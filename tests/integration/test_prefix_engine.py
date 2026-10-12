import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from zero2vllm.engine.lifecycle import LifecycleEngine, Status
    from zero2vllm.engine.prefix_runner import PrefixTinyRunner
    from zero2vllm.engine.scheduler import BatchScheduler
    from zero2vllm.reference.attention import TinyAttentionLM, generate


@unittest.skipUnless(HAS_TORCH, 'optional PyTorch dependency is not installed')
class PrefixEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def run_request(self, runner, prompt, count=3):
        context = runner.new_context()
        context.reserve(len(prompt)+count-1)
        ids, logits = [], []
        try:
            for _ in range(count):
                ids.append(runner.advance(prompt, ids, context))
                logits.append(list(context.last_logits))
        finally:
            context.close()
        return ids, logits, context

    def test_cold_hot_logits_match_full_recompute_across_sizes_and_dtypes(self):
        prompt = [1, 2, 3, 4, 5]
        reference = generate(TinyAttentionLM(max_context=32), prompt, 3, use_cache=False)
        for dtype in (torch.float64, torch.float32):
            for size in (1, 2, 3, 4):
                runner = PrefixTinyRunner(block_size=size, dtype=dtype)
                for run in range(2):
                    ids, logits, ctx = self.run_request(runner, prompt)
                    self.assertEqual(ids, reference.token_ids)
                    error = max(abs(a-b) for row, ref in zip(logits, reference.logits) for a, b in zip(row, ref))
                    self.assertLess(error, 1e-12 if dtype == torch.float64 else 1e-6)
                    hits = ((len(prompt)-1)//size)*size if run else 0
                    self.assertEqual(ctx.hit_positions, hits)
                    self.assertEqual(ctx.projected_positions, 7-hits)
                    self.assertEqual(runner.store.pool_snapshot()['active_block_references'], 0)

    def test_same_prompt_whole_block_still_computes_final_logits(self):
        runner = PrefixTinyRunner(block_size=2)
        self.run_request(runner, [1, 2, 3, 4], 1)
        _, _, ctx = self.run_request(runner, [1, 2, 3, 4], 1)
        self.assertEqual((ctx.hit_positions, ctx.projected_positions), (2, 2))

    def test_branching_live_contexts_share_prefix_and_preserve_outputs(self):
        runner = PrefixTinyRunner()
        a, b = runner.new_context(), runner.new_context()
        for ctx in (a, b):
            ctx.reserve(7)
            self.addCleanup(ctx.close)
        pa, pb = [1, 2, 3, 4, 5], [1, 2, 3, 4, 6]
        ia = [runner.advance(pa, [], a)]
        ib = [runner.advance(pb, [], b)]
        self.assertEqual(b.hit_positions, 4)
        self.assertEqual(runner.store.snapshot(a.handle)['block_refs'][:2], [2, 2])
        a.close()
        ib.append(runner.advance(pb, ib, b))
        self.assertEqual(ib, generate(TinyAttentionLM(max_context=32), pb, 2, use_cache=False).token_ids)
        self.assertEqual(len(ia), 1)

    def test_lifecycle_and_scheduler_cleanup_with_conservative_budget(self):
        runner = PrefixTinyRunner(num_blocks=8)
        contexts = []
        create = runner.new_context
        def capture():
            ctx = create()
            contexts.append(ctx)
            return ctx
        runner.new_context = capture
        with LifecycleEngine(runner) as engine:
            scheduler = BatchScheduler(engine, max_running=2, max_tokens_per_round=10)
            scheduler.submit('a', [1, 2, 3, 4, 5], 3)
            scheduler.submit('b', [1, 2, 3, 4, 6], 3)
            while scheduler.has_pending:
                scheduler.tick()
                self.assertLessEqual(runner.store.pool_snapshot()['active_unique_blocks']*2, engine.used_tokens)
            self.assertEqual(engine.used_tokens, 0)
            self.assertTrue(all(c.closed for c in contexts))
            self.assertEqual([c.hit_positions for c in contexts], [0, 4])
            self.assertGreater(runner.store.pool_snapshot()['idle_cached_blocks'], 0)

    def test_cancelled_request_drops_refs_without_destroying_peer_prefix(self):
        runner = PrefixTinyRunner(num_blocks=8)
        with LifecycleEngine(runner) as engine:
            for name, prompt in [('a', [1, 2, 3]), ('b', [1, 2, 4])]:
                engine.submit(name, prompt, 3)
                engine.admit(name)
                engine.step(name)
            engine.cancel('a')
            while engine.view('b').status == Status.RUNNING:
                engine.step('b')
            self.assertEqual(list(engine.view('b').output_ids), generate(TinyAttentionLM(max_context=32), [1, 2, 4], 3, use_cache=False).token_ids)
            self.assertEqual(runner.store.pool_snapshot()['active_block_references'], 0)

    def test_exception_after_publishing_reclaims_live_references(self):
        runner = PrefixTinyRunner()
        advance = runner.advance
        def fail(prompt, generated, context):
            advance(prompt, generated, context)
            raise RuntimeError('injected after complete KV writes')
        runner.advance = fail
        with LifecycleEngine(runner) as engine:
            engine.submit('a', [1, 2, 3], 3)
            engine.admit('a')
            engine.step('a')
            self.assertEqual(engine.view('a').status, Status.FAILED)
            self.assertEqual(engine.used_tokens, 0)
        self.assertEqual(runner.store.pool_snapshot()['active_block_references'], 0)
        runner.advance = advance
        _, _, ctx = self.run_request(runner, [1, 2, 4])
        self.assertEqual(ctx.hit_positions, 2)

    def test_zero_output_never_binds_cache_and_eos_returns_refs(self):
        runner = PrefixTinyRunner()
        first = generate(TinyAttentionLM(max_context=32), [1, 2, 3], 1, use_cache=False).token_ids[0]
        with LifecycleEngine(runner) as engine:
            engine.submit('zero', [1, 2, 3], 0)
            self.assertEqual(runner.store.events, [])
            engine.submit('eos', [1, 2, 3], 3, first)
            engine.admit('eos')
            engine.step('eos')
            self.assertEqual(engine.view('eos').status, Status.FINISHED)
        self.assertEqual(runner.store.pool_snapshot()['active_block_references'], 0)

    def test_history_mismatch_does_not_mutate_cache(self):
        runner = PrefixTinyRunner()
        ctx = runner.new_context()
        ctx.reserve(6)
        self.addCleanup(ctx.close)
        token = runner.advance([1, 2, 3], [], ctx)
        before = runner.store.snapshot(ctx.handle)
        with self.assertRaises(ValueError):
            runner.advance([1, 2, 4], [token], ctx)
        self.assertEqual(runner.store.snapshot(ctx.handle), before)

    def test_runner_owns_fixed_weight_snapshot(self):
        model = TinyAttentionLM(max_context=32)
        runner = PrefixTinyRunner(model=model)
        first, _, _ = self.run_request(runner, [1, 2, 3])
        model.wv[0][0] += 100
        second, _, _ = self.run_request(runner, [1, 2, 3])
        self.assertEqual(first, second)
        self.assertNotEqual(runner.model_fingerprint, PrefixTinyRunner(model=model).model_fingerprint)

    def test_namespace_changes_for_weights_dtype_or_salt(self):
        base = PrefixTinyRunner()
        self.assertEqual(base.namespace, PrefixTinyRunner().namespace)
        for changed in (PrefixTinyRunner(dtype=torch.float32), PrefixTinyRunner(cache_salt='team-b'),
                        PrefixTinyRunner(model=TinyAttentionLM(max_context=32, seed=9))):
            self.assertNotEqual(base.namespace, changed.namespace)

    def test_context_close_is_idempotent_and_foreign_context_is_rejected(self):
        runner, other = PrefixTinyRunner(), PrefixTinyRunner()
        ctx = other.new_context()
        ctx.reserve(3)
        with self.assertRaises(ValueError):
            runner.advance([1], [], ctx)
        ctx.close()
        ctx.close()
        with self.assertRaises(ValueError):
            other.advance([1], [], ctx)
