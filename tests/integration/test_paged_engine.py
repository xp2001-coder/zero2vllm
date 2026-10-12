import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from zero2vllm.engine.lifecycle import LifecycleEngine, Status
    from zero2vllm.engine.paged_runner import PagedTinyRunner
    from zero2vllm.engine.scheduler import BatchScheduler
    from zero2vllm.reference.attention import WorkCounter, generate


@unittest.skipUnless(HAS_TORCH, 'optional PyTorch dependency is not installed')
class PagedEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def make(self, blocks=4, size=2):
        runner = PagedTinyRunner(num_blocks=blocks, block_size=size)
        engine = LifecycleEngine(runner)
        self.addCleanup(engine.close)
        return runner, engine

    def drain(self, scheduler):
        traces = []
        while scheduler.has_pending:
            self.assertLess(len(traces), 100)
            traces.append(scheduler.tick())
        self.assertEqual(scheduler.engine.used_tokens, 0)
        store = scheduler.engine.runner.store
        self.assertEqual(store.free_blocks, store.num_blocks)
        return traces

    def test_all_logits_match_dense_across_block_and_chunk_boundaries(self):
        for dtype, tolerance in ((torch.float64, 1e-12), (torch.float32, 1e-6)):
            for size in (1, 2, 4):
                runner = PagedTinyRunner(num_blocks=12, block_size=size, dtype=dtype)
                for chunks in ([5], [2, 1, 2], [1]*5):
                    context = runner.new_context()
                    try:
                        context.reserve(5)
                        tokens, outputs, offset = [1, 2, 3, 4, 5], [], 0
                        for count in chunks:
                            outputs += runner.forward_tokens(tokens[offset:offset+count], context)
                            offset += count
                        expected = runner.model.forward_full(tokens, WorkCounter())
                        torch.testing.assert_close(torch.tensor(outputs, dtype=torch.float64), torch.tensor(expected, dtype=torch.float64), atol=tolerance, rtol=tolerance)
                    finally:
                        context.close()

    def test_rounded_admission_prevents_physical_overcommit(self):
        runner, engine = self.make(3, 4)
        scheduler = BatchScheduler(engine)
        for name in ('a', 'b'):
            scheduler.submit(name, [1, 2, 3], 3)  # Logical 5, physical 8 each.
        first = scheduler.tick()
        self.assertEqual((first.admitted, first.waiting, first.admission_blocked), (('a',), ('b',), 'fifo_capacity'))
        self.assertEqual((engine.used_tokens, runner.store.free_blocks), (8, 1))
        self.assertEqual(engine.reservation_for('b'), 8)
        trace = [first] + self.drain(scheduler)
        self.assertEqual(len(trace), 6)

    def test_fragmented_blocks_work_under_continuous_scheduling(self):
        runner, engine = self.make()
        scheduler = BatchScheduler(engine, max_running=3)
        observed = []
        original = runner.advance
        def record(prompt, generated, context):
            token = original(prompt, generated, context)
            observed.append((prompt, runner.store.table(context.handle)))
            return token
        runner.advance = record
        for name, prompt, count in [('a', [1, 2], 1), ('b', [3], 2), ('c', [4, 5], 1)]:
            scheduler.submit(name, prompt, count)
        scheduler.tick()
        scheduler.submit('d', [5, 6, 7], 2)
        self.drain(scheduler)
        self.assertIn(((5, 6, 7), (0, 2)), observed)
        for r in scheduler.records():
            self.assertEqual(list(r['output_ids']), generate(runner.model, r['prompt_ids'], r['max_new_tokens'], use_cache=False).token_ids)

    def test_cancel_returns_blocks_for_waiting_request(self):
        runner, engine = self.make(2, 2)
        s = BatchScheduler(engine)
        s.submit('a', [1, 2], 3)
        s.submit('b', [3, 4], 3)
        s.tick()
        s.cancel('a')
        self.assertEqual(runner.store.free_blocks, 2)
        self.assertEqual(s.tick().admitted, ('b',))
        self.drain(s)

    def test_failure_after_append_releases_pages_and_preserves_peer(self):
        runner, engine = self.make(4, 2)
        original = runner.advance
        def fail_one(prompt, generated, context):
            token = original(prompt, generated, context)
            if prompt == (7,):
                raise RuntimeError('injected after paged write')
            return token
        runner.advance = fail_one
        s = BatchScheduler(engine)
        s.submit('bad', [7], 2)
        s.submit('good', [1, 2], 3)
        self.drain(s)
        self.assertEqual(engine.view('bad').status, Status.FAILED)
        self.assertEqual(list(engine.view('good').output_ids), generate(runner.model, [1, 2], 3, use_cache=False).token_ids)

    def test_reservation_failure_closes_context_and_rolls_back_ledger(self):
        runner, engine = self.make()
        context = runner.new_context()
        original = context.reserve
        def fail_after_reserve(count):
            original(count)
            raise MemoryError('injected after page allocation')
        context.reserve = fail_after_reserve
        runner.new_context = lambda: context
        engine.submit('bad', [1, 2], 3)
        self.assertFalse(engine.admit('bad'))
        self.assertTrue(context.closed)
        self.assertEqual((runner.store.free_blocks, engine.used_tokens), (4, 0))

    def test_eos_and_zero_output_do_not_leak_pages(self):
        runner, engine = self.make()
        eos = generate(runner.model, [1, 2, 3], 1, use_cache=False).token_ids[0]
        s = BatchScheduler(engine)
        s.submit('eos', [1, 2, 3], 4, eos)
        s.submit('zero', [1], 0)
        self.drain(s)
        self.assertEqual(engine.view('eos').reason, 'eos')
        self.assertEqual(engine.view('zero').forward_calls, 0)

    def test_engine_capacity_must_match_dedicated_pool(self):
        runner = PagedTinyRunner(num_blocks=2, block_size=4)
        with self.assertRaisesRegex(ValueError, 'dedicated paged pool'):
            LifecycleEngine(runner, capacity_tokens=9)
        self.assertEqual(LifecycleEngine(runner).capacity_tokens, 8)

    def test_invalid_chunk_and_cross_runner_context_do_not_mutate(self):
        runner, _ = self.make()
        context = runner.new_context()
        context.reserve(3)
        try:
            for tokens in ([1, 16], [1, 2, 3, 4]):
                with self.assertRaises(ValueError):
                    runner.forward_tokens(tokens, context)
                self.assertEqual(context.cached_positions, 0)
            foreign = PagedTinyRunner()
            with self.assertRaises(ValueError):
                foreign.forward_tokens([1], context)
        finally:
            context.close()

    def test_cancel_during_forward_keeps_pages_until_return(self):
        runner, engine = self.make(2, 2)
        original = runner.advance
        observations = []
        def cancel_inside(*args):
            token = original(*args)
            engine.cancel('a')
            observations.append((engine.used_tokens, runner.store.free_blocks))
            return token
        runner.advance = cancel_inside
        engine.submit('a', [1, 2], 3)
        engine.admit('a')
        engine.step('a')
        self.assertEqual(observations, [(4, 0)])
        self.assertEqual(engine.view('a').output_ids, ())
        self.assertEqual((engine.used_tokens, runner.store.free_blocks), (0, 2))

    def test_closed_context_cannot_touch_reused_blocks(self):
        runner, engine = self.make(1, 2)
        context = runner.new_context()
        context.reserve(2)
        runner.forward_tokens([1], context)
        context.close()
        fresh = runner.new_context()
        fresh.reserve(2)
        try:
            with self.assertRaises(ValueError):
                runner.forward_tokens([2], context)
            self.assertEqual(fresh.cached_positions, 0)
        finally:
            fresh.close()


if __name__ == '__main__':
    unittest.main()
