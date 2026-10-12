import dataclasses
import unittest

from zero2vllm.engine.lifecycle import LifecycleEngine, Status
from zero2vllm.engine.runners import TinyCachedRunner
from zero2vllm.reference.attention import TinyAttentionLM, generate


class ObservedRunner(TinyCachedRunner):
    def __init__(self):
        super().__init__(TinyAttentionLM(max_context=16))
        self.contexts = []
        self.after_forward = None

    def new_context(self):
        context = super().new_context()
        self.contexts.append(context)
        return context

    def advance(self, *args):
        token = super().advance(*args)
        if self.after_forward:
            self.after_forward()
        return token


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.runner = ObservedRunner()
        self.engine = LifecycleEngine(self.runner, capacity_tokens=16)
        self.addCleanup(self.engine.close)

    def start(self, name='a', prompt=(1, 2, 3), count=4, eos=None):
        self.engine.submit(name, prompt, count, eos)
        self.assertTrue(self.engine.admit(name))

    def drain(self, name):
        while self.engine.view(name).status == Status.RUNNING:
            self.engine.step(name)
        return self.engine.view(name)

    def test_normal_generation_matches_independent_full_recompute(self):
        self.start()
        view = self.drain('a')
        expected = generate(self.runner.model, [1, 2, 3], 4, use_cache=False).token_ids
        self.assertEqual(list(view.output_ids), expected)
        self.assertEqual((view.status, view.reason, view.peak_cached_positions), (Status.FINISHED, 'length', 6))
        self.assertEqual((view.cached_positions, view.reserved_tokens, self.engine.used_tokens), (0, 0, 0))

    def test_waiting_has_no_context_and_admission_allocates_empty_cache(self):
        view = self.engine.submit('a', [1, 2, 3], 4)
        self.assertEqual((view.status, self.engine.used_tokens, len(self.runner.contexts)), (Status.WAITING, 0, 0))
        self.engine.admit('a')
        self.assertEqual((self.engine.used_tokens, self.engine.view('a').cached_positions), (6, 0))

    def test_generated_count_is_one_ahead_of_cache_after_a_step(self):
        self.start()
        for count in (1, 2, 3):
            view = self.engine.step('a')
            self.assertEqual(len(view.output_ids), count)
            self.assertEqual(view.cached_positions, 3 + count - 1)

    def test_eos_takes_precedence_over_length_and_stays_in_ids(self):
        eos = generate(self.runner.model, [1], 1, use_cache=False).token_ids[0]
        self.start(prompt=(1,), count=1, eos=eos)
        view = self.engine.step('a')
        self.assertEqual((view.reason, view.output_ids), ('eos', (eos,)))

    def test_zero_generation_never_allocates_or_calls_model(self):
        view = self.engine.submit('a', [1], 0)
        self.assertEqual((view.status, view.forward_calls, view.output_ids), (Status.FINISHED, 0, ()))
        self.assertEqual(self.runner.contexts, [])
        self.assertFalse(self.engine.admit('a'))

    def test_waiting_cancel_never_allocates(self):
        self.engine.submit('a', [1], 3)
        self.assertTrue(self.engine.cancel('a'))
        self.assertEqual(self.engine.view('a').status, Status.CANCELLED)
        self.assertEqual(self.runner.contexts, [])
        self.assertFalse(self.engine.admit('a'))

    def test_running_cancel_clears_actual_lists_and_is_idempotent(self):
        self.start()
        first = self.engine.step('a')
        context = self.runner.contexts[0]
        cache = context.kv
        self.assertTrue(cache.keys)
        self.assertTrue(self.engine.cancel('a'))
        events = self.engine.events('a')
        self.assertFalse(self.engine.cancel('a'))
        self.assertEqual(self.engine.step('a').output_ids, first.output_ids)
        self.assertEqual(self.engine.events('a'), events)
        self.assertEqual((cache.keys, cache.values, context.kv), ([], [], None))
        self.assertTrue(context.closed)
        self.assertEqual(sum(e['kind'] == 'finalized' for e in events), 1)

    def test_cancel_during_forward_defers_release_and_discards_token(self):
        self.start()
        observed = []
        def cancel_in_forward():
            self.engine.cancel('a')
            view = self.engine.view('a')
            observed.append((view.status, view.reserved_tokens, view.cached_positions))
        self.runner.after_forward = cancel_in_forward
        view = self.engine.step('a')
        self.assertEqual(observed, [(Status.RUNNING, 6, 3)])
        self.assertEqual((view.status, view.output_ids, self.engine.used_tokens), (Status.CANCELLED, (), 0))
        kinds = [e['kind'] for e in self.engine.events('a')]
        self.assertLess(kinds.index('cancel_requested'), kinds.index('forward_returned'))
        self.assertNotIn('token_committed', kinds)

    def test_forward_error_after_cache_mutation_releases_resources(self):
        self.start()
        def fail():
            raise RuntimeError('injected after KV append')
        self.runner.after_forward = fail
        view = self.engine.step('a')
        self.assertEqual((view.status, view.reason, view.peak_cached_positions), (Status.FAILED, 'model_error', 3))
        self.assertIn('injected', view.error)
        self.assertEqual((view.output_ids, self.engine.used_tokens), ((), 0))
        self.assertTrue(self.runner.contexts[0].closed)

    def test_forward_error_wins_over_pending_cancel(self):
        self.start()
        def cancel_then_fail():
            self.engine.cancel('a')
            raise RuntimeError('forward failed')
        self.runner.after_forward = cancel_then_fail
        view = self.engine.step('a')
        self.assertTrue(view.cancel_requested)
        self.assertEqual(view.status, Status.FAILED)
        self.assertEqual(self.engine.used_tokens, 0)

    def test_initialization_error_rolls_back_reservation(self):
        def fail():
            raise MemoryError('injected allocation failure')
        self.runner.new_context = fail
        self.engine.submit('a', [1], 2)
        self.assertFalse(self.engine.admit('a'))
        view = self.engine.view('a')
        self.assertEqual((view.status, view.reason, self.engine.used_tokens), (Status.FAILED, 'initialization_error', 0))

    def test_capacity_deferral_is_atomic_and_cancel_allows_retry(self):
        engine = LifecycleEngine(self.runner, capacity_tokens=6)
        self.addCleanup(engine.close)
        for name in ('a', 'b'):
            engine.submit(name, [1, 2, 3], 4)
        self.assertTrue(engine.admit('a'))
        self.assertFalse(engine.admit('b'))
        self.assertEqual((engine.used_tokens, engine.view('b').reserved_tokens, len(self.runner.contexts)), (6, 0, 1))
        engine.cancel('a')
        self.assertTrue(engine.admit('b'))
        engine.close()
        self.assertEqual(engine.used_tokens, 0)

    def test_interleaved_requests_have_distinct_cache_and_reference_outputs(self):
        self.start('a', (1, 2, 3), 4)
        self.start('b', (4, 2), 3)
        self.assertIsNot(self.runner.contexts[0].kv, self.runner.contexts[1].kv)
        for _ in range(4):
            self.engine.step('a')
            self.engine.step('b')
        for name, prompt, count in [('a', [1, 2, 3], 4), ('b', [4, 2], 3)]:
            expected = generate(self.runner.model, prompt, count, use_cache=False).token_ids
            self.assertEqual(list(self.engine.view(name).output_ids), expected)
        self.assertEqual(self.engine.used_tokens, 0)

    def test_cancel_one_request_does_not_change_another_cache_or_weights(self):
        self.start('a')
        self.start('b', (4, 5), 3)
        self.engine.step('a')
        self.engine.step('b')
        other = self.runner.contexts[1].kv
        before = [row[:] for row in other.keys]
        weights = [row[:] for row in self.runner.model.wq]
        self.engine.cancel('a')
        self.assertEqual(other.keys, before)
        self.assertEqual(self.runner.model.wq, weights)
        expected = generate(self.runner.model, [4, 5], 3, use_cache=False).token_ids
        self.assertEqual(list(self.drain('b').output_ids), expected)

    def test_invalid_submission_has_no_state_or_budget_side_effect(self):
        bad = [([], 1, None), ([True], 1, None), ([16], 1, None), ([1], -1, None),
               ([1], True, None), ([1], 1, 16), ([1]*17, 0, None), ([1], 17, None)]
        for prompt, count, eos in bad:
            with self.subTest(prompt=prompt, count=count, eos=eos), self.assertRaises(ValueError):
                self.engine.submit('a', prompt, count, eos)
            self.assertEqual(self.engine.used_tokens, 0)
            with self.assertRaises(KeyError):
                self.engine.view('a')
        with self.assertRaises(ValueError):
            LifecycleEngine(self.runner, 0)

    def test_request_that_cannot_fit_total_budget_is_rejected(self):
        engine = LifecycleEngine(self.runner, 2)
        with self.assertRaisesRegex(ValueError, 'never fit'):
            engine.submit('a', [1, 2], 2)

    def test_duplicate_and_unadmitted_step_do_not_overwrite_request(self):
        self.engine.submit('a', [1], 2)
        with self.assertRaises(ValueError):
            self.engine.submit('a', [2], 1)
        with self.assertRaises(ValueError):
            self.engine.step('a')
        self.assertEqual(self.engine.view('a').status, Status.WAITING)

    def test_views_events_and_prompt_do_not_expose_mutable_request_state(self):
        prompt = [1, 2]
        view = self.engine.submit('a', prompt, 2)
        prompt[0] = 15
        with self.assertRaises(dataclasses.FrozenInstanceError):
            view.status = Status.FAILED
        events = self.engine.events('a')
        events[0]['kind'] = 'corrupted'
        self.assertEqual(self.engine.events('a')[0]['kind'], 'submitted')
        self.engine.admit('a')
        expected = generate(self.runner.model, [1, 2], 2, use_cache=False).token_ids
        self.assertEqual(list(self.drain('a').output_ids), expected)

    def test_forget_requires_terminal_and_removes_result(self):
        self.start()
        with self.assertRaises(ValueError):
            self.engine.forget('a')
        self.engine.cancel('a')
        self.engine.forget('a')
        with self.assertRaises(KeyError):
            self.engine.view('a')

    def test_context_manager_closes_on_exception_and_close_is_idempotent(self):
        engine = LifecycleEngine(self.runner)
        with self.assertRaisesRegex(RuntimeError, 'caller failed'):
            with engine:
                engine.submit('a', [1], 3)
                engine.submit('b', [2], 3)
                engine.admit('a')
                engine.step('a')
                raise RuntimeError('caller failed')
        engine.close()
        self.assertEqual(engine.used_tokens, 0)
        self.assertTrue(all(engine.view(n).status == Status.CANCELLED for n in ('a', 'b')))
        with self.assertRaises(RuntimeError):
            engine.submit('c', [1], 1)

    def test_nested_step_is_rejected_without_double_commit(self):
        self.start()
        self.runner.after_forward = lambda: self.engine.step('a')
        view = self.engine.step('a')
        self.assertEqual((view.status, view.forward_calls, view.output_ids), (Status.FAILED, 1, ()))

    def test_keyboard_interrupt_propagates_and_context_manager_cleans_up(self):
        engine = LifecycleEngine(self.runner)
        def interrupt():
            raise KeyboardInterrupt()
        self.runner.after_forward = interrupt
        with self.assertRaises(KeyboardInterrupt):
            with engine:
                engine.submit('a', [1, 2, 3], 4)
                engine.admit('a')
                engine.step('a')
        self.assertEqual((engine.view('a').status, engine.used_tokens), (Status.CANCELLED, 0))
        self.assertTrue(self.runner.contexts[0].closed)

    def test_invalid_runner_token_fails_and_releases(self):
        self.start()
        self.runner.advance = lambda *args: self.runner.vocab_size
        self.assertEqual(self.engine.step('a').status, Status.FAILED)
        self.assertEqual(self.engine.used_tokens, 0)

    def test_context_limit_allows_last_predicted_token_not_being_fed_back(self):
        self.start(prompt=tuple(range(16)), count=1)
        self.assertEqual(self.engine.step('a').status, Status.FINISHED)
