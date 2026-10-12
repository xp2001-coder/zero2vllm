import dataclasses
import unittest

from zero2vllm.engine.lifecycle import LifecycleEngine, Status
from zero2vllm.engine.runners import TinyCachedRunner
from zero2vllm.engine.scheduler import BatchScheduler
from zero2vllm.reference.attention import generate


class SchedulerTests(unittest.TestCase):
    def make(self, policy='continuous', capacity=32, slots=2, budget=8, runner=None):
        engine = LifecycleEngine(runner or TinyCachedRunner(), capacity_tokens=capacity)
        self.addCleanup(engine.close)
        return BatchScheduler(engine, policy=policy, max_running=slots, max_tokens_per_round=budget)

    def drain(self, scheduler, limit=100):
        trace = []
        while scheduler.has_pending:
            self.assertLess(len(trace), limit, 'Scheduler made no bounded progress.')
            trace.append(scheduler.tick())
        self.assertEqual(scheduler.engine.used_tokens, 0)
        return trace

    def mixed(self, policy):
        s = self.make(policy)
        s.submit('a', [1, 2, 3], 6)
        s.submit('b', [4, 2], 2)
        trace = [s.tick()]
        s.submit('c', [5, 6], 2)
        trace.append(s.tick())
        s.submit('d', [3, 1, 2], 2)
        trace += self.drain(s)
        return s, trace

    def test_golden_static_cohorts_and_continuous_refill(self):
        expected = {
            'static': [{'a', 'b'}, {'a', 'b'}, {'a'}, {'a'}, {'a'}, {'a'}, {'c', 'd'}, {'c', 'd'}],
            'continuous': [{'a', 'b'}, {'a', 'b'}, {'a', 'c'}, {'a', 'c'}, {'a', 'd'}, {'a', 'd'}],
        }
        for policy in expected:
            s, trace = self.mixed(policy)
            self.assertEqual([set(t.executed) for t in trace], expected[policy])
            for record in s.records():
                ref = generate(s.engine.runner.model, record['prompt_ids'], record['max_new_tokens'], use_cache=False)
                self.assertEqual(list(record['output_ids']), ref.token_ids)
                self.assertEqual(record['cached_positions'], 0)
            self.assertEqual(sum(i.input_tokens for t in trace for i in t.planned), 18)

    def test_boundary_metrics_have_no_off_by_one(self):
        s, _ = self.mixed('continuous')
        c = next(r for r in s.records() if r['request_id'] == 'c')
        self.assertEqual((c['arrival_boundary'], c['admission_boundary'], c['token_boundaries'], c['terminal_boundary']), (1, 2, [3, 4], 4))
        self.assertEqual((c['first_token_delay_rounds'], c['completion_delay_rounds'], c['inter_token_gaps_rounds']), (2, 3, [1]))

    def test_no_refill_inside_execution_loop(self):
        s = self.make(slots=1)
        s.submit('a', [1], 1)
        s.submit('b', [2], 1)
        first = s.tick()
        self.assertEqual(first.executed, ('a',))
        self.assertEqual(s.engine.view('b').status, Status.WAITING)
        self.assertEqual(s.tick().executed, ('b',))

    def test_active_does_not_mean_selected_and_prefill_cost_is_not_one(self):
        s = self.make(budget=3)
        s.submit('a', [1, 2, 3], 3)
        s.submit('b', [4, 2], 3)
        first = s.tick()
        self.assertEqual(first.running, ('a', 'b'))
        self.assertEqual([(i.phase, i.input_tokens) for i in first.planned], [('prefill', 3)])
        self.assertEqual(first.skipped_budget, ('b',))
        self.assertEqual(s.engine.view('b').cached_positions, 0)
        second = s.tick()
        self.assertEqual([(i.request_id, i.phase, i.input_tokens) for i in second.planned], [('b', 'prefill', 2), ('a', 'decode', 1)])
        self.drain(s)

    def test_rotation_gives_long_prefill_a_turn(self):
        s = self.make(slots=3, budget=4)
        s.submit('a', [1], 6)
        s.submit('b', [2], 6)
        s.tick()
        s.submit('long', [3, 4, 5, 6], 2)
        traces = [s.tick() for _ in range(3)]
        self.assertTrue(any('long' in t.executed for t in traces))
        self.drain(s)

    def test_fifo_capacity_blocks_smaller_follower(self):
        s = self.make(capacity=8, slots=3)
        for n, p, count in [('a', [1, 2, 3], 4), ('b', [2, 3, 4], 3), ('c', [1], 1)]:
            s.submit(n, p, count)
        t = s.tick()
        self.assertEqual((t.admitted, t.waiting, t.admission_blocked), (('a',), ('b', 'c'), 'fifo_capacity'))
        self.assertEqual(s.engine.available_tokens, 2)
        self.drain(s)

    def test_boundary_cancel_fills_slot_but_static_waits_for_other_member(self):
        for policy in ('static', 'continuous'):
            s = self.make(policy)
            for n in ('a', 'b', 'c'):
                s.submit(n, [1, 2], 4)
            s.tick()
            self.assertTrue(s.cancel('a'))
            self.assertFalse(s.cancel('a'))
            t = s.tick()
            self.assertEqual('c' in t.admitted, policy == 'continuous')
            self.assertEqual(s.records()[0]['terminal_boundary'], 1)
            self.drain(s)

    def test_cancel_waiting_head_unblocks_fifo(self):
        s = self.make(capacity=8, slots=3)
        s.submit('a', [1, 2, 3], 4)
        s.submit('b', [1, 2, 3], 3)
        s.submit('c', [1], 1)
        s.tick()
        s.cancel('b')
        self.assertEqual(s.tick().admitted, ('c',))
        self.drain(s)

    def test_zero_output_has_no_first_token_or_execution(self):
        s = self.make(budget=1)
        s.submit('zero', [1, 2, 3], 0)
        self.assertFalse(s.has_pending)
        record = s.records()[0]
        self.assertIsNone(record['first_token_delay_rounds'])
        self.assertEqual(record['completion_delay_rounds'], 0)
        self.assertEqual(s.tick().planned, ())

    def test_eos_shortens_cohort_without_changing_other_outputs(self):
        s = self.make('static')
        eos = generate(s.engine.runner.model, [1, 2, 3], 1, use_cache=False).token_ids[0]
        s.submit('eos', [1, 2, 3], 5, eos)
        s.submit('normal', [2], 2)
        s.submit('next', [3], 1)
        trace = self.drain(s)
        self.assertEqual(s.engine.view('eos').reason, 'eos')
        self.assertEqual([t.admitted for t in trace], [('eos', 'normal'), (), ('next',)])

    def test_initialization_failure_does_not_consume_slot_or_capacity(self):
        runner = TinyCachedRunner()
        original = runner.new_context
        count = 0
        def fail_once():
            nonlocal count
            count += 1
            if count == 1:
                raise MemoryError('injected')
            return original()
        runner.new_context = fail_once
        s = self.make(slots=1, runner=runner)
        s.submit('bad', [1], 1)
        s.submit('good', [2], 1)
        t = s.tick()
        self.assertEqual(t.executed, ('good',))
        self.assertEqual(s.engine.view('bad').status, Status.FAILED)
        self.assertEqual(s.records()[0]['terminal_boundary'], 0)
        self.assertEqual(s.engine.used_tokens, 0)

    def test_forward_failure_isolated_and_slot_reused_next_round(self):
        runner = TinyCachedRunner()
        original = runner.advance
        def fail_one(prompt, generated, context):
            token = original(prompt, generated, context)
            if prompt == (1,):
                raise RuntimeError('injected after KV append')
            return token
        runner.advance = fail_one
        s = self.make(runner=runner)
        for n, p in [('bad', [1]), ('good', [2]), ('next', [3])]:
            s.submit(n, p, 2)
        first = s.tick()
        self.assertEqual(first.terminal, ('bad',))
        self.assertEqual(s.tick().admitted, ('next',))
        self.drain(s)
        self.assertEqual(s.engine.view('good').status, Status.FINISHED)

    def test_cancel_planned_peer_skips_its_forward(self):
        s = self.make()
        original = s.engine.runner.advance
        def cancel_peer(*args):
            token = original(*args)
            s.engine.cancel('b')  # Low-level cooperative hook, as in 2.3.
            return token
        s.engine.runner.advance = cancel_peer
        s.submit('a', [1], 1)
        s.submit('b', [2], 1)
        t = s.tick()
        self.assertEqual(len(t.planned), 2)
        self.assertEqual(t.executed, ('a',))
        self.assertEqual(s.engine.view('b').forward_calls, 0)
        self.assertEqual(s.engine.used_tokens, 0)

    def test_oversized_prefill_rejected_before_submission(self):
        s = self.make(budget=2)
        with self.assertRaisesRegex(ValueError, 'Chunking'):
            s.submit('a', [1, 2, 3], 1)
        self.assertEqual(s.records(), [])
        s.submit('a', [1], 1)
        self.drain(s)

    def test_invalid_settings_and_incompatible_runner(self):
        for options in ({'policy': 'random'}, {'max_running': True}, {'max_tokens_per_round': 0}):
            with self.assertRaises(ValueError):
                BatchScheduler(LifecycleEngine(TinyCachedRunner()), **options)
        runner = TinyCachedRunner()
        runner.supports_cached_steps = False
        with self.assertRaisesRegex(ValueError, 'P-then-1'):
            BatchScheduler(LifecycleEngine(runner))

    def test_reentrant_scheduler_operation_rejected(self):
        s = self.make()
        original = s.engine.runner.advance
        def attempt(*args):
            with self.assertRaisesRegex(RuntimeError, 'between rounds'):
                s.tick()
            return original(*args)
        s.engine.runner.advance = attempt
        s.submit('a', [1], 1)
        self.drain(s)

    def test_trace_and_records_cannot_mutate_scheduler(self):
        s = self.make()
        prompt = [1, 2]
        s.submit('a', prompt, 2)
        prompt[0] = 15
        trace = s.tick()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            trace.planned[0].input_tokens = 99
        records = s.records()
        records[0]['token_boundaries'].clear()
        self.assertEqual(s.records()[0]['token_boundaries'], [1])
        self.assertEqual(s.engine.view('a').prompt_ids, (1, 2))
        self.drain(s)

    def test_idle_rounds_and_later_arrival_use_same_clock(self):
        s = self.make()
        for _ in range(3):
            self.assertEqual(s.tick().executed, ())
        s.submit('late', [1], 1)
        s.tick()
        r = s.records()[0]
        self.assertEqual((r['arrival_boundary'], r['token_boundaries'], r['completion_delay_rounds']), (3, [4], 1))

    def test_limits_hold_across_small_grid_of_workloads(self):
        for policy in ('static', 'continuous'):
            for slots in (1, 2, 3):
                for budget in (3, 5):
                    s = self.make(policy, capacity=8, slots=slots, budget=budget)
                    for i in range(6):
                        s.submit(str(i), [i % 15 + 1] * (i % 3 + 1), i % 4 + 1)
                    for t in self.drain(s):
                        self.assertLessEqual(len(t.running), slots)
                        self.assertLessEqual(sum(i.input_tokens for i in t.planned), budget)
                        self.assertTrue(0 <= t.reserved_before_execution <= 8)
                        self.assertEqual(len(t.executed), len(set(t.executed)))
                    for r in s.records():
                        expected = generate(s.engine.runner.model, r['prompt_ids'], r['max_new_tokens'], use_cache=False).token_ids
                        self.assertEqual(list(r['output_ids']), expected)

    def test_replay_exposes_long_prefill_tradeoff(self):
        import json
        from scripts.experiments.lesson0204 import WORKLOADS, run_case
        data = json.loads(WORKLOADS.read_text(encoding='utf-8'))
        case = next(c for c in data['cases'] if c['name'] == 'long_prefill_turn')
        static, continuous = [run_case(case, p, data['model']) for p in ('static', 'continuous')]
        def request(result, name):
            return next(r for r in result['requests'] if r['request_id'] == name)
        self.assertTrue(static['passed'] and continuous['passed'])
        self.assertLess(request(continuous, 'long')['first_token_delay_rounds'], request(static, 'long')['first_token_delay_rounds'])
        self.assertGreater(max(request(continuous, 'a')['inter_token_gaps_rounds']), max(request(static, 'a')['inter_token_gaps_rounds']))

    def test_replay_cancellation_and_no_output_metrics(self):
        import json
        from scripts.experiments.lesson0204 import WORKLOADS, run_case
        data = json.loads(WORKLOADS.read_text(encoding='utf-8'))
        case = next(c for c in data['cases'] if c['name'] == 'stop_and_cancel')
        result = run_case(case, 'continuous', data['model'])
        records = {r['request_id']: r for r in result['requests']}
        self.assertEqual(records['cancel_running']['output_ids'], (14, 1))
        self.assertEqual(records['cancel_waiting']['output_ids'], ())
        self.assertEqual(records['eos']['reason'], 'eos')
        self.assertIsNone(records['zero']['first_token_delay_rounds'])
        self.assertTrue(result['passed'])

    def test_replay_rejects_bad_arrivals_and_impossible_future_work(self):
        from scripts.experiments.lesson0204 import run_case
        case = {'name': 'invalid', 'description': '',
                'limits': {'capacity_tokens': 8, 'max_running': 2, 'max_tokens_per_round': 2},
                'requests': [{'id': 'a', 'arrival': -1, 'prompt': [1], 'max_new_tokens': 1}]}
        with self.assertRaisesRegex(ValueError, 'arrival'):
            run_case(case, 'continuous', {'max_context': 32})
        case['requests'][0].update(arrival=10, prompt=[1, 2, 3])
        with self.assertRaisesRegex(ValueError, 'Chunking'):
            run_case(case, 'continuous', {'max_context': 32})

    def test_replay_round_guard_fails_instead_of_reporting_partial_success(self):
        from scripts.experiments.lesson0204 import run_case
        case = {'name': 'limit', 'description': '',
                'limits': {'capacity_tokens': 8, 'max_running': 1, 'max_tokens_per_round': 2},
                'requests': [{'id': 'a', 'arrival': 0, 'prompt': [1], 'max_new_tokens': 4}]}
        with self.assertRaisesRegex(RuntimeError, 'max_rounds reached'):
            run_case(case, 'continuous', {'max_context': 32}, max_rounds=1)


if __name__ == '__main__':
    unittest.main()
