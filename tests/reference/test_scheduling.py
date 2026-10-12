"""Boundary and conservation checks for the lesson 3.2 counting model."""
import random
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.experiments import lesson0302
from scripts.experiments.lesson0301 import read_manifest
from zero2vllm.reference.scheduling import BlockLedger, CountRequest, CountScheduler, reusable_prefix_tokens


class SchedulingTests(unittest.TestCase):
    def test_mixed_rounds_have_partial_prefill_and_one_token_decode(self):
        rows = lesson0302.mixed_rounds()
        self.assertEqual([r['synthetic_output_counts'] for r in rows], [{'A': 1}, {'A': 1}, {'A': 1, 'B': 1}])
        self.assertEqual([r['after_feedback']['requests']['B']['computed'] for r in rows], [3, 6, 7])

    def test_threshold_does_not_mean_total_batch_budget(self):
        row = lesson0302.mixed_rounds(token_budget=8, threshold=2, rounds=1)[0]
        self.assertEqual([r['scheduled'] for r in row['scheduled']], [1, 2])
        self.assertEqual(row['unused_token_budget'], 5)

    def test_one_token_budget_delays_waiting_request(self):
        rows = lesson0302.mixed_rounds(token_budget=1)
        self.assertTrue(all([r['request'] for r in row['scheduled']] == ['A'] for row in rows))
        self.assertEqual(rows[-1]['after_feedback']['requests']['B']['computed'], 0)

    def test_chunk_boundaries_and_final_output(self):
        rows = lesson0302.run_case('chunked')['rounds']
        self.assertEqual([r['scheduled'][0]['scheduled'] for r in rows], [4, 4, 1])
        self.assertEqual(rows[-1]['after_feedback']['requests']['P']['known'], 10)

    def test_sequence_limit_leaves_token_budget_unused(self):
        row = lesson0302.run_case('sequence_limit')['rounds'][0]
        self.assertEqual(row['stop_reason'], 'max_seqs')
        self.assertEqual(row['after_schedule']['waiting'], ['B'])

    def test_waiting_head_is_not_skipped_for_smaller_request(self):
        row = lesson0302.run_case('head_blocking')['rounds'][0]
        self.assertEqual(row['scheduled'], [])
        self.assertEqual(row['after_schedule']['blocks']['free_ids'], [1])

    def test_preempted_request_keeps_generated_history_and_resumes(self):
        rows = lesson0302.run_case('preemption')['rounds']
        self.assertEqual(rows[0]['after_schedule']['requests']['B'], {'known': 5, 'computed': 0, 'status': 'PREEMPTED'})
        self.assertEqual(rows[1]['scheduled'][0]['kind'], 'resumed')
        self.assertEqual(rows[1]['scheduled'][0]['computed_before'], 0)

    def test_self_preemption_returns_without_infinite_retry(self):
        row = lesson0302.run_case('self_preemption')['rounds'][0]
        self.assertEqual(row['preempted'], ['A'])
        self.assertEqual(row['after_schedule']['blocks']['free_ids'], [1])

    def test_completion_required_between_rounds(self):
        s = CountScheduler()
        s.add(CountRequest('P', 8))
        s.step()
        with self.assertRaises(ValueError): s.step()
        with self.assertRaises(ValueError): s.add(CountRequest('Q', 2))
        s.complete()
        self.assertEqual(s.step()['scheduled'][0]['computed_before'], 4)

    def test_invalid_partial_output_is_atomic(self):
        s = CountScheduler()
        s.add(CountRequest('P', 8))
        s.step()
        before = s.snapshot()
        with self.assertRaises(ValueError): s.complete({'P': 1})
        self.assertEqual(s.snapshot(), before)
        s.complete()

    def test_feedback_id_and_count_are_checked(self):
        s = CountScheduler()
        s.add(CountRequest('P', 1))
        s.step()
        for feedback in ({'unknown': 1}, {'P': 2}, {'P': -1}, {'P': True}, {}):
            with self.assertRaises(ValueError): s.complete(feedback)
        s.complete({'P': 1}, finished=['P'])
        self.assertEqual(len(s.pool.free), 11)

    def test_limit_can_chunk_but_must_finish_when_caught_up(self):
        s = CountScheduler(token_budget=2, max_model_len=4)
        s.add(CountRequest('P', 4))
        s.step()
        s.complete()
        s.step()
        with self.assertRaises(ValueError): s.complete({'P': 1})
        s.complete(finished=['P'])
        self.assertEqual(s.requests['P'].status, 'FINISHED')

    def test_invalid_prepared_state_does_not_allocate(self):
        s = CountScheduler(num_blocks=2)
        before = s.snapshot()
        for r in (CountRequest('A', 4, 4), CountRequest('', 1), CountRequest('A', 8, 5)):
            with self.assertRaises(ValueError): s.add(r, running=True)
        self.assertEqual(s.snapshot(), before)

    def test_bad_configuration_rejected(self):
        for kwargs in ({'token_budget': 0}, {'max_seqs': True}, {'threshold': -1}, {'block_size': 0}, {'num_blocks': 1}):
            with self.assertRaises(ValueError): CountScheduler(**kwargs)


class PrefixLedgerTests(unittest.TestCase):
    def test_full_block_cap_keeps_one_input_for_logits(self):
        self.assertEqual([reusable_prefix_tokens(n, 100, 4) for n in range(1, 14)],
                         [0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12])
        self.assertEqual(reusable_prefix_tokens(13, 1, 4), 4)

    def test_free_prefix_hit_cannot_also_be_new_storage(self):
        result = lesson0302.run_case('prefix_capacity')
        self.assertEqual(result['failed_before'], result['failed_after'])
        self.assertEqual(result['success']['tables']['P'], [1, 2, 3])

    def test_active_prefix_hit_does_not_consume_free_queue(self):
        p = BlockLedger(4)
        p.seed_cached([1])
        p.reserve('A', 4, hits=[1])
        self.assertEqual(p.reserve('B', 5, hits=[1]), [2])
        self.assertEqual(list(p.free), [3])
        self.assertEqual(p.refs[1], 2)

    def test_release_keeps_shared_block_until_last_owner(self):
        r = lesson0302.run_case('shared_release')
        self.assertNotIn(1, r['after_A']['free_ids'])
        self.assertIn(1, r['after_B']['free_ids'])
        self.assertIn(1, r['after_B']['cached_ids'])

    def test_eviction_removes_old_identity(self):
        self.assertEqual(lesson0302.run_case('eviction')['after']['cached_ids'], [])

    def test_duplicate_or_unverified_hits_are_rejected_atomically(self):
        p = BlockLedger(4)
        p.seed_cached([1])
        before = p.snapshot()
        for hits in ([1, 1], [0], [2], [9]):
            with self.assertRaises(ValueError): p.reserve('A', 8, hits=hits)
        self.assertEqual(before, p.snapshot())

    def test_existing_tail_slot_needs_no_new_block(self):
        p = BlockLedger(2)
        self.assertEqual(p.reserve('A', 3), [1])
        self.assertEqual(p.reserve('A', 4), [])
        before = p.snapshot()
        self.assertIsNone(p.reserve('A', 5))
        self.assertEqual(before, p.snapshot())

    def test_random_allocations_preserve_ownership_and_capacity(self):
        rng, p = random.Random(302), BlockLedger(8)
        lengths = {}
        for _ in range(300):
            rid = str(rng.randrange(6))
            if rid in lengths and rng.random() < .4:
                p.release(rid)
                del lengths[rid]
            else:
                length = lengths.get(rid, 0) + rng.randrange(1, 5)
                before = p.snapshot()
                result = p.reserve(rid, length)
                if result is None: self.assertEqual(before, p.snapshot())
                else: lengths[rid] = length
            p.check()
            self.assertEqual(len(p.free) + sum(len(t) for t in p.tables.values()), 7)
        for rid in list(lengths): p.release(rid)
        self.assertEqual(set(p.free), set(range(1, 8)))


class LessonRunnerTests(unittest.TestCase):
    def test_all_cases_are_offline_without_upstream_source(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(StringIO()), patch.object(lesson0302, 'audit', side_effect=AssertionError('must stay offline')), patch.object(lesson0302, 'fetch_sources', side_effect=AssertionError('must not fetch')):
            self.assertEqual(lesson0302.main(['--output-dir', temp]), 0)
            self.assertTrue((Path(temp)/'summary.json').is_file())

    def test_missing_requested_source_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(StringIO()), redirect_stderr(StringIO()) as errors:
            self.assertEqual(lesson0302.main(['--case', 'prefix_boundary', '--inspect-source', '--source-dir', temp, '--output-dir', temp]), 1)
            self.assertIn('Missing source file', errors.getvalue())

    def test_manifest_is_pinned_and_includes_core_ownership_functions(self):
        m = read_manifest(lesson0302.MANIFEST)
        self.assertEqual(m['commit'], 'ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d')
        self.assertTrue({'Scheduler.schedule', 'KVCacheManager.allocate_slots', 'BlockPool.touch', 'Scheduler.update_from_output'} <= {c['symbol'] for c in m['checkpoints']})


if __name__ == '__main__':
    unittest.main()
