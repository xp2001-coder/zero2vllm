from contextlib import redirect_stdout, redirect_stderr
from dataclasses import replace
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.experiments import lesson0303
from zero2vllm.reference.execution import RequestSlice, pack_inputs, route_candidates


class InputPackingTests(unittest.TestCase):
    def setUp(self):
        self.rows = lesson0303.requests()

    def test_exact_request_positions_and_noncontiguous_slots(self):
        b = pack_inputs(self.rows, ['B', 'A'])
        self.assertEqual(b.input_ids, (23, 12, 13, 14))
        self.assertEqual(b.positions, (3, 2, 3, 4))
        self.assertEqual(b.slot_mapping, (23, 34, 35, 12))

    def test_query_length_is_not_total_key_length(self):
        b = pack_inputs(self.rows, ['B', 'A'])
        self.assertEqual(b.query_start_loc, (0, 1, 4))
        self.assertEqual(b.seq_lens, (4, 5))
        self.assertEqual(b.logits_indices, (0, 3))

    def test_reordering_updates_every_array_and_result_ownership(self):
        b = pack_inputs(self.rows, ['A', 'B'])
        self.assertEqual(b.slot_mapping, (34, 35, 12, 23))
        self.assertEqual(b.logits_indices, (2, 3))
        self.assertEqual(b.req_id_to_index, {'A': 0, 'B': 1})
        self.assertEqual(route_candidates(b, [17, 9]), {'A': [], 'B': [9]})

    def test_zero_or_excess_scheduled_tokens_rejected(self):
        for count in (0, -1, True, 5):
            with self.assertRaises(ValueError):
                pack_inputs((replace(self.rows[0], scheduled=count),), ['A'])

    def test_advanced_scheduler_count_cannot_be_used_as_old_count(self):
        with self.assertRaises(ValueError):
            pack_inputs((replace(self.rows[0], computed=5),), ['A'])

    def test_exact_execution_permutation_required(self):
        for order in ([], ['A'], ['B', 'B'], ['A', 'B', 'A'], ['X', 'B']):
            with self.assertRaises(ValueError): pack_inputs(self.rows, order)

    def test_invalid_block_table_and_shared_storage_rejected(self):
        for blocks in ((8,), (8, 8), (8, 0), (8, 10), (8, 5)):
            with self.assertRaises(ValueError):
                pack_inputs((replace(self.rows[0], block_table=blocks), self.rows[1]), ['A', 'B'])

    def test_padding_has_no_slots_or_query_segment(self):
        b = pack_inputs(self.rows, ['B', 'A'], pad_to=8, padding_token=31)
        self.assertEqual((b.num_actual_tokens, b.num_input_tokens), (4, 8))
        self.assertEqual(b.input_ids[4:], (31, 31, 31, 31))
        self.assertEqual(len(b.slot_mapping), 4)
        self.assertEqual(b.query_start_loc[-1], 4)

    def test_invalid_vocabulary_padding_and_dimensions(self):
        for kwargs in ({'pad_to': 3}, {'padding_token': 32}, {'block_size': 0}, {'num_blocks': True}, {'vocab_size': 20}):
            with self.assertRaises(ValueError): pack_inputs(self.rows, ['A', 'B'], **kwargs)

    def test_input_lists_are_copied_to_immutable_tuples(self):
        ids, blocks = [1, 2], [1]
        r = RequestSlice('C', ids, 0, 2, blocks)
        ids[0], blocks[0] = 30, 9
        self.assertEqual((r.token_ids, r.block_table), ((1, 2), (1,)))

    def test_routing_requires_one_valid_candidate_per_request(self):
        b = pack_inputs(self.rows, ['B', 'A'])
        for candidates in ([], [1], [1, -2], [True, 1]):
            with self.assertRaises(ValueError): route_candidates(b, candidates)

    def test_metadata_mode_never_imports_tensor_execution_or_fetches(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(StringIO()), patch.object(lesson0303, 'tensor_cases', side_effect=AssertionError('not allowed')), patch.object(lesson0303, 'fetch_sources', side_effect=AssertionError('not allowed')):
            self.assertEqual(lesson0303.main(['--mode', 'metadata', '--output-dir', temp]), 0)
            self.assertTrue((Path(temp)/'summary.json').is_file())

    def test_missing_source_reports_correct_opt_in_flag(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(StringIO()), redirect_stderr(StringIO()) as err:
            self.assertEqual(lesson0303.main(['--mode','metadata','--inspect-source','--source-dir',temp,'--output-dir',temp]), 1)
            self.assertIn('--fetch-source', err.getvalue())


if __name__ == '__main__':
    unittest.main()
