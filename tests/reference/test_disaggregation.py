"""Independent hand-calculated oracles and invalid-input checks for 13.1."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.experiments.lesson1301 import DEFAULT_INPUT
from zero2vllm.reference.disaggregation import analyze, percentiles


class DisaggregationTests(unittest.TestCase):
    def setUp(self):
        self.doc = json.loads(DEFAULT_INPUT.read_text(encoding='utf-8'))

    def result(self, name, **overrides):
        return next(r for r in analyze(self.doc, overrides)['runs'] if r['name'] == name)

    def test_hand_calculated_joint_goodput(self):
        result = analyze(self.doc)
        self.assertEqual([r['qualified'] for r in result['runs']], [2, 3, 4, 0, 0, 2])
        self.assertEqual([r['completed'] for r in result['runs']], [4, 4, 4, 4, 4, 3])
        self.assertEqual(result['source_kind'], 'synthetic')

    def test_each_metric_has_its_own_denominator(self):
        row = self.result('mixed_uncontrolled')['requests'][0]
        self.assertEqual(row['ttft_ms'], 100)
        self.assertAlmostEqual(row['tpot_ms'], 190 / 3)
        self.assertEqual(row['max_itl_ms'], 130)
        self.assertEqual(row['violations'], ['tpot_ms', 'max_itl_ms'])

    def test_first_token_can_hide_handoff_from_ttft(self):
        row = self.result('pd_first_token_from_p')['requests'][0]
        self.assertEqual(row['ttft_ms'], 100)
        self.assertAlmostEqual(row['tpot_ms'], 200 / 3)
        self.assertEqual(row['max_itl_ms'], 140)
        self.assertFalse(row['qualified'])

    def test_tpot_does_not_hide_a_single_large_gap(self):
        self.doc['runs'][0]['requests'][0]['token_arrivals_ms'] = [100, 110, 200, 210]
        row = self.result('mixed_uncontrolled')['requests'][0]
        self.assertLess(row['tpot_ms'], 50)
        self.assertEqual(row['violations'], ['max_itl_ms'])

    def test_failures_and_partial_tokens_are_kept(self):
        result = self.result('pd_pool_imbalance')
        self.assertEqual(result['offered'], 4)
        self.assertEqual(result['status_counts'], {'completed': 3, 'incomplete': 1})
        self.assertEqual(result['joint_success_fraction'], 0.5)
        self.assertEqual(result['observed_output_tokens_per_second'], 13)
        self.assertIsNone(result['requests'][-1]['tpot_ms'])

    def test_window_duration_not_last_completion(self):
        self.doc['window']['end_ms'] = 2000
        self.doc['runs'][-1]['requests'][-1]['end_ms'] = 2000
        result = self.result('pd_fast_transfer')
        self.assertEqual(result['goodput_rps'], 2)
        self.assertEqual(result['completed_rps'], 2)

    def test_stricter_slo_changes_goodput_not_throughput(self):
        result = self.result('pd_fast_transfer', ttft_ms=150)
        self.assertEqual(result['goodput_rps'], 0)
        self.assertEqual(result['completed_rps'], 4)

    def test_threshold_equality_qualifies(self):
        self.assertEqual(self.result('pd_fast_transfer', ttft_ms=160, tpot_ms=25, max_itl_ms=25)['qualified'], 4)

    def test_failed_full_output_still_not_qualified(self):
        self.doc['runs'][2]['requests'][0]['status'] = 'error'
        self.assertEqual(self.result('pd_fast_transfer')['qualified'], 3)

    def test_all_failed_populations_are_null(self):
        for r in self.doc['runs'][2]['requests']:
            r.update(status='rejected', token_arrivals_ms=[], end_ms=500)
        result = self.result('pd_fast_transfer')
        self.assertEqual(result['goodput_rps'], 0)
        self.assertEqual(result['completed_request_percentiles']['ttft_ms'], {'p50': None,'p95': None,'p99': None})

    def test_nearest_rank_is_declared_and_hand_checkable(self):
        self.assertEqual(percentiles([100, 230, 110, 120]), {'p50': 110, 'p95': 230, 'p99': 230})

    def test_missing_failed_request_is_rejected(self):
        self.doc['runs'][-1]['requests'].pop()
        with self.assertRaisesRegex(ValueError, 'missing offered'):
            analyze(self.doc)

    def test_unequal_gpu_budget_is_rejected(self):
        self.doc['runs'][2]['allocation'][0]['gpus'] = 2
        with self.assertRaisesRegex(ValueError, 'unequal GPU'):
            analyze(self.doc)

    def test_duplicate_ids_are_rejected(self):
        for field in ('workload', 'run_requests'):
            doc = copy.deepcopy(self.doc)
            rows = doc['workload'] if field == 'workload' else doc['runs'][0]['requests']
            rows.append(copy.deepcopy(rows[0]))
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'duplicate'):
                analyze(doc)

    def test_invalid_window_and_boundary_arrivals(self):
        for change in ('empty', 'arrival_at_end', 'arrival_before_start', 'token_after_end'):
            doc = copy.deepcopy(self.doc)
            if change == 'empty': doc['window']['end_ms'] = 0
            elif change == 'arrival_at_end': doc['workload'][0]['arrival_ms'] = 1000
            elif change == 'arrival_before_start': doc['window']['start_ms'] = 1
            else: doc['runs'][0]['requests'][0]['end_ms'] = 1001
            with self.subTest(change=change), self.assertRaises(ValueError):
                analyze(doc)

    def test_completion_at_end_is_included(self):
        row = self.doc['runs'][2]['requests'][0]
        row.update(token_arrivals_ms=[925, 950, 975, 1000], end_ms=1000)
        result = self.result('pd_fast_transfer', ttft_ms=1000)
        self.assertEqual(result['qualified'], 4)

    def test_nan_infinity_and_booleans_are_rejected(self):
        for bad in (float('nan'), float('inf'), -1, True):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                analyze(self.doc, {'ttft_ms': bad})

    def test_token_order_and_output_contract(self):
        for tokens in ([100, 90, 120, 140], [100], [100]*5):
            self.doc['runs'][0]['requests'][0]['token_arrivals_ms'] = tokens
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                analyze(self.doc)

    def test_single_token_protocol_is_explicitly_rejected(self):
        self.doc['workload'][0]['output_tokens'] = 1
        with self.assertRaisesRegex(ValueError, 'at least two'):
            analyze(self.doc)

    def test_incomplete_record_needs_cutoff(self):
        self.doc['runs'][-1]['requests'][-1]['end_ms'] = 999
        with self.assertRaisesRegex(ValueError, 'observation cutoff'):
            analyze(self.doc)

    def test_mutation_free_analysis(self):
        original = copy.deepcopy(self.doc)
        analyze(self.doc, {'ttft_ms': 300})
        self.assertEqual(self.doc, original)

    def test_cli_writes_report_and_rejects_invalid_slo(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'result.json'
            command = [sys.executable, '-m', 'scripts.experiments.lesson1301', '--output', str(output)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(report['runs'][2]['goodput_rps'], 4)
            self.assertEqual(len(report['input_sha256']), 64)
            crlf = Path(tmp)/'windows-input.json'
            crlf.write_bytes(DEFAULT_INPUT.read_bytes().replace(b'\r\n', b'\n').replace(b'\n', b'\r\n'))
            again = subprocess.run(command + ['--input', str(crlf)], capture_output=True, text=True)
            self.assertEqual(again.returncode, 0, again.stderr)
            self.assertEqual(json.loads(output.read_text(encoding='utf-8'))['input_sha256'], report['input_sha256'])
            before = output.read_bytes()
            bad = subprocess.run(command + ['--ttft-ms', '-1'], capture_output=True, text=True)
            self.assertEqual(bad.returncode, 2)
            self.assertEqual(before, output.read_bytes())


if __name__ == '__main__':
    unittest.main()
