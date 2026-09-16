import copy
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

from zero2vllm.reference.attention import TinyAttentionLM
from zero2vllm.reference.phases import linear_cost, phase_experiment
from zero2vllm.reference.kv_geometry import cache_experiment, grouped_attention, kv_capacity
from zero2vllm.reference.metrics import metrics_experiment, percentile
from zero2vllm.reference.positions import causal_attention, linear_attention, rope, ssm_scan, structure_experiment
from zero2vllm.reference.sampling import SamplingConfig, compare_logits, distribution, draw, generate_sampled

ROOT = Path(__file__).resolve().parents[2]


class PhaseTests(unittest.TestCase):
    def test_gemm_flops_bytes_and_roofline_by_hand(self):
        cost = linear_cost(2, 3, 4, element_bytes=2, peak_tflops=1, bandwidth_gbps=1)
        self.assertEqual(cost['flops'], 48)
        self.assertEqual(cost['ideal_transfer_bytes'], 52)
        self.assertAlmostEqual(cost['arithmetic_intensity_flops_per_byte'], 48 / 52)
        self.assertAlmostEqual(cost['roofline_lower_bound_ms'], 52 / 1e6)
        self.assertEqual(cost['limiting_bound'], 'transfer')

    def test_phase_steps_and_last_unconsumed_token(self):
        report = phase_experiment([1, 2, 3, 4])
        self.assertEqual([x['projected_positions'] for x in report['steps']], [4, 1, 1, 1, 1, 1, 1, 1])
        self.assertEqual([x['attention_score_pairs'] for x in report['steps']], [10, 5, 6, 7, 8, 9, 10, 11])
        self.assertEqual(report['cached_work']['projected_token_positions'], 11)
        self.assertEqual(report['full_recompute_work']['projected_token_positions'], 60)
        self.assertEqual(report['steps'][-1]['absolute_positions'], [10])
        self.assertTrue(report['verification']['passed'])
        self.assertEqual(len(phase_experiment([1], 1)['steps']), 1)

    def test_invalid_estimates_and_trace_budget(self):
        for kwargs in ({'tokens': 0}, {'input_width': True}, {'peak_tflops': float('nan')}, {'bandwidth_gbps': -1}):
            options = {'tokens': 1, 'input_width': 8, 'output_width': 8, **kwargs}
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                linear_cost(**options)
        with self.assertRaises(ValueError):
            phase_experiment([1], 0)
        with self.assertRaises(ValueError):
            phase_experiment([1], 129)


class CacheGeometryTests(unittest.TestCase):
    def test_mha_gqa_mqa_capacity_and_scaling(self):
        self.assertEqual(kv_capacity(kv_heads=32)['payload_gib'], 2)
        self.assertEqual(kv_capacity()['payload_mib'], 512)
        self.assertEqual(kv_capacity(kv_heads=1)['payload_mib'], 64)
        self.assertEqual(kv_capacity(batch_size=3, element_bytes=1)['payload_mib'], 768)
        for kwargs in ({'kv_heads': 3}, {'sequence_length': 0}, {'element_bytes': True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                kv_capacity(**kwargs)

    def test_zero_queries_produce_group_value_means(self):
        # Each head must select its group's values; zero scores give a known mean.
        output = grouped_attention([[0.0], [0.0], [0.0], [0.0]],
                                   [[[2.0], [7.0]], [[4.0], [9.0]]],
                                   [[[1.0], [10.0]], [[3.0], [20.0]]])
        self.assertEqual(output, [[2.0], [2.0], [15.0], [15.0]])
        with self.assertRaises(ValueError):
            grouped_attention([[0.0]] * 3, [[[1.0], [2.0]]], [[[1.0], [2.0]]])

    def test_uneven_chunks_offsets_and_input_preservation(self):
        prompt = [1, 2, 3, 4, 5]
        for size in (1, 2, 5, 8):
            report = cache_experiment(prompt, size)
            self.assertEqual([x['position_offset'] for x in report['chunks']], list(range(0, 5, size)))
            self.assertEqual(len(report['cached_keys']), 5)
            self.assertTrue(report['verification']['passed'])
        self.assertEqual(prompt, [1, 2, 3, 4, 5])


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / 'docs/chapters/part-01-inference/04-performance-metrics/data/request-events.json').read_text(encoding='utf-8'))

    def test_hand_computed_timeline_and_goodput(self):
        result = metrics_experiment(self.data)
        a, b, c, d = result['request_metrics']
        self.assertAlmostEqual(a['ttft_s'], 0.5)
        self.assertAlmostEqual(a['tpot_s'], 0.25)
        self.assertAlmostEqual(a['itl_s'][0], 0.2)
        self.assertIsNone(b['tpot_s'])
        self.assertFalse(c['meets_slo'])
        self.assertFalse(d['meets_slo'])
        self.assertAlmostEqual(result['rates']['output_tokens_per_s'], 6 / 2.3)
        self.assertAlmostEqual(result['rates']['goodput_requests_per_s'], 2 / 2.3)
        self.assertEqual(result['rates']['good_fraction_all_requests'], 0.5)
        self.assertAlmostEqual(result['distributions']['ttft_s']['p95'], 1.16)
        self.assertAlmostEqual(result['distributions']['tpot_s']['mean'], 0.425)
        self.assertAlmostEqual(result['distributions']['itl_s']['mean'], 1.1 / 3)

    def test_percentiles_empty_singleton_and_interpolation(self):
        self.assertIsNone(percentile([], 0.99))
        self.assertEqual(percentile([4], 0.99), 4)
        self.assertEqual(percentile([30, 0, 20, 10], 0.5), 15)
        self.assertAlmostEqual(percentile([30, 0, 20, 10], 0.95), 28.5)
        with self.assertRaises(ValueError):
            percentile([float('nan')], 0.5)

    def test_failed_partial_tokens_do_not_inflate_throughput(self):
        self.data['requests'][3]['token_times_s'] = [1.5, 1.6]
        result = metrics_experiment(self.data)
        self.assertEqual(result['counts']['successful_output_tokens'], 6)
        self.assertEqual(result['counts']['failed'], 1)
        self.assertEqual(result['request_metrics'][3]['output_tokens'], 2)
        self.assertEqual(result['distributions']['ttft_s']['count'], 3)

    def test_all_failed_and_single_token_slo_policy(self):
        self.data['requests'] = [self.data['requests'][3]]
        result = metrics_experiment(self.data)
        self.assertEqual(result['rates']['goodput_requests_per_s'], 0)
        self.assertIsNone(result['distributions']['ttft_s']['p50'])
        self.assertEqual(result['rates']['output_tokens_per_s'], 0)

    def test_reject_bad_clocks_duplicates_chunks_and_empty_success(self):
        for data in ([], {'schema_version': 1, 'timestamp_kind': 'per_token_same_clock',
                          'record_type': 'simulation', 'requests': [None]}):
            with self.assertRaises(ValueError):
                metrics_experiment(data)
        mutations = [lambda d: d.update(timestamp_kind='stream_chunk'),
                     lambda d: d['requests'][0].update(token_times_s=[0.9, 0.5]),
                     lambda d: d['requests'][0].update(work_started_s=1.2),
                     lambda d: d['requests'][0].update(token_times_s=[]),
                     lambda d: d['requests'][1].update(id='A'),
                     lambda d: d['requests'][0].update(arrival_s=float('nan'))]
        for mutation in mutations:
            data = copy.deepcopy(self.data)
            mutation(data)
            with self.assertRaises(ValueError):
                metrics_experiment(data)


class StructureTests(unittest.TestCase):
    def test_rope_known_rotation_norm_and_relative_positions(self):
        output = rope([1.0, 0.0], 1)
        self.assertAlmostEqual(output[0], math.cos(1))
        self.assertAlmostEqual(output[1], math.sin(1))
        self.assertEqual(rope([1.0, 2.0, 3.0, 4.0], 0), [1.0, 2.0, 3.0, 4.0])
        for size in (1, 4, 8):
            self.assertTrue(structure_experiment(size)['verification']['passed'])
        with self.assertRaises(ValueError):
            rope([1, 2, 3], 0)

    def test_window_membership_counts_and_full_limit(self):
        q, k, v = [[0.0]] * 4, [[0.0]] * 4, [[1.0], [2.0], [3.0], [4.0]]
        output = causal_attention(q, k, v, window=2)
        self.assertEqual(output['outputs'], [[1.0], [1.5], [2.5], [3.5]])
        self.assertEqual(output['score_pairs'], 7)
        self.assertEqual(causal_attention(q, k, v), causal_attention(q, k, v, window=10))
        self.assertEqual(causal_attention(q, k, v, window=1)['outputs'], v)

    def test_alibi_hand_calculation_and_causality(self):
        result = causal_attention([[0.0]] * 2, [[0.0]] * 2, [[1.0], [3.0]], alibi_slope=math.log(2))
        self.assertAlmostEqual(result['outputs'][1][0], 7 / 3)
        self.assertEqual(result['visible_positions'][0], [0])

    def test_linear_attention_recurrence_and_non_softmax_semantics(self):
        q, k, v = [[1.0], [2.0]], [[0.0], [1.0]], [[1.0], [3.0]]
        recurrent = linear_attention(q, k, v)
        self.assertAlmostEqual(recurrent[-1][0], 7 / 3)
        self.assertEqual(recurrent, linear_attention(q, k, v, recurrent=False))
        self.assertNotAlmostEqual(recurrent[-1][0], causal_attention(q, k, v)['outputs'][-1][0])

    def test_ssm_scan_matches_closed_form(self):
        result = ssm_scan([1, 2, 3])
        self.assertEqual(result['states'], [1, 2.5, 4.25])
        self.assertEqual(result['outputs'], [2, 5, 8.5])
        self.assertEqual(result['outputs'], result['convolution_reference'])


class ScriptedModel(TinyAttentionLM):
    """Control logits independently of the sampler to test stopping transitions."""
    def __init__(self, tokens):
        super().__init__(vocab_size=3, width=2, max_context=16)
        self.tokens, self.calls = tokens, 0

    def forward_cached(self, tokens, cache, counter):
        self.validate(tokens, len(cache.keys))
        cache.keys.extend([[0, 0] for _ in tokens])
        cache.values.extend([[0, 0] for _ in tokens])
        logits = [-100.0] * 3
        logits[self.tokens[self.calls]] = 100.0
        self.calls += 1
        return [logits for _ in tokens]


class SamplingTests(unittest.TestCase):
    def test_top_k_top_p_min_p_known_probabilities(self):
        logits = [math.log(p) for p in (0.4, 0.3, 0.2, 0.1)]
        self.assertEqual(distribution(logits, SamplingConfig(1, top_k=2, top_p=1))['kept_token_ids'], [0, 1])
        p = distribution(logits, SamplingConfig(1, top_p=0.8))['probabilities']
        for actual, expected in zip(p, [4 / 9, 3 / 9, 2 / 9, 0]):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(distribution(logits, SamplingConfig(1, top_p=1, min_p=0.6))['kept_token_ids'], [0, 1])
        # top-p is applied to the normalized top-k subset in this teaching contract.
        self.assertEqual(distribution(logits, SamplingConfig(1, top_k=2, top_p=0.5))['kept_token_ids'], [0])

    def test_sign_aware_penalties_have_distinct_histories(self):
        config = SamplingConfig(1, top_p=1, repetition_penalty=2, presence_penalty=0.5, frequency_penalty=0.25)
        result = distribution([4, -2, 1], config, prompt=[0, 1], generated=[1, 1, 2])
        self.assertEqual(result['penalized_logits'], [2, -5, -0.25])

    def test_zero_temperature_masks_and_ties(self):
        self.assertEqual(distribution([2, 2, 1], SamplingConfig(0, top_p=0.1))['probabilities'], [1, 0, 0])
        self.assertEqual(distribution([2, 2, 1], SamplingConfig(0), blocked={0})['probabilities'], [0, 1, 0])
        with self.assertRaises(ValueError):
            distribution([-math.inf, -math.inf], SamplingConfig())

    def test_stable_softmax_shift_and_invalid_inputs(self):
        a = distribution([10000, 9999], SamplingConfig(1, top_p=1))['probabilities']
        b = distribution([0, -1], SamplingConfig(1, top_p=1))['probabilities']
        self.assertEqual(a, b)
        for logits in ([math.nan, 0], [math.inf, 0], []):
            with self.assertRaises(ValueError):
                distribution(logits, SamplingConfig())
        for config in (SamplingConfig(-1), SamplingConfig(top_k=-1), SamplingConfig(top_p=0),
                       SamplingConfig(min_p=1.1), SamplingConfig(repetition_penalty=0)):
            with self.assertRaises(ValueError):
                distribution([1, 2], config)

    def test_seed_replay_and_empirical_distribution(self):
        a, b = random.Random(7), random.Random(7)
        seq = [draw([0.4, 0.3, 0.2, 0.1], a) for _ in range(20000)]
        self.assertEqual(seq, [draw([0.4, 0.3, 0.2, 0.1], b) for _ in range(20000)])
        for token, probability in enumerate((0.4, 0.3, 0.2, 0.1)):
            self.assertLess(abs(seq.count(token) / len(seq) - probability), 0.02)

    def test_eos_stop_ids_and_minimum_length(self):
        options = {'prompt': [0], 'vocabulary': ('甲', '乙', '丙'), 'config': SamplingConfig(0), 'max_new_tokens': 3}
        result = generate_sampled(ScriptedModel([2]), eos_token_id=2, **options)
        self.assertEqual((result['new_token_ids'], result['generated_text'], result['finish_reason']), ([2], '', 'eos'))
        result = generate_sampled(ScriptedModel([1]), stop_token_ids=(1,), **options)
        self.assertEqual(result['finish_reason'], 'stop_token')
        result = generate_sampled(ScriptedModel([2, 2]), eos_token_id=2, min_new_tokens=1, **options)
        self.assertEqual(result['new_token_ids'], [0, 2])
        self.assertEqual(result['generated_text'], '甲')

    def test_stop_string_crosses_piece_boundary_and_ignores_prompt(self):
        options = {'prompt': [0], 'vocabulary': ('甲', '乙', '丙'), 'config': SamplingConfig(0), 'max_new_tokens': 3}
        result = generate_sampled(ScriptedModel([2, 1]), stop_strings=('丙乙',), **options)
        self.assertEqual(result['new_token_ids'], [2, 1])
        self.assertEqual(result['generated_text'], '')
        self.assertEqual(result['finish_reason'], 'stop_string')
        result = generate_sampled(ScriptedModel([2, 2, 2]), stop_strings=('甲',), **options)
        self.assertEqual(result['finish_reason'], 'length')

    def test_zero_output_and_invalid_stopping_budget(self):
        model = ScriptedModel([])
        result = generate_sampled(model, [0], ('甲', '乙', '丙'), SamplingConfig(), max_new_tokens=0)
        self.assertEqual(result['new_token_ids'], [])
        self.assertEqual(model.calls, 0)
        with self.assertRaises(ValueError):
            generate_sampled(model, [0], ('甲', '乙', '丙'), SamplingConfig(), stop_strings=('',))

    def test_logit_tolerance_and_token_equality_are_independent(self):
        a = compare_logits([2, 1], [20, 1])
        self.assertTrue(a['same_argmax'])
        self.assertFalse(a['all_within_tolerance'])
        b = compare_logits([1, 1.0000001], [1.0000002, 1])
        self.assertFalse(b['same_argmax'])
        self.assertTrue(b['all_within_tolerance'])
        with self.assertRaises(ValueError):
            compare_logits([1], [1, 2])


class Part1CLITests(unittest.TestCase):
    def test_commands_save_valid_reports_in_unicode_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            for command in ('phase-demo', 'kv-demo', 'metrics-demo', 'structure-demo', 'sampling-demo'):
                with self.subTest(command=command):
                    output = Path(folder) / '第一篇' / f'{command}.json'
                    result = subprocess.run([sys.executable, '-m', 'zero2vllm', command, '--output', str(output)],
                                            cwd=ROOT, capture_output=True, text=True, encoding='utf-8',
                                            env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
                    self.assertEqual(result.returncode, 0, result.stderr)
                    data = json.loads(output.read_text(encoding='utf-8'))
                    self.assertEqual(data, json.loads(result.stdout))
                    self.assertTrue(data['verification']['passed'])

    def test_bad_arguments_fail_without_report(self):
        for command in (['phase-demo', '--width', '0'], ['kv-demo', '--chunk-size', '0'],
                        ['metrics-demo', '--input', 'no-such-timeline.json'], ['structure-demo', '--window', '0'],
                        ['sampling-demo', '--top-p', '0']):
            result = subprocess.run([sys.executable, '-m', 'zero2vllm', *command], cwd=ROOT,
                                    capture_output=True, text=True, encoding='utf-8')
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertEqual(result.stdout, '')


if __name__ == '__main__':
    unittest.main()
