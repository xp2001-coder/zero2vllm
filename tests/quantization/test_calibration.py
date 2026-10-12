import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from zero2vllm.quantization.calibration import (RangeObserver,FixedHistogram,parameters_from_range,
    prepare_samples,valid_rows,feature_rows,validate_corpus,percentile_threshold,mse_clip_search,
    error_metrics,classification_metrics)
from scripts.experiments.lesson0903 import load_fixture,run_experiments,objective_examples,torch_check,main
from zero2vllm.quantization import QuantizedModelPassport


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.fixture=load_fixture()
        self.records=self.fixture['records']

    def prepare(self,**kwargs):
        return prepare_samples(self.records,**dict(split='calibration',domain='general',sample_count=4,max_length=8,seed=0,**kwargs))

    def test_repeat_is_identical(self):
        self.assertEqual(self.prepare(),self.prepare())

    def test_selected_documents_are_nested_for_counts(self):
        a=prepare_samples(self.records,split='calibration',domain='general',sample_count=2,seed=0)
        self.assertEqual(a['sample_ids'],self.prepare()['sample_ids'][:2])

    def test_mask_counts_include_specials_not_padding(self):
        p=self.prepare()
        self.assertEqual(p['valid_tokens'],24)
        self.assertEqual(p['body_tokens']+p['bos_tokens']+p['eos_tokens'],24)
        self.assertEqual(p['padding_tokens'],8)

    def test_exact_budget_cuts_final_sample(self):
        p=self.prepare(token_budget=16)
        self.assertEqual(p['valid_tokens'],16)
        self.assertEqual(p['used_samples'],3)
        self.assertLess(p['eos_tokens'],p['used_samples'])
        self.assertEqual(sum(sum(m) for m in p['attention_mask']),16)

    def test_length_truncation_reserves_eos(self):
        p=prepare_samples(self.records,split='calibration',domain='general',sample_count=8,max_length=4)
        self.assertTrue(all(row[0]==1 and row[-1]==2 for row in p['input_ids']))
        self.assertGreater(p['body_tokens_dropped_by_length'],0)

    def test_specials_can_be_disabled(self):
        p=self.prepare(add_bos=False,add_eos=False)
        self.assertEqual(p['bos_tokens']+p['eos_tokens'],0)

    def test_cross_split_content_leakage_rejected(self):
        records=copy.deepcopy(self.records)
        records[-1]['token_ids']=records[0]['token_ids'][:]
        with self.assertRaisesRegex(ValueError,'leaks'):validate_corpus(records)

    def test_duplicate_ids_rejected(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):validate_corpus(self.records+[self.records[0]])

    def test_impossible_budget_or_insufficient_data_rejected(self):
        with self.assertRaises(ValueError):self.prepare(token_budget=1000)
        with self.assertRaises(ValueError):prepare_samples(self.records,split='calibration',sample_count=100)

    def test_invalid_token_and_preprocessing_contracts(self):
        records=copy.deepcopy(self.records);records[0]['token_ids']=[True]
        with self.assertRaises(ValueError):validate_corpus(records)
        for options in (dict(max_length=2),dict(seed=True),dict(add_eos=1),dict(token_budget=0)):
            args=dict(split='calibration',sample_count=4,max_length=8);args.update(options)
            with self.assertRaises(ValueError):prepare_samples(self.records,**args)

    def test_feature_table_and_padding_sentinel(self):
        p=self.prepare();rows,mask=feature_rows(p,self.fixture['features'])
        self.assertEqual(len(rows),32)
        self.assertEqual(len(valid_rows(rows,mask)),24)
        self.assertTrue(all(abs(v)<999 for row in valid_rows(rows,mask) for v in row))

    def test_mask_shape_and_boolean_type_are_checked(self):
        for mask in ([True,False],[1]):
            with self.assertRaises(ValueError):valid_rows([[1]],mask)
        self.assertEqual(valid_rows([[math.nan]],[False]),[])


class ObserverTests(unittest.TestCase):
    def test_global_minmax_is_batch_order_invariant(self):
        for order in ([[[-1,1]],[[-4,8]]],[[[-4,8]],[[-1,1]]]):
            o=RangeObserver()
            for b in order:o.update(b)
            self.assertEqual((o.minimum,o.maximum),(-4,8))

    def test_ema_hand_calculation(self):
        o=RangeObserver(mode='ema',alpha=.25)
        o.update([[-1,1]]);o.update([[-4,8]])
        self.assertEqual((o.minimum,o.maximum),(-1.75,2.75))

    def test_ema_order_and_partition_matter(self):
        o=RangeObserver(mode='ema',alpha=.25);o.update([[-4,8]]);o.update([[-1,1]])
        self.assertEqual((o.minimum,o.maximum),(-3.25,6.25))
        o.reset();o.update([[-1,1],[-4,8]])
        self.assertEqual((o.minimum,o.maximum),(-4,8))

    def test_freeze_then_reset(self):
        o=RangeObserver();o.update([[-1,2]]);o.freeze()
        with self.assertRaisesRegex(ValueError,'frozen'):o.update([[99]])
        o.reset();self.assertEqual(o.tokens,0);self.assertFalse(o.frozen)

    def test_masked_empty_batch_does_not_initialize(self):
        o=RangeObserver();o.update([[999]],[False])
        self.assertIsNone(o.minimum);self.assertEqual(o.updates,0)
        with self.assertRaises(ValueError):o.freeze()

    def test_asymmetric_zero_point_golden(self):
        p=parameters_from_range(-1,2,scheme='asymmetric')
        self.assertEqual((p.scale,p.zero_point),(.2,5))
        self.assertEqual(p.qdq([[-1,0,1,2]])['q'],[[0,5,10,15]])

    def test_zero_range_has_defined_scale(self):
        for scheme in ('symmetric','asymmetric'):
            p=parameters_from_range(0,0,scheme=scheme)
            self.assertEqual((p.scale,p.zero_point),(1,0))
            self.assertEqual(p.qdq([[0]])['dq'],[[0]])

    def test_saturation_and_rounding(self):
        p=parameters_from_range(-3.5,3.5)
        r=p.qdq([[-4,-.75,.25,.75,4]])
        self.assertEqual(r['q'],[[-7,-2,0,2,7]])
        self.assertEqual(r['saturation_fraction'],.4)

    def test_histogram_edges_and_out_of_range(self):
        h=FixedHistogram(-2,2,4);h.update([[-3,-2,-1,0,1,2,3]])
        self.assertEqual(h.counts,[1,1,1,2]);self.assertEqual((h.underflow,h.overflow),(1,1))
        self.assertEqual(h.state()['edges'],[-2,-1,0,1,2])

    def test_histogram_mask_and_incremental_counts(self):
        h=FixedHistogram(0,2,2);h.update([[0],[100]],[True,False]);h.update([[2]])
        self.assertEqual(h.counts,[1,1]);self.assertEqual(h.overflow,0)

    def test_percentile_convention_is_nearest_rank(self):
        self.assertEqual(percentile_threshold([[-1,2,3,100]],.75),3)
        self.assertEqual(percentile_threshold([[-1,2,3,100]],1),100)

    def test_mse_search_scores_original_not_clipped_target(self):
        r=mse_clip_search([[7]], [.5,7])
        self.assertEqual(r['candidates'][0]['mse'],42.25)
        self.assertEqual(r['best']['threshold'],7)

    def test_search_tie_prefers_larger_threshold(self):
        self.assertEqual(mse_clip_search([[0]],[1,2])['best']['threshold'],2)

    def test_invalid_observer_and_search_configuration(self):
        for alpha in (0,-1,2,math.nan,True):
            with self.assertRaises(ValueError):RangeObserver(mode='ema',alpha=alpha)
        for thresholds in ([],[0],[math.inf]):
            with self.assertRaises(ValueError):mse_clip_search([[1]],thresholds)
        with self.assertRaises(ValueError):FixedHistogram(1,1,4)


class MetricTests(unittest.TestCase):
    def test_metrics_golden(self):
        m=error_metrics([[1,2]],[[1,3]])
        self.assertEqual((m['mse'],m['sse'],m['max_abs'],m['nmse']),(.5,1,1,.2))
        self.assertAlmostEqual(m['cosine'],7/math.sqrt(50))

    def test_zero_norm_null_instead_of_fake_perfect_cosine(self):
        m=error_metrics([[0,0]],[[0,0]])
        self.assertEqual(m['mse'],0);self.assertIsNone(m['nmse']);self.assertIsNone(m['cosine'])
        self.assertIsNone(error_metrics([[1]],[[0]])['cosine'])

    def test_cosine_cannot_detect_scale_error(self):
        m=error_metrics([[1,2]],[[2,4]])
        self.assertAlmostEqual(m['cosine'],1);self.assertEqual(m['nmse'],1)

    def test_shapes_and_nonfinite_are_not_silently_accepted(self):
        for candidate in ([[1]],[[1,2],[3]],[[math.nan],[1]]):
            with self.assertRaises(ValueError):error_metrics([[1],[2]],candidate)

    def test_weight_and_layer_ranking_reverses(self):
        r=objective_examples()['weight_vs_layer']
        self.assertEqual(r['A']['weight']['mse'],.0078125)
        self.assertEqual(r['B']['weight']['mse'],.03125)
        self.assertEqual(r['A']['layer']['mse'],1)
        self.assertEqual(r['B']['layer']['mse'],0)

    def test_fp_and_quantized_upstream_ranking_reverses(self):
        r=objective_examples()['reconstruction']
        self.assertEqual(r['A']['fp_input']['mse'],0)
        self.assertEqual(r['B']['fp_input']['mse'],1)
        self.assertEqual(r['A']['quantized_upstream']['mse'],1)
        self.assertEqual(r['B']['quantized_upstream']['mse'],.0625)

    def test_low_logit_mse_can_have_lower_accuracy(self):
        r=objective_examples()['logits_vs_accuracy']
        self.assertEqual(r['A']['error']['mse'],.0078125)
        self.assertEqual(r['B']['error']['mse'],.28515625)
        self.assertEqual(r['A']['task']['accuracy'],.5)
        self.assertEqual(r['B']['task']['accuracy'],1)

    def test_nll_uniform_and_mask(self):
        m=classification_metrics([[0,0],[0,0],[999,0]],[0,1,-100],[True,True,False])
        self.assertEqual(m['valid_targets'],2);self.assertAlmostEqual(m['perplexity'],2)
        self.assertAlmostEqual(m['nll_sum'],2*math.log(2))

    def test_nll_is_stable_for_large_logits(self):
        for offset in (10000,1e20,-1e20):
            with self.subTest(offset=offset):
                m=classification_metrics([[offset,offset]],[0])
                self.assertAlmostEqual(m['mean_nll'],math.log(2))
                self.assertAlmostEqual(m['perplexity'],2)

    def test_nll_rejects_missing_or_invalid_targets(self):
        for labels,mask in (([],None),([2],None),([0],[False])):
            with self.assertRaises(ValueError):classification_metrics([[0,0]],labels,mask)


class Lesson0903Tests(unittest.TestCase):
    def test_twenty_controlled_cases(self):
        r=run_experiments();self.assertTrue(r['passed'],r['cases']);self.assertEqual(len(r['cases']),20)

    def test_cli_protocol_and_passport_roundtrip(self):
        with tempfile.TemporaryDirectory() as out,patch('builtins.print'):
            self.assertEqual(main(['--output-dir',out]),0)
            p=json.loads((Path(out)/'cpu-passport.json').read_text(encoding='utf8'))
            self.assertEqual(QuantizedModelPassport.from_dict(p).to_dict(),p)
            self.assertEqual(p['sections']['validation']['model_quality']['status'],'unknown')
            protocol=json.loads((Path(out)/'protocol.json').read_text(encoding='utf8'))
            self.assertEqual(protocol['valid_tokens'],50)
            self.assertEqual(protocol['used_samples'],8)

    @unittest.skipUnless(importlib.util.find_spec('torch'),'Optional PyTorch observer comparison')
    def test_observers_against_pytorch(self):
        r=torch_check();self.assertTrue(r['passed'],r)


if __name__=='__main__':unittest.main()
