import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from zero2vllm.reference.quantization_overview import (as_fp32,dequantize_logical,linear_qdq,logical_int4_qdq,
    quantize_affine,dequantize_affine,requantize_affine)
from scripts.experiments.lesson0901 import run_experiments


class QuantizationOverviewTests(unittest.TestCase):
    def test_hand_computed_non_square_linear(self):
        r=linear_qdq([[1,2,-1]],[[-3.5,-0.75],[0.25,1.25],[2.5,3.5]])
        self.assertEqual(r['qdq']['scale'],0.5)
        self.assertEqual(r['qdq']['q'],[[-7,-2],[0,2],[5,7]])
        self.assertEqual(r['original_output'],[[-5.5,-1.75]])
        self.assertEqual(r['quantized_output'],[[-6,-2.5]])
        self.assertEqual(r['qdq']['error']['mse'],0.03125)
        self.assertEqual(r['output_max_abs_error'],0.75)

    def test_rounding_halfway_is_even_for_both_signs(self):
        r=logical_int4_qdq([[-1.25,-0.75,-0.25,0.25,0.75,1.25]],scale=0.5)
        self.assertEqual(r['q'],[[-2,-2,0,0,2,2]])

    def test_zero_tensor_uses_defined_nonzero_scale(self):
        r=logical_int4_qdq([[0,0]])
        self.assertEqual(r['scale'],1)
        self.assertEqual(r['q'],[[0,0]])
        self.assertEqual(r['error']['max_abs'],0)

    def test_clipping_is_not_covered_by_rounding_error_bound(self):
        r=logical_int4_qdq([[-4,4]],scale=0.5)
        self.assertEqual(r['q'],[[-7,7]])
        self.assertEqual(r['error']['max_abs'],0.5)
        self.assertFalse(r['error']['half_scale_bound_applicable'])
        self.assertFalse(r['error']['half_scale_bound_satisfied'])

    def test_fp32_input_conversion_is_real(self):
        self.assertEqual(as_fp32(0.1),0.10000000149011612)
        self.assertEqual(as_fp32(0.25),0.25)
        self.assertEqual(logical_int4_qdq([[0.1]])['fp32_input'],[[as_fp32(0.1)]])

    def test_input_unchanged_and_no_packed_payload(self):
        w=[[0.25,3.5]]
        before=[row[:] for row in w]
        r=logical_int4_qdq(w)
        self.assertEqual(w,before)
        self.assertIsInstance(r['q'][0][0],int)
        self.assertNotIn('packed_bytes',r)

    def test_wrong_scale_changes_dequantized_values(self):
        q=[[-7,0,7]]
        self.assertEqual(dequantize_logical(q,0.5),[[-3.5,0,3.5]])
        self.assertEqual(dequantize_logical(q,1),[[-7,0,7]])

    def test_rejects_invalid_matrix_inputs(self):
        for values in ([],[[]],[[1],[1,2]],[[True]],[[float('inf')]],[[float('nan')]],[[1e50]],[[1e-50]],[[2**10000]],'1'):
            with self.subTest(kind=type(values).__name__),self.assertRaises(ValueError):
                logical_int4_qdq(values)

    def test_rejects_invalid_scale_and_logical_values(self):
        for scale in (0,-1,True,float('nan'),float('inf'),'1',1e308,2**10000):
            with self.assertRaises(ValueError): logical_int4_qdq([[1]],scale=scale)
        for q in ([[-8]],[[8]],[[0.5]],[[True]],[[1],[1,2]]):
            with self.assertRaises(ValueError): dequantize_logical(q,0.5)

    def test_rejects_incompatible_linear(self):
        with self.assertRaises(ValueError): linear_qdq([[1,2]],[[1],[2],[3]])

    def test_all_seventeen_experiments_pass(self):
        _,cases,_=run_experiments()
        self.assertEqual(len(cases),17)
        self.assertTrue(all(c['passed'] for c in cases),cases)

    def test_cli_writes_two_passports_and_replayable_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=subprocess.run([sys.executable,'-m','scripts.experiments.lesson0901','--output-dir',tmp],capture_output=True)
            self.assertEqual(run.returncode,0,run.stderr.decode(errors='replace'))
            root=Path(tmp)
            data=json.loads((root/'summary.json').read_text(encoding='utf8'))
            self.assertTrue(data['passed'])
            self.assertFalse(data['gpu_validated'])
            self.assertFalse(data['packed'])
            p=json.loads((root/'qwen-plan-passport.json').read_text(encoding='utf8'))
            self.assertEqual(p['sections']['execution']['actual_kernel']['status'],'unknown')
            self.assertTrue((root/'cpu-passport.json').is_file())

    def test_affine_zero_and_negative_reals_in_unsigned_codes(self):
        params=dict(scale=0.5,zero_point=3)
        self.assertEqual(quantize_affine([[-1.5,0,0.5,6]],**params),[[0,3,4,15]])
        self.assertEqual(dequantize_affine([[0,3,4,15]],**params),[[-1.5,0,0.5,6]])

    def test_odd_zero_point_must_be_added_after_rounding(self):
        self.assertEqual(quantize_affine([[0.25]],scale=0.5,zero_point=3),[[3]])
        self.assertEqual(round(0.25/0.5+3),4)

    def test_zero_point_error_is_a_constant_offset(self):
        q=[[0,3,4,8,15]]
        actual=dequantize_affine(q,scale=0.5,zero_point=3)
        wrong=dequantize_affine(q,scale=0.5,zero_point=0)
        self.assertEqual([b-a for a,b in zip(actual[0],wrong[0])],[1.5]*5)

    def test_requantization_matches_known_codes_with_new_parameters(self):
        q=requantize_affine([[0,3,4,8,15]],source_scale=0.5,source_zero_point=3,target_scale=1,target_zero_point=2)
        self.assertEqual(q,[[0,2,2,4,8]])
        self.assertEqual(dequantize_affine(q,scale=1,zero_point=2),[[-2,0,0,2,6]])

    def test_requantization_respects_target_saturation(self):
        q=requantize_affine([[0,3,4,8,15]],source_scale=0.5,source_zero_point=3,target_scale=1,target_zero_point=2,target_qmax=3)
        self.assertEqual(q,[[0,2,2,3,3]])

    def test_requantization_cannot_recover_preexisting_rounding_loss(self):
        q1=quantize_affine([[0.625]],scale=0.5,zero_point=3)
        via=requantize_affine(q1,source_scale=0.5,source_zero_point=3,target_scale=1,target_zero_point=2)
        direct=quantize_affine([[0.625]],scale=1,zero_point=2)
        self.assertEqual(via,[[2]])
        self.assertEqual(direct,[[3]])

    def test_identity_requantization_preserves_all_codes(self):
        codes=[list(range(16))]
        self.assertEqual(requantize_affine(codes,source_scale=0.5,source_zero_point=3,target_scale=0.5,target_zero_point=3),codes)

    def test_requantization_does_not_insert_fp32_rounding(self):
        # The restored value lies just above 0.5; an unintended FP32 cast
        # would turn it into a tie and incorrectly round down to zero.
        self.assertEqual(requantize_affine([[1]],source_scale=0.50000000001,source_zero_point=0,target_scale=1,target_zero_point=0),[[1]])

    def test_affine_rejects_malformed_contract_and_codes(self):
        for params in (dict(zero_point=0.5),dict(zero_point=True),dict(zero_point=-1),dict(zero_point=16),dict(qmin=15),dict(qmax=0),dict(qmax=float('inf'))):
            with self.subTest(params=params),self.assertRaises(ValueError):
                quantize_affine([[0]],**{**dict(scale=0.5,zero_point=3),**params})
        for q in ([],[[]],[[True]],[[0.5]],[[-1]],[[16]],[[1],[2,3]]):
            with self.assertRaises(ValueError): dequantize_affine(q,scale=0.5,zero_point=3)


if __name__=='__main__': unittest.main()
