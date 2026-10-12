import importlib.util
import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from zero2vllm.quantization.algorithms import *
from zero2vllm.quantization import QuantizedModelPassport
from scripts.experiments.lesson0904 import load_fixture,run_experiments,torch_check,main


class RoundingTests(unittest.TestCase):
    def setUp(self):
        self.x=[[1,1],[1,0]];self.w=[[.49],[.49]]
        self.p=[AffineParameters(1,0,-7,7,'symmetric')]

    def test_rtn_ties_even_and_saturation(self):
        self.assertEqual(rtn([[-9],[-1.5],[.5],[1.5],[9]],self.p)['q'],[[-7],[-2],[0],[2],[7]])

    def test_nonzero_zero_point_added_after_round(self):
        p=[AffineParameters(.5,3,0,15,'asymmetric')]
        self.assertEqual(rtn([[0],[.25],[-.25]],p)['q'],[[3],[3],[3]])

    def test_per_output_grids_do_not_mix_columns(self):
        g=column_grids([[-7,-14],[7,14]])
        self.assertEqual([p.scale for p in g],[1,2])

    def test_all_zero_grid_is_defined(self):
        self.assertEqual(column_grids([[0],[0]])[0].scale,1)

    def test_exact_four_choices_choose_non_nearest(self):
        r=optimize_rounding(self.x,self.w,self.p)
        self.assertEqual(r['candidates'],4)
        self.assertEqual(r['best']['q'],[[0],[1]])
        self.assertAlmostEqual(r['best']['loss'],.2405)
        self.assertAlmostEqual(layer_metrics(self.x,self.w,r['rtn']['dq'])['sse'],1.2005)

    def test_weight_and_output_objectives_differ(self):
        best=optimize_rounding(self.x,self.w,self.p)['best']['dq']
        self.assertGreater(error_metrics(self.w,best)['mse'],error_metrics(self.w,[[0],[0]])['mse'])

    def test_exact_values_do_not_add_search_variables(self):
        r=optimize_rounding(self.x,[[1],[2]],self.p)
        self.assertEqual(r['candidates'],1);self.assertEqual(r['best']['loss'],0)

    def test_enumeration_guard_is_explicit(self):
        with self.assertRaisesRegex(ValueError,'Too many'):optimize_rounding(self.x,self.w,self.p,max_variables=1)

    def test_clipping_is_scored_against_original(self):
        r=clipping_search([[1]],[[7]],[.5,1])
        self.assertEqual(r['candidates'][0]['objective_mse'],12.25)
        self.assertEqual(r['best']['ratio'],1)

    def test_malformed_matrix_grids_and_ratios_rejected(self):
        for w in ([],[[1],[2,3]],[[math.nan]],[[True]]):
            with self.assertRaises(ValueError):rtn(w)
        with self.assertRaises(ValueError):rtn([[1,2]],self.p)
        with self.assertRaises(ValueError):column_grids([[1]],ratio=0)


class HessianTests(unittest.TestCase):
    def setUp(self):
        self.x=[[1,1],[1,0]];self.w=[[.49],[.49]]
        self.p=[AffineParameters(1,0,-7,7,'symmetric')]

    def test_hessian_shape_factor_and_damping(self):
        h,lam=gram_hessian(self.x,.1)
        self.assertAlmostEqual(lam,.3)
        self.assertEqual(h,[[4.3,2],[2,2.3]])

    def test_inverse_known_two_by_two(self):
        inv=inverse_spd([[4,2],[2,2]])
        for a,b in zip(sum(inv,[]),[.5,-.5,-.5,1]):self.assertAlmostEqual(a,b)

    def test_upper_factor_is_inverse_not_hessian(self):
        r=gptq_reference(self.x,self.w,self.p,damping=0)
        product=matmul(transpose(r['upper']),r['upper'])
        for a,b in zip(sum(product,[]),[.5,-.5,-.5,1]):self.assertAlmostEqual(a,b)

    def test_error_compensation_hand_trace(self):
        r=gptq_reference(self.x,self.w,self.p,damping=0)
        self.assertAlmostEqual(r['trace'][1]['before'][0],.98)
        self.assertEqual(r['q'],[[0],[1]])
        self.assertAlmostEqual(r['layer_sse'],.2405)

    def test_diagonal_gram_reduces_to_rtn(self):
        x=[[1,0],[0,2]]
        self.assertEqual(gptq_reference(x,self.w,self.p,damping=0)['q'],rtn(self.w,self.p)['q'])

    def test_lazy_block_sizes_match_on_non_square_weight(self):
        f=load_fixture()['main'];p=column_grids(f['weight'])
        runs=[gptq_reference(f['calibration'],f['weight'],p,block_size=b) for b in (1,2,3,9)]
        for r in runs:self.assertEqual(r['q'],runs[0]['q'])

    def test_damped_loss_decomposes_into_trace(self):
        r=gptq_reference(self.x,self.w,self.p,damping=.2)
        self.assertGreater(r['regularization_penalty'],0)
        self.assertAlmostEqual(r['trace_objective'],r['regularized_objective'])

    def test_act_order_restores_original_feature_axis(self):
        r=gptq_reference([[0,3],[1,0]],[[.3],[1.2]],self.p,act_order=True,damping=0)
        self.assertEqual(r['order'],[1,0])
        self.assertEqual(r['q'],[[0],[1]])

    def test_rank_deficiency_requires_explicit_damping(self):
        with self.assertRaises(ValueError):gptq_reference([[1,1],[1,1]],self.w,self.p,damping=0)
        self.assertGreater(gptq_reference([[1,1],[1,1]],self.w,self.p)['lambda_absolute'],0)

    def test_all_zero_input_is_not_silently_repaired(self):
        with self.assertRaises(ValueError):gptq_reference([[0,0]],self.w,self.p)

    def test_invalid_factor_inputs_and_controls(self):
        for h in ([[1,2],[0,1]],[[1,2]],[[0]]):
            with self.assertRaises(ValueError):cholesky(h)
        for opts in (dict(damping=-1),dict(block_size=0),dict(act_order=1)):
            with self.assertRaises(ValueError):gptq_reference(self.x,self.w,self.p,**opts)


class TransformTests(unittest.TestCase):
    def test_diagonal_identity_in_non_square_layer(self):
        f=load_fixture()['main'];x,w=f['calibration'],f['weight']
        tx,tw=diagonal_transform(x,w,[.25,2,3])
        self.assertLess(error_metrics(matmul(x,w),matmul(tx,tw))['max_abs'],1e-12)

    def test_omitting_inverse_changes_function(self):
        x,w=[[2,1]],[[1],[1]]
        _,tw=diagonal_transform(x,w,[2,1])
        self.assertEqual(matmul(x,tw),[[5]])
        self.assertEqual(matmul(x,w),[[3]])

    def test_zero_alpha_is_rtn_and_other_alpha_can_worsen(self):
        f=load_fixture()['scaling'];s=scaling_search(f['calibration'],f['weight'])
        self.assertEqual(s['candidates'][0]['effective_w'],rtn(f['weight'])['dq'])
        self.assertGreater(s['candidates'][1]['calibration_mse'],s['candidates'][0]['calibration_mse'])
        self.assertLess(s['best']['calibration_mse'],s['candidates'][0]['calibration_mse'])

    def test_hadamard_is_normalized(self):
        for k in (1,2,4,8):
            h=hadamard(k);a=matmul(transpose(h),h)
            for i in range(k):
                for j in range(k):self.assertAlmostEqual(a[i][j],float(i==j))

    def test_rotation_preserves_output_not_coordinate_peak(self):
        x,w=[[8,0]],[[.25],[1]];rx,rw=rotate(x,w,hadamard(2))
        self.assertLess(error_metrics(matmul(x,w),matmul(rx,rw))['max_abs'],1e-12)
        self.assertAlmostEqual(sum(v*v for v in rx[0]),64)
        self.assertLess(max(abs(v) for v in rx[0]),8)

    def test_relu_does_not_commute_with_rotation(self):
        h=hadamard(2);a=matmul([[-1,1]],h)
        left=[[max(0,v) for v in a[0]]]
        right=matmul([[0,1]],h)
        self.assertGreater(error_metrics(left,right)['max_abs'],.1)

    def test_transform_contracts_reject_invalid_scales(self):
        for s in ([0,1],[-1,1],[1],[math.nan,1]):
            with self.assertRaises(ValueError):diagonal_transform([[1,1]],[[1],[1]],s)
        with self.assertRaises(ValueError):hadamard(3)
        with self.assertRaises(ValueError):rotate([[1,1]],[[1],[1]],[[2,0],[0,1]])


class GradientAndLessonTests(unittest.TestCase):
    def test_chosen_ste_gradients_inside_and_outside(self):
        p=AffineParameters(.5,0,-7,7,'symmetric')
        self.assertAlmostEqual(ste_derivatives(.6,p)['ds'],-.2)
        self.assertEqual(ste_derivatives(.6,p)['dx'],1)
        self.assertEqual(ste_derivatives(4,p),dict(dx=0.0,ds=7.0))
        self.assertEqual(ste_derivatives(-4,p),dict(dx=0.0,ds=-7.0))

    def test_stochastic_rounding_exact_expectation_and_clipping(self):
        p=AffineParameters(1,0,-7,7,'symmetric')
        class Draw:
            def __init__(self,v):self.v=v
            def random(self):return self.v
        self.assertEqual([stochastic_code(.25,p,Draw(v)) for v in (.1,.3)],[1,0])
        self.assertEqual(stochastic_code(99,p,Draw(.1)),7)
        self.assertEqual(stochastic_code(-.25,p,Draw(.8)),-1)

    def test_twenty_three_fixed_cases(self):
        r=run_experiments();self.assertEqual(len(r['cases']),23);self.assertTrue(r['passed'],r['cases'])

    def test_cli_passport_keeps_model_quality_unknown(self):
        with tempfile.TemporaryDirectory() as out,patch('builtins.print'):
            self.assertEqual(main(['--output-dir',out]),0)
            p=json.loads((Path(out)/'cpu-passport.json').read_text(encoding='utf8'))
            self.assertEqual(QuantizedModelPassport.from_dict(p).to_dict(),p)
            self.assertEqual(p['sections']['validation']['model_quality']['status'],'unknown')

    @unittest.skipUnless(importlib.util.find_spec('torch'),'Optional PyTorch math/STE oracle')
    def test_independent_torch_oracles(self):
        r=torch_check(load_fixture());self.assertTrue(r['passed'],r)


if __name__=='__main__':unittest.main()
