from dataclasses import replace
import math
import unittest

from zero2vllm.quantization.tensor_contract import (calculate_effective_bpw,cast_scale,
    dequantize,quantize_asymmetric,quantize_symmetric)
from zero2vllm.reference.quantization import matmul

W=[[-3.5,0,0.25],[-0.75,7,1.75],[0.25,1.5,-3.5],[1.75,3.5,0.75],[2.5,14,-0.5]]


class TensorContractTests(unittest.TestCase):
    def test_non_square_grouped_golden_codes(self):
        t=quantize_symmetric(W)
        self.assertEqual(t.shape,(5,3))
        self.assertEqual(t.q,((-7,0,1),(-2,7,7),(1,3,-7),(7,7,2),(7,7,-7)))
        self.assertEqual(t.scales[0],(0.5,1,0.25))
        self.assertEqual(t.scales[1],(0.25,0.5,0.5))

    def test_non_square_linear_outputs(self):
        x=[[1,0,2,-1,0.5],[0,1,0,1,-1]]
        self.assertEqual(matmul(x,W),[[-3.5,6.5,-7.75],[-1.5,-3.5,3]])
        self.assertEqual(matmul(x,dequantize(quantize_symmetric(W))),[[-3.5,6.5,-8],[-1.75,-3.5,3.25]])

    def test_axis_changes_parameter_shape_and_mapping(self):
        k=quantize_symmetric(W,axis=0);n=quantize_symmetric(W,axis=1)
        self.assertEqual(k.to_dict()['parameter_shape'],(3,3))
        self.assertEqual(n.to_dict()['parameter_shape'],(5,2))
        self.assertEqual(k.parameter_index(4,2),(2,2))
        self.assertEqual(n.parameter_index(4,2),(4,1))
        self.assertNotEqual(dequantize(k),dequantize(n))

    def test_per_tensor_and_channel_are_explicit(self):
        scalar=quantize_symmetric(W,axis=None,group_size=None)
        channel=quantize_symmetric(W,axis=0,group_size=None)
        self.assertEqual(scalar.to_dict()['parameter_shape'],(1,1))
        self.assertEqual(scalar.scales,((2.0,),))
        self.assertEqual(channel.to_dict()['parameter_shape'],(1,3))

    def test_per_token_granularity_uses_rows_of_activation(self):
        t=quantize_symmetric([[0,1,7],[0,2,14]],axis=1,group_size=None)
        self.assertEqual(t.scales,((1.0,),(2.0,)))
        self.assertEqual(t.q,((0,1,7),(0,1,7)))

    def test_tail_statistics_use_only_real_cells(self):
        a=quantize_asymmetric(W);b=quantize_asymmetric(W,pad_to_group=True)
        self.assertEqual(a.scales,b.scales)
        self.assertEqual(a.zero_points,b.zero_points)
        self.assertEqual(dequantize(a),dequantize(b))
        self.assertEqual(b.physical_shape,(6,3))

    def test_padding_encodes_real_zero_with_nonzero_z(self):
        t=quantize_asymmetric(W,pad_to_group=True)
        self.assertEqual(t.zero_points[-1],(0,0,15))
        self.assertEqual(t.q[-1],(0,0,15))
        self.assertEqual(dequantize(t,include_padding=True)[-1],[0,0,0])

    def test_wrong_padding_code_is_rejected(self):
        t=quantize_asymmetric(W,pad_to_group=True)
        with self.assertRaisesRegex(ValueError,'Padded'):
            replace(t,q=t.q[:-1]+((0,0,0),))

    def test_column_padding_and_group_larger_than_axis(self):
        t=quantize_symmetric(W,axis=1,pad_to_group=True)
        self.assertEqual(t.physical_shape,(5,4))
        self.assertTrue(all(row[-1]==0 for row in dequantize(t,include_padding=True)))
        large=quantize_symmetric(W,axis=0,group_size=8,pad_to_group=True)
        self.assertEqual(large.physical_shape,(8,3))
        self.assertEqual(dequantize(large),dequantize(quantize_symmetric(W,group_size=None)))

    def test_asymmetric_include_zero_for_one_sided_and_constant_data(self):
        for values,z in (([[2,4]],0),([[-4,-2]],15),([[3,3]],0),([[-3,-3]],15)):
            t=quantize_asymmetric(values,axis=None,group_size=None)
            self.assertEqual(t.zero_points,((z,),))
            self.assertTrue(t.scales[0][0]>0)

    def test_zero_groups_have_defined_scale(self):
        for fn in (quantize_symmetric,quantize_asymmetric):
            t=fn([[0,0],[0,0]])
            self.assertEqual(dequantize(t),[[0,0],[0,0]])
            self.assertTrue(all(s==1 for row in t.scales for s in row))

    def test_asymmetric_golden_nonzero_zero_point(self):
        t=quantize_asymmetric([[-1,0,1,2]],axis=None,group_size=None)
        self.assertEqual(t.zero_points,((5,),))
        self.assertEqual(t.q,((0,5,10,15),))
        self.assertEqual(dequantize(t),[[-1,0,1,2]])

    def test_integer_widths_and_symmetric_narrow_range(self):
        for b in (2,3,4,8):
            t=quantize_symmetric([[-7,0,7]],bits=b,axis=None,group_size=None)
            qmax=2**(b-1)-1
            self.assertEqual(t.q,((-qmax,0,qmax),))
            self.assertEqual(dequantize(t),[[-7,0,7]])

    def test_scale_dtype_is_applied_before_encoding(self):
        t=quantize_symmetric([[0.7]],scale_dtype='float16')
        self.assertEqual(t.scales,((0.0999755859375,),))
        self.assertEqual(dequantize(t),[[7*0.0999755859375]])
        self.assertNotEqual(dequantize(t),dequantize(quantize_symmetric([[0.7]])))

    def test_scale_underflow_overflow_and_invalid_dtype(self):
        for values in ([[1e-8]],[[1e7]]):
            with self.assertRaises(ValueError): quantize_symmetric(values,scale_dtype='float16')
        with self.assertRaises(ValueError): cast_scale(1,'fp8')
        with self.assertRaises(ValueError): cast_scale(0,'float16')

    def test_parameter_axis_and_code_mismatches_are_rejected(self):
        t=quantize_symmetric(W)
        for kwargs in (dict(axis=1),dict(scales=((1,),)),dict(zero_points=((0,),)),dict(q=((0,),)),dict(q=tuple(tuple(8 for _ in row) for row in W))):
            with self.assertRaises(ValueError): replace(t,**kwargs)

    def test_inputs_and_export_are_detached(self):
        values=[[1,2],[3,4]];t=quantize_symmetric(values)
        values[0][0]=999
        self.assertNotEqual(t.q[0][0],999)
        self.assertIsInstance(t.q,tuple)

    def test_invalid_inputs_and_group_contracts(self):
        for values in ([],[[]],[[True]],[[math.nan]],[[math.inf]],[[1],[2,3]]):
            with self.assertRaises(ValueError): quantize_symmetric(values)
        for kwargs in (dict(axis=True),dict(axis=2),dict(group_size=0),dict(group_size=True),dict(bits=5),dict(axis=None),dict(pad_to_group=1)):
            with self.assertRaises(ValueError): quantize_symmetric(W,**kwargs)

    def test_bpw_counts_tail_scale_zero_point_byte_and_alignment_padding(self):
        t=quantize_asymmetric(W,scale_dtype='float16',pad_to_group=True)
        r=calculate_effective_bpw(t,metadata_bytes=4,alignment_bytes=16)
        self.assertEqual([r[k] for k in ('weight_bits','tensor_padding_bits','scale_bits','zero_point_bits','metadata_bits','byte_padding_bits','alignment_padding_bits')],[60,12,144,36,32,4,96])
        self.assertEqual(r['estimated_bytes'],48)
        self.assertEqual(r['effective_bpw'],25.6)

    def test_bpw_can_count_wider_zero_point_storage(self):
        t=quantize_asymmetric(W,scale_dtype='float16')
        self.assertEqual(calculate_effective_bpw(t,zero_point_bits=32)['zero_point_bits'],9*32)

    def test_bpw_zero_overhead_large_group_limit(self):
        t=quantize_symmetric([[0]*128],axis=1,group_size=128,scale_dtype='float16')
        r=calculate_effective_bpw(t)
        self.assertEqual(r['effective_bpw'],4+16/128)

    def test_bpw_rejects_undersized_or_invalid_metadata(self):
        t=quantize_asymmetric(W)
        for kwargs in (dict(zero_point_bits=0),dict(zero_point_bits=2),dict(metadata_bytes=-1),dict(metadata_bytes=True),dict(alignment_bytes=0)):
            with self.assertRaises(ValueError): calculate_effective_bpw(t,**kwargs)


if __name__=='__main__': unittest.main()
