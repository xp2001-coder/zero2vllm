import math
import unittest
from zero2vllm.quantization.minifloat import (decode_minifloat,encode_minifloat,
    decode_e8m0,decode_scaled_block)


class MinifloatTests(unittest.TestCase):
    def test_e4m3fn_known_bit_patterns(self):
        for code,value in ((0x38,1),(0x3d,1.625),(0x7e,448),(0xfe,-448),(1,2**-9),(8,2**-6)):
            self.assertEqual(decode_minifloat(code,'e4m3fn'),value)
        self.assertTrue(math.isnan(decode_minifloat(0x7f,'e4m3fn')))

    def test_e5m2_range_and_specials(self):
        for code,value in ((0x3c,1),(0x7b,57344),(1,2**-16),(4,2**-14)):
            self.assertEqual(decode_minifloat(code,'e5m2'),value)
        self.assertEqual(decode_minifloat(0x7c,'e5m2'),math.inf)
        self.assertEqual(decode_minifloat(0xfc,'e5m2'),-math.inf)
        self.assertTrue(math.isnan(decode_minifloat(0x7d,'e5m2')))

    def test_fp4_complete_positive_table_and_negative_zero(self):
        self.assertEqual([decode_minifloat(c,'e2m1') for c in range(8)],[0,0.5,1,1.5,2,3,4,6])
        self.assertEqual(math.copysign(1,decode_minifloat(8,'e2m1')),-1)
        self.assertEqual(decode_minifloat(15,'e2m1'),-6)

    def test_same_byte_has_different_float_meanings(self):
        self.assertEqual(decode_minifloat(0x38,'e4m3fn'),1)
        self.assertEqual(decode_minifloat(0x38,'e5m2'),0.5)

    def test_all_finite_codes_roundtrip_with_sign_of_zero(self):
        for name,size in (('e4m3fn',256),('e5m2',256),('e2m1',16)):
            for code in range(size):
                v=decode_minifloat(code,name)
                if math.isfinite(v): self.assertEqual(encode_minifloat(v,name),code,(name,code))

    def test_special_value_counts(self):
        for name,finite,nan,inf in (('e4m3fn',254,2,0),('e5m2',248,6,2),('e2m1',16,0,0)):
            values=[decode_minifloat(c,name) for c in range(16 if name=='e2m1' else 256)]
            self.assertEqual((sum(map(math.isfinite,values)),sum(map(math.isnan,values)),sum(map(math.isinf,values))),(finite,nan,inf))

    def test_midpoint_rounding_and_negative_values(self):
        self.assertEqual(decode_minifloat(encode_minifloat(1.0625,'e4m3fn'),'e4m3fn'),1)
        self.assertEqual(decode_minifloat(encode_minifloat(1.1875,'e4m3fn'),'e4m3fn'),1.25)
        self.assertEqual(decode_minifloat(encode_minifloat(-2.5,'e2m1'),'e2m1'),-2)

    def test_saturation_and_explicit_rejection(self):
        self.assertEqual(encode_minifloat(500,'e4m3fn'),0x7e)
        self.assertEqual(encode_minifloat(-1e300,'e5m2'),0xfb)
        with self.assertRaises(ValueError): encode_minifloat(449,'e4m3fn',overflow='raise')

    def test_ftz_is_an_explicit_lossy_policy(self):
        self.assertEqual(decode_minifloat(1,'e4m3fn'),2**-9)
        self.assertEqual(decode_minifloat(1,'e4m3fn',ftz=True),0)
        self.assertEqual(decode_minifloat(8,'e4m3fn',ftz=True),2**-6)
        self.assertEqual(math.copysign(1,decode_minifloat(0x81,'e4m3fn',ftz=True)),-1)

    def test_e8m0_is_positive_exponent_not_unsigned_integer(self):
        self.assertEqual([decode_e8m0(c) for c in (0,127,128,254)],[2**-127,1,2,2**127])
        self.assertTrue(math.isnan(decode_e8m0(255)))

    def test_mx_and_nv_use_different_scale_structures(self):
        self.assertEqual(decode_scaled_block([5]*32,kind='mxfp4',scale_code=128),[6]*32)
        self.assertEqual(decode_scaled_block([5]*16,kind='nvfp4',scale_code=0x3c,global_scale=2),[9]*16)

    def test_nv_global_scale_is_actually_rounded_to_fp32(self):
        self.assertEqual(decode_scaled_block([2]*16,kind='nvfp4',scale_code=0x38,global_scale=1.00000003),[1]*16)

    def test_mx_nan_scale_propagates_to_entire_block(self):
        self.assertTrue(all(math.isnan(v) for v in decode_scaled_block([0]*32,kind='mxfp4',scale_code=255)))

    def test_invalid_codes_inputs_and_scale_contracts(self):
        for code in (-1,256,True,1.5):
            with self.assertRaises(ValueError): decode_minifloat(code,'e4m3fn')
        for x in (math.nan,math.inf,True,'1'):
            with self.assertRaises(ValueError): encode_minifloat(x,'e5m2')
        with self.assertRaises(ValueError): decode_minifloat(0,'e4m3')
        with self.assertRaises(ValueError): decode_scaled_block([0]*16,kind='mxfp4',scale_code=127)
        with self.assertRaises(ValueError): decode_scaled_block([0]*16,kind='nvfp4',scale_code=0xb8)
        with self.assertRaises(ValueError): decode_scaled_block([0]*32,kind='mxfp4',scale_code=127,global_scale=2)


if __name__=='__main__': unittest.main()
