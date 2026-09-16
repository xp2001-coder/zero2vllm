import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

from zero2vllm.reference.quantization import (
    Int4Weights, dequantize_weights, matmul, pack_int4, packed_linear,
    quantization_experiment, quantize_value, quantize_weights, unpack_int4,
)


class QuantizationTests(unittest.TestCase):
    def test_signed_nibbles_have_known_wire_bytes(self):
        self.assertEqual(pack_int4([-8, -1, 0, 7, -7]), bytes.fromhex("f87009"))
        self.assertEqual(unpack_int4(bytes.fromhex("f87009"), 5), [-8, -1, 0, 7, -7])
        self.assertEqual(pack_int4([]), b"")

    def test_rounding_saturation_and_zero(self):
        cases = [(0.5, 0), (1.5, 2), (2.5, 2), (-1.5, -2), (-2.5, -2),
                 (100, 7), (-100, -7), (0, 0)]
        for value, expected in cases:
            self.assertEqual(quantize_value(value, 1.0), expected)

    def test_known_linear_and_tail_group(self):
        packed = quantize_weights([[-7], [0], [7], [2]], 3)
        self.assertEqual(packed.data.hex(), "0977")
        self.assertEqual(packed.scales, ((1.0,), (2/7,)))
        self.assertEqual(packed_linear([[1, 2, 3, 4]], packed), [[22.0]])

    def test_zero_column_uses_finite_scale_and_stays_zero(self):
        packed = quantize_weights([[1, 0], [-2, 0], [3, 0]], 2)
        self.assertTrue(all(row[1] == 1.0 for row in packed.scales))
        self.assertEqual([row[1] for row in dequantize_weights(packed)], [0, 0, 0])

    def test_error_bound_and_linear_across_shapes(self):
        rng = random.Random(18)
        for k, n, group in [(1, 1, 8), (5, 3, 2), (8, 4, 8), (17, 5, 4)]:
            w = [[rng.uniform(-9, 9) for _ in range(n)] for _ in range(k)]
            x = [[rng.uniform(-1, 1) for _ in range(k)] for _ in range(3)]
            snapshot = [row[:] for row in w]
            q = quantize_weights(w, group)
            dq = dequantize_weights(q)
            self.assertEqual(w, snapshot)
            self.assertEqual(matmul(x, dq), packed_linear(x, q))
            for i in range(k):
                for j in range(n):
                    self.assertLessEqual(abs(w[i][j]-dq[i][j]), q.scales[i//group][j]/2+1e-12)

    def test_rejects_invalid_packing_and_metadata(self):
        for values in ([-9], [8], [1.5], [True]):
            with self.assertRaises(ValueError): pack_int4(values)
        for data, count in [(b"", 1), (b"\xf0", 1), (b"\x00", -1)]:
            with self.assertRaises(ValueError): unpack_int4(data, count)
        with self.assertRaises(ValueError): Int4Weights(1, 1, 1, b"\x08", ((1.0,),))
        with self.assertRaises(ValueError): Int4Weights(1, 1, 1, b"\x00", ((0.0,),))
        with self.assertRaises(ValueError): Int4Weights(1, 1, 1, b"\x00", ())

    def test_invalid_numeric_inputs_and_shapes(self):
        for w in ([], [[]], [[1], [1, 2]], [[float('nan')]], [[float('inf')]], [[True]]):
            with self.assertRaises(ValueError): quantize_weights(w)
        for group in (0, -1, True, 1.5):
            with self.assertRaises(ValueError): quantize_weights([[1]], group)
        with self.assertRaises(ValueError): packed_linear([[1, 2]], quantize_weights([[1]]))
        with self.assertRaises(ValueError): quantize_weights([[5e-324]])

    def test_report_separates_errors_and_counts_scale_bytes(self):
        r = quantization_experiment()
        self.assertTrue(r['verification']['passed'])
        self.assertEqual(r['verification']['implementation_max_abs_error'], 0)
        self.assertGreater(r['verification']['output_quantization_max_abs_error'], 0)
        self.assertEqual(r['storage']['packed_bytes'], 43)
        self.assertEqual(r['storage']['scale_bytes'], 120)
        self.assertEqual(r['storage']['logical_payload_bytes'], 163)

    def test_cli_outputs_replayable_report_and_rejects_bad_group(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'量化.json'
            p = subprocess.run([sys.executable, '-m', 'zero2vllm', 'quant-demo', '--output', str(output)],
                               capture_output=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertTrue(json.loads(output.read_text(encoding='utf-8'))['verification']['passed'])
            p = subprocess.run([sys.executable, '-m', 'zero2vllm', 'quant-demo', '--group-size', '0'],
                               capture_output=True)
            self.assertEqual(p.returncode, 2)


if __name__ == '__main__':
    unittest.main()
