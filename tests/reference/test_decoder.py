import unittest

from zero2vllm.reference.decoder import DecoderConfig, ScalarDecoder, rms_norm, rope


class ScalarDecoderTests(unittest.TestCase):
    def test_config_and_token_contract(self):
        for kwargs in ({'width': 7}, {'heads': 3}, {'width': 6, 'heads': 2},
                       {'layers': 0}, {'eps': 0}, {'rope_base': float('nan')}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                DecoderConfig(**kwargs)
        model = ScalarDecoder()
        for tokens in ([], [-1], [16], [True], [1.0], [0]*33):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                model.forward(tokens)

    def test_rms_norm_hand_calculation_and_zero(self):
        # sqrt(mean([3^2,4^2])+3.5) = 4.
        self.assertEqual(rms_norm([[3., 4.]], [2., 1.], 3.5), [[1.5, 1.]])
        self.assertEqual(rms_norm([[0., 0.]], [1., 1.], 1e-6), [[0., 0.]])

    def test_rope_preserves_norm_in_each_head(self):
        rows = [[1., 2., 3., 4.]]*3
        rotated = rope(rows, heads=2, base=10000.)
        self.assertEqual(rotated[0], rows[0])
        for row in rotated:
            self.assertAlmostEqual(sum(x*x for x in row[:2]), 5.)
            self.assertAlmostEqual(sum(x*x for x in row[2:]), 25.)

    def test_future_tokens_do_not_change_prefix(self):
        model = ScalarDecoder()
        a, _ = model.forward([1, 2, 3])
        b, _ = model.forward([1, 2, 15, 9])
        self.assertEqual(a[:2], b[:2])

    def test_generation_boundary_and_input_ownership(self):
        model = ScalarDecoder(DecoderConfig(max_context=4))
        prompt = [1, 2, 3, 4]
        self.assertEqual(model.generate(prompt, 0), [])
        self.assertEqual(len(model.generate(prompt, 1)), 1)
        self.assertEqual(prompt, [1, 2, 3, 4])
        for count in (-1, True, 2):
            with self.assertRaises(ValueError):
                model.generate(prompt, count)


if __name__ == '__main__':
    unittest.main()
