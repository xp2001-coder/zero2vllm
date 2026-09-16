import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from zero2vllm.modeling.torch_decoder import TorchDecoder, apply_rope, decoder_experiment
    from zero2vllm.reference.decoder import DecoderConfig, ScalarDecoder


@unittest.skipUnless(HAS_TORCH, 'optional PyTorch dependency is not installed')
class TorchDecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Tiny tensors run faster with one thread; restore caller settings afterwards.
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_all_intermediate_stages_float64_and_float32(self):
        for dtype in ('float64', 'float32'):
            for seed, prompt, layers in ((7, [1, 2, 3, 4], 2), (8, [15], 1), (11, [2, 2, 0], 3)):
                with self.subTest(dtype=dtype, seed=seed):
                    report = decoder_experiment(prompt, seed=seed, dtype=dtype, layers=layers)
                    self.assertEqual(len(report['comparisons']), 3 + 14*layers)
                    self.assertTrue(report['verification']['passed'], report['comparisons'])

    def test_batch_matches_independent_requests(self):
        reference = ScalarDecoder()
        model = TorchDecoder(reference)
        tokens = [[1, 2, 3], [3, 2, 1]]
        with torch.inference_mode():
            batch = model(torch.tensor(tokens))
        for i, row in enumerate(tokens):
            expected, _ = reference.forward(row)
            torch.testing.assert_close(batch[i], torch.tensor(expected, dtype=torch.float64), atol=1e-12, rtol=1e-10)

    def test_future_mask_and_prefix_extension(self):
        model = TorchDecoder(ScalarDecoder())
        with torch.inference_mode():
            a = model(torch.tensor([[1, 2, 3]]))
            b = model(torch.tensor([[1, 2, 15, 9]]))
        torch.testing.assert_close(a[:, :2], b[:, :2], atol=1e-12, rtol=1e-10)

    def test_linear_transpose_embedding_and_parameter_registration(self):
        reference = ScalarDecoder()
        model = TorchDecoder(reference)
        self.assertEqual(model.head.weight.shape, (16, 8))
        self.assertEqual(model.head.weight[3, 2].item(), reference.weights['head'][2][3])
        self.assertEqual(model.embedding.weight[3, 2].item(), reference.weights['embedding'][3][2])
        self.assertEqual(sum(p.numel() for p in model.parameters()), 1576)
        self.assertEqual(len(model.state_dict()), 21)

    def test_rotations_use_position_and_do_not_mix_heads(self):
        x = torch.tensor([[[[1., 2., 3., 4.], [1., 2., 3., 4.]]]], dtype=torch.float64)
        y = apply_rope(x, 10000.)
        torch.testing.assert_close(x[:, :, 0], y[:, :, 0])
        torch.testing.assert_close(x.square().sum(-1), y.square().sum(-1))
        self.assertFalse(torch.equal(x[:, :, 1], y[:, :, 1]))

    def test_invalid_input_and_context_boundary(self):
        model = TorchDecoder(ScalarDecoder(DecoderConfig(max_context=4)))
        for tokens in (torch.tensor([1]), torch.tensor([[1.]]), torch.tensor([[16]]),
                       torch.empty((0, 2), dtype=torch.long), torch.empty((1, 0), dtype=torch.long),
                       torch.zeros((1, 5), dtype=torch.long)):
            with self.subTest(shape=tokens.shape), self.assertRaises(ValueError):
                model(tokens)
        self.assertEqual(model.generate([1]*4, 0), [])
        self.assertEqual(len(model.generate([1]*4, 1)), 1)
        with self.assertRaises(ValueError):
            model.generate([1]*4, 2)

    def test_does_not_change_rng_and_generation_disables_grad(self):
        before = torch.random.get_rng_state().clone()
        model = TorchDecoder(ScalarDecoder())
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        observed = []
        hook = model.register_forward_hook(lambda *args: observed.append(torch.is_grad_enabled()))
        try:
            model.generate([1, 2], 2)
        finally:
            hook.remove()
        self.assertEqual(observed, [False, False])

    def test_mutated_projection_fails_oracle_comparison(self):
        reference = ScalarDecoder()
        model = TorchDecoder(reference)
        with torch.no_grad():
            model.layers[0].q.weight.zero_()
        expected, _ = reference.forward([1, 2, 3])
        with torch.inference_mode():
            actual = model(torch.tensor([[1, 2, 3]]))[0]
        self.assertFalse(torch.allclose(actual, torch.tensor(expected, dtype=torch.float64), atol=1e-12, rtol=1e-10))


if __name__ == '__main__':
    unittest.main()
