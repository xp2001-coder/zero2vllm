"""Offline checkpoint contracts and independent Transformers oracle; no downloads."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HAS_CHECKPOINT = all(importlib.util.find_spec(name) is not None for name in
                     ('torch', 'safetensors', 'transformers', 'tokenizers'))
if HAS_CHECKPOINT:
    import torch
    from safetensors.torch import load_file, save_file
    from transformers import LlamaConfig as HFConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from zero2vllm.modeling.llama import LlamaConfig, LlamaDecoder, split_half_rope, greedy_generate
    from zero2vllm.modeling.torch_decoder import apply_rope
    from zero2vllm.modeling.checkpoint import checkpoint_files, load_checkpoint, read_json, sha256
    from scripts.experiments.lesson0202 import compare_models, prepare_snapshot


@unittest.skipUnless(HAS_CHECKPOINT, 'optional checkpoint dependencies are not installed')
class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def fixture(self, tied=True, kv=2, sharded=False):
        config = HFConfig(vocab_size=16, hidden_size=16, intermediate_size=24,
                          num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=kv,
                          max_position_embeddings=32, rms_norm_eps=1e-5, rope_theta=100000.,
                          tie_word_embeddings=tied, attention_dropout=0.)
        config._attn_implementation = 'eager'
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(19)
            reference = LlamaForCausalLM(config).eval()
        reference.save_pretrained(self.directory, safe_serialization=True,
                                  max_shard_size='1KB' if sharded else '1GB')
        return reference

    def change_weights(self, fn):
        path = self.directory/'model.safetensors'
        # Detach from safetensors' file mapping before overwriting on Windows.
        weights = {name: value.clone() for name, value in load_file(path).items()}
        fn(weights)
        save_file(weights, path, metadata={'format': 'pt'})

    def test_gqa_mha_and_tied_untied_match_all_logits(self):
        for tied in (True, False):
            for kv in (1, 2, 4):
                with self.subTest(tied=tied, kv=kv):
                    reference = self.fixture(tied=tied, kv=kv)
                    model, audit = load_checkpoint(self.directory)
                    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
                    with torch.inference_mode():
                        expected = reference(tokens, use_cache=False).logits
                        actual = model(tokens)
                    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
                    self.assertEqual(model.head.weight is model.embedding.weight, tied)
                    self.assertEqual(audit['parameter_count'], sum(p.numel() for p in reference.parameters()))

    def test_indexed_shards_match_reference(self):
        reference = self.fixture(sharded=True)
        model, audit = load_checkpoint(self.directory)
        self.assertGreater(len(audit['files']), 1)
        with torch.inference_mode():
            tokens = torch.tensor([[1, 5, 7]])
            torch.testing.assert_close(model(tokens), reference(tokens, use_cache=False).logits, atol=1e-6, rtol=1e-5)

    def test_end_to_end_tokenizer_stages_and_greedy_oracle(self):
        reference = self.fixture()
        model, _ = load_checkpoint(self.directory)
        backend = Tokenizer(WordLevel({'<eos>': 0, '<unk>': 1, 'hello': 2, 'world': 3}, unk_token='<unk>'))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token='<eos>', bos_token='<eos>', unk_token='<unk>')
        report = compare_models(model, reference, tokenizer, ['hello world', 'world'], 3, 1e-6, 1e-5)
        self.assertTrue(all(case['passed'] for case in report), report)
        self.assertEqual(report[0]['input_ids'], [2, 3])
        self.assertEqual([c['stage'] for c in report[0]['comparisons']],
                         ['embedding', 'layers.0', 'layers.1', 'final_norm', 'logits'])

    def test_missing_and_extra_keys_are_rejected(self):
        self.fixture()
        self.change_weights(lambda weights: weights.pop('model.layers.0.self_attn.q_proj.weight'))
        with self.assertRaisesRegex(ValueError, 'missing='):
            load_checkpoint(self.directory)
        self.fixture()
        self.change_weights(lambda weights: weights.update({'stray.weight': torch.ones(1)}))
        with self.assertRaisesRegex(ValueError, 'extra='):
            load_checkpoint(self.directory)

    def test_untied_head_cannot_be_omitted(self):
        self.fixture(tied=False)
        self.change_weights(lambda weights: weights.pop('lm_head.weight'))
        with self.assertRaisesRegex(ValueError, 'lm_head.weight'):
            load_checkpoint(self.directory)

    def test_explicit_tied_head_must_equal_embedding(self):
        self.fixture()
        self.change_weights(lambda w: w.update({'lm_head.weight': w['model.embed_tokens.weight'].clone()}))
        model, audit = load_checkpoint(self.directory)
        self.assertIs(model.head.weight, model.embedding.weight)
        self.assertFalse(audit['head_alias_reconstructed'])
        self.change_weights(lambda w: w['lm_head.weight'].add_(1))
        with self.assertRaisesRegex(ValueError, 'Tied'):
            load_checkpoint(self.directory)

    def test_shape_and_quantized_dtype_rejected(self):
        key = 'model.layers.0.self_attn.k_proj.weight'
        self.fixture()
        self.change_weights(lambda w: w.update({key: w[key].T.contiguous()}))
        with self.assertRaisesRegex(ValueError, 'Shape mismatch'):
            load_checkpoint(self.directory)
        self.fixture()
        self.change_weights(lambda w: w.update({key: w[key].to(torch.int8)}))
        with self.assertRaisesRegex(ValueError, 'Unsupported dtype'):
            load_checkpoint(self.directory)

    def test_nonfinite_weights_rejected(self):
        for bad in (float('nan'), float('inf')):
            self.fixture()
            self.change_weights(lambda w: w['model.norm.weight'].fill_(bad))
            with self.assertRaisesRegex(ValueError, 'Non-finite'):
                load_checkpoint(self.directory)

    def test_stored_bfloat16_upcasts_without_changing_values(self):
        reference = self.fixture()
        self.change_weights(lambda w: w.update({k: v.bfloat16() for k, v in w.items()}))
        model, report = load_checkpoint(self.directory)
        self.assertEqual(report['mapping'][0]['stored_dtype'], 'torch.bfloat16')
        torch.testing.assert_close(model.embedding.weight, reference.model.embed_tokens.weight.bfloat16().float(), atol=0, rtol=0)

    def test_transposing_square_q_is_caught_by_numeric_oracle(self):
        reference = self.fixture()
        key = 'model.layers.0.self_attn.q_proj.weight'
        self.change_weights(lambda w: w.update({key: w[key].T.contiguous()}))
        model, _ = load_checkpoint(self.directory)  # shape checks legitimately cannot detect this
        tokens = torch.tensor([[1, 2, 3, 4]])
        with torch.inference_mode():
            self.assertFalse(torch.allclose(model(tokens), reference(tokens, use_cache=False).logits, atol=1e-7, rtol=1e-6))

    def test_rope_pairing_and_permutation(self):
        x = torch.tensor([[[[1., 2., 3., 4.], [1., 2., 3., 4.]]]])
        actual = split_half_rope(x, 10000.)
        self.assertFalse(torch.allclose(actual, apply_rope(x, 10000.)))
        # Conjugating adjacent-pair RoPE by this permutation gives split-half RoPE.
        permutation = [0, 2, 1, 3]
        converted = apply_rope(x[..., permutation], 10000.)[..., permutation]
        torch.testing.assert_close(actual, converted)
        torch.testing.assert_close(actual[..., 0, :], x[..., 0, :])
        torch.testing.assert_close(actual.square().sum(-1), x.square().sum(-1))

    def test_causal_prefix_invariance(self):
        self.fixture()
        model, _ = load_checkpoint(self.directory)
        with torch.inference_mode():
            a = model(torch.tensor([[1, 2, 3]]))
            b = model(torch.tensor([[1, 2, 9, 8]]))
        torch.testing.assert_close(a[:, :2], b[:, :2], atol=1e-6, rtol=1e-5)

    def test_unloaded_model_and_input_bounds(self):
        self.fixture()
        config = LlamaConfig.from_dict(read_json(self.directory/'config.json'))
        model = LlamaDecoder(config)
        with self.assertRaisesRegex(RuntimeError, 'Load'):
            model(torch.tensor([[1]]))
        model, _ = load_checkpoint(self.directory)
        for tokens in (torch.tensor([[16]]), torch.tensor([[-1]]), torch.tensor([[1.]]),
                       torch.empty((1, 0), dtype=torch.long), torch.ones((1, 33), dtype=torch.long)):
            with self.assertRaises(ValueError):
                model(tokens)

    def test_eos_zero_output_and_context_boundary(self):
        self.fixture()
        model, _ = load_checkpoint(self.directory)
        tokens = torch.tensor([[1]*32])
        self.assertEqual(greedy_generate(model, tokens, 0), [])
        self.assertEqual(len(greedy_generate(model, tokens, 1)), 1)
        with self.assertRaises(ValueError):
            greedy_generate(model, tokens, 2)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        self.assertEqual(greedy_generate(model, torch.tensor([[1]]), 4, eos_token_id=0), [0])

    def test_unsupported_config_semantics_rejected(self):
        self.fixture()
        original = read_json(self.directory/'config.json')
        for field, bad in [('model_type', 'qwen2'), ('rope_scaling', {'rope_type': 'linear', 'factor': 2}),
                           ('attention_bias', True), ('mlp_bias', True), ('rope_interleaved', True),
                           ('hidden_act', 'gelu'), ('num_key_value_heads', 3), ('hidden_size', 15),
                           ('rms_norm_eps', float('nan')), ('head_dim', 8),
                           ('quantization_config', {}), ('vocab_size', True)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                LlamaConfig.from_dict({**original, field: bad})

    def test_duplicate_json_keys_rejected(self):
        path = self.directory/'config.json'
        path.write_text('{"hidden_size": 16, "hidden_size": 32}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            read_json(path)

    def test_shard_index_missing_file_wrong_keys_and_traversal(self):
        self.fixture(sharded=True)
        path = self.directory/'model.safetensors.index.json'
        original = read_json(path)
        key = next(iter(original['weight_map']))
        for replacement in ('missing.safetensors', '../escape.safetensors', 'C:\\escape.safetensors'):
            changed = {**original, 'weight_map': {**original['weight_map'], key: replacement}}
            path.write_text(json.dumps(changed), encoding='utf-8')
            with self.assertRaises(ValueError):
                checkpoint_files(self.directory)
        changed = {**original, 'weight_map': {k: v for k, v in original['weight_map'].items() if k != key}}
        # Keep that shard referenced to detect the unindexed tensor.
        shard = original['weight_map'][key]
        changed['weight_map']['unexpected.alias'] = shard
        path.write_text(json.dumps(changed), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            checkpoint_files(self.directory)

    def test_snapshot_receipt_detects_config_drift_offline(self):
        self.fixture()
        names = ['config.json', 'model.safetensors']
        recipe = {'repo_id': 'test/offline', 'revision': 'a'*40, 'files': names}
        receipt = {**recipe, 'files': [{'file': name, 'sha256': sha256(self.directory/name)} for name in names]}
        (self.directory/'zero2vllm-snapshot.json').write_text(json.dumps(receipt), encoding='utf-8')
        self.assertEqual(prepare_snapshot(self.directory, recipe), receipt)
        with (self.directory/'config.json').open('a', encoding='utf-8') as target:
            target.write(' ')
        with self.assertRaisesRegex(ValueError, 'changed'):
            prepare_snapshot(self.directory, recipe)


if __name__ == '__main__':
    unittest.main()
