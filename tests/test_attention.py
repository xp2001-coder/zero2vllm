import unittest

from zero2vllm.reference.attention import TinyAttentionLM, WorkCounter, generate


class AttentionTests(unittest.TestCase):
    def assert_logits_close(self, left, right):
        self.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assertAlmostEqual(x, y, delta=1e-12)

    def test_chunked_prefill_matches_full_at_every_position(self):
        for seed in (0, 7, 31):
            model = TinyAttentionLM(seed=seed)
            tokens = [2, 4, 6, 3, 8, 1, 0]
            expected = model.forward_full(tokens, WorkCounter())
            for chunk_sizes in ((7,), (1, 1, 1, 1, 1, 1, 1), (2, 3, 2)):
                with self.subTest(seed=seed, chunks=chunk_sizes):
                    cache = model.new_cache()
                    actual = []
                    offset = 0
                    for size in chunk_sizes:
                        actual.extend(model.forward_cached(tokens[offset:offset + size], cache, WorkCounter()))
                        offset += size
                    self.assert_logits_close(expected, actual)

    def test_future_tokens_do_not_change_earlier_logits(self):
        model = TinyAttentionLM()
        prefix = model.forward_full([1, 2, 3], WorkCounter())
        longer = model.forward_full([1, 2, 3, 9, 10], WorkCounter())
        self.assert_logits_close(prefix, longer[:3])

    def test_cached_generation_matches_full_across_seeds(self):
        for seed in range(10):
            for prompt in ([1], [1, 4, 7], [15, 0, 15, 0]):
                with self.subTest(seed=seed, prompt=prompt):
                    full = generate(TinyAttentionLM(seed=seed), prompt, 12, use_cache=False)
                    cached = generate(TinyAttentionLM(seed=seed), prompt, 12, use_cache=True)
                    self.assertEqual(full.token_ids, cached.token_ids)
                    self.assert_logits_close(full.logits, cached.logits)

    def test_projection_work_and_last_token_convention(self):
        full = generate(TinyAttentionLM(), [1, 2, 3, 4], 8, use_cache=False)
        cached = generate(TinyAttentionLM(), [1, 2, 3, 4], 8, use_cache=True)
        self.assertEqual(full.work.projected_token_positions, sum(range(4, 12)))
        self.assertEqual(cached.work.projected_token_positions, 11)
        self.assertLess(cached.work.attention_score_pairs, full.work.attention_score_pairs)

    def test_zero_generation_does_no_forward_work(self):
        result = generate(TinyAttentionLM(), [1], 0, use_cache=True)
        self.assertEqual(result.token_ids, [])
        self.assertEqual(result.work.projected_token_positions, 0)

    def test_context_limit_allows_final_unconsumed_token(self):
        model = TinyAttentionLM(max_context=3)
        self.assertEqual(len(generate(model, [1, 2], 2, use_cache=True).token_ids), 2)
        with self.assertRaises(ValueError):
            generate(model, [1, 2], 3, use_cache=True)

    def test_invalid_chunk_does_not_mutate_cache(self):
        model = TinyAttentionLM(max_context=4)
        cache = model.new_cache()
        count = WorkCounter()
        model.forward_cached([1, 2], cache, count)
        before = [list(row) for row in cache.keys]
        for tokens in ([3, 99], [3, 4, 5]):
            with self.assertRaises(ValueError):
                model.forward_cached(tokens, cache, count)
            self.assertEqual(cache.keys, before)
            self.assertEqual(count.projected_token_positions, 2)

    def test_cache_rejects_other_model_and_mismatched_lengths(self):
        model = TinyAttentionLM()
        with self.assertRaises(ValueError):
            model.forward_cached([1], TinyAttentionLM().new_cache(), WorkCounter())
        broken = model.new_cache()
        broken.keys.append([0.0] * model.width)
        with self.assertRaises(ValueError):
            model.forward_cached([1], broken, WorkCounter())

    def test_invalid_inputs_and_generation_counts(self):
        model = TinyAttentionLM()
        for tokens in ([], [-1], [16], [True], [1.0]):
            with self.assertRaises(ValueError):
                model.forward_full(tokens, WorkCounter())
        for count in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                generate(model, [1], count, use_cache=True)

    def test_generate_preserves_prompt(self):
        prompt = [1, 2, 3]
        generate(TinyAttentionLM(), prompt, 4, use_cache=True)
        self.assertEqual(prompt, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
