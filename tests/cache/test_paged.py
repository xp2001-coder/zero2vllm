import dataclasses
import importlib.util
import math
import unittest

HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from zero2vllm.cache.paged import PagedContext, PagedKVStore


@unittest.skipUnless(HAS_TORCH, 'optional PyTorch dependency is not installed')
class PagedStoreTests(unittest.TestCase):
    def store(self, blocks=4, size=2, heads=1, dim=2):
        return PagedKVStore(blocks, size, kv_heads=heads, head_dim=dim)

    def tensor(self, rows):
        return torch.tensor(rows, dtype=torch.float64).reshape(-1, 1, 2)

    def test_noncontiguous_pages_preserve_logical_order(self):
        store = self.store(3)
        a, b = store.create(), store.create()
        store.reserve(a, 2)
        store.reserve(b, 2)
        k = self.tensor([[1, 0], [0, 1], [1, 1]])
        v = self.tensor([[10, 0], [20, 10], [30, 20]])
        store.append(a, k[:2], v[:2])
        store.reserve(a, 3)
        self.assertEqual(store.append(a, k[2:], v[2:]), (4,))
        self.assertEqual(store.table(a), (0, 2))
        self.assertEqual(store.slot_mapping(a, range(3)), (0, 1, 4))
        actual_k, actual_v = store.gather(a)
        torch.testing.assert_close(actual_k, k)
        torch.testing.assert_close(actual_v, v)
        q = torch.tensor([[1., 0.]], dtype=torch.float64)
        weights = [math.exp(1/math.sqrt(2)), 1, math.exp(1/math.sqrt(2))]
        expected = [sum(w*float(row[0, d]) for w, row in zip(weights, v))/sum(weights) for d in range(2)]
        torch.testing.assert_close(store.attention(a, q), torch.tensor([expected], dtype=torch.float64))

    def test_reservation_oom_leaves_table_length_and_data_unchanged(self):
        store = self.store(2)
        a = store.create()
        store.reserve(a, 2)
        store.append(a, self.tensor([[1, 2]]), self.tensor([[3, 4]]))
        before = store.snapshot(a)
        with self.assertRaises(MemoryError):
            store.reserve(a, 5)
        self.assertEqual(store.snapshot(a), before)
        self.assertEqual(store.free_blocks, 1)
        with self.assertRaises(IndexError):
            store.locate(a, 2)

    def test_reservation_and_written_prefix_are_distinct(self):
        store = self.store()
        a = store.create()
        store.reserve(a, 5)
        store.append(a, self.tensor([[1, 2]]), self.tensor([[3, 4]]))
        self.assertEqual(store.locate(a, 4), (2, 0))
        with self.assertRaisesRegex(ValueError, 'written prefix'):
            store.gather(a, 5)
        s = store.snapshot(a)
        self.assertEqual((s['allocated_slots'], s['rounding_tail_slots'], s['reserved_unwritten_positions']), (6, 1, 4))

    def test_stale_tail_survives_release_but_is_never_visible(self):
        store = self.store(1)
        old = store.create()
        store.reserve(old, 2)
        store.append(old, self.tensor([[1, 2], [8, 9]]), self.tensor([[3, 4], [99, 99]]))
        store.release(old)
        fresh = store.create()
        store.reserve(fresh, 1)
        store.append(fresh, self.tensor([[0, 0]]), self.tensor([[3, 7]]))
        self.assertEqual(store.physical_snapshot()[1][0, 1].tolist(), [[99., 99.]])
        self.assertEqual(store.gather(fresh)[1].tolist(), [[[3., 7.]]])
        self.assertEqual(store.attention(fresh, torch.zeros(1, 2, dtype=torch.float64)).tolist(), [[3., 7.]])
        for operation in (lambda: store.reserve(old, 1), lambda: store.gather(old), lambda: store.release(old)):
            with self.assertRaises(ValueError):
                operation()

    def test_foreign_or_copied_handle_rejected_even_with_same_key(self):
        a, b = self.store(), self.store()
        handle_a, handle_b = a.create(), b.create()
        self.assertEqual(handle_a.key, handle_b.key)
        for handle in (handle_b, dataclasses.replace(handle_a)):
            with self.assertRaises(ValueError):
                a.reserve(handle, 1)

    def test_gather_returns_copies(self):
        store = self.store()
        a = store.create()
        store.reserve(a, 1)
        store.append(a, self.tensor([[1, 2]]), self.tensor([[3, 4]]))
        k, v = store.gather(a)
        k.zero_()
        v.zero_()
        self.assertEqual(store.gather(a)[0].tolist(), [[[1., 2.]]])
        self.assertEqual(store.gather(a)[1].tolist(), [[[3., 4.]]])

    def test_invalid_append_is_side_effect_free(self):
        store = self.store()
        a = store.create()
        store.reserve(a, 2)
        valid = self.tensor([[1, 2]])
        invalid = [valid.float(), torch.ones(1, 2, dtype=torch.float64), valid[:0],
                   valid.clone().fill_(float('nan')), valid.clone().requires_grad_(True)]
        for bad in invalid:
            with self.assertRaises(ValueError):
                store.append(a, valid, bad)
            self.assertEqual(store.length(a), 0)
        with self.assertRaises(ValueError):
            store.append(a, valid, valid.repeat(2, 1, 1))
        self.assertEqual(store.length(a), 0)

    def test_append_cannot_write_rounding_tail(self):
        store = self.store()
        a = store.create()
        store.reserve(a, 1)  # Two physical slots, only one logical position allowed.
        with self.assertRaises(ValueError):
            store.append(a, self.tensor([[1, 2], [3, 4]]), self.tensor([[5, 6], [7, 8]]))
        self.assertEqual(store.length(a), 0)

    def test_causal_prefix_excludes_later_values(self):
        store = self.store()
        a = store.create()
        store.reserve(a, 2)
        store.append(a, self.tensor([[0, 0], [0, 0]]), self.tensor([[2, 4], [200, 400]]))
        q = torch.zeros(1, 2, dtype=torch.float64)
        self.assertEqual(store.attention(a, q, upto=1).tolist(), [[2., 4.]])
        self.assertEqual(store.attention(a, q, upto=2).tolist(), [[101., 202.]])

    def test_gqa_head_mapping_matches_independent_head_loops(self):
        store = self.store(heads=2)
        a = store.create()
        store.reserve(a, 3)
        k = torch.arange(12, dtype=torch.float64).reshape(3, 2, 2)/10
        v = k+1
        q = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [2., -1.]], dtype=torch.float64)
        store.append(a, k, v)
        rows = []
        for h in range(4):
            probabilities = (k[:, h//2] @ q[h] / math.sqrt(2)).softmax(0)
            rows.append(probabilities @ v[:, h//2])
        torch.testing.assert_close(store.attention(a, q), torch.stack(rows))
        with self.assertRaises(ValueError):
            store.attention(a, q[:3])

    def test_zero_reservation_and_empty_gather(self):
        store = self.store()
        a = store.create()
        self.assertEqual(store.reserve(a, 0), ())
        self.assertEqual(tuple(store.gather(a)[0].shape), (0, 1, 2))
        with self.assertRaises(ValueError):
            store.attention(a, torch.zeros(1, 2, dtype=torch.float64))
        store.release(a)
        self.assertEqual(store.free_blocks, 4)

    def test_invalid_constructor_and_ranges(self):
        for options in ({'num_blocks': 0}, {'block_size': True}, {'kv_heads': 0}, {'head_dim': 0}, {'dtype': torch.float16}):
            with self.assertRaises(ValueError):
                PagedKVStore(**{'num_blocks': 2, 'block_size': 2, **options})
        store = self.store()
        a = store.create()
        store.reserve(a, 2)
        for position in (-1, 2, True):
            with self.assertRaises(IndexError):
                store.locate(a, position)
        for count in (-1, True, 1):
            with self.assertRaises(ValueError):
                store.gather(a, count)

    def test_context_close_is_idempotent_and_releases_once(self):
        store = self.store()
        context = PagedContext(store)
        context.reserve(3)
        self.assertEqual(store.free_blocks, 2)
        context.close()
        context.close()
        self.assertEqual((store.free_blocks, context.cached_positions), (4, 0))
        with self.assertRaises(ValueError):
            context.reserve(1)

    def test_tensor_byte_count_matches_shape_and_dtype(self):
        for dtype, size in ((torch.float32, 4), (torch.float64, 8)):
            store = PagedKVStore(3, 4, kv_heads=2, head_dim=5, dtype=dtype)
            self.assertEqual(store.storage_bytes, 2*3*4*2*5*size)

    def test_other_request_values_and_fragmentation_do_not_change_attention(self):
        store = self.store(3)
        a, b = store.create(), store.create()
        store.reserve(a, 2)
        store.reserve(b, 2)
        k, v = self.tensor([[1, 2], [3, 4]]), self.tensor([[5, 6], [7, 8]])
        store.append(a, k, v)
        q = torch.ones(1, 2, dtype=torch.float64)
        before = store.attention(a, q)
        store.append(b, k*100, v*100)
        torch.testing.assert_close(store.attention(a, q), before)
        store.release(b)
        torch.testing.assert_close(store.attention(a, q), before)


if __name__ == '__main__':
    unittest.main()
