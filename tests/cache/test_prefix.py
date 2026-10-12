import dataclasses
import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from zero2vllm.cache.prefix import PrefixKVStore


@unittest.skipUnless(HAS_TORCH, 'optional PyTorch dependency is not installed')
class PrefixStoreTests(unittest.TestCase):
    def store(self, blocks=8, size=2, **kwargs):
        return PrefixKVStore(blocks, size, head_dim=2, **kwargs)

    def append(self, store, handle, tokens):
        values = torch.tensor([[[float(t), float(t)+0.5]] for t in tokens], dtype=store.dtype)
        return store.append(handle, tokens, values, values)

    def seed(self, store, tokens, namespace='default'):
        handle = store.acquire(tokens, len(tokens), namespace=namespace)
        self.append(store, handle, tokens)
        store.release(handle)

    def test_active_requests_share_identical_physical_blocks(self):
        store = self.store()
        a = store.acquire([1, 2, 3], 5)
        self.append(store, a, [1, 2, 3])
        b = store.acquire([1, 2, 4], 5)
        sa, sb = store.snapshot(a), store.snapshot(b)
        self.assertEqual(sb['hit_positions'], 2)
        self.assertEqual(sa['block_table'][0], sb['block_table'][0])
        self.assertEqual(sb['block_refs'], [2, 1, 1])
        self.append(store, b, [4])
        store.release(a)
        self.assertEqual(store.snapshot(b)['block_refs'], [1, 1, 1])
        self.assertEqual(store.gather(b)[1][:, 0, 0].tolist(), [1, 2, 4])
        store.release(b)
        self.assertEqual(store.pool_snapshot()['active_block_references'], 0)

    def test_release_preserves_full_blocks_and_discards_partial_tail(self):
        store = self.store()
        self.seed(store, [1, 2, 3])
        pool = store.pool_snapshot()
        self.assertEqual((pool['idle_cached_blocks'], pool['empty_blocks']), (1, 7))
        b = store.acquire([1, 2, 3], 3)
        self.assertEqual(store.snapshot(b)['tokens'], [1, 2])
        store.release(b)

    def test_leaves_input_for_final_logits_at_every_boundary(self):
        for size in (1, 2, 3, 4):
            for length in (1, 2, 3, 4, 5, 8):
                with self.subTest(size=size, length=length):
                    store = self.store(20, size)
                    tokens = list(range(length))
                    self.seed(store, tokens)
                    h = store.acquire(tokens, length)
                    self.assertEqual(store.snapshot(h)['hit_positions'], ((length-1)//size)*size)
                    store.release(h)

    def test_same_suffix_with_different_parent_cannot_hit(self):
        store = self.store()
        self.seed(store, [1, 2, 5, 6, 9])
        h = store.acquire([3, 4, 5, 6, 9], 5)
        self.assertEqual(store.snapshot(h)['hit_positions'], 0)
        store.release(h)

    def test_namespace_and_forced_digest_collisions_are_checked_exactly(self):
        store = self.store(12, digest_fn=lambda _: 'forced-collision')
        self.seed(store, [1, 2, 5], 'tenant-a')
        self.seed(store, [3, 4, 5], 'tenant-a')
        for tokens, namespace, hits in [([1, 2, 8], 'tenant-a', 2),
                                       ([3, 4, 8], 'tenant-a', 2),
                                       ([1, 2, 8], 'tenant-b', 0),
                                       ([5, 6, 8], 'tenant-a', 0)]:
            h = store.acquire(tokens, 3, namespace=namespace)
            self.assertEqual(store.snapshot(h)['hit_positions'], hits)
            store.release(h)

    def test_miss_stops_the_contiguous_prefix_search(self):
        store = self.store()
        self.seed(store, [1, 2, 3, 4, 5])
        h = store.acquire([1, 2, 9, 4, 5], 5)
        self.assertEqual(store.snapshot(h)['hit_positions'], 2)
        store.release(h)

    def test_oom_does_not_pin_evict_or_change_recency(self):
        store = self.store(3)
        self.seed(store, [1, 2, 3])
        live = store.acquire([8, 9], 2)
        before = store.pool_snapshot()
        events = list(store.events)
        with self.assertRaises(MemoryError):
            store.acquire([1, 2, 3], 6)
        self.assertEqual(store.pool_snapshot(), before)
        self.assertEqual(store.events, events)
        store.release(live)

    def test_zero_reference_hit_is_protected_during_new_block_selection(self):
        store = self.store(2)
        self.seed(store, [1, 2, 3, 4])
        h = store.acquire([1, 2, 9], 3)
        self.assertEqual(store.snapshot(h)['hit_positions'], 2)
        self.append(store, h, [9])
        self.assertEqual(store.gather(h)[0][:, 0, 0].tolist(), [1, 2, 9])
        store.release(h)

    def test_lru_evicts_only_idle_blocks(self):
        store = self.store(3)
        self.seed(store, [1, 2])
        self.seed(store, [3, 4])
        self.seed(store, [5, 6])
        live = store.acquire([1, 2, 7], 3)
        # Pin the oldest hit (block 0), evict next-oldest block 1 for private tail.
        self.assertEqual(store.snapshot(live)['block_table'], [0, 1])
        self.assertEqual([e['block'] for e in store.events if e['kind'] == 'evicted'], [1])
        other = store.acquire([8], 1)
        self.assertEqual(store.snapshot(other)['block_table'], [2])
        store.release(other)
        store.release(live)

    def test_evicted_hash_no_longer_matches_overwritten_bytes(self):
        store = self.store(2)
        self.seed(store, [1, 2, 3, 4])
        self.seed(store, [8, 9, 10, 11])
        h = store.acquire([1, 2, 3], 3)
        self.assertEqual(store.snapshot(h)['hit_positions'], 0)
        self.append(store, h, [1, 2, 3])
        self.assertEqual(store.gather(h)[0][:, 0, 0].tolist(), [1, 2, 3])
        store.release(h)

    def test_duplicate_cold_requests_do_not_remap_live_tables(self):
        store = self.store(4)
        a, b = store.acquire([1, 2, 3], 3), store.acquire([1, 2, 3], 3)
        table = store.snapshot(b)['block_table']
        self.append(store, a, [1, 2, 3])
        self.append(store, b, [1, 2, 3])
        self.assertEqual(store.snapshot(b)['block_table'], table)
        store.release(a)
        store.release(b)
        self.assertEqual(store.pool_snapshot()['idle_cached_blocks'], 1)

    def test_partial_tail_is_private_and_old_bytes_are_invisible(self):
        store = self.store(2)
        self.seed(store, [1, 2, 9])
        h = store.acquire([1, 2, 3], 3)
        with self.assertRaises(ValueError):
            store.gather(h, 3)
        self.append(store, h, [3])
        self.assertEqual(store.gather(h)[1][:, 0, 0].tolist(), [1, 2, 3])
        store.release(h)

    def test_wrong_tokens_or_tensor_leave_snapshot_unchanged(self):
        store = self.store()
        h = store.acquire([1, 2, 3], 3)
        before = store.snapshot(h)
        for tokens, value in [([9], torch.ones(1, 1, 2, dtype=torch.float64)),
                              ([1], torch.ones(1, 1, 2, dtype=torch.float32)),
                              ([1], torch.full((1, 1, 2), float('nan'), dtype=torch.float64))]:
            with self.assertRaises(ValueError):
                store.append(h, tokens, value, value)
            self.assertEqual(store.snapshot(h), before)
        store.release(h)

    def test_handles_and_gather_copies_preserve_ownership(self):
        store, other = self.store(), self.store()
        h = store.acquire([1, 2], 2)
        foreign = other.acquire([1], 1)
        self.append(store, h, [1, 2])
        store.gather(h)[0].zero_()
        self.assertEqual(store.gather(h)[0][:, 0, 0].tolist(), [1, 2])
        for bad in (foreign, dataclasses.replace(h)):
            with self.assertRaises(ValueError):
                store.gather(bad)
        store.release(h)
        with self.assertRaises(ValueError):
            store.release(h)
        other.release(foreign)

    def test_reset_refuses_live_handles_and_clears_idle_index(self):
        store = self.store()
        h = store.acquire([1, 2, 3], 3)
        self.append(store, h, [1, 2, 3])
        before = store.pool_snapshot()
        with self.assertRaises(RuntimeError):
            store.reset()
        self.assertEqual(store.pool_snapshot(), before)
        store.release(h)
        store.reset()
        self.assertEqual(store.pool_snapshot()['empty_blocks'], store.num_blocks)
        h = store.acquire([1, 2, 3], 3)
        self.assertEqual(store.snapshot(h)['hit_positions'], 0)
        store.release(h)

    def test_disable_lookup_still_computes_and_publishes(self):
        store = self.store()
        self.seed(store, [1, 2, 3])
        h = store.acquire([1, 2, 3], 3, reuse=False)
        self.assertEqual(store.snapshot(h)['hit_positions'], 0)
        self.append(store, h, [1, 2, 3])
        store.release(h)

    def test_attention_uses_explicit_causal_prefix(self):
        store = self.store()
        h = store.acquire([1, 2, 3], 3)
        self.append(store, h, [1, 2, 3])
        q = torch.zeros(1, 2, dtype=torch.float64)
        self.assertEqual(store.attention(h, q, upto=1).tolist(), [[1, 1.5]])
        self.assertEqual(store.attention(h, q).tolist(), [[2, 2.5]])
        store.release(h)

    def test_invalid_configuration_and_reservations(self):
        for args in [(0, 2), (2, 0), (True, 2)]:
            with self.assertRaises(ValueError):
                self.store(*args)
        store = self.store()
        for prompt, reserve in [([], 1), ([True], 1), ([-1], 1), ([1, 2], 1)]:
            with self.assertRaises(ValueError):
                store.acquire(prompt, reserve)
        self.assertEqual(store.pool_snapshot()['active_requests'], 0)
