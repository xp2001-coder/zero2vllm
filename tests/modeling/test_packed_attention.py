from dataclasses import replace
import importlib.util
import math
import unittest

from scripts.experiments import lesson0303

HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from torch.nn import functional as F
    from zero2vllm.modeling.packed_attention import AttentionFixture, fuse_qkv, rope


@unittest.skipUnless(HAS_TORCH, 'optional CPU PyTorch is not installed')
class PackedAttentionTests(unittest.TestCase):
    def setUp(self):
        self.model = AttentionFixture()
        self.rows = lesson0303.requests()

    def test_paged_logits_match_independent_dense_requests(self):
        r = self.model.execute(self.rows, ['B','A'])
        self.assertLess(max(r['errors'].values()), 1e-11)

    def test_float32_contracts(self):
        self.assertTrue(all(c['passed'] for c in lesson0303.tensor_cases('float32')))

    def test_reorder_does_not_change_per_request_results(self):
        a = self.model.execute(self.rows, ['A','B'])
        b = self.model.execute(self.rows, ['B','A'])
        for rid in ('A','B'):
            torch.testing.assert_close(a['logits_by_id'][rid], b['logits_by_id'][rid], atol=1e-11, rtol=0)
        self.assertEqual(a['returned_ids'], b['returned_ids'])

    def test_future_scheduled_token_cannot_change_earlier_queries(self):
        old = self.model.execute(self.rows, ['B','A'])
        rows = (replace(self.rows[0], token_ids=(10,11,12,13,30,15)), self.rows[1])
        new = self.model.execute(rows, ['B','A'])
        torch.testing.assert_close(old['logits_by_id']['A'][:2], new['logits_by_id']['A'][:2], atol=1e-11, rtol=0)
        self.assertGreater(float((old['logits_by_id']['A'][-1]-new['logits_by_id']['A'][-1]).abs().max()), 1e-6)

    def test_unscheduled_future_token_never_enters_forward(self):
        rows = (replace(self.rows[0], token_ids=(10,11,12,13,14,31)), self.rows[1])
        old, new = self.model.execute(self.rows, ['A','B']), self.model.execute(rows, ['A','B'])
        torch.testing.assert_close(old['logits'], new['logits'], atol=0, rtol=0)

    def test_other_request_history_cannot_leak_into_A(self):
        rows = (self.rows[0], replace(self.rows[1], token_ids=(0,1,2,3)))
        old, new = self.model.execute(self.rows, ['A','B']), self.model.execute(rows, ['A','B'])
        torch.testing.assert_close(old['logits_by_id']['A'], new['logits_by_id']['A'], atol=0, rtol=0)

    def test_padding_projections_never_write_cache_or_produce_results(self):
        a = self.model.execute(self.rows, ['B','A'])
        b = self.model.execute(self.rows, ['B','A'], pad_to=8, padding_token=31)
        torch.testing.assert_close(a['logits'], b['logits'], atol=1e-11, rtol=0)
        torch.testing.assert_close(a['cache'], b['cache'], equal_nan=True)
        self.assertEqual(b['shapes']['q'], [8,4,2])
        self.assertEqual(b['shapes']['logits'], [4,32])

    def test_unused_physical_slots_stay_nan_but_outputs_are_finite(self):
        r = self.model.execute(self.rows, ['B','A'])
        self.assertEqual(int(r['written'].sum()), 9)
        self.assertTrue(torch.isnan(r['cache'].view(2,-1,2,2)[:,~r['written']]).all())
        self.assertTrue(torch.isfinite(r['logits']).all())

    def test_partial_prefill_candidate_is_discarded(self):
        r = self.model.execute(self.rows, ['B','A'])
        self.assertEqual(len(r['candidate_ids']), 2)
        self.assertEqual(r['returned_ids']['A'], [])
        self.assertEqual(r['returned_ids']['B'], [r['candidate_ids'][0]])

    def test_complete_prefill_retains_candidate(self):
        rows = (replace(self.rows[0], scheduled=4),)
        r = self.model.execute(rows, ['A'])
        self.assertEqual(len(r['returned_ids']['A']), 1)
        self.assertLess(r['errors']['A'], 1e-11)

    def test_fused_qkv_matches_three_separate_matrices(self):
        x = self.model.embedding[:5]
        q, k, v = F.linear(x, self.model.fused).split((8,4,4),-1)
        for name, value in zip(('q_proj','k_proj','v_proj'), (q,k,v)):
            torch.testing.assert_close(value, F.linear(x, self.model.weights[name]), atol=1e-11, rtol=0)

    def test_wrong_non_square_weight_orientation_rejected(self):
        weights = dict(self.model.weights)
        weights['k_proj'] = weights['k_proj'].T
        with self.assertRaises(ValueError): fuse_qkv(weights, hidden_size=8, query_heads=4, kv_heads=2, head_dim=2)

    def test_square_shape_alone_does_not_prove_orientation(self):
        x = torch.tensor([[1.,2.]],dtype=torch.float64)
        w = torch.tensor([[1.,3.],[2.,4.]],dtype=torch.float64)
        self.assertEqual(F.linear(x,w).tolist(), [[7.,10.]])
        self.assertNotEqual(F.linear(x,w.T).tolist(), [[7.,10.]])

    def test_missing_nan_or_mixed_dtype_weights_are_rejected(self):
        for mode in ('missing','nan','dtype'):
            weights={k:v.clone() for k,v in self.model.weights.items()}
            if mode=='missing': del weights['q_proj']
            elif mode=='nan': weights['v_proj'][0,0]=float('nan')
            else: weights['k_proj']=weights['k_proj'].float()
            with self.assertRaises(ValueError): fuse_qkv(weights,hidden_size=8,query_heads=4,kv_heads=2,head_dim=2)

    def test_fused_storage_does_not_alias_checkpoint_tensors(self):
        before = self.model.fused.clone()
        self.model.weights['q_proj'].zero_()
        torch.testing.assert_close(before, self.model.fused, atol=0, rtol=0)

    def test_rope_uses_absolute_request_position(self):
        x=torch.tensor([[[1.,0.]],[[1.,0.]]],dtype=torch.float64)
        expected=torch.tensor([[[1.,0.]],[[math.cos(3),math.sin(3)]]],dtype=torch.float64)
        torch.testing.assert_close(rope(x,(0,3)),expected,atol=1e-12,rtol=0)

    def test_gqa_maps_pairs_of_query_heads_to_one_kv_head(self):
        _,_,k,_=self.model.project((10,11),(0,1))
        mapped=k[:,torch.arange(4)//2]
        torch.testing.assert_close(mapped[:,0],mapped[:,1])
        torch.testing.assert_close(mapped[:,2],mapped[:,3])
        self.assertFalse(torch.equal(mapped[:,0],mapped[:,2]))


if __name__ == '__main__':
    unittest.main()
