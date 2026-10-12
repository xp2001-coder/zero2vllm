from dataclasses import replace
import math
import random
import unittest

from scripts.experiments.lesson0304 import experiments
from zero2vllm.reference.runner_state import (
    StablePool, StagedTable, copy_race_demo, expand_mapping, graph_dispatch,
    selected_logprobs, gumbel_choice,
)


class RunnerStateTests(unittest.TestCase):
    def pool(self):
        p = StablePool()
        p.add('A', [1, 2, 3], 1)
        p.add('B', [4, 5], 1)
        p.add('C', [6, 7], 0)
        return p

    def test_mapping_order_does_not_move_persistent_rows(self):
        p = self.pool(); before = tuple(p.rows)
        self.assertEqual(p.view(['C', 'A']).indices, (2, 0))
        self.assertEqual([r.request_id for r in p.gather(p.view(['C', 'A']))], ['C', 'A'])
        self.assertEqual(tuple(p.rows), before)

    def test_scatter_updates_only_scheduled_requests(self):
        p = self.pool(); p.advance(p.view(['B', 'A']), [1, 2])
        self.assertEqual([r.computed for r in p.rows], [3, 2, 0])

    def test_invalid_batch_progress_has_no_partial_writes(self):
        p = self.pool(); before = tuple(p.rows)
        with self.assertRaises(ValueError): p.advance(p.view(['A', 'B']), [1, 8])
        self.assertEqual(tuple(p.rows), before)

    def test_same_id_reuse_cannot_accept_old_generation(self):
        p = self.pool(); old = p.view(['A'])
        p.remove('A'); p.add('A', [9], 0)
        with self.assertRaisesRegex(ValueError, 'stale'): p.advance(old, [1])
        self.assertEqual(p.rows[0].computed, 0)

    def test_different_id_reuse_cannot_accept_old_result(self):
        p = self.pool(); old = p.view(['B'])
        p.remove('B'); p.add('D', [8], 0)
        with self.assertRaises(ValueError): p.gather(old)

    def test_duplicate_unknown_and_exhausted_pool(self):
        p = self.pool()
        for call in [lambda:p.add('D',[1]),lambda:p.add('A',[1]),lambda:p.view(['A','A']),lambda:p.view(['Z'])]:
            with self.assertRaises(ValueError): call()

    def test_request_tokens_are_snapshotted(self):
        ids=[1,2];p=StablePool();p.add('A',ids);ids[0]=9
        self.assertEqual(p.rows[0].token_ids,(1,2))

    def test_empty_step_and_repeated_removal(self):
        p=self.pool();p.advance(p.view([]),[]);p.remove('A');p.remove('A')
        self.assertEqual(p.index,{'B':1,'C':2})

    def test_malformed_step_is_rejected(self):
        p=self.pool();s=p.view(['A'])
        for broken in [replace(s,indices=(-1,)),replace(s,generations=()),replace(s,indices=(True,))]:
            with self.assertRaises(ValueError):p.gather(broken)

    def test_stage_snapshots_values_and_apply_clears(self):
        t=StagedTable();v=[8,3];t.stage(0,2,v);v[0]=99
        self.assertEqual(t.table[0],[0]*6)
        t.apply();self.assertEqual(t.table[0],[0,0,8,3,0,0])
        self.assertEqual(t.apply()['contents'],[])

    def test_overlapping_writes_rejected_without_damaging_pending(self):
        t=StagedTable();t.stage(1,1,[3,4]);before=t.packed()
        with self.assertRaises(ValueError):t.stage(1,2,[9])
        self.assertEqual(t.packed(),before)
        t.stage(1,3,[5]);t.apply();self.assertEqual(t.table[1],[0,3,4,5,0,0])

    def test_scatter_bounds_and_int32_limits(self):
        t=StagedTable()
        for row,start,values in [(-1,0,[1]),(3,0,[1]),(0,6,[1]),(0,0,[2**31]),(0,0,[True])]:
            with self.assertRaises(ValueError):t.stage(row,start,values)
        self.assertEqual(t.pending,[])

    def test_scatter_random_disjoint_writes_match_dense(self):
        rng=random.Random(34)
        for _ in range(20):
            t=StagedTable(4,9);dense=[[0]*9 for _ in range(4)]
            for row in range(4):
                start=rng.randrange(9);values=[rng.randrange(-20,20) for _ in range(rng.randrange(10-start))]
                t.stage(row,start,values);dense[row][start:start+len(values)]=values
            t.apply();self.assertEqual(t.table,dense)

    def test_payload_counts_values_and_metadata(self):
        t=StagedTable();t.stage(0,2,[8,3]);t.stage(2,0,[5]);r=t.packed()
        self.assertEqual((r['cumulative_ends'],r['payload_bytes'],r['full_table_bytes']),([2,3],36,72))

    def test_copy_ownership_counterexample(self):
        self.assertEqual(copy_race_demo(False)['received'],[9,7])
        self.assertEqual(copy_race_demo(True)['received'],[3,7])

    def test_expanded_logits_ownership(self):
        self.assertEqual(expand_mapping([2,0],[1,3]),([2,0,0,0],[0,0,1,2]))
        with self.assertRaises(ValueError):expand_mapping([1],[0])

    def test_gumbel_is_invariant_to_log_normalization(self):
        xs=[2.,1.,0.,-1.];us=[.13,.81,.4,.73]
        z=math.log(sum(math.exp(x) for x in xs))
        self.assertEqual(gumbel_choice(xs,us),gumbel_choice([x-z for x in xs],us))
        with self.assertRaises(ValueError):gumbel_choice([1],[0])

    def test_topk_keeps_full_vocab_normalizer_and_is_shift_stable(self):
        a=selected_logprobs([2.,1.,0.,-1.],2);b=selected_logprobs([1002.,1001.,1000.,999.],2)
        self.assertEqual(a,b)
        self.assertLess(sum(math.exp(x) for x in a['logprobs']),1)
        self.assertAlmostEqual(a['logprobs'][0],2-math.log(sum(math.exp(x) for x in [2,1,0,-1])))

    def test_graph_requires_bucket_and_full_capture(self):
        self.assertEqual(graph_dispatch('FULL',[3],captured=[])['mode'],'NONE')
        self.assertEqual(graph_dispatch('FULL',[9],captured=[8])['mode'],'NONE')
        r=graph_dispatch('FULL',[3],captured=[4]);self.assertEqual((r['mode'],r['bucket']),('FULL',4))

    def test_mixed_and_uniform_decode_choose_separate_modes(self):
        self.assertEqual(graph_dispatch('FULL_AND_PIECEWISE',[1,1],captured=[2])['mode'],'FULL')
        self.assertEqual(graph_dispatch('FULL_AND_PIECEWISE',[1,3],captured=[4])['mode'],'PIECEWISE')
        self.assertEqual(graph_dispatch('FULL_DECODE_ONLY',[1,3],captured=[4])['mode'],'NONE')

    def test_invalid_graph_and_sampling_inputs(self):
        for call in [lambda:graph_dispatch('AUTO',[1]),lambda:graph_dispatch('FULL',[]),lambda:selected_logprobs([float('nan')],1),lambda:selected_logprobs([1],2)]:
            with self.assertRaises(ValueError):call()

    def test_all_teaching_cases_pass(self):
        cases=experiments();self.assertEqual(len(cases),10);self.assertTrue(all(c['passed'] for c in cases))


if __name__ == '__main__': unittest.main()
