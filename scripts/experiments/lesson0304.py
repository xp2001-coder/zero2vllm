"""Lesson 3.4: CPU state, copy ownership, sampling math and graph dispatch."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform

from scripts.common import ROOT, source_fingerprint
from scripts.experiments.source_records import inspect_sources
from zero2vllm.reference.runner_state import (
    StablePool, StagedTable, copy_race_demo, expand_mapping, graph_dispatch,
    selected_logprobs, gumbel_choice,
)

LESSON = ROOT/'docs/chapters/part-03-vllm/04-v1-and-modelrunner-evolution'


def case(name, checks, **details):
    return dict(name=name, checks=checks, passed=all(checks.values()), **details)


def rejected(call):
    try: call()
    except ValueError: return True
    return False


def experiments():
    pool = StablePool()
    pool.add('A', [10, 11, 12, 13, 14], 2)
    pool.add('B', [20, 21, 22, 23], 3)
    pool.add('C', [30, 31], 0)
    first = pool.view(['B', 'A'])
    before = tuple(pool.rows)
    gathered = pool.gather(first)
    pool.advance(first, [1, 3])
    advanced = {r.request_id: r.computed for r in pool.rows}
    after = tuple(pool.rows)
    pool.remove('B')
    pool.add('D', [7, 8], 0)
    stale_rejected = rejected(lambda: pool.advance(first, [0, 0]))
    pool.remove('A')  # preemption at the runner-state level
    pool.add('E', [2], 0)
    pool.remove('C')
    new_slot = pool.add('A', [10, 11, 12, 13, 14], 0)
    table = StagedTable()
    table.stage(0, 2, [8, 3])
    table.stage(2, 0, [5])
    untouched_before_apply = all(x == 0 for row in table.table for x in row)
    payload = table.apply()
    dense = [[0, 0, 8, 3, 0, 0], [0]*6, [5, 0, 0, 0, 0, 0]]
    duplicate = StagedTable()
    duplicate.stage(0, 1, [1, 2])
    overlap_rejected = rejected(lambda: duplicate.stage(0, 2, [9]))
    unsafe, safe = copy_race_demo(False), copy_race_demo(True)
    indices, positions = expand_mapping([2, 0], [1, 3])
    logits, uniforms = [2., 1., 0., -1.], [.13, .81, .4, .73]
    shifted = [v - max(logits) for v in logits]
    normalizer = math.log(sum(math.exp(v) for v in shifted))
    log_probs = [v - normalizer for v in shifted]
    top = selected_logprobs(logits, 2)
    graph_cases = {
        'decode_ready': graph_dispatch('FULL_AND_PIECEWISE', [1, 1, 1], captured=[4]),
        'mixed': graph_dispatch('FULL_AND_PIECEWISE', [1, 3], captured=[4]),
        'uncaptured': graph_dispatch('FULL', [1, 1, 1], captured=[]),
        'too_large': graph_dispatch('FULL', [9], captured=[8]),
        'decode_only_mixed': graph_dispatch('FULL_DECODE_ONLY', [1, 3], captured=[4]),
    }
    return [
        case('stable_rows', {'mapping_is_1_0': first.indices == (1, 0),
                             'identity_preserved': [r.request_id for r in gathered] == ['B', 'A'],
                             'unscheduled_C_unchanged': after[2] == before[2]},
             step=asdict(first), persistent_before=[asdict(r) for r in before]),
        case('scatter_progress', {'A_5_B_4_C_0': advanced == {'A': 5, 'B': 4, 'C': 0}}, computed_by_id=advanced),
        case('row_reuse', {'stale_step_rejected': stale_rejected, 'new_D_untouched': pool.rows[pool.index['D']].computed == 0},
             note='Generation is a teaching guard, not an asserted upstream field.'),
        case('preempt_and_resume', {'A_reloaded_elsewhere': new_slot == 2, 'progress_reset_explicitly': pool.rows[new_slot].computed == 0},
             old_slot=0, resumed_slot=new_slot, preserved_history=list(pool.rows[new_slot].token_ids)),
        case('staged_writes', {'staging_did_not_mutate_base': untouched_before_apply,
                               'matches_dense_update': table.table == dense,
                               'pending_cleared': table.packed()['contents'] == []}, payload=payload, result=table.table),
        case('overlap_guard', {'ambiguous_writes_rejected': overlap_rejected}, note='No GPU ordering guarantee is invented.'),
        case('copy_ownership', {'unsafe_reads_new_step': unsafe['received'] == [9, 7],
                                'snapshot_preserves_old_step': safe['received'] == [3, 7]}, unsafe=unsafe, snapshot=safe),
        case('expanded_sampling_map', {'state_indices': indices == [2, 0, 0, 0], 'local_positions': positions == [0, 0, 1, 2]},
             indices=indices, local_positions=positions),
        case('sampling_algebra', {'gumbel_equivalence': gumbel_choice(logits, uniforms) == gumbel_choice(log_probs, uniforms),
                                  'topk_uses_full_normalizer': max(abs(v-log_probs[i]) for i, v in zip(top['indices'], top['logprobs'])) < 1e-12},
             logits=logits, uniforms=uniforms, gumbel_token=gumbel_choice(logits, uniforms), top_logprobs=top,
             note='Fixed-uniform algebra only; not Triton RNG or statistical throughput validation.'),
        case('graph_dispatch', {'ready_full': graph_cases['decode_ready']['mode'] == 'FULL' and graph_cases['decode_ready']['bucket'] == 4,
                                'mixed_piecewise': graph_cases['mixed']['mode'] == 'PIECEWISE',
                                'uncaptured_eager': graph_cases['uncaptured']['mode'] == 'NONE',
                                'large_eager': graph_cases['too_large']['mode'] == 'NONE',
                                'decode_only_mixed_eager': graph_cases['decode_only_mixed']['mode'] == 'NONE'}, decisions=graph_cases),
    ]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inspect-source', action='store_true')
    p.add_argument('--fetch-source', action='store_true')
    p.add_argument('--source-root', type=Path, default=ROOT/'.local/sources')
    p.add_argument('--output-dir', type=Path, default=ROOT/'.local/runs/part-03-vllm/04-runner-evolution')
    a = p.parse_args(argv)
    report = dict(recorded_at=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
                  course_source=source_fingerprint(), verification_level='cpu_semantics_and_math',
                  upstream_executed=False, gpu_validated=False, cases=[], passed=False)
    try:
        report['cases'] = experiments()
        report['passed'] = all(c['passed'] for c in report['cases'])
        if a.inspect_source or a.fetch_source:
            report['source_inspection'] = inspect_sources([LESSON/'data/source-v085.json', LESSON/'data/source-v0171.json'], a.source_root, a.fetch_source)
            report['passed'] &= all(r['passed'] for r in report['source_inspection'])
    except (ValueError, OSError) as error:
        report.update(passed=False, error=str(error))
    a.output_dir.mkdir(parents=True, exist_ok=True)
    (a.output_dir/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    for c in report['cases']: print(c['name'], 'PASS' if c['passed'] else 'FAIL')
    if report.get('error'): print(report['error'])
    for r in report.get('source_inspection', []): print(r['tag'], 'static inspection', r['passed'], r['errors'])
    print('No upstream runner, GPU transfer, Triton sampler or CUDA Graph executed.')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
