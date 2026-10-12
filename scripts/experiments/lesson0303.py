"""Lesson 3.3: packing contracts, real CPU attention tensors and pinned source."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

from scripts.common import ROOT, source_fingerprint
from scripts.experiments.lesson0301 import audit, fetch_sources, read_manifest
from zero2vllm.reference.execution import RequestSlice, pack_inputs

LESSON = ROOT/'docs/chapters/part-03-vllm/03-model-execution-and-attention-backends'
MANIFEST = LESSON/'data/source-manifest.json'


def requests():
    data = json.loads((LESSON/'data/requests.json').read_text(encoding='utf-8'))
    return tuple(RequestSlice(**r) for r in data['requests'])


def case(name, checks, **details):
    return {'name': name, 'checks': checks, 'passed': all(checks.values()), **details}


def metadata_cases():
    rows = requests()
    batch = pack_inputs(rows, ['B', 'A'])
    reverse = pack_inputs(rows, ['A', 'B'])
    return [case('packing', {
        'tokens': batch.input_ids == (23, 12, 13, 14),
        'positions': batch.positions == (3, 2, 3, 4),
        'slots': batch.slot_mapping == (23, 34, 35, 12),
        'query_boundaries': batch.query_start_loc == (0, 1, 4),
        'seq_lengths': batch.seq_lens == (4, 5),
        'logit_rows': batch.logits_indices == (0, 3),
    }, metadata=asdict(batch)), case('packing_reordered', {
        'tokens': reverse.input_ids == (12, 13, 14, 23),
        'positions': reverse.positions == (2, 3, 4, 3),
        'slots': reverse.slot_mapping == (34, 35, 12, 23),
        'boundaries': reverse.query_start_loc == (0, 3, 4),
        'mapping': reverse.req_id_to_index == {'A': 0, 'B': 1},
    }, metadata=asdict(reverse))]


def tensor_cases(dtype_name='float64'):
    import torch
    from torch.nn import functional as F
    from zero2vllm.modeling.packed_attention import AttentionFixture

    dtype = getattr(torch, dtype_name)
    tolerance = 1e-11 if dtype == torch.float64 else 2e-5
    model = AttentionFixture(dtype)
    rows = requests()
    result = model.execute(rows, ['B', 'A'])
    reverse = model.execute(rows, ['A', 'B'])
    padded = model.execute(rows, ['B', 'A'], pad_to=8, padding_token=31)
    x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=dtype)
    mathematical_weight = torch.tensor([[1, 0], [0, 1], [1, 1]], dtype=dtype)
    checkpoint_weight = mathematical_weight.T.contiguous()
    linear = F.linear(x, checkpoint_weight)
    fused_inputs = model.embedding[torch.tensor([23, 12, 13, 14])]
    separate = torch.cat([F.linear(fused_inputs, model.weights[n]) for n in ('q_proj', 'k_proj', 'v_proj')], -1)
    fusion_error = float((F.linear(fused_inputs, model.fused)-separate).abs().max())
    # Perturb A's last SCHEDULED position. Earlier A queries must not see it.
    changed_a = replace(rows[0], token_ids=(10, 11, 12, 13, 30, 15))
    causal = model.execute((changed_a, rows[1]), ['B', 'A'])
    prefix_error = float((causal['logits_by_id']['A'][:2]-result['logits_by_id']['A'][:2]).abs().max())
    changed_last = float((causal['logits_by_id']['A'][-1]-result['logits_by_id']['A'][-1]).abs().max())
    # Change the OTHER request's whole history, including its existing KV.
    changed_b = replace(rows[1], token_ids=(0, 1, 2, 3))
    isolated = model.execute((rows[0], changed_b), ['B', 'A'])
    isolation_error = float((isolated['logits_by_id']['A']-result['logits_by_id']['A']).abs().max())
    # GQA identity: indexed heads versus materialized repetition.
    _, _, k, v = model.project((10, 11, 12), (0, 1, 2))
    groups = torch.arange(4)//2
    written = result['written']
    cache_flat = result['cache'].view(2, -1, 2, 2)
    padding_error = float((padded['logits']-result['logits']).abs().max())
    reordered_error = max(float((result['logits_by_id'][r.request_id]-reverse['logits_by_id'][r.request_id]).abs().max()) for r in rows)
    details = {'dtype': dtype_name, 'tolerance': tolerance, 'shapes': result['shapes'],
               'max_abs_errors': result['errors'], 'candidate_ids': result['candidate_ids'],
               'returned_ids': result['returned_ids']}
    return [
        case('weight_orientation', {'matches_matmul': torch.equal(linear, x@mathematical_weight),
                                   'known_result': linear.tolist() == [[4., 5.], [10., 11.]]},
             mathematical_shape=[3, 2], checkpoint_shape=[2, 3], output=linear.tolist()),
        case('fused_qkv', {'matches_separate': fusion_error <= tolerance},
             input_shape=[4, 8], fused_weight_shape=list(model.fused.shape), split_sizes=[8, 4, 4], max_abs_error=fusion_error),
        case('paged_matches_dense', {'matches_independent_requests': max(result['errors'].values()) <= tolerance}, **details),
        case('reordered_results', {'same_rows_by_request': reordered_error <= tolerance,
                                  'same_returned_tokens': result['returned_ids'] == reverse['returned_ids']}, max_abs_error=reordered_error),
        case('causal_future', {'earlier_queries_unchanged': prefix_error <= tolerance,
                              'perturbation_has_effect': changed_last > tolerance}, earlier_error=prefix_error, changed_last_row=changed_last),
        case('isolated_requests', {'A_unchanged_when_B_changes': isolation_error <= tolerance}, max_abs_error=isolation_error),
        case('gqa_head_groups', {'keys': torch.equal(k[:, groups], k.repeat_interleave(2, 1)),
                               'values': torch.equal(v[:, groups], v.repeat_interleave(2, 1))}, query_to_kv_head=groups.tolist()),
        case('graph_padding', {'real_outputs_unchanged': padding_error <= tolerance,
                              'writes_unchanged': torch.equal(written, padded['written']),
                              'four_write_addresses_only': len(padded['batch'].slot_mapping) == 4,
                              'eight_projected_rows': padded['shapes']['q'][0] == 8},
             num_actual_tokens=4, num_input_tokens=8, max_abs_error=padding_error,
             note='Buffer padding only; no CUDA Graph capture or replay.'),
        case('partial_prefill_output', {'A_discarded': result['returned_ids']['A'] == [],
                                       'B_retained': result['returned_ids']['B'] == [result['candidate_ids'][0]],
                                       'two_candidate_rows': len(result['candidate_ids']) == 2},
             candidate_ids=result['candidate_ids'], returned_ids=result['returned_ids']),
        case('unwritten_tail', {'nine_valid_positions': int(written.sum()) == 9,
                               'unused_storage_stays_nan': bool(torch.isnan(cache_flat[:, ~written]).all()),
                               'outputs_finite': bool(torch.isfinite(result['logits']).all())}, written_slots=written.nonzero().flatten().tolist()),
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('all', 'metadata'), default='all')
    parser.add_argument('--dtype', choices=('float64', 'float32'), default='float64')
    parser.add_argument('--inspect-source', action='store_true')
    parser.add_argument('--fetch-source', action='store_true')
    parser.add_argument('--source-dir', type=Path, default=ROOT/'.local/sources/vllm-v0.8.5')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'.local/runs/part-03-vllm/03-model-execution')
    args = parser.parse_args(argv)
    report = {'recorded_at': datetime.now(timezone.utc).isoformat(), 'python': platform.python_version(),
              'verification_level': 'cpu_tensor_fixture' if args.mode == 'all' else 'metadata_only',
              'upstream_executed': False, 'full_llama_executed': False, 'gpu_validated': False,
              'course_source': source_fingerprint(), 'cases': []}
    try:
        report['cases'] = metadata_cases()
        if args.mode == 'all':
            try:
                import torch
            except ModuleNotFoundError as error:
                raise ValueError('Install the CPU PyTorch version in lesson 3.3 lab.md, or use --mode metadata') from error
            report.update(torch_version=torch.__version__, dtype=args.dtype, device='cpu')
            report['cases'].extend(tensor_cases(args.dtype))
        for item in report['cases']:
            print(f'{item["name"]}: {"PASS" if item["passed"] else "FAIL"}')
        report['passed'] = all(c['passed'] for c in report['cases'])
        if args.inspect_source or args.fetch_source:
            manifest = read_manifest(MANIFEST)
            downloaded = fetch_sources(args.source_dir, manifest) if args.fetch_source else []
            inspected = audit(args.source_dir, manifest)
            inspected['errors'] = [e.replace('first run with --fetch', 'first run with --fetch-source') for e in inspected['errors']]
            inspected['downloaded'] = downloaded
            report['source_inspection'] = inspected
            report['passed'] &= inspected['passed']
            print(f'Static inspection: {"PASS" if inspected["passed"] else "FAIL"}; {len(inspected["checkpoints"])} functions')
            for error in inspected['errors']: print(error, file=sys.stderr)
    except (OSError, ValueError, AssertionError) as error:
        report.update(passed=False, error=str(error))
        print(f'FAIL: {error}', file=sys.stderr)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')
    print('No upstream vLLM execution, pretrained-model inference or GPU timing.')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
