"""Lesson 2.5: CPU tensor KV pages, mapping, visibility and scheduler accounting."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform

try:
    import torch
except ModuleNotFoundError as error:
    raise SystemExit('Lesson 2.5 requires CPU PyTorch. Follow docs/chapters/part-02-engine/05-paged-caching/lab.md.') from error

from scripts.common import ROOT, source_fingerprint
from zero2vllm.cache.paged import PagedKVStore
from zero2vllm.engine.lifecycle import LifecycleEngine, Status
from zero2vllm.engine.paged_runner import PagedTinyRunner
from zero2vllm.engine.scheduler import BatchScheduler
from zero2vllm.reference.attention import TinyAttentionLM, generate

LESSON = ROOT/'docs/chapters/part-02-engine/05-paged-caching'
DATA = LESSON/'data/cases.json'


def case_result(name, details, assertions):
    return {'name': name, **details, 'assertions': assertions, 'passed': all(assertions.values())}


def mapping_case(data):
    spec = data['mapping']
    store = PagedKVStore(3, 2, head_dim=2)
    a, b = store.create(), store.create()
    store.reserve(a, 2)
    store.reserve(b, 2)
    keys = torch.tensor(spec['keys'], dtype=torch.float64).unsqueeze(1)
    values = torch.tensor(spec['values'], dtype=torch.float64).unsqueeze(1)
    store.append(a, keys[:2], values[:2])
    store.reserve(a, 3)
    store.append(a, keys[2:], values[2:])
    snapshot = store.snapshot(a)
    actual_k, actual_v = store.gather(a)
    query = torch.tensor([spec['query']], dtype=torch.float64)
    attended = store.attention(a, query)
    scores = [sum(q*k for q, k in zip(spec['query'], row))/math.sqrt(2) for row in spec['keys']]
    exp = [math.exp(x-max(scores)) for x in scores]
    probabilities = [x/sum(exp) for x in exp]
    expected = [sum(p*row[d] for p, row in zip(probabilities, spec['values'])) for d in range(2)]
    error = float((attended-torch.tensor([expected], dtype=torch.float64)).abs().max())
    store.release(a)
    store.release(b)
    return case_result('mapping', {'input': spec, 'snapshot': snapshot,
                                  'gathered_keys': actual_k.tolist(), 'gathered_values': actual_v.tolist(),
                                  'probabilities': probabilities, 'attention_output': attended.tolist(),
                                  'scalar_reference': expected, 'max_abs_error': error},
                       {'noncontiguous_table': snapshot['block_table'] == (0, 2),
                        'expected_slot_mapping': snapshot['written_slot_mapping'] == (0, 1, 4),
                        'gather_preserves_order': torch.equal(actual_k, keys) and torch.equal(actual_v, values),
                        'attention_matches_scalar': error < 1e-12, 'all_blocks_returned': store.free_blocks == 3})


def reuse_case():
    store = PagedKVStore(1, 2, head_dim=2)
    old = store.create()
    store.reserve(old, 2)
    store.append(old, torch.tensor([[[1., 0.]], [[0., 1.]]], dtype=torch.float64),
                 torch.tensor([[[10., 20.]], [[99., 99.]]], dtype=torch.float64))
    before = store.snapshot(old)
    store.release(old)
    new = store.create()
    store.reserve(new, 1)
    store.append(new, torch.zeros(1, 1, 2, dtype=torch.float64), torch.tensor([[[3., 7.]]], dtype=torch.float64))
    after = store.snapshot(new)
    raw_tail = store.physical_snapshot()[1][0, 1].tolist()
    attended = store.attention(new, torch.zeros(1, 2, dtype=torch.float64)).tolist()
    stale_rejected = False
    try:
        store.gather(old)
    except ValueError:
        stale_rejected = True
    store.release(new)
    return case_result('reuse', {'old': before, 'new': after, 'stale_physical_tail': raw_tail,
                                'visible_attention_output': attended},
                       {'same_physical_block': before['block_table'] == after['block_table'] == (0,),
                        'tail_actually_stale': raw_tail == [[99., 99.]],
                        'only_new_prefix_visible': attended == [[3., 7.]],
                        'old_handle_rejected': stale_rejected, 'all_blocks_returned': store.free_blocks == 1})


def oom_case():
    store = PagedKVStore(2, 2, head_dim=2)
    a = store.create()
    store.reserve(a, 2)
    store.append(a, torch.ones(1, 1, 2, dtype=torch.float64), torch.ones(1, 1, 2, dtype=torch.float64))
    before = store.snapshot(a)
    rejected = False
    try:
        store.reserve(a, 5)
    except MemoryError:
        rejected = True
    after = store.snapshot(a)
    remaining = store.free_blocks
    store.release(a)
    return case_result('oom', {'before': before, 'attempted_positions': 5, 'after': after, 'free_after_failure': remaining},
                       {'rejected': rejected, 'metadata_unchanged': before == after,
                        'free_list_unchanged': remaining == 1, 'all_blocks_returned': store.free_blocks == 2})


def logits_case(data, dtype_name, block_size):
    spec = data['generation']
    prompt, count = spec['prompt'], spec['max_new_tokens']
    required = len(prompt)+count-1
    dtype = getattr(torch, dtype_name)
    runner = PagedTinyRunner(num_blocks=math.ceil(required/block_size)+2, block_size=block_size,
                             dtype=dtype, model=TinyAttentionLM(**data['model']))
    reference = generate(runner.model, prompt, count, use_cache=False)
    context = runner.new_context()
    ids, logits, snapshots = [], [], []
    try:
        context.reserve(required)
        for step in range(count):
            inputs = prompt if step == 0 else [ids[-1]]
            row = runner.forward_tokens(inputs, context)[-1]
            logits.append(row)
            ids.append(max(range(runner.vocab_size), key=row.__getitem__))
            snapshots.append(runner.store.snapshot(context.handle))
    finally:
        context.close()
    error = max(abs(a-b) for row, ref in zip(logits, reference.logits) for a, b in zip(row, ref))
    tolerance = 1e-12 if dtype_name == 'float64' else 1e-6
    return case_result(f'logits-{dtype_name}-b{block_size}',
                       {'dtype': dtype_name, 'block_size': block_size, 'input': spec, 'output_ids': ids,
                        'expected_ids': reference.token_ids, 'logits': logits, 'reference_logits': reference.logits,
                        'snapshots': snapshots, 'max_abs_error': error, 'absolute_tolerance': tolerance},
                       {'ids_match_full_recompute': ids == reference.token_ids,
                        'logits_within_tolerance': error < tolerance,
                        'final_input_length': snapshots[-1]['written_positions'] == required,
                        'all_blocks_returned': runner.store.free_blocks == runner.store.num_blocks})


def engine_case(data, spec):
    runner = PagedTinyRunner(num_blocks=spec['num_blocks'], block_size=spec['block_size'], model=TinyAttentionLM(**data['model']))
    observations, contexts, rounds = [], [], []
    new_context, advance = runner.new_context, runner.advance
    def observed_context():
        context = new_context()
        contexts.append(context)
        return context
    def observed_advance(prompt, generated, context):
        token = advance(prompt, generated, context)
        observations.append({'prompt': prompt, 'generated_before': generated, **runner.store.snapshot(context.handle)})
        if any(tuple(r['prompt']) == prompt and r.get('fail_after_write') for r in spec['requests']):
            raise RuntimeError('injected after paged write')
        return token
    runner.new_context, runner.advance = observed_context, observed_advance
    with LifecycleEngine(runner) as engine:
        scheduler = BatchScheduler(engine, max_running=spec['max_running'], max_tokens_per_round=8)
        last_arrival = max(r['arrival'] for r in spec['requests'])
        while scheduler.has_pending or scheduler.round_index <= last_arrival:
            r = scheduler.round_index
            if r >= 128:
                raise RuntimeError('Replay did not finish within 128 rounds.')
            for request in spec['requests']:
                if request['arrival'] == r:
                    scheduler.submit(request['id'], request['prompt'], request['max_new_tokens'])
                if request.get('cancel_at') == r:
                    scheduler.cancel(request['id'])
            trace = scheduler.tick().to_dict()
            rounds.append({**trace, 'free_blocks_after': runner.store.free_blocks,
                           'physical_allocated_slots_after': (runner.store.num_blocks-runner.store.free_blocks)*runner.store.block_size})
        records = scheduler.records()
        expected = {r['id']: generate(runner.model, r['prompt'], r['max_new_tokens'], use_cache=False).token_ids for r in spec['requests']}
        assertions = {'ledger_matches_physical_allocation': all(r['reserved_after_execution'] == r['physical_allocated_slots_after'] for r in rounds),
                      'all_blocks_returned': runner.store.free_blocks == runner.store.num_blocks,
                      'token_ledger_returned': engine.used_tokens == 0,
                      'contexts_closed': all(c.closed for c in contexts),
                      'finalized_once': all(sum(e['kind'] == 'finalized' for e in engine.events(r['id'])) == 1 for r in spec['requests'])}
        for request, record in zip(spec['requests'], records):
            status = Status.FAILED if request.get('fail_after_write') else Status.CANCELLED if 'cancel_at' in request else Status.FINISHED
            assertions[f"{request['id']}_status"] = record['status'] == status
            output = list(record['output_ids'])
            assertions[f"{request['id']}_output"] = output == (expected[request['id']] if status == Status.FINISHED else expected[request['id']][:len(output)])
        if spec['name'] == 'admission':
            assertions['second_request_waits_for_two_blocks'] = rounds[0]['waiting'] == ('b',) and rounds[0]['reserved_after_execution'] == 8
        if spec['name'] == 'fragmented':
            assertions['d_gets_noncontiguous_blocks'] = any(o['prompt'] == (5, 6, 7) and o['block_table'] == (0, 2) for o in observations)
        if spec['name'] == 'cleanup':
            assertions['cancel_keeps_two_outputs'] = len(records[0]['output_ids']) == 2
        return case_result(spec['name'], {'input': spec, 'rounds': rounds, 'requests': records,
                                         'forward_observations': observations, 'expected_ids_without_stops': expected}, assertions)


def size_case(data):
    spec, rows = data['block_size_sweep'], []
    for size in spec['sizes']:
        store = PagedKVStore(32, size)
        handles, snapshots = [], []
        for reserve, written in zip(spec['reserved_lengths'], spec['written_lengths']):
            handle = store.create()
            handles.append(handle)
            store.reserve(handle, reserve)
            tensor = torch.zeros(written, 1, 8, dtype=torch.float64)
            store.append(handle, tensor, tensor)
            snapshots.append(store.snapshot(handle))
        allocated = sum(s['allocated_slots'] for s in snapshots)
        row = {'block_size': size, 'allocated_blocks': 32-store.free_blocks,
               'allocated_slots': allocated, 'rounding_tail_slots': sum(s['rounding_tail_slots'] for s in snapshots),
               'reserved_unwritten_positions': sum(s['reserved_unwritten_positions'] for s in snapshots),
               'written_positions': sum(s['written_positions'] for s in snapshots),
               'allocated_payload_bytes': allocated*2*8*8,
               'whole_pool_payload_bytes': store.storage_bytes}
        for handle in handles:
            store.release(handle)
        row['all_blocks_returned'] = store.free_blocks == store.num_blocks
        rows.append(row)
    return case_result('block-sizes', {'input': spec, 'rows': rows},
                       {'slot_conservation': all(r['allocated_slots'] == r['written_positions']+r['reserved_unwritten_positions']+r['rounding_tail_slots'] for r in rows),
                        'bounded_rounding_tail': all(0 <= r['rounding_tail_slots'] <= len(spec['reserved_lengths'])*(r['block_size']-1) for r in rows),
                        'all_blocks_returned': all(r['all_blocks_returned'] for r in rows)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('all', 'mapping', 'reuse', 'oom', 'logits', 'admission', 'fragmented', 'cleanup', 'block-sizes'), default='all')
    parser.add_argument('--dtype', choices=('both', 'float64', 'float32'), default='both', help='Applies to logits cases.')
    parser.add_argument('--block-size', type=int, help='Override logits block sizes; default runs 2 and 4.')
    parser.add_argument('--output-dir', type=Path, default=Path('.local/runs/part-02-engine/05-paged-caching'))
    args = parser.parse_args()
    if args.block_size is not None and not 1 <= args.block_size <= 32:
        parser.error('--block-size must be between 1 and 32.')
    torch.set_num_threads(1)
    data = json.loads(DATA.read_text(encoding='utf-8'))
    cases = []
    for name, function in [('mapping', lambda: mapping_case(data)), ('reuse', reuse_case), ('oom', oom_case)]:
        if args.case in ('all', name):
            cases.append(function())
    if args.case in ('all', 'logits'):
        for dtype in (('float64', 'float32') if args.dtype == 'both' else (args.dtype,)):
            for size in ((2, 4) if args.block_size is None else (args.block_size,)):
                cases.append(logits_case(data, dtype, size))
    for spec in data['engine_cases']:
        if args.case in ('all', spec['name']):
            cases.append(engine_case(data, spec))
    if args.case in ('all', 'block-sizes'):
        cases.append(size_case(data))
    report = {'experiment': 'part2-05-paged-kv-storage', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'source': source_fingerprint(),
              'environment': {'python': platform.python_version(), 'system': platform.platform(), 'torch': torch.__version__, 'threads': 1, 'device': 'cpu'},
              'model': data['model'], 'tensor_layout': 'separate K/V [num_blocks, block_size, kv_heads, head_dim]; one layer per store',
              'execution': 'TinyAttentionLM projections + CPU tensor scatter/gather and attention; serial per-request forwards',
              'allocation_policy': 'engine reserves ceil((P+N-1)/B) full blocks up front; standalone store can grow explicitly',
              'cases': cases, 'gpu_status': {'cuda-a100': 'not_run', 'maca-c500': 'not_run'}, 'passed': all(c['passed'] for c in cases)}
    output = ROOT/args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')
    for case in cases:
        detail = f"; max_abs_error={case['max_abs_error']:.3g}" if 'max_abs_error' in case else ''
        print(f"{'PASS' if case['passed'] else 'FAIL'} {case['name']}{detail}")
    print(f"Summary: {output/'summary.json'}")
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
