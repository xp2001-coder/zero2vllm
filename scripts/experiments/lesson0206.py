"""Lesson 2.6: actual CPU prefix reuse, reference counts, LRU and output checks."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

try:
    import torch
except ModuleNotFoundError as error:
    raise SystemExit('Lesson 2.6 requires CPU PyTorch; follow its lab.md.') from error

from scripts.common import ROOT, source_fingerprint
from zero2vllm.cache.prefix import PrefixKVStore
from zero2vllm.engine.lifecycle import LifecycleEngine
from zero2vllm.engine.prefix_runner import PrefixTinyRunner
from zero2vllm.engine.scheduler import BatchScheduler
from zero2vllm.reference.attention import TinyAttentionLM, generate

LESSON = ROOT/'docs/chapters/part-02-engine/06-prefix-caching'


def result(name, data, checks):
    return {'name': name, **data, 'assertions': checks, 'passed': all(checks.values())}


def run_request(runner, model, prompt, count):
    context = runner.new_context()
    ids, logits, snapshots = [], [], []
    try:
        context.reserve(len(prompt)+count-1)
        for _ in range(count):
            ids.append(runner.advance(prompt, ids, context))
            logits.append(list(context.last_logits))
            snapshots.append(runner.store.snapshot(context.handle))
    finally:
        context.close()
    reference = generate(model, prompt, count, use_cache=False)
    error = max(abs(a-b) for row, ref in zip(logits, reference.logits) for a, b in zip(row, ref))
    return {'prompt': prompt, 'output_ids': ids, 'reference_ids': reference.token_ids,
            'logits': logits, 'reference_logits': reference.logits, 'max_abs_error': error,
            'hit_positions': context.hit_positions, 'projected_positions': context.projected_positions,
            'attention_pairs': context.attention_pairs, 'snapshots': snapshots,
            'pool_after_release': runner.store.pool_snapshot()}


def cold_warm(data, dtype, size, reuse=True):
    model = TinyAttentionLM(**data['model'])
    runner = PrefixTinyRunner(num_blocks=24, block_size=size, dtype=getattr(torch, dtype), model=model, reuse=reuse)
    prompt, count = data['generation']['prompt'], data['generation']['max_new_tokens']
    cold = run_request(runner, model, prompt, count)
    warm = run_request(runner, model, prompt, count)
    hits = ((len(prompt)-1)//size)*size if reuse else 0
    tolerance = 1e-12 if dtype == 'float64' else 1e-6
    checks = {'cold_misses': cold['hit_positions'] == 0,
              'expected_warm_hit': warm['hit_positions'] == hits,
              'saved_projection_positions': cold['projected_positions']-warm['projected_positions'] == hits,
              'prefill_work': warm['projected_positions'] == len(prompt)-hits+count-1,
              'ids_match': all(r['output_ids'] == r['reference_ids'] for r in (cold, warm)),
              'logits_match': all(r['max_abs_error'] < tolerance for r in (cold, warm)),
              'no_active_references': runner.store.pool_snapshot()['active_block_references'] == 0}
    return result(f'cold-warm-{dtype}' if reuse else 'no-reuse',
                  {'dtype': dtype, 'block_size': size, 'absolute_tolerance': tolerance, 'cold': cold, 'warm': warm}, checks)


def branching(data):
    model = TinyAttentionLM(**data['model'])
    runner = PrefixTinyRunner(num_blocks=12, block_size=2, model=model)
    a, b = runner.new_context(), runner.new_context()
    pa, pb = data['branches']
    ids, snapshots, checks = [[], []], {}, {}
    try:
        a.reserve(len(pa)+2)
        b.reserve(len(pb)+2)
        ids[0].append(runner.advance(pa, [], a))
        ids[1].append(runner.advance(pb, [], b))
        snapshots['a_live'] = runner.store.snapshot(a.handle)
        snapshots['b_live'] = runner.store.snapshot(b.handle)
        snapshots['both_live_pool'] = runner.store.pool_snapshot()
        a.close()
        snapshots['b_after_a_release'] = runner.store.snapshot(b.handle)
        for _ in range(2):
            ids[1].append(runner.advance(pb, ids[1], b))
        checks = {'four_positions_reused': b.hit_positions == 4,
                  'same_physical_prefix': snapshots['a_live']['block_table'][:2] == snapshots['b_live']['block_table'][:2],
                  'two_live_references': snapshots['b_live']['block_refs'][:2] == [2, 2],
                  'one_after_a_release': snapshots['b_after_a_release']['block_refs'][:2] == [1, 1],
                  'a_output_matches': ids[0] == generate(model, pa, 1, use_cache=False).token_ids,
                  'b_output_matches': ids[1] == generate(model, pb, 3, use_cache=False).token_ids}
    finally:
        a.close()
        b.close()
    pool = runner.store.pool_snapshot()
    checks['no_active_references'] = pool['active_block_references'] == 0
    return result('branching-live', {'prompts': [pa, pb], 'outputs': ids, 'snapshots': snapshots, 'pool_after': pool}, checks)


def append_tokens(store, handle, tokens):
    values = torch.tensor([[[float(t), float(t)+0.5]] for t in tokens], dtype=torch.float64)
    store.append(handle, tokens, values, values)


def seed(store, tokens, namespace='default'):
    handle = store.acquire(tokens, len(tokens), namespace=namespace)
    append_tokens(store, handle, tokens)
    store.release(handle)


def boundaries(data):
    rows = []
    for size in data['boundary_block_sizes']:
        for length in data['boundary_lengths']:
            store = PrefixKVStore(16, size, head_dim=2)
            prompt = list(range(length))
            seed(store, prompt)
            h = store.acquire(prompt, length)
            hit = store.snapshot(h)['hit_positions']
            rows.append({'block_size': size, 'prompt_positions': length,
                         'hit_positions': hit, 'recomputed_prompt_positions': length-hit})
            store.release(h)
    return result('hit-boundaries', {'rows': rows},
                  {'full_blocks_only': all(r['hit_positions'] % r['block_size'] == 0 for r in rows),
                   'one_to_B_inputs_remain': all(1 <= r['recomputed_prompt_positions'] <= r['block_size'] for r in rows),
                   'formula_matches': all(r['hit_positions'] == ((r['prompt_positions']-1)//r['block_size'])*r['block_size'] for r in rows)})


def scopes(data, collision=False):
    store = PrefixKVStore(12, 2, head_dim=2, digest_fn=(lambda _: 'forced-collision') if collision else None)
    seed(store, data['scope_seed'], 'tenant-a')
    rows = []
    for probe in data['scope_probes']:
        h = store.acquire(probe['prompt'], len(probe['prompt']), namespace=probe['namespace'])
        rows.append({**probe, 'actual_hits': store.snapshot(h)['hit_positions']})
        store.release(h)
    return result('forced-collision' if collision else 'parent-and-namespace', {'probes': rows},
                  {'all_expected_hits': all(r['actual_hits'] == r['expected_hits'] for r in rows),
                   'no_active_references': store.pool_snapshot()['active_block_references'] == 0})


def pressure():
    store = PrefixKVStore(3, 2, head_dim=2)
    for tokens in ([1, 2], [3, 4], [5, 6]):
        seed(store, tokens)
    before = store.pool_snapshot()
    h = store.acquire([1, 2, 7], 3)
    pinned = store.snapshot(h)
    append_tokens(store, h, [7])
    q = store.acquire([8], 1)
    append_tokens(store, q, [8])
    visible = store.gather(h)[1][:, 0, 0].tolist()
    evicted = [e['block'] for e in store.events if e['kind'] == 'evicted']
    store.release(h)
    store.release(q)
    return result('lru-and-protection', {'before': before, 'pinned': pinned, 'evicted_blocks': evicted, 'visible_values': visible},
                  {'oldest_hit_protected': pinned['block_table'] == [0, 1],
                   'eviction_order': evicted == [1, 2], 'live_data_preserved': visible == [1, 2, 7],
                   'no_active_references': store.pool_snapshot()['active_block_references'] == 0})


def oom():
    store = PrefixKVStore(3, 2, head_dim=2)
    seed(store, [1, 2, 3])
    h = store.acquire([8, 9], 2)
    before, events = store.pool_snapshot(), list(store.events)
    rejected = False
    try:
        store.acquire([1, 2, 3], 6)
    except MemoryError:
        rejected = True
    after = store.pool_snapshot()
    checks = {'rejected': rejected, 'pool_unchanged': before == after, 'events_unchanged': events == store.events}
    store.release(h)
    checks['no_active_references'] = store.pool_snapshot()['active_block_references'] == 0
    return result('oom-atomic', {'before': before, 'after': after}, checks)


def reset_case():
    store = PrefixKVStore(4, 2, head_dim=2)
    h = store.acquire([1, 2, 3], 3)
    append_tokens(store, h, [1, 2, 3])
    rejected = False
    try:
        store.reset()
    except RuntimeError:
        rejected = True
    store.release(h)
    before = store.pool_snapshot()
    store.reset()
    after = store.pool_snapshot()
    h = store.acquire([1, 2, 3], 3)
    hit = store.snapshot(h)['hit_positions']
    store.release(h)
    return result('reset', {'before': before, 'after': after, 'next_hit_positions': hit},
                  {'live_reset_rejected': rejected, 'idle_cleared': after['empty_blocks'] == 4, 'next_request_misses': hit == 0})


def duplicates():
    store = PrefixKVStore(4, 2, head_dim=2)
    a, b = store.acquire([1, 2, 3], 3), store.acquire([1, 2, 3], 3)
    before = store.snapshot(b)
    append_tokens(store, a, [1, 2, 3])
    append_tokens(store, b, [1, 2, 3])
    after = store.snapshot(b)
    store.release(a)
    store.release(b)
    pool = store.pool_snapshot()
    return result('duplicate-cold', {'b_before': before, 'b_after': after, 'pool_after': pool},
                  {'live_table_unchanged': before['block_table'] == after['block_table'],
                   'one_indexed_copy': pool['idle_cached_blocks'] == 1,
                   'no_active_references': pool['active_block_references'] == 0})


def scheduler_case(data):
    model = TinyAttentionLM(**data['model'])
    runner = PrefixTinyRunner(num_blocks=12, block_size=2, model=model)
    contexts = []
    create, advance = runner.new_context, runner.advance
    def capture():
        ctx = create()
        contexts.append(ctx)
        return ctx
    def fail(prompt, generated, ctx):
        token = advance(prompt, generated, ctx)
        if prompt == (7, 8, 9):
            raise RuntimeError('injected after publishing valid KV')
        return token
    runner.new_context, runner.advance = capture, fail
    rows = []
    with LifecycleEngine(runner) as engine:
        scheduler = BatchScheduler(engine, max_running=3, max_tokens_per_round=15)
        for name, prompt in [('a', data['branches'][0]), ('b', data['branches'][1]), ('bad', [7, 8, 9])]:
            scheduler.submit(name, prompt, 3)
        while scheduler.has_pending:
            if scheduler.round_index == 1:
                scheduler.cancel('a')
            trace = scheduler.tick().to_dict()
            rows.append({**trace, 'pool_after': runner.store.pool_snapshot()})
        records = scheduler.records()
        checks = {'states': [r['status'] for r in records] == ['cancelled', 'finished', 'failed'],
                  'b_output_matches': list(records[1]['output_ids']) == generate(model, data['branches'][1], 3, use_cache=False).token_ids,
                  'conservative_ledger': all(r['pool_after']['active_unique_blocks']*2 <= r['reserved_after_execution'] for r in rows),
                  'b_reuses_four': contexts[1].hit_positions == 4, 'contexts_closed': all(c.closed for c in contexts),
                  'no_active_references': runner.store.pool_snapshot()['active_block_references'] == 0,
                  'ledger_returned': engine.used_tokens == 0}
    return result('scheduler-cleanup', {'rounds': rows, 'requests': records,
                                      'hit_positions': [c.hit_positions for c in contexts]}, checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('all', 'cold-warm', 'branching', 'boundaries', 'scope', 'collision', 'pressure', 'oom', 'reset', 'duplicates', 'scheduler', 'no-reuse'), default='all')
    parser.add_argument('--dtype', choices=('both', 'float64', 'float32'), default='both', help='Only cold-warm cases.')
    parser.add_argument('--block-size', type=int, default=2, help='Only cold-warm cases; fixed other cases use their teaching inputs.')
    parser.add_argument('--output-dir', type=Path, default=Path('.local/runs/part-02-engine/06-prefix-caching'))
    args = parser.parse_args()
    if not 1 <= args.block_size <= 16:
        parser.error('--block-size must be between 1 and 16.')
    torch.set_num_threads(1)
    data = json.loads((LESSON/'data/cases.json').read_text(encoding='utf-8'))
    cases = []
    if args.case in ('all', 'cold-warm'):
        for dtype in (('float64', 'float32') if args.dtype == 'both' else (args.dtype,)):
            cases.append(cold_warm(data, dtype, args.block_size))
    entries = [('branching', lambda: branching(data)), ('boundaries', lambda: boundaries(data)),
               ('scope', lambda: scopes(data)), ('collision', lambda: scopes(data, True)),
               ('pressure', pressure), ('oom', oom), ('reset', reset_case), ('duplicates', duplicates),
               ('scheduler', lambda: scheduler_case(data)), ('no-reuse', lambda: cold_warm(data, 'float64', 2, False))]
    for name, function in entries:
        if args.case in ('all', name):
            cases.append(function())
    report = {'experiment': 'part2-06-prefix-caching', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'source': source_fingerprint(), 'model': data['model'],
              'environment': {'python': platform.python_version(), 'system': platform.platform(), 'torch': torch.__version__, 'device': 'cpu', 'threads': 1},
              'execution': 'One-layer Tiny model; actual shared CPU full-block KV; copied gather reference attention; serial forwards.',
              'admission': 'Conservative rounded full reservation; shared blocks do not expand scheduler admission limits.',
              'cases': cases, 'passed': all(c['passed'] for c in cases),
              'gpu_status': {'cuda-a100': 'not_run', 'maca-c500': 'not_run'}}
    output = ROOT/args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')
    for case in cases:
        print(('PASS ' if case['passed'] else 'FAIL ')+case['name'])
        if not case['passed']:
            print(case['assertions'])
    print(f"Summary: {output/'summary.json'}")
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
