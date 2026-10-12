"""Lesson 3.2: offline counting experiments and opt-in pinned-source inspection."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from scripts.common import ROOT, source_fingerprint
from scripts.experiments.lesson0301 import audit, fetch_sources, read_manifest
from zero2vllm.reference.scheduling import (
    BlockLedger, CountRequest, CountScheduler, reusable_prefix_tokens,
)

LESSON = ROOT / 'docs/chapters/part-03-vllm/02-scheduling-and-kv-management-in-vllm'
MANIFEST = LESSON / 'data/source-manifest.json'
CASES = ('mixed', 'chunked', 'sequence_limit', 'head_blocking', 'preemption',
         'self_preemption', 'prefix_boundary', 'prefix_capacity',
         'shared_release', 'eviction')


def expect(condition, message):
    if not condition:
        raise AssertionError(message)


def mixed_rounds(token_budget=4, threshold=0, rounds=3):
    if type(rounds) is not int or not 1 <= rounds <= 16:
        raise ValueError('Use 1 to 16 rounds')
    scheduler = CountScheduler(token_budget=token_budget, threshold=threshold, num_blocks=32)
    scheduler.add(CountRequest('A', 5, 4), running=True)
    scheduler.add(CountRequest('B', 7))
    records = []
    for _ in range(rounds):
        row = scheduler.step()
        # An explicit teaching assumption, NOT sampled model output.
        outputs = {r['request']: 1 for r in row['scheduled']
                   if scheduler.requests[r['request']].computed == scheduler.requests[r['request']].known}
        scheduler.complete(outputs)
        row.update(synthetic_output_counts=outputs, after_feedback=scheduler.snapshot())
        records.append(row)
    return records


def run_case(name):
    if name == 'mixed':
        rounds = mixed_rounds()
        counts = [{r['request']: r['scheduled'] for r in row['scheduled']} for row in rounds]
        expect(counts == [{'A': 1, 'B': 3}, {'A': 1, 'B': 3}, {'A': 1, 'B': 1}], 'mixed counts')
        return {'rounds': rounds}
    if name == 'chunked':
        s = CountScheduler(token_budget=4)
        s.add(CountRequest('P', 9))
        rows = []
        for outputs in ({}, {}, {'P': 1}):
            row = s.step()
            s.complete(outputs)
            row.update(synthetic_output_counts=outputs, after_feedback=s.snapshot())
            rows.append(row)
        expect([r['scheduled'][0]['scheduled'] for r in rows] == [4, 4, 1], 'chunk counts')
        return {'rounds': rows}
    if name in ('sequence_limit', 'preemption'):
        s = CountScheduler(token_budget=4, max_seqs=1 if name == 'sequence_limit' else 4,
                           num_blocks=12 if name == 'sequence_limit' else 3)
        s.add(CountRequest('A', 5, 4), running=True)
        s.add(CountRequest('B', 5, 4) if name == 'preemption' else CountRequest('B', 7),
              running=name == 'preemption')
        first = s.step()
        if name == 'sequence_limit':
            expect(first['unused_token_budget'] == 3 and first['stop_reason'] == 'max_seqs', 'seq cap')
            s.complete({'A': 1})
            return {'rounds': [first], 'after_feedback': s.snapshot()}
        expect(first['preempted'] == ['B'] and first['unused_token_budget'] == 3, 'tail victim')
        expect(s.requests['B'].known == 5 and s.requests['B'].computed == 0, 'preserve known tokens')
        s.complete({'A': 1}, finished=['A'])
        first['after_feedback'] = s.snapshot()
        second = s.step()
        expect(second['scheduled'][0]['kind'] == 'resumed' and second['scheduled'][0]['scheduled'] == 4, 'resume recompute')
        s.complete()
        second['after_feedback'] = s.snapshot()
        return {'rounds': [first, second]}
    if name == 'head_blocking':
        s = CountScheduler(token_budget=8, num_blocks=2)
        s.add(CountRequest('large', 8))
        s.add(CountRequest('small', 1))
        row = s.step()
        expect(not row['scheduled'] and row['unused_token_budget'] == 8, 'head blocks smaller follower')
        s.complete()
        return {'rounds': [row]}
    if name == 'self_preemption':
        s = CountScheduler(num_blocks=2)
        s.add(CountRequest('A', 5, 4), running=True)
        row = s.step()
        expect(row['preempted'] == ['A'] and not row['scheduled'], 'self preemption')
        s.complete()
        return {'rounds': [row]}
    if name == 'prefix_boundary':
        rows = [{'known': n, 'cached_full_blocks': 2, 'reused': reusable_prefix_tokens(n, 2, 4)}
                for n in (7, 8, 9)]
        expect([r['reused'] for r in rows] == [4, 4, 8], 'full-block boundary')
        return {'block_size': 4, 'rows': rows}
    if name == 'prefix_capacity':
        p = BlockLedger(3)
        p.seed_cached([1, 2])
        before = p.snapshot()
        fresh = p.reserve('P', 9, hits=[1, 2])
        expect(fresh is None and p.snapshot() == before, 'hits cannot be double-counted as free')
        roomy = BlockLedger(4)
        roomy.seed_cached([1, 2])
        expect(roomy.reserve('P', 9, hits=[1, 2]) == [3], 'one genuinely free block suffices')
        return {'failed_before': before, 'failed_after': p.snapshot(), 'success': roomy.snapshot(),
                'required_new': 1, 'free_before_touch': 2, 'evictable_hits': 2}
    if name == 'shared_release':
        p = BlockLedger(4)
        p.seed_cached([1])
        p.reserve('A', 5, hits=[1])
        p.reserve('B', 5, hits=[1])
        shared = p.snapshot()
        expect(p.refs[1] == 2, 'shared refcount')
        p.release('A')
        after_a = p.snapshot()
        expect(p.refs[1] == 1 and 1 not in p.free, 'live shared block not reusable as storage')
        p.release('B')
        expect(p.refs[1] == 0 and 1 in p.free and 1 in p.cached, 'unowned cache retained')
        return {'shared': shared, 'after_A': after_a, 'after_B': p.snapshot()}
    if name == 'eviction':
        p = BlockLedger(2)
        p.seed_cached([1])
        before = p.snapshot()
        p.reserve('new_content', 1)
        expect(1 not in p.cached and p.refs[1] == 1, 'old cache identity invalidated')
        return {'before': before, 'after': p.snapshot()}
    raise ValueError(f'Unknown case: {name}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('all',) + CASES, default='all')
    parser.add_argument('--output-dir', type=Path, default=ROOT / '.local/runs/part-03-vllm/02-scheduling-kv')
    parser.add_argument('--inspect-source', action='store_true', help='Additionally inspect local pinned source')
    parser.add_argument('--fetch-source', action='store_true', help='Explicitly fetch missing files and inspect source')
    parser.add_argument('--source-dir', type=Path, default=ROOT / '.local/sources/vllm-v0.8.5')
    args = parser.parse_args(argv)
    selected = CASES if args.case == 'all' else (args.case,)
    report = {'recorded_at': datetime.now(timezone.utc).isoformat(),
              'verification_level': 'teaching_count_simulation',
              'executed_upstream_code': False, 'model_executed': False, 'gpu_validated': False,
              'course_source': source_fingerprint(), 'cases': []}
    try:
        for name in selected:
            result = run_case(name)
            report['cases'].append({'name': name, 'passed': True, **result})
            print(f'{name}: PASS')
        if args.inspect_source or args.fetch_source:
            manifest = read_manifest(MANIFEST)
            downloaded = fetch_sources(args.source_dir, manifest) if args.fetch_source else []
            report['source_inspection'] = audit(args.source_dir, manifest)
            report['source_inspection']['downloaded'] = downloaded
            expect(report['source_inspection']['passed'], '; '.join(report['source_inspection']['errors']))
            print(f'Static source inspection: PASS ({len(manifest["checkpoints"])} functions)')
        report['passed'] = True
    except (OSError, ValueError, AssertionError) as error:
        report.update(passed=False, error=str(error))
        print(f'FAIL: {error}', file=sys.stderr)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    print('Count simulation only; output counts are assumed, not sampled. No GPU timing.')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
