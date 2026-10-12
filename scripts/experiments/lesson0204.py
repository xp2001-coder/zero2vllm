"""Compare static and continuous admission with real CPU reference KV execution.

Every selected request calls the model serially. Report logical rounds, never
GPU throughput or equal-duration latency. Workload order breaks arrival ties.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

from scripts.common import ROOT, source_fingerprint
from zero2vllm.engine.lifecycle import LifecycleEngine, Status
from zero2vllm.engine.runners import TinyCachedRunner
from zero2vllm.engine.scheduler import BatchScheduler
from zero2vllm.reference.attention import TinyAttentionLM, generate

LESSON = ROOT/'docs/chapters/part-02-engine/04-from-static-to-continuous-batching'
WORKLOADS = LESSON/'data/workloads.json'


def run_case(case, policy, model_config, overrides=None, max_rounds=1000):
    limits = {**case['limits'], **(overrides or {})}
    requests = case['requests']
    names = [r['id'] for r in requests]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate workload request ID.')
    for request in requests:
        if type(request['arrival']) is not int or request['arrival'] < 0:
            raise ValueError('arrival must be a nonnegative boundary index.')
        if 'cancel_at' in request and (type(request['cancel_at']) is not int or request['cancel_at'] < request['arrival']):
            raise ValueError('cancel_at must be an integer at or after arrival.')
    if type(max_rounds) is not int or max_rounds < 1:
        raise ValueError('max_rounds must be positive.')
    runner = TinyCachedRunner(TinyAttentionLM(**model_config))
    contexts = []
    new_context = runner.new_context
    def observe_context():
        context = new_context()
        contexts.append(context)
        return context
    runner.new_context = observe_context
    # Validate every request before replay, including requests arriving much later.
    with LifecycleEngine(runner, capacity_tokens=limits['capacity_tokens']) as validator:
        probe = BatchScheduler(validator, policy=policy,
                               max_running=limits['max_running'], max_tokens_per_round=limits['max_tokens_per_round'])
        for spec in requests:
            probe.submit(spec['id'], spec['prompt'], spec['max_new_tokens'], spec.get('eos_token_id'))

    traces = []
    last_arrival = max((r['arrival'] for r in requests), default=-1)
    with LifecycleEngine(runner, capacity_tokens=limits['capacity_tokens']) as engine:
        scheduler = BatchScheduler(engine, policy=policy,
                                   max_running=limits['max_running'], max_tokens_per_round=limits['max_tokens_per_round'])
        while scheduler.has_pending or scheduler.round_index <= last_arrival:
            boundary = scheduler.round_index
            if boundary >= max_rounds:
                raise RuntimeError('max_rounds reached; inspect arrivals, limits and progress. No partial success report was saved.')
            for spec in requests:
                if spec['arrival'] == boundary:
                    scheduler.submit(spec['id'], spec['prompt'], spec['max_new_tokens'], spec.get('eos_token_id'))
            for spec in requests:
                if spec.get('cancel_at') == boundary:
                    scheduler.cancel(spec['id'])
            if not scheduler.has_pending and boundary >= last_arrival:
                break  # Boundary events alone do not require a model iteration.
            traces.append(scheduler.tick().to_dict())
        records = scheduler.records()
        expected = {}
        for spec in requests:
            ids = generate(runner.model, spec['prompt'], spec['max_new_tokens'], use_cache=False).token_ids
            if spec.get('eos_token_id') in ids:
                ids = ids[:ids.index(spec['eos_token_id'])+1]
            expected[spec['id']] = ids
        agreements = []
        for record in records:
            ref = expected[record['request_id']]
            actual = list(record['output_ids'])
            agreements.append(actual == (ref[:len(actual)] if record['status'] == Status.CANCELLED else ref))
        assertions = {
            'outputs_match_independent_full_recompute': all(agreements),
            'all_terminal_without_failure': all(r['status'] in (Status.FINISHED, Status.CANCELLED) for r in records),
            'capacity_respected': all(0 <= t['reserved_before_execution'] <= limits['capacity_tokens'] and 0 <= t['reserved_after_execution'] <= limits['capacity_tokens'] for t in traces),
            'active_limit_respected': all(len(t['running']) <= limits['max_running'] for t in traces),
            'iteration_token_limit_respected': all(sum(i['input_tokens'] for i in t['planned']) <= limits['max_tokens_per_round'] for t in traces),
            'at_most_one_forward_per_request_per_round': all(len(t['executed']) == len(set(t['executed'])) for t in traces),
            'budget_returned': engine.used_tokens == 0,
            'all_contexts_closed': all(c.closed and c.kv is None for c in contexts),
            'finalized_once': all(sum(e['kind'] == 'finalized' for e in engine.events(n)) == 1 for n in names),
        }
        selections = sum(len(t['executed']) for t in traces)
        slots = len(traces)*limits['max_running']
        metrics = {'rounds': len(traces), 'request_steps': selections,
                   'processed_input_tokens': sum(i['input_tokens'] for t in traces for i in t['planned'] if i['request_id'] in t['executed']),
                   'emitted_tokens': sum(len(r['output_ids']) for r in records),
                   'selected_slot_fraction': selections/slots if slots else None,
                   'empty_rounds': sum(not t['executed'] for t in traces)}
        return {'name': case['name'], 'policy': policy, 'description': case['description'],
                'limits': limits, 'workload': requests, 'traces': traces, 'requests': records,
                'expected_without_cancellation': expected, 'metrics': metrics,
                'assertions': assertions, 'passed': all(assertions.values())}


def print_trace(result):
    print(f"\n{result['name']} / {result['policy']}")
    for t in result['traces']:
        selected = ' '.join(f"{i['request_id']}:{i['phase'][0].upper()}{i['input_tokens']}" for i in t['planned']) or '-'
        print(f"r={t['round_index']:02} [{selected}] wait={','.join(t['waiting']) or '-'} "
              f"skip={','.join(t['skipped_budget']) or '-'} block={t['admission_blocked'] or '-'}")
    print(json.dumps(result['metrics']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', default='all')
    parser.add_argument('--policy', choices=('both', 'static', 'continuous'), default='both')
    parser.add_argument('--max-running', type=int)
    parser.add_argument('--max-tokens-per-round', type=int)
    parser.add_argument('--capacity-tokens', type=int)
    parser.add_argument('--max-rounds', type=int, default=1000)
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=Path('.local/runs/part-02-engine/04-from-static-to-continuous-batching'))
    args = parser.parse_args()
    data = json.loads(WORKLOADS.read_text(encoding='utf-8'))
    cases = [c for c in data['cases'] if args.case in ('all', c['name'])]
    if not cases:
        parser.error('Unknown case. Available: '+', '.join(c['name'] for c in data['cases']))
    overrides = {k: getattr(args, k) for k in ('max_running', 'max_tokens_per_round', 'capacity_tokens') if getattr(args, k) is not None}
    policies = ('static', 'continuous') if args.policy == 'both' else (args.policy,)
    try:
        results = [run_case(c, p, data['model'], overrides, args.max_rounds) for c in cases for p in policies]
    except (ValueError, RuntimeError) as error:
        parser.exit(2, f'Experiment failed: {error}\n')
    report = {'experiment': 'part2-04-iteration-scheduling',
              'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'source': source_fingerprint(),
              'environment': {'python': platform.python_version(), 'system': platform.platform(), 'device': 'cpu'},
              'model': data['model'], 'cache_kind': 'real_python_kv_lists',
              'execution': 'serial per-request cached forwards; a plan is not a tensor batch',
              'clock': 'arrival/cancel at boundary r; selected outputs at boundary r+1; no wall-clock performance measurements',
              'metrics_note': 'selected_slot_fraction is logical selections/(rounds*max_running), not GPU utilization',
              'cases': results, 'gpu_status': {'cuda-a100': 'not_run', 'maca-c500': 'not_run'},
              'passed': all(c['passed'] for c in results)}
    directory = ROOT/args.output_dir
    directory.mkdir(parents=True, exist_ok=True)
    (directory/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    for result in results:
        if args.trace:
            print_trace(result)
        print(f"{'PASS' if result['passed'] else 'FAIL'} {result['name']}/{result['policy']}: {result['metrics']['rounds']} logical rounds")
    print(f"Summary: {directory/'summary.json'}")
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
