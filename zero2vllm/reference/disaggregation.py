"""Audit a fixed-window, fixed-output-length PD comparison; no GPU simulation.

All token arrival times use one client monotonic clock in milliseconds. Records
must already be censored at the observation boundary. Incomplete/failed records
remain in the offered cohort and can never contribute to goodput.
"""
from collections import Counter
import math


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, label, minimum=0):
    require(type(value) in (int, float) and math.isfinite(value) and value >= minimum,
            f'{label}: expected finite number >= {minimum}')
    return value


def positive_int(value, label):
    require(type(value) is int and value > 0, f'{label}: expected positive integer')
    return value


def percentiles(values):
    """Nearest-rank quantiles; undefined empty populations are JSON null."""
    ordered = sorted(values)
    return {f'p{p}': ordered[math.ceil(p / 100 * len(ordered)) - 1] if ordered else None
            for p in (50, 95, 99)}


def analyze(document, slo_override=None):
    """Validate a comparison contract and compute request-level joint SLOs.

    JSON schema/fields are documented in lesson 13.1's lab.md. Matching metadata
    establishes declared comparability, not actual measurement provenance.
    """
    require(document.get('schema_version') == 1, 'unsupported schema_version')
    require(document.get('source_kind') in ('synthetic', 'measured'), 'invalid source_kind')
    require(bool(document.get('provenance')), 'provenance is required')
    require(document.get('clock') == 'client_monotonic_ms', 'one client clock is required')
    contract = document['contract']
    for key in ('model_revision', 'tokenizer_revision', 'precision', 'workload_revision',
                'generation_config', 'cache_policy', 'gpu_model', 'runtime_revision'):
        require(bool(contract.get(key)), f'missing contract.{key}')
    gpu_budget = positive_int(contract['gpu_count'], 'gpu_count')
    start = number(document['window']['start_ms'], 'window.start_ms')
    end = number(document['window']['end_ms'], 'window.end_ms')
    require(end > start, 'observation window must have positive duration')
    seconds = (end - start) / 1000
    slo = dict(document['slo'])
    if slo_override:
        require(set(slo_override) <= {'ttft_ms', 'tpot_ms', 'max_itl_ms'}, 'unknown SLO override')
        slo.update(slo_override)
    require(set(slo) == {'ttft_ms', 'tpot_ms', 'max_itl_ms'}, 'expected three SLO thresholds')
    for key, value in slo.items():
        number(value, f'slo.{key}')
    workload = {}
    for request in document['workload']:
        rid = request['id']
        require(isinstance(rid, str) and rid and rid not in workload, 'duplicate/invalid workload id')
        arrival = number(request['arrival_ms'], f'{rid}.arrival_ms')
        require(start <= arrival < end, f'{rid}: arrival outside [start, end)')
        positive_int(request['input_tokens'], f'{rid}.input_tokens')
        require(positive_int(request['output_tokens'], f'{rid}.output_tokens') >= 2,
                f'{rid}: this protocol requires at least two output tokens')
        workload[rid] = request
    require(bool(workload), 'workload must not be empty')
    results, names = [], set()
    for run in document['runs']:
        name = run['name']
        require(isinstance(name, str) and name and name not in names, 'duplicate/invalid run name')
        names.add(name)
        require(bool(run.get('configuration')), f'{name}: configuration is required')
        allocations = run['allocation']
        require(bool(allocations), f'{name}: allocation is required')
        roles = [item['role'] for item in allocations]
        require(len(set(roles)) == len(roles) and set(roles) in ({'mixed'}, {'prefill', 'decode'}),
                f'{name}: use mixed or prefill/decode roles')
        require(sum(positive_int(a['gpus'], 'allocation.gpus') for a in allocations) == gpu_budget,
                f'{name}: unequal GPU budget')
        rows, ids = [], set()
        for record in run['requests']:
            rid = record['id']
            require(rid in workload and rid not in ids, f'{name}: unknown/duplicate request id')
            ids.add(rid)
            request = workload[rid]
            status = record['status']
            require(status in ('completed', 'rejected', 'error', 'timeout', 'cancelled', 'incomplete'),
                    f'{rid}: unknown status')
            finish = number(record['end_ms'], f'{rid}.end_ms')
            require(request['arrival_ms'] <= finish <= end, f'{rid}: invalid end time')
            tokens = record['token_arrivals_ms']
            require(isinstance(tokens, list), f'{rid}: token times must be a list')
            previous = request['arrival_ms']
            for t in tokens:
                number(t, f'{rid}.token_time')
                require(previous <= t <= finish, f'{rid}: token times must be ordered before end')
                previous = t
            require(len(tokens) <= request['output_tokens'], f'{rid}: too many output tokens')
            if status == 'completed':
                require(len(tokens) == request['output_tokens'], f'{rid}: completed output length mismatch')
            if status == 'rejected':
                require(not tokens, f'{rid}: rejected requests cannot emit tokens')
            if status == 'incomplete':
                require(finish == end, f'{rid}: incomplete records must end at observation cutoff')
            gaps = [b - a for a, b in zip(tokens, tokens[1:])]
            metrics = {'ttft_ms': tokens[0] - request['arrival_ms'] if tokens else None,
                       'tpot_ms': (tokens[-1] - tokens[0]) / (len(tokens) - 1) if gaps else None,
                       'max_itl_ms': max(gaps) if gaps else None}
            violations = [key for key, value in metrics.items() if value is not None and value > slo[key]]
            qualified = status == 'completed' and not violations
            rows.append(dict(id=rid, status=status, output_tokens=len(tokens), **metrics,
                             qualified=qualified, violations=violations,
                             exclusion=None if status == 'completed' else status))
        require(ids == set(workload), f'{name}: missing offered requests')
        complete = [r for r in rows if r['status'] == 'completed']
        qualified_count = sum(r['qualified'] for r in rows)
        results.append(dict(name=name, allocation=allocations, configuration=run['configuration'],
            offered=len(rows), completed=len(complete), status_counts=dict(Counter(r['status'] for r in rows)),
            qualified=qualified_count, joint_success_fraction=qualified_count / len(rows),
            completed_rps=len(complete) / seconds, goodput_rps=qualified_count / seconds,
            observed_output_tokens_per_second=sum(r['output_tokens'] for r in rows) / seconds,
            completed_request_percentiles={key: percentiles([r[key] for r in complete]) for key in slo},
            requests=rows))
    require(bool(results), 'runs must not be empty')
    return dict(schema_version=1, source_kind=document['source_kind'], provenance=document['provenance'],
        contract=contract, window=document['window'], duration_seconds=seconds, slo=slo,
        percentile_method='nearest_rank; completed requests only; max_itl is a per-request maximum',
        scope='Fixed finite window, no pre-window arrivals. Not a steady-state capacity estimate, GPU simulation or correctness test.',
        runs=results)
