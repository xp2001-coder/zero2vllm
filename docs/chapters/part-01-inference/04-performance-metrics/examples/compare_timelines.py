"""Change one fact at a time in the lesson's simulated event data."""
import copy
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from zero2vllm.reference.metrics import metrics_experiment, percentile


def main():
    data = json.loads((Path(__file__).resolve().parents[1]/'data/request-events.json').read_text(encoding='utf-8'))
    base = metrics_experiment(data)
    late = copy.deepcopy(data)
    late['requests'][-1]['finished_s'] = 3.0
    partial = copy.deepcopy(data)
    partial['requests'][-1]['token_times_s'] = [1.5, 1.6]
    reports = [('base', base), ('late_failure', metrics_experiment(late)),
               ('partial_failure', metrics_experiment(partial)),
               ('strict_slo', metrics_experiment(data, 0.4, 0.2))]
    for name, report in reports:
        print(f"{name}: window={report['window']['duration_s']:.1f}s "
              f"success={report['counts']['successful']}/4 "
              f"tokens/s={report['rates']['output_tokens_per_s']:.6f} "
              f"good={report['counts']['good_requests']}")
    assert math.isclose(reports[1][1]['rates']['output_tokens_per_s'], 2.0)
    assert reports[1][1]['distributions']['ttft_s'] == base['distributions']['ttft_s']
    assert reports[2][1]['rates'] == base['rates']
    assert reports[2][1]['request_metrics'][-1]['output_tokens'] == 2
    assert reports[3][1]['counts']['good_requests'] == 0
    assert reports[3][1]['rates']['output_tokens_per_s'] == base['rates']['output_tokens_per_s']
    request_mean = base['distributions']['tpot_s']['mean']
    interval_mean = base['distributions']['itl_s']['mean']
    assert math.isclose(request_mean, 0.425)
    assert math.isclose(interval_mean, 1.1/3)
    print(f'request-mean TPOT={request_mean:.6f}s; interval-mean ITL={interval_mean:.6f}s')
    pooled = percentile([1, 1, 100], 0.95)
    round_average = (percentile([1, 1], 0.95)+percentile([100], 0.95))/2
    assert math.isclose(pooled, 90.1) and math.isclose(round_average, 50.5)
    print(f'pooled P95={pooled:.1f}; average of round P95={round_average:.1f}')
    print('PASS: simulated metrics, weighting, and window boundaries.')


if __name__ == '__main__':
    main()
