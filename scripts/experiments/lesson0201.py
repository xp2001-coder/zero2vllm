"""Execute lesson 2.1 CPU cases in one PyTorch process; save auditable JSON."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.common import source_fingerprint

CASES = [
    ('float64', {'prompt': [1, 2, 3, 4]}),
    ('float32', {'prompt': [1, 2, 3, 4], 'dtype': 'float32'}),
    ('seed8', {'prompt': [1, 2, 3, 4], 'seed': 8}),
    ('single-token', {'prompt': [15], 'layers': 1}),
    ('three-layers', {'prompt': [2, 2, 0], 'layers': 3}),
    ('zero-output', {'prompt': [1, 2], 'max_new_tokens': 0}),
    ('context-boundary', {'prompt': [1]*32, 'max_new_tokens': 1}),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('.local/runs/part-02-engine/01-pytorch-model'))
    parser.add_argument('--skip-tests', action='store_true')
    args = parser.parse_args()
    try:
        import torch
        from zero2vllm.modeling.torch_decoder import decoder_experiment
    except ModuleNotFoundError as error:
        if error.name != 'torch':
            raise
        parser.exit(2, 'Install CPU PyTorch as described in lesson 2.1 before running this script.\n')
    torch.set_num_threads(1)
    directory = (ROOT / args.output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'cases').mkdir(exist_ok=True)
    (directory / 'checks').mkdir(exist_ok=True)
    summary = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
               'source': source_fingerprint(), 'cases': [], 'tests_skipped': args.skip_tests,
               'gpu_status': 'not_run', 'passed': False}

    def save(name, data):
        (directory / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')

    for name, options in CASES:
        result = decoder_experiment(**options)
        result['created_at_utc'] = summary['created_at_utc']
        save('cases/'+name+'.json', result)
        passed = result['verification']['passed']
        summary['cases'].append({'name': name, 'options': options, 'output': 'cases/'+name+'.json', 'passed': passed})
        print(f'{"PASS" if passed else "FAIL"} {name}', flush=True)
    tests_passed = True
    if not args.skip_tests:
        env = {**os.environ, 'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8'}
        result = subprocess.run([sys.executable, '-m', 'unittest', 'tests.reference.test_decoder', 'tests.modeling.test_decoder', '-v'],
                                cwd=ROOT, env=env, capture_output=True, text=True, encoding='utf-8')
        (directory / 'checks/tests.txt').write_text(result.stdout+result.stderr, encoding='utf-8', newline='\n')
        tests_passed = result.returncode == 0
        summary['tests'] = {'passed': tests_passed, 'output': 'checks/tests.txt', 'exit_code': result.returncode}
    summary['passed'] = tests_passed and all(c['passed'] for c in summary['cases'])
    save('summary.json', summary)
    print(f'Summary: {directory / "summary.json"}')
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
