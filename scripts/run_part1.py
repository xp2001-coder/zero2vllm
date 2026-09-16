"""Run the documented Part 1 CPU cases, unit tests and course checks.

All commands use this checkout and the current interpreter; no shell, network,
third-party dependency, model download, GPU call, or git mutation is involved.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    ("01-next-token", ["next-token-demo"]),
    ("02-phases", ["phase-demo"]),
    ("03-cache", ["kv-demo"]),
    ("04-metrics", ["metrics-demo"]),
    ("05-structures", ["structure-demo"]),
    ("06-sampling", ["sampling-demo"]),
    ("cache-generation", ["demo"]),
    ("phase-short", ["phase-demo", "--prompt", "1,2", "--max-new-tokens", "3"]),
    ("phase-batch64", ["phase-demo", "--decode-batch", "64"]),
    ("cache-chunk1", ["kv-demo", "--chunk-size", "1"]),
    ("cache-chunk3", ["kv-demo", "--chunk-size", "3"]),
    ("cache-capacity", ["kv-demo", "--sequence-length", "8192", "--batch-size", "4"]),
    ("cache-gqa4", ["kv-demo", "--kv-heads", "4"]),
    ("metrics-strict", ["metrics-demo", "--ttft-slo", "0.4", "--tpot-slo", "0.2"]),
    ("structure-window1", ["structure-demo", "--window", "1"]),
    ("structure-window8", ["structure-demo", "--window", "8"]),
    ("sampling-greedy", ["sampling-demo", "--temperature", "0", "--max-new-tokens", "3"]),
    ("sampling-temp05", ["sampling-demo", "--temperature", "0.5", "--top-p", "1"]),
    ("sampling-temp15", ["sampling-demo", "--temperature", "1.5", "--top-p", "1"]),
    ("sampling-topk", ["sampling-demo", "--temperature", "1", "--top-k", "2", "--top-p", "1"]),
    ("sampling-topp", ["sampling-demo", "--temperature", "1", "--top-p", "0.8"]),
    ("sampling-minp", ["sampling-demo", "--temperature", "1", "--top-p", "1", "--min-p", "0.6"]),
    ("sampling-penalties", ["sampling-demo", "--repetition-penalty", "1.2", "--presence-penalty", "0.5", "--frequency-penalty", "0.25"]),
    ("sampling-seed8", ["sampling-demo", "--seed", "8"]),
    ("sampling-eos", ["sampling-demo", "--temperature", "0", "--eos-token-id", "2"]),
    ("sampling-stop-id", ["sampling-demo", "--temperature", "0", "--stop-token-id", "2"]),
    ("sampling-stop-string", ["sampling-demo", "--temperature", "0", "--stop", "推理推理"]),
    ("sampling-zero", ["sampling-demo", "--max-new-tokens", "0"]),
]


def source_fingerprint():
    paths = sorted([*ROOT.glob('zero2vllm/**/*.py'), *ROOT.glob('tests/*.py'),
                    *ROOT.glob('scripts/*.py'), ROOT / 'examples/part1/request-events.json'])
    sha = hashlib.sha256()
    for path in paths:
        sha.update(path.relative_to(ROOT).as_posix().encode('utf-8') + b'\0')
        sha.update(path.read_bytes().replace(b'\r\n', b'\n') + b'\0')
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT, capture_output=True, text=True, check=True).stdout)
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {'git_commit': commit, 'working_tree_dirty': dirty, 'source_sha256': sha.hexdigest(),
            'recipe': 'Sorted package/tests/scripts Python files plus event fixture; relative UTF-8 path + NUL + LF-normalized file bytes + NUL.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('runs/part1'), help='Directory for generated reports; relative paths use repository root')
    parser.add_argument('--skip-tests', action='store_true', help='Run all experiments and document checks but omit unit tests (e.g. already run by CI)')
    args = parser.parse_args()
    directory = (ROOT / args.output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
    summary = {'schema_version': 1, 'record_type': 'measured_reference',
               'created_at_utc': datetime.now(timezone.utc).isoformat(),
               'python': platform.python_version(), 'os': platform.system(),
               'source': source_fingerprint(), 'cases': [], 'tests': None, 'course_check': None,
               'passed': False, 'gpu_status': 'not_run',
               'scope': 'Executed CPU reference/analytical/simulation cases. Individual reports retain provenance; no hardware speed measurement.'}

    def save():
        (directory / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')

    for name, options in CASES:
        path = directory / f'{name}.json'
        argv = [sys.executable, '-m', 'zero2vllm', *options, '--output', str(path)]
        process = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True, encoding='utf-8')
        passed, verification = False, None
        if process.returncode == 0:
            data = json.loads(path.read_text(encoding='utf-8'))
            verification = data.get('verification')
            passed = bool(verification and verification.get('passed'))
        summary['cases'].append({'name': name, 'arguments': options, 'output': path.name,
                                 'exit_code': process.returncode, 'passed': passed,
                                 'verification': verification, 'error': process.stderr.strip() or None})
        print(f'{"PASS" if passed else "FAIL"} {name}', flush=True)
        save()
        if not passed:
            return 1
    if not args.skip_tests:
        process = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                                 cwd=ROOT, env=env, capture_output=True, text=True, encoding='utf-8')
        (directory / 'tests.txt').write_text(process.stdout + process.stderr, encoding='utf-8', newline='\n')
        summary['tests'] = {'exit_code': process.returncode, 'passed': process.returncode == 0, 'output': 'tests.txt'}
        print(f'{"PASS" if process.returncode == 0 else "FAIL"} unit tests', flush=True)
        save()
        if process.returncode:
            return 1
    process = subprocess.run([sys.executable, 'scripts/course_docs.py', '--check'], cwd=ROOT, env=env,
                             capture_output=True, text=True, encoding='utf-8')
    (directory / 'course-check.txt').write_text(process.stdout + process.stderr, encoding='utf-8', newline='\n')
    summary['course_check'] = {'exit_code': process.returncode, 'passed': process.returncode == 0, 'output': 'course-check.txt'}
    summary['passed'] = process.returncode == 0
    summary['tests_skipped'] = args.skip_tests
    save()
    print(f'{"PASS" if summary["passed"] else "FAIL"} course check; summary: {directory / "summary.json"}', flush=True)
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
