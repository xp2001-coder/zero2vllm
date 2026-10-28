"""Analyze declared equal-budget PD traces; offline and standard-library only."""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.common import ROOT
from zero2vllm.reference.disaggregation import analyze

LESSON = ROOT / 'docs/chapters/part-13-disaggregation/01-prefill-decode-disaggregation'
DEFAULT_INPUT = LESSON / 'data/comparison.json'
DEFAULT_OUTPUT = ROOT / '.local/runs/part-13-disaggregation/01-pd-boundaries/comparison.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--ttft-ms', type=float)
    parser.add_argument('--tpot-ms', type=float)
    parser.add_argument('--max-itl-ms', type=float)
    args = parser.parse_args()
    try:
        if args.input.resolve() == args.output.resolve():
            raise ValueError('input and output must be different files')
        raw = args.input.read_bytes()
        document = json.loads(raw)
        override = {key: value for key, value in vars(args).items()
                    if key in ('ttft_ms', 'tpot_ms', 'max_itl_ms') and value is not None}
        report = analyze(document, override)
        report['input_sha256'] = hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest()
        report['input_sha256_recipe'] = 'Input bytes with CRLF normalized to LF; no other changes.'
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                               encoding='utf-8', newline='\n')
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f'Invalid comparison: {exc}\n')
    print(f"Source: {report['source_kind']}; observation: {report['duration_seconds']:g}s")
    print('name                          complete qualified goodput(req/s)')
    for run in report['runs']:
        print(f"{run['name']:<30} {run['completed']:>6} {run['qualified']:>9} {run['goodput_rps']:>14.3f}")
    print(f'Report: {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
