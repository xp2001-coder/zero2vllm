"""Show nonzero zero-points, DQ and requantization with explicit values."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from scripts.experiments.lesson0901 import affine_experiment


if __name__ == '__main__':
    report = affine_experiment()
    print('x -> Q1(s=0.5,z=3) -> DQ1 -> Q2(s=1,z=2) -> DQ2')
    for row in zip(report['input'][0], report['q1'][0], report['dq1'][0], report['q2'][0], report['dq2'][0]):
        print(' -> '.join(str(v) for v in row))
    print(json.dumps(report,ensure_ascii=False,indent=2))
