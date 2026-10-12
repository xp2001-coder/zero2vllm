"""Print every Q/DQ step for the chapter's fixed small Linear."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from zero2vllm.reference.quantization_overview import linear_qdq


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scale',type=float,help='Override amax/7 for a controlled clipping experiment')
    args=parser.parse_args()
    fixture=json.loads((Path(__file__).resolve().parents[1]/'data/linear.json').read_text(encoding='utf-8'))
    try:
        report=linear_qdq(fixture['x'],fixture['w'],scale=args.scale)
    except ValueError as error:
        parser.error(str(error))
    qdq=report['qdq']
    print('k n | FP32 input | / scale | q | restored | error')
    for k,row in enumerate(qdq['fp32_input']):
        for n,w in enumerate(row):
            dq=qdq['dequantized'][k][n]
            print(k,n,'|',w,'|',w/qdq['scale'],'|',qdq['q'][k][n],'|',dq,'|',w-dq)
    print(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
