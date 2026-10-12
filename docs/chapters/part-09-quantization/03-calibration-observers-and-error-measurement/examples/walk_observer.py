"""Compare global extrema with EMA and show the calibration/freeze boundary."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[5]))
from zero2vllm.quantization.calibration import RangeObserver

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('minmax','ema'),default='ema')
    parser.add_argument('--reverse',action='store_true')
    args=parser.parse_args()
    batches=[[[-1,1]],[[-4,8]]]
    if args.reverse:batches.reverse()
    observer=RangeObserver(mode=args.mode,alpha=.25)
    for batch in batches:
        observer.update(batch);print(json.dumps(observer.state(),indent=2))
    params=observer.freeze(scheme='asymmetric')
    print('Frozen parameters:',json.dumps(asdict(params)))
    try:observer.update([[999]])
    except ValueError as error:print('Expected rejection:',error)
