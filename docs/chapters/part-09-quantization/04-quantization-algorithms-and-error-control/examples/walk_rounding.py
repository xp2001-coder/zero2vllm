import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[5]))

from scripts.experiments.lesson0904 import load_fixture
from zero2vllm.quantization.algorithms import optimize_rounding
from zero2vllm.quantization.calibration import AffineParameters

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Inspect four fixed-grid rounding choices')
    parser.add_argument('--max-variables',type=int,default=16)
    args=parser.parse_args();f=load_fixture()['rounding']
    try:
        result=optimize_rounding(f['calibration'],f['weight'],[AffineParameters(1,0,-7,7,'symmetric')],max_variables=args.max_variables)
    except ValueError as error:parser.error(str(error))
    print(json.dumps(result,indent=2))
