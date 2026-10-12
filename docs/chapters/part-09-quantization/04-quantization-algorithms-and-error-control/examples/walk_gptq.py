import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[5]))

from scripts.experiments.lesson0904 import load_fixture
from zero2vllm.quantization.algorithms import column_grids,gptq_reference

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Trace GPTQ core on a 3x2 synthetic weight')
    parser.add_argument('--block-size',type=int,default=1)
    parser.add_argument('--act-order',action='store_true')
    args=parser.parse_args();f=load_fixture()['main']
    try:
        result=gptq_reference(f['calibration'],f['weight'],column_grids(f['weight']),block_size=args.block_size,act_order=args.act_order)
    except ValueError as error:parser.error(str(error))
    print(json.dumps(result,indent=2))
