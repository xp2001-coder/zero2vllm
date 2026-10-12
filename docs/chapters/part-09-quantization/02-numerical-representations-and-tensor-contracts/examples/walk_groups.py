"""Print logical/physical tensor shapes, parameters, codes and dequantization."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[5]))
from scripts.experiments.lesson0902 import load_fixture
from zero2vllm.quantization.tensor_contract import quantize_symmetric,quantize_asymmetric,dequantize

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--axis',type=int,choices=(0,1),default=0)
    parser.add_argument('--asymmetric',action='store_true')
    parser.add_argument('--pad',action='store_true')
    args=parser.parse_args()
    fn=quantize_asymmetric if args.asymmetric else quantize_symmetric
    t=fn(load_fixture()['w'],axis=args.axis,pad_to_group=args.pad)
    print(json.dumps(dict(contract=t.to_dict(),dq=dequantize(t,include_padding=args.pad)),ensure_ascii=False,indent=2))
