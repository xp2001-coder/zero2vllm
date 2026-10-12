"""Inspect a bounded graph-mode decision; never capture a CUDA graph."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[5]))
from zero2vllm.reference.runner_state import graph_dispatch
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--mode',choices=('NONE','FULL','PIECEWISE','FULL_DECODE_ONLY','FULL_AND_PIECEWISE'),default='FULL_AND_PIECEWISE')
p.add_argument('--query-lengths',type=int,nargs='+',default=[1,3])
p.add_argument('--sizes',type=int,nargs='*',default=[1,2,4,8])
p.add_argument('--captured',type=int,nargs='*',default=[4])
a=p.parse_args()
try:r=graph_dispatch(a.mode,a.query_lengths,a.sizes,a.captured)
except ValueError as e:p.error(str(e))
print(json.dumps(r,ensure_ascii=False,indent=2))
print('Dispatch policy illustration only; piecewise wrapper readiness and GPU execution are outside scope.')
