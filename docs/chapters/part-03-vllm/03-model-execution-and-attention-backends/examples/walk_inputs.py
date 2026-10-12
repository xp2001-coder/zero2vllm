"""Print request-local coordinates and selected CPU results for lesson 3.3."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[5]))
from scripts.experiments.lesson0303 import requests
from zero2vllm.reference.execution import pack_inputs

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--order', choices=('BA','AB'), default='BA')
parser.add_argument('--pad-to', type=int)
parser.add_argument('--tensors', action='store_true')
args=parser.parse_args()
if args.tensors:
    from zero2vllm.modeling.packed_attention import AttentionFixture
    r=AttentionFixture().execute(requests(),list(args.order),pad_to=args.pad_to)
    data=asdict(r['batch'])
    data.update(shapes=r['shapes'],errors=r['errors'],candidate_ids=r['candidate_ids'],returned_ids=r['returned_ids'])
    print('CPU 单层 Attention 夹具；非 vLLM / 预训练模型 / GPU 执行。')
else:
    data=asdict(pack_inputs(requests(),list(args.order),pad_to=args.pad_to))
    print('仅输入打包；没有执行模型或采样。')
print(json.dumps(data,ensure_ascii=False,indent=2))
