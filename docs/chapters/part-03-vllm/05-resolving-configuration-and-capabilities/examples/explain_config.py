"""Trace selected v0.8.5 rules using synthetic inputs, not a vLLM launch."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[5]))
from zero2vllm.reference.config_resolution import resolve_v1
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'data/demo-config.json',help='Course JSON mapping; NOT vLLM YAML')
p.add_argument('--usage',choices=('server','offline'),default='server')
p.add_argument('--memory-gib',type=float,default=40)
p.add_argument('--model-limit',type=int,default=4096)
p.add_argument('--max-num-batched-tokens',type=int)
p.add_argument('--max-num-seqs',type=int)
p.add_argument('--prefix',choices=('true','false','inherit'),default='inherit')
p.add_argument('--enforce-eager',action='store_true')
a=p.parse_args()
try:
 config=json.loads(a.config.read_text(encoding='utf8'))
 if not isinstance(config,dict):raise ValueError('configuration must be a JSON object')
 cli={k:getattr(a,k) for k in ('max_num_batched_tokens','max_num_seqs') if getattr(a,k) is not None}
 if a.prefix!='inherit':cli['enable_prefix_caching']=a.prefix=='true'
 if a.enforce_eager:cli['enforce_eager']=True
 result=resolve_v1(config,cli,usage=a.usage,memory_gib=a.memory_gib,model_limit=a.model_limit)
except (OSError,ValueError) as error:p.error(str(error))
print(json.dumps(result,ensure_ascii=False,indent=2))
