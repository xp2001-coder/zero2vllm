"""An already classified feature meets absent or explicit V1 intent."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[5]))
from zero2vllm.reference.config_resolution import engine_choice
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--intent',choices=('auto','0','1'),default='auto')
p.add_argument('--feature',choices=('supported','experimental','unsupported'),default='supported')
a=p.parse_args()
try:r=engine_choice(None if a.intent=='auto' else a.intent,a.feature)
except ValueError as error:p.error(str(error))
print(json.dumps(r,ensure_ascii=False,indent=2))
print('Feature classification is supplied, not detected. No model/platform support is established.')
