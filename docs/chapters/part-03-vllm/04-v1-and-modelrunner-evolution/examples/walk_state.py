"""Show stable rows, staged updates or copy ownership. CPU standard library."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[5]))
from scripts.experiments.lesson0304 import experiments
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--case',choices=('mapping','staged','copy','sampling'),default='mapping')
a=p.parse_args()
groups={'mapping':('stable_rows','scatter_progress','row_reuse','preempt_and_resume'),
        'staged':('staged_writes','overlap_guard'),'copy':('copy_ownership',),
        'sampling':('expanded_sampling_map','sampling_algebra')}
for c in experiments():
 if c['name'] in groups[a.case]:print(json.dumps(c,ensure_ascii=False,indent=2))
print('CPU semantics only; no upstream runner or GPU execution.')
