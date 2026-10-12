"""Show clipping search and three hand-computed ranking reversals."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[5]))
from scripts.experiments.lesson0903 import clipping_experiment,objective_examples

if __name__=='__main__':
    print(json.dumps(dict(clipping=clipping_experiment(),objectives=objective_examples()),ensure_ascii=False,indent=2))
