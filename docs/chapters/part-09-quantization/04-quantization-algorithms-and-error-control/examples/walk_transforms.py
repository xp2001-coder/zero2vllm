import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[5]))

from scripts.experiments.lesson0904 import run_experiments

if __name__=='__main__':
    argparse.ArgumentParser(description='Inspect scaling candidates and rotation identity').parse_args()
    report=run_experiments()
    print(json.dumps(dict(scaling=report['scaling_toy'],rotation=report['rotation']),indent=2))
