"""Inspect requested versus effective calibration counts and token masks."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[5]))
from scripts.experiments.lesson0903 import load_fixture,make_protocol

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain',choices=('general','code'),default='general')
    parser.add_argument('--samples',type=int,default=4)
    parser.add_argument('--token-budget',type=int)
    args=parser.parse_args()
    try:
        p=make_protocol(load_fixture(),split='calibration',domain=args.domain,sample_count=args.samples,
            max_length=8,seed=0,token_budget=args.token_budget)
    except ValueError as e:parser.error(str(e))
    print(json.dumps(p,ensure_ascii=False,indent=2))
