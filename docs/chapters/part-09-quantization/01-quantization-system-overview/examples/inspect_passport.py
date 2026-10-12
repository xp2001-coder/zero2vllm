"""Validate a passport's structure and print only known/declared fields."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from zero2vllm.quantization import QuantizedModelPassport


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('passport',type=Path)
    args=parser.parse_args()
    try:
        p=QuantizedModelPassport.from_dict(json.loads(args.passport.read_text(encoding='utf-8')))
    except (ValueError,OSError) as error:
        parser.error(str(error))
    print(p.experiment_id, p.scope)
    print('statuses:',p.status_counts())
    for section,fields in p.to_dict()['sections'].items():
        for name,entry in fields.items():
            if entry['status'] not in ('unknown','not_applicable'):
                print(f'{section}.{name}: {entry["status"]}; expected={entry["expected"]}; observed={entry["observed"]}')
    print('A well-formed citation is not proof; open the evidence and check its scope.')


if __name__=='__main__':
    main()
