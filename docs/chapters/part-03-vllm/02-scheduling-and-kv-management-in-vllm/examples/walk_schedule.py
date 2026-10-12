"""Print a count-only mixed scheduling trace; run from the repository root."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from scripts.experiments.lesson0302 import mixed_rounds

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--token-budget', type=int, default=4)
parser.add_argument('--threshold', type=int, default=0)
parser.add_argument('--rounds', type=int, default=3)
args = parser.parse_args()
rows = mixed_rounds(args.token_budget, args.threshold, args.rounds)
print('教学计数回放；每次追平后假设产生 1 个输出 token，不调用模型。')
for i, row in enumerate(rows, 1):
    print(f'第 {i} 轮：')
    for item in row['scheduled']:
        rid = item['request']
        after = row['after_feedback']['requests'][rid]
        print(f"  {rid}: N={item['known_before']}, C={item['computed_before']}, 本轮={item['scheduled']}, 新块={item['new_block_ids']}, 反馈后 N={after['known']}, C={after['computed']}")
    print(f"  剩余 token 预算={row['unused_token_budget']}；等待={row['after_feedback']['waiting']}")
