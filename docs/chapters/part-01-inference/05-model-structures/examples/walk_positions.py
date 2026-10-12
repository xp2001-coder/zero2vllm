"""Track rotations, absolute positions, and visible key indices."""
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from zero2vllm.reference.positions import causal_attention, rope, structure_experiment


def main():
    dot = lambda a, b: sum(x*y for x, y in zip(a, b))
    rotated = rope([1.0, 0.0], 1)
    assert math.isclose(dot(rotated, rotated), 1.0)
    q, k = [1.0, 0.0], [1.0, 0.0]
    correct = dot(rope(q, 7), rope(k, 6))
    wrong = dot(rope(q, 0), rope(k, 6))
    assert math.isclose(correct, math.cos(1)) and not math.isclose(correct, wrong)
    print('rotate [1,0] at position 1:', [round(x, 6) for x in rotated])
    print(f'query position 7: score={correct:.6f}; wrongly reset to 0: {wrong:.6f}')
    report = structure_experiment(8, 3)
    for position, visible in enumerate(report['sliding_window']['visible_positions']):
        print(f'query {position}: keys {visible}')
    assert report['dense_attention']['score_pairs'] == 36
    assert report['sliding_window']['score_pairs'] == 21
    q, k, v = [[0.0]]*2, [[0.0]]*2, [[1.0], [3.0]]
    alibi = causal_attention(q, k, v, alibi_slope=math.log(2))
    assert math.isclose(alibi['outputs'][1][0], 7/3)
    print('pairs: full=36, window3=21; ALiBi second output=7/3')
    print('PASS: absolute positions, visibility, and distance bias.')


if __name__ == '__main__':
    main()
