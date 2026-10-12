"""Separate numerical tolerance, greedy stability, and rounding order."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from zero2vllm.reference.sampling import compare_logits


def main():
    cases = [('same_token_wrong_scores', [2, 1], [20, 1], False, True),
             ('close_scores_different_token', [1, 1.0000001], [1.0000002, 1], True, False),
             ('gap_exceeds_twice_error', [2, 1], [1.9, 1.1], False, True)]
    for name, reference, candidate, close, same in cases:
        result = compare_logits(reference, candidate)
        assert result['all_within_tolerance'] == close and result['same_argmax'] == same
        print(f"{name}: within_tolerance={close}, same_argmax={same}, max_error={result['max_abs_error']:.8g}")
    a, b = (1e16 + (-1e16)) + 1.0, 1e16 + ((-1e16) + 1.0)
    assert a == 1 and b == 0
    print(f'addition order: left={a:g}, right={b:g}')
    print('PASS: same token and close logits are distinct checks.')


if __name__ == '__main__':
    main()
