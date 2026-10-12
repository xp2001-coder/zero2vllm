"""Inspect a fixed four-candidate distribution before running language generation."""
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from zero2vllm.reference.sampling import SamplingConfig, distribution, draw


class FixedUniform:
    """A specified draw for arithmetic, not a random-number generator."""
    def __init__(self, value):
        self.value = value

    def random(self):
        return self.value


def main():
    base = [0.4, 0.3, 0.2, 0.1]
    logits = [math.log(p) for p in base]
    cases = [
        ('base', dict(temperature=1, top_p=1), base),
        ('temperature_0.5', dict(temperature=0.5, top_p=1), [16/30, 9/30, 4/30, 1/30]),
        ('top_k_2', dict(temperature=1, top_p=1, top_k=2), [4/7, 3/7, 0, 0]),
        ('top_p_0.8', dict(temperature=1, top_p=0.8), [4/9, 3/9, 2/9, 0]),
        ('min_p_0.6', dict(temperature=1, top_p=1, min_p=0.6), [4/7, 3/7, 0, 0]),
        ('k_then_p', dict(temperature=1, top_k=2, top_p=0.5), [1, 0, 0, 0]),
    ]
    for name, config, expected in cases:
        result = distribution(logits, SamplingConfig(**config))
        assert all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(result['probabilities'], expected))
        print(name, 'kept=', result['kept_token_ids'], 'p=', [round(p, 6) for p in result['probabilities']])
    for value, expected in ((0.65, 1), (0.75, 2)):
        token = draw(base, FixedUniform(value))
        assert token == expected
        print(f'u={value} -> ID {token}')
    print('PASS: temperature, filter order, renormalization, and interval sampling.')


if __name__ == '__main__':
    main()
