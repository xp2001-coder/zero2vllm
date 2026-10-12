"""Reproduce the lesson's small latent, recurrent-attention, and SSM identities."""
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from zero2vllm.reference.attention import linear
from zero2vllm.reference.positions import linear_attention, ssm_scan


def main():
    c, wk, q = [1, 2], [[1, 0, 1], [0, 1, 1]], [2, 3, 1]
    k = linear(c, wk)
    absorbed = [sum(a*b for a, b in zip(q, row)) for row in wk]
    expanded_score = sum(a*b for a, b in zip(q, k))
    latent_score = sum(a*b for a, b in zip(absorbed, c))
    assert k == [1, 2, 3] and absorbed == [3, 4]
    assert expanded_score == latent_score == 11
    print(f'latent: expanded K={k}; transformed Q={absorbed}; both scores=11')
    q, k, v = [[1.0], [2.0]], [[0.0], [1.0]], [[1.0], [3.0]]
    explicit = linear_attention(q, k, v, recurrent=False)
    recurrent = linear_attention(q, k, v)
    state = norm = 0.0
    for i in range(2):
        pk, pq = max(k[i][0], 0)+1, max(q[i][0], 0)+1
        state += pk*v[i][0]
        norm += pk
        output = pq*state/(pq*norm)
        assert math.isclose(output, explicit[i][0]) and math.isclose(output, recurrent[i][0])
        print(f'linear step {i}: S={state:g}, z={norm:g}, output={output:.6f}')
    assert math.isclose(recurrent[-1][0], 7/3)
    result = ssm_scan([1.0, 2.0, 3.0])
    assert result['states'] == [1, 2.5, 4.25]
    assert result['outputs'] == result['convolution_reference'] == [2, 5, 8.5]
    print('SSM states:', result['states'], 'outputs:', result['outputs'])
    print('PASS: three separate identities; no cross-model equivalence claimed.')


if __name__ == '__main__':
    main()
