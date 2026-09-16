"""Inspect the lesson's adjacent-pair RoPE and its position convention.

Run from the repository root in the lesson's CPU PyTorch environment:
    python docs/chapters/part-02-engine/01-pytorch-model/examples/rope_math.py
"""
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

import torch

from zero2vllm.modeling.torch_decoder import apply_rope


def rotate_pairs(vector, position, base=10000.0):
    """Scalar calculation with an explicit absolute position, for comparison."""
    result = []
    for index in range(0, len(vector), 2):
        angle = position / base ** (index / len(vector))
        a, b = vector[index:index + 2]
        result.extend((a * math.cos(angle) - b * math.sin(angle),
                       a * math.sin(angle) + b * math.cos(angle)))
    return result


def repeated_positions(vectors, length):
    # One request; each head gets its own content vector at every position.
    heads = [[vector[:] for _ in range(length)] for vector in vectors]
    return torch.tensor([heads], dtype=torch.float64)


@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    base = 10000.0
    q_vectors = [[1.0, 0.0, 1.0, 0.0], [2.0, 1.0, -1.0, 2.0]]
    k_vectors = [[0.0, 1.0, 1.0, 1.0], [1.0, -2.0, 3.0, 0.5]]
    q = repeated_positions(q_vectors, length=3)
    original = q.clone()
    rotated = apply_rope(q, base)
    expected_heads = [[rotate_pairs(vector, pos, base) for pos in range(3)]
                      for vector in q_vectors]
    expected = torch.tensor([expected_heads], dtype=torch.float64)
    torch.testing.assert_close(rotated, expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(q, original, atol=0, rtol=0)
    torch.testing.assert_close(rotated[:, :, 0], q[:, :, 0], atol=0, rtol=0)
    torch.testing.assert_close(
        rotated.reshape(1, 2, 3, 2, 2).square().sum(-1),
        q.reshape(1, 2, 3, 2, 2).square().sum(-1),
        atol=1e-12, rtol=1e-12,
    )
    print('Input shape [B,H,T,Dh]:', tuple(q.shape))
    print('Head 0 input at each position: [1, 0, 1, 0]')
    print('Pair frequencies: [1.0, 0.01] radians per position')
    for pos, row in enumerate(rotated[0, 0].tolist()):
        print(f'position {pos}: ' + ' '.join(f'{v:.6f}' for v in row))
    print('Scalar rotation / pair norms / position zero / input unchanged: PASS')

    # Shift both positions together while keeping the content vectors fixed.
    # This checks the RoPE dot product, not the full decoder's shift behavior.
    q_long = apply_rope(repeated_positions(q_vectors, 6), base)
    k_long = apply_rope(repeated_positions(k_vectors, 6), base)
    original_dot = (q_long[:, :, 2] * k_long[:, :, 1]).sum(-1)
    shifted_dot = (q_long[:, :, 5] * k_long[:, :, 4]).sum(-1)
    relative_values = [sum(a * b for a, b in zip(qv, rotate_pairs(kv, -1, base)))
                       for qv, kv in zip(q_vectors, k_vectors)]
    relative_dot = torch.tensor([relative_values], dtype=torch.float64)
    torch.testing.assert_close(original_dot, shifted_dot, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(original_dot, relative_dot, atol=1e-12, rtol=1e-12)
    print(f'Head 0 dot, Q@2 and K@1: {original_dot[0, 0]:.6f}')
    print(f'Head 0 dot, Q@5 and K@4: {shifted_dot[0, 0]:.6f}')
    print('Joint position shift / relative rotation: PASS')

    # The current function derives positions from its input length. Passing
    # one new token makes it position 0, even if it belongs at position 2.
    reset_q = apply_rope(repeated_positions(q_vectors, 1), base)
    reset_dot = (reset_q[:, :, 0] * k_long[:, :, 1]).sum(-1)
    if torch.allclose(reset_dot, original_dot, atol=1e-12, rtol=1e-12):
        raise AssertionError('This example must expose the reset-position error.')
    print(f'Wrong: one-token Q resets to 0; dot with K@1: {reset_dot[0, 0]:.6f}')
    print('Different score detected: PASS (KV cache needs explicit positions)')


if __name__ == '__main__':
    main()
