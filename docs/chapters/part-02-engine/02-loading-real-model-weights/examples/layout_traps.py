"""Three shape-preserving mistakes: transpose, rotary pairs, GQA head order."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
import torch
from zero2vllm.modeling.llama import split_half_rope
from zero2vllm.modeling.torch_decoder import apply_rope


def main():
    x = torch.tensor([[1., 2.]])
    weight = torch.tensor([[1., 2.], [3., 4.]])
    correct, wrong = x @ weight.T, x @ weight
    print('Linear stored [out,in]:', correct.tolist(), '; transposed on load:', wrong.tolist())
    assert correct.tolist() == [[5., 11.]] and wrong.tolist() == [[7., 10.]]

    heads = torch.tensor([[[[1., 2., 3., 4.], [1., 2., 3., 4.]]]])
    half, adjacent = split_half_rope(heads, 10000.), apply_rope(heads, 10000.)
    print('RoPE at p=1, split half:', half[0, 0, 1].tolist())
    print('RoPE at p=1, adjacent:  ', adjacent[0, 0, 1].tolist())
    permutation = [0, 2, 1, 3]
    converted = apply_rope(heads[..., permutation], 10000.)[..., permutation]
    torch.testing.assert_close(half, converted)
    assert not torch.allclose(half, adjacent)

    kv = torch.arange(3).reshape(1, 3, 1, 1)
    grouped, tiled = kv.repeat_interleave(3, dim=1), kv.repeat(1, 3, 1, 1)
    print('GQA KV head for Q0..Q8:', grouped.flatten().tolist())
    print('Wrong repeated order: ', tiled.flatten().tolist())
    assert grouped.flatten().tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert not torch.equal(grouped, tiled)
    print('PASS: all three mistakes preserve shape but change values or meaning.')


if __name__ == '__main__':
    main()
