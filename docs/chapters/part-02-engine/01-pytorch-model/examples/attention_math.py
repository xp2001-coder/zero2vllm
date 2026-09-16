"""Reproduce the single-head numerical example in lesson 2.1.

These small matrices are chosen for hand calculation. Q and K are treated as
the inputs to scaled dot-product attention, after any positional transform.
They are not activations taken from the lesson's randomly initialized model.
"""
import math

import torch


def main():
    torch.set_printoptions(precision=4, sci_mode=False)
    dtype = torch.float64
    q = torch.tensor([[2., 0., 0., 0.]] * 3, dtype=dtype)
    k = torch.tensor([[1., 0., 0., 0.],
                      [0., 0., 0., 0.],
                      [2., 0., 0., 0.]], dtype=dtype)
    v = torch.tensor([[1., 0., 0., 0.],
                      [0., 2., 0., 0.],
                      [0., 0., 3., 0.]], dtype=dtype)

    with torch.inference_mode():
        scores = q @ k.T / math.sqrt(q.shape[-1])
        future = torch.ones(3, 3, dtype=torch.bool).triu(1)
        masked_scores = scores.masked_fill(future, float('-inf'))
        probs = masked_scores.softmax(dim=-1)
        output = probs @ v
        for name, value in [('scores', scores), ('future (True = blocked)', future),
                            ('masked_scores', masked_scores), ('probs', probs),
                            ('output', output)]:
            print(name)
            print(value)

        # Every row distributes unit mass only among its visible positions.
        torch.testing.assert_close(probs.sum(dim=-1), torch.ones(3, dtype=dtype))
        assert torch.count_nonzero(probs[future]).item() == 0
        expected_last = (math.exp(1) * v[0] + v[1] + math.exp(2) * v[2])
        expected_last /= math.exp(1) + 1 + math.exp(2)
        torch.testing.assert_close(output[-1], expected_last)

        # Adding a 0/1 visibility mask is a different computation: future
        # positions retain finite scores, so softmax assigns them weight.
        visible = (~future).to(dtype)
        wrong_probs = (scores + visible).softmax(dim=-1)
        print('wrong mask: probability on future positions in row 0')
        print(wrong_probs[0, 1:].sum().item())


if __name__ == '__main__':
    main()
