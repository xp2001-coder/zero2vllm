"""Run the lesson's three-token example through the existing decoder.

Run from the repository root with the lesson's CPU PyTorch environment:
    python docs/chapters/part-02-engine/01-pytorch-model/examples/walk_forward.py
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

import torch

from zero2vllm.modeling.torch_decoder import TorchDecoder
from zero2vllm.reference.decoder import ScalarDecoder


def show(name, value):
    print(f'{name:28s} {tuple(value.shape)}')


def main():
    torch.set_num_threads(1)
    reference = ScalarDecoder(seed=7)
    model = TorchDecoder(reference, dtype=torch.float64)
    model.eval()
    tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)
    print('Input IDs:', tokens.tolist())
    print('CPU / float64 / seed=7 / untrained weights')

    with torch.inference_mode():
        # Follow the top-level path using the model's existing modules.
        # Each block also records intermediate tensors for inspection.
        show('tokens', tokens)
        x = model.embedding(tokens)
        show('embedding', x)
        trace = {}
        for index, block in enumerate(model.layers):
            prefix = f'layers.{index}.'
            x = block(x, trace, prefix)
            for stage in ('norm1', 'context', 'attention_residual',
                          'norm2', 'gate', 'up', 'swiglu', 'down', 'output'):
                show(prefix + stage, trace[prefix + stage])

        x = model.final_norm(x)
        show('final_norm', x)
        logits = model.head(x)
        show('logits', logits)

        # This checks that our walkthrough follows the real forward path.
        # Independent numerical correctness is checked by the lesson's
        # scalar-reference experiment; reusing modules alone cannot prove it.
        torch.testing.assert_close(logits, model(tokens), atol=1e-12, rtol=1e-10)
        print('Walkthrough matches model(tokens): PASS')

        last_logits = logits[:, -1, :]
        show('last_logits', last_logits)
        next_ids = last_logits.argmax(dim=-1)
        print('Greedy ID at each position:', logits.argmax(dim=-1).tolist())
        print('Next ID after the full prompt:', next_ids.tolist())
        print('Prompt plus ONE new ID:', tokens[0].tolist() + [int(next_ids[0])])


if __name__ == '__main__':
    main()
