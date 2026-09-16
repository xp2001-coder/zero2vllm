"""Run from the repository root: python docs/chapters/part-02-engine/01-pytorch-model/examples/run.py."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

import torch
from zero2vllm.reference.decoder import ScalarDecoder
from zero2vllm.modeling.torch_decoder import TorchDecoder


def main():
    torch.set_num_threads(1)
    reference = ScalarDecoder(seed=7)
    model = TorchDecoder(reference, dtype=torch.float64)
    tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.inference_mode():
        logits, trace = model(tokens, return_trace=True)
    print('logits shape:', tuple(logits.shape))
    print('parameters:', sum(p.numel() for p in model.parameters()))
    for name, value in trace.items():
        print(name, tuple(value.shape))
    print('generated IDs:', model.generate(tokens[0].tolist(), 4))
    print('These are untrained token IDs, not natural language.')


if __name__ == '__main__':
    main()
