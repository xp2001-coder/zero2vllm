"""Inspect a downloaded local checkpoint without allocating the decoder."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from safetensors import safe_open
from transformers import AutoTokenizer
from zero2vllm.modeling.checkpoint import checkpoint_files, read_json
from zero2vllm.modeling.llama import LlamaConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, default=Path('.local/models/smollm2-135m'))
    args = parser.parse_args()
    config = LlamaConfig.from_dict(read_json(args.snapshot/'config.json'))
    files = checkpoint_files(args.snapshot)
    print(f'Layers={config.num_hidden_layers}, hidden={config.hidden_size}, head_dim={config.head_dim}')
    print(f'Q heads={config.num_attention_heads}, KV heads={config.num_key_value_heads}, tensors={len(files)}')
    for name in ('model.embed_tokens.weight', 'model.layers.0.self_attn.q_proj.weight',
                 'model.layers.0.self_attn.k_proj.weight', 'model.layers.0.mlp.down_proj.weight', 'model.norm.weight'):
        with safe_open(files[name], framework='pt', device='cpu') as source:
            view = source.get_slice(name)
            print(f'{name}: shape={view.get_shape()}, dtype={view.get_dtype()}')
    print('lm_head.weight stored separately:', 'lm_head.weight' in files)
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True, trust_remote_code=False)
    for prompt in ('The capital of France is', 'Once upon a time'):
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        print(repr(prompt), '->', ids, '->', repr(tokenizer.decode(ids)))
    print('Special IDs:', {'bos': tokenizer.bos_token_id, 'eos': tokenizer.eos_token_id, 'pad': tokenizer.pad_token_id})


if __name__ == '__main__':
    main()
