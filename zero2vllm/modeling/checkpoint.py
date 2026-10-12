"""Explicit HF safetensors -> teaching Llama parameter mapping, CPU only."""
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

from zero2vllm.modeling.llama import LlamaConfig, LlamaDecoder


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate JSON key: {key}')
            result[key] = value
        return result
    result = json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique)
    if not isinstance(result, dict):
        raise ValueError(f'Expected a JSON object: {Path(path).name}')
    return result


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def weight_map(config):
    mapping = {'model.embed_tokens.weight': 'embedding.weight',
               'model.norm.weight': 'final_norm.weight', 'lm_head.weight': 'head.weight'}
    for i in range(config.num_hidden_layers):
        source, target = f'model.layers.{i}.', f'layers.{i}.'
        mapping[source+'input_layernorm.weight'] = target+'norm1.weight'
        mapping[source+'post_attention_layernorm.weight'] = target+'norm2.weight'
        for name in ('q', 'k', 'v', 'o'):
            mapping[source+f'self_attn.{name}_proj.weight'] = target+f'{name}.weight'
        for name in ('gate', 'up', 'down'):
            mapping[source+f'mlp.{name}_proj.weight'] = target+f'{name}.weight'
    return mapping


def checkpoint_files(directory):
    """Return tensor -> file, validating a single file or an indexed set of shards."""
    directory = Path(directory).resolve()
    single, index = directory/'model.safetensors', directory/'model.safetensors.index.json'
    if single.exists() and index.exists():
        raise ValueError('Ambiguous checkpoint: both single-file and sharded manifests exist.')
    if single.is_file():
        with safe_open(single, framework='pt', device='cpu') as source:
            return {name: single for name in source.keys()}
    if not index.is_file():
        raise ValueError('Expected model.safetensors or model.safetensors.index.json.')
    mapping = read_json(index).get('weight_map')
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError('Shard index must contain a nonempty weight_map.')
    resolved = {}
    for name, filename in mapping.items():
        if not isinstance(filename, str) or '/' in filename or '\\' in filename or ':' in filename or not filename.endswith('.safetensors'):
            raise ValueError(f'Invalid shard filename: {filename!r}')
        path = (directory/filename).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError(f'Missing shard or shard outside snapshot: {filename}')
        resolved[name] = path
    for path in sorted(set(resolved.values())):
        with safe_open(path, framework='pt', device='cpu') as source:
            actual = set(source.keys())
        expected = {name for name, location in resolved.items() if location == path}
        if actual != expected:
            raise ValueError(f'Shard index disagrees with {path.name}: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}')
    return resolved


def load_checkpoint(directory):
    """Fail closed: missing/extra keys, shape/dtype/finiteness and tied aliases.

    Shards and tensors are read sequentially. Never build a second full state_dict.
    A model is returned only when all checks pass; partial loads cannot be used.
    """
    directory = Path(directory)
    config = LlamaConfig.from_dict(read_json(directory/'config.json'))
    files = checkpoint_files(directory)
    mapping = weight_map(config)
    optional = {'lm_head.weight'} if config.tie_word_embeddings else set()
    missing, extra = set(mapping)-set(files)-optional, set(files)-set(mapping)
    if missing or extra:
        raise ValueError(f'Checkpoint keys: missing={sorted(missing)}, extra={sorted(extra)}')
    model = LlamaDecoder(config)
    targets = dict(model.named_parameters(remove_duplicate=False))
    entries = []
    saved_head = None
    with torch.no_grad():
        for path in sorted(set(files.values())):
            with safe_open(path, framework='pt', device='cpu') as source:
                for name in sorted(key for key, location in files.items() if location == path):
                    tensor = source.get_tensor(name)
                    parameter = targets[mapping[name]]
                    if tuple(tensor.shape) != tuple(parameter.shape):
                        raise ValueError(f'Shape mismatch for {name}: {tuple(tensor.shape)} != {tuple(parameter.shape)}')
                    if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                        raise ValueError(f'Unsupported dtype for {name}: {tensor.dtype}')
                    if not torch.isfinite(tensor).all().item():
                        raise ValueError(f'Non-finite checkpoint tensor: {name}')
                    if config.tie_word_embeddings and name == 'lm_head.weight':
                        saved_head = tensor.float().clone()
                    else:
                        parameter.copy_(tensor)
                    entries.append({'source': name, 'target': mapping[name], 'file': path.name,
                                    'shape': list(tensor.shape), 'stored_dtype': str(tensor.dtype),
                                    'compute_dtype': 'torch.float32'})
        if saved_head is not None and not torch.equal(saved_head, model.embedding.weight):
            raise ValueError('Tied lm_head and embedding weights disagree.')
    model.loaded = True
    report = {
        'tensor_count': len(entries), 'parameter_count': sum(p.numel() for p in model.parameters()),
        'tied_embeddings': config.tie_word_embeddings,
        'head_alias_reconstructed': config.tie_word_embeddings and 'lm_head.weight' not in files,
        'files': [{'file': path.name, 'bytes': path.stat().st_size, 'sha256': sha256(path)}
                  for path in sorted(set(files.values()))],
        'mapping': entries,
    }
    return model, report
