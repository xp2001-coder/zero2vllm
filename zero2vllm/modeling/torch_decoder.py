"""Readable, eager PyTorch decoder for lesson 2.1; CPU float64/float32 only.

No fused attention, KV cache, padding, checkpoint loading or platform claims.
Each Linear weight is explicitly transposed from the scalar oracle's [in,out].
"""
from dataclasses import asdict
import hashlib
import json
import math
import platform

import torch
from torch import nn
from torch.nn import functional as F

from zero2vllm.reference.decoder import DecoderConfig, ScalarDecoder, validate_generation


class RMSNorm(nn.Module):
    def __init__(self, width, eps, dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width, dtype=dtype))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps) * self.weight


def apply_rope(x, base):
    # [B,H,T,Dh]; adjacent pairs share a rotation frequency.
    dim = x.shape[-1]
    positions = torch.arange(x.shape[-2], dtype=x.dtype, device=x.device)
    inv_freq = base ** (-torch.arange(0, dim, 2, dtype=x.dtype, device=x.device) / dim)
    angle = positions[:, None] * inv_freq[None, :]
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even*angle.cos()-odd*angle.sin(),
                        even*angle.sin()+odd*angle.cos()), dim=-1).flatten(-2)


class DecoderBlock(nn.Module):
    def __init__(self, config, dtype):
        super().__init__()
        self.config = config
        self.norm1 = RMSNorm(config.width, config.eps, dtype)
        self.norm2 = RMSNorm(config.width, config.eps, dtype)
        for name in ('q', 'k', 'v', 'o'):
            setattr(self, name, nn.Linear(config.width, config.width, bias=False, dtype=dtype))
        self.gate = nn.Linear(config.width, config.hidden, bias=False, dtype=dtype)
        self.up = nn.Linear(config.width, config.hidden, bias=False, dtype=dtype)
        self.down = nn.Linear(config.hidden, config.width, bias=False, dtype=dtype)

    def forward(self, x, trace, prefix):
        batch, length, width = x.shape
        heads, dim = self.config.heads, width // self.config.heads

        def split(value):
            return value.reshape(batch, length, heads, dim).transpose(1, 2)

        def merge(value):
            return value.transpose(1, 2).contiguous().reshape(batch, length, width)

        n = self.norm1(x)
        trace[prefix+'norm1'] = n
        q, k, v = self.q(n), self.k(n), self.v(n)
        trace.update({prefix+'q': q, prefix+'k': k, prefix+'v': v})
        q, k, v = split(q), split(k), split(v)
        q, k = apply_rope(q, self.config.rope_base), apply_rope(k, self.config.rope_base)
        trace.update({prefix+'q_rope': merge(q), prefix+'k_rope': merge(k)})
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dim)
        future = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
        probs = scores.masked_fill(future, float('-inf')).softmax(dim=-1)
        context = merge(probs @ v)
        trace[prefix+'context'] = context
        x = x + self.o(context)
        trace[prefix+'attention_residual'] = x
        n = self.norm2(x)
        trace[prefix+'norm2'] = n
        gate, up = self.gate(n), self.up(n)
        trace.update({prefix+'gate': gate, prefix+'up': up})
        gated = F.silu(gate) * up
        trace[prefix+'swiglu'] = gated
        down = self.down(gated)
        trace[prefix+'down'] = down
        x = x + down
        trace[prefix+'output'] = x
        return x


class TorchDecoder(nn.Module):
    def __init__(self, reference, dtype=torch.float64):
        super().__init__()
        if dtype not in (torch.float64, torch.float32):
            raise ValueError('Lesson 2.1 supports float64 and float32.')
        c = self.config = reference.config
        # Do not consume a caller's global torch RNG for weights we overwrite.
        with torch.random.fork_rng(devices=[]):
            self.embedding = nn.Embedding(c.vocab_size, c.width, dtype=dtype)
            self.layers = nn.ModuleList([DecoderBlock(c, dtype) for _ in range(c.layers)])
            self.final_norm = RMSNorm(c.width, c.eps, dtype)
            self.head = nn.Linear(c.width, c.vocab_size, bias=False, dtype=dtype)
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                source = torch.tensor(reference.weights[name.removesuffix('.weight')], dtype=dtype)
                # Embedding [vocab,width] stays unchanged; Linear stores [out,in].
                if source.ndim == 2 and name != 'embedding.weight':
                    source = source.T
                parameter.copy_(source)
        self.eval()

    def forward(self, tokens, return_trace=False):
        if tokens.dtype != torch.long or tokens.ndim != 2:
            raise ValueError('tokens must be a torch.long tensor shaped [batch, length].')
        if tokens.device.type != 'cpu' or self.embedding.weight.device.type != 'cpu':
            raise ValueError('This lesson validates CPU execution only.')
        if tokens.shape[0] == 0 or not 1 <= tokens.shape[1] <= self.config.max_context:
            raise ValueError('Require nonempty batch and 1..max_context input positions.')
        if (tokens < 0).any().item() or (tokens >= self.config.vocab_size).any().item():
            raise ValueError('Token IDs must be within the vocabulary.')
        x = self.embedding(tokens)
        trace = {'embedding': x}
        for i, layer in enumerate(self.layers):
            x = layer(x, trace, f'layers.{i}.')
        x = self.final_norm(x)
        trace['final_norm'] = x
        logits = self.head(x)
        trace['logits'] = logits
        return (logits, trace) if return_trace else logits

    @torch.inference_mode()
    def generate(self, tokens, max_new_tokens=4):
        validate_generation(tokens, max_new_tokens, self.config)
        sequence, generated = list(tokens), []
        for _ in range(max_new_tokens):
            logits = self(torch.tensor([sequence], dtype=torch.long))
            token = int(logits[0, -1].argmax())
            generated.append(token)
            sequence.append(token)
        return generated


@torch.inference_mode()
def decoder_experiment(prompt, max_new_tokens=4, seed=7, dtype='float64', layers=2):
    if dtype not in ('float64', 'float32'):
        raise ValueError('dtype must be float64 or float32.')
    config = DecoderConfig(layers=layers)
    validate_generation(prompt, max_new_tokens, config)
    reference = ScalarDecoder(config, seed)
    model = TorchDecoder(reference, getattr(torch, dtype))
    expected, reference_trace = reference.forward(prompt)
    actual, tensor_trace = model(torch.tensor([prompt], dtype=torch.long), return_trace=True)
    atol, rtol = (1e-12, 1e-10) if dtype == 'float64' else (1e-5, 1e-4)
    comparisons = []
    for name, values in reference_trace.items():
        oracle = torch.tensor(values, dtype=torch.float64)
        value = tensor_trace[name][0].to(torch.float64)
        diff = (value-oracle).abs()
        comparisons.append({'stage': name, 'shape': list(tensor_trace[name].shape),
                            'max_abs_error': diff.max().item(),
                            'passed': bool(torch.isfinite(value).all() and (diff <= atol+rtol*oracle.abs()).all())})
    scalar_ids = reference.generate(prompt, max_new_tokens)
    tensor_ids = model.generate(prompt, max_new_tokens)
    return {
        'experiment': 'part2-01-pytorch-decoder',
        'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                        'system': platform.platform(), 'device': 'cpu', 'dtype': dtype},
        'config': asdict(config), 'seed': seed, 'prompt': prompt,
        'max_new_tokens': max_new_tokens,
        'weights_sha256': hashlib.sha256(json.dumps(reference.weights, sort_keys=True, allow_nan=False).encode()).hexdigest(),
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'tolerance': {'atol': atol, 'rtol': rtol}, 'comparisons': comparisons,
        'reference_last_logits': expected[-1], 'torch_last_logits': actual[0, -1].tolist(),
        'reference_generated_ids': scalar_ids, 'torch_generated_ids': tensor_ids,
        'verification': {'passed': all(c['passed'] for c in comparisons) and scalar_ids == tensor_ids,
                         'all_stages_close': all(c['passed'] for c in comparisons),
                         'greedy_ids_equal': scalar_ids == tensor_ids},
        'scope': 'Untrained CPU dense decoder; full-prefix recomputation; no padding/KV cache/checkpoint/GPU performance validation.',
    }
