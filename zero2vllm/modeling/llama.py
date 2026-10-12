"""Lesson 2.2: CPU float32, full-prefix Llama subset for SmolLM2-135M.

The equations extend lesson 2.1 with GQA, split-half RoPE and tied embeddings.
No Transformers dependency in this forward path. No padding or KV cache yet.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from zero2vllm.modeling.torch_decoder import RMSNorm


@dataclass(frozen=True)
class LlamaConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_dict(cls, data):
        if data.get('model_type') != 'llama':
            raise ValueError('Only the documented Llama/SmolLM2 subset is supported.')
        for name, expected in {
            'hidden_act': 'silu', 'attention_bias': False, 'mlp_bias': False,
            'rope_interleaved': False, 'pretraining_tp': 1,
            'partial_rotary_factor': 1.0,
        }.items():
            if data.get(name, expected) != expected:
                raise ValueError(f'Unsupported {name}: {data[name]!r}')
        for name in ('rope_scaling', 'quantization_config', 'sliding_window'):
            if data.get(name) is not None:
                raise ValueError(f'Unsupported {name}; do not silently ignore model semantics.')
        required = (
            'vocab_size', 'hidden_size', 'intermediate_size', 'num_hidden_layers',
            'num_attention_heads', 'num_key_value_heads', 'max_position_embeddings',
            'rms_norm_eps', 'rope_theta', 'tie_word_embeddings',
        )
        missing = set(required) - data.keys()
        if missing:
            raise ValueError(f'Missing config fields: {sorted(missing)}')
        for name in required[:7]:
            if type(data[name]) is not int or data[name] <= 0:
                raise ValueError(f'{name} must be a positive integer.')
        for name in ('rms_norm_eps', 'rope_theta'):
            value = data[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive.')
        if type(data['tie_word_embeddings']) is not bool:
            raise ValueError('tie_word_embeddings must be boolean.')
        result = cls(**{name: data[name] for name in required})
        if result.hidden_size % result.num_attention_heads or result.head_dim % 2:
            raise ValueError('hidden_size must divide into heads with an even head_dim.')
        if result.num_attention_heads % result.num_key_value_heads:
            raise ValueError('num_attention_heads must be divisible by num_key_value_heads.')
        if data.get('head_dim', result.head_dim) != result.head_dim:
            raise ValueError('Custom head_dim is not supported.')
        return result


def split_half_rope(x, base):
    """Rotate (0, Dh/2), (1, Dh/2+1), ...; positions start at zero."""
    dim = x.shape[-1]
    positions = torch.arange(x.shape[-2], dtype=torch.float32, device=x.device)
    inverse = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=x.device) / dim))
    angles = positions[:, None] * inverse[None, :]
    angles = torch.cat((angles, angles), dim=-1)
    first, second = x.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return x * angles.cos() + rotated * angles.sin()


class LlamaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = c = config
        self.norm1 = RMSNorm(c.hidden_size, c.rms_norm_eps, torch.float32)
        self.norm2 = RMSNorm(c.hidden_size, c.rms_norm_eps, torch.float32)
        for name, out in (('q', c.hidden_size), ('k', c.num_key_value_heads*c.head_dim),
                          ('v', c.num_key_value_heads*c.head_dim), ('o', c.hidden_size)):
            setattr(self, name, nn.Linear(c.hidden_size, out, bias=False, dtype=torch.float32))
        self.gate = nn.Linear(c.hidden_size, c.intermediate_size, bias=False, dtype=torch.float32)
        self.up = nn.Linear(c.hidden_size, c.intermediate_size, bias=False, dtype=torch.float32)
        self.down = nn.Linear(c.intermediate_size, c.hidden_size, bias=False, dtype=torch.float32)

    def forward(self, x):
        c = self.config
        batch, length, _ = x.shape
        n = self.norm1(x)
        q = self.q(n).view(batch, length, c.num_attention_heads, c.head_dim).transpose(1, 2)
        k = self.k(n).view(batch, length, c.num_key_value_heads, c.head_dim).transpose(1, 2)
        v = self.v(n).view(batch, length, c.num_key_value_heads, c.head_dim).transpose(1, 2)
        q, k = split_half_rope(q, c.rope_theta), split_half_rope(k, c.rope_theta)
        groups = c.num_attention_heads // c.num_key_value_heads
        # Educational materialization; production kernels can index KV groups directly.
        k, v = k.repeat_interleave(groups, dim=1), v.repeat_interleave(groups, dim=1)
        scores = (q @ k.transpose(-2, -1)) * (c.head_dim ** -0.5)
        future = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
        probs = scores.masked_fill(future, float('-inf')).softmax(dim=-1)
        context = (probs @ v).transpose(1, 2).contiguous().view(batch, length, c.hidden_size)
        x = x + self.o(context)
        n = self.norm2(x)
        return x + self.down(F.silu(self.gate(n)) * self.up(n))


class LlamaDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = c = config
        # Allocate without random initialization; the loader must fill every parameter.
        with torch.device('meta'):
            self.embedding = nn.Embedding(c.vocab_size, c.hidden_size, dtype=torch.float32)
            self.layers = nn.ModuleList([LlamaBlock(c) for _ in range(c.num_hidden_layers)])
            self.final_norm = RMSNorm(c.hidden_size, c.rms_norm_eps, torch.float32)
            self.head = nn.Linear(c.hidden_size, c.vocab_size, bias=False, dtype=torch.float32)
        self.to_empty(device='cpu')
        if c.tie_word_embeddings:
            self.head.weight = self.embedding.weight
        self.loaded = False
        self.eval()

    def forward(self, tokens, return_trace=False):
        if not self.loaded:
            raise RuntimeError('Load and validate every checkpoint parameter before inference.')
        if tokens.device.type != 'cpu' or self.embedding.weight.device.type != 'cpu':
            raise ValueError('This lesson validates CPU execution only.')
        if self.embedding.weight.dtype != torch.float32:
            raise ValueError('This lesson validates float32 computation only.')
        if tokens.dtype != torch.long or tokens.ndim != 2:
            raise ValueError('tokens must be torch.long with shape [batch, length].')
        if tokens.shape[0] < 1 or not 1 <= tokens.shape[1] <= self.config.max_position_embeddings:
            raise ValueError('Require nonempty batch and input within max_position_embeddings.')
        if (tokens < 0).any().item() or (tokens >= self.config.vocab_size).any().item():
            raise ValueError('Token ID is outside the configured vocabulary.')
        x = self.embedding(tokens)
        trace = {'embedding': x} if return_trace else None
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if return_trace:
                trace[f'layers.{i}'] = x
        x = self.final_norm(x)
        if return_trace:
            trace['final_norm'] = x
        logits = self.head(x)
        return (logits, trace) if return_trace else logits


@torch.inference_mode()
def greedy_generate(model, tokens, max_new_tokens, eos_token_id=None):
    """Full-prefix, one request, stop on EOS; no sampling defaults are inherited."""
    if tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.dtype != torch.long or tokens.numel() == 0:
        raise ValueError('Generation requires one nonempty torch.long request.')
    if type(max_new_tokens) is not int or max_new_tokens < 0:
        raise ValueError('max_new_tokens must be a nonnegative integer.')
    if tokens.min() < 0 or tokens.max() >= model.config.vocab_size:
        raise ValueError('Token ID is outside the configured vocabulary.')
    # The final predicted token need not be fed back into another forward pass.
    context = model.config.max_position_embeddings
    if tokens.shape[1] > context or tokens.shape[1] + max(0, max_new_tokens-1) > context:
        raise ValueError('Requested generation would exceed the input context limit.')
    if not model.loaded:
        raise RuntimeError('Checkpoint is not loaded.')
    ids, sequence = [], tokens.clone()
    for _ in range(max_new_tokens):
        token = int(model(sequence)[0, -1].argmax())
        ids.append(token)
        if token == eos_token_id:
            break
        sequence = torch.cat((sequence, torch.tensor([[token]], dtype=torch.long)), dim=1)
    return ids
