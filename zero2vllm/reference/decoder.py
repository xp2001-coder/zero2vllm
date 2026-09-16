"""Part 2.1: independent scalar decoder oracle; no torch or numpy imports.

Matrices use [input, output]. This is a small untrained, dense MHA model,
not a checkpoint-compatible Llama implementation. No cache or padding yet.
"""
from dataclasses import dataclass
import math
import random

from .attention import linear, softmax


@dataclass(frozen=True)
class DecoderConfig:
    vocab_size: int = 16
    width: int = 8
    heads: int = 2
    hidden: int = 16
    layers: int = 2
    max_context: int = 32
    eps: float = 1e-6
    rope_base: float = 10000.0

    def __post_init__(self):
        for value in (self.vocab_size, self.width, self.heads, self.hidden,
                      self.layers, self.max_context):
            if type(value) is not int or value <= 0:
                raise ValueError('Model dimensions must be positive integers.')
        if self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError('width must be divisible by heads, with even head dimension.')
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError('eps must be positive and finite.')
        if not math.isfinite(self.rope_base) or self.rope_base <= 1:
            raise ValueError('rope_base must be finite and greater than 1.')


def validate_tokens(tokens, config):
    if not tokens or len(tokens) > config.max_context:
        raise ValueError('Input must contain 1..max_context tokens.')
    if any(type(t) is not int or not 0 <= t < config.vocab_size for t in tokens):
        raise ValueError('Token IDs must be integers within the vocabulary.')


def validate_generation(tokens, count, config):
    validate_tokens(tokens, config)
    if type(count) is not int or count < 0:
        raise ValueError('max_new_tokens must be a nonnegative integer.')
    if len(tokens) + max(count - 1, 0) > config.max_context:
        raise ValueError('Generation inputs would exceed max_context.')


def make_weights(config, seed=7):
    rng = random.Random(seed)

    def matrix(rows, cols):
        return [[rng.uniform(-0.3, 0.3) for _ in range(cols)] for _ in range(rows)]

    def norm():
        return [rng.uniform(0.8, 1.2) for _ in range(config.width)]

    weights = {'embedding': matrix(config.vocab_size, config.width)}
    for i in range(config.layers):
        prefix = f'layers.{i}.'
        weights[prefix + 'norm1'] = norm()
        for name in ('q', 'k', 'v', 'o'):
            weights[prefix + name] = matrix(config.width, config.width)
        weights[prefix + 'norm2'] = norm()
        weights[prefix + 'gate'] = matrix(config.width, config.hidden)
        weights[prefix + 'up'] = matrix(config.width, config.hidden)
        weights[prefix + 'down'] = matrix(config.hidden, config.width)
    weights['final_norm'] = norm()
    weights['head'] = matrix(config.width, config.vocab_size)
    return weights


def rms_norm(rows, weight, eps):
    return [[value * gain / math.sqrt(sum(x*x for x in row) / len(row) + eps)
             for value, gain in zip(row, weight)] for row in rows]


def rope(rows, heads, base):
    """Adjacent pairs, position starting at zero, one frequency set per head."""
    dim = len(rows[0]) // heads
    result = []
    for pos, row in enumerate(rows):
        rotated = list(row)
        for head in range(heads):
            for pair in range(0, dim, 2):
                angle = pos / base ** (pair / dim)
                a, b = row[head*dim+pair:head*dim+pair+2]
                rotated[head*dim+pair] = a*math.cos(angle) - b*math.sin(angle)
                rotated[head*dim+pair+1] = a*math.sin(angle) + b*math.cos(angle)
        result.append(rotated)
    return result


def add(a, b):
    return [[x+y for x, y in zip(ar, br)] for ar, br in zip(a, b)]


class ScalarDecoder:
    def __init__(self, config=None, seed=7):
        self.config = config or DecoderConfig()
        self.weights = make_weights(self.config, seed)

    def forward(self, tokens):
        c, w = self.config, self.weights
        validate_tokens(tokens, c)
        x = [list(w['embedding'][t]) for t in tokens]
        trace = {'embedding': x}
        dim = c.width // c.heads
        for i in range(c.layers):
            prefix = f'layers.{i}.'
            normalized = rms_norm(x, w[prefix+'norm1'], c.eps)
            trace[prefix+'norm1'] = normalized
            q, k, v = [[linear(row, w[prefix+name]) for row in normalized] for name in ('q', 'k', 'v')]
            trace.update({prefix+'q': q, prefix+'k': k, prefix+'v': v})
            q, k = rope(q, c.heads, c.rope_base), rope(k, c.heads, c.rope_base)
            trace.update({prefix+'q_rope': q, prefix+'k_rope': k})
            context = []
            # Future keys are never visited: independent of torch's mask implementation.
            for pos in range(len(tokens)):
                row = []
                for head in range(c.heads):
                    start = head * dim
                    scores = [sum(q[pos][start+d]*k[j][start+d] for d in range(dim)) / math.sqrt(dim)
                              for j in range(pos+1)]
                    probs = softmax(scores)
                    row.extend(sum(probs[j]*v[j][start+d] for j in range(pos+1)) for d in range(dim))
                context.append(row)
            trace[prefix+'context'] = context
            x = add(x, [linear(row, w[prefix+'o']) for row in context])
            trace[prefix+'attention_residual'] = x
            normalized = rms_norm(x, w[prefix+'norm2'], c.eps)
            trace[prefix+'norm2'] = normalized
            gate = [linear(row, w[prefix+'gate']) for row in normalized]
            up = [linear(row, w[prefix+'up']) for row in normalized]
            trace.update({prefix+'gate': gate, prefix+'up': up})
            gated = [[(g / (1 + math.exp(-g))) * u for g, u in zip(gr, ur)] for gr, ur in zip(gate, up)]
            trace[prefix+'swiglu'] = gated
            down = [linear(row, w[prefix+'down']) for row in gated]
            trace[prefix+'down'] = down
            x = add(x, down)
            trace[prefix+'output'] = x
        x = rms_norm(x, w['final_norm'], c.eps)
        trace['final_norm'] = x
        logits = [linear(row, w['head']) for row in x]
        trace['logits'] = logits
        return logits, trace

    def generate(self, tokens, max_new_tokens=4):
        validate_generation(tokens, max_new_tokens, self.config)
        sequence, generated = list(tokens), []
        for _ in range(max_new_tokens):
            logits, _ = self.forward(sequence)
            token = max(range(self.config.vocab_size), key=lambda j: logits[-1][j])
            generated.append(token)
            sequence.append(token)
        return generated
