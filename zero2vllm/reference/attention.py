"""A single-head causal attention model with deterministic random weights.

The full and cached paths independently implement causal attention. Learned
position embeddings make cache offsets observable. There is no tokenizer,
training, MLP, normalization, RoPE, or claim of meaningful language generation.
"""

from dataclasses import dataclass, field
import math
import random

Vector = list[float]
Matrix = list[Vector]


def linear(vector: Vector, weight: Matrix) -> Vector:
    return [sum(x * row[j] for x, row in zip(vector, weight))
            for j in range(len(weight[0]))]


def softmax(scores: Vector) -> Vector:
    largest = max(scores)
    exp = [math.exp(score - largest) for score in scores]
    total = sum(exp)
    return [value / total for value in exp]


@dataclass
class WorkCounter:
    # One position means computing its Q, K and V once; not a FLOP count.
    projected_token_positions: int = 0
    attention_score_pairs: int = 0


@dataclass
class KVCache:
    owner: object = field(repr=False)
    keys: Matrix = field(default_factory=list)
    values: Matrix = field(default_factory=list)


class TinyAttentionLM:
    def __init__(self, vocab_size: int = 16, width: int = 8,
                 max_context: int = 128, seed: int = 7):
        if any(type(value) is not int or value <= 0
               for value in (vocab_size, width, max_context)):
            raise ValueError("Model dimensions must be positive integers.")
        self.vocab_size = vocab_size
        self.width = width
        self.max_context = max_context
        self._owner = object()
        rng = random.Random(seed)

        def matrix(rows: int, columns: int) -> Matrix:
            return [[rng.uniform(-0.5, 0.5) for _ in range(columns)]
                    for _ in range(rows)]

        self.embedding = matrix(vocab_size, width)
        self.position = matrix(max_context, width)
        self.wq = matrix(width, width)
        self.wk = matrix(width, width)
        self.wv = matrix(width, width)
        self.wo = matrix(width, width)
        self.head = matrix(width, vocab_size)

    def new_cache(self) -> KVCache:
        return KVCache(owner=self._owner)

    def validate(self, tokens: list[int], offset: int = 0) -> None:
        if not tokens:
            raise ValueError("At least one input token is required.")
        if any(type(token) is not int or not 0 <= token < self.vocab_size
               for token in tokens):
            raise ValueError(f"Token IDs must be integers in [0, {self.vocab_size}).")
        if offset < 0 or offset + len(tokens) > self.max_context:
            raise ValueError("Input exceeds the model context capacity.")

    def _embed(self, token: int, position: int) -> Vector:
        return [x + p for x, p in zip(self.embedding[token], self.position[position])]

    def _logits(self, residual: Vector, context: Vector) -> Vector:
        projected = linear(context, self.wo)
        return linear([x + y for x, y in zip(residual, projected)], self.head)

    def forward_full(self, tokens: list[int], counter: WorkCounter) -> Matrix:
        """Recompute all input positions with an explicit causal mask."""
        self.validate(tokens)
        hidden = [self._embed(token, i) for i, token in enumerate(tokens)]
        queries = [linear(x, self.wq) for x in hidden]
        keys = [linear(x, self.wk) for x in hidden]
        values = [linear(x, self.wv) for x in hidden]
        counter.projected_token_positions += len(tokens)
        outputs = []
        for i, query in enumerate(queries):
            # j <= i: future token keys must not affect this position.
            scores = [sum(q * k for q, k in zip(query, keys[j])) / math.sqrt(self.width)
                      for j in range(i + 1)]
            probabilities = softmax(scores)
            context = [sum(probabilities[j] * values[j][d] for j in range(i + 1))
                       for d in range(self.width)]
            counter.attention_score_pairs += i + 1
            outputs.append(self._logits(hidden[i], context))
        return outputs

    def forward_cached(self, tokens: list[int], cache: KVCache,
                       counter: WorkCounter) -> Matrix:
        """Append a prompt chunk or decode token using absolute positions."""
        if cache.owner is not self._owner:
            raise ValueError("This cache belongs to another model instance.")
        if len(cache.keys) != len(cache.values):
            raise ValueError("Cache key/value lengths differ.")
        offset = len(cache.keys)
        # Validate the entire chunk before mutating its cache or counter.
        self.validate(tokens, offset)
        outputs = []
        for i, token in enumerate(tokens):
            hidden = self._embed(token, offset + i)
            query = linear(hidden, self.wq)
            cache.keys.append(linear(hidden, self.wk))
            cache.values.append(linear(hidden, self.wv))
            scores = [sum(q * k for q, k in zip(query, key)) / math.sqrt(self.width)
                      for key in cache.keys]
            probabilities = softmax(scores)
            context = [sum(p * value[d] for p, value in zip(probabilities, cache.values))
                       for d in range(self.width)]
            counter.projected_token_positions += 1
            counter.attention_score_pairs += len(cache.keys)
            outputs.append(self._logits(hidden, context))
        return outputs


@dataclass
class Generation:
    token_ids: list[int]
    logits: Matrix
    work: WorkCounter


def generate(model: TinyAttentionLM, prompt: list[int], max_new_tokens: int,
             *, use_cache: bool) -> Generation:
    model.validate(prompt)
    if type(max_new_tokens) is not int or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a nonnegative integer.")
    # The final generated token is returned, but does not need a forward pass.
    if len(prompt) + max(max_new_tokens - 1, 0) > model.max_context:
        raise ValueError("Generation exceeds the model context capacity.")
    generated: list[int] = []
    trace: Matrix = []
    work = WorkCounter()
    cache = model.new_cache()
    for step in range(max_new_tokens):
        if use_cache:
            inputs = list(prompt) if step == 0 else [generated[-1]]
            logits = model.forward_cached(inputs, cache, work)[-1]
        else:
            logits = model.forward_full(list(prompt) + generated, work)[-1]
        trace.append(logits)
        generated.append(max(range(model.vocab_size), key=logits.__getitem__))
    return Generation(generated, trace, work)
