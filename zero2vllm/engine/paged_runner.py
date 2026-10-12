"""Tiny reference model + CPU tensor KV pages; no real-model or GPU claim."""
import torch

from zero2vllm.cache.paged import PagedContext, PagedKVStore
from zero2vllm.reference.attention import TinyAttentionLM, linear


class PagedTinyRunner:
    supports_cached_steps = True
    cache_kind = 'cpu_tensor_pages_gather_reference'

    def __init__(self, *, num_blocks=8, block_size=4, dtype=torch.float64, model=None):
        self.model = model if model is not None else TinyAttentionLM(max_context=32)
        self.store = PagedKVStore(num_blocks, block_size, head_dim=self.model.width, dtype=dtype)
        self.vocab_size = self.model.vocab_size
        self.max_context = self.model.max_context
        self.allocation_unit = block_size
        self.cache_capacity_tokens = self.store.capacity_tokens

    def new_context(self):
        return PagedContext(self.store)

    def forward_tokens(self, tokens, context):
        if context.closed or context.store is not self.store:
            raise ValueError('Context is closed or belongs to another runner.')
        offset = context.cached_positions
        tokens = list(tokens)
        self.model.validate(tokens, offset)
        if offset+len(tokens) > self.store.snapshot(context.handle)['reserved_positions']:
            raise ValueError('Input exceeds the reserved logical length.')
        logits = []
        # Append one position before reading its causal prefix. A whole prompt
        # never exposes a later position to an earlier query.
        for i, token in enumerate(tokens):
            hidden = self.model._embed(token, offset+i)
            q = torch.tensor([linear(hidden, self.model.wq)], dtype=self.store.dtype)
            k = torch.tensor([[linear(hidden, self.model.wk)]], dtype=self.store.dtype)
            v = torch.tensor([[linear(hidden, self.model.wv)]], dtype=self.store.dtype)
            self.store.append(context.handle, k, v)
            attended = self.store.attention(context.handle, q)[0].tolist()
            logits.append(self.model._logits(hidden, attended))
        return logits

    def advance(self, prompt, generated, context):
        expected = 0 if not generated else len(prompt)+len(generated)-1
        if context.cached_positions != expected:
            raise ValueError('Paged cache length does not match request history.')
        inputs = list(prompt) if not generated else [generated[-1]]
        logits = self.forward_tokens(inputs, context)[-1]
        return max(range(self.vocab_size), key=logits.__getitem__)
