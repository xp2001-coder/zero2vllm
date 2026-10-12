"""Prefix sharing in the existing lifecycle engine, with conservative admission.

Admission still charges the full rounded logical reservation. Actual blocks are
bound at first forward, after the prompt is known. Sharing lowers physical use
and prefill work but does not increase this lesson's scheduler admission limit.
"""
import copy
import hashlib
import json

import torch

from zero2vllm.cache.prefix import PrefixKVStore
from zero2vllm.reference.attention import TinyAttentionLM, linear


class PrefixContext:
    def __init__(self, store, namespace, reuse):
        self.store, self.namespace, self.reuse = store, namespace, reuse
        self.handle = None
        self.reserved = 0
        self.closed = False
        self.prompt = None
        self.hit_positions = 0
        self.projected_positions = self.attention_pairs = 0
        self.last_logits = None

    @property
    def cached_positions(self):
        return 0 if self.closed or self.handle is None else self.store.snapshot(self.handle)['written_positions']

    def reserve(self, positions):
        if self.closed or self.handle is not None:
            raise ValueError('Reservation must precede prefix binding.')
        if type(positions) is not int or positions < 0:
            raise ValueError('Reservation must be nonnegative.')
        self.reserved = positions

    def close(self):
        if not self.closed:
            if self.handle is not None:
                self.store.release(self.handle)
            self.closed = True


class PrefixTinyRunner:
    supports_cached_steps = True
    cache_kind = 'cpu_shared_full_prefix_blocks'

    def __init__(self, *, num_blocks=16, block_size=2, dtype=torch.float64,
                 model=None, cache_salt='public', reuse=True):
        if not isinstance(cache_salt, str) or not cache_salt or type(reuse) is not bool:
            raise ValueError('Use a nonempty cache salt and a boolean reuse setting.')
        # Own a fixed weight snapshot; outside mutation cannot silently stale KV.
        self._model = copy.deepcopy(model) if model is not None else TinyAttentionLM(max_context=32)
        self.vocab_size, self.max_context = self._model.vocab_size, self._model.max_context
        weights = {name: getattr(self._model, name) for name in ('embedding', 'position', 'wq', 'wk', 'wv', 'wo', 'head')}
        payload = json.dumps(['tiny-learned-absolute-v1', str(dtype), weights], sort_keys=True, separators=(',', ':'))
        self.model_fingerprint = hashlib.sha256(payload.encode()).hexdigest()
        self.namespace = json.dumps([self.model_fingerprint, cache_salt], separators=(',', ':'))
        self.reuse = reuse
        self.store = PrefixKVStore(num_blocks, block_size, head_dim=self._model.width, dtype=dtype)
        self.allocation_unit = block_size
        self.cache_capacity_tokens = self.store.capacity_tokens

    def new_context(self):
        return PrefixContext(self.store, self.namespace, self.reuse)

    def advance(self, prompt, generated, context):
        if context.closed or context.store is not self.store:
            raise ValueError('Context is closed or belongs to another runner.')
        prompt, generated = tuple(prompt), tuple(generated)
        self._model.validate(list(prompt)+list(generated))
        if context.handle is None:
            if generated:
                raise ValueError('The first step cannot already have generated tokens.')
            context.handle = self.store.acquire(prompt, context.reserved,
                                                namespace=context.namespace, reuse=context.reuse)
            context.prompt = prompt
            context.hit_positions = context.cached_positions
            inputs = prompt[context.hit_positions:]
        else:
            expected = len(prompt)+len(generated)-1
            snapshot = self.store.snapshot(context.handle)
            if (not generated or context.prompt != prompt or context.cached_positions != expected
                    or snapshot['tokens'] != list(prompt+generated[:-1])):
                raise ValueError('Prefix cache does not match the committed request history.')
            inputs = generated[-1:]
        if context.cached_positions+len(inputs) > context.reserved:
            raise ValueError('Input exceeds the reservation.')
        for token in inputs:
            position = context.cached_positions
            hidden = self._model._embed(token, position)
            query = torch.tensor([linear(hidden, self._model.wq)], dtype=self.store.dtype, device='cpu')
            keys = torch.tensor([[linear(hidden, self._model.wk)]], dtype=self.store.dtype, device='cpu')
            values = torch.tensor([[linear(hidden, self._model.wv)]], dtype=self.store.dtype, device='cpu')
            self.store.append(context.handle, [token], keys, values)
            context.projected_positions += 1
            context.attention_pairs += position+1
            attended = self.store.attention(context.handle, query)[0].tolist()
            context.last_logits = self._model._logits(hidden, attended)
        return max(range(self.vocab_size), key=context.last_logits.__getitem__)
