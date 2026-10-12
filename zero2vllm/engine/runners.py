"""Shared model weights, per-request contexts; no batching or scheduler here."""
from zero2vllm.engine.lifecycle import RequestContext
from zero2vllm.reference.attention import TinyAttentionLM, WorkCounter


class TinyCachedRunner:
    cache_kind = 'real_python_kv_lists'
    # Scheduler cost contract: first step consumes P positions, later steps one.
    supports_cached_steps = True

    def __init__(self, model=None):
        self.model = model if model is not None else TinyAttentionLM(max_context=32)
        self.vocab_size = self.model.vocab_size
        self.max_context = self.model.max_context

    def new_context(self):
        return RequestContext(kv=self.model.new_cache())

    def advance(self, prompt, generated, context):
        if context.closed or context.kv is None:
            raise RuntimeError('Request context is closed or missing KV.')
        expected = 0 if not generated else len(prompt) + len(generated) - 1
        if context.cached_positions != expected:
            raise ValueError('Cache length does not match this request history.')
        inputs = list(prompt) if not generated else [generated[-1]]
        logits = self.model.forward_cached(inputs, context.kv, WorkCounter())[-1]
        return max(range(self.vocab_size), key=logits.__getitem__)


class LlamaFullPrefixRunner:
    cache_kind = 'none_full_prefix_recompute'

    def __init__(self, model):
        # Optional import: the lifecycle core and tiny runner need only Python.
        if not model.loaded:
            raise ValueError('Load the lesson 2.2 checkpoint first.')
        self.model = model
        self.vocab_size = model.config.vocab_size
        self.max_context = model.config.max_position_embeddings

    def new_context(self):
        return RequestContext()

    def advance(self, prompt, generated, context):
        import torch
        if context.closed:
            raise RuntimeError('Request context is closed.')
        with torch.inference_mode():
            tokens = torch.tensor([prompt + generated], dtype=torch.long)
            return int(self.model(tokens)[0, -1].argmax())
