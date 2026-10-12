"""Lesson 3.3 numerical fixture: one CPU attention layer, NOT a Llama model.

    Embedding -> fused QKV -> split-half RoPE -> paged GQA -> O/residual -> head.
No norm, MLP, tokenizer, checkpoint files, sampling RNG, GPU or FlashAttention.
Dense independent-request evaluation supplies the numerical reference.
"""
import math
import torch
from torch.nn import functional as F

from zero2vllm.reference.execution import pack_inputs, route_candidates


def fuse_qkv(weights, *, hidden_size, query_heads, kv_heads, head_dim):
    """Validate all named [out,in] matrices before concatenating rows.

    This unquantized TP=1 exercise is not the upstream packed-weight loader.
    """
    if set(weights) != {'q_proj', 'k_proj', 'v_proj'}:
        raise ValueError('Require exactly q_proj, k_proj and v_proj')
    tensors = [weights[n] for n in ('q_proj', 'k_proj', 'v_proj')]
    sizes = (query_heads*head_dim, kv_heads*head_dim, kv_heads*head_dim)
    for t, out in zip(tensors, sizes):
        if (not isinstance(t, torch.Tensor) or tuple(t.shape) != (out, hidden_size)
                or t.device.type != 'cpu' or t.dtype not in (torch.float32, torch.float64)
                or not torch.isfinite(t).all().item()):
            raise ValueError('Expected finite CPU [out,in] floating-point weights')
    if len({t.dtype for t in tensors}) != 1:
        raise ValueError('Projection weight dtypes must agree')
    return torch.cat(tensors, dim=0).clone()


def rope(x, positions, base=10000.0):
    """[tokens,heads,even_dim], using explicit request-local positions."""
    dim = x.shape[-1]
    if x.ndim != 3 or dim % 2 or len(positions) != x.shape[0]:
        raise ValueError('Invalid RoPE shape or positions')
    pos = torch.as_tensor(positions, dtype=x.dtype)
    inv = base ** (-torch.arange(0, dim, 2, dtype=x.dtype)/dim)
    angles = pos[:, None]*inv[None, :]
    angles = torch.cat([angles, angles], dim=-1)[:, None, :]
    first, second = x.chunk(2, dim=-1)
    return x*angles.cos() + torch.cat([-second, first], dim=-1)*angles.sin()


class AttentionFixture:
    vocab_size = 32
    hidden_size = 8
    query_heads = 4
    kv_heads = 2
    head_dim = 2

    def __init__(self, dtype=torch.float64, seed=303):
        if dtype not in (torch.float32, torch.float64):
            raise ValueError('This fixture supports CPU float32/float64')
        self.dtype = dtype
        generator = torch.Generator(device='cpu').manual_seed(seed)

        def matrix(rows, cols):
            return torch.randn(rows, cols, generator=generator, dtype=dtype)/math.sqrt(cols)

        self.embedding = matrix(self.vocab_size, self.hidden_size)
        self.weights = {'q_proj': matrix(8, 8), 'k_proj': matrix(4, 8), 'v_proj': matrix(4, 8)}
        self.fused = fuse_qkv(self.weights, hidden_size=8, query_heads=4, kv_heads=2, head_dim=2)
        self.output_weight = matrix(8, 8)
        self.head_weight = matrix(32, 8)

    def project(self, ids, positions, *, fused=True):
        x = self.embedding[torch.tensor(ids, dtype=torch.long)]
        if fused:
            q, k, v = F.linear(x, self.fused).split((8, 4, 4), dim=-1)
        else:
            q, k, v = (F.linear(x, self.weights[n]) for n in ('q_proj', 'k_proj', 'v_proj'))
        q = rope(q.reshape(-1, 4, 2), positions)
        k = rope(k.reshape(-1, 2, 2), positions)
        return x, q, k, v.reshape(-1, 2, 2)

    def finish(self, x, attended):
        hidden = x + F.linear(attended.reshape(-1, 8), self.output_weight)
        return F.linear(hidden, self.head_weight)

    @torch.inference_mode()
    def dense(self, ids):
        """All-prefix reference: separate projections, repeated GQA heads, tril."""
        n = len(ids)
        x, q, k, v = self.project(ids, list(range(n)), fused=False)
        k, v = k.repeat_interleave(2, dim=1), v.repeat_interleave(2, dim=1)
        scores = torch.einsum('ihd,jhd->hij', q, k)/math.sqrt(2)
        future = torch.ones(n, n, dtype=torch.bool).triu(1)
        probs = scores.masked_fill(future, float('-inf')).softmax(-1)
        attended = torch.einsum('hij,jhd->ihd', probs, v)
        return self.finish(x, attended)

    @torch.inference_mode()
    def execute(self, requests, order, *, block_size=4, num_blocks=10,
                pad_to=None, padding_token=0):
        """Prepare a fresh one-step fixture; prior KV is recomputed to seed it.

        No persistent engine or allocator is implemented. Only step outputs,
        valid stores and request ownership are compared with dense references.
        """
        batch = pack_inputs(requests, order, block_size=block_size, num_blocks=num_blocks,
                            vocab_size=self.vocab_size, pad_to=pad_to, padding_token=padding_token)
        cache = torch.full((2, num_blocks, block_size, 2, 2), float('nan'), dtype=self.dtype)
        flat = cache.view(2, -1, 2, 2)
        written = torch.zeros(num_blocks*block_size, dtype=torch.bool)

        def write(slots, k, v):
            slots = torch.tensor(slots, dtype=torch.long)
            if written[slots].any().item():
                raise ValueError('Attempt to overwrite an already written position')
            flat[0].index_copy_(0, slots, k)
            flat[1].index_copy_(0, slots, v)
            written[slots] = True

        # Seed only the BEFORE-step valid prefix, never the remaining prompt.
        for r in batch.requests:
            if r.computed:
                _, _, k, v = self.project(r.token_ids[:r.computed], tuple(range(r.computed)))
                slots = [r.block_table[p//block_size]*block_size+p%block_size for p in range(r.computed)]
                write(slots, k, v)
        x, q, k, v = self.project(batch.input_ids, batch.positions)
        total = batch.num_actual_tokens
        # Padded projection rows exist, but have NO cache write addresses.
        write(batch.slot_mapping, k[:total], v[:total])
        out = torch.empty(total, 4, 2, dtype=self.dtype)
        group_indices = torch.arange(4)//2
        for i, r in enumerate(batch.requests):
            begin, end = batch.query_start_loc[i:i+2]
            length = batch.seq_lens[i]
            slots = [r.block_table[p//block_size]*block_size+p%block_size for p in range(length)]
            if not written[slots].all().item():
                raise ValueError('Attention tried to read an unwritten position')
            keys, values = flat[0, slots], flat[1, slots]
            for row in range(begin, end):
                # Request-local absolute position; not row or chunk-local index.
                visible = torch.arange(length) <= batch.positions[row]
                scores = torch.einsum('hd,thd->ht', q[row], keys[:, group_indices])/math.sqrt(2)
                probs = scores.masked_fill(~visible[None, :], float('-inf')).softmax(-1)
                out[row] = torch.einsum('ht,thd->hd', probs, values[:, group_indices])
        logits = self.finish(x[:total], out)
        selected = logits[list(batch.logits_indices)]
        candidates = selected.argmax(-1).tolist()
        by_id = {r.request_id: logits[batch.query_start_loc[i]:batch.query_start_loc[i+1]]
                 for i, r in enumerate(batch.requests)}
        errors = {r.request_id: float((by_id[r.request_id]-self.dense(r.token_ids[:r.computed+r.scheduled])[r.computed:]).abs().max())
                  for r in batch.requests}
        return {'batch': batch, 'cache': cache, 'written': written, 'logits': logits,
                'logits_by_id': by_id, 'candidate_ids': candidates,
                'returned_ids': route_candidates(batch, candidates), 'errors': errors,
                'shapes': {'hidden': list(x.shape), 'q': list(q.shape), 'k': list(k.shape),
                           'v': list(v.shape), 'cache': list(cache.shape),
                           'logits': list(logits.shape), 'selected_logits': list(selected.shape)}}
