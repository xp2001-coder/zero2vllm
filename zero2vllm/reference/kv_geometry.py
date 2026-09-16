"""Lesson 1.3: cache offsets, grouped-query arithmetic, and storage estimates."""

from dataclasses import asdict
import math

from .attention import TinyAttentionLM, WorkCounter, softmax


def kv_capacity(layers: int = 32, query_heads: int = 32, kv_heads: int = 8,
                head_dim: int = 128, sequence_length: int = 4096,
                batch_size: int = 1, element_bytes: int = 2) -> dict:
    if any(type(x) is not int or x <= 0 for x in
           (layers, query_heads, kv_heads, head_dim, sequence_length, batch_size, element_bytes)):
        raise ValueError("Cache dimensions and element_bytes must be positive integers.")
    if query_heads % kv_heads:
        raise ValueError("query_heads must be divisible by kv_heads.")
    per_token = 2 * layers * kv_heads * head_dim * element_bytes
    total = per_token * sequence_length * batch_size
    return {"record_type": "analytical_estimate", "layers": layers, "query_heads": query_heads,
            "kv_heads": kv_heads, "head_dim": head_dim, "sequence_length": sequence_length,
            "batch_size": batch_size, "element_bytes": element_bytes,
            "query_heads_per_kv_head": query_heads // kv_heads,
            "bytes_per_token_all_layers": per_token, "payload_bytes": total,
            "payload_mib": total / 2**20, "payload_gib": total / 2**30,
            "assumptions": "Dense K/V, equal K/V width, all layers same geometry; no pages, padding, scales, replicas or workspace."}


def grouped_attention(queries: list[list[float]], keys: list[list[list[float]]],
                      values: list[list[list[float]]]) -> list[list[float]]:
    """One newest query per Q head; K/V layout [sequence, KV heads, head dim]."""
    if not queries or not keys or len(keys) != len(values) or not keys[0]:
        raise ValueError("Nonempty Q and matching nonempty K/V sequences are required.")
    hq, hkv, width = len(queries), len(keys[0]), len(queries[0])
    if width < 1 or hq % hkv:
        raise ValueError("Q heads must be divisible by KV heads and width must be positive.")
    if any(len(row) != width for row in queries) or any(
            len(token) != hkv or any(len(row) != width for row in token) for token in keys + values):
        raise ValueError("Q/K/V shapes do not match.")
    if any(not math.isfinite(x) for row in queries for x in row) or any(
            not math.isfinite(x) for token in keys + values for row in token for x in row):
        raise ValueError("Q/K/V must contain finite numbers.")
    result = []
    for head, query in enumerate(queries):
        group = head // (hq // hkv)
        probabilities = softmax([sum(q * k for q, k in zip(query, token[group])) / math.sqrt(width)
                                 for token in keys])
        result.append([sum(p * token[group][d] for p, token in zip(probabilities, values))
                       for d in range(width)])
    return result


def cache_experiment(prompt: list[int], chunk_size: int = 2, seed: int = 7,
                     sequence_length: int = 4096, batch_size: int = 1,
                     kv_heads: int = 8) -> dict:
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    model = TinyAttentionLM(seed=seed)
    reference = model.forward_full(prompt, WorkCounter())
    cache, counter, chunks, actual = model.new_cache(), WorkCounter(), [], []
    for start in range(0, len(prompt), chunk_size):
        tokens = prompt[start:start + chunk_size]
        output = model.forward_cached(tokens, cache, counter)
        actual.extend(output)
        chunks.append({"input_token_ids": tokens, "position_offset": start,
                       "positions": list(range(start, start + len(tokens))),
                       "cache_length_after": len(cache.keys), "logits": output})
    error = max(abs(x - y) for row, ref in zip(actual, reference) for x, y in zip(row, ref))
    q = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]]
    k = [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]]
    v = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
    grouped = grouped_attention(q, k, v)
    expanded_k = [[head for head in token for _ in range(2)] for token in k]
    expanded_v = [[head for head in token for _ in range(2)] for token in v]
    expanded = grouped_attention(q, expanded_k, expanded_v)
    group_error = max(abs(a - b) for row, ref in zip(grouped, expanded) for a, b in zip(row, ref))
    estimates = {name: kv_capacity(kv_heads=heads, sequence_length=sequence_length, batch_size=batch_size)
                 for name, heads in (("mha", 32), ("gqa", kv_heads), ("mqa", 1))}
    return {"schema_version": 1, "experiment": "part1_kv_cache", "record_type": "measured_reference",
            "input": {"prompt_token_ids": prompt, "chunk_size": chunk_size, "seed": seed},
            "chunks": chunks, "cached_keys": cache.keys, "cached_values": cache.values,
            "work": asdict(counter), "full_logits": reference,
            "grouped_query_example": {"q": q, "k": k, "v": v, "q_to_kv_head": [0, 0, 1, 1],
                                      "grouped_output": grouped, "expanded_output": expanded},
            "capacity_estimates": estimates,
            "verification": {"max_abs_chunk_logit_error": error, "max_abs_grouped_error": group_error,
                             "passed": error <= 1e-12 and group_error <= 1e-12},
            "limitations": ["CPU cache stores Python lists, not a packed GPU payload.",
                            "Storage estimates and four-head example have independent, explicit dimensions."]}
