"""Lesson 1.2: actual CPU work counts and a separate analytical GEMM model."""

from dataclasses import asdict
import math

from .attention import TinyAttentionLM, WorkCounter, generate


def linear_cost(tokens: int, input_width: int, output_width: int,
                element_bytes: int = 2, peak_tflops: float = 100.0,
                bandwidth_gbps: float = 1000.0) -> dict:
    """Idealized one-GEMM lower bound; decimal TFLOP/s and GB/s, not hardware timing."""
    if any(type(x) is not int or x <= 0 for x in
           (tokens, input_width, output_width, element_bytes)):
        raise ValueError("GEMM dimensions and element_bytes must be positive integers.")
    if any(isinstance(x, bool) or not math.isfinite(x) or x <= 0
           for x in (peak_tflops, bandwidth_gbps)):
        raise ValueError("Assumed peak compute and bandwidth must be finite and positive.")
    m, k, n = tokens, input_width, output_width
    flops = 2 * m * k * n  # conventional multiply-add = two FLOPs
    weights = k * n * element_bytes
    activations = (m * k + m * n) * element_bytes
    traffic = weights + activations
    compute_s = flops / (peak_tflops * 1e12)
    memory_s = traffic / (bandwidth_gbps * 1e9)
    return {
        "record_type": "analytical_estimate", "shape_m_k_n": [m, k, n],
        "element_bytes": element_bytes, "flops": flops,
        "weight_bytes": weights, "input_and_output_bytes": activations,
        "ideal_transfer_bytes": traffic, "arithmetic_intensity_flops_per_byte": flops / traffic,
        "assumed_peak_tflops": peak_tflops, "assumed_bandwidth_gbps": bandwidth_gbps,
        "ridge_point_flops_per_byte": peak_tflops * 1e3 / bandwidth_gbps,
        "compute_lower_bound_ms": compute_s * 1e3,
        "transfer_lower_bound_ms": memory_s * 1e3,
        "roofline_lower_bound_ms": max(compute_s, memory_s) * 1e3,
        "limiting_bound": "compute" if compute_s >= memory_s else "transfer",
        "assumptions": ["One dense GEMM; each weight/input read once and output written once.",
                        "No bias, KV reads, softmax, MLP stack, cache residency, launch, or communication.",
                        "Peak rates are hypothetical inputs, not A100/C500 specifications or achieved rates."],
    }


def phase_experiment(prompt: list[int], max_new_tokens: int = 8, seed: int = 7,
                     prefill_tokens: int = 128, decode_batch: int = 1,
                     width: int = 4096, peak_tflops: float = 100.0,
                     bandwidth_gbps: float = 1000.0) -> dict:
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("Use at least one output token to observe prefill.")
    model = TinyAttentionLM(seed=seed)
    expected = generate(model, prompt, max_new_tokens, use_cache=False)
    cache, work = model.new_cache(), WorkCounter()
    generated, rows = [], []
    max_error = 0.0
    for step in range(max_new_tokens):
        tokens = list(prompt) if step == 0 else [generated[-1]]
        before, projected, pairs = len(cache.keys), work.projected_token_positions, work.attention_score_pairs
        logits = model.forward_cached(tokens, cache, work)[-1]
        error = max(abs(a - b) for a, b in zip(logits, expected.logits[step]))
        max_error = max(max_error, error)
        token = max(range(model.vocab_size), key=logits.__getitem__)
        rows.append({"step": step + 1, "phase": "prefill" if step == 0 else "decode",
                     "input_token_ids": tokens, "absolute_positions": list(range(before, before + len(tokens))),
                     "q_shape": [len(tokens), model.width], "kv_shape_after": [len(cache.keys), model.width],
                     "cache_length_before": before, "cache_length_after": len(cache.keys),
                     "projected_positions": work.projected_token_positions - projected,
                     "attention_score_pairs": work.attention_score_pairs - pairs,
                     "next_token_id": token, "next_token_logits": logits, "max_abs_logit_error": error})
        generated.append(token)
    estimates = {name: linear_cost(m, width, width, peak_tflops=peak_tflops,
                                  bandwidth_gbps=bandwidth_gbps)
                 for name, m in (("prefill", prefill_tokens), ("decode", decode_batch))}
    return {"schema_version": 1, "experiment": "part1_prefill_decode", "record_type": "measured_reference",
            "input": {"prompt_token_ids": prompt, "max_new_tokens": max_new_tokens, "seed": seed},
            "model": {"width": model.width, "vocab_size": model.vocab_size, "max_context": model.max_context},
            "steps": rows, "new_token_ids": generated,
            "cached_work": asdict(work), "full_recompute_work": asdict(expected.work),
            "gemm_estimates": estimates,
            "verification": {"max_abs_logit_error": max_error, "same_token_ids": generated == expected.token_ids,
                             "passed": max_error <= 1e-12 and generated == expected.token_ids},
            "limitations": ["CPU loops expose semantics, not parallel GPU scheduling or measured latency.",
                            "GEMM estimates use different, explicitly listed shapes from the tiny CPU model."]}
