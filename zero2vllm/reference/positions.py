"""Lesson 1.5: small positional, windowed, latent and recurrent math examples.

These are standalone identities, not implementations of production model families.
"""

import math

from .attention import linear, softmax


def rope(vector: list[float], position: int, base: float = 10000.0) -> list[float]:
    """Adjacent-pair convention; other frameworks may store half-split pairs."""
    if not vector or len(vector) % 2 or type(position) is not int or position < 0:
        raise ValueError("RoPE requires a nonempty even width and nonnegative integer position.")
    if isinstance(base, bool) or not math.isfinite(base) or base <= 0 or any(not math.isfinite(x) for x in vector):
        raise ValueError("RoPE inputs must be finite and base positive.")
    output = []
    for pair in range(len(vector) // 2):
        angle = position * base ** (-2 * pair / len(vector))
        a, b = vector[2 * pair:2 * pair + 2]
        output.extend([a * math.cos(angle) - b * math.sin(angle),
                       a * math.sin(angle) + b * math.cos(angle)])
    return output


def _validate_qkv(q, k, v):
    if not q or len(q) != len(k) or len(k) != len(v) or not q[0] or not v[0]:
        raise ValueError("Use matching nonempty Q/K/V sequences.")
    if any(len(row) != len(q[0]) for row in q + k) or any(len(row) != len(v[0]) for row in v):
        raise ValueError("Q/K widths must match; V width must be uniform.")
    if any(not math.isfinite(x) for row in q + k + v for x in row):
        raise ValueError("Q/K/V must be finite.")


def causal_attention(q, k, v, window: int | None = None, alibi_slope: float = 0.0) -> dict:
    _validate_qkv(q, k, v)
    if window is not None and (type(window) is not int or window <= 0):
        raise ValueError("window must be a positive integer, counting the current token.")
    if not math.isfinite(alibi_slope) or alibi_slope < 0:
        raise ValueError("ALiBi slope must be finite and nonnegative.")
    outputs, visible, weights = [], [], []
    for i, query in enumerate(q):
        indices = list(range(max(0, i - window + 1) if window else 0, i + 1))
        scores = [sum(a * b for a, b in zip(query, k[j])) / math.sqrt(len(query))
                  - alibi_slope * (i - j) for j in indices]
        probabilities = softmax(scores)
        outputs.append([sum(p * v[j][d] for p, j in zip(probabilities, indices)) for d in range(len(v[0]))])
        visible.append(indices)
        weights.append(probabilities)
    return {"visible_positions": visible, "attention_weights": weights,
            "outputs": outputs, "score_pairs": sum(map(len, visible))}


def linear_attention(q, k, v, recurrent: bool = True) -> list[list[float]]:
    """Positive phi(x)=max(x,0)+1; no softmax or approximation-equality claim."""
    _validate_qkv(q, k, v)
    phi = lambda row: [max(x, 0.0) + 1.0 for x in row]
    pq, pk = list(map(phi, q)), list(map(phi, k))
    width, value_width = len(q[0]), len(v[0])
    state = [[0.0] * value_width for _ in range(width)]
    norm = [0.0] * width
    outputs = []
    for i, query in enumerate(pq):
        if recurrent:
            for a in range(width):
                norm[a] += pk[i][a]
                for b in range(value_width):
                    state[a][b] += pk[i][a] * v[i][b]
            denominator = sum(a * b for a, b in zip(query, norm))
            outputs.append([sum(query[a] * state[a][b] for a in range(width)) / denominator
                            for b in range(value_width)])
        else:
            weights = [sum(a * b for a, b in zip(query, key)) for key in pk[:i + 1]]
            outputs.append([sum(weights[j] * v[j][b] for j in range(i + 1)) / sum(weights)
                            for b in range(value_width)])
    return outputs


def ssm_scan(inputs: list[float], a: float = 0.5, b: float = 1.0, c: float = 2.0) -> dict:
    if any(not math.isfinite(x) for x in [*inputs, a, b, c]):
        raise ValueError("SSM inputs and coefficients must be finite.")
    states, outputs, state = [], [], 0.0
    for x in inputs:
        state = a * state + b * x
        states.append(state)
        outputs.append(c * state)
    convolution = [sum(c * a ** (i - j) * b * inputs[j] for j in range(i + 1))
                   for i in range(len(inputs))]
    return {"inputs": inputs, "a": a, "b": b, "c": c, "states": states,
            "outputs": outputs, "convolution_reference": convolution}


def _matrix_error(a, b):
    return max(abs(x - y) for row, ref in zip(a, b) for x, y in zip(row, ref))


def structure_experiment(sequence_length: int = 8, window: int = 3) -> dict:
    if type(sequence_length) is not int or not 1 <= sequence_length <= 256:
        raise ValueError("Teaching sequence_length must be in [1, 256].")
    q = [[(i % 3) / 2.0, 0.5] for i in range(sequence_length)]
    k = [[0.2, (i % 4) / 3.0] for i in range(sequence_length)]
    v = [[float(i), float(i % 2)] for i in range(sequence_length)]
    full, sliding = causal_attention(q, k, v), causal_attention(q, k, v, window=window)
    alibi = causal_attention(q, k, v, alibi_slope=0.5)
    recurrent, explicit = linear_attention(q, k, v), linear_attention(q, k, v, recurrent=False)
    vector = [1.0, 2.0, -1.0, 0.5]
    rotated = rope(vector, 3)
    other = [0.5, -0.3, 0.7, 0.2]
    dot = lambda x, y: sum(a * b for a, b in zip(x, y))
    shifted_dot, base_dot = dot(rope(vector, 7), rope(other, 10)), dot(rope(vector, 0), rope(other, 3))
    norm_error = abs(dot(vector, vector) - dot(rotated, rotated))
    # A compressed content-key/value identity; decoupled RoPE is not implemented here.
    latent = [[0.4, -0.2, 0.1], [0.1, 0.3, -0.5]]
    wk, wv, query = [[1.0, 0.2], [-0.1, 0.5], [0.3, 0.4]], [[0.2, 0.6], [0.5, -0.3], [0.7, 0.1]], [0.3, -0.7]
    expanded_keys, expanded_values = [linear(x, wk) for x in latent], [linear(x, wv) for x in latent]
    absorbed_query = [dot(query, row) for row in wk]
    dense_scores, latent_scores = [dot(query, x) for x in expanded_keys], [dot(absorbed_query, x) for x in latent]
    weights = softmax([s / math.sqrt(len(query)) for s in dense_scores])
    dense_output = [sum(p * row[d] for p, row in zip(weights, expanded_values)) for d in range(2)]
    latent_context = [sum(p * row[d] for p, row in zip(weights, latent)) for d in range(3)]
    latent_output = linear(latent_context, wv)
    latent_error = max(abs(a - b) for a, b in zip(dense_scores + dense_output, latent_scores + latent_output))
    ssm = ssm_scan([1.0, 2.0, 3.0])
    errors = {"rope_norm": norm_error, "rope_relative_position": abs(shifted_dot - base_dot),
              "linear_recurrence": _matrix_error(recurrent, explicit), "latent_identity": latent_error,
              "ssm_recurrence": max(abs(a - b) for a, b in zip(ssm["outputs"], ssm["convolution_reference"]))}
    return {"schema_version": 1, "experiment": "part1_model_structures", "record_type": "measured_reference",
            "input": {"sequence_length": sequence_length, "window": window, "q": q, "k": k, "v": v},
            "rope": {"pair_convention": "adjacent", "vector": vector, "position": 3, "rotated": rotated,
                     "original_squared_norm": dot(vector, vector), "shifted_dot": shifted_dot, "base_dot": base_dot},
            "dense_attention": full, "sliding_window": sliding, "alibi_slope_0_5": alibi,
            "linear_attention": {"feature_map": "relu(x)+1", "recurrent": recurrent, "explicit": explicit,
                                 "persistent_state_elements": len(q[0]) * len(v[0]) + len(q[0])},
            "latent_content_identity": {"latent": latent, "wk": wk, "wv": wv, "q": query,
                                        "dense_scores": dense_scores, "latent_scores": latent_scores,
                                        "dense_output": dense_output, "latent_output": latent_output},
            "ssm": ssm, "verification": {"errors": errors, "absolute_tolerance": 1e-12,
                                         "passed": all(x <= 1e-12 for x in errors.values())},
            "limitations": ["Independent mathematical examples, not a full RoFormer, Mistral, DeepSeek, or Mamba implementation.",
                            "Linear feature-map attention changes the model; it is not numerically equal to softmax attention.",
                            "Window code restricts visible positions but still stores input lists; no ring-buffer GPU allocation.",
                            "No extrapolation quality, runtime complexity timing, or GPU performance has been measured."]}
