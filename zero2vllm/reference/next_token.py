"""Inspectable next-token experiment for lesson 1.1.

Weights and five token pieces are hand-picked for arithmetic, not trained.
The trace reuses the reference math primitives and is compared with the existing
model's forward pass. The existing generation loop produces subsequent tokens.
"""

import math

from .attention import TinyAttentionLM, WorkCounter, generate, linear, softmax

VOCABULARY = ("我", "喜欢", "推理", "模型", "。")


def teaching_model() -> TinyAttentionLM:
    model = TinyAttentionLM(vocab_size=len(VOCABULARY), width=2, max_context=16)
    model.embedding = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
    # Position vectors are explicitly zero for the hand calculation.
    model.position = [[0.0, 0.0] for _ in range(model.max_context)]
    model.wq = [[1.0, 0.0], [0.0, 1.0]]
    model.wk = [[1.0, 0.0], [0.0, 1.0]]
    model.wv = [[1.0, 0.0], [0.0, 1.0]]
    model.wo = [[1.0, 0.0], [0.0, 1.0]]
    model.head = [[1.0, 0.0, 1.0, -1.0, 0.0],
                  [0.0, 1.0, 1.0, 0.0, -1.0]]
    return model


def encode_pieces(text: str) -> list[int]:
    pieces = text.split()
    if not pieces:
        raise ValueError("Prompt must contain at least one space-separated teaching token.")
    ids = []
    for piece in pieces:
        if piece not in VOCABULARY:
            raise ValueError(f"Unknown teaching token: {piece!r}. Use space-separated pieces from {VOCABULARY}.")
        ids.append(VOCABULARY.index(piece))
    return ids


def trace_forward(model: TinyAttentionLM, tokens: list[int]) -> dict:
    model.validate(tokens)
    embeddings = [list(model.embedding[token]) for token in tokens]
    positions = [list(model.position[i]) for i in range(len(tokens))]
    hidden = [[e + p for e, p in zip(embedding, position)]
              for embedding, position in zip(embeddings, positions)]
    queries = [linear(row, model.wq) for row in hidden]
    keys = [linear(row, model.wk) for row in hidden]
    values = [linear(row, model.wv) for row in hidden]
    scores = [[sum(q * k for q, k in zip(query, key)) / math.sqrt(model.width)
               for key in keys] for query in queries]
    # JSON null denotes a blocked position. Only visible scores enter softmax.
    masked_scores = [[value if j <= i else None for j, value in enumerate(row)]
                     for i, row in enumerate(scores)]
    attention = [softmax(row[:i + 1]) + [0.0] * (len(tokens) - i - 1)
                 for i, row in enumerate(scores)]
    contexts = [[sum(weight * value[d] for weight, value in zip(row, values))
                 for d in range(model.width)] for row in attention]
    projected = [linear(row, model.wo) for row in contexts]
    residual = [[x + y for x, y in zip(a, b)] for a, b in zip(hidden, projected)]
    logits = [linear(row, model.head) for row in residual]
    reference = model.forward_full(tokens, WorkCounter())
    error = max(abs(x - y) for row, expected in zip(logits, reference)
                for x, y in zip(row, expected))
    return {
        "shapes": {"token_ids": [len(tokens)], "embedding_weight": [model.vocab_size, model.width],
                   "hidden_q_k_v": [len(tokens), model.width],
                   "attention_scores_and_weights": [len(tokens), len(tokens)],
                   "lm_head_weight": [model.width, model.vocab_size],
                   "logits": [len(tokens), model.vocab_size]},
        "token_embeddings": embeddings, "position_embeddings": positions,
        "hidden": hidden, "q": queries, "k": keys, "v": values,
        "scaled_scores": scores, "masked_scores": masked_scores,
        "attention_weights": attention, "context": contexts,
        "attention_output": projected, "residual_hidden": residual,
        "all_position_logits": logits, "next_token_logits": logits[-1],
        "next_token_probabilities": softmax(logits[-1]),
        "max_abs_reference_error": error,
    }


def next_token_experiment(prompt: str = "我 喜欢", max_new_tokens: int = 3) -> dict:
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer for this lesson.")
    tokens = encode_pieces(prompt)
    model = teaching_model()
    # Validate the complete generation budget before producing a partial trace.
    generated = generate(model, tokens, max_new_tokens, use_cache=False)
    trace = trace_forward(model, tokens)
    steps = []
    prefix = list(tokens)
    for number, (token, logits) in enumerate(zip(generated.token_ids, generated.logits), 1):
        steps.append({"step": number, "input_token_ids": list(prefix),
                      "selected_logit_row": len(prefix) - 1,
                      "next_token_logits": logits, "next_token_id": token,
                      "next_token_piece": VOCABULARY[token]})
        prefix.append(token)
    error = trace["max_abs_reference_error"]
    first_error = max(abs(a - b) for a, b in zip(trace["next_token_logits"], generated.logits[0]))
    causal = all(weight == 0 for i, row in enumerate(trace["attention_weights"])
                 for weight in row[i + 1:])
    normalized = all(abs(sum(row) - 1.0) <= 1e-12 for row in trace["attention_weights"])
    return {
        "schema_version": 1, "experiment": "lesson_01_next_token",
        "record_type": "measured_reference",
        "model": {"kind": "hand_set_single_head_attention", "trained": False,
                  "vocabulary": list(VOCABULARY), "width": model.width,
                  "max_context": model.max_context, "embedding": model.embedding,
                  "position_vectors": "all zero for hand calculation",
                  "wq": model.wq, "wk": model.wk, "wv": model.wv,
                  "wo": model.wo, "lm_head": model.head},
        "input": {"text": prompt, "token_pieces": prompt.split(),
                  "token_ids": tokens, "max_new_tokens": max_new_tokens},
        "trace": trace,
        "generation": {"steps": steps, "new_token_ids": generated.token_ids,
                       "new_token_pieces": [VOCABULARY[token] for token in generated.token_ids],
                       "decoded_text": "".join(VOCABULARY[token] for token in prefix)},
        "verification": {"max_abs_reference_error": error,
                         "max_abs_first_step_error": first_error,
                         "future_attention_is_zero": causal,
                         "attention_rows_sum_to_one": normalized,
                         "absolute_tolerance": 1e-12,
                         "passed": error <= 1e-12 and first_error <= 1e-12 and causal and normalized},
        "limitations": ["Five hand-defined token pieces; whitespace splitting is not a real model tokenizer.",
                        "Hand-set, untrained weights and zero position embeddings; no language-quality claim.",
                        "Single head and residual only; no MLP, normalization, RoPE, or full Transformer stack.",
                        "Fixed generation length with greedy selection; no EOS or sampling strategy in this fixture.",
                        "CPU math only; no GPU, vLLM service, timing, or acceleration measurement."],
    }
