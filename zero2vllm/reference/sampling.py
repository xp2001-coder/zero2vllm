"""Lesson 1.6: explicit CPU sampling and stopping contracts, not a vLLM backend."""

from collections import Counter
from dataclasses import asdict, dataclass
import math
import random

from .attention import TinyAttentionLM, WorkCounter
from .next_token import VOCABULARY, encode_pieces, teaching_model


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 0.8
    top_k: int = 0
    top_p: float = 0.9
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    def validate(self):
        values = (self.temperature, self.top_p, self.min_p, self.repetition_penalty,
                  self.presence_penalty, self.frequency_penalty)
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in values):
            raise ValueError("Sampling parameters must be finite numbers.")
        if self.temperature < 0 or self.repetition_penalty <= 0:
            raise ValueError("temperature must be nonnegative; repetition_penalty must be positive.")
        if type(self.top_k) is not int or self.top_k < 0:
            raise ValueError("top_k must be a nonnegative integer; 0 disables it in this lesson.")
        if not 0 < self.top_p <= 1 or not 0 <= self.min_p <= 1:
            raise ValueError("top_p must be in (0, 1]; min_p in [0, 1].")
        if any(not -2 <= x <= 2 for x in (self.presence_penalty, self.frequency_penalty)):
            raise ValueError("presence_penalty and frequency_penalty must be in [-2, 2].")


def distribution(logits: list[float], config: SamplingConfig, prompt: list[int] = (),
                 generated: list[int] = (), blocked: set[int] = frozenset()) -> dict:
    """Penalty -> temperature -> top-k -> top-p -> min-p -> normalize.

    Repetition considers prompt+output; frequency/presence consider output only.
    Greedy uses penalized/masked scores and ignores probability filters.
    """
    config.validate()
    if not logits or any(type(x) not in (int, float) or math.isnan(x) or x == math.inf for x in logits):
        raise ValueError("Use nonempty logits with finite values or -inf masks; NaN/+inf are invalid.")
    if any(type(token) is not int or not 0 <= token < len(logits) for token in [*prompt, *generated, *blocked]):
        raise ValueError("History and blocked IDs must lie within the vocabulary.")
    counts, seen = Counter(generated), set(prompt) | set(generated)
    adjusted = []
    for token, value in enumerate(logits):
        if token in blocked or value == -math.inf:
            adjusted.append(-math.inf)
            continue
        if token in seen:
            value = value / config.repetition_penalty if value > 0 else value * config.repetition_penalty
        value -= config.presence_penalty * (counts[token] > 0) + config.frequency_penalty * counts[token]
        if not math.isfinite(value):
            raise ValueError("Penalty arithmetic overflowed; reduce input magnitude or penalty.")
        adjusted.append(value)
    order = sorted((i for i, x in enumerate(adjusted) if math.isfinite(x)), key=lambda i: (-adjusted[i], i))
    if not order:
        raise ValueError("All candidates are masked; sampling has no valid token.")
    probabilities = [0.0] * len(logits)
    if config.temperature == 0:
        kept = order[:1]
        probabilities[kept[0]] = 1.0
    else:
        largest = adjusted[order[0]]
        weights = {i: math.exp((adjusted[i] - largest) / config.temperature) for i in order}
        if config.top_k:
            order = order[:config.top_k]
        if config.top_p < 1.0:
            total = sum(weights[i] for i in order)
            kept, cumulative = [], 0.0
            for i in order:
                kept.append(i)
                cumulative += weights[i] / total
                if cumulative >= config.top_p:
                    break
            order = kept
        threshold = config.min_p * weights[order[0]]
        kept = [i for i in order if weights[i] >= threshold and weights[i] > 0]
        total = sum(weights[i] for i in kept)
        for i in kept:
            probabilities[i] = weights[i] / total
    return {"raw_logits": [None if x == -math.inf else x for x in logits],
            "penalized_logits": [None if x == -math.inf else x for x in adjusted],
            "kept_token_ids": kept, "probabilities": probabilities,
            "mode": "greedy" if config.temperature == 0 else "sample"}


def draw(probabilities: list[float], rng: random.Random) -> int:
    if not probabilities or any(not math.isfinite(p) or p < 0 for p in probabilities) or abs(sum(probabilities) - 1) > 1e-12:
        raise ValueError("Sampling probabilities must be nonnegative and sum to one.")
    u, total = rng.random(), 0.0
    for token, probability in enumerate(probabilities):
        total += probability
        if u < total:
            return token
    return max(i for i, p in enumerate(probabilities) if p > 0)  # final rounding gap


def generate_sampled(model: TinyAttentionLM, prompt: list[int], vocabulary: list[str] | tuple[str, ...],
                     config: SamplingConfig, max_new_tokens: int = 8, seed: int = 7,
                     eos_token_id: int | None = None, stop_token_ids: tuple[int, ...] = (),
                     stop_strings: tuple[str, ...] = (), min_new_tokens: int = 0) -> dict:
    config.validate()
    model.validate(prompt)
    if type(seed) is not int:
        raise ValueError("seed must be an integer.")
    if any(type(x) is not int or x < 0 for x in (max_new_tokens, min_new_tokens)) or min_new_tokens > max_new_tokens:
        raise ValueError("Require 0 <= min_new_tokens <= max_new_tokens with integer limits.")
    if len(prompt) + max(max_new_tokens - 1, 0) > model.max_context:
        raise ValueError("Generation budget exceeds model context capacity.")
    if len(vocabulary) != model.vocab_size or any(not isinstance(x, str) or not x for x in vocabulary):
        raise ValueError("Provide one nonempty string per vocabulary ID.")
    terminals = set(stop_token_ids)
    if eos_token_id is not None:
        terminals.add(eos_token_id)
    if any(type(x) is not int or not 0 <= x < model.vocab_size for x in terminals):
        raise ValueError("EOS and stop token IDs must be within the vocabulary.")
    if any(not isinstance(x, str) or not x for x in stop_strings):
        raise ValueError("Stop strings must be nonempty strings.")
    cache, work, rng = model.new_cache(), WorkCounter(), random.Random(seed)
    ids, steps, text, reason, stop_value = [], [], "", "length", None
    for step in range(max_new_tokens):
        inputs = list(prompt) if step == 0 else [ids[-1]]
        logits = model.forward_cached(inputs, cache, work)[-1]
        dist = distribution(logits, config, prompt, ids, terminals if len(ids) < min_new_tokens else set())
        token = dist["kept_token_ids"][0] if config.temperature == 0 else draw(dist["probabilities"], rng)
        ids.append(token)
        steps.append({"step": step + 1, "input_token_ids": inputs, "distribution": dist, "selected_token_id": token})
        if token in terminals:
            reason, stop_value = ("eos" if token == eos_token_id else "stop_token"), token
            break
        previous_length = len(text)
        text += vocabulary[token]
        matches = []
        if len(ids) >= min_new_tokens:
            for stop in stop_strings:
                start = text.find(stop, max(0, previous_length - len(stop) + 1))
                if start >= 0:
                    matches.append((start, stop))
        if matches:
            start, stop_value = min(matches)
            text, reason = text[:start], "stop_string"
            break
    return {"new_token_ids": ids, "generated_text": text, "finish_reason": reason, "stop_value": stop_value,
            "steps": steps, "cache_length": len(cache.keys), "work": asdict(work),
            "stop_contract": "Terminal IDs remain in new_token_ids but are omitted from text; stop strings and following text are removed. Prompt text is excluded. Buffered output only."}


def compare_logits(reference: list[float], candidate: list[float], atol: float = 1e-6, rtol: float = 1e-5) -> dict:
    if not reference or len(reference) != len(candidate):
        raise ValueError("Logit vectors must be nonempty and have equal lengths.")
    if any(not math.isfinite(x) for x in [*reference, *candidate, atol, rtol]) or atol < 0 or rtol < 0:
        raise ValueError("Logits and tolerances must be finite; tolerances nonnegative.")
    errors = [abs(a - b) for a, b in zip(reference, candidate)]
    limits = [atol + rtol * abs(x) for x in reference]
    argmax = lambda values: max(range(len(values)), key=values.__getitem__)
    return {"max_abs_error": max(errors), "atol": atol, "rtol": rtol,
            "all_within_tolerance": all(error <= limit for error, limit in zip(errors, limits)),
            "same_argmax": argmax(reference) == argmax(candidate),
            "reference_top2_gap": sorted(reference, reverse=True)[0] - sorted(reference, reverse=True)[1] if len(reference) > 1 else None}


def sampling_experiment(prompt: str = "我 喜欢", config: SamplingConfig = SamplingConfig(),
                        max_new_tokens: int = 8, seed: int = 7, draws: int = 2000,
                        eos_token_id: int | None = None, stop_token_ids: tuple[int, ...] = (),
                        stop_strings: tuple[str, ...] = (), min_new_tokens: int = 0) -> dict:
    if type(draws) is not int or not 1 <= draws <= 100000:
        raise ValueError("draws must be in [1, 100000].")
    tokens = encode_pieces(prompt)
    result = generate_sampled(teaching_model(), tokens, VOCABULARY, config, max_new_tokens,
                              seed, eos_token_id, stop_token_ids, stop_strings, min_new_tokens)
    # A fixed categorical distribution for a frequency experiment, independent of language generation.
    fixed = distribution([math.log(x) for x in (0.4, 0.3, 0.2, 0.1)], config)
    rng, counts = random.Random(seed), [0, 0, 0, 0]
    for _ in range(draws):
        counts[draw(fixed["probabilities"], rng)] += 1
    replay = generate_sampled(teaching_model(), tokens, VOCABULARY, config, max_new_tokens,
                              seed, eos_token_id, stop_token_ids, stop_strings, min_new_tokens)
    return {"schema_version": 1, "experiment": "part1_sampling", "record_type": "measured_reference",
            "input": {"prompt": prompt, "max_new_tokens": max_new_tokens, "min_new_tokens": min_new_tokens,
                      "seed": seed, "draws": draws, "eos_token_id": eos_token_id,
                      "stop_token_ids": list(stop_token_ids), "stop_strings": list(stop_strings)},
            "config": asdict(config), "generation": result,
            "frequency_experiment": {"base_probabilities": [0.4, 0.3, 0.2, 0.1], "filtered": fixed,
                                     "draws": draws, "counts": counts, "frequencies": [n / draws for n in counts]},
            "numerical_examples": {"same_token_wrong_scores": compare_logits([2.0, 1.0], [20.0, 1.0]),
                                   "close_scores_different_token": compare_logits([1.0, 1.0000001], [1.0000002, 1.0])},
            "verification": {"same_seed_replay": result == replay, "passed": result == replay and sum(counts) == draws},
            "limitations": ["Teaching sampler with an explicit filter order; not a replica of every vLLM version/backend.",
                            "Seed replay is checked only for this CPU implementation, not across hardware/frameworks.",
                            "Toy strings have direct token-piece concatenation; no real tokenizer or streaming stop buffer.",
                            "Finite-sample frequencies fluctuate; exact agreement with probabilities is not required."]}
