"""Lesson 1.4: recompute request metrics from explicit, same-clock token events."""

import math


def percentile(values: list[float], probability: float) -> float | None:
    """Linear interpolation on sorted samples, index=(n-1)*p, including n=1."""
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Percentile probability must be in [0, 1].")
    if not values:
        return None
    if any(not math.isfinite(x) for x in values):
        raise ValueError("Percentile samples must be finite.")
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def _summary(values: list[float]) -> dict:
    return {"count": len(values), "mean": sum(values) / len(values) if values else None,
            **{f"p{int(p * 100)}": percentile(values, p) for p in (0.5, 0.95, 0.99)}}


def metrics_experiment(data: dict, ttft_slo_s: float = 0.9, tpot_slo_s: float = 0.35) -> dict:
    if any(isinstance(x, bool) or not math.isfinite(x) or x < 0 for x in (ttft_slo_s, tpot_slo_s)):
        raise ValueError("SLO limits must be finite and nonnegative seconds.")
    if not isinstance(data, dict) or data.get("schema_version") != 1 or data.get("timestamp_kind") != "per_token_same_clock":
        raise ValueError("Use schema 1 and per_token_same_clock events; stream chunks are not token timestamps.")
    if data.get("record_type") not in ("simulation", "measured_gpu"):
        raise ValueError("Input provenance must be simulation or measured_gpu; keep device evidence with measured data.")
    requests = data.get("requests")
    if not isinstance(requests, list) or not requests:
        raise ValueError("At least one request is required.")
    identifiers, rows = set(), []
    for request in requests:
        if not isinstance(request, dict):
            raise ValueError("Each request must be a JSON object.")
        identifier = request["id"]
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError("Request IDs must be nonempty and unique.")
        identifiers.add(identifier)
        arrival, started, finished = (request[key] for key in ("arrival_s", "work_started_s", "finished_s"))
        tokens = request["token_times_s"]
        status = request["status"]
        if not isinstance(tokens, list) or status not in ("ok", "error"):
            raise ValueError("Each request needs token_times_s and status ok/error.")
        times = [arrival, started, finished, *tokens]
        if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in times):
            raise ValueError("Timestamps must be finite nonnegative numbers in seconds.")
        if not arrival <= started <= finished or any(not started <= x <= finished for x in tokens):
            raise ValueError("Timestamps must satisfy arrival <= start <= tokens <= finish.")
        if any(a > b for a, b in zip(tokens, tokens[1:])):
            raise ValueError("Token timestamps must be nondecreasing.")
        if status == "ok" and not tokens:
            raise ValueError("This lesson requires successful requests to contain at least one output token.")
        itl = [b - a for a, b in zip(tokens, tokens[1:])]
        ttft = tokens[0] - arrival if tokens else None
        tpot = (tokens[-1] - tokens[0]) / (len(tokens) - 1) if len(tokens) >= 2 else None
        meets = status == "ok" and ttft <= ttft_slo_s and (tpot is None or tpot <= tpot_slo_s)
        rows.append({"id": identifier, "status": status, "output_tokens": len(tokens),
                     "queue_s": started - arrival, "e2e_s": finished - arrival,
                     "ttft_s": ttft, "tpot_s": tpot, "itl_s": itl, "meets_slo": meets})
    begin = min(r["arrival_s"] for r in requests)
    end = max(r["finished_s"] for r in requests)
    duration = end - begin
    if duration <= 0:
        raise ValueError("Benchmark window must have positive duration.")
    successes = [r for r in rows if r["status"] == "ok"]
    good = sum(r["meets_slo"] for r in rows)
    output_tokens = sum(r["output_tokens"] for r in successes)
    return {"schema_version": 1, "experiment": "part1_request_metrics", "record_type": data["record_type"],
            "input": data, "request_metrics": rows,
            "window": {"begin_s": begin, "end_s": end, "duration_s": duration,
                       "definition": "First arrival through last finish, including failures and queueing."},
            "counts": {"requests": len(rows), "successful": len(successes), "failed": len(rows) - len(successes),
                       "successful_output_tokens": output_tokens, "good_requests": good},
            "rates": {"success_fraction": len(successes) / len(rows),
                      "successful_requests_per_s": len(successes) / duration,
                      "output_tokens_per_s": output_tokens / duration,
                      "goodput_requests_per_s": good / duration, "good_fraction_all_requests": good / len(rows)},
            "slo": {"ttft_s": ttft_slo_s, "tpot_s": tpot_slo_s,
                    "one_token_rule": "TTFT applies; TPOT is undefined and imposes no condition."},
            "distributions": {"ttft_s": _summary([r["ttft_s"] for r in successes]),
                              "tpot_s": _summary([r["tpot_s"] for r in successes if r["tpot_s"] is not None]),
                              "itl_s": _summary([x for r in successes for x in r["itl_s"]]),
                              "e2e_success_s": _summary([r["e2e_s"] for r in successes]),
                              "queue_all_s": _summary([r["queue_s"] for r in rows])},
            "percentile_method": "Sorted samples with linear interpolation at (n-1)*p; latency distributions exclude failed requests except queue_all_s.",
            "verification": {"passed": True, "meaning": "Input schema, clocks and event ordering validated; not proof of measured GPU provenance."},
            "limitations": ["Default data is a hand-written simulation, not a benchmark measurement.",
                            "The caller must provide genuine per-token events and same-clock queue timestamps.",
                            "Recomputing reported measured data does not verify its hardware provenance."]}
