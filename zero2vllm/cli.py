"""Commands that work from a fresh checkout with only Python 3.11+."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from .environment import collect_environment
from .profiles import inspect_profile
from .reference.attention import TinyAttentionLM, generate
from .reference.next_token import next_token_experiment
from .reference.phases import phase_experiment
from .reference.kv_geometry import cache_experiment
from .reference.metrics import metrics_experiment
from .reference.positions import structure_experiment
from .reference.sampling import SamplingConfig, sampling_experiment
from .reference.quantization import quantization_experiment


def kv_experiment(prompt: list[int], max_new_tokens: int, seed: int) -> dict:
    full = generate(TinyAttentionLM(seed=seed), prompt, max_new_tokens, use_cache=False)
    cached = generate(TinyAttentionLM(seed=seed), prompt, max_new_tokens, use_cache=True)
    error = max((abs(x - y) for a, b in zip(full.logits, cached.logits)
                 for x, y in zip(a, b)), default=0.0)
    same = full.token_ids == cached.token_ids
    return {
        "schema_version": 1,
        "experiment": "cpu_kv_cache_reference",
        "record_type": "measured_reference",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": {"kind": "single_head_random_attention", "vocab_size": 16,
                  "width": 8, "max_context": 128, "seed": seed},
        "input": {"prompt_token_ids": prompt, "max_new_tokens": max_new_tokens},
        "full_recompute": asdict(full),
        "kv_cache": asdict(cached),
        "verification": {"same_token_ids": same, "max_abs_logit_error": error,
                         "absolute_tolerance": 1e-12, "passed": same and error <= 1e-12},
        "limitations": ["Random untrained weights; token IDs are not meaningful language.",
                        "Pure Python float reference; not PyTorch or vLLM GPU execution.",
                        "Work counters are not FLOPs, latency, or GPU performance."],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="zero2vllm")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Verify full vs cached attention on CPU")
    demo.add_argument("--prompt", default="1,2,3,4")
    demo.add_argument("--max-new-tokens", type=int, default=8)
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--output", type=Path)
    next_token = commands.add_parser("next-token-demo", help="Trace the hand-calculable model in lesson 1.1")
    next_token.add_argument("--prompt", default="我 喜欢")
    next_token.add_argument("--max-new-tokens", type=int, default=3)
    next_token.add_argument("--output", type=Path)
    phase = commands.add_parser("phase-demo", help="Trace prefill/decode work and estimate one GEMM",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    phase.add_argument("--prompt", default="1,2,3,4", help="Comma-separated tiny-model token IDs")
    phase.add_argument("--max-new-tokens", type=int, default=8, help="Output count for CPU trace")
    phase.add_argument("--seed", type=int, default=7, help="Tiny-model weight seed")
    phase.add_argument("--prefill-tokens", type=int, default=128, help="M for analytical prefill GEMM")
    phase.add_argument("--decode-batch", type=int, default=1, help="M for analytical decode GEMM")
    phase.add_argument("--width", type=int, default=4096, help="K=N for analytical GEMM, separate from CPU model")
    phase.add_argument("--peak-tflops", type=float, default=100.0, help="Hypothetical peak TFLOP/s, not measured")
    phase.add_argument("--bandwidth-gbps", type=float, default=1000.0, help="Hypothetical decimal GB/s, not Gbit/s")
    phase.add_argument("--output", type=Path)
    kv = commands.add_parser("kv-demo", help="Check cache chunks, GQA and KV payload estimates",
                             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    kv.add_argument("--prompt", default="1,2,3,4,5", help="Comma-separated tiny-model token IDs")
    kv.add_argument("--chunk-size", type=int, default=2, help="Tokens appended per prefill chunk")
    kv.add_argument("--seed", type=int, default=7, help="Tiny-model weight seed")
    kv.add_argument("--sequence-length", type=int, default=4096, help="Length for capacity estimate only")
    kv.add_argument("--batch-size", type=int, default=1, help="Equal-length requests in capacity estimate")
    kv.add_argument("--kv-heads", type=int, default=8, help="GQA KV heads, must divide 32")
    kv.add_argument("--output", type=Path)
    metrics = commands.add_parser("metrics-demo", help="Recompute metrics from same-clock per-token events",
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    metrics.add_argument("--input", type=Path, default=Path("examples/part1/request-events.json"), help="Event JSON; bundled input is simulated")
    metrics.add_argument("--ttft-slo", type=float, default=0.9, help="TTFT limit in seconds")
    metrics.add_argument("--tpot-slo", type=float, default=0.35, help="TPOT limit in seconds")
    metrics.add_argument("--output", type=Path)
    structure = commands.add_parser("structure-demo", help="Verify positional and recurrent model identities",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    structure.add_argument("--sequence-length", type=int, default=8, help="Teaching sequence length, at most 256")
    structure.add_argument("--window", type=int, default=3, help="Visible window including current position")
    structure.add_argument("--output", type=Path)
    sampling = commands.add_parser("sampling-demo", help="Explore token sampling, penalties, seeds and stopping",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sampling.add_argument("--prompt", default="我 喜欢", help="Space-separated teaching token pieces")
    sampling.add_argument("--max-new-tokens", type=int, default=8, help="Maximum generated IDs")
    sampling.add_argument("--min-new-tokens", type=int, default=0, help="IDs required before terminal IDs are allowed")
    sampling.add_argument("--seed", type=int, default=7, help="Request-local CPU RNG seed")
    sampling.add_argument("--draws", type=int, default=2000, help="Draws from the independent fixed four-token distribution")
    sampling.add_argument("--temperature", type=float, default=0.8, help="0 selects greedy decoding")
    sampling.add_argument("--top-k", type=int, default=0, help="0 disables top-k")
    sampling.add_argument("--top-p", type=float, default=0.9, help="Cumulative probability threshold in (0,1]")
    sampling.add_argument("--min-p", type=float, default=0.0, help="Threshold relative to largest probability in [0,1]")
    sampling.add_argument("--repetition-penalty", type=float, default=1.0, help="Positive sign-aware penalty for prompt and output IDs")
    sampling.add_argument("--presence-penalty", type=float, default=0.0, help="Subtract once for IDs in generated output")
    sampling.add_argument("--frequency-penalty", type=float, default=0.0, help="Subtract per occurrence in generated output")
    sampling.add_argument("--eos-token-id", type=int, help="Optional teaching EOS ID; none by default")
    sampling.add_argument("--stop-token-id", type=int, action="append", default=[], help="Repeat for multiple stopping IDs")
    sampling.add_argument("--stop", action="append", default=[], help="Stop string in generated text; may cross token boundaries")
    sampling.add_argument("--output", type=Path)
    doctor = commands.add_parser("doctor", help="Collect read-only environment inventory")
    doctor.add_argument("--output", type=Path)
    quant = commands.add_parser("quant-demo", help="Verify INT4 packing and Linear on CPU")
    quant.add_argument("--group-size", type=int, default=8)
    quant.add_argument("--seed", type=int, default=7)
    quant.add_argument("--output", type=Path)
    profile = commands.add_parser("profile", help="Inspect a CUDA/MACA deployment profile")
    profile.add_argument("path", type=Path)
    profile.add_argument("--output", type=Path)
    profile.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            prompt = [int(token.strip()) for token in args.prompt.split(",")]
            result = kv_experiment(prompt, args.max_new_tokens, args.seed)
            code = 0 if result["verification"]["passed"] else 1
        elif args.command == "next-token-demo":
            result = next_token_experiment(args.prompt, args.max_new_tokens)
            result["created_at_utc"] = datetime.now(timezone.utc).isoformat()
            code = 0 if result["verification"]["passed"] else 1
        elif args.command in ("phase-demo", "kv-demo", "metrics-demo", "structure-demo", "sampling-demo"):
            if args.command == "phase-demo":
                result = phase_experiment([int(x.strip()) for x in args.prompt.split(",")], args.max_new_tokens,
                                          args.seed, args.prefill_tokens, args.decode_batch, args.width,
                                          args.peak_tflops, args.bandwidth_gbps)
            elif args.command == "kv-demo":
                result = cache_experiment([int(x.strip()) for x in args.prompt.split(",")], args.chunk_size,
                                          args.seed, args.sequence_length, args.batch_size, args.kv_heads)
            elif args.command == "metrics-demo":
                result = metrics_experiment(json.loads(args.input.read_text(encoding="utf-8")), args.ttft_slo, args.tpot_slo)
            elif args.command == "structure-demo":
                result = structure_experiment(args.sequence_length, args.window)
            else:
                config = SamplingConfig(args.temperature, args.top_k, args.top_p, args.min_p,
                                        args.repetition_penalty, args.presence_penalty, args.frequency_penalty)
                result = sampling_experiment(args.prompt, config, args.max_new_tokens, args.seed, args.draws,
                                             args.eos_token_id, tuple(args.stop_token_id), tuple(args.stop), args.min_new_tokens)
            result["created_at_utc"] = datetime.now(timezone.utc).isoformat()
            code = 0 if result["verification"]["passed"] else 1
        elif args.command == "quant-demo":
            result = quantization_experiment(args.group_size, args.seed)
            result["created_at_utc"] = datetime.now(timezone.utc).isoformat()
            code = 0 if result["verification"]["passed"] else 1
        elif args.command == "doctor":
            result = collect_environment()
            code = 0
        else:
            result = inspect_profile(args.path)
            code = 2 if args.require_complete and not result["configuration_complete"] else 0
        payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload, encoding="utf-8")
        sys.stdout.write(payload)
        return code
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"zero2vllm: {error}", file=sys.stderr)
        return 2
