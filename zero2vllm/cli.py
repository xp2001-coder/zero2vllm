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
