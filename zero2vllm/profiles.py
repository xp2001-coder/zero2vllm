"""Inspect deployment profiles and prepare an argv for an existing container.

This module never starts a service or a container. Versions are declarations
to be checked on hardware, not compatibility claims inferred from strings.
"""

from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
import re
import tomllib


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _known_fields(values: dict, allowed: set[str], section: str = "") -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        names = ", ".join(f"{section}.{key}" if section else key for key in unknown)
        raise ValueError(f"Unknown profile fields: {names}. "
                         "Schema 1 supports only the documented single-GPU baseline fields.")


def inspect_profile(path: Path) -> dict:
    with path.open("rb") as stream:
        profile = tomllib.load(stream)
    _known_fields(profile, {"schema_version", "backend", "hardware", "runtime", "model", "serving"})
    if type(profile.get("schema_version")) is not int or profile["schema_version"] != 1:
        raise ValueError("Only profile schema_version = 1 is supported.")
    backend = profile.get("backend")
    if backend not in ("cuda", "maca"):
        raise ValueError("Deployment backend must be explicitly cuda or maca.")
    required_fields = {
        "hardware": ["gpu_name", "gpu_memory_gib", "host_os", "cpu_arch"],
        "runtime": ["image", "python", "pytorch", "vllm", "driver"],
        "model": ["id", "revision", "path_in_container", "dtype"],
    }
    required_fields["runtime"] += (["cuda"] if backend == "cuda"
                                   else ["maca", "vllm_metax", "mcoplib"])
    missing: list[str] = []
    for section, fields in required_fields.items():
        values = profile.get(section)
        if not isinstance(values, dict):
            raise ValueError(f"Missing [{section}] table.")
        _known_fields(values, set(fields), section)
        for name in fields:
            value = values.setdefault(name, "")
            if not isinstance(value, str):
                raise ValueError(f"{section}.{name} must be a string.")
            if not value.strip():
                missing.append(f"{section}.{name}")

    memory = profile["hardware"]["gpu_memory_gib"].strip()
    if memory:
        try:
            size = Decimal(memory)
        except InvalidOperation:
            raise ValueError("hardware.gpu_memory_gib must contain a positive finite number in GiB.") from None
        if not size.is_finite() or size <= 0:
            raise ValueError("hardware.gpu_memory_gib must contain a positive finite number in GiB.")

    runtime = profile["runtime"]
    model = profile["model"]
    serving = profile.get("serving", {})
    if not isinstance(serving, dict):
        raise ValueError("serving must be a table.")
    _known_fields(serving, {"tensor_parallel_size", "max_model_len", "max_num_seqs",
                            "gpu_memory_utilization", "enforce_eager", "port"}, "serving")
    if type(serving.get("tensor_parallel_size")) is not int or serving["tensor_parallel_size"] != 1:
        raise ValueError("The first release requires tensor_parallel_size = 1.")
    max_len = _positive_int(serving.get("max_model_len"), "max_model_len")
    max_seqs = _positive_int(serving.get("max_num_seqs"), "max_num_seqs")
    port = _positive_int(serving.get("port"), "port")
    if port > 65535:
        raise ValueError("port must be at most 65535.")
    utilization = serving.get("gpu_memory_utilization")
    if type(utilization) not in (float, int) or not 0 < utilization < 1:
        raise ValueError("gpu_memory_utilization must be between 0 and 1.")
    if type(serving.get("enforce_eager")) is not bool:
        raise ValueError("enforce_eager must be a boolean.")
    if model["dtype"] and model["dtype"] not in ("float16", "bfloat16", "float32"):
        raise ValueError("Choose an explicit float16, bfloat16, or float32 baseline dtype.")
    if model["path_in_container"] and not PurePosixPath(model["path_in_container"]).is_absolute():
        raise ValueError("model.path_in_container must be an absolute Linux path.")
    if backend == "maca" and runtime.get("vllm") and runtime.get("vllm_metax"):
        if runtime["vllm"] != runtime["vllm_metax"]:
            raise ValueError("The declared vLLM and vLLM-MetaX versions must match.")

    # A digest makes the intended image immutable. Registry names are user supplied.
    gaps = list(missing)
    if runtime["image"] and not re.fullmatch(r".+@sha256:[0-9a-f]{64}", runtime["image"]):
        gaps.append("runtime.image needs its exact @sha256: digest")
    command = None
    if not gaps:
        command = ["vllm", "serve", model["path_in_container"],
                   "--served-model-name", "zero2vllm",
                   "--dtype", model["dtype"], "--tensor-parallel-size", "1",
                   "--max-model-len", str(max_len), "--max-num-seqs", str(max_seqs),
                   "--gpu-memory-utilization", str(utilization),
                   "--host", "127.0.0.1", "--port", str(port)]
        if serving["enforce_eager"]:
            command.append("--enforce-eager")
    return {
        "schema_version": 1,
        "backend": backend,
        "target_gpu": profile["hardware"]["gpu_name"],
        "configuration_complete": not gaps,
        "gpu_execution_tested": False,
        "missing_or_unlocked": gaps,
        "command_argv": command,
        "execution_context": "Inside the matching GPU container with exactly one full GPU visible.",
        "note": "Configuration inspection only. No command was executed. Run clients in the same container or network namespace.",
    }
