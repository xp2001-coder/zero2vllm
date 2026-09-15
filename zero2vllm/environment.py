"""Read-only environment inventory. Package presence is not GPU validation."""

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import platform
import shutil
import sys


def collect_environment() -> dict:
    packages = {}
    for name in ("torch", "vllm", "vllm-metax", "mcoplib", "transformers"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "schema_version": 1,
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "python_implementation": sys.implementation.name,
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "packages": packages,
        "commands_on_path": {name: shutil.which(name) is not None
                             for name in ("nvidia-smi", "mx-smi", "docker")},
        "gpu_execution_tested": False,
        "note": "Inventory only; no GPU execution, driver probe, hostname, or credentials collected.",
    }
