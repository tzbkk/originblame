from __future__ import annotations

"""Shared utilities for pipeline v2 benchmarks."""

import json
import os
import platform
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from resource import getrusage, RUSAGE_SELF
from typing import Any


# ═══════════════════════════════════════════════════════════════════════════════
# Type aliases
# ═══════════════════════════════════════════════════════════════════════════════

AuthorsList = list[dict[str, str]]


# ═══════════════════════════════════════════════════════════════════════════════
# Tokenizer loader
# ═══════════════════════════════════════════════════════════════════════════════

def load_tokenizer(tokenizer_name_or_path: str):
    """Load tokenizer from local cache, local file, or HuggingFace Hub."""
    from tokenizers import Tokenizer

    local_path = Path(tokenizer_name_or_path)
    if local_path.exists() and local_path.is_file():
        return Tokenizer.from_file(str(local_path))

    cache_candidates = [
        Path.home() / ".cache" / "huggingface" / "tokenizers" / f"{tokenizer_name_or_path}.json",
        Path.home() / ".cache" / "huggingface" / "hub" / f"models--{tokenizer_name_or_path}" / "snapshots" / "tokenizer.json",
    ]
    for cached in cache_candidates:
        if cached.exists():
            return Tokenizer.from_file(str(cached))

    try:
        return Tokenizer.from_pretrained(tokenizer_name_or_path)
    except Exception:
        return Tokenizer.from_file(tokenizer_name_or_path)


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

MAX_CONTEXT_TOKENS = 4096
SKIP_DIRS = frozenset({"Documentation", "scripts", "tools", "samples"})


# ═══════════════════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineLog:
    run_id: str
    pipeline: str
    data_source: str
    scale: str
    with_ob: bool
    timing: dict[str, float] = field(default_factory=dict)
    throughput: dict[str, Any] = field(default_factory=dict)
    ob_metrics: dict[str, Any] = field(default_factory=dict)
    storage_bytes: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    document_stats: dict[str, Any] = field(default_factory=dict)
    tokenizer: dict[str, Any] = field(default_factory=dict)
    output_files: list[str] = field(default_factory=list)
    reproducibility: dict[str, str] = field(default_factory=dict)
    memory: dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# Timer
# ═══════════════════════════════════════════════════════════════════════════════

class PipelineTimer:
    """Context manager that records per-phase wall time in milliseconds."""

    def __init__(self) -> None:
        self._phases: dict[str, float] = {}
        self._current_phase: str | None = None
        self._start: float = 0.0

    def start(self, phase: str) -> None:
        self._current_phase = phase
        self._start = time.perf_counter()

    def stop(self) -> None:
        if self._current_phase is None:
            raise RuntimeError("No phase started")
        elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        self._phases[self._current_phase] = elapsed_ms
        self._current_phase = None

    @property
    def phases(self) -> dict[str, float]:
        return dict(self._phases)

    def __enter__(self) -> PipelineTimer:
        return self

    def __exit__(self, *args: Any) -> None:
        if self._current_phase is not None:
            self.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# Storage measurement
# ═══════════════════════════════════════════════════════════════════════════════

def measure_storage(output_dir: Path) -> dict[str, int]:
    """Walk directory tree, return byte counts for jsonl/packed/ob subdirectories."""
    result: dict[str, int] = {"jsonl": 0, "packed": 0, "ob": 0, "total": 0}

    for subdir, key in [("jsonl", "jsonl"), ("packed", "packed"), (".ob", "ob")]:
        path = output_dir / subdir
        if path.is_dir():
            total = 0
            for root, _dirs, files in os.walk(path):
                for f in files:
                    total += os.path.getsize(os.path.join(root, f))
            result[key] = total

    result["total"] = result["jsonl"] + result["packed"] + result["ob"]
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Memory measurement
# ═══════════════════════════════════════════════════════════════════════════════

def measure_memory() -> int:
    """Return ru_maxrss in bytes (platform-normalized)."""
    maxrss = getrusage(RUSAGE_SELF).ru_maxrss
    # On Linux, ru_maxrss is in KB; on macOS it's in bytes.
    if platform.system() == "Darwin":
        return maxrss
    return maxrss * 1024


# ═══════════════════════════════════════════════════════════════════════════════
# Log I/O
# ═══════════════════════════════════════════════════════════════════════════════

def write_log(log: PipelineLog, path: Path) -> None:
    """JSON dump with indent=2."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "run_id": log.run_id,
        "pipeline": log.pipeline,
        "data_source": log.data_source,
        "scale": log.scale,
        "with_ob": log.with_ob,
        "timing": log.timing,
        "throughput": log.throughput,
        "ob_metrics": log.ob_metrics,
        "storage_bytes": log.storage_bytes,
        "errors": log.errors,
        "document_stats": log.document_stats,
        "tokenizer": log.tokenizer,
        "output_files": log.output_files,
        "reproducibility": log.reproducibility,
        "memory": log.memory,
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════════
# Reproducibility
# ═══════════════════════════════════════════════════════════════════════════════

def get_reproducibility_info() -> dict[str, str]:
    """Collect hostname, git hash, and dependency versions."""
    info: dict[str, str] = {
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
    }

    try:
        import ob
        info["ob_git_hash"] = getattr(ob, "__git_hash__", "unknown")
    except ImportError:
        info["ob_git_hash"] = "not_installed"

    try:
        import datatrove
        info["datatrove_version"] = getattr(datatrove, "__version__", "unknown")
    except ImportError:
        info["datatrove_version"] = "not_installed"

    try:
        import datasets
        info["datasets_version"] = datasets.__version__
    except ImportError:
        info["datasets_version"] = "not_installed"

    try:
        import tokenizers
        info["tokenizers_version"] = tokenizers.__version__
    except ImportError:
        info["tokenizers_version"] = "not_installed"

    return info


# ═══════════════════════════════════════════════════════════════════════════════
# Tokenizer info
# ═══════════════════════════════════════════════════════════════════════════════

def get_tokenizer_info(tokenizer_name: str) -> dict[str, Any]:
    """Load tokenizer via tokenizers library, return name + vocab_size."""
    tok = load_tokenizer(tokenizer_name)
    return {
        "name": tokenizer_name,
        "vocab_size": tok.get_vocab_size(),
        "library": "tokenizers",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Run ID
# ═══════════════════════════════════════════════════════════════════════════════

def format_run_id(pipeline: str, data_source: str, scale: str, with_ob: bool) -> str:
    """Generate run ID like 'datatrove-zhwiki-1k-ob-20260426-143022'."""
    ob_suffix = "-ob" if with_ob else ""
    ts = time.strftime("%Y%m%d-%H%M%S")
    return f"{pipeline}-{data_source}-{scale}{ob_suffix}-{ts}"
