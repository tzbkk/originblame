#!/usr/bin/env python3
from __future__ import annotations

"""Extract overhead metrics from pipeline_v2 benchmark logs.

Reads all ``*/pipeline.log`` files from a directory, pairs OB and baseline
runs by (pipeline, data_source, scale), then computes:

- Throughput overhead: (OB_wall_time - baseline_wall_time) / baseline_wall_time × 100%
- Storage overhead:   (OB_storage    - baseline_storage)  / baseline_storage  × 100%
- Scaling behavior:   how overhead changes with scale
- Cross-framework:    Datatrove vs HF overhead at same (data_source, scale)

Usage:
    python -m benchmarks.scripts.pipeline_v2.extract_metrics
    python -m benchmarks.scripts.pipeline_v2.extract_metrics --log-dir path/to/results
    python -m benchmarks.scripts.pipeline_v2.extract_metrics --output summary.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


# ═══════════════════════════════════════════════════════════════════════════════
# Log discovery and parsing
# ═══════════════════════════════════════════════════════════════════════════════

def discover_logs(log_dir: Path) -> list[Path]:
    """Find all ``{log_dir}/*/pipeline.log`` files."""
    if not log_dir.is_dir():
        print(f"WARNING: log directory not found: {log_dir}", file=sys.stderr)
        return []
    return sorted(log_dir.glob("*/pipeline.log"))


def parse_log(path: Path) -> dict[str, Any] | None:
    """Load a single pipeline.log, return None on parse error or if run had errors."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"WARNING: skipping {path}: {exc}", file=sys.stderr)
        return None

    # Skip runs that recorded errors
    if data.get("errors"):
        print(
            f"NOTE: skipping {path.name} (run has {len(data['errors'])} error(s))",
            file=sys.stderr,
        )
        return None

    return data


# ═══════════════════════════════════════════════════════════════════════════════
# Run grouping and pairing
# ═══════════════════════════════════════════════════════════════════════════════

def group_by_key(logs: list[dict[str, Any]]) -> dict[tuple, list[dict[str, Any]]]:
    """Group logs by (pipeline, data_source, scale)."""
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for entry in logs:
        key = (entry["pipeline"], entry["data_source"], entry["scale"])
        groups[key].append(entry)
    return groups


def find_pairs(
    groups: dict[tuple, list[dict[str, Any]]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Match OB runs with baseline runs within each group.

    Returns list of (baseline, ob) pairs.
    """
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for key, entries in sorted(groups.items()):
        ob_runs = [e for e in entries if e.get("with_ob")]
        baseline_runs = [e for e in entries if not e.get("with_ob")]

        if not ob_runs:
            print(
                f"NOTE: no OB run for "
                f"pipeline={key[0]} source={key[1]} scale={key[2]}",
                file=sys.stderr,
            )
            continue
        if not baseline_runs:
            print(
                f"NOTE: no baseline run for "
                f"pipeline={key[0]} source={key[1]} scale={key[2]}",
                file=sys.stderr,
            )
            continue

        # Use first match; if multiple, pick shortest wall_time as representative
        ob = min(ob_runs, key=lambda e: _wall_time(e))
        baseline = min(baseline_runs, key=lambda e: _wall_time(e))
        pairs.append((baseline, ob))

    return pairs


# ═══════════════════════════════════════════════════════════════════════════════
# Metric extraction helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _wall_time(entry: dict[str, Any]) -> float:
    """Extract wall_time_ms — HF puts it in timing, Datatrove in throughput."""
    t = entry.get("timing", {}).get("wall_time_ms")
    if t is not None:
        return t
    return entry.get("throughput", {}).get("wall_time_ms", float("inf"))


def _storage_total(entry: dict[str, Any]) -> int:
    """Extract total bytes from storage_bytes dict."""
    return entry.get("storage_bytes", {}).get("total", 0)


def _docs_per_sec(entry: dict[str, Any]) -> float:
    """Extract docs_per_sec from throughput dict."""
    return entry.get("throughput", {}).get("docs_per_sec", 0.0)


def _tokens_per_sec(entry: dict[str, Any]) -> float:
    """Extract tokens_per_sec from throughput dict."""
    return entry.get("throughput", {}).get("tokens_per_sec", 0.0)


def _pct_overhead(ob_value: float, baseline_value: float) -> float | None:
    """Compute percentage overhead. Returns None if baseline is zero."""
    if baseline_value == 0:
        return None
    return (ob_value - baseline_value) / baseline_value * 100.0


# ═══════════════════════════════════════════════════════════════════════════════
# Metric computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_pair_metrics(
    baseline: dict[str, Any], ob: dict[str, Any]
) -> dict[str, Any]:
    """Compute overhead metrics for a single OB/baseline pair."""
    b_wall = _wall_time(baseline)
    o_wall = _wall_time(ob)
    b_storage = _storage_total(baseline)
    o_storage = _storage_total(ob)
    b_dps = _docs_per_sec(baseline)
    o_dps = _docs_per_sec(ob)
    b_tps = _tokens_per_sec(baseline)
    o_tps = _tokens_per_sec(ob)

    throughput_overhead = _pct_overhead(o_wall, b_wall)
    storage_overhead = _pct_overhead(o_storage, b_storage)
    throughput_penalty = _pct_overhead(b_dps, o_dps)  # inverted: positive = OB slower

    return {
        "pipeline": baseline["pipeline"],
        "data_source": baseline["data_source"],
        "scale": baseline["scale"],
        "baseline_run_id": baseline.get("run_id", ""),
        "ob_run_id": ob.get("run_id", ""),
        "baseline_wall_time_ms": b_wall,
        "ob_wall_time_ms": o_wall,
        "baseline_storage_bytes": b_storage,
        "ob_storage_bytes": o_storage,
        "baseline_docs_per_sec": b_dps,
        "ob_docs_per_sec": o_dps,
        "baseline_tokens_per_sec": b_tps,
        "ob_tokens_per_sec": o_tps,
        "throughput_overhead_pct": throughput_overhead,
        "storage_overhead_pct": storage_overhead,
        "throughput_penalty_pct": throughput_penalty,
        "ob_metrics": ob.get("ob_metrics", {}),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Scaling behavior
# ═══════════════════════════════════════════════════════════════════════════════

SCALE_ORDER = ["1k", "10k", "100k", "all"]


def _scale_sort_key(scale: str) -> int:
    """Return numeric value for scale string for sorting."""
    mapping = {"1k": 1_000, "10k": 10_000, "100k": 100_000,
               "all": 1_000_000}
    return mapping.get(scale.lower(), 0)


def compute_scaling_behavior(
    pair_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    """Group overhead by scale, showing trends."""
    by_scale: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for m in pair_metrics:
        by_scale[m["scale"]].append(m)

    scaling: dict[str, Any] = {}
    for scale in sorted(by_scale, key=_scale_sort_key):
        entries = by_scale[scale]
        t_overheads = [e["throughput_overhead_pct"] for e in entries
                       if e["throughput_overhead_pct"] is not None]
        s_overheads = [e["storage_overhead_pct"] for e in entries
                       if e["storage_overhead_pct"] is not None]
        t_penalties = [e["throughput_penalty_pct"] for e in entries
                       if e["throughput_penalty_pct"] is not None]

        def _avg(vals: list[float]) -> float | None:
            return sum(vals) / len(vals) if vals else None

        scaling[scale] = {
            "num_pairs": len(entries),
            "throughput_overhead_avg_pct": _avg(t_overheads),
            "storage_overhead_avg_pct": _avg(s_overheads),
            "throughput_penalty_avg_pct": _avg(t_penalties),
        }

    return scaling


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-framework comparison
# ═══════════════════════════════════════════════════════════════════════════════

def compute_cross_framework(
    pair_metrics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compare Datatrove vs HF overhead at same (data_source, scale)."""
    by_combination: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for m in pair_metrics:
        combo = (m["data_source"], m["scale"])
        by_combination[combo][m["pipeline"]] = m

    comparisons: list[dict[str, Any]] = []
    for (source, scale), frameworks in sorted(by_combination.items()):
        if "datatrove" in frameworks and "huggingface" in frameworks:
            dt = frameworks["datatrove"]
            hf = frameworks["huggingface"]
            comparisons.append({
                "data_source": source,
                "scale": scale,
                "datatrove_throughput_overhead_pct": dt["throughput_overhead_pct"],
                "huggingface_throughput_overhead_pct": hf["throughput_overhead_pct"],
                "datatrove_storage_overhead_pct": dt["storage_overhead_pct"],
                "huggingface_storage_overhead_pct": hf["storage_overhead_pct"],
                "datatrove_baseline_tokens_per_sec": dt["baseline_tokens_per_sec"],
                "huggingface_baseline_tokens_per_sec": hf["baseline_tokens_per_sec"],
            })

    return comparisons


# ═══════════════════════════════════════════════════════════════════════════════
# Formatted table output
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:+.1f}%"


def _fmt_num(value: float | None, unit: str = "") -> str:
    if value is None:
        return "N/A"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M{unit}"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}k{unit}"
    return f"{value:.1f}{unit}"


def _fmt_bytes(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f} GB"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f} MB"
    if value >= 1_000:
        return f"{value / 1_000:.1f} KB"
    return f"{value} B"


def print_overhead_table(pair_metrics: list[dict[str, Any]]) -> None:
    """Print a formatted overhead summary table to stdout."""
    if not pair_metrics:
        print("No paired runs found.")
        return

    # Sort: framework, source, scale
    sorted_metrics = sorted(
        pair_metrics,
        key=lambda m: (m["pipeline"], m["data_source"], _scale_sort_key(m["scale"])),
    )

    # Header
    hdr = (
        f"{'Framework':<12} {'Source':<10} {'Scale':<7} "
        f"{'Time(OB)':<12} {'Time(base)':<12} "
        f"{'Thr.OH':<9} {'Stor.OH':<9} "
        f"{'tok/s(base)':<13} {'tok/s(OB)':<13} "
        f"{'Stor(base)':<12} {'Stor(OB)':<12}"
    )
    sep = "-" * len(hdr)

    print("\n=== Overhead Summary ===\n")
    print(hdr)
    print(sep)

    for m in sorted_metrics:
        t_oh = _fmt_pct(m["throughput_overhead_pct"])
        s_oh = _fmt_pct(m["storage_overhead_pct"])
        print(
            f"{m['pipeline']:<12} {m['data_source']:<10} {m['scale']:<7} "
            f"{_fmt_num(m['ob_wall_time_ms'], 'ms'):<12} "
            f"{_fmt_num(m['baseline_wall_time_ms'], 'ms'):<12} "
            f"{t_oh:<9} {s_oh:<9} "
            f"{_fmt_num(m['baseline_tokens_per_sec'], ''):<13} "
            f"{_fmt_num(m['ob_tokens_per_sec'], ''):<13} "
            f"{_fmt_bytes(m['baseline_storage_bytes']):<12} "
            f"{_fmt_bytes(m['ob_storage_bytes']):<12}"
        )

    print()


def print_cross_framework_table(comparisons: list[dict[str, Any]]) -> None:
    """Print cross-framework comparison table."""
    if not comparisons:
        print("No cross-framework comparisons available.")
        return

    sorted_comparisons = sorted(
        comparisons,
        key=lambda c: (c["data_source"], _scale_sort_key(c["scale"])),
    )

    print("\n=== Cross-Framework Comparison ===\n")
    hdr = (
        f"{'Source':<10} {'Scale':<7} "
        f"{'DT Thr.OH':<11} {'HF Thr.OH':<11} "
        f"{'DT Stor.OH':<11} {'HF Stor.OH':<11} "
        f"{'DT tok/s':<11} {'HF tok/s':<11}"
    )
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)

    for c in sorted_comparisons:
        print(
            f"{c['data_source']:<10} {c['scale']:<7} "
            f"{_fmt_pct(c['datatrove_throughput_overhead_pct']):<11} "
            f"{_fmt_pct(c['huggingface_throughput_overhead_pct']):<11} "
            f"{_fmt_pct(c['datatrove_storage_overhead_pct']):<11} "
            f"{_fmt_pct(c['huggingface_storage_overhead_pct']):<11} "
            f"{_fmt_num(c['datatrove_baseline_tokens_per_sec'], ''):<11} "
            f"{_fmt_num(c['huggingface_baseline_tokens_per_sec'], ''):<11}"
        )

    print()


def print_scaling_table(scaling: dict[str, Any]) -> None:
    """Print scaling behavior table."""
    if not scaling:
        print("No scaling data available.")
        return

    print("\n=== Scaling Behavior ===\n")
    hdr = (
        f"{'Scale':<7} {'Pairs':<7} "
        f"{'Thr.OH(avg)':<13} {'Stor.OH(avg)':<13} {'Penalty(avg)':<13}"
    )
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)

    for scale in sorted(scaling, key=_scale_sort_key):
        s = scaling[scale]
        print(
            f"{scale:<7} {s['num_pairs']:<7} "
            f"{_fmt_pct(s['throughput_overhead_avg_pct']):<13} "
            f"{_fmt_pct(s['storage_overhead_avg_pct']):<13} "
            f"{_fmt_pct(s['throughput_penalty_avg_pct']):<13}"
        )

    print()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_LOG_DIR = "benchmarks/results/pipeline_v2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract overhead metrics from pipeline_v2 benchmark logs.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path(DEFAULT_LOG_DIR),
        help=f"Directory containing run subdirectories (default: {DEFAULT_LOG_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write overhead_summary.json to this path (default: stdout only)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # 1. Discover and parse logs
    log_paths = discover_logs(args.log_dir)
    print(f"Found {len(log_paths)} pipeline.log file(s) in {args.log_dir}",
          file=sys.stderr)

    logs: list[dict[str, Any]] = []
    for p in log_paths:
        parsed = parse_log(p)
        if parsed is not None:
            logs.append(parsed)

    if not logs:
        print("No valid logs found. Exiting.", file=sys.stderr)
        return 1

    print(f"Loaded {len(logs)} valid run(s) "
          f"({len(log_paths) - len(logs)} skipped)", file=sys.stderr)

    # 2. Pair OB and baseline runs
    groups = group_by_key(logs)
    print(f"Found {len(groups)} unique (pipeline, source, scale) combination(s)",
          file=sys.stderr)

    pairs = find_pairs(groups)
    print(f"Matched {len(pairs)} OB/baseline pair(s)", file=sys.stderr)

    if not pairs:
        print("No paired runs found. Cannot compute overhead.", file=sys.stderr)
        return 1

    # 3. Compute metrics
    pair_metrics = [compute_pair_metrics(b, o) for b, o in pairs]

    # 4. Scaling behavior
    scaling = compute_scaling_behavior(pair_metrics)

    # 5. Cross-framework comparison
    cross_framework = compute_cross_framework(pair_metrics)

    # 6. Output
    summary = {
        "pair_metrics": pair_metrics,
        "scaling_behavior": scaling,
        "cross_framework": cross_framework,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote metrics to {args.output}", file=sys.stderr)

    # Always print tables to stdout
    print_overhead_table(pair_metrics)
    print_scaling_table(scaling)
    print_cross_framework_table(cross_framework)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
