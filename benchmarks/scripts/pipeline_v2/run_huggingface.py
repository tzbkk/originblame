from __future__ import annotations

"""HuggingFace Datasets pipeline orchestrator for OriginBlame benchmarks.

Runs HuggingFace Datasets pipelines with optional OB provenance tracking.
Produces JSONL output only — no packed binary.

Usage:
    python benchmarks/scripts/pipeline_v2/run_huggingface.py \\
        --data-source zhwiki --scale 1k --with-ob
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure pipeline_v2 is importable from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.scripts.pipeline_v2._native_compat import clean as ob_clean
from pipeline_v2.hf_steps import (
    BaselineMapper,
    OBMapper,
    kernel_generator,
    stream_and_write,
    zhwiki_generator,
)
from pipeline_v2.shared import (
    PipelineLog,
    PipelineTimer,
    format_run_id,
    get_reproducibility_info,
    measure_memory,
    measure_storage,
    write_log,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Scale conversion
# ═══════════════════════════════════════════════════════════════════════════════

_SCALE_MAP: dict[str, int] = {
    "0.1k": 100,
    "0.1K": 100,
    "0.5k": 500,
    "0.5K": 500,
    "1k": 1_000,
    "1K": 1_000,
    "10k": 10_000,
    "10K": 10_000,
    "all": 1_000_000,
    "100k": 100_000,
    "100K": 100_000,
    "1m": 1_000_000,
    "1M": 1_000_000,
}


def parse_scale(scale_str: str) -> int:
    """Convert scale string like '1k' to integer."""
    if scale_str in _SCALE_MAP:
        return _SCALE_MAP[scale_str]
    # Try numeric
    try:
        return int(scale_str)
    except ValueError:
        pass
    raise ValueError(
        f"Unknown scale: {scale_str!r}. "
        f"Supported: {sorted(set(_SCALE_MAP.values()))}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="HuggingFace Datasets pipeline with optional OB provenance",
    )
    p.add_argument(
        "--data-source",
        choices=["zhwiki", "kernel"],
        required=True,
        help="Data source to process",
    )
    p.add_argument(
        "--scale",
        required=True,
        help="Scale: 1k, 10k, 100k, all (case-insensitive)",
    )
    p.add_argument(
        "--with-ob",
        action="store_true",
        default=True,
        dest="with_ob",
        help="Enable OB provenance tracking (default)",
    )
    p.add_argument(
        "--no-ob",
        action="store_false",
        dest="with_ob",
        help="Disable OB provenance tracking",
    )
    p.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("benchmarks/raw_data"),
        help="Path to raw data directory (default: benchmarks/raw_data)",
    )
    p.add_argument(
        "--linux-dir",
        type=Path,
        default=Path("benchmarks/raw_data/linux"),
        help="Path to Linux kernel repo (default: benchmarks/raw_data/linux)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results/pipeline_v2"),
        help="Output directory (default: benchmarks/results/pipeline_v2)",
    )
    return p


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline execution
# ═══════════════════════════════════════════════════════════════════════════════


def run_pipeline(
    data_source: str,
    scale_str: str,
    with_ob: bool,
    raw_data_dir: Path,
    linux_dir: Path,
    output_dir: Path,
) -> PipelineLog:
    """Run a single HuggingFace pipeline configuration and return the log."""
    scale_int = parse_scale(scale_str)

    # Output directory: huggingface-{data_source}-{scale}{-ob}/
    ob_suffix = "-ob" if with_ob else ""
    run_dir_name = f"huggingface-{data_source}-{scale_str.lower()}{ob_suffix}"
    run_output_dir = output_dir / run_dir_name
    run_output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_dir = run_output_dir / "jsonl"
    jsonl_path = jsonl_dir / "data.jsonl"

    run_id = format_run_id("huggingface", data_source, scale_str.lower(), with_ob)
    timer = PipelineTimer()
    errors: list[str] = []

    # Memory before
    mem_before = measure_memory()

    # OB init (before execution)
    if with_ob:
        import ob

        timer.start("ob_init")
        ob.init(ob_dir=run_output_dir, force=True)
        timer.stop()

    try:
        # ── Stream and write ─────────────────────────────────────────────
        timer.start("stream_write")

        if data_source == "zhwiki":
            gen = zhwiki_generator(
                raw_data_dir=raw_data_dir,
                scale=scale_int,
                license="CC-BY-SA-4.0",
            )
        elif data_source == "kernel":
            gen = kernel_generator(
                linux_dir=linux_dir,
                scale=scale_int,
            )
        else:
            raise ValueError(f"Unknown data source: {data_source}")

        ob_mapper: OBMapper | None = None
        mapper = None
        if with_ob:
            ob_mapper = OBMapper(run_output_dir)
            mapper = ob_mapper
        else:
            mapper = BaselineMapper()

        write_stats = stream_and_write(
            generator=gen,
            mapper=mapper,
            jsonl_path=jsonl_path,
        )
        timer.stop()

        doc_count = write_stats["doc_count"]

        # ── OB clean ──────────────────────────────────────────────────────
        if with_ob:
            timer.start("ob_clean")
            ob_clean(str(run_output_dir))
            timer.stop()

        # ── Collect stats ─────────────────────────────────────────────────
        phases = timer.phases
        wall_time_ms = sum(phases.values())

        total_bytes = write_stats.get("total_bytes", 0)
        wall_time_s = wall_time_ms / 1000.0
        docs_per_sec = round(doc_count / wall_time_s) if wall_time_s > 0 else 0
        bytes_per_sec = round(total_bytes / wall_time_s) if wall_time_s > 0 else 0

        ob_metrics: dict = {}
        if with_ob and ob_mapper is not None:
            ob_metrics = dict(ob_mapper.stats)

        storage = measure_storage(run_output_dir)

        mem_after = measure_memory()

        log = PipelineLog(
            run_id=run_id,
            pipeline="huggingface",
            data_source=data_source,
            scale=scale_str.lower(),
            with_ob=with_ob,
            timing={
                "wall_time_ms": round(wall_time_ms, 1),
                "stream_write_ms": round(phases.get("stream_write", 0), 1),
                "ob_init_ms": round(phases.get("ob_init", 0), 1) if with_ob else 0,
                "ob_clean_ms": round(phases.get("ob_clean", 0), 1) if with_ob else 0,
            },
            throughput={
                "docs_processed": doc_count,
                "docs_per_sec": docs_per_sec,
                "bytes_total": total_bytes,
                "bytes_per_sec": bytes_per_sec,
            },
            ob_metrics=ob_metrics,
            storage_bytes={
                "jsonl": storage["jsonl"],
                "ob": storage["ob"],
                "total": storage["total"],
            },
            errors=errors,
            document_stats={
                "doc_count": doc_count,
            },
            output_files=[
                str(jsonl_path),
            ],
            reproducibility=get_reproducibility_info(),
            memory={
                "rss_before_bytes": mem_before,
                "rss_after_bytes": mem_after,
                "rss_delta_bytes": mem_after - mem_before,
            },
        )

    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        phases = timer.phases
        wall_time_ms = sum(phases.values())

        log = PipelineLog(
            run_id=run_id,
            pipeline="huggingface",
            data_source=data_source,
            scale=scale_str.lower(),
            with_ob=with_ob,
            timing={"wall_time_ms": round(wall_time_ms, 1)},
            throughput={},
            ob_metrics={},
            storage_bytes={},
            errors=errors,
            document_stats={},
            output_files=[],
            reproducibility=get_reproducibility_info(),
            memory={
                "rss_before_bytes": mem_before,
                "rss_after_bytes": measure_memory(),
            },
        )

    # Write log
    write_log(log, run_output_dir / "pipeline.log")

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Pipeline: huggingface | Source: {data_source} | Scale: {scale_str}")
    print(f"OB: {with_ob} | Run ID: {run_id}")
    print(f"Output: {run_output_dir}")
    if not errors:
        print(
            f"Docs: {log.throughput.get('docs_processed', 0)} | "
            f"Bytes: {log.throughput.get('bytes_total', 0)} | "
            f"Wall: {log.timing.get('wall_time_ms', 0):.0f}ms"
        )
        print(
            f"Throughput: {log.throughput.get('bytes_per_sec', 0)} B/s | "
            f"Storage: {log.storage_bytes.get('total', 0)} bytes"
        )
    else:
        for err in errors:
            print(f"ERROR: {err}")
    print(f"{'=' * 60}\n")

    return log


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    run_pipeline(
        data_source=args.data_source,
        scale_str=args.scale,
        with_ob=args.with_ob,
        raw_data_dir=args.raw_data_dir,
        linux_dir=args.linux_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
