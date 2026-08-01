#!/usr/bin/env python3
from __future__ import annotations

"""CLI orchestrator for Datatrove pipelines with optional OB provenance.

Assembles and runs Datatrove pipelines using LocalPipelineExecutor with proper
OB integration (init before, clean after) and unified JSON logging.

Datatrove pipelines produce tokenized binary (.ds) output only — no JSONL.
When --with-ob is enabled, token-index entries are written for each document.

Pipeline configurations:
  OB:      [Reader, OBTrack, DocumentTokenizer]
  Baseline: [Reader, DocumentTokenizer]
  Skip tokenizer (kernel all): [Reader] (no OB, no tokenization)

Usage:
    python -m benchmarks.scripts.pipeline_v2.run_datatrove \
        --data-source zhwiki --scale 1k --with-ob
    python -m benchmarks.scripts.pipeline_v2.run_datatrove \
        --data-source kernel --scale all --no-ob --linux-dir /path/to/linux
"""

import argparse
import logging
import sys
import time
from pathlib import Path

log = logging.getLogger("run_datatrove")

from benchmarks.scripts.pipeline_v2._native_compat import clean as ob_clean


# ═══════════════════════════════════════════════════════════════════════════════
# Scale parsing
# ═══════════════════════════════════════════════════════════════════════════════

SCALE_MAP: dict[str, int] = {
    "1k": 1_000,
    "10k": 10_000,
    "100k": 100_000,
    "all": 1_000_000,
}


def parse_scale(scale_str: str) -> int:
    """Convert scale string like '1k', '10K', 'all' to integer."""
    key = scale_str.lower()
    if key in SCALE_MAP:
        return SCALE_MAP[key]
    # Handle numeric pass-through (e.g. "100k" is not in map but should work)
    raise ValueError(
        f"Unknown scale '{scale_str}'. "
        f"Valid options: {', '.join(sorted(SCALE_MAP.keys()))}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Datatrove pipeline with optional OB provenance",
    )
    parser.add_argument(
        "--data-source",
        choices=["zhwiki", "kernel"],
        required=True,
        help="Data source to process",
    )
    parser.add_argument(
        "--scale",
        required=True,
        help="Number of documents (1k, 10k, 100k, all)",
    )
    parser.add_argument(
        "--with-ob",
        action="store_true",
        default=True,
        dest="with_ob",
        help="Enable OB provenance tracking (default)",
    )
    parser.add_argument(
        "--no-ob",
        action="store_false",
        dest="with_ob",
        help="Disable OB provenance tracking",
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("benchmarks/raw_data"),
        help="Directory containing raw data dumps (for zhwiki)",
    )
    parser.add_argument(
        "--linux-dir",
        type=Path,
        default=Path("benchmarks/raw_data/linux"),
        help="Path to Linux kernel source tree (for kernel)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results/pipeline_v2"),
        help="Base output directory",
    )
    parser.add_argument(
        "--tokenizer",
        default="gpt2",
        help="Tokenizer name or path (default: gpt2)",
    )
    return parser


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline assembly
# ═══════════════════════════════════════════════════════════════════════════════


def build_pipeline(
    *,
    data_source: str,
    scale: int,
    with_ob: bool,
    raw_data_dir: Path,
    linux_dir: Path,
    run_output_dir: Path,
    tokenizer: str,
) -> tuple[list, object | None, object | None]:
    """Assemble pipeline steps. Returns (steps, reader_ref, ob_track_ref)."""
    from datatrove.pipeline.tokens import DocumentTokenizer

    from benchmarks.scripts.pipeline_v2.datatrove_steps import (
        KernelReader,
        MediaWikiReader,
        OBTrack,
        PreTokenizedDocumentTokenizer,
    )

    raw_data_dir = Path(raw_data_dir)
    linux_dir = Path(linux_dir)
    run_output_dir = Path(run_output_dir)

    from benchmarks.scripts.pipeline_v2.shared import load_tokenizer
    import tempfile
    tok = load_tokenizer(tokenizer)
    _tok_path = Path(tempfile.gettempdir()) / f"ob-bench-{tokenizer}-tokenizer.json"
    if not _tok_path.exists():
        tok.save(str(_tok_path))
    tokenizer_resolved = str(_tok_path)
    tokenizer_canonical = tokenizer

    packed_dir = run_output_dir / "packed"
    ob_dir = run_output_dir

    # Reader step
    if data_source == "zhwiki":
        reader_step = MediaWikiReader(
            raw_data_dir=raw_data_dir,
            scale=scale,
        )
    elif data_source == "kernel":
        reader_step = KernelReader(
            linux_dir=linux_dir,
            scale=scale,
        )
    else:
        raise ValueError(f"Unknown data source: {data_source}")

    # Datatrove's DocumentTokenizer accumulates metadata/index in memory,
    # which OOMs on large kernel runs (all files × ~5KB avg).
    # For kernel-all: skip packed binary entirely, keep OBTrack for token-index.
    skip_packed_binary = data_source == "kernel" and scale >= 100_000

    ob_track_step = None
    if with_ob:
        ob_track_step = OBTrack(
            ob_dir=ob_dir,
            tokenizer_name=tokenizer_canonical,
        )
        if skip_packed_binary:
            steps = [reader_step, ob_track_step]
        else:
            doc_tokenizer = PreTokenizedDocumentTokenizer(
                output_folder=str(packed_dir),
                tokenizer_name_or_path=tokenizer_resolved,
                save_filename="data",
            )
            steps = [reader_step, ob_track_step, doc_tokenizer]
    else:
        if skip_packed_binary:
            steps = [reader_step]
        else:
            doc_tokenizer = DocumentTokenizer(
                output_folder=str(packed_dir),
                tokenizer_name_or_path=tokenizer_resolved,
                save_filename="data",
            )
            steps = [reader_step, doc_tokenizer]

    return steps, reader_step, ob_track_step


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Resolve scale
    scale_str = args.scale.lower()
    scale_int = parse_scale(scale_str)

    # Resolve paths
    raw_data_dir = Path(args.raw_data_dir)
    linux_dir = Path(args.linux_dir)
    output_dir = Path(args.output_dir)

    # Build output directory name: datatrove-{source}-{scale}{-ob}
    ob_suffix = "-ob" if args.with_ob else ""
    run_dir_name = f"datatrove-{args.data_source}-{scale_str}{ob_suffix}"
    run_output_dir = output_dir / run_dir_name
    run_output_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "Pipeline: data_source=%s scale=%d with_ob=%s output=%s",
        args.data_source, scale_int, args.with_ob, run_output_dir,
    )

    # ── Build pipeline log ──────────────────────────────────────────────
    from benchmarks.scripts.pipeline_v2.shared import (
        PipelineLog,
        PipelineTimer,
        format_run_id,
        get_reproducibility_info,
        get_tokenizer_info,
        measure_memory,
        measure_storage,
        write_log,
    )

    run_id = format_run_id(
        pipeline="datatrove",
        data_source=args.data_source,
        scale=scale_str,
        with_ob=args.with_ob,
    )

    pipeline_log = PipelineLog(
        run_id=run_id,
        pipeline="datatrove",
        data_source=args.data_source,
        scale=scale_str,
        with_ob=args.with_ob,
    )

    errors: list[str] = []
    timer = PipelineTimer()

    # ── OB init ─────────────────────────────────────────────────────────
    if args.with_ob:
        try:
            import ob

            timer.start("ob_init")
            ob.init(ob_dir=run_output_dir, force=True)
            timer.stop()
            log.info("OB initialized at %s/.ob", run_output_dir)
        except Exception as exc:
            msg = f"OB init failed: {exc}"
            log.error(msg)
            errors.append(msg)
            pipeline_log.errors = errors
            write_log(pipeline_log, run_output_dir / "pipeline.log")
            return 1

    # ── Assemble pipeline ───────────────────────────────────────────────
    try:
        steps, reader_step, ob_track_step = build_pipeline(
            data_source=args.data_source,
            scale=scale_int,
            with_ob=args.with_ob,
            raw_data_dir=raw_data_dir,
            linux_dir=linux_dir,
            run_output_dir=run_output_dir,
            tokenizer=args.tokenizer,
        )
    except Exception as exc:
        msg = f"Pipeline assembly failed: {exc}"
        log.error(msg)
        errors.append(msg)
        pipeline_log.errors = errors
        write_log(pipeline_log, run_output_dir / "pipeline.log")
        return 1

    # ── Execute pipeline ────────────────────────────────────────────────
    from datatrove.executor import LocalPipelineExecutor

    timer.start("pipeline")
    executor = LocalPipelineExecutor(
        pipeline=steps, tasks=1, workers=1,
        skip_completed=False,
        log_folder=str(run_output_dir / "datatrove-logs"),
    )
    try:
        executor.run()
    except Exception as exc:
        msg = f"Pipeline execution failed: {exc}"
        log.error(msg)
        errors.append(msg)
    timer.stop()

    # ── OB clean ────────────────────────────────────────────────────────
    if args.with_ob:
        try:
            import ob

            timer.start("ob_clean")
            ob_clean(str(run_output_dir))
            timer.stop()
            log.info("OB cleaned")
        except Exception as exc:
            msg = f"OB clean failed: {exc}"
            log.warning(msg)
            errors.append(msg)

    # ── Collect stats ───────────────────────────────────────────────────
    wall_time_ms = timer.phases.get("pipeline", 0.0)
    wall_time_s = wall_time_ms / 1000.0 if wall_time_ms > 0 else 1.0

    # Reader stats — LocalPipelineExecutor deep-copies steps, so read from executor.pipeline
    _exec_reader = executor.pipeline[0] if executor.pipeline else reader_step
    reader_stats = getattr(_exec_reader, "_reader_stats", {})
    doc_count = reader_stats.get("pages_yielded", reader_stats.get("files_ok", 0))

    # OB stats — find OBTrack in executor's pipeline copy
    ob_metrics: dict = {}
    if ob_track_step is not None:
        _exec_ob = None
        for _s in executor.pipeline:
            if type(_s).__name__ == "OBTrack":
                _exec_ob = _s
                break
        if _exec_ob is None:
            _exec_ob = ob_track_step
        ob_metrics = dict(getattr(_exec_ob, "_ob_stats", {}))

    total_tokens = ob_metrics.get("total_tokens", 0)

    # Throughput
    throughput: dict = {}
    if wall_time_s > 0:
        throughput["docs_per_sec"] = round(doc_count / wall_time_s, 2)
        if total_tokens > 0:
            throughput["tokens_per_sec"] = round(total_tokens / wall_time_s, 2)
        throughput["wall_time_ms"] = round(wall_time_ms, 1)

    # Storage
    storage_bytes = measure_storage(run_output_dir)

    # Tokenizer info
    tokenizer_info: dict = {}
    try:
        tokenizer_info = get_tokenizer_info(args.tokenizer)
    except Exception as exc:
        log.warning("Could not get tokenizer info: %s", exc)
        tokenizer_info = {"name": args.tokenizer, "error": str(exc)}

    # Reproducibility
    reproducibility: dict = {}
    try:
        reproducibility = get_reproducibility_info()
    except Exception as exc:
        log.warning("Could not get reproducibility info: %s", exc)

    # Memory
    memory: dict = {}
    try:
        memory["max_rss_bytes"] = measure_memory()
    except Exception:
        pass

    # Output files
    output_files: list[str] = []
    for subdir in ["packed"]:
        p = run_output_dir / subdir
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file():
                    output_files.append(str(f.relative_to(run_output_dir)))

    # ── Fill PipelineLog ────────────────────────────────────────────────
    pipeline_log.timing = timer.phases
    pipeline_log.throughput = throughput
    pipeline_log.ob_metrics = ob_metrics
    pipeline_log.storage_bytes = storage_bytes
    pipeline_log.errors = errors
    pipeline_log.document_stats = reader_stats
    pipeline_log.tokenizer = tokenizer_info
    pipeline_log.output_files = output_files
    pipeline_log.reproducibility = reproducibility
    pipeline_log.memory = memory

    # ── Write log ───────────────────────────────────────────────────────
    write_log(pipeline_log, run_output_dir / "pipeline.log")
    log.info("Pipeline log written to %s/pipeline.log", run_output_dir)

    # ── Summary ─────────────────────────────────────────────────────────
    log.info("=== Pipeline Complete ===")
    log.info("  Run ID:    %s", run_id)
    log.info("  Docs:      %d", doc_count)
    log.info("  Tokens:    %d", total_tokens)
    log.info("  Wall time: %.1f ms", wall_time_ms)
    if throughput.get("docs_per_sec"):
        log.info("  Docs/s:    %.1f", throughput["docs_per_sec"])
    if throughput.get("tokens_per_sec"):
        log.info("  Tokens/s:  %.1f", throughput["tokens_per_sec"])

    if errors:
        log.warning("  Errors:    %d", len(errors))
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
