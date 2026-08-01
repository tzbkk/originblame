#!/usr/bin/env python3
"""Query benchmark for pipeline_v2 OB outputs.

Measures blame/show/revoke/purge/generate-set latency on each OB dataset
produced by the pipeline_v2 runs.  Uses the Rust native extension directly
via _ob_native.

Layout:
    results/pipeline_v2/{framework}-{source}-{scale}-ob/
      jsonl/data.jsonl
      .ob/
      pipeline.log

Usage:
    PYTHONPATH=src python3 benchmarks/scripts/pipeline_v2/bench_query.py
    PYTHONPATH=src python3 benchmarks/scripts/pipeline_v2/bench_query.py --runs 3
    PYTHONPATH=src python3 benchmarks/scripts/pipeline_v2/bench_query.py --filter huggingface-zhwiki-1m-ob
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import _ob_native as R

log = logging.getLogger("bench_query")


# ═══════════════════════════════════════════════════════════════════════════════
# Discovery
# ═══════════════════════════════════════════════════════════════════════════════

def find_ob_dirs(results_dir: Path) -> list[Path]:
    """Find all OB output directories (those containing .ob/).

    Discovers both HuggingFace (with jsonl/data.jsonl) and Datatrove
    (with packed/ or just .ob/) datasets.
    """
    dirs = []
    for p in sorted(results_dir.iterdir()):
        if not p.is_dir() or not (p / ".ob").is_dir():
            continue
        dirs.append(p)
    return dirs


def _is_hf_dataset(ob_dir: Path) -> bool:
    """Check if dataset has data.jsonl (HuggingFace pipeline output)."""
    return (ob_dir / "jsonl" / "data.jsonl").is_file()


def _load_lines(ob_dir: Path) -> tuple[list[str], str]:
    """Load data lines and return (lines, file_path_for_blame).

    Returns ([], "") for Datatrove datasets that have no JSONL.
    """
    import gzip

    plain = ob_dir / "jsonl" / "data.jsonl"
    if plain.is_file():
        with open(plain, encoding="utf-8") as f:
            return f.read().splitlines(), str(plain)

    # Datatrove: no JSONL output (token-only)
    return [], ""


# ═══════════════════════════════════════════════════════════════════════════════
# Core benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def _find_revoke_target(ob_dir: Path) -> tuple[str, str, int] | None:
    """Find the author with the most manifest records. Returns (name, email, count)."""
    ob_str = str(ob_dir)

    # Build section_hash → author_ids mapping
    section_authors: dict[str, list[str]] = {}
    for sec in R.shard_iterate_all(ob_str, "sections"):
        sh = sec.get("section_hash", "")
        if sh:
            section_authors[sh] = sec.get("authors", [])

    # Build author_id → {name, email} mapping
    author_info_map: dict[str, dict] = {}
    for a in R.shard_iterate_all(ob_str, "authors"):
        author_info_map[a.get("id", "")] = {"name": a.get("name", ""), "email": a.get("email", "")}

    # Count manifest records per author
    author_counts: dict[str, int] = {}
    for rec in R.shard_iterate_all(ob_str, "document-index"):
        for sh in rec.get("sources", []):
            for aid in section_authors.get(sh, []):
                author_counts[aid] = author_counts.get(aid, 0) + 1

    if not author_counts:
        return None

    target_aid = max(author_counts, key=lambda k: author_counts[k])
    info = author_info_map.get(target_aid, {})
    email = info.get("email", "")
    name = info.get("name", "")
    if not email:
        return None
    return (name, email, author_counts[target_aid])


def run_query_bench(ob_dir: Path, runs: int = 5) -> dict:
    """Run blame/show/revoke/purge benchmarks on a single OB dataset.

    For HuggingFace datasets (with data.jsonl): runs all record-level queries.
    For Datatrove datasets (token-only): skips record-level queries entirely.

    Returns a dict with all latency metrics averaged over `runs` iterations.
    """
    results: dict = {}
    ob_str = str(ob_dir)
    is_hf = _is_hf_dataset(ob_dir)

    results["pipeline_type"] = "huggingface" if is_hf else "datatrove"

    # ── Basic counts ──────────────────────────────────────────────────────────
    manifest_records = 0
    for _ in R.shard_iterate_all(ob_str, "document-index"):
        manifest_records += 1

    all_lines, data_path = _load_lines(ob_dir)
    data_rows = len(all_lines)

    results["data_rows"] = data_rows
    results["manifest_records"] = manifest_records

    # Datatrove datasets: no record-level queries
    if not is_hf:
        log.info("  Datatrove dataset (token-only output): skipping record-level queries")
        results["row_coverage"] = 0.0
        for field in ["blame_mean_ms", "show_mean_ms", "show_idx_ms",
                       "revoke_ms", "purge_ms", "purge_author_idx_ms"]:
            results[field] = None
        return results

    results["row_coverage"] = round(manifest_records / data_rows * 100, 1) if data_rows > 0 else 0.0

    # ── Sample lines for blame ────────────────────────────────────────────────
    total_lines = len(all_lines)

    step = max(1, total_lines // 10)
    line_numbers = list(range(1, min(total_lines + 1, step * 10 + 1), step))[:10]
    if total_lines > 0:
        line_numbers = [min(ln, total_lines) for ln in line_numbers]

    # ── Sample author for show ────────────────────────────────────────────────
    all_authors = list(R.shard_iterate_all(ob_str, "authors"))
    sample_name = all_authors[0].get("name", "") if all_authors else None

    # ── Revoke target (find once, reuse across runs) ──────────────────────────
    revoke_target = _find_revoke_target(ob_dir)
    if revoke_target:
        results["revoke_target"] = revoke_target[0]
        results["revoke_affected_rows"] = revoke_target[2]
        affected_pct = (revoke_target[2] / manifest_records * 100) if manifest_records > 0 else 0.0
        results["revoke_affected_pct"] = round(affected_pct, 1)

    # ── Timed operations (multiple runs) ──────────────────────────────────────
    avg_fields = [
        "blame_mean_ms",
        "show_mean_ms",
        "show_idx_ms",
        "revoke_ms",
        "purge_ms",
        "purge_author_idx_ms",
    ]
    has_index = (ob_dir / ".ob" / "index").exists()
    all_runs: list[dict] = []

    for run_idx in range(1, runs + 1):
        run_result: dict = {}

        # Warm-up (3 blame + 1 show + 1 show_idx + 1 purge_idx)
        for ln in line_numbers[:3]:
            line_content = all_lines[ln - 1].rstrip("\n")
            R.blame(ob_str, data_path, line_content)
        if sample_name:
            R.show_by_author(ob_str, sample_name)
        if revoke_target and has_index:
            try:
                R.purge_by_author_indexd(ob_str, revoke_target[1], data_path, True)
            except Exception:
                pass

        # Blame (10 samples)
        blame_times = []
        for ln in line_numbers:
            line_content = all_lines[ln - 1].rstrip("\n")
            t0 = time.perf_counter_ns()
            R.blame(ob_str, data_path, line_content)
            blame_times.append((time.perf_counter_ns() - t0) / 1e6)
        run_result["blame_mean_ms"] = round(sum(blame_times) / len(blame_times), 3)

        # Show (3 samples)
        show_times = []
        if sample_name:
            for _ in range(3):
                t0 = time.perf_counter_ns()
                R.show_by_author(ob_str, sample_name)
                show_times.append((time.perf_counter_ns() - t0) / 1e6)
        run_result["show_mean_ms"] = round(sum(show_times) / len(show_times), 3) if show_times else None

        # Show with index (3 samples) — same function, index auto-used when present
        show_idx_times = []
        if sample_name and has_index:
            for _ in range(3):
                t0 = time.perf_counter_ns()
                R.show_by_author(ob_str, sample_name)
                show_idx_times.append((time.perf_counter_ns() - t0) / 1e6)
        run_result["show_idx_ms"] = round(sum(show_idx_times) / len(show_idx_times), 3) if show_idx_times else None

        # Revoke + Purge
        if revoke_target:
            target_name, target_email, affected_rows = revoke_target
            if target_email:
                t0 = time.perf_counter_ns()
                R.revoke_by_author(ob_str, target_email)
                run_result["revoke_ms"] = round((time.perf_counter_ns() - t0) / 1e6, 3)

                for ln in line_numbers[:3]:
                    line_content = all_lines[ln - 1].rstrip("\n")
                    R.blame(ob_str, data_path, line_content)

                t0 = time.perf_counter_ns()
                try:
                    R.purge_revoked(ob_str, data_path, True)
                    run_result["purge_ms"] = round((time.perf_counter_ns() - t0) / 1e6, 3)
                except Exception as e:
                    log.warning("Purge failed: %s", e)
                    run_result["purge_ms"] = None

                # Purge --author with index (3 samples, cold)
                purge_idx_times = []
                if has_index:
                    try:
                        R.purge_by_author_indexd(ob_str, target_email, data_path, True)
                    except Exception:
                        pass
                    for _ in range(3):
                        t0 = time.perf_counter_ns()
                        try:
                            R.purge_by_author_indexd(ob_str, target_email, data_path, True)
                            purge_idx_times.append((time.perf_counter_ns() - t0) / 1e6)
                        except Exception:
                            pass
                run_result["purge_author_idx_ms"] = (
                    round(sum(purge_idx_times) / len(purge_idx_times), 3)
                    if purge_idx_times else None
                )

                R.revoke_by_author(ob_str, target_email)

        if "revoke_ms" not in run_result:
            run_result["revoke_ms"] = None
            run_result["purge_ms"] = None
            run_result["purge_author_idx_ms"] = None

        all_runs.append(run_result)
        log.info(
            "  Run %d/%d: blame=%.3fms  show=%.3fms  show_idx=%.3fms  revoke=%.3fms  purge=%.3fms  purge_a_idx=%.3fms",
            run_idx, runs,
            run_result.get("blame_mean_ms", 0),
            run_result.get("show_mean_ms", 0) or 0,
            run_result.get("show_idx_ms", 0) or 0,
            run_result.get("revoke_ms", 0) or 0,
            run_result.get("purge_ms", 0) or 0,
            run_result.get("purge_author_idx_ms", 0) or 0,
        )

    # ── Average results ───────────────────────────────────────────────────────
    for field in avg_fields:
        values = [r[field] for r in all_runs if r.get(field) is not None]
        if values:
            results[field] = round(sum(values) / len(values), 3)
            results[f"{field}_min"] = round(min(values), 3)
            results[f"{field}_max"] = round(max(values), 3)
            if len(values) > 1:
                mean = sum(values) / len(values)
                variance = sum((v - mean) ** 2 for v in values) / len(values)
                results[f"{field}_stdev"] = round(variance ** 0.5, 3)
            else:
                results[f"{field}_stdev"] = 0.0
        else:
            results[field] = None

    results["num_runs"] = runs
    results["runs"] = [{k: v for k, v in r.items() if k in avg_fields} for r in all_runs]

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Token-level benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def run_token_bench(ob_dir: Path, tokenizer: str = "gpt2", runs: int = 3) -> dict:
    """Measure show --tokenizer, revoke --tokenizer, generate-set latency."""
    results: dict = {}
    ob_str = str(ob_dir)
    has_token_index = (ob_dir / ".ob" / f"token-index.{tokenizer}").exists() or \
                      (ob_dir / ".ob" / f"token-index.{tokenizer}.bin").exists()
    if not has_token_index:
        results["token_show_ms"] = None
        results["token_revoke_ms"] = None
        results["generate_set_ms"] = None
        return results

    all_authors = list(R.shard_iterate_all(ob_str, "authors"))
    sample = all_authors[0] if all_authors else None
    sample_name = sample.get("name", "") if sample else None
    if not sample_name:
        results["token_show_ms"] = None
        results["token_revoke_ms"] = None
        results["generate_set_ms"] = None
        return results

    # show --tokenizer with binary index (3 runs, averaged)
    show_times = []
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        R.show_by_author_token(ob_str, sample_name, tokenizer)
        show_times.append((time.perf_counter_ns() - t0) / 1e6)
    results["token_show_ms"] = round(sum(show_times) / len(show_times), 3)

    # show --tokenizer WITHOUT binary index (hide .bin, 3 runs, averaged)
    tok_bin = ob_dir / ".ob" / f"token-index.{tokenizer}.bin"
    tok_bin_bak = None
    show_no_idx_times = []
    if tok_bin.exists():
        tok_bin_bak = tok_bin.with_suffix(".bin.bak_bench")
        tok_bin.rename(tok_bin_bak)
    try:
        for _ in range(runs):
            t0 = time.perf_counter_ns()
            R.show_by_author_token(ob_str, sample_name, tokenizer)
            show_no_idx_times.append((time.perf_counter_ns() - t0) / 1e6)
    finally:
        if tok_bin_bak and tok_bin_bak.exists():
            if tok_bin.exists():
                tok_bin.unlink()
            tok_bin_bak.rename(tok_bin)
    results["token_show_no_idx_ms"] = round(sum(show_no_idx_times) / len(show_no_idx_times), 3) if show_no_idx_times else None

    gen_times = []
    try:
        tok_info_str = R.token_index_build_binary_index(ob_str, tokenizer)
        total_entries = int(tok_info_str.split(",")[0].split("=")[1])
        bitmask = [True] * total_entries
    except Exception as e:
        log.warning("token-index-build-binary-index failed: %s", e)
        total_entries = 0
        bitmask = []

    if bitmask:
        for _ in range(runs):
            with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
                tmp_path = tmp.name
            t0 = time.perf_counter_ns()
            try:
                R.write_forget_set(tmp_path, bitmask)
                gen_times.append((time.perf_counter_ns() - t0) / 1e6)
            except Exception as e:
                log.warning("generate-set failed: %s", e)
            finally:
                os.unlink(tmp_path)
    results["generate_set_ms"] = round(sum(gen_times) / len(gen_times), 3) if gen_times else None

    # revoke --tokenizer (single run, destructive — measure + undo)
    t0 = time.perf_counter_ns()
    R.revoke_by_author_token(ob_str, sample_name, tokenizer)
    results["token_revoke_ms"] = round((time.perf_counter_ns() - t0) / 1e6, 3)
    # Undo
    R.revoke_by_author_token(ob_str, sample_name, tokenizer)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Query benchmark: blame/show/revoke/purge on pipeline_v2 OB outputs"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("benchmarks/results/pipeline_v2"),
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Only run on directories matching this substring",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="gpt2",
    )
    parser.add_argument(
        "--skip-token",
        action="store_true",
    )
    args = parser.parse_args()

    # Logging
    ts = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    log_dir = args.results_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"query_bench_{ts}.log"

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )

    # Discover OB dirs
    ob_dirs = find_ob_dirs(args.results_dir)
    if args.filter:
        ob_dirs = [d for d in ob_dirs if args.filter in d.name]

    if not ob_dirs:
        log.error("No OB directories found in %s", args.results_dir)
        return 1

    log.info("Found %d OB datasets to benchmark", len(ob_dirs))

    # System info
    all_results: dict = {
        "timestamp": ts,
        "type": "pipeline_v2_query_bench",
        "num_runs": args.runs,
        "tokenizer": args.tokenizer,
        "system": {
            "python": sys.version,
            "os": sys.platform,
            "cpu_cores": os.cpu_count(),
        },
    }
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True,
        ).stdout.strip()
        all_results["system"]["git_commit"] = git_commit
    except Exception:
        pass

    total_start = time.monotonic()

    for ob_dir in ob_dirs:
        name = ob_dir.name

        if not (ob_dir / ".ob").is_dir():
            log.warning("No .ob/ in %s, skipping", name)
            continue

        is_hf = _is_hf_dataset(ob_dir)
        pipeline_type = "HF" if is_hf else "DT"

        log.info("=" * 70)
        log.info("Dataset: %s (%s)", name, pipeline_type)
        log.info("=" * 70)

        t0 = time.monotonic()
        query_results = run_query_bench(ob_dir, runs=args.runs)

        if not args.skip_token:
            token_results = run_token_bench(ob_dir, tokenizer=args.tokenizer, runs=3)
            query_results.update(token_results)

        query_results["wall_seconds"] = round(time.monotonic() - t0, 1)

        all_results[name] = query_results

        is_dt = query_results.get("pipeline_type") == "datatrove"
        if is_dt:
            log.info(
                ">>> %s (DT): record-level=N/A  tok_show=%s  tok_show_no_idx=%s  gen_set=%s",
                name,
                f'{query_results.get("token_show_ms") or 0:.3f}ms' if query_results.get("token_show_ms") is not None else "N/A",
                f'{query_results.get("token_show_no_idx_ms") or 0:.3f}ms' if query_results.get("token_show_no_idx_ms") is not None else "N/A",
                f'{query_results["generate_set_ms"]:.3f}ms' if query_results.get("generate_set_ms") is not None else "N/A",
            )
        else:
            log.info(
                ">>> %s (HF): blame=%.3fms  show=%.3fms  show_idx=%.3fms  revoke=%.3fms  purge=%.3fms  purge_a_idx=%.3fms  tok_show=%s  tok_show_no_idx=%s  gen_set=%s",
                name,
                query_results.get("blame_mean_ms", 0),
                query_results.get("show_mean_ms", 0) or 0,
                query_results.get("show_idx_ms", 0) or 0,
                query_results.get("revoke_ms", 0) or 0,
                query_results.get("purge_ms", 0) or 0,
                query_results.get("purge_author_idx_ms", 0) or 0,
                f'{query_results.get("token_show_ms") or 0:.3f}ms' if query_results.get("token_show_ms") is not None else "N/A",
                f'{query_results.get("token_show_no_idx_ms") or 0:.3f}ms' if query_results.get("token_show_no_idx_ms") is not None else "N/A",
                f'{query_results["generate_set_ms"]:.3f}ms' if query_results.get("generate_set_ms") is not None else "N/A",
            )

    total_elapsed = time.monotonic() - total_start
    all_results["total_wall_seconds"] = round(total_elapsed, 1)

    # Write results
    results_file = args.results_dir / f"query_bench_{ts}.json"
    results_file.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    latest = args.results_dir / "query_bench_latest.json"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    shutil.copy2(results_file, latest)

    log.info("Results written to %s", results_file)
    log.info("Total wall time: %.1fs", total_elapsed)

    # Summary table
    log.info("")
    log.info("=" * 140)
    log.info("%-35s %4s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s",
             "Dataset", "Type", "rows", "blame", "show", "show_idx", "revoke", "purge", "p_a_idx", "tok_show", "tok_no_idx", "gen_set")
    log.info("-" * 140)
    for ob_dir in ob_dirs:
        name = ob_dir.name
        r = all_results.get(name, {})
        if not r:
            continue
        def _fmt(key):
            v = r.get(key)
            return f"{v:7.3f}ms" if v is not None else "    N/A"
        is_dt = r.get("pipeline_type") == "datatrove"
        ptype = "DT" if is_dt else "HF"
        if is_dt:
            log.info("%-35s %4s %8d %8s %8s %8s %8s %8s %8s %s %s %s",
                     name, ptype,
                     r.get("data_rows", 0),
                     "N/A", "N/A", "N/A", "N/A", "N/A", "N/A",
                     _fmt("token_show_ms"),
                     _fmt("token_show_no_idx_ms"),
                     _fmt("generate_set_ms"),
                     )
        else:
            log.info("%-35s %4s %8d %7.3fms %7.3fms %7.3fms %7.3fms %7.3fms %7.3fms %s %s %s",
                     name, ptype,
                     r.get("data_rows", 0),
                     r.get("blame_mean_ms", 0),
                     r.get("show_mean_ms", 0) or 0,
                     r.get("show_idx_ms", 0) or 0,
                     r.get("revoke_ms", 0) or 0,
                     r.get("purge_ms", 0) or 0,
                     r.get("purge_author_idx_ms", 0) or 0,
                     _fmt("token_show_ms"),
                     _fmt("token_show_no_idx_ms"),
                     _fmt("generate_set_ms"),
                     )
    log.info("=" * 140)

    return 0


if __name__ == "__main__":
    sys.exit(main())
