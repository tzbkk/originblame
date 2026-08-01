#!/usr/bin/env python3
"""Kernel reconcile benchmark.

Runs the two-phase reconcile benchmark on kernel OB datasets to measure
provenance recovery after data mutations.

For each kernel OB dataset found under ``benchmarks/results/pipeline_v2/``:
  1. Back up ``data.jsonl`` and ``.ob/`` to a temp directory.
  2. Mutate the data file (10% edit, 5% delete, 5% insert) with seed=42.
  3. Run ``ob reconcile`` with hash matching (Pass 1) and embedding matching
     (Pass 2).
  4. Record: hash_matched, semantic_matched, new_lines, orphans, duration_ms.
  5. Restore the backup (in a ``finally`` block).

Usage:
    PYTHONPATH=src python3 benchmarks/scripts/pipeline_v2/bench_kernel_reconcile.py
    PYTHONPATH=src python3 benchmarks/scripts/pipeline_v2/bench_kernel_reconcile.py --embedding-api http://localhost:1234/v1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

log = logging.getLogger("bench_kernel_reconcile")


# ═══════════════════════════════════════════════════════════════════════════════
# Mutation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _mutate_chars(text: str, rng: random.Random, char_pct: float = 0.10) -> str:
    """Replace *char_pct* of alphabetic characters with random lowercase letters."""
    chars = list(text)
    alpha_positions = [i for i, c in enumerate(chars) if c.isalpha()]
    n_edit = max(1, int(len(alpha_positions) * char_pct))
    for pos in rng.sample(alpha_positions, min(n_edit, len(alpha_positions))):
        chars[pos] = rng.choice("abcdefghijklmnopqrstuvwxyz")
    return "".join(chars)


def mutate_data(
    data_file: Path,
    seed: int = 42,
    edit_pct: float = 0.10,
    delete_pct: float = 0.05,
    insert_pct: float = 0.05,
) -> dict:
    """Mutate a data file in-place: edit / delete / insert lines.

    Returns mutation stats dict.
    """
    rng = random.Random(seed)

    with open(data_file, encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]

    original_lines = len(lines)

    # Classify each line
    categories: list[str] = []
    for _ in lines:
        r = rng.random()
        if r < edit_pct:
            categories.append("edit")
        elif r < edit_pct + delete_pct:
            categories.append("delete")
        else:
            categories.append("keep")

    edited_lines = categories.count("edit")
    deleted_lines = categories.count("delete")

    # Build synthetic insert lines from existing data
    num_inserts = max(1, int(original_lines * insert_pct))
    insert_lines = []
    for _ in range(num_inserts):
        src = rng.choice(lines)
        insert_lines.append(_mutate_chars(src, rng, char_pct=0.15))

    # Apply edits and deletes
    result_lines: list[str] = []
    for i, line in enumerate(lines):
        cat = categories[i]
        if cat == "delete":
            continue
        if cat == "edit":
            result_lines.append(_mutate_chars(line, rng))
        else:
            result_lines.append(line)

    # Insert new lines at random positions
    for ins in insert_lines:
        pos = rng.randint(0, len(result_lines))
        result_lines.insert(pos, ins)

    with open(data_file, "w", encoding="utf-8") as f:
        for line in result_lines:
            f.write(line + "\n")

    return {
        "original_lines": original_lines,
        "edited_lines": edited_lines,
        "deleted_lines": deleted_lines,
        "inserted_lines": num_inserts,
        "final_lines": len(result_lines),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Core benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def run_reconcile_bench(
    ob_dir: Path,
    embedding_api: str | None = None,
    model: str = "text-embedding-nomic-embed-text-v1.5",
    threshold: float = 0.85,
    seed: int = 42,
) -> dict:
    """Run reconcile benchmark on a single kernel OB dataset.

    Steps:
      1. Back up ``data.jsonl`` and ``.ob/``.
      2. Mutate data in-place.
      3. Run reconcile.
      4. Restore backup (always, in ``finally``).
    """
    ob_str = str(ob_dir)

    # ── Locate data file (HF or Datatrove layout) ────────────────────────────
    data_file = ob_dir / "jsonl" / "data.jsonl"
    if not data_file.is_file():
        log.warning("SKIP %s: no JSONL (Datatrove token-only output)", ob_dir.name)
        return {}

    # ── Backup ────────────────────────────────────────────────────────────────
    backup_dir = tempfile.mkdtemp(prefix="ob_reconcile_backup_")
    data_backup = Path(backup_dir) / "data.jsonl"
    ob_backup = Path(backup_dir) / ".ob"

    try:
        shutil.copy2(data_file, data_backup)
        shutil.copytree(ob_dir / ".ob", ob_backup)

        # ── Mutate ────────────────────────────────────────────────────────────
        log.info("  Mutating data...")
        mutation = mutate_data(data_file, seed=seed)
        log.info(
            "  Mutation: %d edited, %d deleted, %d inserted (%d → %d lines)",
            mutation["edited_lines"],
            mutation["deleted_lines"],
            mutation["inserted_lines"],
            mutation["original_lines"],
            mutation["final_lines"],
        )

        # ── Reconcile ─────────────────────────────────────────────────────────
        # Ensure ob_util is importable.
        # rust-originblame is typically checked out as a sibling or installed.
        # The import may also work directly if ob-util is pip-installed.
        try:
            from ob_util.reconcile import reconcile  # type: ignore[import-untyped]
        except ImportError:
            # Attempt to add the typical source paths.
            _repo_root = Path(__file__).resolve().parents[3]
            _candidates = [
                _repo_root / "rust-originblame" / "python" / "packages" / "ob-util" / "src",
                _repo_root / "rust-originblame" / "python" / "src",
            ]
            for p in _candidates:
                s = str(p)
                if s not in sys.path:
                    sys.path.insert(0, s)
            from ob_util.reconcile import reconcile  # type: ignore[import-untyped]

        t0 = time.perf_counter()

        result = reconcile(
            str(data_file),
            model=model,
            threshold=threshold,
            ob_dir=ob_dir,
            embedding_api=embedding_api,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return {
            "mutation": mutation,
            "hash_matched": result.hash_matched,
            "semantic_matched": result.semantic_matched,
            "new_lines": result.new_lines,
            "orphans": result.orphans,
            "duration_ms": round(elapsed_ms, 1),
            "total_matched": result.hash_matched + result.semantic_matched,
            "recovery_pct": round(
                (result.hash_matched + result.semantic_matched)
                / mutation["original_lines"]
                * 100,
                1,
            )
            if mutation["original_lines"] > 0
            else 0.0,
            "seed": seed,
            "model": model,
            "threshold": threshold,
        }

    except Exception as e:
        log.error("  Reconcile failed: %s", e)
        import traceback

        traceback.print_exc()
        return {"error": str(e)}

    finally:
        # ── Restore backup ────────────────────────────────────────────────────
        log.info("  Restoring backup...")
        try:
            shutil.copy2(data_backup, data_file)
        except Exception:
            log.error("  Failed to restore data.jsonl")

        ob_data_dir = ob_dir / ".ob"
        if ob_backup.exists():
            restore_tmp = ob_dir / ".ob_reconcile_restoring"
            if ob_data_dir.exists():
                ob_data_dir.rename(restore_tmp)
            try:
                ob_backup.rename(ob_data_dir)
                if restore_tmp.exists():
                    shutil.rmtree(restore_tmp)
            except Exception:
                if restore_tmp.exists() and not ob_data_dir.exists():
                    restore_tmp.rename(ob_data_dir)
                log.error("  Failed to restore .ob from backup")

        shutil.rmtree(backup_dir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Kernel reconcile benchmark")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("benchmarks/results/pipeline_v2"),
    )
    parser.add_argument(
        "--embedding-api",
        default="http://localhost:1234/v1",
        help="OpenAI-compatible embedding API URL",
    )
    parser.add_argument(
        "--model",
        default="text-embedding-nomic-embed-text-v1.5",
    )
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    results_dir = args.results_dir.resolve()

    # ── Discover kernel OB datasets ───────────────────────────────────────────
    kernel_dirs: list[Path] = []
    for p in sorted(results_dir.iterdir()):
        name = p.name
        if "kernel" not in name:
            continue
        if not (p / ".ob").is_dir():
            continue
        if not (p / "jsonl").is_dir():
            continue
        # Must have at least data.jsonl for reconcile
        if not (p / "jsonl" / "data.jsonl").is_file():
            continue
        kernel_dirs.append(p)

    if not kernel_dirs:
        log.error("No kernel OB datasets with data.jsonl found in %s", results_dir)
        sys.exit(1)

    log.info("Found %d kernel OB datasets", len(kernel_dirs))

    # ── Run benchmarks ────────────────────────────────────────────────────────
    all_results: dict[str, dict] = {}
    for ob_dir in kernel_dirs:
        name = ob_dir.name
        log.info("Running reconcile benchmark on %s ...", name)
        result = run_reconcile_bench(
            ob_dir,
            embedding_api=args.embedding_api,
            model=args.model,
            threshold=args.threshold,
            seed=args.seed,
        )
        all_results[name] = result

        if "error" not in result:
            log.info(
                "  hash=%d, semantic=%d, new=%d, orphans=%d, recovery=%.1f%%, time=%.0fms",
                result["hash_matched"],
                result["semantic_matched"],
                result["new_lines"],
                result["orphans"],
                result["recovery_pct"],
                result["duration_ms"],
            )

    # ── Save results ──────────────────────────────────────────────────────────
    output_file = results_dir / "kernel_reconcile_bench.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log.info("Results saved to %s", output_file)

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 90}")
    print("Kernel Reconcile Benchmark Results")
    print(f"{'=' * 90}")
    print(
        f"  {'Dataset':<35} {'Hash':>6} {'Semantic':>9} {'Recovery':>9} {'Time':>8}"
    )
    print(f"  {'-' * 35} {'-' * 6} {'-' * 9} {'-' * 9} {'-' * 8}")
    for name, r in all_results.items():
        if "error" in r:
            print(f"  {name:<35} ERROR: {r['error'][:30]}")
        else:
            print(
                f"  {name:<35} {r['hash_matched']:>6} {r['semantic_matched']:>9} "
                f"{r['recovery_pct']:>8.1f}% {r['duration_ms']:>7.0f}ms"
            )
    print()


if __name__ == "__main__":
    main()
