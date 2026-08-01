#!/usr/bin/env python3
"""Kernel revocation precision analysis.

Measures file-level vs record-level revocation precision on kernel OB datasets.
Demonstrates that even at file-level granularity, ob provenance avoids
dataset-level over-deletion.

For each kernel OB dataset, the script:
1. Counts all authors and their contribution sizes (manifest records per author)
2. Selects top-N authors plus preset notables (torvalds, akpm, gregkh)
3. For each selected author, computes:
   - manifest_count: line-level contributions (ob-precise)
   - file_count: number of unique files touched
   - over_deletion_ratio: total_data_rows / file_count
4. Prints a summary table and saves JSON results

Layout:
    results/pipeline_v2/{framework}-kernel-{scale}-ob/
      jsonl/data.jsonl           (HuggingFace)
      jsonl/*.jsonl.gz           (Datatrove)
      .ob/

Usage:
    PYTHONPATH=src python3 benchmarks/scripts/pipeline_v2/bench_kernel_revocation.py
    PYTHONPATH=src python3 benchmarks/scripts/pipeline_v2/bench_kernel_revocation.py --filter datatrove
    PYTHONPATH=src python3 benchmarks/scripts/pipeline_v2/bench_kernel_revocation.py --top-n 15
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import _ob_native as R

log = logging.getLogger("bench_kernel_revocation")

# Preset authors to always include in analysis (matched case-insensitively
# against both name and email).
PRESET_AUTHORS = {"torvalds", "akpm", "gregkh"}


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _has_jsonl(ob_dir: Path) -> bool:
    """Check if dataset has data.jsonl (HuggingFace pipeline output)."""
    return (ob_dir / "jsonl" / "data.jsonl").is_file()


def _count_data_rows(ob_dir: Path) -> int:
    """Count data rows, handling both HF and Datatrove layouts."""
    plain = ob_dir / "jsonl" / "data.jsonl"
    if plain.is_file():
        with open(plain, encoding="utf-8") as f:
            return sum(1 for _ in f)

    # Datatrove: one or more .jsonl.gz files
    total = 0
    for gz in sorted((ob_dir / "jsonl").glob("*.jsonl.gz")):
        with gzip.open(gz, "rt", encoding="utf-8") as f:
            total += sum(1 for _ in f)
    return total


def _matches_preset(name: str, email: str) -> bool:
    """Check if an author matches any preset name substring."""
    combined = f"{name} {email}".lower()
    return any(p in combined for p in PRESET_AUTHORS)


def _select_top_authors(authors: list[dict], top_n: int) -> list[dict]:
    """Select top-N authors by manifest count, plus any preset authors."""
    selected = list(authors[:top_n])
    selected_emails = {a["email"] for a in selected}

    for a in authors[top_n:]:
        if _matches_preset(a["name"], a["email"]) and a["email"] not in selected_emails:
            selected.append(a)
            selected_emails.add(a["email"])

    # Re-sort by manifest_count descending
    selected.sort(key=lambda a: a["manifest_count"], reverse=True)
    return selected


# ═══════════════════════════════════════════════════════════════════════════════
# Core analysis
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_author_precision(ob_dir: Path) -> dict:
    """Analyze revocation precision for all authors in a kernel OB dataset.

    Returns dict with:
        - total_data_rows: int
        - total_manifest_records: int
        - unique_authors: int
        - authors: list of {name, email, share_pct, manifest_count,
                    file_count, over_deletion_ratio}
    """
    ob_str = str(ob_dir)

    # Count data rows (kernel: each row = one C/H source file)
    total_data_rows = _count_data_rows(ob_dir)

    # Build section_hash → author_ids mapping
    section_authors: dict[str, list[str]] = {}
    for sec in R.shard_iterate_all(ob_str, "sections"):
        sh = sec.get("section_hash", "")
        if sh:
            section_authors[sh] = sec.get("authors", [])

    # Build author_id → {name, email} mapping
    author_info_map: dict[str, dict] = {}
    for a in R.shard_iterate_all(ob_str, "authors"):
        author_info_map[a.get("id", "")] = {
            "name": a.get("name", ""),
            "email": a.get("email", ""),
        }

    # Count manifest records per author AND track which files each author touches.
    # For kernel: manifest records are line-level attributions (git blame),
    # and line_hash identifies the file (one record per file).
    author_manifest_counts: dict[str, int] = {}
    author_files: dict[str, set[str]] = {}

    for rec in R.shard_iterate_all(ob_str, "document-index"):
        line_hash = rec.get("line_hash", "")
        for sh in rec.get("sources", []):
            for aid in section_authors.get(sh, []):
                author_manifest_counts[aid] = author_manifest_counts.get(aid, 0) + 1
                if aid not in author_files:
                    author_files[aid] = set()
                author_files[aid].add(line_hash)

    total_manifest_records = sum(author_manifest_counts.values())

    # Build per-author results
    authors: list[dict] = []
    for aid, count in author_manifest_counts.items():
        info = author_info_map.get(aid, {})
        name = info.get("name", aid[:12])
        email = info.get("email", "")
        file_count = len(author_files.get(aid, set()))
        share_pct = (
            round(count / total_manifest_records * 100, 1)
            if total_manifest_records > 0
            else 0.0
        )
        # Over-deletion ratio: how many times more files you'd delete with
        # dataset-level revocation (total_data_rows) vs ob-guided file-level
        # revocation (file_count).  Higher = more over-deletion without ob.
        over_deletion = (
            round(total_data_rows / file_count, 1)
            if file_count > 0
            else float("inf")
        )

        authors.append({
            "name": name,
            "email": email,
            "share_pct": share_pct,
            "manifest_count": count,
            "file_count": file_count,
            "over_deletion_ratio": over_deletion,
        })

    # Sort by manifest_count descending
    authors.sort(key=lambda a: a["manifest_count"], reverse=True)

    return {
        "total_data_rows": total_data_rows,
        "total_manifest_records": total_manifest_records,
        "unique_authors": len(authors),
        "authors": authors,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════════════════════════

def _print_table(name: str, result: dict, top_n: int) -> None:
    """Print a formatted revocation precision table."""
    selected = _select_top_authors(result["authors"], top_n)

    print(f"\n{'=' * 90}")
    print(f"  {name}")
    print(
        f"  Data rows: {result['total_data_rows']:,}  |  "
        f"Manifest records: {result['total_manifest_records']:,}  |  "
        f"Unique authors: {result['unique_authors']:,}"
    )
    print(f"{'=' * 90}")
    print(
        f"  {'Author':<25} {'Share%':>7} {'Manifest':>10} "
        f"{'Files':>9} {'Over-del':>9}"
    )
    print(f"  {'-' * 25} {'-' * 7} {'-' * 10} {'-' * 9} {'-' * 9}")

    for a in selected:
        od = (
            f"{a['over_deletion_ratio']:.1f}×"
            if a["over_deletion_ratio"] != float("inf")
            else "inf"
        )
        print(
            f"  {a['name']:<25} {a['share_pct']:>6.1f}% "
            f"{a['manifest_count']:>10,} {a['file_count']:>9,} {od:>9}"
        )

    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kernel revocation precision analysis: file-level vs record-level"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("benchmarks/results/pipeline_v2"),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top authors to include in table (preset authors always included)",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Only analyze datasets matching this substring",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    results_dir = args.results_dir.resolve()

    # Find kernel OB datasets
    kernel_dirs: list[Path] = []
    for p in sorted(results_dir.iterdir()):
        name = p.name
        if "kernel" not in name or not (p / ".ob").is_dir():
            continue
        if not (p / "jsonl").is_dir():
            continue
        if args.filter and args.filter not in name:
            continue
        kernel_dirs.append(p)

    if not kernel_dirs:
        log.error("No kernel OB datasets found in %s", results_dir)
        sys.exit(1)

    log.info("Found %d kernel OB datasets", len(kernel_dirs))

    all_results: dict = {}
    for ob_dir in kernel_dirs:
        name = ob_dir.name

        if not _has_jsonl(ob_dir):
            log.info("SKIP %s: no JSONL (Datatrove token-only output)", name)
            continue

        log.info("Analyzing %s ...", name)
        try:
            result = analyze_author_precision(ob_dir)
            all_results[name] = result
            _print_table(name, result, args.top_n)
        except Exception as e:
            log.error("Failed to analyze %s: %s", name, e)

    # Save results
    output_file = results_dir / "kernel_revocation_precision.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log.info("Results saved to %s", output_file)


if __name__ == "__main__":
    main()
