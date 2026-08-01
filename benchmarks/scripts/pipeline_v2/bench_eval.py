#!/usr/bin/env python3
"""
bench_eval.py — Reproducible evaluation for paper Tables 3, 4, 5.

Runs on pipeline_v2 HuggingFace datasets (deterministic, no LLM).

Usage:
    python3 benchmarks/scripts/pipeline_v2/bench_eval.py
    python3 benchmarks/scripts/pipeline_v2/bench_eval.py --bench revocation
    python3 benchmarks/scripts/pipeline_v2/bench_eval.py --bench scalability
    python3 benchmarks/scripts/pipeline_v2/bench_eval.py --bench reconcile
    python3 benchmarks/scripts/pipeline_v2/bench_eval.py --runs 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent

NATIVE_PATH = REPO_ROOT.parent / "rust-originblame" / "python" / "src"
if NATIVE_PATH.is_dir():
    sys.path.insert(0, str(NATIVE_PATH))

R = None
try:
    from ob import _ob_native as R  # type: ignore[no-redef]
except ImportError:
    try:
        import _ob_native as R  # type: ignore[no-redef]
    except ImportError:
        pass

if R is None:
    print("ERROR: Cannot import native module from ob._native or _ob_native", file=sys.stderr)
    print("Ensure rust-originblame/python/src is on PYTHONPATH", file=sys.stderr)
    sys.exit(1)

# ── Constants ─────────────────────────────────────────────────────────────────

RESULTS_DIR = Path("benchmarks/results/pipeline_v2")

HF_DATASETS = [
    "huggingface-zhwiki-1k-ob",
    "huggingface-zhwiki-10k-ob",
    "huggingface-zhwiki-100k-ob",
    "huggingface-zhwiki-all-ob",
    "huggingface-kernel-1k-ob",
    "huggingface-kernel-10k-ob",
    "huggingface-kernel-all-ob",
]

REVOCATION_DATASET = "huggingface-zhwiki-10k-ob"
RECONCILE_DATASETS = [
    "huggingface-zhwiki-1k-ob",
    "huggingface-zhwiki-10k-ob",
    "huggingface-zhwiki-100k-ob",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _ob_dir(dataset_name: str) -> Path:
    return RESULTS_DIR / dataset_name


def _data_path(dataset_name: str) -> str | None:
    """Return data.jsonl path for HF datasets, None for Datatrove (token-only)."""
    p = _ob_dir(dataset_name) / "jsonl" / "data.jsonl"
    return str(p) if p.is_file() else None


def _has_jsonl(dataset_name: str) -> bool:
    """Check if dataset has data.jsonl (HuggingFace pipeline output)."""
    return (_ob_dir(dataset_name) / "jsonl" / "data.jsonl").is_file()


def _load_lines(ob_dir: Path) -> list[str]:
    data_file = ob_dir / "jsonl" / "data.jsonl"
    with open(data_file, encoding="utf-8") as f:
        return f.read().splitlines()


def _sample_line_numbers(total: int, n: int = 10) -> list[int]:
    """Return n evenly-spaced 1-based line numbers."""
    if total <= n:
        return list(range(1, total + 1))
    step = total // n
    return list(range(1, total + 1, step))[:n]


def _line_hash_for_text(text: str) -> str:
    """Compute SHA-256 hash matching the pipeline_v2 HF tracking format."""
    payload = json.dumps({"text": text}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _extract_text_from_jsonl_line(line: str) -> str | None:
    """Extract the 'text' field from a JSONL line, or return None."""
    try:
        obj = json.loads(line)
        return obj.get("text")
    except (json.JSONDecodeError, AttributeError):
        return None


def _has_index(ob_dir: Path) -> bool:
    return (ob_dir / ".ob" / "index").is_dir()


def _ns_to_ms(ns: int) -> float:
    return round(ns / 1_000_000, 3)


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark 1: Revocation Precision (Table 3)
# ═══════════════════════════════════════════════════════════════════════════════

def _author_shares(ob_str: str, total_manifest: int) -> list[tuple[str, str, int, float]]:
    """Compute each author's manifest share. Returns (name, email, count, pct)."""

    # Build section_hash -> author_ids
    section_authors: dict[str, list[str]] = {}
    for sec in R.shard_iterate_all(ob_str, "sections"):
        sh = sec.get("section_hash", "")
        if sh:
            section_authors[sh] = sec.get("authors", [])

    # Build author_id -> {name, email}
    author_map: dict[str, dict] = {}
    for a in R.shard_iterate_all(ob_str, "authors"):
        author_map[a.get("id", "")] = {"name": a.get("name", ""), "email": a.get("email", "")}

    # Count manifest records per author
    author_counts: dict[str, int] = {}
    for rec in R.shard_iterate_all(ob_str, "document-index"):
        for sh in rec.get("sources", []):
            for aid in section_authors.get(sh, []):
                author_counts[aid] = author_counts.get(aid, 0) + 1

    result = []
    for aid, count in author_counts.items():
        info = author_map.get(aid, {})
        name = info.get("name", "")
        email = info.get("email", "")
        pct = (count / total_manifest * 100) if total_manifest > 0 else 0.0
        result.append((name, email, count, pct))

    result.sort(key=lambda x: x[2], reverse=True)
    return result


def _pick_quota_authors(shares: list[tuple[str, str, int, float]]) -> list[tuple[str, str, int, float]]:
    """Pick 4 authors at different share levels: high (>50%), medium (5-20%), low (1-5%), tiny (<1%)."""
    if len(shares) <= 4:
        return shares

    targets = []
    # High: first author with >50% share
    for s in shares:
        if s[3] > 50:
            targets.append(s)
            break
    # Medium: first with 5-20%
    for s in shares:
        if 5 <= s[3] <= 20:
            targets.append(s)
            break
    # Low: first with 1-5%
    for s in shares:
        if 1 <= s[3] < 5:
            targets.append(s)
            break
    # Tiny: first with 0.1-1%
    for s in shares:
        if 0.1 <= s[3] < 1:
            targets.append(s)
            break

    # Fallback: if not enough targets, fill from sorted list
    if len(targets) < 4:
        used = {t[1] for t in targets}
        for s in shares:
            if s[1] not in used:
                targets.append(s)
                if len(targets) >= 4:
                    break

    return targets[:4]


def run_revocation(dataset_name: str = REVOCATION_DATASET) -> dict:
    """Benchmark 1: Revocation Precision."""
    if not _has_jsonl(dataset_name):
        print(f"  SKIP {dataset_name}: no JSONL (Datatrove token-only output)")
        return {"dataset": dataset_name, "skipped": True, "reason": "no JSONL"}

    ob_dir = _ob_dir(dataset_name)
    ob_str = str(ob_dir)
    data_path = _data_path(dataset_name)
    lines = _load_lines(ob_dir)
    total_lines = len(lines)

    # Count manifest records
    total_manifest = sum(1 for _ in R.shard_iterate_all(ob_str, "document-index"))

    # Compute author shares
    shares = _author_shares(ob_str, total_manifest)
    targets = _pick_quota_authors(shares)

    rows = []
    for name, email, affected, pct in targets:
        # show_by_author to get affected entries
        entries = R.show_by_author(ob_str, name)
        affected_rows = len(entries) if entries else 0

        # Over-deletion: file-level (delete everything) vs record-level (delete only affected)
        over_deletion = round(total_lines / affected_rows, 1) if affected_rows > 0 else float("inf")

        # Revoke (toggle on)
        R.revoke_by_author(ob_str, name)

        # Purge (dry run)
        purge_result = R.purge_revoked(ob_str, data_path, dry_run=True)
        purged_dry = purge_result.get("purged", 0) if isinstance(purge_result, dict) else 0

        # Restore (toggle back off)
        R.revoke_by_author(ob_str, name)

        rows.append({
            "author_name": name,
            "share_pct": round(pct, 1),
            "affected_rows": affected_rows,
            "purged_dry": purged_dry,
            "over_deletion_factor": over_deletion,
            "total_lines": total_lines,
        })

    result = {
        "dataset": dataset_name,
        "total_lines": total_lines,
        "total_manifest": total_manifest,
        "authors": rows,
    }

    # Print table
    print("\n" + "=" * 90)
    print("Table 3 — Revocation Precision (dataset: {})".format(dataset_name))
    print("=" * 90)
    print("{:<25s} {:>8s} {:>12s} {:>12s} {:>18s}".format(
        "Author", "Share%", "Affected", "Purged(dry)", "Over-deletion"))
    print("-" * 90)
    for r in rows:
        print("{:<25s} {:>7.1f}% {:>12d} {:>12d} {:>17.1f}x".format(
            r["author_name"][:25],
            r["share_pct"],
            r["affected_rows"],
            r["purged_dry"],
            r["over_deletion_factor"],
        ))
    print("=" * 90)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark 2: Scalability (Table 5)
# ═══════════════════════════════════════════════════════════════════════════════

def run_scalability(datasets: list[str] | None = None, runs: int = 3) -> dict:
    """Benchmark 2: Scalability across all HF datasets."""
    if datasets is None:
        datasets = HF_DATASETS

    all_results = {}

    for ds_name in datasets:
        ob_dir = _ob_dir(ds_name)
        ob_str = str(ob_dir)
        data_path = _data_path(ds_name)

        if not ob_dir.is_dir():
            print(f"  SKIP {ds_name}: directory not found")
            continue

        if not _has_jsonl(ds_name):
            print(f"  SKIP {ds_name}: no JSONL (Datatrove token-only output)")
            continue

        lines = _load_lines(ob_dir)
        total_lines = len(lines)
        line_numbers = _sample_line_numbers(total_lines, 10)
        has_idx = _has_index(ob_dir)

        # Get first author for show/revoke targets
        all_authors = list(R.shard_iterate_all(ob_str, "authors"))
        sample_author = all_authors[0] if all_authors else None
        sample_name = sample_author.get("name", "") if sample_author else None
        sample_email = sample_author.get("email", "") if sample_author else None

        # Warm-up
        for ln in line_numbers[:3]:
            content = lines[ln - 1]
            R.blame(ob_str, data_path, content)
        if sample_name:
            R.show_by_author(ob_str, sample_name)

        # Collect per-run timings
        run_data: dict[str, list[float]] = {
            "blame": [],
            "show": [],
            "show_idx": [],
            "revoke": [],
            "purge": [],
            "purge_idx": [],
        }

        for _run in range(runs):
            # -- blame (10 samples averaged per run) --
            blame_times = []
            for ln in line_numbers:
                content = lines[ln - 1]
                t0 = time.perf_counter_ns()
                R.blame(ob_str, data_path, content)
                blame_times.append(time.perf_counter_ns() - t0)
            run_data["blame"].append(sum(blame_times) / len(blame_times) / 1e6)

            # -- show_by_author (3 samples averaged per run) --
            if sample_name:
                show_times = []
                for _ in range(3):
                    t0 = time.perf_counter_ns()
                    R.show_by_author(ob_str, sample_name)
                    show_times.append(time.perf_counter_ns() - t0)
                run_data["show"].append(sum(show_times) / len(show_times) / 1e6)

                # show with index (same call, index auto-used)
                if has_idx:
                    idx_times = []
                    for _ in range(3):
                        t0 = time.perf_counter_ns()
                        R.show_by_author(ob_str, sample_name)
                        idx_times.append(time.perf_counter_ns() - t0)
                    run_data["show_idx"].append(sum(idx_times) / len(idx_times) / 1e6)

            # -- revoke + purge --
            if sample_name:
                # Revoke (toggle on)
                t0 = time.perf_counter_ns()
                R.revoke_by_author(ob_str, sample_name)
                run_data["revoke"].append((time.perf_counter_ns() - t0) / 1e6)

                # Warm-up blame after revoke
                for ln in line_numbers[:3]:
                    content = lines[ln - 1]
                    R.blame(ob_str, data_path, content)

                # Purge (dry run)
                t0 = time.perf_counter_ns()
                R.purge_revoked(ob_str, data_path, dry_run=True)
                run_data["purge"].append((time.perf_counter_ns() - t0) / 1e6)

                # Purge by author indexed (dry run)
                if has_idx and sample_email:
                    # Warm-up
                    try:
                        R.purge_by_author_indexd(ob_str, sample_email, data_path, True)
                    except Exception:
                        pass
                    idx_purge_times = []
                    for _ in range(3):
                        t0 = time.perf_counter_ns()
                        try:
                            R.purge_by_author_indexd(ob_str, sample_email, data_path, True)
                            idx_purge_times.append(time.perf_counter_ns() - t0)
                        except Exception:
                            pass
                    if idx_purge_times:
                        run_data["purge_idx"].append(
                            sum(idx_purge_times) / len(idx_purge_times) / 1e6
                        )

                # Restore (toggle off)
                R.revoke_by_author(ob_str, sample_name)

        # Average across runs
        ds_result: dict = {"total_lines": total_lines, "has_index": has_idx, "runs": runs}
        for op, values in run_data.items():
            if values:
                ds_result[f"{op}_ms"] = round(sum(values) / len(values), 3)
            else:
                ds_result[f"{op}_ms"] = None

        all_results[ds_name] = ds_result

        print(f"  {ds_name}: {total_lines:>8d} lines  "
              f"blame={ds_result.get('blame_ms', 'N/A')}ms  "
              f"show={ds_result.get('show_ms', 'N/A')}ms  "
              f"revoke={ds_result.get('revoke_ms', 'N/A')}ms  "
              f"purge={ds_result.get('purge_ms', 'N/A')}ms")

    # Print summary table
    print("\n" + "=" * 110)
    print("Table 5 — Scalability ({}-run avg, ms)".format(runs))
    print("=" * 110)
    fmt = "{:<30s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s}"
    print(fmt.format("Dataset", "Lines", "blame", "show", "show_idx", "revoke", "purge", "purge_idx"))
    print("-" * 110)
    for ds_name in datasets:
        r = all_results.get(ds_name, {})
        if not r:
            continue
        def _v(key: str) -> str:
            v = r.get(f"{key}_ms")
            return f"{v:.3f}" if v is not None else "N/A"
        print(fmt.format(
            ds_name[:30],
            str(r.get("total_lines", "")),
            _v("blame"),
            _v("show"),
            _v("show_idx"),
            _v("revoke"),
            _v("purge"),
            _v("purge_idx"),
        ))
    print("=" * 110)

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark 3: Reconcile (Table 4)
# ═══════════════════════════════════════════════════════════════════════════════

def _mutate_chars(text: str, rng: random.Random, char_pct: float = 0.10) -> str:
    """Replace char_pct of alphabetic characters with random lowercase letters."""
    chars = list(text)
    alpha_positions = [i for i, c in enumerate(chars) if c.isalpha()]
    n_edit = max(1, int(len(alpha_positions) * char_pct))
    for pos in rng.sample(alpha_positions, min(n_edit, len(alpha_positions))):
        chars[pos] = rng.choice("abcdefghijklmnopqrstuvwxyz")
    return "".join(chars)


def _mutate_data(
    data_file: Path,
    output_file: Path,
    seed: int = 42,
    edit_pct: float = 0.10,
    delete_pct: float = 0.05,
    insert_pct: float = 0.05,
) -> dict:
    """Mutate a data file to simulate edits. Returns mutation stats."""
    rng = random.Random(seed)
    with open(data_file, encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]

    original_lines = len(lines)
    categories = []
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

    num_inserts = max(1, int(original_lines * insert_pct))
    insert_lines = []
    for _ in range(num_inserts):
        src = rng.choice(lines)
        try:
            rec = json.loads(src)
            if "text" in rec:
                rec["text"] = _mutate_chars(rec["text"], rng, char_pct=0.15)
                insert_lines.append(json.dumps(rec, ensure_ascii=False))
            else:
                insert_lines.append(_mutate_chars(src, rng, char_pct=0.15))
        except (json.JSONDecodeError, KeyError):
            insert_lines.append(_mutate_chars(src, rng, char_pct=0.15))

    result_lines = []
    for i, line in enumerate(lines):
        cat = categories[i]
        if cat == "delete":
            continue
        if cat == "edit":
            try:
                rec = json.loads(line)
                if "text" in rec:
                    rec["text"] = _mutate_chars(rec["text"], rng)
                    result_lines.append(json.dumps(rec, ensure_ascii=False))
                else:
                    result_lines.append(_mutate_chars(line, rng))
            except (json.JSONDecodeError, KeyError):
                result_lines.append(_mutate_chars(line, rng))
        else:
            result_lines.append(line)

    for ins in insert_lines:
        pos = rng.randint(0, len(result_lines))
        result_lines.insert(pos, ins)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for line in result_lines:
            f.write(line + "\n")

    return {
        "original_lines": original_lines,
        "edited_lines": edited_lines,
        "deleted_lines": deleted_lines,
        "inserted_lines": num_inserts,
        "final_lines": len(result_lines),
    }


def run_reconcile(datasets: list[str] | None = None) -> dict:
    """Benchmark 3: Reconcile — Pass 1 hash matching only."""
    if datasets is None:
        datasets = RECONCILE_DATASETS

    all_results = {}

    for ds_name in datasets:
        ob_dir = _ob_dir(ds_name)
        ob_str = str(ob_dir)

        if not ob_dir.is_dir():
            print(f"  SKIP {ds_name}: directory not found")
            continue

        if not _has_jsonl(ds_name):
            print(f"  SKIP {ds_name}: no JSONL (Datatrove token-only output)")
            continue

        data_file = ob_dir / "jsonl" / "data.jsonl"

        # Use a temp directory to avoid modifying original data
        with tempfile.TemporaryDirectory(prefix=f"bench_eval_{ds_name}_") as tmpdir:
            tmp_path = Path(tmpdir)

            # Copy data.jsonl
            tmp_data = tmp_path / "data.jsonl"
            shutil.copy2(data_file, tmp_data)

            # Mutate the copy
            mutated_file = tmp_path / "mutated.jsonl"
            mutation = _mutate_data(tmp_data, mutated_file, seed=42)

            # Build hash set from original manifest
            manifest_hashes: set[str] = set()
            for rec in R.shard_iterate_all(ob_str, "document-index"):
                lh = rec.get("line_hash", "")
                if lh:
                    manifest_hashes.add(lh)

            # Read mutated file and check hash matches against kept lines
            hash_matched = 0
            with open(mutated_file, encoding="utf-8") as f:
                for raw_line in f:
                    stripped = raw_line.strip()
                    if not stripped:
                        continue
                    try:
                        obj = json.loads(stripped)
                    except json.JSONDecodeError:
                        continue
                    computed = R.compute_hash(obj)
                    if computed in manifest_hashes:
                        hash_matched += 1

            total_mutated = mutation["final_lines"]
            recovery_pct = round(hash_matched / total_mutated * 100, 1) if total_mutated > 0 else 0.0

            result = {
                "dataset": ds_name,
                "seed": 42,
                "pass1_only": True,
                "mutation": mutation,
                "hash_matched": hash_matched,
                "total_mutated_lines": total_mutated,
                "recovery_pct": recovery_pct,
                "total_manifest_hashes": len(manifest_hashes),
            }
            all_results[ds_name] = result

            print(f"  {ds_name}: orig={mutation['original_lines']}  "
                  f"edit={mutation['edited_lines']}  "
                  f"del={mutation['deleted_lines']}  "
                  f"ins={mutation['inserted_lines']}  "
                  f"hash_matched={hash_matched}  "
                  f"recovery={recovery_pct}%")

    # Print summary table
    print("\n" + "=" * 90)
    print("Table 4 — Reconcile Recovery (Pass 1: hash matching, seed=42)")
    print("=" * 90)
    fmt = "{:<30s} {:>6s} {:>6s} {:>6s} {:>6s} {:>10s} {:>10s}"
    print(fmt.format("Dataset", "Orig", "Edit", "Del", "Ins", "HashMatch", "Recovery%"))
    print("-" * 90)
    for ds_name in datasets:
        r = all_results.get(ds_name, {})
        if not r:
            continue
        m = r.get("mutation", {})
        print(fmt.format(
            ds_name[:30],
            str(m.get("original_lines", "")),
            str(m.get("edited_lines", "")),
            str(m.get("deleted_lines", "")),
            str(m.get("inserted_lines", "")),
            str(r.get("hash_matched", "")),
            f"{r.get('recovery_pct', 0):.1f}%",
        ))
    print("=" * 90)

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    global RESULTS_DIR

    parser = argparse.ArgumentParser(
        description="Reproducible evaluation for paper Tables 3, 4, 5 on pipeline_v2 HF datasets",
    )
    parser.add_argument(
        "--bench",
        choices=["revocation", "scalability", "reconcile"],
        default=None,
        help="Run a single benchmark (default: all three)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs for scalability benchmark (default: 3)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Pipeline v2 results directory",
    )
    args = parser.parse_args()

    RESULTS_DIR = args.results_dir.resolve()

    ts = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    all_output: dict = {
        "timestamp": ts,
    }

    t_start = time.monotonic()

    # ── Revocation (Table 3) ─────────────────────────────────────────────────
    if args.bench is None or args.bench == "revocation":
        print("\n>>> Benchmark 1: Revocation Precision (Table 3)")
        all_output["revocation"] = run_revocation()

    # ── Scalability (Table 5) ────────────────────────────────────────────────
    if args.bench is None or args.bench == "scalability":
        print("\n>>> Benchmark 2: Scalability (Table 5)")
        all_output["scalability"] = run_scalability(runs=args.runs)

    # ── Reconcile (Table 4) ──────────────────────────────────────────────────
    if args.bench is None or args.bench == "reconcile":
        print("\n>>> Benchmark 3: Reconcile (Table 4)")
        all_output["reconcile"] = run_reconcile()

    wall_seconds = round(time.monotonic() - t_start, 1)
    all_output["wall_seconds"] = wall_seconds

    # ── Write JSON output ────────────────────────────────────────────────────
    out_dir = RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "bench_eval_latest.json"
    out_file.write_text(
        json.dumps(all_output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"\nResults written to {out_file}")
    print(f"Total wall time: {wall_seconds}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
