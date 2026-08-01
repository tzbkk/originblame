#!/usr/bin/env python3
"""Retroactive Provenance PoC — 100-record Wikipedia subset.

Proves that retroactive provenance recovery works by:
1. Building a source index from 100 Wikipedia pages (SHA-256 of content)
2. Simulating retroactive matching against a "found" dataset
3. Sweeping 12 confidence thresholds to find optimal P/R/F1
4. Generating a PR curve figure for the paper

Usage:
    python3 benchmarks/scripts/reconstruct_poc.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from ob_util.reconstruct import (
    SourceRecord,
    build_source_index,
    generate_pr_curve,
    load_records,
    match_record,
    simulate_mutations,
    sweep_thresholds,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_JSONL = (
    ROOT / "benchmarks" / "results" / "pipeline_v2"
    / "huggingface-zhwiki-1k-ob" / "jsonl" / "data.jsonl"
)
FIGURE_DIR = ROOT / "paper" / "figures-cikm"
RETRO_DIR = ROOT / "benchmarks" / "results" / "pipeline_v2" / "retroactive"
EVIDENCE_DIR = ROOT / ".sisyphus" / "evidence"

N_RECORDS = 100
THRESHOLDS = [0.50, 0.60, 0.70, 0.75, 0.80, 0.82, 0.85, 0.88, 0.90, 0.92, 0.95, 0.99]
SEED = 42


def main() -> None:
    print("=" * 60)
    print("Retroactive Provenance PoC — 100-record Wikipedia subset")
    print("=" * 60)

    if not DATA_JSONL.exists():
        print(f"ERROR: {DATA_JSONL} not found")
        sys.exit(1)

    rng = random.Random(SEED)

    print("\n[Phase 1] Extract — Build source index")
    source_records = load_records(DATA_JSONL, start=0, n=N_RECORDS)
    source_index = build_source_index(source_records)
    source_texts = [r.text for r in source_index.values()]
    source_records_list = list(source_index.values())

    print("\n[Phase 2] Match — Simulate retroactive recovery")
    mutated = simulate_mutations(source_records, rng)
    distractors = load_records(DATA_JSONL, start=500, n=50)

    found_records: list[dict] = []
    found_is_from_source: list[bool] = []
    for rec in mutated:
        found_records.append(rec)
        found_is_from_source.append(True)
    for rec in distractors:
        found_records.append(rec)
        found_is_from_source.append(False)
    paired = list(zip(found_records, found_is_from_source))
    rng.shuffle(paired)
    found_records = [p[0] for p in paired]
    found_is_from_source = [p[1] for p in paired]
    print(f"  Found dataset: {len(found_records)} records "
          f"({N_RECORDS} target + {len(distractors)} distractors)")

    matches: list[tuple[float, bool]] = []
    matched_records: list[dict] = []
    for rec, is_src in zip(found_records, found_is_from_source):
        conf, matched = match_record(rec["text"], source_index, source_texts, source_records_list)
        matches.append((conf, is_src))
        if matched and conf > 0.5:
            matched_records.append({
                "confidence": conf,
                "found_title": rec["title"],
                "matched_title": matched.title,
                "authors": matched.authors,
                "year": matched.year,
                "license": matched.license,
            })

    n_true = sum(found_is_from_source)
    print(f"  Ground truth: {n_true} of {len(found_records)} records from source set")

    print("\n[Phase 3] Threshold sweep")
    sweep_results = sweep_thresholds(matches, THRESHOLDS)

    hdr = f"  {'θ':>6s}  {'P':>6s}  {'R':>6s}  {'F1':>6s}  {'TP':>4s}  {'FP':>4s}  {'FN':>4s}"
    print(f"\n{hdr}")
    print("  " + "-" * 42)
    for r in sweep_results:
        print(f"  {r.threshold:6.2f}  {r.precision:6.3f}  {r.recall:6.3f}  "
              f"{r.f1:6.3f}  {r.tp:4d}  {r.fp:4d}  {r.fn:4d}")

    best = max(sweep_results, key=lambda r: r.f1)
    print(f"\n  Optimal: θ={best.threshold:.2f} at F1-maximum "
          f"(P={best.precision:.3f}, R={best.recall:.3f}, F1={best.f1:.3f})")

    recovery_rate = best.tp / N_RECORDS if N_RECORDS > 0 else 0
    print(f"  Recovery rate: {recovery_rate:.1%} ({best.tp}/{N_RECORDS} source records)")

    print("\n[Phase 4] Build — Write outputs")
    RETRO_DIR.mkdir(parents=True, exist_ok=True)
    retro_output = RETRO_DIR / "matched_records.jsonl"
    with open(retro_output, "w", encoding="utf-8") as f:
        for mr in sorted(matched_records, key=lambda x: x["confidence"], reverse=True):
            f.write(json.dumps(mr, ensure_ascii=False) + "\n")
    print(f"  Matched records: {retro_output} ({len(matched_records)} records)")

    fig_path = FIGURE_DIR / "fig-reconstruct-threshold.pdf"
    generate_pr_curve(sweep_results, fig_path)

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    evidence_path = EVIDENCE_DIR / "task-0.6-reconstruct-poc.txt"
    with open(evidence_path, "w", encoding="utf-8") as f:
        f.write("Retroactive Provenance PoC — Evidence\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset: {DATA_JSONL}\n")
        f.write(f"Records: {N_RECORDS} source + {len(distractors)} distractors\n")
        f.write(f"Seed: {SEED}\n\n")
        f.write("Threshold Sweep:\n")
        f.write(f"  {'θ':>6s}  {'P':>6s}  {'R':>6s}  {'F1':>6s}\n")
        for r in sweep_results:
            f.write(f"  {r.threshold:6.2f}  {r.precision:6.3f}  "
                    f"{r.recall:6.3f}  {r.f1:6.3f}\n")
        f.write(f"\nOptimal: θ={best.threshold:.2f} at F1-maximum "
                f"(P={best.precision:.3f}, R={best.recall:.3f}, F1={best.f1:.3f})\n")
        f.write(f"Recovery rate: {recovery_rate:.1%}\n")
        f.write(f"Figure: {fig_path}\n")
        f.write(f"Matched records: {retro_output}\n")
    print(f"  Evidence: {evidence_path}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
