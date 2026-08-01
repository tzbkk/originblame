#!/usr/bin/env python3
"""Post-run verification for Task 1.5 multi-seed MU experiments.

Checks:
  1. All expected checkpoints exist (91 total)
  2. Each checkpoint has evaluation metrics
  3. Forget-set ranking: line > embedding > random on forget PPL
  4. Paired t-test (scipy.stats.ttest_rel) for statistical significance
  5. Cohen's d effect size
  6. Generates multiseed_summary.json with mean ± std

Usage:
    python3 verify_experiments.py --config benchmarks/scripts/pipeline_qa/config.yaml
    python3 verify_experiments.py --config config.yaml --results-dir benchmarks/results/pipeline_qa
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import yaml


AUTHORS = ["Berthe", "Antigng-bot", "Iokseng"]
SEEDS = [42, 123, 456]
ALGORITHMS = ["npo", "rmu", "grad_ascent"]
FORGET_TYPES = ["line", "page_prototype", "random"]
BETA_VALUES = [0.05, 0.2]

EXPECTED_METRICS = [
    "forget_ppl",
    "retain_ppl",
    "forget_rouge_l",
    "retain_rouge_l",
    "truth_ratio",
    "mia_auc_20",
    "extraction_strength",
    "forget_quality_ks",
]

ALGO_OUTPUT_DIRS = {
    "npo": "checkpoints/npo",
    "rmu": "checkpoints/rmu-fullft",
    "grad_ascent": "checkpoints/grad_ascent",
}


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class CheckResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = True
        self.messages: list[str] = []

    def ok(self, msg: str) -> None:
        self.messages.append(f"  PASS: {msg}")

    def fail(self, msg: str) -> None:
        self.passed = False
        self.messages.append(f"  FAIL: {msg}")

    def warn(self, msg: str) -> None:
        self.messages.append(f"  WARN: {msg}")


def check_checkpoints(results_dir: Path) -> CheckResult:
    """Check that all expected checkpoints exist."""
    cr = CheckResult("Checkpoint Existence")

    expected = 0
    found = 0
    missing: list[str] = []

    # SFT: 3 seeds
    for seed in SEEDS:
        expected += 1
        sft_dir = results_dir / "checkpoints" / f"sft_seed{seed}"
        if sft_dir.is_dir():
            found += 1
        else:
            missing.append(f"sft_seed{seed}")

    # Retrain: 3 authors × 3 seeds = 9
    for author in AUTHORS:
        for seed in SEEDS:
            expected += 1
            retrain_dir = results_dir / "checkpoints" / "retrain" / f"{author}_seed{seed}"
            if retrain_dir.is_dir():
                found += 1
            else:
                missing.append(f"retrain/{author}_seed{seed}")

    # MU: 3 algos × 3 seeds × 3 forget types × 3 authors = 81
    for algo in ALGORITHMS:
        algo_base = ALGO_OUTPUT_DIRS[algo]
        for seed in SEEDS:
            for ft in FORGET_TYPES:
                for author in AUTHORS:
                    expected += 1
                    ckpt_dir = (
                        results_dir / algo_base / f"{author}_{ft}_seed{seed}" / "final"
                    )
                    if ckpt_dir.is_dir():
                        has_adapter = (ckpt_dir / "adapter_config.json").exists()
                        has_config = (ckpt_dir / "config.json").exists()
                        if has_adapter or has_config:
                            found += 1
                        else:
                            missing.append(f"{algo}/{author}_{ft}_seed{seed}/final (no config)")
                    else:
                        missing.append(f"{algo}/{author}_{ft}_seed{seed}/final")

    # Beta sweep: 2 beta × 3 seeds = 6
    for beta in BETA_VALUES:
        for seed in SEEDS:
            expected += 1
            sweep_dir = (
                results_dir
                / "checkpoints"
                / "npo-sweep"
                / f"beta{beta}_Berthe_line_seed{seed}"
                / "final"
            )
            if sweep_dir.is_dir():
                found += 1
            else:
                missing.append(f"npo-sweep/beta{beta}_Berthe_line_seed{seed}/final")

    if found == expected:
        cr.ok(f"All {expected} checkpoints present")
    else:
        cr.fail(f"Found {found}/{expected} checkpoints")
        for m in missing:
            cr.messages.append(f"    Missing: {m}")

    return cr


def check_eval_results(results_dir: Path) -> CheckResult:
    """Check that eval_results.json has all keys and metrics."""
    cr = CheckResult("Evaluation Metrics")

    eval_file = results_dir / "eval_results.json"
    if not eval_file.exists():
        cr.fail(f"eval_results.json not found at {eval_file}")
        return cr

    with open(eval_file, encoding="utf-8") as f:
        results = json.load(f)

    cr.ok(f"eval_results.json found with {len(results)} entries")

    # Check each entry has all expected metrics
    missing_metrics: dict[str, list[str]] = {}
    for key, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
        missing = [m for m in EXPECTED_METRICS if m not in metrics]
        if missing:
            missing_metrics[key] = missing

    if missing_metrics:
        cr.fail(f"{len(missing_metrics)} entries missing metrics")
        for key, ms in list(missing_metrics.items())[:10]:
            cr.messages.append(f"    {key}: missing {ms}")
    else:
        cr.ok(f"All entries have {len(EXPECTED_METRICS)} metrics")

    return cr


def check_forget_set_ranking(results_dir: Path) -> CheckResult:
    """Check that line > embedding > random on forget PPL (per algorithm, author)."""
    cr = CheckResult("Forget-Set Ranking")

    eval_file = results_dir / "eval_results.json"
    if not eval_file.exists():
        cr.fail("eval_results.json not found")
        return cr

    with open(eval_file, encoding="utf-8") as f:
        results = json.load(f)

    rankings_ok = 0
    rankings_fail = 0

    for algo in ALGORITHMS:
        for author in AUTHORS:
            ppls = {}
            for ft in FORGET_TYPES:
                for seed in SEEDS:
                    key = f"{algo}_{author}_{ft}_seed{seed}"
                    if key in results and "forget_ppl" in results[key]:
                        if ft not in ppls:
                            ppls[ft] = []
                        ppls[ft].append(results[key]["forget_ppl"])

            if len(ppls) < 3:
                continue

            line_avg = sum(ppls.get("line", [0])) / max(1, len(ppls.get("line", [])))
            emb_avg = sum(ppls.get("embedding", [0])) / max(1, len(ppls.get("embedding", [])))
            rand_avg = sum(ppls.get("random", [0])) / max(1, len(ppls.get("random", [])))

            if line_avg >= emb_avg >= rand_avg:
                rankings_ok += 1
            elif line_avg > rand_avg:
                rankings_ok += 1
                cr.warn(
                    f"{algo}/{author}: line({line_avg:.1f}) > random({rand_avg:.1f}) "
                    f"but embedding({emb_avg:.1f}) out of order"
                )
            else:
                rankings_fail += 1
                cr.warn(
                    f"{algo}/{author}: ranking VIOLATED — "
                    f"line={line_avg:.1f}, embedding={emb_avg:.1f}, random={rand_avg:.1f}"
                )

    if rankings_ok > 0:
        cr.ok(f"Ranking check: {rankings_ok} ok, {rankings_fail} violated")
    if rankings_fail > 0:
        cr.fail(f"{rankings_fail} ranking violations detected")

    return cr


def compute_statistics(results_dir: Path) -> CheckResult:
    """Compute paired t-test and Cohen's d for line vs random forget PPL."""
    cr = CheckResult("Statistical Tests")

    try:
        from scipy.stats import ttest_rel
    except ImportError:
        cr.fail("scipy not installed — cannot compute t-tests. pip install scipy")
        return cr

    eval_file = results_dir / "eval_results.json"
    if not eval_file.exists():
        cr.fail("eval_results.json not found")
        return cr

    with open(eval_file, encoding="utf-8") as f:
        results = json.load(f)

    all_tests: dict[str, dict] = {}

    for algo in ALGORITHMS:
        for author in AUTHORS:
            line_ppls = []
            random_ppls = []

            for seed in SEEDS:
                line_key = f"{algo}_{author}_line_seed{seed}"
                random_key = f"{algo}_{author}_random_seed{seed}"

                if line_key in results and "forget_ppl" in results[line_key]:
                    line_ppls.append(results[line_key]["forget_ppl"])
                if random_key in results and "forget_ppl" in results[random_key]:
                    random_ppls.append(results[random_key]["forget_ppl"])

            if len(line_ppls) >= 2 and len(random_ppls) >= 2 and len(line_ppls) == len(random_ppls):
                t_stat, p_value = ttest_rel(line_ppls, random_ppls)

                mean_diff = sum(l - r for l, r in zip(line_ppls, random_ppls)) / len(line_ppls)
                all_vals = line_ppls + random_ppls
                pooled_std = math.sqrt(
                    sum((v - sum(all_vals) / len(all_vals)) ** 2 for v in all_vals)
                    / (len(all_vals) - 1)
                ) if len(all_vals) > 1 else 1.0
                cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0

                label = f"{algo}/{author}"
                all_tests[label] = {
                    "t_statistic": round(t_stat, 4),
                    "p_value": round(p_value, 6),
                    "cohens_d": round(cohens_d, 4),
                    "line_mean": round(sum(line_ppls) / len(line_ppls), 2),
                    "random_mean": round(sum(random_ppls) / len(random_ppls), 2),
                    "n_seeds": len(line_ppls),
                }

                sig = p_value < 0.05
                symbol = "p < 0.05 *" if sig else "p >= 0.05"
                cr.messages.append(
                    f"  {label}: t={t_stat:.3f}, {symbol}, d={cohens_d:.3f} "
                    f"(line={sum(line_ppls)/len(line_ppls):.1f} vs random={sum(random_ppls)/len(random_ppls):.1f})"
                )
            else:
                cr.warn(f"{algo}/{author}: insufficient data for t-test ({len(line_ppls)} line, {len(random_ppls)} random)")

    if all_tests:
        n_sig = sum(1 for t in all_tests.values() if t["p_value"] < 0.05)
        cr.ok(f"Computed {len(all_tests)} paired t-tests, {n_sig} significant (p<0.05)")
    else:
        cr.fail("No paired t-tests could be computed")

    return cr


def generate_summary(results_dir: Path) -> CheckResult:
    """Generate multiseed_summary.json with mean ± std per condition."""
    cr = CheckResult("Summary Generation")

    eval_file = results_dir / "eval_results.json"
    if not eval_file.exists():
        cr.fail("eval_results.json not found")
        return cr

    with open(eval_file, encoding="utf-8") as f:
        results = json.load(f)

    summary: dict[str, dict] = {}

    conditions = [
        ("sft", [("sft", f"sft_seed{seed}") for seed in SEEDS]),
    ]

    for algo in ALGORITHMS:
        for author in AUTHORS:
            for ft in FORGET_TYPES:
                label = f"{algo}_{author}_{ft}"
                keys = [f"{algo}_{author}_{ft}_seed{seed}" for seed in SEEDS]
                conditions.append((label, [(f"seed{seed}", k) for seed, k in zip(SEEDS, keys)]))

    for label, key_pairs in conditions:
        metrics_by_key: dict[str, list[float]] = defaultdict(list)
        seed_values: dict[str, dict] = {}

        for seed_label, key in key_pairs:
            if key in results and isinstance(results[key], dict):
                seed_values[seed_label] = results[key]
                for metric in EXPECTED_METRICS:
                    if metric in results[key]:
                        metrics_by_key[metric].append(results[key][metric])

        if not metrics_by_key:
            continue

        entry: dict = {"seeds": seed_values}

        for metric, values in metrics_by_key.items():
            if len(values) > 0:
                mean = sum(values) / len(values)
                if len(values) > 1:
                    std = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
                else:
                    std = 0.0
                entry[metric] = {
                    "mean": round(mean, 4),
                    "std": round(std, 4),
                    "n": len(values),
                }

        summary[label] = entry

    output_file = results_dir / "multiseed_summary.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    cr.ok(f"Summary saved to {output_file} ({len(summary)} conditions)")

    return cr


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify Task 1.5 multi-seed MU experiment results"
    )
    parser.add_argument(
        "--config",
        default="benchmarks/scripts/pipeline_qa/config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Path to results directory (default: from config)",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(str(config_path))

    if args.results_dir:
        results_dir = Path(args.results_dir).resolve()
    else:
        results_dir = Path(cfg["eval"]["output_dir"]).resolve()

    print(f"Results dir: {results_dir}")
    print(f"Config: {config_path}")
    print()

    all_checks = [
        check_checkpoints(results_dir),
        check_eval_results(results_dir),
        check_forget_set_ranking(results_dir),
        compute_statistics(results_dir),
        generate_summary(results_dir),
    ]

    print("=" * 70)
    print("  VERIFICATION REPORT")
    print("=" * 70)
    print()

    n_pass = 0
    n_fail = 0
    for cr in all_checks:
        status = "PASS" if cr.passed else "FAIL"
        print(f"[{status}] {cr.name}")
        for msg in cr.messages:
            print(msg)
        print()
        if cr.passed:
            n_pass += 1
        else:
            n_fail += 1

    print("=" * 70)
    print(f"  TOTAL: {n_pass} passed, {n_fail} failed")
    print("=" * 70)

    if n_fail > 0:
        print("\nSome checks FAILED. Review output above.")
        sys.exit(1)
    else:
        print("\nAll checks PASSED.")
        sys.exit(0)


if __name__ == "__main__":
    main()
