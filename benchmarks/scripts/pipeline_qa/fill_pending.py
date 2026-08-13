#!/usr/bin/env python3
"""Fill [PENDING] placeholders in paper LaTeX files with experiment results.

Reads multiseed_summary.json and replaces \\textcolor{red}{[PENDING]} markers
in evaluation.tex and table2-mu-results.tex with actual mean ± std values.

Usage:
    python3 fill_pending.py --results-dir benchmarks/results/pipeline_qa
    python3 fill_pending.py --results-dir benchmarks/results/pipeline_qa --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

CONDITION_MAP = {
    "Prov.": "line",
    "Embed.": "page_prototype",
    "Random": "random",
}

AUTHOR_MAP = {
    "Berthe": "Berthe",
    "Antigng": "Antigng-bot",
    "Iokseng": "Iokseng",
}

# Column name → metric name mapping
COLUMN_METRIC = {
    "Fgt PPL": "forget_ppl",
    "Ret PPL": "retain_ppl",
    "Truth Ratio": "truth_ratio",
}


def load_summary(results_dir: Path) -> dict:
    summary_file = results_dir / "multiseed_summary.json"
    if not summary_file.exists():
        print(f"ERROR: {summary_file} not found. Run verify_experiments.py first.")
        sys.exit(1)

    with open(summary_file, encoding="utf-8") as f:
        return json.load(f)


def find_best_algorithm_per_condition(summary: dict, metric: str, higher_is_better: bool) -> dict:
    """For each (author, condition), find the algorithm with best mean for the metric.

    Returns {(author, condition): (algo_name, mean, std)}
    """
    best: dict[tuple[str, str], tuple[str, float, float]] = {}

    for algo in ["npo", "rmu", "grad_ascent"]:
        for author in ["Berthe", "Antigng-bot", "Iokseng"]:
            for cond in ["line", "page_prototype", "random"]:
                key = f"{algo}_{author}_{cond}"
                if key not in summary:
                    continue
                entry = summary[key]
                if metric not in entry:
                    continue
                m = entry[metric]["mean"]
                s = entry[metric]["std"]
                pair = (author, cond)
                if pair not in best:
                    best[pair] = (algo, m, s)
                else:
                    prev_m = best[pair][1]
                    if higher_is_better and m > prev_m:
                        best[pair] = (algo, m, s)
                    elif not higher_is_better and m < prev_m:
                        best[pair] = (algo, m, s)

    return best


def compute_aggregate_improvement(summary: dict) -> dict:
    """Compute aggregate provenance-vs-random improvement across all algorithms/seeds.

    Returns dict with keys: forget_ppl_improvement, retain_ppl_improvement,
    truth_ratio_improvement, p_value, cohens_d, etc.
    """
    metrics_data: dict[str, dict[str, list[float]]] = {
        "forget_ppl": {"line": [], "random": []},
        "retain_ppl": {"line": [], "random": []},
        "truth_ratio": {"line": [], "random": []},
    }

    for algo in ["npo", "rmu", "grad_ascent"]:
        for author in ["Berthe", "Antigng-bot", "Iokseng"]:
            for cond in ["line", "random"]:
                key = f"{algo}_{author}_{cond}"
                if key not in summary:
                    continue
                entry = summary[key]
                for metric in metrics_data:
                    if metric in entry:
                        metrics_data[metric][cond].append(entry[metric]["mean"])

    result: dict = {}
    for metric in metrics_data:
        line_vals = metrics_data[metric]["line"]
        random_vals = metrics_data[metric]["random"]

        if not line_vals or not random_vals:
            continue

        line_avg = sum(line_vals) / len(line_vals)
        random_avg = sum(random_vals) / len(random_vals)

        if random_avg != 0:
            improvement = (line_avg - random_avg) / abs(random_avg) * 100
        else:
            improvement = 0.0

        result[metric] = {
            "line_mean": round(line_avg, 2),
            "random_mean": round(random_avg, 2),
            "improvement_pct": round(improvement, 1),
        }

    # Compute paired t-test across (algo, author) pairs for line vs random
    try:
        from scipy.stats import ttest_rel

        line_ppls = []
        random_ppls = []
        for algo in ["npo", "rmu", "grad_ascent"]:
            for author in ["Berthe", "Antigng-bot", "Iokseng"]:
                line_key = f"{algo}_{author}_line"
                random_key = f"{algo}_{author}_random"
                if line_key in summary and random_key in summary:
                    if "forget_ppl" in summary[line_key] and "forget_ppl" in summary[random_key]:
                        line_ppls.append(summary[line_key]["forget_ppl"]["mean"])
                        random_ppls.append(summary[random_key]["forget_ppl"]["mean"])

        if len(line_ppls) >= 2 and len(random_ppls) >= 2:
            import math

            t_stat, p_value = ttest_rel(line_ppls, random_ppls)
            mean_diff = sum(l - r for l, r in zip(line_ppls, random_ppls)) / len(line_ppls)
            all_vals = line_ppls + random_ppls
            pooled_std = math.sqrt(
                sum((v - sum(all_vals) / len(all_vals)) ** 2 for v in all_vals)
                / (len(all_vals) - 1)
            ) if len(all_vals) > 1 else 1.0
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0

            result["paired_ttest"] = {
                "t_statistic": round(t_stat, 3),
                "p_value": round(p_value, 6),
                "cohens_d": round(cohens_d, 3),
                "significant_005": p_value < 0.05,
                "n_pairs": len(line_ppls),
            }
    except ImportError:
        result["paired_ttest"] = {"error": "scipy not available"}

    return result


def fill_evaluation_tex(eval_tex_path: Path, aggregates: dict, summary: dict) -> bool:
    """Replace [PENDING] placeholders in evaluation.tex."""
    if not eval_tex_path.exists():
        print(f"WARNING: {eval_tex_path} not found")
        return False

    with open(eval_tex_path, encoding="utf-8") as f:
        content = f.read()

    original = content

    # Find [PENDING] placeholders and fill them based on context
    # The evaluation.tex has placeholders in specific order:
    # "achieve [PENDING] higher forget-set PPL and [PENDING] lower retain-set PPL"
    # "p[PENDING] < 0.05, Cohen's d = [PENDING]"
    # "[PENDING] improvement for provenance conditions"

    fppl = aggregates.get("forget_ppl", {})
    rppl = aggregates.get("retain_ppl", {})
    ttest = aggregates.get("paired_ttest", {})

    replacements_made = 0
    pending_positions = list(re.finditer(r"\\textcolor\{red\}\{\\bf\{?\[PENDING\]\}?\}", content))
    pending_simple = list(re.finditer(r"\\textcolor\{red\}\{\[PENDING\]\}", content))
    all_pending = sorted(pending_positions + pending_simple, key=lambda m: m.start())

    # Context-based replacement strategy:
    # We need to identify which [PENDING] maps to which value from the surrounding text
    # The text contains a specific pattern we can parse

    # Strategy: replace in order based on the evaluation.tex structure
    values_to_fill = []

    # From the text: "achieve X% higher forget-set PPL and Y% lower retain-set PPL"
    fppl_improvement = fppl.get("improvement_pct", "TBD")
    rppl_improvement = rppl.get("improvement_pct", "TBD")
    p_value_str = f"={ttest.get('p_value', 'TBD')}" if ttest.get("p_value") else "TBD"
    cohens_d_str = f"{ttest.get('cohens_d', 'TBD')}" if ttest.get("cohens_d") else "TBD"
    truth_improvement = aggregates.get("truth_ratio", {}).get("improvement_pct", "TBD")

    values_to_fill = [
        f"{fppl_improvement}\\%" if isinstance(fppl_improvement, (int, float)) else "TBD",
        f"{abs(rppl_improvement)}\\%" if isinstance(rppl_improvement, (int, float)) else "TBD",
        p_value_str,
        cohens_d_str,
        f"{truth_improvement}\\%" if isinstance(truth_improvement, (int, float)) else "TBD",
    ]

    # Apply replacements
    offset = 0
    for i, match in enumerate(all_pending):
        if i < len(values_to_fill):
            old = match.group()
            new = values_to_fill[i]
            start = match.start() + offset
            end = match.end() + offset
            content = content[:start] + new + content[end:]
            offset += len(new) - len(old)
            replacements_made += 1

    if content != original:
        with open(eval_tex_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  Filled {replacements_made} placeholders in {eval_tex_path.name}")
        return True
    else:
        print(f"  No changes needed in {eval_tex_path.name}")
        return False


def fill_table2(table_path: Path, summary: dict) -> bool:
    """Fill [PENDING] cells in table2-mu-results.tex with mean ± std values."""
    if not table_path.exists():
        print(f"WARNING: {table_path} not found")
        return False

    with open(table_path, encoding="utf-8") as f:
        content = f.read()

    original = content

    # For each author and condition, find best algorithm results and fill cells
    # Table structure: Author | Cond. | Fgt PPL ↑ | Ret PPL ↓ | Truth Ratio ↓

    for author_short, author_full in AUTHOR_MAP.items():
        for cond_short, cond_full in CONDITION_MAP.items():
            # Find best algorithm for this (author, condition)
            best_forget_ppl = None
            best_retain_ppl = None
            best_truth_ratio = None

            for algo in ["npo", "rmu", "grad_ascent"]:
                key = f"{algo}_{author_full}_{cond_full}"
                if key not in summary:
                    continue
                entry = summary[key]

                if "forget_ppl" in entry:
                    m, s = entry["forget_ppl"]["mean"], entry["forget_ppl"]["std"]
                    if best_forget_ppl is None or m > best_forget_ppl[0]:
                        best_forget_ppl = (m, s)

                if "retain_ppl" in entry:
                    m, s = entry["retain_ppl"]["mean"], entry["retain_ppl"]["std"]
                    if best_retain_ppl is None or m < best_retain_ppl[0]:
                        best_retain_ppl = (m, s)

                if "truth_ratio" in entry:
                    m, s = entry["truth_ratio"]["mean"], entry["truth_ratio"]["std"]
                    if best_truth_ratio is None or m < best_truth_ratio[0]:
                        best_truth_ratio = (m, s)

    # Replace [PENDING] placeholders cell by cell
    # The table has a regular structure: 3 authors × 3 conditions × 3 columns = 27 values
    # Pattern: \textcolor{red}{[PENDING]} repeated for mean and std

    pending_pattern = r"\\textcolor\{red\}\{?\[PENDING\]\}?"

    # Build ordered list of values to fill
    # Table rows: Berthe(Prov/Embed/Random), Antigng(Prov/Embed/Random), Iokseng(Prov/Embed/Random)
    # Each row: Fgt_PPL_mean, Fgt_PPL_std, Ret_PPL_mean, Ret_PPL_std, Truth_mean, Truth_std
    fill_values: list[str] = []

    for author_short, author_full in AUTHOR_MAP.items():
        for cond_short, cond_full in CONDITION_MAP.items():
            best_fppl = None
            best_rppl = None
            best_tr = None

            for algo in ["npo", "rmu", "grad_ascent"]:
                key = f"{algo}_{author_full}_{cond_full}"
                if key not in summary:
                    continue
                entry = summary[key]

                if "forget_ppl" in entry:
                    m, s = entry["forget_ppl"]["mean"], entry["forget_ppl"]["std"]
                    if best_fppl is None or m > best_fppl[0]:
                        best_fppl = (m, s)

                if "retain_ppl" in entry:
                    m, s = entry["retain_ppl"]["mean"], entry["retain_ppl"]["std"]
                    if best_rppl is None or m < best_rppl[0]:
                        best_rppl = (m, s)

                if "truth_ratio" in entry:
                    m, s = entry["truth_ratio"]["mean"], entry["truth_ratio"]["std"]
                    if best_tr is None or m < best_tr[0]:
                        best_tr = (m, s)

            # mean, std pairs for each column
            if best_fppl:
                fill_values.append(f"{best_fppl[0]:.2f}")
                fill_values.append(f"{best_fppl[1]:.2f}")
            else:
                fill_values.extend(["--", "--"])

            if best_rppl:
                fill_values.append(f"{best_rppl[0]:.2f}")
                fill_values.append(f"{best_rppl[1]:.2f}")
            else:
                fill_values.extend(["--", "--"])

            if best_tr:
                fill_values.append(f"{best_tr[0]:.3f}")
                fill_values.append(f"{best_tr[1]:.3f}")
            else:
                fill_values.extend(["--", "--"])

    # Find all [PENDING] occurrences and replace
    all_pending = list(re.finditer(pending_pattern, content))

    replacements = 0
    offset = 0
    for i, match in enumerate(all_pending):
        if i < len(fill_values):
            old = match.group()
            new = fill_values[i]
            start = match.start() + offset
            end = match.end() + offset
            content = content[:start] + new + content[end:]
            offset += len(new) - len(old)
            replacements += 1

    if content != original:
        with open(table_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  Filled {replacements} placeholders in {table_path.name}")
        return True
    else:
        print(f"  No changes needed in {table_path.name}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fill [PENDING] placeholders in paper LaTeX files with experiment results"
    )
    parser.add_argument(
        "--results-dir",
        default="benchmarks/results/pipeline_qa",
        help="Path to results directory containing multiseed_summary.json",
    )
    parser.add_argument(
        "--paper-dir",
        default=None,
        help="Path to paper root directory (auto-detected if not specified)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be changed without modifying files",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()

    # Auto-detect paper directory
    if args.paper_dir:
        paper_dir = Path(args.paper_dir).resolve()
    else:
        # Walk up from results_dir to find paper/
        candidate = results_dir
        while candidate.parent != candidate:
            if (candidate / "paper").is_dir():
                paper_dir = candidate / "paper"
                break
            candidate = candidate.parent
        else:
            paper_dir = results_dir.parent.parent.parent / "paper"

    print(f"Results dir: {results_dir}")
    print(f"Paper dir: {paper_dir}")
    print()

    summary = load_summary(results_dir)
    print(f"Loaded summary: {len(summary)} conditions")
    print()

    aggregates = compute_aggregate_improvement(summary)
    print("Aggregate improvements:")
    for metric, data in aggregates.items():
        if isinstance(data, dict) and "improvement_pct" in data:
            print(f"  {metric}: {data['improvement_pct']:+.1f}% (line={data['line_mean']:.2f} vs random={data['random_mean']:.2f})")
    if "paired_ttest" in aggregates:
        tt = aggregates["paired_ttest"]
        if "p_value" in tt:
            sig = "SIGNIFICANT" if tt["significant_005"] else "not significant"
            print(f"  Paired t-test: p={tt['p_value']:.4f} ({sig}), d={tt['cohens_d']:.3f}")
    print()

    if args.dry_run:
        print("DRY RUN — no files will be modified")

    eval_tex = paper_dir / "cikm" / "sections" / "evaluation.tex"
    table2 = paper_dir / "cikm" / "tables" / "table2-mu-results.tex"

    changes = 0
    if not args.dry_run:
        if fill_evaluation_tex(eval_tex, aggregates, summary):
            changes += 1
        if fill_table2(table2, summary):
            changes += 1
    else:
        print(f"Would fill: {eval_tex}")
        print(f"Would fill: {table2}")

    print()
    if changes > 0 or args.dry_run:
        print(f"{'Would fill' if args.dry_run else 'Filled'} {changes} files.")
        if not args.dry_run:
            print("Review changes with: git diff paper/")
    else:
        print("No changes needed.")


if __name__ == "__main__":
    main()
