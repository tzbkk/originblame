#!/usr/bin/env python3
"""Author contribution statistics analysis for any OB dataset.

Usage:
  python analyze_authors.py <ob-dir> [output-dir] [--figures-dir PATH]

Examples:
  python analyze_authors.py benchmarks/results/pipeline_v2/huggingface-zhwiki-all-ob
  python analyze_authors.py benchmarks/results/pipeline_qa/qa_chatml benchmarks/results/pipeline_qa
"""

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

_RUST_OB = Path(__file__).resolve().parents[2] / "rust-originblame" / "python" / "src"
if _RUST_OB.exists():
    sys.path.insert(0, str(_RUST_OB))

import _ob_native as R


def compute_gini(values: list[int]) -> float:
    if not values:
        return 0.0
    n = len(values)
    total = sum(values)
    if total == 0:
        return 0.0
    sorted_v = sorted(values)
    area = sum((n - i) * v for i, v in enumerate(sorted_v))
    return (n + 1 - 2 * area / total) / n


def is_ip_author(name: str, email: str) -> bool:
    if "@" not in email:
        return True
    parts = name.replace(".", "").replace(":", "")
    if parts.isdigit() and len(name) > 6:
        return True
    return False


def is_bot(name: str) -> bool:
    return "bot" in name.lower()


def bracket(count: int) -> str:
    if count == 1:
        return "1"
    elif count <= 10:
        return "2-10"
    elif count <= 50:
        return "11-50"
    elif count <= 100:
        return "51-100"
    elif count <= 500:
        return "101-500"
    elif count <= 1000:
        return "501-1000"
    else:
        return "1001+"


BRACKET_ORDER = ["1", "2-10", "11-50", "51-100", "101-500", "501-1000", "1001+"]


def analyze(ob_dir: str, output_dir: Path, figures_dir: Path):
    has_mpl = False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        has_mpl = True
    except ImportError:
        print("WARNING: matplotlib not available, skipping figures")

    t0 = time.time()
    dataset_name = Path(ob_dir).name
    print(f"[{dataset_name}] Loading from {ob_dir}/.ob/ ...")

    authors = {}
    revoked_author_ids = set()
    for a in R.shard_iterate_all(ob_dir, "authors"):
        authors[a["id"]] = {"name": a["name"], "email": a["email"]}
        if a.get("revoked"):
            revoked_author_ids.add(a["id"])
    print(f"  Authors: {len(authors):,} ({len(revoked_author_ids)} revoked)")

    author_sections = defaultdict(set)
    author_coauthors = defaultdict(set)
    author_year_counts = defaultdict(Counter)
    section_count = 0
    total_author_slots = 0

    for s in R.shard_iterate_all(ob_dir, "sections"):
        section_count += 1
        path = s["path"]
        aids = s["authors"]
        year = s.get("year", "unknown")
        total_author_slots += len(aids)

        for aid in aids:
            author_sections[aid].add(path)
            author_year_counts[aid][year] += 1

        for i, a1 in enumerate(aids):
            for a2 in aids[i + 1:]:
                author_coauthors[a1].add(a2)
                author_coauthors[a2].add(a1)

    elapsed_load = time.time() - t0
    print(f"  Sections: {section_count:,}  Slots: {total_author_slots:,}  Time: {elapsed_load:.1f}s")

    section_counts = []
    per_author = {}

    for aid, meta in authors.items():
        secs = author_sections.get(aid, set())
        n_sections = len(secs)
        section_counts.append(n_sections)

        # Extract document title from path: "raw/Title#Heading" or "qa/Title"
        docs = set()
        for p in secs:
            parts = p.split("/", 1)
            title = parts[1].split("#")[0] if len(parts) > 1 else p.split("#")[0]
            docs.add(title)

        per_author[aid] = {
            "name": meta["name"],
            "email": meta["email"],
            "record_count": n_sections,
            "document_count": len(docs),
            "contribution_pct": round(n_sections / section_count * 100, 4) if section_count else 0,
            "slot_contribution_pct": round(n_sections / total_author_slots * 100, 4) if total_author_slots else 0,
            "coauthor_count": len(author_coauthors.get(aid, set())),
            "is_bot": is_bot(meta["name"]),
            "is_ip": is_ip_author(meta["name"], meta["email"]),
            "is_revoked": aid in revoked_author_ids,
            "year_distribution": dict(author_year_counts.get(aid, Counter())),
        }

    bracket_counts = Counter(bracket(c) for c in section_counts)
    bot_authors = [a for a in per_author.values() if a["is_bot"]]
    human_authors = [a for a in per_author.values() if not a["is_bot"]]
    ip_authors = [a for a in per_author.values() if a["is_ip"]]
    top10 = sorted(per_author.items(), key=lambda x: x[1]["record_count"], reverse=True)[:10]

    global_stats = {
        "total_authors": len(authors),
        "total_sections": section_count,
        "total_author_section_slots": total_author_slots,
        "mean_records_per_author": round(sum(section_counts) / len(section_counts), 4) if section_counts else 0,
        "median_records_per_author": sorted(section_counts)[len(section_counts) // 2] if section_counts else 0,
        "gini_coefficient": round(compute_gini(section_counts), 4),
        "bot_authors": len(bot_authors),
        "human_authors": len(human_authors),
        "ip_authors": len(ip_authors),
        "revoked_authors": len(revoked_author_ids),
        "bracket_distribution": {b: bracket_counts.get(b, 0) for b in BRACKET_ORDER},
    }

    stats_output = {
        "dataset": dataset_name,
        "ob_dir": ob_dir,
        "global": global_stats,
        "top10": [{"rank": i + 1, **v} for i, (_, v) in enumerate(top10)],
        "authors": per_author,
    }

    stats_path = output_dir / "author_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats_output, f, indent=2, ensure_ascii=False)
    print(f"  Wrote {stats_path}")

    # Evidence
    ev = [
        f"Author Distribution Analysis — {dataset_name}",
        f"{'=' * 60}",
        f"Date: {time.strftime('%Y-%m-%d')}",
        f"Dataset: {ob_dir}",
        "",
        f"DATA SUMMARY",
        f"{'-' * 12}",
        f"- Total authors: {global_stats['total_authors']:,}",
        f"- Total sections: {global_stats['total_sections']:,}",
        f"- Total author-section slots: {global_stats['total_author_section_slots']:,}",
        f"- Mean records/author: {global_stats['mean_records_per_author']:.2f}",
        f"- Median records/author: {global_stats['median_records_per_author']}",
        f"- Gini coefficient: {global_stats['gini_coefficient']:.4f}",
        f"- Bot authors: {global_stats['bot_authors']} ({global_stats['bot_authors']/global_stats['total_authors']*100:.1f}%)",
        f"- Human authors: {global_stats['human_authors']} ({global_stats['human_authors']/global_stats['total_authors']*100:.1f}%)",
        f"- IP authors: {global_stats['ip_authors']} ({global_stats['ip_authors']/global_stats['total_authors']*100:.1f}%)",
        f"- Revoked authors: {global_stats['revoked_authors']}",
        "",
        f"BRACKET DISTRIBUTION",
        f"{'-' * 21}",
    ]
    for b in BRACKET_ORDER:
        cnt = bracket_counts.get(b, 0)
        pct = cnt / len(section_counts) * 100 if section_counts else 0
        ev.append(f"{b:>10} records: {cnt:>6} ({pct:.1f}%)")
    ev += ["", "TOP-10 AUTHORS", f"{'-' * 14}"]
    for i, (_, v) in enumerate(top10):
        ev.append(f"{i+1}. {v['name']}: {v['record_count']} ({v['contribution_pct']:.1f}%), coauthors={v['coauthor_count']}, bot={v['is_bot']}")

    evidence_path = Path(f".sisyphus/evidence/author-stats-{dataset_name}.txt")
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text("\n".join(ev))

    if has_mpl:
        print(f"  Generating figures...")
        plt.rcParams.update({"font.size": 11, "figure.figsize": (7, 4)})

        nonzero = [c for c in section_counts if c > 0]
        if nonzero:
            fig, ax1 = plt.subplots()
            log_bins = np.logspace(0, np.log10(max(nonzero)), 50)
            ax1.hist(nonzero, bins=log_bins, color="#4C72B0", alpha=0.8, edgecolor="white", linewidth=0.5)
            ax1.set_xscale("log")
            ax1.set_xlabel("Records per author (log scale)")
            ax1.set_ylabel("Number of authors")
            ax2 = ax1.twinx()
            sorted_c = np.sort(nonzero)
            cdf = np.arange(1, len(sorted_c) + 1) / len(sorted_c)
            ax2.plot(sorted_c, cdf, color="#DD8452", linewidth=2)
            ax2.set_ylabel("CDF")
            ax2.set_ylim(0, 1.05)
            ax1.set_title(f"Author Contribution Distribution ({dataset_name}, {section_count:,} sections)")
            fig.tight_layout()
            fig.savefig(figures_dir / f"fig-author-histogram-{dataset_name}.pdf", dpi=150)
            plt.close(fig)

        if top10:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            names = [v["name"][:20] for _, v in top10]
            counts = [v["record_count"] for _, v in top10]
            colors = ["#4C72B0" for _ in top10]
            bars = ax.barh(range(len(names)), counts, color=colors)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names)
            ax.invert_yaxis()
            ax.set_xlabel("Number of sections")
            ax.set_title(f"Top-10 Authors ({dataset_name})")
            for bar, cnt in zip(bars, counts):
                ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{cnt:,} ({cnt/section_count*100:.1f}%)", va="center", fontsize=9)
            fig.tight_layout()
            fig.savefig(figures_dir / f"fig-author-contribution-{dataset_name}.pdf", dpi=150)
            plt.close(fig)

        coauthor_vals = [v["coauthor_count"] for v in per_author.values()]
        nonzero_co = [c for c in coauthor_vals if c > 0]
        if nonzero_co:
            fig, ax = plt.subplots()
            log_bins_co = np.logspace(0, np.log10(max(nonzero_co)), 50)
            ax.hist(nonzero_co, bins=log_bins_co, color="#55A868", alpha=0.8, edgecolor="white", linewidth=0.5)
            ax.set_xscale("log")
            ax.set_xlabel("Co-authors per author (log scale)")
            ax.set_ylabel("Number of authors")
            ax.set_title(f"Co-author Distribution ({dataset_name})")
            fig.tight_layout()
            fig.savefig(figures_dir / f"fig-author-coauthors-{dataset_name}.pdf", dpi=150)
            plt.close(fig)

        print(f"  Figures saved to {figures_dir}/")

    elapsed = time.time() - t0
    print(f"[{dataset_name}] Done in {elapsed:.1f}s — {stats_path}")
    return stats_output


def main():
    parser = argparse.ArgumentParser(description="Author contribution analysis for any OB dataset")
    parser.add_argument("ob_dir", help="Path to dataset with .ob/ directory")
    parser.add_argument("output_dir", nargs="?", default=None,
                        help="Directory for output JSON (default: same as ob_dir)")
    parser.add_argument("--figures-dir", default="paper/cikm/figures",
                        help="Directory for output figures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.ob_dir)
    figures_dir = Path(args.figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    analyze(args.ob_dir, output_dir, figures_dir)


if __name__ == "__main__":
    main()
