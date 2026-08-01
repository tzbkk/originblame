#!/usr/bin/env python3
"""Two-phase reconcile benchmark (hash + semantic).

Runs reconcile with hash matching (Pass 1) + embedding similarity (Pass 2)
for each dataset. Skips datasets that already have result files.

Writes <dataset>/reconcile_semantic.json per dataset.

Usage:
    python3 recon_semantic.py
    python3 recon_semantic.py --datasets huggingface-zhwiki-10k-ob huggingface-zhwiki-all-ob
    python3 recon_semantic.py --embedding-api http://localhost:1234/v1
    python3 recon_semantic.py --force
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent

sys.path.insert(0, str(REPO_ROOT / "rust-originblame" / "python" / "src"))
sys.path.insert(0, str(REPO_ROOT / "rust-originblame" / "python" / "packages" / "ob-util" / "src"))

from ob_util.reconcile import reconcile

RESULTS_DIR = REPO_ROOT / "benchmarks" / "results" / "pipeline_v2"

ALL_DATASETS = [
    "huggingface-zhwiki-1k-ob",
    "huggingface-zhwiki-10k-ob",
    "huggingface-zhwiki-100k-ob",
    "huggingface-zhwiki-all-ob",
]

MODEL = "nomic-embed-text-v1.5"
THRESHOLD = 0.85
SEED = 42


def _mutate_chars(s: str, rng: random.Random, char_pct: float = 0.10) -> str:
    chars = list(s)
    alpha = [i for i, c in enumerate(chars) if c.isalpha()]
    n = max(1, int(len(alpha) * char_pct))
    for idx in rng.sample(alpha, min(n, len(alpha))):
        chars[idx] = chr((ord(chars[idx]) + rng.randint(1, 25)) % 128)
    return "".join(chars)


def mutate(data_file: Path, output_file: Path) -> dict:
    rng = random.Random(SEED)
    with open(data_file, encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]

    cats = []
    for _ in lines:
        r = rng.random()
        cats.append("edit" if r < 0.10 else ("delete" if r < 0.15 else "keep"))

    result = []
    for i, line in enumerate(lines):
        if cats[i] == "delete":
            continue
        if cats[i] == "edit":
            try:
                rec = json.loads(line)
                if "text" in rec:
                    rec["text"] = _mutate_chars(rec["text"], rng)
                    result.append(json.dumps(rec, ensure_ascii=False))
                else:
                    result.append(_mutate_chars(line, rng))
            except (json.JSONDecodeError, KeyError):
                result.append(_mutate_chars(line, rng))
        else:
            result.append(line)

    n_ins = max(1, int(len(lines) * 0.05))
    for _ in range(n_ins):
        src = rng.choice(lines)
        try:
            rec = json.loads(src)
            if "text" in rec:
                rec["text"] = _mutate_chars(rec["text"], rng, char_pct=0.15)
                ins_line = json.dumps(rec, ensure_ascii=False)
            else:
                ins_line = _mutate_chars(src, rng, char_pct=0.15)
        except (json.JSONDecodeError, KeyError):
            ins_line = _mutate_chars(src, rng, char_pct=0.15)
        result.insert(rng.randint(0, len(result)), ins_line)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for line in result:
            f.write(line + "\n")

    return {
        "original_lines": len(lines),
        "edited_lines": cats.count("edit"),
        "deleted_lines": cats.count("delete"),
        "inserted_lines": n_ins,
        "final_lines": len(result),
    }


def run(ds_name: str, embedding_api: str, results_dir: Path, force: bool = False) -> None:
    ob_dir = results_dir / ds_name
    data_file = ob_dir / "jsonl" / "data.jsonl"
    if not data_file.is_file():
        print(f"SKIP {ds_name}: no data.jsonl", flush=True)
        return

    result_file = ob_dir / "reconcile_semantic.json"
    if result_file.exists() and not force:
        print(f"SKIP {ds_name}: {result_file.name} already exists", flush=True)
        return

    work = ob_dir / "recon_work"
    if work.exists():
        shutil.rmtree(work)

    print(f"=== {ds_name} ===", flush=True)

    work.mkdir()
    work_data = work / "data.jsonl"
    shutil.copy2(data_file, work_data)
    shutil.copytree(ob_dir / ".ob", work / ".ob")

    print(f"  precompute embeddings...", flush=True)
    t0 = time.time()
    reconcile(
        str(work_data),
        model=MODEL,
        threshold=THRESHOLD,
        ob_dir=work,
        embedding_api=embedding_api,
        compute_all_embeddings=True,
    )
    t_pre = time.time() - t0
    n_lines = sum(1 for _ in open(work_data, encoding="utf-8"))
    print(f"    {n_lines} embeddings in {t_pre:.1f}s", flush=True)

    backup = work / "data_orig.jsonl"
    shutil.move(str(work_data), str(backup))
    mut = mutate(backup, work_data)
    print(
        f"  mutate: orig={mut['original_lines']} "
        f"edit={mut['edited_lines']} del={mut['deleted_lines']} "
        f"ins={mut['inserted_lines']} → {mut['final_lines']}",
        flush=True,
    )

    print(f"  reconcile...", flush=True)
    t1 = time.time()
    r = reconcile(
        str(work_data),
        model=MODEL,
        threshold=THRESHOLD,
        ob_dir=work,
        embedding_api=embedding_api,
    )
    t_rec = time.time() - t1

    matched = r.hash_matched + r.semantic_matched
    recovery = round(matched / mut["final_lines"] * 100, 1) if mut["final_lines"] else 0.0

    print(
        f"  hash={r.hash_matched} semantic={r.semantic_matched} "
        f"new={r.new_lines} orphans={r.orphans} "
        f"recovery={recovery}% time={t_rec:.1f}s",
        flush=True,
    )

    result = {
        "timestamp": time.strftime("%Y-%m-%dT%H%M%S"),
        "dataset": ds_name,
        "seed": SEED,
        "model": MODEL,
        "threshold": THRESHOLD,
        "mutation": mut,
        "hash_matched": r.hash_matched,
        "semantic_matched": r.semantic_matched,
        "new_lines": r.new_lines,
        "orphans": r.orphans,
        "recovery_pct": recovery,
        "precompute_embeddings_s": round(t_pre, 1),
        "reconcile_s": round(t_rec, 1),
    }
    result_file.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  saved → {result_file}", flush=True)

    shutil.rmtree(work, ignore_errors=True)
    print(f"  cleaned up {work}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-phase reconcile benchmark")
    parser.add_argument(
        "--datasets", nargs="+", default=ALL_DATASETS,
        help="Datasets to run (default: all zhwiki scales)",
    )
    parser.add_argument(
        "--embedding-api", default=os.environ.get("EMBEDDING_API", "http://localhost:1234/v1"),
        help="OpenAI-compatible embedding API URL (env: EMBEDDING_API)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run even if result file exists",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=RESULTS_DIR,
        help="Pipeline v2 results directory",
    )
    args = parser.parse_args()
    rd = args.results_dir.resolve()

    for ds in args.datasets:
        run(ds, args.embedding_api, rd, args.force)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
