#!/usr/bin/env python3
"""Quick verification: page-level multi-prototype vs average-prototype embedding baseline.

Compares two methods for computing embedding similarity between an author's records
and all other records:

  Method A (current): average all author embeddings → 1 vector → cosine similarity
  Method B (page-prototype): one centroid per wiki page → max cosine similarity

Metrics:
  - sim_gap: mean_sim(author_records) - mean_sim(non_author_records)
  - lift: mean_sim(author_records) / mean_sim(random_subset)
  - top-N overlap: how many of the true author records appear in top-N by similarity
"""

import json
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ob.authors import query_authors
from ob.storage import LAYER_MANIFEST, LAYER_SECTION, shard_iterate_all

OB_DIR = Path("/home/hxue/Projects/originblame/benchmarks/results/pipeline_qa/qa_chatml")
DATA_FILE = OB_DIR / "jsonl/data.jsonl"
EMBED_API = "http://localhost:1234/v1/embeddings"
EMBED_MODEL = "text-embedding-nomic-embed-text-v1.5"
BATCH_SIZE = 256

AUTHORS = ["InternetArchiveBot", "佛祖西来", "Ohtashinichiro"]


def compute_line_hash(data: dict) -> str:
    import hashlib

    content = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def extract_text(record: dict) -> str:
    if "text" in record:
        return record["text"]
    if "messages" in record:
        return " ".join(m.get("content", "") for m in record["messages"])
    return ""


def load_records():
    records = []
    hash_to_idx = {}
    texts = []
    with open(DATA_FILE, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            lh = compute_line_hash(rec)
            records.append(rec)
            hash_to_idx[lh] = len(records) - 1
            texts.append(extract_text(rec))
    return records, hash_to_idx, texts


def load_provenance():
    manifest = {}
    for rec in shard_iterate_all(OB_DIR, LAYER_MANIFEST):
        manifest[rec["line_hash"]] = rec.get("sources", [])
    sections = {}
    for rec in shard_iterate_all(OB_DIR, LAYER_SECTION):
        sections[rec["section_hash"]] = rec
    return manifest, sections


def get_author_records(author_name, hash_to_idx, manifest, sections):
    author_recs = query_authors(OB_DIR, name=author_name)
    if not author_recs:
        return set(), {}
    author_ids = {r["id"] for r in author_recs}
    author_sections = set()
    for sh, srec in sections.items():
        if author_ids & set(srec.get("authors", [])):
            author_sections.add(sh)

    author_indices = set()
    section_map = {}
    for lh, sources in manifest.items():
        matching = [s for s in sources if s in author_sections]
        if matching:
            idx = hash_to_idx.get(lh)
            if idx is not None:
                author_indices.add(idx)
                for s in matching:
                    section_map.setdefault(s, []).append(idx)

    page_map = defaultdict(list)
    for sec_hash, indices in section_map.items():
        srec = sections.get(sec_hash)
        if srec:
            path = srec["path"]
            page = path.split("#")[0] if "#" in path else path
            for i in indices:
                page_map[page].append(i)

    return author_indices, page_map


def embed_texts(texts):
    all_embs = []
    n = len(texts)
    for start in range(0, n, BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        end = min(start + BATCH_SIZE, n)
        print(f"  Embedding: {end}/{n}", end="\r")
        payload = json.dumps({"model": EMBED_MODEL, "input": batch}).encode("utf-8")
        req = urllib.request.Request(
            f"{EMBED_API.rstrip('/')}",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        batch_embs = [d["embedding"] for d in sorted(result["data"], key=lambda x: x["index"])]
        all_embs.append(np.array(batch_embs, dtype=np.float32))
    print(f"  Embedding: {n}/{n} done")
    return np.vstack(all_embs)


def evaluate_method(author_indices, scores, total_lines, method_name):
    author_scores = scores[list(author_indices)]
    non_author_indices = list(set(range(total_lines)) - author_indices)
    non_author_scores = scores[non_author_indices]

    mean_author = np.mean(author_scores)
    mean_non_author = np.mean(non_author_scores)
    sim_gap = mean_author - mean_non_author

    rng = np.random.RandomState(42)
    random_indices = rng.choice(non_author_indices, size=len(author_indices), replace=False)
    mean_random = np.mean(scores[random_indices])
    lift = mean_author / (mean_random + 1e-10)

    target_size = len(author_indices)
    top_n = np.argsort(scores)[-target_size:]
    overlap = len(set(top_n) & author_indices)

    print(f"\n  {method_name}:")
    print(f"    Author mean sim:   {mean_author:.4f}")
    print(f"    Non-author mean:   {mean_non_author:.4f}")
    print(f"    Sim gap:           {sim_gap:.4f}")
    print(f"    Lift over random:  {lift:.2f}x")
    print(f"    Top-{target_size} overlap:  {overlap}/{target_size} ({100*overlap/target_size:.1f}%)")

    return {"sim_gap": sim_gap, "lift": lift, "overlap": overlap, "overlap_pct": 100 * overlap / target_size}


def main():
    print("Loading records...")
    records, hash_to_idx, texts = load_records()
    total = len(records)
    print(f"  {total} records loaded")

    print("Loading provenance...")
    manifest, sections = load_provenance()
    print(f"  {len(manifest)} manifest, {len(sections)} sections")

    print("Computing embeddings...")
    t0 = time.time()
    embs = embed_texts(texts)
    print(f"  Shape: {embs.shape}, time: {time.time()-t0:.1f}s")

    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs_normed = embs / (norms + 1e-10)

    results = {}
    for author_name in AUTHORS:
        print(f"\n{'='*60}")
        print(f"Author: {author_name}")

        author_indices, page_map = get_author_records(author_name, hash_to_idx, manifest, sections)
        if not author_indices:
            print("  NOT FOUND, skipping")
            continue
        print(f"  Records: {len(author_indices)}, Pages: {len(page_map)}")

        author_indices = set(author_indices)
        non_author_indices = np.array(list(set(range(total)) - author_indices))

        # Method A: average prototype
        author_avg = embs_normed[list(author_indices)].mean(axis=0)
        author_avg = author_avg / (np.linalg.norm(author_avg) + 1e-10)
        scores_avg = embs_normed[non_author_indices] @ author_avg

        # Also score author's own records for sim_gap calculation
        scores_author_self = embs_normed[list(author_indices)] @ author_avg
        full_scores_avg = np.zeros(total)
        full_scores_avg[non_author_indices] = scores_avg
        full_scores_avg[list(author_indices)] = scores_author_self
        res_avg = evaluate_method(author_indices, full_scores_avg, total, "Method A (average)")

        # Method B: page-level prototypes + max-similarity
        page_centroids = []
        for page, indices in page_map.items():
            centroid = embs_normed[indices].mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
            page_centroids.append(centroid)
        page_centroids = np.array(page_centroids)
        print(f"  Page centroids: {page_centroids.shape}")

        sim_matrix = embs_normed @ page_centroids.T
        full_scores_max = np.max(sim_matrix, axis=1)
        res_max = evaluate_method(author_indices, full_scores_max, total, "Method B (page max-sim)")

        results[author_name] = {
            "records": len(author_indices),
            "pages": len(page_map),
            "avg": res_avg,
            "page_max": res_max,
            "improvement_gap": res_max["sim_gap"] / (res_avg["sim_gap"] + 1e-10),
            "improvement_lift": res_max["lift"] / (res_avg["lift"] + 1e-10),
            "improvement_overlap": res_max["overlap_pct"] / (res_avg["overlap_pct"] + 1e-10),
        }

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Author':<20} {'Records':>7} {'Pages':>6} | {'Avg gap':>8} {'Page gap':>9} {'Improv':>7} | {'Avg lift':>8} {'Page lift':>9} {'Improv':>7}")
    print("-" * 100)
    for name, r in results.items():
        print(
            f"{name:<20} {r['records']:>7} {r['pages']:>6} | "
            f"{r['avg']['sim_gap']:>8.4f} {r['page_max']['sim_gap']:>9.4f} {r['improvement_gap']:>6.2f}x | "
            f"{r['avg']['lift']:>7.2f}x {r['page_max']['lift']:>9.2f}x {r['improvement_lift']:>6.2f}x"
        )

    out_file = Path(__file__).resolve().parent / "results" / "page_prototype_verification.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=float)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
