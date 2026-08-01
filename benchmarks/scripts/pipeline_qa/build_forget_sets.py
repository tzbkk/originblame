#!/usr/bin/env python3
"""Build line-level and random forget sets from ob provenance.

Given author names and a data directory with .ob/ provenance, produces
JSON files mapping each (author, forget_set_type) to a list of line indices
in data.jsonl.

Forget set types:
  line       — ob provenance: exact lines authored by the target author
  random     — random subset same size as line-level
"""
  
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from ob.authors import query_authors
from ob.storage import LAYER_MANIFEST, LAYER_SECTION, shard_iterate_all


def compute_line_hash(data: dict) -> str:
    try:
        from _ob_native import compute_hash

        return compute_hash(data)
    except ImportError:
        content = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(content.encode("utf-8")).hexdigest()


def load_data(data_file: Path) -> tuple[list[int], dict[str, int], list[str]]:
    """Load data.jsonl: compute hash per line, return (all_indices, hash_to_idx, texts).

    texts[i] is the "text" field of record at line index i (for embedding/ngram).
    """
    all_indices: list[int] = []
    hash_to_idx: dict[str, int] = {}
    texts: list[str] = []
    with open(data_file, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            lh = compute_line_hash(record)
            all_indices.append(idx)
            hash_to_idx[lh] = idx
            if "text" in record:
                texts.append(record["text"])
            elif "messages" in record:
                texts.append(" ".join(m.get("content", "") for m in record["messages"]))
            else:
                texts.append("")
    return all_indices, hash_to_idx, texts


def load_manifest_index(ob_dir: Path) -> dict[str, list[str]]:
    """Load all manifest records, return {line_hash: [section_hash, ...]}."""
    manifest: dict[str, list[str]] = {}
    for rec in shard_iterate_all(ob_dir, LAYER_MANIFEST):
        manifest[rec["line_hash"]] = rec.get("sources", [])
    return manifest


def load_section_index(ob_dir: Path) -> dict[str, dict]:
    """Load all section records, return {section_hash: section_record}."""
    sections: dict[str, dict] = {}
    for rec in shard_iterate_all(ob_dir, LAYER_SECTION):
        sections[rec["section_hash"]] = rec
    return sections


def find_author_section_hashes(
    author_ids: set[str], sections: dict[str, dict]
) -> set[str]:
    """Find all section hashes where any of the given author_ids contributed."""
    result = set()
    for sh, srec in sections.items():
        if author_ids & set(srec.get("authors", [])):
            result.add(sh)
    return result


def build_line_level(
    author_section_hashes: set[str],
    manifest: dict[str, list[str]],
    hash_to_idx: dict[str, int],
) -> set[int]:
    """Find line indices where the author contributed (via their sections)."""
    indices = set()
    for lh, sources in manifest.items():
        if any(s in author_section_hashes for s in sources):
            idx = hash_to_idx.get(lh)
            if idx is not None:
                indices.add(idx)
    return indices


# ── Token-level forget sets ────────────────────────────────────────────────────


def load_token_index(ob_dir: Path, tokenizer: str = "gpt2") -> list[dict] | None:
    """Read all token-index entries from .ob/token-index.{tokenizer}/NNN files.

    Returns a list of dicts with keys: token_count, sources, tokenizer, revoked.
    Entry i corresponds to the i-th tracked document (position-based).
    Returns None if the token-index directory does not exist.
    """
    ti_dir = ob_dir / ".ob" / f"token-index.{tokenizer}"
    if not ti_dir.is_dir():
        return None

    files: list[tuple[int, Path]] = []
    for entry in ti_dir.iterdir():
        name = entry.name
        if entry.is_file():
            try:
                num = int(name)
            except ValueError:
                continue
            files.append((num, entry))
    files.sort(key=lambda x: x[0])

    entries: list[dict] = []
    for _, path in files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entries.append(json.loads(line))
    return entries


def build_token_level(
    author_section_hashes: set[str],
    ob_dir: Path,
    tokenizer: str = "gpt2",
) -> set[int]:
    """Find line indices of documents where the author has token-level contributions.

    Uses .ob/token-index.{tokenizer}/ to find token-index entries whose sources
    overlap with the author's section hashes. Entry position maps 1:1 to line
    index in data.jsonl (position-based).

    This differs from line-level because token-index attribution is per-document:
    a section may contribute to many lines, but token attribution may only cover
    a subset. In practice, token-level is a subset of line-level.
    """
    entries = load_token_index(ob_dir, tokenizer)
    if entries is None:
        print(f"  WARNING: token-index.{tokenizer} not found, skipping token-level")
        return set()

    indices: set[int] = set()
    for idx, entry in enumerate(entries):
        if entry.get("revoked"):
            continue
        sources = set(entry.get("sources", []))
        if sources & author_section_hashes:
            indices.add(idx)
    return indices


def build_token_mask(
    author_section_hashes: set[str],
    ob_dir: Path,
    tokenizer: str = "gpt2",
) -> dict:
    """Build token-level mask data for training.

    Returns a dict with:
      line_indices: sorted list of line indices with any author tokens
      token_counts: {line_idx: author_token_count} for matching entries
      total_author_tokens: sum of token counts across matching entries
      total_tokens: sum of all (non-revoked) token counts in the index

    Note: For now, training scripts treat token_mask the same as token (using
    line indices only). Per-token masking within documents is a future enhancement.
    """
    entries = load_token_index(ob_dir, tokenizer)
    if entries is None:
        print(f"  WARNING: token-index.{tokenizer} not found, skipping token_mask")
        return {
            "line_indices": [],
            "token_counts": {},
            "total_author_tokens": 0,
            "total_tokens": 0,
        }

    line_indices: list[int] = []
    token_counts: dict[int, int] = {}
    total_author_tokens = 0
    total_tokens = 0

    for idx, entry in enumerate(entries):
        if entry.get("revoked"):
            continue
        tc = entry.get("token_count", 0)
        total_tokens += tc
        sources = set(entry.get("sources", []))
        if sources & author_section_hashes:
            line_indices.append(idx)
            token_counts[idx] = tc
            total_author_tokens += tc

    return {
        "line_indices": sorted(line_indices),
        "token_counts": token_counts,
        "total_author_tokens": total_author_tokens,
        "total_tokens": total_tokens,
    }


def build_random(total_lines: int, target_size: int, seed: int = 42) -> set[int]:
    """Random sample of line indices with given size and seed."""
    rng = random.Random(seed)
    return set(rng.sample(range(total_lines), target_size))


# ── Embedding similarity forget set ──────────────────────────────────────────


def embed_texts_api(
    texts: list[str], api_url: str, model: str, batch_size: int = 256
) -> np.ndarray:
    """Embed texts via OpenAI-compatible API."""
    import urllib.request

    all_embeddings: list[np.ndarray] = []
    url = f"{api_url.rstrip('/')}/embeddings"
    n = len(texts)
    for start in range(0, n, batch_size):
        batch = texts[start : start + batch_size]
        end = min(start + batch_size, n)
        print(f"  Embedding via API: {end}/{n}")
        payload = json.dumps({"model": model, "input": batch}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        batch_embs = [
            d["embedding"] for d in sorted(result["data"], key=lambda x: x["index"])
        ]
        all_embeddings.append(np.array(batch_embs, dtype=np.float32))
    return np.vstack(all_embeddings)


def embed_texts_local(
    texts: list[str], model_name: str, batch_size: int = 256
) -> np.ndarray:
    """Embed texts using local sentence-transformers model."""
    from sentence_transformers import SentenceTransformer

    print(f"  Loading local model: {model_name}")
    model = SentenceTransformer(model_name)
    all_embeddings: list[np.ndarray] = []
    n = len(texts)
    for start in range(0, n, batch_size):
        batch = texts[start : start + batch_size]
        end = min(start + batch_size, n)
        print(f"  Embedding locally: {end}/{n}")
        embs = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        all_embeddings.append(embs)
    return np.vstack(all_embeddings)


def compute_all_embeddings(
    texts: list[str], api_url: str | None, model_name: str = "nomic-embed-text-v1.5"
) -> np.ndarray:
    """Compute embeddings for all texts using API or local model."""
    if api_url:
        print(f"Computing embeddings via API: {api_url}")
        return embed_texts_api(texts, api_url, model_name)
    else:
        print(f"Computing embeddings via local model: {model_name}")
        return embed_texts_local(texts, model_name)


def build_page_map(
    author_section_hashes: set[str],
    sections: dict[str, dict],
    manifest: dict[str, list[str]],
    hash_to_idx: dict[str, int],
) -> dict[str, list[int]]:
    author_indices_by_page: dict[str, list[int]] = {}
    for lh, sources in manifest.items():
        matching = [s for s in sources if s in author_section_hashes]
        if not matching:
            continue
        idx = hash_to_idx.get(lh)
        if idx is None:
            continue
        for sec_hash in matching:
            srec = sections.get(sec_hash)
            if srec:
                path = srec["path"]
                page = path.split("#")[0] if "#" in path else path
                author_indices_by_page.setdefault(page, []).append(idx)
    return author_indices_by_page


def build_page_prototype_sim(
    author_line_indices: set[int],
    author_indices_by_page: dict[str, list[int]],
    all_embeddings: np.ndarray,
    total_lines: int,
    target_size: int,
) -> set[int]:
    """Top-N lines by max cosine similarity across page-level prototype centroids.

    Excludes the author's own lines from candidates.
    """
    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    embs_normed = all_embeddings / (norms + 1e-10)

    page_centroids = []
    for page, indices in author_indices_by_page.items():
        centroid = embs_normed[indices].mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
        page_centroids.append(centroid)
    page_centroids = np.array(page_centroids)

    sim_matrix = embs_normed @ page_centroids.T
    max_sim = np.max(sim_matrix, axis=1)

    candidate_mask = np.ones(total_lines, dtype=bool)
    candidate_mask[list(author_line_indices)] = False
    candidate_indices = np.where(candidate_mask)[0]
    candidate_scores = max_sim[candidate_indices]

    top_k = min(target_size, len(candidate_indices))
    top_local = np.argsort(candidate_scores)[-top_k:]
    return set(candidate_indices[top_local].tolist())


# ── N-gram similarity forget set ─────────────────────────────────────────────


def char_ngrams(text: str, n: int = 5) -> set[str]:
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def build_ngram(
    author_line_indices: set[int],
    texts: list[str],
    total_lines: int,
    target_size: int,
    n: int = 5,
) -> set[int]:
    """Top-N lines by char n-gram Jaccard similarity to author's combined profile.

    Excludes the author's own lines from candidates.
    """
    print("  Computing n-gram profiles for all lines...")
    all_ngram_sets = [char_ngrams(t, n) for t in texts]

    author_profile: set[str] = set()
    for idx in author_line_indices:
        author_profile |= all_ngram_sets[idx]

    print(f"  Author profile: {len(author_profile)} unique {n}-grams")

    candidate_indices = [i for i in range(total_lines) if i not in author_line_indices]

    similarities: list[tuple[int, float]] = []
    for i in candidate_indices:
        s = all_ngram_sets[i]
        if not s or not author_profile:
            similarities.append((i, 0.0))
            continue
        intersection = len(s & author_profile)
        union = len(s | author_profile)
        similarities.append((i, intersection / union if union > 0 else 0.0))

    similarities.sort(key=lambda x: x[1], reverse=True)
    top_k = min(target_size, len(similarities))
    return {idx for idx, _ in similarities[:top_k]}


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Build forget sets from ob provenance")
    parser.add_argument(
        "--config",
        default="benchmarks/scripts/pipeline_qa/config.yaml",
        help="Path to config.yaml (default: benchmarks/scripts/pipeline_qa/config.yaml)",
    )
    parser.add_argument(
        "--embedding-api",
        default=None,
        help="OpenAI-compatible embedding API URL (e.g. http://localhost:1234/v1)",
    )
    parser.add_argument(
        "--skip-page-prototype",
        action="store_true",
        help="Skip page prototype embedding computation",
    )
    parser.add_argument(
        "--skip-ngram",
        action="store_true",
        help="Skip n-gram computation",
    )
    parser.add_argument(
        "--tokenizer",
        default="gpt2",
        help="Tokenizer name for token-index (default: gpt2)",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ob_data_dir = Path(config["data"]["ob_data_dir"]).resolve()
    data_file = Path(config["data"]["data_file"]).resolve()

    print(f"Data directory: {ob_data_dir}")
    print(f"Data file:      {data_file}")

    print("\nLoading data.jsonl...")
    all_indices, hash_to_idx, texts = load_data(data_file)
    total_lines = len(all_indices)
    print(f"  {total_lines} lines loaded")

    print("Loading manifest records...")
    manifest = load_manifest_index(ob_data_dir)
    print(f"  {len(manifest)} manifest records")

    print("Loading section records...")
    sections = load_section_index(ob_data_dir)
    print(f"  {len(sections)} section records")

    do_page_prototype = not args.skip_page_prototype
    do_ngram = not args.skip_ngram

    all_embeddings: np.ndarray | None = None
    if do_page_prototype:
        try:
            all_embeddings = compute_all_embeddings(texts, args.embedding_api)
            print(f"  Embeddings shape: {all_embeddings.shape}")
        except Exception as e:
            print(f"  ERROR: Failed to compute embeddings: {e}")
            print("  Falling back: skipping page_prototype")
            do_page_prototype = False

    results: dict[str, dict] = {}

    for author_cfg in config["authors"]:
        author_name = author_cfg["name"]
        print(f"\n{'=' * 60}")
        print(f"Author: {author_name}")

        author_recs = query_authors(ob_data_dir, name=author_name)
        if not author_recs:
            print(
                f"  WARNING: Author '{author_name}' not found in .ob/authors/, skipping"
            )
            continue
        author_ids = {rec["id"] for rec in author_recs}
        print(f"  Resolved {len(author_ids)} author ID(s)")

        author_section_hashes = find_author_section_hashes(author_ids, sections)
        print(f"  Contributes to {len(author_section_hashes)} sections")

        line_indices = build_line_level(author_section_hashes, manifest, hash_to_idx)
        print(f"  Line-level forget set: {len(line_indices)} lines")

        random_indices = build_random(total_lines, len(line_indices), seed=config.get("seed", 42))
        print(f"  Random forget set:      {len(random_indices)} lines")

        target_2x = min(2 * len(line_indices), total_lines)
        rng = random.Random(42)
        indices_2x = set(rng.sample(range(total_lines), target_2x))
        print(f"  2x_random forget set:   {len(indices_2x)} lines")

        author_result: dict[str, dict] = {
            "line": {
                "indices": sorted(line_indices),
                "count": len(line_indices),
            },
            "random": {
                "indices": sorted(random_indices),
                "count": len(random_indices),
            },
            "2x_random": {
                "indices": sorted(indices_2x),
                "count": len(indices_2x),
            },
        }

        if do_page_prototype and all_embeddings is not None:
            author_page_map = build_page_map(
                author_section_hashes, sections, manifest, hash_to_idx
            )
            print(f"  Computing page_prototype forget set ({len(author_page_map)} pages)...")
            page_proto_indices = build_page_prototype_sim(
                line_indices, author_page_map, all_embeddings, total_lines, len(line_indices)
            )
            print(f"  page_prototype forget set: {len(page_proto_indices)} lines")
            author_result["page_prototype"] = {
                "indices": sorted(page_proto_indices),
                "count": len(page_proto_indices),
                "pages": len(author_page_map),
            }

        if do_ngram:
            print(f"  Computing ngram forget set...")
            ngram_indices = build_ngram(
                line_indices, texts, total_lines, len(line_indices)
            )
            print(f"  ngram forget set:       {len(ngram_indices)} lines")
            author_result["ngram"] = {
                "indices": sorted(ngram_indices),
                "count": len(ngram_indices),
            }

        tokenizer_name = args.tokenizer
        print(f"  Computing token forget set (tokenizer={tokenizer_name})...")
        token_indices = build_token_level(
            author_section_hashes, ob_data_dir, tokenizer_name
        )
        print(f"  token forget set:       {len(token_indices)} lines")
        author_result["token"] = {
            "indices": sorted(token_indices),
            "count": len(token_indices),
        }

        print(f"  Computing token_mask forget set...")
        mask_data = build_token_mask(
            author_section_hashes, ob_data_dir, tokenizer_name
        )
        print(
            f"  token_mask forget set:  {mask_data['total_author_tokens']} "
            f"author tokens in {len(mask_data['line_indices'])} lines "
            f"(of {mask_data['total_tokens']} total)"
        )
        author_result["token_mask"] = {
            "indices": mask_data["line_indices"],
            "count": len(mask_data["line_indices"]),
            "token_counts": mask_data["token_counts"],
            "total_author_tokens": mask_data["total_author_tokens"],
            "total_tokens": mask_data["total_tokens"],
        }

        results[author_name] = author_result

    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "forget_sets.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"Results saved to {output_file}")
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    types_to_show = ["line", "random", "2x_random"]
    if any("page_prototype" in d for d in results.values()):
        types_to_show.append("page_prototype")
    if any("ngram" in d for d in results.values()):
        types_to_show.append("ngram")
    if any("token" in d for d in results.values()):
        types_to_show.append("token")
        types_to_show.append("token_mask")

    header = f"{'Author':<20}" + "".join(f"{t:>10}" for t in types_to_show)
    separator = f"{'-' * 20}" + "".join(f"{'-' * 10}" for _ in types_to_show)
    print(header)
    print(separator)
    for author_name, data in results.items():
        row = f"{author_name:<20}"
        for t in types_to_show:
            count = data.get(t, {}).get("count", "-")
            row += f"{str(count):>10}"
        print(row)
    print(separator)
    print(
        f"{'TOTAL lines':<20}"
        + f"{total_lines:>10}"
        + "".join(f"{'':>10}" for _ in types_to_show[1:])
    )


if __name__ == "__main__":
    main()
