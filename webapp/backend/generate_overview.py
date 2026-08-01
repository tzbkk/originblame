#!/usr/bin/env python3
"""Pre-compute overview.json and data.jsonl.idx for a dataset.

Usage: python generate_overview.py /path/to/dataset-ob
"""

import gc
import json
import struct
import sys
from pathlib import Path

OB_PYTHON_SRC = str(Path(__file__).parent.parent.parent / "rust-originblame" / "python" / "src")
sys.path.insert(0, OB_PYTHON_SRC)

from ob.authors import list_all_authors
from ob.indexer import read_all_manifest
from ob.storage import LAYER_SECTION, shard_iterate_all
from ob.util import compute_hash


def generate(dataset_dir: Path):
    jsonl_path = dataset_dir / "jsonl" / "data.jsonl"

    authors = list_all_authors(dataset_dir)
    revoked_aids = [a["id"] for a in authors if a.get("revoked")]
    revoked_set = set(revoked_aids)
    id_to_name: dict[str, str] = {a["id"]: a["name"] for a in authors}
    email_to_id: dict[str, str] = {a["email"]: a["id"] for a in authors}
    del authors
    gc.collect()

    sec_path_revoked: dict[str, bool] = {}
    author_sections: dict[str, int] = {}
    contributor_sections: dict[str, int] = {}
    sec_path_record_count: dict[str, int] = {}
    all_contrib: set[str] = set()
    sec_path_to_author_ids: dict[str, list[str]] = {}
    sec_path_to_contributor_ids: dict[str, list[str]] = {}
    section_count = 0
    revoked_section_n = 0

    for s in shard_iterate_all(dataset_dir, LAYER_SECTION):
        section_count += 1
        p = s.get("path", "")
        sec_authors = s.get("authors", [])
        sec_contributors = s.get("contributors", [])
        is_revoked = s.get("revoked", False) or bool(set(sec_authors) & revoked_set)
        sec_path_revoked[p] = is_revoked
        if is_revoked:
            revoked_section_n += 1
        sec_path_to_author_ids[p] = sec_authors
        sec_path_to_contributor_ids[p] = sec_contributors
        for aid in sec_authors:
            author_sections[aid] = author_sections.get(aid, 0) + 1
        for cid in sec_contributors:
            contributor_sections[cid] = contributor_sections.get(cid, 0) + 1
        all_contrib.update(sec_contributors)
        sec_path_record_count[p] = 0

    sec_hash_to_path: dict[str, str] = {}
    for s in shard_iterate_all(dataset_dir, LAYER_SECTION):
        sh = s.get("section_hash", "")
        p = s.get("path", "")
        if sh:
            sec_hash_to_path[sh] = p

    gc.collect()

    manifest = read_all_manifest(dataset_dir)
    manifest_by_hash: dict[str, dict] = {e.get("line_hash", ""): e for e in manifest}
    revoked_record_n = 0
    total_records = 0
    author_record_ids: dict[str, list[int]] = {}
    sec_path_record_indices: dict[str, list[int]] = {}
    line_hash_to_idx: dict[str, int] = {}
    idx_to_line_hash: dict[int, str] = {}

    offsets: list[int] = []

    if jsonl_path.exists():
        with open(jsonl_path, "rb") as f:
            idx = 0
            while True:
                pos = f.tell()
                line_bytes = f.readline()
                if not line_bytes:
                    break
                offsets.append(pos)
                line = line_bytes.decode("utf-8").strip()
                if not line:
                    idx += 1
                    continue
                total_records += 1
                r = json.loads(line)
                lh = compute_hash(r)
                entry = manifest_by_hash.get(lh)
                if not entry:
                    idx += 1
                    continue
                sources = entry.get("sources", [])
                sp = sec_hash_to_path.get(sources[0], "") if sources else ""
                if not sp:
                    idx += 1
                    continue

                line_hash_to_idx[lh] = idx
                idx_to_line_hash[idx] = lh

                if sp in sec_path_record_count:
                    sec_path_record_count[sp] += 1
                sec_path_record_indices.setdefault(sp, []).append(idx)

                if entry.get("revoked", False) or sec_path_revoked.get(sp, False):
                    revoked_record_n += 1

                for aid in sec_path_to_author_ids.get(sp, []):
                    author_record_ids.setdefault(aid, []).append(idx)
                idx += 1

    unique_author_ids = {aid for aids in sec_path_to_author_ids.values() for aid in aids}
    del sec_path_to_contributor_ids, manifest_by_hash, manifest
    gc.collect()

    author_id_to_name = {aid: id_to_name.get(aid, aid[:8]) for aid in unique_author_ids}
    author_email_map = {a["email"]: a["id"] for a in list_all_authors(dataset_dir) if a["id"] in unique_author_ids}
    gc.collect()

    sorted_sections = sorted(author_sections.items(), key=lambda x: x[1], reverse=True)
    top_authors = [
        {"id": aid, "name": author_id_to_name.get(aid, aid[:8]), "sections": cnt}
        for aid, cnt in sorted_sections[:50]
    ]
    author_ranking = [aid for aid, _ in sorted_sections]

    result = {
        "version": 2,
        "records": total_records,
        "sections": section_count,
        "authors": len(unique_author_ids),
        "contributors": len(all_contrib),
        "revoked": {
            "authors": len(revoked_aids),
            "sections": revoked_section_n,
            "records": revoked_record_n,
        },
        "top_authors": top_authors,
        "author_ranking": author_ranking,
        "author_sections": author_sections,
        "author_record_ids": author_record_ids,
        "revoked_author_ids": revoked_aids,
        "sec_path_revoked": sec_path_revoked,
        "sec_path_record_count": sec_path_record_count,
        "sec_path_record_indices": sec_path_record_indices,
        "line_hash_to_idx": line_hash_to_idx,
        "idx_to_line_hash": idx_to_line_hash,
        "id_to_name": author_id_to_name,
        "author_email_to_id": author_email_map,
    }

    overview_path = dataset_dir / "overview.json"
    overview_path.write_text(json.dumps(result, ensure_ascii=False))
    print(f"Wrote {overview_path} ({overview_path.stat().st_size:,} bytes)")

    lazy = {
        "sec_hash_to_path": sec_hash_to_path,
        "sec_path_to_author_ids": sec_path_to_author_ids,
        "contributor_sections": contributor_sections,
    }
    lazy_path = dataset_dir / "overview_lazy.json"
    lazy_path.write_text(json.dumps(lazy, ensure_ascii=False))
    print(f"Wrote {lazy_path} ({lazy_path.stat().st_size:,} bytes)")

    idx_path = dataset_dir / "jsonl" / "data.jsonl.idx"
    with open(idx_path, "wb") as f:
        for off in offsets:
            f.write(struct.pack("<Q", off))
    print(f"Wrote {idx_path} ({len(offsets)} entries, {idx_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_overview.py /path/to/dataset-ob")
        sys.exit(1)
    dataset_dir = Path(sys.argv[1])
    generate(dataset_dir)
