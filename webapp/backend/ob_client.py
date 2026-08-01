"""OB client wrapper — centralises all ob.* calls with consistent error handling.

Memory-efficient data access layer: overview is loaded from a pre-computed
``overview.json`` (no runtime aggregation across 220k+ records), and individual
records are read via the byte-offset index ``data.jsonl.idx`` for O(1) seeks.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PIPELINE_DIR = Path(os.environ.get(
    "OB_PIPELINE_DIR",
    "/home/hxue/Projects/originblame/benchmarks/results/pipeline_v2",
))
QA_DIR = Path(os.environ.get(
    "OB_QA_DIR",
    "/home/hxue/Projects/originblame/benchmarks/results/pipeline_qa",
))
OB_PYTHON_SRC = os.environ.get(
    "OB_PYTHON_SRC",
    "/home/hxue/Projects/rust-originblame/python/src",
)

# ---------------------------------------------------------------------------
# Bootstrap ob package
# ---------------------------------------------------------------------------
import sys

if OB_PYTHON_SRC not in sys.path:
    sys.path.insert(0, OB_PYTHON_SRC)

from ob.authors import (  # noqa: E402
    get_author as _get_author,
    list_all_authors,
    query_authors,
    revoke_author,
    restore_author,
    invalidate_cache as _invalidate_author_cache,
)
from ob.register import (  # noqa: E402
    get_section as _get_section,
    query_sections,
    revoke_section,
    restore_section,
    find_sections_by_path,
    invalidate_section_cache as _invalidate_section_cache,
)
from ob.storage import (  # noqa: E402
    shard_iterate_all,
    LAYER_SECTION,
    LAYER_AUTHORS,
)
from ob.indexer import read_all_manifest, is_revoked  # noqa: E402
from ob.oplog import query_log  # noqa: E402


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------
def discover_datasets() -> list[str]:
    """Return sorted list of available OB dataset names (must have .ob/ dir)."""
    results = []
    for base in [PIPELINE_DIR, QA_DIR]:
        if not base.exists():
            continue
        for d in base.iterdir():
            if d.is_dir() and (d / "jsonl" / "data.jsonl").exists() and (d / ".ob").is_dir():
                results.append(d.name)
    return sorted(set(results))


def dataset_path(dataset: str) -> Path:
    """Return the Path for a dataset directory."""
    for base in [PIPELINE_DIR, QA_DIR]:
        p = base / dataset
        if p.is_dir() and (p / "jsonl" / "data.jsonl").exists():
            return p
    return PIPELINE_DIR / dataset


def dataset_exists(dataset: str) -> bool:
    p = dataset_path(dataset)
    return p.is_dir() and (p / "jsonl" / "data.jsonl").exists()


# ---------------------------------------------------------------------------
# Overview loading (pre-computed overview.json)
# ---------------------------------------------------------------------------
_overview_cache: dict[str, dict] = {}  # dataset → loaded overview dict
_lazy_cache: dict[str, dict] = {}


def load_overview(dataset: str) -> dict:
    """Load pre-computed overview.json. Falls back to runtime computation if missing."""
    if dataset in _overview_cache:
        return _overview_cache[dataset]
    path = dataset_path(dataset) / "overview.json"
    if path.exists():
        data = json.loads(path.read_text())
        _overview_cache[dataset] = data
        return data
    from generate_overview import generate
    generate(dataset_path(dataset))
    data = json.loads(path.read_text())
    _overview_cache[dataset] = data
    return data


def load_overview_lazy(dataset: str) -> dict:
    """Lazy-load overview_lazy.json (sec_hash_to_path, sec_path_to_author_ids, contributor_sections)."""
    if dataset in _lazy_cache:
        return _lazy_cache[dataset]
    path = dataset_path(dataset) / "overview_lazy.json"
    if path.exists():
        data = json.loads(path.read_text())
        _lazy_cache[dataset] = data
        return data
    return {}


def invalidate_overview_cache(dataset: str) -> None:
    """Clear in-memory cache and delete on-disk overview.json so it regenerates."""
    _overview_cache.pop(dataset, None)
    _lazy_cache.pop(dataset, None)
    path = dataset_path(dataset) / "overview.json"
    if path.exists():
        path.unlink()
    lazy_path = dataset_path(dataset) / "overview_lazy.json"
    if lazy_path.exists():
        lazy_path.unlink()


# ---------------------------------------------------------------------------
# Byte-offset record loading (data.jsonl.idx)
# ---------------------------------------------------------------------------
_offset_cache: dict[str, list[int]] = {}  # dataset → byte offsets


def _load_offsets(dataset: str) -> list[int] | None:
    """Load byte-offset index for data.jsonl."""
    if dataset in _offset_cache:
        return _offset_cache[dataset]
    idx_path = dataset_path(dataset) / "jsonl" / "data.jsonl.idx"
    if not idx_path.exists():
        return None
    data = idx_path.read_bytes()
    offsets = list(struct.unpack(f'<{len(data)//8}Q', data))
    _offset_cache[dataset] = offsets
    return offsets


def load_records_by_indices(dataset: str, indices: list[int]) -> list[dict]:
    """Load specific records by line index using byte-offset O(1) seek."""
    if not indices:
        return []
    offsets = _load_offsets(dataset)
    jsonl_path = dataset_path(dataset) / "jsonl" / "data.jsonl"
    if not jsonl_path.exists():
        return []
    sorted_indices = sorted(set(indices))
    fetched: dict[int, dict] = {}
    with open(jsonl_path, "rb") as f:
        for idx in sorted_indices:
            if offsets:
                if idx >= len(offsets):
                    continue
                f.seek(offsets[idx])
            else:
                # No offset index — fall back to sequential scan
                f.seek(0)
                for _ in range(idx):
                    f.readline()
            line = f.readline().strip()
            if line:
                fetched[idx] = json.loads(line)
    return [fetched[i] for i in indices if i in fetched]


def load_records_range(dataset: str, start: int, count: int) -> list[dict]:
    """Load a slice of records. Uses byte-offset index if available."""
    jsonl_path = dataset_path(dataset) / "jsonl" / "data.jsonl"
    if not jsonl_path.exists():
        return []

    offsets = _load_offsets(dataset)
    if offsets and start < len(offsets):
        # Fast path: byte-offset seek
        records = []
        end = min(start + count, len(offsets))
        with open(jsonl_path, "rb") as f:
            for i in range(start, end):
                f.seek(offsets[i])
                line = f.readline().strip()
                if line:
                    records.append(json.loads(line))
        return records

    # Slow fallback: sequential read
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            line = line.strip()
            if line:
                records.append(json.loads(line))
            if len(records) >= count:
                break
    return records


# ---------------------------------------------------------------------------
# Data loaders (records from JSONL)
# ---------------------------------------------------------------------------
def load_records(dataset: str) -> list[dict]:
    """Load all records from data.jsonl."""
    path = dataset_path(dataset) / "jsonl" / "data.jsonl"
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def count_records(dataset: str) -> int:
    """Count records. Uses byte-offset index length when available."""
    offsets = _load_offsets(dataset)
    if offsets is not None:
        return len(offsets)
    path = dataset_path(dataset) / "jsonl" / "data.jsonl"
    if not path.exists():
        return 0
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Author helpers
# ---------------------------------------------------------------------------
def load_authors(dataset: str) -> list[dict]:
    return list_all_authors(dataset_path(dataset))


def load_authors_by_ids(dataset: str, author_ids: list[str]) -> list[dict]:
    """Load only specific authors by their IDs — much faster than loading all."""
    ob_dir = dataset_path(dataset)
    results = []
    for aid in author_ids:
        a = _get_author(ob_dir, aid)
        if a:
            results.append(a)
    return results


def author_contribution_map(sections: list[dict]) -> Counter:
    """Return Counter: author_id -> number of sections they appear in."""
    c: Counter = Counter()
    for s in sections:
        for aid in s.get("authors", []):
            c[aid] += 1
    return c


# ---------------------------------------------------------------------------
# Section helpers
# ---------------------------------------------------------------------------
def load_sections(dataset: str) -> list[dict]:
    return list(shard_iterate_all(dataset_path(dataset), LAYER_SECTION))


def load_sections_lazy(dataset: str, start: int, count: int) -> list[dict]:
    """Load a slice of sections using the shard iterator."""
    ob_dir = dataset_path(dataset)
    items = []
    for i, s in enumerate(shard_iterate_all(ob_dir, LAYER_SECTION)):
        if i < start:
            continue
        items.append(s)
        if len(items) >= count:
            break
    return items


def sec_path_revoked_map(
    sections: list[dict], revoked_author_ids: set[str]
) -> dict[str, bool]:
    """Map sec_path -> revoked status (lazy cascade)."""
    m = {}
    for s in sections:
        p = s.get("path", "")
        m[p] = s.get("revoked", False) or bool(
            set(s.get("authors", [])) & revoked_author_ids
        )
    return m


def record_sec_path(record: dict) -> str:
    """Build the sec_path join key for a record.

    Matches section path format: ``raw/<title>`` when heading is empty,
    ``raw/<title>#<heading>`` otherwise.  A trailing ``#`` would never
    match a section path and breaks cross-dataset compatibility.
    """
    title = record.get("title", "")
    heading = record.get("heading", "")
    if heading:
        return f"raw/{title}#{heading}"
    return f"raw/{title}"


# ---------------------------------------------------------------------------
# Revocation status
# ---------------------------------------------------------------------------
def revoked_author_ids(authors: list[dict]) -> set[str]:
    return {a["id"] for a in authors if a.get("revoked")}


def is_record_revoked(
    record: dict, sp_revoked: dict[str, bool]
) -> bool:
    return sp_revoked.get(record_sec_path(record), False)


# ---------------------------------------------------------------------------
# Manifest helpers (still used by record detail endpoint fallback)
# ---------------------------------------------------------------------------
def load_manifest_indexed(dataset: str) -> list[dict]:
    """Load manifest entries. Records and manifest are written in same order."""
    return read_all_manifest(dataset_path(dataset))


def get_manifest_entry(dataset: str, idx: int) -> dict | None:
    """Read a single manifest entry by streaming to the target index."""
    manifest = read_all_manifest(dataset_path(dataset))
    if 0 <= idx < len(manifest):
        return manifest[idx]
    return None


# ---------------------------------------------------------------------------
# Contributor lookup helper (replaces contrib_record_ids cache field)
# ---------------------------------------------------------------------------
def _get_contributor_record_indices(
    dataset: str, author_id: str, cached: dict
) -> list[int]:
    """Get record indices where author_id appears as contributor.

    Scans section shards for sections listing ``author_id`` in their
    ``contributors`` array, then expands each matching section's record
    indices from ``sec_path_record_indices`` in the overview.
    """
    ob_dir = dataset_path(dataset)
    sec_path_record_indices = cached.get("sec_path_record_indices", {})
    indices: list[int] = []
    for s in shard_iterate_all(ob_dir, LAYER_SECTION):
        if author_id in s.get("contributors", []):
            sp = s.get("path", "")
            indices.extend(sec_path_record_indices.get(sp, []))
    return sorted(indices)


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------
def load_audit_log(dataset: str) -> list[dict]:
    """Read .ob/log, parse entries, reverse chronological."""
    log_path = dataset_path(dataset) / ".ob" / "log"
    if not log_path.exists():
        return []
    entries = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # JSON format (new)
            if line.startswith("{"):
                try:
                    entries.append(json.loads(line))
                    continue
                except json.JSONDecodeError:
                    pass
            # Plain text format: "YYYYMMDD-HH:MM:SS op args..."
            parts = line.split(None, 1)
            ts = parts[0] if parts else ""
            rest = parts[1] if len(parts) > 1 else ""
            op_parts = rest.split(None, 1)
            op = op_parts[0] if op_parts else rest
            detail_str = op_parts[1] if len(op_parts) > 1 else ""
            entries.append({"ts": ts, "op": op, "detail": detail_str, "cmd": line})
    entries.reverse()
    return entries


# ---------------------------------------------------------------------------
# Backup / restore (demo reset)
# ---------------------------------------------------------------------------
def backup_dir(dataset: str) -> Path:
    return dataset_path(dataset) / ".ob-backup-demo"


def backup_ob(dataset: str) -> None:
    """Copy .ob/ to .ob-backup-demo/ on first revoke (idempotent)."""
    ob_src = dataset_path(dataset) / ".ob"
    bkp = backup_dir(dataset)
    if bkp.exists():
        return
    shutil.copytree(ob_src, bkp)


def restore_ob(dataset: str) -> None:
    ob_dst = dataset_path(dataset) / ".ob"
    bkp = backup_dir(dataset)
    if bkp.exists():
        shutil.rmtree(ob_dst)
        shutil.copytree(bkp, ob_dst)
    _invalidate_author_cache()
    _invalidate_section_cache()
    invalidate_overview_cache(dataset)


# ---------------------------------------------------------------------------
# Revocation actions
# ---------------------------------------------------------------------------
def do_revoke_author(dataset: str, *, email: str | None = None, author_id: str | None = None) -> int:
    backup_ob(dataset)
    count = revoke_author(dataset_path(dataset), email=email, author_id=author_id)
    _invalidate_author_cache()
    _invalidate_section_cache()
    invalidate_overview_cache(dataset)
    return count


def do_revoke_section(dataset: str, *, section_hash: str | None = None, path: str | None = None) -> int:
    backup_ob(dataset)
    count = revoke_section(dataset_path(dataset), section_hash=section_hash, path=path)
    _invalidate_author_cache()
    _invalidate_section_cache()
    invalidate_overview_cache(dataset)
    return count


def do_restore_author(dataset: str, *, email: str | None = None, author_id: str | None = None) -> int:
    count = restore_author(dataset_path(dataset), email=email, author_id=author_id)
    _invalidate_author_cache()
    _invalidate_section_cache()
    invalidate_overview_cache(dataset)
    return count


def do_restore_section(dataset: str, *, section_hash: str | None = None, path: str | None = None) -> int:
    count = restore_section(dataset_path(dataset), section_hash=section_hash, path=path)
    _invalidate_author_cache()
    _invalidate_section_cache()
    invalidate_overview_cache(dataset)
    return count


# ---------------------------------------------------------------------------
# Record enrichment — uses overview data (no per-record precomputed map)
# ---------------------------------------------------------------------------
def enrich_record(r: dict, idx: int, cached: dict, dataset: str = "") -> dict:
    """Add title/heading/authors/sec_path from overview data when missing from JSONL.

    For datasets where JSONL records already have ``title``, ``heading``,
    ``authors_json``, the original fields are kept.  For datasets without
    those fields (e.g. ChatML QA data), the metadata is resolved from
    overview's ``sec_path_record_indices``.

    The reverse map ``_idx_to_sec_path`` is built lazily on first call and
    stored back in ``cached`` so it is reused across enrichment calls within
    the same request.
    """
    if not r.get("line_hash"):
        r["line_hash"] = cached.get("idx_to_line_hash", {}).get(str(idx), "")

    if r.get("title"):
        return r

    idx_map = cached.get("_idx_to_sec_path")
    if idx_map is None:
        idx_map = {}
        for sp, idxs in cached.get("sec_path_record_indices", {}).items():
            for i in idxs:
                idx_map[i] = sp
        cached["_idx_to_sec_path"] = idx_map

    sp = idx_map.get(idx, "")
    if not sp:
        return r

    r = dict(r)
    path_body = sp.lstrip("raw/").lstrip("qa/")
    parts = path_body.split("#", 1)
    r["title"] = parts[0] if parts else ""
    r["heading"] = parts[1] if len(parts) > 1 else ""
    r["sec_path"] = sp

    sec_authors_map = cached.get("sec_path_to_author_ids")
    if sec_authors_map is None:
        sec_authors_map = load_overview_lazy(dataset).get("sec_path_to_author_ids", {})
    sec_authors = sec_authors_map.get(sp, [])
    if sec_authors and not r.get("authors_json"):
        id_to_name = cached.get("id_to_name", {})
        r["authors_json"] = json.dumps([
            {"id": aid, "name": id_to_name.get(aid, ""), "email": ""}
            for aid in sec_authors
        ])

    if not r.get("line_hash"):
        idx_to_lh = cached.get("idx_to_line_hash", {})
        r["line_hash"] = idx_to_lh.get(str(idx), "")

    return r


# ---------------------------------------------------------------------------
# Filtered streaming (used by /records endpoint when filters are applied)
# ---------------------------------------------------------------------------
def stream_records_filtered(
    dataset: str,
    filter_fn,
    page: int = 1,
    limit: int = 20,
) -> tuple[list[dict], int]:
    """Stream JSONL, apply *filter_fn(idx, r)->(bool, r_enriched|None)* per record.

    If the filter returns a truthy value and a dict, that dict is stored
    instead of the raw record (allows enrichment during filtering).
    """
    jsonl_path = dataset_path(dataset) / "jsonl" / "data.jsonl"
    if not jsonl_path.exists():
        return [], 0
    start = (page - 1) * limit
    end = start + limit
    total = 0
    page_items: list[dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            result = filter_fn(r, idx)
            if isinstance(result, tuple):
                keep, enriched = result
            else:
                keep, enriched = result, None
            if keep:
                if start <= total < end:
                    page_items.append(enriched if enriched is not None else r)
                total += 1
    return page_items, total
