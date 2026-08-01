"""FastAPI backend for OriginBlame webapp demo.

Wraps the ``ob`` Python package to expose REST endpoints consumed by the
React SPA frontend.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Bootstrap: ensure ob package is importable
# ---------------------------------------------------------------------------
_OB_PYTHON_SRC = os.environ.get(
    "OB_PYTHON_SRC",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "rust-originblame", "python", "src")),
)
if _OB_PYTHON_SRC not in sys.path:
    sys.path.insert(0, _OB_PYTHON_SRC)

from ob_client import (
    load_overview, invalidate_overview_cache, load_overview_lazy,
    load_authors_by_ids, load_records_range, count_records,
    get_manifest_entry, stream_records_filtered, load_records_by_indices,
    _get_section, _get_contributor_record_indices,
    record_sec_path, discover_datasets, dataset_path, enrich_record,
)


from ob.authors import list_all_authors, revoke_author, restore_author, invalidate_cache as _invalidate_author_cache  # noqa: E402
from ob.oplog import query_log  # noqa: E402
from ob.register import (  # noqa: E402
    revoke_section,
    restore_section,
)
from ob.storage import LAYER_SECTION, shard_iterate_all  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PIPELINE_DIR = os.environ.get(
    "OB_PIPELINE_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "benchmarks", "results", "pipeline_v2")),
)
DEFAULT_DATASET = "qa_chatml"

app = FastAPI(title="OriginBlame API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _discover_datasets() -> list[str]:
    return discover_datasets()


def _ob_dir(dataset: str) -> str:
    return str(dataset_path(dataset))


def _record_preview(r: dict) -> str:
    if r.get("messages"):
        msgs = r["messages"]
        for m in msgs:
            if m.get("role") == "user":
                return m.get("content", "")[:120].replace("\n", " ") + "..."
        return str(r)[:120] + "..."
    return r.get("text", "")[:120].replace("\n", " ") + "..."


def _record_authors_display(r: dict) -> str:
    if r.get("authors_json"):
        names = [a.get("name", "") for a in json.loads(r.get("authors_json", "[]"))[:3]]
        return ", ".join(n for n in names if n)
    return ""


def _backup_dir(dataset: str) -> Path:
    return Path(_ob_dir(dataset)) / ".ob-backup-demo"


def _backup_ob(dataset: str):
    ob_src = Path(_ob_dir(dataset)) / ".ob"
    bkp = _backup_dir(dataset)
    if bkp.exists():
        return
    shutil.copytree(ob_src, bkp)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------
class OverviewResponse(BaseModel):
    records: int
    sections: int
    authors: int
    contributors: int
    revoked: dict
    top_authors: list[dict]
    author_ranking: list[str] = []


class PaginatedResponse(BaseModel):
    items: list
    total: int
    page: int
    limit: int


class AuthorDetailResponse(BaseModel):
    author: dict
    metrics: dict
    author_records: list[dict]
    author_page: int = 1
    author_limit: int = 20
    author_total: int = 0
    contributor_records: list[dict]
    contributor_page: int = 1
    contributor_limit: int = 20
    contributor_total: int = 0


class RecordDetailResponse(BaseModel):
    record: dict
    section: Optional[dict] = None


class ErasureImpactResponse(BaseModel):
    target_name: str
    is_already_revoked: bool
    affected_sections: int
    affected_records: int
    affected_contrib_sections: int = 0
    affected_contrib_records: int = 0
    total_sections: int
    total_records: int
    revoke_desc: str
    comparison: Optional[dict] = None


class RevokeAuthorRequest(BaseModel):
    email: str


class RevokeSectionRequest(BaseModel):
    section_hash: str


class RevokeRecordRequest(BaseModel):
    path: str


class RestoreAuthorRequest(BaseModel):
    author_id: str


class RestoreSectionRequest(BaseModel):
    section_hash: str


class MessageResponse(BaseModel):
    message: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/datasets")
def get_datasets():
    return {"datasets": _discover_datasets(), "default": DEFAULT_DATASET}


@app.get("/api/{dataset}/overview", response_model=OverviewResponse)
def get_overview(dataset: str):
    cached = load_overview(dataset)
    return OverviewResponse(
        records=cached["records"],
        sections=cached["sections"],
        authors=cached["authors"],
        contributors=cached["contributors"],
        revoked=cached["revoked"],
        top_authors=cached["top_authors"],
        author_ranking=cached.get("author_ranking", []),
    )


@app.get("/api/{dataset}/authors")
def get_authors(
    dataset: str,
    search: Optional[str] = None,
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=200),
):
    cached = load_overview(dataset)

    ranking = cached.get("author_ranking", [])
    total = len(ranking)

    if search:
        ob_dir = Path(_ob_dir(dataset))
        all_authors = list_all_authors(ob_dir)
        filtered = sorted(
            [a for a in all_authors if search.lower() in a.get("name", "").lower()],
            key=lambda a: ranking.index(a["id"]) if a["id"] in ranking else len(ranking),
        )
        total = len(filtered)
        start = (page - 1) * limit
        page_authors = filtered[start:start + limit]
    else:
        start = (page - 1) * limit
        page_ids = ranking[start:start + limit]
        page_authors = load_authors_by_ids(dataset, page_ids)

    sections_total = cached.get("sections", 1)
    author_sections = cached.get("author_sections", {})
    items = []
    for a in page_authors:
        aid = a["id"]
        sec_count = author_sections.get(aid, 0)
        items.append({
            "id": aid,
            "name": a.get("name", ""),
            "email": a.get("email", ""),
            "sections": sec_count,
            "contribution_pct": f"{sec_count / max(sections_total, 1) * 100:.1f}%",
            "revoked": a.get("revoked", False),
        })

    return PaginatedResponse(items=items, total=total, page=page, limit=limit)


@app.get("/api/{dataset}/authors/{author_id}", response_model=AuthorDetailResponse)
def get_author_detail(
    dataset: str,
    author_id: str,
    author_page: int = Query(1, ge=1),
    author_limit: int = Query(20, ge=1, le=100),
    contributor_page: int = Query(1, ge=1),
    contributor_limit: int = Query(20, ge=1, le=100),
):
    cached = load_overview(dataset)

    selected = next((a for a in load_authors_by_ids(dataset, [author_id]) if a["id"] == author_id), None)
    if not selected:
        raise HTTPException(404, "Author not found")

    aid = selected["id"]
    sec_count = cached.get("author_sections", {}).get(aid, 0)
    sections_as_contrib = load_overview_lazy(dataset).get("contributor_sections", {}).get(aid, 0)

    author_record_idxs = cached.get("author_record_ids", {}).get(aid, [])
    contrib_record_idxs = _get_contributor_record_indices(dataset, aid, cached)

    a_start = (author_page - 1) * author_limit
    a_end = a_start + author_limit
    c_start = (contributor_page - 1) * contributor_limit
    c_end = c_start + contributor_limit

    author_records = load_records_by_indices(dataset, author_record_idxs[a_start:a_end])
    contributor_records = load_records_by_indices(dataset, contrib_record_idxs[c_start:c_end])

    author_records = [enrich_record(r, ri, cached, dataset) for ri, r in zip(author_record_idxs[a_start:a_end], author_records)]
    contributor_records = [enrich_record(r, ri, cached, dataset) for ri, r in zip(contrib_record_idxs[c_start:c_end], contributor_records)]

    author_sec_paths = {r.get("sec_path", record_sec_path(r)) for r in author_records}

    result = AuthorDetailResponse(
        author={
            "id": aid,
            "name": selected.get("name", ""),
            "email": selected.get("email", ""),
            "revoked": selected.get("revoked", False),
        },
        metrics={
            "sections_as_author": sec_count,
            "records_as_author": len(author_record_idxs),
            "records_as_contributor": len(contrib_record_idxs),
            "sections_as_contributor": sections_as_contrib,
        },
        author_records=[
            {
                "idx": ri,
                "title": r.get("title", ""),
                "heading": r.get("heading", ""),
                "sec_path": r.get("sec_path", record_sec_path(r)),
                "year": r.get("year", ""),
                "license": r.get("license", ""),
                "preview": _record_preview(r),
                "text": r.get("text", ""),
                "authors_json": r.get("authors_json", "[]"),
                "line_hash": r.get("line_hash", ""),
            }
            for ri, r in zip(author_record_idxs[a_start:a_end], author_records)
        ],
        author_page=author_page,
        author_limit=author_limit,
        author_total=len(author_record_idxs),
        contributor_records=[
            {
                "idx": ri,
                "title": r.get("title", ""),
                "heading": r.get("heading", ""),
                "sec_path": r.get("sec_path", record_sec_path(r)),
                "year": r.get("year", ""),
                "license": r.get("license", ""),
                "preview": _record_preview(r),
                "text": r.get("text", ""),
                "authors_json": r.get("authors_json", "[]"),
                "line_hash": r.get("line_hash", ""),
                "is_author": r.get("sec_path", record_sec_path(r)) in author_sec_paths,
            }
            for ri, r in zip(contrib_record_idxs[c_start:c_end], contributor_records)
        ],
        contributor_page=contributor_page,
        contributor_limit=contributor_limit,
        contributor_total=len(contrib_record_idxs),
    )
    return result


@app.get("/api/{dataset}/records")
def get_records(
    dataset: str,
    search: Optional[str] = None,
    author: Optional[str] = None,
    status: Optional[str] = None,
    year: Optional[str] = None,
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=200),
):
    has_filter = search or author or status or year

    if not has_filter:
        total = count_records(dataset)
        start = (page - 1) * limit
        page_records = load_records_range(dataset, start, limit)
        cached = load_overview(dataset)
        items = []
        for i, r in enumerate(page_records):
            r = enrich_record(r, start + i, cached, dataset)
            items.append({
                "title": r.get("title", ""),
                "heading": r.get("heading", ""),
                "preview": _record_preview(r),
                "authors": _record_authors_display(r),
                "year": str(r.get("year", "")),
                "status": "active",
                "sec_path": r.get("sec_path", record_sec_path(r)),
                "line_hash": r.get("line_hash", ""),
            })
        return {"items": items, "total": total, "page": page, "limit": limit}

    # Filtered path: use cache + stream_records_filtered
    cached = load_overview(dataset)

    sec_path_revoked = cached.get("sec_path_revoked", {})
    authors_param = [a.strip() for a in author.split(",")] if author else []
    years_param = [y.strip() for y in year.split(",")] if year else []
    q = search.lower() if search else ""

    def _filter(r: dict, idx: int):
        r = enrich_record(r, idx, cached, dataset)
        title = r.get("title", "")
        text = r.get("text", "")
        if q and q not in title.lower() and q not in text.lower():
            return False, None
        if authors_param:
            r_authors = json.loads(r.get("authors_json", "[]"))
            if not any(a.get("name", "") in authors_param for a in r_authors):
                return False, None
        sp = record_sec_path(r)
        if status == "active" and sec_path_revoked.get(sp, False):
            return False, None
        if status == "revoked" and not sec_path_revoked.get(sp, False):
            return False, None
        if years_param and str(r.get("year", "")) not in years_param:
            return False, None
        return True, r

    page_records, total = stream_records_filtered(dataset, _filter, page, limit)

    items = []
    for i, r in enumerate(page_records):
        sp = record_sec_path(r)
        is_revoked = sec_path_revoked.get(sp, False)
        items.append({
            "title": r.get("title", ""),
            "heading": r.get("heading", ""),
            "preview": _record_preview(r),
            "authors": _record_authors_display(r),
            "year": str(r.get("year", "")),
            "status": "revoked" if is_revoked else "active",
            "sec_path": sp,
            "line_hash": r.get("line_hash", ""),
        })

    return {"items": items, "total": total, "page": page, "limit": limit}
@app.get("/api/{dataset}/records/detail")
def get_record_detail(dataset: str, sec_path: str = "", hash: str = "", idx: int = Query(-1)):
    ob_dir = Path(_ob_dir(dataset))
    cached = load_overview(dataset)

    # Resolve idx from line_hash if provided
    if hash and idx < 0:
        hash_to_idx = cached.get("line_hash_to_idx", {})
        resolved = hash_to_idx.get(hash)
        if resolved is not None:
            idx = int(resolved)

    if idx >= 0:
        recs = load_records_range(dataset, idx, 1)
    else:
        recs = []
        jsonl_path = ob_dir / "jsonl" / "data.jsonl"
        if jsonl_path.exists():
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if record_sec_path(rec) == sec_path or (not rec.get("title") and sec_path):
                        idx = i
                        recs = [rec]
                        break
    if not recs:
        raise HTTPException(404, "Record not found")
    r = recs[0]

    r = enrich_record(r, idx, cached, dataset)
    r_authors = json.loads(r.get("authors_json", "[]"))

    line_hash = hash or r.get("line_hash", "")
    revoked = False
    section_hashes: list[str] = []
    sp = r.get("sec_path", "")
    if sp:
        sec_hash_to_path = load_overview_lazy(dataset).get("sec_hash_to_path", {})
        path_to_hash = {v: k for k, v in sec_hash_to_path.items()}
        sh = path_to_hash.get(sp, "")
        if sh:
            section_hashes = [sh]
    if idx >= 0:
        mentry = get_manifest_entry(dataset, idx)
        if mentry:
            if not line_hash:
                line_hash = mentry.get("line_hash", "")
            revoked = mentry.get("revoked", False)
            if not section_hashes:
                section_hashes = mentry.get("sources", [])

    id_to_name = cached.get("id_to_name", {})
    revoked_author_ids = set(cached.get("revoked_author_ids", []))
    author_email_to_id = cached.get("author_email_to_id", {})

    sections = []
    for sh in section_hashes:
        sec = _get_section(ob_dir, sh)
        if sec:
            sec_revoked = sec.get("revoked", False) or bool(set(sec.get("authors", [])) & revoked_author_ids)
            contrib_names = [id_to_name.get(cid, cid[:8]) for cid in sec.get("contributors", [])]
            sections.append({
                "section_hash": sec.get("section_hash", ""),
                "path": sec.get("path", ""),
                "license": sec.get("license", ""),
                "year": sec.get("year", ""),
                "revoked": sec_revoked,
                "contributors": contrib_names,
            })

    if not revoked and sections:
        revoked = any(s["revoked"] for s in sections)

    if not r_authors and sections:
        sec_author_ids = []
        for sec in sections:
            sec_rec = _get_section(ob_dir, sec["section_hash"])
            if sec_rec:
                sec_author_ids.extend(sec_rec.get("authors", []))
        r_authors = [{"id": aid, "name": id_to_name.get(aid, aid[:8]), "email": ""} for aid in dict.fromkeys(sec_author_ids)]

    display_text = r.get("text", "")
    if not display_text and r.get("messages"):
        display_text = json.dumps(r["messages"], ensure_ascii=False, indent=2)

    return {
        "title": r.get("title", ""),
        "heading": r.get("heading", ""),
        "year": r.get("year", "") or (sections[0]["year"] if sections else ""),
        "license": r.get("license", "") or (sections[0]["license"] if sections else ""),
        "revoked": revoked,
        "authors": [
            {
                "id": a.get("id", "") or author_email_to_id.get(a.get("email", ""), ""),
                "name": a.get("name", ""),
                "email": a.get("email", ""),
            }
            for a in r_authors
        ],
        "text": display_text,
        "line_hash": line_hash,
        "section_hashes": section_hashes,
        "sections": sections,
    }


@app.get("/api/{dataset}/sections")
def get_sections(
    dataset: str,
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=500),
    revoked_only: bool = False,
):
    cached = load_overview(dataset)

    sec_path_revoked = cached.get("sec_path_revoked", {})
    sec_path_record_count = cached.get("sec_path_record_count", {})
    id_to_name = cached.get("id_to_name", {})

    ob_dir = Path(_ob_dir(dataset))
    all_sections_meta: list[dict] = []
    for s in shard_iterate_all(ob_dir, LAYER_SECTION):
        p = s.get("path", "")
        is_revoked = sec_path_revoked.get(p, False)
        if revoked_only and not is_revoked:
            continue
        all_sections_meta.append({
            "section_hash": s.get("section_hash", ""),
            "path": p,
            "authors": s.get("authors", []),
            "license": s.get("license", ""),
            "year": s.get("year", ""),
            "revoked": is_revoked,
        })

    total = len(all_sections_meta)
    start = (page - 1) * limit
    page_items = all_sections_meta[start:start + limit]

    items = []
    for s in page_items:
        p = s["path"].replace("raw/", "")
        parts = p.split("#", 1)
        sec_author_ids = s["authors"]
        sec_author_names = ", ".join(id_to_name.get(a, a[:8]) for a in sec_author_ids[:3])
        rec_count = sec_path_record_count.get(s["path"], 0)
        items.append({
            "section_hash": s["section_hash"],
            "path": s["path"],
            "title": parts[0] if parts else p,
            "heading": parts[1] if len(parts) > 1 else "",
            "authors": sec_author_names,
            "license": s["license"],
            "year": s["year"],
            "revoked": s["revoked"],
            "record_count": rec_count,
        })

    return PaginatedResponse(items=items, total=total, page=page, limit=limit)


@app.get("/api/{dataset}/erasure/impact", response_model=ErasureImpactResponse)
def get_erasure_impact(dataset: str, revoke_type: str, target: str):
    cached = load_overview(dataset)

    sec_path_revoked = cached.get("sec_path_revoked", {})
    sec_path_record_count = cached.get("sec_path_record_count", {})
    id_to_name = cached.get("id_to_name", {})
    revoked_aids = set(cached.get("revoked_author_ids", []))
    author_sections_map = cached.get("author_sections", {})
    total_sections = cached.get("sections", 0)
    total_records = cached.get("records", 0)

    if revoke_type == "author":
        ob_dir = Path(_ob_dir(dataset))
        all_authors = list_all_authors(ob_dir)
        aid_match = next(
            (a["id"] for a in all_authors if a.get("email", "") == target or a.get("name", "") == target),
            None,
        )
        if aid_match:
            selected = load_authors_by_ids(dataset, [aid_match])
        else:
            selected = []
        if not selected:
            raise HTTPException(404, "Author not found")
        aid = selected[0]["id"]
        is_already_revoked = selected[0].get("revoked", False)

        author_sec_paths: set[str] = set()
        contributor_sec_paths: set[str] = set()
        ob_dir = Path(_ob_dir(dataset))
        for s in shard_iterate_all(ob_dir, LAYER_SECTION):
            p = s.get("path", "")
            if aid in s.get("authors", []):
                author_sec_paths.add(p)
            if aid in s.get("contributors", []):
                contributor_sec_paths.add(p)

        record_level = len(author_sec_paths)
        contributor_level = len(contributor_sec_paths)
        affected_records = sum(sec_path_record_count.get(p, 0) for p in author_sec_paths)
        affected_contrib_records = sum(sec_path_record_count.get(p, 0) for p in contributor_sec_paths)
        file_level = total_sections
        factor = file_level / max(record_level, 1)
        return ErasureImpactResponse(
            target_name=selected[0].get("name", target),
            is_already_revoked=is_already_revoked,
            affected_sections=record_level,
            affected_records=affected_records,
            affected_contrib_sections=contributor_level,
            affected_contrib_records=affected_contrib_records,
            total_sections=total_sections,
            total_records=total_records,
            revoke_desc=f"{record_level} sections as author, {contributor_level} as contributor",
            comparison={
                "file_level": file_level,
                "contributor_level": contributor_level,
                "record_level": record_level,
                "factor": round(factor, 1),
            },
        )
    elif revoke_type == "section":
        ob_dir = Path(_ob_dir(dataset))
        sec = None
        for s in shard_iterate_all(ob_dir, LAYER_SECTION):
            if s.get("section_hash", "").startswith(target):
                sec = s
                break
        if not sec:
            raise HTTPException(404, "Section not found")
        target_sec_path = sec.get("path", "")
        author_records = sec_path_record_count.get(target_sec_path, 0)
        sec_author_ids = sec.get("authors", [])
        target_name = id_to_name.get(sec_author_ids[0], sec_author_ids[0][:8]) if sec_author_ids else ""
        return ErasureImpactResponse(
            target_name=target_name,
            is_already_revoked=sec.get("revoked", False) or sec_path_revoked.get(target_sec_path, False),
            affected_sections=1,
            affected_records=author_records,
            total_sections=total_sections,
            total_records=total_records,
            revoke_desc=f"1 section, {author_records} record(s)",
        )
    elif revoke_type == "record":
        rec_sec_path = target
        affected_records = sec_path_record_count.get(rec_sec_path, 0)
        # Get record to find author name
        target_name = ""
        jsonl_path = Path(_ob_dir(dataset)) / "jsonl" / "data.jsonl"
        if jsonl_path.exists():
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if record_sec_path(rec) == rec_sec_path:
                        r_authors = json.loads(rec.get("authors_json", "[]"))
                        target_name = r_authors[0].get("name", "") if r_authors else ""
                        break
        is_rev = sec_path_revoked.get(rec_sec_path, False)
        return ErasureImpactResponse(
            target_name=target_name,
            is_already_revoked=is_rev,
            affected_sections=1,
            affected_records=1,
            total_sections=total_sections,
            total_records=total_records,
            revoke_desc="1 record",
        )
    else:
        raise HTTPException(400, "Invalid revoke_type. Use: author, section, record")


@app.post("/api/{dataset}/revoke/author", response_model=MessageResponse)
def api_revoke_author(dataset: str, req: RevokeAuthorRequest):
    ob_dir = Path(_ob_dir(dataset))
    _backup_ob(dataset)
    try:
        from ob.authors import invalidate_cache
        revoke_author(ob_dir, email=req.email)
        invalidate_cache(ob_dir)
        invalidate_overview_cache(dataset)
        return MessageResponse(message=f"Author {req.email} revoked")
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/{dataset}/revoke/section", response_model=MessageResponse)
def api_revoke_section(dataset: str, req: RevokeSectionRequest):
    ob_dir = Path(_ob_dir(dataset))
    _backup_ob(dataset)
    try:
        revoke_section(ob_dir, section_hash=req.section_hash)
        invalidate_overview_cache(dataset)
        return MessageResponse(message=f"Section {req.section_hash[:16]}... revoked")
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/{dataset}/revoke/record", response_model=MessageResponse)
def api_revoke_record(dataset: str, req: RevokeRecordRequest):
    ob_dir = Path(_ob_dir(dataset))
    _backup_ob(dataset)
    try:
        revoke_section(ob_dir, path=req.path)
        invalidate_overview_cache(dataset)
        return MessageResponse(message=f"Record {req.path} revoked")
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/{dataset}/revoked")
def get_revoked(dataset: str):
    cached = load_overview(dataset)

    revoked_aids = set(cached.get("revoked_author_ids", []))
    id_to_name = cached.get("id_to_name", {})
    sec_path_record_count = cached.get("sec_path_record_count", {})

    revoked_authors_raw = load_authors_by_ids(dataset, list(revoked_aids))
    revoked_authors = [
        {
            "id": a["id"],
            "name": a.get("name", ""),
            "email": a.get("email", ""),
            "affected_sections": sum(1 for s in _stream_sections_for_author(dataset, a["id"], "authors")),
        }
        for a in revoked_authors_raw
    ]

    ob_dir = Path(_ob_dir(dataset))
    revoked_sections = []
    cascade_sections = []
    for s in shard_iterate_all(ob_dir, LAYER_SECTION):
        p = s.get("path", "")
        is_direct = s.get("revoked", False)
        is_cascade = not is_direct and bool(set(s.get("authors", [])) & revoked_aids)
        if is_direct:
            path_display = p.replace("raw/", "")
            parts = path_display.split("#", 1)
            sec_author_ids = s.get("authors", [])
            sec_author_names = ", ".join(id_to_name.get(a, a[:8]) for a in sec_author_ids[:3])
            rec_count = sec_path_record_count.get(p, 0)
            revoked_sections.append({
                "section_hash": s.get("section_hash", ""),
                "path": p,
                "title": parts[0] if parts else path_display,
                "heading": parts[1] if len(parts) > 1 else "",
                "authors": sec_author_names,
                "record_count": rec_count,
            })
        elif is_cascade:
            cascade_sections.append({
                "path": p.replace("raw/", ""),
                "section_hash": s.get("section_hash", ""),
            })

    result = {
        "revoked_authors": revoked_authors,
        "revoked_sections": revoked_sections,
        "cascade_count": len(cascade_sections),
    }
    return result


def _stream_sections_for_author(dataset: str, author_id: str, field: str):
    ob_dir = Path(_ob_dir(dataset))
    for s in shard_iterate_all(ob_dir, LAYER_SECTION):
        if author_id in s.get(field, []):
            yield s


@app.post("/api/{dataset}/restore/author", response_model=MessageResponse)
def api_restore_author(dataset: str, req: RestoreAuthorRequest):
    ob_dir = Path(_ob_dir(dataset))
    try:
        from ob.authors import invalidate_cache
        restore_author(ob_dir, author_id=req.author_id)
        invalidate_cache(ob_dir)
        invalidate_overview_cache(dataset)
        return MessageResponse(message=f"Author restored")
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/{dataset}/restore/section", response_model=MessageResponse)
def api_restore_section(dataset: str, req: RestoreSectionRequest):
    ob_dir = Path(_ob_dir(dataset))
    try:
        restore_section(ob_dir, section_hash=req.section_hash)
        invalidate_overview_cache(dataset)
        return MessageResponse(message=f"Section restored")
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/{dataset}/audit-log")
def get_audit_log(dataset: str, op: Optional[str] = None):
    ob_dir = Path(_ob_dir(dataset))
    entries = query_log(ob_dir, op=op)
    # Convert datetime objects to strings
    for e in entries:
        if "ts" in e and not isinstance(e["ts"], str):
            e["ts"] = str(e["ts"])
    return {"entries": entries[:200], "total": len(entries)}


@app.post("/api/{dataset}/reset", response_model=MessageResponse)
def api_reset(dataset: str):
    ob_dst = Path(_ob_dir(dataset)) / ".ob"
    bkp = _backup_dir(dataset)
    if bkp.exists():
        shutil.rmtree(ob_dst)
        shutil.copytree(bkp, ob_dst)
    invalidate_overview_cache(dataset)
    return MessageResponse(message="Demo state reset. All revocations undone.")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
