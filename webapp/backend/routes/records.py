from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Query

from ob_client import (
    dataset_exists,
    load_authors,
    load_sections,
    load_records,
    revoked_author_ids,
    record_sec_path,
    sec_path_revoked_map,
)

router = APIRouter()

MAX_LIMIT = 200
DEFAULT_LIMIT = 50


@router.get("/{dataset}/records")
def list_records(
    dataset: str,
    search: str = Query("", description="Substring match on title + text"),
    author: str = Query("", description="Filter by author name"),
    status: str = Query("", description="Filter: 'active' or 'revoked'"),
    year: str = Query("", description="Filter by year"),
    page: int = Query(1, ge=1),
    limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
):
    if not dataset_exists(dataset):
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset}")

    authors_list = load_authors(dataset)
    sections = load_sections(dataset)
    records = load_records(dataset)

    revoked_aids = revoked_author_ids(authors_list)
    sp_revoked = sec_path_revoked_map(sections, revoked_aids)

    filtered = records

    if search:
        q = search.lower()
        filtered = [
            r
            for r in filtered
            if q in r.get("title", "").lower() or q in r.get("text", "").lower()
        ]

    if author:
        filtered = [
            r
            for r in filtered
            if any(
                a.get("name", "") == author
                for a in json.loads(r.get("authors_json", "[]"))
            )
        ]

    if year:
        filtered = [r for r in filtered if str(r.get("year", "")) == year]

    if status:
        if status == "active":
            filtered = [
                r for r in filtered if not sp_revoked.get(record_sec_path(r), False)
            ]
        elif status == "revoked":
            filtered = [
                r for r in filtered if sp_revoked.get(record_sec_path(r), False)
            ]

    total = len(filtered)
    start = (page - 1) * limit
    end = start + limit
    page_items = filtered[start:end]

    items = []
    for r in page_items:
        r_authors = json.loads(r.get("authors_json", "[]"))
        author_names = ", ".join(a.get("name", "") for a in r_authors[:5])
        if len(r_authors) > 5:
            author_names += f" (+{len(r_authors) - 5})"
        text = r.get("text", "")
        preview = text[:120].replace("\n", " ") + ("\u2026" if len(text) > 120 else "")
        sp = record_sec_path(r)
        is_revoked = sp_revoked.get(sp, False)
        items.append(
            {
                "title": r.get("title", ""),
                "heading": r.get("heading", ""),
                "preview": preview,
                "authors": author_names,
                "year": r.get("year", ""),
                "status": "revoked" if is_revoked else "active",
            }
        )

    return {"items": items, "total": total, "page": page, "limit": limit}


@router.get("/{dataset}/records/{idx}")
def get_record_detail(dataset: str, idx: int):
    if not dataset_exists(dataset):
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset}")

    records = load_records(dataset)
    sections = load_sections(dataset)

    if idx < 0 or idx >= len(records):
        raise HTTPException(status_code=404, detail=f"Record index out of range: {idx}")

    r = records[idx]
    sp = record_sec_path(r)
    matching_secs = [s for s in sections if s.get("path") == sp]
    sec = matching_secs[0] if matching_secs else None

    r_authors = json.loads(r.get("authors_json", "[]"))
    contributor_count = len(sec.get("contributors", [])) if sec else 0

    return {
        "title": r.get("title", ""),
        "heading": r.get("heading", ""),
        "year": r.get("year", ""),
        "license": r.get("license", ""),
        "text": r.get("text", ""),
        "authors": [{"name": a.get("name", ""), "email": a.get("email", "")} for a in r_authors],
        "section_hash": sec.get("section_hash", "") if sec else "",
        "section_path": sec.get("path", "") if sec else "",
        "contributor_count": contributor_count,
    }
