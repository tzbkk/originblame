from __future__ import annotations

import json
from collections import Counter

from fastapi import APIRouter, HTTPException, Query

from ob_client import (
    dataset_exists,
    load_authors,
    load_sections,
    load_records,
    author_contribution_map,
    revoked_author_ids,
    record_sec_path,
    sec_path_revoked_map,
)

router = APIRouter()

MAX_LIMIT = 200
DEFAULT_LIMIT = 50


@router.get("/{dataset}/authors")
def list_authors(
    dataset: str,
    search: str = Query("", description="Case-insensitive substring match on name"),
    page: int = Query(1, ge=1),
    limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
):
    if not dataset_exists(dataset):
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset}")

    authors = load_authors(dataset)
    sections = load_sections(dataset)
    contrib = author_contribution_map(sections)
    total_sections = len(sections)

    q = search.lower()
    if q:
        filtered = [a for a in authors if q in a.get("name", "").lower()]
    else:
        filtered = authors

    filtered.sort(key=lambda a: contrib.get(a["id"], 0), reverse=True)

    total = len(filtered)
    start = (page - 1) * limit
    end = start + limit
    page_items = filtered[start:end]

    items = []
    for a in page_items:
        aid = a["id"]
        sec_count = contrib.get(aid, 0)
        items.append(
            {
                "name": a.get("name", ""),
                "email": a.get("email", ""),
                "id": aid,
                "sections": sec_count,
                "contribution_pct": round(sec_count / max(total_sections, 1) * 100, 1),
                "revoked": a.get("revoked", False),
            }
        )

    return {"items": items, "total": total, "page": page, "limit": limit}


@router.get("/{dataset}/authors/{author_id}")
def get_author_detail(dataset: str, author_id: str):
    if not dataset_exists(dataset):
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset}")

    authors = load_authors(dataset)
    sections = load_sections(dataset)
    records = load_records(dataset)

    author = next((a for a in authors if a["id"] == author_id), None)
    if author is None:
        raise HTTPException(status_code=404, detail=f"Author not found: {author_id}")

    author_sec_paths = {
        s.get("path") for s in sections if author_id in s.get("authors", [])
    }
    contributor_sec_paths = {
        s.get("path") for s in sections if author_id in s.get("contributors", [])
    }

    author_records = [
        r
        for r in records
        if record_sec_path(r) in author_sec_paths
    ]
    contributor_records = [
        r
        for r in records
        if record_sec_path(r) in contributor_sec_paths
    ]

    author_recs = [
        {
            "title": r.get("title", ""),
            "heading": r.get("heading", ""),
            "year": r.get("year", ""),
            "text_preview": r.get("text", "")[:150].replace("\n", " ") + "\u2026",
        }
        for r in author_records[:200]
    ]
    contributor_recs = [
        {
            "title": r.get("title", ""),
            "heading": r.get("heading", ""),
            "year": r.get("year", ""),
            "text_preview": r.get("text", "")[:150].replace("\n", " ") + "\u2026",
        }
        for r in contributor_records[:200]
    ]

    return {
        "id": author["id"],
        "name": author.get("name", ""),
        "email": author.get("email", ""),
        "revoked": author.get("revoked", False),
        "sections_author": len(author_sec_paths),
        "sections_contributor": len(contributor_sec_paths),
        "records": author_recs,
        "contributor_records": contributor_recs,
    }
