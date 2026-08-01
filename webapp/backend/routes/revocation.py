from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ob_client import (
    dataset_exists,
    load_authors,
    load_sections,
    load_records,
    load_audit_log,
    revoked_author_ids,
    record_sec_path,
    do_revoke_author,
    do_revoke_section,
    do_restore_author,
    do_restore_section,
)

router = APIRouter()


class RevokeRequest(BaseModel):
    type: str
    target: dict


class RestoreRequest(BaseModel):
    type: str
    target: dict


@router.post("/{dataset}/revoke")
def revoke(dataset: str, req: RevokeRequest):
    if not dataset_exists(dataset):
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset}")

    try:
        if req.type == "author":
            email = req.target.get("email")
            author_id = req.target.get("author_id") or req.target.get("id")
            count = do_revoke_author(dataset, email=email, author_id=author_id)
            return {"success": True, "message": f"Revoked {count} author(s)"}

        elif req.type == "section":
            section_hash = req.target.get("section_hash")
            path = req.target.get("path")
            count = do_revoke_section(dataset, section_hash=section_hash, path=path)
            return {"success": True, "message": f"Revoked {count} section(s)"}

        elif req.type == "record":
            path = req.target.get("path")
            if not path:
                title = req.target.get("title", "")
                heading = req.target.get("heading", "")
                path = f"raw/{title}#{heading}"
            count = do_revoke_section(dataset, path=path)
            return {"success": True, "message": f"Revoked {count} section(s) for record"}

        else:
            raise HTTPException(status_code=400, detail=f"Unknown revocation type: {req.type}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{dataset}/restore")
def restore(dataset: str, req: RestoreRequest):
    if not dataset_exists(dataset):
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset}")

    try:
        if req.type == "author":
            email = req.target.get("email")
            author_id = req.target.get("author_id") or req.target.get("id")
            count = do_restore_author(dataset, email=email, author_id=author_id)
            return {"success": True, "message": f"Restored {count} author(s)"}

        elif req.type == "section":
            section_hash = req.target.get("section_hash")
            path = req.target.get("path")
            count = do_restore_section(dataset, section_hash=section_hash, path=path)
            return {"success": True, "message": f"Restored {count} section(s)"}

        else:
            raise HTTPException(status_code=400, detail=f"Unknown restore type: {req.type}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset}/revoked")
def list_revoked(dataset: str):
    if not dataset_exists(dataset):
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset}")

    authors = load_authors(dataset)
    sections = load_sections(dataset)
    records = load_records(dataset)
    id_to_name = {a["id"]: a["name"] for a in authors}

    revoked_authors = [
        {
            "id": a["id"],
            "name": a.get("name", ""),
            "email": a.get("email", ""),
            "sections_affected": sum(
                1 for s in sections if a["id"] in s.get("authors", [])
            ),
        }
        for a in authors
        if a.get("revoked")
    ]

    revoked_aids = revoked_author_ids(authors)
    revoked_sections = []
    for s in sections:
        if s.get("revoked"):
            revoked_sections.append(
                {
                    "section_hash": s.get("section_hash", ""),
                    "path": s.get("path", ""),
                    "authors": [
                        id_to_name.get(aid, aid[:8]) for aid in s.get("authors", [])
                    ],
                    "revoked_direct": True,
                }
            )
        elif bool(set(s.get("authors", [])) & revoked_aids):
            revoked_sections.append(
                {
                    "section_hash": s.get("section_hash", ""),
                    "path": s.get("path", ""),
                    "authors": [
                        id_to_name.get(aid, aid[:8]) for aid in s.get("authors", [])
                    ],
                    "revoked_direct": False,
                }
            )

    return {"authors": revoked_authors, "sections": revoked_sections}
