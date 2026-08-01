from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ob_client import dataset_exists, load_audit_log

router = APIRouter()

ICON_MAP = {
    "init": "rocket",
    "revoke_author": "red_circle",
    "revoke": "red_circle",
    "restore_author": "green_circle",
    "restore_section": "green_circle",
    "clean": "broom",
}


@router.get("/{dataset}/audit")
def get_audit_log(dataset: str):
    if not dataset_exists(dataset):
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset}")

    entries = load_audit_log(dataset)

    items = []
    for e in entries:
        op = e.get("op", "?")
        detail = e.get("detail", "")
        if isinstance(detail, dict):
            detail = str(detail)
        items.append(
            {
                "ts": e.get("ts", ""),
                "op": op,
                "detail": detail,
                "icon": ICON_MAP.get(op, "pin"),
            }
        )

    return items
