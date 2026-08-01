from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ob_client import (
    dataset_exists,
    load_overview,
)

router = APIRouter()


@router.get("/{dataset}/overview")
def get_overview(dataset: str):
    if not dataset_exists(dataset):
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset}")

    return load_overview(dataset)
