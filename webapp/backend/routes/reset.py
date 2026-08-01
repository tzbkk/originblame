from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ob_client import dataset_exists, restore_ob

router = APIRouter()


@router.post("/{dataset}/reset")
def reset_demo(dataset: str):
    if not dataset_exists(dataset):
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset}")

    try:
        restore_ob(dataset)
        return {"success": True, "message": "Demo state reset. All revocations undone."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
