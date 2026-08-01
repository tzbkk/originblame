from fastapi import APIRouter

from ob_client import discover_datasets

router = APIRouter()


@router.get("/datasets")
def list_datasets():
    return discover_datasets()
