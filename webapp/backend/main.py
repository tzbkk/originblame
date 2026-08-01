"""OriginBlame provenance dashboard API."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Ensure ob_client can be imported from same directory
_BACKEND_DIR = str(Path(__file__).parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from routes import datasets, overview, authors, records, revocation, audit, reset  # noqa: E402

app = FastAPI(title="OriginBlame API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_router = FastAPI()

api_router.include_router(datasets.router, tags=["datasets"])
api_router.include_router(overview.router, tags=["overview"])
api_router.include_router(authors.router, tags=["authors"])
api_router.include_router(records.router, tags=["records"])
api_router.include_router(revocation.router, tags=["revocation"])
api_router.include_router(audit.router, tags=["audit"])
api_router.include_router(reset.router, tags=["reset"])

app.mount("/api", api_router)

# Serve React production build as static files (mounted after API routes)
_STATIC_DIR = Path(__file__).parent.parent / "frontend" / "dist"
if _STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")
