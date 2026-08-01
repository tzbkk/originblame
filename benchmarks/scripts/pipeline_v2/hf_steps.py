from __future__ import annotations

"""Generators and mappers for HuggingFace Datasets pipeline.

Provides plain Python generators for Dataset.from_generator() and stateful
callables for ds.map() — all designed for single-process (num_proc=1)
execution due to ob concurrency constraints.

HuggingFace pipelines produce JSONL output only — no packed binary.
"""

import json
from pathlib import Path
import logging

from benchmarks.scripts.pipeline_v2.mediawiki_parser import (
    find_dump_files,
    page_to_authors,
    parse_mediawiki_stream,
)
from benchmarks.scripts.pipeline_v2._native_compat import (
    compute_hash,
    index_document,
    section_add_raw,
)
from benchmarks.scripts.pipeline_v2.kernel_parser import (
    parse_kernel_files,
)

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Generators for Dataset.from_generator()
# ═══════════════════════════════════════════════════════════════════════════════


def zhwiki_generator(
    raw_data_dir: Path,
    scale: int,
    license: str = "CC-BY-SA-4.0",
):
    """Yield dicts for HuggingFace Dataset.from_generator() from MediaWiki dumps.

    Finds dump files via find_dump_files(), iterates dumps in order, parses pages
    via parse_mediawiki_stream(), and yields **one dict per chunk** until *scale*
    chunks reached (cumulative across dump files).  Pages with wikitext < 50 chars
    are skipped by the parser itself.

    Each yielded dict carries ``_ob_source_path`` so that the downstream OBMapper
    can register a per-chunk section (e.g. ``raw/北京#历史``) and link the
    doc-index entry to it.  JSONL granularity == doc-index granularity.
    """
    raw_data_dir = Path(raw_data_dir)
    yielded = 0

    for dump_path in find_dump_files(raw_data_dir):
        for page in parse_mediawiki_stream(dump_path):
            if page.chunks:
                for chunk in page.chunks:
                    yield {
                        "text": chunk.raw_text,
                        "title": page.title,
                        "heading": chunk.heading,
                        "authors_json": json.dumps(
                            [{"name": a, "email": f"{a}@mediawiki"} for a in chunk.authors],
                            ensure_ascii=False,
                        ),
                        "contributors_json": json.dumps(
                            [{"name": c, "email": f"{c}@mediawiki"} for c in page.contributors],
                            ensure_ascii=False,
                        ),
                        "year": page.year,
                        "license": license,
                        "_ob_source_path": chunk.source_path,
                        "_ob_page_contributors": page.contributors,
                    }
                    yielded += 1
                    if yielded >= scale:
                        return
            else:
                yield {
                    "text": page.wikitext,
                    "title": page.title,
                    "authors_json": json.dumps(page_to_authors(page), ensure_ascii=False),
                    "contributors_json": json.dumps(
                        [{"name": c, "email": f"{c}@mediawiki"} for c in page.contributors],
                        ensure_ascii=False,
                    ),
                    "year": page.year,
                    "license": license,
                    "_ob_page_contributors": page.contributors,
                }
                yielded += 1
                if yielded >= scale:
                    return


def kernel_generator(
    linux_dir: Path,
    scale: int,
):
    """Yield dicts for HuggingFace Dataset.from_generator() from Linux kernel source."""
    from benchmarks.scripts.pipeline_v2.kernel_parser import parse_kernel_files

    yielded = 0
    for kf in parse_kernel_files(Path(linux_dir), scale):
        yield {
            "text": kf.text,
            "title": kf.path,
            "authors_json": json.dumps(kf.authors, ensure_ascii=False),
            "year": kf.year,
            "license": kf.license,
        }
        yielded += 1
        if yielded >= scale:
            return


# ═══════════════════════════════════════════════════════════════════════════════
# OBMapper — stateful callable for ds.map()
# ═══════════════════════════════════════════════════════════════════════════════


class OBMapper:
    """Stateful callable with ob provenance (manifest-only, no token-index).

    Single-process only (num_proc=1) due to shared author_cache and ob state.

    For per-chunk records (carrying ``_ob_source_path``), registers a per-chunk
    section and links the doc-index entry to it.  Doc-index granularity always
    matches JSONL granularity (1 JSONL line = 1 doc-index entry).
    """

    def __init__(self, ob_dir: Path):
        import ob

        self.ob_dir = Path(ob_dir)
        self._ob = ob

        self.author_cache: dict[str, str] = {}
        self.stats = {
            "authors_registered": 0,
            "sections_registered": 0,
            "tracks_ok": 0,
            "tracks_failed": 0,
            "total_bytes": 0,
        }

    def __call__(self, example: dict) -> dict:
        year = example["year"]
        license_ = example["license"]
        title = example["title"]
        text = example["text"]

        ob_source_path = example.pop("_ob_source_path", None)
        page_contributors = example.get("_ob_page_contributors", [])

        try:
            authors_meta = json.loads(example["authors_json"])

            for a in authors_meta:
                name, email = a["name"], a["email"]
                if name not in self.author_cache:
                    aid = self._ob.author_add(name, email, ob_dir=self.ob_dir)
                    self.author_cache[name] = aid
                    self.stats["authors_registered"] += 1

            # Register page-level contributors as authors too (Ship of Theseus)
            contributor_ids: list[str] = []
            for name in page_contributors:
                if name not in self.author_cache:
                    aid = self._ob.author_add(name, f"{name}@mediawiki", ob_dir=self.ob_dir)
                    self.author_cache[name] = aid
                    self.stats["authors_registered"] += 1
                aid = self.author_cache[name]
                if aid not in contributor_ids:
                    contributor_ids.append(aid)

            author_ids = [self.author_cache[a["name"]] for a in authors_meta]

            section_path = ob_source_path or f"raw/{title}"
            section_hash = section_add_raw(
                section_path, author_ids, contributor_ids, license_, year, str(self.ob_dir),
            )
            self.stats["sections_registered"] += 1

            line_hash = compute_hash(example)
            index_document(str(self.ob_dir), line_hash, "data.jsonl", [section_hash])
            self.stats["tracks_ok"] += 1
            self.stats["total_bytes"] += len(text)

        except Exception as exc:
            self.stats["tracks_failed"] += 1
            log.error("OBMapper failed for %s: %s", example.get("title", "?"), exc)

        return example


# ═══════════════════════════════════════════════════════════════════════════════
# TokenizeMapper — baseline (no OB), stateful callable for ds.map()
# ═══════════════════════════════════════════════════════════════════════════════


class BaselineMapper:
    """Baseline callable — adds byte count, no OB, no tokenization."""

    def __call__(self, example: dict) -> dict:
        example["byte_count"] = len(example["text"].encode("utf-8"))
        return example


# ═══════════════════════════════════════════════════════════════════════════════
# Streaming writer — single-pass generator → JSONL
# ═══════════════════════════════════════════════════════════════════════════════


def stream_and_write(
    generator,
    mapper,
    jsonl_path: Path,
) -> dict:
    jsonl_path = Path(jsonl_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    doc_count = 0
    total_bytes = 0

    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for doc in generator:
            if mapper is not None:
                doc = mapper(doc)
            doc_count += 1
            total_bytes += len(doc["text"].encode("utf-8"))
            jf.write(json.dumps(doc, ensure_ascii=False) + "\n")

    return {
        "doc_count": doc_count,
        "total_bytes": total_bytes,
    }
