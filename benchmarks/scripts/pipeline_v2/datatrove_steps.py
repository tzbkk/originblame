from __future__ import annotations

"""Datatrove PipelineStep subclasses for pipeline v2 benchmarks.

Four steps:
  - MediaWikiReader: reads zhwiki XML dumps, yields Documents
  - KernelReader: reads Linux kernel C files with git attribution, yields Documents
  - OBTrack: writes token-index entries for each Document
  - PreTokenizedDocumentTokenizer: tokenizes, reusing pre-computed IDs when available
"""

import logging
from pathlib import Path

from datatrove.data import Document
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.tokens.tokenizer import DocumentTokenizer

from benchmarks.scripts.pipeline_v2._native_compat import (
    token_index_write_pid,
)
from benchmarks.scripts.pipeline_v2.kernel_parser import (
    parse_kernel_files,
)
from benchmarks.scripts.pipeline_v2.mediawiki_parser import (
    find_dump_files,
    page_to_authors,
    parse_mediawiki_stream,
)
from benchmarks.scripts.pipeline_v2.shared import load_tokenizer

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# MediaWikiReader
# ═══════════════════════════════════════════════════════════════════════════════


class MediaWikiReader(PipelineStep):
    """Read zhwiki XML dump files and yield Documents.

    Iterates dump files found by ``find_dump_files()``, parses pages via
    ``parse_mediawiki_stream()``, and yields **one Document per chunk**.
    Stops after yielding *scale* chunks total (cumulative across all dumps).
    """

    def __init__(
        self,
        raw_data_dir: Path,
        scale: int,
        license: str = "CC-BY-SA-4.0",
    ) -> None:
        self.raw_data_dir = Path(raw_data_dir)
        self.scale = scale
        self.license = license
        self._reader_stats: dict[str, int] = {}

    def run(self, data, rank: int = 0, world_size: int = 1):
        chunks_yielded = 0
        pages_seen = 0
        dumps_processed = 0

        dump_files = find_dump_files(self.raw_data_dir)
        if not dump_files:
            log.warning("No dump files found in %s", self.raw_data_dir)

        for dump_path in dump_files:
            for page in parse_mediawiki_stream(dump_path):
                page_authors = page_to_authors(page)
                pages_seen += 1

                if page.chunks:
                    for chunk in page.chunks:
                        yield Document(
                            text=chunk.raw_text,
                            id=f"{page.title}#{chunk.heading}",
                            metadata={
                                "authors": [{"name": a, "email": f"{a}@mediawiki"} for a in chunk.authors],
                                "year": page.year,
                                "license": self.license,
                                "title": page.title,
                                "_ob_source_path": chunk.source_path,
                                "_ob_page_contributors": page.contributors,
                            },
                        )
                        chunks_yielded += 1
                        if chunks_yielded >= self.scale:
                            break
                else:
                    yield Document(
                        text=page.wikitext,
                        id=page.title,
                        metadata={
                            "authors": page_authors,
                            "year": page.year,
                            "license": self.license,
                            "title": page.title,
                            "_ob_page_contributors": page.contributors,
                        },
                    )
                    chunks_yielded += 1

                if chunks_yielded >= self.scale:
                    break

            dumps_processed += 1
            if chunks_yielded >= self.scale:
                break

        self._reader_stats = {
            "chunks_yielded": chunks_yielded,
            "pages_seen": pages_seen,
            "dumps_processed": dumps_processed,
        }
        log.info(
            "MediaWikiReader done: %d chunks from %d pages, %d dumps",
            chunks_yielded, pages_seen, dumps_processed,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# KernelReader
# ═══════════════════════════════════════════════════════════════════════════════


class KernelReader(PipelineStep):
    """Read Linux kernel C files with git log attribution and yield Documents."""

    def __init__(
        self,
        linux_dir: Path,
        scale: int,
    ) -> None:
        self.linux_dir = Path(linux_dir)
        self.scale = scale
        self._reader_stats: dict[str, int] = {}

    def run(self, data, rank: int = 0, world_size: int = 1):
        files_ok = 0

        for kf in parse_kernel_files(self.linux_dir, self.scale):
            files_ok += 1
            yield Document(
                text=kf.text,
                id=kf.path,
                metadata={
                    "authors": kf.authors,
                    "year": kf.year,
                    "license": kf.license,
                    "title": kf.path,
                },
            )

            if files_ok >= self.scale:
                break

        self._reader_stats = {
            "files_ok": files_ok,
        }
        log.info("KernelReader done: %d files", files_ok)


# ═══════════════════════════════════════════════════════════════════════════════
# OBTrack
# ═══════════════════════════════════════════════════════════════════════════════


class OBTrack(PipelineStep):
    """Write token-index entries with author attribution for each Document.

    For per-chunk records (carrying ``_ob_source_path`` metadata), registers a
    per-chunk section and links the token-index entry to it.  Token-index
    granularity matches Document granularity (1 Document = 1 token-index entry).
    """

    def __init__(
        self,
        ob_dir: Path,
        tokenizer_name: str = "gpt2",
    ) -> None:
        self.ob_dir = Path(ob_dir)
        self.tokenizer_name = tokenizer_name
        self._tokenizer = None

        self._author_cache: dict[str, str] = {}
        self._ob_stats: dict[str, int] = {
            "tracks_ok": 0,
            "tracks_failed": 0,
            "total_tokens": 0,
            "authors_registered": 0,
            "sections_registered": 0,
        }

    def _get_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = load_tokenizer(self.tokenizer_name)
        return self._tokenizer

    def run(self, data, rank: int = 0, world_size: int = 1):
        for doc in data:
            try:
                self._process_doc(doc)
            except Exception as exc:
                self._ob_stats["tracks_failed"] += 1
                log.error("OBTrack failed for doc %s: %s", doc.id, exc)
            yield doc

    def _process_doc(self, doc: Document) -> None:
        import ob as _ob
        from benchmarks.scripts.pipeline_v2._native_compat import (
            section_add_raw,
        )

        ob_source_path = doc.metadata.pop("_ob_source_path", None)
        page_contributors: list[str] = doc.metadata.pop("_ob_page_contributors", [])
        authors_meta = doc.metadata.get("authors", [])
        if not authors_meta:
            return

        year = doc.metadata.get("year", "")
        license_ = doc.metadata.get("license", "")
        title = doc.id or ""

        author_ids = []
        for a in authors_meta:
            name, email = a["name"], a["email"]
            if name not in self._author_cache:
                aid = _ob.author_add(name, email, ob_dir=self.ob_dir)
                self._author_cache[name] = aid
                self._ob_stats["authors_registered"] += 1
            author_ids.append(self._author_cache[name])

        # Register page-level contributors as authors too (Ship of Theseus)
        contributor_ids: list[str] = []
        for name in page_contributors:
            if name not in self._author_cache:
                aid = _ob.author_add(name, f"{name}@mediawiki", ob_dir=self.ob_dir)
                self._author_cache[name] = aid
                self._ob_stats["authors_registered"] += 1
            aid = self._author_cache[name]
            if aid not in contributor_ids:
                contributor_ids.append(aid)

        section_path = ob_source_path or f"raw/{title}"
        section_hash = section_add_raw(
            section_path, author_ids, contributor_ids, license_, year, str(self.ob_dir),
        )
        self._ob_stats["sections_registered"] += 1

        tok = self._get_tokenizer()
        encoding = tok.encode(doc.text)
        token_count = len(encoding.ids)
        doc.metadata["token_count"] = token_count
        doc.metadata["_ob_token_ids"] = encoding.ids

        token_index_write_pid(
            str(self.ob_dir),
            self.tokenizer_name,
            token_count,
            [section_hash],
        )

        self._ob_stats["tracks_ok"] += 1
        self._ob_stats["total_tokens"] += token_count


# ═══════════════════════════════════════════════════════════════════════════════
# PreTokenizedDocumentTokenizer
# ═══════════════════════════════════════════════════════════════════════════════


class PreTokenizedDocumentTokenizer(DocumentTokenizer):
    """DocumentTokenizer that reuses pre-computed token IDs from metadata.

    When ``doc.metadata["_ob_token_ids"]`` is present (set by OBTrack), those
    IDs are used directly instead of re-tokenizing ``doc.text``.  This avoids
    the double-tokenization overhead when OB provenance is enabled.

    When no pre-computed tokens are found, falls back to normal tokenization.
    """

    def write_unshuffled(self, data, filename):
        """Override that skips tokenization when ``_ob_token_ids`` is present."""
        from datatrove.pipeline.tokens.tokenizer import TokenizedFile
        from datatrove.utils.batching import batched

        unshuff = TokenizedFile(
            self.output_folder
            if (not self.shuffle_documents and not self.shuffle_chunk_size)
            or not self.local_working_dir
            else self.local_working_dir,
            filename,
            save_index=self.save_index
            or self.shuffle_documents
            or self.shuffle_chunk_size,
            save_loss_metadata=self.save_loss_metadata,
            upload_block_size=self.upload_block_size,
            tokenizer_name_or_path=self.tokenizer_name_or_path,
            save_final_metadata=self.save_final_metadata,
            token_size=self.token_size,
        )

        for batch in batched(data, self.batch_size):
            with self.track_time(unit="batch"):
                # Fast path: all docs have pre-computed tokens
                all_pre = True
                for doc in batch:
                    if "_ob_token_ids" not in doc.metadata:
                        all_pre = False
                        break

                if all_pre:
                    for document in batch:
                        tokens = document.metadata.pop("_ob_token_ids")
                        unshuff.write(tokens, None)
                        self.stat_update("tokens", value=len(tokens))
                    continue

                # Slow path: mixed or no pre-computed tokens
                pre_map: dict[int, list[int]] = {}
                to_encode: list[int] = []
                for i, doc in enumerate(batch):
                    if "_ob_token_ids" in doc.metadata:
                        pre_map[i] = doc.metadata.pop("_ob_token_ids")
                    else:
                        to_encode.append(i)

                encoded_list = (
                    self.tokenizer.encode_batch(
                        [batch[i].text for i in to_encode]
                    )
                    if to_encode
                    else []
                )

                enc_iter = iter(encoded_list)
                for i, doc in enumerate(batch):
                    if i in pre_map:
                        tokens = pre_map[i]
                        loss_values = None
                    else:
                        encoded = next(enc_iter)
                        tokens = encoded.ids
                        loss_values = self.get_loss_values(doc, encoded)
                        if (
                            loss_values is not None
                            and len(loss_values) < len(tokens)
                        ):
                            tokens = tokens[: len(loss_values)]
                    unshuff.write(tokens, loss_values)
                    self.stat_update("tokens", value=len(tokens))

        unshuff.close()
        return unshuff
