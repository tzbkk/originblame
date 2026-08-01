"""Pure parsing adapter for MediaWiki XML dump files.

Streams XML line-by-line, reuses ob_util's _extract_page_info() for page parsing,
and yields MediaWikiPage dataclass instances. No OB API calls — pure parsing only.

Supports streaming decompression of .7z / .tar.gz / .tar.bz2 / .xml files
without writing intermediate data to disk.
"""
from __future__ import annotations

import codecs
import subprocess
import sys
import tarfile
from collections.abc import Generator, Iterator
from dataclasses import dataclass, field
from pathlib import Path

# ob-util may live in a sibling repo; resolve relative to this file
try:
    from ob_util.parsers.mediawiki import (
        _extract_all_revisions,
        _extract_page_info,
    )
except ImportError:
    _ob_util_src = (
        Path(__file__).resolve().parents[5]
        / "rust-originblame" / "python" / "packages" / "ob-util" / "src"
    )
    sys.path.insert(0, str(_ob_util_src))
    from ob_util.parsers.mediawiki import (
        _extract_all_revisions,
        _extract_page_info,
    )

_MAX_PAGE_BYTES = 50 * 1024 * 1024
_MIN_WIKITEXT_LEN = 50


@dataclass
class MediaWikiPage:
    title: str
    year: str
    wikitext: str
    contributors: list[str] = field(default_factory=list)
    chunks: list = field(default_factory=list)


@dataclass
class ParseStats:
    pages_parsed: int = 0
    pages_skipped: int = 0
    pages_parse_failed: int = 0


def _open_text_lines(file_path: Path) -> Iterator[str]:
    """Yield decoded text lines from .xml, .7z, .tar.gz, .tar.bz2, or .tgz.

    .7z  — pipes through the ``7z`` CLI (``7z e -so``). Requires ``7z`` on PATH.
    .tar.gz / .tgz / .tar.bz2 — uses stdlib ``tarfile`` in streaming mode
    (``r|gz`` / ``r|bz2``). Extracts the first ``.xml`` member found.
    .xml — plain ``open()``.
    """
    name = file_path.name.lower()

    if name.endswith(".7z"):
        proc = subprocess.Popen(
            ["7z", "e", "-so", str(file_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        assert proc.stdout is not None
        reader = codecs.getreader("utf-8")(proc.stdout)
        try:
            for line in reader:
                yield line.replace("\r\n", "\n").replace("\r", "\n")
        finally:
            proc.kill()
            proc.wait()

    elif name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(file_path, "r|gz") as tf:
            for member in tf:
                if member.isfile() and member.name.lower().endswith(".xml"):
                    f = tf.extractfile(member)
                    if f is None:
                        continue
                    yield from codecs.getreader("utf-8")(f)

    elif name.endswith(".tar.bz2"):
        with tarfile.open(file_path, "r|bz2") as tf:
            for member in tf:
                if member.isfile() and member.name.lower().endswith(".xml"):
                    f = tf.extractfile(member)
                    if f is None:
                        continue
                    yield from codecs.getreader("utf-8")(f)

    else:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            yield from f


def parse_mediawiki_stream(
    file_path: Path,
    max_pages: int | None = None,
) -> Generator[MediaWikiPage, None, None]:
    """Stream-parse a MediaWiki XML dump, yielding MediaWikiPage for each valid page.

    Args:
        file_path: Path to a .xml, .7z, .tar.gz, .tar.bz2, or .tgz dump file.
        max_pages: Stop after yielding this many pages (None = no limit).

    Yields:
        MediaWikiPage instances with title, year, wikitext, and contributors.
    """
    file_path = Path(file_path)
    stats = ParseStats()

    buf: list[str] = []
    in_page = False
    page_bytes = 0
    yielded = 0

    for line in _open_text_lines(file_path):
        if "<page>" in line:
            in_page = True
            buf = [line]
            page_bytes = len(line)
            continue

        if not in_page:
            continue

        page_bytes += len(line)
        if page_bytes > _MAX_PAGE_BYTES:
            if "</page>" in line:
                stats.pages_parse_failed += 1
                in_page = False
                buf = []
            continue

        buf.append(line)
        if "</page>" in line:
            in_page = False

            info = _extract_all_revisions(buf)
            if info is None:
                info = _extract_page_info(buf)
            buf = []

            if info is None:
                stats.pages_skipped += 1
                continue

            wikitext = info["wikitext"]
            if len(wikitext) < _MIN_WIKITEXT_LEN:
                stats.pages_skipped += 1
                continue

            chunks = info.get("chunks", [])
            page = MediaWikiPage(
                title=info["title"],
                year=info["year"],
                wikitext=wikitext,
                contributors=info["contributors"],
                chunks=chunks,
            )
            stats.pages_parsed += 1
            yielded += 1
            yield page

            if max_pages is not None and yielded >= max_pages:
                return


_SUPPORTED_EXTENSIONS = (".xml", ".7z", ".tar.gz", ".tgz", ".tar.bz2")


def _has_supported_extension(name: str) -> bool:
    lower = name.lower()
    return any(lower.endswith(ext) for ext in _SUPPORTED_EXTENSIONS)


def find_dump_files(raw_data_dir: Path) -> list[Path]:
    """Find MediaWiki dump files in sorted order (deterministic).

    Matches ``zhwiki-*-pages-meta-history*`` with any supported extension
    (.xml, .7z, .tar.gz, .tgz, .tar.bz2).

    Returns:
        Sorted list of matching file paths. Empty list if directory
        doesn't exist or no files match.
    """
    raw_data_dir = Path(raw_data_dir)
    if not raw_data_dir.is_dir():
        return []

    candidates: list[Path] = []
    for p in raw_data_dir.iterdir():
        if not p.is_file():
            continue
        if "pages-meta-history" not in p.name or not p.name.startswith("zhwiki-"):
            continue
        if _has_supported_extension(p.name):
            candidates.append(p)

    return sorted(candidates)


def page_to_authors(page: MediaWikiPage) -> list[dict[str, str]]:
    return [{"name": name, "email": f"{name}@mediawiki"} for name in page.contributors]
