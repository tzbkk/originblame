from __future__ import annotations

"""Kernel source files with git log author attribution.

Walks filesystem for source files, uses a single bulk `git log --name-only` to
get all commit authors per file. No line-level blame — git log is sufficient for
record-level attribution (every author listed has at least one commit touching
the file).

No OB API calls — pure filesystem + git parsing.
"""

import logging
import os
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

from benchmarks.scripts.pipeline_v2.shared import SKIP_DIRS

log = logging.getLogger(__name__)

_KERNEL_EXTENSIONS = (".c", ".h", ".S", ".rs", ".dts", ".dtsi")


# ═══════════════════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class KernelFile:
    path: str
    text: str
    authors: list[dict[str, str]]
    license: str
    year: str
    line_count: int


# ═══════════════════════════════════════════════════════════════════════════════
# Attribution: bulk git log
# ═══════════════════════════════════════════════════════════════════════════════

def bulk_git_log_authors(linux_dir: Path) -> dict[str, dict[str, int]]:
    """Single git log call: {file_path: {email: commit_count}} for ALL files.

    Much faster than per-file git log — one subprocess instead of thousands.
    """
    try:
        result = subprocess.run(
            ["git", "log", "--name-only",
             "--format=%ae", "--no-merges"],
            capture_output=True,
            text=True,
            cwd=str(linux_dir),
            timeout=300,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return {}

    file_authors: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    current_email: str | None = None

    for line in result.stdout.split("\n"):
        line = line.strip()
        if not line:
            continue
        if "@" in line and "/" not in line and not line.startswith("/"):
            current_email = line
        elif current_email:
            file_authors[line][current_email] += 1

    return dict(file_authors)


# ═══════════════════════════════════════════════════════════════════════════════
# File selection + parsing
# ═══════════════════════════════════════════════════════════════════════════════

def parse_kernel_files(
    linux_dir: Path,
    target_files: int,
) -> Generator[KernelFile, None, None]:
    """Walk filesystem for source files, attribute via git log, yield KernelFile.

    All files are attributed with ALL commit authors (not just top-N).
    No line-level blame — git log --name-only is sufficient for record-level
    provenance: every author listed has at least one commit touching the file.
    """
    linux_dir = Path(linux_dir)
    if not linux_dir.is_dir():
        return

    # Phase 1: single bulk git log for ALL files (one subprocess)
    bulk_data = bulk_git_log_authors(linux_dir)

    # Phase 2: walk filesystem, yield files with attribution
    yielded = 0
    files_without_attrib = 0

    for root, _dirs, fnames in os.walk(linux_dir):
        rel_root = os.path.relpath(root, linux_dir)
        parts = set(rel_root.split("/") if rel_root != "." else [])
        if parts & SKIP_DIRS:
            continue
        if ".git" in parts:
            continue
        for f in sorted(fnames):
            if not f.endswith(_KERNEL_EXTENSIONS):
                continue
            full = os.path.join(root, f)
            rel = os.path.relpath(full, linux_dir)

            try:
                text = Path(full).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if not text.strip():
                continue

            author_counts = bulk_data.get(rel, {})
            if author_counts:
                authors = [{"name": e.split("@")[0], "email": e} for e in author_counts]
            else:
                authors = []
                files_without_attrib += 1

            yielded += 1
            yield KernelFile(
                path=rel,
                text=text,
                authors=authors,
                license="GPL-2.0",
                year="2024",
                line_count=text.count("\n") + (0 if text.endswith("\n") else 1),
            )

            if yielded >= target_files:
                log.info(
                    "parse_kernel_files: %d files, %d without git-log attribution",
                    yielded, files_without_attrib,
                )
                return

    log.info(
        "parse_kernel_files: %d files, %d without git-log attribution",
        yielded, files_without_attrib,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════════════════════════════════════

def count_kernel_files(linux_dir: Path) -> int:
    linux_dir = Path(linux_dir)
    if not linux_dir.is_dir():
        return 0

    count = 0
    for root, _dirs, fnames in os.walk(linux_dir):
        rel_root = os.path.relpath(root, linux_dir)
        parts = set(rel_root.split("/") if rel_root != "." else [])
        if parts & SKIP_DIRS:
            continue
        if ".git" in parts:
            continue
        for f in fnames:
            if f.endswith(_KERNEL_EXTENSIONS):
                count += 1
    return count
