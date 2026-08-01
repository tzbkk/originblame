"""Compatibility layer for pipeline v2 — uses _ob_native (PyO3) for performance.

The benchmark pipeline always has the Rust build available (_ob_native.so or
installed ob package). This is separate from the public Python API which uses
subprocess fallback for easy installation without maturin.
"""

from __future__ import annotations

import os
from pathlib import Path

from _ob_native import (
    compute_hash,
    index_document,
    register_section,
    token_index_write_pid,
)


def section_add_raw(
    path: str,
    author_ids: list[str],
    contributor_ids: list[str],
    license: str,
    year: str,
    ob_dir_str: str,
) -> str:
    return register_section(path, author_ids, contributor_ids, license, year, ob_dir_str)


def track_hashed(
    ob_dir_str: str,
    line_hash: str,
    file: str,
    section_hashes: list[str],
    token_count: int,
    tokenizer_name: str,
) -> None:
    index_document(ob_dir_str, line_hash, file, section_hashes)
    token_index_write_pid(ob_dir_str, tokenizer_name, token_count, section_hashes)


def clean(ob_dir_str: str) -> None:
    from _ob_native import clean as _clean

    _clean(ob_dir_str, False)
