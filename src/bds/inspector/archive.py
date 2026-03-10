"""Archive extraction utilities for ZIP, GZ, TAR files."""

from __future__ import annotations

import gzip
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import List


def extract_archive(path: Path, dest: Path | None = None) -> List[Path]:
    """Extract an archive and return list of extracted file paths.

    If *dest* is None, extracts to a temporary directory.
    Returns a list of all extracted file paths (non-directory).
    """
    path = Path(path)
    if dest is None:
        dest = Path(tempfile.mkdtemp(prefix="bds_archive_"))
    else:
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)

    ext = path.suffix.lower()
    name_lower = path.name.lower()

    if ext == ".zip":
        return _extract_zip(path, dest)
    elif name_lower.endswith(".tar.gz") or name_lower.endswith(".tgz"):
        return _extract_tar(path, dest, mode="r:gz")
    elif name_lower.endswith(".tar.bz2"):
        return _extract_tar(path, dest, mode="r:bz2")
    elif name_lower.endswith(".tar"):
        return _extract_tar(path, dest, mode="r:")
    elif ext == ".gz":
        return _extract_gzip(path, dest)
    else:
        raise ValueError(f"Unsupported archive format: {path.name}")


def is_archive(path: Path) -> bool:
    """Check if a file is a known archive format."""
    name = path.name.lower()
    return any(
        name.endswith(ext)
        for ext in (".zip", ".tar.gz", ".tgz", ".tar.bz2", ".tar", ".gz")
    )


def _extract_zip(path: Path, dest: Path) -> List[Path]:
    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(dest)
    return _list_files(dest)


def _extract_tar(path: Path, dest: Path, mode: str) -> List[Path]:
    with tarfile.open(path, mode) as tf:
        tf.extractall(dest, filter="data")
    return _list_files(dest)


def _extract_gzip(path: Path, dest: Path) -> List[Path]:
    stem = path.stem  # e.g., "data.csv" from "data.csv.gz"
    out_path = dest / stem
    with gzip.open(path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    return [out_path]


def _list_files(directory: Path) -> List[Path]:
    """Recursively list all files (not directories) in a directory."""
    return sorted(p for p in directory.rglob("*") if p.is_file())
