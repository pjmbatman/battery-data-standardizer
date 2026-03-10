"""Tests for archive extraction."""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

import pytest

from bds.inspector.archive import extract_archive, is_archive


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


class TestIsArchive:
    def test_zip(self):
        assert is_archive(Path("data.zip"))

    def test_tar_gz(self):
        assert is_archive(Path("data.tar.gz"))

    def test_csv_not_archive(self):
        assert not is_archive(Path("data.csv"))


class TestExtractZip:
    def test_extract_zip(self, tmp_dir):
        # Create a zip file with a CSV inside
        csv_content = "col1,col2\n1,2\n3,4\n"
        zip_path = tmp_dir / "test.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("data.csv", csv_content)

        dest = tmp_dir / "extracted"
        files = extract_archive(zip_path, dest)
        assert len(files) == 1
        assert files[0].name == "data.csv"
        assert files[0].read_text() == csv_content
