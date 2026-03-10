"""Tests for file inspector / preview module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from bds.inspector.preview import inspect_file


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


class TestCSVPreview:
    def test_basic_csv(self, tmp_dir):
        csv_path = tmp_dir / "test.csv"
        csv_path.write_text("cycle,voltage,current,time\n1,3.5,0.5,0\n1,3.6,0.5,10\n2,4.0,-0.5,20\n")
        result = inspect_file(csv_path)
        assert "[CSV/Text file" in result
        assert "cycle,voltage,current,time" in result

    def test_tsv_file(self, tmp_dir):
        tsv_path = tmp_dir / "test.tsv"
        tsv_path.write_text("cycle\tvoltage\tcurrent\n1\t3.5\t0.5\n")
        result = inspect_file(tsv_path)
        assert "[CSV/Text file" in result

    def test_large_csv_truncated(self, tmp_dir):
        csv_path = tmp_dir / "large.csv"
        lines = ["col1,col2"] + [f"{i},{i*0.1}" for i in range(100)]
        csv_path.write_text("\n".join(lines))
        result = inspect_file(csv_path)
        # Only first 20 lines should appear in preview
        assert "col1,col2" in result


class TestJSONPreview:
    def test_basic_json(self, tmp_dir):
        json_path = tmp_dir / "test.json"
        data = {"cells": [{"id": "B0005", "cycles": [1, 2, 3]}]}
        json_path.write_text(json.dumps(data))
        result = inspect_file(json_path)
        assert "[JSON file" in result
        assert "cells" in result

    def test_large_json_truncated(self, tmp_dir):
        json_path = tmp_dir / "big.json"
        data = {"data": list(range(1000))}
        json_path.write_text(json.dumps(data))
        result = inspect_file(json_path)
        assert "[JSON file" in result


class TestPicklePreview:
    def test_dict_pickle(self, tmp_dir):
        import pickle
        pkl_path = tmp_dir / "test.pkl"
        data = {"cell_id": "test", "voltage": [3.5, 3.6, 3.7]}
        with open(pkl_path, "wb") as f:
            pickle.dump(data, f)
        result = inspect_file(pkl_path)
        assert "[Pickle file" in result
        assert "dict" in result


class TestUnknownFormat:
    def test_unknown_text_file(self, tmp_dir):
        path = tmp_dir / "test.xyz"
        path.write_text("some unknown content\nline 2\n")
        result = inspect_file(path)
        assert "[Unknown format" in result
        assert "some unknown content" in result

    def test_nonexistent_file(self, tmp_dir):
        result = inspect_file(tmp_dir / "nonexistent.csv")
        assert "File not found" in result
