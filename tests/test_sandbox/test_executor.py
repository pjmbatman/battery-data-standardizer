"""Tests for sandbox executor."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from bds.config import SandboxConfig
from bds.sandbox.executor import SandboxExecutor


@pytest.fixture
def sandbox():
    with tempfile.TemporaryDirectory() as d:
        config = SandboxConfig(timeout=10)
        yield SandboxExecutor(config, work_dir=Path(d))


class TestSandboxExecutor:
    def test_simple_json_output(self, sandbox):
        code = 'import json; print(json.dumps({"hello": "world"}))'
        result = sandbox.execute(code)
        assert result.success
        assert result.output == {"hello": "world"}

    def test_execution_error(self, sandbox):
        code = "raise ValueError('test error')"
        result = sandbox.execute(code)
        assert not result.success
        assert "ValueError" in result.error

    def test_timeout(self, sandbox):
        code = "import time; time.sleep(30)"
        result = sandbox.execute(code, timeout=2)
        assert not result.success
        assert "timed out" in result.error

    def test_no_output(self, sandbox):
        code = "x = 42"
        result = sandbox.execute(code)
        assert not result.success
        assert "no output" in result.error.lower()

    def test_invalid_json_output(self, sandbox):
        code = 'print("not json")'
        result = sandbox.execute(code)
        assert not result.success
        assert "JSON" in result.error

    def test_numpy_computation(self, sandbox):
        code = '''
import json
import numpy as np
arr = np.array([1.0, 2.0, 3.0])
print(json.dumps({"mean": float(arr.mean()), "values": arr.tolist()}))
'''
        result = sandbox.execute(code)
        assert result.success
        assert result.output["mean"] == 2.0
        assert result.output["values"] == [1.0, 2.0, 3.0]
