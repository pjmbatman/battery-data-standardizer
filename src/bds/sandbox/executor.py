"""Sandbox executor — runs LLM-generated code in a subprocess with timeout."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from bds.config import SandboxConfig


@dataclass
class ExecutionResult:
    success: bool
    output: Optional[Any] = None
    error: Optional[str] = None
    partial_output: Optional[str] = None
    return_code: int = 0


class SandboxExecutor:
    """Execute LLM-generated Python code in a subprocess."""

    def __init__(self, config: SandboxConfig, work_dir: Optional[Path] = None):
        self.config = config
        self.work_dir = work_dir or Path.cwd()

    def execute(self, code: str, timeout: Optional[int] = None) -> ExecutionResult:
        """Run a Python code string in a subprocess and parse JSON output from stdout."""
        timeout = timeout or self.config.timeout

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            prefix="bds_exec_",
            delete=False,
            dir=self.work_dir,
        ) as f:
            f.write(code)
            script_path = Path(f.name)

        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.work_dir),
            )

            if proc.returncode == 0:
                stdout = proc.stdout.strip()
                if not stdout:
                    return ExecutionResult(
                        success=False,
                        error="Script produced no output on stdout",
                        partial_output=proc.stderr,
                        return_code=0,
                    )
                try:
                    output = json.loads(stdout)
                    return ExecutionResult(success=True, output=output, return_code=0)
                except json.JSONDecodeError as e:
                    return ExecutionResult(
                        success=False,
                        error=f"Failed to parse JSON output: {e}",
                        partial_output=stdout[:5000],
                        return_code=0,
                    )
            else:
                return ExecutionResult(
                    success=False,
                    error=proc.stderr[:5000],
                    partial_output=proc.stdout[:5000] if proc.stdout else None,
                    return_code=proc.returncode,
                )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                error=f"Execution timed out after {timeout}s",
                return_code=-1,
            )
        finally:
            script_path.unlink(missing_ok=True)
