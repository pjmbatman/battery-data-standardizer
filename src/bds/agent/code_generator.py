"""Code generation agent — LLM generates a complete extraction script."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from bds.agent.llm_client import LLMClient
from bds.agent.prompts import (
    CODE_FIX_USER,
    CODE_GENERATION_SYSTEM,
    CODE_GENERATION_USER,
    TARGET_SCHEMA_DESCRIPTION,
    VALIDATION_FIX_USER,
)
from bds.config import BDSConfig
from bds.sandbox.executor import SandboxExecutor, ExecutionResult
from bds.validator import ValidationResult, Validator

logger = logging.getLogger(__name__)


class CodeGenerationAgent:
    """First-pass agent: ask LLM to generate a full extraction script."""

    def __init__(self, llm: LLMClient, sandbox: SandboxExecutor, config: BDSConfig):
        self.llm = llm
        self.sandbox = sandbox
        self.config = config
        self.validator = Validator()
        self.last_code: Optional[str] = None

    def run(self, file_path: Path, file_preview: str) -> Optional[dict]:
        """Generate extraction code, execute, validate, retry on failure.

        Returns the extracted data dict on success, None on failure.
        """
        max_retries = self.config.llm.max_retries
        code = self._generate_code(file_path, file_preview)
        self.last_code = code

        for attempt in range(max_retries):
            logger.info("Code generation attempt %d/%d", attempt + 1, max_retries)

            result = self.sandbox.execute(code)

            if not result.success:
                logger.warning("Execution failed: %s", (result.error or "")[:500])
                if result.partial_output:
                    logger.warning("Partial output: %s", result.partial_output[:300])
                if attempt < max_retries - 1:
                    code = self._fix_code(
                        code, file_path, file_preview, result.error, result.partial_output
                    )
                    self.last_code = code
                continue

            # Auto-fix: inject cell_id from filename if missing
            data = result.output
            if isinstance(data, dict) and "cell_id" not in data:
                data["cell_id"] = file_path.stem

            validation = self.validator.validate(data)

            if validation.is_valid:
                logger.info("Extraction succeeded with %d cycles", len(data.get("cycles", [])))
                return data

            # Validation failed — ask LLM to fix
            logger.warning("Validation issues: %s", validation.issues[:3])
            if attempt < max_retries - 1:
                code = self._fix_validation(
                    code, file_path, file_preview, data, validation
                )
                self.last_code = code

        logger.warning("Code generation agent exhausted retries")
        return None

    def _generate_code(self, file_path: Path, file_preview: str) -> str:
        """Ask LLM to generate the initial extraction script."""
        system_msg = CODE_GENERATION_SYSTEM.format(schema=TARGET_SCHEMA_DESCRIPTION)
        user_msg = CODE_GENERATION_USER.format(
            file_path=str(file_path),
            file_preview=file_preview[:8000],
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        raw = self.llm.generate(messages)
        return _strip_code_fences(raw)

    def _fix_code(
        self,
        original_code: str,
        file_path: Path,
        file_preview: str,
        error: str,
        partial_output: Optional[str],
    ) -> str:
        """Ask LLM to fix code that failed to execute."""
        system_msg = CODE_GENERATION_SYSTEM.format(schema=TARGET_SCHEMA_DESCRIPTION)
        user_msg = CODE_FIX_USER.format(
            error=error[:3000],
            partial_output=(partial_output or "")[:2000],
            original_code=original_code,
            file_path=str(file_path),
            file_preview=file_preview[:4000],
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        raw = self.llm.generate(messages)
        return _strip_code_fences(raw)

    def _fix_validation(
        self,
        original_code: str,
        file_path: Path,
        file_preview: str,
        data: dict,
        validation: ValidationResult,
    ) -> str:
        """Ask LLM to fix code whose output failed validation."""
        system_msg = CODE_GENERATION_SYSTEM.format(schema=TARGET_SCHEMA_DESCRIPTION)

        cycles = data.get("cycles", [])
        sample_keys = list(cycles[0].keys()) if cycles else []
        sample_cycle = json.dumps(
            {k: (v[:3] if isinstance(v, list) else v) for k, v in (cycles[0].items() if cycles else {})},
            default=str,
        )

        user_msg = VALIDATION_FIX_USER.format(
            issues="\n".join(f"- {i}" for i in validation.issues),
            num_cycles=len(cycles),
            sample_keys=sample_keys,
            sample_data=sample_cycle[:2000],
            original_code=original_code,
            file_path=str(file_path),
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        raw = self.llm.generate(messages)
        return _strip_code_fences(raw)


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences from LLM output."""
    text = text.strip()
    if text.startswith("```python"):
        text = text[len("```python"):]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()
