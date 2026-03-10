"""Agent orchestrator — coordinates code generation and tool-use agents."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional

from bds.agent.code_generator import CodeGenerationAgent
from bds.agent.llm_client import LLMClient
from bds.agent.tool_use import ToolUseAgent
from bds.cache import MappingCache
from bds.config import BDSConfig
from bds.inspector.preview import inspect_file
from bds.sandbox.executor import SandboxExecutor
from bds.validator import Validator

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Top-level orchestrator: preview → code-gen → tool-use → validate."""

    def __init__(self, config: BDSConfig):
        self.config = config
        self.llm = LLMClient(config.llm)
        self.sandbox = SandboxExecutor(config.sandbox)
        self.code_agent = CodeGenerationAgent(self.llm, self.sandbox, config)
        self.tool_agent = ToolUseAgent(self.llm, self.sandbox, config)
        self.cache = MappingCache(config.cache)
        self.validator = Validator()

    def standardize(self, file_path: Path) -> Optional[dict]:
        """Standardize a single file into the target schema.

        Returns the extracted data dict, or None on failure.
        """
        file_path = Path(file_path).resolve()
        logger.info("Standardizing: %s", file_path)

        # 0. Cache check
        if self.config.cache.enabled:
            cached_code = self.cache.get(file_path)
            if cached_code is not None:
                logger.info("Cache hit — replaying cached extraction code")
                adapted_code = _adapt_cached_code(cached_code, file_path)
                result = self.sandbox.execute(adapted_code)
                if result.success:
                    data = result.output
                    # Ensure cell_id matches the current file
                    if isinstance(data, dict):
                        data["cell_id"] = file_path.stem
                    validation = self.validator.validate(data)
                    if validation.is_valid:
                        return data
                logger.warning("Cached code failed, proceeding with fresh extraction")

        # 1. File structure preview
        preview = inspect_file(file_path)
        logger.debug("Preview length: %d chars", len(preview))

        # 2. Code generation agent (primary)
        result = self.code_agent.run(file_path, preview)

        # 3. Tool-use agent (fallback)
        if result is None and self.config.agent.fallback_to_tool_use:
            logger.info("Code generation failed, falling back to tool-use agent")
            result = self.tool_agent.run(file_path, preview)

        # 4. Cache successful extraction
        if result is not None and self.config.cache.enabled:
            code = self.code_agent.last_code
            if code:
                self.cache.save(file_path, code)

        return result

    def standardize_batch(self, file_paths: List[Path]) -> List[Optional[dict]]:
        """Standardize multiple files."""
        results = []
        for fp in file_paths:
            try:
                result = self.standardize(fp)
                results.append(result)
            except Exception as exc:
                logger.error("Failed to standardize %s: %s", fp, exc)
                results.append(None)
        return results


def _adapt_cached_code(code: str, new_file_path: Path) -> str:
    """Replace the hardcoded FILE_PATH in cached code with the new file path."""
    new_path_str = str(new_file_path)
    # Match FILE_PATH = "..." or FILE_PATH = '...'
    code = re.sub(
        r'''(FILE_PATH\s*=\s*)(["']).*?\2''',
        rf'\1"{new_path_str}"',
        code,
    )
    # Also handle: file_path = "..." (lowercase variant some LLMs use)
    code = re.sub(
        r'''(file_path\s*=\s*)(["']).*?\2''',
        rf'\1"{new_path_str}"',
        code,
    )
    return code
