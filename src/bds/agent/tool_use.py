"""Tool-use agent — fallback mode where LLM iteratively explores and extracts data."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from bds.agent.llm_client import LLMClient
from bds.agent.prompts import (
    TARGET_SCHEMA_DESCRIPTION,
    TOOL_USE_SYSTEM,
    TOOL_USE_USER,
)
from bds.agent.tools import TOOL_SCHEMAS, ToolExecutor
from bds.config import BDSConfig
from bds.sandbox.executor import SandboxExecutor

logger = logging.getLogger(__name__)

# Rough estimate: 1 token ≈ 4 chars. Keep well under 32K context.
MAX_CONTEXT_CHARS = 80000  # ~20K tokens, leaving room for output


class ToolUseAgent:
    """Fallback agent: uses tool calls to iteratively explore and extract data."""

    def __init__(self, llm: LLMClient, sandbox: SandboxExecutor, config: BDSConfig):
        self.llm = llm
        self.config = config
        self.tool_executor = ToolExecutor(sandbox_executor=sandbox)

    def run(self, file_path: Path, file_preview: str) -> Optional[dict]:
        """Run the tool-use agent loop."""
        max_steps = self.config.agent.max_tool_steps

        system_msg = TOOL_USE_SYSTEM.format(schema=TARGET_SCHEMA_DESCRIPTION)
        user_msg = TOOL_USE_USER.format(
            file_path=str(file_path),
            file_preview=file_preview[:4000],
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        for step in range(max_steps):
            logger.info("Tool-use step %d/%d", step + 1, max_steps)

            # Trim context if too large
            messages = _trim_context(messages, MAX_CONTEXT_CHARS)

            try:
                response = self.llm.chat_with_tools(messages, TOOL_SCHEMAS)
            except Exception as exc:
                logger.warning("LLM call failed at step %d: %s", step + 1, exc)
                # Try trimming harder
                messages = _trim_context(messages, MAX_CONTEXT_CHARS // 2)
                try:
                    response = self.llm.chat_with_tools(messages, TOOL_SCHEMAS)
                except Exception:
                    logger.error("LLM call failed even after trimming")
                    return None

            if response.has_tool_calls:
                assistant_msg = {"role": "assistant", "content": response.content or ""}
                tool_calls_raw = []
                for tc in response.tool_calls:
                    tool_calls_raw.append({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    })
                assistant_msg["tool_calls"] = tool_calls_raw
                messages.append(assistant_msg)

                for tc in response.tool_calls:
                    logger.debug("Tool call: %s(%s)", tc.name, tc.arguments)
                    result_str = self.tool_executor.execute(tc.name, tc.arguments)
                    # Aggressively truncate tool results
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_str[:2000],
                    })
            else:
                content = response.content or ""
                messages.append({"role": "assistant", "content": content})

                data = _try_parse_json(content)
                if data is not None:
                    logger.info(
                        "Tool-use agent produced result with %d cycles",
                        len(data.get("cycles", [])),
                    )
                    return data
                else:
                    logger.warning("Tool-use agent returned non-JSON content, prompting again")
                    messages.append({
                        "role": "user",
                        "content": (
                            "Your response was not valid JSON. Please use the execute_code tool "
                            "to write a Python script that extracts the data and prints JSON to stdout. "
                            "The code should output the complete target schema JSON."
                        ),
                    })

        logger.warning("Tool-use agent exhausted max steps (%d)", max_steps)
        return None


def _trim_context(messages: list, max_chars: int) -> list:
    """Remove old tool results if context is too large."""
    total = sum(len(m.get("content", "") or "") for m in messages)
    if total <= max_chars:
        return messages

    # Keep system + first user + last 6 messages, trim middle
    keep_start = 2  # system + user
    keep_end = 6
    if len(messages) <= keep_start + keep_end:
        # Just truncate content of tool messages
        trimmed = []
        for m in messages:
            if m.get("role") == "tool" and len(m.get("content", "")) > 500:
                m = {**m, "content": m["content"][:500] + "\n[truncated]"}
            trimmed.append(m)
        return trimmed

    result = messages[:keep_start] + messages[-(keep_end):]
    # Add a note about what was removed
    n_removed = len(messages) - keep_start - keep_end
    result.insert(keep_start, {
        "role": "user",
        "content": f"[{n_removed} earlier tool interactions removed to save context]",
    })
    return result


def _try_parse_json(text: str) -> Optional[dict]:
    """Try to extract a JSON object from text."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[len("```json"):]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        pass
                    break
    return None
