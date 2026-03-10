"""LLM client wrapping vLLM's OpenAI-compatible API."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import OpenAI

from bds.config import LLMConfig

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class ChatResponse:
    content: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    finish_reason: Optional[str] = None

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class LLMClient:
    """Client for vLLM-served EXAONE model using OpenAI SDK."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = OpenAI(
            base_url=config.api_base,
            api_key="not-needed",
        )
        self.model = config.model

    def check_connection(self) -> bool:
        """Check if the vLLM server is reachable and has the model loaded."""
        try:
            models = self.client.models.list()
            available = [m.id for m in models.data]
            if self.model in available:
                logger.info("Connected to vLLM. Model '%s' is available.", self.model)
                return True
            logger.warning(
                "vLLM is up but model '%s' not found. Available: %s",
                self.model, available,
            )
            # Auto-select if only one model is available
            if len(available) == 1:
                self.model = available[0]
                logger.info("Auto-selected model: %s", self.model)
                return True
            return False
        except Exception as exc:
            logger.error("Cannot connect to vLLM at %s: %s", self.config.api_base, exc)
            return False

    def generate(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = False,
        temperature: Optional[float] = None,
        max_tokens: int = 4096,
    ) -> str:
        """Generate a text or JSON completion."""
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        logger.debug("LLM generate: %d messages, json_mode=%s", len(messages), json_mode)
        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        logger.debug("LLM response: %d chars", len(content))
        return content

    def chat_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: int = 4096,
    ) -> ChatResponse:
        """Chat completion with tool use support."""
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens,
        }

        logger.debug("LLM chat_with_tools: %d messages, %d tools", len(messages), len(tools))
        response = self.client.chat.completions.create(**kwargs)
        msg = response.choices[0].message

        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {"raw": tc.function.arguments}
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))

        return ChatResponse(
            content=msg.content,
            tool_calls=tool_calls,
            finish_reason=response.choices[0].finish_reason,
        )
