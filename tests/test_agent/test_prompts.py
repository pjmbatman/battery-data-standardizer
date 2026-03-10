"""Tests for prompt templates."""

from __future__ import annotations

from bds.agent.prompts import (
    CODE_GENERATION_SYSTEM,
    CODE_GENERATION_USER,
    TARGET_SCHEMA_DESCRIPTION,
    TOOL_USE_SYSTEM,
)


class TestPromptTemplates:
    def test_code_generation_system_has_schema_placeholder(self):
        rendered = CODE_GENERATION_SYSTEM.format(schema=TARGET_SCHEMA_DESCRIPTION)
        assert "voltage_v" in rendered
        assert "current_a" in rendered
        assert "cycle_number" in rendered

    def test_code_generation_user_has_placeholders(self):
        rendered = CODE_GENERATION_USER.format(
            file_path="/data/test.csv",
            file_preview="col1,col2\n1,2\n",
        )
        assert "/data/test.csv" in rendered
        assert "col1,col2" in rendered

    def test_tool_use_system_has_tools(self):
        rendered = TOOL_USE_SYSTEM.format(schema=TARGET_SCHEMA_DESCRIPTION)
        assert "inspect" in rendered
        assert "read_sample" in rendered
        assert "extract" in rendered
