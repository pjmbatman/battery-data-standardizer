"""Configuration loading and management."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class LLMConfig:
    api_base: str = "http://localhost:8000/v1"
    model: str = "LGAI-EXAONE/K-EXAONE-236B-A23B"
    temperature: float = 0.1
    enable_thinking: bool = True
    max_retries: int = 3


@dataclass
class SandboxConfig:
    timeout: int = 120
    allowed_imports: List[str] = field(
        default_factory=lambda: [
            "pandas", "numpy", "scipy", "h5py", "openpyxl", "json", "pickle",
        ]
    )


@dataclass
class AgentConfig:
    max_tool_steps: int = 20
    confidence_threshold: float = 0.7
    fallback_to_tool_use: bool = True


@dataclass
class CacheConfig:
    enabled: bool = True
    db_path: str = ".bds_cache/cache.db"


@dataclass
class ExportConfig:
    format: str = "cellrecord"


@dataclass
class BDSConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    export: ExportConfig = field(default_factory=ExportConfig)


def _merge_dataclass(dc: object, overrides: dict) -> None:
    """Recursively merge a dict of overrides into a dataclass instance."""
    for key, value in overrides.items():
        if not hasattr(dc, key):
            continue
        current = getattr(dc, key)
        if isinstance(value, dict) and hasattr(current, "__dataclass_fields__"):
            _merge_dataclass(current, value)
        else:
            setattr(dc, key, value)


def load_config(
    config_path: Optional[str | Path] = None,
    overrides: Optional[dict] = None,
) -> BDSConfig:
    """Load configuration from YAML file with optional overrides.

    Resolution order:
    1. Built-in defaults (dataclass defaults)
    2. YAML file values
    3. Programmatic overrides
    """
    cfg = BDSConfig()

    if config_path is not None:
        path = Path(config_path)
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            _merge_dataclass(cfg, data)

    if overrides:
        _merge_dataclass(cfg, overrides)

    return cfg


_DEFAULT_CONFIG_SEARCH_PATHS = [
    Path("bds.yaml"),
    Path("configs/default.yaml"),
    Path.home() / ".config" / "bds" / "config.yaml",
]


def find_config() -> Optional[Path]:
    """Search common locations for a config file."""
    for p in _DEFAULT_CONFIG_SEARCH_PATHS:
        if p.exists():
            return p
    return None
