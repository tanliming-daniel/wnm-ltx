from __future__ import annotations

from pathlib import Path

from .schema import WorldNarrativeConfig
from .simple_yaml import simple_yaml_load


def load_config(path: str | Path) -> WorldNarrativeConfig:
    path = Path(path).expanduser().resolve()
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ImportError:
        raw = simple_yaml_load(text)
    else:
        raw = yaml.safe_load(text) or {}
    return WorldNarrativeConfig.from_mapping(raw)
