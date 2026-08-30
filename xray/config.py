"""Config loading. The estimator never touches this module."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path is not None else DEFAULT_PATH
    with open(path) as fh:
        return yaml.safe_load(fh)


def merge(cfg: dict, **overrides) -> dict:
    """Shallow-nested override helper: merge(cfg, sim={'n_laps': 3})."""
    out = copy.deepcopy(cfg)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = {**out[key], **value}
        else:
            out[key] = value
    return out
