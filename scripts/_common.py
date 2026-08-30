"""Shared CLI plumbing."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xray.config import load_config, merge  # noqa: E402


def base_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config", default=None)
    p.add_argument("--laps", type=int, default=None)
    p.add_argument("--out", default=None)
    return p


def config_from(args) -> dict:
    cfg = load_config(args.config)
    if getattr(args, "laps", None):
        cfg = merge(cfg, sim={"n_laps": args.laps})
    return cfg
