"""Versioned real-race corpus manifests.

The analysis payload remains the single loader/output of the FastF1 path. This
module records what is already in those payloads, fingerprints it, and writes a
new manifest directory for each corpus build so historical artifacts are never
silently overwritten.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from .regs import regs_for


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fingerprint_json(obj: Any) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode())


def _schema(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _schema(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        if not obj:
            return []
        return {"list_len": len(obj), "item": _schema(obj[0])}
    return type(obj).__name__


def _git_version() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def _fastf1_version() -> str:
    try:
        return metadata.version("fastf1")
    except metadata.PackageNotFoundError:
        return "unavailable"


def _has_any(rows: list[dict], *fields: str) -> bool:
    return any(any(r.get(f) is not None for f in fields) for r in rows)


def record_from_payload(path: Path, config_path: Path | None = None,
                        code_version: str | None = None) -> dict:
    raw = path.read_bytes()
    payload = json.loads(raw)
    laps = list(payload.get("laps", []))
    cars = payload.get("cars", {})
    schema = _schema(payload)
    cfg_raw = config_path.read_bytes() if config_path and config_path.exists() else b""
    geo = payload.get("circuit_geometry", {})
    weather = payload.get("weather", {})
    weather_trace = payload.get("weather_trace", [])
    date = payload.get("date")
    regulation = payload.get("regulation") or {}
    regulation_variant = regulation.get("variant")
    regulation_source = "payload.regulation.variant"
    if regulation_variant is None and date:
        regulation_variant = regs_for(date).variant
        regulation_source = "regs_for(payload date fallback)"
    elif regulation_variant is None:
        regulation_source = "unavailable_missing_payload_regulation_and_date"
    stat = path.stat()
    analysed_at = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()
    return {
        "season": payload.get("year"),
        "event": payload.get("event"),
        "session": payload.get("session"),
        "date": date,
        "drivers": sorted(cars),
        "cars": sorted(cars),
        "fastf1_version": _fastf1_version(),
        "xray_code_version": code_version or _git_version(),
        "config_fingerprint": sha256_bytes(cfg_raw) if cfg_raw else "unavailable",
        "payload_schema": fingerprint_json(schema),
        "payload_schema_shape": schema,
        "regulation_variant": regulation_variant,
        "regulation_variant_source": regulation_source,
        "circuit_metadata_version": fingerprint_json({
            "length": geo.get("length"),
            "has_elevation": geo.get("has_elevation"),
            "zones": geo.get("zones", []),
        }),
        "weather_availability": {
            "summary": bool(weather),
            "trace": bool(weather_trace),
            "n_trace_samples": len(weather_trace),
        },
        "tyre_stint_availability": {
            "compound": _has_any(laps, "compound"),
            "tyre_life": _has_any(laps, "tyre_life"),
            "stint": _has_any(laps, "stint"),
            "fresh_tyre": _has_any(laps, "fresh_tyre"),
        },
        "track_status_availability": _has_any(laps, "track_status", "track_status_s"),
        "pit_availability": _has_any(laps, "pit_in_time_s", "pit_out_time_s"),
        "analysis_timestamp": analysed_at,
        "artifact_path": str(path),
        "artifact_fingerprint": sha256_bytes(raw),
    }


def build_corpus(payload_paths: list[Path], out_root: Path,
                 config_path: Path | None = None,
                 generated_at: datetime | None = None) -> Path:
    generated_at = generated_at or datetime.now(timezone.utc)
    version = generated_at.strftime("%Y%m%dT%H%M%SZ")
    out_dir = out_root / version
    out_dir.mkdir(parents=True, exist_ok=False)
    records = [record_from_payload(Path(p), config_path=config_path)
               for p in sorted(payload_paths)]
    manifest = {
        "corpus_version": version,
        "generated_at": generated_at.isoformat(),
        "n_sessions": len(records),
        "sessions": records,
        "manifest_fingerprint": fingerprint_json(records),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return out_dir
