"""Writes out/release_manifest.json: what shipped.

A judge, a bug report, or a future session needs to be able to pin a claim to
an exact commit, estimator state and model fingerprint without re-deriving any
of them. This reads existing sources of truth -- it does not duplicate the
P3/P3.5 validation data, and it does not compute anything new.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          cwd=ROOT).stdout.strip()


def _api_version() -> str:
    """Read the version FastAPI already declares, rather than restating it."""
    src = (ROOT / "simulation" / "api" / "main.py").read_text(encoding="utf-8")
    m = re.search(r'FastAPI\([^)]*version="([^"]+)"', src)
    return m.group(1) if m else "unknown"


def _model_fingerprint() -> str:
    """A cheap hash over the sources whose drift would change a live decision:
    the (still-invented) pass-model coefficients, the tyre compound table, and
    the estimator-activation registry -- not a duplicate of the P3 validation
    datasets themselves."""
    from xray.overtake import COEFFS
    from xray.registry import registry_payload
    import yaml

    tyres = (ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    compounds = yaml.safe_load(tyres).get("tyres", {}).get("compounds", {})
    blob = json.dumps({
        "pass_coeffs": asdict(COEFFS),
        "tyre_compounds": compounds,
        "energy_inference_status": registry_payload()["energy_inference"]["status"],
    }, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def main() -> None:
    from xray.registry import registry_payload

    manifest = {
        "commit": _git("rev-parse", "HEAD"),
        "dirty_worktree": bool(_git("status", "--porcelain")),
        "estimator_status": registry_payload()["energy_inference"],
        "frontend_build_present": (ROOT / "simulation" / "app" / "dist" / "index.html").exists(),
        "api_version": _api_version(),
        "model_fingerprint": _model_fingerprint(),
        "registry_version": registry_payload()["registry_version"],
    }
    out = ROOT / "out" / "release_manifest.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
