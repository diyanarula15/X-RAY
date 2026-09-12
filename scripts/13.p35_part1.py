#!/usr/bin/env python3
"""P3.5 Part 1: assemble the estimator-diagnosis artifact.

Reads the measurements produced by the P3.5 modules and writes one fingerprinted
summary. Computes nothing itself beyond aggregation -- every number comes from
`xray.reinfer`, `xray.nuisance`, `xray.estimator_diag` or `xray.mismatch`, so the
artifact cannot disagree with the code that produced it.
"""
from __future__ import annotations

import datetime as _dt
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xray.estimator_diag import (DIAG_VERSION, CDA_ADMISSIBLE_HI,  # noqa: E402
                                CDA_ADMISSIBLE_LO, CDA_ADMISSIBLE_NOTE,
                                MATERIAL_ENERGY_J, SIMILAR_FIT_REL,
                                floor_pinning, label_identifiability)
from xray.nuisance import REGISTRY_VERSION, nuisance_registry  # noqa: E402
from xray.reinfer import SCHEMA_VERSION, fingerprint  # noqa: E402

OUT = Path("out/p35")


def _mean(rows, path):
    vals = []
    for r in rows:
        v = r
        for k in path:
            v = (v or {}).get(k) if isinstance(v, dict) else None
        if v is not None and np.isfinite(v):
            vals.append(float(v))
    return float(np.mean(vals)) if vals else None


def production_floor_pinning() -> dict:
    """The pathology, measured on shipped payloads rather than on a rerun."""
    per_race = {}
    for p in sorted(glob.glob("out/races/*.json")):
        with open(p) as fh:
            d = json.load(fh)
        rows = []
        for car in d.get("cars", {}).values():
            tr = car.get("trace") or {}
            if not tr.get("usable_mean"):
                continue
            toJ = lambda a: [None if x is None else x * 1e6 for x in a]
            try:
                r = floor_pinning(toJ(tr["usable_mean"]), toJ(tr["usable_p10"]),
                                  toJ(tr["usable_p90"]))
            except ValueError:
                continue
            if r.get("available"):
                rows.append(r)
        if rows:
            per_race[d.get("id", Path(p).stem)] = {
                "n_cars": len(rows),
                "frac_reported_empty": _mean(rows, ["frac_reported_empty"]),
                "frac_band_degenerate": _mean(rows, ["frac_band_degenerate"]),
                "frac_empty_and_degenerate": _mean(rows, ["frac_empty_and_degenerate"]),
                "mean_width_mj": _mean(rows, ["mean_width_mj"]),
                "mean_usable_mj": _mean(rows, ["mean_usable_mj"])}
    allr = list(per_race.values())
    return {"per_race": per_race,
            "n_races": len(per_race),
            "n_car_races": int(sum(r["n_cars"] for r in allr)),
            "frac_reported_empty": _mean(allr, ["frac_reported_empty"]),
            "frac_band_degenerate": _mean(allr, ["frac_band_degenerate"]),
            "frac_empty_and_degenerate": _mean(allr, ["frac_empty_and_degenerate"]),
            "mean_width_mj": _mean(allr, ["mean_width_mj"]),
            "mean_usable_mj": _mean(allr, ["mean_usable_mj"])}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    diags = json.loads(Path(OUT / "diagnostics.json").read_text()) \
        if (OUT / "diagnostics.json").exists() else []
    sens_files = sorted(glob.glob(str(OUT / "sens_r*.json")))
    sens = [json.loads(Path(f).read_text()) for f in sens_files]
    mm = json.loads((OUT / "mismatch.json").read_text()) \
        if (OUT / "mismatch.json").exists() else {}

    sat = [d["saturation"] for d in diags if d.get("saturation", {}).get("available")]
    saturation = {
        "n_windows": len(sat),
        "frac_upper_edge_at_ceiling": _mean(sat, ["frac_upper_edge_at_ceiling"]),
        "frac_censored": _mean(sat, ["frac_censored"]),
        "frac_band_pinned_at_ceiling": _mean(sat, ["frac_band_pinned_at_ceiling"]),
        "mean_centre_below_ceiling_kw": _mean(sat, ["mean_centre_below_ceiling_kw"]),
        "clipped_fraction_of_band_energy": _mean(sat, ["clipped_fraction_of_band_energy"]),
        "current_observation_semantics": "latent = ceiling (exact)",
        "implied_correct_semantics": "latent <= ceiling (censored from above)",
    }

    from xray.estimator_diag import IdentifiabilityMetrics
    ident_rows = []
    for d in diags:
        m = d.get("identifiability") or {}
        # Labels are RECOMPUTED from the stored metrics with the current
        # `label_identifiability`, never copied from the run that produced them.
        # The stored labels predate the degeneracy guard, and an artifact carrying
        # a label its own code would no longer assign is worse than no label.
        try:
            lab = label_identifiability(IdentifiabilityMetrics(**m))
        except TypeError:
            lab = d.get("label") or {}
        ident_rows.append({"window_id": d["window_id"],
                           "label_recomputed": lab.get("label"),
                           "label_reason": lab.get("reason"),
                           "degenerate": lab.get("degenerate"),
                           "variance_ratio": m.get("variance_ratio"),
                           "energy_particle_width_mj": (m.get("energy_particle_width_j") or 0) / 1e6,
                           "between_nuisance_sd_mj": (m.get("between_nuisance_sd_j") or 0) / 1e6,
                           "cda_set_width": m.get("cda_set_width"),
                           "cda_admissible_overlap": m.get("cda_admissible_overlap"),
                           "profile_width_mj": (d.get("profile") or {}).get("profile_width_mj")})

    ranking = sens[0]["ranking"] if sens else []
    summary = {
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "versions": {"window_schema": SCHEMA_VERSION,
                     "nuisance_registry": REGISTRY_VERSION,
                     "diagnostics": DIAG_VERSION,
                     "mismatch": mm.get("mismatch_version")},
        "estimator_state_semantics": {
            "cda": "identified SET from regulation-derived per-sample intervals; "
                   "midpoint is a reporting convenience, NOT a posterior mean",
            "deployment": "identified per-sample BAND; centre selected by the "
                          "race-long energy-balance closure, not by a likelihood",
            "stored_energy": "particle belief; p10/p90 are particle QUANTILES and "
                             "are not calibrated credible intervals",
        },
        "windows": {"n": len(diags),
                    "ids": [d["window_id"] for d in diags]},
        "nuisance_registry": nuisance_registry(),
        "nuisance_ranking": ranking,
        "saturation": saturation,
        "identifiability": {
            "definition": {
                "variance_ratio": "between-nuisance sd of inferred energy divided "
                                  "by the particle band's 1-sigma equivalent "
                                  "(p10-p90 / 2.563)",
                "material_energy_mj": MATERIAL_ENERGY_J / 1e6,
                "similar_fit_rel": SIMILAR_FIT_REL,
                "degenerate_width_mj": 0.01,
                "cda_admissible_domain": [CDA_ADMISSIBLE_LO, CDA_ADMISSIBLE_HI],
                "cda_admissible_note": CDA_ADMISSIBLE_NOTE,
            },
            "windows": ident_rows,
        },
        "production_floor_pinning": production_floor_pinning(),
        "synthetic_mismatch": {
            "cases": [{"case": c["case"], "note": c["note"], "patch": c["patch"],
                       **(c["aggregate"] or {})} for c in mm.get("cases", [])],
            "verdict": mm.get("verdict", {}),
        },
    }
    summary["summary_fingerprint"] = fingerprint(summary)
    stamp = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    d = OUT / "part1" / stamp
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"wrote {d / 'summary.json'}")
    print(f"fingerprint {summary['summary_fingerprint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
