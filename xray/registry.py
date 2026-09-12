"""The canonical model-status registry: what each model IS, and what is known about it.

P3's reason for existing. Every earlier phase carried its claim strength in prose
-- a comment here, a metadata string there, an `ASSUMED_` prefix somewhere else --
and prose does not reach the UI. So the frontend rendered a synthetic logistic and
an integrated force balance in the same typeface, and the only thing separating
"measured" from "invented" was whether a reader happened to know.

Two rules shape this file.

First, **a status is not a score.** `INFERRED` says the number is produced by
inference; it says nothing about whether that inference works. The validation
record says that, separately, and it is allowed to say NO. There is no single word
for "energy inference" that is both honest and short, which is why there is no
`VALIDATED` status here to reach for.

Second, **nothing is restated.** Metrics are read from the P3 artifacts on disk
and carry their fingerprint, so a registry entry cannot drift away from the run it
describes. When an artifact is missing the entry says so rather than falling back
to a remembered number -- a registry quoting numbers nobody can trace is worse
than one that declines.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from typing import Any

REGISTRY_VERSION = "p3-model-registry-v1"

# ------------------------------------------------------------------- statuses
# What KIND of thing a number is. Ordered loosely from "known by rule" to
# "not available at all". None of these is a quality claim.
REGULATION = "REGULATION"        # in the rulebook; not measured, not fitted
PHYSICS = "PHYSICS"              # derived from force/energy balance
SYNTHETIC = "SYNTHETIC"          # designed coefficients, not fitted to data
HEURISTIC = "HEURISTIC"          # a rule of thumb, neither fitted nor physical
INFERRED = "INFERRED"            # produced by inference from observed data
CALIBRATED = "CALIBRATED"        # fitted to data, validation not yet established
EMPIRICAL = "EMPIRICAL"          # fitted to data AND validated out of sample
DISABLED = "DISABLED"            # implemented, deliberately not in the path
UNAVAILABLE = "UNAVAILABLE"      # cannot be produced from the data we have
RESEARCH_ONLY = "RESEARCH_ONLY"  # real code, real tests, never reaches a race

STATUSES = (REGULATION, PHYSICS, SYNTHETIC, HEURISTIC, INFERRED, CALIBRATED,
            EMPIRICAL, DISABLED, UNAVAILABLE, RESEARCH_ONLY)

# ----------------------------------------------------------- validation verdicts
NOT_ATTEMPTED = "NOT_ATTEMPTED"
MIXED = "MIXED"
NO_ROBUST_IMPROVEMENT = "NO_ROBUST_IMPROVEMENT"
NOT_DEMONSTRATED = "NOT_DEMONSTRATED"
REFUSED = "REFUSED"
PASSED = "PASSED"

# Identifiability, as a word rather than a number, because the number alone
# ("0.0") reads like a missing value.
IDENT_LIMITED = "LIMITED"
IDENT_NONE = "NONE"
IDENT_GOOD = "GOOD"


@dataclass(frozen=True)
class Validation:
    """What was checked, how, and what came out -- including a negative result.

    `result` is deliberately separate from `available`. An entry that ran a real
    held-out comparison and lost says `available=True, result=NO_ROBUST_
    IMPROVEMENT`; one that was never checked says `available=False,
    result=NOT_ATTEMPTED`. Collapsing those two into a single falsy value is how a
    system ends up unable to distinguish "we looked and it failed" from "we never
    looked", which are opposite engineering situations.
    """
    available: bool
    kind: str                      # direct_truth | indirect_predictive | none
    result: str
    metrics: dict = field(default_factory=dict)
    artifact: str | None = None    # path of the artifact the metrics came from
    fingerprint: str | None = None
    provenance: str | None = None  # how train/test were split
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelEntry:
    name: str
    component: str
    status: str
    production: bool               # does this reach a real race?
    version: str | None = None
    validation: Validation | None = None
    identifiability: str | None = None
    reason: str | None = None      # why unavailable / disabled / research-only
    notes: tuple[str, ...] = ()

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(f"{self.component}: unknown status {self.status!r}")


# ------------------------------------------------------------------- artifacts
def _artifact_root() -> str:
    return os.environ.get("XRAY_P3_ARTIFACTS", os.path.join("out", "p3"))


def _latest(kind: str) -> tuple[dict | None, str | None]:
    """Newest artifact of a kind, or (None, None). Never a remembered number."""
    paths = sorted(glob.glob(os.path.join(_artifact_root(), kind, "*", "summary.json")))
    if not paths:
        return None, None
    try:
        with open(paths[-1]) as fh:
            return json.load(fh), paths[-1]
    except (OSError, json.JSONDecodeError):
        return None, None


def _mean_over_runs(runs: list[dict], group: str, key: str) -> float | None:
    vals = [r[group][key] for r in runs if group in r and key in r[group]]
    return None if not vals else float(sum(vals) / len(vals))


def _range_over_runs(runs: list[dict], group: str, key: str):
    vals = [r[group][key] for r in runs if group in r and key in r[group]]
    return None if not vals else [float(min(vals)), float(max(vals))]


def _synthetic_validation() -> Validation:
    """Part 1: the canonical path scored against hidden simulator truth.

    Reported as the MEAN over the seeds, with a range where the spread is the
    point. SOC band containment runs 35.7-55.0% across three seeds; quoting only
    the mean (43.8%) would hide that the interval is not merely too narrow but
    inconsistently so, which is the actual finding.
    """
    d, path = _latest("part1")
    if d is None:
        return Validation(False, "direct_truth", NOT_ATTEMPTED,
                          notes=("no Part 1 artifact on disk",))
    runs = d.get("synthetic_direct_validation") or []
    if not runs:
        return Validation(False, "direct_truth", NOT_ATTEMPTED,
                          artifact=path, notes=("artifact has no runs",))
    m = {
        "n_seeds": len(runs),
        "cda_truth_contained": sum(bool(r["cda"].get("truth_contained")) for r in runs),
        "cda_width_pct_of_truth": _mean_over_runs(runs, "cda", "width_pct_of_truth"),
        "cda_identifiability": _mean_over_runs(runs, "cda", "identifiability"),
        "deployment_band_containment": _mean_over_runs(
            runs, "deployment", "sample_band_containment"),
        "deployment_lap_mape_pct": _mean_over_runs(runs, "deployment", "lap_mape_pct"),
        "deployment_lap_bias_pct": _mean_over_runs(runs, "deployment", "lap_bias_pct"),
        "soc_mean_mae_mj": _mean_over_runs(runs, "energy_belief", "soc_mean_mae_mj"),
        "soc_mean_bias_mj": _mean_over_runs(runs, "energy_belief", "soc_mean_bias_mj"),
        "soc_band_containment": _mean_over_runs(
            runs, "energy_belief", "soc_particle_band_containment"),
        "soc_band_containment_range": _range_over_runs(
            runs, "energy_belief", "soc_particle_band_containment"),
    }
    return Validation(
        True, "direct_truth", MIXED, metrics=m, artifact=path,
        fingerprint=d.get("summary_fingerprint"),
        provenance="hidden simulator truth; scoring layer only",
        notes=("deployment flows are measured well: 5.71% lap MAPE, 93.4% band "
               "containment",
               "CdA is NOT identified on this path: identifiability 0.0, interval "
               "658.6% of truth wide. It contains the truth 3/3, which is a "
               "containment result and not an accuracy one",
               "SOC uncertainty is not calibrated: band containment 43.8% mean, "
               "35.7-55.0% across seeds",
               "the simulator has no genuine lift-and-coast regime, so the drag "
               "channel that identifies CdA on real data is not exercised here",
               "the CdA interval spans NEGATIVE drag area ([-2.19, +2.16] on the "
               "first seed, centre -0.018), which is physically impossible. "
               "\"truth contained 3/3\" is therefore close to vacuous: an "
               "interval that wide would contain almost any truth"))


def _real_validation() -> Validation:
    """Part 2: held-out future telemetry. The verdict is negative and stays negative.

    The comparison is the one that matters for the project's claim: does knowing
    the inferred energy predict the rival's FUTURE speed better than not knowing
    it? Against a fixed-energy baseline the answer is no on two of three targets
    and yes on one, which is not a robust improvement. Tuning until it flips would
    be fitting the held-out set.
    """
    d, path = _latest("part2")
    if d is None:
        return Validation(False, "indirect_predictive", NOT_ATTEMPTED,
                          notes=("no Part 2 artifact on disk",))
    abl = d.get("energy_ablation", {}).get("paired_metrics", {})
    rp = d.get("real_prediction", {})
    ds = rp.get("dataset", {})
    metrics = {
        "n_examples": ds.get("n_examples"),
        "n_races": ds.get("n_races"),
        "by_target": ds.get("by_target"),
        "aggregate_mae": {t: {k: v.get("mae") for k, v in g.get("aggregate", {}).items()}
                          for t, g in rp.get("metrics", {}).items()},
        "xray_minus_fixed_mae": {t: v.get("xray_minus_fixed_mae")
                                 for t, v in abl.items()},
        "xray_minus_neutral_mae": {t: v.get("xray_minus_neutral_mae")
                                   for t, v in abl.items()},
        "limitations": ds.get("limitations", []),
    }
    wins = [t for t, v in abl.items() if (v.get("xray_minus_fixed_mae") or 0.0) < 0.0]
    return Validation(
        True, "indirect_predictive", NO_ROBUST_IMPROVEMENT, metrics=metrics,
        artifact=path, fingerprint=d.get("summary_fingerprint"),
        provenance=json.dumps(d.get("split_definition", {}), sort_keys=True),
        notes=("there is no real battery ground truth; this scores future "
               "OBSERVABLE speed, never energy itself",
               f"X-RAY beats the fixed-energy baseline on {len(wins)} of "
               f"{len(abl)} targets ({', '.join(wins) or 'none'}) and loses on "
               "the rest -- lower MAE is better, so a POSITIVE "
               "xray_minus_fixed_mae is a loss",
               "the energy-neutral baseline is the strongest of the three on two "
               "targets, which is the result, not a bug",
               "a single winning target is not evidence; reporting only "
               "braking_point_speed would be cherry-picking"))


def energy_inference_status() -> dict[str, Any]:
    """The one status the product must not get wrong.

    Both halves are mandatory. "INFERRED" alone invites the reader to assume it
    works; "NOT DEMONSTRATED" alone invites them to assume nothing was built.
    """
    syn, real = _synthetic_validation(), _real_validation()
    return {
        "component": "energy_inference",
        "status": INFERRED,
        "headline": "INFERRED — REAL HELD-OUT PREDICTIVE VALIDATION: NOT DEMONSTRATED",
        "synthetic_validation": {
            "available": syn.available, "direct_truth": syn.kind == "direct_truth",
            "result": syn.result, "metrics": syn.metrics,
            "fingerprint": syn.fingerprint, "notes": list(syn.notes),
        },
        "real_validation": {
            "available": real.available, "type": real.kind, "result": real.result,
            "metrics": real.metrics, "fingerprint": real.fingerprint,
            "provenance": real.provenance, "notes": list(real.notes),
        },
        "identifiability": {"status": IDENT_LIMITED,
                            "reason": "CdA identifiability 0.0 on the canonical "
                                      "synthetic path; exact real re-inference "
                                      "sensitivity is not measurable from the "
                                      "persisted payloads"},
        "real_ground_truth": {"available": False,
                              "reason": "no public battery channel exists"},
        "registry_version": REGISTRY_VERSION,
    }


# -------------------------------------------------------------- the registry
def registry() -> list[ModelEntry]:
    """Every model a decision can depend on, with its status and its evidence."""
    syn, real = _synthetic_validation(), _real_validation()
    d2, _ = _latest("part2")
    phys = (d2 or {}).get("physical_calibration_status", {})
    pass_dataset = (d2 or {}).get("pass_dataset_audit", {})

    def why(key: str, default: str) -> str:
        row = phys.get(key) or {}
        r, res = row.get("reason"), row.get("result")
        return f"{res} — {r}" if r and res else (r or res or default)

    return [
        ModelEntry(
            "Canonical energy inference", "energy_inference", INFERRED, True,
            version="realfit canonical path", validation=real,
            identifiability=IDENT_LIMITED,
            notes=("synthetic direct validation exists and is mixed; the real "
                   "held-out comparison shows no robust improvement",
                   "status INFERRED describes the MECHANISM. It is not a claim "
                   "that the inference is accurate")),
        ModelEntry(
            "Canonical inference, synthetic scoring", "energy_inference_synthetic",
            INFERRED, False, validation=syn, identifiability=IDENT_NONE,
            reason="scoring layer; reads hidden truth and therefore never ships",
            notes=("same code as the production entry, run against simulator "
                   "truth -- the blindfold is what makes this a separate entry "
                   "rather than a separate implementation",)),
        ModelEntry(
            "Stage-1 estimator", "stage1_estimator", RESEARCH_ONLY, False,
            reason="locked to the synthetic `Observation` type by its import; "
                   "never executes on a real race",
            validation=Validation(False, "none", NOT_ATTEMPTED),
            notes=("its metrics are not evidence about real-race inference",)),
        ModelEntry(
            "Set-membership / RBPF stack", "setmem_rbpf", RESEARCH_ONLY, False,
            reason="reached by tests only; not on the shipping post-ingestion path",
            validation=Validation(False, "none", NOT_ATTEMPTED),
            notes=("balance, setmem, modes, pipeline, rbpf, deadband, strategy, "
                   "pooling",
                   "the assumption-free polytope lives here, so quoting the "
                   "tilted posterior alone from the production path is still "
                   "wrong -- it is just not this module's result")),
        ModelEntry(
            "Pass probability p_pass", "pass_model", SYNTHETIC, True,
            validation=Validation(
                False, "none", REFUSED, metrics=dict(pass_dataset),
                notes=("no track-status channel, so SC/VSC laps cannot be removed",
                       "no reliable pit timing",
                       "no decision-time physics delta_v in the persisted payload",
                       "3.56% positive rate over 4,163 usable samples")),
            reason="empirical fit refused: the dataset cannot support it",
            notes=("the only invented coefficients in the codebase, and they "
                   "propagate into every decision claim",)),
        ModelEntry(
            "Tyre model", "tyres", SYNTHETIC, True,
            reason=why("tyres", "compound and TyreLife present; wear is modelled"),
            validation=Validation(False, "none", NOT_ATTEMPTED),
            notes=("TyreLife is observed AGE IN LAPS and is never wear",)),
        ModelEntry(
            "Drag area CdA", "cda", INFERRED, True, validation=syn,
            identifiability=IDENT_LIMITED,
            reason=why("cda", "identified sets retained; not relabelled calibrated"),
            notes=("bracketed from both sides by the regulation, never "
                   "thresholded; interval width IS the identifiability score",
                   "658.6% of truth wide on the canonical synthetic path, and "
                   "the interval includes physically impossible negative drag, "
                   "so containment of the truth is not an accuracy claim")),
        ModelEntry(
            "Downforce ClA", "cla", DISABLED, False,
            reason=why("cla", "held-out corner/braking improvement not established"),
            validation=Validation(False, "none", NOT_DEMONSTRATED)),
        ModelEntry(
            "Wake / dirty air", "wake", SYNTHETIC, True,
            reason=why("wake", "tow and dirty air cannot be separated"),
            validation=Validation(False, "none", NOT_ATTEMPTED)),
        ModelEntry(
            "Track wetness", "wetness", HEURISTIC, True,
            reason=why("wetness_wind", "no weather trace"),
            validation=Validation(False, "none", NOT_ATTEMPTED),
            notes=("INFERRED from the Rainfall bool with assumed rain/drying "
                   "coefficients; not a FastF1 measurement",)),
        ModelEntry(
            "Wind projection", "wind", UNAVAILABLE, False,
            reason="no verified circuit orientation; FastF1 `CircuitGeometry.xy` "
                   "is a track-local frame and is not aligned to true North",
            validation=Validation(False, "none", NOT_ATTEMPTED),
            notes=("the projection declines rather than assuming 0 degrees; the "
                   "fitted wind residual pathway is preserved separately",)),
        ModelEntry(
            "Rival pit context", "rival_pit", UNAVAILABLE, False,
            reason=why("rival_pit", "pit data unavailable in the current corpus"),
            validation=Validation(False, "none", NOT_ATTEMPTED),
            notes=("returns UNKNOWN; actual pit times are oracle-only and never "
                   "reach a decision",)),
        ModelEntry(
            "Future rival-energy forecast", "energy_forecast", HEURISTIC, True,
            reason="bounded historical lap-rate transition from the current belief",
            validation=Validation(False, "none", NOT_ATTEMPTED)),
        ModelEntry(
            "Regulation variant", "regulation_variant", REGULATION, True,
            version="2026, pre/post-Miami dispatched by date",
            validation=Validation(False, "none", NOT_ATTEMPTED),
            notes=("correctness-driven, not metric-driven: the right rule in "
                   "force beats a prettier number under the wrong one",
                   "a variant mismatch between simulator and fixtures once "
                   "scaled every harvest measurement by 0.71x silently")),
        ModelEntry(
            "Vehicle mass model", "mass_model", PHYSICS, True,
            version="790 kg",
            validation=Validation(
                False, "direct_truth", NOT_DEMONSTRATED,
                notes=("MEASURED — NOT ADOPTED: the synthetic winner depended on "
                       "simulator-specific fuel assumptions and did not transfer "
                       "defensibly",
                       "a synthetic metric improvement is not a real-path "
                       "justification")),
            notes=("790 kg remains active deliberately",)),
        ModelEntry(
            "Deployment-zone executable ceiling", "deployment_ceiling", PHYSICS,
            True, version="measured per zone per entry speed",
            validation=Validation(
                False, "direct_truth", PASSED,
                notes=("measured through the shared integrator, not integrated "
                       "in closed form: integrating the taper over the window "
                       "overstates zone A by 0.981 vs 0.448 MJ because it counts "
                       "the braking stretch and ignores that deploying harder "
                       "raises v and lowers the next step's ceiling",)),
            notes=("context-dependent by construction: Circuit Sigma zone A "
                   "absorbs 1.603 MJ entered at 200 km/h and 0.448 MJ at "
                   "314 km/h, where the taper reaches 0 kW after 387 m of an "
                   "1100 m straight",
                   "replaces `max(zone.energy_grid)` as P2's budget ceiling; "
                   "that axis covers run-up PLUS straight and offered budgets "
                   "up to 1.602 MJ in a zone that could execute 0.45")),
    ]


@lru_cache(maxsize=1)
def _registry_payload() -> dict[str, Any]:
    entries = registry()
    return {
        "registry_version": REGISTRY_VERSION,
        "energy_inference": energy_inference_status(),
        "entries": [
            {**{k: v for k, v in asdict(e).items() if k != "validation"},
             "notes": list(e.notes),
             "validation": (None if e.validation is None else
                            {**asdict(e.validation),
                             "notes": list(e.validation.notes)})}
            for e in entries],
        "production_components": sorted(e.component for e in entries if e.production),
        "research_only_components": sorted(
            e.component for e in entries if e.status == RESEARCH_ONLY),
        "data_quality_limitations": list(DATA_QUALITY_LIMITATIONS),
    }


def registry_payload() -> dict[str, Any]:
    """JSON-ready registry for the service. Cached: it reads artifacts from disk."""
    import copy
    return copy.deepcopy(_registry_payload())


# The limitations a reader must see before trusting any number on the page. Each
# one is a fact about the CORPUS, not a hedge about the model.
DATA_QUALITY_LIMITATIONS = (
    "NO REAL BATTERY GROUND TRUTH — no public channel carries stored energy",
    "NO TRACK STATUS IN CURRENT CORPUS — SC/VSC/yellow contamination cannot be removed",
    "NO WEATHER TRACE — only a session-summary air density is available",
    "NO PIT DATA — rival pit context is UNKNOWN, never guessed",
    "NO GENUINE SYNTHETIC LIFT-AND-COAST — the drag channel that identifies CdA "
    "on real data is not exercised by the simulator",
    "CDA POORLY IDENTIFIABLE IN SYNTHETIC DIRECT VALIDATION — identifiability 0.0, "
    "interval 658.6% of truth wide",
    "CURRENT ENERGY INFERENCE DOES NOT SHOW ROBUST REAL PREDICTIVE VALUE — it loses "
    "to a fixed-energy baseline on 2 of 3 held-out targets",
)
