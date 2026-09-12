"""P2: the empirical pass model, and the audit that decides whether to fit one.

`xray.overtake.p_pass` is synthetic. Replacing it needs a labelled dataset of
real overtaking attempts, and the first job is not fitting -- it is finding out
whether the data can support a label at all. This module builds the dataset and
audits it; `fit` refuses to run when the audit says the label is not clean.

The label is the hard part. Lap-resolution position tells you that a car was
behind on lap L and ahead on lap L+1. It does not tell you WHY, and the most
common why is a pit stop: the cars never raced, one of them just spent 22
seconds stationary. A model trained on unfiltered position changes learns the
pit cycle and reports it as overtaking skill.

So every exclusion below is derived from a field that is actually in the
payload. Pit cycles are detected from tyre-life resets and compound changes,
which are real published values. Safety cars are NOT detectable from the current
payload -- there is no track-status channel in it -- and that is reported as a
contamination bound rather than silently ignored.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

FEATURE_SCHEMA_VERSION = "p2-pass-features-v1"

# How long the attacker must stay ahead for the change to count as a completed
# pass rather than a slipstream swap that is handed straight back. ASSUMED, and
# expressible only in laps because lap-resolution position is the finest signal
# the payload has.
CONFIRMATION_LAPS = 2


@dataclass(frozen=True)
class PassSample:
    """One labelled overtaking opportunity.

    Every feature is computed from data at or before the decision lap. The label
    alone looks forward, which is what a label is for.
    """
    race_id: str
    circuit: str
    lap: int
    attacker: str
    defender: str
    # ---- features (decision-time only)
    gap_s: float
    attacker_compound: str | None
    defender_compound: str | None
    attacker_tyre_life: float | None
    defender_tyre_life: float | None
    tyre_life_delta: float | None
    position_delta: int
    # ---- label and its provenance
    pass_success: int | None
    excluded: bool
    exclusion_reason: str


def _pit_laps_from_tyre_metadata(rows: list[dict]) -> set:
    """Laps where a driver plainly changed tyres, from published fields only.

    Two signals, both real columns: the compound changed, or tyre life fell
    instead of rising. Neither is a guess -- a set that gets younger between
    laps was replaced. This is not a complete pit detector (a same-compound stop
    with a stale TyreLife would be missed) and it is not presented as one; it is
    a lower bound on pit cycles, used to EXCLUDE samples, so missing one leaves
    contamination rather than inventing an exclusion.
    """
    out = set()
    ordered = sorted(rows, key=lambda r: int(r.get("lap", 0)))
    for prev, cur in zip(ordered, ordered[1:]):
        lap = int(cur.get("lap", 0))
        if prev.get("compound") and cur.get("compound") and prev["compound"] != cur["compound"]:
            out.add(lap)
            continue
        a, b = prev.get("tyre_life"), cur.get("tyre_life")
        if a is not None and b is not None and float(b) < float(a):
            out.add(lap)
    return out


def build_dataset(payload: dict) -> list[PassSample]:
    """Labelled attacker/defender pairs from one analysed race payload."""
    race_id = str(payload.get("id") or payload.get("event") or "unknown")
    circuit = str(payload.get("circuit") or "unknown")
    laps = payload.get("laps") or []
    gaps = payload.get("gaps") or []
    if not laps or not gaps:
        return []

    by_driver: dict[str, list[dict]] = {}
    for r in laps:
        by_driver.setdefault(str(r.get("driver")), []).append(r)
    pos: dict[tuple, int] = {}
    meta: dict[tuple, dict] = {}
    for r in laps:
        if r.get("position") is None:
            continue
        key = (int(r["lap"]), str(r["driver"]))
        pos[key] = int(r["position"])
        meta[key] = r
    pit_laps = {d: _pit_laps_from_tyre_metadata(rs) for d, rs in by_driver.items()}
    max_lap = max((int(r["lap"]) for r in laps), default=0)

    out = []
    for g in gaps:
        lap = int(g.get("lap", 0))
        atk, dfn = str(g.get("car")), str(g.get("ahead"))
        if not atk or not dfn or dfn == "None":
            continue
        here = (lap, atk), (lap, dfn)
        if here[0] not in pos or here[1] not in pos:
            continue
        p_atk, p_dfn = pos[here[0]], pos[here[1]]
        if p_atk <= p_dfn:
            continue                      # attacker must start behind

        # ---- label: ahead next lap, and still ahead CONFIRMATION_LAPS later
        label, excluded, reason = None, False, ""
        end = lap + CONFIRMATION_LAPS
        if end > max_lap:
            excluded, reason = True, "confirmation window runs past the end of the race"
        else:
            seq = []
            for L in range(lap + 1, end + 1):
                ka, kd = (L, atk), (L, dfn)
                if ka not in pos or kd not in pos:
                    seq = []
                    break
                seq.append(pos[ka] < pos[kd])
            if not seq:
                excluded, reason = True, "missing position data in the confirmation window"
            else:
                label = int(all(seq))

        # ---- exclusions, from published fields only
        if not excluded:
            window = range(lap, end + 1)
            if any(L in pit_laps.get(atk, ()) for L in window):
                excluded, reason = True, "attacker pit cycle in the window"
            elif any(L in pit_laps.get(dfn, ()) for L in window):
                excluded, reason = True, "defender pit cycle in the window"

        ma, md = meta[here[0]], meta[here[1]]
        tl_a, tl_d = ma.get("tyre_life"), md.get("tyre_life")
        out.append(PassSample(
            race_id=race_id, circuit=circuit, lap=lap, attacker=atk, defender=dfn,
            gap_s=float(g.get("gap_s", float("nan"))),
            attacker_compound=ma.get("compound"), defender_compound=md.get("compound"),
            attacker_tyre_life=None if tl_a is None else float(tl_a),
            defender_tyre_life=None if tl_d is None else float(tl_d),
            tyre_life_delta=(None if tl_a is None or tl_d is None
                             else float(tl_a) - float(tl_d)),
            position_delta=int(p_atk - p_dfn),
            pass_success=label, excluded=excluded, exclusion_reason=reason))
    return out


@dataclass
class DatasetAudit:
    """What the data can and cannot support. Produced BEFORE any fitting."""
    n_raw: int = 0
    n_usable: int = 0
    n_positive: int = 0
    n_excluded: int = 0
    exclusions: dict = field(default_factory=dict)
    races: list = field(default_factory=list)
    circuits: list = field(default_factory=list)
    missing_signals: list = field(default_factory=list)
    available_features: list = field(default_factory=list)
    blocking_reasons: list = field(default_factory=list)

    @property
    def fit_permitted(self) -> bool:
        return not self.blocking_reasons

    @property
    def positive_rate(self) -> float:
        return 0.0 if not self.n_usable else self.n_positive / self.n_usable


# A pass model is a probability model. These are the minimums below which a
# grouped held-out calibration claim would be arithmetic rather than evidence.
MIN_USABLE_SAMPLES = 300
MIN_POSITIVES = 40
MIN_RACE_GROUPS = 4


def audit_dataset(samples: list[PassSample], payloads: list[dict] | None = None
                  ) -> DatasetAudit:
    """Decide whether this dataset can support an empirical claim."""
    a = DatasetAudit()
    a.n_raw = len(samples)
    usable = [s for s in samples if not s.excluded and s.pass_success is not None]
    a.n_usable = len(usable)
    a.n_positive = sum(s.pass_success for s in usable)
    a.n_excluded = sum(1 for s in samples if s.excluded)
    for s in samples:
        if s.excluded:
            a.exclusions[s.exclusion_reason] = a.exclusions.get(s.exclusion_reason, 0) + 1
    a.races = sorted({s.race_id for s in usable})
    a.circuits = sorted({s.circuit for s in usable})
    a.available_features = ["gap_s", "position_delta", "attacker_tyre_life",
                            "defender_tyre_life", "tyre_life_delta",
                            "attacker_compound", "defender_compound"]

    # What is NOT in the payload, checked rather than assumed.
    probe = (payloads or [{}])[0]
    if not probe.get("weather_trace"):
        a.missing_signals.append("weather_trace (causal rho/wetness per sample)")
    lap_keys = set((probe.get("laps") or [{}])[0].keys()) if probe.get("laps") else set()
    for key, why in (("pit_in_time_s", "pit in/out timing"),
                     ("stint", "stint index"),
                     ("fresh_tyre", "fresh-tyre flag")):
        if key not in lap_keys:
            a.missing_signals.append(f"{key} ({why})")
    a.missing_signals.append(
        "track status / safety car / VSC / yellow -- no channel in the payload, "
        "so SC-period position changes cannot be excluded")
    a.missing_signals.append(
        "physics delta_v at the decision point -- needs per-driver estimator "
        "belief plus a wear surface per opportunity, not present in these payloads")

    if a.n_usable < MIN_USABLE_SAMPLES:
        a.blocking_reasons.append(
            f"only {a.n_usable} usable samples, below the {MIN_USABLE_SAMPLES} "
            "needed for a grouped held-out calibration claim")
    if a.n_positive < MIN_POSITIVES:
        a.blocking_reasons.append(
            f"only {a.n_positive} positive labels, below {MIN_POSITIVES}")
    if len(a.races) < MIN_RACE_GROUPS:
        a.blocking_reasons.append(
            f"only {len(a.races)} race groups, below {MIN_RACE_GROUPS} needed to "
            "split by race rather than by row")
    a.blocking_reasons.append(
        "no safety-car/VSC channel: label contamination from non-racing position "
        "changes cannot be bounded, so a fitted probability would not be a "
        "probability of overtaking")
    a.blocking_reasons.append(
        "no decision-time delta_v feature: the physics feature that the whole "
        "model is meant to condition on is not reconstructible from these payloads")
    return a


def dataset_fingerprint(samples: list[PassSample]) -> str:
    blob = json.dumps([asdict(s) for s in samples], sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def load_payloads(directory: str | Path) -> list[dict]:
    out = []
    for path in sorted(Path(directory).glob("*.json")):
        try:
            out.append(json.loads(path.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    return out


def fit(samples: list[PassSample], audit: DatasetAudit):
    """Refuses unless the audit permits it. There is no force flag."""
    if not audit.fit_permitted:
        raise RuntimeError(
            "dataset audit blocks fitting:\n  - " + "\n  - ".join(audit.blocking_reasons)
            + "\n`xray.overtake.p_pass` remains SYNTHETIC.")
    raise NotImplementedError(
        "no dataset has yet passed the audit, so no fitting path has been "
        "written. Writing one before the data exists would be writing a model "
        "whose calibration report could never be produced.")
