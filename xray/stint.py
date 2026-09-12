"""Stint state and pit proximity, with the oracle held at arm's length.

The decision engine wants to know how long it will have to live with the tyres
it damages by attacking. In a live race our own team knows its own plan, so
using it is not cheating. Replaying a 2024 race and reading the lap the car
ACTUALLY pitted is cheating, and it is the easiest possible mistake to make
because the field is sitting right there in the Laps table.

So there are two entry points and they do not meet:

    pit_context_from_plan(...)   causal. team_plan / synthetic_plan.
    infer_pit_context(...)       causal. inferred, or unknown -- never a guess.
    oracle_pit_context(...)      NOT causal. source="oracle_eval", for scoring
                                 an inference after the fact, never an input.

`PitContext.is_causal` is the gate, and `decision_service` refuses anything that
fails it. A boolean "close to pit" is deliberately not the interface: 0.5 laps,
2 laps and 8 laps are three different strategic situations and collapsing them
loses exactly the information the DP needs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CAUSAL_SOURCES = ("team_plan", "synthetic_plan", "inferred", "unknown")
ORACLE_SOURCE = "oracle_eval"


@dataclass(frozen=True)
class StintState:
    """Observed stint metadata for one driver on one lap.

    Every field here is published by FastF1 (verified against the installed
    3.8.3 `Laps` columns: Stint, Compound, TyreLife, FreshTyre). `tyre_life` is
    AGE IN LAPS and is never wear -- see xray.tyres.
    """
    driver: str
    lap: int
    stint: int | None = None
    compound: str | None = None
    tyre_life: float | None = None
    fresh_tyre: bool | None = None
    source: str = "lap_metadata"


@dataclass(frozen=True)
class PitContext:
    """How near a tyre change is, and how much that claim is worth."""
    planned_pit_lap: int | None = None
    pit_window_start_lap: int | None = None
    pit_window_end_lap: int | None = None
    laps_to_pit_mean: float | None = None
    pit_within_1_lap_prob: float = 0.0
    pit_within_2_laps_prob: float = 0.0
    source: str = "unknown"
    confidence: float = 0.0

    @property
    def is_causal(self) -> bool:
        """False for anything built from events after the decision."""
        return self.source in CAUSAL_SOURCES

    @property
    def is_known(self) -> bool:
        return self.laps_to_pit_mean is not None


UNKNOWN_PIT = PitContext(source="unknown", confidence=0.0)


def _probs(laps_to_pit: float) -> tuple[float, float]:
    """P(stop within 1 lap), P(within 2), for a deterministic plan.

    A known plan has no uncertainty about WHETHER, only about how far away it
    is, so these are indicators rather than a distribution. A real probabilistic
    model would replace this -- and would have to be validated first.
    """
    return (1.0 if laps_to_pit <= 1.0 else 0.0,
            1.0 if laps_to_pit <= 2.0 else 0.0)


def pit_context_from_plan(current_lap: int, planned_pit_lap: int | None,
                          source: str = "team_plan",
                          confidence: float = 0.95,
                          window: tuple[int, int] | None = None) -> PitContext:
    """A stop we know about because somebody told us.

    Allowed for our own car in a live race, and for either car in the synthetic
    simulator where the schedule is config. Not allowed for a rival in a replay:
    that is `infer_pit_context`.
    """
    if source not in ("team_plan", "synthetic_plan"):
        raise ValueError(f"{source!r} is not a plan source")
    if planned_pit_lap is None:
        return PitContext(source=source, confidence=0.0)
    laps = float(planned_pit_lap) - float(current_lap)
    p1, p2 = _probs(laps)
    start, end = window if window else (planned_pit_lap, planned_pit_lap)
    return PitContext(planned_pit_lap=int(planned_pit_lap),
                      pit_window_start_lap=int(start), pit_window_end_lap=int(end),
                      laps_to_pit_mean=laps, pit_within_1_lap_prob=p1,
                      pit_within_2_laps_prob=p2, source=source,
                      confidence=float(confidence))


def pit_context_from_window(current_lap: int, start_lap: int, end_lap: int,
                            source: str = "team_plan",
                            confidence: float = 0.6) -> PitContext:
    """A stop planned as a window rather than a lap.

    Uniform over the window, which is an assumption and a weak one -- it is here
    because a window is what teams actually commit to, not because uniform is
    right. The probabilities are the fraction of the window that falls inside
    the horizon.
    """
    start, end = int(min(start_lap, end_lap)), int(max(start_lap, end_lap))
    n = end - start + 1
    laps = np.arange(start, end + 1, dtype=float) - float(current_lap)
    laps = laps[laps >= 0.0]
    if len(laps) == 0:
        return PitContext(source=source, confidence=0.0)
    return PitContext(planned_pit_lap=None, pit_window_start_lap=start,
                      pit_window_end_lap=end, laps_to_pit_mean=float(laps.mean()),
                      pit_within_1_lap_prob=float(np.mean(laps <= 1.0)),
                      pit_within_2_laps_prob=float(np.mean(laps <= 2.0)),
                      source=source, confidence=float(confidence))


def infer_pit_context(stint: StintState, laps_seen: int,
                      model=None) -> PitContext:
    """A rival's stop, from what we can see now. Usually: we cannot see it.

    This checkout has NO validated pit-inference model. Tyre age alone does not
    tell you when a team will stop -- it depends on the undercut, traffic, the
    safety-car probability and a strategy call nobody publishes -- so inventing
    a "stops at 25 laps" rule here would be a guess wearing a probability.

    So this returns `unknown` unless a validated model is handed in. That is the
    correct answer, and the DP has an explicit branch for it: no invented reset,
    the worn state simply continues.
    """
    if model is None:
        return UNKNOWN_PIT
    out = model(stint, laps_seen)
    if not isinstance(out, PitContext):
        raise TypeError("pit inference model must return a PitContext")
    if out.source != "inferred":
        raise ValueError("an inference model must label its output 'inferred'")
    return out


def oracle_pit_context(current_lap: int, actual_pit_lap: int | None) -> PitContext:
    """The lap the car really pitted. EVALUATION ONLY.

    Built from `PitInTime`/`PitOutTime`, which are in the future relative to any
    decision being replayed. `is_causal` is False, and `decision_service` raises
    rather than accepting one. It exists so an inference can be SCORED, which
    needs the answer -- it just must not also be the input.
    """
    if actual_pit_lap is None:
        return PitContext(source=ORACLE_SOURCE, confidence=0.0)
    laps = float(actual_pit_lap) - float(current_lap)
    p1, p2 = _probs(laps)
    return PitContext(planned_pit_lap=int(actual_pit_lap),
                      pit_window_start_lap=int(actual_pit_lap),
                      pit_window_end_lap=int(actual_pit_lap),
                      laps_to_pit_mean=laps, pit_within_1_lap_prob=p1,
                      pit_within_2_laps_prob=p2, source=ORACLE_SOURCE,
                      confidence=1.0)


def require_causal(ctx: PitContext) -> PitContext:
    """Gate at the boundary of the decision path."""
    if not ctx.is_causal:
        raise ValueError(
            f"pit context source {ctx.source!r} is not causal; oracle pit data "
            "may score an inference but may never feed a decision")
    return ctx


# ------------------------------------------------------------------ payload
def stint_from_lap_row(driver: str, row: dict) -> StintState:
    """Build from an analysis payload lap row, tolerating missing keys.

    `compound` and `tyre_life` are the existing payload keys and keep their
    meaning. `stint` and `fresh_tyre` are added only where present -- they are
    real FastF1 columns, but a payload written before they were serialised will
    not have them and must not be given invented values.
    """
    def _i(key):
        v = row.get(key)
        if v is None:
            return None
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return int(f) if np.isfinite(f) else None

    fresh = row.get("fresh_tyre")
    life = row.get("tyre_life")
    return StintState(
        driver=driver, lap=int(row.get("lap", 0)), stint=_i("stint"),
        compound=(str(row["compound"]) if row.get("compound") else None),
        tyre_life=(None if life is None else float(life)),
        fresh_tyre=(None if fresh is None else bool(fresh)))


def actual_pit_lap_from_rows(rows: list[dict]) -> int | None:
    """First lap with a recorded pit entry. ORACLE -- evaluation only."""
    for r in sorted(rows, key=lambda x: int(x.get("lap", 0))):
        if r.get("pit_in_time_s") is not None:
            return int(r.get("lap", 0))
    return None
