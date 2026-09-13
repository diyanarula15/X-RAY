"""LLM-as-judge over the precomputed decision bundles in `out/decisions/`.

PRESENTATION/AUDIT layer. The rule for this package is stricter than the layer
rule in `docs/pipeline_layers.md` and is enforced by
`tests/test_judge.py::test_judge_package_imports_only_stdlib_and_the_sdk`:

    xray/judge/ imports only the standard library and google.genai.
    Nothing else from xray/, nothing from simulation/.

The consequence is the point. The judge cannot call `overtake.p_pass`, cannot
call `vehicle.step`, cannot re-solve a DP. Every number it reports came out of a
JSON file on disk, so a verdict can never quietly become a second, unvalidated
physics path. A judge that could compute would eventually be asked to, and the
one thing this package must never do is produce a number the rest of the stack
might believe.

What it judges: whether X-RAY's own ATTACK/HOLD call was justified BY THE
EVIDENCE the engine had. Not whether the driver agreed -- that comparison is
`matches_recommendation`, it is already computed by
`decision_service.historical_replay`, and it needs no LLM. See `evidence.py`
for the redaction that keeps the two questions apart.
"""
from __future__ import annotations

# Bumped when the evidence surface, the loop or the verdict shape changes in a
# way that should invalidate cached verdicts. It is part of the cache key.
JUDGE_VERSION = "judge-v1"

# Bumped when the rubric axes change meaning. Separate from JUDGE_VERSION so a
# plumbing fix does not throw away a run's worth of quota, but a rubric change
# always does.
RUBRIC_VERSION = "rubric-v1"

# Every verdict carries this. `docs/model_inventory.md` has the matching entry.
VERDICT_LABEL = (
    "GENERATED COMMENTARY -- an LLM's reading of the evidence. Not a "
    "measurement, not physics, not a validation of the decision engine."
)

# The bundles are produced by the real stack and by nothing else. Quoted in
# every report so a verdict can never be read as scoring Stage 1 or the
# set-membership group, which produced none of these numbers.
STACK = ("real: FastF1 -> xray/data/ingest.py -> xray/realfit.py -> "
         "xray/analysis.py -> xray/decision_service.py")

__all__ = ["JUDGE_VERSION", "RUBRIC_VERSION", "VERDICT_LABEL", "STACK"]
