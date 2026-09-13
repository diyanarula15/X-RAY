"""Evidence.tsx: the numbers are served, and the status words are the artifacts'.

Same failure mode as `test_frontend_p2.py` / `test_frontend_p4.py`, one step
further down: not "does the view compute", but "does the view REMEMBER". The
ablation chart's eight (Hz, MAPE) pairs were literals in `Method.tsx` once; the
artefact then moved and the page kept showing 5.03% at 100 Hz against a measured
7.68% -- a third of the error, invisible, for two weeks. So the ban here is on
the literals, not on arithmetic.

The other three guards are claim-strength guards. The P3.5 Part 3 candidate was
built, measured and NOT activated; a page that omits that reads as if the
hardened estimator shipped, and a page that quotes a flattering Part 3 number
without the activation verdict reads as if it worked. Both are overclaims in the
same direction.

Every file read passes `encoding="utf-8"` -- invariant 14; the repo has been
broken three times by omitting it.
"""
from __future__ import annotations

import glob
import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
APP = ROOT / "simulation" / "app" / "src"
EVIDENCE = APP / "views" / "Evidence.tsx"
P3_PANEL = APP / "components" / "P3Panel.tsx"
ABLATION = ROOT / "out" / "ablation.json"


def _src(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def _prose(src: str) -> str:
    """Collapse whitespace before asserting a sentence.

    JSX line-wraps prose at whatever column prettier lands on, so a sentence
    asserted as a contiguous string breaks on reflow rather than on meaning. The
    test must fail when the claim changes, not when the indentation does.
    """
    return re.sub(r"\s+", " ", src)


def _strip_comments(src: str) -> str:
    """Comments are prose, and this file's prose cites the numbers it banned.

    The AblationChart docstring quotes "5.0% MAPE at 100 Hz against a measured
    7.68%" on purpose -- that is the history of the bug. Stripping comments is
    what lets the literal ban be strict about rendered output without deleting
    the explanation of why it exists.
    """
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return "\n".join(l for l in src.splitlines() if not l.strip().startswith("//"))


@pytest.fixture(scope="module")
def evidence() -> str:
    assert EVIDENCE.exists(), "Evidence.tsx is gone"
    return _src(EVIDENCE)


@pytest.fixture(scope="module")
def evidence_code(evidence) -> str:
    return _strip_comments(evidence)


# ------------------------------------------------- the ablation is served, not typed
def test_the_ablation_chart_reads_the_api_and_not_a_local_table(evidence_code):
    assert "api.ablation()" in evidence_code, "Evidence no longer fetches /api/ablation"
    assert "ab.rates" in evidence_code and "ab.mape" in evidence_code, (
        "the chart no longer plots the served arrays")
    # A literal array of rates next to the chart is how the old copy started.
    assert not re.search(r"\[\s*100(\.0)?\s*,\s*50", evidence_code), (
        "Evidence.tsx declares a literal rate ladder")


@pytest.mark.skipif(not ABLATION.exists(), reason="out/ablation.json not generated")
def test_no_ablation_figure_is_a_literal_in_the_tsx(evidence_code):
    """Every number in the artefact must be absent from the rendered source.

    Checked at 1 and 2 decimals because that is how a transcription gets written
    down ("7.7%", "7.68%") -- the full float never appears in prose, so banning
    only the exact value would ban nothing.
    """
    ab = json.loads(ABLATION.read_text(encoding="utf-8"))
    series = [*ab["mape"], *ab["coverage"], *ab["cda_abs_err_pct"]]
    for v in series:
        # Bare and percent-suffixed, and bounded on both sides: a plain substring
        # search flagged the decision table's "2.6" laps-chosen against the
        # 3.7 Hz MAPE of 2.61% and the oracle's 0.483 against a coverage of 0.48.
        # A ban that fires on unrelated numbers is a ban somebody deletes.
        for lit in (f"{v:.2f}", f"{v:.1f}%"):
            pattern = rf"(?<![\d.]){re.escape(lit)}(?![\d])"
            m = re.search(pattern, evidence_code)
            assert not m, (
                f"Evidence.tsx hardcodes the ablation figure {lit} "
                f"(from {ABLATION.name}); it must come from /api/ablation")


@pytest.mark.skipif(not ABLATION.exists(), reason="out/ablation.json not generated")
def test_the_artefact_is_not_older_than_the_regulation_it_was_measured_under():
    """A stale ablation is a wrong ablation, and it fails silently.

    `out/ablation.json` predated the corrected taper curve in `xray/regs.py` for
    two weeks and served 5.03% at 100 Hz while the estimator measured 7.68%. The
    endpoint reads the file and the view draws it, so nothing in the stack
    noticed. This is the cheap guard: mtime, not a re-run.
    """
    regs = ROOT / "xray" / "regs.py"
    assert ABLATION.stat().st_mtime >= regs.stat().st_mtime, (
        "out/ablation.json is older than xray/regs.py -- regenerate it with "
        "`python3 scripts/05.run_ablation.py` rather than editing the numbers")


def test_the_ablation_is_labelled_a_simulator_number(evidence_code):
    """Invariant 11. Stage 1 and the real stack are different baselines."""
    assert ("STAGE 1 SIMULATOR — NOT THE SHIPPING REAL-DATA STACK"
            in _prose(evidence_code))
    assert "05.run_ablation.py" in evidence_code, "the producing script is not named"


# ------------------------------------------- production vs candidate vs activation
def test_production_and_candidate_are_named_separately(evidence_code):
    """Two estimators exist. A page that names one name cannot be read correctly."""
    for s in ("PRODUCTION ESTIMATOR", "P3.5 PART 3 CANDIDATE", "ACTIVATION",
              "P3 / OLD canonical", "experimental hardened candidate"):
        assert s in evidence_code, f"Evidence.tsx no longer states {s!r}"


def test_the_activation_status_is_not_activated_and_says_why(evidence_code):
    assert "NOT ACTIVATED" in evidence_code
    assert re.search(r"did not outperform production consistently", _prose(evidence_code)), (
        "the non-activation reason is gone; NOT ACTIVATED without a reason reads "
        "like an oversight rather than a result")


@pytest.mark.skipif(not glob.glob("out/p35/part3/*/summary.json"),
                    reason="no P3.5 Part 3 artifact on disk")
def test_the_rendered_activation_verdict_is_the_artefacts_own_string(evidence_code):
    """The view transcribes this one, so pin it to the artefact it came from.

    Transcription is the weaker arrangement and it is acknowledged in the source:
    there is no `/api/p35/status` endpoint yet. Until there is, this test is what
    makes the transcription safe -- a rerun that flips the verdict fails here
    instead of leaving a stale word on the page.
    """
    path = sorted(glob.glob("out/p35/part3/*/summary.json"))[-1]
    d = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    assert d["activation_decision"] in evidence_code, (
        f"Evidence.tsx does not carry the artefact's activation_decision "
        f"({d['activation_decision']!r} in {path})")
    assert d["production_estimator_status"] in evidence_code
    assert pathlib.Path(path).as_posix() in evidence_code, (
        "the artefact path is not named next to the transcribed words")


@pytest.mark.skipif(not glob.glob("out/p35/part3/*/summary.json"),
                    reason="no P3.5 Part 3 artifact on disk")
def test_the_registry_agrees_that_the_candidate_is_not_in_the_path():
    """The page and the registry must not be two opinions about one fact."""
    from xray import registry as reg

    by = {e.component: e for e in reg.registry()}
    e = by["hardened_estimator_candidate"]
    assert e.production is False
    assert e.status == reg.DISABLED
    assert e.reason, "a non-production entry must say why"
    assert e.validation is not None and e.validation.available is True, (
        "measured-and-rejected must not read as never-checked")
    assert e.validation.result == reg.NO_ROBUST_IMPROVEMENT
    assert e.validation.fingerprint, "metrics with no fingerprint are untraceable"


# ------------------------------------------------------- the limits stay on screen
def test_the_store_reserve_identifiability_limit_is_stated(evidence_code):
    """Invariant 4's consequence, in words, on the page.

    Without it a reader takes the deployable-energy number for a state-of-charge
    reading, which is the one substitution this project's whole reporting
    discipline exists to stop.
    """
    assert ("Public telemetry does not robustly identify the store/reserve "
            "decomposition.") in _prose(evidence_code)


def test_no_claim_of_real_data_energy_validation(evidence_code):
    """There is no public battery channel, so no real energy validation exists.

    The real held-out work scores future OBSERVABLE speed. A page that says
    "validated on real data" about energy would be claiming a measurement that
    cannot be taken -- and the Part 3 candidate's restored synthetic accuracy is
    exactly the number most likely to be written up that way.
    """
    banned = {
        r"validated\s+(?:on|against)\s+real": "a real-data validation claim",
        r"real[-\s]data\s+energy\s+validation": "a real energy-validation claim",
        r"(?:real|measured|true)\s+(?:battery|energy)\s+ground\s*[-\s]?truth\s+"
        r"(?:is\s+)?available": "a claim that real energy truth exists",
        r"\benergy\s+accuracy\s+(?:on|against)\s+real\b": "a real energy-accuracy claim",
        r"\bPASSED\b\s+on\s+real": "a passing real verdict",
    }
    for pattern, what in banned.items():
        m = re.search(pattern, evidence_code, re.I)
        assert not m, f"Evidence.tsx contains {what}: {m.group(0)!r}"
    # And the positive statement is present, not merely the absence of the lie.
    assert "no such public channel exists" in _prose(evidence_code)


def test_the_real_held_out_verdict_is_still_rendered_by_the_backend_panels(evidence):
    """The negative verdict is a backend string, not prose this file can soften."""
    assert "EnergyStatusPanel" in evidence and "ValidationPanel" in evidence
    panel = _src(P3_PANEL)
    assert "real_validation" in panel and "e.headline" in panel
    assert ("CURRENT INFERRED ENERGY DOES NOT ADD ROBUST HELD-OUT PREDICTIVE VALUE"
            in panel)


# --------------------------------------------------- per-race regulation provenance
def test_the_missing_regulation_provenance_is_shown_as_missing(evidence_code):
    """Four of five race artefacts carry no `regulation` block, and the page says
    so by name. The harvest cap they used is correct -- this is a provenance
    claim, not a correctness one, and conflating the two would either frighten a
    reader off good numbers or imply provenance that is not there."""
    races = sorted(glob.glob("out/races/*.json"))
    if not races:
        pytest.skip("no race artefacts on disk")
    have, lack = [], []
    for p in races:
        d = json.loads(pathlib.Path(p).read_text(encoding="utf-8"))
        (have if d.get("regulation") else lack).append(d.get("circuit") or p)
    if not lack:
        pytest.skip("every race artefact now carries a regulation block")
    assert ("Regulation provenance is published for one race, not five."
            in _prose(evidence_code))
    # The circuits without provenance are named, so the absence is checkable.
    names = {"Spa-Francorchamps": "Spa", "Zandvoort": "Zandvoort",
             "Monte Carlo": "Monaco", "Silverstone": "Silverstone"}
    for circuit in lack:
        shown = names.get(circuit, circuit)
        assert shown in evidence_code, (
            f"{circuit} has no regulation provenance and the page does not say so")
    for circuit in have:
        assert circuit in evidence_code, (
            f"{circuit} does carry provenance and the page does not credit it")

# ---------------------------------------------- the replay panel renders no verdict
def test_the_replay_panel_renders_no_driver_action_verdict():
    """`actual_action` is a position delta relabelled as intent, and the panel's
    green/amber "matched the recommendation: yes/no" graded P2 against that
    relabelling. The delta moves for pit stops, retirements ahead, penalties,
    incidents, traffic and safety cars, and a driver who attacked and failed was
    recorded as having held -- so the verdict was unsupported in both directions.
    Banned by pattern in the rendered source, because the legacy fields are still
    on the payload and re-reading one is a one-line regression.
    """
    code = _strip_comments(_src(P3_PANEL))
    for field in ("actual_action", "matches_recommendation"):
        assert f"replay.{field}" not in code, (
            f"P3Panel.tsx reads replay.{field}, a legacy driver-action verdict")
        assert field not in code, (
            f"P3Panel.tsx still mentions {field} outside a comment")
    for banned in ("matched the recommendation", "what the driver actually did",
                   "MATCHED", "DIVERGED", "followed", "disobeyed"):
        assert banned not in code, f"P3Panel.tsx renders the verdict {banned!r}"


def test_the_replay_panel_reads_the_observed_outcome_and_counterfactual_fields():
    """The replacement must be the backend's own string, not a re-derivation."""
    code = _strip_comments(_src(P3_PANEL))
    assert "replay.observed_outcome" in code, "the observed outcome is not rendered"
    assert "replay.observed_outcome_basis" in code, "the outcome's basis is not shown"
    prose = _prose(code)
    assert "unresolved from historical telemetry" in prose, (
        "the counterfactual is not reported as unresolved")
    assert "not observed" in prose, "the driver's choice is not marked unobserved"
    # No re-derivation of the delta into a word, and no right/wrong colouring of
    # the outcome: `observed_outcome` arrives as a finished string.
    assert not re.search(r"observed_position_delta\s*[<>]", code), (
        "P3Panel.tsx grades the position delta client-side")
    assert not re.search(r"observed_outcome[^\n]*\?\s*C\.green", code), (
        "P3Panel.tsx colours the observed outcome as right/wrong")


def test_the_two_views_say_the_same_thing_about_the_counterfactual():
    """Situations.tsx (Agent B's) and the replay panel are one argument. A panel
    that still implied a verdict while the table above it refused to print one
    was the disagreement this guard pins shut."""
    panel = _prose(_strip_comments(_src(P3_PANEL)))
    situations = _prose(_strip_comments(_src(APP / "views" / "Situations.tsx")))
    for claim in ("unresolved from historical telemetry",
                  "position not published for this window"):
        assert claim in panel, f"the replay panel no longer says {claim!r}"
        assert claim in situations, f"Situations.tsx no longer says {claim!r}"
