"""Cockpit: the canonical recommendation is P2's, and P1 is labelled as legacy.

The audit that produced this guard: across 3,298 decision points the P1 per-lap
`attack` flag disagrees with canonical P2 on 961 of them (29%). Cockpit used to
render that flag as bare prose -- 'ATTACK - V(attack) exceeds V(hold)' in green,
the same styling as the headline call -- and fell back to it in the 46px headline
slot whenever `/p2` failed. Either way a reader saw a P1 answer in the place
reserved for the recommendation, wrong roughly one time in three, with nothing on
screen naming the layer.

Static source checks, in the style of `test_frontend_p2.py` and
`test_frontend_p4.py`: they read the TSX and assert on wording and on the absence
of banned patterns. They do not render the page; `tsc -b` / `vite build` are run
separately.
"""
from __future__ import annotations

import pathlib
import re

APP = pathlib.Path(__file__).resolve().parent.parent / "simulation" / "app" / "src"
COCKPIT = APP / "views" / "Cockpit.tsx"


def _strip_comments(src: str) -> str:
    """Comments are prose, and this file's comments quote the wording they
    replaced. A ban that fires on its own rationale is a ban nobody keeps."""
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return "\n".join(l for l in src.splitlines() if not l.strip().startswith("//"))


def _src() -> str:
    # encoding pinned: invariant 14 -- a source-scanning test without it reads
    # cp1252 on Windows and dies on the first non-ASCII character.
    return _strip_comments(COCKPIT.read_text(encoding="utf-8"))


def test_the_legacy_p1_section_is_framed_as_legacy_and_diagnostic():
    src = _src()
    assert "LEGACY P1 TRACE" in src
    assert "historical per-lap diagnostic" in src
    assert "not the canonical" in src
    # the measurement that justifies the framing travels with it
    assert "961 of" in src and "3,298" in src


def test_the_per_lap_p1_flag_is_named_for_its_layer():
    """The three prose strings stay -- the information is genuinely useful -- but
    none of them may stand alone as an unattributed verdict."""
    src = _src()
    i = src.index("V(attack) exceeded V(hold)")
    window = src[max(0, i - 400):i]
    assert "legacy P1 flag" in window, (
        "the per-lap P1 verdict is rendered without naming P1")
    assert "Diagnostic only" in src
    # the old unattributed present-tense wording must be gone
    assert "'ATTACK — V(attack) exceeds V(hold)'" not in src


def test_no_p1_value_is_labelled_as_the_canonical_recommendation():
    src = _src()
    for phrase in ("X-RAY recommendation", "the X-RAY call", "XRAY recommendation"):
        assert phrase not in src, f"Cockpit labels something {phrase!r}"
    # "the call" was the d3 annotation on the P1 row; it now names its layer.
    assert "the call — lap" not in src
    assert "legacy P1 per-lap call" in src
    # P1's confidence sat unlabelled beside the P2 headline.
    assert "`confidence ${pct(row.confidence" not in src
    assert "legacy P1 confidence" in src
    # and the big number in the right rail is no longer a bare "MODEL" headline
    assert "<SectionTitle>MODEL</SectionTitle>" not in src
    assert "LEGACY P1 MODEL" in src
    assert "P1 trace confidence" in src


def test_the_headline_recommendation_comes_from_p2():
    src = _src()
    assert "p2.decision === 'ATTACK'" in src
    assert "api.p2(" in src
    # the headline renders the P2-derived value, not a P1 row field
    assert "{attack == null ? 'NO CANONICAL RECOMMENDATION'" in src


def test_the_headline_declines_rather_than_falling_back_to_p1():
    """Invariant 3, applied to presentation: with no P2 there is no canonical
    recommendation, and the honest output is a refusal. The old expression
    `p2 ? p2.decision === 'ATTACK' : !!row?.attack` promoted P1 into the
    headline silently, and P1 names a different action 29% of the time."""
    src = _src()
    assert "!!row?.attack" not in src, "the headline still falls back to the P1 flag"
    assert "const attack: boolean | null = p2 ? p2.decision === 'ATTACK' : null;" in src
    assert "P2 unavailable" in src
    # the refusal state must be reachable in the render, not just computed
    assert "attack == null" in src


def test_cockpit_adds_no_decision_arithmetic():
    """Same banned list as `test_frontend_p2.py` / `test_frontend_p4.py`, so the
    relabelling cannot have smuggled in a second opinion. Presentation only: the
    fix is wording and styling, no new comparison of P1 against P2 in TS."""
    banned = {
        r"Math\.exp\s*\(\s*-": "a logistic / sigmoid",
        r"1\s*/\s*\(\s*1\s*\+\s*Math\.exp": "a logistic",
        r"\bp_pass\b": "a pass-probability formula",
        r"delta_v\s*=": "a delta_v computation",
        r"value_attack\s*=\s*[^=]": "an expected-value computation",
        r"value_hold\s*=\s*[^=]": "an expected-value computation",
        r"\bbellman\b": "a Bellman step",
        r"expected_regret\s*=\s*[^=]": "a regret computation",
        r"decision_margin\s*=\s*[^=]": "a margin computation",
        r"deployment_budget_mj\s*=\s*[^=]": "a budget choice",
        # a frontend-invented agreement flag would be a third opinion about
        # which layer is right, which is the backend's call, not the view's.
        r"(?:agrees|disagrees|matches)\s*=\s*[^=]": "a client-side P1/P2 comparison",
        r"sel\.attack\s*[=!]==?\s*p2": "a client-side P1-vs-P2 comparison",
        r"row\?*\.attack\s*[=!]==?\s*\(*p2": "a client-side P1-vs-P2 comparison",
    }
    src = _src()
    for pattern, what in banned.items():
        assert not re.search(pattern, src), f"Cockpit.tsx contains {what}"
    # no comparison against a threshold invented here either
    assert not re.search(r"\.tau\s*[<>]", src), "Cockpit compares q against tau itself"


def test_the_p1_verdict_is_visually_subdued_relative_to_the_p2_headline():
    """It was green -- `color: sel.attack ? C.green : C.gray` -- which is the
    colour the canonical ATTACK headline uses, so a green P1 line read as a live
    recommendation."""
    src = _src()
    assert "sel.attack ? C.green" not in src
    i = src.index("legacy P1 flag for lap")
    assert "C.gray" in src[max(0, i - 260):i]
    # the canonical headline keeps its colour coding
    assert "attack ? C.green : C.amber" in src
