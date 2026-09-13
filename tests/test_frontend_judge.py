"""The judge panel still computes nothing, and still says where it came from.

Same discipline as `test_frontend_p2.py` and `test_frontend_p4.py`, applied to
the one genuinely new surface: a panel rendering text a language model wrote.
The extra risk here is not arithmetic drift, it is a generated string reading
as a measurement, so the provenance banner is asserted as hard as the maths
bans are.

Every read passes `encoding="utf-8"`. Omitting it reads cp1252 on Windows and
has broken this suite three times (`docs/dev_readme.md` section 7 item 1).
"""
from __future__ import annotations

import pathlib
import re

APP = pathlib.Path(__file__).resolve().parent.parent / "simulation" / "app" / "src"
PANEL = APP / "components" / "JudgePanel.tsx"
SITUATIONS = APP / "views" / "Situations.tsx"
API = APP / "lib" / "api.ts"


def _strip_comments(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return "\n".join(l for l in src.splitlines() if not l.strip().startswith("//"))


def test_the_panel_exists_and_is_wired_into_situations():
    assert PANEL.exists()
    src = SITUATIONS.read_text(encoding="utf-8")
    assert "JudgePanel" in src
    assert "api.judge(" in src


def test_judge_panel_contains_no_decision_mathematics():
    """Reuses P2's banned-pattern list so the panel cannot drift from the other
    views, plus two bans specific to a verdict surface: the frontend must not
    derive a verdict or a score."""
    from tests.test_frontend_p2 import _ts_sources  # noqa: F401  (shared list)

    banned = {
        r"Math\.exp\s*\(\s*-": "a logistic / sigmoid",
        r"1\s*/\s*\(\s*1\s*\+\s*Math\.exp": "a logistic",
        r"delta_v\s*=": "a delta_v computation",
        r"value_attack\s*=\s*[^=]": "an expected-value computation",
        r"\bbellman\b": "a Bellman step",
        r"decision_margin\s*=\s*[^=]": "a margin computation",
        # `[^={]` rather than `[^=]`: a JSX attribute is `verdict={expr}`,
        # which passes a value down, and banning that would ban rendering it.
        # What must not appear is an assignment -- `verdict = ...` -- which is
        # the frontend deciding a verdict for itself.
        r"\bverdict\s*=\s*[^={]": "a client-side verdict derivation",
        r"\bscore\s*=\s*[^={]": "a client-side score derivation",
        r"reduce\s*\(": "an aggregation over scores",
    }
    src = _strip_comments(PANEL.read_text(encoding="utf-8"))
    for pattern, what in banned.items():
        assert not re.search(pattern, src), f"JudgePanel.tsx contains {what}"


def test_judge_panel_never_types_the_pass_model_identifier():
    """`test_frontend_p2._ts_sources` bans the bare pass-probability identifier
    across every .ts/.tsx outside src/three/, and its comment stripper removes
    comments but NOT string literals -- so the token must be absent from JSX
    prose too. The rubric axis is named `pass_model_honesty`; anything the panel
    says about that model arrives from the server in `note` or `why_wrong`."""
    for path in (PANEL, API):
        src = path.read_text(encoding="utf-8")
        assert "p_pass" not in src, (
            f"{path.name} types the banned pass-probability identifier; use the "
            f"server-supplied text instead")


def test_the_provenance_banner_is_unconditional():
    """A provenance notice behind a hover or a flag is one most readers never
    see. Same precedent as `ReplayPanel`'s `replay.label`
    (`tests/test_frontend_p4.py::test_scenario_walkthrough_renders_the_permanent_disclaimer_unconditionally`)."""
    src = PANEL.read_text(encoding="utf-8")
    assert "{verdict.label}" in src
    assert "verdict.model_id" in src
    assert "verdict.prompt_version" in src
    head = src.split("{verdict.label}")[0][-260:]
    assert "title=" not in head, "the banner is behind a tooltip"
    assert "&&" not in head.split("return (")[-1], "the banner is conditional"


def test_the_panel_states_the_judge_did_not_see_the_outcome():
    """The one thing a reader must not conclude is that a verdict is a second
    opinion about whether the driver agreed."""
    src = PANEL.read_text(encoding="utf-8")
    assert "not shown what the driver" in src


def test_the_panel_holds_no_verdict_label_map():
    """`verdict_label` and `verdict_tone` are server fields. A local enum ->
    string map would be the frontend deciding what a verdict means."""
    src = _strip_comments(PANEL.read_text(encoding="utf-8"))
    assert "verdict.verdict_label" in src
    for enum_name in ("JUSTIFIED", "OVERCLAIMED", "UNSUPPORTED",
                      "SHOULD_HAVE_REFUSED", "CORRECTLY_REFUSED"):
        assert enum_name not in src, (
            f"JudgePanel.tsx names the verdict enum {enum_name}; render "
            f"verdict_label instead")


def test_situations_reads_the_verdict_and_derives_nothing():
    src = _strip_comments(SITUATIONS.read_text(encoding="utf-8"))
    assert "judge?.verdicts?.[judgeKey(s)]?.verdict_label" in src
    # The existing P4 bans must still hold with the new column in place.
    for pattern, what in {
        r"actual_action\s*=\s*[^=]": "a client-side actual_action derivation",
        r"matches_recommendation\s*=\s*[^=]": "a client-side match derivation",
        r"\bverdict\s*=\s*[^={]": "a client-side verdict derivation",
    }.items():
        assert not re.search(pattern, src), f"Situations.tsx contains {what}"


def test_a_missing_judge_never_blocks_the_table():
    """Verdicts are optional commentary that most pairs do not have. A fetch
    failure resolving into the view's error state would make the whole tab
    unusable on a checkout where nobody has run the batch."""
    src = SITUATIONS.read_text(encoding="utf-8")
    block = src.split("api.judge(")[1][:400]
    assert "setJudge(null)" in block
    assert "setErr" not in block


def test_the_api_types_label_verdicts_as_generated_commentary():
    src = API.read_text(encoding="utf-8")
    assert "GENERATED COMMENTARY" in src
    assert "is_measurement: false" in src
    assert "api.judge" in src or "judge:" in src
