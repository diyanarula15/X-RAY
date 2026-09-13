"""Replay (Theatre) says what it is, and says nothing it cannot support.

Same discipline as `test_frontend_p2.py` / `test_frontend_p4.py`: grep the TSX
and assert the wording, because every failure this file guards was a sentence
rather than a number.

Four of them shipped at once in `Theatre.tsx`:

* `subjectAhead: d > 0` — the inverse of its own sign convention. `d = rival.s -
  subject.s`, so `d > 0` is the RIVAL ahead; the header named the subject ahead
  while the 3D scene, reading `s` directly, drew it trailing.
* "Yours is known; theirs is reconstructed from speed" — true of the simulator,
  false of every session this view opens. These are historical public-data races
  and the payload marks no car as internally known; both bars are the same
  particle-filter output off the same public speed trace.
* "The store has reached the floor" — a speed trace pins flows exactly and
  absolute level only up to an unidentified constant, and store vs reserve is
  not separately identifiable from public telemetry at all.
* `gap.seconds < 1 && 'Override eligible'` — the frontend issuing a regulatory
  ruling from one interpolated sample. The 1.000 s Manual Override rule is
  decided at a zone detection point against the regulation in force; the car
  payload carries no eligibility field (`regulation.zone_eligibility` is
  deployment-ZONE geometry, a different rule), so there is nothing to render.
"""
from __future__ import annotations

import pathlib
import re
import shutil
import subprocess

import pytest

APP = pathlib.Path(__file__).resolve().parent.parent / "simulation" / "app" / "src"
THEATRE = APP / "views" / "Theatre.tsx"
APP_TSX = APP / "App.tsx"


def _src(path: pathlib.Path) -> str:
    # encoding is explicit: the default reads cp1252 on Windows and has broken
    # this suite three times on a single non-ASCII character.
    return path.read_text(encoding="utf-8")


def _strip_comments(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return "\n".join(l for l in src.splitlines() if not l.strip().startswith("//"))


# ---------------------------------------------------------------- A1 / A2

def test_the_gap_sign_convention_is_named_for_the_rival_not_the_subject():
    body = _strip_comments(_src(THEATRE))
    assert "let d = rS.s - sS.s;" in body, "the sign convention moved; re-read this test"
    assert "rivalAhead: d > 0" in body
    assert "subjectAhead" not in body, (
        "`subjectAhead: d > 0` is the inverse of the convention: d = rival.s - "
        "subject.s, so d > 0 is the RIVAL ahead")


def test_position_labels_follow_the_sign_and_not_the_other_way_round():
    body = _strip_comments(_src(THEATRE))
    i = body.index("gap.rivalAhead")
    window = body[i:i + 260]
    # rival ahead => subject BEHIND, rival AHEAD; the other branch is the mirror.
    assert re.search(r"\{\s*subject:\s*'BEHIND',\s*rival:\s*'AHEAD'", window)
    assert re.search(r"\{\s*subject:\s*'AHEAD',\s*rival:\s*'BEHIND'", window)


def test_roles_are_constants_and_are_never_reassigned_from_position():
    src = _src(THEATRE)
    assert "const ROLE_SUBJECT = 'CHASER'" in src
    assert "const ROLE_RIVAL = 'TARGET'" in src
    body = _strip_comments(src)
    # No code path may write a role. A pass changes POSITION, never identity:
    # before a pass the chaser is 0.4 s behind the target, after it the chaser is
    # 0.2 s ahead of the target -- still the chaser.
    for banned in (r"ROLE_SUBJECT\s*=\s*[^'\s]", r"ROLE_RIVAL\s*=\s*[^'\s]",
                   r"'CHASER'\s*:", r"\?\s*'CHASER'", r"\?\s*'TARGET'",
                   r"rivalAhead\s*\?\s*ROLE", r"ROLE_\w+\s*=\s*\w+Ahead"):
        assert not re.search(banned, body), f"a role is being derived from position: {banned}"
    # and the labels the energy bars carry are the role constants, not 'YOU'
    assert "${ROLE_SUBJECT} — ${subject?.driver" in src
    assert "${ROLE_RIVAL} — ${rival?.driver" in src
    assert "YOU —" not in src


# ---------------------------------------------------------------- A3

def test_track_state_words_exist_and_are_not_strategy_words():
    body = _strip_comments(_src(THEATRE))
    for word in ("CHASING", "DEFENDING POSITION", "SIDE-BY-SIDE"):
        assert word in body, f"the track-state vocabulary lost {word}"
    # The canonical strategy vocabulary is ATTACK / HOLD and it is produced by
    # the solver. A third strategy word invented in the UI would be read as a
    # call the engine never made -- and `DEFEND` is the one that would reach the
    # solver's vocabulary by accident. \bDEFEND\b does not match DEFENDING.
    assert not re.search(r"\bDEFEND\b", body), "DEFEND must never be a strategy value"

    # This used to ban ATTACK/HOLD from Theatre outright, on the grounds that
    # the call belongs to Cockpit. Replay now shows the call for the decision
    # point the clock has passed -- it ran with no recommendation on screen at
    # all, under a badge that disclaimed one. The hazard the ban existed for is
    # unchanged and still checked below: Theatre may DISPLAY the solver's word,
    # never DERIVE one.
    #
    # The track-state vocabulary is the thing that would drift into a strategy
    # call by accident, so that block specifically must stay clean of both.
    i = body.index("const position")
    state_block = body[i:body.index("windowsOverlap")]
    assert not re.search(r"\bATTACK\b|\bHOLD\b", state_block), (
        "the track-state vocabulary has taken a strategy value; CHASING and "
        "DEFENDING POSITION describe the track, they are not calls")

    # Every strategy word in the file is a read of the server's own field.
    for m in re.finditer(r"\bATTACK\b|\bHOLD\b", body):
        window = body[max(0, m.start() - 90):m.start()]
        assert "call.recommendation" in window, (
            "a strategy word in Theatre that is not a read of "
            "`call.recommendation` -- the call must come from the bundle, not "
            "from anything computed in this view")


# ---------------------------------------------------------------- A4

def test_replay_identifies_itself_as_recorded_history():
    src = _src(THEATRE)
    assert "HISTORICAL REPLAY" in src
    assert "Recorded telemetry" in src
    # and it must not let the advisory recommendation read as an executed action
    assert "X-RAY recommendation is advisory here." in src
    assert "does not execute" in src and "counterfactual branch" in src
    # the label is body text, not a tooltip on something else
    i = src.index("HISTORICAL REPLAY")
    assert "title=" not in src[max(0, i - 300):i], "the replay label must not be a tooltip"


# ---------------------------------------------------------------- A5

def test_no_asymmetric_known_versus_reconstructed_energy_wording():
    body = _strip_comments(_src(THEATRE))
    banned = (
        r"[Yy]ours is known",
        r"[Tt]heirs is reconstructed",
        r"\bis known\b",
        r"\breconstructed\b",
    )
    for pattern in banned:
        assert not re.search(pattern, body), (
            f"asymmetric energy wording is back ({pattern}): these sessions are "
            "historical public data and the payload marks no car as internally known")


def test_both_cars_are_labelled_estimated_and_telemetry_derived():
    src = _src(THEATRE)
    assert "Estimated usable energy" in src
    assert "Telemetry-derived" in src or "telemetry-derived" in src


def test_the_uncertainty_band_stays_on_screen_for_both_cars():
    body = _strip_comments(_src(THEATRE))
    # p10/p90 come from the payload (usable_p10 / usable_p90) and are passed to
    # both bars. A point estimate without its band is the one readout this repo
    # does not ship.
    assert len(re.findall(r"p10=\{", body)) >= 2
    assert len(re.findall(r"p90=\{", body)) >= 2
    assert "p10–p90" in _src(THEATRE) or "p10-p90" in _src(THEATRE)


# ---------------------------------------------------------------- A6

def test_the_cut_out_wording_is_evidence_aware_not_deterministic():
    src = _src(THEATRE)
    assert "Deployment cut-out detected." in src
    assert "consistent with the car being near its deployable-energy floor" in src
    assert ("Public telemetry does not uniquely identify store and reserve "
            "separately.") in src
    body = _strip_comments(src)
    for pattern in (r"has reached the floor", r"store spent", r"battery is empty",
                    r"\bis empty\b", r"exactly zero", r"nothing left to deploy"):
        assert not re.search(pattern, body), (
            f"deterministic cut-out wording is back ({pattern}): a cut-out is "
            "evidence consistent with a floor, not a measurement of one")


# ---------------------------------------------------------------- A7

def test_no_frontend_inference_of_manual_override_eligibility():
    body = _strip_comments(_src(THEATRE))
    assert "Override eligible" not in body, (
        "a definitive eligibility badge is a regulatory ruling the frontend "
        "cannot make")
    # no gap-versus-one-second comparison of any spelling
    for pattern in (r"gap\.seconds\s*[<>]=?\s*1\b", r"seconds\s*[<>]=?\s*1\.0",
                    r"gap_s\s*[<>]=?\s*1\b", r"MOM_GAP", r"1\.000\s*s"):
        assert not re.search(pattern, body), f"eligibility inferred from a gap: {pattern}"
    assert "eligibility not established" in body.lower()


def test_theatre_implements_no_regulation_arithmetic():
    """Reuses the P2 banned-pattern list so the views cannot drift apart."""
    body = _strip_comments(_src(THEATRE))
    banned = {
        r"Math\.exp\s*\(\s*-": "a logistic / sigmoid",
        r"\bp_pass\b": "a pass-probability formula",
        r"delta_v\s*=": "a delta_v computation",
        r"\bP_MGUK\b|\bp_harv_max\b\s*[*/]": "a regulation power term in TypeScript",
        r"0\.5\s*\*\s*rho": "a drag equation",
        r"\bCdA\s*=": "a drag-area computation",
    }
    for pattern, what in banned.items():
        assert not re.search(pattern, body), f"Theatre.tsx contains {what}"


# ---------------------------------------------------------------- A8

def test_a_pair_swap_re_ranges_the_clock_on_every_exit_path():
    body = _strip_comments(_src(APP_TSX))
    i = body.index("async function applyPair(")
    fn = body[i:body.index("async function load(")]
    # Three exits: one car missing, no shared window, and the healthy path. All
    # three must leave the clock describing the CURRENT pair -- `setRange` also
    # resets `raceTime`, which is the clamp.
    assert len(re.findall(r"setRange\(", fn)) >= 3, (
        "an exit path leaves the previous pair's time range on the clock, which "
        "is what made a newly-picked car sit frozen with a null gap")
    assert "timeRange(sc)" in fn and "timeRange(rc)" in fn
    assert "no overlapping telemetry" in fn, "the no-overlap case must be reported"


def test_setrange_resets_the_playback_time():
    """The clamp lives in the store; assert it rather than assuming it."""
    store = _strip_comments(_src(APP / "store" / "playback.ts"))
    assert re.search(r"setRange:\s*\(tRange\)\s*=>\s*set\(\{\s*tRange,\s*raceTime:\s*tRange\[0\]",
                     store)


def test_the_picker_keeps_the_requested_codes_when_a_car_fails_to_load():
    body = _strip_comments(_src(APP_TSX))
    i = body.index("function pick(")
    fn = body[i:i + 900]
    assert "usePlayback.getState()" in fn and "st.subject" in fn and "st.rival" in fn, (
        "`subject?.driver ?? ''` alone makes every later pick return early once "
        "one car is unavailable, and the picker goes silently dead")


def test_the_empty_option_exists_so_a_blank_value_has_something_to_select():
    src = _src(APP_TSX)
    assert '<option value="">' in src, (
        "with value='' and no matching option the browser shows the first driver "
        "while React holds '', so selecting that driver fires no change event")


def test_replay_reports_an_unavailable_car_and_an_empty_shared_window():
    src = _src(THEATRE)
    assert "PAIR NOT PLAYABLE" in src
    assert "NO SHARED TELEMETRY WINDOW" in src
    assert "windowsOverlap" in src


# ------------------------------------------------- behavioural check (optional)

DETERMINISTIC_CASES = """
subject.s=100 rival.s=110 -> subject BEHIND / rival AHEAD
subject.s=110 rival.s=100 -> subject AHEAD  / rival BEHIND
wrap s=5 vs 4995 (L=5000) -> subject AHEAD  / rival BEHIND
wrap s=4995 vs 5 (L=5000) -> subject BEHIND / rival AHEAD
"""

_HARNESS = r"""
const fs = require('fs');
const src = fs.readFileSync(process.argv[2], 'utf8');
function body(a, b) {
  const i = src.indexOf(a), j = src.indexOf(b, i);
  if (i < 0 || j < 0) throw new Error('marker not found: ' + a);
  return src.slice(i, j + b.length);
}
const gapFn = new Function('sS', 'rS', 'L',
  body('if (!sS || !rS || !sS.onTrack', 'rivalAhead: d > 0 };'));
const posFn = new Function('gap', 'CAR_LENGTH_M',
  body('if (!gap) return null;',
       "{ subject: 'AHEAD', rival: 'BEHIND', state: 'DEFENDING POSITION' };"));
const cases = [[100, 110, 'BEHIND'], [110, 100, 'AHEAD'],
               [5, 4995, 'AHEAD'], [4995, 5, 'BEHIND']];
for (const [ss, rs, want] of cases) {
  const p = posFn(gapFn({ s: ss, v: 70, onTrack: true },
                        { s: rs, v: 70, onTrack: true }, 5000), 5.6);
  if (p.subject !== want) throw new Error(`${ss} vs ${rs}: ${p.subject} != ${want}`);
  if (p.rival === p.subject) throw new Error(`${ss} vs ${rs}: both cars ${p.rival}`);
}
console.log('ok');
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="no node on this machine")
def test_the_deterministic_ahead_behind_cases_by_executing_the_real_source(tmp_path):
    """Not a reimplementation: the two memo bodies are lifted verbatim out of
    `Theatre.tsx` and run, so this fails if the shipped expression flips sign.
    Cases: """ + DETERMINISTIC_CASES
    harness = tmp_path / "check.js"
    harness.write_text(_HARNESS, encoding="utf-8")
    r = subprocess.run(["node", str(harness), str(THEATRE)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "ok" in r.stdout
