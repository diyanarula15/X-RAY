"""The two cars in the 3-D replay must be tellable apart at racing speed.

The audit finding this guards: `Car.tsx` gave CHASER and TARGET the same
'#DCDCE4' shell and differed only in the accent colour applied to three small
parts -- the spine stripe (a 60 mm box), the front wing plane and the rear wing
plane. In the duel and chase cameras those parts are a few pixels wide, so the
identity rested almost entirely on one hue step (amber '#FFC300' vs red
'#E10600'), which is exactly the cue that fails at speed, fails in peripheral
vision, and fails outright for a red-green colour-blind viewer.

So the property asserted here is *redundancy*: at least two independent visual
channels must separate the roles, and at least one of them must not be colour.
These are static checks against the TSX source in the style of
`tests/test_frontend_p2.py` -- they do not render the scene; `tsc -b` and
`vite build` are run separately and reported in the audit.

The second half of the file is a containment check. Car identity is a
presentation concern, and the repair must not have smuggled motion, coordinate,
interpolation, physics or camera logic into the car component to get it.
"""
from __future__ import annotations

import pathlib
import re

APP = pathlib.Path(__file__).resolve().parent.parent / "simulation" / "app" / "src"
CAR = APP / "three" / "Car.tsx"
SCENE = APP / "three" / "Scene.tsx"
THEME = APP / "lib" / "theme.ts"


def _src(path: pathlib.Path) -> str:
    # encoding="utf-8" is not optional: a source scan without it reads cp1252 on
    # Windows and has broken this suite three times on a single non-ASCII byte.
    return path.read_text(encoding="utf-8")


def _strip_comments(src: str) -> str:
    """Comments are prose. A ban that fires on prose is a ban nobody keeps."""
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return "\n".join(l for l in src.splitlines()
                     if not l.strip().startswith("//"))


# --------------------------------------------------------------------------
# identity
# --------------------------------------------------------------------------

def test_the_car_component_takes_an_explicit_role():
    """Role must be a prop, not something inferred from the accent colour.

    Inferring it from the colour is what made colour the single point of
    failure in the first place.
    """
    src = _strip_comments(_src(CAR))
    assert "CarRole" in src, "no role type on the car component"
    assert re.search(r"role\s*:\s*CarRole", src), "role is not a declared prop"
    assert "'CHASER'" in src and "'TARGET'" in src, "both roles must be named"


def test_the_scene_labels_which_car_is_which():
    src = _strip_comments(_src(SCENE))
    assert 'role="CHASER"' in src, "the subject car is not marked as the chaser"
    assert 'role="TARGET"' in src, "the rival car is not marked as the target"


def _role_branches(src: str) -> list[str]:
    """Every place the source forks on the role."""
    return re.findall(r"isChaser\s*\?[^\n]*", src) + \
        re.findall(r"role\s*===\s*'(?:CHASER|TARGET)'", src)


def test_at_least_two_independent_channels_separate_the_roles():
    """The core property. Colour is allowed to be one channel; it cannot be
    the only one, and the non-colour channel has to change real geometry or
    text, not a shade."""
    src = _strip_comments(_src(CAR))

    # 1. colour, applied to the whole shell rather than to trim. The old
    #    version hard-coded one shell colour for both cars.
    assert "shellColour" in src, "the shell colour does not depend on the role"
    assert not re.search(r"color:\s*ghost\s*\?\s*'#8A8A94'\s*:\s*'#DCDCE4'", src), \
        "the shell is still one fixed colour for both roles"
    assert re.search(r"color:\s*shellColour", src), \
        "the role colour is not applied to the body shell material"

    # 2. silhouette: the two roles must render different geometry, so the
    #    outline differs with the colour thrown away entirely.
    chaser_only = re.search(r"isChaser\s*\?\s*\(\s*\n\s*<mesh", src)
    assert chaser_only, "no role-conditional geometry: the outline is identical"

    # 3. a marker whose SHAPE differs, not just its colour
    shapes = set(re.findall(r"<(cone|torus|box|sphere|ring)Geometry", src))
    assert {"cone", "torus"} <= shapes, \
        "the role marker must differ in shape (cone vs ring), not only in hue"

    # 4. the role spelled out, on a camera-facing sprite so no rotation can
    #    hide it
    assert "<sprite" in src, "no camera-facing role label"
    assert "roleLabelTexture" in src and "fillText(role" in src, \
        "the label must print the role word itself"

    # and the non-colour channels must be driven by the role, not by anything else
    assert len(_role_branches(src)) >= 3, \
        f"only {len(_role_branches(src))} role forks; identity is too thin"


def test_the_identity_survives_the_cars_overlapping_on_screen():
    """Close proximity is the case the view exists for. A marker that is
    z-buffered away behind the other car's bodywork answers the question for
    one car and not the other, so marker and label are drawn depth-test-free
    with an explicit render order, and the two roles' markers sit at different
    heights so they do not merge into one blob."""
    src = _strip_comments(_src(CAR))
    assert src.count("depthTest: false") >= 2, \
        "marker and label must both survive being occluded by the other car"
    assert "renderOrder" in src, "no explicit draw order for the role overlays"
    heights = re.findall(r"position=\{\[0,\s*isChaser \?\s*([\d.]+)\s*:\s*([\d.]+)", src)
    marker_y = re.findall(r"<mesh position=\{\[0,\s*(0\.\d+),\s*0\]\}", src)
    assert heights or len(set(marker_y)) >= 2, \
        "the two roles' overlays sit at the same height and will merge when close"


def test_the_role_colours_come_from_the_theme_palette():
    """No invented colours. The accent is passed in from `lib/theme.ts`; the
    only derivation allowed is mixing toward the palette's own white."""
    car = _strip_comments(_src(CAR))
    theme = _src(THEME)
    assert "C.white" in car and "white: '#FFFFFF'" in theme
    scene = _strip_comments(_src(SCENE))
    assert "accent={C.amber}" in scene and "accent={C.red}" in scene, \
        "the identity colours must be the palette's, not literals in the scene"
    # no new six-digit literals beyond the greys and the background that were
    # already in the file before the repair
    allowed = {"#17171C", "#5C5C68", "#22222A", "#8A8A94", "#DCDCE4",
               "#39C6E0", "#3A3A44", "#07070A"}
    found = set(re.findall(r"#[0-9A-Fa-f]{6}", car))
    assert found <= allowed, f"new colour literals outside the palette: {found - allowed}"


# --------------------------------------------------------------------------
# containment: this was a visual change and nothing else
# --------------------------------------------------------------------------

BANNED = {
    # motion / physics
    r"\bvelocity\b": "a velocity term",
    r"\bacceler": "an acceleration term",
    r"\bphysics\b": "physics",
    r"\bmass\b": "a mass term",
    r"\bdrag\b": "a drag term",
    r"\bforce\b": "a force term",
    r"\bP_K\b|\bdeploy_kw\b|\bharvest_kw\b": "an energy channel",
    # coordinates / positioning taken over from the scene
    r"\bposeAt\b": "centreline pose resolution (Scene's job)",
    r"\bcurvature\b": "track curvature",
    r"Math\.atan2": "a heading computation",
    # `geo` as a bare word is the local name of the extruded body mesh and
    # predates the repair; what must stay out is the circuit geometry payload.
    r"geo\.(?:x|y|z|s|length|curvature|has_elevation)\b": "circuit geometry data",
    r"lib/api": "an import of the race payload types",
    # camera
    r"\buseThree\b": "camera/renderer access",
    r"\bcamera\b": "camera logic",
    r"\blookAt\b": "camera aiming",
    # interpolation of state over time
    r"\bdamp\b|\beasing\b|\bsmoothstep\b": "an easing helper",
}


def test_no_motion_coordinate_or_camera_logic_entered_the_car_component():
    src = _strip_comments(_src(CAR))
    for pattern, what in BANNED.items():
        assert not re.search(pattern, src), f"Car.tsx now contains {what}"


def test_the_only_frame_work_in_the_car_is_what_was_already_there():
    """Two `useFrame` bodies existed before the repair -- the car's
    position/heading copy and the wheel spin -- plus the trail rebuild. The
    identity channels are declarative geometry and must not have added a third
    kind of per-frame work.
    """
    src = _strip_comments(_src(CAR))
    assert src.count("useFrame(") == 3, \
        f"expected 3 useFrame bodies (car, wheel, trail), found {src.count('useFrame(')}"
    # the car's frame body still does exactly position.copy + rotation.set + spin
    body = src[src.index("useFrame((_, dt) => {"):]
    body = body[:body.index("});")]
    assert "position.copy(position)" in body and "rotation.set(0, heading, bank)" in body
    assert "spin.current +=" in body
    assert "*" in body  # speed / WHEEL_R * dt, unchanged
    assert "Math." not in body, "new mathematics in the car's frame callback"


def test_any_exponential_in_the_car_is_frame_rate_easing():
    """Mirrors tests/test_frontend_p2.py's ban: an exponential here is only
    ever `Math.exp(-dt * k)` frame-rate easing, never a logistic."""
    src = _strip_comments(_src(CAR))
    for m in re.finditer(r"Math\.exp\(([^)]*)\)", src):
        arg = m.group(1)
        assert "dt" in arg or "delta" in arg, \
            f"Car.tsx has a non-easing exponential: Math.exp({arg})"


def test_the_repair_did_not_touch_the_energy_trail():
    """The trail is the other car-attached visual and belongs to a different
    concern (energy flow). Identity work must leave it alone."""
    src = _src(CAR)
    assert "export function EnergyTrail" in src
    assert "vertexColors: true" in src and "AdditiveBlending" in src
