"""Layer hygiene: what may import what, and which numbers are allowed to differ.

`docs/pipeline_layers.md` assigns every module in `xray/` to a layer. Two rules
from it are enforceable and neither was enforced before:

  1. An INFERENCE module may not import a SIMULATION module.
  2. A physical quantity has one definition, or a registered reason to have two.

Rule 1 generalises `test_estimator_is_blind`, which parses exactly one file.
Rule 2 exists because the same quantity currently has three values across the
three inference stacks -- reference mass 768 vs 790, fuel burn 1.4 vs 1.15 vs not
modelled, reserve sigma 1.0e5 vs 5.0e5 -- and nothing noticed. The registry
below does not fix that. It stops it growing, and makes removing an entry a
deliberate act with a test to update.

Every file read here passes `encoding="utf-8"`. Omitting it has broken this
suite three times (`docs/dev_readme.md` section 7 item 1); on Windows the
default is cp1252 and `realfit.py` has a non-ASCII character in its docstring.
"""
from __future__ import annotations

import ast
import inspect
import pathlib

import pytest

XRAY = pathlib.Path(__file__).resolve().parent.parent / "xray"

# Generates synthetic ground truth, including the hidden energy state.
SIMULATION = {"sim", "track", "policy", "config"}

# Turns a feed into a belief. Must work regardless of where the feed came from.
INFERENCE = {"estimator", "realfit", "analysis", "balance", "setmem", "modes",
             "pipeline", "rbpf", "deadband", "strategy", "pooling"}


def _imported_module_names(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[-1])
        elif isinstance(node, ast.Import):
            names.update(a.name.split(".")[-1] for a in node.names)
    return names


@pytest.mark.parametrize("module", sorted(INFERENCE))
def test_no_inference_module_imports_the_simulator(module):
    """The blindfold, applied to the layer rather than to one file.

    `test_estimator_is_blind` covers `estimator.py`. Nothing covered the other
    ten, and `realfit.py` -- the only inference code a real race executes -- was
    among them.
    """
    path = XRAY / f"{module}.py"
    if not path.exists():
        pytest.skip(f"{module}.py does not exist")
    leaked = _imported_module_names(path) & SIMULATION
    assert not leaked, (
        f"xray/{module}.py imports {sorted(leaked)} from the SIMULATION layer. "
        f"Inference must be source-agnostic: anything it needs from the "
        f"simulator is either public regulation (xray/constants.py, xray/regs.py) "
        f"or arrives through the feed.")


JUDGE = XRAY / "judge"


@pytest.mark.parametrize(
    "path", sorted(JUDGE.glob("*.py")) if JUDGE.exists() else [],
    ids=lambda p: p.name)
def test_the_judge_package_is_isolated_from_every_computing_layer(path):
    """`xray/judge/` is PRESENTATION/AUDIT and holds a stricter rule than
    rule 1 above: it may import only the standard library and the LLM SDK.

    The generalisation matters. Rule 1 stops inference importing the simulator;
    it would not stop an LLM judge importing `overtake.p_pass` "just to check
    the number", which would make a generated verdict into a second,
    unvalidated physics path that the rest of the stack might believe. The full
    scan, including the SDK-confinement and redaction guarantees, is in
    `tests/test_judge.py`; this entry exists so the rule is visible from the
    file that assigns modules to layers.
    """
    leaked = _imported_module_names(path) & (SIMULATION | INFERENCE)
    assert not leaked, (
        f"xray/judge/{path.name} imports {sorted(leaked)}. The judge reads "
        f"serialized results and computes nothing.")


# (module, attribute) -> (value, why it is allowed to differ from its siblings)
#
# An entry here is an admission, not an approval. Adding one should be harder
# than fixing the divergence.
CONSTANT_REGISTRY = {
    ("estimator", "RESERVE_SIGMA"): (
        1.0e5,
        "Stage 1. Moved 2.5e5 -> 1.0e5 to hit two synthetic test_band_coverage "
        "criteria. Real independently chose 5x this and synthetic cannot "
        "adjudicate, because real data has no E_true."),
    ("rbpf", "RESERVE_SIGMA_J"): (
        5.0e5,
        "Set-membership stack. Reached only by tests."),
    ("realfit", "RESERVE_SIGMA_REAL"): (
        5.0e5,
        "Real path. Agrees with rbpf, not with Stage 1."),
    ("balance", "M_REF_KG"): (
        790.0,
        "Reference mass for the set-membership stack. Stage 1 uses "
        "constants.MASS_CAR_MIN (768.0). Unresolved: no comment records whether "
        "790 means car-plus-average-fuel or is an oversight."),
    ("constants", "FUEL_BURN_PER_LAP_KG"): (
        1.4,
        "The published figure. The set-membership stack defaults to 1.15 and the "
        "real path models no fuel burn at all -- see FUEL_BURN_DEFAULTS below."),
}

# Functions whose fuel-burn default disagrees with constants.FUEL_BURN_PER_LAP_KG.
# The real path is absent from this list because `realfit.build_kin` holds mass
# constant for a whole race and models no burn.
FUEL_BURN_DEFAULTS = {
    ("balance", "build_window_constraints"): 1.15,
    ("balance", "fuel_closure_constraint"): 1.15,
    ("deadband", "fit_cut_speeds"): 1.15,
    ("pipeline", "identify_car"): 1.15,
    ("rbpf", "run"): 1.15,
}


@pytest.mark.parametrize("key", sorted(CONSTANT_REGISTRY))
def test_registered_constants_still_hold_their_registered_values(key):
    """Pin the divergences so none of them moves without this test noticing."""
    import importlib
    module_name, attr = key
    expected, why = CONSTANT_REGISTRY[key]
    mod = importlib.import_module(f"xray.{module_name}")
    actual = getattr(mod, attr)
    assert actual == pytest.approx(expected), (
        f"xray/{module_name}.py::{attr} is {actual}, registered as {expected}.\n"
        f"Registered because: {why}\n"
        f"If the change is deliberate, say which stack's measured numbers moved "
        f"and update the registry in the same commit.")


@pytest.mark.parametrize("key", sorted(FUEL_BURN_DEFAULTS))
def test_fuel_burn_defaults_are_the_registered_ones(key):
    """1.15 kg/lap is hard-coded as a default in five places against a published
    1.4 in `constants`. Whichever is right, the disagreement is registered rather
    than latent -- a tune of one meaning something different in the other is
    exactly the failure this file exists to catch."""
    import importlib
    module_name, func_name = key
    mod = importlib.import_module(f"xray.{module_name}")
    sig = inspect.signature(getattr(mod, func_name))
    default = sig.parameters["fuel_burn_per_lap"].default
    assert default == pytest.approx(FUEL_BURN_DEFAULTS[key]), (
        f"xray/{module_name}.py::{func_name} defaults fuel_burn_per_lap to "
        f"{default}, registered as {FUEL_BURN_DEFAULTS[key]}.")


def test_config_wind_prior_sigma_matches_the_code():
    """`config/default.yaml` says 3.0, `estimator.fit_nuisance` defaults 1.0, and
    nothing reads the yaml key -- so 1.0 is what runs.

    The identical failure is already documented for `dry_event_e_scale` in
    `config/default.yaml` ("It was edited alone once") and pinned by
    `test_config_dry_event_scale_matches_the_code`. This one was unpinned. The
    test asserts the two are reconciled, so the yaml cannot drift away from the
    behaviour again.
    """
    from xray.config import load_config
    from xray.estimator import fit_nuisance

    cfg = load_config(str(pathlib.Path(__file__).resolve().parent.parent
                          / "config" / "default.yaml"))
    yaml_value = cfg["estimator"]["cda_prior_wind_sigma"]
    code_default = inspect.signature(fit_nuisance).parameters["wind_prior_sigma"].default
    assert yaml_value == pytest.approx(code_default), (
        f"config/default.yaml sets cda_prior_wind_sigma={yaml_value} but "
        f"estimator.fit_nuisance defaults wind_prior_sigma={code_default}, and "
        f"nothing reads the yaml key -- the code value is what runs. Reconcile "
        f"them or delete the key; do not leave a number in config that does "
        f"nothing.")
