"""Marks the test suite as a package.

Not cosmetic. Several test modules do `from tests.simfix import ...`, and a
stray `tests` distribution in site-packages ships its own `__init__.py`. A
regular package beats a namespace package on sys.path, so without this file
site-packages won 6 collections outright: test_balance, test_deadband,
test_golden, test_pipeline, test_rbpf and test_setmem all died at import with
"No module named 'tests.simfix'" -- the golden regression and set-membership
containment suites among them. They did not fail; they never ran.
"""
