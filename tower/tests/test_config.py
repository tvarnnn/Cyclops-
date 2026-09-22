import os

from tower.config import get_settings


def test_cv_experiment_defaults_to_baseline(monkeypatch):
    monkeypatch.delenv("TOWER_CV_EXPERIMENT", raising=False)

    settings = get_settings()

    assert settings.cv_experiment == "baseline"


def test_cv_experiment_respects_env_var(monkeypatch):
    monkeypatch.setenv("TOWER_CV_EXPERIMENT", "edge_detection")

    settings = get_settings()

    assert settings.cv_experiment == "edge_detection"


def test_cv_device_defaults_to_auto(monkeypatch):
    monkeypatch.delenv("TOWER_CV_DEVICE", raising=False)

    settings = get_settings()

    assert settings.cv_device == "auto"


def test_cv_device_respects_env_var(monkeypatch):
    monkeypatch.setenv("TOWER_CV_DEVICE", "cpu")

    settings = get_settings()

    assert settings.cv_device == "cpu"


# -- test isolation: the suite must never read the checkout's own data --
#
# `DEFAULT_OBSERVATION_ROOT` is `TOWER_ROOT / "data" / "object_memory"`,
# and `TOWER_ROOT` is derived from `config.py`'s own file location. That
# is right for the PRODUCT -- a Tower nobody configured still serves the
# wearer their own memory, which is the default that was deliberately
# reversed on 2026-08-26 -- and wrong for a TEST, because a test that
# unsets `TOWER_OBSERVATION_ROOT` does not get an empty store. It gets
# whatever the developer's own checkout has accumulated.
#
# That is a privacy defect before it is a flake: the suite reads a real
# wearer's object-memory history off the developer's disk. It is also
# invisible in CI, because `data/` is gitignored, so a fresh clone has
# nothing there and every assertion about "an empty store" passes for the
# wrong reason. It only shows up on a machine that has actually walked
# around -- as `assert 64 == 0`.
#
# The two tests below are the enforcement. They fail if the conftest
# fixture that redirects this default is removed or weakened.


def test_the_suite_never_inherits_the_checkouts_own_observation_store(monkeypatch):
    """An unset TOWER_OBSERVATION_ROOT must not resolve into the repo.

    Not "must be empty" -- must be OUTSIDE THE CHECKOUT. An emptiness
    assertion passes vacuously on a clean clone; this one cannot.
    """
    from pathlib import Path

    from tower.config import TOWER_ROOT

    monkeypatch.delenv("TOWER_OBSERVATION_ROOT", raising=False)
    monkeypatch.delenv("TOWER_OBSERVATION_ENABLED", raising=False)

    resolved = get_settings().observation_root
    root = Path(resolved).resolve()

    assert Path(resolved).is_absolute(), (
        f"the observation root {resolved!r} is RELATIVE, so where it lands "
        "depends on the directory pytest was started from. The containment "
        "check below would then pass from most working directories while "
        "the store still resolved into the checkout from the one that "
        "matters -- the same cwd-anchored trap that put a stub world in the "
        "real corpus."
    )
    assert root != TOWER_ROOT.resolve(), (
        f"the observation root IS the checkout root, {root}. `Path.parents` "
        "excludes the path itself, so the containment check below does not "
        "catch this on its own."
    )
    assert TOWER_ROOT.resolve() not in root.parents, (
        f"an unconfigured Tower under test resolves its observation root to "
        f"{root}, inside the checkout at {TOWER_ROOT}. Every test that does "
        f"not set TOWER_OBSERVATION_ROOT is reading the developer's real "
        f"object-memory store."
    )


def test_an_unconfigured_tower_still_uses_the_products_own_default(monkeypatch):
    """The POSITIVE half. Without it the negative half proves little.

    The autouse fixture in `conftest.py` repoints
    `DEFAULT_OBSERVATION_ROOT` at a temp path for every test, which is
    what stops the suite reading a real wearer's history. The cost is
    that no test could then see the product's ACTUAL default -- and a
    reviewer proved the cost was already paid: an `_observation_root()`
    mutated to ignore `DEFAULT_OBSERVATION_ROOT` entirely passed the
    whole suite at 2233/58, byte-identical to baseline.

    That made the 2026-08-26 reversal -- "a default that hides data from
    its owner while still storing it protects nobody" -- unprovable by
    any test here. This restores it.

    The real value is put back for this test's duration, and the
    expectation is computed from `TOWER_ROOT` rather than imported, so
    this cannot degrade into comparing the constant with itself.
    """
    from pathlib import Path

    from tower import config

    expected = str(Path(config.TOWER_ROOT) / "data" / "object_memory")
    monkeypatch.setattr(config, "DEFAULT_OBSERVATION_ROOT", expected)
    monkeypatch.delenv("TOWER_OBSERVATION_ROOT", raising=False)

    assert config._observation_root(True) == expected, (
        "an enabled cartridge with no TOWER_OBSERVATION_ROOT must resolve to "
        "the product default. A Tower that stores a wearer's memory and then "
        "answers 404 about it is exactly the defect that default reversed."
    )
    assert config._observation_root(False) is None, (
        "a cartridge switched OFF must get no root at all, whatever the "
        "default says"
    )


def test_an_unconfigured_app_under_test_is_pointed_outside_the_checkout():
    """The same invariant one layer up, at the object the routes read.

    `get_settings()` is not the only way to reach a root; the read routes
    use `app.state.object_memory_root`. If someone ever computes that
    from somewhere other than settings, this catches it and the test
    above does not.
    """
    from pathlib import Path

    from tower.config import TOWER_ROOT
    from tower.main import create_app

    served = create_app().state.object_memory_root
    root = Path(served).resolve()

    assert Path(served).is_absolute(), (
        f"the app serves object memory out of the RELATIVE path {served!r}; "
        "where that lands depends on the working directory"
    )
    assert root != TOWER_ROOT.resolve() and TOWER_ROOT.resolve() not in root.parents, (
        f"the app built under test serves object memory out of {root}, "
        f"inside the checkout at {TOWER_ROOT}"
    )
