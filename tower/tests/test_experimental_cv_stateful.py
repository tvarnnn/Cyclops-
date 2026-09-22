"""The Lab module hosting a STATEFUL experiment.

This file replaces the old test_depth_cv_module.py. Its subject changed
from "a second Module subclass exists for the depth experiment" to "one
Module subclass hosts anything implementing the Experiment protocol",
which is the point of the V1 refactor.
"""

import asyncio

import pytest

from tower.experiments import ExperimentSettings
from tower.modules.base import ModuleState
from tower.modules.experimental_cv import ExperimentalCVModule


class _FakeStatefulExperiment:
    name = "fake"

    def __init__(self) -> None:
        self.loaded_with = None
        self.release_calls = 0
        self.run_calls = 0

    def load(self, settings) -> None:
        self.loaded_with = settings

    def run(self, raw_bytes: bytes):
        self.run_calls += 1
        return f"fake-result:{raw_bytes!r}"

    def release(self) -> None:
        self.release_calls += 1


def test_full_lifecycle_delegates_to_the_experiment():
    fake = _FakeStatefulExperiment()
    module = ExperimentalCVModule(
        "fake", ExperimentSettings(device="cpu"), experiment=fake
    )

    asyncio.run(module.load())
    assert module.state == ModuleState.READY
    assert fake.loaded_with == ExperimentSettings(device="cpu")

    asyncio.run(module.start())
    assert module.state == ModuleState.ACTIVE

    assert module.process(b"frame-bytes") == "fake-result:b'frame-bytes'"
    assert fake.run_calls == 1

    asyncio.run(module.stop())
    asyncio.run(module.unload())
    assert module.state == ModuleState.UNLOADED
    assert fake.release_calls == 1


def test_mark_failed_releases_the_experiment():
    fake = _FakeStatefulExperiment()
    module = ExperimentalCVModule("fake", experiment=fake)
    asyncio.run(module.load())
    asyncio.run(module.start())

    module.mark_failed()

    assert module.state == ModuleState.FAILED
    assert fake.release_calls == 1


def test_release_is_idempotent():
    """mark_failed can be reached from anywhere, including after unload."""
    fake = _FakeStatefulExperiment()
    module = ExperimentalCVModule("fake", experiment=fake)
    asyncio.run(module.load())
    asyncio.run(module.start())
    asyncio.run(module.stop())
    asyncio.run(module.unload())

    module.mark_failed()

    assert fake.release_calls == 1, "the experiment must not be released twice"


def test_a_failure_during_load_still_releases():
    """A partially-loaded experiment must not survive its failed arm.

    Since 2026-09-06 a startup load that raises no longer propagates out
    of `load()` -- the Lab reports `failed` and stays usable -- so the
    release happens inside the Lab, at once, rather than on the module's
    FAILED transition. Either way: released exactly once, and a later
    `mark_failed()` must not release it again.
    """

    class _ExplodingExperiment(_FakeStatefulExperiment):
        def load(self, settings):
            raise RuntimeError("model download failed")

    fake = _ExplodingExperiment()
    module = ExperimentalCVModule("fake", experiment=fake)

    asyncio.run(module.load())  # no longer raises
    assert module.lab.status()["lifecycle"]["state"] == "failed"
    assert "model download failed" in module.lab.status()["lifecycle"]["reason"]
    assert fake.release_calls == 1

    module.mark_failed()
    assert fake.release_calls == 1


def test_an_unknown_experiment_name_is_reported_not_raised_at_load():
    """Loud in the log, `failed` on the wire, and not terminal."""
    module = ExperimentalCVModule("not-a-real-experiment")

    asyncio.run(module.load())  # no longer raises

    status = module.lab.status()
    assert status["lifecycle"]["state"] == "failed"
    assert "unknown experiment" in status["lifecycle"]["reason"]


def test_the_descriptor_still_declares_no_persistence():
    """The refactor must not have quietly changed what the Lab claims."""
    module = ExperimentalCVModule("baseline")

    behaviour = module.descriptor.data_behavior
    assert module.descriptor.id == "experimental-cv"
    assert behaviour.persists_data is False
    assert behaviour.retains_raw_imagery is False
    assert behaviour.transmits_externally is False
