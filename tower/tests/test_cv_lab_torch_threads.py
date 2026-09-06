"""The torch intra-op budget: chosen from a measurement, applied where it counts.

torch's pool defaults to one thread per logical CPU and Intel OpenMP
workers spin-wait between parallel regions, so `object_detection` on
CUDA read 12.6 cores at 20 threads and 1.8 at 2, at the same latency
(2026-09-06, this host). The budget is an `ExperimentSettings` field
resolved per device, and -- the part that is easy to get wrong -- OpenMP
keeps it PER CALLING THREAD, so it has to be applied on the thread that
runs inference, not only on the loader thread that built the model.
"""

import asyncio
import threading

import pytest

from tests.cv_lab_fixtures import armed_lab, jpeg_bytes, start_and_wait
from tower.config import get_settings
from tower.experiments import ExperimentSettings
from tower.experiments.depth import (
    TORCH_THREADS_CPU,
    TORCH_THREADS_CUDA,
    resolve_torch_threads,
)


# -- resolution --------------------------------------------------------


def test_auto_resolves_per_device():
    assert resolve_torch_threads("cuda", "auto") == TORCH_THREADS_CUDA
    assert resolve_torch_threads("cuda:0", "auto") == TORCH_THREADS_CUDA
    assert resolve_torch_threads("cpu", "auto") == TORCH_THREADS_CPU
    assert resolve_torch_threads("cpu") == TORCH_THREADS_CPU


def test_the_defaults_are_the_measured_ones():
    assert (TORCH_THREADS_CUDA, TORCH_THREADS_CPU) == (2, 4)


def test_zero_or_none_leaves_torch_alone():
    assert resolve_torch_threads("cuda", 0) is None
    assert resolve_torch_threads("cpu", "0") is None
    assert resolve_torch_threads("cpu", None) is None


def test_an_explicit_budget_is_honoured_on_any_device():
    assert resolve_torch_threads("cuda", 3) == 3
    assert resolve_torch_threads("cpu", "6") == 6


@pytest.mark.parametrize("bad", ["x", -1, "-2", True, 1.5, ""])
def test_garbage_is_refused_rather_than_defaulted(bad):
    with pytest.raises(ValueError):
        resolve_torch_threads("cpu", bad)


def test_settings_default_to_auto():
    assert ExperimentSettings().torch_threads == "auto"


def test_the_environment_variable_is_read_and_sanitised(monkeypatch):
    monkeypatch.setenv("TOWER_CV_TORCH_THREADS", "3")
    assert get_settings().cv_torch_threads == 3
    monkeypatch.setenv("TOWER_CV_TORCH_THREADS", "0")
    assert get_settings().cv_torch_threads == 0
    monkeypatch.setenv("TOWER_CV_TORCH_THREADS", "auto")
    assert get_settings().cv_torch_threads == "auto"
    monkeypatch.setenv("TOWER_CV_TORCH_THREADS", "banana")
    assert get_settings().cv_torch_threads == "auto"
    monkeypatch.setenv("TOWER_CV_TORCH_THREADS", "-4")
    assert get_settings().cv_torch_threads == "auto"
    monkeypatch.delenv("TOWER_CV_TORCH_THREADS")
    assert get_settings().cv_torch_threads == "auto"


# -- application, with real torch -------------------------------------


@pytest.fixture
def torch_restored():
    """torch's budget on this thread, put back afterwards.

    The setting is per calling thread in OpenMP and the test thread is
    shared with every other test in the session.
    """
    torch = pytest.importorskip("torch")
    before = torch.get_num_threads()
    yield torch
    torch.set_num_threads(before)


def test_object_detection_applies_the_budget_on_the_loader_and_the_inference_thread(
    torch_restored,
):
    torch = torch_restored
    pytest.importorskip("torchvision")
    from tower.experiments.object_detection import ObjectDetectionExperiment

    experiment = ObjectDetectionExperiment()
    seen: dict[str, int] = {}

    def _load():
        experiment.load(ExperimentSettings(device="cpu", torch_threads=2))
        seen["loader"] = torch.get_num_threads()

    loader = threading.Thread(target=_load)
    loader.start()
    loader.join()
    assert seen["loader"] == 2

    # A DIFFERENT thread with a different budget: the first frame must
    # bring it to the experiment's, because a cap set on the loader thread
    # never reached this one.
    torch.set_num_threads(max(4, torch.get_num_threads()))
    try:
        experiment.run(jpeg_bytes(320, 240, textured=True))
        assert torch.get_num_threads() == 2
        assert experiment.describe()["torch_threads"] == 2
    finally:
        experiment.release()


def test_depth_reports_the_budget_it_resolved(torch_restored):
    pytest.importorskip("timm")
    from tower.experiments.depth import DepthEstimation

    experiment = DepthEstimation()
    try:
        experiment.load(ExperimentSettings(device="cpu", torch_threads="auto"))
        assert experiment.describe()["torch_threads"] == TORCH_THREADS_CPU
        assert torch_restored.get_num_threads() == TORCH_THREADS_CPU
    finally:
        experiment.release()


def test_zero_leaves_the_pool_exactly_as_it_was(torch_restored):
    torch = torch_restored
    pytest.importorskip("torchvision")
    from tower.experiments.object_detection import ObjectDetectionExperiment

    before = torch.get_num_threads()
    experiment = ObjectDetectionExperiment()
    try:
        experiment.load(ExperimentSettings(device="cpu", torch_threads=0))
        experiment.run(jpeg_bytes(320, 240, textured=True))
        assert torch.get_num_threads() == before
        assert experiment.describe()["torch_threads"] is None
    finally:
        experiment.release()


# -- the leak this lane was opened for, end to end ---------------------


@pytest.mark.slow
def test_repeated_heavy_arms_do_not_grow_the_thread_count(torch_restored):
    """Eight arms alternating the two model-backed experiments through a
    real `CVLab`. Before the reusable loader thread this grew by 19 OS
    threads per arm; the first pair is excluded because it legitimately
    creates the loader's team and the frame thread's team once."""
    psutil = pytest.importorskip("psutil")
    pytest.importorskip("torchvision")
    pytest.importorskip("timm")
    frame = jpeg_bytes(640, 360, textured=True)
    process = psutil.Process()

    async def scenario():
        lab = await armed_lab("baseline")
        counts = []
        try:
            for _ in range(4):
                for experiment_id in ("object_detection", "depth"):
                    await start_and_wait(lab, experiment_id)
                    status = lab.status()
                    assert status["lifecycle"]["state"] == "running", status["lifecycle"]
                    lab.process(frame)
                    counts.append(process.num_threads())
        finally:
            await lab.shutdown()
        return counts

    counts = asyncio.run(scenario())

    settled = counts[2:]
    assert max(settled) - min(settled) <= 2, counts
