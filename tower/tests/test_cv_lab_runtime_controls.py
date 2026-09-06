"""Nothing the CV Lab does should need a Tower restart to recover from.

Before 2026-09-06 two paths were terminal for the life of the process:
an experiment raising anything but `FrameProcessingError` on a frame,
and a startup default that could not load. Both reached
`ModuleContainer.mark_failed()`, and there is no way back from that. With
object detection driving a machine to 99% CPU, one odd frame is
plausible -- and after it, every experiment was dead until somebody
restarted uvicorn. That is the "restart Tower" the mission brief
describes, and this file is what keeps it gone.

The second half covers what a switch costs: loads run on ONE reusable
thread (a fresh thread per arm leaked 19 OS threads a switch, measured),
exactly one experiment is alive at a time, and the status document says
how long the arm took and what the process is using.
"""

import asyncio
import gc
import logging
import os
import threading
import weakref

import pytest

from tests.cv_lab_fixtures import (  # noqa: F401
    _close_cv_lab_clients,
    armed_lab,
    jpeg_bytes,
    make_client,
    start_and_wait,
)
from tower.cv_lab.contracts import (
    FRAME_REFUSED_FAILED,
    STATE_FAILED,
    STATE_RUNNING,
)
from tower.cv_lab.loader import LOADER_THREAD_NAME
from tower.experiments import EXPERIMENTS, ExperimentResult
from tower.modules.base import FrameProcessingError, FrameSkippedError, ModuleState
from tower.modules.container import ModuleContainer
from tower.modules.experimental_cv import ExperimentalCVModule


def _loader_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name.startswith(LOADER_THREAD_NAME)]


class _Recording:
    """A cheap experiment that remembers what happened to it."""

    name = "recording"
    instances: list["_Recording"] = []

    def __init__(self) -> None:
        self.loaded_on: int | None = None
        self.loaded_thread_name: str | None = None
        self.released = 0
        self.frames = 0
        type(self).instances.append(self)

    def load(self, settings):
        self.loaded_on = threading.get_ident()
        self.loaded_thread_name = threading.current_thread().name

    def run(self, raw_bytes):
        self.frames += 1
        return ExperimentResult(
            result_value=1.0,
            result_label="recorded",
            processing_ms=0.1,
            stage_ms={},
        )

    def release(self):
        self.released += 1


class _CrashesOnFrame(_Recording):
    name = "crasher"

    def run(self, raw_bytes):
        raise RuntimeError("the detector exploded on this frame")


class _Exploding(_Recording):
    name = "boom"

    def load(self, settings):
        raise RuntimeError("weights unavailable")


@pytest.fixture
def swap(monkeypatch):
    """Replace a registry entry for the length of a test."""

    def _swap(experiment_id: str, factory):
        monkeypatch.setitem(EXPERIMENTS, experiment_id, factory)

    _Recording.instances = []
    return _swap


# -- a crash on a frame ends the run, not the process -----------------


def test_an_experiment_that_raises_on_a_frame_fails_the_run_and_frees_itself(swap):
    swap("edge_detection", _CrashesOnFrame)
    lab = asyncio.run(armed_lab("baseline"))
    asyncio.run(start_and_wait(lab, "edge_detection"))
    crasher = _Recording.instances[-1]

    with pytest.raises(FrameProcessingError) as caught:
        lab.process(jpeg_bytes())

    assert caught.value.reason == FRAME_REFUSED_FAILED
    assert "exploded" in str(caught.value)
    status = lab.status()
    assert status["lifecycle"]["state"] == STATE_FAILED
    assert "exploded" in status["lifecycle"]["reason"]
    assert status["run"]["frames_failed"] == 1
    assert status["run"]["ended_at"] is not None
    assert crasher.released == 1, "the crashed experiment must be released at once"
    assert lab.frame_provenance() is None

    # The next frame is refused with the same reason, not re-run.
    with pytest.raises(FrameProcessingError) as again:
        lab.process(jpeg_bytes())
    assert again.value.reason == FRAME_REFUSED_FAILED
    assert crasher.released == 1


def test_after_a_frame_crash_the_next_start_is_accepted_and_works(swap):
    swap("edge_detection", _CrashesOnFrame)
    lab = asyncio.run(armed_lab("baseline"))
    asyncio.run(start_and_wait(lab, "edge_detection"))
    with pytest.raises(FrameProcessingError):
        lab.process(jpeg_bytes())

    recovered = asyncio.run(start_and_wait(lab, "baseline"))

    assert recovered.accepted is True
    assert lab.status()["lifecycle"]["state"] == STATE_RUNNING
    assert lab.process(jpeg_bytes()).result_label == "mean_intensity"


def test_a_frame_crash_reaches_the_container_as_a_skipped_frame_and_the_module_stays_active(swap):
    """The property the whole file exists for: `mark_failed()` is never
    reached from a frame."""
    swap("edge_detection", _CrashesOnFrame)
    module = ExperimentalCVModule("baseline")
    container = ModuleContainer(module)
    asyncio.run(container.load_and_start())
    asyncio.run(start_and_wait(module.lab, "edge_detection"))

    with pytest.raises(FrameSkippedError) as caught:
        container.process(jpeg_bytes())

    assert caught.value.reason == FRAME_REFUSED_FAILED
    assert container.state == ModuleState.ACTIVE
    asyncio.run(start_and_wait(module.lab, "baseline"))
    assert container.process(jpeg_bytes()).result_label == "mean_intensity"


def test_a_crash_that_is_not_an_exception_still_propagates(swap):
    """`KeyboardInterrupt` and friends are not a run failure and must not
    be swallowed into one."""

    class _Interrupts(_Recording):
        def run(self, raw_bytes):
            raise KeyboardInterrupt

    swap("edge_detection", _Interrupts)
    lab = asyncio.run(armed_lab("baseline"))
    asyncio.run(start_and_wait(lab, "edge_detection"))

    with pytest.raises(KeyboardInterrupt):
        lab.process(jpeg_bytes())


# -- a startup default that cannot arm is loud, and recoverable --------


def test_a_startup_default_that_fails_to_load_leaves_the_lab_failed_and_the_module_active(swap, caplog):
    swap("edge_detection", _Exploding)
    module = ExperimentalCVModule("edge_detection")
    container = ModuleContainer(module)

    with caplog.at_level(logging.ERROR):
        asyncio.run(container.load_and_start())

    assert container.state == ModuleState.ACTIVE
    status = module.lab.status()
    assert status["lifecycle"]["state"] == STATE_FAILED
    assert "weights unavailable" in status["lifecycle"]["reason"]
    assert status["run"]["origin"] == "startup_default"
    assert status["run"]["ended_at"] is not None
    assert _Recording.instances[-1].released == 1
    loud = [r for r in caplog.records if "TOWER_CV_EXPERIMENT" in r.getMessage()]
    assert loud and loud[0].levelno >= logging.ERROR

    # Frames are refused with the failed reason rather than killing anything.
    with pytest.raises(FrameSkippedError) as caught:
        container.process(jpeg_bytes())
    assert caught.value.reason == FRAME_REFUSED_FAILED
    assert container.state == ModuleState.ACTIVE

    # And a client can pick anything else without a restart.
    outcome = asyncio.run(start_and_wait(module.lab, "baseline"))
    assert outcome.accepted is True
    assert container.process(jpeg_bytes()).result_label == "mean_intensity"


def test_an_unknown_startup_default_is_loud_but_not_terminal(caplog):
    module = ExperimentalCVModule("not-a-real-experiment")
    container = ModuleContainer(module)

    with caplog.at_level(logging.ERROR):
        asyncio.run(container.load_and_start())

    assert container.state == ModuleState.ACTIVE
    status = module.lab.status()
    assert status["lifecycle"]["state"] == STATE_FAILED
    assert "not-a-real-experiment" in status["lifecycle"]["reason"]
    assert status["selected"] == "not-a-real-experiment"
    assert any("TOWER_CV_EXPERIMENT" in r.getMessage() for r in caplog.records)
    assert asyncio.run(start_and_wait(module.lab, "baseline")).accepted is True


def test_a_failed_startup_default_shows_on_the_wire(monkeypatch, swap):
    swap("edge_detection", _Exploding)
    client = make_client(monkeypatch, "edge_detection")

    document = client.get("/cv-lab").json()["status"]

    assert document["lifecycle"]["state"] == STATE_FAILED
    assert client.get("/health").json()["module_state"] == "active"
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "cv_lab_start", "experiment_id": "baseline"})
        reply = ws.receive_json()
        assert reply["type"] == "cv_lab_status"
        assert reply["accepted_command"] == "cv_lab_start"


# -- what a switch costs -----------------------------------------------


def test_loads_run_on_the_lab_loader_thread_and_reuse_it(swap):
    swap("edge_detection", _Recording)
    swap("frame_quality", _Recording)
    before = {t.ident for t in _loader_threads()}
    lab = asyncio.run(armed_lab("baseline"))

    async def scenario():
        await start_and_wait(lab, "edge_detection")
        await start_and_wait(lab, "frame_quality")
        await start_and_wait(lab, "edge_detection")

    asyncio.run(scenario())

    idents = {inst.loaded_on for inst in _Recording.instances}
    names = {inst.loaded_thread_name for inst in _Recording.instances}
    assert len(idents) == 1, "each arm ran on a different thread"
    assert threading.get_ident() not in idents, "a load ran on the event loop thread"
    assert all(name.startswith(LOADER_THREAD_NAME) for name in names)
    # Other tests' Labs may still hold loader threads; only the ones this
    # test created are ours to count.
    mine = {t.ident for t in _loader_threads()} - before
    assert mine == idents


def test_repeated_switching_keeps_exactly_one_experiment_alive(swap):
    swap("edge_detection", _Recording)
    swap("frame_quality", _Recording)
    swap("optical_flow", _Recording)
    lab = asyncio.run(armed_lab("baseline"))

    async def scenario():
        for _ in range(3):
            for experiment_id in ("edge_detection", "frame_quality", "optical_flow"):
                await start_and_wait(lab, experiment_id)
                lab.process(jpeg_bytes())

    asyncio.run(scenario())
    refs = [weakref.ref(inst) for inst in _Recording.instances]
    current = _Recording.instances[-1]
    del _Recording.instances[:]
    gc.collect()

    alive = [ref() for ref in refs if ref() is not None]
    assert alive == [current], (
        f"{len(alive)} experiment instances are still reachable after nine "
        "switches; a released experiment must not be held anywhere"
    )
    assert all(inst.released == 1 for inst in [r() for r in refs] if inst is not None and inst is not current)


def test_shutdown_closes_the_loader_thread(swap):
    swap("edge_detection", _Recording)
    before = {t.ident for t in _loader_threads()}

    async def scenario():
        lab = await armed_lab("baseline")
        await start_and_wait(lab, "edge_detection")
        assert len([t for t in _loader_threads() if t.ident not in before]) == 1
        await lab.shutdown()

    asyncio.run(scenario())
    mine = [t for t in _loader_threads() if t.ident not in before]
    for thread in mine:
        thread.join(timeout=5.0)
    assert not [t for t in _loader_threads() if t.ident not in before]


def test_the_run_records_how_long_the_arm_took(swap):
    swap("edge_detection", _Recording)
    lab = asyncio.run(armed_lab("baseline"))
    assert lab.status()["run"]["arm_ms"] is not None, "the startup arm is timed too"

    asyncio.run(start_and_wait(lab, "edge_detection"))
    arm_ms = lab.status()["run"]["arm_ms"]

    assert isinstance(arm_ms, (int, float))
    assert 0.0 <= arm_ms < 5_000.0


def test_a_failed_arm_has_no_arm_time(swap):
    swap("edge_detection", _Exploding)
    lab = asyncio.run(armed_lab("baseline"))
    asyncio.run(start_and_wait(lab, "edge_detection"))

    assert lab.status()["lifecycle"]["state"] == STATE_FAILED
    assert lab.status()["run"]["arm_ms"] is None


def test_get_cv_lab_reports_the_process_beside_the_document(monkeypatch):
    """Beside, not inside: the document is byte-equal on three surfaces
    and a live RSS figure would not be."""
    client = make_client(monkeypatch, "baseline")

    body = client.get("/cv-lab").json()

    assert set(body["process"]) == {"pid", "threads", "rss_mb"}
    assert body["process"]["pid"] == os.getpid()
    assert body["process"]["threads"] >= 1
    assert body["process"]["rss_mb"] > 0
    assert "process" not in body["status"]
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "cv_lab_status"})
        assert "process" not in ws.receive_json()["status"]
