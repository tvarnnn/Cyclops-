"""Regressions found by the two shared-runtime reviewers of the 2026-08-27 unification.

Both defects existed on a lane branch and neither was reachable there,
because each needed something the OTHER lane brought: a second worker spec
to contend the supervisor's lock, and a live cartridge that follows the
stream while a wearer can pause it. Merging four lanes is what made them
real, so the tests that pin them live with the integration rather than in
any one lane's file.

Each test was proven RED against the pre-fix code before being kept.
"""

import asyncio
import threading
import time

import pytest

from tower.live_session import STATE_PAUSED, STATE_RUNNING, STATE_STOPPED


class _StubEngine:
    """Loads instantly and observes nothing. The LIFECYCLE is under test,
    not the model, so this deliberately carries no torch."""

    def load(self):
        return None

    def observe(self, frame, received_at, source_seq):
        return None

    def release(self):
        return None


def _session():
    """A real `SceneLive` -- the cartridge that actually has this bug --
    over a stub engine.

    `SceneLive` rather than the `LiveSession` base because the base is not
    constructible on its own, and because Scene Understanding is the
    cartridge whose `follow_stream` defaults ON and which detects PEOPLE.
    Testing the base class would prove the mechanism and miss the product.
    """
    from tower.scene.live import SceneLive

    return SceneLive(_StubEngine, decode=lambda raw: raw, follow_stream=True)


# -- Reviewer A, finding 4 ---------------------------------------------


def test_a_new_stream_does_not_resume_a_session_the_wearer_paused():
    """A socket connecting is not a person pressing a button.

    `start()` promotes PAUSED -> RUNNING on the stated grounds of "an
    operator pressing a button twice". `stream_opened` used to call it
    unconditionally, so a wearer who paused Scene Understanding -- which
    detects PEOPLE, and follows the stream by default -- had that pause
    undone by any new connection: a reconnect, a second phone, or a Mac
    running a physical test. Nobody asked, and nothing said so.

    Object Memory reaches the right answer by construction: its gate is
    re-asked at every `capture_opened`, so a paused session stays paused
    across any number of new captures. This makes the two agree.
    """
    session = _session()
    try:
        session.watcher_joined("phone-a:sub-1", owner="phone-a")
        session.stream_opened(owner="phone-a")
        assert session.status()["state"] in (STATE_RUNNING, "starting")

        session.pause()
        assert session.status()["state"] == STATE_PAUSED

        # A third connection arrives. The wearer has not touched anything.
        session.stream_opened(owner="phone-c")
        assert session.status()["state"] == STATE_PAUSED, (
            "a stream_start resumed a session the wearer paused"
        )
    finally:
        session.stop()


def test_an_operator_pressing_start_still_resumes_a_paused_session():
    """The other half, so the fix above cannot be 'never resume'.

    An explicit Start is a person's decision and must still work, or
    Pause becomes a one-way door out of which only Stop leads.
    """
    session = _session()
    try:
        session.start()
        session.pause()
        assert session.status()["state"] == STATE_PAUSED

        session.start()
        assert session.status()["state"] in (STATE_RUNNING, "starting")
    finally:
        session.stop()


def test_a_stream_start_still_starts_a_stopped_session():
    """And the fix must not break the reason `stream_opened` exists.

    Nothing on the wire could start a session before this hook: opening a
    cartridge on the phone sends no verb, and a test asserts that. What
    it does send is a subscription to the live scene, and stream plus
    subscription is the start signal (`tower/scene/live.py`, WHEN IT
    RUNS). Withholding only the PAUSED promotion must leave that intact.
    """
    session = _session()
    try:
        assert session.status()["state"] == STATE_STOPPED
        session.watcher_joined("phone-a:sub-1", owner="phone-a")
        session.stream_opened(owner="phone-a")
        assert session.status()["state"] in (STATE_RUNNING, "starting")
    finally:
        session.stop()


# -- Reviewer A, finding 1 ---------------------------------------------


class _StubSupervisor:
    """A supervisor whose `capture_opened` blocks, the way a real one does
    while `detach` holds its lock across a terminate-and-wait."""

    def __init__(self, block_seconds: float) -> None:
        self._block = block_seconds
        self.calls = 0

    def capture_opened(self, capture_id, capture_dir, continues=None):
        self.calls += 1
        time.sleep(self._block)


class _StubObserver:
    """The dataset recorder's shape, reduced to what `_start_capture` uses."""

    is_recording = False

    def resumable_capture(self):
        return None

    def start(self, owner=None, continues=None):
        return "cap-1"

    def capture_dir(self, capture_id):
        return f"/tmp/{capture_id}"


class _StubState:
    def __init__(self, supervisor):
        self.capture_workers = supervisor
        self.frame_observers = [_StubObserver()]
        self.frame_consumers = []
        self.live_cartridges = None


class _StubApp:
    def __init__(self, supervisor):
        self.state = _StubState(supervisor)


class _StubWebSocket:
    def __init__(self, supervisor):
        self.app = _StubApp(supervisor)


def test_starting_a_capture_does_not_block_the_event_loop():
    """`capture_opened` is the one supervisor call that ran ON THE LOOP.

    Since the supervisor became multi-spec its lock is contended by Pause
    and Stop, whose `detach` holds it across a `terminate()` and a bounded
    `process.wait()`. A producer that ignores SIGTERM therefore blocked
    `stream_start` for up to the terminate timeout PER SPEC -- and on the
    event loop that is the whole Tower: no frames, no pings, no /health,
    on every connection. Reviewer A measured 1.95 s with one stubborn
    worker.

    This drives the REAL `_start_capture`, not a hand-rolled `to_thread`
    around the inner helper -- a test that does its own offloading would
    pass against the very code it is meant to catch. It asserts the
    property that matters: while a capture is being attached, the loop
    still runs other work. Against the pre-fix code the heartbeat scores
    0.
    """
    from tower.routes import ws as ws_module

    supervisor = _StubSupervisor(block_seconds=0.5)
    websocket = _StubWebSocket(supervisor)
    ticks = 0

    async def scenario():
        nonlocal ticks

        async def heartbeat():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        beat = asyncio.create_task(heartbeat())
        await ws_module._start_capture(websocket, owner="conn-1")
        beat.cancel()

    asyncio.run(scenario())

    assert supervisor.calls == 1, "the supervisor was never told a capture opened"
    # 0.5 s of blocking against a 10 ms heartbeat. A frozen loop scores 0;
    # a healthy one scores tens. The bar sits far below the ideal so this
    # cannot flake on a loaded box.
    assert ticks >= 5, (
        f"the event loop advanced only {ticks} times while a capture was "
        "attaching; the attach is blocking the loop"
    )


def test_a_supervisor_that_raises_does_not_end_the_stream():
    """Attaching a worker is a side errand and must stay one.

    Unchanged by the fix, and pinned here because moving the call
    off-thread is exactly the kind of change that loses an except.
    """
    from tower.routes.ws import _offer_capture_opened

    class _Angry:
        def capture_opened(self, *args, **kwargs):
            raise RuntimeError("no")

    # Must not raise.
    _offer_capture_opened(_Angry(), "cap-1", "/tmp/cap-1", None)


def test_offering_a_capture_to_no_supervisor_is_a_no_op():
    from tower.routes.ws import _offer_capture_opened

    _offer_capture_opened(None, "cap-1", "/tmp/cap-1", None)


# -- Reviewer B, finding 1 (CRITICAL) ----------------------------------


def _client_with_lab(monkeypatch, tmp_path):
    """A real app with a capture root, so the recorder actually writes."""
    from fastapi.testclient import TestClient

    from tower.main import create_app

    monkeypatch.setenv("TOWER_CAPTURE_ROOT", str(tmp_path / "captures"))
    monkeypatch.delenv("TOWER_WORLD_ROOT", raising=False)
    client = TestClient(create_app())
    client.__enter__()
    return client


def _jpeg_message(seq: int) -> dict:
    import base64
    import io as _io

    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(seq)
    array = rng.integers(0, 255, size=(360, 640, 3), dtype=np.uint8)
    buffer = _io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=60)
    return {
        "type": "frame",
        "seq": seq,
        "width": 640,
        "height": 360,
        "format": "jpeg",
        "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }


def _frames_written(client) -> int:
    """How many frames actually reached disk.

    `frames_written` lives on `CaptureStatus`, not on the recorder --
    `status` is a PROPERTY and is None when nothing is recording, which is
    itself a failure worth catching here.
    """
    status = client.app.state.frame_observers[0].status
    assert status is not None, "the recorder is not recording at all"
    return status.frames_written


def test_a_paused_cv_lab_does_not_stop_the_dataset_recorder(monkeypatch, tmp_path):
    """One experimental sandbox must not gate the frame bus.

    `cv_lab_pause` is an ordinary, client-reachable command that any
    connection may send. It used to stop frames reaching the DATASET
    RECORDER, Scene Understanding and Document Memory -- while every one
    of them went on reporting itself healthy. `/health` said `capture:
    armed`, the recorder said `is_recording` with an open capture id, and
    zero bytes were written. The walk was lost and nothing said so.

    A frame that arrived and decoded is a real frame. What the CV module
    thought of it is the CV module's business.
    """
    client = _client_with_lab(monkeypatch, tmp_path)
    try:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "stream_start"})
            for seq in range(1, 4):
                ws.send_json(_jpeg_message(seq))
                ws.receive_json()
            before = _frames_written(client)

            ws.send_json({"type": "cv_lab_pause"})
            # Drain until the Lab acknowledges the pause.
            for _ in range(20):
                if ws.receive_json().get("type") == "cv_lab_status":
                    break

            refusals = 0
            for seq in range(10, 20):
                ws.send_json(_jpeg_message(seq))
                reply = ws.receive_json()
                if reply.get("reason") == "cv_lab_paused":
                    refusals += 1
            after = _frames_written(client)

        assert refusals == 10, "the Lab did not actually refuse these frames"
        assert after - before == 10, (
            f"the recorder wrote {after - before} of 10 frames the Lab refused; "
            "a paused CV Lab is starving the dataset recorder"
        )
    finally:
        client.__exit__(None, None, None)


def test_an_unavailable_cv_module_does_not_stop_the_dataset_recorder(
    monkeypatch, tmp_path
):
    """The terminal case, which is worse than the pause.

    `mark_failed()` is terminal, so a typo in TOWER_CV_EXPERIMENT or one
    load timeout made the Tower record nothing and feed no cartridge for
    the LIFE OF THE PROCESS.
    """
    from tower.modules.container import ModuleUnavailableError

    client = _client_with_lab(monkeypatch, tmp_path)
    try:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "stream_start"})
            ws.send_json(_jpeg_message(1))
            ws.receive_json()
            before = _frames_written(client)

            def _dead(_raw):
                raise ModuleUnavailableError("module is FAILED")

            monkeypatch.setattr(
                client.app.state.module_container, "process", _dead
            )

            for seq in range(10, 16):
                ws.send_json(_jpeg_message(seq))
                reply = ws.receive_json()
                assert reply["reason"] == "module_unavailable"
            after = _frames_written(client)

        assert after - before == 6, (
            f"the recorder wrote {after - before} of 6 frames a FAILED module "
            "refused; a dead CV module is starving the dataset recorder"
        )
    finally:
        client.__exit__(None, None, None)


# -- Final adversarial reviewer, finding 2 (CRITICAL) ------------------


def _memory_record(object_class: str, *, frame_seq=None, session_id=None) -> dict:
    """One observation, as a raw JSON dict, in the shape the store writes.

    Built through the real `ObjectObservation.to_json_dict()` rather than
    hand-rolled, so this test cannot pass by writing a record the parser
    would have rejected anyway -- which is the failure mode that would
    make a read-side guard look like it works.
    """
    from tower.object_memory.records import Confidence, ObjectObservation

    return ObjectObservation(
        object_class=object_class,
        detector_score=0.9,
        confidence=Confidence.HIGH,
        observed_at=1787810000.0,
        time_basis="tower-receipt",
        recorded_at=1787810000.0,
        source="glasses-camera",
        module_id="object-memory",
        session_id=session_id,
        frame_seq=frame_seq,
        bounding_box=(0.31, 0.12, 0.68, 0.94),
        retention_tag="default",
        privacy_tags=("derived-only",),
        spatial_ref=None,
        external_refs=(),
    ).to_json_dict()


def test_a_person_record_that_reached_the_store_is_never_served(tmp_path):
    """The write guard protects the FILE. Nothing protected the WIRE.

    `ObservationStore.append` refuses a class this store may not persist,
    and its message names the threat model exactly: "an `append()` from
    anywhere else -- a script, a future consumer, a careless refactor".
    Every READ path was unguarded, so a record that reached the file by
    any of those routes was parsed and served in full: `object_class:
    "person"`, a normalised bounding box, and a session_id plus frame_seq
    that resolve to the original first-person JPEG through `/frame`. The
    payload said `recordable: false` while handing all of it over.

    `person` is not a tier and not a threshold. It is a separate constant,
    checked first, that no model can reach past -- and a refusal that only
    covers the writer is not that.
    """
    import json

    from tower.object_memory.store import OBSERVATIONS_FILENAME, ObservationStore

    root = tmp_path / "mem"
    root.mkdir()
    store = ObservationStore(root, retention_seconds=None)

    # Written the way the guard's own comment anticipates: by something
    # that is not `append()`.
    with (root / OBSERVATIONS_FILENAME).open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(_memory_record("laptop")) + "\n")
        handle.write(
            json.dumps(
                _memory_record("person", frame_seq=3410, session_id="cap-abc123")
            )
            + "\n"
        )

    classes = {o.object_class for o in store.all_observations()}
    assert classes == {"laptop"}, (
        f"the read side served {classes}; a person record reached the wire"
    )
    assert store.last_seen("person") is None, (
        "last_seen served a person record by name"
    )


def test_the_read_guard_does_not_drop_what_the_store_may_hold(tmp_path):
    """The other half, so the fix cannot be "serve nothing"."""
    import json

    from tower.object_memory.store import OBSERVATIONS_FILENAME, ObservationStore

    root = tmp_path / "mem"
    root.mkdir()
    store = ObservationStore(root, retention_seconds=None)
    with (root / OBSERVATIONS_FILENAME).open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(_memory_record("laptop")) + "\n")
        handle.write(json.dumps(_memory_record("cell phone")) + "\n")

    classes = {o.object_class for o in store.all_observations()}
    assert classes == {"laptop", "cell phone"}, classes
    assert store.last_seen("laptop") is not None


# -- Final adversarial reviewer, finding 4 (MAJOR) ---------------------


def test_the_deletion_cli_defaults_to_the_root_the_producer_writes_to():
    """A wearer asks for erasure and gets a silent no-op, exit code 0.

    `scripts/object_query.py --purge-all` is the only deletion path this
    cartridge has, and `config.py` names it as "where real deletion
    lives". Its default root was `Path("data/object_memory")` -- relative,
    resolved against whatever directory the operator stood in, and never
    consulting TOWER_OBSERVATION_ROOT.

    `config.py` exists to stop precisely this and says why at length: two
    defaults for one directory is "a bug with a settings file in front of
    it". `object_memory_session.py` was fixed then; this one was not, and
    it is the worse place to miss, because `purge()` on a directory
    nothing wrote to removes nothing and reports success --
    `unlink(missing_ok=True)` cannot tell "already gone" from "never
    here".

    Every existing CLI test passes `--root` explicitly, which is why none
    of them caught it.
    """
    import importlib
    import sys
    from pathlib import Path

    from tower.config import DEFAULT_OBSERVATION_ROOT

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    object_query = importlib.import_module("object_query")
    importlib.reload(object_query)

    assert object_query.DEFAULT_ROOT.is_absolute(), (
        f"the deletion CLI's default root is relative ({object_query.DEFAULT_ROOT}); "
        "it resolves against the operator's working directory"
    )
    assert object_query.DEFAULT_ROOT == Path(DEFAULT_OBSERVATION_ROOT), (
        f"the deletion CLI defaults to {object_query.DEFAULT_ROOT} while the "
        f"producer and the read routes use {DEFAULT_OBSERVATION_ROOT}"
    )


# -- Final adversarial reviewer, finding 12 (MEDIUM) -------------------


def test_a_failure_reason_on_the_wire_never_carries_a_filesystem_path():
    """These strings reach an unauthenticated socket.

    An OSError describes a failure by naming the PATH it happened on, so a
    missing world, a missing model file or a torch.hub cache miss would
    disclose the home directory -- and with it the OS username -- to
    anyone who can open `/ws`.

    The split is by KIND, not by caller: this repository's own exceptions
    are written to be read by a person and pass through, because
    "this build does not recognise that pose convention" is the contract
    working. Blanket-suppressing the message was tried first and two
    tests correctly refused it.
    """
    from tower.logging_config import client_safe_reason

    leaky = FileNotFoundError(
        2, "No such file or directory", r"C:\Users\someone\worlds\w\world.json"
    )
    text = client_safe_reason(leaky)
    assert text == "FileNotFoundError", text
    assert "someone" not in text and "world.json" not in text

    class UnknownPoseConventionError(ValueError):
        pass

    explained = UnknownPoseConventionError(
        "world declares a pose convention this build does not recognise"
    )
    text = client_safe_reason(explained)
    assert "does not recognise" in text, (
        "a domain exception's explanation was suppressed; the client is "
        "left with a type name it cannot act on"
    )


def test_an_import_failure_names_the_module_without_naming_the_file():
    """`ImportError` is the second exception that carries a path.

    The rule above reduces `OSError` and passes everything else through,
    which is correct for this repository's own exceptions and WRONG for
    `ImportError`: CPython builds the message for a failed
    `from X import Y` by appending the module's `__file__` in
    parentheses. Measured before this guard, on a real one:

        ImportError: cannot import name 'X' from 'torchvision.transforms'
        (C:\\Users\\tvllo\\Projects\\Glasses\\tower\\.venv\\Lib\\site-packages
        \\torchvision\\transforms\\__init__.py)

    That reached `/cartridges` the moment Scene Understanding began
    reporting WHY it could not be constructed, and it discloses the home
    directory and the OS username -- the exact threat this helper exists
    to stop, arriving through the one exception type it did not cover.

    The module NAME is kept, because it is the actionable half and it is
    structured data rather than prose: `exc.name` is set by the import
    system and never holds a path.
    """
    from tower.logging_config import client_safe_reason

    leaky = ImportError(
        "cannot import name 'InterpolationMode' from 'torchvision.transforms' "
        r"(C:\Users\someone\.venv\Lib\site-packages\torchvision\transforms\__init__.py)",
        name="torchvision.transforms",
        path=r"C:\Users\someone\.venv\Lib\site-packages\torchvision\transforms\__init__.py",
    )
    text = client_safe_reason(leaky)

    assert "someone" not in text, text
    assert ".venv" not in text and "site-packages" not in text, text
    assert "__init__.py" not in text, text
    # Still says what is missing.
    assert "torchvision.transforms" in text, text

    missing = ModuleNotFoundError("No module named 'torch'", name="torch")
    text = client_safe_reason(missing)
    assert "torch" in text, text
    assert "\\" not in text and "/" not in text, text
