"""Press Start; a producer attaches itself. Press Stop; it lets go.

The product claim of this workstream, asserted against the real ASGI app.

What it replaces was measured, not imagined. On 2026-08-26 a real
Ray-Ban walk went through the phone, the Tower, a capture directory and
an object-memory producer, and produced 64 observations the iOS app
displayed correctly. Every one of those observations required a human to:

  1. start a generic recording from the Home screen,
  2. find the capture directory that had just been minted,
  3. run `object_memory_session.py --follow-capture <dir>` in a second
     terminal,
  4. and then set `TOWER_OBSERVATION_ROOT` to the same directory the
     producer had defaulted to, because until they did, every HTTP
     request answered 404 about the memory that had just been written.

None of those four steps exists any more, and each of the four has a
test below.

The worker itself is faked throughout. A real producer would need a real
capture to tail and 13.4 MB of weights to load; what is under test here
is the WIRING and the STATE MACHINE, which is where every one of the four
steps actually lived.
"""

import pytest
from fastapi.testclient import TestClient

from tower.capture_workers import ATTACH_MODE_FROM_NOW, ATTACH_MODE_FROM_START
from tower.main import OBJECT_MEMORY_WORKER, WORLD_BUILD_WORKER, create_app
from tower.results.contracts import CARTRIDGE_OBJECT_MEMORY

SESSION_URL = f"/cartridges/{CARTRIDGE_OBJECT_MEMORY}/session"


class FakeProcess:
    _pids = iter(range(5000, 500000))

    def __init__(self, argv, **kwargs):
        self.args = list(argv)
        self.pid = next(FakeProcess._pids)
        self._returncode = None
        self.terminated = False

    def poll(self):
        return self._returncode

    def wait(self, timeout=None):
        if self._returncode is None:
            raise TimeoutError(timeout)
        return self._returncode

    def terminate(self):
        self.terminated = True
        self._returncode = -15


@pytest.fixture
def app(monkeypatch, tmp_path):
    monkeypatch.setenv("TOWER_CAPTURE_ROOT", str(tmp_path / "capture"))
    monkeypatch.setenv("TOWER_WORLD_ROOT", str(tmp_path / "world"))
    monkeypatch.setenv("TOWER_OBSERVATION_ROOT", str(tmp_path / "memory"))
    monkeypatch.delenv("TOWER_OBSERVATION_ENABLED", raising=False)
    application = create_app()
    application.state.spawned = []

    def spawn(argv, **kwargs):
        process = FakeProcess(argv, **kwargs)
        application.state.spawned.append(process)
        return process

    application.state.capture_workers._spawn = spawn
    return application


@pytest.fixture
def client(app):
    return TestClient(app)


def _open_capture(app, capture_id="cap1", continues=None):
    """Mint a capture the way `stream_start` does, and notify the supervisor."""
    observer = app.state.frame_observers[0]
    directory = observer.capture_dir(capture_id)
    directory.mkdir(parents=True, exist_ok=True)
    app.state.capture_workers.capture_opened(
        capture_id, directory, continues=continues
    )
    return directory


def _producers(app):
    return [
        process
        for process in app.state.spawned
        if "object_memory_session.py" in " ".join(process.args)
    ]


def _builders(app):
    return [
        process
        for process in app.state.spawned
        if "world_build_session.py" in " ".join(process.args)
    ]


# -- step 3: nobody runs a script by hand ------------------------------


def test_a_capture_opening_during_an_active_session_attaches_a_producer(app, client):
    client.post(f"{SESSION_URL}/start")

    _open_capture(app)

    assert len(_producers(app)) == 1


def test_the_producer_is_told_it_saw_the_whole_capture(app, client):
    client.post(f"{SESSION_URL}/start")

    _open_capture(app)

    argv = _producers(app)[0].args
    assert argv[argv.index("--attach-mode") + 1] == ATTACH_MODE_FROM_START


def test_starting_mid_walk_attaches_to_the_capture_already_recording(app, client):
    """Start pressed three minutes in still works, and does not look back.

    The producer is told it arrived late, so it skips the part of the
    journal that was recorded before anybody asked for it. A wearer who
    starts remembering at 15:03 has not asked for the 15:00 part of the
    walk to be remembered, and reading it would be a consent decision no
    script has standing to make.
    """
    observer = app.state.frame_observers[0]
    observer.start()

    client.post(f"{SESSION_URL}/start")

    argv = _producers(app)[0].args
    assert argv[argv.index("--attach-mode") + 1] == ATTACH_MODE_FROM_NOW


# -- the gate: a stopped cartridge remembers nothing -------------------


WORLD_BUILDER_SESSION_URL = "/cartridges/world_builder/session"


def test_a_capture_opening_with_no_session_attaches_nothing(app):
    """Armed is not recording, and the default is armed.

    Since 2026-09-06 the builder is gated too: a world is geometry, but a
    builder with its background solves attached to every capture -- a CV
    Lab camera session included -- and the only way to run another
    cartridge without it was a Tower restart. Each cartridge's producer
    now waits to be asked for.
    """
    _open_capture(app)

    assert _producers(app) == []
    assert _builders(app) == []


def test_a_capture_opening_with_world_builder_active_attaches_only_a_builder(app, client):
    client.post(f"{WORLD_BUILDER_SESSION_URL}/start")
    _open_capture(app)

    assert _producers(app) == []
    assert len(_builders(app)) == 1


def test_pausing_stops_the_producer_and_leaves_the_builder_alone(app, client):
    client.post(f"{WORLD_BUILDER_SESSION_URL}/start")
    client.post(f"{SESSION_URL}/start")
    _open_capture(app)
    producer = _producers(app)[0]
    builder = _builders(app)[0]

    client.post(f"{SESSION_URL}/pause")

    assert producer.terminated is True
    assert builder.terminated is False


def test_a_paused_session_does_not_attach_to_the_next_capture(app, client):
    client.post(f"{SESSION_URL}/start")
    _open_capture(app, "cap1")
    client.post(f"{SESSION_URL}/pause")

    _open_capture(app, "cap2")

    assert len(_producers(app)) == 1


def test_resuming_attaches_to_whatever_is_recording_now(app, client):
    client.post(f"{SESSION_URL}/start")
    _open_capture(app, "cap1")
    client.post(f"{SESSION_URL}/pause")
    app.state.frame_observers[0].start()

    client.post(f"{SESSION_URL}/resume")

    assert len(_producers(app)) == 2


def test_stopping_ends_the_session_and_the_producer(app, client):
    client.post(f"{SESSION_URL}/start")
    _open_capture(app)

    body = client.post(f"{SESSION_URL}/stop").json()

    assert body["state"] == "stopped"
    assert _producers(app)[0].terminated is True


# -- step 4: the producer and the read routes cannot disagree ----------


def test_the_producer_is_given_exactly_the_root_the_routes_read(app, client):
    """The 404 that cost the 2026-08-26 run its first half hour.

    The producer defaulted to `data/object_memory` and the web process
    defaulted to nothing, so the memory was written where nothing served
    it. There is now one value and both sides are handed it.
    """
    client.post(f"{SESSION_URL}/start")
    _open_capture(app)

    argv = _producers(app)[0].args
    assert argv[argv.index("--root") + 1] == app.state.object_memory_root


def test_an_unconfigured_tower_still_serves_its_own_memory(monkeypatch, tmp_path):
    """No environment variable, and the read routes answer anyway.

    This is the reversed default. Nothing has been observed yet, so the
    answer is an empty listing rather than a 404 -- which is a different
    claim, and the right one: "this Tower serves object memory and has
    none" is not "this Tower serves no object memory".
    """
    monkeypatch.delenv("TOWER_OBSERVATION_ROOT", raising=False)
    monkeypatch.delenv("TOWER_OBSERVATION_ENABLED", raising=False)
    monkeypatch.setenv("TOWER_CAPTURE_ROOT", str(tmp_path / "capture"))
    client = TestClient(create_app())

    response = client.get("/object-memory/observations")

    assert response.status_code == 200
    assert response.json()["observation_count"] == 0


def test_switching_the_cartridge_off_is_still_reachable(monkeypatch, tmp_path):
    """The 404 state iOS renders as "this Tower serves no object memory".

    It used to be the default and is now deliberate. The state still
    exists, because a Tower that should not remember anything must be
    able to say so, and the iOS surface has copy for exactly this.
    """
    monkeypatch.setenv("TOWER_OBSERVATION_ENABLED", "false")
    monkeypatch.setenv("TOWER_OBSERVATION_ROOT", str(tmp_path / "memory"))
    monkeypatch.setenv("TOWER_CAPTURE_ROOT", str(tmp_path / "capture"))
    client = TestClient(create_app())

    assert client.get("/object-memory/observations").status_code == 404


def test_a_switched_off_cartridge_has_no_producer_to_start(monkeypatch, tmp_path):
    monkeypatch.setenv("TOWER_OBSERVATION_ENABLED", "false")
    monkeypatch.setenv("TOWER_CAPTURE_ROOT", str(tmp_path / "capture"))
    application = create_app()
    client = TestClient(application)

    assert OBJECT_MEMORY_WORKER not in application.state.capture_workers.worker_names()
    body = client.get(SESSION_URL).json()
    assert body["supported"] is False
    assert client.post(f"{SESSION_URL}/start").status_code == 409


# -- the control surface itself ----------------------------------------


def test_the_session_starts_stopped(client):
    body = client.get(SESSION_URL).json()

    assert body["state"] == "stopped"
    assert body["supported"] is True
    assert body["contract"] == "cartridge_session.control/2026-08-27"


def test_the_payload_says_that_state_is_intent_not_liveness(client):
    """The claim carried as a value, not only as a sentence in a document.

    An ACTIVE session whose producer died is the "looks successful but
    does nothing" failure this whole surface exists to make visible, and
    a client that reads `state` alone will draw "remembering" for the
    rest of a walk that remembered nothing.
    """
    body = client.get(SESSION_URL).json()

    assert body["state_means"] == "intent-not-liveness"
    assert "following" in body


def test_an_unknown_cartridge_is_a_404(client):
    response = client.get("/cartridges/teapot/session")

    assert response.status_code == 404


def test_an_unknown_action_is_refused(client):
    response = client.post(f"{SESSION_URL}/rewind")

    assert response.status_code == 409
    assert response.json()["detail"]["reason"] == "unknown-action"


def test_pausing_a_stopped_session_is_refused_with_its_real_state(client):
    response = client.post(f"{SESSION_URL}/pause")

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["reason"] == "not-active"
    assert detail["state"] == "stopped"


def test_starting_twice_reports_that_nothing_changed(client):
    client.post(f"{SESSION_URL}/start")

    body = client.post(f"{SESSION_URL}/start").json()

    assert body["accepted"] is True
    assert body["changed"] is False


def test_stop_is_never_refused(client):
    """The one action that always works, from any state.

    Refusing it would leave the only way out of a confused state being a
    Tower restart.
    """
    assert client.post(f"{SESSION_URL}/stop").status_code == 200


def test_the_session_lists_the_captures_it_followed(app, client):
    client.post(f"{SESSION_URL}/start")
    _open_capture(app, "cap1")

    body = client.get(SESSION_URL).json()

    assert body["captures"] == ["cap1"]
    assert body["following"] == ["cap1"]


def test_health_reports_the_session_beside_the_workers(app, client):
    client.post(f"{SESSION_URL}/start")

    body = client.get("/health").json()

    assert body["cartridge_sessions"][CARTRIDGE_OBJECT_MEMORY]["state"] == "active"
    assert OBJECT_MEMORY_WORKER in body["capture_workers"]["configured"]
    assert WORLD_BUILD_WORKER in body["capture_workers"]["configured"]


def test_the_session_handlers_are_sync_so_a_pause_cannot_stall_the_frame_path():
    """`detach` waits for a producer to finish its record before killing it.

    Up to `DETACH_GRACE_SECONDS`. On the event loop that is a five-second
    stall on every frame the Tower is serving, which is why these are
    declared `def` and run in FastAPI's threadpool -- the same reason
    `tower/routes/geometry.py` gives for the same choice.
    """
    import inspect

    from tower.routes import sessions

    for handler in (sessions.read_session, sessions.apply_session_action):
        assert not inspect.iscoroutinefunction(handler), handler.__name__


class TestASessionDoesNotOutliveEveryClient:
    """A phone that goes away for good takes its cartridge sessions with it.

    A `CartridgeSession` is INTENT and lives only in this process's memory.
    Until 2026-09-07 nothing ended one when the phone that opened it went
    away: a wearer who pressed Start and whose phone then crashed left the
    gate open for as long as the Tower ran, and the next capture -- from any
    cartridge, days later -- attached a producer with nobody having asked
    for it. For Object Memory that is a recorder starting itself on somebody
    else's walk.

    `main.py` already states the rule these tests restore: cartridge sessions
    are "deliberately NOT persisted anywhere. A Tower that restarts comes
    back with every cartridge stopped, because resuming a memory of what a
    camera sees without anybody asking again is the wrong default."

    The second test is the other half, and is why this is gated on the LAST
    connection rather than on the closing one: iOS reconnects in about half
    a second while uvicorn can take 20-40 s to notice the old socket died,
    so a superseded connection's teardown runs while the new one is already
    live. A WiFi hiccup must not end a walk.
    """

    def test_the_session_stops_when_the_last_connection_closes(self, app, client):
        assert client.post(f"{SESSION_URL}/start").json()["state"] == "active"

        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "ping"})
            ws.receive_json()

        assert client.get(SESSION_URL).json()["state"] == "stopped", (
            "a session nobody is connected to any more must not stay active: "
            "the next capture would attach a producer with nobody asking"
        )

    def test_a_reconnect_does_not_end_the_session(self, app, client):
        assert client.post(f"{SESSION_URL}/start").json()["state"] == "active"

        # The new connection is already live when the old one tears down --
        # the ordering `ConnectionTracker` was written for.
        with client.websocket_connect("/ws") as first:
            first.send_json({"type": "ping"})
            first.receive_json()
            with client.websocket_connect("/ws") as second:
                second.send_json({"type": "ping"})
                second.receive_json()
                # The first socket closes here, inside the second's lifetime.
            assert client.get(SESSION_URL).json()["state"] == "active", (
                "a superseded connection's teardown must not end a session "
                "another connection is still holding"
            )

    def test_a_stopped_session_is_left_alone(self, app, client):
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "ping"})
            ws.receive_json()

        assert client.get(SESSION_URL).json()["state"] == "stopped"
