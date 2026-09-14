"""Start, walk, stop, world -- with nothing typed in a second terminal.

This is the one test that exercises the actual product claim end to end:
a real ASGI app, a real WebSocket, a real `CaptureRecorder`, a real
`world_build_session.py` in a REAL subprocess, a real world on disk, and
the real result channel reporting it.

Everything else in this workstream fakes the spawn, which is right for
testing bookkeeping and wrong for testing that the thing actually runs.
The 2026-08-24 failure was not a bookkeeping error -- every individual
piece worked. What was missing was anybody connecting them.

Slow by nature: a Python subprocess has to start, import OpenCV, tail a
journal and run a build. That cost is the point. A fast version of this
test would be a fake, and a fake is what let the gap exist.
"""

import base64
import asyncio
import json
import time

import numpy as np
import pytest

pytestmark = pytest.mark.slow


def _frames(count: int) -> list[str]:
    """Textured noise that shifts, so tracking survives and keyframes land.

    Noise rather than a rendered scene: this test asserts that a world
    gets BUILT, not that it is geometrically correct. The synthetic
    renderer belongs to the tests that make claims about geometry.
    """
    import cv2

    rng = np.random.default_rng(7)
    base = rng.integers(0, 255, (640, 800, 3), dtype=np.uint8)
    out = []
    for index in range(count):
        # Pan across a wider image: real displacement between frames, so
        # the keyframe policy has motion to measure.
        window = base[:, index * 4 : index * 4 + 360]
        ok, buffer = cv2.imencode(".jpg", window)
        assert ok
        out.append(base64.b64encode(buffer.tobytes()).decode("ascii"))
    return out


def _frame_message(seq: int, data: str) -> dict:
    return {
        "type": "frame",
        "seq": seq,
        "width": 360,
        "height": 640,
        "format": "jpeg",
        "data": data,
    }


def _wait_for(predicate, timeout: float, what: str):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.1)
    raise AssertionError(f"timed out after {timeout}s waiting for {what}")


@pytest.fixture
def tower(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from tower.main import create_app

    monkeypatch.setenv("TOWER_CAPTURE_ROOT", str(tmp_path / "capture"))
    monkeypatch.setenv("TOWER_WORLD_ROOT", str(tmp_path / "world"))
    # Rebuild often: this walk is short, and the whole question is
    # whether geometry appears DURING it rather than only at the end.
    monkeypatch.setenv("TOWER_WORLD_REBUILD_EVERY", "2")
    app = create_app()
    client = TestClient(app)
    # The World Builder workspace is on the phone's screen: since 2026-09-06
    # a builder attaches to a capture only while this session is active,
    # exactly as the object-memory producer always has.
    assert client.post(f"{WORLD_BUILDER_SESSION_URL}/start").status_code == 200
    yield client, app, tmp_path
    # Never leave a follower running, whatever the assertions did.
    app.state.capture_workers.shutdown(grace_seconds=5.0)


WORLD_BUILDER_SESSION_URL = "/cartridges/world_builder/session"


def test_a_capture_with_world_builder_inactive_attaches_no_builder(tower):
    """The cartridge-switch requirement: a camera session started for some
    other cartridge must not spawn a builder and its solver children.
    `TOWER_WORLD_AUTOBUILD` is on; the SESSION is what is off."""
    client, app, tmp_path = tower
    assert client.post(f"{WORLD_BUILDER_SESSION_URL}/stop").status_code == 200
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "stream_start"})
        for index, data in enumerate(_frames(6), start=1):
            ws.send_json(_frame_message(index, data))
            ws.receive_json()
        assert app.state.capture_workers.status() == []
        ws.send_json({"type": "stream_stop"})
        ws.send_json({"type": "ping"})
        ws.receive_json()
    assert not (tmp_path / "world" / "worlds").exists() or not list(
        (tmp_path / "world" / "worlds").iterdir()
    )


def test_entering_world_builder_mid_capture_attaches_a_builder_from_now(tower):
    """The wearer started the camera elsewhere and then opened World
    Builder: `start` attaches to the capture already recording."""
    client, app, tmp_path = tower
    assert client.post(f"{WORLD_BUILDER_SESSION_URL}/stop").status_code == 200
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "stream_start"})
        for index, data in enumerate(_frames(6), start=1):
            ws.send_json(_frame_message(index, data))
            ws.receive_json()
        assert app.state.capture_workers.status() == []
        body = client.post(f"{WORLD_BUILDER_SESSION_URL}/start").json()
        capture_id = app.state.frame_observers[0].status.capture_id
        assert body["attached_capture_id"] == capture_id
        workers = _wait_for(app.state.capture_workers.status, 10.0, "a builder")
        assert workers[0]["capture_id"] == capture_id
        ws.send_json({"type": "stream_stop"})
        ws.send_json({"type": "ping"})
        ws.receive_json()
    _wait_for(lambda: not app.state.capture_workers.status(), 90.0, "the builder to finish")


def test_start_walk_stop_produces_a_world_with_no_manual_step(tower):
    """The whole product claim, from the socket to the persisted world.

    Nothing in this test names a capture id. That is the point: on
    2026-08-24 a human had to read one off a directory listing and pass
    it to a second process by hand.
    """
    client, app, tmp_path = tower
    frames = _frames(24)

    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "stream_start"})
        for index, data in enumerate(frames, start=1):
            ws.send_json(_frame_message(index, data))
            ws.receive_json()

        recorder = app.state.frame_observers[0]
        capture_id = recorder.status.capture_id

        # A worker was attached to THIS capture, without being told.
        workers = _wait_for(
            app.state.capture_workers.status, 10.0, "a worker to be running"
        )
        assert workers[0]["capture_id"] == capture_id

        ws.send_json({"type": "stream_stop"})
        ws.send_json({"type": "ping"})
        ws.receive_json()

        # The capture closed, so the follower observes completion,
        # finalises and exits on its own. Nothing kills it -- WHILE THE
        # PHONE STAYS ON THE SCREEN, which is what this socket is. It is
        # held open until the builder is done because that is what the
        # phone does after Stop (the World Builder screen stays up,
        # subscribed, until the world is ready) and because the product's
        # documented answer to "the last client left" is a SOFT stop:
        # the builder stops observing, closes the session `interrupted`
        # and skips the final solve (`StopRequest` in
        # `world_build_session.py`; §14.13 of the handoff). This test
        # used to close the socket first and passed anyway, because the
        # test client's cancellation swallowed that stop; once the
        # teardown was made cancellation-proof the builder was soft-
        # stopped 0.6 s in, with zero keyframes, five times in five.
        _wait_for(
            lambda: not app.state.capture_workers.status(),
            90.0,
            "the worker to finish and be reaped",
        )

    worlds = list((tmp_path / "world" / "worlds").iterdir())
    assert len(worlds) == 1, f"expected exactly one world, got {worlds}"
    world_dir = worlds[0]

    sessions = list((world_dir / "sessions").iterdir())
    assert len(sessions) == 1
    session = json.loads((sessions[0] / "session.json").read_text(encoding="utf-8"))

    assert session["capture_id"] == capture_id, (
        "the world was built from a different capture than the one the "
        "phone streamed"
    )
    assert session["frame_source"] == "live-capture"
    assert session["ended_at"] is not None, "the session was never closed"
    assert session["keyframes_accepted"] > 0

    manifest = json.loads(
        (world_dir / "derived" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["keyframes"] > 0


def test_the_running_world_is_reported_to_a_subscriber_while_it_builds(tower):
    """Live, over the wire the phone actually uses.

    Not a disk assertion: iOS reads the result channel, and "a world
    exists on disk" is a different claim from "the phone is being told
    about it". On 2026-08-24 the phone was told there was no world, and
    that was true.
    """
    client, app, tmp_path = tower
    frames = _frames(24)

    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "stream_start"})
        for index, data in enumerate(frames, start=1):
            ws.send_json(_frame_message(index, data))
            ws.receive_json()

        ws.send_json(
            {
                "type": "result_subscribe",
                "cartridge": "world_builder",
                "result_type": "status",
            }
        )

        # Drain until the channel reports a world with keyframes. The
        # first snapshot may legitimately arrive before the follower has
        # written anything.
        deadline = time.monotonic() + 60.0
        snapshot = None
        while time.monotonic() < deadline:
            message = ws.receive_json()
            if message.get("type") != "cartridge_result":
                continue
            payload = message["payload"]
            if (payload.get("world_snapshot") or {}).get("keyframe_count"):
                snapshot = payload["world_snapshot"]
                break
        assert snapshot is not None, (
            "no world with keyframes was reported within 60s while the "
            "stream was open"
        )

        assert snapshot["keyframe_count"] > 0
        assert snapshot["world_id"]
        # Uncalibrated, so this must stay honest whatever else it says.
        assert snapshot["calibration"] == "uncalibrated"
        assert snapshot["scale"] == "unknown"
        assert snapshot["trajectory"]["pose_count"] in (0, None), (
            "an uncalibrated build reported camera poses; this is the "
            "2026-08-24 defect, over the wire"
        )

        ws.send_json({"type": "stream_stop"})

    app.state.capture_workers.shutdown(grace_seconds=60.0)


def test_a_reconnect_keeps_one_world_and_one_worker(tower):
    """A WiFi hiccup mid-walk must not fork the walk into two worlds.

    `handoff.md` 9.3 makes this the EXPECTED case on this link, not an
    edge case: the socket dies, iOS reconnects in about half a second and
    re-sends stream_start with `seq` continuing. The recorder declares
    the new capture a successor; the follower chains into it; and the
    supervisor must NOT start a second builder.
    """
    client, app, tmp_path = tower
    frames = _frames(24)
    recorder = app.state.frame_observers[0]

    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "stream_start"})
        for index, data in enumerate(frames[:12], start=1):
            ws.send_json(_frame_message(index, data))
            ws.receive_json()
        first_capture = recorder.status.capture_id
        _wait_for(app.state.capture_workers.status, 10.0, "the first worker")

    # The socket dropped without a stream_stop, exactly as a dead WiFi
    # link does. The phone comes back and re-opens the bracket.
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "stream_start"})
        for index, data in enumerate(frames[12:], start=13):
            ws.send_json(_frame_message(index, data))
            ws.receive_json()
        second_capture = recorder.status.capture_id

        assert second_capture != first_capture
        manifest = json.loads(
            (recorder.capture_dir(second_capture) / "capture.json").read_text(
                encoding="utf-8"
            )
        )
        assert manifest["continues_capture"] == first_capture, (
            "the recorder did not record the lineage, so nothing downstream "
            "can know this is one walk"
        )

        workers = app.state.capture_workers.status()
        assert len(workers) == 1, (
            f"a reconnect started a second builder: {workers}"
        )
        assert second_capture in workers[0]["lineage"]

        ws.send_json({"type": "stream_stop"})
        ws.send_json({"type": "ping"})
        ws.receive_json()

    _wait_for(
        lambda: not app.state.capture_workers.status(),
        90.0,
        "the worker to finish",
    )

    worlds = list((tmp_path / "world" / "worlds").iterdir())
    assert len(worlds) == 1, (
        f"one walk produced {len(worlds)} worlds; a reconnect forked it"
    )


def test_a_superseding_stream_start_continues_the_walk(tower):
    """The FIRST of two ways one reconnect became two worlds.

    iOS reconnects in about half a second while uvicorn takes 20-40 s to
    notice the old socket died, so the new connection's `stream_start`
    arrives while the old capture is still recording. `_start_capture`
    stops that capture to open the new one -- and it stopped it as
    `END_REASON_STOP`, which `resumable_capture()` does not offer as a
    predecessor. `continues_capture` was never set; the old builder saw a
    politely closed capture and finalised world 1; the new capture began
    world 2. A dress-rehearsal reviewer drove it through a real Tower at
    four timings and got two half-walks, "Complete", every time.

    `test_a_reconnect_keeps_one_world_and_one_worker` above exercises the
    disconnect path -- the first socket is gone before the second opens.
    This one holds BOTH open, which is the supersession path, and the one
    the field link actually takes.
    """
    client, app, tmp_path = tower
    frames = _frames(24)
    recorder = app.state.frame_observers[0]
    with client.websocket_connect("/ws") as first:
        first.send_json({"type": "stream_start"})
        for index, data in enumerate(frames[:12], start=1):
            first.send_json(_frame_message(index, data))
            first.receive_json()
        first_capture = recorder.status.capture_id
        assert first_capture is not None

        # The old socket is still open -- the Tower has not noticed
        # anything -- when the phone's new socket re-opens the bracket.
        with client.websocket_connect("/ws") as second:
            second.send_json({"type": "stream_start"})
            for index, data in enumerate(frames[12:24], start=13):
                second.send_json(_frame_message(index, data))
                second.receive_json()
            second_capture = recorder.status.capture_id
            assert second_capture != first_capture

            first_manifest = json.loads(
                (recorder.capture_dir(first_capture) / "capture.json").read_text(
                    encoding="utf-8"
                )
            )
            second_manifest = json.loads(
                (recorder.capture_dir(second_capture) / "capture.json").read_text(
                    encoding="utf-8"
                )
            )
            assert first_manifest["end_reason"] == "disconnect", (
                "a superseded capture was closed as a polite stop, which is "
                "what told the old builder the walk was over"
            )
            assert second_manifest["continues_capture"] == first_capture, (
                "the superseding capture did not declare its predecessor, so "
                "this is two walks on disk"
            )
            second.send_json({"type": "stream_stop"})


def test_the_last_client_leaving_does_not_end_a_walk_inside_the_resume_grace(tower):
    """The SECOND way: the Tower notices the drop before the phone returns.

    A capture that ended by disconnect is inside its 90 s resume grace,
    and the builder is sitting in `_await_successor` for it. The last
    connection going away used to stop every cartridge session -- which
    closed that builder's stdin, which `should_stop` honours at once:
    world 1 finalised, and the phone that came back 60 s later started
    world 2. Only World Builder is deferred; it is the cartridge whose
    session follows a capture lineage, and its builder finishes on its
    own at the end of the grace if nobody returns.
    """
    client, app, tmp_path = tower
    frames = _frames(24)
    recorder = app.state.frame_observers[0]
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "stream_start"})
        for index, data in enumerate(frames[:12], start=1):
            ws.send_json(_frame_message(index, data))
            ws.receive_json()
        _wait_for(app.state.capture_workers.status, 10.0, "the first worker")
    # The socket dropped without a stream_stop and nobody else is
    # connected: this is the moment every session used to be stopped.
    assert recorder.resumable_capture() is not None, (
        "the fixture's drop did not leave a resumable capture; the test "
        "cannot say anything"
    )
    # CALLED DIRECTLY, not observed through the client's teardown.
    # Starlette's TestClient cancels the handler at the first `await` of
    # the disconnect `finally`, so `_stop_cartridge_sessions` never runs
    # under it -- a reviewer measured `stop calls: []` for every
    # TestClient run -- and a test that waited for it would pass whatever
    # the function did. The real-uvicorn reconnect harness is the
    # end-to-end proof; this pins the decision itself.
    from types import SimpleNamespace

    from tower.routes.ws import _stop_cartridge_sessions

    _stop_cartridge_sessions(SimpleNamespace(app=app))
    sessions = app.state.cartridge_sessions
    assert sessions["world_builder"].state != "stopped", (
        "the World Builder session was stopped while its capture was "
        "inside the resume grace -- the builder's stdin was closed and the "
        "walk ended at the first WiFi blip"
    )

def test_a_deferred_walk_is_stopped_once_its_grace_is_over(tower, monkeypatch):
    """The deferral is a delay, not an exemption -- and a narrow one.

    Nothing revisited the decision to leave World Builder running for a
    walk inside its resume grace, so a phone that never came back left
    the session `active` forever -- and the NEXT phone on ANY screen got
    a builder attached that nobody asked for, and a second world built
    from its frames. Reproduced end to end by a reviewer. Once the grace
    is over and nobody has asked for World Builder since, that walk is
    stopped.

    THAT walk, and nothing else. The first follow-up re-ran the whole
    last-client stop, and a reviewer drove it on a real Tower: a phone
    that never touched World Builder left; a second phone started Object
    Memory and walked; 106 s later its producer was SIGBREAK'd mid-walk.
    """
    from types import SimpleNamespace

    from tower.routes import ws as ws_module

    client, app, tmp_path = tower
    frames = _frames(12)
    recorder = app.state.frame_observers[0]
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "stream_start"})
        for index, data in enumerate(frames, start=1):
            ws.send_json(_frame_message(index, data))
            ws.receive_json()
        _wait_for(app.state.capture_workers.status, 10.0, "the first worker")
    assert recorder.resumable_capture() is not None

    websocket = SimpleNamespace(app=app)
    monkeypatch.setattr(ws_module, "RESUME_GRACE_SECONDS", 0.0)
    monkeypatch.setattr(ws_module, "_DEFERRED_STOP_MARGIN_SECONDS", 0.0)
    sessions = app.state.cartridge_sessions
    world_builder = sessions["world_builder"]
    walk = world_builder.snapshot()["session_id"]

    # Another phone's Object Memory, started after the drop: not this
    # task's business, whatever it decides about World Builder.
    assert client.post("/cartridges/object_memory/session/start").json()["state"] == "active"

    # Somebody did ask: `session/start` moved `requested_at` -- even
    # though the session was still active and nothing else changed. Left
    # alone, even with nobody connected.
    requested_before = world_builder.snapshot()["requested_at"]
    assert client.post(f"{WORLD_BUILDER_SESSION_URL}/start").status_code == 200
    asyncio.run(ws_module._stop_world_builder_after_grace(
        websocket, walk, requested_before))
    assert world_builder.state != "stopped", (
        "a walk somebody had restarted was stopped by the grace follow-up"
    )

    # A follow-up armed for a DIFFERENT walk finds this one and leaves it.
    requested_before = world_builder.snapshot()["requested_at"]
    asyncio.run(ws_module._stop_world_builder_after_grace(
        websocket, "some-earlier-walk", requested_before))
    assert world_builder.state != "stopped"

    # Nobody asked since the deferral: stopped, even though some OTHER
    # client is on the socket now -- and only World Builder.
    asyncio.run(ws_module._stop_world_builder_after_grace(
        websocket, walk, requested_before))
    assert world_builder.state == "stopped", (
        "a walk nobody asked for again was left active because a different "
        "client was connected"
    )
    assert sessions["object_memory"].state == "active", (
        "the World Builder follow-up stopped another cartridge's session"
    )


def test_the_grace_follow_up_is_armed_only_for_a_deferred_walk(tower, monkeypatch):
    """Nothing is armed unless the stop is leaving a World Builder walk
    running, and a walk gets one clock, not one per drop.

    A reviewer traced that arming on every last-client disconnect (and
    cancelling the previous task each time) meant a phone reconnecting
    and dropping every 60 s never let the follow-up fire.
    """
    from types import SimpleNamespace

    from tower.routes import ws as ws_module

    client, app, tmp_path = tower
    websocket = SimpleNamespace(app=app)
    state = app.state

    async def arm():
        ws_module._arm_world_builder_follow_up(websocket)
        return getattr(state, "world_builder_grace_stop", None)

    # World Builder never asked for: nothing to defer, nothing armed.
    assert asyncio.run(arm()) is None

    # A deferred walk: armed once; a second drop for the same walk keeps
    # the first task and its deadline.
    frames = _frames(12)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "stream_start"})
        for index, data in enumerate(frames, start=1):
            ws.send_json(_frame_message(index, data))
            ws.receive_json()
        _wait_for(app.state.capture_workers.status, 10.0, "the first worker")
    assert app.state.frame_observers[0].resumable_capture() is not None

    async def arm_twice():
        ws_module._arm_world_builder_follow_up(websocket)
        first = state.world_builder_grace_stop
        ws_module._arm_world_builder_follow_up(websocket)
        second = state.world_builder_grace_stop
        same = first is second and not first.done()
        first.cancel()
        return same

    assert asyncio.run(arm_twice()), (
        "a second disconnect for the same walk restarted its grace clock"
    )


def test_a_successor_is_not_chained_into_a_worker_that_was_asked_to_stop(tower):
    """The rest of the walk was recorded and built by nobody.

    A dress-rehearsal reviewer cut the link, left the World Builder screen
    during the outage (`session/stop` -> the builder's stdin closed while
    it sat in `_await_successor`), came back and reconnected. The
    successor capture was chained into that worker because it was still
    `is_alive()` -- it was finishing its final build -- and it exited
    without ever following: 1,200 frames on disk, `health.workers=[]`,
    while the phone showed the sentence §15 says means success.

    An asked-to-stop worker is alive and not following. A successor that
    names its lineage must get a builder of its own.
    """
    client, app, tmp_path = tower
    frames = _frames(24)
    recorder = app.state.frame_observers[0]
    supervisor = app.state.capture_workers

    with client.websocket_connect("/ws") as first:
        first.send_json({"type": "stream_start"})
        for index, data in enumerate(frames[:12], start=1):
            first.send_json(_frame_message(index, data))
            first.receive_json()
        first_capture = recorder.status.capture_id
        _wait_for(supervisor.status, 10.0, "the first worker")
        original = supervisor.status()[0]["pid"]

        # The wearer leaves the screen: the builder is ASKED to stop. It
        # stays registered, alive, finishing -- and will not follow.
        assert client.post(f"{WORLD_BUILDER_SESSION_URL}/stop").status_code == 200

        # ...and comes back. The app's reconnect gesture is resubscribe,
        # `session/start`, then `stream_start` if the camera is streaming
        # -- the start is what reopens the World Builder gate; without it
        # no builder could attach to anything and this test would prove
        # nothing.
        assert client.post(f"{WORLD_BUILDER_SESSION_URL}/start").status_code == 200
        with client.websocket_connect("/ws") as second:
            second.send_json({"type": "stream_start"})
            for index, data in enumerate(frames[12:24], start=13):
                second.send_json(_frame_message(index, data))
                second.receive_json()
            second_capture = recorder.status.capture_id
            assert second_capture != first_capture

            def a_fresh_builder():
                return any(
                    w["pid"] != original and second_capture in w["lineage"]
                    for w in supervisor.status()
                )
            _wait_for(a_fresh_builder, 10.0, "a builder of its own for the successor")
            assert not any(
                w["pid"] == original and second_capture in w["lineage"]
                for w in supervisor.status()
            ), "the successor was chained into the worker that was asked to stop"
            second.send_json({"type": "stream_stop"})
