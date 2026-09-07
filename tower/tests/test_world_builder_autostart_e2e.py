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

    # The capture closed, so the follower observes completion, finalises
    # and exits on its own. Nothing kills it.
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
