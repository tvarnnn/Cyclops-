"""Isolation: the channel must not touch the frame path or another cartridge.

The requirement these pin is negative -- "publishing changes nothing else"
-- which is the kind that quietly stops being true. So they are written
against observable behaviour (the bytes on the wire, the frames on disk,
the module's state) rather than against intent.
"""

import ast
import pathlib

import pytest

from tests.result_channel_fixtures import (  # noqa: F401
    _close_result_channel_clients,
    build_world,
    drain,
    jpeg_frame,
    make_client,
    pump,
    subscribe,
)

TOWER = pathlib.Path("tower")


@pytest.fixture(scope="module")
def world_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("isolation")
    build_world(root, frames=10)
    return root


def _frame(seq: int, data: str) -> dict:
    return {
        "type": "frame",
        "seq": seq,
        "width": 640,
        "height": 360,
        "format": "jpeg",
        "data": data,
    }


# -- the frame path is unchanged ---------------------------------------


def test_frame_results_are_identical_with_and_without_a_subscription(
    monkeypatch, world_root
):
    """The frame reply must not gain, lose or reorder a single field.

    iOS decodes `frame_result` in a Debug build today. A subscription is a
    separate concern and must be invisible to it.
    """
    client = make_client(monkeypatch, world_root)
    data = jpeg_frame()

    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "stream_start"})
        ws.send_json(_frame(1, data))
        without = drain(ws, expect="frame_result")

    with client.websocket_connect("/ws") as ws:
        subscribe(ws)
        drain(ws, expect="cartridge_result")
        ws.send_json({"type": "stream_start"})
        ws.send_json(_frame(1, data))
        with_subscription = drain(ws, expect="frame_result")

    assert sorted(without) == sorted(with_subscription)
    for field in ("type", "seq", "result_label", "result_value", "mean_intensity"):
        assert without[field] == with_subscription[field]
    # processing_ms legitimately differs run to run; its PRESENCE and type
    # are the contract, not its value.
    assert isinstance(with_subscription["processing_ms"], float)


def test_publishing_does_not_change_what_capture_records(
    monkeypatch, tmp_path, world_root
):
    """Recording is a side errand and must stay one.

    A subscription running alongside must not add, drop or reorder a
    recorded frame -- the dataset is the authoritative artifact and a
    reporting surface has no business perturbing it.
    """
    from tower.capture import CaptureRecorder

    data = jpeg_frame()

    def _record(subscribe_first: bool, root):
        monkeypatch.setenv("TOWER_CAPTURE_ROOT", str(root))
        client = make_client(monkeypatch, world_root)
        with client.websocket_connect("/ws") as ws:
            if subscribe_first:
                subscribe(ws)
                drain(ws, expect="cartridge_result")
            ws.send_json({"type": "stream_start"})
            for seq in range(1, 6):
                ws.send_json(_frame(seq, data))
                drain(ws, expect="frame_result")
            ws.send_json({"type": "stream_stop"})
        recorder = client.app.state.frame_observers[0]
        assert isinstance(recorder, CaptureRecorder)
        status = recorder.status
        return recorder.read_frames(status.capture_id)

    plain = _record(False, tmp_path / "plain")
    watched = _record(True, tmp_path / "watched")

    assert len(plain) == len(watched) == 5
    assert [f["wire_seq"] for f in plain] == [f["wire_seq"] for f in watched]
    assert [f["width"] for f in plain] == [f["width"] for f in watched]


def test_a_failing_result_channel_leaves_the_frame_path_working(
    monkeypatch, world_root
):
    """A push-channel failure must never implicate the connection.

    The receive loop is what answers frames; a broken snapshot producer
    must not reach it.
    """
    client = make_client(monkeypatch, world_root)

    def _explode(*args, **kwargs):
        raise RuntimeError("the producer is broken")

    client.app.state.result_hub._snapshot_for = _explode

    with client.websocket_connect("/ws") as ws:
        reply = subscribe(ws)
        # Subscribing computes the first snapshot, so the failure surfaces
        # HERE as a legible refusal -- not as silence, and not as a
        # connection error.
        assert reply["type"] == "result_error"
        assert reply["reason"] == "snapshot_failed"
        assert "RuntimeError" in reply["message"]

        ws.send_json({"type": "stream_start"})
        ws.send_json(_frame(1, jpeg_frame()))
        result = drain(ws, expect="frame_result")

    assert result["result_label"] == "mean_intensity"


def test_the_module_container_is_untouched_by_a_subscription(
    monkeypatch, world_root
):
    """The cartridge's own state must not move because someone watched."""
    client = make_client(monkeypatch, world_root)
    container = client.app.state.module_container
    before = (container.state.value, container.descriptor.id)

    with client.websocket_connect("/ws") as ws:
        subscribe(ws)
        drain(ws, expect="cartridge_result")
        pump(client, times=3)

    assert (container.state.value, container.descriptor.id) == before


# -- cross-cartridge isolation -----------------------------------------


def test_a_world_builder_subscription_only_ever_carries_world_builder(
    monkeypatch, world_root
):
    client = make_client(monkeypatch, world_root)
    with client.websocket_connect("/ws") as ws:
        reply = subscribe(ws)
        envelopes = [drain(ws, expect="cartridge_result")]
        for _ in range(3):
            pump(client)
            envelopes.append(drain(ws, expect="cartridge_result"))

    for envelope in envelopes:
        assert envelope["cartridge"] == "world_builder"
        assert envelope["result_type"] == "status"
        assert envelope["subscription_id"] == reply["subscription_id"]


def test_no_other_cartridge_can_be_subscribed_to(monkeypatch, world_root):
    """The registry is the only gate, and it refuses everything else.

    Two lists, because there are two different refusals and a client
    switches on the difference. `unknown_cartridge` means this Tower has
    never heard of it -- iOS's "not built yet". `cartridge_unavailable`
    means the contract exists and this Tower is not configured to serve
    it -- iOS's "connect". Collapsing them would tell a person to give up
    on a cartridge that one environment variable would turn on.

    `experimental_cv`, `document_memory` and `scene_understanding` all
    left the first list on 2026-08-27, each having gained a contract of
    its own. Which list they land in now depends on configuration rather
    than on this build: `make_client` builds the real app, so a CV Lab
    always exists and `experimental_cv` is genuinely subscribable here --
    it is covered by `tests/test_cv_lab_protocol.py`. The fixture sets a
    world root and nothing else, so the other two are declared and
    UNAVAILABLE, which is the second loop below.

    That leaves two cartridges this Tower really has never heard of:
    `translator`, which does not exist, and `object_memory`, which has a
    control surface and a store but is deliberately absent from
    `registry.declare()` until the iOS lane can take the declaration and
    its pinned test together. Its presence here is load-bearing -- if
    somebody declares Object Memory without that coordination, this line
    is what notices.
    """
    # Document Memory has a managed root since 2026-09-07, so on a stock
    # Tower it IS subscribable. "Declared and unavailable" is reachable
    # only by switching it off, which is the configuration this test
    # describes.
    monkeypatch.setenv("TOWER_DOCUMENT_ENABLED", "false")
    client = make_client(monkeypatch, world_root)
    with client.websocket_connect("/ws") as ws:
        for cartridge in ("object_memory", "translator"):
            ws.send_json(
                {
                    "type": "result_subscribe",
                    "cartridge": cartridge,
                    "result_type": "status",
                }
            )
            error = drain(ws, expect="result_error")
            assert error["reason"] == "unknown_cartridge", cartridge

        for cartridge, result_type in (
            ("scene_understanding", "live"),
            ("document_memory", "status"),
        ):
            ws.send_json(
                {
                    "type": "result_subscribe",
                    "cartridge": cartridge,
                    "result_type": result_type,
                }
            )
            error = drain(ws, expect="result_error")
            assert error["reason"] == "cartridge_unavailable", cartridge
            # The refusal has to name the configuration that would fix it.
            # "Unavailable" with no reason is indistinguishable from
            # broken, and a person cannot act on it.
            assert "TOWER_" in error["message"], cartridge


def test_subscriptions_on_different_connections_are_independent(
    monkeypatch, world_root
):
    """Multiple clients, each with its own sequence and its own slot."""
    client = make_client(monkeypatch, world_root)

    with client.websocket_connect("/ws") as first:
        with client.websocket_connect("/ws") as second:
            first_reply = subscribe(first)
            first_envelope = drain(first, expect="cartridge_result")
            second_reply = subscribe(second)
            second_envelope = drain(second, expect="cartridge_result")

            assert first_reply["subscription_id"] == second_reply["subscription_id"], (
                "ids are per connection, so both connections start at sub-1"
            )
            assert first_envelope["seq"] == second_envelope["seq"] == 1
            assert first_envelope["revision"] == second_envelope["revision"], (
                "one shared reader serves both, so they see the same state"
            )

            pump(client)
            assert drain(first, expect="cartridge_result")["seq"] == 2
            assert drain(second, expect="cartridge_result")["seq"] == 2


def test_one_connection_disconnecting_does_not_disturb_another(
    monkeypatch, world_root
):
    client = make_client(monkeypatch, world_root)

    with client.websocket_connect("/ws") as survivor:
        subscribe(survivor)
        drain(survivor, expect="cartridge_result")

        with client.websocket_connect("/ws") as doomed:
            subscribe(doomed)
            drain(doomed, expect="cartridge_result")

        pump(client)
        assert drain(survivor, expect="cartridge_result")["seq"] == 2


# -- vocabulary ---------------------------------------------------------


def test_the_result_channel_uses_no_gaze_or_identity_vocabulary():
    """The same ban the cartridges already carry, extended to the wire.

    IOS-to-Tower.md 4.2 asks explicitly: do not send a field named `gaze`,
    `looking_at` or `attention`, and 3.3 adds `viewing_duration` -- "the
    name is the failure". A wire contract is the worst place for one,
    because a phone ships against it.
    """
    banned = (
        "looking_at",
        "gaze",
        "attention",
        "viewing_duration",
        "is_looking",
        "face_id",
        "person_id",
    )
    offenders = []
    paths = list((TOWER / "results").rglob("*.py"))
    paths.append(TOWER / "routes" / "results_ws.py")
    paths.append(TOWER / "routes" / "cartridges.py")
    # Widened 2026-08-27. The two cartridge control routes serve payloads
    # to the same consumers this channel does, and a review pointed out
    # they were outside the scan -- `SceneEngine.describe()` builds a dict
    # containing the literal string "orientation evidence, not gaze; the
    # camera cannot see attention", and a route that returned it would
    # have shipped the word and passed this test.
    for extra in (
        TOWER / "routes" / "scene.py",
        TOWER / "routes" / "documents.py",
    ):
        if extra.exists():
            paths.append(extra)

    for path in paths:
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Name):
                names.append(node.id)
            elif isinstance(node, ast.Attribute):
                names.append(node.attr)
            elif isinstance(node, ast.arg):
                names.append(node.arg)
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                names.append(node.name)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                names.append(node.value)
            for name in names:
                for word in banned:
                    if word in name:
                        offenders.append(f"{path.name}: {word!r} in {name!r}")

    assert offenders == []
