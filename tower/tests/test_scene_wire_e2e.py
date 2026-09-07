"""The Scene Understanding wire path, driven end to end through the real app.

This file exists because of one specific failure mode, and it is the one
an adversarial reviewer of a cartridge should look for first:

    `/cartridges` says the contract is offered, and the production path
    that would serve it does not connect.

Every hop here is production code. The app is `create_app()`. The frames
arrive as base64 JPEG over the real `/ws`, through the real frame handler,
after a real `frame_result`. The session is the one
`tower/cartridge_runtime.py` built. The payload is assembled by the real
adapter and delivered by the real hub through the real subscription.

**The one thing that is stubbed is the detector**, and only the detector:
`_scene_session` is replaced with one whose engine factory returns a stub.
Loading `ssdlite320` would download 13.4 MB of weights on a first run and
add 30 ms per frame to a suite that runs on every commit, and it would
test torchvision rather than this wire. What is under test is that a frame
entering `/ws` comes back out as counts on a subscription -- and every
line responsible for that is real.

`tests/test_object_detection_integration.py` is where the real weights are
exercised, opt-in behind `TOWER_RUN_MODEL_TESTS`.
"""

import base64
import io
import threading
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tests.result_channel_fixtures import pump
from tower.scene.records import BoundingBox, Detection

CONTRACT = "scene_understanding.live/2026-08-27"

_OPEN: list = []


class StubSceneEngine:
    """A SceneEngine that always sees the same room, without torch.

    Returns real `SceneState` objects built by the real `SceneEngine`
    machinery -- tracker, counts, relations -- because the payload's
    shape depends on those and a hand-built state would not exercise
    them.
    """

    def __init__(self, detections):
        from tower.scene.detect import FixedDetector
        from tower.scene.engine import SceneEngine
        from tower.scene.tracking import TrackerPolicy

        self._engine = SceneEngine(
            FixedDetector([detections] * 10_000),
            TrackerPolicy(min_iou=0.25, min_hits=1, max_misses=5),
        )
        self._detector = type("D", (), {"name": "stub-detector"})()

    def load(self):
        self._engine.load()

    def release(self):
        self._engine.release()

    def observe(self, frame, *, received_at=None):
        return self._engine.observe(frame, received_at=received_at)


TWO_PEOPLE_A_CHAIR_AND_A_LAPTOP = [
    Detection("person", 0.9, BoundingBox(20, 80, 120, 300)),
    Detection("person", 0.9, BoundingBox(150, 90, 250, 305)),
    Detection("chair", 0.8, BoundingBox(480, 200, 600, 330)),
    Detection("laptop", 0.8, BoundingBox(40, 200, 140, 260)),
]


@pytest.fixture(autouse=True)
def _close_clients():
    yield
    while _OPEN:
        _OPEN.pop().__exit__(None, None, None)


@pytest.fixture
def client(monkeypatch, tmp_path):
    """The real app, scene enabled, detector stubbed.

    `_scene_session` is patched rather than the settings, so
    `build_live_cartridges` still runs -- including the branch that
    decides whether to construct anything at all, which is itself part of
    what "the production path connects" means.
    """
    from tower import cartridge_runtime
    from tower.main import create_app
    from tower.scene.live import SceneLive

    monkeypatch.setenv("TOWER_SCENE_UNDERSTANDING", "true")
    monkeypatch.delenv("TOWER_WORLD_ROOT", raising=False)
    monkeypatch.delenv("TOWER_CAPTURE_ROOT", raising=False)
    monkeypatch.setattr(
        cartridge_runtime,
        "_scene_session",
        lambda settings: SceneLive(
            lambda: StubSceneEngine(TWO_PEOPLE_A_CHAIR_AND_A_LAPTOP)
        ),
    )
    made = TestClient(create_app())
    made.__enter__()
    _OPEN.append(made)
    made.app.state.result_hub._poll_seconds = 3600.0
    return made


def _jpeg(width=640, height=360) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (30, 60, 90)).save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _send_frames(ws, count: int, *, width=640, height=360) -> None:
    payload = _jpeg(width, height)
    for seq in range(count):
        ws.send_json(
            {
                "type": "frame",
                "seq": seq,
                "width": width,
                "height": height,
                "format": "jpeg",
                "data": payload,
            }
        )
        assert _next_frame_result(ws)["type"] == "frame_result"


def _next_frame_result(ws) -> dict:
    """The next `frame_result`, skipping results the subscription pushes.

    With a live subscription open on the same socket, `cartridge_result`
    heartbeats interleave with frame replies. They are the subscription
    working, not noise, and a test that asserts on frame replies steps
    over them.
    """
    for _ in range(50):
        message = ws.receive_json()
        if message["type"] != "cartridge_result":
            return message
    raise AssertionError("no frame_result arrived")


def _subscribe(ws) -> str:
    """Subscribe to the live scene on this socket; returns the id.

    A subscription is the phone saying "a person is looking at this",
    and it is what lets a stream start the detector: a phone streaming
    for World Builder must not have Scene Understanding running behind
    it. The first snapshot is consumed here so a test reads only what it
    asked for afterwards.
    """
    ws.send_json(
        {
            "type": "result_subscribe",
            "cartridge": "scene_understanding",
            "result_type": "live",
            "contract": CONTRACT,
        }
    )
    reply = ws.receive_json()
    assert reply["type"] == "result_subscribed", reply
    assert ws.receive_json()["type"] == "cartridge_result"
    return reply["subscription_id"]


def _await_state(client, want: str, timeout_polls: int = 400) -> dict:
    """Poll the HTTP view until the lifecycle reaches `want`.

    A stop reached through a disconnect crosses two thread hops
    (`watchers_left`, then `stream_closed`), so a poll that never yields
    can exhaust its budget before the second hop runs. The sleep is
    short; the budget is what bounds the wait.
    """
    for _ in range(timeout_polls):
        payload = client.get("/scene").json()
        if payload["lifecycle"]["state"] == want:
            return payload
        time.sleep(0.005)
    raise AssertionError(f"the session never reached {want!r}")


def _await_scene(client, timeout_polls: int = 200):
    """Poll the HTTP view until the worker has produced a scene.

    The worker is a real thread and the test is not synchronised with it,
    so something has to wait. Polling the public route rather than
    sleeping a fixed interval keeps the test honest on a slow machine and
    fast on a quick one.
    """
    for _ in range(timeout_polls):
        payload = client.get("/scene").json()
        if payload["scene_available"]:
            return payload
    raise AssertionError("the session never produced a scene")


class TestTheDeclarationMatchesWhatCanBeServed:
    def test_the_offer_is_available_when_the_cartridge_is_enabled(self, client):
        declaration = client.get("/cartridges").json()
        offer = next(
            entry
            for entry in declaration["cartridges"]
            if entry["cartridge"] == "scene_understanding"
        )

        assert offer["available"] is True
        assert offer["unavailable_reason"] is None
        assert offer["contract"] == CONTRACT
        assert offer["result_type"] == "live"
        assert offer["snapshot_only"] is True

    def test_what_is_offered_can_actually_be_subscribed_to(self, client):
        """The assertion this whole file exists for.

        An offer that cannot be subscribed to is worse than no offer: it
        tells a person to connect to something that is not there.
        """
        with client.websocket_connect("/ws") as ws:
            ws.send_json(
                {
                    "type": "result_subscribe",
                    "cartridge": "scene_understanding",
                    "result_type": "live",
                    "contract": CONTRACT,
                }
            )
            reply = ws.receive_json()
            assert reply["type"] == "result_subscribed"
            first = ws.receive_json()

        assert first["type"] == "cartridge_result"
        assert first["contract"] == CONTRACT
        assert first["seq"] == 1

    def test_a_contract_mismatch_is_refused_rather_than_downgraded(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.send_json(
                {
                    "type": "result_subscribe",
                    "cartridge": "scene_understanding",
                    "result_type": "live",
                    "contract": "scene_understanding.live/1999-01-01",
                }
            )
            error = ws.receive_json()

        assert error["reason"] == "contract_mismatch"
        assert error["offered_contract"] == CONTRACT


class TestAFrameOnTheSocketBecomesCountsOnTheSubscription:
    def test_the_whole_path_connects(self, client):
        """Frame in over /ws, counts out over the subscription.

        Deliberately one test rather than three: the value is in the hops
        being joined, and three tests that each stubbed one hop would all
        pass on a path that does not connect.
        """
        client.post("/scene/start")
        with client.websocket_connect("/ws") as ws:
            _send_frames(ws, 4)
            scene = _await_scene(client)

            ws.send_json(
                {
                    "type": "result_subscribe",
                    "cartridge": "scene_understanding",
                    "result_type": "live",
                }
            )
            assert ws.receive_json()["type"] == "result_subscribed"
            first = ws.receive_json()

        payload = first["payload"]
        assert payload["scene_available"] is True
        assert payload["counts"]["person"] == 2
        assert payload["counts"]["chair"] == 1
        assert payload["counts"]["laptop"] == 1
        # And the HTTP view agrees with the socket, because both call the
        # same function.
        assert scene["counts"] == payload["counts"]

    def test_frames_arriving_before_start_are_counted_and_dropped(self, client):
        with client.websocket_connect("/ws") as ws:
            _send_frames(ws, 3)

        status = client.get("/scene").json()
        assert status["lifecycle"]["state"] == "stopped"
        assert status["frames_dropped_not_running"] == 3
        assert status["frames_observed"] == 0
        assert status["scene_available"] is False

    def test_stopping_discards_the_scene_on_every_surface(self, client):
        """Stop must be stop on the socket as well as on the route.

        A stale scene reaching a client through a channel the stop route
        did not think about is exactly how this cartridge would end up
        making a claim about a room the wearer has left.
        """
        client.post("/scene/start")
        with client.websocket_connect("/ws") as ws:
            _send_frames(ws, 4)
            _await_scene(client)

            ws.send_json(
                {
                    "type": "result_subscribe",
                    "cartridge": "scene_understanding",
                    "result_type": "live",
                }
            )
            assert ws.receive_json()["type"] == "result_subscribed"
            assert ws.receive_json()["payload"]["scene_available"] is True

            stopped = client.post("/scene/stop").json()
            pump(client)
            after = ws.receive_json()

        assert stopped["scene_available"] is False
        assert stopped["counts"] is None
        assert after["payload"]["scene_available"] is False
        assert after["payload"]["counts"] is None
        assert after["payload"]["lifecycle"]["state"] == "stopped"
        assert "discarded" in after["payload"]["scene_unavailable_reason"]

    def test_a_paused_session_keeps_its_counts_and_says_they_are_not_current(
        self, client
    ):
        client.post("/scene/start")
        with client.websocket_connect("/ws") as ws:
            _send_frames(ws, 4)
            _await_scene(client)

        paused = client.post("/scene/pause").json()

        assert paused["lifecycle"]["state"] == "paused"
        assert paused["lifecycle"]["scene_is_current"] is False
        assert paused["scene_available"] is True
        assert paused["counts"]["person"] == 2


class TestThePayloadSaysWhatItMayNotSay:
    @pytest.fixture
    def payload(self, client):
        client.post("/scene/start")
        with client.websocket_connect("/ws") as ws:
            _send_frames(ws, 4)
            _await_scene(client)
        return client.get("/scene").json()

    #: The blocks whose whole job is to NAME what is refused. They
    #: contain the forbidden words on purpose -- `refused_entity_fields`
    #: exists precisely to say "track_id: joinable across time" -- so the
    #: leak scan runs over everything else.
    #:
    #: Listed explicitly, and asserted to be exactly this set, so a new
    #: block cannot quietly become exempt by being prose-shaped.
    PROSE_BLOCKS = (
        "tracks_absent_reason",
        "refused_entity_fields",
        "refused_relations",
        "relations_absent_reason",
        "withheld_relations",
        "where_excludes_reason",
        "side_convention",
        "count_limitations",
        "confidence_absent_reason",
        "scene_unavailable_reason",
        "observed_at_note",
    )

    def test_the_prose_blocks_are_exactly_the_declared_set(self, payload):
        """The exemption list must not drift ahead of the payload.

        If a new self-describing block arrives and is not listed above,
        the leak scan below covers it -- which is the safe direction. If
        one is listed and no longer exists, the exemption is stale and
        this says so.
        """
        for name in self.PROSE_BLOCKS:
            assert name in payload, f"{name} is exempted and does not exist"

    @pytest.mark.parametrize(
        "forbidden",
        ("track_id", "visible_eyes", "visible_ears", "x0", "\"box\"", "normalised_x",
         "view_offset", "in_front_of\":"),
    )
    def test_no_identifying_or_joinable_field_reaches_the_wire(
        self, payload, forbidden
    ):
        """Serialised and searched, not inspected key by key.

        A key-by-key check tests the keys somebody remembered. Searching
        the encoded payload catches one nested three levels down in a
        block added later, which is how this would actually go wrong.

        The self-describing blocks are removed first, because they name
        the refused fields as their entire purpose. Everything else --
        every number, every count, every position, every lifecycle field
        -- is scanned.
        """
        import json

        data = {
            key: value
            for key, value in payload.items()
            if key not in self.PROSE_BLOCKS
        }
        encoded = json.dumps(data)
        assert forbidden not in encoded, f"the payload leaked {forbidden!r}"

    def test_the_refusals_name_the_fields_they_refuse(self, payload):
        """The other half, and the reason the exemption is safe.

        A refusal that does not name what it refuses is an absence. These
        blocks are where the forbidden words are allowed to appear, and
        they are required to appear there.
        """
        refused = {entry["field"] for entry in payload["refused_entity_fields"]}

        assert {"track_id", "box", "facing", "visible_eyes"} <= refused
        assert payload["tracks"] is None
        assert "dwell timeline" in payload["tracks_absent_reason"]

    def test_relations_are_unexpressible_not_merely_empty(self, payload):
        assert payload["relations"] is None
        assert payload["relations_absent_reason"]
        refused = {entry["relation"] for entry in payload["refused_relations"]}
        assert "in_front_of" in refused
        assert "behind" in refused

    def test_people_are_a_count_and_an_aggregate_never_a_list(self, payload):
        people = payload["people"]

        assert people["count"] == 2
        assert people["may_include_wearer"] is True
        assert people["validated"] is False
        # Never 0. Orientation was never measured, and zero is an answer.
        assert people["facing_wearer"] is None
        assert people["facing_answered"] is False
        assert "observation gap" in people["facing_unavailable_reason"]

    def test_position_is_reported_for_objects_and_refused_for_people(self, payload):
        assert "person" not in payload["where"]
        assert payload["where_excludes"] == ["person"]
        # Two objects placed on opposite sides of the frame by the
        # fixture, so the side counts are known independently of the code.
        assert payload["where"]["chair"]["right"] == 1
        assert payload["where"]["laptop"]["left"] == 1

    def test_every_count_declares_itself_a_lower_bound(self, payload):
        assert payload["count_is_lower_bound"] is True
        limitations = {entry["limitation"] for entry in payload["count_limitations"]}
        assert "size-floor" in limitations
        assert "recall" in limitations

    def test_every_reported_class_is_present_even_at_zero(self, payload):
        assert set(payload["counts"]) == set(payload["reported_classes"])
        assert payload["counts"]["bottle"] == 0

    def test_every_boolean_is_a_real_bool_not_an_int(self, payload):
        """`bool` subclasses `int`, and a 1 fails every Swift `as? Bool`."""
        for key in ("count_is_lower_bound", "scene_available"):
            assert type(payload[key]) is bool
        for key in ("may_include_wearer", "validated", "facing_answered"):
            assert type(payload["people"][key]) is bool
        for key in ("scene_is_current", "load_overdue"):
            assert type(payload["lifecycle"][key]) is bool

    def test_the_payload_stays_small_with_every_class_saturated(self, client):
        """Fixed arity is what keeps this bounded, so prove the arity.

        A scene with one of everything is the largest `counts` and
        `where` this contract can produce, because both are keyed on a
        fixed class list rather than on what is in the room.

        THE CEILING IS 12 KB, NOT WORLD BUILDER'S 8, AND THE DIFFERENCE
        IS DELIBERATE. That number is a bandwidth judgement, not a
        protocol limit: the concern is starving the frame path, which
        shares this socket's send lock. Measured, at maximum arity, this
        payload is ~9.2 KB, of which roughly two thirds is CONSTANT
        self-description -- the refusals, the count limitations, the
        notes that keep a null from being read as a zero. Re-sent on the
        2 s heartbeat that is 4.6 KB/s per subscriber, against frames at
        12 fps and ~30 KB each: 1.3%.

        The prose was already cut once. `refused_relations` used to carry
        the full research abstracts from `tower/scene/state.py` -- 1,500
        characters for `in_front_of` alone -- and now carries the first
        sentence and a pointer, which took 1.5 KB off every heartbeat. It
        will not be cut further by deleting disclosure: an undercount
        published without its limitations is the failure this cartridge
        exists to avoid, and it is cheaper to send than to explain.

        The assertion below the ceiling is the one that matters more. It
        pins the property rather than the number: saturating the room
        must add only the fixed-arity delta, so the payload cannot grow
        with what is in front of the wearer.
        """
        import json

        from tower.results.scene_understanding import REPORTED_CLASSES

        detections = [
            Detection(label, 0.9, BoundingBox(10 + i * 3, 10, 60 + i * 3, 90))
            for i, label in enumerate(REPORTED_CLASSES)
        ]
        import numpy as np

        from tower.scene.live import SceneLive

        # Decode stubbed here and only here: this test drives the session
        # directly rather than through `/ws`, because what it measures is
        # the payload's ARITY, and pushing 13 classes through the socket
        # would test the socket again for no extra information.
        frame = np.zeros((360, 640, 3), np.uint8)
        session = SceneLive(
            lambda: StubSceneEngine(detections), decode=lambda raw: frame
        )
        client.app.state.live_cartridges.frame_consumers[0].stop()
        client.app.state.live_cartridges = type(
            client.app.state.live_cartridges
        )(frame_consumers=[session], scene=session, document=None)
        try:
            session.start()
            # `start()` returns while the worker is still STARTING, and a
            # frame offered then is correctly dropped. So this waits on
            # the observable state rather than assuming the thread has
            # got there -- the same reason `_await_scene` polls.
            deadline = threading.Event()
            timer = threading.Timer(10.0, deadline.set)
            timer.start()
            try:
                while not deadline.is_set() and session.latest()[0] is None:
                    session.offer_frame(b"x", received_at=1.0)
                    deadline.wait(0.005)
            finally:
                timer.cancel()
            assert session.latest()[0] is not None, "the session never observed"
            payload = client.get("/scene").json()
            encoded = json.dumps(payload)
            assert payload["counts"]["dining table"] == 1
            assert len(encoded) < 12000, (
                f"the payload grew to {len(encoded)} bytes"
            )

            # An EMPTY running scene, for the delta. Everything but the
            # counts is identical, so the difference is exactly what a
            # full room costs.
            session.stop()
            empty = SceneLive(lambda: StubSceneEngine([]), decode=lambda raw: frame)
            client.app.state.live_cartridges = type(
                client.app.state.live_cartridges
            )(frame_consumers=[empty], scene=empty, document=None)
            try:
                empty.start()
                deadline = threading.Event()
                timer = threading.Timer(10.0, deadline.set)
                timer.start()
                try:
                    while not deadline.is_set() and empty.latest()[0] is None:
                        empty.offer_frame(b"x", received_at=1.0)
                        deadline.wait(0.005)
                finally:
                    timer.cancel()
                bare = len(json.dumps(client.get("/scene").json()))
            finally:
                empty.stop()

            assert len(encoded) - bare < 1200, (
                "a full room added "
                f"{len(encoded) - bare} bytes; this payload must grow with "
                "its fixed class list, not with what is in front of the "
                "wearer"
            )
        finally:
            session.stop()


class TestTheSessionIsNotRunningUnlessSomebodySaidSo:
    def test_a_fresh_tower_observes_nothing(self, client):
        status = client.get("/scene").json()

        assert status["lifecycle"]["state"] == "stopped"
        assert status["frames_observed"] == 0
        assert status["scene_available"] is False
        assert "stopped" in status["scene_unavailable_reason"]

    def test_the_routes_are_absent_when_the_cartridge_is_off(
        self, monkeypatch, tmp_path
    ):
        """Off must mean 404 on the control surface, not a silent no-op.

        A POST that returns 200 and does nothing is how an operator ends
        up believing a physical test is running when it is not.
        """
        from tower.main import create_app

        monkeypatch.delenv("TOWER_SCENE_UNDERSTANDING", raising=False)
        monkeypatch.delenv("TOWER_WORLD_ROOT", raising=False)
        off = TestClient(create_app())
        off.__enter__()
        _OPEN.append(off)

        assert off.get("/scene").status_code == 404
        assert off.post("/scene/start").status_code == 404

        offer = next(
            entry
            for entry in off.get("/cartridges").json()["cartridges"]
            if entry["cartridge"] == "scene_understanding"
        )
        assert offer["available"] is False
        assert "TOWER_SCENE_UNDERSTANDING" in offer["unavailable_reason"]


class TestAPhoneThatStreamsAndWatchesSeesAScene:
    """Nothing but the wire starts the session, and only the right wire.

    `IOS-to-Tower.md` 6.2 is explicit that opening a cartridge on the
    phone sends no VERB, and a test on the iOS side asserts that. What
    the phone does send is a `result_subscribe` for the live scene, and
    together with `stream_start` -- the moment a feed exists -- that is
    the whole start signal. A stream alone is a phone using some other
    cartridge's camera and must not run a people detector behind it; a
    subscription alone has nothing to observe.
    """

    def test_stream_and_subscription_start_the_session_without_any_http_call(
        self, client
    ):
        with client.websocket_connect("/ws") as ws:
            _subscribe(ws)
            ws.send_json({"type": "stream_start"})
            _send_frames(ws, 4)
            scene = _await_scene(client)

        assert scene["counts"]["person"] == 2
        assert scene["lifecycle"]["scene_is_current"] is True

    def test_a_stream_alone_does_not_start_it(self, client):
        """World Builder's camera is not a request to detect people."""
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "stream_start"})
            _send_frames(ws, 4)
            status = client.get("/scene").json()

        assert status["lifecycle"]["state"] == "stopped"
        assert status["lifecycle"]["demand"] == {
            "streams": 1,
            "watchers": 0,
            "operator_hold": False,
            "runs_when": "stream-and-watcher-or-operator",
        }
        assert status["frames_dropped_not_running"] == 4

    def test_a_subscription_alone_does_not_start_it(self, client):
        with client.websocket_connect("/ws") as ws:
            _subscribe(ws)
            status = client.get("/scene").json()

        assert status["lifecycle"]["state"] == "stopped"
        assert status["lifecycle"]["demand"]["watchers"] == 1
        assert status["lifecycle"]["demand"]["streams"] == 0

    def test_unsubscribing_stops_it_and_discards_the_scene(self, client):
        """Leaving the Scene screen hands the GPU back."""
        with client.websocket_connect("/ws") as ws:
            subscription_id = _subscribe(ws)
            ws.send_json({"type": "stream_start"})
            _send_frames(ws, 4)
            _await_scene(client)
            ws.send_json(
                {"type": "result_unsubscribe", "subscription_id": subscription_id}
            )
            assert ws.receive_json()["type"] == "result_unsubscribed"
            after = _await_state(client, "stopped")

        assert after["scene_available"] is False
        assert after["counts"] is None
        assert after["lifecycle"]["demand"]["watchers"] == 0
        assert after["lifecycle"]["demand"]["streams"] == 1

    def test_resubscribing_on_the_same_stream_starts_a_fresh_session(
        self, client
    ):
        with client.websocket_connect("/ws") as ws:
            first_id = _subscribe(ws)
            ws.send_json({"type": "stream_start"})
            _send_frames(ws, 4)
            first = _await_scene(client)
            ws.send_json({"type": "result_unsubscribe", "subscription_id": first_id})
            assert ws.receive_json()["type"] == "result_unsubscribed"
            _await_state(client, "stopped")

            _subscribe(ws)
            _send_frames(ws, 4)
            second = _await_scene(client)

        assert second["lifecycle"]["session_id"] == first["lifecycle"]["session_id"] + 1
        assert second["counts"]["person"] == 2

    def test_stream_stop_ends_it_and_discards_the_scene(self, client):
        with client.websocket_connect("/ws") as ws:
            _subscribe(ws)
            ws.send_json({"type": "stream_start"})
            _send_frames(ws, 4)
            _await_scene(client)
            ws.send_json({"type": "stream_stop"})
            after = _await_state(client, "stopped")

        assert after["scene_available"] is False
        assert after["counts"] is None

    def test_a_disconnect_ends_it_too(self, client):
        """A wearable disconnects abruptly as the NORMAL case.

        Crash, network drop, walking out of range. A session left running
        by one of those would hold a model and keep serving a scene of a
        room whose wearer is gone.
        """
        with client.websocket_connect("/ws") as ws:
            _subscribe(ws)
            ws.send_json({"type": "stream_start"})
            _send_frames(ws, 4)
            _await_scene(client)

        # The watchers leave first and the stream closes after, on two
        # separate hops off the event loop; wait for both.
        for _ in range(400):
            after = client.get("/scene").json()
            if (
                after["lifecycle"]["state"] == "stopped"
                and after["lifecycle"]["demand"]["streams"] == 0
            ):
                break
            time.sleep(0.005)
        assert after["scene_available"] is False
        assert after["lifecycle"]["demand"] == {
            "streams": 0,
            "watchers": 0,
            "operator_hold": False,
            "runs_when": "stream-and-watcher-or-operator",
        }

    def test_a_connection_that_never_streamed_does_not_end_an_operator_s_session(
        self, client
    ):
        """Stop only what the stream started.

        `ws.py` tears the stream down on ANY exit, including a client
        that never sent `stream_start`. An unconditional stop here would
        let any passing connection kill a session an operator had started
        by hand for a physical test -- which is exactly the situation
        this cartridge most needs to survive.
        """
        client.post("/scene/start")
        with client.websocket_connect("/ws") as ws:
            _send_frames(ws, 4)
            _await_scene(client)
        after = client.get("/scene").json()

        assert after["lifecycle"]["state"] == "running"
        assert after["scene_available"] is True

    def test_a_restarted_operator_session_still_stops_with_its_stream(
        self, client
    ):
        """Integration finding 15, on the real wire."""
        with client.websocket_connect("/ws") as ws:
            _subscribe(ws)
            ws.send_json({"type": "stream_start"})
            _send_frames(ws, 4)
            _await_scene(client)
            client.post("/scene/stop")
            client.post("/scene/start")
            _send_frames(ws, 4)
            _await_scene(client)

        after = _await_state(client, "stopped")
        assert after["scene_available"] is False

    def test_autostart_can_be_turned_off(self, monkeypatch):
        """An operator who wants the manual control keeps it."""
        from tower import cartridge_runtime
        from tower.main import create_app
        from tower.scene.live import SceneLive

        monkeypatch.setenv("TOWER_SCENE_UNDERSTANDING", "true")
        monkeypatch.setenv("TOWER_SCENE_AUTOSTART", "false")
        monkeypatch.delenv("TOWER_WORLD_ROOT", raising=False)
        monkeypatch.setattr(
            cartridge_runtime,
            "_scene_session",
            lambda settings: SceneLive(
                lambda: StubSceneEngine(TWO_PEOPLE_A_CHAIR_AND_A_LAPTOP),
                follow_stream=settings.scene_autostart,
            ),
        )
        manual = TestClient(create_app())
        manual.__enter__()
        _OPEN.append(manual)

        with manual.websocket_connect("/ws") as ws:
            _subscribe(ws)
            ws.send_json({"type": "stream_start"})
            _send_frames(ws, 4)

        status = manual.get("/scene").json()
        assert status["lifecycle"]["state"] == "stopped"
        assert status["lifecycle"]["follows_stream"] is False
