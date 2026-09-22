"""Start, Pause, Stop -- and what each of them must actually do.

The three failure modes a reviewer of a live cartridge should hunt for are
all lifecycle failures, and all three are asserted here:

- Start/Stop that does not control the work. `frames_observed` must stop
  moving after `stop()` and must not move before `start()`.
- Stale state served after Stop. The single most damaging thing this
  cartridge could do is answer "what is around you" about a room the
  wearer left, so `stop()` DISCARDS the last state rather than freezing
  it.
- An unbounded queue. There is one slot; the hundredth frame offered to a
  busy worker must cost the same memory as the second, and the drops must
  be counted where a client can see them.

No torch here. The engine is a stub, because what is under test is the
session around the engine, not the detector -- and a test that had to
load a 13 MB model to prove that `pause()` stops feeding it would not be
run often enough to be worth writing.
"""

import threading

import pytest

from tower.scene.live import (
    STATE_FAILED,
    STATE_PAUSED,
    STATE_RUNNING,
    STATE_STOPPED,
    SceneLive,
)


class StubEngine:
    """Counts what it was asked to do, and can be made slow or hostile."""

    def __init__(self, *, gate: threading.Event | None = None, explode=False):
        self.loaded = False
        self.released = False
        self.observed = []
        self._gate = gate
        self._explode = explode
        self._detector = type("D", (), {"name": "stub"})()

    def load(self):
        if self._explode == "load":
            raise RuntimeError("no weights here")
        self.loaded = True

    def release(self):
        self.released = True

    def observe(self, frame, *, received_at=None):
        if self._gate is not None:
            self._gate.wait(timeout=5.0)
        if self._explode == "observe":
            raise RuntimeError("bad frame")
        self.observed.append(received_at)
        return ("state", received_at)


def _live(engine=None, **kwargs):
    engine = engine if engine is not None else StubEngine()
    live = SceneLive(
        lambda: engine,
        decode=lambda raw: raw,
        **kwargs,
    )
    return live, engine


def _await_state(live, wanted, timeout=5.0):
    """Poll the session until it reaches a state, or fail loudly.

    A `threading.Event` on the session would be a second source of truth
    about the same transition; polling the public accessor tests the
    thing a caller would actually observe.
    """
    deadline = threading.Event()
    timer = threading.Timer(timeout, deadline.set)
    timer.start()
    try:
        while not deadline.is_set():
            if live.state == wanted:
                return
        raise AssertionError(f"session never reached {wanted!r}; it is {live.state!r}")
    finally:
        timer.cancel()


def _await(predicate, timeout=5.0):
    deadline = threading.Event()
    timer = threading.Timer(timeout, deadline.set)
    timer.start()
    try:
        while not deadline.is_set():
            if predicate():
                return
        raise AssertionError("condition never became true")
    finally:
        timer.cancel()


@pytest.fixture
def stopped_session():
    sessions = []

    def make(engine=None, **kwargs):
        live, engine = _live(engine, **kwargs)
        sessions.append(live)
        return live, engine

    yield make
    for live in sessions:
        live.stop()


class TestStartAndStopControlTheWork:
    def test_a_session_that_was_never_started_observes_nothing(self, stopped_session):
        live, engine = stopped_session()

        for _ in range(10):
            live.offer_frame(b"frame")

        assert engine.observed == []
        assert live.status()["frames_dropped_not_running"] == 10
        assert live.status()["state"] == STATE_STOPPED

    def test_a_started_session_observes_what_it_is_offered(self, stopped_session):
        live, engine = stopped_session()
        live.start()
        _await_state(live, STATE_RUNNING)

        live.offer_frame(b"frame", received_at=1.0)
        _await(lambda: live.status()["frames_observed"] == 1)

        assert engine.observed == [1.0]

    def test_a_stopped_session_observes_nothing_further(self, stopped_session):
        live, engine = stopped_session()
        live.start()
        _await_state(live, STATE_RUNNING)
        live.offer_frame(b"frame", received_at=1.0)
        _await(lambda: live.status()["frames_observed"] == 1)

        live.stop()
        for _ in range(5):
            live.offer_frame(b"frame", received_at=2.0)

        assert engine.observed == [1.0]

    def test_stopping_releases_the_engine(self, stopped_session):
        live, engine = stopped_session()
        live.start()
        _await_state(live, STATE_RUNNING)

        live.stop()

        assert engine.released is True

    def test_starting_twice_does_not_start_two_workers(self, stopped_session):
        """Idempotent in the direction the callers mean it.

        Both plausible callers -- an operator pressing a button twice, a
        client reconnecting -- mean "make sure this is on".
        """
        live, engine = stopped_session()
        first = live.start()
        _await_state(live, STATE_RUNNING)
        before = threading.active_count()
        second = live.start()

        assert second["session_id"] == first["session_id"]
        # The DELTA across the second start, not the process-wide total.
        #
        # This assertion used to read `threading.active_count() < 50`,
        # which is a whole-process measurement standing in for a claim
        # about one session. Run on its own it passed; run inside the full
        # suite on a host with torch installed it reached 54 and failed,
        # because the number counts torch's native pools and every thread
        # any other cartridge's tests left in the interpreter -- none of
        # which this session started and none of which it can control. A
        # test that fails for what its neighbours did says nothing about
        # the code it names.
        #
        # What "does not start two workers" actually means is that the
        # second start adds no thread of its own, and that is what is
        # measured now. It is a strictly tighter bound: the old form would
        # have passed a genuine second worker as long as the process
        # happened to be under fifty threads.
        assert threading.active_count() == before, (
            "a second start must add no thread: it is the same session"
        )


class TestStopDiscardsTheScene:
    def test_the_last_state_does_not_survive_a_stop(self, stopped_session):
        """The single most important assertion about this cartridge.

        A scene retained past the end of a session is a claim about a room
        the wearer has left. There is no staleness number large enough to
        make that safe, because a client that renders counts above
        staleness shows the room first.
        """
        live, _ = stopped_session()
        live.start()
        _await_state(live, STATE_RUNNING)
        live.offer_frame(b"frame", received_at=1.0)
        _await(lambda: live.latest()[0] is not None)

        live.stop()

        state, observed_at, computed_at = live.latest()
        assert state is None
        assert observed_at is None
        assert computed_at is None
        assert live.status()["has_state"] is False
        assert live.status()["staleness_seconds"] is None

    def test_a_paused_session_keeps_its_scene_and_says_it_is_paused(
        self, stopped_session
    ):
        """Pause is the deliberately different case.

        `IOS-to-Tower.md` 4.7 asks the Tower to keep `observing` and
        `lastKnown` apart rather than flatten them. Pause is lastKnown.
        """
        live, engine = stopped_session()
        live.start()
        _await_state(live, STATE_RUNNING)
        live.offer_frame(b"frame", received_at=1.0)
        _await(lambda: live.latest()[0] is not None)

        live.pause()

        assert live.state == STATE_PAUSED
        assert live.latest()[0] is not None
        assert live.status()["has_state"] is True

    def test_a_paused_session_observes_nothing(self, stopped_session):
        live, engine = stopped_session()
        live.start()
        _await_state(live, STATE_RUNNING)
        live.pause()

        for _ in range(5):
            live.offer_frame(b"frame", received_at=1.0)

        assert engine.observed == []
        assert live.status()["frames_dropped_not_running"] == 5

    def test_resuming_observes_again_without_reloading(self, stopped_session):
        live, engine = stopped_session()
        live.start()
        _await_state(live, STATE_RUNNING)
        live.pause()
        live.resume()
        _await_state(live, STATE_RUNNING)

        live.offer_frame(b"frame", received_at=2.0)
        _await(lambda: live.status()["frames_observed"] == 1)

        assert engine.released is False
        assert engine.observed == [2.0]

    def test_a_fresh_start_after_a_stop_begins_a_new_session(self, stopped_session):
        """Session identity must change, because track ids restart.

        A client that kept counting across a stop would be joining two
        different tracker sessions, which is exactly the joinability this
        cartridge refuses.
        """
        live, _ = stopped_session()
        first = live.start()["session_id"]
        _await_state(live, STATE_RUNNING)
        live.stop()
        second = live.start()["session_id"]

        assert second != first


class TestTheSlotIsOneDeep:
    def test_a_hundred_frames_offered_to_a_busy_worker_hold_one(
        self, stopped_session
    ):
        gate = threading.Event()
        engine = StubEngine(gate=gate)
        live, _ = stopped_session(engine)
        live.start()
        _await_state(live, STATE_RUNNING)

        # The first frame enters the worker and blocks on the gate.
        live.offer_frame(b"frame", received_at=0.0)
        _await(lambda: live._pending is None)

        for index in range(100):
            live.offer_frame(b"frame", received_at=float(index + 1))

        assert live._pending is not None
        assert live._pending[1] == 100.0, "the newest frame wins the slot"
        assert live.status()["frames_skipped"] == 99
        gate.set()

    def test_a_skipped_frame_is_counted_where_a_client_can_see_it(
        self, stopped_session
    ):
        """A silently dropped frame is indistinguishable from a quiet room.

        That is the whole reason this counter is on the wire rather than
        only in a log.
        """
        gate = threading.Event()
        engine = StubEngine(gate=gate)
        live, _ = stopped_session(engine)
        live.start()
        _await_state(live, STATE_RUNNING)
        live.offer_frame(b"a", received_at=0.0)
        _await(lambda: live._pending is None)

        live.offer_frame(b"b", received_at=1.0)
        live.offer_frame(b"c", received_at=2.0)

        assert live.status()["frames_skipped"] == 1
        gate.set()


class TestFailuresAreReportedRatherThanSwallowed:
    def test_a_load_that_fails_leaves_a_failed_session_with_a_reason(
        self, stopped_session
    ):
        live, _ = stopped_session(StubEngine(explode="load"))
        live.start()
        _await_state(live, STATE_FAILED)

        status = live.status()
        assert status["state"] == STATE_FAILED
        assert "could not be loaded" in status["failure_reason"]
        assert status["has_state"] is False

    def test_a_frame_that_explodes_does_not_end_the_session(self, stopped_session):
        live, engine = stopped_session(StubEngine(explode="observe"))
        live.start()
        _await_state(live, STATE_RUNNING)

        live.offer_frame(b"frame", received_at=1.0)
        live.offer_frame(b"frame", received_at=2.0)

        assert live.state == STATE_RUNNING

    def test_a_frame_that_will_not_decode_is_counted_not_observed(
        self, stopped_session
    ):
        engine = StubEngine()
        live = SceneLive(lambda: engine, decode=lambda raw: None)
        try:
            live.start()
            _await_state(live, STATE_RUNNING)
            live.offer_frame(b"rubbish", received_at=1.0)
            _await(lambda: live.status()["decode_failures"] == 1)

            assert engine.observed == []
            assert live.status()["frames_observed"] == 0
        finally:
            live.stop()

    def test_offer_frame_never_raises_whatever_the_session_is_doing(
        self, stopped_session
    ):
        """It runs on the event loop. It is not allowed to have opinions."""
        live, _ = stopped_session(StubEngine(explode="load"))
        live.offer_frame(b"a")
        live.start()
        live.offer_frame(b"b")
        _await_state(live, STATE_FAILED)
        live.offer_frame(b"c")
        live.stop()
        live.offer_frame(b"d")


class TestTheReportedClockIsHonest:
    def test_staleness_is_measured_from_the_frame_not_from_the_computation(
        self, stopped_session
    ):
        """`observed_at` is when the Tower received the frame.

        Not when the detector finished with it. A 30 ms detection would
        otherwise make every scene look 30 ms fresher than it is, and the
        error grows with the cost of the model.
        """
        now = [100.0]
        engine = StubEngine()
        live = SceneLive(lambda: engine, decode=lambda raw: raw, clock=lambda: now[0])
        try:
            live.start()
            _await_state(live, STATE_RUNNING)
            live.offer_frame(b"frame", received_at=90.0)
            _await(lambda: live.latest()[0] is not None)

            now[0] = 105.0
            status = live.status()
            assert status["observed_at"] == 90.0
            assert status["staleness_seconds"] == pytest.approx(15.0)
        finally:
            live.stop()

    def test_a_load_that_overruns_is_reported_without_being_killed(
        self, stopped_session
    ):
        """Overdue is a report, not a kill.

        Nothing in Python can interrupt a blocking model load, and
        pretending otherwise would be the lie. A first-run weight download
        on a slow link is slow and still correct; what an operator needs
        is to be able to tell that from a wedge.
        """
        now = [0.0]
        gate = threading.Event()

        class SlowLoad(StubEngine):
            def load(self):
                gate.wait(timeout=5.0)
                super().load()

        live = SceneLive(
            lambda: SlowLoad(), decode=lambda raw: raw, clock=lambda: now[0]
        )
        try:
            live.start()
            now[0] = 500.0
            status = live.status()

            assert status["state"] == "starting"
            assert status["load_overdue"] is True
            assert status["loading_seconds"] == pytest.approx(500.0)
        finally:
            gate.set()
            live.stop()
