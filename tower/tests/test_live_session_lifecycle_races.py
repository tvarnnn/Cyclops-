"""Lifecycle races in the shared live-session base, each reproduced first.

These are integration findings 13 and 15, carried open since the
2026-08-27 unification and deferred by the runtime-fitness lane because
fixing them meant touching `stop()`, whose docstring calls its four-step
ordering "the whole correctness of this method" and records what both
previous inversions cost.

The shape they share is the one the unification's final reviewer named:
every falsifiable NUMBER in that lane held, and every claim that failed
was about **lifecycle ownership or session scope** -- written from design
intent rather than read off a running object. So these tests drive two
threads at a real session and read what actually happened.

`_on_pause` is the window in both. It is what `stop()` runs at step 2,
off the lock, and for Document Memory it is `engine.flush()` -- an OCR
pass measured at 1.19 s a page, up to two pages. So this window is
SECONDS wide in the product, not microseconds. Blocking it on an event
here is not an artificial widening; it is the ordinary case held still.
"""

import threading
import time

import pytest

from tower.live_session import STATE_RUNNING, STATE_STOPPED
from tower.scene.live import SceneLive


class _Engine:
    """Loads instantly, carries no torch, and records its own release.

    The LIFECYCLE is under test, not the model -- the same reason
    `tests/test_unified_lifecycle_regressions.py` uses a stub engine.
    Identity matters here in a way it does not there, so each instance
    is numbered and remembers whether it was released.
    """

    _made = 0

    def __init__(self) -> None:
        _Engine._made += 1
        self.ident = _Engine._made
        self.released = False

    def load(self):
        return None

    def observe(self, frame, received_at=None, source_seq=None):
        return None

    def release(self):
        self.released = True


class _BlockingFlush(SceneLive):
    """Scene, but its stop-time flush is held open on an event."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.flush_entered = threading.Event()
        self.release_flush = threading.Event()

    def _on_pause(self, engine):
        self.flush_entered.set()
        assert self.release_flush.wait(15), "the test never released the flush"


def _await_state(session, want, timeout=10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if session.state == want:
            return True
        time.sleep(0.005)
    return False


def _session(cls=SceneLive):
    return cls(_Engine, decode=lambda raw: raw, follow_stream=True)


class TestAStopStillFlushing:
    """Integration finding 13, reproduced.

    `stop()` holds `_lifecycle` for all four of its steps. `start()`
    takes only `_condition`. Step 1 sets STOPPED under `_condition` and
    RELEASES it; steps 2 (flush), 3 (join) and 4 (release) run outside
    it. A `start()` landing in that window sees STOPPED and stands up a
    second session -- and step 4 then reached for `self._engine`, which
    by then is the SECOND session's.
    """

    def test_a_stop_releases_the_engine_it_captured_not_the_current_one(self):
        session = _session(_BlockingFlush)

        session.start()
        assert _await_state(session, STATE_RUNNING)
        engine1 = session._engine

        stopper = threading.Thread(target=session.stop, name="http-stop")
        stopper.start()
        assert session.flush_entered.wait(10), "stop() never reached its flush"

        # A wearer pressing Start, or a phone reconnecting. The state is
        # already STOPPED, so nothing refuses this.
        session.start()
        assert _await_state(session, STATE_RUNNING)
        engine2 = session._engine
        assert engine2 is not engine1, "session 2 must build its own engine"

        session.release_flush.set()
        stopper.join(15)
        assert not stopper.is_alive()

        assert engine1.released, (
            "the stopping session's own engine was never released: it "
            "leaked, and the model it holds is never returned"
        )
        assert not engine2.released, (
            "the stop released the NEXT session's engine. That session is "
            "still running and now has no model behind it"
        )
        assert session._engine is engine2, (
            "the running session lost its engine reference; every frame "
            "it is offered now fails inside _consume and is swallowed by "
            "'a frame failed; the session continues', while status() "
            "still reports running and carries no field that says so"
        )
        assert session.state == STATE_RUNNING


class TestTheWorkerIsReused:
    """The per-session thread, and the OpenMP team it stranded.

    torch's ATen parallel backend is OpenMP here. The first ATen call on
    any thread spins up a team of `cores - 1`, and **that team is not
    reclaimed when the thread exits** -- so a `threading.Thread` per
    session cost +19 OS threads, ~+7 MB RSS and +38 handles on every
    Start/Stop, linear, with no plateau. Twelve cycles took a real Tower
    from 29 to 257 threads.
    """

    def test_repeated_sessions_do_not_grow_the_os_thread_count(self):
        """The leak itself, as a property rather than as a number.

        A REAL ATen CALL IS REQUIRED AND IS THE WHOLE POINT. A pure
        Python stub engine creates no OpenMP team, so this test would
        pass against the broken code and prove nothing. That is the
        trap this docstring exists to stop the next person falling into.

        Cycle 1 is excluded from the comparison deliberately: the first
        session legitimately creates one team and one worker, and that
        cost is CONSTANT. What must not grow is cycle 2 onwards.
        """
        torch = pytest.importorskip("torch")
        psutil = pytest.importorskip("psutil")

        class TorchEngine(_Engine):
            def load(self):
                with torch.inference_mode():
                    torch.nn.functional.conv2d(
                        torch.randn(1, 1, 16, 16), torch.randn(1, 1, 3, 3)
                    )

        session = SceneLive(TorchEngine, decode=lambda raw: raw)
        process = psutil.Process()

        def cycle():
            session.start()
            assert _await_state(session, STATE_RUNNING)
            session.stop()

        cycle()
        settled = process.num_threads()
        for _ in range(7):
            cycle()

        assert process.num_threads() == settled, (
            f"seven further Start/Stop cycles moved the OS thread count "
            f"from {settled} to {process.num_threads()}. Each session is "
            "stranding an OpenMP team that outlives its thread"
        )

    def test_two_sessions_run_on_the_same_worker(self):
        """Reuse, asserted directly, so a silent regression is visible.

        The count test above would also pass if a fresh thread happened
        to create no team. This one cannot: it reads the thread identity.
        """
        idents = []

        class Recording(_Engine):
            def load(self):
                idents.append(threading.get_ident())

        session = SceneLive(Recording, decode=lambda raw: raw)
        for _ in range(3):
            session.start()
            assert _await_state(session, STATE_RUNNING)
            session.stop()

        assert len(idents) == 3
        assert len(set(idents)) == 1, (
            f"three sessions ran on {len(set(idents))} different threads; "
            "the worker is not being reused"
        )

    def test_rapid_cycling_does_not_strand_parked_workers(self):
        """The window between `done.set()` and the worker going un-busy.

        Found by re-reading the fix rather than by a failing test, which
        is why it is written down rather than only fixed.

        `stop()` waits on the session's completion Event. The worker sets
        that Event in its `finally` and only THEN reacquires its own
        condition to mark itself free. So a `stop()` could return, and
        the next `start()` call `submit()`, while the worker it is about
        to reuse still read as busy. `submit()` refuses, a replacement is
        minted, and the original parks forever -- never reused, never
        exited. One stranded thread per occurrence, which is WORSE than
        the per-session thread this class replaced, because that one at
        least exited.

        Two changes close it: the worker clears `_busy` BEFORE signalling
        the session done, so anything that observes the Event also
        observes the worker free; and a worker that is replaced anyway is
        RETIRED rather than left parked, so the fallback costs at most
        one exiting thread instead of one immortal one.

        HONEST STATUS: THIS TEST PASSED BEFORE THE FIX TOO. 40 rapid
        cycles did not hit the window even once, and it is easy to see
        why -- between `done.set()` and the next `submit()` the stopping
        thread still has to run step 4's teardown and two lock
        acquisitions, which is ample time for the worker to mark itself
        free. So this is a window removed by reading, not a failure
        reproduced. It is closed anyway because the fix is a free
        reordering of two statements, and because "unlikely" stops being
        a defence the moment a loaded host preempts the worker in exactly
        the wrong microsecond -- which is the same argument the module
        already makes about the flush window it DOES reproduce.

        Kept as a stress guard rather than sold as a reproduction.
        """
        idents = []

        class Recording(_Engine):
            def load(self):
                idents.append(threading.get_ident())

        session = SceneLive(Recording, decode=lambda raw: raw)
        for _ in range(40):
            session.start()
            assert _await_state(session, STATE_RUNNING)
            session.stop()

        assert len(idents) == 40
        assert len(set(idents)) == 1, (
            f"40 rapid cycles ran on {len(set(idents))} distinct worker "
            "threads. Every extra one is a parked thread that will never "
            "be reused and never exit, holding its OpenMP team for the "
            "life of the process"
        )

    def test_a_wedged_load_is_abandoned_and_the_next_start_still_runs(self):
        """The one virtue of a thread per session, kept.

        A worker stuck inside a model load cannot be interrupted --
        nothing in Python can. It must never delay the next Start. The
        reused worker is RETIRED rather than waited on, and the next
        session mints a fresh one, which is exactly what happened on
        every cycle before the worker was reused.
        """
        wedged = threading.Event()
        entered = threading.Event()
        built = []

        class Wedging(_Engine):
            def load(self):
                if not built:
                    built.append(self)
                    entered.set()
                    assert wedged.wait(20), "the test never released the load"

        session = SceneLive(Wedging, decode=lambda raw: raw, stop_join_timeout_s=0.3)

        session.start()
        assert entered.wait(10), "the first load never started"
        began = time.monotonic()
        session.stop()
        assert time.monotonic() - began < 5.0, "stop() waited out a wedged load"

        session.start()
        assert _await_state(session, STATE_RUNNING, timeout=10.0), (
            "a session wedged in a model load blocked the next Start; the "
            "worker was waited on instead of retired"
        )
        wedged.set()

    def test_a_stop_does_not_wait_for_the_next_sessions_work(self):
        """Step 3 must wait on THIS session, never on the worker.

        The single way worker reuse goes wrong. If step 3 asks "is the
        worker idle?" instead of "is my session done?", a Stop racing a
        Start waits out the whole `STOP_JOIN_TIMEOUT_S` on session 2's
        load -- measured at 5.01 s, on a path reached from a websocket
        disconnect -- and then abandons the healthy worker session 2 is
        running on.
        """
        session = _session(_BlockingFlush)
        session.start()
        assert _await_state(session, STATE_RUNNING)

        stopper = threading.Thread(target=session.stop, name="http-stop")
        stopper.start()
        assert session.flush_entered.wait(10)

        session.start()
        assert _await_state(session, STATE_RUNNING)

        began = time.monotonic()
        session.release_flush.set()
        stopper.join(15)
        assert not stopper.is_alive()
        assert time.monotonic() - began < 3.0, (
            "stop() waited on the worker rather than on its own session, "
            "so it paid its full bound for session 2's work"
        )

    def test_a_reused_worker_produces_the_same_results_as_a_fresh_one(self):
        """What pays for the new risk.

        Session 2's `_create` now runs on a thread that previously ran
        session 1's `_consume`. torch's per-thread state is grad mode and
        RNG; this asserts the observable consequence rather than the
        mechanism.
        """
        seen = []

        class Counting(_Engine):
            def observe(self, frame, received_at=None, source_seq=None):
                seen.append((self.ident, frame))
                # NOT None: `SceneLive._publish` treats None as "the
                # frame would not decode" and undoes the base's
                # `frames_observed` increment, so a None-returning stub
                # makes this test unable to see its own frame.
                return object()

        session = SceneLive(Counting, decode=lambda raw: raw)
        for _ in range(3):
            session.start()
            assert _await_state(session, STATE_RUNNING)
            session.offer_frame(b"frame", source_seq=1)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if session.status()["frames_observed"] == 1:
                    break
                time.sleep(0.005)
            assert session.status()["frames_observed"] == 1
            session.stop()

        assert len(seen) == 3
        assert len({ident for ident, _ in seen}) == 3, (
            "each session must observe through its OWN engine even though "
            "they share a worker thread"
        )


class TestStopThenStart:
    """Integration finding 15, reproduced here as an xfail and now closed.

    `stop()` clears `_stream_owners` ("a manual stop disowns the
    stream") and `_begin_session_locked` clears it again, so after
    Stop -> Start on the HTTP routes a session built on the base class's
    owner set was owned by nobody, and its own stream closing could not
    end it. For a cartridge whose subject is detecting PEOPLE that was a
    privacy defect and not only a resource one.

    `SceneLive` now tracks OPEN streams apart from who started the
    session -- the split this class's original docstring said the fix
    needed -- and the last stream closing stops the session whoever
    pressed Start. `tests/test_scene_activation.py` holds the full rule;
    this test stays here, unmarked, because it is the reproduction that
    found the defect and should keep running against the fix.
    """

    def test_a_restarted_session_is_still_stopped_by_its_stream_closing(self):
        token = "connection-token-A"
        session = _session()

        session.stream_opened(owner=token)
        session.watcher_joined("sub", owner=token)
        assert _await_state(session, STATE_RUNNING), (
            "a phone that streams and watches starts it"
        )

        session.stop()
        session.start()
        assert _await_state(session, STATE_RUNNING)

        # The SAME phone -- never disconnected -- now drops.
        session.stream_closed(owner=token)

        assert _await_state(session, STATE_STOPPED, timeout=5.0), (
            "the stream that started this session closed and the session "
            "is still running: it is owned by nobody, keeps consuming "
            "frames and holding its model, and no disconnect can stop it. "
            "Only an explicit HTTP Stop can -- which is exactly the state "
            "a wearer cannot reach by putting the glasses down"
        )
