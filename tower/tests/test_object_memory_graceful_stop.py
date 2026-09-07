"""Pressing Stop must let the producer finish writing what it saw.

WHAT WAS BROKEN, AND IT WAS THE BUTTON'S WHOLE PURPOSE.

`POST /cartridges/object_memory/session/stop` reaches
`CartridgeSession._detach` -> `CaptureWorkerSupervisor.detach` ->
`Popen.terminate()`. On Windows `terminate()` is `TerminateProcess`:
no unwinding, no `finally`, no `atexit`. The producer's
`finally: engine.release()` -- the only code that closes the sightings
still open, writes the ones that had matured, and refreshes the
duration, `frame_count` and `best_*` of the ones already on disk --
therefore never ran.

The bias is the bad part. The sighting still OPEN when a wearer stops
walking is, by construction, the object they had been looking at
longest. Stop was throwing away the best-observed memories and keeping
the rest.

The fix has three pieces and this file exercises all three:

1. `scripts/object_memory_session.py` installs `_StopRequest`, which
   turns SIGTERM / SIGINT / SIGBREAK into a flag the frame loop reads.
2. `tower/capture_workers._ask_to_stop` asks on two channels BEFORE
   terminating: it closes the child's stdin, and it signals. The stdin
   close is the one that works everywhere -- a console control event
   needs a console, and under a pseudoconsole
   `GenerateConsoleCtrlEvent` reports success and delivers nothing.
   That was measured here, with a trivial child that registered all
   three handlers and was still alive thirty seconds later.
3. `CartridgeSession.DETACH_GRACE_SECONDS` is a real window again,
   because there is finally something for it to wait for.

The subprocess case is the load-bearing one. Piece 1 and piece 2 can
both be asserted with fakes and both be wrong about the platform; only a
real child process proves a request actually crossed one.
"""

import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import pytest

from tower.capture_workers import (
    CaptureWorkerSupervisor,
    WorkerSpec,
    _ask_to_stop,
)


# How long a child gets to notice the request and run its flush. Generous:
# the point of the assertion is "it exited at all", and a machine under a
# full test suite is not a machine to measure latency on.
STOP_TIMEOUT_SECONDS = 45.0


def _new_group_flags():
    return subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0


@pytest.fixture
def open_capture(tmp_path):
    """A capture that is still recording: a journal and no `ended_at`.

    A producer pointed at this polls forever, which is the state a wearer
    is in for the whole of a walk and the only state in which Stop is
    interesting. `capture.json` is deliberately absent -- `CaptureFollower`
    reads a missing manifest as "not closed", which is what a capture
    looks like between `stream_start` and the first flush.
    """
    directory = tmp_path / "captures" / "cap-1"
    (directory / "frames").mkdir(parents=True)
    (directory / "frames.jsonl").write_text("", encoding="utf-8")
    return directory


@pytest.fixture
def recorded_capture(tmp_path):
    """A finished capture with real frames in it.

    Unlike `open_capture`, this one HAS content -- which is the only way
    to tell "the stop channel waited" from "the stop channel fired and
    there was nothing to read anyway".
    """
    directory = tmp_path / "captures" / "cap-2"
    (directory / "frames").mkdir(parents=True)
    image = np.full((64, 48, 3), 120, np.uint8)
    lines = []
    count = 3
    for seq in range(1, count + 1):
        relpath = f"frames/{seq:08d}.jpg"
        (directory / relpath).write_bytes(cv2.imencode(".jpg", image)[1].tobytes())
        lines.append(
            json.dumps(
                {
                    "schema_version": 1,
                    "source_seq": seq,
                    "received_at": 1000.0 + seq,
                    "relpath": relpath,
                }
            )
        )
    (directory / "frames.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return directory, count


class TestTheProducerCanBeAskedToStop:
    """A real child process, a real request, a real exit code.

    The load-bearing case. Both halves of the mechanism can be asserted
    with fakes and both can be wrong about the platform; only a real
    child proves a request actually crossed a process boundary.
    """

    def _spawn(self, capture, root, extra=()):
        return subprocess.Popen(
            [
                sys.executable,
                "scripts/object_memory_session.py",
                "--follow-capture",
                str(capture),
                "--root",
                str(root),
                "--detector",
                "none",
                "--verifier",
                "none",
                # Long enough that an idle exit cannot be mistaken for a
                # stop that worked. If the request path is broken this
                # test times out rather than passing for the wrong reason.
                "--max-idle-polls",
                "40000",
                "--format",
                "json",
                "--stop-on-stdin-close",
                *extra,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=_new_group_flags(),
        )

    def _run_and_ask(self, capture, root):
        process = self._spawn(capture, root)
        try:
            # Long enough to be well inside the poll loop -- `--help`
            # returns in a quarter of a second on this host, so the
            # imports are not the wait.
            time.sleep(3.0)
            assert process.poll() is None, (
                "the producer exited before it could be asked to stop"
            )
            process.stdin.close()
            # Detached before `communicate()` sees it. Closing stdin IS the
            # stop request under test, and CPython's `Popen._communicate`
            # then flushes the closed writer -- which raises `ValueError:
            # I/O operation on closed file`, a error that code tolerates
            # only as `BrokenPipeError`. `communicate()` also sets
            # `_communication_started` in its own `finally`, so the retry
            # below would fail again on a missing `_fileobj2output` and
            # bury the first error.
            process.stdin = None
            stdout, stderr = process.communicate(timeout=STOP_TIMEOUT_SECONDS)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()
        return process, stdout, stderr

    def test_it_exits_cleanly_when_asked_rather_than_polling_forever(
        self, open_capture, tmp_path
    ):
        process, stdout, stderr = self._run_and_ask(
            open_capture, tmp_path / "memory"
        )

        assert process.returncode == 0, (
            "the producer did not exit cleanly: "
            f"{process.returncode} / stdout: {stdout} / stderr: {stderr}"
        )
        report = json.loads(stdout)
        # `frames-ended` here would mean the run ended for a reason other
        # than being asked, and every other assertion would pass anyway.
        assert report["stopped_because"] == "stdin-closed"

    def test_the_flush_ran_rather_than_being_cut_off(
        self, open_capture, tmp_path
    ):
        """The report is printed AFTER `finally: engine.release()`.

        So a report on stdout is proof that execution reached past the
        flush rather than being cut off inside it, which is exactly what
        `TerminateProcess` used to do -- it produced no report at all.
        """
        process, stdout, _ = self._run_and_ask(open_capture, tmp_path / "memory")

        report = json.loads(stdout)
        # Every one of these is computed after the `finally`.
        assert "frames_observed" in report
        assert "stored_observations" in report
        assert "pruned_expired" in report
        assert report["seconds"] >= 0

    def test_the_stop_is_noticed_during_a_quiet_stretch(
        self, open_capture, tmp_path
    ):
        """The defect the first attempt at this shipped with.

        Checking the flag in a wrapper AROUND the frame generator only
        gets control back when a frame arrives. A wearer who puts the
        glasses on a desk and presses Stop is in the quietest stretch
        there is, and that version did not stop at all. The capture in
        this fixture has never produced a frame, so nothing but a check
        inside the poll loop can end this run.
        """
        process, stdout, stderr = self._run_and_ask(
            open_capture, tmp_path / "memory"
        )

        assert process.returncode == 0, f"{stdout} / {stderr}"
        assert json.loads(stdout)["frames_observed"] == 0


class TestTheSupervisorAsksBeforeItShoots:
    class _Process:
        """A child that only dies once it has been asked.

        `wait` raises until `_ask_to_stop` has been used, which is the
        exact shape of the defect: the old code waited out the whole grace
        window on a process nobody had told anything.
        """

        def __init__(self):
            self.pid = 4242
            self.asked = False
            self.terminated = False
            self.waits = []

        def wait(self, timeout=None):
            self.waits.append(timeout)
            if self.asked or self.terminated:
                return 0
            raise subprocess.TimeoutExpired("worker", timeout or 0)

        def terminate(self):
            self.terminated = True

        def poll(self):
            return 0 if (self.asked or self.terminated) else None

    @pytest.fixture
    def supervisor(self):
        return CaptureWorkerSupervisor(
            [
                WorkerSpec(
                    argv=("python", "-c", "pass"),
                    name="w",
                    # OPTED IN. Without this the supervisor must not ask,
                    # and the case below pins that.
                    stop_via_stdin=True,
                )
            ]
        )

    @pytest.fixture
    def unasked_supervisor(self):
        """A spec that never said it understood being asked.

        The world builder's spec is this one. It installs no signal
        handler, so a `CTRL_BREAK_EVENT` reaches it as "die now" -- and
        its ten-second grace exists precisely so a follower can finish
        the build that releases its writer lock. Sending it a signal
        would convert that grace into an instant kill and make the result
        channel report a world `failed` for a build seconds from done.
        """
        return CaptureWorkerSupervisor(
            [WorkerSpec(argv=("python", "-c", "pass"), name="w")]
        )

    def _attach(self, supervisor, process, tmp_path):
        supervisor._spawn = lambda *a, **k: process
        supervisor.capture_opened("cap-1", tmp_path)
        return supervisor

    def test_a_worker_is_asked_and_never_terminated(
        self, supervisor, tmp_path, monkeypatch
    ):
        process = self._Process()

        def asking(target):
            target.asked = True
            return True

        monkeypatch.setattr("tower.capture_workers._ask_to_stop", asking)
        self._attach(supervisor, process, tmp_path)

        supervisor.detach("w", grace_seconds=3.0)

        assert process.asked is True
        assert process.terminated is False, (
            "a producer that stopped when asked must not also be shot"
        )

    def test_a_worker_that_ignores_the_request_is_still_terminated(
        self, supervisor, tmp_path, monkeypatch
    ):
        process = self._Process()
        monkeypatch.setattr("tower.capture_workers._ask_to_stop", lambda p: True)
        self._attach(supervisor, process, tmp_path)

        supervisor.detach("w", grace_seconds=0.01)

        assert process.terminated is True

    def test_a_zero_grace_detach_does_not_ask(
        self, supervisor, tmp_path, monkeypatch
    ):
        """`grace_seconds=0` means "gone now".

        Asking and then immediately terminating is WORSE than not asking:
        a producer that had begun its flush gets killed halfway through
        one, and a half-written record is the thing the grace exists to
        avoid.
        """
        process = self._Process()
        asks = []
        monkeypatch.setattr(
            "tower.capture_workers._ask_to_stop", lambda p: asks.append(p)
        )
        self._attach(supervisor, process, tmp_path)

        supervisor.detach("w", grace_seconds=0.0)

        assert asks == []
        assert process.terminated is True

    def test_a_worker_that_never_opted_in_is_not_asked(
        self, unasked_supervisor, tmp_path, monkeypatch
    ):
        """THE REGRESSION THIS GATE EXISTS FOR.

        A first version of this change asked every worker. On a host with
        a real console that would have delivered a `CTRL_BREAK_EVENT` to
        the world builder, which handles none, at every shutdown -- and
        its grace, the whole point of which is to let a final build
        finish, would have become an immediate kill. It was measured as
        harmless on THIS host only because a pseudoconsole swallows the
        event, which is the worst possible reason for a bug to look fine.
        """
        process = self._Process()
        asks = []
        monkeypatch.setattr(
            "tower.capture_workers._ask_to_stop", lambda p: asks.append(p)
        )
        self._attach(unasked_supervisor, process, tmp_path)

        unasked_supervisor.detach("w", grace_seconds=0.01)

        assert asks == [], (
            "a child that installed no handler must be waited for and then "
            "terminated, never signalled"
        )
        assert process.terminated is True

    def test_asking_a_process_that_cannot_be_signalled_is_contained(self):
        class Refusing:
            pid = 1

            def terminate(self):
                raise OSError("no")

        # Contained as "not asked" so the caller falls through to
        # terminate() rather than leaving a worker alive because an
        # optional courtesy raised.
        assert _ask_to_stop(Refusing()) is False


class TestTheGraceIsWorthHavingAgain:
    def test_the_detach_grace_is_no_longer_zero(self):
        """Pinned because the zero was correct until this change.

        Its comment argued, correctly, that a wait on a process nobody had
        asked to stop is pure cost. That argument is now void -- the
        producer is asked -- and a future reader who restores the zero
        without restoring the reasoning would silently reinstate the lost
        flush.
        """
        from tower.cartridge_session import DETACH_GRACE_SECONDS

        assert DETACH_GRACE_SECONDS > 0.0
        # And bounded: this blocks a synchronous HTTP handler, and the iOS
        # client's own timeout is ten seconds.
        assert DETACH_GRACE_SECONDS <= 5.0

    def test_stop_passes_the_grace_through_to_the_supervisor(self):
        from tower.cartridge_session import CartridgeSession, DETACH_GRACE_SECONDS

        class Supervisor:
            def __init__(self):
                self.detached = []

            def worker_names(self):
                return ("w",)

            def attach(self, name, capture_id, capture_dir):
                return False

            def detach(self, name, grace_seconds=10.0):
                self.detached.append((name, grace_seconds))

            def following(self, name):
                return []

        supervisor = Supervisor()
        session = CartridgeSession(
            cartridge="object_memory",
            worker="w",
            supervisor=supervisor,
            open_capture=lambda: None,
            clock=lambda: 0.0,
        )
        session.apply("start")
        session.apply("stop")

        assert supervisor.detached[-1] == ("w", DETACH_GRACE_SECONDS)


class TestTheStopChannelDoesNotFireOnItsOwn:
    """A held-open pipe must not read as a stop request.

    THE SHARP EDGE, FOUND BY RUNNING THE THING RATHER THAN READING IT.

    `--stop-on-stdin-close` means "the parent is holding my stdin, and
    closing it is a stop". `CaptureWorkerSupervisor` honours that by
    spawning with `stdin=subprocess.PIPE` and keeping the write end, and
    `main.py` sets the flag and `stop_via_stdin` adjacently so a Tower
    cannot pass one without the other.

    A HUMAN CAN. A verification script written during this change used
    `subprocess.run(capture_output=True)`, which hands the child whatever
    stdin the parent had -- a null device. The watcher saw EOF before the
    first frame, the run ended in milliseconds, and the report was a wall
    of zeroes with `stopped_because: stdin-closed` buried in the middle
    of it. Twenty minutes went into "why did the detector find nothing".

    The producer now names that case on stderr when it happens. These two
    cases pin the behaviour either side of it: a pipe held open reads
    every frame, and a pipe closed afterwards still stops the run.
    """

    def _spawn(self, capture, root):
        return subprocess.Popen(
            [
                sys.executable,
                "scripts/object_memory_session.py",
                "--follow-capture",
                str(capture),
                "--root",
                str(root),
                "--detector",
                "none",
                "--verifier",
                "none",
                "--max-idle-polls",
                "40000",
                "--format",
                "json",
                "--stop-on-stdin-close",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=_new_group_flags(),
        )

    def test_a_held_pipe_lets_every_frame_through_before_the_stop(
        self, recorded_capture, tmp_path
    ):
        capture, frame_count = recorded_capture
        process = self._spawn(capture, tmp_path / "memory")
        try:
            # Long enough for a capture this small to be read entirely.
            # The producer polls at 0.25 s and there are three frames in
            # one journal read.
            time.sleep(3.0)
            assert process.poll() is None, (
                "the producer exited before anything closed its stdin, which "
                "is the whole defect this case exists for"
            )
            process.stdin.close()
            # Detached before `communicate()` sees it. Closing stdin IS the
            # stop request under test, and CPython's `Popen._communicate`
            # then flushes the closed writer -- which raises `ValueError:
            # I/O operation on closed file`, a error that code tolerates
            # only as `BrokenPipeError`. `communicate()` also sets
            # `_communication_started` in its own `finally`, so the retry
            # below would fail again on a missing `_fileobj2output` and
            # bury the first error.
            process.stdin = None
            stdout, stderr = process.communicate(timeout=STOP_TIMEOUT_SECONDS)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()

        assert process.returncode == 0, f"{stdout} / {stderr}"
        report = json.loads(stdout)
        assert report["stopped_because"] == "stdin-closed"
        assert report["frames_observed"] == frame_count, (
            "every frame in the journal must have been read before the stop; "
            "a watcher that fires on its own reads none of them"
        )

    def test_a_run_that_read_nothing_says_why_on_stderr(
        self, open_capture, tmp_path
    ):
        """The diagnostic, asserted rather than trusted.

        Closing stdin immediately is indistinguishable in code from a
        supervisor stopping a producer promptly, so this is not something
        the producer can refuse -- only something it can explain.
        """
        process = self._spawn(open_capture, tmp_path / "memory")
        try:
            process.stdin.close()
            # Detached before `communicate()` sees it. Closing stdin IS the
            # stop request under test, and CPython's `Popen._communicate`
            # then flushes the closed writer -- which raises `ValueError:
            # I/O operation on closed file`, a error that code tolerates
            # only as `BrokenPipeError`. `communicate()` also sets
            # `_communication_started` in its own `finally`, so the retry
            # below would fail again on a missing `_fileobj2output` and
            # bury the first error.
            process.stdin = None
            stdout, stderr = process.communicate(timeout=STOP_TIMEOUT_SECONDS)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()

        assert process.returncode == 0, f"{stdout} / {stderr}"
        assert json.loads(stdout)["frames_observed"] == 0
        assert "--stop-on-stdin-close" in stderr, (
            "a run that ended before reading a frame must name the reason "
            "where somebody is already looking"
        )
