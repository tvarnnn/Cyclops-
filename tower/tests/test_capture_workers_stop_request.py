"""Asking a worker to stop without killing it, and owning what it spawned.

Three things the supervisor learnt from the 2026-09-06 physical walk,
each one measured rather than imagined (see
`docs/superpowers/specs/2026-09-06-world-builder-live-history-lifecycle-design.md`):

1. **A soft stop is a request, not a deadline.** `request_stop` closes a
   worker's stdin and returns. The worker stays registered until it exits
   on its own and is reaped -- a builder that is finalizing is allowed to
   finish, and one that is still observing interrupts within a build. No
   signal, no terminate, no wait.

2. **The spec knows how long its worker needs.** `stop_grace_seconds` on
   the spec replaces the caller's grace at shutdown and detach, because
   the builder wraps up in one build (30 s) and the caller cannot know
   that; the caller's value is the default for specs that say nothing.

3. **One process, in a job.** A spec whose argv starts with
   `sys.executable` is spawned through `tower.process_ownership`, so the
   pid held is the interpreter's and not a venv launcher's, and the job it
   is placed in is what a terminate kills.
"""

import os
import subprocess
import sys

import pytest

from tower.capture_workers import (
    DEFAULT_GRACE_SECONDS,
    CaptureWorkerSupervisor,
    WorkerSpec,
)
from tower.process_ownership import interpreter_environment, interpreter_executable


class _Pipe:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakeProcess:
    """A Popen stand-in with a stdin the supervisor may close."""

    _pids = iter(range(70_000, 80_000))

    def __init__(self, argv, **kwargs):
        self.args = list(argv)
        self.kwargs = kwargs
        self.pid = next(FakeProcess._pids)
        self.stdin = _Pipe() if kwargs.get("stdin") is subprocess.PIPE else None
        self._returncode = None
        self.terminated = False
        self.wait_timeouts = []

    def poll(self):
        return self._returncode

    def wait(self, timeout=None):
        self.wait_timeouts.append(timeout)
        if self._returncode is None:
            raise subprocess.TimeoutExpired(self.args, timeout)
        return self._returncode

    def terminate(self):
        self.terminated = True
        self._returncode = -15

    def kill(self):
        self._returncode = -9

    def exit_with(self, code):
        self._returncode = code


@pytest.fixture
def spawned():
    return []


@pytest.fixture
def spawn(spawned):
    def _spawn(argv, **kwargs):
        process = FakeProcess(argv, **kwargs)
        spawned.append(process)
        return process

    return _spawn


@pytest.fixture
def no_signals(monkeypatch):
    """`os.kill` on a made-up pid would reach a stranger's process group."""
    sent = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: sent.append((pid, sig)))
    return sent


def _spec(name, **overrides):
    fields = dict(
        argv=("python", f"{name}.py", "--follow-capture", "{capture_dir}"),
        name=name,
    )
    fields.update(overrides)
    return WorkerSpec(**fields)


def _open(supervisor, tmp_path, capture_id="cap-1"):
    directory = tmp_path / "captures" / capture_id
    directory.mkdir(parents=True, exist_ok=True)
    supervisor.capture_opened(capture_id, directory)
    return directory


# -- request_stop: the soft channel ------------------------------------


class TestRequestStop:
    def test_closes_stdin_and_nothing_else(self, spawn, spawned, tmp_path, no_signals):
        supervisor = CaptureWorkerSupervisor(
            [_spec("builder", stop_via_stdin=True)], spawn=spawn
        )
        _open(supervisor, tmp_path)
        process = spawned[0]

        asked = supervisor.request_stop("builder")

        assert asked == 1
        assert process.stdin.closed is True
        assert process.terminated is False
        assert no_signals == [], "a soft stop must not send a console event"
        assert process.wait_timeouts == [], "a soft stop must not wait"

    def test_the_worker_stays_registered_until_it_exits(
        self, spawn, spawned, tmp_path, no_signals
    ):
        """The builder finishes its final build on its own time.

        Forgetting it here would make a finalizing builder an orphan the
        supervisor cannot see -- and `following` is how the phone learns
        whether the world it is watching is still being written.
        """
        supervisor = CaptureWorkerSupervisor(
            [_spec("builder", stop_via_stdin=True)], spawn=spawn
        )
        _open(supervisor, tmp_path)

        supervisor.request_stop("builder")

        assert [row["capture_id"] for row in supervisor.status()] == ["cap-1"]
        assert supervisor.following("builder") == ["cap-1"]

        spawned[0].exit_with(0)

        assert supervisor.status() == []

    def test_a_worker_that_cannot_be_asked_is_not_counted(
        self, spawn, spawned, tmp_path, no_signals
    ):
        """No stdin pipe, no request. It is not terminated either.

        A spec that never opted in has no reader on its stdin and would
        gain nothing from an EOF; the honest answer is zero, and the log
        says what will actually stop it.
        """
        supervisor = CaptureWorkerSupervisor([_spec("legacy")], spawn=spawn)
        _open(supervisor, tmp_path)

        assert supervisor.request_stop("legacy") == 0
        assert spawned[0].terminated is False
        assert supervisor.status() != []

    def test_an_unknown_spec_is_zero(self, spawn):
        supervisor = CaptureWorkerSupervisor([_spec("builder")], spawn=spawn)

        assert supervisor.request_stop("teapot") == 0

    def test_shutdown_afterwards_still_terminates_a_worker_that_ignored_it(
        self, spawn, spawned, tmp_path, no_signals
    ):
        """The soft request is a courtesy; shutdown is not."""
        supervisor = CaptureWorkerSupervisor(
            [_spec("builder", stop_via_stdin=True)], spawn=spawn
        )
        _open(supervisor, tmp_path)
        supervisor.request_stop("builder")

        supervisor.shutdown(grace_seconds=0.01)

        assert spawned[0].terminated is True
        assert supervisor.status() == []

    def test_is_logged_per_worker(self, spawn, spawned, tmp_path, no_signals, caplog):
        supervisor = CaptureWorkerSupervisor(
            [_spec("builder", stop_via_stdin=True)], spawn=spawn
        )
        _open(supervisor, tmp_path)

        with caplog.at_level("INFO"):
            supervisor.request_stop("builder")

        assert any(
            "asked" in record.getMessage() and str(spawned[0].pid) in record.getMessage()
            for record in caplog.records
        ), caplog.text


# -- stop_grace_seconds: the spec's own bound --------------------------


class TestStopGraceSeconds:
    def test_the_spec_grace_replaces_the_callers_at_shutdown(
        self, spawn, spawned, tmp_path, no_signals
    ):
        supervisor = CaptureWorkerSupervisor(
            [_spec("builder", stop_via_stdin=True, stop_grace_seconds=0.02)],
            spawn=spawn,
        )
        _open(supervisor, tmp_path)

        supervisor.shutdown(grace_seconds=DEFAULT_GRACE_SECONDS)

        assert spawned[0].wait_timeouts[0] == 0.02

    def test_the_spec_grace_replaces_the_callers_at_detach(
        self, spawn, spawned, tmp_path, no_signals
    ):
        supervisor = CaptureWorkerSupervisor(
            [_spec("builder", stop_via_stdin=True, stop_grace_seconds=0.02)],
            spawn=spawn,
        )
        _open(supervisor, tmp_path)

        supervisor.detach("builder", grace_seconds=3.0)

        assert spawned[0].wait_timeouts[0] == 0.02

    def test_a_spec_without_one_gets_the_callers(
        self, spawn, spawned, tmp_path, no_signals
    ):
        supervisor = CaptureWorkerSupervisor(
            [_spec("memory", stop_via_stdin=True)], spawn=spawn
        )
        _open(supervisor, tmp_path)

        supervisor.detach("memory", grace_seconds=0.03)

        assert spawned[0].wait_timeouts[0] == 0.03

    def test_each_spec_gets_its_own_at_a_shared_shutdown(
        self, spawn, spawned, tmp_path, no_signals
    ):
        supervisor = CaptureWorkerSupervisor(
            [
                _spec("builder", stop_via_stdin=True, stop_grace_seconds=0.02),
                _spec("memory", stop_via_stdin=True),
            ],
            spawn=spawn,
        )
        _open(supervisor, tmp_path)

        supervisor.shutdown(grace_seconds=0.01)

        by_name = {process.args[1]: process for process in spawned}
        assert by_name["builder.py"].wait_timeouts[0] == 0.02
        assert by_name["memory.py"].wait_timeouts[0] == 0.01


# -- one process, in a job ---------------------------------------------


class TestProcessOwnership:
    def test_a_sys_executable_argv_is_spawned_as_one_interpreter(
        self, spawn, spawned, tmp_path
    ):
        """`main.py` writes `sys.executable`; the supervisor knows better.

        The rewrite happens here rather than at the wiring point so the
        wiring stays a plain argv and so every spec, present and future,
        gets one process without knowing why it needs one.
        """
        supervisor = CaptureWorkerSupervisor(
            [_spec("builder", argv=(sys.executable, "builder.py", "{capture_dir}"))],
            spawn=spawn,
        )
        _open(supervisor, tmp_path)

        process = spawned[0]
        assert process.args[0] == interpreter_executable()
        assert process.kwargs["env"] == interpreter_environment()

    def test_any_other_argv_is_left_alone(self, spawn, spawned, tmp_path):
        supervisor = CaptureWorkerSupervisor([_spec("builder")], spawn=spawn)
        _open(supervisor, tmp_path)

        process = spawned[0]
        assert process.args[0] == "python"
        assert process.kwargs.get("env") is None

    def test_the_worker_is_placed_in_a_job_and_the_job_is_what_terminates_it(
        self, spawn, spawned, tmp_path, monkeypatch
    ):
        """A terminate goes through the job, so it reaches the whole tree."""

        class FakeJob:
            closed = False

            def __init__(self):
                self.terminated = False

            def terminate(self, exit_code=1):
                self.terminated = True
                # What TerminateJobObject does to the process inside it.
                spawned[0].exit_with(-1)
                return True

            def close(self):
                self.closed = True

        jobs = []

        def assign(process):
            job = FakeJob()
            jobs.append((process, job))
            return job

        monkeypatch.setattr("tower.capture_workers.assign_to_job", assign)
        supervisor = CaptureWorkerSupervisor([_spec("builder")], spawn=spawn)
        _open(supervisor, tmp_path)

        assert jobs and jobs[0][0] is spawned[0]

        supervisor.detach("builder", grace_seconds=0.0)

        assert jobs[0][1].terminated is True
        assert spawned[0].terminated is False, (
            "the job terminated the tree; a second TerminateProcess on the "
            "root would race it for nothing"
        )
        assert supervisor.status() == []

    def test_the_status_pid_is_the_process_the_supervisor_spawned(
        self, spawn, spawned, tmp_path
    ):
        supervisor = CaptureWorkerSupervisor([_spec("builder")], spawn=spawn)
        _open(supervisor, tmp_path)

        assert supervisor.status()[0]["pid"] == spawned[0].pid
