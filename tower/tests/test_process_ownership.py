"""The supervisor owns the process it started, and everything under it.

WHAT WAS MEASURED ON THIS BOX, 2026-09-06 (audit F2).

`sys.executable` inside the Tower's venv is `.venv\\Scripts\\python.exe`,
which on Windows is a LAUNCHER: it spawns the real interpreter
(`sys._base_executable`) as a child and waits. So every worker the
supervisor spawned via `sys.executable` was a PAIR of processes, and the
pid the supervisor held -- the one in `/health`, the one it terminated at
shutdown -- was the launcher's. The launcher puts the interpreter in a
job with `KILL_ON_JOB_CLOSE | SILENT_BREAKAWAY_OK`, so terminating the
launcher does kill the interpreter, but every grandchild (a solve child
in the middle of a 30-135 s mapping run) survives, and that is exactly
what the physical walk left behind: a solver writing a solution nothing
ever merged.

Two changes, and each has a real-process test here because a fake would
prove nothing about the platform:

1. spawn ONE process (`sys._base_executable` with `__PYVENV_LAUNCHER__`
   pointing at the venv), so the pid held is the pid doing the work;
2. put it in a supervisor-owned Job Object without breakaway, so a
   terminate kills the tree and a Tower that dies takes its tree with
   it.
"""

import json
import os
import subprocess
import sys
import time

import psutil
import pytest

from tower.process_ownership import (
    assign_to_job,
    interpreter_argv,
    interpreter_command,
    interpreter_environment,
    terminate_tree,
)

windows_only = pytest.mark.skipif(os.name != "nt", reason="Job Objects are Windows")

# A child that reports itself and then spawns a grandchild that sleeps, so
# a test can hold three generations and ask what a terminate reached.
_PARENT_WITH_GRANDCHILD = """
import json, subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
print(json.dumps({"grandchild": child.pid}), flush=True)
time.sleep(60)
"""


def _wait_until(predicate, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return bool(predicate())


def _gone(pid: int) -> bool:
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return True
    try:
        return proc.status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


def _kill_quietly(*pids):
    for pid in pids:
        try:
            psutil.Process(pid).kill()
        except psutil.Error:
            pass


# -- one process, venv-aware ------------------------------------------


def test_the_interpreter_command_names_a_real_interpreter():
    command = interpreter_command("-c", "pass")

    assert os.path.isfile(command[0]), command
    assert command[1:] == ("-c", "pass")


def test_interpreter_argv_puts_the_script_first():
    argv = interpreter_argv("some/script.py", "--flag", "value")

    assert argv[0] == interpreter_command()[0]
    assert argv[1:] == ("some/script.py", "--flag", "value")


def test_interpreter_environment_does_not_mutate_its_input():
    base = {"A": "1"}

    env = interpreter_environment(base)

    assert base == {"A": "1"}
    assert env["A"] == "1"


def test_the_child_is_one_venv_aware_process():
    """The recipe gives ONE process whose `sys.prefix` is this venv.

    Asserted against a live child rather than by reading `sys` flags,
    because the claim is about what the operating system sees: a launcher
    pair looks like one process to `Popen` and two to `psutil`.
    """
    child = subprocess.Popen(
        interpreter_command(
            "-c",
            "import sys, json, psutil\n"
            "print(json.dumps([sys.prefix, sys.executable]), flush=True)\n"
            "input()\n",
        ),
        env=interpreter_environment(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        line = child.stdout.readline()
        prefix, executable = json.loads(line)
        assert os.path.normcase(prefix) == os.path.normcase(sys.prefix), (
            "the child does not see this venv, so it cannot import the "
            "Tower's packages"
        )
        assert executable, "the child has no idea what it is"
        if os.name == "nt":
            assert psutil.Process(child.pid).children() == [], (
                "the child spawned something of its own: it is still the "
                "launcher pair, and the supervisor holds the wrong pid"
            )
    finally:
        child.stdin.close()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()


# -- the job owns the tree ---------------------------------------------


@windows_only
def test_a_terminate_through_the_supervisor_reaches_the_grandchild(tmp_path):
    """The failure of the 2026-09-06 walk, reproduced and closed.

    A child that itself spawned a solver is terminated; the solver must
    be gone too, within seconds, without anybody knowing its pid.
    """
    child = subprocess.Popen(
        interpreter_command("-c", _PARENT_WITH_GRANDCHILD),
        env=interpreter_environment(),
        stdout=subprocess.PIPE,
        text=True,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    job = assign_to_job(child)
    grandchild = None
    try:
        assert job is not None, "the child could not be placed in a job"
        grandchild = json.loads(child.stdout.readline())["grandchild"]
        assert not _gone(grandchild)

        assert terminate_tree(child, job=job, timeout=5.0) is True

        assert _wait_until(lambda: _gone(grandchild), 5.0), (
            "the grandchild survived: the job did not own it"
        )
        assert child.poll() is not None
    finally:
        _kill_quietly(child.pid, *([grandchild] if grandchild else []))


@windows_only
def test_closing_the_job_handle_kills_the_tree():
    """A Tower that dies must take its builders and their solvers with it.

    `KILL_ON_JOB_CLOSE` is the mechanism: the last handle to the job
    closing is what a dead Tower looks like to the kernel, and nothing
    the child does can opt out because the job has no breakaway flag.
    """
    child = subprocess.Popen(
        interpreter_command("-c", _PARENT_WITH_GRANDCHILD),
        env=interpreter_environment(),
        stdout=subprocess.PIPE,
        text=True,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    job = assign_to_job(child)
    grandchild = None
    try:
        assert job is not None
        grandchild = json.loads(child.stdout.readline())["grandchild"]

        job.close()

        assert _wait_until(lambda: child.poll() is not None, 5.0)
        assert _wait_until(lambda: _gone(grandchild), 5.0), (
            "the grandchild outlived the job handle"
        )
    finally:
        _kill_quietly(child.pid, *([grandchild] if grandchild else []))


def test_terminate_tree_without_a_job_still_kills_the_grandchild():
    """The POSIX path, and the Windows fallback when a job cannot be had.

    `psutil` walks the tree first, because terminating only the handle
    we hold orphans exactly the process doing the work.
    """
    child = subprocess.Popen(
        interpreter_command("-c", _PARENT_WITH_GRANDCHILD),
        env=interpreter_environment(),
        stdout=subprocess.PIPE,
        text=True,
    )
    grandchild = None
    try:
        grandchild = json.loads(child.stdout.readline())["grandchild"]

        assert terminate_tree(child, job=None, timeout=5.0) is True

        assert _wait_until(lambda: _gone(grandchild), 5.0)
        assert child.poll() is not None
    finally:
        _kill_quietly(child.pid, *([grandchild] if grandchild else []))


def test_a_process_stand_in_is_never_walked_with_psutil():
    """A fake's pid may belong to a stranger; nothing may be killed by it.

    Every supervisor test in this repository uses a `FakeProcess` with a
    made-up pid, and `terminate_tree` is now on the path those tests
    exercise. A pid lookup on a made-up number would find whatever real
    process happens to hold it.
    """

    class Fake:
        pid = 4  # the Windows System process; never a valid target

        def __init__(self):
            self.terminated = False
            self._code = None

        def poll(self):
            return self._code

        def terminate(self):
            self.terminated = True
            self._code = -15

        def wait(self, timeout=None):
            if self._code is None:
                raise subprocess.TimeoutExpired("fake", timeout)
            return self._code

    fake = Fake()

    assert terminate_tree(fake, job=None, timeout=0.1) is True
    assert fake.terminated is True
    assert assign_to_job(fake) is None


def test_a_terminate_that_raises_reports_not_gone():
    class Immortal:
        pid = 4

        def poll(self):
            return None

        def terminate(self):
            raise PermissionError("access is denied")

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired("immortal", timeout)

    assert terminate_tree(Immortal(), job=None, timeout=0.1) is False
