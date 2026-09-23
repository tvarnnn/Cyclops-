"""The stdin stop channel: seen when it closes, and never a lock anybody waits on.

THE 95-MINUTE FINISH, REPRODUCED.

`scripts/world_finish_pending.py` held world 2f447162 for 95 minutes at 0%
CPU. Its stop watcher sat in `os.read(0, 1)` on the pipe the Tower holds, and
its main thread sat in `LoadLibraryExW` on scipy's OpenBLAS, waiting for a
UCRT lock on descriptor 0 that the parked read held. The earlier fix loaded
the known offenders before arming the watcher and called the rest a race
that could not be reproduced once DLLs were warm.

It is not a race. With a live pipe held open behind a parked read, a fresh
`import numpy` hangs until the pipe closes, every time, on the Tower host --
and so does `import pycolmap` after the whole prewarm list, through a second,
kernel-level path. `tower/stdin_stop.py` has both native stacks.

What is asserted here:

* the channel's MEANING is unchanged -- a close is seen, a written byte is
  seen, a null stdin is seen at once, an open pipe is not mistaken for a
  close, a process with no stdin is not watched;
* on Windows, the two measured deadlocks no longer form with the new watcher
  armed, in a spawned interpreter shaped exactly as the Tower spawns one;
* and, as a control, that the OLD watcher still deadlocks here -- so the
  two tests above are not passing because this host stopped reproducing it.
  If it ever does stop, the control skips and says so rather than passing.
"""

import importlib.util
import os
import pathlib
import subprocess
import sys
import queue
import textwrap
import threading
import time

import pytest

from tower.process_ownership import interpreter_command, interpreter_environment
from tower.stdin_stop import watch_stdin_close

TOWER_ROOT = pathlib.Path(__file__).resolve().parent.parent

WINDOWS = os.name == "nt"

# The new watcher, armed exactly as the workers arm it. Nothing numeric is
# imported before BODY runs: the whole point of the Windows cases is that
# BODY is the first thing to load those DLLs, AFTER the watcher is parked.
_NEW_WATCHER = """
import sys, threading, time
sys.path.insert(0, {root!r})
from tower.stdin_stop import watch_stdin_close
closed = threading.Event()
armed = watch_stdin_close(closed.set, name="test-stop-watch")
print("ARMED", armed is not None, "numpy" in sys.modules, flush=True)
"""

# The watcher as it was until 2026-09-23, for the control.
_OLD_WATCHER = """
import os, sys, threading, time
closed = threading.Event()
def _wait():
    try:
        os.read(0, 1)
    except OSError:
        pass
    closed.set()
threading.Thread(target=_wait, name="old-stop-watch", daemon=True).start()
print("ARMED", True, "numpy" in sys.modules, flush=True)
"""


class _Child:
    """A spawned interpreter, and ONE reader of its stdout.

    One reader for the child's whole life, feeding a queue. A reader per
    line would leave a timed-out `readline` still parked on the pipe, and
    it would swallow the very line the next wait is for -- the control
    below times out on purpose and then waits again.
    """

    def __init__(self, prologue: str, body: str, *, stdin=subprocess.PIPE):
        source = prologue.format(root=str(TOWER_ROOT)) + textwrap.dedent(body)
        self.process = subprocess.Popen(
            interpreter_command("-c", source),
            cwd=str(TOWER_ROOT),
            env=interpreter_environment(),
            stdin=stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            # The Tower's own spawn flags: `_BackgroundChore.start` and
            # `capture_workers._start` both put the child in its own group.
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if WINDOWS else 0,
        )
        self.stdin = self.process.stdin
        self._lines: queue.Queue = queue.Queue()
        self.seen: list[str] = []
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        for raw in self.process.stdout:
            self._lines.put(raw.decode(errors="replace").strip())
        self._lines.put(None)

    def line(self, timeout: float) -> str | None:
        """The next line, or None if none came in time (or the child ended)."""
        try:
            got = self._lines.get(timeout=timeout)
        except queue.Empty:
            return None
        if got is not None:
            self.seen.append(got)
        return got

    def reap(self) -> str:
        if self.process.poll() is None:
            self.process.kill()
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:  # pragma: no cover
            pass
        return "\n".join(self.seen[-20:])


class TestTheChannelMeansWhatItMeant:
    def test_closing_the_pipe_is_seen(self):
        child = _Child(_NEW_WATCHER, """
            print("SEEN" if closed.wait(20) else "MISSED", flush=True)
        """)
        try:
            assert (child.line(60) or "").startswith("ARMED True")
            time.sleep(0.5)
            child.stdin.close()
            started = time.monotonic()
            assert child.line(30) == "SEEN"
            # Poll latency, not a grace: the finisher is given five seconds.
            assert time.monotonic() - started < 3.0
        finally:
            child.reap()

    def test_a_byte_written_is_also_the_request(self):
        child = _Child(_NEW_WATCHER, """
            print("SEEN" if closed.wait(20) else "MISSED", flush=True)
        """)
        try:
            assert (child.line(60) or "").startswith("ARMED True")
            child.stdin.write(b"x")
            child.stdin.flush()
            assert child.line(30) == "SEEN"
        finally:
            child.reap()

    def test_an_open_pipe_is_not_mistaken_for_a_close(self):
        child = _Child(_NEW_WATCHER, """
            print("SEEN" if closed.wait(2.0) else "HELD", flush=True)
        """)
        try:
            assert (child.line(60) or "").startswith("ARMED True")
            assert child.line(30) == "HELD"
        finally:
            child.reap()

    def test_a_null_stdin_is_the_request_at_once(self):
        child = _Child(_NEW_WATCHER, """
            print("SEEN" if closed.wait(10) else "MISSED", flush=True)
        """, stdin=subprocess.DEVNULL)
        try:
            assert (child.line(60) or "").startswith("ARMED True")
            assert child.line(30) == "SEEN"
        finally:
            child.reap()

    def test_a_process_with_no_stdin_is_not_watched(self, monkeypatch):
        monkeypatch.setattr(sys, "stdin", None)
        calls = []
        assert watch_stdin_close(lambda: calls.append(1), name="x") is None
        assert calls == []


def _assert_loads_behind_the_watcher(statement: str, what: str):
    child = _Child(_NEW_WATCHER, f"""
        time.sleep(1.0)  # let the watcher reach its first poll
        started = time.perf_counter()
        {statement}
        print("LOADED", round(time.perf_counter() - started, 2), flush=True)
        print("SEEN" if closed.wait(20) else "MISSED", flush=True)
    """)
    try:
        armed = child.line(60) or ""
        # Nothing numeric may be resident before the load under test, or the
        # load would not be the first and this would prove nothing.
        assert armed == "ARMED True False", armed
        # THE PIPE STAYS OPEN until the load has reported. That is the half
        # of the deadlock the parent contributes; closing it early would
        # release the lock and test nothing.
        loaded = child.line(90)
        assert loaded is not None and loaded.startswith("LOADED"), (
            f"`{what}` never returned with the stdin watcher armed and the "
            "pipe held open -- the descriptor-0 deadlock is back. Dump it "
            "with `py-spy dump --native --pid <pid>`.\n" + child.reap()
        )
        child.stdin.close()
        assert child.line(30) == "SEEN"
    finally:
        child.reap()


@pytest.mark.skipif(not WINDOWS, reason="the deadlock is a Windows descriptor-0 path")
class TestTheDeadlockNoLongerForms:
    def test_a_first_numpy_load_behind_the_watcher(self):
        """The UCRT path: libgfortran inside numpy's OpenBLAS, descriptor 0's lock."""
        _assert_loads_behind_the_watcher("import numpy", "import numpy")

    def test_a_first_scipy_linalg_load_behind_the_watcher(self):
        """The DLL on the finisher's own py-spy stack, 2026-09-22."""
        _assert_loads_behind_the_watcher("import scipy.linalg", "import scipy.linalg")

    @pytest.mark.skipif(
        importlib.util.find_spec("pycolmap") is None, reason="pycolmap is not installed"
    )
    def test_a_first_pycolmap_load_behind_the_watcher(self):
        """The kernel path: msvcrt `fstat64(0)` -> `PeekNamedPipe` on the pipe.

        The one the prewarm could never have covered: it hung with numpy,
        scipy, cv2, torch and the CUDA driver all resident first.
        """
        _assert_loads_behind_the_watcher("import pycolmap", "import pycolmap")


@pytest.mark.skipif(not WINDOWS, reason="the deadlock is a Windows descriptor-0 path")
class TestTheControl:
    def test_the_old_parked_read_still_deadlocks_on_this_host(self):
        """If this stops hanging, the tests above stop proving anything."""
        child = _Child(_OLD_WATCHER, """
            time.sleep(1.0)
            import numpy
            print("LOADED", flush=True)
        """)
        try:
            assert (child.line(60) or "") == "ARMED True False"
            loaded = child.line(15)
            if loaded is not None:
                pytest.skip(
                    "this host no longer reproduces the descriptor-0 deadlock "
                    "with the old watcher; TestTheDeadlockNoLongerForms is "
                    "now weaker evidence than it was written to be"
                )
            # And the close releases it, which is the whole shape of the bug:
            # hung until the Tower stops, then finishing as if nothing had
            # happened.
            child.stdin.close()
            assert child.line(30) == "LOADED"
        finally:
            child.reap()
