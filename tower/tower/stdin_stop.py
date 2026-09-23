"""Notice that the parent closed this process's stdin, without READING it.

WHY THIS MODULE EXISTS, REPRODUCED ON DEMAND ON 2026-09-23.

A worker the Tower spawns with `stdin=PIPE` is stopped by closing that pipe:
nothing is ever written to it, the request IS the close, and a pipe needs no
console. Until this module, each worker noticed the close the obvious way --
a daemon thread parked in `os.read(0, 1)` for the whole run.

On Windows that parked read is half of a deadlock, and it is deterministic,
not a race. Two paths were measured on the Tower host, each with a fresh
interpreter, a live pipe held open by the parent, and the old watcher armed:

  * `import numpy` (or `import scipy.linalg`) hangs at 0% CPU until the pipe
    is closed, EVERY time. The OpenBLAS DLLs numpy and scipy ship statically
    link libgfortran, whose load-time constructor (`init_units`) calls a UCRT
    function on descriptor 0. The UCRT takes descriptor 0's lock for that --
    and `os.read(0, 1)` is UCRT `_read`, which holds the same lock for as
    long as its `ReadFile` blocks. Main thread: `LoadLibraryExW` ->
    `libscipy_openblas*.dll` -> `ucrtbase` -> `RtlEnterCriticalSection`.
    Watcher: `_read` -> `ReadFile`. This is the exact native stack py-spy
    took from the finisher that held world 2f447162 for 95 minutes.
  * `import pycolmap` hangs the same way even with numpy, scipy, cv2, torch
    and the CUDA driver all loaded first. Its bundled `libgfortran-5.dll` is
    linked against `msvcrt.dll`, not the UCRT, so it never touches that lock;
    its `init_units` reaches `fstat64(0)`, which calls `PeekNamedPipe` on the
    stdin pipe -- and Windows serialises every call on a synchronous file
    object behind a pending synchronous `ReadFile`. Blocked in the kernel,
    in `ZwFsControlFile`, until the read returns.

So the hazard is not "some DLL that spawns threads" and it is not
cold-versus-warm DLLs: it is ANY library whose load-time code touches
descriptor 0 while another thread has a read pending on it. Loading known
offenders before arming the watcher (`tower/native_prewarm.py`) removed the
two that had been caught and left every other one in place -- pycolmap is
in the same environment.

WHAT THIS DOES INSTEAD. On Windows, when stdin is a pipe, no read is ever
left pending: a daemon thread asks `PeekNamedPipe` a few times a second.
Each call returns at once -- nothing is pending, so nothing waits behind it
and nothing it holds is held for longer than the call -- and it fails with
`ERROR_BROKEN_PIPE` the moment the parent's write end is closed. Descriptor
0, its UCRT lock and the pipe's file object are left exactly as they were
for every library that wants to look at them.

A stdin that is NOT a pipe on Windows (a console, `NUL`, a file -- a person
running a worker by hand) is read with `_winapi.ReadFile` on the OS handle,
which bypasses the UCRT descriptor lock; a console handle is not a pipe, so
the kernel path above does not apply to it. On POSIX nothing here was ever a
hazard -- `read(2)` takes no user-space lock -- and the blocking read stays.

NEVER THROUGH `sys.stdin`. Every path here uses the raw descriptor or its OS
handle. A read through the buffered object holds that object's lock while it
blocks, and interpreter shutdown -- the ordinary ending, the parent still
holding the pipe -- then aborts with "Fatal Python error:
_enter_buffered_busy: could not acquire lock for <_io.BufferedReader
name='<stdin>'>", a non-zero exit for a run that succeeded.

SEMANTICS ARE UNCHANGED. The close is the request; a byte written by the
parent is also the request, since it can mean nothing else; an unreadable or
already-closed stdin is the request; a process with no stdin at all is not
watched. The cost is up to `PIPE_POLL_SECONDS` of latency on a stop that
every caller gives seconds of grace.

**Cartridge-blind**, like `process_ownership.py`: it knows about pipes and
threads and names no cartridge.
"""

from __future__ import annotations

import os
import stat
import sys
import threading
import time
from collections.abc import Callable

# How often a Windows pipe is asked whether its writer is still there.
# Small against every stop grace the Tower gives (five seconds for the
# finisher, thirty for the builder), and cheap: one non-blocking syscall.
PIPE_POLL_SECONDS = 0.2


def stdin_descriptor() -> int | None:
    """Descriptor 0 as this process sees it, or None when there is none."""
    try:
        return sys.stdin.fileno() if sys.stdin is not None else None
    except (AttributeError, ValueError, OSError):
        return None


def _wait_windows_pipe(handle: int, poll_seconds: float) -> None:
    """Return once the pipe's writer has gone, or has written anything."""
    import _winapi  # noqa: PLC0415 -- Windows only

    while True:
        try:
            available, _ = _winapi.PeekNamedPipe(handle, 0)
        except OSError:
            # ERROR_BROKEN_PIPE is the close. Anything else -- an invalid
            # handle, a pipe that cannot be asked -- is an unreadable pipe,
            # and an unreadable pipe is itself the request: the alternative
            # is a worker nobody can stop.
            return
        if available:
            return
        time.sleep(poll_seconds)


def _wait_windows_handle(handle: int) -> None:
    """Block on the OS handle, NOT through the UCRT descriptor lock."""
    import _winapi  # noqa: PLC0415 -- Windows only

    try:
        _winapi.ReadFile(handle, 1)
    except OSError:
        pass


def _wait_posix(fd: int) -> None:
    try:
        os.read(fd, 1)
    except OSError:
        pass


def _waiter(fd: int, poll_seconds: float) -> Callable[[], None]:
    if os.name != "nt":
        return lambda: _wait_posix(fd)
    import msvcrt  # noqa: PLC0415 -- Windows only

    try:
        handle = msvcrt.get_osfhandle(fd)
        is_pipe = stat.S_ISFIFO(os.fstat(fd).st_mode)
    except OSError:
        # No handle behind the descriptor: unreadable, which is the request.
        return lambda: None
    if is_pipe:
        return lambda: _wait_windows_pipe(handle, poll_seconds)
    return lambda: _wait_windows_handle(handle)


def watch_stdin_close(
    on_close: Callable[[], None],
    *,
    name: str,
    poll_seconds: float = PIPE_POLL_SECONDS,
) -> threading.Thread | None:
    """Call `on_close()` once, from a daemon thread, when stdin closes.

    Returns the thread, or None when this process has no stdin to watch --
    in which case nothing is armed and `on_close` is never called, exactly
    as before this module existed.

    A DAEMON thread, because the ordinary ending of every worker is that it
    finishes its own work while the parent still holds the pipe; a
    non-daemon thread waiting for a close that is not coming would keep the
    process alive after that.
    """
    fd = stdin_descriptor()
    if fd is None:
        return None
    wait = _waiter(fd, poll_seconds)

    def run() -> None:
        wait()
        on_close()

    thread = threading.Thread(target=run, name=name, daemon=True)
    thread.start()
    return thread
