"""Spawn a child as ONE process, own it, and be able to kill its tree.

WHY THIS MODULE EXISTS, MEASURED ON 2026-09-06.

The first physical test of the combined stack ended with a builder dead
and its background solve child still running for another 32 seconds,
writing a solution nothing would ever merge. The process audit that
followed (`Glasses-scratch\\wb-live-history\\process-audit`) found the
mechanism, and it is not a bug in anybody's code -- it is how a Windows
venv works:

  * `sys.executable` in the Tower is `.venv\\Scripts\\python.exe`, which
    is a LAUNCHER. It spawns the real interpreter (`sys._base_executable`)
    as a child and waits for it. Every worker the supervisor started was
    therefore two processes, and the pid it held -- the one in `/health`,
    the one in the log, the one `terminate()` shot -- was the launcher's.
  * The launcher puts its interpreter in a Job Object with
    `KILL_ON_JOB_CLOSE | SILENT_BREAKAWAY_OK`. Terminating the launcher
    closes that job and kills the interpreter, which is why things looked
    fine. But SILENT_BREAKAWAY_OK means every process the interpreter
    spawns is quietly created OUTSIDE the job. Grandchildren survive.

Two facts were verified on this box before anything here was written:
spawning `sys._base_executable` with `__PYVENV_LAUNCHER__` set to the
venv's python gives ONE process whose `sys.prefix` is the venv and whose
packages import; and nested Job Objects work (Windows 8+), so a child
that is already in some outer job can still be placed in ours.

**Cartridge-blind, like `capture_workers.py`.** This module knows how to
start an interpreter and how to stop a tree. It imports no cartridge and
names none, and it must stay that way: the second thing that ever needs
to own a process gets this for free only if it contains no trace of the
first.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# The variable the venv launcher sets for the interpreter it spawns. An
# interpreter started with it set reports the venv as `sys.prefix` and
# the launcher's path as `sys.executable` -- which is what makes it
# venv-aware without the launcher being in the picture at all.
PYVENV_LAUNCHER_VARIABLE = "__PYVENV_LAUNCHER__"


# -- one process --------------------------------------------------------


def _launcher_pair() -> str | None:
    """The real interpreter behind a venv launcher, or None.

    None means "`sys.executable` already IS the interpreter": POSIX, where
    a venv's python is a symlink and one process; or Windows outside a
    venv; or a build with no `_base_executable`. In every one of those
    the unchanged recipe is already correct.
    """
    if os.name != "nt":
        return None
    base = getattr(sys, "_base_executable", None)
    if not base or not os.path.isfile(base):
        return None
    if os.path.normcase(os.path.abspath(base)) == os.path.normcase(
        os.path.abspath(sys.executable)
    ):
        return None
    return base


def interpreter_executable() -> str:
    """argv[0] for a child that must be one process and see this venv."""
    return _launcher_pair() or sys.executable


def interpreter_command(*args: str) -> tuple[str, ...]:
    """`(python, *args)` for any interpreter invocation, `-c` included."""
    return (interpreter_executable(), *(str(arg) for arg in args))


def interpreter_argv(script: str | Path, *args: str) -> tuple[str, ...]:
    """`(python, script, *args)` -- the shape every worker spec has."""
    return interpreter_command(str(script), *args)


def interpreter_environment(base: dict | None = None) -> dict:
    """The environment that makes `interpreter_executable()` venv-aware.

    A COPY, never the caller's mapping. On a launcher pair it carries
    `__PYVENV_LAUNCHER__` pointing at the venv's python, which is the only
    thing the launcher itself would have done. Elsewhere it is the input,
    unchanged, so a caller can pass it unconditionally.
    """
    env = dict(os.environ if base is None else base)
    if _launcher_pair() is not None:
        env[PYVENV_LAUNCHER_VARIABLE] = sys.executable
    return env


# -- the job ------------------------------------------------------------

# JOBOBJECT_BASIC_LIMIT_INFORMATION.LimitFlags. KILL_ON_JOB_CLOSE alone:
# no BREAKAWAY_OK, no SILENT_BREAKAWAY_OK, which is the whole difference
# between this job and the launcher's.
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9

# Said once per Tower, not once per worker. A host where jobs cannot be
# created (a policy, an exotic sandbox) says so at the first spawn and is
# then quiet; every later worker simply runs without one, and
# `terminate_tree` falls back to walking the tree with psutil.
_job_failure_logged = False


class JobHandle:
    """A Job Object handle kept alive for as long as the worker is owned.

    THE HANDLE IS THE OWNERSHIP. `KILL_ON_JOB_CLOSE` fires when the last
    handle to the job closes, so this object must live on the worker
    record: drop it, and the kernel reads that as the Tower having died
    and kills the tree. That is the behaviour wanted at Tower death and
    exactly not the behaviour wanted from a garbage collector, which is
    why `_Worker` holds it and nothing else does.
    """

    __slots__ = ("_handle",)

    def __init__(self, handle: int) -> None:
        self._handle = handle

    @property
    def closed(self) -> bool:
        return self._handle is None

    def terminate(self, exit_code: int = 1) -> bool:
        """`TerminateJobObject`: every process in the job, at once."""
        if self._handle is None:
            return False
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        ok = kernel32.TerminateJobObject(
            ctypes.c_void_p(self._handle), ctypes.c_uint(exit_code)
        )
        if not ok:
            logger.warning(
                "[Tower][Worker] TerminateJobObject failed (error %s); falling "
                "back to terminating the tree by hand",
                ctypes.get_last_error(),
            )
        return bool(ok)

    def close(self) -> None:
        """Release the handle. With no other handle open, this kills the job."""
        if self._handle is None:
            return
        import ctypes

        handle, self._handle = self._handle, None
        ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(
            ctypes.c_void_p(handle)
        )

    def __del__(self) -> None:  # pragma: no cover - interpreter teardown
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


def _job_limits_structure():
    """The ctypes shape of JOBOBJECT_EXTENDED_LIMIT_INFORMATION."""
    import ctypes
    from ctypes import wintypes

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
            ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    return JOBOBJECT_EXTENDED_LIMIT_INFORMATION


def assign_to_job(process) -> JobHandle | None:
    """Put a freshly spawned child in a job this process owns.

    Windows only, and only for a real `Popen` (one with a `_handle`); a
    stand-in process object in a test gets None without a word. None is
    also the answer when the kernel refuses, logged once, because a
    worker that runs without a job is still a worker -- `terminate_tree`
    walks its tree by hand instead.

    THERE IS A WINDOW, AND IT IS ACCEPTED. The child exists before it is
    assigned, so a grandchild spawned in the first microseconds would
    escape the job. A Python interpreter takes tens of milliseconds to
    reach its first line and the assignment happens immediately after
    `Popen` returns; the alternative, `CREATE_SUSPENDED` plus a resume, is
    not something `subprocess` exposes.
    """
    global _job_failure_logged

    if os.name != "nt":
        return None
    handle = getattr(process, "_handle", None)
    if handle is None:
        return None

    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = ctypes.c_void_p
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        _log_job_failure("CreateJobObject", ctypes.get_last_error())
        return None

    limits = _job_limits_structure()()
    limits.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel32.SetInformationJobObject(
        ctypes.c_void_p(job),
        ctypes.c_int(_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION),
        ctypes.byref(limits),
        ctypes.sizeof(limits),
    ):
        _log_job_failure("SetInformationJobObject", ctypes.get_last_error())
        kernel32.CloseHandle(ctypes.c_void_p(job))
        return None

    if not kernel32.AssignProcessToJobObject(
        ctypes.c_void_p(job), ctypes.c_void_p(int(handle))
    ):
        _log_job_failure("AssignProcessToJobObject", ctypes.get_last_error())
        kernel32.CloseHandle(ctypes.c_void_p(job))
        return None

    return JobHandle(job)


def _log_job_failure(call: str, error: int) -> None:
    global _job_failure_logged
    if _job_failure_logged:
        logger.debug("[Tower][Worker] %s failed (error %s)", call, error)
        return
    _job_failure_logged = True
    logger.warning(
        "[Tower][Worker] %s failed (error %s): workers on this host run "
        "WITHOUT a job object. They are still stopped at shutdown, by "
        "walking each process tree, but a Tower that dies uncleanly will "
        "not take its workers with it. Said once.",
        call,
        error,
    )


# -- the tree -----------------------------------------------------------


def terminate_tree(
    process, *, job: JobHandle | None = None, timeout: float, hard: bool = False
) -> bool:
    """Stop a child and everything under it. True when all of it is gone.

    With a job, `TerminateJobObject` does the whole tree in one call and
    the wait afterwards is only for the handle to notice. Without one --
    POSIX, or a Windows host that refused a job -- the tree is walked
    with psutil FIRST, because terminating only the handle we hold is
    precisely how the 2026-09-06 solve child was orphaned.

    ONLY A REAL `Popen` IS WALKED. A process stand-in carries a made-up
    pid, and the number it made up belongs to some real process on this
    machine; looking it up and terminating its children would be the one
    defect in this file that could reach outside the test suite.

    `terminate()` raising is reported as "not gone" rather than raised:
    the caller keeps such a worker in its registry so it stays visible,
    which is the contract `CaptureWorkerSupervisor._stop_worker` has
    always had.

    `hard` skips the polite half on POSIX: SIGKILL, never SIGTERM first. A
    SIGTERM runs the child's own handler, and for a stalled recovery
    finisher that handler is the one that gives its attempt back -- the
    stall would then never count (review, 2026-09-23). On Windows every path
    here is already a TerminateProcess, which runs no handler.
    """
    if job is not None and not job.closed and job.terminate():
        return _wait_gone(process, timeout)

    descendants = _descendants(process)
    for proc in descendants:
        try:
            if hard:
                proc.kill()
            else:
                proc.terminate()
        except Exception:  # noqa: BLE001 -- already gone, or not ours to touch
            pass

    try:
        if hard:
            process.kill()
        else:
            process.terminate()
    except Exception:
        logger.exception(
            "[Tower][Worker] could NOT terminate pid %s; it stays in the "
            "registry so it is still visible to /health and to the next "
            "shutdown, and nothing else will be attached in its place",
            getattr(process, "pid", "?"),
        )
        return False

    gone = _wait_gone(process, timeout)
    if not gone:
        try:
            process.kill()
            gone = _wait_gone(process, timeout)
        except Exception:  # noqa: BLE001
            pass

    for proc in descendants:
        try:
            if proc.is_running():
                proc.kill()
        except Exception:  # noqa: BLE001
            pass
    for proc in descendants:
        try:
            proc.wait(timeout=timeout)
        except Exception:  # noqa: BLE001
            gone = False
    return gone


def _descendants(process) -> list:
    if not isinstance(process, subprocess.Popen):
        return []
    # A process that has already exited may have had its pid reused;
    # asking psutil about it would answer for a stranger.
    if process.poll() is not None:
        return []
    try:
        import psutil

        return psutil.Process(process.pid).children(recursive=True)
    except Exception:  # noqa: BLE001 -- psutil missing, or the process just left
        return []


def _wait_gone(process, timeout: float) -> bool:
    try:
        process.wait(timeout=timeout)
        return True
    except Exception:  # noqa: BLE001 -- TimeoutExpired, or a stand-in's own error
        return process.poll() is not None
