#!/usr/bin/env python
"""Drive a World Builder mapping session offline, from frames on disk.

This is the V1 driver. It calls exactly the same `engine.observe()` a
future in-process module adapter would, which is why live-versus-offline
is a driver choice rather than an architecture choice: nothing about the
engine changes when frames arrive over a WebSocket instead of from a
directory.

Frame sources:

  --frames DIR     a directory of .jpg files, processed in sorted order.
                   This is what a recorded capture session looks like.
  --synthetic      render a synthetic walk instead. SYNTHETIC, NOT
                   PHYSICAL -- for exercising the pipeline with no
                   hardware, never for any claim about the real camera.
  --follow-capture DIR
                   tail a capture directory the Tower is writing RIGHT NOW,
                   observing each frame as it lands. This is the live path:
                   arm the recorder (TOWER_CAPTURE_ROOT), walk the room,
                   and the world builds while you walk. It runs in a
                   SEPARATE PROCESS from the Tower on purpose -- the frame
                   path never pays for reconstruction, which is why an
                   expensive rebuild can run repeatedly mid-session.

                   Unlike --frames, this reads the JOURNAL, so source_seq,
                   tx_seq and receipt time survive. A directory glob would
                   throw them away and with them any ability to reason
                   about dropped frames.

Intrinsics come from the intrinsics store, keyed by the resolution the
frames MEASURE at -- not by anything this process declares. `--intrinsics`
overrides the store. When neither yields a calibration for the observed
resolution the intrinsics stay unknown, the unposed backend runs, and the
engine honestly produces no poses. There is no flag that invents a focal
length, and nothing rescales a calibration from another resolution.

WHEN THE LOOKUP HAPPENS, AND WHY THAT IS THE HARD PART

The mapping session does not open until a frame has actually been
observed. That ordering is not tidiness; it is the fix for a bug that
cost the 2026-08-25 physical walk.

The Tower does not wait for a frame before attaching a builder. It
attaches at `stream_start`, from `_start_capture`, on the line after the
capture id is minted -- so on the live path this process routinely opens
against a capture directory whose journal has no rows in it yet. This
file used to resolve intrinsics right there, get "resolution not
observed", and freeze `unknown()` into the session record. The 360x640
frames arrived about a second later, a 360x640 calibration was sitting
in the store the whole time, and nothing ever looked again: 75 keyframes,
`backend: unposed`, `downgraded_from: classical`, zero poses.

It was not a race that could be tuned away by starting later. Importing
this module with OpenCV takes ~135ms on the Tower host and the first
frame lands ~1s after `stream_start`, so the worker won every time.

The session's intrinsics are a property of the PIXELS, so the session
cannot honestly be opened before a pixel exists. The world is created
immediately -- it costs nothing and gives the result channel something to
report -- and then this process blocks on the first frame, measures it,
and asks the store about the size it actually saw. A capture that closes
without ever delivering a frame ends that wait and opens an empty session
rather than hanging.

    .venv\\Scripts\\python.exe scripts/world_build_session.py --synthetic --name "Test Room"
    .venv\\Scripts\\python.exe scripts/world_build_session.py --frames data/capture/xyz
"""

import argparse
import io
import itertools
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tower.artifact_paths import artifact_root_arg  # noqa: E402
from tower.native_prewarm import prewarm_world_builder  # noqa: E402
from tower.stdin_stop import watch_stdin_close  # noqa: E402
from tower.capture import (  # noqa: E402
    END_REASON_DISCONNECT as END_REASON_CAPTURE_DISCONNECT,
    END_REASON_BOUNDED_LIMIT as END_REASON_CAPTURE_BOUNDED,
    END_REASON_STOP as END_REASON_CAPTURE_STOP,
    CaptureFollower,
)
from tower.world_builder.backends import (  # noqa: E402
    BACKEND_AUTO,
    BACKEND_NAMES,
    select_backend,
)
from tower.world_builder.engine import WorldBuilderEngine  # noqa: E402
from tower.world_builder.intrinsics_store import IntrinsicsStore  # noqa: E402
from tower.world_builder.records import (  # noqa: E402
    FINAL_SOLVE_FAILED,
    FINAL_SOLVE_SKIPPED,
    FINAL_SOLVE_SOLVED,
    FINAL_SOLVE_UNAVAILABLE,
    FINALIZATION_COMPLETE,
    FINALIZATION_INTERRUPTED,
    STAGE_APPEARANCE,
    STAGE_DENSE,
    STAGE_STATE_FAILED,
    STAGE_STATE_OK,
    STAGE_STATE_RUNNING,
    STAGE_STATE_STOPPED,
    STAGE_STATE_UNAVAILABLE,
    STAGE_SURFACE,
    CameraIntrinsics,
    camera_intrinsics_from_json_dict,
)
from tower.world_builder.schema import (  # noqa: E402
    END_REASON_ERROR,
    END_REASON_INTERRUPTED,
    END_REASON_STOP,
)
from tower.process_ownership import (  # noqa: E402
    interpreter_environment,
    interpreter_executable,
)
from tower.world_builder.store import WorldStore  # noqa: E402

DEFAULT_ROOT = Path("data/world_builder")
TOWER_ROOT = Path(__file__).resolve().parents[1]
# Accepted keyframes between background global solves. At ~3.6 keyframes per
# second of walk this is roughly every 15 s; a solve over a two-minute walk
# costs ~40-80 s on this host (ledger E8/E9), so a longer walk simply gets
# fewer, larger solves rather than a queue.
DEFAULT_SOLVE_EVERY = 50
# How long Stop waits for a background solve before running the final one.
DEFAULT_SOLVE_WAIT_SECONDS = 120.0

logger = logging.getLogger("tower.world_build_session")


@dataclass(frozen=True)
class ObservedFrame:
    """One frame plus whatever the wire actually told us about it.

    `received_at` is None for a source that has no recorded timestamp --
    a directory of loose jpegs. None means unknown and the engine stamps
    its own receipt time; inventing one here would fabricate a clock
    (Rule 3).
    """

    payload: bytes
    source_seq: int
    wire_seq: int | None = None
    tx_seq: int | None = None
    received_at: float | None = None
    # The size the RECORDER measured off this frame after decoding it,
    # when the source knows it. Carried on the frame rather than looked
    # up from the capture again, because the whole failure this guards
    # against was asking a directory a question about a frame that had
    # not been written into it yet. None means the source did not say,
    # and `observed_size_of` decodes the bytes instead.
    width: int | None = None
    height: int | None = None
    # Where the raw frame lives on disk, when it lives anywhere. The global
    # solver reads raw frames in preference to the session's redacted
    # copies (global_solve.py, ledger E6), and only the process that
    # observed the frame knows the path.
    source_path: Path | None = None


def load_frames(directory: Path) -> list[ObservedFrame]:
    paths = sorted(directory.glob("*.jpg"))
    if not paths:
        raise SystemExit(f"no .jpg frames found under {directory}")
    return [
        ObservedFrame(
            payload=path.read_bytes(), source_seq=index, wire_seq=index,
            source_path=path,
        )
        for index, path in enumerate(paths)
    ]


def first_observed_frame(frames):
    """Take the first frame off a source, and hand back the whole run.

    This is where the live path now WAITS. `follow_capture` is a
    generator that polls, so `next` blocks until the recorder appends a
    line or gives up on the capture -- which is exactly the wait that
    makes the resolution knowable.

    Returns `(first, frames)` where `frames` still yields that first
    frame: nothing may be consumed for measurement and then dropped, or
    the session silently starts one frame into the walk. `(None, empty)`
    when the source ends without ever producing one, which is a real
    state -- a phone that connects and drops -- and must not hang.
    """
    iterator = iter(frames)
    first = next(iterator, None)
    if first is None:
        return None, iter(())
    return first, itertools.chain((first,), iterator)


def observed_size_of(frame: ObservedFrame) -> tuple[int, int] | None:
    """The size of one frame, preferring what the recorder measured.

    The recorder decodes every frame it stores and journals the resulting
    width and height, so on the live path the answer is already in hand
    and costs nothing. Falling back to decoding the payload covers a
    source that carries no metadata at all.

    Never guesses. None means "this frame did not say and could not be
    read", and the caller turns that into unknown intrinsics rather than
    into a default resolution.
    """
    if isinstance(frame.width, int) and isinstance(frame.height, int):
        return (frame.width, frame.height)
    try:
        from PIL import Image

        with Image.open(io.BytesIO(frame.payload)) as image:
            return image.size
    except Exception:  # noqa: BLE001 -- an undecodable frame is not fatal here
        return None


def observed_size_from_frames(directory: Path) -> tuple[int, int] | None:
    """The size of the first jpeg in a directory, read from its header."""
    paths = sorted(directory.glob("*.jpg"))
    if not paths:
        return None
    try:
        from PIL import Image

        with Image.open(paths[0]) as image:
            return image.size
    except Exception:  # noqa: BLE001 -- an undecodable frame is not fatal here
        return None


def resolve_intrinsics(store: IntrinsicsStore, observed_size, *, frame_source):
    """Look up a calibration for the size the frames MEASURE at.

    Never falls back to another resolution and never rescales: a
    calibration wrong by a crop factor produces a plausible trajectory
    that is wrong, which is the worst failure available. A miss returns
    `unknown()` and the run proceeds exactly as an uncalibrated Tower has
    always proceeded -- but it now says so, loudly, at the top of the log.
    """
    if observed_size is None:
        known = store.list_resolutions()
        logger.warning(
            "[Tower][WorldBuilder] could not observe the frame resolution of "
            "this %s source, so no calibration was looked up. Intrinsics stay "
            "unknown: expect the unposed backend, 0 poses and 0 points. "
            "Calibrations on file: %s",
            frame_source,
            ", ".join(f"{w}x{h}" for w, h in known) or "none",
        )
        return CameraIntrinsics.unknown()

    width, height = observed_size
    intrinsics = store.lookup(width, height)
    if intrinsics.is_known:
        return intrinsics

    # The store already logged where it looked. This line adds what the
    # operator can DO about it, and is the answer to the question the
    # 2026-08-24 walk could not answer: "why is there no geometry?"
    known = store.list_resolutions()
    logger.warning(
        "[Tower][WorldBuilder] NO CALIBRATION for the observed %sx%s frames "
        "(looked for %s). Intrinsics stay unknown, so the unposed backend "
        "runs and this session will produce 0 poses and 0 points. "
        "Calibrations on file: %s. Fix: see docs/CALIBRATION.md and run "
        "scripts/calibrate_charuco.py on board views captured at %sx%s.",
        width,
        height,
        store.path_for(width, height),
        ", ".join(f"{w}x{h}" for w, h in known) or "none",
        width,
        height,
    )
    return CameraIntrinsics.unknown()


def follow_capture(directory: Path, *, poll_seconds: float, max_idle_polls,
                   should_stop=None, handle: dict | None = None):
    """Yield frames from a capture directory as the Tower writes them.

    THE SPLIT BELOW IS THE WHOLE POINT, AND IT IS NOT STYLE.

    This used to be one generator function with the check as its first
    statement. A `def` containing `yield` is a generator function, so
    calling it runs NONE of the body -- the check did not execute until
    something advanced the generator for the first time. `main()` calls
    this at the frame-source step and does not advance it until after

        store  = WorldStore(args.root)
        engine = WorldBuilderEngine(store, ...)
        world_id = args.world or engine.create_world(args.name)

    so a session pointed at a capture directory that does not exist
    MINTED A WORLD and only then exited nonzero. The world stayed.

    That is the mechanism behind the empty worlds that accumulate with
    install age: 86 of the 123 worlds in the corpus on this host hold a
    `world.json` and no sessions. Every failed follow left one, and
    nothing ever collected them.

    Creating the world before the first frame is DELIBERATE and is
    preserved -- `main()` says why: "a Tower whose phone has connected
    but not yet sent a frame reports a world that exists and is empty
    rather than no world at all." That claim is about a session that can
    start. This function now refuses before `main()` reaches the store,
    so a session that cannot start leaves nothing behind.

    Validating in a plain function that RETURNS the generator is the
    standard way to make a generator's preconditions eager. The
    alternative -- moving the check up into `main()` -- would put the
    precondition somewhere other than the thing it is a precondition
    for, and the next caller would not get it.
    """
    if not directory.exists():
        raise SystemExit(f"no capture directory at {directory}")
    return _follow_capture(
        directory,
        poll_seconds=poll_seconds,
        max_idle_polls=max_idle_polls,
        should_stop=should_stop,
        handle=handle,
    )


def _follow_capture(directory: Path, *, poll_seconds: float, max_idle_polls,
                    should_stop=None, handle: dict | None = None):
    follower = CaptureFollower(directory, poll_seconds=poll_seconds)
    # The caller needs to ask, AFTER the loop, whether the capture it was
    # following had closed -- see `end_reason` in `main()`. The follower
    # retargets `_directory` onto a successor across a reconnect, so its
    # own `is_closed()` is the only answer that stays right; the directory
    # this generator was called with may be two captures old by then.
    if handle is not None:
        handle["follower"] = follower
    # `should_stop` is asked inside the poll loop, which is where this
    # process spends a quiet walk. A stop that arrived while the follower
    # slept is noticed at the next poll, not at the next frame.
    for frame in follower.follow(max_idle_polls=max_idle_polls, should_stop=should_stop):
        yield ObservedFrame(
            payload=frame.raw_bytes,
            source_seq=frame.source_seq,
            wire_seq=frame.wire_seq,
            tx_seq=frame.tx_seq,
            received_at=frame.received_at,
            width=frame.width,
            height=frame.height,
            # `follower.directory`, NOT the `directory` this generator was
            # called with. `relpath` is relative to the capture the frame
            # CAME FROM, and a reconnect retargets the follower onto a
            # successor -- the comment fifteen lines above says so about
            # `is_closed()` and this line was left reading the closure.
            #
            # Measured on the 2026-09-09 field walk, which reconnected once:
            # `sources.json` named capture 6a1b544c for all 643 keyframes,
            # and 523 of them were actually in dd885cca. Every one of those
            # 523 resolved to a path that does not exist, so `_source_frame`
            # fell back to the session's face-redacted copies and COLMAP
            # was fed those instead of the raw frames -- for 81% of the
            # walk. This is the mechanism behind what the handoff had
            # recorded as "sources.json is already 523/643 stale"; nothing
            # was stale, the ledger was wrong when it was written.
            #
            # It could have been worse than missing. The phone's source
            # index happened to run 1..953 in the first capture and
            # 1309..6109 in the second, so no path collided; had the
            # counter restarted at 1, the same bug would have handed
            # COLMAP a DIFFERENT REAL PHOTOGRAPH under the right name, and
            # nothing anywhere would have noticed.
            source_path=follower.directory / frame.relpath,
        )


def synthetic_frames(count: int, width: int, height: int):
    """Render a synthetic walk. Returns (jpegs, ground-truth intrinsics).

    Imports the test harness deliberately: it is the only renderer that
    exists, and duplicating it into production code to avoid a test import
    would be worse than the import.
    """
    from tests import synthetic_scene as ss

    camera_matrix = ss.camera_matrix(width, height)
    scene = ss.furnished_room()
    poses = ss.strafe(count, step=0.09)
    images = ss.render_sequence(scene, poses, camera_matrix, width, height)
    intrinsics = CameraIntrinsics(
        source="self_calibrated",
        model="pinhole",
        fx=float(camera_matrix[0, 0]),
        fy=float(camera_matrix[1, 1]),
        cx=float(camera_matrix[0, 2]),
        cy=float(camera_matrix[1, 2]),
        calibrated_width=width,
        calibrated_height=height,
    )
    frames = [
        ObservedFrame(payload=ss.encode_jpeg(image), source_seq=index, wire_seq=index)
        for index, image in enumerate(images)
    ]
    return frames, intrinsics


class StopRequest:
    """A stop asked for from outside this process, at one of two levels.

    THE 2026-09-06 PHYSICAL WALK IS WHY THIS EXISTS.

    Until then the builder could not be asked anything. The supervisor
    waited its grace and called `TerminateProcess`, which on Windows means
    no unwinding, no `finally`, no `atexit` -- and a builder that died
    between `observe()` and `stop_session()` left a lock naming a dead pid,
    a session record with `ended_at: null`, and a derived tree nobody
    could call anything but "failed". Twenty-eight sessions on the
    Windows box ended that way before the walk that made it visible.

    TWO LEVELS, BECAUSE TWO DIFFERENT THINGS ARE BEING ASKED.

    * **Soft** -- the parent closed this process's stdin. It means "you are
      no longer wanted for NEW frames": the wearer left World Builder, or
      the cartridge was stopped. A builder still observing stops
      observing, closes the session as `interrupted`, skips the final
      solve (nobody is waiting for it) and writes its final build. A
      builder already finalizing carries on: finalization is bounded and
      holds no camera.
    * **Hard** -- `SIGBREAK` / `SIGTERM` / `SIGINT`. It means "wrap up
      now": the Tower is shutting down. Any solve child is terminated,
      the session is closed if it is still open, the final build is
      written from what exists, and the process exits. Bounded by one
      build.

    Both are FLAGS set from a handler or a daemon thread and acted on by
    the main thread, for the reason `object_memory_session._StopRequest`
    gives: a handler that took the store's lock or joined a child is how a
    shutdown deadlocks.
    """

    SOFT = "soft"
    HARD = "hard"

    def __init__(self) -> None:
        self.level: str | None = None
        self.source: str | None = None
        self._lock = threading.Lock()

    def install(self, *, watch_stdin: bool = False) -> None:
        if watch_stdin:
            self._watch_stdin()
        for name in ("SIGTERM", "SIGINT", "SIGBREAK"):
            handler_signal = getattr(signal, name, None)
            if handler_signal is None:
                continue
            try:
                signal.signal(handler_signal, self._handle_signal)
            except (ValueError, OSError):
                # Not the main thread, or a signal this platform will not
                # let a process take. The process can still be terminated;
                # it just cannot be asked on that channel.
                continue

    @property
    def asked(self) -> bool:
        return self.level is not None

    @property
    def hard(self) -> bool:
        return self.level == self.HARD

    def asked_for(self) -> bool:
        """A callable for `CaptureFollower.follow(should_stop=...)`."""
        return self.level is not None

    def hard_asked_for(self) -> bool:
        return self.level == self.HARD

    def request(self, level: str, source: str) -> None:
        with self._lock:
            # A hard request outranks a soft one; a soft one never lowers
            # a hard one already recorded.
            if self.level == self.HARD:
                return
            self.level = level
            self.source = source

    def _handle_signal(self, signum, _frame) -> None:
        try:
            name = signal.Signals(signum).name
        except ValueError:
            name = str(signum)
        self.request(self.HARD, name)

    def _watch_stdin(self) -> None:
        """Soft stop when the parent closes the pipe it holds.

        Nothing is ever written to this pipe, the request IS the close, and
        a pipe needs no console -- which is why it is the channel that works
        under a pseudoconsole, a service, or a job object.

        NOT A PARKED READ. This used to be a daemon thread blocked in
        `os.read(0, 1)` for the whole run, and on Windows that read is half
        of a deterministic deadlock with any library whose load-time code
        touches descriptor 0 -- numpy's and scipy's OpenBLAS, pycolmap's
        libgfortran. It is what held world 2f447162 for 95 minutes. See
        `tower/stdin_stop.py` for the two native paths, measured.
        """
        watch_stdin_close(
            lambda: self.request(self.SOFT, "stdin-closed"),
            name="world-builder-stop-watch",
        )

    def bounded(self, frames):
        """`frames`, ending at the next frame after a stop was asked for.

        The follower's poll loop is the primary check on the live path;
        this is the only one a `--frames` replay has, and it is what keeps
        a live stop from being missed by the one frame the follower had
        already yielded.
        """
        for frame in frames:
            if self.asked:
                return
            yield frame


# How long a solve child gets after `terminate()` before it is killed, and
# how often a waiting builder looks at its stop request.
CHILD_TERMINATE_TIMEOUT_S = 5.0
CHILD_POLL_S = 0.25


def python_executable() -> str:
    """The interpreter a solve child runs under.

    `tower.process_ownership` decides: on a Windows venv `sys.executable`
    is a launcher that spawns the real interpreter underneath it with
    silent breakaway, so a child started that way is a PAIR of processes
    and `_terminate_process_tree` reaches only the outer one. The recipe
    gives one process, provided `child_environment()` travels with it.
    """
    return interpreter_executable()


def child_environment() -> dict:
    """The other half of `python_executable()`: what makes it venv-aware."""
    return interpreter_environment()


# `scripts/world_surface.py` exits with this when the stage cannot run on
# this machine at all, as opposed to cannot run on this session yet.
SURFACE_EXIT_CANNOT_RUN_HERE = 4


class BackgroundSurface:
    """The live surface reconstruction, as a CHILD this builder owns.

    Why it is driven by the SOLVE and not by keyframes. During a walk the only
    geometry worth reconstructing from is the global solution: the per-segment
    chain fragments -- 34 segments on a measured walk, the largest holding 13.6%
    -- and a field built from it would place the same wall in several places at
    several scales. GLOMAP puts nearly every keyframe in one component. So a
    live surface job is launched when a background solve LANDS, and at no other
    time. Between solves the surface is honestly unchanged rather than
    dishonestly growing.

    Why a child process rather than a thread. The builder observes frames on one
    thread and a GPU job in that thread would stall ingestion. The child also
    gives a stop something to kill: `close()` from the builder's `finally`
    leaves nothing behind, which is the lesson `BackgroundSolver` was written
    for after an orphaned solve ran on for 32 s past its parent.

    One at a time. NOT held back while a solve runs: an earlier version was,
    on the theory that the two share the GPU, and on a real-time replay of the
    canonical walk it built ONE surface in 140 s -- every time a solve landed
    the next solve was already due and launched first, so the surface never
    got its turn. The theory was also wrong on this machine: pycolmap has no
    CUDA build on Windows, so the solve is CPU work and the surface is GPU
    work. What they do share is CPU, so the surface child runs at below-normal
    priority and the solve, and frame ingestion, win it.

    A solve that lands while a surface is still building is not dropped: it
    marks the surface stale, and the next `poll` launches a rebuild against the
    newest solve as soon as the running one finishes.
    """

    def __init__(self, *, root: Path, world_id: str, session_id: str,
                 script: Path | None = None, spawn=None, appearance: bool = False):
        self.root = root
        # Build the appearance on each live surface, in the same child.
        self.appearance = appearance
        self.world_id = world_id
        self.session_id = session_id
        self.script = Path(script) if script is not None             else TOWER_ROOT / "scripts" / "world_surface.py"
        self._spawn = spawn if spawn is not None else subprocess.Popen
        self._child = None
        self._launches = 0
        self._log = None
        self._stale = False
        self._built_from = None
        self._pending_stamp = None
        self._disabled = False

    @property
    def running(self) -> bool:
        return self._child is not None and self._child.poll() is None

    @property
    def launches(self) -> int:
        return self._launches

    def solve_landed(self, store: WorldStore) -> bool:
        """A solve finished. If it published a NEW solution, build against it
        now, or as soon as the surface already building is done.

        A finished solve is not necessarily a new solution: a solve child that
        crashed or found nothing to add still "finishes". Rebuilding the surface
        for it costs a GPU minute and changes nothing, so the solution file's
        own stamp is compared with the one last built from.
        """
        stamp = self._solution_stamp()
        if stamp is not None and stamp == self._built_from:
            return False
        self._pending_stamp = stamp
        self._stale = True
        return self.poll(store)

    def _solution_stamp(self):
        path = (self.root / "worlds" / self.world_id / "solve" / self.session_id
                / "solution.json")
        try:
            st = path.stat()
        except OSError:
            return None
        return (st.st_mtime_ns, st.st_size)

    def poll(self, store: WorldStore) -> bool:
        """Launch the pending rebuild if one is owed and nothing is running."""
        self._reap()
        if not self._stale or self.running or self._disabled:
            return False
        self._stale = False
        launched = self.maybe_launch(store)
        if launched:
            self._built_from = self._pending_stamp
        return launched

    @property
    def child_pid(self) -> int | None:
        return self._child.pid if self._child is not None else None

    def _reap(self) -> None:
        if self._child is not None and self._child.poll() is not None:
            if self._child.returncode == SURFACE_EXIT_CANNOT_RUN_HERE:
                # The depth network is not installed on this machine. Every
                # later solve would launch a child that fails the same way;
                # say so once and stop.
                self._disabled = True
                logger.warning(
                    "[Tower][WorldBuilder] live surface cannot run on this "
                    "machine; no further live surfaces this walk")
            self._child = None
            if self._log is not None:
                self._log.close()
                self._log = None

    def maybe_launch(self, store: WorldStore, *, solver_running: bool = False) -> bool:
        # `solver_running` is accepted and ignored; see the class docstring for
        # why the surface no longer waits for the solve.
        self._reap()
        if self.running:
            return False
        log_dir = self.root / "worlds" / self.world_id / "surface" / self.session_id
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            self._log = open(log_dir / "surface.log", "ab")
        except OSError:
            self._log = None
        argv = [
            python_executable(), str(self.script),
            "--root", str(self.root), "--world", self.world_id,
            "--session", self.session_id, "--live", "--force",
            *(["--appearance"] if self.appearance else []),
        ]
        extra = {}
        if os.name == "nt":
            # The solve and frame ingestion are CPU work and must win it.
            extra["creationflags"] = subprocess.BELOW_NORMAL_PRIORITY_CLASS
        self._child = self._spawn(
            argv, cwd=str(TOWER_ROOT), stdout=self._log or subprocess.DEVNULL,
            stderr=subprocess.STDOUT, env=child_environment(),
            stdin=subprocess.DEVNULL, **extra,
        )
        self._launches += 1
        logger.info(
            "[Tower][WorldBuilder] live surface %s launched (pid %s)",
            self._launches, self._child.pid,
        )
        return True

    def close(self) -> None:
        """Terminate rather than abandon. A live surface is disposable: the
        final one replaces it, so there is nothing to wait for at Stop.

        `_terminate_process_tree` and not `Popen.terminate`, because the child
        is a python launcher that spawns the real interpreter: killing only the
        launcher leaves the GPU job running and the lock file held.
        """
        self._reap()
        if self._child is None:
            return
        try:
            _terminate_process_tree(self._child)
        except Exception:  # noqa: BLE001 -- teardown must not mask the real exit
            logger.exception("[Tower][WorldBuilder] could not stop the live surface child")
        self._child = None
        if self._log is not None:
            self._log.close()
            self._log = None


class BackgroundSolver:
    """Every global solve this builder runs, as a CHILD it owns.

    One child at a time. `maybe_launch` starts `scripts/world_solve.py`
    in the background when no solve is running and at least `every`
    keyframes have been accepted since the last launch; `finished()`
    reports (once) that a launch has completed, which is the builder's cue
    to rebuild so the new solution reaches the derived tree without
    waiting for the next rebuild interval. `run_final` runs the
    finalization solve the same way, so that a stop request can end it.

    OWNERSHIP IS THE POINT OF THE CLASS. On the 2026-09-06 walk the
    builder died with a background solve in flight; the child ran on for
    32 s as an orphan and wrote a solution nothing ever merged. Every
    child is now tracked, `wait()` terminates rather than abandons, and
    `close()` -- called from the builder's `finally` -- leaves nothing
    behind. Output goes to `solve/<session>/solve.log`.
    """

    def __init__(self, *, root: Path, world_id: str, session_id: str, every: int,
                 capture_dirs, threads: int | None = None, script: Path | None = None,
                 spawn=None):
        self.root = root
        self.world_id = world_id
        self.session_id = session_id
        self.every = max(1, every)
        self.capture_dirs = [Path(d) for d in capture_dirs]
        self.threads = threads
        self.script = Path(script) if script is not None else TOWER_ROOT / "scripts" / "world_solve.py"
        self._spawn = spawn if spawn is not None else subprocess.Popen
        self._child = None
        self._launched_at_keyframes = 0
        self._launches = 0
        self._completed_unseen = False
        self._log = None

    @property
    def running(self) -> bool:
        return self._child is not None and self._child.poll() is None

    @property
    def launches(self) -> int:
        return self._launches

    @property
    def child_pid(self) -> int | None:
        return self._child.pid if self._child is not None else None

    def _reap(self) -> None:
        if self._child is not None and self._child.poll() is not None:
            self._child = None
            self._completed_unseen = True
            if self._log is not None:
                self._log.close()
                self._log = None

    def finished(self) -> bool:
        """True once per completed launch."""
        self._reap()
        if self._completed_unseen:
            self._completed_unseen = False
            return True
        return False

    def _argv(self, *, final: bool) -> list[str]:
        argv = [
            python_executable(), str(self.script),
            "--root", str(self.root), "--world", self.world_id, "--session", self.session_id,
        ]
        for capture_dir in self.capture_dirs:
            argv += ["--capture-dir", str(capture_dir)]
        # LOOP DETECTION ON EVERY SOLVE, not only the final one.
        #
        # Sequential matching reaches 20 keyframes either side and no
        # further, so a live solve can only ever chain forwards: it cannot
        # discover that the wearer has walked back into a room it already
        # mapped. The consequence is not a slightly worse world, it is a
        # world that comes APART as the walk goes on. Measured on the
        # 2026-09-09 capture, sequential only, at the field run's own solve
        # horizons: 6 components at 156 keyframes, 11 at 311, 14 at 526,
        # 16 at 646, with the largest holding 24% of posed keyframes.
        #
        # The same capture re-solved with loop detection at the same
        # horizons, in one workspace, the way a live session accumulates:
        #
        #     horizon   components   largest component's share
        #        156        3              0.75
        #        311        6              0.77
        #        526        5              0.90
        #        646        6              0.91
        #        795        5              0.95
        #
        # It CONVERGES instead of fragmenting, which is the whole product
        # requirement: geometry that becomes more recognisable while the
        # wearer walks, not less.
        #
        # The cost is 1.2-1.9x the sequential solve -- 50.8 s against
        # 42.6 s at 646 keyframes, 17.1 s against 9.1 s at 156 -- and it is
        # paid in matching, which is incremental: pairs already tested stay
        # in the database, so each solve only matches what is new. That is
        # far cheaper than it looks next to a from-scratch mapping stage.
        #
        # `--final` still differs, and still matters: it is the one solve
        # that sees every keyframe including the tail no live solve reached.
        argv += ["--loop-detection"]
        if final:
            argv += ["--final"]
        if self.threads is not None:
            argv += ["--threads", str(self.threads)]
        return argv

    def maybe_launch(self, store: WorldStore, accepted: int, sources: dict) -> bool:
        self._reap()
        if self.running or accepted - self._launched_at_keyframes < self.every or accepted < 2:
            return False
        from tower.world_builder import global_solve  # noqa: PLC0415

        global_solve.write_sources(store, self.world_id, self.session_id, sources)
        workspace = global_solve.workspace_for(store, self.world_id, self.session_id)
        workspace.root.mkdir(parents=True, exist_ok=True)
        self._log = open(workspace.root / "solve.log", "ab")
        self._child = self._spawn(
            self._argv(final=False), cwd=str(TOWER_ROOT), stdout=self._log,
            stderr=subprocess.STDOUT, env=child_environment(), stdin=subprocess.DEVNULL,
        )
        self._launched_at_keyframes = accepted
        self._launches += 1
        logger.info(
            "[Tower][WorldBuilder] background solve %s launched at %s keyframes (pid %s)",
            self._launches, accepted, self._child.pid,
        )
        return True

    def wait(self, timeout: float | None, should_stop=None) -> bool:
        """Wait for a running child. False if it had to be terminated.

        A child that outlives `timeout` is TERMINATED, not abandoned: the
        final solve that follows uses the same workspace (one feature
        database, one `sparse/` tree it deletes first), and an abandoned
        child still mapping into it is a corruption waiting to happen.
        `should_stop` ends the wait early the same way.
        """
        if self._child is None:
            return True
        deadline = None if timeout is None else time.monotonic() + timeout
        try:
            while self._child.poll() is None:
                if should_stop is not None and should_stop():
                    logger.warning(
                        "[Tower][WorldBuilder] background solve pid %s terminated: "
                        "a stop was requested", self._child.pid,
                    )
                    self._terminate_child()
                    return False
                if deadline is not None and time.monotonic() >= deadline:
                    logger.warning(
                        "[Tower][WorldBuilder] background solve pid %s still running after "
                        "%ss; terminating it so the final solve owns the workspace",
                        self._child.pid, timeout,
                    )
                    self._terminate_child()
                    return False
                time.sleep(CHILD_POLL_S)
            return True
        finally:
            self._reap()

    def run_final(self, store: WorldStore, sources: dict, *, should_stop=None) -> dict:
        """The finalization solve, as a child, until it ends or a stop arrives.

        Returns the child's own summary (`solved`, `solver`, `components`,
        `timing`, ...) with `attempted: True`, or `{"attempted": True,
        "solved": False, "interrupted": True, ...}` when `should_stop`
        ended it. Never raises: a walk that reconstructed locally is worth
        keeping even if the global solve failed.
        """
        from tower.world_builder import global_solve  # noqa: PLC0415

        started = time.perf_counter()
        try:
            global_solve.write_sources(store, self.world_id, self.session_id, sources)
            workspace = global_solve.workspace_for(store, self.world_id, self.session_id)
            workspace.root.mkdir(parents=True, exist_ok=True)
            self._log = open(workspace.root / "solve.log", "ab")
            # `stdin=DEVNULL` ON BOTH SPAWNS, AND IT IS NOT HYGIENE. This
            # process's stdin is the supervisor's stop pipe, and a daemon
            # thread sits in a blocking ReadFile on it for the whole session
            # (see StopRequest). A child that inherits that handle inherits
            # a synchronous file object with an I/O in flight, and Windows
            # serialises every operation on such an object -- so the child's
            # own startup, which queries its fd 0, blocks until the pipe
            # closes. Measured: a final solve child sat at 0.02 s of CPU for
            # 90 s and started the instant stdin was closed.
            self._child = self._spawn(
                self._argv(final=True), cwd=str(TOWER_ROOT), stdout=subprocess.PIPE,
                stderr=self._log, env=child_environment(), stdin=subprocess.DEVNULL,
            )
        except Exception as error:  # noqa: BLE001 -- see the docstring
            logger.warning(
                "[Tower][WorldBuilder] final global solve could not start for %s: %s",
                self.session_id, error,
            )
            return {"attempted": True, "solved": False, "error": f"{type(error).__name__}: {error}"}
        logger.info(
            "[Tower][WorldBuilder] final global solve launched (pid %s)", self._child.pid
        )
        child = self._child
        # Drain stdout on a thread: the summary is small, but a pipe nobody
        # reads is a child that blocks on its last print and never exits.
        chunks: list[bytes] = []

        def drain() -> None:
            try:
                chunks.append(child.stdout.read())
            except Exception:  # noqa: BLE001
                pass

        reader = threading.Thread(target=drain, name="world-builder-final-solve-stdout", daemon=True)
        reader.start()
        interrupted = False
        try:
            while child.poll() is None:
                if should_stop is not None and should_stop():
                    interrupted = True
                    logger.warning(
                        "[Tower][WorldBuilder] final global solve pid %s terminated: a hard "
                        "stop was requested; the last background solution stands", child.pid,
                    )
                    self._terminate_child()
                    break
                time.sleep(CHILD_POLL_S)
        finally:
            reader.join(timeout=2.0)
            self._reap()
        # A hard stop that reached the child FIRST (a console control event
        # goes to the whole process group) ends it before this loop sees
        # the flag; the outcome is still "interrupted by the stop", not "the
        # solver crashed".
        if should_stop is not None and should_stop():
            interrupted = True
        elapsed = round(time.perf_counter() - started, 3)
        if interrupted:
            return {"attempted": True, "solved": False, "interrupted": True, "seconds": elapsed}
        summary: dict = {}
        text = b"".join(chunks).decode("utf-8", errors="replace").strip()
        if text:
            try:
                summary = json.loads(text)
            except ValueError:
                summary = {"solved": False, "error": f"unreadable summary: {text[-200:]}"}
        if child.returncode not in (0, None) and not summary.get("solved"):
            summary.setdefault("error", f"world_solve.py exited {child.returncode}")
        summary["attempted"] = True
        summary["seconds"] = elapsed
        logger.info(
            "[Tower][WorldBuilder] final global solve: solved=%s solver=%s posed=%s/%s "
            "components=%s in %.2fs",
            summary.get("solved"), summary.get("solver"), summary.get("keyframes_posed"),
            summary.get("keyframes"), len(summary.get("components") or []), elapsed,
        )
        return summary

    def close(self) -> None:
        """Leave no child behind. Called from the builder's `finally`."""
        if self._child is not None and self._child.poll() is None:
            logger.warning(
                "[Tower][WorldBuilder] solve child pid %s still running at exit; terminating",
                self._child.pid,
            )
            self._terminate_child()
        self._reap()
        if self._log is not None:
            self._log.close()
            self._log = None

    def _terminate_child(self) -> None:
        child = self._child
        if child is None:
            return
        _terminate_process_tree(child)


def _terminate_process_tree(process) -> None:
    """Terminate a child and everything it spawned, then make sure.

    Children first, because on Windows a venv interpreter may itself be a
    launcher with the real interpreter underneath it, and terminating
    only the handle we hold would orphan exactly the process doing the
    work.
    """
    try:
        import psutil

        try:
            descendants = psutil.Process(process.pid).children(recursive=True)
        except psutil.Error:
            descendants = []
    except Exception:  # noqa: BLE001 -- psutil is a hard dependency; be safe anyway
        descendants = []
    for proc in descendants:
        try:
            proc.terminate()
        except Exception:  # noqa: BLE001
            pass
    try:
        process.terminate()
    except Exception:  # noqa: BLE001
        pass
    try:
        process.wait(timeout=CHILD_TERMINATE_TIMEOUT_S)
    except Exception:  # noqa: BLE001
        try:
            process.kill()
            process.wait(timeout=CHILD_TERMINATE_TIMEOUT_S)
        except Exception:  # noqa: BLE001
            pass
    for proc in descendants:
        try:
            if proc.is_running():
                proc.kill()
        except Exception:  # noqa: BLE001
            pass


def solve_session(store: WorldStore, world_id: str, session_id: str, *, capture_dirs,
                  sources: dict, loop_detection: bool = True) -> dict:
    """The finalisation solve, in this process, after the last frame.

    Never raises: a walk that reconstructed locally is worth keeping even
    if the global solve failed, exactly as `register_session` reasons.
    """
    from tower.world_builder import global_solve  # noqa: PLC0415
    from tower.world_builder.store import compute_input_digest  # noqa: PLC0415

    started = time.perf_counter()
    try:
        global_solve.write_sources(store, world_id, session_id, sources)
        keyframes = store.read_keyframes(world_id, session_id)
        summary = global_solve.solve(
            store, world_id, session_id, capture_dirs=capture_dirs, final=True,
            num_threads=-1, loop_detection=loop_detection,
            input_digest=compute_input_digest(keyframes),
        )
    except Exception as error:  # noqa: BLE001 -- see the docstring
        logger.warning(
            "[Tower][WorldBuilder] final global solve failed for %s: %s", session_id, error
        )
        return {"attempted": True, "solved": False, "error": f"{type(error).__name__}: {error}"}
    summary["seconds"] = round(time.perf_counter() - started, 3)
    summary["attempted"] = True
    logger.info(
        "[Tower][WorldBuilder] final global solve: solved=%s solver=%s posed=%s/%s "
        "components=%s in %.2fs",
        summary.get("solved"), summary.get("solver"), summary.get("keyframes_posed"),
        summary.get("keyframes"), len(summary.get("components") or []), summary["seconds"],
    )
    return summary


# How many keyframes a rebuild is worth, at a given size of world.
#
# `--rebuild-every 4` is a fixed count and the rebuild is not a fixed cost:
# `write_derived` rewrites poses, points, support and the manifest IN FULL
# every time, so it grows with the world while the interval does not.
# Measured against derived trees at the field walk's own ratios (33.5
# points and 26.2 support rows per keyframe):
#
#     keyframes   tree size   write_derived
#           795      3.7 MB       0.34 s
#          2000      9.4 MB       0.86 s
#          4000     18.8 MB       1.85 s
#          6000     28.2 MB       3.48 s
#
# At the measured 3.2 keyframes/sec, four keyframes is 1.21 s of wall time.
# The write alone crosses that at about 2,700 keyframes -- roughly FOURTEEN
# MINUTES into a walk -- and from there the builder falls behind for the
# rest of the session, with the capture directory as the only queue. The
# stated target is twenty to thirty minutes.
#
# So the interval grows with the world, doubling each time the keyframe
# count doubles past the knee. That keeps the rebuild a bounded FRACTION of
# the wall clock instead of a growing one, rather than 27% -> 69% -> 148%
# -> 288%, which is what a fixed four gives and is why the builder fell
# behind for the back half of a long walk.
#
# EVERYTHING ABOVE IS THE ORIGINAL REASONING AND IT IS SOUND. THE NUMBERS
# IN IT ARE NOT: they cost `write_derived`, and the loop calls
# `engine.build()`. The knee was 750 and the schedule was 795->4,
# 1500->8, 3000->16, 6000->32. The block immediately below supersedes all
# of that -- read it, not this.
#
# The wearer loses nothing that matters. A rebuild is a redraw of a world
# that is already mostly settled by then, and the two triggers that carry
# real news are untouched: a completed background solve still forces a
# rebuild immediately, and the final build still runs at Stop.
# RE-ANCHORED ON WHAT THE LOOP ACTUALLY PAYS.
#
# The first version of this modelled `write_derived` -- 0.342 s at 795
# keyframes -- and set the knee at 750 so that "nothing about the
# 2026-09-09 walk moves". The loop below does not call `write_derived`; it
# calls `engine.build()`, which is that write plus the merge, the placement
# pass and the manifest. Measured over the 204 rebuilds of a replay of the
# real field capture, against that walk's own arrival rate of 3.23
# keyframes/second:
#
#     keyframes     mean build     share of wall clock at interval 4
#       1- 200        0.141 s              11.4%
#     201- 400        0.386 s              31.1%
#     401- 600        0.585 s              47.2%
#     601- 800        1.006 s              81.2%
#     801-1000        0.895 s              72.2%
#
# 81%, where the model said 27%. So the knee was in the wrong place AND
# arrived one doubling late: `(accepted // 750).bit_length() - 1` is zero
# for everything below 1500, which is about eight minutes -- the interval
# did not widen until long after the builder had stopped keeping up.
#
# 600 and no `- 1`, so the first doubling lands where the measured share
# crosses a half: 8 from 601, 16 from 1200, 32 from 2400, 64 from 4800,
# and capped there. `min(doublings, 4)` still caps at `4 << 4` = 64; it
# now engages at 9,600 rather than 12,000, which is past where 64 is
# first reached either way. At 3.23 keyframes/second that is a live refresh every
# 2.5 s at the start of the widening and every 20 s at the cap, against a
# builder that otherwise falls permanently behind the camera with the
# capture directory as its only queue.
REBUILD_KNEE_KEYFRAMES = 600


def rebuild_interval(base: int, accepted: int) -> int:
    """The rebuild interval for a world of `accepted` keyframes."""
    if accepted <= REBUILD_KNEE_KEYFRAMES:
        return base
    doublings = (accepted // REBUILD_KNEE_KEYFRAMES).bit_length()
    return base << min(doublings, 4)


def session_manifest(store, world_id: str, session_id: str) -> dict:
    """The manifest that describes THIS session, from either copy.

    THE READER WAS FIXED AND THE WRITERS WERE NOT. `usable_placements`
    judges a placement by the session's own manifest; the two places that
    STAMP a placement's `input_digest` still read the world's, which names
    whichever session built last. In the live flow they are the same file's
    contents, so nothing showed -- but `world_registration.py --write
    --session <older>` on a world walked twice stamps the newer session's
    digest, and then every one of those placements is refused by the reader
    for disagreeing. A reviewer found it by asking what else read the
    world-level copy.

    The session's own first, then the world's but only if it names this
    session; a manifest about another session is not evidence about this
    one.
    """
    manifest = store.read_session_manifest(world_id, session_id)
    if isinstance(manifest, dict) and manifest.get("session_id") == session_id:
        return manifest
    world = store.read_derived_manifest(world_id)
    if isinstance(world, dict) and world.get("session_id") == session_id:
        return world
    return {}


def should_register(result) -> bool:
    """Whether the Sim3 registrar should run after this build.

    A FUNCTION, not an expression at the call site, because the thing it
    decides has already been got wrong once and the wrong version was
    untestable. An adversarial review pointed out that the first fix left
    the decision inline in `main()`, where the only test that could reach it
    re-typed the condition into the test file and asserted the copy -- a
    tautology that would have passed against any implementation at all.

    The rule: the registrar answers "where do these fragments sit relative
    to each other", pairwise and weakly. When a global solve has already
    answered it from one reconstruction, a second weaker answer must not
    overwrite the first. Otherwise the registrar is the only producer there
    is, and it must run.

    `placements_source` is set by `engine.build()`, which is the only code
    that knows whether the build it just did wrote placements from a
    solution. Asking anything else has been tried: the guard used to ask
    whether the FINAL solve had succeeded, which is a different question,
    and on the 2026-09-09 walk the answer to it was "no" while the answer to
    this one was "yes, 72 segments across 14 components". The registrar ran
    and replaced them with 120 refusals.
    """
    return (getattr(result, "diagnostics", None) or {}).get("placements_source") is None


def register_session(store: WorldStore, world_id: str, session_id: str) -> dict:
    """Place what can be placed, and say so. Never raises.

    Registration is the step that turns a bag of independently
    reconstructed fragments into a world. It was implemented, tested,
    persisted and served long before anything called it, so every walk
    up to now finalised with `placements.json` absent and iOS drew every
    segment as its own disconnected island -- not because the pairs were
    refused, but because the question was never asked. On the 2026-08-29
    drawer walk asking it places 5 of 36 segments and 4,704 of 13,050
    points; the recorded session shipped 0 of both.

    Three properties make this safe to run automatically:

    - It is NON-DESTRUCTIVE. `register()` writes nothing; poses, points
      and support are untouched, and a segment's own geometry never
      moves. Only `transform_to_world` is added, in a separate file.
    - It is REFUSAL-BY-DEFAULT. A pair is admitted only on two
      independent solves that agree; an unplaced segment is served
      exactly as it is served today.
    - It is DIGEST-BOUND. The placement records the build it was solved
      against, and the serving layer refuses any placement whose digest
      does not match, so a later rebuild cannot resurrect a stale
      transform.

    It runs HERE -- in the builder subprocess, after the last build --
    and not in the web process, for the same reason `build()` does: this
    is seconds of work, and the frame path must never pay for it. It is
    also why failure is swallowed. A world that reconstructed is worth
    keeping even if it could not be placed, so a registration that
    raises is reported and does not take the session down with it.
    """
    from scripts.world_registration import (  # noqa: PLC0415
        SupportMissingError,
        placements_from_report,
        register,
    )

    started = time.perf_counter()
    try:
        # NEVER OVERWRITE A GLOBAL SOLVE'S PLACEMENTS. The caller's guard is
        # the first line of defence and this is the second, because the first
        # one can be told the wrong thing.
        #
        # `load_solution` absorbs any unreadable solution and returns None --
        # the right answer for a torn archive, and also the answer it gives
        # if numpy changes an exception type, a schema drifts, or the box
        # runs out of memory. In every one of those `engine.build()` writes
        # no placements, reports `placements_source: None`, and
        # `should_register` concludes there is no global solve to defer to.
        # It would then call this function, which used to write
        # unconditionally -- destroying exactly the placements the fix was
        # written to protect, from a cause whose only symptom is one
        # `logger.warning`. An adversarial review found that path.
        #
        # A placement set is trusted here only if it is REGISTERED and
        # CURRENT: the serving layer already drops any placement whose
        # `input_digest` disagrees with the manifest, so a stale set is not
        # worth preserving and re-registering it is the correct outcome.
        #
        # INSIDE the try, and that is not tidiness. This function promises
        # in its own docstring never to raise, and the first version of this
        # check read the store above the guard -- which `_Boom`, the test
        # store whose every read fails, turned straight back into the
        # session-ending exception the guard exists to prevent.
        existing = store.read_placements(world_id, session_id) or []
        manifest_now = session_manifest(store, world_id, session_id)
        digest_now = manifest_now.get("input_digest")
        current_registered = [
            p for p in existing
            if p.state == "registered" and p.input_digest == digest_now
        ]
        if current_registered:
            logger.info(
                "[Tower][WorldBuilder] registration stood down for session %s: %s "
                "current registered placements already exist",
                session_id, len(current_registered),
            )
            return {
                "attempted": False,
                "wrote_placements": False,
                "reason": (
                    f"{len(current_registered)} current registered placements "
                    "already exist and were not replaced"
                ),
            }
        report = register(store, world_id, session_id)
        # Inside the guard, not after it. Persisting is not the safe part
        # of this: `placements_from_report` runs every placement through
        # `SegmentPlacement.__post_init__`, which raises ValueError on a
        # NaN scale or a non-unit quaternion -- precisely what a
        # degenerate Sim3 produces, and precisely the failure this guard
        # exists for. Measured with these three lines outside the try: a
        # raising `write_placements` gave exit code 1 and zero bytes of
        # report, losing a walk that had reconstructed perfectly well.
        manifest = session_manifest(store, world_id, session_id)
        placements = placements_from_report(
            report, input_digest=manifest.get("input_digest")
        )
        store.write_placements(world_id, session_id, placements)
    except SupportMissingError as error:
        return {"attempted": True, "wrote_placements": False, "refusal": str(error)}
    except Exception as error:  # noqa: BLE001 -- see the docstring
        logger.warning(
            "[Tower][WorldBuilder] registration failed for session %s: %s",
            session_id,
            error,
        )
        return {
            "attempted": True,
            "wrote_placements": False,
            "error": f"{type(error).__name__}: {error}",
        }

    elapsed = time.perf_counter() - started

    logger.info(
        "[Tower][WorldBuilder] registration: %s of %s segments placed, "
        "%s of %s points, %s admitted pairs of %s candidates, in %.2fs",
        report["segments_registered"],
        report["segments_with_geometry"],
        report["points_registered"],
        report["points_total"],
        len(report["admitted_pairs"]),
        report["candidate_pairs"],
        elapsed,
    )
    return {
        "attempted": True,
        "wrote_placements": True,
        "reference_segment": report["reference_segment"],
        "segments_registered": report["segments_registered"],
        "segments_with_geometry": report["segments_with_geometry"],
        "points_registered": report["points_registered"],
        "points_total": report["points_total"],
        "candidate_pairs": report["candidate_pairs"],
        "admitted_pairs": report["admitted_pairs"],
        "cycles_checked": report["cycles_checked"],
        "cycle_refusal": report["cycle_refusal"],
        "seconds": round(elapsed, 3),
    }


def _terminal_detail(result):
    """A stage result's `detail`, but only when it explains a non-`ok` state.

    The surface pipeline puts its whole report in `detail` on success. That
    belongs in the stage's own manifest, where it already is, and not on the
    session record that the world listing reads for every session of every
    world.
    """
    return None if getattr(result, "state", None) == STAGE_STATE_OK else result.detail


def _record_raise(record, stage: str) -> None:
    """Name the exception currently being handled on the session record.

    `sys.exc_info()` rather than a bound `as exc`, so the call sites stay
    `except BaseException:` -- there is nothing to do with the exception here
    except write down what it was and let it keep going.
    """
    exc = sys.exc_info()[1]
    record(stage, state=STAGE_STATE_FAILED, detail=f"{type(exc).__name__}: {exc}")


def _solution_gated(store, world_id: str, session_id: str) -> bool:
    """Whether the PUBLISHED solve ran the evidence gate (its `solution.json` carries a
    `gate` record): then its depth stage was told the camera's FoV, and the surface asks
    for the same to reuse it. Keyed on the solve, not the environment (review V7, L-a): a
    world solved before the gate keeps today's surface whatever the setting is now.
    Never raises; unreadable is "not gated", today's surface."""
    try:
        from tower.storage import read_json_closed  # noqa: PLC0415

        path = store.world_dir(world_id) / "solve" / session_id / "solution.json"
        meta = read_json_closed(path) if path.exists() else None
        gate = (meta or {}).get("gate")
        return isinstance(gate, dict) and gate.get("state") == "applied"
    except Exception:  # noqa: BLE001
        return False


def _gate_setting() -> bool:
    """`TOWER_WORLD_SOLVE_GATE` (off). Never raises: a malformed environment is
    the gate off, which is today's surface."""
    try:
        from tower.config import world_solve_gate_setting  # noqa: PLC0415

        return bool(world_solve_gate_setting())
    except Exception:  # noqa: BLE001
        return False


def final_surface_stages(store: WorldStore, world_id: str, session_id: str, *,
                         solved: bool, appearance: bool, prune_depth_work: bool,
                         should_stop, stop_source=lambda: None,
                         record=None) -> dict:
    """The finished world's surface stages, after Stop, in THIS process.

    Returns the report entries (`surface`, and `appearance` when asked). In order,
    each stage skipped rather than truncated on a hard stop (`should_stop`):

    1. `surfacify(force=True)` with the FINAL parameters (`SurfaceParams()`):
       depth, then the transient detector masks under **union** (Grounding DINO +
       SAM 2 on top of the OneFormer masks the live child already cached --
       the cache is per component, so OneFormer is not run twice), then the depth
       consistency field **warm-started** from the live child's field (the key
       differs: final outer iterations and a later solve), then fusion, the plane
       snap and the levels;
    2. the final appearance on that surface, with the final parameters (union
       masks, read from the same cache), only if the surface is `ok`;
    3. pruning the per-frame depth work, only after both, because the appearance
       reads each frame's fill mask and raw prediction from it.

    The live child (`BackgroundSurface`, `scripts/world_surface.py --live`) runs
    the same order with the live presets (OneFormer only, fewer outer iterations)
    and is terminated before this starts. Extracted from `main` so the order,
    the parameters and the stop behaviour are tested by running it.

    `record(stage, state=, detail=, attempted=)` -- the builder passes
    `engine.mark_stage` -- persists each outcome on the session record. It is
    optional and defaults to a no-op, so callers that only want the report
    are unchanged. It exists because the report this returns is printed and
    then discarded: by the time these stages run the record already says
    `finalization: complete`, and without this a `surfacify()` that raised
    left a world sparse forever with nothing on disk to say so. Every stage
    is marked `running` before it starts and terminal afterwards -- including
    when it RAISES, which is re-raised unchanged once recorded.
    """
    report: dict = {}
    record = record or (lambda *a, **kw: None)

    if not appearance:
        # NOT REQUESTED IS A FACT, written HERE rather than by the builder
        # after this returns, so the recovery finisher -- which calls this with
        # `appearance=False` on a Tower whose appearance is switched off --
        # writes it too. Without it an appearance left `stopped` by an earlier
        # Tower stayed `stopped` through every rebuilt surface: owed for ever,
        # rebuilt until the attempt bound retired a perfectly good surface.
        record(STAGE_APPEARANCE, state=STAGE_STATE_UNAVAILABLE, attempted=False,
               detail="not requested (--appearance was not passed)")

    def _skip(reason: str, state: str) -> dict:
        """Neither stage ran. Both say why, rather than the appearance being
        absent and indistinguishable from a Tower that never recorded it."""
        record(STAGE_SURFACE, state=state, attempted=False, detail=reason)
        if appearance:
            record(STAGE_APPEARANCE, state=state, attempted=False, detail=reason)
        return {"surface": {"attempted": False, "reason": reason}}

    if should_stop():
        return _skip(f"hard stop ({stop_source()}) during finalization",
                     STAGE_STATE_STOPPED)
    if not solved:
        return _skip("a surface needs a global solve; there is none",
                     STAGE_STATE_UNAVAILABLE)
    from tower.world_builder.surface import SurfaceParams  # noqa: PLC0415
    from tower.world_builder.surface_pipeline import surfacify  # noqa: PLC0415

    # `running` BEFORE the call, terminal after it, on every path including
    # the one that raises. The Job Object kills this tree on a thirty-second
    # grace and a six-minute surface will not get to say anything afterwards;
    # a stage left saying `running` by a pid that is gone is the truth.
    record(STAGE_SURFACE, state=STAGE_STATE_RUNNING)
    try:
        surface_result = surfacify(
            store, world_id, session_id,
            # The final preset, named rather than defaulted, so the union detector
            # and the final consistency iterations are what this line says.
            params=SurfaceParams(),
            # `force`, because the live stage has almost certainly left a
            # COARSE artifact for this same solve behind. Without it the
            # "already built from this solve" short-circuit would see a
            # matching digest and keep the walk-time reconstruction as the
            # finished world -- the exact failure the final stage exists
            # to prevent. The parameters differ, so the params digest
            # differs too and the short-circuit would not in fact fire;
            # this is belt and braces on the thing that would be worst to
            # get wrong.
            force=True,
            should_stop=should_stop,
            # With the evidence gate on (`TOWER_WORLD_SOLVE_GATE`) the final
            # solve already ran the depth stage told the camera's FoV
            # (`coherence_publish.py`); asking for the same here is what makes
            # this stage REUSE it. Off: not passed at all -- today's call.
            **({"depth_known_fov": True} if _solution_gated(store, world_id, session_id) else {}),
        )
    except BaseException:
        # RECORDED, THEN RE-RAISED UNCHANGED. The exception is how the
        # supervisor and the exit code learn; the record is how anyone learns
        # afterwards, and before today there was no afterwards -- the world
        # stayed sparse forever beside a session saying `complete`.
        _record_raise(record, STAGE_SURFACE)
        if appearance:
            record(STAGE_APPEARANCE, state=STAGE_STATE_UNAVAILABLE, attempted=False,
                   detail="the surface stage raised; there was nothing to shade")
        raise
    # THE APPEARANCE IS OWED FROM THIS INSTANT, and it is written down BEFORE
    # the surface's `ok`. The other order left a moment -- two record writes
    # and the appearance modules' cold import -- in which the record said
    # `{surface: ok}` and nothing about the appearance: a finished world to
    # every reader, "Saved" on the phone, and, if the process died right
    # there, a grey mesh that said "Saved" for ever and that the recovery
    # finisher would never select (two independent reviewers, 2026-09-23).
    # Written first, a death anywhere from here leaves `running` under a pid
    # that is gone, which is exactly the signature the finisher recovers.
    if appearance and surface_result.state == "ok":
        record(STAGE_APPEARANCE, state=STAGE_STATE_RUNNING)
    # `detail` on the session record is for a reader asking WHY, and on a
    # successful build the pipeline's `detail` is its entire report -- about
    # ten kilobytes of JSON-inside-a-string, already written verbatim to
    # `surface/<session>/manifest.json`. Copying it here would put it in every
    # `session.json`, which the world listing parses for every session of
    # every world on every poll. Keep it for the states a reader needs it for.
    record(STAGE_SURFACE, state=surface_result.state,
           detail=_terminal_detail(surface_result))
    report["surface"] = {"attempted": True, **surface_result.as_dict()}
    appearance_interrupted = False
    appearance_built = not appearance
    if appearance and surface_result.state != "ok":
        record(STAGE_APPEARANCE, state=STAGE_STATE_UNAVAILABLE, attempted=False,
               detail=f"the surface is {surface_result.state!r}, not 'ok'")
    # The final appearance, on the final surface, BEFORE the depth work
    # is pruned below: it reads each frame's fill mask and raw depth
    # prediction from that work. Skipped on a hard stop like the rest.
    if appearance and surface_result.state == "ok":
        if should_stop():
            appearance_interrupted = True
            record(STAGE_APPEARANCE, state=STAGE_STATE_STOPPED, attempted=False,
                   detail=f"hard stop ({stop_source()}) during finalization")
            report["appearance"] = {
                "attempted": False,
                "reason": f"hard stop ({stop_source()}) during finalization",
            }
        else:
            from tower.world_builder.appearance import AppearanceParams  # noqa: PLC0415
            from tower.world_builder.appearance_pipeline import (  # noqa: PLC0415
                build_appearance,
            )

            # `running` was recorded above, before the surface's `ok`.
            try:
                appearance_result = build_appearance(
                    store, world_id, session_id, params=AppearanceParams(),
                    should_stop=should_stop,
                )
            except BaseException:
                _record_raise(record, STAGE_APPEARANCE)
                raise
            if (appearance_result.state == STAGE_STATE_UNAVAILABLE
                    and getattr(appearance_result, "retryable", False)):
                # A refusal about the MOMENT -- another build holding the
                # session's lock, the redaction label moving under the build
                # -- is an interrupted stage, not a failed one. Recorded
                # `unavailable` with `attempted`, it read as a photographic
                # build that FAILED: terminal, never retried, "Saved" on the
                # phone over a grey mesh. `stopped` is picked up again by the
                # recovery finisher and is bounded by its attempt ledger.
                record(STAGE_APPEARANCE, state=STAGE_STATE_STOPPED,
                       detail=f"{appearance_result.detail}; not a failure of "
                              "the build -- it is tried again")
                appearance_interrupted = True
            else:
                record(STAGE_APPEARANCE, state=appearance_result.state,
                       detail=_terminal_detail(appearance_result))
                appearance_interrupted = appearance_result.state == "stopped"
            report["appearance"] = {"attempted": True, **appearance_result.as_dict()}
            appearance_built = appearance_result.state == "ok"
    # Decided by what the appearance REPORTED as well as by asking again: a
    # stop predicate need not stay true once the stage it stopped has returned
    # (measured: a run stopped in the appearance then pruned the work it needed).
    #
    # And only after an appearance that BUILT (review 1, m2). `unavailable` is
    # most often transient -- a live child's lock whose process has not died
    # yet, a redactor that could not load -- and pruning after it left
    # `world_appearance.py` unable to rebuild without `world_surface.py --force`
    # first. The work is kept until an appearance has used it.
    if (prune_depth_work and surface_result.state == "ok" and not appearance_interrupted
            and appearance_built and not should_stop()):
        # The per-frame depth work is ~0.5 GB for a walk and nothing in
        # the product reads it once the final surface exists; a later
        # rebuild recomputes it. The dense stage prunes its own when on.
        # Not after a hard stop: an appearance stopped half-way would be
        # rebuilt by the next run, and it needs this work to be rebuilt.
        from tower.world_builder.dense_pipeline import (  # noqa: PLC0415
            dense_dir,
            prune_intermediates,
        )

        report["surface"]["depth_work_pruned_bytes"] = prune_intermediates(
            dense_dir(store, world_id, session_id))
    return report


def finalization_notice(summary: dict | None) -> str | None:
    """`finalization.notice` for a final solve that was just published (contract
    WORLD-BUILDER-COMPONENTS.md v6, §3.1): `coherence_publish.publish_notice`'s sentences
    -- what the evidence gate could not do and who can fix it -- or None, which REMOVES
    the key.

    ONLY FOR A SOLVE THE GATE RAN ON (a `gate` record in the summary). `publish_notice`
    also words a GPU-out-of-memory masks sentence for a masked solve with the gate off,
    and that stays where it has always been, in `detail`; §3.1 is explicit that the
    notice is written only for a session whose final solve went through the gate. So
    every ungated world -- every world before the gate, and every Tower with it off --
    keeps a finalization block byte for byte as before."""
    if not isinstance(summary, dict) or not isinstance(summary.get("gate"), dict):
        return None
    from tower.world_builder.coherence_publish import publish_notice  # noqa: PLC0415

    return publish_notice(summary)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a World Builder mapping session over frames on disk."
    )
    parser.add_argument(
        "--root", type=artifact_root_arg, default=str(DEFAULT_ROOT)
    )
    parser.add_argument("--frames", type=Path, help="Directory of .jpg frames.")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Render a synthetic walk instead of reading frames.",
    )
    parser.add_argument("--synthetic-frames", type=int, default=16)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--name", default=None, help="World display name.")
    parser.add_argument("--world", default=None, help="Add to an existing world.")
    parser.add_argument(
        "--intrinsics",
        type=Path,
        help=(
            "JSON file holding a CameraIntrinsics record. Overrides the "
            "intrinsics store. Normally unnecessary: a calibration written "
            "by calibrate_charuco.py is discovered automatically from "
            "<root>/intrinsics/ by the observed frame resolution."
        ),
    )
    parser.add_argument("--backend", choices=BACKEND_NAMES, default=BACKEND_AUTO)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument(
        "--follow-capture",
        type=Path,
        default=None,
        help="Tail a capture directory the Tower is writing, building live.",
    )
    parser.add_argument(
        "--rebuild-every",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Rebuild derived geometry after every N accepted keyframes so a "
            "viewer sees the world grow. 0 (default) builds once at the end."
        ),
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=0.25,
        help="How often to check a followed capture for new frames.",
    )
    parser.add_argument(
        "--max-idle-polls",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Give up after N quiet polls on a capture that never closes. "
            "Unset waits for the recorder to close it."
        ),
    )
    parser.add_argument(
        "--solve",
        action="store_true",
        help=(
            "Run the global solver (tower/world_builder/global_solve.py): in "
            "the background every --solve-every accepted keyframes during the "
            "walk, and once more, in this process, after the last frame. Its "
            "solution is merged into the derived tree by build(). When it "
            "produces a solution, --register is skipped: the placements come "
            "from one reconstruction rather than from Sim3 fits between "
            "fragments."
        ),
    )
    parser.add_argument(
        "--solve-every",
        type=int,
        default=DEFAULT_SOLVE_EVERY,
        help=(
            "MINIMUM accepted keyframes between background solves "
            "(0 = final solve only). The launch is checked inside the "
            "rebuild block, so the effective spacing is this value "
            "rounded up to the rebuild interval -- which widens with the "
            "world (see rebuild_interval): 16x --rebuild-every past 4,800 "
            "keyframes, so 64 at the default. A value below the current "
            "interval cannot be honoured."
        ),
    )
    parser.add_argument(
        "--solve-wait-seconds",
        type=float,
        default=DEFAULT_SOLVE_WAIT_SECONDS,
        help="how long Stop waits for a running background solve before the final solve",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help=(
            "After the final build, try to place the session's segments in "
            "one coordinate frame and persist the result as placements.json. "
            "Runs once, at the end, in this process -- never on the frame "
            "path. A segment's own geometry is never moved and a refusal is "
            "the default, so the worst case is the unregistered world you "
            "would have had anyway."
        ),
    )
    parser.add_argument(
        "--surface",
        action="store_true",
        help="reconstruct a surface: coarsely during the walk, whenever a "
             "background solve lands, and at full resolution after Stop. "
             "Needs --solve.",
    )
    parser.add_argument(
        "--appearance",
        action="store_true",
        help="build the appearance artifact (redacted keyframes prepared for "
             "view-dependent blending on the phone) after each live surface and "
             "after the final surface. Needs --surface.",
    )
    parser.add_argument(
        "--surface-script",
        type=Path,
        default=None,
        help="override the live surface child's script (tests)",
    )
    parser.add_argument(
        "--densify",
        action="store_true",
        help=(
            "After the final build and the final solve, reconstruct a dense "
            "point cloud from the solved cameras and persist it under "
            "<world>/dense/<session>. Runs once, at the end, in this process "
            "-- never on the frame path. It reads the world's own redacted "
            "keyframe imagery, adds an artifact nothing else reads, and "
            "changes neither derived/ nor the world's scale semantics, so a "
            "failure leaves exactly the world you would have had. Costs "
            "minutes: skipped outright on a hard stop."
        ),
    )
    parser.add_argument(
        "--stop-on-stdin-close",
        action="store_true",
        help=(
            "Treat EOF on stdin as a SOFT stop request: stop observing, close "
            "the session as interrupted, skip the final solve, write the final "
            "build. The Tower's worker supervisor holds the other end of that "
            "pipe. SIGBREAK/SIGTERM/SIGINT are always a HARD stop (wrap up now)."
        ),
    )
    parser.add_argument(
        "--solve-script",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,  # a test seam: run this instead of world_solve.py
    )
    args = parser.parse_args(argv)

    # Configured here rather than at import, so importing this module for
    # a test does not reconfigure the test runner's logging. Only added
    # if nothing else has set logging up: when the Tower spawns this as a
    # capture worker its output is inherited, and a second handler would
    # double every line in that console.
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )

    # BEFORE the watcher is armed, for the reason `tower/native_prewarm.py`
    # gives at length: this process runs `final_surface_stages` after Stop,
    # which loads the same OpenBLAS-backed stack that deadlocked the
    # recovery finisher against its own parked stdin reader on 2026-09-22.
    # The builder is the other process that runs those stages, so it is the
    # other process that has to be warmed.
    prewarm_world_builder()

    # Installed before the follower is built and before the first frame is
    # read, because the poll loop it arms is the thing being armed.
    stop_request = StopRequest()
    stop_request.install(watch_stdin=args.stop_on_stdin_close)

    chosen = [
        name
        for name, value in (
            ("--frames", args.frames),
            ("--synthetic", args.synthetic),
            ("--follow-capture", args.follow_capture),
        )
        if value
    ]
    if len(chosen) != 1:
        parser.error(
            "exactly one of --frames, --synthetic or --follow-capture is required"
        )
    if args.rebuild_every < 0:
        parser.error("--rebuild-every must not be negative")

    # Keyed by the resolution frames MEASURE at, never by --width/--height.
    intrinsics_store = IntrinsicsStore(args.root)
    # Absolute, always, and before anything else. `--root data/world_builder`
    # resolves against THIS process's working directory, which the Tower
    # sets to TOWER_ROOT rather than inheriting from the shell that
    # started it. A relative path printed as-is cannot show that the
    # store and the calibrator disagreed about which directory they meant.
    logger.info(
        "[Tower][WorldBuilder] world root %s (cwd %s) -- calibrations on "
        "file: %s",
        args.root.resolve(),
        Path.cwd(),
        ", ".join(f"{w}x{h}" for w, h in intrinsics_store.list_resolutions())
        or "NONE",
    )

    capture_id = None
    synthetic_intrinsics = None
    capture_handle: dict = {}
    if args.follow_capture:
        frames = follow_capture(
            args.follow_capture,
            poll_seconds=args.poll_seconds,
            max_idle_polls=args.max_idle_polls,
            # Asked inside the poll loop, which is where this process
            # spends a quiet walk. See `StopRequest`.
            should_stop=stop_request.asked_for,
            handle=capture_handle,
        )
        frame_source = "live-capture"
        capture_id = args.follow_capture.name
        # The sender chooses the stream size and DAT may change it
        # mid-walk. This process measures each frame; it declares nothing.
        declared_size = None
    elif args.synthetic:
        frames, synthetic_intrinsics = synthetic_frames(
            args.synthetic_frames, args.width, args.height
        )
        frame_source = "synthetic"
        # The only source whose size this process actually chose.
        declared_size = (args.width, args.height)
    else:
        frames = load_frames(args.frames)
        frame_source = "recorded-capture"
        # A directory of jpegs whose size was decided by whatever wrote
        # them. Measured per keyframe, not declared here.
        declared_size = None

    # The world BEFORE the wait below, not after. Creating it is a couple
    # of small writes and it is what the result channel looks for, so a
    # Tower whose phone has connected but not yet sent a frame reports a
    # world that exists and is empty rather than no world at all.
    store = WorldStore(args.root)
    engine = WorldBuilderEngine(store, backend_name=args.backend)
    world_id = args.world or engine.create_world(args.name)

    # Only now, with somewhere to put the answer, ask what size the
    # frames are -- and on the live path, wait until there IS a frame to
    # ask about. See "WHEN THE LOOKUP HAPPENS" in the module docstring.
    # What actually answered the calibration question, for the log below.
    # A path when the store was asked, a sentence when it was not: a log
    # that always prints a store path implies the store was consulted
    # even for a synthetic run, and this line exists precisely so nobody
    # has to guess which file the numbers came from.
    consulted = "the store"
    if args.synthetic:
        intrinsics = synthetic_intrinsics
        observed_size = declared_size
        consulted = "not consulted (the synthetic renderer supplies its own)"
    elif args.follow_capture:
        first, frames = first_observed_frame(frames)
        if first is None:
            # The capture closed, or gave up, without a single frame.
            # Not an error: it is a phone that connected and dropped. The
            # session still opens, honestly empty.
            logger.warning(
                "[Tower][WorldBuilder] capture %s delivered no frames, so no "
                "resolution was ever observed and no calibration was looked "
                "up. This session will be empty.",
                capture_id,
            )
            observed_size = None
        else:
            observed_size = observed_size_of(first)
            logger.info(
                "[Tower][WorldBuilder] first frame of capture %s observed at "
                "%s (source_seq=%s); resolving intrinsics against THAT, not "
                "against anything declared",
                capture_id,
                f"{observed_size[0]}x{observed_size[1]}"
                if observed_size
                else "an unreadable size",
                first.source_seq,
            )
        intrinsics = resolve_intrinsics(
            intrinsics_store, observed_size, frame_source=frame_source
        )
    else:
        observed_size = observed_size_from_frames(args.frames)
        intrinsics = resolve_intrinsics(
            intrinsics_store, observed_size, frame_source=frame_source
        )

    if consulted == "the store":
        consulted = (
            str(intrinsics_store.path_for(*observed_size))
            if observed_size
            else "not consulted (no resolution was ever observed)"
        )

    # An explicit file always wins over the store: it is the escape hatch
    # for a calibration that lives elsewhere, and for reproducing an old
    # build against the intrinsics it originally used. Unlike the store
    # this does NOT check the observed resolution -- the engine's
    # `_require_matching_resolution` does, per keyframe, and a hard
    # failure is the right answer when an operator names a file by hand.
    if args.intrinsics:
        intrinsics = camera_intrinsics_from_json_dict(
            json.loads(args.intrinsics.read_text(encoding="utf-8"))
        )
        consulted = f"{args.intrinsics} (--intrinsics override)"
        logger.info(
            "[Tower][WorldBuilder] using --intrinsics %s (source=%s, %sx%s), "
            "overriding the intrinsics store",
            args.intrinsics,
            intrinsics.source,
            intrinsics.calibrated_width,
            intrinsics.calibrated_height,
        )

    session_id = engine.start_session(
        world_id,
        intrinsics=intrinsics,
        frame_source=frame_source,
        # Only for a source whose size this process actually CHOSE.
        #
        # --width/--height default to 480x360 and describe the synthetic
        # renderer. Passing them for a followed capture records a size
        # nobody measured: the 2026-08-24 session says `declared_width:
        # 480, declared_height: 360` while every one of its 155 keyframes
        # is 360x640. It was harmless only because unknown intrinsics
        # skip the resolution check -- the moment a calibration exists,
        # `_require_matching_resolution` turns it into a hard failure, or
        # worse, invites calibrating at the wrong resolution.
        #
        # Per-keyframe width/height are measured off the decoded frame
        # and were always right. None means unknown, which is the honest
        # value for a stream whose size the sender decides.
        declared_size=declared_size,
        capture_id=capture_id,
    )

    # Everything a physical run needs in order to answer "is calibration
    # active?" without opening a session record afterwards, on one line.
    # After the 2026-08-25 walk the answer existed only on disk, hours
    # later: the log said `intrinsics=unknown` and nothing about which
    # resolution had been observed or which file had been consulted, so
    # "the calibration is wrong" and "the calibration was never read"
    # looked identical.
    #
    # `announce=False` because the engine announces the selection itself
    # a few lines from now, in `_open_live_solve`. This asks the same
    # deterministic function what it will decide; it does not decide.
    selection = select_backend(args.backend, intrinsics, announce=False)
    logger.info(
        "[Tower][WorldBuilder] session %s in world %s: source=%s capture=%s "
        "root=%s observed=%s calibration=%s intrinsics=%s backend=%s "
        "(requested %s) rebuild_every=%s",
        session_id,
        world_id,
        frame_source,
        capture_id,
        args.root.resolve(),
        f"{observed_size[0]}x{observed_size[1]}" if observed_size else "UNOBSERVED",
        consulted,
        intrinsics.source,
        selection.backend.capabilities.backend_id,
        args.backend,
        args.rebuild_every,
    )
    if selection.was_downgraded:
        logger.warning(
            "[Tower][WorldBuilder] backend downgraded from %s: %s",
            selection.downgraded_from,
            selection.downgrade_reason,
        )

    started = time.perf_counter()
    rebuilds = 0
    since_rebuild = 0
    accepted = 0
    warned_frame_size = False
    # keyframe_id -> raw frame path, for the global solver (see ObservedFrame).
    sources: dict = {}
    capture_dirs = [d for d in (args.follow_capture, args.frames) if d is not None]
    solver = None
    if args.solve:
        solver = BackgroundSolver(
            root=args.root.resolve(), world_id=world_id, session_id=session_id,
            # 0 means "final solve only": no background cadence, but the
            # final solve still runs as a child this process owns.
            every=args.solve_every if args.solve_every > 0 else 0,
            capture_dirs=capture_dirs, script=args.solve_script,
        )
    background_solves = args.solve and args.solve_every > 0
    # The live surface needs a global solve to exist at all, so it is only
    # constructed when background solves are on. With `--solve-every 0` there
    # is one solve, at the end, and nothing to show during the walk.
    surfacer = None
    if args.surface and background_solves:
        surfacer = BackgroundSurface(
            root=args.root.resolve(), world_id=world_id, session_id=session_id,
            script=args.surface_script, appearance=args.appearance,
        )

    # THE LIFECYCLE, IN ONE PLACE, AND IT UNWINDS.
    #
    # Everything from the first frame to the last write sits inside one
    # try. A stop request ends the frame loop at the next poll; an
    # exception ends it with `end_reason: error`; either way the session is
    # CLOSED on disk, the finalization is RECORDED, the final build is
    # attempted, every solve child is reaped and the lock is released.
    # Before 2026-09-06 none of that was guaranteed, and a builder that
    # died mid-walk left a lock naming a dead pid as the only account of
    # what happened.
    end_reason = END_REASON_STOP
    exit_code = 0
    summary = None
    solve_report = None
    result = None
    observe_seconds = 0.0
    finalization_state = FINALIZATION_COMPLETE
    final_solve_state = None
    finalization_detail = None
    # `finalization.notice` (contract v6, §3.1): set only when a gated final solve is
    # published and owes something; None writes no key. `stop_session` has just given the
    # record a fresh `pending` block, so there is no older notice here to keep.
    finalization_notice_text = None
    try:
        for frame in stop_request.bounded(frames):
            outcome = engine.observe(
                frame.payload,
                received_at=frame.received_at,
                source_seq=frame.source_seq,
                wire_seq=frame.wire_seq,
                tx_seq=frame.tx_seq,
            )
            if outcome.keyframe_id is None:
                if (
                    getattr(outcome, "reason", None) == "frame_size_changed"
                    and not warned_frame_size
                ):
                    # ONCE PER SESSION, at WARNING, because the engine's
                    # rejection is silent here otherwise: this loop
                    # `continue`d past it with no log, and a whole walk at
                    # the wrong rung left nothing but journal lines. The
                    # count reaches the phone through the status channel;
                    # this is for the operator reading the Tower log.
                    warned_frame_size = True
                    logger.warning(
                        "[Tower][WorldBuilder] session %s is receiving frames "
                        "of a different size from its first frame; they are "
                        "being rejected, because the calibration is exact "
                        "per resolution. Every further frame at that size "
                        "will be rejected too",
                        session_id,
                    )
                continue
            accepted += 1
            since_rebuild += 1
            if frame.source_path is not None:
                sources[outcome.keyframe_id] = str(frame.source_path)
            # A finished background solve is worth a rebuild now: the solution
            # reaches the derived tree only through build(), and the wearer
            # should see the world snap together as soon as it is known.
            solve_landed = background_solves and solver.finished()
            # Two keyframes is the minimum a two-view backend can say anything
            # about. Rebuilding on one would burn a build to produce an anchor
            # pose and nothing else.
            if (args.rebuild_every and since_rebuild >= rebuild_interval(
                    args.rebuild_every, accepted) and accepted >= 2) or (
                solve_landed and accepted >= 2
            ):
                rebuild_started = time.perf_counter()
                try:
                    interim = engine.build(world_id, session_id)
                except OSError as exc:
                    # An interim rebuild is a best-effort view for the wearer;
                    # the next one rewrites every derived file. Windows refuses
                    # the atomic replace while any reader holds the destination
                    # (the Tower's web thread reading points.json for the
                    # phone, descheduled under solver load), and on 2026-09-06
                    # that exception ended a live session mid-walk: no
                    # session_stopped, a LOCK with a dead pid, a torn derived
                    # tree. Say so, and try again at the next rebuild.
                    since_rebuild = 0
                    logger.warning(
                        "[Tower][WorldBuilder] rebuild %s failed and will be retried "
                        "at the next one: %s: %s",
                        rebuilds + 1, type(exc).__name__, exc,
                    )
                    continue
                rebuilds += 1
                since_rebuild = 0
                # One line per rebuild, not per frame. Over a 15-minute walk
                # this process used to print nothing at all until it was
                # over, so "why isn't World Builder changing?" had no
                # answer short of reading the world directory by hand.
                logger.info(
                    "[Tower][WorldBuilder] rebuild %s: %s keyframes -> %s "
                    "positioned poses, %s points, %s segments in %.2fs",
                    rebuilds,
                    interim.keyframes,
                    interim.poses_solved,
                    interim.points,
                    interim.segments,
                    time.perf_counter() - rebuild_started,
                )
                if background_solves:
                    solver.maybe_launch(store, accepted, sources)
                # A solve landing is the one moment the live surface can be
                # rebuilt from geometry worth trusting, so the launch is here
                # and not on a keyframe count. It is asked AFTER the solver,
                # so a solve that is due wins the card.
                if surfacer is not None:
                    if solve_landed:
                        surfacer.solve_landed(store)
                    else:
                        surfacer.poll(store)
        observe_seconds = time.perf_counter() - started
        if surfacer is not None:
            # The walk is over, so a live surface still building is already
            # obsolete: the final one after the final solve replaces it. Left
            # running it competed with finalization -- on a real-time replay
            # of the canonical walk the last live build finished 39 s after
            # Stop, while the final solve waited. Ended here rather than only
            # in the `finally`, which runs after finalization.
            surfacer.close()

        # WAS THE CAPTURE STILL RUNNING WHEN WE WERE TOLD TO GO?
        #
        # That is the question, and this used to ask a different one: any
        # stop request at all made the session `interrupted`. But the
        # wearer leaving the World Builder screen IS a soft stop -- iOS
        # posts `session/stop` from `.onDisappear` -- and it arrives right
        # after the Stop that closed the capture. So the ordinary way to
        # finish a walk produced `end_reason: interrupted`, and
        # `results/world_builder.py` maps that to Interrupted ahead of ever
        # looking at `finalization`. Measured on a real 12 fps capture: a
        # wearer who leaves immediately gets `interrupted`, one who lingers
        # a second gets `stop`, on identical geometry.
        #
        # A capture that has written its end reason ended because somebody
        # pressed Stop. Whether this process was also told to go afterwards
        # says nothing about the walk. `is_closed()` is asked of the
        # FOLLOWER, not of the directory we started with, because a
        # reconnect retargets it onto a successor capture.
        follower = capture_handle.get("follower")
        # `bounded_limit` is NOT the wearer. The recorder stops itself at a
        # configured bound and its own log says a follower sees that "exactly
        # as if it were" a disconnect -- so treating a closed capture as an
        # ordinary end would finalise a world at the bound, under the label
        # `stop`, while the wearer is still walking. The bound is forty
        # minutes now, but a walk that reaches it should say so.
        capture_end = follower.end_reason() if follower is not None else None
        # A reconnect still in flight when the stop arrived is NOT a
        # finished walk, even though the capture it was following ended
        # `disconnect` and `disconnect` counts as finished. The wearer was
        # still walking; the link died and nobody waited for it. Only the
        # follower knows, so it is asked -- see
        # `CaptureFollower.stopped_awaiting_successor`.
        abandoned_reconnect = (
            follower is not None and follower.stopped_awaiting_successor()
        )
        capture_finished = (
            capture_end in (END_REASON_CAPTURE_STOP, END_REASON_CAPTURE_DISCONNECT)
            and not abandoned_reconnect
        )
        if stop_request.asked and not capture_finished:
            # Now it means what it says: frames were still coming and
            # somebody asked this process to go.
            end_reason = END_REASON_INTERRUPTED
            logger.warning(
                "[Tower][WorldBuilder] stop requested (%s, %s) while the capture was "
                "still open; the session ends as %r",
                stop_request.level, stop_request.source, end_reason,
            )
        elif stop_request.asked:
            logger.info(
                "[Tower][WorldBuilder] stop requested (%s, %s) after the capture "
                "closed (%s); this is an ordinary end and the session ends as %r",
                stop_request.level, stop_request.source, capture_end, end_reason,
            )
        if capture_end == END_REASON_CAPTURE_BOUNDED:
            # A BOUND IS NOT A STOP, and this used only to say so in a log
            # line while recording `stop` anyway -- a warning that claimed
            # "the truncation is not reported as a clean finish" beside the
            # clean-finish label. Caught by an adversarial review running
            # these very lines against a real follower.
            #
            # Nobody asked for this walk to end: the recorder reached forty
            # minutes and stopped itself while the wearer was still
            # walking, and everything after that moment is missing from the
            # world. `interrupted` is what that is.
            end_reason = END_REASON_INTERRUPTED
            logger.warning(
                "[Tower][WorldBuilder] the capture stopped ITSELF at a configured "
                "bound, not because anyone asked; the walk was longer than the "
                "world. The session ends as %r, and whatever came after the bound "
                "is not in it.",
                end_reason,
            )
        summary = engine.stop_session(
            end_reason, hold_lock=True, capture_end_reason=capture_end
        )

        # -- finalization: the lock is still held, the record says pending --
        if solver is None:
            final_solve_state = None
        elif stop_request.hard:
            # ONLY a hard stop skips it. This used to be `stop_request.asked`,
            # so a SOFT stop -- the wearer leaving the World Builder screen,
            # or the cartridge session being stopped -- skipped the final
            # solve too, on the reasoning that "nobody is waiting for it".
            #
            # The 2026-09-09 walk falsifies that premise. Nobody watches a
            # final solve; they open the world afterwards. And the final
            # solve is what MAKES the world: re-running the one that walk
            # never got, on its own images, took 16 components to 6 and put
            # 652 of 795 keyframes into one at 0.82 px, against a largest
            # component of 156 before. Skipping it does not save the wearer
            # a wait, it costs them the reconstruction.
            #
            # A hard stop is different and still skips: the Tower is going
            # down, and `run_final` below would be killed mid-solve anyway.
            final_solve_state = FINAL_SOLVE_SKIPPED
            finalization_detail = (
                f"final solve skipped: hard stop ({stop_request.source}) "
                "while observing"
            )
            solver.wait(0.0)
        else:
            # A background solve still running at Stop is given a bounded
            # wait and then TERMINATED, never abandoned: the final solve is
            # about to reuse its workspace.
            solver.wait(args.solve_wait_seconds, should_stop=stop_request.hard_asked_for)
            if stop_request.hard:
                final_solve_state = FINAL_SOLVE_SKIPPED
                finalization_detail = (
                    f"final solve skipped: hard stop ({stop_request.source}) during finalization"
                )
            else:
                solve_report = solver.run_final(
                    store, sources, should_stop=stop_request.hard_asked_for
                )
                if solve_report.get("solved"):
                    final_solve_state = FINAL_SOLVE_SOLVED
                    # What the published solve still owes, on the row (review V7, H2 and
                    # L-c): masks lost to GPU memory (an owner re-finishes this walk), a
                    # gate the idle Tower re-runs. None for every ungated solve.
                    from tower.world_builder.coherence_publish import (  # noqa: PLC0415
                        publish_notice,
                    )

                    finalization_detail = publish_notice(solve_report) or finalization_detail
                    # And the phone's copy of it (v6, §3.1; `detail` keeps its meaning and
                    # still carries the sentence).
                    finalization_notice_text = finalization_notice(solve_report)
                elif solve_report.get("interrupted"):
                    final_solve_state = FINAL_SOLVE_SKIPPED
                    finalization_detail = (
                        f"final solve terminated: hard stop ({stop_request.source}) "
                        "during finalization; the last background solution stands"
                    )
                elif solve_report.get("error"):
                    final_solve_state = FINAL_SOLVE_FAILED
                    finalization_detail = f"final solve failed: {solve_report['error']}"
                else:
                    final_solve_state = FINAL_SOLVE_UNAVAILABLE
                    finalization_detail = (
                        f"final solve produced no solution: {solve_report.get('reason')}"
                    )
        if solver is not None:
            if solve_report is None:
                solve_report = {"attempted": False, "solved": False}
            solve_report["background_launches"] = solver.launches
            solve_report["final_solve"] = final_solve_state

        built = time.perf_counter()
        result = engine.build(world_id, session_id)
        build_seconds = time.perf_counter() - built
        logger.info(
            "[Tower][WorldBuilder] session %s finished: %s frames, %s keyframes, "
            "%s segments, backend=%s (downgraded_from=%s), %s solved poses, "
            "%s points, scale=%s, final build %.2fs",
            session_id,
            summary.frames_observed,
            summary.keyframes_accepted,
            summary.segments,
            result.backend_id,
            result.downgraded_from,
            result.poses_solved,
            result.points,
            result.scale_state,
            build_seconds,
        )
    except BaseException as exc:  # noqa: BLE001 -- the unwind IS the point
        exit_code = 1
        finalization_state = FINALIZATION_INTERRUPTED
        finalization_detail = f"{type(exc).__name__}: {exc}"
        logger.exception(
            "[Tower][WorldBuilder] session %s ended by %s; closing the record and "
            "keeping what was built",
            session_id, type(exc).__name__,
        )
        if engine.session_active:
            try:
                summary = engine.stop_session(END_REASON_ERROR, hold_lock=True)
            except Exception:  # noqa: BLE001
                logger.exception("[Tower][WorldBuilder] could not close the session record")
        if result is None:
            # A best-effort last build: whatever the journal holds is worth
            # a derived tree. It may raise for the same reason the loop did;
            # that is logged, not propagated over the record-keeping below.
            try:
                result = engine.build(world_id, session_id)
            except Exception:  # noqa: BLE001
                logger.exception("[Tower][WorldBuilder] the final build after the error failed too")
        if isinstance(exc, KeyboardInterrupt):
            exit_code = 130
    finally:
        if surfacer is not None:
            # Before the solver, because the live surface is the disposable
            # one: whatever it was part-way through is replaced by the final
            # build below, and leaving it holding the GPU would slow that down.
            surfacer.close()
        if solver is not None:
            solver.close()
        try:
            engine.mark_finalization(
                world_id, session_id,
                state=finalization_state,
                final_solve=final_solve_state,
                detail=finalization_detail,
                notice=finalization_notice_text,
            )
        except Exception:  # noqa: BLE001
            logger.exception("[Tower][WorldBuilder] could not record the finalization")

        # THE OWED WORK IS WRITTEN DOWN BEFORE THE LOCK IS DROPPED, and the
        # order is the whole point.
        #
        # Releasing the lock here is correct and stays: the phone must be
        # able to read the world while the photographic stages build it. But
        # between this release and `final_surface_stages`' own first record,
        # a hundred lines below, there used to be NOTHING ON DISK saying a
        # photographic room was coming. The status channel read that window
        # as a finished world and the phone said **Saved**, then went back to
        # Improving when the surface finally wrote its first `running`
        # status. That is the first half of the Mac validation's T2, and it
        # is the half the record could not cover because the record did not
        # exist yet.
        #
        # `running` rather than a new word, and it is not a lie: this process
        # IS working on this session, in this function, microseconds from
        # calling the stage. It is also the right thing to find afterwards --
        # a builder that dies in this window leaves `running` under a dead
        # pid, which is precisely the signature
        # `scripts/world_finish_pending.py` selects on and recovers, where
        # before it left nothing at all and the work was lost silently.
        #
        # Guarded on `args.surface` because a Tower with the stages switched
        # off owes nothing and must not be made to look as though it does --
        # and on the SOLVE, because `final_surface_stages` refuses without
        # one and `world_finish_pending.assess()` will not pick such a
        # session up either (`no-final-solve`). A record nothing will ever
        # act on is a world that reads "Improving" forever.
        if (
            args.surface
            and finalization_state == FINALIZATION_COMPLETE
            and final_solve_state == FINAL_SOLVE_SOLVED
        ):
            try:
                engine.mark_stage(
                    world_id, session_id, STAGE_SURFACE,
                    state=STAGE_STATE_RUNNING,
                    detail="the photographic stages are about to run",
                )
            except Exception:  # noqa: BLE001 -- a record is not worth the walk
                logger.exception(
                    "[Tower][WorldBuilder] could not record that the "
                    "photographic stages are owed"
                )

        engine.release_world(world_id)

    if result is None or summary is None:
        return exit_code

    report = {
        "world_id": world_id,
        "session_id": session_id,
        "frame_source": frame_source,
        "end_reason": end_reason,
        "finalization": finalization_state,
        "frames_observed": summary.frames_observed,
        "keyframes_accepted": summary.keyframes_accepted,
        "rejected_by_reason": summary.rejected_by_reason,
        "segments": summary.segments,
        "rebuilds": rebuilds,
        "backend_id": result.backend_id,
        "downgraded_from": result.downgraded_from,
        "poses_solved": result.poses_solved,
        "poses_refused": result.poses_refused,
        "points": result.points,
        "scale_state": result.scale_state,
        "observe_ms_per_frame": round(
            observe_seconds * 1000 / max(summary.frames_observed, 1), 3
        ),
        "build_seconds": round(build_seconds, 3) if exit_code == 0 else None,
    }

    # After the final build, never between rebuilds: registration reads
    # the derived tree and binds its answer to that build's digest, so a
    # mid-walk run would solve against geometry the next rebuild
    # replaces and be discarded at serve time anyway.
    if solve_report is not None:
        report["global_solve"] = solve_report
    # The Sim3 registrar places fragments against each other; when the
    # global solve produced a solution the placements already come from one
    # reconstruction and a second, weaker answer must not overwrite them.
    #
    # THE QUESTION IS WHO WROTE placements.json, NOT WHETHER THE FINAL SOLVE
    # RAN. This used to read `not (solve_report or {}).get("solved")`, and
    # `solve_report` is the FINAL solve's report -- None whenever the session
    # did not reach finalization normally. On the 2026-09-09 walk the session
    # died in the observe loop, so `solve_report` was None, so the guard
    # concluded there was no solution and ran the registrar. It ran at
    # 20:24:48.3, sixteen seconds AFTER the last build had written its
    # placements at 20:24:32.5, and overwrote them.
    #
    # What it destroyed is on record in the manifest beside the file it
    # replaced: `global_solve` reports 72 segments registered into 14
    # components from nine successful background solves, while the
    # `placements.json` the phone actually reads was left saying 120 refused
    # and 2 registered. That single substitution is why a walk that
    # reconstructed most of a room was drawn as 87 disconnected fragments.
    #
    # `engine.build()` now says which producer owns the file, so the guard
    # asks the build that actually wrote it.
    if args.register:
        report["registration"] = (
            register_session(store, world_id, session_id)
            if should_register(result)
            else {
                "attempted": False,
                "reason": "the global solve placed these segments",
            }
        )

    # EVERYTHING BELOW RUNS AFTER THE RECORD ALREADY SAYS `complete` AND THE
    # WORLD LOCK IS ALREADY RELEASED, and both of those are deliberate: the
    # lock is dropped so the phone can read the world while the surface
    # builds (`results/world_builder_render.py`), which is six to eleven
    # minutes on a real walk. What was NOT deliberate is that the outcome of
    # these stages existed only in `report` below -- printed once, persisted
    # nowhere. A `surfacify()` that raised left a world sparse forever with a
    # session record proudly saying `complete`, and no operator or client
    # could tell that from a world that was never asked for a surface.
    #
    # `record_stage` writes the second record, beside `finalization` and
    # never into it. It moves nothing: not one call below changed position,
    # the lock stays released, and `finalization` still says exactly what it
    # said before.
    def record_stage(stage, *, state, detail=None, attempted=True):
        try:
            engine.mark_stage(
                world_id, session_id, stage,
                state=state, detail=detail, attempted=attempted,
            )
        except Exception:  # noqa: BLE001
            # A record that cannot be written must not take the artifact
            # down with it: by here the world itself is already on disk.
            logger.exception(
                "[Tower][WorldBuilder] could not record the %s stage", stage
            )

    # The final surface, after everything load-bearing is on disk. It runs
    # BEFORE the dense stage, and the order is not cosmetic: the dense stage
    # prunes the per-frame depth work when it finishes, so a surface built
    # after it found every prediction gone and ran the depth network again over
    # the whole walk. Built first, it computes the depth stage and names it for
    # this solve, and the dense stage then reuses it by its cache key.
    #
    # Skipped rather than truncated on a hard stop. The Job Object kills this
    # tree on a 30-second grace and a half-written surface would be worse than
    # none, which is why the artifact is published atomically and why the
    # format checks its own length on read.
    if args.surface:
        report.update(final_surface_stages(
            store, world_id, session_id,
            solved=bool((solve_report or {}).get("solved")),
            appearance=args.appearance,
            prune_depth_work=not args.densify,
            should_stop=stop_request.hard_asked_for,
            stop_source=lambda: stop_request.source,
            record=record_stage,
        ))
    else:
        # NOT REQUESTED IS A FACT, AND IT IS NOT THE SAME FACT AS SILENCE.
        # An absent key means "a Tower that never recorded this"; this means
        # "nobody asked for a photographic world", which is why a sparse
        # world here is not a failure.
        for stage in (STAGE_SURFACE, STAGE_APPEARANCE):
            record_stage(stage, state=STAGE_STATE_UNAVAILABLE, attempted=False,
                         detail="not requested (--surface was not passed)")
    # `--surface` without `--appearance` is recorded by `final_surface_stages`
    # itself, so the recovery finisher records it too.

    # THE AREAS' "OWED" MUST BE A PROMISE SOMETHING KEEPS (WORLD-BUILDER-COMPONENTS.md
    # §3.4, §7 rule 5; review V6, M2). An area the evidence gate named reads `owed`
    # until the idle finisher builds it. On a Tower where no finisher will run, or
    # where area builds are switched off, that is never, so the room's end is where
    # they are recorded as declined instead. Nothing for a session with no components
    # record, and nothing when the finisher will build them. Never raises.
    from tower.world_builder.components import (  # noqa: PLC0415
        settle_areas_nobody_will_build,
    )

    settled = settle_areas_nobody_will_build(store, world_id, session_id)
    if settled:
        report["areas_declined"] = settled

    # Dense reconstruction last (after the surface, which shares its depth
    # stage), because it is the most expensive thing here
    # and the least load-bearing: every other artifact is already on disk and
    # complete before it starts. It is skipped rather than truncated on a hard
    # stop -- the Job Object kills this tree on a 30-second grace and a
    # half-written dense tree would be worse than none. Its own stages are
    # checkpointed, so a later `scripts/world_densify.py` resumes rather than
    # restarting.
    if args.densify:
        if stop_request.hard_asked_for():
            reason = f"hard stop ({stop_request.source}) during finalization"
            record_stage(STAGE_DENSE, state=STAGE_STATE_STOPPED, attempted=False,
                         detail=reason)
            report["dense"] = {"attempted": False, "reason": reason}
        elif not (solve_report or {}).get("solved"):
            reason = "dense reconstruction needs a global solve; there is none"
            record_stage(STAGE_DENSE, state=STAGE_STATE_UNAVAILABLE, attempted=False,
                         detail=reason)
            report["dense"] = {"attempted": False, "reason": reason}
        else:
            from tower.world_builder.dense_pipeline import densify  # noqa: PLC0415

            record_stage(STAGE_DENSE, state=STAGE_STATE_RUNNING)
            try:
                dense_result = densify(
                    store, world_id, session_id,
                    should_stop=stop_request.hard_asked_for,
                )
            except BaseException:
                _record_raise(record_stage, STAGE_DENSE)
                raise
            record_stage(STAGE_DENSE, state=dense_result.state,
                         detail=dense_result.detail)
            report["dense"] = {"attempted": True, **dense_result.as_dict()}
    else:
        record_stage(STAGE_DENSE, state=STAGE_STATE_UNAVAILABLE, attempted=False,
                     detail="not requested (--densify was not passed)")

    if args.format == "json":
        print(json.dumps(report, indent=2))
    else:
        for key, value in report.items():
            print(f"{key:22s} {value}")
        if frame_source == "synthetic":
            print(
                "\nSYNTHETIC, NOT PHYSICAL: nothing here says anything about "
                "the Ray-Ban camera."
            )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
