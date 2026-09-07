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
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tower.artifact_paths import artifact_root_arg  # noqa: E402
from tower.capture import CaptureFollower  # noqa: E402
from tower.world_builder.backends import (  # noqa: E402
    BACKEND_AUTO,
    BACKEND_NAMES,
    select_backend,
)
from tower.world_builder.engine import WorldBuilderEngine  # noqa: E402
from tower.world_builder.intrinsics_store import IntrinsicsStore  # noqa: E402
from tower.world_builder.records import (  # noqa: E402
    CameraIntrinsics,
    camera_intrinsics_from_json_dict,
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


def follow_capture(directory: Path, *, poll_seconds: float, max_idle_polls):
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
        directory, poll_seconds=poll_seconds, max_idle_polls=max_idle_polls
    )


def _follow_capture(directory: Path, *, poll_seconds: float, max_idle_polls):
    follower = CaptureFollower(directory, poll_seconds=poll_seconds)
    for frame in follower.follow(max_idle_polls=max_idle_polls):
        yield ObservedFrame(
            payload=frame.raw_bytes,
            source_seq=frame.source_seq,
            wire_seq=frame.wire_seq,
            tx_seq=frame.tx_seq,
            received_at=frame.received_at,
            width=frame.width,
            height=frame.height,
            source_path=directory / frame.relpath,
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


class BackgroundSolver:
    """The global solve, run as a child of the builder so the frame path never
    waits for it.

    One child at a time. `maybe_launch` starts `scripts/world_solve.py`
    when no solve is running and at least `every` keyframes have been
    accepted since the last launch; `finished()` reports (once) that a
    launch has completed, which is the builder's cue to rebuild so the new
    solution reaches the derived tree without waiting for the next
    rebuild interval. Output goes to `solve/<session>/solve.log`.
    """

    def __init__(self, *, root: Path, world_id: str, session_id: str, every: int,
                 capture_dirs, threads: int | None = None):
        self.root = root
        self.world_id = world_id
        self.session_id = session_id
        self.every = max(1, every)
        self.capture_dirs = [Path(d) for d in capture_dirs]
        self.threads = threads
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

    def maybe_launch(self, store: WorldStore, accepted: int, sources: dict) -> bool:
        self._reap()
        if self.running or accepted - self._launched_at_keyframes < self.every or accepted < 2:
            return False
        from tower.world_builder import global_solve  # noqa: PLC0415

        global_solve.write_sources(store, self.world_id, self.session_id, sources)
        workspace = global_solve.workspace_for(store, self.world_id, self.session_id)
        workspace.root.mkdir(parents=True, exist_ok=True)
        argv = [
            sys.executable, str(TOWER_ROOT / "scripts" / "world_solve.py"),
            "--root", str(self.root), "--world", self.world_id, "--session", self.session_id,
        ]
        for capture_dir in self.capture_dirs:
            argv += ["--capture-dir", str(capture_dir)]
        if self.threads is not None:
            argv += ["--threads", str(self.threads)]
        self._log = open(workspace.root / "solve.log", "ab")
        self._child = subprocess.Popen(
            argv, cwd=str(TOWER_ROOT), stdout=self._log, stderr=subprocess.STDOUT
        )
        self._launched_at_keyframes = accepted
        self._launches += 1
        logger.info(
            "[Tower][WorldBuilder] background solve %s launched at %s keyframes (pid %s)",
            self._launches, accepted, self._child.pid,
        )
        return True

    def wait(self, timeout: float | None) -> bool:
        """Wait for a running child. False if it had to be abandoned."""
        if self._child is None:
            return True
        try:
            self._child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning(
                "[Tower][WorldBuilder] background solve still running after %ss; "
                "the final solve proceeds without it", timeout,
            )
            return False
        finally:
            self._reap()
        return True


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
        report = register(store, world_id, session_id)
        # Inside the guard, not after it. Persisting is not the safe part
        # of this: `placements_from_report` runs every placement through
        # `SegmentPlacement.__post_init__`, which raises ValueError on a
        # NaN scale or a non-unit quaternion -- precisely what a
        # degenerate Sim3 produces, and precisely the failure this guard
        # exists for. Measured with these three lines outside the try: a
        # raising `write_placements` gave exit code 1 and zero bytes of
        # report, losing a walk that had reconstructed perfectly well.
        manifest = store.read_derived_manifest(world_id) or {}
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
        help="accepted keyframes between background solves (0 = final solve only)",
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
    if args.follow_capture:
        frames = follow_capture(
            args.follow_capture,
            poll_seconds=args.poll_seconds,
            max_idle_polls=args.max_idle_polls,
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
    # keyframe_id -> raw frame path, for the global solver (see ObservedFrame).
    sources: dict = {}
    capture_dirs = [d for d in (args.follow_capture, args.frames) if d is not None]
    solver = None
    if args.solve and args.solve_every > 0:
        solver = BackgroundSolver(
            root=args.root.resolve(), world_id=world_id, session_id=session_id,
            every=args.solve_every, capture_dirs=capture_dirs,
        )
    for frame in frames:
        outcome = engine.observe(
            frame.payload,
            received_at=frame.received_at,
            source_seq=frame.source_seq,
            wire_seq=frame.wire_seq,
            tx_seq=frame.tx_seq,
        )
        if outcome.keyframe_id is None:
            continue
        accepted += 1
        since_rebuild += 1
        if frame.source_path is not None:
            sources[outcome.keyframe_id] = str(frame.source_path)
        # A finished background solve is worth a rebuild now: the solution
        # reaches the derived tree only through build(), and the wearer
        # should see the world snap together as soon as it is known.
        solve_landed = solver is not None and solver.finished()
        # Two keyframes is the minimum a two-view backend can say anything
        # about. Rebuilding on one would burn a build to produce an anchor
        # pose and nothing else.
        if (args.rebuild_every and since_rebuild >= args.rebuild_every and accepted >= 2) or (
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
            if solver is not None:
                solver.maybe_launch(store, accepted, sources)
    observe_seconds = time.perf_counter() - started
    summary = engine.stop_session()

    solve_report = None
    if args.solve:
        if solver is not None:
            solver.wait(args.solve_wait_seconds)
        solve_report = solve_session(
            store, world_id, session_id, capture_dirs=capture_dirs, sources=sources,
        )
        if solver is not None:
            solve_report["background_launches"] = solver.launches

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

    report = {
        "world_id": world_id,
        "session_id": session_id,
        "frame_source": frame_source,
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
        "build_seconds": round(build_seconds, 3),
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
    if args.register and not (solve_report or {}).get("solved"):
        report["registration"] = register_session(store, world_id, session_id)
    elif args.register:
        report["registration"] = {
            "attempted": False, "reason": "placements come from the global solve",
        }

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
