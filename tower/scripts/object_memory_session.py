#!/usr/bin/env python
r"""Remember which objects were in view, and when.

Runs in a SEPARATE PROCESS from the Tower, like every other cartridge
producer here. That is what makes this slice possible at all: Object
Memory has been considered blocked on whether a module lifecycle can
bound a synchronous model load, and a follower that tails a capture
journal needs no module lifecycle, no load timeout and no module slot.
The Tower's frame path pays nothing for this.

Frame sources:

  --frames DIR          a recorded capture directory, or any directory of
                        .jpg files. If the directory has a `frames.jsonl`
                        journal it is read instead of globbing, so the
                        real receipt times and source_seq survive; a bare
                        directory of jpegs has no timestamps and the
                        records say so.
  --follow-capture DIR  tail a capture the Tower is writing RIGHT NOW, so
                        objects are remembered while the wearer walks

WHAT THIS WILL AND WILL NOT REMEMBER

The classes `tower/object_memory/classes.py` admits, which is not one
list but two. `laptop` and `cell phone` are written on the detector's
word, because 36 of 36 inspected crops of them over the real corpus were
right. A dozen more -- `remote`, `backpack`, `bottle`, `cup` and the
rest -- are written only if a VERIFIER agrees, because inspection found
the same detector calling a ceiling fan `airplane` at 0.99 and a laptop
keyboard `remote` at 0.87. With `--verifier none`, which is the default,
nothing in the second list is ever written and this script behaves
exactly as the one that was physically validated.

`person` is excluded, and there is deliberately NO FLAG to widen the
list: re-admitting `person` commits the project to persisting a record
per detected bystander, which is an open ruling no human has settled
here. A command-line switch would let that decision happen by accident.

A record says a CATEGORY was visible over a span of time, with a
confidence. It is not a claim that the object is there now, not a claim
about WHICH laptop, and not a position in a room -- `spatial_ref` is null
and stays null.

The unit is a SIGHTING: a run of frames in which a class stayed in view,
broken by a gap of more than three seconds. A record is written once the
sighting is three frames old -- about a quarter of a second -- stamped
with the FIRST frame, so a killed session loses at most that. What the
sighting later becomes (its duration, its frame count, its strongest
look) is folded back into the same record rather than becoming a second
one. None of the scores is a calibrated probability.

`--retention-days` is recorded in the store's manifest at first append,
and every later read clamps to min(persisted, requested). A reader can
narrow that window; nothing a reader passes can widen it.

AND EACH RECORD KEEPS A PICTURE THIS CARTRIDGE OWNS.

`--keep-imagery`, on by default, writes one small filtered crop per
record under `<root>/keyframes/`. Without it a record's only picture is
a frame in `data/captures/`, which this cartridge does not own and whose
lifetime it does not set -- so the first thing that prunes captures
takes the picture out of every memory at once, which is a race rather
than a retention policy. The owned crop is pruned when the record
expires and deleted by `object_query.py --purge-all`.

It needs a face-detection model and FAILS CLOSED without one: no model
means no keyframes at all, never an unfiltered crop on disk, and this
script says so once at start rather than refusing quietly all walk.
Measured at ~11.7 KB a record, about 4.3 MB an hour of walking.

    .venv\Scripts\python.exe scripts/object_memory_session.py --frames data/captures/<id>
    .venv\Scripts\python.exe scripts/object_memory_session.py --follow-capture data/captures/<id>
"""

import argparse
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tower.artifact_paths import artifact_root_arg  # noqa: E402
from tower.capture import CaptureFollower, FRAMES_FILENAME  # noqa: E402
from tower.capture_workers import (  # noqa: E402
    ATTACH_MODE_FROM_NOW,
    ATTACH_MODE_FROM_START,
)
from tower.config import DEFAULT_OBSERVATION_ROOT  # noqa: E402
from tower.object_memory.detector import (  # noqa: E402
    SCORE_THRESHOLD,
    FixedDetector,
    TorchvisionDetector,
)
from tower.object_memory.engine import ObjectMemoryEngine  # noqa: E402
from tower.object_memory.imagery import FaceFilter  # noqa: E402
from tower.object_memory.keyframes import KeyframeStore  # noqa: E402
from tower.object_memory.relevance import (  # noqa: E402
    RelevancePolicy,
    recordable_classes,
)
from tower.object_memory.verification import (  # noqa: E402
    VerificationQueue,
)
from tower.object_memory.store import (  # noqa: E402
    DEFAULT_RETENTION_DAYS,
    ObservationStore,
)
from tower.storage import read_raw_jsonl  # noqa: E402

# THE SAME DEFAULT THE WEB PROCESS USES, IMPORTED RATHER THAN RESTATED.
#
# This line used to read `Path("data/object_memory")` while
# `tower/config.py` defaulted its observation root to None, and the two
# were never compared. The 2026-08-26 physical run is what that cost: a
# real walk was remembered into this directory, and every HTTP request
# answered 404 about it until an operator set an environment variable by
# hand. One constant, in the settings module, imported here.
DEFAULT_ROOT = Path(DEFAULT_OBSERVATION_ROOT)


def journal_frames(directory: Path):
    """A recorded capture, read through its journal.

    The journal is what carries `received_at` and `source_seq`. Globbing
    the image directory instead would throw both away and leave every
    record stamped with processing time, which makes the resample window
    measure the wrong thing entirely.
    """
    records, _ = read_raw_jsonl(directory / FRAMES_FILENAME)
    for record in records:
        relpath = record.get("relpath")
        if not relpath:
            continue
        path = directory / relpath
        if not path.exists():
            continue
        yield (
            path.read_bytes(),
            record.get("source_seq"),
            record.get("received_at"),
            relpath,
        )


def loose_frames(directory: Path):
    """Frames from a directory of jpegs. Validates EAGERLY -- see below.

    The split is the same one `world_build_session.py: follow_capture`
    needed, and for the same reason. A `def` containing `yield` is a
    generator function: calling it runs none of the body, so a check
    written as the first statement does not fire until something advances
    the generator -- which, in `main()`, is after the store and the model
    have been built. There the equivalent ordering minted a permanent
    empty world on every failed session.

    Latent rather than live here: `ObservationStore.__init__` and
    `engine.load()` create nothing on disk, so today this only wastes a
    model load. Fixed anyway, because "it is harmless because of what two
    other functions happen not to do" is not a property anyone maintains.
    """
    paths = sorted(directory.glob("*.jpg"))
    if not paths:
        raise SystemExit(f"no .jpg frames found under {directory}")
    return _loose_frames(paths)


def _loose_frames(paths):
    for index, path in enumerate(paths):
        # received_at None, not a fabricated interval: this source has no
        # clock, and the records must not pretend otherwise (Rule 3).
        yield path.read_bytes(), index, None, path.name


VERIFIERS = ("none", "owlv2")


def _resolve_device(requested: str) -> str:
    """A requested device, as one this host actually has.

    Same word and same rule as `TOWER_CV_DEVICE` and
    `cartridge_runtime._resolve_device`: **auto downgrades, cuda does
    not** -- an unnoticed downgrade turns a GPU deployment into a CPU one
    with a GPU label on it, which is worse than a failure.

    With one deliberate difference, and it is about what this process is.
    The Lab's version RAISES when `cuda` is unavailable, and raising
    there costs an experiment. Raising here costs a walk: this is a
    producer a person started from their phone, mid-session, and a Tower
    whose GPU disappeared under a driver update would answer their Start
    by remembering nothing at all. So an explicit `cuda` that cannot be
    honoured downgrades too -- and says so at WARNING, which is what
    `OwlV2Verifier` already did for the same reason. `auto` is silent
    about it because `auto` asked for whatever was there.
    """
    if requested == "cpu":
        return "cpu"

    import torch

    available = torch.cuda.is_available()
    if available:
        return "cuda"
    if requested == "cuda":
        print(
            "[Tower][ObjectMemory] cuda was asked for and torch reports none "
            "on this host; running on cpu",
            file=sys.stderr,
            flush=True,
        )
    return "cpu"


def _build_verifier(name: str, device: str):
    """Whatever may second-guess a detector label, or None.

    A NAME rather than a flag, because the answer is a model identifier
    and a boolean could never have become one. `none` returns None, and
    None is what makes `verification_available` False -- which keeps the
    `verify` tier unreachable on a Tower with no semantic model, rather
    than reachable and always refused.

    `none` is the DEFAULT, and that is a deliberate conservatism rather
    than a lack of confidence in the alternative. `owlv2` measured ~93%
    acceptance of correct labels and ~94% rejection of wrong ones over
    94 human-labelled crops -- but those 94 crops are from one home, and
    turning it on downloads ~600 MB of weights and takes ~620 MB of VRAM
    on a GPU this cartridge shares. One environment variable enables it;
    the evidence for doing so is in
    `docs/superpowers/research/2026-08-27-object-memory-corpus-precision.md`.
    """
    if name == "none":
        return None
    if name == "owlv2":
        try:
            from tower.object_memory.classes import prompt_for, verifier_vocabulary
            from tower.object_memory.verification import OwlV2Verifier

            return OwlV2Verifier(
                device=device,
                vocabulary=verifier_vocabulary(),
                prompt_for=prompt_for,
            )
        except Exception as exc:  # noqa: BLE001
            # SOFT, and only in the narrowing direction.
            #
            # `owlv2` became the default on 2026-08-29, which means a
            # host that has never fetched the weights, or has no
            # `transformers`, is now a host where the DEFAULT
            # configuration can fail. A producer that died here would
            # answer a wearer's Start with a walk that remembered
            # nothing and a warning in a log they are not reading.
            #
            # Running without it records two classes instead of
            # fourteen. `recordable_classes` in the report says which,
            # and `verifier` in the report says `none`, so the run
            # cannot claim a verification tier it did not have.
            print(
                "[Tower][ObjectMemory] the owlv2 verifier could not be "
                f"loaded ({exc.__class__.__name__}: {exc}); continuing with "
                "no verifier, which records only the classes the detector "
                "is trusted on",
                file=sys.stderr,
                flush=True,
            )
            return None
    raise SystemExit(
        f"unknown verifier {name!r}; this build offers {', '.join(VERIFIERS)}"
    )


def _build_keyframe_store(root, keep_imagery: bool):
    """`(KeyframeStore, FaceFilter)`, or `(None, None)`, and one loud line.

    Both or neither. A keyframe store with no usable filter would write
    nothing and count a refusal per sighting, which is the correct
    behaviour and a terrible thing to discover at the end of a walk -- so
    this reports the condition ONCE, before the first frame, next to the
    device and verifier lines an operator is already reading.

    The filter is still handed to the engine even when it is
    unavailable, rather than being replaced with None. `KeyframeStore.
    write` refuses an unavailable filter by the same branch it refuses a
    missing one, and the counter says `display-filter-unavailable` either
    way -- a report that said "no keyframe store" when the truth is "no
    model on this host" would send an operator to the wrong setting.
    """
    if not keep_imagery:
        # Not a degraded configuration. It reproduces exactly the
        # behaviour that shipped, where this cartridge persisted no
        # pixels at all, and the report says `keep_imagery: false` so a
        # run cannot be mistaken for one that tried and failed.
        return None, None

    face_filter = FaceFilter()
    if not face_filter.available:
        print(
            "[Tower][ObjectMemory] KEEPING NO IMAGERY THIS RUN: "
            f"{face_filter.unavailable_reason}. Every record will still be "
            "written and will still point at its capture frame, but none "
            "will get a picture of its own -- so when the capture is "
            "pruned or deleted, those memories lose their pictures. "
            "Nothing unfiltered is ever written instead.",
            file=sys.stderr,
            flush=True,
        )
    return KeyframeStore(root), face_filter


class _StopRequest:
    """A stop asked for from outside this process, and whether it arrived.

    THE PROBLEM THIS SOLVES, AND IT WAS COSTING REAL MEMORIES.

    Pressing Stop (or Pause) in the app reaches
    `CartridgeSession._detach` -> `CaptureWorkerSupervisor.detach` ->
    `Popen.terminate()`. On Windows that is `TerminateProcess`: no
    unwinding, no `finally`, no `atexit`. So `engine.release()` -- which
    is the ONLY thing that calls `finish()` on the open sightings, writes
    the ones that matured, and refreshes the duration, `frame_count` and
    `best_*` of the ones already on disk -- never ran.

    What a wearer lost by pressing the button they were told to press:
    every sighting still open at that instant. A laptop that had been in
    view for the whole walk and was still in view when they stopped is
    exactly the sighting that is open, so this was biased towards losing
    the MOST-seen objects rather than the least.

    The old detach grace was zero, and its comment said why: "nothing
    SIGNALS the producer: it is a follower tailing a journal that is
    still being written, so it has no reason to stop and never does. The
    wait measured 5.01 seconds every single time." That was true, and it
    was true because of this file, not because of the supervisor. Now
    something does signal it, the wait ends when the work is done, and
    the grace is worth having again.

    WHY A FLAG AND NOT WORK IN THE HANDLER.

    A signal handler runs between bytecodes on whatever thread the
    interpreter chooses. `engine.release()` takes the store's writer lock
    and joins the verification queue's worker; doing that from a handler
    is how a shutdown deadlocks. So the handler sets a bool and returns,
    the frame loop notices within one frame, and the ordinary
    `finally: engine.release()` does the work on the main thread exactly
    as it does when a capture closes by itself.

    WHICH SIGNALS.

    `SIGTERM` and `SIGINT` on POSIX. On Windows `SIGTERM` is not
    deliverable between processes at all -- `Popen.terminate()` is
    `TerminateProcess` and never becomes a Python signal -- so the
    supervisor sends `CTRL_BREAK_EVENT` to the process group it
    deliberately created for each worker, and that arrives here as
    `SIGBREAK`. Registering all four is cheap and means neither side has
    to be right about the platform.
    """

    def __init__(self) -> None:
        self.requested = False
        self.signal_name: str | None = None

    def install(self, *, watch_stdin: bool = False) -> None:
        if watch_stdin:
            self._watch_stdin()
        for name in ("SIGTERM", "SIGINT", "SIGBREAK"):
            handler_signal = getattr(signal, name, None)
            if handler_signal is None:
                continue
            try:
                signal.signal(handler_signal, self._handle)
            except (ValueError, OSError):
                # Not the main thread, or a signal this platform will not
                # let a process take. A producer that cannot be asked
                # nicely still gets terminated; it just loses the flush,
                # which is where this started.
                continue

    def asked_for(self) -> bool:
        """A callable, so a generator deep in `tower.capture` can ask
        without importing anything from this script."""
        return self.requested

    def _handle(self, signum, _frame) -> None:
        self.requested = True
        self.signal_name = signal.Signals(signum).name

    def _watch_stdin(self) -> None:
        """Stop when the parent closes the pipe it is holding.

        THIS IS THE CHANNEL THAT ACTUALLY WORKS ON WINDOWS.

        A console control event needs a console, and the Tower does not
        reliably have one: under a pseudoconsole -- an editor's integrated
        terminal, a CI runner, a service --
        `GenerateConsoleCtrlEvent` reports success and nothing arrives. A
        pipe needs nothing. `CaptureWorkerSupervisor` keeps the write end
        for any spec that set `stop_via_stdin`, and closing it is an EOF
        this thread cannot miss.

        A DAEMON thread, and it must be: the read blocks until the parent
        closes, and a non-daemon thread blocked on that would keep this
        process alive after the frame loop had finished by itself, which
        is the ordinary ending.

        The read is one byte, not a line. Nothing is ever WRITTEN to this
        pipe -- the request is the close -- so waiting for a newline would
        wait for something that is never coming, and a parent that did
        write something has still said the only thing this pipe can mean.
        """

        def wait_for_close() -> None:
            # The RAW descriptor, never `sys.stdin.buffer.read(1)`. A read
            # through the buffered object holds that object's lock for as
            # long as it blocks, and when this process exits NORMALLY --
            # the capture closed, the supervisor still holding the pipe --
            # interpreter shutdown tries to take the same lock to close
            # stdin and aborts: "Fatal Python error: _enter_buffered_busy:
            # could not acquire lock for <_io.BufferedReader name='<stdin>'>",
            # a non-zero exit for a run that succeeded. `os.read` takes no
            # Python lock; the pending ReadFile dies with the process.
            try:
                fd = sys.stdin.fileno() if sys.stdin is not None else None
            except (AttributeError, ValueError, OSError):
                return
            if fd is None:
                return
            try:
                os.read(fd, 1)
            except Exception:
                # A closed or unreadable pipe is itself the request. The
                # alternative -- treating an unreadable stdin as "keep
                # going" -- is a producer nobody can stop.
                pass
            self.requested = True
            if self.signal_name is None:
                self.signal_name = "stdin-closed"

        threading.Thread(
            target=wait_for_close, name="object-memory-stop-watch", daemon=True
        ).start()

    def bounded(self, frames):
        """`frames`, ending at the next frame after a stop was asked for.

        THE SECOND OF TWO CHECKS, AND THE ONLY ONE ON A REPLAY.

        The primary check is inside `CaptureFollower.follow`'s poll loop,
        which is where a live producer spends a quiet walk. This one is
        the only check a `--frames` replay has: a replay reads a
        directory, never polls, and would otherwise run to the end of the
        corpus after being asked to stop.

        It is also the reason a stop during a LIVE run cannot be missed
        by one frame -- the follower may already have yielded before the
        flag was set.

        Wrapping the generator rather than checking inside the loop body
        keeps the check on ONE path: the loop already has a `--limit`
        break and a `finally`, and a second `break` in the body is a
        second place for a future edit to return early past the flush.
        """
        for frame in frames:
            if self.requested:
                return
            yield frame


def _prewarm_native_libraries() -> None:
    """Load the OpenBLAS-backed native stack BEFORE any watcher thread runs.

    THE ZERO-FRAME DEADLOCK, AND WHY THIS ONE LINE PREVENTS IT.

    The physical run on 2026-09-06 recorded a healthy capture -- 244
    frames, ~21 s -- and this producer, attached `from-start`, observed
    NONE of them. It was not slow and it was not misattached: it was
    deadlocked inside its own model load and never reached the frame
    loop. `frames_observed: 0`, then killed at Stop because it could not
    exit.

    Reproduced deterministically on this host and dissected with py-spy
    (`--native`). Two threads, one lock:

      * the main thread is inside `transformers` loading the OWLv2
        verifier, which imports `scipy.linalg`, which loads
        `libscipy_openblas*.dll`. OpenBLAS spins up its thread pool in
        `DllMain`, so the load sits under the Windows loader lock
        (`LdrpDrainWorkQueue` / `ZwWaitForAlertByThreadId`);
      * the `object-memory-stop-watch` daemon thread is blocked in a
        synchronous `ReadFile` on the stdin pipe the supervisor holds
        (`_StopRequest._watch_stdin`).

    A thread parked in a blocking pipe read while the loader is bringing
    up a DLL that creates threads is a documented Windows loader-lock
    hazard, and here it is a hard hang: 0% CPU, forever. It is invisible
    to the whole test suite because every subprocess test pins
    `--verifier none` to avoid downloading weights -- so nothing ever
    loaded transformers/scipy in a spawned worker while the stdin watcher
    was armed. Production defaults to `owlv2`, so production hit it and
    the tests could not.

    The fix is ordering, not luck. Loading `scipy.linalg` HERE, on the
    main thread, before `_StopRequest` arms the watcher, brings OpenBLAS
    up single-threaded with no pipe-reader parked behind the loader. When
    `transformers` imports `scipy.linalg` later it is already in
    `sys.modules` and no DLL work happens -- so the watcher can stay armed
    for the whole run, including the load, and a Stop pressed during the
    ~8 s load is still honoured.

    Contained: a host without `scipy` cannot load `owlv2` either
    (`transformers` needs it), so it will run with no verifier regardless,
    and there is no OpenBLAS-via-scipy to preload. Anything else that
    fails here is a warning, never a refusal to start -- warming a library
    must not be the thing that stops a wearer's session from beginning.
    """
    try:
        import scipy.linalg  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(
            "[Tower][ObjectMemory] could not pre-warm the native numerical "
            f"stack ({exc.__class__.__name__}: {exc}); continuing. If a "
            "verifier is configured and this host is Windows, watch for a "
            "worker that loads its model and never observes a frame.",
            file=sys.stderr,
            flush=True,
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Record which object categories were visible, and when."
    )
    parser.add_argument(
        "--root", type=artifact_root_arg, default=str(DEFAULT_ROOT)
    )
    parser.add_argument("--frames", type=Path, default=None)
    parser.add_argument("--follow-capture", type=Path, default=None)
    parser.add_argument(
        "--detector",
        choices=("ssdlite320", "none"),
        default="ssdlite320",
        help=(
            "'none' runs the whole pipeline over no detections, which "
            "exercises the producer without downloading a model and "
            "honestly remembers nothing."
        ),
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument(
        "--verifier",
        default="none",
        help=(
            "What may second-guess a detector label. 'none' (the default) "
            "agrees with nothing, so only the classes measured reliable on "
            "the detector's own word are written. 'owlv2' loads "
            "google/owlv2-base-patch16-ensemble (~600 MB, Apache-2.0) and "
            "unlocks the verify tier."
        ),
    )
    parser.add_argument(
        "--verifier-device",
        choices=("auto", "cpu", "cuda"),
        default="cuda",
        help=(
            "Where the verifier runs. CUDA by default even though the "
            "detector defaults to CPU, and deliberately: measured 128 ms "
            "a crop on this GPU against 2,473 ms on this CPU, and putting "
            "the two stages on different devices is what keeps a 2.5-second "
            "burst off the cores the detector is using."
        ),
    )
    parser.add_argument("--score-threshold", type=float, default=SCORE_THRESHOLD)
    parser.add_argument(
        "--min-score",
        type=float,
        default=RelevancePolicy.min_score,
        help="Detections below this are seen but never remembered.",
    )
    parser.add_argument(
        "--gap-seconds",
        type=float,
        default=RelevancePolicy.gap_seconds,
        help=(
            "How long a class may be out of view before its next "
            "appearance counts as a new sighting."
        ),
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=RelevancePolicy.min_frames,
        help=(
            "How many frames a sighting must last before it is written. A "
            "third of the sightings in the real corpus are shorter than "
            "three frames, and those are flickers."
        ),
    )
    parser.add_argument(
        "--retention-days",
        type=float,
        default=DEFAULT_RETENTION_DAYS,
        help=(
            "Forget observations older than this. 0 means keep forever. "
            "Recorded in the store manifest at first append, and every "
            "later read is clamped to it."
        ),
    )
    parser.add_argument(
        "--keep-imagery",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Give each record a small filtered crop of its OWN, under the "
            "observation root, deleted when the record expires. On by "
            "default. Without it a record's only picture is a frame in "
            "data/captures/, which this cartridge does not own and whose "
            "lifetime it does not set -- so the first thing that prunes "
            "captures takes the picture out of every memory at once. "
            "Measured at ~11.7 KB a record, about 4.3 MB an hour of "
            "walking. Needs a face-detection model; without one nothing "
            "is written and this says so at start."
        ),
    )
    parser.add_argument(
        "--attach-mode",
        choices=(ATTACH_MODE_FROM_START, ATTACH_MODE_FROM_NOW),
        default=ATTACH_MODE_FROM_START,
        help=(
            "Only meaningful with --follow-capture. 'from-start' reads the "
            "whole capture, which is right for a producer attached when the "
            "recording opened. 'from-now' skips whatever the journal "
            "already holds: a producer attached three minutes into a walk "
            "was not asked for the first three minutes, and reading them "
            "would be a consent decision this script has no standing to "
            "make."
        ),
    )
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    parser.add_argument("--max-idle-polls", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Stop after N frames.")
    parser.add_argument(
        "--stop-on-stdin-close",
        action="store_true",
        help=(
            "Finish the run when the parent closes this process's stdin. "
            "How CaptureWorkerSupervisor asks for a clean stop on a host "
            "with no console, where a signal cannot be delivered."
        ),
    )
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)

    chosen = [
        name
        for name, value in (
            ("--frames", args.frames),
            ("--follow-capture", args.follow_capture),
        )
        if value
    ]
    if len(chosen) != 1:
        parser.error("exactly one of --frames or --follow-capture is required")

    # BEFORE the stdin watcher is armed, and that order is load-bearing on
    # Windows: warming the OpenBLAS-backed native stack while no thread is
    # parked in a blocking pipe read is what stops the loader-lock deadlock
    # that made a real walk observe zero frames. See
    # `_prewarm_native_libraries`.
    _prewarm_native_libraries()

    # Installed before the follower is built and before the first frame
    # is read, because the poll loop it arms is the thing being armed.
    stop_request = _StopRequest()
    stop_request.install(watch_stdin=args.stop_on_stdin_close)

    if args.follow_capture:
        if not args.follow_capture.exists():
            raise SystemExit(f"no capture directory at {args.follow_capture}")
        follower = CaptureFollower(
            args.follow_capture,
            poll_seconds=args.poll_seconds,
            start_at_end=args.attach_mode == ATTACH_MODE_FROM_NOW,
        )
        frames = (
            (frame.raw_bytes, frame.source_seq, frame.received_at, frame.relpath)
            for frame in follower.follow(
                max_idle_polls=args.max_idle_polls,
                # Asked inside the poll loop, which is where this process
                # spends a quiet walk. See `_StopRequest`.
                should_stop=stop_request.asked_for,
            )
        )
        session_id = args.follow_capture.name
        source = "live-capture"
        timing = "capture-journal"
        attach_mode = args.attach_mode
    else:
        if not args.frames.exists():
            raise SystemExit(f"no frame directory at {args.frames}")
        if (args.frames / FRAMES_FILENAME).exists():
            frames = journal_frames(args.frames)
            timing = "capture-journal"
        else:
            frames = loose_frames(args.frames)
            timing = "none"
        session_id = args.frames.name
        source = "recorded-frames"
        # A replay reads what it was pointed at. There is no live journal
        # to arrive late to, so the flag has nothing to describe and the
        # report says so rather than reporting a default that would look
        # like a decision.
        attach_mode = None

    # Resolved HERE rather than in the web process, which deliberately
    # imports no torch. Printed before anything is loaded so a person
    # reading the Tower console sees what this run is actually using,
    # rather than the word "auto" and a guess.
    device = (
        _resolve_device(args.device) if args.detector == "ssdlite320" else args.device
    )
    verifier_device = (
        _resolve_device(args.verifier_device) if args.verifier != "none" else args.verifier_device
    )
    print(
        f"[Tower][ObjectMemory] detector={args.detector} device={device} "
        f"verifier={args.verifier} verifier_device={verifier_device} "
        f"root={args.root}",
        file=sys.stderr,
        flush=True,
    )

    detector = (
        TorchvisionDetector(
            score_threshold=args.score_threshold, device=device
        )
        if args.detector == "ssdlite320"
        else FixedDetector()
    )
    retention = None if args.retention_days <= 0 else args.retention_days * 86400.0
    store = ObservationStore(args.root, retention_seconds=retention)
    verifier = _build_verifier(args.verifier, verifier_device)
    # Synchronous for a replay, threaded for a live follow. A replay has
    # no reason to be asynchronous and every reason to be deterministic;
    # a live session must not stall the frame path on a model.
    verification = (
        None
        if verifier is None
        else VerificationQueue(verifier, workers=0 if args.frames else 1)
    )
    policy = RelevancePolicy(
        min_score=args.min_score,
        gap_seconds=args.gap_seconds,
        min_frames=args.min_frames,
        verification_available=verifier is not None,
    )
    # ONE LINE, AT START, RATHER THAN A REFUSAL PER SIGHTING.
    #
    # A Tower with no face-detection weights writes no keyframes at all,
    # by design -- `KeyframeStore.write` fails closed. The failure mode
    # that matters is not the refusal, it is the SILENCE: a whole walk of
    # them, counted in `keyframes_refused` at the end and visible to
    # nobody in between, on a run whose operator believes their memories
    # are keeping their pictures. So the condition is stated once, before
    # the first frame, in the same place the device and verifier are.
    keyframes, face_filter = _build_keyframe_store(args.root, args.keep_imagery)
    engine = ObjectMemoryEngine(
        store,
        detector,
        policy=policy,
        verification=verification,
        keyframes=keyframes,
        face_filter=face_filter,
        # The capture id, which is a FRAME identity owned by shared
        # transport -- not a World Builder session. This cartridge cannot
        # see a world and must not imply it can.
        session_id=session_id,
        source="glasses-camera",
    )

    started = time.perf_counter()
    engine.load()
    try:
        for index, frame in enumerate(stop_request.bounded(frames)):
            if args.limit is not None and index >= args.limit:
                break
            payload, source_seq, received_at, relpath = frame
            engine.observe(
                payload,
                received_at=received_at,
                source_seq=source_seq,
                relpath=relpath,
            )
    finally:
        engine.release()
    elapsed = time.perf_counter() - started

    pruned = store.prune_expired()

    report = {
        "root": str(args.root),
        "frame_source": source,
        "session_id": session_id,
        "detector": detector.name,
        "device": device if detector.name != "fixed" else None,
        # What RAN. `args.verifier` is what was asked for, and the two
        # differ exactly when the weights could not be loaded -- which is
        # the case a report must not paper over, because it is the
        # difference between fourteen recordable classes and two.
        "verifier": args.verifier if verifier is not None else "none",
        "verifier_requested": args.verifier,
        "verifier_device": verifier_device if verifier is not None else None,
        "timing_source": timing,
        "attach_mode": attach_mode,
        # Whether records got pictures of their own, and what actually
        # filtered them. `keep_imagery` is what was ASKED FOR;
        # `keyframe_filter` is what RAN, and is null exactly when this
        # host has no face-detection model -- which is the case a report
        # must not paper over, because it is the difference between a
        # memory that keeps its picture for thirty days and one whose
        # picture belongs to a capture directory. The per-outcome
        # counters are `keyframes_written` and `keyframes_refused`, which
        # arrive with the engine's counters below.
        "keep_imagery": bool(args.keep_imagery),
        "keyframe_filter": (
            face_filter.label
            if face_filter is not None and face_filter.available
            else None
        ),
        # What this run could have written, which is narrower than what
        # the store would have accepted: the verify tier is only in reach
        # when something can second-guess the detector.
        "recordable_classes": list(recordable_classes(verifier is not None)),
        # Every counter the engine kept, including why detections did NOT
        # become observations. "wrote 11 records" means nothing without
        # "and declined 4,000, nearly all of them for classes it has no
        # evidence it can read".
        **engine.counters(),
        "stored_observations": len(store.all_observations()),
        "pruned_expired": pruned,
        "retention_days": args.retention_days,
        # Why the run ended. "asked" is a Pause or a Stop reaching the
        # producer as a signal; "frames-ended" is the capture closing or
        # the idle bound expiring. A report that does not say which turns
        # "the walk ended" and "a person stopped it" into the same line.
        "stopped_because": (
            stop_request.signal_name if stop_request.requested else "frames-ended"
        ),
        "seconds": round(elapsed, 3),
        "ms_per_frame": round(elapsed * 1000 / max(engine.frames_observed, 1), 3),
    }

    # A RUN THAT DID NOTHING AND WAS ASKED TO DO NOTHING SAYS SO.
    #
    # `--stop-on-stdin-close` means "the parent is holding my stdin, and
    # closing it is a stop request". `CaptureWorkerSupervisor` honours
    # that by spawning with `stdin=subprocess.PIPE` and keeping the write
    # end, so the flag and the pipe are set together in `main.py` and a
    # Tower cannot get this wrong.
    #
    # A HUMAN CAN. Run this by hand from anything that hands a child a
    # null or already-closed stdin -- a scheduler, a service, a script
    # using `subprocess.run(capture_output=True)` -- and the watcher sees
    # EOF before the first frame, the run ends immediately, and the
    # report is a wall of zeroes with `stopped_because: stdin-closed`
    # buried in it. That happened while writing this, cost twenty
    # minutes, and the fix is not to guess at the cause in code -- a
    # supervisor closing the pipe promptly is a legitimate stop that
    # looks identical -- but to NAME the likely reason once, here, where
    # somebody is already reading.
    if report["stopped_because"] == "stdin-closed" and engine.frames_observed == 0:
        print(
            "[Tower][ObjectMemory] this run ended before it read a single "
            "frame, because stdin reached EOF and --stop-on-stdin-close was "
            "passed. If you started this by hand, that flag expects a parent "
            "holding the other end of a pipe; drop it, or spawn with "
            "stdin=PIPE and keep it open.",
            file=sys.stderr,
            flush=True,
        )

    if args.format == "json":
        print(json.dumps(report, indent=2))
    else:
        for key, value in report.items():
            print(f"{key:22s} {value}")
        print(
            "\nOBSERVED, NOT LOCATED: a category was visible in a frame. "
            "There is no position in a room here -- spatial_ref is null."
        )
        print(
            "Only "
            + ", ".join(report["recordable_classes"])
            + " could be remembered on this run. `person` is excluded on "
            "purpose; see object_memory.classes.EXCLUDED_CLASSES."
        )
        print("Times are tower-receipt, never on-glasses capture time.")
        print(
            "Records point back at data/captures/ by session and frame. "
            "That imagery is governed by capture retention, not by this "
            "store's."
        )
        if report["keyframe_filter"]:
            print(
                f"{report['keyframes_written']} record(s) also kept a small "
                f"crop of their own under {args.root}\\keyframes, filtered by "
                f"{report['keyframe_filter']} before being written. Those ARE "
                "governed by this store's retention: they are pruned when the "
                "record expires and deleted by --purge-all. The filter names "
                "what ran; it is not a claim that faces were removed."
            )
        elif args.keep_imagery:
            print(
                "NO record kept a crop of its own: this host has no "
                "face-detection model, and nothing unfiltered is ever "
                "written. Every picture here belongs to data/captures/ and "
                "disappears with it."
            )
    return 0


def _run_and_exit() -> None:
    """Run the producer, then leave the process HARD -- on EVERY path.

    A CLEAN FINISH THAT NO LONGER LOOKS LIKE A CRASH.

    The stdin-stop watcher (`_StopRequest._watch_stdin`) is a daemon thread
    parked in a blocking native `ReadFile` for as long as the parent holds
    the write end of the pipe -- which it does for the whole of the
    ordinary ending, where the walk stops because the capture closed rather
    than because anyone pressed Stop. Letting the interpreter FINALIZE with
    that thread still blocked races CUDA/torch teardown and access-violates
    (`0xC0000005`) on Windows, so `CaptureWorkerSupervisor.reap` logs a walk
    that finished as `EXITED 3221225477`. Measured on this host with the
    verifier OFF as well as on -- it is the blocked reader and the GPU
    teardown, not anything this cartridge computes.

    `os._exit` skips that finalization, and it must be reached on EVERY exit,
    not only success. The watcher is armed (line ~669) BEFORE `engine.load()`
    builds the CUDA context, so an exception from `engine.load()` (a CUDA
    OOM, a weights download that fails -- ordinary errors), from the frame
    loop, or from building the report escapes into the SAME finalization with
    the SAME blocked watcher, and the crash then masks the real error behind
    exit 3221225477. An earlier version of this function only guarded the
    success path and claimed errors "happen before a model or a watcher
    exists"; a reviewer showed that was false. So every ending goes through
    `os._exit` here.

    The error is not lost. A real exception's traceback is printed to stderr
    -- which the Tower console inherits -- before the hard exit, and the code
    is a plain `1`, not a crash code. A `SystemExit` (argument parsing, or an
    explicit one) keeps its exact code and, for a message, prints it as
    Python would; those happen before a model or watcher exists, so a normal
    finalize would be safe too, but a hard exit there costs nothing and keeps
    one rule instead of two. The report and flush are unchanged: by the time
    a clean `main()` returns, `finally: engine.release()` has already flushed.
    """
    try:
        code = main()
    except SystemExit as exc:
        # Preserve SystemExit semantics without running finalization.
        sys.stdout.flush()
        sys.stderr.flush()
        value = exc.code
        if isinstance(value, int):
            os._exit(value)
        if value is None:
            os._exit(0)
        # A string (or other) code: Python's default handler prints it to
        # stderr and exits 1. Do the same, since catching it skipped that.
        print(value, file=sys.stderr, flush=True)
        os._exit(1)
    except BaseException:  # noqa: BLE001
        # Any real failure after the watcher and the CUDA context exist.
        # Show it -- masking it behind the teardown crash is the whole
        # defect -- then leave hard so the crash cannot happen at all.
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code if isinstance(code, int) else 0)


if __name__ == "__main__":
    _run_and_exit()
