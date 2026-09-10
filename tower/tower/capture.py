"""Recording raw frames to disk, as an explicit dataset session.

SHARED infrastructure, not a cartridge's. It lives here rather than under
tower/world_builder/ because the transport arms it and any cartridge might
want it -- Text/Document will want OCR-quality stills, Object Memory
occasional high-value frames. Versioning a recording made by the SHARED
transport with a CARTRIDGE's schema constant would mean a geometry-driven
schema bump invalidating capture journals that have nothing to do with
geometry, so this module owns its own version and time basis.

Separate from any keyframe corpus on purpose. A mapper persists a SELECTED
subset of frames; this records everything, because the two jobs it exists
for both need unselected frames:

- camera calibration, which needs board views chosen by a human holding a
  board, not by a parallax policy;
- re-running the V0.9.3 experiments on real DAT footage, which is the
  standing acceptance gate on every dataset-based conclusion the project
  has drawn.

Neither is possible today: a filesystem search found no stored Ray-Ban
footage anywhere, which is why this exists at all.

It is OFF BY DEFAULT and bounded in both duration and bytes. Under
06-PRIVACY-DATA this is an Explicit Dataset-Recording Session: manually
started and stopped, bounded, purgeable, and visibly recording. It is not
incidental capture, and it must never become the default path.
"""

import json
import logging
import os
import pathlib
import time
from dataclasses import dataclass, field

from tower.storage import (
    append_jsonl,
    new_id,
    read_json_closed,
    read_raw_jsonl,
    write_json_atomic,
)

# This module's own schema version and clock label. Deliberately NOT
# imported from a cartridge: a recording made by shared transport must not
# be versioned by a consumer's schema.
CAPTURE_SCHEMA_VERSION = 1
TIME_BASIS = "tower-receipt"

END_REASON_STOP = "stop"
END_REASON_DISCONNECT = "disconnect"
END_REASON_BOUNDED_LIMIT = "bounded_limit"

# How long after a capture ends by DISCONNECT a new `stream_start` is
# treated as the same walk continuing.
#
# `handoff.md` 6.4: iOS retries on a [0.5, 1, 2, 4, 8] s backoff and gives
# up after five attempts, which takes roughly 45 s because each attempt can
# burn a 6 s pong timeout. 90 s is comfortably past the point where iOS has
# stopped trying, so anything arriving later is genuinely a new walk.
RESUME_GRACE_SECONDS = 90.0

# How long a follower tolerates a capture whose manifest is still OPEN but
# has stopped growing, expressed as polls at the default 0.25 s interval.
#
# `CaptureFollower.follow` has always accepted this bound and its docstring
# has always promised it -- "Bounded by construction (Rule 15): a capture
# whose manifest never closes -- a crashed recorder -- ends the follow after
# `max_idle_polls` quiet polls rather than waiting forever". Neither worker
# spec in `main.py` passed it, so the parameter defaulted to None and the
# promise was never kept. A producer whose Tower died without closing the
# manifest polled that directory forever. On Windows that is the ordinary
# way a Tower dies: `terminate()` is `TerminateProcess`, which runs no
# lifespan and closes nothing.
#
# 900 s, and the number is chosen against what a LEGITIMATE silence can be
# rather than picked for roundness. Frames arrive at ~12 fps, so a live
# capture is never quiet for long; every ordinary interruption CLOSES the
# capture and is handled by the successor path instead. The longest
# plausible silence with the manifest still open is the window in which the
# phone has stopped sending and uvicorn has not yet noticed the dead
# socket, which `stop()` records as 20-40 s. This is an order of magnitude
# above that, and ten times `RESUME_GRACE_SECONDS`, which is this file's own
# allowance for a walk that goes quiet and comes back.
#
# The failure directions are asymmetric and that is why the bound is
# generous: firing early costs a wearer the rest of a mapped walk, while
# firing late only costs an idle process a few more minutes.
IDLE_FOLLOW_TIMEOUT_SECONDS = 900.0
DEFAULT_FOLLOW_POLL_SECONDS = 0.25
DEFAULT_MAX_IDLE_POLLS = int(
    IDLE_FOLLOW_TIMEOUT_SECONDS / DEFAULT_FOLLOW_POLL_SECONDS
)

logger = logging.getLogger(__name__)

CAPTURE_FILENAME = "capture.json"
FRAMES_FILENAME = "frames.jsonl"
FRAMES_DIRNAME = "frames"


@dataclass(frozen=True)
class CaptureLimits:
    """Hard bounds. Rule 15 -- no unbounded operation on the live path."""

    # Sized for the session length the product actually asks for.
    #
    # 900 s -- fifteen minutes -- was the bound until an adversarial review
    # pointed out what it does to the stated target of "potentially 20-30
    # minute sessions". At the bound the recorder stops ITSELF, cleanly,
    # and the warning it logs says a follower "will see the capture close
    # exactly as if it were" a disconnect. So a thirty-minute walk recorded
    # fifteen minutes, the builder finalised a world at the halfway point,
    # and -- since a closed capture now reads as an ordinary wearer stop --
    # it did so under the label `stop`. Silent truncation that looks like
    # success is the worst shape this bug could have taken.
    #
    # 2400 s is forty minutes: the target plus a third, so a wearer who
    # goes long is not truncated by a round number. It is still a BOUND,
    # which is the point of Rule 15 -- an unbounded recorder on the live
    # path is how a disk fills during a walk.
    max_seconds: float = 2400.0
    # The byte bound has to move with it or it becomes the binding one.
    # Measured on the 2026-09-09 captures: 2,865 frames over 7 captures,
    # mean 24.6 KB per recorded JPEG at 360x640. Forty minutes at the
    # measured 12 fps is ~28,800 frames, ~708 MB. 2 GiB leaves room for a
    # larger frame without the bound arriving unannounced mid-walk.
    max_bytes: int = 2_147_483_648


@dataclass
class CaptureStatus:
    capture_id: str
    started_at: float
    frames_written: int = 0
    bytes_written: int = 0
    ended_at: float | None = None
    end_reason: str | None = None
    rejected: dict = field(default_factory=dict)

    @property
    def is_open(self) -> bool:
        return self.ended_at is None


class CaptureRecorder:
    """Writes raw frames plus a journal into one capture directory.

    Every persisted pixel passes through `write_frame`. That single choke
    point is deliberate: whatever redaction policy is eventually chosen,
    it becomes a change to one function rather than an archaeology
    exercise across the frame path. This decides no policy -- it makes one
    implementable.
    """

    def __init__(self, root, limits: CaptureLimits | None = None, clock=time.time):
        self._owner = None
        # The last capture this recorder closed because a socket dropped,
        # and when. Only ever used to LINK a successor to it; never to
        # reopen it.
        self._interrupted: tuple[str, float] | None = None
        self._root = pathlib.Path(root)
        self._limits = limits or CaptureLimits()
        self._clock = clock
        self._status: CaptureStatus | None = None

    @property
    def status(self) -> CaptureStatus | None:
        return self._status

    @property
    def is_recording(self) -> bool:
        return self._status is not None and self._status.is_open

    def capture_dir(self, capture_id: str):
        return self._root / "captures" / capture_id

    @property
    def owner(self):
        """Who armed the recording, or None.

        A recorder is a process-global object shared by every connection,
        but a recording belongs to the connection whose `stream_start`
        opened it. Without that, one connection's teardown stops another
        connection's capture -- see `stop`.
        """
        return self._owner

    def resumable_capture(self) -> str | None:
        """The capture a new `stream_start` should declare itself a successor to.

        `handoff.md` 9.3 makes a repeated `stream_start` on a fresh
        connection the EXPECTED case on this link, not an edge case: iOS
        re-opens the stream bracket after a reconnect while the camera is
        still running, and `seq` continues from where it left off. Tower
        must treat that as one walk.

        This does NOT reopen the old capture. Reopening would leave a
        capture that can sit open forever if the phone never comes back --
        and a follower waiting on it would never return, while the result
        channel reported `receiving` for eternity. The old capture stays
        closed and durable; the NEW one records that it continues it.

        Lineage is therefore decided by the WRITER, which knows, and
        written down. A reader is never asked to guess which of several
        capture directories continues its own.
        """
        if self._interrupted is None:
            return None
        capture_id, ended_at = self._interrupted
        if (self._clock() - ended_at) > RESUME_GRACE_SECONDS:
            self._interrupted = None
            return None
        return capture_id

    def start(self, owner=None, continues: str | None = None) -> str:
        capture_id = new_id()
        self._owner = owner
        self._continues = continues
        if continues is not None:
            logger.info(
                "[Tower][Capture] recording %s continues %s: a reconnect, not "
                "a new walk",
                capture_id,
                continues,
            )
        self._interrupted = None
        self._status = CaptureStatus(
            capture_id=capture_id, started_at=self._clock()
        )
        write_json_atomic(
            self.capture_dir(capture_id) / CAPTURE_FILENAME,
            self._manifest(self._status),
        )
        return capture_id

    def write_frame(
        self,
        raw_bytes: bytes,
        *,
        source_seq: int,
        wire_seq: int | None = None,
        tx_seq: int | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> bool:
        """Persist one frame. Returns False once a bound is reached.

        Never raises for a bound: hitting a limit is an expected, recorded
        outcome, not an error, and turning it into an exception on the
        frame path would put session teardown at the mercy of a disk quota.
        """
        if not self.is_recording:
            return False

        status = self._status
        now = self._clock()
        if now - status.started_at >= self._limits.max_seconds:
            (
                logger.warning(
                    "[Tower][Capture] capture %s reached a configured bound "
                    "and stopped itself; this is NOT a client disconnect, and "
                    "a follower will see the capture close exactly as if it "
                    "were one",
                    self._status.capture_id,
                ),
                self.stop(END_REASON_BOUNDED_LIMIT),
            )[1]
            return False
        if status.bytes_written + len(raw_bytes) > self._limits.max_bytes:
            self.stop(END_REASON_BOUNDED_LIMIT)
            return False

        directory = self.capture_dir(status.capture_id)
        frames_dir = directory / FRAMES_DIRNAME
        frames_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{source_seq:08d}.jpg"
        path = frames_dir / filename

        # Image first with fsync, journal line second. A journal line
        # pointing at a missing image is corruption; an orphan image is
        # harmless and gets swept by purge.
        temp_path = path.with_name(path.name + ".tmp")
        try:
            with temp_path.open("wb") as handle:
                handle.write(raw_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            temp_path.replace(path)
        finally:
            temp_path.unlink(missing_ok=True)

        append_jsonl(
            directory / FRAMES_FILENAME,
            {
                "schema_version": CAPTURE_SCHEMA_VERSION,
                "source_seq": source_seq,
                "wire_seq": wire_seq,
                "tx_seq": tx_seq,
                "received_at": now,
                "time_basis": TIME_BASIS,
                "relpath": f"{FRAMES_DIRNAME}/{filename}",
                "byte_count": len(raw_bytes),
                "width": width,
                "height": height,
            },
        )
        status.frames_written += 1
        status.bytes_written += len(raw_bytes)
        return True

    def stop(self, reason: str = END_REASON_STOP, owner=None) -> CaptureStatus | None:
        """Close the recording -- but only if the caller owns it.

        `owner=None` is an unconditional stop, which is what an operator
        or a test wants. A CONNECTION must always pass its own token,
        because of a race that was measured rather than imagined:

        uvicorn does not learn a WebSocket is dead for 20-40 seconds
        (ws_ping_interval and ws_ping_timeout are both 20 s), while iOS
        reconnects in 0.5 s and re-sends `stream_start`. So the ordering
        on a WiFi hiccup is: the NEW connection arms a recording, and then
        the OLD connection's `finally` block runs and stops it.

        Measured before this guard existed: the phone streamed on, the
        Tower answered every frame_result, `/health` said
        `recording: false`, and ZERO frames reached disk for the rest of
        the walk. Silently. That is worse than losing the capture -- it
        looks like success.
        """
        if self._status is None or not self._status.is_open:
            return self._status
        if owner is not None and self._owner is not None and owner != self._owner:
            logger.info(
                "[Tower][Capture] ignoring a stop from a superseded connection; "
                "capture %s belongs to another",
                self._status.capture_id,
            )
            return self._status
        self._status.ended_at = self._clock()
        self._status.end_reason = reason
        self._owner = None
        # Only a DISCONNECT leaves a walk that might continue. A polite
        # stream_stop, or a configured bound, ends it deliberately.
        self._interrupted = (
            (self._status.capture_id, self._status.ended_at)
            if reason == END_REASON_DISCONNECT
            else None
        )
        write_json_atomic(
            self.capture_dir(self._status.capture_id) / CAPTURE_FILENAME,
            self._manifest(self._status),
        )
        return self._status

    def read_frames(self, capture_id: str) -> list[dict]:
        raw_records, _ = read_raw_jsonl(
            self.capture_dir(capture_id) / FRAMES_FILENAME
        )
        return raw_records

    def purge(self, capture_id: str) -> tuple[int, int]:
        """Really delete a capture. Returns (removed, retained).

        Retained is reported rather than swallowed: a purge that could not
        remove everything must not claim success, because 06-PRIVACY-DATA
        requires real deletion and the caller has no other way to learn
        the imagery is still on disk.
        """
        directory = self.capture_dir(capture_id)
        if not directory.exists():
            return 0, 0
        removed = retained = 0
        for path in sorted(
            directory.rglob("*"), key=lambda p: len(p.parts), reverse=True
        ):
            try:
                path.rmdir() if path.is_dir() else path.unlink()
                removed += 1
            except OSError as exc:
                logger.warning("capture: could not remove %s: %s", path, exc)
                retained += 1
        try:
            directory.rmdir()
            removed += 1
        except OSError:
            retained += 1
        return removed, retained

    def manifest(self, capture_id: str) -> dict | None:
        """The recorded manifest, or None if this capture was never started."""
        path = self.capture_dir(capture_id) / CAPTURE_FILENAME
        if not path.exists():
            return None
        return read_json_closed(path)

    def _manifest(self, status: CaptureStatus) -> dict:
        return {
            "schema_version": CAPTURE_SCHEMA_VERSION,
            "capture_id": status.capture_id,
            "started_at": status.started_at,
            "ended_at": status.ended_at,
            "end_reason": status.end_reason,
            "time_basis": TIME_BASIS,
            "frames_written": status.frames_written,
            "bytes_written": status.bytes_written,
            "max_seconds": self._limits.max_seconds,
            "max_bytes": self._limits.max_bytes,
            # The capture this one continues, or null. Set when a
            # `stream_start` arrives soon after a disconnect, which is the
            # normal shape of a WiFi hiccup mid-walk.
            "continues_capture": getattr(self, "_continues", None),
            "retains_raw_imagery": True,
            "redaction": "none",
            "privacy_tags": ["raw-imagery", "first-person", "dataset-recording"],
        }


class _JournalTail:
    """Reads only what has been appended since the last read.

    The obvious implementation re-reads and re-parses the whole journal on
    every poll, and that is what shipped first. It is O(n) per poll and
    therefore O(n squared) over a capture: measured at 23.8 ms for a single
    poll against a full 10,800-line journal, which is roughly 43 seconds of
    CPU across one 15-minute walk at 4 Hz -- spent in the same process that
    is trying to run `observe()` on every frame.

    Seeking to a remembered byte offset makes a poll cost the same whether
    the journal holds ten lines or ten thousand.

    A partial trailing line is expected, not exceptional: the recorder
    appends without fsync, so a reader can arrive mid-write. The remainder
    is carried forward and completed on the next poll rather than being
    discarded as corruption -- which is what `read_raw_jsonl` does for a
    whole-file read, and what this has to reproduce incrementally.
    """

    __slots__ = ("_path", "_offset", "_remainder")

    def __init__(self, path, *, start_at_end: bool = False) -> None:
        self._path = path
        # Where reading begins. Zero -- the whole journal -- unless a
        # caller has said it arrived late and must not read the part of
        # the recording that happened before it was asked for. A journal
        # that does not exist yet is empty, so both answers are 0.
        self._offset = 0
        if start_at_end:
            try:
                self._offset = path.stat().st_size
            except OSError:
                self._offset = 0
        self._remainder = b""

    def read_new(self) -> list:
        try:
            size = self._path.stat().st_size
        except OSError:
            return []
        if size < self._offset:
            # The file shrank, which for an append-only journal means it
            # was replaced. Start again rather than reading from a stale
            # offset into unrelated bytes.
            self._offset = 0
            self._remainder = b""
        if size == self._offset:
            return []

        try:
            with self._path.open("rb") as handle:
                handle.seek(self._offset)
                chunk = handle.read()
        except OSError:
            return []
        self._offset += len(chunk)

        buffer = self._remainder + chunk
        lines = buffer.split(b"\n")
        # The last element is whatever follows the final newline: empty if
        # the journal ends cleanly, a partial record if a write is in
        # flight.
        self._remainder = lines.pop()

        records = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except (ValueError, UnicodeDecodeError):
                # A line that is complete and still unparseable is real
                # corruption. Skipped, exactly as a whole-file read would.
                logger.warning("[Tower][Capture] skipping an unreadable journal line")
        return records


@dataclass(frozen=True)
class FollowedFrame:
    """One recorded frame, handed back with the metadata the wire carried."""

    source_seq: int
    received_at: float
    raw_bytes: bytes
    relpath: str
    wire_seq: int | None = None
    tx_seq: int | None = None
    width: int | None = None
    height: int | None = None


class CaptureFollower:
    """Yields a capture's frames, including ones not written yet.

    A capture directory is an append-only journal beside a directory of
    images, and the recorder fsyncs each image BEFORE appending its line.
    That ordering is the whole reason this is safe: a line that exists
    always points at a complete file, so a reader never sees a half-written
    JPEG and needs no lock, no handshake and no shared memory with the
    writer.

    This is what makes live processing possible today without touching the
    module lifecycle. The recorder runs on the Tower event loop; whoever
    consumes the frames runs in a separate process and can take as long as
    it likes, because falling behind costs the frame path nothing.

    Bounded by construction (Rule 15): a capture whose manifest never
    closes -- a crashed recorder -- ends the follow after `max_idle_polls`
    quiet polls rather than waiting forever.

    Follows a capture ACROSS a reconnect. When a capture ends by
    disconnect, the follower waits briefly for a successor -- a capture
    whose manifest names this one in `continues_capture` -- and continues
    into it. Without that, a WiFi hiccup ends the mapping session at the
    hiccup: the follower returns, the driver stops the world session, and
    the rest of the walk sits in a second directory that nothing reads.
    """

    def __init__(
        self,
        directory,
        *,
        poll_seconds: float = DEFAULT_FOLLOW_POLL_SECONDS,
        sleep=time.sleep,
        follow_reconnects: bool = True,
        resume_grace_seconds: float = RESUME_GRACE_SECONDS,
        start_at_end: bool = False,
    ):
        self._directory = pathlib.Path(directory)
        self._poll_seconds = poll_seconds
        self._sleep = sleep
        self._follow_reconnects = follow_reconnects
        self._resume_grace_seconds = resume_grace_seconds
        # Set when a stop arrives WHILE waiting out a reconnect. See
        # `stopped_awaiting_successor`.
        self._stopped_awaiting_successor = False
        # Skip whatever the journal already holds, and yield only frames
        # recorded from now on.
        #
        # Off by default, because every existing caller follows a capture
        # from the moment it opens and must see all of it. It is turned
        # on by a consumer ATTACHED LATE -- a cartridge a wearer started
        # three minutes into a walk. Reading the earlier frames would be
        # cheap and wrong: nobody asked for the first three minutes to be
        # processed, and a follower is not the right place to decide that
        # they should be.
        #
        # It applies to THIS directory only. A successor capture after a
        # reconnect is read whole, because by then the consumer has been
        # attached the entire time that capture existed.
        self._start_at_end = start_at_end

    @property
    def directory(self):
        return self._directory

    def is_closed(self) -> bool:
        """True once the recorder has written an end reason."""
        return self.end_reason() is not None

    def end_reason(self) -> str | None:
        """WHY the capture ended, or None while it is still open.

        Carried rather than collapsed into `is_closed`, because the three
        reasons are not the same event to a consumer. `stop` is the wearer.
        `disconnect` is the link. `bounded_limit` is the recorder stopping
        ITSELF at a configured bound while the wearer is very likely still
        walking -- and a builder that treats that as an ordinary end
        finalises a world at the bound and says nothing.
        """
        path = self._directory / CAPTURE_FILENAME
        if not path.exists():
            return None
        try:
            manifest = read_json_closed(path)
        except (OSError, ValueError):
            # A manifest caught mid-replace is not an ended capture. Say
            # "still open" and re-read next poll rather than truncating
            # the session on a transient read.
            return None
        if manifest.get("ended_at") is None:
            return None
        return manifest.get("end_reason") or END_REASON_STOP

    def follow(self, *, max_idle_polls: int | None = None, should_stop=None):
        """Frames, until the capture ends, the idle bound expires, or a
        caller says stop.

        `should_stop` is asked once per poll, and the poll loop is the
        only place it CAN be asked. A driver that wrapped this generator
        and checked between yields would only get control back when a
        frame arrived -- so a producer asked to stop during a quiet
        stretch, which is most of a walk and all of a walk that has just
        been put down, would not notice until the next thing moved. It
        was written that way first and it did not stop.

        Asked before the journal read rather than after, so a stop that
        arrived while this generator was sleeping does not first pull in
        another frame the person had already asked not to be remembered.
        """
        journal = self._directory / FRAMES_FILENAME
        tail = _JournalTail(journal, start_at_end=self._start_at_end)
        idle_polls = 0
        # Per FOLLOW, not per follower: a generator re-entered would
        # otherwise carry the previous run's answer forward.
        self._stopped_awaiting_successor = False

        while True:
            if should_stop is not None and should_stop():
                return
            fresh = tail.read_new()

            for record in fresh:
                frame = self._load(record)
                if frame is not None:
                    yield frame

            # Journal first, manifest second, then ONE more journal read.
            # The recorder appends a line and only later rewrites the
            # manifest, so a follower that stopped the instant it saw an
            # end reason would drop whatever landed in between.
            if self.is_closed():
                for record in tail.read_new():
                    frame = self._load(record)
                    if frame is not None:
                        yield frame

                successor = self._await_successor(should_stop=should_stop)
                if successor is None:
                    return
                if should_stop is not None and should_stop():
                    # THE SAME WINDOW, ONE INSTRUCTION WIDE.
                    #
                    # `_await_successor` samples `should_stop` and then
                    # returns; the rebind below happens next, and the loop's
                    # own check is after that. A stop set between those two
                    # samples used to bind to a capture this follower then
                    # read ZERO frames of -- moving `end_reason()` onto a
                    # capture that is still open, and reporting
                    # `stopped_awaiting_successor()` as False. That is the
                    # defect the wait was fixed for, surviving in a window
                    # that shrank from ninety seconds to a few statements.
                    # A reviewer drove it. Asking once more, before the
                    # rebind, closes it: the flag is the same fact either
                    # way, and the follower stays where it read from.
                    self._stopped_awaiting_successor = True
                    return
                # Same walk, new directory. Rebind and keep going, so the
                # driver never learns a reconnect happened and the mapping
                # session stays continuous.
                logger.info(
                    "[Tower][Capture] following %s into %s after a reconnect",
                    self._directory.name,
                    successor.name,
                )
                self._directory = successor
                tail = _JournalTail(self._directory / FRAMES_FILENAME)
                idle_polls = 0
                continue

            if fresh:
                idle_polls = 0
            else:
                idle_polls += 1
                if max_idle_polls is not None and idle_polls >= max_idle_polls:
                    return

            self._sleep(self._poll_seconds)

    def _ended_by_disconnect(self) -> bool:
        try:
            manifest = read_json_closed(self._directory / CAPTURE_FILENAME)
        except (OSError, ValueError):
            return False
        return manifest.get("end_reason") == END_REASON_DISCONNECT

    def _find_successor(self):
        """A capture whose manifest names this one as its predecessor.

        NEWEST FIRST, AND ONLY CAPTURES YOUNGER THAN THIS ONE. A successor
        is by definition created after its predecessor ended, so anything
        older cannot be one, and the one we want is almost always the
        newest directory there is.

        This used to open and parse every `capture.json` under the root, in
        whatever order `iterdir` gave, with no early exit. A reviewer
        measured it at **11 ms per scan against 104 captures** -- and this
        runs once per poll for up to 360 polls, so the "ninety second"
        grace window actually ran 94 s, an overrun that grows with a
        directory that only ever grows (about 109 s at 500 captures). The
        wait is supposed to be bounded by the same window the recorder
        uses, and it was not.

        THE PRUNE CARRIES THE GRACE WINDOW AS SLACK, and the first version
        did not. It cut at this capture's own mtime, and a reviewer showed
        that mtime moves FORWARD at stop -- `write_json_atomic` on
        `capture.json` bumps the parent directory (measured, +0.30 s). So
        a successor created before the predecessor's `finally` ran was
        pruned by 0.4 s and never found: 0 of 360 polls, not "a missed
        successor on that poll" as the comment then claimed. Both mtimes
        are frozen for the whole window, so a prune that misses once
        misses every time.
        `RESUME_GRACE_SECONDS` of slack is the same bound the recorder
        uses to decide whether to link a successor at all, so nothing the
        recorder would link can fall outside it -- and it absorbs a
        backwards clock step of up to ninety seconds as well.
        """
        captures_root = self._directory.parent
        mine = self._directory.name
        try:
            floor = self._directory.stat().st_mtime - self._resume_grace_seconds
        except OSError:
            floor = None
        candidates = []
        try:
            # `os.scandir`, NOT `iterdir()` + `is_dir()` + `stat()`.
            # Windows returns the type and the timestamps in the directory
            # entry itself, so scandir answers both from data it already
            # has. The three-call version cost two stats per entry and was
            # where the 11 ms went -- the JSON reads were never the
            # expensive part.
            with os.scandir(captures_root) as entries:
                for entry in entries:
                    if entry.name == mine or not entry.is_dir():
                        continue
                    try:
                        mtime = entry.stat().st_mtime
                    except OSError:
                        continue
                    if floor is not None and mtime < floor:
                        continue
                    candidates.append((mtime, pathlib.Path(entry.path)))
        except OSError:
            return None
        for _mtime, entry in sorted(candidates, key=lambda pair: pair[0], reverse=True):
            try:
                manifest = read_json_closed(entry / CAPTURE_FILENAME)
            except (OSError, ValueError):
                continue
            if isinstance(manifest, dict) and manifest.get("continues_capture") == mine:
                return entry
        return None

    def stopped_awaiting_successor(self) -> bool:
        """Whether this follower was stopped mid-reconnect.

        A caller deciding whether a walk finished has to tell two things
        apart that look identical from the capture alone: a walk that
        ended, and a walk whose link died and was then abandoned before it
        could come back. The predecessor's own end reason is `disconnect`
        in both, and `disconnect` counts as finished -- deliberately, see
        `scripts/world_build_session.py`. Only the follower knows that a
        reconnect was still in flight when it was told to go.

        TRUE MEANS "A SUCCESSOR HAD APPEARED AND I LEFT IT UNREAD". It does
        not mean "the walk was cut short", which is a larger set: when the
        stop lands before the successor's directory exists, a cut-short
        walk is indistinguishable on disk from a finished one and this
        reports False. See `_await_successor` for why that is the right way
        to resolve an ambiguity nothing on disk can settle.
        """
        return self._stopped_awaiting_successor

    def _await_successor(self, *, should_stop=None):
        """Wait out a reconnect, but only for a capture that was CUT OFF.

        A capture that ended politely, or at a configured bound, is
        finished -- waiting on those would add a fixed delay to the end of
        every ordinary session for nothing.

        The wait is bounded by the same grace window the recorder uses to
        decide whether to link a successor at all, so the two cannot
        disagree about how long a reconnect may take.

        AND IT ASKS `should_stop`, WHICH IT DID NOT. Ninety seconds is a
        long time to be deaf. `routes/ws.py` stops every cartridge session
        when the last connection goes, and it is exactly a slow reconnect
        that gets it there: a reviewer measured this Tower's own 52
        reconnect chains and found three at 31-35 s, against a socket the
        server notices as dead in 20-40 s. So the stop lands while this
        loop is running.

        What happened then was worse than the delay. The loop ran to the
        end of the grace window, found the successor, rebound onto it, and
        `continue`d -- straight into the `should_stop` check at the top of
        `follow`, which returned having read ZERO frames of the capture it
        had just bound to. The walk was truncated at the reconnect, and
        the rebind quietly moved `end_reason()` onto a capture that was
        still open, so nothing downstream could say what had happened.

        Now the wait ends when the stop does, the follower stays bound to
        the capture it actually read, and `stopped_awaiting_successor()`
        records that a reconnect was in flight -- which is the one fact
        that distinguishes an abandoned walk from a finished one.
        """
        if not self._follow_reconnects or not self._ended_by_disconnect():
            return None
        polls = max(1, int(self._resume_grace_seconds / max(self._poll_seconds, 1e-6)))
        for _ in range(polls):
            successor = self._find_successor()
            stopping = should_stop is not None and should_stop()
            if successor is not None:
                if not stopping:
                    return successor
                # A RECONNECT WAS IN FLIGHT AND WE ARE LEAVING ANYWAY.
                #
                # This is the only shape that can be DETECTED as a walk cut
                # short -- not the only one that is. The first version set
                # the flag on any stop that landed inside the window, which
                # is also the ordinary shape of a phone that disconnects for
                # good: the socket dies, `routes/ws.py` stops the session
                # 20-40 s later, comfortably inside the 90 s grace. That
                # quietly reversed the policy two files away that says a
                # `disconnect` capture counts as finished, and a reviewer
                # measured a permanent disconnect reporting `interrupted` --
                # this campaign's headline symptom, back through the door it
                # was pushed out of.
                #
                # THE RESIDUE, STATED PLAINLY. When the stop lands BEFORE
                # the successor's directory appears, a walk that was cut
                # short is indistinguishable on disk from a walk that
                # ended: the same `disconnect` capture, the same absent
                # successor. A second reviewer measured the overlap and it
                # is not small -- reconnect chains at 31-35 s against a
                # socket noticed dead at 20-40 s. Neither answer is
                # derivable, so this resolves it the way the rest of the
                # system resolves an ambiguous end: toward FINISHED, which
                # shows a real world as Saved rather than showing a real
                # world as broken. Closing it needs the phone to say which
                # it was, not more cleverness here.
                self._stopped_awaiting_successor = True
                return None
            if stopping:
                # Nobody came back. The walk was over either way, so this
                # is an ordinary end and the caller is told nothing
                # special -- it just stops waiting for it.
                return None
            self._sleep(self._poll_seconds)
        return None

    def _load(self, record: dict) -> FollowedFrame | None:
        relpath = record.get("relpath")
        if not relpath:
            return None
        path = self._directory / relpath
        try:
            raw_bytes = path.read_bytes()
        except OSError:
            # Should be impossible given the write ordering. A reader that
            # trusted the invariant absolutely would turn one deleted file
            # into a crash mid-session.
            logger.warning("capture: journal references missing image %s", path)
            return None
        return FollowedFrame(
            source_seq=record["source_seq"],
            received_at=record["received_at"],
            raw_bytes=raw_bytes,
            relpath=relpath,
            wire_seq=record.get("wire_seq"),
            tx_seq=record.get("tx_seq"),
            width=record.get("width"),
            height=record.get("height"),
        )
