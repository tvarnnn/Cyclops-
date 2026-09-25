"""The per-frame quality log: what the builder measured and decided about EVERY frame.

OFF BY DEFAULT. `TOWER_WORLD_FRAME_QUALITY_LOG` (`config.world_frame_quality_log_setting`),
read by the builder at each session start. Off, nothing here runs: no file is created, and
every output of a session is byte-identical to the builder before this module existed
(tests/golden/world_builder_frame_quality_ceb203a.json, recorded from ceb203a).

WHY IT EXISTS (P5-SHARP, manager 091 B3). Only keyframes carry a quality record
(`keyframes.jsonl`), so a walk that refused a quarter of its frames as `blurred` -- walk 3:
851 blurred, 1,458 insufficient_motion -- leaves no per-frame series to calibrate a turn
governor against. This is that series.

WHY A FILE OF ITS OWN, `sessions/<sid>/frames_quality.jsonl`, and not fields on an existing
record:

* `events.jsonl` is the journal a live reader tails (events.py); a line per frame would swamp
  it, and its kind set is closed because consumers switch on it.
* `keyframes.jsonl` holds keyframes only; its parser (`keyframe_from_json_dict`) is strict
  and every build reads it, so a non-keyframe row there would break old readers.
* `session.json` is rewritten whole and atomically; a per-frame series in it would make every
  rewrite O(frames).

A new file changes no existing byte, and no reader knows it is there: nothing in the Tower
reads it, nothing reaches the phone, so there is no contract change. A world without it --
every world written before this, and every world written with the switch off -- loads,
renders and finishes exactly as before. `WorldStore.purge_world` removes it with the world
(it walks the whole directory); `WorldStore.world_bytes` counts it under `journals`.

ONE LINE PER OBSERVED FRAME: one per `engine.observe()` call that returned, so the line count
equals the session's `frames_observed` for a walk that ended normally. A frame the frontend
never scored (`malformed_frame`, `frame_size_changed`) still gets its line, with every
measurement null, so the series has no holes to misalign against an IMU clock.

EVERY VALUE IS ONE THE BUILDER ALREADY COMPUTED on the live path; nothing here decodes,
measures or re-derives anything. A line holds (null where the value does not exist):

  v                       FRAME_QUALITY_VERSION
  source_seq              the builder's sequence number, the keyframe record's -- NOT always
                          the capture's (see JOINS below)
  received_at             as observe() received it: the capture record's own value
  sharpness               `FrameQuality.sharpness`, the variance of the Laplacian: THE number
                          `KeyframeSelector._is_sharp_enough` compares, full float precision
  tracker                 `reference` (motion was measured against the current reference
                          keyframe) or `no_reference` (session start, or the first frame after
                          a loss: `evaluate` got motion None); null if the frame was not scored
  feature_count           `MotionSummary.seeded_count`, the reference's seeds
  tracked_count           `MotionSummary.tracked_count` (the motion-evidence gate's count)
  survival_ratio          `MotionSummary.survival_ratio` (the loss and degraded gates)
  overlap_ratio           `MotionSummary.overlap_ratio` (the overlap-floor gate)
  median_parallax_px      `MotionSummary.median_displacement_px`, the median LK displacement
                          against the reference -- the keyframe record's name for it; the
                          parallax gates divide it by the image diagonal
  homography_residual_px  `MotionSummary.homography_residual_px`: not a gate, a rotation
                          detector (frontend.MotionSummary), logged because it is already there
  frames_since_keyframe   the selector's counter as `evaluate` saw it (after `note_frame`): how
                          stale the reference was
  segment_index           the segment the frame was measured in (before a loss moves it on)
  outcome, reason         the selector's decision exactly (`KeyframeDecision`), or the
                          engine's own reject for an unscored frame
  keyframe_id             the persisted keyframe's id, or null: whether it became a keyframe

JOINS USE `received_at`, NEVER `source_seq`. When the sender restarts inside a capture lineage
its sequence starts again, and the builder RENUMBERS from there onto a monotonic sequence
(engine.observe; the journal says `source_seq_restarted`) so no keyframe is overwritten. From
that frame on, this file's `source_seq` is the builder's number -- equal to `keyframes.jsonl`'s,
different from the capture's `frames.jsonl`. `received_at` is passed through untouched from the
capture record (capture.py's follower), so it joins the capture, and an IMU clock, exactly. (A
source with no recorded timestamp -- a directory of loose JPEGs -- gets the engine's receipt
time instead, as the keyframe record does.) Within this file and against `keyframes.jsonl`,
`keyframe_id` is the join.

THE BLUR VERDICT IS REPRODUCIBLE FROM THIS FILE ALONE. The ratio test compares `sharpness`
against the median (`sorted(w)[len(w) // 2]`) of the last `KeyframePolicy.sharpness_window`
(30) scored frames INCLUDING this one, once at least 5 are held; the absolute floor is
`min_sharpness` (25.0). Every scored frame has a line and nothing else enters that window, so
the lines with a non-null `sharpness`, in order, rebuild it exactly (a test pins that).

WRITING NEVER COSTS THE WALK ANYTHING. One handle per session, opened at session start and
buffered (`io.DEFAULT_BUFFER_SIZE`, 8 KB: about 20 lines per system call), no fsync; closed --
so flushed -- first thing in `stop_session`, which also runs when a walk ends in error. Any
failure -- building a line (`FrameQualityLog.record` builds it inside the guard), opening,
writing or flushing -- is logged ONCE and turns the log off for the rest of that session; the
builder never sees an exception from here.

THE PRICE OF BUFFERING, and what a reader must tolerate. A builder killed outright loses the
unflushed tail: at most one buffer, 8 KB, about 20 lines -- under 2 s at 12 fps, but about 6 s
at today's ~3.3 fps delivery. And the buffer reaches the disk in 8 KB pieces cut at byte
boundaries, not at line ends, so such a kill can also leave the LAST LINE TORN. Readers must
skip a torn line rather than fail on it: `read_frames_quality` does (as `read_raw_jsonl` does for
every journal here), and a log opened on a file whose last line is torn starts on a fresh line
first, so the tear never fuses with the next record. Flushing per line would close that window
at the cost of a system call per frame; for a calibration series the tolerant reader is the
cheaper answer.

MEASURED (RUN/experiments/P5-SHARP, this host): 5.3 us per line for the row, the JSON and the
buffered write (perf.py), against an 83 ms frame interval at 12 fps. Replaying a real frozen
1,251-frame capture (0892c308, replay_real.py) twice each way, observe()'s median was
3.88-4.01 ms off and 3.94-4.09 ms on, and the whole replay 8.82-9.05 s off and 8.81-9.23 s
on: no difference outside run-to-run noise.

SIZE: 415 bytes a line on that real capture (519,487 bytes, 1,251 lines); 405 for a measured
frame, 444 for a keyframe (its 41-character id) and 346 for an unscored one with a Tower epoch
`received_at` (perf.py). A 5-minute walk at 12 fps (3,600 frames) is therefore about 1.5 MB;
at today's ~3.3 fps delivery (990 frames), 0.4 MB. The names are the keyframe record's,
spelled out, on purpose: 237 of those bytes are keys, and a reader joins the two files without
a legend. Retention is the session's: kept with the world, purged with it. It holds numbers
only, never imagery.
"""

import io
import json
import logging
import os
from pathlib import Path

from tower.storage import read_raw_jsonl

logger = logging.getLogger(__name__)

FRAMES_QUALITY_FILENAME = "frames_quality.jsonl"
FRAME_QUALITY_VERSION = 1

TRACKER_REFERENCE = "reference"
TRACKER_NO_REFERENCE = "no_reference"


def frames_quality_path(store, world_id: str, session_id: str) -> Path:
    """Where a session's log lives. Beside `events.jsonl`, never inside it."""
    return store.session_dir(world_id, session_id) / FRAMES_QUALITY_FILENAME


def read_frames_quality(path) -> list[dict]:
    """A session's lines, in order, for analysis. Nothing in the Tower calls this.

    An absent file is `[]`: every world written before this module, and every world written
    with the switch off. A torn line -- the last one, after a builder was killed mid-buffer
    (see the module docstring) -- is skipped with a warning, never raised: `read_raw_jsonl`'s
    rule for every journal here. Only JSON objects are returned.
    """
    records, _ = read_raw_jsonl(Path(path))
    return [record for record in records if isinstance(record, dict)]


def frame_row(
    *,
    source_seq: int,
    received_at: float,
    outcome: str,
    reason: str,
    quality=None,
    motion=None,
    frames_since_keyframe: int | None = None,
    segment_index: int | None = None,
    keyframe_id: str | None = None,
) -> dict:
    """One line, from the objects the engine already holds. Copies; computes nothing."""
    measured = motion is not None
    return {
        "v": FRAME_QUALITY_VERSION,
        "source_seq": source_seq,
        "received_at": received_at,
        "sharpness": quality.sharpness if quality is not None else None,
        "tracker": (
            None if quality is None
            else TRACKER_REFERENCE if measured else TRACKER_NO_REFERENCE
        ),
        "feature_count": motion.seeded_count if measured else None,
        "tracked_count": motion.tracked_count if measured else None,
        "survival_ratio": motion.survival_ratio if measured else None,
        "overlap_ratio": motion.overlap_ratio if measured else None,
        "median_parallax_px": motion.median_displacement_px if measured else None,
        "homography_residual_px": motion.homography_residual_px if measured else None,
        "frames_since_keyframe": frames_since_keyframe,
        "segment_index": segment_index,
        "outcome": outcome,
        "reason": reason,
        "keyframe_id": keyframe_id,
    }


class FrameQualityLog:
    """One session's log: one buffered handle, never an exception to the caller."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._handle = None
        self._failed = False
        self._closed = False
        self._lines = 0
        try:
            # A torn last line (a killed builder) is ended before anything is appended,
            # so it never fuses with the next record -- `append_jsonl`'s rule.
            torn = False
            if self._path.exists() and self._path.stat().st_size > 0:
                with self._path.open("rb") as tail:
                    tail.seek(-1, os.SEEK_END)
                    torn = tail.read(1) != b"\n"
            # Append, never truncate: a session id is fresh, so the file is new, and if it
            # somehow is not, nothing already in it is destroyed. "\n" on every platform,
            # so the bytes (and the size quoted above) do not depend on the host.
            self._handle = open(  # noqa: SIM115 -- held for the session, closed in close()
                self._path, "a", encoding="utf-8", newline="\n",
                buffering=io.DEFAULT_BUFFER_SIZE,
            )
            if torn:
                self._handle.write("\n")
        except Exception as error:  # noqa: BLE001 -- a log is never worth a walk
            self.fail("open", error)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def lines(self) -> int:
        """Lines handed to the buffer. A later flush failure can still lose the tail."""
        return self._lines

    @property
    def failed(self) -> bool:
        return self._failed

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def active(self) -> bool:
        """Still writing: opened, not failed, not closed. Off, callers skip the work."""
        return self._handle is not None

    def record(self, measured: dict | None = None, **fields) -> None:
        """Build one line from `frame_row`'s fields AND write it, both inside the guard.

        This is the engine's entry point (review V16 MED-1): an exception from building the
        row -- a renamed attribute on a `FrameQuality` or `MotionSummary`, a field the row
        does not take, a field given twice -- is the log's failure like any I/O error: logged
        once, the log off for the session, and nothing reaches `observe()`. `measured` (what
        `evaluate` saw) is merged in here rather than at the call, for the same reason.
        """
        if self._handle is None:
            return
        try:
            row = frame_row(**fields, **(measured or {}))
        except Exception as error:  # noqa: BLE001 -- see the module docstring
            self.fail("build a line for", error)
            return
        self.write(row)

    def write(self, row: dict) -> None:
        handle = self._handle
        if handle is None:
            return
        try:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        except Exception as error:  # noqa: BLE001 -- see the module docstring
            self.fail("write", error)
            return
        self._lines += 1

    def close(self) -> None:
        """Flush and close. Idempotent; never raises."""
        handle, self._handle = self._handle, None
        self._closed = True
        if handle is None:
            return
        try:
            handle.close()
        except Exception as error:  # noqa: BLE001 -- the flush is a write too
            self.fail("flush", error)

    def fail(self, doing: str, error: BaseException) -> None:
        """Log once, drop the handle, stay off for the rest of the session. Never raises.

        Public so the engine can hand the log a failure of its own side of the line (reading
        what `evaluate` saw) under the same rule."""
        handle, self._handle = self._handle, None
        already = self._failed
        self._failed = True
        if not already:
            logger.warning(
                "[Tower][WorldBuilder] the frame quality log could not %s %s (%s: %s); it is "
                "off for the rest of this session and the walk continues",
                doing, self._path, type(error).__name__, error,
            )
        if handle is not None:
            try:
                handle.close()
            except Exception:  # noqa: BLE001 -- the same failure again; already logged
                pass
