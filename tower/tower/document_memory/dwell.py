"""Has this page been held in view long enough to be worth reading?

The expensive path costs ~0.3 s of GPU OCR per page (1.9 s on CPU), so
the whole design depends on this stage being both cheap and stingy. A
single frame containing text is not a reading event; it is a glance, a
reflection, or a poster on a wall the wearer walked past.

**Both a frame count and a wall-clock duration are required.** Frame rate
is not fixed here -- at ~3.3 fps, "six frames" is nearly two seconds; at
the 12 fps now delivered it is half of one. A policy expressed only in
frames would silently mean something different every time the transport
changed, which is why the loss tolerance below is in SECONDS.

**A dwell is made of segments.** A wearer who turns a page without
moving their head presents the same region with different content. The
region tracker cannot see that; the content check can (a phase
correlation of the region against the segment's first frame collapses
when the words change). Each segment keeps its own best frames, so a
three-page read within one dwell yields three pages, not the two
sharpest frames of the whole thing.

None of this measures attention. It measures a text region being present
and steady. `07-PLATFORM-CONSTRAINTS.md` Limitation 8 is explicit that
the camera cannot establish that the wearer looked at, noticed or read
anything, and no threshold in this file changes that.
"""

from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from tower.document_memory.records import (
    END_REASON_LOST,
    END_REASON_MAX_DURATION,
    END_REASON_STOPPED,
)


@dataclass(frozen=True)
class DwellPolicy:
    """Every threshold, in one place, each with a reason.

    Deliberately a value object rather than module constants: the
    benchmark sweeps these, and a threshold that cannot be swept cannot be
    chosen from data.
    """

    # A glance is not a reading event. Both must be satisfied.
    min_frames: int = 4
    min_seconds: float = 1.0
    # A page turn, a blink of occlusion, a hand across the page or a run
    # of blurred frames must not end a dwell; walking away must. In
    # SECONDS, because at 12 fps three frames is a quarter of a second
    # and a page turn takes a full one. The frame bound is a backstop.
    max_missing_seconds: float = 1.5
    max_missing_frames: int = 60
    # Rule 15: nothing unbounded. A page left on a desk in view all
    # afternoon becomes one bounded observation, not an ever-growing one.
    max_seconds: float = 180.0
    # A gap this large between two detections is not one continuous
    # reading. Timestamps come from the capture journal and are WALL
    # CLOCK, so this also contains the damage when a clock jumps: an
    # adversarial review produced a document claiming 92 days of reading
    # from a single bad timestamp.
    max_frame_gap_s: float = 5.0
    # How much the region may move between frames and still be "the same
    # page", as a fraction of the frame diagonal. A person holding a page
    # is not a tripod.
    max_centre_shift_fraction: float = 0.18
    # And how much it may change size. Leaning in is the same page.
    min_area_ratio: float = 0.5
    max_area_ratio: float = 2.0
    # How many frames get the expensive path, per SEGMENT. Two, because
    # OCR is not deterministic across views and the better reading wins.
    best_frames: int = 2
    # And how many segments a dwell may hold. Bounds the OCR a flush at
    # Stop can cost: 6 x 2 x ~0.3 s on the GPU.
    max_segments: int = 6
    # A frame this blurry is not worth OCR even if it is the best one seen.
    # Variance of the Laplacian over the text region. Real frames are
    # lower-contrast than the renderer; the selector is best-of-window,
    # this is only the floor under it.
    min_sharpness: float = 40.0
    # Content check: phase-correlation response of the region against the
    # segment's reference crop. A page shifted slightly stays above 0.9;
    # different words on the same region drop to ~0.24 (synthetic).
    content_change_response: float = 0.45
    # And it must fail this many consecutive frames before a new segment
    # opens, so a noisy frame or two cannot split a page in two. Three
    # is a quarter of a second at 12 fps, well inside a real page turn.
    content_change_frames: int = 3
    # Width of the crop the content check compares at.
    content_probe_width: int = 160


@dataclass
class ScoredFrame:
    """A candidate frame, kept only if it might earn the expensive path."""

    source_seq: int | None
    at: float
    candidate: object
    gray: np.ndarray
    segment: int = 0

    @property
    def score(self) -> float:
        """Sharpness first, squareness as the tie-break.

        Both matter and they are not interchangeable: a razor-sharp page
        seen at 60 degrees warps into something OCR reads badly, and a
        perfectly square page that is out of focus reads worse still.
        Multiplying lets one veto the other.
        """
        return self.candidate.sharpness * max(self.candidate.squareness, 0.05)


@dataclass
class Dwell:
    """One sustained observation in progress, or just finished."""

    started_at: float
    last_seen_at: float
    frames_seen: int = 0
    frames_considered: int = 0
    missing_frames: int = 0
    # The best frames of the CURRENT segment.
    best: list = field(default_factory=list)
    # The best frames of every CLOSED segment, in order.
    closed: list = field(default_factory=list)
    segment: int = 0
    end_reason: str = END_REASON_LOST
    reference: object | None = None
    # Whatever the caller said the frames belonged to when this dwell
    # STARTED, kept opaque so this module stays free of any notion of a
    # capture. Set once, in `_start`, and never touched by `_extend`.
    #
    # Set once is the entire point. Document Memory stamps a capture id
    # onto the record a dwell produces, and it used to read that id at
    # RECORD time -- so a `stream_start` arriving mid-dwell (a phone
    # reconnect re-arms the recorder) moved every frame of that reading
    # onto a capture it did not come from, and a `stream_stop` erased it
    # entirely. Both were measured. A pointer that resolves to the wrong
    # frame is worse than no pointer at all.
    lineage: object | None = None
    # Accumulated FORWARD-ONLY, one inter-frame delta at a time.
    elapsed_seconds: float = 0.0
    clock_regressions: int = 0
    # The content check's reference for the current segment, and how
    # many consecutive frames have disagreed with it.
    content_reference: np.ndarray | None = None
    content_disagreements: int = 0

    @property
    def seconds(self) -> float:
        """How long the page was observed, immune to a clock that moves back.

        Accumulated from positive inter-frame deltas rather than computed
        as `last_seen_at - started_at`. The difference is not academic:
        timestamps are wall-clock from the capture journal, so an NTP
        correction on the LAST frame of an otherwise perfect dwell used to
        clamp this to zero, fail `qualifies()`, and discard the whole
        document with no error and no trace.
        """
        return self.elapsed_seconds

    @property
    def selected(self) -> list:
        """Every frame that earned OCR, closed segments first, in order."""
        frames = []
        for segment in self.closed:
            frames.extend(segment)
        frames.extend(self.best)
        return frames

    @property
    def segments(self) -> int:
        return len(self.closed) + (1 if self.best else 0)

    def qualifies(self, policy: DwellPolicy) -> bool:
        return (
            self.frames_seen >= policy.min_frames
            and self.seconds >= policy.min_seconds
            and bool(self.selected)
        )


class DwellTracker:
    """The state machine. Cheap, and the only thing that opens the wallet.

    Returns a completed `Dwell` from `observe()` on the frame where it
    ends -- so a caller never has to poll, and a dwell that never
    qualifies is simply dropped without the caller learning about it.
    """

    def __init__(self, policy: DwellPolicy | None = None) -> None:
        self._policy = policy or DwellPolicy()
        self._current: Dwell | None = None
        self._segments_opened = 0

    @property
    def policy(self) -> DwellPolicy:
        return self._policy

    @property
    def in_dwell(self) -> bool:
        return self._current is not None

    @property
    def current(self) -> Dwell | None:
        return self._current

    @property
    def segments_opened(self) -> int:
        """Page turns detected inside dwells, over the tracker's life."""
        return self._segments_opened

    def observe(
        self,
        candidate,
        *,
        at: float,
        gray: np.ndarray | None = None,
        source_seq: int | None = None,
        frame_diagonal: float | None = None,
        lineage: object | None = None,
    ) -> Dwell | None:
        """Feed one frame. Returns a completed dwell, or None."""
        if candidate is None:
            return self._miss(at)

        if self._current is None:
            self._start(candidate, at, gray, source_seq, lineage)
            return None

        if lineage != self._current.lineage:
            # The frames stopped belonging to the recording this dwell
            # started in. A reconnect re-arms the recorder, mints a new
            # capture id, and `source_seq` restarts with it.
            #
            # Ending the dwell is the only honest answer. Keeping it open
            # froze `capture_id` at the old capture while letting frames
            # from the new one win the OCR slots -- so the published
            # `page_source_seqs` resolved into the OLD journal and named
            # completely different frames. Silent, plausible and wrong,
            # which is worse than no pointer at all. Same reasoning
            # `_is_same_region` applies to a different page.
            finished = self._finish(END_REASON_LOST)
            self._start(candidate, at, gray, source_seq, lineage)
            return finished

        if at - self._current.last_seen_at > self._policy.max_frame_gap_s:
            # Too long since the last detection for this to be one
            # continuous reading -- a stalled stream, or a clock jump.
            # Ending the dwell contains the damage either way.
            finished = self._finish(END_REASON_LOST)
            self._start(candidate, at, gray, source_seq, lineage)
            return finished

        if not self._is_same_region(candidate, frame_diagonal):
            # A different page is a NEW dwell, not a continuation. Without
            # this, turning from one document to another would merge two
            # documents into one record with interleaved text.
            finished = self._finish(END_REASON_LOST)
            self._start(candidate, at, gray, source_seq, lineage)
            return finished

        self._extend(candidate, at, gray, source_seq)

        if self._current.seconds >= self._policy.max_seconds:
            return self._finish(END_REASON_MAX_DURATION)
        return None

    def flush(self, reason: str = END_REASON_STOPPED) -> Dwell | None:
        """End an open dwell because the stream did, not because the page did."""
        if self._current is None:
            return None
        return self._finish(reason)

    # -- internals -----------------------------------------------------

    def _start(self, candidate, at, gray, source_seq, lineage=None) -> None:
        self._current = Dwell(
            started_at=at,
            last_seen_at=at,
            reference=candidate,
            lineage=lineage,
        )
        self._extend(candidate, at, gray, source_seq)

    def _extend(self, candidate, at, gray, source_seq) -> None:
        dwell = self._current
        delta = at - dwell.last_seen_at
        if delta < 0:
            # The clock moved back. Contribute nothing rather than
            # subtracting, and record that it happened so a short
            # observation is explainable.
            dwell.clock_regressions += 1
        else:
            dwell.elapsed_seconds += delta
        dwell.frames_seen += 1
        dwell.frames_considered += 1
        dwell.last_seen_at = at
        dwell.missing_frames = 0
        # The reference tracks the latest accepted region rather than the
        # first: a wearer drifting slowly across a page would otherwise
        # eventually fail the same-region test against where they started.
        dwell.reference = candidate

        if gray is not None:
            self._check_content(dwell, candidate, gray)

        if gray is None or candidate.sharpness < self._policy.min_sharpness:
            return
        scored = ScoredFrame(
            source_seq=source_seq,
            at=at,
            candidate=candidate,
            gray=gray,
            segment=dwell.segment,
        )
        dwell.best.append(scored)
        dwell.best.sort(key=lambda frame: frame.score, reverse=True)
        # Bounded: only the frames that will actually be OCR'd are
        # retained, so a long dwell cannot accumulate imagery.
        del dwell.best[self._policy.best_frames :]

    def _check_content(self, dwell: Dwell, candidate, gray) -> None:
        """Same region, different words? Then this is a new segment."""
        probe = self._content_probe(candidate, gray)
        if probe is None or float(probe.std()) < 1.0:
            # A featureless crop correlates with nothing, including
            # itself. It is inconclusive, not a page turn.
            return
        reference = dwell.content_reference
        if reference is None or reference.shape != probe.shape:
            dwell.content_reference = probe
            dwell.content_disagreements = 0
            return
        _shift, response = cv2.phaseCorrelate(reference, probe)
        if response >= self._policy.content_change_response:
            dwell.content_disagreements = 0
            return
        dwell.content_disagreements += 1
        if dwell.content_disagreements < self._policy.content_change_frames:
            return
        # A page turn. Close the segment, keep its best frames, and let a
        # new one begin with this frame as its reference.
        if dwell.best:
            dwell.closed.append(list(dwell.best))
            dwell.best = []
        dwell.segment += 1
        self._segments_opened += 1
        dwell.content_reference = probe
        dwell.content_disagreements = 0
        if len(dwell.closed) >= self._policy.max_segments:
            # Bounded. The oldest segment's frames are dropped rather than
            # the newest: what the wearer is looking at NOW is the page
            # they are most likely to ask about.
            del dwell.closed[0]

    def _content_probe(self, candidate, gray) -> np.ndarray | None:
        from tower.document_memory.gate import crop_region

        crop = crop_region(gray, candidate)
        if crop.size == 0:
            return None
        # The INNER part of the crop. The region is padded by design and
        # its border is whatever surrounds the page; a page held steady
        # against a moving background (a wearer walking with a sheet in
        # hand) turned that border into four spurious page turns in one
        # 48-frame replay. The words are in the middle.
        height, width_px = crop.shape[:2]
        inset_y, inset_x = int(height * 0.1), int(width_px * 0.1)
        if height - 2 * inset_y >= 8 and width_px - 2 * inset_x >= 8:
            crop = crop[inset_y : height - inset_y, inset_x : width_px - inset_x]
        width = self._policy.content_probe_width
        # A FIXED probe size, square, whatever the crop's aspect: a region
        # that grows as the wearer leans in, or whose text block is a
        # different shape after a page turn, still compares against its
        # own reference. An aspect-preserving probe changed shape on a
        # page turn and was silently adopted as the new reference, so no
        # turn was ever detected -- found by the test written for it.
        small = cv2.resize(crop, (width, width), interpolation=cv2.INTER_AREA)
        return np.float32(small)

    def _miss(self, at: float) -> Dwell | None:
        if self._current is None:
            return None
        self._current.missing_frames += 1
        self._current.frames_considered += 1
        gone_for = at - self._current.last_seen_at
        if (
            self._current.missing_frames > self._policy.max_missing_frames
            or gone_for > self._policy.max_missing_seconds
        ):
            return self._finish(END_REASON_LOST)
        return None

    def _finish(self, reason: str) -> Dwell | None:
        dwell = self._current
        self._current = None
        if dwell is None:
            return None
        dwell = replace(dwell, end_reason=reason)
        return dwell if dwell.qualifies(self._policy) else None

    def _is_same_region(self, candidate, frame_diagonal: float | None) -> bool:
        reference = self._current.reference
        if reference is None:
            return True

        if frame_diagonal:
            shift = float(
                np.linalg.norm(
                    np.asarray(candidate.centre) - np.asarray(reference.centre)
                )
            )
            if shift / frame_diagonal > self._policy.max_centre_shift_fraction:
                return False

        if reference.area_fraction <= 0:
            return True
        ratio = candidate.area_fraction / reference.area_fraction
        return self._policy.min_area_ratio <= ratio <= self._policy.max_area_ratio
