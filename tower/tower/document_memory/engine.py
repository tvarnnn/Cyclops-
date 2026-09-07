"""Frames in, observed documents out.

The whole shape of this file follows from one measurement: OCR costs
~0.3 s per page on the GPU and ~1.9 s on the CPU, while deciding whether
a frame is worth OCR costs ~3 ms. So the pipeline is built to make the
expensive stage rare rather than fast.

    every frame     decode, steady-and-sharp gate                 ~3 ms
    stable frames   text detector, at most 4x a second           ~35 ms
    per dwell       keep the sharpest one or two frames/segment   free
    per dwell       crop, OCR, judge readability                 ~0.3 s each
    per dwell       is this a page already on record?             ~1 ms

Nothing here runs on the Tower event loop. Like World Builder, this is an
engine plus an offline/live driver, because the module contract is a
registry of one with a scalar-shaped result and a page of OCR could not
sit on the frame path regardless.

WHAT CHANGED ON 2026-09-07 AND WHY

The per-frame stage used to be a contour-quad detector with a glyph
statistic (`detect.py`). On 9,199 real frames it fired six times, all on
blinds and keyboards, and zero times after re-derivation; on a rendered
page it needed four clean convex corners. It is replaced by
`gate.PageFinder`: a motion/sharpness gate plus the OCR engine's own text
detector, which finds text where a text detector finds it and needs no
corners. `detect.py` is retained for its warp and for the corpus tests
that measure it; nothing on this path calls it.

The other change is identity. A dwell used to become a new record every
time; now a dwell whose page is already on record -- same words, same
look, per `identity.compare` -- becomes a SIGHTING of that record. That
is what stops twenty seconds of looking at one page from becoming a
library of one page.
"""

import logging
import time
from dataclasses import dataclass, replace

import cv2
import numpy as np

from tower.confidence import Confidence
from tower.document_memory.dwell import DwellPolicy, DwellTracker
from tower.document_memory.gate import PageFinder, crop_region
from tower.document_memory.identity import (
    MIN_WORDS_FOR_TEXT,
    SAME_PAGE_TOKEN_OVERLAP,
    Reading,
    compare,
    perceptual_hash,
    token_overlap,
    tokenise,
)
from tower.document_memory.ocr import OcrResult, TextRecogniser
from tower.document_memory.records import (
    END_REASON_STOPPED,
    REDACTION_NONE,
    TIMING_ASSUMED_INTERVAL,
    TIMING_CAPTURE_JOURNAL,
    TIMING_MIXED,
    DocumentObservation,
    PageObservation,
    Sighting,
)
from tower.storage import new_id

logger = logging.getLogger(__name__)

__all__ = [
    "DocumentMemoryEngine",
    "ObserveResult",
    "SAME_PAGE_TOKEN_OVERLAP",
    "MIN_WORDS_FOR_TEXT",
    "is_same_page",
    "token_overlap",
    "tokenise",
]

# How many recently recorded documents a new dwell is compared against.
# Recent by RECORDING order: what a wearer looks back at is what they
# read a moment ago. Two hundred is far above any real session and
# costs under a millisecond of set arithmetic.
RESIGHT_CANDIDATES = 200


def is_same_page(existing: str, incoming: str) -> bool:
    """Are these two OCR views of the same physical page?

    Only ever asked WITHIN one dwell, which matters: the dwell tracker has
    already established that both frames show the same region of the same
    size in the same place. So the only way they are different pages is
    that the page TURNED while the wearer held position -- which shows up
    as the same rectangle carrying different words.

    **Two views that both recognised NO text are the same page**, and this
    is the case worth spelling out. Overlap between two empty strings is
    zero, so a plain similarity test splits them into two pages and the
    record then claims two pages were observed when nothing at all was
    read. That is an observation gap reported as content.
    """
    existing_blank = not existing.strip()
    incoming_blank = not incoming.strip()
    if existing_blank and incoming_blank:
        return True
    if existing_blank != incoming_blank:
        # One view read text and the other did not. Same page, badly seen
        # once -- merging keeps the reading that succeeded.
        return True
    return token_overlap(existing, incoming) >= SAME_PAGE_TOKEN_OVERLAP


@dataclass(frozen=True)
class ObserveResult:
    """What happened to one frame. Cheap to produce, cheap to log."""

    outcome: str  # "no_page" | "dwelling" | "document" | "resighted"
    document_id: str | None = None
    in_dwell: bool = False
    page_detected: bool = False
    sharpness: float | None = None
    # True when this frame ended a dwell whose page was already on
    # record; `document_id` then names THAT record.
    resighted: bool = False
    # True when the record this frame produced has no readable page: OCR
    # looked and found only noise. Persisted all the same, and counted.
    unreadable: bool = False


@dataclass(frozen=True)
class RecordOutcome:
    """What `_record` did with a finished dwell."""

    document_id: str | None
    resighted: bool = False
    unreadable: bool = False
    pages_ocred: int = 0


# How many frames a FLUSH may OCR. A Stop or a Pause runs the flush on the
# caller's thread -- an HTTP handler, usually -- and a dwell with six
# segments of two frames is twelve pages: ~4 s on the GPU and ~23 s on a
# CPU that `auto` fell back to. Four keeps the worst case near a second
# on the GPU and under eight on the CPU, and the NEWEST frames are the
# ones kept, because the page in view when a person pressed Stop is the
# page they will ask about.
FLUSH_MAX_FRAMES = 4


class DocumentMemoryEngine:
    """Observes documents. Never claims anyone read one.

    `world_id` / `world_session_id` are **supplied by the caller or
    absent**. Nothing here derives them, and this module does not import
    World Builder -- a test enforces that. Absent means unknown, which is
    not the same as "nowhere".
    """

    def __init__(
        self,
        store,
        recogniser: TextRecogniser,
        policy: DwellPolicy | None = None,
        clock=time.time,
        *,
        finder: PageFinder | None = None,
        capture_id: str | None = None,
        world_id: str | None = None,
        world_session_id: str | None = None,
        keep_page_images: bool = False,
        assumed_frame_interval_s: float | None = None,
        resight: bool = True,
    ) -> None:
        self._store = store
        self._recogniser = recogniser
        self._policy = policy or DwellPolicy()
        self._tracker = DwellTracker(self._policy)
        # The detector is the recogniser itself unless a caller says
        # otherwise: one model, loaded once, both stages.
        self._finder = finder or PageFinder(recogniser)
        self._clock = clock
        self._capture_id = capture_id
        self._world_id = world_id
        self._world_session_id = world_session_id
        # OFF by default and it must stay that way. A document's whole
        # point is to be readable, this platform has no redaction, and
        # 06-PRIVACY-DATA is explicit that a crop is not inherently safe.
        self._keep_page_images = keep_page_images
        # A directory of loose jpegs carries no timestamps. Replaying one
        # has to assume a frame interval, and the assumption is recorded
        # on every document it produces rather than hidden here.
        self._assumed_interval = assumed_frame_interval_s
        self._resight = resight
        # The last timestamp handed out, real or synthetic. A synthetic
        # one continues from here rather than restarting at "now", so a
        # single frame with a missing timestamp cannot detach the clock
        # from the stream around it.
        self._last_at: float | None = None
        self._used_real_time = False
        self._used_assumed_time = False
        self._frames_observed = 0
        self._documents_recorded = 0
        self._documents_resighted = 0
        self._dwells_unreadable = 0
        self._pages_ocred = 0
        self._last_outcome: RecordOutcome | None = None

    @property
    def frames_observed(self) -> int:
        return self._frames_observed

    @property
    def documents_recorded(self) -> int:
        return self._documents_recorded

    @property
    def documents_resighted(self) -> int:
        return self._documents_resighted

    @property
    def dwells_unreadable(self) -> int:
        return self._dwells_unreadable

    @property
    def pages_ocred(self) -> int:
        return self._pages_ocred

    @property
    def pages_turned(self) -> int:
        return self._tracker.segments_opened

    @property
    def detections(self) -> int:
        return self._finder.detections

    @property
    def in_dwell(self) -> bool:
        return self._tracker.in_dwell

    @property
    def capture_id(self) -> str | None:
        return self._capture_id

    @property
    def last_gate_verdict(self):
        return self._finder.last_verdict

    def set_capture_id(self, capture_id: str | None) -> None:
        """Adopt the lineage of the frames now arriving.

        A capture id does not exist until a phone connects -- the
        recording that mints it starts at `stream_start` -- so a
        long-lived engine cannot be constructed holding one. This is how
        it learns.

        Takes effect on the NEXT DWELL, never retroactively, and that is
        the point rather than a limitation: a dwell already in progress
        was fed by frames from the previous lineage, and restamping it
        would attach a reading to a recording it did not come from.
        Provenance that can be rewritten after the fact is not
        provenance. The id travels ON THE DWELL, fixed when it started;
        see `Dwell.lineage`.
        """
        self._capture_id = capture_id

    def observe(
        self,
        raw_bytes: bytes,
        *,
        received_at: float | None = None,
        source_seq: int | None = None,
    ) -> ObserveResult:
        """The cheap per-frame path. Raw pixels never leave this method."""
        at = self._timestamp(received_at)
        self._frames_observed += 1

        gray = _decode_gray(raw_bytes)
        if gray is None:
            # A bad frame is a missing frame, not a lost document: a
            # single undecodable frame in the middle of a dwell must not
            # discard the reading in progress.
            finished = self._tracker.observe(None, at=at)
            return self._resolve(finished, page_detected=False)

        candidate = self._finder.find(gray, at=at)
        diagonal = float(np.hypot(*gray.shape[:2]))
        finished = self._tracker.observe(
            candidate,
            at=at,
            gray=gray if candidate is not None else None,
            source_seq=source_seq,
            frame_diagonal=diagonal,
            lineage=self._capture_id,
        )
        return self._resolve(
            finished,
            page_detected=candidate is not None,
            sharpness=None if candidate is None else candidate.sharpness,
        )

    def flush(
        self, reason: str = END_REASON_STOPPED, *, max_frames: int | None = FLUSH_MAX_FRAMES
    ) -> str | None:
        """End an open dwell because the stream ended.

        A wearer who is still reading when the session stops has still
        read; discarding that observation would lose exactly the document
        they were most engaged with. Bounded to `max_frames` OCR calls,
        newest first; `last_outcome` says what was recorded.
        """
        finished = self._tracker.flush(reason)
        if finished is None:
            return None
        return self._record(finished, max_frames=max_frames).document_id

    def abandon(self) -> bool:
        """Drop an open dwell WITHOUT reading it. True if one was open.

        For the one caller that must not run OCR on its own thread: the
        idle timer that stops a session nobody has streamed to for ten
        minutes. Whatever dwell was open then ended at least that long
        ago, and reading it there would build a torch thread pool on a
        thread that is about to die. Logged by the caller, never silent.
        """
        return self._tracker.flush(END_REASON_STOPPED) is not None

    @property
    def last_outcome(self) -> RecordOutcome | None:
        """What the most recent `_record` did. None until a dwell ended."""
        return self._last_outcome

    def release(self) -> None:
        self._recogniser.release()

    def _read(self, page_image):
        """OCR one page, or record that it could not be read.

        A recogniser can fail for ordinary reasons -- a model OOM, a CUDA
        hiccup, a corrupted tensor. Before this guard existed the
        exception propagated out of `observe()` and, in a live
        `--follow-capture` session, ended observation for the REST OF THE
        WEARER'S SESSION with a traceback and nothing persisted.

        Recording an empty page instead is consistent with this module's
        own rule: "we looked and found no readable text" is a real answer,
        and losing the document entirely is not.
        """
        self._pages_ocred += 1
        try:
            return self._recogniser.read(page_image)
        except Exception:
            logger.exception(
                "document memory: OCR failed on a page; recording it as "
                "unreadable and continuing"
            )
            return OcrResult(text="")

    def _timestamp(self, received_at: float | None) -> float:
        """Real receipt time, or a stated assumption -- never a guess.

        With `assumed_frame_interval_s` set, a frame with no timestamp
        advances a synthetic clock at that rate. That is the only way a
        batch of undated jpegs can produce a dwell duration at all, and
        every document it touches is stamped so the duration can never be
        read as measured.

        **The synthetic clock continues from the last timestamp handed
        out, not from "now".** An earlier version anchored to
        `self._clock()` the first time it was needed, so ONE frame with a
        missing timestamp inside a stream of real ones jumped the clock to
        wall-time -- an adversarial review produced a document claiming
        ninety-two days of reading from exactly that.
        """
        if received_at is not None:
            self._used_real_time = True
            self._last_at = received_at
            return received_at

        if self._assumed_interval is None:
            at = self._clock()
            self._last_at = at
            return at

        self._used_assumed_time = True
        at = self._clock() if self._last_at is None else self._last_at + self._assumed_interval
        self._last_at = at
        return at

    def _timing_source(self) -> str:
        if self._used_real_time and self._used_assumed_time:
            return TIMING_MIXED
        if self._used_assumed_time:
            return TIMING_ASSUMED_INTERVAL
        return TIMING_CAPTURE_JOURNAL

    # -- internals -----------------------------------------------------

    def _resolve(self, finished, *, page_detected, sharpness=None) -> ObserveResult:
        if finished is not None:
            recorded = self._record(finished)
            return ObserveResult(
                outcome="resighted" if recorded.resighted else "document",
                document_id=recorded.document_id,
                in_dwell=self._tracker.in_dwell,
                page_detected=page_detected,
                sharpness=sharpness,
                resighted=recorded.resighted,
                unreadable=recorded.unreadable,
            )
        return ObserveResult(
            outcome="dwelling" if self._tracker.in_dwell else "no_page",
            in_dwell=self._tracker.in_dwell,
            page_detected=page_detected,
            sharpness=sharpness,
        )

    def _record(self, dwell, *, max_frames: int | None = None) -> RecordOutcome:
        """The expensive path: crop, OCR, dedup, identify, persist.

        Every qualifying dwell is persisted, readable or not: "we looked
        and found no readable text" is a real answer and a different one
        from "we never looked". The outcome says which, and the session
        counts the unreadable ones so a library filling with them is
        visible.
        """
        outcome = self._record_inner(dwell, max_frames=max_frames)
        self._last_outcome = outcome
        return outcome

    def _record_inner(self, dwell, *, max_frames: int | None) -> RecordOutcome:
        document_id = new_id()
        pages: list[PageObservation] = []
        # Which segment each page came from, parallel to `pages`, so the
        # blank-merge rule cannot fold an unreadable NEXT page into the
        # previous one across a page turn.
        page_segments: list[int] = []
        # OCR returns one region per detected line, so the first region of
        # the first page IS the title line. Reconstructing a title by
        # splitting the joined text would be guessing at structure the
        # recogniser already gave us.
        title_candidate: str | None = None
        selected = dwell.selected
        if max_frames is not None and len(selected) > max_frames:
            # Newest first: the last segments are the pages in view when
            # the session ended, which are the ones a person asks about.
            selected = selected[-max_frames:]

        for index, frame in enumerate(selected):
            page_image = crop_region(frame.gray, frame.candidate)
            result = self._read(page_image)
            readable = _is_readable(result)
            text = result.text if readable else ""
            visual_hash = perceptual_hash(page_image)

            duplicate = _find_duplicate(
                pages, text, segment=frame.segment, page_segments=page_segments
            )
            if duplicate is not None:
                # The same page seen twice within one dwell -- which is the
                # NORMAL case, since best-frame selection deliberately picks
                # two views of one page. Merge rather than duplicate, and
                # keep the higher-confidence reading.
                pages[pages.index(duplicate)] = _merge(
                    duplicate, result, frame, text=text, readable=readable,
                    visual_hash=visual_hash,
                )
                continue

            if title_candidate is None and readable:
                title_candidate = _title_from_regions(result.regions)

            relpath = None
            if self._keep_page_images:
                relpath = self._persist_page_image(document_id, index, page_image)

            page_segments.append(frame.segment)
            pages.append(
                PageObservation(
                    page_index=len(pages),
                    text=text,
                    region_count=result.region_count,
                    mean_region_confidence=result.mean_confidence,
                    min_region_confidence=result.min_confidence,
                    confidence=result.confidence_label if readable else Confidence.UNKNOWN,
                    sharpness=frame.candidate.sharpness,
                    squareness=frame.candidate.squareness,
                    source_seq=frame.source_seq,
                    observed_at=frame.at,
                    image_relpath=relpath,
                    visual_hash=visual_hash,
                    box_count=getattr(frame.candidate, "box_count", 0),
                    readable=readable,
                )
            )

        unreadable = not any(page.readable for page in pages)
        if unreadable:
            # The detector said text; the recogniser could not read it.
            # PERSISTED all the same, as a page that is not readable, and
            # counted.
            self._dwells_unreadable += 1

        if self._resight:
            existing = self._find_resighting(pages)
            if existing is not None:
                sighting = Sighting(
                    observed_at=dwell.started_at,
                    observed_seconds=dwell.seconds,
                    capture_id=dwell.lineage,
                    source_seq=selected[0].source_seq if selected else None,
                    end_reason=dwell.end_reason,
                    frames_considered=dwell.frames_considered,
                )
                updated = _with_sighting(existing, sighting, pages)
                if self._store.update(updated):
                    self._documents_resighted += 1
                    return RecordOutcome(
                        existing.document_id,
                        resighted=True,
                        unreadable=unreadable,
                        pages_ocred=len(selected),
                    )
                # The record vanished between read and write -- a purge
                # raced us. Fall through and record it afresh.

        document = DocumentObservation(
            document_id=document_id,
            observed_at=dwell.started_at,
            recorded_at=self._clock(),
            observed_seconds=dwell.seconds,
            pages=tuple(pages),
            title=title_candidate or _title_of(pages),
            summary=_summarise(pages),
            frames_considered=dwell.frames_considered,
            frames_ocred=len(selected),
            end_reason=dwell.end_reason,
            confidence=_document_confidence(pages),
            # From the DWELL, not from this engine's current field: the
            # frames that produced this document belonged to whatever
            # capture was open when the dwell began. See
            # `set_capture_id`.
            capture_id=dwell.lineage,
            timing_source=self._timing_source(),
            assumed_frame_interval_s=(
                self._assumed_interval if self._used_assumed_time else None
            ),
            world_id=self._world_id,
            world_session_id=self._world_session_id,
            retains_raw_imagery=self._keep_page_images,
            redaction=REDACTION_NONE,
        )
        self._store.append(document)
        self._documents_recorded += 1
        return RecordOutcome(
            document_id, unreadable=unreadable, pages_ocred=len(selected)
        )

    def _find_resighting(self, pages) -> DocumentObservation | None:
        """The recorded document these pages are a later look at, or None.

        The FIRST readable page decides. A multi-page dwell whose first
        page matches a record is that record seen again; whether its
        later pages are new to the record is a question this version
        does not answer -- it keeps the record's pages and adds a
        sighting, which loses nothing that was on disk.
        """
        incoming = next((page for page in pages if page.readable), None)
        if incoming is None:
            return None
        probe = Reading(
            text=incoming.text,
            mean_confidence=incoming.mean_region_confidence,
            visual_hash=incoming.visual_hash,
        )
        if not probe.can_testify:
            return None
        try:
            candidates = self._store.read_recent(RESIGHT_CANDIDATES)
        except Exception:
            logger.exception(
                "document memory: could not read the library for dedup; "
                "recording the dwell as new"
            )
            return None
        for document in reversed(candidates):
            for page in document.pages:
                if not page.readable:
                    continue
                verdict = compare(
                    Reading(page.text, page.mean_region_confidence, page.visual_hash),
                    probe,
                )
                if verdict.same:
                    return document
        return None

    def _persist_page_image(self, document_id, index, page_image) -> str | None:
        ok, buffer = cv2.imencode(
            ".jpg", page_image, [int(cv2.IMWRITE_JPEG_QUALITY), 88]
        )
        if not ok:
            logger.warning("document memory: could not encode page image")
            return None
        filename = f"{document_id}-{index:02d}.jpg"
        self._store.write_page_image(filename, buffer.tobytes())
        from tower.document_memory.store import IMAGES_DIRNAME

        return f"{IMAGES_DIRNAME}/{filename}"


# Below this mean confidence what OCR returned is its noise floor: on
# real 360x640 screen frames it returns fragments at median confidence
# 0.056, and storing those as text makes a library that matches queries
# it should not. Deliberately BELOW `identity.MIN_CONFIDENCE_TO_TESTIFY`
# (0.30): a low-confidence page is kept and labelled LOW, it just cannot
# vouch for whether another page is the same one.
READABLE_MIN_CONFIDENCE = 0.15


def _is_readable(result: OcrResult) -> bool:
    """Did OCR produce words a person could search for?

    At least one token of two characters at a mean confidence above the
    noise floor. The regions are counted either way; the words are kept
    only when they clear this.
    """
    if not result.region_count:
        return False
    if (result.mean_confidence or 0.0) < READABLE_MIN_CONFIDENCE:
        return False
    words = [token for token in tokenise(result.text) if len(token) >= 2]
    return len(words) >= 1


def _with_sighting(existing, sighting, pages) -> DocumentObservation:
    """The record, with one more sighting and the better reading kept.

    The first observation's provenance is untouched: `observed_at`,
    `capture_id`, `pages[*].source_seq` stay what they were. What may
    improve is the TEXT, page by page, when the new look read the same
    page with higher confidence -- a re-sighting is often a better look.
    """
    improved = list(existing.pages)
    for incoming in pages:
        if not incoming.readable:
            continue
        for index, page in enumerate(improved):
            if not page.readable:
                continue
            if token_overlap(page.text, incoming.text) < SAME_PAGE_TOKEN_OVERLAP:
                continue
            better = (incoming.mean_region_confidence or 0.0) > (
                page.mean_region_confidence or 0.0
            )
            improved[index] = replace(
                page,
                text=incoming.text if better else page.text,
                region_count=incoming.region_count if better else page.region_count,
                mean_region_confidence=(
                    incoming.mean_region_confidence
                    if better
                    else page.mean_region_confidence
                ),
                min_region_confidence=(
                    incoming.min_region_confidence
                    if better
                    else page.min_region_confidence
                ),
                confidence=incoming.confidence if better else page.confidence,
                observation_count=page.observation_count + 1,
            )
            break
    return replace(
        existing,
        pages=tuple(improved),
        sightings=existing.sightings + (sighting,),
        summary=_summarise(improved) or existing.summary,
        confidence=_document_confidence(improved),
    )


def _find_duplicate(pages, incoming: str, *, segment: int = 0, page_segments=None):
    """Which already-recorded page, if any, this reading belongs to.

    Two rules, and the difference between them matters.

    A TEXT-to-TEXT match may be against any page in the document: OCR
    reading the same words again is the same page wherever it sits.

    A BLANK match is restricted to the page recorded IMMEDIATELY BEFORE,
    AND ONLY WITHIN THE SAME SEGMENT. The dwell's best frames are the
    sharpest views of a segment, not guaranteed to be the same physical
    page -- a wearer can turn a page without the region moving, and the
    segment boundary is where the tracker saw that happen. An adversarial
    review showed the unrestricted rule erasing a genuinely different
    page whenever OCR happened to fail on it, and a later one showed the
    same-segment restriction missing: an unreadable next page folded
    into the previous page's observation count.
    """
    previous_same_segment = bool(pages) and (
        page_segments is None or page_segments[-1] == segment
    )
    incoming_blank = not incoming.strip()
    if incoming_blank:
        if previous_same_segment:
            # Blank after blank: the same unreadable page. Blank after a
            # reading: the same page, seen badly once -- keep the reading
            # that worked. Either way it is this segment's page.
            return pages[-1]
        return None
    for page in pages:
        if not page.text.strip():
            continue
        if token_overlap(page.text, incoming) >= SAME_PAGE_TOKEN_OVERLAP:
            return page
    # A blank page followed by a reading, in the same segment: the
    # reading belongs to it.
    if previous_same_segment and not pages[-1].text.strip():
        return pages[-1]
    return None


def _decode_gray(raw_bytes: bytes):
    """Decode, or return None. Never raises for bad input.

    Unlike the Lab, a bad frame here is not even a frame-scoped error --
    it is a missing observation in a sequence, and the dwell state machine
    already knows what to do with one of those.
    """
    if not raw_bytes:
        return None
    try:
        array = np.frombuffer(raw_bytes, dtype=np.uint8)
        return cv2.imdecode(array, cv2.IMREAD_GRAYSCALE)
    except cv2.error:
        return None


def _merge(page: PageObservation, result, frame, *, text, readable, visual_hash):
    """Keep the better reading of the same page, and count both views."""
    incoming_confidence = result.mean_confidence or 0.0
    existing_confidence = page.mean_region_confidence or 0.0
    if not readable or incoming_confidence <= existing_confidence:
        return replace(
            page,
            observation_count=page.observation_count + 1,
            visual_hash=page.visual_hash or visual_hash,
        )
    return replace(
        page,
        text=text,
        region_count=result.region_count,
        mean_region_confidence=result.mean_confidence,
        min_region_confidence=result.min_confidence,
        confidence=result.confidence_label,
        sharpness=frame.candidate.sharpness,
        squareness=frame.candidate.squareness,
        observation_count=page.observation_count + 1,
        visual_hash=visual_hash or page.visual_hash,
        box_count=getattr(frame.candidate, "box_count", page.box_count),
        readable=True,
    )


TITLE_MAX_CHARS = 90


def _title_from_regions(regions) -> str | None:
    """The document's first LINE, assembled from the regions that share it.

    A recogniser splits a heading across several regions -- "Attention",
    "Is All You Need" -- so taking the first region alone yields a
    one-word title. Regions arrive in reading order, so the title is the
    leading run whose vertical centre stays within half a line height of
    the first one.

    Extractive throughout. A GENERATED title for text that may have been
    captured only partially would read as a description of the document
    rather than of what was observed, which is the fabrication the
    anti-hallucination requirement forbids.
    """
    if not regions:
        return None
    first = regions[0]
    if first.y_centre is None or not first.height:
        return first.text.strip()[:TITLE_MAX_CHARS] or None

    tolerance = max(first.height * 0.6, 1.0)
    parts = []
    for region in regions:
        centre = region.y_centre
        if centre is None or abs(centre - first.y_centre) > tolerance:
            break
        parts.append(region.text.strip())
    title = " ".join(part for part in parts if part)
    return title[:TITLE_MAX_CHARS] or None


def _title_of(pages) -> str | None:
    """Fallback title when OCR returned no region structure.

    Extractive, always. A GENERATED title for text that may have been
    captured only partially is precisely the fabrication the
    anti-hallucination requirement forbids -- it would read as a
    description of the document rather than of what was observed.
    """
    for page in pages:
        words = page.text.split()
        if len(words) >= 2:
            title = " ".join(words)[:TITLE_MAX_CHARS]
            return title.rsplit(" ", 1)[0] if len(title) == TITLE_MAX_CHARS else title
    return None


def _summarise(pages) -> str:
    """The opening of what was actually captured. Extractive, always.

    Deliberately NOT an LLM summary. There is no local serving path, and
    more importantly a generated summary of partially-captured text would
    read as authoritative while describing pages nobody observed.
    """
    words = []
    for page in pages:
        words.extend(page.text.split())
        if len(words) >= 40:
            break
    if not words:
        return ""
    summary = " ".join(words[:40])
    return summary + ("..." if len(words) > 40 else "")


def _document_confidence(pages) -> Confidence:
    """The WEAKEST page's confidence, not the average.

    A document is only as trustworthy as its worst-read page, and
    averaging would let one crisp page hide one that was barely legible.
    """
    labels = [page.confidence for page in pages if page.text]
    if not labels:
        return Confidence.UNKNOWN
    order = [
        Confidence.UNKNOWN,
        Confidence.LOW,
        Confidence.MEDIUM,
        Confidence.HIGH,
    ]
    return min(labels, key=order.index)
