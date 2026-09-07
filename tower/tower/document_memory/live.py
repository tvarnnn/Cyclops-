"""Document Memory as a live session: Start, Pause, Stop, and provenance.

The lifecycle, the single-slot frame path and the abandoned-load latch are
`tower/live_session.py`, shared with Scene Understanding. What is here is
what is specific to this cartridge, and there are five things.

**1. Stop KEEPS what it recorded.** Scene Understanding's Stop discards
its state; this one must not. That asymmetry is the difference between the
two cartridges rather than an inconsistency: a scene expires the moment
nobody is looking, and a record of what a person read is exactly as true
afterwards.

**2. A dwell in progress is FLUSHED, not dropped.** A wearer still reading
when a session pauses or stops has read something, and throwing it away
would lose a real observation to a UI action. `_on_pause` runs off the
session lock precisely so it can afford the OCR that flushing may trigger.
The flush is bounded by `DwellPolicy.max_segments * best_frames` pages,
which at ~0.3 s each on the GPU sits inside the session's 5 s join.

**3. Provenance comes from the capture, and it comes from outside.** A
`DocumentObservation` carries the `capture_id` its frames came from and,
per page, the `source_seq` of the frame that was actually read. Neither
exists until a phone connects and the recorder mints a capture id, which
is why `capture_started` is a hook the connection handler calls rather
than something this class could look up. When no recorder is armed, the
capture id is None -- and the adapter publishes
`capture_id_validated: false` alongside it rather than implying a pointer
that resolves.

**4. Retention is enforced HERE, on a real cadence.** Before this existed,
`prune_expired` had exactly one production caller -- the end of
`scripts/document_memory_session.py` -- so a long-running Tower would
never have pruned at all. This session prunes at start and after every
document it records.

**5. A session nobody is streaming to winds itself down.** The OCR reader
holds ~1.4 GB of GPU memory while loaded. A session started by a phone
that then disconnected and never came back would hold it forever, on a
Tower whose next cartridge wants that memory. So when the stream closes,
a timer starts; if no stream reopens within `idle_stop_s`, the session
stops itself -- flushing, as any Stop does. A reconnect within the
window cancels it and the session carries on, which is the case a phone
dropping Wi-Fi for ten seconds needs.

WHAT THIS COSTS, AND WHY IT IS SAFE TO LEAVE RUNNING

The per-frame path is the steady-and-sharp gate at ~3 ms, plus the text
detector at ~35 ms on at most a quarter of the steady frames. The
expensive path -- crop plus OCR at ~0.3 s a page on the GPU -- runs at
most `best_frames` times per segment of a completed dwell, capped
structurally rather than by a caller remembering to be careful.

The OCR reader itself costs about 1.3 s to construct on the GPU (5 s on
CPU), ONCE per session. It is loaded in `_create`, on the worker thread,
so that cost is paid inside `state: "starting"` where a client can see
it -- rather than lazily, inside the first frame that happens to complete
a dwell, which is where `EasyOcrRecogniser.read` would otherwise put it.
"""

import logging
import threading

from tower.document_memory.dwell import DwellPolicy
from tower.document_memory.engine import DocumentMemoryEngine
from tower.document_memory.records import END_REASON_STOPPED
from tower.document_memory.store import DocumentStore
from tower.live_session import LiveSession

logger = logging.getLogger(__name__)

# The default window a live session writes under, in days.
#
# 30, matching `scripts/document_memory_session.py`'s own default rather
# than `DocumentStore`'s constructor default of None. That constructor
# default means KEEP FOREVER, and the store's own docstring says
# "'forever' has to be chosen" -- a web process that adopted it would be
# choosing it by omission, which is the one way it must not be chosen.
DEFAULT_RETENTION_DAYS = 30.0

SECONDS_PER_DAY = 86400.0

# How long a running session survives with no stream before stopping
# itself. Ten minutes: long enough for a phone to drop and rejoin a
# network, far shorter than "until somebody restarts the Tower".
DEFAULT_IDLE_STOP_S = 600.0

# Above this many stored documents the session says so on every status.
#
# It does NOT delete. Evicting a wearer's memories because a count got
# large is a policy decision with a privacy dimension, and the honest
# thing for this lane to do is surface the number rather than quietly
# pick an answer. Retention -- an age, chosen by an operator -- is the
# eviction rule this cartridge has, and it is enforced.
LIBRARY_SOFT_LIMIT = 10_000


class DocumentLive(LiveSession):
    """One document capture session, driven by frames from elsewhere."""

    name = "Document"

    # `engine.observe()` appends to the store before this session ever
    # sees the result, so a result that arrives after a Pause or a Stop
    # is already on disk. Reporting it is the only way the counters can
    # agree with what is there.
    commits_during_consume = True

    def __init__(
        self,
        root,
        *,
        retention_days: float | None = DEFAULT_RETENTION_DAYS,
        policy: DwellPolicy | None = None,
        recogniser_factory=None,
        keep_page_images: bool = False,
        follow_stream: bool = False,
        device: str = "auto",
        idle_stop_s: float | None = DEFAULT_IDLE_STOP_S,
        **kwargs,
    ) -> None:
        # OFF by default, unlike Scene Understanding's, and the asymmetry
        # is the difference between the two cartridges: this one WRITES.
        # A session that persists what a wearer read gets an explicit
        # start, which is the standard `06-PRIVACY-DATA.md` holds the
        # dataset recorder to -- "arming is not recording".
        #
        # The cost is smaller than it looks: the half of this cartridge a
        # phone reaches is the library, over HTTP, and that works whether
        # or not anything is currently recording.
        super().__init__(follow_stream=follow_stream, **kwargs)
        self._root = root
        self._retention_days = retention_days
        self._policy = policy
        self._recogniser_factory = recogniser_factory
        self._device = device
        self._idle_stop_s = idle_stop_s
        # OFF, and it must stay off by default. A document's whole point
        # is to be readable and this platform has no redaction, so a
        # persisted page image is an unredacted photograph of whatever
        # the wearer was reading. `engine.py` says the same.
        self._keep_page_images = bool(keep_page_images)

        self._store = None
        self._capture_id = None
        self._documents_recorded = 0
        self._documents_resighted = 0
        self._dwells_unreadable = 0
        self._pages_detected = 0
        self._dwells_started = 0
        self._in_dwell = False
        self._last_document_id = None
        self._last_document_at = None
        self._library_count = None
        self._pruned_documents = 0
        self._prune_incomplete = False
        self._ocr_device = None
        self._idle_timer: threading.Timer | None = None
        self._idle_stops = 0
        # True only while the idle timer's own thread is inside `stop()`,
        # so `_on_pause` knows not to run OCR there.
        self._idle_abandon = False
        self._flushed_document_id = None

    # -- capture lineage -----------------------------------------------

    def capture_started(self, capture_id) -> None:
        """Adopt the lineage of the frames about to arrive.

        Takes effect on the NEXT document, not retroactively: an engine
        already mid-dwell keeps the capture id it was built with, because
        rewriting it would attach frames to a recording they did not come
        from. That is the whole reason this field exists.
        """
        with self._condition:
            self._capture_id = capture_id
            engine = self._engine
        if engine is not None:
            # The engine stamps `capture_id` onto each document it
            # records. Setting it here rather than rebuilding the engine
            # keeps an in-progress dwell alive across a `stream_start`
            # that merely re-armed the recorder.
            engine.set_capture_id(capture_id)

    def capture_stopped(self, capture_id) -> None:
        with self._condition:
            if self._capture_id == capture_id:
                self._capture_id = None
                engine = self._engine
            else:
                engine = None
        if engine is not None:
            engine.set_capture_id(None)

    # -- idle wind-down ------------------------------------------------

    def stream_opened(self, owner=None) -> None:
        self._cancel_idle_timer()
        super().stream_opened(owner)

    def stream_closed(self, owner=None) -> None:
        super().stream_closed(owner)
        # Only a session that is still running after the base class had
        # its say needs a timer: one that followed the stream has already
        # stopped, and one that was never started has nothing to release.
        if self._idle_stop_s is None:
            return
        with self._condition:
            running = self._state in ("running", "starting", "paused")
        if running:
            self._arm_idle_timer()

    def offer_frame(self, raw_bytes, *, received_at=None, source_seq=None) -> None:
        # A frame arriving is a stream that is open, whatever the hooks
        # said: a Tower without a recorder never calls `stream_opened`.
        if self._idle_timer is not None:
            self._cancel_idle_timer()
        super().offer_frame(raw_bytes, received_at=received_at, source_seq=source_seq)

    def _arm_idle_timer(self) -> None:
        self._cancel_idle_timer()
        timer = threading.Timer(self._idle_stop_s, self._idle_stop)
        timer.daemon = True
        timer.name = "tower-Document-idle"
        self._idle_timer = timer
        timer.start()
        logger.info(
            "[Tower][Document] no stream; the session will stop itself in "
            "%.0f s unless one opens",
            self._idle_stop_s,
        )

    def _cancel_idle_timer(self) -> None:
        timer, self._idle_timer = self._idle_timer, None
        if timer is not None:
            timer.cancel()

    def _idle_stop(self) -> None:
        self._idle_timer = None
        with self._condition:
            running = self._state in ("running", "starting", "paused")
        if not running:
            return
        logger.info(
            "[Tower][Document] stopping an idle session: no stream for %.0f s",
            self._idle_stop_s,
        )
        self._idle_stops += 1
        self._idle_abandon = True
        try:
            self.stop()
        finally:
            self._idle_abandon = False

    # -- hooks ---------------------------------------------------------

    def _retention_seconds(self):
        if self._retention_days is None or self._retention_days <= 0:
            return None
        return self._retention_days * SECONDS_PER_DAY

    def _create(self):
        store = DocumentStore(self._root, retention_seconds=self._retention_seconds())
        recogniser = self._make_recogniser()
        load = getattr(recogniser, "load", None)
        if load is not None:
            # Explicitly, here, on the worker thread. `read()` would do it
            # lazily on the first completed dwell instead, which puts the
            # load inside a frame and inside `state: "running"`, where
            # nothing reports it.
            load()
        with self._condition:
            self._ocr_device = getattr(recogniser, "device", None)
        engine = DocumentMemoryEngine(
            store,
            recogniser,
            self._policy,
            capture_id=self._capture_id,
            keep_page_images=self._keep_page_images,
        )
        self._store = store
        self._prune(store)
        return engine

    def _make_recogniser(self):
        if self._recogniser_factory is not None:
            return self._recogniser_factory()
        from tower.document_memory.ocr import EasyOcrRecogniser

        return EasyOcrRecogniser(device=self._device)

    def _engine_name(self, engine):
        return getattr(getattr(engine, "_recogniser", None), "name", None)

    def _consume(self, engine, raw_bytes, received_at, source_seq):
        result = engine.observe(
            raw_bytes, received_at=received_at, source_seq=source_seq
        )
        if result.outcome in ("document", "resighted"):
            # Off the session lock, and only when something was written.
            # Retention that runs once at process exit is retention a
            # long-lived Tower never applies.
            self._prune(self._store)
        return result

    def _publish(self, result, received_at: float, now: float) -> None:
        if result.page_detected:
            self._pages_detected += 1
        if result.in_dwell and not self._in_dwell:
            self._dwells_started += 1
        self._in_dwell = bool(result.in_dwell)
        if result.unreadable:
            self._dwells_unreadable += 1
        if result.document_id is not None:
            if result.resighted:
                self._documents_resighted += 1
            else:
                self._documents_recorded += 1
            self._last_document_id = result.document_id
            self._last_document_at = received_at

    def _on_pause(self, engine) -> None:
        """Close an open dwell rather than losing it.

        Returns a document id when a dwell qualified. It is not counted
        into `documents_recorded` here and that is deliberate: the count
        a status reports is of documents this SESSION recorded while
        observing, and a flush at teardown is reported separately as
        `flushed_document_id` so an operator can see that the last
        document arrived because the session ended rather than because
        the wearer looked away.
        """
        if self._idle_abandon:
            # The idle timer's thread. No OCR here: a torch thread pool
            # built on a `threading.Timer` thread is never reclaimed, and
            # the dwell it would read ended ten minutes ago in any case.
            if engine.abandon():
                logger.warning(
                    "[Tower][Document] an idle stop dropped a dwell that "
                    "was still open when the stream closed; it was not read"
                )
            return
        document_id = engine.flush(END_REASON_STOPPED)
        if document_id is None:
            return
        outcome = engine.last_outcome
        with self._condition:
            if outcome is not None and outcome.resighted:
                self._documents_resighted += 1
            else:
                self._documents_recorded += 1
            if outcome is not None and outcome.unreadable:
                self._dwells_unreadable += 1
            self._last_document_id = document_id
            self._flushed_document_id = document_id
        self._prune(self._store)

    def _teardown(self, engine) -> None:
        engine.release()

    def _on_start_locked(self) -> None:
        self._documents_recorded = 0
        self._documents_resighted = 0
        self._dwells_unreadable = 0
        self._pages_detected = 0
        self._dwells_started = 0
        self._in_dwell = False
        self._last_document_id = None
        self._last_document_at = None
        self._flushed_document_id = None
        self._pruned_documents = 0
        self._prune_incomplete = False
        self._ocr_device = None

    def _on_stop_locked(self) -> None:
        """Nothing is discarded. See the module header.

        `_in_dwell` is cleared because it describes the engine, which is
        about to be released; the counters and the document ids survive so
        an operator can read what the session did after it ended.
        """
        self._in_dwell = False
        self._cancel_idle_timer()

    def _prune(self, store) -> None:
        """Apply retention. Never raises, always reports what it could not do.

        A prune that fails silently is a retention promise that is not
        kept, and the only visible symptom would be a directory that grew.
        `DocumentStore.purge`/`prune_expired` already report incomplete
        deletion rather than raising -- a locked image on Windows is the
        routine case -- so this carries that report onto the status
        instead of dropping it.
        """
        if store is None:
            return
        try:
            report = store.prune_expired()
            count = store.count()
        except Exception:
            logger.exception(
                "[Tower][Document] retention could not be applied; the store "
                "may hold records past its window"
            )
            with self._condition:
                self._prune_incomplete = True
            return
        with self._condition:
            self._pruned_documents += int(report.get("documents_removed", 0))
            if not report.get("complete", True):
                self._prune_incomplete = True
            self._library_count = count

    def _extra_status(self) -> dict:
        return {
            "recogniser": getattr(self, "_engine_label", None),
            "ocr_device": self._ocr_device,
            "capture_id": self._capture_id,
            "capture_id_validated": False,
            "in_dwell": bool(self._in_dwell),
            "dwells_started": self._dwells_started,
            "pages_detected": self._pages_detected,
            "documents_recorded": self._documents_recorded,
            "documents_resighted": self._documents_resighted,
            "dwells_unreadable": self._dwells_unreadable,
            "last_document_id": self._last_document_id,
            "last_document_at": self._last_document_at,
            "flushed_document_id": self._flushed_document_id,
            "keeps_page_images": bool(self._keep_page_images),
            "follows_stream": bool(self._follow_stream),
            "idle_stop_seconds": self._idle_stop_s,
            "idle_stop_pending": self._idle_timer is not None,
            "retention_days": self._retention_days,
            "documents_pruned": self._pruned_documents,
            # True when a deletion could not be completed -- a locked
            # image file, most often. Reported rather than logged,
            # because a retention promise that quietly failed looks
            # exactly like one that was kept.
            "retention_incomplete": bool(self._prune_incomplete),
            "library_count": self._library_count,
            "library_soft_limit": LIBRARY_SOFT_LIMIT,
            "library_over_soft_limit": bool(
                self._library_count is not None
                and self._library_count > LIBRARY_SOFT_LIMIT
            ),
            "library_soft_limit_note": (
                "a soft limit is reported, never enforced. This session "
                "evicts by AGE only: deleting a wearer's memories because a "
                "count grew is a policy decision, not a cleanup"
            ),
        }
