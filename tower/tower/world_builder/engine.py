"""The World Builder engine: observe cheaply, build expensively.

Two entry points with deliberately different cost profiles:

``observe()`` runs per delivered frame and is cheap (~5 ms measured at
360x640 against a ~300 ms interval). It decodes, scores, tracks, decides,
and on acceptance persists one keyframe. It never calls a geometry
backend.

``build()`` runs at stop time, reads the persisted keyframe journal back,
and writes the reconstruction. It is never reachable from the frame
path, which is what keeps a multi-second write off the event loop.

It used to be a full re-solve as well -- re-reading every keyframe,
re-decoding every JPEG and re-detecting features on all N -- which made
a walk rebuilt every k keyframes cost O(N^2/k), so asking for MORE live
updates cost more than the walk. It now extends one live solve as each
keyframe is accepted (see _LiveSolve) and a rebuild is a flush. The
from-scratch path is still here and still correct: it is what a cold
rebuild, a re-derive, and any keyframe set this engine did not itself
observe fall back to.

That split is also why live-versus-offline is a *driver* choice rather
than an architecture choice: the offline script calls exactly the same
observe() a future module adapter would.
"""

import logging
import time
from dataclasses import dataclass, field, replace

from tower.confidence import Confidence
from tower.world_builder.backend import KeyframeInput
from tower.world_builder.backends import BACKEND_AUTO, select_backend
from tower.world_builder.events import EventLog, WorldEvent
from tower.world_builder.frontend import FrameTracker, analyse_frame, decode_gray
from tower.world_builder.keyframes import (
    KeyframePolicy,
    KeyframeSelector,
)
from tower.world_builder.redaction import REDACTION_NONE, FaceRedactor
from tower.world_builder.records import (
    FINAL_SOLVE_PENDING,
    FINALIZATION_PENDING,
    FINALIZATION_STATES,
    POST_FINALIZATION_STAGES,
    STAGE_STATES,
    CameraIntrinsics,
    Keyframe,
    KeyframeEdge,
    ScaleState,
    Session,
    World,
    make_keyframe_id,
    new_id,
)
from tower.world_builder.schema import (
    END_REASON_STOP,
    POSE_STATUS_ANCHOR,
    POSE_STATUS_SOLVED,
    SCALE_MEASURED,
    SCALE_RELATIVE,
    SCALE_UNKNOWN,
)
from tower.world_builder import global_solve
from tower.world_builder.store import WorldStore, compute_input_digest

logger = logging.getLogger(__name__)


class SessionNotActiveError(RuntimeError):
    """observe()/stop_session() called without a live session."""


class ImagesPurgedError(RuntimeError):
    """A rebuild was attempted on a world whose imagery has been deleted.

    Raised loudly rather than producing an empty reconstruction: silently
    returning nothing would look like a mapping failure rather than the
    consequence of a deliberate privacy action.
    """


@dataclass(frozen=True)
class ObserveResult:
    outcome: str
    reason: str
    keyframe_id: str | None = None
    frames_observed: int = 0
    keyframes_accepted: int = 0


@dataclass(frozen=True)
class SessionSummary:
    session_id: str
    frames_observed: int
    keyframes_accepted: int
    rejected_by_reason: dict
    segments: int
    end_reason: str


@dataclass(frozen=True)
class BuildResult:
    world_id: str
    session_id: str
    backend_id: str
    keyframes: int
    poses_solved: int
    poses_refused: int
    points: int
    segments: int
    scale_state: str
    downgraded_from: str | None = None
    diagnostics: dict = field(default_factory=dict)


# How many consecutive segments may produce NOTHING before a solve break
# stops being allowed to start another one.
#
# Two earlier rules were measured and refused, both for the same reason.
#
# `MIN_SOLVED_BEFORE_RESTART = 2` (restart only if the broken chain had
# solved two poses) leaves the cascade fully intact for the case that
# matters most: when the SEED PAIR fails the chain has solved nothing --
# the anchor is not a solved pose -- so no threshold >= 1 is ever met.
# Measured on capture 4fea31e2, one seed-pair failure stranded 48 of 50
# keyframes.
#
# `MIN_KEYFRAMES_BEFORE_RESTART` (wait for N keyframes) fails for a
# subtler reason: `chain_broken` is an EDGE. Decline the restart at that
# instant and no second opportunity ever arrives, however many keyframes
# accumulate. Swept at 3, 4, 6 and 8 it reproduced the solved-pose
# numbers exactly and never recovered 4fea31e2.
#
# Delay is the wrong axis anyway. Once a chain is broken every further
# keyframe in that segment is refused without geometry, so waiting buys
# nothing -- the only real question is whether restarting is worth it.
#
# It is worth it unless the region has already proven unmappable. A
# segment that ends having solved NOTHING is that evidence. At 1, a
# restart is allowed unless the segment immediately before it produced
# nothing -- so a region that is building geometry restarts freely, and
# genuinely untrackable footage stops after one wasted attempt instead of
# shattering into dozens of two-keyframe shards.
#
# THE VALUE IS CHOSEN ON MARGINAL EFFICIENCY, measured over the pinned
# eight. Every larger cap buys more geometry and costs more fragments,
# monotonically, with no knee -- so the question is what each added
# fragment card is worth:
#
#   cap   segments   solved   points   poses per ADDED segment
#     -        127      346    47429   (baseline)
#     1        230      591    75369   2.38
#     2        306      695    86230   1.95
#     3        360      749    91130   1.73
#  none        470      863   107005   1.51
#
# The return declines monotonically, so 1 is where a fragment card buys
# the most reconstruction.
#
# What settled it against a larger cap: REGISTRATION IS INVARIANT across
# the whole range. On e1c52b9f, caps of 1, 2 and none all produce exactly
# the same registered cluster -- 3 segments, 5603 points, the same two
# admitted pairs. The extra fragments buy raw geometry, not cross-segment
# coherence, so paying more fragment cards for them is a bad trade while
# the viewer cannot rank or collapse them.
#
# The larger caps remain available and are worth revisiting the moment
# fragments can be ranked: the uncapped variant is +517 solved poses over
# baseline and grows the largest single coherent piece 28656 -> 32756.
MAX_BARREN_SEGMENTS = 1

class WorldBuilderEngine:
    def __init__(
        self,
        store: WorldStore,
        policy: KeyframePolicy | None = None,
        backend_name: str = BACKEND_AUTO,
        clock=time.time,
        redactor_factory=None,
        relocalizer: str | None = None,
    ) -> None:
        self._store = store
        # The look-back relocalizer's mode (`off` / `prompt` / `silent`,
        # relocalizer.py). None reads `TOWER_WORLD_RELOCALIZER` at each
        # session start -- the builder process inherits the Tower's
        # environment -- and unset is `off`: nothing runs, nothing is
        # journaled, and every journal is byte-identical to before.
        self._relocalizer_mode = relocalizer
        self._reloc = None
        self._policy = policy or KeyframePolicy()
        self._backend_name = backend_name
        # A factory rather than an instance: a redactor holds a loaded
        # detector, and a session that never starts should not pay for
        # one. Injectable so a test can drive both halves without a model
        # file, and so an operator can point at different weights.
        self._redactor_factory = redactor_factory or FaceRedactor
        # None until a session starts. `_persist_keyframe` only ever runs
        # inside one, and a half-built placeholder here would be a worse
        # failure than an AttributeError if that ever stopped being true.
        self._redactor = None
        self._clock = clock

        self._session: Session | None = None
        self._selector: KeyframeSelector | None = None
        self._tracker: FrameTracker | None = None
        self._events: EventLog | None = None
        self._segment_index = 0
        # The sequence renumbering. See `observe`.
        self._seq_offset = 0
        self._last_effective_seq: int | None = None
        # The size every frame of this session must be. Set by the
        # first frame rather than from the declared size, because
        # the declared size is what the phone SAYS and this is what
        # it actually sent. See `observe`.
        self._frame_shape = None
        self._segment_solved = 0
        self._barren_segments = 0
        self._segments_used: set[int] = set()
        self._rejected: dict[str, int] = {}
        # Survives stop_session() on purpose: the usual order is
        # observe... stop_session() build(), and throwing the solve away
        # at stop would put the whole cost straight back.
        self._live: _LiveSolve | None = None

    # -- lifecycle -----------------------------------------------------

    def create_world(self, display_name: str | None = None) -> str:
        now = self._clock()
        world = World(
            world_id=new_id(),
            created_at=now,
            updated_at=now,
            display_name=display_name,
        )
        self._store.write_world(world)
        return world.world_id

    def start_session(
        self,
        world_id: str,
        *,
        intrinsics: CameraIntrinsics | None = None,
        frame_source: str = "unknown",
        declared_size: tuple[int, int] | None = None,
        capture_id: str | None = None,
    ) -> str:
        world = self._store.read_world(world_id)
        self._store.acquire_writer_lock(world_id)

        session = Session(
            session_id=new_id(),
            world_id=world_id,
            started_at=self._clock(),
            frame_source=frame_source,
            capture_id=capture_id,
            declared_width=declared_size[0] if declared_size else None,
            declared_height=declared_size[1] if declared_size else None,
            intrinsics=intrinsics or CameraIntrinsics.unknown(),
        )
        self._store.write_session(session)
        self._store.write_world(
            replace(
                world,
                updated_at=self._clock(),
                session_ids=world.session_ids + (session.session_id,),
            )
        )

        self._session = session
        self._redactor = self._redactor_factory()
        # See _persist_keyframe: once any keyframe of this session has been
        # written unredacted, the session can never again say otherwise.
        self._unredacted_keyframes = 0
        if not self._redactor.available:
            logger.warning(
                "[Tower][WorldBuilder] persisting UNREDACTED keyframes: %s",
                self._redactor.unavailable_reason,
            )
        self._selector = KeyframeSelector(self._policy)
        self._tracker = FrameTracker()
        self._events = EventLog(
            self._store, world_id, session.session_id, clock=self._clock
        )
        self._segment_index = 0
        self._seq_offset = 0
        self._last_effective_seq = None
        # The size every frame of this session must be. Set by the
        # first frame rather than from the declared size, because
        # the declared size is what the phone SAYS and this is what
        # it actually sent. See `observe`.
        self._frame_shape = None
        # Reset with its sibling. start_session resets every other piece
        # of per-session state; leaving this one behind let a new
        # session inherit a restart budget the previous one earned.
        self._segment_solved = 0
        self._barren_segments = 0
        self._segments_used: set[int] = set()
        self._rejected = {}
        self._events.append("session_started", {"frame_source": frame_source})
        self._start_relocalizer(session)
        self._open_live_solve(session)
        return session.session_id

    def observe(
        self,
        raw_bytes: bytes,
        *,
        received_at: float | None = None,
        source_seq: int,
        wire_seq: int | None = None,
        tx_seq: int | None = None,
    ) -> ObserveResult:
        if self._session is None:
            raise SessionNotActiveError("observe() requires an active session")

        session = self._session
        received_at = self._clock() if received_at is None else received_at
        self._session = replace(
            session, frames_observed=session.frames_observed + 1
        )
        session = self._session

        # SEQUENCE NUMBERS RESTART WHEN THE GLASSES DO, AND THE SESSION
        # DOES NOT. `make_keyframe_id` says it: `source_seq` "resets when
        # the glasses session restarts". A builder follows a capture
        # LINEAGE -- the reconnect work of this campaign -- so one session
        # can now see the sequence start again mid-walk (a re-pair, an app
        # relaunch inside the resume grace). The keyframe id and the image
        # file name are both `source_seq`, so a repeated number OVERWROTE
        # the earlier keyframe's image on disk and gave the journal two
        # keyframes with one id; the final solve keys COLMAP on the file
        # name and died with a SQLite constraint abort -- "Partial" over a
        # full walk, in four of a dress rehearsal's runs. The sequence is
        # renumbered onto a monotonic one the moment it goes backwards;
        # `wire_seq` and `tx_seq` keep the raw numbers.
        effective_seq = source_seq + self._seq_offset
        if (
            self._last_effective_seq is not None
            and effective_seq <= self._last_effective_seq
        ):
            self._seq_offset = self._last_effective_seq + 1 - source_seq
            effective_seq = source_seq + self._seq_offset
            logger.warning(
                "[WorldBuilder] source_seq went backwards (%s after %s): the "
                "sender restarted; renumbering from %s so no keyframe is "
                "overwritten",
                source_seq, self._last_effective_seq, effective_seq,
            )
            self._events.append(
                "source_seq_restarted",
                {"source_seq": source_seq, "renumbered_to": effective_seq},
            )
        self._last_effective_seq = effective_seq
        source_seq = effective_seq

        try:
            gray = decode_gray(raw_bytes)
        except ValueError:
            # One undecodable frame is frame-scoped: drop it, keep going.
            self._note_rejected("malformed_frame")
            self._events.append("frame_rejected", {"reason": "malformed_frame"})
            return self._result("reject", "malformed_frame")

        # A FRAME OF A DIFFERENT SIZE IS REJECTED, NOT TRACKED.
        #
        # `MotionTracker.measure` feeds this frame and a stored reference
        # frame straight into `cv2.calcOpticalFlowPyrLK`, which asserts
        # they are the same size -- in C, as a `cv2.error`, which is not a
        # `ValueError` and so walks straight past the guard above.
        # `world_build_session.py` catches only `OSError` around the frame
        # loop, so it reaches the outermost `except BaseException`: the
        # session is closed `end_reason: error`, finalization
        # `interrupted`, and every remaining frame of the walk is
        # discarded. A reviewer drove exactly that through a real Tower --
        # 220 frames with a rung change at frame 120 -- and the wearer,
        # who walked the whole room, gets "Interrupted".
        #
        # Rejecting is not merely safer than crashing, it is the correct
        # answer: the calibration is per-resolution and exact, so a frame
        # at a size this session is not calibrated for could not have
        # produced a usable pose anyway. `_require_matching_resolution`
        # would refuse the build at the next rebuild for the same reason
        # -- and that exception is raised in one place and caught
        # NOWHERE, so this guard is what stops it ever being reached.
        #
        # Counted and named, so a walk that quietly changed rung is
        # visible afterwards instead of merely short.
        if self._frame_shape is None:
            self._frame_shape = gray.shape[:2]
        elif gray.shape[:2] != self._frame_shape:
            self._note_rejected("frame_size_changed")
            self._events.append("frame_rejected", {
                "reason": "frame_size_changed",
                "expected": list(self._frame_shape),
                "received": list(gray.shape[:2]),
            })
            return self._result("reject", "frame_size_changed")

        quality = analyse_frame(gray)
        self._selector.note_frame(quality)
        motion = self._tracker.measure(gray)
        decision = self._selector.evaluate(quality, motion)

        if decision.lost:
            # A new segment: poses either side are NOT in a common frame,
            # and the records say so rather than implying continuity.
            self._tracker.reset()
            self._selector.note_lost()
            self._segment_index += 1
            # A new segment starts with nothing solved. Without this the
            # restart budget would be inherited from the segment that
            # just died.
            self._segment_solved = 0
            # A tracking loss is not evidence that the region is
            # unmappable -- the wearer moved too fast, that is all. It
            # must not spend the restart budget.
            self._barren_segments = 0
            if self._live is not None:
                self._live.close_segment(self._segment_index)
            self._note_rejected(decision.reason)
            lost = self._events.append(
                "tracking_lost", {"segment_index": self._segment_index}
            )
            # The look-back relocalizer: a loss with no episode open opens
            # one (the lost frame is its first scan); one inside an open
            # episode joins it. Never waits on the matcher.
            self._recovery(
                lambda r: r.note_lost(lost.at)
                + r.note_frame(gray, source_seq, None, self._clock())
            )
            return self._result(decision.outcome, decision.reason)

        if not decision.accepted:
            self._note_rejected(decision.reason)
            self._recovery(
                lambda r: r.note_frame(gray, source_seq, None, self._clock())
            )
            return self._result(decision.outcome, decision.reason)

        keyframe, image_bytes = self._persist_keyframe(
            gray_shape=gray.shape,
            raw_bytes=raw_bytes,
            received_at=received_at,
            source_seq=source_seq,
            wire_seq=wire_seq,
            tx_seq=tx_seq,
            quality=quality,
            motion=motion,
            reason=decision.reason,
        )
        chain_broke_pending = False
        if self._live is not None:
            # The REDACTED bytes, because those are what landed on disk
            # and therefore what build() decodes. Feeding `gray` here
            # would solve against pixels no rebuild can ever reproduce --
            # redaction costs about 9% of the point cloud when a face is
            # in frame, so the two would quietly disagree. `is` because
            # redact() hands back the very object it was given whenever
            # it changed nothing, which is the overwhelmingly common case
            # and makes this free.
            step = self._live.extend(
                keyframe.keyframe_id,
                gray if image_bytes is raw_bytes else decode_gray(image_bytes),
            )
            if step is not None and step.pose.status == POSE_STATUS_SOLVED:
                self._segment_solved += 1
            if (
                step is not None
                and step.chain_broken
                and self._barren_segments < MAX_BARREN_SEGMENTS
            ):
                # The solve chain failed on this keyframe. Everything
                # after it shares no coordinate frame with what came
                # before -- which is precisely what a segment boundary
                # means here, and what the comment on the tracking-loss
                # increment above already says.
                #
                # Without this, `chain.broken` latches and every later
                # keyframe in the segment is refused WITHOUT ORB
                # detection, matching, or any geometry attempted.
                # Measured on capture 22e9d428: 354 refusals from 26
                # decisions, 328 cascade, 0 of 26 segments ever
                # recovering. classical.py has claimed for a long time
                # that "the engine turns this into a new segment"; it did
                # not, and this makes the claim true.
                #
                # The tracker keeps its reference. Tracking is healthy;
                # only the solve failed, and discarding the reference
                # would manufacture a tracking loss out of a geometry
                # failure -- measured at 424 -> 389 solved poses when the
                # reference is genuinely lost here.
                #
                # Honest note: inserting a reset at THIS line is an
                # equivalent mutation, because set_reference() below runs
                # unconditionally and re-establishes it. The invariant is
                # real and worth stating; this position does not enforce
                # it. The test asserts the outcome (the tracker still has
                # a reference), not this line.
                # A segment that ends having solved nothing is evidence
                # the region is unmappable; a run of them stops further
                # restarts. One that produced geometry clears the count.
                self._barren_segments = (
                    self._barren_segments + 1 if self._segment_solved == 0 else 0
                )
                self._segment_index += 1
                self._segment_solved = 0
                self._live.close_segment(self._segment_index)
                chain_broke_pending = True

        self._segments_used.add(keyframe.segment_index)
        self._tracker.set_reference(gray)
        self._selector.note_accepted()
        self._session = replace(
            self._session,
            keyframes_accepted=self._session.keyframes_accepted + 1,
        )
        if chain_broke_pending:
            # AFTER the keyframe it belongs to. `tracking_lost` always
            # precedes the keyframes of the segment it announces, because
            # that branch returns before persisting. This one cannot: the
            # breaker was already stamped with the OLD index. Emitting it
            # first inverted the convention, so a consumer rebuilding
            # segmentation from the journal misattributed exactly one
            # keyframe per break.
            self._events.append(
                "solve_chain_broken", {"segment_index": self._segment_index}
            )
        self._events.append(
            "keyframe_accepted",
            {
                "keyframe_id": keyframe.keyframe_id,
                "reason": decision.reason,
                "segment_index": keyframe.segment_index,
            },
        )
        # A reference for the NEXT loss, an anchor candidate for an open
        # episode, and -- while one is open -- a scan frame that anchors
        # itself. The relocalizer holds pixels in memory only; it journals
        # numbers and keyframe ids, never imagery.
        self._recovery(
            lambda r: (
                r.note_keyframe(keyframe.keyframe_id, source_seq, gray)
                or r.note_frame(
                    gray, source_seq, keyframe.keyframe_id, self._clock()
                )
            )
        )
        return self._result(
            decision.outcome, decision.reason, keyframe_id=keyframe.keyframe_id
        )

    def stop_session(
        self,
        reason: str = END_REASON_STOP,
        *,
        hold_lock: bool = False,
        capture_end_reason: str | None = None,
    ) -> SessionSummary:
        """Close the session record. Optionally keep the writer lock.

        `hold_lock=True` is the live builder's path. The final solve and
        the final build run AFTER this call and take up to a couple of
        minutes; while they run, the lock -- held by a process the Tower
        can see is alive -- is the only fact on disk that says "somebody
        is still finishing this world" rather than "this world was left
        half-built". The record is also given a `finalization` block in
        state `pending`, so a builder that dies inside that window leaves
        a lock naming a dead pid AND a pending finalization, which is a
        different, truthful story from a builder that was never asked to
        finalize. `release_world()` drops the lock when the caller is done;
        `mark_finalization()` moves the record on.

        Default `False` keeps every offline caller exactly as it was.
        """
        if self._session is None:
            raise SessionNotActiveError("stop_session() requires an active session")

        now = self._clock()
        finalization = None
        if hold_lock:
            finalization = {
                "state": FINALIZATION_PENDING,
                "final_solve": FINAL_SOLVE_PENDING,
                "started_at": now,
                "updated_at": now,
                "detail": None,
            }
        session = replace(
            self._session,
            ended_at=now,
            end_reason=reason,
            rejected_by_reason=dict(self._rejected),
            finalization=finalization,
        )
        self._store.write_session(session)
        # `capture_end_reason` IS NOT `reason`, AND THAT IS THE POINT.
        #
        # `reason` is what happened to the WORLD; the capture's own end is
        # what happened to the LINK. They differ in a case the field walk
        # actually produced: a capture that ended `disconnect` counts as
        # finished, so a walk whose phone never came back is recorded --
        # deliberately, see `world_build_session.py` -- as an ordinary
        # `stop`. That choice errs toward calling a real, openable world
        # Saved rather than putting the campaign's headline symptom back,
        # and it is defensible only while the artifact still says which it
        # was. `data/captures/<id>/capture.json` says, but the session
        # names only the FIRST capture it followed, and a reconnect starts
        # a new one; after that the link is a timestamp search.
        #
        # One key on an event that is already written exactly once. Not a
        # new periodic write on the live path -- that is the family §14.6
        # of the handoff declines to open in the last hour of a campaign,
        # and this is not it. Absent when the caller does not know, so
        # every offline caller's journal is byte-identical to before.
        stopped_payload = {"end_reason": reason}
        if capture_end_reason is not None:
            stopped_payload["capture_end_reason"] = capture_end_reason
        # An episode still open when the walk ends can no longer recover:
        # it is closed `timed_out` (why: session_stopped) BEFORE the stop
        # line, so a stopped session never reads "searching" forever.
        self._recovery(lambda r: r.close(self._clock()))
        self._reloc = None
        self._events.append("session_stopped", stopped_payload)
        if not hold_lock:
            self._store.release_writer_lock(session.world_id)

        summary = SessionSummary(
            session_id=session.session_id,
            frames_observed=session.frames_observed,
            keyframes_accepted=session.keyframes_accepted,
            rejected_by_reason=dict(self._rejected),
            # Distinct indices that received a keyframe, NOT the
            # high-water mark -- see _segments_used. This is the same
            # quantity build() reports, so the two cannot disagree.
            segments=len(self._segments_used),
            end_reason=reason,
        )
        self._session = None
        self._selector = None
        self._tracker = None
        self._events = None
        return summary

    @property
    def session_active(self) -> bool:
        """Whether a session is open: started and not yet stopped."""
        return self._session is not None

    def release_world(self, world_id: str) -> None:
        """Drop the writer lock a `stop_session(hold_lock=True)` kept."""
        self._store.release_writer_lock(world_id)

    def mark_finalization(
        self,
        world_id: str,
        session_id: str,
        *,
        state: str,
        final_solve: str | None,
        detail: str | None = None,
    ) -> None:
        """Rewrite the session's finalization block, and nothing else.

        The counts, the end reason and the timestamps written by
        `stop_session` are re-read from disk and kept; only the
        finalization moves. `started_at` is preserved from the pending
        record when there is one, so "how long did finalization take" stays
        answerable from the record alone.
        """
        if state not in FINALIZATION_STATES:
            raise ValueError(f"unknown finalization state {state!r}")
        session = self._store.read_session(world_id, session_id)
        now = self._clock()
        previous = session.finalization or {}
        self._store.write_session(
            replace(
                session,
                finalization={
                    "state": state,
                    "final_solve": final_solve,
                    "started_at": previous.get("started_at", now),
                    "updated_at": now,
                    "detail": detail,
                },
            )
        )

    def mark_stage(
        self,
        world_id: str,
        session_id: str,
        stage: str,
        *,
        state: str,
        detail: str | None = None,
        attempted: bool = True,
    ) -> None:
        """Record what became of one POST-finalization stage, and nothing else.

        The surface, the appearance and the dense stages run AFTER
        `mark_finalization(complete)` and AFTER `release_world()`, and they
        take six to eleven minutes on a real walk. That order is deliberate:
        the lock is dropped so the phone can read the world while they build
        (`results/world_builder_render.py`). What was missing is the record.
        A world could sit at `finalization: {state: complete}` with no
        photographic representation, no surface, and nothing on disk saying
        whether one was attempted, skipped, or raised -- the report dict the
        builder prints is not persisted anywhere.

        So this writes beside `finalization` rather than into it. Callers are
        expected to mark a stage `running` BEFORE starting it: this process
        can be killed by the Job Object on a thirty-second grace, and a stage
        left saying `running` by a pid that is gone is the honest record of
        exactly that.

        Like `mark_finalization`, the record is re-read from disk and only
        the one stage moves; `started_at` is preserved from the `running`
        write so "how long did the surface take" is answerable from the
        record alone. Unlike `mark_finalization`, this is called with the
        world lock already RELEASED, which is why it must stay a
        read-modify-write of a single small file published atomically.
        """
        if stage not in POST_FINALIZATION_STAGES:
            raise ValueError(
                f"unknown post-finalization stage {stage!r}; the set is "
                f"closed because consumers switch on it: "
                f"{POST_FINALIZATION_STAGES}"
            )
        if state not in STAGE_STATES:
            raise ValueError(
                f"unknown stage state {state!r}; the vocabulary is the one "
                f"every status.json already uses: {STAGE_STATES}"
            )
        session = self._store.read_session(world_id, session_id)
        now = self._clock()
        stages = dict(session.stages or {})
        previous = stages.get(stage) or {}
        stages[stage] = {
            "attempted": attempted,
            "state": state,
            "started_at": previous.get("started_at", now),
            "updated_at": now,
            "detail": detail,
        }
        self._store.write_session(replace(session, stages=stages))

    # -- build ---------------------------------------------------------

    def build(self, world_id: str, session_id: str) -> BuildResult:
        """Reconstruct from the persisted journal. Expensive; offline only."""
        world = self._store.read_world(world_id)
        if world.images_purged:
            raise ImagesPurgedError(
                f"world {world_id} has had its imagery purged; a rebuild "
                "would produce an empty reconstruction rather than a map"
            )

        session = self._store.read_session(world_id, session_id)
        keyframes = self._store.read_keyframes(world_id, session_id)
        _require_matching_resolution(session, keyframes)
        # Silent here, deliberately. `_open_live_solve` already announced
        # this at session start, where an operator can still act on it,
        # and build() now runs once per rebuild -- so announcing here
        # turns one actionable warning into one per rebuild. The
        # selection itself is unchanged and still recorded on the
        # session below.
        selection = select_backend(
            self._backend_name, session.intrinsics, announce=False
        )
        backend = selection.backend
        backend.prepare(session.intrinsics)

        if selection.was_downgraded:
            self._store.write_session(
                replace(
                    session,
                    backend_id=backend.capabilities.backend_id,
                    backend_requires_intrinsics=(
                        backend.capabilities.requires_intrinsics
                    ),
                    backend_downgraded_from=selection.downgraded_from,
                    backend_downgrade_reason=selection.downgrade_reason,
                )
            )
        else:
            self._store.write_session(
                replace(
                    session,
                    backend_id=backend.capabilities.backend_id,
                    backend_requires_intrinsics=(
                        backend.capabilities.requires_intrinsics
                    ),
                )
            )

        # Edges are recomputed from the keyframes on every build, so they
        # are derived output despite living in a journal. Without this,
        # each rebuild appends a duplicate set and the reported edge count
        # doubles, triples, and so on.
        self._store.clear_edges(world_id, session_id)

        # Segment -> (keyframe ids fed, estimate). Empty for a cold
        # rebuild, a different session, or a live solve that gave up.
        solved_live = self._live_estimates(world_id, session_id, session, backend)

        poses_solved = poses_refused = 0
        # `poses_refused` alone reads as N independent judgments about the
        # world. It is not: once any pose in a segment is refused the
        # chain latches and every later keyframe is refused WITHOUT ORB
        # detection, matching, or any geometry being attempted. Measured
        # on the real 33-segment world from capture 22e9d428: 354
        # refusals, 26 root decisions, 328 cascaded, and 0 of 26 segments
        # ever recovered. "26 chains died once" and "354 frames were each
        # judged ungeometric" are different bugs with different fixes.
        poses_refused_root = poses_refused_cascaded = 0
        refusal_degeneracy: dict[str, int] = {}
        refusals_by_segment: dict[int, dict] = {}
        # Counted, not derived by subtraction. `keyframes - poses_refused`
        # silently promotes every anchor to a camera position, and an
        # anchor is definitional rather than measured: identity rotation,
        # zero translation, by construction. On the 2026-08-24 physical
        # walk that arithmetic turned 36 origin markers into "36 camera
        # poses" on the phone while poses_solved was zero.
        #
        # An anchor IS a real position when the chain it anchors resolved
        # -- it is that segment's origin, and dropping it would
        # under-report every segment by one. So the rule is per segment,
        # and it needs the per-segment solve count to state.
        poses_anchor = 0
        poses_positioned = 0
        total_points = 0
        # Per segment, so a specific unreadable fragment can be chased
        # without re-running the walk. Summed for the manifest.
        discards_by_segment: dict[int, dict[str, int]] = {}
        total_triangulated = 0
        segments = sorted({keyframe.segment_index for keyframe in keyframes})

        pose_rows: list[dict] = []
        point_rows: list[list[float]] = []
        # [segment_index, frame_index, feature_index, point_index] per row.
        # Flat integers rather than dicts: this table is several times
        # longer than points.json's and carries no names worth repeating a
        # few hundred thousand times.
        support_rows: list[list[int]] = []

        for segment in segments:
            members = [k for k in keyframes if k.segment_index == segment]
            if not members:
                continue
            member_ids = tuple(keyframe.keyframe_id for keyframe in members)
            carried = solved_live.get(segment)
            if carried is not None and carried[0] == member_ids:
                # The flush. Nothing is re-read, re-decoded or re-solved;
                # this is the same estimate estimate_window() would
                # return, which tests/test_world_builder_incremental.py
                # pins bit-for-bit.
                estimate = carried[1]
            else:
                # Whatever this engine did not observe itself: a cold
                # rebuild, a re-derive, a session whose journal no longer
                # matches what was fed. Ids are compared rather than
                # counted because a matching count with different
                # keyframes is exactly the failure worth catching.
                window = [
                    KeyframeInput(
                        keyframe_id=keyframe.keyframe_id,
                        image_gray=self._load_gray(world_id, session_id, keyframe),
                    )
                    for keyframe in members
                ]
                estimate = backend.estimate_window(window)

            segment_solved = 0
            segment_anchors = 0
            first_refusal_index = None
            first_refusal_degeneracy = None
            solved_after_refusal = 0
            for position, (keyframe, pose) in enumerate(
                zip(members, estimate.poses)
            ):
                if pose.status == POSE_STATUS_SOLVED:
                    poses_solved += 1
                    segment_solved += 1
                elif pose.status == POSE_STATUS_ANCHOR:
                    poses_anchor += 1
                    segment_anchors += 1
                else:
                    poses_refused += 1
                    if first_refusal_index is None:
                        first_refusal_index = position
                        first_refusal_degeneracy = pose.degeneracy
                        poses_refused_root += 1
                        refusal_degeneracy[pose.degeneracy] = (
                            refusal_degeneracy.get(pose.degeneracy, 0) + 1
                        )
                    else:
                        poses_refused_cascaded += 1
                if (
                    pose.status == POSE_STATUS_SOLVED
                    and first_refusal_index is not None
                ):
                    # Cannot happen while the chain latches. Counted anyway,
                    # so that if a recovery mechanism is ever added its
                    # effect is visible instead of being invisible.
                    solved_after_refusal += 1
                pose_rows.append(
                    self._pose_row(keyframe, pose, segment)
                )
            # An anchor counts as a position only if something in its
            # segment actually solved against it. A lone anchor in a
            # segment that resolved nothing is an origin marker for an
            # empty coordinate frame.
            poses_positioned += segment_solved
            if segment_solved:
                poses_positioned += segment_anchors

            if first_refusal_index is not None:
                refusals_by_segment[segment] = {
                    "first_refusal_index": first_refusal_index,
                    "degeneracy": first_refusal_degeneracy,
                    # Keyframes the segment held but never attempted
                    # geometry for, because the chain had already latched.
                    "abandoned_keyframes": (
                        len(members) - first_refusal_index - 1
                    ),
                    "recovered": solved_after_refusal > 0,
                }

            for previous, current, pose in zip(
                members, members[1:], estimate.poses[1:]
            ):
                self._store.append_edge(
                    world_id,
                    session_id,
                    KeyframeEdge(
                        from_keyframe_id=previous.keyframe_id,
                        to_keyframe_id=current.keyframe_id,
                        matches=pose.matches,
                        inliers=pose.inliers,
                        inlier_ratio=pose.inlier_ratio,
                        median_parallax_px=pose.median_displacement_px,
                        median_parallax_deg=pose.median_triangulation_deg,
                        cheirality_fraction=pose.cheirality_fraction,
                        r_h=pose.r_h,
                        rotation_dominant=(pose.status != POSE_STATUS_SOLVED),
                        pose_status=pose.status,
                        degeneracy=pose.degeneracy,
                        quality=Confidence.from_score(pose.inlier_ratio),
                    ),
                )

            segment_discards = estimate.diagnostics.get("points_discarded") or {}
            discards_by_segment[segment] = {
                "low_parallax": int(segment_discards.get("low_parallax", 0)),
                "high_reprojection": int(
                    segment_discards.get("high_reprojection", 0)
                ),
            }
            total_triangulated += int(
                estimate.diagnostics.get("points_triangulated", 0)
            )

            if estimate.points is not None:
                # Tagged with the segment that produced them. Segments do
                # NOT share a coordinate frame or a unit, so concatenating
                # them untagged produces one cloud that silently merges
                # incompatible geometry.
                point_rows.extend(
                    {"segment_index": segment, "xyz": xyz}
                    for xyz in estimate.points.xyz.tolist()
                )
                total_points += len(estimate.points)

                # Which 2-D feature in which keyframe produced which of
                # those points. Tagged with the same segment, and BOTH
                # indices stay segment-local: `frame_index` is a position
                # within this segment's ordered keyframes, so it joins
                # against the poses this loop just appended, and
                # `point_index` is a position within this segment's
                # points, so it joins against the rows just above. Neither
                # is a session-wide ordinal -- a segment is the only frame
                # of reference the two share, since segments do not share
                # a coordinate frame either.
                support = estimate.points.support_views
                if support is not None:
                    support_rows.extend(
                        [segment, frame, feature, point]
                        for frame, feature, point in support.tolist()
                    )

        backend.release()

        # A global solution (tower/world_builder/global_solve.py), when one
        # has been persisted for this session, is expressed through the
        # derived tree here and nowhere else: build() stays the single
        # writer of poses.json / points.json / support.json, and the
        # placements the solution implies are written beside them under the
        # same input digest, so the geometry route serves them as current.
        # Counts below are recomputed from the merged rows; the chain's own
        # root/cascaded refusal split is kept as a diagnostic of the chain.
        input_digest = compute_input_digest(keyframes)
        solve_summary = None
        placements = None
        solution = global_solve.load_solution(self._store, world_id, session_id)
        if solution is not None:
            merged = global_solve.merge(
                keyframes, pose_rows, point_rows, support_rows, solution,
                input_digest=input_digest,
            )
            pose_rows = merged.pose_rows
            point_rows = merged.point_rows
            support_rows = merged.support_rows
            placements = merged.placements
            solve_summary = {**merged.summary, "segments": merged.segments}
            poses_solved = sum(1 for r in pose_rows if r["status"] == POSE_STATUS_SOLVED)
            poses_anchor = sum(1 for r in pose_rows if r["status"] == POSE_STATUS_ANCHOR)
            poses_refused = len(pose_rows) - poses_solved - poses_anchor
            solved_by_segment: dict[int, int] = {}
            anchors_by_segment: dict[int, int] = {}
            for r in pose_rows:
                if r["status"] == POSE_STATUS_SOLVED:
                    solved_by_segment[r["segment_index"]] = solved_by_segment.get(r["segment_index"], 0) + 1
                elif r["status"] == POSE_STATUS_ANCHOR:
                    anchors_by_segment[r["segment_index"]] = anchors_by_segment.get(r["segment_index"], 0) + 1
            poses_positioned = sum(
                n + anchors_by_segment.get(seg, 0) for seg, n in solved_by_segment.items()
            )
            total_points = len(point_rows)

        # Scale becomes "relative" only once something actually solved:
        # an internally consistent world with an arbitrary unit. Without a
        # solved pose there is no unit at all, so it stays "unknown".
        #
        # A world with MORE THAN ONE segment is not internally consistent
        # either. Each segment is solved in its own window, so each has its
        # own arbitrary unit -- measured 4x apart between two segments of
        # one session. Calling that "relative" would assert a coherence
        # the reconstruction does not have.
        if not poses_solved:
            scale_state = SCALE_UNKNOWN
        elif len(segments) > 1:
            scale_state = SCALE_UNKNOWN
        else:
            scale_state = SCALE_RELATIVE

        # Never clobber a scale this build did not earn. A measured scale
        # carries meters_per_unit, method, confidence and history that a
        # rebuild has no business discarding -- ScaleState promises
        # superseded estimates are appended, never overwritten.
        existing = world.scale
        if existing.state == SCALE_MEASURED:
            scale = existing
        elif existing.state == scale_state:
            scale = existing
        else:
            scale = ScaleState(state=scale_state)

        self._store.write_world(
            replace(world, updated_at=self._clock(), scale=scale)
        )
        self._store.write_derived(
            world_id,
            session_id,
            poses=pose_rows,
            points=point_rows,
            support=support_rows,
            manifest={
                "schema_version": world.schema_version,
                "input_digest": input_digest,
                "built_at": self._clock(),
                "backend_id": backend.capabilities.backend_id,
                "session_id": session_id,
                "keyframes": len(keyframes),
                "poses_solved": poses_solved,
                "poses_refused": poses_refused,
                # Split, because the total conflates a decision with its
                # consequences. root + cascaded == poses_refused, always.
                "poses_refused_root": poses_refused_root,
                "poses_refused_cascaded": poses_refused_cascaded,
                # Over ROOT refusals only -- the cascaded ones carry the
                # root's label and would trebly count one decision.
                "refusal_degeneracy_counts": refusal_degeneracy,
                # Both reported. Suppressing the anchors would replace one
                # misleading number with a missing one; a reader should be
                # able to see "36 segment origins and no trajectory",
                # which is a precise description of an uncalibrated walk.
                "poses_anchor": poses_anchor,
                "poses_positioned": poses_positioned,
                "points": total_points,
                # What was triangulated but refused, and why. Stated
                # rather than left to be inferred: a consumer seeing only
                # the surviving points cannot otherwise tell a sparse
                # world from a heavily filtered one, and those call for
                # different responses from whoever is wearing the glasses.
                # Zero is written explicitly -- absent would mean "this
                # build predates the counter", which is a different fact.
                "points_discarded": {
                    "low_parallax": sum(
                        d["low_parallax"] for d in discards_by_segment.values()
                    ),
                    "high_reprojection": sum(
                        d["high_reprojection"]
                        for d in discards_by_segment.values()
                    ),
                },
                "points_triangulated": total_triangulated,
                "segments": len(segments),
                "scale_state": scale_state,
                "global_solve": solve_summary,
            },
        )
        if placements is not None:
            self._store.write_placements(world_id, session_id, placements)

        return BuildResult(
            world_id=world_id,
            session_id=session_id,
            backend_id=backend.capabilities.backend_id,
            keyframes=len(keyframes),
            poses_solved=poses_solved,
            poses_refused=poses_refused,
            points=total_points,
            segments=len(segments),
            scale_state=scale_state,
            downgraded_from=selection.downgraded_from,
            diagnostics={
                "points_discarded_by_segment": discards_by_segment,
                "refusals_by_segment": refusals_by_segment,
                # WHO OWNS placements.json after this build.
                #
                # "global_solve" means the merge above wrote every placement
                # from one reconstruction. The Sim3 registrar must then not
                # run: it answers the same question pairwise and weaker, and
                # it OVERWRITES the same file.
                #
                # This is reported rather than inferred because the builder
                # used to infer it from the wrong thing -- whether the FINAL
                # solve had succeeded. On the 2026-09-09 walk the final solve
                # never ran (the session died in the observe loop), so the
                # builder concluded no solution existed and ran the
                # registrar 16 seconds after the last build. It replaced 72
                # segments registered into 14 components with 120 refusals
                # and 2 registrations, and that is the world the phone then
                # drew. Nine good background solves were discarded by a
                # question about a tenth that never happened.
                #
                # "PLACED SOMETHING", not "ran". An adversarial review found
                # the first version of this asking `placements is not None`,
                # which is true whenever `merge()` ran at all -- including
                # when it returns [] because every segment is still pending,
                # and when it returns nothing but refusals because the solve
                # posed none of their keyframes. Both of those place zero
                # segments, and both would have stood the registrar down and
                # left the world with no placements at all. The registrar is
                # the correct fallback there, and this must not suppress it.
                "placements_source": (
                    "global_solve"
                    if placements and any(p.state == "registered" for p in placements)
                    else None
                ),
            },
        )

    # -- internals -----------------------------------------------------

    def _start_relocalizer(self, session) -> None:
        """The session's look-back relocalizer, when the setting asks for one.

        `relocalizer_started` is journaled right after `session_started`
        and records what THIS builder runs with (contract s6.2: the payload
        reads the limiter and acceptance from here, never from the web
        process's configuration). No relocalizer -- `off`, no calibration,
        or a failure to build one -- journals nothing, and the session's
        `tracking.recovery` is null exactly as for every older session.
        """
        if self._reloc is not None:
            try:
                self._reloc.close(self._clock())
            except Exception:
                logger.exception("[Tower][WorldBuilder] closing a stale relocalizer failed")
            self._reloc = None
        mode = self._relocalizer_mode
        if mode is None:
            from tower.config import world_relocalizer_setting

            mode = world_relocalizer_setting()
        if mode not in ("prompt", "silent"):
            return
        try:
            from tower.world_builder import relocalizer

            reloc = relocalizer.from_session(session.intrinsics, mode=mode)
        except Exception:
            logger.exception(
                "[Tower][WorldBuilder] look-back relocalizer unavailable for "
                "session %s; the walk continues without it",
                session.session_id,
            )
            return
        if reloc is None:
            return
        self._events.append("relocalizer_started", reloc.started_payload())
        self._reloc = reloc

    def _recovery(self, step) -> None:
        """Run one relocalizer step and journal what it decided, in order.

        The relocalizer never writes the journal itself; its worker never
        touches it at all. A relocalizer that raises is dropped for the rest
        of the session -- it must never cost the walk a frame or a keyframe.
        """
        reloc = self._reloc
        if reloc is None:
            return
        try:
            events = step(reloc)
        except Exception:
            logger.exception(
                "[Tower][WorldBuilder] look-back relocalizer failed; it is off "
                "for the rest of this session"
            )
            self._reloc = None
            try:
                reloc.close(self._clock())
            except Exception:
                pass
            return
        for kind, payload in events or ():
            self._events.append(kind, payload)

    def _open_live_solve(self, session) -> None:
        """Start the solve that observe() will extend.

        Backend selection is deterministic in (name, intrinsics), so the
        instance chosen here is the same one build() would choose; build()
        re-checks both anyway before trusting anything this produces.
        """
        if self._live is not None:
            self._live.release()
            self._live = None
        try:
            selection = select_backend(self._backend_name, session.intrinsics)
            backend = selection.backend
            backend.begin(session.intrinsics)
        except Exception:
            # A live solve is an optimisation. Losing it must never cost
            # the session its keyframes -- build() still has the
            # from-scratch path, and it will raise there, loudly, with
            # the whole journal in hand.
            logger.exception(
                "[Tower][WorldBuilder] live geometry unavailable for session "
                "%s; build() will solve from scratch",
                session.session_id,
            )
            return
        self._live = _LiveSolve(
            world_id=session.world_id,
            session_id=session.session_id,
            backend=backend,
            intrinsics=session.intrinsics,
            segment_index=self._segment_index,
        )

    def _live_estimates(self, world_id, session_id, session, backend) -> dict:
        """What the live solve has, if it is still the right answer.

        Every one of these is a way the carried solve could be answering
        a question nobody asked: a different world, a different session,
        intrinsics rewritten since the session opened, or a backend
        selection that has since changed. Any of them and the whole thing
        is discarded rather than partially believed.
        """
        live = self._live
        if live is None or not live.usable:
            return {}
        if live.world_id != world_id or live.session_id != session_id:
            return {}
        if live.intrinsics != session.intrinsics:
            return {}
        if live.backend_id != backend.capabilities.backend_id:
            return {}
        return live.estimates()

    def _pose_row(self, keyframe, pose, segment) -> dict:
        """Convert a backend pose into the persisted T_world_camera contract.

        The backends work in the convention OpenCV hands them:
        `recoverPose` and `solvePnPRansac` both return (R, t) mapping a
        WORLD point into the CAMERA frame, so `t` is not a position at all
        -- it is where the world origin sits as seen by the camera.

        schema.POSE_CONVENTION declares the opposite: `T_world_camera`,
        whose translation IS the camera's position in world coordinates,
        chosen precisely so no consumer ever has to invert anything.

        Writing the raw `t` under that contract mirrors every camera
        through the origin. It is not a loud failure -- the trajectory
        stays smooth, monotonic and entirely plausible -- which is exactly
        why the convention is frozen and why this conversion lives here
        rather than being left to whoever renders it. An earlier version
        of this function shipped the raw value; a strafe along +X
        persisted as -X, and nothing caught it until the values were
        compared against ground-truth camera positions.

            R_world_camera = R.T
            C              = -R.T @ t
        """
        import numpy as np

        row = {
            "keyframe_id": keyframe.keyframe_id,
            "segment_index": segment,
            "status": pose.status,
            "degeneracy": pose.degeneracy,
            "rotation": None,
            "translation": None,
        }
        if pose.status == POSE_STATUS_ANCHOR:
            row["rotation"] = [1.0, 0.0, 0.0, 0.0]
            row["translation"] = [0.0, 0.0, 0.0]
        elif pose.rotation is not None:
            rotation = np.asarray(pose.rotation, dtype=np.float64)
            row["rotation"] = _rotation_to_quaternion_wxyz(rotation.T)
            if pose.translation is not None:
                translation = np.asarray(
                    pose.translation, dtype=np.float64
                ).reshape(3)
                row["translation"] = [
                    float(v) for v in (-rotation.T @ translation)
                ]
            # A rotation_only pose keeps its rotation and leaves
            # translation null. Discarding both would throw away the real
            # information the degeneracy path exists to preserve.
        return row

    def _load_gray(self, world_id, session_id, keyframe):
        path = self._store.session_dir(world_id, session_id) / keyframe.image_relpath
        return decode_gray(path.read_bytes())

    def _persist_keyframe(
        self, *, gray_shape, raw_bytes, received_at, source_seq, wire_seq,
        tx_seq, quality, motion, reason,
    ) -> tuple[Keyframe, bytes]:
        session = self._session
        filename = f"{source_seq:08d}.jpg"

        # The privacy transformation, and it happens HERE because this is
        # the one place every persisted pixel passes through. Before the
        # write, never after: redacting on read would leave the raw frames
        # on disk, which is a display filter rather than the
        # transformation 06-PRIVACY-DATA asks for.
        #
        # The bytes the reconstruction later reads are the redacted ones,
        # and that was measured before it was chosen: at the ~5% of frame a
        # real face occupies, keyframe acceptance and pose solving are
        # completely unaffected and the point cloud loses about 9%. See
        # redaction.py.
        redaction = self._redactor.redact(raw_bytes)
        image_bytes = redaction.image_bytes

        # Image first, fsynced, THEN the journal line. A journal line
        # pointing at a missing image is corruption; an orphan image is
        # harmless and gets swept.
        self._store.write_keyframe_image(
            session.world_id, session.session_id, filename, image_bytes
        )
        keyframe = Keyframe(
            keyframe_id=make_keyframe_id(session.session_id, source_seq),
            session_id=session.session_id,
            source_seq=source_seq,
            wire_seq=wire_seq,
            tx_seq=tx_seq,
            received_at=received_at,
            image_relpath=f"images/{filename}",
            width=quality.width,
            height=quality.height,
            byte_count=len(image_bytes),
            segment_index=self._segment_index,
            sharpness=quality.sharpness,
            selection_reason=reason,
            median_parallax_px=(
                motion.median_displacement_px if motion else None
            ),
            overlap_ratio=motion.overlap_ratio if motion else None,
            survival_ratio=motion.survival_ratio if motion else None,
            tracked_count=motion.tracked_count if motion else None,
            feature_count=motion.seeded_count if motion else None,
            homography_residual_px=(
                motion.homography_residual_px if motion else None
            ),
            quality=Confidence.from_score(
                motion.survival_ratio if motion else None
            ),
        )
        self._store.append_keyframe(session.world_id, keyframe)
        # The session records what was APPLIED, not what was configured, and
        # it records the WEAKEST thing applied to any of its keyframes.
        #
        # This used to be "whatever the last keyframe got". `redact` never
        # raises: when detection throws on one frame it hands back the
        # ORIGINAL bytes labelled `none`, and those are what was just written.
        # The next frame that redacted cleanly then rewrote the session label,
        # and the dense and surface stages -- which trust `images/` as-is for
        # any session not labelled `none` -- read that raw frame as a redacted
        # keyframe and fused it. One later success laundered one failure.
        #
        # So `none` is sticky. It is a session-wide demotion rather than a
        # per-keyframe record, and that is deliberate: the dense stage already
        # handles a `none` session by redacting every keyframe itself before
        # reading a pixel, and refusing any frame it cannot redact. The cost of
        # one bad frame is therefore recomputation, not lost frames, and it
        # needs no schema change that every reader of `keyframes.jsonl` would
        # have to learn to honour before it was safe.
        if redaction.label == REDACTION_NONE:
            self._unredacted_keyframes += 1
            if self._unredacted_keyframes == 1 and self._redactor.available:
                logger.warning(
                    "[Tower][WorldBuilder] keyframe %s persisted UNREDACTED (%s); "
                    "session %s is recorded as redaction=none from here on",
                    keyframe.keyframe_id, redaction.unavailable_reason,
                    session.session_id,
                )
        label = (
            REDACTION_NONE if self._unredacted_keyframes else redaction.label
        )
        if self._session.redaction != label:
            self._session = replace(self._session, redaction=label)
        # The bytes as well as the record: they are what the live solve
        # must see, because they are what a rebuild will read back.
        return keyframe, image_bytes

    def _note_rejected(self, reason: str) -> None:
        self._rejected[reason] = self._rejected.get(reason, 0) + 1

    def _result(self, outcome, reason, keyframe_id=None) -> ObserveResult:
        return ObserveResult(
            outcome=outcome,
            reason=reason,
            keyframe_id=keyframe_id,
            frames_observed=self._session.frames_observed,
            keyframes_accepted=self._session.keyframes_accepted,
        )


class _LiveSolve:
    """One geometry solve carried across observe() calls.

    The whole reason this class exists is a cost measurement. build()
    re-solved from scratch every time, at roughly O(N^1.2) in the backend
    alone -- 303 ms for 32 keyframes and 641 ms for 64, extrapolating to
    about 2 s at the 155 keyframes of the 2026-08-24 physical walk, plus
    a JPEG decode per keyframe on top. A walk rebuilt every k keyframes
    therefore paid O(N^2/k): 5.9 s of backend work over 64 keyframes at
    --rebuild-every 4, against 0.8 s for the same walk extended. Turning
    the live updates UP made the whole session slower, which is why the
    cadence defaulted to zero and why nothing appeared until a walk had
    ended.

    A segment gets exactly one solve, and it never crosses a
    tracking_lost: segments do not share a coordinate frame or a unit,
    they are independent windows today, and they must stay so. Closing a
    segment freezes its estimate and resets the backend.

    Nothing here is allowed to cost the session a keyframe. Every backend
    call is guarded, and a solve that fails simply stops offering
    answers; build() then does what it always did.
    """

    def __init__(self, *, world_id, session_id, backend, intrinsics, segment_index):
        self.world_id = world_id
        self.session_id = session_id
        self.backend = backend
        self.intrinsics = intrinsics
        self.backend_id = backend.capabilities.backend_id
        self.usable = True
        self._segment_index = segment_index
        self._open: list[str] = []
        self._frozen: dict[int, tuple[tuple[str, ...], object]] = {}

    def extend(self, keyframe_id: str, gray):
        """Extend the open solve and return the backend's Extension.

        The return value was previously discarded. It carries
        `chain_broken`, the signal that lets the engine split a segment
        when the solve fails instead of refusing every later keyframe in
        it without looking.
        """
        if not self.usable:
            return None
        try:
            step = self.backend.extend(
                KeyframeInput(keyframe_id=keyframe_id, image_gray=gray)
            )
        except Exception:
            self._give_up("extending")
            return None
        self._open.append(keyframe_id)
        return step

    def close_segment(self, segment_index: int) -> None:
        if not self.usable:
            return
        try:
            if self._open:
                self._frozen[self._segment_index] = (
                    tuple(self._open),
                    self.backend.snapshot(),
                )
            self.backend.reset()
        except Exception:
            self._give_up("closing a segment of")
            return
        self._open = []
        self._segment_index = segment_index

    def estimates(self) -> dict:
        """Frozen segments plus a live view of the open one.

        Non-destructive: a mid-walk rebuild reads this and the walk keeps
        extending the same solve afterwards. If it were destructive,
        watching a world build would change the world.
        """
        if not self.usable:
            return {}
        carried = dict(self._frozen)
        if self._open:
            try:
                carried[self._segment_index] = (
                    tuple(self._open),
                    self.backend.snapshot(),
                )
            except Exception:
                self._give_up("snapshotting")
                return {}
        return carried

    def release(self) -> None:
        self.usable = False
        try:
            self.backend.release()
        except Exception:
            logger.exception("[Tower][WorldBuilder] backend release failed")

    def _give_up(self, doing: str) -> None:
        self.usable = False
        self._frozen = {}
        self._open = []
        logger.exception(
            "[Tower][WorldBuilder] live geometry gave up %s session %s; "
            "build() will solve from scratch",
            doing,
            self.session_id,
        )


class IntrinsicsResolutionMismatchError(RuntimeError):
    """Intrinsics were calibrated at a resolution the frames are not."""


def _require_matching_resolution(session, keyframes) -> None:
    """Refuse to apply intrinsics to frames of a different size.

    Applying a 480x360 calibration to 720x1280 frames does not fail --
    it silently produces a reconstruction wrong by the resolution ratio.
    CameraIntrinsics.scaled_to() already refuses to rescale without
    established linearity; this is the check that actually routes callers
    into that refusal instead of around it.
    """
    intrinsics = session.intrinsics
    if not intrinsics.is_known or not keyframes:
        return
    sizes = {(keyframe.width, keyframe.height) for keyframe in keyframes}
    calibrated = (intrinsics.calibrated_width, intrinsics.calibrated_height)
    mismatched = sorted(size for size in sizes if size != calibrated)
    if mismatched:
        raise IntrinsicsResolutionMismatchError(
            f"intrinsics were calibrated at {calibrated[0]}x{calibrated[1]} "
            f"but this session contains keyframes at {mismatched}. Refusing "
            "to apply them: the reconstruction would be silently wrong by "
            "the resolution ratio. Calibrate at the delivered resolution, "
            "or establish scales_linearly_across_resolutions and rescale."
        )


def _rotation_to_quaternion_wxyz(rotation) -> list[float]:
    """Rotation matrix to quaternion in the frozen wxyz order.

    Hand-rolled because scipy is not a dependency. Uses the largest
    diagonal term to pick a numerically stable branch rather than the
    trace-only formula, which loses precision near 180 degrees.
    """
    import numpy as np

    m = np.asarray(rotation, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return [float(w), float(x), float(y), float(z)]
