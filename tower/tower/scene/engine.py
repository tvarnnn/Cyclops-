"""Frames in, a live scene state out.

    every frame     detect, associate into tracks, place     ~17 ms CUDA (RT-DETRv2-R18)
                                                             ~30 ms CPU  (SSDLite320)
    at a cadence    face visibility on person tracks          ~9 ms per face, CPU

Nothing is persisted and nothing runs on the Tower event loop: this
engine is driven by `tower/scene/live.py` on a worker thread inside the
web process, one frame at a time, newest frame wins.

The detector is chosen by device (`tower/scene/detect.py`) and the
orientation stage is a face detector on each tracked person's box
(`tower/scene/orientation.py`); both decisions and their measurements are
in `docs/superpowers/research/2026-09-07-scene-understanding-architecture.md`.
"""

import logging
import time

from tower.scene.detect import SCORE_THRESHOLD, Detector
from tower.scene.orientation import VOTE_WINDOW, age_estimate, vote
from tower.scene.records import FacingEstimate
from tower.scene.state import (
    SceneState,
    apparent_size,
    assign_side,
    describe_position,
    is_partial_at_bottom_edge,
    relate,
)
from tower.scene.tracking import (
    DELIVERED_FRAME_INTERVAL_S,
    Tracker,
    TrackerPolicy,
)

logger = logging.getLogger(__name__)

# `DELIVERED_FRAME_INTERVAL_S` is 83.5 ms -- 12.0 fps, measured from the
# corpus's own `frames.jsonl` receipt timestamps. It is imported rather
# than defined here because `TrackerPolicy.max_misses` is derived from it
# too, and it has to live below both consumers. It stays importable from
# here, which is where a driver looks for a frame-rate fact.

# How many delivered frames one orientation estimate may skip: the
# tracker's confirmation streak, currently three. Estimating facing more
# often than a track can be confirmed buys nothing; less often, and a
# track can appear, be reported and be dropped without its facing ever
# being measured once.
#
# Written as the coupling rather than as 3, because the coupling is the
# reason. A test used to be the only thing holding these two equal; a
# retune of `min_hits` would have passed review and broken it.
ORIENTATION_FRAME_STRIDE = TrackerPolicy.min_hits

# How often the orientation stage may run, in seconds: ~250 ms.
#
# The stage is cheap now -- ~9 ms per face on CPU, measured 2026-09-07,
# against the 43 ms CUDA / 956 ms CPU keypoint model it replaced -- so
# the cadence is no longer a cost decision. It is a VOTING decision:
# `orientation.vote` needs two agreeing estimates out of three before it
# will say "facing", so at this stride a claim needs about half a second
# of evidence and stops about half a second after the face is gone. Per-
# frame estimates would make that window 250 ms, which is the flicker
# the vote exists to remove. A person's facing does not change in 83 ms.
ORIENTATION_INTERVAL_S = ORIENTATION_FRAME_STRIDE * DELIVERED_FRAME_INTERVAL_S


class SceneEngine:
    """Maintains what is around the wearer right now.

    Deliberately has no store. `state()` returns the current answer and
    the previous one is gone -- which is the whole difference between this
    cartridge and Environmental Memory.
    """

    def __init__(
        self,
        detector: Detector,
        tracker_policy: TrackerPolicy | None = None,
        clock=time.time,
        *,
        facing_estimator=None,
        orientation_interval_s: float = ORIENTATION_INTERVAL_S,
        score_threshold: float | None = None,
    ) -> None:
        self._detector = detector
        self._tracker = Tracker(tracker_policy)
        self._clock = clock
        # None means no orientation stage at all: `orientation_enabled`
        # stays False and every facing is `unknown`. The runtime supplies
        # one when the face model file is present.
        self._facing = facing_estimator
        self._orientation_interval = orientation_interval_s
        # The detector's own floor when it has one, so the published
        # `score_threshold` is the one the counts were actually taken at.
        if score_threshold is None:
            score_threshold = getattr(detector, "score_threshold", SCORE_THRESHOLD)
        self._score_threshold = score_threshold
        self._last_orientation_at: float | None = None
        # Whether orientation has ever produced a real estimate, as
        # opposed to merely being configured. An estimator that raises on
        # every call -- a missing model, a bad build -- used to leave
        # `orientation_enabled` True and the query layer answering a
        # confident 0, which is the same observation-gap trap the refusal
        # exists to prevent, one layer further down.
        self._orientation_succeeded = False
        self._frames_observed = 0
        self._frame_size = (0, 0)
        self._loaded = False

    @property
    def frames_observed(self) -> int:
        return self._frames_observed

    @property
    def orientation_enabled(self) -> bool:
        """Configured AND has actually produced an estimate.

        "An object was passed to the constructor" is not evidence that
        anyone's orientation was ever measured.
        """
        return self._facing is not None and self._orientation_succeeded

    @property
    def orientation_configured(self) -> bool:
        return self._facing is not None

    @property
    def orientation_method(self) -> str | None:
        return None if self._facing is None else getattr(self._facing, "name", None)

    @property
    def detector_name(self) -> str:
        return getattr(self._detector, "name", "unknown")

    def load(self) -> None:
        self._detector.load()
        if self._facing is not None:
            self._facing.load()
        self._loaded = True

    def release(self) -> None:
        self._detector.release()
        if self._facing is not None:
            self._facing.release()
        self._tracker.reset()

    def observe(self, frame_bgr, *, received_at: float | None = None) -> SceneState:
        """One frame in, the current scene out.

        Returns the state rather than storing it. A caller that wants the
        previous state has to have kept it, which is the correct default
        for something whose whole claim is to describe *now*.
        """
        at = self._clock() if received_at is None else received_at
        self._frames_observed += 1
        if frame_bgr is not None and getattr(frame_bgr, "size", 0):
            self._frame_size = (frame_bgr.shape[1], frame_bgr.shape[0])
        width, height = self._frame_size

        detections = self._detect(frame_bgr)
        self._tracker.update(detections, at=at)
        self._place(width)

        if self._should_estimate_orientation(at):
            self._estimate_orientation(frame_bgr, at)
        self._age_orientation(at)

        # Copied, not referenced. `Tracker.counted()` hands back its own
        # mutable `Track` objects, and the tracker rewrites them on every
        # subsequent frame -- so a state that kept them would keep
        # changing after the moment it claims to describe. See
        # tests/test_scene_snapshot_isolation.py.
        counted = [track.snapshot() for track in self._tracker.counted()]
        partial = tuple(
            track for track in counted if is_partial_at_bottom_edge(track, height)
        )
        tracks = tuple(track for track in counted if track not in partial)
        counts: dict = {}
        for track in tracks:
            counts[track.label] = counts.get(track.label, 0) + 1

        return SceneState(
            at=at,
            frame_width=width,
            frame_height=height,
            tracks=tracks,
            partial_people=partial,
            relations=tuple(relate(tracks, width, height)),
            counts=counts,
            frames_observed=self._frames_observed,
            detector=self.detector_name,
            score_threshold=self._score_threshold,
            orientation_enabled=self.orientation_enabled,
            orientation_method=self.orientation_method,
        )

    def describe(self, state: SceneState) -> dict:
        """The scene as an answer, not as a data structure."""
        payload = state.to_json_dict()
        payload["positions"] = {
            track.track_id: describe_position(
                track, state.frame_width, state.frame_height
            )
            for track in state.tracks
        }
        payload["apparent_sizes"] = {
            track.track_id: apparent_size(track, state.frame_height)
            for track in state.tracks
            if track.label == "person"
        }
        return payload

    # -- internals -----------------------------------------------------

    def _detect(self, frame_bgr):
        """Detect, or report an empty frame rather than dying.

        A detector can fail for ordinary reasons -- a model OOM, a bad
        frame. Losing the whole session over one of them would be the same
        defect Document Memory's review found in its OCR path, and the
        answer is the same: an empty result is honest, a crash is not.
        """
        try:
            return self._detector.detect(frame_bgr)
        except Exception:
            logger.exception(
                "scene: detection failed on a frame; treating it as empty "
                "and continuing"
            )
            return []

    def _place(self, width: int) -> None:
        """Give every live track a side, with hysteresis against its last."""
        for track in self._tracker.tracks:
            normalised_x = track.box.centre[0] / width if width else None
            track.side = assign_side(track.side, normalised_x)

    def _should_estimate_orientation(self, at: float) -> bool:
        if self._facing is None:
            return False
        if self._last_orientation_at is None:
            return True
        return at - self._last_orientation_at >= self._orientation_interval

    def _estimate_orientation(self, frame_bgr, at: float) -> None:
        """Ask the estimator about each tracked person's box, then vote.

        The boxes are the TRACKER'S: the detector decides what exists and
        the estimator only describes it, so two models can never
        disagree about how many people there are. An estimator that
        raises leaves every facing where it was, ageing.
        """
        person_tracks = [
            track for track in self._tracker.tracks if track.label == "person"
        ]
        if not person_tracks:
            return
        try:
            estimates = self._facing.estimate(
                frame_bgr, [track.box for track in person_tracks]
            )
        except Exception:
            logger.exception("scene: facing estimation failed; leaving facing unknown")
            return
        self._last_orientation_at = at
        if len(estimates) != len(person_tracks):
            logger.error(
                "scene: facing estimator returned %d estimates for %d boxes; "
                "ignoring the run",
                len(estimates),
                len(person_tracks),
            )
            return
        for track, raw in zip(person_tracks, estimates):
            history = track.facing_history
            track.facing = age_estimate(vote(history, raw), 0.0)
            track.facing_history = (*history, raw.state)[-VOTE_WINDOW:]
            track.facing_estimated_at = at
            self._orientation_succeeded = True

    def _age_orientation(self, at: float) -> None:
        """Age each estimate from when THAT TRACK was last estimated.

        Not from the last orientation RUN. The two diverge precisely when
        staleness matters most: the estimator runs, is not asked about
        this person -- because the tracker lost them for a frame -- and
        the track keeps its old reading. Ageing from the run time would
        reset that reading's age to nearly zero on every run, so a
        ten-second-old "facing toward you" would report as one second old
        and would never reach the expiry that exists to catch it.

        `age_estimate` also carries a correctness guard unrelated to
        cost: its clamp exists because a backward NTP step produced a
        negative age, which pushed the expiry deadline further into the
        future -- the one direction it must never move.
        """
        if self._facing is None:
            return
        for track in self._tracker.tracks:
            if track.label != "person":
                continue
            if track.facing_estimated_at is None:
                # Never estimated -- a track that appeared since the last
                # run. Unknown, which is the honest answer, and not the
                # previous occupant's reading.
                track.facing = FacingEstimate()
                continue
            track.facing = age_estimate(
                track.facing, at - track.facing_estimated_at
            )
