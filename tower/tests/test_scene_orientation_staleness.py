"""A facing estimate runs at a cadence, is voted on, and carries its age.

Orientation runs at a cadence rather than per frame, so the estimates it
produces are stale by construction, and every one carries its age. The
cadence is no longer a cost decision -- the face-visibility estimator is
~9 ms per face on CPU -- it is what makes the 2-of-3 vote mean "half a
second of agreeing evidence". A consumer that cannot see how stale an
estimate is is being told something false with a true-looking shape.

These tests use `FixedFacingEstimator`, so the estimate for a box is
chosen here rather than read back from a model.
"""

import numpy as np
import pytest

from tower.confidence import Confidence
from tower.scene.detect import FixedDetector
from tower.scene.engine import (
    DELIVERED_FRAME_INTERVAL_S,
    ORIENTATION_FRAME_STRIDE,
    ORIENTATION_INTERVAL_S,
    SceneEngine,
)
from tower.scene.orientation import (
    EVIDENCE_FACE,
    EVIDENCE_NO_FACE,
    EVIDENCE_TOO_SMALL,
    EVIDENCE_WEAK_FACE,
    FACE_SCORE_THRESHOLD,
    MAX_ESTIMATE_AGE_S,
    VOTE_NEEDED,
    VOTE_WINDOW,
    FixedFacingEstimator,
    age_estimate,
    estimate_from_face,
    vote,
)
from tower.scene.records import (
    FACING_TOWARD,
    FACING_UNKNOWN,
    BoundingBox,
    Detection,
    FacingEstimate,
)
from tower.scene.tracking import TrackerPolicy

FRAME = np.zeros((360, 640, 3), np.uint8)
POLICY = TrackerPolicy(min_iou=0.25, min_hits=2, max_misses=5)
PERSON_BOX = (100, 80, 220, 320)
TOWARD = estimate_from_face(0.95)
NO_FACE = estimate_from_face(None)


def _person():
    return [Detection(label="person", score=0.9, box=BoundingBox(*PERSON_BOX))]


def _sees(estimate=TOWARD, box=PERSON_BOX):
    return [(BoundingBox(*box), estimate)]


def _engine(facing, interval=2.0):
    engine = SceneEngine(
        FixedDetector([_person()] * 200),
        POLICY,
        clock=lambda: 0.0,
        facing_estimator=facing,
        orientation_interval_s=interval,
    )
    engine.load()
    return engine


class TestOneEstimateFromOneFace:
    def test_a_strong_face_is_toward_at_medium_never_high(self):
        estimate = estimate_from_face(0.95)
        assert estimate.state == FACING_TOWARD
        assert estimate.confidence is Confidence.MEDIUM
        assert estimate.evidence == EVIDENCE_FACE

    def test_a_weak_face_is_unknown_and_says_why(self):
        estimate = estimate_from_face(FACE_SCORE_THRESHOLD - 0.01)
        assert estimate.state == FACING_UNKNOWN
        assert estimate.evidence == EVIDENCE_WEAK_FACE

    def test_no_face_is_unknown_not_away(self):
        """The camera did not establish the orientation. It did not
        establish "away" either, and the state must not say so."""
        estimate = estimate_from_face(None)
        assert estimate.state == FACING_UNKNOWN
        assert estimate.evidence == EVIDENCE_NO_FACE

    def test_a_box_too_small_for_a_face_is_unknown(self):
        estimate = estimate_from_face(0.99, too_small=True)
        assert estimate.state == FACING_UNKNOWN
        assert estimate.evidence == EVIDENCE_TOO_SMALL

    def test_the_threshold_is_the_measured_one(self):
        """0.9: 0.83 precision on 966 COCO persons; 0.8 gave 0.69."""
        assert FACE_SCORE_THRESHOLD == pytest.approx(0.9)


class TestTheVote:
    def test_one_flash_of_a_face_is_not_a_claim(self):
        assert vote((), TOWARD).state == FACING_UNKNOWN

    def test_two_of_three_agreeing_is_a_claim(self):
        assert vote((FACING_TOWARD,), TOWARD).state == FACING_TOWARD
        assert vote((FACING_TOWARD, FACING_UNKNOWN), TOWARD).state == FACING_TOWARD

    def test_a_person_who_turned_away_stops_being_facing_within_two_estimates(self):
        history = (FACING_TOWARD, FACING_TOWARD, FACING_TOWARD)
        first = vote(history, NO_FACE)
        assert first.state == FACING_TOWARD, "one missing face is a gap"
        second = vote((*history, NO_FACE)[-VOTE_WINDOW:], NO_FACE)
        assert second.state == FACING_UNKNOWN

    def test_the_window_and_the_majority_are_what_they_say(self):
        assert VOTE_WINDOW == 3
        assert VOTE_NEEDED == 2

    def test_a_vote_keeps_the_latest_evidence(self):
        """So a consumer can still see WHY it is unknown."""
        assert vote((), NO_FACE).evidence == EVIDENCE_NO_FACE
        assert vote((), TOWARD).evidence == EVIDENCE_FACE


class TestTheEstimateIsRunAtACadence:
    def test_it_does_not_run_on_every_frame(self):
        """Explicitly at a 2.0 s interval, not the default.

        Pinned here so this test keeps testing the cadence MECHANISM
        rather than whatever the cadence constant currently is.
        """
        facing = FixedFacingEstimator([_sees()])
        engine = _engine(facing, interval=2.0)

        for index in range(20):
            engine.observe(FRAME, received_at=index * 0.3)

        # 20 frames over 5.7 s at a 2 s cadence: 4 calls at most.
        assert facing.calls <= 4, f"facing ran {facing.calls} times in 20 frames"

    def test_it_runs_at_least_once(self):
        facing = FixedFacingEstimator([_sees()])
        engine = _engine(facing)

        for index in range(6):
            engine.observe(FRAME, received_at=index * 0.3)

        assert facing.calls >= 1

    def test_it_is_asked_about_the_trackers_boxes_not_its_own(self):
        """The detector decides what exists; the estimator only describes it."""
        facing = FixedFacingEstimator([_sees()])
        engine = _engine(facing)
        for index in range(4):
            engine.observe(FRAME, received_at=index * 0.3)

        assert facing.asked, "the estimator was never asked"
        assert all(box == BoundingBox(*PERSON_BOX) for box in facing.asked[0])

    def test_it_is_not_asked_when_nobody_is_in_view(self):
        facing = FixedFacingEstimator([_sees()])
        engine = SceneEngine(
            FixedDetector([[]] * 20), POLICY, clock=lambda: 0.0, facing_estimator=facing
        )
        engine.load()
        for index in range(6):
            engine.observe(FRAME, received_at=index * 0.3)
        assert facing.calls == 0

    def test_it_never_runs_when_orientation_is_disabled(self):
        engine = SceneEngine(
            FixedDetector([_person()] * 20), POLICY, clock=lambda: 0.0
        )
        engine.load()

        state = None
        for index in range(10):
            state = engine.observe(FRAME, received_at=index * 0.3)

        assert engine.orientation_enabled is False
        assert all(
            track.facing.state == FACING_UNKNOWN for track in state.tracks
        )


class TestTheVoteInsideTheEngine:
    def test_a_single_run_does_not_publish_toward(self):
        facing = FixedFacingEstimator([_sees()])
        engine = _engine(facing, interval=100.0)
        state = None
        for index in range(4):
            state = engine.observe(FRAME, received_at=index * 0.3)
        assert facing.calls == 1
        assert state.of_class("person")[0].facing.state == FACING_UNKNOWN

    def test_two_agreeing_runs_publish_toward(self):
        facing = FixedFacingEstimator([_sees()])
        engine = _engine(facing, interval=0.5)
        state = None
        for index in range(6):
            state = engine.observe(FRAME, received_at=index * 0.3)
        assert facing.calls >= 2
        assert state.of_class("person")[0].facing.state == FACING_TOWARD
        assert state.of_class("person")[0].facing.confidence is Confidence.MEDIUM


class TestEveryEstimateCarriesItsAge:
    def _facing_twice(self):
        # Two agreeing runs, then no more runs for a long time.
        return FixedFacingEstimator([_sees(), _sees()])

    def test_a_fresh_estimate_reports_a_small_age(self):
        facing = self._facing_twice()
        engine = _engine(facing, interval=0.5)

        state = None
        for index in range(4):
            state = engine.observe(FRAME, received_at=index * 0.3)

        person = state.of_class("person")[0]
        assert person.facing.state == FACING_TOWARD
        assert person.facing.age_seconds == pytest.approx(0.3, abs=0.01)

    def test_the_age_grows_between_estimates(self):
        facing = FixedFacingEstimator([_sees()])
        engine = _engine(facing, interval=100.0)

        ages = []
        for index in range(8):
            state = engine.observe(FRAME, received_at=index * 0.5)
            if state.of_class("person"):
                ages.append(state.of_class("person")[0].facing.age_seconds)

        assert ages == sorted(ages)
        assert ages[-1] > ages[0]

    def test_an_estimate_older_than_the_limit_becomes_unknown(self):
        """Not silently kept, and not deleted either.

        Deleting it would leave a consumer reading a missing field as
        "not facing", which is the same observation-gap error in a
        different costume.
        """
        facing = FixedFacingEstimator([_sees()])
        engine = SceneEngine(
            FixedDetector([_person()] * 200),
            POLICY,
            clock=lambda: 0.0,
            facing_estimator=facing,
            orientation_interval_s=0.5,
        )
        engine.load()
        # Two runs establish `toward`; then the estimator is starved by a
        # huge interval.
        state = None
        for index in range(4):
            state = engine.observe(FRAME, received_at=index * 0.3)
        assert state.of_class("person")[0].facing.state == FACING_TOWARD
        engine._orientation_interval = 1000.0

        at = 1.2
        while at <= MAX_ESTIMATE_AGE_S + 2.0:
            state = engine.observe(FRAME, received_at=at)
            at += 0.5

        person = state.of_class("person")[0]
        assert person.facing.state == FACING_UNKNOWN
        assert person.facing.age_seconds > MAX_ESTIMATE_AGE_S

    def test_ageing_an_estimate_preserves_its_evidence_until_it_expires(self):
        estimate = FacingEstimate(
            state=FACING_TOWARD, confidence=Confidence.MEDIUM, evidence=EVIDENCE_FACE
        )

        fresh = age_estimate(estimate, 1.0)
        expired = age_estimate(estimate, MAX_ESTIMATE_AGE_S + 1.0)

        assert fresh.state == FACING_TOWARD
        assert fresh.evidence == EVIDENCE_FACE
        assert expired.state == FACING_UNKNOWN
        assert expired.confidence is Confidence.UNKNOWN
        assert expired.evidence == EVIDENCE_FACE, "why it was what it was survives"


class TestFailuresAreLocal:
    def test_an_estimate_for_a_box_nobody_asked_about_is_ignored(self):
        """The detector decides what exists.

        An answer about a box far from any person must not create a
        phantom or move anyone's facing.
        """
        facing = FixedFacingEstimator([_sees(box=(500, 20, 560, 90))])
        engine = _engine(facing)

        state = None
        for index in range(6):
            state = engine.observe(FRAME, received_at=index * 0.3)

        assert state.count("person") == 1
        assert state.of_class("person")[0].facing.state == FACING_UNKNOWN

    def test_a_failing_estimator_does_not_end_the_session(self):
        """One model failure must not cost the whole scene."""

        class _Exploding:
            name = "exploding"

            def load(self):
                return None

            def estimate(self, frame_bgr, boxes):
                raise RuntimeError("model OOM")

            def release(self):
                return None

        engine = _engine(_Exploding())

        state = None
        for index in range(8):
            state = engine.observe(FRAME, received_at=index * 0.3)

        assert state.count("person") == 1
        assert engine.frames_observed == 8
        assert engine.orientation_enabled is False

    def test_an_estimator_that_answers_the_wrong_number_of_boxes_is_ignored(self):
        class _Wrong:
            name = "wrong"

            def load(self):
                return None

            def estimate(self, frame_bgr, boxes):
                return [TOWARD, TOWARD]

            def release(self):
                return None

        engine = _engine(_Wrong(), interval=0.1)
        state = None
        for index in range(8):
            state = engine.observe(FRAME, received_at=index * 0.3)
        assert state.of_class("person")[0].facing.state == FACING_UNKNOWN
        assert engine.orientation_enabled is False


class TestAFailingDetectorDoesNotEndTheSession:
    def test_detection_failure_is_an_empty_frame_not_a_crash(self):
        class _Exploding:
            name = "exploding"

            def load(self):
                return None

            def detect(self, frame_bgr):
                raise RuntimeError("model OOM")

            def release(self):
                return None

        engine = SceneEngine(_Exploding(), POLICY, clock=lambda: 0.0)
        engine.load()

        state = None
        for index in range(6):
            state = engine.observe(FRAME, received_at=index * 0.3)

        assert engine.frames_observed == 6
        assert state.counts == {}


class TestAgeIsPerTrackNotPerRun:
    """A run that fails to find someone must not refresh THEIR estimate.

    The age was once computed from the last orientation RUN, not from
    when this track was last estimated. Those diverge exactly when
    staleness matters most. Measured before the fix: an estimate made at
    t=0 still reported `age=1.0` and `toward_wearer` at **t=11**.

    With boxes now asked for rather than discovered, the case is a
    track the tracker has LOST for a frame (misses > 0 keeps it alive
    but it is still asked about) -- so the engine-level guard is kept
    and tested through the ageing path directly.
    """

    def test_a_track_that_was_never_estimated_reports_unknown(self):
        """And not the previous occupant's reading."""
        facing = FixedFacingEstimator([[]])
        engine = SceneEngine(
            FixedDetector([_person()] * 20),
            POLICY,
            clock=lambda: 0.0,
            facing_estimator=facing,
        )
        engine.load()

        state = None
        for index in range(8):
            state = engine.observe(FRAME, received_at=index * 0.3)

        person = state.of_class("person")[0]
        assert person.facing.state == FACING_UNKNOWN

    def test_a_stale_estimate_eventually_expires_to_unknown(self):
        """The expiry only works if the age is honest."""
        facing = FixedFacingEstimator([_sees(), _sees(), [], [], [], [], [], [], [], [], [], [], [], []])
        engine = _engine(facing, interval=0.5)

        state = None
        at = 0.0
        toward_seen = False
        while at <= MAX_ESTIMATE_AGE_S + 3.0:
            state = engine.observe(FRAME, received_at=at)
            people = state.of_class("person")
            if people and people[0].facing.state == FACING_TOWARD:
                toward_seen = True
            at += 0.5

        assert toward_seen, "the two agreeing runs must have produced toward"
        person = state.of_class("person")[0]
        assert person.facing.state == FACING_UNKNOWN


class TestTheCadenceIsDerivedNotGuessed:
    """The cadence constant must show its arithmetic."""

    def test_the_delivered_frame_interval_is_the_measured_one(self):
        """83.5 ms == 12.0 fps, from the corpus receipt timestamps."""
        assert DELIVERED_FRAME_INTERVAL_S == pytest.approx(0.0835)
        assert 1.0 / DELIVERED_FRAME_INTERVAL_S == pytest.approx(12.0, abs=0.05)

    def test_the_cadence_is_a_whole_number_of_delivered_frames(self):
        assert ORIENTATION_FRAME_STRIDE == int(ORIENTATION_FRAME_STRIDE)
        assert ORIENTATION_INTERVAL_S == pytest.approx(
            ORIENTATION_FRAME_STRIDE * DELIVERED_FRAME_INTERVAL_S
        )

    def test_the_stride_is_one_tracker_confirmation_window(self):
        assert ORIENTATION_FRAME_STRIDE == TrackerPolicy.min_hits

    def test_a_claim_needs_about_half_a_second_of_evidence(self):
        """Two agreeing estimates at this cadence."""
        assert VOTE_NEEDED * ORIENTATION_INTERVAL_S == pytest.approx(0.5, abs=0.05)
        assert ORIENTATION_INTERVAL_S < MAX_ESTIMATE_AGE_S / 10
