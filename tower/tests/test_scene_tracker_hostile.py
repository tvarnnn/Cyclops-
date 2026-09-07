"""The tracker defects an adversarial review found, pinned.

Each one produced a *wrong answer*, not a crash — a phantom person
permanently counted, a real person silently replaced by an artifact, one
stranger's orientation reported as another's. Those are the failures that
matter here, because a wrong count looks exactly like a right one.

The review also named why the original suite missed all of them, and the
reason is worth carrying: every existing tracking test used widely
separated, non-competing boxes. Nothing ever gave two tracks a shared
candidate, so nothing exercised association at all.
"""

import numpy as np
import pytest

from tower.confidence import Confidence
from tower.scene.detect import FixedDetector
from tower.scene.engine import SceneEngine
from tower.scene.records import (
    FACING_TOWARD,
    FACING_UNKNOWN,
    BoundingBox,
    Detection,
    FacingEstimate,
)
from tower.scene.tracking import Tracker, TrackerPolicy


def _person(box, score=0.9) -> Detection:
    return Detection(label="person", score=score, box=BoundingBox(*box))


class TestAFlickerNeverBecomesPermanent:
    """`hits` was a lifetime total, so intermittence accumulated.

    A reflection, a TV showing a person, or a poster glimpsed once every
    six frames reached three lifetime hits and confirmed — then stayed
    confirmed for the rest of the session, having never been seen twice in
    a row.
    """

    def test_a_detection_present_one_frame_in_six_is_never_confirmed(self):
        tracker = Tracker(TrackerPolicy(min_hits=3, max_misses=5))
        real = _person((0, 0, 100, 200))
        ghost = _person((400, 0, 460, 60))

        for index in range(40):
            detections = [real] + ([ghost] if index % 6 == 0 else [])
            tracker.update(detections, at=index * 0.3)

        assert tracker.count("person") == 1
        ghosts = [track for track in tracker.tracks if track.box.x0 == 400]
        assert ghosts and ghosts[0].hits >= 3, (
            "the ghost must have accumulated lifetime hits -- that is the trap"
        )
        assert not ghosts[0].is_confirmed

    def test_confirmation_needs_consecutive_frames(self):
        tracker = Tracker(TrackerPolicy(min_hits=3, max_misses=5))
        person = _person((0, 0, 100, 200))

        # Present, absent, present, absent, present: three hits, never
        # two in a row.
        for index in range(5):
            tracker.update([person] if index % 2 == 0 else [], at=index * 0.3)

        assert tracker.count("person") == 0

    def test_confirmation_latches_and_survives_a_dropout(self):
        """The other direction, and why a bare streak test would be wrong.

        Once a track has earned confirmation it must keep it through a
        dropout, or the count flickers exactly as it would from raw
        detections.
        """
        tracker = Tracker(TrackerPolicy(min_hits=3, max_misses=5))
        person = _person((0, 0, 100, 200))

        for index in range(5):
            tracker.update([person], at=index * 0.3)
        assert tracker.count("person") == 1

        counts = []
        for index in range(4):
            tracker.update([], at=2.0 + index * 0.3)
            counts.append(tracker.count("person"))

        assert counts == [1, 1, 1, 1]


class TestAssociationDoesNotStarveATrack:
    """Greedy took the single best pair and stole another track's only
    qualifying detection. The victim then missed every subsequent frame
    and was dropped, while a phantom confirmed in its place — sometimes
    with the total count still reading correct, which is worse."""

    @staticmethod
    def _ious():
        """The review's geometry, checked against BoundingBox.iou itself."""
        t1 = BoundingBox(0, 0, 100, 100)
        t2 = BoundingBox(60, 0, 160, 100)
        d1 = BoundingBox(0, 0, 100, 100)
        d2 = BoundingBox(-20, 0, 80, 100)
        return t1, t2, d1, d2

    def test_the_geometry_really_is_ambiguous(self):
        """Pin the counterexample so it cannot silently stop being one.

        The property that matters is structural, not the exact numbers:
        T1 has TWO qualifying candidates and T2 has exactly ONE, and T2's
        only candidate is also T1's best. Greedy takes the highest pair
        first and starves T2; a maximum-cardinality matching gives T1 its
        second choice and both tracks survive.
        """
        t1, t2, d1, d2 = self._ious()
        threshold = 0.2

        assert t1.iou(d1) == pytest.approx(1.0)
        assert t1.iou(d2) == pytest.approx(0.667, abs=0.01)
        assert t2.iou(d1) == pytest.approx(0.25, abs=0.01)
        assert t2.iou(d2) == pytest.approx(0.111, abs=0.01)

        t1_options = sum(1 for box in (d1, d2) if t1.iou(box) >= threshold)
        t2_options = sum(1 for box in (d1, d2) if t2.iou(box) >= threshold)
        assert (t1_options, t2_options) == (2, 1)
        assert t1.iou(d1) > t1.iou(d2), "greedy would take T2's only option"

    def test_both_tracks_still_get_a_detection(self):
        tracker = Tracker(TrackerPolicy(min_iou=0.2, min_hits=1))
        t1, t2, d1, d2 = self._ious()

        tracker.update(
            [Detection("person", 0.9, t1), Detection("person", 0.9, t2)], at=0.0
        )
        assert len(tracker.tracks) == 2

        tracker.update(
            [Detection("person", 0.9, d1), Detection("person", 0.9, d2)], at=0.3
        )

        assert tracker.count("person") == 2, "a phantom third track appeared"
        assert all(track.misses == 0 for track in tracker.tracks), (
            "a track starved while a detection it qualified for went unused"
        )

    def test_no_real_track_is_replaced_by_an_artifact(self):
        """With the default min_hits the count looked right while the
        second real person was quietly swapped for a new track."""
        tracker = Tracker(TrackerPolicy(min_iou=0.2, min_hits=3, max_misses=5))
        t1, t2, d1, d2 = self._ious()

        for index in range(4):
            tracker.update(
                [Detection("person", 0.9, t1), Detection("person", 0.9, t2)],
                at=index * 0.3,
            )
        original_ids = {track.track_id for track in tracker.tracks}

        for index in range(6):
            tracker.update(
                [Detection("person", 0.9, d1), Detection("person", 0.9, d2)],
                at=2.0 + index * 0.3,
            )

        assert {track.track_id for track in tracker.tracks} == original_ids


class TestReacquisitionDropsStaleEvidence:
    """A track id could be reused for a DIFFERENT person across a short
    gap, and the old person's orientation went with it."""

    def test_a_re_matched_track_forgets_its_facing(self):
        tracker = Tracker(TrackerPolicy(min_iou=0.25, min_hits=2, max_misses=5))
        box = (100, 80, 220, 320)
        for index in range(4):
            tracker.update([_person(box)], at=index * 0.3)

        track = tracker.tracks[0]
        track.facing = FacingEstimate(
            state=FACING_TOWARD, confidence=Confidence.MEDIUM, age_seconds=0.0
        )
        track.facing_estimated_at = 1.2

        tracker.update([], at=1.5)
        tracker.update([], at=1.8)
        tracker.update([_person((105, 82, 225, 322))], at=2.1)

        track = tracker.tracks[0]
        assert track.facing.state == FACING_UNKNOWN, (
            "one person's orientation was reported as another's"
        )
        assert track.facing_estimated_at is None

    def test_continuous_tracking_keeps_its_facing(self):
        """The fix must not throw away a still-valid estimate every frame."""
        tracker = Tracker(TrackerPolicy(min_iou=0.25, min_hits=2, max_misses=5))
        box = (100, 80, 220, 320)
        for index in range(4):
            tracker.update([_person(box)], at=index * 0.3)

        track = tracker.tracks[0]
        track.facing = FacingEstimate(state=FACING_TOWARD, age_seconds=0.0)
        track.facing_estimated_at = 1.2

        for index in range(4):
            tracker.update([_person(box)], at=2.0 + index * 0.3)

        assert tracker.tracks[0].facing.state == FACING_TOWARD


class TestClockRegressionDoesNotExtendAnEstimatesLife:
    def test_a_backward_step_does_not_produce_a_negative_age(self):
        """A negative age pushes the expiry deadline into the future --
        the one direction it must never move. `Track.age_seconds` already
        clamped; `FacingEstimate` did not."""
        from tower.scene.orientation import age_estimate

        estimate = FacingEstimate(state=FACING_TOWARD, age_seconds=0.0)

        aged = age_estimate(estimate, -2.0)

        assert aged.age_seconds == 0.0


class TestOrientationMustHaveActuallyRun:
    def test_a_permanently_failing_pose_model_leaves_orientation_disabled(self):
        class _AlwaysFails:
            name = "always-fails"

            def load(self):
                return None

            def estimate(self, frame_bgr, boxes):
                raise RuntimeError("bad weights")

            def release(self):
                return None

        engine = SceneEngine(
            FixedDetector([[_person((100, 80, 220, 320))]] * 20),
            TrackerPolicy(min_hits=2),
            clock=lambda: 0.0,
            facing_estimator=_AlwaysFails(),
        )
        engine.load()

        state = None
        for index in range(8):
            state = engine.observe(np.zeros((360, 640, 3), np.uint8), received_at=index * 0.3)

        assert engine.orientation_configured is True
        assert engine.orientation_enabled is False, (
            "configured is not measured"
        )
        assert state.orientation_enabled is False
