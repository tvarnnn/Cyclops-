"""Counting must use tracking, and this is where that is proven.

The brief singles this out: *person counting must use tracking rather
than naively summing detections*. Two failure modes make it
non-negotiable, and they pull in opposite directions:

- a detector that misses someone on one frame in five reports a count
  flickering between 2 and 3 while nothing in the room changed;
- a detector that fires twice on one person reports two people.

Every assertion here is against a detection script this file wrote, so
the correct count is known independently of the tracker.
"""

import pytest

from tower.scene.records import BoundingBox, Detection
from tower.scene.tracking import (
    DELIVERED_FRAME_INTERVAL_S,
    MAX_ABSENCE_S,
    Tracker,
    TrackerPolicy,
    detections_from_boxes,
    frames_in,
)

POLICY = TrackerPolicy(min_iou=0.25, min_hits=3, max_misses=5)


def _person(box, score=0.9) -> Detection:
    return Detection(label="person", score=score, box=BoundingBox(*box))


def _feed(tracker, frames, start=0.0, step=0.3):
    at = start
    for detections in frames:
        tracker.update(detections, at=at)
        at += step
    return tracker


class TestOneThingIsCountedOnce:
    def test_a_person_standing_still_is_one_track(self):
        tracker = Tracker(POLICY)
        _feed(tracker, [[_person((100, 100, 200, 400))]] * 10)

        assert tracker.count("person") == 1

    def test_a_person_walking_across_the_view_stays_one_track(self):
        """Association must survive real motion.

        The IoU floor is permissive because a box that moves 30 px
        between frames still has to associate. Measured on the corpus at
        the real 12.0 fps, the 1st percentile of same-object
        consecutive-frame IoU is 0.525 for `person` and 0.386 for `cell
        phone`; this synthetic step is harsher than either.
        """
        tracker = Tracker(POLICY)
        frames = [[_person((x, 100, x + 100, 400))] for x in range(0, 300, 30)]
        _feed(tracker, frames)

        assert tracker.count("person") == 1

    def test_two_people_are_two_tracks(self):
        tracker = Tracker(POLICY)
        frames = [
            [_person((50, 100, 150, 400)), _person((400, 100, 500, 400))]
        ] * 8
        _feed(tracker, frames)

        assert tracker.count("person") == 2


class TestADetectorDropoutDoesNotChangeTheCount:
    """The headline requirement, stated as a test.

    A count that flickers with the detector is worse than useless: it
    makes the wearer distrust every answer.
    """

    def test_a_missed_frame_does_not_drop_the_count(self):
        tracker = Tracker(POLICY)
        boxes = [_person((100, 100, 200, 400)), _person((400, 100, 500, 400))]

        # Establish, then lose ONE person for a single frame, then recover.
        _feed(tracker, [boxes] * 5)
        before = tracker.count("person")
        tracker.update([boxes[0]], at=2.0)
        during = tracker.count("person")
        tracker.update(boxes, at=2.3)
        after = tracker.count("person")

        assert (before, during, after) == (2, 2, 2)

    def test_a_sustained_dropout_of_one_in_five_frames_holds_the_count(self):
        tracker = Tracker(POLICY)
        both = [_person((100, 100, 200, 400)), _person((400, 100, 500, 400))]

        counts = []
        at = 0.0
        for index in range(30):
            detections = [both[0]] if index % 5 == 4 else both
            tracker.update(detections, at=at)
            at += 0.3
            if index >= 3:
                counts.append(tracker.count("person"))

        assert set(counts) == {2}, f"count flickered: {sorted(set(counts))}"

    def test_someone_who_leaves_stops_being_counted(self):
        """The other direction: a track that is truly gone must expire."""
        tracker = Tracker(TrackerPolicy(min_hits=3, max_misses=3))
        both = [_person((100, 100, 200, 400)), _person((400, 100, 500, 400))]
        _feed(tracker, [both] * 6)
        assert tracker.count("person") == 2

        for index in range(6):
            tracker.update([both[0]], at=3.0 + index * 0.3)

        assert tracker.count("person") == 1


class TestAFlickerIsNotAPerson:
    def test_a_single_detection_is_not_counted(self):
        """One frame is a flicker, not a person."""
        tracker = Tracker(POLICY)
        tracker.update([_person((100, 100, 200, 400))], at=0.0)

        assert tracker.count("person") == 0
        assert len(tracker.tracks) == 1, "it is tracked, just not confirmed"

    def test_confirmation_needs_the_full_hit_streak(self):
        tracker = Tracker(TrackerPolicy(min_hits=3))
        counts = []
        for index in range(4):
            tracker.update([_person((100, 100, 200, 400))], at=index * 0.3)
            counts.append(tracker.count("person"))

        assert counts == [0, 0, 1, 1]

    def test_a_one_frame_false_positive_never_reaches_the_count(self):
        tracker = Tracker(POLICY)
        real = _person((100, 100, 200, 400))
        ghost = _person((500, 50, 560, 120))

        for index in range(6):
            detections = [real, ghost] if index == 2 else [real]
            tracker.update(detections, at=index * 0.3)

        assert tracker.count("person") == 1


class TestAssociationRules:
    def test_a_chair_never_becomes_a_person(self):
        """Association is within a class.

        Cross-class association would let one label flip rewrite a
        track's identity, so a chair briefly detected as a person would
        merge the two.
        """
        tracker = Tracker(POLICY)
        box = (100, 100, 200, 400)
        _feed(tracker, [detections_from_boxes("chair", [box])] * 6)
        tracker.update(detections_from_boxes("person", [box]), at=3.0)

        assert tracker.count("chair") == 1
        assert tracker.count("person") == 0
        assert len(tracker.tracks) == 2

    def test_two_boxes_that_barely_overlap_are_two_things(self):
        tracker = Tracker(TrackerPolicy(min_iou=0.5, min_hits=1))
        tracker.update([_person((0, 0, 100, 100))], at=0.0)
        tracker.update([_person((90, 90, 190, 190))], at=0.3)

        assert tracker.count("person") == 2

    def test_one_detection_cannot_be_claimed_by_two_tracks(self):
        """Greedy association must not double-assign."""
        tracker = Tracker(TrackerPolicy(min_iou=0.1, min_hits=1))
        tracker.update(
            [_person((0, 0, 100, 100)), _person((50, 0, 150, 100))], at=0.0
        )
        assert len(tracker.tracks) == 2

        tracker.update([_person((25, 0, 125, 100))], at=0.3)

        matched = [track for track in tracker.tracks if track.misses == 0]
        assert len(matched) == 1, "one detection updated two tracks"


class TestTracksCarryNoIdentity:
    def test_ids_do_not_survive_a_reset(self):
        """A track id means "the same blob as last frame", nothing more.

        If ids survived a reset they would begin to mean "the same person
        as before", which is the identity this cartridge must never have.
        """
        tracker = Tracker(POLICY)
        _feed(tracker, [[_person((100, 100, 200, 400))]] * 5)
        first_ids = [track.track_id for track in tracker.tracks]

        tracker.reset()
        _feed(tracker, [[_person((100, 100, 200, 400))]] * 5)
        second_ids = [track.track_id for track in tracker.tracks]

        assert first_ids == second_ids == [1], (
            "ids restart, so nothing can be matched across sessions"
        )

    def test_a_person_who_leaves_and_returns_is_a_new_track(self):
        """Re-identification is deliberately absent.

        Recognising someone as "the same person as five minutes ago" needs
        appearance matching, which is the first step toward identity.
        """
        tracker = Tracker(TrackerPolicy(min_hits=2, max_misses=2))
        box = (100, 100, 200, 400)
        _feed(tracker, [detections_from_boxes("person", [box])] * 4)
        original = tracker.tracks[0].track_id

        for index in range(5):
            tracker.update([], at=3.0 + index * 0.3)
        _feed(tracker, [detections_from_boxes("person", [box])] * 4, start=10.0)

        assert tracker.tracks[0].track_id != original


class TestBoundingBoxGeometry:
    def test_iou_of_a_box_with_itself_is_one(self):
        box = BoundingBox(10, 20, 110, 220)
        assert box.iou(box) == pytest.approx(1.0)

    def test_disjoint_boxes_have_zero_iou(self):
        assert BoundingBox(0, 0, 10, 10).iou(BoundingBox(50, 50, 60, 60)) == 0.0

    def test_iou_matches_a_hand_computed_overlap(self):
        """Two 100x100 boxes offset by 50 in x: overlap 50x100 = 5000,
        union 10000 + 10000 - 5000 = 15000, so 1/3."""
        left = BoundingBox(0, 0, 100, 100)
        right = BoundingBox(50, 0, 150, 100)

        assert left.iou(right) == pytest.approx(1 / 3)

    def test_a_degenerate_box_does_not_divide_by_zero(self):
        assert BoundingBox(5, 5, 5, 5).iou(BoundingBox(0, 0, 10, 10)) == 0.0


class TestTheSharedDetectorIsAdaptedNotAdopted:
    """The platform reports a 4-tuple; this cartridge reports a BoundingBox.

    The conversion is the whole of what `scene/detect.py`'s adapter does
    now that the loader itself is shared, so it is worth an assertion of
    its own. An x/y transposition here would be invisible everywhere
    except in a relationship that came out mirrored.
    """

    def test_the_box_keeps_its_corner_order(self):
        from tower.detection import Detection as PlatformDetection
        from tower.scene.detect import to_scene_detection

        converted = to_scene_detection(
            PlatformDetection(label="cup", score=0.77, box=(1.0, 2.0, 30.0, 40.0))
        )

        assert converted.label == "cup"
        assert converted.score == 0.77
        assert isinstance(converted.box, BoundingBox)
        assert (converted.box.x0, converted.box.y0) == (1.0, 2.0)
        assert (converted.box.x1, converted.box.y1) == (30.0, 40.0)
        assert converted.box.width == 29.0
        assert converted.box.height == 38.0

    def test_the_converted_detection_still_renders(self):
        """`to_json_dict` is why this cartridge keeps its own type at all."""
        from tower.detection import Detection as PlatformDetection
        from tower.scene.detect import to_scene_detection

        rendered = to_scene_detection(
            PlatformDetection(label="chair", score=0.5, box=(0.0, 0.0, 2.0, 2.0))
        ).to_json_dict()

        assert rendered["label"] == "chair"
        assert rendered["box"] == {"x0": 0.0, "y0": 0.0, "x1": 2.0, "y1": 2.0}


class TestTheThresholdsAreDerivedNotGuessed:
    """Every threshold must show its arithmetic, or its measurement.

    `max_misses = 5` was justified in this module as "roughly 1.5 seconds
    of absence" against an assumed ~3.3 fps. The corpus's own journals
    put the delivered rate at 12.0 fps, so the number bought 0.42 s --
    short enough that a person walking behind a doorframe was dropped and
    recounted as a new person, which is the one failure the counting
    requirement exists to prevent.

    The sweep behind these values is in
    `docs/superpowers/research/2026-08-26-tracker-retune.md`: 9,145 real
    corpus frames, the shipped detector, every constant swept.
    """

    def test_the_tracker_knows_the_measured_frame_interval(self):
        """83.5 ms == 12.0 fps, from the corpus receipt timestamps.

        The tracker owns this constant rather than importing it from the
        engine, because two of its three thresholds are now derived from
        it and the engine imports the tracker, not the other way round.
        """
        assert DELIVERED_FRAME_INTERVAL_S == pytest.approx(0.0835)
        assert 1.0 / DELIVERED_FRAME_INTERVAL_S == pytest.approx(12.0, abs=0.05)

    def test_the_miss_budget_is_an_absence_in_seconds(self):
        """An occlusion is a wall-clock duration, so the budget is one.

        This is the whole defect: a miss budget written as a frame count
        silently changes meaning when the frame rate does.
        """
        assert TrackerPolicy.max_misses == frames_in(MAX_ABSENCE_S)
        assert MAX_ABSENCE_S == pytest.approx(1.0)

    def test_the_old_budget_would_not_survive_this_derivation(self):
        """5 frames is 0.42 s at the real rate, not the 1.5 s claimed."""
        assert 5 * DELIVERED_FRAME_INTERVAL_S < 0.5
        assert TrackerPolicy.max_misses > 5
        assert TrackerPolicy.max_misses * DELIVERED_FRAME_INTERVAL_S == (
            pytest.approx(1.0, abs=0.01)
        )

    def test_the_confirmation_streak_is_frames_and_not_a_duration(self):
        """min_hits counts evidence, so it does NOT scale with the rate.

        A detector's false positives are per frame, not per second, so
        this one is a frame count by nature and the 3.6x rate error did
        not corrupt it. The sweep confirms 3 from both sides: 4
        regresses count stability under detector dropout (0.774 at 40%
        against 0.965), and 2 doubles the frames on which a track the
        detector was never confident about is counted.
        """
        assert TrackerPolicy.min_hits == 3

    def test_the_iou_floor_admits_the_motion_the_corpus_actually_shows(self):
        """The 1st percentile of same-object consecutive-frame IoU.

        Measured per label on 9,145 corpus frames. The floor has to sit
        below all of them or a real object loses its own track.
        """
        measured_p1 = {"person": 0.525, "laptop": 0.613, "cell phone": 0.386}

        assert TrackerPolicy.min_iou < min(measured_p1.values())


class TestAnOcclusionIsNotANewPerson:
    """The failure the corrected budget exists to prevent.

    Counting holds through dropout because a confirmed track survives it.
    A track that does NOT survive comes back with a new `track_id` and is
    counted as somebody new -- so the miss budget is the constant that
    protects the headline capability, and it was the one that was wrong.
    """

    @staticmethod
    def _occlude(policy, absent_frames):
        tracker = Tracker(policy)
        box = (100, 100, 200, 400)
        at = 0.0
        for _ in range(6):
            tracker.update([_person(box)], at=at)
            at += DELIVERED_FRAME_INTERVAL_S
        original = tracker.tracks[0].track_id

        for _ in range(absent_frames):
            tracker.update([], at=at)
            at += DELIVERED_FRAME_INTERVAL_S
        tracker.update([_person(box)], at=at)

        return original, tracker

    def test_a_person_hidden_for_most_of_a_second_keeps_their_track_id(self):
        absent = int(0.9 / DELIVERED_FRAME_INTERVAL_S)
        original, tracker = self._occlude(TrackerPolicy(), absent)

        assert [track.track_id for track in tracker.tracks] == [original]
        assert tracker.count("person") == 1

    def test_the_old_budget_recounted_exactly_that_person(self):
        """Not a hypothetical: the same script, the shipped constant."""
        absent = int(0.9 / DELIVERED_FRAME_INTERVAL_S)
        original, tracker = self._occlude(TrackerPolicy(max_misses=5), absent)

        assert tracker.tracks[0].track_id != original

    def test_someone_who_genuinely_leaves_is_still_dropped(self):
        """The other side of the trade, which is real and is bounded.

        A budget large enough to bridge an occlusion is also large enough
        to keep claiming somebody is in the room after they have gone.
        That window is exactly `MAX_ABSENCE_S`, and it must be a window
        rather than a licence.
        """
        tracker = Tracker()
        at = 0.0
        for _ in range(6):
            tracker.update([_person((100, 100, 200, 400))], at=at)
            at += DELIVERED_FRAME_INTERVAL_S
        assert tracker.count("person") == 1

        for _ in range(TrackerPolicy.max_misses + 1):
            tracker.update([], at=at)
            at += DELIVERED_FRAME_INTERVAL_S

        assert tracker.count("person") == 0
        assert tracker.tracks == []

    def test_the_stale_window_is_a_second_and_no_longer(self):
        """One frame before the budget runs out the track is still KEPT.

        Kept is continuity: if they come back now it is the same track.
        COUNTED is a stronger claim and stops sooner -- see the next test.
        """
        tracker = Tracker()
        at = 0.0
        for _ in range(6):
            tracker.update([_person((100, 100, 200, 400))], at=at)
            at += DELIVERED_FRAME_INTERVAL_S

        for _ in range(TrackerPolicy.max_misses):
            tracker.update([], at=at)
            at += DELIVERED_FRAME_INTERVAL_S

        assert len(tracker.confirmed()) == 1, "still inside the budget"
        assert at - 6 * DELIVERED_FRAME_INTERVAL_S == pytest.approx(
            MAX_ABSENCE_S, abs=0.01
        )

    def test_the_count_window_is_half_a_second_and_no_longer(self):
        """Counted while seen within 0.5 s; kept for 1.0 s.

        Measured 2026-09-07 on labelled sequences: departure lag was the
        count's largest error, and counting only tracks seen within
        0.5 s took the exact-count rate from 0.315 to 0.370 with
        continuity unchanged.
        """
        from tower.scene.tracking import MAX_COUNT_ABSENCE_S

        tracker = Tracker()
        at = 0.0
        for _ in range(6):
            tracker.update([_person((100, 100, 200, 400))], at=at)
            at += DELIVERED_FRAME_INTERVAL_S

        for _ in range(TrackerPolicy.max_count_misses):
            tracker.update([], at=at)
            at += DELIVERED_FRAME_INTERVAL_S
        assert tracker.count("person") == 1, "still inside the count window"
        assert at - 6 * DELIVERED_FRAME_INTERVAL_S == pytest.approx(
            MAX_COUNT_ABSENCE_S, abs=0.01
        )

        tracker.update([], at=at)
        assert tracker.count("person") == 0, "no longer counted"
        assert len(tracker.confirmed()) == 1, "but still kept"
        assert TrackerPolicy.max_count_misses < TrackerPolicy.max_misses
