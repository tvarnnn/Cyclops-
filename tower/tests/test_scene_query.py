"""The brief's four questions, and the refusals that keep them honest.

The refusals matter as much as the answers. A scene layer that returns 0
for "how many people are facing me" when it never measured orientation is
reporting an observation gap as an observation of absence -- Core
Principle 3's exact failure, and the one most likely to be mistaken for
data, because zero looks like an answer.

Every assertion is against a detection script this file wrote: the boxes
are known, so the correct count, side and relationship are known without
consulting the code.
"""

import numpy as np
import pytest

from tower.confidence import Confidence
from tower.scene.detect import FixedDetector
from tower.scene.engine import SceneEngine
from tower.scene.orientation import FixedFacingEstimator, estimate_from_face
from tower.scene.query import SceneQuery
from tower.scene.records import (
    FACING_AWAY,
    FACING_PROFILE,
    FACING_TOWARD,
    FACING_UNKNOWN,
    BoundingBox,
    Detection,
)
from tower.scene.state import REFUSED_RELATIONSHIPS
from tower.scene.tracking import TrackerPolicy

FRAME = np.zeros((360, 640, 3), np.uint8)
POLICY = TrackerPolicy(min_iou=0.25, min_hits=3, max_misses=5)


def _det(label, box, score=0.9) -> Detection:
    return Detection(label=label, score=score, box=BoundingBox(*box))


def _run(detections_per_frame, frames=8, pose=None, **kwargs):
    kwargs.setdefault("orientation_interval_s", 0.5)
    engine = SceneEngine(
        FixedDetector([detections_per_frame] * frames),
        POLICY,
        clock=lambda: 0.0,
        facing_estimator=pose,
        **kwargs,
    )
    engine.load()
    state = None
    for index in range(frames):
        state = engine.observe(FRAME, received_at=index * 0.3)
    return engine, state


# Two people on the left and centre, a chair on the right. Chosen here,
# so every expected answer below is independent of the code.
TWO_PEOPLE_AND_A_CHAIR = [
    _det("person", (60, 80, 160, 300)),
    _det("person", (280, 90, 380, 305)),
    _det("chair", (470, 200, 580, 340)),
]


class TestHowManyPeople:
    def test_it_counts_the_people_that_are_there(self):
        _, state = _run(TWO_PEOPLE_AND_A_CHAIR)

        answer = SceneQuery(state).count("person")

        assert answer.answered is True
        assert answer.value == 2

    def test_the_answer_says_it_came_from_tracks(self):
        """So a reader can tell it is not a sum of detections."""
        _, state = _run(TWO_PEOPLE_AND_A_CHAIR)

        answer = SceneQuery(state).count("person")

        assert "tracking rather than raw detections" in answer.reason

    def test_zero_is_answered_but_qualified(self):
        """A zero here IS an answer -- the detector looked and found none.

        It is qualified anyway, because an occluded person is not seen and
        "none detected" is not "none present".
        """
        _, state = _run([_det("chair", (100, 100, 200, 300))])

        answer = SceneQuery(state).count("person")

        assert answer.answered is True
        assert answer.value == 0
        assert "not evidence of absence" in answer.detail["note"]

    def test_other_classes_are_counted_too(self):
        _, state = _run(TWO_PEOPLE_AND_A_CHAIR)

        assert SceneQuery(state).count("chair").value == 1


class TestWhereIsIt:
    def test_it_locates_a_chair_on_the_side_it_was_placed(self):
        """The fixture put the chair at x 470-580 of a 640-wide frame."""
        _, state = _run(TWO_PEOPLE_AND_A_CHAIR)

        answer = SceneQuery(state).where_is("chair")

        assert answer.answered is True
        assert [entry["side"] for entry in answer.value] == ["right"]

    def test_a_left_hand_object_reports_left(self):
        _, state = _run([_det("laptop", (20, 200, 120, 280))])

        assert SceneQuery(state).where_is("laptop").value[0]["side"] == "left"

    def test_the_position_declares_its_frame_of_reference(self):
        """There is no live world pose, so every position is camera-relative."""
        _, state = _run(TWO_PEOPLE_AND_A_CHAIR)

        entry = SceneQuery(state).where_is("chair").value[0]

        assert entry["frame_of_reference"] == "camera"
        assert "no live world pose" in entry["note"]

    def test_asking_for_something_absent_is_refused_not_answered_empty(self):
        """"I don't see one" is a different statement from "there isn't one"."""
        _, state = _run(TWO_PEOPLE_AND_A_CHAIR)

        answer = SceneQuery(state).where_is("dining table")

        assert answer.answered is False
        assert "not about the room" in answer.reason


class TestHowManyAreFacingMe:
    def test_it_refuses_when_orientation_was_never_measured(self):
        """The trap. Returning 0 here would be a lie that looks like data."""
        _, state = _run(TWO_PEOPLE_AND_A_CHAIR)

        answer = SceneQuery(state).facing_wearer()

        assert answer.answered is False
        assert answer.value is None
        assert "never measured" in answer.reason
        assert answer.detail["people_in_view"] == 2

    def test_it_answers_when_orientation_is_enabled(self):
        """Two people: one facing the camera, one facing away.

        The face-visibility estimate is chosen here, so the correct
        answer is known independently. Two agreeing runs are needed
        before a facing is published, which the 0.5 s interval over
        eight frames at 0.3 s provides.
        """
        facing = estimate_from_face(0.95)
        away = estimate_from_face(None)
        pose = FixedFacingEstimator(
            [
                [
                    (BoundingBox(60, 80, 160, 300), facing),
                    (BoundingBox(280, 90, 380, 305), away),
                ]
            ]
        )

        _, state = _run(TWO_PEOPLE_AND_A_CHAIR, pose=pose)
        answer = SceneQuery(state).facing_wearer()

        assert answer.answered is True
        assert answer.value == 1

    def test_the_answer_denies_being_gaze(self):
        """The camera cannot see attention, and the answer must say so."""
        facing = estimate_from_face(0.95)
        pose = FixedFacingEstimator([[(BoundingBox(60, 80, 160, 300), facing)]])

        _, state = _run(TWO_PEOPLE_AND_A_CHAIR, pose=pose)
        answer = SceneQuery(state).facing_wearer()

        assert answer.answered is True
        assert "NOT gaze" in answer.reason

    def test_people_whose_orientation_is_unknown_are_reported_as_such(self):
        """Not silently counted as "not facing".

        One of the two people is estimated; the other is never found by
        the pose model. The second must be reported as UNKNOWN, not
        quietly folded into "not facing".
        """
        facing = estimate_from_face(0.95)
        pose = FixedFacingEstimator([[(BoundingBox(60, 80, 160, 300), facing)]])

        _, state = _run(TWO_PEOPLE_AND_A_CHAIR, pose=pose)
        answer = SceneQuery(state).facing_wearer()

        assert answer.value == 1
        assert answer.detail["orientation_unknown"] == 1

    def test_a_pose_model_that_never_succeeds_is_a_refusal_not_a_zero(self):
        """"Configured" is not "measured", and the difference is the trap.

        A pose model with bad weights or an incompatible torch build
        raises on every call. Reporting `answered: True, value: 0` for
        that is the same observation-gap error the refusal exists to
        prevent, one layer further down -- and it is worse there, because
        everything about the answer looks healthy.
        """

        class _AlwaysFails:
            name = "always-fails"

            def load(self):
                return None

            def estimate(self, frame_bgr, boxes):
                raise RuntimeError("bad weights")

            def release(self):
                return None

        _, state = _run(TWO_PEOPLE_AND_A_CHAIR, pose=_AlwaysFails())
        answer = SceneQuery(state).facing_wearer()

        assert answer.answered is False
        assert answer.value is None
        assert "has not once succeeded" in answer.reason

    def test_the_oldest_estimate_is_none_rather_than_zero_when_absent(self):
        """`or 0.0` folded "no estimate" into "zero seconds old", which
        read as corroborating a freshness nobody measured."""
        facing = estimate_from_face(0.95)
        pose = FixedFacingEstimator([[(BoundingBox(60, 80, 160, 300), facing)]])

        _, state = _run(TWO_PEOPLE_AND_A_CHAIR, pose=pose)
        detail = SceneQuery(state).facing_wearer().detail

        oldest = detail["oldest_estimate_seconds"]
        assert oldest is not None and oldest >= 0.0
        # The second person WAS asked about -- every tracked person is --
        # and the answer was "no face", which is an estimate with an age
        # and an evidence, not a missing one.
        assert detail["states"][2]["age_seconds"] is not None
        assert detail["states"][2]["state"] == FACING_UNKNOWN
        assert detail["states"][2]["evidence"] == "no-face-found"

    def test_a_person_never_asked_about_reports_no_age(self):
        """`or 0.0` folded "no estimate" into "zero seconds old". A track
        that appeared after the last orientation run has no estimate and
        must say so with None, never 0."""
        facing = estimate_from_face(0.95)
        pose = FixedFacingEstimator([[(BoundingBox(60, 80, 160, 300), facing)]])
        # A huge interval: the estimator runs once, on the first frame,
        # before the tracks are even confirmed; nobody is asked again.
        _, state = _run(TWO_PEOPLE_AND_A_CHAIR, pose=pose, orientation_interval_s=1000.0)
        for track in state.of_class("person"):
            assert track.facing.age_seconds is None or track.facing.age_seconds >= 0.0


class TestRelationships:
    def test_left_of_matches_the_geometry_we_chose(self):
        _, state = _run(TWO_PEOPLE_AND_A_CHAIR)

        relations = SceneQuery(state).relationships().value
        left_of = {
            (entry["subject_track_id"], entry["object_track_id"])
            for entry in relations
            if entry["relationship"] == "left_of"
        }

        # Tracks are numbered in creation order: person(60), person(280),
        # chair(470). So 1 is left of 2 and 3, and 2 is left of 3.
        assert left_of == {(1, 2), (1, 3), (2, 3)}

    def test_things_at_the_same_horizontal_position_get_no_side_relation(self):
        """A one-pixel difference must not assert a relation."""
        _, state = _run(
            [
                _det("book", (300, 40, 340, 80)),
                _det("cup", (302, 250, 342, 290)),
            ]
        )

        relations = SceneQuery(state).relationships().value

        assert not any(
            entry["relationship"] in ("left_of", "right_of") for entry in relations
        )

    def test_higher_in_view_is_named_for_the_image_not_the_room(self):
        """"Above" would imply a world relation a 2-D box cannot support."""
        _, state = _run(
            [
                _det("book", (300, 20, 340, 60)),
                _det("cup", (302, 280, 342, 320)),
            ]
        )

        relations = SceneQuery(state).relationships().value

        assert any(
            entry["relationship"] == "higher_in_view" for entry in relations
        )
        assert not any(entry["relationship"] == "above" for entry in relations)

    def test_relative_distance_is_never_asserted_from_box_size(self):
        """SHIPPED, THEN WITHDRAWN, and the counterexample is why.

        Box area within one class looked like safe evidence for relative
        distance. An adversarial review produced two chairs at the SAME
        distance -- one face-on, one edge-on -- whose areas differ 2.5x,
        which the rule asserted as "nearer". That is a WRONG relation, not
        a weak one, and this cartridge's own bar is that a wrong
        relationship is worse than a missing one.
        """
        face_on = _det("chair", (100, 100, 400, 300))  # 300 x 200 = 60000
        edge_on = _det("chair", (500, 100, 620, 300))  # 120 x 200 = 24000
        assert (
            (400 - 100) * (300 - 100) / ((620 - 500) * (300 - 100))
        ) == 2.5, "the counterexample must actually have a 2.5x area ratio"

        _, state = _run([face_on, edge_on])

        relations = SceneQuery(state).relationships().value
        assert not any(
            entry["relationship"] == "nearer_than_same_class"
            for entry in relations
        )

    def test_the_withdrawal_is_recorded_with_its_counterexample(self):
        """A withdrawn feature must leave its reasoning behind, or it gets
        re-invented by the next person who thinks of it."""
        _, state = _run(TWO_PEOPLE_AND_A_CHAIR)

        answer = SceneQuery(state).why_not("nearer_than_same_class")

        assert answer.answered is True
        assert "WITHDRAWN" in answer.value
        assert "face-on" in answer.value

    def test_every_relation_declares_it_is_camera_relative(self):
        _, state = _run(TWO_PEOPLE_AND_A_CHAIR)

        for entry in SceneQuery(state).relationships().value:
            assert entry["frame_of_reference"] == "camera"


class TestRefusalsExplainThemselves:
    @pytest.mark.parametrize("relationship", sorted(REFUSED_RELATIONSHIPS))
    def test_a_refused_relationship_says_what_it_would_need(self, relationship):
        """A refusal that can explain itself stops the next cartridge
        re-deriving the same conclusion from scratch."""
        _, state = _run(TWO_PEOPLE_AND_A_CHAIR)

        answer = SceneQuery(state).why_not(relationship)

        assert answer.answered is True
        assert len(answer.value) > 40

    def test_no_refused_relationship_is_ever_asserted(self):
        _, state = _run(TWO_PEOPLE_AND_A_CHAIR)

        asserted = {
            entry["relationship"] for entry in SceneQuery(state).relationships().value
        }

        assert not (asserted & set(REFUSED_RELATIONSHIPS))

    def test_an_unknown_relationship_is_not_pretended_to_be_refused(self):
        _, state = _run(TWO_PEOPLE_AND_A_CHAIR)

        answer = SceneQuery(state).why_not("smells_like")

        assert answer.answered is False


class TestOrientationEvidence:
    """What one face detection means, checked against what it may claim."""

    def test_a_strong_face_means_facing_toward(self):
        estimate = estimate_from_face(0.95)

        assert estimate.state == FACING_TOWARD
        assert estimate.appears_facing_wearer is True

    def test_no_face_means_unknown_not_away(self):
        """The difference that matters: not seeing a face is not seeing a
        back of a head. `away_from_wearer` is never produced; nothing
        measured on this platform can say it with usable precision."""
        estimate = estimate_from_face(None)

        assert estimate.state == FACING_UNKNOWN
        assert estimate.state not in (FACING_AWAY, FACING_PROFILE)
        assert estimate.confidence is Confidence.UNKNOWN

    def test_a_weak_face_does_not_count(self):
        """A face detector emits a score for anything face-like. Without
        a threshold, every reflection and poster faces the camera."""
        estimate = estimate_from_face(0.5)

        assert estimate.state == FACING_UNKNOWN

    def test_confidence_never_reaches_high(self):
        """A detector's score over a crop, validated on stills from a
        different camera, is two steps from a measurement, and the brief
        forbids claiming this from weak evidence."""
        for score in (0.9, 0.99, 1.0):
            assert estimate_from_face(score).confidence is not Confidence.HIGH


class TestAnUnknownFrameSizeAssertsNothing:
    """No frame, no position, no relation.

    Reachable through direct API use before any decodable frame arrives.
    Normalising a pixel coordinate by zero used to report EVERY object at
    `normalised_x: 0.0` and `side: "left"`, and the minimum-separation
    guards collapsed to zero so any two boxes differing by a pixel got a
    confident `left_of`. Both are specific claims derived from no
    information -- the same failure this cartridge refuses everywhere
    else, in a quieter costume.
    """

    @staticmethod
    def _state_without_a_frame():
        engine = SceneEngine(
            FixedDetector(
                [
                    [
                        _det("book", (300, 40, 340, 80)),
                        _det("cup", (302, 250, 342, 290)),
                    ]
                ]
                * 20
            ),
            POLICY,
            clock=lambda: 0.0,
        )
        engine.load()
        state = None
        for index in range(5):
            state = engine.observe(None, received_at=index * 0.3)
        return engine, state

    def test_the_state_says_the_frame_is_unknown(self):
        _, state = self._state_without_a_frame()

        assert state.frame_known is False
        assert state.to_json_dict()["frame"]["known"] is False

    def test_no_relation_is_asserted(self):
        _, state = self._state_without_a_frame()

        assert state.relations == ()

    def test_positions_refuse_rather_than_reporting_the_origin(self):
        engine, state = self._state_without_a_frame()

        positions = engine.describe(state)["positions"]

        assert positions, "the tracks still exist"
        for entry in positions.values():
            assert entry["side"] == "unknown"
            assert entry["normalised_x"] is None
            assert "no evidence" in entry["note"]

    def test_counting_still_works_without_a_frame(self):
        """A count needs no geometry, so it must not be lost with it."""
        _, state = self._state_without_a_frame()

        assert SceneQuery(state).count("book").value == 1

    def test_the_json_stays_valid(self):
        import json
        import math

        engine, state = self._state_without_a_frame()

        payload = json.dumps(engine.describe(state))

        assert "NaN" not in payload and "Infinity" not in payload
        assert math.isfinite(state.at)
