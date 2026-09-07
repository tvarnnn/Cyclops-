"""The small decisions the scene makes about where, how big, and whose.

Each has a number behind it (`tower/scene/state.py`, `tracking.py`,
`detect.py`) and each is the kind of constant that gets "tidied" later
by someone who does not know why it is the value it is. These tests are
where that person finds out.
"""

import pytest

from tower.scene.detect import (
    DETECTOR_CHOICES,
    HUB_SCORE_THRESHOLD,
    SCORE_THRESHOLD,
    resolve_choice,
    score_threshold_for,
)
from tower.scene.records import (
    SIDE_CENTRE,
    SIDE_LEFT,
    SIDE_RIGHT,
    SIDE_UNKNOWN,
    BoundingBox,
    Track,
)
from tower.scene.state import (
    SIDE_HYSTERESIS,
    SIDE_LEFT_BELOW,
    SIDE_RIGHT_ABOVE,
    SIZE_LARGE,
    SIZE_MEDIUM,
    SIZE_SMALL,
    SIZE_UNKNOWN,
    apparent_size,
    assign_side,
    is_partial_at_bottom_edge,
)
from tower.scene.tracking import (
    Tracker,
    TrackerPolicy,
    maximum_weight_assignment,
)


def _track(label, box, track_id=1):
    return Track(
        track_id=track_id,
        label=label,
        box=BoundingBox(*box),
        score=0.9,
        first_seen_at=0.0,
        last_seen_at=0.0,
    )


class TestSides:
    def test_the_centre_band_is_wide_enough_to_hold_a_person(self):
        """0.45/0.55 was 4.7 degrees of a 44.7-degree field: narrower than
        a torso at two metres. 0.35/0.65 is 14 degrees."""
        assert SIDE_LEFT_BELOW == pytest.approx(0.35)
        assert SIDE_RIGHT_ABOVE == pytest.approx(0.65)

    def test_a_fresh_placement_uses_the_plain_boundaries(self):
        assert assign_side(None, 0.2) == SIDE_LEFT
        assert assign_side(None, 0.5) == SIDE_CENTRE
        assert assign_side(None, 0.8) == SIDE_RIGHT

    def test_unknown_frame_is_unknown_side_whatever_came_before(self):
        assert assign_side(SIDE_LEFT, None) == SIDE_UNKNOWN

    def test_a_thing_on_the_line_keeps_its_word(self):
        """Hysteresis: crossing back needs the margin."""
        just_inside_left = SIDE_LEFT_BELOW + SIDE_HYSTERESIS / 2
        assert assign_side(SIDE_LEFT, just_inside_left) == SIDE_LEFT
        assert assign_side(SIDE_CENTRE, just_inside_left) == SIDE_CENTRE
        assert assign_side(None, just_inside_left) == SIDE_CENTRE

    def test_a_real_move_still_changes_the_word(self):
        assert assign_side(SIDE_LEFT, SIDE_LEFT_BELOW + SIDE_HYSTERESIS * 2) == SIDE_CENTRE
        assert assign_side(SIDE_RIGHT, 0.2) == SIDE_LEFT

    def test_the_hysteresis_is_above_frame_jitter_and_below_a_move(self):
        """Corpus median inter-frame box motion is 4 px of 360; 0.03 is 11."""
        assert 4 / 360 < SIDE_HYSTERESIS < 0.1


class TestApparentSize:
    def test_buckets_by_box_height_fraction(self):
        assert apparent_size(_track("person", (0, 0, 50, 300)), 360) == SIZE_LARGE
        assert apparent_size(_track("person", (0, 0, 50, 150)), 360) == SIZE_MEDIUM
        assert apparent_size(_track("person", (0, 0, 50, 80)), 360) == SIZE_SMALL

    def test_unknown_frame_is_unknown_size(self):
        assert apparent_size(_track("person", (0, 0, 50, 300)), 0) == SIZE_UNKNOWN


class TestPartialAtBottomEdge:
    def test_a_box_cut_off_at_the_bottom_with_no_head_region_is_partial(self):
        # 640 high: bottom at 632 (>= 0.97), top at 320 (>= 0.45)
        assert is_partial_at_bottom_edge(_track("person", (0, 320, 200, 632)), 640)

    def test_a_person_whose_head_is_in_view_is_not_partial(self):
        assert not is_partial_at_bottom_edge(_track("person", (0, 100, 200, 632)), 640)

    def test_a_box_that_stops_short_of_the_edge_is_not_partial(self):
        assert not is_partial_at_bottom_edge(_track("person", (0, 320, 200, 560)), 640)

    def test_only_people_are_ever_partial(self):
        assert not is_partial_at_bottom_edge(_track("chair", (0, 320, 200, 632)), 640)

    def test_unknown_frame_is_never_partial(self):
        assert not is_partial_at_bottom_edge(_track("person", (0, 320, 200, 632)), 0)


class TestAssignment:
    def test_cardinality_wins_before_overlap(self):
        """The starvation geometry: two tracks, T2's only option is T1's
        best. Both must be matched."""
        weights = [[1.0 + 1.0, 1.0 + 0.33], [1.0 + 0.25, 0.0]]
        pairs = maximum_weight_assignment(weights)
        assert pairs == [(0, 1), (1, 0)]

    def test_among_complete_matchings_overlap_decides(self):
        """Two adjacent people under a head turn: the higher-IoU pairing
        wins, which is what Kuhn's visit order got wrong."""
        weights = [[1.0 + 0.9, 1.0 + 0.6], [1.0 + 0.6, 1.0 + 0.9]]
        assert maximum_weight_assignment(weights) == [(0, 0), (1, 1)]

    def test_zero_weight_pairs_are_dropped_by_the_tracker(self):
        tracker = Tracker(TrackerPolicy(min_iou=0.25, min_hits=1))
        from tower.scene.tracking import detections_from_boxes

        tracker.update(detections_from_boxes("person", [(0, 0, 100, 100)]), at=0.0)
        # A detection nowhere near the track: a new track, not a match.
        tracker.update(detections_from_boxes("person", [(400, 400, 500, 500)]), at=0.1)
        assert len(tracker.tracks) == 2
        assert tracker.tracks[0].misses == 1

    def test_rectangular_and_empty_matrices(self):
        assert maximum_weight_assignment([]) == []
        assert maximum_weight_assignment([[0.5, 0.9, 0.1]]) == [(0, 1)]
        assert maximum_weight_assignment([[0.5], [0.9], [0.1]]) == [(1, 0)]

    def test_it_agrees_with_brute_force_on_random_matrices(self):
        import itertools
        import random

        rng = random.Random(7)
        for _ in range(200):
            rows = rng.randint(1, 5)
            cols = rng.randint(1, 5)
            weights = [[rng.choice([0.0, rng.random()]) for _ in range(cols)] for _ in range(rows)]
            best = -1.0
            for perm in itertools.permutations(range(cols), min(rows, cols)):
                for chosen_rows in itertools.combinations(range(rows), min(rows, cols)):
                    total = sum(weights[r][c] for r, c in zip(chosen_rows, perm))
                    best = max(best, total)
            got = sum(weights[r][c] for r, c in maximum_weight_assignment(weights))
            assert got == pytest.approx(best), (weights, got, best)


class TestDetectorChoice:
    def test_auto_is_by_device(self):
        assert resolve_choice("cuda", "auto") == "rtdetr_v2_r18"
        assert resolve_choice("cuda:0", "auto") == "rtdetr_v2_r18"
        assert resolve_choice("cpu", "auto") == "ssdlite320"

    def test_a_named_choice_is_kept_on_either_device(self):
        assert resolve_choice("cpu", "rtdetr_v2_r18") == "rtdetr_v2_r18"
        assert resolve_choice("cuda", "ssdlite320") == "ssdlite320"

    def test_an_unknown_choice_fails_loudly(self):
        with pytest.raises(ValueError):
            resolve_choice("cuda", "yolo")

    def test_every_choice_is_described(self):
        assert "auto" in DETECTOR_CHOICES
        assert all(isinstance(text, str) and text for text in DETECTOR_CHOICES.values())

    def test_the_thresholds_are_per_detector(self):
        """SSDLite keeps the platform's 0.4; the hub detectors count best
        at 0.5 (per-image people count exact on 73% of labelled images
        against 64% at 0.4)."""
        assert score_threshold_for("ssdlite320") == SCORE_THRESHOLD == pytest.approx(0.4)
        assert score_threshold_for("rtdetr_v2_r18") == HUB_SCORE_THRESHOLD == pytest.approx(0.5)
        assert score_threshold_for("dfine_s") == HUB_SCORE_THRESHOLD


class TestTheEnginePublishesTheDetectorsThreshold:
    def test_a_detector_with_a_threshold_sets_the_states(self):
        from tower.scene.detect import FixedDetector
        from tower.scene.engine import SceneEngine

        detector = FixedDetector([[]])
        detector.score_threshold = 0.5
        engine = SceneEngine(detector, clock=lambda: 0.0)
        engine.load()
        import numpy as np

        state = engine.observe(np.zeros((360, 640, 3), np.uint8), received_at=0.0)
        assert state.score_threshold == 0.5


class TestThePerEntityQueryLayerNeverReachesTheWire:
    def test_no_route_or_result_module_imports_the_query_layer(self):
        """`tower/scene/query.py` answers questions with track ids and
        boxes -- the per-entity shape the wire refuses. It exists for the
        CLI and for tests; a route that imported it would be the debug
        endpoint that publishes a movement trace."""
        import pathlib

        offenders = []
        for path in pathlib.Path("tower").rglob("*.py"):
            if "scene" in path.parts and path.name == "query.py":
                continue
            text = path.read_text(encoding="utf-8")
            if "tower.scene.query" in text or "from tower.scene import query" in text:
                if path.parts[1] in ("routes", "results") or path.name == "main.py":
                    offenders.append(path.as_posix())
        assert offenders == []
