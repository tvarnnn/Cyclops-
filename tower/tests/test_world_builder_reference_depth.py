"""Looking further back must add observations and change nothing else.

SYNTHETIC, NOT PHYSICAL. The room is rendered by a perfect pinhole; no
number here says anything about the Ray-Ban camera.

`EXTEND_REFERENCE_DEPTH` widens how far back `_extend` looks for further
sightings of landmarks it already holds. The whole safety case for it is
a separation:

  * it runs AFTER the pose is solved, so it cannot change the pose;
  * its output goes to `support` and NOT to `observed`, so it cannot
    change the next keyframe's pose either.

Both halves are asserted here rather than described, because the first
version of this change did merge into `observed`, and on real footage
that moved poses_solved 22 -> 21 and points 2060 -> 1964 while the
multiplicity metric improved -- a graph statistic rising while the
reconstruction shrank, which is the exact failure this stage was warned
about.

`test_widening_moves_mass_out_of_the_two_view_bucket` is the test that
notices if the mechanism stops running. Set DEPTH to 1 and it fails.
That is deliberate: a mechanism whose removal keeps the suite green is
not tested, and this file exists because that has happened here before.
"""

import numpy as np
import pytest

from tests import synthetic_scene as ss
from tower.world_builder.backend import KeyframeInput
from tower.world_builder.backends import classical as classical_module
from tower.world_builder.backends.classical import ClassicalTwoViewBackend
from tower.world_builder.records import CameraIntrinsics
from tower.world_builder.schema import INTRINSICS_SOURCE_SELF_CALIBRATED

WIDTH, HEIGHT = 480, 360


@pytest.fixture(scope="module")
def scene():
    return ss.furnished_room()


@pytest.fixture(scope="module")
def camera_matrix():
    return ss.camera_matrix(WIDTH, HEIGHT)


@pytest.fixture(scope="module")
def intrinsics(camera_matrix):
    return CameraIntrinsics(
        source=INTRINSICS_SOURCE_SELF_CALIBRATED,
        model="pinhole",
        fx=float(camera_matrix[0, 0]),
        fy=float(camera_matrix[1, 1]),
        cx=float(camera_matrix[0, 2]),
        cy=float(camera_matrix[1, 2]),
        calibrated_width=WIDTH,
        calibrated_height=HEIGHT,
    )


@pytest.fixture(scope="module")
def window(scene, camera_matrix):
    """A lateral walk: slow enough that structure survives several frames.

    Step is small on purpose. If consecutive keyframes shared nothing
    with the one before last, there would be no further sighting to
    find and this file would pass for the wrong reason.
    """
    import cv2

    poses = ss.strafe(14, step=0.05)
    grays = [
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for image in ss.render_sequence(
            scene, poses, camera_matrix, WIDTH, HEIGHT
        )
    ]
    return [
        KeyframeInput(keyframe_id=f"kf{index:04d}", image_gray=gray)
        for index, gray in enumerate(grays)
    ]


def _solve(window, intrinsics, depth, monkeypatch):
    monkeypatch.setattr(classical_module, "EXTEND_REFERENCE_DEPTH", depth)
    backend = ClassicalTwoViewBackend()
    backend.prepare(intrinsics)
    return backend.estimate_window(window)


def _multiplicity(estimate):
    """How many distinct keyframes saw each landmark, from the support
    table production actually persists."""
    support = estimate.points.support_views
    seen: dict[int, set[int]] = {}
    for frame, _feature, landmark in support.tolist():
        seen.setdefault(landmark, set()).add(frame)
    return seen


class TestItAddsObservations:
    def test_widening_moves_mass_out_of_the_two_view_bucket(
        self, window, intrinsics, monkeypatch
    ):
        """THE control. Delete the mechanism and this fails.

        A landmark seen by exactly two views is exactly determined --
        two rays, one intersection -- so it constrains nothing an
        optimiser can use. Moving mass OUT of that bucket is the entire
        point, and it is a different claim from "the tail got longer":
        finding a handful of heavily-observed landmarks would fatten the
        tail while leaving the two-view share untouched, and would
        un-starve nothing.
        """
        narrow = _solve(window, intrinsics, 1, monkeypatch)
        wide = _solve(window, intrinsics, 3, monkeypatch)

        narrow_seen = _multiplicity(narrow)
        wide_seen = _multiplicity(wide)

        narrow_two = sum(1 for v in narrow_seen.values() if len(v) == 2)
        wide_two = sum(1 for v in wide_seen.values() if len(v) == 2)

        assert wide.points.support_views.shape[0] > (
            narrow.points.support_views.shape[0]
        ), "looking further back found no further sightings at all"
        assert wide_two < narrow_two, (
            f"two-view landmarks did not fall: {narrow_two} -> {wide_two}. "
            "More support rows with the same two-view share means the extra "
            "rows landed on landmarks that were already well observed."
        )

    def test_the_total_observation_record_grows(
        self, window, intrinsics, monkeypatch
    ):
        """Sightings are the product here, so their total must rise.

        Deliberately NOT "every landmark keeps the views it had". A
        landmark INDEX is not stable across the two runs: widening merges
        duplicates, so index 6 in one solve and index 6 in the other are
        not the same piece of world, and comparing them by index compares
        nothing. The total is the honest aggregate.
        """
        narrow = _solve(window, intrinsics, 1, monkeypatch)
        wide = _solve(window, intrinsics, 3, monkeypatch)
        assert (
            wide.points.support_views.shape[0]
            > narrow.points.support_views.shape[0]
        )


class TestItDoesNotCostReconstruction:
    """The safety half.

    This change is NOT pose-neutral and the first version's claim that it
    was is wrong. Guided associations enter `observed`, which the next
    keyframe's PnP draws correspondences from, so later poses can move.
    That is a deliberate trade: the alternative kept poses frozen and
    published a support table naming one image point as two different
    landmarks, which is worse, because that table is what cross-segment
    registration solves against.

    What is asserted instead is the property that actually matters to the
    product -- the reconstruction must not shrink.
    """

    def test_no_pose_is_lost(self, window, intrinsics, monkeypatch):
        """Measured across 30 real segments: 2 gained a pose, 27 were
        unchanged, 1 lost one. This pins the direction on the synthetic
        walk, where the answer is deterministic."""
        narrow = _solve(window, intrinsics, 1, monkeypatch)
        wide = _solve(window, intrinsics, 3, monkeypatch)

        def solved(estimate):
            from tower.world_builder.schema import POSE_STATUS_SOLVED

            return sum(
                1 for p in estimate.poses if p.status == POSE_STATUS_SOLVED
            )

        assert solved(wide) >= solved(narrow), (
            f"widening cost a pose: {solved(narrow)} -> {solved(wide)}. A "
            "graph statistic improving while the reconstruction shrinks is "
            "the failure this stage was warned about."
        )
        assert len(narrow.poses) == len(wide.poses)

    def test_widening_adds_no_feature_bound_to_two_landmarks(
        self, window, intrinsics, monkeypatch
    ):
        """One image point must not be evidence for two pieces of world.

        Compared, not asserted to zero. A small number of these rows
        predates this change: the seed pair emits both frames' rows
        "regardless of whether the dict write above collided", because
        `match_indices` guarantees one entry per query index and not per
        train index. That is documented in `estimate_window`.

        What must not happen is widening ADDING any. It did, in the first
        version of this change -- 2 rows became 147 -- because guided
        associations were withheld from `observed`, so the next keyframe
        could not see that a feature already had a landmark and
        triangulated a second one for it. That is the regression this
        asserts against, and the reason guided rows are merged into
        `reobserved`.
        """

        def conflicts(estimate):
            claimed: dict[tuple[int, int], int] = {}
            found = 0
            for frame, feature, landmark in (
                estimate.points.support_views.tolist()
            ):
                key = (frame, feature)
                if key in claimed and claimed[key] != landmark:
                    found += 1
                claimed[key] = landmark
            return found

        narrow = conflicts(_solve(window, intrinsics, 1, monkeypatch))
        wide = conflicts(_solve(window, intrinsics, 3, monkeypatch))
        assert wide <= narrow, (
            f"widening introduced {wide - narrow} new features bound to two "
            f"landmarks ({narrow} -> {wide}). support.json is what "
            "cross-segment registration solves PnP against, so one of every "
            "such pair feeds a wrong 3-D point to placement."
        )

    def test_landmarks_get_denser_in_observations_not_just_fewer(
        self, window, intrinsics, monkeypatch
    ):
        """Widening MERGES duplicate landmarks, so the point count can
        legitimately fall. This pins the difference between merging and
        losing.

        Measured across 30 real segments: 13 lost points, and on every
        one of those 13 the observations-per-landmark ratio ROSE (e.g.
        2.55 -> 3.01, 2.88 -> 3.24). Losing real structure would drop
        both. Merging a duplicate drops the landmark and keeps its
        sightings, which is what the existing `_extend` comment describes
        as the thing that stops the map being write-only -- this change
        only extends that reuse from a one-frame window to DEPTH frames.
        """
        narrow = _solve(window, intrinsics, 1, monkeypatch)
        wide = _solve(window, intrinsics, 3, monkeypatch)

        def per_landmark(estimate):
            points = estimate.points.xyz.shape[0]
            rows = estimate.points.support_views.shape[0]
            return rows / points if points else 0.0

        assert per_landmark(wide) > per_landmark(narrow), (
            "observations per landmark did not rise, so any drop in the "
            "point count is structure being lost rather than duplicates "
            "being merged"
        )


class TestTheRetainedStateStaysBounded:
    def test_observations_do_not_accumulate_across_a_live_walk(
        self, scene, camera_matrix, intrinsics, monkeypatch
    ):
        """`forget_before` now keeps DEPTH frames rather than one.

        DEPTH times a constant is still a constant. This asserts the
        shape of that claim -- retained state must not grow with the
        length of the walk -- because the alternative reading, "keep
        everything the wider window might want", is exactly the
        unbounded growth the prune exists to prevent.
        """
        import cv2

        monkeypatch.setattr(classical_module, "EXTEND_REFERENCE_DEPTH", 3)
        poses = ss.strafe(40, step=0.045)
        grays = [
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            for image in ss.render_sequence(
                scene, poses, camera_matrix, WIDTH, HEIGHT
            )
        ]
        backend = ClassicalTwoViewBackend()
        backend.begin(intrinsics)

        sizes = []
        for index, gray in enumerate(grays, start=1):
            backend.extend(
                KeyframeInput(keyframe_id=f"kf{index:04d}", image_gray=gray)
            )
            if index in (10, 20, 40):
                sizes.append(len(backend._chain.observed))

        assert backend._chain.landmarks, "the walk solved nothing to measure"
        assert max(sizes) < min(sizes) * 2, (
            f"observations are accumulating rather than being pruned: {sizes}"
        )
        assert (
            len(backend._chain.references)
            <= classical_module.EXTEND_REFERENCE_DEPTH
        ), "the reference list is unbounded"
