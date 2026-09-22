"""How trajectory error actually behaves along a chain.

SYNTHETIC, NOT PHYSICAL.

This file exists because a single drift percentage was misread. 21.6% at
16 keyframes was recorded as "unbounded drift"; the dominant cause was
that the walk producing it left the room at that exact keyframe. With the
walk kept inside the scene, error accumulates *smoothly* and no cliff
exists at 16 at all.

Bounds here are deliberately loose. `findEssentialMat(USAC_MAGSAC)` and
`solvePnPRansac(SQPNP)` are unseeded, and a committed test's own docstring
claims 1.32% where the same test measures 1.62% on this OpenCV build. A
tight bound would pin a build, not a behaviour. What is asserted is the
SHAPE of the error.

## THAT SHAPE CHANGED, and this file is the record of it

Until the local bundle adjustment landed, the shape was accumulation:
error grew with every keyframe, and the assertion here was the
comparison `long > short * 3`. That was true and it was the largest
single defect in the reconstruction. Measured on a strafe that stays
inside the room, with no refusal and nothing blurred:

    keyframes   rotation error med / max   max drift / path   step scale
         6         0.95 / 1.69 deg               4.7%          7.63
        12         1.69 / 3.46 deg               2.7%          8.47
        20         3.19 / 9.17 deg               9.8%          8.85
        30         5.71 / 18.88 deg             11.3%          3.46
        40         9.21 / 33.98 deg             18.2%          2.43

The last column is the per-step ratio of recovered to true camera
motion. It is flat for twenty keyframes and then falls by a factor of
three: the reconstruction CONTRACTS. A segment longer than about twenty
keyframes was internally warped, which is why cross-segment registration
-- which solves a Sim3 and refuses two pieces whose geometry disagrees
-- could place so few of them, and why simply cutting fewer segments
made the world worse rather than better.

With `world_builder/bundle.py` running over a sliding window, measured on
the same fixture:

    keyframes    8      16      24      40
    drift      0.57%   0.12%   0.16%   0.23%

Drift no longer grows with chain length, so the assertion below is now
the opposite comparison: a long chain must NOT be markedly worse than a
short one. The eight-keyframe case is the worst of the four because the
window has barely filled -- which is the honest shape of a windowed
adjustment and not a defect.
"""

import cv2
import numpy as np
import pytest

from tower.world_builder.backend import KeyframeInput
from tower.world_builder.backends.classical import ClassicalTwoViewBackend
from tower.world_builder.records import CameraIntrinsics
from tests import synthetic_scene as ss

WIDTH, HEIGHT = 480, 360


@pytest.fixture(scope="module")
def rig():
    camera_matrix = ss.camera_matrix(WIDTH, HEIGHT)
    intrinsics = CameraIntrinsics(
        source="self_calibrated",
        model="pinhole",
        fx=float(camera_matrix[0, 0]),
        fy=float(camera_matrix[1, 1]),
        cx=float(camera_matrix[0, 2]),
        cy=float(camera_matrix[1, 2]),
        calibrated_width=WIDTH,
        calibrated_height=HEIGHT,
    )
    return ss.furnished_room(), camera_matrix, intrinsics


def _in_room_walk(count: int):
    """A walk whose step shrinks so the whole chain stays inside the room."""
    usable_half_width = ss.ROOM_WIDTH_M / 2 - 0.5
    step = min(0.20, usable_half_width / max(count - 1, 1))
    poses = ss.strafe(count, step=step)
    assert ss.poses_outside_room(poses) == [], (
        "this test measures drift, not what happens when the camera walks "
        "into a wall -- the walk must stay inside the scene"
    )
    return poses


def _max_drift_fraction(rig, poses) -> float:
    scene, camera_matrix, intrinsics = rig
    images = ss.render_sequence(scene, poses, camera_matrix, WIDTH, HEIGHT)
    grays = [
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        for image in images
    ]
    backend = ClassicalTwoViewBackend()
    backend.prepare(intrinsics)
    estimate = backend.estimate_window(
        [KeyframeInput(f"k{index}", gray) for index, gray in enumerate(grays)]
    )

    centres, kept = [], []
    for index, pose in enumerate(estimate.poses):
        if pose.status == "anchor":
            centres.append(np.zeros(3))
            kept.append(index)
        elif pose.translation is not None:
            centres.append(-pose.rotation.T @ pose.translation)
            kept.append(index)

    assert len(kept) >= len(poses) - 1, "too many poses refused to measure drift"

    centres = np.asarray(centres)
    truth = np.asarray([poses[index].position for index in kept])
    scale, rotation, translation = ss.umeyama_similarity(centres, truth)
    aligned = (scale * (rotation @ centres.T).T) + translation

    residual = np.linalg.norm(aligned - truth, axis=1)
    full_truth = np.asarray([pose.position for pose in poses])
    path_length = float(np.linalg.norm(np.diff(full_truth, axis=0), axis=1).sum())
    return float(residual.max() / path_length)


class TestDriftShape:
    def test_a_short_in_room_chain_tracks_truth_closely(self, rig):
        """Eight keyframes: measured 0.83% here, bounded at 5%."""
        assert _max_drift_fraction(rig, _in_room_walk(8)) < 0.05

    def test_sixteen_keyframes_is_not_a_cliff(self, rig):
        """The whole point.

        This is the count that produced 21.6% and a diagnosis of
        "unbounded drift". Kept inside the room it measures ~1%, which
        means the earlier figure was mostly a fact about the scene.
        """
        assert _max_drift_fraction(rig, _in_room_walk(16)) < 0.05

    def test_a_long_chain_does_not_drift_further_than_a_short_one(self, rig):
        """The bundle adjustment's whole reason for existing.

        This assertion used to be `long > short * 3`, and it passed --
        error accumulated, 24 keyframes drifted several times as far as
        8, and by 40 keyframes the reconstruction had contracted by a
        factor of three. The module docstring carries those numbers.

        Now the comparison runs the other way. It is deliberately a
        comparison and not a threshold: the ratio survives an OpenCV
        upgrade, an absolute percentage does not.

        THE MUTATION THAT PROVES THIS TEST. Set
        `classical.BUNDLE_WINDOW = 0` and it fails, because that is
        exactly the engine this file used to describe.
        """
        short = _max_drift_fraction(rig, _in_room_walk(8))
        long = _max_drift_fraction(rig, _in_room_walk(24))

        assert long <= max(short, 0.01) * 1.5, (
            f"drift is accumulating again: short={short:.4f} long={long:.4f}"
        )

    def test_a_forty_keyframe_chain_stays_bounded(self, rig):
        """The length at which the old chain had failed outright.

        40 keyframes measured 18.2% drift, 9.21 deg median rotation error
        and a 3x scale contraction before the adjustment existed. 5% is
        far above what it measures now (0.23%) and far below what it
        measured then, so this fails loudly on a regression without
        pinning an OpenCV build.
        """
        assert _max_drift_fraction(rig, _in_room_walk(40)) < 0.05


class TestInlierRatioIsNotADegeneracySignal:
    """A measured negative result -- WHICH THIS BRANCH REVERSED.

    The obvious response to degrading geometry is to gate PnP on inlier
    ratio, mirroring the initial pair's gate. Measurement said that would
    be backwards: the walk that stays safely inside the room reported
    LOWER inlier ratios than the one heading into a wall, and a ratio
    floor would have refused the good configuration and admitted the bad
    one. The reading offered was that a large healthy match set with
    plenty of far-field structure has proportionally more outliers.

    THAT WAS A MEASUREMENT OF POSE ERROR, NOT OF MATCHING. The inlier
    ratio here is `solvePnPRansac`'s, so it reports how many
    correspondences agree with the pose the solver found -- and on a
    chain drifting by 9 degrees, the interior walk's own correct
    correspondences were being called outliers. With the drift removed by
    the local bundle adjustment, the ordering flips and stays flipped,
    deterministically across runs:

        interior walk (step 0.09)        0.7550
        walk into the wall (step 0.20)   0.7022

    The negative result is therefore retired, and this class now pins the
    reversal so it cannot be silently undone. It does NOT follow that an
    inlier-ratio gate should be added: the separation is 0.05 on one
    synthetic fixture, and a gate needs a corpus sweep and an adversarial
    pass of its own. What follows is only that the reason for refusing to
    consider one is gone.
    """

    def test_the_good_walk_now_reports_the_higher_inlier_ratio(self, rig):
        scene, camera_matrix, intrinsics = rig

        def ratios(poses):
            images = ss.render_sequence(scene, poses, camera_matrix, WIDTH, HEIGHT)
            grays = [
                cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
                for image in images
            ]
            backend = ClassicalTwoViewBackend()
            backend.prepare(intrinsics)
            estimate = backend.estimate_window(
                [KeyframeInput(f"k{i}", gray) for i, gray in enumerate(grays)]
            )
            return [
                pose.inlier_ratio
                for pose in estimate.poses
                if pose.inlier_ratio is not None
            ]

        interior = ratios(ss.strafe(16, step=0.09))
        toward_the_wall = ratios(ss.strafe(16, step=0.20))

        assert float(np.median(interior[4:])) > float(
            np.median(toward_the_wall[4:])
        ), (
            "the interior walk reporting the LOWER ratio again would mean "
            "pose error has returned to the level that inverted this "
            "measurement in the first place -- read it as a drift "
            "regression, not as an argument about matching"
        )
