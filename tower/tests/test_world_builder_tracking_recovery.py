"""One unsolvable keyframe must not end a coordinate frame.

WHY THIS FILE EXISTS

`_Chain.broken` was a one-way latch. The first keyframe whose pose could
not be solved set it, and every keyframe after it in that segment was
returned `unavailable` from `classical.py`'s early return WITHOUT ORB
detection, matching, or any geometry attempted. The engine's only
response was to cut a new segment -- a new coordinate frame, a new
arbitrary scale, a new fragment for the viewer to fail to connect.

Measured on the 2026-09-01 long-loop walk (2613 frames, 434 keyframes):
81 pose refusals, of which the manifest attributes 21 to a root decision
and 60 to cascade. Across every calibrated session in the corpus the
split is 137 root attempts against 1812 cascaded refusals -- 93% of all
refused poses were keyframes at which no solver ever ran.

The mechanism this file pins is the one mature visual SLAM systems all
have and this pipeline did not: a bounded RECOVERY state between
"tracking" and "give up". A failed solve leaves the map, the poses and
the references alone and the NEXT keyframe is solved against the last
keyframes that actually have poses. Only a sustained run of failures --
`MAX_RECOVERY_KEYFRAMES` of them -- breaks the chain.

WHAT IS DELIBERATELY NOT CHANGED, and what these tests therefore pin

No acceptance threshold moves. `MIN_PNP_CORRESPONDENCES`,
`PNP_REPROJECTION_ERROR_PX`, `MIN_INLIERS`, `MIN_INLIER_RATIO` and
`MIN_TRIANGULATION_ANGLE_DEG` are the same numbers before and after.
Recovery is asking the question again on a later keyframe, not lowering
the bar for the answer -- which is exactly the distinction
`test_recovery_still_refuses_when_the_geometry_is_absent` exists to
enforce, and why it must keep failing.
"""

import cv2
import numpy as np
import pytest

from tests import synthetic_scene as ss
from tower.world_builder.backend import KeyframeInput
from tower.world_builder.backends import classical
from tower.world_builder.backends.classical import ClassicalTwoViewBackend
from tower.world_builder.records import CameraIntrinsics
from tower.world_builder.schema import (
    POSE_STATUS_ANCHOR,
    POSE_STATUS_SOLVED,
    POSE_STATUS_UNAVAILABLE,
)

WIDTH, HEIGHT = 480, 360


def _intrinsics(camera):
    return CameraIntrinsics(
        source="self_calibrated",
        fx=float(camera[0][0]),
        fy=float(camera[1][1]),
        cx=float(camera[0][2]),
        cy=float(camera[1][2]),
        calibrated_width=WIDTH,
        calibrated_height=HEIGHT,
    )


def _walk(count=12, step=0.12):
    camera = ss.camera_matrix(WIDTH, HEIGHT)
    images = ss.render_sequence(
        ss.furnished_room(), ss.strafe(count, step=step), camera, WIDTH, HEIGHT
    )
    frames = [
        KeyframeInput(
            keyframe_id=f"kf{i}",
            image_gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
            image_bgr=image,
        )
        for i, image in enumerate(images)
    ]
    return camera, frames


def _noise_frame(keyframe_id, seed=7):
    """A frame with texture but no relationship to the scene.

    Not a black rectangle: a featureless frame fails at ORB detection,
    which is a different refusal path from "plenty of features, none of
    them ours". The second is what a wearer walking past a blank wall or
    swinging past a window actually produces, and it is the one that has
    to not kill the segment.
    """
    rng = np.random.default_rng(seed)
    gray = rng.integers(0, 255, (HEIGHT, WIDTH), dtype=np.uint8)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return KeyframeInput(
        keyframe_id=keyframe_id,
        image_gray=gray,
        image_bgr=cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
    )


# The SHIPPED budget is 1 -- see MAX_RECOVERY_KEYFRAMES for the
# adversarial measurement that put it there. Every test below that
# exercises the recovery MECHANISM therefore raises it explicitly, so
# that what is being tested is the mechanism and not the policy, and so
# that changing the policy cannot silently make these tests vacuous.
MECHANISM_BUDGET = 6


@pytest.fixture
def recovery_enabled(monkeypatch):
    monkeypatch.setattr(classical, "MAX_RECOVERY_KEYFRAMES", MECHANISM_BUDGET)
    return MECHANISM_BUDGET


def _run(frames, camera):
    backend = ClassicalTwoViewBackend()
    backend.begin(_intrinsics(camera))
    return [backend.extend(frame) for frame in frames], backend


def test_the_shipped_budget_is_one():
    """A policy, pinned with its reason, so a later change is deliberate.

    1 is the largest reference gap the adversarial measurement supports.
    Over texture that repeats -- an ordinary room -- a gap of 2 already
    loses 20 cm of real walking in six of nine samples, and a gap of 8
    publishes 1 mm for 1.500 m with 169 PnP inliers and support
    reprojecting at 0.22 px. Nothing in this backend refuses that, which
    is why the bound and not a gate has to carry it.

    See tests/test_world_builder_recovery_safety.py for the tables, and
    MAX_RECOVERY_KEYFRAMES for what would earn a larger value.
    """
    assert classical.MAX_RECOVERY_KEYFRAMES == 1


def test_one_unsolvable_keyframe_does_not_end_the_segment(recovery_enabled):
    """The regression. On the parent branch every pose after index 5 is
    `unavailable`; the chain latched on the noise frame and no solver ran
    again."""
    camera, frames = _walk(12)
    frames[5] = _noise_frame("kf5")

    steps, backend = _run(frames, camera)
    statuses = [step.pose.status for step in steps]

    assert statuses[0] == POSE_STATUS_ANCHOR
    assert statuses[5] == POSE_STATUS_UNAVAILABLE, (
        "the noise frame must still be refused -- recovery is not leniency"
    )
    recovered = [s for s in statuses[6:] if s == POSE_STATUS_SOLVED]
    assert len(recovered) >= 5, (
        f"expected the walk to keep solving after one bad keyframe, got "
        f"{statuses}"
    )
    assert not any(step.chain_broken for step in steps), (
        "one refusal is not a broken chain"
    )


def test_the_map_keeps_growing_across_a_refused_keyframe(recovery_enabled):
    """Recovery that produced no structure would be recovery in name only."""
    camera, frames = _walk(12)
    frames[5] = _noise_frame("kf5")

    steps, backend = _run(frames, camera)
    after = sum(
        len(step.new_points) for step in steps[6:] if step.new_points is not None
    )
    assert after > 0, "no landmarks were triangulated after the refusal"

    snapshot = backend.snapshot()
    assert snapshot.points is not None and len(snapshot.points) > 0


def test_a_sustained_run_of_failures_still_breaks_the_chain(recovery_enabled):
    """Honest refusal survives. Recovery is BOUNDED, and the bound is what
    keeps a genuinely untrackable stretch from being carried forward on a
    coordinate frame nothing supports."""
    camera, frames = _walk(4 + MECHANISM_BUDGET + 4)
    steps = []
    backend = ClassicalTwoViewBackend()
    backend.begin(_intrinsics(camera))

    for index, frame in enumerate(frames):
        if index >= 4:
            frame = _noise_frame(f"noise{index}", seed=index)
        steps.append(backend.extend(frame))

    assert sum(1 for step in steps if step.chain_broken) == 1, (
        "the break is an EDGE: exactly one keyframe reports it"
    )
    broke_at = next(i for i, step in enumerate(steps) if step.chain_broken)
    assert broke_at == 4 + recovery_enabled - 1, (
        f"the chain broke at {broke_at}; the bound says it should break on "
        f"the {recovery_enabled}th consecutive failure"
    )


def test_recovery_still_refuses_when_the_geometry_is_absent():
    """A walk of unrelated frames must produce no poses at all.

    The mutation this guards: if recovery were implemented by lowering an
    acceptance threshold rather than by retrying, this test would start
    passing poses.
    """
    camera, _ = _walk(4)
    frames = [_noise_frame(f"noise{i}", seed=100 + i) for i in range(8)]
    steps, _ = _run(frames, camera)

    solved = [s for s in steps if s.pose.status == POSE_STATUS_SOLVED]
    assert solved == [], (
        f"unrelated frames must not yield poses; got {len(solved)}"
    )


def test_the_seed_pair_retries_against_the_same_anchor(recovery_enabled):
    """A failed seed pair must not throw the anchor away.

    Restarting the segment resets the baseline to zero, so a pair refused
    for want of parallax is refused again on the next attempt, and again.
    Holding the anchor and waiting means the next attempt has a WIDER
    baseline than the one that failed -- which is the direction that fixes
    a parallax refusal.
    """
    camera, frames = _walk(10)
    frames[1] = _noise_frame("kf1")

    steps, _ = _run(frames, camera)
    assert steps[0].pose.status == POSE_STATUS_ANCHOR
    assert steps[1].pose.status == POSE_STATUS_UNAVAILABLE
    assert steps[2].pose.status == POSE_STATUS_SOLVED, (
        "keyframe 2 should have seeded against the retained anchor"
    )
    assert not any(step.chain_broken for step in steps)


def test_thresholds_are_unchanged():
    """The acceptance bar is a fact about this branch, pinned so that a
    later 'improvement' cannot quietly buy coherence with leniency."""
    from tower.world_builder import geometry

    assert classical.MIN_PNP_CORRESPONDENCES == 12
    assert classical.PNP_REPROJECTION_ERROR_PX == 3.0
    assert geometry.MIN_INLIERS == 15
    assert geometry.MIN_INLIER_RATIO == 0.05
    assert geometry.MIN_TRIANGULATION_ANGLE_DEG == 0.5
    assert geometry.RANSAC_THRESHOLD_PX == 1.0
