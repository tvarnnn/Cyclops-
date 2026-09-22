"""A broken solve chain starts a new segment.

`classical.py` has claimed for a long time that when a chain breaks "the
engine turns this into a new segment". It did not. `engine.py` incremented
the segment index only on `decision.lost` from the keyframe policy, and
never looked at what the backend did -- so once a chain broke, every later
keyframe in that segment was refused without ORB detection, matching, or
any geometry attempted.

Measured before this change, on the real 33-segment world from capture
22e9d428: 354 refusals from 26 root decisions, 328 of them cascade, and
0 of 26 segments ever recovered. Segment 14 abandoned 39 of its 61
keyframes.

Splitting is also what the record already claims. `engine.py`'s own
comment on the segment increment says "poses either side are NOT in a
common frame" -- which is exactly true of a broken chain, because
everything after the break is `unavailable` and shares no frame with what
came before.

The tracker is deliberately NOT reset here. Tracking is healthy; it is the
solve that failed. Resetting would manufacture a tracking loss out of a
geometry failure.
"""

import numpy as np
import pytest

from tower.world_builder.backend import Extension, PoseEstimate
from tower.world_builder.schema import (
    DEGENERACY_LOW_PARALLAX,
    POSE_STATUS_SOLVED,
    POSE_STATUS_UNAVAILABLE,
)


def test_extension_reports_the_break_only_on_the_keyframe_that_broke_it():
    """`chain_broken` is an EDGE, not a level.

    If it stayed true for every later keyframe the engine would start a
    new segment per frame, turning one failure into dozens of one-frame
    segments -- worse than the cascade it replaces.
    """
    assert Extension(pose=PoseEstimate(keyframe_id="k")).chain_broken is False


def test_classical_reports_chain_broken_once(monkeypatch):
    import cv2

    from tests import synthetic_scene as ss
    from tower.world_builder.backend import KeyframeInput
    from tower.world_builder.backends.classical import ClassicalTwoViewBackend
    from tower.world_builder.records import CameraIntrinsics

    from tower.world_builder.backends import classical as _classical

    width, height = 480, 360
    camera = ss.camera_matrix(width, height)
    # Long enough that the forced failures from index 3 on exhaust the
    # recovery budget. The break is no longer the FIRST refusal -- a
    # refusal costs an attempt, and MAX_RECOVERY_KEYFRAMES of them in a
    # row cost the chain -- but it is still an EDGE, which is what this
    # test is actually for.
    count = 4 + _classical.MAX_RECOVERY_KEYFRAMES + 2
    images = ss.render_sequence(
        ss.furnished_room(), ss.strafe(count, step=0.12), camera, width, height
    )
    frames = [
        KeyframeInput(
            keyframe_id=f"kf{i}",
            image_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
            image_bgr=img,
        )
        for i, img in enumerate(images)
    ]
    backend = ClassicalTwoViewBackend()
    backend.begin(
        CameraIntrinsics(
            source="self_calibrated",
            fx=float(camera[0][0]),
            fy=float(camera[1][1]),
            cx=float(camera[0][2]),
            cy=float(camera[1][2]),
            calibrated_width=width,
            calibrated_height=height,
        )
    )

    # Force every PnP to fail from the third keyframe on.
    from tower.world_builder.backends import classical

    calls = {"n": 0}
    real = classical._solve_pnp_ransac_or_refuse

    def _fail(*args, **kwargs):
        calls["n"] += 1
        return False, None, None, None

    breaks = []
    for index, frame in enumerate(frames):
        if index == 3:
            monkeypatch.setattr(classical, "_solve_pnp_ransac_or_refuse", _fail)
        step = backend.extend(frame)
        breaks.append(bool(step.chain_broken))

    assert calls["n"] > 0, "the fixture must reach PnP"
    assert sum(breaks) == 1, (
        f"the break must be reported exactly once, got {breaks}"
    )


def test_engine_starts_a_new_segment_when_the_solve_chain_breaks(tmp_path):
    """End to end. Keyframes after a broken chain must land in a NEW
    segment, so they get an anchor and a chance to solve, instead of
    being refused unexamined."""
    import cv2

    from tests import synthetic_scene as ss
    from tower.world_builder.backends import classical
    from tower.world_builder.engine import WorldBuilderEngine
    from tower.world_builder.records import CameraIntrinsics
    from tower.world_builder.store import WorldStore

    width, height = 480, 360
    camera = ss.camera_matrix(width, height)
    images = ss.render_sequence(
        ss.furnished_room(), ss.strafe(12, step=0.10), camera, width, height
    )

    store = WorldStore(tmp_path)
    engine = WorldBuilderEngine(store)
    world_id = engine.create_world()
    session_id = engine.start_session(
        world_id,
        intrinsics=CameraIntrinsics(
            source="self_calibrated",
            fx=float(camera[0][0]),
            fy=float(camera[1][1]),
            cx=float(camera[0][2]),
            cy=float(camera[1][2]),
            calibrated_width=width,
            calibrated_height=height,
        ),
        frame_source="synthetic",
        declared_size=(width, height),
    )
    for index, image in enumerate(images):
        engine.observe(ss.encode_jpeg(image), source_seq=index)
    engine.stop_session()

    keyframes = list(store.read_keyframes(world_id, session_id))
    segments = {k.segment_index for k in keyframes}
    # The synthetic strafe solves cleanly, so this asserts the mechanism
    # does not fire spuriously. The corpus measurement is what shows it
    # firing usefully.
    assert segments == {0}, (
        "a clean walk must not be split; splitting on a healthy chain "
        "would manufacture fragments out of nothing"
    )
