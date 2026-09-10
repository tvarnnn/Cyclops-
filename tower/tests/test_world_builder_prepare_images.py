"""Undistorting a session's keyframes into the solve workspace.

`prepare_images` had NO TESTS at all, and an adversarial review found two
defects in it that a test would have caught, both silent.

1.  On a calibration change it does `shutil.rmtree(ignore_errors=True)` and
    then skips any frame whose target still exists. On Windows a file a
    reader holds open cannot be unlinked, and `ignore_errors` turns that
    into silence -- so the tree survives, the skip fires, and COLMAP is
    handed a MIX OF TWO CALIBRATIONS under one `camera.json`. The review
    demonstrated one image from calibration A and three from B, with no
    exception and no warning.

2.  That same `rmtree` deletes `sources.json` three lines before
    `read_sources` reads it. `sources.json` maps each keyframe to the RAW
    capture frame it came from, and the builder that observed those frames
    is the only thing that knows it. Losing it silently demotes every
    subsequent solve to the face-redacted session copies -- the
    experiment ledger's measured 337 images down to 307.
"""

import json

import cv2
import numpy as np
import pytest

from tower.world_builder import global_solve
from tower.world_builder.global_solve import PinholeCamera, prepare_images
from tower.world_builder.records import Keyframe

WIDTH, HEIGHT = 64, 48


def _keyframe(session_id: str, i: int) -> Keyframe:
    return Keyframe(
        keyframe_id=f"{session_id}:{i:08d}", session_id=session_id, source_seq=i,
        received_at=float(i), image_relpath=f"images/{i:08d}.jpg",
        width=WIDTH, height=HEIGHT, byte_count=1, segment_index=0,
    )


@pytest.fixture
def session(tmp_path):
    """A real world and session with four distinguishable keyframe images."""
    from dataclasses import replace

    from tower.world_builder.engine import WorldBuilderEngine
    from tower.world_builder.store import WorldStore

    store = WorldStore(tmp_path / "worlds")
    engine = WorldBuilderEngine(store)
    world_id = engine.create_world()
    session_id = engine.start_session(
        world_id, frame_source="synthetic", intrinsics=_intrinsics(50.0, 0.12)
    )
    rng = np.random.default_rng(0)
    keyframes = []
    for i in range(4):
        frame = rng.integers(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8)
        ok, buf = cv2.imencode(".jpg", frame)
        assert ok
        store.write_keyframe_image(world_id, session_id, f"{i:08d}.jpg", buf.tobytes())
        keyframe = _keyframe(session_id, i)
        store.append_keyframe(world_id, keyframe)
        keyframes.append(keyframe)
    engine.stop_session("stopped")
    workspace = global_solve.workspace_for(store, world_id, session_id)
    return store, world_id, session_id, keyframes, workspace


def _intrinsics(fx: float, k1: float = 0.0):
    """A calibration. `k1` must differ between the two the recalibration
    test uses: with zero distortion the undistort map is the identity for
    ANY focal length, so the output bytes would match and the test would
    pass without proving anything."""
    from tower.world_builder.records import CameraIntrinsics

    return CameraIntrinsics(
        source="self_calibrated", model="pinhole_radtan",
        fx=fx, fy=fx, cx=WIDTH / 2, cy=HEIGHT / 2,
        dist_coeffs=(k1, 0.0, 0.0, 0.0, 0.0),
        calibrated_width=WIDTH, calibrated_height=HEIGHT,
    )


def _recalibrate(store, world_id, session_id, fx, k1=0.0):
    from dataclasses import replace

    session = store.read_session(world_id, session_id)
    store.write_session(replace(session, intrinsics=_intrinsics(fx, k1)))


def _prepare(store, world_id, session_id, keyframes):
    return prepare_images(store, world_id, session_id, keyframes)


def test_it_undistorts_every_keyframe_once(session):
    store, world_id, session_id, keyframes, workspace = session
    _camera, written = _prepare(store, world_id, session_id, keyframes)
    assert written == 4
    assert len(list(workspace.images_dir.glob("*.jpg"))) == 4
    # A second call with the SAME calibration is a no-op: that skip is the
    # incrementality the live path depends on.
    _camera, again = _prepare(store, world_id, session_id, keyframes)
    assert again == 0


def test_a_recalibration_re_undistorts_a_frame_the_rmtree_could_not_delete(
    session, monkeypatch
):
    """The defect, staged deterministically.

    `shutil.rmtree(ignore_errors=True)` is a NO-OP on a tree containing a
    file Windows will not unlink -- a file any reader holds open -- and
    `ignore_errors` is what turns the refusal into silence. So the tree
    survives, and before the fix the loop's `if target.exists(): continue`
    skipped every surviving frame and COLMAP was handed a MIX OF TWO
    CALIBRATIONS under one `camera.json`.

    The rmtree is stubbed rather than raced: what is under test is what
    happens WHEN the delete does not happen, and reproducing that by timing
    a real file handle makes the test a coin flip rather than a check.
    """
    store, world_id, session_id, keyframes, workspace = session
    _prepare(store, world_id, session_id, keyframes)
    survivor = workspace.images_dir / "00000001.jpg"
    before = survivor.read_bytes()

    _recalibrate(store, world_id, session_id, 80.0, -0.25)
    monkeypatch.setattr(global_solve.shutil, "rmtree", lambda *a, **k: None)
    _camera, written = _prepare(store, world_id, session_id, keyframes)

    assert survivor.exists(), "the fixture did not reproduce the surviving file"
    assert survivor.read_bytes() != before, (
        "a frame that survived the delete kept its OLD calibration; COLMAP "
        "would be handed two calibrations under one camera.json"
    )
    assert written == 4, "not every frame was re-undistorted after a recalibration"


def test_the_skip_still_holds_when_the_calibration_did_not_change(session):
    """The fix must not have turned every solve into a full re-undistort.

    Skipping frames already in the workspace is the incrementality the live
    path depends on: the field run undistorted 120 of 646 images on its
    ninth solve.
    """
    store, world_id, session_id, keyframes, workspace = session
    _prepare(store, world_id, session_id, keyframes)
    stamps = {p.name: p.stat().st_mtime_ns for p in workspace.images_dir.glob("*.jpg")}
    _camera, written = _prepare(store, world_id, session_id, keyframes)
    assert written == 0
    assert {p.name: p.stat().st_mtime_ns
            for p in workspace.images_dir.glob("*.jpg")} == stamps


def test_a_recalibration_keeps_the_raw_frame_provenance(session):
    """`sources.json` is deleted by the rmtree three lines before it is read.

    Nothing can recreate it: it maps each keyframe to the raw capture frame
    the observing builder saw, and a replay stages frames under enumeration
    indices so the name alone does not find them.
    """
    store, world_id, session_id, keyframes, workspace = session
    _prepare(store, world_id, session_id, keyframes)
    session_dir = store.session_dir(world_id, session_id)
    raw = {k.keyframe_id: str(session_dir / "images" / f"{i:08d}.jpg")
           for i, k in enumerate(keyframes)}
    global_solve.write_sources_records(workspace, raw)
    assert global_solve.read_sources(workspace) == raw

    _recalibrate(store, world_id, session_id, 80.0, -0.25)
    _prepare(store, world_id, session_id, keyframes)

    assert global_solve.read_sources(workspace) == raw, (
        "a recalibration lost the raw-frame provenance; every later solve "
        "silently falls back to the redacted session copies"
    )


def test_the_camera_written_is_the_camera_asked_for(session):
    store, world_id, session_id, keyframes, workspace = session
    camera, _written = _prepare(store, world_id, session_id, keyframes)
    stored = PinholeCamera.from_json_dict(
        json.loads(workspace.camera_path.read_text(encoding="utf-8"))
    )
    assert stored == camera
