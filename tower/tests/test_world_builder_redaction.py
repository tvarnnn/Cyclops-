"""Face redaction before persistence: does it work, and what does it cost?

Uses REAL face imagery rather than synthetic blobs. `scikit-image` (here
as an easyocr dependency) ships `astronaut.png` and an LFW subset of 100
distinct faces, so a detector can be measured against faces rather than
against something face-shaped.

SYNTHETIC SCENES, REAL FACES. The rooms are rendered; the faces are
photographs of people. Nothing here says anything about the Ray-Ban
camera's own optics.
"""

import numpy as np
import pytest

from tower.world_builder.redaction import (
    HEAD_DILATION,
    REDACTION_NONE,
    FaceRedactor,
    model_path,
)

cv2 = pytest.importorskip("cv2")
skimage_data = pytest.importorskip("skimage.data")

pytestmark = pytest.mark.skipif(
    model_path() is None,
    reason="no face-detection model is vendored on this host",
)

WIDTH, HEIGHT = 640, 360


def _encode(image) -> bytes:
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    assert ok
    return buffer.tobytes()


def _room() -> np.ndarray:
    """A textured backdrop, so the frame is not a flat field."""
    from tests import synthetic_scene as ss

    matrix = ss.camera_matrix(WIDTH, HEIGHT)
    images = ss.render_sequence(
        ss.furnished_room(), ss.strafe(1, step=0.09), matrix, WIDTH, HEIGHT
    )
    return images[0].copy()


def _face_patch(size: int) -> np.ndarray:
    face = skimage_data.astronaut()[20:220, 150:350]
    patch = cv2.resize(face, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)


def _frame_with_face(size: int = 90, at=(200, 80)):
    frame = _room()
    patch = _face_patch(size)
    x, y = at
    frame[y : y + size, x : x + size] = patch
    return frame, (x, y, size)


def test_a_real_face_is_filled(tmp_path):
    frame, (x, y, size) = _frame_with_face()
    result = FaceRedactor().redact(_encode(frame))

    assert result.applied
    assert result.regions >= 1

    out = cv2.imdecode(np.frombuffer(result.image_bytes, np.uint8), cv2.IMREAD_COLOR)
    centre = out[y + size // 2, x + size // 2]
    assert int(centre.max()) == 0, "the middle of the face was not filled"


def test_the_fill_covers_more_than_the_face_box():
    """A face box is not a head. Hair, ears and jaw are outside it."""
    frame, (x, y, size) = _frame_with_face(size=120, at=(220, 100))
    out_bytes = FaceRedactor().redact(_encode(frame)).image_bytes
    out = cv2.imdecode(np.frombuffer(out_bytes, np.uint8), cv2.IMREAD_COLOR)

    filled = (out.max(axis=2) == 0)
    assert filled.sum() > (size * size) * 0.5, "implausibly little was filled"
    # Dilation is bounded: a redactor that filled the frame would "pass"
    # every privacy test and destroy the reconstruction.
    assert filled.sum() < filled.size * 0.35, "the fill is far too large"


def test_a_frame_with_no_face_is_left_alone():
    """False positives cost geometry for nothing."""
    frame = _room()
    original = _encode(frame)
    result = FaceRedactor().redact(original)

    assert result.applied
    assert result.regions == 0
    assert result.image_bytes == original, (
        "an untouched frame must be persisted byte-identically, not re-encoded"
    )


def test_many_distinct_real_faces_are_detected():
    """One face is an anecdote. The LFW subset is 100 different people."""
    lfw = skimage_data.lfw_subset()
    redactor = FaceRedactor()
    hits = 0
    total = 0
    for sample in lfw[:40]:
        total += 1
        face = (sample * 255).astype(np.uint8)
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2BGR)
        frame = _room()
        patch = cv2.resize(face, (110, 110), interpolation=cv2.INTER_CUBIC)
        frame[90:200, 240:350] = patch
        if redactor.redact(_encode(frame)).regions >= 1:
            hits += 1

    assert hits / total >= 0.85, f"only {hits}/{total} real faces were detected"


# -- honesty ------------------------------------------------------------


def test_the_label_names_the_detector_and_threshold():
    label = FaceRedactor().label
    assert label.startswith("faces-detected-and-filled/")
    assert "@" in label
    for outcome_claim in ("anonymised", "anonymized", "privacy-safe", "removed"):
        assert outcome_claim not in label


def test_an_absent_model_is_reported_not_silently_skipped(tmp_path):
    redactor = FaceRedactor(path=tmp_path / "nothing.onnx")
    assert not redactor.available
    assert redactor.label == REDACTION_NONE

    payload = _encode(_room())
    result = redactor.redact(payload)
    assert result.image_bytes == payload
    assert result.label == REDACTION_NONE
    assert result.applied is False
    assert "no face-detection model" in result.unavailable_reason


def test_an_undecodable_image_is_persisted_unchanged_not_lost():
    """A redactor that raised would trade privacy for DATA LOSS.

    The keyframe still has to be persisted, and the session has to be able
    to say honestly that nothing was applied to it.
    """
    result = FaceRedactor().redact(b"not a jpeg at all")
    assert result.image_bytes == b"not a jpeg at all"
    assert result.label == REDACTION_NONE
    assert result.applied is False
    assert "ValueError" in result.unavailable_reason


def test_a_detector_that_explodes_does_not_stop_persistence(monkeypatch):
    redactor = FaceRedactor()

    def _explode(*args, **kwargs):
        raise RuntimeError("the detector fell over")

    monkeypatch.setattr(redactor, "_detect", _explode)
    payload = _encode(_room())
    result = redactor.redact(payload)

    assert result.image_bytes == payload
    assert result.label == REDACTION_NONE
    assert "RuntimeError" in result.unavailable_reason


def test_a_resolution_change_mid_session_is_handled():
    """DAT's adaptive ladder changes resolution mid-stream."""
    redactor = FaceRedactor()
    small, _ = _frame_with_face(size=90, at=(200, 80))
    assert redactor.redact(_encode(small)).applied

    large = cv2.resize(small, (1280, 720), interpolation=cv2.INTER_CUBIC)
    assert redactor.redact(_encode(large)).applied


# -- the cost to the reconstruction ------------------------------------


def test_redaction_does_not_cost_keyframes_or_poses(tmp_path):
    """The objection that redaction damages the geometry, measured.

    A real face box is a few percent of the frame. At that scale keyframe
    acceptance and pose solving were completely insensitive across ten
    scene seeds; only feature density and the point count move.

    This runs the FULL pipeline both ways over identical frames -- the
    only difference is whether the persisted pixels were filled.
    """
    from tests import synthetic_scene as ss
    from tower.world_builder.engine import WorldBuilderEngine
    from tower.world_builder.records import CameraIntrinsics
    from tower.world_builder.store import WorldStore

    matrix = ss.camera_matrix(WIDTH, HEIGHT)
    intrinsics = CameraIntrinsics(
        source="self_calibrated",
        model="pinhole",
        fx=float(matrix[0, 0]),
        fy=float(matrix[1, 1]),
        cx=float(matrix[0, 2]),
        cy=float(matrix[1, 2]),
        calibrated_width=WIDTH,
        calibrated_height=HEIGHT,
    )
    frames = ss.render_sequence(
        ss.furnished_room(), ss.strafe(12, step=0.09), matrix, WIDTH, HEIGHT
    )
    patch = _face_patch(80)
    payloads = []
    for image in frames:
        frame = image.copy()
        frame[60:140, 180:260] = patch
        payloads.append(_encode(frame))

    def _run(root, factory):
        engine = WorldBuilderEngine(
            WorldStore(root), redactor_factory=factory
        )
        world_id = engine.create_world("Cost")
        session_id = engine.start_session(
            world_id,
            intrinsics=intrinsics,
            frame_source="synthetic",
            declared_size=(WIDTH, HEIGHT),
        )
        for index, payload in enumerate(payloads):
            engine.observe(payload, source_seq=index)
        summary = engine.stop_session()
        result = engine.build(world_id, session_id)
        return summary, result

    plain_summary, plain = _run(
        tmp_path / "plain", lambda: FaceRedactor(path=tmp_path / "absent.onnx")
    )
    redacted_summary, redacted = _run(tmp_path / "redacted", FaceRedactor)

    assert redacted_summary.keyframes_accepted == plain_summary.keyframes_accepted, (
        "redaction changed which frames became keyframes"
    )
    assert redacted.poses_solved == plain.poses_solved, (
        "redaction cost a pose solve"
    )
    assert redacted.segments == plain.segments
    # Points may thin; they must not collapse.
    assert redacted.points >= plain.points * 0.5, (
        f"the point cloud collapsed: {plain.points} -> {redacted.points}"
    )


def test_the_persisted_image_is_the_redacted_one(tmp_path):
    """What is on disk is what was filled -- not the original bytes."""
    from tests import synthetic_scene as ss
    from tower.world_builder.engine import WorldBuilderEngine
    from tower.world_builder.store import WorldStore

    matrix = ss.camera_matrix(WIDTH, HEIGHT)
    frames = ss.render_sequence(
        ss.furnished_room(), ss.strafe(6, step=0.09), matrix, WIDTH, HEIGHT
    )
    patch = _face_patch(90)
    store = WorldStore(tmp_path)
    engine = WorldBuilderEngine(store)
    world_id = engine.create_world("Persisted")
    session_id = engine.start_session(world_id, frame_source="synthetic")
    for index, image in enumerate(frames):
        frame = image.copy()
        frame[70:160, 200:290] = patch
        engine.observe(_encode(frame), source_seq=index)
    engine.stop_session()

    images = sorted(store.images_dir(world_id, session_id).glob("*.jpg"))
    assert images, "precondition: keyframes were persisted"
    for path in images:
        stored = cv2.imdecode(
            np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR
        )
        region = stored[70:160, 200:290]
        assert int(region.min()) == 0, f"{path.name} kept an unfilled face"


def test_the_model_is_found_from_any_working_directory(tmp_path, monkeypatch):
    """It used to be `Path("models")/...`, resolved against the cwd. A Tower or
    a script started from anywhere but `tower/` therefore found no model, and
    the consequence is not an error: `engine.py` logs a warning and persists
    the keyframe UNREDACTED. A privacy transformation was silently conditional
    on where a caller happened to be standing.

    Found when a batch script run from a scratch directory refused 429 of 429
    frames -- refusing is the safe direction, and only the dense stage's own
    outcome check made it visible at all.
    """
    from tower.world_builder.redaction import FaceRedactor, model_path

    monkeypatch.delenv("TOWER_FACE_REDACTION_MODEL", raising=False)
    monkeypatch.chdir(tmp_path)              # no models/ here
    found = model_path()
    assert found is not None, "the vendored model must be found from any cwd"
    assert found.is_absolute()
    assert found.exists()
    assert FaceRedactor().available is True


def test_the_environment_override_still_wins_and_a_missing_one_is_absent(tmp_path, monkeypatch):
    """The override is how a deployment points at its own weights, and an
    override naming a file that is not there must be reported as absent rather
    than silently falling back to a different model."""
    from tower.world_builder.redaction import model_path

    monkeypatch.setenv("TOWER_FACE_REDACTION_MODEL", str(tmp_path / "nope.onnx"))
    assert model_path() is None

    real = tmp_path / "mine.onnx"
    real.write_bytes(b"not really a model")
    monkeypatch.setenv("TOWER_FACE_REDACTION_MODEL", str(real))
    assert model_path() == real
