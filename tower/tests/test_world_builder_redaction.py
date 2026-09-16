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


# -- the false positives ------------------------------------------------
#
# The claim at the top of `redaction.py` -- "0 false positives on 40 face-free
# frames" -- was measured on SYNTHETIC room renders, and it does not survive
# contact with a real capture. Re-measured on the canonical world's 398 raw
# source frames: 240 detections at 0.30, of which TWENTY are a face. 81 are the
# wearer's own hand, 135 are a printed cup logo, a lit PC case, a wall, a
# carpet, a doorway. 12.6% of every pixel in the capture was filled, 38 frames
# lost more than half of themselves, and the worst lost 93.8%.
#
# Every number in the cases below is a REAL detection: the box and the five
# landmarks YuNet returned, in that frame, at that score. They are written out
# rather than recomputed so the test needs no capture corpus and still
# measures the thing that went wrong.
#
# Evidence and method:
#   Glasses-scratch/wb-final-recon/redaction/REDACTION.md

# (box, landmarks, score, what it actually is)
HANDS = [
    ((5.9, 405.1, 101.4, 134.9),
     [69.2, 455.0, 87.9, 454.0, 96.3, 473.9, 72.6, 503.3, 84.1, 501.8],
     0.82, "keyframe 362, a hand on a laptop keyboard"),
    ((21.4, 409.6, 103.0, 138.3),
     [90.5, 459.2, 100.7, 462.9, 109.7, 481.1, 86.4, 507.8, 91.4, 510.5],
     0.80, "keyframe 361, the same hand a frame later"),
    ((86.7, 426.7, 89.1, 166.5),
     [163.0, 500.6, 158.3, 502.2, 181.2, 523.9, 161.2, 550.1, 151.7, 550.4],
     0.77, "keyframe 324, a hand holding a phone"),
]

SCENE = [
    ((15.9, 397.9, 18.0, 25.4),
     [27.5, 408.0, 30.4, 407.3, 33.0, 411.8, 29.3, 418.1, 31.0, 416.7],
     0.56, "keyframe 131, a lit PC case on carpet"),
    ((0.3, 353.9, 49.2, 111.6),
     [-2.3, 393.2, 8.5, 393.5, -4.6, 420.0, -0.7, 439.0, 9.4, 438.0],
     0.32, "keyframe 359, the wearer's leg and a desk edge"),
]

# The twenty true positives in that capture are one object: a printed portrait
# of a bearded man hanging on the bedroom wall. It is a real, identifiable
# human face and the change must not cost a single one of them.
PORTRAIT = [
    ((49.1, 303.3, 56.5, 67.9),
     [68.5, 332.0, 92.3, 332.5, 80.9, 345.3, 69.8, 353.0, 91.1, 353.3],
     0.78, "keyframe 87, the portrait close up"),
    ((69.5, 233.2, 30.5, 40.4),
     [77.4, 251.2, 87.3, 252.2, 78.9, 259.2, 76.8, 264.4, 82.9, 265.6],
     0.65, "keyframe 85, the portrait at 30 px"),
    ((150.0, 16.9, 46.9, 54.0),
     [169.7, 37.8, 182.5, 37.9, 178.3, 49.0, 170.1, 57.7, 179.1, 56.6],
     0.75, "keyframe 153, the portrait across the room"),
    ((96.2, 24.4, 83.7, 101.1),
     [132.5, 75.7, 159.0, 78.7, 149.8, 97.0, 128.7, 103.3, 151.8, 106.3],
     0.43, "keyframe 119, the portrait at an angle"),
]

# LIVE faces, from the wider capture corpus: the canonical world has none, but
# 53 841 frames across 104 capture directories do -- a face in a bathroom
# mirror, and a group photograph displayed on a monitor. These are the cases
# that decide whether the gate is safe.
LIVE_FACES = [
    ((191.8, 262.8, 153.6, 215.3),
     [228.0, 345.1, 300.5, 344.5, 261.5, 388.6, 233.7, 420.2, 297.3, 419.9],
     0.92, "capture 0fc400, a face in a mirror, 14% of the frame"),
    ((282.1, 222.5, 29.8, 35.1),
     [288.6, 236.0, 301.5, 236.3, 293.6, 242.9, 289.5, 249.7, 299.5, 248.6],
     0.92, "capture ab10cb, the same wearer further from the mirror"),
    ((177.0, 379.3, 26.2, 34.0),
     [184.9, 388.7, 196.4, 388.7, 191.3, 395.4, 185.8, 402.0, 195.7, 402.6],
     0.91, "capture a0845f, one face in a group photo on a monitor"),
    ((191.1, 305.6, 18.1, 22.6),
     [197.7, 314.1, 206.2, 314.1, 203.1, 318.9, 198.2, 322.1, 205.1, 322.4],
     0.91, "capture ece060, an 18 px face in the same photograph"),
]


def test_a_hand_is_not_a_face():
    """The single largest class of false positive, and it scores HIGH.

    Raising the confidence threshold cannot reach these -- 0.82, 0.80 and 0.77
    are above all but three of the twenty true positives. The landmarks are
    what give them away: a "nose" 0.9 to 2.5 eye-separations off the midline,
    against 0.03-0.27 for a real face.
    """
    from tower.world_builder.redaction import landmarks_are_facelike

    for box, landmarks, score, what in HANDS:
        assert not landmarks_are_facelike(box, landmarks), (
            f"{what} (score {score:.2f}) was accepted as a face"
        )


def test_scene_texture_is_not_a_face():
    from tower.world_builder.redaction import landmarks_are_facelike

    for box, landmarks, score, what in SCENE:
        assert not landmarks_are_facelike(box, landmarks), (
            f"{what} (score {score:.2f}) was accepted as a face"
        )


def test_every_true_positive_in_the_capture_still_passes():
    """The constraint that dominates: no recall may be traded for precision.

    Four of these are the printed portrait -- at 30 px, at 47 px, at 56 px and
    at an angle -- and four are live faces from the wider corpus. If a change
    to the bands makes any of them fail, the change is wrong however much it
    improves the pixel count.
    """
    from tower.world_builder.redaction import (
        MAX_BOX_AREA_FRACTION,
        box_area_fraction,
        landmarks_are_facelike,
    )

    for box, landmarks, score, what in PORTRAIT + LIVE_FACES:
        assert landmarks_are_facelike(box, landmarks), f"{what} was rejected"
        assert box_area_fraction(box, (640, 360)) <= MAX_BOX_AREA_FRACTION, (
            f"{what} was rejected as too large"
        )


def test_the_geometry_is_free_of_head_roll():
    """A tilted head must score exactly as an upright one.

    The vertical measures are taken along the eye line's own perpendicular
    rather than the image's y-axis, so this is a property of the maths and not
    a happy accident. The detector is documented to hold through 45 degrees of
    tilt; a plausibility test that did not would give that back.
    """
    import math

    from tower.world_builder.redaction import (
        landmark_geometry,
        landmarks_are_facelike,
    )

    box, landmarks, _score, _what = PORTRAIT[0]
    upright = landmark_geometry(box, landmarks)
    cx = box[0] + box[2] / 2.0
    cy = box[1] + box[3] / 2.0
    for degrees in (-45, -20, 20, 45):
        a = math.radians(degrees)
        rotated = []
        for i in range(0, len(landmarks), 2):
            dx, dy = landmarks[i] - cx, landmarks[i + 1] - cy
            rotated += [cx + dx * math.cos(a) - dy * math.sin(a),
                        cy + dx * math.sin(a) + dy * math.cos(a)]
        turned = landmark_geometry(box, rotated)
        for key, value in upright.items():
            assert abs(turned[key] - value) < 1e-6, (
                f"{key} moved by {turned[key] - value:.4f} at {degrees} degrees"
            )
        assert landmarks_are_facelike(box, rotated)


def test_a_detection_that_covers_the_frame_is_not_a_face():
    """The measured worst case: 47.8% of the frame in one raw box, which after
    the head dilation blacked out 93.8% of a frame whose content is a wall, a
    PC tower and carpet.

    The bound is not tight. The largest raw box over 840 composited faces is
    30.1% of the frame and the 95th percentile is 19.9% -- a face about 20 cm
    from the lens -- so the cap sits above every face this was measured on.
    """
    from tower.world_builder.redaction import (
        MAX_BOX_AREA_FRACTION,
        box_area_fraction,
    )

    worst = (3.0, -64.1, 318.8, 345.2)          # keyframe 51, score 0.32
    assert box_area_fraction(worst, (640, 360)) > MAX_BOX_AREA_FRACTION

    arms_length = (100.0, 150.0, 200.0, 256.0)  # 22% of the frame
    assert box_area_fraction(arms_length, (640, 360)) <= MAX_BOX_AREA_FRACTION


def test_degenerate_landmarks_are_rejected_not_crashed_on():
    from tower.world_builder.redaction import (
        landmark_geometry,
        landmarks_are_facelike,
    )

    box = (10.0, 10.0, 40.0, 50.0)
    both_eyes_at_one_point = [20.0, 20.0, 20.0, 20.0, 25.0, 30.0,
                              18.0, 40.0, 26.0, 40.0]
    assert landmark_geometry(box, both_eyes_at_one_point) is None
    assert not landmarks_are_facelike(box, both_eyes_at_one_point)
    assert not landmarks_are_facelike((0.0, 0.0, 0.0, 0.0), [0.0] * 10)


def test_a_large_detection_must_survive_a_second_look_at_native_scale():
    """Where the destroyed pixels are: big boxes.

    A detection covering more than 5% of the frame is re-sought at UPSCALE 1.
    A real face is an object and survives resampling -- 88.9% of composited
    faces do, and every one at or above 95 px wide. A coincidence of a few
    pixels usually does not: only 23.7% of the scene false positives came back.
    """
    redactor = FaceRedactor()
    frame = _room()
    big = (40.0, 40.0, 200.0, 256.0)            # 22% of a 640x360 frame
    landmarks = [100.0, 130.0, 180.0, 130.0, 140.0, 175.0,
                 105.0, 215.0, 175.0, 215.0]

    def _only_at_upscale_two(image, upscale):
        return [(big, landmarks)] if upscale != 1 else []

    redactor._raw_detect = _only_at_upscale_two
    assert redactor._detect(frame) == [], (
        "a large detection that only exists at one scale was still filled"
    )

    def _at_both(image, upscale):
        return [(big, landmarks)]

    redactor._raw_detect = _at_both
    assert len(redactor._detect(frame)) == 1


def test_a_small_detection_is_exempt_from_the_second_look():
    """Recall on a DISTANT face is the thing the second look would cost.

    At UPSCALE 1 only 20% of 24-px composited faces and 80% of 35-px ones come
    back -- which is precisely the range UPSCALE 2 exists to find. So the test
    applies only where the detection is too big to be a distant bystander.
    """
    redactor = FaceRedactor()
    frame = _room()
    small = (100.0, 100.0, 40.0, 50.0)          # 0.9% of the frame
    landmarks = [111.0, 118.0, 129.0, 118.0, 120.0, 130.0,
                 112.0, 140.0, 128.0, 140.0]

    def _only_at_upscale_two(image, upscale):
        return [(small, landmarks)] if upscale != 1 else []

    redactor._raw_detect = _only_at_upscale_two
    assert len(redactor._detect(frame)) == 1, (
        "a small face was dropped for failing a test it is exempt from"
    )


def test_the_second_look_failing_keeps_the_detection():
    """FAILS TOWARDS FILLING.

    An unnecessary fill costs pixels. A skipped one puts a face on disk. Those
    are not the same kind of mistake, so a corroboration pass that cannot run
    must not be read as a refusal.
    """
    redactor = FaceRedactor()
    frame = _room()
    big = (40.0, 40.0, 200.0, 256.0)
    landmarks = [100.0, 130.0, 180.0, 130.0, 140.0, 175.0,
                 105.0, 215.0, 175.0, 215.0]

    def _explodes_at_native(image, upscale):
        if upscale == 1:
            raise RuntimeError("the second pass fell over")
        return [(big, landmarks)]

    redactor._raw_detect = _explodes_at_native
    assert len(redactor._detect(frame)) == 1


def test_the_label_records_that_the_gate_ran():
    """A reader who saw only `yunet-2023mar@0.30` would look up that detector's
    recall and over-estimate this imagery. The suffix says a filter ran after
    it, and which one.
    """
    from tower.world_builder.redaction import PLAUSIBILITY_ID

    label = FaceRedactor().label
    assert label.startswith("faces-detected-and-filled/")
    assert label.endswith(f"+{PLAUSIBILITY_ID}")
    for outcome_claim in ("anonymised", "anonymized", "privacy-safe", "removed"):
        assert outcome_claim not in label


def test_a_small_detection_with_unfacelike_landmarks_is_still_filled():
    """The landmark test is not applied to small boxes, and must not be.

    Its bands were set from frontal faces; a distant bystander in three-quarter
    view is the face they describe worst, and every such face is small. Hands,
    which the test exists for, are never under 2.68% of the frame. So a small
    box keeps its fill however its landmarks lie -- and the same landmarks on a
    large box are still refused, which is what proves the exemption is by size
    and not a test that stopped running.
    """
    from tower.world_builder.redaction import (
        LANDMARK_TEST_ABOVE_AREA,
        landmarks_are_facelike,
    )

    # Keyframe 324's hand (nose 2.5 eye-separations off the midline), shrunk
    # about its own box to a distant-face size, 0.9% of a 640x360 frame.
    (hx, hy, hw, hh), hand, _score, _what = HANDS[2]

    def _scaled(factor):
        box = (100.0, 100.0, hw * factor, hh * factor)
        lm = []
        for i in range(0, len(hand), 2):
            lm += [box[0] + (hand[i] - hx) * factor,
                   box[1] + (hand[i + 1] - hy) * factor]
        return box, lm

    small_box, small_lm = _scaled(0.4)
    assert small_box[2] * small_box[3] / (640 * 360) < LANDMARK_TEST_ABOVE_AREA
    assert not landmarks_are_facelike(small_box, small_lm), (
        "precondition: these landmarks really do fail the test"
    )

    redactor = FaceRedactor()
    frame = _room()
    redactor._raw_detect = lambda image, upscale: [(small_box, small_lm)]
    assert len(redactor._detect(frame)) == 1, (
        "a small detection was left unfilled because its landmarks were odd"
    )

    large_box, large_lm = _scaled(1.0)
    assert large_box[2] * large_box[3] / (640 * 360) >= LANDMARK_TEST_ABOVE_AREA
    redactor._raw_detect = lambda image, upscale: [(large_box, large_lm)]
    assert redactor._detect(frame) == [], "the landmark test stopped running"
