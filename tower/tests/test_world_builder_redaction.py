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
    from tower.world_builder.redaction import landmark_verdict

    for box, landmarks, score, what in PORTRAIT + LIVE_FACES:
        assert landmark_verdict(box, landmarks) is True, f"{what} was rejected"


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


# -- a gate that cannot judge must fill ---------------------------------
#
# Found by an adversarial review of the gate above. Each of the first three
# returned `[]` -- no fill -- before the fix.


def _box_of_area(fraction, width=640, height=360):
    import math

    side = math.sqrt(fraction * width * height)
    return (100.0, 20.0, side, side)


def _facelike_landmarks(box):
    x, y, w, h = box
    return [x + .3 * w, y + .4 * h, x + .7 * w, y + .4 * h, x + .5 * w, y + .6 * h,
            x + .35 * w, y + .8 * h, x + .65 * w, y + .8 * h]


def _hand_landmarks(box):
    """The keyframe-324 hand, mapped into `box`: measurably not a face."""
    (hx, hy, hw, hh), hand, _score, _what = HANDS[2]
    x, y, w, h = box
    out = []
    for i in range(0, len(hand), 2):
        out += [x + (hand[i] - hx) / hw * w, y + (hand[i + 1] - hy) / hh * h]
    return out


def _gate(detections_at_two, detections_at_one=None, *, raises_at=()):
    """Run `_detect` on scripted detector output.

    `detections_at_one` is what every OTHER scale returns -- native, 1/2, 1/4 --
    or a dict {scale: detections}; a scale missing from the dict finds nothing.
    `raises_at` lists scales whose pass throws.
    """
    from tower.world_builder.redaction import UPSCALE

    redactor = FaceRedactor()
    asked = []

    def _raw(image, upscale):
        asked.append(upscale)
        if upscale in raises_at:
            raise RuntimeError(f"pass at {upscale} fell over")
        if upscale == UPSCALE or detections_at_one is None:
            return list(detections_at_two)
        if isinstance(detections_at_one, dict):
            return list(detections_at_one.get(upscale, []))
        return list(detections_at_one)

    redactor._raw_detect = _raw
    boxes = redactor._detect(_room())
    _gate.asked = asked
    return boxes


def test_coincident_eye_landmarks_on_a_large_box_are_filled():
    """YuNet collapses both eyes onto one point on a near-profile face. The
    geometry is then undefined, which is no judgement at all -- and was being
    read as "not a face". Nothing corroborates it here either, on purpose.
    """
    from tower.world_builder.redaction import landmark_verdict

    box = _box_of_area(0.04)
    x, y, w, h = box
    eyes_coincide = [x + .5 * w, y + .4 * h, x + .5 * w, y + .4 * h,
                     x + .6 * w, y + .6 * h, x + .45 * w, y + .8 * h,
                     x + .6 * w, y + .8 * h]
    assert landmark_verdict(box, eyes_coincide) is None
    assert len(_gate([(box, eyes_coincide)], detections_at_one=[])) == 1


def test_non_finite_landmarks_on_a_large_box_are_filled():
    from tower.world_builder.redaction import landmark_verdict

    box = _box_of_area(0.04)
    nan = float("nan")
    good = _facelike_landmarks(box)
    for landmarks in ([nan] * 10, good[:4] + [nan] + good[5:], [float("inf")] * 10):
        assert landmark_verdict(box, landmarks) is None
        assert len(_gate([(box, landmarks)], detections_at_one=[])) == 1, (
            f"landmarks {landmarks} made the gate skip the fill"
        )


def test_a_box_that_is_not_a_number_fills_the_frame_rather_than_raising():
    """Raising here is not neutral: `redact` would persist the ORIGINAL bytes."""
    nan = float("nan")
    boxes = _gate([((nan, 10.0, 50.0, 60.0), [0.0] * 10)], detections_at_one=[])
    assert boxes == [(0.0, 0.0, 640.0, 360.0)]


# -- a box over a quarter of the frame ----------------------------------
#
# Landmark geometry carries no evidence at this size: all 40 false boxes over
# 25% on the canonical capture are "facelike". What does is being found again at
# another resolution -- native, 1/2 or 1/4 -- which a genuinely large face
# survives (65 of 65 composited close faces) and a wall rarely does (6 of 40).


def test_a_facelike_box_over_a_quarter_is_not_filled_on_its_landmarks_alone():
    """The rule that stored a plain wall and a PC tower fully black."""
    box = _box_of_area(0.30)
    assert _gate([(box, _facelike_landmarks(box))], detections_at_one=[]) == []


def test_a_box_over_a_quarter_is_filled_when_found_again_at_any_scale():
    box = _box_of_area(0.30)
    for scale in (1.0, 0.5, 0.25):
        for landmarks in (_facelike_landmarks(box), _hand_landmarks(box)):
            boxes = _gate([(box, landmarks)],
                          detections_at_one={scale: [(box, landmarks)]})
            assert len(boxes) == 1, (
                f"a large box found again at scale {scale} was not filled"
            )


def test_a_large_box_found_again_elsewhere_does_not_count():
    """Corroboration means the SAME place, not any detection at all."""
    box = _box_of_area(0.30)
    elsewhere = (500.0, 300.0, 40.0, 50.0)
    boxes = _gate([(box, _facelike_landmarks(box))],
                  detections_at_one={s: [(elsewhere, _facelike_landmarks(elsewhere))]
                                     for s in (1.0, 0.5, 0.25)})
    assert boxes == []


def test_a_large_box_is_filled_if_any_corroboration_pass_cannot_run():
    """FAILS TOWARDS FILLING, per scale: one broken pass is not a "no"."""
    box = _box_of_area(0.30)
    for scale in (1.0, 0.5, 0.25):
        boxes = _gate([(box, _facelike_landmarks(box))], detections_at_one={},
                      raises_at=(scale,))
        assert len(boxes) == 1, f"a pass failing at {scale} dropped the fill"


def test_unjudgeable_landmarks_fill_a_large_box_without_any_second_look():
    box = _box_of_area(0.30)
    nan = float("nan")
    boxes = _gate([(box, [nan] * 10)], detections_at_one={})
    assert len(boxes) == 1
    assert _gate.asked == [2], "an unjudgeable box was sent for corroboration"


def test_the_downscaled_pass_reports_boxes_in_original_pixels():
    """A pass at 1/4 must come back in the same coordinates as the pass at 2,
    or every IoU it is compared on is meaningless."""
    frame, (x, y, size) = _frame_with_face(size=150, at=(200, 60))
    redactor = FaceRedactor()
    full = redactor._raw_detect(frame, 2)
    quarter = redactor._raw_detect(frame, 0.25)
    assert full and quarter, "precondition: the face is found at both scales"
    from tower.world_builder.redaction import _iou

    best = max(_iou(a, b) for a, _ in full for b, _ in quarter)
    assert best >= 0.5, f"the 1/4 pass disagrees with the 2x pass (IoU {best:.2f})"


def test_a_real_face_covering_more_than_a_quarter_of_the_frame_is_filled():
    """End to end, real detector, the capture's own 360x640 portrait frame: a
    tightly cropped face ~20 cm from the lens, whose detector box is ~30% of
    the frame -- so it goes down the large-box branch, not the easy one."""
    from tower.world_builder.redaction import (
        LARGE_BOX_AREA_FRACTION,
        box_area_fraction,
    )

    face = skimage_data.astronaut()[40:160, 170:290]
    patch = cv2.cvtColor(
        cv2.resize(face, (300, 300), interpolation=cv2.INTER_CUBIC),
        cv2.COLOR_RGB2BGR,
    )
    frame = cv2.resize(_room(), (360, 640))
    frame[150:450, 5:305] = patch

    redactor = FaceRedactor()
    raw = redactor._raw_detect(frame, 2)
    assert any(box_area_fraction(b, frame.shape) > LARGE_BOX_AREA_FRACTION
               for b, _ in raw), "precondition: the detection is a large box"

    result = redactor.redact(_encode(frame))
    out = cv2.imdecode(np.frombuffer(result.image_bytes, np.uint8), cv2.IMREAD_COLOR)
    assert result.regions >= 1
    assert int(out[300, 155].max()) == 0, "the middle of a close face was not filled"


# -- a box over a quarter of the frame, cut by the frame edge (plausibility3) --
#
# Review 3, R1. A close face cut by the edge is what a detector sees worst at
# reduced resolution, so "found again at native, 1/2 or 1/4" left such faces on
# disk: 4 of 594 off-frame composites, 32 of a 3,232 held-out set. At the edge,
# facelike landmarks are enough again.


def test_a_facelike_large_box_at_the_frame_edge_is_filled_without_a_second_look():
    from tower.world_builder.redaction import EDGE_MARGIN, box_near_frame_edge

    for box in (
        (-40.0, 20.0, 263.0, 263.0),                    # crosses the left edge
        (0.0, 20.0, 263.0, 263.0),                      # touches it
        (100.0, 360.0 - 263.0 - 17.0, 263.0, 263.0),    # within 5% of the bottom
    ):
        assert box_near_frame_edge(box, (360, 640)), box
        boxes = _gate([(box, _facelike_landmarks(box))], detections_at_one={})
        assert len(boxes) == 1, f"a close face cut by the edge was left unfilled: {box}"
    assert EDGE_MARGIN == 0.05


def test_an_interior_large_box_still_needs_a_second_look():
    """The edge rule must not quietly become plausibility1 everywhere: the
    wall-and-PC-tower box was interior."""
    from tower.world_builder.redaction import box_near_frame_edge

    box = (100.0, 40.0, 263.0, 263.0)     # 40 px from the top: outside 5% (18 px)
    assert not box_near_frame_edge(box, (360, 640))
    assert _gate([(box, _facelike_landmarks(box))], detections_at_one={}) == []


def test_a_large_box_at_the_edge_with_unfacelike_landmarks_still_needs_a_second_look():
    box = (0.0, 20.0, 263.0, 263.0)
    assert _gate([(box, _hand_landmarks(box))], detections_at_one={}) == []
    assert len(_gate([(box, _hand_landmarks(box))],
                     detections_at_one={0.5: [(box, _hand_landmarks(box))]})) == 1


def test_a_real_close_face_cut_by_the_frame_edge_is_filled():
    """End to end, real detector, a 360x640 portrait frame: a face 420 px wide
    whose top half is above the frame, as someone leaning into the wearer is.
    YuNet finds it at 2x (a box crossing the top edge, ~35% of the frame) and at
    no reduced scale, so plausibility2 filled none of it."""
    from tower.world_builder.redaction import (
        LARGE_BOX_AREA_FRACTION,
        LARGE_BOX_CORROBORATION_SCALES,
        _iou,
        box_area_fraction,
        box_near_frame_edge,
    )

    face = cv2.cvtColor((skimage_data.lfw_subset()[30] * 255).astype(np.uint8),
                        cv2.COLOR_GRAY2BGR)
    patch = cv2.resize(face, (420, 420), interpolation=cv2.INTER_CUBIC)
    frame = cv2.resize(_room(), (360, 640))
    x0, y0 = -30, -197
    frame[0:y0 + 420, 0:360] = patch[-y0:, -x0:-x0 + 360]

    redactor = FaceRedactor()
    raw = redactor._raw_detect(frame, 2)
    large = [b for b, _ in raw if box_area_fraction(b, frame.shape) > LARGE_BOX_AREA_FRACTION]
    assert large, "precondition: the detection is a large box"
    assert all(box_near_frame_edge(b, frame.shape) for b in large), "precondition: at the edge"
    for scale in LARGE_BOX_CORROBORATION_SCALES:
        again = redactor._raw_detect(frame, scale)
        assert not any(_iou(b, a) >= 0.30 for b in large for a, _ in again), (
            f"precondition: plausibility2 would have corroborated this at {scale}")

    result = redactor.redact(_encode(frame))
    out = cv2.imdecode(np.frombuffer(result.image_bytes, np.uint8), cv2.IMREAD_COLOR)
    assert result.regions >= 1
    # the face's in-frame part: centre columns, from the top edge to the mouth
    face_region = out[0:int(y0 + 0.8 * 420), int(x0 + 0.2 * 420):int(x0 + 0.8 * 420)]
    assert float((face_region.max(axis=2) <= 8).mean()) >= 0.8, (
        "a close face cut by the frame edge was left on disk")


# -- one failed frame must not be laundered by the next success ---------


def test_one_unredacted_keyframe_makes_the_whole_session_unredacted(
    tmp_path, monkeypatch
):
    """`redact` never raises: when detection throws on one frame it returns
    the ORIGINAL bytes labelled `none`, and those are persisted. The session
    label used to be whatever the LAST keyframe got, so the next clean frame
    restored `faces-detected-and-filled/...`, and the dense stage -- which
    trusts `images/` as-is for any session not labelled `none` -- read the raw
    frame as a redacted keyframe.

    Real engine, real redactor; `_detect` throws on the second keyframe only.
    Then the dense stage's own reader is asked what it would do with that frame.
    """
    from tests import synthetic_scene as ss
    from tower.world_builder import redaction as RD
    from tower.world_builder.dense_pipeline import keyframe_image_bytes
    from tower.world_builder.engine import WorldBuilderEngine
    from tower.world_builder.records import CameraIntrinsics
    from tower.world_builder.store import WorldStore

    calls = {"n": 0}
    real_detect = RD.FaceRedactor._detect

    def flaky(self, image):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("detector fault on one frame")
        return real_detect(self, image)

    monkeypatch.setattr(RD.FaceRedactor, "_detect", flaky)

    width, height = 480, 360
    matrix = ss.camera_matrix(width, height)
    intrinsics = CameraIntrinsics(
        source="self_calibrated", model="pinhole",
        fx=float(matrix[0, 0]), fy=float(matrix[1, 1]),
        cx=float(matrix[0, 2]), cy=float(matrix[1, 2]),
        calibrated_width=width, calibrated_height=height,
    )
    frames = ss.render_sequence(
        ss.furnished_room(), ss.strafe(14, step=0.09), matrix, width, height
    )
    store = WorldStore(tmp_path)
    engine = WorldBuilderEngine(store, redactor_factory=RD.FaceRedactor)
    world_id = engine.create_world("laundering")
    session_id = engine.start_session(
        world_id, intrinsics=intrinsics, frame_source="synthetic",
        declared_size=(width, height),
    )
    for index, image in enumerate(frames):
        engine.observe(ss.encode_jpeg(image), source_seq=index)
    engine.stop_session()

    assert calls["n"] >= 3, "precondition: a success followed the failure"
    session = store.read_session(world_id, session_id)
    assert session.redaction == RD.REDACTION_NONE, (
        f"a keyframe was persisted unredacted but the session says "
        f"{session.redaction!r}"
    )

    failed = store.read_keyframes(world_id, session_id)[1]
    trusted = session.redaction != RD.REDACTION_NONE
    _data, origin, _mask = keyframe_image_bytes(
        store, world_id, session_id, failed.keyframe_id, None,
        RD.FaceRedactor(), keyframes_are_redacted=trusted,
    )
    assert origin != "world-keyframe", (
        "the dense stage read the unredacted frame as a redacted keyframe"
    )


def test_a_session_with_no_failures_keeps_its_redaction_label(tmp_path):
    """The sticky label must not demote a clean session."""
    from tests import synthetic_scene as ss
    from tower.world_builder.engine import WorldBuilderEngine
    from tower.world_builder.store import WorldStore

    matrix = ss.camera_matrix(WIDTH, HEIGHT)
    frames = ss.render_sequence(
        ss.furnished_room(), ss.strafe(6, step=0.09), matrix, WIDTH, HEIGHT
    )
    store = WorldStore(tmp_path)
    engine = WorldBuilderEngine(store)
    world_id = engine.create_world("clean")
    session_id = engine.start_session(world_id, frame_source="synthetic")
    for index, image in enumerate(frames):
        engine.observe(_encode(image), source_seq=index)
    engine.stop_session()

    assert store.read_session(world_id, session_id).redaction == FaceRedactor().label


# -- the fourth test: something that is not YuNet has to agree -----------
#
# Everything above is the detector arguing with itself. After all of it,
# precision on the canonical capture is 24 boxes of 140. These pin the stage
# that takes it to 24 of 104, and -- far more important -- pin every way it
# can go wrong to FILLING.


def _scripted_verifier(scores, *, available=True, raises=False,
                       below_area=0.08, threshold=0.44):
    """A verifier that answers from a list, in the order it is asked."""

    class _V:
        def __init__(self):
            self.asked = []
            self.available = available
            self.unavailable_reason = None if available else "scripted"
            self.below_area = below_area
            self.threshold = threshold

        def scores(self, image, boxes):
            self.asked.append(list(boxes))
            if raises:
                raise RuntimeError("the verifier fell over")
            return list(scores)[: len(boxes)]

    return _V()


def _detect_with(verifier, boxes_and_landmarks, image=None):
    redactor = FaceRedactor(verifier=verifier)
    redactor._raw_detect = lambda img, upscale: list(boxes_and_landmarks)
    return redactor, redactor._detect(_room() if image is None else image)


def test_the_verifier_drops_a_small_box_it_scores_below_the_threshold():
    box = _box_of_area(0.03)
    verifier = _scripted_verifier([0.10])
    redactor, boxes = _detect_with(verifier, [(box, _facelike_landmarks(box))])
    assert redactor.verifies
    assert verifier.asked == [[box]]
    assert boxes == []


def test_the_verifier_keeps_a_small_box_it_scores_above_the_threshold():
    box = _box_of_area(0.03)
    verifier = _scripted_verifier([0.90])
    _redactor, boxes = _detect_with(verifier, [(box, _facelike_landmarks(box))])
    assert len(boxes) == 1


def test_a_box_too_large_to_judge_is_never_offered_to_the_verifier():
    """Above the scope the crop is mostly padding and the face is partial. At
    a 25% cut the same threshold loses 79 of 2,210 held-out off-frame
    composites; at 12%, six; at 8%, none (PRECISION.md). So the verifier is
    not asked, and plausibility3's own evidence decides alone.
    """
    box = _box_of_area(0.20)
    verifier = _scripted_verifier([0.0])
    _redactor, boxes = _detect_with(verifier, [(box, _facelike_landmarks(box))])
    assert verifier.asked == []
    assert len(boxes) == 1


@pytest.mark.parametrize("kind", ["absent", "unloadable", "throws",
                                  "no-crop", "not-a-number"])
def test_every_way_the_verifier_can_fail_keeps_the_fill(kind):
    """A verifier that cannot judge must never be the reason a face is not
    filled. Each of these would have to DROP the box to be a privacy bug.
    """
    verifier = {
        "absent": False,
        "unloadable": _scripted_verifier([0.0], available=False),
        "throws": _scripted_verifier([0.0], raises=True),
        "no-crop": _scripted_verifier([None]),
        "not-a-number": _scripted_verifier([float("nan")]),
    }[kind]
    box = _box_of_area(0.03)
    redactor = FaceRedactor(verifier=verifier)
    redactor._raw_detect = lambda img, upscale: [(box, _facelike_landmarks(box))]
    assert len(redactor._detect(_room())) == 1


def test_the_label_names_the_gate_that_actually_ran():
    """A session that filled plausibility3's pixels must not claim
    plausibility4's. The suffix is the one thing a later reader has."""
    from tower.world_builder.redaction import (
        PLAUSIBILITY_ID,
        PLAUSIBILITY_WITHOUT_VERIFIER,
    )

    assert PLAUSIBILITY_ID == "plausibility4"
    with_verifier = FaceRedactor(verifier=_scripted_verifier([1.0]))
    without = FaceRedactor(verifier=False)
    assert with_verifier.verifies
    assert with_verifier.label.endswith("+" + PLAUSIBILITY_ID)
    assert not without.verifies
    assert without.label.endswith("+" + PLAUSIBILITY_WITHOUT_VERIFIER)
    assert with_verifier.label != without.label


def test_the_reredaction_step_writes_the_gate_that_ran():
    """The re-redacted set's name and label, and the allowlist the appearance
    build reads, all move together with the gate -- otherwise a p4 set is
    written under p3's name, or a p4 session is not trusted at all."""
    from tower.world_builder import appearance as A
    from tower.world_builder import reredaction as RR
    from tower.world_builder.redaction import PLAUSIBILITY_ID

    assert RR.TARGET_LABEL.endswith("+" + PLAUSIBILITY_ID)
    assert RR.set_name() == "images.redacted-" + PLAUSIBILITY_ID
    # p3 is now an OLDER gate, and its fill contains p4's on the same frame,
    # so a frame kept for any reason still meets the target label.
    assert RR.REREDACTABLE_LABELS[
        "faces-detected-and-filled/yunet-2023mar@0.30+plausibility3"] is True
    assert RR.TARGET_LABEL in A.TRUSTED_REDACTION_LABELS


def test_an_unloadable_verifier_is_reported_rather_than_assumed(tmp_path,
                                                                monkeypatch):
    from tower.world_builder.redaction import FaceVerifier

    monkeypatch.setenv("TOWER_FACE_VERIFIER", str(tmp_path / "nothing.json"))
    absent = FaceVerifier()
    assert not absent.available
    assert "TOWER_FACE_VERIFIER" in absent.unavailable_reason

    monkeypatch.setenv("TOWER_FACE_VERIFIER", "off")
    assert not FaceVerifier().available

    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    monkeypatch.setenv("TOWER_FACE_VERIFIER", str(broken))
    bad = FaceVerifier()
    assert not bad.available
    assert "could not be loaded" in bad.unavailable_reason


def test_the_crop_is_the_neighbourhood_padded_and_stretched():
    """The crop recipe is part of the rule: a probe fitted on 2.5x stretched
    crops means nothing if the product hands it something else."""
    from tower.world_builder.redaction import (
        VERIFIER_CONTEXT,
        VERIFIER_SIZE,
        verifier_crop,
    )

    frame = _room()
    inside = verifier_crop(frame, (300.0, 150.0, 40.0, 40.0))
    assert inside.shape == (VERIFIER_SIZE, VERIFIER_SIZE, 3)
    # a box at the very corner still yields a crop rather than raising
    corner = verifier_crop(frame, (-20.0, -20.0, 60.0, 60.0))
    assert corner is not None and corner.shape[0] == VERIFIER_SIZE
    assert VERIFIER_CONTEXT > 1.0
    # a dark, low-contrast patch comes back using the whole range
    dark = np.zeros((360, 640, 3), np.uint8)
    ramp = np.linspace(8, 26, 100).astype(np.uint8)
    dark[100:200, 100:200] = ramp[:, None, None]
    stretched = verifier_crop(dark, (120.0, 120.0, 40.0, 40.0))
    assert int(stretched.max()) - int(stretched.min()) > 200
    # a box with no area is refused rather than guessed at
    assert verifier_crop(frame, (10.0, 10.0, 0.0, 10.0)) is None


def test_the_verifier_ranks_a_real_face_above_the_room_it_is_in():
    """Recall and precision, on fixtures: a real photograph of a face, at
    three sizes, against six real non-face textures and the bare room, all
    composited into the same rendered room.

    A synthetic scene's absolute scores are not the canonical capture's --
    everything here scores high, which is the safe direction -- so what is
    pinned is the recall and the SEPARATION: every face is accepted, and
    every face outscores every non-face.
    """
    from tower.world_builder.redaction import shared_verifier

    verifier = shared_verifier()
    if not verifier.available:
        pytest.skip(f"no face verifier here: {verifier.unavailable_reason}")
    room = _room()

    def _with(patch, size=90, at=(200, 80)):
        frame = room.copy()
        resized = cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)
        frame[at[1]:at[1] + size, at[0]:at[0] + size] = resized
        return frame, (float(at[0]), float(at[1]), float(size), float(size))

    faces = []
    for size in (60, 90, 140):
        frame, box = _with(_face_patch(size), size)
        faces.append(verifier.scores(frame, [box])[0])

    others = {"room": verifier.scores(room, [(200.0, 80.0, 90.0, 90.0)])[0]}
    for name in ("brick", "grass", "gravel", "coffee", "coins", "text"):
        texture = getattr(skimage_data, name)()
        texture = (cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)
                   if texture.ndim == 2
                   else cv2.cvtColor(texture, cv2.COLOR_RGB2BGR))
        frame, box = _with(texture)
        others[name] = verifier.scores(frame, [box])[0]

    assert min(faces) >= verifier.threshold, (
        f"a real face was rejected: {faces} against {verifier.threshold}")
    assert min(faces) > max(others.values()) + 0.03, (
        f"faces {faces} did not separate from {others}")
