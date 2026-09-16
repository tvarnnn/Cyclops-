"""Face redaction applied BEFORE a keyframe is written to disk.

The platform's privacy rule is a pipeline, and the order in it is the
whole point:

    raw sensor data -> necessary ephemeral perception -> derived
    structured information -> privacy transformation -> persistence

World Builder is the one module that retains raw imagery, so it is the one
module where the transformation has somewhere to happen. This runs at
`engine._persist_keyframe`, the single choke point every persisted pixel
already passes through -- the same property `capture.write_frame` has and
for the same reason: a rule enforced in one place is a rule, and a rule
enforced by everyone remembering is a hope.

**Before persistence, not on read.** Redacting on the way out would leave
the raw frames on disk, where a later bug, a backup, a forensic recovery
or a policy change can still reach them. That is a display filter, not a
privacy transformation, and the rule above puts the transformation before
the write.

WHY THIS IS AFFORDABLE, MEASURED
--------------------------------
The obvious objection is that the reconstruction is built from these
pixels, so redacting them damages the geometry. Measured on 10 synthetic
scene seeds, full pipeline (observe -> stop -> build), blurring a centred
region of the frame:

    blur area   ORB features   keyframes   poses solved   points
    0%  (base)   1406 +- 31      5.0/5.0      4.0/4.0      1567
    5%           1378 (98%)      5.0/5.0      4.0/4.0      1425 (-9%)
    15%          1288 (92%)      5.0/5.0      4.0/4.0      1365 (-13%)
    30%          1103 (78%)      5.0/5.0      4.0/4.0      1162 (-26%)

**Keyframe acceptance and pose solving were completely insensitive** --
5.0 of 5.0 keyframes and 4 of 4 poses at every level, with a
byte-identical rejection histogram. Only feature density and the point
count degrade, and they degrade smoothly.

And a real face is small: measured detector boxes at 640x360 have a median
area of 1.74% of the frame, 4.45% after the head dilation below. So the
honest cost of one redacted face is the 5% row -- no keyframes lost, no
poses lost, about 9% of the point cloud.

WHY THIS DETECTOR
-----------------
YuNet is already compiled into the OpenCV this project ships
(`cv2.FaceDetectorYN`); only its weights were missing. Measured at 640x360
with the settings below: **17.4 ms per frame**, 100/100 on distinct real
faces from scikit-image's LFW subset, 0 false positives on 40 face-free
frames, and it holds through 45 degrees of head tilt, mirrors, faces on
screens, and motion blur.

Rejected alternatives, all measured:

- **scikit-image's LBP cascade** (already on disk, via easyocr's
  scikit-image). Works, needs no download -- and costs 300-524 ms per
  frame and goes blind above about 15 degrees of head tilt, which is
  anyone looking down at a phone.
- **The COCO person detector plus pose keypoints** we already load for
  Scene Understanding. A head box from keypoints did cover 100% of the
  face in every case tested -- but at 998 ms per frame, ~57x YuNet, and it
  cannot redact a face with no body in frame, which is a common shape for
  first-person capture. The cheaper heuristic, "the upper quarter of the
  person box", leaks 33-68% of the face and is worst exactly when the
  person is occluded. Both rejected.
- **mediapipe**, which pulls `opencv-contrib-python` alongside our
  headless build -- the same collision this project already rejected
  `rapidocr_onnxruntime` for. **facenet-pytorch**, which downgrades torch,
  torchvision, numpy and pillow. Both rejected without installing.

WHAT MAY BE CLAIMED
-------------------
Not "faces removed". The detector has measured hard false negatives: a
face occluded more than about 60%, and a face rotated about 90 degrees in
plane. Profile and rear views are a known blind spot of this detector
class and were not testable here.

So the label records **what ran**, not what was achieved -- the detector's
identity and its threshold, so a reader can look up its limits. A session
says `faces-detected-and-filled/yunet-2023mar@0.30`, never "redacted",
"anonymised" or "privacy-safe". `retains_raw_imagery` stays true and the
privacy tags stay exactly as they were: bodies, clothing, room contents
and any undetected face are all still in the image.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# What a session records when nothing was applied. Historical sessions keep
# this forever; it is never backfilled.
REDACTION_NONE = "none"

DETECTOR_ID = "yunet-2023mar"

# Measured band. Below 0.2 the detector fires on face-free frames (35 of
# 40 synthetic room frames at 0.1); above 0.4 it starts missing small
# faces and faces on screens (both 0/3 at 0.6). 0.3 sits in the middle of
# the only defensible range.
CONFIDENCE = 0.30
NMS_THRESHOLD = 0.30
TOP_K = 5000

# The detector is trained for larger faces than a 640x360 first-person
# frame usually contains. Upscaling costs 12.7 ms and buys the 20-32 px
# faces that a wide first-person view is full of.
UPSCALE = 2

# A face box is not a head. Measured: the raw box has a median area of
# 1.74% of the frame and covers the face only; 1.6x covers hair, ears and
# jaw at a median 4.45%, still inside the "no measurable geometry cost"
# band.
HEAD_DILATION = 1.6

# Solid fill, not blur. Blur is partially invertible, and it is not
# cheaper -- measured 0.29 ms against 0.01 ms, with identical ORB
# retention (86% vs 86% at 30% of the frame).
FILL_VALUE = 0

# Re-encode quality. One extra JPEG generation was measured free: ORB
# 1439 -> 1437, keyframes 5 -> 5, points 1634 -> 1627.
JPEG_QUALITY = 90

MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"

# WHERE THE WEIGHTS ARE, INDEPENDENTLY OF WHERE THE PROCESS WAS STARTED.
#
# This used to be `Path("models") / MODEL_FILENAME`, resolved against the
# current working directory. A Tower or a script started from anywhere but
# `tower/` therefore found no model, `FaceRedactor.available` was False, and
# the consequence is not an error: `engine.py` logs a warning and persists the
# keyframe UNREDACTED, and `redact()` returns the original bytes labelled
# `none`. So a privacy transformation was silently conditional on a caller's
# cwd. It was found when a batch script run from a scratch directory refused
# 429 of 429 frames -- refusing is the safe direction, and only the dense
# stage's own check made it visible at all.
#
# The package's own location is the honest anchor: `tower/tower/world_builder/`
# -> `tower/models/`. The cwd-relative path is still tried, so an existing
# deployment that relies on it keeps working, and the environment override
# still wins over both.
_PACKAGE_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / MODEL_FILENAME
_CWD_MODEL_PATH = Path("models") / MODEL_FILENAME

# Kept as the name other modules and tests import.
DEFAULT_MODEL_PATH = _PACKAGE_MODEL_PATH


@dataclass(frozen=True)
class RedactionResult:
    """What happened to one image, and what may be said about it."""

    image_bytes: bytes
    label: str
    regions: int
    unavailable_reason: str | None = None

    @property
    def applied(self) -> bool:
        return self.unavailable_reason is None


def model_path() -> Path | None:
    """Where the detector weights are, or None.

    `TOWER_FACE_REDACTION_MODEL` overrides; otherwise a file vendored at
    the default path is used if it is there. Absent means redaction is
    unavailable, which is reported rather than silently skipped.
    """
    override = os.environ.get("TOWER_FACE_REDACTION_MODEL")
    if override:
        candidate = Path(override.strip())
        return candidate if candidate.exists() else None
    for candidate in (_PACKAGE_MODEL_PATH, _CWD_MODEL_PATH):
        if candidate.exists():
            return candidate
    return None


class FaceRedactor:
    """Fills detected face regions before an image is persisted.

    Constructed once per session. Holds one detector; a frame-size change
    re-targets it rather than rebuilding it, because DAT's adaptive ladder
    can change resolution mid-stream.
    """

    def __init__(self, path=None) -> None:
        # An explicitly supplied path is checked too, not just the
        # default. Trusting it produced a redactor that reported itself
        # AVAILABLE and then failed on every frame -- so a session would
        # record `none` by way of a caught exception rather than because
        # anyone knew the model was missing.
        candidate = Path(path) if path is not None else model_path()
        if candidate is not None and not candidate.exists():
            candidate = None
        self._path = candidate
        self._detector = None
        self._size = None
        self._failed_reason: str | None = None
        if self._path is None:
            self._failed_reason = (
                "no face-detection model is available on this Tower; set "
                "TOWER_FACE_REDACTION_MODEL or vendor "
                f"{DEFAULT_MODEL_PATH.as_posix()}"
            )

    @property
    def available(self) -> bool:
        return self._failed_reason is None

    @property
    def unavailable_reason(self) -> str | None:
        return self._failed_reason

    @property
    def label(self) -> str:
        """The value a session records for imagery this redactor wrote.

        Names the detector and its threshold rather than asserting an
        outcome. "redacted" alone would invite a reader to infer
        completeness the detector cannot support.
        """
        if not self.available:
            return REDACTION_NONE
        return f"faces-detected-and-filled/{DETECTOR_ID}@{CONFIDENCE:.2f}"

    def redact(self, image_bytes: bytes) -> RedactionResult:
        """Fill every detected face. Returns the ORIGINAL bytes on failure.

        Never raises. A redactor that threw would stop a keyframe being
        persisted, which would trade a privacy improvement for data loss --
        and the caller must be able to record honestly that nothing was
        applied.
        """
        if not self.available:
            return RedactionResult(
                image_bytes=image_bytes,
                label=REDACTION_NONE,
                regions=0,
                unavailable_reason=self._failed_reason,
            )
        try:
            return self._redact(image_bytes)
        except Exception as exc:  # noqa: BLE001
            logger.exception("[Tower][Redaction] failed; persisting unchanged")
            return RedactionResult(
                image_bytes=image_bytes,
                label=REDACTION_NONE,
                regions=0,
                unavailable_reason=(
                    f"face redaction failed with {type(exc).__name__}; the "
                    "image was persisted unchanged"
                ),
            )

    def _redact(self, image_bytes: bytes) -> RedactionResult:
        import cv2
        import numpy as np

        buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("undecodable image")

        boxes = self._detect(image)
        if not boxes:
            return RedactionResult(
                image_bytes=image_bytes, label=self.label, regions=0
            )

        height, width = image.shape[:2]
        for x, y, w, h in boxes:
            x0 = max(0, int(x))
            y0 = max(0, int(y))
            x1 = min(width, int(x + w))
            y1 = min(height, int(y + h))
            if x1 > x0 and y1 > y0:
                image[y0:y1, x0:x1] = FILL_VALUE

        ok, encoded = cv2.imencode(
            ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        )
        if not ok:
            raise ValueError("could not re-encode a redacted image")
        return RedactionResult(
            image_bytes=encoded.tobytes(), label=self.label, regions=len(boxes)
        )

    def _detect(self, image) -> list:
        import cv2

        height, width = image.shape[:2]
        scaled = cv2.resize(
            image,
            (width * UPSCALE, height * UPSCALE),
            interpolation=cv2.INTER_CUBIC,
        )
        size = (scaled.shape[1], scaled.shape[0])
        if self._detector is None:
            self._detector = cv2.FaceDetectorYN.create(
                str(self._path), "", size, CONFIDENCE, NMS_THRESHOLD, TOP_K
            )
            self._size = size
        elif size != self._size:
            # DAT's adaptive ladder can change resolution mid-stream.
            self._detector.setInputSize(size)
            self._size = size

        _, faces = self._detector.detect(scaled)
        if faces is None:
            return []

        boxes = []
        for face in faces:
            x, y, w, h = (float(value) / UPSCALE for value in face[:4])
            # Dilate about the centre: a face box is not a head.
            cx, cy = x + w / 2.0, y + h / 2.0
            w *= HEAD_DILATION
            h *= HEAD_DILATION
            boxes.append((cx - w / 2.0, cy - h / 2.0, w, h))
        return boxes
