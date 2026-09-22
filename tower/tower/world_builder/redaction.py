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

# WHAT THE DETECTOR FIRES ON WHEN THERE IS NO FACE, AND WHAT TO DO ABOUT IT
# ------------------------------------------------------------------------
# The claim above -- "0 false positives on 40 face-free frames" -- was measured
# on SYNTHETIC room renders. On a real capture it does not hold, and the gap is
# not marginal. Re-measured on the canonical world's 398 raw source frames
# (`Glasses-scratch/wb-final-recon/redaction/`):
#
#     240 detections at 0.30, of which 20 are a face. Precision 8.3%.
#     81 are the wearer's own HAND holding a phone or on a keyboard.
#     135 are scene: a printed cup logo, a lit PC case, bare wall, carpet,
#     a doorway, a pile of laundry, a desk edge.
#     12.6% of every pixel in the capture was filled; 92 frames lost more
#     than 20% of themselves, 38 more than 50%, the worst 93.8%.
#
# The 20 true positives are a printed portrait of a man hanging on the bedroom
# wall. There is no live face anywhere in the 398 keyframes, and none in the
# 53 841 frames of the whole capture corpus either.
#
# THE OBVIOUS FIX IS THE WRONG ONE. "A face has texture, a blank wall does not,
# so reject low-contrast detections" inverts the data here: the real faces are
# the LOW-contrast detections (median within-box contrast 12.2) and the false
# ones are the high-contrast ones (hand 36.3, scene 30.9). AUC 0.096 -- that
# test is a better than 9-to-1 detector of the wrong class. Raising CONFIDENCE
# is nearly as bad: 0.35 already costs two of the twenty true positives and
# 0.40 costs four, because a small, dim, real face scores no better than a cup.
#
# What does separate them is three things the detector already knows, or can
# be asked again cheaply.

# 1. THE FIVE LANDMARKS HAVE TO LIE LIKE A FACE.
#
# YuNet returns the two eyes, the nose and the two mouth corners with every
# box, at no extra cost, and on a false positive they are usually nonsense --
# a nose 6.5 eye-separations off the midline, a mouth above the eyes. Measured
# over the 20 in-situ faces and 517 composited real faces, and widened, because
# 100 frontal LFW faces and one portrait are a floor on the real spread and not
# a description of it. Each band is (min, max); all five must hold.
#
# These are ratios of landmark distances, so they do not move with exposure,
# resolution or the DAT ladder, and because the vertical ones are measured
# along the eye line's own perpendicular, head ROLL is free.
EYE_SEPARATION_BAND = (0.15, 0.62)      # over box width; faces 0.186..0.535
MAX_NOSE_OFFSET = 0.80                  # over eye sep; faces 0.000..0.647
EYE_TO_NOSE_BAND = (0.25, 1.60)         # over eye sep; faces 0.361..1.279
NOSE_TO_MOUTH_BAND = (0.20, 1.90)       # over eye sep; faces 0.299..1.557
MOUTH_TO_EYE_BAND = (0.40, 1.25)        # mouth width over eye sep

# ...BUT ONLY WHERE IT CAN BE AFFORDED. A small detection is exempt and kept.
#
# The bands were set from frontal faces and one portrait; the case they cover
# worst is a DISTANT bystander in three-quarter view, whose five landmarks YuNet
# places least cleanly. That risk lives entirely in small boxes. The pixels the
# test buys back live entirely in large ones: the wearer's hands are never
# smaller than 2.68% of the frame (81 of 81, median 9.05%). Measured over the
# 398 raw frames, exempting everything under 2% leaves the filled fraction at
# 1.83% -- unchanged to two decimals -- recovers the same 12 of 13 fill-caused
# refusals, and lets back exactly ONE extra false positive. At 2% a face is
# about 65 px wide, roughly a metre from the lens; 95% of the 73 real faces
# measured (portrait and live) are smaller, so they never meet the landmark
# test at all. 3% would start exempting hands.
LANDMARK_TEST_ABOVE_AREA = 0.02

# 2. A BOX THAT COVERS THE FRAME NEEDS NO EXTRA EVIDENCE, BUT IT NEEDS SOME.
#
# The largest raw box among 840 composited faces is 30.1% of the frame and the
# 95th percentile is 19.9%. The largest false box is 47.8%, and after
# HEAD_DILATION it took 93.8% of a frame whose content is a wall, a PC tower
# and carpet.
#
# Two rules have been tried here and both were wrong, so both are recorded.
#
# A hard cap (above 25%, never filled) left a real face closer than about
# 20 cm -- someone leaning into the wearer -- on disk: 0 of 65 composited close
# faces filled.
#
# "Facelike landmarks OR found again at native resolution" then filled 65 of
# 65, but landmark geometry carries NO evidence at this size: all 40 false
# boxes over 25% on the canonical capture are "facelike", because a box that
# large spreads its five points like a face whatever is inside it. So the rule
# filled every large box: 7.51% of the capture's pixels, 31 frames over half
# black, and a replay keyframe of a plain wall and a PC tower stored fully black.
#
# What does carry evidence is the one thing a genuinely large face is best at:
# being seen at LOW resolution. A face filling a third of the frame is still a
# 90-px face at a quarter of the resolution, while a coincidence of texture
# spread over that area usually dissolves. Measured over the same 40 false
# boxes and 65 close composited faces (REDACTION.md, section 13):
#
#     re-detected at    false boxes kept   close faces filled   filled pixels
#     native                  4 / 40             64 / 65             2.57%
#     1/2                     1 / 40             65 / 65             1.99%
#     1/4                     2 / 40             65 / 65             2.19%
#     any of the three        6 / 40             65 / 65             2.72%
#
# The union is used, not the cheapest single scale: each scale misses faces the
# others see (native missed one composite that 1/2 and 1/4 found), a composite
# set is 65 frontal faces rather than every close face there is, and the cost of
# the union over the best single scale is 0.7 points of fill. Landmark geometry
# is NOT consulted for a large box, except that landmarks too broken to judge
# still fill it without asking.
LARGE_BOX_AREA_FRACTION = 0.25
LARGE_BOX_CORROBORATION_SCALES = (1.0, 0.5, 0.25)

# 2b. EXCEPT AT THE FRAME EDGE, WHERE A REAL FACE IS CUT AND LOSES THE EVIDENCE.
#
# The composites behind the table above were faces placed wholly inside the
# frame. A person leaning into the wearer is usually cut by the frame edge, and
# a cut face is exactly what YuNet sees worst at reduced resolution: the part
# that would make it a face at 1/4 is outside the image. Review 3 pasted close
# faces 5% off each edge: plausibility2 left 4 of 594 composites unfilled that
# plausibility1 had filled, and on a larger held-out set (every LFW face and the
# astronaut, 300-500 px wide, cut by each edge, 3,232 composites) it left 32.
# On 3 of the first 4 no scale found ANY box, native included.
#
# So a large box within EDGE_MARGIN of the frame edge -- 5% of the short side,
# 18 px on 360x640 -- goes back to plausibility1's rule: facelike landmarks OR
# found again. Padding the frame before the smaller passes and low-confidence
# re-detection were measured too, and each still lost faces the edge rule
# fills. Measured (REDACTION.md section 14, fix-r4/r1_margin.py):
#
#                   raw fill  >50% frames  worst   close  off-frame lost  held-out lost
#   plausibility1     7.51%        31      93.2%   65/65      0 / 594        0 / 3232
#   plausibility2     2.72%         6      79.8%   65/65      4 / 594       32 / 3232
#   edge <= 0 px      5.73%        20      93.2%   65/65      2 / 594        4 / 3232
#   edge <= 5% (this) 5.99%        22      93.2%   65/65      0 / 594        1 / 3232
#   edge <= 10%       7.21%        29      93.2%   65/65      0 / 594        0 / 3232
#
# The price is stated, not hidden: 25 of the capture's 40 false large boxes
# touch an edge, so this gives back most of plausibility2's pixels (2.72% ->
# 5.99%, and keyframe 188 is 93.2% black again). It is paid because the cost of
# an unnecessary fill is pixels, and the cost of a skipped one is a face on disk.
# 10% would recover the last held-out face (a box 23 px from the left edge whose
# face is cut at the BOTTOM -- the box is misplaced, not truncated) at the cost of
# almost every remaining false box, and adding a confidence-0.10 re-detection at
# 1/2 or 1/4 recovers it at 6.58% fill and 25 frames over half black. That one
# face is recorded as a known miss instead.
EDGE_MARGIN = 0.05

# 3. A BIG CLAIM ON THE FRAME HAS TO SURVIVE BEING LOOKED AT AGAIN, SMALLER.
#
# Detecting the same frame at UPSCALE 1 instead of 2 keeps 80% of the in-situ
# faces and 88.9% of the composited ones, but only 23.7% of the scene false
# positives: a real face is an object and survives resampling, a coincidence
# of a few pixels does not. Applied to everything it would cost real recall on
# SMALL faces (20% survive at 24 px wide, 80% at 35 px) -- which is exactly
# what UPSCALE 2 exists to find. So it is applied only where the detection is
# big enough that losing it cannot be a distant bystander AND big enough to
# matter for the pixels: at 5% of the frame a face is ~95 px wide, and every
# composited face at or above that width survived the second look.
#
# The second pass runs at UPSCALE 1, costs about a quarter of the first, and
# only runs at all when a large detection survived the first two tests.
CORROBORATE_ABOVE_AREA = 0.05
CORROBORATION_IOU = 0.30

# Named in the session label, so a reader of an old world can tell which
# imagery went through the gate and which predates it.
#
# plausibility1: landmark bands, native corroboration from 5%, and above 25%
#                facelike OR native-corroborated (53be0c3).
# plausibility2: the same, except above 25% the landmarks are ignored and the
#                box must be found again at native, 1/2 or 1/4 resolution.
# plausibility3: the same, except that a box over 25% within EDGE_MARGIN of the
#                frame edge is filled on facelike landmarks OR being found again.
# plausibility4: the same, plus the VERIFIER below on every surviving box small
#                enough to judge. Only ever removes a box.
PLAUSIBILITY_ID = "plausibility4"

# The gate this reduces to when the verifier is not on this Tower. The label
# records which of the two actually ran, because they fill different pixels.
PLAUSIBILITY_WITHOUT_VERIFIER = "plausibility3"

# 4. AND THEN SOMETHING THAT IS NOT YUNET HAS TO AGREE IT IS A FACE.
# -----------------------------------------------------------------
# Everything above is YuNet arguing with itself: its own landmarks, its own
# output at another resolution. After all of it, precision on the canonical
# capture is 24 boxes of 140 -- 17.1% counting the wall portrait and the second
# print, 14.3% counting the portrait alone. Eleven of every twelve regions the
# product blacks out are a wall, a cup logo, a desk, a hand or a pile of
# laundry. That is affordable as pixels; it stops being affordable the moment
# anything PROPAGATES a fill region (`appearance.RedactionConsensus`), because
# then every false positive removes a piece of the room from every frame.
#
# So the last test asks a model that has never heard of YuNet. The candidate's
# neighbourhood is cut out, put on a scale a general vision model has actually
# seen, embedded by DINOv2-base, and scored by a logistic probe. Measured
# (`Glasses-scratch/wb-final-recon/fixit/precision/PRECISION.md`):
#
#     rule                          boxes kept   faces kept   precision
#     plausibility3                     140         24/24        17.1%
#     plausibility4 (this)              104         24/24        23.1%
#     ... among regions <=10% of the frame, which are the ones a consensus
#         propagates:                    67 -> 49  24/24   27.6% -> 37.6%
#
# WHAT IT IS FITTED ON, AND WHAT IT IS NOT. Five banks, none of them from the
# capture it is measured on: 284 eye-labelled detections from twelve other
# captures (53 of them real live faces), 1,500 more from 87 further captures,
# and three synthetic banks pasted into other captures' frames -- frontal
# faces, faces as dim warped PRINTS on a wall, and close faces cut by the frame
# edge. The canonical capture's own 140 boxes were only ever a test set.
#
# WHERE IT IS ALLOWED TO JUDGE. Only a box at most VERIFY_BELOW_AREA of the
# frame. Above that the crop is mostly padding and the face is partial, and the
# measurement says so plainly: at a 25% cut the same threshold loses 79 of
# 2,210 off-frame composites; at 12%, six; at 8%, none. A close bystander lives
# above that line and keeps plausibility3's own evidence untouched.
#
# THE THRESHOLD. The lowest score any in-scope box carrying a real or
# composited face received, over every recall set, is 0.464. 0.44 is that with
# a margin. It is chosen from POSITIVES only; the 116 labelled non-faces it is
# measured against never entered the choice. Calibrated instead only on
# off-capture held-out faces it would be 0.32, and precision 17.5% -- the gap
# is what this capture's own portrait contributes, and is stated rather than
# hidden.
#
# FAILS TOWARDS FILLING, LOUDLY. No coefficients, no torch, no weights, a load
# that throws, a crop that cannot be cut, a forward pass that raises: every box
# is kept and the label says `plausibility3`, because a session that filled
# plausibility3's pixels must not claim plausibility4's.
VERIFIER_ID = "dinov2b-logreg-v1"
VERIFIER_BACKBONE = "facebook/dinov2-base"
VERIFY_BELOW_AREA = 0.08
VERIFY_THRESHOLD = 0.44
VERIFIER_CONTEXT = 2.5
VERIFIER_SIZE = 224
VERIFIER_FILENAME = "face_verifier_dinov2b_v1.json"

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

_PACKAGE_VERIFIER_PATH = (
    Path(__file__).resolve().parents[2] / "models" / VERIFIER_FILENAME
)
_CWD_VERIFIER_PATH = Path("models") / VERIFIER_FILENAME
DEFAULT_VERIFIER_PATH = _PACKAGE_VERIFIER_PATH


def landmark_geometry(box, landmarks) -> dict:
    """The five landmarks as ratios, in the face's own frame.

    `box` is (x, y, w, h) and `landmarks` the five (x, y) pairs YuNet returns
    with it, in the order (right eye, left eye, nose, right mouth corner, left
    mouth corner) -- "right" being the viewer's left.

    Every quantity is a RATIO, so nothing here moves with exposure, resolution
    or DAT's adaptive ladder. The two vertical ones are measured along the
    perpendicular to the eye line rather than along the image's y-axis, so a
    head tilted 45 degrees scores exactly as an upright one: roll is free, and
    the detector is documented to hold through that much tilt.

    Returns `None` for a degenerate detection -- one whose two eyes are on top
    of each other -- which is itself a finding about the detection.
    """
    import numpy as np

    x, y, w, h = (float(v) for v in box)
    pts = np.asarray(landmarks, dtype=float).reshape(5, 2)
    right_eye, left_eye, nose, right_mouth, left_mouth = pts
    eye = left_eye - right_eye
    eye_separation = float(np.hypot(*eye))
    if eye_separation <= 1e-6 or w <= 0 or h <= 0:
        return None

    axis = eye / eye_separation
    down = np.array([-axis[1], axis[0]])
    if float(np.dot(down, (0.0, 1.0))) < 0:
        down = -down
    eye_mid = (left_eye + right_eye) / 2.0
    mouth_mid = (right_mouth + left_mouth) / 2.0

    return {
        "eye_separation": eye_separation / w,
        "nose_offset": abs(float(np.dot(nose - eye_mid, axis))) / eye_separation,
        "eye_to_nose": float(np.dot(nose - eye_mid, down)) / eye_separation,
        "nose_to_mouth": float(np.dot(mouth_mid - nose, down)) / eye_separation,
        "mouth_to_eye": float(np.hypot(*(left_mouth - right_mouth)))
        / eye_separation,
    }


def landmarks_are_facelike(box, landmarks) -> bool:
    """Do the five landmarks lie the way a face's do?

    This is the cheapest of the three tests and the one that removes the
    wearer's own hands, which are the largest single class of false positive
    on this corpus: a hand holding a phone puts YuNet's "nose" a median 0.89
    eye-separations off the midline against 0.14 for a real face, and its
    "mouth" 1.40 below the nose against 0.60.
    """
    g = landmark_geometry(box, landmarks)
    if g is None:
        return False
    lo, hi = EYE_SEPARATION_BAND
    if not lo <= g["eye_separation"] <= hi:
        return False
    if g["nose_offset"] > MAX_NOSE_OFFSET:
        return False
    lo, hi = EYE_TO_NOSE_BAND
    if not lo <= g["eye_to_nose"] <= hi:
        return False
    lo, hi = NOSE_TO_MOUTH_BAND
    if not lo <= g["nose_to_mouth"] <= hi:
        return False
    lo, hi = MOUTH_TO_EYE_BAND
    if not lo <= g["mouth_to_eye"] <= hi:
        return False
    return True


def landmark_verdict(box, landmarks):
    """True (facelike), False (measurably not), or None (cannot be judged).

    The distinction is the whole point. `landmarks_are_facelike` folds "cannot
    judge" into False, which is right for a yes/no question and wrong for a
    gate: YuNet collapses both eyes onto one point on a near-profile face, and
    a NaN compares False against every band. Both used to DROP the fill. A gate
    that cannot judge must fill, so the caller needs the third answer.
    """
    import math

    try:
        values = [float(v) for v in box] + [float(v) for v in landmarks]
    except (TypeError, ValueError):
        return None
    if len(values) != 14 or not all(math.isfinite(v) for v in values):
        return None
    if landmark_geometry(box, landmarks) is None:
        return None
    return landmarks_are_facelike(box, landmarks)


def box_area_fraction(box, frame_shape) -> float:
    """The raw detection's area as a fraction of the frame, before dilation."""
    height, width = frame_shape[:2]
    if height <= 0 or width <= 0:
        return 0.0
    return (float(box[2]) * float(box[3])) / float(width * height)


def box_near_frame_edge(box, frame_shape, margin: float = None) -> bool:
    """Whether the raw box touches, crosses, or comes within `margin` (a
    fraction of the frame's SHORT side) of any frame edge."""
    height, width = frame_shape[:2]
    margin = EDGE_MARGIN if margin is None else margin
    slack = margin * min(width, height)
    x, y, w, h = (float(v) for v in box)
    return (x <= slack or y <= slack
            or x + w >= width - slack or y + h >= height - slack)


def verifier_crop(image, box, context: float = None, size: int = None):
    """The square neighbourhood the verifier looks at, or None.

    `context` times the box's longer side, about its centre, replicate-padded
    where it leaves the frame, resized to `size`, and then stretched between
    its own 1st and 99th percentile.

    The stretch is not cosmetic. This capture is a dark room, and its real
    faces are its LOWEST-contrast detections (REDACTION.md section 3): a raw
    crop of one is nearly black, and a backbone trained on ordinary
    photographs has never seen anything like it. Measured, the stretch takes
    set-A ranking from AUC 0.911 to 0.925 on its own.
    """
    import cv2
    import numpy as np

    context = VERIFIER_CONTEXT if context is None else context
    size = VERIFIER_SIZE if size is None else size
    height, width = image.shape[:2]
    x, y, w, h = (float(v) for v in box)
    if not (w > 0 and h > 0):
        return None
    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(w, h) * context
    x0, y0 = int(round(cx - side / 2.0)), int(round(cy - side / 2.0))
    x1, y1 = int(round(cx + side / 2.0)), int(round(cy + side / 2.0))
    left, top = max(0, -x0), max(0, -y0)
    right, bottom = max(0, x1 - width), max(0, y1 - height)
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(width, x1), min(height, y1)
    if cx1 <= cx0 or cy1 <= cy0:
        return None
    patch = image[cy0:cy1, cx0:cx1]
    if left or top or right or bottom:
        patch = cv2.copyMakeBorder(patch, top, bottom, left, right,
                                   cv2.BORDER_REPLICATE)
    if patch.size == 0:
        return None
    patch = cv2.resize(
        patch, (size, size),
        interpolation=cv2.INTER_CUBIC if patch.shape[0] < size else cv2.INTER_AREA,
    )
    values = patch.astype(np.float32)
    low = float(np.percentile(values, 1.0))
    high = float(np.percentile(values, 99.0))
    if high - low < 4.0:
        # A flat patch has nothing to stretch, and stretching it would amplify
        # sensor noise into structure the probe was never shown.
        return patch
    return np.clip((values - low) * (255.0 / (high - low)), 0, 255).astype("uint8")


def verifier_path() -> Path | None:
    """Where the probe coefficients are, or None.

    `TOWER_FACE_VERIFIER` overrides, and the single word `off` disables the
    stage deliberately -- which is a supported configuration, not a failure:
    the gate is then `plausibility3` and the label says so.
    """
    override = os.environ.get("TOWER_FACE_VERIFIER")
    if override:
        if override.strip().lower() == "off":
            return None
        candidate = Path(override.strip())
        return candidate if candidate.exists() else None
    for candidate in (_PACKAGE_VERIFIER_PATH, _CWD_VERIFIER_PATH):
        if candidate.exists():
            return candidate
    return None


class FaceVerifier:
    """Does a general vision model agree that this box is a face?

    One instance per process (`shared_verifier`), because the backbone is
    350 MB and `FaceRedactor` is constructed per session and per test.

    Every failure path leads to the same place: `available` is False, and the
    caller keeps every box. There is no path on which this class causes a box
    NOT to be filled because something went wrong.
    """

    def __init__(self, path=None) -> None:
        self._weights = None
        self._bias = 0.0
        self.threshold = VERIFY_THRESHOLD
        self.below_area = VERIFY_BELOW_AREA
        self.identifier = VERIFIER_ID
        self._model = None
        self._processor = None
        self._failed_reason: str | None = None

        candidate = Path(path) if path is not None else verifier_path()
        if candidate is None or not candidate.exists():
            self._failed_reason = (
                "no face-verifier coefficients on this Tower; set "
                f"TOWER_FACE_VERIFIER or vendor {VERIFIER_FILENAME}"
            )
            return
        try:
            import json

            import numpy as np

            doc = json.loads(Path(candidate).read_text())
            self._weights = np.asarray(doc["weights"], dtype=np.float32)
            self._bias = float(doc["bias"])
            self.threshold = float(doc.get("threshold", VERIFY_THRESHOLD))
            self.below_area = float(
                doc.get("apply_below_area_fraction", VERIFY_BELOW_AREA)
            )
            self.identifier = str(doc.get("id", VERIFIER_ID))
            backbone = os.environ.get("TOWER_FACE_VERIFIER_BACKBONE") or str(
                doc.get("backbone", VERIFIER_BACKBONE)
            )
            import torch
            from transformers import AutoImageProcessor, AutoModel

            self._torch = torch
            self._processor = AutoImageProcessor.from_pretrained(backbone)
            model = AutoModel.from_pretrained(backbone)
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = model.to(self._device).eval()
        except Exception as exc:  # noqa: BLE001
            self._model = None
            self._failed_reason = (
                f"the face verifier could not be loaded ({type(exc).__name__}: "
                f"{exc}); every detection will be filled"
            )
            logger.warning("[Tower][Redaction] %s", self._failed_reason)

    @property
    def available(self) -> bool:
        return self._failed_reason is None and self._model is not None

    @property
    def unavailable_reason(self) -> str | None:
        return self._failed_reason

    def scores(self, image, boxes) -> list:
        """P(face) per box. Raises rather than guessing; the caller fills."""
        import cv2
        import numpy as np

        crops = [verifier_crop(image, b) for b in boxes]
        out = [None] * len(boxes)
        usable = [i for i, c in enumerate(crops) if c is not None]
        if not usable:
            return out
        torch = self._torch
        images = [cv2.cvtColor(crops[i], cv2.COLOR_BGR2RGB) for i in usable]
        with torch.no_grad():
            batch = self._processor(images=images, return_tensors="pt")
            pixel_values = batch["pixel_values"].to(self._device)
            result = self._model(pixel_values=pixel_values)
            features = torch.cat(
                [result.pooler_output, result.last_hidden_state.mean(dim=1)],
                dim=-1,
            )
            features = features / features.norm(dim=-1, keepdim=True)
        matrix = features.float().cpu().numpy()
        if matrix.shape[1] != self._weights.shape[0]:
            raise ValueError(
                f"verifier coefficients are {self._weights.shape[0]}-d and the "
                f"backbone produced {matrix.shape[1]}-d features"
            )
        logits = matrix @ self._weights + self._bias
        for slot, value in zip(usable, 1.0 / (1.0 + np.exp(-logits))):
            out[slot] = float(value)
        return out


_SHARED_VERIFIER = None


def shared_verifier() -> FaceVerifier:
    """The process's one verifier. Loaded on first ask, then reused."""
    global _SHARED_VERIFIER
    if _SHARED_VERIFIER is None:
        _SHARED_VERIFIER = FaceVerifier()
    return _SHARED_VERIFIER


def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx0, by0, bx1, by1 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    inter = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(
        0.0, min(ay1, by1) - max(ay0, by0)
    )
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


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

    def __init__(self, path=None, verifier=None) -> None:
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
        # Decided ONCE, here, so a session's label cannot change halfway
        # through it. `verifier=False` is the explicit "do not verify".
        if verifier is False:
            self._verifier = None
        elif verifier is not None:
            self._verifier = verifier if verifier.available else None
        else:
            shared = shared_verifier()
            self._verifier = shared if shared.available else None
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

        Names the detector, its threshold AND the plausibility gate rather
        than asserting an outcome. "redacted" alone would invite a reader to
        infer completeness the detector cannot support.

        The gate is in the label because it changes what was filled. A reader
        who saw only `yunet-2023mar@0.30` would look up that detector's recall
        and over-estimate this imagery; the suffix tells them a filter ran
        after it, and which one.
        """
        if not self.available:
            return REDACTION_NONE
        gate = (PLAUSIBILITY_ID if self._verifier is not None
                else PLAUSIBILITY_WITHOUT_VERIFIER)
        return (
            f"faces-detected-and-filled/{DETECTOR_ID}@{CONFIDENCE:.2f}"
            f"+{gate}"
        )

    @property
    def verifies(self) -> bool:
        """Whether the fourth test -- the one that is not YuNet -- is running."""
        return self._verifier is not None

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

    def _raw_detect(self, image, upscale) -> list:
        """(box, landmarks) pairs in ORIGINAL image coordinates.

        The upscale is a parameter because the corroboration passes below look
        at the same frame at other scales -- including below 1 -- and every
        pass has to come back in the same coordinates to be compared.
        """
        import cv2

        height, width = image.shape[:2]
        if upscale == 1:
            scaled = image
        else:
            target = (max(1, int(round(width * upscale))),
                      max(1, int(round(height * upscale))))
            scaled = cv2.resize(
                image,
                target,
                # Area averaging to shrink: cubic aliases, and aliasing is
                # exactly the texture coincidence this pass exists to dissolve.
                interpolation=cv2.INTER_CUBIC if upscale > 1 else cv2.INTER_AREA,
            )
        size = (scaled.shape[1], scaled.shape[0])
        scale_x = size[0] / float(width)
        scale_y = size[1] / float(height)
        # One detector, re-targeted. Building a second one per frame was
        # measured at 8 ms and this runs on every keyframe.
        if self._detector is None:
            self._detector = cv2.FaceDetectorYN.create(
                str(self._path), "", size, CONFIDENCE, NMS_THRESHOLD, TOP_K
            )
            self._size = size
        elif size != self._size:
            # DAT's adaptive ladder can change resolution mid-stream, and the
            # corroboration pass changes it deliberately.
            self._detector.setInputSize(size)
            self._size = size

        _, faces = self._detector.detect(scaled)
        if faces is None:
            return []
        out = []
        for face in faces:
            box = (float(face[0]) / scale_x, float(face[1]) / scale_y,
                   float(face[2]) / scale_x, float(face[3]) / scale_y)
            landmarks = []
            for i in range(4, 14, 2):
                landmarks += [float(face[i]) / scale_x, float(face[i + 1]) / scale_y]
            out.append((box, landmarks))
        return out

    def _detect(self, image) -> list:
        """The boxes to fill: detected, judged plausible, then dilated.

        Order matters. The plausibility tests run on the RAW box, because the
        head dilation is a deliberate over-reach and a test applied after it
        would be measuring the over-reach rather than the detection.
        """
        # (box, scales). Everything not in this list was refused by a MEASURED
        # landmark verdict; everything in it is filled unless `scales` is
        # non-empty and the detector finds it again at none of them.
        candidates = []
        for box, landmarks in self._raw_detect(image, UPSCALE):
            area = box_area_fraction(box, image.shape)
            if area < LANDMARK_TEST_ABOVE_AREA:
                candidates.append((box, ()))
                continue
            verdict = landmark_verdict(box, landmarks)
            if verdict is None:
                # Degenerate or non-finite landmarks: no judgement was made, so
                # nothing may be dropped on the strength of one. Fill.
                candidates.append((box, ()))
            elif area > LARGE_BOX_AREA_FRACTION:
                if verdict and box_near_frame_edge(box, image.shape):
                    # A close face cut by the frame edge can be found at no
                    # other scale (EDGE_MARGIN): facelike is enough here.
                    candidates.append((box, ()))
                else:
                    # The landmark verdict carries no evidence at this size;
                    # only being found again at some other resolution does.
                    candidates.append((box, LARGE_BOX_CORROBORATION_SCALES))
            elif not verdict:
                continue
            elif area >= CORROBORATE_ABOVE_AREA:
                candidates.append((box, (1.0,)))
            else:
                candidates.append((box, ()))
        if not candidates:
            return []

        kept = self._verify(image, self._corroborate(image, candidates))

        boxes = []
        import math

        height, width = image.shape[:2]
        for x, y, w, h in kept:
            if not all(math.isfinite(float(v)) for v in (x, y, w, h)):
                # A detection whose box is not a number cannot be located, only
                # believed. Fill the frame rather than raise, which would
                # persist the ORIGINAL bytes.
                boxes.append((0.0, 0.0, float(width), float(height)))
                continue
            # Dilate about the centre: a face box is not a head.
            cx, cy = x + w / 2.0, y + h / 2.0
            w *= HEAD_DILATION
            h *= HEAD_DILATION
            boxes.append((cx - w / 2.0, cy - h / 2.0, w, h))
        return boxes

    def _corroborate(self, image, candidates: list) -> list:
        """Ask again, at other resolutions, about the boxes that need it.

        `candidates` is (box, scales). A box with no scales -- small enough to
        be a distant bystander, or unjudgeable -- is kept without a second pass
        being run. Otherwise it is kept if ANY of its scales finds it again.
        Each scale is run at most once per frame, and only if some box asks.

        FAILS TOWARDS FILLING. If a pass cannot be run at all -- a detector
        that throws, a resize that fails -- every box that asked for that scale
        is kept. The cost of an unnecessary fill is pixels; the cost of a
        skipped one is a face on disk, and those are not the same kind of
        mistake.
        """
        passes = {}

        def found_at(box, scale):
            if scale not in passes:
                try:
                    # This leaves the detector sized for another scale; the
                    # next call at UPSCALE sees a size change and re-targets,
                    # which is the same path DAT's ladder already takes.
                    passes[scale] = self._raw_detect(image, scale)
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "[Tower][Redaction] corroboration pass at scale %s "
                        "failed; keeping the detection", scale,
                    )
                    passes[scale] = None
            second = passes[scale]
            if second is None:
                return True
            return any(_iou(box, other) >= CORROBORATION_IOU for other, _ in second)

        survivors = []
        for box, scales in candidates:
            if not scales or any(found_at(box, s) for s in scales):
                survivors.append(box)
        return survivors

    def _verify(self, image, boxes: list) -> list:
        """Drop a small box a general vision model says is not a face.

        Runs on the RAW boxes, before the head dilation, like every other
        test, and only on boxes at most `below_area` of the frame.

        FAILS TOWARDS FILLING, in every direction: no verifier, a box too big
        to judge, a crop that cannot be cut, a score that is not a number, a
        forward pass that raises. Each of those keeps the box. The only way a
        box is dropped here is a finite score below the threshold.
        """
        import math

        if self._verifier is None or not boxes:
            return boxes
        judged = [i for i, b in enumerate(boxes)
                  if box_area_fraction(b, image.shape) <= self._verifier.below_area]
        if not judged:
            return boxes
        try:
            scores = self._verifier.scores(image, [boxes[i] for i in judged])
        except Exception:  # noqa: BLE001
            logger.warning(
                "[Tower][Redaction] the face verifier failed on this frame; "
                "keeping every detection"
            )
            return boxes
        drop = set()
        for slot, score in zip(judged, scores):
            if score is None or not math.isfinite(float(score)):
                continue
            if float(score) < self._verifier.threshold:
                drop.add(slot)
        return [b for i, b in enumerate(boxes) if i not in drop]
