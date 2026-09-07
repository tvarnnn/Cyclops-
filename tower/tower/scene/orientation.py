"""Does that person appear to be facing the wearer -- and never, are they looking.

`07-PLATFORM-CONSTRAINTS.md` Limitation 8: the camera cannot establish
that anyone looked at, noticed or read anything, and there is no eye
tracking on this hardware. What it CAN see is whether the front of a
person's head is visible, and that is the whole claim this module makes.

So the state is `toward_wearer`, the property is `appears_facing_wearer`,
and there is deliberately no value meaning "looking at you".

THE SIGNAL IS A FACE, VISIBLE, INSIDE A PERSON BOX

A face detector (`cv2.FaceDetectorYN`, the vendored YuNet model that
World Builder already uses for redaction) is run on the upper part of
each tracked person's box. A face found with a high score means the
front of that head is towards the camera; anything else -- no face, a
weak face, a box too small to hold one -- means the orientation is
**not established**, and is published as `unknown`.

There are deliberately only two answers. "Facing away" and "side-on" are
not produced, because nothing measured here can produce them with usable
precision (see below), and a positive claim with 0.02-0.34 precision is
worse than no claim.

WHY NOT KEYPOINTS. The first version of this module inferred facing from
which COCO keypoints a `keypointrcnn_resnet50_fpn` reported as visible
(both eyes and an ear -> toward, both ears and no eye -> away, one ear ->
profile). Validated on 2026-09-07 against 966 human-labelled persons
from COCO val2017 (`Glasses-scratch/scene-understanding-v1/orientation/`,
GT derived from the annotators' own visibility flags) it called 220 of
228 true-profile people `toward`: the keypoint model reports a confident
score and a plausible coordinate for an OCCLUDED eye, so score-thresholded
"visibility" does not track human visibility exactly where it matters.
`toward` precision was 0.56-0.62 across every threshold swept, and the
`away`/`profile` states were 0.02-0.34 precise -- wrong more often than
right. The same evaluation gave YuNet-in-person-box, at a face score of
0.9, **0.83 precision / 0.64 recall for `toward`**, at 9 ms per face on
CPU with no VRAM against 43 ms per frame on CUDA (956 ms on CPU) for the
keypoint model. A landmark-yaw refinement and a keypoint+face AND
combination were both measured and added nothing over raising YuNet's
own threshold.

WHAT THAT VALIDATION IS NOT. COCO stills are third-party photography:
front-lit, in focus, no motion blur, and 47% of eligible people face the
camera because photographers point cameras at faces. The precision above
is anchored to that base rate and is an UPPER BOUND for a glasses camera
in a room where most people are not looking at the wearer. The corpus on
this host contains no bystander at all (2026-09-07 audit, 45,594 frames),
so nothing here is validated on this camera. The feature ships as
EXPERIMENTAL, its confidence is capped at MEDIUM, and the wire says so.

PRIVACY. The face detector's output -- a box, a score and five landmark
coordinates -- is read, reduced to one boolean per person box, and
discarded. No crop is kept, no embedding is computed, and the score
itself never reaches a track or the wire. `test_scene_understanding_
persists_nothing` walks this file.

TEMPORAL VOTING. One frame's face detection is one vote. The engine
publishes `toward_wearer` for a track only when a majority of its last
`VOTE_WINDOW` estimates agree (see `vote`), so a single flash of a face
in one frame is not a claim, and a person turning away stops being
"facing" within two estimates rather than one. Every estimate still
carries its age and expires (`age_estimate`): a person who turned around
six seconds ago is not described by a six-second-old reading.
"""

import logging
from dataclasses import replace
from pathlib import Path
from typing import Protocol, runtime_checkable

from tower.confidence import Confidence
from tower.scene.records import (
    FACING_TOWARD,
    FACING_UNKNOWN,
    BoundingBox,
    FacingEstimate,
)

logger = logging.getLogger(__name__)

# The vendored YuNet weights. The same file World Builder's redaction
# loads, resolved the same way, so a Tower has one face model or none.
DEFAULT_MODEL_PATH = Path("models") / "face_detection_yunet_2023mar.onnx"

# A face scoring below this is not evidence the front of the head is
# visible. 0.9, from the 2026-09-07 sweep: 0.6 gives 0.63 precision, 0.8
# gives 0.69, 0.9 gives 0.83 -- and recall falls from 0.94 to 0.64 over
# the same range. The product says "facing your direction" out loud, so
# it buys precision with recall, and says `unknown` for the rest.
FACE_SCORE_THRESHOLD = 0.9

# YuNet's own NMS and candidate cap. The candidate cap is small because
# one person box holds at most a face or two.
NMS_THRESHOLD = 0.3
TOP_K = 50

# Only the top of a person box can hold that person's head. Cropping to
# it halves the pixels the detector sees and, more importantly, keeps a
# second person's face lower in a tall overlapping box from being
# counted as this one's.
HEAD_REGION_FRACTION = 0.6

# YuNet is trained for faces larger than a distant head at 360x640. A
# crop shorter than this is upscaled by `UPSCALE` before detection, which
# is what World Builder's redaction measured as the difference between
# finding and missing a 20-32 px face.
UPSCALE_BELOW_PX = 160
UPSCALE = 2

# A person box shorter than this cannot hold a face the detector can
# resolve even upscaled. Its orientation is unknown, and the estimate
# says why.
MIN_PERSON_HEIGHT_PX = 48

# How stale an estimate may be before it is reported as unknown rather
# than as an answer. Generous, because orientation is slow-moving -- but
# finite, because a person who turned around ten seconds ago is not
# described by a ten-second-old estimate. 6.0 s is ~24 cadence windows.
MAX_ESTIMATE_AGE_S = 6.0

# Temporal voting: `toward` needs at least `VOTE_NEEDED` of the last
# `VOTE_WINDOW` per-track estimates to be `toward`. 2 of 3 at the
# engine's ~250 ms cadence means a claim needs about half a second of
# agreeing evidence and drops about half a second after it stops.
VOTE_WINDOW = 3
VOTE_NEEDED = 2

# Why an estimate is what it is. On the estimate, never on the wire.
EVIDENCE_FACE = "face-visible"
EVIDENCE_NO_FACE = "no-face-found"
EVIDENCE_WEAK_FACE = "face-below-threshold"
EVIDENCE_TOO_SMALL = "box-too-small"
EVIDENCE_NONE = "none"


def estimate_from_face(
    face_score: float | None,
    *,
    too_small: bool = False,
    threshold: float = FACE_SCORE_THRESHOLD,
) -> FacingEstimate:
    """One person box's estimate, from whether a face was found in it.

    Confidence is deliberately never HIGH. This is a detector's score
    over a crop, two layers away from a measurement, validated on stills
    from a different camera; MEDIUM is the ceiling.
    """
    if too_small:
        return FacingEstimate(
            state=FACING_UNKNOWN,
            confidence=Confidence.UNKNOWN,
            evidence=EVIDENCE_TOO_SMALL,
        )
    if face_score is None:
        return FacingEstimate(
            state=FACING_UNKNOWN,
            confidence=Confidence.UNKNOWN,
            evidence=EVIDENCE_NO_FACE,
        )
    if face_score < threshold:
        return FacingEstimate(
            state=FACING_UNKNOWN,
            confidence=Confidence.UNKNOWN,
            evidence=EVIDENCE_WEAK_FACE,
        )
    return FacingEstimate(
        state=FACING_TOWARD,
        confidence=Confidence.MEDIUM,
        evidence=EVIDENCE_FACE,
    )


def vote(history: tuple, latest: FacingEstimate) -> FacingEstimate:
    """The estimate a track may publish, from its recent raw estimates.

    `history` is the track's last raw states, oldest first, NOT including
    `latest`. Returns `toward` if a majority of the window (latest
    included) is `toward`; otherwise an unknown estimate that keeps the
    latest evidence, so a consumer can still see why.

    A single `toward` in a window of unknowns is a flash, not a claim.
    """
    window = (*history, latest.state)[-VOTE_WINDOW:]
    if window.count(FACING_TOWARD) >= VOTE_NEEDED:
        return replace(latest, state=FACING_TOWARD, confidence=Confidence.MEDIUM)
    return replace(latest, state=FACING_UNKNOWN, confidence=Confidence.UNKNOWN)


def age_estimate(estimate: FacingEstimate, seconds: float) -> FacingEstimate:
    """Advance an estimate's age, expiring it once it is too old.

    An expired estimate becomes UNKNOWN rather than being deleted, so the
    consumer sees "we do not know" instead of a missing field it might
    read as "not facing".

    Clamped at zero, matching `Track.age_seconds`. Timestamps come from
    the capture journal and are wall clock: a backward NTP step produced a
    NEGATIVE age, which quietly pushed the expiry deadline further into
    the future -- the one direction it must never move. That clamp is a
    CLOCK guard, not a latency guard, and survives any change of model.
    """
    seconds = max(seconds, 0.0)
    if seconds > MAX_ESTIMATE_AGE_S:
        return FacingEstimate(
            state=FACING_UNKNOWN,
            confidence=Confidence.UNKNOWN,
            age_seconds=seconds,
            evidence=estimate.evidence,
        )
    return replace(estimate, age_seconds=seconds)


@runtime_checkable
class FacingEstimator(Protocol):
    """Anything that can say, per person box, whether a face is visible.

    `estimate` takes the frame and the boxes the TRACKER is asking about,
    and returns one `FacingEstimate` per box in the same order. The
    detector decides what exists; this stage only describes it. Boxes
    are asked for rather than discovered so two models cannot disagree
    about how many people there are.
    """

    name: str

    def load(self) -> None: ...

    def estimate(self, frame_bgr, boxes: list) -> list: ...

    def release(self) -> None: ...


class FixedFacingEstimator:
    """Returns the estimates the caller chose, matched to boxes by IoU.

    Each entry in `frames` is one call's answer: a list of
    `(BoundingBox, FacingEstimate)`. A requested box takes the estimate
    of the first entry overlapping it at IoU >= 0.25, else an unknown
    with `EVIDENCE_NO_FACE` -- the same shape the real estimator
    produces, so a test asserts against a state it wrote down rather
    than a model's opinion. The last entry repeats once exhausted.
    """

    name = "fixed"

    def __init__(self, frames=None) -> None:
        self._frames = [list(frame) for frame in (frames or [])]
        self.calls = 0
        self.asked: list = []

    def load(self) -> None:
        return None

    def estimate(self, frame_bgr, boxes: list) -> list:
        self.calls += 1
        self.asked.append(list(boxes))
        if not self._frames:
            answers = []
        else:
            answers = self._frames[min(self.calls - 1, len(self._frames) - 1)]
        out = []
        for box in boxes:
            found = None
            for answer_box, estimate in answers:
                if box.iou(answer_box) >= 0.25:
                    found = estimate
                    break
            out.append(found if found is not None else estimate_from_face(None))
        return out

    def release(self) -> None:
        return None


class FaceVisibilityEstimator:
    """YuNet, on the head region of each person box. CPU, ~9 ms a face.

    Loads lazily, from the vendored model file, and refuses to construct
    without it: a Tower with no face model has no orientation stage, and
    the engine reports `orientation_enabled: false` rather than guessing.
    """

    name = "yunet-face-visibility"

    def __init__(
        self,
        path=None,
        *,
        score_threshold: float = FACE_SCORE_THRESHOLD,
        min_person_height_px: int = MIN_PERSON_HEIGHT_PX,
    ) -> None:
        self._path = Path(path) if path is not None else DEFAULT_MODEL_PATH
        if not self._path.exists():
            raise FileNotFoundError(
                f"no face model at {self._path.as_posix()}; orientation "
                "needs the vendored YuNet weights"
            )
        self._score_threshold = score_threshold
        self._min_height = min_person_height_px
        self._detector = None
        self._size = None

    def load(self) -> None:
        import cv2

        # Constructed at a nominal size; every call re-sizes to its crop.
        # YuNet's own score floor is set low here and the threshold is
        # applied in `estimate_from_face`, so the constant that decides
        # the claim is the one this module documents.
        self._detector = cv2.FaceDetectorYN.create(
            str(self._path), "", (320, 320), 0.5, NMS_THRESHOLD, TOP_K
        )
        self._size = (320, 320)

    def estimate(self, frame_bgr, boxes: list) -> list:
        if self._detector is None:
            self.load()
        height, width = frame_bgr.shape[:2]
        return [self._one(frame_bgr, width, height, box) for box in boxes]

    def _one(self, frame_bgr, width: int, height: int, box: BoundingBox) -> FacingEstimate:
        import cv2

        x0 = int(max(box.x0, 0))
        x1 = int(min(box.x1, width))
        y0 = int(max(box.y0, 0))
        y1 = int(min(box.y0 + box.height * HEAD_REGION_FRACTION, height))
        if box.height < self._min_height or x1 - x0 < 8 or y1 - y0 < 8:
            return estimate_from_face(None, too_small=True)

        crop = frame_bgr[y0:y1, x0:x1]
        if crop.shape[0] < UPSCALE_BELOW_PX:
            crop = cv2.resize(
                crop,
                (crop.shape[1] * UPSCALE, crop.shape[0] * UPSCALE),
                interpolation=cv2.INTER_CUBIC,
            )
        size = (crop.shape[1], crop.shape[0])
        if size != self._size:
            self._detector.setInputSize(size)
            self._size = size

        _, faces = self._detector.detect(crop)
        if faces is None or len(faces) == 0:
            return estimate_from_face(None)
        # The strongest face in this head region, and only its score.
        # Landmarks and box are discarded here; nothing below this line
        # ever sees them.
        best = max(float(face[14]) for face in faces)
        return estimate_from_face(best, threshold=self._score_threshold)

    def release(self) -> None:
        self._detector = None
        self._size = None


def model_path() -> Path | None:
    """Where the face model is, or None. The same rule as redaction."""
    import os

    override = os.environ.get("TOWER_FACE_REDACTION_MODEL")
    if override:
        path = Path(override)
        return path if path.exists() else None
    return DEFAULT_MODEL_PATH if DEFAULT_MODEL_PATH.exists() else None
