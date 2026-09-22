"""Is this frame worth looking at, and where is the text in it?

Two stages, and the split is the whole design.

**Stage 1 runs on every frame and costs ~3 ms.** It answers one narrow
question -- is the camera steady and the image sharp -- with a phase
correlation against the previous frame and a variance-of-Laplacian
sharpness with a rolling baseline. It says nothing about text, on
purpose: the previous gate (`detect.py`) tried to prove "this is a page
of text" from 360x640 pixels on every frame with a contour finder and a
glyph statistic, and on 9,199 real frames it fired six times, all on
blinds and keyboards, and zero times after re-derivation. Proving text
from pixels is a text detector's job.

**Stage 2 runs on stable frames, rate-limited, and costs ~35 ms on the
GPU.** It is the text DETECTOR half of the OCR engine -- CRAFT, in
EasyOCR's case -- which answers "is there text and where" in a fraction
of what recognition costs. Measured 2026-09-06 on an RTX 5070: 31-47 ms
at 360x640, 55-66 ms at 504x896, zero boxes on a noise frame, and the
recogniser reads a page from those boxes at 0.98 word recall when the
page fills the frame. No quadrilateral is required: a partial page, a
screen, a book with its spine in view or a sheet under a hand all yield
text boxes and none reliably yields four convex corners, which is why
4.9% was the quad survival rate on real footage.

The region a dwell tracks is the union of those boxes. That is a weaker
geometric claim than a quad and a stronger evidential one: it is where
a text detector found text, not where an edge finder found a rectangle.

What this file does NOT do: measure attention. `07-PLATFORM-CONSTRAINTS.md`
Limitation 8 stands. A steady, sharp frame with text in it is a frame
worth reading; it is not evidence that the wearer read anything.
"""

import logging
from collections import deque
from dataclasses import dataclass
from statistics import median

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GatePolicy:
    """Every stage-1 threshold, in one place, each with its reason.

    A value object rather than module constants so a benchmark can sweep
    them and a test can pin them; a threshold that cannot be swept
    cannot be chosen from data.
    """

    # Width the frame is downscaled to for phase correlation. 180 px is
    # enough to resolve a 1-px shift at 360 wide and keeps the FFT under
    # a millisecond.
    motion_probe_width: int = 180
    # A frame is steady when it moved less than this fraction of the
    # frame diagonal since the previous one: 7 px at 360x640, 10 at
    # 504x896. Measured as a fraction so the same policy means the same
    # thing at every rung the glasses can deliver.
    max_shift_fraction: float = 0.01
    # Phase-correlation response below which two consecutive frames are
    # not the same picture. A pure shift keeps it above 0.9 through 32
    # px; a page turned in place without the camera moving drops it to
    # ~0.24 (synthetic). 0.6 leaves room for real sensor noise.
    min_response: float = 0.6
    # And below THIS, with a small shift, the content changed while the
    # camera held still -- the page-turn signal a quad tracker lacks.
    content_change_response: float = 0.45
    # Absolute sharpness floor, variance of the Laplacian over the whole
    # frame. World Builder's keyframe selector uses 25 at 360x640,
    # measured on a real walk; it is a sanity check, not the selector.
    min_sharpness: float = 25.0
    # The real blur gate: sharpness relative to the rolling median of
    # recent frames. A blink of motion blur in a steady hold reads as a
    # tenth of its neighbours; a page held in dim light reads as low in
    # absolute terms and normal relative to itself. World Builder's
    # `min_sharpness_ratio` is 0.55, from the same walk.
    min_sharpness_ratio: float = 0.55
    sharpness_window: int = 30


@dataclass(frozen=True)
class GateVerdict:
    """What stage 1 concluded about one frame."""

    stable: bool
    sharpness: float
    sharpness_ratio: float
    shift_px: float
    response: float
    # True when the camera held still and the picture changed anyway.
    content_changed: bool


def measure_sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian, via a 16-bit pass and `meanStdDev`.

    The same computation `detect.py` made with `CV_64F` and `.var()`,
    at a fifth of the cost: 0.19 ms against ~1 ms at 360x640. Exact to
    rounding.
    """
    if gray is None or gray.size == 0:
        return 0.0
    laplacian = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    _mean, std = cv2.meanStdDev(laplacian)
    return float(std[0][0] ** 2)


class FrameGate:
    """Stage 1. Steady and sharp, or not; nothing about text."""

    def __init__(self, policy: GatePolicy | None = None) -> None:
        self._policy = policy or GatePolicy()
        self._previous: np.ndarray | None = None
        self._window: np.ndarray | None = None
        self._shape: tuple | None = None
        self._probe_scale = 1.0
        self._recent = deque(maxlen=self._policy.sharpness_window)

    @property
    def policy(self) -> GatePolicy:
        return self._policy

    def reset(self) -> None:
        self._previous = None
        self._recent.clear()

    def observe(self, gray: np.ndarray) -> GateVerdict:
        policy = self._policy
        shape = tuple(gray.shape[:2])
        if shape != self._shape:
            # A rung change. Regions and shifts are in pixels of a frame
            # that no longer exists; start over rather than compare.
            self._shape = shape
            self._previous = None
            self._recent.clear()
        sharpness = measure_sharpness(gray)
        baseline = median(self._recent) if self._recent else sharpness
        self._recent.append(sharpness)
        ratio = sharpness / baseline if baseline > 0 else 1.0

        probe = self._probe(gray)
        shift_px, response = self._motion(probe)
        self._previous = probe

        height, width = gray.shape[:2]
        diagonal = float(np.hypot(height, width))
        small_shift = shift_px < policy.max_shift_fraction * diagonal
        # A sharp frame that does not correlate with the last one is a
        # picture that changed. The shift estimate is meaningless when
        # the content differs -- the peak lands anywhere -- so it is not
        # consulted here; sharpness is, because a frame smeared by motion
        # also fails to correlate and is a different case.
        content_changed = (
            response < policy.content_change_response
            and sharpness >= policy.min_sharpness
            and ratio >= policy.min_sharpness_ratio
        )
        stable = (
            small_shift
            and response >= policy.min_response
            and sharpness >= policy.min_sharpness
            and ratio >= policy.min_sharpness_ratio
        )
        return GateVerdict(
            stable=stable,
            sharpness=sharpness,
            sharpness_ratio=float(ratio),
            shift_px=float(shift_px),
            response=float(response),
            content_changed=bool(content_changed),
        )

    def _probe(self, gray: np.ndarray) -> np.ndarray:
        height, width = gray.shape[:2]
        target_width = min(self._policy.motion_probe_width, width)
        target_height = max(1, int(round(height * target_width / width)))
        small = cv2.resize(
            gray, (target_width, target_height), interpolation=cv2.INTER_AREA
        )
        probe = np.float32(small)
        # The probe is `target_width` wide; a shift measured on it scales
        # back up to full-frame pixels by this factor.
        self._probe_scale = width / float(target_width)
        if self._window is None or self._window.shape != probe.shape:
            self._window = cv2.createHanningWindow(
                (probe.shape[1], probe.shape[0]), cv2.CV_32F
            )
        return probe

    def _motion(self, probe: np.ndarray) -> tuple[float, float]:
        """(shift in FULL-frame pixels, response) against the previous frame."""
        previous = self._previous
        if previous is None or previous.shape != probe.shape:
            # First frame, or a rung change: nothing to compare against.
            # Reported as "moved a lot" so a dwell cannot start on a frame
            # whose steadiness nobody measured.
            return float("inf"), 0.0
        (dx, dy), response = cv2.phaseCorrelate(previous, probe, self._window)
        return float(np.hypot(dx, dy) * self._probe_scale), float(response)


# -- stage 2: the text region -------------------------------------------


@dataclass(frozen=True)
class RegionPolicy:
    """When a set of text boxes counts as a page worth tracking."""

    # Fewer boxes than this is a label, a clock or a logo, not a page.
    min_boxes: int = 3
    # Median box height in pixels. Body text at 360x640 is ~8 px tall;
    # EasyOCR reads it at that height when the page fills the frame and
    # not at 5. Below this the recogniser would produce noise, and a
    # region that cannot be read is not worth a dwell.
    min_median_box_height_px: float = 7.0
    # The union of the boxes must be a real portion of the view.
    min_area_fraction: float = 0.03
    # Padding around the union, as a fraction of its own size, so a
    # crop does not clip descenders and margins.
    padding_fraction: float = 0.08
    # The detector is run at most this often on a steady view; between
    # runs the last region is carried forward. 4 Hz at ~35 ms is ~14%
    # of the worker's time at 12 fps.
    detect_interval_s: float = 0.25
    # A floor under `detect_interval_s`, so a policy that sets the
    # interval to zero to "detect everything" still cannot spend the
    # detector on every frame of a 12 fps stream. A steady view of a
    # screen with busy text is the expensive case: the detector's own
    # post-processing grows with the number of boxes it finds.
    min_detect_interval_s: float = 0.1
    # And a region older than this is not carried forward at all.
    carry_forward_s: float = 1.5


@dataclass(frozen=True)
class RegionCandidate:
    """Where the text is in one frame, and how good a look it is.

    The interface the dwell tracker consumes: `corners`, `centre`,
    `area_fraction`, `sharpness`, `squareness`. The rest is evidence.
    """

    corners: np.ndarray  # (4, 2) float32, clockwise from top-left
    area_fraction: float
    sharpness: float
    box_count: int
    median_box_height: float
    coverage: float
    squareness: float = 1.0
    source: str = "text-detector"

    @property
    def centre(self) -> tuple[float, float]:
        return (
            float(self.corners[:, 0].mean()),
            float(self.corners[:, 1].mean()),
        )

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        """x0, y0, x1, y1 as ints, for cropping."""
        xs = self.corners[:, 0]
        ys = self.corners[:, 1]
        return int(xs.min()), int(ys.min()), int(np.ceil(xs.max())), int(np.ceil(ys.max()))


def crop_region(gray: np.ndarray, candidate) -> np.ndarray:
    """The region's pixels, clipped to the frame. Never empty.

    Works from `corners` alone, so a legacy quad candidate from
    `detect.py` crops exactly as a text region does.
    """
    height, width = gray.shape[:2]
    corners = np.asarray(candidate.corners, dtype=np.float32).reshape(-1, 2)
    xs, ys = corners[:, 0], corners[:, 1]
    x0, y0 = int(np.floor(xs.min())), int(np.floor(ys.min()))
    x1, y1 = int(np.ceil(xs.max())), int(np.ceil(ys.max()))
    x0 = max(0, min(x0, width - 1))
    y0 = max(0, min(y0, height - 1))
    x1 = max(x0 + 1, min(x1, width))
    y1 = max(y0 + 1, min(y1, height))
    return gray[y0:y1, x0:x1]


def region_from_boxes(
    boxes, frame_shape, *, policy: RegionPolicy | None = None, sharpness=None
) -> RegionCandidate | None:
    """The union of the detector's boxes as one region, or None.

    `sharpness` is supplied by the caller when it has already been
    measured on the crop; otherwise the candidate carries 0 and the
    caller is expected to fill it in.
    """
    policy = policy or RegionPolicy()
    boxes = [box for box in boxes if box.height > 0 and box.area > 0]
    if len(boxes) < policy.min_boxes:
        return None
    heights = sorted(box.height for box in boxes)
    median_height = heights[len(heights) // 2]
    if median_height < policy.min_median_box_height_px:
        return None

    height, width = frame_shape[:2]
    x0 = min(box.x0 for box in boxes)
    y0 = min(box.y0 for box in boxes)
    x1 = max(box.x1 for box in boxes)
    y1 = max(box.y1 for box in boxes)
    pad_x = (x1 - x0) * policy.padding_fraction
    pad_y = (y1 - y0) * policy.padding_fraction
    x0 = max(0.0, x0 - pad_x)
    y0 = max(0.0, y0 - pad_y)
    x1 = min(float(width), x1 + pad_x)
    y1 = min(float(height), y1 + pad_y)
    union_area = max(x1 - x0, 1.0) * max(y1 - y0, 1.0)
    area_fraction = union_area / float(height * width)
    if area_fraction < policy.min_area_fraction:
        return None

    corners = np.array(
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32
    )
    coverage = min(1.0, sum(box.area for box in boxes) / union_area)
    return RegionCandidate(
        corners=corners,
        area_fraction=float(area_fraction),
        sharpness=float(sharpness or 0.0),
        box_count=len(boxes),
        median_box_height=float(median_height),
        coverage=float(coverage),
    )


class PageFinder:
    """Stage 1 and stage 2 together: one frame in, a region or None out.

    `detector` is anything with `detect(gray) -> boxes`; in production
    it is the OCR recogniser itself, so the detector and the reader are
    one model loaded once. In tests it is `classical_text_boxes`.
    """

    def __init__(
        self,
        detector,
        *,
        gate_policy: GatePolicy | None = None,
        region_policy: RegionPolicy | None = None,
    ) -> None:
        self._detector = detector
        self._gate = FrameGate(gate_policy)
        self._policy = region_policy or RegionPolicy()
        self._last_region: RegionCandidate | None = None
        self._last_detect_at: float | None = None
        self._detections = 0
        self._last_verdict: GateVerdict | None = None

    @property
    def detections(self) -> int:
        return self._detections

    @property
    def last_verdict(self) -> GateVerdict | None:
        return self._last_verdict

    def reset(self) -> None:
        self._gate.reset()
        self._last_region = None
        self._last_detect_at = None

    def find(self, gray: np.ndarray, *, at: float) -> RegionCandidate | None:
        verdict = self._gate.observe(gray)
        self._last_verdict = verdict
        if not verdict.stable:
            # An unsteady or blurred frame is not a miss of the PAGE; the
            # dwell tracker tolerates a bounded run of these. It is a
            # frame the detector should not be spent on.
            return None

        # On a cadence, never per frame. A content change does NOT ask
        # for an early re-run: a frame whose content changed is by
        # definition not steady, so it never reaches here, and the page
        # turn it signals is the dwell tracker's business (it compares
        # the region against the segment's own reference).
        since = None if self._last_detect_at is None else at - self._last_detect_at
        due = since is None or since >= max(
            self._policy.detect_interval_s, self._policy.min_detect_interval_s
        )
        if due:
            region = self._detect(gray)
            self._last_detect_at = at
            self._last_region = region
        else:
            region = self._carry_forward(gray, at)
        if region is None:
            return None
        return self._with_sharpness(region, gray, content_changed=verdict.content_changed)

    def _detect(self, gray) -> RegionCandidate | None:
        self._detections += 1
        # A recogniser without a detector stage -- a test double written
        # before one existed -- gets the classical one. Production
        # recognisers have `detect`; the fallback is not a silent
        # downgrade of anything a person deployed.
        detect = getattr(self._detector, "detect", None) or classical_text_boxes
        try:
            boxes = detect(gray)
        except Exception:
            logger.exception(
                "document memory: the text detector failed on a frame; "
                "treating it as no text"
            )
            return None
        return region_from_boxes(boxes, gray.shape, policy=self._policy)

    def _carry_forward(self, gray, at) -> RegionCandidate | None:
        if self._last_region is None or self._last_detect_at is None:
            return None
        if at - self._last_detect_at > self._policy.carry_forward_s:
            return None
        return self._last_region

    @staticmethod
    def _with_sharpness(region, gray, *, content_changed: bool) -> RegionCandidate:
        from dataclasses import replace

        crop = crop_region(gray, region)
        return replace(
            region,
            sharpness=measure_sharpness(crop),
            source="text-detector" if not content_changed else "text-detector/changed",
        )


# -- a detector that needs no model --------------------------------------


def classical_text_boxes(gray: np.ndarray):
    """Text-line boxes from morphology alone. The test suite's detector.

    A black-hat transform picks out dark marks smaller than its kernel
    -- glyphs -- and ignores the large dark-to-light edge of the page
    itself, which a plain gradient would fuse with the first line of
    text into one blob spanning the sheet. Otsu, then a wide horizontal
    closing so the glyphs of one line fuse; each blob's bounding box is
    one line. Deterministic, offline and free, which is what a default
    test suite needs. It is NOT the production detector and is not
    claimed to work on real footage: brick courses fuse into lines too,
    which is exactly why production uses a learned detector.
    """
    from tower.document_memory.ocr import TextBox

    if gray is None or gray.size == 0:
        return ()
    height, width = gray.shape[:2]
    if min(height, width) < 16:
        return ()
    glyph = max(9, height // 20) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (glyph, glyph))
    marks = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, binary = cv2.threshold(marks, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, width // 30), 1))
    fused = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, line_kernel)
    contours, _ = cv2.findContours(fused, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # A line of text is wider than it is tall and taller than a
        # scratch. Very tall blobs are edges of the page or the frame.
        if w < h * 2 or h < 4 or h > height * 0.25:
            continue
        # Reject blobs that are nearly empty inside: a page border.
        fill = float(np.count_nonzero(binary[y : y + h, x : x + w])) / float(w * h)
        if fill < 0.10:
            continue
        boxes.append(TextBox(float(x), float(y), float(x + w), float(y + h)))
    return tuple(boxes)


class ClassicalTextDetector:
    """`classical_text_boxes` behind the detector interface."""

    name = "classical"

    def detect(self, gray):
        return classical_text_boxes(gray)
