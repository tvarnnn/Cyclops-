"""The capture gate: steady and sharp, then where the text is.

Synthetic throughout. The page is rendered, so the answer -- is there
text, where is it, was the camera steady -- is known independently of
the code under test. No number here is a claim about real footage; the
real-footage claims live in the handoff, measured by replaying captures.
"""

import cv2
import numpy as np
import pytest

from tests import document_fixtures as fx
from tower.document_memory.gate import (
    ClassicalTextDetector,
    FrameGate,
    GatePolicy,
    PageFinder,
    RegionPolicy,
    classical_text_boxes,
    crop_region,
    measure_sharpness,
    region_from_boxes,
)
from tower.document_memory.ocr import TextBox

PORTRAIT = (360, 640)


def _gray(frame_bgr):
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)


def _page_gray(lines=fx.TRANSFORMER_PAPER, frame_size=PORTRAIT, corners=None):
    frame, _ = fx.place_page(fx.render_page(lines), frame_size=frame_size, corners=corners)
    return _gray(frame)


def _shifted(gray, dx: int, dy: int = 0):
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(gray, matrix, (gray.shape[1], gray.shape[0]), borderMode=cv2.BORDER_REPLICATE)


class TestSharpness:
    def test_blur_lowers_it_monotonically(self):
        gray = _page_gray()
        sharp = measure_sharpness(gray)
        soft = measure_sharpness(cv2.GaussianBlur(gray, (5, 5), 0))
        softer = measure_sharpness(cv2.GaussianBlur(gray, (15, 15), 0))

        assert sharp > soft > softer

    def test_an_empty_image_is_not_sharp(self):
        assert measure_sharpness(np.zeros((0, 0), np.uint8)) == 0.0


class TestTheSteadinessGate:
    def test_the_first_frame_is_never_stable(self):
        """Nothing precedes it to be steady against."""
        gate = FrameGate()

        verdict = gate.observe(_page_gray())

        assert verdict.stable is False
        assert verdict.shift_px == float("inf")

    def test_a_held_view_is_stable(self):
        gate = FrameGate()
        gray = _page_gray()
        gate.observe(gray)

        verdict = gate.observe(_shifted(gray, 1))

        assert verdict.stable is True
        assert verdict.shift_px < 3.0
        assert verdict.response > 0.6

    def test_a_large_move_is_not(self):
        gate = FrameGate()
        gray = _page_gray()
        gate.observe(gray)

        verdict = gate.observe(_shifted(gray, 40))

        assert verdict.stable is False
        assert verdict.shift_px > 20.0

    def test_the_shift_threshold_is_a_fraction_of_the_diagonal(self):
        """The same policy means the same thing at every rung."""
        policy = GatePolicy()
        small = _page_gray(frame_size=(360, 640))
        large = _page_gray(frame_size=(720, 1280))
        limit_small = policy.max_shift_fraction * np.hypot(*small.shape)
        limit_large = policy.max_shift_fraction * np.hypot(*large.shape)

        assert limit_large == pytest.approx(2 * limit_small)

    def test_a_blink_of_blur_in_a_steady_hold_is_rejected(self):
        """The rolling baseline is the real blur gate, not the floor."""
        gate = FrameGate()
        gray = _page_gray()
        for _ in range(8):
            gate.observe(gray)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)

        verdict = gate.observe(blurred)

        assert verdict.stable is False
        assert verdict.sharpness_ratio < GatePolicy().min_sharpness_ratio

    def test_a_page_turned_in_place_reads_as_a_content_change(self):
        """Same camera pose, different words: the response collapses."""
        gate = FrameGate()
        gate.observe(_page_gray(fx.TRANSFORMER_PAPER))

        verdict = gate.observe(_page_gray(fx.DEPTH_NOTES))

        assert verdict.content_changed is True
        assert verdict.stable is False

    def test_a_rung_change_resets_the_comparison(self):
        gate = FrameGate()
        gate.observe(_page_gray(frame_size=(360, 640)))

        verdict = gate.observe(_page_gray(frame_size=(504, 896)))

        assert verdict.stable is False


class TestTheRegionFromBoxes:
    def _boxes(self, count, height=12.0, x0=40.0, width=200.0, top=100.0, gap=8.0):
        return [
            TextBox(x0, top + index * (height + gap), x0 + width, top + index * (height + gap) + height)
            for index in range(count)
        ]

    def test_enough_tall_boxes_make_a_region(self):
        region = region_from_boxes(self._boxes(5), (640, 360))

        assert region is not None
        assert region.box_count == 5
        assert region.median_box_height == 12.0
        assert 0 < region.area_fraction < 1

    def test_too_few_boxes_is_a_label_not_a_page(self):
        assert region_from_boxes(self._boxes(2), (640, 360)) is None

    def test_tiny_text_is_not_worth_a_dwell(self):
        """Below ~7 px the recogniser returns noise; a region that cannot
        be read is not worth tracking."""
        assert region_from_boxes(self._boxes(6, height=4.0), (640, 360)) is None

    def test_the_region_is_padded_and_clipped_to_the_frame(self):
        region = region_from_boxes(self._boxes(4, x0=0.0, top=0.0), (640, 360))

        x0, y0, x1, y1 = region.bounds
        assert x0 == 0 and y0 == 0
        assert x1 <= 360 and y1 <= 640

    def test_a_region_below_the_area_floor_is_rejected(self):
        boxes = self._boxes(3, height=8.0, width=20.0, gap=1.0)

        assert region_from_boxes(boxes, (640, 360), policy=RegionPolicy(min_area_fraction=0.5)) is None

    def test_crop_never_returns_an_empty_image(self):
        gray = np.full((640, 360), 128, np.uint8)
        region = region_from_boxes(
            self._boxes(4, x0=350.0, width=100.0),
            (640, 360),
            policy=RegionPolicy(min_area_fraction=0.0),
        )

        crop = crop_region(gray, region)

        assert crop.size > 0


class TestTheClassicalDetector:
    """The suite's detector. Not production; must still tell text from not."""

    def test_it_finds_the_lines_of_a_rendered_page(self):
        boxes = classical_text_boxes(_page_gray())

        assert len(boxes) >= 6

    def test_it_finds_nothing_on_a_blank_page(self):
        assert region_from_boxes(classical_text_boxes(_gray(fx.blank_page_frame())), (480, 640)) is None

    def test_it_finds_nothing_in_a_textured_room(self):
        assert region_from_boxes(classical_text_boxes(_gray(fx.no_page_frame())), (480, 640)) is None

    def test_it_survives_a_tiny_image(self):
        assert classical_text_boxes(np.zeros((8, 8), np.uint8)) == ()


class TestThePageFinder:
    def _finder(self, **region):
        return PageFinder(ClassicalTextDetector(), region_policy=RegionPolicy(**region))

    def test_a_held_page_yields_a_region_after_the_first_frame(self):
        finder = self._finder()
        gray = _page_gray()

        assert finder.find(gray, at=0.0) is None
        region = finder.find(_shifted(gray, 1), at=0.1)

        assert region is not None
        assert region.box_count >= 6
        assert region.sharpness > 0

    def test_the_detector_is_rate_limited_and_the_region_carried_forward(self):
        finder = self._finder(detect_interval_s=0.5)
        gray = _page_gray()
        finder.find(gray, at=0.0)
        for index in range(1, 6):
            region = finder.find(_shifted(gray, index % 2), at=index * 0.08)
            assert region is not None

        # One detection at t=0.08 (the first stable frame); the next four
        # frames arrive inside the interval and reuse it.
        assert finder.detections == 1

    def test_a_stale_region_is_not_carried_forever(self):
        finder = self._finder(detect_interval_s=10.0, carry_forward_s=0.5)
        gray = _page_gray()
        finder.find(gray, at=0.0)
        assert finder.find(_shifted(gray, 1), at=0.1) is not None

        # Well past the carry-forward window but before the next
        # detection is due: no region, rather than a two-second-old one.
        assert finder.find(_shifted(gray, 1), at=2.0) is None

    def test_an_unsteady_frame_yields_nothing_and_costs_no_detection(self):
        finder = self._finder()
        gray = _page_gray()
        finder.find(gray, at=0.0)
        finder.find(_shifted(gray, 1), at=0.1)
        before = finder.detections

        assert finder.find(_shifted(gray, 60), at=0.2) is None
        assert finder.detections == before

    def test_a_room_with_no_text_yields_nothing(self):
        finder = self._finder()
        gray = _gray(fx.no_page_frame(frame_size=PORTRAIT))
        finder.find(gray, at=0.0)

        assert finder.find(gray, at=0.1) is None

    def test_a_detector_that_raises_is_a_miss_not_a_crash(self):
        class Broken:
            def detect(self, gray):
                raise RuntimeError("cuda hiccup")

        finder = PageFinder(Broken())
        gray = _page_gray()
        finder.find(gray, at=0.0)

        assert finder.find(_shifted(gray, 1), at=0.1) is None

    def test_a_recogniser_without_a_detector_gets_the_classical_one(self):
        class ReadOnly:
            def read(self, page_gray):  # pragma: no cover - never called here
                raise AssertionError

        finder = PageFinder(ReadOnly())
        gray = _page_gray()
        finder.find(gray, at=0.0)

        assert finder.find(_shifted(gray, 1), at=0.1) is not None
