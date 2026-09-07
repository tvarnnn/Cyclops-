"""Is this the page I already have? Both witnesses must agree.

The failure modes the mission names, each pinned: the same frame twice,
the same page shifted, the same page under perspective, the same page
revisited later, two pages that look alike but differ, and the same
paragraph in two documents.
"""

import cv2
import numpy as np

from tests import document_fixtures as fx
from tower.document_memory.identity import (
    SAME_LOOK_MAX_DISTANCE,
    SAME_PAGE_TOKEN_OVERLAP,
    Reading,
    compare,
    containment,
    hash_distance,
    numbers_agree,
    perceptual_hash,
    token_overlap,
)

PAPER = fx.page_text(fx.TRANSFORMER_PAPER)
NOTES = fx.page_text(fx.DEPTH_NOTES)
RECEIPT = fx.page_text(fx.RECEIPT)


def _crop(lines, tilt=0.0, shift=(0, 0), blur=0, gain=1.0):
    page = fx.render_page(lines)
    frame, corners = fx.place_page(page, frame_size=(720, 1280), tilt=tilt)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if shift != (0, 0):
        matrix = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
        gray = cv2.warpAffine(gray, matrix, (gray.shape[1], gray.shape[0]))
        corners = corners + np.float32(shift)
    if blur:
        gray = cv2.GaussianBlur(gray, (blur, blur), 0)
    if gain != 1.0:
        gray = np.clip(gray.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    x0, y0 = corners.min(axis=0).astype(int)
    x1, y1 = corners.max(axis=0).astype(int)
    return gray[max(y0, 0) : y1, max(x0, 0) : x1]


def _reading(text, crop=None, confidence=0.9):
    return Reading(text, confidence, None if crop is None else perceptual_hash(crop))


class TestThePerceptualHash:
    def test_sixteen_hex_characters(self):
        digest = perceptual_hash(_crop(fx.TRANSFORMER_PAPER))

        assert len(digest) == 16
        int(digest, 16)

    def test_the_same_page_hashes_alike_under_shift_blur_and_exposure(self):
        reference = perceptual_hash(_crop(fx.TRANSFORMER_PAPER))
        for variant in (
            _crop(fx.TRANSFORMER_PAPER, shift=(12, 5)),
            _crop(fx.TRANSFORMER_PAPER, blur=5),
            _crop(fx.TRANSFORMER_PAPER, gain=0.75),
            _crop(fx.TRANSFORMER_PAPER, tilt=0.3),
        ):
            assert hash_distance(reference, perceptual_hash(variant)) <= SAME_LOOK_MAX_DISTANCE

    def test_different_prose_hashes_apart(self):
        paper = perceptual_hash(_crop(fx.TRANSFORMER_PAPER))
        notes = perceptual_hash(_crop(fx.DEPTH_NOTES))

        assert hash_distance(paper, notes) > SAME_LOOK_MAX_DISTANCE

    def test_a_missing_hash_has_no_distance(self):
        assert hash_distance(None, "0" * 16) is None
        assert perceptual_hash(np.zeros((0, 0), np.uint8)) is None


class TestTheTextWitness:
    def test_identical_text_overlaps_completely(self):
        assert token_overlap(PAPER, PAPER) == 1.0

    def test_unrelated_text_barely_overlaps(self):
        assert token_overlap(PAPER, NOTES) < 0.25

    def test_a_partial_view_is_contained_even_when_overlap_is_low(self):
        half = " ".join(PAPER.split()[: len(PAPER.split()) // 2])

        assert containment(PAPER, half) > 0.95
        assert token_overlap(PAPER, half) < SAME_PAGE_TOKEN_OVERLAP

    def test_numbers_veto_a_template_twin(self):
        first = "Invoice 1042 total 17.49 due 2026-09-30 ref 88213"
        second = "Invoice 1043 total 22.10 due 2026-10-15 ref 88214"

        assert numbers_agree(first, first)
        assert not numbers_agree(first, second)

    def test_too_few_numbers_do_not_vote(self):
        assert numbers_agree("page 1 of the notes", "page 2 of the notes")


class TestTheVerdict:
    def test_the_same_frame_twice_is_the_same_page(self):
        crop = _crop(fx.TRANSFORMER_PAPER)

        assert compare(_reading(PAPER, crop), _reading(PAPER, crop)).same

    def test_the_same_page_shifted_and_re_read_is_the_same_page(self):
        """OCR of a re-view drops and splits a few words; the overlap
        stays well above the threshold and the look agrees."""
        words = PAPER.split()
        noisy = " ".join(words[:-3] + ["dispensing", "wlth", "recurrance"])

        verdict = compare(
            _reading(PAPER, _crop(fx.TRANSFORMER_PAPER)),
            _reading(noisy, _crop(fx.TRANSFORMER_PAPER, shift=(15, 8))),
        )

        assert verdict.same
        assert verdict.text_overlap >= SAME_PAGE_TOKEN_OVERLAP

    def test_the_same_page_under_perspective_is_the_same_page(self):
        verdict = compare(
            _reading(PAPER, _crop(fx.TRANSFORMER_PAPER)),
            _reading(PAPER, _crop(fx.TRANSFORMER_PAPER, tilt=0.4)),
        )

        assert verdict.same

    def test_different_pages_that_look_alike_are_not_merged(self):
        """Two invoices on one template: the picture agrees, the numbers do not."""
        first = "Invoice 1042 total 17.49 due 2026-09-30 ref 88213 hardware store"
        second = "Invoice 1043 total 22.10 due 2026-10-15 ref 88214 hardware store"
        crop = _crop(fx.RECEIPT)

        verdict = compare(_reading(first, crop), _reading(second, crop))

        assert not verdict.same
        assert verdict.numbers_agree is False

    def test_the_same_paragraph_in_two_documents_is_not_merged(self):
        """The words agree; the pages do not look alike. Kept separate,
        because a false merge destroys a page."""
        verdict = compare(
            _reading(PAPER, _crop(fx.TRANSFORMER_PAPER)),
            _reading(PAPER, _crop(fx.DEPTH_NOTES)),
        )

        assert not verdict.same
        assert verdict.decision == "shares-text"

    def test_a_page_seen_again_minutes_later_is_the_same_page(self):
        """Time does not enter the comparison at all: a revisit is a
        re-view with a different capture id, which is provenance, not
        identity."""
        verdict = compare(
            _reading(RECEIPT, _crop(fx.RECEIPT)),
            _reading(RECEIPT, _crop(fx.RECEIPT, shift=(4, 2), gain=1.1)),
        )

        assert verdict.same

    def test_an_unreadable_reading_abstains(self):
        crop = _crop(fx.TRANSFORMER_PAPER)

        verdict = compare(_reading(PAPER, crop), _reading("", crop, confidence=None))

        assert verdict.decision == "abstain"
        assert not verdict.same

    def test_a_low_confidence_reading_abstains(self):
        crop = _crop(fx.TRANSFORMER_PAPER)

        verdict = compare(_reading(PAPER, crop), _reading(PAPER, crop, confidence=0.1))

        assert verdict.decision == "abstain"

    def test_words_alone_may_merge_when_no_hash_exists(self):
        """A record written before hashes existed still deduplicates on
        its words -- but only on a strong overlap."""
        verdict = compare(_reading(PAPER), _reading(PAPER))

        assert verdict.same
        assert verdict.hash_distance is None

    def test_a_partial_view_of_a_page_is_the_same_page_by_containment(self):
        half = " ".join(PAPER.split()[: int(len(PAPER.split()) * 0.6)])
        verdict = compare(_reading(PAPER), _reading(half))

        assert verdict.same
        assert verdict.text_containment >= 0.85
