"""The pipeline, end to end, against rendered pages with known text.

SYNTHETIC, NOT PHYSICAL.

`FixedTextRecogniser` stands in for OCR here so the suite stays fast,
offline and deterministic. It is not a mock of EasyOCR's behaviour -- it
returns the same SHAPE (regions with boxes, joined into text) so nothing
in the pipeline can come to depend on structure the real engine never
provides. OCR *accuracy* is measured separately, against the real engine,
in the opt-in integration test.

The word throughout is **observed**. The camera cannot establish that
anyone read anything (`07-PLATFORM-CONSTRAINTS.md` Limitation 8).
"""

import pytest

from tower.confidence import Confidence
from tower.document_memory.dwell import DwellPolicy
from tower.document_memory.engine import (
    DocumentMemoryEngine,
    is_same_page,
    token_overlap,
)
from tower.document_memory.ocr import FixedTextRecogniser, OcrResult, TextRegion
from tower.document_memory.records import (
    TIMING_ASSUMED_INTERVAL,
    TIMING_CAPTURE_JOURNAL,
)
from tower.document_memory.store import DocumentStore
from tests import document_fixtures as fx

POLICY = DwellPolicy(min_frames=3, min_seconds=0.6)


class _Clock:
    def __init__(self, start=1000.0, step=0.3):
        self.now = start
        self.step = step

    def __call__(self):
        return self.now

    def tick(self):
        self.now += self.step
        return self.now


@pytest.fixture
def store(tmp_path) -> DocumentStore:
    return DocumentStore(tmp_path)


def _run(store, frames, recogniser=None, clock=None, **kwargs):
    clock = clock or _Clock()
    recogniser = recogniser or FixedTextRecogniser(
        pages=[fx.page_regions(fx.TRANSFORMER_PAPER)]
    )
    engine = DocumentMemoryEngine(
        store, recogniser, policy=POLICY, clock=clock, **kwargs
    )
    for index, frame in enumerate(frames):
        engine.observe(frame, received_at=clock.tick(), source_seq=index)
    engine.flush()
    return engine


class TestItObservesADocument:
    def test_a_held_page_becomes_one_stored_document(self, store):
        _run(store, fx.document_frames(fx.TRANSFORMER_PAPER, 8))

        assert store.count() == 1

    def test_a_scene_with_no_page_stores_nothing(self, store):
        _run(store, [fx.encode(fx.no_page_frame())] * 20)

        assert store.count() == 0

    def test_a_blank_page_held_in_view_stores_nothing(self, store):
        """A whiteboard is not a document, however long it is in view."""
        _run(store, [fx.encode(fx.blank_page_frame())] * 20)

        assert store.count() == 0

    def test_a_single_glance_stores_nothing(self, store):
        """Two frames of a page is not a reading event."""
        _run(store, fx.document_frames(fx.TRANSFORMER_PAPER, 2))

        assert store.count() == 0

    def test_the_stored_text_is_what_the_recogniser_returned(self, store):
        _run(store, fx.document_frames(fx.TRANSFORMER_PAPER, 8))

        document = store.read_all()[0]

        assert "Transformer" in document.text
        assert document.title == "Attention Is All You Need"
        assert document.word_count > 20

    def test_ocr_runs_only_on_the_selected_frames(self, store):
        """The whole architecture in one assertion.

        OCR costs ~1.2 s a page. Twenty frames must not mean twenty
        seconds.
        """
        recogniser = FixedTextRecogniser(pages=[fx.page_regions(fx.RECEIPT)])
        _run(store, fx.document_frames(fx.RECEIPT, 20), recogniser=recogniser)

        assert recogniser.calls <= POLICY.best_frames
        # Nineteen, not twenty: the first frame of a stream has nothing
        # to be steady AGAINST, so the gate never admits it. Since
        # 2026-09-07 a dwell starts on the second frame at the earliest.
        assert store.read_all()[0].frames_considered >= 19


class TestItSeparatesAndDeduplicates:
    def test_two_documents_in_sequence_become_two_records(self, store):
        frames = (
            fx.document_frames(fx.TRANSFORMER_PAPER, 8)
            + [fx.encode(fx.no_page_frame())] * 6
            + fx.document_frames(fx.DEPTH_NOTES, 8)
        )
        recogniser = FixedTextRecogniser(
            pages=[
                fx.page_regions(fx.TRANSFORMER_PAPER),
                fx.page_regions(fx.TRANSFORMER_PAPER),
                fx.page_regions(fx.DEPTH_NOTES),
                fx.page_regions(fx.DEPTH_NOTES),
            ]
        )
        _run(store, frames, recogniser=recogniser)

        titles = sorted(document.title for document in store.read_all())

        assert len(titles) == 2
        assert titles[0].startswith("Attention")
        assert titles[1].startswith("Monocular")

    def test_two_views_of_one_page_do_not_become_two_pages(self, store):
        """Best-frame selection deliberately picks two views of one page.

        Recording them as two would mean the record claims twice as many
        pages were observed as actually were.
        """
        _run(store, fx.document_frames(fx.TRANSFORMER_PAPER, 10))

        document = store.read_all()[0]

        assert document.frames_ocred == 2
        assert document.pages_observed == 1
        assert document.pages[0].observation_count == 2

    def test_two_text_free_views_are_still_one_page(self, store):
        """The case a plain similarity test gets backwards.

        Overlap between two empty strings is zero, so a naive check
        splits them and the record then claims two pages were observed
        when nothing at all was read.
        """
        _run(
            store,
            fx.document_frames(fx.TRANSFORMER_PAPER, 10),
            recogniser=FixedTextRecogniser(pages=[""]),
        )

        assert store.read_all()[0].pages_observed == 1

    def test_a_page_turn_within_one_dwell_becomes_two_pages(self, store):
        """Same rectangle, different words: the page turned."""
        recogniser = FixedTextRecogniser(
            pages=[fx.page_regions(fx.TRANSFORMER_PAPER), fx.page_regions(fx.RECEIPT)]
        )
        _run(store, fx.document_frames(fx.TRANSFORMER_PAPER, 10), recogniser=recogniser)

        assert store.read_all()[0].pages_observed == 2


class TestHonesty:
    def test_no_spatial_anchor_is_invented(self, store):
        """The brief: do not fabricate spatial anchors.

        This cartridge cannot see a world. Absent means unknown, which is
        not the same as "nowhere".
        """
        _run(store, fx.document_frames(fx.TRANSFORMER_PAPER, 8))

        document = store.read_all()[0]

        assert document.world_id is None
        assert document.world_session_id is None
        assert document.frame_revision is None

    def test_a_supplied_anchor_is_carried_verbatim(self, store):
        """Supplied by a caller is fine. Derived here is not."""
        _run(
            store,
            fx.document_frames(fx.TRANSFORMER_PAPER, 8),
            world_id="w-1",
            world_session_id="s-1",
        )

        document = store.read_all()[0]

        assert document.world_id == "w-1"
        assert document.world_session_id == "s-1"

    def test_real_receipt_times_are_labelled_as_measured(self, store):
        _run(store, fx.document_frames(fx.TRANSFORMER_PAPER, 8))

        document = store.read_all()[0]

        assert document.timing_source == TIMING_CAPTURE_JOURNAL
        assert document.assumed_frame_interval_s is None

    def test_an_assumed_frame_rate_is_labelled_as_assumed(self, tmp_path):
        """A directory of loose jpegs has no timestamps.

        Replaying one has to assume an interval, and the assumption must
        be readable on the record rather than hidden in a driver.
        """
        store = DocumentStore(tmp_path)
        engine = DocumentMemoryEngine(
            store,
            FixedTextRecogniser(pages=[fx.page_regions(fx.RECEIPT)]),
            policy=POLICY,
            assumed_frame_interval_s=0.3,
        )
        for index, frame in enumerate(fx.document_frames(fx.RECEIPT, 8)):
            engine.observe(frame, source_seq=index)
        engine.flush()

        document = store.read_all()[0]

        assert document.timing_source == TIMING_ASSUMED_INTERVAL
        assert document.assumed_frame_interval_s == 0.3
        # Six intervals over seven admitted frames: the first frame of
        # the stream is never admitted (nothing to be steady against).
        assert document.observed_seconds == pytest.approx(0.3 * 6, abs=0.01)

    def test_a_document_is_only_as_confident_as_its_worst_page(self, store):
        """Averaging would let one crisp page hide a barely legible one."""

        class _MixedRecogniser:
            name = "mixed"

            def __init__(self):
                self.calls = 0

            def read(self, page_gray):
                self.calls += 1
                confidence = 0.95 if self.calls == 1 else 0.2
                text = "alpha beta" if self.calls == 1 else "zeta eta"
                return OcrResult(
                    text=text,
                    regions=(
                        TextRegion(text=text, confidence=confidence, box=(0, 0, 10, 10)),
                    ),
                )

            def release(self):
                return None

        _run(
            store,
            fx.document_frames(fx.TRANSFORMER_PAPER, 10),
            recogniser=_MixedRecogniser(),
        )

        assert store.read_all()[0].confidence is Confidence.LOW

    def test_the_summary_is_extractive_not_generated(self, store):
        """Every word of the summary must appear in the captured text."""
        _run(store, fx.document_frames(fx.TRANSFORMER_PAPER, 8))

        document = store.read_all()[0]
        captured = set(document.text.lower().split())

        for word in document.summary.replace("...", "").split():
            assert word.lower() in captured, f"{word!r} was not captured"

    def test_page_images_are_not_persisted_by_default(self, store):
        """A document's whole point is to be readable, and this platform
        cannot redact. Off is the only safe default."""
        _run(store, fx.document_frames(fx.TRANSFORMER_PAPER, 8))

        document = store.read_all()[0]

        assert document.retains_raw_imagery is False
        assert all(page.image_relpath is None for page in document.pages)
        assert not store.images_dir().exists()

    def test_persisting_page_images_is_opt_in_and_declares_no_redaction(self, store):
        _run(
            store,
            fx.document_frames(fx.TRANSFORMER_PAPER, 8),
            keep_page_images=True,
        )

        document = store.read_all()[0]

        assert document.retains_raw_imagery is True
        assert document.redaction == "none", (
            "no redaction exists on this platform; saying otherwise would "
            "be a false privacy claim"
        )
        assert document.pages[0].image_relpath is not None
        assert (store.directory / document.pages[0].image_relpath).exists()


class TestBadInput:
    def test_an_undecodable_frame_is_a_gap_not_a_lost_document(self, store):
        """One bad frame mid-dwell must not discard the reading in progress."""
        frames = fx.document_frames(fx.TRANSFORMER_PAPER, 10)
        frames.insert(5, b"not a jpeg")

        _run(store, frames)

        assert store.count() == 1

    @pytest.mark.parametrize("payload", [b"", b"garbage", bytes.fromhex("ffd8ff")])
    def test_hostile_bytes_never_raise(self, store, payload):
        engine = DocumentMemoryEngine(
            store, FixedTextRecogniser(), policy=POLICY, clock=_Clock()
        )

        result = engine.observe(payload, source_seq=0)

        assert result.outcome == "no_page"


class TestSamePageComparison:
    def test_identical_text_is_the_same_page(self):
        assert is_same_page("alpha beta gamma", "alpha beta gamma")

    def test_completely_different_text_is_not(self):
        assert not is_same_page("alpha beta gamma", "zeta eta theta")

    def test_two_blanks_are_the_same_page(self):
        assert is_same_page("", "")
        assert is_same_page("   ", "")

    def test_a_blank_and_a_reading_are_the_same_page(self):
        """One view read text and the other did not -- same page, seen badly."""
        assert is_same_page("", "alpha beta")
        assert is_same_page("alpha beta", "")

    def test_overlap_is_symmetric_and_bounded(self):
        assert token_overlap("a b", "b a") == 1.0
        assert token_overlap("a b", "c d") == 0.0
        assert token_overlap("", "a") == 0.0
