"""A dwell is made of segments: a page turned in place is a new page.

And the engine's identity rule across dwells: a page seen again is a
sighting of the record it was, not a second record.
"""

import cv2
import numpy as np
import pytest

from tests import document_fixtures as fx
from tower.document_memory.dwell import DwellPolicy, DwellTracker
from tower.document_memory.engine import DocumentMemoryEngine
from tower.document_memory.gate import ClassicalTextDetector, PageFinder
from tower.document_memory.ocr import FixedTextRecogniser
from tower.document_memory.store import DocumentStore

POLICY = DwellPolicy(min_frames=3, min_seconds=0.5, max_missing_seconds=1.0)


class _Clock:
    def __init__(self, start=1000.0, step=0.1):
        self.now = start
        self.step = step

    def __call__(self):
        return self.now

    def tick(self):
        self.now += self.step
        return self.now


@pytest.fixture
def store(tmp_path):
    return DocumentStore(tmp_path)


def _engine(store, pages, clock, **kwargs):
    recogniser = FixedTextRecogniser(pages=pages)
    return DocumentMemoryEngine(
        store,
        recogniser,
        policy=POLICY,
        clock=clock,
        finder=PageFinder(ClassicalTextDetector()),
        **kwargs,
    ), recogniser


def _feed(engine, frames, clock, start_seq=0):
    results = []
    for index, frame in enumerate(frames):
        results.append(engine.observe(frame, received_at=clock.tick(), source_seq=start_seq + index))
    return results


class TestSegments:
    def test_a_page_turned_in_place_becomes_a_second_page(self, store):
        """Same region, different words, no camera motion."""
        clock = _Clock()
        engine, recogniser = _engine(
            store,
            [fx.page_regions(fx.TRANSFORMER_PAPER), fx.page_regions(fx.DEPTH_NOTES)],
            clock,
        )
        first = fx.document_frames(fx.TRANSFORMER_PAPER, 8, frame_size=(360, 640))
        second = fx.document_frames(fx.DEPTH_NOTES, 8, frame_size=(360, 640))

        _feed(engine, first + second, clock)
        engine.flush()

        documents = store.read_all()
        assert len(documents) == 1
        assert documents[0].pages_observed == 2
        assert engine.pages_turned == 1
        assert {page.page_index for page in documents[0].pages} == {0, 1}

    def test_one_noisy_frame_does_not_split_a_page(self, store):
        """The content check must disagree on consecutive frames."""
        tracker = DwellTracker(POLICY)
        gray = cv2.cvtColor(
            fx.place_page(fx.render_page(fx.TRANSFORMER_PAPER), frame_size=(360, 640))[0],
            cv2.COLOR_BGR2GRAY,
        )
        finder = PageFinder(ClassicalTextDetector())
        at = 0.0
        finder.find(gray, at=at)
        noise = np.random.default_rng(1).integers(0, 255, gray.shape, dtype=np.uint8)
        for index in range(1, 8):
            at += 0.1
            frame = noise if index == 4 else gray
            candidate = finder.find(frame, at=at)
            if candidate is None:
                continue
            tracker.observe(candidate, at=at, gray=frame, frame_diagonal=735.0)

        assert tracker.segments_opened == 0

    def test_segments_are_bounded(self, store):
        """Six pages turned in one dwell is six pages of OCR at most, and
        the OLDEST segment is the one dropped when a seventh arrives."""
        policy = DwellPolicy(**{**POLICY.__dict__, "max_segments": 2, "best_frames": 1})
        tracker = DwellTracker(policy)
        finder = PageFinder(ClassicalTextDetector())
        at = 0.0
        texts = [fx.TRANSFORMER_PAPER, fx.DEPTH_NOTES, fx.RECEIPT]
        for lines in texts:
            gray = cv2.cvtColor(
                fx.place_page(fx.render_page(lines), frame_size=(360, 640))[0],
                cv2.COLOR_BGR2GRAY,
            )
            for _ in range(4):
                at += 0.1
                candidate = finder.find(gray, at=at)
                if candidate is not None:
                    tracker.observe(candidate, at=at, gray=gray, frame_diagonal=735.0)

        dwell = tracker.flush()
        assert dwell is not None
        assert tracker.segments_opened == 2
        assert len(dwell.selected) <= policy.max_segments * policy.best_frames + policy.best_frames


class TestSightings:
    def test_the_same_page_seen_again_is_a_sighting_not_a_record(self, store):
        clock = _Clock()
        engine, _ = _engine(store, [fx.page_regions(fx.TRANSFORMER_PAPER)], clock)
        frames = fx.document_frames(fx.TRANSFORMER_PAPER, 8, frame_size=(360, 640))
        gap = [fx.encode(fx.no_page_frame(frame_size=(360, 640)))] * 20

        _feed(engine, frames, clock)
        _feed(engine, gap, clock, start_seq=100)
        _feed(engine, frames, clock, start_seq=200)
        results = _feed(engine, gap, clock, start_seq=300)

        documents = store.read_all()
        assert len(documents) == 1
        assert documents[0].sighting_count == 2
        assert engine.documents_recorded == 1
        assert engine.documents_resighted == 1
        assert any(result.outcome == "resighted" for result in results)

    def test_a_sighting_keeps_the_first_observation_untouched(self, store):
        clock = _Clock()
        engine, _ = _engine(store, [fx.page_regions(fx.RECEIPT)], clock, capture_id="cap-1")
        frames = fx.document_frames(fx.RECEIPT, 8, frame_size=(360, 640))
        gap = [fx.encode(fx.no_page_frame(frame_size=(360, 640)))] * 20

        _feed(engine, frames, clock)
        engine.flush()
        first = store.read_all()[0]
        _feed(engine, gap, clock, start_seq=100)
        engine.set_capture_id("cap-2")
        _feed(engine, frames, clock, start_seq=200)
        engine.flush()

        merged = store.read_all()[0]
        assert merged.document_id == first.document_id
        assert merged.observed_at == first.observed_at
        assert merged.capture_id == "cap-1"
        assert [p.source_seq for p in merged.pages] == [p.source_seq for p in first.pages]
        assert merged.sightings[0].capture_id == "cap-2"
        assert merged.last_observed_at > first.observed_at

    def test_a_different_page_is_a_new_record(self, store):
        clock = _Clock()
        engine, _ = _engine(
            store,
            [fx.page_regions(fx.TRANSFORMER_PAPER), fx.page_regions(fx.DEPTH_NOTES)],
            clock,
        )
        gap = [fx.encode(fx.no_page_frame(frame_size=(360, 640)))] * 20

        _feed(engine, fx.document_frames(fx.TRANSFORMER_PAPER, 8, frame_size=(360, 640)), clock)
        _feed(engine, gap, clock, start_seq=100)
        _feed(engine, fx.document_frames(fx.DEPTH_NOTES, 8, frame_size=(360, 640)), clock, start_seq=200)
        engine.flush()

        assert len(store.read_all()) == 2
        assert engine.documents_resighted == 0

    def test_an_unreadable_re_view_does_not_merge(self, store):
        """Blank OCR abstains; the dwell is recorded as its own unreadable record."""
        clock = _Clock()
        engine, recogniser = _engine(store, [fx.page_regions(fx.TRANSFORMER_PAPER), ""], clock)
        frames = fx.document_frames(fx.TRANSFORMER_PAPER, 8, frame_size=(360, 640))
        gap = [fx.encode(fx.no_page_frame(frame_size=(360, 640)))] * 20

        _feed(engine, frames, clock)
        _feed(engine, gap, clock, start_seq=100)
        recogniser._pages = [""]
        recogniser.calls = 0
        _feed(engine, frames, clock, start_seq=200)
        engine.flush()

        documents = store.read_all()
        assert len(documents) == 2
        assert documents[0].sighting_count == 1
        assert not documents[1].pages[0].readable
        assert engine.dwells_unreadable == 1

    def test_resighting_can_be_switched_off(self, store):
        clock = _Clock()
        engine, _ = _engine(store, [fx.page_regions(fx.TRANSFORMER_PAPER)], clock, resight=False)
        frames = fx.document_frames(fx.TRANSFORMER_PAPER, 8, frame_size=(360, 640))
        gap = [fx.encode(fx.no_page_frame(frame_size=(360, 640)))] * 20

        _feed(engine, frames, clock)
        _feed(engine, gap, clock, start_seq=100)
        _feed(engine, frames, clock, start_seq=200)
        engine.flush()

        assert len(store.read_all()) == 2


class TestTheStoreUpdate:
    def test_update_replaces_one_record_and_leaves_the_rest(self, store):
        from dataclasses import replace

        clock = _Clock()
        engine, _ = _engine(
            store,
            [fx.page_regions(fx.TRANSFORMER_PAPER), fx.page_regions(fx.DEPTH_NOTES)],
            clock,
            resight=False,
        )
        gap = [fx.encode(fx.no_page_frame(frame_size=(360, 640)))] * 20
        _feed(engine, fx.document_frames(fx.TRANSFORMER_PAPER, 8, frame_size=(360, 640)), clock)
        _feed(engine, gap, clock, start_seq=100)
        _feed(engine, fx.document_frames(fx.DEPTH_NOTES, 8, frame_size=(360, 640)), clock, start_seq=200)
        engine.flush()
        first, second = store.read_all()

        assert store.update(replace(first, title="renamed")) is True

        renamed, untouched = store.read_all()
        assert renamed.title == "renamed"
        assert untouched.document_id == second.document_id

    def test_update_of_a_missing_record_is_false(self, store, tmp_path):
        clock = _Clock()
        engine, _ = _engine(store, [fx.page_regions(fx.RECEIPT)], clock)
        _feed(engine, fx.document_frames(fx.RECEIPT, 8, frame_size=(360, 640)), clock)
        engine.flush()
        from dataclasses import replace

        document = replace(store.read_all()[0], document_id="nobody")

        assert store.update(document) is False

    def test_a_record_with_sightings_round_trips(self, store):
        clock = _Clock()
        engine, _ = _engine(store, [fx.page_regions(fx.RECEIPT)], clock)
        frames = fx.document_frames(fx.RECEIPT, 8, frame_size=(360, 640))
        gap = [fx.encode(fx.no_page_frame(frame_size=(360, 640)))] * 20
        _feed(engine, frames, clock)
        _feed(engine, gap, clock, start_seq=100)
        _feed(engine, frames, clock, start_seq=200)
        engine.flush()

        document = DocumentStore(store.directory).read_all()[0]

        assert document.sighting_count == 2
        assert document.sightings[0].observed_seconds > 0
        assert document.pages[0].visual_hash is not None
        assert document.pages[0].readable is True
