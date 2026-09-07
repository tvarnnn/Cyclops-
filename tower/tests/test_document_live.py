"""The live session's own behaviour: what it reports, and when it stops itself.

Driven directly, with a fixed recogniser, so nothing here loads a model
or opens a socket. The wire path is `test_documents_wire_e2e.py`.
"""

import threading
import time

import pytest

from tests import document_fixtures as fx
from tower.document_memory.dwell import DwellPolicy
from tower.document_memory.live import DocumentLive
from tower.document_memory.ocr import FixedTextRecogniser
from tower.document_memory.store import DocumentStore

POLICY = DwellPolicy(min_frames=3, min_seconds=0.2, max_missing_seconds=0.5)
FRAME = (360, 640)


class _Recogniser(FixedTextRecogniser):
    """A fixed recogniser that also reports a device, like the real one."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = None
        self.loaded = 0
        self.released = 0

    def load(self):
        self.loaded += 1
        self.device = "cpu"

    def release(self):
        self.released += 1
        self.device = None


def _wait(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def _session(tmp_path, recognisers, **kwargs):
    made = []

    def factory():
        recogniser = _Recogniser(pages=[fx.page_regions(fx.RECEIPT)])
        made.append(recogniser)
        recognisers.append(recogniser)
        return recogniser

    session = DocumentLive(
        tmp_path, policy=POLICY, recogniser_factory=factory, retention_days=1.0, **kwargs
    )
    return session


def _feed(session, frames, *, start=1000.0, step=0.1, seq=0):
    at = start
    for index, frame in enumerate(frames):
        at += step
        session.offer_frame(frame, received_at=at, source_seq=seq + index)
        # The single slot drops a frame offered while the worker is busy;
        # pacing keeps the test about lifecycle rather than backpressure.
        time.sleep(0.03)
    return at


@pytest.fixture
def frames():
    return fx.document_frames(fx.RECEIPT, 10, frame_size=FRAME)


class TestWhatTheStatusSays:
    def test_the_device_and_the_new_counters_are_reported(self, tmp_path, frames):
        recognisers = []
        session = _session(tmp_path, recognisers)
        session.start()
        assert _wait(lambda: session.status()["state"] == "running")

        status = session.status()
        assert status["ocr_device"] == "cpu"
        assert status["documents_resighted"] == 0
        assert status["dwells_unreadable"] == 0
        assert status["idle_stop_pending"] is False
        assert status["idle_stop_seconds"] == 600.0
        session.stop()

    def test_a_page_seen_twice_counts_once_recorded_and_once_resighted(
        self, tmp_path, frames
    ):
        recognisers = []
        session = _session(tmp_path, recognisers)
        session.start()
        assert _wait(lambda: session.status()["state"] == "running")
        gap = [fx.encode(fx.no_page_frame(frame_size=FRAME))] * 12

        at = _feed(session, frames)
        at = _feed(session, gap, start=at, seq=100)
        assert _wait(lambda: session.status()["documents_recorded"] == 1)
        at = _feed(session, frames, start=at, seq=200)
        _feed(session, gap, start=at, seq=300)
        assert _wait(lambda: session.status()["documents_resighted"] == 1)

        session.stop()
        assert DocumentStore(tmp_path).count() == 1
        assert DocumentStore(tmp_path).read_all()[0].sighting_count == 2

    def test_an_unreadable_dwell_is_recorded_and_counted(self, tmp_path, frames):
        recognisers = []

        def factory():
            recogniser = _Recogniser(pages=[""])
            recognisers.append(recogniser)
            return recogniser

        session = DocumentLive(
            tmp_path, policy=POLICY, recogniser_factory=factory, retention_days=1.0
        )
        session.start()
        assert _wait(lambda: session.status()["state"] == "running")
        at = _feed(session, frames)
        _feed(session, [fx.encode(fx.no_page_frame(frame_size=FRAME))] * 12, start=at, seq=100)

        assert _wait(lambda: session.status()["dwells_unreadable"] == 1)
        assert session.status()["documents_recorded"] == 1
        session.stop()
        document = DocumentStore(tmp_path).read_all()[0]
        assert not any(page.readable for page in document.pages)

    def test_stop_flushes_a_dwell_in_progress_and_releases_the_reader(
        self, tmp_path, frames
    ):
        recognisers = []
        session = _session(tmp_path, recognisers)
        session.start()
        assert _wait(lambda: session.status()["state"] == "running")
        _feed(session, frames)

        session.stop()

        assert session.status()["state"] == "stopped"
        assert session.status()["flushed_document_id"] is not None
        assert DocumentStore(tmp_path).count() == 1
        assert recognisers[0].released == 1


class TestASecondSession:
    def test_a_second_session_records_into_the_same_library(self, tmp_path, frames):
        recognisers = []
        session = _session(tmp_path, recognisers)
        for _ in range(2):
            session.start()
            assert _wait(lambda: session.status()["state"] == "running")
            at = _feed(session, frames)
            _feed(session, [fx.encode(fx.no_page_frame(frame_size=FRAME))] * 12, start=at, seq=100)
            assert _wait(
                lambda: session.status()["documents_recorded"]
                + session.status()["documents_resighted"]
                >= 1
            )
            session.stop()

        assert len(recognisers) == 2
        assert all(r.released == 1 for r in recognisers)
        # The same page, twice: one record with two sightings.
        assert DocumentStore(tmp_path).count() == 1
        assert DocumentStore(tmp_path).read_all()[0].sighting_count == 2

    def test_counters_start_fresh_each_session(self, tmp_path, frames):
        recognisers = []
        session = _session(tmp_path, recognisers)
        session.start()
        assert _wait(lambda: session.status()["state"] == "running")
        at = _feed(session, frames)
        _feed(session, [fx.encode(fx.no_page_frame(frame_size=FRAME))] * 12, start=at, seq=100)
        assert _wait(lambda: session.status()["documents_recorded"] == 1)
        session.stop()

        session.start()
        assert _wait(lambda: session.status()["state"] == "running")
        assert session.status()["documents_recorded"] == 0
        assert session.status()["ocr_device"] == "cpu"
        session.stop()

    def test_no_thread_is_left_behind_by_repeated_sessions(self, tmp_path, frames):
        recognisers = []
        session = _session(tmp_path, recognisers)
        session.start()
        assert _wait(lambda: session.status()["state"] == "running")
        session.stop()
        baseline = threading.active_count()

        for _ in range(3):
            session.start()
            assert _wait(lambda: session.status()["state"] == "running")
            session.stop()

        assert threading.active_count() <= baseline


class TestTheIdleWindDown:
    def test_a_session_whose_stream_closed_stops_itself(self, tmp_path):
        recognisers = []
        session = _session(tmp_path, recognisers, idle_stop_s=0.3)
        session.start()
        assert _wait(lambda: session.status()["state"] == "running")

        session.stream_opened("phone")
        session.stream_closed("phone")

        assert session.status()["idle_stop_pending"] is True
        assert _wait(lambda: session.status()["state"] == "stopped", timeout=3.0)
        assert recognisers[0].released == 1

    def test_an_idle_stop_never_runs_ocr_on_the_timer_thread(self, tmp_path, frames):
        """A dwell open when the stream closed is dropped, not read: the
        timer's thread must not build a torch thread pool, and the dwell
        ended ten minutes ago in any case. Logged, never silent."""
        recognisers = []
        session = _session(tmp_path, recognisers, idle_stop_s=0.3)
        session.start()
        assert _wait(lambda: session.status()["state"] == "running")
        _feed(session, frames[:6])
        assert _wait(lambda: session.status()["in_dwell"] is True)
        calls_before = recognisers[0].calls

        session.stream_closed("phone")
        assert _wait(lambda: session.status()["state"] == "stopped", timeout=3.0)

        assert recognisers[0].calls == calls_before
        assert session.status()["flushed_document_id"] is None
        assert DocumentStore(tmp_path).count() == 0

    def test_a_reconnect_cancels_the_wind_down(self, tmp_path, frames):
        recognisers = []
        session = _session(tmp_path, recognisers, idle_stop_s=0.4)
        session.start()
        assert _wait(lambda: session.status()["state"] == "running")

        session.stream_opened("phone")
        session.stream_closed("phone")
        assert session.status()["idle_stop_pending"] is True
        session.stream_opened("phone")
        assert session.status()["idle_stop_pending"] is False

        time.sleep(0.6)
        assert session.status()["state"] == "running"
        session.stop()

    def test_a_frame_cancels_the_wind_down(self, tmp_path, frames):
        """A Tower without a recorder never calls `stream_opened`; the
        frames themselves are the evidence a stream is open."""
        recognisers = []
        session = _session(tmp_path, recognisers, idle_stop_s=0.4)
        session.start()
        assert _wait(lambda: session.status()["state"] == "running")
        session.stream_closed("phone")
        assert session.status()["idle_stop_pending"] is True

        _feed(session, frames[:2])

        assert session.status()["idle_stop_pending"] is False
        time.sleep(0.6)
        assert session.status()["state"] == "running"
        session.stop()

    def test_stopping_by_hand_cancels_the_timer(self, tmp_path):
        recognisers = []
        session = _session(tmp_path, recognisers, idle_stop_s=0.3)
        session.start()
        assert _wait(lambda: session.status()["state"] == "running")
        session.stream_closed("phone")
        session.stop()

        assert session.status()["idle_stop_pending"] is False
        time.sleep(0.5)
        assert session.status()["state"] == "stopped"
        assert recognisers[0].released == 1

    def test_the_wind_down_can_be_switched_off(self, tmp_path):
        recognisers = []
        session = _session(tmp_path, recognisers, idle_stop_s=None)
        session.start()
        assert _wait(lambda: session.status()["state"] == "running")
        session.stream_closed("phone")

        assert session.status()["idle_stop_pending"] is False
        assert session.status()["idle_stop_seconds"] is None
        session.stop()
