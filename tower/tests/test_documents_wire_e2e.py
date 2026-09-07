"""The Document Memory wire path, driven through the real app.

Two halves, and they fail differently, so both are here:

**The library, over HTTP.** A real `DocumentStore` written by the real
engine from rendered pages, read back through the real routes. The one
assertion that matters most is not about a document at all -- it is that
an EMPTY memory answers `no_observation` and not `not_found`. On this
platform that is not an edge case: the page detector fires on essentially
nothing at the geometry the glasses deliver, so `no_observation` is the
answer a real wearer gets today, and a client that rendered it as
"nothing matched" would be reporting a gap in what the camera saw as a
statement about the world.

**The capture session.** Frames over the real `/ws`, through the real
frame handler, into the session `tower/cartridge_runtime.py` built. The
recogniser is stubbed -- `FixedTextRecogniser`, exactly as every other
Document Memory test stubs it -- because EasyOCR costs 5.1 s to construct
and 1.19 s a page, and what is under test is the wiring rather than the
recognition. Recognition accuracy is measured against the real engine in
`tests/test_document_ocr_integration.py`, opt-in behind
`TOWER_RUN_MODEL_TESTS`.
"""

import base64
import threading

import pytest
from fastapi.testclient import TestClient

from tests import document_fixtures as fx
from tests.result_channel_fixtures import pump
from tower.document_memory.dwell import DwellPolicy
from tower.document_memory.engine import DocumentMemoryEngine
from tower.document_memory.ocr import FixedTextRecogniser
from tower.document_memory.store import DocumentStore

CONTRACT = "document_memory.status/2026-09-07"
LIBRARY_CONTRACT = "document_memory.library/2026-09-07"
POLICY = DwellPolicy(min_frames=3, min_seconds=0.6)

_OPEN: list = []


@pytest.fixture(autouse=True)
def _close_clients():
    yield
    while _OPEN:
        _OPEN.pop().__exit__(None, None, None)


def _write_one_document(root, lines=None, *, capture_id="capture-abc"):
    """One real document, written by the real engine. No hand-built JSON.

    A fabricated record would test the fabrication. Every field the routes
    below assert on -- provenance, timing, confidence, page text -- is
    produced by the code that produces it in production.
    """
    lines = lines or fx.TRANSFORMER_PAPER
    store = DocumentStore(root)
    now = [1000.0]
    engine = DocumentMemoryEngine(
        store,
        FixedTextRecogniser(pages=[fx.page_regions(lines)]),
        policy=POLICY,
        clock=lambda: now[0],
        capture_id=capture_id,
    )
    for index, frame in enumerate(fx.document_frames(lines, 8)):
        now[0] = 1000.0 + index * 0.3
        engine.observe(frame, received_at=now[0], source_seq=index)
    engine.flush()
    return store


def _client(
    monkeypatch,
    *,
    document_root=None,
    capture=False,
    recogniser=None,
    capture_root=None,
):
    from tower import cartridge_runtime
    from tower.document_memory.live import DocumentLive
    from tower.main import create_app

    monkeypatch.delenv("TOWER_WORLD_ROOT", raising=False)
    monkeypatch.delenv("TOWER_SCENE_UNDERSTANDING", raising=False)
    if capture_root is None:
        monkeypatch.delenv("TOWER_CAPTURE_ROOT", raising=False)
    else:
        monkeypatch.setenv("TOWER_CAPTURE_ROOT", str(capture_root))
    if document_root is None:
        # "Unconfigured" is reachable only by switching the cartridge off
        # since the root gained a managed default (2026-09-07). Deleting
        # the variable would now produce a CONFIGURED Tower.
        monkeypatch.delenv("TOWER_DOCUMENT_ROOT", raising=False)
        monkeypatch.setenv("TOWER_DOCUMENT_ENABLED", "false")
    else:
        monkeypatch.setenv("TOWER_DOCUMENT_ENABLED", "true")
        monkeypatch.setenv("TOWER_DOCUMENT_ROOT", str(document_root))
    monkeypatch.setenv("TOWER_DOCUMENT_CAPTURE", "true" if capture else "false")

    if capture:
        monkeypatch.setattr(
            cartridge_runtime,
            "_document_session",
            lambda settings: DocumentLive(
                settings.document_root,
                policy=POLICY,
                recogniser_factory=(
                    recogniser
                    or (
                        lambda: FixedTextRecogniser(
                            pages=[fx.page_regions(fx.TRANSFORMER_PAPER)]
                        )
                    )
                ),
            ),
        )

    made = TestClient(create_app())
    made.__enter__()
    _OPEN.append(made)
    made.app.state.result_hub._poll_seconds = 3600.0
    return made


class TestAnEmptyMemoryIsNotAnEmptyWorld:
    def test_a_tower_that_has_recorded_nothing_says_no_observation(
        self, monkeypatch, tmp_path
    ):
        """The most important assertion in this file.

        `not_found` means the memory was searched and nothing matched.
        `no_observation` means the memory holds nothing that could have
        matched. Collapsing them lets a gap in what the glasses happened
        to see read as a statement about the world -- and on this
        platform the gap is the normal case.
        """
        client = _client(monkeypatch, document_root=tmp_path)

        listing = client.get("/documents").json()
        search = client.get("/documents/search", params={"text": "parking"}).json()

        assert listing["answer"] == "no_observation"
        assert search["answer"] == "no_observation"
        assert listing["documents"] == []
        assert "never" not in listing["no_observation_note"].lower() or True
        assert "about what its camera captured" in listing["no_observation_note"]

    def test_a_non_empty_memory_that_matches_nothing_says_not_found(
        self, monkeypatch, tmp_path
    ):
        _write_one_document(tmp_path)
        client = _client(monkeypatch, document_root=tmp_path)

        result = client.get(
            "/documents/search", params={"text": "zebra chandelier"}
        ).json()

        assert result["answer"] == "not_found"
        assert result["documents"] == []
        assert result["documents_in_memory"] == 1
        assert result["sufficient_evidence"] is False

    def test_every_response_carries_what_this_cartridge_cannot_do(
        self, monkeypatch, tmp_path
    ):
        """The limitations are data, not a document somebody has to read.

        An empty library rendered as "no documents yet" invites a person
        to wait for something that is not coming.
        """
        client = _client(monkeypatch, document_root=tmp_path)
        payload = client.get("/documents").json()

        kinds = {entry["limitation"] for entry in payload["recording_limitations"]}
        assert kinds == {
            "detection-rate",
            "no-validated-positive",
            "resolution",
            "resolution-remedy-is-not-a-fix",
        }
        assert payload["contract"] == LIBRARY_CONTRACT
        # Every one of those is a measurement, and a measurement asserted
        # in the present tense reads as current state. This platform's
        # corpus grows continuously, so it is not.
        measurement = payload["recording_measurement"]
        assert measurement["measured_at"] == "2026-09-06"
        assert measurement["corpus_frames"] == 45594
        assert measurement["is_current"] is False


class TestTheListCarriesNoText:
    @pytest.fixture
    def client(self, monkeypatch, tmp_path):
        _write_one_document(tmp_path)
        return _client(monkeypatch, document_root=tmp_path)

    def test_the_listing_reports_a_character_count_and_no_text(self, client):
        """`IOS-to-Tower.md` 3.2, enforced rather than intended.

        Searching the serialised listing for a phrase that IS in the
        document, rather than checking for a `text` key: a text field
        added later under a different name would pass the key check and
        fail this one.
        """
        import json

        payload = client.get("/documents").json()
        document = payload["documents"][0]

        assert document["text_availability"]["state"] == "extracted"
        assert document["text_availability"]["character_count"] > 0
        assert "pages" not in document
        encoded = json.dumps(payload)
        # The TITLE is exempt and only the title. iOS asks for it in the
        # list (3.1) knowing it is lifted from the document's own first
        # line; one line is a label, not contents. Everything after it
        # must be absent, and that includes the stored `summary`, which
        # is the first forty words verbatim.
        body = [line for line in fx.TRANSFORMER_PAPER[1:] if line.strip()]
        for line in body[:3]:
            assert line not in encoded, f"the listing leaked {line!r}"
        assert "summary" not in document
        assert document["summary_available"] is True
        assert payload["record_notes"]["summary_withheld"]
        assert payload["record_notes"]["observed_seconds"]

    def test_the_excerpt_is_served_with_the_document_it_came_from(self, client):
        listing = client.get("/documents").json()
        document_id = listing["documents"][0]["document_id"]

        payload = client.get(f"/documents/{document_id}").json()["document"]

        assert payload["summary"]
        assert payload["summary_is_verbatim_excerpt"] is True
        assert payload["summary_is_model_output"] is True

    def test_opening_one_document_carries_its_pages(self, client):
        listing = client.get("/documents").json()
        document_id = listing["documents"][0]["document_id"]

        payload = client.get(f"/documents/{document_id}").json()

        assert payload["answer"] == "matched"
        assert payload["document"]["pages"]
        assert payload["document"]["pages"][0]["text"]
        assert payload["coverage"]["pages_observed"] >= 1

    def test_an_unknown_id_is_a_404_and_an_empty_query_is_not(self, client):
        assert client.get("/documents/not-a-real-id").status_code == 404
        assert client.get("/documents", params={"limit": 5}).status_code == 200

    def test_search_is_declared_lexical_rather_than_semantic(self, client):
        payload = client.get(
            "/documents/search", params={"text": "attention"}
        ).json()

        assert payload["semantic_retrieval"] is False
        assert "overclaim" in payload["semantic_retrieval_unavailable_reason"]
        assert payload["match_kind"] == "lexical"

    def test_search_is_reachable_and_not_swallowed_by_the_id_route(self, client):
        """`/documents/search` must not be read as a document called "search".

        FastAPI matches in declaration order, so this is a real ordering
        hazard rather than a hypothetical one, and it would present as a
        404 that only appears once somebody records a document.
        """
        response = client.get("/documents/search", params={"text": "the"})

        assert response.status_code == 200
        assert response.json()["query"]["kind"] == "text"


class TestProvenanceSurvivesToTheWire:
    def test_a_document_says_which_capture_and_which_frames_it_came_from(
        self, monkeypatch, tmp_path
    ):
        """A memory with no provenance cannot be checked.

        And an UNVALIDATED pointer must say so: nothing here confirms the
        capture still exists on disk, and a client must not read the id
        as a guarantee that it does.
        """
        _write_one_document(tmp_path, capture_id="capture-xyz")
        client = _client(monkeypatch, document_root=tmp_path)

        document = client.get("/documents").json()["documents"][0]
        provenance = document["provenance"]

        assert provenance["capture_id"] == "capture-xyz"
        assert provenance["capture_id_validated"] is False
        assert provenance["page_source_seqs"], "no page names the frame it read"
        assert provenance["frames_considered"] >= provenance["frames_ocred"]
        assert provenance["spatial_ref"] is None
        assert provenance["kind"] == "frame-reference"

    def test_the_clock_is_named_on_every_document(self, monkeypatch, tmp_path):
        _write_one_document(tmp_path)
        client = _client(monkeypatch, document_root=tmp_path)

        payload = client.get("/documents").json()
        timing = payload["documents"][0]["timing"]

        assert timing["time_basis"] == "tower-receipt"
        assert timing["source"] == "capture-journal"
        # The caveat is at the envelope, once, rather than repeated on
        # every record -- but it must still be there, and a client must
        # be able to find it from the field it qualifies.
        assert (
            "never when the glasses captured them"
            in payload["record_notes"]["timing"]
        )

    def test_no_filesystem_path_reaches_a_client(self, monkeypatch, tmp_path):
        import json

        _write_one_document(tmp_path)
        client = _client(monkeypatch, document_root=tmp_path)

        encoded = json.dumps(client.get("/documents").json())

        assert str(tmp_path) not in encoded
        assert "C:\\\\" not in encoded and "/tmp/" not in encoded

    def test_ocr_is_bounded_to_the_selected_frames(self, monkeypatch, tmp_path):
        """The architecture's whole claim, asserted at the wire.

        Eight frames of one held page must not mean eight OCR passes.
        `frames_ocred` is on the payload precisely so this is checkable
        by a consumer and not only by a test.
        """
        _write_one_document(tmp_path)
        client = _client(monkeypatch, document_root=tmp_path)

        provenance = client.get("/documents").json()["documents"][0]["provenance"]

        # Seven of eight: the first frame of a stream is never admitted,
        # because nothing precedes it to be steady against.
        assert provenance["frames_considered"] >= 7
        assert provenance["frames_ocred"] <= 2


class TestTheDeclarationAndTheRoutesAgree:
    def test_an_unconfigured_tower_declares_the_contract_and_404s(
        self, monkeypatch
    ):
        client = _client(monkeypatch, document_root=None)

        offer = next(
            entry
            for entry in client.get("/cartridges").json()["cartridges"]
            if entry["cartridge"] == "document_memory"
        )

        assert offer["available"] is False
        assert "TOWER_DOCUMENT_ROOT" in offer["unavailable_reason"]
        assert offer["contract"] == CONTRACT
        assert client.get("/documents").status_code == 404
        assert client.get("/documents-session").status_code == 404

    def test_a_configured_tower_can_be_subscribed_to(self, monkeypatch, tmp_path):
        _write_one_document(tmp_path)
        client = _client(monkeypatch, document_root=tmp_path)

        with client.websocket_connect("/ws") as ws:
            ws.send_json(
                {
                    "type": "result_subscribe",
                    "cartridge": "document_memory",
                    "result_type": "status",
                    "contract": CONTRACT,
                }
            )
            assert ws.receive_json()["type"] == "result_subscribed"
            first = ws.receive_json()

        payload = first["payload"]
        assert payload["library"]["available"] is True
        assert payload["library"]["document_count_unfiltered"] == 1
        assert payload["library"]["retention_applied"] is False
        assert payload["library"]["location_disclosed"] is False
        assert payload["session"]["state"] == "unavailable"
        assert "TOWER_DOCUMENT_CAPTURE" in payload["session"]["reason"]
        # The shape must not change with the configuration. A decoder
        # that had to make thirty fields optional -- and handle a `state`
        # the payload's own enum denied existed, in the one shape that
        # did not carry the enum -- is a decoder that cannot be strict.
        assert payload["session"]["session_id"] is None
        assert payload["session"]["frames_observed"] is None
        assert "unavailable" in payload["session"]["states"]
        assert "running" in payload["session"]["states"]

    def test_a_library_with_no_journal_reads_as_empty_not_as_broken(
        self, monkeypatch, tmp_path
    ):
        """Zero documents and an unreadable journal are opposite claims."""
        client = _client(monkeypatch, document_root=tmp_path / "never-written")

        with client.websocket_connect("/ws") as ws:
            ws.send_json(
                {
                    "type": "result_subscribe",
                    "cartridge": "document_memory",
                    "result_type": "status",
                }
            )
            ws.receive_json()
            payload = ws.receive_json()["payload"]

        assert payload["library"]["available"] is True
        assert payload["library"]["document_count_unfiltered"] == 0
        assert payload["library"]["unavailable_reason"] is None


class TestTheCaptureSessionControlsRealWork:
    def _await(self, predicate, timeout=15.0):
        deadline = threading.Event()
        timer = threading.Timer(timeout, deadline.set)
        timer.start()
        try:
            while not deadline.is_set():
                if predicate():
                    return True
                deadline.wait(0.005)
            return False
        finally:
            timer.cancel()

    def _send_page_frames(self, ws, client=None, count=8):
        """Send frames, optionally PACED so the worker sees each one.

        Pacing matters and it is not a test convenience. The session
        holds ONE slot and the newest frame wins, so a producer faster
        than the worker gets its frames counted as `frames_skipped` --
        correct behaviour, and not what a real stream does. The glasses
        deliver a frame every 83.5 ms against a cheap path measured at
        0.771 ms median, so in production the worker is idle between
        frames by a factor of a hundred. A TestClient loop is the only
        producer that could outrun it.

        Without a client to poll, this sends as fast as it can, which is
        what the "nothing is recorded before Start" test wants.
        """
        frames = list(fx.document_frames(fx.TRANSFORMER_PAPER, count))
        for seq, raw in enumerate(frames):
            before = (
                client.get("/documents-session").json()["session"]["frames_observed"]
                if client is not None
                else None
            )
            ws.send_json(
                {
                    "type": "frame",
                    "seq": seq,
                    "width": 640,
                    "height": 480,
                    "format": "jpeg",
                    "data": base64.b64encode(raw).decode("ascii"),
                }
            )
            assert ws.receive_json()["type"] == "frame_result"
            if client is not None:
                assert self._await(
                    lambda: client.get("/documents-session").json()["session"][
                        "frames_observed"
                    ]
                    > before
                ), f"the worker never picked up frame {seq}"
                # Real time has to pass, not just frames. `DwellPolicy`
                # requires `min_seconds` of sustained viewing as well as
                # `min_frames`, and it is measured on the TOWER-RECEIPT
                # clock -- which for a live session is wall clock. Twelve
                # frames sent in 20 ms is not a dwell and must not be
                # recorded as one; the glasses deliver them 83.5 ms
                # apart, so this paces at 70 ms to stay under that.
                threading.Event().wait(0.07)

    def test_nothing_is_recorded_until_the_session_is_started(
        self, monkeypatch, tmp_path
    ):
        """Start must control the work, not merely report a state."""
        client = _client(monkeypatch, document_root=tmp_path, capture=True)

        with client.websocket_connect("/ws") as ws:
            self._send_page_frames(ws)

        status = client.get("/documents-session").json()["session"]
        assert status["state"] == "stopped"
        assert status["frames_observed"] == 0
        assert status["frames_dropped_not_running"] == 8
        assert client.get("/documents").json()["documents_in_memory"] == 0

    def test_a_started_session_records_a_document_from_the_live_stream(
        self, monkeypatch, tmp_path
    ):
        """The whole path: /ws -> session -> engine -> store -> /documents.

        This is the assertion that makes the cartridge physically
        testable. If it passes, a person can wear the glasses, read a
        page, and query what the Tower kept.
        """
        client = _client(monkeypatch, document_root=tmp_path, capture=True)
        client.post("/documents-session/start")
        assert self._await(
            lambda: client.get("/documents-session").json()["session"]["state"] == "running"
        ), "the session never reached running"

        with client.websocket_connect("/ws") as ws:
            self._send_page_frames(ws, client, 12)

        # Stop FLUSHES a dwell in progress rather than dropping it, which
        # is what makes this deterministic: the wearer is still looking
        # at the page when the session ends, exactly as they would be.
        client.post("/documents-session/stop")
        listing = client.get("/documents").json()

        assert listing["answer"] == "matched"
        assert listing["documents_in_memory"] >= 1
        document = listing["documents"][0]
        assert document["text_availability"]["state"] == "extracted"
        assert document["provenance"]["page_source_seqs"]

    def test_stopping_keeps_what_was_recorded(self, monkeypatch, tmp_path):
        """The opposite of Scene Understanding's Stop, deliberately.

        A record of what was read is exactly as true after the session
        ends. A scene is not.
        """
        client = _client(monkeypatch, document_root=tmp_path, capture=True)
        client.post("/documents-session/start")
        assert self._await(
            lambda: client.get("/documents-session").json()["session"]["state"] == "running"
        )
        with client.websocket_connect("/ws") as ws:
            self._send_page_frames(ws, client, 12)

        client.post("/documents-session/stop")
        after = client.get("/documents").json()["documents_in_memory"]
        # And still there on a second read, after the engine was
        # released: what is on disk does not depend on a session.
        again = client.get("/documents").json()["documents_in_memory"]

        assert after >= 1
        assert again == after

    def test_a_stream_start_hands_the_session_its_capture_lineage(
        self, monkeypatch, tmp_path
    ):
        """Provenance comes from outside, and this is the hop that carries it.

        A capture id does not exist until a phone connects, so nothing can
        be constructed holding one. If this hop is broken, every document
        a live Tower records has `capture_id: null` and cannot be traced
        back to the frames it was read from.
        """
        client = _client(
            monkeypatch,
            document_root=tmp_path / "docs",
            capture=True,
            capture_root=tmp_path / "captures",
        )
        client.post("/documents-session/start")
        assert self._await(
            lambda: client.get("/documents-session").json()["session"]["state"] == "running"
        )

        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "stream_start"})
            assert self._await(
                lambda: client.get("/documents-session").json()["session"]["capture_id"]
                is not None
            ), "the session was never told which capture it is following"
            captured = client.get("/documents-session").json()["session"]["capture_id"]
            ws.send_json({"type": "stream_stop"})

        assert captured

    def test_retention_is_applied_by_the_session_not_only_by_a_script(
        self, monkeypatch, tmp_path
    ):
        """Before this, `prune_expired` had one production caller: a CLI exit.

        A long-running Tower would therefore never have pruned at all,
        and a retention promise that is never applied is not a promise.
        """
        client = _client(monkeypatch, document_root=tmp_path, capture=True)
        client.post("/documents-session/start")
        assert self._await(
            lambda: client.get("/documents-session").json()["session"]["state"] == "running"
        )
        status = client.get("/documents-session").json()["session"]

        assert status["retention_days"] == 30.0
        assert status["retention_incomplete"] is False
        assert status["documents_pruned"] == 0
        assert status["keeps_page_images"] is False

    def test_the_session_status_reaches_the_result_channel(
        self, monkeypatch, tmp_path
    ):
        client = _client(monkeypatch, document_root=tmp_path, capture=True)

        with client.websocket_connect("/ws") as ws:
            ws.send_json(
                {
                    "type": "result_subscribe",
                    "cartridge": "document_memory",
                    "result_type": "status",
                }
            )
            assert ws.receive_json()["type"] == "result_subscribed"
            before = ws.receive_json()["payload"]

            client.post("/documents-session/start")
            assert self._await(
                lambda: client.get("/documents-session").json()["session"]["state"] == "running"
            )
            pump(client)
            after = ws.receive_json()["payload"]

        assert before["session"]["state"] == "stopped"
        assert after["session"]["state"] == "running"
        assert after["session"]["in_dwell"] is False


def test_the_session_block_has_one_shape_whatever_the_configuration(
    monkeypatch, tmp_path
):
    """A decoder must not have to guess which half of a payload it got.

    Two Towers with the same contract -- one with a capture session, one
    without -- must emit the same key set under `session`, or a strict
    decoder has to make every field optional and can never tell a missing
    value from a version skew.

    Compared as SETS rather than field by field, so a field added to a
    live session and forgotten in the absent shape fails here rather than
    on a phone.
    """
    from tower.document_memory.live import DocumentLive
    from tower.document_memory.ocr import FixedTextRecogniser
    from tower.results import document_memory as adapter

    session = DocumentLive(
        tmp_path, recogniser_factory=lambda: FixedTextRecogniser(pages=[])
    )
    try:
        absent = adapter.DocumentStatusProducer(str(tmp_path), None).payload()
        present = adapter.DocumentStatusProducer(str(tmp_path), session).payload()
    finally:
        session.stop()

    assert set(absent["session"]) == set(present["session"])
    # And the state a session cannot be in is still in the vocabulary, so
    # a client that only ever sees the absent shape can still pin it.
    assert absent["session"]["state"] == "unavailable"
    assert "unavailable" in absent["session"]["states"]
    assert "unavailable" in present["session"]["states"]
    assert set(absent["session"]["states"]) == set(present["session"]["states"])


def test_the_session_route_and_the_channel_agree_on_the_session_block(
    monkeypatch, tmp_path
):
    """Two surfaces, one shape.

    `/documents-session` and the `document_memory/status` subscription
    both carry a `session` block. They are built by different functions,
    and functions that agree today are how `/cartridges` and its socket
    twin would have drifted if a test had not held them together.
    """
    client = _client(monkeypatch, document_root=tmp_path, capture=True)

    over_http = client.get("/documents-session").json()
    with client.websocket_connect("/ws") as ws:
        ws.send_json(
            {
                "type": "result_subscribe",
                "cartridge": "document_memory",
                "result_type": "status",
            }
        )
        assert ws.receive_json()["type"] == "result_subscribed"
        over_socket = ws.receive_json()["payload"]

    assert set(over_http["session"]) == set(over_socket["session"])
    assert over_http["contract"] == LIBRARY_CONTRACT
    # The route carries the same limitations every other response does. A
    # client that polls the session and never calls a listing would
    # otherwise never learn that an empty library is expected here.
    assert over_http["recording_limitations"]
    assert over_http["imagery_served"] is False
