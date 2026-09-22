"""Object Memory's first wire path: a read-only HTTP view of observations.

Two things are under test here that are not ordinary transport concerns.

RETENTION CANNOT BE WIDENED. The store clamps every read to
min(persisted, requested), and a review of this cartridge already found
exactly that hole once at the CLI layer -- `--retention-days 3650` served
expired records. A new layer is a new chance to reintroduce it, so the
widening attempt is made HERE, over the route, against a store holding a
record that is genuinely out of window.

THE PAYLOAD MUST NOT OVERCLAIM. `spatial_ref` is None on every record and
nothing in this cartridge knows where anything is in a room. The tests
below pin the field names, because field names are what a client codes
against -- a disclaimer in a doc is not a constraint on a decoder.
"""

import ast
import pathlib
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tower.confidence import Confidence
from tower.object_memory.records import ObjectObservation, privacy_tags_for
from tower.object_memory.store import DEFAULT_RETENTION_SECONDS, ObservationStore
from tower.results.object_memory import (
    OBSERVATIONS_CONTRACT,
    build_last_seen,
    build_observations,
)
from tower.routes import observations as observation_routes

DAY = 86400.0
# The real clock, not a frozen one. The route builds its own store and
# passes no clock -- it must not, or a wall-clock retention promise would
# depend on a test seam. So the fixture places its records relative to
# real time and the expired record is genuinely, unfakeably expired.
NOW = time.time()


def _observation(
    object_class: str = "laptop",
    *,
    recorded_at: float = NOW,
    observed_at: float | None = None,
    detector_score: float | None = 0.61,
    best_score: float | None = 0.97,
    session_id: str | None = "capture0",
    frame_seq: int | None = 1821,
) -> ObjectObservation:
    return ObjectObservation(
        object_class=object_class,
        detector_score=detector_score,
        confidence=Confidence.HIGH,
        observed_at=recorded_at if observed_at is None else observed_at,
        time_basis="tower-receipt",
        recorded_at=recorded_at,
        source="glasses-camera",
        module_id="object-memory",
        session_id=session_id,
        frame_seq=frame_seq,
        bounding_box=(0.1, 0.2, 0.3, 0.4),
        retention_tag="default",
        privacy_tags=privacy_tags_for(session_id, frame_seq),
        spatial_ref=None,
        external_refs=(),
        best_score=best_score,
    )


@pytest.fixture
def corpus(tmp_path):
    """A store written under the 30-day default holding one expired record.

    The expired one is 40 days old, so no honest read under the persisted
    window may serve it -- and a dishonest one is exactly what the
    widening test is looking for.
    """
    root = tmp_path / "object_memory"
    store = ObservationStore(
        root, retention_seconds=DEFAULT_RETENTION_SECONDS, clock=lambda: NOW
    )
    store.append(_observation("laptop", recorded_at=NOW - 1 * DAY))
    store.append(_observation("cell phone", recorded_at=NOW - 2 * DAY))
    store.append(_observation("laptop", recorded_at=NOW - 40 * DAY))
    return root


def _client(root) -> TestClient:
    app = FastAPI()
    app.include_router(observation_routes.router)
    app.state.object_memory_root = root
    return TestClient(app)


def _store(root, retention_seconds=None) -> ObservationStore:
    return ObservationStore(
        root, retention_seconds=retention_seconds, clock=lambda: NOW
    )


# --- The contract identifier ---------------------------------------------


def test_the_contract_identifier_is_exact():
    """Opaque and dated, compared for equality only -- never parsed."""
    assert OBSERVATIONS_CONTRACT == "object_memory.observations/2026-08-26"


def test_every_payload_carries_the_contract(corpus):
    listing = build_observations(_store(corpus))
    answer = build_last_seen(_store(corpus), "laptop")

    assert listing["contract"] == OBSERVATIONS_CONTRACT
    assert answer["contract"] == OBSERVATIONS_CONTRACT


# --- `since`: this recording's memories, apart from history ---------------


def test_since_returns_only_records_written_at_or_after_it(corpus):
    """The primitive that stops history masquerading as the current run.

    The corpus holds a laptop at NOW-1d and a cell phone at NOW-2d inside
    the window. A session that started 1.5 days ago asks `since` and gets
    only the laptop -- the record its recording created -- not the phone
    that predates it.
    """
    listing = build_observations(_store(corpus), since=NOW - 1.5 * DAY)

    assert listing["observation_count"] == 1
    assert listing["observations"][0]["object_class"] == "laptop"
    assert all(o["recorded_at"] >= NOW - 1.5 * DAY for o in listing["observations"])


def test_since_is_echoed_and_absent_means_the_whole_store(corpus):
    """`null` since and a real since are different claims, said in the payload."""
    whole = build_observations(_store(corpus))
    scoped = build_observations(_store(corpus), since=NOW - 1.5 * DAY)

    assert whole["since"] is None
    assert whole["observation_count"] == 2
    assert scoped["since"] == NOW - 1.5 * DAY


def test_a_recording_that_wrote_nothing_gets_an_empty_answer_not_history(corpus):
    """The zero-frame failure, answered honestly.

    A session that starts NOW and records nothing asks `since=NOW`; the
    store has 116-record-shaped history older than that, and the answer is
    empty rather than borrowing it.
    """
    listing = build_observations(_store(corpus), since=NOW)

    assert listing["observation_count"] == 0
    assert listing["observations"] == []
    # History is still there for the view that asks for it.
    assert build_observations(_store(corpus))["observation_count"] == 2


def test_since_over_the_route_and_the_contract_is_unchanged(corpus):
    response = _client(corpus).get(
        "/object-memory/observations", params={"since": NOW - 1.5 * DAY}
    )
    body = response.json()

    assert response.status_code == 200
    assert body["observation_count"] == 1
    assert body["since"] == NOW - 1.5 * DAY
    # Additive: the identifier a shipped iOS build compares for equality
    # does not move because a new optional field appeared.
    assert body["contract"] == OBSERVATIONS_CONTRACT


def test_a_negative_since_is_refused_by_the_route(corpus):
    response = _client(corpus).get(
        "/object-memory/observations", params={"since": -1}
    )
    assert response.status_code == 422


# --- Retention cannot be widened over the wire ---------------------------


def test_a_reader_cannot_widen_retention_over_the_route(corpus):
    """The hole a review already found in the CLI, tried again at HTTP.

    The store was written under 30 days and holds a 40-day-old record.
    Asking the route for 3650 days must serve two records, not three.
    """
    response = _client(corpus).get(
        "/object-memory/observations", params={"retention_days": 3650}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["observation_count"] == 2
    assert all(o["recorded_at"] >= NOW - 30 * DAY for o in body["observations"])


def test_asking_for_an_unbounded_window_over_the_route_does_not_widen_it(corpus):
    """0 means "no limit of my own", and it still gets the persisted one."""
    response = _client(corpus).get(
        "/object-memory/observations", params={"retention_days": 0}
    )

    assert response.status_code == 200
    assert response.json()["observation_count"] == 2


def test_the_widening_attempt_is_reported_back_as_clamped(corpus):
    """The clamp is visible, not merely applied.

    A client that asked for 3650 days and silently got 30 would have no
    way to know its question was refused.
    """
    body = _client(corpus).get(
        "/object-memory/observations", params={"retention_days": 3650}
    ).json()

    assert body["retention"]["requested_days"] == 3650.0
    assert body["retention"]["effective_days"] == 30.0
    assert body["retention"]["clamped"] is True


def test_a_reader_may_still_narrow_the_window_over_the_route(corpus):
    """Narrowing is the half that must keep working."""
    body = _client(corpus).get(
        "/object-memory/observations", params={"retention_days": 1.5}
    ).json()

    assert body["observation_count"] == 1
    assert body["observations"][0]["object_class"] == "laptop"
    assert body["retention"]["clamped"] is False


def test_last_seen_cannot_be_widened_either(corpus):
    """The expired laptop is the MOST RECENT... only if you can see it.

    It is not the most recent -- it is the oldest -- so this test would
    pass by accident against `last_seen`. It is asserted against a class
    whose ONLY record is expired instead.
    """
    body = _client(corpus).get(
        "/object-memory/last-seen/laptop", params={"retention_days": 3650}
    ).json()

    assert body["observed"] is True
    assert body["observation"]["recorded_at"] == NOW - 1 * DAY


def test_a_class_whose_only_record_expired_is_not_served_at_any_request(tmp_path):
    root = tmp_path / "om"
    store = ObservationStore(
        root, retention_seconds=DEFAULT_RETENTION_SECONDS, clock=lambda: NOW
    )
    store.append(_observation("cell phone", recorded_at=NOW - 90 * DAY))

    body = _client(root).get(
        "/object-memory/last-seen/cell phone", params={"retention_days": 3650}
    ).json()

    assert body["observed"] is False
    assert body["observation"] is None


def test_a_negative_retention_request_is_refused_by_the_route(corpus):
    """Rather than reaching the store, which raises ValueError -> 500."""
    response = _client(corpus).get(
        "/object-memory/observations", params={"retention_days": -1}
    )

    assert response.status_code == 422


# --- Read-only: no mutation reaches the wire -----------------------------


def test_the_router_exposes_only_reads(corpus):
    methods = set()
    for route in observation_routes.router.routes:
        methods |= set(getattr(route, "methods", set()))

    assert methods == {"GET"}


def test_no_wire_module_can_reach_purge_or_prune():
    """An unauthenticated endpoint must not be able to delete a memory.

    An AST scan rather than a grep, so a comment explaining WHY these are
    absent does not trip the rule.
    """
    forbidden = {"purge", "prune_expired"}
    offenders = []
    for path in (
        pathlib.Path("tower") / "routes" / "observations.py",
        pathlib.Path("tower") / "results" / "object_memory.py",
    ):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "attr", None) or getattr(
                    node.func, "id", None
                )
                if name in forbidden:
                    offenders.append(f"{path.name} calls {name}()")

    assert offenders == []


def test_the_handlers_are_sync_so_disk_reads_stay_off_the_event_loop():
    """Same reasoning as `tower/routes/geometry.py`.

    FastAPI runs a sync endpoint in its threadpool. An `async def` here
    would put a blocking JSONL read on the event loop with no executor.
    """
    tree = ast.parse(
        (pathlib.Path("tower") / "routes" / "observations.py").read_text(
            encoding="utf-8"
        )
    )
    coroutines = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef)
    ]

    assert coroutines == []


# --- The payload must not overclaim --------------------------------------


def test_spatial_ref_travels_as_an_explicit_null(corpus):
    """Carried, not omitted: a consumer must see the field exists and is empty."""
    body = _client(corpus).get("/object-memory/observations").json()

    assert body["spatial_ref"] is None
    for observation in body["observations"]:
        assert observation["where"]["spatial_ref"] is None
        assert "spatial_ref" in observation["where"]


def test_where_is_a_frame_reference_and_says_so(corpus):
    body = _client(corpus).get("/object-memory/observations").json()
    where = body["observations"][0]["where"]

    assert where["kind"] == "frame-reference"
    assert where["session_id"] == "capture0"
    assert where["frame_seq"] == 1821
    assert where["camera"] == "glasses-camera"


def test_the_only_position_offered_is_inside_a_frame(corpus):
    """The bounding box lives UNDER the frame reference, not beside it.

    A box at the top level reads as a position. Nested under
    `where.kind == "frame-reference"` it reads as what it is: where in a
    picture, not where in a room.
    """
    body = _client(corpus).get("/object-memory/observations").json()
    observation = body["observations"][0]

    assert "bounding_box" not in observation
    assert observation["where"]["bounding_box_normalized"] == [0.1, 0.2, 0.3, 0.4]


def test_a_record_claims_visibility_once_and_not_presence(corpus):
    body = _client(corpus).get("/object-memory/observations").json()

    assert body["claim"] == "category-was-visible-once"
    assert body["identity"] == "category-not-instance"
    for observation in body["observations"]:
        assert observation["claim"] == "category-was-visible-once"
        assert "present" not in observation


def test_absence_of_a_record_is_not_evidence_of_absence(corpus):
    """And the payload says so in a field, not only in a document."""
    body = _client(corpus).get("/object-memory/last-seen/laptop").json()

    assert body["absence_means"] == "not-observed-by-this-cartridge"


def test_an_unobserved_class_answers_200_and_not_404(corpus):
    """404 would read as "there is no laptop", which is a claim about the world.

    The resource -- what this cartridge knows about a class -- exists. The
    answer is that it knows nothing.
    """
    response = _client(corpus).get("/object-memory/last-seen/keys")

    assert response.status_code == 200
    body = response.json()
    assert body["observed"] is False
    assert body["observation"] is None
    assert body["where"] is None


def test_a_class_this_cartridge_never_records_says_so(corpus):
    body = _client(corpus).get("/object-memory/last-seen/keys").json()

    assert body["recordable"] is False
    assert body["recorded_classes"] == ["laptop", "cell phone"]


def test_a_recordable_class_with_no_record_is_distinguishable(corpus):
    """"We never look for keys" and "we looked and saw no toaster" differ."""
    body = _client(corpus).get("/object-memory/last-seen/cell phone").json()

    assert body["recordable"] is True
    assert body["observed"] is True


def test_the_imagery_a_record_points_at_is_not_this_cartridges_to_promise(corpus):
    """session_id + frame_seq resolves into data/captures/, governed elsewhere."""
    body = _client(corpus).get("/object-memory/observations").json()

    assert body["observations"][0]["where"]["imagery_retention"] == "capture-side"
    assert body["observations"][0]["privacy_tags"] == [
        "derived-only",
        "frame-referenced",
    ]


# --- Project disciplines --------------------------------------------------


def test_booleans_are_booleans_and_not_ints(corpus):
    """`bool` subclasses `int`; a 1 here fails every Swift `as? Bool` decode."""
    body = _client(corpus).get("/object-memory/last-seen/laptop").json()
    listing = _client(corpus).get(
        "/object-memory/observations", params={"retention_days": 3650}
    ).json()

    for value in (
        body["observed"], body["recordable"], listing["retention"]["clamped"]
    ):
        assert type(value) is bool
    assert build_last_seen(_store(corpus), "laptop")["observed"] is True
    assert build_observations(
        _store(corpus, 3650 * DAY), requested_retention_days=3650.0
    )["retention"]["clamped"] is True


def test_a_missing_score_is_null_and_never_zero(tmp_path):
    """Records written before best_score existed must not borrow a number."""
    root = tmp_path / "om"
    store = ObservationStore(
        root, retention_seconds=DEFAULT_RETENTION_SECONDS, clock=lambda: NOW
    )
    store.append(
        _observation("laptop", recorded_at=NOW, detector_score=None, best_score=None)
    )

    observation = _client(root).get(
        "/object-memory/observations"
    ).json()["observations"][0]

    assert observation["detector_score"] is None
    assert observation["best_score"] is None


def test_a_record_with_no_frame_pointer_carries_nulls_not_empties(tmp_path):
    root = tmp_path / "om"
    store = ObservationStore(
        root, retention_seconds=DEFAULT_RETENTION_SECONDS, clock=lambda: NOW
    )
    store.append(
        _observation("laptop", recorded_at=NOW, session_id=None, frame_seq=None)
    )

    where = _client(root).get(
        "/object-memory/observations"
    ).json()["observations"][0]["where"]

    assert where["session_id"] is None
    assert where["frame_seq"] is None


# --- Filtering and wiring -------------------------------------------------


def test_the_listing_can_be_narrowed_to_one_class(corpus):
    body = _client(corpus).get(
        "/object-memory/observations", params={"object_class": "cell phone"}
    ).json()

    assert body["object_class"] == "cell phone"
    assert body["observation_count"] == 1
    assert body["observations"][0]["object_class"] == "cell phone"


def test_observations_are_newest_first(corpus):
    body = _client(corpus).get("/object-memory/observations").json()
    observed = [o["observed_at"] for o in body["observations"]]

    assert observed == sorted(observed, reverse=True)


def test_an_unconfigured_root_answers_404_on_both_routes(tmp_path):
    app = FastAPI()
    app.include_router(observation_routes.router)
    app.state.object_memory_root = None
    client = TestClient(app)

    assert client.get("/object-memory/observations").status_code == 404
    assert client.get("/object-memory/last-seen/laptop").status_code == 404


def test_the_router_is_registered_on_the_application():
    """The gap this work exists to close: a cartridge with no route."""
    from tower.main import create_app

    # Read off the OpenAPI schema rather than walking `app.routes`: this
    # FastAPI wraps an included router in an opaque `_IncludedRouter` with
    # no `.path`, so the obvious walk silently finds nothing and the
    # assertion passes for the wrong reason.
    paths = create_app().openapi()["paths"]

    assert set(paths["/object-memory/observations"]) == {"get"}
    assert set(paths["/object-memory/last-seen/{object_class}"]) == {"get"}


def test_the_store_reports_the_window_it_will_honour(corpus):
    """The accessor the wire echo is built on. Read-only; it changes nothing."""
    assert _store(corpus, 3650 * DAY).effective_retention_seconds() == (
        DEFAULT_RETENTION_SECONDS
    )
    assert _store(corpus, 1.0 * DAY).effective_retention_seconds() == 1.0 * DAY
    assert _store(corpus, None).effective_retention_seconds() == (
        DEFAULT_RETENTION_SECONDS
    )
