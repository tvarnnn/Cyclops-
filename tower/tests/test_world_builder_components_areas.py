"""Components on the wire, the row's `scope`, E13, and the area routes.

Contract: docs/contracts/WORLD-BUILDER-COMPONENTS.md (§2, §3.1-3.4, §5, §7), with the
Mac's C1 edits E3 (`scope`), E4 (the four stable 404 sentences) and E13
(`build_in_progress` for a running area).

The components record is the evidence gate's (P3.2 module 2). It is written here by a
FIXTURE WRITER in the contract's shape, so these tests pin what the reader and the
routes do with it, not how the gate decides it. The area artifacts are built by the
REAL appearance pipeline through `components.AreaStore`, from the synthetic solved,
surfaced world `test_world_builder_appearance.World` builds -- so the area routes are
tested against an artifact the product itself wrote.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import shutil
import time

import pytest

from tests.test_world_builder_appearance import SESSION, WORLD, World, _app, _never_redact

from tower.world_builder import appearance as A
from tower.world_builder import appearance_pipeline as AP
from tower.world_builder import components as C

ROOM = "0123456789abcdef"
AREA1 = "a1a1a1a1a1a1a1a1"
AREA2 = "a2a2a2a2a2a2a2a2"
SHORT = "0000000000000001"

FINALIZED = {"state": "complete", "final_solve": "solved",
             "started_at": 1.0, "updated_at": 2.0, "detail": ""}
ROOM_DONE = {"surface": {"state": "ok", "attempted": True},
             "appearance": {"state": "ok", "attempted": True}}


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _finalize(world, stages=None):
    s = world.store.read_session(WORLD, SESSION)
    world.store.write_session(dataclasses.replace(
        s, ended_at=5.0, end_reason="stop", finalization=dict(FINALIZED),
        stages=dict(stages if stages is not None else ROOM_DONE)))


def _entries(kids):
    """A record in the contract's shape: the room, two areas and a short stretch."""
    return [
        {"id": SHORT, "state": "unplaced", "reason": "no-verified-link",
         "reasons": ["no-verified-link"], "shown_as": "none", "keyframes": 1,
         "capture_spans_s": [[40.0, 41.0]], "keyframe_ids": kids[7:8]},
        {"id": AREA2, "state": "unplaced", "reason": "solved-separately",
         "reasons": ["solved-separately"], "shown_as": "area", "keyframes": 30,
         "capture_spans_s": [[20.0, 25.5]], "keyframe_ids": kids[5:7]},
        {"id": ROOM, "state": "placed", "reason": None, "reasons": [], "shown_as": "room",
         "keyframes": 128, "capture_spans_s": [[0.0, 10.0], [30.0, 39.0]],
         "keyframe_ids": kids[0:3]},
        {"id": AREA1, "state": "unplaced", "reason": "single-unconfirmed-link",
         "reasons": ["single-unconfirmed-link", "scale-mismatch"], "shown_as": "area",
         "keyframes": 55, "capture_spans_s": [[86.7, 97.1], [106.4, 109.6]],
         "keyframe_ids": kids[3:5]},
    ]


def write_components(store, entries, *, world_id=WORLD, session_id=SESSION):
    """THE FIXTURE WRITER: `solve/<session>/components.json` as the gate writes it."""
    path = C.components_path(store, world_id, session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"components": entries, "gate": "fixture"}), encoding="utf-8")
    return C.read_components_record(store, world_id, session_id)


def build_area(world, area_id, record, *, appearance=True):
    """An area built by the real appearance pipeline, through `AreaStore`, from a copy
    of the room's solve / depth / surface (the geometry is the test world's box)."""
    view = C.AreaStore(world.store, WORLD, SESSION, area_id)
    for stage in ("solve", "dense", "surface"):
        shutil.copytree(world.store.world_dir(WORLD) / stage / SESSION,
                        view.area_dir / stage / SESSION,
                        ignore=shutil.ignore_patterns(C.COMPONENTS_FILENAME))
    C.write_area_record(world.store, WORLD, SESSION, area_id, components_sha1=record.sha1,
                        stage="surface", state="ok", levelled=True)
    if appearance:
        result = AP.build_appearance(view, WORLD, SESSION,
                                     params=A.AppearanceParams(selection_samples=4000),
                                     device="cpu", redactor_factory=_never_redact)
        assert result.state == AP.STATE_OK, result.detail
        C.write_area_record(world.store, WORLD, SESSION, area_id,
                            components_sha1=record.sha1, stage="appearance", state="ok")
    else:
        C.write_area_record(world.store, WORLD, SESSION, area_id,
                            components_sha1=record.sha1, stage="appearance",
                            state="unavailable", attempted=False, detail="not requested")
    return view


@pytest.fixture
def old_world(tmp_path):
    """A finished world with a photographic room and NO components record: every
    world on every Tower today."""
    world = World(tmp_path)
    world.build(redactor_factory=_never_redact)
    _finalize(world)
    return world


@pytest.fixture
def area_world(old_world):
    record = write_components(old_world.store, _entries(old_world.kids))
    old_world.record = record
    return old_world


@pytest.fixture
def engine_world(tmp_path):
    """A world built by the REAL engine -- observed, stopped (journal and all), built
    -- with a finished photographic room recorded, so the status channel's lifecycle
    reaches `_still_building` exactly as it does after a real walk. Returns
    `(store, world_id, session_id, keyframe_ids)`."""
    from tests.result_channel_fixtures import build_world
    from tower.world_builder.store import WorldStore

    root = tmp_path / "worlds"
    world_id, session_id = build_world(root, frames=6)
    store = WorldStore(root)
    s = store.read_session(world_id, session_id)
    store.write_session(dataclasses.replace(s, finalization=dict(FINALIZED),
                                            stages=dict(ROOM_DONE)))
    kids = [k.keyframe_id for k in store.read_keyframes(world_id, session_id)]
    kids = (kids * 8)[:8]
    return store, world_id, session_id, kids


def _row(store):
    return next(s for w in __import__("tower.results.world_builder_library",
                                      fromlist=["x"]).build_world_listing(store)["worlds"]
                for s in w["sessions"])


def _lifecycle(store, world_id=WORLD, session_id=SESSION):
    from tower.results.world_builder import WorldBuilderStatusProducer

    snap = WorldBuilderStatusProducer(store.root, lambda: 1e9).snapshot(
        world_id=world_id, session_id=session_id)
    return getattr(snap, "payload", snap)["lifecycle"]


def _client(world):
    from fastapi.testclient import TestClient

    return TestClient(_app(world.root))


def _meta(html, name):
    m = re.search(rf'<meta name="{name}" content="([^"]*)">', html[:4096])
    return m.group(1) if m else None


def _config(html):
    m = re.search(r"const CONFIG = (\{.*?\});\n", html)
    assert m, "the page carries its configuration"
    return json.loads(m.group(1))


def _area_url(area_id, tail="render", world_id=WORLD, session_id=SESSION):
    return f"/worlds/{world_id}/areas/{session_id}/{area_id}/{tail}"


# ---------------------------------------------------------------------------
# the record: shape, order, refusal
# ---------------------------------------------------------------------------


class TestTheRecord:

    def test_it_is_read_in_the_contracts_order(self, area_world):
        record = area_world.record
        assert [e["id"] for e in record.entries] == [ROOM, AREA1, AREA2, SHORT]
        assert [e["id"] for e in record.areas()] == [AREA1, AREA2]

    def test_ties_are_broken_by_the_earliest_span(self):
        base = {"state": "unplaced", "reason": "no-verified-link",
                "reasons": ["no-verified-link"], "shown_as": "area", "keyframes": 40}
        room = {"id": ROOM, "state": "placed", "reason": None, "reasons": [],
                "shown_as": "room", "keyframes": 9, "capture_spans_s": [[0, 1]]}
        late = dict(base, id=AREA1, capture_spans_s=[[50.0, 60.0]])
        early = dict(base, id=AREA2, capture_spans_s=[[5.0, 9.0]])
        rec = C.parse_components({"components": [late, room, early]})
        assert [e["id"] for e in rec.entries] == [ROOM, AREA2, AREA1]

    @pytest.mark.parametrize("mutate", [
        lambda es: es.pop(2),                                     # no placed entry
        lambda es: es[0].update(id="NOT-HEX-0123456"),            # id alphabet
        lambda es: es[0].update(id="0123"),                       # id length
        lambda es: es[1].update(id=ROOM),                         # duplicate id
        lambda es: es[2].update(shown_as="area"),                 # placed but not room
        lambda es: es[1].update(reasons=[]),                      # unplaced, no reason
        lambda es: es[1].update(reason="scale-mismatch"),         # reason is not reasons[0]
        lambda es: es[1].update(shown_as="island"),               # unknown shown_as
        lambda es: es[1].update(keyframes=True),                  # a bool is not a count
        lambda es: es[1].update(capture_spans_s=[[1, 0]]),        # a backwards span
        lambda es: es[1].update(capture_spans_s=[[i, i + 1] for i in range(9)]),  # > 8
        lambda es: es[1].update(keyframe_ids="kf"),               # not a list
    ])
    def test_a_record_that_is_not_the_contracts_shape_is_absent(self, tmp_path, mutate):
        from tower.world_builder.store import WorldStore

        entries = _entries([f"k{i}" for i in range(8)])
        mutate(entries)
        store = WorldStore(tmp_path)
        assert write_components(store, entries) is None

    def test_an_empty_or_unparseable_record_is_absent(self, tmp_path):
        from tower.world_builder.store import WorldStore

        store = WorldStore(tmp_path)
        assert write_components(store, []) is None
        C.components_path(store, WORLD, SESSION).write_text("{not json", encoding="utf-8")
        assert C.read_components_record(store, WORLD, SESSION) is None


# ---------------------------------------------------------------------------
# §7: every world today has no components, and nothing about it changes
# ---------------------------------------------------------------------------


class TestAnOldWorld:

    def test_its_row_says_null_and_its_photographic_block_is_unchanged(self, old_world):
        row = _row(old_world.store)
        assert "components" in row and row["components"] is None
        # No `scope` on a session without a record: the block is byte for byte
        # what it was (absent `scope` means "room", §3.4).
        assert set(row["photographic"]) == {"state", "stage", "detail"}
        assert row["photographic"]["state"] == "complete"
        assert row["state"] == "complete"

    def test_its_status_block_is_unchanged(self, engine_world):
        store, world_id, session_id, _kids = engine_world
        row = _row(store)
        assert row["components"] is None
        lifecycle = _lifecycle(store, world_id, session_id)
        assert lifecycle["photographic"] == row["photographic"]
        assert set(lifecycle["photographic"]) == {"state", "stage", "detail"}
        assert lifecycle["state"] == "ready"

    def test_its_revision_says_null_and_is_otherwise_unchanged(self, old_world):
        from tower.results.world_builder_render import build_render_revision

        body = build_render_revision(old_world.store, WORLD, SESSION, viewer="appearance-1")
        assert body["components"] is None
        assert set(body) == {"session_id", "representation", "revision", "live",
                             "appearance", "components"}
        assert body["representation"] == "appearance"

    @pytest.mark.parametrize("area_id", [AREA1, ROOM, "zzzz", "0" * 16, "A1A1A1A1A1A1A1A1"])
    @pytest.mark.parametrize("tail", ["render", "render/revision", "appearance/manifest",
                                      "appearance/chunk/" + "0" * 32,
                                      "appearance/proxy/" + "0" * 32])
    def test_no_area_route_answers_but_this_session_has_no_areas(self, old_world, area_id,
                                                                  tail):
        r = _client(old_world).get(_area_url(area_id, tail))
        assert r.status_code == 404
        assert r.json()["detail"] == "this session has no areas"

    def test_the_room_page_is_the_same_page_with_or_without_a_record(self, old_world):
        """§4: the room page draws the placed component exactly as today; areas are
        never a rung of the room's ladder and never drawn on its pages."""
        client = _client(old_world)
        before = client.get(f"/worlds/{WORLD}/render?viewer=appearance-1")
        write_components(old_world.store, _entries(old_world.kids))
        after = client.get(f"/worlds/{WORLD}/render?viewer=appearance-1")
        assert before.status_code == after.status_code == 200
        assert before.text == after.text
        assert "/areas/" not in after.text


# ---------------------------------------------------------------------------
# §2 on the wire
# ---------------------------------------------------------------------------


class TestOnTheWire:

    def test_the_row_carries_exactly_the_contracts_fields_in_order(self, area_world):
        build_area(area_world, AREA1, area_world.record)
        row = _row(area_world.store)
        comps = row["components"]
        assert [c["id"] for c in comps] == [ROOM, AREA1, AREA2, SHORT]
        fields = {"id", "state", "reason", "reasons", "shown_as", "keyframes",
                  "keyframes_phone", "capture_spans_s", "has_geometry", "photographic"}
        for c in comps:
            assert set(c) == fields, "no other field, and never keyframe_ids (§2.1)"
        room, a1, a2, short = comps
        assert room["state"] == "placed" and room["reason"] is None and room["reasons"] == []
        assert room["shown_as"] == "room"
        # The room's figures ARE the row's own.
        assert room["has_geometry"] is row["has_geometry"] is True
        assert room["keyframes_phone"] == row["appearance"]["keyframes_phone"]
        assert room["photographic"]["state"] == "complete"
        assert a1["reason"] == "single-unconfirmed-link"
        assert a1["reasons"] == ["single-unconfirmed-link", "scale-mismatch"]
        assert a1["capture_spans_s"] == [[86.7, 97.1], [106.4, 109.6]]
        assert a1["has_geometry"] is True
        assert a1["keyframes_phone"] > 0
        assert a1["photographic"]["state"] == "complete"
        # The unbuilt area: nothing to draw, and it is owed.
        assert a2["has_geometry"] is False and a2["keyframes_phone"] is None
        assert a2["photographic"]["state"] == "owed"
        # Counted only: never built, never drawable, no build word.
        assert short["has_geometry"] is False and short["photographic"] is None
        assert short["keyframes_phone"] is None

    def test_the_revision_carries_the_same_array_outside_the_revision(self, area_world):
        from tower.results.world_builder_render import build_render_revision

        build_area(area_world, AREA1, area_world.record)
        body = build_render_revision(area_world.store, WORLD, SESSION, viewer="appearance-1")
        assert body["components"] == _row(area_world.store)["components"]
        before = body["revision"]
        build_area(area_world, AREA2, area_world.record)
        after = build_render_revision(area_world.store, WORLD, SESSION, viewer="appearance-1")
        # An area finishing moves the array and never the room's revision (§3.2).
        assert after["revision"] == before
        assert after["components"] != body["components"]

    def test_the_route_serves_it(self, area_world):
        r = _client(area_world).get("/worlds")
        assert r.status_code == 200
        assert r.json()["contract"] == "world_builder.worlds/2026-09-10"
        comps = r.json()["worlds"][0]["sessions"][0]["components"]
        assert [c["id"] for c in comps] == [ROOM, AREA1, AREA2, SHORT]


# ---------------------------------------------------------------------------
# §3.4: the row's photographic covers the room and its areas; scope; E13
# ---------------------------------------------------------------------------


class TestScopeAndLiveness:

    @staticmethod
    def _built_by_record(store, world_id, session_id, record, area_id):
        for stage in ("surface", "appearance"):
            C.write_area_record(store, world_id, session_id, area_id,
                                components_sha1=record.sha1, stage=stage, state="ok")

    def test_an_owed_area_keeps_the_world_finalizing_with_scope_area(self, engine_world):
        store, world_id, session_id, kids = engine_world
        record = write_components(store, _entries(kids), world_id=world_id,
                                  session_id=session_id)
        self._built_by_record(store, world_id, session_id, record, AREA1)  # AREA2 owed
        row = _row(store)
        ph = row["photographic"]
        assert ph["scope"] == "area" and ph["state"] == "owed"
        assert ph["detail"].startswith("an area of this walk")
        assert AREA2 not in ph["detail"] and "Area" not in ph["detail"]  # no name, no number
        assert row["state"] == "finalizing"
        lifecycle = _lifecycle(store, world_id, session_id)
        assert lifecycle["photographic"] == ph
        assert lifecycle["state"] == "finalizing"
        assert lifecycle["build_in_progress"] is False     # owed: nothing IS running
        # Settled: the row's word comes back, scoped to the room.
        self._built_by_record(store, world_id, session_id, record, AREA2)
        assert _row(store)["photographic"]["scope"] == "room"
        lifecycle = _lifecycle(store, world_id, session_id)
        assert lifecycle["state"] == "ready"
        assert lifecycle["photographic"]["scope"] == "room"

    def test_a_running_area_is_build_in_progress(self, engine_world):
        """C1 E13: `running` => `build_in_progress: true`, for an area as for the room.
        The liveness probe sees the AREA's status file."""
        store, world_id, session_id, kids = engine_world
        record = write_components(store, _entries(kids), world_id=world_id,
                                  session_id=session_id)
        self._built_by_record(store, world_id, session_id, record, AREA1)
        view = C.AreaStore(store, world_id, session_id, AREA2)
        C.write_area_record(store, world_id, session_id, AREA2,
                            components_sha1=record.sha1, stage="surface", state="running")
        status = view.world_dir(world_id) / "surface" / session_id / "status.json"
        status.parent.mkdir(parents=True, exist_ok=True)
        status.write_text(json.dumps({"state": "running", "stage": "depth",
                                      "pid": os.getpid(), "updated_at": time.time()}))
        row = _row(store)
        assert row["photographic"]["state"] == "running"
        assert row["photographic"]["scope"] == "area"
        assert row["state"] == "finalizing"
        lifecycle = _lifecycle(store, world_id, session_id)
        assert lifecycle["state"] == "finalizing"
        assert lifecycle["build_in_progress"] is True
        assert lifecycle["build_in_progress_unavailable_reason"] is None
        assert lifecycle["photographic"]["scope"] == "area"
        # The area's own revision says it is live.
        from tower.results.world_builder_render import _area_live, resolve_area

        assert _area_live(resolve_area(store, world_id, session_id, AREA2)) is True
        # ...and when the process is gone, it is owed again, never "running".
        status.write_text(json.dumps({"state": "running", "stage": "depth",
                                      "pid": 2 ** 22 + 7, "updated_at": 1.0}))
        assert _row(store)["photographic"]["state"] == "owed"
        assert _lifecycle(store, world_id, session_id)["build_in_progress"] is False

    def test_an_area_that_failed_never_reaches_the_row(self, area_world):
        build_area(area_world, AREA1, area_world.record)
        C.write_area_record(area_world.store, WORLD, SESSION, AREA2,
                            components_sha1=area_world.record.sha1, stage="surface",
                            state="failed", detail="depth network unavailable")
        row = _row(area_world.store)
        assert row["photographic"] == {"state": "complete", "stage": "appearance",
                                       "detail": "the appearance stage finished",
                                       "scope": "room"}
        assert row["state"] == "complete"
        a2 = row["components"][2]
        assert a2["photographic"]["state"] == "failed"
        assert a2["has_geometry"] is False

    def test_an_unsettled_room_outranks_its_areas(self, area_world):
        _finalize(area_world, stages={"surface": {"state": "ok"},
                                      "appearance": {"state": "stopped"}})
        ph = _row(area_world.store)["photographic"]
        assert ph["scope"] == "room" and ph["state"] == "owed"
        assert ph["stage"] == "appearance"

    def test_a_record_for_another_solve_is_not_this_areas_build(self, area_world):
        build_area(area_world, AREA1, area_world.record)
        build_area(area_world, AREA2, area_world.record)
        assert _row(area_world.store)["photographic"]["scope"] == "room"
        # The gate rewrote the record (a new solve): every area is owed again.
        entries = _entries(area_world.kids)
        entries[1]["keyframes"] = 31
        write_components(area_world.store, entries)
        ph = _row(area_world.store)["photographic"]
        assert ph["scope"] == "area" and ph["state"] == "owed"


# ---------------------------------------------------------------------------
# §5: the area routes
# ---------------------------------------------------------------------------


class TestTheAreaRoutes:

    def test_the_four_sentences_are_pinned(self):
        """C1 E4: stable identifiers, compared for equality by the phone. Changing one
        is a contract change."""
        assert C.NO_SUCH_AREA == "no such area in this session"
        assert C.NO_AREAS == "this session has no areas"
        assert C.AREA_NOT_BUILT_YET == "this area has not been built yet"
        assert C.AREA_COULD_NOT_BE_BUILT == "this area could not be built"

    def test_the_appearance_page_is_the_rooms_program_on_the_areas_addresses(self,
                                                                              area_world):
        build_area(area_world, AREA1, area_world.record)
        r = _client(area_world).get(_area_url(AREA1))
        assert r.status_code == 200
        assert r.headers["cache-control"] == "no-store"
        assert r.headers["content-type"].startswith("text/html")
        csp = r.headers["content-security-policy"]
        assert csp.endswith("connect-src glasses-world:")
        html = r.text
        assert _meta(html, "wb-representation") == "appearance"
        epoch = AP.read_appearance_manifest(C.AreaStore(area_world.store, WORLD, SESSION,
                                                        AREA1), WORLD, SESSION)["epoch"]
        assert _meta(html, "wb-revision") == f"{SESSION}/area:{AREA1}/appearance:1@{epoch}"
        assert _meta(html, "wb-area") == AREA1
        config = _config(html)
        prefix = f"/worlds/{WORLD}/areas/{SESSION}/{AREA1}"
        assert config["routes"] == {"manifest": f"{prefix}/appearance/manifest",
                                    "chunk": f"{prefix}/appearance/chunk/",
                                    "proxy": f"{prefix}/appearance/proxy/",
                                    "revision": f"{prefix}/render/revision"}
        assert config["area"] == {"id": AREA1, "levelled": True}
        assert config["base"] == "glasses-world://tower"
        # Never the room's routes (§5.6 rule 2).
        assert f"/worlds/{WORLD}/appearance/" not in html
        assert f"/worlds/{WORLD}/render/revision" not in html

    def test_viewer_is_accepted_and_ignored(self, area_world):
        build_area(area_world, AREA1, area_world.record)
        client = _client(area_world)
        plain = client.get(_area_url(AREA1))
        declared = client.get(_area_url(AREA1) + "?viewer=anything,at-all")
        assert declared.status_code == 200
        assert _meta(declared.text, "wb-representation") == "appearance"
        assert _meta(plain.text, "wb-revision") == _meta(declared.text, "wb-revision")

    def test_the_tower_transport_is_self(self, area_world):
        build_area(area_world, AREA1, area_world.record)
        r = _client(area_world).get(_area_url(AREA1) + "?transport=tower")
        assert r.headers["content-security-policy"].endswith("connect-src 'self'")
        assert _config(r.text)["base"] == ""

    def test_the_revision_names_the_same_page(self, area_world):
        build_area(area_world, AREA1, area_world.record)
        client = _client(area_world)
        page = client.get(_area_url(AREA1)).text
        r = client.get(_area_url(AREA1, "render/revision"))
        assert r.status_code == 200 and r.headers["cache-control"] == "no-store"
        body = r.json()
        assert set(body) == {"session_id", "area_id", "representation", "revision", "live",
                             "appearance"}
        assert body["session_id"] == SESSION and body["area_id"] == AREA1
        assert body["representation"] == "appearance"
        assert body["revision"] == _meta(page, "wb-revision")
        assert body["revision"].startswith(f"{SESSION}/area:{AREA1}/")
        assert body["live"] is False
        assert body["appearance"]["state"] == "served"
        assert body["appearance"]["current"] is True
        assert set(body["appearance"]) == {"revision", "current", "state", "epoch"}

    def test_a_surface_only_area_serves_the_strict_surface_page(self, area_world):
        build_area(area_world, AREA1, area_world.record, appearance=False)
        client = _client(area_world)
        r = client.get(_area_url(AREA1))
        assert r.status_code == 200
        assert r.headers["content-security-policy"] == (
            "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'")
        assert _meta(r.text, "wb-representation") == "surface"
        built = json.loads((C.AreaStore(area_world.store, WORLD, SESSION, AREA1).area_dir
                            / "surface" / SESSION / "manifest.json").read_text())["built_at"]
        assert _meta(r.text, "wb-revision") == f"{SESSION}/area:{AREA1}/surface:{built}"
        assert _meta(r.text, "wb-area") == AREA1
        rev = client.get(_area_url(AREA1, "render/revision")).json()
        assert rev["representation"] == "surface"
        assert rev["revision"] == _meta(r.text, "wb-revision")
        pinned = client.get(_area_url(AREA1) + "?representation=appearance")
        assert pinned.status_code == 404

    @pytest.mark.parametrize("area_id", [ROOM, SHORT, "ffffffffffffffff", "not-an-id",
                                         AREA1.upper(), AREA1 + "0"])
    def test_no_such_area_in_this_session(self, area_world, area_id):
        client = _client(area_world)
        for tail in ("render", "render/revision", "appearance/manifest"):
            r = client.get(_area_url(area_id, tail))
            assert r.status_code == 404
            assert r.json()["detail"] == "no such area in this session"

    def test_not_built_yet_is_transient_and_could_not_be_built_is_terminal(self,
                                                                           area_world):
        client = _client(area_world)
        for tail in ("render", "render/revision"):
            r = client.get(_area_url(AREA2, tail))
            assert (r.status_code, r.json()["detail"]) == (
                404, "this area has not been built yet")
        C.write_area_record(area_world.store, WORLD, SESSION, AREA2,
                            components_sha1=area_world.record.sha1, stage="surface",
                            state="failed", detail="boom")
        for tail in ("render", "render/revision"):
            r = client.get(_area_url(AREA2, tail))
            assert (r.status_code, r.json()["detail"]) == (404, "this area could not be built")

    def test_sparse_and_dense_are_404_and_anything_else_is_422(self, area_world):
        build_area(area_world, AREA1, area_world.record)
        client = _client(area_world)
        for rung in ("sparse", "dense"):
            r = client.get(_area_url(AREA1) + f"?representation={rung}")
            assert (r.status_code, r.json()["detail"]) == (
                404, "an area has no sparse or dense rung")
        assert client.get(_area_url(AREA1) + "?representation=mesh").status_code == 422
        assert client.get(_area_url(AREA1) + "?max_points=0").status_code == 422
        assert client.get(_area_url(AREA1) + "?transport=cdn").status_code == 422

    def test_world_and_session_404s_are_the_rooms(self, area_world):
        client = _client(area_world)
        r = client.get(_area_url(AREA1, world_id="nope"))
        assert (r.status_code, r.json()["detail"]) == (404, "no world 'nope'")
        r = client.get(_area_url(AREA1, session_id="s9"))
        assert (r.status_code, r.json()["detail"]) == (404, "world 'w1' has no session 's9'")
        r = client.get(_area_url(AREA1, world_id="..%5C..%5Cetc"))
        assert r.status_code == 404

    def test_the_appearance_routes_serve_the_areas_artifact_privately(self, area_world):
        build_area(area_world, AREA1, area_world.record)
        client = _client(area_world)
        r = client.get(_area_url(AREA1, "appearance/manifest"),
                       headers={"accept-encoding": "identity"})
        assert r.status_code == 200
        for key, value in (("cache-control", "no-store"), ("pragma", "no-cache"),
                           ("x-content-type-options", "nosniff")):
            assert r.headers[key] == value
        assert "etag" not in r.headers and "last-modified" not in r.headers
        assert r.headers["x-world-redaction"]
        assert r.headers["x-world-imagery"] == "redacted"
        man = r.json()
        assert man["format"] == A.APPEARANCE_FORMAT
        assert man["area"] == {"id": AREA1, "levelled": True}
        assert man["currency"]["current"] is True
        proxy = client.get(_area_url(AREA1, f"appearance/proxy/{man['proxy']['digest']}"))
        assert proxy.status_code == 200
        assert proxy.headers["cache-control"] == "no-store"
        digest = next(iter(man["keyframes"][0]["chunks"].values()))["digest"]
        chunk = client.get(_area_url(AREA1, f"appearance/chunk/{digest}"),
                           headers={"accept-encoding": "gzip"})
        assert chunk.status_code == 200
        assert chunk.headers["x-world-imagery"] == "redacted"
        bad = client.get(_area_url(AREA1, "appearance/chunk/" + "0" * 32))
        assert bad.status_code == 404 and bad.headers["cache-control"] == "no-store"
        assert client.get(_area_url(AREA1, "appearance/chunk/xyz")).status_code == 404
        # The room's artifact is never answered on an area path, nor the reverse.
        room_digest = next(iter(
            AP.read_appearance_manifest(area_world.store, WORLD, SESSION)["keyframes"][0]
            ["chunks"].values()))["digest"]
        if room_digest != digest:
            assert client.get(
                _area_url(AREA1, f"appearance/chunk/{room_digest}")).status_code == 404

    def test_a_relabelled_session_withdraws_the_areas_imagery(self, area_world):
        build_area(area_world, AREA1, area_world.record)
        area_world.set_label("none")
        _finalize(area_world)
        client = _client(area_world)
        r = client.get(_area_url(AREA1, "appearance/manifest"))
        assert r.status_code == 404
        assert r.json()["detail"] == AP.STALE_LABEL_DETAIL
        rev = client.get(_area_url(AREA1, "render/revision"))
        # The surface is still drawable; the appearance is not served.
        assert rev.status_code == 200
        assert rev.json()["appearance"]["revision"] is None

    def test_a_purged_world_refuses_the_areas_imagery(self, area_world):
        build_area(area_world, AREA1, area_world.record)
        world = area_world.store.read_world(WORLD)
        area_world.store.write_world(dataclasses.replace(world, images_purged=True))
        r = _client(area_world).get(_area_url(AREA1, "appearance/manifest"))
        assert r.status_code == 404
        assert r.json()["detail"] == "appearance imagery was purged"

    def test_an_area_inherits_no_scale(self, area_world):
        from tower.world_builder.records import ScaleState

        world = area_world.store.read_world(WORLD)
        area_world.store.write_world(dataclasses.replace(
            world, scale=ScaleState(state="metric", meters_per_unit=0.5)))
        view = C.AreaStore(area_world.store, WORLD, SESSION, AREA1)
        assert view.read_world(WORLD).scale.state == "unknown"
        assert area_world.store.read_world(WORLD).scale.state == "metric"
        # Everything about the session is the real world's.
        assert view.session_dir(WORLD, SESSION) == area_world.store.session_dir(WORLD, SESSION)
        assert view.world_dir(WORLD) == area_world.store.world_dir(WORLD) / "areas" / AREA1
