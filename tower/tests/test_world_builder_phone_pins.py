"""What the phone assumes about the Tower (Mac `mac-002`, iOS 4ddcf839), pinned Tower-side.

The phone's side of WORLD-BUILDER-COMPONENTS.md is code-complete and reads the Tower
strictly: a `components` list it cannot trust is treated as null (every area hidden),
an area page whose metas name another area is refused, and a fetch outside the area's
whitelist never reaches the Tower. Each assumption below is one the Tower must keep:

1. `components`: exactly one `placed` entry, first; area ids 16 lower-hex; `id`,
   `state`, `shown_as`, an int `keyframes` and a bool `has_geometry` on every entry;
   `capture_spans_s` a list of `[start, end]` with `0 <= start <= end`. A record that
   breaks any of it is served as null -- on the row AND the revision body -- never
   partially.
2. The area page's `wb-area` and `wb-revision` metas: written exactly as the room page
   writes its own (`<meta name="..." content="...">`, double quotes), within the first
   4096 characters.
3. The area page's CONFIG routes: all four are the area's paths, and none -- in
   particular `R.revision` -- carries a query (the room's carries `?session_id=`).
4. The four area 404 sentences (pinned in `test_world_builder_components_areas.py`).
5. An area's unsettled word always carries `scope: "area"`, on the row and on
   `lifecycle.photographic`; absent `scope` is the room; `build_in_progress` is true
   while a stage runs, for an area as for the room.
"""

from __future__ import annotations

import copy
import json
import os
import re
import time

import pytest

from tests.test_world_builder_components_areas import (  # noqa: F401 -- fixtures
    AREA1,
    AREA2,
    ROOM,
    SESSION,
    WORLD,
    _client,
    _entries,
    _lifecycle,
    _row,
    area_world,
    build_area,
    engine_world,
    old_world,
    write_components,
)

from tower.world_builder import components as C

_HEX16 = re.compile(r"^[0-9a-f]{16}$")


def _assert_phone_readable(comps):
    """Every clause of mac-002 item 1, on a served list."""
    assert isinstance(comps, list) and comps
    assert comps[0]["state"] == "placed"
    assert sum(1 for c in comps if c["state"] == "placed") == 1
    for c in comps:
        for key in ("id", "state", "shown_as", "keyframes", "has_geometry",
                    "capture_spans_s"):
            assert key in c, key
        assert type(c["keyframes"]) is int
        assert type(c["has_geometry"]) is bool
        assert isinstance(c["id"], str) and isinstance(c["state"], str)
        assert isinstance(c["shown_as"], str)
        if c["shown_as"] == "area":
            assert _HEX16.match(c["id"]), c["id"]
        assert isinstance(c["capture_spans_s"], list)
        for span in c["capture_spans_s"]:
            assert isinstance(span, list) and len(span) == 2
            assert 0 <= span[0] <= span[1]


# ---------------------------------------------------------------------------
# 1. components: strictly readable, or null
# ---------------------------------------------------------------------------


def test_a_valid_record_is_served_phone_readable_on_the_row_and_the_revision(area_world):
    from tower.results.world_builder_render import build_render_revision

    build_area(area_world, AREA1, area_world.record)
    row = _row(area_world.store)["components"]
    _assert_phone_readable(row)
    body = build_render_revision(area_world.store, WORLD, SESSION, viewer="appearance-1")
    _assert_phone_readable(body["components"])
    assert body["components"] == row
    # Order on the wire whatever the order in the file: placed first.
    assert row[0]["id"] == ROOM


def _break(mutate):
    def apply(kids):
        entries = _entries(kids)
        mutate(entries)
        return entries
    return apply


_BY_ID = {"short": 0, "area2": 1, "room": 2, "area1": 3}   # positions in `_entries`

MALFORMED = {
    "two placed": _break(lambda es: es[_BY_ID["area1"]].update(
        state="placed", reason=None, reasons=[], shown_as="room")),
    "no placed": _break(lambda es: es[_BY_ID["room"]].update(
        state="unplaced", reason="no-verified-link", reasons=["no-verified-link"],
        shown_as="none")),
    "area id upper-case": _break(lambda es: es[_BY_ID["area1"]].update(id=AREA1.upper())),
    "area id 15 hex": _break(lambda es: es[_BY_ID["area1"]].update(id=AREA1[:15])),
    "area id 17 hex": _break(lambda es: es[_BY_ID["area1"]].update(id=AREA1 + "0")),
    "missing id": _break(lambda es: es[_BY_ID["area2"]].pop("id")),
    "missing state": _break(lambda es: es[_BY_ID["area2"]].pop("state")),
    "missing shown_as": _break(lambda es: es[_BY_ID["area2"]].pop("shown_as")),
    "missing keyframes": _break(lambda es: es[_BY_ID["area2"]].pop("keyframes")),
    "keyframes a string": _break(lambda es: es[_BY_ID["area2"]].update(keyframes="30")),
    "keyframes a float": _break(lambda es: es[_BY_ID["area2"]].update(keyframes=30.0)),
    "keyframes a bool": _break(lambda es: es[_BY_ID["area2"]].update(keyframes=True)),
    "span backwards": _break(lambda es: es[_BY_ID["area2"]].update(
        capture_spans_s=[[25.5, 20.0]])),
    "span negative start": _break(lambda es: es[_BY_ID["area2"]].update(
        capture_spans_s=[[-0.1, 20.0]])),
    "span one number": _break(lambda es: es[_BY_ID["area2"]].update(capture_spans_s=[[1.0]])),
    "spans not a list": _break(lambda es: es[_BY_ID["area2"]].update(capture_spans_s="0-1")),
    "unknown shown_as": _break(lambda es: es[_BY_ID["area2"]].update(shown_as="island")),
}


@pytest.mark.parametrize("name", sorted(MALFORMED))
def test_a_malformed_record_is_null_never_partial(area_world, name):
    """The phone would null the whole list; the Tower never sends it at all, so the
    two sides cannot disagree about which areas exist."""
    from tower.results.world_builder_render import build_render_revision

    build_area(area_world, AREA1, area_world.record)
    assert _row(area_world.store)["components"] is not None
    entries = MALFORMED[name](area_world.kids)
    path = C.components_path(area_world.store, WORLD, SESSION)
    path.write_text(json.dumps({"components": entries}), encoding="utf-8")
    assert _row(area_world.store)["components"] is None
    body = build_render_revision(area_world.store, WORLD, SESSION, viewer="appearance-1")
    assert body["components"] is None
    # And nothing about an area is offered anywhere else either.
    r = _client(area_world).get(f"/worlds/{WORLD}/areas/{SESSION}/{AREA1}/render")
    assert (r.status_code, r.json()["detail"]) == (404, "this session has no areas")


def test_the_served_list_never_carries_the_towers_internal_keys(area_world):
    comps = _row(area_world.store)["components"]
    for c in comps:
        assert "keyframe_ids" not in c


# ---------------------------------------------------------------------------
# 2. the area page's metas, byte format as the room's, in the first 4096 chars
# ---------------------------------------------------------------------------

_META = re.compile(r'<meta name="(wb-[a-z]+)" content="([^"]*)">')


def _metas(page):
    head = page[:4096]
    found = {name: value for name, value in _META.findall(head)}
    for name in found:
        tag = f'<meta name="{name}" content="{found[name]}">'
        assert page.index(tag) + len(tag) <= 4096, name
    return found


@pytest.mark.parametrize("appearance", [True, False])
def test_the_area_metas_are_written_as_the_rooms(area_world, appearance):
    build_area(area_world, AREA1, area_world.record, appearance=appearance)
    client = _client(area_world)
    room = client.get(f"/worlds/{WORLD}/render?viewer=appearance-1"
                      + ("" if appearance else "&representation=surface")).text
    area = client.get(f"/worlds/{WORLD}/areas/{SESSION}/{AREA1}/render").text
    room_metas, area_metas = _metas(room), _metas(area)
    assert "wb-revision" in room_metas and "wb-area" not in room_metas
    assert area_metas["wb-area"] == AREA1
    assert area_metas["wb-revision"].startswith(f"{SESSION}/area:{AREA1}/")
    assert area_metas["wb-representation"] == ("appearance" if appearance else "surface")
    # The same byte form: attribute order, double quotes, no self-closing slash.
    for page, metas in ((room, room_metas), (area, area_metas)):
        for name, value in metas.items():
            assert f'<meta name="{name}" content="{value}">' in page[:4096]
    # The revision the page carries is the one the revision route reports.
    body = client.get(f"/worlds/{WORLD}/areas/{SESSION}/{AREA1}/render/revision").json()
    assert body["revision"] == area_metas["wb-revision"]


# ---------------------------------------------------------------------------
# 3. the area page fetches only the area's paths, with no query
# ---------------------------------------------------------------------------


def _config(page):
    m = re.search(r"const CONFIG = (\{.*?\});\n", page)
    return json.loads(m.group(1))


def test_the_area_pages_routes_are_the_areas_and_carry_no_query(area_world):
    build_area(area_world, AREA1, area_world.record)
    client = _client(area_world)
    area_routes = _config(client.get(f"/worlds/{WORLD}/areas/{SESSION}/{AREA1}/render").text)[
        "routes"]
    room_routes = _config(client.get(f"/worlds/{WORLD}/render?viewer=appearance-1").text)[
        "routes"]
    prefix = f"/worlds/{WORLD}/areas/{SESSION}/{AREA1}/"
    assert set(area_routes) == {"manifest", "chunk", "proxy", "revision"}
    for key, value in area_routes.items():
        assert value.startswith(prefix), key
        assert "?" not in value and "&" not in value, key
    assert area_routes["revision"] == prefix + "render/revision"
    assert area_routes["manifest"] == prefix + "appearance/manifest"
    assert area_routes["chunk"] == prefix + "appearance/chunk/"
    assert area_routes["proxy"] == prefix + "appearance/proxy/"
    # The room's is the one with the query, which the area handler refuses.
    assert "?session_id=" in room_routes["revision"]


def test_the_page_program_fetches_only_through_its_routes():
    """The page composes every address as `R.<route>` (+ a digest): no other
    address, and no query appended to one, anywhere in the program."""
    from tower.world_builder import appearance_render as AR

    template = AR.viewer_template_path().read_text(encoding="utf-8")
    calls = re.findall(r"(?:fetchBytes|fetchJSON)\(([^,]+),", template)
    calls = [c.strip() for c in calls if c.strip() not in ("route",)]
    assert sorted(calls) == sorted(["R.proxy + man.proxy.digest", "R.chunk + digest",
                                    "R.manifest", "R.revision"])
    assert template.count("fetch(") == 1   # inside fetchBytes, on url(route)


# ---------------------------------------------------------------------------
# 4. the four sentences (pinned in test_world_builder_components_areas.py)
# ---------------------------------------------------------------------------


def test_the_four_area_sentences_are_still_the_contracts():
    assert (C.NO_SUCH_AREA, C.AREA_COULD_NOT_BE_BUILT, C.NO_AREAS, C.AREA_NOT_BUILT_YET) == (
        "no such area in this session", "this area could not be built",
        "this session has no areas", "this area has not been built yet")


# ---------------------------------------------------------------------------
# 5. scope and build_in_progress
# ---------------------------------------------------------------------------


def _live_status(path, pid):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"state": "running", "stage": "depth", "pid": pid,
                                "updated_at": time.time()}))


@pytest.mark.parametrize("word", ["owed", "running", "unobservable"])
def test_an_areas_unsettled_word_always_carries_scope_area(engine_world, monkeypatch, word):
    import tower.results.world_builder_render as render

    store, world_id, session_id, kids = engine_world
    record = write_components(store, _entries(kids), world_id=world_id,
                              session_id=session_id)
    for stage in ("surface", "appearance"):
        C.write_area_record(store, world_id, session_id, AREA1,
                            components_sha1=record.sha1, stage=stage, state="ok")
    if word in ("running", "unobservable"):
        C.write_area_record(store, world_id, session_id, AREA2,
                            components_sha1=record.sha1, stage="surface", state="running")
        _live_status(C.area_root(store, world_id, session_id, AREA2) / "surface"
                     / session_id / "status.json", os.getpid())
    if word == "unobservable":
        real = render._stage_running

        def broken(path, is_stale):
            if "areas" in str(path):
                raise OSError("the area's liveness probe is broken")
            return real(path, is_stale)

        monkeypatch.setattr(render, "_stage_running", broken)
    row = _row(store)
    lifecycle = _lifecycle(store, world_id, session_id)
    assert row["photographic"]["state"] == word
    assert row["photographic"]["scope"] == "area"
    assert lifecycle["photographic"] == row["photographic"]
    assert lifecycle["state"] == "finalizing" and row["state"] == "finalizing"
    if word == "running":
        assert lifecycle["build_in_progress"] is True
    if word == "owed":
        assert lifecycle["build_in_progress"] is False


def test_a_running_room_is_scope_room_and_build_in_progress(engine_world):
    store, world_id, session_id, kids = engine_world
    write_components(store, _entries(kids), world_id=world_id, session_id=session_id)
    s = store.read_session(world_id, session_id)
    import dataclasses

    store.write_session(dataclasses.replace(
        s, stages={"surface": {"state": "running", "attempted": True}}))
    _live_status(store.world_dir(world_id) / "surface" / session_id / "status.json",
                 os.getpid())
    lifecycle = _lifecycle(store, world_id, session_id)
    assert lifecycle["photographic"]["state"] == "running"
    assert lifecycle["photographic"]["scope"] == "room"
    assert lifecycle["build_in_progress"] is True
    assert _row(store)["photographic"]["scope"] == "room"


def test_absent_scope_is_the_room_on_every_older_session(engine_world):
    store, world_id, session_id, _kids = engine_world
    row = _row(store)
    lifecycle = _lifecycle(store, world_id, session_id)
    assert row["components"] is None
    assert "scope" not in row["photographic"]
    assert "scope" not in lifecycle["photographic"]
