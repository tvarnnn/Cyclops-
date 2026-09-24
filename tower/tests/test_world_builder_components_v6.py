"""Review V6 (RV2) findings on module 3, pinned: L1 (ids are matched whole), L2 (a record
of another solve is not this solve's), L4 (a complete area that cannot be composed is
transient), L6 (copy), M2 (owed areas settle when nothing will build them).

H1, M1, M3 and L3 are in `test_world_builder_refinish.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.test_world_builder_components_areas import (  # noqa: F401 -- fixtures
    AREA1,
    AREA2,
    SESSION,
    WORLD,
    _client,
    _entries,
    _row,
    area_world,
    build_area,
    engine_world,
    old_world,
    write_components,
)

from tower.world_builder import components as C


# -- L1 -------------------------------------------------------------------------


@pytest.mark.parametrize("value", [AREA1 + "\n", AREA1 + " ", "\n" + AREA1, AREA1.upper()])
def test_an_id_is_matched_whole(value):
    assert C.is_area_id(AREA1)
    assert not C.is_area_id(value)


def test_a_record_with_a_newline_in_an_id_is_absent():
    rec = C.parse_components({"components": [
        {"id": "0123456789abcdef", "state": "placed", "reason": None, "reasons": [],
         "shown_as": "room", "keyframes": 4, "capture_spans_s": [[0, 1]]},
        {"id": AREA1 + "\n", "state": "unplaced", "reason": "no-verified-link",
         "reasons": ["no-verified-link"], "shown_as": "area", "keyframes": 40,
         "capture_spans_s": [[2, 9]]}]})
    assert rec is None


# -- L2 -------------------------------------------------------------------------


def _write_with_digest(store, kids, digest):
    path = C.components_path(store, WORLD, SESSION)
    doc = {"components": _entries(kids)}
    if digest is not None:
        doc["input_digest"] = digest
    path.write_text(json.dumps(doc), encoding="utf-8")


def test_a_record_of_the_published_solve_is_served(old_world):
    _write_with_digest(old_world.store, old_world.kids, "digest-1")   # the fixture's solve
    assert C.read_components_record(old_world.store, WORLD, SESSION) is not None
    assert _row(old_world.store)["components"] is not None


def test_a_record_of_another_solve_is_absent(old_world):
    _write_with_digest(old_world.store, old_world.kids, "a-solve-that-no-longer-exists")
    assert C.read_components_record(old_world.store, WORLD, SESSION) is None
    assert _row(old_world.store)["components"] is None
    r = _client(old_world).get(f"/worlds/{WORLD}/areas/{SESSION}/{AREA1}/render")
    assert r.json()["detail"] == "this session has no areas"


def test_a_record_of_a_solve_that_is_gone_is_absent(old_world):
    _write_with_digest(old_world.store, old_world.kids, "digest-1")
    solution = old_world.store.world_dir(WORLD) / "solve" / SESSION / "solution.json"
    solution.rename(solution.with_name("solution.moved.json"))
    assert C.read_components_record(old_world.store, WORLD, SESSION) is None


def test_a_record_that_names_no_solve_is_taken_as_it_is(old_world):
    _write_with_digest(old_world.store, old_world.kids, None)
    assert C.read_components_record(old_world.store, WORLD, SESSION) is not None


def test_a_republished_solve_is_seen_at_once(old_world):
    """The digest cache is keyed by the solution file's stat: a new solve is read."""
    from tower.world_builder.global_solve import load_solution, workspace_for, write_solution

    _write_with_digest(old_world.store, old_world.kids, "digest-1")
    assert C.read_components_record(old_world.store, WORLD, SESSION) is not None
    sol = load_solution(old_world.store, WORLD, SESSION)
    sol.input_digest = "digest-2"
    write_solution(workspace_for(old_world.store, WORLD, SESSION), sol)
    assert C.read_components_record(old_world.store, WORLD, SESSION) is None


# -- L4 -------------------------------------------------------------------------


def test_a_complete_area_that_cannot_be_composed_is_transient(area_world):
    """Its record says both stages `ok`, but nothing under it is drawable right now."""
    for stage in ("surface", "appearance"):
        C.write_area_record(area_world.store, WORLD, SESSION, AREA1,
                            components_sha1=area_world.record.sha1, stage=stage, state="ok")
    client = _client(area_world)
    for tail in ("render", "render/revision"):
        r = client.get(f"/worlds/{WORLD}/areas/{SESSION}/{AREA1}/{tail}")
        assert (r.status_code, r.json()["detail"]) == (404, "this area has not been built yet")


# -- L6 -------------------------------------------------------------------------


def test_areas_nobody_will_show_are_not_counted_in_the_room_caption(area_world):
    client = _client(area_world)
    page = client.get(f"/worlds/{WORLD}/render?viewer=appearance-1").text
    assert "2 more areas shown separately" in page
    for area_id in (AREA1, AREA2):
        for stage in ("surface", "appearance"):
            C.write_area_record(area_world.store, WORLD, SESSION, area_id,
                                components_sha1=area_world.record.sha1, stage=stage,
                                state="unavailable", attempted=False, detail="declined")
    page = client.get(f"/worlds/{WORLD}/render?viewer=appearance-1").text
    assert "more area" not in page
    C.write_area_record(area_world.store, WORLD, SESSION, AREA1,
                        components_sha1=area_world.record.sha1, stage="surface",
                        state="running")
    page = client.get(f"/worlds/{WORLD}/render?viewer=appearance-1").text
    assert "1 more area shown separately" in page


def test_the_area_page_loads_the_areas_surface(area_world):
    build_area(area_world, AREA1, area_world.record)
    client = _client(area_world)
    area = client.get(f"/worlds/{WORLD}/areas/{SESSION}/{AREA1}/render").text
    room = client.get(f"/worlds/{WORLD}/render?viewer=appearance-1").text
    assert 'status("Loading the area\'s surface…");' in area
    assert "Loading the room's surface" not in area
    assert 'status("Loading the room\'s surface…");' in room


def test_an_owed_areas_detail_does_not_repeat_itself(engine_world):
    store, world_id, session_id, kids = engine_world
    write_components(store, _entries(kids), world_id=world_id, session_id=session_id)
    detail = _row(store)["photographic"]["detail"]
    assert detail == "an area of this walk: not built yet, and no process is working on it"


# -- M2 -------------------------------------------------------------------------


@pytest.mark.parametrize("env,expected", [
    ({}, C.AREA_BUILDS_OFF_DETAIL),
    ({"TOWER_WORLD_AREA_BUILDS": "1", "TOWER_WORLD_FINISH_PENDING": "false"},
     C.NO_FINISHER_DETAIL),
    ({"TOWER_WORLD_AREA_BUILDS": "1", "TOWER_WORLD_SURFACE": "0"}, C.NO_FINISHER_DETAIL),
    ({"TOWER_WORLD_AREA_BUILDS": "1", "TOWER_WORLD_SOLVE": "off"}, C.NO_FINISHER_DETAIL),
    ({"TOWER_WORLD_AREA_BUILDS": "1"}, None),
])
def test_who_will_build_the_areas(monkeypatch, env, expected):
    for name in ("TOWER_WORLD_AREA_BUILDS", "TOWER_WORLD_FINISH_PENDING",
                 "TOWER_WORLD_SURFACE", "TOWER_WORLD_SOLVE"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert C.areas_nobody_will_build_reason() == expected


def test_owed_areas_settle_when_nothing_will_build_them(engine_world, monkeypatch):
    """"Owed" always means a finisher will build it; otherwise the builder records the
    decline when the room finishes, and the row stops saying the walk is unfinished."""
    store, world_id, session_id, kids = engine_world
    write_components(store, _entries(kids), world_id=world_id, session_id=session_id)
    monkeypatch.setenv("TOWER_WORLD_AREA_BUILDS", "1")
    monkeypatch.setenv("TOWER_WORLD_FINISH_PENDING", "false")
    assert _row(store)["photographic"]["scope"] == "area"
    settled = C.settle_areas_nobody_will_build(store, world_id, session_id)
    assert sorted(settled) == sorted([AREA1, AREA2])
    row = _row(store)
    assert row["photographic"]["scope"] == "room"
    assert row["state"] != "finalizing"
    areas = [c for c in row["components"] if c["shown_as"] == "area"]
    assert {c["photographic"]["state"] for c in areas} == {"unattempted"}
    assert {c["photographic"]["detail"] for c in areas} == {C.NO_FINISHER_DETAIL}


def test_owed_areas_stay_owed_when_the_finisher_will_build_them(engine_world, monkeypatch):
    store, world_id, session_id, kids = engine_world
    write_components(store, _entries(kids), world_id=world_id, session_id=session_id)
    monkeypatch.setenv("TOWER_WORLD_AREA_BUILDS", "1")
    for name in ("TOWER_WORLD_FINISH_PENDING", "TOWER_WORLD_SURFACE", "TOWER_WORLD_SOLVE"):
        monkeypatch.delenv(name, raising=False)
    assert C.settle_areas_nobody_will_build(store, world_id, session_id) == []
    assert _row(store)["photographic"]["state"] == "owed"


def test_settling_never_touches_a_session_without_a_record(engine_world, monkeypatch):
    store, world_id, session_id, _kids = engine_world
    monkeypatch.delenv("TOWER_WORLD_AREA_BUILDS", raising=False)
    assert C.settle_areas_nobody_will_build(store, world_id, session_id) == []
    assert not (store.world_dir(world_id) / "areas").exists()


def test_the_builder_settles_the_areas_when_the_room_finishes():
    """The builder's post-Stop path calls it after the room's stages, whichever way
    they went (`scripts/world_build_session.py`)."""
    source = (Path(__file__).resolve().parents[1] / "scripts" / "world_build_session.py"
              ).read_text(encoding="utf-8")
    call = "settle_areas_nobody_will_build(store, world_id, session_id)"
    assert source.count(call) == 1
    assert source.index(call) > source.index("report.update(final_surface_stages(")
    assert source.index(call) < source.index("if args.densify:")
