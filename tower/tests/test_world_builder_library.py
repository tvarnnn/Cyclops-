"""`GET /worlds`: the index a viewer opens old worlds from."""

import json
import os

from fastapi import FastAPI
from fastapi.testclient import TestClient

from tower.results.world_builder_library import WORLDS_CONTRACT, build_world_listing
from tower.routes import geometry as geometry_routes
from tower.world_builder.store import WorldStore


def _client(store) -> TestClient:
    app = FastAPI()
    app.include_router(geometry_routes.router)
    app.state.world_root = store.root
    return TestClient(app)


def test_listing_names_the_derived_world_and_its_session(derived_world):
    store, world_id, session_id = derived_world
    listing = build_world_listing(store)
    assert listing["contract"] == WORLDS_CONTRACT
    assert listing["world_count"] == 1
    world = listing["worlds"][0]
    assert world["world_id"] == world_id
    assert world["live"] is False
    assert world["session_count"] == 1
    session = world["sessions"][0]
    assert session["session_id"] == session_id
    assert session["has_geometry"] is True
    assert "ended_at" in session and "end_reason" in session


def test_a_live_lock_marks_the_world_live_and_a_dead_one_does_not(derived_world):
    store, world_id, _ = derived_world
    lock = store.lock_path(world_id)
    lock.parent.mkdir(parents=True, exist_ok=True)
    # A pid that cannot be running: pid 0 is never a user process holder.
    lock.write_text(json.dumps({"pid": 999999999}))
    assert build_world_listing(store)["worlds"][0]["live"] is False
    # This process is running, but it is *us*; a viewer asking from inside
    # the writer would not call that "live", and the store's own lock
    # logic makes the same exception.
    lock.write_text(json.dumps({"pid": os.getpid()}))
    assert build_world_listing(store)["worlds"][0]["live"] is False


def test_route_serves_the_listing_and_404s_without_a_root(derived_world):
    store, world_id, _ = derived_world
    client = _client(store)
    response = client.get("/worlds")
    assert response.status_code == 200
    body = response.json()
    assert body["contract"] == WORLDS_CONTRACT
    assert body["worlds"][0]["world_id"] == world_id

    app = FastAPI()
    app.include_router(geometry_routes.router)
    assert TestClient(app).get("/worlds").status_code == 404


def test_an_unreadable_world_is_omitted_not_invented(derived_world, tmp_path):
    store, world_id, _ = derived_world
    broken = store.world_dir("deadbeefdeadbeefdeadbeefdeadbeef")
    broken.mkdir(parents=True)
    (broken / "world.json").write_text("{not json")
    listing = build_world_listing(store)
    assert [w["world_id"] for w in listing["worlds"]] == [world_id]


def test_worlds_are_newest_first_and_sessions_oldest_first(tmp_path):
    from tower.world_builder.engine import WorldBuilderEngine

    store = WorldStore(tmp_path)
    engine = WorldBuilderEngine(store)
    first = engine.create_world("first")
    second = engine.create_world("second")
    listing = build_world_listing(store)
    ids = [w["world_id"] for w in listing["worlds"]]
    assert ids.index(second) < ids.index(first)
    assert listing["worlds"][0]["display_name"] == "second"
    assert all(w["sessions"] == [] for w in listing["worlds"])
