"""An interrupted world stays inspectable: the 2026-09-06 physical walk, as a fixture.

World `fcbca9e9…` on the Windows box is the real thing and is preserved
untouched in the canonical world root. This is its STATE, rebuilt from
synthetic frames so the suite carries no capture data: a session record
still holding the zeros written at start, no `session_stopped`, a lock
naming a pid that is gone, and a derived tree from the last interim build.

What the product must do with that world, on every surface it has:
say `interrupted` rather than `failed`, keep the geometry it has, list it
with the keyframes the journal holds, and serve its picture.
"""

import json
import time

import psutil
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.result_channel_fixtures import start_live_world
from tower.results.world_builder import WorldBuilderStatusProducer
from tower.results.world_builder_library import build_world_listing
from tower.routes import geometry as geometry_routes
from tower.world_builder.store import WorldStore


@pytest.fixture
def interrupted_world(tmp_path):
    root = tmp_path / "worlds"
    world_id, session_id, engine = start_live_world(root, frames=8)
    engine.build(world_id, session_id)          # the last interim rebuild
    store = WorldStore(root)
    dead = next(pid for pid in range(100_000, 200_000) if not psutil.pid_exists(pid))
    store.lock_path(world_id).write_text(json.dumps({"pid": dead}), encoding="utf-8")
    # The engine object still believes its session is open; drop it
    # without touching the disk, as a killed process would.
    engine._session = None
    engine._events = None
    return root, store, world_id, session_id


def _client(store) -> TestClient:
    app = FastAPI()
    app.include_router(geometry_routes.router)
    app.state.world_root = store.root
    return TestClient(app)


def test_the_status_channel_calls_it_interrupted_and_keeps_its_geometry(interrupted_world):
    root, store, world_id, session_id = interrupted_world
    payload = WorldBuilderStatusProducer(root, time.time).snapshot(None, None).payload
    assert payload["lifecycle"]["state"] == "interrupted"
    assert payload["model_state"] == "interrupted"
    assert payload["geometry"]["available"] is True
    assert payload["world_snapshot"]["world_id"] == world_id
    assert payload["world_snapshot"]["keyframe_count"] >= 2
    assert payload["session"]["ended_at"] is None
    assert payload["selection"]["mode"] == "latest"


def test_the_listing_says_interrupted_with_the_journal_count(interrupted_world):
    root, store, world_id, session_id = interrupted_world
    listing = build_world_listing(store)
    session = listing["worlds"][0]["sessions"][0]
    assert session["state"] == "interrupted"
    assert session["abandoned"] is True
    assert session["has_geometry"] is True
    assert session["keyframes_accepted"] == 0          # the record never moved
    assert session["keyframes_journaled"] >= 2         # the journal did
    assert listing["worlds"][0]["live"] is False


def test_its_picture_and_geometry_are_still_served(interrupted_world):
    root, store, world_id, session_id = interrupted_world
    client = _client(store)
    render = client.get(f"/worlds/{world_id}/render")
    assert render.status_code == 200
    assert "text/html" in render.headers["content-type"]
    manifest = client.get(f"/worlds/{world_id}/geometry/manifest", params={"session_id": session_id})
    assert manifest.status_code == 200
    assert manifest.json()["segments"]
