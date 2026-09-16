"""`GET /worlds/{id}/render/revision` as the phone follows it, and the CSP
every render page carries in its own head.

Contract: `docs/contracts/WORLD-BUILDER-WORLDS.md` §4 rule 6 and §4a. The
surface-backed revision cases live beside the surface pipeline's tests; this
file pins what the iOS follower relies on without building a surface:

- the route accepts `view`, and reports the rung §4 would serve for it;
- the payload names its `representation` (so the phone never parses the
  opaque revision for a rung) and says whether a build is `live`;
- `live` is per SESSION, and false for a stage whose process is gone;
- the sparse and dense pages carry the CSP as a meta tag, after
  `<meta charset>` and before `wb-representation`, inside 4096 characters.
"""

import json
import os
import re
import time

from fastapi import FastAPI
from fastapi.testclient import TestClient

from tower.results import world_builder_render as R
from tower.routes import geometry as geometry_routes
from tower.world_builder.records import Session, World
from tower.world_builder.store import WorldStore

CSP_META = ('<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
            'script-src \'unsafe-inline\'; style-src \'unsafe-inline\'">')


def _client(store) -> TestClient:
    app = FastAPI()
    app.include_router(geometry_routes.router)
    app.state.world_root = store.root
    return TestClient(app)


# ---------------------------------------------------------------------------
# the payload
# ---------------------------------------------------------------------------


def test_the_route_names_the_rung_and_whether_a_build_is_live(derived_world):
    store, world_id, session_id = derived_world
    response = _client(store).get(f"/worlds/{world_id}/render/revision",
                                  params={"session_id": session_id})
    assert response.status_code == 200
    body = response.json()
    assert body["representation"] == "sparse"
    assert body["revision"] == f"{session_id}/sparse"
    assert body["live"] is False, "a finished fixture world has nothing building"


def test_the_route_honours_view_diagnostics(derived_world, monkeypatch):
    """The iOS review's M1, at the ROUTE: without `view` reaching the adapter,
    a diagnostics page (always sparse) would be followed against the surface
    revision and refetched on every rebuild."""
    store, world_id, session_id = derived_world
    monkeypatch.setattr(R, "surface_artifact_drawable", lambda *_args: True)
    # A manifest to read the revision from: a surface with none is answered as
    # absent now, never as `surface:None` (review 2, iOS m5).
    manifest = store.world_dir(world_id) / "surface" / session_id / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"built_at": 1.0}), encoding="utf-8")
    client = _client(store)

    product = client.get(f"/worlds/{world_id}/render/revision",
                         params={"session_id": session_id}).json()
    assert product["representation"] == "surface"

    diagnostics = client.get(f"/worlds/{world_id}/render/revision",
                             params={"session_id": session_id, "view": "diagnostics"}).json()
    assert diagnostics["representation"] == "sparse"
    assert diagnostics["revision"] == f"{session_id}/sparse"


def test_an_unmatched_route_and_a_contract_404_are_distinguishable(derived_world):
    """§4a rule 5: the phone stops only on FastAPI's own `Not Found`."""
    store, world_id, _session_id = derived_world
    client = _client(store)
    unmatched = client.get(f"/worlds/{world_id}/render/nope")
    assert unmatched.status_code == 404
    assert unmatched.json() == {"detail": "Not Found"}

    contract = client.get("/worlds/no-such-world/render/revision")
    assert contract.status_code == 404
    assert contract.json()["detail"] != "Not Found"


# ---------------------------------------------------------------------------
# live
# ---------------------------------------------------------------------------


def _alive_lock(monkeypatch):
    # The lock's own pid is excluded by the listing's probe (a Tower cannot be
    # its own builder), so a held lock is simulated at the probe it calls.
    monkeypatch.setattr(WorldStore, "lock_holder",
                        lambda self, world_id: {"pid": 1, "alive": True, "unreadable": False})


def test_a_held_lock_makes_the_session_being_written_live(derived_world, monkeypatch):
    store, world_id, session_id = derived_world
    _alive_lock(monkeypatch)
    # The fixture's session has ended and has no finalization record: a
    # builder holding the world lock is writing some OTHER session.
    assert R.session_build_running(store, world_id, session_id) is False

    session = store.read_session(world_id, session_id)
    store.write_session(Session(session_id=session_id, world_id=world_id,
                                started_at=session.started_at, ended_at=session.ended_at,
                                finalization={"state": "pending"}))
    assert R.session_build_running(store, world_id, session_id) is True, (
        "a session in finalization under a live lock is still being built")

    store.write_session(Session(session_id="s-open", world_id=world_id,
                                started_at=5.0, ended_at=None))
    world = store.read_world(world_id)
    store.write_world(World(world_id=world_id, created_at=world.created_at,
                            updated_at=6.0, session_ids=(session_id, "s-open")))
    assert R.session_build_running(store, world_id, "s-open") is True


def _write_status(store, world_id, session_id, stage, **fields):
    root = store.world_dir(world_id) / stage / session_id
    root.mkdir(parents=True, exist_ok=True)
    (root / "status.json").write_text(json.dumps(fields), encoding="utf-8")


def test_a_running_surface_stage_is_live_after_the_lock_is_gone(derived_world):
    """The final surface runs AFTER the builder releases the world lock."""
    store, world_id, session_id = derived_world
    _write_status(store, world_id, session_id, "surface",
                  state="running", pid=os.getpid(), updated_at=time.time())
    assert R.session_build_running(store, world_id, session_id) is True


def test_a_stage_that_has_published_its_manifest_is_not_live_before_ok(derived_world):
    """Review 3, R5. A build writes its manifest, prunes, and only then `ok`. A
    poll in that gap saw the new revision with `live: true`, so the phone offered
    the finished world instead of swapping it in, and the `live: false` poll
    after it was not new. The manifest being newer than the `running` status
    is the build having published."""
    store, world_id, session_id = derived_world
    for stage in ("surface", "dense"):
        _write_status(store, world_id, session_id, stage,
                      state="running", pid=os.getpid(), updated_at=time.time())
        root = store.world_dir(world_id) / stage / session_id
        status_ns = (root / "status.json").stat().st_mtime_ns
        manifest = root / "manifest.json"
        # an OLDER manifest, from the previous build: still live
        manifest.write_text(json.dumps({"built_at": 1.0}), encoding="utf-8")
        os.utime(manifest, ns=(status_ns - 10**9, status_ns - 10**9))
        assert R.session_build_running(store, world_id, session_id) is True, stage
        # this build's manifest, published after its `running` status: not live
        os.utime(manifest, ns=(status_ns + 10**9, status_ns + 10**9))
        assert R.session_build_running(store, world_id, session_id) is False, stage
        # and the next build's `running` status makes it live again
        _write_status(store, world_id, session_id, stage,
                      state="running", pid=os.getpid(), updated_at=time.time())
        os.utime(root / "status.json", ns=(status_ns + 2 * 10**9, status_ns + 2 * 10**9))
        assert R.session_build_running(store, world_id, session_id) is True, stage
        (root / "status.json").write_text(json.dumps({"state": "ok"}), encoding="utf-8")


def test_the_surface_stage_publishes_its_manifest_before_ok():
    """The probe above reads "manifest newer than running" as published, which
    is only sound while every `running` write precedes the manifest write and
    `ok` follows it. Pinned on the source, since the order is the contract."""
    import inspect

    from tower.world_builder import surface_pipeline

    body = inspect.getsource(surface_pipeline.surfacify)
    publish = body.index("_write_manifest(root")
    assert body.rindex("state=STATE_RUNNING", 0, publish) < publish
    assert body.index("_status(root, state=STATE_OK", publish) > publish


def test_a_running_dense_stage_is_live(derived_world):
    store, world_id, session_id = derived_world
    _write_status(store, world_id, session_id, "dense",
                  state="running", pid=os.getpid(), updated_at=time.time())
    assert R.session_build_running(store, world_id, session_id) is True


def test_a_stage_whose_process_is_gone_is_not_live(derived_world):
    store, world_id, session_id = derived_world
    # A pid no process holds, on both stages, and a finished stage.
    _write_status(store, world_id, session_id, "surface",
                  state="running", pid=2**31 - 7, updated_at=time.time())
    _write_status(store, world_id, session_id, "dense", state="ok", pid=os.getpid())
    assert R.session_build_running(store, world_id, session_id) is False


def test_an_unreadable_status_is_not_live_and_does_not_raise(derived_world):
    store, world_id, session_id = derived_world
    root = store.world_dir(world_id) / "surface" / session_id
    root.mkdir(parents=True, exist_ok=True)
    (root / "status.json").write_text("{not json", encoding="utf-8")
    assert R.session_build_running(store, world_id, session_id) is False


# ---------------------------------------------------------------------------
# the CSP in the page's own head
# ---------------------------------------------------------------------------


def _head_order(html: str) -> None:
    charset = html.find('<meta charset="utf-8">')
    csp = html.find(CSP_META)
    rung = html.find('<meta name="wb-representation"')
    assert 0 <= charset < csp < rung, (charset, csp, rung)
    assert rung + 64 < 4096, "the phone reads the rung from the first 4096 characters"
    assert html.count("http-equiv=\"Content-Security-Policy\"") == 1


def test_the_sparse_page_carries_the_policy_as_a_meta_tag(derived_world):
    store, world_id, session_id = derived_world
    html = R.build_world_render(store, world_id, session_id)
    _head_order(html)
    assert re.search(r'<meta name="wb-revision" content="[^"]+">', html[:4096])


def test_the_dense_template_carries_the_policy_as_a_meta_tag():
    from tower.world_builder.dense_render import viewer_template_path

    _head_order(viewer_template_path().read_text(encoding="utf-8"))


def test_the_meta_policy_is_the_response_header(derived_world):
    store, world_id, session_id = derived_world
    response = _client(store).get(f"/worlds/{world_id}/render",
                                  params={"session_id": session_id})
    header = response.headers["content-security-policy"]
    assert f'content="{header}"' in response.text[:4096]


def test_a_running_dense_stage_whose_pid_was_recycled_is_not_live(derived_world):
    """Review 2, I2. A densify killed mid-run leaves `running`; once Windows
    hands its pid to a later process, a bare pid probe said `live: true` for as
    long as that process lived. This process started after a status written a
    day ago, so it cannot have written it."""
    store, world_id, session_id = derived_world
    _write_status(store, world_id, session_id, "dense",
                  state="running", pid=os.getpid(), updated_at=time.time() - 86400)
    assert R.session_build_running(store, world_id, session_id) is False
