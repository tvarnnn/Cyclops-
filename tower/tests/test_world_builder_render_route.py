"""`GET /worlds/{id}/render`: the interactive viewer a phone opens.

The composition itself is pinned by `test_world_render.py` (the Sim3 is
applied as the store defines it, unregistered segments stay apart). This
file pins the ROUTE: what a client gets for a world that exists, and what
it gets -- and is told -- for one that does not.
"""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tower.results.world_builder_render import (
    MAX_POINTS_CEILING,
    MOBILE_MAX_POINTS,
    WorldRenderUnavailable,
    build_world_render,
    resolve_session,
)
from tower.routes import geometry as geometry_routes
from tower.world_builder.records import Session, World
from tower.world_builder.render import CAPTION, CAPTION_BEHIND


def _client(store) -> TestClient:
    app = FastAPI()
    app.include_router(geometry_routes.router)
    app.state.world_root = store.root
    return TestClient(app)


def _frames_payload(html: str) -> list:
    start = html.index("const FRAMES = ") + len("const FRAMES = ")
    end = html.index(";\n", start)
    return json.loads(html[start:end].replace("<\\/", "</"))


def test_the_route_serves_a_self_contained_page(derived_world):
    store, world_id, session_id = derived_world
    response = _client(store).get(f"/worlds/{world_id}/render",
                                  params={"session_id": session_id})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.headers["cache-control"] == "no-store"
    html = response.text
    assert html.startswith("<!doctype html>")
    # Nothing fetched from anywhere: the phone's web view refuses outbound
    # navigation, and the page must not need any.
    assert "<script src=" not in html and "<link " not in html
    assert 'name="viewport"' in html
    assert "touchstart" in html and "touchmove" in html
    # The picture says what it is, every time.
    assert CAPTION in html
    assert CAPTION_BEHIND not in html  # the fixture's tree is current


def test_the_page_draws_the_fixture_segments(derived_world):
    store, world_id, session_id = derived_world
    html = build_world_render(store, world_id, session_id)
    frames = _frames_payload(html)
    # No placements in the fixture: each segment with points or cameras is
    # its own frame, never composed.
    assert frames and all(f["shared"] is False for f in frames)
    seg0 = next(f for f in frames if "segment 0" in f["name"])
    assert [s["index"] for s in seg0["segments"]] == [0]
    assert len(seg0["segments"][0]["xyz"]) == 2 * 3
    assert len(seg0["cameras"]) == 2  # the anchor and the solved pose
    seg1 = next(f for f in frames if "segment 1" in f["name"])
    assert seg1["segments"] == [] and len(seg1["cameras"]) == 1  # refused pose not drawn


def test_a_stale_tree_is_served_and_says_so(derived_world):
    from tests.conftest import _keyframe

    store, world_id, session_id = derived_world
    # One more accepted keyframe moves the journal digest past the build.
    store.append_keyframe(world_id, _keyframe(session_id, 5, 1))
    html = build_world_render(store, world_id, session_id)
    assert CAPTION_BEHIND in html


def test_the_session_defaults_to_the_newest_with_geometry(derived_world):
    store, world_id, session_id = derived_world
    # A newer session with no derived tree must not be chosen over the one
    # that has geometry; a newer one WITH geometry must be.
    store.write_session(Session(session_id="s1", world_id=world_id,
                                started_at=10.0, ended_at=11.0))
    world = store.read_world(world_id)
    store.write_world(World(world_id=world_id, created_at=world.created_at,
                            updated_at=12.0, session_ids=(session_id, "s1")))
    assert resolve_session(store, world_id, None) == session_id

    # AN EMPTY TREE IS NOT GEOMETRY, and this used to write one and assert
    # it won. `engine.build` calls `write_derived` unconditionally, so a
    # walk that solved nothing leaves both files with empty arrays -- 14
    # bytes of points.json, and eleven sessions on the real 163-world root
    # look exactly like this. Choosing that over the older walk that has a
    # reconstruction opens the world to a blank page.
    derived = store.derived_dir(world_id) / "s1"
    derived.mkdir(parents=True)
    (derived / "poses.json").write_text(json.dumps({"poses": []}))
    (derived / "points.json").write_text(json.dumps({"points": []}))
    assert resolve_session(store, world_id, None) == session_id, (
        "a newer walk that reconstructed nothing outranked one that did"
    )

    # ...and a newer one that DID build must win, which is what the
    # comment above this test always claimed to check.
    (derived / "poses.json").write_text(json.dumps(
        {"poses": [{"keyframe_id": "k", "segment_index": 0,
                    "status": "solved", "degeneracy": "",
                    "rotation": [1.0, 0.0, 0.0, 0.0],
                    "translation": [0.0, 0.0, 0.0]}]}
    ))
    (derived / "points.json").write_text(json.dumps(
        {"points": [{"segment_index": 0, "xyz": [0.0, 0.0, 1.0]}]}
    ))
    assert resolve_session(store, world_id, None) == "s1"
    assert resolve_session(store, world_id, session_id) == session_id


def test_absence_is_a_404_that_names_what_is_missing(derived_world):
    store, world_id, session_id = derived_world
    client = _client(store)

    response = client.get("/worlds/nope/render")
    assert response.status_code == 404
    assert "no world" in response.json()["detail"]

    response = client.get(f"/worlds/{world_id}/render", params={"session_id": "nope"})
    assert response.status_code == 404
    assert "no session" in response.json()["detail"]

    store.write_session(Session(session_id="s1", world_id=world_id,
                                started_at=10.0, ended_at=None))
    store.write_world(World(world_id=world_id, created_at=1.0, updated_at=12.0,
                            session_ids=(session_id, "s1")))
    response = client.get(f"/worlds/{world_id}/render", params={"session_id": "s1"})
    assert response.status_code == 404
    assert "no geometry yet" in response.json()["detail"]

    store.clear_derived(world_id)
    response = client.get(f"/worlds/{world_id}/render")
    assert response.status_code == 404
    assert "no session with geometry" in response.json()["detail"]

    app = FastAPI()
    app.include_router(geometry_routes.router)
    assert TestClient(app).get(f"/worlds/{world_id}/render").status_code == 404


def test_the_point_budget_is_bounded(derived_world):
    store, world_id, session_id = derived_world
    client = _client(store)
    assert client.get(f"/worlds/{world_id}/render", params={"max_points": 0}).status_code == 422
    assert client.get(f"/worlds/{world_id}/render",
                      params={"max_points": MAX_POINTS_CEILING + 1}).status_code == 422
    assert client.get(f"/worlds/{world_id}/render",
                      params={"max_points": 1}).status_code == 200
    assert 0 < MOBILE_MAX_POINTS <= MAX_POINTS_CEILING


def test_a_world_id_cannot_close_the_script_block(tmp_path):
    """The id is written into the page. A hostile one must stay text."""
    from tower.world_builder.render import canvas_html

    html = canvas_html([], [], [], 'x</script><script>alert(1)</script>"')
    assert "</script><script>alert" not in html
    assert "&lt;/script&gt;" in html


def test_the_payload_carries_no_angle_bracket(derived_world):
    """`<` in the JSON is what could end or escape the script block; it is
    written as \\u003c, which the browser's JSON parser reads back."""
    store, world_id, session_id = derived_world
    html = build_world_render(store, world_id, session_id)
    start = html.index("const FRAMES = ") + len("const FRAMES = ")
    end = html.index(";\n", start)
    assert "<" not in html[start:end]


def test_the_response_forbids_every_external_resource(derived_world):
    store, world_id, session_id = derived_world
    response = _client(store).get(f"/worlds/{world_id}/render")
    csp = response.headers["content-security-policy"]
    assert "default-src 'none'" in csp and "script-src 'unsafe-inline'" in csp


def test_a_world_id_that_escapes_the_root_is_no_world(derived_world):
    """The geometry routes' containment guard applies here too: on Windows a
    backslash is a separator the route pattern does not exclude, and every
    store path is joined from this id. Checked at the adapter, where the
    id arrives however the router spelled it, and over HTTP for the
    backslash spelling the router lets through."""
    store, world_id, session_id = derived_world
    for spelling in ("..\\..\\elsewhere\\worlds\\victim", "../../elsewhere/worlds/victim",
                     "..", "."):
        with pytest.raises(WorldRenderUnavailable) as excinfo:
            build_world_render(store, spelling, None)
        assert "no world" in excinfo.value.reason, spelling
    response = _client(store).get("/worlds/..%5C..%5Celsewhere%5Cworlds%5Cvictim/render")
    assert response.status_code == 404
    # A non-canonical spelling that stays inside the root names the real
    # world and is served as it (POSIX resolves the `..`; the geometry
    # transport test covers the Windows spelling).
    assert build_world_render(store, f"junk/../{world_id}", session_id).startswith("<!doctype")


def test_a_detail_never_carries_a_filesystem_path(derived_world):
    store, world_id, session_id = derived_world
    # Geometry gone between the existence check and the read.
    import tower.results.world_builder_render as adapter

    original = adapter.render_html

    def vanish(*args, **kwargs):
        raise FileNotFoundError(str(store.root / "worlds" / world_id / "derived" / "points.json"))

    adapter.render_html = vanish
    try:
        with pytest.raises(WorldRenderUnavailable) as excinfo:
            build_world_render(store, world_id, session_id)
    finally:
        adapter.render_html = original
    assert str(store.root) not in excinfo.value.reason
    assert "no geometry yet" in excinfo.value.reason


def test_the_adapter_raises_its_own_error_for_a_missing_world(tmp_path):
    from tower.world_builder.store import WorldStore

    with pytest.raises(WorldRenderUnavailable):
        build_world_render(WorldStore(tmp_path), "w", None)


# -- the diagnostics view, which only the route can deliver ----------------
#
# The viewer carries both views on one payload and can toggle between them
# in a browser. The phone cannot use that: iOS loads this page with
# `loadHTMLString(_:baseURL: nil)`, so there is no URL for the page to read
# a mode from, and the web view's navigation policy cancels the page's own
# links. So `?view=` has to be answered HERE, when the page is composed, or
# the wearer has no route to the sparse solver output at all -- which the
# product requirement keeps as reachable diagnostics, not as a deleted one.


def test_the_route_serves_the_diagnostic_view_on_request(derived_world):
    store, world_id, session_id = derived_world
    response = _client(store).get(
        f"/worlds/{world_id}/render",
        params={"session_id": session_id, "view": "diagnostics"},
    )
    assert response.status_code == 200
    assert "let mode = 'diag';" in response.text


def test_the_route_serves_the_world_view_by_default(derived_world):
    store, world_id, session_id = derived_world
    response = _client(store).get(f"/worlds/{world_id}/render",
                                  params={"session_id": session_id})
    assert response.status_code == 200
    assert "let mode = 'world';" in response.text


@pytest.mark.parametrize("view", ["", "world", "PRODUCT", "nonsense", "../x", "<script>"])
def test_an_unrecognised_view_still_serves_the_world(derived_world, view):
    """A display mode on an unauthenticated route.

    A 422 here would answer "show me my world" with a validation error, so
    anything unrecognised opens the world view instead.
    """
    store, world_id, session_id = derived_world
    response = _client(store).get(
        f"/worlds/{world_id}/render",
        params={"session_id": session_id, "view": view},
    )
    assert response.status_code == 200
    assert "let mode = 'world';" in response.text


def test_the_view_parameter_cannot_inject_into_the_page(derived_world):
    """The mode is substituted into a <script> block."""
    store, world_id, session_id = derived_world
    response = _client(store).get(
        f"/worlds/{world_id}/render",
        params={"session_id": session_id, "view": "'; alert(1); //"},
    )
    assert response.status_code == 200
    assert "alert(1)" not in response.text
