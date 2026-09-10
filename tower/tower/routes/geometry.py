"""World Builder geometry, over HTTP.

HTTP and not the WebSocket, for a reason that is in the code rather than
in taste: `tower/routes/ws.py` gives the result sender and the frame path
one shared `asyncio.Lock`. One session's `points.json` is 1.07 MB against
a 3,884-byte status snapshot, so pushing geometry down that socket would
hold the lock and starve `frame_result` -- the exact thing
`CARTRIDGE-RESULTS.md` forbids in Tower responsibility #3.

Both handlers are declared `def` rather than `async def` on purpose.
FastAPI runs a sync endpoint in its threadpool, which keeps the disk read
and the hash off the event loop with no executor of our own.
"""

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

from tower.results.world_builder_geometry import (
    build_manifest,
    build_segment,
    store_from_root,
)
from tower.results.world_builder_library import build_world_listing
from tower.results.world_builder_render import (
    MAX_POINTS_CEILING,
    WorldRenderUnavailable,
    build_world_render,
)

router = APIRouter()


def _store(request: Request):
    root = getattr(request.app.state, "world_root", None)
    if root is None:
        raise HTTPException(status_code=404, detail="no world root is configured")
    return store_from_root(root)


@router.get("/worlds")
def world_listing(request: Request) -> dict:
    """Every saved world and its sessions, so a viewer can open an old one.

    Contract `world_builder.worlds/2026-09-06`. Read-only; a directory
    walk over `world.json` / `session.json`, no geometry. Sync `def` like
    the geometry handlers, and for the same reason.
    """
    return build_world_listing(_store(request))


@router.get("/worlds/{world_id}/geometry/manifest")
def geometry_manifest(world_id: str, session_id: str, request: Request) -> dict:
    manifest = build_manifest(_store(request), world_id, session_id)
    if manifest is None:
        # 404 now means ABSENT only. Geometry that is real but behind the
        # newest keyframes is served with `current: false` instead of
        # hidden -- during a walk the digest moves with every keyframe, so
        # refusing it meant the gallery stayed empty for the whole capture.
        raise HTTPException(status_code=404, detail="no geometry for this session")
    return manifest


@router.get("/worlds/{world_id}/geometry/segment/{segment_index}")
def geometry_segment(
    world_id: str, segment_index: int, session_id: str, request: Request,
    max_points: int | None = Query(default=None, ge=1),
) -> dict:
    chunk = build_segment(
        _store(request), world_id, session_id, segment_index, max_points=max_points
    )
    if chunk is None:
        raise HTTPException(status_code=404, detail="no such segment")
    return chunk


@router.get("/worlds/{world_id}/render", response_class=HTMLResponse)
def world_render(
    world_id: str, request: Request,
    session_id: str | None = Query(default=None),
    max_points: int | None = Query(default=None, ge=1, le=MAX_POINTS_CEILING),
    # DECLARED, not left to the page to read off its own URL.
    #
    # The viewer can open in either the world view or the diagnostic one,
    # and it used to pick between them from `location.search`. iOS -- the
    # only client this page has -- loads it with
    # `loadHTMLString(_:baseURL: nil)`: no origin, no URL, so
    # `location.search` is always empty, and its navigation policy cancels
    # the page's own links so an in-page href could not reach the other
    # view either. The affordance existed and did nothing on the device it
    # was built for. Declaring it here is what makes it real.
    #
    # Not validated to an enum on purpose: an unrecognised value opens the
    # world view. This is a display mode on an unauthenticated route, and
    # the safe reading of a value nobody understands is "show the wearer
    # their world", not a 422.
    #
    # `None` rather than a named default, because naming it would mean
    # importing it, and `test_shared_code_does_not_import_a_cartridge`
    # refuses a route that imports `tower.world_builder` -- correctly: the
    # web process knows a world builder only through the adapter below.
    # The adapter owns what "no view asked for" means.
    view: str | None = Query(default=None),
) -> HTMLResponse:
    """The interactive viewer of one saved world, as a self-contained page.

    Contract `WORLD-BUILDER-WORLDS.md` §4. Composed on request from the
    derived tree by the same code `scripts/world_render.py` writes
    `world.html` with. Sync `def` for the reason the geometry handlers
    are: the read and the composition stay off the event loop.

    404 means ABSENT -- no root, no such world, no such session, or no
    geometry built for it yet -- and `detail` says which, because the
    phone shows it. Geometry behind the newest keyframes is served with
    its own caption saying so, never hidden.
    """
    try:
        html = build_world_render(
            _store(request), world_id, session_id, max_points=max_points,
            view=view,
        )
    except WorldRenderUnavailable as exc:
        raise HTTPException(status_code=404, detail=exc.reason) from None
    # The phone fetches with its cache bypassed, and this header says the
    # same thing from this side: a world under construction changes with
    # every build, and there are no validators to revalidate against.
    #
    # The policy states what the contract promises -- a page that loads
    # nothing from anywhere -- so a browser enforces it too: its own inline
    # script and style, and no other resource of any kind.
    return HTMLResponse(html, headers={
        "Cache-Control": "no-store",
        "Content-Security-Policy": (
            "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'"
        ),
    })
