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
from fastapi.responses import HTMLResponse, JSONResponse, Response

from tower.results.envelope import json_safe
from tower.results.world_builder_appearance import (
    NO_STORE_HEADERS,
    AppearanceNotServed,
    appearance_file,
    appearance_manifest,
    encode_body,
)
from tower.results.world_builder_geometry import (
    build_manifest,
    build_segment,
    store_from_root,
)
from tower.results.world_builder_library import build_world_listing
from tower.results.world_builder_render import (
    MAX_POINTS_CEILING,
    WorldRenderUnavailable,
    build_render_revision,
    build_world_render,
    render_content_security_policy,
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

    Contract `world_builder.worlds/2026-09-10`. Read-only; a directory
    walk over `world.json` / `session.json`, no geometry. Sync `def` like
    the geometry handlers, and for the same reason.

    `json_safe` for the reason `routes/ws.py` applies it to every send:
    Starlette's `JSONResponse` serialises with `allow_nan=False`, so ONE
    `NaN` anywhere in the listing is a 500 that loses every world.
    MEASURED (fastapi 0.141.1, pydantic 2.13.5): that 500 is reached
    only by a handler with no return annotation. The `-> dict` above
    sends the payload through pydantic's serialiser first, and its
    default (`ser_json_inf_nan="null"`) already turns a non-finite float
    into `null`. So today the wrap changes nothing on the wire; it is
    here so that the guarantee is this route's own rather than a side
    effect of an annotation somebody could remove, or of a serialiser
    default somebody could change. The producer refuses a row whose own
    timestamps are not numbers, but it passes `finalization` through as
    the builder wrote it, and a clock can write `NaN` there. `None` is
    the contract's word for "not established", which is what a
    non-finite number honestly is.
    """
    return json_safe(build_world_listing(_store(request)))


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


def _appearance_store(request: Request):
    root = getattr(request.app.state, "world_root", None)
    if root is None:
        raise HTTPException(status_code=404, detail="no world root is configured",
                            headers=NO_STORE_HEADERS)
    return store_from_root(root)


def _appearance_headers(label: str) -> dict:
    return {**NO_STORE_HEADERS, "X-World-Redaction": label}


def _appearance_response(request: Request, data: bytes, media_type: str, label: str) -> Response:
    """A 200 of appearance bytes, gzip/deflate when the client accepts it.

    The privacy headers are the same either way (`no-store`, `nosniff`, no
    validators); only `Content-Encoding` and `Vary` are added. 404s are not
    compressed: they are a sentence.
    """
    body, encoding = encode_body(data, request.headers.get("accept-encoding"))
    return Response(content=body, media_type=media_type,
                    headers={**_appearance_headers(label), **encoding})


@router.get("/worlds/{world_id}/appearance/{session_id}/manifest")
def appearance_manifest_route(world_id: str, session_id: str, request: Request) -> Response:
    """The appearance artifact's manifest (`WORLD-BUILDER-APPEARANCE.md` §9).

    Imagery metadata of a private space: `no-store`, no validators, and the
    session's redaction label re-checked on every request -- a relabelled
    session answers 404 rather than its old textures.
    """
    try:
        payload, label = appearance_manifest(_appearance_store(request), world_id, session_id)
    except AppearanceNotServed as exc:
        raise HTTPException(status_code=404, detail=exc.reason,
                            headers=NO_STORE_HEADERS) from None
    # Rendered exactly as JSONResponse would, then encoded like the bytes routes:
    # the manifest is ~0.4 MB of JSON for a 374-keyframe walk.
    data = JSONResponse(json_safe(payload)).body
    return _appearance_response(request, data, "application/json", label)


def _appearance_bytes(request: Request, world_id: str, session_id: str, kind: str,
                      digest: str) -> Response:
    try:
        data, label = appearance_file(_appearance_store(request), world_id, session_id,
                                      kind, digest)
    except AppearanceNotServed as exc:
        raise HTTPException(status_code=404, detail=exc.reason,
                            headers=NO_STORE_HEADERS) from None
    return _appearance_response(request, data, "application/octet-stream", label)


@router.get("/worlds/{world_id}/appearance/{session_id}/chunk/{digest}")
def appearance_chunk_route(world_id: str, session_id: str, digest: str,
                           request: Request) -> Response:
    """One keyframe bundle, by the content digest the manifest names."""
    return _appearance_bytes(request, world_id, session_id, "chunk", digest)


@router.get("/worlds/{world_id}/appearance/{session_id}/proxy/{digest}")
def appearance_proxy_route(world_id: str, session_id: str, digest: str,
                           request: Request) -> Response:
    """The proxy mesh the appearance was built against (`WBSURF01`)."""
    return _appearance_bytes(request, world_id, session_id, "proxy", digest)


@router.get("/worlds/{world_id}/render/revision")
def world_render_revision(
    world_id: str, request: Request,
    session_id: str | None = Query(default=None),
    view: str | None = Query(default=None),
    viewer: str | None = Query(default=None),
) -> JSONResponse:
    """Which picture `GET /worlds/{id}/render` would serve now, as a revision.

    Contract `WORLD-BUILDER-WORLDS.md` §4a. A few hundred bytes, so the phone
    can keep a picture open during a walk and learn that a better one has been
    built without re-downloading megabytes to find out. The revision it is
    compared with is stamped into the page it already has.
    """
    try:
        payload = build_render_revision(_store(request), world_id, session_id, view=view,
                                        viewer=viewer)
    except WorldRenderUnavailable as exc:
        raise HTTPException(status_code=404, detail=exc.reason) from None
    return JSONResponse(json_safe(payload), headers={"Cache-Control": "no-store"})


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
    # Which reconstruction to serve. "auto" is the product default and means
    # the best one this session actually has -- appearance (only for a client
    # declaring `viewer=appearance-1`), surface, then dense points, then
    # sparse. The named values exist so a developer, a test, or the
    # diagnostics screen can pin one and compare; naming a representation
    # the session does not have falls back rather than failing, because the
    # caller asking for a better picture should never get no picture.
    representation: str = Query(
        default="auto", pattern="^(auto|sparse|dense|surface|appearance)$"),
    # Where the APPEARANCE page fetches its imagery from (WORLD-BUILDER-WORLDS.md
    # §4). `app`, the default and the phone's only mode: the app's private
    # `glasses-world:` scheme, whose native handler whitelists this world's
    # routes. `tower`: a desktop debug mode, only when named, fetching from this
    # origin. Every other page fetches nothing and ignores it.
    transport: str = Query(default="app", pattern="^(app|tower)$"),
    # What the client can draw (WORLD-BUILDER-WORLDS.md §4). `auto` offers the
    # appearance page only to a client declaring `appearance-1`: an iOS build
    # older than that page has no scheme handler to fetch its imagery through,
    # and was served a page that could fetch nothing. Comma-separated tokens;
    # unknown ones are ignored, never a 422.
    viewer: str | None = Query(default=None),
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
            view=view, representation=representation, transport=transport,
            viewer=viewer,
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
    #
    # The appearance page is the one exception: it fetches its imagery, and its
    # policy allows exactly that transport (`connect-src glasses-world:`, or
    # `'self'` in the named desktop debug mode) and nothing else.
    return HTMLResponse(html, headers={
        "Cache-Control": "no-store",
        "Content-Security-Policy": render_content_security_policy(html, transport),
    })
