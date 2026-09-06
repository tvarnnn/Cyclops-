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

from tower.results.world_builder_geometry import (
    build_manifest,
    build_segment,
    store_from_root,
)
from tower.results.world_builder_library import build_world_listing

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
