"""The interactive viewer of a saved world, for a phone to open.

Contract: `docs/contracts/WORLD-BUILDER-WORLDS.md` §4 (`GET /worlds/{id}/render`).

The fourth adapter named after this cartridge, and for the same reason
the other three are: it reads World Builder's record shapes, so it is the
one file outside the package that may. It answers a different question
from the geometry producer -- "let a person LOOK at the room" rather than
"give a client the points" -- and answers it with a self-contained HTML
page rather than JSON, which is why it is a separate file and a separate
route rather than a query flag on the manifest.

The page is composed on request from the derived tree, by the same code
`scripts/world_render.py` uses for its `world.html`, so what the phone
shows is what the operator's render shows. Nothing is cached and nothing
is written: a world under construction changes with every build, and a
finished one is cheap to compose (a 438-keyframe walk is well under a
second at the mobile point budget).
"""

from __future__ import annotations

from tower.world_builder.render import DEFAULT_MAX_POINTS, render_html
from tower.world_builder.store import WorldStore, WorldStoreError

# A phone draws every point on a 2-D canvas on every gesture, so the
# budget is lower than the operator's 200k default. Fractional-stride
# sampling spans the whole cloud (contract §3), so the picture is the same
# room, thinner. A client may ask for more, up to the operator's default.
MOBILE_MAX_POINTS = 80_000
MAX_POINTS_CEILING = DEFAULT_MAX_POINTS


class WorldRenderUnavailable(Exception):
    """The route's 404: the world, the session, or its geometry is absent.
    `reason` is what the phone shows; it names what is missing."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _has_geometry(store: WorldStore, world_id: str, session_id: str) -> bool:
    derived = store.derived_dir(world_id) / session_id
    return (derived / "poses.json").exists() and (derived / "points.json").exists()


def resolve_session(store: WorldStore, world_id: str, session_id: str | None) -> str:
    """The session to draw. An explicit id is taken as given; otherwise
    the newest session of the world that has geometry -- the listing
    orders sessions oldest first by `started_at`, and this follows it."""
    try:
        world = store.read_world(world_id)
    except (WorldStoreError, OSError, ValueError, KeyError):
        raise WorldRenderUnavailable(f"no world {world_id!r}") from None
    if session_id is not None:
        if session_id not in world.session_ids:
            raise WorldRenderUnavailable(
                f"world {world_id!r} has no session {session_id!r}")
        if not _has_geometry(store, world_id, session_id):
            raise WorldRenderUnavailable(
                f"session {session_id!r} of world {world_id!r} has no geometry yet")
        return session_id
    candidates = []
    for candidate in world.session_ids:
        if not _has_geometry(store, world_id, candidate):
            continue
        try:
            started = store.read_session(world_id, candidate).started_at
        except (WorldStoreError, OSError, ValueError, KeyError):
            continue
        candidates.append((started, candidate))
    if not candidates:
        raise WorldRenderUnavailable(f"world {world_id!r} has no session with geometry yet")
    candidates.sort()
    return candidates[-1][1]


def build_world_render(store: WorldStore, world_id: str, session_id: str | None, *,
                       max_points: int | None = None) -> str:
    """The viewer page for one session of one world, or
    `WorldRenderUnavailable` naming what is missing."""
    chosen = resolve_session(store, world_id, session_id)
    budget = MOBILE_MAX_POINTS if max_points is None else min(max_points, MAX_POINTS_CEILING)
    try:
        return render_html(store, world_id, chosen, max_points=budget)
    except FileNotFoundError as exc:
        # Raced a `clear_derived` between the existence check and the read.
        raise WorldRenderUnavailable(str(exc)) from None
