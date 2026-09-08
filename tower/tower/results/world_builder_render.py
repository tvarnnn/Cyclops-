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

import logging

from tower.results.world_builder_geometry import contained_world_id
from tower.world_builder.render import DEFAULT_MAX_POINTS, render_html
from tower.world_builder.store import WorldStore, WorldStoreError

logger = logging.getLogger(__name__)

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


def _clip(value: str, limit: int = 80) -> str:
    """An id as it appears in a 404 detail: bounded, like `_echo_safe` on
    the result channel, because the detail echoes a request-supplied
    string and the phone shows it verbatim."""
    return value if len(value) <= limit else value[:limit] + "..."


def resolve_session(store: WorldStore, world_id: str, session_id: str | None) -> str:
    """The session to draw. An explicit id is taken as given; otherwise
    the newest session of the world that has geometry -- the listing
    orders sessions oldest first by `started_at`, and this follows it.

    `session_id` is checked against the world's own sessions before any
    path is joined from it, so it cannot name a directory outside the
    world. `world_id` is contained by the caller (`build_world_render`).
    """
    try:
        world = store.read_world(world_id)
    except (WorldStoreError, OSError, ValueError, KeyError):
        raise WorldRenderUnavailable(f"no world {_clip(world_id)!r}") from None
    # The directory scan the listing uses, not `world.json`'s list: the
    # engine writes `session.json` before it appends to `world.json`, and a
    # session the listing offers must be one the render can find.
    sessions = set(store.list_session_ids(world_id)) | set(world.session_ids)
    if session_id is not None:
        if session_id not in sessions:
            raise WorldRenderUnavailable(
                f"world {_clip(world_id)!r} has no session {_clip(session_id)!r}")
        if not _has_geometry(store, world_id, session_id):
            raise WorldRenderUnavailable(
                f"session {session_id!r} of world {world_id!r} has no geometry yet")
        return session_id
    candidates = []
    for candidate in sorted(sessions):
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


REPRESENTATION_AUTO = "auto"
REPRESENTATION_SPARSE = "sparse"
REPRESENTATION_DENSE = "dense"


def build_world_render(store: WorldStore, world_id: str, session_id: str | None, *,
                       max_points: int | None = None,
                       representation: str = REPRESENTATION_AUTO) -> str:
    """The viewer page for one session of one world, or
    `WorldRenderUnavailable` naming what is missing.

    `world_id` goes through the geometry adapter's containment guard
    first, for the reason its docstring gives: on Windows a backslash is
    a separator Starlette's route pattern does not exclude, and every
    store path below is joined from this id. An id that escapes the root
    is "no world", and the id is answered under its canonical spelling.
    """
    contained = contained_world_id(store, world_id)
    if contained is None:
        raise WorldRenderUnavailable(f"no world {_clip(world_id)!r}")
    world_id = contained
    chosen = resolve_session(store, world_id, session_id)
    # A session that has a dense reconstruction gets the dense viewer, because
    # that is the whole point of having built one. `representation=sparse`
    # forces the old page, and a session with no dense artifact -- which is
    # every world built before this stage -- falls through to it unchanged.
    if representation != REPRESENTATION_SPARSE:
        # THE IMPORT IS OUTSIDE THE try THAT CATCHES ITS EXCEPTION.
        #
        # `DenseViewerUnavailable` used to be bound by an import inside the
        # same `try` whose first `except` names it. If that import raised --
        # the one class of bug the fallback below exists for, a broken dense
        # module -- Python evaluated the first except clause, hit a NameError
        # on the unbound name, and propagated THAT. Later clauses of the same
        # try are not tried, so the "never lose the sparse page to a dense bug"
        # fallback never ran and the route returned 500.
        try:
            from tower.world_builder.dense import (  # noqa: PLC0415
                POINT_STRIDE_BYTES,
            )
            from tower.world_builder.dense_render import (  # noqa: PLC0415
                MOBILE_BYTE_BUDGET,
                DenseViewerUnavailable,
                build_dense_page,
            )
        except Exception:  # noqa: BLE001 -- a dense module that will not import
            logger.exception(
                "[Tower][WorldBuilder] the dense viewer module did not import "
                "for %s; serving sparse", world_id,
            )
            DenseViewerUnavailable = build_dense_page = None  # noqa: N806

        try:
            if build_dense_page is None:
                raise RuntimeError("dense viewer unavailable")

            # `max_points` is validated by the route and must not then be
            # ignored: the worlds contract calls it "point budget", and a
            # client that asks for fewer points has to get fewer. It was
            # dropped on this path, so `max_points=1` returned 295,000 points
            # and a 6 MB page. Converted to the byte budget this viewer speaks,
            # and only ever downwards -- the phone default stays the default.
            budget = MOBILE_BYTE_BUDGET
            if max_points is not None:
                budget = min(budget, max(1, int(max_points)) * POINT_STRIDE_BYTES)
            return build_dense_page(store, world_id, chosen, budget_bytes=budget)
        except DenseViewerUnavailable as exc:
            if representation == REPRESENTATION_DENSE:
                raise WorldRenderUnavailable(exc.reason) from None
        except Exception:  # noqa: BLE001 -- never lose the sparse page to a dense bug
            logger.exception(
                "[Tower][WorldBuilder] dense viewer failed for %s; serving sparse",
                world_id,
            )

    budget = MOBILE_MAX_POINTS if max_points is None else min(max_points, MAX_POINTS_CEILING)
    try:
        return render_html(store, world_id, chosen, max_points=budget)
    except FileNotFoundError:
        # Raced a `clear_derived` between the existence check and the read.
        # Worded here rather than from the exception: the phone shows the
        # detail verbatim, and an OSError's text carries a filesystem path
        # (contract §3 rule 3: no paths on the wire).
        raise WorldRenderUnavailable(
            f"session {chosen!r} of world {world_id!r} has no geometry yet") from None
