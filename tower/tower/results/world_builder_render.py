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

import json
import logging
import re
from html import escape as html_escape

from tower.results.world_builder_geometry import contained_world_id
from tower.world_builder.render import (
    DEFAULT_MAX_POINTS,
    VIEW_DIAGNOSTICS,
    VIEW_PRODUCT,
    render_html,
)
from tower.results.world_builder_library import _sortable
from tower.world_builder.store import (
    WorldStore,
    WorldStoreError,
    built_from_an_inactive_keyframe_set,
    dense_artifact_drawable,
    session_has_drawable_geometry,
    surface_artifact_drawable,
)

logger = logging.getLogger(__name__)

# A phone draws every point on a 2-D canvas on every gesture, so the
# budget is lower than the operator's 200k default. Fractional-stride
# sampling spans the whole cloud (contract §3), so the picture is the same
# room, thinner. A client may ask for more, up to the operator's default.
#
# 80,000 WAS THE NUMBER UNTIL IT WAS MEASURED. The reasoning above was
# right and the value was a guess. Timing the page's own `draw()` -- the
# function an orbit drag calls once per frame -- on the recovered field
# world in desktop Chrome, 1920x819, by cloning its segments to reach each
# count:
#
#      19,329 pts    6.2 ms   161 fps
#      38,658 pts   13.6 ms    74 fps
#      77,316 pts   35.3 ms    28 fps      <- roughly the old budget
#     115,974 pts   65.0 ms    15 fps
#
# Superlinear, and 28 fps is what a FAST DESKTOP manages at the budget the
# phone was being handed. A phone is slower than that at canvas fill by
# some factor this host cannot measure -- so the old value was not a
# margin, it was the cliff.
#
# 40,000 sits at 13.6 ms / 74 fps here, which is a 2.6x margin on the
# measured cliff and still twice the 19,866 points the 2026-09-09 walk
# produced in total. A 20-30 minute walk is the case this protects: at
# ~3.2 keyframes/sec it reaches several times this and would otherwise be
# handed the whole cloud up to the old ceiling.
#
# UNVERIFIED: the desktop-to-phone factor. The next physical test is the
# first chance to measure it, and this number should be revisited with
# that measurement rather than argued about.
MOBILE_MAX_POINTS = 40_000
MAX_POINTS_CEILING = DEFAULT_MAX_POINTS


class WorldRenderUnavailable(Exception):
    """The route's 404: the world, the session, or its geometry is absent.
    `reason` is what the phone shows; it names what is missing."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _has_geometry(store: WorldStore, world_id: str, session_id: str) -> bool:
    """Whether drawing this session would put anything on the page.

    **Existence of the files was the wrong question here specifically.**
    `resolve_session` uses this to CHOOSE which session to draw, newest
    first -- so on a world walked twice where the second walk solved
    nothing, an empty `points.json` (14 bytes, and `engine.build` writes
    one unconditionally) outranked the older walk that has geometry, and
    the wearer got a blank page for a world with a reconstruction in it.
    Found by `test_the_three_surfaces_ask_the_same_question_of_the_same_files`
    the moment the listing's copy was corrected.
    """
    return session_has_drawable_geometry(store, world_id, session_id)


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
    # `world.session_ids` COMES OFF DISK UNCOERCED, and this line both
    # hashes it and (below) sorts it. A `world.json` carrying
    # `session_ids: [{"a": 1}]` raises `TypeError: unhashable` right here;
    # one carrying `[123, "s0"]` hashes fine and then raises in the sort
    # three lines down. Either is an HTTP **500 on the route the phone
    # opens a saved world with**.
    #
    # A reviewer found both immediately after the round that hardened
    # `candidates.sort` on line 141 and walked past line 127 -- in a round
    # whose own title was about fixing things one line above the line it
    # fixed.
    #
    # Ids are strings by contract; anything else is a corrupt record and
    # is dropped rather than allowed to take the whole render with it. The
    # directory scan is authoritative anyway, so a dropped `world.json`
    # entry costs nothing that is actually on disk.
    listed = {sid for sid in world.session_ids if isinstance(sid, str)}
    sessions = set(store.list_session_ids(world_id)) | listed
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
    # `_sortable`, for the reason `build_world_listing` gives at length:
    # `started_at` comes off disk uncoerced, and a string beside a float
    # raises `TypeError` out of a route with no handler. Here that is a
    # 500 on the render page instead of the world.
    # `pair[1]` is a session id and is now guaranteed to be a string --
    # the union above drops anything else -- so the tiebreak cannot raise
    # on a rank tie either.
    candidates.sort(key=lambda pair: (_sortable(pair[0]), pair[1]))
    return candidates[-1][1]


class _ViewerModuleMissing(Exception):
    """Stands in for a viewer's own "unavailable" type when its module did
    not import at all.

    The rungs below deliberately put the import OUTSIDE the `try` that
    catches its exception, because binding the exception name inside that
    `try` meant a broken viewer module left the name unbound and the
    `except` clause raised `NameError` instead of falling back. Binding it
    to `None` instead trades that for `TypeError: catching classes that do
    not inherit from BaseException` -- the same failure with a different
    word. It has to be bound to a real exception class, and one nothing
    raises, so the clause is legal and never matches.
    """


REPRESENTATION_AUTO = "auto"
REPRESENTATION_APPEARANCE = "appearance"
REPRESENTATION_SPARSE = "sparse"
REPRESENTATION_DENSE = "dense"
REPRESENTATION_SURFACE = "surface"

# The fallback ladder, best first. `auto` walks it and serves the first rung
# this session actually has; a named representation starts the walk at its own
# rung so "give me dense" never silently serves something better or worse
# without saying so. The rung that is reached is reported in the page and in
# the worlds listing, so "what am I looking at" is never a guess.
#
# `appearance` (2026-09-17) sits above the surface: the wearer's own redacted
# keyframes blended over the surface they were prepared against. It is served
# whenever the appearance routes would serve (the label still matches, the
# world is not purged, the manifest reads) -- NOT only when the artifact is
# `current`. During a walk every new surface makes the appearance "built on an
# earlier surface" for the minute its rebuild takes; demoting the rung for that
# minute would swap the page down and back up on every solve, resetting the
# wearer's camera twice. The page says it is behind instead
# (WORLD-BUILDER-APPEARANCE.md §8: currency is reported, never enforced).
REPRESENTATION_LADDER = (REPRESENTATION_APPEARANCE, REPRESENTATION_SURFACE,
                         REPRESENTATION_DENSE, REPRESENTATION_SPARSE)

# What a client says it can draw, `WORLD-BUILDER-WORLDS.md` §4 (`viewer`).
#
# `auto` offers the appearance rung ONLY to a client that declares it. The
# appearance page is not self-contained: it fetches its imagery through the
# app's `glasses-world:` scheme handler, which an iOS build older than the rung
# does not have. Such a build loads every page with `loadHTMLString`, sends no
# `viewer`, and before this gate was served an appearance page that could fetch
# nothing -- a broken picture where the surface used to be. Without the
# declaration `auto` starts at the surface, exactly the page that build got
# before the rung existed.
#
# A query parameter rather than a header: the page is fetched natively, but the
# page's own revision poll goes through the scheme handler, and a parameter in
# the proxied URL is visible in the handler's whitelist and its tests, where a
# header would be one more thing to forget to copy. Comma-separated tokens, so
# a later client can declare more than one; unknown tokens are ignored, never a
# 422 (a newer app talking to an older Tower still gets a picture).
VIEWER_APPEARANCE = "appearance-1"


def viewer_capabilities(viewer: str | None) -> frozenset:
    """The capability tokens a `viewer` value declares (empty for `None`)."""
    if not viewer:
        return frozenset()
    return frozenset(token.strip() for token in str(viewer).split(",") if token.strip())


def viewer_draws_appearance(viewer: str | None) -> bool:
    """Whether the client can load the appearance page (§4 `viewer`)."""
    return VIEWER_APPEARANCE in viewer_capabilities(viewer)


# Transports of the appearance page (`appearance_render.TRANSPORTS`), named
# here so the route has a default without importing the cartridge.
TRANSPORT_APP = "app"
TRANSPORT_TOWER = "tower"


_REPRESENTATION_META = re.compile(
    r'<meta name="wb-representation" content="([a-z]+)">')


def render_revision(store: WorldStore, world_id: str, session_id: str,
                    rung: str) -> str | None:
    """An opaque string that changes exactly when the page for `rung` would.

    The phone keeps a saved-world picture open while the Tower is still
    building it -- during a walk the live surface is rebuilt each time a global
    solve lands -- and asks this to learn whether a better picture exists
    without downloading a multi-megabyte page to find out.

    The sparse rung's revision is a CONSTANT on purpose. The derived tree is
    rewritten every few keyframes, and a picture that reloaded itself that
    often would be unusable to look at; what the wearer is waiting for is the
    step UP the ladder, which does change the revision because the rung is in
    it.

    `None` for a surface whose manifest cannot be read now, never
    `"surface:None"`. A bare read racing the `os.replace` of a landing build
    fails on Windows, and the phone took that string for a new same-rung
    revision: one needless page download, and when the page's own stamp hit
    the same race, a swap onto an identical mesh and a second swap back
    (review 2, iOS m5). The read now survives a replace; what still fails is
    reported as no surface revision, and the caller answers the next rung.
    """
    if rung == REPRESENTATION_APPEARANCE:
        # The page PROGRAM's revision, and the appearance EPOCH. Appearance
        # builds are followed by the page itself and are deliberately not in
        # it (§4a `appearance`) -- but a build that an open page must not
        # overwrite in place (a relabel, a purge and rebuild, a re-redaction
        # switch) starts a new epoch, and that does move the page revision. The
        # page revision used to be a constant, so a page that had dropped its
        # textures was stamped exactly like the rebuilt one and the app never
        # replaced it (review 1, B1).
        appearance = _appearance_revision(store, world_id, session_id)
        if appearance.get("revision") is None:
            return None
        try:
            from tower.world_builder.appearance_render import PAGE_REVISION  # noqa: PLC0415
        except Exception:  # noqa: BLE001 -- no page module, no rung
            return None
        epoch = appearance.get("epoch")
        return f"{PAGE_REVISION}@{epoch}" if epoch else PAGE_REVISION
    if rung == REPRESENTATION_SURFACE:
        from tower.world_builder.store import _read_json_past_a_replace  # noqa: PLC0415

        path = store.world_dir(world_id) / "surface" / session_id / "manifest.json"
        if not path.exists():
            # Every page asks for every rung's revision; a world with no
            # surface must not pay the replace-retry backoff for it.
            return None
        try:
            manifest = _read_json_past_a_replace(path)
        except ValueError:
            manifest = None
        built = manifest.get("built_at") if isinstance(manifest, dict) else None
        return None if built is None else f"surface:{built}"
    if rung == REPRESENTATION_DENSE:
        path = store.world_dir(world_id) / "dense" / session_id / "manifest.json"
        try:
            stamp = path.stat().st_mtime_ns
        except OSError:
            stamp = None
        return f"dense:{stamp}"
    return REPRESENTATION_SPARSE


def _stamp_revision(store: WorldStore, world_id: str, session_id: str,
                    html: str, revisions: dict | None = None) -> str:
    """Write the page's own revision into its head, beside its rung.

    So the phone knows which revision it is showing from the page itself, with
    no second request -- and so no race in which a build lands between
    fetching a page and asking what revision it was.
    """
    match = _REPRESENTATION_META.search(html, 0, 4096)
    if match is None:
        return html
    rung = match.group(1)
    # `revisions` is read BEFORE the page is composed. Read after, a build
    # landing between the two stamped an OLD page with the NEW revision, and
    # the phone -- told it already had the latest -- never fetched the final
    # surface. Read before, the same race stamps a NEW page with an OLD
    # revision, which costs the phone one extra fetch and is always safe.
    revision = (revisions or {}).get(rung)
    if revision is None:
        own = render_revision(store, world_id, session_id, rung)
        if own is None:
            # No stamp rather than a false one: the phone then records the
            # revision it polled, which is what it did before pages carried one.
            return html
        revision = f"{session_id}/{own}"
    tag = f'<meta name="wb-revision" content="{html_escape(revision, quote=True)}">'
    return html[:match.end()] + tag + html[match.end():]


def build_render_revision(store: WorldStore, world_id: str,
                          session_id: str | None, view: str | None = None,
                          viewer: str | None = None) -> dict:
    """What `GET /worlds/{id}/render` would serve now, as a revision.

    Walks the same ladder in the same order, but by the artifact checks the
    Saved Worlds listing uses rather than by composing the page. The two can
    disagree only when an artifact passes its header check and then fails to
    parse, and that disagreement costs one extra page fetch, not a loop: the
    phone records the revision it was told before comparing pages.

    `viewer` gates the appearance rung exactly as it gates `auto` in
    `build_world_render`: a client that does not declare it is told the rung
    it would be served, so an old app's follower never chases a page it cannot
    draw. `appearance` is reported either way (it is additive data).
    """
    contained = contained_world_id(store, world_id)
    if contained is None:
        raise WorldRenderUnavailable(f"no world {_clip(world_id)!r}")
    world_id = contained
    chosen = resolve_session(store, world_id, session_id)
    revision = None
    appearance = _appearance_revision(store, world_id, chosen)
    if view == VIEW_DIAGNOSTICS:
        rung = REPRESENTATION_SPARSE
    elif (viewer_draws_appearance(viewer)
          and appearance.get("revision") is not None
          and (revision := render_revision(
              store, world_id, chosen, REPRESENTATION_APPEARANCE)) is not None):
        rung = REPRESENTATION_APPEARANCE
    elif (surface_artifact_drawable(store, world_id, chosen)
          and (revision := render_revision(
              store, world_id, chosen, REPRESENTATION_SURFACE)) is not None):
        # A surface whose manifest cannot be read right now is answered as
        # absent -- the next rung -- never as `surface:None`.
        rung = REPRESENTATION_SURFACE
    elif dense_artifact_drawable(store, world_id, chosen):
        rung = REPRESENTATION_DENSE
    else:
        rung = REPRESENTATION_SPARSE
    if rung not in (REPRESENTATION_SURFACE, REPRESENTATION_APPEARANCE):
        revision = render_revision(store, world_id, chosen, rung)
    return {"session_id": chosen, "representation": rung,
            # The session is in the revision, so an open picture whose session
            # the Tower chose notices when the Tower would choose a newer one.
            "revision": f"{chosen}/{revision}",
            "live": session_build_running(store, world_id, chosen),
            # WORLD-BUILDER-APPEARANCE.md §9. Separate from `revision` on
            # purpose: the page follows appearance builds itself, and folding
            # them into the page revision would swap the page -- and reset the
            # wearer's camera -- on every one. `null` exactly when the
            # appearance route would 404, including a changed redaction label.
            "appearance": appearance,
            # WORLD-BUILDER-COMPONENTS.md §3.2: the same array as this session's
            # listing row, or null. NOT part of `revision`: an area finishing, or
            # its build state moving, does not change the room page and must not
            # swap it. The phone compares it by value to keep its areas row current.
            "components": _components_or_none(store, world_id, chosen)}


def _room_captions(store: WorldStore, world_id: str, session_id: str) -> dict | None:
    """`{"more_areas": N}` when this session's components record shows at least one
    area, else None. Never raises: a caption must not cost the room its page."""
    try:
        from tower.world_builder.components import read_components_record  # noqa: PLC0415

        record = read_components_record(store, world_id, session_id)
        count = len(record.areas()) if record is not None else 0
    except Exception:  # noqa: BLE001
        logger.debug("[Tower][WorldBuilder] components caption probe failed", exc_info=True)
        return None
    return {"more_areas": count} if count >= 1 else None


def _components_or_none(store: WorldStore, world_id: str, session_id: str):
    """`components.components_for_session`, never an exception: the revision must
    answer whatever the area half does. None -- one `stat` -- for every session
    without a components record."""
    try:
        from tower.world_builder.components import components_for_session  # noqa: PLC0415

        return components_for_session(store, world_id, session_id)
    except Exception:  # noqa: BLE001
        logger.warning("[Tower][WorldBuilder] components for %s/%s could not be "
                       "computed; the revision says components: null",
                       world_id, session_id, exc_info=True)
        return None


def _appearance_page(store: WorldStore, world_id: str, session_id: str, revisions: dict,
                     transport: str, *, pinned: bool,
                     captions: dict | None = None) -> str | None:
    """The appearance page, or None to walk on down the ladder.

    Served exactly when the appearance routes would serve -- the same probe
    the revision route uses -- so the rung a page declares and the rung §4a
    reports agree. A pinned `representation=appearance` that cannot be served
    is a 404, like every other pinned rung.
    """
    appearance = _appearance_revision(store, world_id, session_id)
    if appearance.get("revision") is None:
        if pinned:
            raise WorldRenderUnavailable("this session has no appearance that can be served")
        return None
    try:
        from tower.world_builder.appearance_render import (  # noqa: PLC0415
            AppearanceViewerUnavailable,
            build_appearance_page,
        )
    except Exception:  # noqa: BLE001 -- an appearance module that will not import
        logger.exception("[Tower][WorldBuilder] the appearance viewer module did not import "
                         "for %s; falling back", world_id)
        if pinned:
            raise WorldRenderUnavailable("the appearance viewer is unavailable") from None
        return None
    try:
        page = build_appearance_page(store, world_id, session_id, transport=transport,
                                     appearance_revision=appearance["revision"],
                                     **({"captions": captions} if captions else {}))
        return _stamp_revision(store, world_id, session_id, page, revisions)
    except AppearanceViewerUnavailable as exc:
        if pinned:
            raise WorldRenderUnavailable(exc.reason) from None
        logger.error(
            "[Tower][WorldBuilder] appearance for %s/%s is served by its routes but the "
            "page could not be composed (%s); serving a lower rung while the revision "
            "route reports appearance", world_id, session_id, exc.reason)
    except Exception:  # noqa: BLE001 -- never lose the world to an appearance bug
        logger.exception("[Tower][WorldBuilder] appearance viewer failed for %s; falling back",
                         world_id)
        if pinned:
            raise WorldRenderUnavailable("the appearance viewer failed") from None
    return None


STRICT_PAGE_POLICY = "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'"


def render_content_security_policy(html: str, transport: str = TRANSPORT_APP) -> str:
    """The route's CSP header for a page this module composed.

    Every rung but the appearance page loads nothing from anywhere. The
    appearance page fetches its imagery -- from the app's private scheme, or in
    the explicitly named desktop debug transport from the Tower's own origin --
    and its header states the same policy as its `<meta>`
    (WORLD-BUILDER-WORLDS.md §4 rule 6).
    """
    match = _REPRESENTATION_META.search(html, 0, 4096)
    if match is not None and match.group(1) == REPRESENTATION_APPEARANCE:
        try:
            from tower.world_builder.appearance_render import (  # noqa: PLC0415
                content_security_policy,
            )

            return content_security_policy(transport)
        except Exception:  # noqa: BLE001 -- the strictest policy, then
            logger.exception("[Tower][WorldBuilder] appearance CSP unavailable")
    return STRICT_PAGE_POLICY


def _appearance_revision(store: WorldStore, world_id: str, session_id: str) -> dict:
    """The appearance half of the revision, or `unavailable`.

    Surviving an appearance bug is right: the page revision must still answer.
    Spelling the survival `withdrawn` was not (review 2, m-15). `withdrawn` is
    the word the Tower uses for a RELABEL or a PURGE, and both consumers act on
    it as such: the page tells the wearer *"its redaction record changed, or its
    imagery was removed"*, and the app records a privacy withdrawal. So a
    traceback in `world_builder_appearance.py` told a wearer their redaction
    record had changed.

    `unavailable` is the fifth state (`WORLD-BUILDER-APPEARANCE.md` §9). It does
    everything `withdrawn` does -- the textures go, because nothing re-checked
    the label and a copy must not outlive that check, and the page keeps polling
    -- and it says the true thing while doing it. Logged at `exception` rather
    than `debug`: a bug that reaches here is a bug, and the Tower's own log was
    the only place it could be seen.
    """
    try:
        from tower.results.world_builder_appearance import (  # noqa: PLC0415
            appearance_revision,
        )

        return appearance_revision(store, world_id, session_id)
    except Exception:  # noqa: BLE001 -- the page revision must survive an appearance bug
        logger.exception("[Tower][WorldBuilder] appearance revision failed")
        return {"revision": None, "current": False, "state": "unavailable", "epoch": None}


def _stage_running(status_path, is_stale) -> bool:
    """Whether a surface or dense stage's `status.json` says `running` and the
    process that wrote it is still that process.

    **A manifest written after the `running` status means the build has
    published**, and is not live any more even though `ok` has not landed yet.
    Both stages write the manifest, then prune, then `ok` -- and a revision
    poll in that gap saw the NEW revision with `live: true`. The phone offered
    it instead of swapping it in, recorded it as handled, and the `live: false`
    poll that followed was not new: no auto-swap after Stop (review 3, R5).
    Every `running` write precedes the manifest of the same build, so "manifest
    newer than status" can only mean "this build already published". Writing
    `ok` first instead would claim a result that is not on disk yet if the
    process dies between the two.
    """
    try:
        status_stat = status_path.stat()
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(status, dict) or status.get("state") != "running":
        return False
    try:
        manifest_mtime = (status_path.parent / "manifest.json").stat().st_mtime_ns
    except OSError:
        manifest_mtime = None
    if manifest_mtime is not None and manifest_mtime > status_stat.st_mtime_ns:
        return False
    try:
        return not is_stale(status)
    except Exception:  # noqa: BLE001 -- a liveness probe must not 500 the route
        return False


def session_build_running(store: WorldStore, world_id: str, session_id: str) -> bool:
    """Whether something is building THIS session right now, so its revision
    may still change.

    Three facts, any one of which is enough:

    1. The world's writer lock is held by a live builder (the listing's `live`,
       same probe) AND this session is the one it is writing: its record is
       still open, or its finalization is still `pending`. The lock is per
       world, so an older, finished session of a world being walked again is
       not live.
    2. The surface stage's `status.json` for this session says `running` and
       its process is alive.
    3. The same for the dense stage.

    **Not a promise that nothing will change when it is false.** The builder
    releases the world lock at the end of finalization and only THEN starts
    the final surface and dense stages (`scripts/world_build_session.py`), so
    there is a gap of up to a registration's length in which this answers
    `false` and a better picture is still coming. A client must therefore
    slow down on `false`, not stop -- `WORLD-BUILDER-WORLDS.md` §4a rule 6.
    Never raises: a probe that cannot read something answers `false`, which
    costs a client latency, never a 500.
    """
    try:
        from tower.results.world_builder_library import _world_is_live  # noqa: PLC0415
        from tower.world_builder.records import FINALIZATION_PENDING  # noqa: PLC0415

        if _world_is_live(store, world_id):
            session = store.read_session(world_id, session_id)
            finalization = session.finalization or {}
            if session.ended_at is None or finalization.get("state") == FINALIZATION_PENDING:
                return True
    except Exception:  # noqa: BLE001 -- see the docstring
        logger.debug("[Tower][WorldBuilder] world lock probe failed for %s", world_id,
                     exc_info=True)
    world_dir = store.world_dir(world_id)
    try:
        from tower.world_builder.surface_pipeline import (  # noqa: PLC0415
            status_is_stale as surface_status_is_stale,
        )

        if _stage_running(world_dir / "surface" / session_id / "status.json",
                          surface_status_is_stale):
            return True
        # The appearance stage runs after the surface in the same child and
        # writes the same kind of status; while it runs, a better picture is
        # still coming.
        if _stage_running(world_dir / "appearance" / session_id / "status.json",
                          surface_status_is_stale):
            return True
    except Exception:  # noqa: BLE001 -- a surface module that will not import
        logger.debug("[Tower][WorldBuilder] surface liveness probe failed", exc_info=True)
    try:
        from tower.world_builder.dense_pipeline import (  # noqa: PLC0415
            status_is_stale as dense_status_is_stale,
        )

        if _stage_running(world_dir / "dense" / session_id / "status.json",
                          dense_status_is_stale):
            return True
    except Exception:  # noqa: BLE001 -- a dense module that will not import
        logger.debug("[Tower][WorldBuilder] dense liveness probe failed", exc_info=True)
    return False


def build_world_render(store: WorldStore, world_id: str, session_id: str | None, *,
                       max_points: int | None = None,
                       view: str | None = None,
                       representation: str = REPRESENTATION_AUTO,
                       transport: str = TRANSPORT_APP,
                       viewer: str | None = None) -> str:
    """The viewer page for one session of one world, or
    `WorldRenderUnavailable` naming what is missing.

    `world_id` goes through the geometry adapter's containment guard
    first, for the reason its docstring gives: on Windows a backslash is
    a separator Starlette's route pattern does not exclude, and every
    store path below is joined from this id. An id that escapes the root
    is "no world", and the id is answered under its canonical spelling.

    `viewer` is what the client declared it can draw (`viewer_draws_appearance`).
    `auto` starts at the appearance rung only for a client that declared it,
    and at the surface otherwise; a PINNED `representation=appearance` is served
    regardless, because a caller that names the rung is asking for that page.
    """
    contained = contained_world_id(store, world_id)
    if contained is None:
        raise WorldRenderUnavailable(f"no world {_clip(world_id)!r}")
    world_id = contained
    chosen = resolve_session(store, world_id, session_id)
    # Walk the ladder from the requested rung down. A session that has a
    # surface gets the surface, because that is the whole point of having
    # built one; one that has only points gets points; every world built
    # before either stage existed falls through to the sparse page unchanged.
    revisions = {}
    for rung in REPRESENTATION_LADDER:
        own = render_revision(store, world_id, chosen, rung)
        if own is not None:
            revisions[rung] = f"{chosen}/{own}"
    wanted = (representation if representation in REPRESENTATION_LADDER
              else REPRESENTATION_APPEARANCE if viewer_draws_appearance(viewer)
              else REPRESENTATION_SURFACE)
    # The solver's diagnostic view IS the sparse page -- only it has the
    # diagnostic rendering -- so asking for it with no representation pinned
    # starts the ladder at sparse. Starting at the surface served the surface
    # for "open the solver's view", and the app's text describing a
    # diagnostics rendering was false.
    if view == VIEW_DIAGNOSTICS and representation == REPRESENTATION_AUTO:
        wanted = REPRESENTATION_SPARSE
    start = REPRESENTATION_LADDER.index(wanted)

    # WORLD-BUILDER-COMPONENTS.md §4: " · N more areas shown separately" on the room
    # page of a session with at least one area; None -- the page exactly as before --
    # for every other session, including every one with `components: null`.
    room_captions = _room_captions(store, world_id, chosen)
    if start <= REPRESENTATION_LADDER.index(REPRESENTATION_APPEARANCE):
        page = _appearance_page(store, world_id, chosen, revisions, transport,
                                pinned=representation == REPRESENTATION_APPEARANCE,
                                captions=room_captions)
        if page is not None:
            return page

    # A surface or dense artifact whose colours came from a re-redacted keyframe
    # set the session no longer reads (a `--revert`) is not served until it is
    # rebuilt; the ladder steps past it, as the revision route does
    # (`store.built_from_an_inactive_keyframe_set`, review 1 m4).
    stale = {rung: built_from_an_inactive_keyframe_set(store, world_id, chosen, rung)
             for rung in (REPRESENTATION_SURFACE, REPRESENTATION_DENSE)}
    if representation in stale and stale[representation]:
        raise WorldRenderUnavailable(
            f"this session's {representation} was built from a re-redacted keyframe set it "
            "no longer reads; it is served again once rebuilt")

    if start <= REPRESENTATION_LADDER.index(REPRESENTATION_SURFACE) and not stale[REPRESENTATION_SURFACE]:
        # Same shape as the dense rung below, and for the same reason: the
        # import sits OUTSIDE the try that catches its exception, so a
        # surface module that will not import degrades to the next rung
        # instead of raising a NameError out of the except clause.
        try:
            from tower.world_builder.surface_render import (  # noqa: PLC0415
                SurfaceViewerUnavailable,
                build_surface_page,
            )
        except Exception:  # noqa: BLE001 -- a surface module that will not import
            logger.exception(
                "[Tower][WorldBuilder] the surface viewer module did not import "
                "for %s; falling back", world_id,
            )
            SurfaceViewerUnavailable = _ViewerModuleMissing  # noqa: N806
            build_surface_page = None
        try:
            if build_surface_page is None:
                raise RuntimeError("surface viewer unavailable")
            page = build_surface_page(store, world_id, chosen, max_points=max_points,
                                      **({"captions": room_captions} if room_captions else {}))
            return _stamp_revision(store, world_id, chosen, page, revisions)
        except SurfaceViewerUnavailable as exc:
            if representation == REPRESENTATION_SURFACE:
                raise WorldRenderUnavailable(exc.reason) from None
            if surface_artifact_drawable(store, world_id, chosen):
                # The revision route (§4a) decides the rung by this same
                # header check, so it is now telling the phone "surface"
                # about a page stamped with a lower rung. The phone pays one
                # fetch per rebuild for that, not a loop, but the operator
                # must hear about an artifact that passes its check and still
                # cannot be drawn.
                logger.error(
                    "[Tower][WorldBuilder] surface artifact for %s/%s passes its "
                    "header check but the page could not be built (%s); serving a "
                    "lower rung while the revision route reports surface",
                    world_id, chosen, exc.reason,
                )
        except Exception:  # noqa: BLE001 -- never lose the world to a surface bug
            logger.exception(
                "[Tower][WorldBuilder] surface viewer failed for %s; falling back",
                world_id,
            )

    # Gated on where the walk STARTED, not on the representation asked for:
    # `view=diagnostics` arrives as `auto` and starts at sparse, and gating on
    # `representation != sparse` served it the dense page whenever a dense
    # artifact existed -- while the revision route said sparse (review 3, R3).
    if start <= REPRESENTATION_LADDER.index(REPRESENTATION_DENSE) and not stale[REPRESENTATION_DENSE]:
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
            DenseViewerUnavailable = _ViewerModuleMissing  # noqa: N806
            build_dense_page = None

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
            return _stamp_revision(store, world_id, chosen,
                                   build_dense_page(store, world_id, chosen,
                                                    budget_bytes=budget),
                                   revisions)
        except DenseViewerUnavailable as exc:
            if representation == REPRESENTATION_DENSE:
                raise WorldRenderUnavailable(exc.reason) from None
            if dense_artifact_drawable(store, world_id, chosen):
                # Same disagreement as the surface rung above, one rung down.
                logger.error(
                    "[Tower][WorldBuilder] dense artifact for %s/%s passes its "
                    "header check but the page could not be built (%s); serving "
                    "sparse while the revision route reports dense",
                    world_id, chosen, exc.reason,
                )
        except Exception:  # noqa: BLE001 -- never lose the sparse page to a dense bug
            logger.exception(
                "[Tower][WorldBuilder] dense viewer failed for %s; serving sparse",
                world_id,
            )

    budget = MOBILE_MAX_POINTS if max_points is None else min(max_points, MAX_POINTS_CEILING)
    try:
        page = render_html(store, world_id, chosen, max_points=budget,
                           view=view or VIEW_PRODUCT)
        return _stamp_revision(store, world_id, chosen, page, revisions)
    except FileNotFoundError:
        # Raced a `clear_derived` between the existence check and the read.
        # Worded here rather than from the exception: the phone shows the
        # detail verbatim, and an OSError's text carries a filesystem path
        # (contract §3 rule 3: no paths on the wire).
        raise WorldRenderUnavailable(
            f"session {chosen!r} of world {world_id!r} has no geometry yet") from None


# ---------------------------------------------------------------------------
# AREAS: `GET /worlds/{w}/areas/{s}/{a}/...` (WORLD-BUILDER-COMPONENTS.md §5)
# ---------------------------------------------------------------------------
#
# A part of a walk the evidence gate could not place, drawn ON ITS OWN, in its own
# levelled frame, by the same page programs as the room fed the area's artifacts.
# Nothing here composites an area with the room or names the room's geometry: the
# area's artifacts live under `<world>/areas/<area>/` and are reached through
# `components.AreaStore`, a view that answers everything about the SESSION (its
# keyframes, its redaction label, `world.json`) from the real world, so every check the
# room's pages and appearance routes make is made again, unchanged, for the area.
#
# Kept in this adapter rather than a new file so the architecture boundary's adapter
# list does not move: it is the render route's own family.


class AreaUnavailable(Exception):
    """An area route's 404. `reason` is the `detail` the phone shows; four of them are
    stable identifiers (`components.NO_SUCH_AREA` and its siblings, C1 E4)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class _Area:
    __slots__ = ("world_id", "session_id", "area_id", "world", "session", "record",
                 "entry", "view")

    def __init__(self, **fields) -> None:
        for key, value in fields.items():
            setattr(self, key, value)


def resolve_area(store: WorldStore, world_id: str, session_id: str, area_id: str) -> _Area:
    """The area a request names, checked in the order a path is joined from it.

    The world is contained first (`contained_world_id`), the session is checked
    against the world's own sessions before the record path is joined from it, and
    the area id is checked against the CURRENT components record -- 16 lower-hex AND
    a `shown_as: "area"` entry -- before the area directory is joined from it. The
    room's id and a `none` entry's id are `no such area in this session` (§5).
    """
    from tower.world_builder import components as C  # noqa: PLC0415

    contained = contained_world_id(store, world_id)
    if contained is None:
        raise AreaUnavailable(f"no world {_clip(world_id)!r}")
    try:
        world = store.read_world(contained)
    except (WorldStoreError, OSError, ValueError, KeyError):
        raise AreaUnavailable(f"no world {_clip(world_id)!r}") from None
    listed = {sid for sid in world.session_ids if isinstance(sid, str)}
    if session_id not in set(store.list_session_ids(contained)) | listed:
        raise AreaUnavailable(
            f"world {_clip(world_id)!r} has no session {_clip(session_id)!r}")
    # NULL FIRST: every older world answers this and only this (§7 rule 1), whatever
    # the id looks like.
    record = C.read_components_record(store, contained, session_id)
    if record is None:
        raise AreaUnavailable(C.NO_AREAS)
    entry = record.entry(area_id) if C.is_area_id(area_id) else None
    if entry is None or entry["shown_as"] != C.SHOWN_AREA:
        raise AreaUnavailable(C.NO_SUCH_AREA)
    try:
        session = store.read_session(contained, session_id)
    except (WorldStoreError, OSError, ValueError, KeyError):
        raise AreaUnavailable(
            f"world {_clip(world_id)!r} has no session {_clip(session_id)!r}") from None
    return _Area(world_id=contained, session_id=session_id, area_id=area_id, world=world,
                 session=session, record=record, entry=entry,
                 view=C.AreaStore(store, contained, session_id, area_id))


def _area_word(store: WorldStore, area: _Area) -> dict:
    from tower.world_builder import components as C  # noqa: PLC0415

    return C.area_photographic_state(store, area.world_id, area.session_id, area.session,
                                     area.area_id, area.record.sha1)


def _area_not_drawable(store: WorldStore, area: _Area) -> AreaUnavailable:
    """Nothing is drawable: transient while its build is owed, running or
    unobservable; terminal once it is settled (failed, or declined)."""
    from tower.world_builder import components as C  # noqa: PLC0415
    from tower.world_builder.photographic import is_unsettled  # noqa: PLC0415

    try:
        word = _area_word(store, area)
    except Exception:  # noqa: BLE001 -- not known is not terminal
        return AreaUnavailable(C.AREA_NOT_BUILT_YET)
    if is_unsettled(word.get("state")):
        return AreaUnavailable(C.AREA_NOT_BUILT_YET)
    return AreaUnavailable(C.AREA_COULD_NOT_BE_BUILT)


def _area_levelled(area: _Area):
    """`area.levelled` as the build recorded it, or None when unknown."""
    from tower.world_builder import components as C  # noqa: PLC0415

    record = C.read_area_record(area.view.base, area.world_id, area.session_id,
                                area.area_id) or {}
    value = record.get("levelled")
    return value if isinstance(value, bool) else None


def _area_appearance_revision(store: WorldStore, area: _Area) -> dict:
    """WORLDS §4a's `appearance` object, for the area's artifact (§5.2). Never raises."""
    from tower.world_builder import appearance_pipeline as AP  # noqa: PLC0415
    from tower.world_builder import components as C  # noqa: PLC0415

    not_served = {"revision": None, "current": False, "state": AP.ABSENT, "epoch": None}
    try:
        manifest, reason = C.area_appearance_servable(store, area.world_id, area.session_id,
                                                      area.area_id, world=area.world)
    except Exception:  # noqa: BLE001 -- the safe answer is to drop
        logger.debug("[Tower][WorldBuilder] area appearance probe failed", exc_info=True)
        return {**not_served, "state": AP.WITHDRAWN}
    if manifest is None:
        if reason == AP.STALE_LABEL_DETAIL:
            try:
                state = AP.withdrawal_state(
                    area.view, area.world_id, area.session_id,
                    AP.read_appearance_manifest(area.view, area.world_id, area.session_id))
            except Exception:  # noqa: BLE001
                state = AP.WITHDRAWN
        elif reason == "appearance imagery was purged":
            state = AP.WITHDRAWN
        else:
            state = AP.ABSENT
        return {**not_served, "state": state}
    build = manifest.get("build_id")
    if not isinstance(build, str):
        return {**not_served, "state": AP.WITHDRAWN}
    try:
        current = bool(AP.appearance_currency(area.view, area.world_id, area.session_id,
                                              manifest)["current"])
    except Exception:  # noqa: BLE001
        current = False
    epoch = manifest.get("epoch")
    return {"revision": f"{area.session_id}/area:{area.area_id}/appearance:{build}",
            "current": current, "state": AP.SERVED,
            "epoch": epoch if isinstance(epoch, str) else None}


def _area_page_revision(area: _Area, appearance: dict) -> str | None:
    """`<session>/area:<area>/appearance:1@<epoch>` (§5.1), or None."""
    try:
        from tower.world_builder.appearance_render import PAGE_REVISION  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return None
    epoch = appearance.get("epoch")
    own = f"{PAGE_REVISION}@{epoch}" if epoch else PAGE_REVISION
    return f"{area.session_id}/area:{area.area_id}/{own}"


def _area_surface_revision(area: _Area) -> str | None:
    """`<session>/area:<area>/surface:<built_at>` (§5.2), or None."""
    own = render_revision(area.view, area.world_id, area.session_id, REPRESENTATION_SURFACE)
    return None if own is None else f"{area.session_id}/area:{area.area_id}/{own}"


def _area_live(area: _Area) -> bool:
    """§5.2 `live`: this area's surface or appearance stage is running under a live
    process and has not yet published (§4a's rule, per area). Never raises."""
    try:
        from tower.world_builder.surface_pipeline import (  # noqa: PLC0415
            status_is_stale,
        )
    except Exception:  # noqa: BLE001
        return False
    root = area.view.world_dir(area.world_id)
    for stage in ("surface", "appearance"):
        try:
            if _stage_running(root / stage / area.session_id / "status.json",
                              status_is_stale):
                return True
        except Exception:  # noqa: BLE001
            continue
    return False


def _stamp_area(html: str, revision: str | None, area_id: str) -> str:
    """`wb-revision` and `wb-area` right after the page's `wb-representation`, inside
    its first 4096 characters (§5.1). No revision stamp rather than a false one."""
    match = _REPRESENTATION_META.search(html, 0, 4096)
    if match is None:
        return html
    tags = ""
    if revision is not None:
        tags += f'<meta name="wb-revision" content="{html_escape(revision, quote=True)}">'
    tags += f'<meta name="wb-area" content="{html_escape(area_id, quote=True)}">'
    return html[:match.end()] + tags + html[match.end():]


def area_routes(world_id: str, session_id: str, area_id: str) -> dict:
    """The four addresses the area's appearance page fetches, relative to its base
    (§5.4) -- the §5.5 whitelist, and nothing else. The revision carries no query."""
    from urllib.parse import quote  # noqa: PLC0415

    w, s, a = (quote(v, safe="") for v in (world_id, session_id, area_id))
    prefix = f"/worlds/{w}/areas/{s}/{a}"
    return {
        "manifest": f"{prefix}/appearance/manifest",
        "chunk": f"{prefix}/appearance/chunk/",
        "proxy": f"{prefix}/appearance/proxy/",
        "revision": f"{prefix}/render/revision",
    }


def _area_caption_info(area: _Area) -> dict:
    """What the area page's captions need (§5.4): its number among the session's
    areas (§2.4 rule 2, the record's order), how many there are, when in the walk it
    was captured (first span start to last span end, the Tower's receipt clock), and
    whether it was levelled."""
    areas = area.record.areas()
    ids = [e["id"] for e in areas]
    spans = area.entry.get("capture_spans_s") or []
    return {"number": ids.index(area.area_id) + 1, "of": len(ids),
            "from_s": spans[0][0] if spans else None,
            "to_s": spans[-1][1] if spans else None,
            "levelled": _area_levelled(area)}


def _area_appearance_page(area: _Area, appearance: dict, transport: str) -> str | None:
    """The room's appearance page program, fed the area's artifacts, addresses and
    captions. None to fall to the surface rung."""
    try:
        from tower.world_builder import appearance_render as AR  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        logger.exception("[Tower][WorldBuilder] the appearance viewer module did not import")
        return None
    try:
        page = AR.build_appearance_page(
            area.view, area.world_id, area.session_id, transport=transport,
            appearance_revision=appearance["revision"],
            routes=area_routes(area.world_id, area.session_id, area.area_id),
            # Additive (§5.4): which area, and whether its vertical could be estimated.
            extra_config={"area": {"id": area.area_id, "levelled": _area_levelled(area)}},
            captions={"area": _area_caption_info(area)})
    except Exception as exc:  # noqa: BLE001 -- never lose the area to an appearance bug
        logger.warning("[Tower][WorldBuilder] area appearance page for %s/%s/%s could not "
                       "be composed (%s); falling back", area.world_id, area.session_id,
                       area.area_id, getattr(exc, "reason", type(exc).__name__))
        return None
    return _stamp_area(page, _area_page_revision(area, appearance), area.area_id)


def _pinned_rung_missing(store: WorldStore, area: _Area, rung: str) -> AreaUnavailable:
    """A named rung the area does not have (§5.1, "as in §4")."""
    from tower.world_builder import components as C  # noqa: PLC0415
    from tower.world_builder.photographic import is_unsettled  # noqa: PLC0415

    try:
        if is_unsettled(_area_word(store, area).get("state")):
            return AreaUnavailable(C.AREA_NOT_BUILT_YET)
    except Exception:  # noqa: BLE001
        return AreaUnavailable(C.AREA_NOT_BUILT_YET)
    return AreaUnavailable(f"this area has no {rung} that can be served")


def build_area_render(store: WorldStore, world_id: str, session_id: str, area_id: str, *,
                      max_points: int | None = None,
                      representation: str = REPRESENTATION_AUTO,
                      transport: str = TRANSPORT_APP,
                      viewer: str | None = None) -> str:
    """`GET /worlds/{w}/areas/{s}/{a}/render` (§5.1): the area's page, or
    `AreaUnavailable`.

    The ladder is the room's, shortened to the two rungs an area is built as:
    appearance, then surface. `viewer` is accepted and ignored -- `auto` offers the
    appearance rung unconditionally, because every client that can name this route
    postdates the scheme handler (§5.1, OPEN M12).
    """
    del viewer  # accepted, ignored (§5.1)
    from tower.world_builder import components as C  # noqa: PLC0415

    area = resolve_area(store, world_id, session_id, area_id)
    if representation in (REPRESENTATION_SPARSE, REPRESENTATION_DENSE):
        raise AreaUnavailable(C.AREA_NO_SPARSE_OR_DENSE)
    if representation in (REPRESENTATION_AUTO, REPRESENTATION_APPEARANCE):
        appearance = _area_appearance_revision(store, area)
        if appearance.get("revision") is not None:
            page = _area_appearance_page(area, appearance, transport)
            if page is not None:
                return page
        if representation == REPRESENTATION_APPEARANCE:
            raise _pinned_rung_missing(store, area, REPRESENTATION_APPEARANCE)
    if C.area_surface_drawable(store, area.world_id, area.session_id, area.area_id):
        try:
            from tower.world_builder.surface_render import (  # noqa: PLC0415
                SurfaceViewerUnavailable,
                build_surface_page,
            )
        except Exception:  # noqa: BLE001
            logger.exception("[Tower][WorldBuilder] the surface viewer module did not import")
            SurfaceViewerUnavailable = _ViewerModuleMissing  # noqa: N806
            build_surface_page = None
        try:
            if build_surface_page is None:
                raise RuntimeError("surface viewer unavailable")
            revision = _area_surface_revision(area)
            page = build_surface_page(area.view, area.world_id, area.session_id,
                                      max_points=max_points,
                                      captions={"area": _area_caption_info(area)})
            return _stamp_area(page, revision, area.area_id)
        except SurfaceViewerUnavailable as exc:
            logger.error("[Tower][WorldBuilder] area surface for %s/%s/%s passes its header "
                         "check but the page could not be built (%s)", area.world_id,
                         area.session_id, area.area_id, exc.reason)
        except Exception:  # noqa: BLE001
            logger.exception("[Tower][WorldBuilder] area surface viewer failed")
    elif representation == REPRESENTATION_SURFACE:
        raise _pinned_rung_missing(store, area, REPRESENTATION_SURFACE)
    raise _area_not_drawable(store, area)


def build_area_revision(store: WorldStore, world_id: str, session_id: str,
                        area_id: str) -> dict:
    """`GET /worlds/{w}/areas/{s}/{a}/render/revision` (§5.2). 404 exactly when the
    render route would: the same resolution, the same ladder, by artifact checks."""
    from tower.world_builder import components as C  # noqa: PLC0415

    area = resolve_area(store, world_id, session_id, area_id)
    appearance = _area_appearance_revision(store, area)
    representation = revision = None
    if appearance.get("revision") is not None:
        revision = _area_page_revision(area, appearance)
        if revision is not None:
            representation = REPRESENTATION_APPEARANCE
    if representation is None and C.area_surface_drawable(
            store, area.world_id, area.session_id, area.area_id):
        revision = _area_surface_revision(area)
        if revision is not None:
            representation = REPRESENTATION_SURFACE
    if representation is None:
        raise _area_not_drawable(store, area)
    return {"session_id": area.session_id, "area_id": area.area_id,
            "representation": representation, "revision": revision,
            "live": _area_live(area), "appearance": appearance}


def area_appearance_manifest(store: WorldStore, world_id: str, session_id: str,
                             area_id: str):
    """(payload, effective label, imagery) of the area's appearance manifest, with
    `currency`, under APPEARANCE §9 in every respect (§5.3). Raises the appearance
    adapter's `AppearanceNotServed`."""
    from tower.results.world_builder_appearance import (  # noqa: PLC0415
        AppearanceNotServed,
        _imagery,
        _label,
    )
    from tower.world_builder import appearance_pipeline as AP  # noqa: PLC0415
    from tower.world_builder import components as C  # noqa: PLC0415

    try:
        area = resolve_area(store, world_id, session_id, area_id)
    except AreaUnavailable as exc:
        raise AppearanceNotServed(exc.reason) from None
    manifest, reason = C.area_appearance_servable(store, area.world_id, area.session_id,
                                                  area.area_id, world=area.world)
    if manifest is None:
        raise AppearanceNotServed(reason)
    payload = dict(manifest)
    payload["currency"] = AP.appearance_currency(area.view, area.world_id, area.session_id,
                                                 manifest)
    payload.setdefault("area", {"id": area.area_id, "levelled": _area_levelled(area)})
    return payload, _label(manifest), _imagery(manifest)


def area_appearance_file(store: WorldStore, world_id: str, session_id: str, area_id: str,
                         kind: str, digest: str):
    """(bytes, effective label, imagery) of a chunk or the proxy the area's current
    manifest names (or one superseded less than 120 s ago), §5.3."""
    from tower.results.world_builder_appearance import (  # noqa: PLC0415
        AppearanceNotServed,
        _imagery,
        _label,
    )
    from tower.world_builder import appearance_pipeline as AP  # noqa: PLC0415
    from tower.world_builder import components as C  # noqa: PLC0415

    if kind not in ("chunk", "proxy") or not AP.is_digest(digest):
        raise AppearanceNotServed("no such appearance file")
    try:
        area = resolve_area(store, world_id, session_id, area_id)
    except AreaUnavailable as exc:
        raise AppearanceNotServed(exc.reason) from None
    manifest, reason = C.area_appearance_servable(store, area.world_id, area.session_id,
                                                  area.area_id, world=area.world)
    if manifest is None:
        raise AppearanceNotServed(reason)
    data = AP.read_appearance_file(area.view, area.world_id, area.session_id, kind, digest,
                                   manifest)
    if data is None:
        raise AppearanceNotServed("no such appearance file")
    return data, _label(manifest), _imagery(manifest)
