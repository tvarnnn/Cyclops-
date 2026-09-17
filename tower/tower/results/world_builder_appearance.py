"""The appearance artifact, over HTTP.

Contract: `docs/contracts/WORLD-BUILDER-APPEARANCE.md` §9.

The fifth adapter named after this cartridge, for the reason the others are:
it reads World Builder's record shapes, so it is a file outside the package
that may. It answers "give the phone the keyframes it blends", which is
imagery -- so every answer is re-checked against the session's redaction record
at the moment it is served, not only when it was built.

Nothing here is cached and nothing is written.
"""

from __future__ import annotations

import logging

from tower.results.world_builder_geometry import contained_world_id
from tower.world_builder import appearance_pipeline as AP
from tower.world_builder.store import WorldStore, WorldStoreError

logger = logging.getLogger(__name__)

# Every response, 200 or 404: this is first-person imagery of a private space.
# No validators either -- an ETag or Last-Modified invites a cache to keep it.
NO_STORE_HEADERS = {
    "Cache-Control": "no-store",
    "Pragma": "no-cache",
    "X-Content-Type-Options": "nosniff",
}


class AppearanceNotServed(Exception):
    """The route's 404, with the sentence the client shows."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _clip(value: str, limit: int = 80) -> str:
    return value if len(value) <= limit else value[:limit] + "..."


def _servable_manifest(store: WorldStore, world_id: str, session_id: str):
    """(canonical world id, manifest) when this session's appearance may be
    served right now, else `AppearanceNotServed`."""
    contained = contained_world_id(store, world_id)
    if contained is None:
        raise AppearanceNotServed(f"no world {_clip(world_id)!r}")
    try:
        world = store.read_world(contained)
    except (WorldStoreError, OSError, ValueError, KeyError):
        raise AppearanceNotServed(f"no world {_clip(world_id)!r}") from None
    listed = {sid for sid in world.session_ids if isinstance(sid, str)}
    if session_id not in set(store.list_session_ids(contained)) | listed:
        raise AppearanceNotServed(
            f"world {_clip(world_id)!r} has no session {_clip(session_id)!r}")
    if getattr(world, "images_purged", False):
        raise AppearanceNotServed("appearance imagery was purged")
    manifest = AP.read_appearance_manifest(store, contained, session_id)
    if manifest is None:
        raise AppearanceNotServed("no appearance for this session")
    if not AP.label_matches(store, contained, session_id, manifest):
        raise AppearanceNotServed(AP.STALE_LABEL_DETAIL)
    return contained, manifest


def _label(manifest: dict) -> str:
    return str((manifest.get("appearance_provenance") or {}).get("redaction_effective"))


def appearance_manifest(store: WorldStore, world_id: str, session_id: str):
    """(payload, effective label). The manifest as built, plus `currency`."""
    contained, manifest = _servable_manifest(store, world_id, session_id)
    payload = dict(manifest)
    payload["currency"] = AP.appearance_currency(store, contained, session_id, manifest)
    return payload, _label(manifest)


def appearance_file(store: WorldStore, world_id: str, session_id: str, kind: str,
                    digest: str):
    """(bytes, effective label) of a chunk or the proxy the manifest names."""
    if kind not in ("chunk", "proxy") or not AP.is_digest(digest):
        raise AppearanceNotServed("no such appearance file")
    contained, manifest = _servable_manifest(store, world_id, session_id)
    data = AP.read_appearance_file(store, contained, session_id, kind, digest, manifest)
    if data is None:
        raise AppearanceNotServed("no such appearance file")
    return data, _label(manifest)


def appearance_revision(store: WorldStore, world_id: str, session_id: str) -> dict:
    """`{revision, current}` for the revision route (§9). Never raises: a
    revision probe that cannot read something answers `null`."""
    try:
        contained, manifest = _servable_manifest(store, world_id, session_id)
    except AppearanceNotServed:
        return {"revision": None, "current": False}
    except Exception:  # noqa: BLE001 -- a probe must not 500 the revision route
        logger.debug("[Tower][WorldBuilder] appearance revision probe failed", exc_info=True)
        return {"revision": None, "current": False}
    build = manifest.get("build_id")
    if not isinstance(build, str):
        return {"revision": None, "current": False}
    try:
        current = bool(AP.appearance_currency(store, contained, session_id, manifest)["current"])
    except Exception:  # noqa: BLE001
        current = False
    return {"revision": f"{session_id}/appearance:{build}", "current": current}
