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
import zlib

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


# Content-Encoding for the appearance bodies (WORLD-BUILDER-APPEARANCE.md §9).
#
# The canonical world's phone load, measured over these routes: 18,229,238 B
# in 10 requests uncompressed, 14,103,933 B gzipped (0.774): ASTC chunks 0.87,
# the WBSURF01 proxy 0.57, the manifest JSON 0.20. On the largest canonical files
# level 6 is 0.794 at ~35 ms per MB, level 1 0.803 at ~22 ms per MB; 6 is chosen
# because the bytes cross a phone's radio and the CPU is the Tower's.
#
# Only what the client says it accepts, only on a 200, and nothing about the
# privacy headers changes: `no-store`, `nosniff`, no validators. `Vary:
# Accept-Encoding` is added so no intermediary could hand a compressed body to
# a client that did not ask for one (and nothing may store it anyway).
# Compressed per request and not kept: nothing here is cached.
COMPRESS_LEVEL = 6
COMPRESS_MIN_BYTES = 1024
ENCODINGS = ("gzip", "deflate")


def negotiate_encoding(accept_encoding: str | None) -> str | None:
    """`gzip`, `deflate` or `None` from an `Accept-Encoding` header value.

    gzip is preferred when both are acceptable. A coding with `q=0` is refused,
    and `*` stands for any coding not named. `identity` is always acceptable,
    so a header that allows neither gets the plain body, never a 406.
    """
    if not accept_encoding:
        return None
    quality: dict[str, float] = {}
    for part in str(accept_encoding).split(","):
        fields = [f.strip() for f in part.split(";")]
        coding = fields[0].lower()
        if not coding:
            continue
        q = 1.0
        for param in fields[1:]:
            name, _, value = param.partition("=")
            if name.strip().lower() == "q":
                try:
                    q = float(value.strip())
                except ValueError:
                    q = 0.0
        quality[coding] = q
    for coding in ENCODINGS:
        q = quality.get(coding, quality.get("*", 0.0))
        if q > 0.0:
            return coding
    return None


def encode_body(data: bytes, accept_encoding: str | None) -> tuple[bytes, dict]:
    """The body to send and the headers that describe its encoding.

    `deflate` is the zlib format (RFC 9110 §8.4.1.2), which is what browsers
    and `URLSession` decode under that name.
    """
    coding = negotiate_encoding(accept_encoding) if len(data) >= COMPRESS_MIN_BYTES else None
    if coding == "gzip":
        # A fixed header timestamp, so the same bytes always encode identically.
        compressor = zlib.compressobj(COMPRESS_LEVEL, zlib.DEFLATED, 16 + zlib.MAX_WBITS)
        body = compressor.compress(data) + compressor.flush()
    elif coding == "deflate":
        body = zlib.compress(data, COMPRESS_LEVEL)
    else:
        return data, {"Vary": "Accept-Encoding"}
    return body, {"Content-Encoding": coding, "Vary": "Accept-Encoding"}


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


def _not_served_state(store: WorldStore, world_id: str, session_id: str, reason: str) -> str:
    """`rebuilding`, `withdrawn` or `absent` for an appearance the routes do
    not serve now (§9 `appearance.state`)."""
    if reason == AP.STALE_LABEL_DETAIL:
        contained = contained_world_id(store, world_id)
        manifest = (AP.read_appearance_manifest(store, contained, session_id)
                    if contained is not None else None)
        return AP.withdrawal_state(store, contained, session_id, manifest)
    if reason == "appearance imagery was purged":
        return AP.WITHDRAWN
    return AP.ABSENT


def appearance_revision(store: WorldStore, world_id: str, session_id: str) -> dict:
    """`{revision, current, state, epoch}` for the revision route (§9). Never
    raises: a revision probe that cannot read something answers `null`.

    `state` is `served` exactly when `revision` is a string. Otherwise it says
    what an open page should do with what it already drew:

    - `rebuilding`: the label changed in a way its textures carry over
      (`appearance_pipeline.textures_carry_over`; the ordinary Stop, `none` ->
      the real label, over a walk-time build that re-redacted every frame). Keep
      drawing, keep asking; the next build is expected.
    - `withdrawn`: the label changed any other way, or the imagery was purged.
      Drop the textures now; keep asking.
    - `absent`: there is no appearance (never built, or the world is gone).

    `epoch` is the served manifest's (§9): it changes exactly when an open page
    must drop its textures before drawing the new build.
    """
    not_served = {"revision": None, "current": False, "state": AP.ABSENT, "epoch": None}
    try:
        contained, manifest = _servable_manifest(store, world_id, session_id)
    except AppearanceNotServed as exc:
        try:
            state = _not_served_state(store, world_id, session_id, exc.reason)
        except Exception:  # noqa: BLE001 -- the safe answer is to drop
            logger.debug("[Tower][WorldBuilder] appearance state probe failed", exc_info=True)
            state = AP.WITHDRAWN
        return {**not_served, "state": state}
    except Exception:  # noqa: BLE001 -- a probe must not 500 the revision route
        logger.debug("[Tower][WorldBuilder] appearance revision probe failed", exc_info=True)
        return {**not_served, "state": AP.WITHDRAWN}
    build = manifest.get("build_id")
    if not isinstance(build, str):
        return {**not_served, "state": AP.WITHDRAWN}
    try:
        current = bool(AP.appearance_currency(store, contained, session_id, manifest)["current"])
    except Exception:  # noqa: BLE001
        current = False
    epoch = manifest.get("epoch")
    return {"revision": f"{session_id}/appearance:{build}", "current": current,
            "state": AP.SERVED, "epoch": epoch if isinstance(epoch, str) else None}
