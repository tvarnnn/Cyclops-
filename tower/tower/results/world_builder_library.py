"""The list of saved worlds, for a viewer that wants to open an old one.

Contract: `world_builder.worlds/2026-09-06` (docs/contracts/WORLD-BUILDER-WORLDS.md).

Read-only, and deliberately thin: it is the index a phone needs to choose
a `(world_id, session_id)` pair, which the status subscription
(`result_subscribe` with `world_id`/`session_id`) and the geometry routes
already accept. It carries no geometry, no imagery and no per-keyframe
data -- those stay where they are.

`live` is answered from the world's writer lock: a lock file whose pid is
still running means a builder is writing that world right now. That is
the same liveness question `WorldStore.acquire_writer_lock` asks, and it
is asked of the OS rather than of a timestamp for the same reason.
"""

from __future__ import annotations

import os

from tower.world_builder.records import FINALIZATION_COMPLETE
from tower.world_builder.store import WorldStore, WorldStoreError

WORLDS_CONTRACT = "world_builder.worlds/2026-09-06"

# Per-session `state`, the same vocabulary the status channel's lifecycle
# uses (`tower/results/world_builder.py`), minus the two words that only
# make sense against a live subscription (`idle`, `unavailable`). A row in
# the picker and the panel it opens must not disagree about what a
# session is.
SESSION_RECEIVING = "receiving"
SESSION_FINALIZING = "finalizing"
SESSION_COMPLETE = "complete"
SESSION_INTERRUPTED = "interrupted"
SESSION_UNBUILT = "unbuilt"


def _world_is_live(store: WorldStore, world_id: str) -> bool:
    holder = store.lock_holder(world_id)
    return (
        holder is not None
        and holder["alive"]
        and holder["pid"] != os.getpid()
    )


def _has_geometry(store: WorldStore, world_id: str, session_id: str) -> bool:
    derived = store.derived_dir(world_id) / session_id
    return (derived / "poses.json").exists() and (derived / "points.json").exists()


def _dense_summary(store: WorldStore, world_id: str, session_id: str) -> dict | None:
    """What dense reconstruction this session has, or None.

    ADDITIVE, and the contract identifier deliberately does not move. iOS
    parses these payloads with `JSONSerialization` into `[String: Any]` and
    reads them key by key, so a key it does not know is a key it never looks
    at -- but it equality-tests `contract` on the first line of every guard, so
    bumping that would empty the gallery on every older build. A world with no
    dense artifact reports `null` and behaves exactly as it does today.
    """
    from tower.world_builder.dense_pipeline import read_dense_manifest  # noqa: PLC0415

    manifest = read_dense_manifest(store, world_id, session_id)
    if not manifest:
        return None
    levels = manifest.get("levels") or []
    canonical = manifest.get("canonical_level", 0)
    mobile = manifest.get("mobile_level", len(levels) - 1)
    return {
        "format": manifest.get("format"),
        "levels": len(levels),
        "canonical_points": (levels[canonical]["points"]
                             if canonical < len(levels) else None),
        "mobile_points": (levels[mobile]["points"] if mobile < len(levels) else None),
        # Repeated from the manifest rather than re-derived. The dense stage
        # makes no scale claim the sparse solve did not already make.
        "scale": manifest.get("scale"),
    }


def _keyframes_journaled(store: WorldStore, world_id: str, session_id: str) -> int:
    """How many keyframes the journal holds, whatever the record says.

    `session.keyframes_accepted` is written at start (zero) and rewritten
    at stop; a session that never stopped keeps the zero forever. The
    journal is one line per keyframe, so counting lines is the honest
    figure and costs one sequential read.
    """
    path = store.keyframes_path(world_id, session_id)
    try:
        with path.open("rb") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def session_state(session, *, live: bool, has_geometry: bool) -> str:
    """One word for what a session IS, from the record, the lock and the tree.

    Mirrors `_lifecycle` in the status producer for the facts a listing
    has (it does not compute geometry currency, so `complete` on a record
    that predates finalization means "stopped and built", not "current").
    """
    finalization = session.finalization
    stopped = session.ended_at is not None
    if live and not stopped:
        return SESSION_RECEIVING
    if live and stopped:
        return SESSION_FINALIZING
    if not stopped:
        # Open record, nobody writing: killed mid-walk.
        return SESSION_INTERRUPTED
    if session.end_reason in ("error", "interrupted"):
        return SESSION_INTERRUPTED
    if finalization is not None:
        if finalization.get("state") == FINALIZATION_COMPLETE:
            return SESSION_COMPLETE
        return SESSION_INTERRUPTED
    return SESSION_COMPLETE if has_geometry else SESSION_UNBUILT


def build_world_listing(store: WorldStore) -> dict:
    """Every world with a readable `world.json`, newest first, with its
    sessions oldest first. A world whose sessions cannot be read is listed
    with what could be read; a world that cannot be read at all is left out
    rather than invented."""
    worlds = []
    for world_id in store.list_world_ids():
        try:
            world = store.read_world(world_id)
        except (WorldStoreError, OSError, ValueError, KeyError):
            continue
        live = _world_is_live(store, world_id)
        sessions = []
        for session_id in store.list_session_ids(world_id):
            try:
                session = store.read_session(world_id, session_id)
            except (WorldStoreError, OSError, ValueError, KeyError):
                continue
            has_geometry = _has_geometry(store, world_id, session_id)
            sessions.append({
                "session_id": session.session_id,
                "started_at": session.started_at,
                "ended_at": session.ended_at,
                "end_reason": session.end_reason,
                "frame_source": session.frame_source,
                "capture_id": session.capture_id,
                "keyframes_accepted": session.keyframes_accepted,
                # The journal's own count, beside the record's. On a record
                # that never stopped the record says zero and the journal
                # says what actually landed (467 on the 2026-09-06 walk).
                "keyframes_journaled": _keyframes_journaled(store, world_id, session_id),
                "has_geometry": has_geometry,
                # The record is only finalised by `stop_session`; a builder
                # killed before that (the supervisor's shutdown grace, a
                # hard kill) leaves `ended_at: null` behind forever. Open
                # with nobody writing is not "still open", and the phone
                # would otherwise say exactly that. Counts on such a record
                # are the start-of-session values, not the journal length.
                "abandoned": session.ended_at is None and not live,
                # One word, the status channel's vocabulary (additive,
                # 2026-09-06): receiving | finalizing | complete |
                # interrupted | unbuilt.
                "state": session_state(session, live=live, has_geometry=has_geometry),
                # The builder's own account of how finalization went, or
                # null on a record written before it existed.
                "finalization": session.finalization,
                # Additive, and null on every world built before the dense
                # stage existed: what dense reconstruction this session holds.
                "dense": _dense_summary(store, world_id, session_id),
            })
        sessions.sort(key=lambda s: s["started_at"])
        worlds.append({
            "world_id": world.world_id,
            "display_name": world.display_name,
            "created_at": world.created_at,
            "updated_at": world.updated_at,
            "live": live,
            "session_count": len(sessions),
            "sessions": sessions,
        })
    worlds.sort(key=lambda w: w["updated_at"], reverse=True)
    return {"contract": WORLDS_CONTRACT, "world_count": len(worlds), "worlds": worlds}
