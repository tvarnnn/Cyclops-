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

import json
import os

from tower.storage import read_json_closed
from tower.world_builder.store import WorldStore, WorldStoreError, _pid_is_running

WORLDS_CONTRACT = "world_builder.worlds/2026-09-06"


def _world_is_live(store: WorldStore, world_id: str) -> bool:
    path = store.lock_path(world_id)
    if not path.exists():
        return False
    try:
        holder = read_json_closed(path)
    except (OSError, json.JSONDecodeError):
        return False
    pid = holder.get("pid")
    return isinstance(pid, int) and pid != os.getpid() and _pid_is_running(pid)


def _has_geometry(store: WorldStore, world_id: str, session_id: str) -> bool:
    derived = store.derived_dir(world_id) / session_id
    return (derived / "poses.json").exists() and (derived / "points.json").exists()


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
            sessions.append({
                "session_id": session.session_id,
                "started_at": session.started_at,
                "ended_at": session.ended_at,
                "end_reason": session.end_reason,
                "frame_source": session.frame_source,
                "capture_id": session.capture_id,
                "keyframes_accepted": session.keyframes_accepted,
                "has_geometry": _has_geometry(store, world_id, session_id),
                # The record is only finalised by `stop_session`; a builder
                # killed before that (the supervisor's shutdown grace, a
                # hard kill) leaves `ended_at: null` behind forever. Open
                # with nobody writing is not "still open", and the phone
                # would otherwise say exactly that. Counts on such a record
                # are the start-of-session values, not the journal length.
                "abandoned": session.ended_at is None and not live,
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
