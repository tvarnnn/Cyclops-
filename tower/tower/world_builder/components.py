"""The pieces of one session's final solve, on the wire, and the areas built from them.

Contract: `docs/contracts/WORLD-BUILDER-COMPONENTS.md` (`world_builder.components/2026-09-23`),
§2 `components[]`, §3.4 the row's `photographic` and its `scope`, §5 the area routes'
storage, §7 saved-world compatibility.

WHAT IS READ, AND WHO WRITES IT. The evidence gate (P3.2 module 2, `coherence_gate`
wired into the final solve) writes `solve/<session>/components.json` with the published
solve. This module never computes a component: it READS that record, validates it
against the contract's shape, and adds the three fields that describe what exists NOW
rather than what the gate decided (`keyframes_phone`, `has_geometry`, `photographic`).
A session without the record -- every session built before this change, and every
session of a Tower with the gate switched off -- is `components: null`, which the
contract defines as "not computed", never "no areas" (§2.4 rule 1, §7 rule 1).

THE RECORD'S SHAPE (Tower-internal; the gate's writer and this reader agree on it):

    {"components": [ {<the §2.1 fields the gate decides: id, state, reason, reasons,
                       shown_as, keyframes, capture_spans_s>,
                      "keyframe_ids": [<the member keyframe ids>]}, ... ],
     ...any other keys (gate params, digests) are ignored here}

`keyframe_ids` never leaves the Tower; it is what an area build is built from. A bare
list is accepted as the `components` array. A record that does not hold the contract's
shape is read as ABSENT (and logged): serving a malformed list would fail the phone's
decoder, and absent is the one reading the phone already handles (today's screens).

WHERE AN AREA LIVES (contract T9, chosen here): `<world>/areas/<area_id>/`, laid out as
a world directory of its one session -- `solve/<session>/` (the area's own levelled
solution), `surface/<session>/`, `appearance/<session>/`, `dense/<session>/` -- so the
existing surface and appearance pipelines build and serve it unchanged through
`AreaStore`, and `record.json`, the area's stage record (which names its session).

NOT `<world>/areas/<session>/<area_id>/` as T9 first proposed, and the reason is
measured: that layout repeats the 32-hex session id (`areas/<session>/<area>/<stage>/
<session>/`), and the first real area build on this machine failed with
`FileNotFoundError` on a staging file at 261 characters. The area id already hashes the
session id (§2.4 rule 3), so it is unique within the world without the extra segment.

The record lives beside the area rather than in `session.json`: `session.json` is
rewritten by the builder and the finisher, and an area whose id is retired by a
re-finish keeps its record with it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import re
import time
from pathlib import Path

from tower.world_builder.records import (
    STAGE_APPEARANCE,
    STAGE_STATE_FAILED,
    STAGE_STATE_OK,
    STAGE_STATE_RUNNING,
    STAGE_STATE_STOPPED,
    STAGE_STATE_UNAVAILABLE,
    STAGE_SURFACE,
    ScaleState,
)
from tower.world_builder.store import WorldStore

logger = logging.getLogger(__name__)

COMPONENTS_CONTRACT = "world_builder.components/2026-09-23"
COMPONENTS_FILENAME = "components.json"
AREAS_DIRNAME = "areas"
AREA_RECORD_FILENAME = "record.json"

STATE_PLACED = "placed"
STATE_UNPLACED = "unplaced"
SHOWN_ROOM = "room"
SHOWN_AREA = "area"
SHOWN_NONE = "none"
_STATES = (STATE_PLACED, STATE_UNPLACED)
_SHOWN = (SHOWN_ROOM, SHOWN_AREA, SHOWN_NONE)

# §2.1: 16 lower-hex, because the id is a path segment of the area routes.
_AREA_ID = re.compile(r"^[0-9a-f]{16}$")

# §2.4 rule 5: at most 8 spans.
MAX_SPANS = 8

SCOPE_ROOM = "room"
SCOPE_AREA = "area"

# THE FOUR AREA SENTENCES (§5.1, C1 E4). Stable identifiers: the phone compares them for
# equality, exactly as it compares FastAPI's `Not Found`, to tell a terminal answer from
# a transient one. Their text never changes without a contract change, and
# `tests/test_world_builder_area_routes.py` pins it.
NO_SUCH_AREA = "no such area in this session"           # terminal
NO_AREAS = "this session has no areas"                   # terminal
AREA_NOT_BUILT_YET = "this area has not been built yet"  # transient
AREA_COULD_NOT_BE_BUILT = "this area could not be built"  # terminal
AREA_NO_SPARSE_OR_DENSE = "an area has no sparse or dense rung"

# The photographic stages an area has. Never dense: an area is a surface and an
# appearance (§5.1, "an area is built as a surface and an appearance only").
AREA_STAGES = (STAGE_SURFACE, STAGE_APPEARANCE)


def is_area_id(value) -> bool:
    return isinstance(value, str) and bool(_AREA_ID.match(value))


# ---------------------------------------------------------------------------
# the record
# ---------------------------------------------------------------------------


def components_path(store: WorldStore, world_id: str, session_id: str) -> Path:
    """`<world>/solve/<session>/components.json` (contract T9)."""
    return store.world_dir(world_id) / "solve" / session_id / COMPONENTS_FILENAME


@dataclasses.dataclass(frozen=True)
class ComponentsRecord:
    """A validated components record. `entries` are in §2.4 rule 2's order and carry
    the gate's fields plus `keyframe_ids` (internal). `sha1` names the file's bytes, so
    an area build can say which record it was built for."""

    entries: tuple
    sha1: str

    def entry(self, component_id: str) -> dict | None:
        for e in self.entries:
            if e["id"] == component_id:
                return e
        return None

    def areas(self) -> list:
        return [e for e in self.entries if e["shown_as"] == SHOWN_AREA]


_WARNED: set = set()


def _warn_once(message: str) -> None:
    if message in _WARNED:
        logger.debug(message)
        return
    if len(_WARNED) < 64:
        _WARNED.add(message)
    logger.warning(message)


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite(value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _valid_spans(spans) -> list | None:
    if not isinstance(spans, list) or len(spans) > MAX_SPANS:
        return None
    out = []
    for span in spans:
        if (not isinstance(span, (list, tuple)) or len(span) != 2
                or not all(_finite(v) for v in span) or span[0] > span[1]):
            return None
        out.append([float(span[0]), float(span[1])])
    return out


def _validate_entry(raw) -> dict | None:
    if not isinstance(raw, dict):
        return None
    cid = raw.get("id")
    state = raw.get("state")
    shown = raw.get("shown_as")
    reasons = raw.get("reasons")
    keyframes = raw.get("keyframes")
    if not is_area_id(cid) or state not in _STATES or shown not in _SHOWN:
        return None
    if (state == STATE_PLACED) != (shown == SHOWN_ROOM):
        return None
    if not isinstance(reasons, list) or not all(isinstance(r, str) and r for r in reasons):
        return None
    if state == STATE_PLACED and reasons:
        return None
    if state == STATE_UNPLACED and not reasons:
        return None
    reason = raw.get("reason", reasons[0] if reasons else None)
    if reason != (reasons[0] if reasons else None):
        return None
    if not _is_int(keyframes) or keyframes < 0:
        return None
    spans = _valid_spans(raw.get("capture_spans_s"))
    if spans is None:
        return None
    kids = raw.get("keyframe_ids")
    if kids is not None and (not isinstance(kids, list)
                             or not all(isinstance(k, str) and k for k in kids)):
        return None
    return {"id": cid, "state": state, "reason": reason, "reasons": list(reasons),
            "shown_as": shown, "keyframes": int(keyframes), "capture_spans_s": spans,
            "keyframe_ids": list(kids) if kids is not None else None}


def _order_key(entry: dict):
    """§2.4 rule 2: placed first; then areas, then counted-only pieces, each by
    `keyframes` descending; ties by the earliest span start."""
    rank = {SHOWN_ROOM: 0, SHOWN_AREA: 1, SHOWN_NONE: 2}[entry["shown_as"]]
    spans = entry["capture_spans_s"]
    first = spans[0][0] if spans else math.inf
    return (rank, -entry["keyframes"], first, entry["id"])


def parse_components(payload, *, sha1: str = "") -> ComponentsRecord | None:
    """A `ComponentsRecord` from the record's JSON, or None when it is not the
    contract's shape. Pure; `read_components_record` is the reader."""
    raw = payload.get("components") if isinstance(payload, dict) else payload
    if not isinstance(raw, list) or not raw:
        # §2.4 rule 1: when computed the list is never empty.
        return None
    entries = []
    for item in raw:
        entry = _validate_entry(item)
        if entry is None:
            return None
        entries.append(entry)
    if sum(1 for e in entries if e["state"] == STATE_PLACED) != 1:
        return None
    if len({e["id"] for e in entries}) != len(entries):
        return None
    entries.sort(key=_order_key)
    return ComponentsRecord(entries=tuple(entries), sha1=sha1)


def _read_bytes_past_a_replace(path: Path, attempts: int = 5) -> bytes:
    """`path.read_bytes()`, retried across a writer's `os.replace` (WinError 5 on
    Windows while the replace lands), the store's `_read_json_past_a_replace`
    ladder for raw bytes -- the record's sha1 is taken over the bytes."""
    last = None
    for attempt in range(attempts):
        try:
            return path.read_bytes()
        except PermissionError as exc:
            last = exc
            time.sleep(0.001 * (2 ** attempt))
    raise last


def read_components_record(store: WorldStore, world_id: str,
                           session_id: str) -> ComponentsRecord | None:
    """The session's components record, or None: absent, unreadable, or not the
    contract's shape. Never raises -- this is on the listing's and the status
    channel's path, and `None` is the answer every older world already gives."""
    path = components_path(store, world_id, session_id)
    try:
        if not path.exists():
            return None
    except OSError:
        return None
    try:
        data = _read_bytes_past_a_replace(path)
        payload = json.loads(data.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 -- absent is the safe reading
        _warn_once(f"[Tower][WorldBuilder] components record {world_id}/{session_id} is "
                   f"unreadable ({type(exc).__name__}); the session reports "
                   "components: null")
        return None
    record = parse_components(payload, sha1=hashlib.sha1(data).hexdigest())
    if record is None:
        _warn_once(f"[Tower][WorldBuilder] components record {world_id}/{session_id} is "
                   "not the contract's shape (WORLD-BUILDER-COMPONENTS.md §2); the "
                   "session reports components: null")
    return record


# ---------------------------------------------------------------------------
# where an area lives, and the store view that builds and serves it
# ---------------------------------------------------------------------------


def area_root(store: WorldStore, world_id: str, session_id: str, area_id: str) -> Path:
    """`<world>/areas/<area_id>` (see the module docstring for why not per session).
    `area_id` must be 16 lower-hex (`is_area_id`) -- it is joined into a path -- and
    the session it belongs to is named by its record, not by the path."""
    del session_id  # the area's session is its record's; the id is unique in the world
    if not is_area_id(area_id):
        raise ValueError("not an area id")
    return store.world_dir(world_id) / AREAS_DIRNAME / area_id


def areas_dir(store: WorldStore, world_id: str) -> Path:
    """`<world>/areas`: every area of every session of the world."""
    return store.world_dir(world_id) / AREAS_DIRNAME


def session_area_dirs(store: WorldStore, world_id: str, session_id: str) -> list:
    """The area directories whose record names this session."""
    out = []
    root = areas_dir(store, world_id)
    if not root.is_dir():
        return out
    for entry in sorted(root.iterdir()):
        if not (entry.is_dir() and is_area_id(entry.name)):
            continue
        record = read_area_record(store, world_id, session_id, entry.name) or {}
        if record.get("session_id") == session_id:
            out.append(entry)
    return out


class AreaStore(WorldStore):
    """One area, seen as a world directory of its own session.

    The surface and appearance pipelines, the appearance routes' checks and the pages
    all find their artifacts at `store.world_dir(world) / <stage> / <session>`, their
    solution at `.../solve/<session>`, and the session's keyframes, images, redaction
    label and `world.json` through `session_dir` and `world_path`. This view answers
    the first two from the area's own directory and everything about the SESSION from
    the real world -- so an area is built from the session's own redacted keyframes,
    re-checked against the session's own label on every request (§5.3), and written
    nowhere but under its own directory.

    `read_world` answers the real world with its scale reset to unknown: an area
    inherits no scale from the room, so it claims none (§5.4, "Scale").
    """

    def __init__(self, base: WorldStore, world_id: str, session_id: str,
                 area_id: str) -> None:
        super().__init__(base.root)
        self._base = base
        self._area_world = world_id
        self._area_session = session_id
        self._area_id = area_id
        self._area_dir = area_root(base, world_id, session_id, area_id)

    @property
    def area_id(self) -> str:
        return self._area_id

    @property
    def area_dir(self) -> Path:
        return self._area_dir

    @property
    def base(self) -> WorldStore:
        return self._base

    def world_dir(self, world_id: str) -> Path:
        if world_id == self._area_world:
            return self._area_dir
        return self._base.world_dir(world_id)

    def world_path(self, world_id: str) -> Path:
        return self._base.world_path(world_id)

    def session_dir(self, world_id: str, session_id: str) -> Path:
        return self._base.session_dir(world_id, session_id)

    def derived_dir(self, world_id: str) -> Path:
        return self._base.derived_dir(world_id)

    def lock_path(self, world_id: str) -> Path:
        return self._base.lock_path(world_id)

    def list_session_ids(self, world_id: str) -> list[str]:
        return self._base.list_session_ids(world_id)

    def read_world(self, world_id: str):
        world = self._base.read_world(world_id)
        return dataclasses.replace(world, scale=ScaleState())


# ---------------------------------------------------------------------------
# the area's stage record
# ---------------------------------------------------------------------------


def area_record_path(store: WorldStore, world_id: str, session_id: str,
                     area_id: str) -> Path:
    return area_root(store, world_id, session_id, area_id) / AREA_RECORD_FILENAME


def read_area_record(store: WorldStore, world_id: str, session_id: str,
                     area_id: str) -> dict | None:
    path = area_record_path(store, world_id, session_id, area_id)
    try:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def write_area_record(store: WorldStore, world_id: str, session_id: str, area_id: str,
                      *, components_sha1: str, stage: str | None = None,
                      state: str | None = None, detail: str | None = None,
                      attempted: bool = True, **fields) -> dict:
    """Record one area stage's outcome (the room's `mark_stage`, for an area).

    A record written for a DIFFERENT components record (a re-solve or a re-finish
    since) is started afresh: its outcomes describe another area.
    """
    from tower.storage import write_json_atomic  # noqa: PLC0415

    now = time.time()
    record = read_area_record(store, world_id, session_id, area_id) or {}
    if record.get("components_sha1") != components_sha1:
        record = {}
    record.update({"area_id": area_id, "session_id": session_id,
                   "components_sha1": components_sha1, "updated_at": now})
    record.update(fields)
    stages = dict(record.get("stages") or {})
    if stage is not None:
        prior = stages.get(stage) or {}
        started = now if state == STAGE_STATE_RUNNING else prior.get("started_at", now)
        stages[stage] = {"state": state, "attempted": bool(attempted), "detail": detail,
                         "started_at": started, "updated_at": now}
    record["stages"] = stages
    write_json_atomic(area_record_path(store, world_id, session_id, area_id), record)
    return record


# ---------------------------------------------------------------------------
# the area's photographic word (§2.1 `photographic`, §3.4)
# ---------------------------------------------------------------------------


def area_photographic_state(store: WorldStore, world_id: str, session_id: str, session,
                            area_id: str, components_sha1: str) -> dict:
    """The area's own build, in WORLDS §2a's seven words. Record first, like the
    room's (`photographic.room_photographic_state`); liveness only where the record is
    ambiguous. Never raises: a probe that fails is `unobservable`.

    NO RECORD MEANS OWED, and it can, because nothing old can reach here: an area
    exists only in a components record, which only the new pipeline writes, and the
    area build writes its record BEFORE it starts. So an area with no record is one the
    finisher (`scripts/world_finish_pending.py`) has not reached yet -- exactly what
    `owed` promises (§7 rule 5: owed area work exists only for a session whose record
    names an area whose build is unsettled).
    """
    from tower.world_builder import photographic as P  # noqa: PLC0415

    if not P._appearance_is_expected(session):
        return {"state": P.PHOTOGRAPHIC_UNATTEMPTED, "stage": None,
                "detail": "this session has no completed global solve to build an area from"}
    try:
        record = read_area_record(store, world_id, session_id, area_id)
    except Exception as exc:  # noqa: BLE001
        return {"state": P.PHOTOGRAPHIC_UNOBSERVABLE, "stage": None,
                "detail": f"the area's record could not be read: {type(exc).__name__}"}
    stages = {}
    if (record and record.get("components_sha1") == components_sha1
            and record.get("session_id") in (None, session_id)):
        stages = record.get("stages") or {}
    if not stages:
        return {"state": P.PHOTOGRAPHIC_OWED, "stage": STAGE_SURFACE,
                "detail": "the area has not been built yet and no process is working on it"}

    def entry(stage):
        value = stages.get(stage)
        return value if isinstance(value, dict) else {}

    if entry(STAGE_APPEARANCE).get("state") == STAGE_STATE_OK:
        return {"state": P.PHOTOGRAPHIC_COMPLETE, "stage": STAGE_APPEARANCE,
                "detail": "the area's appearance stage finished"}
    for stage in AREA_STAGES:
        if entry(stage).get("state") == STAGE_STATE_FAILED:
            return {"state": P.PHOTOGRAPHIC_FAILED, "stage": stage,
                    "detail": entry(stage).get("detail")
                    or f"the area's {stage} stage is recorded failed"}
    for stage in AREA_STAGES:
        if entry(stage).get("state") in (STAGE_STATE_RUNNING, STAGE_STATE_STOPPED):
            # The room's rule (`photographic._liveness`), asked of the AREA's own
            # status files: the view puts `<area>/surface/<session>/status.json` where
            # the probe looks. This is what makes a running area `running` (C1 E13).
            try:
                view = AreaStore(store, world_id, session_id, area_id)
            except Exception as exc:  # noqa: BLE001
                return {"state": P.PHOTOGRAPHIC_UNOBSERVABLE, "stage": stage,
                        "detail": f"the area could not be probed: {type(exc).__name__}"}
            return P._liveness(view, world_id, session_id, stage)
    for stage in AREA_STAGES:
        e = entry(stage)
        if e.get("state") == STAGE_STATE_UNAVAILABLE and e.get("attempted"):
            return {"state": P.PHOTOGRAPHIC_FAILED, "stage": stage,
                    "detail": e.get("detail")
                    or f"the area's {stage} stage ran and could not produce it"}
    return {"state": P.PHOTOGRAPHIC_UNATTEMPTED, "stage": None,
            "detail": entry(STAGE_APPEARANCE).get("detail") or entry(STAGE_SURFACE).get("detail")
            or "no photographic stage was attempted for this area"}


def area_words(store: WorldStore, world_id: str, session_id: str, session,
               record: ComponentsRecord) -> dict:
    """`{area_id: word}` for every `shown_as: "area"` entry of the record."""
    from tower.world_builder import photographic as P  # noqa: PLC0415

    out = {}
    for e in record.areas():
        try:
            out[e["id"]] = area_photographic_state(store, world_id, session_id, session,
                                                   e["id"], record.sha1)
        except Exception as exc:  # noqa: BLE001 -- one area, not the row
            out[e["id"]] = {"state": P.PHOTOGRAPHIC_UNOBSERVABLE, "stage": None,
                            "detail": f"the area's state could not be computed: "
                                      f"{type(exc).__name__}"}
    return out


# Which area word the row carries when several are unsettled. `running` first, so a
# running area always makes `build_in_progress` true (C1 E13); then `unobservable`,
# which nobody could answer; then `owed`.
_AREA_PRECEDENCE = ("running", "unobservable", "owed")


def combine_photographic(room: dict | None, record: ComponentsRecord | None,
                         words: dict | None) -> dict | None:
    """The row's `photographic` block: the room AND its areas (§3.4, C1 E3).

    - the room's word, when the room is `running`, `owed` or `unobservable`;
    - otherwise the first unsettled area word (running, unobservable, owed), with the
      area's `stage` and a `detail` that says *an area of this walk* -- no name, no
      number;
    - otherwise the room's word. An area's `failed` never reaches the row.

    `scope` is present exactly when the session HAS a components record: `"area"` when
    the word is an area's, `"room"` otherwise. A session with `components: null` gets
    the room's block byte for byte as before (absent `scope` means `"room"`, §3.4).
    """
    from tower.world_builder.photographic import is_unsettled  # noqa: PLC0415

    if room is None or record is None:
        return room
    if is_unsettled(room.get("state")):
        return {**room, "scope": SCOPE_ROOM}
    unsettled = [(w, aid) for aid, w in (words or {}).items() if is_unsettled(w.get("state"))]
    if unsettled:
        order = {s: i for i, s in enumerate(_AREA_PRECEDENCE)}
        # The record's own order among equals, so the chosen area is stable.
        position = {e["id"]: i for i, e in enumerate(record.entries)}
        word, _aid = min(unsettled, key=lambda pair: (order.get(pair[0]["state"], 9),
                                                      position.get(pair[1], 0)))
        return {"state": word["state"], "stage": word.get("stage"),
                "detail": f"an area of this walk: {word.get('detail')}",
                "scope": SCOPE_AREA}
    return {**room, "scope": SCOPE_ROOM}


def with_areas(store: WorldStore, world_id: str, session_id: str, session,
               room: dict) -> dict:
    """`combine_photographic` for one session, reading its record. The helper both
    the listing row and the status channel's `lifecycle.photographic` go through."""
    record = read_components_record(store, world_id, session_id)
    if record is None:
        return room
    return combine_photographic(room, record,
                                area_words(store, world_id, session_id, session, record))


# ---------------------------------------------------------------------------
# what exists now: an area's drawability and its phone keyframes
# ---------------------------------------------------------------------------


def area_appearance_manifest(store: WorldStore, world_id: str, session_id: str,
                             area_id: str) -> dict | None:
    """The area's appearance manifest as built (no checks), or None."""
    try:
        from tower.world_builder import appearance_pipeline as AP  # noqa: PLC0415

        return AP.read_appearance_manifest(AreaStore(store, world_id, session_id, area_id),
                                           world_id, session_id)
    except Exception:  # noqa: BLE001
        return None


def area_appearance_servable(store: WorldStore, world_id: str, session_id: str,
                             area_id: str, world=None) -> tuple[dict | None, str | None]:
    """`(manifest, None)` when the area's appearance may be served right now, else
    `(None, reason)`. The same checks, in the same order, as the room's appearance
    routes (APPEARANCE §9): purged, readable, imagery source, and the redaction label
    and keyframe set re-checked against the SESSION's."""
    from tower.world_builder import appearance_pipeline as AP  # noqa: PLC0415
    from tower.world_builder import raw_imagery as RAWIMG  # noqa: PLC0415

    if world is None:
        world = store.read_world(world_id)
    if getattr(world, "images_purged", False):
        return None, "appearance imagery was purged"
    view = AreaStore(store, world_id, session_id, area_id)
    manifest = AP.read_appearance_manifest(view, world_id, session_id)
    if manifest is None:
        return None, "no appearance for this area"
    wanted = RAWIMG.imagery_source_from_env()
    if not AP.imagery_matches(manifest, wanted):
        return None, AP.imagery_mismatch_detail(manifest, wanted)
    if not AP.label_matches(view, world_id, session_id, manifest, imagery_source=wanted):
        return None, AP.STALE_LABEL_DETAIL
    return manifest, None


def area_surface_drawable(store: WorldStore, world_id: str, session_id: str,
                          area_id: str) -> bool:
    from tower.world_builder.store import surface_artifact_drawable  # noqa: PLC0415

    try:
        return surface_artifact_drawable(AreaStore(store, world_id, session_id, area_id),
                                         world_id, session_id)
    except Exception:  # noqa: BLE001
        return False


def area_has_geometry(store: WorldStore, world_id: str, session_id: str, area_id: str,
                      world=None) -> bool:
    """Would `GET /worlds/{w}/areas/{s}/{a}/render` answer 200 now (§2.1
    `has_geometry`)? The appearance rung when its routes would serve it, else a
    drawable surface -- the route's own ladder, by the same artifact checks."""
    try:
        manifest, _reason = area_appearance_servable(store, world_id, session_id, area_id,
                                                     world=world)
        if manifest is not None and any(k.get("tier") == "phone"
                                        for k in manifest.get("keyframes") or []):
            return True
    except Exception:  # noqa: BLE001
        logger.debug("[Tower][WorldBuilder] area appearance probe failed", exc_info=True)
    return area_surface_drawable(store, world_id, session_id, area_id)


def _phone_keyframes(manifest: dict | None) -> int | None:
    if not isinstance(manifest, dict):
        return None
    return sum(1 for k in manifest.get("keyframes") or [] if k.get("tier") == "phone")


# ---------------------------------------------------------------------------
# §2.1 on the wire
# ---------------------------------------------------------------------------


def wire_components(store: WorldStore, world_id: str, session_id: str, session, *,
                    record: ComponentsRecord | None,
                    room_has_geometry: bool,
                    room_keyframes_phone: int | None,
                    room_word: dict | None,
                    words: dict | None = None,
                    world=None) -> list | None:
    """The §2 array for one session, or None (`components: null`).

    Exactly the §2.1 fields, in §2.4 rule 2's order. `keyframe_ids` and anything else
    the record holds stay in the Tower.
    """
    if record is None:
        return None
    if words is None:
        words = area_words(store, world_id, session_id, session, record)
    if world is None:
        try:
            world = store.read_world(world_id)
        except Exception:  # noqa: BLE001
            world = None
    out = []
    for e in record.entries:
        item = {"id": e["id"], "state": e["state"], "reason": e["reason"],
                "reasons": list(e["reasons"]), "shown_as": e["shown_as"],
                "keyframes": e["keyframes"]}
        if e["shown_as"] == SHOWN_ROOM:
            item["keyframes_phone"] = room_keyframes_phone
            item["capture_spans_s"] = [list(s) for s in e["capture_spans_s"]]
            item["has_geometry"] = bool(room_has_geometry)
            item["photographic"] = ({k: room_word.get(k) for k in ("state", "stage", "detail")}
                                    if room_word else None)
        elif e["shown_as"] == SHOWN_AREA:
            manifest = area_appearance_manifest(store, world_id, session_id, e["id"])
            item["keyframes_phone"] = _phone_keyframes(manifest)
            item["capture_spans_s"] = [list(s) for s in e["capture_spans_s"]]
            item["has_geometry"] = (world is not None
                                    and area_has_geometry(store, world_id, session_id,
                                                          e["id"], world=world))
            word = words.get(e["id"])
            item["photographic"] = ({k: word.get(k) for k in ("state", "stage", "detail")}
                                    if word else None)
        else:
            item["keyframes_phone"] = None
            item["capture_spans_s"] = [list(s) for s in e["capture_spans_s"]]
            item["has_geometry"] = False
            item["photographic"] = None
        out.append(item)
    return out


def components_for_session(store: WorldStore, world_id: str, session_id: str, *,
                           session=None, room_has_geometry: bool | None = None,
                           room_keyframes_phone=None, room_word: dict | None = None,
                           world=None) -> list | None:
    """`wire_components` with every input it was not handed computed here. The render
    revision route's entry point (§3.2); None, cheaply, for every older session."""
    record = read_components_record(store, world_id, session_id)
    if record is None:
        return None
    if session is None:
        session = store.read_session(world_id, session_id)
    if room_has_geometry is None:
        from tower.world_builder.store import session_has_drawable_geometry  # noqa: PLC0415

        room_has_geometry = session_has_drawable_geometry(store, world_id, session_id)
    if room_keyframes_phone is None:
        try:
            from tower.world_builder import appearance_pipeline as AP  # noqa: PLC0415

            room_keyframes_phone = _phone_keyframes(
                AP.read_appearance_manifest(store, world_id, session_id))
        except Exception:  # noqa: BLE001
            room_keyframes_phone = None
    if room_word is None:
        from tower.world_builder.photographic import room_photographic_state  # noqa: PLC0415

        room_word = room_photographic_state(store, world_id, session_id, session)
    return wire_components(store, world_id, session_id, session, record=record,
                           room_has_geometry=room_has_geometry,
                           room_keyframes_phone=room_keyframes_phone,
                           room_word=room_word, world=world)
