"""The World Builder status producer: a READER, never a second pipeline.

The web process does not build worlds and must not start. Reconstruction
runs in its own process (`scripts/world_build_session.py`), which is the
decision `plan.md` 28 protects and the closeout report defends at length:
the frame path pays nothing for a rebuild precisely because the rebuild is
somewhere else. This module reads what that process has already persisted
and shapes it for the wire. It calls no engine method, holds no engine
object, and starts nothing.

Everything reported here is derived from four things on disk:

    LOCK              a pid file, held for the LIFETIME OF A SESSION
    events.jsonl      the append-only journal, dense event_id per session
    session.json      written at start and rewritten at stop -- see below
    derived/manifest  what the last build produced

The single most important fact about the second and third of those:

    session.json is written at start_session() with frames_observed=0 and
    keyframes_accepted=0, and is not rewritten until stop_session().

(engine.py start_session, stop_session.) So DURING a live session those
counts on disk are stale zeros. Reporting them would be the exact failure
`IOS-to-Tower.md` 1.8 warns about -- "nil and 0 are different claims and
are kept different all the way to the screen" -- with the added insult
that the zero looks like a measurement. The live keyframe count is
therefore counted from `keyframe_accepted` events instead, and
`frames_observed` is reported as UNAVAILABLE while a session is live,
because no event is written for an ordinary rejected frame (engine.py
observe: `_note_rejected` without an append for every reason except
malformed_frame). Tower genuinely does not know it yet, and says so.
"""

import json
import logging
import math
from pathlib import Path

from tower.logging_config import client_safe_reason
from tower.results.contracts import TIME_BASIS
from tower.storage import read_raw_jsonl
from tower.results.envelope import Snapshot, compute_revision
from tower.world_builder.records import FINAL_SOLVE_SOLVED, format_distance
from tower.world_builder.schema import (
    INTRINSICS_SOURCE_UNKNOWN,
    POSE_STATUS_ANCHOR,
    POSE_STATUS_SOLVED,
    SCHEMA_VERSION,
    SCALE_ESTIMATED,
    SCALE_MEASURED,
    SCALE_RELATIVE,
    SCALE_UNKNOWN,
)
from tower.world_builder.store import WorldStore, WorldStoreError

logger = logging.getLogger(__name__)

# Lifecycle, named for the evidence rather than for an intention. Tower
# cannot see a process's intent; it can see a lock, a journal and a
# manifest.
LIFECYCLE_RECEIVING = "receiving"
# NOT "finalizing", and the reason is narrower than an earlier version of
# this comment claimed.
#
# That version said the files are BYTE-IDENTICAL while build() runs. An
# adversarial review disproved it: build() rewrites edges.jsonl,
# session.json and world.json BEFORE the manifest lands, so the directory
# does change. What is true -- and is what matters -- is that those writes
# are indistinguishable from a build that made them and then DIED. The
# writer lock is already released (engine.stop_session releases it before
# the driver calls build), and no event is written, so there is no marker
# that says "a process is working right now".
#
# So Tower cannot observe that work is continuing, and a state named
# "finalizing" would assert exactly that.
#
# This name says only what is visible: capture stopped, and the stored
# geometry is not current with the keyframes. iOS may render its own
# .finalizing from it -- lifecycle.build_in_progress carries the caveat
# that makes that an informed choice rather than an inherited guess.
LIFECYCLE_STOPPED_UNBUILT = "stopped_unbuilt"
# Since 2026-09-06 the builder KEEPS its writer lock through the final
# solve and the final build, and writes a `finalization` block on the
# session record when it stops. So the comment above is now history for
# the live path: a lock held by a running process AFTER `session_stopped`
# is a process that is finishing, and that is what this state says. The
# old `stopped_unbuilt` remains for records written before the change.
LIFECYCLE_FINALIZING = "finalizing"
LIFECYCLE_READY = "ready"
# A session that did not end the way a walk ends. The lock names a dead
# process, or the record says `error`/`interrupted`, or finalization was
# left pending by a process that is gone. Named for what happened to the
# SESSION, not for what exists on disk: the geometry block beside it says
# whether a reconstruction is there, and on the 2026-09-06 walk it was.
LIFECYCLE_INTERRUPTED = "interrupted"
# Kept in the vocabulary for readers; nothing on disk maps to it any more
# -- every fact that used to be `failed` is a more specific `interrupted`.
LIFECYCLE_FAILED = "failed"
LIFECYCLE_IDLE = "idle"
LIFECYCLE_UNAVAILABLE = "unavailable"

# Why THIS world is the one on the wire. The unpinned default answers with
# a live world if any, else the newest world on disk -- and until
# 2026-09-06 nothing said which, so a phone opening World Builder with
# nothing live drew the newest saved world as if it were the live one.
SELECTION_PINNED = "pinned"          # the client named it
SELECTION_LIVE = "live"              # a running builder holds its lock, session open
SELECTION_FINALIZING = "finalizing"  # a running builder holds its lock, session stopped
SELECTION_LATEST = "latest"          # nothing is live; this is the most recently updated
SELECTION_NONE = "none"              # nothing to report at all

# Tracking. `limited` is deliberately NEVER emitted -- see _tracking_block.
TRACKING_GOOD = "good"
TRACKING_LOST = "lost"
TRACKING_UNKNOWN = "unknown"

CALIBRATION_UNKNOWN = "unknown"
CALIBRATION_UNCALIBRATED = "uncalibrated"
CALIBRATION_CALIBRATED = "calibrated"

# iOS's vocabulary for scale (IOS-to-Tower.md 1.5), mapped from Tower's.
# `unknown` maps to nothing on purpose: a figure that cannot be labelled
# with one of the three is not sent as a distance at all, which leaves
# iOS's rule -- "a figure that arrives unlabelled is simply not shown as a
# distance" -- with nothing to catch.
SCALE_SEMANTICS = {
    SCALE_RELATIVE: "relative",
    SCALE_ESTIMATED: "inferredMetric",
    SCALE_MEASURED: "measuredMetric",
}

# Fields whose value advances without anything having happened. Excluded
# from the change revision so a UI can tell new data from repeated data
# (IOS-to-Tower.md 1.2).
VOLATILE_PATHS = (
    "progress.mapping_seconds",
    "world_snapshot.mapping_seconds",
    # Not volatile but not CONTENT: two subscriptions -- one pinned to the
    # newest world, one unpinned -- describe the same bytes on disk and
    # must agree on the revision, or a client switching between them sees
    # a phantom change.
    "selection",
    # Self-referential rather than volatile: `world_snapshot.revision` IS
    # the revision, so it cannot be an input to computing it. Excluded so
    # the hash stays stable when the field is filled in afterwards.
    "world_snapshot.revision",
)

# The Tower's own name for what it builds. `handoff.md` 9.5 says iOS
# displays this VERBATIM and never matches it against a known set, so it
# is prose for a person, not an identifier.
GEOMETRY_REPRESENTATION = "sparse point cloud"

# --- the iOS projection ------------------------------------------------
#
# `handoff.md` 16 names the one contract shape that costs the phone
# nothing: "a self-contained, coarsely-updated world report whose fields
# map 1:1 onto WorldSnapshot, plus an explicit lifecycle state mapping
# onto WorldModelState". Everything else in this payload is Tower-native
# and carries the EVIDENCE for these values; this block is the part iOS
# decodes.
#
# It is a projection, not a second source of truth: it is computed from
# the same payload in the same pass, so the two cannot disagree, and a
# test asserts every projected value against the block it came from.
#
# Tower deliberately does the mapping. The alternative -- shipping only
# Tower's vocabulary and asking iOS to translate -- would put this table
# on the phone, where the knowledge is not, and where getting it wrong is
# an App Store release rather than a Tower restart.

# WorldModelState cases (handoff.md 8.2). `awaiting_first_update` is
# deliberately absent: it means "frames are going out and the Tower has
# said nothing yet", which is a fact about the PHONE's own situation. Only
# iOS can know it, and it reaches it by not having received a snapshot.
MODEL_STATE_UNSUPPORTED = "unsupported"
MODEL_STATE_IDLE = "idle"
MODEL_STATE_RECEIVING = "receiving"
MODEL_STATE_FINALIZING = "finalizing"
MODEL_STATE_FINALIZED = "finalized"
# New at `world_builder.status/2026-09-06`, and the reason the identifier
# moved: a session that ended abnormally is neither `failed` (which the
# phone drew with no world at all) nor `finalized` (which would present a
# half-built walk as a finished one). The snapshot and the geometry travel
# with it, so the phone can show what exists and say what happened.
MODEL_STATE_INTERRUPTED = "interrupted"
MODEL_STATE_FAILED = "failed"

_MODEL_STATE_BY_LIFECYCLE = {
    LIFECYCLE_RECEIVING: MODEL_STATE_RECEIVING,
    # A live process is finishing; `lifecycle.build_in_progress` is True on
    # the evidence of the lock.
    LIFECYCLE_FINALIZING: MODEL_STATE_FINALIZING,
    # "capture ended, Tower still working; figures may change" is exactly
    # what `stopped_unbuilt` means -- the stored figures are not the final
    # figures. On a pre-finalization record Tower still cannot see whether
    # a build is RUNNING, and lifecycle.build_in_progress carries that
    # caveat unchanged.
    LIFECYCLE_STOPPED_UNBUILT: MODEL_STATE_FINALIZING,
    LIFECYCLE_READY: MODEL_STATE_FINALIZED,
    LIFECYCLE_INTERRUPTED: MODEL_STATE_INTERRUPTED,
    LIFECYCLE_FAILED: MODEL_STATE_FAILED,
    LIFECYCLE_IDLE: MODEL_STATE_IDLE,
}

# WorldTrackingQuality (handoff.md 8.3). Tower's `unknown` is iOS's
# `unavailable`; `limited` is never produced -- see _tracking_block.
_IOS_TRACKING = {
    TRACKING_GOOD: "good",
    TRACKING_LOST: "lost",
    TRACKING_UNKNOWN: "unavailable",
}

# WorldPersistenceState. Tower always persists, so `session` (meaning
# "this session only, nothing stored") is unreachable.
IOS_SCALE_UNKNOWN = "unknown"


class _FileCache:
    """Parse a file only when it has actually changed.

    A measured necessity, not an optimisation. `WorldStore.read_events`
    reads and JSON-parses the ENTIRE journal on every call -- the
    `after_event_id` cursor filters *after* the full read (store.py), so a
    cursor buys nothing. Measured on this host:

        100 events        0.27 ms
        1,000 events      2.35 ms
        10,000 events    26.6 ms
        50,000 events   209 ms

    against a measured frame reply of 1.98 ms average and 15.25 ms worst
    ever observed. A poll loop doing that read twice a second would spend
    more time parsing a journal than the Tower spends answering frames,
    and `asyncio.to_thread` does not save it: the work is `json.loads`,
    which holds the GIL, so offloading turns one 35 ms stall into many
    5 ms ones.

    A `stat()` costs **0.0135 ms** -- roughly 2,000x less at 10k events.
    And because every file here is either append-only or replaced whole,
    (size, mtime_ns) is a sound fingerprint: an append always grows the
    file, and an atomic replace always changes both.

    This also shrinks a genuine WINDOWS hazard. An open read handle in
    this process makes `Path.replace()` fail with WinError 5 in the
    *builder* process -- so a reader that opens files it did not need to
    can break the writer it is only supposed to be watching. Not opening
    them is the strongest available mitigation.
    """

    # Bounded, even though a remote client cannot drive it: `resolve`
    # refuses a world id that is not on disk before anything is cached, so
    # growth follows the OPERATOR's data, not a subscriber's requests.
    # Capped anyway. An unbounded-in-principle cache is a latent defect
    # whether or not today's callers can reach it, and the recovery here
    # is free -- every entry is a pure function of a file that is still
    # there, so dropping the lot costs one re-read.
    MAX_ENTRIES = 64

    __slots__ = ("_entries",)

    def __init__(self) -> None:
        self._entries: dict = {}

    def read(self, path, reader):
        try:
            stat = path.stat()
            fingerprint = (stat.st_size, stat.st_mtime_ns)
        except OSError:
            # Absent or unreadable. Do not cache: a file that appears
            # later must be picked up on the next poll.
            self._entries.pop(str(path), None)
            return reader()
        key = str(path)
        cached = self._entries.get(key)
        if cached is not None and cached[0] == fingerprint:
            return cached[1]
        value = reader()
        if len(self._entries) >= self.MAX_ENTRIES:
            self._entries.clear()
        self._entries[key] = (fingerprint, value)
        return value

    def fingerprint(self, path):
        try:
            stat = path.stat()
        except OSError:
            return None
        return (stat.st_size, stat.st_mtime_ns)


class WorldBuilderStatusProducer:
    """Builds one status snapshot per call. Holds only a small cache."""

    def __init__(self, world_root, clock) -> None:
        self._root = Path(world_root)
        self._clock = clock
        self._files = _FileCache()
        # Path length needs the full poses file, which the manifest does
        # not summarise. Reading it on every poll would be the one
        # genuinely unbounded read in this module, so it is computed once
        # per geometry revision and remembered. One entry per target,
        # replaced rather than accumulated -- see _path_length.
        self._path_length_cache: dict[str, tuple[str, dict | None]] = {}

    # -- target selection ---------------------------------------------

    def resolve(self, world_id: str | None, session_id: str | None):
        """Pick which world and session to report on.

        Returns `(world_id, session_id, problem)`; `resolve_with_selection`
        adds WHY that world was picked, which the payload now carries.

        An explicit world_id is iOS's inspection mode
        (`WorldInspectionMode.inspecting(worldID:)`, 1.7), where "there is
        no capture to start, and a counter that moved would be a bug".
        With none given, a live session is preferred over the most
        recently updated world, because a client that did not name one is
        asking about now.
        """
        chosen, session, problem, _ = self.resolve_with_selection(world_id, session_id)
        return chosen, session, problem

    def resolve_with_selection(self, world_id: str | None, session_id: str | None):
        """`resolve`, plus the `selection` block for the payload."""
        store = WorldStore(self._root)
        try:
            world_ids = store.list_world_ids()
        except OSError:
            return None, None, "world root is not readable", None
        if not world_ids:
            return None, None, "no worlds exist under this Tower's world root", None

        if world_id is not None:
            if world_id not in world_ids:
                return None, None, f"no world with id {world_id!r}", None
            chosen = world_id
            mode, reason = SELECTION_PINNED, "the client named this world"
        else:
            chosen, mode, reason = self._most_relevant(store, world_ids)
            if chosen is None:
                return None, None, "no world could be read", None

        if session_id is not None:
            if session_id not in store.list_session_ids(chosen):
                return (
                    None,
                    None,
                    f"world {chosen!r} has no session with id {session_id!r}",
                    None,
                )
            resolved_session = session_id
        else:
            sessions = store.list_session_ids(chosen)
            resolved_session = (
                self._latest_session(store, chosen, sessions) if sessions else None
            )
        selection = {
            "mode": mode,
            "world_id": chosen,
            "session_id": resolved_session,
            "reason": reason,
        }
        return chosen, resolved_session, None, selection

    def _most_relevant(self, store, world_ids):
        """A LIVE world if one exists, else the most recently updated.

        Returns `(world_id, selection_mode, reason)`.

        "Live" means a lock held by a process that is still running. An
        earlier version accepted the mere existence of a lock file, so one
        leftover lock from a crashed builder permanently hijacked every
        default subscription -- an adversarial review demonstrated a
        stale-locked world outranking a newer, cleanly stopped one.

        A live lock on a STOPPED session is a builder finalizing, and it
        still outranks every saved world: it is what "now" looks like in
        the two minutes after Stop. The selection names it `finalizing`
        so a client can say so.
        """
        live = [
            wid
            for wid in world_ids
            if (holder := store.lock_holder(wid)) is not None
            and holder.get("alive")
        ]
        candidates = live or world_ids
        best, best_at = None, -math.inf
        for wid in candidates:
            try:
                world = store.read_world(wid)
            except (WorldStoreError, KeyError, OSError):
                continue
            if world.updated_at > best_at:
                best, best_at = wid, world.updated_at
        if best is None:
            return None, SELECTION_NONE, "no world could be read"
        if not live:
            return (
                best,
                SELECTION_LATEST,
                "nothing is live; this is the most recently updated world",
            )
        if self._newest_session_is_stopped(store, best):
            return (
                best,
                SELECTION_FINALIZING,
                "a live builder holds this world's writer lock and its session has stopped",
            )
        return best, SELECTION_LIVE, "a live builder holds this world's writer lock"

    def _newest_session_is_stopped(self, store, world_id) -> bool:
        sessions = store.list_session_ids(world_id)
        if not sessions:
            return False
        latest = self._latest_session(store, world_id, sessions)
        summary = self._files.read(
            store.events_path(world_id, latest),
            lambda: _summarise_events(*read_raw_jsonl(store.events_path(world_id, latest))),
        )
        return bool(summary["stopped"])

    def _latest_session(self, store, world_id, session_ids):
        best, best_at = session_ids[0], -math.inf
        for sid in session_ids:
            try:
                session = store.read_session(world_id, sid)
            except (WorldStoreError, KeyError, OSError):
                continue
            if session.started_at > best_at:
                best, best_at = sid, session.started_at
        return best

    # -- the snapshot --------------------------------------------------

    def snapshot(self, world_id: str | None, session_id: str | None) -> Snapshot:
        """One complete status payload. Never partial, never a delta."""
        resolved_world, resolved_session, problem, selection = (
            self.resolve_with_selection(world_id, session_id)
        )
        if problem is not None:
            return self._unavailable(problem)
        try:
            return self._snapshot(resolved_world, resolved_session, selection)
        except (WorldStoreError, KeyError, ValueError, OSError) as exc:
            # A world this build cannot read is a real answer, not a
            # crash. Refusing to interpret an unknown schema is the store's
            # documented behaviour and it must survive to the wire rather
            # than becoming a dropped subscription.
            logger.warning("result channel: world %s unreadable: %s", world_id, exc)
            return self._unavailable(
                f"world could not be read: {client_safe_reason(exc)}"
            )

    def _unavailable(self, reason: str, *, supported: bool = True) -> Snapshot:
        """Nothing to report, and whether that is a Tower limitation.

        `supported=False` is "this Tower cannot serve World Builder at all"
        -- no world root configured. That maps to iOS's `.unsupported`,
        which tells a person the Tower cannot do this. Everything else --
        no worlds yet, an unreadable world -- maps to `.idle`, which does
        not.
        """
        payload = {
            "world": None,
            "session": None,
            "lifecycle": {
                "state": LIFECYCLE_UNAVAILABLE,
                "evidence": "nothing to read",
                "reason": reason,
                **_BUILD_UNOBSERVABLE,
            },
            "progress": None,
            "tracking": None,
            "calibration": None,
            "scale": None,
            "geometry": None,
            "trajectory": None,
            "persistence": None,
            "artifacts": None,
            "time_basis": TIME_BASIS,
        }
        payload["model_state"] = (
            MODEL_STATE_IDLE if supported else MODEL_STATE_UNSUPPORTED
        )
        payload["model_state_reason"] = reason
        payload["world_snapshot"] = None
        payload["selection"] = {
            "mode": SELECTION_NONE, "world_id": None, "session_id": None, "reason": reason,
        }
        return Snapshot(
            payload=payload,
            revision=compute_revision(payload, VOLATILE_PATHS),
            volatile_fields=VOLATILE_PATHS,
        )

    def _snapshot(self, world_id: str, session_id: str | None, selection=None) -> Snapshot:
        store = WorldStore(self._root)
        world = store.read_world(world_id)

        if session_id is None:
            payload = self._payload_no_session(store, world)
        else:
            payload = self._payload(store, world, session_id)
        payload["selection"] = selection or {
            "mode": SELECTION_PINNED, "world_id": world_id, "session_id": session_id,
            "reason": "the client named this world",
        }

        _attach_ios_projection(payload)

        revision = compute_revision(payload, VOLATILE_PATHS)
        if payload.get("world_snapshot") is not None:
            # iOS holds the revision INSIDE the snapshot (handoff.md 8.3),
            # so it survives being handed around as one value. It is the
            # same string the envelope carries.
            payload["world_snapshot"]["revision"] = revision
        return Snapshot(
            payload=payload,
            revision=revision,
            volatile_fields=VOLATILE_PATHS,
        )

    def _payload_no_session(self, store, world) -> dict:
        return {
            "world": _world_block(world),
            "session": None,
            "lifecycle": {
                "state": LIFECYCLE_IDLE,
                "evidence": "the world exists and has no sessions",
                "reason": None,
                **_BUILD_UNOBSERVABLE,
            },
            "progress": None,
            "tracking": None,
            "calibration": None,
            "scale": _scale_block(world, attributable=False),
            "geometry": _geometry_unavailable("this world has no sessions"),
            "trajectory": _trajectory_unavailable("this world has no sessions"),
            "persistence": _persistence_block(world),
            "artifacts": _artifacts_block(store, world.world_id, None, world),
            "time_basis": TIME_BASIS,
        }

    def _payload(self, store, world, session_id: str) -> dict:
        session = store.read_session(world.world_id, session_id)
        # A SUMMARY, not the parsed journal. Two reasons, both measured.
        #
        # Memory: caching the parsed list would hold every event dict for
        # as long as anyone is subscribed -- tens of megabytes for a long
        # session, in a cache whose whole purpose is to be cheap.
        #
        # Time: stat-gating stops the journal being re-PARSED, but the
        # blocks below scan it, and a scan is O(events) on every poll.
        # Measured at 50,000 events: 9.26 ms per snapshot when the parsed
        # list was cached and re-scanned, 0.79 ms when the summary is
        # cached instead. The parse was never the only cost.
        events = self._files.read(
            store.events_path(world.world_id, session_id),
            lambda: _summarise_events(
                *read_raw_jsonl(store.events_path(world.world_id, session_id))
            ),
        )
        holder = store.lock_holder(world.world_id)
        manifest = self._files.read(
            store.derived_manifest_path(world.world_id),
            lambda: _read_manifest(store, world.world_id),
        )
        keyframes_current = self._files.read(
            store.keyframes_path(world.world_id, session_id),
            lambda: _keyframes_digest(store, world.world_id, session_id),
        )

        if manifest is not None and manifest.get("session_id") != session_id:
            # THE session check, and the only one. A world with two built
            # sessions has ONE manifest, describing whichever built last.
            # Attributing it to the other session would report one
            # session's geometry as another's -- a confident wrong answer,
            # and the reason _read_manifest deliberately does not filter:
            # its result is cached per FILE, and that file is shared.
            manifest = None

        stopped = events['stopped']
        geometry_current = (
            manifest is not None
            and keyframes_current is not None
            and manifest.get("input_digest") == keyframes_current
        )

        lifecycle = _lifecycle(
            holder=holder,
            stopped=stopped,
            session=session,
            geometry_current=geometry_current,
            has_manifest=manifest is not None,
        )
        # Which counts are trustworthy is decided by whether the session
        # was ever STOPPED -- not by whether it is currently `receiving`.
        #
        # session.json holds the zeros written at start_session until
        # stop_session rewrites it, and `failed` and `idle` are both
        # states in which that never happened. An earlier version keyed
        # this on `receiving`, so a builder that crashed mid-session
        # reported `keyframes_accepted: 0, provenance: "measured"` beside
        # `tracking: good, evidence: the most recent event was
        # keyframe_accepted` -- the same payload denying and confirming
        # the same fact. Found by adversarial review.
        counts_are_final = stopped and session.ended_at is not None
        progress = _progress_block(
            session, events, counts_are_final, self._clock()
        )
        keyframes_now = progress["keyframes_accepted"]

        return {
            "world": _world_block(world),
            "session": _session_block(session),
            "lifecycle": lifecycle,
            "progress": progress,
            "tracking": _tracking_block(events),
            "calibration": _calibration_block(session),
            # Scale is attributed to THIS session or not at all. It
            # lives on the World record and is written by build(), so a
            # session that was never built would otherwise inherit a
            # `relative` scale earned by a different session -- an
            # adversarial review found exactly that, and the same argument
            # _calibration_block makes about a world-level calibration
            # state applies here unchanged.
            "scale": _scale_block(world, attributable=manifest is not None),
            "geometry": _geometry_block(manifest, geometry_current, keyframes_now),
            "trajectory": self._trajectory_block(
                store, world, session_id, manifest, geometry_current,
                keyframes_now, events,
            ),
            "persistence": _persistence_block(world),
            "artifacts": _artifacts_block(
                store, world.world_id, session_id, world, session
            ),
            "time_basis": TIME_BASIS,
        }

    def _trajectory_block(
        self, store, world, session_id, manifest, current, keyframes_now,
        events,
    ) -> dict:
        # Same reasoning as _geometry_block: a trajectory over the first N
        # keyframes is a correct answer to an older question, not a wrong
        # answer, and hiding it makes a live session look idle.
        if manifest is None:
            return _trajectory_unavailable(
                "no build has run for this session, so no poses exist"
            )
        revision = compute_revision(
            {
                "digest": manifest.get("input_digest"),
                "built_at": manifest.get("built_at"),
                "solved": manifest.get("poses_solved"),
                "refused": manifest.get("poses_refused"),
                "segments": manifest.get("segments"),
            }
        )
        return {
            "available": True,
            "current": current,
            "built_from_keyframes": manifest.get("keyframes"),
            "keyframes_now": keyframes_now,
            "stale_reason": (
                None
                if current
                else (
                    "keyframes have been accepted since this build ran; this "
                    "path covers built_from_keyframes of them"
                )
            ),
            # Poses that actually carry a position, which is neither
            # poses_solved nor the keyframe count. See _pose_count: the
            # first keyframe of a segment that resolved is a real point
            # on the path, and the only keyframe of a segment that
            # resolved nothing is not.
            "pose_count": _pose_count(manifest),
            "poses_solved": manifest.get("poses_solved"),
            "poses_refused": manifest.get("poses_refused"),
            # Reported beside the count rather than folded into it. An
            # uncalibrated walk should read as "36 segment origins, no
            # trajectory" -- a precise description -- instead of either
            # a fabricated pose count or a silent absence.
            "poses_anchor": manifest.get("poses_anchor"),
            "keyframes": manifest.get("keyframes"),
            "segments": manifest.get("segments"),
            # Beside `segments`, and never derivable from it. See
            # `_events_summary`: 122 segments on the 2026-09-09 walk were
            # 64 tracking losses plus 57 solve-chain breaks plus the one
            # the session started with.
            "tracking_restarts": events.get("tracking_restarts"),
            "chain_breaks": events.get("chain_breaks"),
            "path_length": self._path_length(
                store, world, session_id, manifest, revision
            ),
            "revision": revision,
            "provenance": "inferred",
            "confidence": None,
            "unavailable_reason": None,
        }

    def _path_length(self, store, world, session_id, manifest, revision):
        """Total distance along the camera path, or an honest refusal.

        Refused whenever the session has more than one segment. A segment
        break means tracking was lost and the poses either side are NOT in
        a common coordinate frame (records.py Keyframe.segment_index), so
        adding distances across the break sums numbers that share neither
        a unit nor an origin. The result would be a plausible number that
        means nothing -- worse than no number, because a UI cannot tell.

        Cached per geometry revision: this is the only read here that
        touches the full poses file, and a poll loop must not repeat it
        for an unchanged build.
        """
        key = f"{world.world_id}:{session_id}"
        cached = self._path_length_cache.get(key)
        if cached is not None and cached[0] == revision:
            return cached[1]

        value = self._compute_path_length(store, world, session_id, manifest)
        # Replaced, never appended: one entry per (world, session), so the
        # cache is bounded by the number of distinct targets a subscriber
        # names, not by session length or poll count. Capped for the same
        # reason as _FileCache -- and recomputing is one file read.
        if len(self._path_length_cache) >= _FileCache.MAX_ENTRIES:
            self._path_length_cache.clear()
        self._path_length_cache[key] = (revision, value)
        return value

    def _compute_path_length(self, store, world, session_id, manifest):
        refused = manifest.get("poses_refused")
        if refused is None or refused > 0:
            # A refused pose is a HOLE in the path, not a shorter path.
            # Summing across it draws a straight line between the two
            # keyframes either side of the gap and calls that distance
            # walked -- and the wearer may have walked a loop through it.
            # A length with holes in it is not a length.
            return {
                "available": False,
                # Interpolating `refused` here used to produce "None of
                # this session's poses were refused, so the path has
                # gaps" -- a sentence that states the opposite of its own
                # conclusion -- in the branch that fires BECAUSE the
                # figure is missing. Prose shown to a person must not be
                # assembled from a value the branch exists to handle.
                "reason": (
                    "this session's build did not record how many poses it "
                    "refused, so gaps in the path cannot be ruled out"
                    if refused is None
                    else (
                        f"{refused} of this session's poses were refused, so "
                        "the path has gaps; a total would draw straight lines "
                        "across them and count the result as distance "
                        "travelled"
                    )
                ),
            }
        segments = manifest.get("segments")
        if segments is None or segments > 1:
            return {
                "available": False,
                "reason": (
                    "this session's build did not record its segment count, "
                    "so a common coordinate frame cannot be assumed"
                    if segments is None
                    else (
                        # NOT "tracking was lost between them", which this
                        # string said until 2026-09-09. A segment boundary is
                        # opened by a tracking loss OR by the solver failing
                        # to extend its chain while tracking is healthy, and
                        # on that walk 57 of 121 boundaries were the second.
                        # The refusal does not depend on which: either way
                        # the poses share no frame.
                        f"this session is in {segments} segments whose poses "
                        "share no coordinate frame, so a total length would "
                        "sum incomparable distances"
                    )
                ),
            }
        semantics = SCALE_SEMANTICS.get(world.scale.state)
        if semantics is None:
            return {
                "available": False,
                "reason": (
                    "this world has no scale state, so a distance figure "
                    "could not be labelled and would not be renderable"
                ),
            }
        try:
            derived = store.read_derived(world.world_id, session_id)
        except (WorldStoreError, KeyError, ValueError, OSError):
            derived = None
        if derived is None:
            return {"available": False, "reason": "the derived poses are unreadable"}

        total = 0.0
        previous = None
        for row in derived["poses"]:
            # Status AND translation, matching inspect.trajectory's own
            # test. A rotation_only row carries a rotation with a null
            # translation; anything else is not a position on the path.
            if row.get("status") not in (POSE_STATUS_SOLVED, POSE_STATUS_ANCHOR):
                return {
                    "available": False,
                    "reason": (
                        f"a pose has status {row.get('status')!r}, so the path "
                        "is not continuous"
                    ),
                }
            translation = row.get("translation")
            if translation is None:
                return {
                    "available": False,
                    "reason": "a pose carries no translation, so the path has a gap",
                }
            if previous is not None:
                total += math.dist(previous, translation)
            previous = translation
        if previous is None:
            return {"available": False, "reason": "no pose was solved"}

        return {
            "available": True,
            "value": total,
            "unit": "world units",
            "scale_semantics": semantics,
            "display": format_distance(total, world.scale),
            "provenance": "inferred",
        }


# -- blocks -------------------------------------------------------------


def _world_block(world) -> dict:
    return {
        "world_id": world.world_id,
        # None, never a derived name. IOS-to-Tower.md 1.2: "If the Tower
        # does not name worlds, iOS shows no name rather than deriving
        # one."
        "display_name": world.display_name,
        "schema_version": world.schema_version,
        "created_at": world.created_at,
        "updated_at": world.updated_at,
    }


def _session_block(session) -> dict:
    return {
        "session_id": session.session_id,
        "started_at": session.started_at,
        "ended_at": session.ended_at,
        "end_reason": session.end_reason,
        "frame_source": session.frame_source,
        "capture_id": session.capture_id,
        # Stated on the record itself. This is the one module that retains
        # raw imagery and the posture travels with the data.
        "retains_raw_imagery": session.retains_raw_imagery,
    }


# Attached to every stopped-and-not-current lifecycle. `null` is the
# whole point: this is not False. Tower does not know whether a build is
# running, and False would be a claim that none is.
_BUILD_UNOBSERVABLE_REASON = (
    "the writer lock is released before build() is called, and build() "
    "emits no event and writes nothing until it finishes, so a build in "
    "progress is indistinguishable on disk from one that never started "
    "and from one that crashed"
)
_BUILD_UNOBSERVABLE = {
    "build_in_progress": None,
    "build_in_progress_unavailable_reason": _BUILD_UNOBSERVABLE_REASON,
}


def _lifecycle(*, holder, stopped, session, geometry_current, has_manifest) -> dict:
    """What the Tower can SEE about whether a world is being built.

    This is `IOS-to-Tower.md` 1.1's central ask -- "a start/stop/failed
    signal **distinct from 'frames are arriving'**" -- and the writer lock
    answers it exactly, because it is held for the lifetime of a mapping
    session and by nothing else. Since 2026-09-06 the live builder keeps
    it through finalization too (`engine.stop_session(hold_lock=True)`),
    and writes a `finalization` block on the record, so five states are
    now distinguishable on disk:

        lock alive, not stopped                 -> receiving
        lock alive, stopped                     -> finalizing
        lock dead                               -> interrupted
        stopped by error/interrupted, or a
          finalization left pending/interrupted -> interrupted
        stopped, finalization complete          -> ready
        stopped, no finalization record (older
          builder), geometry current / behind  -> ready / stopped_unbuilt

    `interrupted` is deliberately NOT `failed`. On the 09-06 walk the
    process died mid-walk and left 463 keyframes of geometry; the state
    describes the session, and the geometry block beside it describes
    what exists.
    """
    finalization = session.finalization
    alive = holder is not None and holder["alive"]
    lock_dead = holder is not None and not holder["alive"]

    if alive and not stopped:
        return {
            "state": LIFECYCLE_RECEIVING,
            "evidence": (
                f"a live process (pid {holder['pid']}) holds the writer lock"
            ),
            "reason": None,
            "build_in_progress": False,
            "build_in_progress_unavailable_reason": None,
            "finalization": finalization,
        }
    if alive and stopped:
        return {
            "state": LIFECYCLE_FINALIZING,
            "evidence": (
                f"a live process (pid {holder['pid']}) holds the writer lock and "
                "session_stopped was written"
            ),
            "reason": (
                "the builder is finishing this world: the final solve and the "
                "final build run after the session stops, and the lock is "
                "released when they are done"
            ),
            # True on the evidence of the lock: the process that finishes a
            # world is the process holding it, and it is alive.
            "build_in_progress": True,
            "build_in_progress_unavailable_reason": None,
            "finalization": finalization,
        }
    if lock_dead and not stopped:
        return {
            "state": LIFECYCLE_INTERRUPTED,
            "evidence": (
                "a writer lock exists but names no readable process id"
                if holder.get("unreadable")
                else (
                    f"the writer lock is held by pid {holder['pid']}, which is "
                    "no longer running"
                )
            ),
            "reason": (
                "the process building this world exited without stopping its "
                "session; its keyframes are persisted but the session was "
                "never closed"
            ),
            "build_in_progress": False,
            "build_in_progress_unavailable_reason": None,
            "finalization": finalization,
        }
    if lock_dead and stopped:
        return {
            "state": LIFECYCLE_INTERRUPTED,
            "evidence": (
                f"the writer lock is held by pid {holder['pid']}, which is no "
                "longer running, and session_stopped was written"
            ),
            "reason": (
                "the process finalizing this world exited before it finished; "
                "the geometry stored is the last build it completed"
            ),
            "build_in_progress": False,
            "build_in_progress_unavailable_reason": None,
            "finalization": finalization,
        }
    # A COMPLETED FINALIZATION OUTRANKS HOW THE CAPTURE ENDED.
    #
    # `end_reason` describes the CAPTURE; `finalization` describes the
    # WORLD, and they are different questions. This block used to answer
    # both with the first, so a session whose capture ended badly could
    # never be reported as finished however it was repaired -- which made
    # `scripts/world_finalize.py` unable to deliver what it exists for.
    # Measured on the recovered 2026-09-09 artifact: finalization
    # `{state: complete, final_solve: solved}`, 88 of 122 segments
    # registered, and this function still said `interrupted`.
    #
    # `end_reason` is NOT rewritten to achieve this -- that walk really did
    # end in an error and the record should keep saying so. It is carried
    # into the reason string instead, so the phone can say a world was
    # finished after an interrupted capture rather than having to choose
    # which half of the truth to show.
    if (
        session.end_reason in ("error", "interrupted")
        and (finalization or {}).get("state") == "complete"
        and (finalization or {}).get("final_solve") == FINAL_SOLVE_SOLVED
    ):
        return {
            "state": LIFECYCLE_READY,
            "evidence": (
                f"the capture ended with end_reason={session.end_reason!r}, and "
                "the finalization record is complete with a solved final solve"
            ),
            "reason": (
                f"this walk's capture ended with {session.end_reason!r}; the "
                "world was finished afterwards and is complete"
            ),
            "build_in_progress": False,
            "build_in_progress_unavailable_reason": None,
            "finalization": finalization,
        }
    if session.end_reason in ("error", "interrupted"):
        detail = (finalization or {}).get("detail")
        return {
            "state": LIFECYCLE_INTERRUPTED,
            "evidence": f"the session recorded end_reason={session.end_reason!r}",
            "reason": (
                f"the mapping session ended with {session.end_reason!r}"
                + (f": {detail}" if detail else "")
            ),
            "build_in_progress": False,
            "build_in_progress_unavailable_reason": None,
            "finalization": finalization,
        }
    if not stopped:
        return {
            "state": LIFECYCLE_IDLE,
            "evidence": (
                "no writer lock is held and no session_stopped event was "
                "written"
            ),
            "reason": None,
            **_BUILD_UNOBSERVABLE,
            "finalization": finalization,
        }
    if finalization is not None and finalization.get("state") != "complete":
        return {
            "state": LIFECYCLE_INTERRUPTED,
            "evidence": (
                f"session_stopped was written and the finalization record is "
                f"{finalization.get('state')!r} with no process holding the lock"
            ),
            "reason": (
                "finalization did not complete; the geometry stored is the last "
                "build that finished"
                + (f": {finalization.get('detail')}" if finalization.get("detail") else "")
            ),
            "build_in_progress": False,
            "build_in_progress_unavailable_reason": None,
            "finalization": finalization,
        }
    if finalization is not None:
        # A finished record from a builder that keeps the lock through
        # finalization: the lock is gone because it was RELEASED, and the
        # record says the build completed.
        return {
            "state": LIFECYCLE_READY,
            "evidence": (
                "session_stopped was written, the finalization record is complete "
                "and the lock was released"
            ),
            "reason": None,
            "build_in_progress": False,
            "build_in_progress_unavailable_reason": None,
            "finalization": finalization,
        }
    if not has_manifest:
        return {
            "state": LIFECYCLE_STOPPED_UNBUILT,
            "evidence": "capture ended and no build output exists for this session",
            "reason": "no geometry has been built for this session yet",
            **_BUILD_UNOBSERVABLE,
            "finalization": None,
        }
    if not geometry_current:
        return {
            "state": LIFECYCLE_STOPPED_UNBUILT,
            "evidence": (
                "capture ended and the stored geometry is older than the "
                "keyframes"
            ),
            "reason": (
                "a rebuild is outstanding; the figures stored are not the "
                "figures these keyframes would produce"
            ),
            **_BUILD_UNOBSERVABLE,
            "finalization": None,
        }
    return {
        "state": LIFECYCLE_READY,
        "evidence": "capture ended and the stored geometry matches the keyframes",
        "reason": None,
        "build_in_progress": None,
        "build_in_progress_unavailable_reason": _BUILD_UNOBSERVABLE_REASON,
        "finalization": None,
    }


def _summarise_events(events, corrupt_lines: int = 0) -> dict:
    """Everything any block needs from the journal, in fixed size.

    Computed once per journal change and cached. Deliberately returns
    scalars: nothing downstream is allowed to hold the event list, so the
    cost of a long session is a few integers rather than the session.

    `corrupt_lines` is carried out rather than discarded. `WorldStore`'s
    readers drop that count, so a torn journal silently lowered the
    keyframe total while still labelling it `measured` -- a review
    demonstrated a count going 4 to 3 with nothing on the wire saying why.
    A journal is appended without fsync, so ONE torn line at the tail is
    routine and means "a write is in flight"; several mean corruption.
    Either way the client is told rather than quietly given a smaller
    number.
    """
    accepted = 0
    last_tracking = None
    stopped = False
    tracking_restarts = 0
    chain_breaks = 0
    for event in events:
        kind = event.get("kind")
        if kind == "keyframe_accepted":
            accepted += 1
            last_tracking = kind
        elif kind == "tracking_lost":
            last_tracking = kind
            tracking_restarts += 1
        elif kind == "solve_chain_broken":
            chain_breaks += 1
        elif kind == "session_stopped":
            stopped = True
    return {
        "keyframes_accepted": accepted,
        "last_tracking": last_tracking,
        "stopped": stopped,
        "corrupt_lines": corrupt_lines,
        # COUNTED, not inferred from the segment total.
        #
        # A segment boundary is not a tracking loss. Two independent
        # causes open a segment (engine.py:314 on a tracking loss,
        # engine.py:400 on a solve-chain break), and the engine is
        # explicit that the second must not be read as the wearer having
        # lost the world -- `solve_chain_broken` deliberately does not
        # move `last_tracking`.
        #
        # iOS had no counted number to show, so it rendered
        # `segments - 1` under the words "Tracking restarted N times".
        # On the 2026-09-09 walk that read "121" against 64 actual
        # losses: 57 of the 121 were the solver failing to place a
        # keyframe while tracking was healthy. Both counts ride here now
        # so the phone can stop doing arithmetic on a number that does
        # not mean what its label says.
        "tracking_restarts": tracking_restarts,
        "chain_breaks": chain_breaks,
    }


def _progress_block(session, events, counts_are_final: bool, now: float) -> dict:
    """What the Tower actually counts, and nothing it does not.

    `counts_are_final` means the session was stopped and its record
    rewritten. Until then session.json still holds the zeros written at
    start_session, so the keyframe count comes from the journal instead,
    and `frames_observed` has no source at all -- an ordinary rejected
    frame writes no event, so the number is simply not knowable yet.
    """
    ended = session.ended_at
    elapsed = (ended if ended is not None else now) - session.started_at
    if elapsed < 0.0:
        # A clock that stepped backwards. Reported as UNKNOWN rather than
        # clamped to zero: a session that has been mapping for five
        # minutes must not become indistinguishable from one that just
        # started. Clamping turns an impossible value into a plausible
        # one, which is the worse failure -- and this field is excluded
        # from the revision, so it would arrive with
        # `revision_changed: false` telling a client to skip the redraw.
        mapping_seconds = None
        clock_note = (
            "the Tower's wall clock moved backwards during this session, so "
            "elapsed mapping time cannot be computed"
        )
    else:
        mapping_seconds = elapsed
        clock_note = None
    return {
        "keyframes_accepted": (
            session.keyframes_accepted
            if counts_are_final
            else events["keyframes_accepted"]
        ),
        "keyframes_accepted_provenance": "measured",
        "keyframes_accepted_source": (
            "session record" if counts_are_final else "event journal"
        ),
        "frames_observed": session.frames_observed if counts_are_final else None,
        "frames_observed_unavailable_reason": (
            None
            if counts_are_final
            else (
                "no event is written for an ordinary rejected frame, and this "
                "session's record has not been finalised, so this count is "
                "not knowable yet"
            )
        ),
        "rejected_by_reason": (
            dict(session.rejected_by_reason) if counts_are_final else None
        ),
        "journal_corrupt_lines": events["corrupt_lines"],
        # On the TOWER's clock, per IOS-to-Tower.md 1.8: "the iPhone's idea
        # of elapsed time is not the Tower's idea of mapping time".
        "mapping_seconds": mapping_seconds,
        "mapping_seconds_unavailable_reason": clock_note,
        "mapping_clock": "tower",
        "time_basis": TIME_BASIS,
    }


def _tracking_block(events) -> dict:
    """Coarse tracking state, and why `limited` is never sent.

    `IOS-to-Tower.md` 1.6 accepts good / limited / lost. Tower emits only
    two of the three plus unknown, because `tracking_lost` and
    `keyframe_accepted` are real events with real meanings, while
    "limited" would require a threshold on rejection rate that nobody has
    defined. Inventing one would put a state on screen that looks measured
    and is not -- and iOS explicitly refuses a percentage for that same
    reason.
    """
    last = events["last_tracking"]
    if last == "tracking_lost":
        return {
            "state": TRACKING_LOST,
            "evidence": "the most recent tracking event was tracking_lost",
            "limited_ever_reported": False,
        }
    if last == "keyframe_accepted":
        return {
            "state": TRACKING_GOOD,
            "evidence": "the most recent tracking event was keyframe_accepted",
            "limited_ever_reported": False,
        }
    return {
        "state": TRACKING_UNKNOWN,
        "evidence": "no keyframe has been accepted and no loss recorded",
        "limited_ever_reported": False,
    }


def _calibration_block(session) -> dict:
    """Coarse calibration state. `calibrating` is unreachable in V1.

    Calibration is a property of the SESSION, not the world: intrinsics
    are recorded per session and keyed by resolution, because DAT's
    adaptive ladder changes resolution mid-stream (records.py
    CameraIntrinsics). A world whose sessions were captured at different
    resolutions has no single calibration state.

    There is no in-session calibration procedure -- calibrate_charuco.py
    runs offline, before a session -- so `calibrating` is never emitted.
    """
    intrinsics = session.intrinsics
    if intrinsics.is_known:
        state = CALIBRATION_CALIBRATED
    elif intrinsics.source == INTRINSICS_SOURCE_UNKNOWN:
        state = CALIBRATION_UNCALIBRATED
    else:
        # A source is declared but the numbers do not survive
        # CameraIntrinsics.is_known -- absent, non-finite, or a
        # non-positive focal length. "Unknown" rather than "uncalibrated":
        # something was attempted and this build cannot vouch for it.
        state = CALIBRATION_UNKNOWN
    return {
        "state": state,
        "source": intrinsics.source,
        "calibrated_width": intrinsics.calibrated_width,
        "calibrated_height": intrinsics.calibrated_height,
        "reprojection_rms_px": intrinsics.reprojection_rms_px,
        "view_count": intrinsics.view_count,
        # No percentage, ever. IOS-to-Tower.md 1.5: "'62% calibrated'
        # implies a denominator nobody has defined."
        "calibrating_ever_reported": False,
        "scope": "session",
    }


def _scale_block(world, *, attributable: bool = True) -> dict:
    """The world's scale, but only where it can be attributed to a session.

    `attributable` is false for a session this build has produced no
    geometry for. A scale is EARNED by a build; reporting one for a
    session that was never built would credit it with another session's
    reconstruction.
    """
    scale = world.scale
    if not attributable:
        return {
            "state": SCALE_UNKNOWN,
            "semantics": None,
            "meters_per_unit": None,
            "method": None,
            "confidence": scale.confidence.value,
            "unit": None,
            "allows_metres": False,
            "unavailable_reason": (
                "no build has produced geometry for this session, so it has "
                "no scale of its own; this world's scale was earned by a "
                "different session"
            ),
        }
    return {
        "state": scale.state,
        "semantics": SCALE_SEMANTICS.get(scale.state),
        "meters_per_unit": scale.meters_per_unit,
        "method": scale.method,
        "confidence": scale.confidence.value,
        # The unit string IOS-to-Tower.md 0.5 requires beside every
        # figure. Null when there is no unit at all, which is the honest
        # value for a world with no solved pose.
        "unit": None if scale.state == SCALE_UNKNOWN else "world units",
        "allows_metres": scale.allows_metres,
        "unavailable_reason": None,
    }


def _geometry_block(manifest, current: bool, keyframes_now) -> dict:
    """Geometry, including geometry that is real but BEHIND.

    An earlier version reported anything not matching the current
    keyframes as simply unavailable. That is honest and it is too strict,
    and running the actual product claim showed why: with
    `world_build_session.py --rebuild-every N`, a build finishes and the
    very next keyframe makes its output stale, so a walk that was
    genuinely producing geometry every few keyframes reported **none at
    all** until it stopped. The whole point of --rebuild-every is to watch
    the world grow, and the channel was hiding it.

    A build over the first N keyframes is not wrong; it is a correct
    answer to an older question. So it is reported, with `current: false`
    and BOTH counts, and a consumer can show real progress while knowing
    exactly how far behind it is. Hiding it discarded true information;
    reporting it without the flags would have let a viewer mistake it for
    the finished world. The flags are the whole difference.
    """
    if manifest is None:
        return _geometry_unavailable(
            "no build has run for this session, so no geometry exists"
        )
    return {
        "available": True,
        # Whether this geometry reflects every keyframe accepted so far.
        # False is normal DURING a session that rebuilds as it goes.
        "current": current,
        "built_from_keyframes": manifest.get("keyframes"),
        "keyframes_now": keyframes_now,
        # The Tower's own word, displayed verbatim and never parsed
        # (IOS-to-Tower.md 1.3).
        "representation": GEOMETRY_REPRESENTATION,
        "element_count": manifest.get("points"),
        "element_name": "point",
        # False, and stated. A build replaces the whole derived tree; it
        # never emits a delta. A UI that assumed otherwise "will draw a
        # partial world as a complete one".
        "is_incremental": False,
        # built_at is in here deliberately, even though it makes an
        # identical rebuild look like a change. input_digest covers only
        # the keyframe IDS (store.compute_input_digest), so a rebuild with
        # a different backend, policy or code version produces DIFFERENT
        # GEOMETRY UNDER THE SAME DIGEST. Between a revision that
        # occasionally cries change when nothing changed and one that can
        # stay silent while the geometry moves underneath a viewer, only
        # the first is safe: the cost is a redundant redraw, and the cost
        # of the second is a stale world shown as current.
        "revision": compute_revision(
            {
                "digest": manifest.get("input_digest"),
                "built_at": manifest.get("built_at"),
                "points": manifest.get("points"),
                "solved": manifest.get("poses_solved"),
                "segments": manifest.get("segments"),
                "scale": manifest.get("scale_state"),
            }
        ),
        "provenance": "inferred",
        # Tower keeps per-keyframe and per-edge confidence labels but has
        # never defined an aggregate for a whole reconstruction. Null
        # rather than an average nobody specified.
        "confidence": None,
        "backend_id": manifest.get("backend_id"),
        "built_at": manifest.get("built_at"),
        "time_basis": TIME_BASIS,
        "unavailable_reason": None,
        "stale_reason": (
            None
            if current
            else (
                "keyframes have been accepted since this build ran; these "
                "figures are correct for the keyframes named in "
                "built_from_keyframes and are not the final world"
            )
        ),
    }


def _geometry_unavailable(reason: str) -> dict:
    return {
        "available": False,
        "current": False,
        "built_from_keyframes": None,
        "keyframes_now": None,
        "stale_reason": None,
        "representation": None,
        "element_count": None,
        "element_name": None,
        "is_incremental": False,
        "revision": None,
        "provenance": None,
        "confidence": None,
        "backend_id": None,
        "built_at": None,
        "time_basis": TIME_BASIS,
        "unavailable_reason": reason,
    }


def _trajectory_unavailable(reason: str) -> dict:
    return {
        "available": False,
        "current": False,
        "built_from_keyframes": None,
        "keyframes_now": None,
        "stale_reason": None,
        "pose_count": None,
        "poses_solved": None,
        "poses_refused": None,
        "poses_anchor": None,
        "keyframes": None,
        "segments": None,
        "tracking_restarts": None,
        "chain_breaks": None,
        "path_length": None,
        "revision": None,
        "provenance": None,
        "confidence": None,
        "unavailable_reason": reason,
    }


def _persistence_block(world) -> dict:
    """Did the world survive the session? Always yes, and say so.

    IOS-to-Tower.md 1.7 wants `session` / `saved(revision)` / `reloading`
    kept distinct from "did not say", because "silence is not a promise
    that a world was discarded". World Builder persists everything by
    construction, so this is `saved` with the world's own revision.
    """
    return {
        "state": "saved",
        "revision": compute_revision(
            {
                "world_id": world.world_id,
                "frame_revision": world.frame_revision,
                "sessions": list(world.session_ids),
            }
        ),
        "images_purged": world.images_purged,
        # The Tower owns persistence entirely and iOS stores nothing.
        # Where it is stored is NOT REQUESTED (1.7) and is not sent: a
        # filesystem path on the Tower is useless to a phone and names a
        # machine's layout to a remote consumer.
        "location_disclosed": False,
    }


def _artifacts_block(store, world_id, session_id, world, session=None) -> dict:
    """What imagery exists, and why none of it is offered.

    `IOS-to-Tower.md` 5 is the strictest rule in the document: an image
    whose treatment is not stated is "handled exactly as strictly as raw
    -- withheld", and there is "deliberately no `.probablySafe` and no
    lenient default".

    Since 2026-08-23 keyframes are face-redacted before they are written,
    and the session records WHICH detector ran at WHICH threshold. That is
    reported here verbatim rather than being collapsed to a boolean,
    because the value is a process claim ("this detector's hits were
    filled") and not an outcome claim ("there are no faces"). Sessions
    captured before that keep `none` forever.

    They are still reported as NOT FETCHABLE, and no id or URL is minted.
    A best-effort filter with measured false negatives is not grounds to
    start shipping first-person imagery over the wire, and iOS holds "no
    URL, no id format, and no bytes" -- inventing a fetch scheme would be
    exactly the fabricated contract that document refuses to produce.
    """
    # `present` is TRI-STATE. False is a positive claim that no imagery
    # exists, and this flag cannot support it: a review found a world with
    # images_purged=True still holding three JPEGs, while the wire said
    # `present: false` -- the block's own docstring and the contract both
    # forbid rendering the flag as "the imagery is gone", and the payload
    # did exactly that. Null means "not established".
    present = None
    count = None
    if session_id is not None:
        images = store.images_dir(world_id, session_id)
        try:
            if images.exists():
                count = sum(1 for _ in images.glob("*.jpg"))
                present = count > 0
            else:
                count = 0
                present = False
        except OSError:
            count = None
    return {
        "keyframe_images": {
            "present": present,
            "count": count,
            # What the SESSION recorded, not a constant. A hardcoded
            # "none" survived the arrival of real redaction for exactly as
            # long as it took someone to look.
            "redaction": (
                session.redaction if session is not None else "none"
            ),
            "fetchable": False,
            "reason": (
                "no artifact transfer contract exists, and these remain "
                "first-person frames whose redaction is best-effort with "
                "measured false negatives; a consumer must withhold imagery "
                "it cannot verify"
            ),
        },
        # The FLAG, reported as a flag. An audit confirmed it deletes
        # nothing -- a world with images_purged=True still had every JPEG
        # on disk. What it actually does is make build() refuse
        # (engine.ImagesPurgedError). Reporting it as "the imagery is
        # gone" would be the false assurance 06-PRIVACY-DATA forbids: a
        # purge that cannot delete everything must never report success.
        "images_purged_declared": world.images_purged,
        "images_purged_verified": None,
        "images_purged_meaning": (
            "a declaration that rebuilds are refused for this world, not a "
            "verified deletion of the imagery"
        ),
    }


def _attach_ios_projection(payload: dict) -> None:
    """Add `model_state` and `world_snapshot` -- the fields iOS decodes.

    Derived from the payload that is already built, so the projection and
    the evidence beside it cannot drift.
    """
    lifecycle = payload["lifecycle"]
    state = _MODEL_STATE_BY_LIFECYCLE.get(lifecycle["state"], MODEL_STATE_IDLE)
    payload["model_state"] = state
    payload["model_state_reason"] = lifecycle.get("reason")

    world = payload.get("world")
    if world is None:
        payload["world_snapshot"] = None
        return

    progress = payload.get("progress") or {}
    tracking = payload.get("tracking") or {}
    scale = payload.get("scale") or {}
    calibration = payload.get("calibration") or {}
    geometry = payload.get("geometry") or {}
    trajectory = payload.get("trajectory") or {}
    persistence = payload.get("persistence") or {}

    scale_word = scale.get("semantics") or IOS_SCALE_UNKNOWN
    path = trajectory.get("path_length") or {}
    path_available = bool(path.get("available"))

    payload["world_snapshot"] = {
        "name": world.get("display_name"),
        "world_id": world.get("world_id"),
        "keyframe_count": progress.get("keyframes_accepted"),
        # The SNAPSHOT's revision, which is the envelope's revision: iOS
        # compares it for equality to decide whether anything changed.
        # Filled in by the caller, which is the only place that knows it.
        "revision": None,
        "tracking": _IOS_TRACKING.get(tracking.get("state"), "unavailable"),
        "scale": scale_word,
        "mapping_seconds": progress.get("mapping_seconds"),
        "calibration": calibration.get("state") or "unknown",
        "geometry": {
            "representation": geometry.get("representation"),
            "element_count": geometry.get("element_count"),
            # Never null: iOS uses this to know whether it is looking at a
            # whole world or a delta, and a null would leave that open.
            "is_incremental": bool(geometry.get("is_incremental")),
        },
        "trajectory": {
            "pose_count": trajectory.get("pose_count"),
            "path_length": path.get("value") if path_available else None,
            "path_length_unit": path.get("unit") if path_available else None,
            # Carried separately from the snapshot's scale, because
            # handoff.md 9.6 says every spatial figure carries its own.
            "scale": (
                path.get("scale_semantics") if path_available else IOS_SCALE_UNKNOWN
            ),
        },
        "persistence": {
            "state": persistence.get("state") or "unknown",
            "revision": persistence.get("revision"),
        },
    }


# -- disk helpers -------------------------------------------------------


# Keys a manifest must carry before it counts as evidence of geometry.
# `engine.build` writes all of them; a manifest missing any is truncated,
# hand-edited, or from a writer this build does not understand.
#
# Gating on "the file exists" instead let a stripped manifest produce
# `available: true` with every figure null -- "we have geometry" asserted
# with nothing to show for it -- which an adversarial review demonstrated.
REQUIRED_MANIFEST_KEYS = (
    "input_digest",
    "session_id",
    "keyframes",
    "points",
    "poses_solved",
    "poses_refused",
    "segments",
)


def _read_manifest(store, world_id):
    """The world's derived manifest, unfiltered, or None.

    Deliberately NOT filtered by session, despite the manifest carrying a
    session_id -- the caller does that, and it matters where the check
    lives. This function is called through a file-fingerprint cache keyed
    on the manifest PATH, and that path is shared by every session in a
    world (store.derived_manifest_path). A cached value filtered for one
    session would be handed to another, which is the exact
    misattribution the check exists to prevent.

    So: this reads and validates the schema; `_payload` decides whether
    the manifest describes the session being reported on. An earlier
    version of this function claimed to do both and did neither after the
    cache was introduced.
    """
    manifest = store.read_derived_manifest(world_id)
    if manifest is None:
        return None
    if manifest.get("schema_version") != SCHEMA_VERSION:
        # store.derived_is_current checks this; nothing else does. A
        # manifest from another schema describes fields whose meaning this
        # build does not know.
        return None
    missing = [key for key in REQUIRED_MANIFEST_KEYS if manifest.get(key) is None]
    if missing:
        logger.warning(
            "result channel: derived manifest for %s is missing %s; "
            "treating it as absent rather than reporting geometry with no "
            "figures",
            world_id,
            missing,
        )
        return None
    return manifest


def _pose_count(manifest):
    """Poses carrying a position that is EVIDENCE.

    This used to be `keyframes - poses_refused`, and that arithmetic is
    what put "Camera poses: 36" on the phone during the 2026-08-24
    physical walk, from a manifest reading `poses_solved: 0, points: 0,
    backend_id: "unposed", segments: 36`.

    `engine.build` counts an ANCHOR as neither solved nor refused, so
    subtraction quietly promotes every anchor to a camera position. An
    anchor is definitional, not measured -- identity rotation, zero
    translation -- and `backends/unposed.py` says the status exists
    precisely so "a downstream consumer [cannot] count it as evidence".
    All 36 of those anchors were the same point.

    But an anchor IS a real position when the chain it anchors resolved:
    it is that segment's origin, and dropping it would under-report
    every segment by one. The rule is therefore per segment, which only
    the build can evaluate, so the build now records the answer as
    `poses_positioned` instead of leaving it to be inferred here.
    """
    positioned = manifest.get("poses_positioned")
    if isinstance(positioned, bool) or not isinstance(positioned, int):
        # No fallback. The old arithmetic -- keyframes - poses_refused --
        # counted a segment ANCHOR as a camera position, and an anchor is
        # definitional: identity rotation, zero translation, one per segment.
        # That is where "Camera poses: 36" came from on a world whose manifest
        # read poses_solved: 0. A manifest without poses_positioned predates
        # the fix, and absent is the only honest answer for it.
        return None
    return max(0, positioned)


def _keyframes_digest(store, world_id, session_id):
    from tower.world_builder.store import compute_input_digest

    try:
        return compute_input_digest(store.read_keyframes(world_id, session_id))
    except (WorldStoreError, KeyError, ValueError, OSError):
        return None


