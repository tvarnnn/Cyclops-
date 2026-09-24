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
import time
from pathlib import Path

from tower.logging_config import client_safe_reason
from tower.results.contracts import TIME_BASIS
from tower.results.world_builder_library import _sortable, lock_speaks_for
from tower.storage import read_json_closed, read_raw_jsonl
from tower.results.envelope import Snapshot, compute_revision
from tower.world_builder.records import FINAL_SOLVE_SOLVED, format_distance
from tower.world_builder.relocalizer import RecoveryJournalReader
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
from tower.world_builder.store import (
    WorldStore,
    WorldStoreError,
    compute_input_digest,
    validate_manifest,
)

logger = logging.getLogger(__name__)

# The settled photographic vocabulary, imported at module scope because it
# is a tuple of strings: `tower.world_builder.photographic` pulls in nothing
# heavy at import time, and the two probes it uses are themselves imported
# lazily inside it.
from tower.world_builder.photographic import (  # noqa: E402
    PHOTOGRAPHIC_OWED,
    PHOTOGRAPHIC_UNOBSERVABLE,
    is_unsettled,
)

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
    # "capture ended, Tower still working; figures may change" is what
    # `stopped_unbuilt` means HERE, and here it is right: the only state
    # that still reaches this mapping is a session whose geometry is
    # BEHIND its journal. A rebuild is outstanding, the world is intact,
    # and "wait" is the honest word.
    #
    # THE PREVIOUS ROUND POINTED THIS AT `interrupted` AND THAT WAS TOO
    # BROAD. `stopped_unbuilt` was carrying two states -- "built and
    # behind" and "nothing here at all" -- and only the second is settled.
    # A reviewer built both: the first started rendering a red
    # "Interrupted ... what was built before it stopped is here" over a
    # complete world that merely needed a rebuild. The same reviewer also
    # showed the mapping's stated premise to be false: `stop_session()`
    # DEFAULTS to `hold_lock=False`, so a caller that takes the default
    # releases the lock and THEN builds, and a build really can be running
    # here. Narrower than it first read, and a later reviewer measured the
    # difference: the shipped offline driver,
    # `scripts/world_build_session.py`, passes `hold_lock=True` at both of
    # its stop sites, so the callers that take the default are the
    # research and benchmark scripts -- which have no phone watching
    # them.
    #
    # The empty case is separated at the branch instead, where it can be
    # said precisely. See `_lifecycle`.
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
    # SIX reads per session per snapshot -- events, the session manifest,
    # the world manifest, the keyframe digest, and (only for a session no
    # manifest describes) a pose summary and a point count. It was four
    # until the recount was added, and this comment still said four; a
    # reviewer counted. 256 therefore covers about 42 sessions rather
    # than the 64 the next paragraph was written for, which is still more
    # than any subscriber reaches.
    #
    # And the entries are NOT all small: a reviewer parsed the preserved
    # field artifact's manifest and measured **63.5 KB** for one, against
    # 768 bytes for an events summary. The bound was 64, and 256 is
    # ~7 MiB worst case while covering 64 sessions -- more than any
    # subscriber reaches. (An earlier version of this comment argued
    # against 512, a number that was never in this file; a reviewer
    # checked it against the diff. A comment that cites a value has to
    # cite the one the code had.) The eviction below is one-at-a-time and
    # least-recently-used, so crossing the bound now costs one re-read
    # rather than every reader's.
    MAX_ENTRIES = 256

    __slots__ = ("_entries",)

    def __init__(self) -> None:
        self._entries: dict = {}

    def read(self, kind: str, path, reader):
        """`reader()`'s result for `path`, reparsed only when it changes.

        `kind` NAMES THE READ, and it is not decoration. The key used to
        be the path alone, so two callers asking different questions of
        one file shared an entry: the first answer won and the second
        caller silently got it, with no error anywhere.

        That is not hypothetical. It happened during this campaign: a
        reader was added that asked what a manifest file CLAIMS, beside
        the existing reader that asks `_validate_manifest` what it is
        WORTH. The second reader got the first's `None` and concluded the
        file made no claim, so a check that was supposed to fire never
        did. The hostile suite caught it, and the reader that hit it was
        later removed for unrelated reasons -- but the hazard is a
        property of a path-keyed cache, not of that reader, and the next
        one to ask a second question of a file would have found it again.

        One file, two questions, two entries.
        """
        try:
            stat = path.stat()
            fingerprint = (stat.st_size, stat.st_mtime_ns)
        except OSError:
            # Absent or unreadable. Do not cache: a file that appears
            # later must be picked up on the next poll.
            self._entries.pop((kind, str(path)), None)
            return reader()
        key = (kind, str(path))
        cached = self._entries.get(key)
        if cached is not None and cached[0] == fingerprint:
            # MOVE TO THE END ON A HIT: least-recently-USED, not
            # first-in-first-out. `d[existing] = v` does not reorder a
            # dict, so the hottest entry -- the live session's journal,
            # refreshed every poll -- kept its original slot and was
            # evicted FIRST. A reviewer demonstrated it on a three-key
            # dict. `pop` then reinsert is the reorder.
            #
            # `pop(key, None)`, NOT `pop(key)`. This producer is a
            # process-lifetime singleton and `_snapshot_for` is reached
            # from `asyncio.to_thread` on BOTH the publisher's poll loop
            # and every websocket's subscribe handler, so two threads read
            # one cache. Check-then-act on a bare `pop` raised KeyError 15
            # times in 16 threads under a reviewer's stress harness -- and
            # KeyError is in `snapshot()`'s except tuple, so the world
            # blinked out of existence on the phone for a poll.
            self._entries.pop(key, None)
            self._entries[key] = cached
            return cached[1]
        value = reader()
        # OLDEST OUT, not everything out. Clearing the lot turned the bound
        # into a cliff: a reviewer measured the hit rate falling from 95.8%
        # to 14.0% the moment the entry count crossed it, and this producer
        # is a process-lifetime singleton shared by every subscriber on a
        # host that holds 163 worlds. Python dicts keep insertion order, so
        # the first key is the oldest touch.
        #
        # `list(...)`, so the eviction does not iterate a dict another
        # thread may be writing: the same reviewer's harness produced
        # "RuntimeError: dictionary changed size during iteration", and
        # RuntimeError is NOT in `snapshot()`'s except tuple -- it escapes
        # to the publisher's consecutive-failure counter and can fail the
        # target.
        while len(self._entries) >= self.MAX_ENTRIES:
            oldest = next(iter(list(self._entries)), None)
            if oldest is None:
                break
            self._entries.pop(oldest, None)
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
            # `_sortable`, FOR THE REASON THE LISTING GIVES -- and this
            # is the surface that is up during the walk.
            #
            # `world_from_json_dict` does not coerce `updated_at`, and
            # `best_at` starts at `-math.inf`, so a `world.json` carrying
            # a string raises `TypeError` HERE, in
            # `resolve_with_selection`, which runs BEFORE `snapshot()`'s
            # try. Two rounds hardened the three HTTP surfaces against
            # exactly this and did not grep for the other readers of the
            # same fields. A reviewer built all 24 corruption shapes and
            # found `GET /worlds`, `/render` and `/geometry/manifest`
            # surviving every one while the status channel died on all of
            # them.
            if _sortable(world.updated_at) > _sortable(best_at):
                best, best_at = wid, world.updated_at
        if best is None:
            return None, SELECTION_NONE, "no world could be read"
        if not live:
            # A PHOTOGRAPHIC BUILD IS ALSO "NOW", AND THE SELECTION HAS TO
            # SAY SO OR ONE PAYLOAD CONTRADICTS ITSELF.
            #
            # `live` is the writer lock, and the surface, appearance and
            # dense stages run with it RELEASED -- so for the six to sixteen
            # minutes they take this answered `latest` / "nothing is live" in
            # the same payload whose `lifecycle` said `finalizing` with
            # `build_in_progress: true`. A reviewer saw it in every run.
            #
            # Not cosmetic. On iOS `WorldSelection.isHistoryOfferedAsLive` is
            # `mode == .latest`, and `TowerWorldBuilderClient` then presents
            # the report as `.idle` -- "No world yet" -- unless `followedWalk`
            # still names the world; and `followedWalk` stops being refreshed
            # the moment the selection turns `latest`. So a relaunch or a
            # jetsam kill during the build dropped the live screen to "No
            # world yet" while the Tower was reporting a build in progress.
            #
            # ONLY `best` IS PROBED, not every world. Reaching here means no
            # world holds a live lock, so nothing is capturing, and the world
            # being built is the one that was just walked -- which is the
            # most recently updated. Probing every world would cost a lock
            # read and three stats per world on a 2 Hz poll (about 20 ms on
            # the 163-world root) to change the answer for a world that
            # cannot be the newest.
            if self._newest_session_is_being_built(store, best):
                return (
                    best,
                    SELECTION_FINALIZING,
                    "no writer lock is held, and a photographic build is "
                    "still running for this world's newest session",
                )
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

    def _newest_session_is_being_built(self, store, world_id) -> bool:
        """Whether this world's newest session has a photographic stage running.

        The same probe `_lifecycle` uses, so the `selection` block and the
        `lifecycle` block in one payload cannot say opposite things about one
        world. An unobservable answer is NOT treated as "yes": the selection
        has no field to carry a caveat, and promoting a world on a probe that
        just failed would be a guess. `lifecycle` carries the caveat instead.
        """
        sessions = store.list_session_ids(world_id)
        if not sessions:
            return False
        latest = self._latest_session(store, world_id, sessions)
        evidence, _unobservable = _photographic_build_evidence(
            store, world_id, latest
        )
        return evidence is not None

    def _newest_session_is_stopped(self, store, world_id) -> bool:
        sessions = store.list_session_ids(world_id)
        if not sessions:
            return False
        latest = self._latest_session(store, world_id, sessions)
        summary = self._files.read(
            "events",
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
            # See the world loop above: `started_at` is uncoerced too.
            if _sortable(session.started_at) > _sortable(best_at):
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
        except (
            WorldStoreError,
            KeyError,
            ValueError,
            OSError,
            # `TypeError` AND `OverflowError`, because a record field that
            # is the wrong TYPE is a corrupt-input problem exactly like a
            # missing key, and neither was in this tuple.
            #
            # `_elapsed_seconds` subtracts `session.started_at` from the
            # clock; a string raises TypeError and a `10**400` raises
            # OverflowError, both from inside the snapshot rather than
            # from the resolver above. Uncaught, they reach
            # `publisher.poll_once`'s bare `except Exception`, and after
            # `MAX_CONSECUTIVE_TARGET_FAILURES` the subscriber is sent
            # `fail_target` and the panel stops updating **for the rest of
            # the walk**. A reviewer traced that path; the alternative is
            # one poll reporting `unavailable`, which is what every other
            # corrupt input here already does.
            TypeError,
            OverflowError,
        ) as exc:
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
            "events",
            store.events_path(world.world_id, session_id),
            lambda: _summarise_events(
                *read_raw_jsonl(store.events_path(world.world_id, session_id))
            ),
        )
        holder = store.lock_holder(world.world_id)
        # THE COPY BESIDE THE GEOMETRY DECIDES, and it decides here too.
        #
        # This read the WORLD's manifest first and fell back to the
        # session's; `store.derived_currency` and
        # `world_builder_geometry._session_manifest` do the opposite. A
        # reviewer built the eleven states where the two copies disagree
        # and found the readers picking different manifests -- reproducing
        # BOTH of this campaign's named failures at once: `ready` beside a
        # 404, and a route serving geometry the phone was told was still
        # finalizing. Four readers, one rule.
        manifest = self._files.read(
            "validated-manifest",
            store.session_manifest_path(world.world_id, session_id),
            lambda: _validate_manifest(
                store.read_session_manifest(world.world_id, session_id),
                world.world_id,
                source="session manifest",
            ),
        )
        if manifest is not None and manifest.get("session_id") != session_id:
            # A session's own copy naming somebody else is corruption.
            manifest = None
        keyframes_current = self._files.read(
            "keyframe-digest",
            store.keyframes_path(world.world_id, session_id),
            lambda: _keyframes_digest(store, world.world_id, session_id),
        )

        if manifest is None:
            # THE WORLD'S COPY, only because this session has none of its
            # own -- a world built before `write_derived` wrote one. It is
            # still filtered by session: a manifest naming somebody else is
            # not evidence about this session, and attributing it would
            # report one session's geometry as another's.
            #
            # `_read_manifest` deliberately does not filter, because its
            # result is cached per FILE and that file is shared by every
            # session in the world. The check lives here, at the point of
            # use, for that reason.
            #
            # When both are absent the four hand-written branches below
            # take over. They exist for legacy worlds only.
            manifest = self._files.read(
                "validated-manifest",
                store.derived_manifest_path(world.world_id),
                lambda: _read_manifest(store, world.world_id),
            )
            if manifest is not None and manifest.get("session_id") != session_id:
                # The world's copy is about another session. Attributing it
                # here would report one session's geometry as another's.
                manifest = None

        stopped = events['stopped']
        geometry_current = (
            manifest is not None
            and keyframes_current is not None
            and manifest.get("input_digest") == keyframes_current
        )

        session_geometry = _has_session_geometry(store, world.world_id, session_id)
        # The figures a manifest would have carried, recovered from the
        # files it would have described. Only when there is no manifest
        # and there IS a tree, so the ordinary path costs one `is None`.
        tree_figures = (
            _figures_from_the_tree(self._files, store, world.world_id, session_id)
            if manifest is None and session_geometry
            else None
        )
        lifecycle = _lifecycle(
            # The WORLD's lock, only where it speaks for THIS session: a
            # finisher or a new walk holding it says nothing about a sibling
            # whose record is closed (`world_builder_library.lock_speaks_for`).
            holder=holder if lock_speaks_for(session) else None,
            stopped=stopped,
            session=session,
            # The files the phone opens, for THIS session -- not the
            # world's manifest, which names whichever session built last.
            has_session_geometry=session_geometry,
            geometry_current=geometry_current,
            # A manifest FILE, not the figures. `tree_figures` gives the
            # blocks below their numbers back, and deliberately does not
            # make this True: `has_manifest` decides which lifecycle state
            # this session is in, and "a summary was recomputed from the
            # files" is not "a build recorded what it did".
            has_manifest=manifest is not None,
            # THE FIGURES, NOT THE PARSE. `tree_figures is not None`
            # means only that poses.json and points.json parsed; a
            # featureless walk parses perfectly and counts zero. Saying
            # "the world opens" over that is the same mistake
            # `_has_drawable_geometry` was written to stop the projection
            # making, made again in prose two branches away, and shipped
            # to the phone as `lifecycle.reason` while the phone's own
            # predicate drew "Needs retry". A reviewer built it.
            has_readable_figures=_figures_are_drawable(tree_figures),
            # The files a photographic stage writes while it runs. Passed,
            # not probed here, because only `_lifecycle` knows whether the
            # question is worth asking -- it asks on exactly one arm, the one
            # where the lock has been released and the disk otherwise looks
            # finished. Direct callers that pass only a record get the old
            # behaviour, which is the honest answer for a caller with no
            # files to look at.
            store=store,
            world_id=world.world_id,
            session_id=session_id,
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
            "geometry": _geometry_block(
                manifest, geometry_current, keyframes_now,
                has_session_geometry=session_geometry,
                tree_figures=tree_figures,
            ),
            "trajectory": self._trajectory_block(
                store, world, session_id, manifest, geometry_current,
                keyframes_now, events,
                has_session_geometry=session_geometry,
                tree_figures=tree_figures,
            ),
            "persistence": _persistence_block(world),
            "artifacts": _artifacts_block(
                store, world.world_id, session_id, world, session
            ),
            "time_basis": TIME_BASIS,
        }

    def _trajectory_block(
        self, store, world, session_id, manifest, current, keyframes_now,
        events, *, has_session_geometry: bool = False, tree_figures=None,
    ) -> dict:
        # Same reasoning as _geometry_block: a trajectory over the first N
        # keyframes is a correct answer to an older question, not a wrong
        # answer, and hiding it makes a live session look idle.
        if manifest is not None and not has_session_geometry:
            # See `_geometry_block`: the same payload, the same reason.
            return _trajectory_unavailable(
                "a build ran for this session and its poses are no longer "
                "on disk; the capture is still there and it can be rebuilt"
            )
        if manifest is None:
            if has_session_geometry:
                # Counted from poses.json, for the reason `_geometry_block`
                # gives at length. This block previously said the poses
                # "cannot be summarised here" while sitting a `stat()` away
                # from the file that holds them.
                if tree_figures is not None:
                    manifest = tree_figures
                else:
                    return _trajectory_unavailable(
                        "this session has poses on disk and they could not "
                        "be read, so nothing here can summarise them"
                    )
            else:
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
                "tree": manifest.get("tree_fingerprint"),
            }
        )
        return {
            "available": True,
            "current": current,
            "built_from_keyframes": manifest.get("keyframes"),
            "keyframes_now": keyframes_now,
            "stale_reason": _stale_reason(
                current,
                manifest,
                (
                    "keyframes have been accepted since this build ran; this "
                    "path covers built_from_keyframes of them"
                ),
                (
                    "these poses were counted from the file on disk because "
                    "no manifest describes them, so nothing here can say "
                    "which keyframes produced them"
                ),
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
            # "UNREADABLE" IS ONE OF TWO REASONS AND IT USED TO BE THE
            # ONLY SENTENCE. `read_derived` returns None for a tree it
            # will not SERVE as well as for one it cannot read, and the
            # first is much the commoner: a build older than its
            # keyframes is refused by the verify gate by design. A
            # reviewer printed this block's `"the derived poses are
            # unreadable"` beside `pose_count: 4` and `element_count:
            # 1347` counted from those same files, in one payload.
            #
            # The two are told apart by asking the gate what it thinks,
            # rather than by guessing from a None.
            stale = False
            try:
                digest = compute_input_digest(
                    store.read_keyframes(world.world_id, session_id)
                )
                stale = store.derived_currency(
                    world.world_id, digest, session_id
                ) is False
            except (WorldStoreError, KeyError, ValueError, OSError):
                stale = False
            return {
                "available": False,
                "reason": (
                    "this session's stored poses are older than its "
                    "keyframes, so a distance along them would not be the "
                    "distance walked"
                    if stale
                    else "the derived poses could not be read"
                ),
            }

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


def _has_session_geometry(store, world_id: str, session_id: str) -> bool:
    """Whether a derived tree was BUILT for this session.

    Existence, not openability, and the difference is not a detail. The
    serving path (`routes/geometry.py` -> `WorldStore.read_derived`) opens
    and parses both files; a reviewer built empty ones, truncated ones and
    wrong-shaped ones and got `ready` from this beside a 404 from the route.
    So this answers "was there a build", which is a question about the
    session, and the route answers "can it be read", which is a question
    about the bytes. They are different failures and they read differently
    on the phone: nothing here versus something corrupt.

    It stays a stat rather than a parse because it runs on the 0.5 s status
    poll and `points.json` is megabytes on a real walk. Two `stat()` calls
    measure 72 microseconds; parsing would be four orders of magnitude
    worse for an answer the route is about to give properly anyway.

    Deliberately the SAME two files `world_builder_library._has_geometry`
    asks about, because the listing and the status producer answering that
    question differently is how a picker row and the canvas it opens end up
    disagreeing. There are three copies of this predicate -- here,
    `world_builder_library.py` and `world_builder_render.py` -- because the
    modules must not import each other, and a test pins all three together.
    """
    derived = store.derived_dir(world_id) / session_id
    return (derived / "poses.json").exists() and (derived / "points.json").exists()


# The stages that make the PHOTOGRAPHIC representation, in the order
# `world_build_session.py` runs them. Deliberately the same three
# `world_builder_render.session_build_running` probes, and in the same
# order, so the state and the evidence given for it cannot name different
# stages.
_PHOTOGRAPHIC_STAGES = ("surface", "appearance", "dense")

# HOW LONG A `running` STATUS IS BELIEVED WITHOUT MOVING.
#
# A ceiling is necessary because the pid probe underneath this one has a
# documented one-way failure: `store._holder_is_running` treats
# `psutil.AccessDenied` as ALIVE -- "cannot judge is not dead", which is the
# right default for a writer lock. So a builder killed mid-surface leaves
# `status.json` saying `running` under pid P, and if the OS later recycles P
# onto a process this Tower may not inspect, the status reads alive forever.
# Nothing else clears it: only another `surfacify()` rewrites the file, and
# `_stage_running`'s "manifest newer than the status" escape hatch needs a
# manifest that a killed build never wrote. Without a bound the phone would
# sit on "Improving" permanently over a finished world -- this module's own
# lie, pointing the other way.
#
# THE NUMBER IS DELIBERATELY LOOSE, because the two errors are not
# symmetrical. Too tight reintroduces "Saved" in the middle of a real build,
# which is the failure this whole change exists to kill and which destroys
# work; too loose only means a wedged status takes longer to clear itself.
#
# What the ceiling must clear is the longest SUB-STAGE, not the longest
# stage: `_status()` rewrites `running` with a fresh `updated_at` at every
# sub-stage boundary (surface_pipeline.py:651, 673, 696, 1009, 1121, 1264),
# six writes across a 6-16 minute build. Measured on this machine: surface
# 374 s and appearance 85-91 s for ~385 keyframes, and a 690-keyframe walk
# scales about 1.75x, so ~655 s for the longest whole stage. No sub-stage can
# exceed its stage, so 655 s is a hard upper bound on the longest sub-stage,
# and it is a generous one -- six boundaries mean the true figure is a
# fraction of it.
#
# One hour is therefore about 5.5x an upper bound that is itself
# conservative. It could be tightened once someone measures a per-sub-stage
# histogram on a long walk, and it should not be tightened before that: what
# this buys is turning "forever" into "an hour", and an hour of a stale
# "Improving" row costs a wearer patience, while a minute of premature
# "Saved" costs them the reconstruction.
_STAGE_STATUS_MAX_AGE_S = 3600.0


def _stage_staleness_probe(stage: str):
    """`status_is_stale` for one stage, imported at call time.

    ONE STAGE PER CALL, and that is the whole point. The first version of this
    imported the dense and surface probes inside a single `try`, so a dense
    module that would not import made this function answer "nothing is
    building" for surface and appearance too -- the fix silently disabling
    itself, and the phone back to "Saved". `session_build_running` separates
    them for exactly this reason and this now matches it.
    """
    if stage == "dense":
        from tower.world_builder.dense_pipeline import (  # noqa: PLC0415
            status_is_stale,
        )
        return status_is_stale
    from tower.world_builder.surface_pipeline import (  # noqa: PLC0415
        status_is_stale,
    )
    return status_is_stale


def _stage_status_is_fresh(status_path, now: float) -> bool:
    """Whether a `running` status has moved recently enough to be believed.

    `updated_at` is the pipeline's own clock and is written on every state; a
    status that lost it falls back to the file's mtime, which is always there
    and is written by the same atomic replace. See `_STAGE_STATUS_MAX_AGE_S`
    for why a bound is needed at all and why it is this loose.
    """
    written = None
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if isinstance(status, dict):
            candidate = status.get("updated_at")
            if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
                written = float(candidate)
    except (OSError, ValueError):
        return False
    if written is None:
        try:
            written = status_path.stat().st_mtime
        except OSError:
            return False
    return (now - written) <= _STAGE_STATUS_MAX_AGE_S


def _stage_status_pid(status_path) -> int | None:
    try:
        pid = json.loads(status_path.read_text(encoding="utf-8")).get("pid")
    except (OSError, ValueError, AttributeError):
        return None
    return pid if isinstance(pid, int) and not isinstance(pid, bool) else None


# WARN ONCE, THEN DEBUG. The status poll runs twice a second, so a persistent
# import failure would write two warnings a second for the life of the Tower.
# The payload carries `build_in_progress_unavailable_reason` on every poll, so
# the log only has to make the failure visible, not repeat it.
_PROBE_FAILURES_SEEN: set[str] = set()


def _warn_probe_failure(message: str) -> None:
    if message in _PROBE_FAILURES_SEEN:
        logger.debug(message, exc_info=True)
        return
    # Bounded: a flood of DISTINCT messages is itself a bug, and an unbounded
    # set in a 2 Hz poll would be a slow leak.
    if len(_PROBE_FAILURES_SEEN) < 64:
        _PROBE_FAILURES_SEEN.add(message)
    logger.warning(message, exc_info=True)


def _photographic_build_evidence(store, world_id, session_id):
    """Whether a photographic stage is building this session, and how sure.

    Returns `(evidence, unobservable_reason)`, exactly one of which is
    non-None, or `(None, None)` for "nothing is building, and that is a real
    answer".

    THE LOCK IS NOT THE ONLY EVIDENCE THAT A WORLD IS STILL BEING BUILT, and
    treating it as the only evidence is what made a phone say "Saved" over a
    reconstruction that was sixteen minutes away. On 2026-09-22 a wearer
    walked a bedroom, pressed Stop, read "Saved" ninety seconds later, and
    shut the Tower down; the surface build died with it.

    The ordering that opens the gap is CORRECT and must stay:
    `scripts/world_build_session.py` marks finalization complete and releases
    the world writer lock in its `finally`, and only THEN runs `--surface`,
    so the phone can read the world while the better picture is being made
    (`world_builder_render.session_build_running`'s docstring says why, under
    "Not a promise that nothing will change when it is false"). What was
    wrong is that `_lifecycle` read "a build is running" off the LOCK alone,
    and the lock is exactly what that ordering gives up.

    `session_build_running` already knows how to close the gap -- it reads
    each stage's own `status.json` and refuses one whose process is gone --
    so this asks it rather than re-deriving it, and adds the one thing it
    does not have: the age ceiling in `_STAGE_STATUS_MAX_AGE_S`.

    A STRING, not a bool, because `_lifecycle` publishes its evidence to the
    phone and the honest sentence here is not the lock-held arm's. The lock
    is released. "A live process holds the writer lock" would be a second
    untruth told to cover the first.

    IT DOES NOT FAIL SILENTLY, and the first version did. Every probe was
    wrapped in `except Exception: logger.debug(...)` returning None, so any
    import or read error quietly restored the behaviour this whole change
    exists to remove: the Tower answers `ready`, iOS says "Saved" over a
    world that is still building, and nothing anywhere says why. A reviewer
    hit it for real -- another lane briefly made `world_builder/surface.py`
    raise `NameError` at import. The failure mode of a fix must not be the
    bug it fixes, so a swallowed exception now comes back as the second
    element, which `_lifecycle` publishes as
    `build_in_progress_unavailable_reason`, and is logged at WARNING once.

    THE ONE PLACE THIS IS ANSWERED. `world_builder_library` imports it rather
    than deriving its own, because the panel saying "Improving" while the
    picker row for the same session says "Complete" is how a wearer decides
    the world is finished and shuts the Tower down -- the same failure, told
    twice. Two independent reviewers found that divergence.
    """
    if store is None or world_id is None or session_id is None:
        # A caller that handed over only the record and no store -- the
        # direct `_lifecycle(holder=..., stopped=..., session=...)` callers
        # in `test_world_builder_finalize_cli` and
        # `test_world_builder_lifecycle` do exactly this. No files to probe,
        # so no claim and no complaint: the old answer is the honest one for
        # a caller that asked only about the record.
        return None, None
    try:
        # Imported INSIDE the function, the convention `world_builder_render`
        # itself uses a few lines below `session_build_running`. That module
        # reaches into `results/` and `world_builder/` at call time; a
        # module-scope import would put a render module, a surface pipeline
        # and a dense pipeline into the import graph of every status poll for
        # a probe that most polls never make, and would leave this module one
        # future edit away from an import cycle it cannot see.
        from tower.results.world_builder_render import (  # noqa: PLC0415
            _stage_running,
            session_build_running,
        )
    except Exception as exc:  # noqa: BLE001 -- reported, not swallowed
        detail = client_safe_reason(exc)
        _warn_probe_failure(
            f"[Tower][WorldBuilder] the photographic build probe cannot be "
            f"loaded ({detail}); whether a build is running is now "
            f"unobservable and the status says so"
        )
        return None, f"the photographic build probe could not be loaded: {detail}"
    try:
        if not session_build_running(store, world_id, session_id):
            return None, None
        world_dir = store.world_dir(world_id)
    except Exception as exc:  # noqa: BLE001 -- documented never to raise; belt too
        detail = client_safe_reason(exc)
        _warn_probe_failure(
            f"[Tower][WorldBuilder] the photographic build gate failed "
            f"({detail}); whether a build is running is now unobservable"
        )
        return None, f"the photographic build probe failed: {detail}"
    now = time.time()
    unobservable = None
    for stage in _PHOTOGRAPHIC_STAGES:
        # ONE `try` PER STAGE. A module that will not import, or a probe that
        # throws, must cost this function that stage and nothing else --
        # `session_build_running` separates its surface and dense probes for
        # exactly this reason, and the first version of this did not.
        try:
            status_path = world_dir / stage / session_id / "status.json"
            # `_stage_running`, NOT a fresh `state == "running"` read. It is
            # the same per-stage predicate the gate above is made of --
            # staleness AND the published-manifest rule (review 3, R5) -- so
            # the stage this names is a stage that is genuinely running, and
            # the evidence cannot drift from the decision.
            if not _stage_running(status_path, _stage_staleness_probe(stage)):
                continue
            if not _stage_status_is_fresh(status_path, now):
                # Running, its pid reads alive, and it has not moved in an
                # hour. See `_STAGE_STATUS_MAX_AGE_S`.
                continue
            pid = _stage_status_pid(status_path)
        except Exception as exc:  # noqa: BLE001 -- this stage only
            detail = client_safe_reason(exc)
            _warn_probe_failure(
                f"[Tower][WorldBuilder] the {stage} liveness probe failed "
                f"({detail}); a build of that stage would not be reported"
            )
            unobservable = f"the {stage} build probe failed: {detail}"
            continue
        where = f"{stage}/{session_id}/status.json"
        if pid is not None:
            return (
                f"{where} says 'running' and the process that wrote it "
                f"(pid {pid}) is still alive"
            ), None
        return f"{where} says 'running' and its process is still alive", None
    if unobservable is not None:
        # The gate said something is running and the stage that would name it
        # is the one that threw. "Nothing is building" would be a guess.
        return None, unobservable
    # The gate said yes and no stage owns it. Either the WORLD LOCK was taken
    # between the two calls (a new capture starting on this world, which the
    # caller's `holder` describes better), or every `running` status is past
    # the age ceiling. Say nothing rather than guess.
    return None, None


def _photographic_state_or_none(store, world_id, session_id, session):
    """`photographic_state`, wrapped so the status channel cannot be broken by it.

    Imported and called at request time, like every other probe in this
    module, and every failure becomes the `unobservable` WORD rather than an
    exception. This function is on the 2 Hz status path: a traceback here
    would take out the whole payload, and the payload is how the wearer
    learns anything at all.
    """
    if store is None or world_id is None or session_id is None:
        # THE SAME GUARD `_photographic_build_evidence` MAKES, and for the
        # same reason: `_lifecycle` takes these three as optional, and a
        # caller that asked only about the record gets only the record. An
        # absent store is not a broken probe -- turning it into
        # `unobservable` would make every such caller report "Improving",
        # which is the second of the two failures this fix has to avoid.
        return None
    try:
        from tower.world_builder.photographic import (  # noqa: PLC0415
            photographic_state,
        )

        return photographic_state(store, world_id, session_id, session)
    except Exception as exc:  # noqa: BLE001 -- reported, never swallowed
        detail = client_safe_reason(exc)
        _warn_probe_failure(
            f"[Tower][WorldBuilder] the photographic state could not be "
            f"computed ({detail}); this world will not be reported finished"
        )
        return {
            "state": PHOTOGRAPHIC_UNOBSERVABLE,
            "stage": None,
            "detail": f"the photographic state could not be computed: {detail}",
        }


def _still_building(base: dict, building: str | None,
                    unobservable: str | None = None,
                    photographic: dict | None = None) -> dict:
    """The settled answer, corrected by the one present-tense fact it cannot see.

    A SETTLED WORD AND A PRESENT-TENSE CLAIM ARE DIFFERENT QUESTIONS, and
    returning both from one branch is what conflated them. `state` and
    `reason` describe what HAPPENED to this session; `build_in_progress`
    describes whether a process is working on it RIGHT NOW. The first version
    of this fix owned a branch of its own placed below the two
    `end_reason in ("error", "interrupted")` arms, which meant it only ever
    corrected the `end_reason: "stop"` case -- and `"stop"` is not the
    ordinary shape. Leaving the World Builder screen sends `session/stop`
    while the capture is still open, so the record reads
    `end_reason: interrupted` and then finalises complete and solved
    (`world_builder_library.session_state` says so in its own comments); a
    walk that trips the recorder's 40-minute bound ends `interrupted` too, by
    design; and `stop_session(END_REASON_ERROR, ...)` still runs a best-effort
    final build, so `--surface` still runs after that as well. A reviewer
    measured the result: `lifecycle: ready, model_state: finalized,
    build_in_progress: False` beside `/render/revision` saying `live: true`.
    `finalized` is the word iOS maps to **Saved**. The ninety-second lie, from
    the common door.

    So this is applied to EVERY stopped arm instead of owning one.

    `LIFECYCLE_FINALIZING`, NOT merely `build_in_progress: True`. iOS reaches
    "Improving" from `finalizing` AND `buildInProgress == true`; `finalized`
    with `buildInProgress` still renders "Saved", so flipping the boolean
    alone would have fixed nothing the wearer can see.

    `finalizing` IS NOT A CLAIM THAT THE WALK WENT WELL. It is a claim that a
    process is working, which is true here whatever the capture did. What
    happened to the capture is not erased: the settled arm's evidence and its
    reason are both carried into the composed strings, so an `error` walk
    still says `error` and still carries its detail. The settled word comes
    back on its own as soon as the stage finishes.

    `finalization` travels untouched by way of `**base`: it is the builder's
    record of the final SOLVE, and the photographic stages are a different
    fact.
    """
    photographic = photographic or {}
    photo_state = photographic.get("state")
    # THE BLOCK TRAVELS ON EVERY ANSWER, settled or not. T3 was not that the
    # Tower computed the wrong photographic state -- it never computed one at
    # all, and `session.stages` was read by `world_finish_pending.py` and by
    # nothing on the wire. A phone cannot tell "the photographic build
    # failed" from "the photographic room is ready" unless somebody sends it
    # the difference.
    base = {**base, "photographic": photographic} if photographic else base

    if building is None:
        # T2 AND T4, AND THEY ARE THE SAME FIX. `building` is a present-tense
        # fact about a PROCESS and it is false in both of the gaps between
        # the photographic stages; `photo_state` is a settled fact about the
        # WORLD and it is true across them. A world that still owes a
        # photographic room is not "Saved" merely because no process happens
        # to be mid-stage in the instant this poll arrived.
        if not is_unsettled(photo_state):
            if unobservable is None:
                return base
            # A probe broke but the record is settled enough to answer
            # without it (`complete`, `failed`, `never_recorded`). Carry the
            # caveat and keep the settled word: promoting THIS to Improving
            # is how a broken probe would park 165 historical worlds on
            # "Improving" forever, which the brief forbids as plainly as it
            # forbids the false "Saved".
            return {
                **base,
                "build_in_progress": None,
                "build_in_progress_unavailable_reason": unobservable,
            }
        owed_reason = photographic.get("detail") or "the photographic build is unfinished"
        from tower.world_builder.photographic import (  # noqa: PLC0415
            PHOTOGRAPHIC_RUNNING,
        )

        if photo_state == PHOTOGRAPHIC_RUNNING and "scope" in photographic:
            # `running` IS A LIVE PROCESS, WHOEVER'S IT IS (C1 E13). The word
            # is only ever `running` on evidence of a live pid, but the
            # present-tense probe above reads the ROOM's stage files alone, so
            # a running AREA build (`scope: "area"`, its status files under
            # `areas/<session>/<area>/`) arrived here with `building` None and
            # was reported `build_in_progress: False` -- "Finalizing" with
            # "the Tower does not report whether a build is running" on the
            # phone. The word and the boolean now cannot disagree.
            #
            # ONLY FOR A SESSION WITH COMPONENTS (review V8, LOW-c). `scope` is
            # on the word exactly when the session has a components record
            # (`components.combine_photographic`); a `components: null`
            # session -- every saved world today -- gets the answer below,
            # exactly as before this branch existed (contract §7 rule 1, §3.4).
            area = photographic.get("scope") == "area"
            return {
                **base,
                "state": LIFECYCLE_FINALIZING,
                "evidence": f"{base['evidence']}, and {owed_reason}",
                "reason": (
                    "the room is saved and an area of this walk is still being "
                    "built: " + owed_reason
                    if area else
                    "the photographic build for this world is still running: "
                    + owed_reason
                ),
                "build_in_progress": True,
                "build_in_progress_unavailable_reason": None,
            }
        return {
            **base,
            # NOT merely `build_in_progress`. iOS reaches "Improving" from
            # `finalizing`; `finalized` with the boolean set still renders
            # "Saved", so flipping the boolean alone would fix nothing a
            # wearer can see. The same note the `building` arm below makes.
            "state": LIFECYCLE_FINALIZING,
            "evidence": f"{base['evidence']}, and {owed_reason}",
            "reason": (
                # NOT "the Tower finishes this at its next start". It
                # usually does -- and `tower/main.py` refuses to spawn the
                # finisher at all when TOWER_WORLD_FINISH_PENDING, the
                # surface or the solve is switched off, and `config.py`
                # already warns that such a world "stays sparse until
                # somebody runs scripts/world_finish_pending.py by hand".
                # A reviewer caught the first version promising a rescue
                # that, on those Towers, is never coming. The world really
                # is unfinished, so "Improving" is the honest word; the
                # promise attached to it was the part that was not.
                "this world does not have the photographic representation it "
                "is owed yet: " + owed_reason + ". A Tower with the "
                "photographic stages enabled finishes owed work the next time "
                "it is idle; one that has them switched off will not, and this "
                "world then stays as it is until somebody runs "
                "scripts/world_finish_pending.py by hand"
                if photo_state == PHOTOGRAPHIC_OWED else
                "whether the photographic build for this world is still "
                "running could not be determined, so this world is not "
                "reported as finished: " + owed_reason
            ),
            # `False` for OWED (nothing IS in progress, and saying otherwise
            # would be a second lie), `None` for UNOBSERVABLE (not known).
            "build_in_progress": (
                None if photo_state == PHOTOGRAPHIC_UNOBSERVABLE else False
            ),
            "build_in_progress_unavailable_reason": (
                unobservable or (
                    owed_reason
                    if photo_state == PHOTOGRAPHIC_UNOBSERVABLE else None
                )
            ),
        }
    settled = base.get("reason")
    return {
        **base,
        "state": LIFECYCLE_FINALIZING,
        # NOT "a live process holds the writer lock". It does not. The
        # settled evidence is kept and the live fact is appended, so the
        # sentence names the file and the pid actually being relied on.
        "evidence": f"{base['evidence']}, and {building}",
        "reason": (
            "the photographic build for this world is still running: the "
            "surface, appearance and dense stages run after the writer lock "
            "is released, and the finished world looks very different from "
            "this one"
            + (
                f". What is already recorded about this session: {settled}"
                if settled
                else ""
            )
        ),
        "build_in_progress": True,
        "build_in_progress_unavailable_reason": None,
    }


def _lifecycle(*, holder, stopped, session, geometry_current, has_manifest,
               has_session_geometry, has_readable_figures: bool = False,
               store=None, world_id: str | None = None,
               session_id: str | None = None) -> dict:
    """What the Tower can SEE about whether a world is being built.

    Two readings, composed. `_lifecycle_from_the_record` answers from the
    record, the lock and the tree -- everything that is already settled. Then,
    for a session that has stopped with no live lock, `_still_building`
    corrects it with the one fact the record cannot hold: whether a
    photographic stage is working on this session right now.

    THE CORRECTION IS SKIPPED ON EXACTLY TWO KINDS OF ARM, and the
    condition below says which:

    * `alive` -- the two arms that read a LIVE writer lock. A live lock is
      strictly better evidence about the same process, and `receiving`
      deliberately reports `build_in_progress: False` because "frames are
      arriving" is not "a build is running".
    * `not stopped` -- a session whose journal has no `session_stopped`. The
      shipped builder cannot produce a photographic stage for one, and
      inventing a word for a state nobody has evidence about is how the last
      three review rounds went.

    Everything else is a stopped session with no live writer, which is the
    whole window this correction exists for.
    """
    base = _lifecycle_from_the_record(
        holder=holder,
        stopped=stopped,
        session=session,
        geometry_current=geometry_current,
        has_manifest=has_manifest,
        has_session_geometry=has_session_geometry,
        has_readable_figures=has_readable_figures,
    )
    alive = holder is not None and holder["alive"]
    if alive or not stopped:
        return base
    building, unobservable = _photographic_build_evidence(
        store, world_id, session_id
    )
    # The settled half of the same question. `_photographic_build_evidence`
    # answers "is a process working right now" and names the pid, which is
    # the better sentence when one IS; `photographic_state` answers "does
    # this world still owe a photographic room", which is the question the
    # wearer is actually asking and the one that survives the gaps between
    # stages. Both, because each says something the other cannot.
    photographic = _photographic_state_or_none(
        store, world_id, session_id, session
    )
    return _still_building(base, building, unobservable, photographic)


def _owner_facing_detail(detail) -> str | None:
    """`finalization.detail` as `lifecycle.reason` may quote it, or None when there is none.

    THE DETAIL IS RAW DIAGNOSTIC TEXT (review V10, MED-5). Its writers put exception text in it --
    `"{type(exc).__name__}: {exc}"`, `"final solve failed: ..."`, the gate's raw depth failure -- and
    `lifecycle.reason` becomes the phone's `model_state_reason`, shown word for word on an
    interrupted session and sent on the unauthenticated `/ws`. Quoted raw, it carried a
    `C:\\Users\\<user>\\...` path to both. So it is quoted as
    `coherence_publish.owner_facing_detail` makes it: one line, no path, no traceback, and no
    exception class name -- an owner reads "CUDA out of memory", not "RuntimeError: ..." (and the
    phone's own guard, `WorldTowerText` at Mac 3bb4431, would swap a sentence holding a class name
    for a generic one). A detail with none of those is quoted exactly as before."""
    if not detail:
        return None
    from tower.world_builder.coherence_publish import owner_facing_detail  # noqa: PLC0415

    return owner_facing_detail(detail) or None


def _client_safe_finalization(finalization):
    """The session's `finalization` record as `lifecycle.finalization` carries it on `/ws`: a COPY whose
    `detail` is `coherence_publish.client_safe_detail` of the record's (review V10, MED-5 -- one line,
    no path; the exception class stays, this is the diagnostic field). The record on disk is not
    touched, and a record whose detail has nothing to scrub is returned as it is, the same object."""
    if not isinstance(finalization, dict) or not isinstance(finalization.get("detail"), str):
        return finalization
    from tower.world_builder.coherence_publish import client_safe_detail  # noqa: PLC0415

    safe = client_safe_detail(finalization["detail"])
    return finalization if safe == finalization["detail"] else dict(finalization, detail=safe)


def _lifecycle_from_the_record(*, holder, stopped, session, geometry_current,
                              has_manifest, has_session_geometry,
                              has_readable_figures: bool = False) -> dict:
    """What the RECORD, the lock and the tree say about this session.

    Everything here is settled: it reads what is already on disk. The one
    present-tense question -- is a photographic stage working on this session
    right now -- belongs to `_still_building`, which corrects this answer on
    every stopped arm. Callers want `_lifecycle`.

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
        stopped, finalization complete, tree    -> ready
        stopped, finalization complete, no tree -> interrupted
        stopped, a manifest but no tree         -> interrupted
        stopped, neither                        -> stopped_unbuilt
        stopped, a tree no manifest describes   -> ready, currency unknown
        stopped, no finalization record (older
          builder), geometry current / behind   -> ready / stopped_unbuilt

    `interrupted` is deliberately NOT `failed`. On the 09-06 walk the
    process died mid-walk and left 463 keyframes of geometry; the state
    describes the session, and the geometry block beside it describes
    what exists.

    `has_manifest` AND `has_session_geometry` ARE DIFFERENT QUESTIONS, and
    conflating them cost a review round. `has_manifest` is the WORLD's
    `derived/manifest.json`, already discarded by the caller when it names
    another session -- a world walked twice has one manifest, describing
    whichever session built last. `has_session_geometry` is
    `derived/<sid>/poses.json` and `points.json`: the files the phone
    actually opens, which every built session has.

    So "is this session's geometry current" is `has_manifest`, and "is
    there anything here to open" is `has_session_geometry`. A guard that
    asks the first when it means the second calls every older session of a
    multi-session world unopenable -- measured, on two synthetic sessions
    in one world with both trees on disk.
    """
    finalization = _client_safe_finalization(session.finalization)
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
        # AND THE GEOMETRY IT IMPLIES EXISTS. A repair whose build failed
        # can leave `complete / solved` on a world whose derived tree is
        # gone, and the first version of this returned READY without ever
        # asking -- an adversarial review reached it end-to-end. `ready` on
        # a world with nothing to open is the same class of lie as
        # "Nothing mapped yet" over 26,634 points, pointing the other way.
        #
        # `has_session_geometry`, NOT `has_manifest`. The first version of
        # this guard asked the manifest, which is the world's and names
        # only the session that built last -- so it called every OLDER
        # session of a multi-session world unopenable, with both derived
        # trees sitting on disk. See the docstring.
        and has_session_geometry
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
        detail = _owner_facing_detail((finalization or {}).get("detail"))
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
        detail = _owner_facing_detail(finalization.get("detail"))
        return {
            "state": LIFECYCLE_INTERRUPTED,
            "evidence": (
                f"session_stopped was written and the finalization record is "
                f"{finalization.get('state')!r} with no process holding the lock"
            ),
            "reason": (
                "finalization did not complete; the geometry stored is the last "
                "build that finished"
                + (f": {detail}" if detail else "")
            ),
            "build_in_progress": False,
            "build_in_progress_unavailable_reason": None,
            "finalization": finalization,
        }
    if finalization is not None:
        # A finished record from a builder that keeps the lock through
        # finalization: the lock is gone because it was RELEASED, and the
        # record says the build completed.
        if has_session_geometry:
            # AND the geometry it implies exists. This branch did not
            # check, while the `error`/`interrupted` branch above -- added
            # by the same campaign -- did, so a normally-stopped world
            # whose derived tree had gone still read READY. The comment
            # above claimed the hole was "noted in the handoff"; it was
            # not, and an audit of the handoff caught the claim rather
            # than the hole.
            #
            # The first fix asked `has_manifest` here too, and a reviewer
            # measured what that does to a world walked twice: the older
            # session read `interrupted`, over a reason saying its
            # geometry was "no longer on disk", with `poses.json` right
            # there. Worse than the hole it closed -- that one lied about
            # a world with nothing in it; this lied about an intact one
            # and invited the wearer to redo the walk.
            return {
                "state": LIFECYCLE_READY,
                "evidence": (
                    "session_stopped was written, the finalization record is "
                    "complete, the lock was released and the derived tree is "
                    "there"
                ),
                "reason": None,
                "build_in_progress": False,
                "build_in_progress_unavailable_reason": None,
                "finalization": finalization,
            }
        # AND IT IS NOT ENOUGH TO REFUSE READY; SOMETHING TRUE HAS TO BE
        # SAID INSTEAD.
        #
        # The first version of this guard let the refusal fall through to
        # `stopped_unbuilt`, which is what an unbuilt world is. A reviewer
        # took the fall-through end to end and found it renders on the
        # phone as a PERMANENT "Finalizing": `stopped_unbuilt` maps to
        # `model_state: finalizing`, and `WorldPresentation` reads that
        # stage as `isStillChanging == true` while suppressing the
        # sentence that would explain it. Nothing is running and nothing
        # ever will be. Its reason -- "no geometry has been built for this
        # session yet" -- was also simply false: a build ran and finished,
        # and its output is gone. "yet" is the future tense of something
        # already past.
        #
        # `stopped_unbuilt` also hard-codes `"finalization": None`, which
        # was harmless while no finalised session could reach it. The only
        # evidence that the solve completed would have been deleted on the
        # way to the phone.
        #
        # So this returns what the sibling branch returns for the same
        # physical condition -- INTERRUPTED, which iOS renders as "Needs
        # retry": settled, not-changing, and carrying its reason. The
        # record travels with it.
        return {
            "state": LIFECYCLE_INTERRUPTED,
            "evidence": (
                "session_stopped was written and the finalization record is "
                "complete, but there is no derived tree to open"
            ),
            "reason": (
                "this world was finished and its geometry is no longer on "
                "disk; the capture is still there and it can be rebuilt"
            ),
            "build_in_progress": False,
            "build_in_progress_unavailable_reason": None,
            "finalization": finalization,
        }
    if has_manifest and not has_session_geometry:
        # A MANIFEST PROVES A BUILD RAN. Saying "no geometry has been built
        # for this session yet" over one is the same false sentence the
        # branch above was fixed for, and it projects to the phone as a
        # permanent `finalizing` -- now rendered as "it usually takes a few
        # minutes… worth waiting for Saved", forever. A reviewer reached it
        # through the ordinary case: `finalization is None` is what every
        # offline caller leaves, so the branch above never fires for them.
        return {
            "state": LIFECYCLE_INTERRUPTED,
            "evidence": (
                "a manifest describes a build for this session and its "
                "derived tree is not there"
            ),
            "reason": (
                "this session was built and its geometry is no longer on "
                "disk; the capture is still there and it can be rebuilt"
            ),
            **_BUILD_UNOBSERVABLE,
            "finalization": finalization,
        }
    if not has_session_geometry:
        # NOTHING WAS BUILT, which is a different statement from
        # `stopped_unbuilt`'s other meaning ("built, and behind"). Both
        # keep this state name -- it is on the wire and iOS decodes it --
        # and they are told apart in the PROJECTION, where the difference
        # actually matters. See `_attach_ios_projection`.
        #
        # `has_session_geometry`, NOT `has_manifest`, for the same reason as
        # the two branches above -- and this one was left behind when they
        # were fixed. A reviewer walked it: a world built twice, both
        # derived trees on disk, an OLDER session with no finalization
        # record, and the producer said `stopped_unbuilt` / "no geometry has
        # been built for this session yet" / a permanent `finalizing` on the
        # phone. Verbatim the outcome the comment twenty lines up spends a
        # paragraph saying must never happen again, reached by a different
        # door.
        #
        # `finalization is None` is not an exotic state either:
        # `stop_session` writes the block only when `hold_lock=True`, and
        # `hold_lock` DEFAULTS to False -- so any caller taking the default
        # arrives here. Not "every offline caller", which is what this
        # said until a reviewer checked: `scripts/world_build_session.py`,
        # the offline driver that actually ships, passes `hold_lock=True`
        # at both of its stop sites. Research and benchmark scripts take
        # the default.
        # `finalization`, not `None`. It is provably None on every path
        # that reaches here -- the block above returns on both arms -- so
        # this is a no-op today and a reviewer said so. It is spelled this
        # way because the two sibling `stopped_unbuilt` returns below said
        # `None` outright, and three identical situations spelled two ways
        # is how the next person picks the wrong one.
        return {
            "state": LIFECYCLE_STOPPED_UNBUILT,
            "evidence": "capture ended and no build output exists for this session",
            "reason": (
                "this walk produced no geometry; the capture is still there "
                "and it can be rebuilt"
            ),
            **_BUILD_UNOBSERVABLE,
            "finalization": finalization,
        }
    if not has_manifest:
        # A DERIVED TREE THAT NO MANIFEST DESCRIBES, AND THE ONE HONEST
        # WORD LEFT.
        #
        # Two ways in, and an earlier version of this branch told the
        # first one's story about both. Either no manifest FILE names this
        # session -- a world built before `write_derived` wrote one per
        # session -- or a file is there and cannot be used: unreadable
        # bytes, a top-level list, a schema version from the future, a
        # required key set to null. `_validate_manifest` was extracted so
        # that a corrupt manifest gives a clean refusal, and the refusal
        # was then laundered into a confident wrong sentence about a
        # manifest "naming another session". A reviewer built all four
        # corrupt shapes and got that sentence for every one of them.
        #
        # Neither case can judge currency, and the difference does not
        # change what to do, so both get this state -- and the evidence
        # below describes the ONLY thing both actually establish.
        #
        # `ready` OVERCLAIMS, and a reviewer said so with a measurement: a
        # SECOND, unrelated session building is what flips this session
        # from "a rebuild is outstanding" to `ready`, without anything
        # about this session changing. The answer it replaced overclaimed
        # in the other direction -- `stopped_unbuilt`, "no geometry has
        # been built", projected to the phone as a permanent `finalizing`
        # over a reconstruction sitting on disk. Between two overclaims,
        # the one that lets a wearer open a real world wins, and the
        # uncertainty is SAID rather than hidden.
        #
        # AND "THE WORLD ITSELF OPENS NORMALLY" WAS NOT TRUE WHEN THIS
        # BRANCH FIRST CLAIMED IT. The geometry and trajectory blocks
        # reported `available: false, element_count: null`, and iOS decides
        # what to draw from those numbers, so a complete reconstruction --
        # 1,347 points, 4 camera poses, served 200 by the geometry route --
        # rendered as "Needs retry: nothing usable came of this session".
        # The figures are recovered from the files now
        # (`_figures_from_the_tree`), which is what makes the sentence
        # true; `has_readable_figures` says whether that worked.
        #
        # Worlds built from now on do not reach this branch at all.
        return {
            "state": LIFECYCLE_READY,
            "evidence": (
                "capture ended and this session has a derived tree that no "
                "manifest describes -- either none was written beside it or "
                "the one that was cannot be read"
            ),
            "reason": (
                "no manifest describes this session's geometry, so how "
                "current it is cannot be judged here; its figures were "
                "counted from the poses and points themselves and the "
                "world opens"
                if has_readable_figures
                else (
                    "no manifest describes this session's geometry, and "
                    "counting the poses and points themselves found "
                    "nothing to show; the capture is still there and it "
                    "can be rebuilt"
                )
            ),
            **_BUILD_UNOBSERVABLE,
            "finalization": finalization,
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
            "finalization": finalization,
        }
    return {
        "state": LIFECYCLE_READY,
        "evidence": "capture ended and the stored geometry matches the keyframes",
        "reason": None,
        "build_in_progress": None,
        "build_in_progress_unavailable_reason": _BUILD_UNOBSERVABLE_REASON,
        "finalization": finalization,
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
    rejected_wrong_size = 0
    rejected_malformed = 0
    # `tracking.recovery` (WORLD-BUILDER-COMPONENTS.md s6), folded in the
    # same single pass. It holds a fixed-arity block, never events.
    recovery = RecoveryJournalReader()
    for event in events:
        kind = event.get("kind")
        recovery.feed(event)
        if kind == "frame_rejected":
            # ONLY the rejections the engine journals -- an ordinary
            # rejected frame writes no event (engine.py `observe`), and
            # this does not pretend otherwise. What IS journaled is the
            # kind worth a sentence on the phone: a frame of a size this
            # session is not calibrated for. A reviewer drove seven of
            # eight frames into that rejection and found no trace of it
            # in the live payload, the session record or the follower's
            # log; a whole walk at the wrong rung read "Mapping" with a
            # frozen keyframe count and then "Saved" with a truncated
            # world.
            # The reason rides in the event's PAYLOAD -- `WorldEvent` is
            # `{event_id, kind, at, payload}` -- which the first version
            # of this read at the top level and counted nothing.
            # Two SCALARS, not a dict keyed by reason. The summary is
            # cached for as long as anyone is subscribed and this
            # function's contract -- pinned by
            # `test_the_journal_cache_holds_a_summary_not_the_journal` --
            # is fixed size and scalars only. The engine journals exactly
            # two rejection kinds, so two counters lose nothing.
            payload = event.get("payload")
            reason = payload.get("reason") if isinstance(payload, dict) else None
            if reason == "frame_size_changed":
                rejected_wrong_size += 1
            elif reason == "malformed_frame":
                rejected_malformed += 1
            continue
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
    summary = {
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
        "frames_rejected_wrong_size": rejected_wrong_size,
        "frames_rejected_malformed": rejected_malformed,
    }
    # ONLY when the journal records a relocalizer: every older journal keeps
    # exactly the summary it had (and its scalar-only shape). The block is
    # fixed-arity -- its size does not grow with the session.
    if recovery.recorded:
        summary["recovery"] = recovery.block()
    return summary


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
        # LIVE, from the journal, and only for the rejections the engine
        # journals. `rejected_by_reason` below is the session record's
        # full tally and is final-only by design; this is the subset a
        # wearer needs to hear about while still walking. Additive.
        "frames_rejected_wrong_size": events.get("frames_rejected_wrong_size", 0),
        "frames_rejected_malformed": events.get("frames_rejected_malformed", 0),
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

    `recovery` is the live look-back relocalizer's block
    (WORLD-BUILDER-COMPONENTS.md s6.2), or None when the journal records no
    relocalizer -- every session today, and every older world. None means
    "not recorded", never "no losses". It carries no volatile field, so an
    unchanged episode never moves `revision`, and a new prompt always does.
    """
    last = events["last_tracking"]
    recovery = events.get("recovery")
    if last == "tracking_lost":
        return {
            "state": TRACKING_LOST,
            "evidence": "the most recent tracking event was tracking_lost",
            "limited_ever_reported": False,
            "recovery": recovery,
        }
    if last == "keyframe_accepted":
        return {
            "state": TRACKING_GOOD,
            "evidence": "the most recent tracking event was keyframe_accepted",
            "limited_ever_reported": False,
            "recovery": recovery,
        }
    return {
        "state": TRACKING_UNKNOWN,
        "evidence": "no keyframe has been accepted and no loss recorded",
        "limited_ever_reported": False,
        "recovery": recovery,
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


def _geometry_block(manifest, current: bool, keyframes_now, *,
                    has_session_geometry: bool = False,
                    tree_figures=None) -> dict:
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
    if manifest is not None and not has_session_geometry:
        # A MANIFEST IS NOT GEOMETRY. It describes a build; the poses and
        # points are the build. When the tree is gone the figures in it are
        # a memory, and reporting them as live put `available: true,
        # current: true, element_count: 26,634` in the same payload as a
        # lifecycle reason saying the geometry is no longer on disk -- with
        # the route answering 404. A reviewer printed both halves of that
        # payload side by side. It is the "Nothing mapped yet over 26,634
        # points" failure this campaign is named for, inverted.
        return _geometry_unavailable(
            "a build ran for this session and its poses and points are no "
            "longer on disk; the capture is still there and it can be "
            "rebuilt"
        )
    if manifest is None:
        if has_session_geometry:
            # A BUILD DID RUN AND NO MANIFEST DESCRIBES IT. A world built
            # before `write_derived` wrote one per session, or one whose
            # manifest is corrupt.
            #
            # "The counts genuinely cannot be given from here" is what a
            # previous version of this comment said, and it was wrong.
            # They come from the manifest, which is a SUMMARY of poses.json
            # and points.json -- and those are on disk, so the summary can
            # be recomputed. Until it was, this returned `available: false,
            # element_count: null` over a real reconstruction, and iOS
            # (`WorldEvidence.hasGeometry`) drew "Needs retry -- nothing
            # usable came of this session" on top of 1,347 points and 4
            # camera poses. Measured, not reasoned about.
            if tree_figures is not None:
                manifest = tree_figures
            else:
                # The files are there and cannot be READ. Different from
                # "there are none", and reported as itself rather than as
                # a count of zero.
                return _geometry_unavailable(
                    "this session has a derived tree and neither its poses "
                    "nor its points could be read, so nothing here can "
                    "summarise it; the geometry route reads the same files"
                )
        else:
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
                # None on every manifest read from a file. Set only when
                # these figures were counted from the tree, where there is
                # no `built_at` and no digest to move the revision when a
                # rebuild lands on the same counts.
                "tree": manifest.get("tree_fingerprint"),
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
        "stale_reason": _stale_reason(
            current,
            manifest,
            (
                "keyframes have been accepted since this build ran; these "
                "figures are correct for the keyframes named in "
                "built_from_keyframes and are not the final world"
            ),
            (
                "these figures were counted from the poses and points on "
                "disk because no manifest describes them, so nothing here "
                "can say which keyframes produced them or whether a "
                "rebuild is outstanding"
            ),
        ),
    }


def _stale_reason(current, manifest, behind: str, unjudgeable: str):
    """Why these figures are not the final world -- and which "not".

    `current` is False in two quite different situations and the block it
    came from used to give the first sentence for both. "Behind" is a
    build that is genuinely older than the keyframes, which is what the
    first message describes. "Unjudgeable" is a build with no manifest,
    where currency is not false but UNKNOWN: telling a wearer keyframes
    have arrived since a build ran, when nothing here knows when it ran,
    is a fabrication of the same family as the one this whole campaign is
    about.
    """
    if current:
        return None
    if manifest.get("input_digest") is None and manifest.get("built_at") is None:
        return unjudgeable
    return behind


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
    # A re-redacted keyframe set the session was switched to
    # (`world_reredact.py --apply`), reported beside the stored one rather
    # than in place of it: `redaction` stays what the SESSION recorded, and
    # `images/` is still on disk. Null when builds read `images/`.
    reredacted_set = None
    if session_id is not None:
        try:
            image_set = store.keyframe_image_set(world_id, session_id)
        except Exception:  # noqa: BLE001 -- a report, never a failure
            image_set = None
        if image_set is not None and image_set.reredacted:
            try:
                set_count = sum(1 for _ in image_set.directory.glob("*.jpg"))
            except OSError:
                set_count = None
            reredacted_set = {"redaction": image_set.redaction, "count": set_count}
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
            "reredacted_set": reredacted_set,
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


def _figures_are_drawable(figures) -> bool:
    """Whether a recount produced something a viewer could draw.

    Not `figures is not None`, which asks only whether the files parsed.
    The same distinction `_has_drawable_geometry` makes of the payload,
    made of the figures before they become one.
    """
    if figures is None:
        return False
    return (figures.get("points") or 0) > 0 or (
        figures.get("poses_positioned") or 0
    ) > 0


def _figures_from_the_tree(files, store, world_id: str, session_id: str):
    """Count the poses and points THEMSELVES, when no manifest can say.

    **This is the fix for a defect this campaign itself introduced.** A
    session whose derived tree is on disk with no manifest describing it
    -- a world built before `write_derived` wrote one per session, or one
    whose manifest is corrupt -- was given `lifecycle: ready` and a
    geometry block reading `available: false, element_count: null`. The
    branch's own comment promised "the world itself opens normally". It
    does not: iOS decides what to draw with `WorldEvidence.hasGeometry`,
    which is `(elements ?? 0) > 0 || (poses ?? 0) > 0`, so a *complete*
    reconstruction rendered as **"Needs retry -- nothing usable came of
    this session. Walking the space again is what produces another one."**
    Measured on a real build: 1,347 points and 4 poses on disk, and that
    sentence over them. That is the exact failure this campaign is named
    for, reintroduced through a branch written to fix it.

    A manifest is meant to be a summary of these files, so when the
    summary is missing, counting them is neither a guess nor a
    fabrication -- it is the same arithmetic the build did, done later.
    What CANNOT be recovered is which keyframes produced them, so
    `keyframes` and `input_digest` stay `None` and currency stays
    unjudgeable; the caller says so in words.

    "Meant to be", because on this Tower's own disk two of 49 sessions
    disagree: `fcbca9e9…/158ef0ef…`'s manifest counts 463 poses where
    `poses.json` holds 467, so the manifest and the file it sits beside
    describe different builds. **Neither is reachable through this
    function** -- both carry a usable manifest, so the recount never
    runs -- and which of the two is right is a question about
    `engine.build`, not about this. Recorded because a reviewer measured
    it and because the sentence above would otherwise read as a
    guarantee.

    Returns None when the files cannot be read, which is a different
    answer from "there are none" and is reported differently.

    **A MANIFEST THIS BUILD CANNOT USE IS NOT A REASON TO REFUSE THE
    FILES, and a round of this campaign spent a fix believing it was.**
    `_validate_manifest` refuses a `schema_version` it does not know, on
    the sound ground that such a manifest "describes fields whose meaning
    this build does not know" -- and the first version of this recount
    read that as evidence the POSES AND POINTS were also unreadable, and
    refused. A reviewer built the state and showed what that produces: the
    status channel reporting `element_count: null` and the phone drawing
    *"Needs retry -- nothing usable came of this session"* over a tree the
    geometry route was serving 200 for at the same moment, because
    `world_builder_geometry._read` uses `read_derived(verify=False)` and
    never looks at `schema_version` at all.

    One refusal in one of the two readers is worse than none: it is the
    Tower disagreeing with itself, which is the failure this campaign is
    named for. The manifest's FIGURES are refused -- they are what the
    schema version is about -- and the files are counted, which is what
    both readers already do.

    Both reads go through `_FileCache`, so the poll loop parses these
    files once per change rather than once per second. poses.json is the
    larger of the two only in pose count; points.json is the megabytes,
    and only its LENGTH is retained -- the rows are dropped before the
    cache stores anything.
    """
    derived = store.derived_dir(world_id) / session_id
    poses_path = derived / "poses.json"
    points_path = derived / "points.json"
    poses = files.read(
        "pose-summary", poses_path, lambda: _summarise_pose_rows(poses_path)
    )
    points = files.read(
        "point-count", points_path, lambda: _count_rows(points_path, "points")
    )
    if poses is None or points is None:
        return None
    return {
        "points": points,
        # Unknowable from the tree, and left unknown rather than guessed.
        # `built_from_keyframes: null` beside a real `element_count` is a
        # precise statement: here is what was built, and nothing here can
        # say what it was built from.
        "keyframes": None,
        "input_digest": None,
        "built_at": None,
        "backend_id": None,
        "scale_state": None,
        # An opaque change detector. `built_at` and `input_digest` are
        # what normally make a revision move, and neither exists here, so
        # a rebuild that happened to produce the same counts would leave
        # the revision unchanged and the phone would not redraw. The
        # files' own (size, mtime_ns) is the fingerprint `_FileCache`
        # already trusts for exactly this question.
        "tree_fingerprint": _fingerprint(poses_path, points_path),
        **poses,
    }


def _fingerprint(*paths) -> str | None:
    parts = []
    for path in paths:
        try:
            stat = path.stat()
        except OSError:
            return None
        parts.append(f"{stat.st_size}:{stat.st_mtime_ns}")
    return "|".join(parts)


def _count_rows(path, key: str):
    """`len(json[key])`, holding on to nothing else.

    points.json is the one genuinely large file here -- tens of thousands
    of rows -- and the count is all any caller of this wants. Parsing it
    to a list and returning the length lets the list be collected
    immediately; returning the list would put megabytes into a
    process-lifetime cache.
    """
    rows = _read_past_a_replace(path, key)
    return len(rows) if isinstance(rows, list) else None


def _read_past_a_replace(path, key: str):
    """Read a file that a builder may be replacing underneath us.

    **A WINDOWS MEASUREMENT, NOT A PRECAUTION.** `write_json_atomic`
    finishes with `os.replace`, and on Windows a replace onto a path a
    reader has open fails with WinError 5 -- and so, symmetrically, does
    the reader's `open()` during the writer's window. A reviewer ran a
    reader and a writer against one file for three seconds and measured
    **3,519 failures in 14,486 reads, 24%**, every one a `PermissionError`
    from the open rather than a torn parse.

    That matters here because this reads the LARGE file. Reported as
    "could not be read", a 24% failure rate at the publisher's 2 Hz makes
    `geometry.available` flicker between true and false while a session
    rebuilds, and the phone flickers with it.

    A replace is over in microseconds, so a couple of immediate retries
    cover it without a sleep in a poll path. What is still failing after
    them is a real fault and is reported as one.

    `ValueError` is NOT retried: a file that parsed and was wrong will
    parse and be wrong again.
    """
    last: OSError | None = None
    for _ in range(3):
        try:
            return read_json_closed(path)[key]
        except (KeyError, TypeError, ValueError):
            return None
        except OSError as exc:
            last = exc
    logger.debug("world builder: %s stayed unreadable: %r", path, last)
    return None


def _summarise_pose_rows(path):
    """`engine.build`'s own pose arithmetic, recomputed from poses.json.

    Deliberately identical to the counting in `engine.build`, including
    the rule `_pose_count` exists for: an ANCHOR is a real position only
    when something in its segment actually solved against it. A lone
    anchor in a segment that resolved nothing is an origin marker for an
    empty coordinate frame, and counting it is where "Camera poses: 36"
    over a world with `poses_solved: 0` came from on the 2026-08-24 walk.

    `refused` is everything that is neither solved nor an anchor -- the
    same definition `engine.build` uses at both of its counting sites.
    """
    rows = _read_past_a_replace(path, "poses")
    if not isinstance(rows, list):
        return None
    solved = anchors = refused = 0
    by_segment: dict = {}
    for row in rows:
        if not isinstance(row, dict):
            return None
        segment = row.get("segment_index")
        # Unhashable ids would raise inside the dict below, and a corrupt
        # file must read as unreadable rather than as an exception out of
        # a function whose contract is "None or a summary".
        if isinstance(segment, (list, dict)):
            return None
        bucket = by_segment.setdefault(segment, [0, 0])
        status = row.get("status")
        if status == POSE_STATUS_SOLVED:
            solved += 1
            bucket[0] += 1
        elif status == POSE_STATUS_ANCHOR:
            anchors += 1
            bucket[1] += 1
        else:
            refused += 1
    positioned = sum(
        solved_here + (anchors_here if solved_here else 0)
        for solved_here, anchors_here in by_segment.values()
    )
    return {
        "poses_solved": solved,
        "poses_refused": refused,
        "poses_anchor": anchors,
        "poses_positioned": positioned,
        "segments": len(by_segment),
    }


def _has_drawable_geometry(payload: dict) -> bool:
    """Whether this payload carries FIGURES a viewer could draw.

    Deliberately the same question `WorldEvidence.hasGeometry` asks on
    iOS -- `(elements ?? 0) > 0 || (poses ?? 0) > 0` -- and deliberately
    not `geometry.available`, which says only that a build ran and left a
    tree behind. A build that solved nothing leaves one too.

    Either figure alone is enough, for the reason iOS gives: a build can
    place cameras and recover few points, or recover points across
    segments whose cameras were never placed. Both are geometry.
    """
    geometry = payload.get("geometry") or {}
    trajectory = payload.get("trajectory") or {}
    elements = geometry.get("element_count")
    poses = trajectory.get("pose_count")
    return (elements or 0) > 0 or (poses or 0) > 0


def _attach_ios_projection(payload: dict) -> None:
    """Add `model_state` and `world_snapshot` -- the fields iOS decodes.

    Derived from the payload that is already built, so the projection and
    the evidence beside it cannot drift.
    """
    lifecycle = payload["lifecycle"]
    state = _MODEL_STATE_BY_LIFECYCLE.get(lifecycle["state"], MODEL_STATE_IDLE)
    if state == MODEL_STATE_FINALIZING and lifecycle["state"] == LIFECYCLE_STOPPED_UNBUILT:
        # `stopped_unbuilt` CARRIES TWO STATES AND ONLY ONE OF THEM MEANS
        # "WAIT".
        #
        # "Built, and behind" is a world that is intact and merely needs a
        # rebuild: `finalizing` is right, and the phone showing "Finalizing"
        # over it is right. "Nothing was built" is a walk that produced no
        # geometry, and telling a wearer to wait for it is the permanent
        # "Finalizing" four separate reviews found by four separate routes.
        #
        # The previous round fixed it by pointing the whole state at
        # `interrupted`, which a fifth reviewer showed was too broad -- the
        # behind-case started rendering a red "Interrupted ... what was
        # built before it stopped is here" over a complete world. The
        # distinction belongs here, where both facts are in hand, and NOT
        # in the state name, which is on the wire and which iOS decodes.
        #
        # One caveat, measured: `stop_session()` defaults to
        # `hold_lock=False`, so an offline caller can be building right now
        # with no lock to show for it, and such a session reads "Needs
        # retry" until its build lands. The live path holds the lock
        # through finalization and never reaches here.
        #
        # THE PREDICATE IS "IS THERE ANYTHING TO DRAW", NOT
        # "DID A BUILD RUN", and the first version of this fix used the
        # second. `geometry.available` is true as soon as a manifest and a
        # derived tree exist -- and `engine.build()` calls `write_derived`
        # UNCONDITIONALLY, so a walk down a dark corridor writes
        # poses.json, points.json and a manifest saying `points: 0,
        # poses_solved: 0`. Gating on `available` therefore kept saying
        # `finalizing` over a world with nothing in it: the permanent
        # "Finalizing" back through a different door, one round after it
        # was closed.
        #
        # iOS decides what to draw with `WorldEvidence.hasGeometry`, which
        # is `(elements ?? 0) > 0 || (poses ?? 0) > 0` -- the FIGURES, not
        # the flag. This must ask the same question of the same numbers,
        # or the Tower tells the wearer to wait for a screen the phone
        # will never have anything to put on. Reproduced with a
        # featureless capture: `available: true, element_count: 0,
        # pose_count: 0, model_state: "finalizing"`.
        if not _has_drawable_geometry(payload):
            state = MODEL_STATE_INTERRUPTED
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
    return _validate_manifest(store.read_derived_manifest(world_id), world_id)


def _validate_manifest(manifest, world_id, source="derived manifest"):
    """The store's rule, reached through this module's private name.

    The checks themselves moved to `WorldStore.validate_manifest` so that
    every reader of a manifest is held to them -- the status channel, the
    geometry route, `usable_placements` and the saved-worlds listing --
    rather than only this one. See that function for what the split
    produced.
    """
    return validate_manifest(manifest, world_id, source=source)


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


