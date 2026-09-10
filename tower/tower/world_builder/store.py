"""On-disk world storage.

Deliberately the same shape as tower/object_memory/store.py: atomic JSON
via temp+fsync+replace inside try/finally, append-only JSONL journals,
raw-dict rewrites, corrupt-line tolerance, and a real purge that removes
every artifact including temp files.

That store was chosen as the model over the numbered-checkpoint/HEAD/mmap
design in the readiness report for a concrete reason. Three Windows
behaviours were verified on this host:

    directory fsync           -> PermissionError [Errno 13]
    replace onto an OPEN dest -> PermissionError [WinError 5]
    unlink an mmap'd file     -> PermissionError [WinError 32]

The checkpoint design needs a validating-recovery path, a fallback policy,
and a retrying garbage collector purely to survive those. Two of the three
do not arise here at all: no directory is renamed and nothing is mmap'd.
The third is already solved by the lock this store inherits. V1 also has
no concurrent reader -- capture, build, and inspect are separate processes
-- so the isolation those subsystems buy has no consumer yet.

Derived output is a single rebuildable directory rather than numbered
checkpoints. Staleness is detected by comparing a digest of the inputs
that produced it, so a stale derived tree is detected rather than trusted.
"""

import hashlib
import json
import os
import time
import logging
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path

from tower.storage import (
    staging_path,
    TEMP_SUFFIX,
    append_jsonl,
    read_json_closed,
    read_raw_jsonl,
    write_json_atomic,
)

from tower.world_builder.records import (
    SegmentPlacement,
    Keyframe,
    KeyframeEdge,
    Session,
    World,
    keyframe_edge_from_json_dict,
    keyframe_from_json_dict,
    session_from_json_dict,
    world_from_json_dict,
)
from tower.world_builder.schema import (
    POSE_CONVENTION,
    SCHEMA_VERSION,
)

logger = logging.getLogger(__name__)

WORLD_FILENAME = "world.json"
SESSION_FILENAME = "session.json"
KEYFRAMES_FILENAME = "keyframes.jsonl"
EDGES_FILENAME = "edges.jsonl"
EVENTS_FILENAME = "events.jsonl"
LOCK_FILENAME = "LOCK"
DERIVED_DIRNAME = "derived"
DERIVED_MANIFEST = "manifest.json"
IMAGES_DIRNAME = "images"


# How many times `acquire_writer_lock` will lose the exclusive create
# before it gives up. Contention here is two processes reclaiming one
# dead lock, which resolves in one round; this is a bound, not a wait.
_LOCK_ACQUIRE_ATTEMPTS = 8

# How long an unreadable lock is given to become readable before it is
# treated as a writer that died between its create and its write. The write
# is one `json.dumps` and an `fsync`; a tenth of a second is four orders of
# magnitude more than that and still imperceptible to a wearer.
_LOCK_UNREADABLE_GRACE_S = 0.1
# A pause between attempts, so eight of them span long enough to outlast a
# peer's reclaim rather than burning through in a millisecond. Without it an
# adversarial review measured every loser of a natural race exhausting all
# eight attempts in 1.2 ms and reporting "contending" instead of naming the
# live holder -- 104 of 104 times.
_LOCK_RETRY_SLEEP_S = 0.02


class WorldStoreError(Exception):
    """Base for storage-layer failures."""


class UnsupportedSchemaError(WorldStoreError):
    """A persisted artifact declares a schema version we cannot read."""


class UnknownPoseConventionError(WorldStoreError):
    """A world declares a pose convention this build does not recognise.

    Raised rather than defaulted. A bare [x, y, z] read under the wrong
    convention does not fail loudly -- it yields a plausible, mirrored,
    wrong answer, and every downstream reference inherits the error.
    """


class WorldLockedError(WorldStoreError):
    """Another live process holds this world's writer lock."""


@dataclass(frozen=True)
class PurgeReport:
    """What a purge actually managed to delete.

    Carries `retained` because a purge that could not remove everything
    must never report success: 06-PRIVACY-DATA requires real deletion, and
    a false claim of deletion is worse than an honest failure.
    """

    removed: tuple[str, ...]
    retained: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.retained


def require_schema(data: dict, what: str) -> None:
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise UnsupportedSchemaError(
            f"{what} declares schema_version {version!r}, this build reads "
            f"{SCHEMA_VERSION}. Refusing rather than guessing at the meaning "
            "of fields written by a different version."
        )


def require_pose_convention(convention: dict) -> None:
    if convention != POSE_CONVENTION:
        differing = sorted(
            key
            for key in set(convention) | set(POSE_CONVENTION)
            if convention.get(key) != POSE_CONVENTION.get(key)
        )
        raise UnknownPoseConventionError(
            "world declares a pose convention this build does not recognise; "
            f"differing keys: {differing}. Refusing to interpret coordinates "
            "rather than silently reading them under the wrong convention."
        )






class WorldStore:
    """Filesystem storage for one root directory of worlds."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        # Guards journal appends (live path) against rewrite/purge, both
        # of which replace or unlink files out from under a concurrent
        # append -- PermissionError on Windows, silent line loss on POSIX.
        # Non-reentrant: locked public methods call `_locked` helpers, not
        # each other.
        self._lock = threading.Lock()

    # -- paths ---------------------------------------------------------

    @property
    def root(self) -> Path:
        return self._root

    def world_dir(self, world_id: str) -> Path:
        return self._root / "worlds" / world_id

    def world_path(self, world_id: str) -> Path:
        return self.world_dir(world_id) / WORLD_FILENAME

    def session_dir(self, world_id: str, session_id: str) -> Path:
        return self.world_dir(world_id) / "sessions" / session_id

    def session_path(self, world_id: str, session_id: str) -> Path:
        return self.session_dir(world_id, session_id) / SESSION_FILENAME

    def keyframes_path(self, world_id: str, session_id: str) -> Path:
        return self.session_dir(world_id, session_id) / KEYFRAMES_FILENAME

    def edges_path(self, world_id: str, session_id: str) -> Path:
        return self.session_dir(world_id, session_id) / EDGES_FILENAME

    def events_path(self, world_id: str, session_id: str) -> Path:
        return self.session_dir(world_id, session_id) / EVENTS_FILENAME

    def images_dir(self, world_id: str, session_id: str) -> Path:
        return self.session_dir(world_id, session_id) / IMAGES_DIRNAME

    def derived_dir(self, world_id: str) -> Path:
        return self.world_dir(world_id) / DERIVED_DIRNAME

    def lock_path(self, world_id: str) -> Path:
        return self.world_dir(world_id) / LOCK_FILENAME

    # -- worlds --------------------------------------------------------

    def list_world_ids(self) -> list[str]:
        worlds_root = self._root / "worlds"
        if not worlds_root.exists():
            return []
        return sorted(
            entry.name
            for entry in worlds_root.iterdir()
            if entry.is_dir() and (entry / WORLD_FILENAME).exists()
        )

    def write_world(self, world: World) -> None:
        with self._lock:
            write_json_atomic(self.world_path(world.world_id), world.to_json_dict())

    def read_world(self, world_id: str) -> World:
        path = self.world_path(world_id)
        if not path.exists():
            raise WorldStoreError(f"no world at {path}")
        data = read_json_closed(path)
        require_schema(data, f"world {world_id}")
        require_pose_convention(data["pose_convention"])
        return world_from_json_dict(data)

    # -- sessions ------------------------------------------------------

    def write_session(self, session: Session) -> None:
        with self._lock:
            write_json_atomic(
                self.session_path(session.world_id, session.session_id),
                session.to_json_dict(),
            )

    def read_session(self, world_id: str, session_id: str) -> Session:
        data = read_json_closed(self.session_path(world_id, session_id))
        require_schema(data, f"session {session_id}")
        return session_from_json_dict(data)

    def list_session_ids(self, world_id: str) -> list[str]:
        sessions_root = self.world_dir(world_id) / "sessions"
        if not sessions_root.exists():
            return []
        return sorted(
            entry.name
            for entry in sessions_root.iterdir()
            if entry.is_dir() and (entry / SESSION_FILENAME).exists()
        )

    # -- keyframes and edges -------------------------------------------

    def append_keyframe(self, world_id: str, keyframe: Keyframe) -> None:
        with self._lock:
            append_jsonl(
                self.keyframes_path(world_id, keyframe.session_id),
                keyframe.to_json_dict(),
            )

    def read_keyframes(self, world_id: str, session_id: str) -> list[Keyframe]:
        raw_records, _ = read_raw_jsonl(self.keyframes_path(world_id, session_id))
        return _parse_all(raw_records, keyframe_from_json_dict, "keyframe")

    def clear_edges(self, world_id: str, session_id: str) -> None:
        """Drop the edge journal before a rebuild recomputes it.

        Edges between keyframes are recomputed from the keyframes on every
        build, so despite living in an append-only journal they are
        derived. Appending without clearing duplicates the whole set per
        rebuild.
        """
        with self._lock:
            self.edges_path(world_id, session_id).unlink(missing_ok=True)

    def append_edge(
        self, world_id: str, session_id: str, edge: KeyframeEdge
    ) -> None:
        with self._lock:
            append_jsonl(self.edges_path(world_id, session_id), edge.to_json_dict())

    def read_edges(self, world_id: str, session_id: str) -> list[KeyframeEdge]:
        raw_records, _ = read_raw_jsonl(self.edges_path(world_id, session_id))
        return _parse_all(raw_records, keyframe_edge_from_json_dict, "edge")

    def append_event(self, world_id: str, session_id: str, event) -> None:
        with self._lock:
            append_jsonl(
                self.events_path(world_id, session_id), event.to_json_dict()
            )

    def read_events(
        self, world_id: str, session_id: str, after_event_id: int | None = None
    ) -> list[dict]:
        """Every event, or only those strictly newer than a cursor.

        The cursor is what turns this journal from an archive into a live
        stream: a reader keeps the last `event_id` it saw and asks for
        what came after. Because `event_id` is dense within a session, a
        gap in what comes back means an event was genuinely dropped -- so
        a viewer can tell it missed something rather than quietly showing
        an incomplete world.

        A record with no `event_id` is skipped when a cursor is given.
        There is no way to advance past it, so returning it would hand it
        back on every poll forever.
        """
        raw_records, _ = read_raw_jsonl(self.events_path(world_id, session_id))
        if after_event_id is None:
            return raw_records
        return [
            record
            for record in raw_records
            if isinstance(record.get("event_id"), int)
            and record["event_id"] > after_event_id
        ]

    # -- images --------------------------------------------------------

    def write_keyframe_image(
        self, world_id: str, session_id: str, filename: str, jpeg_bytes: bytes
    ) -> Path:
        """Persist one keyframe image, durably, BEFORE its journal line.

        The ordering is deliberate and asymmetric: a journal line pointing
        at a missing image is corruption, while an image with no journal
        line is a harmless orphan that purge and verification both sweep.
        Never invert this.
        """
        images = self.images_dir(world_id, session_id)
        images.mkdir(parents=True, exist_ok=True)
        path = images / filename
        # `staging_path`, not `name + TEMP_SUFFIX`: a staging name derived only
        # from the destination is shared by every writer of it. Keyframe
        # filenames are unique per session so a collision is unlikely here,
        # but "unlikely" is what the same pattern was called in
        # `write_bytes_atomic` before it was measured producing 656 torn
        # reads. One convention, one place.
        temp_path = staging_path(path)
        try:
            with temp_path.open("wb") as handle:
                handle.write(jpeg_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            temp_path.replace(path)
        finally:
            temp_path.unlink(missing_ok=True)
        return path

    # -- derived -------------------------------------------------------

    def derived_manifest_path(self, world_id: str) -> Path:
        return self.derived_dir(world_id) / DERIVED_MANIFEST

    def clear_derived(self, world_id: str) -> None:
        """Delete the whole derived tree.

        Safe by construction: every byte under derived/ is rebuildable
        from the authoritative journals plus the keyframe images. This is
        also the operation the "derived is genuinely derived" test drives.
        """
        derived = self.derived_dir(world_id)
        if derived.exists():
            shutil.rmtree(derived, ignore_errors=True)

    def write_derived_manifest(self, world_id: str, manifest: dict) -> None:
        write_json_atomic(self.derived_manifest_path(world_id), manifest)

    def read_derived_manifest(self, world_id: str) -> dict | None:
        path = self.derived_manifest_path(world_id)
        if not path.exists():
            return None
        try:
            data = read_json_closed(path)
        except json.JSONDecodeError:
            logger.warning("world builder: derived manifest unreadable at %s", path)
            return None
        return data

    def write_derived(
        self,
        world_id: str,
        session_id: str,
        *,
        poses,
        points,
        manifest,
        support=None,
    ) -> None:
        """Write the rebuildable reconstruction outputs.

        Poses and points are JSON rather than .npy: at V1 scale a session
        is hundreds of poses and low tens of thousands of points, JSON
        round-trips exactly without a dtype contract to get wrong, and it
        stays inspectable by eye. The trigger to switch to .npy is a world
        large enough that a viewer needs partial or memory-mapped reads --
        the same "small enough that rewriting wholesale is fine" reasoning
        object memory's store already documents for itself.

        `support` is the 2-D/3-D association -- which feature in which
        keyframe produced which point -- and goes in its OWN file rather
        than into a points.json row. points.json is the source for a live
        cross-platform wire contract (docs/contracts/
        WORLD-BUILDER-GEOMETRY.md) whose row shape is pinned by test; a
        second file costs a reader one open and costs that contract
        nothing. `None` writes no file at all, which is what every world
        built before this existed looks like on disk.
        """
        # Under the same lock as purge_world. Without it an in-flight
        # build can recreate a world directory that purge has just
        # reported as completely deleted -- and the resurrected tree is
        # then invisible to list_world_ids(), so no later purge targets it.
        with self._lock:
            derived = self.derived_dir(world_id) / session_id
            write_json_atomic(derived / "poses.json", {"poses": poses})
            write_json_atomic(derived / "points.json", {"points": points})
            if support is not None:
                write_json_atomic(derived / "support.json", {"support": support})
            write_json_atomic(self.derived_manifest_path(world_id), manifest)

    def read_derived(
        self, world_id: str, session_id: str, *, verify: bool = True
    ) -> dict | None:
        """Read the derived reconstruction, refusing stale output.

        The digest is checked by default rather than on request. Serving a
        derived tree that no longer matches the journal that produced it
        is silent corruption of exactly the kind this store is built to
        prevent: the numbers look fine, they are just answers to an older
        question. Returning None makes a stale tree indistinguishable from
        an absent one, which is the honest outcome -- both mean "rebuild".

        `support` is OPTIONAL and reads as None when the file is not
        there. It arrived after ~29 worlds were already on disk, and a
        reconstruction is complete without it -- it is an index into the
        reconstruction, not part of it. A missing support.json is
        therefore absent, never an error, and never a reason to refuse
        poses and points that are perfectly good.
        """
        if verify:
            digest = compute_input_digest(
                self.read_keyframes(world_id, session_id)
            )
            if not self.derived_is_current(world_id, digest):
                logger.warning(
                    "world builder: derived output for %s is stale; "
                    "treating as absent",
                    world_id,
                )
                return None
        derived = self.derived_dir(world_id) / session_id
        poses_path = derived / "poses.json"
        points_path = derived / "points.json"
        if not poses_path.exists() or not points_path.exists():
            return None
        try:
            return {
                "poses": read_json_closed(poses_path)["poses"],
                "points": read_json_closed(points_path)["points"],
                "support": self._read_support(derived),
            }
        except (json.JSONDecodeError, KeyError):
            logger.warning("world builder: derived output unreadable for %s", world_id)
            return None

    def write_placements(self, world_id: str, session_id: str, placements) -> None:
        """Persist where each segment sits, or why it does not sit anywhere.

        Its own file, for the same reason `support` has one: geometry is
        immutable once solved and PLACEMENT is not, so a later registration
        pass must be able to change where a segment sits without rewriting
        the points that decide its content hash. Folding placement into
        points.json would couple the two and make every re-placement look
        like new geometry.
        """
        with self._lock:
            derived = self.derived_dir(world_id) / session_id
            payload = {"placements": [p.to_json_dict() for p in placements]}
            # allow_nan=False: Python writes NaN and Infinity as bare
            # tokens that JSON.parse, Swift's JSONSerialization and Go's
            # encoding/json all reject. A file only this runtime can read
            # is not a wire artifact. Raising here leaves no file, because
            # the write is atomic.
            json.dumps(payload, allow_nan=False)
            write_json_atomic(derived / "placements.json", payload)

    def read_placements(self, world_id: str, session_id: str):
        """The placements, or None. Never raises, never refuses a read.

        Absent and unreadable are the same answer, exactly as for
        `support`: every world built before placements existed has no such
        file, and a reconstruction is complete without one. Refusing a
        world over a truncated index beside it would turn an optional file
        into a hard dependency by the back door.

        A row that fails its own invariants is dropped rather than
        poisoning the rest -- a placement that cannot be represented is
        one that must not be drawn, and the others are still good.
        """
        path = self.derived_dir(world_id) / session_id / "placements.json"
        if not path.exists():
            return None
        # Broad on purpose. An earlier version caught only JSONDecodeError
        # and KeyError, which covers a truncated file and nothing else --
        # `{"placements": null}`, a top-level list, a string, or a null row
        # are all VALID JSON and each raised straight through
        # build_manifest into an HTTP 500, taking poses and points that
        # were perfectly good down with an optional index beside them.
        # That is exactly what this method's contract exists to prevent,
        # so the contract is enforced rather than described.
        try:
            document = read_json_closed(path)
            rows = document["placements"]
            if not isinstance(rows, list):
                raise TypeError(f"placements must be a list, got {type(rows)}")
        except Exception:
            logger.warning(
                "world builder: placements unreadable at %s; treating as "
                "absent",
                path,
            )
            return None
        kept = []
        for row in rows:
            try:
                kept.append(SegmentPlacement.from_json_dict(row))
            except Exception:
                logger.warning(
                    "world builder: dropping unrepresentable placement in "
                    "%s: %r",
                    path,
                    row,
                )
        return kept

    def _read_support(self, derived: Path):
        """The association, or None. Never raises, never refuses a read.

        Unreadable is treated the same as missing on purpose. Poses and
        points do not become wrong because an index beside them is
        truncated, and refusing the whole derived tree over it would turn
        an optional file into a hard dependency by the back door -- which
        is precisely what keeping it out of points.json was meant to
        avoid. Logged, because a corrupt file is still worth knowing about.
        """
        path = derived / "support.json"
        if not path.exists():
            return None
        try:
            support = read_json_closed(path)["support"]
            # Shape-checked, not just parsed. A top-level list raised
            # TypeError straight out of a method whose docstring promises
            # it never raises, and a string was returned AS the support
            # table -- which cross-segment registration then consumes.
            if not isinstance(support, list):
                raise TypeError(
                    f"support must be a list, got {type(support)}"
                )
            return support
        except Exception:
            logger.warning(
                "world builder: support association unreadable at %s; "
                "treating as absent",
                path,
            )
            return None

    def derived_is_current(self, world_id: str, input_digest: str) -> bool:
        manifest = self.read_derived_manifest(world_id)
        if manifest is None:
            return False
        return (
            manifest.get("schema_version") == SCHEMA_VERSION
            and manifest.get("input_digest") == input_digest
        )

    def world_bytes(self, world_id: str) -> dict:
        """On-disk size of one world, split by tier.

        Reported rather than capped. A hard cap that stopped mapping
        mid-session would trade a storage problem for a silently truncated
        world, which is worse: the operator would get an incomplete map
        with no obvious sign it was cut short. Making growth visible lets
        retention be decided deliberately -- and the derived figure is the
        reclaimable half, since every byte under derived/ is rebuildable.
        """
        directory = self.world_dir(world_id)
        if not directory.exists():
            return {"total": 0, "images": 0, "derived": 0, "journals": 0}

        totals = {"total": 0, "images": 0, "derived": 0, "journals": 0}
        derived_root = self.derived_dir(world_id)
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            size = path.stat().st_size
            totals["total"] += size
            if path.suffix == ".jpg":
                totals["images"] += size
            elif derived_root in path.parents:
                totals["derived"] += size
            elif path.suffix in (".jsonl", ".json"):
                totals["journals"] += size
        return totals

    # -- locking -------------------------------------------------------

    def acquire_writer_lock(self, world_id: str) -> None:
        """Take the single-writer lock, reclaiming it from a dead process.

        Liveness by pid AND process start time rather than by timeout. A
        stale-lock timer has to guess how long a legitimate writer might
        pause; asking the OS whether the pid is still running does not
        guess, and psutil is already a dependency. The start time is
        there because Windows recycles pids aggressively: a lock left by a
        builder that died could otherwise name a pid that now belongs to
        an unrelated process, and the next session would be refused for
        as long as that stranger lived (see `lock_holder`).
        """
        path = self.lock_path(world_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        # CREATE-EXCLUSIVE, NOT CHECK-THEN-WRITE.
        #
        # This read the holder and then wrote the lock, with nothing atomic
        # in between, so two processes arriving in that window both saw "no
        # lock" and both proceeded. The second write overwrote the first's
        # record, so the file named only one of them -- and when the first
        # finished, `release_writer_lock` unlinked a lock the OTHER writer
        # still believed it held. An adversarial review measured TWO
        # SIMULTANEOUS WRITERS ADMITTED IN 8 OF 8 TRIALS, and drove two
        # concurrent `world_finalize.py` runs to `finalized: True` on one
        # world. `world_finalize.py` calls this "the whole safety story".
        #
        # `O_CREAT | O_EXCL` is one atomic operation on Windows and POSIX
        # alike: exactly one caller creates the file. Everyone else falls
        # through to the liveness check, and a caller that decides the
        # holder is gone RECLAIMS by unlinking and trying the exclusive
        # create again -- so two processes that both find a dead lock still
        # cannot both win, and the loser re-reads and sees the winner.
        unreadable_since: float | None = None
        for _attempt in range(_LOCK_ACQUIRE_ATTEMPTS):
            try:
                handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                pass
            else:
                with os.fdopen(handle, "w", encoding="utf-8") as file:
                    file.write(json.dumps(_lock_record(os.getpid())))
                    file.flush()
                    os.fsync(file.fileno())
                # WON THE CREATE -- NOW CONFIRM WE STILL HOLD IT.
                #
                # The create is atomic; the reclaim below is not. A peer
                # that decided this lock was dead unlinks whatever is at
                # the path -- not the file it read -- so it can delete a
                # lock created since, and then create its own. An
                # adversarial review drove that to BOTH PROCESSES
                # ACQUIRING, 5 of 5, with a stall injected in the reclaim
                # window (0 of 120 naturally, so it is narrow, not
                # imaginary).
                #
                # Reading our own record back closes the outcome that
                # matters: whoever's record is on disk owns the world, and
                # a process whose record was replaced goes round rather
                # than returning to write underneath the winner.
                mine = self.lock_holder(world_id)
                if mine is not None and mine["pid"] == os.getpid():
                    return
                continue
            holder = self.lock_holder(world_id)
            if holder is None:
                # Unreadable. Two very different things look like this and
                # only time tells them apart.
                #
                # TRANSIENT: a peer is between its `O_CREAT|O_EXCL` and its
                # write, so the file exists and is empty for microseconds.
                # Reclaiming here would delete a live writer's lock.
                #
                # ORPHANED: that peer was killed in the same window, and
                # the zero-byte file it left names nobody. This is a NEW
                # possibility -- the old code wrote the lock through
                # `write_json_atomic`, which is never partial -- and
                # refusing it forever bricks the world: an adversarial
                # review measured `acquire_writer_lock` raising in 1.3 ms
                # and every retry, and `world_finalize.py`, refused
                # identically. Permanently.
                #
                # So: wait out the transient case, then reclaim. The write
                # window is measured in microseconds; anything unreadable
                # for a tenth of a second is not a writer in progress.
                now = time.monotonic()
                if unreadable_since is None:
                    unreadable_since = now
                elif now - unreadable_since >= _LOCK_UNREADABLE_GRACE_S:
                    logger.warning(
                        "world builder: reclaiming an unreadable lock on %s; it "
                        "names no process and has not become readable in %ss, so "
                        "it is a writer that died mid-write",
                        world_id, _LOCK_UNREADABLE_GRACE_S,
                    )
                    try:
                        path.unlink()
                    except OSError:
                        pass
                    unreadable_since = None
                time.sleep(_LOCK_RETRY_SLEEP_S)
                continue
            unreadable_since = None
            if holder["alive"] and holder["pid"] != os.getpid():
                raise WorldLockedError(
                    f"world {world_id} is locked by live pid {holder['pid']}; "
                    "refusing a second writer"
                )
            logger.warning(
                "world builder: reclaiming lock on %s held by pid %r",
                world_id,
                holder["pid"],
            )
            # Unlink and go round: the create is what decides, not this.
            try:
                path.unlink()
            except OSError:
                pass
            time.sleep(_LOCK_RETRY_SLEEP_S)
        raise WorldLockedError(
            f"world {world_id}: could not take the writer lock in "
            f"{_LOCK_ACQUIRE_ATTEMPTS} attempts; another writer is contending "
            "for it"
        )

    def lock_holder(self, world_id: str) -> dict | None:
        """Who holds the writer lock, and whether that process is alive.

        None when no readable lock exists. Otherwise
        `{"pid": int|None, "alive": bool, "unreadable": bool}`:
        `unreadable` is a lock file that names no usable pid, which is NOT
        "no lock" -- reporting a healthy idle world over a file that is
        right there would downgrade a crashed builder to nothing at all.

        One answer for the store, the status producer and the listing, so
        the three cannot disagree about whether a world is live.
        """
        path = self.lock_path(world_id)
        try:
            if not path.exists():
                return None
            holder = read_json_closed(path)
        except (OSError, json.JSONDecodeError, ValueError):
            return None
        if not isinstance(holder, dict):
            return {"pid": None, "alive": False, "unreadable": True}
        pid = holder.get("pid")
        if not isinstance(pid, int) or isinstance(pid, bool):
            return {"pid": None, "alive": False, "unreadable": True}
        return {
            "pid": pid,
            "alive": _holder_is_running(pid, holder.get("created_at")),
            "unreadable": False,
        }

    def release_writer_lock(self, world_id: str) -> None:
        self.lock_path(world_id).unlink(missing_ok=True)

    # -- purge ---------------------------------------------------------

    def purge_world(self, world_id: str) -> PurgeReport:
        """Really delete a world: images, journals, derived tree, temps.

        Reports what could not be removed rather than claiming success.
        06-PRIVACY-DATA requires real deletion, and the failure mode this
        guards against has already shipped once in this repository: a
        purge that returned a success count while leaving a temp file
        holding live data behind.
        """
        with self._lock:
            world_dir = self.world_dir(world_id)
            if not world_dir.exists():
                return PurgeReport(removed=(), retained=())

            removed: list[str] = []
            retained: list[str] = []

            for path in sorted(
                world_dir.rglob("*"), key=lambda p: len(p.parts), reverse=True
            ):
                try:
                    if path.is_dir():
                        path.rmdir()
                    else:
                        path.unlink()
                    removed.append(str(path))
                except OSError as exc:
                    logger.warning("world builder: could not remove %s: %s", path, exc)
                    retained.append(str(path))

            try:
                world_dir.rmdir()
                removed.append(str(world_dir))
            except OSError as exc:
                logger.warning(
                    "world builder: could not remove %s: %s", world_dir, exc
                )
                retained.append(str(world_dir))

            return PurgeReport(removed=tuple(removed), retained=tuple(retained))


def compute_input_digest(keyframes: list[Keyframe]) -> str:
    """Digest the authoritative inputs a derived build consumed.

    Lets a later read detect that derived output no longer matches the
    journal that produced it, instead of trusting whatever is on disk.
    """
    hasher = hashlib.sha256()
    for keyframe in keyframes:
        hasher.update(keyframe.keyframe_id.encode("utf-8"))
        hasher.update(b"\x00")
    return hasher.hexdigest()


def _parse_all(raw_records: list[dict], parser, what: str) -> list:
    parsed = []
    for raw in raw_records:
        try:
            require_schema(raw, what)
            parsed.append(parser(raw))
        except (KeyError, ValueError, UnsupportedSchemaError):
            # Valid JSON, schema this build cannot interpret. Not
            # corruption: skipped from the parsed view without touching
            # the file, matching object memory's two-stage discipline.
            logger.warning(
                "world builder: skipping %s record with an unrecognised schema",
                what,
            )
    return parsed


# How far apart two readings of one process's start time may be and still
# name the same process. psutil reports the value at millisecond precision
# on Windows and the two readings here are taken by different processes.
_CREATE_TIME_TOLERANCE_S = 1.0


def _lock_record(pid: int) -> dict:
    """What a writer puts in its lock: the pid, and when that pid started.

    The start time is what makes the pid mean one process rather than
    whichever process the OS next hands that number to.
    """
    record = {"pid": pid}
    try:
        import psutil

        record["created_at"] = float(psutil.Process(pid).create_time())
    except Exception:  # pragma: no cover - a lock without a start time still works
        pass
    return record


def _pid_is_running(pid: int) -> bool:
    return _holder_is_running(pid, None)


def _holder_is_running(pid: int, created_at) -> bool:
    """Whether the process a lock names is the process that is running.

    With a start time on the lock, a running pid whose start time differs
    is a DIFFERENT process -- the builder that wrote the lock is dead and
    its number was recycled. Without one (a lock written before start
    times were recorded) the pid alone decides, as it always did.
    """
    try:
        import psutil

        if not psutil.pid_exists(pid):
            return False
        if created_at is None:
            return True
        try:
            actual = psutil.Process(pid).create_time()
        except psutil.Error:
            return False
        return abs(float(actual) - float(created_at)) <= _CREATE_TIME_TOLERANCE_S
    except Exception:  # pragma: no cover - psutil is a hard dependency
        return False
