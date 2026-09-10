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
        except ValueError:
            # `ValueError`, NOT `json.JSONDecodeError`. The latter is a
            # subclass, and `UnicodeDecodeError` -- which is what invalid
            # UTF-8 raises -- is a sibling. A reviewer wrote the same three
            # bad bytes into each copy of one manifest and got opposite
            # answers: the session copy (which already caught `ValueError`)
            # refused cleanly, the world copy raised out of every reader
            # that touches it, including out of `read_derived`'s verify
            # gate, which sits ABOVE its own `try`. Two files meant to be
            # identical have to fail identically.
            logger.warning("world builder: derived manifest unreadable at %s", path)
            return None
        return data if isinstance(data, dict) else None

    def session_manifest_path(self, world_id: str, session_id: str) -> Path:
        """The manifest beside a session's own poses and points.

        Absent for anything built before `write_derived` started writing it.
        `read_derived_manifest` above is the WORLD's, which names whichever
        session built last; this one always describes the session it sits
        in.
        """
        return self.derived_dir(world_id) / session_id / DERIVED_MANIFEST

    def read_session_manifest(self, world_id: str, session_id: str) -> dict | None:
        path = self.session_manifest_path(world_id, session_id)
        if not path.exists():
            return None
        try:
            data = read_json_closed(path)
        except (json.JSONDecodeError, ValueError):
            logger.warning("world builder: session manifest unreadable at %s", path)
            return None
        return data if isinstance(data, dict) else None

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
            # THE SAME MANIFEST, BESIDE THE FILES IT DESCRIBES.
            #
            # A world has ONE `derived/manifest.json` and it names whichever
            # session built last. That is the root of a whole family of
            # defects this campaign kept fixing one symptom at a time: the
            # status producer discards the manifest for any other session
            # (correctly -- attributing one session's figures to another is
            # worse), and then has no figures at all, so an older session of
            # a world walked twice reported no geometry, no poses, no
            # currency, and the phone rendered a red "Needs retry" over a
            # reconstruction sitting on disk. Four separate branches were
            # written to paper over that, three of them wrong, before the
            # question "why is there only one copy" got asked.
            #
            # A session that describes itself needs none of them. Cheap
            # (one small JSON per build, beside megabytes of points),
            # atomic like everything else here, and additive: a world built
            # before this has no per-session copy and reads exactly as it
            # did.
            write_json_atomic(derived / DERIVED_MANIFEST, manifest)
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
            # `is False`, NOT `not ...`. `None` is "no manifest here can
            # judge this" -- a legacy world walked twice, where the only
            # manifest describes another session. Refusing that is a
            # guaranteed 404 for a reconstruction that is sitting on disk
            # and perfectly good; serving it with the wire contract's
            # `current` flag OFF is the honest compromise, and the status
            # channel says in words why it cannot be judged. Only a
            # manifest that actually disagrees is stale.
            if self.derived_currency(world_id, digest, session_id) is False:
                logger.warning(
                    "world builder: derived output for %s/%s is stale; "
                    "treating as absent",
                    world_id, session_id,
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
        except (KeyError, TypeError, ValueError) as exc:
            # `TypeError` BELONGS HERE, and its absence was the one gap in
            # this file. `_read_support` and `read_placements` below both
            # carry explicit comments about a top-level list raising
            # TypeError "straight out of a method whose docstring promises
            # it never raises"; this method, which has the same promise and
            # the same subscript, was never given the same guard. A reviewer
            # fed it `[]` and watched the TypeError come out through
            # `build_manifest` as an HTTP 500 where every other corrupt
            # derived tree gives a 404. A corrupt world and an absent one
            # are both "nothing to serve"; neither is a server fault.
            # (`json.JSONDecodeError` is a `ValueError`, so it is in here
            # by inheritance rather than by being listed twice.)
            #
            # `OSError` IS NOT IN THIS TUPLE, and it was, briefly. A
            # reviewer injected EIO and watched a disk fault become
            # "no geometry for this session" -- a 404 on a route whose own
            # comment says "404 now means ABSENT only", and an empty
            # reconstruction in `world_inspect`. A server that cannot read
            # its own storage should say so, loudly, and 500 is how. The
            # widening was a guess dressed as symmetry.
            logger.warning(
                "world builder: derived output unreadable for %s/%s: %s: %s",
                world_id, session_id, type(exc).__name__, exc,
            )
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

    def derived_is_current(
        self, world_id: str, input_digest: str, session_id: str | None = None
    ) -> bool:
        """Whether the stored geometry answers the question these keyframes ask.

        THIS HAS NO PRODUCTION CALLERS LEFT. `read_derived`, the geometry
        route and the render page all call `derived_currency` directly,
        because two of the three answers it gives are not booleans. Two
        tests still call this, and a boolean is what they want. A previous
        version of this docstring named those three as its callers; they
        had already moved.

        What the world-level version cost: on a world walked twice, every
        reader that gated on it refused the OLDER session outright, logging
        "stale; treating as absent". So "open an earlier walk from Saved
        Worlds" returned a 404 for a reconstruction sitting on disk.
        The status channel's version of the same bug was found and fixed
        four times in four review rounds; this half of it, on the path that
        actually serves the geometry, was found by asking what the phone
        does after the status channel says `ready`.

        With a session id, a session that has its own manifest is judged by
        it. Anything built before those existed falls back to the world's,
        which is exactly as right and as wrong as it was.
        """
        return self.derived_currency(world_id, input_digest, session_id) is True

    def derived_currency(
        self, world_id: str, input_digest: str, session_id: str | None = None
    ):
        """True, False, or **None for "nothing here can judge it"**.

        THE THIRD ANSWER IS THE POINT, and folding it into `False` is what
        the first version of this fix did wrong. Two different questions
        were being asked of one boolean:

          * *is this geometry current?* -- what the wire contract's
            `current` flag means, "reflects every keyframe accepted so
            far", and what the status channel reports.
          * *may this geometry be served at all?* -- what
            `read_derived`'s verify gate decides.

        For a world built before per-session manifests existed, walked
        twice, the honest answer to the first is "unknown" and to the
        second is "yes, with the flag off". Returning `True` from one
        boolean made the ROUTE assert `current: true` over geometry a
        reviewer then made genuinely stale by appending a keyframe -- while
        the status channel beside it said `current: false`. Returning
        `False` refuses to serve a good reconstruction. Neither is the
        answer; there are three.
        """
        manifest = None
        if session_id is not None:
            manifest = self.read_session_manifest(world_id, session_id)
            if manifest is not None and manifest.get("session_id") != session_id:
                # A session's own copy naming somebody else is corruption,
                # not a world walked twice.
                manifest = None
        if manifest is None:
            world_manifest = self.read_derived_manifest(world_id)
            if not isinstance(world_manifest, dict):
                world_manifest = None
            if (
                session_id is not None
                and world_manifest is not None
                and world_manifest.get("session_id") != session_id
            ):
                # The world's manifest is about another session and this
                # one has no copy of its own: a legacy world walked twice.
                return None
            manifest = world_manifest
        if manifest is None:
            # NO MANIFEST ANYWHERE IS THE THIRD ANSWER, NOT `False`.
            #
            # This returned False, and the docstring above spends a
            # paragraph explaining why that is wrong -- for the one case it
            # DID handle, a world manifest naming another session. The case
            # where there is no manifest at all fell through to here and got
            # the answer the docstring rejects.
            #
            # `False` means "a manifest exists and disagrees", and
            # `read_derived`'s gate refuses on exactly that. So a world
            # built before per-session manifests existed, or one whose
            # manifests were lost, was 404 for a reconstruction sitting on
            # disk -- the campaign's named failure, on the serving path,
            # in the function written to prevent it.
            #
            # It surfaced when the status channel learned to recount such a
            # session's poses and points and report it `ready`: the channel
            # promised a world and the wearer's next tap got nothing. A
            # promise the next tap breaks is worse than the old
            # consistently-wrong pair, which is why this is a defect the
            # recount created rather than one it merely revealed.
            #
            # Nothing is lost by serving it. The wire contract carries
            # `current`, the route sets it from this same three-valued
            # answer, and the status channel says in words that currency
            # cannot be judged.
            return None
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


REQUIRED_MANIFEST_KEYS = (
    "input_digest",
    "session_id",
    "keyframes",
    "points",
    "poses_solved",
    "poses_refused",
    "segments",
)


def session_has_drawable_geometry(store, world_id, session_id, manifest=None) -> bool:
    """Would opening this session SHOW the wearer anything?

    **Not "do the files exist", which is what two separate copies of this
    used to ask.** `engine.build` calls `write_derived` unconditionally,
    so a walk that solved nothing still leaves `poses.json` and a
    `points.json` holding `{"points": []}` -- 14 bytes. Eleven sessions on
    the real 163-world root are exactly that, and they were listed as
    `complete, has_geometry: true` while the status channel for the same
    session projected `needsRetry`.

    Two surfaces need this answer and they must not compute it apart:

      * `results/world_builder_library` puts it on the wire as
        `has_geometry`, which `WorldPickerView` branches on;
      * `results/world_builder_render.resolve_session` uses it to choose
        WHICH session to draw, so an empty newer walk would otherwise be
        picked over an older one that has geometry, and the page drawn
        blank.

    A third predicate, `results/world_builder._has_session_geometry`,
    deliberately still answers EXISTENCE -- it decides which lifecycle
    state a session is in ("was there a build"), which is a different
    question, and its own test says so.

    `manifest` is passed in when the caller already has it, which the
    listing does; otherwise it is read here. The count of `points.json` is
    the fallback for a session no manifest describes -- 0 of the 49 real
    sessions with a tree, and every one of those manifests carries both
    figures.
    """
    derived = store.derived_dir(world_id) / session_id
    if not ((derived / "poses.json").exists() and (derived / "points.json").exists()):
        return False
    if manifest is None:
        try:
            manifest = store.read_session_manifest(world_id, session_id)
            if not (
                isinstance(manifest, dict)
                and manifest.get("session_id") == session_id
            ):
                world = store.read_derived_manifest(world_id)
                manifest = (
                    world
                    if isinstance(world, dict)
                    and world.get("session_id") == session_id
                    else None
                )
        except (WorldStoreError, OSError, ValueError, KeyError):
            manifest = None
    if isinstance(manifest, dict):
        points = manifest.get("points")
        positioned = manifest.get("poses_positioned")
        if isinstance(points, int) or isinstance(positioned, int):
            return (points or 0) > 0 or (positioned or 0) > 0
    return _points_on_disk(store, world_id, session_id) > 0


def _points_on_disk(store, world_id, session_id) -> int:
    """`len(points.json)`, for a session no manifest summarises.

    The rows are dropped as soon as they are counted; only the length is
    kept. Unreadable is not empty -- but the only honest answer inside a
    bool is the one that does not promise a wearer something to look at,
    and the status channel says the difference in words.
    """
    path = store.derived_dir(world_id) / session_id / "points.json"
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))["points"]
    except (KeyError, TypeError, ValueError, OSError):
        return 0
    return len(rows) if isinstance(rows, list) else 0


def validate_manifest(
    manifest, world_id, source="derived manifest", *, require_figures=True
):
    """Schema-check a manifest, wherever it was read from, or None.

    **HERE, RATHER THAN IN ONE READER, BECAUSE THERE ARE FOUR READERS.**
    This lived in `results/world_builder.py` and the status producer was
    the only caller. `results/world_builder_geometry._session_manifest`
    -- which the geometry route, `usable_placements` and the saved-worlds
    listing all reach -- checked only `isinstance(dict)` and
    `session_id`, so the two disagreed about any manifest that was
    readable and wrong.

    A reviewer measured what that produced: for a session whose manifest
    declares a schema this build does not know, and whose derived tree is
    gone, the picker said `interrupted` -- "a build ran and its output is
    gone" -- while the status channel behind that same row said
    `stopped_unbuilt`, "this walk produced no geometry". The two surfaces
    this campaign spent a round reconciling, contradicting each other,
    through a comment that claimed "four readers, one rule".

    A no-op on real data: all 49 sessions with a derived tree on the
    163-world root pass both the loose and the strict rule.
    """
    if not isinstance(manifest, dict):
        # `isinstance`, not `is None`. A world manifest holding a
        # top-level list or string reached `.get()` and came out as an
        # AttributeError -- a 500 on the status channel, where the
        # identically corrupt per-session copy gave a clean refusal.
        return None
    if manifest.get("schema_version") != SCHEMA_VERSION:
        # A manifest from another schema describes fields whose meaning
        # this build does not know. It is refused as a SUMMARY; the poses
        # and points beside it are a separate question, and
        # `_figures_from_the_tree` answers it separately.
        return None
    if not require_figures:
        # IDENTITY AND PROVENANCE ONLY, which is all some readers need.
        #
        # The figure check below exists because the STATUS CHANNEL reports
        # those figures, and "geometry: available with every count null"
        # is a claim with nothing behind it. A reader asking "which build
        # produced this" needs only `session_id` and `input_digest`, and
        # holding it to the figures refuses manifests that answer its
        # question perfectly well.
        #
        # Unifying the two readers without this split was itself a defect:
        # `usable_placements` started refusing every placement of a
        # session whose manifest carries a digest and no counts, so a
        # registered segment stopped serving its transform. Caught by
        # `test_world_builder_placements.py`, five tests at once, in the
        # full suite rather than in the targeted one I had been running.
        return manifest
    missing = [key for key in REQUIRED_MANIFEST_KEYS if manifest.get(key) is None]
    if missing:
        logger.warning(
            "world builder: %s for %s is missing %s; treating it as absent "
            "rather than reporting geometry with no figures",
            source, world_id, missing,
        )
        return None
    return manifest


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
