#!/usr/bin/env python
"""Rebuild one saved session with the product pipeline: components, the room and its areas.

    .venv\\Scripts\\python.exe scripts/world_refinish.py --root <world root> --world <id>
        [--session <id>] [--seed 0] [--threads -1] [--no-appearance]
        [--capture-dir <dir> ... | --no-capture]
        [--keep-depth-work] [--dry-run] [--format json|text]

WHAT IT IS FOR (WORLD-BUILDER-COMPONENTS.md §7 rule 4). Every world saved before the
evidence gate has `components: null` and shows today's screens; nothing rebuilds it on
its own -- the Tower's start-up and idle finisher deliberately never computes
components (§7 rule 5). An owner who wants an OLD world shown the new way runs this,
once, for that world. Nothing else runs it.

WHAT IT DOES, IN ORDER, from the session's own authoritative data (its journal and its
stored, redacted keyframes; refused for a world whose imagery was purged, while a live
writer holds the world, and while any stage of the session -- the room's or an area's
-- is running under a live process, which the builder's post-Stop surface does WITHOUT
the lock):

1.  SETS THE PREVIOUS RESULT ASIDE, AND DELETES NOTHING (contract T11). Under the
    world's writer lock, ledger first, copies before moves, areas before the solve, and
    every completed move undone if a later one fails (a file held open on Windows), so
    it never stops half done (`set_aside`). Then the room's stages are recorded
    `stopped` ("re-finish in progress"), so a run that ends before step 3 leaves a room
    the finisher rebuilds:
    - `solve/<session>` is MOVED to `<world>/refinish/<stamp>/solve/<session>`, and
      then the walk's own feature database (`database.db`, with any SQLite
      `-wal`/`-shm` beside it), the solver images it was extracted from (`images/`),
      `sources.json`, `camera.json`, the solver's content-keyed transient-mask
      cache (`transients/`) and the record of the walk database's frozen matching
      (`database.matching.json`) are COPIED BACK into a fresh
      `solve/<session>` -- never moved back, so the set-aside copy stays byte for
      byte what the walk left, for rollback. The copy is what makes the masked
      solve filter THE WALK'S OWN DATABASE (`solve.masking:
      "walk-database-filtered"`, the approved arm A1h). Without it the solve finds
      no walk database, re-extracts under the masks (`"re-extracted"`, arm A1),
      and on the target walk that splits the room (reviewer RV1, finding M1-1).
      A solver image goes back ONLY when it is proven to be exactly what step 2
      plans for its keyframe (review V9, M-7; `carry_back_decisions`); every other
      one is withheld -- the solve writes it afresh from its planned frame -- and
      its features are cleared from the copied database. Each image's provenance
      is recorded in `images.provenance.json`, which `prepare_images` honours;
    - `surface/`, `appearance/` and `dense/<session>` (without its depth-prediction
      cache, review V9 M-12), `derived/` and the session record are COPIED there,
      because the rebuild replaces them in place;
    - every `areas/<area>` of this session is MOVED there: a new solve names new areas;
    - `<world>/refinish/<stamp>/refinish.json` records what was set aside, from where,
      and the session's previous finalization and stage records. Restoring is moving
      those directories back; deleting them needs a human's approval.
2.  THE FINAL SOLVE, WITH THE PRODUCT SETTINGS: `scripts/world_finalize.py` in a child
    process with `TOWER_WORLD_SOLVE_MASKS=1` (hand/arm/held-phone masks),
    `TOWER_WORLD_SOLVE_SEED=<--seed>` (the seeded single-thread solve) and
    `TOWER_WORLD_SOLVE_GATE=1` (depth before publish, the evidence gate, and
    `solve/<session>/components.json`). It takes the lock itself, re-derives the tree
    and records the finalization. Before it, each keyframe's OWN raw frame -- found
    by its capture identity (`source_seq` + `received_at`) in the journals of the
    captures named by `--capture-dir`, else of the session's capture chain under
    `TOWER_CAPTURE_ROOT`; `--no-capture` for neither -- is written into the fresh
    solve directory's `sources.json`, as the builder records it; a keyframe with no
    unambiguous frame keeps its stored redacted copy. The ledger's `solver_frames`
    says where every solver frame comes from (`plan_solver_frames`).
3.  THE ROOM: the builder's own final surface and appearance
    (`world_build_session.final_surface_stages`) under the lock.
4.  THE AREAS: every `shown_as: "area"` component, built on its own and levelled
    (`tower/world_builder/area_build.py`), whatever `TOWER_WORLD_AREA_BUILDS` says --
    asking for this command is asking for the areas.

THE RESULT. `components` appears on the session's row; the room may change (the gate
may move pieces out of it), so its revision changes and an open viewer offers *A newer
reconstruction is ready* (IOS §10). If the gate is not wired on this Tower the solve
still runs, the room is rebuilt, and the report says `components: null`.

A final solve that publishes nothing puts the previous result back: the solve, the
areas, the session's own derived tree and the session record (what the failed solve left is
kept under `refinish/<stamp>/failed-*`). So does ANY exception between the set-aside and a
published solve -- a session record that cannot be written, a child that cannot be
started -- and the ledger then says `restored-after-an-error` and names the step and
the error (`restore-incomplete`, with what could not be done, if it could not all go
back). And so does the idle Tower's finisher, or the next run of this command, when a
re-finish was KILLED before it published -- a console closed, a reboot (review V9,
M-5; `recover_dead_refinish`): the ledger then says
`restored-after-the-refinish-ended`. The finisher's attempt counters are restarted only
once the new solve is published, so a put-back leaves them as they were.

THE PUT-BACK READS THE DISK, NOT THE LEDGER, AND CAN BE RUN AGAIN (review V10, MED-2): a
put-back killed part-way is finished by the next one, and one an owner did by hand is
recognised, never undone (`restore_after_failed_solve`). It puts back only THIS session's
`derived/<session>` and the world's `derived/manifest.json` while that still names this
session, and it parks rather than touching anything if another session's derived output
changed since the set-aside (MED-3). The final solve's child is recorded in the ledger as
soon as it exists, and nothing is put back while it runs (V10 Q3).

Exit status: 0 done; 1 a step failed (the report says which); 2 refused (no such world
or session, imagery purged, a live writer holds the world, a build is running, or the
set-aside could not complete and was rolled back).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tower.artifact_paths import artifact_root_arg  # noqa: E402
from tower.world_builder.store import WorldStore, WorldStoreError  # noqa: E402

logger = logging.getLogger(__name__)

REFINISH_DIRNAME = "refinish"
LEDGER_FILENAME = "refinish.json"
EXIT_OK, EXIT_FAILED, EXIT_REFUSED = 0, 1, 2
FINALIZE_SCRIPT = Path(__file__).resolve().parent / "world_finalize.py"

# What the fresh solve directory gets back, AS COPIES, from the set-aside one: the
# walk's feature database (and SQLite's companions, if a writer left them), the
# solver images its keypoints came from, where the raw frames were, the camera the
# images were undistorted with, and the solver's transient-mask cache
# (`transients/`, keyed by image name AND the SHA-1 of its bytes, so an entry can
# only ever be used for the exact image it was computed on -- it saves the masked
# solve its GPU minutes), and the record of the walk database's FROZEN matching
# (`database.matching.json`, `global_solve.FROZEN_MATCHING_FILENAME`: the seeded
# final solve that made it maps that database as it is instead of matching again,
# which is not reproducible -- review V8 H2). NOT `masks/` (the COLMAP masks are
# rewritten from that cache on every masked solve), nor `database.masked.*` /
# `reverify_pairs.txt` (made by a solve, for that solve). See `set_aside`.
#
# `images/` GOES BACK ONLY IMAGE BY IMAGE, AND ONLY WHERE ITS PROVENANCE IS PROVEN
# (review V9, M-7). A carried-back solver image used to outrank capture identity:
# `prepare_images` never rewrites an image that exists, so a walk first solved from
# its redacted keyframes stayed redacted after a re-finish that found every raw frame
# (RV9-D D1), a frame another capture's by-name lookup had given it survived every
# re-finish (D2), and duplicate-id keyframes were solved from a raw image instead of
# their stored copy (D3). Now each image is carried back only when it is exactly what
# this re-finish plans for it (`carry_back_decisions`), its provenance is recorded
# (`SOLVER_IMAGES_PROVENANCE`), and every other image is WITHHELD -- rewritten by the
# solve from the planned frame -- and its features are cleared from the copied
# database (`global_solve.clear_image_features`, the one implementation; review V10,
# L-16c), so the rewritten image is extracted and matched afresh instead of mapped with
# the old image's keypoints. One exception (review V10, L-8): a walk image the
# provenance RECORD proves was undistorted from the keyframe's own raw frame still goes
# back when that frame is no longer on disk -- see `carry_back_decisions`.
SOLVE_COPY_BACK = ("database.db", "database.db-wal", "database.db-shm", "images",
                   "sources.json", "camera.json", "transients", "database.matching.json")

# THE SOLVER-IMAGE PROVENANCE RECORD (review V9, M-7; the format is agreed with
# `global_solve.prepare_images`, which honours it when the file exists):
#
#     {"record": "wb-solver-image-provenance/1",
#      "images": {"<image name>": {"keyframe_id": ..., "source": "raw" | "redacted",
#                                  "frame": ..., "sha1": ...}}}
#
# `source` is where the image's pixels were undistorted from: a raw capture frame, or
# the session's stored (face-redacted) keyframe. `frame` is the raw frame's path AS
# `sources.json` RECORDS IT (compared after `global_solve.resolve_source_path`), or the
# keyframe's `image_relpath` for a redacted copy. `sha1` is the image file's
# (`solve_masks.file_sha1`). One entry per image NAME -- for a name two keyframes share,
# the first keyframe in keyframe order, which is the one `prepare_images` writes. A
# re-finish writes it into every fresh solve directory, with an entry for each image it
# carried back; an absent file is a solve directory no re-finish has prepared, and means
# today's behaviour.
SOLVER_IMAGES_PROVENANCE = "images.provenance.json"
PROVENANCE_RECORD = "wb-solver-image-provenance/1"
IMAGE_SOURCE_RAW = "raw"
IMAGE_SOURCE_REDACTED = "redacted"
# `prepare_images` writes a solver image as `cv2.imwrite(..., [IMWRITE_JPEG_QUALITY, 95])`
# of the undistorted, cropped frame; an image is VERIFIED BY REPRODUCTION when the same
# steps on the planned frame give the same bytes (`_solver_image_sha1`). Measured on the
# frozen walk 6839fb8f: 689 of its 690 solver images reproduce byte for byte from the raw
# frame `sources.json` names; the 690th (keyframe 00000936) reproduces from its REDACTED
# copy -- the walk solved it before its raw frame was recorded -- and is withheld.
SOLVER_JPEG_QUALITY = 95

# The depth-prediction cache (`dense_pipeline.PREDICTIONS_DIRNAME`, the gate's depth
# stage) is left out of the set-aside COPY of `dense/<session>` (review V9, M-12): it is
# re-derivable from the redacted keyframes and the network, it is about twenty times
# the size of the keyframe images (311 MB for 678 frames), and every re-finish would
# keep another full copy of it, which policy forbids deleting. The live cache is not
# touched -- the rebuild reuses it -- and the ledger's copy entry says it was skipped.
DENSE_PREDICTIONS_DIRNAME = "predictions"
DENSE_PREDICTIONS_SKIPPED = ("re-derivable depth predictions (the gate's depth cache, about "
                             "twenty times the keyframe images); left in place for the "
                             "rebuild to reuse, not copied")

# The product settings a re-finish runs the final solve with (§7 rule 4).
PRODUCT_SOLVE_ENV = {
    "TOWER_WORLD_SOLVE_MASKS": "1",
    "TOWER_WORLD_SOLVE_GATE": "1",
}


class Refused(Exception):
    pass


def _latest_session(store: WorldStore, world_id: str) -> str | None:
    from scripts.world_finalize import latest_session  # noqa: PLC0415

    return latest_session(store, world_id)


def plan(store: WorldStore, world_id: str, session_id: str, stamp: str) -> dict:
    """What would be set aside, and where. Reads only."""
    world_dir = store.world_dir(world_id)
    aside = world_dir / REFINISH_DIRNAME / stamp
    moves, copies = [], []
    for kind, how in (("solve", "move"), ("surface", "copy"), ("appearance", "copy"),
                      ("dense", "copy")):
        src = world_dir / kind / session_id
        if src.exists():
            entry = {"kind": kind, "from": str(src), "to": str(aside / kind / session_id)}
            if kind == "dense" and (src / DENSE_PREDICTIONS_DIRNAME).is_dir():
                entry["skipped"] = [DENSE_PREDICTIONS_DIRNAME]
                entry["skipped_why"] = DENSE_PREDICTIONS_SKIPPED
            (moves if how == "move" else copies).append(entry)
    from tower.world_builder.components import session_area_dirs  # noqa: PLC0415

    for area in session_area_dirs(store, world_id, session_id):
        moves.append({"kind": "areas", "from": str(area),
                      "to": str(aside / "areas" / area.name)})
    if (world_dir / "derived").exists():
        copies.append({"kind": "derived", "from": str(world_dir / "derived"),
                       "to": str(aside / "derived")})
    copies.append({"kind": "session", "from": str(store.session_path(world_id, session_id)),
                   "to": str(aside / "session.json")})
    solve_src = world_dir / "solve" / session_id
    copy_back = [name for name in SOLVE_COPY_BACK if (solve_src / name).exists()]
    return {"aside": str(aside), "moves": moves, "copies": copies,
            "copy_back_into_fresh_solve": copy_back}


def _restart_attempts(store: WorldStore, world_id: str, session_id: str,
                      stamp: str) -> dict:
    """Restart `world_finish_pending`'s counters for this session (room and areas);
    return what they were. Called once the new solve is published (review V9, M-6 and
    LOW): the counters describe the build being REPLACED, and until the publish nothing
    has replaced it -- a put-back leaves them as they were."""
    from scripts import world_finish_pending as wfp  # noqa: PLC0415

    # The room's, the areas' AND the re-gate's counters (FIN, review V8): a re-finished
    # world must not inherit an exhausted re-gate count from the build it replaces. And the
    # put-back's (review V10, L-7b): an earlier dead re-finish's put-back attempts bound
    # nothing once a new solve is published.
    keys = (session_id, wfp.area_ledger_key(session_id), wfp.regate_ledger_key(session_id),
            wfp.refinish_ledger_key(session_id))
    with wfp._LEDGER_LOCK:
        sessions, unreadable = wfp._read_ledger(store, world_id)
        if unreadable:
            return {"unreadable": True}
        # A counter this stamp already restarted is not restarted again, and not reported as
        # what it WAS: a second call (the finisher's, after a kill between the first and its
        # ledger write) must not overwrite the record of the build that was replaced.
        prior = {k: sessions[k] for k in keys if k in sessions
                 and (sessions[k] or {}).get("detail") != _restart_detail(stamp)}
        if prior:
            sessions = dict(sessions)
            for key in prior:
                sessions[key] = {"attempts": 0, "forgiven": 0,
                                 "detail": _restart_detail(stamp)}
            wfp._write_ledger(store, world_id, sessions)
    return prior


# Windows refuses to rename a directory while any file under it is open (WinError 5),
# and the Tower may be serving one of them -- an area chunk, a surface level. So every
# move is retried over a few seconds, and a set-aside that still cannot complete is
# ROLLED BACK rather than left half done (review V6, M1).
MOVE_ATTEMPTS = 5
MOVE_BACKOFF_S = 0.2
FRESH_SOLVE_STAGING = "fresh-solve"

LEDGER_SETTING_ASIDE = "setting-aside"
LEDGER_SET_ASIDE = "set-aside"
LEDGER_ROLLED_BACK = "rolled-back"
LEDGER_ROLLBACK_INCOMPLETE = "rollback-incomplete"
LEDGER_RESTORED = "restored-after-a-failed-solve"
# An exception between the set-aside and a published solve (review P3, PV attempt 1:
# a read-only `session.json` made `mark_stage` raise WinError 5 after the solve was
# set aside, and the world was left with its solve moved and a session saying
# `complete`). The previous result is put back exactly as for a failed solve.
LEDGER_RESTORED_AFTER_ERROR = "restored-after-an-error"
# The previous result could not be put back, or only part of it; `restore_report`
# and `restore_errors` in the ledger say what is where.
LEDGER_RESTORE_INCOMPLETE = "restore-incomplete"
# After a published solve (FIN, review V8): the room and the areas are being rebuilt
# (`published`), then the run's TERMINAL state -- `done`, or `stopped` when a stop was
# asked for or the room could not be rebuilt (`detail` says which). A ledger left at a
# non-terminal state by a process that is gone is a re-finish that died there; the
# ledger's `process` (pid and start time) is what tells a reader so
# (`refinish_process_alive`).
LEDGER_PUBLISHED = "published"
LEDGER_DONE = "done"
LEDGER_STOPPED = "stopped"
# A re-finish whose process ended in step 1 or 2 -- before its new solve was published --
# put back by the idle Tower's finisher (review V9, M-5; `recover_dead_refinish`).
LEDGER_RESTORED_AFTER_DEATH = "restored-after-the-refinish-ended"
LEDGER_TERMINAL_STATES = (LEDGER_ROLLED_BACK, LEDGER_ROLLBACK_INCOMPLETE, LEDGER_RESTORED,
                          LEDGER_RESTORED_AFTER_ERROR, LEDGER_RESTORE_INCOMPLETE, LEDGER_DONE,
                          LEDGER_STOPPED, LEDGER_RESTORED_AFTER_DEATH)
# The two states a re-finish is in before its new solve is published: the only ones in
# which a re-finish that died leaves the world with its solve set aside (M-5).
LEDGER_BEFORE_PUBLISH = (LEDGER_SETTING_ASIDE, LEDGER_SET_ASIDE)


def refinish_process_alive(ledger: dict) -> bool | None:
    """Whether the re-finish that wrote this ledger is still running -- its own process, OR
    the final solve's child it started (review V10, Q3): True, False, or None when it
    cannot be told. `refinish_liveness` says which."""
    return refinish_liveness(ledger)["alive"]


def refinish_liveness(ledger: dict) -> dict:
    """`{"alive", "process", "child"}`, each True, False or None (unknown).

    THE PROCESS: pid AND start time, the writer lock's own test
    (`store._holder_is_running`), so a recycled pid is not taken for the re-finish that
    died. None for a ledger written before the pid was recorded, AND for one with a pid but
    no start time (review V9, LOW): a bare pid cannot tell the re-finish from whatever
    process reuses its number, so it is UNKNOWN -- the finisher then decides by what the
    files show, which keeps it out of a marked room (`world_finish_pending`) -- never
    "alive" or "dead" on the pid alone. Logged, because a ledger should not lack it.

    THE CHILD (review V10, Q3): `world_finalize.py`, recorded in the ledger (`child`) as
    soon as it exists (`run_final_solve`). A re-finish whose parent was killed while its
    child runs on is NOT dead: the child may still publish, and a put-back under it would
    hand it the ORIGINAL solve directory to overwrite. Dead once it ended (`exit_code`
    recorded) or is gone by pid and start time; a ledger without a child (none was started,
    or it predates V10) is decided by the process alone.

    ALIVE if either runs; dead only when both are known to have ended."""
    process = _recorded_alive((ledger or {}).get("process"), ledger, "pid")
    child_record = (ledger or {}).get("child")
    child = False if child_record is None else _recorded_alive(child_record, ledger,
                                                               "solve child pid")
    if process or child:
        alive = True
    elif process is None or child is None:
        alive = None
    else:
        alive = False
    return {"alive": alive, "process": process, "child": child}


def _recorded_alive(record, ledger, what: str) -> bool | None:
    from tower.world_builder.store import _holder_is_running  # noqa: PLC0415

    record = record if isinstance(record, dict) else {}
    if "exit_code" in record:
        return False            # it was seen to end
    pid = record.get("pid")
    if not isinstance(pid, int) or isinstance(pid, bool):
        return None
    created_at = record.get("created_at")
    if not isinstance(created_at, (int, float)) or isinstance(created_at, bool):
        logger.warning("[Tower][WorldBuilder] re-finish ledger %s names %s %s without its "
                       "start time; whether it is running cannot be told, so it is treated "
                       "as unknown", (ledger or {}).get("stamp"), what, pid)
        return None
    return bool(_holder_is_running(pid, created_at))


class SetAsideFailed(Refused):
    """The previous result could not be set aside. Everything that had moved was moved
    back (or the ledger says exactly what could not be), so the world is as it was."""


def _replace_with_retry(src: Path, dst: Path) -> None:
    last = None
    for attempt in range(MOVE_ATTEMPTS):
        try:
            os.replace(src, dst)
            return
        except PermissionError as exc:
            last = exc
            time.sleep(MOVE_BACKOFF_S * (2 ** attempt))
    raise last


def _write_ledger(aside: Path, ledger: dict) -> None:
    from tower.storage import write_json_atomic  # noqa: PLC0415

    write_json_atomic(aside / LEDGER_FILENAME, ledger)


def _write_ledger_quietly(aside: Path, ledger: dict) -> str | None:
    """`_write_ledger`, for a path that is already failing: None when written, else why
    not. A ledger that cannot be written must not stop what puts the world back."""
    try:
        _write_ledger(aside, ledger)
    except Exception as exc:  # noqa: BLE001 -- reported by the caller
        return f"{type(exc).__name__}: {exc}"
    return None


def _move_back(done: list, ledger: dict, aside: Path) -> bool:
    """Undo completed moves, newest first. True when every one went back."""
    ok = True
    for m in reversed(done):
        try:
            _replace_with_retry(Path(m["to"]), Path(m["from"]))
            m["moved_back"] = True
        except OSError as exc:
            ok = False
            m["moved_back"] = False
            m["move_back_error"] = f"{type(exc).__name__}: {exc}"
    _write_ledger_quietly(aside, ledger)
    return ok


def _skipping(root: Path, skipped):
    """A `copytree` ignore that leaves out `skipped` names directly under `root` only."""
    if not skipped:
        return None
    top = os.path.normcase(os.path.abspath(root))

    def ignore(directory, names):
        if os.path.normcase(os.path.abspath(directory)) != top:
            return []
        return [n for n in names if n in skipped]

    return ignore


def set_aside(store: WorldStore, world_id: str, session_id: str, stamp: str, *,
              solver_images: dict | None = None) -> dict:
    """Step 1, under the caller's lock. Never deletes; returns the ledger it wrote.

    IN AN ORDER THAT CANNOT STRAND THE WORLD (review V6, M1; review V9, M-6):

    1. the LEDGER first, naming everything that is about to move, so nothing can move
       without a record of where it went;
    2. the snapshot COPIES (surface, appearance, dense without its prediction cache,
       derived, the session record), and the COPY-BACK of the walk's solve inputs into
       a staging directory beside the ledger -- only the solver images whose provenance
       is proven, with the withheld images' features cleared from the COPIED database
       (M-7) -- both copies, both before anything moves, so the copy-back does not
       depend on any move having happened;
    3. the MOVES, the session's areas before its solve, each retried against a file
       another process holds open;
    4. the staged copy put in place as the fresh `solve/<session>`, and the ledger's
       `set-aside`.

    ONE ROLLBACK COVERS EVERYTHING FROM THE FIRST MOVE TO THE LAST LEDGER WRITE (M-6).
    Any exception there -- not only a move's `OSError` -- puts the staged copy back
    beside the ledger if it was placed, undoes every completed move, and raises
    `SetAsideFailed`, which `main` reports as a refusal (an interrupt is re-raised as
    itself, after the same rollback). It used to cover the moves alone: the finisher's
    counters were restarted after them, a read-only `finish_attempts.json` raised
    `PermissionError`, and the solve and areas stayed moved with the ledger at
    `setting-aside` and nothing reporting it (RV9-D probe A). The counters are now
    restarted after the publish, and never fatally (`refinish`). A failure before the
    first move moved nothing, and says so.

    `solver_images` is `plan_solver_frames`' answer, whose `carry_back` decides which
    solver images go back; None plans from the walk's own `sources.json`, searching no
    capture.
    """
    p = plan(store, world_id, session_id, stamp)
    aside = Path(p["aside"])
    try:
        aside.mkdir(parents=True, exist_ok=False)
        session = store.read_session(world_id, session_id)
    except Exception as exc:  # noqa: BLE001 -- nothing has moved: a refusal
        raise SetAsideFailed(f"the previous result could not be set aside "
                             f"({type(exc).__name__}: {exc}); nothing was moved") from exc
    moves = [dict(m) for m in p["moves"]]
    # Areas before the solve: an open area file is the likeliest refusal, and failing
    # there leaves nothing of the solve to put back.
    moves.sort(key=lambda m: 0 if m["kind"] == "areas" else 1)
    for m in moves:
        if m["kind"] == "solve":
            # What the ORIGINAL is, so a put-back can tell it from anything else at its place
            # once the set-aside copy is gone (review V10, MED-2; `_put_move_back`).
            m["solution_sha1"] = _sha1_or_none(Path(m["from"]) / "solution.json")
    staging = aside / FRESH_SOLVE_STAGING
    from tower.world_builder.store import _lock_record  # noqa: PLC0415

    ledger = {
        "command": "scripts/world_refinish.py",
        "state": LEDGER_SETTING_ASIDE,
        "world_id": world_id, "session_id": session_id, "stamp": stamp,
        "set_aside_at": time.time(),
        # Which process is doing this (FIN, review V8): pid and start time, as in the
        # writer lock. See `refinish_process_alive`.
        "process": _lock_record(os.getpid()),
        "moved": moves, "copied": p["copies"], "copied_back": [],
        "previous": {"finalization": session.finalization, "stages": session.stages},
        "restore": "move the rebuild's own solve/<session> (and areas) aside, then move "
                   "each `moved` entry back from `to` to `from`; the `copied` entries "
                   "are snapshots of what the rebuild replaced in place, and the "
                   "`copied_back` entries are the rebuild's COPIES of the set-aside "
                   "walk database and images (the originals were never modified)",
        "deletion": "requires a human's approval (Glasses filesystem policy rule 14)",
    }
    try:
        _write_ledger(aside, ledger)
    except Exception as exc:  # noqa: BLE001 -- nothing has moved: a refusal
        raise SetAsideFailed(f"the previous result could not be set aside (writing its "
                             f"ledger: {type(exc).__name__}: {exc}); nothing was moved: "
                             f"{aside}") from exc

    def fail(state: str, what: str, exc: BaseException) -> "SetAsideFailed":
        ledger["state"] = state
        ledger["error"] = f"{what}: {type(exc).__name__}: {exc}"
        unwritten = _write_ledger_quietly(aside, ledger)
        return SetAsideFailed(
            f"the previous result could not be set aside ({ledger['error']}); "
            + ("nothing was moved" if state == LEDGER_ROLLED_BACK else
               "SOME MOVES COULD NOT BE UNDONE -- see the ledger")
            + f": {aside / LEDGER_FILENAME}"
            + (f" (the ledger itself could not be written: {unwritten})" if unwritten else ""))

    # 2. copies: snapshots, then the copy-back staged beside the ledger.
    solve_live = store.world_dir(world_id) / "solve" / session_id
    try:
        for c in p["copies"]:
            src, dst = Path(c["from"]), Path(c["to"])
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, dst, ignore=_skipping(src, c.get("skipped")))
            else:
                shutil.copy2(src, dst)
        # THE WALK'S OWN DATABASE GOES BACK, AS A COPY (RV1 M1-1). The masked final
        # solve maps a filtered copy of the walk database when one is there
        # (`global_solve`, `walk-database-filtered`, arm A1h) and re-extracts under
        # the masks when it is not (`re-extracted`, arm A1) -- which splits the
        # target room. So the fresh solve directory gets a COPY of the database and
        # of the solver images its keypoints were extracted from (so the masks are
        # computed on the same pixels) -- the ones whose provenance is this
        # re-finish's plan (M-7) --, where the raw frames were (`sources.json`), the
        # camera the images were undistorted with (`camera.json`, without which
        # `prepare_images` would re-undistort them) and the content-keyed mask cache.
        # The set-aside originals are never touched again.
        if solve_live.exists():
            if solver_images is None:
                solver_images = plan_solver_frames(store, world_id, session_id, ())
            carry = solver_images.get("carry_back") or {}
            carried = carry.get("carried") or {}
            staging.mkdir()
            database_entry = None
            changed_while_copying: list = []
            for name in SOLVE_COPY_BACK:
                old = solve_live / name
                entry = {"kind": "solve", "name": name,
                         "from": str(aside / "solve" / session_id / name),
                         "to": str(solve_live / name)}
                if name == "images":
                    if not old.is_dir():
                        continue
                    entry.update(_carry_back_images(old, staging / name, carried))
                    changed_while_copying = entry["images_changed_while_copying"]
                elif old.is_dir():
                    shutil.copytree(old, staging / name)
                elif old.is_file():
                    shutil.copy2(old, staging / name)
                else:
                    continue
                ledger["copied_back"].append(entry)
                if name == "database.db":
                    database_entry = entry
            # What `_carry_back_images` did not copy is withheld too, and has no entry.
            _write_provenance_record(staging, carried)
            ledger["copied_back"].append({
                "kind": "solve", "name": SOLVER_IMAGES_PROVENANCE, "written": True,
                "to": str(solve_live / SOLVER_IMAGES_PROVENANCE), "images": len(carried)})
            if database_entry is not None:
                # Every image not carried back -- withheld when judged, AND one that changed
                # between being judged and being copied (it was read from the IMAGES entry;
                # it used to be read from the database's, which never has it, so such an
                # image kept its old features).
                clear = set(carry.get("clear_features") or ()) | set(changed_while_copying)
                from tower.world_builder.global_solve import (  # noqa: PLC0415
                    clear_image_features,
                )

                database_entry["features_cleared"] = clear_image_features(
                    staging / "database.db", sorted(clear))
    except Exception as exc:  # noqa: BLE001 -- nothing has moved yet
        raise fail(LEDGER_ROLLED_BACK, "copying", exc) from exc

    # 3 and 4, under ONE rollback (M-6).
    done: list = []
    placed = False
    doing = "moving"
    try:
        for m in moves:
            src, dst = Path(m["from"]), Path(m["to"])
            doing = f"moving {m['kind']} {src.name}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            _replace_with_retry(src, dst)
            m["moved"] = True
            done.append(m)
            _write_ledger(aside, ledger)
        if staging.exists():
            doing = "placing the fresh solve directory"
            _replace_with_retry(staging, solve_live)
            placed = True
        doing = "recording the set-aside"
        ledger["state"] = LEDGER_SET_ASIDE
        _write_ledger(aside, ledger)
    except BaseException as exc:
        all_back = True
        if placed:
            # The staged copy goes back beside the ledger first: the solve cannot be
            # moved back over it.
            try:
                _replace_with_retry(solve_live, staging)
                placed = False
            except OSError as undo:
                all_back = False
                ledger["unplace_error"] = f"{type(undo).__name__}: {undo}"
        back = [m for m in done if not (placed and m["kind"] == "solve")]
        for m in done:
            if m not in back:
                m["moved_back"] = False
                m["move_back_error"] = ("skipped: the fresh solve directory still occupies "
                                        f"{m['from']}")
        all_back = _move_back(back, ledger, aside) and all_back
        failure = fail(LEDGER_ROLLED_BACK if all_back else LEDGER_ROLLBACK_INCOMPLETE,
                       doing, exc)
        if not isinstance(exc, Exception):
            raise          # an interrupt or an exit, after the world was put back
        raise failure from exc
    return ledger


# What each put-back attempt moves out of the way is kept under its own name (review V10,
# MED-2): the first attempt's are the names every earlier ledger used, and a later attempt --
# after one that was killed part-way -- gets `failed-solve-2`, `failed-derived-2`, ..., so it
# never collides with what an earlier one kept. Nothing is ever moved or copied ONTO a path
# that exists (`_unused`).
FAILED_SOLVE = "failed-solve"
FAILED_AREAS = "failed-areas"
FAILED_DERIVED = "failed-derived"
# Where an attempt stages its copy of the derived snapshot before renaming it into place.
PUT_BACK_STAGING = "putting-back"
DERIVED_DIRNAME = "derived"
DERIVED_MANIFEST = "manifest.json"


def _attempt_dir(aside: Path, name: str, attempt: int) -> Path:
    return aside / (name if attempt <= 1 else f"{name}-{attempt}")


def _unused(path: Path) -> Path:
    """`path`, or the first `path.<k>` (k = 2, 3, ...) that does not exist."""
    if not os.path.lexists(path):
        return path
    k = 2
    while os.path.lexists(path.with_name(f"{path.name}.{k}")):
        k += 1
    return path.with_name(f"{path.name}.{k}")


def restore_after_failed_solve(store: WorldStore, world_id: str, session_id: str,
                               ledger: dict, *, state: str = LEDGER_RESTORED) -> dict:
    """Put the set-aside result back after a final solve that did not publish (review
    V6, L3), after any exception between the set-aside and a published solve, or after the
    re-finish died (review V9, M-5) -- under the caller's lock. Nothing is deleted: what
    took the previous result's place is moved under `refinish/<stamp>/failed-*`. The
    session record is written back from its snapshot, so the session says what is on disk
    again.

    THE DISK DECIDES, NOT THE LEDGER, SO IT CAN BE RUN AGAIN (review V10, MED-2). A put-back
    can itself be killed part-way (the Tower closing: each retried move can take seconds),
    and it used to write its ledger only at the end: the next one trusted the stale ledger,
    took the ORIGINAL solve it had already moved back for the rebuild's, and filed it as
    `failed-solve` (RV10 R1, RV10-C probe 1a); or collided with its own `failed-solve/` and
    parked a world that was whole (R2, 1b); or undid an owner's own put-back done by the
    ledger's `restore` text (probe 2). Now, for every move the set-aside recorded:
    - the set-aside original (`to`) there and `from` free: moved back;
    - `from` there and `to` gone: it IS back -- never moved, or moved back by an earlier
      attempt or by hand -- and nothing is touched (`found_on_disk`);
    - both there: what occupies `from` came after the original was moved (the rebuild's
      fresh solve directory), and it is moved aside -- ONLY then, because the original it
      makes room for is verified to exist;
    - neither: an error, and the session is parked.
    The ledger is written after every move, and every attempt is recorded in it
    (`restore_attempts`) BEFORE it moves anything, with its own `failed-*` names.

    ONLY THIS SESSION'S DERIVED TREE (review V10, MED-3). `derived/` is the WORLD's:
    `derived/<session>/` per session, and `derived/manifest.json` naming whichever session
    built last. A put-back that runs later than the re-finish (the finisher's) used to swap
    the whole tree for the set-aside snapshot, and a walk of another session in between lost
    its derived output from the live world (RV10-C probe 3). Now `derived/<session>` goes
    back, and the world manifest only while it still names this session -- each only where
    it differs from the snapshot. And when either must go back while anything else under
    `derived/` differs from the snapshot (another session was built since the set-aside,
    maybe against the rebuilt tree), nothing at all is moved: the session is PARKED
    (`restore-incomplete`, and the finisher says so on the row with a sentence of the
    closed set), for an owner. See `derived_put_back_plan`.

    EVERY STEP IS ATTEMPTED, whatever an earlier one did: the moves that put the old
    solve and areas back must not be skipped because, say, the session record cannot
    be written (the very file whose write failed in PV's attempt 1). What could not be
    done is listed in `errors`, and the ledger then says `restore-incomplete` rather
    than `state`. A session record that already equals its snapshot byte for byte is
    left alone -- nothing moved it, so there is nothing to write back.

    Raises only when the ledger cannot be written before the first move (then nothing
    moved) or at the end."""
    from tower.storage import write_json_atomic  # noqa: PLC0415

    world_dir = store.world_dir(world_id)
    aside = world_dir / REFINISH_DIRNAME / ledger["stamp"]
    attempts = ledger.get("restore_attempts")
    if not isinstance(attempts, list):
        attempts = ledger["restore_attempts"] = []
    number = len(attempts) + 1
    out = {"attempt": number, "restored": [], "kept": [], "errors": [], "found_on_disk": []}
    attempts.append({"attempt": number, "started_at": time.time(), "for": state,
                     "failed_under": [_attempt_dir(aside, n, number).name
                                      for n in (FAILED_SOLVE, FAILED_AREAS, FAILED_DERIVED)],
                     "report": out})
    # The ledger FIRST, naming where this attempt keeps what it moves: nothing moves
    # without a record of where it went.
    _write_ledger(aside, ledger)

    def attempt(what: str, step) -> bool:
        try:
            step()
            return True
        except Exception as exc:  # noqa: BLE001 -- recorded; the next step still runs
            out["errors"].append(f"{what}: {type(exc).__name__}: {exc}")
            return False

    def finish() -> dict:
        ledger["state"] = LEDGER_RESTORE_INCOMPLETE if out["errors"] else state
        ledger["restore_report"] = out
        attempts[-1]["ended_at"] = time.time()
        _write_ledger(aside, ledger)
        return out

    # A set-aside that never completed (a re-finish that died in step 1) rebuilt nothing
    # in place, and its snapshots may be partial: only its moves are undone.
    snapshots = ledger.get("state") != LEDGER_SETTING_ASIDE
    derived = None
    if snapshots:
        derived = derived_put_back_plan(store, world_id, session_id, ledger)
        out["derived"] = {k: v for k, v in derived.items() if k != "park"}
        if derived["park"]:
            # Decided before anything moves: the world stays exactly as the re-finish left
            # it, and an owner decides.
            out["errors"].append(f"the derived tree: {derived['park']}; nothing was moved")
            out["parked"] = derived["park"]
            return finish()

    # The solve first (it frees nothing the areas need), then the areas, newest last.
    moves = [m for m in ledger.get("moved") or () if isinstance(m, dict)]
    moves.sort(key=lambda m: 0 if m.get("kind") == "solve" else 1)
    for m in moves:
        attempt(f"putting {m.get('kind')} {Path(str(m.get('from'))).name} back",
                lambda m=m: _put_move_back(m, aside, number, ledger, out))
    if not any(m.get("kind") == "solve" for m in moves):
        live_solve = world_dir / "solve" / session_id
        if live_solve.exists():
            # The session had no solve to set aside: what is there now is the rebuild's, and
            # there is no original to make room for, so it stays (review V10, MED-2: never
            # moved aside unless the original it makes room for exists).
            out["solve_left_in_place"] = str(live_solve)
    if derived is not None and (derived["revert_session"] or derived["revert_manifest"]):
        attempt("restoring the derived tree",
                lambda: _put_derived_back(store, world_id, session_id, ledger, derived,
                                          number, out))
    session_copy = aside / "session.json"
    if snapshots and session_copy.is_file():
        live_session = store.session_path(world_id, session_id)

        def restore_session():
            if live_session.is_file() and live_session.read_bytes() == session_copy.read_bytes():
                out["unchanged"] = str(live_session)
                return
            write_json_atomic(live_session,
                              json.loads(session_copy.read_text(encoding="utf-8")))
            out["restored"].append(str(live_session))
            _write_ledger_quietly(aside, ledger)

        attempt("restoring the session record", restore_session)
    # THE FINISHER'S COUNTERS (review V9, LOW). A re-finish now restarts them only after
    # its solve is published (`refinish`), so a put-back never meets restarted counters.
    # A ledger from before that restarted them in step 1 and kept the old ones here: they
    # go back, each only while it still carries THIS stamp's restart -- a counter the
    # finisher has moved on since is its own.
    prior = (ledger.get("previous") or {}).get("finish_attempts")
    if isinstance(prior, dict) and prior and not prior.get("unreadable"):
        attempt("restoring the finisher's attempt counters",
                lambda: _restore_attempts(store, world_id, ledger["stamp"], prior, out))
    return finish()


def _sha1_or_none(path: Path) -> str | None:
    import hashlib  # noqa: PLC0415

    try:
        return hashlib.sha1(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _put_move_back(m: dict, aside: Path, number: int, ledger: dict, out: dict) -> None:
    """One recorded move of the set-aside, put back as the disk says (see
    `restore_after_failed_solve`). Raises when it cannot; the caller records it."""
    home, away = Path(m["from"]), Path(m["to"])
    home_there, away_there = os.path.lexists(home), os.path.lexists(away)
    if not away_there:
        if not home_there:
            raise FileNotFoundError(f"neither the set-aside {away} nor {home} exists")
        # In place: never moved, or back already -- when it is the original. A solve whose
        # set-aside copy is gone is back only if `from` holds the solution the set-aside
        # recorded (a ledger from before V10 recorded none, and is taken at the disk's word).
        recorded = m.get("solution_sha1")
        if (m.get("kind") == "solve" and m.get("moved") and recorded is not None
                and _sha1_or_none(home / "solution.json") != recorded):
            raise FileNotFoundError(f"the set-aside solve {away} is gone, and what is at {home} "
                                    "is not it; nothing was moved")
        if m.get("moved") and not m.get("moved_back"):
            m["moved_back"] = True
            m["found_back_on_disk"] = True
            out["found_on_disk"].append(str(home))
            _write_ledger_quietly(aside, ledger)
        elif not m.get("moved") and m.get("kind") == "solve":
            out["solve_never_moved"] = str(home)
        return
    if home_there:
        # The original is at `away`, so what is at `home` took its place after the move --
        # the rebuild's fresh solve directory. Kept, never deleted.
        kind = FAILED_SOLVE if m.get("kind") == "solve" else FAILED_AREAS
        kept = _unused(_attempt_dir(aside, kind, number) / home.name)
        kept.parent.mkdir(parents=True, exist_ok=True)
        _replace_with_retry(home, kept)
        m.setdefault("displaced_to", []).append(str(kept))
        out["kept"].append(str(kept))
        _write_ledger_quietly(aside, ledger)
    home.parent.mkdir(parents=True, exist_ok=True)
    _replace_with_retry(away, home)
    m["moved"] = True
    m["moved_back"] = True
    out["restored"].append(str(home))
    _write_ledger_quietly(aside, ledger)


def _is_staging_name(name: str) -> bool:
    """A writer's staging file (`storage.staging_path`: `<name>.p<pid>.<hex>.tmp`, and
    `prepare_images`' `.tmp.jpg`): a write in flight or a dead writer's leftover, never
    content."""
    parts = name.split(".")
    return "tmp" in parts and any(len(p) > 1 and p[0] == "p" and p[1:].isdigit()
                                  for p in parts)


def _files_under(root: Path) -> dict:
    """relative POSIX path -> file, for every file under `root` ({} when there is none)."""
    if not root.is_dir():
        return {}
    return {p.relative_to(root).as_posix(): p for p in root.rglob("*")
            if p.is_file() and not _is_staging_name(p.name)}


def _same_bytes(a: Path | None, b: Path | None) -> bool:
    """Whether two files hold the same bytes (two absent files are the same; an unreadable
    one is different -- the careful answer here)."""
    if a is None or b is None:
        return a is None and b is None
    try:
        sa, sb = a.stat(), b.stat()
        if sa.st_size != sb.st_size:
            return False
        if sa.st_mtime_ns == sb.st_mtime_ns:
            # The snapshot is a `copy2` (the time kept), and every writer of this tree replaces
            # a file whole (`write_json_atomic`), which gives it a new time: the same size and
            # time is the same file, without reading a world's megabytes twice.
            return True
        with a.open("rb") as fa, b.open("rb") as fb:
            while True:
                x, y = fa.read(1 << 20), fb.read(1 << 20)
                if x != y:
                    return False
                if not x:
                    return True
    except OSError:
        return False


def _manifest_session(path: Path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data.get("session_id") if isinstance(data, dict) else None


def derived_put_back_plan(store: WorldStore, world_id: str, session_id: str,
                          ledger: dict) -> dict:
    """What putting back this session's derived output means, read off the disk against the
    set-aside snapshot (review V10, MED-3). Reads only. Returns
    `{"revert_session", "revert_manifest", "changed_elsewhere", "park"}`:

    - `revert_session`: `derived/<session>/` differs from the snapshot's (a file added,
      gone or changed -- the rebuild's `engine.build` wrote it);
    - `revert_manifest`: the world's `derived/manifest.json` names THIS session and
      differs from the snapshot's (the rebuild wrote it last). One that names another
      session is that session's, and never this put-back's to change;
    - `changed_elsewhere`: every other file under `derived/` that differs from the snapshot
      -- another session built since the set-aside (a writer's staging file is not
      content);
    - `park`: why nothing may be moved, or None. When this session's output must go back
      while other output changed since the set-aside, that other output may have been built
      against the rebuilt tree, and reverting under it could leave the world inconsistent:
      an owner decides. When nothing of this session's must go back, other sessions'
      changes are simply theirs.

    A world that had no `derived/` when it was set aside (no `derived` copy in the ledger)
    is compared with an empty snapshot; one that had one whose snapshot is gone parks."""
    world_dir = store.world_dir(world_id)
    aside = world_dir / REFINISH_DIRNAME / ledger["stamp"]
    snapshot = aside / DERIVED_DIRNAME
    snapshotted = any(isinstance(c, dict) and c.get("kind") == "derived"
                      for c in ledger.get("copied") or ())
    live = world_dir / DERIVED_DIRNAME
    if snapshotted and not snapshot.is_dir():
        return {"revert_session": False, "revert_manifest": False, "changed_elsewhere": [],
                "park": f"its set-aside snapshot {snapshot} is gone"}
    snap = _files_under(snapshot) if snapshotted else {}
    here = _files_under(live)
    prefix = f"{session_id}/"
    mine = {rel for rel in set(snap) | set(here) if rel.startswith(prefix)}
    revert_session = any(not _same_bytes(snap.get(rel), here.get(rel)) for rel in mine)
    manifest_mine = _manifest_session(live / DERIVED_MANIFEST) == session_id
    revert_manifest = manifest_mine and not _same_bytes(snap.get(DERIVED_MANIFEST),
                                                        here.get(DERIVED_MANIFEST))
    elsewhere = sorted(rel for rel in set(snap) | set(here)
                       if rel not in mine and not (manifest_mine and rel == DERIVED_MANIFEST)
                       and not _same_bytes(snap.get(rel), here.get(rel)))
    park = None
    if (revert_session or revert_manifest) and elsewhere:
        shown = ", ".join(elsewhere[:5]) + (f" and {len(elsewhere) - 5} more"
                                             if len(elsewhere) > 5 else "")
        park = (f"this session's derived output must go back, but other derived output "
                f"changed after the set-aside ({shown}); putting it back could undo what was "
                "built on it")
    return {"revert_session": revert_session, "revert_manifest": revert_manifest,
            "changed_elsewhere": elsewhere, "park": park}


def _put_derived_back(store: WorldStore, world_id: str, session_id: str, ledger: dict,
                      plan: dict, number: int, out: dict) -> None:
    """`derived/<session>` and (while it names this session) the world manifest, back from
    the snapshot, each only where `plan` says it differs. The snapshot is COPIED to a staging
    directory beside the ledger first and renamed into place, so a kill leaves either the
    rebuild's tree, or none (the next attempt then puts a fresh copy in), never a half-copied
    one; the rebuild's is kept under `failed-derived`."""
    world_dir = store.world_dir(world_id)
    aside = world_dir / REFINISH_DIRNAME / ledger["stamp"]
    snapshot = aside / DERIVED_DIRNAME
    live = world_dir / DERIVED_DIRNAME
    failed = _attempt_dir(aside, FAILED_DERIVED, number)
    staging = _attempt_dir(aside, PUT_BACK_STAGING, number)
    if plan["revert_session"]:
        target = live / session_id
        staged = None
        if (snapshot / session_id).is_dir():
            staged = _unused(staging / DERIVED_DIRNAME / session_id)
            staged.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(snapshot / session_id, staged)
        if os.path.lexists(target):
            kept = _unused(failed / session_id)
            kept.parent.mkdir(parents=True, exist_ok=True)
            _replace_with_retry(target, kept)
            out["kept"].append(str(kept))
            _write_ledger_quietly(aside, ledger)
        if staged is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            _replace_with_retry(staged, target)
            out["restored"].append(str(target))
            _write_ledger_quietly(aside, ledger)
    if plan["revert_manifest"]:
        target = live / DERIVED_MANIFEST
        kept = _unused(failed / DERIVED_MANIFEST)
        kept.parent.mkdir(parents=True, exist_ok=True)
        if (snapshot / DERIVED_MANIFEST).is_file():
            # Copied aside, then replaced in ONE rename: the live manifest is never absent.
            shutil.copy2(target, kept)
            staged = _unused(staging / DERIVED_MANIFEST)
            staged.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(snapshot / DERIVED_MANIFEST, staged)
            _replace_with_retry(staged, target)
            out["restored"].append(str(target))
        else:
            _replace_with_retry(target, kept)        # the world had none before
        out["kept"].append(str(kept))
        _write_ledger_quietly(aside, ledger)


def _restart_detail(stamp: str) -> str:
    return f"restarted by scripts/world_refinish.py ({stamp})"


def _restore_attempts(store: WorldStore, world_id: str, stamp: str, prior: dict,
                      out: dict) -> None:
    """Write the finisher's counters `prior` back, where a re-finish of `stamp` restarted
    them and nothing has touched them since."""
    from scripts import world_finish_pending as wfp  # noqa: PLC0415

    with wfp._LEDGER_LOCK:
        sessions, unreadable = wfp._read_ledger(store, world_id)
        if unreadable:
            raise RuntimeError("the finisher's attempt ledger is unreadable")
        restored = [key for key, entry in prior.items() if isinstance(entry, dict)
                    and (sessions.get(key) or {}).get("detail") == _restart_detail(stamp)]
        if restored:
            sessions = dict(sessions)
            for key in restored:
                sessions[key] = prior[key]
            wfp._write_ledger(store, world_id, sessions)
    out["finish_attempts_restored"] = restored


# A RE-FINISH THAT DIED BEFORE IT PUBLISHED (review V9, M-5). A hard kill in step 1 or 2 --
# a console closed, a reboot -- has no `finally`: the solve stays set aside, the room stays
# marked `stopped`, and the ledger stays at `setting-aside` or `set-aside`. The finisher
# used to read the dead process as "not holding the world", owe the stopped room, and run
# the room's stages on a session with NO solution: `surface: unavailable` written into the
# live status, the ledger never moving on, no notice (RV9-D probe C, RV9-E F4).
#
# WHY PUT IT BACK, AND NOT PARK THE WORLD. Everything the re-finish moved is intact under
# `refinish/<stamp>/` and named by its ledger (renames, recorded move by move), so putting
# it back is exactly what the re-finish itself does after any failure before a publish
# (`restore_after_failed_solve`), under the same lock, deleting nothing: the world is
# again the published, consistent world it was before, and an owner who still wants the
# re-finish runs it again. Parking would leave the phone a world with its solve moved
# away -- no components, no room to rebuild from -- until an owner noticed; and the
# process being gone is established by pid AND start time, so a live re-finish is never
# put back under its own feet. The finisher parks a world only if the put-back itself
# fails (`world_finish_pending`).


def session_ledgers(store: WorldStore, world_id: str, session_id: str) -> list:
    """`(stamp, ledger)` for every readable re-finish ledger of this session, oldest first
    (stamps sort by time). Reads only; [] when there is no `refinish/` or it cannot be
    listed."""
    from tower.storage import read_json_closed  # noqa: PLC0415

    root = store.world_dir(world_id) / REFINISH_DIRNAME
    try:
        stamps = sorted(entry for entry in root.iterdir() if entry.is_dir())
    except OSError:
        return []
    out = []
    for aside in stamps:
        try:
            ledger = read_json_closed(aside / LEDGER_FILENAME)
        except (OSError, ValueError):
            continue
        if isinstance(ledger, dict) and ledger.get("session_id") == session_id:
            out.append((aside.name, ledger))
    return out


def dead_before_publish(store: WorldStore, world_id: str, session_id: str) -> dict | None:
    """The re-finish of this session that died before its new solve was published and has
    not been put back: `{"stamp", "state"}`, or None. Only the session's NEWEST ledger
    counts -- a later re-finish has since acted on what the earlier one left, and putting
    the earlier one back would move things under it. Dead is established by pid and start
    time (`refinish_process_alive` False); unknown is not dead. Reads only."""
    ledgers = session_ledgers(store, world_id, session_id)
    if not ledgers:
        return None
    stamp, ledger = ledgers[-1]
    if ledger.get("state") in LEDGER_BEFORE_PUBLISH and refinish_process_alive(ledger) is False:
        return {"stamp": stamp, "state": ledger.get("state")}
    return None


def recover_dead_refinish(store: WorldStore, world_id: str, stamp: str) -> dict:
    """Put back what the re-finish `stamp` set aside, when its process ended before its new
    solve was published. Under the caller's writer lock (the finisher's). Returns
    `{"state": <the ledger's state now>, "recovered": bool, ...}`; raises only when the
    ledger cannot be read or written.

    Per what the process left (the ledger is written after every move, so a kill between
    a rename and its record is the only gap, and the disk closes it):
    - `setting-aside`: step 1 did not complete. A move the ledger does not record but the
      disk shows (its source gone, its destination there) counts as moved; the moves are
      undone, and nothing else -- the rebuild replaced nothing in place yet, and its
      snapshots may be partial. A solve that never moved is left where it is.
    - `set-aside`: the moves are undone and the session record and this session's derived
      output go back from their snapshots (or the session parks: see
      `restore_after_failed_solve`) -- unless the solve child had PUBLISHED before the re-finish
      ended (a solution in the fresh solve directory, and a completed, solved
      finalization written after the set-aside). Then the new solve stands: the ledger
      says `published`, the finisher's counters are restarted (as the re-finish itself
      does after its publish; review V10, L-6), and the room the re-finish marked is owed
      to the finisher, as for any re-finish that died after its publish.

    Dead means the re-finish's process AND the final solve's child it started are both
    known to have ended (`refinish_liveness`; review V10, Q3): nothing is put back while the
    child may still publish into the directory being put back. A put-back that was itself
    interrupted is finished by the next call (`restore_after_failed_solve` reads the disk)."""
    from tower.storage import read_json_closed  # noqa: PLC0415

    aside = store.world_dir(world_id) / REFINISH_DIRNAME / stamp
    ledger = read_json_closed(aside / LEDGER_FILENAME)
    if not isinstance(ledger, dict):
        raise ValueError(f"{aside / LEDGER_FILENAME} is not a re-finish ledger")
    state = ledger.get("state")
    if state not in LEDGER_BEFORE_PUBLISH:
        return {"state": state, "recovered": False,
                "why": "the ledger is not at a step before the publish"}
    liveness = refinish_liveness(ledger)
    if liveness["alive"] is not False:
        return {"state": state, "recovered": False, "liveness": liveness,
                "why": ("its final solve's child is still running" if liveness["child"]
                        else "its process is not known to have ended")}
    session_id = ledger.get("session_id")
    found = []
    for m in ledger.get("moved") or ():
        if m.get("moved") or m.get("moved_back"):
            continue
        if Path(m["to"]).exists() and not Path(m["from"]).exists():
            m["moved"] = True
            m["found_moved_on_disk"] = True
            found.append(m["from"])
    noticed = ledger.get("ended_before_publish")
    if not isinstance(noticed, dict):
        noticed = ledger["ended_before_publish"] = {
            "state": state, "noticed_at": time.time(), "by": "scripts/world_finish_pending.py",
            "moves_found_on_disk": []}
    noticed.setdefault("moves_found_on_disk", []).extend(found)
    if state == LEDGER_SET_ASIDE and _published_since_set_aside(store, world_id, session_id,
                                                                ledger):
        # The counters first, then the state: a kill in between is published again by the
        # next call, and `_restart_attempts` does not restart a counter twice.
        restart_error = None
        try:
            restarted = _restart_attempts(store, world_id, session_id, stamp)
            previous = ledger.get("previous")
            if not isinstance(previous, dict):
                previous = ledger["previous"] = {}
            if not isinstance(previous.get("finish_attempts"), dict):
                previous["finish_attempts"] = {}
            previous["finish_attempts"].update(restarted)
            ledger["finish_attempts_restarted_at"] = time.time()
        except Exception as exc:  # noqa: BLE001 -- recorded; the new solve stands either way
            restart_error = ledger["finish_attempts_error"] = (
                f"the finisher's attempt counters could not be restarted: "
                f"{type(exc).__name__}: {exc}")
        ledger["state"] = LEDGER_PUBLISHED
        ledger["published_at"] = time.time()
        ledger["detail"] = ("the final solve had published before the re-finish's process "
                            "ended; the room it marked is owed to the finisher")
        _write_ledger(aside, ledger)
        return {"state": LEDGER_PUBLISHED, "recovered": True, "published": True,
                **({"finish_attempts_error": restart_error} if restart_error else {})}
    out = restore_after_failed_solve(store, world_id, session_id, ledger,
                                     state=LEDGER_RESTORED_AFTER_DEATH)
    return {"state": ledger["state"], "recovered": True, "restore_report": out}


def _published_since_set_aside(store: WorldStore, world_id: str, session_id: str,
                               ledger: dict) -> bool:
    """Whether the re-finish's solve child published before the re-finish ended: the
    previous solve moved out -- and still out, ON DISK (review V10, MED-2: a put-back that
    was killed after moving it back left the ledger saying otherwise, and the solution
    then at `solve/<session>` is the ORIGINAL's) --, a solution in its place (the fresh
    directory is copied WITHOUT one), and a complete, solved finalization written after the
    set-aside."""
    from tower.world_builder.global_solve import load_solution  # noqa: PLC0415
    from tower.world_builder.records import (  # noqa: PLC0415
        FINAL_SOLVE_SOLVED,
        FINALIZATION_COMPLETE,
    )

    solve_move = next((m for m in ledger.get("moved") or () if m.get("kind") == "solve"), None)
    if solve_move is None or not solve_move.get("moved") or solve_move.get("moved_back"):
        return False
    if not os.path.lexists(solve_move.get("to") or ""):
        return False
    try:
        if load_solution(store, world_id, session_id) is None:
            return False
        fin = store.read_session(world_id, session_id).finalization or {}
    except Exception:  # noqa: BLE001 -- unreadable is not published
        return False
    at, since = fin.get("updated_at"), ledger.get("set_aside_at")
    return (fin.get("state") == FINALIZATION_COMPLETE
            and fin.get("final_solve") == FINAL_SOLVE_SOLVED
            and isinstance(at, (int, float)) and isinstance(since, (int, float))
            and at > since)


def build_in_progress(store: WorldStore, world_id: str, session_id: str) -> str | None:
    """Why this session is being built right now, or None (review V6, H1).

    The writer lock is NOT enough: the builder releases it at the end of
    finalization and only then runs the room's final surface and appearance (six to
    sixteen minutes), and an area build may be live too. So: the Tower's own probe for
    the room (`session_build_running`: the lock while the session is open or
    finalizing, and every room stage status `running` under a live pid), and the same
    per-stage probe over this session's areas."""
    from tower.results.world_builder_render import (  # noqa: PLC0415
        _stage_running,
        session_build_running,
    )
    from tower.world_builder.components import AreaStore, session_area_dirs  # noqa: PLC0415
    from tower.world_builder.surface_pipeline import status_is_stale  # noqa: PLC0415

    if session_build_running(store, world_id, session_id):
        return ("a photographic stage of this session is running under a live process "
                "(the builder finishes the room after it releases the world's lock)")
    for area in session_area_dirs(store, world_id, session_id):
        view = AreaStore(store, world_id, session_id, area.name)
        for stage in ("surface", "appearance"):
            status = view.world_dir(world_id) / stage / session_id / "status.json"
            if _stage_running(status, status_is_stale):
                return f"area {area.name} of this session is being built right now"
    return None


REFINISH_IN_PROGRESS = (
    "re-finish in progress (scripts/world_refinish.py): the solve this was built from "
    "is set aside under refinish/{stamp}/, and the room is rebuilt after the new solve")


# WHERE THE SOLVER'S FRAMES COME FROM (P3 finding 6: 991e5a15 and af47007c were
# re-finished from the face-redacted session keyframes although their raw captures
# were on disk). The builder records, per keyframe, the raw frame it observed
# (`sources.json`); a world without that record fell back to the session copies.
#
# WHICH CAPTURES. `--capture-dir DIR` (repeatable) when the owner names them;
# otherwise the capture this session RECORDS (`session.capture_id`, set only for a
# live-capture session) under the owner's configured capture root,
# `TOWER_CAPTURE_ROOT` -- the setting that arms the raw recorder at all, so there is
# raw imagery on disk only where the owner configured it -- together with every
# capture that CONTINUES it after a reconnect (`continues_capture`), as the builder's
# follower walked them (`_capture_chain`). A relative root is anchored where
# `sources.json` paths are (`global_solve.resolve_source_path`), never the cwd.
# `--no-capture`, or nothing resolvable: today's behaviour.
#
# WHICH FRAME, AND NEVER ANOTHER CAPTURE'S (review of 3abe763, ACC). A reconnect
# restarts the phone's frame numbering, so the captures of one walk REUSE frame
# names: on the live walk adc75972 (3 captures, 353 keyframes) 152 keyframes have a
# same-named frame in another capture of the chain, and 20 keyframe ids occur twice
# in the session itself. Matching by name gave 67 keyframes another moment's image.
# So a keyframe's raw frame is found by ITS OWN IDENTITY in the captures' journals
# (`frames.jsonl`): the one journal record with the keyframe's `source_seq` AND its
# `received_at` (the Tower's receipt time, which the follower copies from that very
# record into the keyframe -- measured equal to the microsecond for 353 of 353 on
# adc75972) and, where both carry one, its `wire_seq`. No record, or more than one:
# that keyframe keeps its stored redacted copy. A keyframe id (or image name) that
# occurs more than once in the session cannot be told apart by anything the solver
# is given, so those keyframes keep the stored copy too. A capture directory with no
# journal is not searched at all: a bare directory of frames has names only.
#
# HOW THE SOLVE GETS THEM. As `sources.json` in the fresh solve directory, keyframe
# id -> exact frame -- the same record the builder writes -- and NEVER as
# `--capture-dir`, whose by-name lookup (`global_solve._source_frame`) is the defect.
# An entry the walk's builder recorded is kept (it observed the frame), except for an
# ambiguous keyframe id. The raw frames are solver input only, exactly as the
# builder's: they never leave this machine, point colours are withheld
# (`global_solve.WITHHELD_POINT_RGB`), and the room's surface and appearance still
# read the redacted keyframes.
CAPTURE_ROOT_ENV = "TOWER_CAPTURE_ROOT"
CAPTURE_FROM_ARGUMENT = "--capture-dir"
CAPTURE_FROM_SESSION = "TOWER_CAPTURE_ROOT + the session's capture_id"
CAPTURE_MANIFEST = "capture.json"
CAPTURE_JOURNAL = "frames.jsonl"
# `received_at` is compared at microsecond resolution: the follower copies the float
# from the journal, so equal frames are equal to the last digit; a microsecond only
# absorbs a float's text round-trip. Two frames of one capture are milliseconds apart.
RECEIVED_AT_DECIMALS = 6


def resolve_capture_dirs(store: WorldStore, world_id: str, session_id: str,
                         explicit=(), *, use_capture: bool = True) -> dict:
    """The raw capture directories to SEARCH, by frame identity, and why. Reads only."""
    if not use_capture:
        return {"capture_dirs": [], "from": None, "why": "--no-capture"}
    if explicit:
        # ABSOLUTE, from the owner's working directory, which is where they named it
        # (review V9, LOW). The frames found in it are written into `sources.json`, whose
        # relative paths every reader resolves against the Tower's `tower/`
        # (`resolve_source_path`), never the cwd: a relative `--capture-dir` was counted
        # raw here and then not found by the solve, which fell back to the redacted
        # copies without a word (RV9-D probe G).
        return {"capture_dirs": [str(Path(d).resolve()) for d in explicit],
                "from": CAPTURE_FROM_ARGUMENT}
    capture_id = getattr(store.read_session(world_id, session_id), "capture_id", None)
    if not capture_id:
        return {"capture_dirs": [], "from": None,
                "why": "the session records no capture (it was not a live capture)"}
    if Path(capture_id).name != capture_id or capture_id in (".", ".."):
        return {"capture_dirs": [], "from": None, "capture_id": capture_id,
                "why": "the session's capture_id is not a plain directory name"}
    root = (os.environ.get(CAPTURE_ROOT_ENV) or "").strip()
    if not root:
        return {"capture_dirs": [], "from": None, "capture_id": capture_id,
                "why": f"{CAPTURE_ROOT_ENV} is not set"}
    from tower.world_builder.global_solve import resolve_source_path  # noqa: PLC0415

    captures_root = resolve_source_path(root) / "captures"
    candidate = captures_root / capture_id
    if not (candidate / "frames").is_dir():
        return {"capture_dirs": [], "from": None, "capture_id": capture_id,
                "why": f"no capture frames at {candidate}"}
    chain = _capture_chain(captures_root, capture_id)
    return {"capture_dirs": [str(captures_root / c["capture_id"]) for c in chain],
            "from": CAPTURE_FROM_SESSION, "capture_id": capture_id, "captures": chain}


def _capture_chain(captures_root: Path, first: str) -> list[dict]:
    """The session's capture and every capture that CONTINUES it (transitively), in
    walk order. A walk whose link drops goes on in a successor capture whose manifest
    names its predecessor in `continues_capture`, and the builder follows it
    (`tower.capture.CaptureFollower`) -- but the session records only the capture it
    started on. Measured on the run's frozen captures: 991e5a15's session spans 3
    captures (132 of its 229 keyframes in the first), af47007c's 3 (164 of 218).
    Every descendant is included, a fork's branches too: frames are matched by
    identity (`map_raw_frames`), so a capture that holds none of the session's frames
    contributes none. Reads only."""
    manifests: dict[str, dict] = {}
    try:
        with os.scandir(captures_root) as entries:
            for entry in entries:
                if not entry.is_dir():
                    continue
                try:
                    manifest = json.loads((Path(entry.path) / CAPTURE_MANIFEST)
                                          .read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if isinstance(manifest, dict):
                    manifests[entry.name] = manifest
    except OSError:
        pass
    chain, frontier = [first], [first]
    while frontier:
        parent = frontier.pop(0)
        children = sorted((c for c, m in manifests.items()
                           if m.get("continues_capture") == parent and c not in chain),
                          key=lambda c: (_as_float(manifests[c].get("started_at")), c))
        chain.extend(children)
        frontier.extend(children)
    return [{"capture_id": c,
             "continues_capture": manifests.get(c, {}).get("continues_capture"),
             "retains_raw_imagery": manifests.get(c, {}).get("retains_raw_imagery"),
             "redaction": manifests.get(c, {}).get("redaction")} for c in chain]


def _as_float(value) -> float:
    return float(value) if isinstance(value, (int, float)) else float("inf")


def _frame_key(source_seq, received_at) -> tuple | None:
    try:
        return int(source_seq), round(float(received_at), RECEIVED_AT_DECIMALS)
    except (TypeError, ValueError):
        return None


def _wire_seq(value):
    """A `wire_seq` as an int; None when absent; `False` when present but not an integer
    (review V9, LOW: one malformed journal field used to raise out of the whole plan).
    A record whose `wire_seq` cannot be read cannot be matched by it, so it is no match."""
    if value is None:
        return None
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else False
    try:
        return int(str(value).strip(), 10)
    except (TypeError, ValueError):
        return False


def _wire_seq_agrees(recorded, keyframe) -> bool:
    a, b = _wire_seq(recorded), _wire_seq(keyframe)
    if a is False or b is False:
        return False
    return a is None or b is None or a == b


def _journal_index(capture_dirs) -> tuple[dict, list[str]]:
    """(source_seq, received_at) -> every journal record with that identity, across
    `capture_dirs`, as `{"path", "wire_seq", "time_basis", "capture"}`; and notes on
    directories that could not be searched."""
    index: dict = {}
    notes: list[str] = []
    for capture_dir in capture_dirs:
        capture_dir = Path(capture_dir)
        journal = capture_dir / CAPTURE_JOURNAL
        try:
            lines = journal.read_text(encoding="utf-8").splitlines()
        except OSError:
            notes.append(f"{capture_dir}: no {CAPTURE_JOURNAL}; not searched (a frame "
                         "cannot be matched to a keyframe by name alone)")
            continue
        for line in lines:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if not isinstance(rec, dict) or not rec.get("relpath"):
                continue
            key = _frame_key(rec.get("source_seq"), rec.get("received_at"))
            if key is None:
                continue
            index.setdefault(key, []).append({
                "path": capture_dir / str(rec["relpath"]), "wire_seq": rec.get("wire_seq"),
                "time_basis": rec.get("time_basis"), "capture": capture_dir.name})
    return index, notes


def map_raw_frames(keyframes, capture_dirs) -> tuple[dict, dict]:
    """keyframe_id -> its OWN raw frame, found by identity (see above), for every
    keyframe that has exactly one; and why the others have none. Reads only."""
    from collections import Counter  # noqa: PLC0415

    from tower.world_builder.global_solve import keyframe_image_name  # noqa: PLC0415

    index, notes = _journal_index(capture_dirs)
    ids = Counter(k.keyframe_id for k in keyframes)
    names = Counter(keyframe_image_name(k) for k in keyframes)
    mapping: dict = {}
    why: dict = {"ambiguous_in_session": [], "no_frame": [], "several_frames": []}
    for kf in keyframes:
        if ids[kf.keyframe_id] > 1 or names[keyframe_image_name(kf)] > 1:
            why["ambiguous_in_session"].append(kf.keyframe_id)
            continue
        key = _frame_key(kf.source_seq, kf.received_at)
        hits = [h for h in index.get(key, []) if h["path"].is_file()
                and _wire_seq_agrees(h["wire_seq"], kf.wire_seq)
                and (h["time_basis"] is None or h["time_basis"] == kf.time_basis)]
        if len(hits) == 1:
            mapping[kf.keyframe_id] = str(hits[0]["path"])
        elif hits:
            why["several_frames"].append(kf.keyframe_id)
        else:
            why["no_frame"].append(kf.keyframe_id)
    why["notes"] = notes
    return mapping, why


# What `source` sums a plan up as. `walk-solver-images` was the answer when every
# keyframe reused the walk's solver image, whatever it had been made from; since review
# V9 (M-7) an image is reused only when its provenance is known, so a plan now always
# says where the pixels come from, and this label appears only in older ledgers.
SOURCE_WALK_IMAGES = "walk-solver-images"
SOURCE_RAW = "raw-capture"
SOURCE_REDACTED = "redacted-session-keyframes"
SOURCE_MIXED = "mixed"
# Keys of `plan_solver_frames`' answer that are for the re-finish itself, not the ledger.
PLAN_INTERNAL_KEYS = ("sources", "image_plan", "carry_back")


def plan_solver_frames(store: WorldStore, world_id: str, session_id: str,
                       capture_dirs) -> dict:
    """Where each keyframe's solver image will come from, the `sources.json` that makes it
    so, and which of the walk's solver images may be carried back. Reads only: `sources`
    in the result is what `apply_solver_frames` writes, `sources_changed` whether it
    differs from the record on disk, `carry_back` what `set_aside` copies back.

    Per keyframe record, in `global_solve.prepare_images`' own order: the raw frame the
    builder recorded (`raw_from_sources_json`); the raw frame found by identity in the
    captures (`raw_from_capture_dir`); else the stored redacted keyframe
    (`redacted_session_copies`, split by why). That is where its solver image's PIXELS
    come from, whether the image is written afresh or reused: `already_undistorted`
    counts the keyframes whose walk image is reused because it is proven to be exactly
    that (`carry_back_decisions`; review V9, M-7), and `walk_images` says what became of
    each of the walk's solver images."""
    from tower.world_builder.global_solve import (  # noqa: PLC0415
        keyframe_image_name,
        read_sources,
        resolve_source_path,
        workspace_for,
    )

    workspace = workspace_for(store, world_id, session_id)
    recorded = read_sources(workspace)
    keyframes = store.read_keyframes(world_id, session_id)
    mapping, why = map_raw_frames(keyframes, capture_dirs)
    ambiguous = set(why["ambiguous_in_session"])
    several = set(why["several_frames"])
    sources = {kid: path for kid, path in recorded.items() if kid not in ambiguous}
    counts = {"keyframes": 0, "already_undistorted": 0, "raw_from_sources_json": 0,
              "raw_from_capture_dir": 0, "redacted_session_copies": 0,
              "redacted_ambiguous_in_session": 0, "redacted_no_capture_frame": 0,
              "redacted_several_capture_frames": 0}
    for kf in keyframes:
        counts["keyframes"] += 1
        kid = kf.keyframe_id
        builder = None if kid in ambiguous else resolve_source_path(recorded.get(kid))
        builder_ok = builder is not None and builder.is_file()
        if not builder_ok and kid in mapping:
            sources[kid] = mapping[kid]
        if builder_ok:
            counts["raw_from_sources_json"] += 1
        elif kid in mapping:
            counts["raw_from_capture_dir"] += 1
        else:
            counts["redacted_session_copies"] += 1
            counts["redacted_ambiguous_in_session" if kid in ambiguous else
                   "redacted_several_capture_frames" if kid in several else
                   "redacted_no_capture_frame"] += 1
    image_plan = plan_solver_images(store, world_id, session_id, keyframes, sources,
                                    walk_sources=recorded)
    carry = carry_back_decisions(store, world_id, session_id, keyframes, image_plan)
    carried = carry["carried"]
    counts["already_undistorted"] = sum(1 for kf in keyframes
                                        if keyframe_image_name(kf) in carried)
    withheld: dict = {}
    for reason in carry["withheld"].values():
        withheld[reason] = withheld.get(reason, 0) + 1
    frame_gone = sum(1 for e in carried.values() if e.get("verified") == "record-frame-gone")
    counts["walk_images"] = {
        "carried_back": len(carried),
        "by_record": sum(1 for e in carried.values() if e.get("verified") == "record"),
        "by_reproduction": sum(1 for e in carried.values()
                               if e.get("verified") == "reproduced"),
        # L-8: the keyframe's raw frame is gone, and its record proves the walk image is it.
        "by_record_frame_gone": frame_gone,
        "withheld": withheld,
        "withheld_images": dict(sorted(carry["withheld"].items())),
        "features_to_clear": len(carry["clear_features"]),
        # L-9: reproduction's own account, and the differences nothing explains.
        "reproduction": carry["reproduction"],
    }
    if carry["unexplained"]:
        counts["walk_images"]["unexplained_images"] = carry["unexplained"]
    # Pixels from a raw frame: planned raw, or a raw walk image kept for a frame now gone (L-8).
    counts["raw_walk_images_frame_gone"] = frame_gone
    raw = counts["raw_from_sources_json"] + counts["raw_from_capture_dir"] + frame_gone
    redacted = counts["redacted_session_copies"] - frame_gone
    counts["source"] = (None if counts["keyframes"] == 0 else
                        SOURCE_RAW if redacted == 0 else
                        SOURCE_REDACTED if raw == 0 else SOURCE_MIXED)
    counts["matched_by"] = "source_seq + received_at (+ wire_seq) in the captures' journals"
    counts["ambiguous_keyframe_ids"] = sorted(ambiguous)
    counts["dropped_builder_entries"] = sorted(k for k in recorded if k in ambiguous)
    if why["notes"]:
        counts["capture_notes"] = why["notes"]
    counts["sources"] = sources
    counts["sources_changed"] = ({k: str(v) for k, v in sources.items()}
                                 != {k: str(v) for k, v in recorded.items()})
    counts["image_plan"] = image_plan
    counts["carry_back"] = carry
    return counts


def plan_solver_images(store: WorldStore, world_id: str, session_id: str, keyframes,
                       sources: dict, walk_sources: dict | None = None) -> dict:
    """image name -> where the solve child will undistort it from, given the planned
    `sources.json`: `{"keyframe_id", "source", "frame", "path"}`. The child's own rule
    (`global_solve._source_frame` with no capture directory, which a re-finish never
    passes): the entry's raw frame when that file exists, else the session's stored copy.
    For a name two keyframes share, the first in keyframe order -- the one
    `prepare_images` writes. Reads only.

    Two more keys, for `carry_back_decisions`: `raw_frame_gone`, the planned `sources.json`
    frame of a keyframe that falls back to its stored copy ONLY because that frame is no
    longer on disk (review V10, L-8); and `alternatives`, the other frames the walk's image
    could have been undistorted from -- the stored copy, and the frame the WALK's own
    `sources.json` (`walk_sources`) names -- which tell an image made from another source
    apart from one nothing reproduces (L-9)."""
    from tower.world_builder.global_solve import (  # noqa: PLC0415
        keyframe_image_name,
        resolve_source_path,
    )

    session_dir = store.session_dir(world_id, session_id)
    out: dict = {}
    for kf in keyframes:
        name = keyframe_image_name(kf)
        if name in out:
            continue
        recorded = sources.get(kf.keyframe_id)
        path = resolve_source_path(recorded)
        stored = session_dir / kf.image_relpath
        walk = resolve_source_path((walk_sources or {}).get(kf.keyframe_id))
        walk = walk if walk is not None and walk.is_file() else None
        if path is not None and path.is_file():
            out[name] = {"keyframe_id": kf.keyframe_id, "source": IMAGE_SOURCE_RAW,
                         "frame": str(recorded), "path": path,
                         "alternatives": [stored] + ([walk] if walk is not None
                                                     and not _same_path(walk, path) else [])}
        else:
            out[name] = {"keyframe_id": kf.keyframe_id, "source": IMAGE_SOURCE_REDACTED,
                         "frame": kf.image_relpath, "path": stored,
                         "alternatives": [walk] if walk is not None else []}
            if recorded is not None:
                out[name]["raw_frame_gone"] = str(recorded)
    return out


def read_image_provenance(solve_dir: Path) -> dict:
    """image name -> its provenance entry, from `SOLVER_IMAGES_PROVENANCE` in `solve_dir`;
    {} when there is none or it cannot be read. Reads only."""
    from tower.storage import read_json_closed  # noqa: PLC0415

    try:
        data = read_json_closed(Path(solve_dir) / SOLVER_IMAGES_PROVENANCE)
    except (OSError, ValueError):
        return {}
    if (not isinstance(data, dict) or data.get("record") != PROVENANCE_RECORD
            or not isinstance(data.get("images"), dict)):
        return {}
    return {k: v for k, v in data["images"].items() if isinstance(v, dict)}


def _same_path(a, b) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def _same_provenance(entry: dict, planned: dict) -> bool:
    """Whether a provenance entry names the frame the plan names."""
    from tower.world_builder.global_solve import resolve_source_path  # noqa: PLC0415

    if entry.get("source") != planned["source"]:
        return False
    if planned["source"] == IMAGE_SOURCE_RAW:
        frame = resolve_source_path(entry.get("frame"))
        return frame is not None and _same_path(frame, planned["path"])
    return (str(entry.get("frame") or "").replace("\\", "/")
            == str(planned["frame"]).replace("\\", "/"))


def _undistortion(store: WorldStore, world_id: str, session_id: str, keyframes):
    """`prepare_images`' undistortion for this session, or None when there is none."""
    from tower.world_builder.global_solve import _undistort_maps  # noqa: PLC0415

    if not keyframes:
        return None
    width, height = keyframes[0].width, keyframes[0].height
    try:
        m1, m2, roi, _camera = _undistort_maps(
            store.read_session(world_id, session_id).intrinsics, width, height)
    except Exception:  # noqa: BLE001 -- nothing can be verified by reproduction
        return None
    return m1, m2, roi, width, height


def _solver_image_sha1(source: Path, undistortion) -> str | None:
    """The SHA-1 of the solver image `prepare_images` would write from `source`: the same
    read, size check, remap, crop and JPEG quality. None when it would write none."""
    import hashlib  # noqa: PLC0415

    import cv2  # noqa: PLC0415

    m1, m2, (x, y, rw, rh), width, height = undistortion
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None or image.shape[1] != width or image.shape[0] != height:
        return None
    undistorted = cv2.remap(image, m1, m2, cv2.INTER_LINEAR)[y:y + rh, x:x + rw]
    ok, encoded = cv2.imencode(".jpg", undistorted,
                               [cv2.IMWRITE_JPEG_QUALITY, SOLVER_JPEG_QUALITY])
    return hashlib.sha1(encoded.tobytes()).hexdigest() if ok else None


# REPRODUCTION FAILURES THAT NOTHING EXPLAINS (review V10, L-9). A walk image that does not
# reproduce from its planned frame is withheld, which is safe -- but when the reason is that
# THIS Tower's undistortion or JPEG encoder no longer gives the bytes the walk's gave (an
# OpenCV or libjpeg upgrade), every image of every walk is withheld, re-extracted and loses
# the frozen matching, and nothing said so. So a differing image is checked against the
# other frames it could have been made from (`plan_solver_images`' `alternatives`); one no
# frame reproduces is UNEXPLAINED, logged, and counted. Measured on the five frozen walks
# (RUN/experiments/P3-REF/probe/carry_back_frozen.out.txt): 2,501 of 2,502 walk images
# reproduce from their planned frame, and the one that does not reproduces from its stored
# copy -- no unexplained image at all. So when unexplained images outnumber reproduced ones,
# the pipeline, not the walk, changed: `encoder_drift_suspected`, and a louder warning.


def carry_back_decisions(store: WorldStore, world_id: str, session_id: str, keyframes,
                         image_plan: dict) -> dict:
    """Which of the walk's solver images may go back into the fresh solve directory
    (review V9, M-7). Reads only. Returns
    `{"carried": {name: provenance entry + "verified"}, "withheld": {name: why},
    "clear_features": [names], "reproduction": {...}, "unexplained": [names]}`.

    An image goes back ONLY when it is proven to be exactly what this re-finish plans for
    its keyframe -- the raw frame found by identity, or the stored redacted copy for a
    keyframe that falls back:
    - `record`: the set-aside solve's provenance record names the planned frame, and its
      SHA-1 is the file's; or
    - `reproduced`: undistorting the planned frame as `prepare_images` does gives the
      very same bytes; or
    - `record-frame-gone` (review V10, L-8): the plan falls back to the stored copy ONLY
      because the keyframe's `sources.json` frame is no longer on disk (a capture moved or
      purged), and the provenance record proves the image IS that frame -- `source` raw,
      that very frame, this keyframe, its SHA-1 the file's. It keeps its record entry
      (`source: raw`): the raw pixels are the better solver input, and nothing can
      re-derive them. A walk image with no record (a walk's own solves write none) is not
      proven by `sources.json` alone -- the frozen walk 6839fb8f holds an image its
      `sources.json` attributes to a raw frame that was undistorted from the stored copy --
      and is withheld as before.
    Every other image is WITHHELD, and says why: `no-keyframe` (no keyframe of the session
    has that name), `planned-frame-unreadable`, `not-reproducible` (the session has no
    undistortion to reproduce it with), `differs-from-plan`. The solve then writes it
    afresh from the planned frame. `clear_features` is every keyframe image NOT carried
    back -- withheld, or absent -- whose features the copied walk database must lose, and
    every image whose clear the walk's record still owes (`features_clear_owed`).
    `reproduction` counts what reproduction found, including the UNEXPLAINED differences
    (see above), which are logged."""
    from tower.world_builder.global_solve import workspace_for  # noqa: PLC0415
    from tower.world_builder.solve_masks import file_sha1  # noqa: PLC0415

    workspace = workspace_for(store, world_id, session_id)
    recorded = read_image_provenance(workspace.root)
    carried: dict = {}
    withheld: dict = {}
    differing: list = []
    undistortion = None
    undistortion_known = False
    if workspace.images_dir.is_dir():
        for entry in sorted(workspace.images_dir.iterdir()):
            if not entry.is_file():
                continue
            name = entry.name
            planned = image_plan.get(name)
            if planned is None:
                withheld[name] = "no-keyframe"
                continue
            sha1 = file_sha1(entry)
            provenance = {k: planned[k] for k in ("keyframe_id", "source", "frame")}
            old = recorded.get(name)
            if old is not None and old.get("sha1") == sha1 and _same_provenance(old, planned):
                carried[name] = dict(provenance, sha1=sha1, verified="record")
                continue
            if old is not None and _record_is_the_gone_frame(old, planned, sha1):
                carried[name] = {"keyframe_id": planned["keyframe_id"],
                                 "source": IMAGE_SOURCE_RAW, "frame": old.get("frame"),
                                 "sha1": sha1, "verified": "record-frame-gone"}
                continue
            if not undistortion_known:
                undistortion = _undistortion(store, world_id, session_id, keyframes)
                undistortion_known = True
            if undistortion is None:
                withheld[name] = "not-reproducible"
                continue
            reproduced = _solver_image_sha1(planned["path"], undistortion)
            if reproduced is None:
                withheld[name] = "planned-frame-unreadable"
            elif reproduced != sha1:
                withheld[name] = "differs-from-plan"
                differing.append((name, sha1, planned))
            else:
                carried[name] = dict(provenance, sha1=sha1, verified="reproduced")
    unexplained = sorted(name for name, sha1, planned in differing
                         if not any(_solver_image_sha1(alt, undistortion) == sha1
                                    for alt in planned.get("alternatives") or ()))
    reproduced_n = sum(1 for e in carried.values() if e.get("verified") == "reproduced")
    reproduction = {"reproduced": reproduced_n, "differs": len(differing),
                    "differs_explained": len(differing) - len(unexplained),
                    "differs_unexplained": len(unexplained),
                    "encoder_drift_suspected": len(unexplained) > reproduced_n}
    if unexplained:
        if reproduction["encoder_drift_suspected"]:
            logger.warning(
                "[Tower][WorldBuilder] %s/%s: %d of the walk's solver images cannot be "
                "reproduced from ANY frame they could have been made from, and only %d can: "
                "this Tower's undistortion or JPEG encoder (OpenCV, libjpeg) probably no longer "
                "gives the bytes the walk's did. Every such image is withheld and re-extracted, "
                "and the walk's frozen matching no longer applies to it (e.g. %s)", world_id,
                session_id, len(unexplained), reproduced_n, ", ".join(unexplained[:3]))
        else:
            logger.warning(
                "[Tower][WorldBuilder] %s/%s: %d walk solver image(s) match neither the planned "
                "frame nor any other frame they could have been made from, and are withheld "
                "(e.g. %s)", world_id, session_id, len(unexplained), ", ".join(unexplained[:3]))
    # CLEARS THE WALK STILL OWES (review V10, L-11; SOL's `features_clear_owed`): a solve that
    # rewrote an image and could not clear its old features records it in the provenance
    # record. The image itself may be carried back -- it is what it says -- but the copied
    # database still holds the OLD image's features for it, and the fresh record written here
    # does not carry the debt on, so they are cleared here too.
    from tower.world_builder.global_solve import features_clear_owed  # noqa: PLC0415

    owed = set(features_clear_owed(workspace))
    return {"carried": carried, "withheld": withheld,
            "clear_features": sorted({n for n in image_plan if n not in carried} | owed),
            "features_clear_owed": sorted(owed),
            "reproduction": reproduction, "unexplained": unexplained}


def _record_is_the_gone_frame(entry: dict, planned: dict, sha1: str) -> bool:
    """L-8: the provenance entry proves the image was undistorted from this keyframe's own
    `sources.json` frame, which the plan skips only because it is no longer on disk --
    `global_solve.recorded_raw_frame_gone`, the rule `prepare_images` keeps such an image by
    (one statement of it), for THIS keyframe's entry."""
    from tower.world_builder.global_solve import recorded_raw_frame_gone  # noqa: PLC0415

    gone = planned.get("raw_frame_gone")
    if gone is None or entry.get("keyframe_id") != planned.get("keyframe_id"):
        return False
    return recorded_raw_frame_gone(entry, planned.get("source"), gone, sha1)


def _carry_back_images(src: Path, dst: Path, carried: dict) -> dict:
    """Copy the carried-back images from `src` to `dst`, each verified by its SHA-1 as it
    is copied; an image that changed since it was judged is not copied, and leaves
    `carried`. Returns the ledger's account."""
    import hashlib  # noqa: PLC0415

    dst.mkdir(parents=True, exist_ok=True)
    changed = []
    for name in sorted(carried):
        try:
            data = (src / name).read_bytes()
        except OSError:
            data = None          # gone since it was judged: withheld like a changed one
        if data is None or hashlib.sha1(data).hexdigest() != carried[name]["sha1"]:
            changed.append(name)
            continue
        (dst / name).write_bytes(data)
        shutil.copystat(src / name, dst / name)
    for name in changed:
        carried.pop(name)
    present = sum(1 for p in src.iterdir() if p.is_file())
    return {"carried_back": len(carried), "withheld": present - len(carried),
            "images_changed_while_copying": changed}


def _write_provenance_record(solve_dir: Path, carried: dict) -> None:
    from tower.storage import write_json_atomic  # noqa: PLC0415

    write_json_atomic(solve_dir / SOLVER_IMAGES_PROVENANCE, {
        "record": PROVENANCE_RECORD,
        "images": {name: {k: entry[k] for k in ("keyframe_id", "source", "frame", "sha1",
                                                "verified") if k in entry}
                   for name, entry in sorted(carried.items())}})


def apply_solver_frames(store: WorldStore, world_id: str, session_id: str,
                        planned: dict) -> bool:
    """Write the planned `sources.json` into the (fresh) solve directory when it
    differs from the one there. True when written."""
    from tower.world_builder.global_solve import (  # noqa: PLC0415
        workspace_for,
        write_sources_records,
    )

    if not planned.get("sources_changed"):
        return False
    write_sources_records(workspace_for(store, world_id, session_id), planned["sources"])
    return True


def run_final_solve(root: Path, world_id: str, session_id: str, *, seed: int,
                    threads: int, runner=None, on_start=None) -> dict:
    """Step 2: `world_finalize.py` with the product solve settings, in a child. No
    `--capture-dir`, ever: its lookup is by frame name, which a walk's captures reuse
    (the raw frames go in `sources.json`, `plan_solver_frames`).

    `on_start(process)` is called as soon as the child exists -- the re-finish records its
    pid in the ledger there (review V10, Q3). `runner` (tests) replaces the child with a
    `subprocess.run`-shaped callable, and then there is no process to record."""
    env = dict(os.environ)
    env.update(PRODUCT_SOLVE_ENV)
    env["TOWER_WORLD_SOLVE_SEED"] = str(int(seed))
    argv = [sys.executable, str(FINALIZE_SCRIPT), "--root", str(root), "--world", world_id,
            "--session", session_id, "--threads", str(int(threads)), "--format", "json"]
    if runner is None:
        done = _run_recorded(argv, env, on_start)
    else:
        done = runner(argv, env=env, capture_output=True, text=True)
    try:
        report = json.loads(done.stdout) if done.stdout.strip() else {}
    except ValueError:
        report = {"stdout_tail": done.stdout[-2000:]}
    report["exit_code"] = done.returncode
    if done.returncode != 0:
        report["stderr_tail"] = (done.stderr or "")[-2000:]
    return report


def _run_recorded(argv, env, on_start):
    """`subprocess.run(argv, env=env, capture_output=True, text=True)`, with `on_start` given
    the child as soon as it exists. ANY exception -- `on_start`'s (the ledger could not
    record the child) or an interrupt -- kills the child and waits for it before it
    propagates, as `subprocess.run` does: the put-back that follows is then never under a
    live child (review V10, Q3)."""
    with subprocess.Popen(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True) as process:
        try:
            if on_start is not None:
                on_start(process)
            stdout, stderr = process.communicate()
        except BaseException:
            process.kill()
            process.wait()
            raise
    return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)


def _child_record(process) -> dict:
    """What the ledger says about the final solve's child: pid and start time, as the writer
    lock records a process (`refinish_liveness`). A child that is already gone when this is
    taken has no start time to read; it is recorded as ended."""
    from tower.world_builder.store import _lock_record  # noqa: PLC0415

    record = dict(_lock_record(process.pid), script="scripts/world_finalize.py",
                  started_at=time.time())
    if "created_at" not in record:
        code = process.poll()
        if code is not None:
            record["exit_code"] = code
    return record


def _put_back(store: WorldStore, engine, world_id: str, session_id: str, ledger: dict,
              report: dict, *, stamp: str, what: str, state: str) -> None:
    """Restore the set-aside result under the world's lock and say so in `report`
    (and, through `restore_after_failed_solve`, in the ledger)."""
    try:
        store.acquire_writer_lock(world_id)
    except Exception as exc:  # noqa: BLE001
        report["reason"] = (f"{what}, and the previous result could not be put back "
                            f"({type(exc).__name__}: {exc}); it is under "
                            f"refinish/{stamp}/")
        ledger["state"] = LEDGER_RESTORE_INCOMPLETE
        ledger["restore_errors"] = [f"the world's lock: {type(exc).__name__}: {exc}"]
        try:
            _write_ledger(store.world_dir(world_id) / REFINISH_DIRNAME / stamp, ledger)
        except OSError:
            pass
        return
    try:
        restored = restore_after_failed_solve(store, world_id, session_id, ledger,
                                              state=state)
        report["restored"] = restored
        if restored["errors"]:
            report["reason"] = (f"{what}, and putting the previous result back stopped "
                                f"part-way ({'; '.join(restored['errors'])}); see "
                                f"refinish/{stamp}/{LEDGER_FILENAME}")
        else:
            report["reason"] = f"{what}; the previous result was put back (see restored)"
    except Exception as exc:  # noqa: BLE001 -- the ledger could not even be written
        report["reason"] = (f"{what}, and putting the previous result back stopped "
                            f"part-way ({type(exc).__name__}: {exc}); see "
                            f"refinish/{stamp}/{LEDGER_FILENAME}")
    finally:
        engine.release_world(world_id)


def refinish(store: WorldStore, root: Path, world_id: str, session_id: str, *,
             seed: int = 0, threads: int = -1, appearance: bool = True,
             prune_depth_work: bool = True, should_stop=lambda: False,
             stop_source=lambda: None, solve_runner=None, stamp: str | None = None,
             capture_dirs=(), use_capture: bool = True) -> dict:
    """Steps 1-4. Raises `Refused` before anything is written when it must refuse:
    no such world or session, purged imagery, a session that never stopped, a live
    writer, a build of this session running without the lock (V6 H1), or solver frames
    that cannot be planned. Raises `SetAsideFailed` (a `Refused`) when step 1 could not
    complete; the world is then as it was (V9 M-6). A final solve that publishes nothing
    puts the previous result back (L3), and so does ANY exception between the set-aside
    and a published solve (the ledger then says `restored-after-an-error` and names the
    error). After the publish the ledger always ends in a terminal state: `done`, or
    `stopped` with why -- a stop, or a room or area stage that raised (V9 LOW).

    `capture_dirs` / `use_capture`: where the solver's raw frames are
    (`resolve_capture_dirs`); the ledger's `solver_frames` records what was used."""
    from scripts.world_build_session import final_surface_stages  # noqa: PLC0415
    from scripts.world_finish_pending import (  # noqa: PLC0415
        _recorder,
        build_session_areas,
    )
    from tower.world_builder.components import read_components_record  # noqa: PLC0415
    from tower.world_builder.engine import WorldBuilderEngine  # noqa: PLC0415

    try:
        world = store.read_world(world_id)
    except (WorldStoreError, OSError, ValueError, KeyError):
        raise Refused(f"no world {world_id!r}") from None
    if session_id not in store.list_session_ids(world_id):
        raise Refused(f"world {world_id!r} has no session {session_id!r}")
    if getattr(world, "images_purged", False):
        raise Refused("this world's keyframe imagery was purged; it cannot be rebuilt")
    session = store.read_session(world_id, session_id)
    if session.ended_at is None:
        raise Refused("this session never stopped; scripts/world_finalize.py first")

    # H1: a build of this session running without the lock (the builder's post-Stop
    # surface, an area build) is not something to move a solve out from under.
    busy = build_in_progress(store, world_id, session_id)
    if busy is not None:
        raise Refused(busy)

    stamp = stamp or time.strftime("%Y%m%d-%H%M%S")
    report: dict = {"world_id": world_id, "session_id": session_id}
    engine = WorldBuilderEngine(store)
    try:
        store.acquire_writer_lock(world_id)
    except Exception as exc:  # noqa: BLE001 -- a live writer owns the world
        raise Refused(f"{type(exc).__name__}: {exc}") from None
    try:
        # Asked again under the lock: a finisher cannot start now, but the builder's
        # post-Stop stages need no lock.
        busy = build_in_progress(store, world_id, session_id)
        if busy is not None:
            raise Refused(busy)
        # An earlier re-finish of this session that died before it published (review V9,
        # M-5) is put back first, exactly as the idle Tower's finisher would: what it set
        # aside IS the previous result, and setting aside its half-built replacement
        # instead would bury it one level deeper.
        dead = dead_before_publish(store, world_id, session_id)
        if dead is not None:
            try:
                report["earlier_refinish"] = recover_dead_refinish(store, world_id,
                                                                   dead["stamp"])
            except Exception as exc:  # noqa: BLE001
                raise Refused(f"an earlier re-finish ({dead['stamp']}) of this session "
                              f"ended before it published and could not be put back "
                              f"({type(exc).__name__}: {exc})") from exc
            if report["earlier_refinish"].get("state") == LEDGER_RESTORE_INCOMPLETE:
                raise Refused(f"an earlier re-finish ({dead['stamp']}) of this session ended "
                              "before it published, and what it set aside could not all be "
                              f"put back; see {REFINISH_DIRNAME}/{dead['stamp']}/"
                              f"{LEDGER_FILENAME}")
        # THE SOLVER'S FRAMES ARE PLANNED BEFORE ANYTHING IS SET ASIDE (review V9, M-7):
        # which of the walk's solver images may be carried back depends on the plan.
        # Reads only, so a failure here is a refusal with nothing written.
        try:
            capture = resolve_capture_dirs(store, world_id, session_id, capture_dirs,
                                           use_capture=use_capture)
            planned = plan_solver_frames(store, world_id, session_id,
                                         capture["capture_dirs"])
        except Exception as exc:  # noqa: BLE001
            raise Refused(f"the solver's frames could not be planned "
                          f"({type(exc).__name__}: {exc}); nothing was set aside") from exc
        report["set_aside"] = ledger = set_aside(store, world_id, session_id, stamp,
                                                 solver_images=planned)
    except BaseException:
        engine.release_world(world_id)
        raise

    # FROM HERE UNTIL A SOLVE IS PUBLISHED, ANY EXCEPTION PUTS THE PREVIOUS RESULT BACK
    # (P3, PV attempt 1: `mark_stage` raised WinError 5 on a read-only session.json
    # right after the set-aside, and the world was left with its solve moved aside and
    # a session saying `complete`). `step` names where it happened, for the ledger.
    step = "marking the room's stages"
    try:
        try:
            # M3: the room's stages are now built from a solve that is set aside. Say
            # so, under the same lock, so a re-finish that stops before step 3 leaves
            # a room the finisher rebuilds (`stopped` is owed) rather than one that
            # claims `ok`.
            detail = REFINISH_IN_PROGRESS.format(stamp=stamp)
            for stage in ("surface", "appearance"):
                engine.mark_stage(world_id, session_id, stage, state="stopped",
                                  detail=detail)
            # The solver's raw frames, each found by its own capture identity and
            # written as `sources.json` into the fresh solve directory -- under the
            # same lock. Never handed to the solve as a capture directory.
            step = "writing the solver's frames"
            capture["sources_json_written"] = apply_solver_frames(
                store, world_id, session_id, planned)
        finally:
            engine.release_world(world_id)
        capture.update({k: v for k, v in planned.items() if k not in PLAN_INTERNAL_KEYS})
        report["solver_frames"] = ledger["solver_frames"] = capture
        _write_ledger(store.world_dir(world_id) / REFINISH_DIRNAME / stamp, ledger)

        # Step 2 takes the lock itself, in its own process. The child is in the ledger
        # before it can take the lock (review V10, Q3): if THIS process is killed while the
        # child runs on, the finisher sees a live re-finish and puts nothing back under it.
        # A ledger that cannot record it stops the child before anything is put back.
        step = "running the final solve"

        def record_child(process):
            ledger["child"] = _child_record(process)
            _write_ledger(store.world_dir(world_id) / REFINISH_DIRNAME / stamp, ledger)

        report["final_solve"] = run_final_solve(
            root, world_id, session_id, seed=seed, threads=threads, on_start=record_child,
            **({"runner": solve_runner} if solve_runner is not None else {}))
        if isinstance(ledger.get("child"), dict):
            ledger["child"].update(exit_code=report["final_solve"].get("exit_code"),
                                   ended_at=time.time())
        step = "reading the published solution"
        from tower.world_builder.global_solve import load_solution  # noqa: PLC0415

        published = (report["final_solve"].get("exit_code") == 0
                     and load_solution(store, world_id, session_id) is not None)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        report["done"] = False
        report["error"] = f"{step}: {error}"
        ledger["error_after_set_aside"] = {"step": step, "error": error,
                                           "at": time.time()}
        _put_back(store, engine, world_id, session_id, ledger, report, stamp=stamp,
                  what=f"{step} failed after the set-aside ({error})",
                  state=LEDGER_RESTORED_AFTER_ERROR)
        if not isinstance(exc, Exception):
            raise      # an interrupt or an exit, after the world was put back
        return report
    if not published:
        # L3: no new solve was published. Put the old one back, so the session is
        # again what is on disk; nothing is deleted.
        report["done"] = False
        _put_back(store, engine, world_id, session_id, ledger, report, stamp=stamp,
                  what="the final solve did not publish a solution",
                  state=LEDGER_RESTORED)
        return report
    aside_dir = store.world_dir(world_id) / REFINISH_DIRNAME / stamp

    def ledger_state(state: str, detail: str | None = None) -> None:
        ledger["state"] = state
        ledger[f"{state}_at"] = time.time()
        if detail:
            ledger["detail"] = detail
        try:
            _write_ledger(aside_dir, ledger)
        except OSError as exc:      # the ledger is a record; the rebuild goes on
            report["ledger_error"] = f"{type(exc).__name__}: {exc}"

    ledger_state(LEDGER_PUBLISHED)
    # The finisher's counters for this session describe the build that was just
    # replaced: they are kept in the ledger and restarted, so a rebuild interrupted later
    # is not retired on the old build's account. NOW, after the publish, and never
    # fatal (review V9, M-6): the build is not replaced until then -- a put-back leaves
    # them as they were -- and a counter file that cannot be written (read-only, RV9-D
    # probe A) costs the restart, not the re-finish.
    try:
        ledger["previous"]["finish_attempts"] = _restart_attempts(store, world_id,
                                                                 session_id, stamp)
        ledger["finish_attempts_restarted_at"] = time.time()
    except Exception as exc:  # noqa: BLE001 -- recorded; the rebuild goes on
        report["finish_attempts_error"] = ledger["finish_attempts_error"] = (
            f"the finisher's attempt counters could not be restarted: "
            f"{type(exc).__name__}: {exc}")
    unwritten = _write_ledger_quietly(aside_dir, ledger)
    if unwritten:
        report["ledger_error"] = unwritten
    record = read_components_record(store, world_id, session_id)
    report["components"] = (None if record is None else
                            [{"id": e["id"], "shown_as": e["shown_as"],
                              "keyframes": e["keyframes"], "reasons": e["reasons"]}
                             for e in record.entries])
    if record is None:
        report["components_note"] = (
            "the evidence gate wrote no components record: it is not wired into this "
            "Tower's final solve, or it failed (see final_solve); the session reports "
            "components: null")

    # Steps 3 and 4, under the lock, the stop checked between them. `main` has put
    # PRODUCT_SOLVE_ENV into this process's environment, so the room's surface asks for
    # the depth the gate already ran (`final_surface_stages`) and reuses it.
    try:
        store.acquire_writer_lock(world_id)
    except Exception as exc:  # noqa: BLE001
        report.update({"done": False, "reason": f"{type(exc).__name__}: {exc}"})
        ledger_state(LEDGER_STOPPED, f"the room was not rebuilt: {report['reason']}")
        return report
    # ANY EXCEPTION IN STEPS 3 AND 4 ENDS THE LEDGER TRUTHFULLY (review V9, LOW): it used
    # to stay at `published` -- a state that says the rebuild is still going on -- over a
    # room whose stage had raised. Now `stopped`, naming the error; the new solve stays
    # published, and the room's own record says what became of it.
    try:
        try:
            step = "rebuilding the room"
            report["room"] = final_surface_stages(
                store, world_id, session_id, solved=True, appearance=appearance,
                prune_depth_work=prune_depth_work, should_stop=should_stop,
                stop_source=stop_source, record=_recorder(engine, world_id, session_id))
            if record is not None and not should_stop():
                step = "building the areas"
                report["areas"] = build_session_areas(
                    store, world_id, session_id, appearance=appearance,
                    prune_depth_work=prune_depth_work, should_stop=should_stop,
                    stop_source=stop_source, build=True,
                    area_ids=[e["id"] for e in record.areas()])
        finally:
            engine.release_world(world_id)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        report.update({"done": False, "error": f"{step}: {error}"})
        ledger_state(LEDGER_STOPPED, f"{step} failed after the new solve was published "
                                     f"({error}); the new solve stands")
        if not isinstance(exc, Exception):
            raise
        return report
    report["done"] = not should_stop()
    ledger_state(LEDGER_DONE if report["done"] else LEDGER_STOPPED,
                 None if report["done"] else "a stop was asked for")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog=("Sets the previous solve and builds aside under "
                "<world>/refinish/<stamp>/ (moved or copied, never deleted), then runs "
                "the masked, seeded, gated final solve, the room and its areas. "
                "WORLD-BUILDER-COMPONENTS.md section 7, rule 4."),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=artifact_root_arg, required=True,
                        help="the world root (a Tower's data/world_builder)")
    parser.add_argument("--world", required=True)
    parser.add_argument("--session", help="defaults to the world's most recent session")
    parser.add_argument("--seed", type=int, default=0,
                        help="TOWER_WORLD_SOLVE_SEED for the seeded single-thread final "
                             "solve (default 0)")
    parser.add_argument("--threads", type=int, default=-1,
                        help="passed to world_finalize.py")
    parser.add_argument("--appearance", action=argparse.BooleanOptionalAction,
                        default=True, help="build the appearance of the room and areas")
    parser.add_argument("--keep-depth-work", action="store_true",
                        help="do not prune the per-frame depth work afterwards")
    capture = parser.add_mutually_exclusive_group()
    capture.add_argument("--capture-dir", action="append", default=[],
                         help="a capture directory (with its frames.jsonl) to search for "
                              "each keyframe's own raw frame, by source_seq and receipt "
                              "time (repeatable). Default: the session's capture and "
                              "those continuing it, under TOWER_CAPTURE_ROOT. A keyframe "
                              "with no unambiguous frame keeps its redacted copy")
    capture.add_argument("--no-capture", action="store_true",
                         help="search no capture: the walk's own sources.json, else the "
                              "redacted session keyframes")
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would be set aside and run; write nothing")
    parser.add_argument("--format", choices=("json", "text"), default="json")
    args = parser.parse_args(argv)

    root = Path(args.root)
    store = WorldStore(root)
    session_id = args.session or _latest_session(store, args.world)
    if session_id is None:
        _emit({"done": False, "refused": "this world has no sessions"}, args.format)
        return EXIT_REFUSED
    if args.dry_run:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        frames = resolve_capture_dirs(store, args.world, session_id, args.capture_dir,
                                      use_capture=not args.no_capture)
        planned = plan_solver_frames(store, args.world, session_id, frames["capture_dirs"])
        frames.update({k: v for k, v in planned.items() if k not in PLAN_INTERNAL_KEYS})
        _emit({"dry_run": True, "world_id": args.world, "session_id": session_id,
               "plan": plan(store, args.world, session_id, stamp),
               "final_solve_env": {**PRODUCT_SOLVE_ENV,
                                   "TOWER_WORLD_SOLVE_SEED": str(args.seed)},
               "solver_frames": frames},
              args.format)
        return EXIT_OK

    from scripts.world_build_session import StopRequest  # noqa: PLC0415
    from tower.native_prewarm import prewarm_world_builder  # noqa: PLC0415

    # This process runs the room's and the areas' stages itself: with the product
    # settings, like the child that solves.
    os.environ.update(PRODUCT_SOLVE_ENV)
    prewarm_world_builder()
    stop = StopRequest()
    stop.install()
    try:
        report = refinish(store, root, args.world, session_id, seed=args.seed,
                          threads=args.threads, appearance=args.appearance,
                          prune_depth_work=not args.keep_depth_work,
                          should_stop=stop.asked_for, stop_source=lambda: stop.source,
                          capture_dirs=args.capture_dir, use_capture=not args.no_capture)
    except Refused as exc:
        _emit({"done": False, "refused": str(exc)}, args.format)
        return EXIT_REFUSED
    _emit(report, args.format)
    return EXIT_OK if report.get("done") else EXIT_FAILED


def _emit(report: dict, fmt: str) -> None:
    if fmt == "json":
        print(json.dumps(report, indent=2, default=str))
        return
    for key, value in report.items():
        print(f"{key:16s} {value}")


if __name__ == "__main__":
    raise SystemExit(main())
