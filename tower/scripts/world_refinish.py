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
      and on the target walk that splits the room (reviewer RV1, finding M1-1);
    - `surface/`, `appearance/` and `dense/<session>`, `derived/` and the session
      record are COPIED there, because the rebuild replaces them in place;
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
areas, the derived tree and the session record (what the failed solve left is kept under
`refinish/<stamp>/failed-*`). So does ANY exception between the set-aside and a
published solve -- a session record that cannot be written, a child that cannot be
started -- and the ledger then says `restored-after-an-error` and names the step and
the error (`restore-incomplete`, with what could not be done, if it could not all go
back).

Exit status: 0 done; 1 a step failed (the report says which); 2 refused (no such world
or session, imagery purged, a live writer holds the world, a build is running, or the
set-aside could not complete and was rolled back).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tower.artifact_paths import artifact_root_arg  # noqa: E402
from tower.world_builder.store import WorldStore, WorldStoreError  # noqa: E402

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
SOLVE_COPY_BACK = ("database.db", "database.db-wal", "database.db-shm", "images",
                   "sources.json", "camera.json", "transients", "database.matching.json")

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
            (moves if how == "move" else copies).append(
                {"kind": kind, "from": str(src), "to": str(aside / kind / session_id)})
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
    return what they were."""
    from scripts import world_finish_pending as wfp  # noqa: PLC0415

    # The room's, the areas' AND the re-gate's counters (FIN, review V8): a re-finished
    # world must not inherit an exhausted re-gate count from the build it replaces.
    keys = (session_id, wfp.area_ledger_key(session_id), wfp.regate_ledger_key(session_id))
    with wfp._LEDGER_LOCK:
        sessions, unreadable = wfp._read_ledger(store, world_id)
        if unreadable:
            return {"unreadable": True}
        prior = {k: sessions[k] for k in keys if k in sessions}
        if prior:
            sessions = dict(sessions)
            for key in prior:
                sessions[key] = {"attempts": 0, "forgiven": 0,
                                 "detail": f"restarted by scripts/world_refinish.py ({stamp})"}
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
LEDGER_TERMINAL_STATES = (LEDGER_ROLLED_BACK, LEDGER_ROLLBACK_INCOMPLETE, LEDGER_RESTORED,
                          LEDGER_RESTORED_AFTER_ERROR, LEDGER_RESTORE_INCOMPLETE, LEDGER_DONE,
                          LEDGER_STOPPED)


def refinish_process_alive(ledger: dict) -> bool | None:
    """Whether the process that wrote this ledger is still running: True, False, or None
    for a ledger written before the pid was recorded. Pid AND start time, the writer
    lock's own test (`store._holder_is_running`), so a recycled pid is not taken for the
    re-finish that died."""
    from tower.world_builder.store import _holder_is_running  # noqa: PLC0415

    process = (ledger or {}).get("process") or {}
    pid = process.get("pid")
    if not isinstance(pid, int):
        return None
    return bool(_holder_is_running(pid, process.get("created_at")))


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
    _write_ledger(aside, ledger)
    return ok


def set_aside(store: WorldStore, world_id: str, session_id: str, stamp: str) -> dict:
    """Step 1, under the caller's lock. Never deletes; returns the ledger it wrote.

    IN AN ORDER THAT CANNOT STRAND THE WORLD (review V6, M1):

    1. the LEDGER first, naming everything that is about to move, so nothing can move
       without a record of where it went;
    2. the snapshot COPIES (surface, appearance, dense, derived, the session record),
       and the COPY-BACK of the walk's solve inputs into a staging directory beside
       the ledger -- both copies, both before anything moves, so the copy-back does
       not depend on any move having happened;
    3. the MOVES, the session's areas before its solve, each retried against a file
       another process holds open, and every completed move undone if a later one
       fails;
    4. the staged copy put in place as the fresh `solve/<session>`.

    A failure at any point leaves the world as it was and raises `SetAsideFailed`;
    what was copied stays under `refinish/<stamp>/`, named by the ledger.
    """
    p = plan(store, world_id, session_id, stamp)
    aside = Path(p["aside"])
    aside.mkdir(parents=True, exist_ok=False)
    session = store.read_session(world_id, session_id)
    moves = [dict(m) for m in p["moves"]]
    # Areas before the solve: an open area file is the likeliest refusal, and failing
    # there leaves nothing of the solve to put back.
    moves.sort(key=lambda m: 0 if m["kind"] == "areas" else 1)
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
    _write_ledger(aside, ledger)

    def fail(state: str, what: str, exc: BaseException) -> "SetAsideFailed":
        ledger["state"] = state
        ledger["error"] = f"{what}: {type(exc).__name__}: {exc}"
        _write_ledger(aside, ledger)
        return SetAsideFailed(
            f"the previous result could not be set aside ({ledger['error']}); "
            + ("nothing was moved" if state == LEDGER_ROLLED_BACK else
               "SOME MOVES COULD NOT BE UNDONE -- see the ledger")
            + f": {aside / LEDGER_FILENAME}")

    # 2. copies: snapshots, then the copy-back staged beside the ledger.
    solve_live = store.world_dir(world_id) / "solve" / session_id
    try:
        for c in p["copies"]:
            src, dst = Path(c["from"]), Path(c["to"])
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        # THE WALK'S OWN DATABASE GOES BACK, AS A COPY (RV1 M1-1). The masked final
        # solve maps a filtered copy of the walk database when one is there
        # (`global_solve`, `walk-database-filtered`, arm A1h) and re-extracts under
        # the masks when it is not (`re-extracted`, arm A1) -- which splits the
        # target room. So the fresh solve directory gets a COPY of the database and
        # of the solver images its keypoints were extracted from (so the masks are
        # computed on the same pixels), where the raw frames were (`sources.json`),
        # the camera the images were undistorted with (`camera.json`, without which
        # `prepare_images` would re-undistort them) and the content-keyed mask cache.
        # The set-aside originals are never touched again.
        if solve_live.exists():
            staging.mkdir()
            for name in SOLVE_COPY_BACK:
                old = solve_live / name
                if old.is_dir():
                    shutil.copytree(old, staging / name)
                elif old.is_file():
                    shutil.copy2(old, staging / name)
                else:
                    continue
                ledger["copied_back"].append({
                    "kind": "solve", "name": name,
                    "from": str(aside / "solve" / session_id / name),
                    "to": str(solve_live / name)})
    except OSError as exc:
        raise fail(LEDGER_ROLLED_BACK, "copying", exc) from exc

    # 3. moves, undone on failure.
    done = []
    for m in moves:
        src, dst = Path(m["from"]), Path(m["to"])
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            _replace_with_retry(src, dst)
        except OSError as exc:
            all_back = _move_back(done, ledger, aside)
            raise fail(LEDGER_ROLLED_BACK if all_back else LEDGER_ROLLBACK_INCOMPLETE,
                       f"moving {m['kind']} {src.name}", exc) from exc
        m["moved"] = True
        done.append(m)
        _write_ledger(aside, ledger)

    # 4. the staged copy becomes the fresh solve directory.
    if staging.exists():
        try:
            _replace_with_retry(staging, solve_live)
        except OSError as exc:
            all_back = _move_back(done, ledger, aside)
            raise fail(LEDGER_ROLLED_BACK if all_back else LEDGER_ROLLBACK_INCOMPLETE,
                       "placing the fresh solve directory", exc) from exc

    # The finisher's attempt counters for this session describe the build being set
    # aside. They are kept in the ledger and restarted, so a rebuild interrupted
    # later is not retired on the old build's account.
    ledger["previous"]["finish_attempts"] = _restart_attempts(store, world_id, session_id,
                                                             stamp)
    ledger["state"] = LEDGER_SET_ASIDE
    _write_ledger(aside, ledger)
    return ledger


def restore_after_failed_solve(store: WorldStore, world_id: str, session_id: str,
                               ledger: dict, *, state: str = LEDGER_RESTORED) -> dict:
    """Put the set-aside result back after a final solve that did not publish (review
    V6, L3) or after any exception between the set-aside and a published solve, under
    the caller's lock. Nothing is deleted: what the failed solve left is moved under
    `refinish/<stamp>/failed-solve/`, the derived tree it rebuilt under
    `failed-derived/`. The session record is written back from its snapshot, so the
    session says what is on disk again.

    EVERY STEP IS ATTEMPTED, whatever an earlier one did: the moves that put the old
    solve and areas back must not be skipped because, say, the session record cannot
    be written (the very file whose write failed in PV's attempt 1). What could not be
    done is listed in `errors`, and the ledger then says `restore-incomplete` rather
    than `state`. A session record that already equals its snapshot byte for byte is
    left alone -- nothing moved it, so there is nothing to write back."""
    from tower.storage import write_json_atomic  # noqa: PLC0415

    world_dir = store.world_dir(world_id)
    aside = world_dir / REFINISH_DIRNAME / ledger["stamp"]
    out = {"restored": [], "kept": [], "errors": []}

    def attempt(what: str, step) -> bool:
        try:
            step()
            return True
        except Exception as exc:  # noqa: BLE001 -- recorded; the next step still runs
            out["errors"].append(f"{what}: {type(exc).__name__}: {exc}")
            return False

    failed = world_dir / "solve" / session_id
    solve_cleared = True
    if failed.exists():
        dst = aside / "failed-solve" / session_id

        def keep_failed_solve():
            dst.parent.mkdir(parents=True, exist_ok=True)
            _replace_with_retry(failed, dst)
            out["kept"].append(str(dst))

        solve_cleared = attempt("keeping the new solve directory aside", keep_failed_solve)
    for m in ledger["moved"]:
        if not m.get("moved") or m.get("moved_back"):
            continue
        if m["kind"] == "solve" and not solve_cleared:
            out["errors"].append(f"moving {m['kind']} back: skipped, {m['from']} is "
                                 "still occupied by the new solve directory")
            continue

        def move_back(m=m):
            _replace_with_retry(Path(m["to"]), Path(m["from"]))
            m["moved_back"] = True
            out["restored"].append(m["from"])

        attempt(f"moving {m['kind']} {Path(m['from']).name} back", move_back)
    snapshot = aside / "derived"
    if snapshot.is_dir():
        live = world_dir / "derived"

        def restore_derived():
            if live.exists():
                dst = aside / "failed-derived"
                _replace_with_retry(live, dst)
                out["kept"].append(str(dst))
            shutil.copytree(snapshot, live)
            out["restored"].append(str(live))

        attempt("restoring the derived tree", restore_derived)
    session_copy = aside / "session.json"
    if session_copy.is_file():
        live_session = store.session_path(world_id, session_id)

        def restore_session():
            if live_session.is_file() and live_session.read_bytes() == session_copy.read_bytes():
                out["unchanged"] = str(live_session)
                return
            write_json_atomic(live_session,
                              json.loads(session_copy.read_text(encoding="utf-8")))
            out["restored"].append(str(live_session))

        attempt("restoring the session record", restore_session)
    ledger["state"] = LEDGER_RESTORE_INCOMPLETE if out["errors"] else state
    ledger["restore_report"] = out
    _write_ledger(aside, ledger)
    return out


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
        return {"capture_dirs": [str(Path(d)) for d in explicit],
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
                and (h["wire_seq"] is None or kf.wire_seq is None
                     or int(h["wire_seq"]) == int(kf.wire_seq))
                and (h["time_basis"] is None or h["time_basis"] == kf.time_basis)]
        if len(hits) == 1:
            mapping[kf.keyframe_id] = str(hits[0]["path"])
        elif hits:
            why["several_frames"].append(kf.keyframe_id)
        else:
            why["no_frame"].append(kf.keyframe_id)
    why["notes"] = notes
    return mapping, why


SOURCE_WALK_IMAGES = "walk-solver-images"
SOURCE_RAW = "raw-capture"
SOURCE_REDACTED = "redacted-session-keyframes"
SOURCE_MIXED = "mixed"


def plan_solver_frames(store: WorldStore, world_id: str, session_id: str,
                       capture_dirs) -> dict:
    """Where each keyframe's solver image will come from, and the `sources.json` that
    makes it so. Reads only: `sources` in the result is what `apply_solver_frames`
    writes, `sources_changed` whether it differs from the record on disk.

    Per keyframe record, in `global_solve.prepare_images`' own order: already
    undistorted in the solve directory (the walk's solver images, reused); the raw
    frame the builder recorded (`raw_from_sources_json`); the raw frame found by
    identity in the captures (`raw_from_capture_dir`); else the stored redacted
    keyframe (`redacted_session_copies`, split by why)."""
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
        if (workspace.images_dir / keyframe_image_name(kf)).exists():
            counts["already_undistorted"] += 1
        elif builder_ok:
            counts["raw_from_sources_json"] += 1
        elif kid in mapping:
            counts["raw_from_capture_dir"] += 1
        else:
            counts["redacted_session_copies"] += 1
            counts["redacted_ambiguous_in_session" if kid in ambiguous else
                   "redacted_several_capture_frames" if kid in several else
                   "redacted_no_capture_frame"] += 1
    raw = counts["raw_from_sources_json"] + counts["raw_from_capture_dir"]
    redacted = counts["redacted_session_copies"]
    counts["source"] = (None if counts["keyframes"] == 0 else
                        SOURCE_WALK_IMAGES if raw + redacted == 0 else
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
    return counts


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
                    threads: int, runner=subprocess.run) -> dict:
    """Step 2: `world_finalize.py` with the product solve settings, in a child. No
    `--capture-dir`, ever: its lookup is by frame name, which a walk's captures reuse
    (the raw frames go in `sources.json`, `plan_solver_frames`)."""
    env = dict(os.environ)
    env.update(PRODUCT_SOLVE_ENV)
    env["TOWER_WORLD_SOLVE_SEED"] = str(int(seed))
    argv = [sys.executable, str(FINALIZE_SCRIPT), "--root", str(root), "--world", world_id,
            "--session", session_id, "--threads", str(int(threads)), "--format", "json"]
    done = runner(argv, env=env, capture_output=True, text=True)
    try:
        report = json.loads(done.stdout) if done.stdout.strip() else {}
    except ValueError:
        report = {"stdout_tail": done.stdout[-2000:]}
    report["exit_code"] = done.returncode
    if done.returncode != 0:
        report["stderr_tail"] = (done.stderr or "")[-2000:]
    return report


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
    writer, or a build of this session running without the lock (V6 H1). Raises
    `SetAsideFailed` (a `Refused`) when step 1 could not complete; the world is then as
    it was. A final solve that publishes nothing puts the previous result back (L3), and
    so does ANY exception between the set-aside and a published solve (the ledger then
    says `restored-after-an-error` and names the error).

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
        report["set_aside"] = ledger = set_aside(store, world_id, session_id, stamp)
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
            step = "resolving the solver's frames"
            capture = resolve_capture_dirs(store, world_id, session_id, capture_dirs,
                                           use_capture=use_capture)
            planned = plan_solver_frames(store, world_id, session_id,
                                         capture["capture_dirs"])
            capture["sources_json_written"] = apply_solver_frames(
                store, world_id, session_id, planned)
        finally:
            engine.release_world(world_id)
        capture.update({k: v for k, v in planned.items() if k != "sources"})
        report["solver_frames"] = ledger["solver_frames"] = capture
        _write_ledger(store.world_dir(world_id) / REFINISH_DIRNAME / stamp, ledger)

        # Step 2 takes the lock itself, in its own process.
        step = "running the final solve"
        report["final_solve"] = run_final_solve(
            root, world_id, session_id, seed=seed, threads=threads,
            **({"runner": solve_runner} if solve_runner is not None else {}))
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
    try:
        report["room"] = final_surface_stages(
            store, world_id, session_id, solved=True, appearance=appearance,
            prune_depth_work=prune_depth_work, should_stop=should_stop,
            stop_source=stop_source, record=_recorder(engine, world_id, session_id))
        if record is not None and not should_stop():
            report["areas"] = build_session_areas(
                store, world_id, session_id, appearance=appearance,
                prune_depth_work=prune_depth_work, should_stop=should_stop,
                stop_source=stop_source, build=True,
                area_ids=[e["id"] for e in record.areas()])
    finally:
        engine.release_world(world_id)
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
        frames.update({k: v for k, v in planned.items() if k != "sources"})
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
