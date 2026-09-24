#!/usr/bin/env python
"""Rebuild one saved session with the product pipeline: components, the room and its areas.

    .venv\\Scripts\\python.exe scripts/world_refinish.py --root <world root> --world <id>
        [--session <id>] [--seed 0] [--threads -1] [--no-appearance]
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
      `sources.json`, `camera.json` and the solver's content-keyed transient-mask
      cache (`transients/`) are COPIED BACK into a fresh
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
    and records the finalization.
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
`refinish/<stamp>/failed-*`).

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
# solve its GPU minutes). NOT `masks/` (the COLMAP masks are rewritten from that
# cache on every masked solve), nor `database.masked.*` / `reverify_pairs.txt`
# (made by a solve, for that solve). See `set_aside`.
SOLVE_COPY_BACK = ("database.db", "database.db-wal", "database.db-shm", "images",
                   "sources.json", "camera.json", "transients")

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

    keys = (session_id, wfp.area_ledger_key(session_id))
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
    ledger = {
        "command": "scripts/world_refinish.py",
        "state": LEDGER_SETTING_ASIDE,
        "world_id": world_id, "session_id": session_id, "stamp": stamp,
        "set_aside_at": time.time(),
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
                               ledger: dict) -> dict:
    """Put the set-aside result back after a final solve that did not publish (review
    V6, L3), under the caller's lock. Nothing is deleted: what the failed solve left
    is moved under `refinish/<stamp>/failed-solve/`, the derived tree it rebuilt under
    `failed-derived/`. The session record is written back from its snapshot, so the
    session says what is on disk again."""
    from tower.storage import write_json_atomic  # noqa: PLC0415

    world_dir = store.world_dir(world_id)
    aside = world_dir / REFINISH_DIRNAME / ledger["stamp"]
    out = {"restored": [], "kept": []}
    failed = world_dir / "solve" / session_id
    if failed.exists():
        dst = aside / "failed-solve" / session_id
        dst.parent.mkdir(parents=True, exist_ok=True)
        _replace_with_retry(failed, dst)
        out["kept"].append(str(dst))
    for m in ledger["moved"]:
        if m.get("moved") and not m.get("moved_back"):
            _replace_with_retry(Path(m["to"]), Path(m["from"]))
            m["moved_back"] = True
            out["restored"].append(m["from"])
    snapshot = aside / "derived"
    if snapshot.is_dir():
        live = world_dir / "derived"
        if live.exists():
            dst = aside / "failed-derived"
            _replace_with_retry(live, dst)
            out["kept"].append(str(dst))
        shutil.copytree(snapshot, live)
        out["restored"].append(str(live))
    session_copy = aside / "session.json"
    if session_copy.is_file():
        write_json_atomic(store.session_path(world_id, session_id),
                          json.loads(session_copy.read_text(encoding="utf-8")))
        out["restored"].append(str(store.session_path(world_id, session_id)))
    ledger["state"] = LEDGER_RESTORED
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


def run_final_solve(root: Path, world_id: str, session_id: str, *, seed: int,
                    threads: int, runner=subprocess.run) -> dict:
    """Step 2: `world_finalize.py` with the product solve settings, in a child."""
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


def refinish(store: WorldStore, root: Path, world_id: str, session_id: str, *,
             seed: int = 0, threads: int = -1, appearance: bool = True,
             prune_depth_work: bool = True, should_stop=lambda: False,
             stop_source=lambda: None, solve_runner=None, stamp: str | None = None) -> dict:
    """Steps 1-4. Raises `Refused` before anything is written when it must refuse:
    no such world or session, purged imagery, a session that never stopped, a live
    writer, or a build of this session running without the lock (V6 H1). Raises
    `SetAsideFailed` (a `Refused`) when step 1 could not complete; the world is then as
    it was. A final solve that publishes nothing puts the previous result back (L3)."""
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
        # M3: the room's stages are now built from a solve that is set aside. Say so,
        # under the same lock, so a re-finish that stops before step 3 leaves a room
        # the finisher rebuilds (`stopped` is owed) rather than one that claims `ok`.
        detail = REFINISH_IN_PROGRESS.format(stamp=stamp)
        for stage in ("surface", "appearance"):
            engine.mark_stage(world_id, session_id, stage, state="stopped", detail=detail)
    finally:
        engine.release_world(world_id)

    # Step 2 takes the lock itself, in its own process.
    report["final_solve"] = run_final_solve(
        root, world_id, session_id, seed=seed, threads=threads,
        **({"runner": solve_runner} if solve_runner is not None else {}))
    from tower.world_builder.global_solve import load_solution  # noqa: PLC0415

    if (report["final_solve"].get("exit_code") != 0
            or load_solution(store, world_id, session_id) is None):
        # L3: no new solve was published. Put the old one back, so the session is
        # again what is on disk; nothing is deleted.
        report["done"] = False
        report["reason"] = ("the final solve did not publish a solution; the previous "
                            "result was put back (see restored)")
        try:
            store.acquire_writer_lock(world_id)
        except Exception as exc:  # noqa: BLE001
            report["reason"] = (f"the final solve did not publish a solution, and the "
                                f"previous result could not be put back "
                                f"({type(exc).__name__}: {exc}); it is under "
                                f"refinish/{stamp}/")
            return report
        try:
            report["restored"] = restore_after_failed_solve(store, world_id, session_id,
                                                            ledger)
        except OSError as exc:
            report["reason"] = (f"the final solve did not publish a solution, and putting "
                                f"the previous result back stopped part-way "
                                f"({type(exc).__name__}: {exc}); see refinish/{stamp}/"
                                f"{LEDGER_FILENAME}")
        finally:
            engine.release_world(world_id)
        return report
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
        _emit({"dry_run": True, "world_id": args.world, "session_id": session_id,
               "plan": plan(store, args.world, session_id, stamp),
               "final_solve_env": {**PRODUCT_SOLVE_ENV,
                                   "TOWER_WORLD_SOLVE_SEED": str(args.seed)}},
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
                          should_stop=stop.asked_for, stop_source=lambda: stop.source)
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
