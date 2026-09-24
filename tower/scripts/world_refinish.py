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
stored, redacted keyframes; refused for a world whose imagery was purged):

1.  SETS THE PREVIOUS RESULT ASIDE, AND DELETES NOTHING (contract T11). Under the
    world's writer lock:
    - `solve/<session>` is MOVED to `<world>/refinish/<stamp>/solve/<session>`, and
      then the walk's own feature database (`database.db`, with any SQLite
      `-wal`/`-shm` beside it), the solver images it was extracted from (`images/`),
      `sources.json` and `camera.json` are COPIED BACK into a fresh
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

Exit status: 0 done; 1 a step failed (the report says which; what was set aside stays
set aside and is named); 2 refused (no such world or session, imagery purged, a live
writer holds the world).
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
# solver images its keypoints came from, where the raw frames were, and the camera
# the images were undistorted with. See `set_aside`.
SOLVE_COPY_BACK = ("database.db", "database.db-wal", "database.db-shm", "images",
                   "sources.json", "camera.json")

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


def set_aside(store: WorldStore, world_id: str, session_id: str, stamp: str) -> dict:
    """Step 1, under the caller's lock. Never deletes; returns the ledger it wrote."""
    from tower.storage import write_json_atomic  # noqa: PLC0415

    p = plan(store, world_id, session_id, stamp)
    aside = Path(p["aside"])
    aside.mkdir(parents=True, exist_ok=False)
    session = store.read_session(world_id, session_id)
    for c in p["copies"]:
        src, dst = Path(c["from"]), Path(c["to"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    for m in p["moves"]:
        src, dst = Path(m["from"]), Path(m["to"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Same volume (the world's own directory): a rename, instant and atomic.
        os.replace(src, dst)
    # THE WALK'S OWN DATABASE GOES BACK, AS A COPY (RV1 M1-1). The masked final
    # solve maps a filtered copy of the walk database when one is there
    # (`global_solve`, `walk-database-filtered`, arm A1h) and re-extracts under the
    # masks when it is not (`re-extracted`, arm A1) -- which splits the target room.
    # So the fresh solve directory gets a COPY of the database and of the solver
    # images its keypoints were extracted from (so the masks are computed on the
    # same pixels), plus where the raw frames were (`sources.json`) and the camera
    # the images were undistorted with (`camera.json`, without which
    # `prepare_images` would re-undistort them). The set-aside originals are never
    # touched again: the rebuild extracts and matches into the copy.
    solve_moved = next((m for m in p["moves"] if m["kind"] == "solve"), None)
    copied_back = []
    if solve_moved is not None:
        old_root = Path(solve_moved["to"])
        fresh = store.world_dir(world_id) / "solve" / session_id
        fresh.mkdir(parents=True, exist_ok=True)
        for name in SOLVE_COPY_BACK:
            old = old_root / name
            if old.is_dir():
                shutil.copytree(old, fresh / name)
            elif old.is_file():
                shutil.copy2(old, fresh / name)
            else:
                continue
            copied_back.append({"kind": "solve", "name": name, "from": str(old),
                                "to": str(fresh / name)})
    # The finisher's attempt counters for this session describe the build being set
    # aside. They are kept in the ledger below and restarted, so a rebuild interrupted
    # later is not retired on the old build's account.
    prior_attempts = _restart_attempts(store, world_id, session_id, stamp)
    ledger = {
        "command": "scripts/world_refinish.py",
        "world_id": world_id, "session_id": session_id, "set_aside_at": time.time(),
        "moved": p["moves"], "copied": p["copies"], "copied_back": copied_back,
        "previous": {"finalization": session.finalization, "stages": session.stages,
                     "finish_attempts": prior_attempts},
        "restore": "move the rebuild's own solve/<session> (and areas) aside, then move "
                   "each `moved` entry back from `to` to `from`; the `copied` entries "
                   "are snapshots of what the rebuild replaced in place, and the "
                   "`copied_back` entries are the rebuild's COPIES of the set-aside "
                   "walk database and images (the originals were never modified)",
        "deletion": "requires a human's approval (Glasses filesystem policy rule 14)",
    }
    write_json_atomic(aside / LEDGER_FILENAME, ledger)
    return ledger


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
    """Steps 1-4. Raises `Refused` before anything is written when it must refuse."""
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

    stamp = stamp or time.strftime("%Y%m%d-%H%M%S")
    report: dict = {"world_id": world_id, "session_id": session_id}
    engine = WorldBuilderEngine(store)
    try:
        store.acquire_writer_lock(world_id)
    except Exception as exc:  # noqa: BLE001 -- a live writer owns the world
        raise Refused(f"{type(exc).__name__}: {exc}") from None
    try:
        report["set_aside"] = set_aside(store, world_id, session_id, stamp)
    finally:
        engine.release_world(world_id)

    # Step 2 takes the lock itself, in its own process.
    report["final_solve"] = run_final_solve(
        root, world_id, session_id, seed=seed, threads=threads,
        **({"runner": solve_runner} if solve_runner is not None else {}))
    if report["final_solve"].get("exit_code") != 0:
        report["done"] = False
        report["reason"] = "the final solve did not finish; see final_solve"
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
