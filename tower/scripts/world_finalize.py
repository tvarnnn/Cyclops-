#!/usr/bin/env python
"""Finish a session whose finalization did not finish.

    python scripts/world_finalize.py --root <world root> --world <id>
        [--session <id>] [--capture-dir <dir> ...] [--skip-solve]
        [--threads N] [--format json|text]

WHY THIS EXISTS, AND WHAT IT COST NOT TO HAVE IT.

`mark_finalization` had exactly one caller in the whole system: the
builder's own `finally`, in `world_build_session.py`. So a session whose
builder died between the last keyframe and the finished world could not be
repaired by any supported operation. Re-running `world_build_session.py`
does not repair it -- that opens a NEW session against the same world --
and `world_solve.py` writes a solution but never merges it, because the
engine's `build()` is the only writer of the derived tree.

The 2026-09-09 walk landed exactly there: 795 keyframes, 643 positioned
poses and 26,634 points on disk, `finalization: interrupted`, detail
`BadZipFile: File is not a zip file`, and nothing in the product able to
turn that into the world it already almost was. Recovering it took three
hand-written steps in a scratch directory. This is those three steps, as
an operation the system supports.

WHAT IT DOES

1.  Takes the world's writer lock, so it cannot race a live session. A
    lock naming a dead process is reclaimed; a lock naming a live one is
    an error, not something to force.
2.  Runs the final global solve -- every keyframe, loop detection on, all
    cores -- unless `--skip-solve`. This is usually the whole point: the
    final solve is what merges the fragments. On the 2026-09-09 capture it
    took 16 components to 6 and put 652 of 795 keyframes into one at
    0.82 px, where the last live solve had 156 in its largest.
3.  Rebuilds the derived tree, which merges that solution into the poses,
    points and placements the phone reads.
4.  Records the finalization the builder never got to record.

IDEMPOTENT. Every step rewrites rather than appends: the solve replaces
`solution.*`, `build()` replaces the whole derived tree, and
`mark_finalization` rewrites one block of the session record. Running it
twice produces the same world as running it once, and running it on an
already-complete session is a supported no-op that still re-derives.

IT NEVER INVENTS DATA. It reads the session's own journals and images and
nothing else. A session whose images were purged cannot be rebuilt and
says so; a solve that finds no model is a recorded outcome, not a crash,
and the derived tree is still rebuilt from whatever the local chain has.

Exit status is 0 when the world was finalized, 1 when it could not be,
and 2 for a usage error.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tower.artifact_paths import artifact_root_arg  # noqa: E402
from tower.world_builder import global_solve  # noqa: E402
from tower.world_builder.engine import WorldBuilderEngine  # noqa: E402
from tower.world_builder.records import (  # noqa: E402
    FINAL_SOLVE_FAILED,
    FINAL_SOLVE_SKIPPED,
    FINAL_SOLVE_SOLVED,
    FINAL_SOLVE_UNAVAILABLE,
    FINALIZATION_COMPLETE,
    FINALIZATION_INTERRUPTED,
)
from tower.world_builder.store import WorldStore, compute_input_digest  # noqa: E402


def latest_session(store: WorldStore, world_id: str) -> str | None:
    """The session most worth finishing: the newest one.

    Deliberately not "the newest UNFINISHED one". A caller who names no
    session means the walk they just took, and silently skipping it
    because its record already says `complete` would hide the case this
    tool is most often pointed at -- a finalization that completed while
    its final solve was skipped.
    """
    session_ids = store.list_session_ids(world_id)
    if not session_ids:
        return None
    sessions = [store.read_session(world_id, sid) for sid in session_ids]
    return max(sessions, key=lambda s: (s.started_at or 0.0, s.session_id)).session_id


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=artifact_root_arg, required=True)
    parser.add_argument("--world", required=True)
    parser.add_argument(
        "--session",
        help="defaults to the world's most recent session",
    )
    parser.add_argument(
        "--capture-dir", action="append", default=[],
        help="where the raw capture frames live; they beat the redacted "
             "session copies as solver input, measured 337 images against 307",
    )
    parser.add_argument(
        "--skip-solve", action="store_true",
        help="rebuild the derived tree from the solution already on disk, "
             "without re-solving",
    )
    parser.add_argument("--threads", type=int, default=-1)
    parser.add_argument("--format", choices=("json", "text"), default="json")
    args = parser.parse_args(argv)

    store = WorldStore(Path(args.root))
    session_id = args.session or latest_session(store, args.world)
    if session_id is None:
        print(json.dumps({"finalized": False, "reason": "this world has no sessions"}))
        return 1

    report: dict = {"world_id": args.world, "session_id": session_id}
    before = store.read_session(args.world, session_id)
    report["was"] = dict(before.finalization or {}) or None

    engine = WorldBuilderEngine(store)
    # The lock is the whole safety story. `acquire_writer_lock` reclaims a
    # lock whose holder is gone (pid AND process start time, so a recycled
    # pid cannot masquerade) and refuses one whose holder is alive. A live
    # builder owns this world; this tool waits for another day rather than
    # writing underneath it.
    try:
        store.acquire_writer_lock(args.world)
    except Exception as exc:  # noqa: BLE001 -- reported, not raised
        report.update({"finalized": False, "reason": f"{type(exc).__name__}: {exc}"})
        _emit(report, args.format)
        return 1

    final_solve_state: str | None = None
    detail: str | None = None
    try:
        if args.skip_solve:
            final_solve_state = FINAL_SOLVE_SKIPPED
            detail = "final solve skipped: --skip-solve"
            report["global_solve"] = {"attempted": False, "reason": "--skip-solve"}
        else:
            keyframes = store.read_keyframes(args.world, session_id)
            summary = global_solve.solve(
                store, args.world, session_id,
                capture_dirs=[Path(d) for d in args.capture_dir],
                final=True,
                num_threads=args.threads,
                loop_detection=True,
                input_digest=compute_input_digest(keyframes),
            )
            report["global_solve"] = summary
            if summary.get("solved"):
                final_solve_state = FINAL_SOLVE_SOLVED
            elif summary.get("error"):
                final_solve_state = FINAL_SOLVE_FAILED
                detail = f"final solve failed: {summary['error']}"
            else:
                final_solve_state = FINAL_SOLVE_UNAVAILABLE
                detail = f"final solve produced no solution: {summary.get('reason')}"

        # The build is the step that matters even when the solve did not
        # land: it re-derives poses, points and placements from the journal,
        # merging whatever solution IS on disk. A world whose final solve
        # failed is still worth rebuilding from its last background one.
        result = engine.build(args.world, session_id)
        report["build"] = {
            "keyframes": result.keyframes,
            "poses_solved": result.poses_solved,
            "poses_refused": result.poses_refused,
            "points": result.points,
            "segments": result.segments,
            "scale_state": result.scale_state,
            "placements_source": (result.diagnostics or {}).get("placements_source"),
        }
        # THE FALLBACK THE BUILDER HAS AND THIS DID NOT. When no global
        # solve placed anything, the Sim3 registrar is the only producer of
        # placements there is, and `world_build_session.py` runs it for
        # exactly that case. Without this, a repair of a session whose solve
        # found nothing rebuilt the derived tree and left the world with no
        # placements at all -- every fragment its own island, which is the
        # outcome this whole campaign is about. Found by an adversarial
        # review of the repair tool.
        #
        # `register_session` never raises and refuses to replace placements
        # that are registered and current, so calling it is safe whichever
        # way the solve went; `should_register` decides.
        from scripts.world_build_session import (  # noqa: PLC0415
            register_session,
            should_register,
        )

        if should_register(result):
            report["registration"] = register_session(store, args.world, session_id)
        else:
            report["registration"] = {
                "attempted": False,
                "reason": "the global solve placed these segments",
            }
        state = FINALIZATION_COMPLETE
        report["finalized"] = True
    except Exception as exc:  # noqa: BLE001
        # A REPAIR THAT FAILS MUST NOT LEAVE THE WORLD WORSE THAN IT FOUND
        # IT. This wrote `interrupted` unconditionally, so pointing the tool
        # at an already-complete world and hitting any error -- a purged
        # world, a full disk, a raising solve -- DOWNGRADED a healthy record
        # to interrupted, with no way back: every re-run hits the same
        # error. An adversarial review demonstrated it on a purged world,
        # `complete` -> `interrupted`, permanently.
        #
        # The record only moves if this run actually had something to
        # improve on. The authoritative journals are untouched either way,
        # so a failure is still retryable once its cause is fixed.
        detail = f"{type(exc).__name__}: {exc}"
        was_complete = (before.finalization or {}).get("state") == FINALIZATION_COMPLETE
        state = FINALIZATION_COMPLETE if was_complete else FINALIZATION_INTERRUPTED
        if was_complete:
            # Keep the record exactly as it was, including its final_solve
            # and its detail: this run has nothing truer to say about it.
            final_solve_state = (before.finalization or {}).get("final_solve")
            detail_to_record = (before.finalization or {}).get("detail")
        else:
            detail_to_record = detail
        detail = detail_to_record
        report.update({
            "finalized": False,
            "reason": f"{type(exc).__name__}: {exc}",
            "record_left_as": state,
        })
    finally:
        try:
            engine.mark_finalization(
                args.world, session_id,
                state=state, final_solve=final_solve_state, detail=detail,
            )
        except Exception as exc:  # noqa: BLE001
            report["record_error"] = f"{type(exc).__name__}: {exc}"
        engine.release_world(args.world)

    after = store.read_session(args.world, session_id)
    report["now"] = dict(after.finalization or {}) or None
    _emit(report, args.format)
    return 0 if report.get("finalized") else 1


def _emit(report: dict, fmt: str) -> None:
    if fmt == "json":
        print(json.dumps(report, indent=2))
        return
    for key, value in report.items():
        print(f"{key:18s} {value}")


if __name__ == "__main__":
    raise SystemExit(main())
