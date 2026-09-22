#!/usr/bin/env python
"""Finish the photographic work a world was promised and never got.

    python scripts/world_finish_pending.py --root <world root>
        [--max-worlds 1] [--max-attempts 3] [--no-appearance]
        [--keep-depth-work] [--dry-run] [--stop-on-stdin-close]
        [--format json|text]

WHY THIS EXISTS, AND WHAT IT COST NOT TO HAVE IT.

The surface and the appearance -- the stages that turn a scatter of feature
points into the room the wearer walked through -- run exactly ONCE, in the
builder child, in the six to sixteen minutes after Stop and after the world
writer lock is released. That ordering is deliberate: the lock is dropped so
the phone can read the world while they build.

What was missing is what happens when that child does not survive those
sixteen minutes. Nothing retried it. Not `scripts/world_finalize.py`, which
rebuilds the SPARSE derived tree and never runs a surface. Not the serving
path, which is strictly read-only -- `tower/results/world_builder*.py`
performs no write and spawns no process, and never generates on demand. Not
the Tower at startup, which had no reconciliation of any kind. A Tower
shutdown, a machine sleep, a crash, or the supervisor's thirty-second stop
grace discarded the work permanently, and the world stayed sparse forever
beside a session record proudly saying `finalization: complete`.

It happened for real on 2026-09-22: the owner shut the Tower down eight
minutes into a build, and the next start recovered nothing.

WHAT IT DOES

1.  Reads every session record under the root and decides which ones OWE
    photographic work (see `assess`). Reading only; nothing is taken and
    nothing is written for a session that is not owed.
2.  Takes the world's writer lock for each owed session it finishes, with
    `world_finalize.py`'s discipline unchanged: a lock naming a dead process
    is reclaimed, a lock naming a LIVE one is an error, not something to
    force.
3.  Runs `world_build_session.final_surface_stages` -- the builder's own
    stage path, by import rather than by copy, so the parameters, the order,
    the stop behaviour and the record writes cannot drift from the builder's.
4.  Releases the lock, and reports.

IDEMPOTENT AND BOUNDED. A session that has been finished is no longer owed,
so running this twice finishes the work once. It finishes at most
`--max-worlds` worlds per run (one, by default) because the next thing the
wearer does may need the GPU. And it counts its own attempts on disk: this
run marks the surface `running` before it starts, exactly as the builder
does, so a finisher that is itself killed leaves behind the very signature it
selects on. Without the bound that is an infinite loop across restarts on a
world it can never finish.

IT NEVER INVENTS WORK. A session with NO stage record at all is NOT owed.
Absent is not a state in the vocabulary; it means "a Tower that never
recorded this", which is every world built before 2026-09-22 -- 165 of them
on the machine this was written on. Reading absence as "interrupted" would
queue tens of hours of GPU against worlds that are finished and fine.

Exit status is 0 whenever the run itself was sound -- including a run that
found nothing owed, which is the common case -- and 1 when a session that was
owed could not be finished.
"""

import argparse
import collections
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.world_build_session import (  # noqa: E402
    StopRequest,
    final_surface_stages,
)
from tower.artifact_paths import artifact_root_arg  # noqa: E402
from tower.results.world_builder_render import session_build_running  # noqa: E402
from tower.storage import read_json_closed, write_json_atomic  # noqa: E402
from tower.world_builder.engine import WorldBuilderEngine  # noqa: E402
from tower.world_builder.records import (  # noqa: E402
    FINAL_SOLVE_SOLVED,
    FINALIZATION_COMPLETE,
    STAGE_APPEARANCE,
    STAGE_STATE_FAILED,
    STAGE_STATE_RUNNING,
    STAGE_STATE_STOPPED,
    STAGE_SURFACE,
)
from tower.world_builder.store import WorldStore, WorldStoreError  # noqa: E402

logger = logging.getLogger(__name__)

# The two stages this tool owes. The dense stage is deliberately NOT here:
# it is off by default, it is the most expensive thing this Tower can be
# asked to do, and it is not what makes a saved world recognisable. The
# photographic representation is the surface and the shading on it.
PHOTOGRAPHIC_STAGES = (STAGE_SURFACE, STAGE_APPEARANCE)

# The states that mean "this stage was interrupted", as opposed to the three
# that are terminal. `ok` is done. `failed` will fail the same way again --
# a depth network that is not installed does not become installed by being
# asked twice -- and `unavailable` means nobody asked for a photographic
# world at all, which is a configuration and not a fault.
INTERRUPTED_STATES = (STAGE_STATE_RUNNING, STAGE_STATE_STOPPED)

# How many times this tool may start the same session before it gives up and
# says so on the record. Three, because the failure it bounds is "something
# keeps killing me" -- a machine that sleeps on a timer, a Tower an operator
# keeps restarting -- and the second attempt is genuinely often the one that
# lands.
DEFAULT_MAX_ATTEMPTS = 3

# One world per run. Not a throughput knob: the machine belongs to whoever
# picks the glasses up next, and six to sixteen minutes is already a long
# time to be holding a GPU on nobody's behalf. Two owed worlds are two Tower
# starts.
DEFAULT_MAX_WORLDS = 1

# Where the attempt counter lives. Beside the world rather than on the
# session record, because it is THIS TOOL's bookkeeping and not a fact about
# the session: `Session.stages` is a record of what the builder did, and
# adding a field to it for a retry counter would make every reader of a
# published record carry this tool's private state.
ATTEMPTS_FILENAME = "finish_attempts.json"


@dataclass(frozen=True)
class Verdict:
    """Whether one session owes photographic work, and why or why not.

    A REASON ON EVERY ANSWER, including the negative ones, because the
    negative ones are the interesting half. This tool's whole risk is doing
    expensive work nobody asked for, so "why did you skip that world" has to
    be answerable from the run's own report rather than by reading this file.
    """

    world_id: str
    session_id: str
    owed: bool
    reason: str
    # A stable, short name for the reason, so a run over a root with seventy
    # sessions can report "37 not-finalized, 29 never-stopped" instead of
    # seventy sentences. The sentence is still there for the ones that matter.
    code: str = ""
    stage: str | None = None
    attempts: int = 0
    # True when this session WAS owed and has used up its attempts. Distinct
    # from a plain `owed: False`: it is the one negative answer that still
    # calls for a write, because it has something to say about a world that
    # will otherwise stay sparse with nothing on disk explaining it.
    exhausted: bool = False

    def as_dict(self) -> dict:
        return {
            "world_id": self.world_id,
            "session_id": self.session_id,
            "owed": self.owed,
            "code": self.code,
            "reason": self.reason,
            "stage": self.stage,
            "attempts": self.attempts,
            "exhausted": self.exhausted,
        }


# -- the attempt ledger ------------------------------------------------


def _attempts_path(store: WorldStore, world_id: str) -> Path:
    return store.world_dir(world_id) / ATTEMPTS_FILENAME


def _read_ledger(store: WorldStore, world_id: str) -> tuple[dict, bool]:
    """This world's attempt entries, and whether the ledger was unreadable.

    The two are reported separately because they must not be confused. An
    unreadable ledger read as "no attempts yet" is the direction that loops
    forever; read as "stop", it costs one world a rebuild a human can ask
    for. So it is neither -- it is its own answer, and `assess` refuses on it.
    """
    path = _attempts_path(store, world_id)
    if not path.exists():
        return {}, False
    try:
        data = read_json_closed(path)
    except (OSError, ValueError):
        logger.warning(
            "[Tower][WorldBuilder] the finish ledger for %s is unreadable",
            world_id,
        )
        return {}, True
    if not isinstance(data, dict) or not isinstance(data.get("sessions"), dict):
        return {}, True
    return data["sessions"], False


def read_attempts(store: WorldStore, world_id: str, session_id: str) -> int | None:
    """How often this tool has started this session, or None if it cannot say."""
    sessions, unreadable = _read_ledger(store, world_id)
    if unreadable:
        return None
    count = (sessions.get(session_id) or {}).get("attempts")
    return count if isinstance(count, int) and not isinstance(count, bool) else 0


def record_attempt(
    store: WorldStore, world_id: str, session_id: str, *, detail: str
) -> int:
    """Count one attempt, BEFORE the work it counts.

    Before, deliberately and for the same reason `mark_stage` writes
    `running` before a stage starts: the failure being bounded here is a
    process that does not live to write anything afterwards. An attempt
    counted on completion counts only the attempts that did not need
    counting.
    """
    sessions, _ = _read_ledger(store, world_id)
    sessions = dict(sessions)
    count = (sessions.get(session_id) or {}).get("attempts")
    count = (count if isinstance(count, int) and not isinstance(count, bool) else 0) + 1
    sessions[session_id] = {"attempts": count, "detail": detail}
    path = _attempts_path(store, world_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, {"schema": 1, "sessions": sessions})
    return count


# -- the predicate -----------------------------------------------------


def assess(
    store: WorldStore,
    world_id: str,
    session_id: str,
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> Verdict:
    """Whether this session owes a photographic representation.

    CONSERVATIVE BY CONSTRUCTION. Every clause below is a reason to do
    nothing, and the positive answer is what is left when none of them fire.
    That direction is the point: the cost of a false negative is a world that
    stays sparse until someone runs a script, and the cost of a false
    positive is six to sixteen minutes of somebody's GPU, times however many
    worlds the mistake matches.
    """

    def no(code: str, reason: str, **kwargs) -> Verdict:
        return Verdict(world_id, session_id, False, reason, code=code, **kwargs)

    try:
        session = store.read_session(world_id, session_id)
    except (WorldStoreError, OSError, ValueError, KeyError) as exc:
        return no(
            "unreadable-session",
            f"the session record is unreadable: {type(exc).__name__}: {exc}",
        )

    if session.ended_at is None:
        # A walk in progress, or a builder that died mid-walk. Either way the
        # photographic stages are not what it is missing, and
        # `world_finalize.py` is the tool for the second shape.
        return no("never-stopped", "this session never stopped")

    finalization = session.finalization or {}
    if finalization.get("state") != FINALIZATION_COMPLETE:
        return no(
            "not-finalized",
            f"finalization is {finalization.get('state')!r}, not "
            f"{FINALIZATION_COMPLETE!r}; there is no finished world to build from"
        )
    if finalization.get("final_solve") != FINAL_SOLVE_SOLVED:
        # `final_surface_stages` refuses without a solve and records
        # `unavailable`. Queuing it would take a lock to write that down again.
        return no(
            "no-final-solve",
            f"the final solve is {finalization.get('final_solve')!r}; a surface "
            "needs a global solve and there is none"
        )

    # THE HISTORICAL-WORLD GUARD, and the most consequential line in the file.
    #
    # `stages` absent is not a state. It means "a Tower that never recorded
    # this", which is every session written before 2026-09-22 -- 165 worlds on
    # the machine this was written for. An empty object says exactly as
    # little. Reading either as "interrupted" would put tens of hours of GPU
    # into a queue nobody asked for, against worlds that are finished.
    if not session.stages:
        return no(
            "no-stage-record",
            "this session has no stage record, which means a Tower that never "
            "recorded one -- not an interrupted stage"
        )

    stage = None
    for candidate in PHOTOGRAPHIC_STAGES:
        entry = session.stages.get(candidate) or {}
        if entry.get("state") in INTERRUPTED_STATES:
            stage = candidate
            break
    if stage is None:
        return no("nothing-interrupted",
                  "no photographic stage is in an interrupted state")

    # "RUNNING" IS TWO DIFFERENT FACTS AND ONLY LIVENESS TELLS THEM APART.
    #
    # Asked of `session_build_running` rather than of a pid check written
    # here. That is the same probe `_lifecycle` uses to report
    # `build_in_progress`, and it already answers both halves of this
    # question: the world's writer lock held by a live builder writing THIS
    # session, and a stage `status.json` saying `running` under a live pid
    # (treating a status whose process is gone as stale). Two answers to one
    # question is how they come to disagree.
    if session_build_running(store, world_id, session_id):
        return no("building-now",
                  f"the {stage} stage is genuinely running right now")

    # And the lock on its own, which the probe above deliberately does not
    # cover: it reports a live lock only while the session it names is still
    # open. A builder walking this world into a NEW session holds the same
    # world's lock, and writing underneath it is exactly what
    # `world_finalize.py` calls "the whole safety story".
    holder = store.lock_holder(world_id)
    if holder is not None and (holder["alive"] or holder["unreadable"]):
        return no(
            "locked",
            "this world's writer lock names "
            + (
                f"live pid {holder['pid']}"
                if holder["alive"]
                else "no readable process, so it is not safe to reclaim here"
            )
        )

    attempts = read_attempts(store, world_id, session_id)
    if attempts is None:
        return no(
            "ledger-unreadable",
            "this tool's attempt ledger for this world is unreadable, so it "
            "cannot tell a first attempt from a hundredth"
        )
    if attempts >= max_attempts:
        return no(
            "attempt-bound",
            f"this tool has already started this session {attempts} times "
            f"(the bound is {max_attempts}) and something ended it every time",
            stage=stage,
            attempts=attempts,
            exhausted=True,
        )

    return Verdict(
        world_id,
        session_id,
        True,
        f"the {stage} stage is {session.stages[stage].get('state')!r} and "
        "nothing is building it",
        code="owed",
        stage=stage,
        attempts=attempts,
    )


def survey(store: WorldStore, *, max_attempts: int = DEFAULT_MAX_ATTEMPTS) -> list:
    """Every session under the root, assessed. Reads only."""
    verdicts = []
    for world_id in store.list_world_ids():
        for session_id in store.list_session_ids(world_id):
            verdicts.append(assess(store, world_id, session_id, max_attempts=max_attempts))
    return verdicts


# -- doing the work ----------------------------------------------------


def _retire(engine: WorldBuilderEngine, verdict: Verdict, max_attempts: int) -> dict:
    """Write down that this tool has given up on a session, once.

    "Bounded" on its own would leave a world sparse with nothing on disk
    saying why, which is the exact failure the stage record was added to fix
    -- it would just move it one level up. `failed` is the honest word and it
    is terminal, so this also stops the session being reassessed at every
    boot forever.
    """
    detail = (
        f"scripts/world_finish_pending.py started this stage {verdict.attempts} "
        f"times (bound {max_attempts}) and was interrupted every time; not "
        "retrying. Run scripts/world_surface.py --force by hand to try again."
    )
    engine.mark_stage(
        verdict.world_id, verdict.session_id, verdict.stage,
        state=STAGE_STATE_FAILED, detail=detail,
    )
    return {"world_id": verdict.world_id, "session_id": verdict.session_id,
            "retired": verdict.stage, "attempts": verdict.attempts}


def finish(
    store: WorldStore,
    verdict: Verdict,
    *,
    appearance: bool,
    prune_depth_work: bool,
    should_stop,
    stop_source=lambda: None,
) -> dict:
    """One owed session, finished under the world's writer lock.

    THE BUILDER'S OWN PATH, BY IMPORT. `final_surface_stages` is what
    `world_build_session.py` calls after Stop, with the same final presets,
    the same ordering, the same stop behaviour and the same `mark_stage`
    writes. Copying any of that here would produce two photographic pipelines
    that agree on the day this was written.

    It re-runs the surface even when only the appearance was interrupted, and
    that is not laziness. `surfacify(force=True)` is how the per-frame depth
    work the appearance reads gets regenerated; the builder prunes that work
    after a successful pair, so an appearance-only rebuild has, in the common
    case, nothing left to read.
    """
    report: dict = {
        "world_id": verdict.world_id,
        "session_id": verdict.session_id,
        "stage": verdict.stage,
    }
    engine = WorldBuilderEngine(store)
    try:
        store.acquire_writer_lock(verdict.world_id)
    except Exception as exc:  # noqa: BLE001 -- reported, not raised
        # Including the lock we checked for in `assess` and lost in the
        # microseconds since. A live writer owns this world; this tool waits
        # for another day.
        report.update({"finished": False, "reason": f"{type(exc).__name__}: {exc}"})
        return report

    try:
        report["attempts"] = record_attempt(
            store, verdict.world_id, verdict.session_id,
            detail=f"finishing the {verdict.stage} stage",
        )
        report["stages"] = final_surface_stages(
            store, verdict.world_id, verdict.session_id,
            # Guaranteed by `assess`: a session is not owed unless its
            # finalization recorded a SOLVED final solve.
            solved=True,
            appearance=appearance,
            prune_depth_work=prune_depth_work,
            should_stop=should_stop,
            stop_source=stop_source,
            record=_recorder(engine, verdict.world_id, verdict.session_id),
        )
        report["finished"] = True
    except Exception as exc:  # noqa: BLE001 -- recorded, then reported
        # `final_surface_stages` has already recorded the stage as `failed`
        # and re-raised; there is nothing truer to write here. What must not
        # happen is the lock surviving this frame, which is why the release
        # is in the `finally` below and not after it.
        #
        # `Exception`, NOT `BaseException`. A `KeyboardInterrupt` or a
        # `SystemExit` caught here would be a Ctrl-C this tool swallowed and
        # then reported as a failed world. Both still run the `finally`, so
        # the lock is released on the way past either way.
        report.update({"finished": False, "reason": f"{type(exc).__name__}: {exc}"})
    finally:
        engine.release_world(verdict.world_id)
    return report


def _recorder(engine: WorldBuilderEngine, world_id: str, session_id: str):
    """`engine.mark_stage`, as `final_surface_stages` expects it, and never
    fatal. The builder wraps it identically and for the same reason: by the
    time these stages run the artifact itself is already on disk, and a
    record that cannot be written must not take it down."""

    def record(stage, *, state, detail=None, attempted=True):
        try:
            engine.mark_stage(
                world_id, session_id, stage,
                state=state, detail=detail, attempted=attempted,
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "[Tower][WorldBuilder] could not record the %s stage", stage
            )

    return record


# -- the CLI -----------------------------------------------------------


def main(argv=None, *, should_stop=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=artifact_root_arg, required=True)
    parser.add_argument(
        "--max-worlds", type=int, default=DEFAULT_MAX_WORLDS,
        help="how many owed worlds to finish in this run; the default is one, "
             "because the machine belongs to whoever picks the glasses up next",
    )
    parser.add_argument(
        "--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS,
        help="how many times this tool may start the same session before it "
             "gives up and records that it did",
    )
    parser.add_argument(
        "--appearance", action=argparse.BooleanOptionalAction, default=True,
        help="shade the surface with the wearer's redacted keyframes, as the "
             "builder does; pass --no-appearance when the Tower's "
             "TOWER_WORLD_APPEARANCE is off",
    )
    parser.add_argument(
        "--keep-depth-work", action="store_true",
        help="do not prune the per-frame depth work afterwards, so a later "
             "scripts/world_densify.py can reuse it",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="report what is owed and write nothing at all",
    )
    parser.add_argument(
        "--stop-on-stdin-close", action="store_true",
        help="the parent is holding this process's stdin; its close is a stop "
             "request. The other half of the agreement in tower/main.py",
    )
    parser.add_argument("--format", choices=("json", "text"), default="json")
    args = parser.parse_args(argv)

    stop_request = StopRequest()
    if should_stop is None:
        # Armed BEFORE the survey, so a Tower that starts and immediately
        # begins a walk is obeyed at the first opportunity rather than at the
        # first one after a directory scan.
        stop_request.install(watch_stdin=args.stop_on_stdin_close)
        should_stop = stop_request.asked_for

    store = WorldStore(Path(args.root))
    verdicts = survey(store, max_attempts=args.max_attempts)
    report: dict = {
        "root": str(args.root),
        "sessions_seen": len(verdicts),
        "owed": [v.as_dict() for v in verdicts if v.owed],
        # COUNTS, NOT SENTENCES. This process inherits the Tower's stdout and
        # runs at every start, and the root it reads has seventy sessions on
        # the machine this was written for -- so the full list of reasons was
        # five hundred lines of console at every boot, in front of the one
        # line that matters. The sentences are still one `--dry-run` away.
        "skipped": collections.Counter(v.code for v in verdicts if not v.owed),
        "finished": [],
        "retired": [],
    }
    if args.dry_run:
        report["skipped_detail"] = [v.as_dict() for v in verdicts if not v.owed]
        _emit(report, args.format)
        return 0

    engine = WorldBuilderEngine(store)
    for verdict in verdicts:
        if not verdict.exhausted or should_stop():
            continue
        try:
            report["retired"].append(_retire(engine, verdict, args.max_attempts))
        except Exception as exc:  # noqa: BLE001 -- a record is not worth an exit
            logger.warning(
                "[Tower][WorldBuilder] could not retire %s/%s: %s",
                verdict.world_id, verdict.session_id, exc,
            )

    failures = 0
    done = 0
    # ONE AT A TIME, and the stop checked between each. A run asked to stop
    # between two worlds stops there rather than starting a second
    # six-minute stage nobody is waiting for.
    for verdict in verdicts:
        if not verdict.owed:
            continue
        if should_stop():
            report["stopped"] = f"stop requested ({stop_request.source})"
            break
        if done >= args.max_worlds:
            report["bounded_at"] = args.max_worlds
            break
        outcome = finish(
            store, verdict,
            appearance=args.appearance,
            prune_depth_work=not args.keep_depth_work,
            should_stop=should_stop,
            stop_source=lambda: stop_request.source,
        )
        report["finished"].append(outcome)
        done += 1
        if not outcome.get("finished"):
            failures += 1

    _emit(report, args.format)
    return 1 if failures else 0


def _emit(report: dict, fmt: str) -> None:
    payload = dict(report)
    # `Counter` is a dict and serialises as one; made explicit so a reader of
    # the JSON sees an object rather than wondering what shape it is.
    payload["skipped"] = dict(payload.get("skipped") or {})
    if fmt == "json":
        print(json.dumps(payload, indent=2))
        return
    print(f"root               {payload['root']}")
    print(f"sessions seen      {payload['sessions_seen']}")
    for code, count in sorted(payload["skipped"].items()):
        print(f"skipped            {count:4d}  {code}")
    for key in ("owed", "finished", "retired", "skipped_detail"):
        for row in payload.get(key) or []:
            print(f"{key:18s} {row}")


if __name__ == "__main__":
    raise SystemExit(main())
