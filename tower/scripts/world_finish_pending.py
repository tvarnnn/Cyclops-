#!/usr/bin/env python
"""Finish the photographic work a world was promised and never got.

    python scripts/world_finish_pending.py --root <world root>
        [--max-worlds 1] [--max-attempts 3] [--max-forgiven 5]
        [--no-appearance] [--keep-depth-work] [--dry-run]
        [--stop-on-stdin-close] [--format json|text]

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

THE BOUND COUNTS ONLY ATTEMPTS THAT ENDED ON THEIR OWN. "Boot the Tower,
then go for a walk" is the EXPECTED event on every start, and it ends this
process -- so counting it would retire a perfectly recoverable world to
`failed` in three ordinary days. An attempt that ended in a stop request is
given back, from the stop watcher's own thread, in the five seconds before
`terminate_tree` arrives. See `forgive_attempt`.

AND FORGIVENESS IS ITSELF BOUNDED, because unconditional it made the attempt
bound unreachable in the one case that bound exists for. On 2026-09-22 this
tool deadlocked in the native loader, did nothing at all, was killed by the
stop grace and was forgiven -- at every Tower start, leaving `{"attempts": 0,
"detail": "attempt given back: stopped (stdin-closed)"}` on disk after all of
them. The world would have said "Improving" across unlimited restarts. After
`--max-forgiven` stops the attempts count anyway, the retry bound is reached,
and the world reports honestly as a photographic build that failed. See
`DEFAULT_MAX_FORGIVEN`.

IT NEVER INVENTS WORK, AND IT IS NOT A NO-OP EITHER. Those two failures pull
in opposite directions and `assess` has to hold both.

A session with no stage record and no photographic artifact is NOT owed:
absent is not a state in the vocabulary, it means "a Tower that never tried
to make a picture of this", and reading it as "interrupted" would queue tens
of hours of GPU against 165 worlds that are finished and fine.

But the stage record was added on 2026-09-22 and is written by nothing that
has run yet, so selecting on it ALONE made this tool a no-op on the only
interrupted world in existence -- including the very one the incident above
is about, whose `surface/<session>/status.json` had said `stopped` since 82
seconds after its finalization completed. So there is a second signal, and it
is safe for exactly the same reason the first is: `surface_pipeline` alone
writes that file, so a Tower that never ran a photographic stage cannot have
left one. Measured: 166 worlds here, ONE with a `surface/` directory, and it
is the interrupted one. See `interrupted_stage_on_disk`.

Exit status is 0 whenever the run itself was sound and left nothing it could
have done -- including a run that found nothing owed, which is the common
case -- 1 when a session that was owed could not be finished, and 3
(`EXIT_MORE_OWED`) when the run finished what `--max-worlds` allowed and
another world is still owed. The Tower reads that last one as "run me again
at the next idle moment" (`tower/main.py`, `CHORE_EXIT_MORE_OWED`), and 4
(`EXIT_WAITING`) as "try again after a backoff": what is left is waiting on
somebody else -- another writer, a re-gate whose solve an owner must re-finish,
an owner's re-finish in progress.

WITH THE EVIDENCE GATE ON (review V8, `test_world_builder_finish_pending_gate.py`).
A re-gate that cannot start is decided before any work and waits without
spending the run's one-world budget (M1a); a re-gate at its attempt bound is
given up on the row and stops hiding the room and areas behind it (M1b); the
room is owed from before the re-gate publishes, so no interruption leaves it
stale (M1c); and a world an owner is re-finishing is left alone (M3b). A world
whose solve carries no gate record, and that nobody is re-finishing, takes none
of these paths.
"""

import argparse
import collections
import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.world_build_session import (  # noqa: E402
    StopRequest,
    final_surface_stages,
)
from tower.artifact_paths import artifact_root_arg  # noqa: E402
from tower.native_prewarm import prewarm_world_builder  # noqa: E402
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
# keeps killing me" -- a machine that sleeps on a timer, an out-of-memory
# kill, a driver that falls over -- and the second attempt is genuinely often
# the one that lands.
#
# AN ATTEMPT IS SPENT ONLY WHEN THIS TOOL WAS LEFT ALONE AND STILL DID NOT
# FINISH. See `forgive_attempt`: the expected event on every boot is "the
# wearer went for a walk", and counting that would retire a perfectly
# recoverable world to `failed` in three ordinary days.
DEFAULT_MAX_ATTEMPTS = 3

# HOW MANY TIMES FORGIVENESS ITSELF MAY BE GRANTED, AND WHY THE BOUND ABOVE
# WAS WORTH NOTHING WITHOUT THIS ONE.
#
# `forgive_attempt` gives the attempt back whenever a stop was REQUESTED, and
# that is right for the event it was written for. It is also unconditional,
# and on 2026-09-22 that made the attempt bound unreachable IN THE ONE CASE IT
# EXISTS FOR. The finisher deadlocked loading OpenBLAS -- `surfacify` pulling
# it in through moge -> scipy.linalg -- behind its own stdin watcher's parked
# read on descriptor 0 (`tower/stdin_stop.py` has the mechanism and the fix),
# and sat at 0% CPU making no progress whatever.
# Every Tower start ran it again, every start ended in a stop, and every stop
# handed the attempt back. The ledger on disk read, after all of them:
#
#     {"attempts": 0, "detail": "attempt given back: stopped (stdin-closed)"}
#
# Zero. The counter that was the only mechanism able to retire that world
# never passed zero, so the world would have reported "Improving" across
# unlimited restarts, for ever, with nothing on disk saying why. The root
# cause is fixed (see `tower/native_prewarm.py` and the prewarm call in
# `main`); this is the containment, so that the NEXT thing that hangs or gets
# killed instantly cannot imply progress for ever either.
#
# FIVE, AGAINST THE REAL NUMBERS. The event being forgiven is "boot the Tower,
# then go for a walk", which ends this process on an ordinary start; a busy
# day is a handful of Tower starts, not fifty. Five free passes means a
# genuinely recoverable world survives a whole day of ordinary use -- and
# several such days, because ONE start that is left alone long enough to
# finish clears the world out of the owed set entirely and the counters stop
# mattering. Past five, the attempts accumulate normally and the three above
# apply, so a world that is truly unfinishable is retired to `failed` after
# nine starts: roughly two to three ordinary days, rather than never.
#
# WHY A COUNTER AND NOT A PROGRESS TEST. The obvious better rule is "forgive
# only an attempt that actually got somewhere", and the material is nearly
# there: `surface_pipeline._status` stamps `updated_at` into
# `<world>/<stage>/<session>/status.json` at each step, so the finisher could
# sample it before the stage and again from the stop watcher's thread. It is
# the wrong rule anyway, and would have been wrong here in BOTH directions.
# The hang produced no status file at all, but neither does an honest stop
# forty seconds into a start -- the wearer pressed Start before `surfacify`
# reached its first write -- and treating that as a spent attempt puts back
# exactly the "three ordinary days retire a recoverable world" failure
# `forgive_attempt` exists to prevent. In the other direction a status file
# proves only that the process wrote one line before it wedged, which the
# deadlock would eventually have managed too. Progress is also not a fact the
# watcher thread can establish cheaply under a five-second `terminate_tree`
# grace: it would be a second read of a file another thread may be mid-write
# on. So the bound is on the number of forgiven attempts -- a fact this tool
# owns, writes itself, and can always read back.
#
# DELIBERATELY NOT A WALL CLOCK. Nothing here times a running build. A long
# walk is a long build, sixteen minutes and more, and a timer around work that
# is going fine would kill the very thing this tool was written to finish.
DEFAULT_MAX_FORGIVEN = 5

# One world per run. Not a throughput knob: the machine belongs to whoever
# picks the glasses up next, and six to sixteen minutes is already a long
# time to be holding a GPU on nobody's behalf. Two owed worlds are two runs:
# this one exits `EXIT_MORE_OWED`, and the Tower starts the next at its next
# idle moment rather than at its next start.
DEFAULT_MAX_WORLDS = 1

# "I finished what I was allowed to, and another world is still owed." Its
# twin is `tower/main.py`'s `CHORE_EXIT_MORE_OWED`; a test holds them equal.
EXIT_MORE_OWED = 3

# "A world I might owe work to is being written by somebody else right now."
# The Tower reads anything but 0 and 3 as "try again after a backoff", which
# is what this wants: not nothing-to-do, and not a spawn every tick while a
# foreign writer holds the lock.
EXIT_WAITING = 4
# Two more with the evidence gate on (review V8), and both are "somebody else must act
# first", never "run me again now": a re-gate that cannot start (its solve will not load,
# or the solve's database is gone -- an owner's re-finish), and a world an owner is
# re-finishing right now. `EXIT_MORE_OWED` is a respawn with no backoff; reporting either
# of these as owed made the refused world loop and starved every other one (M1a).
REGATE_REFUSED = "regate-refused"
REFINISH_IN_PROGRESS_CODE = "refinish-in-progress"
WAITING_CODES = ("locked", "building-now", REGATE_REFUSED, REFINISH_IN_PROGRESS_CODE)

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
    # Every ENTRY too. An entry that is not an object used to reach
    # `_counter`'s `entry.get` and raise out of `survey` -- one malformed
    # line took down the survey of EVERY world on the Tower, not just its
    # own (adversarial review, 2026-09-23).
    if not all(isinstance(entry, dict) for entry in data["sessions"].values()):
        return {}, True
    return data["sessions"], False


def _counter(entry: dict, field: str) -> int:
    """One counter out of a ledger entry, treating anything else as zero.

    ABSENT IS ZERO, AND THAT IS WHAT KEEPS OLD LEDGERS READABLE. `forgiven`
    was added on 2026-09-22 to a file that already exists beside every world
    this tool has ever touched, and those entries have `attempts` and
    `detail` and nothing else. They are not corrupt and they are not a
    special case: a session nobody has forgiven yet has been forgiven zero
    times, which is exactly what the new field would have said. `bool` is
    excluded because `True` is an `int` in Python and a counter of `True` is
    a bug, not a count.
    """
    value = entry.get(field)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def read_attempts(store: WorldStore, world_id: str, session_id: str) -> int | None:
    """How often this tool has started this session, or None if it cannot say."""
    sessions, unreadable = _read_ledger(store, world_id)
    if unreadable:
        return None
    return _counter(sessions.get(session_id) or {}, "attempts")


def read_forgiven(store: WorldStore, world_id: str, session_id: str) -> int | None:
    """How often an attempt on this session has been given back, or None.

    Reported separately from `attempts` rather than folded into it because
    the two answer different questions. `attempts` is "how many times was
    this tool left alone and still did not finish", which is what the retry
    bound is about. `forgiven` is "how many times did this tool run and end
    in a stop", which is what tells a hung finisher apart from a wearer who
    keeps going for walks -- the distinction the incident of 2026-09-22 had
    no way to make, because every one of those runs left `attempts` at zero.
    """
    sessions, unreadable = _read_ledger(store, world_id)
    if unreadable:
        return None
    return _counter(sessions.get(session_id) or {}, "forgiven")


# One writer at a time WITHIN this process. Between processes the world's
# writer lock is the guarantee; inside it, `forgive_attempt` is called from
# the stop watcher's thread while the main thread may still be in
# `record_attempt`'s read-modify-write.
_LEDGER_LOCK = threading.Lock()


def _write_ledger(store: WorldStore, world_id: str, sessions: dict) -> None:
    path = _attempts_path(store, world_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, {"schema": 1, "sessions": sessions})


def record_attempt(
    store: WorldStore, world_id: str, session_id: str, *, detail: str
) -> int:
    """Count one attempt, BEFORE the work it counts.

    Before, deliberately and for the same reason `mark_stage` writes
    `running` before a stage starts: the failure being bounded here is a
    process that does not live to write anything afterwards. An attempt
    counted on completion counts only the attempts that did not need
    counting.

    RAISES rather than shrugging when it cannot write. An attempt that could
    not be counted is an UNCOUNTED attempt, and a read-only or full disk made
    this the one path that turned the bound into its opposite: every boot
    started the same six-minute stage, none of them was ever counted, and the
    bound never advanced. The caller refuses the work instead.
    """
    with _LEDGER_LOCK:
        sessions, unreadable = _read_ledger(store, world_id)
        if unreadable:
            # AN UNREADABLE LEDGER IS SET ASIDE, NOT OBEYED FOR EVER. It used
            # to make `assess` refuse the world permanently while the serving
            # path, reading the stage record, called it `owed` -- "Improving"
            # with nothing that would ever end it (two reviewers, 2026-09-23).
            # Now it is moved beside itself, untouched, and counting starts
            # again from this attempt: the retry bound still applies, from
            # here. Called under the world's writer lock, so nothing else is
            # writing it.
            _set_aside_unreadable_ledger(store, world_id)
            sessions = {}
        sessions = dict(sessions)
        # THE ENTRY IS EDITED, NOT REPLACED. It used to be rewritten whole,
        # which was harmless while `attempts` was the only number in it and
        # fatal the moment it was not: `forgiven` is written by the stop
        # watcher on the way out of one run and read by the next one, so a
        # `record_attempt` that dropped it would reset the forgiveness bound
        # to zero at the start of every attempt -- restoring, exactly, the
        # unreachable bound of 2026-09-22 through a different door.
        entry = dict(sessions.get(session_id) or {})
        count = _counter(entry, "attempts") + 1
        entry["attempts"] = count
        entry["forgiven"] = _counter(entry, "forgiven")
        entry["detail"] = detail
        sessions[session_id] = entry
        _write_ledger(store, world_id, sessions)
    return count


def _set_aside_unreadable_ledger(store: WorldStore, world_id: str) -> None:
    """Move an unreadable ledger out of the way, keeping it for a human."""
    path = _attempts_path(store, world_id)
    if not path.exists():
        return
    aside = path.with_name(f"{path.stem}.unreadable-{int(time.time())}{path.suffix}")
    os.replace(path, aside)
    logger.warning(
        "[Tower][WorldBuilder] the finish ledger for %s was unreadable; it was "
        "moved to %s and counting starts again from this attempt",
        world_id, aside.name,
    )


def forgive_attempt(
    store: WorldStore,
    world_id: str,
    session_id: str,
    *,
    detail: str,
    max_forgiven: int = DEFAULT_MAX_FORGIVEN,
) -> None:
    """Give an attempt back, because this run was ASKED to stop.

    THE EXPECTED EVENT ON EVERY BOOT MUST BE FREE. Boot the Tower, then go
    for a walk: the chore starts a six-minute surface, the wearer presses
    Start a minute later, `capture_opened` closes the pipe, and five seconds
    later `terminate_tree` kills a process that never reached a `should_stop`
    checkpoint inside a depth pass. Counting that spends an attempt for no
    work, and three ordinary days of doing exactly what the product expects
    would retire a perfectly recoverable world to `failed` for ever.

    So the bound counts only the attempts that ENDED ON THEIR OWN: a machine
    that slept, an out-of-memory kill, a driver that fell over. Those leave
    no stop request behind, which is precisely what makes them the failure
    the bound was written for.

    AND FORGIVENESS IS ITSELF BOUNDED, because unconditional it made the
    retry bound unreachable in the one case that bound exists for. On
    2026-09-22 a finisher deadlocked in the native loader, did nothing at
    all, was killed by the stop grace, and was forgiven -- at every single
    Tower start, leaving `{"attempts": 0, "detail": "attempt given back:
    stopped (stdin-closed)"}` on disk however many times it ran. A stop
    request is evidence that THIS run was interrupted; it is not evidence
    that the run would ever have finished, and after `max_forgiven` of them
    this tool stops accepting it as such. The attempt then stands, the
    attempts accumulate, `assess` reaches the retry bound and `_retire`
    writes `failed` -- so the world reports honestly as a photographic build
    that did not happen, instead of "Improving" for ever. See
    `DEFAULT_MAX_FORGIVEN` for the number and the arithmetic behind it.

    THE COUNT IS KEPT EVEN WHEN THE FORGIVENESS IS REFUSED, and it has to be:
    it is the only durable trace that this run existed at all. A run that is
    always killed before it writes anything else is exactly the shape being
    bounded, so the increment happens on both branches and the ledger's
    `detail` says which one was taken.

    CALLED FROM THE STOP WATCHER'S THREAD, milliseconds after the pipe
    closes, and that is the whole design. The grace ends in `terminate_tree`,
    so anything this process would have done after the stage returned does
    not run -- there is no `finally` on a `TerminateProcess`. Never raises:
    a forgiveness that cannot be written costs one attempt, and taking the
    process down with it would cost the world its lock.
    """
    try:
        with _LEDGER_LOCK:
            sessions, unreadable = _read_ledger(store, world_id)
            if unreadable:
                return
            sessions = dict(sessions)
            entry = dict(sessions.get(session_id) or {})
            count = _counter(entry, "attempts")
            forgiven = _counter(entry, "forgiven")
            entry["forgiven"] = forgiven + 1
            if forgiven >= max_forgiven:
                # Refused. `attempts` is left exactly as `record_attempt`
                # wrote it, which is the whole containment: from here on the
                # retry bound advances by one at every start.
                entry["detail"] = (
                    f"attempt kept: {detail}; this session has already been "
                    f"forgiven {forgiven} times (the bound is {max_forgiven}), "
                    "so a stop is no longer taken as evidence that it would "
                    "have finished"
                )
            else:
                entry["attempts"] = max(0, count - 1)
                entry["detail"] = f"attempt given back: {detail}"
            sessions[session_id] = entry
            _write_ledger(store, world_id, sessions)
    except Exception:  # noqa: BLE001 -- see the docstring
        logger.warning(
            "[Tower][WorldBuilder] could not give back the attempt on %s/%s",
            world_id, session_id, exc_info=True,
        )


# -- the second signal -------------------------------------------------


def _stage_status(store: WorldStore, world_id: str, session_id: str,
                  stage: str) -> dict | None:
    """One stage's own `status.json`, or None when there is not a readable one.

    The same file `session_build_running` and `_lifecycle` already read. It
    is opened here rather than through them because they answer "is this
    live?", and the question here is the opposite one.
    """
    path = store.world_dir(world_id) / stage / session_id / "status.json"
    try:
        status = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # Unreadable is not "interrupted". It is nothing at all, and the
        # conservative reading of nothing is to leave the world alone.
        return None
    return status if isinstance(status, dict) else None


def interrupted_stage_on_disk(store: WorldStore, world_id: str,
                              session_id: str) -> str | None:
    """The photographic stage whose own artifact says it was interrupted.

    WHY THIS EXISTS, AND WHAT SELECTING ON THE RECORD ALONE COST.

    `Session.stages` was added on 2026-09-22 and is the precise signal. It
    is also, on the day it was added, written by nothing that has run yet.
    An adversarial review measured the first version of this tool against the
    real root: 70 sessions, ZERO owed -- and the skipped list included world
    2f447162 / session cb308801, the Tower shut down eight minutes into a
    build that this whole tool was written for. Its
    `surface/cb308801.../status.json` had said `{"state": "stopped", "stage":
    "depth"}` since 82 seconds after its finalization completed. A selector
    that is a no-op on the only evidence in existence has protected nothing,
    and would have stayed a no-op until a Tower carrying the new record had
    itself been interrupted.

    IT CANNOT WIDEN THE NET, which is the property that makes it safe. The
    stage-record clause exists to guarantee that a historical backlog cannot
    be discovered; this file gives the same guarantee from a different
    direction. A `status.json` under `<world>/surface/<session>/` is written
    by `surface_pipeline` and by nothing else, so a Tower that never ran a
    photographic stage cannot have left one. Measured on this machine: ONE
    world in 166 has a `surface/` directory at all, and it is the interrupted
    one.

    Only the two interrupted words count, exactly as in the record: `stopped`,
    or `running` whose process is gone -- decided by `status_is_stale`, the
    helper the surface pipeline itself publishes, rather than by a pid check
    invented here.
    """
    for stage in PHOTOGRAPHIC_STAGES:
        status = _stage_status(store, world_id, session_id, stage)
        if status is None:
            continue
        state = status.get("state")
        if state == STAGE_STATE_STOPPED:
            return stage
        if state == STAGE_STATE_RUNNING:
            # Lazily, and ONLY HERE: `surface_pipeline` pulls in the whole
            # reconstruction stack, and the answer for almost every session is
            # "there is no such file". Imported up front, a module that failed
            # to import took down the judgement of every historical world
            # rather than of the one session with a `running` status to judge.
            from tower.world_builder.surface_pipeline import (  # noqa: PLC0415
                status_is_stale,
            )

            if status_is_stale(status):
                return stage
    return None


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

    # AN OWNER IS RE-FINISHING THIS WORLD (review V8, M3b). Asked first, so the whole
    # re-finish -- its steps and the lock gaps between them -- reads as one waiting answer,
    # whatever the session record says at the instant this looks. See
    # `refinish_in_progress`.
    refinish = refinish_in_progress(store, world_id)
    if refinish is not None:
        logger.info("[Tower][WorldBuilder] %s/%s is left alone: %s", world_id, session_id,
                    refinish)
        return no(REFINISH_IN_PROGRESS_CODE,
                  f"{refinish}; the finisher stays out of a world while it is re-finished")

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

    # A GATED SOLVE THAT OWES ITS GATE AGAIN (review V7, H1b and L-c): the depth stage could
    # not give it a metric scale (the network missing, CUDA out of memory, the surface lock
    # held), or the gate itself raised. Asked before the photographic stages because the
    # re-gate can move pieces out of the room, which the room's stages then rebuild. Only a
    # solve the gate RAN on can owe one: this tool never computes components for a world
    # the gate never saw. A solve that lost its MASKS owes nothing here -- that needs a new
    # solve, which an owner runs attended (`world_refinish.py`; V7, H2).
    #
    # A RE-GATE THE FINISHER CANNOT RUN HIDES NOTHING (review V8, M1a and M1b). Given up at
    # its attempt bound (and said so on the row), or refused because its solve will not
    # load, the room and the areas behind it are asked exactly as for a world with no gate:
    # a re-gate that will never happen must not leave owed room or area work unfinished for
    # ever. A refused re-gate is still the answer when the room and areas owe nothing, so
    # the run reports WAITING rather than nothing-to-do.
    regate = _assess_regate(store, world_id, session_id, session, max_attempts=max_attempts)
    if regate is not None and regate.code != REGATE_REFUSED:
        return regate
    room = _assess_room(store, world_id, session_id, session, max_attempts=max_attempts)
    if regate is not None and not (room.owed or room.exhausted or room.code in WAITING_CODES):
        return regate
    return room


def _assess_room(store: WorldStore, world_id: str, session_id: str, session, *,
                 max_attempts: int) -> Verdict:
    """`assess`, for the room's photographic stages and then the session's areas: what a
    session owes once the re-gate owes nothing this tool can do."""

    def no(code: str, reason: str, **kwargs) -> Verdict:
        return Verdict(world_id, session_id, False, reason, code=code, **kwargs)

    # TWO SIGNALS, IN PRECEDENCE ORDER, AND NEITHER CAN DISCOVER A BACKLOG.
    #
    # THE RECORD FIRST, wherever there is one. `Session.stages` is written
    # last, by the builder itself, and it is the more precise of the two: a
    # record saying the surface finished is finished, whatever a `status.json`
    # from an earlier COARSE build during the walk still says. Reading the
    # file over the record would resurrect worlds that are already done.
    #
    # THE STAGE'S OWN ARTIFACT otherwise. `stages` absent is not a state in
    # the vocabulary -- it means "a Tower that never recorded this", which is
    # every session written before 2026-09-22 -- but it is not the same fact
    # as "no photographic work was ever attempted", and conflating the two is
    # what made the first version of this tool a no-op on the one world the
    # incident was about. See `interrupted_stage_on_disk`: a
    # `surface/<session>/status.json` cannot exist on a Tower that never ran
    # the stage, so this reads the same guarantee off a different file.
    #
    # Absent record AND no interrupted artifact is the historical world, and
    # it stays untouched -- 165 of the 166 on this machine.
    by_record = bool(session.stages)
    if by_record:
        stage = None
        for candidate in PHOTOGRAPHIC_STAGES:
            entry = session.stages.get(candidate) or {}
            if entry.get("state") in INTERRUPTED_STATES:
                stage = candidate
                break
        if stage is None:
            # THE ROOM OWES NOTHING; ITS AREAS MAY (WORLD-BUILDER-COMPONENTS.md
            # §3.4, §7 rule 5). Asked only here, after the room, because the row
            # carries an area's word only when the room's is settled -- and never
            # of a session without a components record: `null` owes nothing, so
            # no world is rebuilt into components by this tool.
            areas = _assess_areas(store, world_id, session_id, session,
                                  max_attempts=max_attempts)
            if areas is not None:
                return areas
            return no("nothing-interrupted",
                      "no photographic stage is in an interrupted state")
    else:
        stage = interrupted_stage_on_disk(store, world_id, session_id)
        if stage is None:
            # ASKED HERE, not only below. `interrupted_stage_on_disk` reads a
            # LIVE `running` status as "not interrupted", which is correct --
            # and would then have been reported as "no Tower ever tried to
            # make a picture of this", which is the opposite of true. The
            # probe is worth its cost on this branch alone: three sessions on
            # the real root reach it.
            if session_build_running(store, world_id, session_id):
                return no("building-now",
                          "a photographic stage for this session is running "
                          "right now")
            areas = _assess_areas(store, world_id, session_id, session,
                                  max_attempts=max_attempts)
            if areas is not None:
                return areas
            return no(
                "no-stage-record",
                "this session has no stage record and no interrupted "
                "photographic artifact, which means a Tower that never tried "
                "to make a picture of it -- not an interrupted stage"
            )

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
    #
    # OUR OWN LOCK IS NOT A FOREIGN WRITER, the same exclusion
    # `world_builder_library._world_is_live` makes and for the same reason.
    # Without it `_retire` could not re-`assess` underneath the lock it just
    # took -- and re-checking under the lock is the entire point of taking it.
    holder = store.lock_holder(world_id)
    if holder is not None and holder["pid"] == os.getpid():
        holder = None
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
        # Owed, not refused: `record_attempt` sets the unreadable ledger aside
        # under the world's lock and counts from there, so the bound still
        # holds. Refusing here was "Improving" for ever on the phone.
        attempts = 0
    if attempts >= max_attempts:
        return no(
            "attempt-bound",
            f"this tool has already started this session {attempts} times "
            f"(the bound is {max_attempts}) and something ended it every time",
            stage=stage,
            attempts=attempts,
            exhausted=True,
        )

    if by_record:
        evidence = (
            f"the {stage} stage is recorded "
            f"{(session.stages[stage] or {}).get('state')!r}"
        )
    else:
        evidence = (
            f"this session has no stage record, but its {stage} stage left an "
            "interrupted status.json behind"
        )
    return Verdict(
        world_id,
        session_id,
        True,
        f"{evidence} and nothing is building it",
        # The two signals are told apart in the report as well as in the code:
        # "owed-by-status" is the one that reaches back past the record, and
        # an operator should be able to see which of them fired.
        code="owed" if by_record else "owed-by-status",
        stage=stage,
        attempts=attempts,
    )


# -- areas (WORLD-BUILDER-COMPONENTS.md §5.4, §7 rule 5) ---------------
#
# An AREA is a part of a walk the evidence gate did not place; it is built on its
# own (`tower/world_builder/area_build.py`) after the room. Its work is owed exactly
# when the session's components record -- written by the gate with the published
# solve, and by nothing here -- names an area whose build is unsettled, which is the
# same judgement the row's `photographic` block makes (`components.area_words`), so
# the tool that builds the work and the row that reports it cannot disagree about it.
#
# THIS TOOL NEVER COMPUTES COMPONENTS. A session with no record (`components: null`,
# every world built before the gate) owes no area work, whatever it looks like.

# The verdict's `stage` for owed area work, and the attempt ledger's key for it: a
# counter of its own, so a room that needed two attempts does not leave its areas one.
AREAS_STAGE = "areas"


def area_ledger_key(session_id: str) -> str:
    return f"{session_id}#{AREAS_STAGE}"


def _owed_areas(store: WorldStore, world_id: str, session_id: str, session):
    """`(record, {area_id: word})` of the unsettled areas, or `(None, {})`.

    Light on purpose: the survey runs at every Tower start over every session, and a
    session without a record costs one `stat` here and imports nothing heavy."""
    from tower.world_builder.components import (  # noqa: PLC0415
        area_words,
        read_components_record,
    )
    from tower.world_builder.photographic import is_unsettled  # noqa: PLC0415

    record = read_components_record(store, world_id, session_id)
    if record is None:
        return None, {}
    words = area_words(store, world_id, session_id, session, record)
    return record, {aid: w for aid, w in words.items() if is_unsettled(w.get("state"))}


def _assess_areas(store: WorldStore, world_id: str, session_id: str, session, *,
                  max_attempts: int) -> "Verdict | None":
    """A verdict about this session's AREAS, or None when they owe nothing (which
    includes every session with no components record)."""
    try:
        _record, owed = _owed_areas(store, world_id, session_id, session)
    except Exception as exc:  # noqa: BLE001 -- one session's areas, not the survey
        logger.warning("[Tower][WorldBuilder] could not assess the areas of %s/%s: %s: %s",
                       world_id, session_id, type(exc).__name__, exc)
        return None
    if not owed:
        return None
    if any(word.get("state") == "running" for word in owed.values()):
        return Verdict(world_id, session_id, False,
                       "an area of this session is being built right now",
                       code="building-now", stage=AREAS_STAGE)
    holder = store.lock_holder(world_id)
    if holder is not None and holder["pid"] == os.getpid():
        holder = None
    if holder is not None and (holder["alive"] or holder["unreadable"]):
        return Verdict(world_id, session_id, False,
                       "this world's writer lock is held by another process",
                       code="locked", stage=AREAS_STAGE)
    attempts = read_attempts(store, world_id, area_ledger_key(session_id))
    if attempts is None:
        attempts = 0
    if attempts >= max_attempts:
        return Verdict(world_id, session_id, False,
                       f"this tool has already started this session's areas {attempts} "
                       f"times (the bound is {max_attempts}) and something ended it "
                       "every time", code="attempt-bound", stage=AREAS_STAGE,
                       attempts=attempts, exhausted=True)
    return Verdict(world_id, session_id, True,
                   f"{len(owed)} area(s) of this session are unsettled and nothing is "
                   "building them", code="owed-area", stage=AREAS_STAGE,
                   attempts=attempts)


# -- a re-gate owed to a gate that could not finish (review V7, H1b and L-c) ------------
#
# The evidence gate's scale fail-safe publishes the room as its unsplit anchor block, which
# on GT's masked arms leaves [129, 89, 120, 44] misplaced where the gate leaves [24, 0, 0,
# 38] (V7) -- tolerable for an interim publish, not as a world's final answer. When it was
# caused by the depth stage (`gate.retryable`, `gate.cause` depth-unavailable) or the gate
# raised (gate-failed), the session is owed ONE thing: the depth stage and the gate run
# again IN PLACE on the published solve (`coherence_publish.regate_published`) -- the solve
# is not redone and nothing is moved aside -- then the derived tree and the room's stages
# are rebuilt from it; its areas are owed by the new record. Under the writer lock, on its
# own attempt ledger key, bounded like every other stage.
#
# There is deliberately no unattended RE-SOLVE (V7, H2): a solve that lost its masks keeps
# `transients.retryable` and says so on the row; an owner re-finishes it attended.

REGATE_STAGE = "regate"


def regate_ledger_key(session_id: str) -> str:
    return f"{session_id}#{REGATE_STAGE}"


def _assess_regate(store: WorldStore, world_id: str, session_id: str, session, *,
                   max_attempts: int) -> "Verdict | None":
    """A verdict about the published solve's re-gate, or None when it owes none -- which
    includes every solve the gate never ran on, and a re-gate given up on the record.

    Reads only. Every answer that is not `owed-regate` is decided HERE, before any lock,
    attempt or warm is spent on it (review V8, M1a)."""
    from tower.world_builder.coherence_publish import regate_owed  # noqa: PLC0415

    try:
        cause = regate_owed(store, world_id, session_id)
    except Exception as exc:  # noqa: BLE001 -- one session, not the survey
        logger.warning("[Tower][WorldBuilder] could not read the gate record of %s/%s: %s",
                       world_id, session_id, exc)
        return None
    if cause is None:
        return None
    if session_build_running(store, world_id, session_id):
        return Verdict(world_id, session_id, False, "this session is being built right now",
                       code="building-now", stage=REGATE_STAGE)
    holder = store.lock_holder(world_id)
    if holder is not None and holder["pid"] == os.getpid():
        holder = None
    if holder is not None and (holder["alive"] or holder["unreadable"]):
        return Verdict(world_id, session_id, False,
                       "this world's writer lock is held by another process",
                       code="locked", stage=REGATE_STAGE)
    attempts = read_attempts(store, world_id, regate_ledger_key(session_id)) or 0
    if attempts >= max_attempts:
        # GIVEN UP, AND SAID SO ON THE ROW (review V8, M1b). The published fail-safe stands.
        # What the row said until now was `publish_notice`'s promise that the idle Tower
        # re-runs the gate, which is no longer true -- so the bound is `exhausted` until
        # `_retire_regate` has put the given-up sentence in its place. THE RECORD IS THE
        # SENTENCE: once the finalization detail says it, the re-gate owes nothing and
        # `assess` asks the room and the areas behind it, which it used to hide for ever.
        # Anything that puts the promise back (a hand-run `world_finalize.py`) is corrected
        # at the next run.
        notice = regate_given_up_notice(_published_meta(store, world_id, session_id),
                                        attempts)
        if (session.finalization or {}).get("detail") == notice:
            return None
        return Verdict(world_id, session_id, False,
                       f"this tool has already re-run the gate of this session {attempts} "
                       f"times (the bound is {max_attempts}); {cause} every time",
                       code="attempt-bound", stage=REGATE_STAGE, attempts=attempts,
                       exhausted=True)
    refusal = _regate_refusal(store, world_id, session_id)
    if refusal is not None:
        # Not owed, and not `exhausted`: nothing was attempted, so nothing is given up.
        # WAITING -- an owner's re-finish is what it needs -- and read-only: the run never
        # takes the lock, counts an attempt or warms the native stack for it (M1a).
        logger.info("[Tower][WorldBuilder] the re-gate of %s/%s cannot start: %s",
                    world_id, session_id, refusal)
        return Verdict(world_id, session_id, False,
                       f"the published solve's evidence gate could not finish ({cause}), and "
                       f"a re-gate cannot start: {refusal}", code=REGATE_REFUSED,
                       stage=REGATE_STAGE, attempts=attempts)
    return Verdict(world_id, session_id, True,
                   f"the published solve's evidence gate could not finish ({cause}); it is "
                   "owed a re-gate in place", code="owed-regate", stage=REGATE_STAGE,
                   attempts=attempts)


def _published_meta(store: WorldStore, world_id: str, session_id: str) -> dict:
    """The published `solution.json`, or {} when there is no readable one."""
    path = store.world_dir(world_id) / "solve" / session_id / "solution.json"
    try:
        meta = read_json_closed(path) if path.exists() else {}
    except (OSError, ValueError):
        return {}
    return meta if isinstance(meta, dict) else {}


def _regate_refusal(store: WorldStore, world_id: str, session_id: str) -> str | None:
    """Why `coherence_publish.regate_published` would refuse this session, or None. Reads
    only.

    THE SAME TWO REFUSALS, WORD FOR WORD: the published solve does not load as a gated
    solution, or the feature database it was solved from is gone. Restated here rather than
    shared because `coherence_publish.py` is another lane's file; a test holds the two to
    one answer on every shape (`test_the_read_only_refusal_agrees_with_regate_published`).
    `regate_published` still refuses under the lock, and `finish_regate` still gives that
    attempt back -- this only moves the common case in front of the lock."""
    from tower.world_builder.global_solve import load_solution, workspace_for  # noqa: PLC0415

    solution = load_solution(store, world_id, session_id)
    if solution is None or not isinstance(solution.gate, dict):
        return "no gated solution is published for this session"
    workspace = workspace_for(store, world_id, session_id)
    name = (solution.solve or {}).get("database") or workspace.database_path.name
    if not (workspace.root / name).is_file():
        return f"the solve's database {name} is gone; an owner can re-finish this walk"
    return None


# The row's sentence once the finisher has stopped re-running a gate: the cause, as
# `publish_notice` states it, and the owner's way out instead of the idle Tower's promise.
REGATE_GIVEN_UP = ("{what}; the idle Tower re-ran the gate {attempts} times without "
                   "finishing it and has stopped trying; an owner can re-finish this walk")


def regate_given_up_notice(meta: dict, attempts: int) -> str:
    """The finalization detail for a re-gate given up at its bound (review V8, M1b).

    `publish_notice`'s sentence for the same record, with its promise -- "the Tower re-runs
    the gate when it is idle" -- replaced by what is now true. The masks sentence, when the
    solve has one, is kept exactly as `publish_notice` words it. Deterministic in the
    published record and the count, because it is also the record that the give-up
    happened (`_assess_regate`)."""
    from tower.world_builder import coherence_publish as CP  # noqa: PLC0415

    gate = (meta or {}).get("gate") or {}
    parts = []
    masks = CP.publish_notice({"transients": (meta or {}).get("transients") or {}, "gate": {}})
    if masks:
        parts.append(masks)
    if gate.get("state") == CP.GATE_STATE_FAILED:
        what = f"the evidence gate failed ({gate.get('detail') or 'an error'})"
    else:
        why = (gate.get("depth") or {}).get("detail") or gate.get("cause") or "no depth"
        what = f"the evidence gate could not measure metric scale ({why})"
    parts.append(REGATE_GIVEN_UP.format(what=what, attempts=attempts))
    return "; ".join(parts)


# -- a re-finish in progress (review V8, M3b) --------------------------
#
# `scripts/world_refinish.py` rebuilds one session in steps, and between them it holds no
# lock: after step 1 (the previous solve set aside, the room's stages recorded `stopped`
# "re-finish in progress") and after step 2 (the new solve published by its child
# `world_finalize.py`, which took and released the lock itself). A `stopped` room is owed
# work, so an idle Tower's finisher picked it up in exactly those gaps -- building the room
# from a solve that was set aside or not yet rebuilt, and holding the lock the re-finish's
# next step then could not take.
#
# WHAT "IN PROGRESS" MEANS, from PF's `world_refinish.py` as it is: its ledger,
# `<world>/refinish/<stamp>/refinish.json`, and the room marker it writes.
#
# * `state: setting-aside` -- step 1 has not completed. Under the world's lock while it
#   runs; left behind, it means a set-aside that never finished, and nothing should build
#   on that world until its owner looks.
# * otherwise, the room of a session of this world still carries THIS stamp's marker
#   (`REFINISH_IN_PROGRESS`, `stopped`). The marker is written under the step-1 lock and is
#   gone as soon as step 3 records the rebuilt room, or a put-back restores the session
#   record -- so it covers precisely the gaps. The ledger's `state` alone cannot say this:
#   a re-finish that SUCCEEDED leaves it `set-aside` for ever (nothing writes a terminal
#   state), and reading that as live would park the world for good.
#
# A MALFORMED LEDGER (unreadable, not an object, an unknown state) is therefore decided by
# the marker too, and that is the fail-safe choice in both directions: with the marker on
# the room the finisher stays out (a live re-finish is never interfered with, which is the
# failure that can end in a half-restored world), and without it the world takes today's
# path (a corrupt file cannot park a world on "Improving" for ever).
#
# THE COST, stated: a re-finish that DIED between its steps looks exactly like a live one
# on disk -- the ledger records no pid -- so the finisher leaves that world alone until its
# owner runs the re-finish again (which is what a set-aside solve needs anyway). OPEN for
# the re-finish's owner: a pid and a terminal state in the ledger would let this tell the
# two apart.


def refinish_in_progress(store: WorldStore, world_id: str) -> str | None:
    """Why an owner's re-finish of this world is live, or None. Reads only; one `stat` for
    a world that was never re-finished."""
    try:
        from scripts import world_refinish as WR  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001 -- then no re-finish can be running either
        logger.warning("[Tower][WorldBuilder] scripts/world_refinish.py does not import "
                       "(%s: %s); no re-finish is looked for", type(exc).__name__, exc)
        return None
    root = store.world_dir(world_id) / WR.REFINISH_DIRNAME
    try:
        if not root.is_dir():
            return None
        stamps = sorted(entry for entry in root.iterdir() if entry.is_dir())
    except OSError as exc:
        # Not "no re-finish": it could not be looked for. Staying out costs a wait.
        return f"this world's {WR.REFINISH_DIRNAME}/ could not be read ({exc})"
    for aside in stamps:
        ledger_path = aside / WR.LEDGER_FILENAME
        if not ledger_path.is_file():
            continue            # the ledger is written first: nothing was set aside here
        try:
            ledger = read_json_closed(ledger_path)
        except (OSError, ValueError):
            ledger = None
        state = ledger.get("state") if isinstance(ledger, dict) else None
        if state == WR.LEDGER_SETTING_ASIDE:
            return (f"re-finish {aside.name} is setting the previous result aside "
                    f"({WR.REFINISH_DIRNAME}/{aside.name}/{WR.LEDGER_FILENAME})")
        marker = WR.REFINISH_IN_PROGRESS.format(stamp=aside.name)
        marked = _room_marked(store, world_id, marker)
        if marked is not None:
            return (f"re-finish {aside.name} (ledger "
                    f"{state if isinstance(state, str) else 'unreadable'}) has set the "
                    f"solve of session {marked} aside and not rebuilt its room yet")
    return None


def _refinish_under_the_lock(store: WorldStore, verdict: Verdict, report: dict) -> bool:
    """True (and `report` says so) when a re-finish of the verdict's world is live, asked
    once the world's lock is ours. A verdict comes from the survey at the start of the run,
    and an owner may have started a re-finish since; the re-finish marks the room under the
    same lock, so the answer given under it is final. Waiting, for the whole world."""
    busy = refinish_in_progress(store, verdict.world_id)
    if busy is None:
        return False
    logger.info("[Tower][WorldBuilder] %s/%s is left alone: %s", verdict.world_id,
                verdict.session_id, busy)
    report.update({"finished": False, "waiting": True, "world_busy": True, "reason": busy})
    return True


def _room_marked(store: WorldStore, world_id: str, marker: str) -> str | None:
    """The session of this world whose room a re-finish marked `stopped` with `marker`."""
    for session_id in store.list_session_ids(world_id):
        try:
            stages = store.read_session(world_id, session_id).stages or {}
        except (WorldStoreError, OSError, ValueError, KeyError):
            continue
        for stage in PHOTOGRAPHIC_STAGES:
            entry = stages.get(stage) or {}
            if entry.get("state") == STAGE_STATE_STOPPED and entry.get("detail") == marker:
                return session_id
    return None


def _give_back_attempt(store: WorldStore, world_id: str, key: str, *, detail: str) -> None:
    """Undo one `record_attempt` for work that was REFUSED before it began. Not
    `forgive_attempt`, whose bound is about runs that were stopped: a refusal did no
    work at all, so it is not counted, and not bounded either."""
    with _LEDGER_LOCK:
        sessions, unreadable = _read_ledger(store, world_id)
        if unreadable:
            return
        sessions = dict(sessions)
        entry = dict(sessions.get(key) or {})
        entry["attempts"] = max(0, _counter(entry, "attempts") - 1)
        entry["forgiven"] = _counter(entry, "forgiven")
        entry["detail"] = detail
        sessions[key] = entry
        _write_ledger(store, world_id, sessions)


# What the room's stages say between the moment the re-gate is about to publish and the
# moment the room is rebuilt from it (review V8, M1c).
REGATE_IN_PROGRESS = ("re-gate in progress (scripts/world_finish_pending.py): the evidence "
                      "gate is re-run on the published solve, and the room is rebuilt from "
                      "it afterwards")


def _room_record(store: WorldStore, world_id: str, session_id: str) -> dict | None:
    """The session's stage record as it is now (None for a session that has none)."""
    stages = store.read_session(world_id, session_id).stages
    return None if stages is None else dict(stages)


def _put_room_record_back(store: WorldStore, world_id: str, session_id: str,
                          before: dict | None) -> None:
    """Restore the room's stage entries exactly as `_room_record` read them, under the
    caller's lock, leaving any other stage's entry as it is now. Only for a re-gate that
    was REFUSED: it did nothing, so the room it marked owed is exactly as good as it was."""
    from dataclasses import replace  # noqa: PLC0415

    session = store.read_session(world_id, session_id)
    stages = dict(session.stages or {})
    for stage in PHOTOGRAPHIC_STAGES:
        if before is not None and stage in before:
            stages[stage] = before[stage]
        else:
            stages.pop(stage, None)
    store.write_session(replace(session, stages=None if before is None and not stages
                                else stages))


def finish_regate(store: WorldStore, verdict: Verdict, *, appearance: bool, prune_depth_work: bool,
                  stop_request, max_forgiven: int = DEFAULT_MAX_FORGIVEN, regate=None,
                  surface_stages=None) -> dict:
    """The owed re-gate, under the world's writer lock: `regate_published`, then the derived
    tree (`engine.build`) and the room's stages (`final_surface_stages`) from the relabelled
    solve. A refusal before any work (`RegateRefused`) gives the attempt back: waiting.

    THE ROOM IS OWED FROM BEFORE THE PUBLISH (review V8, M1c). Its stages are recorded
    `stopped` (`REGATE_IN_PROGRESS`) under this lock, right before `regate_published`
    replaces the solve and its components. A stop, a crash or a kill anywhere after that --
    between the publish and the rebuild there is no `finally` a `terminate_tree` would run
    -- leaves a room the next run rebuilds from the new partition, where it used to leave
    the OLD room standing over the new solve with nothing owed. The mark is the builder's
    own discipline (`mark_stage` `running` before a stage) and `world_refinish.py`'s."""
    from tower.world_builder import coherence_publish as CP  # noqa: PLC0415

    key = regate_ledger_key(verdict.session_id)
    report: dict = {"world_id": verdict.world_id, "session_id": verdict.session_id,
                    "stage": REGATE_STAGE}
    engine = WorldBuilderEngine(store)
    try:
        store.acquire_writer_lock(verdict.world_id)
    except Exception as exc:  # noqa: BLE001
        # `world_busy`: another writer has this WORLD, so the run does not try the world's
        # other sessions either (`main`).
        report.update({"finished": False, "waiting": True, "world_busy": True,
                       "reason": f"{type(exc).__name__}: {exc}"})
        return report
    disarm = None
    try:
        if _refinish_under_the_lock(store, verdict, report):
            return report
        try:
            report["attempts"] = record_attempt(store, verdict.world_id, key,
                                                detail="re-running the evidence gate in place")
        except Exception as exc:  # noqa: BLE001
            report.update({"finished": False,
                           "reason": f"the attempt could not be counted, so it was not started: "
                                     f"{type(exc).__name__}: {exc}"})
            return report
        disarm = _forgive_on_stop(
            stop_request,
            lambda source: forgive_attempt(store, verdict.world_id, key,
                                           detail=f"stopped ({source})", max_forgiven=max_forgiven))
        prewarm_world_builder()
        if stop_request.asked:
            # Asked before anything is published: nothing to mark and nothing to rebuild.
            # (The attempt is given back on the way out, below.)
            report.update({"finished": False, "reason": f"stopped ({stop_request.source})"})
            return report
        before = _room_record(store, verdict.world_id, verdict.session_id)
        for stage in PHOTOGRAPHIC_STAGES:
            engine.mark_stage(verdict.world_id, verdict.session_id, stage,
                              state=STAGE_STATE_STOPPED, detail=REGATE_IN_PROGRESS)
        try:
            report["regate"] = (regate or CP.regate_published)(
                store, verdict.world_id, verdict.session_id, should_stop=stop_request.asked_for)
        except CP.RegateRefused as exc:
            _give_back_attempt(store, verdict.world_id, key,
                               detail=f"not started, waiting: {exc}")
            try:
                _put_room_record_back(store, verdict.world_id, verdict.session_id, before)
            except Exception:  # noqa: BLE001 -- the room then stays owed: rebuilt, not stale
                logger.warning("[Tower][WorldBuilder] could not put the room's stage record "
                               "of %s/%s back after a refused re-gate", verdict.world_id,
                               verdict.session_id, exc_info=True)
            report.update({"finished": False, "waiting": True, "reason": str(exc)})
            return report
        # The row's sentence follows the re-gate: cleared, or the new reason.
        fin = store.read_session(verdict.world_id, verdict.session_id).finalization or {}
        if fin.get("state"):
            engine.mark_finalization(verdict.world_id, verdict.session_id, state=fin["state"],
                                     final_solve=fin.get("final_solve"),
                                     detail=(report["regate"] or {}).get("notice"))
        if stop_request.asked:
            # The room stays `stopped` -- owed -- and the next run rebuilds it from the new
            # partition (M1c).
            report.update({"finished": False, "reason": f"stopped ({stop_request.source})"})
            return report
        report["build"] = {"poses_solved": getattr(engine.build(verdict.world_id, verdict.session_id),
                                                   "poses_solved", None)}
        report["stages"] = (surface_stages or final_surface_stages)(
            store, verdict.world_id, verdict.session_id, solved=True, appearance=appearance,
            prune_depth_work=prune_depth_work, should_stop=stop_request.asked_for,
            stop_source=lambda: stop_request.source,
            record=_recorder(engine, verdict.world_id, verdict.session_id))
        report["finished"] = True
    except Exception as exc:  # noqa: BLE001 -- recorded, then reported
        report.update({"finished": False, "reason": f"{type(exc).__name__}: {exc}"})
    finally:
        if disarm is not None:
            forgiven = disarm()
            if not forgiven and stop_request.asked:
                forgive_attempt(store, verdict.world_id, key, detail=f"stopped ({stop_request.source})",
                                max_forgiven=max_forgiven)
        engine.release_world(verdict.world_id)
    return report


def build_session_areas(store: WorldStore, world_id: str, session_id: str, *,
                        appearance: bool, prune_depth_work: bool, should_stop,
                        stop_source=lambda: None, build: bool | None = None,
                        area_ids=None) -> list:
    """Build (or, with area builds off, decline) this session's unsettled areas, one
    at a time, the stop checked between each. The caller holds the world's lock.

    `build` None reads `TOWER_WORLD_AREA_BUILDS` (off: every owed area is recorded as
    declined, a settled word, rather than left owed for ever). `area_ids` restricts
    the set (the re-finish command passes every area). Returns one report per area.
    """
    from tower.config import world_area_builds_setting  # noqa: PLC0415
    from tower.world_builder import area_build as AB  # noqa: PLC0415

    session = store.read_session(world_id, session_id)
    record, owed = _owed_areas(store, world_id, session_id, session)
    if record is None:
        return []
    targets = list(area_ids) if area_ids is not None else list(owed)
    build = world_area_builds_setting() if build is None else build
    # The depth the area's levelling runs must be the depth its surface stage will
    # ask for, or the predictions are made twice: `final_surface_stages` asks for the
    # FoV-told depth exactly when the evidence gate is on.
    # Keyed on the published solve's gate record, as the room's surface is (review V7, L-a).
    from scripts.world_build_session import _solution_gated  # noqa: PLC0415

    known_fov = _solution_gated(store, world_id, session_id)
    reports = []
    for area_id in targets:
        if should_stop():
            reports.append({"area_id": area_id, "built": False,
                            "reason": f"stop requested ({stop_source()})"})
            break
        if not build:
            AB.decline_area(store, world_id, session_id, area_id, record.sha1,
                            AB.AREA_BUILDS_OFF_DETAIL)
            reports.append({"area_id": area_id, "built": False, "declined": True,
                            "reason": AB.AREA_BUILDS_OFF_DETAIL})
            continue
        reports.append(AB.build_area(
            store, world_id, session_id, area_id, record,
            final_surface_stages=final_surface_stages, appearance=appearance,
            prune_depth_work=prune_depth_work, should_stop=should_stop,
            stop_source=stop_source, depth_known_fov=known_fov))
    return reports


def finish_areas(store: WorldStore, verdict: Verdict, *, appearance: bool,
                 prune_depth_work: bool, stop_request,
                 max_forgiven: int = DEFAULT_MAX_FORGIVEN) -> dict:
    """`finish`, for owed area work: the same lock, attempt and forgiveness
    discipline, on the areas' own ledger key."""
    key = area_ledger_key(verdict.session_id)
    report: dict = {"world_id": verdict.world_id, "session_id": verdict.session_id,
                    "stage": AREAS_STAGE}
    engine = WorldBuilderEngine(store)
    try:
        store.acquire_writer_lock(verdict.world_id)
    except Exception as exc:  # noqa: BLE001
        report.update({"finished": False, "reason": f"{type(exc).__name__}: {exc}"})
        return report
    disarm = None
    try:
        if _refinish_under_the_lock(store, verdict, report):
            return report
        try:
            report["attempts"] = record_attempt(store, verdict.world_id, key,
                                                detail="finishing this session's areas")
        except Exception as exc:  # noqa: BLE001
            report.update({"finished": False,
                           "reason": f"the attempt could not be counted, so it was not "
                                     f"started: {type(exc).__name__}: {exc}"})
            return report
        disarm = _forgive_on_stop(
            stop_request,
            lambda source: forgive_attempt(store, verdict.world_id, key,
                                           detail=f"stopped ({source})",
                                           max_forgiven=max_forgiven))
        prewarm_world_builder()
        report["areas"] = build_session_areas(
            store, verdict.world_id, verdict.session_id, appearance=appearance,
            prune_depth_work=prune_depth_work, should_stop=stop_request.asked_for,
            stop_source=lambda: stop_request.source)
        report["finished"] = True
    except Exception as exc:  # noqa: BLE001 -- recorded on the area, then reported
        report.update({"finished": False, "reason": f"{type(exc).__name__}: {exc}"})
    finally:
        if disarm is not None:
            forgiven = disarm()
            if not forgiven and stop_request.asked:
                forgive_attempt(store, verdict.world_id, key,
                                detail=f"stopped ({stop_request.source})",
                                max_forgiven=max_forgiven)
        engine.release_world(verdict.world_id)
    return report


def _retire_areas(store: WorldStore, verdict: Verdict, max_attempts: int) -> dict:
    """`_retire` for area work: every still-unsettled area is recorded `failed`,
    once, under the lock, after re-assessing underneath it."""
    from tower.world_builder import area_build as AB  # noqa: PLC0415

    out = {"world_id": verdict.world_id, "session_id": verdict.session_id,
           "retired": AREAS_STAGE, "attempts": verdict.attempts}
    try:
        store.acquire_writer_lock(verdict.world_id)
    except Exception as exc:  # noqa: BLE001
        out.update({"retired": None, "reason": f"{type(exc).__name__}: {exc}"})
        return out
    try:
        again = assess(store, verdict.world_id, verdict.session_id,
                       max_attempts=max_attempts)
        if not (again.exhausted and again.stage == AREAS_STAGE):
            out.update({"retired": None, "reason": f"no longer {verdict.code}: {again.code}"})
            return out
        session = store.read_session(verdict.world_id, verdict.session_id)
        record, owed = _owed_areas(store, verdict.world_id, verdict.session_id, session)
        detail = (f"scripts/world_finish_pending.py started this session's areas "
                  f"{verdict.attempts} times (bound {max_attempts}) and something ended "
                  "it on its own every time; not retrying. scripts/world_refinish.py "
                  "rebuilds them by hand.")
        for area_id in owed:
            AB.area_recorder(store, verdict.world_id, verdict.session_id, area_id,
                             record.sha1)(STAGE_SURFACE, state=STAGE_STATE_FAILED,
                                          detail=detail)
        out["areas"] = list(owed)
    finally:
        store.release_writer_lock(verdict.world_id)
    return out


def _retire_regate(store: WorldStore, verdict: Verdict, max_attempts: int) -> dict:
    """`_retire` for a re-gate at its bound (review V8, M1b): the row's promise that the
    idle Tower re-runs the gate is replaced by the given-up sentence -- the existing notice
    mechanism, the session's finalization detail -- once, under the lock, after
    re-assessing underneath it. Nothing else changes: the published fail-safe stands, and
    `assess` now asks the room and the areas behind it."""
    out = {"world_id": verdict.world_id, "session_id": verdict.session_id,
           "retired": REGATE_STAGE, "attempts": verdict.attempts}
    try:
        store.acquire_writer_lock(verdict.world_id)
    except Exception as exc:  # noqa: BLE001
        out.update({"retired": None, "reason": f"{type(exc).__name__}: {exc}"})
        return out
    try:
        again = assess(store, verdict.world_id, verdict.session_id, max_attempts=max_attempts)
        if not (again.exhausted and again.stage == REGATE_STAGE):
            out.update({"retired": None, "reason": f"no longer {verdict.code}: {again.code}"})
            return out
        notice = regate_given_up_notice(
            _published_meta(store, verdict.world_id, verdict.session_id), again.attempts)
        fin = store.read_session(verdict.world_id, verdict.session_id).finalization or {}
        WorldBuilderEngine(store).mark_finalization(
            verdict.world_id, verdict.session_id, state=fin["state"],
            final_solve=fin.get("final_solve"), detail=notice)
        out["notice"] = notice
    finally:
        store.release_writer_lock(verdict.world_id)
    return out


def survey(store: WorldStore, *, max_attempts: int = DEFAULT_MAX_ATTEMPTS) -> list:
    """Every session under the root, assessed. Reads only."""
    verdicts = []
    for world_id in store.list_world_ids():
        for session_id in store.list_session_ids(world_id):
            try:
                verdicts.append(
                    assess(store, world_id, session_id, max_attempts=max_attempts))
            except Exception as exc:  # noqa: BLE001 -- one session, not the root
                # ONE SESSION'S PROBLEM STAYS ONE SESSION'S. An exception here
                # used to end the whole survey, so every owed world on the
                # Tower waited on the one that could not be read.
                logger.warning(
                    "[Tower][WorldBuilder] could not assess %s/%s: %s: %s",
                    world_id, session_id, type(exc).__name__, exc,
                )
                verdicts.append(Verdict(
                    world_id, session_id, False,
                    f"this session could not be assessed: {type(exc).__name__}: {exc}",
                    code="assess-error",
                ))
    return verdicts


# -- doing the work ----------------------------------------------------


def _retire(store: WorldStore, verdict: Verdict, max_attempts: int) -> dict:
    """Write down that this tool has given up on a session, once.

    "Bounded" on its own would leave a world sparse with nothing on disk
    saying why, which is the exact failure the stage record was added to fix
    -- it would just move it one level up. `failed` is the honest word and it
    is terminal, so this also stops the session being reassessed at every
    boot forever.

    UNDER THE WRITER LOCK, AND RE-CHECKED UNDERNEATH IT. `mark_stage` is a
    read-modify-write of `session.json` with no lock of its own, and the
    survey that produced this verdict ran over every session in the root
    before anything was written -- so a builder can take the world in
    between, and the write that would be lost is that builder's own. The
    lock closes the window, and the second `assess` is what makes taking it
    worth anything.
    """
    if verdict.stage == AREAS_STAGE:
        return _retire_areas(store, verdict, max_attempts)
    if verdict.stage == REGATE_STAGE:
        return _retire_regate(store, verdict, max_attempts)
    out = {"world_id": verdict.world_id, "session_id": verdict.session_id,
           "retired": verdict.stage, "attempts": verdict.attempts}
    try:
        store.acquire_writer_lock(verdict.world_id)
    except Exception as exc:  # noqa: BLE001 -- reported, never forced
        out.update({"retired": None, "reason": f"{type(exc).__name__}: {exc}"})
        return out
    try:
        again = assess(store, verdict.world_id, verdict.session_id,
                       max_attempts=max_attempts)
        if not again.exhausted:
            out.update({"retired": None, "reason": f"no longer {verdict.code}: "
                                                   f"{again.code}"})
            return out
        detail = (
            f"scripts/world_finish_pending.py started this stage "
            f"{verdict.attempts} times (bound {max_attempts}) and something "
            "ended it on its own every time; not retrying. To try again by "
            "hand: scripts/world_surface.py --force, then "
            "scripts/world_appearance.py -- a saved world is the surface AND "
            "the shading on it, and the surface alone is a grey mesh."
        )
        WorldBuilderEngine(store).mark_stage(
            verdict.world_id, verdict.session_id, verdict.stage,
            state=STAGE_STATE_FAILED, detail=detail,
        )
    finally:
        store.release_writer_lock(verdict.world_id)
    return out


def _forgive_on_stop(stop_request, forgive):
    """Arrange for `forgive` to run the instant a stop is ASKED FOR.

    WRAPPING THE INSTANCE, and that is the point rather than a shortcut.
    `StopRequest._watch_stdin` and `_handle_signal` both reach the flag
    through `self.request(...)`, so shadowing it on the instance puts
    `forgive` on the watcher thread, milliseconds after the pipe closes.

    Doing it at any later point does not work at all. The supervisor's grace
    ends in `terminate_tree`, and a stage inside a depth pass will not reach
    a `should_stop` checkpoint within it -- so there is no `finally`, no
    `atexit` and no return value. Whatever is going to be written down about
    this attempt has to be written while the process is still alive, in the
    five seconds it has left.

    Returns a callable that puts the original method back, and forgives at
    most once however many times a stop is asked for.
    """
    original = stop_request.request
    done = threading.Event()

    def request(level, source):
        original(level, source)
        if not done.is_set():
            done.set()
            forgive(source)

    stop_request.request = request

    def disarm():
        stop_request.request = original
        return done.is_set()

    return disarm


def finish(
    store: WorldStore,
    verdict: Verdict,
    *,
    appearance: bool,
    prune_depth_work: bool,
    stop_request,
    max_forgiven: int = DEFAULT_MAX_FORGIVEN,
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
    if verdict.stage == AREAS_STAGE:
        return finish_areas(store, verdict, appearance=appearance,
                            prune_depth_work=prune_depth_work,
                            stop_request=stop_request, max_forgiven=max_forgiven)
    if verdict.stage == REGATE_STAGE:
        return finish_regate(store, verdict, appearance=appearance,
                             prune_depth_work=prune_depth_work,
                             stop_request=stop_request, max_forgiven=max_forgiven)
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

    disarm = None
    try:
        if _refinish_under_the_lock(store, verdict, report):
            return report
        try:
            report["attempts"] = record_attempt(
                store, verdict.world_id, verdict.session_id,
                detail=f"finishing the {verdict.stage} stage",
            )
        except Exception as exc:  # noqa: BLE001
            # AN UNCOUNTED ATTEMPT IS AN UNBOUNDED ONE. A read-only or full
            # disk made this raise before any work began, so nothing was
            # counted, the bound never advanced, and every boot started the
            # same six-minute stage again -- the loop the bound exists to
            # prevent, arriving through the bound's own front door. A root
            # that cannot be written to has nowhere to put a surface either.
            report.update({
                "finished": False,
                "reason": f"the attempt could not be counted, so it was not "
                          f"started: {type(exc).__name__}: {exc}",
            })
            return report
        disarm = _forgive_on_stop(
            stop_request,
            lambda source: forgive_attempt(
                store, verdict.world_id, verdict.session_id,
                detail=f"stopped ({source})",
                max_forgiven=max_forgiven,
            ),
        )
        # THE WARM, AFTER THE ATTEMPT IS COUNTED. It is the native imports and
        # the CUDA attach -- exactly the class of work that hung on 2026-09-22
        # -- and it used to run before `record_attempt`, so a hang in it was a
        # stall the Tower could kill but the ledger never counted: retried
        # for ever, never retired (review, 2026-09-23). Since
        # `tower/stdin_stop.py` it no longer has to precede the stop watcher.
        prewarm_world_builder()
        report["stages"] = final_surface_stages(
            store, verdict.world_id, verdict.session_id,
            # Guaranteed by `assess`: a session is not owed unless its
            # finalization recorded a SOLVED final solve.
            solved=True,
            appearance=appearance,
            prune_depth_work=prune_depth_work,
            should_stop=stop_request.asked_for,
            stop_source=lambda: stop_request.source,
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
        if disarm is not None:
            # BELT AND BRACES, for the two windows the watcher thread cannot
            # cover: a stop that landed between `record_attempt` and the wrap
            # (microseconds, but it would spend an attempt on no work), and a
            # stage that HONOURED the stop and returned normally, which is the
            # happy version of the same event. `forgive_attempt` runs at most
            # once whichever path gets there.
            forgiven = disarm()
            if not forgiven and stop_request.asked:
                forgive_attempt(
                    store, verdict.world_id, verdict.session_id,
                    detail=f"stopped ({stop_request.source})",
                    max_forgiven=max_forgiven,
                )
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


def main(argv=None, *, stop_request=None) -> int:
    """`stop_request` is the stop channel, injectable so a test can BE the
    capture that opens mid-surface rather than approximate one."""
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
        "--max-forgiven", type=int, default=DEFAULT_MAX_FORGIVEN,
        help="how many times a stop request may give an attempt back before "
             "the attempts start counting anyway; without this bound a "
             "finisher that hangs and is killed at every boot is forgiven at "
             "every boot and --max-attempts is never reached",
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

    # THE SURVEY IS FREE AND THE WARM IS NOT, so the survey goes first.
    #
    # `tower/main.py` promises that "a Tower with nothing owed spawns a
    # process that reads some small JSON files, prints an empty report and
    # exits", and that is the common case by a wide margin: measured on the
    # root this was written against, 70 sessions, survey 0.04 s, **zero**
    # owed. Warming the native stack before finding that out cost 1.4 s and
    # some 650 MB of resident torch at every single Tower start, for nothing
    # -- a regression an adversarial reviewer caught in the first version of
    # this change.
    #
    # WHAT IS GIVEN UP, AND WHY IT IS AFFORDABLE. The watcher is armed after
    # the survey rather than before it, so a stop asked for during those
    # 0.04 s is not seen by this process. Nothing is at stake in that window:
    # no lock is taken, no attempt is recorded and nothing is written until
    # `finish` below, and the supervisor's `terminate_tree` reaps the process
    # either way. The ordering that IS load-bearing is unchanged and is the
    # reason these two lines are adjacent: the native stack is warmed while
    # no thread is parked in a blocking pipe read. See
    # `tower/native_prewarm.py` for the 95-minute deadlock that buys.
    has_work = any(v.owed or v.exhausted for v in verdicts)
    if not has_work:
        _emit(report, args.format)
        # Nothing THIS run can do -- but a world held by another writer may
        # still owe work once that writer lets go. See `EXIT_WAITING`.
        return EXIT_WAITING if any(v.code in WAITING_CODES for v in verdicts) else 0

    if stop_request is None:
        # The native warm is in `finish`, after the attempt is counted; the
        # watcher no longer has to wait for it (`tower/stdin_stop.py`).
        stop_request = StopRequest()
        stop_request.install(watch_stdin=args.stop_on_stdin_close)
    should_stop = stop_request.asked_for

    retire_failures = 0
    retire_waiting = 0
    # A WORKLIST, because retiring a re-gate can uncover more (review V8, M1b): once its
    # give-up is on the record, the same session is asked again, and the room or areas it
    # was hiding are finished -- or retired -- in THIS run rather than at the next start.
    # Bounded: a given-up re-gate is never `exhausted` again, so each session re-enters
    # the list at most once.
    position = {(v.world_id, v.session_id): i for i, v in enumerate(verdicts)}
    pending = [v for v in verdicts if v.exhausted]
    while pending:
        verdict = pending.pop(0)
        if should_stop():
            break
        try:
            retired = _retire(store, verdict, args.max_attempts)
            report["retired"].append(retired)
            reason = retired.get("reason") or ""
            if retired.get("retired") is None and not reason.startswith("no longer"):
                # Not retired and not moot. A live writer holding the lock is
                # WAITING -- try again later; anything else went wrong.
                if "WorldLockedError" in reason:
                    retire_waiting += 1
                else:
                    retire_failures += 1
        except Exception as exc:  # noqa: BLE001 -- a record is not worth an exit
            retire_failures += 1
            logger.warning(
                "[Tower][WorldBuilder] could not retire %s/%s: %s",
                verdict.world_id, verdict.session_id, exc,
            )
            continue
        if verdict.stage == REGATE_STAGE and retired.get("retired") == REGATE_STAGE:
            try:
                again = assess(store, verdict.world_id, verdict.session_id,
                               max_attempts=args.max_attempts)
            except Exception as exc:  # noqa: BLE001 -- one session, as in `survey`
                logger.warning("[Tower][WorldBuilder] could not assess %s/%s: %s",
                               verdict.world_id, verdict.session_id, exc)
                continue
            verdicts[position[(verdict.world_id, verdict.session_id)]] = again
            if again.exhausted and again.stage != REGATE_STAGE:
                pending.append(again)
    if any(r.get("retired") == REGATE_STAGE for r in report["retired"]):
        report["owed"] = [v.as_dict() for v in verdicts if v.owed]
        report["skipped"] = collections.Counter(v.code for v in verdicts if not v.owed)

    failures = retire_failures
    done = 0
    # Worlds this run found busy -- another writer holding the lock, or an owner's
    # re-finish -- whose other sessions would only meet the same answer, so they are not
    # picked again in this run (review V8, M1a).
    held = set()
    # ONE AT A TIME, and the stop checked between each. A run asked to stop
    # between two worlds stops there rather than starting a second
    # six-minute stage nobody is waiting for.
    for verdict in verdicts:
        if not verdict.owed or verdict.world_id in held:
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
            stop_request=stop_request,
            max_forgiven=args.max_forgiven,
        )
        report["finished"].append(outcome)
        if outcome.get("waiting"):
            # Refused before it began (a re-gate whose solve went, a lock another writer
            # took): no work was done, so it does not spend the run's budget, and the run
            # goes on to the next owed world. It used to count, and with another world
            # owed the run exited EXIT_MORE_OWED -- a respawn with no backoff that picked
            # the same refused world first, every time, and starved the rest (review V8,
            # M1a). The Tower asks again after its backoff (EXIT_WAITING).
            retire_waiting += 1
            if outcome.get("world_busy"):
                held.add(verdict.world_id)
            continue
        done += 1
        if not outcome.get("finished"):
            failures += 1

    _emit(report, args.format)
    if failures:
        return 1
    if "bounded_at" in report:
        return EXIT_MORE_OWED
    if retire_waiting or any(v.code in WAITING_CODES for v in verdicts):
        # Somebody else is writing a world this run would otherwise have
        # looked at -- a hand-run tool, another Tower on the same root. Not
        # nothing-to-do: the Tower tries again later, after a backoff.
        return EXIT_WAITING
    return 0


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
