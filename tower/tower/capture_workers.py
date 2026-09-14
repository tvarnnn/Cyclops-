"""Run worker processes for the lifetime of a capture, and reap them.

This is the piece whose absence made the first physical World Builder
test (2026-08-24) require a second terminal, a directory listing, and a
copied UUID. The Tower recorded ten captures that evening and a human
attached a follower to one of them by hand. The other nine were never
read by anything, so the wearer walked for five minutes with the camera
reporting LIVE and the world frozen at the figures from the first two.

It happened a second time, to a different cartridge, on 2026-08-26: a
real Ray-Ban walk produced 2,203 frames and 64 remembered observations,
and every one of them existed because a human found the capture
directory and started a producer against it in another terminal. The
supervisor could not have done it, because it ran exactly ONE spec and
that slot was taken.

**One spec was never the design; it was the number of specs that
existed.** A supervisor is now given a LIST, each entry named, each
optionally GATED by a predicate the wiring point supplies. The list is
still argv and strings.

**This module is deliberately cartridge-blind.** It knows how to run an
argv when a capture opens, how to stop one on request, and how to reap.
It does not know what any worker computes, and it names no cartridge --
which is what lets `test_the_capture_worker_supervisor_is_cartridge_blind`
stay green without an exemption. The argv and the gate both come from
`main.py`, the wiring point, as plain strings and a plain callable.

**The web process still does not build, and does not remember.** It
supervises children that do. That separation is the reason a rebuild can
take seconds without the frame path noticing, and it is the architecture
decision `docs/agent-handoffs/WORLD-BUILDER.md` section 1 exists to
protect.
"""

import logging
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field

# Cartridge-blind, like this module: it knows how to start an interpreter
# as ONE process and how to stop a process tree, and nothing about what
# the process computes.
from tower.process_ownership import (
    assign_to_job,
    interpreter_environment,
    interpreter_executable,
    terminate_tree,
)

logger = logging.getLogger(__name__)

# How long a worker gets to finish on its own at shutdown before it is
# terminated. A follower whose capture has closed is usually inside its
# final build, and that build is what releases the writer lock -- kill it
# and the result channel correctly, but needlessly, reports the world
# `failed` with a pid that no longer exists.
DEFAULT_GRACE_SECONDS = 10.0

# How long a TERMINATED worker gets to actually die before it is treated
# as un-killable. Distinct from the grace above, which is how long a
# worker gets to finish on its own BEFORE being terminated: this one is
# waiting on a process that has already been shot, and on Windows that
# wait is real -- `TerminateProcess` is asynchronous and a `poll()`
# straight afterwards routinely still returns None.
TERMINATE_TIMEOUT_SECONDS = 2.0

PLACEHOLDER_CAPTURE_DIR = "{capture_dir}"
PLACEHOLDER_CAPTURE_ID = "{capture_id}"
# Substituted with one of the two values below. A worker whose argv does
# not mention it is unaffected; a worker whose argv does gets told
# something it cannot work out for itself.
PLACEHOLDER_ATTACH_MODE = "{attach_mode}"

# A worker started at the moment the capture opened. Every frame the
# capture will ever hold is still ahead of it.
ATTACH_MODE_FROM_START = "from-start"
# A worker started against a capture that was ALREADY RUNNING. Whatever
# is already in the journal happened before this worker was asked for,
# and a worker that decides on its own to go back and read it is making a
# consent decision it has no standing to make.
ATTACH_MODE_FROM_NOW = "from-now"


def _ask_to_stop(process, *, send_signal: bool = True) -> bool:
    """Ask a worker to stop, without killing it. Returns whether it was asked.

    TWO HALVES, SEPARABLE SINCE 2026-09-06. `send_signal=False` closes
    stdin and does nothing else: that is the SOFT request `request_stop`
    makes on behalf of a cartridge's Stop button, meaning "you are no
    longer wanted for new frames", which a builder answers by ending its
    session and writing its final build on its own time. The default sends
    both halves, and that is the HARD request `shutdown` makes, meaning
    "wrap up now". The world builder tells them apart (stdin EOF versus
    `SIGBREAK`); the object memory producer treats both as stop. Neither
    half is a kill.

    THE GRACE WINDOW WAS WAITING ON A REQUEST NOBODY HAD MADE.

    `_stop_worker` used to `wait(grace)` first and terminate second. That
    is the right shape, but nothing in between ever told the worker
    anything, so the wait measured the full grace every single time and
    the process was shot at the end of it regardless. The Object Memory
    producer's `finally: engine.release()` -- the only code that closes
    open sightings and writes the ones that matured -- therefore never
    ran when a wearer pressed Stop. The fix is not a longer wait. It is
    to ask.

    TWO CHANNELS, BECAUSE ONE OF THEM DOES NOT WORK WHERE THIS RUNS.

    The obvious channel is a signal, and on POSIX `terminate()` is
    `SIGTERM` and that is the whole story. On Windows `terminate()` is
    `TerminateProcess`, which is not a signal and cannot be caught, so
    the only route to a catchable stop is a console control event --
    which is why `_start` passes `CREATE_NEW_PROCESS_GROUP`, and
    `CTRL_BREAK_EVENT` rather than `CTRL_C_EVENT` because a new process
    group starts with Ctrl-C disabled.

    A SIGNAL IS NOT ENOUGH, AND THE REASON IS NOT THE ONE FIRST WRITTEN
    HERE.

    This docstring used to claim that a console control event is never
    delivered under a pseudoconsole and that the child hears nothing. **A
    reviewer measured the opposite and was right.** Under
    `GetConsoleWindow() == 0` the event IS delivered, and a child that
    installed no handler dies of it with `STATUS_CONTROL_C_EXIT` and no
    unwinding at all -- which is the more dangerous half of the truth,
    because it means a signal sent to the wrong worker destroys exactly
    the grace it was meant to protect. That is what
    `_Worker.handles_stop_request` gates, and it is why the gate is not
    optional.

    The stdin channel stays, and is still the first one tried, for
    reasons that survive the correction: a pipe needs no console at all,
    so it works where an event genuinely cannot be delivered (a detached
    service, a job object, a host that revoked the group); it is
    unambiguous, where a control event's disposition depends on what the
    child did with its handlers; and closing it costs nothing when the
    child is already gone. Two independent channels for a request that
    must not be missed is the shape, and neither is required to succeed.

    So the FIRST channel is closing the child's stdin. The supervisor
    holds the write end for any spec that asked for it
    (`WorkerSpec.stop_via_stdin`), closing it is an EOF the child's
    reader thread cannot miss, and a pipe needs no console, no signal
    disposition and no permission. It also fixes something adjacent for
    free: a producer orphaned by a Tower that died gets its EOF from the
    operating system rather than polling for its full fifteen-minute idle
    bound.

    Both are attempted. Neither is required to succeed, and each is
    contained separately -- a signal that raises must not stop the pipe
    from being closed. Contained overall as "not asked" so the caller
    falls through to `terminate()` rather than leaving a worker alive
    because an optional courtesy failed.
    """
    # A process that has already exited is not asked, and this is not
    # tidiness. `os.kill(pid, CTRL_BREAK_EVENT)` treats the pid as a
    # process GROUP id, Windows recycles pids aggressively, and a reaped
    # worker's pid can belong to something else by the time a shutdown
    # gets here. Signalling a stranger's process group is the one failure
    # in this file that would not look like a failure.
    try:
        if process.poll() is not None:
            return False
    except Exception:  # noqa: BLE001
        # A process object that cannot say. Fall through and try: the
        # caller terminates either way, and refusing to ask on a
        # can't-tell would silently drop the flush.
        pass

    asked = False

    stdin = getattr(process, "stdin", None)
    if stdin is not None:
        try:
            stdin.close()
            asked = True
        except Exception:
            logger.debug(
                "[Tower][Worker] could not close stdin for pid %s",
                getattr(process, "pid", "?"),
                exc_info=True,
            )

    if not send_signal:
        return asked

    try:
        if os.name == "nt":
            os.kill(process.pid, signal.CTRL_BREAK_EVENT)
        else:
            process.terminate()
        asked = True
    except Exception:
        logger.debug(
            "[Tower][Worker] could not signal pid %s; it will be terminated "
            "instead",
            getattr(process, "pid", "?"),
            exc_info=True,
        )

    return asked


@dataclass(frozen=True)
class WorkerSpec:
    """What to run, once per capture lineage.

    `argv` may contain `{capture_dir}`, `{capture_id}` and
    `{attach_mode}`, substituted per capture. Substitution is positional
    over the argv LIST, never over a joined string: a capture root chosen
    by an operator can contain spaces, and building a command line by
    concatenation is how that becomes a quoting bug on Windows.
    """

    argv: tuple[str, ...]
    cwd: str | None = None
    name: str = "capture-worker"
    # Whether the child writes into this process's stdout and stderr.
    #
    # True by default, and that default is the point. A worker sent to
    # DEVNULL is a worker whose failure is invisible: the operator sees a
    # world that is not growing and has nothing to read. Interleaving is
    # the lesser problem, and small in practice -- the world builder logs
    # once at session start, once per rebuild, and once at the end.
    inherit_output: bool = True
    # Whether this worker is handed a stdin pipe the supervisor keeps,
    # so that closing it is a stop request the child can act on.
    #
    # False by default, and that default is not timidity: a child that
    # does not WATCH its stdin gains nothing from being given a pipe, and
    # a child that inherits a stdin already at EOF -- a Tower started with
    # `< nul`, or as a service -- would read that as an immediate stop.
    # A spec sets this only alongside the argv flag that makes its child
    # watch, so the two halves of the agreement are written in one place.
    stop_via_stdin: bool = False
    # How long this worker needs to wrap up once ASKED, at shutdown or
    # detach, before it is terminated. None means "whatever the caller
    # passes", which is what every spec meant before this existed.
    #
    # THE SPEC'S VALUE REPLACES THE CALLER'S WHEN SET. The rule is not
    # max() and not min(): the spec is the one place that knows what its
    # worker does when asked -- the world builder ends its session and
    # writes one final build, which is 30 s on a long walk, and the
    # object memory producer flushes a queue in well under 3 -- while the
    # caller (`lifespan`, a `CartridgeSession`) only knows a default that
    # has to suit every worker at once. A 10 s default against a 30 s
    # build is exactly how the builder came to be shot mid-finalization
    # and its world reported `failed` (2026-09-06). The grace is the
    # BOUND, not the wait: a worker that finishes sooner is reaped sooner.
    stop_grace_seconds: float | None = None
    # Asked at every capture open: may this spec run right now?
    #
    # None means "always", which is what every spec meant before gates
    # existed. A predicate rather than a flag because the answer CHANGES
    # while the Tower is up -- a wearer starting or pausing a cartridge
    # between two captures must not need a restart to be obeyed -- and a
    # predicate rather than a state machine in here because whose
    # question it is belongs at the wiring point, not in the supervisor.
    gate: object = None


@dataclass
class _Worker:
    capture_id: str
    process: object
    argv: tuple[str, ...]
    started_at: float
    # Every capture id this worker is following, in order. The follower
    # chains into successors on its own, so one worker legitimately owns
    # a whole lineage.
    lineage: list[str] = field(default_factory=list)
    # Whether this child understands being ASKED to stop, rather than
    # only being terminated. Copied off its spec at spawn, because
    # `_stop_worker` has a worker and not a spec.
    #
    # It gates the signal as well as the pipe, and that is the point. A
    # `CTRL_BREAK_EVENT` reaches a child that has installed no handler as
    # a request to die immediately -- so sending one to the world
    # builder, which installs none, would have converted its ten-second
    # grace into an instant kill and made the result channel report a
    # world `failed` for a build that was seconds from finishing. That is
    # exactly what the grace exists to prevent, and it was nearly
    # destroyed by a change meant to make a DIFFERENT worker's grace
    # useful.
    handles_stop_request: bool = False
    # Set by `request_stop`. An asked-to-stop worker is still ALIVE -- it
    # is finishing its final build -- but it will not follow a successor,
    # and chaining one into it loses the rest of the walk. See
    # `_attach_to_registry`.
    stop_requested: bool = False
    # The Job Object this worker was placed in, or None off Windows and on
    # a host that refused one. HELD HERE ON PURPOSE: `KILL_ON_JOB_CLOSE`
    # fires when the last handle closes, so the handle living on this
    # record is what makes "the Tower died" and "the tree died" the same
    # event. See `process_ownership.JobHandle`.
    job: object = None

    def is_alive(self) -> bool:
        return self.process.poll() is None


class _SpecRegistry:
    """One spec's workers, and which lineage each of them owns.

    Per spec, not shared, and that separation is the whole reason two
    cartridges can follow one capture safely. A shared table keyed by
    lineage alone would let the first spec's worker answer "yes, this
    lineage is followed" on behalf of a second spec whose worker had
    died -- and the second cartridge would then silently record nothing
    for the rest of the walk, which is the exact failure the supervisor
    exists to prevent.
    """

    __slots__ = ("spec", "workers", "roots")

    def __init__(self, spec: WorkerSpec) -> None:
        self.spec = spec
        # lineage root capture id -> worker
        self.workers: dict[str, _Worker] = {}
        # any capture id -> the lineage root that owns it
        self.roots: dict[str, str] = {}

    def owner_of(self, capture_id: str) -> _Worker | None:
        root = self.roots.get(capture_id)
        return self.workers.get(root) if root is not None else None

    def forget(self, root: str) -> None:
        worker = self.workers.pop(root, None)
        if worker is None:
            return
        for capture_id in worker.lineage:
            if self.roots.get(capture_id) == root:
                del self.roots[capture_id]


class CaptureWorkerSupervisor:
    """One worker per capture LINEAGE, per spec.

    The lineage distinction is the older half of the design.
    `CaptureFollower` already follows a capture across a reconnect:
    seeing a capture close by disconnect it waits out
    `RESUME_GRACE_SECONDS` for a successor whose manifest names it, and
    continues into that one. So a successor capture already has a reader.
    Starting a second worker for it would put two followers, two mapping
    sessions and two writer locks on one walk -- and the result channel,
    which prefers "a LIVE world if one exists", would then have two live
    worlds to choose between and would pick by `updated_at`.

    Lineage is decided by the WRITER and written into the manifest
    (`CaptureRecorder.resumable_capture`). This class only reads the
    `continues` value it is handed; it never guesses which capture
    continues which.

    The per-spec half is newer. Specs are addressed BY NAME -- `attach`
    and `detach` take one -- so names must be unique, and the constructor
    refuses a duplicate rather than letting one cartridge's Pause stop
    another cartridge's producer.

    SERIALISED. `capture_opened` runs on the event loop, from the
    connection that just received `stream_start`; `attach` and `detach`
    run in FastAPI's threadpool, from a session control request. Those
    are genuinely concurrent, and unserialised they raced: a Start
    pressed as a capture opened ran the "is anything already following
    this lineage" check twice, both times seeing nothing, and spawned two
    producers on one capture. The second overwrote the first in the
    registry, so the orphan was invisible to `reap`, `detach`,
    `shutdown` and `/health` -- and two producers appending to one JSONL
    store lose each other's writes, because `update_sighting` rewrites
    the whole file. A reviewer reproduced both the lost write and a
    duplicate record with a colliding `observation_id`.

    Re-entrant, because `capture_opened` and `attach` both call `reap`.
    """

    def __init__(self, specs=None, *, spawn=None, clock=time.monotonic):
        # A bare spec is still a legal argument. Every call site that
        # existed before this class grew a list passes one, and the
        # single-spec Tower is not a legacy shape to be migrated away
        # from -- it is what a Tower running one cartridge looks like.
        if specs is None:
            specs = ()
        elif isinstance(specs, WorkerSpec):
            specs = (specs,)
        else:
            specs = tuple(specs)

        names = [spec.name for spec in specs]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                "worker spec names must be unique; `attach` and `detach` "
                f"address a spec by name. Duplicated: {duplicates}"
            )

        self._spawn = spawn if spawn is not None else subprocess.Popen
        self._clock = clock
        # Re-entrant: `capture_opened` and `attach` both call `reap`, and
        # `shutdown` and `detach` both call `_stop_worker`.
        self._lock = threading.RLock()
        self._registries: dict[str, _SpecRegistry] = {
            spec.name: _SpecRegistry(spec) for spec in specs
        }

    @property
    def enabled(self) -> bool:
        """Whether anything will be started at all.

        Reported rather than inferred: "no worker is configured" and "a
        worker is configured and failed" call for opposite responses from
        whoever is reading /health.
        """
        return bool(self._registries)

    def spec_for(self, name: str) -> WorkerSpec | None:
        """The spec registered under a name, or None.

        Public because a wiring test has to be able to assert what the
        Tower will actually run -- which flags, which root, which
        cadence -- and the alternative it used before this existed was
        reaching into a private attribute. That worked while there was
        exactly one spec and stopped compiling the moment there were two,
        which is the usual way a private attribute charges for its use.
        """
        registry = self._registries.get(name)
        return registry.spec if registry is not None else None

    def worker_names(self) -> tuple[str, ...]:
        """Every spec this supervisor could ever start, gated or not.

        The gate decides whether a worker runs; this says whether the
        Tower knows how to run one at all. `/health` needs both, and a
        reader that only had the running list could not tell "configured
        and paused" from "not configured".
        """
        return tuple(self._registries)

    # -- lifecycle ----------------------------------------------------

    def capture_opened(self, capture_id: str, capture_dir, *, continues=None) -> None:
        """A recording has started. Attach every gated-open spec to it."""
        if not self._registries:
            return

        with self._lock:
            self.reap()

            for registry in self._registries.values():
                if not self._gate_open(registry.spec):
                    continue
                self._attach_to_registry(
                    registry,
                    capture_id,
                    capture_dir,
                    continues=continues,
                    attach_mode=ATTACH_MODE_FROM_START,
                )

    def attach(self, name: str, capture_id: str, capture_dir) -> bool:
        """Start ONE named spec against a capture that is already open.

        The gate is deliberately not consulted. `capture_opened` asks
        "should this be running", which is a standing question about
        configuration; this is a direct instruction from whoever owns the
        gate, arriving at the moment they changed their mind. Asking the
        gate here would make the instruction depend on the caller having
        already published its own new state, which is a race with no
        upside.

        Returns whether a process was started. False covers both "no such
        spec" and "one is already following this lineage" -- neither is
        an error, and an instruction repeated twice must not put two
        producers on one store.
        """
        with self._lock:
            registry = self._registries.get(name)
            if registry is None:
                logger.warning(
                    "[Tower][Worker] asked to attach unknown worker %r; "
                    "nothing will follow capture %s for it",
                    name,
                    capture_id,
                )
                return False
            self.reap()
            return self._attach_to_registry(
                registry,
                capture_id,
                capture_dir,
                continues=None,
                attach_mode=ATTACH_MODE_FROM_NOW,
            )

    def detach(self, name: str, grace_seconds: float = DEFAULT_GRACE_SECONDS) -> int:
        """Stop every worker belonging to ONE spec. Returns how many.

        `grace_seconds` is how long a worker gets to exit on its own
        first. The caller chooses it and the caller should usually choose
        ZERO: nothing here SIGNALS a worker, and a follower tailing a
        journal that is still being written has no reason to stop. Waiting
        on one measures the full grace every time and then terminates it
        anyway. `shutdown` is the case where waiting is right, because
        there the capture has closed and the follower really will finish.

        Detaching does NOT disable the spec. The gate is the one source
        of truth for whether a spec should be running; a latch in here
        would be a second one, and the two would disagree the first time
        either changed.
        """
        with self._lock:
            registry = self._registries.get(name)
            if registry is None:
                return 0
            pending = [
                (registry, root, worker)
                for root, worker in list(registry.workers.items())
            ]
            if not pending:
                return 0

            # Concurrently, for the same reason `shutdown` is, and it
            # matters MORE here: `shutdown` runs once at teardown, while
            # this runs on every Pause and every Stop, and
            # `capture_closed` takes this same lock ON THE EVENT LOOP.
            # Serially this held the lock for the SUM of every worker's
            # grace and terminate window; measured, a three-worker detach
            # blocked a concurrent `capture_closed` for 5.60 s.
            #
            # The caller usually passes grace 0 -- see the docstring above
            # -- so the common case was already cheap and this is about
            # the case that is not.
            stopped_flags = self._stop_all(pending, grace_seconds)

            stopped = 0
            for (_, root, _), gone in zip(pending, stopped_flags):
                if not gone:
                    # Still alive. KEEPING it in the registry is the
                    # point: forgetting a worker that could not be
                    # terminated makes it an orphan nothing can see --
                    # not `status`, not `/health`, not the next
                    # `shutdown` -- and three Stop/Start cycles then
                    # leave four producers running with the supervisor
                    # aware of one.
                    continue
                registry.forget(root)
                stopped += 1
            return stopped

    def request_stop(self, name: str) -> int:
        """ASK every worker of one spec to stop. Returns how many were asked.

        The soft channel, and nothing but the soft channel: each worker
        that opted in (`stop_via_stdin`) has its stdin closed, and this
        returns at once. No signal, no wait, no terminate, and -- the
        part that matters -- NO FORGETTING. The worker stays in its
        registry until it exits and `reap` notices, so `status`,
        `/health` and `following` keep telling the truth about a builder
        that is spending the next thirty seconds on its final build.

        This is what a World Builder Stop means. The wearer left the
        workspace; the walk's geometry is still worth finishing, and a
        builder that is mid-finalization must be allowed to. A Stop that
        terminated would turn every "I looked at another tab" into an
        interrupted world -- which is the state the 2026-09-06 walk
        ended in for a different reason, and it was bad enough once.
        `shutdown()` remains the hard path.

        A worker that never opted in is not asked and not counted: it has
        no reader on its stdin, so an EOF would tell it nothing. It is
        left running and the log says so, because the only thing that
        will stop it is the next shutdown.
        """
        with self._lock:
            registry = self._registries.get(name)
            if registry is None:
                return 0
            self._reap_locked()
            asked = 0
            for worker in registry.workers.values():
                if not worker.handles_stop_request:
                    logger.warning(
                        "[Tower][Worker] %s worker pid %s for capture %s cannot "
                        "be asked to stop (its spec has no stdin channel); it "
                        "runs on until it finishes or the Tower shuts down",
                        registry.spec.name,
                        worker.process.pid,
                        worker.capture_id,
                    )
                    continue
                if _ask_to_stop(worker.process, send_signal=False):
                    asked += 1
                    worker.stop_requested = True
                    logger.info(
                        "[Tower][Worker] asked %s worker pid %s for capture %s "
                        "to stop (stdin closed); it stays registered until it "
                        "exits on its own",
                        registry.spec.name,
                        worker.process.pid,
                        worker.capture_id,
                    )
            return asked

    def capture_closed(self, capture_id: str) -> None:
        """A recording has ended.

        Workers are deliberately NOT stopped here. A capture that ended
        by disconnect may still get a successor within the resume grace
        window, and the follower is the thing that waits for it. Killing
        it on `stream_stop` would turn every WiFi hiccup back into the
        end of the walk.
        """
        if not self._registries:
            return
        for registry in self._registries.values():
            worker = registry.owner_of(capture_id)
            if worker is None:
                continue
            logger.info(
                "[Tower][Worker] capture %s closed; %s worker pid %s continues "
                "until it observes completion",
                capture_id,
                registry.spec.name,
                worker.process.pid,
            )
        self.reap()

    def reap(self) -> None:
        """Notice workers that have exited, and say how they went."""
        with self._lock:
            self._reap_locked()

    def _reap_locked(self) -> None:
        for registry in self._registries.values():
            for root, worker in list(registry.workers.items()):
                code = worker.process.poll()
                if code is None:
                    continue
                elapsed = self._clock() - worker.started_at
                if code == 0:
                    logger.info(
                        "[Tower][Worker] %s worker pid %s for capture %s "
                        "finished after %.1fs (lineage: %s)",
                        registry.spec.name,
                        worker.process.pid,
                        worker.capture_id,
                        elapsed,
                        ", ".join(worker.lineage),
                    )
                else:
                    # Loud, and named against its capture. A follower that
                    # dies before acquiring the writer lock leaves the
                    # result channel with nothing to read, and iOS renders
                    # that as "no world" -- indistinguishable from "you
                    # have not walked yet".
                    logger.warning(
                        "[Tower][Worker] %s worker pid %s for capture %s "
                        "EXITED %s after %.1fs; nothing is following this "
                        "capture for it. argv: %s",
                        registry.spec.name,
                        worker.process.pid,
                        worker.capture_id,
                        code,
                        elapsed,
                        " ".join(worker.argv),
                    )
                registry.forget(root)

    def shutdown(self, grace_seconds: float = DEFAULT_GRACE_SECONDS) -> None:
        """Let every worker finish, then insist. Every worker at once.

        The waits run CONCURRENTLY, so teardown costs the slowest worker
        rather than the sum of all of them. Each one still gets its own
        full grace window and the same terminate-then-confirm sequence, so
        this changes the cost and nothing else -- no grace is shortened,
        no worker is signalled earlier than it was.

        Serially this was `N * (grace + TERMINATE_TIMEOUT_SECONDS)`, and
        the whole of it ran holding `self._lock`. MEASURED with two
        stubborn workers: shutdown took 24.00 s, a concurrent `status()`
        -- which is what `/health` calls -- blocked for 23.95 s of it, and
        two workers were still alive at the end. The lock is still held
        here, because the registries must not move underneath a teardown;
        what shrinks is how long that is true for.

        `_stop_worker` is safe to run from several threads at once: it
        touches only its own `worker.process` and the logger. Nothing
        mutates a registry until every wait has returned, on this thread.
        """
        with self._lock:
            pending = [
                (registry, root, worker)
                for registry in self._registries.values()
                for root, worker in list(registry.workers.items())
            ]
            if not pending:
                return

            stopped = self._stop_all(pending, grace_seconds)

            # `zip` cannot misalign: `pool.map` yields in ARGUMENT order,
            # not completion order, and the serial fallback preserves it
            # by construction.
            for (registry, root, _), gone in zip(pending, stopped):
                if gone:
                    registry.forget(root)

    def _stop_all(self, pending, grace_seconds) -> list[bool]:
        """Stop every pending worker, concurrently if a pool can be had.

        THE FALLBACK IS NOT DEFENSIVE NOISE. `ThreadPoolExecutor` raises
        `RuntimeError` when the interpreter is shutting down, and again
        when the OS refuses a thread -- and it raises BEFORE any worker is
        stopped, so the concurrent path is all-or-nothing. Without this,
        one raise stopped ZERO workers and left every registry entry
        standing: orphan producers the supervisor no longer knows about,
        which is the precise failure this module exists to prevent.

        Both halves are reachable and neither is exotic. Shutdown runs at
        interpreter teardown by definition, and this Tower has a measured
        thread leak of ~19 threads per live-session cycle -- so the state
        in which `shutdown()` most needs to work is the state in which a
        new pool is most likely to be refused.

        It was invisible on a one-cartridge Tower, because a single worker
        never enters the pool path at all, and total on a two-cartridge
        one.
        """
        # Each worker gets ITS SPEC'S grace when the spec names one, and
        # the caller's otherwise. See `WorkerSpec.stop_grace_seconds` for
        # why the spec's replaces rather than caps or floors the caller's.
        def stop(item):
            registry, _, worker = item
            return self._stop_worker(
                worker, _grace_for(registry.spec, grace_seconds)
            )

        if len(pending) == 1:
            return [stop(pending[0])]

        from concurrent.futures import ThreadPoolExecutor

        try:
            with ThreadPoolExecutor(
                # Capped: `pending` grows with captures, and a teardown
                # that cannot get 3 threads will not get 300.
                max_workers=min(len(pending), 8),
                thread_name_prefix="capture-shutdown",
            ) as pool:
                return list(pool.map(stop, pending))
        except RuntimeError:
            logger.warning(
                "[Tower][Worker] could not start a shutdown pool; stopping "
                "%s worker(s) one at a time instead. This is slower -- it "
                "costs the SUM of every grace window rather than the "
                "longest -- but every worker is still stopped and every "
                "one that dies is still forgotten",
                len(pending),
            )
            return [stop(item) for item in pending]

    # -- reporting ----------------------------------------------------

    def status(self) -> list[dict]:
        """One row per live worker. Safe to call from /health."""
        with self._lock:
            self._reap_locked()
            return self._status_locked()

    def _status_locked(self) -> list[dict]:
        return [
            {
                # Which spec this row belongs to. Without it two rows for
                # one capture id are indistinguishable, and "a worker is
                # alive on this capture" stops being an answer to any
                # particular cartridge's question.
                "worker": registry.spec.name,
                "capture_id": worker.capture_id,
                "lineage": list(worker.lineage),
                "pid": worker.process.pid,
                "alive": True,
                "uptime_seconds": self._clock() - worker.started_at,
            }
            for registry in self._registries.values()
            for worker in registry.workers.values()
        ]

    def mark(self) -> float:
        """A reading of THIS supervisor's clock, for use with `following`.

        Exists so that nothing outside compares a `started_at` against a
        timestamp from somewhere else. That is not a hypothetical
        tidiness: this supervisor's clock defaults to `time.monotonic`
        and `CartridgeSession`'s to `time.time`, so the first version of
        `following(since=...)` compared an uptime against a Unix epoch,
        every worker looked older than every session, and a correct
        implementation reported nothing at all. A caller that takes its
        mark from here cannot make that mistake.

        Monotonic is also the right clock for the question. "Did this
        start after that" must survive a clock adjustment, and a wall
        clock stepping backwards mid-walk would make a live producer look
        like a leftover.
        """
        return self._clock()

    def following(self, name: str, *, since: float | None = None) -> list[str]:
        """The capture ids one named spec currently has a live worker on.

        `since` narrows the answer to workers STARTED at or after a
        moment, and it exists because this supervisor is deliberately
        cartridge-blind and therefore knows nothing about sessions. A
        `CartridgeSession` asking "what is MY producer on" and a
        `/health` reader asking "what is running on this Tower" are
        different questions, and answering the first with the second is
        how a brand-new session that attached nothing came to render as
        recording: a previous session's un-killable worker is still in
        this registry, and it was being reported under the new session's
        id.

        Unfiltered is still the default, and still the honest answer to
        the second question. A worker nobody can kill must stay visible
        somewhere, or the one control a wearer has over being remembered
        fails open silently.
        """
        with self._lock:
            registry = self._registries.get(name)
            if registry is None:
                return []
            self._reap_locked()
            return [
                worker.capture_id
                for worker in registry.workers.values()
                if since is None or worker.started_at >= since
            ]

    # -- internals ----------------------------------------------------

    @staticmethod
    def _gate_open(spec: WorkerSpec) -> bool:
        """Whether this spec may run right now. A broken gate means no.

        Contained, and contained CLOSED. `capture_opened` runs on the
        connection that just received `stream_start`, so an exception
        escaping a gate would drop a recording -- and defaulting the
        other way would start a producer whose enablement could not be
        established, which is the direction that writes data nobody
        asked for.
        """
        gate = spec.gate
        if gate is None:
            return True
        try:
            return bool(gate())
        except Exception:
            logger.exception(
                "[Tower][Worker] the gate for worker %r raised; treating it "
                "as closed and starting nothing",
                spec.name,
            )
            return False

    def _attach_to_registry(
        self,
        registry: _SpecRegistry,
        capture_id: str,
        capture_dir,
        *,
        continues,
        attach_mode: str,
    ) -> bool:
        if continues is not None:
            root = registry.roots.get(continues)
            worker = registry.workers.get(root) if root is not None else None
            if worker is not None and worker.is_alive() and worker.stop_requested:
                # ALIVE, AND NOT FOLLOWING. This worker has been asked to
                # stop -- `session/stop`, which is what leaving the World
                # Builder screen sends -- and is spending its last seconds
                # on a final build. It will never walk into this capture.
                # Chaining the successor into it recorded a lineage nobody
                # served: a dress-rehearsal reviewer cut the link, left the
                # screen during the outage, came back, and watched 1,200
                # frames get recorded and built by NOBODY while the phone
                # showed the "success" sentence. So: fall through and start
                # a builder on this capture. It builds the second half as
                # its own world, which is a loss the wearer can see rather
                # than one they cannot.
                logger.warning(
                    "[Tower][Worker] capture %s continues %s, but the %s "
                    "worker pid %s following it has been asked to stop and "
                    "is finishing; starting a new builder on the successor "
                    "instead of chaining into a worker that will not follow",
                    capture_id, continues, registry.spec.name,
                    worker.process.pid,
                )
            elif worker is not None and worker.is_alive():
                # The existing follower will walk into this capture by
                # itself. Record the mapping so the NEXT successor --
                # which names this capture, not the one we spawned on --
                # is recognised too.
                registry.roots[capture_id] = root
                worker.lineage.append(capture_id)
                logger.info(
                    "[Tower][Worker] capture %s continues %s; %s worker pid %s "
                    "(following %s) will chain into it, not restarted",
                    capture_id,
                    continues,
                    registry.spec.name,
                    worker.process.pid,
                    root,
                )
                return False
            # Either this Tower never saw the predecessor (a restart
            # mid-walk), or this spec's worker has died. Both mean
            # nothing is reading this capture FOR THIS SPEC, so follow it.
            logger.info(
                "[Tower][Worker] capture %s continues %s but no live %s worker "
                "owns that lineage; starting one",
                capture_id,
                continues,
                registry.spec.name,
            )
        elif registry.owner_of(capture_id) is not None:
            # Already followed by this spec. Attaching again would put two
            # producers on one store.
            return False

        return self._start(registry, capture_id, capture_dir, attach_mode)

    def _start(
        self,
        registry: _SpecRegistry,
        capture_id: str,
        capture_dir,
        attach_mode: str,
    ) -> bool:
        spec = registry.spec
        argv = tuple(
            part.replace(PLACEHOLDER_CAPTURE_DIR, str(capture_dir))
            .replace(PLACEHOLDER_CAPTURE_ID, capture_id)
            .replace(PLACEHOLDER_ATTACH_MODE, attach_mode)
            for part in spec.argv
        )
        # ONE PROCESS, NOT A LAUNCHER PAIR. A spec written against
        # `sys.executable` -- which is every spec `main.py` builds -- would
        # on a Windows venv start a launcher that starts the interpreter,
        # and the pid held here, logged below and reported by `/health`
        # would be the launcher's. Terminating it kills the interpreter
        # (the launcher's own job does that) but every grandchild
        # survives, which is how a solve child outlived its builder on
        # 2026-09-06. The rewrite lives here rather than at the wiring
        # point so `main.py` stays a plain argv and no future spec has to
        # know why it needs this.
        env = None
        if argv and argv[0] == sys.executable:
            argv = (interpreter_executable(), *argv[1:])
            env = interpreter_environment()
        try:
            process = self._spawn(
                argv,
                cwd=spec.cwd,
                env=env,
                stdout=None if spec.inherit_output else subprocess.DEVNULL,
                stderr=None if spec.inherit_output else subprocess.DEVNULL,
                # See `_ask_to_stop`. The write end lives on the Popen and
                # closing it is the one stop request that needs no console.
                stdin=subprocess.PIPE if spec.stop_via_stdin else None,
                # A worker must not die because the operator pressed
                # Ctrl-C in the Tower's console: on Windows a console
                # Ctrl-C goes to the whole process group, and a follower
                # killed mid-build leaves a stale writer lock behind.
                creationflags=(
                    subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
                ),
            )
        except Exception:
            logger.exception(
                "[Tower][Worker] could not start the %s worker for capture %s; "
                "this capture will record frames but NOTHING will follow it "
                "for that worker. argv: %s",
                spec.name,
                capture_id,
                " ".join(argv),
            )
            return False

        registry.workers[capture_id] = _Worker(
            capture_id=capture_id,
            process=process,
            argv=argv,
            started_at=self._clock(),
            lineage=[capture_id],
            handles_stop_request=bool(spec.stop_via_stdin),
            # Immediately, so the window in which a grandchild could be
            # born outside the job is the width of one function return.
            # None off Windows, on a refused host, and for a stand-in
            # process in a test; `terminate_tree` copes with all three.
            job=assign_to_job(process),
        )
        registry.roots[capture_id] = capture_id
        logger.info(
            "[Tower][Worker] started %s pid %s for capture %s: %s",
            spec.name,
            process.pid,
            capture_id,
            " ".join(argv),
        )
        return True

    def _stop_worker(self, worker: _Worker, grace_seconds: float) -> bool:
        """Stop one worker. Returns whether it is actually gone.

        The return value is what stops an orphan. A `terminate()` that
        raises used to be logged and then forgotten anyway, which removed
        the worker from the registry while leaving the process running --
        invisible to `status`, to `/health` and to the next `shutdown`.
        Three Stop/Start cycles left four producers alive with the
        supervisor aware of one.
        """
        process = worker.process
        # ASK FIRST, IF THIS CHILD UNDERSTANDS BEING ASKED. A grace
        # window that waits without asking measures the whole window
        # every time and then shoots the process anyway -- see
        # `_ask_to_stop`.
        #
        # Two conditions, and both are load-bearing:
        #
        # `grace_seconds` -- `detach(grace_seconds=0)` means "gone now",
        # and asking a process to stop and then immediately terminating
        # it is worse than not asking, because a producer that had begun
        # its flush gets killed halfway through one.
        #
        # `handles_stop_request` -- a child that installed no handler
        # dies on the signal instead of finishing. Sending one to a
        # worker that never opted in would turn its grace into an instant
        # kill, which is the opposite of what a grace is for. See
        # `_Worker.handles_stop_request`.
        if grace_seconds and worker.handles_stop_request:
            _ask_to_stop(process)
        try:
            process.wait(timeout=grace_seconds)
            logger.info(
                "[Tower][Worker] worker pid %s for capture %s exited within "
                "its grace window",
                process.pid,
                worker.capture_id,
            )
            return True
        except Exception:
            if grace_seconds:
                logger.warning(
                    "[Tower][Worker] worker pid %s for capture %s did not "
                    "exit within %.1fs of being asked; terminating. Anything "
                    "it had not finished writing is lost.",
                    process.pid,
                    worker.capture_id,
                    grace_seconds,
                )
        # THE TREE, NOT THE PID. Through the worker's job on Windows, so
        # a builder's solve child dies with it; by walking descendants
        # elsewhere. A plain `terminate()` here is what left a solver
        # running for 32 s after its builder was gone on 2026-09-06.
        #
        # `terminate()` is asynchronous. On Windows it is
        # `TerminateProcess`, and a `poll()` immediately afterwards
        # routinely still returns None -- so `terminate_tree` waits
        # `TERMINATE_TIMEOUT_SECONDS` for the process to actually go.
        # That wait is not the grace window: the grace waited on a
        # process that had been asked; this waits on one that has just
        # been shot, and it is measured in milliseconds.
        if terminate_tree(process, job=worker.job, timeout=TERMINATE_TIMEOUT_SECONDS):
            return True
        logger.warning(
            "[Tower][Worker] pid %s is not gone within %.1fs of being "
            "terminated; it stays in the registry so it is still visible, "
            "and nothing else will be attached in its place",
            process.pid,
            TERMINATE_TIMEOUT_SECONDS,
        )
        return False


def _grace_for(spec: WorkerSpec, caller_grace: float) -> float:
    """The grace one worker gets: its spec's when set, else the caller's."""
    if spec.stop_grace_seconds is None:
        return caller_grace
    return spec.stop_grace_seconds
