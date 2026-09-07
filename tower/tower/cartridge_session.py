"""Start, Pause and Stop for a cartridge that produces out of process.

**Why this is shared code rather than one cartridge's.** Every producer
in this repository runs in its own process and tails a capture journal --
World Builder, Scene, Document Memory and Object Memory all do. None of
them had a way for a wearer to say "start remembering" or "stop", because
the only thing that ever started a producer was a human typing a command.
The first cartridge to need the button would have built it privately, and
the second would have built it again slightly differently. So the state
machine lives here, knows no cartridge, and is handed a worker NAME.

**What a session is not.**

  * Not a capture. The capture id is minted by the phone's
    `stream_start` and is shared by everything watching the camera; a
    session says what one cartridge DOES about it.
  * Not persisted. A Tower that restarts comes back `stopped`. Resuming
    a memory of what a camera sees, without anybody asking again, is the
    wrong direction to fail in -- and the alternative, a state file,
    would be a promise this process could not keep across a crash.
  * Not the producer. It starts and stops one. Whether a process is
    actually alive is the supervisor's answer, and `snapshot()` reports
    that separately from the state, because "the wearer asked for this"
    and "it is happening" are different claims and the gap between them
    is exactly the failure worth showing.

**Three states, and the reason there is no fourth.** `stopped`, `active`,
`paused`. There is no `starting`: attaching is a `Popen` and a dict
update, and a transient state that no client can ever observe is a state
that only exists to be got wrong.

**SERIALISED, because the transport is concurrent.** The routes are sync
`def` so a five-second detach stays off the event loop, which means
FastAPI runs them in its threadpool and two actions can be in flight at
once. Unserialised, a Stop arriving while a Start was between "set
ACTIVE" and "attach" left the session `stopped` with a producer running
-- and `stop()` from `stopped` returned early without detaching, so a
second Stop could not recover it. The one control a wearer has over being
remembered failed OPEN. A reviewer reproduced it: `state=stopped`,
`following=['cap-1']`, live pid.

Every action holds the lock for its whole duration, including the
attach and the detach. That makes a Pause take as long as stopping a
process takes and makes a concurrent Start wait for it, which is correct:
these are a person pressing buttons, not a hot path.
"""

import logging
import threading
import uuid

logger = logging.getLogger(__name__)

STOPPED = "stopped"
ACTIVE = "active"
PAUSED = "paused"

STATES = (STOPPED, ACTIVE, PAUSED)

START = "start"
PAUSE = "pause"
RESUME = "resume"
STOP = "stop"

ACTIONS = (START, PAUSE, RESUME, STOP)

# How long a producer gets to exit on its own before Pause or Stop
# terminates it.
#
# THIS WAS ZERO, AND ZERO WAS RIGHT AT THE TIME.
#
# The five seconds before that were worse than useless: nothing SIGNALLED
# the producer -- it is a follower tailing a journal that is still being
# written, so it had no reason to stop and never did -- and the wait
# measured 5.01 seconds every single time before the process was
# terminated anyway. The grace bought nothing and cost a wearer five
# seconds of a control whose whole purpose is to stop recording NOW.
#
# What changed is the producer. `scripts/object_memory_session.py` now
# installs `_StopRequest`, `CaptureWorkerSupervisor._ask_to_stop` sends
# the signal that sets it (a `CTRL_BREAK_EVENT` to the process group on
# Windows, where `terminate()` is `TerminateProcess` and cannot be
# caught), and the frame loop ends at the next frame so its
# `finally: engine.release()` runs. That call is the ONLY thing that
# closes the sightings still open at the instant of a Stop -- and the
# sighting still open when a wearer stops walking is, by construction,
# the object they were looking at most. Zero seconds threw those away.
#
# THREE, and not more, because this blocks an HTTP handler.
#
# The measured flush is a `wait_idle()` on the verification queue plus a
# `_settle()` per open sighting; the queue is bounded at 8 pending and a
# verdict measured 126 ms on the GPU, so the worst case is well inside
# this. The route is a sync `def` and the iOS client's timeout is ten
# seconds, so three leaves both room. A producer that overruns is still
# terminated, and the log says so and says what was lost.
DETACH_GRACE_SECONDS = 3.0

# What Stop (and Pause) do to a running producer.
#
# `terminate`: ask, wait DETACH_GRACE_SECONDS, terminate -- Object Memory's
# contract, where "stop recording NOW" is the whole point of the button.
#
# `request`: close the producer's stdin and return. The producer stays
# registered with the supervisor until it exits on its own and is reaped.
# World Builder's contract since 2026-09-06: a builder that is still
# observing ends its session and writes one final build within seconds; a
# builder already finalizing (30-135 s of solving after the wearer's Stop)
# is allowed to finish, because terminating it would turn "I looked at
# another tab" into an interrupted world. The hard path is `shutdown()`.
STOP_POLICY_TERMINATE = "terminate"
STOP_POLICY_REQUEST = "request"
STOP_POLICIES = (STOP_POLICY_TERMINATE, STOP_POLICY_REQUEST)


class SessionRefused(Exception):
    """An action this session cannot honour from the state it is in.

    Carries a `reason` short enough to be a wire value and a `message`
    long enough to be read by a person, because the transport has to
    serve both and inventing the split at the route would put half the
    vocabulary somewhere no test looks.
    """

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message


class CartridgeSession:
    """One cartridge's Start/Pause/Stop, over a capture it does not own.

    `open_capture` is a callable returning `(capture_id, capture_dir)` for
    whatever is recording right now, or None. Injected rather than
    reached for, because the recorder lives behind `app.state` and this
    class must stay testable without an app -- and because a session that
    imported the recorder would make the recorder's absence an
    AttributeError on the wearer's Start button.
    """

    def __init__(
        self,
        *,
        cartridge: str,
        worker: str,
        supervisor,
        open_capture,
        clock,
        detach_grace_seconds: float = DETACH_GRACE_SECONDS,
        stop_policy: str = STOP_POLICY_TERMINATE,
    ) -> None:
        if stop_policy not in STOP_POLICIES:
            raise ValueError(
                f"unknown stop policy {stop_policy!r}; expected one of {STOP_POLICIES}"
            )
        self._cartridge = cartridge
        self._worker = worker
        self._supervisor = supervisor
        self._open_capture = open_capture
        self._clock = clock
        self._stop_policy = stop_policy
        # A reading of the SUPERVISOR's clock, taken when this session
        # last went active. See `_go_active` and `mark`.
        self._attached_since: float | None = None
        # Whether this session has EVER been activated.
        #
        # Kept beside `_attached_since` because the two `None`s that
        # variable can hold mean opposite things to a client, and
        # collapsing them was a defect a reviewer found: "this session has
        # started nothing" is an empty list, and "this session cannot tell
        # you what it started" is a null. Without this flag both came out
        # as the empty list, and a client would then draw the loud "a
        # producer you did not start is recording" alarm over a producer
        # the session had started itself.
        self._activated = False
        self._detach_grace_seconds = detach_grace_seconds

        # Held for the whole of every action, attach and detach
        # included. See the module docstring: the transport is concurrent
        # and the failure was open.
        self._lock = threading.RLock()
        self._state = STOPPED
        self._session_id: str | None = None
        self._started_at: float | None = None
        self._changed_at: float | None = None
        # Every capture this session's worker has been seen following, in
        # the order they were first seen. Accumulated rather than
        # declared: a session started before the walk attaches nothing
        # itself -- the capture opens later and the supervisor consults
        # the gate -- so the only honest source is what the supervisor
        # says it is actually following.
        self._captures: list[str] = []

    # -- what the gate asks -------------------------------------------

    @property
    def state(self) -> str:
        return self._state

    def is_active(self) -> bool:
        """The predicate handed to the worker spec as its gate.

        A plain bool and nothing else. `capture_opened` runs on the
        connection that just received `stream_start`, so this is on the
        frame path's critical section and must not read a file, take a
        lock, or raise.
        """
        return self._state == ACTIVE

    @property
    def supported(self) -> bool:
        """Whether this Tower has a producer to start at all.

        False on a Tower where the cartridge is switched off. Reported
        rather than hidden: a Start button that silently does nothing is
        worse than one that says why it cannot.
        """
        try:
            return self._worker in self._supervisor.worker_names()
        except Exception:
            logger.exception(
                "[Tower][Session] could not ask the supervisor about worker %r",
                self._worker,
            )
            return False

    # -- actions -------------------------------------------------------

    def apply(self, action: str) -> dict:
        """Run a named action, and say in the log that somebody did.

        THE LOGGING IS NOT DECORATION.

        Until 2026-08-29 this module logged only exceptions, so a person
        pressing Start on their phone left no trace at all in the Tower
        console. Physically testing this cartridge therefore meant a human
        reading values off a screen and reciting them to whoever was
        debugging -- which is exactly the workflow this whole product pass
        exists to delete.

        One line per action and one per refusal. That is not noisy: these
        are human button presses, four of them exist, and the frame path
        does not come through here. Everything per-frame stays where it
        was, in the producer's own process, and reaches this console once
        as a report when the run ends.

        The line carries `following` deliberately. `state` alone would say
        what was asked for; the pair is what makes "the button worked and
        nothing is recording" legible from a log after the fact.
        """
        try:
            result = self._dispatch(action)
        except SessionRefused as refusal:
            logger.info(
                "[Tower][Session] %s %s REFUSED (%s): %s",
                self._cartridge,
                action,
                refusal.reason,
                refusal.message,
            )
            raise
        logger.info(
            "[Tower][Session] %s %s -> state=%s changed=%s session=%s "
            "attached=%s following=%s",
            self._cartridge,
            action,
            result.get("state"),
            result.get("changed"),
            result.get("session_id"),
            result.get("attached_capture_id"),
            result.get("following"),
        )
        return result

    def _dispatch(self, action: str) -> dict:
        """The one entry point a transport needs, without the logging."""
        if action == START:
            return self.start()
        if action == PAUSE:
            return self.pause()
        if action == RESUME:
            return self.resume()
        if action == STOP:
            return self.stop()
        raise SessionRefused(
            "unknown-action",
            f"{action!r} is not an action; expected one of {', '.join(ACTIONS)}",
        )

    def start(self) -> dict:
        """Begin remembering. From `stopped` or `paused`, and idempotent.

        `start` is deliberately accepted from `paused` as well as from
        `stopped`, while `resume` is not accepted from `stopped`. The
        asymmetry is on purpose: Start is what a wearer presses, and it
        should mean "be running" whatever the app thought the state was,
        whereas Resume is a claim about continuing something and should
        not quietly invent a session that was never started.
        """
        with self._lock:
            self._require_supported()
            if self._state == ACTIVE:
                return self._result(changed=False)
            if self._state == STOPPED:
                self._session_id = uuid.uuid4().hex
                self._started_at = self._clock()
                self._captures = []
            return self._go_active()

    def resume(self) -> dict:
        with self._lock:
            self._require_supported()
            if self._state == ACTIVE:
                return self._result(changed=False)
            if self._state != PAUSED:
                raise SessionRefused(
                    "not-paused",
                    "there is no paused session to resume; this cartridge is "
                    f"{self._state}",
                )
            return self._go_active()

    def pause(self) -> dict:
        """Stop remembering, keep the session.

        Pausing DETACHES the producer rather than signalling it to idle.
        The alternative was a control file the producer polls, and it was
        rejected: it would give the web process a write into the
        cartridge's own directory, add a second source of truth for
        whether the cartridge is running, and leave a stale "active" file
        behind after a crash. Stopping the process makes the pause
        observable in the process table, costs one model load to undo,
        and cannot go stale.

        It is also PROMPT. See DETACH_GRACE_SECONDS: the producer is not
        signalled and has no reason to exit on its own, so waiting for it
        bought nothing and cost a wearer five seconds of a control whose
        whole purpose is to stop recording now.

        What that costs is the producer's in-memory state -- which track
        was open, which class was last recorded. Losing it errs towards
        one extra honest observation after a resume, never towards a
        suppressed real one, which is the direction this cartridge
        already errs on restart.
        """
        with self._lock:
            self._require_supported()
            if self._state == PAUSED:
                return self._result(changed=False)
            if self._state != ACTIVE:
                raise SessionRefused(
                    "not-active",
                    f"there is nothing to pause; this cartridge is {self._state}",
                )
            self._detach()
            self._state = PAUSED
            self._changed_at = self._clock()
            return self._result(changed=True)

    def stop(self) -> dict:
        """End the session. Never refused, from any state.

        Whatever a confused client believes, "stop" is a request to end
        up stopped, and it can always be honoured. Refusing it would
        leave the only way out of a bad state being a Tower restart.

        Observations already written are untouched. Stopping ends the
        producing, not the memory.

        IT ALWAYS DETACHES, EVEN FROM `stopped`. This used to return
        early when the state was already `stopped`, on the reasonable
        view that there was nothing to do -- and that made the one
        recoverable state unrecoverable. Stop means "end up with nothing
        recording", and the only way to keep that promise is to check.
        Detaching when nothing is attached costs a dictionary lookup.
        """
        with self._lock:
            already_stopped = self._state == STOPPED
            self._detach()
            self._state = STOPPED
            self._session_id = None
            self._started_at = None
            # `_attached_since` is deliberately NOT cleared. A producer
            # this session started and could not kill is still this
            # session's producer, and it is still recording; saying so
            # after a Stop is the whole point. Clearing the mark would
            # widen the filter to every worker on the Tower, which is the
            # opposite of what it is for.
            if already_stopped:
                return self._result(changed=False)
            self._changed_at = self._clock()
            return self._result(changed=True)

    # -- reporting -----------------------------------------------------

    def snapshot(self) -> dict:
        """Everything a client needs to draw the cartridge honestly.

        Accumulating `captures` here means a read has a side effect, and
        that is the lesser evil. The alternative is a callback from the
        supervisor into every session, which would put a cartridge's
        bookkeeping on the connection that handles `stream_start`; this
        way the frame path stays untouched and the list is refreshed
        exactly when somebody is looking.
        """
        following = self._following()
        mine = self._following_this_session()
        # `mine` is None when this session cannot scope the question. The
        # history then accumulates from the WIDE list, which is what it
        # did before scoping existed and is the only answer available.
        for capture_id in following if mine is None else mine:
            if capture_id not in self._captures:
                self._captures.append(capture_id)
        return {
            "cartridge": self._cartridge,
            "worker": self._worker,
            "supported": self.supported,
            "state": self._state,
            # How Stop treats a running producer (see STOP_POLICY_*). A
            # client showing "stopped" must know whether that meant "the
            # process is gone" or "the process was asked and is finishing".
            "stop_policy": self._stop_policy,
            "session_id": self._session_id,
            "started_at": self._started_at,
            "changed_at": self._changed_at,
            # What the wearer asked for is `state`. What is actually
            # happening is this. An ACTIVE session with an empty
            # `following` while a capture is recording is a producer that
            # died, and the phone should be able to say so.
            #
            # EVERY live producer for this worker, including one left
            # over from a session that could not kill it. That breadth is
            # deliberate and must not be narrowed: a worker nobody can
            # stop is the one thing a Stop button failing open looks
            # like, and hiding it here would make the failure silent.
            "following": following,
            # The subset THIS session actually started -- the field a
            # liveness claim should be drawn from -- or **null** when this
            # session cannot scope the question at all, which a client
            # must read as "fall back to `following`" and never as
            # "nothing".
            #
            # `following` alone produced a documented false positive:
            # press Stop against an un-killable producer, press Start
            # again, and the new session reports the old session's
            # capture under a new `session_id` having attached nothing.
            # A client keying "remembering" off that tells a person their
            # memory is being written when this session wrote none of it.
            #
            # Scoped by START TIME rather than by a list of ids this
            # object keeps, because the interesting attachments are the
            # ones nothing here decided: a capture that opens while the
            # gate is open is spawned by the supervisor without asking
            # the session, and a session that only counted its own
            # `_attach` returns would miss every one of them -- which is
            # most of them, since Start before the camera is the normal
            # order.
            "following_this_session": mine,
            "captures": list(self._captures),
        }

    # -- internals -----------------------------------------------------

    def _require_supported(self) -> None:
        if not self.supported:
            raise SessionRefused(
                "unsupported",
                f"this Tower has no {self._cartridge} producer configured, so "
                "there is nothing to start",
            )

    def _go_active(self) -> dict:
        self._state = ACTIVE
        self._changed_at = self._clock()
        # BEFORE the attach, and on the supervisor's own clock. A mark
        # taken afterwards could be later than the `started_at` of the
        # worker this very call spawns, and the session would then report
        # that it had attached nothing. Re-taken on every activation, so
        # a producer that survived a Pause it was supposed to be killed
        # by is not counted as this session's after the Resume.
        self._attached_since = self._supervisor_mark()
        self._activated = True
        attached = self._attach()
        result = self._result(changed=True)
        result["attached_capture_id"] = attached
        return result

    def _attach(self) -> str | None:
        """Attach to whatever is recording NOW, if anything.

        Deliberately re-asked on every start and resume rather than
        remembered. A pause long enough to matter is long enough for the
        phone to have reconnected, and a session that re-attached to the
        capture it remembered would follow a directory nothing is writing
        to any more.
        """
        try:
            current = self._open_capture()
        except Exception:
            logger.exception(
                "[Tower][Session] could not find out what is recording; %s is "
                "%s and will attach when the next capture opens",
                self._cartridge,
                self._state,
            )
            return None
        if current is None:
            return None
        capture_id, capture_dir = current
        try:
            started = self._supervisor.attach(self._worker, capture_id, capture_dir)
        except Exception:
            logger.exception(
                "[Tower][Session] could not attach the %s worker to capture %s",
                self._worker,
                capture_id,
            )
            return None
        if not started:
            # Either one was already following -- which is fine and
            # common, a Start pressed twice -- or the spawn failed, which
            # the supervisor has already logged loudly. Neither is a
            # reason to refuse the wearer's Start, and `following` in the
            # snapshot tells the two apart.
            return None
        return capture_id

    def _detach(self) -> None:
        try:
            if self._stop_policy == STOP_POLICY_REQUEST:
                asked = self._supervisor.request_stop(self._worker)
                logger.info(
                    "[Tower][Session] %s asked %s worker(s) to stop; they finish "
                    "on their own and stay visible until they do",
                    self._cartridge,
                    asked,
                )
                return
            self._supervisor.detach(
                self._worker, grace_seconds=self._detach_grace_seconds
            )
        except Exception:
            # The state change goes ahead regardless. Closing the gate is
            # what stops the NEXT capture being followed, and a session
            # that refused to move because a terminate failed would keep
            # attaching a producer to every future capture.
            logger.exception(
                "[Tower][Session] could not stop the %s worker; the gate is "
                "closed and no new capture will be followed",
                self._worker,
            )

    def _supervisor_mark(self) -> float | None:
        try:
            return self._supervisor.mark()
        except Exception:
            # A supervisor from before `mark` existed, or one that raised.
            # `following_this_session` then stays EMPTY rather than
            # falling back to the unscoped list, and the direction is
            # chosen deliberately.
            #
            # This field is what a client draws "remembering" from. An
            # unscoped fallback would let a leftover producer light that
            # up for a session that started nothing, which is the exact
            # false positive this whole change exists to remove -- a
            # false success is the one outcome worse than no answer.
            # `following` is unaffected and still carries every live
            # producer, so nothing becomes invisible; the claim just
            # stops being made.
            logger.debug(
                "[Tower][Session] supervisor has no usable clock mark; "
                "%s will not scope `following_this_session`",
                self._cartridge,
                exc_info=True,
            )
            return None

    def _scoped_following(self, since: float) -> list[str] | None:
        """The scoped list, or None when this supervisor cannot scope.

        SEPARATE FROM `_following` ON PURPOSE, and a reviewer found the
        defect that made it necessary.

        `_following` WIDENS on a `TypeError`: it retries without `since`,
        because a supervisor from before that keyword existed should
        still be able to answer the public `following` field, and
        over-reporting what is running is the safe direction there.

        It is the unsafe direction here. Handing that widened list back
        as `following_this_session` tells a client "these are the
        producers YOU started" when the answer is "every producer on this
        Tower" -- which is exactly the false positive this field was
        added to remove, and it would raise the loud "something you did
        not start is recording, and Stop will not reach it" warning about
        a recording the person started themselves. `_supervisor_mark`
        already refuses to guess for that reason; this refuses for it
        too.

        The `TypeError` catch is narrow and deliberate. One raised INSIDE
        a modern `following()` is a bug in the supervisor rather than an
        old signature, and swallowing it as "cannot scope" would hide it
        -- so it is logged at exception level in the general branch below
        rather than folded into the quiet one.
        """
        try:
            return list(self._supervisor.following(self._worker, since=since))
        except TypeError:
            logger.debug(
                "[Tower][Session] this supervisor cannot answer a scoped "
                "`following`; %s reports null rather than a list it cannot "
                "stand behind",
                self._cartridge,
                exc_info=True,
            )
            return None
        except Exception:
            logger.exception(
                "[Tower][Session] could not ask the supervisor what worker %r "
                "is following for this session",
                self._worker,
            )
            return None

    def _following_this_session(self) -> list[str] | None:
        """Live producers THIS session started, or None if it cannot say.

        THREE ANSWERS, NOT TWO, AND THE THIRD IS WHY THIS IS NOT A LIST.

        - a **list** -- these captures, and no others, have a producer
          this session started still alive on them;
        - the **empty list** -- this session has started nothing that is
          still running. A positive claim;
        - **None** -- this session cannot scope the question at all,
          because the supervisor has no clock mark to offer or cannot
          answer a scoped question. See `_scoped_following`.

        The third is not pedantry. A client draws "you are being
        recorded" from this field and draws a loud "a producer you did
        NOT start is recording, and Stop will not reach it" from the
        difference between it and `following`. Answering "I cannot tell"
        with the empty list turns every producer into somebody else's,
        including this session's own, and prints an alarm about a
        recording the person started themselves. A reviewer found exactly
        that, and the fix is to say null and let the client fall back to
        `following`.

        Not gated on the state. An earlier draft returned the empty list
        whenever the session was `stopped`, and that was wrong in the one
        direction that matters: a producer a Stop failed to kill is still
        this session's and is still recording. The honest answer to "what
        did I start that is running" is its capture, not nothing. The
        same holds for `paused`.
        """
        if not self._activated:
            # A positive claim, and the one case in which the empty list
            # is unambiguous: nothing has ever been asked for here, so
            # nothing here started anything.
            return []
        if self._attached_since is None:
            # Activated, and no clock mark to scope by -- a supervisor
            # from before `mark()` existed, or one that raised. The claim
            # is unavailable, which is not the same as false.
            return None
        return self._scoped_following(self._attached_since)

    def _following(self, *, since: float | None = None) -> list[str]:
        try:
            return list(self._supervisor.following(self._worker, since=since))
        except TypeError:
            # A supervisor from before `since` existed. Answering the
            # narrower question with the broader one is the direction that
            # OVER-reports what is running, which is the safe direction
            # for a field a wearer reads as "you are being recorded".
            return list(self._supervisor.following(self._worker))
        except Exception:
            logger.exception(
                "[Tower][Session] could not ask the supervisor what worker %r "
                "is following",
                self._worker,
            )
            return []

    def _result(self, *, changed: bool) -> dict:
        result = self.snapshot()
        result["changed"] = changed
        result.setdefault("attached_capture_id", None)
        return result
