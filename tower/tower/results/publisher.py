"""Fan-out of cartridge snapshots to subscribers, with hard bounds.

Three decisions shape everything here, and all three follow from one fact:
**a World Builder status is a SNAPSHOT, not a log entry.** It is recomputed
from files on disk that are themselves the durable record. Nothing is lost
by discarding an older snapshot, because a newer one answers the same
question better.

So:

1. **There is no queue.** Each subscription owns exactly ONE slot holding
   the most recent snapshot it has not yet been sent. A new snapshot
   REPLACES whatever is in that slot. Memory per subscription is therefore
   one payload, not N -- and the "unbounded event queue" the brief warns
   about is not bounded here, it is absent by construction. A bounded
   queue of N would have been worse than either: it drops the NEWEST
   updates once full, which is precisely backwards for a freshness-first
   channel.

2. **One poller serves everyone.** The reader is per-app, not per
   connection, so ten subscribers watching one world cost one disk read
   per interval, not ten. It runs only while at least one subscription
   exists: a Tower nobody is watching does no work.

3. **A slow consumer is dropped, never tolerated indefinitely.** Sends are
   bounded by `SEND_TIMEOUT_S`. A client that stops reading gets its
   subscription closed with a reason. This matters because the frame path
   shares the socket: one TCP stream means a client that is not reading
   blocks everything, and a result send must never be the thing holding
   that up without limit.

The poll loop touches no cartridge state. It reads files. It cannot alter
capture cadence, frame routing, World Builder processing or cartridge
correctness, because it has no reference to any of them.
"""

import asyncio
import logging

from tower.results.contracts import RESULT_TYPE_STATUS
from tower.results.envelope import ResultEnvelope

logger = logging.getLogger(__name__)

# How often the shared reader looks at disk. A World Builder keyframe is
# accepted at most a few times a second and a rebuild is far rarer, so
# polling faster would spend IO to observe nothing. Measured read cost
# governs how low this can go; see the contract document.
DEFAULT_POLL_SECONDS = 0.5

# A snapshot is re-sent this often even when the revision has not changed,
# so a live figure that legitimately advances -- mapping seconds -- does
# not freeze on screen. `revision_changed: false` marks these, so a client
# can tell a heartbeat from news.
DEFAULT_HEARTBEAT_SECONDS = 2.0

# A send that takes longer than this means the consumer has stopped
# reading. The subscription is closed rather than waited on.
#
# 1 second, and the number is a FRAME-PATH bound rather than a patience
# setting. `handoff.md` 13.7 is explicit that blocking the socket for >2 s
# makes iOS treat the connection as wedged and replace it, so this stays
# comfortably under that with the lock wait added on top. The result sender holds the connection's send lock for
# the duration of a send, so a stuck result is also the longest a
# `frame_result` can queue behind one. Both are blocked anyway when the
# client has stopped reading -- one socket is one TCP stream -- but this
# is the bound on how long a merely SLOW consumer can delay the path that
# is measured.
SEND_TIMEOUT_S = 1.0

# How long to wait for the send lock itself, measured separately. Lumping
# the two together let a slow FRAME send consume a result's whole budget
# and trigger a spurious "consumer did not accept" drop -- the result had
# not been offered to the socket at all, it was queued behind the frame
# path. Raised by an adversarial review.
LOCK_TIMEOUT_S = 1.0

# The backstop the sender task applies around the whole operation. The
# inner two bounds give the precise cause; this one guarantees the task
# cannot sit in a send forever if a transport ever fails to honour them.
TOTAL_SEND_TIMEOUT_S = LOCK_TIMEOUT_S + SEND_TIMEOUT_S

# Consecutive failed snapshot attempts for one target before its
# subscribers are told and dropped.
#
# One failure is transient -- a file being replaced underneath us is
# routine and the next poll succeeds. Failing every time is not transient,
# and swallowing it forever leaves the client waiting on a channel that is
# never coming back. An earlier version logged each failure and continued
# indefinitely, which is the silence this module's header calls the worst
# outcome. Three, so a burst of contention cannot trip it.
# How long one cartridge's snapshot may take before the pass moves on
# without it. Generous by design: the slowest measured real snapshot is
# World Builder's at ~40 ms on a 163-world root, and `GET /worlds` under a
# disk fault that touches every manifest is ~2.1 s. This is not a
# performance budget -- it is the line between "slow" and "wedged", and
# only the second is worth telling a subscriber about.
SNAPSHOT_TIMEOUT_SECONDS = 10.0

MAX_CONSECUTIVE_TARGET_FAILURES = 3

# Per connection. A client with more than this many open subscriptions is
# either confused or hostile; either way the answer is a refusal, not
# unbounded growth in a dict a remote party controls.
MAX_SUBSCRIPTIONS_PER_CONNECTION = 8

# How a reader failure reaches a client: as a VALUE in the subscription's
# slot, never as an exception. The sender already swallows ordinary send
# failures (a closing socket is routine), so an exception carrying "the
# reader is dead" would be eaten by that same clause and the client would
# wait forever for a channel that is never coming back. Silence is the
# worst of the three outcomes -- worse than a crash, because nothing
# anywhere reports it.
CURSOR_MATCHED = "matched"
CURSOR_STALE = "stale"
CURSOR_UNRECOGNISED = "unrecognised"
CURSOR_ABSENT = "absent"


class Subscription:
    """One client's standing interest in one cartridge result.

    Holds at most one undelivered snapshot. `coalesced` counts how many
    were superseded in that slot since the last successful send, which is
    the only honest way to tell a client it was slow -- a sequence gap
    would say "you missed something", and it did not.
    """

    __slots__ = (
        "subscription_id",
        "cartridge",
        "result_type",
        "contract",
        "world_id",
        "session_id",
        "seq",
        "last_revision",
        "last_sent_at",
        "coalesced",
        "cursor_status",
        "_pending",
        "_failure",
    )

    def __init__(
        self,
        *,
        subscription_id: str,
        cartridge: str,
        result_type: str,
        contract: str,
        world_id: str | None,
        session_id: str | None,
        cursor_status: str,
    ) -> None:
        self.subscription_id = subscription_id
        self.cartridge = cartridge
        self.result_type = result_type
        self.contract = contract
        self.world_id = world_id
        self.session_id = session_id
        self.seq = 0
        self.last_revision: str | None = None
        self.last_sent_at: float | None = None
        self.coalesced = 0
        self.cursor_status = cursor_status
        self._pending = None
        self._failure: str | None = None

    @property
    def target(self) -> tuple:
        """What the shared reader keys its work on.

        Two subscriptions naming the same cartridge, result type, world and
        session are answered by ONE snapshot computation, however many
        connections asked.
        """
        return (self.cartridge, self.result_type, self.world_id, self.session_id)

    def offer(self, snapshot) -> None:
        if self._pending is not None:
            if self._pending.revision == snapshot.revision:
                # Same content. Not a supersession, so it must not count
                # as one: `coalesced` tells a client it was too slow to
                # see intermediate STATES, and re-offering an identical
                # snapshot is not an intermediate state. Counting it would
                # report drops that never happened -- most visibly right
                # after subscribe, where the first snapshot is seeded
                # directly and the next poll re-offers the same one.
                return
            # The previous snapshot was never sent. Replaced, not queued.
            self.coalesced += 1
        self._pending = snapshot

    def take(self):
        snapshot, self._pending = self._pending, None
        return snapshot

    def fail(self, reason: str) -> None:
        self._failure = reason

    def take_failure(self):
        failure, self._failure = self._failure, None
        return failure

    @property
    def has_pending(self) -> bool:
        return self._pending is not None or self._failure is not None


class ConnectionChannel:
    """Every subscription belonging to one WebSocket, plus its sender.

    One sender task per CONNECTION rather than per subscription: a client
    with eight subscriptions should not cost eight tasks, and messages on
    one socket have to be serialised anyway.
    """

    def __init__(self, hub, send, clock) -> None:
        self._hub = hub
        self._send = send
        self._clock = clock
        self._subscriptions: dict[str, Subscription] = {}
        self._wakeup = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._closed = False
        self._counter = 0

    # -- lifecycle ------------------------------------------------------

    def next_subscription_id(self) -> str:
        self._counter += 1
        return f"sub-{self._counter}"

    @property
    def subscription_count(self) -> int:
        return len(self._subscriptions)

    def get(self, subscription_id: str):
        return self._subscriptions.get(subscription_id)

    async def add(self, subscription: Subscription) -> None:
        self._subscriptions[subscription.subscription_id] = subscription
        if self._task is None:
            self._task = asyncio.create_task(self._run())
        await self._hub.attach(self)

    async def remove(self, subscription_id: str) -> bool:
        removed = self._subscriptions.pop(subscription_id, None) is not None
        if removed and not self._subscriptions:
            await self._hub.detach(self)
        return removed

    async def close(self) -> None:
        """Tear down on ANY exit from the connection, polite or not.

        Symmetric with ws.py's own `finally:` handling of capture. A
        subscription that outlived its socket would keep a shared reader
        polling disk on behalf of a client that is gone -- the exact leak
        that makes a push channel a liability rather than a feature.
        """
        if self._closed:
            return
        self._closed = True
        self._subscriptions.clear()
        await self._hub.detach(self)
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                # The sender task's cancellation, not ours. If THIS task is
                # also being cancelled -- app shutdown racing a disconnect
                # -- that must not be swallowed, or the shutdown waits
                # forever for a cancellation nobody delivered.
                current = asyncio.current_task()
                if current is not None and current.cancelling() > 0:
                    raise
            except Exception:  # noqa: BLE001
                # Anything else is already logged in _run. Nothing else may
                # propagate: this runs in the connection's cleanup path,
                # where an exception would skip the capture and session
                # teardown that follows it.
                logger.debug(
                    "[Tower][Results] sender task ended badly", exc_info=True
                )

    # -- delivery -------------------------------------------------------

    def offer(self, target, snapshot, *, now: float, heartbeat: float) -> None:
        """Give a new snapshot to every matching subscription.

        Called by the hub, on the event loop, and does no IO: it drops a
        reference into a slot and sets an event. A slow client therefore
        cannot slow the reader down, only itself.
        """
        woken = False
        for subscription in self._subscriptions.values():
            if subscription.target != target:
                continue
            changed = snapshot.revision != subscription.last_revision
            due = (
                subscription.last_sent_at is None
                or (now - subscription.last_sent_at) >= heartbeat
            )
            if changed or due:
                subscription.offer(snapshot)
                woken = True
        if woken:
            self._wakeup.set()

    def fail_all(self, reason: str) -> None:
        for subscription in self._subscriptions.values():
            subscription.fail(reason)
        self._wakeup.set()

    def fail_target(self, target, reason: str) -> None:
        """Fail only the subscriptions watching one target.

        A world nobody can read must not take down a subscription to a
        different world on the same connection.
        """
        woken = False
        for subscription in self._subscriptions.values():
            if subscription.target == target:
                subscription.fail(reason)
                woken = True
        if woken:
            self._wakeup.set()

    async def _run(self) -> None:
        try:
            while True:
                await self._wakeup.wait()
                self._wakeup.clear()
                await self._drain()
        except asyncio.CancelledError:
            raise
        except Exception:
            # A push-channel failure must never take the connection with
            # it. The receive loop keeps running, frames keep being
            # answered, and this is logged with a traceback rather than
            # disappearing into a task nobody awaits.
            logger.exception(
                "[Tower][Results] sender task failed; the frame path is "
                "unaffected and this connection will publish no further "
                "results"
            )

    async def _drain(self) -> None:
        for subscription in list(self._subscriptions.values()):
            if not subscription.has_pending:
                continue
            failure = subscription.take_failure()
            if failure is not None:
                # In-band, on the same socket, and then the subscription
                # is closed. A client is told once and never left holding
                # a subscription that will never speak again.
                try:
                    await asyncio.wait_for(
                        self._send(
                            {
                                "type": "result_error",
                                "reason": "channel_failed",
                                "subscription_id": subscription.subscription_id,
                                "cartridge": subscription.cartridge,
                                "result_type": subscription.result_type,
                                "message": failure,
                            }
                        ),
                        timeout=SEND_TIMEOUT_S,
                    )
                except Exception:
                    return
                await self.remove(subscription.subscription_id)
                continue
            snapshot = subscription.take()
            envelope = _envelope(subscription, snapshot, self._clock())
            try:
                await asyncio.wait_for(
                    self._send(envelope.to_json_dict()),
                    timeout=TOTAL_SEND_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "[Tower][Results] subscription %s: consumer did not accept "
                    "a result within %.1fs; closing it",
                    subscription.subscription_id,
                    SEND_TIMEOUT_S,
                )
                # TELL the client before dropping it. An earlier version
                # closed the subscription with no message at all, and the
                # contract says `channel_failed` is the only unsolicited
                # error -- so a conforming client had no way to learn its
                # subscription was gone and would wait forever. This
                # module's own header argues silence is the worst
                # outcome; it was doing exactly that. Best-effort and
                # short: the consumer is already not reading, so this
                # very likely fails too, and that is fine.
                try:
                    await asyncio.wait_for(
                        self._send(
                            {
                                "type": "result_error",
                                "reason": "consumer_too_slow",
                                "subscription_id": subscription.subscription_id,
                                "cartridge": subscription.cartridge,
                                "result_type": subscription.result_type,
                                "message": (
                                    "this subscription was closed because a "
                                    f"result was not accepted within "
                                    f"{SEND_TIMEOUT_S:.0f}s; subscribe again "
                                    "to resume"
                                ),
                            }
                        ),
                        timeout=0.5,
                    )
                except Exception:
                    pass
                await self.remove(subscription.subscription_id)
                continue
            except Exception:
                # Includes WebSocketDisconnect. The socket is gone; the
                # connection's own finally: will call close(). Stop
                # draining rather than trying every remaining subscription
                # against a dead socket.
                logger.info(
                    "[Tower][Results] subscription %s: send failed, the "
                    "connection is closing",
                    subscription.subscription_id,
                )
                return
            subscription.seq += 1
            subscription.last_revision = snapshot.revision
            subscription.last_sent_at = self._clock()
            subscription.coalesced = 0
            # A cursor status describes the FIRST reply to a subscribe and
            # nothing after it.
            subscription.cursor_status = None


def _envelope(subscription: Subscription, snapshot, now: float) -> ResultEnvelope:
    return ResultEnvelope(
        cartridge=subscription.cartridge,
        result_type=subscription.result_type,
        contract=subscription.contract,
        subscription_id=subscription.subscription_id,
        seq=subscription.seq + 1,
        revision=snapshot.revision,
        revision_changed=snapshot.revision != subscription.last_revision,
        tower_sent_at=now,
        payload=snapshot.payload,
        coalesced=subscription.coalesced,
        cursor_status=subscription.cursor_status,
    )


class ResultHub:
    """The shared reader. One per app; owns the poll task.

    Deliberately holds no cartridge object. It is handed a callable that
    turns a target into a snapshot, and that callable reads files. Nothing
    here can reach into World Builder, start a build, or touch a frame.
    """

    def __init__(
        self,
        snapshot_for,
        *,
        clock,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
        heartbeat_seconds: float = DEFAULT_HEARTBEAT_SECONDS,
    ) -> None:
        self._snapshot_for = snapshot_for
        self._clock = clock
        self._poll_seconds = poll_seconds
        self._heartbeat_seconds = heartbeat_seconds
        self._channels: set = set()
        self._task: asyncio.Task | None = None
        # Bounded by the number of live targets, and pruned every pass.
        self._failures: dict = {}
        # One running snapshot per target, at most. See `poll_once`.
        self._in_flight: dict = {}
        # Set after every completed poll pass. Tests wait on this instead
        # of sleeping, which is what makes them deterministic.
        self.polled = asyncio.Event()

    async def attach(self, channel) -> None:
        self._channels.add(channel)
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def detach(self, channel) -> None:
        self._channels.discard(channel)
        if not self._channels and self._task is not None:
            # Nobody is watching. Stop reading disk entirely rather than
            # spinning a timer forever on a Tower whose client went home.
            #
            # Cancel and DROP -- deliberately no `await task` here. This
            # runs from whichever task called remove(), and that is often
            # the per-connection SENDER task: awaiting the reader from
            # inside the sender couples two cancellations together, and a
            # test that drove a persistently-failing reader deadlocked the
            # event loop exactly there (the loop went idle in select with
            # nothing scheduled and the main coroutine waiting forever).
            #
            # Nothing needs the task's result. It is cancelled, it will
            # unwind on its own, and `shutdown()` -- which runs from the
            # app's own teardown, never from a sender -- is where a real
            # join belongs.
            task, self._task = self._task, None
            task.cancel()

    async def shutdown(self) -> None:
        """Stop the reader on app teardown. Never raises.

        Awaiting a task that already died would re-raise its exception
        inside the shutdown handler, turning a dead push channel into a
        failed application shutdown.
        """
        self._channels.clear()
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            # The task we just cancelled, not this one. Swallowing our own
            # cancellation here would strand the shutdown that requested
            # it, so re-raise if the enclosing task is also being
            # cancelled.
            if asyncio.current_task() is not None and (
                asyncio.current_task().cancelling() > 0
            ):
                raise
        except Exception:  # noqa: BLE001
            # A reader that already died must not turn a clean shutdown
            # into a failed one.
            logger.debug("[Tower][Results] reader had already failed", exc_info=True)

    async def _run(self) -> None:
        try:
            while True:
                await self.poll_once()
                await asyncio.sleep(self._poll_seconds)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception(
                "[Tower][Results] shared reader failed; no further results "
                "will be published. The frame path and every cartridge are "
                "unaffected"
            )
            # Tell every subscriber rather than going quiet. A dead
            # channel that still looks alive is the failure mode this
            # whole path is built to avoid.
            reason = (
                "the Tower's shared result reader stopped with "
                f"{type(exc).__name__}; no further results will be published "
                "on this connection"
            )
            for channel in list(self._channels):
                try:
                    channel.fail_all(reason)
                except Exception:
                    logger.exception(
                        "[Tower][Results] could not notify a channel of the "
                        "reader failure"
                    )

    def _note_failure(self, target, reason: str, *, log=None) -> None:
        """Count one failure for a target, and tell its subscribers after three.

        One place for both failure kinds -- a snapshot that raised and one
        that is still running at the deadline -- so they escalate the same
        way and say the same thing. Persistent, not transient: after
        `MAX_CONSECUTIVE_TARGET_FAILURES` in a row the subscribers watching
        this target are told rather than left holding a subscription that
        has gone quiet.
        """
        if log is not None:
            logger.warning(*log)
        failures = self._failures.get(target, 0) + 1
        self._failures[target] = failures
        if failures < MAX_CONSECUTIVE_TARGET_FAILURES:
            return
        message = f"{reason}, {failures} times in a row"
        for channel in list(self._channels):
            try:
                channel.fail_target(target, message)
            except Exception:
                logger.exception(
                    "[Tower][Results] could not notify a channel of a "
                    "persistent target failure"
                )
        self._failures.pop(target, None)

    async def poll_once(self) -> None:
        """One pass: compute each distinct target once, offer it to all.

        Snapshot computation is pushed off the event loop with
        `asyncio.to_thread`, because it reads and JSON-parses files and
        the loop is also answering frames. A disk stall must cost this
        channel latency, never the frame path.
        """
        targets = {}
        for channel in self._channels:
            for subscription in channel._subscriptions.values():
                targets[subscription.target] = subscription

        # Forget the failure counts of targets nobody is watching any
        # more. `_failures` was pruned on success and on escalation, but
        # not when a subscription simply went away, so a target that
        # failed once and was then unsubscribed left an entry forever.
        #
        # It is small and it is genuinely unbounded: `Subscription.target`
        # includes the client-chosen `world_id` and `session_id`, so a
        # connection can mint distinct targets at will. Bounded now by the
        # live subscriptions rather than by the client's imagination. The
        # comment at the field says "pruned every pass"; this is what
        # makes that true.
        if len(self._failures) > len(targets):
            self._failures = {
                target: count
                for target, count in self._failures.items()
                if target in targets
            }

        # ONE SNAPSHOT IN FLIGHT PER TARGET, ALL TARGETS AT ONCE, ONE
        # DEADLINE FOR THE PASS.
        #
        # Three defects lived in the loop this replaces, and the second
        # was introduced by the fix for the first.
        #
        # First, targets were polled SEQUENTIALLY with no deadline, so a
        # wedged World Builder read delivered nothing, for any cartridge,
        # across 12 poll windows with no error and no `fail_target`; and a
        # merely slow (2 s) producer made Document Memory and Scene
        # Understanding 5x slower, because the pass length was the SUM of
        # the targets.
        #
        # Second, the fix wrapped each call in `asyncio.wait_for`, which
        # cancels the await and NOT the thread -- and then re-dispatched
        # the same target on the very next pass while the previous thread
        # was still running. A reviewer measured one leaked thread per
        # poll per wedged target into the default executor, which the
        # capture path shares: at the default 0.5 s poll and 10 s deadline
        # a single wedged target exhausted a 24-worker pool in ~252 s,
        # after which EVERY `asyncio.to_thread` in the process queued
        # forever -- `stream_start`'s `capture_opened`, the disconnect
        # cleanup, the subscribe-time snapshot, `supervisor.shutdown`.
        # Before that fix the wedge silenced the result channel and
        # nothing else. That is a fix worse than its defect, and it is
        # this campaign's own.
        #
        # Third, the sequential shape meant the 10 s deadline only turned
        # ">10 s" into a failure; a 2 s producer still cost everyone 2 s.
        #
        # So: a target that already has a snapshot running is NOT
        # dispatched again -- its future is kept in `_in_flight` and
        # simply checked -- which bounds the threads at one per target by
        # construction, however long a read wedges. Every target that is
        # free is dispatched, and the pass waits on all of them together
        # with one bounded `asyncio.wait`, which does not cancel: a slow
        # snapshot survives into the next pass and is delivered when it
        # lands, as the freshest state there is. The pass length is the
        # SLOWEST target capped at the deadline, not the sum. A target
        # still pending at the deadline counts one failure, and the
        # existing escalation tells its subscribers after three.
        for target in list(self._in_flight):
            if target not in targets and self._in_flight[target].done():
                # Nobody is watching this any more and its thread has
                # finished: forget it. A still-running thread for a
                # departed target stays until it finishes, so it can
                # never be dispatched twice.
                del self._in_flight[target]

        waiting: dict = {}
        for target, sample in targets.items():
            future = self._in_flight.get(target)
            if future is None or future.done() and target not in self._in_flight:
                future = None
            if future is None:
                future = asyncio.ensure_future(
                    asyncio.to_thread(
                        self._snapshot_for, sample.cartridge,
                        sample.result_type, sample.world_id,
                        sample.session_id,
                    )
                )
                self._in_flight[target] = future
            waiting[future] = target

        if waiting:
            await asyncio.wait(
                list(waiting), timeout=SNAPSHOT_TIMEOUT_SECONDS
            )

        now = self._clock()
        for future, target in waiting.items():
            if not future.done():
                self._note_failure(
                    target,
                    f"the Tower could not build this cartridge's state within "
                    f"{SNAPSHOT_TIMEOUT_SECONDS:.0f}s",
                    log=(
                        "[Tower][Results] snapshot for %s has exceeded %.1fs "
                        "and is still running; the other cartridges are "
                        "being served without it",
                        target, SNAPSHOT_TIMEOUT_SECONDS,
                    ),
                )
                continue
            # Done, one way or the other: this target is free for the
            # next pass whether it succeeded or raised.
            self._in_flight.pop(target, None)
            try:
                snapshot = future.result()
            except Exception as exc:
                # One unreadable target must not stop the others, and must
                # not stop the loop. The producer already turns expected
                # storage failures into an `unavailable` payload; reaching
                # here means something genuinely unexpected.
                logger.exception(
                    "[Tower][Results] could not build a snapshot for %s", target
                )
                self._note_failure(
                    target,
                    f"the Tower could not read this cartridge's state; the "
                    f"last failure was {type(exc).__name__}",
                )
                continue
            self._failures.pop(target, None)
            for channel in list(self._channels):
                channel.offer(
                    target, snapshot, now=now, heartbeat=self._heartbeat_seconds
                )

        self.polled.set()
        self.polled.clear()


def classify_cursor(since_revision, current_revision) -> str:
    """What to make of a cursor a reconnecting client supplied.

    A cursor here can never cause data loss, and that is a property of the
    design rather than of this function: every subscription begins with a
    complete snapshot regardless of what the client sent. There is no
    delta stream to resume into and therefore no gap to mis-handle, which
    is why an unrecognised cursor is reported and not refused.

    The status exists so a client can tell "nothing changed while I was
    away" from "I have no idea what you are referring to" -- the first
    lets it skip a redraw, and the second tells it its cached revision is
    worthless.
    """
    if since_revision is None:
        return CURSOR_ABSENT
    if not isinstance(since_revision, str) or not since_revision:
        return CURSOR_UNRECOGNISED
    if since_revision == current_revision:
        return CURSOR_MATCHED
    return CURSOR_STALE


def world_builder_result_type(result_type: str) -> bool:
    return result_type == RESULT_TYPE_STATUS
