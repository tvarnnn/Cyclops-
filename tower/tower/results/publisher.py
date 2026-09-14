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
import threading
import time

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
# A snapshot still running this many deadlines after it was dispatched is
# ABANDONED: forgotten by the hub, so the next pass dispatches the target
# afresh, on the chance the fault has cleared. The abandoned thread is a
# daemon and finishes, or does not, on its own. See `_dispatch`.
SNAPSHOT_ABANDON_MULTIPLIER = 3
# The cap doubles per consecutive abandonment of one target, up to this.
# 120 s, not 600: this is also the longest a target stays refused after a
# fault CLEARS, because recovery needs an abandonment (a reviewer measured
# 592 s of silent refusal at 600).
SNAPSHOT_ABANDON_MAX_SECONDS = 120.0
# The most distinct targets that may have a snapshot running at once. A
# target is `(cartridge, result_type, world_id, session_id)` and the last
# two are the client's to choose: a reviewer subscribed 400 distinct
# world_ids against a wedged read and got 400 live threads in 0.3 s, none
# of them ever swept because a failed subscribe registers nothing and so
# starts no poll loop. Eight subscriptions per connection, a handful of
# connections: 64 is generous for a phone and a wall for anything else.
MAX_IN_FLIGHT_TARGETS = 64

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


class TooManyTargetsInFlight(TimeoutError):
    """A subscribe refused because `MAX_IN_FLIGHT_TARGETS` snapshots are
    already running. A `TimeoutError` so the route answers it the same
    way: `snapshot_failed`, one log line."""


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
        age_clock=time.monotonic,
    ) -> None:
        self._snapshot_for = snapshot_for
        self._clock = clock
        # Every DURATION here -- how long a snapshot has been running, the
        # subscribe's remaining wait, the abandonment cap -- is measured on
        # this clock, never on `clock`. `build_hub` hands in `time.time`,
        # which a reviewer stepped: 600 s backwards left a wedged target
        # unreachable by the cap for the length of the step, with the
        # full inline wait restored on every subscribe; 10 s forwards aged
        # every in-flight snapshot past its deadline at once and dropped
        # every subscriber three passes later. `clock` is for the
        # heartbeat and for what the client is told; this is for ages.
        self._age_clock = age_clock
        # A sub-millisecond poll is a hot loop, not a setting.
        self._poll_seconds = max(float(poll_seconds), 0.001)
        self._heartbeat_seconds = heartbeat_seconds
        self._channels: set = set()
        self._task: asyncio.Task | None = None
        # Bounded by the number of live targets, and pruned every pass.
        self._failures: dict = {}
        # One running snapshot per target, at most. See `poll_once`.
        self._in_flight: dict = {}
        # When each in-flight snapshot was dispatched, by target, so a
        # pass can tell "still running past the deadline" from "started
        # a moment ago by someone else".
        self._dispatched_at: dict = {}
        # Passes do not interleave. See `poll_once`.
        self._pass_lock = asyncio.Lock()
        # How many times in a row a target's snapshot has been abandoned,
        # for the backoff in `_past_abandonment`. Cleared on a delivery.
        self._abandon_streak: dict = {}
        # The subscription ids watching each target when its in-flight
        # snapshot was dispatched. A result whose watchers have ALL gone
        # was computed for nobody who is still here. See the discard rule
        # in `_poll_once_locked`.
        self._watchers_at_dispatch: dict = {}
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
            # And forget what the cancelled pass would have collected. A
            # snapshot that finishes now finishes for nobody: a done
            # future left here was handed to the NEXT subscription as its
            # first snapshot, computed before that phone connected -- a
            # reviewer measured 4.0 s old, against a newer revision on
            # disk, on a real socket. A still-running one stays, so the
            # target is never computed twice at once; its watchers are
            # cleared, so the pass that finds it done discards it unless
            # a subscribe has joined it since.
            for target in list(self._in_flight):
                if self._in_flight[target].done():
                    self._forget(target)
                else:
                    self._watchers_at_dispatch[target] = set()

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
                started = time.monotonic()
                await self.poll_once()
                # The pass may have waited up to `poll_seconds` for its
                # own fresh futures; that wait counts toward the interval,
                # so the cadence is the interval and not "pass + poll".
                # A refinement, not the fix: the fix for the 20%-cadence
                # measurement is the pass's budget (see `_poll_once_locked`),
                # and a mutation that restores the full sleep here is NOT
                # caught by the cadence test, because a slow future is
                # fresh -- and waited on -- only on the pass that
                # dispatched it.
                elapsed = time.monotonic() - started
                # Never less than 20 ms: a pass that takes longer than
                # the poll must not turn the loop into a spin. The first
                # floor was a quarter of the interval, which at a 1 ms
                # test poll is a quarter of a millisecond -- 6,000 passes
                # a second, measured. An absolute floor, not a fraction
                # of the thing being floored -- and ABOVE THE EVENT LOOP'S
                # CLOCK RESOLUTION, which on Windows is 15.6 ms: a timer
                # due inside that resolution is treated as already due
                # whenever the loop has anything else ready, and the
                # snapshot threads' deliveries keep it ready, so a 10 ms
                # floor measured 0.17 ms.
                await asyncio.sleep(max(0.02, self._poll_seconds - elapsed))
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

    async def first_snapshot(self, subscription, *, timeout: float):
        """The snapshot a new subscription is answered with, WITHOUT a new thread
        if one is already computing it.

        `results_ws` used to run its own `asyncio.to_thread` here under a
        `wait_for`. The deadline stopped the socket hanging, and left the
        thread running -- "bounded at one per subscribe attempt, which is
        client-driven". It is client-driven at the fastest rate the client
        has: iOS's `sendStallTimeout` is 2 s, so while the first snapshot
        stalls the phone replaces the socket and re-subscribes every ~2.5 s,
        and each attempt minted a thread. A reviewer drove that against a
        wedged read: **24 executor threads in 60 s, then `stream_start`
        never answered for any phone** -- the exact exhaustion the poll
        loop had just been cured of, one route over.

        Sharing `_in_flight` bounds it at one thread per target across
        BOTH paths. A subscribe that finds a snapshot already running waits
        on that one, for whatever is LEFT of its deadline; if it is still
        running the caller gets `TimeoutError` and the future stays where
        it is for the next pass to collect. Nothing is cancelled and
        nothing is duplicated.

        Two things it must NOT do, both measured by the next reviewer:

        * Wait on a future that is already past its deadline. This runs
          inline in the connection's message loop, and every subscribe to
          a wedged target was sitting here for the full 10 s -- six
          consecutive subscribes, all timing out, three of them after the
          fault had cleared -- while the frames behind it went
          unanswered. A future older than the deadline is a known wedge;
          the answer is immediate.
        * Hand over a future that is already DONE. Nobody collected it,
          so it was computed for a subscription that has gone -- a phone
          that dropped mid-pass -- and it is stale by construction: a
          reconnected phone was given a revision 4.0 s old while disk
          held a newer one, and told its own current cursor was "stale".
          It is discarded and the target dispatched afresh.
        """
        target = subscription.target
        # The same sweep the pass runs, because a target nobody has
        # managed to subscribe to has no channel, no poll loop and so no
        # pass: without this, wedged futures minted by failed subscribes
        # were never abandoned at all.
        abandoned_here = False
        for other in list(self._in_flight):
            other_future = self._in_flight[other]
            if not other_future.done() and self._past_abandonment(other):
                self._abandon(other)
                if other == target:
                    abandoned_here = True
        future = self._in_flight.get(target)
        if future is not None and future.done():
            # DONE AND UNCOLLECTED. Stale if it is OLD -- a reviewer
            # measured 4.0 s, behind the revision on disk, computed for a
            # phone that had dropped mid-pass -- and the freshest state
            # there is if it is YOUNG: a 2.1 s read that finished 0.4 s
            # ago, for a phone whose socket stalled at 2 s and came back
            # to re-subscribe. Discarding that one meant dispatching
            # afresh, stalling again, and never subscribing: 0 of 8 on a
            # real socket at a healthy 2.1 s read. The line is the
            # heartbeat, which is how old a snapshot the channel itself
            # is content to send.
            age = self._age_clock() - self._dispatched_at.get(target, self._age_clock())
            if age <= max(self._poll_seconds, self._heartbeat_seconds):
                self._forget(target)
                self._abandon_streak.pop(target, None)
                return future.result()
            self._forget(target)
            future = None
        if future is not None and self._past_abandonment(target):
            # A subscribe is a chance to try the target afresh, and it
            # was the only chance: once the escalation has failed every
            # watcher off a wedged target, no pass looks at it again, so
            # the abandon sweep in the pass cannot reach it. The first
            # version of the cap sat only in the pass -- a reviewer
            # counted 33 re-subscribes over 100 s against a future 95 s
            # old, every one refused in 0.00 s. Poisoned for the life of
            # the process, with the fix for that present and unreachable.
            self._abandon(target)
            future = None
            abandoned_here = True
        if future is None:
            if (
                target not in self._in_flight
                and len(self._in_flight) >= MAX_IN_FLIGHT_TARGETS
            ):
                raise TooManyTargetsInFlight(
                    f"{len(self._in_flight)} results are already being "
                    f"computed; not starting one for {target}"
                )
            future = self._dispatch(
                target, subscription, self._watchers_of(target) | {subscription}
            )
        else:
            self._watchers_at_dispatch.setdefault(target, set()).add(subscription)
        age = self._age_clock() - self._dispatched_at.get(target, self._age_clock())
        # THE INLINE WAIT IS THE SNAPSHOT DEADLINE, and a round of this
        # campaign that capped it lower is why this comment exists. This
        # runs in the connection's message loop; while it waits, no frame
        # on that socket is answered, and iOS replaces a socket that has
        # stalled 2 s. That looks like a reason to answer within 2 s --
        # and a 1.5 s cap was shipped, measured, and reverted: on the
        # phone, `snapshot_failed` is TERMINAL (`.failed`, no retry; only
        # `channel_failed` is retried), so every healthy read slower than
        # the cap left the World Builder screen dead for the life of the
        # connection -- 0 of 8 subscribes at a 2.1 s read, which is this
        # file's own number for a manifest read under a disk fault. The
        # stall, by contrast, self-heals: the phone replaces the socket,
        # the re-subscribe joins the same future, and is answered when it
        # lands. So the wait is the deadline; what must not happen is
        # waiting it MORE than once per fault, and that is the rule
        # below: a subscribe that itself abandoned a wedge does not wait
        # on its own replacement at all (the abandon reset the age, and
        # the first version then waited the full deadline again at every
        # cap interval -- 25 s of every 30 blocked, measured).
        waited = 0.0
        remaining = min(timeout, SNAPSHOT_TIMEOUT_SECONDS - age)
        if abandoned_here:
            remaining = 0.0
        if remaining > 0:
            started = self._age_clock()
            await asyncio.wait({future}, timeout=remaining)
            waited = self._age_clock() - started
        if not future.done():
            # This subscription will not be registered: take it back out
            # of the watcher set, or every failed subscribe to a wedged
            # target leaves a dead `Subscription` there for as long as
            # the future lives (a reviewer counted 34 in 100 s). Only
            # THIS one: the set was seeded with the target's live
            # watchers, and the first version clobbered it with just this
            # subscription and then emptied it here, so the snapshot a
            # registered watcher was waiting for was discarded as
            # "computed for nobody".
            watchers = self._watchers_at_dispatch.get(target)
            if watchers is not None:
                watchers.discard(subscription)
            if abandoned_here:
                detail = "was abandoned and dispatched afresh; not waited on"
            elif remaining <= 0:
                detail = f"is {age:.0f}s old, past the {SNAPSHOT_TIMEOUT_SECONDS:.0f}s deadline"
            else:
                detail = f"is still running after {waited:.1f}s"
            raise TimeoutError(f"the first snapshot for {target} {detail}")
        if self._in_flight.get(target) is future:
            self._forget(target)
        self._abandon_streak.pop(target, None)
        return future.result()

    def _watchers_of(self, target) -> set:
        """Every registered subscription watching `target`, across channels."""
        watchers: set = set()
        for channel in self._channels:
            for subscription in channel._subscriptions.values():
                if subscription.target == target:
                    watchers.add(subscription)
        return watchers

    def _abandonment_cap(self, target) -> float:
        """How long this target's snapshot may run before it is abandoned.

        Doubles with every consecutive abandonment, up to
        `SNAPSHOT_ABANDON_MAX_SECONDS`: a read that stays wedged is
        retried at 30 s, 60 s, 120 s ... rather than minting a new daemon
        thread every 30 s for as long as the fault lasts (a reviewer
        counted one per 30 s, 120 an hour, each holding the wedged
        handle). A delivery clears the streak.
        """
        streak = self._abandon_streak.get(target, 0)
        cap = SNAPSHOT_ABANDON_MULTIPLIER * SNAPSHOT_TIMEOUT_SECONDS * (2 ** streak)
        return min(cap, SNAPSHOT_ABANDON_MAX_SECONDS)

    def _past_abandonment(self, target) -> bool:
        age = self._age_clock() - self._dispatched_at.get(target, self._age_clock())
        return age >= self._abandonment_cap(target)

    def _abandon(self, target) -> None:
        """Forget a snapshot that has been running past its abandonment cap.
        Its thread may still return, into a future nothing references any
        more -- whose exception, if it raises, is retrieved by the callback
        so asyncio does not log it as never retrieved."""
        age = self._age_clock() - self._dispatched_at.get(target, self._age_clock())
        self._abandon_streak[target] = self._abandon_streak.get(target, 0) + 1
        logger.warning(
            "[Tower][Results] snapshot for %s has been running for %.0fs; "
            "giving up on that thread and trying the target afresh "
            "(abandoned %d time(s) in a row; next cap %.0fs)",
            target, age, self._abandon_streak[target], self._abandonment_cap(target),
        )
        future = self._in_flight.get(target)
        if future is not None and not future.done():
            future.add_done_callback(
                lambda f: None if f.cancelled() else f.exception()
            )
        self._forget(target)

    def _dispatch(self, target, sample, watchers):
        """Start ONE snapshot for `target` on a thread of its own, and record it.

        A plain daemon thread, not `asyncio.to_thread`, for two reasons a
        reviewer measured. The default executor is shared with the capture
        path -- `stream_start`, the disconnect teardown, `supervisor.shutdown`
        -- and a snapshot thread that never returns must not be able to
        take a worker from it (this is what turned one wedged read into a
        Tower that answered nobody, twice). And `asyncio.run` joins the
        default executor's threads on the way out: a Tower shut down with
        one wedged snapshot thread sat at "Application shutdown complete"
        for **300 s** before the interpreter gave up on the join. A daemon
        thread is joined by nobody. The result crosses back to the loop
        with `call_soon_threadsafe`, and is dropped on the floor if the
        loop has closed or the hub has since abandoned the future.
        """
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        def deliver(result, error):
            if future.cancelled():
                return
            if error is not None:
                future.set_exception(error)
            else:
                future.set_result(result)

        def run():
            try:
                result = self._snapshot_for(
                    sample.cartridge, sample.result_type,
                    sample.world_id, sample.session_id,
                )
            except Exception as exc:  # noqa: BLE001 -- crosses a thread
                result, error = None, exc
            except BaseException as exc:  # noqa: BLE001
                # A `SystemExit` or `KeyboardInterrupt` raised by a
                # producer must not cross the thread as itself: the
                # collector catches `Exception`, and a reviewer measured
                # the Tower process exiting with code 3 / 130 when it
                # did. It is a failure of that target, reported as one.
                result = None
                error = RuntimeError(
                    f"the producer raised {type(exc).__name__}"
                )
            else:
                error = None
            try:
                loop.call_soon_threadsafe(deliver, result, error)
            except RuntimeError:
                # The loop is closed: the Tower is gone, and so is
                # anyone who wanted this.
                pass

        threading.Thread(
            target=run, name=f"tower-result-snapshot", daemon=True
        ).start()
        self._in_flight[target] = future
        self._dispatched_at[target] = self._age_clock()
        self._watchers_at_dispatch[target] = set(watchers)
        return future

    def _forget(self, target) -> None:
        """Drop a target's in-flight record, retrieving a raised exception
        so asyncio does not log "Future exception was never retrieved"."""
        future = self._in_flight.pop(target, None)
        self._dispatched_at.pop(target, None)
        self._watchers_at_dispatch.pop(target, None)
        if future is not None and future.done() and not future.cancelled():
            future.exception()

    def _collect(self, target, future, moment: float) -> None:
        """A finished snapshot: freed, then offered or counted.

        THIS future is forgotten, not whatever the table holds for the
        target now. `first_snapshot` takes no pass lock, so a subscribe
        that ran while a pass was parked in its wait may have discarded
        this future's table entry and dispatched a replacement;
        forgetting by target popped the replacement, and the next pass
        dispatched a third -- two live threads for one target, measured
        by a reviewer, which is the one bound this table exists to hold.
        """
        if self._in_flight.get(target) is future:
            self._forget(target)
        elif not future.cancelled():
            future.exception()
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
                f"the Tower could not read this cartridge's state; "
                f"the last failure was {type(exc).__name__}",
            )
            return
        self._failures.pop(target, None)
        self._abandon_streak.pop(target, None)
        for channel in list(self._channels):
            channel.offer(
                target, snapshot, now=moment, heartbeat=self._heartbeat_seconds
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
        """One pass over every watched target. Passes never interleave.

        **A RACE THAT WITHHELD DELIVERIES, found by the suite.** The
        background loop and a forced pass (`pump()` in the tests, with
        the heartbeat set to zero so it MUST deliver) ran concurrently on
        one loop. The background pass dispatched a target's snapshot; the
        forced pass, arriving a moment later, found that future "already
        in flight", counted it a failure and returned without offering
        anything; the background pass then completed it and offered under
        the ordinary heartbeat rule, which sends nothing for an unchanged
        world. Net: a pass that had to deliver delivered nothing, and the
        waiting test hung the whole run for 33 minutes at
        `test_a_partially_deleted_world_reports_honestly_and_does_not_crash`.
        Before the in-flight table every pass owned its own thread, so the
        race could not exist; it is the table's cost, paid here with a
        lock. A pass waiting for the previous one to finish is cheap --
        passes are ~40 ms on the real root -- and correct: the second pass
        then finds the target free, or done, and does its own work.
        """
        async with self._pass_lock:
            await self._poll_once_locked()

    async def _poll_once_locked(self) -> None:
        """One pass: compute each distinct target once, offer it to all.

        Snapshot computation is pushed off the event loop onto a thread
        per dispatch (`_dispatch`), because it reads and JSON-parses files
        and the loop is also answering frames. A disk stall must cost
        this channel latency, never the frame path.
        """
        targets = {}
        for channel in self._channels:
            for subscription in channel._subscriptions.values():
                targets[subscription.target] = subscription

        # Forget the failure counts of targets nobody is watching any
        # more. `_failures` was pruned on success and on escalation, but
        # not when a subscription simply went away, so a target that
        # failed once and was then unsubscribed left an entry forever --
        # and, when the target came back, escalated it after ONE failure
        # instead of three (the first version pruned only when the dict
        # had outgrown the target set, which let a count survive a
        # reconnect). Every pass, unconditionally: it is a dict the size
        # of the live targets.
        #
        # It is small and it is genuinely unbounded: `Subscription.target`
        # includes the client-chosen `world_id` and `session_id`, so a
        # connection can mint distinct targets at will. Bounded now by the
        # live subscriptions rather than by the client's imagination.
        if self._failures:
            self._failures = {
                target: count
                for target, count in self._failures.items()
                if target in targets
            }
        if self._abandon_streak:
            self._abandon_streak = {
                target: streak
                for target, streak in self._abandon_streak.items()
                if target in targets or target in self._in_flight
            }

        # ONE SNAPSHOT IN FLIGHT PER TARGET, ALL TARGETS AT ONCE, EACH
        # OFFERED THE MOMENT IT LANDS, ONE DEADLINE FOR THE PASS.
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
        # The version after it waited for ALL of the pass's futures before
        # offering any, which turned sum into max and was not a fix: a
        # reviewer measured CV Lab, Object Memory and Document Memory at
        # 7 deliveries in 20 s beside a 2.1 s World Builder read -- 17.5%
        # of their cadence -- and a full 10 s blackout for every cartridge
        # on the pass that first meets a wedge. Each future is now offered
        # as it completes; the slow one costs only its own subscribers.
        #
        # So: a target that already has a snapshot running is NOT
        # dispatched again -- its future is kept in `_in_flight` and
        # simply checked -- which bounds the threads at one per target by
        # construction, however long a read wedges. Every target that is
        # free is dispatched, and the pass waits on this pass's own
        # futures with a bounded wait that does not cancel: a slow
        # snapshot survives into the next pass and is delivered when it
        # lands, as the freshest state there is. A target still pending
        # at the deadline counts one failure, and the existing escalation
        # tells its subscribers after three. A target still pending
        # `SNAPSHOT_ABANDON_MULTIPLIER` deadlines later is forgotten and
        # dispatched afresh next pass -- the version before this never
        # re-dispatched, so a thread that never returned poisoned its
        # target for the life of the process, and a fault that cleared
        # was never noticed.
        watchers_now: dict = {}
        for channel in self._channels:
            for subscription in channel._subscriptions.values():
                watchers_now.setdefault(subscription.target, set()).add(
                    subscription
                )
        for target in list(self._in_flight):
            if target not in targets and self._in_flight[target].done():
                # Nobody is watching this any more and its thread has
                # finished: forget it. A still-running thread for a
                # departed target stays until it finishes, so it can
                # never be dispatched twice.
                self._forget(target)
            elif (
                not self._in_flight[target].done()
                and self._past_abandonment(target)
            ):
                # ABANDONED -- watched or not. The first version swept
                # only the watched targets, and a wedged target has no
                # watchers by the time the cap is reached: the escalation
                # fails them off after three deadlines (~11 s), the cap is
                # three deadlines PLUS (30 s), and nothing in between
                # looks at an unwatched pending future. Dead code, and
                # the target stayed poisoned.
                self._abandon(target)

        now = self._clock()
        waiting: dict = {}
        fresh: set = set()
        for target, sample in targets.items():
            future = self._in_flight.get(target)
            if (
                future is not None
                and future.done()
                and (
                    self._watchers_at_dispatch.get(target, set())
                    & watchers_now.get(target, set())
                )
            ):
                # CARRIED OVER, DONE, STILL WATCHED: collected now AND
                # replaced now. Finished after its own pass's budget,
                # for a subscriber who is still here -- the freshest
                # state there is. The version before this collected it
                # and did not dispatch its replacement until the NEXT
                # pass, so a healthy-but-slow target lost a whole poll
                # interval per delivery: a reviewer measured a 0.55 s
                # read and a 0.80 s read both delivering every 1.52 s.
                self._collect(target, future, now)
                future = None
            elif (
                future is not None
                and future.done()
            ):
                # DONE, AND EVERYONE IT WAS COMPUTED FOR HAS GONE: discarded.
                #
                # The test is the WATCHERS, not "did a pass run while nobody
                # watched" -- no pass may run in that gap at all (the phone
                # drops and returns between two polls), and the first
                # version of this rule missed exactly that and handed a
                # reconnected phone the stale result. And not merely
                # `done()` either: a target watched by the same subscriber
                # throughout, whose snapshot simply finished after its
                # pass's deadline, is the freshest state there is; the
                # version before that discarded it and re-dispatched -- one
                # extra producer entry, caught by the leak test's count.
                #
                # The watchers are the `Subscription` OBJECTS, not their
                # ids: ids are minted per connection, so every phone's
                # first subscription is "sub-1", and a reconnected phone's
                # "sub-1" matched the dead socket's "sub-1" -- a false
                # "still watching" for exactly the case this rule exists
                # for. A reviewer parametrised the shipped test with the
                # reused id and it failed.
                #
                # This is a snapshot computed for a subscription that has
                # since gone -- the target left `targets` for a pass while
                # its thread was still running, and came back. Delivering
                # it sent a stale revision to a NEW subscription: a
                # reviewer measured `rev 1` then **`rev 0`**, 7.5 s old,
                # on a reconnected phone's default target, which is the
                # one every phone subscribes to. Age was unbounded, not
                # "at most one poll" as the previous comment claimed.
                self._forget(target)
                future = None
            if future is None:
                future = self._dispatch(
                    target, sample, watchers_now.get(target, set())
                )
                fresh.add(future)
            waiting[future] = target

        offered: set = set()

        def collect(done_futures) -> None:
            moment = self._clock()
            for future in done_futures:
                offered.add(future)
                self._collect(waiting[future], future, moment)

        # ONLY THE FUTURES DISPATCHED THIS PASS are waited on. A
        # carried-over future that is already past the deadline is not
        # waited on again: the previous version put it back into the wait
        # every pass, so one wedged target held EVERY pass at the full
        # deadline until its subscribers were told -- a reviewer measured
        # the healthy cartridges dropping from ~160 deliveries to 38 in
        # 16 s. A wedge now costs the others nothing after its first
        # pass; it is checked with `done()` and counted below.
        #
        # And each is offered AS IT LANDS: `FIRST_COMPLETED`, in a loop,
        # against a wall-clock budget (the hub's own clock may be a test's
        # frozen one, and a frozen clock must not make this loop wait the
        # full deadline again and again).
        # The budget is the POLL INTERVAL, not the snapshot deadline: a
        # pass that waits for its slowest future waits for the slowest
        # target, and with the loop's sleep on top of it every cartridge
        # ran at "slowest + poll" -- 20% of cadence beside a 2.1 s read,
        # and a 10 s blackout for everyone on the first pass to meet a
        # wedge, both measured twice. A future that outlives the budget
        # is simply carried into the next pass, where it is collected
        # done or counted against the deadline by its age.
        pending = set(fresh)
        budget_ends = time.monotonic() + min(
            self._poll_seconds, SNAPSHOT_TIMEOUT_SECONDS
        )
        while pending:
            remaining = budget_ends - time.monotonic()
            if remaining <= 0:
                break
            done, pending = await asyncio.wait(
                pending, timeout=remaining, return_when=asyncio.FIRST_COMPLETED
            )
            collect(done)

        now = self._age_clock()
        for future, target in waiting.items():
            if future in offered or future.done():
                continue
            # ONLY ONCE IT HAS ACTUALLY AGED PAST THE DEADLINE. A future
            # dispatched moments ago by a subscribe (`first_snapshot`)
            # or by the previous pass is not a wedge, and the first
            # version counted it as one -- logging "has exceeded
            # 10.0s" six milliseconds after the connection opened.
            # Three subscribes during three passes would have told a
            # healthy target's subscribers it had failed.
            age = now - self._dispatched_at.get(target, now)
            if age < SNAPSHOT_TIMEOUT_SECONDS:
                continue
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
