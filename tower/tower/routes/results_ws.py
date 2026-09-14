"""The WebSocket half of the cartridge result channel.

Kept out of `ws.py` on purpose. That module owns the frame path -- the
latency-measured, privacy-sensitive part of this system -- and the result
channel is a side surface that must be able to fail without implicating
it. Separate module, separate failure domain, and `ws.py` gains four small
dispatch branches rather than three hundred lines.

Every handler here returns without raising -- with ONE exception that is
not a result-channel problem at all. A malformed subscribe, an unknown
cartridge, a hostile payload: all become a `result_error` on the wire, and
the receive loop never learns the result channel had a problem, because the
receive loop is what answers frames. But a `WebSocketDisconnect` means the
SOCKET is gone, which is the receive loop's business and not a subscription
bug; it propagates, so the connection ends cleanly rather than being
swallowed here and re-surfacing as an uncaught `RuntimeError` from the next
`receive_json`. See `handle`.
"""

import asyncio
import logging

from fastapi import WebSocketDisconnect

from tower.results import registry
from tower.results.contracts import ENVELOPE_CONTRACT
from tower.results.publisher import (
    SNAPSHOT_TIMEOUT_SECONDS,
    LOCK_TIMEOUT_S,
    MAX_SUBSCRIPTIONS_PER_CONNECTION,
    SEND_TIMEOUT_S,
    ConnectionChannel,
    Subscription,
    classify_cursor,
)

logger = logging.getLogger(__name__)

# Client -> Tower
MSG_CARTRIDGES = "cartridges"
MSG_SUBSCRIBE = "result_subscribe"
MSG_UNSUBSCRIBE = "result_unsubscribe"

RESULT_MESSAGE_TYPES = frozenset({MSG_CARTRIDGES, MSG_SUBSCRIBE, MSG_UNSUBSCRIBE})

# Tower -> client
MSG_SUBSCRIBED = "result_subscribed"
MSG_UNSUBSCRIBED = "result_unsubscribed"
MSG_ERROR = "result_error"

# Error reasons. A closed set: a client switches on these, so adding one
# is a contract change.
ERR_MALFORMED = "malformed_request"
ERR_UNKNOWN_CARTRIDGE = "unknown_cartridge"
ERR_UNKNOWN_RESULT_TYPE = "unknown_result_type"
ERR_CONTRACT_MISMATCH = "contract_mismatch"
ERR_UNAVAILABLE = "cartridge_unavailable"
ERR_TOO_MANY = "too_many_subscriptions"
ERR_UNKNOWN_SUBSCRIPTION = "unknown_subscription"
ERR_SNAPSHOT_FAILED = "snapshot_failed"

# How much of a client-supplied identifier comes back in a refusal.
#
# `cartridge`, `result_type` and `subscription_id` are echoed into both a
# message string and a field, and were bounded by nothing. MEASURED at
# exactly 2.00x and unbounded: a 1,000,000-character `cartridge` produced
# a 2,000,311-character reply.
#
# It costs more than its size. These replies are sent while holding the
# send lock the FRAME PATH shares, so an oversized echo is paid by every
# `frame_result` queued behind it -- which is the starvation
# `CARTRIDGE-RESULTS.md` forbids in Tower responsibility #3.
#
# 120 to match the two guards that already exist for exactly this, in the
# same words: `routes/ws.py: _echo_safe` ("the alternative is letting a
# remote party choose the size of our messages") and `cv_lab/lab.py:
# _clip` ("A remote party must not be able to choose the size of a message
# this Tower sends"). Not imported from either: `ws.py`'s helper passes
# numbers through untouched because it bounds a numeric `seq`, and
# `cv_lab`'s lives inside a cartridge that this module is forbidden to
# import -- `test_the_result_channel_core_is_cartridge_blind` is what
# keeps the result-channel core cartridge-blind, and reaching through it
# for a string helper would honour the letter of that rule while breaking
# it.
ECHO_LIMIT = 120


def _echo_safe(value) -> str:
    """A client-supplied identifier on its way back out, bounded.

    Always a string: these three fields are typed as strings by the
    contract, an ill-formed client can send anything, and the refusal has
    to name what arrived so a person can see their own typo.
    """
    text = str(value)
    if len(text) <= ECHO_LIMIT:
        return text
    return text[: ECHO_LIMIT - 1] + "…"


async def handle(message: dict, *, websocket, sender, channel_holder) -> None:
    """Dispatch one result-channel message.

    Returns without raising for every result-channel FAULT -- see the module
    docstring -- but lets a `WebSocketDisconnect` through, because that is
    the socket dying, not a subscription going wrong.
    """
    try:
        message_type = message.get("type")
        if message_type == MSG_CARTRIDGES:
            await _cartridges(websocket, sender)
        elif message_type == MSG_SUBSCRIBE:
            await _subscribe(message, websocket, sender, channel_holder)
        elif message_type == MSG_UNSUBSCRIBE:
            await _unsubscribe(message, sender, channel_holder)
    except WebSocketDisconnect:
        # NOT a result-channel fault, and the one thing the broad handler
        # below must not eat. A send or a subscribe on a socket the client
        # has already dropped raises this; swallowing it here leaves the
        # receive loop to discover the dead socket on its NEXT
        # `receive_json`, which raises a bare `RuntimeError: WebSocket is
        # not connected` that ws.py does not catch and uvicorn logs as
        # "Exception in ASGI application". Propagated, it reaches the
        # endpoint's own `except WebSocketDisconnect` and the connection
        # ends the way every other disconnect does.
        raise
    except Exception:
        # Deliberately broad, and deliberately swallowed after logging.
        # This handler is called from the frame-serving receive loop; an
        # escape here would end a connection that is successfully
        # answering frames because a status subscription went wrong.
        logger.exception(
            "[Tower][Results] handler failed for %r; the connection continues",
            message.get("type"),
        )


async def _cartridges(websocket, sender) -> None:
    """The capability declaration, identical to `GET /cartridges`.

    Served from the same `registry.declare()` so the two surfaces cannot
    drift. A test asserts they are byte-identical.
    """
    await sender.send(registry.declare(**_declaration_inputs(websocket)))


async def _subscribe(message, websocket, sender, channel_holder) -> None:
    cartridge = message.get("cartridge")
    result_type = message.get("result_type")
    if not isinstance(cartridge, str) or not isinstance(result_type, str):
        await _error(
            sender,
            ERR_MALFORMED,
            "result_subscribe requires string 'cartridge' and 'result_type'",
        )
        return

    # Bounded HERE, once, rather than at each of the eight sites that echo
    # them. A guard applied per call site is a guard someone adds a ninth
    # call site next to, and the ninth one is the hole -- this function
    # already echoes `cartridge` and `result_type` into six different
    # refusals, and an audit that listed the sites missed
    # `requested_contract` below.
    #
    # Safe to rebind before the lookup rather than only on the way out:
    # every cartridge and result type this Tower serves is a short
    # identifier from a closed set in `contracts.py`, so truncation can
    # only ever affect a value that was already going to be refused. A
    # name long enough to be clipped is not a name `find_offer` knows.
    cartridge = _echo_safe(cartridge)
    result_type = _echo_safe(result_type)

    world_id = message.get("world_id")
    session_id = message.get("session_id")
    if world_id is not None and not isinstance(world_id, str):
        await _error(sender, ERR_MALFORMED, "'world_id' must be a string or absent")
        return
    if session_id is not None and not isinstance(session_id, str):
        await _error(sender, ERR_MALFORMED, "'session_id' must be a string or absent")
        return

    # These two are bounded HERE rather than with the pair above only
    # because they are read here -- and that gap is exactly the failure
    # the comment above predicts. The first version of this guard bound
    # four fields at the top of the handler and these two were declared
    # eleven lines further down, checked for TYPE and echoed verbatim.
    #
    # They are the WORSE half, because a subscribe carrying them
    # SUCCEEDS. Measured: 2,000,000 characters in, 4,001,438 bytes back --
    # the same 2.00x -- and then the string persists on the
    # `Subscription` and inside `Subscription.target`, so `poll_once`
    # re-serialises it every 0.5 s for the life of the subscription,
    # across the send lock the frame path shares. Eight per connection.
    #
    # `None` survives as `None`: absent is a meaningful value on both, and
    # `_echo_safe` would turn it into the string "None".
    #
    # Truncating before use is safe for the same reason it is above: a
    # world id is a 32-character hex string, so a value long enough to be
    # clipped is one no world was ever going to match.
    world_id = None if world_id is None else _echo_safe(world_id)
    session_id = None if session_id is None else _echo_safe(session_id)

    inputs = _declaration_inputs(websocket)
    offer = registry.find_offer(
        inputs["world_root"],
        cartridge,
        result_type,
        document_root=inputs["document_root"],
        scene_enabled=inputs["scene_enabled"],
        # The offer's `unavailable_reason` goes on the wire below, so this
        # surface must be told the same thing `/cartridges` was.
        scene_unavailable_reason=inputs["scene_unavailable_reason"],
        cv_lab=inputs["cv_lab"],
    )
    if offer is None:
        known = registry.known_cartridges(
            inputs["world_root"],
            document_root=inputs["document_root"],
            scene_enabled=inputs["scene_enabled"],
            cv_lab=inputs["cv_lab"],
        )
        if cartridge not in known:
            await _error(
                sender,
                ERR_UNKNOWN_CARTRIDGE,
                f"this Tower offers no contract for cartridge {cartridge!r}",
                cartridge=cartridge,
                result_type=result_type,
                offered=sorted(known),
            )
        else:
            await _error(
                sender,
                ERR_UNKNOWN_RESULT_TYPE,
                f"cartridge {cartridge!r} offers no result type {result_type!r}",
                cartridge=cartridge,
                result_type=result_type,
            )
        return

    requested = message.get("contract")
    if requested is not None and requested != offer["contract"]:
        # Compared for EQUALITY and nothing else. iOS holds contract
        # identifiers opaque, so a mismatch is not "older" or "newer" --
        # it is "we are not talking about the same agreement", and the
        # only safe answer is to refuse rather than to serve a payload
        # the client will decode under different rules.
        await _error(
            sender,
            ERR_CONTRACT_MISMATCH,
            "this Tower serves a different contract for that result type",
            cartridge=cartridge,
            result_type=result_type,
            offered_contract=offer["contract"],
            # Client-supplied and echoed, exactly like the two above.
            requested_contract=_echo_safe(requested),
        )
        return

    if not offer["available"]:
        await _error(
            sender,
            ERR_UNAVAILABLE,
            offer["unavailable_reason"],
            cartridge=cartridge,
            result_type=result_type,
            contract=offer["contract"],
        )
        return

    channel = channel_holder.ensure(websocket, sender)
    if channel.subscription_count >= MAX_SUBSCRIPTIONS_PER_CONNECTION:
        await _error(
            sender,
            ERR_TOO_MANY,
            f"a connection may hold at most {MAX_SUBSCRIPTIONS_PER_CONNECTION} "
            "subscriptions",
            cartridge=cartridge,
            result_type=result_type,
        )
        return

    since = message.get("since_revision")
    subscription = Subscription(
        subscription_id=channel.next_subscription_id(),
        cartridge=cartridge,
        result_type=result_type,
        contract=offer["contract"],
        world_id=world_id,
        session_id=session_id,
        # Provisional: replaced below once the first snapshot is known,
        # because "stale" versus "matched" can only be decided against a
        # revision we have actually computed.
        cursor_status=None,
    )

    hub = websocket.app.state.result_hub
    # The first snapshot is computed HERE, synchronously with the reply,
    # rather than waiting for the next poll. A subscriber that had to wait
    # up to a poll interval to learn anything would make reconnection feel
    # broken, and the whole contract rests on "a subscription always
    # begins with a complete snapshot".

    try:
        # THE SAME DEADLINE THE POLL LOOP HAS, because this runs inline in
        # the connection's message loop: while it waits, nothing else on
        # this socket is answered. A reviewer injected a 3 s stall and
        # watched a frame sent behind a subscribe wait the full 3 s for
        # its `frame_result`, and iOS sends `result_subscribe` on every
        # reconnect of the World Builder screen. A wedged read here hung
        # that phone's socket outright. `TimeoutError` is an `Exception`,
        # so it takes the reply below rather than leaving the client
        # waiting on an answer that never comes. The thread outlives the
        # cancel -- bounded at one per subscribe attempt, which is
        # client-driven and not a 2 Hz loop.
        # Through the hub's in-flight table, NOT a thread of this call's
        # own. See `ResultHub.first_snapshot` for the measurement: a
        # wedged read plus iOS's 2 s stall timeout minted one thread per
        # reconnect and exhausted the executor in 60 s.
        snapshot = await hub.first_snapshot(
            subscription, timeout=SNAPSHOT_TIMEOUT_SECONDS
        )
    except Exception as exc:
        # A subscribe that cannot produce its first snapshot must SAY so.
        # The outer handler would have logged this and returned, leaving
        # the client waiting on a reply that was never coming -- the
        # silent no-op IOS-to-Tower.md 2.2 rules out, and the worst of the
        # available failures because nothing on either side reports it.
        if isinstance(exc, TimeoutError):
            # One line. A phone retrying every ~2.5 s against a wedged
            # read logged a full traceback per attempt -- a reviewer
            # counted 216 lines a minute, indefinitely.
            logger.warning(
                "[Tower][Results] could not build the first snapshot for "
                "%s/%s: %s",
                cartridge, result_type, exc,
            )
        else:
            logger.exception(
                "[Tower][Results] could not build the first snapshot for %s/%s",
                cartridge,
                result_type,
            )
        await _error(
            sender,
            ERR_SNAPSHOT_FAILED,
            f"the Tower could not read this cartridge's state: "
            f"{type(exc).__name__}",
            cartridge=cartridge,
            result_type=result_type,
            contract=offer["contract"],
        )
        return
    subscription.cursor_status = classify_cursor(since, snapshot.revision)

    await sender.send(
        {
            "type": MSG_SUBSCRIBED,
            "envelope_contract": ENVELOPE_CONTRACT,
            "subscription_id": subscription.subscription_id,
            "cartridge": cartridge,
            "result_type": result_type,
            "contract": offer["contract"],
            "snapshot_only": offer["snapshot_only"],
            "world_id": world_id,
            "session_id": session_id,
            "cursor_status": subscription.cursor_status,
        }
    )

    await channel.add(subscription)
    # Delivered through the same path every later result takes, so the
    # first snapshot is not a special case a client has to decode twice.
    subscription.offer(snapshot)
    channel._wakeup.set()
    # AFTER the subscription exists, so the session it may start is one
    # somebody is already listening to. This is the phone saying "show
    # me the scene", and for Scene Understanding it is what starts the
    # detector -- see `tower/scene/live.py`, WHEN IT RUNS.
    channel_holder.watcher_joined(cartridge, subscription.subscription_id)


async def _unsubscribe(message, sender, channel_holder) -> None:
    subscription_id = message.get("subscription_id")
    if not isinstance(subscription_id, str):
        await _error(
            sender, ERR_MALFORMED, "result_unsubscribe requires 'subscription_id'"
        )
        return
    # Same rebinding as `_subscribe`, same reason. Ids are minted by
    # `next_subscription_id()` and are short, so a value long enough to be
    # clipped is one `remove()` was never going to match.
    subscription_id = _echo_safe(subscription_id)
    channel = channel_holder.existing()
    removed = False
    cartridge = None
    if channel is not None:
        existing = channel.get(subscription_id)
        cartridge = None if existing is None else existing.cartridge
        removed = await channel.remove(subscription_id)
    if not removed:
        await _error(
            sender,
            ERR_UNKNOWN_SUBSCRIPTION,
            f"no open subscription with id {subscription_id!r}",
            subscription_id=subscription_id,
        )
        return
    await sender.send(
        {"type": MSG_UNSUBSCRIBED, "subscription_id": subscription_id}
    )
    if cartridge is not None:
        await channel_holder.watcher_left(cartridge, subscription_id)


async def _error(sender, reason: str, message: str, **extra) -> None:
    payload = {
        "type": MSG_ERROR,
        "envelope_contract": ENVELOPE_CONTRACT,
        "reason": reason,
        "message": message,
    }
    payload.update(extra)
    await sender.send(payload)


def _declaration_inputs(websocket) -> dict:
    """What this Tower's declaration depends on, off one app state.

    Delegated to `registry.declaration_inputs` rather than reading the
    attributes here, so this surface and `/cartridges` over HTTP cannot
    come to disagree about what "configured" means. That byte-identity is
    asserted by a test and is the reason the helper exists at all.
    """
    return registry.declaration_inputs(websocket.app.state)


async def _off_loop_even_if_cancelled(function, *args, **kwargs) -> None:
    """Run `function` on a thread; if this task is cancelled meanwhile,
    let the thread finish on its own and re-raise.

    A connection teardown must complete whether or not the handler task
    survives it. `to_thread` alone abandons the await on cancellation and
    the work with it -- the thread keeps running, but a SECOND
    cancellation (a test client's teardown delivers several) can land
    before the thread was even started. Starting a plain thread first
    guarantees the work happens; awaiting its completion is best effort.
    """
    import threading

    done = threading.Event()

    def run():
        try:
            function(*args, **kwargs)
        except Exception:
            logger.exception("[Tower][Results] teardown hook failed")
        finally:
            done.set()

    threading.Thread(target=run, name="tower-results-teardown", daemon=True).start()
    try:
        await asyncio.to_thread(done.wait, 30.0)
    except asyncio.CancelledError:
        raise


class ChannelHolder:
    """Lazily creates one ConnectionChannel per WebSocket.

    Lazy because the overwhelming majority of connections -- every current
    iOS build, and every test in this repository predating this work --
    never subscribe to anything. Those connections must pay nothing: no
    task, no event, no registration with the shared reader.
    """

    __slots__ = ("_channel", "_clock", "owner", "_live")

    def __init__(self, clock, *, owner=None, live=None) -> None:
        self._channel = None
        self._clock = clock
        # The connection's identity, the same token `ws.py` hands the
        # recorder and the live cartridges. Demand is reported per
        # subscription and released per connection, so both need it.
        self.owner = owner
        # The `LiveCartridges` demand surface, or None on a Tower with
        # no live cartridge. Handed in rather than read off `app.state`
        # so a test can construct a holder without an app.
        self._live = live

    def ensure(self, websocket, sender) -> ConnectionChannel:
        if self._channel is None:

            async def _send(payload):
                # Bounded on BOTH waits. The push task shares the
                # connection's send lock with the frame path, so an
                # unbounded lock wait here would let a slow frame consume
                # a result's budget and drop a subscription that was
                # never actually offered to the socket.
                await sender.send_bounded(
                    payload,
                    lock_timeout=LOCK_TIMEOUT_S,
                    send_timeout=SEND_TIMEOUT_S,
                )

            self._channel = ConnectionChannel(
                websocket.app.state.result_hub, _send, self._clock
            )
        return self._channel

    def existing(self):
        return self._channel

    def watcher_token(self, subscription_id: str) -> tuple:
        return (self.owner, subscription_id)

    def watcher_joined(self, cartridge: str, subscription_id: str) -> None:
        if self._live is not None:
            self._live.watcher_joined(
                cartridge, self.watcher_token(subscription_id), owner=self.owner
            )

    async def watcher_left(self, cartridge: str, subscription_id: str) -> None:
        # OFF the event loop: the last watcher leaving reaches
        # `LiveSession.stop()` and its bounded join, for the same reason
        # `ws.py` runs `stream_closed` through `to_thread`.
        if self._live is not None:
            await asyncio.to_thread(
                self._live.watcher_left,
                cartridge,
                self.watcher_token(subscription_id),
            )

    async def close(self) -> None:
        channel, self._channel = self._channel, None
        if channel is None:
            # Never subscribed, so never a watcher. Returning without an
            # await keeps this connection's teardown synchronous, which
            # `ConnectionTracker` relies on: a superseded connection's
            # teardown must finish before the next one is measured.
            return
        try:
            await channel.close()
        finally:
            # In a `finally`, because `ConnectionChannel.close` re-raises
            # a cancellation that is aimed at the connection handler
            # itself -- and a connection that is being cancelled has
            # still gone. Its watchers leave with it either way; a
            # session kept running for a subscription whose socket is
            # closed would be the leak this method exists to prevent.
            if self._live is not None:
                await _off_loop_even_if_cancelled(
                    self._live.watchers_left, owner=self.owner
                )
