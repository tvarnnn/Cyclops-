"""Resources, slow consumers, cleanup, and failure containment.

The properties that decide whether this channel is safe to leave running,
as opposed to whether it produces the right JSON.
"""

import asyncio

import pytest

from tests.result_channel_fixtures import (  # noqa: F401
    _close_result_channel_clients,
    build_world,
    drain,
    make_client,
    pump,
    subscribe,
)
from tower.results.publisher import (
    MAX_SUBSCRIPTIONS_PER_CONNECTION,
    ConnectionChannel,
    ResultHub,
    Subscription,
)
from tower.results.envelope import Snapshot


@pytest.fixture(scope="module")
def world_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("worlds")
    build_world(root, frames=10)
    return root


def _snapshot(revision: str) -> Snapshot:
    return Snapshot(payload={"revision_marker": revision}, revision=revision)


def _subscription() -> Subscription:
    return Subscription(
        subscription_id="sub-1",
        cartridge="world_builder",
        result_type="status",
        contract="c",
        world_id=None,
        session_id=None,
        cursor_status=None,
    )


# -- bounded memory -----------------------------------------------------


def test_a_subscription_holds_exactly_one_snapshot_however_many_arrive():
    """There is no queue, and this is what that means concretely.

    A hundred snapshots offered to a subscriber that never reads leave ONE
    in memory. A bounded queue of N would have held N -- and, worse, would
    have had to choose between dropping the newest (discarding the only
    snapshot that matters) and dropping the oldest (a slower coalesce at
    N times the memory).
    """
    subscription = _subscription()
    for index in range(100):
        subscription.offer(_snapshot(f"rev-{index}"))

    assert subscription._pending.revision == "rev-99"
    assert subscription.coalesced == 99

    taken = subscription.take()
    assert taken.revision == "rev-99"
    assert subscription._pending is None
    assert subscription.has_pending is False


def test_an_identical_snapshot_is_not_counted_as_a_supersession():
    """`coalesced` must mean "you missed intermediate states".

    Re-offering the same revision is not an intermediate state, and
    counting it would report drops that never happened.
    """
    subscription = _subscription()
    subscription.offer(_snapshot("same"))
    subscription.offer(_snapshot("same"))
    subscription.offer(_snapshot("same"))

    assert subscription.coalesced == 0


def test_coalescing_is_reported_to_the_client(world_root, monkeypatch):
    """A slow consumer learns it was slow, without a gap in the sequence.

    Every poll pass runs before the sender task gets a turn, because they
    are driven inside one portal call. So the client sees one message
    carrying the newest state plus the count it skipped.
    """
    client = make_client(monkeypatch, world_root)
    with client.websocket_connect("/ws") as ws:
        subscribe(ws)
        first = drain(ws, expect="cartridge_result")
        assert first["coalesced"] == 0

        channel = client.app.state.result_hub

        async def _flood():
            hub = channel
            for index in range(20):
                for connection in list(hub._channels):
                    for subscription in connection._subscriptions.values():
                        subscription.offer(_snapshot(f"flood-{index}"))
                    connection._wakeup.set()

        client.portal.call(_flood)
        latest = drain(ws, expect="cartridge_result")

    assert latest["revision"] == "flood-19"
    assert latest["coalesced"] == 19
    assert latest["seq"] == 2, "the sequence stays dense; coalescing is not a gap"


def test_a_connection_cannot_open_unbounded_subscriptions(world_root, monkeypatch):
    """A remote party must not be able to grow a server-side dict at will."""
    client = make_client(monkeypatch, world_root)
    request = {
        "type": "result_subscribe",
        "cartridge": "world_builder",
        "result_type": "status",
    }
    with client.websocket_connect("/ws") as ws:
        for _ in range(MAX_SUBSCRIPTIONS_PER_CONNECTION):
            ws.send_json(request)
        ws.send_json(request)

        # Each accepted subscribe produces an acknowledgement AND a
        # snapshot, and the refusal is one more message; read them all
        # and classify, rather than assuming an interleaving that nothing
        # guarantees.
        seen = [
            ws.receive_json()
            for _ in range(2 * MAX_SUBSCRIPTIONS_PER_CONNECTION + 1)
        ]

    accepted = [m for m in seen if m["type"] == "result_subscribed"]
    refused = [m for m in seen if m["type"] == "result_error"]

    assert len(accepted) == MAX_SUBSCRIPTIONS_PER_CONNECTION
    assert len(refused) == 1
    assert refused[0]["reason"] == "too_many_subscriptions"


def test_the_journal_cache_holds_a_summary_not_the_journal(tmp_path):
    """Poll cost, and cache memory, must not grow with session length.

    Stat-gating alone stopped the journal being re-PARSED, but the blocks
    that read it SCANNED it, so cost stayed O(events) per poll -- and
    caching the parsed list would have held every event dict for as long
    as anyone was subscribed. Measured at 50,000 events: 117 ms per
    snapshot with neither, 9.26 ms with gating alone, 0.73 ms with a
    cached summary.
    """
    from tower.results.world_builder import _summarise_events

    events = (
        [{"kind": "session_started"}]
        + [{"kind": "keyframe_accepted"}] * 40
        + [{"kind": "tracking_lost"}]
        + [{"kind": "frame_rejected"}] * 5
    )
    summary = _summarise_events(events)

    assert summary == {
        "keyframes_accepted": 40,
        "last_tracking": "tracking_lost",
        "stopped": False,
        "corrupt_lines": 0,
        # Counted here rather than derived downstream, because the only other
        # number a consumer had was the segment total and `segments - 1` is
        # not a count of tracking losses -- a segment is also opened when the
        # solver cannot extend its chain while tracking is fine. Both are
        # scalars, so the memory bound below is unaffected.
        "tracking_restarts": 1,
        "chain_breaks": 0,
    }
    # Fixed arity whatever the journal length: this is the memory bound.
    assert len(_summarise_events(events * 1000)) == len(summary)
    assert all(
        isinstance(value, (int, str, bool, type(None)))
        for value in summary.values()
    ), "the summary must hold scalars, never the events"


def test_the_producer_caches_are_capped(tmp_path):
    """No cache in the producer may grow without bound.

    A remote client cannot drive these -- `resolve` refuses a world id
    that is not on disk before anything is cached -- so growth follows the
    operator's data. Capped anyway: an unbounded-in-principle cache is a
    latent defect whether or not today's callers can reach it, and
    recovery is one re-read of a file that is still there.
    """
    from tower.results.world_builder import _FileCache

    cache = _FileCache()
    absent = tmp_path / "nope"
    for index in range(_FileCache.MAX_ENTRIES * 4):
        path = tmp_path / f"f{index}"
        path.write_text("x", encoding="utf-8")
        cache.read("test", path, lambda: index)

    assert len(cache._entries) <= _FileCache.MAX_ENTRIES

    # A file that does not exist is never cached: it must be picked up the
    # moment it appears.
    cache.read("test", absent, lambda: None)
    assert str(absent) not in cache._entries


# -- slow consumers -----------------------------------------------------


def test_a_consumer_that_never_reads_is_dropped_not_waited_on():
    """The bound that protects the FRAME path.

    One socket is one TCP stream, so a client that stops reading blocks
    everything on it. A result send that waited indefinitely would be the
    thing holding the frame path up.
    """
    hub = ResultHub(lambda *args: _snapshot("x"), clock=lambda: 0.0)
    started = asyncio.Event()

    async def _never_returns(_payload):
        started.set()
        await asyncio.Event().wait()

    async def _run():
        import tower.results.publisher as publisher

        original = (publisher.SEND_TIMEOUT_S, publisher.TOTAL_SEND_TIMEOUT_S)
        publisher.SEND_TIMEOUT_S = 0.05
        publisher.TOTAL_SEND_TIMEOUT_S = 0.05
        try:
            channel = ConnectionChannel(hub, _never_returns, lambda: 0.0)
            subscription = _subscription()
            await channel.add(subscription)
            subscription.offer(_snapshot("first"))
            channel._wakeup.set()
            await asyncio.wait_for(started.wait(), timeout=2.0)
            # Long enough for the 50 ms send timeout to fire and the
            # subscription to be closed.
            for _ in range(60):
                await asyncio.sleep(0.01)
                if channel.subscription_count == 0:
                    break
            return channel.subscription_count
        finally:
            publisher.SEND_TIMEOUT_S, publisher.TOTAL_SEND_TIMEOUT_S = original
            await channel.close()

    assert asyncio.run(_run()) == 0


# -- cleanup ------------------------------------------------------------


def test_disconnect_removes_every_subscription_and_stops_the_reader(
    world_root, monkeypatch
):
    """A subscription that outlived its socket would keep polling disk forever."""
    client = make_client(monkeypatch, world_root)
    hub = client.app.state.result_hub

    with client.websocket_connect("/ws") as ws:
        subscribe(ws)
        drain(ws, expect="cartridge_result")
        assert client.portal.call(_channel_count(hub)) == 1

    assert client.portal.call(_channel_count(hub)) == 0
    assert client.portal.call(_task_alive(hub)) is False


def test_unsubscribing_the_last_subscription_stops_the_reader(
    world_root, monkeypatch
):
    """A Tower nobody is watching must do no disk IO on anyone's behalf."""
    client = make_client(monkeypatch, world_root)
    hub = client.app.state.result_hub

    with client.websocket_connect("/ws") as ws:
        reply = subscribe(ws)
        drain(ws, expect="cartridge_result")
        assert client.portal.call(_task_alive(hub)) is True

        ws.send_json(
            {
                "type": "result_unsubscribe",
                "subscription_id": reply["subscription_id"],
            }
        )
        drain(ws, expect="result_unsubscribed")

        assert client.portal.call(_channel_count(hub)) == 0
        assert client.portal.call(_task_alive(hub)) is False


def _channel_count(hub):
    async def _call():
        return len(hub._channels)

    return _call


def _task_alive(hub):
    async def _call():
        return hub._task is not None and not hub._task.done()

    return _call


# -- failure containment ------------------------------------------------


def test_a_reader_failure_reaches_the_client_instead_of_going_quiet():
    """The worst outcome is silence, so it is the one that is ruled out.

    Drives the REAL `ResultHub._run` loop rather than calling `fail_all`
    by hand. An adversarial review pointed out that the earlier version
    raised its own RuntimeError, invoked fail_all directly, and then
    asserted the exception name against a string the test itself had
    written -- so deleting the entire `except Exception -> fail_all` block
    in `_run` would have left it green. It now fails if that block goes.
    """
    def _explode(*args):
        raise RuntimeError("the disk fell over")

    hub = ResultHub(_explode, clock=lambda: 0.0, poll_seconds=0.001)
    sent = []

    async def _capture(payload):
        sent.append(payload)

    async def _run():
        channel = ConnectionChannel(hub, _capture, lambda: 0.0)
        # add() starts the hub's own task; nothing here touches fail_all.
        await channel.add(_subscription())
        for _ in range(200):
            await asyncio.sleep(0.01)
            if sent:
                break
        await channel.close()
        return sent

    messages = asyncio.run(_run())
    assert messages, (
        "the reader died and the client was never told -- silence is the "
        "failure this path exists to prevent"
    )
    assert messages[0]["reason"] == "channel_failed"
    assert "RuntimeError" in messages[0]["message"]


def test_a_slow_consumer_is_told_before_it_is_dropped():
    """Closing a subscription in silence leaves a client waiting forever.

    The contract names the unsolicited errors; a drop that sends nothing
    is not one of them, so a conforming client could never learn it
    happened.
    """
    import tower.results.publisher as publisher

    hub = ResultHub(lambda *args: _snapshot("x"), clock=lambda: 0.0)
    seen = []
    blocked = asyncio.Event()

    async def _slow(payload):
        if payload.get("type") == "cartridge_result":
            blocked.set()
            await asyncio.Event().wait()
        seen.append(payload)

    async def _run():
        original = (publisher.SEND_TIMEOUT_S, publisher.TOTAL_SEND_TIMEOUT_S)
        publisher.SEND_TIMEOUT_S = 0.05
        publisher.TOTAL_SEND_TIMEOUT_S = 0.05
        try:
            channel = ConnectionChannel(hub, _slow, lambda: 0.0)
            subscription = _subscription()
            await channel.add(subscription)
            subscription.offer(_snapshot("first"))
            channel._wakeup.set()
            await asyncio.wait_for(blocked.wait(), timeout=2.0)
            for _ in range(200):
                await asyncio.sleep(0.01)
                if seen:
                    break
            await channel.close()
            return seen
        finally:
            publisher.SEND_TIMEOUT_S, publisher.TOTAL_SEND_TIMEOUT_S = original

    messages = asyncio.run(_run())
    assert messages, "the subscription was closed without telling the client"
    assert messages[0]["reason"] == "consumer_too_slow"
    assert "subscribe again" in messages[0]["message"]


def test_one_unreadable_target_does_not_stop_the_others():
    """A poll pass must not be all-or-nothing across subscribers."""
    calls = []

    def _snapshot_for(cartridge, result_type, world_id, session_id):
        calls.append(world_id)
        if world_id == "broken":
            raise ValueError("unreadable")
        return _snapshot("fine")

    hub = ResultHub(_snapshot_for, clock=lambda: 0.0)
    delivered = []

    async def _capture(payload):
        delivered.append(payload)

    async def _run():
        channel = ConnectionChannel(hub, _capture, lambda: 0.0)
        for index, world in enumerate(("broken", "healthy")):
            subscription = Subscription(
                subscription_id=f"sub-{index}",
                cartridge="world_builder",
                result_type="status",
                contract="c",
                world_id=world,
                session_id=None,
                cursor_status=None,
            )
            await channel.add(subscription)
        await hub.poll_once()
        for _ in range(50):
            await asyncio.sleep(0.01)
            if delivered:
                break
        await channel.close()
        return delivered

    messages = asyncio.run(_run())
    assert "broken" in calls and "healthy" in calls
    assert [m["payload"]["revision_marker"] for m in messages] == ["fine"]


def test_hub_shutdown_survives_a_reader_that_already_died():
    """Shutdown must not re-raise a dead task's exception.

    That would turn a dead push channel into a failed application
    shutdown -- a small problem escalating into a visible one.
    """
    def _explode(*args):
        raise RuntimeError("boom")

    hub = ResultHub(_explode, clock=lambda: 0.0, poll_seconds=0.001)

    async def _run():
        channel = ConnectionChannel(hub, _noop_send, lambda: 0.0)
        await channel.add(_subscription())
        for _ in range(50):
            await asyncio.sleep(0.01)
            if hub._task is not None and hub._task.done():
                break
        await hub.shutdown()
        await channel.close()

    asyncio.run(_run())


async def _noop_send(_payload):
    return None


def test_a_wedged_cartridge_does_not_silence_the_others(monkeypatch):
    """One producer that never returns must not take the Tower with it.

    The targets are polled SEQUENTIALLY in one pass, so before this a
    producer that blocked blocked everything behind it. A reviewer
    measured both ends: a wedged World Builder read delivered **nothing
    at all, for any cartridge, across 12 poll windows** -- no error, no
    `fail_target`, the loop simply never came back -- and a merely slow
    (2 s) producer made Document Memory and Scene Understanding exactly
    5x slower.

    That is the brief's own non-negotiable: a World Builder failure must
    not degrade CV Lab, Object Memory, Document Memory, Scene
    Understanding or the Tower's networking.
    """
    import threading

    from tower.results import publisher as publisher_module

    monkeypatch.setattr(publisher_module, "SNAPSHOT_TIMEOUT_SECONDS", 0.2)
    release = threading.Event()
    calls = []

    def _snapshot_for(cartridge, result_type, world_id, session_id):
        calls.append(world_id)
        if world_id == "wedged":
            # Never returns within the deadline. Released at the end so
            # the worker thread does not outlive the test.
            release.wait(timeout=30)
            return _snapshot("late")
        return _snapshot("fine")

    hub = ResultHub(_snapshot_for, clock=lambda: 0.0)
    delivered = []

    async def _capture(payload):
        delivered.append(payload)

    async def _run():
        channel = ConnectionChannel(hub, _capture, lambda: 0.0)
        for index, world in enumerate(("wedged", "healthy")):
            subscription = Subscription(
                subscription_id=f"sub-{index}",
                cartridge="world_builder",
                result_type="status",
                contract="c",
                world_id=world,
                session_id=None,
                cursor_status=None,
            )
            await channel.add(subscription)
        await hub.poll_once()
        for _ in range(200):
            await asyncio.sleep(0.01)
            if delivered:
                break
        await channel.close()
        return delivered

    try:
        messages = asyncio.run(_run())
    finally:
        release.set()

    assert "wedged" in calls and "healthy" in calls, calls
    assert [m["payload"]["revision_marker"] for m in messages] == ["fine"], (
        "the healthy cartridge was silenced by the wedged one"
    )


def test_a_cartridge_that_keeps_timing_out_tells_its_own_subscribers(monkeypatch):
    """And it must say so, rather than going quiet forever.

    The consecutive-failure path exists so a persistent problem reaches
    the people watching it instead of the log. A timeout is exactly such
    a problem and was not routed into it.
    """
    import threading

    from tower.results import publisher as publisher_module

    monkeypatch.setattr(publisher_module, "SNAPSHOT_TIMEOUT_SECONDS", 0.05)
    release = threading.Event()

    def _snapshot_for(cartridge, result_type, world_id, session_id):
        release.wait(timeout=30)
        return _snapshot("late")

    hub = ResultHub(_snapshot_for, clock=lambda: 0.0)
    delivered = []

    async def _capture(payload):
        delivered.append(payload)

    async def _run():
        channel = ConnectionChannel(hub, _capture, lambda: 0.0)
        await channel.add(Subscription(
            subscription_id="sub-0", cartridge="world_builder",
            result_type="status", contract="c", world_id="wedged",
            session_id=None, cursor_status=None,
        ))
        for _ in range(publisher_module.MAX_CONSECUTIVE_TARGET_FAILURES):
            await hub.poll_once()
        for _ in range(200):
            await asyncio.sleep(0.01)
            if delivered:
                break
        await channel.close()
        return delivered

    try:
        messages = asyncio.run(_run())
    finally:
        release.set()

    assert messages, "a target that timed out repeatedly said nothing at all"
    reasons = " ".join(str(m) for m in messages)
    assert "within" in reasons and "times in a row" in reasons, reasons
