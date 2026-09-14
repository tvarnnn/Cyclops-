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
        # The fixture's rejections carry no reason, so neither counter
        # moves; the truthfulness suite drives the real event.
        "frames_rejected_wrong_size": 0,
        "frames_rejected_malformed": 0,
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
        release.set()
        return delivered

    messages = asyncio.run(_run())

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

    import time as _time

    # A REAL CLOCK. A failure is counted only once a snapshot has actually
    # aged past the deadline -- a future dispatched a moment ago is not a
    # wedge -- so a hub frozen at 0.0 could never escalate.
    hub = ResultHub(_snapshot_for, clock=_time.monotonic)
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
        for _ in range(publisher_module.MAX_CONSECUTIVE_TARGET_FAILURES + 1):
            await hub.poll_once()
            await asyncio.sleep(publisher_module.SNAPSHOT_TIMEOUT_SECONDS * 1.5)
        for _ in range(200):
            await asyncio.sleep(0.01)
            if delivered:
                break
        await channel.close()
        release.set()
        return delivered

    messages = asyncio.run(_run())

    assert messages, "a target that timed out repeatedly said nothing at all"
    reasons = " ".join(str(m) for m in messages)
    assert "within" in reasons and "times in a row" in reasons, reasons


def test_a_wedged_target_is_never_dispatched_twice(monkeypatch):
    """The fix for the wedge must not leak a thread per poll.

    The first deadline used `asyncio.wait_for`, which cancels the await
    and NOT the thread -- and then dispatched the same target again on
    the very next pass while the previous thread was still running. A
    reviewer measured one leaked thread per poll per wedged target into
    the default executor, which the capture path shares: at the default
    0.5 s poll and 10 s deadline, one wedged target exhausted a 24-worker
    pool in ~252 s, after which every `asyncio.to_thread` in the process
    queued forever -- `stream_start`, the disconnect cleanup, the
    subscribe-time snapshot, shutdown. Before that fix the wedge silenced
    the result channel and nothing else. A fix worse than its defect.

    A target with a snapshot in flight is not dispatched again. Over 25
    polls the wedged producer must be entered exactly ONCE, and the
    healthy target beside it must be served on every one of them.
    """
    import threading

    from tower.results import publisher as publisher_module

    monkeypatch.setattr(publisher_module, "SNAPSHOT_TIMEOUT_SECONDS", 0.05)
    release = threading.Event()
    entered = {"wedged": 0, "healthy": 0}

    def _snapshot_for(cartridge, result_type, world_id, session_id):
        entered[world_id] += 1
        if world_id == "wedged":
            release.wait(timeout=30)
            return _snapshot("late")
        return _snapshot(f"fine-{entered['healthy']}")

    hub = ResultHub(_snapshot_for, clock=lambda: 0.0)
    delivered = []

    async def _capture(payload):
        delivered.append(payload)

    async def _run():
        channel = ConnectionChannel(hub, _capture, lambda: 0.0)
        for index, world in enumerate(("wedged", "healthy")):
            await channel.add(Subscription(
                subscription_id=f"sub-{index}", cartridge="world_builder",
                result_type="status", contract="c", world_id=world,
                session_id=None, cursor_status=None,
            ))
        for _ in range(25):
            await hub.poll_once()
            await asyncio.sleep(0)
        for _ in range(100):
            await asyncio.sleep(0.01)
            if len(delivered) >= 1:
                break
        await channel.close()
        release.set()

    asyncio.run(_run())

    assert entered["wedged"] == 1, (
        f"the wedged target was dispatched {entered['wedged']} times across 25 "
        "polls -- one leaked thread per poll, which is how the executor the "
        "capture path shares was exhausted"
    )
    # AT LEAST 25, not exactly: the hub's own polling loop can run a pass of
    # its own beside the explicit ones, and once passes were serialised
    # that extra pass stopped being swallowed by interleaving and showed up
    # as a 26th healthy entry. The claim under test is the wedge; the
    # healthy target must simply be served on every explicit pass.
    assert entered["healthy"] >= 25, entered
    assert delivered and delivered[0]["payload"]["revision_marker"].startswith("fine")


def test_targets_are_polled_together_not_one_after_another():
    """A slow producer must not cost every other cartridge its own delay.

    Sequential polling made a 2 s World Builder snapshot cost Document
    Memory and Scene Understanding 2 s each -- measured at exactly 5x
    slower -- and the first deadline did not change that: it only turned
    "more than 10 s" into a failure. Two targets that each take 0.3 s
    must finish in one pass of ~0.3 s, not ~0.6 s.
    """
    import time as _time

    def _snapshot_for(cartridge, result_type, world_id, session_id):
        _time.sleep(0.3)
        return _snapshot(world_id)

    hub = ResultHub(_snapshot_for, clock=lambda: 0.0)
    delivered = []

    async def _capture(payload):
        delivered.append(payload)

    async def _run():
        channel = ConnectionChannel(hub, _capture, lambda: 0.0)
        for index, world in enumerate(("slow-a", "slow-b")):
            await channel.add(Subscription(
                subscription_id=f"sub-{index}", cartridge="world_builder",
                result_type="status", contract="c", world_id=world,
                session_id=None, cursor_status=None,
            ))
        started = _time.perf_counter()
        await hub.poll_once()
        elapsed = _time.perf_counter() - started
        for _ in range(100):
            await asyncio.sleep(0.01)
            if len(delivered) >= 2:
                break
        await channel.close()
        return elapsed

    elapsed = asyncio.run(_run())
    markers = sorted(m["payload"]["revision_marker"] for m in delivered)
    assert markers == ["slow-a", "slow-b"], markers
    # Sequential is >= 0.6 s; concurrent is ~0.3 s. The margin is 50 ms
    # on the wrong side, which is coarse enough to survive suite load.
    assert elapsed < 0.55, f"one pass over two 0.3 s targets took {elapsed:.2f}s"

def test_a_flapping_subscribe_shares_one_thread_with_the_poll_loop(monkeypatch):
    """The subscribe-time snapshot must ride the poll loop's in-flight table.

    Round 20 gave the inline first snapshot a deadline and called the
    leaked thread "bounded at one per subscribe attempt, which is
    client-driven". iOS's `sendStallTimeout` is 2 s: while that snapshot
    stalls, the phone replaces the socket and re-subscribes every ~2.5 s,
    and every attempt minted a thread. A reviewer drove it against a
    wedged read on a real Tower: 24 executor threads in 60 s, then
    `stream_start` never answered for any phone -- the exhaustion the
    poll loop had just been cured of, one route over.

    Twelve subscribes to one wedged target, with the poll loop running
    beside them, must enter the producer exactly ONCE.
    """
    import threading

    from tower.results import publisher as publisher_module

    monkeypatch.setattr(publisher_module, "SNAPSHOT_TIMEOUT_SECONDS", 0.05)
    release = threading.Event()
    entered = {"n": 0}

    def _snapshot_for(cartridge, result_type, world_id, session_id):
        entered["n"] += 1
        release.wait(timeout=30)
        return _snapshot("late")

    hub = ResultHub(_snapshot_for, clock=lambda: 0.0)

    async def _run():
        subscription = Subscription(
            subscription_id="sub", cartridge="world_builder",
            result_type="status", contract="c", world_id="wedged",
            session_id=None, cursor_status=None,
        )
        timeouts = 0
        for _ in range(12):
            try:
                await hub.first_snapshot(subscription, timeout=0.05)
            except TimeoutError:
                timeouts += 1
            await hub.poll_once()
        release.set()
        return timeouts

    timeouts = asyncio.run(_run())
    assert timeouts == 12
    assert entered["n"] == 1, (
        f"the producer was entered {entered['n']} times for one wedged target "
        "-- one thread per reconnect, which is how the executor was exhausted"
    )


def test_a_snapshot_finished_while_unwatched_is_not_delivered(monkeypatch):
    """A result computed for a subscription that has gone is stale by definition.

    A target left the poll set while its thread was running -- the phone
    dropped -- and came back. The finished future was still in the
    in-flight table and was delivered to the NEW subscription: a reviewer
    measured `rev 1` and then **`rev 0`**, 7.5 s old, on a reconnected
    phone's default target. The revision went backwards. Such a result is
    discarded and the target dispatched afresh.
    """
    import threading

    from tower.results import publisher as publisher_module

    # Generous: the stall is gated, not timed, and a fresh thread on a
    # loaded box can take longer than 50 ms merely to start.
    monkeypatch.setattr(publisher_module, "SNAPSHOT_TIMEOUT_SECONDS", 1.0)
    gate = threading.Event()
    produced = []

    def _snapshot_for(cartridge, result_type, world_id, session_id):
        marker = f"rev-{len(produced)}"
        produced.append(marker)
        if marker == "rev-0":
            gate.wait(timeout=30)   # the first computation outlives the subscriber
        return _snapshot(marker)

    hub = ResultHub(_snapshot_for, clock=lambda: 0.0)
    # Separate inboxes: what the FIRST channel is sent on close is not
    # evidence about the second.
    first_inbox, delivered = [], []

    async def _capture_first(payload):
        first_inbox.append(payload)

    async def _capture(payload):
        delivered.append(payload)

    def _sub(n):
        return Subscription(
            subscription_id=f"sub-{n}", cartridge="world_builder",
            result_type="status", contract="c", world_id="w",
            session_id=None, cursor_status=None,
        )

    async def _run():
        first = ConnectionChannel(hub, _capture_first, lambda: 0.0)
        await first.add(_sub(1))
        await hub.poll_once()          # dispatches rev-0, which stalls
        await first.close()            # the phone drops; target unwatched
        gate.set()                     # rev-0 finishes while nobody watches
        await asyncio.sleep(0.3)
        second = ConnectionChannel(hub, _capture, lambda: 0.0)
        await second.add(_sub(2))      # the phone is back
        for _ in range(5):
            await hub.poll_once()
            for _ in range(50):
                await asyncio.sleep(0.01)
                if delivered:
                    break
            if delivered:
                break
        await second.close()

    asyncio.run(_run())
    kinds = [m.get("type") for m in delivered]
    markers = [
        m["payload"]["revision_marker"] for m in delivered if "payload" in m
    ]
    assert markers, f"nothing was delivered to the reconnected subscription: {kinds}"
    assert markers[0] != "rev-0", (
        f"the reconnected subscription was handed the stale rev-0: {markers}"
    )
    # And discarding the stale result must not be reported to the NEW
    # subscription as a failure of its own.
    assert "result_error" not in kinds, kinds

def test_a_forced_pass_beside_the_hubs_own_loop_still_delivers():
    """Two passes on one target must not lose the one that had to deliver.

    The hub's own loop (started by the first subscription) and a forced
    pass -- `pump()` in the tests, heartbeat zero, so it MUST deliver --
    ran concurrently. The loop's pass dispatched the snapshot; the forced
    pass found it "already in flight", counted a failure and offered
    nothing; the loop's pass then offered under the ordinary heartbeat
    rule, which sends nothing for an unchanged world. Net: nothing, and a
    test waiting for that message hung the whole suite for 33 minutes.
    Passes are serialised now. Ten forced passes beside the running loop
    must each deliver.
    """
    import time as _time

    from tower.results import publisher as publisher_module

    def _snapshot_for(cartridge, result_type, world_id, session_id):
        return _snapshot("same")

    hub = ResultHub(_snapshot_for, clock=_time.monotonic, poll_seconds=0.01)
    delivered = []

    async def _capture(payload):
        delivered.append(payload)

    async def _run():
        channel = ConnectionChannel(hub, _capture, _time.monotonic)
        await channel.add(Subscription(
            subscription_id="sub", cartridge="world_builder",
            result_type="status", contract="c", world_id="w",
            session_id=None, cursor_status=None,
        ))
        # Let the hub's own loop get going beside us.
        await asyncio.sleep(0.05)
        forced = 0
        for _ in range(10):
            before = len(delivered)
            previous = hub._heartbeat_seconds
            hub._heartbeat_seconds = 0.0
            try:
                await hub.poll_once()
            finally:
                hub._heartbeat_seconds = previous
            for _ in range(50):
                await asyncio.sleep(0.005)
                if len(delivered) > before:
                    forced += 1
                    break
        await channel.close()
        return forced

    forced = asyncio.run(_run())
    assert forced == 10, (
        f"only {forced} of 10 forced passes delivered beside the running loop"
    )


def test_a_snapshot_started_moments_ago_is_not_a_failure():
    """A failure is a snapshot that has AGED past the deadline, nothing else.

    A future dispatched a moment ago by a subscribe or by the previous
    pass is not a wedge, and the first version counted it as one --
    logging "has exceeded 10.0s" six milliseconds after the connection
    opened. Three subscribes during three passes would have told a
    healthy target's subscribers it had failed.
    """
    import threading
    import time as _time

    from tower.results import publisher as publisher_module

    gate = threading.Event()

    def _snapshot_for(cartridge, result_type, world_id, session_id):
        gate.wait(timeout=30)
        return _snapshot("eventually")

    hub = ResultHub(_snapshot_for, clock=_time.monotonic)
    delivered = []

    async def _capture(payload):
        delivered.append(payload)

    async def _run():
        channel = ConnectionChannel(hub, _capture, _time.monotonic)
        subscription = Subscription(
            subscription_id="sub", cartridge="world_builder",
            result_type="status", contract="c", world_id="w",
            session_id=None, cursor_status=None,
        )
        await channel.add(subscription)
        # A subscribe dispatches the snapshot; it is still running when
        # the pass comes round. Young, not wedged.
        try:
            await hub.first_snapshot(subscription, timeout=0.05)
        except TimeoutError:
            pass
        await hub.poll_once()
        failures = dict(hub._failures)
        gate.set()
        await asyncio.sleep(0.2)
        await channel.close()
        return failures

    failures = asyncio.run(_run())
    assert not failures, (
        f"a snapshot that had just been dispatched was counted as a failure: {failures}"
    )


# ---------------------------------------------------------------------------
# Round 22: what the third reviewer of the in-flight table found.
# ---------------------------------------------------------------------------


def _fake_clock():
    """A clock the test advances by hand."""
    state = {"now": 0.0}

    def clock():
        return state["now"]

    clock.advance = lambda seconds: state.__setitem__("now", state["now"] + seconds)
    return clock


def test_a_subscribe_does_not_wait_on_a_snapshot_already_past_its_deadline(monkeypatch):
    """A future older than the deadline is a known wedge; the answer is immediate.

    `first_snapshot` runs inline in the connection's message loop. Every
    subscribe to a wedged target sat there for the full deadline -- a
    reviewer measured six consecutive subscribes, all timing out, three
    of them after the fault had cleared -- while the frames queued behind
    it went unanswered.
    """
    import threading
    import time as _time

    from tower.results import publisher as publisher_module

    monkeypatch.setattr(publisher_module, "SNAPSHOT_TIMEOUT_SECONDS", 1.0)
    gate = threading.Event()

    def _snapshot_for(cartridge, result_type, world_id, session_id):
        gate.wait(timeout=30)
        return _snapshot("late")

    clock = _fake_clock()
    hub = ResultHub(_snapshot_for, clock=clock, age_clock=clock)

    async def _run():
        subscription = Subscription(
            subscription_id="sub-1", cartridge="world_builder",
            result_type="status", contract="c", world_id="wedged",
            session_id=None, cursor_status=None,
        )
        try:
            await hub.first_snapshot(subscription, timeout=0.05)
        except TimeoutError:
            pass
        clock.advance(2.0)   # past the deadline, short of the abandonment cap
        started = _time.monotonic()
        with pytest.raises(TimeoutError):
            await hub.first_snapshot(subscription, timeout=1.0)
        gate.set()
        return _time.monotonic() - started

    waited = asyncio.run(_run())
    assert waited < 0.2, (
        f"a subscribe waited {waited:.2f}s on a snapshot already past its deadline"
    )


def test_a_wedged_target_is_abandoned_and_tried_afresh(monkeypatch):
    """A thread that never returns must not poison its target for the life
    of the process.

    The version before this never re-dispatched a target whose future
    was still pending, so a snapshot thread that never came back meant
    no pass and no subscribe ever computed that target again -- and a
    fault that cleared was never noticed. After
    `SNAPSHOT_ABANDON_MULTIPLIER` deadlines the hub forgets the thread
    and tries again.
    """
    import threading

    from tower.results import publisher as publisher_module

    monkeypatch.setattr(publisher_module, "SNAPSHOT_TIMEOUT_SECONDS", 0.02)
    never = threading.Event()
    entered = {"n": 0}

    def _snapshot_for(cartridge, result_type, world_id, session_id):
        entered["n"] += 1
        if entered["n"] == 1:
            never.wait(timeout=60)      # the wedge: this thread never returns
            return _snapshot("wedged")
        return _snapshot("recovered")

    clock = _fake_clock()
    hub = ResultHub(_snapshot_for, clock=clock, age_clock=clock)
    delivered = []

    async def _capture(payload):
        delivered.append(payload)

    async def _run():
        channel = ConnectionChannel(hub, _capture, clock)
        await channel.add(Subscription(
            subscription_id="sub-1", cartridge="world_builder",
            result_type="status", contract="c", world_id="w",
            session_id=None, cursor_status=None,
        ))
        await hub.poll_once()               # dispatches the wedge
        assert entered["n"] == 1
        clock.advance(0.02 * 2)             # past the deadline, not the cap
        await hub.poll_once()
        assert entered["n"] == 1, "re-dispatched before the abandonment cap"
        clock.advance(0.02 * publisher_module.SNAPSHOT_ABANDON_MULTIPLIER)
        await hub.poll_once()               # abandons, dispatches afresh
        for _ in range(100):
            await asyncio.sleep(0.01)
            if delivered:
                break
        await channel.close()
        never.set()

    asyncio.run(_run())
    assert entered["n"] == 2, f"the producer was entered {entered['n']} times"
    assert [m["payload"]["revision_marker"] for m in delivered] == ["recovered"]


def test_a_snapshot_finished_before_a_subscribe_is_not_its_first_snapshot():
    """A done future nobody collected was computed for a subscription that
    has gone; a new subscription gets its own.

    Reproduced by a reviewer on a real socket: the phone dropped mid-pass,
    `detach` cancelled the pass, the future finished for nobody, and the
    reconnected phone's `first_snapshot` returned it verbatim -- 4.0 s
    old, behind the revision on disk, and the phone's own current cursor
    was classified "stale" against it.
    """
    import threading

    gate = threading.Event()
    produced = []

    def _snapshot_for(cartridge, result_type, world_id, session_id):
        marker = f"rev-{len(produced)}"
        produced.append(marker)
        if marker == "rev-0":
            gate.wait(timeout=30)     # rev-0 outlives the phone that asked
        return _snapshot(marker)

    clock = _fake_clock()
    hub = ResultHub(_snapshot_for, clock=lambda: 0.0, age_clock=clock)

    def _sub(n):
        return Subscription(
            subscription_id="sub-1", cartridge="world_builder",
            result_type="status", contract="c", world_id="w",
            session_id=None, cursor_status=None,
        )

    async def _run():
        first = ConnectionChannel(hub, lambda payload: None, lambda: 0.0)
        await first.add(_sub(1))
        # The pass dispatches rev-0 ...
        pass_task = asyncio.ensure_future(hub.poll_once())
        await asyncio.sleep(0)
        # ... and the phone drops before it is collected: detach cancels
        # the hub's loop; the pass we started by hand is cancelled too.
        await first.close()
        pass_task.cancel()
        try:
            await pass_task
        except asyncio.CancelledError:
            pass
        gate.set()
        await asyncio.sleep(0.1)         # rev-0 finishes, for nobody
        clock.advance(4.0)               # ... and sits there, 4.0 s old (the reviewer's number)
        # The phone is back.
        snapshot = await hub.first_snapshot(_sub(2), timeout=1.0)
        return snapshot.revision

    assert asyncio.run(_run()) == "rev-1", "the reconnected phone was handed rev-0"


@pytest.mark.parametrize("reconnected_id", ["sub-2", "sub-1"])
def test_a_snapshot_finished_while_unwatched_is_not_delivered_even_to_a_reused_id(
    monkeypatch, reconnected_id
):
    """Subscription ids are minted PER CONNECTION, so every phone's first
    subscription is "sub-1". The watcher rule keyed on ids matched a
    reconnected phone's "sub-1" to the dead socket's "sub-1" and kept the
    stale result -- a reviewer parametrised the shipped test with the
    reused id and it failed. Keyed on the Subscription objects now.
    """
    import threading

    from tower.results import publisher as publisher_module

    monkeypatch.setattr(publisher_module, "SNAPSHOT_TIMEOUT_SECONDS", 0.05)
    gate = threading.Event()
    walk_calls = []

    def _snapshot_for(cartridge, result_type, world_id, session_id):
        if cartridge != "world_builder":
            return _snapshot(f"{cartridge}-bystander")
        marker = f"rev-{len(walk_calls)}"
        walk_calls.append(marker)
        if marker == "rev-0":
            gate.wait(timeout=30)     # the walk's first snapshot outlives its phone
        return _snapshot(marker)

    hub = ResultHub(_snapshot_for, clock=lambda: 0.0)
    delivered = []

    async def _capture(payload):
        delivered.append(payload)

    def _sub(subscription_id):
        return Subscription(
            subscription_id=subscription_id, cartridge="world_builder",
            result_type="status", contract="c", world_id="w",
            session_id=None, cursor_status=None,
        )

    async def _run():
        # A bystander phone on another target, here throughout: the hub
        # stays attached, so nothing but the watcher rule can catch this.
        bystander = ConnectionChannel(hub, lambda payload: None, lambda: 0.0)
        await bystander.add(Subscription(
            subscription_id="sub-1", cartridge="cv_lab",
            result_type="status", contract="c", world_id=None,
            session_id=None, cursor_status=None,
        ))
        first = ConnectionChannel(hub, _capture, lambda: 0.0)
        await first.add(_sub("sub-1"))
        await hub.poll_once()
        await first.close()
        gate.set()
        await asyncio.sleep(0.2)
        second = ConnectionChannel(hub, _capture, lambda: 0.0)
        await second.add(_sub(reconnected_id))
        await hub.poll_once()
        for _ in range(100):
            await asyncio.sleep(0.01)
            if delivered:
                break
        await second.close()
        await bystander.close()

    asyncio.run(_run())
    markers = [m["payload"]["revision_marker"] for m in delivered]
    assert markers and markers[0] != "rev-0", (
        f"the reconnected subscription {reconnected_id!r} was handed the stale rev-0: {markers}"
    )


def test_a_slow_target_does_not_delay_the_others_within_a_pass():
    """Each snapshot is offered as it lands, not when the slowest one does.

    A reviewer measured CV Lab, Object Memory and Document Memory at 7
    deliveries in 20 s beside a 2.1 s World Builder read -- 17.5% of
    their cadence -- because the pass waited for ALL of its futures
    before offering any. Sum had become max, and max was not a fix.
    """
    import threading
    import time as _time

    slow = threading.Event()

    def _snapshot_for(cartridge, result_type, world_id, session_id):
        if cartridge == "world_builder":
            slow.wait(timeout=30)
        return _snapshot(f"{cartridge}-1")

    hub = ResultHub(_snapshot_for, clock=_time.monotonic)
    arrivals = {}

    async def _capture(payload):
        arrivals.setdefault(payload["cartridge"], _time.monotonic())

    async def _run():
        channel = ConnectionChannel(hub, _capture, _time.monotonic)
        for cartridge in ("world_builder", "cv_lab", "object_memory"):
            await channel.add(Subscription(
                subscription_id=f"sub-{cartridge}", cartridge=cartridge,
                result_type="status", contract="c", world_id="w",
                session_id=None, cursor_status=None,
            ))
        started = _time.monotonic()
        pass_task = asyncio.ensure_future(hub.poll_once())
        for _ in range(200):
            await asyncio.sleep(0.005)
            if {"cv_lab", "object_memory"} <= arrivals.keys():
                break
        fast_arrived = _time.monotonic() - started
        slow.set()
        await pass_task
        await channel.close()
        return fast_arrived

    fast_arrived = asyncio.run(_run())
    assert fast_arrived < 0.5, (
        f"the fast cartridges waited {fast_arrived:.2f}s for the slow one"
    )


def test_the_loop_exits_promptly_with_a_wedged_snapshot_thread_still_running():
    """A snapshot thread nobody can reach must not hold the process open.

    `asyncio.run` joins the default executor's threads on the way out; a
    reviewer measured a Tower sitting at "Application shutdown complete"
    for 300 s with one wedged snapshot thread, until the interpreter gave
    up on the join. Snapshot threads are daemons of their own now.
    """
    import threading
    import time as _time

    never = threading.Event()

    def _snapshot_for(cartridge, result_type, world_id, session_id):
        never.wait(timeout=120)
        return _snapshot("never")

    hub = ResultHub(_snapshot_for, clock=_time.monotonic)

    async def _run():
        channel = ConnectionChannel(hub, lambda payload: None, _time.monotonic)
        await channel.add(Subscription(
            subscription_id="sub-1", cartridge="world_builder",
            result_type="status", contract="c", world_id="w",
            session_id=None, cursor_status=None,
        ))
        try:
            await hub.first_snapshot(
                channel.get("sub-1"), timeout=0.05
            )
        except TimeoutError:
            pass
        await channel.close()
        await hub.shutdown()

    started = _time.monotonic()
    asyncio.run(_run())
    elapsed = _time.monotonic() - started
    never.set()
    assert elapsed < 2.0, f"the loop took {elapsed:.1f}s to exit behind a wedged thread"


def test_a_departed_targets_failure_count_is_forgotten():
    """A target that failed once, left, and came back starts from zero.

    The prune ran only when `_failures` had outgrown the target set, so a
    count survived a reconnect and the returning target escalated after
    ONE failure instead of three.
    """
    def _snapshot_for(cartridge, result_type, world_id, session_id):
        if cartridge == "world_builder":
            raise RuntimeError("the producer exploded")
        return _snapshot("fine")

    hub = ResultHub(_snapshot_for, clock=lambda: 0.0)

    async def _run():
        channel = ConnectionChannel(hub, lambda payload: None, lambda: 0.0)
        await channel.add(Subscription(
            subscription_id="sub-1", cartridge="world_builder",
            result_type="status", contract="c", world_id="w",
            session_id=None, cursor_status=None,
        ))
        await hub.poll_once()
        for _ in range(50):
            await asyncio.sleep(0.01)
            if hub._failures:
                break
        assert hub._failures, "the raising producer was not counted"
        await channel.remove("sub-1")
        # A different, HEALTHY target is watched now -- one entry each
        # side, so a prune that waits for the dict to outgrow the targets
        # never fires. The old count must go regardless.
        other = ConnectionChannel(hub, lambda payload: None, lambda: 0.0)
        await other.add(Subscription(
            subscription_id="sub-1", cartridge="cv_lab",
            result_type="status", contract="c", world_id=None,
            session_id=None, cursor_status=None,
        ))
        await hub.poll_once()
        await asyncio.sleep(0.05)
        left_over = [t for t in hub._failures if t[0] == "world_builder"]
        await other.close()
        return left_over

    assert asyncio.run(_run()) == []


def test_an_exception_from_an_unwatched_snapshot_is_retrieved():
    """A raising producer whose subscriber left mid-flight must not dump a
    bare "Future exception was never retrieved" through asyncio's handler
    with no `[Tower][Results]` prefix.
    """
    import threading

    gate = threading.Event()
    unhandled = []

    def _snapshot_for(cartridge, result_type, world_id, session_id):
        gate.wait(timeout=30)
        raise RuntimeError("the producer exploded")

    hub = ResultHub(_snapshot_for, clock=lambda: 0.0)

    async def _run():
        asyncio.get_running_loop().set_exception_handler(
            lambda loop, context: unhandled.append(context)
        )
        channel = ConnectionChannel(hub, lambda payload: None, lambda: 0.0)
        await channel.add(Subscription(
            subscription_id="sub-1", cartridge="world_builder",
            result_type="status", contract="c", world_id="w",
            session_id=None, cursor_status=None,
        ))
        await hub.poll_once()          # dispatches; the thread is gated
        await channel.remove("sub-1")  # the subscriber leaves
        gate.set()
        await asyncio.sleep(0.1)       # the thread raises, for nobody
        other = ConnectionChannel(hub, lambda payload: None, lambda: 0.0)
        await other.add(Subscription(
            subscription_id="sub-1", cartridge="cv_lab",
            result_type="status", contract="c", world_id=None,
            session_id=None, cursor_status=None,
        ))
        await hub.poll_once()          # prunes the departed target
        await other.close()
        await hub.shutdown()

    asyncio.run(_run())
    import gc
    gc.collect()
    assert not [c for c in unhandled if "never retrieved" in str(c.get("message", ""))], unhandled


# ---------------------------------------------------------------------------
# Round 23: what the fourth reviewer of the in-flight table found.
# ---------------------------------------------------------------------------


def test_a_slow_target_does_not_cost_the_others_their_cadence():
    """The other cartridges keep their rate beside a slow World Builder read.

    Written to the symptom this time: the previous test measured how soon
    the fast cartridges were offered WITHIN one pass and passed against
    code that delivered them at 20% of their cadence, because the pass
    still waited for its slowest future and the loop slept the full poll
    on top. Beside a 0.3 s World Builder read at a 0.05 s poll, CV Lab
    must get at least 70% of its ideal deliveries over one second (the
    fixed code measures 85-90%; a loop that sleeps the full poll on top of
    the pass's own wait measures 50%).
    """
    import threading
    import time as _time

    def _snapshot_for(cartridge, result_type, world_id, session_id):
        if cartridge == "world_builder":
            _time.sleep(0.3)
        return _snapshot(f"{cartridge}-{_time.monotonic():.4f}")

    hub = ResultHub(
        _snapshot_for, clock=_time.monotonic, poll_seconds=0.05,
        heartbeat_seconds=0.0,
    )
    deliveries = {"cv_lab": 0, "world_builder": 0}

    async def _capture(payload):
        deliveries[payload["cartridge"]] += 1

    async def _run():
        channel = ConnectionChannel(hub, _capture, _time.monotonic)
        for cartridge in ("world_builder", "cv_lab"):
            await channel.add(Subscription(
                subscription_id=f"sub-{cartridge}", cartridge=cartridge,
                result_type="status", contract="c", world_id="w",
                session_id=None, cursor_status=None,
            ))
        await asyncio.sleep(1.0)
        await channel.close()

    asyncio.run(_run())
    ideal = 1.0 / 0.05
    assert deliveries["cv_lab"] >= ideal * 0.7, (
        f"cv_lab got {deliveries['cv_lab']} deliveries in 1 s beside a 0.3 s "
        f"world_builder read; ideal is ~{ideal:.0f}"
    )


def test_a_poisoned_target_recovers_once_its_watchers_have_been_failed_off(monkeypatch):
    """The abandonment cap must reach a target nobody is watching any more.

    The cap sat only inside the pass's loop over WATCHED targets, and a
    wedged target has no watchers by the time it is reached: the
    escalation fails them off after three deadlines, the cap is three
    deadlines plus. A reviewer counted 33 re-subscribes over 100 s
    against a future 95 s old, every one refused in 0.00 s. Here the
    subscriber is gone, the cap passes, and the next subscribe must get
    a fresh snapshot.
    """
    import threading

    from tower.results import publisher as publisher_module

    monkeypatch.setattr(publisher_module, "SNAPSHOT_TIMEOUT_SECONDS", 0.02)
    never = threading.Event()
    entered = {"n": 0}

    def _snapshot_for(cartridge, result_type, world_id, session_id):
        entered["n"] += 1
        if entered["n"] == 1:
            never.wait(timeout=60)
            return _snapshot("wedged")
        return _snapshot("recovered")

    clock = _fake_clock()
    hub = ResultHub(_snapshot_for, clock=clock, age_clock=clock)

    def _sub():
        return Subscription(
            subscription_id="sub-1", cartridge="world_builder",
            result_type="status", contract="c", world_id="w",
            session_id=None, cursor_status=None,
        )

    async def _run():
        # The wedge is dispatched by a subscribe that times out; nobody
        # is left watching.
        with pytest.raises(TimeoutError):
            await hub.first_snapshot(_sub(), timeout=0.01)
        clock.advance(0.02 * (publisher_module.SNAPSHOT_ABANDON_MULTIPLIER + 1))
        # No pass has looked at it; a subscribe must not be refused on
        # the strength of a thread that will never return. The subscribe
        # that abandons it dispatches afresh and answers at once (it
        # will not wait on its own replacement -- round 24); the next
        # subscribe collects the fresh snapshot.
        with pytest.raises(TimeoutError):
            await hub.first_snapshot(_sub(), timeout=1.0)
        snapshot = await hub.first_snapshot(_sub(), timeout=1.0)
        never.set()
        return snapshot.revision

    assert asyncio.run(_run()) == "recovered"
    assert entered["n"] == 2


def test_an_unwatched_wedge_is_abandoned_by_the_pass(monkeypatch):
    """The pass's own sweep reaches every in-flight future, watched or not."""
    import threading

    from tower.results import publisher as publisher_module

    monkeypatch.setattr(publisher_module, "SNAPSHOT_TIMEOUT_SECONDS", 0.02)
    never = threading.Event()

    def _snapshot_for(cartridge, result_type, world_id, session_id):
        if cartridge == "world_builder":
            never.wait(timeout=60)
        return _snapshot("x")

    clock = _fake_clock()
    hub = ResultHub(_snapshot_for, clock=clock, age_clock=clock)
    target = ("world_builder", "status", "w", None)

    async def _run():
        with pytest.raises(TimeoutError):
            await hub.first_snapshot(Subscription(
                subscription_id="sub-1", cartridge="world_builder",
                result_type="status", contract="c", world_id="w",
                session_id=None, cursor_status=None,
            ), timeout=0.01)
        assert target in hub._in_flight
        # Only a healthy, unrelated target is watched now.
        bystander = ConnectionChannel(hub, lambda payload: None, clock)
        await bystander.add(Subscription(
            subscription_id="sub-1", cartridge="cv_lab",
            result_type="status", contract="c", world_id=None,
            session_id=None, cursor_status=None,
        ))
        clock.advance(0.02 * (publisher_module.SNAPSHOT_ABANDON_MULTIPLIER + 1))
        await hub.poll_once()
        gone = target not in hub._in_flight
        await bystander.close()
        never.set()
        return gone

    assert asyncio.run(_run()), "an unwatched wedge was never abandoned"


def test_a_pass_does_not_forget_a_future_it_did_not_collect():
    """`collect` forgets THE FUTURE IT COLLECTED, not whatever the table holds.

    `first_snapshot` takes no pass lock. A subscribe that ran while a
    pass was parked in its wait discarded that pass's (done) future and
    dispatched a replacement; the pass then forgot BY TARGET and popped
    the replacement, and the next pass dispatched a third: two live
    threads for one target, which is the one bound the table exists to
    hold. Measured by a reviewer.
    """
    hub = ResultHub(lambda *args: _snapshot("x"), clock=lambda: 0.0)
    target = ("world_builder", "status", "w", None)

    async def _run():
        loop = asyncio.get_running_loop()
        # The pass's future: done, waiting to be collected.
        first = loop.create_future()
        first.set_result(_snapshot("first"))
        # The subscribe's replacement: installed meanwhile, still running.
        replacement = loop.create_future()
        hub._in_flight[target] = replacement
        hub._dispatched_at[target] = 0.0
        hub._watchers_at_dispatch[target] = set()
        hub._collect(target, first, 0.0)
        kept = hub._in_flight.get(target) is replacement
        replacement.cancel()
        return kept

    assert asyncio.run(_run()), "the pass popped a future it had not collected"


def test_a_failed_subscribe_leaves_no_watcher_behind(monkeypatch):
    """A subscription that timed out is never registered, so it must not
    stay in the watcher set for as long as the wedge lives (34 dead
    objects in 100 s, measured)."""
    import threading

    from tower.results import publisher as publisher_module

    monkeypatch.setattr(publisher_module, "SNAPSHOT_TIMEOUT_SECONDS", 1.0)
    never = threading.Event()

    def _snapshot_for(cartridge, result_type, world_id, session_id):
        never.wait(timeout=60)
        return _snapshot("x")

    hub = ResultHub(_snapshot_for, clock=lambda: 0.0)
    target = ("world_builder", "status", "w", None)

    async def _run():
        for n in range(5):
            with pytest.raises(TimeoutError):
                await hub.first_snapshot(Subscription(
                    subscription_id="sub-1", cartridge="world_builder",
                    result_type="status", contract="c", world_id="w",
                    session_id=None, cursor_status=None,
                ), timeout=0.01)
        left = len(hub._watchers_at_dispatch.get(target, set()))
        never.set()
        return left

    assert asyncio.run(_run()) == 0


@pytest.mark.parametrize("raised", [SystemExit, KeyboardInterrupt])
def test_a_producer_raising_a_base_exception_is_a_failure_not_a_shutdown(raised):
    """`SystemExit` from a producer thread must not exit the Tower.

    The thread transferred every `BaseException` into the future as
    itself and the collector caught `Exception`, so a producer's
    `SystemExit` ended the process with code 3 (130 for
    `KeyboardInterrupt`); a reviewer measured both. It is a failure of
    that target, counted and reported like any other.
    """
    def _snapshot_for(cartridge, result_type, world_id, session_id):
        raise raised()

    hub = ResultHub(_snapshot_for, clock=lambda: 0.0)

    async def _run():
        channel = ConnectionChannel(hub, lambda payload: None, lambda: 0.0)
        await channel.add(Subscription(
            subscription_id="sub-1", cartridge="world_builder",
            result_type="status", contract="c", world_id="w",
            session_id=None, cursor_status=None,
        ))
        await hub.poll_once()
        for _ in range(100):
            await asyncio.sleep(0.01)
            if hub._failures:
                break
        counted = dict(hub._failures)
        await channel.close()
        return counted

    assert asyncio.run(_run()), f"{raised.__name__} from the producer was not counted as a failure"


# ---------------------------------------------------------------------------
# Round 24: what the fifth reviewer of the in-flight table found.
# ---------------------------------------------------------------------------


def test_a_subscribe_that_abandoned_a_wedge_does_not_wait_on_its_replacement(monkeypatch):
    """After an abandon, the subscribe answers at once; the fresh thread is
    for the next one.

    The abandon-then-dispatch reset the target's age to zero, so the
    subscribe that did it then waited the FULL deadline inline in the
    connection's message loop -- and did so again at every cap interval
    for the life of the wedge. A reviewer measured 25 s of every 30
    blocked at iOS's re-subscribe cadence.
    """
    import threading
    import time as _time

    from tower.results import publisher as publisher_module

    monkeypatch.setattr(publisher_module, "SNAPSHOT_TIMEOUT_SECONDS", 1.0)
    never = threading.Event()
    entered = {"n": 0}

    def _snapshot_for(cartridge, result_type, world_id, session_id):
        entered["n"] += 1
        never.wait(timeout=60)
        return _snapshot("never")

    clock = _fake_clock()
    hub = ResultHub(_snapshot_for, clock=clock, age_clock=clock)

    def _sub():
        return Subscription(
            subscription_id="sub-1", cartridge="world_builder",
            result_type="status", contract="c", world_id="w",
            session_id=None, cursor_status=None,
        )

    async def _run():
        with pytest.raises(TimeoutError):
            await hub.first_snapshot(_sub(), timeout=0.01)
        clock.advance(1.0 * (publisher_module.SNAPSHOT_ABANDON_MULTIPLIER + 1))
        started = _time.monotonic()
        with pytest.raises(TimeoutError):
            await hub.first_snapshot(_sub(), timeout=10.0)
        waited = _time.monotonic() - started
        never.set()
        return waited

    waited = asyncio.run(_run())
    assert entered["n"] == 2, "the abandoning subscribe did not dispatch afresh"
    assert waited < 0.2, f"the abandoning subscribe waited {waited:.2f}s on its own replacement"


def test_ages_are_measured_on_the_age_clock_not_the_wall_clock(monkeypatch):
    """A wall-clock step must not re-poison or mass-fail anything.

    `build_hub` hands the hub `time.time`. A reviewer stepped it 600 s
    backwards: the abandonment cap became unreachable for the length of
    the step and every subscribe waited the full deadline again. Ages are
    on their own clock now; the wall clock is for the heartbeat only.
    """
    import threading

    from tower.results import publisher as publisher_module

    monkeypatch.setattr(publisher_module, "SNAPSHOT_TIMEOUT_SECONDS", 1.0)
    never = threading.Event()

    def _snapshot_for(cartridge, result_type, world_id, session_id):
        never.wait(timeout=60)
        return _snapshot("never")

    wall = _fake_clock()
    age = _fake_clock()
    hub = ResultHub(_snapshot_for, clock=wall, age_clock=age)
    target = ("world_builder", "status", "w", None)

    def _sub():
        return Subscription(
            subscription_id="sub-1", cartridge="world_builder",
            result_type="status", contract="c", world_id="w",
            session_id=None, cursor_status=None,
        )

    async def _run():
        with pytest.raises(TimeoutError):
            await hub.first_snapshot(_sub(), timeout=0.01)
        wall.advance(-600.0)          # the wall clock steps back
        age.advance(1.0 * (publisher_module.SNAPSHOT_ABANDON_MULTIPLIER + 1))
        past = hub._past_abandonment(target)
        never.set()
        return past

    assert asyncio.run(_run()), "a wall-clock step hid a target past its abandonment cap"


def test_abandonment_backs_off_for_a_read_that_stays_wedged(monkeypatch):
    """A permanently wedged read is retried at 30 s, 60 s, 120 s ... not
    every 30 s forever (one daemon thread per 30 s, 120 an hour, measured
    by a reviewer over 195 s of real time).
    """
    import threading

    from tower.results import publisher as publisher_module

    monkeypatch.setattr(publisher_module, "SNAPSHOT_TIMEOUT_SECONDS", 1.0)
    never = threading.Event()
    entered = {"n": 0}

    def _snapshot_for(cartridge, result_type, world_id, session_id):
        entered["n"] += 1
        never.wait(timeout=120)
        return _snapshot("never")

    clock = _fake_clock()
    hub = ResultHub(_snapshot_for, clock=clock, age_clock=clock)

    async def _run():
        channel = ConnectionChannel(hub, lambda payload: None, clock)
        await channel.add(Subscription(
            subscription_id="sub-1", cartridge="world_builder",
            result_type="status", contract="c", world_id="w",
            session_id=None, cursor_status=None,
        ))
        # Ten simulated minutes in 3 s steps of passes.
        for _ in range(200):
            clock.advance(3.0)
            await hub.poll_once()
        await channel.close()
        never.set()

    asyncio.run(_run())
    # Caps of 3, 6, 12, 24, 48, 96, then 120, 120, 120 s (the ceiling;
    # cumulative 549 s) plus the first dispatch: about ten entries over
    # 600 s, against twenty at a flat 30 s.
    assert 6 <= entered["n"] <= 12, f"the producer was entered {entered['n']} times in 600 s"


def test_a_carried_over_snapshot_is_replaced_in_the_pass_that_collects_it():
    """A healthy-but-slow target is re-dispatched the moment its previous
    snapshot is collected, not one poll later.

    The pass that collected a carried-over done future dispatched
    nothing for that target until the next pass: a reviewer measured a
    0.55 s read and a 0.80 s read both delivering every 1.52 s at a
    0.5 s poll -- a whole interval of dead time per delivery.
    """
    import threading

    gates = [threading.Event(), threading.Event()]
    entered = {"n": 0}

    def _snapshot_for(cartridge, result_type, world_id, session_id):
        entered["n"] += 1
        if entered["n"] <= 2:
            gates[entered["n"] - 1].wait(timeout=30)   # slow, both times
        return _snapshot(f"rev-{entered['n']}")

    hub = ResultHub(_snapshot_for, clock=lambda: 0.0, poll_seconds=0.01)
    target = ("world_builder", "status", "w", None)
    delivered = []

    async def _capture(payload):
        delivered.append(payload)

    async def _run():
        channel = ConnectionChannel(hub, _capture, lambda: 0.0)
        await channel.add(Subscription(
            subscription_id="sub-1", cartridge="world_builder",
            result_type="status", contract="c", world_id="w",
            session_id=None, cursor_status=None,
        ))
        await hub.poll_once()                 # dispatches rev-1, which stalls
        assert target in hub._in_flight
        first = hub._in_flight[target]
        gates[0].set()
        for _ in range(100):
            await asyncio.sleep(0.005)
            if first.done():
                break
        await hub.poll_once()                 # collects rev-1 ...
        replaced = hub._in_flight.get(target) # ... and must have dispatched rev-2
        gates[1].set()
        await asyncio.sleep(0.05)
        await channel.close()
        return replaced is not None and replaced is not first

    assert asyncio.run(_run()), "the collecting pass left the target idle until the next one"
    assert entered["n"] >= 2
    assert [m["payload"]["revision_marker"] for m in delivered][:1] == ["rev-1"], (
        "the carried-over snapshot was replaced but its delivery was lost"
    )


# ---------------------------------------------------------------------------
# Round 25: what the sixth publisher reviewer and the fourth rehearsal found.
# ---------------------------------------------------------------------------


def test_a_slow_but_healthy_first_snapshot_is_answered_not_failed():
    """A first snapshot slower than the phone's stall bound is still ANSWERED.

    Round 24 capped the inline wait at 1.5 s. On the phone `snapshot_failed`
    is terminal -- `.failed`, no retry; only `channel_failed` is retried --
    so every healthy read slower than the cap left the World Builder
    screen dead for the life of the connection: 0 of 8 subscribes on a
    real socket at a 2.1 s read. The stall self-heals instead (the phone
    replaces the socket and the re-subscribe joins the same future). The
    wait is the deadline; a 1.7 s read is answered.
    """
    import time as _time

    def _snapshot_for(cartridge, result_type, world_id, session_id):
        _time.sleep(1.7)
        return _snapshot("slow-but-fine")

    hub = ResultHub(_snapshot_for, clock=_time.monotonic)

    async def _run():
        return await hub.first_snapshot(Subscription(
            subscription_id="sub-1", cartridge="world_builder",
            result_type="status", contract="c", world_id="w",
            session_id=None, cursor_status=None,
        ), timeout=10.0)

    assert asyncio.run(_run()).revision == "slow-but-fine"


def test_a_young_uncollected_snapshot_is_handed_over_not_discarded():
    """A done future that finished a moment ago is the freshest state there
    is, HOWEVER LONG IT TOOK to compute.

    The phone's socket stalled at 2 s of a 2.1 s read and it came back to
    re-subscribe; the read had finished 0.4 s earlier, uncollected. The
    first handover measured age from DISPATCH -- 2.5 s, over the line --
    so the case it was written for could never fire: 0 of 10 sockets at
    any read between 2.0 and 2.4 s, measured on a real socket. Staleness
    is how long a result has sat (here: since it completed), not how long
    it took. Real time, on purpose.
    """
    import time as _time

    entered = {"n": 0}

    def _snapshot_for(cartridge, result_type, world_id, session_id):
        entered["n"] += 1
        _time.sleep(0.3)                    # slower than the heartbeat below
        return _snapshot(f"rev-{entered['n'] - 1}")

    hub = ResultHub(
        _snapshot_for, clock=_time.monotonic, poll_seconds=0.05, heartbeat_seconds=0.1,
    )

    def _sub():
        return Subscription(
            subscription_id="sub-1", cartridge="world_builder",
            result_type="status", contract="c", world_id="w",
            session_id=None, cursor_status=None,
        )

    async def _run():
        target = ("world_builder", "status", "w", None)
        future = hub._dispatch(target, _sub(), set())     # the dropped phone's read
        for _ in range(200):
            await asyncio.sleep(0.005)
            if future.done():
                break
        await asyncio.sleep(0.02)                         # finished 0.02 s ago; took 0.3 s
        young = (await hub.first_snapshot(_sub(), timeout=1.0)).revision
        future = hub._dispatch(target, _sub(), set())
        for _ in range(200):
            await asyncio.sleep(0.005)
            if future.done():
                break
        await asyncio.sleep(0.25)                         # sat for 0.25 s > the 0.1 s heartbeat
        old = (await hub.first_snapshot(_sub(), timeout=1.0)).revision
        return young, old

    young, old = asyncio.run(_run())
    assert young == "rev-0", "a result that finished 0.02 s ago was discarded because it took 0.3 s"
    assert old == "rev-2", "a result that sat past the heartbeat was handed over as fresh"


def test_recovery_after_a_fault_clears_is_bounded():
    """A target is never refused for more than `SNAPSHOT_ABANDON_MAX_SECONDS`
    after its read recovers, whatever the streak -- driven, not just the
    constant.

    Recovery needs an abandonment, and the cap doubled to 600 s: a
    reviewer measured 592 s of silent refusal after a fault cleared.
    """
    import threading

    from tower.results import publisher as publisher_module

    never = threading.Event()
    entered = {"n": 0}

    def _snapshot_for(cartridge, result_type, world_id, session_id):
        entered["n"] += 1
        if entered["n"] == 1:
            never.wait(timeout=60)          # the fault ...
        return _snapshot("recovered")      # ... which has cleared for new reads

    clock = _fake_clock()
    hub = ResultHub(_snapshot_for, clock=clock, age_clock=clock)
    target = ("world_builder", "status", "w", None)

    def _sub():
        return Subscription(
            subscription_id="sub-1", cartridge="world_builder",
            result_type="status", contract="c", world_id="w",
            session_id=None, cursor_status=None,
        )

    async def _run():
        hub._abandon_streak[target] = 9     # a long-standing wedge
        with pytest.raises(TimeoutError):
            await hub.first_snapshot(_sub(), timeout=0.01)
        # The fault clears now. Subscribe every 10 s; how long until answered?
        waited = 0.0
        while waited <= 1000.0:
            clock.advance(10.0)
            waited += 10.0
            await asyncio.sleep(0.01)       # the loop runs between a phone's attempts
            try:
                snapshot = await hub.first_snapshot(_sub(), timeout=1.0)
            except TimeoutError:
                continue
            never.set()
            return waited, snapshot.revision
        never.set()
        return waited, None

    waited, revision = asyncio.run(_run())
    assert revision == "recovered"
    assert waited <= publisher_module.SNAPSHOT_ABANDON_MAX_SECONDS + 10.0, (
        f"the target stayed refused for {waited:.0f}s after the fault cleared"
    )
    assert publisher_module.SNAPSHOT_ABANDON_MAX_SECONDS <= 120.0


def test_distinct_targets_cannot_mint_unbounded_threads():
    """A connection choosing world_ids cannot mint a wedged thread per
    choice -- and cannot lock anyone else out either.

    A reviewer subscribed 400 distinct world_ids against a wedged read
    and got 400 live threads in 0.3 s. The first cap was GLOBAL, and the
    next reviewer showed ~21 hostile connections holding it refused every
    legitimate phone's subscribe on every target, with a reason the phone
    treats as terminal. The cap is per connection: this one is walled at
    `MAX_IN_FLIGHT_PER_CONNECTION`, and a second connection's healthy
    target is answered while the wall stands.
    """
    import threading

    from tower.results import publisher as publisher_module
    from tower.results.publisher import TooManyTargetsInFlight

    never = threading.Event()
    entered = {"n": 0}

    def _snapshot_for(cartridge, result_type, world_id, session_id):
        entered["n"] += 1
        if world_id.startswith("hostile"):
            never.wait(timeout=60)
        return _snapshot("fine")

    hub = ResultHub(_snapshot_for, clock=lambda: 0.0)
    hostile, phone = object(), object()

    async def _run():
        refused = 0
        for n in range(400):
            try:
                await hub.first_snapshot(Subscription(
                    subscription_id="sub-1", cartridge="world_builder",
                    result_type="status", contract="c", world_id=f"hostile-{n}",
                    session_id=None, cursor_status=None,
                ), timeout=0.001, owner=hostile)
            except TooManyTargetsInFlight:
                refused += 1
            except TimeoutError:
                pass
        in_flight = len(hub._in_flight)
        answered = await hub.first_snapshot(Subscription(
            subscription_id="sub-1", cartridge="world_builder",
            result_type="status", contract="c", world_id="the-phones-world",
            session_id=None, cursor_status=None,
        ), timeout=1.0, owner=phone)
        never.set()
        return refused, in_flight, answered.revision

    refused, in_flight, answered = asyncio.run(_run())
    limit = publisher_module.MAX_IN_FLIGHT_PER_CONNECTION
    assert in_flight <= limit, f"{in_flight} snapshots in flight for one hostile connection"
    assert refused == 400 - limit
    assert answered == "fine", "a legitimate phone was locked out by another connection's wedges"


def test_wedged_futures_minted_by_failed_subscribes_are_swept_by_the_next_subscribe(monkeypatch):
    """The abandon sweep must not depend on a poll loop that a failed
    subscribe never started."""
    import threading

    from tower.results import publisher as publisher_module

    monkeypatch.setattr(publisher_module, "SNAPSHOT_TIMEOUT_SECONDS", 1.0)
    never = threading.Event()

    def _snapshot_for(cartridge, result_type, world_id, session_id):
        never.wait(timeout=60)
        return _snapshot("never")

    clock = _fake_clock()
    hub = ResultHub(_snapshot_for, clock=clock, age_clock=clock)

    def _sub(world_id):
        return Subscription(
            subscription_id="sub-1", cartridge="world_builder",
            result_type="status", contract="c", world_id=world_id,
            session_id=None, cursor_status=None,
        )

    async def _run():
        with pytest.raises(TimeoutError):
            await hub.first_snapshot(_sub("first"), timeout=0.01)
        first = ("world_builder", "status", "first", None)
        assert first in hub._in_flight
        clock.advance(1.0 * (publisher_module.SNAPSHOT_ABANDON_MULTIPLIER + 1))
        with pytest.raises(TimeoutError):
            await hub.first_snapshot(_sub("second"), timeout=0.01)
        swept = first not in hub._in_flight
        never.set()
        return swept

    assert asyncio.run(_run()), "a wedge nobody could subscribe to was never abandoned"


def test_a_failed_subscribe_does_not_erase_a_registered_watcher():
    """The watcher set is seeded with the target's live watchers, and a
    failed subscribe removes only itself.

    `_dispatch` clobbered the set with just the new subscription; the
    failure path then emptied it, and the pass that met the finished
    future discarded it as "computed for nobody" while a registered
    watcher waited -- an extra full read cycle, measured.
    """
    import threading

    gate = threading.Event()
    produced = []

    def _snapshot_for(cartridge, result_type, world_id, session_id):
        produced.append(len(produced))
        gate.wait(timeout=30)
        return _snapshot(f"rev-{len(produced) - 1}")

    hub = ResultHub(_snapshot_for, clock=lambda: 0.0)
    target = ("world_builder", "status", "w", None)
    delivered = []

    async def _capture(payload):
        delivered.append(payload)

    def _sub(subscription_id):
        return Subscription(
            subscription_id=subscription_id, cartridge="world_builder",
            result_type="status", contract="c", world_id="w",
            session_id=None, cursor_status=None,
        )

    async def _run():
        watcher = ConnectionChannel(hub, _capture, lambda: 0.0)
        registered = _sub("sub-A")
        await watcher.add(registered)
        # The target is free (no pass has run yet); a second phone's
        # subscribe dispatches it and times out.
        with pytest.raises(TimeoutError):
            await hub.first_snapshot(_sub("sub-B"), timeout=0.05)
        still_watching = registered in hub._watchers_at_dispatch.get(target, set())
        gate.set()
        for _ in range(200):
            await asyncio.sleep(0.01)
            if delivered:
                break
        await watcher.close()
        return still_watching

    assert asyncio.run(_run()), "the registered watcher was erased by a failed subscribe"
    assert [m["payload"]["revision_marker"] for m in delivered][:1] == ["rev-0"], (
        "the snapshot the registered watcher was waiting for was discarded"
    )


def test_the_loop_sleeps_at_least_ten_milliseconds():
    """A 1 ms poll is a 10 ms loop, not a spin (6,000 passes a second, measured)."""
    import time as _time

    entered = {"n": 0}

    def _snapshot_for(cartridge, result_type, world_id, session_id):
        entered["n"] += 1
        return _snapshot("x")

    hub = ResultHub(_snapshot_for, clock=_time.monotonic, poll_seconds=0.001)

    async def _run():
        channel = ConnectionChannel(hub, lambda payload: None, _time.monotonic)
        await channel.add(Subscription(
            subscription_id="sub-1", cartridge="world_builder",
            result_type="status", contract="c", world_id="w",
            session_id=None, cursor_status=None,
        ))
        await asyncio.sleep(0.3)
        await channel.close()

    asyncio.run(_run())
    assert entered["n"] <= 60, f"{entered['n']} passes in 0.3 s at a 1 ms poll"


def test_ages_default_to_the_monotonic_clock():
    """`build_hub` passes only `clock`; the durations must not follow it."""
    import threading

    never = threading.Event()

    def _snapshot_for(cartridge, result_type, world_id, session_id):
        never.wait(timeout=60)
        return _snapshot("never")

    wall = _fake_clock()
    hub = ResultHub(_snapshot_for, clock=wall)
    target = ("world_builder", "status", "w", None)

    async def _run():
        with pytest.raises(TimeoutError):
            await hub.first_snapshot(Subscription(
                subscription_id="sub-1", cartridge="world_builder",
                result_type="status", contract="c", world_id="w",
                session_id=None, cursor_status=None,
            ), timeout=0.01)
        wall.advance(10_000.0)
        past = hub._past_abandonment(target)
        never.set()
        return past

    assert not asyncio.run(_run()), "a wall-clock jump aged a snapshot past its cap"


# ---------------------------------------------------------------------------
# Round 26: what the seventh publisher reviewer found.
# ---------------------------------------------------------------------------


def test_the_young_handover_reaches_the_registered_watchers():
    """A result handed to a newcomer is offered to the watchers it was
    computed for as well -- not taken from them.

    The handover returned the result to the subscriber alone and forgot
    the target; a reviewer watched the watcher's connection receive `r3`
    and `r4` and never `r2`, which the newcomer had taken.
    """
    import threading

    gate = threading.Event()
    produced = []

    def _snapshot_for(cartridge, result_type, world_id, session_id):
        produced.append(len(produced))
        gate.wait(timeout=30)
        return _snapshot(f"rev-{len(produced) - 1}")

    hub = ResultHub(_snapshot_for, clock=lambda: 0.0, heartbeat_seconds=2.0)
    target = ("world_builder", "status", "w", None)
    delivered = []

    async def _capture(payload):
        delivered.append(payload)

    def _sub(subscription_id):
        return Subscription(
            subscription_id=subscription_id, cartridge="world_builder",
            result_type="status", contract="c", world_id="w",
            session_id=None, cursor_status=None,
        )

    async def _run():
        watcher = ConnectionChannel(hub, _capture, lambda: 0.0)
        registered = _sub("sub-A")
        await watcher.add(registered)
        # No pass may race this: the hub's loop is stopped, the channel
        # stays attached with its sender.
        hub._task.cancel()
        hub._task = None
        future = hub._dispatch(target, registered, {registered})
        gate.set()
        for _ in range(200):
            await asyncio.sleep(0.005)
            if future.done():
                break
        newcomer = (await hub.first_snapshot(_sub("sub-B"), timeout=1.0)).revision
        for _ in range(100):
            await asyncio.sleep(0.01)
            if delivered:
                break
        await watcher.close()
        return newcomer

    assert asyncio.run(_run()) == "rev-0"
    assert [m["payload"]["revision_marker"] for m in delivered] == ["rev-0"], (
        "the registered watcher never received the result a newcomer was handed"
    )
    assert len(produced) == 1


def test_a_replaced_socket_takes_its_subscription_back():
    """A subscribe cancelled mid-wait -- the phone replaced its socket --
    leaves no dead Subscription in the watcher set (four retained per
    wedge, measured)."""
    import threading

    never = threading.Event()

    def _snapshot_for(cartridge, result_type, world_id, session_id):
        never.wait(timeout=60)
        return _snapshot("never")

    hub = ResultHub(_snapshot_for, clock=lambda: 0.0)
    target = ("world_builder", "status", "w", None)

    async def _run():
        subscription = Subscription(
            subscription_id="sub-1", cartridge="world_builder",
            result_type="status", contract="c", world_id="w",
            session_id=None, cursor_status=None,
        )
        task = asyncio.ensure_future(hub.first_snapshot(subscription, timeout=5.0))
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        left = subscription in hub._watchers_at_dispatch.get(target, set())
        never.set()
        return left

    assert not asyncio.run(_run()), "a cancelled subscribe stayed in the watcher set"


def test_a_cached_exception_is_counted_and_not_re_raised_for_the_heartbeat():
    """A young done future holding an exception is a failure to count and
    move past, not a result to hand over."""
    entered = {"n": 0}

    def _snapshot_for(cartridge, result_type, world_id, session_id):
        entered["n"] += 1
        if entered["n"] == 1:
            raise RuntimeError("the producer exploded once")
        return _snapshot("fine")

    hub = ResultHub(_snapshot_for, clock=lambda: 0.0, heartbeat_seconds=2.0)
    target = ("world_builder", "status", "w", None)

    def _sub():
        return Subscription(
            subscription_id="sub-1", cartridge="world_builder",
            result_type="status", contract="c", world_id="w",
            session_id=None, cursor_status=None,
        )

    async def _run():
        future = hub._dispatch(target, _sub(), set())
        for _ in range(200):
            await asyncio.sleep(0.005)
            if future.done():
                break
        snapshot = await hub.first_snapshot(_sub(), timeout=1.0)
        return snapshot.revision, dict(hub._failures)

    revision, failures = asyncio.run(_run())
    assert revision == "fine"
    assert failures.get(target) == 1, "the cached exception was not counted as a failure"


# ---------------------------------------------------------------------------
# Round 27: what the eighth publisher reviewer found.
# ---------------------------------------------------------------------------


def test_done_results_nobody_waits_for_do_not_fill_the_table():
    """Subscribe-and-drop sockets against a HEALTHY read must not disable
    the channel.

    300 sockets each subscribed to a distinct world_id and dropped 20 ms
    later; the reads finished for nobody, the table held 256 done futures
    nobody watched, the global cap was reached, and every later subscribe
    was refused -- so no channel ever attached, no pass ever ran, and
    nothing ever swept them. Still refused 130 s later. The subscribe
    path frees done futures nobody is waiting for.
    """
    import time as _time

    from tower.results import publisher as publisher_module

    def _snapshot_for(cartridge, result_type, world_id, session_id):
        _time.sleep(0.05)
        return _snapshot("fine")

    hub = ResultHub(_snapshot_for, clock=_time.monotonic)

    async def _run():
        for n in range(publisher_module.MAX_IN_FLIGHT_TARGETS + 40):
            task = asyncio.ensure_future(hub.first_snapshot(Subscription(
                subscription_id="sub-1", cartridge="world_builder",
                result_type="status", contract="c", world_id=f"w{n}",
                session_id=None, cursor_status=None,
            ), timeout=10.0, owner=object()))
            await asyncio.sleep(0.002)
            task.cancel()                          # the socket drops mid-wait
            try:
                await task
            except asyncio.CancelledError:
                pass
        await asyncio.sleep(0.3)                   # every read finishes, for nobody
        stranded = len(hub._in_flight)
        answered = await hub.first_snapshot(Subscription(
            subscription_id="sub-1", cartridge="world_builder",
            result_type="status", contract="c", world_id="the-phone",
            session_id=None, cursor_status=None,
        ), timeout=5.0, owner=object())
        return stranded, answered.revision, len(hub._in_flight)

    stranded, answered, after = asyncio.run(_run())
    assert answered == "fine", f"a phone was refused behind {stranded} done results nobody waited for"
    assert after <= 1, f"{after} entries remain after the sweep"


def test_the_global_cap_still_stands_against_pending_wedges():
    """The global wall (256) refuses a 257th DISTINCT connection's wedge."""
    import threading

    from tower.results import publisher as publisher_module
    from tower.results.publisher import TooManyTargetsInFlight

    never = threading.Event()

    def _snapshot_for(cartridge, result_type, world_id, session_id):
        never.wait(timeout=60)
        return _snapshot("never")

    hub = ResultHub(_snapshot_for, clock=lambda: 0.0)
    limit = publisher_module.MAX_IN_FLIGHT_TARGETS

    async def _run():
        refused = 0
        for n in range(limit + 8):
            try:
                await hub.first_snapshot(Subscription(
                    subscription_id="sub-1", cartridge="world_builder",
                    result_type="status", contract="c", world_id=f"w{n}",
                    session_id=None, cursor_status=None,
                ), timeout=0.001, owner=object())   # a new connection each time
            except TooManyTargetsInFlight:
                refused += 1
            except TimeoutError:
                pass
        in_flight = len(hub._in_flight)
        never.set()
        return refused, in_flight

    refused, in_flight = asyncio.run(_run())
    assert in_flight == limit
    assert refused == 8


def test_a_closed_channel_is_not_kept_alive_by_its_dispatches():
    """`detach` drops the owner references of the channel leaving, so a
    closed channel (and its socket) is not retained for as long as a
    wedged future lives, and does not count against anyone."""
    import threading

    never = threading.Event()

    def _snapshot_for(cartridge, result_type, world_id, session_id):
        never.wait(timeout=60)
        return _snapshot("never")

    hub = ResultHub(_snapshot_for, clock=lambda: 0.0)

    async def _run():
        channel = ConnectionChannel(hub, lambda payload: None, lambda: 0.0)
        with pytest.raises(TimeoutError):
            await hub.first_snapshot(Subscription(
                subscription_id="sub-1", cartridge="world_builder",
                result_type="status", contract="c", world_id="w",
                session_id=None, cursor_status=None,
            ), timeout=0.01, owner=channel)
        assert channel in hub._owner_of.values()
        await hub.detach(channel)
        retained = channel in hub._owner_of.values()
        never.set()
        return retained

    assert not asyncio.run(_run()), "a detached channel was still referenced as an owner"


def test_a_future_is_collected_once():
    """The young handover and a parked pass may both reach one future; a
    failure must be counted once, not twice (two genuine failures then
    tripped the three-strike escalation)."""
    hub = ResultHub(lambda *args: _snapshot("x"), clock=lambda: 0.0)
    target = ("world_builder", "status", "w", None)

    async def _run():
        future = asyncio.get_running_loop().create_future()
        future.set_exception(RuntimeError("once"))
        hub._in_flight[target] = future
        hub._dispatched_at[target] = 0.0
        hub._watchers_at_dispatch[target] = set()
        hub._collect(target, future, 0.0)
        hub._collect(target, future, 0.0)
        return dict(hub._failures)

    assert asyncio.run(_run()).get(target) == 1


def test_an_abandoned_threads_late_return_does_not_stamp_the_replacement(monkeypatch):
    """`_completed_at` for a target's NEW future is never overwritten by
    the OLD, abandoned future's callback."""
    import threading

    from tower.results import publisher as publisher_module

    monkeypatch.setattr(publisher_module, "SNAPSHOT_TIMEOUT_SECONDS", 1.0)
    old_gate = threading.Event()
    entered = {"n": 0}

    def _snapshot_for(cartridge, result_type, world_id, session_id):
        entered["n"] += 1
        if entered["n"] == 1:
            old_gate.wait(timeout=60)
        return _snapshot(f"rev-{entered['n']}")

    clock = _fake_clock()
    hub = ResultHub(_snapshot_for, clock=clock, age_clock=clock)
    target = ("world_builder", "status", "w", None)

    def _sub():
        return Subscription(
            subscription_id="sub-1", cartridge="world_builder",
            result_type="status", contract="c", world_id="w",
            session_id=None, cursor_status=None,
        )

    async def _run():
        with pytest.raises(TimeoutError):
            await hub.first_snapshot(_sub(), timeout=0.01)     # the wedge
        clock.advance(1.0 * (publisher_module.SNAPSHOT_ABANDON_MULTIPLIER + 1))
        with pytest.raises(TimeoutError):
            await hub.first_snapshot(_sub(), timeout=0.01)     # abandons, re-dispatches
        replacement = hub._in_flight[target]
        for _ in range(200):
            await asyncio.sleep(0.005)
            if replacement.done():
                break
        stamped = hub._completed_at.get(target)
        clock.advance(100.0)
        old_gate.set()                                         # the old thread returns now
        await asyncio.sleep(0.1)
        return stamped, hub._completed_at.get(target)

    stamped, after = asyncio.run(_run())
    assert stamped is not None and after == stamped, (
        "the abandoned thread's late return re-stamped the replacement's completion"
    )


def test_forgetting_a_target_clears_every_table():
    """`_forget` is the one place a target leaves its in-flight record, and
    it leaves every table keyed on the target. (`_failures` and
    `_abandon_streak` deliberately outlive it -- they are the target's
    history, pruned by the pass -- and `_collected` is keyed on the
    future and weak.)"""
    import time as _time

    hub = ResultHub(lambda *args: _snapshot("x"), clock=_time.monotonic)

    async def _run():
        await hub.first_snapshot(Subscription(
            subscription_id="sub-1", cartridge="world_builder",
            result_type="status", contract="c", world_id="w",
            session_id=None, cursor_status=None,
        ), timeout=5.0, owner=object())
        return {
            name: len(getattr(hub, name)) for name in (
                "_in_flight", "_dispatched_at", "_watchers_at_dispatch",
                "_completed_at", "_owner_of",
            )
        }

    sizes = asyncio.run(_run())
    assert all(size == 0 for size in sizes.values()), sizes


def test_the_young_threshold_is_the_heartbeat_not_the_poll():
    """A result that sat longer than the poll but shorter than the
    heartbeat is still fresh -- the heartbeat is how old a snapshot the
    channel itself is content to send."""
    import time as _time

    entered = {"n": 0}

    def _snapshot_for(cartridge, result_type, world_id, session_id):
        entered["n"] += 1
        return _snapshot(f"rev-{entered['n'] - 1}")

    hub = ResultHub(
        _snapshot_for, clock=_time.monotonic, poll_seconds=0.05, heartbeat_seconds=0.5,
    )

    def _sub():
        return Subscription(
            subscription_id="sub-1", cartridge="world_builder",
            result_type="status", contract="c", world_id="w",
            session_id=None, cursor_status=None,
        )

    async def _run():
        target = ("world_builder", "status", "w", None)
        future = hub._dispatch(target, _sub(), set())
        for _ in range(200):
            await asyncio.sleep(0.005)
            if future.done():
                break
        await asyncio.sleep(0.2)                    # past the poll, inside the heartbeat
        return (await hub.first_snapshot(_sub(), timeout=1.0)).revision

    assert asyncio.run(_run()) == "rev-0"
    assert entered["n"] == 1


# ---------------------------------------------------------------------------
# Round 28: what the ninth publisher reviewer found (no blocking defect).
# ---------------------------------------------------------------------------


def test_collected_futures_do_not_accumulate_with_nobody_attached():
    """The once-per-future record must die with the future.

    It was a plain set cleared at the start of each pass; a pass runs only
    while a channel is attached, and the cached-exception handover ends in
    `snapshot_failed`, which attaches nothing -- 35,000 retained futures
    an hour, measured, each holding its exception, its traceback and
    through it the hub.
    """
    import gc

    calls = {"n": 0}

    def _snapshot_for(cartridge, result_type, world_id, session_id):
        calls["n"] += 1
        if calls["n"] % 2 == 1:
            raise RuntimeError("every other read explodes")
        return _snapshot("fine")

    hub = ResultHub(_snapshot_for, clock=lambda: 0.0, heartbeat_seconds=2.0)

    def _sub(n):
        return Subscription(
            subscription_id="sub-1", cartridge="world_builder",
            result_type="status", contract="c", world_id=f"w{n}",
            session_id=None, cursor_status=None,
        )

    # pytest's log capture would keep every `logger.exception` record --
    # and through its traceback the frame, the closure and the future --
    # alive for the test; the Tower has no such capture. Silence the
    # logger so what is measured is the hub's own retention.
    from tower.results import publisher as publisher_module
    publisher_module.logger.disabled = True

    async def _run():
        for n in range(60):
            target = ("world_builder", "status", f"w{n}", None)
            future = hub._dispatch(target, _sub(n), set())     # raises
            for _ in range(200):
                await asyncio.sleep(0.002)
                if future.done():
                    break
            # The handover finds the cached exception, counts it, and
            # dispatches afresh -- which succeeds.
            await hub.first_snapshot(_sub(n), timeout=1.0)
            del future
        gc.collect()
        return len(hub._collected), len(hub._in_flight)

    try:
        collected, in_flight = asyncio.run(_run())
    finally:
        publisher_module.logger.disabled = False
    assert in_flight == 0
    assert collected <= 2, f"{collected} collected futures retained with nobody attached"


def test_a_producers_own_timeout_is_not_echoed_onto_the_wire():
    """Only the hub's own timeouts and refusals put their text on the wire;
    a producer's `TimeoutError` may carry a path, an errno and a pid."""
    from tower.results.publisher import SnapshotTimeout, TooManyTargetsInFlight
    from tower.routes.results_ws import _first_snapshot_failure_message

    internals = TimeoutError(r"C:\Users\somebody\world\manifest.json errno 13 pid 4242")
    assert "manifest.json" not in _first_snapshot_failure_message(internals)
    assert "TimeoutError" in _first_snapshot_failure_message(internals)
    assert _first_snapshot_failure_message(SnapshotTimeout("still running after 10.0s")) == (
        "still running after 10.0s"
    )
    assert "already has" in _first_snapshot_failure_message(
        TooManyTargetsInFlight("this connection already has 8 results being computed")
    )


def test_a_departing_channel_unowns_only_its_own_dispatches():
    """`detach` releases the leaving channel's owner references and nobody
    else's: a still-live connection keeps its per-connection count."""
    import threading

    from tower.results import publisher as publisher_module

    never = threading.Event()

    def _snapshot_for(cartridge, result_type, world_id, session_id):
        never.wait(timeout=60)
        return _snapshot("never")

    hub = ResultHub(_snapshot_for, clock=lambda: 0.0)

    def _sub(world_id):
        return Subscription(
            subscription_id="sub-1", cartridge="world_builder",
            result_type="status", contract="c", world_id=world_id,
            session_id=None, cursor_status=None,
        )

    async def _run():
        leaving = ConnectionChannel(hub, lambda payload: None, lambda: 0.0)
        staying = ConnectionChannel(hub, lambda payload: None, lambda: 0.0)
        for n in range(3):
            with pytest.raises(TimeoutError):
                await hub.first_snapshot(_sub(f"leaving-{n}"), timeout=0.01, owner=leaving)
        for n in range(publisher_module.MAX_IN_FLIGHT_PER_CONNECTION):
            with pytest.raises(TimeoutError):
                await hub.first_snapshot(_sub(f"staying-{n}"), timeout=0.01, owner=staying)
        await hub.detach(leaving)
        owned_by_staying = sum(1 for holder in hub._owner_of.values() if holder is staying)
        owned_by_leaving = sum(1 for holder in hub._owner_of.values() if holder is leaving)
        # ...and the staying connection is still at its wall.
        from tower.results.publisher import TooManyTargetsInFlight
        with pytest.raises(TooManyTargetsInFlight):
            await hub.first_snapshot(_sub("staying-more"), timeout=0.01, owner=staying)
        never.set()
        return owned_by_staying, owned_by_leaving

    owned_by_staying, owned_by_leaving = asyncio.run(_run())
    assert owned_by_leaving == 0
    assert owned_by_staying == 8, "a departing channel unowned another connection's dispatches"


def test_the_sweep_leaves_a_watched_result_for_its_watcher():
    """The subscribe path frees finished work NOBODY is waiting for -- not
    a result a registered watcher is about to receive."""
    import threading

    gate = threading.Event()
    entered = {"n": 0}

    def _snapshot_for(cartridge, result_type, world_id, session_id):
        entered["n"] += 1
        if cartridge == "world_builder":
            gate.wait(timeout=30)
            return _snapshot(f"wb-{entered['n']}")
        return _snapshot("cv")

    hub = ResultHub(_snapshot_for, clock=lambda: 0.0, heartbeat_seconds=2.0)
    wb_target = ("world_builder", "status", "w", None)
    delivered = []

    async def _capture(payload):
        delivered.append(payload)

    async def _run():
        watcher = ConnectionChannel(hub, _capture, lambda: 0.0)
        registered = Subscription(
            subscription_id="sub-A", cartridge="world_builder",
            result_type="status", contract="c", world_id="w",
            session_id=None, cursor_status=None,
        )
        await watcher.add(registered)
        hub._task.cancel()                          # no pass may race this
        hub._task = None
        future = hub._dispatch(wb_target, registered, {registered})
        gate.set()
        for _ in range(200):
            await asyncio.sleep(0.005)
            if future.done():
                break
        wb_entries = entered["n"]
        # Another connection subscribes to something else: its sweep runs.
        await hub.first_snapshot(Subscription(
            subscription_id="sub-1", cartridge="cv_lab",
            result_type="status", contract="c", world_id=None,
            session_id=None, cursor_status=None,
        ), timeout=1.0, owner=object())
        still_there = hub._in_flight.get(wb_target) is future
        # The watcher's own pass collects it.
        await hub.poll_once()
        for _ in range(100):
            await asyncio.sleep(0.01)
            if delivered:
                break
        await watcher.close()
        return still_there, wb_entries

    still_there, wb_entries = asyncio.run(_run())
    assert still_there, "another connection's subscribe swept a result a watcher was waiting for"
    # The first delivery is THAT result (later passes may add newer ones).
    assert [m["payload"]["revision_marker"] for m in delivered][:1] == [f"wb-{wb_entries}"]
