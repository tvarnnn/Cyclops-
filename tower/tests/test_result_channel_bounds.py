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
    hub = ResultHub(_snapshot_for, clock=clock)

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
        clock.advance(5.0)   # well past the deadline
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
    hub = ResultHub(_snapshot_for, clock=clock)
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

    hub = ResultHub(_snapshot_for, clock=lambda: 0.0)

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
