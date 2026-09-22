"""The Tower's Windows listener must survive a connection reset in the backlog.

The failure this guards is CPython gh-93821: `asyncio.ProactorEventLoop`
closes its LISTENING socket, and never re-arms accept, when a connection
queued in the kernel backlog is reset before the loop accepts it. The
process and the loop keep running; the port just disappears. A physical
Object Memory test hit it -- Tower resident, `/health` refused everywhere,
nothing listening on 8000 -- and `tower/serve_loop.py` is the fix.

The reproduction is the one that was measured: fire a burst of connections
that RST on close (SO_LINGER 0) while the event loop is deliberately stalled,
so the connections sit in the backlog and are reset while queued. A stock
Proactor loop loses its listener; the resilient one keeps it.

Windows-only, because the bug, the loop and the `ResilientProactorEventLoop`
class are all Windows-only. Off win32 there is no Proactor loop and nothing
to guard.
"""

import asyncio
import socket
import sys
import threading
import time

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "win32",
    reason="gh-93821 and ResilientProactorEventLoop are Windows-only",
)


def _rst_burst(port: int, count: int) -> None:
    """Connect `count` times and reset each connection immediately.

    SO_LINGER on with a zero timeout makes `close()` send a RST rather than
    a FIN, which is what puts a connection into the state that trips the
    accept-loop bug when it is reset while queued in the backlog.
    """
    linger = b"\x01\x00\x00\x00"  # l_onoff=1, l_linger=0 -> RST on close
    for _ in range(count):
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, linger)
        try:
            client.connect(("127.0.0.1", port))
        except OSError:
            client.close()
            continue
        client.close()


def _probe(port: int) -> bool:
    """Whether a fresh connection is still accepted -- the real question."""
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.settimeout(2.0)
    try:
        client.connect(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        client.close()


def _listener_survives(make_loop, *, rounds: int = 4, burst: int = 200) -> bool:
    """Run the stall+RST attack on a loop and report whether the port lives.

    Returns True only if, after every round, the listening socket is still
    open AND a fresh connection is still accepted.
    """
    loop = make_loop()

    async def scenario() -> bool:
        async def handler(reader, writer):
            try:
                await reader.read(16)
            except Exception:
                pass
            finally:
                writer.close()

        server = await asyncio.start_server(
            handler, "127.0.0.1", 0, backlog=100
        )
        listening = server.sockets[0]
        port = listening.getsockname()[1]
        try:
            for _ in range(rounds):
                attacker = threading.Thread(
                    target=_rst_burst, args=(port, burst), daemon=True
                )
                attacker.start()
                # Stall the single-threaded loop so the burst piles up in the
                # backlog and is reset while queued -- the enabling condition.
                time.sleep(0.4)
                attacker.join(timeout=10)
                # Let the accept loop work through whatever is queued.
                await asyncio.sleep(0.4)
                if listening.fileno() == -1:
                    return False
                if not await asyncio.to_thread(_probe, port):
                    return False
            return True
        finally:
            server.close()
            try:
                await asyncio.wait_for(server.wait_closed(), 5)
            except (asyncio.TimeoutError, Exception):
                pass

    try:
        return loop.run_until_complete(scenario())
    finally:
        loop.close()


def test_the_resilient_loop_keeps_its_listener_through_a_reset_burst(caplog):
    """The guarantee: the fixed loop stays up and keeps accepting.

    Asserts BOTH that the listener survived AND that the per-connection
    re-arm branch actually fired (it logs a warning the first time). Without
    the second check the test could pass on a run where the attack never
    landed -- the same race that makes the stock-loop test a non-strict
    xfail -- and would then not catch a regression that broke the re-arm.
    """
    import logging

    from tower.serve_loop import ResilientProactorEventLoop

    with caplog.at_level(logging.WARNING, logger="tower.serve_loop"):
        survived = _listener_survives(ResilientProactorEventLoop)

    assert survived is True, (
        "the resilient listener died under a reset-in-backlog burst -- the "
        "re-arm in tower/serve_loop.py has regressed"
    )
    rearmed = [
        r for r in caplog.records if "reset before it was accepted" in r.message
    ]
    assert rearmed, (
        "the listener survived but the per-connection re-arm branch never "
        "fired -- the attack did not land, so this run proves nothing. If "
        "this is persistent, the burst/stall in _listener_survives needs to "
        "be stronger."
    )


@pytest.mark.xfail(
    reason="CPython gh-93821: the stock Proactor loop closes its listener on "
    "a per-connection accept error. Documented here; not strict because the "
    "attack is a race and the stock loop occasionally survives a run.",
    strict=False,
)
def test_the_stock_proactor_loop_loses_its_listener():
    """The bug itself, pinned as documentation.

    This is why `serve_loop.py` exists. Expected to FAIL -- the stock loop
    loses its listener -- so a run in which it passes means the attack did
    not land this time, not that CPython was fixed.
    """
    assert _listener_survives(asyncio.ProactorEventLoop) is True
