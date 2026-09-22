"""A dropped socket must leave the result-channel handler, not be eaten by it.

`results_ws.handle` swallows result-channel FAULTS on purpose -- a broken
subscription must not end a connection that is answering frames. But a
`WebSocketDisconnect` is not a fault of the result channel; it is the socket
being gone. Swallowed here, the receive loop only discovers it on the next
`receive_json`, which raises a bare `RuntimeError: WebSocket is not
connected` that `ws.py` does not catch and uvicorn logs as "Exception in
ASGI application". So the disconnect must propagate, and a real fault must
still be swallowed. Both halves are pinned here.
"""

import asyncio

import pytest
from fastapi import WebSocketDisconnect

from tower.routes import results_ws


def test_a_disconnect_raised_by_a_handler_propagates(monkeypatch):
    async def _dropped(*args, **kwargs):
        raise WebSocketDisconnect(1006)

    monkeypatch.setattr(results_ws, "_cartridges", _dropped)

    with pytest.raises(WebSocketDisconnect):
        asyncio.run(
            results_ws.handle(
                {"type": results_ws.MSG_CARTRIDGES},
                websocket=None,
                sender=None,
                channel_holder=None,
            )
        )


def test_a_result_channel_fault_is_still_swallowed(monkeypatch):
    async def _broken(*args, **kwargs):
        raise RuntimeError("a subscription went wrong")

    monkeypatch.setattr(results_ws, "_cartridges", _broken)

    # Must NOT raise: a fault becomes a logged line, and the connection --
    # which is answering frames -- continues.
    asyncio.run(
        results_ws.handle(
            {"type": results_ws.MSG_CARTRIDGES},
            websocket=None,
            sender=None,
            channel_holder=None,
        )
    )
