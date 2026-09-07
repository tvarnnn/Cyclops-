"""A Windows event loop whose listener survives a per-connection accept error.

WHY THIS FILE EXISTS: THE LISTENER THAT VANISHED WHILE THE PROCESS LIVED.

An Object Memory physical test on Windows ended with a Tower that was still
a running process -- ~1.7 GB resident, answering nothing -- but with no
socket on port 8000 at all: `Get-NetTCPConnection -LocalPort 8000` empty,
`/health` refused on localhost AND over Tailscale, while the Python process
tree stayed alive. The console carried, just before it went deaf:

    OSError: [WinError 64] The specified network name is no longer available
    ConnectionResetError: [WinError 10054] ...
    ERROR asyncio Accept failed on a socket

That is not "the phone disconnected". It is CPython bug gh-93821, unfixed
in 3.12 and on `main`. uvicorn runs on `asyncio.ProactorEventLoop` on
Windows, and that loop's accept path (`asyncio/proactor_events.py`,
`_start_serving.loop`) does this: when the `AcceptEx` completion raises an
`OSError` -- which happens when a connection that was sitting in the kernel
backlog is reset by its peer BEFORE the loop accepted it -- it logs "Accept
failed on a socket", **closes the listening socket, and never re-arms
accept.** The loop keeps running, every existing connection keeps working,
and the port is simply gone. A restart is the only recovery, which is
exactly what was observed.

The enabling condition is a busy loop: while the event loop is occupied
(decoding a 7 MB frame, an fsync, a `reap()` under the supervisor lock,
DEBUG console logging), only one `AcceptEx` is armed, so connections pile
up in the backlog, and a wearable client that reconnects -- iOS retries on
a backoff after a capture's `disconnect`, Tailscale re-paths, a URLSession
attempt is cancelled -- resets one of them and trips the bug. A shared,
long-lived Tower that a phone connects to over a flaky link is precisely
the workload that hits it.

WHAT THIS CHANGES, AND WHAT IT DELIBERATELY DOES NOT.

`ResilientProactorEventLoop` overrides exactly one private method,
`_start_serving`, and its accept closure differs from CPython's in exactly
one place: an `OSError` raised by the per-connection `f.result()` whose
Winsock code names the CONNECTION rather than the listener is logged and
SKIPPED, and accept is re-armed, instead of closing the listener. Every
other error keeps CPython's behaviour to the letter -- a failure to arm a
NEW accept, a genuinely broken listening socket, cancellation at shutdown
-- because those are not this bug and must still tear the listener down.

It is a copy of a private method, so it is pinned to a CPython layout
(verified against 3.12.5, which is this host). That is an accepted cost:
the upstream fix (PR #124779) is still open, `--loop asyncio` does NOT help
(it still yields a Proactor loop on win32), and the alternative -- a
`SelectorEventLoop`, which does not have the bug -- forfeits IOCP for the
whole Tower to fix an accept-path defect. When CPython ships the fix this
file can be deleted and the `--loop` flag dropped.

This is half of the long-lived-Tower story. The other half is bounding
shutdown, which is a uvicorn flag (`--timeout-graceful-shutdown`) rather
than code and lives in `scripts/start_tower.ps1`.
"""

from __future__ import annotations

import asyncio
import logging
import sys

logger = logging.getLogger("tower.serve_loop")

# Winsock / Win32 error codes that describe the ACCEPTED connection, not the
# listening socket. A completion carrying one of these means "the connection
# that was waiting in the backlog is gone", which is a normal event on a
# flaky wireless link and must never take the listener down with it.
#
#   64    ERROR_NETNAME_DELETED     -- the network name is no longer available
#   1236  ERROR_CONNECTION_ABORTED  -- the local system aborted the connection
#   10053 WSAECONNABORTED           -- software caused connection abort
#   10054 WSAECONNRESET             -- connection reset by peer
#   10057 WSAENOTCONN               -- socket is not connected
#   10058 WSAESHUTDOWN              -- socket has been shut down
_PER_CONNECTION_WINERRORS = frozenset({64, 1236, 10053, 10054, 10057, 10058})


if sys.platform == "win32":
    from asyncio import exceptions as _asyncio_exceptions
    from asyncio import trsock as _asyncio_trsock

    class ResilientProactorEventLoop(asyncio.ProactorEventLoop):
        """A ProactorEventLoop that re-arms accept after a per-connection error.

        The body of `loop` below is copied from CPython 3.12's
        `BaseProactorEventLoop._start_serving` and differs in one place,
        marked THE FIX. Keeping the rest byte-for-byte is deliberate: the
        goal is to change nothing except whether one specific, transient
        Winsock error closes the listener.
        """

        def _start_serving(
            self,
            protocol_factory,
            sock,
            sslcontext=None,
            server=None,
            backlog=100,
            ssl_handshake_timeout=None,
            ssl_shutdown_timeout=None,
        ):
            def loop(f=None):
                try:
                    if f is not None:
                        # THE FIX. CPython lets an OSError from this
                        # `f.result()` fall through to the outer handler,
                        # which closes the LISTENING socket. When the error
                        # is about the accepted connection -- reset in the
                        # backlog before we took it -- skip that one
                        # connection and re-arm accept instead. The listener
                        # is fine; only a client went away.
                        try:
                            conn, addr = f.result()
                        except OSError as exc:
                            if (
                                getattr(exc, "winerror", None)
                                in _PER_CONNECTION_WINERRORS
                                and sock.fileno() != -1
                            ):
                                logger.warning(
                                    "[Tower][Serve] a queued connection was "
                                    "reset before it was accepted (%s); the "
                                    "listener stays up and keeps accepting",
                                    exc,
                                )
                                conn = None
                            else:
                                # Not this bug -- a broken listener, or an
                                # error that is not a per-connection reset.
                                # Fall through to CPython's handling.
                                raise
                        if conn is not None:
                            if self._debug:
                                logger.debug(
                                    "%r got a new connection from %r: %r",
                                    server,
                                    addr,
                                    conn,
                                )
                            protocol = protocol_factory()
                            if sslcontext is not None:
                                self._make_ssl_transport(
                                    conn,
                                    protocol,
                                    sslcontext,
                                    server_side=True,
                                    extra={"peername": addr},
                                    server=server,
                                    ssl_handshake_timeout=ssl_handshake_timeout,
                                    ssl_shutdown_timeout=ssl_shutdown_timeout,
                                )
                            else:
                                self._make_socket_transport(
                                    conn,
                                    protocol,
                                    extra={"peername": addr},
                                    server=server,
                                )
                    if self.is_closed():
                        return
                    f = self._proactor.accept(sock)
                except OSError as exc:
                    if sock.fileno() != -1:
                        self.call_exception_handler(
                            {
                                "message": "Accept failed on a socket",
                                "exception": exc,
                                "socket": _asyncio_trsock.TransportSocket(sock),
                            }
                        )
                        sock.close()
                    elif self._debug:
                        logger.debug(
                            "Accept failed on socket %r", sock, exc_info=True
                        )
                except _asyncio_exceptions.CancelledError:
                    sock.close()
                else:
                    self._accept_futures[sock.fileno()] = f
                    f.add_done_callback(loop)

            self.call_soon(loop)


def resilient_loop_factory(
    use_subprocess: bool = False,
) -> asyncio.AbstractEventLoop:
    """A NEW event loop for uvicorn, resilient on Windows.

    THE CONTRACT, AND THE TRAP IN IT. uvicorn's `--loop module:callable`
    imports this name and hands it to `asyncio.run(..., loop_factory=...)`
    WITHOUT calling it. `asyncio.Runner` then calls it with NO arguments and
    expects a fresh LOOP INSTANCE back. So this returns an instance, not a
    class -- returning the class (which is what the built-in
    `uvicorn.loops.asyncio.asyncio_loop_factory` does, because uvicorn calls
    THAT one itself first) would hand asyncio a class where it wants a loop
    and fail at startup. The `use_subprocess` parameter exists only to match
    the built-in's signature; the custom-loop path never passes it.

    Off Windows there is no bug and no Proactor loop, so a plain
    `SelectorEventLoop` is returned and this file is inert. On Windows, when
    uvicorn is driving a worker subprocess, it also wants a selector loop
    (the Proactor loop cannot watch a subprocess's pipes the same way);
    `use_subprocess` carries that, exactly as the built-in factory reads it.
    """
    if sys.platform == "win32" and not use_subprocess:
        return ResilientProactorEventLoop()
    return asyncio.SelectorEventLoop()
