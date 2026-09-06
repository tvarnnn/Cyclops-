"""One loader thread per Lab, reused across arms, retired when abandoned.

`tower/loading.py`'s `run_abandonable` starts a fresh daemon thread per
call and never joins it, which is the right shape for the one place it
was written for -- a module's single load at boot, bounded by a timeout
that can only abandon. The CV Lab arms an experiment every time a person
taps a row, and that changes the arithmetic:

A thread that runs a torch model load creates torch's intra-op OpenMP
team on itself, and the team does not leave when the thread does.
Measured on this host on 2026-09-06, six consecutive arms of
`object_detection` on fresh threads took the process from 48 to 143 OS
threads (+19 per arm, ~8 MB RSS each), and six arms of `depth` from 163 to
258, with no ceiling in sight; the same twelve arms on ONE persistent
thread moved the count by zero. The team is the price of using torch on a
thread, paid once per thread. So the Lab pays it once.

What this keeps from `run_abandonable`, because it is still true:

* the awaiter is bounded by `asyncio.wait_for` and a timeout ABANDONS the
  load -- nothing in Python can stop a thread mid-`torch.hub.load`;
* an abandoned load runs to completion on its own thread and DISCARDS
  what it built, through the experiment's `LoadInvalidation` latch, which
  this module does not touch and does not need to;
* nothing here is ever joined on the event loop.

What it adds: when a load is abandoned, its worker is **retired** -- the
next `run()` gets a fresh worker at once rather than queueing behind a
thread that may be minutes from finishing a download -- and the retired
worker exits by itself once it is free, so the loader-thread count is
one again. A worker is retired only if it is actually holding the
abandoned job; a cancellation that lands after the load returned costs
nothing.
"""

import asyncio
import contextvars
import logging
import queue
import threading
import weakref
from typing import Any, Callable

logger = logging.getLogger(__name__)

LOADER_THREAD_NAME = "tower-cv-lab-loader"


class _Job:
    __slots__ = ("fn", "args", "loop", "future", "context", "done", "abandoned")

    def __init__(self, fn, args, loop, future) -> None:
        self.fn = fn
        self.args = args
        self.loop = loop
        self.future = future
        self.context = contextvars.copy_context()
        # Set by the worker once `fn` has returned or raised, before the
        # result is delivered. Read by the awaiter's cancellation path to
        # decide whether the worker is still busy with us.
        self.done = False
        # Set by an awaiter that gave up while the job was still QUEUED.
        # The worker skips it: there is nobody to deliver to and nothing
        # was built yet, so running it would only build something to
        # throw away.
        self.abandoned = False

    def execute(self) -> None:
        try:
            result = self.context.run(self.fn, *self.args)
        except BaseException as exc:  # relayed verbatim to the awaiter
            payload = (self.future.set_exception, exc)
        else:
            payload = (self.future.set_result, result)
        self.done = True
        try:
            self.loop.call_soon_threadsafe(_deliver, self.future, *payload)
        except RuntimeError:
            # The loop that was waiting has closed. That is the abandoned
            # case working as designed: there is nobody left to tell, and
            # the load's own invalidation token has already made sure it
            # built nothing that survives.
            pass


def _deliver(future: asyncio.Future, setter: Callable[[Any], None], value: Any) -> None:
    # The awaiter may have been cancelled by its timeout while the job was
    # still running; setting either half of a done future raises
    # InvalidStateError, and setting an exception nobody will retrieve
    # logs a spurious "exception was never retrieved".
    if not future.done():
        setter(value)


class _Worker:
    def __init__(self, serial: int) -> None:
        self.queue: "queue.SimpleQueue[_Job | None]" = queue.SimpleQueue()
        self.retired = False
        self.thread = threading.Thread(
            target=self._loop, name=f"{LOADER_THREAD_NAME}-{serial}", daemon=True
        )

    def _loop(self) -> None:
        while True:
            job = self.queue.get()
            if job is None:
                return
            if job.abandoned:
                continue
            job.execute()


class ExperimentLoader:
    """Runs blocking experiment loads on one reusable daemon thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._worker: _Worker | None = None
        self._serial = 0

    def _current_worker_locked(self) -> _Worker:
        worker = self._worker
        if worker is None or worker.retired:
            self._serial += 1
            worker = _Worker(self._serial)
            worker.thread.start()
            self._worker = worker
            # A Lab that is dropped without `release()` -- a test that
            # built one and moved on -- must not leave a thread parked on
            # an empty queue forever. The finalizer holds the WORKER, not
            # `self`, so it cannot keep the loader alive.
            weakref.finalize(self, _retire, worker)
        return worker

    async def run(self, fn: Callable[..., Any], *args: Any) -> Any:
        """Run `fn(*args)` on the loader thread and await its result.

        Cancelling the awaiter (directly, or through `asyncio.wait_for`)
        abandons the job: if it is still queued it is skipped; if it is
        running, its worker is retired and a fresh one serves the next
        call. Either way the cancellation propagates.
        """
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        job = _Job(fn, args, loop, future)
        with self._lock:
            worker = self._current_worker_locked()
            worker.queue.put(job)
        try:
            return await future
        except asyncio.CancelledError:
            with self._lock:
                if not job.done:
                    job.abandoned = True
                    if worker is self._worker and not worker.retired:
                        worker.retired = True
                        # The sentinel queues BEHIND whatever the worker is
                        # doing, so it exits when it is free and not before.
                        worker.queue.put(None)
                        logger.info(
                            "[Tower][CVLab] a load was abandoned mid-flight; "
                            "its loader thread is retired and will exit when "
                            "the load returns"
                        )
            raise

    def close(self) -> None:
        """Let the current worker exit once it is idle. Idempotent.

        Not joined: the thread is a daemon and may be inside an abandoned
        load. A later `run()` starts a fresh worker, so a Lab that is
        released and loaded again is not stuck with a closed loader.
        """
        with self._lock:
            worker, self._worker = self._worker, None
            _retire(worker)


def _retire(worker: "_Worker | None") -> None:
    """Queue the exit sentinel behind whatever the worker is doing."""
    if worker is not None and not worker.retired:
        worker.retired = True
        worker.queue.put(None)
