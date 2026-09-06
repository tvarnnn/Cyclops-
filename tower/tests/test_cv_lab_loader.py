"""One loader thread per Lab, reused across arms, retired when abandoned.

Why this file exists: a `load()` that runs a torch model leaves torch's
intra-op team behind on the thread that ran it. Measured on this host on
2026-09-06, a fresh thread per arm grew the Tower by exactly 19 OS
threads and ~8 MB per switch to `depth` or `object_detection`, linearly,
with no ceiling; one persistent loader thread grew it by zero over the
same twelve arms. The team is not a leak when it belongs to a thread
that is still working; it is a leak when its thread has exited.

The properties an adversarial reviewer would check, in order:

* two consecutive loads run on the SAME OS thread;
* a load that is abandoned (its awaiter cancelled) still completes on its
  own thread and its result is discarded, not delivered;
* the abandoned worker is RETIRED -- the next load runs on a fresh thread
  rather than queueing behind the stuck one;
* a retired worker exits once it is free, so the count of loader threads
  returns to one;
* an exception inside a load reaches the awaiter as that exception.
"""

import asyncio
import threading
import time

import pytest

from tower.cv_lab.loader import LOADER_THREAD_NAME, ExperimentLoader


def _loader_threads(before: set[int] | None = None) -> list[threading.Thread]:
    """Loader threads, minus any that existed before the test began.

    Other test modules build Labs they never release, and a Lab's loader
    exits only when the Lab is collected. Counting those here would make
    this file's assertions depend on test order.
    """
    return [
        t
        for t in threading.enumerate()
        if t.name.startswith(LOADER_THREAD_NAME)
        and (before is None or t.ident not in before)
    ]


@pytest.fixture
def before() -> set[int]:
    return {t.ident for t in _loader_threads()}


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_two_loads_run_on_the_same_thread():
    loader = ExperimentLoader()
    try:

        async def scenario():
            first = await loader.run(threading.get_ident)
            second = await loader.run(threading.get_ident)
            return first, second

        first, second = asyncio.run(scenario())
    finally:
        loader.close()

    assert first == second
    assert first != threading.get_ident()


def test_the_loader_thread_is_named_and_there_is_one_of_it(before):
    loader = ExperimentLoader()
    try:
        asyncio.run(loader.run(lambda: None))
        live = _loader_threads(before)
        assert len(live) == 1
        assert live[0].daemon is True
    finally:
        loader.close()
    assert _wait_until(lambda: not _loader_threads(before))


def test_arguments_and_results_are_relayed():
    loader = ExperimentLoader()
    try:
        assert asyncio.run(loader.run(lambda a, b: a + b, 2, 3)) == 5
    finally:
        loader.close()


def test_an_exception_in_the_load_reaches_the_awaiter():
    loader = ExperimentLoader()

    def _boom():
        raise RuntimeError("weights unavailable")

    try:
        with pytest.raises(RuntimeError, match="weights unavailable"):
            asyncio.run(loader.run(_boom))
        # And the worker survived it: the next load still runs.
        assert asyncio.run(loader.run(lambda: "alive")) == "alive"
    finally:
        loader.close()


def test_an_abandoned_load_finishes_on_its_own_and_is_discarded(before):
    """The `run_abandonable` promise, kept by a reusable worker."""
    loader = ExperimentLoader()
    started = threading.Event()
    proceed = threading.Event()
    finished: list[int] = []

    def _stuck():
        started.set()
        proceed.wait(5)
        finished.append(threading.get_ident())
        return "late"

    try:

        async def scenario():
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(loader.run(_stuck), timeout=0.05)
            assert started.is_set()
            # The Lab moves on: a second load must not wait for the first.
            fresh = await asyncio.wait_for(
                loader.run(threading.get_ident), timeout=2.0
            )
            proceed.set()
            return fresh

        fresh_thread = asyncio.run(scenario())
    finally:
        loader.close()

    assert _wait_until(lambda: bool(finished)), "the abandoned load never ran to completion"
    assert finished[0] != fresh_thread, "the second load queued behind the abandoned one"
    # The retired worker exits once it is free; only the closed current
    # worker's exit is also pending, so nothing named ours remains.
    assert _wait_until(lambda: not _loader_threads(before))


def test_a_cancelled_awaiter_retires_the_worker_only_while_it_is_busy(before):
    """A cancellation that lands AFTER the load completed must not retire
    an idle worker for nothing -- otherwise every stop-after-arm would
    cost a thread."""
    loader = ExperimentLoader()
    try:

        async def scenario():
            await loader.run(lambda: None)
            first = {t.ident for t in _loader_threads(before)}
            task = asyncio.ensure_future(loader.run(lambda: None))
            await task
            task.cancel()  # a no-op on a finished task, as in the Lab
            await loader.run(lambda: None)
            second = {t.ident for t in _loader_threads(before)}
            return first, second

        first, second = asyncio.run(scenario())
    finally:
        loader.close()

    assert first == second


def test_close_is_idempotent_and_run_after_close_starts_a_fresh_worker(before):
    loader = ExperimentLoader()
    asyncio.run(loader.run(lambda: None))
    loader.close()
    loader.close()
    assert _wait_until(lambda: not _loader_threads(before))
    # A Lab that is unloaded and loaded again (the container has no such
    # path today, but the loader should not be the reason it cannot).
    try:
        assert asyncio.run(loader.run(lambda: "again")) == "again"
    finally:
        loader.close()
