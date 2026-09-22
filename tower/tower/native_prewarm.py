"""Load the thread-spawning native stack BEFORE a watcher thread parks in a read.

WHY THIS MODULE EXISTS, MEASURED TWICE ON THIS HOST.

A worker that the Tower spawns with `stdin=PIPE` and `--stop-on-stdin-close`
arms a daemon thread that blocks in a synchronous `ReadFile` on that pipe:
nothing is ever written to it, the request IS the close. That is the only
stop channel that works for a process with no console, and it is correct.

It is also one half of a Windows loader-lock deadlock. The other half is any
DLL that spawns threads in its `DllMain` -- OpenBLAS, shipped inside
`scipy.linalg`, is the one that keeps doing it here. When the main thread
calls `LoadLibraryExW` on such a DLL it holds the loader lock while the DLL
creates its thread pool; a thread already parked in a blocking pipe read is
enough to make that never complete. The result is a hard hang at 0% CPU,
forever, with no traceback and no log line.

IT HAS NOW HAPPENED TO TWO SUBSYSTEMS.

  * 2026-09-06, Object Memory. A healthy 244-frame walk was observed with
    `frames_observed: 0`: the producer deadlocked loading the OWLv2
    verifier, which imports `scipy.linalg`. Fixed in
    `scripts/object_memory_session.py` by warming `scipy.linalg` on the
    main thread before the watcher was armed.

  * 2026-09-22, World Builder. `scripts/world_finish_pending.py` -- the
    recovery finisher -- held world `2f44716237544569b5f2faf782d9f877` for
    over 95 minutes against an ~8 minute job, at 0% CPU, GPU idle, writing
    nothing, while the phone read "Finishing" the whole time. py-spy
    `--native` on the live process showed the identical two threads:

        MainThread          LdrLoadDll / LoadLibraryExW
                            libscipy_openblas-*.dll
                            <module> (scipy/linalg/blas.py:247)
                            ... moge/model/v2.py -> dense.py:509 _load
                            surfacify -> final_surface_stages -> finish

        world-builder-stop-watch
                            NtReadFile / ReadFile
                            wait_for_close (world_build_session.py:529)

The first fix was written where the first bug was found, so the second
subsystem inherited the bug rather than the fix. This module is the fix
extracted to one place, so the third subsystem inherits neither.

WHAT IT DOES NOT DO. It does not change the stop channel, and it does not
delay arming it by any meaningful amount: once the pipe's write end is
closed the close is latched, so a stop asked for during the warm is still
honoured at the first read. And it never refuses to start -- warming a
library is an optimisation of ORDER, not a dependency. Anything that fails
here is reported and stepped over, because a host that cannot import torch
cannot run the stage that would have needed it either, and it must still be
allowed to say so in its own words rather than through this.

**Cartridge-blind**, like `process_ownership.py`: it knows about DLLs and
threads and names no cartridge. The module TUPLES below are the only
subsystem-shaped thing in it, and they are data.
"""

from __future__ import annotations

import importlib
import sys

# The libraries whose load must not race a parked pipe reader.
#
# `scipy.linalg` is the proven offender in both incidents: it is what
# carries `libscipy_openblas*.dll`, and OpenBLAS builds its thread pool in
# `DllMain`. The others are here because they ship their own thread-spawning
# runtimes (torch's OpenMP, OpenCV's parallel backend) and are loaded by the
# same stages a few frames later -- warming them costs the seconds their
# import would have cost anyway, and removes them from the hazard window.
#
# ORDER MATTERS ONLY IN ONE DIRECTION: numpy before scipy, because scipy
# imports it regardless and doing it explicitly keeps the failure message
# about the library that actually could not be found.
OBJECT_MEMORY_MODULES = ("numpy", "scipy.linalg")
WORLD_BUILDER_MODULES = ("numpy", "scipy.linalg", "cv2", "torch")


def prewarm(modules, *, subsystem: str, stream=None) -> tuple[str, ...]:
    """Import `modules` on the calling thread. Returns the names that failed.

    MUST be called on the MAIN thread, and BEFORE any stop watcher is
    armed. Both halves are the point: the deadlock needs a loader load on
    one thread and a blocking kernel read on another, and this removes the
    overlap by doing the loads while no such reader exists yet.

    Every failure is a warning on `stream` (stderr by default) and never an
    exception. A missing library is a real problem for the stage that needs
    it, and that stage will say so with the context this function does not
    have.
    """
    if stream is None:
        stream = sys.stderr
    failed: list[str] = []
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 -- warming is never fatal
            failed.append(name)
            print(
                f"[Tower][{subsystem}] could not pre-warm '{name}' "
                f"({exc.__class__.__name__}: {exc}); continuing. On Windows, "
                "watch for a worker that loads a model and then makes no "
                "progress at all -- see tower/native_prewarm.py.",
                file=stream,
                flush=True,
            )
    return tuple(failed)


def _warm_cuda(stream) -> bool:
    """Ask torch for a CUDA device, so the NVIDIA driver attaches HERE.

    FOUND BY AN ADVERSARIAL REVIEW OF THE FIRST VERSION OF THIS MODULE, which
    warmed the import list above and stopped there. `import torch` does NOT
    load the driver: the first CUDA *query* does, and on this host that query
    maps seventeen further DLLs -- `nvcuda64.dll`, `nvapi64.dll`,
    `nvobjectloader64.dll`, `nvcudart_hybrid64.dll` and friends. `nvcuda64`
    starts worker and watchdog threads when it attaches, which is exactly the
    `DllMain` behaviour this whole module exists to keep away from a parked
    pipe reader.

    And the query that does it is `tower/world_builder/dense.py`:

        from moge.model.v2 import MoGeModel        # line 507
        ...
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    -- one line after the import the py-spy stack blamed, in the same frame,
    under the same loader lock, with the same watcher in `NtReadFile`. So the
    first version removed one thread-spawning DLL from that frame and left a
    second one in it.

    Measured on this host: **0.02 s**, against 1.4 s for the imports above.
    There is no argument for leaving it out.

    Returns False and says so if torch is absent or the query raises; a host
    with no CUDA still runs, on the CPU, and the stage says so itself.
    """
    try:
        import torch  # noqa: PLC0415 -- already resident after `prewarm`

        torch.cuda.is_available()
        return True
    except Exception as exc:  # noqa: BLE001 -- warming is never fatal
        print(
            f"[Tower][WorldBuilder] could not pre-warm the CUDA driver "
            f"({exc.__class__.__name__}: {exc}); continuing. The stage will "
            "resolve its own device and say which it got.",
            file=stream if stream is not None else sys.stderr,
            flush=True,
        )
        return False


def prewarm_world_builder(stream=None) -> tuple[str, ...]:
    """The World Builder warm: the stages that draw the photographic room.

    Called by `scripts/world_build_session.py` and
    `scripts/world_finish_pending.py`, which run the same
    `final_surface_stages` path and therefore load the same DLLs.
    """
    failed = prewarm(WORLD_BUILDER_MODULES, subsystem="WorldBuilder", stream=stream)
    # AFTER the imports and still BEFORE any watcher: the driver attaches on
    # the first CUDA query, not on `import torch`. See `_warm_cuda`.
    _warm_cuda(stream)
    return failed


def prewarm_object_memory(stream=None) -> tuple[str, ...]:
    """The Object Memory warm: the verifier's numerical stack."""
    return prewarm(OBJECT_MEMORY_MODULES, subsystem="ObjectMemory", stream=stream)
