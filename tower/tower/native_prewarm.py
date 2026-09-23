"""Load the heavy native stack before a worker's stop watcher is armed.

NO LONGER THE FIX, AND WHY IT IS STILL HERE.

This module was written on 2026-09-22 as the fix for the recovery finisher
that held world 2f44716237544569b5f2faf782d9f877 for 95 minutes at 0% CPU,
on the theory that a DLL spawning threads in `DllMain` cannot finish loading
under the Windows loader lock while another thread is parked in a pipe read.
The theory was wrong about the mechanism, and so the fix was narrower than
it looked. Reproduced on demand on 2026-09-23 (`tower/stdin_stop.py` has both
native stacks): the stop watcher's `os.read(0, 1)` held descriptor 0 -- the
UCRT's per-descriptor lock, and the pipe's synchronous file object -- and
ANY library whose load-time code touched descriptor 0 waited for it.
numpy's and scipy's OpenBLAS do (their libgfortran's `init_units`); so does
pycolmap's bundled libgfortran, which hung even with this whole list loaded
first. It was never cold-versus-warm DLLs, and never a race: with a live pipe
behind a parked read it hangs every time.

The real fix is `tower/stdin_stop.py`: the watcher no longer leaves a read
pending, so there is nothing for a library to wait behind, whatever loads
and whenever. This warm is kept for two modest reasons -- it front-loads
import cost to a moment when nothing is held, and it is belt and braces for
the libraries it names should a watcher ever park a read again -- and it is
covered by the ordering tests that already existed. It must not be mistaken
for the thing that makes the workers safe.

IT HAS HAPPENED TO TWO SUBSYSTEMS.

  * 2026-09-06, Object Memory. A healthy 244-frame walk was observed with
    `frames_observed: 0`: the producer hung loading the OWLv2 verifier,
    which imports `scipy.linalg`.

  * 2026-09-22, World Builder. `scripts/world_finish_pending.py` held world
    2f447162 for over 95 minutes. py-spy `--native` on the live process:

        MainThread          LoadLibraryExW -> libscipy_openblas-*.dll
                            -> ucrtbase -> RtlEnterCriticalSection
                            <module> (scipy/linalg/blas.py:247)
                            ... moge/model/v2.py -> dense.py:509 _load
                            surfacify -> final_surface_stages -> finish

        world-builder-stop-watch
                            ucrtbase _read -> ReadFile / NtReadFile
                            wait_for_close (world_build_session.py:529)

WHAT IT DOES NOT DO. It does not change the stop channel, and it never
refuses to start -- warming a library is an optimisation of ORDER, not a
dependency. Anything that fails here is reported and stepped over, because a
host that cannot import torch cannot run the stage that would have needed it
either, and it must still be allowed to say so in its own words rather than
through this.

**Cartridge-blind**, like `process_ownership.py`: it knows about DLLs and
threads and names no cartridge. The module TUPLES below are the only
subsystem-shaped thing in it, and they are data.
"""

from __future__ import annotations

import importlib
import sys

# The libraries the stages load first, warmed while nothing is held.
#
# `scipy.linalg` is the library on both incidents' stacks: it carries
# `libscipy_openblas*.dll`, whose libgfortran touches descriptor 0 at load.
# The others are loaded by the same stages a few frames later, and warming
# them costs the seconds their import would have cost anyway.
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
    `nvobjectloader64.dll`, `nvcudart_hybrid64.dll` and friends -- loading
    that no stage should be doing for the first time while anything is held.

    And the query that does it is `tower/world_builder/dense.py`:

        from moge.model.v2 import MoGeModel        # line 507
        ...
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    -- one line after the import the py-spy stack blamed, in the same frame.

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
