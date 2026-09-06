"""The CV Lab itself: one slot, many experiments, one truthful document.

What this replaces
------------------
Choosing an experiment used to mean editing `TOWER_CV_EXPERIMENT`,
restarting the Tower, starting a generic recording somewhere else, and
reading an anonymous number off a debug panel. Every part of that is a
consequence of one fact: the experiment was decided at process start and
nothing on the wire could say which one it was. This module moves the
decision to runtime and puts a name on every number that leaves.

What it is NOT
--------------
It is not a module registry and it does not swap modules.
`ModuleContainer` still holds exactly one `Module`, constructed once, with
no discovery and no swap path -- `04-MODULE-SYSTEM.md` forbids dynamic
discovery before V1.0 and nothing here works around that. What changed is
what is *inside* the one Lab slot, which is what the module's own
docstring has said all along: "one Lab slot, many experiments".

Three clocks' worth of care, in one place
-----------------------------------------
1. **The frame path is synchronous and on the event loop.** `process()`
   never awaits, so no command can interleave *within* a frame. That is
   what makes `frame_provenance()` -- read by `ws.py` immediately after
   `process()`, with no await between -- describe the frame it just
   answered rather than some later one.

2. **A load is not.** A model-backed experiment's `load()` downloads
   weights and blocks; `depth` fetches 119 MB the first time. Awaiting
   that inside a connection's receive loop would stop that socket reading
   for minutes, and `handoff.md` 13.7 says iOS replaces a connection it
   cannot write to for two seconds. So `start()` validates, publishes
   `starting`, and hands the load to a background task. The command
   returns immediately; the outcome arrives as state, which is exactly
   the shape iOS's `run(_:)` already has.

3. **The status document is read from another thread.** The result
   channel's poller computes snapshots with `asyncio.to_thread`, so
   `status()` runs concurrently with the loop that mutates this object.
   Every state transition and every status build therefore happens under
   `_guard`. The FRAME PATH deliberately does not take it: it mutates
   only per-run counters, whose individual reads are atomic, and putting
   a lock on the measured path to make a diagnostic one frame fresher
   would be paying in the wrong currency.

Staleness is structural, not checked
------------------------------------
A run is the unit of provenance and a new experiment is a new run. The
old experiment is released BEFORE the new run id is published, so there
is no window in which a result produced by one experiment can carry
another's name. `run_id` travels on every `frame_result` and on the
status document, and a client that sees a result whose `run_id` is not
the one it is watching should discard it -- not because the Tower is
expected to send one, but because that is the only way a reconnect across
a restart cannot show a previous Tower's numbers.
"""

import asyncio
import functools
import importlib.util
import math
import logging
import os
import threading
import time
import uuid

import psutil

from tower.cv_lab import catalog
from tower.logging_config import client_safe_reason
from tower.cv_lab.contracts import (
    CONTROL_CONTRACT,
    ERR_EXPERIMENT_UNAVAILABLE,
    ERR_INVALID_STATE,
    ERR_LAB_BUSY,
    ERR_LAB_UNAVAILABLE,
    ERR_MALFORMED,
    ERR_STALE_RUN,
    ERR_UNKNOWN_EXPERIMENT,
    FRAME_REFUSAL_REASONS,
    FRAME_REFUSED_FAILED,
    FRAME_RESULT_CONTRACT,
    ORIGIN_CLIENT_REQUEST,
    ORIGIN_STARTUP_DEFAULT,
    STATE_FAILED,
    STATE_IDLE,
    STATE_PAUSED,
    STATE_RUNNING,
    STATE_STARTING,
    STATE_STOPPED,
    STATE_UNAVAILABLE,
    STATUS_CONTRACT,
    STREAM_IDLE_AFTER_S,
    TIME_BASIS,
)
from tower.cv_lab.preview import LivePreview, PreviewPolicy
from tower.cv_lab.loader import ExperimentLoader
from tower.cv_lab.run import LabRun
from tower.experiments import EXPERIMENTS, ExperimentSettings, experiment_metadata
from tower.modules.base import FrameProcessingError

# Imported for `json_safe` alone, and the direction is deliberate.
# `NaN` and `Infinity` are not JSON: Python emits them bare because
# `allow_nan` defaults True, and a strict decoder -- Swift's, for one --
# then rejects the WHOLE message rather than one field. The result
# channel already sanitises at its envelope boundary, but this document
# has three surfaces and only one of them is that envelope. Sanitising
# where the document is BUILT covers all three at once, and means no
# future surface has to remember.
from tower.results.envelope import json_safe

logger = logging.getLogger(__name__)

# The bound on arming an experiment. The same number and the same reason
# as `ModuleContainer.LOAD_TIMEOUT_S`: 119 MB of MiDaS weights inside a
# 10 s bound would demand ~95 Mbit/s from the first byte, so enforcing the
# tight bound would convert "slow once, then fast forever" into "fails
# every first run". Imported rather than restated so the two cannot drift.
from tower.modules.container import LOAD_TIMEOUT_S as ARM_TIMEOUT_S


class CommandOutcome:
    """What a command did, and the full status afterwards.

    Carries the status either way. A refusal that said only "no" would
    leave a client guessing what state it is now in, and the most common
    reason for a refusal -- somebody else changed the experiment -- is
    precisely the case where the client's picture is out of date.
    """

    __slots__ = ("accepted", "reason", "message", "status", "extra")

    def __init__(
        self,
        *,
        accepted: bool,
        status: dict,
        reason: str | None = None,
        message: str | None = None,
        extra: dict | None = None,
    ) -> None:
        self.accepted = accepted
        self.reason = reason
        self.message = message
        self.status = status
        self.extra = extra or {}


# What each model-backed experiment needs on the import path. Per
# experiment, not one global "is torch there": `depth` needs torch and
# timm (MiDaS's hubconf imports a DPT backbone chain unconditionally),
# and `object_detection` needs torchvision. A single torch probe reported
# `object_detection` available on a Tower that could not run it.
#
# What this canNOT check is the network. `depth` fetches MiDaS weights
# through `torch.hub` on first use, so an offline Tower still accepts the
# start and reports `failed` afterwards with the reason. That is why
# `failed` is recoverable, and why the contract says so.
_REQUIRED_MODULES: dict[str, tuple[str, ...]] = {
    "depth": ("torch", "timm"),
    "object_detection": ("torch", "torchvision"),
}


@functools.lru_cache(maxsize=None)
def _module_is_installed(name: str) -> bool:
    """Whether an import would succeed, WITHOUT performing it.

    CACHED, because this runs on every status build -- once per required
    module of every model-backed experiment -- and `find_spec` walks the
    filesystem. Uncached it cost 0.65 ms of a 1.25 ms status build,
    measured; the whole document costs 0.35 ms without it. A module does
    not appear or disappear while a process runs, and installing the [ml]
    extra into a live Tower needs a restart anyway: the module system
    loads its experiment once.

    `find_spec` locates a module and does not execute it, which is the
    whole point: `test_importing_the_lab_does_not_import_torch` asserts
    that importing `tower.main` leaves torch out of `sys.modules`, and an
    availability check that imported torch to answer "is torch available"
    would be a 2 GB answer to a yes/no question.
    """
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _missing_extra(experiment_id) -> str | None:
    """The first module this experiment needs and this Tower lacks."""
    for name in _REQUIRED_MODULES.get(experiment_id, ()):
        if not _module_is_installed(name):
            return name
    return None


def _clip(value, limit: int = 120) -> str:
    """Bound a string that came from a client before it goes back out.

    `request_id` is capped because it is echoed onto the wire; an
    `experiment_id` embedded in a refusal message is echoed onto the wire
    too, and so is the `str(exc)` of a failed load. A remote party must
    not be able to choose the size of a message this Tower sends.
    """
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "\u2026"


def _wire_safe(value):
    """A runtime fact, reduced to something JSON can carry."""
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return _clip(value, 120)


def _own_process():
    """A handle on this process for the status document, or None.

    Built once: `psutil.Process()` resolves a creation time on
    construction, and the document is built twice a second.
    """
    try:
        return psutil.Process()
    except Exception:  # pragma: no cover - psutil could not see itself
        logger.exception("[Tower][CVLab] psutil could not open this process")
        return None


class CVLab:
    """The one Lab slot's contents, its lifecycle, and what it reports."""

    def __init__(
        self,
        initial_experiment_id: str,
        settings: ExperimentSettings | None = None,
        *,
        experiment=None,
        clock=time.time,
        connection_count=None,
        instance_id: str | None = None,
        preview: PreviewPolicy | None = None,
    ) -> None:
        self._initial_experiment_id = initial_experiment_id
        self._settings = settings or ExperimentSettings()
        self._clock = clock
        self._connection_count = connection_count
        # Minted per process. A client that reconnects to a RESTARTED
        # Tower would otherwise see `run-1` again and could read it as the
        # run it was already watching. Part of every run id for that
        # reason, rather than a field beside it that a client might not
        # compare.
        self.instance_id = instance_id or uuid.uuid4().hex[:12]

        # Injected instance wins, for tests and for a caller that has
        # already built one. Otherwise the registry factory runs in
        # `load_initial`, never at construction: building a detector here
        # would load model weights merely because someone constructed a
        # Lab.
        self._injected_experiment = experiment
        self._experiment = None

        self._guard = threading.Lock()
        self._state = STATE_IDLE
        self._state_reason: str | None = None
        self._state_since = clock()
        self._selected_id: str | None = None
        self._run: LabRun | None = None
        self._run_counter = 0
        # Checked and set with no await between, which is what makes it
        # atomic on a single-threaded event loop. An asyncio.Lock would
        # bind itself to the first loop that used it, and `create_app()`
        # drives startup through `asyncio.run` while everything after runs
        # on uvicorn's loop -- a different one.
        self._switching = False
        self._arm_task: asyncio.Task | None = None
        self._released = False

        # Lab-scoped, not run-scoped, and deliberately: "frames are
        # arriving but the Lab is idle" is the single most useful thing to
        # know when nothing is happening, and a run-scoped counter cannot
        # say it because there is no run.
        self._frames_offered_total = 0
        # Frames that arrived and never reached the Lab, because the
        # transport could not decode them. Counted here rather than only
        # in the session summary because they are the difference between
        # two situations a person diagnoses differently: "nothing is
        # arriving" and "things are arriving and they are broken". Without
        # it those look identical from the status document, and the answer
        # was a server-side log line -- which is exactly what `GET /cv-lab`
        # exists because nobody can see over Tailscale.
        self._frames_rejected_before_lab = 0
        self._last_frame_at: float | None = None

        self._last_frame_provenance: dict | None = None

        # The live preview. One slot, one encoding, no queue and no file.
        # Constructed here rather than injected, for the same reason the
        # Lab itself is constructed inside the module: one Lab, one
        # preview, and no way for two of them to disagree about which run
        # a picture belongs to. What IS injected is the policy, because
        # whether this Tower draws anything at all is an operator's
        # decision and not this class's.
        #
        # It is driven entirely from `_set_state_locked`, which is the one
        # choke point every state change passes through. Hooking pause,
        # resume, stop, fail and release individually would have been five
        # places to forget, and the one that got forgotten would be the
        # one that left a stopped run's last frame on somebody's screen.
        self._preview = LivePreview(preview or PreviewPolicy(), clock=clock)

        # ONE loader thread for every arm this Lab ever performs, started
        # on first use. A fresh thread per arm leaked torch's intra-op team
        # every switch to a model-backed experiment -- 19 OS threads and
        # ~8 MB a time, measured -- see tower/cv_lab/loader.py.
        self._loader = ExperimentLoader()
        self._process = _own_process()

    # -- module-facing lifecycle ---------------------------------------

    async def load_initial(self) -> None:
        """Arm the startup default. Called from `Module._do_load`.

        Loud, and no longer terminal. Until 2026-09-06 an unknown name or
        a failed load here propagated to `ModuleContainer`, which marked
        the module FAILED -- a state with no way back -- so a typo in
        `TOWER_CV_EXPERIMENT`, or a `depth` default on a Tower that could
        not reach torch.hub, killed every experiment for the life of the
        process. The mission brief's "restart Tower" step was this.

        Now the failure is logged at ERROR naming the variable, the Lab
        reports `failed` with the reason on every surface, the module
        reaches ACTIVE, and a client picks any other experiment with
        `cv_lab_start`. What still propagates is `CancelledError`: the
        container's load TIMEOUT cancels this coroutine and must go on
        marking the module FAILED, because a load that overran the bound
        has been abandoned on its thread and the invalidation latch is
        what makes that safe (`tower/loading.py`).
        """
        # A fresh load is not a released Lab. Nothing reaches this
        # today -- `ModuleContainer` has no reload path and `unload()`
        # runs only from `shutdown()` -- but `_released` gates every
        # command, and a sticky flag on a Lab whose `process()` works
        # would refuse every request while answering every frame.
        self._released = False
        experiment_id = self._initial_experiment_id
        experiment = None
        try:
            if self._injected_experiment is None:
                factory = EXPERIMENTS.get(experiment_id)
                if factory is None:
                    raise ValueError(
                        f"unknown experiment {experiment_id!r}; "
                        f"available: {sorted(EXPERIMENTS)}"
                    )
                experiment = factory()
            else:
                experiment = self._injected_experiment

            # Published BEFORE the load, so a status read during a slow
            # startup says `starting` and names what it is starting rather
            # than reporting an idle Lab that is in fact busy.
            self._transition(
                STATE_STARTING,
                selected=experiment_id,
                run=self._new_run(experiment_id, ORIGIN_STARTUP_DEFAULT),
            )
            # Installed BEFORE the load, unlike the interactive path in
            # `_arm`. A partially-loaded startup experiment must still be
            # reachable by `release()`: the container's load timeout
            # ABANDONS the loading thread and then marks the module
            # FAILED, and if nothing holds the instance at that moment,
            # nothing closes `LoadInvalidation`'s latch and the abandoned
            # thread installs a live model -- on CUDA, resident GPU
            # memory -- into a module that will never be released again.
            # `process()` cannot reach it in the meantime because the
            # state is STARTING, not RUNNING.
            with self._guard:
                self._experiment = experiment

            # On the Lab's loader thread, not inline. `asyncio.wait_for`
            # can cancel only at an await point, and a model-backed
            # `load()` is a torch import, a weight download and a
            # `.to(device)`, none of which yield one.
            await self._loader.run(experiment.load, self._settings)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._fail_startup(experiment_id, exc)
            return

        with self._guard:
            self._experiment = experiment
            self._set_state_locked(STATE_RUNNING)
            self._record_runtime_locked(experiment)
            self._record_arm_locked()

    def _fail_startup(self, experiment_id: str, exc: BaseException) -> None:
        """The startup default could not arm. Say so, loudly, and go on."""
        logger.error(
            "[Tower][CVLab] TOWER_CV_EXPERIMENT=%r could not be armed at "
            "startup: %s. The Lab reports `failed`; a client may select any "
            "other experiment with cv_lab_start, and no restart is needed",
            experiment_id,
            exc,
            exc_info=True,
        )
        reason = (
            f"{experiment_id} could not be armed at startup: "
            f"{client_safe_reason(exc)}"
        )
        self._drop_experiment()
        with self._guard:
            self._selected_id = experiment_id
            if self._run is None or self._run.experiment_id != experiment_id:
                # The name was unknown, so no run was ever minted. Mint one
                # so that the document has a run to hang the reason on --
                # a `failed` state with `run: null` would say less than
                # the log did.
                self._run = self._new_run(experiment_id, ORIGIN_STARTUP_DEFAULT)
            self._set_state_locked(STATE_FAILED, reason=_clip(reason))
            if self._run.ended_at is None:
                self._run.ended_at = self._clock()

    async def shutdown(self, reason: str = "the Tower is shutting down") -> None:
        """Release, and WAIT for an in-flight arm to unwind.

        `release()` cannot wait -- it is reachable from `mark_failed()`,
        which the frame path can reach, with no loop to await against. So
        it cancels the arm task and moves on, which is right for a
        failure and wrong for a clean shutdown: cancellation is delivered
        at the next await point, and if the loop closes first the task
        never reaches the `except CancelledError` clause that releases
        what it built. On CUDA that is resident GPU memory with no owner,
        and the process is exiting anyway -- but a shutdown that leaves
        "Task was destroyed but it is pending" in the log is a shutdown
        nobody can read.

        Called from `lifespan`, which is the one place that has both a
        running loop and the authority to wait.
        """
        task = self._arm_task
        self.release(reason)
        if task is None:
            return
        try:
            await task
        except asyncio.CancelledError:
            # The task we just cancelled, not this one. Swallowing our own
            # cancellation would strand the shutdown that requested it.
            current = asyncio.current_task()
            if current is not None and current.cancelling() > 0:
                raise
        except Exception:  # noqa: BLE001
            logger.debug(
                "[Tower][CVLab] the arm task ended badly during shutdown",
                exc_info=True,
            )

    def release(self, reason: str = "the Lab was released") -> None:
        """Free whatever the current experiment holds, and stop serving.

        Called from `Module._do_unload` (clean teardown) and from
        `Module._do_release` (a FAILED transition, reachable from
        anywhere, including from inside a load). Must be safe after a
        partial load and safe to call twice.

        Terminal. The module owning this Lab is gone either way, and the
        Lab reports `unavailable` rather than `idle` so that nobody reads
        a dead slot as one waiting for a request.
        """
        task, self._arm_task = self._arm_task, None
        if task is not None:
            task.cancel()
        with self._guard:
            self._released = True
            # Cleared here as well as on every completion path. Leaving it
            # set after a release mid-arm made `_switching` stop being an
            # invariant: nothing could reach `start()` past the
            # `_released` check today, but the next person to reorder
            # those checks would get a permanently busy Lab.
            self._switching = False
            self._set_state_locked(STATE_UNAVAILABLE, reason=reason)
            # A run that will never process another frame has ended.
            # Without this the document keeps publishing `ended_at: null`
            # with `elapsed_s` growing and `processed_fps` decaying
            # towards zero, for a Lab that is gone.
            if self._run is not None and self._run.ended_at is None:
                self._run.ended_at = self._clock()
            experiment, self._experiment = self._experiment, None
            self._last_frame_provenance = None
            # Unconditional, and not only through `_set_state_locked`:
            # releasing an already-`unavailable` Lab is not a state
            # CHANGE, so the transition hook would not fire, and the one
            # thing a release must guarantee is that nothing is left
            # holding a picture.
            self._preview.end()
        if experiment is not None:
            experiment.release()
        # The loader thread exits once it is idle. An arm that is still
        # loading keeps it alive until the load returns and is discarded.
        self._loader.close()

    # -- the frame path -------------------------------------------------

    def process(self, raw_bytes: bytes):
        """Run the armed experiment over one frame, or refuse it legibly.

        Synchronous and free of awaits, which is load-bearing: it is what
        makes a switch unable to interleave inside a frame, and what makes
        `frame_provenance()` describe this frame.
        """
        now = self._clock()
        self._frames_offered_total += 1
        self._last_frame_at = now
        run = self._run
        state = self._state
        experiment = self._experiment

        if state != STATE_RUNNING or experiment is None:
            self._last_frame_provenance = None
            if run is not None:
                run.record_refused(now)
            raise FrameProcessingError(
                self._refusal_message(state),
                reason=FRAME_REFUSAL_REASONS.get(state, "cv_lab_unavailable"),
            )

        self._arm_preview_for_frame(now)
        try:
            result = experiment.run(raw_bytes)
        except FrameProcessingError:
            # The experiment refused THIS frame and said why -- it would
            # not decode, it was too small. The run goes on.
            self._last_frame_provenance = None
            if run is not None:
                run.record_failed(now)
            raise
        except Exception as exc:
            # Anything else is a bug in the experiment, and until
            # 2026-09-06 it propagated to `ModuleContainer.process`,
            # which marks the module FAILED -- terminal, for the life of
            # the process. One odd frame ended every experiment until a
            # restart. Now it ends the RUN: the experiment is released,
            # the state is `failed` with the reason, and the container
            # sees a refused frame. The next `cv_lab_start` arms a fresh
            # instance. `BaseException` (KeyboardInterrupt, SystemExit)
            # is not a run failure and still propagates below.
            self._last_frame_provenance = None
            if run is not None:
                run.record_failed(now)
            self._fail_run_on_frame(run, experiment, exc)
            raise FrameProcessingError(
                f"{self._selected_id} failed while processing a frame and "
                f"its run has ended: {client_safe_reason(exc)}. Select an "
                "experiment to continue.",
                reason=FRAME_REFUSED_FAILED,
            ) from exc
        except BaseException:
            self._last_frame_provenance = None
            if run is not None:
                run.record_failed(now)
            raise

        result_seq = run.record_result(result, now) if run is not None else 0
        try:
            self._last_frame_provenance = self._provenance(
                run, result, result_seq, now
            )
        except Exception:
            # Attribution is a diagnostic. A result that does not look
            # like an `ExperimentResult` is a real bug and is logged as
            # one -- but raising here would leave `ModuleContainer` no
            # choice but `mark_failed()`, which is TERMINAL, so one odd
            # frame would end CV processing for the life of the process.
            self._last_frame_provenance = None
            logger.exception(
                "[Tower][CVLab] could not attribute a result from %s; the "
                "frame is answered without provenance",
                self._selected_id,
            )
        self._offer_preview(run, result_seq, now)
        return result

    def _arm_preview_for_frame(self, now: float) -> None:
        """Tell the experiment whether to build a picture for THIS frame.

        Called BEFORE `run()`, which is the only place it can be called
        from. Four of the seven visual experiments derive something to
        draw -- a line drawing of the room, a thinned keypoint set, the
        boxes pulled off a tensor -- and that derivation has to happen
        while the frame's intermediates are still in scope. Deciding
        afterwards would mean either keeping the intermediates alive
        between frames, which is imagery this module has always declared
        it does not hold, or recomputing them, which is the redundant CV
        work this whole design exists to avoid.

        So the throttle is asked here, one frame early, and its answer is
        one boolean assignment on the experiment. A frame the throttle
        turns down costs exactly that assignment: no line drawing, no
        keypoints, no boxes, and nothing retained.

        Never raises, for the usual reason -- `ModuleContainer` turns
        anything that is not a `FrameProcessingError` into a TERMINAL
        module failure, and a picture must not be able to end a run.
        """
        preview = self._preview
        if not preview.is_live:
            return
        setter = getattr(self._experiment, "set_preview_capture", None)
        if setter is None:
            return
        try:
            setter(preview.wants_capture(now))
        except Exception:
            logger.exception(
                "[Tower][CVLab] %s refused a per-frame preview arm; the "
                "frame is processed without one",
                self._selected_id,
            )

    def _offer_preview(self, run: LabRun | None, result_seq: int, now: float) -> None:
        """Hand this frame's derived array to the preview. Never raises.

        The whole of the visualisation cost on the measured path, and it
        is deliberately almost nothing: three attribute reads to find out
        nobody is watching, or a clock comparison and two assignments to
        keep an array the experiment had already built. No resize, no
        colour conversion, no encode, no copy and no lock. The expensive
        half runs in `LivePreview.render`, on a worker thread, when a
        client actually asks -- which is the whole reason a viewer cannot
        backpressure this pipeline.

        Placed AFTER the result and after provenance, so a preview that
        went wrong cannot affect either. And wrapped, because
        `ModuleContainer` turns anything that is not a
        `FrameProcessingError` into a TERMINAL module failure: a bug in a
        picture would otherwise end CV processing for the life of the
        process, which is the exact trade this feature must never make.
        """
        preview = self._preview
        try:
            if not preview.is_live:
                return
            if not preview.wants_capture(now):
                preview.note_throttled()
                return
            take = getattr(self._experiment, "take_preview", None)
            if take is None:
                preview.note_empty()
                return
            array = take()
            if array is None:
                preview.note_empty()
                return
            preview.capture(
                array,
                run_id=run.run_id if run is not None else None,
                result_seq=result_seq,
                now=now,
            )
        except Exception:
            logger.exception(
                "[Tower][CVLab] capturing a preview from %s failed; the "
                "result is unaffected and telemetry continues",
                self._selected_id,
            )

    def note_frame_rejected_before_processing(self, now: float | None = None) -> None:
        """A frame arrived and the transport refused it. Never raises.

        Called from `ws.py` when `parse_and_decode_frame` fails, which is
        the one thing that happens to a frame before the Lab can see it.
        It updates `last_frame_at` too: a malformed frame is still
        evidence that something is streaming, and `receiving_frames`
        would otherwise say no while a phone was sending as fast as it
        could.
        """
        self._frames_rejected_before_lab += 1
        self._last_frame_at = self._clock() if now is None else now

    # -- the live preview, from a worker thread ------------------------
    #
    # Three thin passes through to `LivePreview`, and thin on purpose:
    # `routes/cv_lab_preview.py` holds `app.state.cv_lab`, not the
    # preview inside it, for the same reason `GET /cv-lab` holds the Lab
    # rather than the run -- a route that reached into an object's
    # internals would be a second place that has to know how the Lab is
    # put together.

    def render_preview(self, *, run_id: str | None = None, if_none_match=None):
        """The newest preview, encoded. Never raises. See `LivePreview`."""
        return self._preview.render(run_id=run_id, if_none_match=if_none_match)

    def preview_descriptor(self) -> dict | None:
        """What a preview would be, without fetching one."""
        return self._preview.descriptor()

    def preview_unavailable_reason(self) -> str | None:
        """Why there is no descriptor, in a sentence, when there is none."""
        return self._preview.why_none()

    def frame_provenance(self) -> dict | None:
        """Who produced the frame result `process()` just returned.

        Read by `ws.py` immediately after `process()` and before the
        `await` that sends the reply. `None` means the frame produced no
        result, so there is nothing to attribute.

        Deliberately NOT attached to the `ExperimentResult` itself: a
        module returns what its experiment returned, and wrapping the
        result would make `Module.process()` mean something different for
        this module than for every other one.
        """
        provenance, self._last_frame_provenance = self._last_frame_provenance, None
        return provenance

    def _provenance(self, run, result, result_seq: int, now: float) -> dict:
        descriptor = run.descriptor if run is not None else {}
        return {
            "contract": FRAME_RESULT_CONTRACT,
            # Which Tower process. A reconnect to a restarted Tower cannot
            # otherwise be told from a reconnect to the same one.
            "tower_instance_id": self.instance_id,
            # Which run. THE staleness check: a client watching run X must
            # discard a result carrying run Y.
            "run_id": run.run_id if run is not None else None,
            # Dense within the run, starting at 1. The wire `seq` is the
            # phone's capture index and skips by design, so it cannot be
            # used to order results.
            "result_seq": result_seq,
            "experiment_id": descriptor.get("id"),
            "experiment_name": descriptor.get("name"),
            # Measured or inferred, on every frame, not only in the
            # aggregate. Rule 16 / Core Principle 2.
            "provenance": descriptor.get("provenance"),
            "backend": descriptor.get("backend"),
            "device": run.runtime.get("device") if run is not None else None,
            "device_requested": self._settings.device,
            "result_label": result.result_label,
            # The Tower measuring itself. Repeated here rather than only
            # at the top level so that the whole attribution block is
            # self-contained -- a consumer keeping one of these does not
            # have to keep the message around it too.
            "processing_ms": result.processing_ms,
            "tower_received_at": now,
            "time_basis": TIME_BASIS,
        }

    def _refusal_message(self, state: str) -> str:
        if state == STATE_IDLE:
            return (
                "the CV Lab is idle: no experiment is armed, so this frame "
                "produced no result. Send cv_lab_start to arm one."
            )
        if state == STATE_STARTING:
            return (
                "the CV Lab is arming an experiment; frames are refused "
                "until it is ready rather than answered by the previous one"
            )
        if state == STATE_PAUSED:
            return "the CV Lab is paused; send cv_lab_resume to continue"
        if state == STATE_STOPPED:
            return (
                "the CV Lab is stopped; the last run's figures are final. "
                "Send cv_lab_start to begin another"
            )
        if state == STATE_FAILED:
            return (
                "the CV Lab's last start failed: "
                f"{self._state_reason or 'no reason recorded'}"
            )
        return (
            "the CV Lab cannot run experiments on this Tower: "
            f"{self._state_reason or 'no reason recorded'}"
        )

    # -- commands -------------------------------------------------------

    def start(self, experiment_id, *, origin: str = ORIGIN_CLIENT_REQUEST):
        """Arm an experiment. Returns as soon as the request is decided.

        Synchronous on purpose. The load runs in a background task, so a
        `depth` start that spends two minutes downloading weights does not
        stop the connection that asked for it from sending frames -- and
        iOS's `run(_:)` already returns nothing and reports through state,
        so there is nothing here for it to wait on.
        """
        refusal = self._unavailable_refusal()
        if refusal is not None:
            return refusal
        if not isinstance(experiment_id, str) or not experiment_id:
            return self._refuse(
                ERR_MALFORMED, "cv_lab_start requires a string 'experiment_id'"
            )
        if not catalog.is_registered(experiment_id):
            return self._refuse(
                ERR_UNKNOWN_EXPERIMENT,
                f"this Tower has no experiment {_clip(experiment_id)!r}",
                extra={"available": sorted(EXPERIMENTS)},
            )
        missing = _missing_extra(experiment_id)
        if experiment_metadata(experiment_id).requires_model and missing:
            return self._refuse(
                ERR_EXPERIMENT_UNAVAILABLE,
                f"{experiment_id!r} needs the optional [ml] extra "
                f"({missing}), which is not installed on this Tower",
                extra={"experiment_id": experiment_id},
            )
        if self._switching:
            return self._refuse(
                ERR_LAB_BUSY,
                "another start is already in flight; the Lab holds one "
                "experiment and will not queue a second request behind it",
            )

        # Set with no await in between, so no second command can pass this
        # check before the flag is up.
        self._switching = True
        run = self._new_run(experiment_id, origin)
        # The old experiment goes BEFORE the new run id is published.
        # After this line nothing can produce a result, so no result can
        # be produced under the wrong name.
        self._drop_experiment()
        self._transition(STATE_STARTING, selected=experiment_id, run=run)
        self._arm_task = asyncio.get_running_loop().create_task(
            self._arm(experiment_id, run.run_id)
        )
        return CommandOutcome(accepted=True, status=self.status())

    async def _arm(self, experiment_id: str, run_id: str) -> None:
        """Load the experiment and publish the outcome. Never raises."""
        experiment = None
        try:
            experiment = EXPERIMENTS[experiment_id]()
            # Bounded, and the bound is the container's. A stalled weight
            # download must not leave the Lab in `starting` forever with
            # no state transition anyone can see -- Rule 15 wants a
            # defined failure transition, and `failed` is it.
            await asyncio.wait_for(
                self._loader.run(experiment.load, self._settings),
                timeout=ARM_TIMEOUT_S,
            )
        except asyncio.CancelledError:
            # A stop landed mid-load. Release BEFORE unwinding: the load
            # runs on a thread nobody joins, and closing the latch here is
            # what makes that thread free what it built instead of
            # installing it into an object with no owner.
            self._release_quietly(experiment)
            raise
        except BaseException as exc:
            self._release_quietly(experiment)
            # Reported as STATE, never as a refusal. The command was
            # already answered `accepted` -- an arm is asynchronous and
            # that is the whole reason it is -- so the outcome arrives
            # the way iOS's own `run(_:)` expects it to: through the
            # status document, pushed on the result channel or read with
            # `cv_lab_status`. There is no `start_failed` refusal code
            # for the same reason there is no reply to a reply.
            self._fail_arm(
                run_id,
                f"{experiment_id} could not be armed: {client_safe_reason(exc)}",
            )
            logger.exception(
                "[Tower][CVLab] arming %s failed; the Lab stays available "
                "and another start may be sent",
                experiment_id,
            )
            return

        # Whether THIS task is still the arm the Lab is waiting on. A
        # cancelled task does not always raise: `stop()` cancels and then
        # a `start()` may arm again, and if this task had already passed
        # its last await it runs to completion regardless. Clearing
        # `_switching` unconditionally then cleared the NEW start's flag
        # and let a third start race the second. The wrong experiment
        # still could not be installed -- the run-id check below sees to
        # that -- but `lab_busy` stopped meaning what it says, and a
        # guarantee that holds only most of the time is not one.
        mine = self._arm_task is asyncio.current_task()
        with self._guard:
            if (
                self._released
                or self._state != STATE_STARTING
                or self._run is None
                or self._run.run_id != run_id
            ):
                # Superseded while we were loading. Whatever we built must
                # not be installed -- see tower/loading.py for why release
                # running first is the whole problem.
                stale = experiment
            else:
                stale = None
                self._experiment = experiment
                self._set_state_locked(STATE_RUNNING)
                self._record_runtime_locked(experiment)
                self._record_arm_locked()
            if mine:
                self._switching = False
        if stale is not None:
            self._release_quietly(stale)
            logger.info(
                "[Tower][CVLab] %s finished arming after run %s was "
                "superseded; discarded",
                experiment_id,
                run_id,
            )

    async def wait_until_armed(self) -> None:
        """Await the in-flight arm, if there is one.

        Not part of the wire surface: a client learns that arming finished
        from the status document. This exists so a test does not have to
        sleep. `shutdown()` is what a teardown calls -- it releases FIRST
        and then joins, which is the order that makes the arm task free
        what it built rather than install it.
        """
        task = self._arm_task
        if task is None:
            return
        try:
            await task
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None and current.cancelling() > 0:
                raise
        except Exception:  # noqa: BLE001
            logger.debug("[Tower][CVLab] arm task ended badly", exc_info=True)

    def _fail_arm(self, run_id: str, reason: str) -> None:
        # Same ownership check as the success path, for the same reason:
        # a superseded task must not clear the flag its successor set.
        try:
            mine = self._arm_task is asyncio.current_task()
        except RuntimeError:
            # Called from outside a task (a test drives it directly).
            mine = True
        with self._guard:
            if mine:
                self._switching = False
            if self._run is not None and self._run.run_id == run_id:
                self._set_state_locked(STATE_FAILED, reason=_clip(reason))
                # The run never started and never will. Ending it keeps
                # `elapsed_s` from growing for a run that processed
                # nothing and is not going to.
                self._run.ended_at = self._clock()

    def pause(self, run_id=None):
        refusal = self._unavailable_refusal() or self._check_run_id(run_id)
        if refusal is not None:
            return refusal
        if self._state != STATE_RUNNING:
            return self._refuse(
                ERR_INVALID_STATE,
                f"cannot pause: the Lab is {self._state}, not running",
            )
        # The experiment stays loaded. That is the whole difference from
        # stop: a paused depth run resumes instantly, a stopped one pays
        # the load again.
        self._transition(STATE_PAUSED)
        return CommandOutcome(accepted=True, status=self.status())

    def resume(self, run_id=None):
        refusal = self._unavailable_refusal() or self._check_run_id(run_id)
        if refusal is not None:
            return refusal
        if self._state != STATE_PAUSED:
            return self._refuse(
                ERR_INVALID_STATE,
                f"cannot resume: the Lab is {self._state}, not paused",
            )
        if self._experiment is None:
            return self._refuse(
                ERR_INVALID_STATE,
                "cannot resume: the paused run holds no armed experiment",
            )
        self._transition(STATE_RUNNING)
        return CommandOutcome(accepted=True, status=self.status())

    def stop(self, run_id=None):
        refusal = self._unavailable_refusal() or self._check_run_id(run_id)
        if refusal is not None:
            return refusal
        if self._state not in (STATE_RUNNING, STATE_PAUSED, STATE_STARTING):
            return self._refuse(
                ERR_INVALID_STATE,
                f"cannot stop: the Lab is {self._state}, and there is no run "
                "to end",
            )
        # A stop lands during an arm too, and must win: the task checks
        # the run id it was given and discards what it built.
        self._switching = False
        task, self._arm_task = self._arm_task, None
        if task is not None:
            task.cancel()
        self._drop_experiment()
        with self._guard:
            if self._run is not None:
                self._run.ended_at = self._clock()
            self._set_state_locked(STATE_STOPPED)
        return CommandOutcome(accepted=True, status=self.status())

    def _unavailable_refusal(self):
        """`lab_unavailable`, or None. Checked before anything else.

        Without it a dead Lab answered `invalid_state` -- which tells a
        client "try again from another state", of a condition no state
        change can fix. Two codes on a closed set mean two different
        things and only one of them is true here.
        """
        if self._released or self._state == STATE_UNAVAILABLE:
            return self._refuse(
                ERR_LAB_UNAVAILABLE,
                "this Tower cannot run experiments: "
                f"{self._state_reason or 'the Lab module is not available'}",
            )
        return None

    def _check_run_id(self, run_id):
        if run_id is None:
            return None
        if not isinstance(run_id, str):
            return self._refuse(
                ERR_MALFORMED, "'run_id' must be a string or absent"
            )
        current = self._run.run_id if self._run is not None else None
        if run_id != current:
            return self._refuse(
                ERR_STALE_RUN,
                f"run {_clip(run_id)!r} is not the current run; the Lab is "
                f"now on {current!r}",
                extra={"current_run_id": current},
            )
        return None

    def _refuse(self, reason: str, message: str, *, extra: dict | None = None):
        return CommandOutcome(
            accepted=False,
            reason=reason,
            message=message,
            status=self.status(),
            extra=extra,
        )

    # -- state ----------------------------------------------------------

    def _new_run(self, experiment_id: str, origin: str) -> LabRun:
        self._run_counter += 1
        return LabRun(
            run_id=f"{self.instance_id}-{self._run_counter}",
            experiment_id=experiment_id,
            descriptor=catalog.descriptor(experiment_id)
            if catalog.is_registered(experiment_id)
            else {"id": experiment_id, "name": experiment_id, "summary": None},
            origin=origin,
            started_at=self._clock(),
        )

    def _transition(self, state: str, *, reason=None, selected=None, run=None) -> None:
        with self._guard:
            if selected is not None:
                self._selected_id = selected
            if run is not None:
                self._run = run
            self._set_state_locked(state, reason=reason)

    def _set_state_locked(self, state: str, *, reason: str | None = None) -> None:
        previous = self._state
        self._state = state
        self._state_reason = reason
        self._state_since = self._clock()
        if state != previous:
            self._sync_preview_locked(state)

    def _sync_preview_locked(self, state: str) -> None:
        """The live preview follows RUNNING and nothing else. Never raises.

        Every lifecycle question the preview has -- start, pause, resume,
        stop, a failed arm, a released module -- is the same question:
        is the Lab running right now. Answering it here, at the one place
        every state change passes through, is what makes "a paused run
        shows no picture" and "a stopped run's last frame is gone"
        properties of the state machine rather than five separate
        promises that each had to be remembered.

        Two things happen together and neither is optional. The slot is
        begun or emptied, and the EXPERIMENT is told whether to keep its
        array at all -- so a Tower that is not running, or has previews
        off, holds no derived imagery anywhere, which is what
        `ExperimentalCVModule`'s `retains_raw_imagery=False` has always
        claimed and must go on being true.
        """
        try:
            if state == STATE_RUNNING:
                run = self._run
                watching = self._preview.begin(
                    run.run_id if run is not None else None,
                    self._preview_kind_locked(),
                )
            else:
                self._preview.suspend()
                watching = False
            self._set_capture_locked(watching)
        except Exception:
            # Diagnostics-shaped failure on a state transition. Raising
            # here would propagate out of `stop()` or, worse, out of
            # `release()` -- which runs on the FAILED transition, where
            # there is nothing left to fail into.
            logger.exception(
                "[Tower][CVLab] could not follow the %s transition with the "
                "live preview; the run is unaffected",
                state,
            )

    def _preview_kind_locked(self) -> str | None:
        """What the current experiment would draw, or None for nothing.

        Read from the DECLARATION rather than from the loaded object.
        An experiment that implements `take_preview` without declaring a
        `preview_kind` has not said how its array should be read, and a
        renderer that guessed would be inventing the contract this whole
        change exists to write down.
        """
        run = self._run
        if run is None or not catalog.is_registered(run.experiment_id):
            return None
        return experiment_metadata(run.experiment_id).preview_kind

    def _set_capture_locked(self, enabled: bool) -> None:
        """Tell the experiment whether anybody is watching. Never raises.

        Optional on the protocol, exactly like `describe()`: six of the
        eight registered experiments have no picture and implement
        neither, and the Lab treats their absence as "nothing to keep"
        rather than as an error.
        """
        setter = getattr(self._experiment, "set_preview_capture", None)
        if setter is None:
            return
        try:
            setter(enabled)
        except Exception:
            logger.exception(
                "[Tower][CVLab] an experiment refused set_preview_capture(%r); "
                "the run is unaffected and no preview will be served",
                enabled,
            )

    @staticmethod
    def _release_quietly(experiment) -> None:
        """Release an experiment nobody else holds. Never raises.

        Every call site is already handling a failure, and a release that
        raised would replace a legible one with an illegible one.
        """
        if experiment is None:
            return
        try:
            experiment.release()
        except Exception:
            logger.exception("[Tower][CVLab] releasing an experiment raised")

    def _drop_experiment(self) -> None:
        with self._guard:
            experiment, self._experiment = self._experiment, None
            self._last_frame_provenance = None
        self._release_quietly(experiment)

    def _fail_run_on_frame(self, run, experiment, exc: BaseException) -> None:
        """A frame crash ends the run and frees the experiment. Never raises.

        Guarded on identity: `process()` snapshots the experiment without
        the lock, so by the time a slow frame raises, a switch may already
        have installed a successor. Failing the successor's run for the
        predecessor's crash would be the misattribution the run id exists
        to prevent, so the state changes only if the crashed instance is
        still the one installed.
        """
        logger.exception(
            "[Tower][CVLab] %s raised on a frame; run %s is failed and the "
            "experiment released. The Lab stays available and another start "
            "may be sent",
            self._selected_id,
            run.run_id if run is not None else None,
        )
        reason = _clip(
            f"{self._selected_id} failed on a frame: {client_safe_reason(exc)}"
        )
        with self._guard:
            still_current = self._experiment is experiment
            if still_current:
                self._experiment = None
                self._last_frame_provenance = None
                self._set_state_locked(STATE_FAILED, reason=reason)
                if self._run is not None and self._run.ended_at is None:
                    self._run.ended_at = self._clock()
        if still_current:
            self._release_quietly(experiment)

    def _record_arm_locked(self) -> None:
        """How long the current run took to arm. Set once, on install."""
        run = self._run
        if run is not None and run.arm_ms is None:
            run.arm_ms = round(
                (time.perf_counter() - run.arm_started_perf) * 1000.0, 1
            )

    def _record_runtime_locked(self, experiment) -> None:
        """Ask the experiment what it actually loaded, if it will say.

        Optional on the `Experiment` protocol. An experiment that does not
        implement `describe()` reports nothing, which is different from
        reporting a device it does not have.
        """
        if self._run is None:
            return
        describe = getattr(experiment, "describe", None)
        if describe is None:
            return
        try:
            described = describe()
        except Exception:
            logger.exception(
                "[Tower][CVLab] %s.describe() raised; runtime facts omitted",
                self._run.experiment_id,
            )
            return
        if isinstance(described, dict):
            # Bounded and stringified: this is a diagnostic block, not a
            # channel for an experiment to put arbitrary objects on the
            # wire. `None` survives as `null` rather than becoming the
            # string "None" -- "we do not know which device" and "the
            # device is called None" are different claims.
            self._run.runtime = {
                str(key)[:64]: _wire_safe(value)
                for key, value in list(described.items())[:8]
            }

    # -- reporting ------------------------------------------------------

    def availability(self) -> tuple[bool, str | None]:
        """Whether this Tower can serve the CV Lab contract at all.

        Duck-typed by `tower/results/registry.py`, which must not import
        this module: the result channel core is cartridge-blind and stays
        that way.
        """
        with self._guard:
            if self._state == STATE_UNAVAILABLE:
                return False, (
                    self._state_reason
                    or "the Lab module is not available on this Tower"
                )
        if not EXPERIMENTS:
            return False, "this Tower has no registered experiments"
        return True, None

    def status(self) -> dict:
        """The one document. Same bytes on every surface that serves it.

        Called from the event loop (a command reply) and from a worker
        thread (the result channel's poller), which is what `_guard` is
        for.
        """
        with self._guard:
            state = self._state
            reason = self._state_reason
            since = self._state_since
            selected = self._selected_id
            run = self._run
            frames_offered_total = self._frames_offered_total
            rejected_before_lab = self._frames_rejected_before_lab
            last_frame_at = self._last_frame_at

        now = self._clock()
        # Sanitised HERE, once, so that all three surfaces are covered.
        # The result channel sanitises at its envelope boundary; `GET
        # /cv-lab` goes through Starlette with `allow_nan=False` and would
        # answer 500; `cv_lab_status` goes through `send_json`, whose
        # `allow_nan` defaults True and would put a bare `NaN` on the wire
        # for a strict decoder to reject the whole message over. Three
        # different failures from one non-finite float.
        return json_safe({
            "contract": STATUS_CONTRACT,
            "control_contract": CONTROL_CONTRACT,
            "frame_result_contract": FRAME_RESULT_CONTRACT,
            "tower_instance_id": self.instance_id,
            "time_basis": TIME_BASIS,
            "lifecycle": {
                "state": state,
                # Prose, for a person. Present only when the state needs
                # explaining; `null` is not "no reason", it is "the state
                # speaks for itself".
                "reason": reason,
                "since": since,
                "run_id": run.run_id if run is not None else None,
            },
            "available": self._available_experiments(),
            # The id the Lab is on, whatever it is doing with it. Distinct
            # from `lifecycle.run_id`: a stopped run still names the
            # experiment it ran.
            "selected": selected,
            # What this process would arm at boot. Reported so that a
            # person reading "running edge_detection" can see whether that
            # survives a restart.
            "default_experiment": self._initial_experiment_id,
            "device_requested": self._settings.device,
            "run": self._run_document(run, now),
            "source": {
                # Whether anything is feeding this Lab, which is the
                # question "I pressed Start and nothing happened" is
                # really asking.
                "clients_connected": self._clients_connected(),
                "receiving_frames": (
                    last_frame_at is not None
                    and (now - last_frame_at) <= STREAM_IDLE_AFTER_S
                ),
                "last_frame_at": last_frame_at,
                "frames_offered_total": frames_offered_total,
                # Arriving, and refused before the Lab could look. Non-zero
                # with `frames_offered_total` at zero means the stream is
                # alive and its frames are undecodable, which is a
                # different fix from "nothing is streaming".
                "frames_rejected_before_lab": rejected_before_lab,
                "idle_after_s": STREAM_IDLE_AFTER_S,
            },
        })

    def process_stats(self) -> dict:
        """What this Tower process is using right now.

        Served beside the status document on `GET /cv-lab`, and
        deliberately NOT inside it. The document is one object on three
        surfaces and a test holds them byte-equal; thread and RSS
        figures differ between two reads a millisecond apart, and the
        result channel would republish the document on every poll for a
        number that changed by 0.1 MB. This is an operator's reading --
        "did the last switch give its memory back" -- for `curl` and the
        switch soak, from a Tower nobody has a shell on.
        """
        threads = rss_mb = None
        if self._process is not None:
            try:
                threads = self._process.num_threads()
                rss_mb = round(self._process.memory_info().rss / (1024 * 1024), 1)
            except Exception:
                logger.debug("[Tower][CVLab] could not read process stats", exc_info=True)
        return {"pid": os.getpid(), "threads": threads, "rss_mb": rss_mb}

    def _clients_connected(self) -> int | None:
        if self._connection_count is None:
            return None
        try:
            return int(self._connection_count())
        except Exception:
            logger.exception("[Tower][CVLab] could not read the connection count")
            return None

    def _available_experiments(self) -> list[dict]:
        return [self._with_availability(entry) for entry in catalog.catalog()]

    def _with_availability(self, entry: dict) -> dict:
        """One catalog entry plus whether this Tower can run it.

        Applied to the entry inside `run.experiment` as well as to the
        catalog, so a client has ONE shape to decode. They differed by two
        keys before, which is exactly the kind of difference a hand-written
        decoder discovers by dropping a message.
        """
        row = dict(entry)
        missing = _missing_extra(entry.get("id"))
        if entry.get("requires_model") and missing is not None:
            row["available"] = False
            row["unavailable_reason"] = (
                f"needs the optional [ml] extra ({missing}), which is not "
                "installed on this Tower"
            )
        else:
            row["available"] = True
            row["unavailable_reason"] = None
        return row

    def _run_document(self, run: LabRun | None, now: float) -> dict | None:
        if run is None:
            return None
        metadata = (
            experiment_metadata(run.experiment_id)
            if catalog.is_registered(run.experiment_id)
            else None
        )
        if metadata is None:
            metrics, omitted = [], 0
        else:
            metrics, omitted = run.metric_rows(metadata)
        elapsed = max((run.ended_at or now) - run.started_at, 0.0)
        # The three counters read ONCE, into locals, and
        # `frames_offered` derived from those. Reading `run.frames_offered`
        # and then the three attributes separately was atomic only because
        # CPython happens to schedule no eval-breaker check between a
        # property's return and the loads that follow it -- which is not a
        # guarantee, and is certainly not the "true by construction" this
        # design claims. Four reads of one consistent triple are.
        processed = run.frames_processed
        refused = run.frames_refused
        failed = run.frames_failed
        return {
            "run_id": run.run_id,
            "experiment": self._with_availability(run.descriptor),
            # Who asked for this run. `startup_default` means nobody did.
            "origin": run.origin,
            "started_at": run.started_at,
            "ended_at": run.ended_at,
            "elapsed_s": round(elapsed, 3),
            # From `cv_lab_start` accepted to the first frame this run
            # could answer. `null` while arming, and `null` forever for a
            # run whose arm failed.
            "arm_ms": run.arm_ms,
            # What the experiment says it actually loaded -- device,
            # weights, versions. Empty for an experiment that holds none.
            "runtime": dict(run.runtime),
            "frames_offered": processed + refused + failed,
            "frames_processed": processed,
            "frames_refused": refused,
            "frames_failed": failed,
            "metrics": metrics,
            "metrics_omitted": omitted,
            # An experiment emitting a metric it never classified. Empty
            # is the only correct value and a test enforces it for every
            # registered experiment; this is what the wire says if one
            # ever reaches production anyway.
            "unclassified_metrics": run.unclassified_metrics,
            "annotation": self._annotation(run, metadata),
            # How the picture is doing, kept apart from how the
            # EXPERIMENT is doing. `timings.processing_ms` beside this is
            # the model's cost and must stay comparable against every
            # figure recorded before previews existed; `preview.render_ms`
            # is what a picture costs, on a different thread, at a
            # different rate, and mixing the two would destroy the one
            # measurement that answers "how much did the viewer cost us".
            "preview": self._preview.stats(),
            "timings": {
                "processing_ms": (
                    round(run.processing_ms.average, 4)
                    if run.processing_ms.count
                    else None
                ),
                "processing_ms_max": (
                    round(run.processing_ms.maximum, 4)
                    if run.processing_ms.count
                    else None
                ),
                "stage_ms": {
                    name: round(stage.average, 4)
                    for name, stage in sorted(run.stage_ms.items())
                },
                # When the Tower last produced a result for this run.
                # Tower-receipt time -- there is no capture timestamp
                # anywhere on the wire, so this is when the TOWER saw a
                # frame, never when the glasses did.
                "observed_at": run.last_result_at,
                "time_basis": TIME_BASIS,
            },
            "throughput": {
                # Frames the Lab PROCESSED per second of run wall-clock.
                # Not the capture rate and not the link rate: the current
                # sender forwards roughly one frame in thirty, so this
                # figure is bounded by what arrives, not by what the Lab
                # could do. `frames_offered` beside it is what makes the
                # difference visible.
                # `null`, not `0.0`, when there is no window yet. A
                # rate over a zero-length interval is undefined, not zero
                # -- and on Windows `time.time()` has ~15.6 ms
                # granularity, so the reply to `cv_lab_start` almost
                # always lands inside the same tick as the run's own
                # start. A client sees `null` for the first few
                # milliseconds and a number thereafter.
                "processed_fps": (
                    round(processed / elapsed, 3) if elapsed > 0 else None
                ),
                "offered_fps": (
                    round((processed + refused + failed) / elapsed, 3)
                    if elapsed > 0
                    else None
                ),
                # The other direction: how fast the Lab could go if frames
                # never stopped arriving. Derived from measured
                # per-frame cost alone.
                "capacity_fps": (
                    round(1000.0 / run.processing_ms.average, 2)
                    if run.processing_ms.count and run.processing_ms.average > 0
                    else None
                ),
            },
        }

    def _annotation(self, run: LabRun, metadata) -> dict:
        """The annotation half of iOS's `CVAnnotationReport`.

        `count` is `null` when the experiment does not produce one and a
        NUMBER when it does, including zero. `0` is a real result meaning
        "found nothing" and must not merge with "did not say".

        `artifact` is no longer always `null`, and the sentence that used
        to be here is worth keeping in view because it is the standard
        this had to meet. It said: `IOS-to-Tower.md` 5 withholds any
        image whose treatment is unstated, and artifact fetching itself
        is UNKNOWN -- iOS "holds no URL, no id format, and no bytes,
        because inventing a fetch scheme would be exactly the fabricated
        contract this work refuses to produce", so "serving an inline
        image here would be the Tower inventing that scheme
        unilaterally". It ended: "The field exists so that a later
        contract adds a payload where a `null` is, rather than adding a
        field."

        This is that contract, and it is the payload going where the
        `null` was. What makes it not the fabrication that sentence
        refused:

        - it landed on BOTH sides in one change, with
          `docs/contracts/EXPERIMENTAL-CV-PREVIEW.md` written down,
          rather than one side guessing what the other would accept;
        - the treatment is STATED, in iOS's own vocabulary, and it is
          the strict value: `raw_ephemeral`, live view only, never
          persisted and never re-served;
        - no image goes INLINE. What is here is a path, a media type and
          a treatment -- the bytes are a separate fetch a client makes
          only if it wants one, which is what `IOS-to-Tower.md`'s own
          `notFetched`/`fetching`/`available` state machine was written
          around;
        - it is `null` again, with a reason, the moment there is nothing
          honest to put here -- previews off, or an experiment with
          nothing to draw.
        """
        count = None
        if metadata is not None and metadata.annotation_metric:
            total = run.metric_total(metadata.annotation_metric)
            # `isfinite` before `int(round(...))`, because `int(round(nan))`
            # RAISES. `json_safe` wraps the finished document and cannot
            # protect a computation that happens while the document is
            # being built: one non-finite detection count made `status()`
            # raise, which answered `GET /cv-lab` with a 500, the socket
            # with an error, and the result channel with `snapshot_failed`
            # -- permanently for that run, because the accumulator never
            # resets.
            if total is not None and math.isfinite(total):
                count = int(round(total))
        artifact = self._preview.descriptor()
        return {
            "count": count,
            "count_unavailable_reason": (
                None
                if count is not None
                else "this experiment reports no annotation count"
            ),
            "artifact": artifact,
            # Mutually exclusive with `artifact`, and never both null. A
            # client that finds neither has met a Tower that is broken
            # rather than one that is being quiet.
            "artifact_unavailable_reason": (
                None if artifact is not None else self._preview.why_none()
            ),
        }
