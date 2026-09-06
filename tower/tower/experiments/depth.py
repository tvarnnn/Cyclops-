import logging
import time

import cv2
import numpy as np

from tower.experiments import (
    ExperimentPreview,
    ExperimentResult,
    ExperimentSettings,
    decode_color,
)
from tower.instrumentation import StageTimer
from tower.loading import LoadInvalidation
from tower.modules.base import FrameProcessingError

logger = logging.getLogger(__name__)


def resolve_device(requested: str) -> str:
    """Turn a requested device into one that actually exists.

    "auto" means CUDA if it is genuinely usable, else CPU. Asking for
    "cuda" on a machine without it is an error rather than a silent
    downgrade -- an unnoticed downgrade turns a GPU benchmark into a CPU
    benchmark with a GPU label on it, which is worse than a failure.
    """
    if requested == "cpu":
        return "cpu"

    import torch

    available = torch.cuda.is_available()
    if requested == "auto":
        return "cuda" if available else "cpu"
    if requested == "cuda" and not available:
        raise RuntimeError("cuda requested but torch reports it is unavailable")
    return requested


# torch's intra-op pool defaults to one thread per logical CPU -- twenty
# on this host -- and Intel OpenMP workers spin-wait for 200 ms after every
# parallel region. At a 40-80 ms frame interval they never sleep, so every
# core that ever joined a team reads busy, which is what "object detection
# pegs the machine at 99% CPU" was. Measured on this host on 2026-09-06
# (20 logical CPUs, RTX 5070, torch 2.13.0+cu132, 40 timed frames a row):
#
#     object_detection  CUDA  threads 20 / 4 / 2 / 1
#                             cores   12.6 / 3.6 / 1.8 / 0.9
#                             p50 ms  43.6 / 40.0 / 40.1 / 40.0
#     depth             CUDA  threads 20 / 2 / 1
#                             cores    6.2 / 1.1 / 0.9
#                             p50 ms  11.8 / 11.7 / 16.3
#     object_detection  CPU   threads 20 / 4      cores 13.1 / 3.8   p50 46.6 / 35.9
#     depth             CPU   threads 20 / 8 / 4  cores 12.5 / 7.7 / 3.9  p50 42.3 / 34.1 / 48.2
#
# On CUDA the CPU work is preprocessing, and two threads cost nothing in
# latency while returning ten cores. On CPU, four is object detection's
# best and within 15% of depth's best (eight) at half the cores; a
# machine that also runs the World Builder's followers wants the cores
# more than depth wants the 8 ms.
TORCH_THREADS_CUDA = 2
TORCH_THREADS_CPU = 4


def resolve_torch_threads(device: str, requested="auto") -> int | None:
    """The intra-op budget to apply, or None to leave torch's default alone.

    `requested` is `ExperimentSettings.torch_threads`: "auto", a
    non-negative integer, or the string of one. Garbage raises rather than
    silently becoming a default -- `config.py` already sanitises the
    environment variable, so anything that reaches here came from code.
    """
    if requested is None:
        return None
    if isinstance(requested, str):
        text = requested.strip().lower()
        if text == "auto":
            return (
                TORCH_THREADS_CUDA
                if str(device).startswith("cuda")
                else TORCH_THREADS_CPU
            )
        if not text.isdigit():
            raise ValueError(f"torch_threads must be 'auto' or an integer, not {requested!r}")
        requested = int(text)
    if isinstance(requested, bool) or not isinstance(requested, int) or requested < 0:
        raise ValueError(f"torch_threads must be 'auto' or a non-negative integer, not {requested!r}")
    return requested or None


def apply_torch_threads(threads: int | None) -> None:
    """Apply the budget on THIS thread.

    OpenMP keeps its thread budget per calling thread, so a cap set on the
    loader thread does not reach the event loop thread that runs
    inference. Experiments therefore call this twice: once in `load()`,
    on the loader, so that building the model does not spin twenty
    workers; and on every `run()`, on whichever thread runs frames, where
    it costs one `omp_get_max_threads()` read when nothing has to change.
    """
    if threads is None:
        return
    import torch

    if torch.get_num_threads() != threads:
        torch.set_num_threads(threads)


class DepthEstimation:
    """Stateful monocular depth estimation (MiDaS-small).

    Output is relative (inverse) depth, not metric distance -- MiDaS-
    small does not produce metric output. result_label says "relative"
    explicitly; treat as model inference, not measurement, per
    07-PLATFORM-CONSTRAINTS.md Limitation 1 / Core Principle 2.
    """

    name = "depth"

    def __init__(self, capture_depth_array: bool = False) -> None:
        self._model = None
        self._transform = None
        self._device = None
        # What was ASKED for, kept beside what was resolved.
        # `auto` resolving to `cpu` on a machine with no CUDA is a
        # correct outcome; `cuda` resolving to `cpu` would be a
        # silent downgrade, and only both numbers together can
        # tell the two apart after the fact.
        self._requested_device = None
        self._torch_threads: int | None = None
        # Opt-in, bounded observability hook for offline research analysis
        # (World Builder Experiment 1, depth_temporal_consistency): the wire
        # protocol only carries the scalar mean, but temporal-stability
        # analysis needs the full per-frame array. Off by default so the
        # serving path is unchanged; holds only the most recent frame, never
        # a growing list, so enabling it cannot grow without bound.
        self.capture_depth_array = capture_depth_array
        self.last_depth_array = None
        # The live preview's slot, and deliberately NOT the same one.
        # `capture_depth_array` is an offline-research hook that a corpus
        # harness turns on for the length of an analysis; this is turned
        # on and off by the Lab as a person starts and stops watching. If
        # they shared a flag, a phone opening the viewer would silently
        # switch on a research capture, and closing it would switch off
        # somebody's analysis. They share the array itself -- one object,
        # two references -- which is the part worth sharing.
        self._preview = ExperimentPreview()
        # Guards the handover from a load that may have been abandoned.
        # See tower/loading.py for the ordering bug this exists for.
        self._invalidation = LoadInvalidation()

    def _install(self, model, transform, device) -> None:
        """Hand the loaded model to `self`. Runs under the token's lock."""
        self._model = model
        self._transform = transform
        self._device = device

    def load(self, settings: ExperimentSettings | None = None) -> None:
        requested = "auto" if settings is None else settings.device
        self._requested_device = requested
        device = resolve_device(requested)
        self._torch_threads = resolve_torch_threads(
            device, "auto" if settings is None else settings.torch_threads
        )
        apply_torch_threads(self._torch_threads)
        import torch  # local import: torch is an optional [ml] extra;
        # nothing outside a depth-selected module may require it.

        start = time.perf_counter()
        # Built into LOCALS, installed onto `self` only at the end and
        # only through the invalidation token. This load runs on a worker
        # thread now (see ExperimentalCVModule._do_load), and a timeout
        # abandons that thread rather than stopping it: by the time these
        # lines run, `release()` may already have happened. Assigning
        # `self._model` directly -- as this method used to -- installs a
        # live model, and on CUDA resident GPU memory, into a FAILED
        # module that nothing will ever release again.
        torch_device = torch.device(device)
        # Pinned: floating on the default branch is a reproducibility risk
        # for a measured baseline, not theoretical -- see the spec's
        # 2026-08-20 Amendment for how this was discovered.
        midas_ref = "intel-isl/MiDaS:454597711a62eabcbf7d1e89f3fb9f569051ac9b"
        model = torch.hub.load(midas_ref, "MiDaS_small", trust_repo=True)
        try:
            model.to(torch_device)
            model.eval()
            # A SECOND hub.load, after the weights are already resident.
            # It can raise -- a corrupt hub cache, a transient network
            # error on a fresh clone -- and the invalidation token guards
            # only the publish, so nothing else covers this window.
            # Without the `except`, `release()` sees `_device is None`,
            # skips `empty_cache()`, and the model stays alive anyway:
            # the container catches this exception and calls
            # `mark_failed()` from inside the `except` block, so the live
            # traceback still holds this frame and its `model` local.
            # Measured with a weakref: alive during `release()`.
            transform = torch.hub.load(
                midas_ref, "transforms", trust_repo=True
            ).small_transform
        except BaseException:
            # This frame owns the model and nothing installed it, so this
            # frame frees it -- before the traceback that would otherwise
            # pin it escapes.
            del model
            if torch_device.type == "cuda":
                torch.cuda.empty_cache()
            raise
        load_ms = (time.perf_counter() - start) * 1000

        if not self._invalidation.publish(
            lambda: self._install(model, transform, torch_device)
        ):
            # Abandoned mid-load. This thread holds the only reference
            # left, so this thread frees it -- nobody is coming back.
            del model, transform
            if torch_device.type == "cuda":
                torch.cuda.empty_cache()
            logger.warning(
                "[Tower][Module] depth model finished loading after %.1fms "
                "but the module had already been released; discarded",
                load_ms,
            )
            return

        # `torch_device`, not `self._device`: a release racing this log
        # line would leave the attribute None, and a crash while logging a
        # success is a silly way to fail a load that worked.
        if torch_device.type == "cuda":
            allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            logger.info(
                "[Tower][Module] depth model loaded on %s in %.1fms "
                "(torch %s, cuda runtime %s, %.1fMB allocated)",
                torch_device,
                load_ms,
                torch.__version__,
                torch.version.cuda,
                allocated_mb,
            )
        else:
            logger.info(
                "[Tower][Module] depth model loaded on %s in %.1fms (torch %s)",
                torch_device,
                load_ms,
                torch.__version__,
            )

    def _clear(self) -> None:
        """Forget everything the load installed. Runs under the token's lock."""
        self._model = None
        self._transform = None
        self._device = None
        self.last_depth_array = None
        self._preview.set_preview_capture(False)

    def release(self) -> None:
        # `self._device` is read INSIDE the teardown, so the same lock
        # covers the question and the answer. Reading it out here first
        # -- as this used to -- is a TOCTOU: an abandoned loader that
        # publishes between the read and the invalidation makes
        # `publish()` return True, so the loader skips its own
        # `empty_cache()`; `_clear` then drops that CUDA model; and
        # `is_cuda` is still the stale False, so `empty_cache()` runs
        # NOWHERE and the freed blocks stay in torch's caching allocator.
        freed: dict[str, float] = {}

        def _teardown() -> None:
            # Runs under the token's lock, and must not touch the token.
            if self._device is not None and self._device.type == "cuda":
                import torch

                freed["peak_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
            self._clear()

        # Invalidate and clear together, under the token's lock. Clearing
        # first and invalidating afterwards would leave the exact window
        # this is here to close: a loader that passed the check installs
        # into a slot that has just been emptied.
        self._invalidation.invalidate(_teardown)

        # Outside the lock on purpose: `empty_cache()` synchronises with
        # the device, and no correctness depends on it holding the lock.
        if freed:
            import torch

            torch.cuda.empty_cache()
            logger.info(
                "[Tower][Module] depth experiment released; peak cuda allocation %.1fMB",
                freed["peak_mb"],
            )

    def describe(self) -> dict:
        """Runtime facts about what is actually loaded. Never raises.

        OPTIONAL on the `Experiment` protocol, and the Lab treats a
        missing `describe()` as "this experiment holds nothing worth
        reporting" rather than as an error. It exists because
        `TOWER_CV_DEVICE=auto` is a REQUEST and `resolve_device` decides
        the answer: a run that says "auto" has not told anyone whether it
        used the GPU, and a CPU figure with a GPU label on it is the
        specific failure `resolve_device` was written to prevent.

        Read outside the invalidation lock on purpose. This is a
        diagnostic, and taking the lock that guards a model handover in
        order to print a device name would let a status read contend with
        a load. A single attribute read is atomic; the worst case is a
        `null` device on a run that was mid-load a microsecond ago.
        """
        device = self._device
        return {
            "backend": "torch",
            "device": "unknown" if device is None else str(device),
            "device_requested": self._requested_device or "auto",
            "model": "MiDaS_small",
            "output": "relative inverse depth, not metric distance",
            "torch_threads": self._torch_threads,
        }

    def set_preview_capture(self, enabled: bool) -> None:
        """Whether to keep this frame's depth array for the live viewer.

        Off until the Lab says otherwise. See `ExperimentPreview`.
        """
        self._preview.set_preview_capture(enabled)

    def take_preview(self):
        """The newest depth array, if one was kept. Relative, not metric.

        Whatever renders this is responsible for saying so. The array
        carries no unit and cannot be given one: MiDaS-small produces
        inverse depth on an arbitrary per-frame scale, so even the same
        wall photographed twice does not produce the same numbers.
        """
        return self._preview.take_preview()

    def run(self, raw_bytes: bytes) -> ExperimentResult:
        timer = StageTimer()

        with timer.stage("decode"):
            image_rgb = cv2.cvtColor(decode_color(raw_bytes), cv2.COLOR_BGR2RGB)

        import torch  # deferred: only the stages below need it, so an
        # undecodable frame never requires torch to be installed at all.

        # On the inference thread, every frame: see `apply_torch_threads`.
        apply_torch_threads(self._torch_threads)

        with timer.stage("preprocess"):
            try:
                input_tensor = self._transform(image_rgb).to(self._device)
            except cv2.error as exc:
                # MiDaS's transform resizes, and its internal cv2.resize
                # asserts on a non-positive scale for an extreme aspect
                # ratio (measured: 1x1000 and 1000x1 both fail; 1x1 and
                # 16x16 are fine, so this is about the RATIO, not size).
                # Without this the bare cv2.error marks the whole module
                # FAILED, terminally, for the life of the process.
                raise FrameProcessingError(
                    f"frame {image_rgb.shape[1]}x{image_rgb.shape[0]} could "
                    f"not be prepared for the depth model: {exc}"
                ) from exc

        with timer.stage("inference"):
            with torch.inference_mode():
                prediction = self._model(input_tensor)

        with timer.stage("postprocess"):
            depth = prediction.squeeze().detach().cpu().numpy()
            mean_relative_depth = float(depth.mean())
            if self.capture_depth_array:
                self.last_depth_array = depth
        # Outside the timed stages, like `edge_detection`'s. `stage_ms`
        # is what the depth model costs, and a physical baseline of
        # `inference: 9.399 ms` must keep meaning the same thing after
        # this file learned to hand the array on.
        #
        # `depth` is handed over by reference and nothing downstream
        # mutates it. The renderer resizes, which allocates; it never
        # writes into what it was given.
        self._preview.offer(depth)

        return ExperimentResult(
            result_value=mean_relative_depth,
            result_label="mean_relative_depth",
            processing_ms=timer.total_ms,
            stage_ms=timer.snapshot(),
            metrics={
                "mean_relative_depth": mean_relative_depth,
                "min_relative_depth": float(depth.min()),
                "max_relative_depth": float(depth.max()),
                "std_relative_depth": float(depth.std()),
            },
        )
