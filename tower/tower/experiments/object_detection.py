"""What objects are in view, as a measurement rather than a product.

This exists for one concrete reason: a Scene Understanding cartridge
needs to choose a detector, and the module doc's promotion path says
measure in the Lab before promoting. COCO gives `person`, `chair` and
`dining table` -- literally the example questions that cartridge is meant
to answer -- so this is the measurement that decision needs.

`ssdlite320_mobilenet_v3_large` from torchvision, which is already an
installed dependency: 13.4 MB of weights, ~31 ms per frame on this CPU at
640x360. No new dependency earns its cost here.

**Output is model inference, not measured fact** (Rule 16, Core Principle
2). A detection is evidence that something scored above a threshold, and
the score is reported alongside the count so a consumer can see how thin
the evidence is. Nothing here establishes identity, and nothing persists.
"""

import logging

import cv2
import numpy as np

from tower.experiments import (
    DetectionPreview,
    ExperimentPreview,
    ExperimentResult,
    ExperimentSettings,
    MetricKind,
    ScenePreview,
    scene_structure,
)
from tower.instrumentation import StageTimer
from tower.loading import LoadInvalidation
from tower.modules.base import FrameProcessingError

logger = logging.getLogger(__name__)

# Below this a COCO detection from a mobile-class detector is noise more
# often than not. Reported as a metric so the choice stays visible, and
# raw counts at 0.0 are reported too so the threshold can be re-chosen
# from data instead of from this comment.
SCORE_THRESHOLD = 0.4

# Reported individually because a Scene Understanding cartridge asks about
# them by name. Everything else is folded into `detections`.
TRACKED_CLASSES = ("person", "chair", "couch", "dining table", "tv", "laptop")

# The most boxes one preview draws. Every detection at or above the
# threshold is drawn; the remaining slots go to the highest-scoring
# detections BELOW it, faded, because those are the ones somebody
# investigating a wrong answer is looking for. SSD returns up to 300 raw
# boxes and a 320-pixel panel with 300 rectangles on it is a hatch
# pattern.
PREVIEW_MAX_BOXES = 24

# How far below the threshold a detection may be and still be drawn.
#
# Half of it, which is a principled line rather than a taste: a
# NEAR-MISS is something that got at least halfway to being believed,
# and everything under that is the detector saying no. Measured on a
# synthetic desk scene: without this floor the picture drew one accepted
# box and twenty-three refusals scoring 0.02 to 0.03, whose label chips
# covered the room and buried the one detection that mattered. That is
# not showing somebody the near-misses; it is showing them the tail.
PREVIEW_LOW_SCORE_FLOOR = SCORE_THRESHOLD / 2.0

# `score_threshold` is SCORE_THRESHOLD echoed back on every frame: a
# constant, and the old harness AVERAGED it, which happened to give the
# right number for the wrong reason. The per-class entries are derived
# from TRACKED_CLASSES rather than typed out, so a class added above
# cannot arrive unclassified.
METRIC_KINDS: dict[str, MetricKind] = {
    "detections": MetricKind.COUNT,
    "raw_detections": MetricKind.COUNT,
    "score_threshold": MetricKind.CONSTANT,
    "mean_score": MetricKind.RATE,
    "max_score": MetricKind.RATE,
    **{
        f"count_{name.replace(' ', '_')}": MetricKind.COUNT
        for name in TRACKED_CLASSES
    },
}


class ObjectDetectionExperiment:
    name = "object_detection"

    def __init__(self) -> None:
        self._model = None
        self._transform = None
        self._device = None
        self._categories = None
        # What was ASKED for, kept beside what was resolved.
        # `auto` resolving to `cpu` on a machine with no CUDA is a
        # correct outcome; `cuda` resolving to `cpu` would be a
        # silent downgrade, and only both numbers together can
        # tell the two apart after the fact.
        self._requested_device = None
        self._torch_threads: int | None = None
        # Guards the handover from a load that may have been abandoned by
        # the module's load timeout. See tower/loading.py.
        self._invalidation = LoadInvalidation()
        self._preview = ExperimentPreview()

    def _install(self, model, transform, categories, device) -> None:
        """Hand the loaded model to `self`. Runs under the token's lock."""
        self._model = model
        self._transform = transform
        self._categories = categories
        self._device = device

    def load(self, settings: ExperimentSettings | None = None) -> None:
        # Local imports: torch/torchvision are an optional [ml] extra, and
        # nothing outside a detection-selected module may require them.
        import torch
        from torchvision.models.detection import (
            SSDLite320_MobileNet_V3_Large_Weights,
            ssdlite320_mobilenet_v3_large,
        )

        from tower.experiments.depth import (
            apply_torch_threads,
            resolve_device,
            resolve_torch_threads,
        )

        requested = "auto" if settings is None else settings.device
        self._requested_device = requested
        # "auto" prefers CUDA here, the same as everywhere else. It did
        # not until 2026-08-29, and the pin that made it CPU is worth
        # understanding rather than deleting, because it was RIGHT about
        # what it measured.
        #
        # What it measured (2026-08-27, this file's previous comment):
        # an idle developer machine, same-process interleaved A/B, 480
        # timed frames per device, CUDA warmed up first --
        #
        #     object_detection   cpu 29.41 ms   cuda 38.17 ms   CPU faster
        #     depth              cpu 20.03 ms   cuda 10.41 ms   CUDA faster
        #
        # CUDA lost all eight blocks, and the explanation was sound:
        # MobileNetV3 at an internal 320 px is bound by kernel-launch
        # overhead rather than arithmetic, so a GPU has nothing to win.
        #
        # What a real Tower measured. Physical testing on the glasses ->
        # phone -> Tower path, 2026-08-29, reported a mean of 199 ms per
        # frame with a worst frame of 4,483 ms -- seven times the figure
        # above, and a stall long enough that the event loop this runs
        # synchronously on stopped answering every socket for four and a
        # half seconds. Reproduced here under CPU contention: elevated
        # latency persisted for THIRTY consecutive frames and then
        # decayed, which is not the shape of a warm-up (paid once) and is
        # exactly the shape of a shared thread pool losing cores. Under
        # the SAME contention CUDA held 40-50 ms per frame, every frame.
        #
        # The two measurements do not contradict each other. The first
        # was taken with nothing else running; a Tower ships with
        # `scene_autostart` on and `scene_device` = "cpu", so the machine
        # this experiment actually runs on always has another CPU-resident
        # detector on it, and `config.py`'s own `scene_torch_threads`
        # comment already says the torch thread pool is process-global and
        # shared. The idle case is the one that does not happen.
        #
        # So the trade is: give up about 9 ms in a condition that does not
        # occur, to avoid losing 150 ms and multi-second event-loop stalls
        # in the one that does. The outputs are equivalent -- same labels,
        # boxes agreeing to 0.018 px and scores to 0.0004 -- so nothing is
        # bought with correctness.
        #
        # `config.py`'s `scene_device` comment still says CUDA is 30.4 ms
        # against CPU's 32.9 ms for this model. Left standing rather than
        # edited: a THIRD figure, on a different harness, that nothing
        # here re-ran. A successor re-measuring `scene_device` should know
        # all three exist, and should measure under load.
        #
        # `TOWER_CV_DEVICE=cpu` still forces CPU. This changes what
        # "auto" means, not what is reachable.
        device = resolve_device(requested)
        self._torch_threads = resolve_torch_threads(
            device, "auto" if settings is None else settings.torch_threads
        )
        apply_torch_threads(self._torch_threads)
        weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
        model = ssdlite320_mobilenet_v3_large(weights=weights)
        model.eval()
        # Locals until the very last line, then installed through the
        # invalidation token. This runs on a worker thread so that the
        # module's load timeout can actually bound the weight download --
        # and a timeout ABANDONS this thread rather than stopping it, so
        # `release()` may already have run by the time we get here.
        # Assigning `self._model` directly would hand a live model, and on
        # CUDA resident GPU memory, to a FAILED module that will never be
        # released again.
        torch_device = torch.device(device)
        try:
            model.to(torch_device)
            # Built out here rather than inside the publish lambda: they
            # can raise, and raising inside `install` would do so while
            # the token's lock is held, with the model already resident
            # and nothing covering it. The token guards the publish, not
            # the build, so the build guards itself.
            transform = weights.transforms()
            categories = list(weights.meta["categories"])
            if torch_device.type == "cuda":
                # One throwaway inference, here, on the worker thread the
                # module's load timeout already bounds -- rather than on
                # the wearer's first frame, on the event loop.
                #
                # Measured 2026-08-29: the first CUDA inference costs
                # ~500 ms (context creation plus cuDNN algorithm
                # selection) and every one after it costs 40-50 ms. Half a
                # second added to an arm that already downloads weights is
                # invisible; half a second added to the first frame is a
                # stall a person feels and a figure that poisons the run's
                # `processing_ms_max` for as long as the run lasts.
                #
                # CPU shows no comparable one-time cost, so it pays
                # nothing here. Inside the existing `try`, so a warm-up
                # that raises is freed by the same `del model` /
                # `empty_cache()` the rest of this block already does.
                with torch.inference_mode():
                    model(
                        [
                            transform(
                                torch.zeros(3, 360, 640, dtype=torch.uint8)
                            ).to(torch_device)
                        ]
                    )
                torch.cuda.synchronize()
        except BaseException:
            # This frame owns the model and nothing installed it, so this
            # frame frees it -- before the traceback that would otherwise
            # pin it escapes to the container's `except` block, which
            # calls `release()` while the traceback is still live.
            del model
            if torch_device.type == "cuda":
                torch.cuda.empty_cache()
            raise
        if not self._invalidation.publish(
            lambda: self._install(model, transform, categories, torch_device)
        ):
            del model
            if torch_device.type == "cuda":
                torch.cuda.empty_cache()
            logger.warning(
                "[Tower][Module] object detection weights finished loading "
                "after the module was released; discarded"
            )

    def _clear(self) -> None:
        """Forget everything the load installed. Runs under the token's lock."""
        self._model = None
        self._transform = None
        self._categories = None
        self._device = None

    def release(self) -> None:
        self._preview.set_preview_capture(False)
        # The device is read INSIDE the teardown so that one lock covers
        # both the question and the answer. Reading it before invalidating
        # -- as this used to -- is a TOCTOU: an abandoned loader that
        # publishes in between makes `publish()` return True, so the
        # loader skips its own `empty_cache()`, `_clear` drops the CUDA
        # model, and the stale False here means `empty_cache()` runs
        # nowhere at all. Same shape as depth.py's `release()`.
        was_cuda: list[bool] = []

        def _teardown() -> None:
            # Runs under the token's lock, and must not touch the token.
            if self._device is not None and self._device.type == "cuda":
                was_cuda.append(True)
            self._clear()

        # Invalidate and clear as one critical section: clearing first
        # would leave a window in which an abandoned loader installs into
        # a slot that has just been emptied.
        self._invalidation.invalidate(_teardown)
        if was_cuda:
            import torch

            torch.cuda.empty_cache()

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
            "model": "ssdlite320_mobilenet_v3_large",
            "weights": "COCO_V1",
            "score_threshold": SCORE_THRESHOLD,
            "torch_threads": self._torch_threads,
        }

    def set_preview_capture(self, enabled: bool) -> None:
        self._preview.set_preview_capture(enabled)

    def take_preview(self):
        return self._preview.take_preview()

    def _preview_payload(self, image, prediction, scores, labels):
        """Boxes, names and scores, in the line drawing's coordinates.

        The one view that turns a wrong answer into a bug report. Physical
        testing produced 160 `person` detections in a room the wearer
        described as empty, and the numbers alone could not say whether
        that was a broken class map, a broken colour order, or the model
        doing exactly what a COCO detector does when a head-mounted camera
        points at the wearer's own hands. Investigation found the third --
        87% of those boxes touch the bottom edge and sit dead centre --
        and no amount of staring at `count_person: 160` would have shown
        it. A box drawn round a forearm shows it immediately.

        Below-threshold detections are kept, not dropped. They are the
        near-misses, and a viewer that hid them would be answering a
        different question than the one somebody debugging is asking.

        What IS dropped is everything past the highest-scoring
        `PREVIEW_MAX_BOXES`, and the payload carries the true totals so the
        caption can say so. SSD returns up to 300 boxes and a 320-pixel
        panel with 300 rectangles on it is a hatch pattern -- but a picture
        that quietly drew a quarter of them and captioned itself as
        complete would be worse than the hatch pattern.

        `prediction["boxes"]` is in the ORIGINAL frame's coordinates:
        torchvision's `GeneralizedRCNNTransform` records the input size
        and maps its boxes back before returning them, so the 320x320 the
        model resized to internally never leaks out here.
        """
        boxes = prediction["boxes"].detach().cpu().numpy()
        structure = scene_structure(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        scale = structure.shape[1] / float(image.shape[1] or 1)

        # Everything worth looking at, best first: accepted, plus the
        # refusals that came close. `argsort` on the whole array rather
        # than a partition -- SSD returns at most 300 boxes, so this is a
        # sort of 300 floats and the clarity is worth more than the
        # microsecond.
        drawable = np.flatnonzero(scores >= PREVIEW_LOW_SCORE_FLOOR)
        order = drawable[np.argsort(-scores[drawable])][:PREVIEW_MAX_BOXES]
        chosen_boxes = (boxes[order] * scale).astype(np.float32)
        chosen_scores = scores[order].astype(np.float32)
        names = tuple(
            self._categories[int(index)]
            if 0 <= int(index) < len(self._categories)
            else str(int(index))
            for index in labels[order]
        )
        return DetectionPreview(
            scene=ScenePreview(structure=structure),
            boxes=chosen_boxes,
            labels=names,
            scores=chosen_scores,
            threshold=SCORE_THRESHOLD,
            accepted_total=int((scores >= SCORE_THRESHOLD).sum()),
            raw_total=int(len(scores)),
        )

    def run(self, raw_bytes: bytes) -> ExperimentResult:
        timer = StageTimer()

        with timer.stage("decode"):
            array = np.frombuffer(raw_bytes, dtype=np.uint8)
            image = cv2.imdecode(array, cv2.IMREAD_COLOR)
            if image is None:
                raise FrameProcessingError("undecodable frame")
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        import torch

        from tower.experiments.depth import apply_torch_threads

        # On the inference thread, every frame: see `apply_torch_threads`.
        apply_torch_threads(self._torch_threads)

        with timer.stage("preprocess"):
            tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)
            batch = [self._transform(tensor).to(self._device)]

        with timer.stage("inference"):
            with torch.inference_mode():
                outputs = self._model(batch)

        with timer.stage("summarize"):
            prediction = outputs[0]
            scores = prediction["scores"].detach().cpu().numpy()
            labels = prediction["labels"].detach().cpu().numpy()

            kept = scores >= SCORE_THRESHOLD
            kept_scores = scores[kept]
            kept_labels = labels[kept]

            metrics = {
                "detections": float(kept.sum()),
                "raw_detections": float(len(scores)),
                "score_threshold": SCORE_THRESHOLD,
                "mean_score": float(kept_scores.mean()) if kept.any() else 0.0,
                "max_score": float(scores.max()) if len(scores) else 0.0,
            }
            for name in TRACKED_CLASSES:
                index = self._categories.index(name)
                metrics[f"count_{name.replace(' ', '_')}"] = float(
                    (kept_labels == index).sum()
                )

        if self._preview.wanted:
            with timer.stage("preview"):
                self._preview.offer(
                    self._preview_payload(image, prediction, scores, labels)
                )

        return ExperimentResult(
            result_value=metrics["detections"],
            result_label="detections",
            processing_ms=timer.total_ms,
            stage_ms=timer.snapshot(),
            metrics=metrics,
        )
