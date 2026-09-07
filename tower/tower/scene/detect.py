"""This cartridge's detectors: one for the GPU, one for the CPU, chosen by device.

Two detectors, because the device decides what is affordable and the
measurement decides what is worth it (`docs/superpowers/research/
2026-09-07-scene-understanding-architecture.md`, on 700 human-labelled
COCO val2017 images resized to this camera's 640 px long side, and 300
real corpus frames on a quiet RTX 5070):

                          mAP50   person AP50   chair AP50   people count   CUDA ms   CPU ms   VRAM
                                                             exact / MAE
    ssdlite320 @0.4       0.367   0.604         0.199        0.659 / 1.15   29.5      43.5     56 MB
    rtdetr_v2_r18 @0.5    0.669   0.856         0.540        0.733 / 0.48   17.3     226       298 MB
    lwdetr_small* @0.4    0.689   0.872         0.606        0.783 / 0.42   18.2       -       138 MB
    dfine_s_obj2coco* @0.5 0.702  0.876         0.596        0.729 / 0.48   28.3     148       152 MB
    dfine_m_obj2coco* @0.5 0.737  0.895         0.640          -            35.4       -       210 MB

    * speed needs fp16 autocast on CUDA. In fp32 D-FINE-S (the plain
      coco checkpoint) measured 121 ms and the obj2coco checkpoint
      110 ms under contention; LW-DETR-small 51 ms under contention.
      "people count" is per-image exact rate / mean error over all
      labelled persons, at the threshold shown.

**On CUDA, RT-DETRv2-R18.** Nearly twice the baseline's accuracy at
60% of its latency, in plain fp32, from the model's own authors' hub
organisation (PekingU, Apache-2.0), through the `transformers` package
this repository already carries for Object Memory's verifier.

**What it is NOT the best at, and why it is the default anyway.**
LW-DETR-small counts people better on the labelled stills (exact 0.783
against 0.733) and is two mAP50 points ahead at the same latency, and
D-FINE-S/M are ahead on mAP50. Both reach that speed only under fp16
autocast -- in fp32 they are three to six times slower -- and LW-DETR's
hub weights are a contributor's conversion (Apache-2.0 on the card;
upstream Apache-2.0) whose lineage to the authors' checkpoints the card
does not state, while D-FINE's carry a documented conversion caveat
(transformers issue #40253). A default that depends on an autocast
path and on weights whose provenance is unstated is a worse default
than one that runs in plain fp32 from the authors' own organisation,
for a five-point difference measured on stills of a different camera.
Both are selectable with `TOWER_SCENE_DETECTOR`, each at its own best
threshold, and a physical validation that favours one is what should
move the default.

**On CPU, SSDLite320 stays.** At 43 ms it is the only candidate that
keeps up with a 12 fps feed on a CPU; RT-DETRv2-R18 is 226 ms there and
D-FINE-N 66 ms at four times the cores. A CPU Tower gets the light
detector, the same tracker, and the wire says which detector produced
the counts.

**Weights are fetched from the Hugging Face hub on first load** (~80 MB
for RT-DETRv2-R18) and cached under the user's HF cache, exactly as
Object Memory's verifier does. A Tower without network access on first
run fails to load, reports `state: "failed"` with the reason, and can be
pointed at the light detector with `TOWER_SCENE_DETECTOR=ssdlite320`.

**Still not the Experimental CV Lab**, and a test still enforces it: the
Lab's `ExperimentResult` is scalars and cannot carry a box, and nothing
that may be thrown away belongs upstream of a production consumer.

Everything here is **model inference, not measured fact**
(`07-PLATFORM-CONSTRAINTS.md` Core Principle 2). A detection is evidence
that something scored above a threshold.
"""

import logging

from tower.detection import SCORE_THRESHOLD, Detector, FixedDetector, SSDLite320Detector
from tower.scene.records import BoundingBox, Detection

logger = logging.getLogger(__name__)

# Re-exported so this cartridge's callers keep importing from this
# cartridge, and so a driver never has to know which names came from the
# platform and which were written here.
__all__ = [
    "SCORE_THRESHOLD",
    "CLASSES_OF_INTEREST",
    "DETECTOR_CHOICES",
    "Detector",
    "FixedDetector",
    "TorchvisionDetector",
    "TransformersDetector",
    "detector_for",
    "resolve_choice",
    "score_threshold_for",
    "to_scene_detection",
]

# The classes this cartridge reports. Not all 80: the brief's questions
# are about people and furniture, and a scene state cluttered with every
# COCO class would bury them. Adding one is a one-line change.
#
# Narrowing HERE, at the detector, rather than downstream: this cartridge
# answers "what is around me now" and everything it keeps is tracked and
# rendered immediately. Object Memory does the opposite and keeps every
# class, because a memory cannot pre-judge what will matter later.
#
# What the real domain contains, from the 2026-09-07 audit of all 97
# corpus captures: laptop, tv/monitor, cell phone, keyboard and mouse in
# most desk captures; chair, bed, couch, dining table, cup and bottle
# occasionally; book never; person never (no bystander in 45,594
# frames). Every class here is one the detector was measured on.
CLASSES_OF_INTEREST = (
    "person",
    "chair",
    "couch",
    "bed",
    "dining table",
    "tv",
    "laptop",
    "book",
    "bottle",
    "cup",
    "keyboard",
    "mouse",
    "cell phone",
)

# Hub label spellings that differ from torchvision's COCO names.
_LABEL_ALIASES = {
    "sofa": "couch",
    "tvmonitor": "tv",
    "diningtable": "dining table",
    "cellphone": "cell phone",
}

# What `TOWER_SCENE_DETECTOR` may name, and what each is.
DETECTOR_CHOICES = {
    "auto": "rtdetr_v2_r18 on CUDA, ssdlite320 on CPU",
    "ssdlite320": "torchvision SSDLite320 MobileNetV3, COCO (BSD-3)",
    "rtdetr_v2_r18": "PekingU/rtdetr_v2_r18vd (Apache-2.0), fp32",
    "rtdetr_v2_r34": "PekingU/rtdetr_v2_r34vd (Apache-2.0), fp32",
    "dfine_s": "ustc-community/dfine-small-obj2coco (Apache-2.0), fp16 on CUDA",
    "dfine_m": "ustc-community/dfine-medium-obj2coco (Apache-2.0), fp16 on CUDA",
    "lwdetr_small": "stevenbucaille/lwdetr_small_60e_coco (Apache-2.0, contributor conversion), fp16 on CUDA",
}

# repo, transformers class, wants fp16 on CUDA, counting threshold.
# Thresholds are each model's best per-image people-count operating
# point on the 700 labelled images (see the table above).
_HUB_MODELS = {
    "rtdetr_v2_r18": ("PekingU/rtdetr_v2_r18vd", "RTDetrV2ForObjectDetection", False, 0.5),
    "rtdetr_v2_r34": ("PekingU/rtdetr_v2_r34vd", "RTDetrV2ForObjectDetection", False, 0.5),
    "dfine_s": ("ustc-community/dfine-small-obj2coco", "DFineForObjectDetection", True, 0.5),
    "dfine_m": ("ustc-community/dfine-medium-obj2coco", "DFineForObjectDetection", True, 0.5),
    "lwdetr_small": ("stevenbucaille/lwdetr_small_60e_coco", "LwDetrForObjectDetection", True, 0.4),
}


def to_scene_detection(detection) -> Detection:
    """A platform detection, in this cartridge's vocabulary.

    A free function so the conversion is testable on its own, without
    weights: it is the only thing the adapter below actually does, and an
    untested conversion between two box conventions is exactly where an
    x/y transposition hides.
    """
    return Detection(
        label=detection.label,
        score=detection.score,
        box=BoundingBox(*detection.box),
    )


class TorchvisionDetector:
    """The shared SSDLite detector, reporting this cartridge's Detection.

    Composition rather than a subclass, deliberately: `detect` returns a
    different type from the one the shared class returns, and a subclass
    that changes its parent's return type is a substitution bug waiting
    for the first caller who holds the base type. Wrapping says what is
    true -- this is a scene-shaped view of a platform detector.
    """

    name = "ssdlite320"

    def __init__(
        self,
        score_threshold: float = SCORE_THRESHOLD,
        classes=CLASSES_OF_INTEREST,
        device: str = "cpu",
    ) -> None:
        self.score_threshold = score_threshold
        self._inner = SSDLite320Detector(
            score_threshold=score_threshold,
            classes=classes,
            device=device,
            owner="Scene",
        )

    def load(self) -> None:
        self._inner.load()

    def detect(self, frame_bgr) -> list[Detection]:
        return [
            to_scene_detection(detection)
            for detection in self._inner.detect(frame_bgr)
        ]

    def release(self) -> None:
        self._inner.release()


class TransformersDetector:
    """A hub detector (RT-DETRv2 or D-FINE) through `transformers`.

    torch and transformers are imported inside methods, never at module
    load, so a Tower without them still imports this cartridge and can
    still run the light detector. One instance owns one model; nothing
    here is shared or cached across sessions, so releasing one cannot
    empty another's.
    """

    def __init__(
        self,
        choice: str,
        *,
        score_threshold: float = SCORE_THRESHOLD,
        classes=CLASSES_OF_INTEREST,
        device: str = "cuda",
    ) -> None:
        if choice not in _HUB_MODELS:
            raise ValueError(f"unknown hub detector {choice!r}")
        self.name = choice
        self._repo, self._class_name, self._wants_fp16, _ = _HUB_MODELS[choice]
        self.score_threshold = score_threshold
        self._score_threshold = score_threshold
        self._classes = set(classes) if classes else None
        self._device_name = device
        self._model = None
        self._processor = None
        self._device = None
        self._id2label = None
        self._fp16 = False

    def load(self) -> None:
        import torch
        import transformers

        model_class = getattr(transformers, self._class_name)
        self._processor = transformers.AutoImageProcessor.from_pretrained(self._repo)
        model = model_class.from_pretrained(self._repo)
        model.eval()
        self._device = torch.device(self._device_name)
        model.to(self._device)
        self._model = model
        self._id2label = {
            int(index): _LABEL_ALIASES.get(label, label)
            for index, label in model.config.id2label.items()
        }
        self._fp16 = bool(self._wants_fp16 and self._device.type == "cuda")
        logger.info(
            "[Tower][Scene] detector %s loaded on %s (fp16=%s, torch %s)",
            self.name,
            self._device,
            self._fp16,
            torch.__version__,
        )

    def detect(self, frame_bgr) -> list[Detection]:
        import numpy as np
        import torch

        if self._model is None:
            self.load()
        height, width = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        inputs = self._processor(images=rgb, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with torch.inference_mode():
            if self._fp16:
                with torch.autocast("cuda", dtype=torch.float16):
                    outputs = self._model(**inputs)
            else:
                outputs = self._model(**inputs)
        # Post-process in fp32 whatever the forward ran in.
        if outputs.logits.dtype != torch.float32:
            outputs.logits = outputs.logits.float()
            outputs.pred_boxes = outputs.pred_boxes.float()
        results = self._processor.post_process_object_detection(
            outputs,
            threshold=self._score_threshold,
            target_sizes=torch.tensor([[height, width]], device=self._device),
        )[0]
        detections = []
        for box, score, label_index in zip(
            results["boxes"].cpu().numpy(),
            results["scores"].cpu().numpy(),
            results["labels"].cpu().numpy(),
        ):
            label = self._id2label.get(int(label_index))
            if label is None or (self._classes is not None and label not in self._classes):
                continue
            detections.append(
                Detection(
                    label=label,
                    score=float(score),
                    box=BoundingBox(*(float(value) for value in box)),
                )
            )
        return detections

    def release(self) -> None:
        was_cuda = self._device is not None and self._device.type == "cuda"
        self._model = None
        self._processor = None
        self._device = None
        self._id2label = None
        if was_cuda:
            import torch

            torch.cuda.empty_cache()


# Where each detector's scores are worth believing, for COUNTING. The
# platform's 0.4 was set for SSDLite and stays right for it. The
# transformer detectors score more generously, and on the 700 labelled
# images RT-DETRv2-R18 counts people exactly on 73% of images at 0.5
# against 64% at 0.4, with the mean count error halving (1.04 -> 0.48);
# precision over the 13 classes goes 0.55 -> 0.70 for a recall cost of
# 0.70 -> 0.61. A count that is right more often beats a count that is
# rarely an undercount, because the wire already says every count is a
# floor.
HUB_SCORE_THRESHOLD = 0.5


def score_threshold_for(choice: str) -> float:
    if choice == "ssdlite320":
        return SCORE_THRESHOLD
    return _HUB_MODELS[choice][3] if choice in _HUB_MODELS else HUB_SCORE_THRESHOLD


def resolve_choice(device: str, choice: str = "auto") -> str:
    """"auto" resolved for this device; a named choice checked and kept."""
    if choice not in DETECTOR_CHOICES:
        raise ValueError(
            f"unknown scene detector {choice!r}; one of {sorted(DETECTOR_CHOICES)}"
        )
    if choice == "auto":
        return "rtdetr_v2_r18" if device.startswith("cuda") else "ssdlite320"
    return choice


def detector_for(device: str, choice: str = "auto", *, score_threshold: float | None = None):
    """The detector this device should run, or the one that was asked for.

    `device` is already resolved ("cuda" or "cpu"). "auto" picks by
    device; a named choice is honoured on either device, so an operator
    can measure a hub detector on CPU if they want to, and can pin the
    light one on CUDA. The threshold defaults per detector
    (`score_threshold_for`).
    """
    choice = resolve_choice(device, choice)
    threshold = score_threshold_for(choice) if score_threshold is None else score_threshold
    if choice == "ssdlite320":
        return TorchvisionDetector(score_threshold=threshold, device=device)
    return TransformersDetector(choice, score_threshold=threshold, device=device)
