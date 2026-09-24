"""Transient masks: the wearer's hands, arms and held phone, found by detectors.

Contracts: `docs/contracts/WORLD-BUILDER-APPEARANCE.md` §5.3a (what the mask is
and how appearance uses it), `WORLD-BUILDER-SURFACE.md` §2 (fusion gives it zero
weight), `WORLD-BUILDER-DENSE.md` §3a (the cache files beside the depth work).

WHY A DETECTOR. The appearance stage's own occluder tests are a depth test and a
photometric vote. Neither recognises a hand, and on the canonical capture they
left the wearer's hands on the desk in ki 308, 330 and 351. The fix-it hands
lane (`Glasses-scratch/wb-final-recon/fixit/hands/HANDS.md`) labelled 62 frames
and measured every open-weight candidate it could run; the recommendation this
module implements scored 97.7% hand/arm pixel recall, 99.7% held-phone recall
and 3.1% of static pixels flagged, against 44-49% / 1-6% / 13-15% for the
geometry rules.

THE RECIPE (HANDS.md §6, as measured, `final_eval.py`):

    per keyframe, on the gamma-2-lifted, undistorted, REDACTED keyframe:
      gdsam      Grounding DINO-base boxes for "hand. arm. sleeve. mobile phone."
                 (box/text 0.2; hand|arm|sleeve >= 0.30, phone >= 0.25, boxes
                 >= 4 px), turned into masks by SAM 2.1 hiera base-plus box
                 prompts; a mask over 60% of the image, or more than 70% inside
                 redaction fill, is dropped (the detector reads fill boxes as
                 hands).
      oneformer  OneFormer Swin-L COCO panoptic at 0.5: `person` -> hand,
                 `cell phone` -> phone.
    compose, CPU, per component: a phone counts only when a connected component
    of it touches the hand mask dilated 12 px (a phone resting on the desk is
    scene; masking every phone deleted it from 19-90% of its frames);
    then the union over components, dilated by a 12 px ellipse at 359 px width
    (scaled with width).

MODES. `union` is the recipe above. `oneformer` is the cheaper mode (one model,
~0.35 s a keyframe; 97.5% / 96.4% / 3.05% on the same labels) and is what the
live preset uses. `off` computes nothing. Every mask records the mode and the
rule that made it.

PIXELS COME FROM ONE PLACE. Masks are computed only on what
`appearance.keyframe_source` returns: the session's redacted keyframe,
undistorted with the solve's own maps, with its unobserved (fill) mask. This
module never opens an image itself. The one exception is opt-in: when the
final solve ran with `TOWER_WORLD_SOLVE_MASKS`, it computed these masks on its
own images first (`solve_masks.py`), and a keyframe's mask is copied from that
cache, marked `origin: solve`, instead of being computed twice.

WHAT IS CACHED, AND WHY IN THE DEPTH WORK. One file per keyframe per component,
`dense/<sid>/work/depth/<ki:05d>_transient.<component>.npz`, holding the
component's hand and phone masks (bit-packed) and the key they were made under:
the keyframe id, the SHA-1 of its stored JPEG, the effective redaction label
(the redacted bytes are a function of those two), the fill and unobserved
rules, the model ids at pinned revisions, and the component's own parameters.
Composition (phone rule, dilation) happens when a mask is read, so a change to
it costs no inference. The depth work is the right home: `_fill.npy`, which the
mask is computed against, lives there under the same `ki` naming; both
consumers (surface fusion, then appearance) already read that directory; and
surface fusion runs BEFORE appearance, so an appearance-owned path would invert
the order of ownership. The price: `prune_intermediates` removes these with the
rest of the work, so a rebuild after a prune pays for inference again, exactly
as it pays for depth.

GPU. Models are loaded one at a time, run over every keyframe that needs them,
and freed (`del`, `torch.cuda.empty_cache()`) before the next is loaded: at
most one checkpoint is on the card (measured <= 2.6 GB allocated each).

UNAVAILABLE. Missing packages, weights that are neither cached nor
downloadable, or `TOWER_WORLD_TRANSIENTS=off` produce `state: unavailable` with
the reason, logged once per process per reason. Nothing is claimed masked: the
consumers record `transients: unavailable` and proceed with their other masks.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

TRANSIENT_SCHEMA = 1

MODE_UNION = "union"
MODE_ONEFORMER = "oneformer"
MODE_OFF = "off"
MODES = (MODE_UNION, MODE_ONEFORMER, MODE_OFF)

COMPONENT_GDSAM = "gdsam"
COMPONENT_ONEFORMER = "oneformer"
MODE_COMPONENTS = {
    MODE_UNION: (COMPONENT_GDSAM, COMPONENT_ONEFORMER),
    MODE_ONEFORMER: (COMPONENT_ONEFORMER,),
    MODE_OFF: (),
}

STATE_OK = "ok"
STATE_UNAVAILABLE = "unavailable"
STATE_OFF = "off"
STATE_STOPPED = "stopped"
STATE_FAILED = "failed"

ENV_SWITCH = "TOWER_WORLD_TRANSIENTS"

# Checkpoints, pinned to the revisions the hands lane measured. A revision is a
# cache key: a new one recomputes every mask, which is the point of pinning.
GDINO_MODEL = ("IDEA-Research/grounding-dino-base", "12bdfa3120f3e7ec7b434d90674b3396eccf88eb")
SAM_MODEL = ("facebook/sam2.1-hiera-base-plus", "b7320756a13354e7530a63935656d35b2f91a290")
ONEFORMER_MODEL = ("shi-labs/oneformer_coco_swin_large", "3a263017ca5c75adbea145f25f81b118243d4394")
# OneFormer's processor reads its class table from this dataset repo.
ONEFORMER_CLASS_INFO = ("shi-labs/oneformer_demo", "4d683bd5bf84e9c8b5537dce306230bde409fe89",
                        "coco_panoptic.json")
# Exactly the files each checkpoint loads from: the repos also hold duplicate
# weights in other formats (Grounding DINO's pytorch_model.bin, SAM's .pt,
# OneFormer's original .pth), about 2.2 GB nothing reads.
MODEL_FILES = {
    GDINO_MODEL[0]: ["config.json", "preprocessor_config.json", "special_tokens_map.json",
                     "tokenizer.json", "tokenizer_config.json", "vocab.txt", "model.safetensors"],
    SAM_MODEL[0]: ["config.json", "preprocessor_config.json", "processor_config.json",
                   "video_preprocessor_config.json", "model.safetensors"],
    ONEFORMER_MODEL[0]: ["config.json", "preprocessor_config.json", "special_tokens_map.json",
                         "tokenizer_config.json", "vocab.json", "merges.txt", "pytorch_model.bin"],
}
COMPONENT_MODELS = {
    COMPONENT_GDSAM: (GDINO_MODEL, SAM_MODEL),
    COMPONENT_ONEFORMER: (ONEFORMER_MODEL,),
}
# What a first run downloads, for the unavailable message.
COMPONENT_DOWNLOAD_HINT = {
    COMPONENT_GDSAM: "about 0.9 GB for Grounding DINO-base and 0.3 GB for SAM 2.1 base-plus",
    COMPONENT_ONEFORMER: "about 0.85 GB for OneFormer Swin-L COCO",
}

HAND_WORDS = ("hand", "arm", "sleeve")


class TransientDetectorUnavailable(RuntimeError):
    """This machine cannot run a transient detector: a package or its weights
    are missing and cannot be fetched."""


@dataclass(frozen=True)
class TransientParams:
    mode: str = MODE_UNION
    # detection input: I' = 255 (I/255)^(1/gamma); the frames are dark
    gamma: float = 2.0
    # gdsam
    gdino_prompt: str = "hand. arm. sleeve. mobile phone."
    gdino_box_threshold: float = 0.2
    gdino_text_threshold: float = 0.2
    hand_score: float = 0.30
    phone_score: float = 0.25
    min_box_px: int = 4
    max_mask_frac: float = 0.6
    fill_drop_frac: float = 0.7
    # oneformer
    oneformer_threshold: float = 0.5
    # compose
    phone_near_px: int = 12
    dilate_px: int = 12
    reference_width: int = 359

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"unknown transient detector mode {self.mode!r}; one of {MODES}")

    @classmethod
    def live(cls, **overrides) -> "TransientParams":
        """OneFormer only: the union costs about twice as much a keyframe."""
        base = dict(mode=MODE_ONEFORMER)
        base.update(overrides)
        return cls(**base)

    @property
    def components(self) -> tuple:
        return MODE_COMPONENTS[self.mode]

    def component_params(self, component: str) -> dict:
        """Everything that changes a component's CACHED output."""
        if component == COMPONENT_GDSAM:
            return {"gamma": self.gamma, "prompt": self.gdino_prompt,
                    "box": self.gdino_box_threshold, "text": self.gdino_text_threshold,
                    "hand": self.hand_score, "phone": self.phone_score,
                    "min_box_px": self.min_box_px, "max_mask_frac": self.max_mask_frac,
                    "fill_drop_frac": self.fill_drop_frac}
        if component == COMPONENT_ONEFORMER:
            return {"gamma": self.gamma, "threshold": self.oneformer_threshold}
        raise ValueError(component)

    def rule_id(self) -> str:
        """The rule a mask was made under, as one string. Recorded per mask and
        per build, and part of every consumer's digest."""
        if self.mode == MODE_OFF:
            return "off"
        parts = []
        if COMPONENT_GDSAM in self.components:
            parts.append(f"gdino-base@{self.hand_score:.2f}/{self.phone_score:.2f}"
                         f"+sam2.1-b+|filldrop{self.fill_drop_frac:g}|maxfrac{self.max_mask_frac:g}")
        if COMPONENT_ONEFORMER in self.components:
            parts.append(f"oneformer-swinl-coco@{self.oneformer_threshold:g}")
        parts.append(f"phone-near{self.phone_near_px}|dil{self.dilate_px}@w{self.reference_width}"
                     f"|gamma{self.gamma:g}|v{TRANSIENT_SCHEMA}")
        return "|".join(parts)

    def models(self) -> dict:
        out = {mid: rev for c in self.components for mid, rev in COMPONENT_MODELS[c]}
        if COMPONENT_ONEFORMER in self.components:
            out[ONEFORMER_CLASS_INFO[0]] = ONEFORMER_CLASS_INFO[1]
        return out

    def as_dict(self) -> dict:
        return dict(self.__dict__)


# ---------------------------------------------------------------------------
# composition (CPU)
# ---------------------------------------------------------------------------


def _ellipse_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    import cv2  # noqa: PLC0415

    if radius <= 0 or not mask.any():
        return mask.astype(bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask.astype(np.uint8), k).astype(bool)


def held_phones(hand: np.ndarray, phone: np.ndarray, near_px: int) -> np.ndarray:
    """`hand` plus every connected component of `phone` within `near_px` of it."""
    import cv2  # noqa: PLC0415

    out = hand.astype(bool).copy()
    if not phone.any() or not hand.any():
        return out
    n, labels = cv2.connectedComponents(phone.astype(np.uint8))
    near = _ellipse_dilate(hand, near_px)
    touching = np.unique(labels[near & (labels > 0)])
    if len(touching):
        out |= np.isin(labels, touching)
    return out


def dilation_radius(params: TransientParams, width: int) -> int:
    return max(0, int(round(params.dilate_px * width / max(1, params.reference_width))))


def compose(parts, params: TransientParams, shape) -> np.ndarray:
    """The transient mask from each component's (hand, phone) masks: held
    phones per component, union over components, dilated."""
    H, W = shape
    out = np.zeros((H, W), bool)
    for hand, phone in parts:
        out |= held_phones(np.asarray(hand, bool), np.asarray(phone, bool), params.phone_near_px)
    return _ellipse_dilate(out, dilation_radius(params, W))


def filter_instance_masks(masks, labels, scores, unobserved, params: TransientParams):
    """Grounding DINO + SAM instance masks -> (hand, phone). Drops a mask over
    `max_mask_frac` of the image or more than `fill_drop_frac` inside the
    unobserved (fill) mask: the detector reads black redaction boxes as hands,
    and those pixels are already excluded, so all the mask would add is a rim."""
    H, W = unobserved.shape
    hand = np.zeros((H, W), bool)
    phone = np.zeros((H, W), bool)
    for m, label, score in zip(masks, labels, scores):
        m = np.asarray(m, bool)
        area = int(m.sum())
        if not area or m.mean() > params.max_mask_frac:
            continue
        if int((m & unobserved).sum()) > params.fill_drop_frac * area:
            continue
        label = str(label).lower()
        if any(w in label for w in HAND_WORDS) and score >= params.hand_score:
            hand |= m
        elif "phone" in label and score >= params.phone_score:
            phone |= m
    return hand, phone


# ---------------------------------------------------------------------------
# the cache
# ---------------------------------------------------------------------------


def cache_path(depth_dir: Path, ki: int, component: str) -> Path:
    return Path(depth_dir) / f"{int(ki):05d}_transient.{component}.npz"


def component_key(component: str, params: TransientParams, *, keyframe_id: str,
                  source_sha1: str, redaction_effective: str) -> dict:
    from tower.world_builder.appearance import UNOBSERVED_RULE  # noqa: PLC0415
    from tower.world_builder.dense_pipeline import FILL_RULE  # noqa: PLC0415

    return {
        "schema": TRANSIENT_SCHEMA,
        "component": component,
        "keyframe_id": keyframe_id,
        "source_sha1": source_sha1,
        "redaction_effective": redaction_effective,
        "fill_rule": FILL_RULE,
        "unobserved_rule": UNOBSERVED_RULE,
        "models": {mid: rev for mid, rev in COMPONENT_MODELS[component]},
        "params": params.component_params(component),
    }


def key_digest(key: dict) -> str:
    return hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()


def _log_bad_cache_file(path, why: str) -> None:
    logger.warning("[Tower][WorldBuilder][transients] the cached mask %s is unreadable (%s); "
                   "it is a miss, computed again and rewritten", Path(path).name, why)


def write_component(path: Path, key: dict, hand: np.ndarray, phone: np.ndarray, *,
                    image_sha1: str | None, seconds: float | None = None) -> None:
    """Atomically: a reader sees the old file or the whole new one.

    Raises `OSError` when the file cannot be written (a full disk, a cached file another
    process holds, MAX_PATH), after taking back its own staging file: the CALLER decides
    what a lost cache entry costs (review V10, L-12b), and every caller here keeps the
    mask it computed for the build in hand."""
    buf = io.BytesIO()
    record = dict(key, image_sha1=image_sha1, computed_at=time.time(),
                  seconds=None if seconds is None else round(float(seconds), 4))
    np.savez(buf, hand=np.packbits(np.asarray(hand, bool), axis=-1),
             phone=np.packbits(np.asarray(phone, bool), axis=-1),
             shape=np.asarray(hand.shape, np.int32),
             key=np.frombuffer(json.dumps(record, sort_keys=True).encode(), np.uint8))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.p{os.getpid()}.tmp")
    try:
        tmp.write_bytes(buf.getvalue())
        os.replace(tmp, path)
    except OSError:
        try:
            tmp.unlink(missing_ok=True)     # this call's own staging file, never the cache's
        except OSError:
            pass
        raise


def pack_masks(hand: np.ndarray, phone: np.ndarray) -> tuple:
    """A component's (hand, phone) held in memory when its cache write failed (review V10,
    L-12b), as bits: a full disk fails EVERY write, and a long walk's masks at one byte a
    pixel would be gigabytes. `unpack_masks` gives them back."""
    hand, phone = np.asarray(hand, bool), np.asarray(phone, bool)
    return np.packbits(hand, axis=-1), np.packbits(phone, axis=-1), int(hand.shape[-1])


def unpack_masks(packed: tuple) -> tuple:
    hand, phone, width = packed
    return (np.unpackbits(hand, axis=-1)[..., :width].astype(bool),
            np.unpackbits(phone, axis=-1)[..., :width].astype(bool))


def read_component(path: Path, key: dict | None = None, shape=None):
    """(hand, phone, record) when the file exists, reads, and was made under
    `key` (and has `shape`); else None. A missing mask is never an empty one.

    A file that exists but cannot be read whole -- empty, truncated, not an
    archive, missing an array, arrays of the wrong shape -- is a MISS like a
    missing one, and says so in the log: the caller computes the mask again and
    `write_component` replaces the file (review V9, LOW). An empty file raised
    `EOFError` and a torn archive `BadZipFile`, neither of which was caught, so
    one bad file failed every later mask step of its session for good."""
    try:
        if not Path(path).is_file():
            return None
    except OSError:
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            record = json.loads(bytes(z["key"]).decode())
            hs, ws = (int(v) for v in z["shape"])
            hand = np.unpackbits(z["hand"], axis=-1)[..., :ws].astype(bool)
            phone = np.unpackbits(z["phone"], axis=-1)[..., :ws].astype(bool)
        if not isinstance(record, dict):
            raise ValueError("its key is not a record")
    except Exception as exc:  # noqa: BLE001 -- EOFError, BadZipFile, zlib, ValueError, ...: a miss
        _log_bad_cache_file(path, f"{type(exc).__name__}: {exc}")
        return None
    if hand.shape != (hs, ws) or phone.shape != (hs, ws):
        _log_bad_cache_file(path, f"its masks are not the {hs}x{ws} it records")
        return None
    if shape is not None and (hs, ws) != tuple(shape):
        return None
    if key is not None and any(record.get(k) != v for k, v in key.items()):
        return None
    return hand, phone, record


# ---------------------------------------------------------------------------
# backends (one per component)
# ---------------------------------------------------------------------------


class ComponentBackend:
    """Runs one component over a batch of keyframes.

    `run(items, params, emit, should_stop)`: `items` is a list of
    `(ki, rgb (H, W, 3) uint8, unobserved (H, W) bool)`; `emit(ki, hand, phone,
    seconds)` is called once per keyframe as soon as its masks are known.
    Returns a dict of timings, or raises `TransientDetectorUnavailable`.
    `probe()` is cheap: a reason this backend cannot run, or None.
    """

    component = "abstract"

    def probe(self) -> str | None:  # pragma: no cover - interface
        return None

    def run(self, items, params: TransientParams, emit, should_stop=None) -> dict:  # pragma: no cover
        raise NotImplementedError


def _packages_missing(*modules) -> str | None:
    import importlib.util  # noqa: PLC0415

    missing = [m for m in modules if importlib.util.find_spec(m) is None]
    if missing:
        return (f"the transient detector needs {', '.join(missing)}, which "
                "is not installed (pyproject extra `transients`)")
    return None


def _gamma_lut(gamma: float) -> np.ndarray:
    return np.clip((np.arange(256) / 255.0) ** (1.0 / gamma) * 255, 0, 255).astype(np.uint8)


def _fetch(component: str, repo_id: str, revision: str, files, repo_type: str = "model") -> str:
    """The local snapshot directory of a pinned checkpoint, downloading what is
    missing. Through `dense.load_hub_weights`: offline and uncached is
    `TransientDetectorUnavailable` naming the repo and cache, and a first
    download logs its size.

    Fetched explicitly, and the model then loaded from the local directory,
    because `transformers` does not always say "not cached": measured with
    `HF_HUB_OFFLINE=1` and an empty cache, OneFormer's image processor raised a
    bare OSError with no hub error behind it, which read as a crash."""
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    from tower.world_builder.dense import load_hub_weights  # noqa: PLC0415

    return load_hub_weights(
        component, repo_id,
        lambda: snapshot_download(repo_id, revision=revision, allow_patterns=list(files),
                                  repo_type=repo_type),
        what="transient detector model", error=TransientDetectorUnavailable,
        hint=COMPONENT_DOWNLOAD_HINT[component], log_tag="transients")


def _free():
    """Collect what the caller just dropped and hand the card back."""
    import gc  # noqa: PLC0415

    import torch  # noqa: PLC0415

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _sync():
    import torch  # noqa: PLC0415

    if torch.cuda.is_available():
        torch.cuda.synchronize()


class GroundingDinoSamBackend(ComponentBackend):
    component = COMPONENT_GDSAM

    def probe(self) -> str | None:
        return _packages_missing("torch", "transformers", "PIL")

    def run(self, items, params, emit, should_stop=None) -> dict:
        import torch  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        try:
            from transformers import (  # noqa: PLC0415
                AutoProcessor,
                GroundingDinoForObjectDetection,
                Sam2Model,
                Sam2Processor,
            )
        except ImportError as exc:
            raise TransientDetectorUnavailable(f"transformers cannot load Grounding DINO / SAM 2 ({exc})") from None
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        lut = _gamma_lut(params.gamma)
        times = {"gdino_load": 0.0, "gdino": 0.0, "sam_load": 0.0, "sam": 0.0,
                 "frames": len(items), "frames_with_boxes": 0}

        t = time.time()
        gdino_dir = _fetch(self.component, *GDINO_MODEL, MODEL_FILES[GDINO_MODEL[0]])
        sam_dir = _fetch(self.component, *SAM_MODEL, MODEL_FILES[SAM_MODEL[0]])
        times["fetch"] = round(time.time() - t, 3)
        t = time.time()
        dproc = AutoProcessor.from_pretrained(gdino_dir)
        dmodel = GroundingDinoForObjectDetection.from_pretrained(gdino_dir).to(dev).eval()
        times["gdino_load"] = round(time.time() - t, 3)
        boxes_by_ki = {}
        per_frame = {}
        t = time.time()
        try:
            for ki, rgb, unobserved in items:
                if should_stop is not None and should_stop():
                    return dict(times, stopped=True)
                t1 = time.time()
                H, W = rgb.shape[:2]
                img = Image.fromarray(lut[rgb])
                with torch.inference_mode():
                    inp = dproc(images=img, text=params.gdino_prompt, return_tensors="pt").to(dev)
                    out = dmodel(**inp)
                    r = dproc.post_process_grounded_object_detection(
                        out, inp.input_ids, threshold=params.gdino_box_threshold,
                        text_threshold=params.gdino_text_threshold, target_sizes=[(H, W)])[0]
                boxes = r["boxes"].float().cpu().numpy()
                scores = r["scores"].float().cpu().numpy()
                labels = [str(x) for x in r.get("text_labels", r.get("labels"))]
                if len(boxes):
                    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W - 1)
                    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H - 1)
                    keep = (((boxes[:, 2] - boxes[:, 0]) > params.min_box_px)
                            & ((boxes[:, 3] - boxes[:, 1]) > params.min_box_px))
                    boxes, scores = boxes[keep], scores[keep]
                    labels = [lab for lab, k in zip(labels, keep) if k]
                boxes_by_ki[ki] = (boxes, scores, labels)
                per_frame[ki] = time.time() - t1
        finally:
            _sync()
            times["gdino"] = round(time.time() - t, 3)
            dmodel = dproc = None
            _free()

        with_boxes = [(ki, rgb, unob) for ki, rgb, unob in items if len(boxes_by_ki[ki][0])]
        times["frames_with_boxes"] = len(with_boxes)
        for ki, rgb, _unob in items:
            if not len(boxes_by_ki[ki][0]):
                z = np.zeros(rgb.shape[:2], bool)
                emit(ki, z, z, per_frame[ki])
        if not with_boxes:
            return times

        t = time.time()
        sproc = Sam2Processor.from_pretrained(sam_dir)
        smodel = Sam2Model.from_pretrained(sam_dir).to(dev).eval()
        times["sam_load"] = round(time.time() - t, 3)
        t = time.time()
        try:
            for ki, rgb, unobserved in with_boxes:
                if should_stop is not None and should_stop():
                    return dict(times, stopped=True)
                t1 = time.time()
                boxes, scores, labels = boxes_by_ki[ki]
                img = Image.fromarray(lut[rgb])
                with torch.inference_mode():
                    sin = sproc(images=img, input_boxes=[[b.tolist() for b in boxes]],
                                return_tensors="pt").to(dev)
                    sout = smodel(**sin, multimask_output=False)
                    m = sproc.post_process_masks(sout.pred_masks.cpu(), sin["original_sizes"].cpu())[0]
                masks = m[:, 0].numpy().astype(bool)
                hand, phone = filter_instance_masks(masks, labels, scores, unobserved, params)
                emit(ki, hand, phone, per_frame[ki] + time.time() - t1)
        finally:
            _sync()
            times["sam"] = round(time.time() - t, 3)
            smodel = sproc = None
            _free()
        return times


class OneFormerBackend(ComponentBackend):
    component = COMPONENT_ONEFORMER

    def probe(self) -> str | None:
        return _packages_missing("torch", "transformers", "PIL", "scipy")

    def run(self, items, params, emit, should_stop=None) -> dict:
        import torch  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        try:
            from transformers import (  # noqa: PLC0415
                OneFormerForUniversalSegmentation,
                OneFormerProcessor,
            )
        except ImportError as exc:
            raise TransientDetectorUnavailable(f"transformers cannot load OneFormer ({exc})") from None
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        lut = _gamma_lut(params.gamma)
        times = {"oneformer_load": 0.0, "oneformer": 0.0, "frames": len(items)}
        t = time.time()
        model_dir = _fetch(self.component, *ONEFORMER_MODEL, MODEL_FILES[ONEFORMER_MODEL[0]])
        info_repo, info_rev, info_file = ONEFORMER_CLASS_INFO
        info_dir = _fetch(self.component, info_repo, info_rev, [info_file], repo_type="dataset")
        times["fetch"] = round(time.time() - t, 3)
        t = time.time()
        proc = OneFormerProcessor.from_pretrained(model_dir, repo_path=info_dir,
                                                  class_info_file=info_file)
        model = OneFormerForUniversalSegmentation.from_pretrained(model_dir).to(dev).eval()
        times["oneformer_load"] = round(time.time() - t, 3)
        id2label = model.config.id2label
        t = time.time()
        try:
            for ki, rgb, _unobserved in items:
                if should_stop is not None and should_stop():
                    return dict(times, stopped=True)
                t1 = time.time()
                H, W = rgb.shape[:2]
                img = Image.fromarray(lut[rgb])
                with torch.inference_mode():
                    inp = proc(images=img, task_inputs=["panoptic"], return_tensors="pt").to(dev)
                    out = model(**inp)
                    r = proc.post_process_panoptic_segmentation(
                        out, target_sizes=[(H, W)], threshold=params.oneformer_threshold)[0]
                seg = r["segmentation"].cpu().numpy()
                hand = np.zeros((H, W), bool)
                phone = np.zeros((H, W), bool)
                for s in r["segments_info"]:
                    label = id2label[s["label_id"]]
                    if label == "person":
                        hand |= seg == s["id"]
                    elif label == "cell phone":
                        phone |= seg == s["id"]
                emit(ki, hand, phone, time.time() - t1)
        finally:
            _sync()
            times["oneformer"] = round(time.time() - t, 3)
            model = proc = None
            _free()
        return times


class DisabledBackend(ComponentBackend):
    def __init__(self, component: str, reason: str):
        self.component = component
        self.reason = reason

    def probe(self):
        return self.reason

    def run(self, items, params, emit, should_stop=None):  # pragma: no cover - never run
        raise TransientDetectorUnavailable(self.reason)


def default_backend_factory(component: str) -> ComponentBackend:
    switch = os.environ.get(ENV_SWITCH, "").strip().lower()
    if switch in ("0", "off", "false", "no"):
        return DisabledBackend(component, f"transient detection is disabled on this Tower "
                                          f"({ENV_SWITCH}={os.environ.get(ENV_SWITCH)})")
    if component == COMPONENT_GDSAM:
        return GroundingDinoSamBackend()
    if component == COMPONENT_ONEFORMER:
        return OneFormerBackend()
    raise ValueError(component)


# Looked up at call time, so a test (and the suite's conftest) can replace it.
BACKEND_FACTORY: Callable[[str], ComponentBackend] = default_backend_factory

_LOGGED: set = set()


def _log_unavailable_once(reason: str) -> None:
    if reason in _LOGGED:
        return
    _LOGGED.add(reason)
    logger.warning("[Tower][WorldBuilder][transients] no detector masks: %s", reason)


# ---------------------------------------------------------------------------
# the entry point
# ---------------------------------------------------------------------------


@dataclass
class TransientReport:
    state: str
    params: TransientParams
    detail: str | None = None
    depth_dir: Path | None = None
    shape: tuple | None = None
    # ki -> [(component, key)] for every keyframe whose every component is cached
    keys: dict = field(default_factory=dict)
    cached: int = 0
    computed: int = 0
    refused: dict = field(default_factory=dict)
    seconds: dict = field(default_factory=dict)
    gpu_peak_mb: float | None = None
    # A `union` build that could run only OneFormer (review 1, m1): the mode
    # that was asked for, and why the masks are OneFormer's alone. `params` is
    # then the OneFormer rule the masks were actually made under, so every
    # consumer digest differs from a whole union build's and a later build with
    # the network rebuilds.
    requested: TransientParams | None = None
    partial: str | None = None
    # Keyframes whose masks were taken from the final solve's own cache
    # (`solve_masks.solve_mask_donor`) instead of being computed here.
    reused_from_solve: int = 0
    # Masks computed by THIS build whose cache write failed (a full disk, a held file,
    # MAX_PATH; review V10, L-12b): (ki, component) -> `pack_masks(hand, phone)`, used by
    # `mask` for this build, and counted. The failed write used to fail the whole stage. The
    # next build finds no cache entry and computes them again.
    cache_write_failed: int = 0
    kept: dict = field(default_factory=dict, repr=False)

    @property
    def available(self) -> bool:
        return self.state == STATE_OK

    def mask(self, ki: int) -> np.ndarray | None:
        """The composed mask of keyframe `ki`, or None when it has none. Keys
        are checked again on read: a file replaced since is not trusted."""
        if not self.available or ki not in self.keys:
            return None
        parts = []
        for component, key in self.keys[ki]:
            packed = self.kept.get((ki, component))
            if packed is not None:
                got = unpack_masks(packed)
            else:
                got = read_component(cache_path(self.depth_dir, ki, component), key, self.shape)
            if got is None:
                return None
            parts.append(got[:2])
        return compose(parts, self.params, self.shape)

    def frames_digest(self) -> str | None:
        """SHA-1 over every (ki, component, key digest) the masks came from."""
        if not self.available:
            return None
        h = hashlib.sha1()
        for ki in sorted(self.keys):
            for component, key in self.keys[ki]:
                h.update(f"{ki}\t{component}\t{key_digest(key)}\n".encode())
        return h.hexdigest()

    def record(self) -> dict:
        """What a consumer's manifest says about its transient masks."""
        out = {
            "state": self.state,
            "detail": self.detail,
            "mode": self.params.mode,
            "rule": self.params.rule_id() if self.state == STATE_OK else None,
            "requested_rule": (self.requested or self.params).rule_id(),
            "partial": self.partial,
            "models": self.params.models() if self.state == STATE_OK else {},
            "frames_masked": len(self.keys) if self.state == STATE_OK else 0,
            "computed": self.computed,
            "cached": self.cached,
            "refused": dict(self.refused),
            "frames_digest": self.frames_digest(),
            "seconds": dict(self.seconds),
            "gpu_peak_mb": self.gpu_peak_mb,
        }
        # Only when it happened: a build that took nothing from the solve --
        # every build of a world solved without `TOWER_WORLD_SOLVE_MASKS` --
        # writes exactly the record it wrote before.
        if self.reused_from_solve:
            out["reused_from_solve"] = self.reused_from_solve
        if self.cache_write_failed:
            out["cache_write_failed"] = self.cache_write_failed
        return out


def _take_from_donor(donor, kid, component, params, shape, path, key) -> bool:
    """Copy one component's masks from the solve's cache into this stage's own
    cache, under this stage's key plus where they came from. False when the
    solve has none for this keyframe under the same component rule."""
    try:
        got = donor(kid, component, params, shape)
    except Exception:  # noqa: BLE001 -- a donor is an optimisation, never a failure
        logger.exception("[Tower][WorldBuilder][transients] solve mask lookup failed for %s", kid)
        return False
    if got is None:
        return False
    hand, phone, provenance = got
    try:
        write_component(path, dict(key, **provenance), hand, phone,
                        image_sha1=provenance.get("solve_image_sha1"))
    except OSError as exc:
        # Not taken (review V10, L-12b): the detector computes this mask instead, and a
        # write that fails again there is kept in memory for the build.
        logger.warning("[Tower][WorldBuilder][transients] could not copy the solve's mask of %s "
                       "(%s: %s); it is computed here instead", kid, type(exc).__name__, exc)
        return False
    return True


def ensure_transient_masks(store, world_id: str, session_id: str, frames, *,
                           intrinsics, camera: dict, align_records: dict, depth_dir,
                           params: TransientParams, policy=None, redactor_factory=None,
                           backend_factory=None, should_stop=None, progress=None,
                           donor=None) -> TransientReport:
    """Make sure every keyframe in `frames` ((ki, keyframe_id) pairs) has its
    transient mask cached under `params`, computing only what is missing.

    Pixels come only from `appearance.keyframe_source` -- with ONE exception,
    and only for a world whose final solve was run with
    `TOWER_WORLD_SOLVE_MASKS`: a mask the solve already computed for the same
    keyframe under the same component rule is copied rather than recomputed
    (`solve_masks`, "one computation, two consumers"). The solve's detector saw
    the solver image -- the same undistorted pixel grid, from the raw capture
    frame where one exists, without redaction fill -- and the copied file says
    so (`origin: solve`).

    ONLY FOR A RAW-IMAGERY BUILD. The solver image is the unredacted capture
    frame, which a redacted (product) build must never draw from -- not even
    as the shape of a mask that becomes the published alpha
    (`test_masks_are_computed_only_from_the_redacted_session_keyframes`). A
    research build (`TOWER_WORLD_RAW_IMAGERY`) already reads those frames, so
    for it the copy changes nothing about provenance. `donor=None` looks the
    solve's cache up; `False` turns the lookup off. Never raises for a machine
    that cannot run the detector: the report says `unavailable`.
    """
    from tower.world_builder import appearance as A  # noqa: PLC0415

    t0 = time.time()
    depth_dir = Path(depth_dir)
    W, H = int(camera["width"]), int(camera["height"])
    report = TransientReport(state=STATE_OK, params=params, depth_dir=depth_dir, shape=(H, W))
    if params.mode == MODE_OFF:
        report.state = STATE_OFF
        report.detail = "transient detection is off for this build"
        return report
    factory = backend_factory or BACKEND_FACTORY
    backends = [factory(c) for c in params.components]
    missing_components = {}
    for b in backends:
        reason = b.probe()
        if reason:
            _log_unavailable_once(reason)
            missing_components[b.component] = reason
    if missing_components:
        if not _can_fall_back(params, missing_components):
            report.state, report.detail = STATE_UNAVAILABLE, next(iter(missing_components.values()))
            return report
        params, backends = _fall_back(report, params, backends, missing_components)
    if policy is None:
        try:
            policy = A.resolve_label_policy(store, world_id, session_id, redactor_factory)
        except A.AppearanceUnavailable as exc:
            _log_unavailable_once(exc.reason)
            report.state, report.detail = STATE_UNAVAILABLE, exc.reason
            return report
    undistorter = A.Undistorter(intrinsics, camera)
    if not getattr(policy, "raw", False) or donor is False:
        donor = None   # a redacted build: never a mask made from raw pixels
    elif donor is None:
        from tower.world_builder.solve_masks import solve_mask_donor  # noqa: PLC0415

        donor = solve_mask_donor(store, world_id, session_id)

    # -- the cheap pass: which (frame, component) are already cached ---------
    wanted: dict = {}                     # ki -> [(component, key)]
    missing: dict = {c: [] for c in params.components}
    kids = {}
    for ki, kid in frames:
        ki = int(ki)
        src = A.keyframe_source(store, world_id, session_id, kid, ki, policy=policy,
                                align_record=None, depth_dir=None,
                                undistorter=undistorter, hash_only=True)
        if src.refused:
            report.refused[src.refused] = report.refused.get(src.refused, 0) + 1
            continue
        kids[ki] = kid
        keys = [(c, component_key(c, params, keyframe_id=kid, source_sha1=src.source_sha1,
                                  redaction_effective=policy.effective))
                for c in params.components]
        wanted[ki] = keys
        any_missing = False
        took = False
        for c, key in keys:
            path = cache_path(depth_dir, ki, c)
            if read_component(path, key, (H, W)) is None:
                if donor is not None and _take_from_donor(donor, kid, c, params, (H, W), path, key):
                    took = True
                    continue
                missing[c].append(ki)
                any_missing = True
        if took:
            report.reused_from_solve += 1
        elif not any_missing:
            report.cached += 1
    report.seconds["lookup"] = round(time.time() - t0, 3)

    need = sorted({ki for kis in missing.values() for ki in kis})
    if need:
        t1 = time.time()
        pixels = {}
        for n, ki in enumerate(need):
            if should_stop is not None and should_stop():
                report.state = STATE_STOPPED
                return report
            src = A.keyframe_source(store, world_id, session_id, kids[ki], ki, policy=policy,
                                    align_record=align_records.get(ki), depth_dir=depth_dir,
                                    undistorter=undistorter)
            if src.refused:
                report.refused[src.refused] = report.refused.get(src.refused, 0) + 1
                wanted.pop(ki, None)
                continue
            pixels[ki] = (src.rgb, src.unobserved, src.image_sha1)
            if progress is not None and n % 50 == 0:
                progress("transients-provenance", n, len(need))
        report.seconds["provenance"] = round(time.time() - t1, 3)

        peak = _reset_peak()
        computed = set()
        for backend in list(backends):
            if backend.component not in params.components:
                continue
            c = backend.component
            todo = [ki for ki in missing[c] if ki in pixels]
            if not todo:
                continue
            keymap = {ki: dict(wanted[ki])[c] for ki in todo}

            def emit(ki, hand, phone, seconds, c=c, keymap=keymap):
                try:
                    write_component(cache_path(depth_dir, ki, c), keymap[ki], hand, phone,
                                    image_sha1=pixels[ki][2], seconds=seconds)
                except OSError as exc:
                    # Not the detector's failure (review V10, L-12b): kept for this build.
                    report.cache_write_failed += 1
                    report.kept[(ki, c)] = pack_masks(hand, phone)
                    if report.cache_write_failed == 1:
                        logger.warning("[Tower][WorldBuilder][transients] %s/%s: could not keep a "
                                       "mask in the cache (%s: %s); this build uses the mask it "
                                       "computed", world_id, session_id, type(exc).__name__, exc)
                computed.add(ki)

            items = [(ki, pixels[ki][0], pixels[ki][1]) for ki in todo]
            t2 = time.time()
            try:
                timings = backend.run(items, params, emit, should_stop)
            except TransientDetectorUnavailable as exc:
                _log_unavailable_once(str(exc))
                if _can_fall_back(params, {c: str(exc)}):
                    # Its weights could not be fetched at load (offline): keep
                    # the OneFormer masks rather than discard every mask.
                    params, backends = _fall_back(report, params, backends, {c: str(exc)})
                    wanted = {ki: [(comp, key) for comp, key in keys if comp in params.components]
                              for ki, keys in wanted.items()}
                    continue
                report.state, report.detail = STATE_UNAVAILABLE, str(exc)
                report.keys = {}
                return report
            except Exception as exc:  # noqa: BLE001 -- a detector crash is not a build crash
                logger.exception("[Tower][WorldBuilder][transients] %s/%s: the %s detector failed",
                                 world_id, session_id, c)
                report.state, report.detail = STATE_FAILED, f"{c}: {type(exc).__name__}: {exc}"
                report.keys = {}
                return report
            report.seconds[c] = round(time.time() - t2, 3)
            for k, v in (timings or {}).items():
                if k != "stopped":
                    report.seconds[f"{c}.{k}"] = v
            if (timings or {}).get("stopped"):
                report.state = STATE_STOPPED
                report.detail = f"stopped during {c}"
                return report
        report.computed = len(computed)
        report.gpu_peak_mb = _peak_mb(peak)

    # Only keyframes whose every component is on disk under its key -- or was computed by this
    # build and could not be written (`kept`, review V10 L-12b).
    report.keys = {ki: keys for ki, keys in wanted.items()
                   if all((ki, c) in report.kept
                          or read_component(cache_path(depth_dir, ki, c), key, (H, W)) is not None
                          for c, key in keys)}
    report.seconds["total"] = round(time.time() - t0, 3)
    if need:
        logger.info("[Tower][WorldBuilder][transients] %s/%s: %d keyframes masked "
                    "(%d computed, %d cached) under %s in %.1f s",
                    world_id, session_id, len(report.keys), report.computed, report.cached,
                    params.mode, report.seconds["total"])
    return report


def _can_fall_back(params: TransientParams, missing: dict) -> bool:
    """A `union` whose Grounding DINO + SAM component cannot run still has
    OneFormer -- which the live child ran all walk, so its weights are on disk.
    Nothing else falls back: `oneformer` without OneFormer has no masks."""
    return (params.mode == MODE_UNION and COMPONENT_ONEFORMER not in missing
            and set(missing) <= {COMPONENT_GDSAM})


def _fall_back(report: TransientReport, params: TransientParams, backends, missing: dict):
    """Continue in `oneformer` mode and say so (review 1, m1): the final build
    after Stop on a machine that cannot fetch Grounding DINO or SAM used to
    record `unavailable` with no masks at all, discarding the OneFormer masks
    the walk had cached -- so the finished world showed hands the live one had
    masked."""
    why = "; ".join(f"{c}: {r}" for c, r in sorted(missing.items()))
    effective = replace(params, mode=MODE_ONEFORMER)
    report.requested = report.requested or params
    report.params = effective
    report.partial = (f"{params.mode} was requested but only {MODE_ONEFORMER} could run ({why}); "
                      "these masks are OneFormer's alone")
    logger.warning("[Tower][WorldBuilder][transients] %s", report.partial)
    return effective, [b for b in backends if b.component in effective.components]


def _reset_peak():
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            return True
    except Exception:  # noqa: BLE001
        pass
    return False


def _peak_mb(active) -> float | None:
    if not active:
        return None
    import torch  # noqa: PLC0415

    return round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
