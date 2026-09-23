"""Transient masks (the wearer's hands, arms and held phone) on SOLVER images.

`transients.py` computes these masks for the surface and appearance stages,
downstream of the solve. The solve itself reads raw, unmasked frames, and on
the target world the only glue between two rigid islands is 7 verified pairs
whose inliers sit on the phone in the wearer's hand (FORENSICS H-A). This
module runs the SAME recipe (`transients.TransientParams()`: union mode --
Grounding DINO-base + SAM 2.1 hiera-b+ boxes -> masks, plus OneFormer Swin-L
COCO person / cell phone; the held-phone rule; a 12 px ellipse dilation at
359 px width) on the images the solver is actually given: the staged
canonical (undistorted, cropped) frames.

Differences from the downstream use, stated because they matter:

  * the input is the staged solver image -- raw and unredacted where raw
    capture exists -- not the redacted keyframe, so there are no fill boxes
    and `unobserved` is empty (the fill-drop rule never fires);
  * masks are keyed by the staged image name AND the SHA-1 of its bytes, so
    a keyframe and a solver-only frame are cached the same way and every arm
    of every experiment reuses them.

GPU work happens only in `compute_masks`, which the CLI `masks` subcommand
runs under the machine-wide GPU lock. Reading (`load_mask`,
`write_colmap_masks`) is CPU only and never computes: a missing mask is
reported, never silently treated as empty.

Licences (HANDS.md): Grounding DINO Apache-2.0, SAM 2.1 Apache-2.0,
OneFormer MIT. No non-commercial or copyleft component.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from tower.world_builder import transients as tr

MASK_SCHEMA = 1
INPUT_RULE = "staged-canonical-image|no-fill"


class MasksMissing(RuntimeError):
    """Some staged images have no cached mask; run the `masks` step (GPU) first."""

    def __init__(self, names):
        self.names = list(names)
        head = ", ".join(self.names[:5])
        super().__init__(f"{len(self.names)} staged images have no cached transient mask "
                         f"(first: {head}); run `world_coherence_variant.py masks` under the GPU lock")


def default_params() -> tr.TransientParams:
    return tr.TransientParams()  # union mode, as HANDS.md recommends


def cache_dir(cache_root, world_id: str) -> Path:
    return Path(cache_root) / "masks" / str(world_id)


def component_path(cdir: Path, name: str, component: str, sha1: str) -> Path:
    """One file per (staged name, image bytes, component): the raw and the
    redacted version of one keyframe stage under the same name and must not
    overwrite each other's masks."""
    return Path(cdir) / f"{Path(name).stem}.{sha1[:12]}.{component}.npz"


def mask_key(component: str, params: tr.TransientParams, name: str, sha1: str) -> dict:
    return {
        "schema": MASK_SCHEMA,
        "component": component,
        "image": name,
        "image_sha1": sha1,
        "input_rule": INPUT_RULE,
        "models": {mid: rev for mid, rev in tr.COMPONENT_MODELS[component]},
        "params": params.component_params(component),
    }


def load_parts(cdir: Path, name: str, sha1: str, params: tr.TransientParams, shape=None):
    """[(hand, phone)] per component, or None if any component is missing/stale."""
    parts = []
    for comp in params.components:
        r = tr.read_component(component_path(cdir, name, comp, sha1), mask_key(comp, params, name, sha1), shape)
        if r is None:
            return None
        parts.append((r[0], r[1]))
    return parts


def load_mask(cdir: Path, name: str, sha1: str, params: tr.TransientParams, shape=None):
    """The composed transient mask (True = transient), or None."""
    parts = load_parts(cdir, name, sha1, params, shape)
    if parts is None:
        return None
    H, W = parts[0][0].shape
    return tr.compose(parts, params, (H, W))


def missing(cdir: Path, images, params: tr.TransientParams) -> list[str]:
    """Names of `images` ({name, sha1}) without a complete cached mask."""
    out = []
    for im in images:
        for comp in params.components:
            if tr.read_component(component_path(cdir, im["name"], comp, im["sha1"]),
                                 mask_key(comp, params, im["name"], im["sha1"])) is None:
                out.append(im["name"])
                break
    return out


def _read_rgb(path) -> np.ndarray:
    import cv2

    bgr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"unreadable image {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def compute_masks(images, cdir: Path, params: tr.TransientParams | None = None, *, log=print,
                  chunk: int = 400, backend_factory=tr.default_backend_factory) -> dict:
    """Run every component over the images that lack it; cache per image.

    `images`: list of {name, path, sha1}. GPU: the caller holds the lock."""
    params = params or default_params()
    cdir = Path(cdir)
    cdir.mkdir(parents=True, exist_ok=True)
    report = {"images": len(images), "components": {}, "rule": params.rule_id()}
    for comp in params.components:
        todo = [im for im in images
                if tr.read_component(component_path(cdir, im["name"], comp, im["sha1"]),
                                     mask_key(comp, params, im["name"], im["sha1"])) is None]
        rep = {"todo": len(todo), "computed": 0, "timings": []}
        report["components"][comp] = rep
        if not todo:
            continue
        backend = backend_factory(comp)
        reason = backend.probe()
        if reason:
            raise tr.TransientDetectorUnavailable(reason)
        t0 = time.time()
        for start in range(0, len(todo), chunk):
            batch = todo[start:start + chunk]
            items = []
            for i, im in enumerate(batch):
                rgb = _read_rgb(im["path"])
                items.append((i, rgb, np.zeros(rgb.shape[:2], bool)))

            def emit(ki, hand, phone, seconds, batch=batch, comp=comp):
                im = batch[ki]
                tr.write_component(component_path(cdir, im["name"], comp, im["sha1"]),
                                   mask_key(comp, params, im["name"], im["sha1"]),
                                   np.asarray(hand, bool), np.asarray(phone, bool),
                                   image_sha1=im["sha1"], seconds=seconds)
                rep["computed"] += 1

            timings = backend.run(items, params, emit)
            rep["timings"].append(timings)
            log(f"masks {comp}: {min(start + chunk, len(todo))}/{len(todo)} "
                f"({time.time() - t0:.0f} s)")
        rep["seconds"] = round(time.time() - t0, 2)
    return report


def write_colmap_masks(images, cdir: Path, out_dir: Path, params: tr.TransientParams | None = None) -> dict:
    """COLMAP feature-extraction masks: `<out_dir>/<image name>.png`, 0 = ignore
    (transient), 255 = use. Raises MasksMissing if any mask is not cached.
    Returns per-image area stats."""
    import cv2

    params = params or default_params()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lost = missing(cdir, images, params)
    if lost:
        raise MasksMissing(lost)
    stats = {}
    for im in images:
        parts = load_parts(cdir, im["name"], im["sha1"], params)
        H, W = parts[0][0].shape
        m = tr.compose(parts, params, (H, W))
        hand = np.zeros((H, W), bool)
        phone = np.zeros((H, W), bool)
        for h, p in parts:
            hand |= h
            phone |= p
        png = np.where(m, 0, 255).astype(np.uint8)
        ok, buf = cv2.imencode(".png", png)
        if not ok:
            raise OSError(f"cannot encode mask for {im['name']}")
        (out_dir / f"{im['name']}.png").write_bytes(buf.tobytes())
        stats[im["name"]] = {"masked_frac": float(m.mean()), "hand_raw_frac": float(hand.mean()),
                             "phone_raw_frac": float(phone.mean()),
                             "held_phone": bool((phone & ~hand).any() and m.any())}
    return stats


def area_summary(stats: dict) -> dict:
    """Distribution of the masked fraction over images."""
    if not stats:
        return {"images": 0}
    v = np.array([s["masked_frac"] for s in stats.values()])
    return {"images": int(len(v)), "any_mask": int((v > 0).sum()),
            "frac_images_masked": float((v > 0).mean()),
            "masked_frac_mean": float(v.mean()), "masked_frac_median": float(np.median(v)),
            "masked_frac_p90": float(np.percentile(v, 90)), "masked_frac_max": float(v.max()),
            "images_over_50pct": int((v > 0.5).sum()), "images_over_80pct": int((v > 0.8).sum())}
