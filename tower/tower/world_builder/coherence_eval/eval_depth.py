"""Monocular metric depth per keyframe, cached ONCE per frozen world.

The scale-drift metric compares each keyframe's SfM depths with an
independent, image-only estimate. The estimate depends on nothing but the
keyframe images, so it is computed once per world and reused for every
variant: two variants are then compared against literally the same numbers.

MODEL: MoGe-2 ViT-L (``Ruicheng/moge-2-vitl``, MIT), the same checkpoint the
dense/surface stage uses (`dense.py` registry ``moge2-vitl``), loaded through
the same `dense.load_hub_weights`. It predicts a METRIC point map; the z
channel is depth along the optical axis, which is exactly the quantity an SfM
camera-frame point's z is. The known horizontal field of view of the canonical
camera is passed (``fov_x``), so the prediction uses the true intrinsics
instead of guessing them. ``resolution_level`` 9 as in `dense.MoGeBackend`.
Pixels MoGe masks as invalid are stored as NaN.

INPUT: the canonical undistorted keyframe (`eval_world.WorldInfo.canonical_image`,
i.e. ``solve/<sid>/images`` when present, otherwise the raw capture frame
undistorted by the solver's own maps). Every input's SHA-1 is recorded.

LAYOUT (``<cache>/<world_id>/depth/``)::

    manifest.json          model, parameters, per-keyframe input source + sha1
    <image stem>.npy       (H, W) float16 metric depth in metres, NaN = invalid

LIMITATIONS: a monocular metric depth network has a per-image scale error
(several percent, scene dependent: close-up clutter vs an open room). The
per-keyframe log-ratio therefore carries that error as noise; drift is read
from robust statistics over many keyframes, never from one keyframe.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from tower.world_builder.coherence_eval.eval_world import WorldInfo

DEPTH_CACHE_VERSION = 1
DEFAULT_BACKEND = "moge2-vitl"
RESOLUTION_LEVEL = 9


def depth_dir(cache_root, world_id: str) -> Path:
    return Path(cache_root) / world_id / "depth"


class DepthCache:
    """Read side: nearest-pixel sampling of cached depth maps."""

    def __init__(self, root) -> None:
        self.root = Path(root)
        self.manifest = None
        p = self.root / "manifest.json"
        if p.is_file():
            self.manifest = json.loads(p.read_text(encoding="utf-8"))
        self._memo: dict[str, np.ndarray] = {}

    @property
    def available(self) -> bool:
        m = self.manifest or {}
        return bool(m.get("complete")) and not m.get("failed_frames")

    def digest(self) -> str | None:
        """The maps' content digest (recorded by the build; recomputed when an
        older manifest lacks it)."""
        m = self.manifest or {}
        if m.get("content_digest"):
            return m["content_digest"]
        names = [n for n, _ in sorted(((n, r.get("index", 0)) for n, r in (m.get("frames") or {}).items()),
                                      key=lambda x: x[1])]
        return content_digest(self.root, names) if names else None

    def camera(self) -> dict | None:
        return (self.manifest or {}).get("camera")

    def load(self, image_name: str) -> np.ndarray | None:
        if image_name in self._memo:
            return self._memo[image_name]
        p = self.root / (Path(image_name).stem + ".npy")
        if not p.is_file():
            return None
        d = np.load(p).astype(np.float32)
        if len(self._memo) > 64:
            self._memo.clear()
        self._memo[image_name] = d
        return d

    def sample(self, image_name: str, uv) -> np.ndarray:
        """Depth at canonical pixels (nearest pixel centre); NaN when outside or invalid."""
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        out = np.full(len(uv), np.nan)
        d = self.load(image_name)
        if d is None or not len(uv):
            return out
        h, w = d.shape
        uv = np.where(np.isfinite(uv), uv, -1.0)
        # COLMAP pixel convention (pixel i spans [i, i+1)), which is what the
        # solve's observations are in; for OpenCV-convention inputs this is at
        # most half a pixel off, far below the depth map's own resolution.
        iu = np.floor(uv[:, 0]).astype(np.int64)
        iv = np.floor(uv[:, 1]).astype(np.int64)
        ok = (iu >= 0) & (iu < w) & (iv >= 0) & (iv < h) & np.isfinite(uv).all(1)
        out[ok] = d[iv[ok], iu[ok]]
        return out


def content_digest(root: Path, image_names) -> str | None:
    """SHA-1 over the stored maps' bytes, in capture order (16 hex chars)."""
    h = hashlib.sha1()
    for name in image_names:
        p = Path(root) / (Path(name).stem + ".npy")
        if not p.is_file():
            return None
        h.update(name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_depth_cache(world: WorldInfo, cache_root, *, backend: str = DEFAULT_BACKEND,
                      device: str | None = None, log=print, limit: int | None = None) -> dict:
    """Run the depth network over every keyframe not yet cached. Resumable:
    an existing map with a matching input sha1 is kept."""
    import cv2  # noqa: F401 -- imported before torch on the main thread (loader-lock rule)
    import torch

    from tower.world_builder import dense

    root = depth_dir(cache_root, world.world_id)
    root.mkdir(parents=True, exist_ok=True)
    mpath = root / "manifest.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8")) if mpath.is_file() else {}
    spec = dense.make_backend(backend)
    model_id = getattr(spec, "model_id", None)
    cam = world.canonical_camera
    fov_x = float(np.degrees(2.0 * np.arctan(cam["width"] / (2.0 * cam["fx"]))))
    params = {"version": DEPTH_CACHE_VERSION, "backend": backend, "model_id": model_id,
              "resolution_level": RESOLUTION_LEVEL, "fov_x_deg": round(fov_x, 6),
              "use_fp16": True, "mask": "moge-mask->NaN", "dtype": "float16"}
    if manifest.get("params") != params:
        manifest = {"params": params, "frames": {}}
    manifest.update({"world_id": world.world_id, "session_id": world.session_id, "camera": cam,
                     "complete": False})
    frames = manifest.setdefault("frames", {})
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = None

    def load_model():
        from moge.model.v2 import MoGeModel

        return dense.load_hub_weights(spec.name, model_id,
                                      lambda: MoGeModel.from_pretrained(model_id)).to(device).eval()

    t0 = time.time()
    done = 0
    todo = range(world.n) if limit is None else range(min(limit, world.n))
    for i in todo:
        name = world.image_name(i)
        src, kind = world.canonical_image_source(i)
        if src is None:
            frames[name] = {"index": i, "ok": False, "why": kind}
            continue
        sha = _sha1(src)
        out = root / (Path(name).stem + ".npy")
        rec = frames.get(name)
        if rec and rec.get("ok") and rec.get("input_sha1") == sha and out.is_file():
            continue
        image, kind = world.canonical_image(i)
        if image is None:
            frames[name] = {"index": i, "ok": False, "why": kind}
            continue
        if model is None:  # loaded only when a map is actually missing
            model = load_model()
        rgb = image[:, :, ::-1].copy()
        t = torch.tensor(rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
        with torch.no_grad():
            res = model.infer(t, resolution_level=RESOLUTION_LEVEL, apply_mask=False, fov_x=fov_x,
                              use_fp16=True)
        z = res["points"][..., 2].float().cpu().numpy()
        mask = res.get("mask")
        if mask is not None:
            z = np.where(mask.cpu().numpy().astype(bool), z, np.nan)
        np.save(out, z.astype(np.float16))
        valid = np.isfinite(z)
        frames[name] = {"index": i, "ok": True, "input_kind": kind, "input_sha1": sha,
                        "shape": list(z.shape), "valid_fraction": round(float(valid.mean()), 4),
                        "median_m": round(float(np.nanmedian(z)), 4) if valid.any() else None}
        done += 1
        if done % 50 == 0:
            log(f"[depth] {world.world_id[:8]} {done} new maps, {time.time() - t0:.0f}s")
            mpath.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    # complete = EVERY keyframe has a map (a failed frame is not complete;
    # review V2 L3), and the content digest pins the maps themselves.
    failed = sorted(world.image_name(i) for i in range(world.n)
                    if not (frames.get(world.image_name(i)) or {}).get("ok"))
    manifest["failed_frames"] = failed
    manifest["complete"] = not failed
    manifest["content_digest"] = content_digest(root, [world.image_name(i) for i in range(world.n)])
    manifest["new_maps_this_run"] = done
    manifest["seconds_this_run"] = round(time.time() - t0, 2)
    if device == "cuda":
        manifest["peak_vram_mb_this_run"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    mpath.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    model = None
    if device == "cuda":
        torch.cuda.empty_cache()
    return manifest
