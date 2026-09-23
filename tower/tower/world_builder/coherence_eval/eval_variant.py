"""The variant interchange format, and adapters into it.

A VARIANT is any reconstruction of a frozen world's keyframes: the saved world
itself (the baseline), a COLMAP incremental re-solve, GLOMAP with other pairs,
a learned SLAM, a pose-graph variant... Every one is evaluated by the SAME
code (`metrics.py`) against the SAME per-world caches, so an experiment only
has to emit this format (or a COLMAP sparse model, which is adapted here).

ON DISK: a directory holding

``reconstruction.json`` (required)::

    {
      "format": "wb-coherence-variant/1",
      "world_id": "<32 hex>",             # the frozen world this reconstructs
      "session_id": "<32 hex>",
      "meta": {                            # free-form provenance, copied to metrics.json
        "variant": "glomap-exhaustive",    # short name
        "source_world": "<world dir or id>",
        "code_commit": "<git sha>", "params": {...},
        "runtime": {"wall_s": ..., "peak_rss_mb": ..., "peak_vram_mb": ...},
        "notes": "..."
      },
      "image_space": "canonical" | "raw" | "custom",
          # which pixels obs_uv are in: "canonical" = solve/<sid>/images
          # (undistorted pinhole, 359x639); "raw" = the 360x640 capture frames
          # with the session distortion; "custom" = anything else (depth
          # sampling then goes through the variant's own camera, approximately)
      "cameras": {"default": {"model": "PINHOLE", "width": 359, "height": 639,
                              "params": [fx, fy, cx, cy]}},
          # COLMAP model names and parameter order; also PINHOLE_RADTAN
          # (fx fy cx cy k1 k2 p1 p2 k3). Used for reprojection only.
      "keyframes": [                       # POSED keyframes only; absent = not posed
        {"keyframe_id": "<sid>:00000042",  # or "image": "00000042.jpg", or "index": 1
         "T_world_camera": [16 floats, row-major 4x4]  (or a 4x4 nested list),
         "component": "0",                 # frame id; poses in different components
                                           # are different gauges. Default "0".
         "status": "published" | "posed",  # published = the variant would show it;
                                           # posed = it has a pose the variant does
                                           # not trust. Default "published".
         "camera": "default"}
      ],
      "segments": [                        # optional; per-segment Sim(3) placements
        {"id": "3", "keyframe_ids": [...], "scale": 1.0, "state": "registered"}
      ]
    }

``points.npz`` (optional; enables reprojection and depth-based scale drift)::

    xyz            (N, 3) float    points, in the frame of their component
    point_component (N,) str/int   optional; else the component of the observers
    keyframe_ids   (K,) str        index space for obs_keyframe
    obs_keyframe   (M,) int        index into keyframe_ids
    obs_point      (M,) int        index into xyz
    obs_uv         (M, 2) float    pixel of the observation, in `image_space`

POSES are T_world_camera with OpenCV camera axes (x right, y down, z forward),
the world convention of `world.json`. The frame and scale are arbitrary: every
metric is gauge-invariant (Sim(3) per component).

KEYFRAME MAPPING: a row may name its keyframe by `keyframe_id` (preferred), by
image name (``00000042.jpg`` -- the name COLMAP knows it by, the file name in
``solve/<sid>/images``), or by capture-order `index`. See `eval_world`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tower.world_builder.coherence_eval.eval_world import CANONICAL, CUSTOM, RAW, WorldInfo

FORMAT = "wb-coherence-variant/1"
PUBLISHED = "published"
POSED = "posed"

# The product's publish rule, mirrored for COLMAP models so that a re-solve
# and the saved world are held to the same bar: `global_solve.MIN_IMAGE_OBSERVATIONS`
# (a camera with fewer 3-D observations is not a measurement) and
# `global_solve.MIN_MODEL_IMAGES` (a model smaller than this is not a component).
DEFAULT_PUBLISH_MIN_OBSERVATIONS = 30
DEFAULT_PUBLISH_MIN_MODEL_IMAGES = 5


@dataclass
class Variant:
    name: str
    world_id: str | None
    session_id: str | None
    poses: dict[str, np.ndarray] = field(default_factory=dict)       # kid -> 4x4 T_world_camera
    component: dict[str, str] = field(default_factory=dict)          # kid -> component id
    status: dict[str, str] = field(default_factory=dict)             # kid -> published|posed
    camera_of: dict[str, str] = field(default_factory=dict)          # kid -> camera id
    cameras: dict[str, dict] = field(default_factory=dict)
    image_space: str = CANONICAL
    component_stated: bool = True
    # points / observations (optional)
    xyz: np.ndarray | None = None
    point_component: np.ndarray | None = None                        # (N,) str
    obs_keyframe_ids: list[str] | None = None
    obs_keyframe: np.ndarray | None = None
    obs_point: np.ndarray | None = None
    obs_uv: np.ndarray | None = None
    segments: list[dict] | None = None
    meta: dict = field(default_factory=dict)

    @property
    def has_observations(self) -> bool:
        return self.xyz is not None and self.obs_point is not None and len(self.obs_point) > 0

    def published_ids(self) -> list[str]:
        return [k for k, s in self.status.items() if s == PUBLISHED]


# ---------------------------------------------------------------------------
# Pose helpers


def T_from_Rt_cw(R_cw, t_cw) -> np.ndarray:
    """COLMAP cam_from_world (R, t) -> T_world_camera."""
    R_cw = np.asarray(R_cw, dtype=np.float64).reshape(3, 3)
    t_cw = np.asarray(t_cw, dtype=np.float64).reshape(3)
    T = np.eye(4)
    T[:3, :3] = R_cw.T
    T[:3, 3] = -R_cw.T @ t_cw
    return T


def quat_wxyz_to_R(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _as_T(value) -> np.ndarray:
    a = np.asarray(value, dtype=np.float64)
    if a.size == 16:
        return a.reshape(4, 4)
    if a.size == 12:
        T = np.eye(4)
        T[:3, :] = a.reshape(3, 4)
        return T
    raise ValueError(f"T_world_camera must have 16 (or 12) numbers, got {a.size}")


# ---------------------------------------------------------------------------
# Interchange IO


def save_variant(variant: Variant, out_dir) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for kid in sorted(variant.poses):
        rows.append({
            "keyframe_id": kid,
            "T_world_camera": [round(float(v), 12) for v in variant.poses[kid].reshape(-1)],
            "component": str(variant.component.get(kid, "0")),
            "status": variant.status.get(kid, PUBLISHED),
            "camera": variant.camera_of.get(kid, "default"),
        })
    doc = {
        "format": FORMAT,
        "world_id": variant.world_id,
        "session_id": variant.session_id,
        "meta": dict(variant.meta, variant=variant.name),
        "image_space": variant.image_space,
        "component_stated": bool(variant.component_stated),
        "cameras": variant.cameras,
        "keyframes": rows,
    }
    if variant.segments is not None:
        doc["segments"] = variant.segments
    with open(out_dir / "reconstruction.json", "w", encoding="utf-8") as handle:
        json.dump(doc, handle, indent=1, sort_keys=True)
    if variant.has_observations:
        arrays = {
            "xyz": np.asarray(variant.xyz, dtype=np.float64),
            "keyframe_ids": np.asarray(variant.obs_keyframe_ids, dtype=str),
            "obs_keyframe": np.asarray(variant.obs_keyframe, dtype=np.int32),
            "obs_point": np.asarray(variant.obs_point, dtype=np.int32),
            "obs_uv": np.asarray(variant.obs_uv, dtype=np.float64),
        }
        if variant.point_component is not None:
            arrays["point_component"] = np.asarray(variant.point_component, dtype=str)
        np.savez_compressed(out_dir / "points.npz", **arrays)
    return out_dir


def load_variant(path, world: WorldInfo | None = None) -> Variant:
    """Read an interchange directory. Keyframe references are resolved through
    `world` when given (image names, indices); unresolved rows are counted in
    ``meta['unresolved_keyframes']`` and dropped."""
    path = Path(path)
    doc = json.loads((path / "reconstruction.json").read_text(encoding="utf-8"))
    if doc.get("format") != FORMAT:
        raise ValueError(f"{path}: format {doc.get('format')!r}, expected {FORMAT!r}")
    meta = dict(doc.get("meta") or {})
    v = Variant(name=str(meta.get("variant") or path.name), world_id=doc.get("world_id"),
                session_id=doc.get("session_id"), meta=meta,
                image_space=doc.get("image_space") or CANONICAL,
                cameras=dict(doc.get("cameras") or {}), segments=doc.get("segments"))
    stated_any = False
    unresolved = 0
    for row in doc.get("keyframes") or []:
        key = row.get("keyframe_id", row.get("image", row.get("index")))
        kid = world.resolve_keyframe(key) if world is not None else (str(key) if key is not None else None)
        if kid is None:
            unresolved += 1
            continue
        v.poses[kid] = _as_T(row["T_world_camera"])
        if "component" in row:
            stated_any = True
        v.component[kid] = str(row.get("component", "0"))
        v.status[kid] = str(row.get("status", PUBLISHED))
        v.camera_of[kid] = str(row.get("camera", "default"))
    v.component_stated = bool(doc.get("component_stated", stated_any))
    if unresolved:
        v.meta["unresolved_keyframes"] = unresolved
    pts = path / "points.npz"
    if pts.is_file():
        with np.load(pts, allow_pickle=False) as z:
            v.xyz = np.asarray(z["xyz"], dtype=np.float64).reshape(-1, 3)
            ids = [str(s) for s in z["keyframe_ids"]]
            if world is not None:
                ids = [world.resolve_keyframe(s) or s for s in ids]
            v.obs_keyframe_ids = ids
            v.obs_keyframe = np.asarray(z["obs_keyframe"], dtype=np.int64)
            v.obs_point = np.asarray(z["obs_point"], dtype=np.int64)
            v.obs_uv = np.asarray(z["obs_uv"], dtype=np.float64).reshape(-1, 2)
            if "point_component" in z.files:
                v.point_component = np.asarray([str(s) for s in z["point_component"]])
    return v


# ---------------------------------------------------------------------------
# Adapter: a saved world (derived tree + global solution)


def _derived_world_poses(world: WorldInfo):
    """Published derived poses composed into their reference segment's frame.

    Returns (poses, component, segment_rows, local) where `local` holds the
    chain-built segments whose placement is not `registered` (their poses are
    in the segment's own frame and are NOT composited with anything)."""
    ddir = world.derived_dir
    rows = json.loads((ddir / "poses.json").read_text(encoding="utf-8"))["poses"]
    placements = {}
    pl_path = ddir / "placements.json"
    if pl_path.is_file():
        for p in json.loads(pl_path.read_text(encoding="utf-8")).get("placements") or []:
            placements[int(p["segment_index"])] = p
    poses, component, local = {}, {}, {}
    for r in rows:
        if r.get("status") not in ("solved", "anchor") or r.get("translation") is None:
            continue
        seg = int(r["segment_index"])
        T_sc = np.eye(4)
        T_sc[:3, :3] = quat_wxyz_to_R(r["rotation"])
        T_sc[:3, 3] = np.asarray(r["translation"], dtype=np.float64)
        p = placements.get(seg)
        if p is not None and p.get("state") == "registered":
            S = np.eye(4)
            s = float(p.get("scale") or 1.0)
            S[:3, :3] = s * quat_wxyz_to_R(p["rotation_wxyz"])
            S[:3, 3] = np.asarray(p["translation"], dtype=np.float64)
            T = S @ T_sc
            # remove the scale from the rotation block (keep T a rigid pose)
            T[:3, :3] = T[:3, :3] / s
            poses[r["keyframe_id"]] = T
            component[r["keyframe_id"]] = f"ref{int(p['reference_segment'])}"
        else:
            local.setdefault(seg, {})[r["keyframe_id"]] = T_sc
    return poses, component, placements, local


def _rigid_fit(A: np.ndarray, B: np.ndarray):
    """R, t minimising |R A + t - B| (Kabsch, no scale)."""
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def variant_from_world(world: WorldInfo, *, name: str = "saved-world") -> Variant:
    """The saved world as a variant: what the product published.

    With a global solution (`solve/<sid>/solution.json` + `.npz`): poses,
    components, points and observations come from the solution, in its own
    per-component frame; a keyframe is `published` iff the derived tree has a
    solved/anchor row for it inside a `registered` placement (the product's
    support floor and component floor decide that), else `posed`. The derived
    poses are checked against the solution per component (they must differ by
    one rigid transform) and the residual is recorded in meta.

    Chain-built segments that the solve did not place keep LOCAL poses in the
    derived tree; each one with >= 2 posed keyframes becomes its own component
    ``local<seg>`` with status `posed` (a one-keyframe local segment is an
    identity by definition, not a measurement, and is left unposed).

    Without a solution: the derived poses composed through placements, one
    component per reference segment, no observations.
    """
    v = Variant(name=name, world_id=world.world_id, session_id=world.session_id,
                image_space=CANONICAL)
    v.meta = {"variant": name, "source_world": str(world.world_dir), "adapter": "world"}
    d_poses, d_comp, placements, local = ({}, {}, {}, {})
    if (world.derived_dir / "poses.json").is_file():
        d_poses, d_comp, placements, local = _derived_world_poses(world)
    sol_path = world.solve_dir / "solution.json"
    npz_path = world.solve_dir / "solution.npz"
    if sol_path.is_file() and npz_path.is_file():
        sol = json.loads(sol_path.read_text(encoding="utf-8"))
        v.meta.update({"solver": sol.get("solver"), "solved_at": sol.get("solved_at"),
                       "input_digest": sol.get("input_digest"), "solve_timing": sol.get("timing"),
                       "pose_source": "solve/solution.json (+ derived publish state)"})
        cam = sol.get("camera") or world.canonical_camera
        v.cameras = {"default": {"model": "PINHOLE", "width": int(cam["width"]), "height": int(cam["height"]),
                                 "params": [float(cam["fx"]), float(cam["fy"]), float(cam["cx"]), float(cam["cy"])]}}
        for kid, e in sol["poses"].items():
            v.poses[kid] = T_from_Rt_cw(e["rotation"], e["translation"])
            v.component[kid] = str(int(e["component"]))
            v.camera_of[kid] = "default"
            v.status[kid] = PUBLISHED if kid in d_poses else POSED
        if not d_poses:
            # no derived tree to say what was published: apply the product's floor
            for kid, e in sol["poses"].items():
                v.status[kid] = PUBLISHED if int(e.get("observations") or 0) >= DEFAULT_PUBLISH_MIN_OBSERVATIONS else POSED
        # derived vs solution consistency, per component
        checks = {}
        for c in sorted(set(v.component.values())):
            ids = [k for k in d_poses if v.component.get(k) == c]
            if len(ids) < 3:
                continue
            A = np.array([v.poses[k][:3, 3] for k in ids])
            B = np.array([d_poses[k][:3, 3] for k in ids])
            R, t = _rigid_fit(A, B)
            res = np.linalg.norm((R @ A.T).T + t - B, axis=1)
            ext = np.percentile(np.linalg.norm(A - np.median(A, 0), axis=1), 90) or 1.0
            rot = [np.degrees(np.arccos(np.clip((np.trace((R @ v.poses[k][:3, :3]).T @ d_poses[k][:3, :3]) - 1) / 2, -1, 1)))
                   for k in ids]
            checks[c] = {"keyframes": len(ids), "max_centre_residual_rel_p90": float(res.max() / ext),
                         "max_rotation_residual_deg": float(max(rot))}
        v.meta["derived_vs_solution"] = checks
        with np.load(npz_path, allow_pickle=False) as z:
            v.xyz = np.asarray(z["xyz"], dtype=np.float64)
            v.point_component = np.asarray([str(int(c)) for c in z["component"]])
            obs = np.asarray(z["observations"]).reshape(-1, 3)
            xy = np.asarray(z["observation_xy"]).reshape(-1, 2) if "observation_xy" in z.files else None
        v.obs_keyframe_ids = list(sol["keyframe_ids"])
        if xy is not None and len(xy) == len(obs):
            v.obs_keyframe = obs[:, 0].astype(np.int64)
            v.obs_point = obs[:, 2].astype(np.int64)
            v.obs_uv = xy.astype(np.float64)
        else:
            v.meta["observations_missing"] = "solution.npz has no observation_xy"
    elif d_poses:
        v.meta["pose_source"] = "derived/poses.json composed through placements.json"
        for kid, T in d_poses.items():
            v.poses[kid] = T
            v.component[kid] = d_comp[kid]
            v.status[kid] = PUBLISHED
            v.camera_of[kid] = "default"
        v.cameras = {"default": {"model": "PINHOLE", **{k: world.canonical_camera[k] for k in ("width", "height")},
                                 "params": [world.canonical_camera[k] for k in ("fx", "fy", "cx", "cy")]}}
    else:
        v.meta["pose_source"] = "none (world has neither a solution nor a derived tree)"
    local_used = {}
    for seg, members in sorted(local.items()):
        fresh = {k: T for k, T in members.items() if k not in v.poses}
        if len(fresh) < 2:
            continue
        for kid, T in fresh.items():
            v.poses[kid] = T
            v.component[kid] = f"local{seg}"
            v.status[kid] = POSED
            v.camera_of[kid] = "default"
        local_used[str(seg)] = len(fresh)
    v.meta["local_chain_segments"] = local_used
    if placements:
        segs = []
        by_seg: dict[int, list[str]] = {}
        for k in world.keyframes:
            by_seg.setdefault(int(k.get("segment_index", -1)), []).append(k["keyframe_id"])
        for seg in sorted(placements):
            p = placements[seg]
            segs.append({"id": str(seg), "state": p.get("state"), "scale": p.get("scale"),
                         "reference_segment": p.get("reference_segment"),
                         "keyframe_ids": by_seg.get(seg, [])})
        v.segments = segs
    return v


# ---------------------------------------------------------------------------
# Adapter: a COLMAP sparse model (or a directory of numbered models)


def _colmap_model_dirs(path: Path) -> list[tuple[str, Path]]:
    def is_model(p: Path) -> bool:
        return (p / "images.bin").is_file() or (p / "images.txt").is_file()

    if is_model(path):
        return [("0", path)]
    subs = sorted((p for p in path.iterdir() if p.is_dir() and is_model(p)),
                  key=lambda p: (not p.name.isdigit(), int(p.name) if p.name.isdigit() else 0, p.name))
    if not subs and (path / "sparse").is_dir():
        return _colmap_model_dirs(path / "sparse")
    return [(p.name, p) for p in subs]


def is_colmap_path(path) -> bool:
    path = Path(path)
    return path.is_dir() and bool(_colmap_model_dirs(path))


def variant_from_colmap(path, world: WorldInfo, *, name: str | None = None,
                        publish_min_observations: int = DEFAULT_PUBLISH_MIN_OBSERVATIONS,
                        publish_min_model_images: int = DEFAULT_PUBLISH_MIN_MODEL_IMAGES,
                        image_space: str | None = None) -> Variant:
    """A COLMAP reconstruction (one model, or numbered sub-models) as a variant.

    Each model is one component, named by its directory ("0", "1", ...).
    Images map to keyframes by file name (see `eval_world`); unmatched names
    are counted in meta. A registered image is `published` iff its model has
    >= `publish_min_model_images` registered images and the image has >=
    `publish_min_observations` 3-D observations -- the product's own publish
    rule, so a re-solve is held to the same bar as the saved world. Pass 0/0
    to publish every registered image.

    image_space: inferred from the camera size unless given -- the world's
    canonical size -> "canonical", the raw capture size -> "raw", else "custom".
    """
    import pycolmap

    path = Path(path)
    v = Variant(name=name or path.name, world_id=world.world_id, session_id=world.session_id)
    v.meta = {"variant": v.name, "source_world": str(world.world_dir), "adapter": "colmap",
              "colmap_path": str(path),
              "publish_rule": {"min_observations": publish_min_observations,
                               "min_model_images": publish_min_model_images}}
    xyz, pcomp, ok_, op_, ouv = [], [], [], [], []
    obs_ids = world.keyframe_ids
    unmatched = 0
    sizes = set()
    base = 0
    for comp, mdir in _colmap_model_dirs(path):
        rec = pycolmap.Reconstruction(str(mdir))
        n_reg = int(rec.num_reg_images())
        cam_ids = {}
        for cid, cam in rec.cameras.items():
            key = f"{comp}:{cid}"
            v.cameras[key] = {"model": cam.model.name, "width": int(cam.width), "height": int(cam.height),
                              "params": [float(p) for p in cam.params]}
            cam_ids[cid] = key
            sizes.add((int(cam.width), int(cam.height)))
        image_kf = {}
        for image in rec.images.values():
            if not image.has_pose:
                continue
            kid = world.resolve_keyframe(image.name)
            if kid is None:
                unmatched += 1
                continue
            cfw = image.cam_from_world() if callable(image.cam_from_world) else image.cam_from_world
            T = T_from_Rt_cw(np.asarray(cfw.rotation.matrix()), np.asarray(cfw.translation))
            if kid in v.poses:
                # the same keyframe in two models: keep the larger model's
                if n_reg <= v.meta.setdefault("_nreg", {}).get(v.component[kid], 0):
                    continue
            v.poses[kid] = T
            v.component[kid] = comp
            v.camera_of[kid] = cam_ids.get(image.camera_id, f"{comp}:{image.camera_id}")
            published = (n_reg >= publish_min_model_images
                         and int(image.num_points3D) >= publish_min_observations)
            v.status[kid] = PUBLISHED if published else POSED
            image_kf[image.image_id] = (world.index_of[kid], image)
        v.meta.setdefault("_nreg", {})[comp] = n_reg
        n_pts = 0
        for _, point in rec.points3D.items():
            elements = [e for e in point.track.elements if e.image_id in image_kf]
            if not elements:
                continue
            xyz.append(np.asarray(point.xyz, dtype=np.float64))
            pcomp.append(comp)
            for e in elements:
                kf_index, image = image_kf[e.image_id]
                ok_.append(kf_index)
                op_.append(base + n_pts)
                ouv.append(np.asarray(image.points2D[e.point2D_idx].xy, dtype=np.float64))
            n_pts += 1
        base += n_pts
    v.meta.pop("_nreg", None)
    v.meta["unmatched_images"] = unmatched
    if xyz:
        v.xyz = np.asarray(xyz).reshape(-1, 3)
        v.point_component = np.asarray(pcomp)
        v.obs_keyframe_ids = obs_ids
        v.obs_keyframe = np.asarray(ok_, dtype=np.int64)
        v.obs_point = np.asarray(op_, dtype=np.int64)
        v.obs_uv = np.asarray(ouv, dtype=np.float64).reshape(-1, 2)
    if image_space is None:
        can = world.canonical_camera
        raw = world.raw_camera
        if sizes and can and all(s == (int(can["width"]), int(can["height"])) for s in sizes):
            image_space = CANONICAL
        elif sizes and raw and all(s == (int(raw["width"]), int(raw["height"])) for s in sizes):
            image_space = RAW
        else:
            image_space = CUSTOM
    v.image_space = image_space
    return v


def load_any(path, world: WorldInfo, **colmap_kwargs) -> Variant:
    """Interchange dir, COLMAP model, or a world-shaped directory."""
    path = Path(path)
    if (path / "reconstruction.json").is_file():
        return load_variant(path, world)
    if (path / "world.json").is_file():
        from tower.world_builder.coherence_eval.eval_world import open_world

        return variant_from_world(open_world(path, world.session_id), name=path.name)
    if is_colmap_path(path):
        return variant_from_colmap(path, world, **colmap_kwargs)
    raise ValueError(f"{path}: not a variant directory (reconstruction.json), a COLMAP model, or a world")
