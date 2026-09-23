"""Forensic renders of a world's layers from the coherence viewpoint set.

Read-only. Given a world directory (or a variant directory of the same
shape: ``solve/<sid>/``, ``surface/<sid>/``, ``sessions/<sid>/``), this
renders, from every view of `viewpoints.viewpoint_set`:

* ``cameras``      -- the global solve's camera frusta and the main
                      component's capture-order path, coloured by component;
* ``sparse``       -- the global solve's sparse points, coloured by component;
* ``sparse_time``  -- the same points coloured by the capture order of the
                      first keyframe that observed them (turbo: early = blue);
* ``surface``      -- the published surface level, grey Lambert shading from
                      the geometric face normals (headlight);
* ``surface_rgb``  -- the same level with its stored vertex colours.

Optionally, when a region-label CSV (forensic annotation, never an input to
reconstruction) is given, ``cameras_region`` and ``sparse_region`` colour by
the region of the keyframe (for points: of their first observing keyframe).
Any label gets a colour (`region_rgb`): the known vocabulary keeps fixed
colours, anything else a deterministic colour derived from the label text.

MAIN COMPONENT ONLY BY DEFAULT. Cameras and points of the viewpoint frame's
component are drawn; cameras below the observation floor (unsupported: the
product does not publish them) are drawn grey. Minor components are opt-in
(`include_minor=True`) and are then drawn in THEIR OWN solver gauge, which is
not registered to the main one -- the index says so. Component colours never
wrap (`component_rgb`).

WHAT IS AND IS NOT A RENDERER CHOICE. Surfaces are ray cast exactly
(Open3D/Embree ``RaycastingScene``) against the stored triangles; nothing is
smoothed, filled or re-coloured. TOP/ORBIT views cull back faces (a face is
back-facing when its stored winding normal points away from the viewer) so an
outside view can see into a room -- the view dict says so (``cull_backfaces``)
and TRAJ views never cull.

Nothing here writes anywhere but ``out_dir``.
"""

from __future__ import annotations

import colorsys
import csv
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tower.world_builder.coherence_eval import viewpoints as vp

BG = (24, 24, 28)
COMPONENT_RGB = [(120, 170, 255), (255, 150, 40), (60, 220, 90), (240, 70, 200),
                 (240, 230, 60), (60, 230, 230), (230, 60, 60), (170, 120, 255)]
REGION_RGB = {
    "desk": (80, 140, 255), "closet": (255, 140, 30), "bathroom": (40, 230, 230),
    "bed": (70, 220, 70), "hallway": (200, 90, 255), "other": (200, 200, 200),
    "unclear": (110, 110, 110), None: (70, 70, 70),
}
UNSUPPORTED_RGB = (125, 125, 125)
LAYERS_DEFAULT = ("cameras", "sparse", "sparse_time", "surface", "surface_rgb")
MINOR_GAUGE_NOTE = ("minor components are drawn in their OWN solver gauge, not registered to "
                    "the main component; positions relative to the main component mean nothing")


def component_rgb(index) -> tuple:
    """Colour of component `index` (0 = main). The first 8 are fixed; beyond
    them, golden-angle hues -- a palette that never wraps silently."""
    i = int(index)
    if 0 <= i < len(COMPONENT_RGB):
        return COMPONENT_RGB[i]
    h = (0.61803398875 * i) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.95)
    return (int(r * 255), int(g * 255), int(b * 255))


def region_rgb(label) -> tuple:
    """Evaluation-only colour of a region label. Known labels keep fixed
    colours; any other label gets a deterministic colour from its text."""
    if label in REGION_RGB:
        return REGION_RGB[label]
    h = int(hashlib.sha1(str(label).encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    r, g, b = colorsys.hsv_to_rgb(h, 0.55, 0.9)
    return (int(r * 255), int(g * 255), int(b * 255))


# ---------------------------------------------------------------------------
# loading (read-only)
# ---------------------------------------------------------------------------


@dataclass
class WorldLayers:
    world_dir: Path
    session_id: str
    solution: dict
    poses: dict                 # kid -> 4x4 T_world_camera
    component_of: dict          # kid -> component
    capture_order: list
    keyframe_ids: list          # solution order (observations index into this)
    xyz: np.ndarray
    point_component: np.ndarray
    point_first_kid: list
    observations: np.ndarray
    camera: dict
    session: dict
    surface_manifest: dict | None
    regions: dict = field(default_factory=dict)   # kid -> region label
    surface_dir: Path | None = None               # default: world_dir/surface/<sid>
    pose_observations: dict | None = None         # kid -> observation count (None: all supported)
    min_observations: int = 30

    def supported(self, kid) -> bool:
        if self.pose_observations is None:
            return kid in self.poses
        return int(self.pose_observations.get(kid) or 0) >= int(self.min_observations)

    def main_component(self):
        """Largest component by supported cameras (ties: lowest id)."""
        counts: dict = {}
        for k in self.poses:
            if self.supported(k):
                c = self.component_of.get(k)
                counts[c] = counts.get(c, 0) + 1
        if not counts:
            return None
        return sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))[0][0]

    @property
    def world_id(self) -> str:
        return self.world_dir.name

    def surface_level_path(self, level: int = 0) -> Path | None:
        """The PUBLISHED file of `level`, as the manifest names it."""
        if not self.surface_manifest:
            return None
        base = self.surface_dir or (self.world_dir / "surface" / self.session_id)
        for lv in self.surface_manifest.get("levels") or []:
            if int(lv.get("level", -1)) == int(level) and lv.get("file"):
                return base / lv["file"]
        return None

    def load_surface(self, level: int = 0):
        from tower.world_builder.surface import read_mesh_bytes

        p = self.surface_level_path(level)
        if p is None or not p.exists():
            return None
        return read_mesh_bytes(p.read_bytes())


def read_regions(csv_path) -> dict:
    out = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[row["keyframe_id"]] = (row.get("region") or "").strip() or None
    return out


def load_world(world_dir, session_id: str | None = None, regions_csv=None,
               surface_dir=None) -> WorldLayers:
    """`surface_dir` evaluates a surface built elsewhere (an experiment's
    output, same session) against this world's solve and keyframes."""
    world_dir = Path(world_dir)
    sid = session_id or vp.single_session_id(world_dir)
    solve = world_dir / "solve" / sid
    meta = json.loads((solve / "solution.json").read_text(encoding="utf-8"))
    with np.load(solve / "solution.npz") as z:
        xyz = np.asarray(z["xyz"], np.float64)
        pc = np.asarray(z["component"], np.int32)
        fk = np.asarray(z["first_keyframe"], np.int64)
        obs = np.asarray(z["observations"], np.int64).reshape(-1, 3)
    poses, comp = vp.poses_from_solution(meta)
    kids = list(meta["keyframe_ids"])
    order = vp.capture_order_from_keyframes(world_dir / "sessions" / sid / "keyframes.jsonl")
    session = json.loads((world_dir / "sessions" / sid / "session.json").read_text(encoding="utf-8"))
    sdir = Path(surface_dir) if surface_dir else world_dir / "surface" / sid
    sm = sdir / "manifest.json"
    manifest = json.loads(sm.read_text(encoding="utf-8")) if sm.exists() else None
    return WorldLayers(
        world_dir=world_dir, session_id=sid, solution=meta, poses=poses, component_of=comp,
        capture_order=order, keyframe_ids=kids, xyz=xyz, point_component=pc,
        point_first_kid=[kids[i] if 0 <= i < len(kids) else None for i in fk],
        observations=obs, camera=dict(meta["camera"]), session=session,
        surface_manifest=manifest,
        regions=read_regions(regions_csv) if regions_csv else {},
        surface_dir=Path(surface_dir) if surface_dir else None,
        pose_observations=vp.observations_from_solution(meta),
    )


def undistorted_photo(world: WorldLayers, kid: str):
    """The keyframe's stored image, undistorted into the solve's camera with
    the solve's own maps (`global_solve._undistort_maps`). RGB uint8 or None."""
    import cv2

    from tower.world_builder.global_solve import _undistort_maps

    seq = kid.split(":")[-1]
    p = world.world_dir / "sessions" / world.session_id / "images" / f"{seq}.jpg"
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        return None
    intr = SimpleNamespace(**{k: world.session["intrinsics"].get(k) for k in
                              ("fx", "fy", "cx", "cy", "dist_coeffs",
                               "calibrated_width", "calibrated_height")})
    h, w = img.shape[:2]
    try:
        m1, m2, (x, y, rw, rh), _cam = _undistort_maps(intr, w, h)
    except Exception:  # noqa: BLE001
        return None
    out = cv2.remap(img, m1, m2, cv2.INTER_LINEAR)[y:y + rh, x:x + rw]
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# rays and the mesh
# ---------------------------------------------------------------------------


def view_rays(view: dict, scale: float = 1.0):
    """(H, W, origins (N,3), dirs (N,3) unit) for a view, pixel centres."""
    W = max(1, int(round(view["width"] * scale)))
    H = max(1, int(round(view["height"] * scale)))
    T = vp.view_T_world_camera(view)
    R, C = T[:3, :3], T[:3, 3]
    u = (np.arange(W) + 0.5) / scale
    v = (np.arange(H) + 0.5) / scale
    uu, vv = np.meshgrid(u, v)
    if view["projection"] == "orthographic":
        x = (uu / view["width"] - 0.5) * 2.0 * view["half_width"]
        y = (vv / view["height"] - 0.5) * 2.0 * view["half_height"]
        o = C + np.stack([x, y, np.zeros_like(x)], -1).reshape(-1, 3) @ R.T
        d = np.tile(R[:, 2], (H * W, 1))
    else:
        x = (uu - view["cx"]) / view["fx"]
        y = (vv - view["cy"]) / view["fy"]
        dc = np.stack([x, y, np.ones_like(x)], -1).reshape(-1, 3)
        d = dc @ R.T
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        o = np.tile(C, (H * W, 1))
    return H, W, o, d


class MeshCaster:
    """Exact ray casting against a stored surface level (Open3D / Embree)."""

    def __init__(self, V, F, C=None):
        self.V = np.asarray(V, np.float32)
        self.F = np.asarray(F, np.int64)
        self.C = None if C is None else np.asarray(C, np.uint8)
        a = self.V[self.F[:, 1]] - self.V[self.F[:, 0]]
        b = self.V[self.F[:, 2]] - self.V[self.F[:, 0]]
        n = np.cross(a, b)
        self.face_n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)
        self.centroid = self.V[self.F].mean(axis=1)
        self._full = self._scene(np.arange(len(self.F)))

    def _scene(self, face_idx):
        import open3d as o3d

        sc = o3d.t.geometry.RaycastingScene()
        if len(face_idx):
            sc.add_triangles(o3d.core.Tensor(self.V),
                             o3d.core.Tensor(self.F[face_idx].astype(np.uint32)))
        return sc, np.asarray(face_idx)

    def front_faces(self, view) -> np.ndarray:
        T = vp.view_T_world_camera(view)
        if view["projection"] == "orthographic":
            to_view = -T[:3, 2][None, :]
        else:
            to_view = T[:3, 3][None, :] - self.centroid
        return np.nonzero((self.face_n * to_view).sum(1) > 0)[0]

    def cast(self, origins, dirs, scene=None):
        import open3d as o3d

        sc, idx = scene or self._full
        rays = np.concatenate([origins, dirs], 1).astype(np.float32)
        r = sc.cast_rays(o3d.core.Tensor(rays))
        t = r["t_hit"].numpy()
        prim = r["primitive_ids"].numpy().astype(np.int64)
        hit = np.isfinite(t)
        face = np.full(len(t), -1, np.int64)
        face[hit] = idx[prim[hit]]
        return t, face, r["primitive_uvs"].numpy()

    def render(self, view, scale=1.0, mode="shaded", cull=None):
        H, W, o, d = view_rays(view, scale)
        cull = view.get("cull_backfaces", False) if cull is None else cull
        scene = self._scene(self.front_faces(view)) if cull else None
        t, face, uv = self.cast(o, d, scene)
        img = np.empty((H * W, 3), np.float32)
        img[:] = BG
        hit = face >= 0
        if hit.any():
            n = self.face_n[face[hit]]
            lam = np.abs((n * d[hit]).sum(1))
            if mode == "shaded":
                g = 40 + 200 * lam
                img[hit] = np.stack([g, g, g * 0.97 + 6], 1)
            else:
                if self.C is None:
                    raise ValueError("mesh has no colours")
                w1, w2 = uv[hit, 0:1], uv[hit, 1:2]
                f = self.F[face[hit]]
                col = ((1 - w1 - w2) * self.C[f[:, 0]] + w1 * self.C[f[:, 1]]
                       + w2 * self.C[f[:, 2]])
                img[hit] = col
        return img.reshape(H, W, 3).clip(0, 255).astype(np.uint8), {
            "hit_fraction": float(hit.mean()), "faces_cast": int(len(scene[1]) if scene else len(self.F))}


# ---------------------------------------------------------------------------
# points and lines
# ---------------------------------------------------------------------------


def render_points(view, X, colors, radius_px: int = 1, scale: float = 1.0, img=None):
    W = int(round(view["width"] * scale))
    H = int(round(view["height"] * scale))
    if img is None:
        img = np.empty((H, W, 3), np.uint8)
        img[:] = BG
    X = np.asarray(X, np.float64).reshape(-1, 3)
    if not len(X):
        return img
    colors = np.asarray(colors, np.uint8).reshape(-1, 3)
    uv, z = vp.project(view, X)
    uv = uv * scale
    ok = np.isfinite(uv).all(1) & (z > 1e-6)
    uv, z, colors = uv[ok], z[ok], colors[ok]
    order = np.argsort(-z, kind="stable")  # far first, near overwrites
    uv, colors = uv[order], colors[order]
    ui = np.floor(uv[:, 0]).astype(np.int64)
    vi = np.floor(uv[:, 1]).astype(np.int64)
    r = int(radius_px)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy > r * r + r:
                continue
            x, y = ui + dx, vi + dy
            m = (x >= 0) & (x < W) & (y >= 0) & (y < H)
            img[y[m], x[m]] = colors[m]
    return img


def _segments_to_points(view, segs, colors, scale):
    """Sample 3D segments densely enough to look continuous in the view."""
    P0, P1 = segs[:, 0], segs[:, 1]
    uv0, z0 = vp.project(view, P0)
    uv1, z1 = vp.project(view, P1)
    with np.errstate(invalid="ignore"):
        L = np.linalg.norm((uv1 - uv0) * scale, axis=1)
    L = np.where(np.isfinite(L) & (z0 > 0) & (z1 > 0), L, 64.0)
    n = np.clip(np.ceil(L), 2, 1500).astype(int)
    pts, cols = [], []
    for i in range(len(segs)):
        s = np.linspace(0, 1, n[i])[:, None]
        pts.append(P0[i] * (1 - s) + P1[i] * s)
        cols.append(np.repeat(np.asarray(colors[i], np.uint8)[None], n[i], 0))
    return np.concatenate(pts), np.concatenate(cols)


def frustum_segments(T, size):
    R, C = T[:3, :3], T[:3, 3]
    corners = np.array([[-0.35, -0.6, 1.0], [0.35, -0.6, 1.0], [0.35, 0.6, 1.0], [-0.35, 0.6, 1.0]])
    Pw = C + (corners * size) @ R.T
    segs = [(C, p) for p in Pw] + [(Pw[i], Pw[(i + 1) % 4]) for i in range(4)]
    up_tick = C + (np.array([0, -1.0, 0]) * 0.5 * size) @ R.T
    segs.append((C, up_tick))
    return segs


def render_cameras(view, world: WorldLayers, frame_radius: float, color_by="component",
                   scale: float = 1.0, *, component=None, include_minor: bool = False):
    """Frusta + the main component's capture-order path.

    Default: the main component only (`component`, else the largest by
    supported cameras); unsupported cameras grey. `include_minor=True` adds
    the other components in their own gauge (see MINOR_GAUGE_NOTE)."""
    main_comp = world.main_component() if component is None else component
    segs, cols = [], []
    size = 0.04 * frame_radius
    for k in world.capture_order:
        T = world.poses.get(k)
        if T is None:
            continue
        comp = world.component_of.get(k)
        if comp != main_comp and not include_minor:
            continue
        if not world.supported(k):
            col = UNSUPPORTED_RGB
        elif color_by == "region":
            col = region_rgb(world.regions.get(k))
        else:
            col = component_rgb(0 if comp == main_comp else _minor_rank(world, comp, main_comp))
        for sgm in frustum_segments(T, size):
            segs.append(sgm)
            cols.append(col)
    path = [world.poses[k][:3, 3] for k in world.capture_order
            if k in world.poses and world.component_of.get(k) == main_comp and world.supported(k)]
    for a, b in zip(path[:-1], path[1:]):
        segs.append((a, b))
        cols.append((230, 230, 230))
    if not segs:
        return render_points(view, np.zeros((0, 3)), np.zeros((0, 3)), scale=scale)
    P, C = _segments_to_points(view, np.array(segs), cols, scale)
    return render_points(view, P, C, radius_px=0, scale=scale)


def _minor_rank(world: WorldLayers, comp, main_comp) -> int:
    """1-based colour rank of a minor component (by camera count, then id)."""
    counts: dict = {}
    for k in world.poses:
        c = world.component_of.get(k)
        if c != main_comp:
            counts[c] = counts.get(c, 0) + 1
    order = sorted(counts, key=lambda c: (-counts[c], str(c)))
    return 1 + order.index(comp) if comp in order else len(order) + 1


def turbo(x):
    import matplotlib

    cm = matplotlib.colormaps["turbo"]
    return (np.asarray(cm(np.clip(x, 0, 1)))[:, :3] * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# labels and sheets
# ---------------------------------------------------------------------------


def label(img, text, scale=0.5):
    import cv2

    out = np.ascontiguousarray(img.copy())
    lines = text.split("\n")
    for i, line in enumerate(lines):
        y = 16 + i * int(22 * scale / 0.5)
        cv2.putText(out, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def fit(img, w, h):
    import cv2

    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    r = cv2.resize(img, (max(1, int(iw * s)), max(1, int(ih * s))), interpolation=cv2.INTER_AREA)
    out = np.empty((h, w, 3), np.uint8)
    out[:] = BG
    y0, x0 = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    out[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return out


def contact_sheet(tiles, cols=6, tile=(320, 320), title=None):
    """tiles: list of (caption, image)."""
    rows = max(1, math.ceil(len(tiles) / cols))
    tw, th = tile
    top = 34 if title else 0
    sheet = np.empty((top + rows * th, cols * tw, 3), np.uint8)
    sheet[:] = (10, 10, 12)
    for i, (cap, im) in enumerate(tiles):
        r, c = divmod(i, cols)
        t = label(fit(im, tw - 4, th - 4), cap, 0.42)
        sheet[top + r * th + 2: top + r * th + 2 + t.shape[0], c * tw + 2: c * tw + 2 + t.shape[1]] = t
    if title:
        sheet[:top] = label(sheet[:top], title, 0.6)
    return sheet


def save_png(path, img):
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2BGR))
    if not ok:
        raise OSError(f"could not write {path}")
    return path


# ---------------------------------------------------------------------------
# in-situ evidence: the keyframe's own photo against the surface from its pose
# ---------------------------------------------------------------------------


def insitu_view(world: WorldLayers, kid: str) -> dict:
    """A TRAJ-style in-situ view of any posed keyframe (the solve's camera)."""
    cam = world.camera
    T = world.poses[kid]
    return {"name": f"KF_{kid.split(':')[-1]}", "family": "TRAJ_INSITU",
            "projection": "perspective", "T_world_camera": T.tolist(),
            "width": int(cam["width"]), "height": int(cam["height"]),
            "fx": float(cam["fx"]), "fy": float(cam["fy"]),
            "cx": float(cam["cx"]), "cy": float(cam["cy"]),
            "fov_y_deg": math.degrees(2 * math.atan(0.5 * cam["height"] / cam["fy"])),
            "cull_backfaces": False, "keyframe_id": kid}


def insitu_compare(world: WorldLayers, caster: "MeshCaster", view: dict):
    """[photo | surface shaded | surface rgb | 50/50 photo+shaded] for one
    in-situ view. The photo is the stored (redacted) keyframe undistorted
    into the solve camera, so pixel (u, v) of every panel is the same ray."""
    kid = view["keyframe_id"]
    photo = undistorted_photo(world, kid)
    shaded, st = caster.render(view, 1.0, "shaded", cull=False)
    rgb, _ = caster.render(view, 1.0, "rgb", cull=False)
    if photo is None or photo.shape[:2] != shaded.shape[:2]:
        photo = np.zeros_like(shaded)
    mix = (0.5 * photo.astype(np.float32) + 0.5 * shaded.astype(np.float32)).astype(np.uint8)
    reg = world.regions.get(kid) or "?"
    tiles = [label(photo, f"photo {kid.split(':')[-1]} {reg}", 0.4),
             label(shaded, f"surface cover {st['hit_fraction']:.2f}", 0.4),
             label(rgb, "surface rgb", 0.4), label(mix, "photo+surface", 0.4)]
    return np.concatenate(tiles, axis=1), st


def region_insitu_keyframes(world: WorldLayers, per_region: int = 3) -> list:
    """Evaluation-only choice of extra in-situ keyframes: per region label,
    the posed main-component keyframes at within-region capture quantiles
    (i + 0.5) / per_region. Region labels choose WHERE TO LOOK, nothing else."""
    main = world.main_component()
    by = {}
    for k in world.capture_order:
        if k in world.poses and world.component_of.get(k) == main and world.supported(k):
            by.setdefault(world.regions.get(k), []).append(k)
    out = []
    for reg in sorted(by, key=lambda r: str(r)):
        ks = by[reg]
        for i in range(per_region):
            out.append((reg, ks[min(len(ks) - 1, int((i + 0.5) / per_region * len(ks)))]))
    return out


# ---------------------------------------------------------------------------
# per-keyframe surface metrics
# ---------------------------------------------------------------------------


def keyframe_surface_metrics(world: WorldLayers, caster: "MeshCaster", kid: str, *,
                             voxel: float, scale: float = 0.5,
                             doubled_range_vox=(2.0, 20.0), doubled_cos=0.8,
                             sparse_rel=0.10) -> dict:
    """What the published surface looks like FROM THIS KEYFRAME'S OWN POSE.

    coverage       fraction of the frame's rays (at `scale`) that hit the surface
    front          of those hits, the fraction whose face points at the camera
    doubled        of front-facing hits, the fraction behind which, 2..20 voxels
                   further along the same ray, lies ANOTHER front-facing face
                   within ~37 deg of parallel: the signature of one wall fused
                   twice at an offset (it also fires on genuinely layered
                   structure -- a poster on a wall, clothes before a closet
                   back -- so compare regions, not absolutes)
    sparse_*       the sparse points THIS keyframe observed (solve observations)
                   cast along their own rays: relative depth error of the
                   surface against them. in_front = the surface is > 10%
                   nearer than the point (occluder/floater/doubled layer);
                   behind_or_missing = > 10% farther, or no surface at all.
    """
    view = insitu_view(world, kid)
    H, Wd, o, d = view_rays(view, scale)
    t, face, _uv = caster.cast(o, d)
    hit = face >= 0
    res = {"kid": kid, "coverage": float(hit.mean())}
    if hit.any():
        n = caster.face_n[face[hit]]
        cosv = -(n * d[hit]).sum(1)
        front = cosv > 0
        res["front"] = float(front.mean())
        # continue each front-facing hit a hair past the surface
        hi = np.nonzero(hit)[0][front]
        eps = 0.5 * voxel
        o2 = o[hi] + d[hi] * (t[hi] + eps)[:, None]
        t2, f2, _ = caster.cast(o2, d[hi])
        ok2 = f2 >= 0
        dbl = np.zeros(len(hi), bool)
        if ok2.any():
            n1 = caster.face_n[face[hi[ok2]]]
            n2 = caster.face_n[f2[ok2]]
            gap = (t2[ok2] + eps) / voxel
            fr2 = -(n2 * d[hi[ok2]]).sum(1) > 0
            par = (n1 * n2).sum(1) > doubled_cos
            dbl[ok2] = fr2 & par & (gap >= doubled_range_vox[0]) & (gap <= doubled_range_vox[1])
        res["doubled"] = float(dbl.mean()) if len(dbl) else 0.0
    else:
        res["front"] = None
        res["doubled"] = None
    # sparse agreement
    ki = world.keyframe_ids.index(kid) if kid in world.keyframe_ids else -1
    m = world.observations[:, 0] == ki
    if m.sum():
        T = world.poses[kid]
        C = T[:3, 3]
        P = world.xyz[world.observations[m, 2]]
        v = P - C
        dist = np.linalg.norm(v, axis=1)
        good = dist > 1e-9
        v, dist = v[good] / dist[good, None], dist[good]
        ts, fs, _ = caster.cast(np.tile(C, (len(v), 1)), v)
        has = fs >= 0
        rel = np.where(has, (ts - dist) / dist, np.inf)
        res.update({
            "sparse_n": int(len(dist)),
            "sparse_hit": float(has.mean()),
            "sparse_abs_rel_med": float(np.median(np.abs(rel[has]))) if has.any() else None,
            "sparse_in_front": float((rel < -sparse_rel).mean()),
            "sparse_behind_or_missing": float((rel > sparse_rel).mean()),
            "sparse_agree": float((np.abs(rel) <= sparse_rel).mean()),
        })
    return res


# ---------------------------------------------------------------------------
# the driver
# ---------------------------------------------------------------------------


def render_world(world_dir, out_dir, *, session_id=None, viewpoints=None, regions_csv=None,
                 layers=LAYERS_DEFAULT, level: int = 0, orbit_scale: float = 1.0,
                 point_radius: int = 1, surface_dir=None, include_minor: bool = False,
                 log=print) -> dict:
    """Render `layers` of a world from its viewpoint set into `out_dir`.

    Layout: out_dir/<layer>/<view>.png, out_dir/sheets/<layer>.png,
    out_dir/index.json. `viewpoints` is a viewpoint-set dict (or path); by
    default the rule is applied to this world's own trajectory. For A/B
    renders of variants, pass a set transferred from the reference
    (`viewpoints.transfer_viewpoints`). Main component only unless
    `include_minor`.
    """
    out_dir = Path(out_dir)
    world = load_world(world_dir, session_id, regions_csv, surface_dir)
    if viewpoints is None:
        vs = vp.viewpoints_for_world(world.world_dir, world.session_id)
    elif isinstance(viewpoints, (str, Path)):
        vs = vp.load_viewpoints(viewpoints)
    else:
        vs = viewpoints
    r = float(vs["frame"]["radius"])
    main = world.main_component()
    layers = list(layers)
    if regions_csv and world.regions:
        for extra in ("cameras_region", "sparse_region"):
            if extra not in layers:
                layers.append(extra)
    index = {"world_dir": str(world.world_dir), "session_id": world.session_id,
             "viewpoint_rule": vs["rule"], "viewpoint_mode": vs.get("mode", "native"),
             "frame": vs["frame"], "layers": {},
             "main_component": main, "include_minor": bool(include_minor),
             "surface_level": level,
             "surface_file": (str(world.surface_level_path(level))
                              if world.surface_level_path(level) else None),
             "notes": [
                 "sparse/cameras: global solve (solve/<sid>/solution.*); "
                 + ("main component + " + MINOR_GAUGE_NOTE if include_minor
                    else "main component only; unsupported main cameras grey"),
                 "surface: published level from surface/<sid>/manifest.json; exact ray "
                 "cast; TOP/ORBIT cull back faces; TRAJ views do not",
             ]}
    caster = None
    mesh = None
    if any(l.startswith("surface") for l in layers):
        mesh = world.load_surface(level)
        if mesh is None:
            index["notes"].append(f"no published surface level {level}")
        else:
            V, F, C, _N = mesh
            t0 = time.time()
            caster = MeshCaster(V, F, C)
            log(f"surface level {level}: {len(V)} vertices, {len(F)} faces "
                f"(scene {time.time() - t0:.1f}s)")
    # points: the main component only unless include_minor
    keep = (np.ones(len(world.xyz), bool) if include_minor
            else np.asarray(world.point_component) == (main if main is not None else 0))
    X = world.xyz[keep]
    pcomp = np.asarray(world.point_component)[keep]
    pfirst = [k for k, m in zip(world.point_first_kid, keep) if m]
    kid_rank = {k: i / max(1, len(world.capture_order) - 1)
                for i, k in enumerate(world.capture_order)}
    minor_rank = {c: _minor_rank(world, c, main) for c in set(pcomp.tolist()) if c != main}
    pc_comp = np.array([component_rgb(0 if c == main else minor_rank[c]) for c in pcomp],
                       np.uint8).reshape(-1, 3)
    pc_time = turbo(np.array([kid_rank.get(k, 0.0) for k in pfirst]))
    pc_region = np.array([region_rgb(world.regions.get(k)) for k in pfirst],
                         np.uint8).reshape(-1, 3)
    index["points_drawn"] = int(len(X))
    index["points_not_drawn_minor"] = int((~keep).sum())
    index["component_colours"] = {str(main): list(component_rgb(0)),
                                  **{str(c): list(component_rgb(i)) for c, i in minor_rank.items()}}
    for layer in layers:
        tiles = []
        stats = {}
        t0 = time.time()
        for view in vs["views"]:
            s = orbit_scale if view["family"] in ("TOP", "ORBIT") else 1.0
            if layer == "cameras":
                img = render_cameras(view, world, r, "component", s, component=main,
                                     include_minor=include_minor)
            elif layer == "cameras_region":
                img = render_cameras(view, world, r, "region", s, component=main,
                                     include_minor=include_minor)
            elif layer == "sparse":
                img = render_points(view, X, pc_comp, point_radius, s)
            elif layer == "sparse_time":
                img = render_points(view, X, pc_time, point_radius, s)
            elif layer == "sparse_region":
                img = render_points(view, X, pc_region, point_radius, s)
            elif layer in ("surface", "surface_rgb"):
                if caster is None:
                    continue
                img, st = caster.render(view, s, "shaded" if layer == "surface" else "rgb")
                stats[view["name"]] = st
            else:
                raise ValueError(f"unknown layer {layer!r}")
            save_png(out_dir / layer / f"{view['name']}.png", img)
            cap = view["name"]
            if "keyframe_id" in view:
                cap += "\n" + view["keyframe_id"].split(":")[-1] + (" (subst)" if view.get("substituted") else "")
                reg = world.regions.get(view["keyframe_id"])
                if reg:
                    cap += " " + reg
            tiles.append((cap, img))
        if tiles:
            sheet = contact_sheet(tiles, cols=6, tile=(300, 300),
                                  title=f"{world.world_id[:8]} {layer}  ({vs['rule']})")
            save_png(out_dir / "sheets" / f"{layer}.png", sheet)
        index["layers"][layer] = {"views": [t[0].split("\n")[0] for t in tiles],
                                  "seconds": round(time.time() - t0, 2), "stats": stats}
        log(f"{layer}: {len(tiles)} views in {time.time() - t0:.1f}s")
    (out_dir).mkdir(parents=True, exist_ok=True)
    (out_dir / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    return index


def insitu_and_metrics(world_dir, out_dir, *, session_id=None, viewpoints=None,
                       regions_csv=None, level: int = 0, per_region: int = 3,
                       metrics_scale: float = 0.5, surface_dir=None,
                       include_minor: bool = False, log=print) -> dict:
    """In-situ photo-vs-surface comparisons and per-keyframe surface metrics.

    Writes out_dir/insitu_compare/<name>.png (+ sheets), and
    out_dir/surface_keyframe_metrics.csv (every posed keyframe of every
    component; minor components are NOT in the surface's frame, which the
    `component` column makes visible), plus a per-region summary JSON.
    """
    out_dir = Path(out_dir)
    world = load_world(world_dir, session_id, regions_csv, surface_dir)
    mesh = world.load_surface(level)
    if mesh is None:
        raise ValueError(f"no published surface level {level}")
    V, F, C, _N = mesh
    caster = MeshCaster(V, F, C)
    voxel = float((world.surface_manifest or {}).get("voxel") or 0.0) or 0.02 * float(
        np.linalg.norm(V.max(0) - V.min(0)))
    if viewpoints is None:
        vs = vp.viewpoints_for_world(world.world_dir, world.session_id)
    elif isinstance(viewpoints, (str, Path)):
        vs = vp.load_viewpoints(viewpoints)
    else:
        vs = viewpoints
    # comparisons: the rule's TRAJ keyframes, then the region picks
    picks = [(v["name"], v) for v in vs["views"] if v["family"] == "TRAJ_INSITU"]
    if world.regions:
        for reg, kid in region_insitu_keyframes(world, per_region):
            v = insitu_view(world, kid)
            picks.append((f"REGION_{reg}_{kid.split(':')[-1]}", v))
    tiles = []
    for name, v in picks:
        img, _st = insitu_compare(world, caster, v)
        save_png(out_dir / "insitu_compare" / f"{name}.png", img)
        tiles.append((name, img))
    for i in range(0, len(tiles), 12):
        sheet = contact_sheet(tiles[i:i + 12], cols=2, tile=(900, 400),
                              title=f"{world.world_id[:8]} in-situ: photo | surface | rgb | mix")
        save_png(out_dir / "sheets" / f"insitu_compare_{i // 12}.png", sheet)
    log(f"insitu: {len(tiles)} comparisons")
    # metrics for every posed keyframe
    rows = []
    t0 = time.time()
    rank = {k: i for i, k in enumerate(world.capture_order)}
    main = world.main_component()
    for kid in world.capture_order:
        if kid not in world.poses:
            continue
        if world.component_of.get(kid) != main and not include_minor:
            continue
        r = keyframe_surface_metrics(world, caster, kid, voxel=voxel, scale=metrics_scale)
        r.update({"capture_index": rank[kid], "component": world.component_of.get(kid),
                  "region": world.regions.get(kid), "supported": world.supported(kid)})
        rows.append(r)
    log(f"metrics: {len(rows)} keyframes in {time.time() - t0:.1f}s")
    cols = ["capture_index", "kid", "component", "supported", "region", "coverage", "front", "doubled",
            "sparse_n", "sparse_hit", "sparse_abs_rel_med", "sparse_agree", "sparse_in_front",
            "sparse_behind_or_missing"]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "surface_keyframe_metrics.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    summary = {}
    for r in rows:
        key = f"c{r['component']}:{r['region']}"
        summary.setdefault(key, []).append(r)
    agg = {}
    for key, rs in sorted(summary.items()):
        def med(field):
            vals = [x[field] for x in rs if x.get(field) is not None]
            return round(float(np.median(vals)), 4) if vals else None
        agg[key] = {"n": len(rs), **{f: med(f) for f in cols[5:] if f != "sparse_n"}}
    (out_dir / "surface_keyframe_metrics_summary.json").write_text(
        json.dumps({"voxel": voxel, "level": level, "scale": metrics_scale,
                    "by_component_region_median": agg}, indent=1), encoding="utf-8")
    return agg
