"""Make the frames agree before they are fused: a smooth per-keyframe depth
correction, solved jointly over every gated frame.

WHY THIS EXISTS

Each keyframe's monocular depth is fitted to the solve's sparse points by one
affine per frame (`dense.align_frame`). Every frame on its own is smooth to a
fraction of a voxel, but frames disagree with EACH OTHER about where a wall is
by 6-12 voxels (median std of a 0.3-unit patch across frames on the canonical
capture, `fixit/diagnose/DIAGNOSIS.md` section 5.2) -- about the width of the
truncation band. The TSDF averages sheets that are each smooth but sit at
different offsets and tilts, and the zero crossing steps wherever the set of
contributing frames changes: the crumpled, faceted, holey walls. Controlled
fusion showed the fusion code is faithful: give it consistent depth and it is
clean.

THE MODEL

For keyframe i, over the plain affine depth `z_aff`:

    z'(u, v) = exp(g_i(u, v)) * z_aff(u, v) + o_i(u, v) * median_i(z_aff)

`g_i` and `o_i` are uniform cubic B-splines over the image, `cells` = (3, 6)
cells (6 x 9 control points each), so 108 numbers per frame. Both start at
zero, which IS the plain affine.

THE SOLVE (LBFGS, torch)

  (a) SfM anchors: Cauchy on log(z' / z_sfm), sigma 2%, at every observation
      of a gated frame outside the redaction fill -- except a seeded 10% of
      3-D POINTS, held out of every term and every frame.
  (b) Cross-frame consistency: samples of frame i, back-projected with z'_i,
      projected into its co-visible frames j (top-k by shared sparse points
      plus revisits >= `loop_gap` keyframes away). Point-to-plane residual
      against j's corrected surface, relative to depth, Cauchy sigma 1%.
      OCCLUSION-GATED: only projections landing on j's eroded valid pixels,
      at a usable incidence, and within a gate that starts at 6% and becomes
      clamp(4 MAD, 2%, 6%). Correspondences are recomputed every outer
      iteration.
  (c) A weak prior toward the plain affine: ridge on g and o, second
      differences on the control grid.

THE DECISION IS MADE ON DATA THE SOLVE NEVER SAW

After solving, held-out anchor error and the point-to-plane disagreement on
frame pairs disjoint from the fitted ones (fresh pixels) are measured for the
plain affine and for the field. A field that makes EITHER worse, that is not
finite, or whose scale field runs away, is REFUSED and the surface is fused
from the plain affine; the record says so. The baseline is the stored affine,
which was fitted on the held-out points too, so the comparison favours the
baseline: a field is applied only when it beats an affine that has seen the
answers.

Measured on the canonical capture (352 frames, RTX 5070): held-out anchor
error 2.07% -> 1.07%, held-out pair disagreement 2.32% -> 0.58%; 33 s cold
with a 1.9 GB peak, 8.6 s warm-started (one outer iteration) from a field
solved over the first 80% of the walk. 800 synthetic keyframes: 49 s, 2.6 GB.

Persisted beside `align.json` in `dense/<session>/`:

    consistency.json        the record: key, state, metrics, timing
    consistency_field.npz   ki, g and o control grids, per-frame median

`align.json` is not touched: its records keep the plain affine, which is the
field's base and the provenance of every frame.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

CONSISTENCY_VERSION = 1
RECORD_FILE = "consistency.json"
FIELD_FILE = "consistency_field.npz"

STATE_APPLIED = "applied"
STATE_REFUSED = "refused"
STATE_FAILED = "failed"
STATE_SKIPPED = "skipped"
STATE_STOPPED = "stopped"


@dataclass(frozen=True)
class ConsistencyParams:
    """The solver. Every field is part of the cache key."""

    cells: tuple = (3, 6)
    """(across, down) B-spline cells. 3x6 is the measured knee: finer fields
    (8x16) win every held-out number and FUSE WORSE -- contradicted faces
    71k -> 129k, wall facets 51% -> 71% -- because a frame bends locally to
    agree with its co-visible pairs and farther frames then see through it."""
    offset_field: bool = True
    sig_sfm: float = 0.02
    sig_cross: float = 0.01
    lam_cross: float = 1.0
    sig_g: float = 0.10
    lam_g: float = 0.02
    sig_o: float = 0.05
    lam_o: float = 0.02
    sig_smooth: float = 0.02
    lam_smooth: float = 0.02
    per_frame: int = 768
    k_top: int = 10
    k_loop: int = 6
    min_shared: int = 15
    loop_gap: int = 15
    outer: int = 5
    inner: int = 60
    warm_outer: int = 2
    warm_gate0: float = 0.03
    gate0: float = 0.06
    gate_min: float = 0.02
    holdout_frac: float = 0.10
    heldout_k_top: int = 8
    heldout_k_loop: int = 6
    heldout_skip: int = 16
    heldout_pixels: int = 256
    min_heldout_anchors: int = 30
    max_correspondences: int = 4_000_000
    """Bound on the cross term's size. On a long walk the samples per frame
    shrink so frames x pairs x samples stays under this, which bounds both
    memory and time; the canonical 352-frame walk (4.5k pairs x 768) is
    3.5M and unaffected."""
    chunk: int = 1_000_000
    max_log_scale_p99: float = 0.5
    """A scale field whose 99th percentile |log scale| exceeds this (65%)
    has run away, and is refused. The canonical field's p99 is 0.17."""
    max_seconds: float = 240.0
    """Wall-clock budget for the optimisation. Outer iterations stop when it
    is spent; the held-out checks still decide."""
    seed: int = 1

    def digest(self) -> str:
        payload = json.dumps(dataclasses.asdict(self), sort_keys=True, default=list)
        return hashlib.sha1(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# the field
# ---------------------------------------------------------------------------


def _bspline_weights(t):
    """Uniform cubic B-spline weights for local parameter t in [0, 1)."""
    t2, t3 = t * t, t * t * t
    return ((1 - t) ** 3 / 6, (3 * t3 - 6 * t2 + 4) / 6,
            (-3 * t3 + 3 * t2 + 3 * t + 1) / 6, t3 / 6)


def axis_basis(n_pixels: int, cells: int) -> np.ndarray:
    """(n_pixels, cells + 3) dense basis along one image axis. Pixel p sits at
    p / (n_pixels - 1) of the axis, the convention of every evaluation here."""
    s = (np.arange(n_pixels, dtype=np.float64) / max(n_pixels - 1, 1)) * cells
    i = np.minimum(np.floor(s), cells - 1)
    w = _bspline_weights(s - i)
    B = np.zeros((n_pixels, cells + 3))
    rows = np.arange(n_pixels)
    for k in range(4):
        B[rows, i.astype(int) + k] = w[k]
    return B


class ConsistencyField:
    """The solved correction, ready to apply to a frame's affine depth."""

    def __init__(self, ki, G, O, med, cells, offset_field, shape):
        self.cells = (int(cells[0]), int(cells[1]))
        self.offset_field = bool(offset_field)
        self.shape = (int(shape[0]), int(shape[1]))
        nx, ny = self.cells[0] + 3, self.cells[1] + 3
        self.G = np.asarray(G, np.float32).reshape(len(ki), ny, nx)
        O = np.asarray(O, np.float32)
        self.O = (O.reshape(len(ki), ny, nx) if self.offset_field
                  else O.reshape(len(ki), 1, 1))
        self.med = np.asarray(med, np.float32).reshape(-1)
        self.index = {int(k): i for i, k in enumerate(ki)}
        H, W = self.shape
        self._By = axis_basis(H, self.cells[1]).astype(np.float32)
        self._Bx = axis_basis(W, self.cells[0]).astype(np.float32)
        self._dev_basis = {}

    def __contains__(self, ki) -> bool:
        return int(ki) in self.index

    def maps(self, ki):
        """(g, o) images of one frame, numpy float32 (H, W)."""
        i = self.index[int(ki)]
        g = self._By @ self.G[i] @ self._Bx.T
        o = (self._By @ self.O[i] @ self._Bx.T) if self.offset_field else \
            np.full(self.shape, float(self.O[i, 0, 0]), np.float32)
        return g, o

    def correct_numpy(self, ki, z):
        if int(ki) not in self.index:
            return z
        g, o = self.maps(ki)
        return (np.exp(g) * z + o * self.med[self.index[int(ki)]]).astype(np.float32)

    def correct(self, ki, z):
        """Corrected depth for a torch (H, W) affine depth map; unchanged for a
        frame the field does not know."""
        import torch

        if int(ki) not in self.index or tuple(z.shape) != self.shape:
            return z
        dev = z.device
        key = str(dev)
        if key not in self._dev_basis:
            self._dev_basis[key] = (torch.as_tensor(self._By, device=dev),
                                    torch.as_tensor(self._Bx, device=dev))
        By, Bx = self._dev_basis[key]
        i = self.index[int(ki)]
        g = By @ torch.as_tensor(self.G[i], device=dev) @ Bx.T
        if self.offset_field:
            o = By @ torch.as_tensor(self.O[i], device=dev) @ Bx.T
        else:
            o = float(self.O[i, 0, 0])
        out = torch.exp(g) * z + o * float(self.med[i])
        return torch.where(torch.isfinite(z), out, z)

    # -- persistence ------------------------------------------------------
    def to_bytes(self, key: str) -> bytes:
        buf = io.BytesIO()
        ki = np.array(sorted(self.index, key=self.index.get), np.int64)
        np.savez_compressed(buf, version=np.int64(CONSISTENCY_VERSION), key=np.array(key),
                            ki=ki, G=self.G, O=self.O, med=self.med,
                            cells=np.array(self.cells), offset_field=np.bool_(self.offset_field),
                            shape=np.array(self.shape))
        return buf.getvalue()

    @classmethod
    def load(cls, path: Path):
        """(field, key), or (None, None) when absent or unreadable."""
        try:
            with np.load(path) as z:
                if int(z["version"]) != CONSISTENCY_VERSION:
                    return None, None
                f = cls(z["ki"], z["G"], z["O"], z["med"], tuple(int(x) for x in z["cells"]),
                        bool(z["offset_field"]), tuple(int(x) for x in z["shape"]))
                return f, str(z["key"])
        except (OSError, ValueError, KeyError):
            return None, None


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------


def consistency_key(frames, solution, surface_params, cparams: ConsistencyParams,
                    outer: int, warm_outer: int) -> str:
    """Everything the solve reads. A new solve, a refit affine, a changed
    image, a changed gate or validity rule, or a changed solver: a new key."""
    K = np.asarray(frames.K, float).round(6).tolist()
    items = [(int(ki), frames.kids.get(ki), round(float(a), 9), round(float(b), 9),
              getattr(frames, "image_sha1", {}).get(ki),
              None if zmax is None else round(float(zmax), 6))
             for ki, a, b, _R, _t, _ho, zmax in frames.items]
    p = surface_params
    payload = json.dumps([
        CONSISTENCY_VERSION, cparams.digest(), int(outer), int(warm_outer),
        getattr(solution, "input_digest", None), frames.kind, K, items,
        (p.edge_rel, p.max_grazing_deg, p.anchor_depth_multiple, p.max_depth_frac,
         p.fill_margin_px, p.gate_rel),
    ], default=str)
    return hashlib.sha1(payload.encode()).hexdigest()


class _Data:
    """The gated frames on the solver's device: affine depth, validity, poses,
    and the SfM anchors split into fit and held out."""

    def __init__(self, frames, solution, surface_params, median_depth, cparams, device):
        import torch

        from tower.world_builder.surface import depth_bound, depth_validity
        from tower.world_builder.surface_pipeline import _depth_from_prediction, _dilate_fill

        self.dev = device
        self.K = np.asarray(frames.K, float)
        self.fx, self.fy = float(self.K[0, 0]), float(self.K[1, 1])
        self.cx, self.cy = float(self.K[0, 2]), float(self.K[1, 2])
        kis, zaff, valid, R, t, fills, meds = [], [], [], [], [], [], []
        shape = None
        for ki, a, b, Ri, ti, _ho, zmax in frames.items:
            pred, _img, fill = frames.load(ki, image=False)
            if pred is None:
                continue
            if shape is None:
                shape = pred.shape
            if pred.shape != shape:
                continue
            z = np.asarray(_depth_from_prediction(pred, a, b, frames.kind), np.float32)
            z = np.where(np.isfinite(z) & (z > 0), z, 0.0).astype(np.float32)
            zt = torch.as_tensor(z, device=device)
            ok, _cos = depth_validity(zt, self.K, surface_params, median_depth,
                                      max_depth=depth_bound(surface_params, median_depth, zmax))
            ok &= zt > 0
            fill_np = np.zeros(shape, bool) if fill is None else np.asarray(fill, bool)
            if fill_np.any():
                ok &= ~_dilate_fill(torch.as_tensor(fill_np, device=device),
                                    surface_params.fill_margin_px)
            zs = z[::7, ::7]
            zs = zs[zs > 0]
            kis.append(int(ki))
            zaff.append(zt.to(torch.float16))
            valid.append(ok)
            fills.append(fill_np)
            meds.append(float(np.median(zs)) if zs.size else 1.0)
            R.append(np.asarray(Ri, np.float64))
            t.append(np.asarray(ti, np.float64))
        self.N = len(kis)
        self.ki = np.array(kis, np.int64)
        self.H, self.W = shape if shape is not None else (0, 0)
        if not self.N:
            return
        self.zaff = torch.stack(zaff)
        self.valid = torch.stack(valid)
        # Eroded in chunks: a float copy of the whole stack is 0.7 GB at 800
        # keyframes, twice over, for a mask that is one byte a pixel.
        self.valid_er = torch.cat([
            (-torch.nn.functional.max_pool2d(
                -self.valid[s:s + 32].to(torch.float32)[:, None], 5, 1, 2))[:, 0] > 0.5
            for s in range(0, self.N, 32)])
        self.med = torch.as_tensor(meds, device=device, dtype=torch.float32)
        self.R_np, self.t_np = np.stack(R), np.stack(t)
        self.R = torch.as_tensor(self.R_np, device=device, dtype=torch.float32)
        self.t = torch.as_tensor(self.t_np, device=device, dtype=torch.float32)
        self._anchors(solution, np.stack(fills), cparams)

    def _anchors(self, solution, fills, cparams):
        H, W = self.H, self.W
        obs = np.asarray(getattr(solution, "observations", np.zeros((0, 3))),
                         np.int64).reshape(-1, 3)
        xyz = np.asarray(getattr(solution, "xyz", np.zeros((0, 3))), np.float64).reshape(-1, 3)
        xy = np.asarray(getattr(solution, "observation_xy", np.zeros((0, 2))),
                        np.float64).reshape(-1, 2)
        empty = dict(f=np.zeros(0, np.int64), u=np.zeros(0), v=np.zeros(0), z=np.zeros(0),
                     held=np.zeros(0, bool))
        if not len(obs) or not len(xyz):
            self.anc = empty
            return
        n_kf = int(max(obs[:, 0].max(), self.ki.max())) + 1
        pos = np.full(n_kf, -1, np.int64)
        pos[self.ki] = np.arange(self.N)
        k = obs[:, 0]
        m = (k >= 0) & (k < n_kf)
        m[m] = pos[k[m]] >= 0
        m &= (obs[:, 2] >= 0) & (obs[:, 2] < len(xyz))
        fi = pos[k[m]]
        pid = obs[m, 2]
        X = xyz[pid]
        pc = np.einsum("nij,nj->ni", self.R_np[fi], X) + self.t_np[fi]
        zc = pc[:, 2]
        if len(xy) == len(obs):
            u, v = xy[m, 0], xy[m, 1]
        else:
            # Older solves record no observation pixels: the projection of the
            # point is the observation to within the solve's reprojection error.
            with np.errstate(divide="ignore", invalid="ignore"):
                u = pc[:, 0] / zc * self.fx + self.cx
                v = pc[:, 1] / zc * self.fy + self.cy
        g = np.isfinite(u) & np.isfinite(v) & (zc > 1e-3) & (u >= 0) & (u < W - 1) \
            & (v >= 0) & (v < H - 1)
        ui = np.clip(np.rint(np.nan_to_num(u)).astype(np.int64), 0, W - 1)
        vi = np.clip(np.rint(np.nan_to_num(v)).astype(np.int64), 0, H - 1)
        g &= ~fills[fi, vi, ui]
        # Held out by POINT, so a held-out track is unseen by every frame.
        rng = np.random.default_rng(12345)
        held_pt = rng.random(len(xyz)) < cparams.holdout_frac
        self.anc = dict(f=fi[g], u=u[g], v=v[g], z=zc[g], held=held_pt[pid[g]], pid=pid[g])

    def covis(self):
        A = self.anc
        m = ~A["held"]
        S = np.zeros((self.N, self.N))
        if not m.any():
            return S
        from scipy.sparse import coo_matrix

        f, pid = A["f"][m], A["pid"][m]
        M = coo_matrix((np.ones(len(f)), (f, pid)), shape=(self.N, int(pid.max()) + 1)).tocsr()
        M.data[:] = 1
        S = (M @ M.T).toarray()
        np.fill_diagonal(S, 0)
        return S


def select_pairs(ki, S, k_top, k_loop, min_shared, loop_gap, skip=0):
    """Per frame: its top-k co-visible frames by shared sparse points, plus
    k_loop revisits at least `loop_gap` keyframes away. `skip` drops that many
    top-ranked candidates of each list, for pairs disjoint from the fitted."""
    pairs = set()
    for i in range(len(ki)):
        order = [j for j in np.argsort(-S[i], kind="stable") if S[i, j] >= min_shared]
        near = order[skip:skip + k_top]
        far = [j for j in order if abs(int(ki[j]) - int(ki[i])) >= loop_gap][skip:skip + k_loop]
        for j in list(near) + list(far):
            pairs.add((i, int(j)))
    return np.array(sorted(pairs), np.int64).reshape(-1, 2)


# ---------------------------------------------------------------------------
# the solver
# ---------------------------------------------------------------------------


class _Field:
    def __init__(self, N, cells, offset_field, device):
        import torch

        self.N = N
        self.cells = cells
        self.nx, self.ny = cells[0] + 3, cells[1] + 3
        self.nc = self.nx * self.ny
        self.offset_field = offset_field
        self.G = torch.zeros((N, self.nc), device=device, requires_grad=True)
        self.O = torch.zeros((N, self.nc if offset_field else 1), device=device,
                             requires_grad=True)

    def basis(self, u, v, W, H):
        import torch

        cx, cy = self.cells
        sx = (u / (W - 1)).clamp(0, 1) * cx
        sy = (v / (H - 1)).clamp(0, 1) * cy
        ix = sx.floor().clamp(max=cx - 1)
        iy = sy.floor().clamp(max=cy - 1)
        bx = torch.stack(_bspline_weights(sx - ix), -1)
        by = torch.stack(_bspline_weights(sy - iy), -1)
        off = torch.arange(4, device=u.device)
        gx = ix.long().unsqueeze(-1) + off
        gy = iy.long().unsqueeze(-1) + off
        idx = (gy.unsqueeze(-1) * self.nx + gx.unsqueeze(-2)).reshape(u.shape + (16,))
        w = (by.unsqueeze(-1) * bx.unsqueeze(-2)).reshape(u.shape + (16,))
        return idx, w

    def eval(self, f, u, v, zaff, med, W, H):
        import torch

        idx, w = self.basis(u, v, W, H)
        flat = f.unsqueeze(-1) * self.nc + idx
        g = (self.G.reshape(-1)[flat] * w).sum(-1)
        if self.offset_field:
            o = (self.O.reshape(-1)[flat] * w).sum(-1)
        else:
            o = self.O[f, 0]
        return torch.exp(g) * zaff + o * med[f]

    def precompute(self, f, u, v, med, W, H):
        """The basis of each sample, once per set of correspondences: the
        LBFGS closure then only gathers."""
        import torch

        idx, w = self.basis(u, v, W, H)
        flat = (f.unsqueeze(-1) * self.nc + idx).to(torch.int32)
        return flat, w, med[f]

    def eval_pre(self, flat, w, zaff, medf):
        import torch

        fl = flat.long()
        g = (self.G.reshape(-1)[fl] * w).sum(-1)
        if self.offset_field:
            o = (self.O.reshape(-1)[fl] * w).sum(-1)
        else:
            o = self.O.reshape(-1)[fl[:, 0] // self.nc]
        return torch.exp(g) * zaff + o * medf

    def reg(self):
        G = self.G.reshape(self.N, self.ny, self.nx)
        d2x = G[:, :, 2:] - 2 * G[:, :, 1:-1] + G[:, :, :-2]
        d2y = G[:, 2:] - 2 * G[:, 1:-1] + G[:, :-2]
        return ((self.G ** 2).mean(), (self.O ** 2).mean(),
                (d2x.pow(2).sum() + d2y.pow(2).sum()) / max(d2x.numel() + d2y.numel(), 1))


def _bilinear(img, fidx, u, v, W, H):
    import torch.nn.functional as TF

    gx = u / (W - 1) * 2 - 1
    gy = v / (H - 1) * 2 - 1
    import torch

    grid = torch.stack([gx, gy], -1).unsqueeze(2)
    return TF.grid_sample(img[fidx].unsqueeze(1).float(), grid, mode="bilinear",
                          align_corners=True)[:, 0, :, 0]


def _nearest(mask, fidx, u, v, W, H):
    ui = u.round().long().clamp(0, W - 1)
    vi = v.round().long().clamp(0, H - 1)
    return mask[fidx.unsqueeze(1).expand_as(ui), vi, ui]


def _pixel_samples(D, per_frame, seed):
    import torch

    g = torch.Generator(device="cpu").manual_seed(int(seed))
    us, vs = [], []
    for i in range(D.N):
        nz = torch.nonzero(D.valid[i].reshape(-1)).squeeze(1).cpu()
        if nz.numel() == 0:
            sel = torch.zeros(per_frame, dtype=torch.long)
        else:
            sel = nz[torch.randint(nz.numel(), (per_frame,), generator=g)]
        us.append((sel % D.W).float() + torch.rand(per_frame, generator=g) - 0.5)
        vs.append((sel // D.W).float() + torch.rand(per_frame, generator=g) - 0.5)
    return (torch.stack(us).to(D.dev).clamp(0, D.W - 1),
            torch.stack(vs).to(D.dev).clamp(0, D.H - 1))


def _correspondences(D, field, pairs, su, sv, gate, return_all=False, chunk=32):
    """Project frame i's samples with its current corrected depth into j; keep
    projections that land on j's eroded valid pixels at a usable incidence.
    The residual is point-to-plane against j's local surface, linear in z'_i
    and z'_j."""
    import torch

    W, H = D.W, D.H
    fx, fy, cx, cy = D.fx, D.fy, D.cx, D.cy
    keys = ("fi", "ui", "vi", "zi", "fj", "uj", "vj", "zj", "A", "B", "Dn", "den")
    out = {k: [] for k in keys}
    n_valid = 0
    with torch.no_grad():
        for s in range(0, len(pairs), chunk):
            pc = torch.as_tensor(pairs[s:s + chunk], device=D.dev)
            I, J = pc[:, 0], pc[:, 1]
            u, v = su[I], sv[I]
            zi_aff = _bilinear(D.zaff, I, u, v, W, H)
            fI = I.unsqueeze(1).expand_as(u)
            zi = field.eval(fI, u, v, zi_aff, D.med, W, H)
            ray = torch.stack([(u - cx) / fx, (v - cy) / fy, torch.ones_like(u)], -1)
            Ri, ti, Rj, tj = D.R[I], D.t[I], D.R[J], D.t[J]
            Rij = Rj @ Ri.transpose(1, 2)
            tij = tj - (Rij @ ti.unsqueeze(-1)).squeeze(-1)
            av = torch.einsum("pab,psb->psa", Rij, ray)
            xj = av * zi.unsqueeze(-1) + tij.unsqueeze(1)
            zc = xj[..., 2]
            uj = xj[..., 0] / zc.clamp(min=1e-4) * fx + cx
            vj = xj[..., 1] / zc.clamp(min=1e-4) * fy + cy
            fin = (zc > 1e-3) & (uj >= 3) & (uj <= W - 4) & (vj >= 3) & (vj <= H - 4) \
                & (zi_aff > 0)
            fin &= _nearest(D.valid_er, J, uj, vj, W, H)
            ujc, vjc = uj.clamp(0, W - 1), vj.clamp(0, H - 1)
            zj_aff = _bilinear(D.zaff, J, ujc, vjc, W, H)

            def pt(du, dv):
                uu, vv = (uj + du).clamp(0, W - 1), (vj + dv).clamp(0, H - 1)
                zz = _bilinear(D.zaff, J, uu, vv, W, H)
                return torch.stack([(uu - cx) / fx * zz, (vv - cy) / fy * zz, zz], -1)

            n = torch.cross(pt(2, 0) - pt(-2, 0), pt(0, 2) - pt(0, -2), dim=-1)
            n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            rayj = torch.stack([(uj - cx) / fx, (vj - cy) / fy, torch.ones_like(uj)], -1)
            fJ = J.unsqueeze(1).expand_as(uj)
            zj = field.eval(fJ, ujc, vjc, zj_aff, D.med, W, H)
            Dn = (n * rayj).sum(-1)
            cosj = Dn.abs() / rayj.norm(dim=-1)
            fin &= torch.isfinite(n).all(-1) & (cosj > 0.25) & (zj_aff > 0)
            A = (n * av).sum(-1)
            B = (n * tij.unsqueeze(1)).sum(-1)
            den = zc.clamp(min=1e-3)
            r0 = (A * zi + B - Dn * zj) / den
            fin &= torch.isfinite(r0)
            n_valid += int(fin.sum())
            keep = fin if return_all else (fin & (r0.abs() < gate))
            for k_, val in (("fi", fI), ("ui", u), ("vi", v), ("zi", zi_aff), ("fj", fJ),
                            ("uj", ujc), ("vj", vjc), ("zj", zj_aff), ("A", A), ("B", B),
                            ("Dn", Dn), ("den", den)):
                out[k_].append(val[keep])
    res = {k: (torch.cat(v) if v else torch.zeros(0, device=D.dev)) for k, v in out.items()}
    n = int(res["fi"].numel())
    with torch.no_grad():
        for side, (fk, uk, vk) in (("i", ("fi", "ui", "vi")), ("j", ("fj", "uj", "vj"))):
            parts = [field.precompute(res[fk][s0:s0 + 1_000_000], res[uk][s0:s0 + 1_000_000],
                                      res[vk][s0:s0 + 1_000_000], D.med, W, H)
                     for s0 in range(0, n, 1_000_000)]
            if parts:
                res["p" + side] = torch.cat([q[0] for q in parts])
                res["w" + side] = torch.cat([q[1] for q in parts])
                res["m" + side] = torch.cat([q[2] for q in parts])
            else:
                res["p" + side] = torch.zeros((0, 16), dtype=torch.int32, device=D.dev)
                res["w" + side] = torch.zeros((0, 16), device=D.dev)
                res["m" + side] = torch.zeros(0, device=D.dev)
    for k in ("fi", "ui", "vi", "fj", "uj", "vj"):
        del res[k]
    res["n"] = n
    res["n_valid"] = n_valid
    return res


def _cross_residual(field, D, c, s=None):
    sl = slice(None) if s is None else s
    zi = field.eval_pre(c["pi"][sl], c["wi"][sl], c["zi"][sl], c["mi"][sl])
    zj = field.eval_pre(c["pj"][sl], c["wj"][sl], c["zj"][sl], c["mj"][sl])
    return (c["A"][sl] * zi + c["B"][sl] - c["Dn"][sl] * zj) / c["den"][sl]


def _anchor_tensors(D, held: bool):
    import torch

    A = D.anc
    m = A["held"] if held else ~A["held"]
    if not m.any():
        z = torch.zeros(0, device=D.dev)
        return dict(f=z.long(), u=z, v=z, z=z, za=z)
    f = torch.as_tensor(A["f"][m], device=D.dev)
    u = torch.as_tensor(A["u"][m], device=D.dev, dtype=torch.float32)
    v = torch.as_tensor(A["v"][m], device=D.dev, dtype=torch.float32)
    z = torch.as_tensor(A["z"][m], device=D.dev, dtype=torch.float32)
    za = torch.zeros_like(z)
    for i in torch.unique(f).tolist():
        mi = f == i
        za[mi] = _bilinear(D.zaff, torch.tensor([i], device=D.dev), u[mi][None], v[mi][None],
                           D.W, D.H)[0]
    ok = za > 0
    return dict(f=f[ok], u=u[ok], v=v[ok], z=z[ok], za=za[ok])


def _cauchy(x):
    import torch

    return torch.log1p(x * x)


def _eval_sfm(D, field, anc):
    import torch

    if anc["f"].numel() == 0:
        return None
    with torch.no_grad():
        z = field.eval(anc["f"], anc["u"], anc["v"], anc["za"], D.med, D.W, D.H)
        rel = ((z - anc["z"]) / anc["z"]).abs()
        rel = rel[torch.isfinite(rel)]
    if rel.numel() == 0:
        return None
    return dict(n=int(rel.numel()), median=float(rel.median()), mean=float(rel.mean()))


def _eval_cross(D, field, pairs, samples):
    import torch

    if not len(pairs):
        return None
    c = _correspondences(D, field, pairs, samples[0], samples[1], gate=10.0, return_all=True)
    if c["n"] == 0:
        return None
    with torch.no_grad():
        r = _cross_residual(field, D, c).abs()
    return dict(n=int(r.numel()), median=float(r.median()),
                frac_lt_1pct=float((r < 0.01).float().mean()))


def _optimise(D, field, anc_fit, pairs, samples, cparams, outer, gate0, should_stop, log):
    """LBFGS over the field. Loss terms are evaluated in chunks with gradients
    accumulated, so memory follows `chunk`, not the walk."""
    import torch

    su, sv = samples
    gate = gate0
    anc_pre = (field.precompute(anc_fit["f"], anc_fit["u"], anc_fit["v"], D.med, D.W, D.H)
               if anc_fit["f"].numel() else None)
    hist = []
    t0 = time.time()
    budget_spent = False
    for it in range(outer):
        if should_stop is not None and should_stop():
            return hist, "stopped"
        if time.time() - t0 > cparams.max_seconds:
            budget_spent = True
            break
        corr = _correspondences(D, field, pairs, su, sv, gate) if cparams.lam_cross > 0 else None
        n_corr = int(corr["n"]) if corr is not None else 0
        opt = torch.optim.LBFGS([field.G, field.O], lr=1.0, max_iter=cparams.inner,
                                history_size=20, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            total = 0.0
            if anc_fit["f"].numel():
                zs = field.eval_pre(anc_pre[0], anc_pre[1], anc_fit["za"], anc_pre[2])
                rs = torch.log(zs.clamp(min=1e-4)) - torch.log(anc_fit["z"])
                L = _cauchy(rs / cparams.sig_sfm).mean()
                L.backward()
                total += float(L.detach())
            if n_corr:
                for s0 in range(0, n_corr, cparams.chunk):
                    rc = _cross_residual(field, D, corr, slice(s0, s0 + cparams.chunk))
                    L = cparams.lam_cross * _cauchy(rc / cparams.sig_cross).sum() / n_corr
                    L.backward()
                    total += float(L.detach())
            rg, ro, rsm = field.reg()
            L = (cparams.lam_g * rg / cparams.sig_g ** 2 + cparams.lam_o * ro / cparams.sig_o ** 2
                 + cparams.lam_smooth * rsm / cparams.sig_smooth ** 2)
            L.backward()
            total += float(L.detach())
            return torch.tensor(total)

        loss = float(opt.step(closure))
        rec = dict(outer=it, loss=loss, gate=gate, correspondences=n_corr)
        if not math.isfinite(loss) or not bool(torch.isfinite(field.G).all()) \
                or not bool(torch.isfinite(field.O).all()):
            hist.append(rec)
            return hist, "diverged"
        if n_corr:
            with torch.no_grad():
                rc = torch.cat([_cross_residual(field, D, corr, slice(s0, s0 + cparams.chunk))
                                for s0 in range(0, n_corr, cparams.chunk)])
                mad = float(1.4826 * (rc - rc.median()).abs().median())
                rec.update(cross_median=float(rc.abs().median()), cross_mad=mad)
                gate = float(np.clip(4 * mad, cparams.gate_min, cparams.gate0))
        del corr
        hist.append(rec)
        log(f"[Tower][WorldBuilder][consistency] {json.dumps(rec)}")
    return hist, ("budget" if budget_spent else "done")


def solve_field(frames, solution, surface_params, median_depth, *,
                cparams: ConsistencyParams | None = None, outer: int | None = None,
                warm: ConsistencyField | None = None, warm_outer: int | None = None,
                device=None, should_stop=None, log=None):
    """Solve and judge a field. Returns (field or None, record). Never raises
    for a solve that fails; the record's state says what happened."""
    import torch

    cparams = cparams or ConsistencyParams()
    log = log or logger.info
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    record = dict(version=CONSISTENCY_VERSION, device=str(dev), cells=list(cparams.cells),
                  offset_field=cparams.offset_field, params=dataclasses.asdict(cparams))
    D = _Data(frames, solution, surface_params, median_depth, cparams, dev)
    record["frames"] = int(D.N)
    anc_fit = _anchor_tensors(D, held=False) if D.N else None
    anc_ho = _anchor_tensors(D, held=True) if D.N else None
    record["anchors_fit"] = int(anc_fit["f"].numel()) if D.N else 0
    record["anchors_held_out"] = int(anc_ho["f"].numel()) if D.N else 0
    if D.N < 2 or record["anchors_fit"] < 10:
        record.update(state=STATE_SKIPPED,
                      reason=f"{D.N} frames and {record['anchors_fit']} fit anchors: "
                             "nothing to make consistent")
        record["seconds"] = round(time.time() - t0, 2)
        return None, record
    if record["anchors_held_out"] < cparams.min_heldout_anchors:
        record.update(state=STATE_SKIPPED,
                      reason=f"only {record['anchors_held_out']} held-out anchors; a field "
                             "that cannot be checked on unseen data is not applied")
        record["seconds"] = round(time.time() - t0, 2)
        return None, record

    S = D.covis()
    pairs = select_pairs(D.ki, S, cparams.k_top, cparams.k_loop, cparams.min_shared,
                         cparams.loop_gap)
    fit_set = {tuple(p) for p in pairs.tolist()}
    ho = select_pairs(D.ki, S, cparams.heldout_k_top, cparams.heldout_k_loop,
                      cparams.min_shared, cparams.loop_gap, skip=cparams.heldout_skip)
    ho = np.array([p for p in ho.tolist() if tuple(p) not in fit_set
                   and (p[1], p[0]) not in fit_set], np.int64).reshape(-1, 2)
    cross_check = "held-out-pairs"
    if not len(ho):
        # A walk too short to have pairs beyond the fitted ranks: the fitted
        # pairs at pixels the solve never sampled, and the record says so.
        ho, cross_check = pairs, "fit-pairs-fresh-pixels"
    per_frame = cparams.per_frame
    if len(pairs):
        per_frame = int(max(64, min(per_frame, cparams.max_correspondences // len(pairs))))
    record.update(pairs=int(len(pairs)), heldout_pairs=int(len(ho)), cross_check=cross_check,
                  samples_per_frame=per_frame)

    field = _Field(D.N, tuple(cparams.cells), cparams.offset_field, dev)
    warm_frames = 0
    if (warm is not None and warm.cells == tuple(cparams.cells)
            and warm.offset_field == cparams.offset_field and warm.shape == (D.H, D.W)):
        with torch.no_grad():
            for i, ki in enumerate(D.ki):
                j = warm.index.get(int(ki))
                if j is None:
                    continue
                field.G[i] = torch.as_tensor(warm.G[j].reshape(-1), device=dev)
                field.O[i] = torch.as_tensor(warm.O[j].reshape(-1), device=dev)
                warm_frames += 1
    warm_used = warm_frames > 0
    n_outer = (warm_outer if warm_used else outer)
    n_outer = int(n_outer if n_outer is not None else
                  (cparams.warm_outer if warm_used else cparams.outer))
    record["warm_start"] = dict(used=warm_used, frames=warm_frames, outer=n_outer)

    try:
        samples = _pixel_samples(D, per_frame, cparams.seed)
        ho_samples = _pixel_samples(D, cparams.heldout_pixels, cparams.seed + 7919)
        zero = _Field(D.N, tuple(cparams.cells), cparams.offset_field, dev)
        before = dict(sfm=_eval_sfm(D, zero, anc_ho), cross=_eval_cross(D, zero, ho, ho_samples))
        t_opt = time.time()
        hist, how = _optimise(D, field, anc_fit, pairs, samples, cparams, n_outer,
                              cparams.warm_gate0 if warm_used else cparams.gate0,
                              should_stop, log)
        record["optimise_seconds"] = round(time.time() - t_opt, 2)
        record["history"] = hist
        record["ended"] = how
        if how == "stopped":
            record.update(state=STATE_STOPPED, reason="stopped during the solve")
            return None, record
        if how == "diverged":
            record.update(state=STATE_FAILED, reason="the solve diverged (non-finite field)")
            return None, record
        after = dict(sfm=_eval_sfm(D, field, anc_ho), cross=_eval_cross(D, field, ho, ho_samples))
        with torch.no_grad():
            absg = field.G.detach().abs().flatten()
            p99 = float(torch.quantile(absg[:: max(1, absg.numel() // 1_000_000)], 0.99))
            p50 = float(absg.median())
    except torch.cuda.OutOfMemoryError as exc:  # pragma: no cover - hardware
        record.update(state=STATE_FAILED, reason=f"out of GPU memory: {exc}")
        return None, record
    finally:
        record["seconds"] = round(time.time() - t0, 2)
        if dev.type == "cuda":
            record["gpu_peak_mb"] = round(torch.cuda.max_memory_allocated(dev) / 2 ** 20, 1)

    record["heldout"] = dict(before=before, after=after)
    record["scale_field_abs_p50_p99"] = [p50, p99]
    refusals = []
    if after["sfm"] is None or before["sfm"] is None:
        refusals.append("held-out anchor error could not be measured")
    elif not after["sfm"]["median"] <= before["sfm"]["median"]:
        refusals.append(f"held-out anchor error got worse ({before['sfm']['median']:.4f} -> "
                        f"{after['sfm']['median']:.4f})")
    if before["cross"] is not None and after["cross"] is not None \
            and not after["cross"]["median"] <= before["cross"]["median"]:
        refusals.append(f"held-out cross-frame disagreement got worse "
                        f"({before['cross']['median']:.4f} -> {after['cross']['median']:.4f})")
    if not p99 <= cparams.max_log_scale_p99:
        refusals.append(f"scale field ran away (p99 |log scale| {p99:.3f})")
    if refusals:
        record.update(state=STATE_REFUSED, reason="; ".join(refusals))
        return None, record
    record.update(state=STATE_APPLIED, reason=None)
    out = ConsistencyField(D.ki, field.G.detach().cpu().numpy(), field.O.detach().cpu().numpy(),
                           D.med.cpu().numpy(), tuple(cparams.cells), cparams.offset_field,
                           (D.H, D.W))
    return out, record


# ---------------------------------------------------------------------------
# the cached entry point
# ---------------------------------------------------------------------------


@dataclass
class ConsistencyResult:
    state: str
    record: dict
    field: ConsistencyField | None = None
    reused: bool = False

    def summary(self) -> dict:
        r = self.record or {}
        keep = ("version", "state", "reason", "frames", "cells", "offset_field", "heldout",
                "warm_start", "seconds", "optimise_seconds", "gpu_peak_mb", "pairs",
                "samples_per_frame", "cross_check", "scale_field_abs_p50_p99", "key", "ended")
        return {**{k: r.get(k) for k in keep if k in r}, "reused": self.reused}


def ensure_consistency(dense_root: Path, frames, solution, surface_params, median_depth, *,
                       cparams: ConsistencyParams | None = None, device=None,
                       should_stop=None, solve=None) -> ConsistencyResult:
    """The field for exactly these frames, from cache when it is theirs.

    A cached `applied` or `refused` decision with this key is reused; a
    `failed` one is not (it may be the GPU, not the data), so the next build
    tries again. When the key differs -- a new global solve during a walk --
    the previous applied field warm-starts the new solve for the frames both
    know. On any exception the surface is fused from the plain affine and the
    record says why.
    """
    from tower.storage import write_bytes_atomic, write_json_atomic

    cparams = cparams or ConsistencyParams()
    solve = solve or solve_field
    outer = int(getattr(surface_params, "consistency_outer", cparams.outer))
    warm_outer = int(getattr(surface_params, "consistency_warm_outer", cparams.warm_outer))
    key = consistency_key(frames, solution, surface_params, cparams, outer, warm_outer)
    rec_path = Path(dense_root) / RECORD_FILE
    field_path = Path(dense_root) / FIELD_FILE
    try:
        cached = json.loads(rec_path.read_text())
    except (OSError, ValueError):
        cached = None
    prev_field, prev_key = ConsistencyField.load(field_path) if field_path.exists() else (None, None)
    if isinstance(cached, dict) and cached.get("key") == key \
            and cached.get("version") == CONSISTENCY_VERSION:
        if cached.get("state") in (STATE_REFUSED, STATE_SKIPPED):
            return ConsistencyResult(cached["state"], cached, None, reused=True)
        if cached.get("state") == STATE_APPLIED and prev_field is not None and prev_key == key:
            return ConsistencyResult(STATE_APPLIED, cached, prev_field, reused=True)
    warm = prev_field if prev_field is not None else None
    try:
        field, record = solve(frames, solution, surface_params, median_depth, cparams=cparams,
                              outer=outer, warm=warm, warm_outer=warm_outer, device=device,
                              should_stop=should_stop)
    except Exception as exc:  # noqa: BLE001 -- the surface falls back, it does not fail
        logger.exception("[Tower][WorldBuilder][consistency] solve failed; fusing the plain affine")
        record = dict(version=CONSISTENCY_VERSION, state=STATE_FAILED,
                      reason=f"{type(exc).__name__}: {exc}")
        field = None
    try:
        import torch

        if torch.cuda.is_available():
            # The fusion that follows wants the memory the solve's caches hold.
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    record.setdefault("version", CONSISTENCY_VERSION)
    record["key"] = key
    record["solve_digest"] = getattr(solution, "input_digest", None)
    record["written_at"] = time.time()
    state = record.get("state", STATE_FAILED)
    if state == STATE_STOPPED:
        return ConsistencyResult(state, record, None)
    if state == STATE_APPLIED and field is not None:
        data = field.to_bytes(key)
        write_bytes_atomic(field_path, lambda handle, d=data: handle.write(d))
    write_json_atomic(rec_path, record)
    logger.info("[Tower][WorldBuilder][consistency] %s over %s frames in %.1fs%s",
                state, record.get("frames"), record.get("seconds") or 0.0,
                f": {record.get('reason')}" if record.get("reason") else "")
    return ConsistencyResult(state, record, field if state == STATE_APPLIED else None)
