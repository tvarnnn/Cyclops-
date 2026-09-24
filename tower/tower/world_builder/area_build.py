"""Build one AREA: a part of the walk the evidence gate did not place, on its own.

Contract: `docs/contracts/WORLD-BUILDER-COMPONENTS.md` §5.4 ("built in the area's own
frame and levelled") and §2.3 (`shown_as: "area"`: at least 30 keyframes or 5 s).

THE APPROACH IS P2-PX's (run `experiments/P2-PX/tools/px_build.py`, C2 built the
target's areas offline in 45-90 s each), moved into the product:

1. **Its own solution.** The published solve, cut to the area's keyframes
   (`components.json`'s Tower-internal `keyframe_ids`), every one of them component 0,
   with the 3-D points they observe -- the one component the existing surface and
   appearance stages build. `keyframe_ids` stays the session's full list so a
   keyframe's `ki` (its depth files' name) is the room's.
2. **Levelled.** Rotated so the vertical estimated from the AREA'S OWN images is up
   (-Y, OpenCV y-down) and centred on its cameras' median, NOT scaled: an area has no
   scale relative to the room (§5.4, "Scale"). The vertical is P2-R5's `group_up`
   (level head + level floors/tables/ceilings from MoGe normals), ported below with
   its pre-registered constants; P2-PX levelled the target's two areas with it. The
   normals come from the area's own depth stage, run once on the unlevelled solution;
   the surface stage then re-fits those predictions to the levelled solution rather
   than predicting again (`dense_pipeline.reusable_predictions`).
3. **The room's stages, unchanged.** `scripts/world_build_session.final_surface_stages`
   -- the builder's own final path -- runs on `components.AreaStore`, so the area is
   built with the final presets from the session's own redacted keyframes and lands
   under `<world>/areas/<area>/`, and each stage's outcome is recorded in
   the area's record rather than in `session.json`.

When the vertical cannot be estimated (fewer normals than `group_up`'s own floor) the
area is built in the solve's own orientation and `levelled` is false (§5.4). No other
support floor is applied: the only measured case (the target's Area 2, support 0.058)
rendered level (OPEN T3).

This module computes no component and decides nothing about placement.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from tower.world_builder import components as C
from tower.world_builder.records import (
    STAGE_APPEARANCE,
    STAGE_STATE_FAILED,
    STAGE_STATE_RUNNING,
    STAGE_STATE_STOPPED,
    STAGE_STATE_UNAVAILABLE,
    STAGE_SURFACE,
)

logger = logging.getLogger(__name__)

# --- P2-R5's pre-registered vertical constants (RUN experiments/P2-R5/scripts/r5_lib.py,
# validated there; used by P2-PX for the target's areas). Not tuned here. ---
UP_GATE_DEG = 5.0      # level-head gate: RMS roll within +5 deg of the best candidate
UP_NORMAL_TOL = 8.0    # a normal supports u if within 8 deg of +-u
UP_PITCH_CONE = 60.0   # u within 60 deg of the mean camera up (head pitch)
UP_MIN_NORMALS = 1000  # below this `group_up` could not use normals: not levelled
PIX_STEP = 8           # depth-map subsampling
DEPTH_VALID_M = (0.1, 12.0)
NORMALS_MAX = 40000
# The harness's roll-free up (`coherence_eval/eval_placement.roll_free_up`,
# `PLACEMENT_PARAMS` tilt_*), P2-R5's fallback when there are too few normals: at
# least 5 cameras, the 10 % least level-headed trimmed, well conditioned at an
# eigenvalue ratio >= 5. Not tuned here.
RF_MIN_CAMERAS = 5
RF_TRIM_FRACTION = 0.1
RF_MIN_CONDITIONING = 5.0
# The unit `levelled` direction, OpenCV y-down: up is -Y.
LEVEL_UP = np.array([0.0, -1.0, 0.0])

AREA_BUILDS_OFF_DETAIL = (
    "area builds are switched off on this Tower (TOWER_WORLD_AREA_BUILDS); "
    "scripts/world_refinish.py builds them on request")


# WHERE AN AREA IS BUILT, as opposed to where it lives. The area lives at
# `<world>/areas/<area>/` (contract v3 §5.3), and its stages append
# `<stage>/<session>/` and their own file names, so the deepest file an area build
# WRITES -- an appearance chunk's staging name, `c.<32 hex>.bin.p<pid>.<8 hex>.tmp`
# -- sat at root + 166 characters: 222 on this Tower's live root and 265 in a
# scratch copy, past Windows' MAX_PATH (260), where every area appearance failed
# (run wb-coherence-run-2026-09-23, P3-PG re-finish). So the stages run in a SHORT
# directory at the root of the store, `<root>/.ab/<area>` (root + 21 instead of
# root + 64), and the finished stage directories are moved into the area's own
# directory when the build ends, on the same volume. What is written peaks at
# root + 124; what is left in place, and only ever read, at root + 146.
AREA_BUILD_DIRNAME = ".ab"


def area_build_dir(store, area_id: str) -> Path:
    """`<root>/.ab/<area id>`: where one area's stages run before being moved into place."""
    if not C.is_area_id(area_id):
        raise ValueError("not an area id")
    return Path(store.root) / AREA_BUILD_DIRNAME / area_id


class _BuildView(C.AreaStore):
    """The area's view with its directory at the short build directory; everything
    about the session is still the real world's (`components.AreaStore`)."""

    def __init__(self, base, world_id: str, session_id: str, area_id: str, build_dir: Path) -> None:
        super().__init__(base, world_id, session_id, area_id)
        self.final_dir = self._area_dir
        self._area_dir = Path(build_dir)


# THE OWNER MARKER. A build directory holds keyframe imagery (undistorted frames,
# appearance chunks) OUTSIDE its world's directory, so `store.purge_world` would not see
# it -- a privacy rule, not tidiness. Every build directory therefore names its world
# before anything else is written into it, and `purge_area_builds` removes a world's
# build directories by that name, whether or not the area's record still exists.
OWNER_FILENAME = "owner.json"


def _read_owner(bdir: Path) -> dict | None:
    try:
        owner = json.loads((Path(bdir) / OWNER_FILENAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return owner if isinstance(owner, dict) else None


def _pid_alive(pid) -> bool:
    try:
        import psutil  # noqa: PLC0415

        return bool(pid) and psutil.pid_exists(int(pid))
    except Exception:  # noqa: BLE001 -- unknown is alive: never remove a live build's files
        return True


def _remove_tree(path: Path, removed: list, retained: list) -> None:
    """Delete `path` bottom-up like `purge_world`, reporting what could not go."""
    path = Path(path)
    if not path.exists():
        return
    entries = sorted(path.rglob("*"), key=lambda q: len(q.parts), reverse=True) + [path]
    for entry in entries:
        try:
            if entry.is_dir():
                entry.rmdir()
            else:
                entry.unlink()
            removed.append(str(entry))
        except OSError as exc:
            logger.warning("[Tower][WorldBuilder] could not remove %s: %s", entry, exc)
            retained.append(str(entry))


def _belongs_to(store, bdir: Path, world_id: str) -> bool:
    """Whether a build directory is this world's: its marker says so, or -- a directory
    left without one -- its area id is one of this world's areas on disk."""
    owner = _read_owner(bdir)
    if owner is not None:
        return owner.get("world_id") == world_id
    return (Path(store.world_dir(world_id)) / C.AREAS_DIRNAME / bdir.name).exists()


def purge_area_builds(store, world_id: str, *, only_stale: bool = False) -> tuple[list, list]:
    """Remove every area build directory of `world_id` (and `.old.*` leftovers inside
    them). `only_stale`: keep one whose building process is still alive. Returns
    (removed, retained) paths. Called by `store.purge_world` and by the re-redaction
    switch, which invalidate the imagery these directories hold."""
    removed: list = []
    retained: list = []
    root = Path(store.root) / AREA_BUILD_DIRNAME
    if not root.is_dir():
        return removed, retained
    for bdir in sorted(root.iterdir()):
        if not bdir.is_dir() or not _belongs_to(store, bdir, world_id):
            continue
        if only_stale and _pid_alive((_read_owner(bdir) or {}).get("pid")):
            continue
        _remove_tree(bdir, removed, retained)
    try:
        root.rmdir()
    except OSError:
        pass
    return removed, retained


def _sweep_orphans(store) -> list:
    """Build directories whose world no longer exists and whose builder is gone: never
    reused, never served, removed."""
    removed: list = []
    root = Path(store.root) / AREA_BUILD_DIRNAME
    if not root.is_dir():
        return removed
    for bdir in sorted(root.iterdir()):
        if not bdir.is_dir():
            continue
        owner = _read_owner(bdir)
        world = (owner or {}).get("world_id")
        if owner is not None and world and Path(store.world_dir(world)).exists():
            continue
        if owner is None and any(Path(store.world_dir(w)).joinpath(C.AREAS_DIRNAME, bdir.name).exists()
                                 for w in _world_ids(store)):
            continue
        if _pid_alive((owner or {}).get("pid")) and owner is not None:
            continue
        _remove_tree(bdir, removed, [])
    return removed


def _world_ids(store) -> list:
    try:
        return list(store.list_world_ids())
    except Exception:  # noqa: BLE001
        return []


def _fresh_build_dir(store, world_id: str, session_id: str, area_id: str) -> Path:
    """An empty build directory, marked with its owner BEFORE anything is written into
    it. What a killed build left there is its own unpublished output: never reused,
    removed. Orphans of worlds that no longer exist are swept on the way."""
    bdir = area_build_dir(store, area_id)
    _sweep_orphans(store)
    if bdir.exists():
        _remove_tree(bdir, [], [])
        shutil.rmtree(bdir, ignore_errors=True)
    bdir.mkdir(parents=True, exist_ok=True)
    from tower.storage import write_json_atomic  # noqa: PLC0415

    write_json_atomic(bdir / OWNER_FILENAME, {"world_id": world_id, "session_id": session_id,
                                              "area_id": area_id, "pid": os.getpid(),
                                              "created_at": time.time()})
    return bdir


# The order stage directories are moved into place: the surface's inputs first, then
# the surface, then the appearance built on it -- the room's own publishing order.
_PUBLISH_ORDER = ("solve", "dense", "surface", "appearance")


def publish_area_build(store, world_id: str, session_id: str, area_id: str) -> list:
    """Move every stage directory the build produced into the area's own directory.
    Returns the stage names moved. Never raises for a build that produced nothing.

    NEVER HALF OF TWO BUILDS. Every stage directory this build replaces is moved OUT of
    the area first (into the build directory, removed with it), so the area never holds
    one build's surface beside another's appearance; then the new ones go in. The
    area's record says the build finished only after this returns (`build_area`), so a
    process killed half-way through leaves `running` under a dead pid -- owed, rebuilt
    -- never an area recorded complete with half its artifacts."""
    bdir = area_build_dir(store, area_id)
    if not bdir.is_dir():
        return []
    final = C.area_root(store, world_id, session_id, area_id)
    final.mkdir(parents=True, exist_ok=True)
    new = [c.name for c in bdir.iterdir() if c.is_dir() and not c.name.startswith(".old.")]
    new.sort(key=lambda n: (_PUBLISH_ORDER.index(n) if n in _PUBLISH_ORDER else -1, n))
    for name in new:
        target = final / name
        if target.exists():
            old = bdir / f".old.{name}"
            if old.exists():
                shutil.rmtree(old, ignore_errors=True)
            target.replace(old)
    moved = []
    for name in new:
        (bdir / name).replace(final / name)
        moved.append(name)
    _remove_tree(bdir, [], [])
    shutil.rmtree(bdir, ignore_errors=True)
    try:
        bdir.parent.rmdir()   # `.ab` itself, when no other build is running
    except OSError:
        pass
    return moved


class AreaNotBuildable(Exception):
    """This area cannot be prepared, with a sentence for its record."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


# ---------------------------------------------------------------------------
# geometry helpers (P2-R5)
# ---------------------------------------------------------------------------


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / (np.linalg.norm(v) + 1e-15)


def rot_between(a, b) -> np.ndarray:
    """The minimal rotation taking unit a onto unit b."""
    a, b = _unit(a), _unit(b)
    v = np.cross(a, b)
    c = float(a @ b)
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        p = _unit(np.cross(a, [1.0, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1.0, 0]))
        return 2 * np.outer(p, p) - np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / s ** 2)


def camera_normals(z: np.ndarray, K: np.ndarray, step: int = PIX_STEP,
                   exclude: np.ndarray | None = None) -> np.ndarray:
    """Unit normals (camera frame, oriented toward the camera) of a depth map, on a
    `step`-subsampled grid, where the depth is valid. `r5_lib.cam_points`."""
    h, w = z.shape
    vs = np.arange(step // 2, h, step)
    us = np.arange(step // 2, w, step)
    uu, vv = np.meshgrid(us, vs)
    zz = z[vv, uu].astype(np.float64)
    x = (uu + 0.5 - K[0, 2]) / K[0, 0] * zz
    y = (vv + 0.5 - K[1, 2]) / K[1, 1] * zz
    P = np.stack([x, y, zz], -1)
    dx = np.full_like(P, np.nan)
    dy = np.full_like(P, np.nan)
    dx[:, 1:-1] = P[:, 2:] - P[:, :-2]
    dy[1:-1, :] = P[2:, :] - P[:-2, :]
    n = np.cross(dx, dy)
    nn = np.linalg.norm(n, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        n = n / np.where(nn > 0, nn, np.nan)
    flip = (n * P).sum(-1) > 0
    n[flip] *= -1
    ok = np.isfinite(zz) & (zz > DEPTH_VALID_M[0]) & (zz < DEPTH_VALID_M[1])
    if exclude is not None:
        ok &= ~exclude[vv, uu]
    n = n[ok]
    return n[np.isfinite(n).all(1)]


def roll_free_up(Rwc: np.ndarray) -> dict | None:
    """The up of a set of cameras from the level head alone (the harness's
    `eval_placement.roll_free_up`, which P2-R5 falls back to): the direction the
    camera x axes are most nearly perpendicular to, signed by the mean camera up,
    re-estimated once without the least level-headed 10 %. None below
    `RF_MIN_CAMERAS`."""
    Rwc = np.asarray(Rwc, dtype=np.float64).reshape(-1, 3, 3)
    if len(Rwc) < RF_MIN_CAMERAS:
        return None
    x = Rwc[:, :, 0]
    cam_up = -Rwc[:, :, 1]
    keep = np.ones(len(x), bool)
    u = ev = None
    for _ in range(2):
        ev, V = np.linalg.eigh(x[keep].T @ x[keep])
        u = V[:, 0]
        if u @ cam_up[keep].mean(0) < 0:
            u = -u
        k = int(math.floor(RF_TRIM_FRACTION * len(x)))
        if k == 0:
            break
        keep = np.ones(len(x), bool)
        keep[np.argsort(-np.abs(x @ u), kind="stable")[:k]] = False
    cond = float(ev[1] / max(ev[0], 1e-12))
    return {"up": _unit(u), "conditioning": cond, "well_conditioned": cond >= RF_MIN_CONDITIONING,
            "cameras": int(len(x))}


def group_up(Rwc: np.ndarray, normals_world: np.ndarray) -> dict:
    """The up of a rigid group of cameras (`r5_lib.group_up`), or `levelled: False`.

    Candidates lie on the circle spanned by the two smallest eigenvectors of the
    camera x axes (a level head keeps them horizontal), gated to an RMS roll within
    UP_GATE_DEG of the best and to UP_PITCH_CONE of the mean camera up; the winner is
    the candidate most normals align with (floors, tables, ceilings), refined by the
    aligned normals.
    """
    N = normals_world
    if len(N) < UP_MIN_NORMALS or len(Rwc) < 2:
        # P2-R5's FALLBACK (review V6, L5): the level head alone. Without it such an
        # area was drawn in GLOMAP's arbitrary frame, possibly upside down. The up's
        # SIGN is the mean camera up's, so it is never upside down; `levelled` says
        # whether the head-level estimate is well conditioned (the caption warns
        # otherwise). Only with too few cameras for even that is it not estimated.
        rf = roll_free_up(Rwc)
        if rf is None:
            return {"levelled": False, "up": None, "support": None, "ambiguity": None,
                    "normals": int(len(N)),
                    "source": "not estimated: too few depth normals and too few cameras"}
        return {"levelled": bool(rf["well_conditioned"]), "up": [float(v) for v in rf["up"]],
                "support": None, "ambiguity": None, "normals": int(len(N)),
                "conditioning": rf["conditioning"],
                "source": "level head only (P2-R5 roll-free up: too few depth normals)"}
    x = Rwc[:, :, 0]
    _ev, V = np.linalg.eigh(x.T @ x)
    e0, e1 = V[:, 0], V[:, 1]
    cu = _unit((-Rwc[:, :, 1]).mean(0))
    al = np.radians(np.arange(-90, 90, 0.5))
    U = np.cos(al)[:, None] * e0 + np.sin(al)[:, None] * e1
    U *= np.sign(U @ cu)[:, None]
    ok = (U @ cu) > math.cos(math.radians(UP_PITCH_CONE))
    if not ok.any():
        ok[:] = True
    rms = np.degrees(np.arcsin(np.clip(np.sqrt(((x @ U.T) ** 2).mean(0)), 0, 1)))
    ok &= rms <= rms[ok].min() + UP_GATE_DEG
    ct = math.cos(math.radians(UP_NORMAL_TOL))
    sc = (np.abs(N @ U.T) > ct).sum(0).astype(float)
    sc[~ok] = -1
    j = int(np.argmax(sc))
    u = U[j]
    far = ok & (np.abs(U @ u) < math.cos(math.radians(30)))
    amb = float(sc[far].max() / max(sc[j], 1)) if far.any() else 0.0
    for _ in range(5):
        c = N @ u
        w = np.abs(c) > ct
        if w.sum() < 50:
            break
        u = _unit((N[w] * np.sign(c[w])[:, None]).sum(0))
    if u @ cu < 0:
        u = -u
    return {"levelled": True, "up": [float(v) for v in u],
            "support": float(sc[j] / len(N)), "ambiguity": amb, "normals": int(len(N)),
            "source": "level head + level surfaces (P2-R5 group_up)"}


# ---------------------------------------------------------------------------
# the area's solution
# ---------------------------------------------------------------------------


def _pose_wc(entry: dict):
    r_cw = np.asarray(entry["rotation"], dtype=np.float64).reshape(3, 3)
    t_cw = np.asarray(entry["translation"], dtype=np.float64).reshape(3)
    r_wc = r_cw.T
    return r_wc, -r_wc @ t_cw


def area_solution(solution, keyframe_ids, *, rotation=None, centre=None, tag: str = ""):
    """The published solution cut to `keyframe_ids`, all component 0, optionally
    rotated about `centre` (x' = R (x - c)). Returns `(solution, members)`."""
    from tower.world_builder.global_solve import Solution, withheld_rgb  # noqa: PLC0415

    wanted = set(keyframe_ids)
    index = {k: i for i, k in enumerate(solution.keyframe_ids)}
    members = []
    for kid in solution.keyframe_ids:
        pose = (solution.poses or {}).get(kid)
        if (kid in wanted and pose and pose.get("rotation") is not None
                and pose.get("translation") is not None):
            members.append(kid)
    if not members:
        raise AreaNotBuildable("none of this area's keyframes is posed in the published solve")
    R = np.eye(3) if rotation is None else np.asarray(rotation, np.float64)
    c0 = np.zeros(3) if centre is None else np.asarray(centre, np.float64)
    member_idx = np.array(sorted(index[k] for k in members), np.int64)
    obs = np.asarray(solution.observations, np.int64).reshape(-1, 3)
    keep = np.isin(obs[:, 0], member_idx)
    used = np.unique(obs[keep, 2]) if keep.any() else np.zeros(0, np.int64)
    remap = -np.ones(len(solution.xyz), np.int64)
    remap[used] = np.arange(len(used))
    o = obs[keep]
    new_obs = np.stack([o[:, 0], o[:, 1], remap[o[:, 2]]], 1) if len(o) else np.zeros((0, 3), np.int64)
    oxy = np.asarray(solution.observation_xy, np.float32).reshape(-1, 2)
    new_xy = oxy[keep] if len(oxy) == len(obs) else np.zeros((0, 2), np.float32)
    xyz = (np.asarray(solution.xyz, np.float64)[used] - c0) @ R.T
    first = np.full(len(used), np.iinfo(np.int64).max, np.int64)
    if len(new_obs):
        np.minimum.at(first, new_obs[:, 2], new_obs[:, 0])
    track = np.bincount(new_obs[:, 2], minlength=len(used)) if len(new_obs) else np.zeros(len(used), np.int64)
    err = np.asarray(solution.error, np.float32).reshape(-1)
    error = err[used] if len(err) == len(solution.xyz) else np.zeros(len(used), np.float32)
    poses = {}
    for kid in members:
        entry = dict(solution.poses[kid])
        r_wc, centre_w = _pose_wc(entry)
        r_wc2 = R @ r_wc
        c2 = R @ (centre_w - c0)
        r_cw2 = r_wc2.T
        entry.update({"component": 0, "rotation": r_cw2.reshape(-1).tolist(),
                      "translation": (-r_cw2 @ c2).tolist()})
        poses[kid] = entry
    supported = sum(1 for k in members if int(poses[k].get("observations") or 0) >= 30)
    digest = hashlib.sha256(json.dumps({
        "solution": solution.input_digest, "members": sorted(members), "tag": tag,
        "rotation": np.round(R, 9).tolist(), "centre": np.round(c0, 9).tolist(),
    }, sort_keys=True).encode()).hexdigest()
    sub = Solution(
        solver=f"{solution.solver} (area)", solved_at=solution.solved_at,
        input_digest=digest, keyframe_ids=list(solution.keyframe_ids), poses=poses,
        components=[{"index": 0, "images": len(members), "images_supported": supported,
                     "points": int(len(used)), "mean_error_px": None}],
        xyz=xyz.astype(np.float32), rgb=withheld_rgb(len(used)),
        component=np.zeros(len(used), np.int32),
        first_keyframe=np.where(first == np.iinfo(np.int64).max, 0, first).astype(np.int32),
        track_length=track.astype(np.int32), error=error.astype(np.float32),
        observations=new_obs.astype(np.int32), observation_xy=new_xy,
        camera=solution.camera,
        timing={"area": {"source_input_digest": solution.input_digest, "tag": tag}},
        transients=getattr(solution, "transients", None),
        solve=getattr(solution, "solve", None),
        # The room's gate record travels with the area's solution: its stages key their
        # depth (the camera's FoV told or not) on it, as the room's do (review V7, L-a).
        gate=getattr(solution, "gate", None))
    return sub, members


def estimate_up(solution, members, align: dict, work: Path) -> dict:
    """The area's vertical from its own depth stage's predictions (camera frame
    normals rotated by the area's poses)."""
    cam = solution.camera or {}
    K = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1.0]])
    W, H = int(cam["width"]), int(cam["height"])
    kind = align.get("kind", "depth")
    rwc_of = {k: _pose_wc(solution.poses[k])[0] for k in members}
    normals = []
    for rec in align.get("records") or []:
        if not rec.get("ok"):
            continue
        ki = int(rec.get("ki", -1))
        kid = rec.get("kid") or (solution.keyframe_ids[ki]
                                 if 0 <= ki < len(solution.keyframe_ids) else None)
        if kid not in rwc_of:
            continue
        pred_path = work / "depth" / f"{ki:05d}_pred.npy"
        try:
            pred = np.load(pred_path).astype(np.float64)
        except (OSError, ValueError):
            continue
        if kind == "depth":
            # MoGe's z: metric depth, the harness's own normals source (P2-R5).
            z = pred
        else:
            from tower.world_builder.dense import depth_from_prediction  # noqa: PLC0415

            z = depth_from_prediction(pred, float(rec.get("a", 1.0)),
                                      float(rec.get("b", 0.0)), kind)
        Kz = K.copy()
        if z.shape != (H, W):
            Kz[0] *= z.shape[1] / W
            Kz[1] *= z.shape[0] / H
        fill = None
        try:
            fill_arr = np.load(work / "depth" / f"{ki:05d}_fill.npy")
            if fill_arr.shape == z.shape:
                fill = fill_arr.astype(bool)
        except (OSError, ValueError):
            fill = None
        n = camera_normals(z, Kz, exclude=fill)
        if len(n):
            normals.append(n @ rwc_of[kid].T)
    N = np.concatenate(normals) if normals else np.zeros((0, 3))
    if len(N) > NORMALS_MAX:
        N = N[np.random.default_rng(0).choice(len(N), NORMALS_MAX, replace=False)]
    Rwc = np.array([rwc_of[k] for k in members])
    return group_up(Rwc, N)


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------


def area_recorder(store, world_id: str, session_id: str, area_id: str, components_sha1: str):
    """`record(stage, state=, detail=, attempted=)` for `final_surface_stages`, writing
    the AREA's record. Never fatal, like the builder's own recorder."""

    def record(stage, *, state, detail=None, attempted=True):
        try:
            C.write_area_record(store, world_id, session_id, area_id,
                                components_sha1=components_sha1, stage=stage,
                                state=state, detail=detail, attempted=attempted)
        except Exception:  # noqa: BLE001
            logger.exception("[Tower][WorldBuilder] could not record the area's %s stage",
                             stage)

    return record


def decline_area(store, world_id: str, session_id: str, area_id: str,
                 components_sha1: str, detail: str) -> None:
    """Both stages `unavailable`, not attempted: a settled "nothing is coming"
    (`unattempted`), never owed for ever."""
    record = area_recorder(store, world_id, session_id, area_id, components_sha1)
    for stage in (STAGE_SURFACE, STAGE_APPEARANCE):
        record(stage, state=STAGE_STATE_UNAVAILABLE, attempted=False, detail=detail)


def owed_area_ids(store, world_id: str, session_id: str, session, record) -> dict:
    """`{area_id: word}` for this session's areas whose build is unsettled (owed,
    running or unobservable), in the record's order. Reads only."""
    from tower.world_builder.photographic import is_unsettled  # noqa: PLC0415

    words = C.area_words(store, world_id, session_id, session, record)
    return {aid: w for aid, w in words.items() if is_unsettled(w.get("state"))}


def _stamp_manifest(path: Path, area_id: str, levelled: bool) -> None:
    """The additive `area: {id, levelled}` key on an area manifest (§5.3)."""
    from tower.storage import write_json_atomic  # noqa: PLC0415

    try:
        man = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if not isinstance(man, dict):
        return
    man["area"] = {"id": area_id, "levelled": bool(levelled)}
    write_json_atomic(path, man)


def stamp_area_manifests(view: C.AreaStore, world_id: str, session_id: str,
                         levelled: bool) -> None:
    root = view.world_dir(world_id)
    for stage in ("surface", "appearance"):
        _stamp_manifest(root / stage / session_id / "manifest.json", view.area_id, levelled)


# ---------------------------------------------------------------------------
# preparing an area
# ---------------------------------------------------------------------------


def prepare_area(store, world_id: str, session_id: str, area_id: str, record, *,
                 should_stop=None, level: bool = True, depth_known_fov: bool = False,
                 depth_backend: str | None = None,
                 build_dir: Path | None = None) -> tuple[C.AreaStore, dict]:
    """Write the area's own levelled solution under its directory; return the view to
    build it through and the levelling record. Raises `AreaNotBuildable`.

    The depth stage runs here once, on the unlevelled solution, for the vertical;
    `surfacify` then refits those predictions to the levelled one. `depth_known_fov`
    must be what the surface stage will run with, or the predictions are made twice
    (correct, only slower).
    """
    from tower.world_builder.global_solve import (  # noqa: PLC0415
        load_solution,
        workspace_for,
        write_solution,
    )
    from tower.world_builder.surface import SurfaceParams  # noqa: PLC0415

    entry = record.entry(area_id)
    if entry is None or entry["shown_as"] != C.SHOWN_AREA:
        raise AreaNotBuildable("this is not an area of the current components record")
    kids = entry.get("keyframe_ids")
    if not kids:
        raise AreaNotBuildable("the components record does not name this area's keyframes")
    solution = load_solution(store, world_id, session_id)
    if solution is None:
        raise AreaNotBuildable("no global solution for this session")
    session = store.read_session(world_id, session_id)
    intrinsics = session.intrinsics
    if intrinsics is None or getattr(intrinsics, "fx", None) is None:
        raise AreaNotBuildable("session has no intrinsics")

    view = (C.AreaStore(store, world_id, session_id, area_id) if build_dir is None
            else _BuildView(store, world_id, session_id, area_id, build_dir))
    src = store.world_dir(world_id) / "solve" / session_id
    dst = view.world_dir(world_id) / "solve" / session_id
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("sources.json", "camera.json"):
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)

    first, members = area_solution(solution, kids, tag=f"{area_id}:unlevelled")
    centres = np.array([_pose_wc(first.poses[k])[1] for k in members])
    centre = np.median(centres, axis=0)
    levelling = {"levelled": False, "up": None, "support": None, "ambiguity": None,
                 "normals": 0, "source": "not attempted", "members": len(members)}
    rotation = np.eye(3)
    if level:
        write_solution(workspace_for(view, world_id, session_id), first)
        from tower.world_builder.surface_pipeline import (  # noqa: PLC0415
            _status as surface_status,
        )
        from tower.world_builder.surface_pipeline import (  # noqa: PLC0415
            ensure_depth_stage,
            surface_dir,
        )

        # A LIVE PROCESS IS WORKING ON THIS AREA FROM HERE, and the status file is what
        # the liveness probe reads (C1 E13): without it the ~10 s depth pass before
        # `surfacify` writes its own status read `owed`, not `running`. `surfacify`
        # overwrites it; a process killed here leaves `running` under a dead pid,
        # which reads as owed again.
        surface_status(surface_dir(view, world_id, session_id), state=STAGE_STATE_RUNNING,
                       stage="level")

        params = SurfaceParams()
        kwargs = {"known_fov": True} if depth_known_fov else {}
        align, work = ensure_depth_stage(
            view, world_id, session_id, first, intrinsics, gate_rel=params.gate_rel,
            backend=depth_backend, should_stop=should_stop,
            imagery_source=params.imagery_source, **kwargs)
        if should_stop is not None and should_stop():
            raise AreaNotBuildable("stopped before the area's vertical was estimated")
        levelling.update(estimate_up(first, members, align, work))
        # Rotated whenever an up was estimated -- a poorly conditioned head-level up
        # still has the right SIGN, and not rotating is how an area came out upside
        # down (review V6, L5); `levelled` stays what the estimate supports.
        if levelling.get("up") is not None:
            rotation = rot_between(np.asarray(levelling["up"]), LEVEL_UP)
    levelled = bool(levelling.get("levelled"))
    final, _members = area_solution(solution, kids, rotation=rotation, centre=centre,
                                    tag=f"{area_id}:{'levelled' if levelled else 'solve-orientation'}")
    write_solution(workspace_for(view, world_id, session_id), final)
    levelling.update({"rotation": rotation.tolist(), "centre": centre.tolist(),
                      "levelled": levelled, "scaled": False,
                      "note": "its own frame and units; never placed or scaled to the room"})
    C.write_area_record(store, world_id, session_id, area_id,
                        components_sha1=record.sha1, levelled=levelled, levelling=levelling)
    return view, levelling


def _mark_running(store, world_id: str, session_id: str, area_id: str) -> None:
    """`running` (stage `level`) in the area's OWN surface directory. Never fatal."""
    try:
        from tower.world_builder.surface_pipeline import (  # noqa: PLC0415
            _status as surface_status,
        )
        from tower.world_builder.surface_pipeline import surface_dir  # noqa: PLC0415

        surface_status(surface_dir(C.AreaStore(store, world_id, session_id, area_id), world_id,
                                   session_id), state=STAGE_STATE_RUNNING, stage="level")
    except Exception:  # noqa: BLE001
        logger.debug("[Tower][WorldBuilder] could not mark the area running", exc_info=True)


def _settle_status(store, world_id: str, session_id: str, area_id: str, state: str,
                   detail: str) -> None:
    """Replace a `running` status the preparation wrote, when it ends without
    `surfacify` (which writes its own). Never fatal."""
    try:
        from tower.world_builder.surface_pipeline import (  # noqa: PLC0415
            _status as surface_status,
        )
        from tower.world_builder.surface_pipeline import surface_dir  # noqa: PLC0415

        root = surface_dir(C.AreaStore(store, world_id, session_id, area_id), world_id,
                           session_id)
        status = root / "status.json"
        if status.exists() and json.loads(status.read_text()).get("stage") == "level":
            surface_status(root, state=state, detail=detail)
    except Exception:  # noqa: BLE001
        logger.debug("[Tower][WorldBuilder] could not settle the area's status",
                     exc_info=True)


def build_area(store, world_id: str, session_id: str, area_id: str, record, *,
               final_surface_stages, appearance: bool = True, prune_depth_work: bool = True,
               should_stop=lambda: False, stop_source=lambda: None, level: bool = True,
               depth_known_fov: bool = False) -> dict:
    """Prepare one area and run the builder's final stages on it.

    `final_surface_stages` is `scripts/world_build_session.final_surface_stages`,
    passed in by the script that calls this (a product module does not import a
    script). The surface is marked `running` BEFORE the preparation, so a process
    killed anywhere from here leaves the signature the finisher recovers.
    """
    record_now = area_recorder(store, world_id, session_id, area_id, record.sha1)
    report = {"area_id": area_id}
    record_now(STAGE_SURFACE, state=STAGE_STATE_RUNNING)
    # TERMINAL RECORDS WAIT FOR THE MOVE. The stages record `ok` / `failed` as they end,
    # which is before their output is in the area's directory; recorded then, a process
    # killed during the move left an area recorded complete with half its artifacts.
    # `running` is written at once (liveness); everything else after the publish.
    deferred: list = []

    def record_stage(stage, *, state, detail=None, attempted=True):
        if state == STAGE_STATE_RUNNING:
            record_now(stage, state=state, detail=detail, attempted=attempted)
        else:
            deferred.append((stage, state, detail, attempted))

    def flush_records():
        while deferred:
            stage, state, detail, attempted = deferred.pop(0)
            record_now(stage, state=state, detail=detail, attempted=attempted)
    t0 = time.time()
    # THE SHORT BUILD DIRECTORY (`AREA_BUILD_DIRNAME`). The liveness probe reads the
    # area's OWN directory, so the `running` status goes there too: a process killed
    # anywhere from here leaves `running` under a dead pid there -- owed -- whatever
    # the build directory holds.
    bdir = _fresh_build_dir(store, world_id, session_id, area_id)
    _mark_running(store, world_id, session_id, area_id)
    try:
        view, levelling = prepare_area(store, world_id, session_id, area_id, record,
                                       should_stop=should_stop, level=level,
                                       depth_known_fov=depth_known_fov, build_dir=bdir)
    except AreaNotBuildable as exc:
        report["published"] = publish_area_build(store, world_id, session_id, area_id)
        stopped = should_stop()
        state = STAGE_STATE_STOPPED if stopped else STAGE_STATE_UNAVAILABLE
        _settle_status(store, world_id, session_id, area_id, state, exc.reason)
        record_stage(STAGE_SURFACE, state=state, attempted=not stopped, detail=exc.reason)
        record_stage(STAGE_APPEARANCE, state=state, attempted=False,
                     detail=f"the surface was not built: {exc.reason}")
        flush_records()
        report.update({"built": False, "reason": exc.reason})
        return report
    except BaseException:
        exc = sys.exc_info()[1]
        try:
            publish_area_build(store, world_id, session_id, area_id)
        except Exception:  # noqa: BLE001 -- the original error is the one to raise
            logger.exception("[Tower][WorldBuilder] could not move the area's build into place")
        record_stage(STAGE_SURFACE, state=STAGE_STATE_FAILED,
                     detail=f"{type(exc).__name__}: {exc}")
        _settle_status(store, world_id, session_id, area_id, STAGE_STATE_FAILED,
                       f"{type(exc).__name__}: {exc}")
        record_stage(STAGE_APPEARANCE, state=STAGE_STATE_UNAVAILABLE, attempted=False,
                     detail="the area could not be prepared; there was nothing to shade")
        flush_records()
        raise
    report["levelling"] = {k: levelling.get(k) for k in
                           ("levelled", "support", "ambiguity", "normals", "source")}
    report["prepare_s"] = round(time.time() - t0, 1)
    t1 = time.time()
    try:
        report["stages"] = final_surface_stages(
            view, world_id, session_id, solved=True, appearance=appearance,
            prune_depth_work=prune_depth_work, should_stop=should_stop,
            stop_source=stop_source, record=record_stage)
        report["stages_s"] = round(time.time() - t1, 1)
        try:
            stamp_area_manifests(view, world_id, session_id, bool(levelling.get("levelled")))
        except Exception:  # noqa: BLE001 -- the key is additive; the artifact stands
            logger.exception("[Tower][WorldBuilder] could not stamp the area manifests")
    finally:
        # Whatever the stages produced -- finished, stopped or failed, with its status
        # saying which -- goes into the area's directory, where it is read; and only
        # then does the record say how the stages ended.
        report["published"] = publish_area_build(store, world_id, session_id, area_id)
        flush_records()
    report["built"] = True
    return report
