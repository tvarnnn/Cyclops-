"""The consistency field and the plane snap, on synthetic worlds whose truth
is known exactly.

`depth_consistency.py` exists because frames that each look smooth disagree
with each other about where a wall is, and the TSDF turns that into crumple.
These tests build a box room seen by overlapping cameras, give every frame a
low-order depth error an affine fit cannot remove (a tilt and a warp), and ask:

  * do the frames become consistent, and closer to the truth
  * is a field that makes held-out anchors worse refused
  * does a failed solve fall back to the plain affine, and say so
  * is the cached field reused exactly when it is theirs, and warm-started
    when it is not
  * does surface fusion actually use the corrected depth

and of `surface.snap_planes`:

  * is a wall's undulation flattened, and a small object on it left alone
  * does a plane never reach past the surface that was reconstructed
"""

from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import pytest

from tests.test_world_builder_surface import ROOM, _look_from

from tower.world_builder import depth_consistency as DC
from tower.world_builder import surface as S
from tower.world_builder import surface_pipeline as SP

WORLD, SESSION = "wc", "sc"
FX, W, H = 110.0, 160, 120


def _K():
    return np.array([[FX, 0, W / 2], [0, FX, H / 2], [0, 0, 1.0]])


def _ray_depth(R, t, K, w=W, h=H):
    """Exact depth of the box room, production pixel convention (pixel u's ray
    is (u - cx) / fx, as the fusion and the fit use)."""
    C = -R.T @ t
    uu, vv = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
    dc = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu)], -1)
    d = dc @ R
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (ROOM - C) / d
        t0 = (-ROOM - C) / d
    tmax = np.maximum(t0, t1).min(-1)
    return (tmax * 1.0).astype(np.float32)  # d has unit z in camera: tmax IS depth


def _perturb(z, i):
    """A per-frame error no affine removes: a tilt across the image and a
    vertical warp, plus scale and offset the fit does remove."""
    uu, vv = np.meshgrid(np.arange(W) / W - 0.5, np.arange(H) / H - 0.5)
    tilt = 0.8 * ((i % 3) - 1)
    warp = 2.0 * np.cos(1.3 * i)
    return (z * (1.0 + 0.03 * np.sin(i)) + 0.05 + tilt * uu + warp * vv * vv).astype(np.float32)


def _world(tmp_path, *, n_frames=10, digest="d1", perturb=True, seed=0):
    """A solved box room with shared sparse tracks and a depth stage on disk."""
    import cv2

    from tower.world_builder.dense import DenseParams, align_frame
    from tower.world_builder.dense_pipeline import FILL_RULE
    from tower.world_builder.global_solve import Solution, load_solution, workspace_for, write_solution
    from tower.world_builder.records import CameraIntrinsics, Session, World
    from tower.world_builder.store import WorldStore

    store = WorldStore(tmp_path)
    store.write_world(World(world_id=WORLD, created_at=1.0, updated_at=2.0,
                            session_ids=(SESSION,)))
    K = _K()
    store.write_session(Session(
        session_id=SESSION, world_id=WORLD, started_at=1.0,
        redaction="faces-detected-and-filled/yunet-2023mar@0.30+plausibility1",
        intrinsics=CameraIntrinsics(source="self_calibrated", model="pinhole", fx=FX, fy=FX,
                                    cx=W / 2, cy=H / 2, calibrated_width=W,
                                    calibrated_height=H)))
    rng = np.random.default_rng(seed)
    # points on the walls, floor and ceiling
    face = rng.integers(0, 6, 3000)
    P = rng.uniform(-ROOM, ROOM, (3000, 3))
    P[np.arange(3000), face // 2] = np.where(face % 2 == 0, -ROOM, ROOM)
    kids, poses, truth, obs, obs_xy = [], {}, [], [], []
    for i in range(n_frames):
        x = -1.2 + 2.4 * i / max(n_frames - 1, 1)
        eye = np.array([x, 0.3 * np.sin(i), 1.0])
        target = np.array([0.4 * x, -0.4 + 0.2 * np.cos(i), -ROOM])
        R, t = _look_from(eye, target)
        kid = f"{SESSION}:{i:08d}"
        kids.append(kid)
        poses[kid] = {"component": 0, "rotation": R.ravel().tolist(),
                      "translation": t.tolist(), "observations": 50}
        z = _ray_depth(R, t, K)
        truth.append(z)
        pc = P @ R.T + t
        with np.errstate(divide="ignore", invalid="ignore"):
            u = pc[:, 0] / pc[:, 2] * FX + W / 2
            v = pc[:, 1] / pc[:, 2] * FX + H / 2
        inb = (pc[:, 2] > 0.1) & (u >= 1) & (u < W - 2) & (v >= 1) & (v < H - 2)
        ui = np.clip(np.rint(np.nan_to_num(u)).astype(int), 0, W - 1)
        vi = np.clip(np.rint(np.nan_to_num(v)).astype(int), 0, H - 1)
        seen = inb & (np.abs(z[vi, ui] - pc[:, 2]) < 0.02 * pc[:, 2])
        for pid in np.nonzero(seen)[0]:
            obs.append((i, len(obs), pid))
            obs_xy.append((u[pid], v[pid]))
    camera = {"fx": FX, "fy": FX, "cx": W / 2, "cy": H / 2, "width": W, "height": H}
    obs = np.array(obs, np.int32)
    write_solution(workspace_for(store, WORLD, SESSION), Solution(
        solver="glomap", solved_at=time.time(), input_digest=digest, keyframe_ids=kids,
        poses=poses, components=[{"index": 0, "images": n_frames, "points": len(P)}],
        xyz=P.astype(np.float32), rgb=np.full((len(P), 3), 128, np.uint8),
        component=np.zeros(len(P), np.int32), first_keyframe=np.zeros(len(P), np.int32),
        track_length=np.full(len(P), 3, np.int32), error=np.full(len(P), 0.5, np.float32),
        observations=obs, observation_xy=np.array(obs_xy, np.float32), camera=camera))
    solution = load_solution(store, WORLD, SESSION)

    dense = store.world_dir(WORLD) / "dense" / SESSION
    (dense / "work" / "depth").mkdir(parents=True, exist_ok=True)
    (dense / "work" / "undist").mkdir(parents=True, exist_ok=True)
    images = store.images_dir(WORLD, SESSION)
    images.mkdir(parents=True, exist_ok=True)
    records = []
    for i, (kid, z) in enumerate(zip(kids, truth)):
        pred = _perturb(z, i) if perturb else z
        np.save(dense / "work" / "depth" / f"{i:05d}.npy", pred.astype(np.float16))
        np.save(dense / "work" / "depth" / f"{i:05d}_pred.npy", pred.astype(np.float16))
        np.save(dense / "work" / "depth" / f"{i:05d}_fill.npy", np.zeros(z.shape, bool))
        img = np.full((H, W, 3), 140, np.uint8)
        cv2.imwrite(str(dense / "work" / "undist" / f"{i:05d}.jpg"), img)
        _ok, enc = cv2.imencode(".jpg", img)
        (images / f"{i:08d}.jpg").write_bytes(enc.tobytes())
        m = obs[:, 0] == i
        pc = P[obs[m, 2]] @ np.asarray(poses[kid]["rotation"]).reshape(3, 3).T \
            + np.asarray(poses[kid]["translation"])
        xy = np.array(obs_xy)[m]
        ui, vi = np.rint(xy[:, 0]).astype(int), np.rint(xy[:, 1]).astype(int)
        a, b, ho = align_frame(pred[vi, ui].astype(float), pc[:, 2], "depth")
        records.append({"ki": i, "kid": kid, "ok": True, "a": a, "b": b,
                        "held_out_rel": min(ho or 0.01, 0.07), "n_points": int(m.sum()),
                        "z_sparse_min": float(pc[:, 2].min()),
                        "z_sparse_max": float(pc[:, 2].max()),
                        "image_sha1": hashlib.sha1(enc.tobytes()).hexdigest(),
                        "fill_rule": FILL_RULE})
    align = {"kind": "depth", "camera": camera, "backend": DenseParams().backend,
             "input_digest": digest, "digest": digest, "fill_rule": FILL_RULE,
             "records": records}
    (dense / "align.json").write_text(json.dumps(align))
    return store, solution, dense, align, truth


def _frames(align, dense, solution, params=None):
    return SP._Frames(align, dense / "work", solution, params or S.SurfaceParams())


CP = DC.ConsistencyParams(per_frame=256, outer=4, inner=30, warm_outer=1, k_top=6, k_loop=0,
                          min_shared=8, max_seconds=600.0)


def _cpu():
    import torch

    return torch.device("cpu")


def _depth_errors(frames, truth, field):
    """Median relative error against the truth, plain affine vs corrected, and
    the median across-frame spread of depth at the same world point."""
    import torch

    params = S.SurfaceParams()
    med = 3.0
    errs_aff, errs_cor = [], []
    for ki, a, b, R, t, _ho, zmax in frames.items:
        pred, _img, _fill = frames.load(ki, image=False)
        z = pred.astype(np.float32)  # kind depth
        z = (z - b) / a
        zc = field.correct_numpy(ki, z) if field is not None else z
        ok, _ = S.depth_validity(torch.as_tensor(z), frames.K, params, med,
                                 max_depth=S.depth_bound(params, med, zmax))
        ok = ok.numpy()
        errs_aff.append(np.abs(z[ok] - truth[ki][ok]) / truth[ki][ok])
        errs_cor.append(np.abs(zc[ok] - truth[ki][ok]) / truth[ki][ok])
    return float(np.median(np.concatenate(errs_aff))), float(np.median(np.concatenate(errs_cor)))


# ---------------------------------------------------------------------------
# the field
# ---------------------------------------------------------------------------


class TestFramesBecomeConsistent:

    def test_a_tilt_and_a_warp_the_affine_cannot_remove_are_removed(self, tmp_path):
        store, solution, dense, align, truth = _world(tmp_path)
        frames = _frames(align, dense, solution)
        field, rec = DC.solve_field(frames, solution, S.SurfaceParams(), 3.0, cparams=CP,
                                    device=_cpu())
        assert rec["state"] == DC.STATE_APPLIED, rec.get("reason")
        before, after = rec["heldout"]["before"], rec["heldout"]["after"]
        # the frames agree with each other, on pixels the solve never sampled
        assert after["cross"]["median"] < 0.5 * before["cross"]["median"], rec["heldout"]
        # and with sparse points the solve never saw
        assert after["sfm"]["median"] < before["sfm"]["median"]
        # and with the truth
        e_aff, e_cor = _depth_errors(frames, truth, field)
        assert e_aff > 0.01, "the fixture must start inconsistent"
        assert e_cor < 0.5 * e_aff, (e_aff, e_cor)

    def test_the_field_is_the_same_evaluated_densely_and_at_a_point(self):
        rng = np.random.default_rng(3)
        G = rng.normal(0, 0.05, (1, 9 * 6))
        O = rng.normal(0, 0.05, (1, 9 * 6))
        f = DC.ConsistencyField([7], G, O, [2.0], (3, 6), True, (H, W))
        z = np.full((H, W), 3.0, np.float32)
        dense = f.correct_numpy(7, z)
        import torch

        fld = DC._Field(1, (3, 6), True, _cpu())
        with torch.no_grad():
            fld.G.copy_(torch.as_tensor(G, dtype=torch.float32))
            fld.O.copy_(torch.as_tensor(O, dtype=torch.float32))
        u = torch.tensor([0.0, 37.0, 159.0]); v = torch.tensor([0.0, 64.0, 119.0])
        pt = fld.eval(torch.zeros(3, dtype=torch.long), u, v, torch.full((3,), 3.0),
                      torch.tensor([2.0]), W, H).detach().numpy()
        assert np.allclose(pt, dense[[0, 64, 119], [0, 37, 159]], atol=1e-4)


class TestHeldOutChecksDecide:

    def test_a_field_that_makes_held_out_anchors_worse_is_refused(self, tmp_path, monkeypatch):
        store, solution, dense, align, truth = _world(tmp_path)
        frames = _frames(align, dense, solution)

        def harmful(D, field, *a, **k):
            import torch

            with torch.no_grad():
                field.G.fill_(0.2)       # every frame 22% too deep
            return [], "done"

        monkeypatch.setattr(DC, "_optimise", harmful)
        field, rec = DC.solve_field(frames, solution, S.SurfaceParams(), 3.0, cparams=CP,
                                    device=_cpu())
        assert rec["state"] == DC.STATE_REFUSED
        assert "held-out anchor error got worse" in rec["reason"]
        assert field is None

    def test_a_refused_field_leaves_the_fused_depth_the_plain_affine(self, tmp_path, monkeypatch):
        import torch

        store, solution, dense, align, truth = _world(tmp_path)
        frames = _frames(align, dense, solution)
        monkeypatch.setattr(DC, "_optimise",
                            lambda D, field, *a, **k: (field.G.data.fill_(0.2), ([], "done"))[1])
        res = DC.ensure_consistency(dense, frames, solution, S.SurfaceParams(), 3.0, cparams=CP,
                                    device=_cpu())
        assert res.state == DC.STATE_REFUSED and res.field is None
        assert not (dense / DC.FIELD_FILE).exists()
        frames.correction = res.field
        got = next(frames.prepared(S.SurfaceParams(), 3.0, torch.device("cpu")))[0]
        ki, a, b = frames.items[0][:3]
        pred, _i, _f = frames.load(ki, image=False)
        assert np.allclose(got.numpy(), (pred - b) / a, atol=1e-5)


class TestFailureFallsBackToTheAffine:

    def test_a_solve_that_raises_is_recorded_and_not_cached(self, tmp_path, monkeypatch):
        store, solution, dense, align, truth = _world(tmp_path)
        frames = _frames(align, dense, solution)

        def boom(*a, **k):
            raise RuntimeError("CUDA error: an illegal memory access")

        res = DC.ensure_consistency(dense, frames, solution, S.SurfaceParams(), 3.0,
                                    cparams=CP, solve=boom)
        assert res.state == DC.STATE_FAILED and res.field is None
        rec = json.loads((dense / DC.RECORD_FILE).read_text())
        assert rec["state"] == DC.STATE_FAILED and "illegal memory access" in rec["reason"]
        calls = []
        DC.ensure_consistency(dense, frames, solution, S.SurfaceParams(), 3.0, cparams=CP,
                              solve=lambda *a, **k: (calls.append(1), (None, {"state": "skipped"}))[1])
        assert calls, "a FAILED solve must be retried, not reused"

    def test_a_diverged_solve_is_failed_not_applied(self, tmp_path, monkeypatch):
        store, solution, dense, align, truth = _world(tmp_path)
        frames = _frames(align, dense, solution)
        monkeypatch.setattr(DC, "_optimise", lambda *a, **k: ([], "diverged"))
        field, rec = DC.solve_field(frames, solution, S.SurfaceParams(), 3.0, cparams=CP,
                                    device=_cpu())
        assert field is None and rec["state"] == DC.STATE_FAILED

    def test_the_surface_still_builds_and_its_manifest_says_why(self, tmp_path, monkeypatch):
        store, solution, dense, align, truth = _world(tmp_path, perturb=False)

        def boom(*a, **k):
            raise RuntimeError("out of memory")

        monkeypatch.setattr(DC, "solve_field", boom)
        params = S.SurfaceParams(voxel_frac=0.03, lod_face_targets=(0,), mobile_page_bytes=0,
                                 min_weight=0.5, min_support_frames=1, min_component_frac=0.0,
                                 smooth_iterations=1, plane_snap=False)
        r = SP.surfacify(store, WORLD, SESSION, params=params)
        assert r.state == SP.STATE_OK, r.detail
        man = SP.read_surface_manifest(store, WORLD, SESSION)
        assert man["detail"]["depth_consistency"]["state"] == "failed"
        assert "out of memory" in man["detail"]["depth_consistency"]["reason"]
        # and a failed field is not "already built": the next build tries again
        again = []
        monkeypatch.setattr(DC, "solve_field",
                            lambda *a, **k: (again.append(1), (None, {"state": "skipped",
                                                                      "reason": "x"}))[1])
        SP.surfacify(store, WORLD, SESSION, params=params)
        assert again


class TestTheCacheIsTheirs:

    def _spy(self, calls, state="applied"):
        def solve(frames, solution, sp, med, **kw):
            calls.append(kw)
            ki = [it[0] for it in frames.items]
            f = DC.ConsistencyField(ki, np.zeros((len(ki), 54)), np.zeros((len(ki), 54)),
                                    np.ones(len(ki)), (3, 6), True, (H, W))
            return (f if state == "applied" else None), {"state": state, "frames": len(ki)}
        return solve

    def test_unchanged_inputs_reuse_and_changed_inputs_resolve(self, tmp_path):
        store, solution, dense, align, truth = _world(tmp_path)
        frames = _frames(align, dense, solution)
        calls = []
        sp = S.SurfaceParams()
        r1 = DC.ensure_consistency(dense, frames, solution, sp, 3.0, cparams=CP,
                                   solve=self._spy(calls))
        r2 = DC.ensure_consistency(dense, frames, solution, sp, 3.0, cparams=CP,
                                   solve=self._spy(calls))
        assert len(calls) == 1 and r2.reused and r2.field is not None
        # a refit affine
        align2 = json.loads(json.dumps(align))
        align2["records"][3]["a"] *= 1.01
        DC.ensure_consistency(dense, _frames(align2, dense, solution), solution, sp, 3.0,
                              cparams=CP, solve=self._spy(calls))
        assert len(calls) == 2
        # a changed gate or validity rule
        sp2 = S.SurfaceParams(edge_rel=0.05)
        DC.ensure_consistency(dense, _frames(align2, dense, solution, sp2), solution, sp2, 3.0,
                              cparams=CP, solve=self._spy(calls))
        assert len(calls) == 3
        # a changed solver
        DC.ensure_consistency(dense, _frames(align2, dense, solution, sp2), solution, sp2, 3.0,
                              cparams=DC.ConsistencyParams(cells=(2, 4)), solve=self._spy(calls))
        assert len(calls) == 4

    def test_a_new_solve_changes_the_key(self, tmp_path):
        store, solution, dense, align, truth = _world(tmp_path, digest="d1")
        frames = _frames(align, dense, solution)
        sp = S.SurfaceParams()
        k1 = DC.consistency_key(frames, solution, sp, CP, 5, 2)
        solution.input_digest = "d2"
        assert DC.consistency_key(frames, solution, sp, CP, 5, 2) != k1

    def test_a_missing_field_file_is_not_a_cache(self, tmp_path):
        store, solution, dense, align, truth = _world(tmp_path)
        frames = _frames(align, dense, solution)
        calls = []
        DC.ensure_consistency(dense, frames, solution, S.SurfaceParams(), 3.0, cparams=CP,
                              solve=self._spy(calls))
        (dense / DC.FIELD_FILE).unlink()
        DC.ensure_consistency(dense, frames, solution, S.SurfaceParams(), 3.0, cparams=CP,
                              solve=self._spy(calls))
        assert len(calls) == 2

    def test_the_consistency_solver_is_in_the_surface_params_digest(self):
        on, off = S.SurfaceParams(), S.SurfaceParams(depth_consistency=False)
        assert SP._params_digest(on, "d") != SP._params_digest(off, "d")
        live = S.SurfaceParams.live()
        assert live.consistency_warm_outer <= on.consistency_warm_outer


class TestWarmStart:

    def test_a_new_solve_starts_from_the_previous_field(self, tmp_path, monkeypatch):
        store, solution, dense, align, truth = _world(tmp_path, digest="d1")
        frames = _frames(align, dense, solution)
        sp = S.SurfaceParams()
        first = DC.ensure_consistency(dense, frames, solution, sp, 3.0, cparams=CP, device=_cpu())
        assert first.state == DC.STATE_APPLIED
        seen = {}
        real = DC._optimise

        def spy(D, field, anc, pairs, samples, cparams, outer, gate0, should_stop, log):
            seen["G0"] = field.G.detach().clone().numpy()
            seen["outer"] = outer
            seen["gate0"] = gate0
            return real(D, field, anc, pairs, samples, cparams, outer, gate0, should_stop, log)

        monkeypatch.setattr(DC, "_optimise", spy)
        solution.input_digest = "d2"          # the next live solve landed
        second = DC.ensure_consistency(dense, frames, solution, sp, 3.0, cparams=CP,
                                       device=_cpu())
        assert second.record["warm_start"] == {"used": True, "frames": len(frames),
                                                "outer": sp.consistency_warm_outer}
        assert seen["outer"] == sp.consistency_warm_outer and seen["gate0"] == CP.warm_gate0
        assert np.allclose(seen["G0"], first.field.G.reshape(len(frames), -1))
        assert second.state == DC.STATE_APPLIED

    def test_a_cold_solve_starts_at_the_affine(self, tmp_path, monkeypatch):
        store, solution, dense, align, truth = _world(tmp_path)
        frames = _frames(align, dense, solution)
        seen = {}

        def spy(D, field, *a, **k):
            seen["G0"] = field.G.detach().clone().numpy()
            return [], "stopped"

        monkeypatch.setattr(DC, "_optimise", spy)
        _f, rec = DC.solve_field(frames, solution, S.SurfaceParams(), 3.0, cparams=CP,
                                 device=_cpu())
        assert not rec["warm_start"]["used"] and not seen["G0"].any()
        assert rec["state"] == DC.STATE_STOPPED


class TestFusionUsesTheCorrectedDepth:

    def test_prepared_yields_corrected_depth(self, tmp_path):
        import torch

        store, solution, dense, align, truth = _world(tmp_path, perturb=False)
        frames = _frames(align, dense, solution)
        ki = [it[0] for it in frames.items]
        G = np.full((len(ki), 54), np.log(1.1))
        frames.correction = DC.ConsistencyField(ki, G, np.zeros((len(ki), 54)), np.ones(len(ki)),
                                                (3, 6), True, (H, W))
        z, ok, *_ = next(frames.prepared(S.SurfaceParams(), 3.0, torch.device("cpu")))
        k0, a, b = frames.items[0][:3]
        pred, _i, _f = frames.load(k0, image=False)
        base = (pred - b) / a
        assert np.allclose(z.numpy()[ok.numpy()], 1.1 * base[ok.numpy()], rtol=1e-4)

    def test_surfacify_applies_the_field_and_records_it(self, tmp_path, monkeypatch):
        store, solution, dense, align, truth = _world(tmp_path)
        real = DC.solve_field
        monkeypatch.setattr(DC, "solve_field",
                            lambda *a, **k: real(*a, **{**k, "cparams": CP}))
        # ensure_consistency passes its own cparams; route them through ours
        monkeypatch.setattr(DC, "ConsistencyParams", lambda: CP)
        params = S.SurfaceParams(voxel_frac=0.03, lod_face_targets=(0,), mobile_page_bytes=0,
                                 min_weight=0.5, min_support_frames=1, min_component_frac=0.0,
                                 smooth_iterations=1, plane_snap=False)
        r = SP.surfacify(store, WORLD, SESSION, params=params)
        assert r.state == SP.STATE_OK, r.detail
        man = SP.read_surface_manifest(store, WORLD, SESSION)
        dc = man["detail"]["depth_consistency"]
        assert dc["state"] == "applied", dc
        assert (dense / DC.FIELD_FILE).exists() and (dense / DC.RECORD_FILE).exists()
        # align.json keeps the plain affine for provenance
        assert json.loads((dense / "align.json").read_text())["records"] == align["records"]
        assert "consistency" in man["seconds"]


# ---------------------------------------------------------------------------
# the plane snap
# ---------------------------------------------------------------------------


VOX = 0.02


def _grid_mesh(xs, ys, height):
    """A triangulated height field z = height(x, y) over a grid."""
    X, Y = np.meshgrid(xs, ys)
    Z = height(X, Y)
    V = np.stack([X, Y, Z], -1).reshape(-1, 3)
    nx = len(xs)
    idx = np.arange(len(V)).reshape(len(ys), nx)
    a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    c, d = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    F = np.concatenate([np.stack([a, c, b], 1), np.stack([b, c, d], 1)])
    return V.astype(np.float32), F.astype(np.int64)


def _wall_views(n=10, zmin=None):
    """Cameras 3 units in front of the plane z = 0, looking at it (-z)."""
    import torch

    K = np.array([[120.0, 0, 80], [0, 120.0, 60], [0, 0, 1]])
    views = []
    for i in range(n):
        eye = np.array([0.5 + 2.0 * i / max(n - 1, 1), 1.5, 3.0])
        R, t = _look_from(eye, eye * np.array([1, 1, 0]), up=(0.0, -1.0, 0.0))
        C = eye
        uu, vv = np.meshgrid(np.arange(160, dtype=float), np.arange(120, dtype=float))
        d = np.stack([(uu - 80) / 120, (vv - 60) / 120, np.ones_like(uu)], -1) @ R
        depth = (0.0 - C[2]) / d[..., 2]   # camera z of the plane hit
        views.append((torch.as_tensor(depth.astype(np.float16)),
                      torch.ones(depth.shape, dtype=torch.bool),
                      torch.as_tensor(R), torch.as_tensor(t)))
    return views, K


def _snap_params(**kw):
    base = dict(snap_min_area_frac=0.1, snap_tol_voxels=2.5, snap_min_frames=4)
    base.update(kw)
    return S.SurfaceParams(**base)


def _snap(V, F, views, K, params):
    return S.snap_planes(V, F, views, K, params, VOX, 3.0, device=_cpu())


class TestPlaneSnap:

    def _wall(self, bump=True):
        xs = np.arange(0, 3.0 + 1e-9, VOX)
        ys = np.arange(0, 3.0 + 1e-9, VOX)
        box = (0.9, 1.1, 1.4, 1.6)      # a 0.2 x 0.2 object standing 0.15 off the wall

        def h(X, Y):
            z = 0.8 * VOX * np.sin(X * 7.0) * np.cos(Y * 5.0)      # 0.8 voxel undulation
            if bump:
                inside = (X >= box[0]) & (X <= box[1]) & (Y >= box[2]) & (Y <= box[3])
                z = np.where(inside, 0.15, z)
            return z
        V, F = _grid_mesh(xs, ys, h)
        return V, F, box

    def test_a_wall_is_flattened_and_a_small_object_on_it_is_not(self):
        V, F, box = self._wall()
        views, K = _wall_views()
        Vs, rec = _snap(V, F, views, K, _snap_params())
        assert rec["plane_count"] >= 1, rec
        on_wall = np.abs(V[:, 2]) < 0.05
        assert np.std(Vs[on_wall, 2]) < 0.25 * np.std(V[on_wall, 2])
        top = V[:, 2] > 0.14
        assert top.any()
        assert np.allclose(Vs[top], V[top], atol=1e-6), "the object was flattened"

    def test_a_plane_never_reaches_past_the_reconstructed_surface(self):
        """Past x = 3 the surface lifts away from the wall at 15 degrees -- inside
        the normal gate, outside the distance gate. Snapping must not pull it
        onto the wall: that would extend the plane over a surface that is not
        on it."""
        xs = np.arange(0, 4.0 + 1e-9, VOX)
        ys = np.arange(0, 3.0 + 1e-9, VOX)
        slope = np.tan(np.radians(15))

        def h(X, Y):
            return np.where(X > 3.0, (X - 3.0) * slope, 0.8 * VOX * np.sin(X * 7) * np.cos(Y * 5))
        V, F = _grid_mesh(xs, ys, h)
        views, K = _wall_views()
        Vs, rec = _snap(V, F, views, K, _snap_params())
        assert rec["plane_count"] >= 1
        tol = 2.5 * VOX
        # nothing moves off its own place in the plane, and nothing moves far
        assert np.allclose(Vs[:, :2], V[:, :2], atol=0.1 * VOX)
        assert np.abs(Vs - V).max() <= 2 * tol + 1e-6
        # the flat region does not grow over the slope
        flat_before = V[np.abs(V[:, 2]) < 0.5 * VOX, 0].max()
        flat_after = Vs[np.abs(Vs[:, 2]) < 0.5 * VOX, 0].max()
        assert flat_after <= flat_before + 3 * VOX, (flat_before, flat_after)

    def test_a_plane_no_frame_measured_is_not_snapped(self):
        V, F, _ = self._wall(bump=False)
        views, K = _wall_views(n=3)
        Vs, rec = _snap(V, F, views, K, _snap_params(snap_min_frames=8))
        assert rec["plane_count"] == 0 and rec["rejected"] >= 1
        assert np.array_equal(Vs, V)

    def test_a_small_plane_is_not_snapped(self):
        V, F, _ = self._wall(bump=False)
        views, K = _wall_views()
        # 9 u^2 of wall against a gate of 1.2 * 3^2 = 10.8
        Vs, rec = _snap(V, F, views, K, _snap_params(snap_min_area_frac=1.2))
        assert rec["plane_count"] == 0 and np.array_equal(Vs, V)

    def test_the_snap_is_in_the_surface_params_digest(self):
        a, b = S.SurfaceParams(), S.SurfaceParams(snap_tol_voxels=3.0)
        assert SP._params_digest(a, "d") != SP._params_digest(b, "d")

    def test_the_manifest_records_planes_and_areas(self, tmp_path):
        from tests.test_world_builder_surface_pipeline import _synthetic_world

        store = _synthetic_world(tmp_path, anchors=True)
        params = S.SurfaceParams(voxel_frac=0.02, lod_face_targets=(0,), mobile_page_bytes=0,
                                 min_weight=0.5, min_support_frames=1, min_component_frac=0.0,
                                 smooth_iterations=2, snap_min_frames=1,
                                 snap_min_area_frac=0.1, depth_consistency=False)
        r = SP.surfacify(store, "w1", "s1", params=params)
        assert r.state == SP.STATE_OK, r.detail
        snap = SP.read_surface_manifest(store, "w1", "s1")["detail"]["plane_snap"]
        assert snap["plane_count"] >= 1
        assert len(snap["plane_areas"]) == snap["plane_count"]
        assert all(a >= snap["min_area"] for a in snap["plane_areas"])
        assert "seconds" in snap and "rejected_examples" not in snap
