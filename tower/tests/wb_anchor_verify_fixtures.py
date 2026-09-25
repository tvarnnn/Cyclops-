"""Synthetic solves for the anchor-verification tests (`test_world_builder_anchor_verify*.py`) and the golden
record of the gate's and the publish step's outputs with `TOWER_WORLD_ANCHOR_VERIFY` unset.

KEEP IN STEP with RUN/experiments/P4-PROD/golden_record.py, which ran `golden_outputs` against the product tree
BEFORE P4-PROD changed it (working tree of d649f9f) and wrote `golden/world_builder_gate_d649f9f.json`.

Not a test module: no `test_` prefix. Everything here is deterministic.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SID = "s1"
ALWAYS = {"solved_at", "solve_identity", "timing", "seconds", "gate_seconds", "map_s", "gate_s", "database",
          "swept", "frozen_at", "workspace", "started_at", "updated_at", "at", "path", "detail_path"}


def rz(deg):
    t = math.radians(deg)
    return np.array([[math.cos(t), -math.sin(t), 0.0], [math.sin(t), math.cos(t), 0.0], [0.0, 0.0, 1.0]])


def name(i: int) -> str:
    return f"{i:08d}.jpg"


def keyframes(n, t0=1000.0, dt=0.5, segments=None):
    from tower.world_builder.records import Keyframe

    return [Keyframe(keyframe_id=f"{SID}:{i:08d}", session_id=SID, source_seq=i, received_at=t0 + dt * i,
                     image_relpath=f"images/{i:08d}.jpg", width=320, height=240, byte_count=1,
                     segment_index=0 if segments is None else int(segments[i]))
            for i in range(n)]


def islands_of(sizes):
    out, s = [], 0
    for n in sizes:
        out.append(list(range(s, s + n)))
        s += n
    return out


def multi_solution(sizes, offsets=None, *, centres=None, transients=("state", "applied")):
    """Islands of one solver component (island 0 = the room's). Camera i has R_cw = Rz(3 i) Rz(offset of its
    island): an island's offset turns it as a whole, so its own links stay honoured and its links to the
    others are contradicted by the offset. Every camera sees 30 tracks with each of its two neighbours;
    60 points couple each island to island 0."""
    from tower.world_builder.global_solve import Solution

    isl = islands_of(sizes)
    n = sum(sizes)
    off = dict(offsets or {})
    kids = [f"{SID}:{i:08d}" for i in range(n)]
    island_of = {c: k for k, idx in enumerate(isl) for c in idx}
    obs, p = [], 0
    for idx in isl:
        for j in range(len(idx)):
            for _ in range(30):
                for c in idx[j:j + 3]:
                    obs.append([c, len(obs), p])
                p += 1
    for k in range(1, len(isl)):
        for q in range(60):
            obs.append([isl[0][q % len(isl[0])], len(obs), p])
            obs.append([isl[k][q % len(isl[k])], len(obs), p])
            p += 1
    obs = np.asarray(obs, np.int32)
    counts = np.bincount(obs[:, 0], minlength=n)
    poses = {}
    for i, kid in enumerate(kids):
        R = rz(3.0 * i) @ rz(float(off.get(island_of[i], 0.0)))
        C = np.zeros(3) if centres is None else np.asarray(centres[i], np.float64)
        poses[kid] = {"component": 0, "rotation": R.ravel().tolist(), "translation": (-R @ C).tolist(),
                      "observations": int(counts[i])}
    first = np.zeros(p, np.int32)
    for row in obs[::-1]:
        first[row[2]] = row[0]
    return Solution(
        solver="glomap", solved_at=1234.5, input_digest="digest-final", keyframe_ids=kids, poses=poses,
        components=[{"index": 0, "images": n, "points": p}],
        xyz=np.zeros((p, 3), np.float32), rgb=np.zeros((p, 3), np.uint8), component=np.zeros(p, np.int32),
        first_keyframe=first, track_length=np.full(p, 3, np.int32), error=np.full(p, 0.5, np.float32),
        observations=obs, observation_xy=np.zeros((len(obs), 2), np.float32),
        camera={"fx": 300.0, "fy": 300.0, "cx": 160.0, "cy": 120.0, "width": 320, "height": 240},
        transients=dict([transients]) if transients else None, solve={"seed": 7})


def multi_links(sizes, cross):
    """Within each island: gaps 1 and 2 at 100 inliers. `cross`: extra links at 40 inliers. The two-view
    rotations are the draw-0 solve's own (no offsets), so an island turned by an offset contradicts them."""
    out = {}
    for idx in islands_of(sizes):
        for j in range(len(idx)):
            for d in (1, 2):
                if j + d < len(idx):
                    out[(name(idx[j]), name(idx[j + d]))] = 100
    for a, b in cross:
        out[(name(a), name(b))] = 40
    rots = {(a, b): rz(3.0 * int(b[:8])) @ rz(3.0 * int(a[:8])).T for a, b in out}
    return out, rots


class Store:
    def __init__(self, root: Path | None = None):
        self.root = root

    def read_session(self, world_id, session_id):
        return SimpleNamespace(started_at=1000.0, intrinsics=None)

    def world_dir(self, world_id):
        return Path(self.root) / world_id


def patch_gate_inputs(mp, links, rots, levels):
    """The gate's readers and its depth and scale stages, faked on `mp` (a pytest MonkeyPatch)."""
    from tower.world_builder import coherence_gate as CG
    from tower.world_builder import coherence_publish as CP

    mp.setattr(CG, "read_verified_links", lambda db, min_inliers=15: dict(links))
    mp.setattr(CG, "read_link_rotations", lambda db, cam, min_inliers=15: dict(rots))

    def depth(store, world_id, session_id, solution, intrinsics, should_stop=None, **kw):
        return ({"backend": "moge2-vitl", "known_fov": 42.0, "targets": len(solution.keyframe_ids),
                 "records": []}, "work-dir", CP._depth_params())

    def metric(solution, name_of, database_path, work):
        ml = {name(i): float(levels[i]) for i in range(len(levels)) if levels[i] is not None}
        return {"metric_log": ml, "cameras_published": len(levels), "cameras_measured": len(ml)}

    mp.setattr(CP, "run_gate_depth", depth)
    mp.setattr(CP, "measure_metric_scale", metric)


def normalise(obj):
    """The comparison's ALWAYS exclusions (RUN/lead/dchk/compare_dchk.py): timestamps, timings, paths."""
    if isinstance(obj, dict):
        return {k: normalise(v) for k, v in obj.items() if k not in ALWAYS}
    if isinstance(obj, list):
        return [normalise(v) for v in obj]
    return obj


def _read(path: Path):
    return normalise(json.loads(path.read_text(encoding="utf-8"))) if path.exists() else None


# ---------------------------------------------------------------------------------------------------------------
# the golden scenarios


CONSENSUS_SIZES = (30, 20, 20, 6)
# Island k >= 1 is joined to the room by a closed triangle through room camera 30 - k.
CONSENSUS_CROSS = ((29, 30), (29, 31), (28, 50), (28, 51), (27, 70), (27, 71))
# Draw 0 attaches B, C and D; draw 1 turns C and D away, draw 2 B and D: the consensus is {A, B, C}, draw 0
# agrees best (it differs on D's 6 only), so D is WITHHELD by the re-gate.
CONSENSUS_OFFSETS = ({}, {2: 40.0, 3: 40.0}, {1: 40.0, 3: 40.0})


def _publish(tmp: Path, sizes, cross, levels, *, offsets=({},), draws=1):
    import pytest

    from tower.world_builder import coherence_publish as CP
    from tower.world_builder.global_solve import SolveWorkspace, write_solution

    links, rots = multi_links(sizes, cross)
    n = sum(sizes)
    tmp.mkdir(parents=True, exist_ok=True)
    ws = SolveWorkspace(tmp / "w1" / "solve" / SID)
    with pytest.MonkeyPatch.context() as mp:
        patch_gate_inputs(mp, links, rots, levels)
        plan = None
        if draws >= 2:
            cands = [multi_solution(sizes, o) for o in offsets]
            plan = CP.ConsensusPlan(draws=draws, seed=7, map_draw=lambda seed: cands[seed - 7])
        out, record = CP.gate_and_publish(Store(tmp), "w1", SID, ws, multi_solution(sizes, offsets[0]),
                                          final=True, gate=True, database_path="db", keyframes=keyframes(n),
                                          write=write_solution, consensus=plan)
    return {"record": normalise(json.loads(json.dumps(record, default=str))),
            "solution.json": _read(ws.solution_path),
            "components.json": _read(ws.root / CP.COMPONENTS_FILENAME),
            "consensus.json": _read(ws.root / CP.CONSENSUS_FILENAME)}


def mid_stretch_model(n=60, lo=25, hi=35):
    """One chain (a biconnected anchor) whose middle stretch [lo, hi) sits at x2 the room's metric level."""
    from tower.world_builder import coherence_gate as CG

    names = [name(i) for i in range(n)]
    R = np.stack([rz(3.0 * i) for i in range(n)])
    obs_i, obs_p, p = [], [], 0
    for j in range(n):
        for _ in range(13):
            for c in range(j, min(n, j + 3)):
                obs_i.append(c)
                obs_p.append(p)
            p += 1
    for c in range(n):
        for _ in range(30):
            obs_i.append(c)
            obs_p.append(p)
            p += 1
    obs_i, obs_p = np.asarray(obs_i), np.asarray(obs_p)
    model = CG.SolveModel(names=names, component=np.zeros(n, np.int64), R_cw=R,
                          n_obs=np.bincount(obs_i, minlength=n), obs_image=obs_i, obs_point=obs_p, n_points=p)
    links = {(name(i), name(i + d)): 100 for i in range(n) for d in (1, 2) if i + d < n}
    rots = {(a, b): R[int(b[:8])] @ R[int(a[:8])].T for a, b in links}
    levels = {name(i): (math.log(2.0) if lo <= i < hi else 0.0) + 0.001 * (i % 3) for i in range(n)}
    return model, links, rots, levels


def golden_outputs(tmp: Path) -> dict:
    """Every output the golden pins, for the code as it is imported now."""
    from tower.world_builder import coherence_gate as CG

    out = {}
    n = 50
    out["single_triangle"] = _publish(tmp / "a", (30, 20), ((29, 30), (29, 31)), [0.0] * n)
    out["single_link_other_level"] = _publish(tmp / "b", (30, 20), ((29, 30),),
                                              [0.0] * 30 + [math.log(2.0)] * 20)
    n = sum(CONSENSUS_SIZES)
    out["consensus_withhold"] = _publish(tmp / "c", CONSENSUS_SIZES, CONSENSUS_CROSS, [0.0] * n,
                                         offsets=CONSENSUS_OFFSETS, draws=3)
    model, links, rots, levels = mid_stretch_model()
    out["apply_gate_mid_stretch"] = normalise(json.loads(json.dumps(
        CG.apply_gate(model, links, levels, link_rotations=rots, masks_applied=True))))
    out["apply_gate_hooks"] = normalise(json.loads(json.dumps(
        CG.apply_gate(model, links, levels, link_rotations=rots, masks_applied=True, withhold=[name(40)],
                      room=[name(0), name(40)]))))
    out["params_digest"] = CG.GateParams().digest()
    return out


def golden_json(tmp: Path) -> str:
    return json.dumps(golden_outputs(tmp), sort_keys=True, indent=0)



# ---------------------------------------------------------------------------------------------------------------
# synthetic evidence for the anchor verification


def line_centres(n, step=0.05):
    """Camera centres along a line (solve units), so translation directions are defined."""
    return [np.array([step * i, 0.02 * (i % 7), 0.0]) for i in range(n)]


def poses_of(solution, n):
    """(R_cw, C) per keyframe index of a `multi_solution`."""
    R, C = {}, {}
    for i in range(n):
        p = solution.poses[f"{SID}:{i:08d}"]
        Rw = np.asarray(p["rotation"], np.float64).reshape(3, 3)
        R[i] = Rw
        C[i] = -Rw.T @ np.asarray(p["translation"], np.float64)
    return R, C


def synthetic_pairs(R, C, pairs, *, error_deg=None, t_reliable=True, parallax=10.0, inliers=200):
    """The masked tier's arrays for `pairs` [(i, j)], each consistent with the solve (R, C) up to a rotation error
    about the x axis of `error_deg[(i, j)]` degrees (default 0)."""
    err = dict(error_deg or {})
    I, J, Rs, ts = [], [], [], []
    for i, j in sorted((min(a, b), max(a, b)) for a, b in pairs):
        Rij = R[j] @ R[i].T
        e = math.radians(float(err.get((i, j), 0.0)))
        Rx = np.array([[1, 0, 0], [0, math.cos(e), -math.sin(e)], [0, math.sin(e), math.cos(e)]])
        t = R[j] @ (C[i] - C[j])
        nt = np.linalg.norm(t)
        I.append(i)
        J.append(j)
        Rs.append(Rx @ Rij)
        ts.append(t / nt if nt > 0 else np.array([1.0, 0.0, 0.0]))
    P = len(I)
    return {"i": np.asarray(I, np.int32), "j": np.asarray(J, np.int32), "R": np.asarray(Rs).reshape(-1, 3, 3),
            "t": np.asarray(ts).reshape(-1, 3), "t_reliable": np.full(P, bool(t_reliable)),
            "parallax_deg": np.full(P, float(parallax), np.float32), "n_inliers": np.full(P, int(inliers), np.int32),
            "n_matches": np.full(P, int(inliers), np.int32)}


def chain_pairs(n, breaks=()):
    """Adjacent pairs (gaps 1 and 2) over 0..n-1, none crossing a break b (between b-1 and b)."""
    out = []
    for i in range(n):
        for d in (1, 2):
            j = i + d
            if j < n and not any(i < b <= j for b in breaks):
                out.append((i, j))
    return out
