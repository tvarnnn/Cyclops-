"""The coherence solver-variant driver (experiment code) on synthetic data.

Covered: capture-chain ordering and staged names (lexical order == capture
order, extras interleave, chained captures continue the sequence); staging
undistorts exactly as `global_solve.prepare_images` and refuses a
non-canonical extra; the rigid gate on a synthetic two-island model (zero
shared points -> split; strongly shared and seed-stable -> kept; shared but
seed-unstable -> split; one seed -> spread unmeasured); the keyframes-only
export; the VariantConfig JSON round trip; the product-recipe predicate.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tower.world_builder.coherence_exp import driver as D
from tower.world_builder.coherence_exp import gate as G

W, H = 360, 640
SID = "a" * 32
WID = "b" * 32


# ---------------------------------------------------------------------------
# a synthetic frozen world with a chained capture


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def make_world(tmp_path: Path, *, chain_sizes=(6, 5), keyframe_every=2, seed=0):
    import cv2

    rng = np.random.default_rng(seed)
    captures = tmp_path / "captures"
    cids = [f"{i:032x}" for i in range(1, len(chain_sizes) + 1)]
    source_seq = 1
    all_frames = []
    for n, (cid, size) in enumerate(zip(cids, chain_sizes)):
        cdir = captures / cid
        (cdir / "frames").mkdir(parents=True)
        rows = []
        for _ in range(size):
            name = f"{source_seq:08d}.jpg"
            img = (rng.random((H, W, 3)) * 255).astype(np.uint8)
            cv2.imwrite(str(cdir / "frames" / name), img)
            rows.append({"source_seq": source_seq, "relpath": f"frames/{name}", "received_at": 100.0 + source_seq})
            all_frames.append((cid, source_seq, name))
            source_seq += 3  # gaps in source_seq, as on real captures
        _write_jsonl(cdir / "frames.jsonl", rows)
        (cdir / "capture.json").write_text(json.dumps({
            "capture_id": cid, "started_at": 50.0 + n, "continues_capture": cids[n - 1] if n else None}),
            encoding="utf-8")
    world = tmp_path / "worlds" / WID
    sdir = world / "sessions" / SID
    (sdir / "images").mkdir(parents=True)
    (world / "world.json").write_text(json.dumps({"world_id": WID}), encoding="utf-8")
    (sdir / "session.json").write_text(json.dumps({
        "session_id": SID, "capture_id": cids[0], "started_at": 0.0, "ended_at": 1e9,
        "intrinsics": {"fx": 400.0, "fy": 400.0, "cx": 180.0, "cy": 320.0,
                       "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0], "calibrated_width": W, "calibrated_height": H}}),
        encoding="utf-8")
    kfs = []
    for i, (cid, seq, name) in enumerate(all_frames):
        if i % keyframe_every:
            continue
        kfs.append({"keyframe_id": f"{SID}:{seq:08d}", "source_seq": seq, "image_relpath": f"images/{name}",
                    "width": W, "height": H, "segment_index": 0})
        cv2.imwrite(str(sdir / "images" / name), np.zeros((H, W, 3), np.uint8))  # a "redacted" copy
    _write_jsonl(sdir / "keyframes.jsonl", kfs)
    return world, captures, all_frames, kfs


def test_capture_index_follows_the_chain_and_orders_keyframes(tmp_path):
    world, captures, frames, kfs = make_world(tmp_path)
    ci = D.capture_index(world, captures)
    assert len(ci.chain) == 2
    assert [f["capture_seq"] for f in ci.frames] == list(range(len(frames)))
    assert [f["source_seq"] for f in ci.frames] == [s for _, s, _ in frames]
    seqs = [ci.keyframe_seq(k["keyframe_id"]) for k in kfs]
    assert seqs == sorted(seqs)
    # a keyframe in the second capture continues the sequence
    last = kfs[-1]
    assert ci.keyframe_seq(last["keyframe_id"]) >= 6
    between = ci.between(seqs[0], seqs[1])
    assert len(between) == 1 and not between[0]["is_keyframe"]


def test_staged_names_sort_in_capture_order_with_extras_interleaved(tmp_path):
    import cv2

    world_dir, captures, frames, kfs = make_world(tmp_path)
    world = D.open_world(world_dir)
    ci = D.capture_index(world_dir, captures)
    gaps = [f for f in ci.frames if not f["is_keyframe"]][:3]
    extras = []
    for f in gaps:
        raw = cv2.imread(f["path"])
        und = D.undistort_frame(world_dir, raw, world=world)
        p = tmp_path / f"extra_{f['capture_seq']}.jpg"
        cv2.imwrite(str(p), und)
        extras.append(D.ExtraImage(name_key=f"{f['capture_id']}/{f['source_seq']}", undistorted_path=str(p),
                                   capture_seq=f["capture_seq"]))
    cfg = D.VariantConfig(extra_images=extras)
    ws = tmp_path / "ws"
    st = D.stage_images(world, ci, ws, cfg, log=lambda s: None)
    names = [r["name"] for r in st["images"]]
    assert names == sorted(names)
    seqs = [r["capture_seq"] for r in st["images"]]
    assert seqs == sorted(seqs)
    kinds = [r["kind"] for r in st["images"]]
    assert kinds[:4] == ["keyframe", "extra", "keyframe", "extra"]
    assert all(n.endswith("_k.jpg") for n, k in zip(names, kinds) if k == "keyframe")
    assert all(n.endswith("_g.jpg") for n, k in zip(names, kinds) if k == "extra")
    # keyframes come from the RAW capture (product: raw frame when on disk)
    assert all(r["source"] == "raw" for r in st["images"] if r["kind"] == "keyframe")
    doc = json.loads((ws / "staging.json").read_text(encoding="utf-8"))
    assert doc["keyframes"] == len(kfs) and doc["extras"] == 3


def test_staging_matches_prepare_images_bytes(tmp_path):
    """The staged keyframe is the image the product's prepare_images writes."""
    import cv2

    from tower.world_builder import global_solve as gs

    world_dir, captures, frames, kfs = make_world(tmp_path)
    world = D.open_world(world_dir)
    ci = D.capture_index(world_dir, captures)
    st = D.stage_images(world, ci, tmp_path / "ws", D.VariantConfig(), log=lambda s: None)
    first = st["images"][0]
    raw = cv2.imread(first["source_path"])

    class Intr:
        fx, fy, cx, cy = 400.0, 400.0, 180.0, 320.0
        dist_coeffs = [0.0] * 5
        calibrated_width, calibrated_height = W, H

    m1, m2, (x, y, rw, rh), cam = gs._undistort_maps(Intr(), W, H)
    und = cv2.remap(raw, m1, m2, cv2.INTER_LINEAR)[y:y + rh, x:x + rw]
    ok, buf = cv2.imencode(".jpg", und, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert (tmp_path / "ws" / "images" / first["name"]).read_bytes() == buf.tobytes()
    assert st["camera"] == cam.to_json_dict()


def test_a_non_canonical_extra_is_refused(tmp_path):
    import cv2

    world_dir, captures, frames, kfs = make_world(tmp_path)
    world = D.open_world(world_dir)
    ci = D.capture_index(world_dir, captures)
    p = tmp_path / "raw.jpg"
    cv2.imwrite(str(p), np.zeros((H, W, 3), np.uint8))  # raw size, not canonical
    cfg = D.VariantConfig(extra_images=[D.ExtraImage("x", str(p), 1)])
    with pytest.raises(ValueError, match="canonical"):
        D.stage_images(world, ci, tmp_path / "ws", cfg, log=lambda s: None)


# ---------------------------------------------------------------------------
# config


def _augment_stub(database_path, name_to_image_id, images_dir, workspace, **kw):
    return {"called": True, **kw}


def test_config_round_trip_and_product_predicate():
    cfg = D.VariantConfig(name="A3", masks="transients", gate="rigid", seeds=[3, 1],
                          extra_images=[{"name_key": "k", "undistorted_path": "p.jpg", "capture_seq": 7}],
                          augment=_augment_stub, augment_kwargs={"a": 1}, gate_params={"k_shared": 5})
    d = json.loads(json.dumps(cfg.to_json()))
    back = D.VariantConfig.from_json(d)
    assert back == cfg
    assert back.augment == f"{__name__}:_augment_stub"
    assert D.resolve_augment(back.augment) is _augment_stub
    assert isinstance(back.extra_images[0], D.ExtraImage)
    assert not back.is_product_recipe()
    assert D.VariantConfig().is_product_recipe()
    with pytest.raises(ValueError):
        D.VariantConfig(masks="everything")
    with pytest.raises(ValueError):
        D.VariantConfig(gate="maybe")


# ---------------------------------------------------------------------------
# the gate on a synthetic two-island model


def _rot(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    Kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * Kx + (1 - math.cos(angle)) * Kx @ Kx


def two_islands(seed, *, shared_between=0, dispersed=True, b_offset=np.zeros(3), b_rot=0.0, noise=0.0,
                n_a=30, n_b=20, pts_per_cam=40):
    """Island A cameras on a line x in [0, 10]; island B on a line beyond. Each
    camera observes points shared with its neighbours (tracks of length 3).
    `shared_between` points link A and B: `dispersed` = each by a different
    (A camera, B camera) pair, so no camera pair shares more than one point
    (separate rigid groups at k >= 2, coupled at group level); else all by the
    last A and first B camera (one rigid group).
    B is moved by (b_rot about z, b_offset) -- the free gauge of a detached island."""
    rng = np.random.default_rng(seed)
    n = n_a + n_b
    names = [f"{i:06d}_k.jpg" for i in range(n)]
    centres = np.zeros((n, 3))
    centres[:n_a, 0] = np.linspace(0, 10, n_a)
    centres[n_a:, 0] = np.linspace(12, 20, n_b)
    centres += rng.normal(0, noise, centres.shape)
    Rb = _rot([0, 0, 1], b_rot)
    pivot = centres[n_a:].mean(0)
    centres[n_a:] = (Rb @ (centres[n_a:] - pivot).T).T + pivot + b_offset
    R_cw = np.repeat(np.eye(3)[None], n, axis=0)
    t_cw = -np.einsum("nij,nj->ni", R_cw, centres)
    obs_i, obs_p, xyz, pcomp = [], [], [], []
    p = 0
    for island in (range(0, n_a), range(n_a, n)):
        idx = list(island)
        for j, i in enumerate(idx):
            for _ in range(pts_per_cam // 3):
                track = [idx[k] for k in range(j, min(j + 3, len(idx)))]
                for c in track:
                    obs_i.append(c)
                    obs_p.append(p)
                xyz.append(centres[i] + [0, 0, 3])
                pcomp.append(0)
                p += 1
    for j in range(shared_between):
        for c in ((j % n_a, n_a + j % n_b) if dispersed else (n_a - 1, n_a)):
            obs_i.append(c)
            obs_p.append(p)
        xyz.append(centres[n_a] + [0, 0, 3])
        pcomp.append(0)
        p += 1
    counts = np.bincount(obs_i, minlength=n)
    return G.SeedModel(seed=seed, names=names, component=np.zeros(n, np.int64), R_cw=R_cw, t_cw=t_cw,
                       n_obs=counts.astype(np.int64), xyz=np.asarray(xyz), point_component=np.asarray(pcomp),
                       obs_image=np.asarray(obs_i), obs_point=np.asarray(obs_p),
                       obs_uv=np.zeros((len(obs_i), 2)))


def test_rigid_groups_split_at_zero_shared_points():
    m = two_islands(0)
    groups = G.rigid_groups(m, range(m.n), 3)
    assert [len(g) for g in groups] == [30, 20]
    # one camera pair sharing >= K points makes one rigid body
    m = two_islands(0, shared_between=5, dispersed=False)
    assert [len(g) for g in G.rigid_groups(m, range(m.n), 3)] == [50]
    # dispersed sharing (one point per camera pair) does not, at K = 3
    m = two_islands(0, shared_between=60)
    assert [len(g) for g in G.rigid_groups(m, range(m.n), 3)] == [30, 20]


def test_gate_splits_a_detached_island_whose_placement_moves_between_seeds():
    models = [two_islands(0), two_islands(1, b_offset=np.array([0, 4.0, 1.0]), b_rot=0.6),
              two_islands(2, b_offset=np.array([0, -3.0, 0]), b_rot=-0.4)]
    out = G.apply_rigid_gate(models, G.GateParams(k_shared=3, max_spread=0.05, min_obs=5))
    labels = np.array([out["labels"][n] for n in models[0].names])
    assert set(labels[:30]) == {0} and set(labels[30:]) == {1}
    assert out["splits"] == 1


def test_gate_keeps_a_strongly_shared_seed_stable_island():
    models = [two_islands(s, shared_between=60, noise=0.001) for s in range(3)]
    out = G.apply_rigid_gate(models, G.GateParams(k_shared=3, max_spread=0.05, min_obs=5))
    assert set(out["labels"].values()) == {0}
    assert out["splits"] == 0


def test_gate_splits_a_shared_but_seed_unstable_island():
    models = [two_islands(0, shared_between=60), two_islands(1, shared_between=60, b_offset=np.array([0, 5.0, 0]))]
    out = G.apply_rigid_gate(models, G.GateParams(k_shared=3, max_spread=0.05, min_obs=5))
    assert len(set(out["labels"].values())) == 2
    dec = [d for r in out["rounds"] for d in r["decisions"] if not d["kept"]]
    assert dec and "seed spread" in dec[0]["why"]


def test_gate_with_one_seed_uses_shared_points_only():
    out = G.apply_rigid_gate([two_islands(0, shared_between=60)], G.GateParams(k_shared=3, min_obs=5))
    assert set(out["labels"].values()) == {0}
    out = G.apply_rigid_gate([two_islands(0)], G.GateParams(k_shared=3, min_obs=5))
    assert len(set(out["labels"].values())) == 2


def test_seed_spread_is_zero_for_identical_seeds():
    models = [two_islands(0), two_islands(0)]
    models[1].seed = 1
    s = G.seed_spread(models)
    assert s["measured"] and s["per_seed"]["1"]["median"] < 1e-9


# ---------------------------------------------------------------------------
# export: keyframes only


def test_export_publishes_keyframes_only(tmp_path):
    from tower.world_builder.coherence_eval.eval_variant import load_variant
    from tower.world_builder.coherence_eval.eval_world import open_world

    world_dir, captures, frames, kfs = make_world(tmp_path, chain_sizes=(40, 40), keyframe_every=2)
    world = open_world(world_dir)
    m = two_islands(0, n_a=30, n_b=10)
    # name the model's images like staging: even = keyframes, odd = extras
    ci = D.capture_index(world_dir, captures)
    kf_seqs = [ci.keyframe_seq(k["keyframe_id"]) for k in kfs]
    rows = []
    names = []
    for i in range(m.n):
        if i % 2 == 0:
            seq = kf_seqs[i // 2]
            name = D.staged_name(seq, D.KEYFRAME_TAG)
            rows.append({"name": name, "kind": "keyframe", "keyframe_id": kfs[i // 2]["keyframe_id"],
                         "capture_seq": seq, "sha1": "x"})
        else:
            seq = kf_seqs[i // 2] + 1
            name = D.staged_name(seq, D.EXTRA_TAG)
            rows.append({"name": name, "kind": "extra", "keyframe_id": None, "capture_seq": seq, "sha1": "x"})
        names.append(name)
    m.names = names
    staging = {"images": rows, "camera": {"fx": 400.0, "fy": 400.0, "cx": 180.0, "cy": 320.0,
                                          "width": 359, "height": 639}}
    cfg = D.VariantConfig(min_image_observations=5)
    info = D.export_variant(m, None, staging, world, tmp_path / "v", cfg, name="t", meta={})
    v = load_variant(tmp_path / "v", world)
    assert len(v.poses) == 20 == info["keyframes_posed"]
    assert set(v.poses) <= set(world.keyframe_ids)
    assert v.obs_keyframe is not None and len(v.obs_keyframe)
    # every exported observation is a keyframe's, in its point's component
    comps = {k: v.component[k] for k in v.poses}
    for kf, pt in zip(v.obs_keyframe, v.obs_point):
        assert comps[v.obs_keyframe_ids[kf]] == v.point_component[pt]
    rep = D.component_report(m, None, staging, world, cfg)
    assert sum(r["keyframes"] for r in rep) == 20 and sum(r["extras"] for r in rep) == 20


def test_medoid_reference_is_the_seed_that_agrees_with_the_others():
    models = [two_islands(0, shared_between=60, dispersed=False), two_islands(1, shared_between=60, dispersed=False),
              two_islands(2, shared_between=60, dispersed=False, b_offset=np.array([0, 6.0, 0]))]
    models[2].names = list(models[2].names)
    ch = G.choose_reference(models, "medoid")
    assert ch["seed"] in (0, 1) and ch["rule"] == "medoid"
    # put the odd seed first: it must not be chosen
    ch = G.choose_reference([models[2], models[0], models[1]], "medoid")
    assert ch["seed"] != 2
    assert G.choose_reference(models, 2)["index"] == 2
