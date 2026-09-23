"""E2 bridge: gap detection, the live retention rule, staged names, and the
learned-keypoint append on a tiny synthetic COLMAP database (no world on disk)."""

import numpy as np
import pandas as pd
import pytest

from tower.world_builder.coherence_exp import bridge as B


def _labels(outcomes):
    reasons = {"accept": "parallax", "skip": "insufficient_motion", "reject": "blurred",
               "tracking_lost": "tracking_lost", "bad": "malformed_frame"}
    rows = []
    for i, o in enumerate(outcomes):
        rows.append({"frame_index": i, "outcome": "reject" if o == "bad" else o, "reason": reasons[o]})
    return pd.DataFrame(rows)


def test_loss_gaps_run_from_last_keyframe_to_next_and_merge_repeated_losses():
    #        0       1       2        3              4        5       6       7
    seq = ["accept", "skip", "reject", "tracking_lost", "reject", "accept", "skip", "accept",
           #  8        9              10             11     12       13
           "reject", "tracking_lost", "tracking_lost", "bad", "reject", "accept",
           #  14       15
           "reject", "tracking_lost"]
    gaps = B.loss_gaps(_labels(seq))
    assert [(g.start_frame, g.end_frame) for g in gaps] == [(0, 5), (7, 13), (13, None)]
    assert gaps[0].refused == (1, 2, 3, 4)
    assert gaps[0].lost_frames == (3,)
    assert gaps[0].preloss_refused == 2
    # two losses before any accept are ONE gap; the undecodable frame is never retained
    assert gaps[1].lost_frames == (9, 10)
    assert gaps[1].refused == (8, 9, 10, 12)
    # a run with refusals but no loss (frame 6) is not a gap
    assert all(6 not in g.refused for g in gaps)
    # the walk ended inside a gap
    assert gaps[2].end_frame is None and gaps[2].refused == (14, 15)


@pytest.mark.parametrize("policy", B.POLICIES)
def test_live_retainer_equals_the_batch_rule(policy):
    rng = np.random.default_rng(7)
    for trial in range(30):
        n = int(rng.integers(5, 300))
        seq = ["accept"] + list(rng.choice(["accept", "skip", "reject", "tracking_lost", "bad"], size=n - 1,
                                           p=[0.15, 0.3, 0.4, 0.1, 0.05]))
        labels = _labels(seq)
        for k in (1, 2, 3):
            live = B.GapFrameRetainer(policy, k).run(labels)
            assert live == B.select_frames(labels, policy, k), (trial, k)


def test_every_k_keeps_positions_counted_from_the_keyframe():
    seq = ["accept", "reject", "reject", "reject", "tracking_lost", "reject", "reject", "accept"]
    labels = _labels(seq)
    assert B.select_frames(labels, B.POLICY_LOSS_GAPS_ALL) == [1, 2, 3, 4, 5, 6]
    assert B.select_frames(labels, B.POLICY_LOSS_GAPS_EVERY_K, 2) == [1, 3, 5]


def test_live_retainer_commits_nothing_before_the_loss_and_respects_its_buffer():
    r = B.GapFrameRetainer(B.POLICY_LOSS_GAPS_ALL, max_buffer=2)
    assert r.observe(0, "accept") == []
    assert r.observe(1, "reject", "blurred") == []
    assert r.observe(2, "reject", "blurred") == []
    assert r.observe(3, "reject", "blurred") == []
    # the loss commits the buffered pre-loss frames (only the newest 2 fit) and itself
    assert r.observe(4, "tracking_lost", "tracking_lost") == [2, 3, 4]
    assert r.observe(5, "reject", "blurred") == [5]
    assert r.observe(6, "accept") == []
    # a refused run that never loses is dropped at the next keyframe
    assert r.observe(7, "reject", "blurred") == []
    assert r.observe(8, "accept") == []
    assert r.peak_buffer == 2


def test_staged_names_sort_in_capture_order_and_round_trip():
    from tower.world_builder.coherence_exp import driver

    seqs = [3, 12, 101, 1000, 99999]
    names = [B.staged_name(s, B.KIND_KEYFRAME if s % 2 else B.KIND_GAP) for s in seqs]
    assert sorted(names) == names
    assert [B.parse_staged_name(n) for n in names] == [(s, "k" if s % 2 else "g") for s in seqs]
    assert B.staged_name(584, "k") == driver.staged_name(584, "k") == "000584_k.jpg"
    assert B.parse_staged_name("00000001.jpg") is None
    with pytest.raises(ValueError):
        B.staged_name(1, "x")
    with pytest.raises(ValueError):
        B.staged_name(10 ** 6, "g")


def test_chains_hold_context_keyframes_and_the_gap_frames_between():
    seq = ["accept", "accept", "accept", "reject", "tracking_lost", "reject", "accept", "accept", "accept"]
    labels = _labels(seq)
    gaps = B.loss_gaps(labels)
    kept = B.select_frames(labels, B.POLICY_LOSS_GAPS_ALL)
    kf = [0, 1, 2, 6, 7, 8]
    chains = B.build_chains(gaps, kept, kf, {i: 0.1 * i for i in range(9)}, context=2)
    assert len(chains) == 1
    c = chains[0]
    assert c.names == ("000000_k.jpg", "000001_k.jpg", "000002_k.jpg", "000003_g.jpg", "000004_g.jpg",
                       "000005_g.jpg", "000006_k.jpg", "000007_k.jpg", "000008_k.jpg")
    assert (c.names[c.kf_before], c.names[c.kf_after]) == ("000002_k.jpg", "000006_k.jpg")
    assert (0, 1) in c.pairs(2) and (0, 2) in c.pairs(2) and (0, 3) not in c.pairs(2)


def _rot(axis, deg):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    a = np.radians(deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def test_consensus_keeps_a_link_that_closes_its_triangles_and_refuses_one_that_does_not():
    Rw = [_rot([0, 1, 0], 4.0 * i) for i in range(4)]          # camera i from world
    rel = lambda i, j: Rw[j] @ Rw[i].T                           # noqa: E731  j from i
    rot = {(0, 1): rel(0, 1), (1, 2): rel(1, 2), (2, 3): rel(2, 3),
           (0, 2): rel(0, 2),                                    # learned, true
           (1, 3): _rot([1, 0, 0], 20.0) @ rel(1, 3)}            # learned, hallucinated
    dt = {k: 0.08 * (k[1] - k[0]) for k in rot}
    v = B.consensus(4, rot, dt, {(0, 2), (1, 3)}, cycle_bound_deg=2.0)
    assert v[(0, 2)]["keep"] and v[(0, 2)]["cycle_median_deg"] < 1e-6
    assert not v[(1, 3)]["keep"] and abs(v[(1, 3)]["cycle_median_deg"] - 20.0) < 1e-6
    # a head cannot turn 90 deg in 80 ms (300 deg/s * dt + 30 deg = 54 deg)
    rot2 = {(0, 1): _rot([0, 1, 0], 90.0)}
    v2 = B.consensus(2, rot2, {(0, 1): 0.08}, {(0, 1)}, cycle_bound_deg=2.0)
    assert not v2[(0, 1)]["keep"] and "physical" in v2[(0, 1)]["why"]
    # and a learned link nothing can check is not kept
    v3 = B.consensus(2, {(0, 1): rel(0, 1)}, {(0, 1): 0.08}, {(0, 1)}, cycle_bound_deg=2.0)
    assert not v3[(0, 1)]["keep"] and v3[(0, 1)]["why"] == "no closing triangle"


def test_quantiser_merges_one_scene_point_seen_in_two_pairs():
    q = B.KeypointQuantiser(2.0)
    q.set_base(7, 100)
    a = q.indices(7, np.array([[10.2, 20.1], [50.0, 50.0]]))
    b = q.indices(7, np.array([[9.7, 19.8], [80.0, 12.0]]))   # same cell as the first point
    assert list(a) == [100, 101] and list(b) == [100, 102]
    rows = q.new_keypoints(7)
    assert rows.shape == (3, 6)
    assert np.allclose(rows[0, :2], [10.5, 20.5])              # COLMAP convention: +0.5
    assert np.allclose(rows[:, 2], 1) and np.allclose(rows[:, 5], 1) and np.allclose(rows[:, 3:5], 0)


def _synthetic_database(path, n_sift=40, seed=0):
    """3 images of a random 3-D scene; SIFT-like keypoints that verify nothing."""
    import pycolmap

    rng = np.random.default_rng(seed)
    f, cx, cy, W, H = 500.0, 180.0, 320.0, 360, 640
    db = pycolmap.Database.open(str(path))
    cid = db.write_camera(pycolmap.Camera(model="PINHOLE", width=W, height=H, params=[f, f, cx, cy]))
    names = ["000010_k.jpg", "000011_g.jpg", "000012_g.jpg"]
    ids = {}
    for n in names:
        iid = db.write_image(pycolmap.Image(name=n, camera_id=cid))
        ids[n] = iid
        kp = np.zeros((n_sift, 6), np.float32)
        kp[:, 0] = rng.uniform(0, W, n_sift)
        kp[:, 1] = rng.uniform(0, H, n_sift)
        kp[:, 2] = kp[:, 5] = 1.0
        db.write_keypoints(iid, kp)
        db.write_descriptors(iid, pycolmap.FeatureDescriptors(pycolmap.FeatureExtractorType.SIFT,
                                                              rng.integers(0, 255, (n_sift, 128), dtype=np.uint8)))
    # a handful of SIFT matches that cannot verify (fewer than 15)
    db.write_matches(ids[names[0]], ids[names[1]], np.array([[0, 0], [1, 1], [2, 2]], np.uint32))
    db.close()
    # the scene and the (learned) correspondences, in OpenCV pixel convention
    X = np.column_stack([rng.uniform(-2, 2, 300), rng.uniform(-3, 3, 300), rng.uniform(4, 8, 300)])
    poses = [(np.eye(3), np.zeros(3)), (_rot([0, 1, 0], 3.0), np.array([-0.15, 0, 0])),
             (_rot([0, 1, 0], 6.0), np.array([-0.3, 0.02, 0]))]

    def project(i):
        R, t = poses[i]
        c = (R @ X.T).T + t
        return np.column_stack([f * c[:, 0] / c[:, 2] + cx, f * c[:, 1] / c[:, 2] + cy]) - 0.5

    uv = [project(i) for i in range(3)]
    ok = np.all([(u[:, 0] > 0) & (u[:, 0] < W - 1) & (u[:, 1] > 0) & (u[:, 1] < H - 1) for u in uv], axis=0)
    uv = [u[ok] for u in uv]
    eloftr = {f"{names[i]}|{names[j]}": (uv[i], uv[j], np.ones(len(uv[i]))) for i, j in [(0, 1), (1, 2), (0, 2)]}
    chain = B.Chain(0, tuple(names), 0, 2, (10, 11, 12), (0.0, 0.08, 0.16))
    return ids, names, chain, eloftr, n_sift


def test_learned_keypoints_are_appended_after_sift_with_offset_matches(tmp_path):
    import pycolmap

    db_path = tmp_path / "database.db"
    ids, names, chain, eloftr, n_sift = _synthetic_database(db_path)
    before = pycolmap.Database.open(str(db_path))
    sift_rows = {n: before.read_keypoints(ids[n]).copy() for n in names}
    before.close()

    rep = B.bridge_augment(db_path, ids, [chain], eloftr, cycle_bound_deg=2.0, quant_px=2.0)
    assert rep["learned_links_verified"] == 3 and rep["learned_links_kept"] == 3
    assert rep["chains"][0]["bridging"]["bridged_by"] == "eloftr"

    db = pycolmap.Database.open(str(db_path))
    try:
        for n in names:
            kp = db.read_keypoints(ids[n])
            assert len(kp) > n_sift
            assert np.array_equal(kp[:n_sift], sift_rows[n])                # SIFT rows untouched, first
            assert np.allclose(np.mod(kp[n_sift:, :2] - 0.5, 2.0), 0.0)      # quantised, COLMAP +0.5
            assert db.num_descriptors_for_image(ids[n]) == len(kp)          # padded to match
        a, b, c = (ids[n] for n in names)
        m01 = np.asarray(db.read_matches(a, b))
        assert m01.shape[0] > 3 and np.array_equal(m01[:3], [[0, 0], [1, 1], [2, 2]])  # SIFT matches kept
        assert (m01[3:] >= n_sift).all()                                    # learned ones offset
        g01 = db.read_two_view_geometry(a, b)
        g12 = db.read_two_view_geometry(b, c)
        assert len(g01.inlier_matches) >= B.MIN_VERIFIED_INLIERS
        assert int(g01.config.value if hasattr(g01.config, "value") else g01.config) not in B.UNVERIFIED_CONFIGS
        # the middle image's learned keypoints are SHARED by both of its pairs: tracks of 3 views
        shared = set(np.asarray(g01.inlier_matches)[:, 1]) & set(np.asarray(g12.inlier_matches)[:, 0])
        assert len(shared) > 0.8 * min(len(g01.inlier_matches), len(g12.inlier_matches))
    finally:
        db.close()


def test_sift_only_arm_classifies_and_writes_nothing(tmp_path):
    import pycolmap

    db_path = tmp_path / "database.db"
    ids, names, chain, _eloftr, n_sift = _synthetic_database(db_path)
    rep = B.bridge_augment(db_path, ids, [chain], None, cycle_bound_deg=2.0)
    assert rep["writes"] == 0 and rep["chains"][0]["bridging"]["bridged_by"] is None
    db = pycolmap.Database.open(str(db_path))
    try:
        assert all(len(db.read_keypoints(ids[n])) == n_sift for n in names)
    finally:
        db.close()


def test_cut_strength_sees_a_bridge_through_gap_frames():
    order = ["k0", "k1", "g1", "g2", "k2", "k3"]
    kf = ["k0", "k1", "k2", "k3"]
    base = [("k0", "k1"), ("k2", "k3"), ("k0", "k1")]
    assert B.cut_strengths(order, base, kf, window=10) == [2, 0, 1]
    bridged = base + [("k1", "g1"), ("g1", "g2"), ("g2", "k2")]
    assert B.cut_strengths(order, bridged, kf, window=10) == [2, 1, 1]


def test_track_cross_section_is_local_so_a_loop_track_does_not_bridge_a_cut():
    order = ["k0", "k1", "k2", "k3", "k4", "k5"]
    names = ["k0", "k1", "k2", "k5"]                  # registered images
    obs_image = [0, 3, 1, 2]                            # point 0: k0 + k5 (a loop); point 1: k1 + k2
    obs_point = [0, 0, 1, 1]
    out = B.model_cut_tracks_from_arrays(names, obs_image, obs_point, order, order, window=2)
    assert [o["tracks_min_cross_section"] for o in out] == [0, 1, 0, 0, 0]
    assert [o["keyframe_tracks_window"] for o in out] == [0, 1, 0, 0, 0]


def test_gap_link_consensus_keeps_a_closing_triangle_and_refuses_an_uncorroborated_pair(tmp_path):
    import pycolmap

    db_path = tmp_path / "database.db"
    ids, names, chain, eloftr, _ = _synthetic_database(db_path)
    B.bridge_augment(db_path, ids, [chain], eloftr, cycle_bound_deg=2.0)
    t = {n: 0.08 * i for i, n in enumerate(names)}
    is_gap = lambda n: n.endswith("_g.jpg")                     # noqa: E731
    rep = B.gap_link_consensus(db_path, is_gap, t, cycle_bound_deg=2.0, dry_run=True)
    assert rep["gap_pairs"] == 3 and rep["refused"] == 0
    # without the (1, 2) link nothing can corroborate the other two: both are refused and removed
    db = pycolmap.Database.open(str(db_path))
    db.delete_two_view_geometry(ids[names[1]], ids[names[2]])
    db.close()
    rep = B.gap_link_consensus(db_path, is_gap, t, cycle_bound_deg=2.0)
    assert rep["gap_pairs"] == 2 and rep["refused"] == 2
    assert rep["refused_by_reason"] == {"no closing triangle": 2}
    db = pycolmap.Database.open(str(db_path))
    try:
        assert not db.exists_two_view_geometry(ids[names[0]], ids[names[1]])
        assert db.exists_matches(ids[names[0]], ids[names[1]])      # raw matches stay
    finally:
        db.close()


def test_restore_keyframe_graph_transplants_the_reference_pairs_and_keeps_gap_pairs(tmp_path):
    import shutil

    import pycolmap

    ref_path, arm_path = tmp_path / "ref.db", tmp_path / "arm.db"
    ids, names, chain, eloftr, _ = _synthetic_database(ref_path)
    kf2 = "000013_k.jpg"
    db = pycolmap.Database.open(str(ref_path))
    kid2 = db.write_image(pycolmap.Image(name=kf2, camera_id=1))
    db.write_keypoints(kid2, db.read_keypoints(ids[names[0]]))
    db.close()
    shutil.copyfile(ref_path, arm_path)
    # the reference holds a keyframe pair the arm lost; the arm holds one the reference never had
    ref = pycolmap.Database.open(str(ref_path))
    ref.write_matches(ids[names[0]], kid2, np.array([[5, 6], [7, 8]], np.uint32))
    ref.close()
    arm = pycolmap.Database.open(str(arm_path))
    arm.write_matches(ids[names[1]], ids[names[2]], np.array([[1, 1]], np.uint32))   # a gap-gap pair: must stay
    arm.close()
    B.bridge_augment(arm_path, ids, [chain], eloftr, cycle_bound_deg=2.0)          # learned gap links written
    rep = B.restore_keyframe_graph(arm_path, ref_path, scratch_dir=tmp_path)
    assert rep["keyframes"] == 2 and rep["added"]["matches"] == 1
    arm = pycolmap.Database.open(str(arm_path))
    try:
        assert np.array_equal(arm.read_matches(ids[names[0]], kid2), [[5, 6], [7, 8]])
        assert arm.exists_two_view_geometry(ids[names[0]], ids[names[1]])          # keyframe-gap pair untouched
    finally:
        arm.close()


def test_eloftr_keypoints_stay_sub_pixel():
    kps = np.array([[[0.5013, 0.25], [0.1, 0.9]], [[0.5031, 0.2507], [0.2, 0.8]]])   # (2, N, 2) normalised
    matches = np.array([[0, -1], [0, -1]])
    scores = np.array([[0.9, 0.9], [0.9, 0.9]])
    k0, k1, sc = B.matched_keypoints_float(kps, matches, scores, (639, 359), (639, 359), threshold=0.2)
    assert k0.shape == (1, 2) and k1.shape == (1, 2) and sc.tolist() == [pytest.approx(0.9)]
    assert np.allclose(k0[0], [0.5013 * 359, 0.25 * 639])        # 179.97, not truncated to 179
    assert np.allclose(k1[0], [0.5031 * 359, 0.2507 * 639])


def test_quantiser_absorbs_integer_truncation():
    rng = np.random.default_rng(3)
    xy = rng.uniform(0, 640, size=(5000, 2))
    qa, qb = B.KeypointQuantiser(2.0), B.KeypointQuantiser(2.0)
    qa.set_base(1, 0)
    qb.set_base(1, 0)
    assert np.array_equal(qa.indices(1, xy), qb.indices(1, np.floor(xy)))
    assert np.array_equal(qa.new_keypoints(1), qb.new_keypoints(1))


def test_restrict_gap_links_keeps_chain_pairs_and_drops_the_rest(tmp_path):
    import pycolmap

    db_path = tmp_path / "database.db"
    ids, names, chain, eloftr, _ = _synthetic_database(db_path)
    B.bridge_augment(db_path, ids, [chain], eloftr, cycle_bound_deg=2.0)
    db = pycolmap.Database.open(str(db_path))
    far = db.write_image(pycolmap.Image(name="000900_k.jpg", camera_id=1))
    db.write_keypoints(far, db.read_keypoints(ids[names[0]]))
    g = db.read_two_view_geometry(ids[names[0]], ids[names[1]])
    db.write_two_view_geometry(far, ids[names[1]], g)                 # a far keyframe <-> gap frame pair
    db.close()
    import shutil
    shutil.copyfile(db_path, tmp_path / "copy.db")
    assert B.restrict_gap_links(db_path, [chain], "chain_only") == {"mode": "chain_only", "removed": 1}
    assert B.restrict_gap_links(tmp_path / "copy.db", [chain], "none")["removed"] == 4
    db = pycolmap.Database.open(str(db_path))
    try:
        assert db.exists_two_view_geometry(ids[names[0]], ids[names[1]])
        assert not db.exists_two_view_geometry(far, ids[names[1]])
    finally:
        db.close()
