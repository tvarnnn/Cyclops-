"""The anchor verification (world_builder/anchor_verify.py; RUN P4-IV RULE.md, digest 8efd726d8a1548d8) on
synthetic solves.

Pinned here:
  (a) the product's `scale_split` alone cannot see a middle stretch shorter than both of its flanks; cut into capture
      runs, (a) -- RULE-a2.md's median-CI test -- isolates it and seals it `scale-mismatch`; a noisy but on-level
      stretch is NOT sealed; the no-majority GUARD; an untestable short stretch; `median_ci` and `anchor_scale_ci`
      equal P4-IV's `iv_rule.py` bit for bit (golden/world_builder_anchor_a2_iv_rule.json);
  (c) the motion flag's bounds (2.0 m/s * dt + 0.3 m, 300 deg/s * dt + 30 deg, dt in [0.05, 2] s), in metres of the
      gate's level, and that a flag only adds a cut;
  (b) the verification groups (cuts, windows, merges); direct, two-hop and unverifiable evidence, the 3/3/3
      independence floor; peeling;
  end to end through `gate_and_publish` with the switch on: the seal re-gate, its post-checks, collateral, the record.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tests import wb_anchor_verify_fixtures as F
from tower.world_builder import anchor_verify as AV
from tower.world_builder import coherence_gate as CG
from tower.world_builder import coherence_publish as CP

GP = CG.GateParams()
VP = AV.VerifyParams()
L2 = math.log(2.0)


def _levels(n, spans):
    """Per-camera level: 0 except `spans` [(lo, hi, value)], plus a little per-camera noise."""
    r = np.array([0.001 * (i % 3) for i in range(n)], float)
    for lo, hi, v in spans:
        r[lo:hi] += v
    return r


def _world(n, dt=0.5, segments=None, t=None):
    times = t if t is not None else [1000.0 + dt * i for i in range(n)]
    return AV.World(times, segments if segments is not None else [0] * n)


def _pairs(n, pairs, R=None, C=None, error_deg=None, **kw):
    if R is None:
        R = {i: F.rz(3.0 * i) for i in range(n)}
        C = {i: c for i, c in enumerate(F.line_centres(n))}
    return AV.Pairs(F.synthetic_pairs(R, C, pairs, error_deg=error_deg, **kw), n)


# ---------------------------------------------------------------------------------------------------------------
# (a) the anchor block's own scale segmentation


def test_scale_split_alone_cannot_see_a_middle_stretch_shorter_than_both_flanks():
    r = _levels(60, [(25, 35, L2)])
    assert len(CG.scale_split(np.arange(60), r, np.arange(60), GP)) == 1        # RULE.md section 3.1


def test_a_isolates_a_middle_stretch_cut_into_capture_runs_and_seals_it():
    n = 60
    r = _levels(n, [(25, 35, L2)])
    w = _world(n)
    pairs = _pairs(n, F.chain_pairs(n, breaks=(25,)))            # the images do not chain across 24 | 25
    runs = AV.capture_runs(list(range(n)), {i: "anchor" for i in range(n)}, set(range(n)), w, pairs, VP)
    assert [(r_[0], r_[-1]) for r_ in runs] == [(0, 24), (25, 59)]
    a = AV.anchor_scale_ci(runs, r, np.arange(n), GP)
    assert not a["refused_no_majority"]
    assert [d["cameras"] for d in a["detached"]] == [list(range(25, 35))]
    assert a["detached"][0]["factor"] == pytest.approx(2.0, rel=0.01)
    assert a["detached"][0]["ci_gap"] > math.log(1.25)


def test_a_seals_nothing_in_one_uncut_run():
    n = 60
    r = _levels(n, [(25, 35, L2)])
    a = AV.anchor_scale_ci([list(range(n))], r, np.arange(n), GP)
    assert a["detached"] == [] and not a["refused_no_majority"]


def test_the_guard_refuses_when_no_level_holds_a_majority():
    n = 40
    r = _levels(n, [(20, 40, L2)])
    a = AV.anchor_scale_ci([list(range(20)), list(range(20, 40))], r, np.arange(n), GP)
    assert a["refused_no_majority"] and a["detached"] == [] and a["would_detach"]


def test_a_segment_under_ten_ratios_is_untestable_and_left_alone():
    n = 60
    r = _levels(n, [(25, 33, L2)])                                # 8 cameras: under scale_min_cameras
    a = AV.anchor_scale_ci([list(range(25)), list(range(25, 33)), list(range(33, 60))], r, np.arange(n), GP)
    assert a["detached"] == [] and len(a["untestable"]) == 1


def test_a_noisy_but_on_level_stretch_is_not_sealed_where_the_declared_test_would_seal_it():
    # the control's failure (P4-IV PHASE2.md): 10 scattered levels around the room's, a run of their own
    n = 70
    r = 1.72 + 0.001 * (np.arange(n) % 5)
    r[30:40] = [0.29, 3.84, 0.5, 3.5, 0.8, 3.0, 2.9, 3.2, 3.6, 3.3]
    assert CG.scale_levels_differ(r, np.r_[0:30, 40:70], np.arange(30, 40), GP)        # RULE.md's (a) would
    a = AV.anchor_scale_ci([list(range(30)), list(range(30, 40)), list(range(40, 70))], r, np.arange(n), GP)
    assert a["detached"] == [] and not a["refused_no_majority"]
    assert a["tested"][1]["n"] == 10 and a["tested"][1]["gap"] < 0                     # tested, not differing


def test_a_x3_middle_stretch_shorter_than_both_neighbours_is_sealed():
    n = 70
    r = 1.0 + 0.02 * np.sin(np.arange(n))
    r[30:42] += math.log(3.0)
    a = AV.anchor_scale_ci([list(range(30)), list(range(30, 70))], r, np.arange(n), GP)
    assert [d["cameras"] for d in a["detached"]] == [list(range(30, 42))]
    assert a["detached"][0]["factor"] == pytest.approx(3.0, rel=0.05)


def test_median_ci_is_the_sign_test_interval():
    assert AV.median_ci([], 0.05) is None and AV.median_ci([1.0] * 5, 0.05) is None    # n = 5: no k reaches 95 %
    x = np.arange(10, dtype=float)[::-1]
    assert AV.median_ci(x, 0.05) == (1.0, 8.0)                                         # [x_(2), x_(9)], RULE-a2 2.4
    assert AV.median_ci(np.arange(20, dtype=float), 0.05) == (5.0, 14.0)               # [x_(6), x_(15)]


def _a2_golden():
    return json.loads((Path(__file__).parent / "golden" / "world_builder_anchor_a2_iv_rule.json")
                      .read_text(encoding="utf-8"))


def test_median_ci_equals_p4s_reference_bit_for_bit():
    cases = _a2_golden()["median_ci"]
    assert len(cases) == 34
    for c in cases:
        got = AV.median_ci(c["values"], c["alpha"])
        assert (None if got is None else list(got)) == c["ci"], (len(c["values"]), c["alpha"])


@pytest.mark.parametrize("name", ["x3_middle_stretch", "noisy_on_level", "no_majority", "noise_with_gaps"])
def test_anchor_scale_ci_equals_p4s_reference_bit_for_bit(name):
    case = _a2_golden()["anchor_scale_ci"][name]
    r = np.asarray([np.nan if v is None else v for v in case["r"]], np.float64)
    got = AV.anchor_scale_ci(case["runs"], r, np.arange(len(r)), GP, 0.05)
    assert json.loads(json.dumps(got, sort_keys=True)) == case["out"]


def test_the_a2_digest_is_the_declared_one():
    assert AV.a2_digest() == "8f5d494796345cd7"


# ---------------------------------------------------------------------------------------------------------------
# (c) the physical-motion flag


def _moving(n, dt, step_m, turn_deg=0.0):
    w = _world(n, dt=dt)
    R = {i: F.rz(turn_deg * i) for i in range(n)}
    C = {i: np.array([step_m * i, 0.0, 0.0]) for i in range(n)}
    return w, AV.Solve(R, C)


@pytest.mark.parametrize("step, flagged", [(1.29, False), (1.31, True)])
def test_the_walking_bound_is_two_metres_a_second_plus_thirty_centimetres(step, flagged):
    w, s = _moving(3, 0.5, step)
    assert bool(AV.motion_flags([0, 1, 2], w, s, 1.0)) == flagged              # 2.0 * 0.5 + 0.3 = 1.3 m


@pytest.mark.parametrize("turn, flagged", [(59.0, False), (61.0, True)])
def test_the_head_turn_bound_is_three_hundred_degrees_a_second_plus_thirty(turn, flagged):
    w, s = _moving(3, 0.1, 0.0, turn)
    assert bool(AV.motion_flags([0, 1, 2], w, s, 1.0)) == flagged              # 300 * 0.1 + 30 = 60 deg


def test_the_bound_is_in_metres_of_the_gates_level_and_long_gaps_are_not_judged():
    w, s = _moving(3, 0.5, 0.7)
    assert not AV.motion_flags([0, 1, 2], w, s, 1.0)                            # 0.7 m
    assert AV.motion_flags([0, 1, 2], w, s, 2.0)                                # 1.4 m
    w, s = _moving(3, 2.5, 100.0)
    assert not AV.motion_flags([0, 1, 2], w, s, 1.0)                            # dt > 2 s: no bound applies


def test_dt_is_floored_at_fifty_milliseconds():
    w = _world(2, t=[1000.0, 1000.0])
    s = AV.Solve({0: np.eye(3), 1: np.eye(3)}, {0: np.zeros(3), 1: np.array([0.39, 0.0, 0.0])})
    assert not AV.motion_flags([0, 1], w, s, 1.0)                                # 2.0 * 0.05 + 0.3 = 0.4 m
    s.C[1] = np.array([0.41, 0.0, 0.0])
    assert AV.motion_flags([0, 1], w, s, 1.0)


def test_a_flag_only_adds_a_cut():
    n = 20
    w = _world(n)
    pairs = _pairs(n, F.chain_pairs(n))
    gg = {i: "g" for i in range(n)}
    assert len(AV.capture_runs(range(n), gg, set(range(n)), w, pairs, VP)) == 1
    runs = AV.capture_runs(range(n), gg, set(range(n)), w, pairs, VP, extra_cuts=[(9, 10)])
    assert [(r[0], r[-1]) for r in runs] == [(0, 9), (10, 19)]


# ---------------------------------------------------------------------------------------------------------------
# (b) groups


def test_capture_runs_cut_on_group_pause_outside_camera_and_chain_break():
    n = 40
    t = [1000.0 + 0.5 * i + (5.0 if i >= 30 else 0.0) for i in range(n)]    # a 5.5 s pause between 29 and 30
    w = _world(n, t=t)
    pairs = _pairs(n, F.chain_pairs(n, breaks=(20,)))
    gg = {i: ("a" if i < 10 else "b") for i in range(n)}
    kept = [i for i in range(n) if i != 25]                                  # 25 is published, not kept
    runs = AV.capture_runs(kept, gg, set(range(n)), w, pairs, VP)
    assert [(r[0], r[-1]) for r in runs] == [(0, 9), (10, 19), (20, 24), (26, 29), (30, 39)]


def test_long_runs_are_windowed_and_small_runs_merge_into_the_nearest_of_their_gate_group():
    n = 75
    w = _world(n)
    pairs = _pairs(n, F.chain_pairs(n, breaks=(70,)))
    gg = {i: "g" for i in range(n)}
    groups = AV.verification_groups(range(n), gg, set(range(n)), w, pairs, VP)
    # 0..69 -> ceil(70 / 30) = 3 near-equal windows; 70..74 (5) stands alone at min_group
    assert [(int(g[0]), int(g[-1])) for g in groups] == [(0, 23), (24, 46), (47, 69), (70, 74)]
    pairs = _pairs(n, F.chain_pairs(n, breaks=(72,)))                        # 72..74: 3 < 5, merges back
    groups = AV.verification_groups(range(n), gg, set(range(n)), w, pairs, VP)
    assert [(int(g[0]), int(g[-1])) for g in groups] == [(0, 23), (24, 47), (48, 74)]


# ---------------------------------------------------------------------------------------------------------------
# (b) evidence and decisions


def _revisit_world(n=100):
    """Segment 0 = 0..49, segment 1 = 50..99; 0.5 s apart, so a capture gap > 30 is >= 15.5 s."""
    return _world(n, segments=[0] * 50 + [1] * 50)


def _solve(n):
    return AV.Solve({i: F.rz(3.0 * i) for i in range(n)}, {i: c for i, c in enumerate(F.line_centres(n))})


def test_direct_pairs_contradict_above_the_rotation_bound_and_confirm_below():
    n = 100
    w, s = _revisit_world(n), _solve(n)
    ends = [(60, 5), (62, 8), (64, 12), (66, 15)]
    G, P = set(range(60, 70)), set(range(0, 50))
    bad = _pairs(n, ends, s.R, s.C, error_deg={(min(a, b), max(a, b)): 8.0 for a, b in ends})
    v = AV.judge(G, P, w, bad, s, VP)
    assert (v["evidence"], v["verdict"], v["why"]) == ("direct", AV.VERDICT_CONTRADICTED, "rotation")
    good = _pairs(n, ends, s.R, s.C, error_deg={(min(a, b), max(a, b)): 3.0 for a, b in ends})
    assert AV.judge(G, P, w, good, s, VP)["verdict"] == AV.VERDICT_CONFIRMED


def test_two_images_are_not_independent_evidence():
    n = 100
    w, s = _revisit_world(n), _solve(n)
    ends = [(60, 5), (60, 8), (61, 12), (61, 15)]                            # 4 pairs, 2 group cameras
    pairs = _pairs(n, ends, s.R, s.C, error_deg={(min(a, b), max(a, b)): 15.0 for a, b in ends})
    v = AV.judge(set(range(60, 70)), set(range(50)), w, pairs, s, VP)
    assert v["evidence"] is None and v["verdict"] == AV.VERDICT_UNVERIFIABLE


def test_only_revisit_pairs_are_evidence():
    n = 100
    w, s = _revisit_world(n), _solve(n)
    ends = [(60, 45), (62, 44), (64, 43)]                                    # gap <= 30: not a revisit
    pairs = _pairs(n, ends, s.R, s.C, error_deg={(min(a, b), max(a, b)): 20.0 for a, b in ends})
    assert AV.judge(set(range(60, 70)), set(range(50)), w, pairs, s, VP)["verdict"] == AV.VERDICT_UNVERIFIABLE


def test_two_hop_chains_decide_where_direct_pairs_are_thin():
    n = 100
    w, s = _revisit_world(n), _solve(n)
    # a in G -- m (adjacent, not revisit) -- r in P (a revisit leg); no direct G-P pair
    legs = [(60, 58), (62, 58), (64, 57), (58, 5), (58, 8), (57, 12), (57, 15), (58, 20)]
    for err, verdict in ((30.0, AV.VERDICT_CONTRADICTED), (5.0, AV.VERDICT_CONFIRMED)):
        pairs = _pairs(n, legs, s.R, s.C, error_deg={(5, 58): err, (8, 58): err, (12, 57): err, (15, 57): err,
                                                      (20, 58): err})
        v = AV.judge(set(range(60, 70)), set(range(50)), w, pairs, s, VP)
        assert (v["evidence"], v["verdict"]) == ("two-hop", verdict)
        assert v["two_hop_ends"] >= 3 and v["two_hop_legs"] >= 3


def test_no_revisit_evidence_is_unverifiable():
    n = 100
    w, s = _revisit_world(n), _solve(n)
    pairs = _pairs(n, F.chain_pairs(n), s.R, s.C)
    assert AV.judge(set(range(60, 70)), set(range(50)), w, pairs, s, VP)["verdict"] == AV.VERDICT_UNVERIFIABLE


def test_translation_direction_alone_can_contradict():
    n = 100
    w, s = _revisit_world(n), _solve(n)
    ends = [(60, 5), (62, 8), (64, 12), (66, 15)]
    arrays = F.synthetic_pairs(s.R, s.C, ends)
    arrays["t"] = -arrays["t"]                                               # 180 deg off, rotations exact
    v = AV.judge(set(range(60, 70)), set(range(50)), w, AV.Pairs(arrays, n), s, VP)
    assert (v["verdict"], v["why"]) == (AV.VERDICT_CONTRADICTED, "translation")


def test_peeling_seals_one_of_two_groups_that_only_contradict_each_other():
    n = 100
    w, s = _revisit_world(n), _solve(n)
    ends = [(0, 60), (2, 62), (4, 64), (6, 66)]
    pairs = _pairs(n, ends, s.R, s.C, error_deg={e: 10.0 for e in ends})
    groups = [np.arange(0, 10), np.arange(60, 70)]
    res = AV.verify(groups, w, pairs, s, VP)
    assert res["sealed"] == [0]                                              # a tie: the earliest first camera
    assert res["groups"][1]["verdict"] == AV.VERDICT_UNVERIFIABLE            # re-judged without it


def test_peeling_seals_the_worse_group_and_clears_its_partners():
    n = 100
    w, s = _revisit_world(n), _solve(n)
    bad = [(0, 60), (2, 62), (4, 64), (20, 66), (22, 68), (24, 69)]
    good = [(0, 80), (2, 82), (4, 84), (6, 86), (20, 88), (22, 90), (24, 92), (26, 94)]
    err = {e: 12.0 for e in bad} | {e: 1.0 for e in good}
    pairs = _pairs(n, bad + good, s.R, s.C, error_deg=err)
    groups = [np.arange(0, 10), np.arange(20, 30), np.arange(60, 70), np.arange(80, 100)]
    res = AV.verify(groups, w, pairs, s, VP)
    assert res["sealed"] == [2]
    assert [res["groups"][g]["verdict"] for g in (0, 1, 3)] == [AV.VERDICT_CONFIRMED] * 3


# ---------------------------------------------------------------------------------------------------------------
# the masked tier


def _texture(seed, w=320, h=240):
    import cv2

    rng = np.random.default_rng(seed)
    im = (rng.random((h // 8, w // 8)) * 255).astype(np.uint8)
    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_NEAREST)
    return cv2.GaussianBlur(im, (3, 3), 0)


def test_keypoints_under_the_mask_are_dropped_before_matching_and_no_mask_is_no_keypoint():
    im = _texture(1)
    mask = np.full(im.shape, 255, np.uint8)
    mask[:, :160] = 0
    xy, d, n_all, n_kept = AV.masked_features(im, mask, 4000)
    assert n_all > n_kept > 0 and len(xy) == n_kept and (xy[:, 0] >= 160).all()
    xy, d, n_all, n_kept = AV.masked_features(im, None, 4000)
    assert n_all > 0 and n_kept == 0 and len(xy) == 0


def test_the_masked_tier_is_built_once_and_then_read_from_its_content_addressed_cache(tmp_path):
    import cv2

    base = _texture(7, 400, 300)
    names = [f"{i:08d}.jpg" for i in range(4)]
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    for i, nm in enumerate(names):
        cv2.imwrite(str(tmp_path / "images" / nm), cv2.cvtColor(base[10 * i:10 * i + 240, 12 * i:12 * i + 320],
                                                                 cv2.COLOR_GRAY2BGR))
        cv2.imwrite(str(tmp_path / "masks" / f"{nm}.png"), np.full((240, 320), 255, np.uint8))
    cam = {"fx": 300.0, "fy": 300.0, "cx": 160.0, "cy": 120.0, "width": 320, "height": 240}
    calls = []

    def fake_desc(images):
        calls.append(len(images))
        return np.eye(len(images), 8, dtype=np.float32)

    arrays, rec = AV.build_masked_pairs(tmp_path, names, cam, descriptor_fn=fake_desc, workers=2)
    assert rec["source"] == "built" and rec["pairs"] == len(arrays["i"]) and calls == [4]
    again, rec2 = AV.build_masked_pairs(tmp_path, names, cam, descriptor_fn=fake_desc, workers=2)
    assert rec2["source"] == "cached" and rec2["digest"] == rec["digest"] and calls == [4]
    # a changed mask is another pair set
    cv2.imwrite(str(tmp_path / "masks" / f"{names[0]}.png"), np.zeros((240, 320), np.uint8))
    _, rec3 = AV.build_masked_pairs(tmp_path, names, cam, descriptor_fn=fake_desc, workers=2)
    assert rec3["source"] == "built" and rec3["key"] != rec["key"]


# ---------------------------------------------------------------------------------------------------------------
# end to end: gate_and_publish with the switch on


def _publish_on(tmp_path, monkeypatch, *, value, n, cross, levels, pairs, segments=None, centres=None):
    from tower.world_builder.global_solve import SolveWorkspace, write_solution

    monkeypatch.setenv("TOWER_WORLD_ANCHOR_VERIFY", value)
    sizes = (n,)
    sol = F.multi_solution(sizes, centres=centres if centres is not None else F.line_centres(n))
    links, rots = F.multi_links(sizes, cross)
    F.patch_gate_inputs(monkeypatch, links, rots, list(levels))
    R, C = F.poses_of(sol, n)
    arrays = F.synthetic_pairs(R, C, pairs["pairs"], error_deg=pairs.get("error"))
    built = []

    def builder(root, names, camera):
        built.append(len(names))
        return arrays, {"key": "synthetic", "digest": AV.pair_digest(arrays), "pairs": len(arrays["i"]),
                        "source": "built", "seconds": 0.0}

    monkeypatch.setattr(AV, "_default_pair_builder", builder)
    ws = SolveWorkspace(tmp_path / "w1" / "solve" / F.SID)
    out, record = CP.gate_and_publish(F.Store(tmp_path), "w1", F.SID, ws, sol, final=True, gate=True,
                                      database_path="db", keyframes=F.keyframes(n, segments=segments),
                                      write=write_solution)
    comps = json.loads((ws.root / CP.COMPONENTS_FILENAME).read_text())["components"]
    return out, record, comps, built


def _by_first(comps):
    return {int(e["keyframe_ids"][0][-8:]): (e["keyframes"], e["reasons"], e["shown_as"]) for e in comps}


def test_end_to_end_a_noisy_on_level_stretch_is_left_in_the_room(tmp_path, monkeypatch):
    n = 60
    levels = 1.72 + 0.001 * (np.arange(n) % 5)
    levels[25:35] = [0.29, 3.84, 0.5, 3.5, 0.8, 3.0, 2.9, 3.2, 3.6, 3.3]
    out, record, comps, built = _publish_on(
        tmp_path, monkeypatch, value="scale", n=n, cross=((10, 45), (10, 46), (11, 46)), levels=levels,
        pairs={"pairs": F.chain_pairs(n, breaks=(25, 35))})
    av = record["anchor_verify"]
    assert av["state"] == AV.STATE_APPLIED and av["sealed_kf"] == 0 and built == [n]
    assert av["a2"]["digest"] == "8f5d494796345cd7" and av["anchor_scale"]["test"] == "median-ci-sign"
    assert _by_first(comps) == {0: (60, [], "room")}


def test_end_to_end_scale_seals_the_mid_stretch_as_scale_mismatch(tmp_path, monkeypatch):
    n = 60
    levels = _levels(n, [(25, 35, L2)])
    out, record, comps, built = _publish_on(
        tmp_path, monkeypatch, value="scale", n=n, cross=((10, 45), (10, 46), (11, 46)), levels=levels,
        pairs={"pairs": F.chain_pairs(n, breaks=(25,))})
    av = record["anchor_verify"]
    assert av["state"] == AV.STATE_APPLIED and av["params_digest"] == "8efd726d8a1548d8"
    assert av["sealed"][CG.REASON_SCALE_MISMATCH]["keyframes"] == 10 and av["collateral"]["keyframes"] == 0
    assert _by_first(comps) == {0: (50, [], "room"), 25: (10, [CG.REASON_SCALE_MISMATCH], "none")}
    assert built == [n] and av["seconds"] >= 0 and record["params_digest"] == CG.GateParams().digest()
    assert out.gate["anchor_verify"]["state"] == AV.STATE_APPLIED


def test_end_to_end_images_seal_a_contradicted_group_and_keep_the_rest(tmp_path, monkeypatch):
    n = 100
    ends = [(5, 60), (8, 62), (12, 64), (15, 66)]
    good = [(5, 75), (8, 80), (12, 85), (15, 90), (20, 95), (35, 76), (38, 82), (41, 88)]
    pairs = {"pairs": F.chain_pairs(n, breaks=(60, 70)) + ends + good, "error": {e: 10.0 for e in ends}}
    out, record, comps, _ = _publish_on(
        tmp_path, monkeypatch, value="images", n=n, cross=((8, 80), (8, 81), (9, 81)), levels=[0.0] * n,
        pairs=pairs, segments=[0] * 50 + [1] * 50)
    av = record["anchor_verify"]
    iv = av["image_verification"]
    assert iv[AV.VERDICT_CONTRADICTED]["groups"] == 1 and iv[AV.VERDICT_CONTRADICTED]["keyframes"] == 10
    assert iv[AV.VERDICT_CONFIRMED]["groups"] >= 2
    assert _by_first(comps) == {0: (90, [], "room"), 60: (10, [CG.REASON_LINK_CONTRADICTED], "none")}
    # manager 086: the wire says `link-contradicted`; the Tower-internal record says which part sealed it, and why
    (sg,) = av["sealed_groups"]
    assert sg["source"] == "anchor-image-verification" and sg["reason"] == CG.REASON_LINK_CONTRADICTED
    assert (sg["evidence"], sg["direct_pairs"], sg["keyframes"]) == ("direct", 4, 10)
    assert sg["direct_rot_med"] == pytest.approx(10.0, abs=0.01) and sg["statistic"] == sg["direct_rot_med"]
    assert "image-contradicted" not in json.dumps(comps) and "anchor-image" not in json.dumps(comps)
    # `images` is exactly (b), the seal re-gate and the imports rule: never (a) or (c)
    assert av["parts"] == ["images", "imports"] and "import_units" in record
    assert all("anchor_scale" not in rd and "motion_flags" not in rd for rd in av["rounds"])


def test_end_to_end_collateral_is_reported_with_the_gates_own_reasons(tmp_path, monkeypatch):
    n = 60
    levels = _levels(n, [(25, 35, L2)])
    _, record, comps, _ = _publish_on(tmp_path, monkeypatch, value="scale", n=n, cross=(), levels=levels,
                                      pairs={"pairs": F.chain_pairs(n, breaks=(25,))})
    av = record["anchor_verify"]
    assert av["state"] == AV.STATE_APPLIED and av["collateral"]["keyframes"] == 25
    got = _by_first(comps)
    assert got[0] == (25, [], "room") and got[25][1] == [CG.REASON_SCALE_MISMATCH]
    assert got[35][0] == 25 and got[35][1] and CG.REASON_SCALE_MISMATCH not in got[35][1]
    assert len(av["rounds"]) == 2 and av["cap_hit"] is False                # round 2 found nothing new


def test_a_failed_post_check_publishes_the_room_as_gated(monkeypatch):
    n = 60
    levels = _levels(n, [(25, 35, L2)])
    sol = F.multi_solution((n,), centres=F.line_centres(n))
    links, rots = F.multi_links((n,), ((10, 45), (10, 46), (11, 46)))
    F.patch_gate_inputs(monkeypatch, links, rots, list(levels))
    kfs = F.keyframes(n)
    result = CP.gate_final_solution(F.Store(), "w1", F.SID, sol, database_path="db", keyframes=kfs)
    R, C = F.poses_of(sol, n)
    arrays = F.synthetic_pairs(R, C, F.chain_pairs(n, breaks=(25,)))
    out = AV.verify_published(result, keyframes=kfs, parts={"scale"}, regate=lambda **kw: result,
                              workspace_root=None, pair_builder=lambda *a: (arrays, {"source": "built"}))
    av = out.record["anchor_verify"]
    assert av["state"] == AV.STATE_NOT_APPLIED and "failed its check" in av["why"]
    assert out.components == result.components and out.solution is result.solution


def test_nothing_to_verify_when_the_gate_attached_nothing(tmp_path, monkeypatch):
    n = 60
    out, record, comps, built = _publish_on(tmp_path, monkeypatch, value="on", n=n, cross=(),
                                            levels=[None] * n, pairs={"pairs": []})
    assert record["attach"] is False and record["anchor_verify"]["state"] == AV.STATE_NOT_RUN and built == []


def test_a_failure_publishes_the_room_as_gated_and_says_so(tmp_path, monkeypatch):
    n = 60
    levels = _levels(n, [(25, 35, L2)])

    def broken(*a, **kw):
        raise AV.DescriptorUnavailable("DINOv2-small unavailable (test)")

    monkeypatch.setattr(AV, "build_masked_pairs", broken)
    monkeypatch.setenv("TOWER_WORLD_ANCHOR_VERIFY", "images")
    sol = F.multi_solution((n,), centres=F.line_centres(n))
    links, rots = F.multi_links((n,), ())
    F.patch_gate_inputs(monkeypatch, links, rots, list(levels))
    result = CP.gate_final_solution(F.Store(), "w1", F.SID, sol, database_path="db", keyframes=F.keyframes(n))
    out = CP.anchor_verified(F.Store(), "w1", F.SID, result, database_path="db", keyframes=F.keyframes(n),
                             workspace_root=tmp_path)
    av = out.record["anchor_verify"]
    assert av["state"] == AV.STATE_FAILED and "DINOv2" in av["detail"]
    assert out.components == result.components


def test_with_the_switch_off_the_publish_step_never_imports_the_verification(tmp_path, monkeypatch):
    monkeypatch.delenv("TOWER_WORLD_ANCHOR_VERIFY", raising=False)
    n = 50
    sol = F.multi_solution((n,))
    links, rots = F.multi_links((n,), ())
    F.patch_gate_inputs(monkeypatch, links, rots, [0.0] * n)
    result = CP.gate_final_solution(F.Store(), "w1", F.SID, sol, database_path="db", keyframes=F.keyframes(n))
    assert CP.anchor_verified(F.Store(), "w1", F.SID, result, database_path="db", keyframes=F.keyframes(n),
                              workspace_root=tmp_path) is result
    assert "anchor_verify" not in result.record and "import_units" not in result.record


# ---------------------------------------------------------------------------------------------------------------
# (i) one import = one link: the database's imported pairs, the journal's episodes


def _import_world(tmp_path, created, journal):
    import sqlite3

    db = tmp_path / "database.db"
    con = sqlite3.connect(str(db))
    con.execute("create table images (image_id integer primary key, name text)")
    con.executemany("insert into images values (?, ?)", [(i + 1, F.name(i)) for i in range(60)])
    con.execute("create table wb_revisit_imported (pair_id integer primary key)")
    for a, b in created:
        lo, hi = sorted((a + 1, b + 1))
        con.execute("insert into wb_revisit_imported values (?)", (lo * 2147483647 + hi,))
    con.commit()
    con.close()
    session = tmp_path / "sessions" / F.SID
    session.mkdir(parents=True)
    (session / "keyframes.jsonl").write_text("\n".join(json.dumps(
        {"keyframe_id": f"{F.SID}:{i:08d}", "image_relpath": f"images/{F.name(i)}"}) for i in range(60)) + "\n")
    (session / "events.jsonl").write_text("\n".join(json.dumps(e) for e in journal) + "\n")

    class Store(F.Store):
        def session_dir(self, world_id, session_id):
            return session

    return Store(tmp_path), db


def _accepted(episode, refs, anchor, inliers=80):
    return {"kind": "recovery_accepted", "payload": {
        "episode": episode, "links": [{"ref_keyframe_id": f"{F.SID}:{r:08d}", "inliers": inliers} for r in refs],
        "anchor": {"keyframe_id": f"{F.SID}:{anchor:08d}", "identity": True, "inliers": None}}}


def test_one_accepted_triangle_is_one_unit(tmp_path):
    store, db = _import_world(tmp_path, [(10, 40), (11, 40)], [_accepted(1, [10, 11], 40)])
    units, audit = AV.import_link_units(store, "w1", F.SID, db, {})
    assert units == {(F.name(10), F.name(40)): "import:1", (F.name(11), F.name(40)): "import:1"}
    assert audit["imported_pairs"] == 2 and audit["units"] == 1


def test_a_listed_pair_the_database_already_held_is_a_unit_of_its_own(tmp_path):
    # the journal lists (12, 40) too, but the import did not create it (kept_existing): not in the table
    store, db = _import_world(tmp_path, [(10, 40), (11, 40)], [_accepted(1, [10, 11, 12], 40)])
    units, _ = AV.import_link_units(store, "w1", F.SID, db, {})
    assert (F.name(12), F.name(40)) not in units and len(units) == 2


def test_no_table_or_no_database_is_todays_gate(tmp_path):
    units, audit = AV.import_link_units(F.Store(tmp_path), "w1", F.SID, tmp_path / "missing.db", {})
    assert units == {} and audit["imported_pairs"] == 0


def test_imports_on_through_the_gate_refuses_a_group_held_only_by_one_import(tmp_path, monkeypatch):
    sizes = (30, 20)
    store, db = _import_world(tmp_path, [(29, 30), (29, 31)], [_accepted(4, [30, 31], 29)])
    sol = F.multi_solution(sizes)
    links, rots = F.multi_links(sizes, ((29, 30), (29, 31)))
    F.patch_gate_inputs(monkeypatch, links, rots, [0.0] * 50)
    kfs = F.keyframes(50)
    monkeypatch.delenv("TOWER_WORLD_ANCHOR_VERIFY", raising=False)
    today = CP.gate_final_solution(store, "w1", F.SID, sol, database_path=db, keyframes=kfs)
    assert today.record["components"] == {"placed": 1, "area": 0, "none": 0} and "import_units" not in today.record
    monkeypatch.setenv("TOWER_WORLD_ANCHOR_VERIFY", "imports")
    one = CP.gate_final_solution(store, "w1", F.SID, sol, database_path=db, keyframes=kfs)
    assert one.record["components"] == {"placed": 1, "area": 1, "none": 0}
    assert one.record["import_units"]["units"] == 1 and one.record["params_digest"] == CG.GateParams().digest()
    reasons = [e["reasons"] for e in one.components["components"] if e["state"] != "placed"]
    assert reasons == [[CG.REASON_SINGLE_UNCONFIRMED_LINK]]


# ---------------------------------------------------------------------------------------------------------------
# once, on the chosen draw, after the consensus


def test_after_the_consensus_the_withheld_group_stays_withheld_and_a_contradicted_group_is_sealed(tmp_path,
                                                                                                    monkeypatch):
    from tower.world_builder.global_solve import SolveWorkspace, write_solution

    monkeypatch.setenv("TOWER_WORLD_ANCHOR_VERIFY", "images")
    sizes, n = F.CONSENSUS_SIZES, sum(F.CONSENSUS_SIZES)
    links, rots = F.multi_links(sizes, F.CONSENSUS_CROSS)
    F.patch_gate_inputs(monkeypatch, links, rots, [0.0] * n)
    cands = [F.multi_solution(sizes, o) for o in F.CONSENSUS_OFFSETS]
    R, C = F.poses_of(cands[0], n)
    bad = [(0, 31), (2, 34), (4, 36), (6, 38)]                               # B (30..49) against the room
    good = [(0, 52), (3, 55), (6, 58), (9, 61), (12, 64)]                    # C (50..69) against the room
    arrays = F.synthetic_pairs(R, C, F.chain_pairs(n, breaks=(30, 50, 70)) + bad + good,
                               error_deg={e: 10.0 for e in bad} | {e: 1.0 for e in good})
    monkeypatch.setattr(AV, "_default_pair_builder", lambda root, names, camera: (arrays, {"source": "built"}))
    plan = CP.ConsensusPlan(draws=3, seed=7, map_draw=lambda seed: cands[seed - 7])
    segments = [0] * 30 + [1] * 20 + [2] * 20 + [3] * 6
    ws = SolveWorkspace(tmp_path / "w1" / "solve" / F.SID)
    out, record = CP.gate_and_publish(F.Store(tmp_path), "w1", F.SID, ws, cands[0], final=True, gate=True,
                                      database_path="db", keyframes=F.keyframes(n, segments=segments),
                                      write=write_solution, consensus=plan)
    assert record["consensus"]["state"] == CP.CONSENSUS_APPLIED and record["consensus"]["detached"]
    av = record["anchor_verify"]
    assert av["state"] == AV.STATE_APPLIED and av["sealed"][AV.IMAGE_SEAL_REASON]["keyframes"] == 20
    comps = {int(e["keyframe_ids"][0][-8:]): (e["keyframes"], e["reasons"])
             for e in json.loads((ws.root / CP.COMPONENTS_FILENAME).read_text())["components"]}
    assert comps == {0: (50, []), 30: (20, [AV.IMAGE_SEAL_REASON]), 70: (6, [CG.REASON_SEED_UNSTABLE])}
    assert (ws.root / CP.CONSENSUS_FILENAME).exists()
    assert out.gate["anchor_verify"]["image_verification"][AV.VERDICT_CONTRADICTED]["groups"] == 1


# ---------------------------------------------------------------------------------------------------------------
# (c) feeds (a): a flagged transition is a cut; alone it seals nothing


def _jump_centres(n, at, metres):
    return [c + (np.array([metres, 0.0, 0.0]) if i >= at else 0.0) for i, c in enumerate(F.line_centres(n))]


def test_on_a_motion_flag_cuts_the_run_so_a2_seals_the_stretch_behind_it(tmp_path, monkeypatch):
    n = 60
    levels = _levels(n, [(25, 35, L2)])
    # the images chain straight across 24 | 25, but the camera jumps 5 m in 0.5 s there (bound: 1.3 m)
    out, record, comps, _ = _publish_on(
        tmp_path, monkeypatch, value="on", n=n, cross=((10, 45), (10, 46), (11, 46)), levels=levels,
        pairs={"pairs": F.chain_pairs(n)}, centres=_jump_centres(n, 25, 5.0))
    av = record["anchor_verify"]
    assert av["parts"] == ["images", "imports", "motion", "scale"]
    assert [(f["a"][-2:], f["b"][-2:]) for f in av["motion_flags"]] == [("24", "25")]
    assert av["sealed"][CG.REASON_SCALE_MISMATCH]["keyframes"] == 10
    assert _by_first(comps) == {0: (50, [], "room"), 25: (10, [CG.REASON_SCALE_MISMATCH], "none")}
    assert av["sealed_groups"][0]["source"] == "anchor-scale-segmentation"


def test_without_the_flag_the_same_stretch_stays(tmp_path, monkeypatch):
    n = 60
    levels = _levels(n, [(25, 35, L2)])
    _, record, comps, _ = _publish_on(
        tmp_path, monkeypatch, value="scale,images", n=n, cross=((10, 45), (10, 46), (11, 46)), levels=levels,
        pairs={"pairs": F.chain_pairs(n)}, centres=_jump_centres(n, 25, 5.0))
    assert record["anchor_verify"]["sealed_kf"] == 0 and _by_first(comps) == {0: (60, [], "room")}


def test_motion_alone_flags_and_seals_nothing(tmp_path, monkeypatch):
    n = 60
    levels = _levels(n, [(25, 35, L2)])
    _, record, comps, built = _publish_on(
        tmp_path, monkeypatch, value="motion", n=n, cross=((10, 45), (10, 46), (11, 46)), levels=levels,
        pairs={"pairs": F.chain_pairs(n)}, centres=_jump_centres(n, 25, 5.0))
    av = record["anchor_verify"]
    assert len(av["motion_flags"]) == 1 and av["sealed_kf"] == 0 and built == []
    assert _by_first(comps) == {0: (60, [], "room")} and "import_units" not in record
