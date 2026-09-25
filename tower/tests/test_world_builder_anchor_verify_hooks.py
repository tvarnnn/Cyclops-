"""The anchor verification's switch and the gate's new hooks (RUN P4-IV RULE.md sections 1, 7 and 9), and the
no-room-round fix in `coherence_gate._finish` (the lead's 01:42 live case; NOT behind the switch).

Pinned here:
  * `TOWER_WORLD_ANCHOR_VERIFY`: unset, blank and `off` are off; `on` is every part; a comma list turns on its
    parts; anything else is off and logged;
  * `apply_gate(seal=None | {}, link_units=None | {})` is today's gate, exactly;
  * `seal`: sealed cameras leave the graph before its blocks form, become ONE piece per reason and honoured-link
    connected set with that reason only, and a group attached only through them is collateral;
  * `link_units`: two links of one unit are one link to the redundancy test;
  * a solve with no posed camera gates cleanly (no StopIteration), and publishes `components: null`.
"""

from __future__ import annotations

import json
import logging
import math

import numpy as np
import pytest

from tests import wb_anchor_verify_fixtures as F
from tower import config
from tower.world_builder import coherence_gate as CG
from tower.world_builder import coherence_publish as CP

# ---------------------------------------------------------------------------------------------------------------
# the switch


@pytest.mark.parametrize("value", [None, "", "   ", "off", "OFF", " off "])
def test_unset_blank_and_off_are_off(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(config.WORLD_ANCHOR_VERIFY_ENV, raising=False)
    else:
        monkeypatch.setenv(config.WORLD_ANCHOR_VERIFY_ENV, value)
    assert config.world_anchor_verify_setting() == frozenset()


@pytest.mark.parametrize("value, parts", [
    ("on", {"scale", "images", "motion", "imports"}),
    ("ON", {"scale", "images", "motion", "imports"}),
    ("scale", {"scale"}),
    ("images", {"images", "imports"}),              # manager 086: walk 4 runs `images` = (b) + the imports rule
    ("motion", {"motion"}),
    ("imports", {"imports"}),
    ("scale,images", {"scale", "images", "imports"}),
    (" Scale , MOTION ", {"scale", "motion"}),
    ("scale,images,motion,imports", {"scale", "images", "motion", "imports"}),
])
def test_on_and_comma_lists_turn_on_their_parts(monkeypatch, value, parts):
    monkeypatch.setenv(config.WORLD_ANCHOR_VERIFY_ENV, value)
    assert config.world_anchor_verify_setting() == frozenset(parts)


@pytest.mark.parametrize("value", ["yes", "1", "true", "scale,foo", "off,scale", "on,scale", "sclae", ","])
def test_garbage_is_off_and_logged(monkeypatch, caplog, value):
    monkeypatch.setenv(config.WORLD_ANCHOR_VERIFY_ENV, value)
    with caplog.at_level(logging.WARNING, logger="tower.config"):
        assert config.world_anchor_verify_setting() == frozenset()
    assert config.WORLD_ANCHOR_VERIFY_ENV in caplog.text


def test_the_declared_digests():
    from tower.world_builder import anchor_verify as AV

    assert AV.VerifyParams().digest() == "8efd726d8a1548d8"          # RULE.md section 4.6
    assert CG.GateParams().digest() == "6ce602286efb999f"            # the gate's, unchanged
    assert (AV.VerifyParams().theta_rot_deg, AV.VerifyParams().theta_t_deg,
            AV.VerifyParams().theta_2hop_deg) == (4.66, 23.22, 23.04)


def test_the_image_seal_reason_is_the_contracts_link_contradicted():
    # manager 086: walk 4 seals (b)'s groups `link-contradicted`; `image-contradicted` is never written
    from tower.world_builder import anchor_verify as AV

    assert AV.IMAGE_SEAL_REASON == CG.REASON_LINK_CONTRADICTED == "link-contradicted"
    assert AV.IMAGE_SEAL_REASON in CG.REASONS
    assert not hasattr(CG, "REASON_IMAGE_CONTRADICTED")


# ---------------------------------------------------------------------------------------------------------------
# the hooks: None or empty is today's gate


def _dump(out):
    return json.dumps(out, sort_keys=True, default=str)


@pytest.mark.parametrize("hooks", [{"seal": None}, {"seal": {}}, {"link_units": None}, {"link_units": {}},
                                   {"seal": {}, "link_units": {}}])
def test_empty_hooks_are_todays_gate(hooks):
    model, links, rots, levels = F.mid_stretch_model()
    today = CG.apply_gate(model, links, levels, link_rotations=rots, masks_applied=True)
    assert _dump(CG.apply_gate(model, links, levels, link_rotations=rots, masks_applied=True, **hooks)) == _dump(today)
    held = CG.apply_gate(model, links, levels, link_rotations=rots, masks_applied=True, withhold=[F.name(40)],
                         room=[F.name(0)])
    assert _dump(CG.apply_gate(model, links, levels, link_rotations=rots, masks_applied=True, withhold=[F.name(40)],
                               room=[F.name(0)], **hooks)) == _dump(held)


# ---------------------------------------------------------------------------------------------------------------
# seal


def _loop_links(n, extra):
    links = {(F.name(i), F.name(i + d)): 100 for i in range(n) for d in (1, 2) if i + d < n}
    for a, b in extra:
        links[(F.name(a), F.name(b))] = 60
    return links


def test_a_sealed_stretch_is_one_piece_with_its_reason_and_the_rest_attached_only_through_it_is_collateral():
    model, links, rots, levels = F.mid_stretch_model()
    seal = {F.name(i): CG.REASON_SCALE_MISMATCH for i in range(25, 35)}
    out = CG.apply_gate(model, links, levels, link_rotations=rots, masks_applied=True, seal=seal,
                        room=[F.name(i) for i in range(60)])
    lab = out["labels"]
    stretch = {lab[F.name(i)] for i in range(25, 35)}
    assert len(stretch) == 1 and 0 not in stretch
    stretch_label = stretch.pop()
    comp = next(c for c in out["components"] if c["label"] == stretch_label)
    assert comp["reasons"] == [CG.REASON_SCALE_MISMATCH]
    # the chain 35..59 was joined to 0..24 only through the stretch: collateral, with the gate's own reason
    assert {lab[F.name(i)] for i in range(0, 25)} == {0}
    tail = {lab[F.name(i)] for i in range(35, 60)}
    assert len(tail) == 1 and 0 not in tail
    tail_label = tail.pop()
    tail_comp = next(c for c in out["components"] if c["label"] == tail_label)
    assert tail_comp["reasons"] and CG.REASON_SCALE_MISMATCH not in tail_comp["reasons"]


def test_a_group_with_its_own_redundant_links_to_the_room_stays_when_a_stretch_is_sealed():
    model, _, rots0, levels = F.mid_stretch_model()
    links = _loop_links(60, [(10, 45), (10, 46), (11, 46)])          # a revisit: 35..59 sees 10..11 again
    rots = {(a, b): model.R_cw[int(b[:8])] @ model.R_cw[int(a[:8])].T for a, b in links}
    seal = {F.name(i): CG.REASON_LINK_CONTRADICTED for i in range(25, 35)}
    out = CG.apply_gate(model, links, levels, link_rotations=rots, masks_applied=True, seal=seal,
                        room=[F.name(i) for i in range(60)])
    lab = out["labels"]
    assert all(lab[F.name(i)] == 0 for i in list(range(0, 25)) + list(range(35, 60)))
    piece = {lab[F.name(i)] for i in range(25, 35)}
    assert len(piece) == 1
    assert next(c for c in out["components"] if c["label"] in piece)["reasons"] == [CG.REASON_LINK_CONTRADICTED]


def test_sealed_cameras_with_two_reasons_are_two_pieces():
    model, links, rots, levels = F.mid_stretch_model()
    seal = {F.name(i): CG.REASON_SCALE_MISMATCH for i in range(25, 30)}
    seal.update({F.name(i): CG.REASON_LINK_CONTRADICTED for i in range(30, 35)})
    out = CG.apply_gate(model, links, levels, link_rotations=rots, masks_applied=True, seal=seal)
    lab = out["labels"]
    a, b = {lab[F.name(i)] for i in range(25, 30)}, {lab[F.name(i)] for i in range(30, 35)}
    assert len(a) == len(b) == 1 and a != b
    reasons = {c["label"]: c["reasons"] for c in out["components"]}
    assert reasons[a.pop()] == [CG.REASON_SCALE_MISMATCH] and reasons[b.pop()] == [CG.REASON_LINK_CONTRADICTED]


def test_the_camera_allow_list_bars_a_group_outside_the_room():
    # B (30..49) joined by a triangle; with a seal and a room that does not hold B, B may not join
    sizes = (30, 20)
    sol = F.multi_solution(sizes)
    links, rots = F.multi_links(sizes, ((29, 30), (29, 31)))
    model = CP.solve_model(sol, {k.keyframe_id: F.name(i) for i, k in enumerate(F.keyframes(50))})
    lv = {F.name(i): 0.0 for i in range(50)}
    free = CG.apply_gate(model, links, lv, link_rotations=rots, masks_applied=True)
    assert all(v == 0 for v in free["labels"].values())
    # seal the room's first camera (the rest of the chain stays one block): B is barred by the allow-list
    out = CG.apply_gate(model, links, lv, link_rotations=rots, masks_applied=True,
                        seal={F.name(0): CG.REASON_SCALE_MISMATCH}, room=[F.name(i) for i in range(30)])
    assert all(out["labels"][F.name(i)] != 0 for i in range(30, 50))
    assert all(out["labels"][F.name(i)] == 0 for i in range(1, 30))
    # the same seal with the room allow-listing B too: B joins as today
    out = CG.apply_gate(model, links, lv, link_rotations=rots, masks_applied=True,
                        seal={F.name(0): CG.REASON_SCALE_MISMATCH}, room=[F.name(i) for i in range(50)])
    assert all(out["labels"][F.name(i)] == 0 for i in range(1, 50))


# ---------------------------------------------------------------------------------------------------------------
# link_units (part (i))


def _triangle_gate(units):
    sizes = (30, 20)
    sol = F.multi_solution(sizes)
    links, rots = F.multi_links(sizes, ((29, 30), (29, 31)))
    model = CP.solve_model(sol, {k.keyframe_id: F.name(i) for i, k in enumerate(F.keyframes(50))})
    lv = {F.name(i): 0.0 for i in range(50)}
    return CG.apply_gate(model, links, lv, link_rotations=rots, masks_applied=True, link_units=units)


def test_one_import_is_one_link_so_its_triangle_alone_does_not_attach():
    one = {tuple(sorted((F.name(29), F.name(30)))): "import:1", tuple(sorted((F.name(29), F.name(31)))): "import:1"}
    out = _triangle_gate(one)
    assert all(out["labels"][F.name(i)] != 0 for i in range(30, 50))
    reasons = {c["label"]: c["reasons"] for c in out["components"]}
    assert reasons[out["labels"][F.name(30)]] == [CG.REASON_SINGLE_UNCONFIRMED_LINK]


def test_two_imports_or_an_independent_link_restore_todays_attachment():
    two = {tuple(sorted((F.name(29), F.name(30)))): "import:1", tuple(sorted((F.name(29), F.name(31)))): "import:2"}
    assert all(v == 0 for v in _triangle_gate(two)["labels"].values())
    only_one_imported = {tuple(sorted((F.name(29), F.name(30)))): "import:1"}
    assert all(v == 0 for v in _triangle_gate(only_one_imported)["labels"].values())


def test_redundant_links_with_units():
    adj = [set(), set(), {3}, {2}]
    cross = [(0, 2), (0, 3)]                   # one group image, two kept ends that are linked: a triangle
    assert CG.redundant_links(cross, adj)
    assert CG.redundant_links(cross, adj, ["u1", "u2"])
    assert not CG.redundant_links(cross, adj, ["u1", "u1"])


# ---------------------------------------------------------------------------------------------------------------
# no room round (the lead's 01:42 live case): NOT behind the switch


def test_a_solve_with_no_camera_gates_cleanly():
    model = CG.SolveModel(names=[], component=np.zeros(0, np.int64), R_cw=np.zeros((0, 3, 3)),
                          n_obs=np.zeros(0, np.int64), obs_image=np.zeros(0, np.int64),
                          obs_point=np.zeros(0, np.int64), n_points=0)
    out = CG.apply_gate(model, {}, {}, link_rotations={}, masks_applied=True)
    assert out["labels"] == {} and out["components"] == [] and out["rounds"] == []
    assert out["metric_available"] is False and out["params_digest"] == CG.GateParams().digest()


def test_a_false_start_walk_of_three_keyframes_and_no_pose_publishes_components_null(monkeypatch):
    from tower.world_builder.global_solve import Solution

    sol = Solution(solver="glomap", solved_at=1.0, input_digest="d",
                   keyframe_ids=[f"{F.SID}:{i:08d}" for i in range(3)], poses={}, components=[], xyz=np.zeros((0, 3), np.float32), rgb=np.zeros((0, 3), np.uint8),
                   component=np.zeros(0, np.int32), first_keyframe=np.zeros(0, np.int32),
                   track_length=np.zeros(0, np.int32), error=np.zeros(0, np.float32),
                   observations=np.zeros((0, 3), np.int32), camera=None, transients={"state": "applied"})
    F.patch_gate_inputs(monkeypatch, {}, {}, [None] * 3)
    result = CP.gate_final_solution(F.Store(), "w1", F.SID, sol, database_path="db", keyframes=F.keyframes(3))
    assert result.record["state"] == CP.GATE_STATE_APPLIED            # not "failed": no exception
    assert result.components is None and result.record["components"] is None
    assert result.record["attach"] is False


def test_posed_cameras_always_have_a_room_label():
    # every posed camera unsupported (< 30 observations): still one round, labelled 0, never a StopIteration
    model = CG.SolveModel(names=[F.name(i) for i in range(3)], component=np.zeros(3, np.int64),
                          R_cw=np.stack([np.eye(3)] * 3), n_obs=np.array([3, 2, 1]),
                          obs_image=np.array([0, 1, 2]), obs_point=np.array([0, 0, 0]), n_points=1)
    out = CG.apply_gate(model, {}, {}, link_rotations={}, masks_applied=True)
    assert set(out["labels"].values()) == {0}
    assert [c["label"] for c in out["components"]] == [0]
    assert math.isclose(out["evidence"]["metric_fraction"], 0.0)


# ---------------------------------------------------------------------------------------------------------------
# the seal re-gate's allow-list is CAMERA BY CAMERA (the lead, from P4-IV phase 2)


def _reshaping_model():
    """Anchor A = 0..49 (a chain); Q = 50..89, densely linked (gap <= 12), joined to A by a triangle through 49 and
    another through 40; levels 0 on 0..69 and x2 on 70..89. The gate keeps 0..69 and leaves 70..89 out
    (`scale-mismatch`). Sealing 50..60 cuts Q off A except through 40, and Q's scale split then re-forms as
    [61..70] | [71..89]: a segment that MIXES room cameras (61..69) with a non-room one (70)."""
    n = 90
    names = [F.name(i) for i in range(n)]
    R = np.stack([F.rz(3.0 * i) for i in range(n)])
    obs_i, obs_p, p = [], [], 0
    for j in range(n):
        for _ in range(13):
            for c in range(j, min(n, j + 3)):
                obs_i.append(c)
                obs_p.append(p)
            p += 1
        for _ in range(30):
            obs_i.append(j)
            obs_p.append(p)
            p += 1
    obs_i, obs_p = np.asarray(obs_i), np.asarray(obs_p)
    model = CG.SolveModel(names=names, component=np.zeros(n, np.int64), R_cw=R,
                          n_obs=np.bincount(obs_i, minlength=n), obs_image=obs_i, obs_point=obs_p, n_points=p)
    links = {(F.name(i), F.name(i + d)): 100 for i in range(50) for d in (1, 2) if i + d < 50}
    links.update({(F.name(i), F.name(j)): 100 for i in range(50, n) for j in range(i + 1, min(n, i + 13))})
    for a, b in ((49, 50), (49, 51), (40, 62), (40, 63)):
        links[(F.name(a), F.name(b))] = 60
    rots = {(a, b): R[int(b[:8])] @ R[int(a[:8])].T for a, b in links}
    levels = {F.name(i): (math.log(2.0) if i >= 70 else 0.0) + 0.001 * (i % 3) for i in range(n)}
    return model, links, rots, levels


def test_a_seal_that_reshapes_the_blocks_never_brings_a_non_room_camera_into_the_room():
    model, links, rots, levels = _reshaping_model()
    chosen = CG.apply_gate(model, links, levels, link_rotations=rots, masks_applied=True)
    room = [nm for nm, lab in chosen["labels"].items() if lab == 0]
    assert room == [F.name(i) for i in range(70)]
    outside = {chosen["labels"][F.name(i)] for i in range(70, 90)}
    assert len(outside) == 1
    seal = {F.name(i): CG.REASON_LINK_CONTRADICTED for i in range(50, 61)}
    out = CG.apply_gate(model, links, levels, link_rotations=rots, masks_applied=True, seal=seal, room=room)
    lab = out["labels"]
    assert all(lab[F.name(i)] != 0 for i in range(70, 90)), "a non-room camera entered the room"
    assert all(lab[F.name(i)] == 0 for i in list(range(50)) + list(range(61, 70)))
    assert len({lab[F.name(i)] for i in range(70, 90)}) == 1              # the outside piece, whole
    # every group the re-gate formed lies wholly on one side of the room
    room_set = set(room)
    for g in out["groups"]:
        sides = {nm in room_set for nm in g["members"]}
        assert len(sides) == 1, g["first_camera"]
    # without the camera-by-camera split the re-formed [61..70] segment would mix the sides
    mixed = CG.scale_split(np.arange(61, 90), np.asarray([levels[F.name(i)] for i in range(90)]),
                           np.arange(90), CG.GateParams())
    assert any(61 in g and 70 in g for g in mixed)
