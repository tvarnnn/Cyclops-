"""The experiment driver hands the evidence gate its two inputs.

`GateParams(rule="evidence")` refuses to run without a per-camera metric scale
and is blind to redundancy without the verified view graph. The driver must
pass both (`links` from the solver's own database, `metric_log` = the harness
TRI ratio); the shared-point rule needs neither and must not touch them.
"""

from __future__ import annotations

import json
import sqlite3

import numpy as np
import pytest

from tower.world_builder.coherence_exp import driver as D
from tower.world_builder.coherence_exp import gate as G

from tests.test_world_coherence_variant import ladder_links, metric, two_islands  # noqa: E402

BASE = 2147483647


def _db(path, links):
    """A COLMAP-shaped database holding `links` as verified two-view geometries."""
    names = sorted({n for pair in links for n in pair})
    ids = {n: i + 1 for i, n in enumerate(names)}
    con = sqlite3.connect(path)
    con.execute("create table images (image_id integer primary key, name text, camera_id integer)")
    con.execute("create table two_view_geometries (pair_id integer primary key, rows integer, cols integer, "
                "data blob, config integer)")
    con.executemany("insert into images values (?, ?, 1)", [(i, n) for n, i in ids.items()])
    rows = []
    for (a, b), inl in links.items():
        ia, ib = sorted((ids[a], ids[b]))
        rows.append((ia * BASE + ib, inl, 2))
    ia, ib = ids["000010_k.jpg"], ids["000045_k.jpg"]   # an unlinked pair ...
    rows.append((ia * BASE + ib, 3, 2))                  # ... below the verification floor: not a link
    con.executemany("insert into two_view_geometries values (?, ?, 2, null, ?)", rows)
    con.commit()
    con.close()
    return path


def test_the_shared_point_rule_gets_no_extra_inputs(tmp_path):
    extra, info = D.gate_inputs(G.GateParams(), [two_islands(0)], {"images": []}, tmp_path / "no-world",
                                tmp_path / "no.db", tmp_path / "no-cache")
    assert extra == {} and info == {"rule": "shared_k"}


def test_the_evidence_rule_gets_verified_links_and_the_metric_scale(tmp_path, monkeypatch):
    links = ladder_links(cross=[(k, 30 + k, 50) for k in range(5)])
    db = _db(tmp_path / "database.db", links)
    seen = {}

    def fake_tri(model, staging, world_dir, metric_cache):
        seen["seed"], seen["cache"] = model.seed, metric_cache
        return metric(4.0, 4.0), {"keyframes_with_ratio": 50}

    monkeypatch.setattr(D, "metric_log_tri", fake_tri)
    models = [two_islands(s, shared_between=60, noise=0.001) for s in (3, 1, 2)]
    extra, info = D.gate_inputs(G.GateParams(rule="evidence"), models, {"images": []}, tmp_path,
                                db, tmp_path / "cache")
    assert extra["links"] == links                  # the floor-level pair is not a link
    assert extra["metric_log"] == metric(4.0, 4.0)
    assert seen == {"seed": 3, "cache": tmp_path / "cache"}   # the reference (first) model's poses
    assert info["links"] == len(links) and info["metric_log"]["keyframes_with_ratio"] == 50


def test_the_wired_inputs_decide_the_gate(tmp_path, monkeypatch):
    """Same models and links; only the metric scale differs -> the wiring is what splits."""
    links = ladder_links(cross=[(k, 30 + k, 50) for k in range(5)])
    db = _db(tmp_path / "database.db", links)
    models = [two_islands(s, shared_between=60, noise=0.001) for s in range(3)]
    gp = G.GateParams(rule="evidence", min_obs=5)
    for level_b, n_groups in ((4.0, 1), (4.0 * 1.6, 2)):
        monkeypatch.setattr(D, "metric_log_tri", lambda *a, lb=level_b: (metric(4.0, lb), {}))
        extra, _ = D.gate_inputs(gp, models, {"images": []}, tmp_path, db, tmp_path)
        out = G.apply_rigid_gate(models, gp, **extra)
        assert len(set(out["labels"].values())) == n_groups, level_b


def test_the_evidence_rule_refuses_without_the_harness_caches(tmp_path):
    w = tmp_path / "world"
    (w / "sessions" / "s1").mkdir(parents=True)
    (w / "world.json").write_text(json.dumps({"world_id": "w1"}))
    (w / "sessions" / "s1" / "session.json").write_text(json.dumps({"session_id": "s1"}))
    (w / "sessions" / "s1" / "keyframes.jsonl").write_text(
        json.dumps({"keyframe_id": "s1:00000001", "image_relpath": "images/00000001.jpg"}) + "\n")
    with pytest.raises(RuntimeError, match="harness depth and pair caches"):
        D.metric_log_tri(two_islands(0), {"images": []}, w, tmp_path / "empty-cache")


def test_run_variant_passes_the_inputs_and_records_them():
    src = open(D.__file__, encoding="utf-8").read()
    assert "gate_mod.apply_rigid_gate(models, gp, **extra)" in src
    assert "inputs=gate_inputs_info" in src and "gate_inputs=gate_inputs_info" in src
    cfg = D.VariantConfig.from_json({"metric_cache": "x", "gate": "rigid", "gate_params": {"rule": "evidence"}})
    assert cfg.metric_cache == "x" and not cfg.is_product_recipe()
    assert D.VariantConfig().metric_cache is None


def test_each_seed_is_gated_on_its_own(tmp_path, monkeypatch):
    """CAND scored seeds/<s>/variant, which is exported BEFORE the gate, as if it were gated. The
    per-seed gated export must gate each seed alone, with that seed's own metric scale."""
    links = ladder_links(cross=[(k, 30 + k, 50) for k in range(5)])
    db = _db(tmp_path / "database.db", links)
    models = [two_islands(s, shared_between=60, noise=0.001) for s in range(3)]
    gp = G.GateParams(rule="evidence", min_obs=5)
    seen = []

    def fake_tri(model, *a):
        seen.append(model.seed)
        return metric(4.0, 4.0 * 1.6 if model.seed == 1 else 4.0), {}

    monkeypatch.setattr(D, "metric_log_tri", fake_tri)
    out = D.gate_each_seed(gp, models, {"images": []}, tmp_path, db, tmp_path)
    assert [g["seed"] for g in out] == [0, 1, 2] and seen == [0, 1, 2]
    groups = {g["seed"]: len(set(g["report"]["labels"].values())) for g in out}
    assert groups == {0: 1, 1: 2, 2: 1}          # only the seed whose own scale steps is split
    assert all(set(g["report"]["labels"]) == set(m.names) for g, m in zip(out, models))
    src = open(D.__file__, encoding="utf-8").read()
    assert '"variant_gated"' in src and "BEFORE any gate" in src


def test_uncalibrated_links_can_be_excluded_from_the_evidence(tmp_path, monkeypatch):
    """6839fb8f kf 92-98 was attached by four UNCALIBRATED (config 3) links that the solve contradicts by
    34-96 deg. With `exclude_uncalibrated_links` such links are not evidence, so a group they alone attach
    is split off; the default keeps today's behaviour."""
    links = ladder_links(cross=[(k, 30 + k, 50) for k in range(5)])
    cross = {tuple(sorted((f"{k:06d}_k.jpg", f"{30 + k:06d}_k.jpg"))) for k in range(5)}
    db = _db(tmp_path / "database.db", links)
    con = sqlite3.connect(db)
    ids = dict(con.execute("select name, image_id from images").fetchall())
    for a, b in cross:
        ia, ib = sorted((ids[a], ids[b]))
        con.execute("update two_view_geometries set config = 3 where pair_id = ?", (ia * BASE + ib,))
    con.commit()
    con.close()
    models = [two_islands(s, shared_between=60, noise=0.001) for s in range(3)]
    monkeypatch.setattr(D, "metric_log_tri", lambda *a: (metric(4.0, 4.0), {}))
    for exclude, n_groups in ((False, 1), (True, 2)):
        gp = G.GateParams(rule="evidence", min_obs=5, exclude_uncalibrated_links=exclude)
        extra, info = D.gate_inputs(gp, models, {"images": []}, tmp_path, db, tmp_path)
        assert (3 in info["links_excluded_configs"]) is exclude
        out = G.apply_rigid_gate(models, gp, **extra)
        assert len(set(out["labels"].values())) == n_groups, exclude


def test_a_group_without_measurable_scale_attaches_only_under_the_lenient_reading(tmp_path, monkeypatch):
    """2f447162: a 14-keyframe island at scale x0.02-0.04 attached on two links because its metric level
    could not be measured, and the default reading lets an unmeasured group pass. `require_group_scale`
    attaches a group only when its level is measured and matches."""
    # B joins A through the cut vertex 30 (linked to 28 and 29): its own block, attached by a closed triangle
    links = ladder_links(cross=[(29, 30, 50), (28, 30, 50)])
    db = _db(tmp_path / "database.db", links)
    models = [two_islands(s, shared_between=60, noise=0.001) for s in range(3)]
    only_a = {k: v for k, v in metric(4.0, 4.0).items() if int(k[:6]) <= 30}   # B's own cameras unmeasured
    monkeypatch.setattr(D, "metric_log_tri", lambda *a: (only_a, {}))
    for strict, n_groups in ((False, 1), (True, 2)):
        gp = G.GateParams(rule="evidence", min_obs=5, require_group_scale=strict)
        extra, _ = D.gate_inputs(gp, models, {"images": []}, tmp_path, db, tmp_path)
        out = G.apply_rigid_gate(models, gp, **extra)
        assert len(set(out["labels"].values())) == n_groups, strict
        if strict:
            why = [d["why"] for r in out["rounds"] for d in r["decisions"] if not d["kept"]]
            assert any("not measurable" in w for w in why), why


def _rot_z(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t), 0.0], [np.sin(t), np.cos(t), 0.0], [0.0, 0.0, 1.0]])


def test_only_links_the_solve_honours_are_evidence():
    """6839fb8f kf 92-98 was held by links the solve contradicts by 34-96 deg. With max_link_disagreement_deg a
    link counts only when its own rotation agrees with the solve's; the ladders stay, the contradicted cross
    links go, and island B is no longer one block with A."""
    links = ladder_links(cross=[(k, 30 + k, 50) for k in range(5)])
    models = [two_islands(s, shared_between=60, noise=0.001) for s in range(3)]
    ref = models[0]
    idx = ref.index()
    cross = {tuple(sorted((f"{k:06d}_k.jpg", f"{30 + k:06d}_k.jpg"))) for k in range(5)}
    level = metric(4.0, 4.0)
    for off_deg, n_groups in ((2.0, 1), (60.0, 2)):
        rots = {}
        for a, b in links:
            R = ref.R_cw[idx[b]] @ ref.R_cw[idx[a]].T
            rots[(a, b)] = (_rot_z(off_deg) @ R) if (a, b) in cross else R
        gp = G.GateParams(rule="evidence", min_obs=5, max_link_disagreement_deg=25.0)
        out = G.apply_rigid_gate(models, gp, links=links, metric_log=level, link_rotations=rots)
        assert len(set(out["labels"].values())) == n_groups, off_deg
        assert "set aside" in out["evidence"]["links_not_honoured"]
    with pytest.raises(ValueError, match="link_rotations"):
        G.apply_rigid_gate(models, G.GateParams(rule="evidence", min_obs=5, max_link_disagreement_deg=25.0),
                           links=links, metric_log=level)


def test_attach_groups_false_keeps_only_the_anchor():
    """The masks-unavailable fail-safe (manager 011): no group is attached to a component's anchor block."""
    links = ladder_links(cross=[(29, 30, 50), (28, 30, 50)])
    models = [two_islands(s, shared_between=60, noise=0.001) for s in range(3)]
    level = metric(4.0, 4.0)
    for attach, n_groups in ((True, 1), (False, 2)):
        gp = G.GateParams(rule="evidence", min_obs=5, attach_groups=attach)
        out = G.apply_rigid_gate(models, gp, links=links, metric_log=level)
        assert len(set(out["labels"].values())) == n_groups, attach
