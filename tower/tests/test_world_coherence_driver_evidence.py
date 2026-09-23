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
