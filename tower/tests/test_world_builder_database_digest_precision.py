"""`solve.database_digest` is taken at a STATED precision for the two-view geometry
matrices (review V8 H2, R4; the lead's decision on H2's OPEN "F noise").

Two same-seed re-finishes of one walk (6839fb8f copy, P3-H2 runs B and C) mapped
identically and gated identically, yet their filtered databases differed in 3 of 16,793
pairs' F by at most 1.3e-14 -- last-bit noise of the mask filter's re-verification on
planar/panoramic pairs, where F is not used. The exact content digest (which the frozen
matching compares on the WALK database) must stay exact; the digest the solve reports
must not count that noise, and must still change for anything that can move the map.
"""

from __future__ import annotations

import sqlite3

import numpy as np

from tower.world_builder import global_solve as GS


def _db(path, *, F_delta=0.0, rows=40, data_seed=0, F_sign=1.0, t_sign=1.0):
    con = sqlite3.connect(path)
    con.execute("create table images (image_id integer primary key, name text)")
    con.execute(
        "create table two_view_geometries (pair_id integer primary key not null, rows integer not null, "
        "cols integer not null, data blob, config integer not null, F blob, E blob, H blob, qvec blob, "
        "tvec blob, camera1 blob, camera2 blob)")
    con.executemany("insert into images values (?, ?)", [(1, "a.jpg"), (2, "b.jpg"), (3, "c.jpg")])
    rng = np.random.default_rng(7)
    for pid, (i, j) in ((2147483649 * 1 + 1, (1, 2)), (2147483647 * 2 + 3, (2, 3))):
        F = rng.normal(size=(3, 3)) * np.array([[1e-7, 1e-6, 1e-3], [1e-6, 1e-7, 1e-3], [1e-3, 1e-3, 1.0]])
        F = (F + F_delta) * F_sign
        data = np.random.default_rng(data_seed + pid % 97).integers(0, 4000, size=(rows, 2), dtype=np.uint32)
        con.execute(
            "insert into two_view_geometries values (?, ?, 2, ?, 3, ?, ?, ?, ?, ?, NULL, NULL)",
            (pid, rows, data.tobytes(), F.tobytes(), (F * 2).tobytes(), np.eye(3).tobytes(),
             (np.array([1.0, 0, 0, 0]) * F_sign).tobytes(), (np.array([0.2, 0, 1]) * t_sign).tobytes()))
    con.commit()
    con.close()
    return path


def test_last_bit_noise_in_F_does_not_change_the_stated_digest(tmp_path):
    a = GS.database_digest(_db(tmp_path / "a.db"))
    b = GS.database_digest(_db(tmp_path / "b.db", F_delta=1.3e-14))
    assert a["content"] != b["content"], "the exact digest must still see the noise"
    assert a["stated"] == b["stated"]


def test_a_real_change_in_F_changes_the_stated_digest(tmp_path):
    a = GS.database_digest(_db(tmp_path / "a.db"))
    b = GS.database_digest(_db(tmp_path / "b.db", F_delta=1e-6))
    assert a["stated"] != b["stated"]


def test_a_changed_inlier_set_changes_the_stated_digest(tmp_path):
    a = GS.database_digest(_db(tmp_path / "a.db"))
    b = GS.database_digest(_db(tmp_path / "b.db", data_seed=5))
    c = GS.database_digest(_db(tmp_path / "c.db", rows=39))
    assert a["stated"] != b["stated"]
    assert a["stated"] != c["stated"]
    assert a["verified"] != c["verified"]


def test_the_rule_is_named():
    assert "10" in GS.DATABASE_DIGEST_RULE and "F" in GS.DATABASE_DIGEST_RULE


def test_F_and_the_quaternion_are_digested_up_to_sign(tmp_path):
    # Measured on 6839fb8f (P3-H2 runs B and C): one planar pair's F came back as -F.
    a = GS.database_digest(_db(tmp_path / "a.db"))
    b = GS.database_digest(_db(tmp_path / "b.db", F_sign=-1.0))
    assert a["content"] != b["content"]
    assert a["stated"] == b["stated"]


def test_the_translation_keeps_its_sign(tmp_path):
    a = GS.database_digest(_db(tmp_path / "a.db"))
    b = GS.database_digest(_db(tmp_path / "b.db", t_sign=-1.0))
    assert a["stated"] != b["stated"]
