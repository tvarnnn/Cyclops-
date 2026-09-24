"""A seeded final solve freezes its matching, and a re-finish maps it again (review V8 H2,
manager 019, part R of the H2 brief).

Matching is not reproducible even on one thread (RUN P3-PF, var step 2), and a final solve
matches INTO the walk's database, so every re-finish started from a different one. Pinned:

  * R1: a re-finish's masked solve takes every mask from the cache the re-finish carried
    back (`cache_hits == images`, the detector never runs);
  * R2: a seeded final solve writes `database.matching.json` beside the walk database; the
    next seeded final solve with the same images, key and database skips extraction and
    matching entirely (`solve.matching: "frozen"`); any change matches again, into the same
    database, and freezes the new state (`"matched"`, `matching_detail` says why);
  * the record goes back with the walk database in a re-finish;
  * an unseeded solve (the default) neither reads nor writes it: today's trace exactly.

pycolmap is the recording fake of `test_world_builder_solve_masks`.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_world_builder_solve_masks import (  # noqa: E402,F401 -- fixtures
    N,
    StubDetector,
    _B,
    _blob,
    _masked,
    _solve,
    _solution_json,
    _todays_trace,
    colmap,
    session,
    walked,
)
from tower.world_builder import global_solve as GS  # noqa: E402
from tower.world_builder import relocalizer  # noqa: E402

CALLS = ("extract_features", "match_sequential", "match_image_pairs")


def _calls(colmap):
    return [e[1] for e in colmap.log if e[0] == "call" and e[1] in CALLS]


def _record(s) -> dict:
    return json.loads((s.workspace.root / GS.FROZEN_MATCHING_FILENAME).read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def _no_links(monkeypatch):
    monkeypatch.setattr(relocalizer, "revisit_pairs", lambda session_dir: [])


# ---------------------------------------------------------------------------
# the default: nothing new


def test_an_unseeded_final_solve_neither_reads_nor_writes_a_record(walked, colmap):
    summary = _solve(walked, final=True)
    assert colmap.log == _todays_trace(walked)
    assert not (walked.workspace.root / GS.FROZEN_MATCHING_FILENAME).exists()
    assert not {"matching", "matching_detail", "database_digest"} & set(summary["solve"])
    # ... and a record left by a seeded solve is not read by an unseeded one
    _solve(walked, final=True, seed=4)
    colmap.log.clear()
    _solve(walked, final=True)
    assert colmap.log == _todays_trace(walked)


def test_a_background_solve_never_freezes(walked, colmap):
    summary = _solve(walked, final=False, seed=4)
    assert not (walked.workspace.root / GS.FROZEN_MATCHING_FILENAME).exists()
    assert "matching" not in summary["solve"]


# ---------------------------------------------------------------------------
# R2: freeze, then map what was frozen


@pytest.mark.parametrize("masked", [False, True])
def test_a_seeded_final_solve_freezes_and_the_next_maps_it_without_matching(walked, colmap, masked):
    run = (lambda: _masked(walked, StubDetector(), seed=7)) if masked else \
        (lambda: _solve(walked, final=True, seed=7))
    first = run()
    assert _calls(colmap) == ["extract_features", "match_sequential"]
    assert first["solve"]["matching"] == GS.MATCHING_MATCHED
    assert first["solve"]["matching_detail"] == ("no frozen matching record; this solve's "
                                                 "matching is frozen now")
    record = _record(walked)
    assert record["record"] == GS.FROZEN_MATCHING_RECORD and record["database"] == "database.db"
    assert sorted(record["images"]) == [f"{i:08d}.jpg" for i in range(N)]
    assert record["key"]["verification_seed"] == 7 and record["key"]["pycolmap"] == "fake"
    assert record["database_digest"] == GS.database_digest(walked.workspace.database_path)

    colmap.log.clear()
    second = run()
    assert _calls(colmap) == [], "a frozen matching is neither extracted nor matched again"
    assert colmap.calls("global_mapping"), "... and it is mapped"
    assert second["solve"]["matching"] == GS.MATCHING_FROZEN
    assert second["solve"]["matching_detail"] is None
    assert second["solve"]["database_digest"] == first["solve"]["database_digest"] is not None
    assert second["solve"]["verified_pairs"] == first["solve"]["verified_pairs"]
    assert _solution_json(walked)["solve"]["matching"] == GS.MATCHING_FROZEN
    if masked:
        # the mask filter still runs (on the frozen walk database) and is re-verified
        assert colmap.calls("verify_matches")
        assert second["solve"]["masking"] == "walk-database-filtered"


def _change_an_image(s):
    path = s.workspace.images_dir / "00000002.jpg"
    ok, buf = cv2.imencode(".jpg", np.full((10, 10, 3), 7, np.uint8))
    path.write_bytes(buf.tobytes())


def _change_the_database(s):
    con = sqlite3.connect(str(s.workspace.database_path))
    con.execute("insert into matches values (?, ?, ?, ?)", (1 * _B + 4, *_blob([[2, 2]])))
    con.commit()
    con.close()


CHANGES = {
    "seed": ({"seed": 8}, None, "matching parameters changed (verification_seed)"),
    "overlap": ({"overlap": 5}, None, "matching parameters changed (sequential_overlap)"),
    "image": ({}, _change_an_image, "1 solver images changed"),
    "database": ({}, _change_the_database, "the database changed since its matching was frozen"),
}


@pytest.mark.parametrize("change", sorted(CHANGES))
def test_any_change_matches_again_and_freezes_the_new_state(walked, colmap, change):
    kwargs, mutate, why = CHANGES[change]
    _solve(walked, final=True, seed=7)
    if mutate is not None:
        mutate(walked)
    colmap.log.clear()
    summary = _solve(walked, final=True, **{"seed": 7, **kwargs})
    assert _calls(colmap) == ["extract_features", "match_sequential"]
    assert summary["solve"]["matching"] == GS.MATCHING_MATCHED
    assert why in summary["solve"]["matching_detail"]
    assert summary["solve"]["matching_detail"].endswith("this solve's matching is frozen now")
    # ... and the next identical solve maps the new state without matching
    colmap.log.clear()
    again = _solve(walked, final=True, **{"seed": 7, **kwargs})
    assert _calls(colmap) == [] and again["solve"]["matching"] == GS.MATCHING_FROZEN


@pytest.mark.parametrize("masked", [False, True])
def test_a_walk_database_written_after_the_freeze_check_is_not_called_frozen(walked, colmap,
                                                                             monkeypatch, masked):
    """Review V9 Q1: the freeze check reads the walk database before extraction; the mask
    filter copies it (unmasked: the mapper reads it) later. A hand-run `world_solve.py
    --final` takes no writer lock and can write it in between. Such a solve is not
    `frozen`: what it mapped is not the frozen matching, and it says so."""
    run = (lambda: _masked(walked, StubDetector(), seed=7)) if masked else \
        (lambda: _solve(walked, final=True, seed=7))
    run()
    checked = GS._frozen_matching_refusal

    def check_then_another_solve_writes(*a, **kw):
        out = checked(*a, **kw)
        _change_the_database(walked)          # the other solve, inside the window
        return out

    monkeypatch.setattr(GS, "_frozen_matching_refusal", check_then_another_solve_writes)
    colmap.log.clear()
    summary = run()
    assert _calls(colmap) == [], "nothing was matched: the check had passed"
    assert summary["solve"]["matching"] == GS.MATCHING_MATCHED
    assert summary["solve"]["matching_detail"] == GS.FROZEN_REFUSAL_CHANGED_UNDER_THE_SOLVE


def test_a_new_pycolmap_matches_again(walked, colmap):
    _solve(walked, final=True, seed=7)
    colmap.__version__ = "fake-2"
    colmap.log.clear()
    summary = _solve(walked, final=True, seed=7)
    assert _calls(colmap) == ["extract_features", "match_sequential"]
    assert "(pycolmap)" in summary["solve"]["matching_detail"]


def test_revisit_links_leave_the_walk_databases_frozen_matching_alone(walked, colmap, monkeypatch):
    """Review V9 M-10: the imported links never enter the walk database, so they are not in
    its key. A frozen walk database STAYS frozen when the relocalizer lists links; they are
    matched into the solve's own filtered copy, on every solve, and never into the walk's."""
    from tower.world_builder import coherence_publish as CP

    monkeypatch.setattr(CP, "gate_final_solution", lambda store, w, s, solution, **kw: CP.GateResult(
        solution=solution, components=None, record={"state": CP.GATE_STATE_APPLIED, "seconds": 0.0}))
    _masked(walked, StubDetector(), seed=7, gate=True)
    assert "revisit_pairs" not in _record(walked)["key"] and "revisit_pairs" not in _record(walked)
    monkeypatch.setattr(relocalizer, "revisit_pairs",
                        lambda session_dir: [("00000000.jpg", "00000003.jpg")])
    for _ in range(2):
        colmap.log.clear()
        summary = _masked(walked, StubDetector(), seed=7, gate=True)
        assert _calls(colmap) == ["match_image_pairs"], "frozen: only the import, into the copy"
        assert summary["solve"]["matching"] == GS.MATCHING_FROZEN
        assert colmap.calls("match_image_pairs")[0][2] == summary["solve"]["database"] != "database.db"
        assert summary["solve"]["revisit_pairs"]["listed"] == 1
        # ... on one thread when seeded: the copy is made again, import included, every solve
        assert colmap.calls("match_image_pairs")[0][3] == (
            "matching_options", "pairing_options", "verification_options")


@pytest.mark.parametrize("imported", [False, True])
def test_a_record_frozen_before_v9_m10(walked, colmap, imported):
    """A record written at 6d4b567 carries `revisit_pairs` in its key. Without an import the
    walk database it describes is exactly what today's key describes, and it stays frozen;
    with one, the walk database holds imported pairs and the freeze is refused, with why."""
    _solve(walked, final=True, seed=7)
    path = walked.workspace.root / GS.FROZEN_MATCHING_FILENAME
    record = _record(walked)
    record["key"]["revisit_pairs"] = {"imported": imported, "count": int(imported),
                                      "sha1": "0" * 40 if imported else None,
                                      "min_inliers": 50 if imported else None}
    record["revisit_pairs"] = [["00000000.jpg", "00000003.jpg"]] if imported else []
    path.write_text(json.dumps(record), encoding="utf-8")
    colmap.log.clear()
    summary = _solve(walked, final=True, seed=7)
    if imported:
        assert summary["solve"]["matching"] == GS.MATCHING_MATCHED
        assert summary["solve"]["matching_detail"].startswith(GS.FROZEN_REFUSAL_LEGACY_IMPORT)
    else:
        assert _calls(colmap) == [] and summary["solve"]["matching"] == GS.MATCHING_FROZEN


def test_the_verification_seed_stays_in_the_key_and_says_why():
    """Review V9 LOW: a different two-view RANSAC seed verifies differently, so a database
    matched under one seed is not frozen for another -- kept, and documented."""
    key = GS.matching_key(camera_params="1,1,1,1", overlap=20, loop_detection=True, seed=3,
                          pycolmap_version="x")
    assert key["verification_seed"] == 3 and "revisit_pairs" not in key
    assert "verifies the same" in GS.matching_key.__doc__


# ---------------------------------------------------------------------------
# an unreadable solver image refuses the freeze, not the solve (review V9, LOW; RV9-B P3)


@pytest.mark.parametrize("masked", [False, True])
def test_an_unreadable_solver_image_refuses_the_freeze_and_the_solve_goes_on(walked, colmap,
                                                                             monkeypatch, masked):
    from tower.world_builder import solve_masks as SM

    real = SM.file_sha1
    locked = walked.workspace.images_dir / "00000002.jpg"

    def file_sha1(path):
        if Path(path) == locked:
            raise PermissionError(13, "The process cannot access the file", str(path))
        return real(path)

    monkeypatch.setattr(SM, "file_sha1", file_sha1)
    if masked:
        summary = _masked(walked, StubDetector(), seed=7)
    else:
        summary = _solve(walked, final=True, seed=7)
    assert summary["solved"] is True
    assert summary["solve"]["matching"] == GS.MATCHING_MATCHED
    detail = summary["solve"]["matching_detail"]
    assert "00000002.jpg could not be read to freeze" in detail and "PermissionError" in detail
    assert detail.endswith("this solve's matching is not frozen")
    assert not (walked.workspace.root / GS.FROZEN_MATCHING_FILENAME).exists()
    assert _calls(colmap)[:2] == ["extract_features", "match_sequential"]


def test_an_unreadable_database_is_matched_and_not_frozen(session, colmap):
    """The fake's database is an empty file: nothing to freeze, and the solve goes on."""
    summary = _solve(session, final=True, seed=7)
    assert summary["solved"] is True
    assert summary["solve"]["matching"] == GS.MATCHING_MATCHED
    assert "could not be read to freeze" in summary["solve"]["matching_detail"]
    assert summary["solve"]["database_digest"] is None
    assert not (session.workspace.root / GS.FROZEN_MATCHING_FILENAME).exists()


def test_a_re_extracted_masked_database_is_never_frozen(session, colmap):
    summary = _masked(session, StubDetector(), seed=7)
    assert summary["solve"]["masking"] == "re-extracted"
    assert summary["solve"]["matching"] == GS.MATCHING_MATCHED
    assert "re-extracted" in summary["solve"]["matching_detail"]
    assert not (session.workspace.root / GS.FROZEN_MATCHING_FILENAME).exists()


# ---------------------------------------------------------------------------
# the digest


def test_the_digest_is_of_content_not_bytes(tmp_path):
    from tests.test_world_builder_solve_masks import _colmap_walk_database

    a, b = tmp_path / "a.db", tmp_path / "b.db"
    _colmap_walk_database(a)
    _colmap_walk_database(b)
    con = sqlite3.connect(str(b))
    con.execute("create table descriptors (image_id integer primary key, data blob)")
    con.execute("insert into descriptors values (1, x'00ff')")    # not read by the mapper
    con.commit()
    con.execute("vacuum")
    con.close()
    da, db = GS.database_digest(a), GS.database_digest(b)
    assert da == db and da["verified_pairs"] == 3
    con = sqlite3.connect(str(b))
    con.execute("update two_view_geometries set rows = rows where pair_id = ?", (3 * _B + 4,))
    con.execute("update keypoints set data = ? where image_id = 1", (b"\x00" * 96,))
    con.commit()
    con.close()
    assert GS.database_digest(b)["content"] != da["content"]
    assert GS.database_digest(b)["verified"] == da["verified"], "the verified pair set is unchanged"
    assert GS.database_digest(tmp_path / "absent.db") is None
    (tmp_path / "empty.db").write_bytes(b"")
    assert GS.database_digest(tmp_path / "empty.db") is None


def _geometry_db(path, F, t, q=(1.0, 0.0, 0.0, 0.0)):
    con = sqlite3.connect(str(path))
    con.execute("create table images (image_id integer primary key, name text)")
    con.execute("create table two_view_geometries (pair_id integer primary key, rows integer, "
                "cols integer, data blob, config integer, F blob, E blob, H blob, qvec blob, "
                "tvec blob)")
    con.executemany("insert into images values (?, ?)", [(1, "a.jpg"), (2, "b.jpg")])
    F = np.asarray(F, np.float64)
    con.execute("insert into two_view_geometries values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (1 * _B + 2, 1, 2, np.zeros(2, np.uint32).tobytes(), 2, F.tobytes(), F.tobytes(),
                 F.tobytes(), np.asarray(q, np.float64).tobytes(),
                 np.asarray(t, np.float64).tobytes()))
    con.commit()
    con.close()
    return GS.database_digest(path)


def test_the_stated_digest_is_up_to_scale_with_nothing_appended(tmp_path):
    """Review V9, LOW (RV9-B D1): F, E, H, qvec and tvec are defined up to scale, as
    `DATABASE_DIGEST_RULE` says, so a pure rescale of every matrix leaves the stated digest
    unchanged -- it appended the magnitude, so F and 2F digested differently. F, E, H and
    qvec are also up to sign; tvec keeps its sign. `content` stays exact."""
    rng = np.random.default_rng(0)
    F = rng.normal(size=9) * np.array([1e-7, 1e-6, 1e-3, 1e-6, 1e-7, 1e-3, 1e-3, 1e-3, 1.0])
    t = np.array([0.2, -0.1, 0.97])
    q = np.array([0.9, 0.1, -0.3, 0.2])
    base = _geometry_db(tmp_path / "a.db", F, t, q)
    for i, (Fs, ts, qs) in enumerate([(2 * F, 2 * t, q), (1e-3 * F, 7.5 * t, q),
                                      (-F, t, -q), (-3 * F, 0.5 * t, -q)]):
        other = _geometry_db(tmp_path / f"b{i}.db", Fs, ts, qs)
        assert other["stated"] == base["stated"], i
        assert other["content"] != base["content"], i
    flipped = _geometry_db(tmp_path / "c.db", F, -t, q)
    assert flipped["stated"] != base["stated"], "tvec's sign is kept"
    for up_to_sign in (True, False):
        assert GS._stated_bytes((5 * F).tobytes(), up_to_sign=up_to_sign) == \
            GS._stated_bytes(F.tobytes(), up_to_sign=up_to_sign)
        assert len(GS._stated_bytes(F.tobytes(), up_to_sign=up_to_sign)) == F.nbytes


def test_an_opposite_sign_tie_for_the_pivot_does_not_flip_the_digest():
    """RV9-B D3: `[t]x` holds +1 and -1; one ulp on either used to move the pivot to the other
    entry and flip the sign of the whole normalised matrix."""
    E = np.array([0, 0, 0, 0, 0, -1.0, 0, 1.0, 0])
    for k, toward in ((5, -2.0), (7, 2.0)):
        E2 = E.copy()
        E2[k] = np.nextafter(E[k], toward)
        assert GS._stated_bytes(E.tobytes(), up_to_sign=True) == \
            GS._stated_bytes(E2.tobytes(), up_to_sign=True), k
    q = np.array([np.sqrt(0.5), -np.sqrt(0.5), 0, 0])
    q2 = q.copy()
    q2[1] = np.nextafter(q[1], -1.0)
    assert GS._stated_bytes(q.tobytes(), up_to_sign=True) == \
        GS._stated_bytes(q2.tobytes(), up_to_sign=True)


# ---------------------------------------------------------------------------
# R1 and the re-finish: the cache and the record go back with the walk database


def _set_aside(s, stamp):
    from scripts import world_refinish as wr

    return wr.set_aside(s.store, s.world_id, s.session_id, stamp)


@pytest.fixture
def no_attempt_ledger(monkeypatch):
    from scripts import world_refinish as wr

    monkeypatch.setattr(wr, "_restart_attempts", lambda *a, **k: {})


def test_a_refinish_maps_the_frozen_matching_and_hits_every_cached_mask(walked, colmap,
                                                                       no_attempt_ledger):
    stub = StubDetector()
    first = _masked(walked, stub, seed=0)
    assert first["transients"]["computed"] == N
    assert first["solve"]["matching"] == GS.MATCHING_MATCHED
    ledger = _set_aside(walked, "r1")
    assert {c["name"] for c in ledger["copied_back"]} >= {"database.db", "transients",
                                                          GS.FROZEN_MATCHING_FILENAME}
    calls_before = len(stub.calls)
    colmap.log.clear()
    second = _masked(walked, stub, seed=0)
    # R1: every mask from the carried-back cache; the detector never ran
    assert second["transients"]["cache_hits"] == second["transients"]["images"] == N
    assert second["transients"]["computed"] == 0 and len(stub.calls) == calls_before
    # R2: the frozen matching went back with the walk database
    assert second["solve"]["matching"] == GS.MATCHING_FROZEN
    assert _calls(colmap) == []
    assert second["solve"]["database_digest"] == first["solve"]["database_digest"]


def test_the_refinish_copy_back_names_the_record():
    from scripts import world_refinish as wr

    assert GS.FROZEN_MATCHING_FILENAME in wr.SOLVE_COPY_BACK
