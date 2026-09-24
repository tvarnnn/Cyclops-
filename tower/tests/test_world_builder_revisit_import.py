"""The relocalizer's revisit links in the final solve (review V8, M4; `global_solve`).

Pinned here:
  * only a MASKED and GATED final solve imports them; any other says why, and a solve
    with no links at all records exactly what it always did;
  * they are matched only into the database the solve MAPS, never into the walk's own
    (review V9 M-10), so no later solve maps an imported pair it did not import;
  * an imported pair counts only at the relocalizer's own floor
    (`relocalizer.REVISIT_MIN_INLIERS`), judged in the mapped database after the masks,
    and a pair THE IMPORT CREATED below it is removed from that database, its matches and
    its geometry (review V9 M-9); a listed pair the database already held (sequential,
    loop detection, an earlier solve) is COLMAP's and stays as it is.

pycolmap is the recording fake of `test_world_builder_solve_masks`; the databases are
real SQLite in COLMAP's schema.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from tests.test_world_builder_solve_masks import (  # noqa: F401 -- fixtures
    StubDetector,
    _B,
    _KEYPOINTS,
    _blob,
    _colmap_walk_database,
    _masked,
    _solve,
    _solution_json,
    colmap,
    session,
    walked,
)
from tower.world_builder import global_solve
from tower.world_builder import relocalizer

LINK = ("00000000.jpg", "00000003.jpg")     # images 1 and 4 of the fixture's database


@pytest.fixture(autouse=True)
def _stub_gate(monkeypatch):
    from tower.world_builder import coherence_publish as CP

    def fake(store, world_id, session_id, solution, *, database_path, keyframes, **kw):
        return CP.GateResult(solution=solution, components=None,
                             record={"state": CP.GATE_STATE_APPLIED, "seconds": 0.0})

    monkeypatch.setattr(CP, "gate_final_solution", fake)


def _links(monkeypatch, pairs):
    monkeypatch.setattr(relocalizer, "revisit_pairs", lambda session_dir: list(pairs))


def _add_pair(db, a_id, b_id, inliers, *, config=2):
    """A matched and verified pair on the fixture's CLEAN keypoints (2 and 3), so the
    mask filter leaves it exactly as it is."""
    rows = [[2 + (i % 2), 2 + (i % 2)] for i in range(inliers)]
    con = sqlite3.connect(str(db))
    con.execute("insert into matches values (?, ?, ?, ?)", (a_id * _B + b_id, *_blob(rows)))
    con.execute("insert into two_view_geometries values (?, ?, ?, ?, ?)",
                (a_id * _B + b_id, *_blob(rows), config))
    con.commit()
    con.close()


def _pair_rows(db, pid):
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        m = con.execute("select rows from matches where pair_id = ?", (pid,)).fetchone()
        g = con.execute("select rows from two_view_geometries where pair_id = ?", (pid,)).fetchone()
        return (m[0] if m else None), (g[0] if g else None)
    finally:
        con.close()


# ---------------------------------------------------------------------------
# which solves import them


def test_the_floor_is_the_relocalizers_own_constant():
    assert global_solve.revisit_floor() == relocalizer.REVISIT_MIN_INLIERS
    assert relocalizer.REVISIT_MIN_INLIERS == relocalizer.AcceptanceParams().triangle_min_link_inliers


@pytest.mark.parametrize("masked,gated", [(False, False), (False, True), (True, False), (True, True)])
def test_no_links_is_todays_record_whatever_the_settings(walked, colmap, monkeypatch, masked, gated):
    _links(monkeypatch, [])
    if masked:
        summary = _masked(walked, StubDetector(), gate=gated, seed=3)
    else:
        summary = _solve(walked, final=True, gate=gated)
    assert summary["solve"]["revisit_pairs"] == {"listed": 0, "verified": 0, "detail": None}
    assert colmap.calls("match_image_pairs") == []


@pytest.mark.parametrize("masked,gated", [(False, False), (False, True), (True, False)])
def test_a_solve_that_is_not_masked_and_gated_imports_none_and_says_why(walked, colmap,
                                                                        monkeypatch, masked, gated):
    _links(monkeypatch, [LINK])
    before = walked.workspace.database_path.read_bytes()
    if masked:
        summary = _masked(walked, StubDetector(), gate=gated)
    else:
        summary = _solve(walked, final=True, gate=gated)
    assert colmap.calls("match_image_pairs") == []
    record = summary["solve"]["revisit_pairs"]
    assert record["imported"] is False and record["listed"] == 1 and record["verified"] == 0
    assert record["detail"] == (global_solve.REVISIT_IMPORT_UNGATED if masked
                                else global_solve.REVISIT_IMPORT_UNMASKED)
    assert "M4" in record["detail"]
    assert walked.workspace.database_path.read_bytes() == before


def test_a_background_solve_never_reads_them(walked, colmap, monkeypatch):
    def must_not_read(_d):
        raise AssertionError("a background solve read the revisit links")

    monkeypatch.setattr(relocalizer, "revisit_pairs", must_not_read)
    summary = _solve(walked, final=False, masks=True, transient_backend_factory=StubDetector(),
                     mask_device_probe=lambda: None, gate=True)
    assert summary["solve"]["revisit_pairs"] == {"listed": 0, "verified": 0, "detail": None}


def test_masks_that_are_not_applied_are_not_masked(walked, colmap, monkeypatch):
    """A detector that cannot run leaves an UNMASKED solve (transients `unavailable`):
    the links are not imported even with the gate on."""
    _links(monkeypatch, [LINK])
    summary = _masked(walked, StubDetector(reason="no CUDA device"), gate=True)
    assert summary["transients"]["state"] != "applied"
    assert summary["solve"]["revisit_pairs"]["detail"] == global_solve.REVISIT_IMPORT_UNMASKED
    assert colmap.calls("match_image_pairs") == []


# ---------------------------------------------------------------------------
# where they are matched (review V9 M-10), and the floor on what the import made (M-9)


def _colmap_writes_listed_pairs(colmap, inliers, rows=None):
    """The fake pycolmap, made to behave like COLMAP: `match_image_pairs` WRITES every listed
    pair into the database it is given, matched and verified at `inliers` (on the fixture's
    clean keypoints 2 and 3, unless `rows` says otherwise) -- and, like pycolmap 4.2.0
    (`V9/frozen/probe_skip.py`), leaves a pair that is already matched and verified as it
    is. Returns the databases it wrote."""
    original = colmap.match_image_pairs
    wrote = []

    def match_image_pairs(database_path, **kwargs):
        original(database_path, **kwargs)
        listed = kwargs["pairing_options"]._values["match_list_path"]
        with open(listed, encoding="utf-8") as handle:
            names = handle.read().split()
        con = sqlite3.connect(str(database_path))
        ids = dict(con.execute("select name, image_id from images"))
        for a, b in zip(names[::2], names[1::2]):
            lo, hi = sorted((ids[a], ids[b]))
            if (con.execute("select 1 from matches where pair_id = ?", (lo * _B + hi,)).fetchone()
                    and con.execute("select 1 from two_view_geometries where pair_id = ?",
                                    (lo * _B + hi,)).fetchone()):
                continue
            data = rows if rows is not None else [[2 + (i % 2), 2 + (i % 2)] for i in range(inliers)]
            con.execute("insert or replace into matches values (?, ?, ?, ?)", (lo * _B + hi, *_blob(data)))
            con.execute("insert or replace into two_view_geometries values (?, ?, ?, ?, 2)",
                        (lo * _B + hi, *_blob(data)))
        con.commit()
        con.close()
        wrote.append(Path(database_path).name)

    colmap.match_image_pairs = match_image_pairs
    return wrote


@pytest.mark.parametrize("inliers,kept", [(49, False), (50, True), (400, True)])
def test_a_pair_the_import_creates_below_the_floor_is_removed_from_the_mapped_database(
        walked, colmap, monkeypatch, inliers, kept):
    """The walk database does NOT hold the revisit pair (1, 4); the import creates it at
    `inliers`, in the database the solve maps -- never in the walk's own (M-10). At 49 it is
    REMOVED from the mapped copy, matches and geometry; at the floor (inclusive) it counts as
    verified. The walk's own database is byte for byte what it was."""
    before = walked.workspace.database_path.read_bytes()
    wrote = _colmap_writes_listed_pairs(colmap, inliers)
    _links(monkeypatch, [LINK])
    summary = _masked(walked, StubDetector(), gate=True, overlap=1)
    mapped = walked.workspace.root / summary["solve"]["database"]
    (_c, _n, database, _kw, listed), = colmap.calls("match_image_pairs")
    assert listed == "00000000.jpg 00000003.jpg\n"
    assert database == mapped.name != "database.db" and wrote == [mapped.name]
    record = summary["solve"]["revisit_pairs"]
    assert record["imported"] is True and record["min_inliers"] == relocalizer.REVISIT_MIN_INLIERS
    assert record["created_by_import"] == 1
    assert record["verified"] == (1 if kept else 0)
    assert record["removed_below_floor"] == (0 if kept else 1)
    assert record["kept_existing"] == 0
    assert _pair_rows(mapped, 1 * _B + 4) == ((inliers, inliers) if kept else (None, None))
    assert walked.workspace.database_path.read_bytes() == before
    # what the mapper was given is the database the floor was applied to
    assert colmap.calls("global_mapping")[0][2] == mapped.name


def test_the_floor_is_judged_after_the_masks(walked, colmap, monkeypatch):
    """The import creates (1, 4) with 60 inliers, one of them on a masked keypoint (0): in
    the filter's copy it was matched on EVERY keypoint, so that match is removed, the pair
    loses its geometry and is re-verified (faked here: nothing comes back) -- and under the
    floor it is removed, although COLMAP gave it 60."""
    _colmap_writes_listed_pairs(colmap, 60, rows=[[0, 0]] + [[2, 2]] * 59)
    _links(monkeypatch, [LINK])
    summary = _masked(walked, StubDetector(), gate=True, overlap=1)
    record = summary["solve"]["revisit_pairs"]
    assert record["created_by_import"] == 1 and record["reverified_after_masks"] == 1
    assert record["verified"] == 0 and record["removed_below_floor"] == 1
    mapped = walked.workspace.root / summary["solve"]["database"]
    reverify = [c for c in colmap.calls("verify_matches")
                if c[3] == "00000000.jpg 00000003.jpg\n"]
    assert len(reverify) == 1 and reverify[0][2] == mapped.name
    assert _pair_rows(mapped, 1 * _B + 4) == (None, None)


@pytest.mark.parametrize("relocalizer_lists_it", [False, True])
def test_a_listed_pair_the_database_already_holds_is_not_the_imports(walked, colmap, monkeypatch,
                                                                     relocalizer_lists_it):
    """Review V9 M-9 (RV9-B P1). (1, 4) is in the walk database at 30 verified inliers before
    any revisit matching -- what loop detection leaves (overlap 1 makes it no sequential pair,
    as a loop pair is on a real walk). It is COLMAP's pair, not the import's: it is not matched
    again, and it reaches the mapper at 30 whether or not the relocalizer lists it. It used to
    be deleted when the relocalizer listed it -- switching the relocalizer on removed
    evidence."""
    _add_pair(walked.workspace.database_path, 1, 4, 30)
    _colmap_writes_listed_pairs(colmap, 400)
    _links(monkeypatch, [LINK] if relocalizer_lists_it else [])
    summary = _masked(walked, StubDetector(), gate=True, seed=3, overlap=1)
    mapped = walked.workspace.root / summary["solve"]["database"]
    assert _pair_rows(mapped, 1 * _B + 4) == (30, 30)
    assert colmap.calls("match_image_pairs") == []
    record = summary["solve"]["revisit_pairs"]
    if relocalizer_lists_it:
        assert record["created_by_import"] == 0 and record["removed_below_floor"] == 0
        assert record["kept_existing"] == 1 and record["verified"] == 0
    else:
        assert record == {"listed": 0, "verified": 0, "detail": None}


@pytest.mark.parametrize("second", ["unmasked", "ungated"])
def test_an_imported_pair_never_enters_the_walk_database(walked, colmap, monkeypatch, second):
    """Review V9 M-10 (RV9-B P2). A masked, gated final solve imports (1, 4) at 20 inliers
    (under the floor, so it is removed from its copy). The walk database never holds it, so
    a later unmasked or ungated solve -- which records `imported: false` -- does not map it.
    It used to: the import matched into the walk database before the filter copied it."""
    _colmap_writes_listed_pairs(colmap, 20)
    _links(monkeypatch, [LINK])
    first = _masked(walked, StubDetector(), gate=True, seed=3, overlap=1)
    assert first["solve"]["revisit_pairs"]["removed_below_floor"] == 1
    assert _pair_rows(walked.workspace.database_path, 1 * _B + 4) == (None, None)

    colmap.log.clear()
    if second == "unmasked":
        summary = _solve(walked, final=True, masks=False, gate=True, seed=3, overlap=1)
    else:
        summary = _masked(walked, StubDetector(), gate=False, seed=3, overlap=1)
    assert summary["solve"]["revisit_pairs"]["imported"] is False
    mapped = walked.workspace.root / summary["solve"]["database"]
    assert colmap.calls("global_mapping")[0][2] == mapped.name
    assert _pair_rows(mapped, 1 * _B + 4) == (None, None)


def _extraction_makes_a_database(colmap):
    """The fake's `extract_features` only touches the file; make it leave a COLMAP database
    (the fixture's schema and keypoints) when there is none, as COLMAP would."""
    original = colmap.extract_features

    def extract_features(database_path, image_path, **kwargs):
        path = Path(database_path)
        fresh = not path.exists() or path.stat().st_size == 0
        original(database_path, image_path, **kwargs)
        if fresh:
            con = sqlite3.connect(str(path))
            con.executescript(
                "create table images (image_id integer primary key, name text, camera_id integer);"
                "create table keypoints (image_id integer primary key, rows integer, cols integer,"
                " data blob);"
                "create table matches (pair_id integer primary key, rows integer, cols integer,"
                " data blob);"
                "create table two_view_geometries (pair_id integer primary key, rows integer,"
                " cols integer, data blob, config integer);")
            for i, name in enumerate(kwargs["image_names"]):
                con.execute("insert into images values (?, ?, 1)", (i + 1, name))
                con.execute("insert into keypoints values (?, 4, 6, ?)", (i + 1, _KEYPOINTS.tobytes()))
            con.commit()
            con.close()

    colmap.extract_features = extract_features


def test_a_re_extracted_database_seeded_from_an_importing_solve_maps_no_import(session, colmap,
                                                                               monkeypatch):
    """With no walk database the masks go to extraction, into a database of the solve's own,
    and the next masked solve is SEEDED with a copy of it (`solve_masks.masked_database`). A
    gated solve imports (1, 4) at 60 and keeps it; a later UNGATED solve seeded from that
    database imports nothing -- and maps nothing imported: the pairs the import made are
    listed in the database itself and taken out first (M-10)."""
    _extraction_makes_a_database(colmap)
    _colmap_writes_listed_pairs(colmap, 60)
    _links(monkeypatch, [LINK])
    first = _masked(session, StubDetector(), gate=True, overlap=1)
    assert first["solve"]["masking"] == "re-extracted"
    first_db = session.workspace.root / first["solve"]["database"]
    assert _pair_rows(first_db, 1 * _B + 4) == (60, 60)

    second = _masked(session, StubDetector(), gate=False, overlap=1)
    assert second["transients"]["database_reused"] is True
    assert second["solve"]["revisit_pairs"]["imported"] is False
    mapped = session.workspace.root / second["solve"]["database"]
    assert mapped != first_db and _pair_rows(mapped, 1 * _B + 4) == (None, None)
    assert second["solve"]["revisit_imports_cleared"] == 1
    record = first["solve"]["revisit_pairs"]
    assert record["created_by_import"] == 1 and record["verified"] == 1
    # ... and a gated one seeded from THAT imports it again, under its own floor
    third = _masked(session, StubDetector(), gate=True, overlap=1)
    assert third["solve"]["revisit_pairs"]["created_by_import"] == 1
    assert _pair_rows(session.workspace.root / third["solve"]["database"], 1 * _B + 4) == (60, 60)


def test_the_floor_removes_only_what_the_import_created(tmp_path):
    db = tmp_path / "database.masked.p1.00000000.db"
    _colmap_walk_database(db)
    _add_pair(db, 1, 4, 10)
    out = global_solve._apply_revisit_floor(db, [LINK], floor=50, created=set())
    assert out == {"verified": 0, "removed_below_floor": 0, "kept_existing": 1}
    assert _pair_rows(db, 1 * _B + 4) == (10, 10)
    out = global_solve._apply_revisit_floor(db, [LINK], floor=50, created={1 * _B + 4})
    assert out == {"verified": 0, "removed_below_floor": 1, "kept_existing": 0}
    assert _pair_rows(db, 1 * _B + 4) == (None, None)


def test_an_unreadable_mapped_database_is_recorded_not_fatal(tmp_path):
    db = tmp_path / "database.masked.p1.00000000.db"
    db.write_bytes(b"")
    out = global_solve._apply_revisit_floor(db, [LINK], floor=50, created=set())
    assert "floor could not be applied" in out["detail"]
