"""The relocalizer's revisit links in the final solve (review V8, M4; `global_solve`).

Pinned here:
  * only a MASKED and GATED final solve imports them; any other says why, and a solve
    with no links at all records exactly what it always did;
  * an imported pair counts only at the relocalizer's own floor
    (`relocalizer.REVISIT_MIN_INLIERS`), judged in the database the solve MAPS -- after the
    masks -- and a pair of that source below it is removed from that database, its
    matches and its geometry, while the walk's own database is never touched;
  * a listed pair the sequential matcher proposes anyway is not "of this source".

pycolmap is the recording fake of `test_world_builder_solve_masks`; the databases are
real SQLite in COLMAP's schema.
"""

from __future__ import annotations

import sqlite3

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
# the floor, in the database the solve maps


@pytest.mark.parametrize("inliers,kept", [(49, False), (50, True), (400, True)])
def test_an_imported_pair_below_the_floor_is_removed_from_the_mapped_database(
        walked, colmap, monkeypatch, inliers, kept):
    """The walk database holds the revisit pair (1, 4) at `inliers`; with a pairing
    overlap of 1 it is no sequential pair. At 49 it is REMOVED from the mapped copy --
    matches and geometry -- and at the floor (inclusive) it counts as verified. The
    walk's own database is byte for byte what it was."""
    _add_pair(walked.workspace.database_path, 1, 4, inliers)
    before = walked.workspace.database_path.read_bytes()
    _links(monkeypatch, [LINK])
    summary = _masked(walked, StubDetector(), gate=True, overlap=1)
    (_c, _n, database, _kw, listed), = colmap.calls("match_image_pairs")
    assert listed == "00000000.jpg 00000003.jpg\n" and database == "database.db"
    record = summary["solve"]["revisit_pairs"]
    assert record["imported"] is True and record["min_inliers"] == relocalizer.REVISIT_MIN_INLIERS
    assert record["verified"] == (1 if kept else 0)
    assert record["removed_below_floor"] == (0 if kept else 1)
    mapped = walked.workspace.root / summary["solve"]["database"]
    assert mapped != walked.workspace.database_path
    assert _pair_rows(mapped, 1 * _B + 4) == ((inliers, inliers) if kept else (None, None))
    assert walked.workspace.database_path.read_bytes() == before
    # what the mapper was given is the database the floor was applied to
    assert colmap.calls("global_mapping")[0][2] == mapped.name


def test_the_floor_is_judged_after_the_masks(walked, colmap, monkeypatch):
    """A pair whose 60 inliers include a masked keypoint loses its geometry in the mask
    filter and is re-verified (faked here: nothing comes back); in the mapped database it
    is under the floor and is removed, although the walk database held it at 60."""
    rows = [[0, 0]] + [[2, 2]] * 59
    con = sqlite3.connect(str(walked.workspace.database_path))
    con.execute("insert into matches values (?, ?, ?, ?)", (1 * _B + 4, *_blob(rows)))
    con.execute("insert into two_view_geometries values (?, ?, ?, ?, 2)", (1 * _B + 4, *_blob(rows)))
    con.commit()
    con.close()
    _links(monkeypatch, [LINK])
    summary = _masked(walked, StubDetector(), gate=True, overlap=1)
    record = summary["solve"]["revisit_pairs"]
    assert record["verified"] == 0 and record["removed_below_floor"] == 1
    mapped = walked.workspace.root / summary["solve"]["database"]
    assert _pair_rows(mapped, 1 * _B + 4) == (None, None)


def test_a_listed_pair_the_sequential_matcher_proposes_is_not_of_this_source(walked, colmap,
                                                                            monkeypatch):
    """Inside the pairing overlap (COLMAP's name order) the pair is a sequential pair
    first: held to COLMAP's 15 like every other, never removed by the revisit floor."""
    _add_pair(walked.workspace.database_path, 1, 4, 20)
    _links(monkeypatch, [LINK])
    summary = _masked(walked, StubDetector(), gate=True)       # overlap 20: 1 and 4 are 3 apart
    record = summary["solve"]["revisit_pairs"]
    assert record["removed_below_floor"] == 0 and record["kept_as_sequential"] == 1
    assert record["verified"] == 0
    mapped = walked.workspace.root / summary["solve"]["database"]
    assert _pair_rows(mapped, 1 * _B + 4) == (20, 20)


def test_the_floor_never_touches_the_walks_database(tmp_path):
    db = tmp_path / "database.db"
    _colmap_walk_database(db)
    _add_pair(db, 1, 4, 10)
    before = db.read_bytes()
    out = global_solve._apply_revisit_floor(db, [LINK], floor=50, overlap=1)
    assert out == {"verified": 0, "removed_below_floor": 1, "kept_as_sequential": 0}
    assert db.read_bytes() != before, "the helper edits whatever it is given..."
    # ... which is why `solve` gives it only the mapped database (the tests above).


def test_an_unreadable_mapped_database_is_recorded_not_fatal(tmp_path):
    db = tmp_path / "database.masked.p1.00000000.db"
    db.write_bytes(b"")
    out = global_solve._apply_revisit_floor(db, [LINK], floor=50, overlap=1)
    assert "floor could not be applied" in out["detail"]
