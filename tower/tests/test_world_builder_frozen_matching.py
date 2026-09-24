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


def test_a_new_pycolmap_matches_again(walked, colmap):
    _solve(walked, final=True, seed=7)
    colmap.__version__ = "fake-2"
    colmap.log.clear()
    summary = _solve(walked, final=True, seed=7)
    assert _calls(colmap) == ["extract_features", "match_sequential"]
    assert "(pycolmap)" in summary["solve"]["matching_detail"]


def test_new_revisit_links_match_again(walked, colmap, monkeypatch):
    """The imported links are part of the key: the frozen database holds what they
    added, and a different list asks for different matching."""
    from tower.world_builder import coherence_publish as CP

    monkeypatch.setattr(CP, "gate_final_solution", lambda store, w, s, solution, **kw: CP.GateResult(
        solution=solution, components=None, record={"state": CP.GATE_STATE_APPLIED, "seconds": 0.0}))
    _masked(walked, StubDetector(), seed=7, gate=True)
    monkeypatch.setattr(relocalizer, "revisit_pairs",
                        lambda session_dir: [("00000000.jpg", "00000003.jpg")])
    colmap.log.clear()
    summary = _masked(walked, StubDetector(), seed=7, gate=True)
    assert _calls(colmap) == ["extract_features", "match_sequential", "match_image_pairs"]
    assert "(revisit_pairs)" in summary["solve"]["matching_detail"]
    colmap.log.clear()
    again = _masked(walked, StubDetector(), seed=7, gate=True)
    assert _calls(colmap) == [] and again["solve"]["matching"] == GS.MATCHING_FROZEN
    assert again["solve"]["revisit_pairs"]["listed"] == 1


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
