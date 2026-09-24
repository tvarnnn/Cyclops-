"""Review V10 fixes in `global_solve` (P3.7, SOL2).

- MED-1(a): a consensus that maps further draws publishes draw 0 FIRST, `deferred`, so a kill
  during the further draws cannot lose the final solve; a caller's stop never reaches draw 0's
  own gate (the depth fake here HONOURS the stop, unlike the V9 tests' fake).
- L-11: a failed walk-database feature clear stays owed and heals.
- L-12: one image this solve cannot replace does not make it raise.
- L-13: a revisit matcher that fails after a partial write is still masked and floored.
- L-13c: an earlier solve's revisit imports are cleared before anything is matched.
- L-16 / L-16b: the frozen-matching end check, and the digest rule's text.
- L-16c: one `clear_image_features`, the fast one that raises.

pycolmap is the recording fake of `test_world_builder_solve_masks`; the mapper, the gate's
links and the depth network are the fakes of `test_world_builder_reproducible_finish`.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from tests.test_world_builder_reproducible_finish import N_A, engines, walk  # noqa: F401
from tests.test_world_builder_solve_masks import (  # noqa: F401 -- fixtures and helpers
    HEIGHT,
    N,
    WIDTH,
    StubDetector,
    _B,
    _blob,
    _masked,
    colmap,
    session,
    walked,
)
from tower.world_builder import coherence_publish as CP
from tower.world_builder import global_solve as GS
from tower.world_builder import relocalizer

RECORD_FILE = "images.provenance.json"


@pytest.fixture(autouse=True)
def _no_links(monkeypatch):
    monkeypatch.setattr(relocalizer, "revisit_pairs", lambda session_dir: [])
    monkeypatch.setenv("TOWER_WORLD_TRANSIENTS", "off")


# ---------------------------------------------------------------------------
# MED-1(a): draw 0 is published before any further draw is mapped


@pytest.fixture
def depth_calls(engines, monkeypatch):
    """The engines' depth network, HONOURING the stop as the product's depth stage does (it
    asks `should_stop` before its first frame, `dense_pipeline.run_depth_stage`). Records each
    call's candidate and whether a stop was handed to it."""
    real = CP.run_gate_depth
    calls = []

    def depth(store, world_id, session_id, solution, intrinsics, should_stop=None):
        calls.append({"identity": GS.solve_identity(solution), "stop": should_stop is not None})
        if should_stop is not None and should_stop():
            raise CP.DepthUnavailable(f"{CP.DEPTH_STOPPED}: the depth stage was stopped after 0 "
                                      "frames")
        return real(store, world_id, session_id, solution, intrinsics, should_stop=should_stop)

    monkeypatch.setattr(CP, "run_gate_depth", depth)
    return calls


@pytest.fixture
def seeds(engines, monkeypatch):
    """The mapper seeds asked for, in order (draw 0 is the solve's own mapping)."""
    asked = []
    real = GS._map_candidate

    def counting(*a, **kw):
        asked.append(kw.get("seed"))
        return real(*a, **kw)

    monkeypatch.setattr(GS, "_map_candidate", counting)
    return asked


def _finish(walk, **kw):
    """The builder child's final solve with a consensus of 3 (the child passes no stop)."""
    return GS.solve(walk.store, walk.world_id, walk.session_id, final=True, masks=True, seed=5,
                    gate=True, consensus=3, transient_backend_factory=StubDetector(),
                    mask_device_probe=lambda: None, input_digest="walk-digest", **kw)


def _published(walk) -> dict:
    return json.loads(walk.workspace.solution_path.read_text(encoding="utf-8"))


def _room(published: dict) -> int:
    return sum(1 for p in published["poses"].values()
               if int(p.get("component", 0)) == 0 and int(p.get("observations", 0)) >= 30)


def test_a_kill_during_a_further_draw_leaves_draw_0_published_and_owed(walk, seeds, depth_calls,
                                                                       colmap, monkeypatch):
    """RV10-B `test_rv10b_m3_kill.py`: the builder's final solve is a child with no stop, and a
    hard stop kills it while draw 1 is being mapped. At e5f7151 the whole final solve was lost:
    `solution.json` stayed the walk's background solve and there was no components record. Now
    draw 0 was published BEFORE draw 1 was mapped -- attached, its consensus `deferred` -- and the
    finisher's re-gate owes the consensus."""
    real = GS._map_candidate
    at_draw_1 = {}

    def dying(*a, **kw):
        if kw.get("seed") == 6:
            at_draw_1.update(_published(walk))       # what a reader sees while draw 1 maps
            raise KeyboardInterrupt("hard stop while mapping draw 1")
        return real(*a, **kw)

    monkeypatch.setattr(GS, "_map_candidate", dying)
    with pytest.raises(KeyboardInterrupt):
        _finish(walk)
    published = _published(walk)
    assert published == at_draw_1, "nothing was published after draw 1 began"
    assert published["timing"]["final"] is True, "the final solve, not the walk's background one"
    gate = published["gate"]
    assert gate["consensus"]["state"] == CP.CONSENSUS_DEFERRED
    assert gate["attach"] is True and gate["depth"]["state"] == CP.DEPTH_OK
    assert gate["retryable"] is True and gate["cause"] == CP.CAUSE_CONSENSUS_DEFERRED
    assert gate["consensus"]["why"] == CP.WHY_PUBLISHED_FIRST
    assert published["solve"]["seed"] == 5
    assert _room(published) == N_A, "draw 0's room, not an anchor-only fail-safe"
    assert (walk.workspace.root / CP.COMPONENTS_FILENAME).is_file()
    assert CP.regate_owed(walk.store, walk.world_id, walk.session_id) == CP.CAUSE_CONSENSUS_DEFERRED


def test_without_a_kill_the_full_consensus_is_published_over_draw_0(walk, seeds, depth_calls,
                                                                     colmap):
    summary = _finish(walk)
    assert seeds == [5, 6, 7]
    published = _published(walk)
    assert published["gate"]["consensus"]["state"] == CP.CONSENSUS_APPLIED
    assert CP.regate_owed(walk.store, walk.world_id, walk.session_id) is None
    assert summary["gate"]["consensus"]["state"] == CP.CONSENSUS_APPLIED
    # DRAW 0 IS GATED ONCE (review V11, LOW-1): the early publish's gate of it is handed to the
    # consensus, which gates each further draw once. At a559fc0 draw 0 was gated twice (4 depth
    # stages). No gate is handed a stop the caller did not give.
    assert len(depth_calls) == 3 and not any(c["stop"] for c in depth_calls)
    ids = [c["identity"] for c in depth_calls]
    assert ids.count(ids[0]) == 1, "draw 0's depth stage ran once (this fake maps draws 1 and 2 alike)"
    # what draw 0 cost is on the record, and it is the gate that ran: the first, cold one -- at
    # a559fc0 the record said {cached: all, predicted: 0}, the second gate's
    draw0 = published["gate"]["consensus"]["draws"][0]
    assert draw0["draw"] == 0 and isinstance(draw0["gate_s"], float)
    assert draw0["predictions"]["predicted"] > 0 and draw0["predictions"]["cached"] == 0
    if published["gate"]["consensus"]["chosen"]["draw"] == 0:
        assert published["gate"]["depth"]["predictions"] == draw0["predictions"]
        assert published["gate"]["seconds"] == draw0["gate_s"]


@pytest.mark.parametrize("stop_at", ["after-draw-0", "during-draw-1-gate", "after-draw-1"])
def test_a_callers_stop_never_publishes_a_stopped_draw_0(walk, seeds, depth_calls, colmap, stop_at):
    """With a depth stage that HONOURS the stop (review V10, MED-1: the V9 tests' fake ignored
    it, so they passed for the wrong reason). Wherever the stop is asked -- before the further
    draws, while a further draw is gated, or between draws -- the published solve is draw 0
    ATTACHED, `deferred`, owed: never the anchor-only fail-safe a stopped depth stage gives. At
    e5f7151 a stop asked before the further draws reached draw 0's own gate.

    (V10's "during-draw-0-regate" case -- a stop reaching draw 0's SECOND gate -- cannot happen
    since review V11, LOW-1: draw 0 is gated once. That is asserted below; the stop that reaches
    a further draw's gate takes its place.)"""
    def stop() -> bool:
        if stop_at == "after-draw-0":
            return len(seeds) >= 1
        if stop_at == "after-draw-1":
            return len(seeds) >= 2
        # asked once draw 1's depth stage has begun: it reaches that draw's own gate
        return len(depth_calls) >= 2

    summary = _finish(walk, should_stop=stop)
    ids = [c["identity"] for c in depth_calls]
    assert ids.count(ids[0]) == 1, "draw 0 is gated once"
    published = _published(walk)
    gate = published["gate"]
    assert gate["consensus"]["state"] == CP.CONSENSUS_DEFERRED
    assert gate["attach"] is True, "draw 0 attached, not a stopped fail-safe"
    assert gate["depth"]["state"] == CP.DEPTH_OK
    assert _room(published) == N_A
    assert published["solve"]["seed"] == 5
    assert CP.regate_owed(walk.store, walk.world_id, walk.session_id) == CP.CAUSE_CONSENSUS_DEFERRED
    assert summary["gate"]["consensus"]["state"] == CP.CONSENSUS_DEFERRED
    assert seeds == {"after-draw-0": [5], "during-draw-1-gate": [5, 6], "after-draw-1": [5, 6]}[stop_at], \
        "no draw is mapped after the stop is seen"
    if stop_at == "during-draw-1-gate":
        # the stop cost draw 1 its vote: draw 0 -- the early publish's own gate of it -- is
        # published over the early publish, deferred by the stop
        assert gate["consensus"]["why"] == CP.WHY_STOPPED
        assert gate["consensus"]["draws"][0]["predictions"]["predicted"] > 0


def test_a_draw_0_fail_safe_is_published_once_and_no_draw_is_mapped(walk, seeds, engines, colmap,
                                                                     monkeypatch):
    """Draw 0's gate attaches nothing (no metric level: the scale fail-safe): there is nothing to
    vote on, so the early publish is the only one, exactly as without it."""
    monkeypatch.setattr(CP, "measure_metric_scale", lambda solution, name_of, db, work: {
        "metric_log": {}, "cameras_published": 0, "cameras_measured": 0})
    calls = []
    real = CP.gate_and_publish

    def recording(*a, **kw):
        calls.append(kw.get("stopped", False))
        return real(*a, **kw)

    monkeypatch.setattr(CP, "gate_and_publish", recording)
    summary = _finish(walk)
    assert seeds == [5]
    assert len(calls) == 1
    assert summary["gate"]["attach"] is False
    assert summary["gate"]["consensus"]["state"] in (CP.CONSENSUS_NOT_NEEDED, CP.CONSENSUS_DEFERRED)


def test_a_single_draw_publishes_once_exactly_as_today(walk, seeds, engines, colmap, monkeypatch):
    seen = []
    real = CP.gate_and_publish

    def recording(*a, **kw):
        seen.append(sorted(kw))
        return real(*a, **kw)

    monkeypatch.setattr(CP, "gate_and_publish", recording)
    GS.solve(walk.store, walk.world_id, walk.session_id, final=True, masks=True, seed=5, gate=True,
             consensus=1, transient_backend_factory=StubDetector(), mask_device_probe=lambda: None,
             input_digest="walk-digest", should_stop=lambda: False)
    assert seen == [sorted(["final", "gate", "database_path", "keyframes", "write", "consensus"])]
    assert seeds == [5]


# ---------------------------------------------------------------------------
# L-11: a failed walk-database feature clear stays owed, and heals


def _frame(path: Path, value: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", np.full((HEIGHT, WIDTH, 3), value, np.uint8),
                           [cv2.IMWRITE_JPEG_QUALITY, 100])
    assert ok
    path.write_bytes(buf.tobytes())
    return path


def _sha1(path: Path) -> str:
    from tower.world_builder.solve_masks import file_sha1

    return file_sha1(path)


def _record(s, images: dict) -> None:
    (s.workspace.root / RECORD_FILE).write_text(json.dumps(
        {"record": "wb-solver-image-provenance/1", "images": images}), encoding="utf-8")


def _record_json(s) -> dict:
    return json.loads((s.workspace.root / RECORD_FILE).read_text(encoding="utf-8"))


def _redacted_entry(s, i: int) -> dict:
    name = f"{i:08d}.jpg"
    return {"keyframe_id": f"{s.session_id}:{i:08d}", "source": "redacted",
            "frame": f"images/{name}", "sha1": _sha1(s.workspace.images_dir / name)}


def _keypoints(db: Path, image_id: int) -> int:
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        return con.execute("select count(*) from keypoints where image_id = ?",
                           (image_id,)).fetchone()[0]
    finally:
        con.close()


@pytest.fixture
def raw_planned(walked, tmp_path):
    """A re-finish's record says image 00000002.jpg (image id 3) came from its stored copy,
    while the solve now plans its raw frame (bright 200): the next solve rewrites it."""
    walked.raw = _frame(tmp_path / "capA" / "frames" / "00000002.jpg", 200)
    _record(walked, {f"{i:08d}.jpg": _redacted_entry(walked, i) for i in range(N)})
    GS.write_sources_records(walked.workspace, {f"{walked.session_id}:00000002": str(walked.raw)})
    return walked


def test_a_feature_clear_that_fails_stays_owed_and_the_next_solve_heals_it(raw_planned, colmap,
                                                                           monkeypatch):
    """RV10-B M7-1: another writer holds the walk database while the solve clears the rewritten
    image's features. At e5f7151 the provenance record already named the new image and the error
    was swallowed, so no later solve cleared them: old keypoints mapped with new pixels for
    good. Now the clear stays owed in the record and the next solve makes it."""
    s = raw_planned
    ws = s.workspace
    monkeypatch.setattr(GS, "CLEAR_FEATURES_BUSY_TIMEOUT_MS", 200, raising=False)
    other = sqlite3.connect(str(ws.database_path), timeout=0)
    other.execute("BEGIN EXCLUSIVE")
    try:
        first = GS.solve(s.store, s.world_id, s.session_id, final=True)
    finally:
        other.rollback()
        other.close()
    rewritten = first["solve"]["solver_images_rewritten"]
    assert rewritten["examples"] == ["00000002.jpg"] and "detail" in rewritten["features_cleared"]
    assert _keypoints(ws.database_path, 3) == 1, "the clear failed: the old features are there"
    owed = _record_json(s).get("features_clear_owed")
    second = GS.solve(s.store, s.world_id, s.session_id, final=True)
    assert _keypoints(ws.database_path, 3) == 0, "the owed clear was made"
    assert owed == ["00000002.jpg"]
    healed = second["solve"]["solver_images_rewritten"]
    assert healed["count"] == 0 and healed["features_cleared"] == {"images": 1, "pairs": 2}
    assert healed["owed_earlier"] == {"count": 1, "examples": ["00000002.jpg"]}
    assert GS.FEATURES_CLEAR_OWED_KEY not in _record_json(s), "nothing is owed any more"
    third = GS.solve(s.store, s.world_id, s.session_id, final=True)
    assert "solver_images_rewritten" not in third["solve"]


def test_a_kill_between_the_rewrite_and_the_clear_leaves_the_clear_owed(raw_planned, colmap,
                                                                        monkeypatch):
    """The process dies right after the new bytes replace the image, before the clear. At
    e5f7151 the next solve rewrote the same bytes again, saw nothing change, and never cleared:
    the owed clear was recorded nowhere. Now it is recorded before the replace."""
    s = raw_planned
    ws = s.workspace
    real = GS.replace_with_retry

    def dies_after(tmp, target):
        real(tmp, target)
        if Path(target).name == "00000002.jpg":
            raise KeyboardInterrupt("killed after the replace")

    monkeypatch.setattr(GS, "replace_with_retry", dies_after)
    with pytest.raises(KeyboardInterrupt):
        GS.solve(s.store, s.world_id, s.session_id, final=True)
    monkeypatch.setattr(GS, "replace_with_retry", real)
    assert _keypoints(ws.database_path, 3) == 1
    GS.solve(s.store, s.world_id, s.session_id, final=True)
    assert _keypoints(ws.database_path, 3) == 0, "the owed clear was made after the kill"
    assert GS.FEATURES_CLEAR_OWED_KEY not in _record_json(s)


def test_a_solve_that_owes_nothing_writes_no_owed_key(raw_planned, colmap):
    first = GS.solve(raw_planned.store, raw_planned.world_id, raw_planned.session_id, final=True)
    assert first["solve"]["solver_images_rewritten"]["features_cleared"] == {"images": 1, "pairs": 2}
    record = _record_json(raw_planned)
    assert set(record) == {"record", "images"}, "the agreed format, with nothing owed"


# ---------------------------------------------------------------------------
# L-12: an image this solve cannot replace does not make it raise


def test_an_image_the_solve_cannot_replace_is_left_and_the_solve_goes_on(raw_planned, colmap,
                                                                         monkeypatch):
    """RV10-B M7-3, portable: replacing 00000002.jpg fails (another process holds it). With a
    provenance record the solve used to raise; now it is left as it is, its entry unchanged,
    it owes no clear (its features are still its own), and the record says so."""
    s = raw_planned
    before = _record_json(s)["images"]["00000002.jpg"]
    image_before = (s.workspace.images_dir / "00000002.jpg").read_bytes()
    real = GS.replace_with_retry

    def held(tmp, target):
        if Path(target).name == "00000002.jpg":
            raise PermissionError(13, "held by another process", str(target))
        return real(tmp, target)

    monkeypatch.setattr(GS, "replace_with_retry", held)
    summary = GS.solve(s.store, s.world_id, s.session_id, final=True)
    assert summary["solved"] is True
    record = summary["solve"]["solver_images_rewritten"]
    assert record["count"] == 0
    assert record["not_rewritten"] == {"count": 1, "examples": ["00000002.jpg"]}
    assert (s.workspace.images_dir / "00000002.jpg").read_bytes() == image_before
    assert _record_json(s)["images"]["00000002.jpg"] == before
    assert GS.FEATURES_CLEAR_OWED_KEY not in _record_json(s)
    assert _keypoints(s.workspace.database_path, 3) == 1


@pytest.mark.skipif(sys.platform != "win32", reason="Windows share modes")
def test_an_exclusively_held_image_with_a_record_does_not_make_the_solve_raise(walked, colmap,
                                                                              monkeypatch):
    """RV10-B M7-3 as it was found: a handle with no sharing on one solver image, and a record."""
    import ctypes
    from ctypes import wintypes

    from tower import storage

    monkeypatch.setattr(storage, "REPLACE_BUDGET_S", 0.2)
    _record(walked, {f"{i:08d}.jpg": _redacted_entry(walked, i) for i in range(N)})
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.restype = wintypes.HANDLE
    handle = k32.CreateFileW(str(walked.workspace.images_dir / "00000002.jpg"), 0x80000000, 0, None,
                             3, 0x80, None)
    try:
        summary = GS.solve(walked.store, walked.world_id, walked.session_id, final=True)
    finally:
        k32.CloseHandle(handle)
    assert summary["solved"] is True
    assert summary["solve"]["solver_images_rewritten"]["not_rewritten"]["examples"] == ["00000002.jpg"]


# ---------------------------------------------------------------------------
# L-8 (the solve's side of REF2's fix): a raw image whose recorded frame is gone is kept


def _prepare(s):
    return GS.prepare_images(s.store, s.world_id, s.session_id,
                             s.store.read_keyframes(s.world_id, s.session_id))


@pytest.fixture
def raw_gone(session, colmap, tmp_path):
    """Image 00000002.jpg was undistorted from the keyframe's own raw frame -- the record says
    so, from exactly the frame `sources.json` names, and its SHA-1 is the file's -- and that raw
    frame is no longer on disk (a moved or purged capture: the path never exists here)."""
    _prepare(session)
    s = session
    s.kid2 = f"{s.session_id}:00000002"
    s.gone = tmp_path / "moved-capture" / "frames" / "00000002.jpg"
    s.image2 = s.workspace.images_dir / "00000002.jpg"
    ok, buf = cv2.imencode(".jpg", np.full((HEIGHT, WIDTH, 3), 200, np.uint8))
    s.image2.write_bytes(buf.tobytes())            # the raw-derived image the walk left
    images = {f"{i:08d}.jpg": _redacted_entry(s, i) for i in range(N)}
    images["00000002.jpg"] = {"keyframe_id": s.kid2, "source": "raw", "frame": str(s.gone),
                              "sha1": _sha1(s.image2)}
    _record(s, images)
    GS.write_sources_records(s.workspace, {s.kid2: str(s.gone)})
    return s


def test_a_raw_image_whose_recorded_frame_is_gone_is_kept(raw_gone):
    """RV10-C `test_rv10c_rawgone.py` (L-8): the stored copy is planned ONLY because the
    recorded raw frame is gone, and the record proves the image is that frame's. At e5f7151 the
    image was rewritten from the redacted copy, losing the raw frame's texture."""
    before = raw_gone.image2.read_bytes()
    _camera, written = _prepare(raw_gone)
    assert written == 0
    assert raw_gone.image2.read_bytes() == before
    assert _record_json(raw_gone)["images"]["00000002.jpg"]["source"] == "raw"


@pytest.mark.parametrize("why", ["frame-on-disk-elsewhere", "other-frame", "sha1"])
def test_otherwise_the_planned_frame_still_wins(raw_gone, why, tmp_path):
    """Only that case: an entry naming ANOTHER frame than `sources.json`, or whose SHA-1 is not
    the file's, is rewritten from the stored copy as before; and when the capture directory
    holds the keyframe's raw frame, that frame is planned and judged as usual."""
    s = raw_gone
    capture_dirs = ()
    if why == "other-frame":
        GS.write_sources_records(s.workspace, {s.kid2: str(tmp_path / "another" / "00000002.jpg")})
    elif why == "sha1":
        record = _record_json(s)
        record["images"]["00000002.jpg"]["sha1"] = "0" * 40
        (s.workspace.root / RECORD_FILE).write_text(json.dumps(record), encoding="utf-8")
    else:
        found = tmp_path / "capB"
        (found / "frames").mkdir(parents=True)
        ok, buf = cv2.imencode(".jpg", np.full((HEIGHT, WIDTH, 3), 90, np.uint8))
        (found / "frames" / "00000002.jpg").write_bytes(buf.tobytes())
        capture_dirs = (found,)
    GS.prepare_images(s.store, s.world_id, s.session_id,
                      s.store.read_keyframes(s.world_id, s.session_id), capture_dirs=capture_dirs)
    entry = _record_json(s)["images"]["00000002.jpg"]
    if why == "frame-on-disk-elsewhere":
        assert entry["source"] == "raw" and entry["frame"] == str(found / "frames" / "00000002.jpg")
    else:
        assert entry["source"] == "redacted"
    assert entry["sha1"] == _sha1(s.image2)


# ---------------------------------------------------------------------------
# L-13 / L-13c: the revisit import


LINK = ("00000000.jpg", "00000003.jpg")


@pytest.fixture
def stub_gate(monkeypatch):
    def fake(store, world_id, session_id, solution, *, database_path, keyframes, **kw):
        return CP.GateResult(solution=solution, components=None,
                             record={"state": CP.GATE_STATE_APPLIED, "seconds": 0.0})

    monkeypatch.setattr(CP, "gate_final_solution", fake)


def _add_pair(db, a_id, b_id, inliers, config=2):
    rows = [[2 + (i % 2), 2 + (i % 2)] for i in range(inliers)]
    con = sqlite3.connect(str(db))
    con.execute("insert or replace into matches values (?, ?, ?, ?)", (a_id * _B + b_id, *_blob(rows)))
    con.execute("insert or replace into two_view_geometries values (?, ?, ?, ?, ?)",
                (a_id * _B + b_id, *_blob(rows), config))
    con.commit()
    con.close()


def _geometry_rows(db, pid):
    con = sqlite3.connect(f"file:{Path(db).as_posix()}?mode=ro", uri=True)
    try:
        row = con.execute("select rows from two_view_geometries where pair_id = ?", (pid,)).fetchone()
        return row[0] if row else None
    finally:
        con.close()


def test_a_revisit_matcher_that_fails_after_writing_is_still_floored(walked, colmap, monkeypatch,
                                                                     stub_gate):
    """RV10-B P6: COLMAP's matcher writes a listed pair (20 inliers, under the floor of 50) and
    then raises. At e5f7151 the import returned at the failure, before the masks and the floor,
    so the 20-inlier pair reached the mapper. Now it is floored like any pair the import made,
    and the record keeps why the matcher failed."""
    original = colmap.match_image_pairs

    def writes_then_raises(database_path, **kwargs):
        original(database_path, **kwargs)
        _add_pair(database_path, 1, 4, 20)
        raise RuntimeError("CUDA out of memory (after the first pair)")

    colmap.match_image_pairs = writes_then_raises
    monkeypatch.setattr(relocalizer, "revisit_pairs", lambda d: [LINK])
    summary = _masked(walked, StubDetector(), gate=True, seed=3, overlap=1)
    mapped = walked.workspace.root / summary["solve"]["database"]
    record = summary["solve"]["revisit_pairs"]
    assert _geometry_rows(mapped, 1 * _B + 4) is None, "under the floor: taken out again"
    assert record["created_by_import"] == 1 and record["removed_below_floor"] == 1
    assert "CUDA out of memory" in record["detail"]
    assert colmap.calls("global_mapping")[0][2] == mapped.name


def test_earlier_imports_are_cleared_before_the_re_extracted_database_is_matched(session, colmap,
                                                                                 monkeypatch,
                                                                                 stub_gate):
    """RV10-B P5: no walk database, so each masked solve re-extracts into its own database,
    seeded with a copy of the newest earlier one. Solve 1 imports (1,4) at 60 inliers. Solve 2
    lists no links, and its loop detection finds (1,4) at 30 on its own. At e5f7151 the seed
    copy still held solve 1's import when the matcher ran, so the matcher skipped the pair, and
    the clear after matching then took it out: solve 2 mapped no (1,4). Now the import is
    cleared first, and solve 2 maps loop detection's pair, as a fresh database would."""
    ws = session.workspace
    ws.database_path.unlink(missing_ok=True)

    def make_schema(db):
        con = sqlite3.connect(str(db))
        con.executescript(
            "create table if not exists images (image_id integer primary key, name text, camera_id integer);"
            "create table if not exists keypoints (image_id integer primary key, rows integer, cols integer, data blob);"
            "create table if not exists matches (pair_id integer primary key, rows integer, cols integer, data blob);"
            "create table if not exists two_view_geometries (pair_id integer primary key, rows integer,"
            " cols integer, data blob, config integer);")
        for i in range(4):
            con.execute("insert or ignore into images values (?, ?, 1)", (i + 1, f"{i:08d}.jpg"))
        con.commit()
        con.close()

    loop_finds = {"on": False}
    extract = colmap.extract_features

    def extract_features(database_path, image_path, **kw):
        extract(database_path, image_path, **kw)
        make_schema(database_path)

    def match_sequential(database_path, **kwargs):
        colmap.log.append(("call", "match_sequential", Path(database_path).name, ()))
        # COLMAP skips a pair the database already holds; loop detection finds (1,4) at 30.
        if loop_finds["on"] and _geometry_rows(database_path, 1 * _B + 4) is None:
            _add_pair(database_path, 1, 4, 30)

    def match_image_pairs(database_path, **kwargs):
        listed = Path(kwargs["pairing_options"]._values["match_list_path"]).read_text().split()
        con = sqlite3.connect(str(database_path))
        ids = dict(con.execute("select name, image_id from images"))
        con.close()
        for a, b in zip(listed[::2], listed[1::2]):
            lo, hi = sorted((ids[a], ids[b]))
            _add_pair(database_path, lo, hi, 60)

    colmap.extract_features = extract_features
    colmap.match_sequential = match_sequential
    colmap.match_image_pairs = match_image_pairs
    monkeypatch.setattr(relocalizer, "revisit_pairs", lambda d: [LINK])
    first = _masked(session, StubDetector(), gate=True, seed=3, overlap=1)
    assert first["solve"]["masking"] == "re-extracted"
    assert _geometry_rows(ws.root / first["solve"]["database"], 1 * _B + 4) == 60
    loop_finds["on"] = True
    monkeypatch.setattr(relocalizer, "revisit_pairs", lambda d: [])
    second = _masked(session, StubDetector(), gate=True, seed=3, overlap=1)
    assert second["transients"]["database_reused"] is True
    assert second["solve"]["revisit_imports_cleared"] == 1
    assert _geometry_rows(ws.root / second["solve"]["database"], 1 * _B + 4) == 30


# ---------------------------------------------------------------------------
# L-16 / L-16b: the frozen matching's end check, and the rule


def test_the_end_check_compares_with_the_digest_the_freeze_check_accepted(walked, colmap,
                                                                          monkeypatch):
    """RV10-B Q1: inside the window, a lockless hand-run solve matches into the walk database
    under ANOTHER key (verification seed 99) and freezes that state. At e5f7151 the end check
    compared the walk database with the record re-read at the end -- the other solve's -- so
    this one was labelled `frozen` although its own key was never checked against what it
    mapped. Now it compares with the digest its own check accepted."""
    _masked(walked, StubDetector(), seed=7)                 # freezes under seed 7
    checked = GS._frozen_matching_refusal
    ws = walked.workspace

    def check_then_another_solve_refreezes(workspace, database_path, key, images, *a, **kw):
        out = checked(workspace, database_path, key, images, *a, **kw)
        con = sqlite3.connect(str(ws.database_path))
        con.execute("insert or replace into matches values (?, ?, ?, ?)", (1 * _B + 3, *_blob([[2, 3], [3, 2]])))
        con.execute("insert or replace into two_view_geometries values (?, ?, ?, ?, 2)",
                    (1 * _B + 3, *_blob([[2, 3], [3, 2]])))
        con.commit()
        con.close()
        GS._freeze_matching(workspace, ws.database_path, dict(key, verification_seed=99), images)
        return out

    monkeypatch.setattr(GS, "_frozen_matching_refusal", check_then_another_solve_refreezes)
    summary = _masked(walked, StubDetector(), seed=7)
    assert summary["solve"]["matching"] == GS.MATCHING_MATCHED
    assert summary["solve"]["matching_detail"] == GS.FROZEN_REFUSAL_CHANGED_UNDER_THE_SOLVE


def test_the_digest_rule_states_the_tie_safe_pivot_and_nothing_appended():
    """P3.6 changed the algorithm (no appended magnitude, a tie-safe pivot); the rule, recorded
    in every seeded final solution, must say so (review V10, L-16b)."""
    rule = GS.DATABASE_DIGEST_RULE
    assert "first entry in position order" in rule and "1e-09" in rule
    assert "nothing appended" in rule and "10 decimals" in rule
    # the rule the text states: a pure rescale and a sign flip of F leave `stated` unchanged,
    # one entry's tie with the pivot resolves by position
    f = np.array([0.3, -1.0, 0.2, 1.0 - 1e-12, 0.1, 0.0, 0.0, 0.5, 0.25])
    assert GS._stated_bytes(f.tobytes(), up_to_sign=True) == \
        GS._stated_bytes((-3.7 * f).tobytes(), up_to_sign=True)


# ---------------------------------------------------------------------------
# L-16c: one clear_image_features, the fast one that raises


def _features_db(path: Path) -> Path:
    con = sqlite3.connect(str(path))
    con.executescript(
        "create table images (image_id integer primary key, name text, camera_id integer);"
        "create table keypoints (image_id integer primary key, rows integer, cols integer, data blob);"
        "create table descriptors (image_id integer primary key, rows integer, cols integer, data blob);"
        "create table matches (pair_id integer primary key, rows integer, cols integer, data blob);"
        "create table two_view_geometries (pair_id integer primary key, rows integer, cols integer,"
        " data blob, config integer);")
    for i in range(1, 5):
        con.execute("insert into images values (?, ?, 1)", (i, f"{i - 1:08d}.jpg"))
        con.execute("insert into keypoints values (?, 1, 6, x'00')", (i,))
        con.execute("insert into descriptors values (?, 1, 128, x'00')", (i,))
    for a, b in ((1, 2), (2, 3), (3, 4), (1, 4)):
        con.execute("insert into matches values (?, 1, 2, x'00')", (a * _B + b,))
        con.execute("insert into two_view_geometries values (?, 1, 2, x'00', 2)", (a * _B + b,))
    con.execute("insert into matches values (?, 1, 2, x'00')", (2 * _B + 4,))    # matches only
    con.commit()
    con.close()
    return path


def test_clear_image_features_is_one_exported_implementation(tmp_path):
    assert callable(GS.clear_image_features)
    assert not hasattr(GS, "_clear_image_features"), "no second implementation"
    db = _features_db(tmp_path / "a.db")
    out = GS.clear_image_features(db, ["00000001.jpg", "missing.jpg"])     # image id 2
    assert out == {"images": 1, "pairs": 3}                                # (1,2) (2,3) (2,4)
    con = sqlite3.connect(str(db))
    try:
        assert con.execute("select count(*) from keypoints where image_id = 2").fetchone()[0] == 0
        assert con.execute("select count(*) from descriptors where image_id = 2").fetchone()[0] == 0
        assert con.execute("select count(*) from images").fetchone()[0] == 4
        assert {p for (p,) in con.execute("select pair_id from matches")} == {3 * _B + 4, 1 * _B + 4}
        assert {p for (p,) in con.execute("select pair_id from two_view_geometries")} == \
            {3 * _B + 4, 1 * _B + 4}
    finally:
        con.close()
    assert GS.clear_image_features(db, []) == {"images": 0, "pairs": 0}
    assert GS.clear_image_features(tmp_path / "absent.db", ["x"]) == {"images": 0, "pairs": 0}


def test_clear_image_features_raises_when_another_writer_holds_the_database(tmp_path, monkeypatch):
    monkeypatch.setattr(GS, "CLEAR_FEATURES_BUSY_TIMEOUT_MS", 100)
    db = _features_db(tmp_path / "a.db")
    other = sqlite3.connect(str(db), timeout=0)
    other.execute("BEGIN EXCLUSIVE")
    try:
        with pytest.raises(sqlite3.OperationalError):
            GS.clear_image_features(db, ["00000001.jpg"])
    finally:
        other.rollback()
        other.close()
    assert GS.clear_image_features(db, ["00000001.jpg"]) == {"images": 1, "pairs": 3}


def test_clear_image_features_leaves_a_database_that_is_not_the_walks_alone(tmp_path):
    empty = tmp_path / "empty.db"
    sqlite3.connect(str(empty)).close()
    out = GS.clear_image_features(empty, ["00000001.jpg"])
    assert out["images"] == 0 and "skipped" in out
