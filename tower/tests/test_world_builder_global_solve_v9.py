"""Review V9 fixes in `global_solve` (P3.6, SOL): the solver images' provenance (M-7, the
prepare side), a consensus draw's own records and scratch (LOW), and the stop reaching the
consensus (M-3, the stop side).

The revisit import (M-9, M-10) is pinned in `test_world_builder_revisit_import.py`, the
frozen matching and the digest (LOW) in `test_world_builder_frozen_matching.py`.

pycolmap is the recording fake of `test_world_builder_solve_masks`; the mapper, the gate's
links and the depth network are the fakes of `test_world_builder_reproducible_finish`.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from tests.test_world_builder_reproducible_finish import engines, walk  # noqa: F401 -- fixtures
from tests.test_world_builder_solve_masks import (  # noqa: F401 -- fixtures and helpers
    HEIGHT,
    N,
    WIDTH,
    StubDetector,
    _B,
    _masked,
    colmap,
    session,
    walked,
)
from tower.world_builder import coherence_publish as CP
from tower.world_builder import global_solve as GS
from tower.world_builder import relocalizer


@pytest.fixture(autouse=True)
def _no_links(monkeypatch):
    monkeypatch.setattr(relocalizer, "revisit_pairs", lambda session_dir: [])


# ---------------------------------------------------------------------------
# M-7: an existing solver image is kept only when its provenance is what this solve plans


def _frame(path: Path, value: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", np.full((HEIGHT, WIDTH, 3), value, np.uint8),
                           [cv2.IMWRITE_JPEG_QUALITY, 100])
    assert ok
    path.write_bytes(buf.tobytes())
    return path


def _mean(path: Path) -> float:
    return float(cv2.imread(str(path), cv2.IMREAD_COLOR).mean())


def _sha1(path: Path) -> str:
    from tower.world_builder.solve_masks import file_sha1

    return file_sha1(path)


def _prepare(s, **kw):
    return GS.prepare_images(s.store, s.world_id, s.session_id,
                             s.store.read_keyframes(s.world_id, s.session_id), **kw)


# The agreed format, spelled out (it is REF's too): a missing constant is not what is tested.
RECORD_FILE = "images.provenance.json"
RAW, REDACTED = "raw", "redacted"


def _record(s, images: dict) -> None:
    (s.workspace.root / RECORD_FILE).write_text(json.dumps(
        {"record": "wb-solver-image-provenance/1", "images": images}), encoding="utf-8")


def _recorded(s) -> dict:
    return json.loads((s.workspace.root / RECORD_FILE).read_text(encoding="utf-8"))["images"]


def _redacted_entry(s, i: int) -> dict:
    name = f"{i:08d}.jpg"
    return {"keyframe_id": f"{s.session_id}:{i:08d}", "source": REDACTED,
            "frame": f"images/{name}", "sha1": _sha1(s.workspace.images_dir / name)}


@pytest.fixture
def prepared(session, colmap, tmp_path):
    """The walk's solver images, undistorted from the stored (redacted) copies -- what a
    walk without its raw frames, or a re-finish without TOWER_CAPTURE_ROOT, leaves -- and
    raw capture frames on disk that the stored copies are not (bright, 200 and 90)."""
    _prepare(session)
    session.raw = _frame(tmp_path / "capA" / "frames" / "00000002.jpg", 200)
    session.other = _frame(tmp_path / "capB" / "frames" / "00000002.jpg", 90)
    session.kid2 = f"{session.session_id}:00000002"
    session.image2 = session.workspace.images_dir / "00000002.jpg"
    return session


def test_without_a_record_an_existing_image_is_kept_and_nothing_is_written(prepared):
    """Today's behaviour, exactly: no provenance record, no rewrite and no record written,
    although `sources.json` now names a raw frame."""
    before = prepared.image2.read_bytes()
    GS.write_sources_records(prepared.workspace, {prepared.kid2: str(prepared.raw)})
    _camera, written = _prepare(prepared)
    assert written == 0
    assert prepared.image2.read_bytes() == before
    assert not (prepared.workspace.root / RECORD_FILE).exists()


def test_a_redacted_image_is_rewritten_when_its_raw_frame_is_planned(prepared):
    """RV9-D D1: the record says the image came from the stored (redacted) copy; this solve
    plans the raw frame found by identity. It is REWRITTEN from the raw frame, and recorded."""
    _record(prepared, {f"{i:08d}.jpg": _redacted_entry(prepared, i) for i in range(N)})
    GS.write_sources_records(prepared.workspace, {prepared.kid2: str(prepared.raw)})
    _camera, written = _prepare(prepared)
    assert written == 1
    assert abs(_mean(prepared.image2) - 200) < 3
    entry = _recorded(prepared)["00000002.jpg"]
    assert entry == {"keyframe_id": prepared.kid2, "source": RAW,
                     "frame": str(prepared.raw), "sha1": _sha1(prepared.image2)}
    # ... and the next solve keeps it: its provenance is now the plan
    assert _prepare(prepared)[1] == 0


def test_an_image_from_another_raw_frame_is_rewritten(prepared):
    """RV9-D D2: the image was undistorted from ANOTHER capture's frame of the same name; the
    plan names this keyframe's own. Rewritten."""
    GS.write_sources_records(prepared.workspace, {prepared.kid2: str(prepared.other)})
    prepared.image2.unlink()
    _prepare(prepared)                                    # the image a by-name run left
    assert abs(_mean(prepared.image2) - 90) < 3
    images = {f"{i:08d}.jpg": _redacted_entry(prepared, i) for i in range(N)}
    images["00000002.jpg"] = {"keyframe_id": prepared.kid2, "source": RAW,
                              "frame": str(prepared.other), "sha1": _sha1(prepared.image2)}
    _record(prepared, images)
    GS.write_sources_records(prepared.workspace, {prepared.kid2: str(prepared.raw)})
    assert _prepare(prepared)[1] == 1
    assert abs(_mean(prepared.image2) - 200) < 3


def test_an_image_of_unknown_provenance_is_rewritten_when_a_raw_frame_is_planned(prepared):
    """A record exists (a re-finish prepared this directory) but names nothing for this
    image: unknown provenance, and a raw frame is planned -- rewritten. The images whose
    stored copy is planned and recorded are kept."""
    _record(prepared, {f"{i:08d}.jpg": _redacted_entry(prepared, i) for i in range(N) if i != 2})
    GS.write_sources_records(prepared.workspace, {prepared.kid2: str(prepared.raw)})
    assert _prepare(prepared)[1] == 1
    assert abs(_mean(prepared.image2) - 200) < 3


def test_a_corrupt_record_is_present_and_names_nothing(prepared):
    (prepared.workspace.root / RECORD_FILE).write_text("{torn", "utf-8")
    GS.write_sources_records(prepared.workspace, {prepared.kid2: str(prepared.raw)})
    assert _prepare(prepared)[1] == N        # every image of unknown provenance is rewritten
    assert abs(_mean(prepared.image2) - 200) < 3


def test_a_raw_image_is_rewritten_when_the_stored_copy_is_planned(prepared):
    """RV9-D D3: the carried-back image is a raw frame, but this keyframe's plan is its
    stored (redacted) copy -- its raw frame is not found by identity. Rewritten from the copy."""
    stored = _mean(prepared.image2)
    GS.write_sources_records(prepared.workspace, {prepared.kid2: str(prepared.raw)})
    prepared.image2.unlink()
    _prepare(prepared)                                    # a raw image, as the walk left it
    assert abs(_mean(prepared.image2) - 200) < 3
    images = {f"{i:08d}.jpg": _redacted_entry(prepared, i) for i in range(N)}
    images["00000002.jpg"] = {"keyframe_id": prepared.kid2, "source": RAW,
                              "frame": str(prepared.raw), "sha1": _sha1(prepared.image2)}
    _record(prepared, images)
    GS.write_sources_records(prepared.workspace, {})     # this re-finish: no raw frame for it
    assert _prepare(prepared)[1] == 1
    assert abs(_mean(prepared.image2) - stored) < 3
    assert _recorded(prepared)["00000002.jpg"]["source"] == \
        REDACTED


def test_an_entry_whose_sha1_is_not_the_files_is_rewritten(prepared):
    images = {f"{i:08d}.jpg": _redacted_entry(prepared, i) for i in range(N)}
    images["00000002.jpg"]["sha1"] = "0" * 40
    _record(prepared, images)
    before = prepared.image2.read_bytes()
    assert _prepare(prepared)[1] == 1
    assert prepared.image2.read_bytes() == before, "the same planned copy, the same bytes"
    assert _recorded(prepared)["00000002.jpg"]["sha1"] == _sha1(prepared.image2)


def test_a_matching_entry_keeps_the_image_whatever_the_path_spelling(prepared, monkeypatch, tmp_path):
    """A raw frame recorded relative (the builder's `sources.json`) and planned absolute, or
    the other way round, is one frame."""
    monkeypatch.setenv(GS.SOURCES_ROOT_ENV, str(tmp_path))
    relative = "capA/frames/00000002.jpg"
    GS.write_sources_records(prepared.workspace, {prepared.kid2: relative})
    prepared.image2.unlink()
    _prepare(prepared)
    images = {f"{i:08d}.jpg": _redacted_entry(prepared, i) for i in range(N)}
    images["00000002.jpg"] = {"keyframe_id": prepared.kid2, "source": RAW,
                              "frame": str(prepared.raw), "sha1": _sha1(prepared.image2)}
    _record(prepared, images)
    assert _prepare(prepared)[1] == 0


def test_a_rewritten_image_misses_the_mask_cache_and_the_frozen_matching_and_loses_its_features(
        walked, colmap, tmp_path):
    """A stale image must not survive through the caches keyed by its SHA-1, nor through
    the walk database's features: COLMAP skips a name that has keypoints. A masked, seeded
    final solve freezes the matching and caches every mask; then a re-finish's record says
    image 00000002.jpg came from its stored copy while this solve plans its raw frame. The
    next solve rewrites it, computes its mask afresh (the other three hit the cache), matches
    again ("1 solver images changed"), and the walk database has lost that image's keypoints
    and every pair touching it (image id 3: pairs (2,3) and (3,4))."""
    first = _masked(walked, StubDetector(), seed=7)
    assert first["transients"]["computed"] == N and first["solve"]["matching"] == GS.MATCHING_MATCHED
    walked.raw = _frame(tmp_path / "capA" / "frames" / "00000002.jpg", 200)
    kid2 = f"{walked.session_id}:00000002"
    _record(walked, {f"{i:08d}.jpg": _redacted_entry(walked, i) for i in range(N)})
    GS.write_sources_records(walked.workspace, {kid2: str(walked.raw)})
    colmap.log.clear()
    second = _masked(walked, StubDetector(), seed=7)
    assert abs(_mean(walked.workspace.images_dir / "00000002.jpg") - 200) < 3
    assert second["transients"]["computed"] == 1 and second["transients"]["cache_hits"] == N - 1
    assert second["solve"]["matching"] == GS.MATCHING_MATCHED
    assert "1 solver images changed" in second["solve"]["matching_detail"]
    rewritten = second["solve"]["solver_images_rewritten"]
    assert rewritten["count"] == 1 and rewritten["examples"] == ["00000002.jpg"]
    assert rewritten["features_cleared"] == {"images": 1, "pairs": 2}
    con = sqlite3.connect(f"file:{walked.workspace.database_path.as_posix()}?mode=ro", uri=True)
    try:
        assert con.execute("select count(*) from keypoints where image_id = 3").fetchone()[0] == 0
        assert con.execute("select count(*) from images where image_id = 3").fetchone()[0] == 1
        pairs = {p for (p,) in con.execute("select pair_id from matches")}
        assert pairs == {1 * _B + 2}
    finally:
        con.close()


def test_a_default_solve_writes_no_provenance_record(walked, colmap):
    GS.solve(walked.store, walked.world_id, walked.session_id, final=True)
    GS.solve(walked.store, walked.world_id, walked.session_id, final=False)
    assert not (walked.workspace.root / RECORD_FILE).exists()


# ---------------------------------------------------------------------------
# LOW: a consensus draw's own seed and timing, and its scratch


def test_a_consensus_draw_carries_its_own_seed_and_mapping_time_and_leaves_no_model(
        session, colmap, monkeypatch):
    _prepare(session)
    ws = session.workspace
    stale = ws.root / GS.CONSENSUS_SPARSE_DIRNAME / "seed-99"
    stale.mkdir(parents=True)
    (stale / "images.bin").write_bytes(b"x" * 64)           # a killed consensus's leftover

    def mapped(pycolmap, database_path, workspace, sparse_dir, keyframes, *, seed, threads,
               input_digest, min_image_observations, camera):
        Path(sparse_dir).mkdir(parents=True, exist_ok=True)
        (Path(sparse_dir) / "points3D.bin").write_bytes(b"\0" * 1024)   # the draw's model
        monkeypatch.setattr(time, "perf_counter", lambda: 12.5)        # 2.5 s of mapping
        return GS.Solution(
            solver="glomap", solved_at=1.0, input_digest=input_digest, keyframe_ids=[], poses={},
            components=[], xyz=np.zeros((0, 3), np.float32), rgb=np.zeros((0, 3), np.uint8),
            component=np.zeros(0, np.int32), first_keyframe=np.zeros(0, np.int32),
            track_length=np.zeros(0, np.int32), error=np.zeros(0, np.float32),
            observations=np.zeros((0, 3), np.int32), camera=camera.to_json_dict())

    monkeypatch.setattr(GS, "_map_candidate", mapped)
    base = GS.Solution(
        solver="glomap", solved_at=0.0, input_digest="d", keyframe_ids=[], poses={}, components=[],
        xyz=np.zeros((0, 3), np.float32), rgb=np.zeros((0, 3), np.uint8),
        component=np.zeros(0, np.int32), first_keyframe=np.zeros(0, np.int32),
        track_length=np.zeros(0, np.int32), error=np.zeros(0, np.float32),
        observations=np.zeros((0, 3), np.int32),
        camera=json.loads(ws.camera_path.read_text(encoding="utf-8")),
        transients={"state": "applied"},
        solve={"seed": 5, "verification_seed": 5, "database": "database.masked.x.db", "threads": 1},
        timing={"prepare_s": 1.0, "match_s": 2.0, "map_s": 99.0})
    map_draw = GS.frozen_draw_mapper(session.store, session.world_id, session.session_id,
                                     ws.database_path, base)
    assert not (ws.root / GS.CONSENSUS_SPARSE_DIRNAME).exists(), "the leftover is cleared"
    monkeypatch.setattr(time, "perf_counter", lambda: 10.0)
    draw = map_draw(7)
    assert draw.solve == {"seed": 7, "verification_seed": 5, "database": "database.masked.x.db",
                          "threads": 1}
    assert draw.timing == {"prepare_s": 1.0, "match_s": 2.0, "map_s": 2.5}
    assert draw.transients == base.transients
    assert base.solve["seed"] == 5 and base.timing["map_s"] == 99.0, "draw 0's are untouched"
    assert not (ws.root / GS.CONSENSUS_SPARSE_DIRNAME).exists(), "no draw's model is left"


# ---------------------------------------------------------------------------
# M-3: the stop reaches the consensus, and a stop never loses the finish


@pytest.fixture
def counted(engines, monkeypatch):
    """How many draws the mapper was asked for (the engines' fake mapper, counted)."""
    calls = []
    real = GS._map_candidate

    def counting(*a, **kw):
        calls.append(kw.get("seed"))
        return real(*a, **kw)

    monkeypatch.setattr(GS, "_map_candidate", counting)
    return calls


def _finish(walk, **kw):
    return GS.solve(walk.store, walk.world_id, walk.session_id, final=True, masks=True, seed=5,
                    gate=True, consensus=3, transient_backend_factory=StubDetector(),
                    mask_device_probe=lambda: None, input_digest="walk-digest", **kw)


@pytest.mark.parametrize("after_draws", [1, 2])
def test_a_stop_during_the_consensus_publishes_draw_0_deferred(walk, counted, colmap, after_draws):
    """A stop after draw 0 is mapped (before any further draw), or after draw 1 (between
    draws): the finish is NOT lost. Draw 0 is published as one draw publishes it -- its own
    seed -- with the consensus `deferred`, and the re-gate in place owes it."""
    summary = _finish(walk, should_stop=lambda: len(counted) >= after_draws)
    assert counted == [5, 6][:after_draws], "no draw is mapped after the stop"
    c = summary["gate"]["consensus"]
    assert c["state"] == CP.CONSENSUS_DEFERRED
    published = json.loads(walk.workspace.solution_path.read_text(encoding="utf-8"))
    assert published["solve"]["seed"] == 5
    assert published["gate"]["consensus"]["state"] == CP.CONSENSUS_DEFERRED
    assert CP.regate_owed(walk.store, walk.world_id, walk.session_id) is not None


def test_without_a_stop_the_consensus_runs_every_draw(walk, counted, colmap):
    summary = _finish(walk, should_stop=lambda: False)
    assert counted == [5, 6, 7]
    assert summary["gate"]["consensus"]["state"] != CP.CONSENSUS_DEFERRED


def test_without_a_callers_stop_the_publish_call_is_todays(walk, engines, colmap, monkeypatch):
    seen = []
    real = CP.gate_and_publish

    def recording(*a, **kw):
        seen.append(sorted(kw))
        return real(*a, **kw)

    monkeypatch.setattr(CP, "gate_and_publish", recording)
    _finish(walk)
    assert seen == [sorted(["final", "gate", "database_path", "keyframes", "write", "consensus"])]
