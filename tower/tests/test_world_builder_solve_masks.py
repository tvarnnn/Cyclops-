"""Transient masks and the seeded single-thread solve in the FINAL global solve.

Modules: `tower/world_builder/solve_masks.py`, `global_solve.solve`, the two
settings in `tower/config.py`, and the surface's reuse in `transients.py`.
Contract: WORLD-BUILDER-COMPONENTS §2.5 (`transients.state`, `transients.rule`,
`solve.seed`, `solve.threads`).

No pycolmap and no model is used here. pycolmap is a FAKE module that records
every option the solve sets and every call it makes -- the trace IS the input
to pycolmap -- and the transient detector is a stub backend. What is tested:

  * with the settings off, the trace is exactly today's, call for call;
  * the seeded solve sets every seed and maps on one thread;
  * masks reach extraction as COLMAP masks, into their own database;
  * a missing or corrupt cache entry is recomputed, never fatal;
  * a detector that cannot run (no GPU, load failure, out of memory) leaves
    a solve that completes UNMASKED and says `unavailable`, never `applied`;
  * the surface stage takes the solver's masks only for a raw-imagery build.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

from tower.world_builder import global_solve
from tower.world_builder import solve_masks as SM
from tower.world_builder import transients as T
from tower.world_builder.records import Keyframe

WIDTH, HEIGHT = 64, 48
N = 4
HAND_RECT = (4, 30, 2, 26)   # y0, y1, x0, x1: a "hand" in every image


# ---------------------------------------------------------------------------
# a pycolmap that records what it is given
# ---------------------------------------------------------------------------


class _Opts:
    """An options object that logs every assignment, at any depth."""

    def __init__(self, kind, log, prefix=""):
        object.__setattr__(self, "_kind", kind)
        object.__setattr__(self, "_log", log)
        object.__setattr__(self, "_prefix", prefix)
        object.__setattr__(self, "_children", {})
        object.__setattr__(self, "_values", {})

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        values = object.__getattribute__(self, "_values")
        if name in values:
            return values[name]
        children = object.__getattribute__(self, "_children")
        if name not in children:
            children[name] = _Opts(self._kind, self._log, f"{self._prefix}{name}.")
        return children[name]

    def __setattr__(self, name, value):
        self._values[name] = value
        self._log.append(("set", self._kind, self._prefix + name, value))


class FakeColmap(types.ModuleType):
    OPTION_KINDS = ("ImageReaderOptions", "FeatureExtractionOptions", "FeatureMatchingOptions",
                    "SequentialPairingOptions", "TwoViewGeometryOptions", "ImportedPairingOptions",
                    "GlobalPipelineOptions", "IncrementalPipelineOptions")

    def __init__(self):
        super().__init__("pycolmap")
        self.log = []
        self.__version__ = "fake"
        self.logging = types.SimpleNamespace(minloglevel=0)
        self.CameraMode = types.SimpleNamespace(SINGLE="SINGLE")
        self.masks_read = {}          # image name -> the mask array extraction would read
        self.database_existed = []    # at each extract_features call
        for kind in self.OPTION_KINDS:
            setattr(self, kind, (lambda k: (lambda: _Opts(k, self.log)))(kind))

    def extract_features(self, database_path, image_path, image_names=None, camera_mode=None,
                         reader_options=None, extraction_options=None):
        self.database_existed.append(Path(database_path).exists())
        self.log.append(("call", "extract_features", Path(database_path).name,
                         tuple(image_names), camera_mode))
        mask_path = reader_options._values.get("mask_path")
        if mask_path is not None:
            for name in image_names:
                # COLMAP's convention: <mask_path>/<image name>.png
                png = Path(mask_path) / f"{name}.png"
                assert png.is_file(), f"no COLMAP mask for {name}"
                self.masks_read[name] = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
        Path(database_path).touch()

    def match_sequential(self, database_path, **kwargs):
        self.log.append(("call", "match_sequential", Path(database_path).name, tuple(sorted(kwargs))))

    def match_image_pairs(self, database_path, **kwargs):
        listed = Path(kwargs["pairing_options"]._values["match_list_path"])
        self.log.append(("call", "match_image_pairs", Path(database_path).name,
                         tuple(sorted(kwargs)), listed.read_text(encoding="utf-8")))

    def verify_matches(self, database_path, pairs_path, options=None):
        self.log.append(("call", "verify_matches", Path(database_path).name,
                         Path(pairs_path).read_text(encoding="utf-8"), options))

    def set_random_seed(self, seed):
        self.log.append(("call", "set_random_seed", seed))

    def global_mapping(self, database_path, image_path, output_path, options=None):
        self.log.append(("call", "global_mapping", Path(database_path).name))
        return {}

    def incremental_mapping(self, database_path, image_path, output_path, options=None):
        self.log.append(("call", "incremental_mapping", Path(database_path).name))
        return {}

    def sets(self, kind):
        return {path: value for op, k, path, value in
                (e for e in self.log if e[0] == "set") if k == kind}

    def calls(self, name):
        return [e for e in self.log if e[0] == "call" and e[1] == name]


@pytest.fixture
def colmap(monkeypatch):
    fake = FakeColmap()
    monkeypatch.setitem(sys.modules, "pycolmap", fake)
    return fake


@pytest.fixture(autouse=True)
def _settings_unset(monkeypatch):
    monkeypatch.delenv("TOWER_WORLD_SOLVE_MASKS", raising=False)
    monkeypatch.delenv("TOWER_WORLD_SOLVE_SEED", raising=False)


# ---------------------------------------------------------------------------
# a real session: four keyframe images, a calibration without distortion
# ---------------------------------------------------------------------------


@pytest.fixture
def session(tmp_path):
    from tower.world_builder.engine import WorldBuilderEngine
    from tower.world_builder.records import CameraIntrinsics
    from tower.world_builder.store import WorldStore

    store = WorldStore(tmp_path / "worlds")
    engine = WorldBuilderEngine(store)
    world_id = engine.create_world()
    session_id = engine.start_session(
        world_id, frame_source="synthetic",
        intrinsics=CameraIntrinsics(source="self_calibrated", model="pinhole_radtan",
                                    fx=50.0, fy=50.0, cx=WIDTH / 2, cy=HEIGHT / 2,
                                    dist_coeffs=(0.0, 0.0, 0.0, 0.0, 0.0),
                                    calibrated_width=WIDTH, calibrated_height=HEIGHT))
    rng = np.random.default_rng(0)
    for i in range(N):
        frame = rng.integers(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8)
        ok, buf = cv2.imencode(".jpg", frame)
        assert ok
        store.write_keyframe_image(world_id, session_id, f"{i:08d}.jpg", buf.tobytes())
        store.append_keyframe(world_id, Keyframe(
            keyframe_id=f"{session_id}:{i:08d}", session_id=session_id, source_seq=i,
            received_at=float(i), image_relpath=f"images/{i:08d}.jpg",
            width=WIDTH, height=HEIGHT, byte_count=1, segment_index=0))
    engine.stop_session("stopped")
    return types.SimpleNamespace(store=store, world_id=world_id, session_id=session_id,
                                 workspace=global_solve.workspace_for(store, world_id, session_id))


def _is_masked_db(name) -> bool:
    """This solve's own masked database: `database.masked.p<pid>.<8hex>.db` (review V5, M1-5)."""
    import re

    return bool(re.fullmatch(r"database\.masked\.p\d+\.[0-9a-f]{8}\.db", str(name)))


def _solve(s, **kw):
    return global_solve.solve(s.store, s.world_id, s.session_id, **kw)


def _solution_json(s) -> dict:
    return json.loads(s.workspace.solution_path.read_text(encoding="utf-8"))


class StubDetector:
    """A backend factory: `gdsam` marks HAND_RECT as a hand, `oneformer` marks
    nothing. It can refuse at probe, fail at model load, or run out of memory
    at inference."""

    def __init__(self, reason=None, load_error=None, run_error=None, rect=HAND_RECT):
        self.reason, self.load_error, self.run_error, self.rect = reason, load_error, run_error, rect
        self.calls = []

    def __call__(self, component):
        stub = self

        class Backend(T.ComponentBackend):
            def probe(self):
                return stub.reason

            def run(self, items, params, emit, should_stop=None):
                if stub.load_error is not None:
                    raise stub.load_error
                for i, rgb, _unobserved in items:
                    stub.calls.append((component, i))
                    if stub.run_error is not None:
                        raise stub.run_error
                    hand = np.zeros(rgb.shape[:2], bool)
                    if component == T.COMPONENT_GDSAM:
                        y0, y1, x0, x1 = stub.rect
                        hand[y0:y1, x0:x1] = True
                    emit(i, hand, np.zeros(rgb.shape[:2], bool), 0.001)
                return {"frames": len(items)}

        Backend.component = component
        return Backend()


def _gpu():
    return None


def _masked(s, stub, **kw):
    return _solve(s, final=True, masks=True, transient_backend_factory=stub,
                  mask_device_probe=kw.pop("device_probe", _gpu), **kw)


# ---------------------------------------------------------------------------
# 1. the settings
# ---------------------------------------------------------------------------


def test_both_settings_default_off(monkeypatch):
    from tower.config import get_settings, world_solve_masks_setting, world_solve_seed_setting

    assert world_solve_masks_setting() is False
    assert world_solve_seed_setting() is None
    settings = get_settings()
    assert settings.world_solve_masks is False
    assert settings.world_solve_seed is None


@pytest.mark.parametrize("value,expected", [("1", True), ("on", True), ("true", True),
                                            ("0", False), ("off", False), ("", False)])
def test_the_mask_setting_reads_like_every_other_flag(monkeypatch, value, expected):
    from tower.config import world_solve_masks_setting

    monkeypatch.setenv("TOWER_WORLD_SOLVE_MASKS", value)
    assert world_solve_masks_setting() is expected


@pytest.mark.parametrize("value,expected", [("7", 7), ("0", 0), (" 12 ", 12), ("", None),
                                            ("off", None), ("-3", None), ("seven", None)])
def test_the_seed_setting_is_a_non_negative_integer_or_off(monkeypatch, value, expected):
    from tower.config import world_solve_seed_setting

    monkeypatch.setenv("TOWER_WORLD_SOLVE_SEED", value)
    assert world_solve_seed_setting() == expected


def test_only_the_final_solve_reads_the_settings(monkeypatch):
    monkeypatch.setenv("TOWER_WORLD_SOLVE_MASKS", "1")
    monkeypatch.setenv("TOWER_WORLD_SOLVE_SEED", "7")
    resolve = global_solve.resolve_run_options
    assert resolve(final=True, masks=None, seed=global_solve.FROM_SETTINGS) == (True, 7)
    # The walk's background solves keep today's recipe whatever the settings say.
    assert resolve(final=False, masks=None, seed=global_solve.FROM_SETTINGS) == (False, None)
    # An explicit argument wins either way.
    assert resolve(final=True, masks=False, seed=None) == (False, None)
    assert resolve(final=False, masks=True, seed=3) == (True, 3)


# ---------------------------------------------------------------------------
# 2. off is today, input for input
# ---------------------------------------------------------------------------


def _todays_trace(s) -> list:
    """What `global_solve.solve` handed pycolmap at 17e6d3d, before this change,
    for a solve with default arguments: written out, so the test pins the
    inputs rather than whatever the code happens to do now."""
    cam = json.loads(s.workspace.camera_path.read_text(encoding="utf-8"))
    names = tuple(f"{i:08d}.jpg" for i in range(N))
    return [
        ("set", "ImageReaderOptions", "camera_model", "PINHOLE"),
        ("set", "ImageReaderOptions", "camera_params",
         ",".join(str(v) for v in (cam["fx"], cam["fy"], cam["cx"], cam["cy"]))),
        ("set", "FeatureExtractionOptions", "num_threads", -1),
        ("set", "FeatureExtractionOptions", "sift.max_num_features", global_solve.MAX_FEATURES),
        ("call", "extract_features", "database.db", names, "SINGLE"),
        ("set", "FeatureMatchingOptions", "num_threads", -1),
        ("set", "SequentialPairingOptions", "overlap", global_solve.SEQUENTIAL_OVERLAP),
        ("set", "SequentialPairingOptions", "quadratic_overlap", False),
        ("set", "SequentialPairingOptions", "loop_detection", False),
        ("call", "match_sequential", "database.db", ("matching_options", "pairing_options")),
        ("set", "GlobalPipelineOptions", "num_threads", -1),
        ("set", "GlobalPipelineOptions", "mapper.bundle_adjustment.refine_focal_length", False),
        ("set", "GlobalPipelineOptions", "mapper.bundle_adjustment.refine_principal_point", False),
        ("set", "GlobalPipelineOptions", "mapper.bundle_adjustment.refine_extra_params", False),
        ("call", "global_mapping", "database.db"),
        ("set", "IncrementalPipelineOptions", "num_threads", -1),
        ("set", "IncrementalPipelineOptions", "ba_refine_focal_length", False),
        ("set", "IncrementalPipelineOptions", "ba_refine_principal_point", False),
        ("set", "IncrementalPipelineOptions", "ba_refine_extra_params", False),
        ("call", "incremental_mapping", "database.db"),
    ]


@pytest.mark.parametrize("final,env", [
    (True, {}),                                                        # settings unset
    (True, {"TOWER_WORLD_SOLVE_MASKS": "0", "TOWER_WORLD_SOLVE_SEED": ""}),
    (True, {"TOWER_WORLD_SOLVE_MASKS": "off", "TOWER_WORLD_SOLVE_SEED": "off"}),
    (False, {"TOWER_WORLD_SOLVE_MASKS": "1", "TOWER_WORLD_SOLVE_SEED": "7"}),  # background
])
def test_with_the_settings_off_pycolmap_gets_exactly_todays_inputs(session, colmap, monkeypatch,
                                                                    final, env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    summary = _solve(session, final=final)
    assert summary["solved"] is True
    assert colmap.log == _todays_trace(session)
    assert not (session.workspace.root / SM.MASKS_DIRNAME).exists()
    assert not SM.masked_database_path(session.workspace).exists()
    meta = _solution_json(session)
    assert meta["schema_version"] == global_solve.SOLUTION_SCHEMA_VERSION == 1
    assert meta["transients"]["state"] == SM.RECORD_UNAVAILABLE
    assert meta["transients"]["requested"] is False
    assert meta["transients"]["rule"] is None
    assert meta["solve"]["seed"] is None and meta["solve"]["threads"] == -1


# ---------------------------------------------------------------------------
# 3. the seeded single-thread solve
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("how", ["argument", "setting"])
def test_a_seeded_solve_sets_every_seed_and_maps_on_one_thread(session, colmap, monkeypatch, how):
    if how == "setting":
        monkeypatch.setenv("TOWER_WORLD_SOLVE_SEED", "7")
        summary = _solve(session, final=True)
    else:
        summary = _solve(session, final=True, seed=7)
    assert summary["solved"] is True
    g = colmap.sets("GlobalPipelineOptions")
    assert g["num_threads"] == 1 and g["mapper.num_threads"] == 1
    for path in ("random_seed", "mapper.random_seed", "mapper.rotation_averaging.random_seed",
                 "mapper.global_positioning.random_seed", "mapper.retriangulation.random_seed"):
        assert g[path] == 7, path
    inc = colmap.sets("IncrementalPipelineOptions")
    assert inc["num_threads"] == 1 and inc["mapper.num_threads"] == 1
    assert inc["random_seed"] == inc["mapper.random_seed"] == inc["triangulation.random_seed"] == 7
    assert colmap.sets("TwoViewGeometryOptions") == {"ransac.random_seed": 7}
    assert colmap.calls("match_sequential")[0][3] == (
        "matching_options", "pairing_options", "verification_options")
    # pycolmap's own seed, before each mapper.
    order = [e[1] for e in colmap.log if e[0] == "call"]
    assert order == ["extract_features", "match_sequential", "set_random_seed", "global_mapping",
                     "set_random_seed", "incremental_mapping"]
    assert all(e[2] == 7 for e in colmap.calls("set_random_seed"))
    # Extraction and matching keep their threads.
    assert colmap.sets("FeatureExtractionOptions")["num_threads"] == -1
    assert colmap.sets("FeatureMatchingOptions")["num_threads"] == -1
    solve = _solution_json(session)["solve"]
    assert solve["seed"] == 7 and solve["threads"] == 1 and solve["seeded"] is True
    assert solve["verification_seed"] == 7 and solve["extraction_threads"] == -1
    assert summary["solve"] == solve


# ---------------------------------------------------------------------------
# 4. masks reach extraction, into their own database
# ---------------------------------------------------------------------------


def test_a_masked_solve_masks_every_solver_image_before_extraction(session, colmap):
    stub = StubDetector()
    summary = _masked(session, stub)
    assert summary["solved"] is True
    ws = session.workspace
    names = [f"{i:08d}.jpg" for i in range(N)]

    # The reader is pointed at the masks and extraction writes the MASKED database.
    assert colmap.sets("ImageReaderOptions")["mask_path"] == str(ws.root / SM.MASKS_DIRNAME)
    (extracted,) = [c[2] for c in colmap.calls("extract_features")]
    assert _is_masked_db(extracted)
    assert colmap.calls("match_sequential")[0][2] == extracted
    assert colmap.calls("global_mapping")[0][2] == extracted
    assert not ws.database_path.exists(), "the background solves' database is not touched"

    # Every image's COLMAP mask: 0 (never extracted) on the hand, 255 away from it.
    assert sorted(colmap.masks_read) == names
    for name, m in colmap.masks_read.items():
        y0, y1, x0, x1 = HAND_RECT
        assert (m[y0:y1, x0:x1] == 0).all(), name
        assert (m[:, 50:] == 255).all(), name

    record = _solution_json(session)["transients"]
    assert record["state"] == SM.RECORD_APPLIED
    assert record["rule"] == T.TransientParams().rule_id()
    assert record["requested"] is True and record["extraction_masked"] is True
    assert record["images"] == record["images_masked"] == N
    assert record["images_unmasked"] == 0
    assert record["computed"] == N and record["cache_hits"] == 0
    assert _is_masked_db(record["database"])
    assert summary["transients"] == record
    assert "masks_s" in _solution_json(session)["timing"]


def _walk_database(path: Path, pairs):
    """A walk `database.db` with just the two tables the masked solve reads."""
    import sqlite3

    path.unlink(missing_ok=True)
    con = sqlite3.connect(str(path))
    con.execute("create table images (image_id integer primary key, name text)")
    con.execute("create table two_view_geometries (pair_id integer primary key, rows integer)")
    for i in range(N):
        con.execute("insert into images values (?, ?)", (i + 1, f"{i:08d}.jpg"))
    con.execute("insert into images values (?, ?)", (99, "gone.jpg"))
    for a, b, inliers in pairs:
        con.execute("insert into two_view_geometries values (?, ?)",
                    (a * 2147483647 + b, inliers))
    con.commit()
    con.close()


@pytest.mark.parametrize("masked,seed", [(False, None), (True, 5)])
def test_the_final_solve_matches_the_relocalizers_revisit_links(session, colmap, monkeypatch,
                                                                 masked, seed):
    from tower.world_builder import relocalizer

    seen_dirs = []

    def links(session_dir):
        seen_dirs.append(Path(session_dir))
        return [("00000000.jpg", "00000003.jpg"), ("00000001.jpg", "not-a-solver-image.jpg")]

    monkeypatch.setattr(relocalizer, "revisit_pairs", links)
    monkeypatch.setattr(global_solve, "_verified_pair_count", lambda db, pairs: len(pairs))
    if masked:
        summary = _masked(session, StubDetector(), seed=seed)
    else:
        summary = _solve(session, final=True, seed=seed)
    assert seen_dirs == [session.store.session_dir(session.world_id, session.session_id)]
    calls = colmap.calls("match_image_pairs")
    assert len(calls) == 1
    _call, _name, database, kwargs, listed = calls[0]
    assert listed == "00000000.jpg 00000003.jpg\n", "only pairs of images this solve has"
    assert (_is_masked_db(database) if masked else database == "database.db")
    assert ("verification_options" in kwargs) is (seed is not None)
    order = [e[1] for e in colmap.log if e[0] == "call"]
    assert order.index("match_sequential") < order.index("match_image_pairs") < \
        order.index("global_mapping")
    record = summary["solve"]["revisit_pairs"]
    assert record == {"listed": 1, "verified": 1, "detail": None}
    assert _solution_json(session)["solve"]["revisit_pairs"] == record


def test_revisit_links_are_for_the_final_solve_only(session, colmap, monkeypatch):
    from tower.world_builder import relocalizer

    monkeypatch.setattr(relocalizer, "revisit_pairs",
                        lambda d: [("00000000.jpg", "00000003.jpg")])
    _solve(session, final=False)
    assert colmap.calls("match_image_pairs") == []


def test_no_revisit_links_is_todays_solve_exactly(session, colmap, monkeypatch):
    from tower.world_builder import relocalizer

    monkeypatch.setattr(relocalizer, "revisit_pairs", lambda d: [])
    summary = _solve(session, final=True)
    assert colmap.log == _todays_trace(session)
    assert summary["solve"]["revisit_pairs"] == {"listed": 0, "verified": 0, "detail": None}


def test_unreadable_revisit_links_are_recorded_not_fatal(session, colmap, monkeypatch):
    from tower.world_builder import relocalizer

    def broken(_d):
        raise ValueError("torn journal")

    monkeypatch.setattr(relocalizer, "revisit_pairs", broken)
    summary = _solve(session, final=True)
    assert summary["solved"] is True
    assert colmap.log == _todays_trace(session)
    assert "torn journal" in summary["solve"]["revisit_pairs"]["detail"]


def test_a_verified_revisit_is_counted_by_colmaps_pair_id(tmp_path):
    """COLMAP stores a pair under min(id) * 2147483647 + max(id), whichever
    way round it was listed."""
    db = tmp_path / "database.db"
    _walk_database(db, [(1, 3, 40), (2, 4, 10)])
    pairs = [("00000002.jpg", "00000000.jpg"),     # listed reversed, verified
             ("00000001.jpg", "00000003.jpg"),     # under the floor
             ("00000000.jpg", "nowhere.jpg")]
    assert global_solve._verified_pair_count(db, pairs) == 1
    assert global_solve._verified_pair_count(tmp_path / "absent.db", pairs) in (None, 0)


# ---------------------------------------------------------------------------
# 4b. the walk database, filtered (arm A1h) -- the path when one exists
# ---------------------------------------------------------------------------

_B = 2147483647
# Four keypoints per image: 0 and 1 inside HAND_RECT (masked), 2 and 3 outside.
_KEYPOINTS = np.array([[10.5, 10.5, 1, 0, 0, 1], [20.5, 15.5, 1, 0, 0, 1],
                       [50.5, 40.5, 1, 0, 0, 1], [60.5, 5.5, 1, 0, 0, 1]], np.float32)


def _blob(rows):
    a = np.asarray(rows, np.uint32).reshape(-1, 2)
    return len(a), 2, a.tobytes()


def _colmap_walk_database(path: Path):
    """A walk `database.db` in COLMAP's schema (the columns the filter reads)."""
    import sqlite3

    path.unlink(missing_ok=True)
    con = sqlite3.connect(str(path))
    con.executescript(
        "create table images (image_id integer primary key, name text, camera_id integer);"
        "create table keypoints (image_id integer primary key, rows integer, cols integer, data blob);"
        "create table matches (pair_id integer primary key, rows integer, cols integer, data blob);"
        "create table two_view_geometries (pair_id integer primary key, rows integer, cols integer,"
        " data blob, config integer);")
    for i in range(N):
        con.execute("insert into images values (?, ?, 1)", (i + 1, f"{i:08d}.jpg"))
        con.execute("insert into keypoints values (?, 4, 6, ?)", (i + 1, _KEYPOINTS.tobytes()))
    # (1,2): two raw matches on masked keypoints, one of them a verified inlier.
    con.execute("insert into matches values (?, ?, ?, ?)", (1 * _B + 2, *_blob([[0, 0], [2, 2], [3, 3], [1, 2]])))
    con.execute("insert into two_view_geometries values (?, ?, ?, ?, 2)", (1 * _B + 2, *_blob([[0, 0], [2, 2], [3, 3]])))
    # (3,4): clean.
    con.execute("insert into matches values (?, ?, ?, ?)", (3 * _B + 4, *_blob([[2, 3], [3, 2]])))
    con.execute("insert into two_view_geometries values (?, ?, ?, ?, 2)", (3 * _B + 4, *_blob([[2, 3], [3, 2]])))
    # (2,3): clean raw matches, but an inlier on a masked keypoint.
    con.execute("insert into matches values (?, ?, ?, ?)", (2 * _B + 3, *_blob([[2, 2], [3, 3]])))
    con.execute("insert into two_view_geometries values (?, ?, ?, ?, 2)", (2 * _B + 3, *_blob([[0, 2], [3, 3]])))
    con.commit()
    con.close()


def _rows(db: Path, table: str) -> dict:
    import sqlite3

    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        return {pid: np.frombuffer(data, np.uint32).reshape(-1, 2).tolist()
                for pid, data in con.execute(f"select pair_id, data from {table}")}
    finally:
        con.close()


@pytest.fixture
def walked(session, colmap):
    """The session after its walk: undistorted images and a walk database."""
    _solve(session, final=True)
    _colmap_walk_database(session.workspace.database_path)
    colmap.log.clear()
    return session


@pytest.mark.parametrize("seed", [None, 5])
def test_with_a_walk_database_the_solve_maps_a_filtered_copy_of_it(walked, colmap, seed):
    ws = walked.workspace
    before = ws.database_path.read_bytes()
    summary = _masked(walked, StubDetector(), seed=seed)
    assert summary["solved"] is True

    # The walk database is extracted and matched exactly as today: unmasked.
    assert "mask_path" not in colmap.sets("ImageReaderOptions")
    assert [c[2] for c in colmap.calls("extract_features")] == ["database.db"]
    assert colmap.calls("match_sequential")[0][2] == "database.db"
    # ... and never modified by the filter.
    assert ws.database_path.read_bytes() == before

    # The copy loses every match and inlier on a masked keypoint.
    masked = ws.root / summary["transients"]["database"]
    assert _is_masked_db(masked.name)
    matches = _rows(masked, "matches")
    geometries = _rows(masked, "two_view_geometries")
    assert matches[1 * _B + 2] == [[2, 2], [3, 3]]
    assert matches[3 * _B + 4] == [[2, 3], [3, 2]] and matches[2 * _B + 3] == [[2, 2], [3, 3]]
    assert set(geometries) == {3 * _B + 4}, "the touched geometries are gone, the clean one kept"

    # The changed pairs are re-verified, under the solve's own two-view options,
    # and the solve maps the copy.
    (_c, _n, database, listed, options), = colmap.calls("verify_matches")
    assert database == masked.name
    assert sorted(listed.splitlines()) == ["00000000.jpg 00000001.jpg", "00000001.jpg 00000002.jpg"]
    if seed is None:
        assert options._values == {} and options._children == {}
    else:
        assert options.ransac._values == {"random_seed": seed}
    order = [e[1] for e in colmap.log if e[0] == "call"]
    assert order.index("match_sequential") < order.index("verify_matches") < order.index("global_mapping")
    assert colmap.calls("global_mapping")[0][2] == masked.name

    record = summary["transients"]
    assert record["state"] == SM.RECORD_APPLIED
    assert record["masking"] == SM.MASKING_FILTERED == summary["solve"]["masking"]
    assert record["database"] == database
    assert record["walk_database"] == "filtered" == summary["solve"]["walk_database"]
    f = record["filter"]
    assert (f["keypoints_total"], f["keypoints_in_mask"]) == (4 * N, 2 * N)
    assert f["matches_dropped"] == 2 and f["pairs_changed"] == 2 and f["pairs_reverified"] == 2
    assert f["geometries_dropped"] == 1 and f["images_filtered"] == N
    assert _solution_json(walked)["solve"]["masking"] == SM.MASKING_FILTERED


def test_without_a_walk_database_the_masks_go_to_extraction(session, colmap):
    summary = _masked(session, StubDetector())
    assert summary["solve"]["masking"] == SM.MASKING_REEXTRACTED
    assert summary["transients"]["masking"] == SM.MASKING_REEXTRACTED
    assert summary["transients"]["walk_database"] == "absent" == summary["solve"]["walk_database"]
    assert colmap.calls("verify_matches") == []


def test_an_unreadable_walk_database_is_not_filtered(walked, colmap):
    walked.workspace.database_path.write_bytes(b"not a database at all, not even close")
    summary = _masked(walked, StubDetector())
    assert summary["solve"]["masking"] == SM.MASKING_REEXTRACTED
    assert summary["solve"]["walk_database"] == "unusable"
    (extracted,) = [c[2] for c in colmap.calls("extract_features")]
    assert _is_masked_db(extracted)


def test_a_filter_that_fails_falls_back_to_re_extraction_and_says_so(walked, colmap, monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(SM, "filter_walk_database", broken)
    summary = _masked(walked, StubDetector())
    assert summary["solved"] is True
    walk_db, extracted = [c[2] for c in colmap.calls("extract_features")]
    assert walk_db == "database.db" and _is_masked_db(extracted)
    assert colmap.sets("ImageReaderOptions")["mask_path"] == str(walked.workspace.root / SM.MASKS_DIRNAME)
    assert colmap.calls("global_mapping")[0][2] == extracted
    assert summary["solve"]["walk_database"] == "filter-failed"
    record = summary["transients"]
    assert record["masking"] == SM.MASKING_REEXTRACTED == summary["solve"]["masking"]
    assert "disk full" in record["filter_failed"]
    assert record["state"] == SM.RECORD_APPLIED


def test_with_the_settings_off_a_walk_database_is_used_exactly_as_today(walked, colmap):
    before = walked.workspace.database_path.read_bytes()
    summary = _solve(walked, final=True)
    assert colmap.log == _todays_trace(walked)
    assert walked.workspace.database_path.read_bytes() == before
    assert not SM.masked_database_path(walked.workspace).exists()
    assert list(walked.workspace.root.glob("database.masked*")) == []
    assert summary["solve"]["masking"] is None and summary["solve"]["walk_database"] is None


def test_a_filtered_database_is_never_reused_as_a_re_extracted_one(walked, colmap):
    _masked(walked, StubDetector())                      # filtered: record says so
    walked.workspace.database_path.unlink()              # the walk database is gone
    summary = _masked(walked, StubDetector())
    assert summary["solve"]["masking"] == SM.MASKING_REEXTRACTED
    assert summary["transients"]["database_reused"] is False
    assert "walk-database-filtered" in summary["transients"]["database_rebuilt_because"]


def test_the_two_spellings_of_the_masking_paths_agree():
    assert global_solve._MASKING_FILTERED == SM.MASKING_FILTERED
    assert global_solve._MASKING_REEXTRACTED == SM.MASKING_REEXTRACTED


def test_a_second_solve_reuses_the_cache_and_the_masked_database(session, colmap):
    stub = StubDetector()
    _masked(session, stub)
    calls = len(stub.calls)
    summary = _masked(session, stub)
    record = summary["transients"]
    assert len(stub.calls) == calls, "no inference for cached masks"
    assert record["state"] == SM.RECORD_APPLIED
    assert record["cache_hits"] == N and record["computed"] == 0
    assert record["database_reused"] is True
    assert colmap.database_existed == [False, True]


def test_a_missing_or_corrupt_mask_cache_is_recomputed_not_fatal(session, colmap):
    stub = StubDetector()
    _masked(session, stub)
    cdir = SM.cache_dir(session.workspace)
    gdsam = sorted(cdir.glob(f"*.{T.COMPONENT_GDSAM}.npz"))
    oneformer = sorted(cdir.glob(f"*.{T.COMPONENT_ONEFORMER}.npz"))
    assert len(gdsam) == len(oneformer) == N
    gdsam[0].unlink()                           # missing
    oneformer[1].write_bytes(b"not an npz")     # corrupt
    stub.calls.clear()
    summary = _masked(session, stub)
    record = summary["transients"]
    assert summary["solved"] is True
    assert record["state"] == SM.RECORD_APPLIED
    assert record["computed"] == 2 and record["cache_hits"] == N - 2
    assert sorted(c for c, _ in stub.calls) == [T.COMPONENT_GDSAM, T.COMPONENT_ONEFORMER]


def test_an_image_whose_mask_changed_rebuilds_the_masked_database(session, colmap):
    _masked(session, StubDetector())
    # The same image, new bytes: its old keypoints were extracted under the
    # old mask, and COLMAP cannot re-extract one image in place.
    target = session.workspace.images_dir / "00000002.jpg"
    image = cv2.imread(str(target))
    ok, buf = cv2.imencode(".jpg", np.full_like(image, 90))
    target.write_bytes(buf.tobytes())
    summary = _masked(session, StubDetector())
    record = summary["transients"]
    assert record["database_reused"] is False
    assert "00000002.jpg" in record["database_rebuilt_because"]
    assert colmap.database_existed == [False, False], "the stale database was removed first"
    assert record["computed"] == 1 and record["cache_hits"] == N - 1


def test_an_image_the_detector_cannot_read_makes_the_solve_partial(session, colmap):
    _solve(session, final=True)            # undistort once
    colmap.log.clear()
    bad = session.workspace.images_dir / "00000001.jpg"
    bad.write_bytes(b"not a jpeg")
    summary = _masked(session, StubDetector())
    record = summary["transients"]
    assert record["state"] == SM.RECORD_PARTIAL
    assert record["images_masked"] == N - 1 and record["images_unmasked"] == 1
    assert record["unmasked_examples"] == ["00000001.jpg"]
    # An explicit "use everything" mask, not a missing file.
    assert (colmap.masks_read["00000001.jpg"] == 255).all()


# ---------------------------------------------------------------------------
# 5. a detector that cannot run: an unmasked solve, and it says so
# ---------------------------------------------------------------------------


FAILURES = {
    "no-gpu": dict(device_probe=lambda: "no CUDA device, and no CPU fallback"),
    "not-installed": dict(stub=StubDetector(reason="the transient detector needs transformers")),
    "load-unavailable": dict(stub=StubDetector(
        load_error=T.TransientDetectorUnavailable("weights not cached and offline"))),
    "load-crash": dict(stub=StubDetector(load_error=OSError("model.safetensors is truncated"))),
    "out-of-memory": dict(stub=StubDetector(run_error=RuntimeError(
        "CUDA out of memory. Tried to allocate 1.2 GiB"))),
}


@pytest.mark.parametrize("failure", sorted(FAILURES))
def test_a_detector_that_fails_leaves_an_unmasked_solve_that_says_so(session, colmap, failure):
    spec = dict(FAILURES[failure])
    stub = spec.pop("stub", StubDetector())
    summary = _masked(session, stub, **spec)
    assert summary["solved"] is True, "the solve must complete"
    record = _solution_json(session)["transients"]
    assert record["state"] == SM.RECORD_UNAVAILABLE, "never silently applied"
    assert record["requested"] is True and record["extraction_masked"] is False
    assert isinstance(record["detail"], str) and record["detail"]
    assert record["rule"] is None
    # Unmasked means today's database and no mask path at all.
    assert "mask_path" not in colmap.sets("ImageReaderOptions")
    assert [c[2] for c in colmap.calls("extract_features")] == ["database.db"]
    assert record["database"] == "database.db"
    assert summary["transients"] == record


def test_the_failure_detail_names_the_cause(session, colmap):
    summary = _masked(session, StubDetector(run_error=RuntimeError("CUDA out of memory")))
    assert "out of memory" in summary["transients"]["detail"]
    assert "gdsam" in summary["transients"]["detail"]
    summary = _masked(session, StubDetector(),
                      device_probe=lambda: "no CUDA device, and no CPU fallback")
    assert "CPU fallback" in summary["transients"]["detail"]


def test_a_union_without_grounding_dino_is_masked_by_oneformer_and_is_not_applied(session, colmap):
    """The surface stage's own fallback: the masks are real, under the
    OneFormer rule, and the record names both rules. But the evidence was
    measured with the union rule, so the state is `partial`, never `applied`,
    and the gate falls back to its fail-safe."""

    class Half(StubDetector):
        def __call__(self, component):
            if component == T.COMPONENT_GDSAM:
                return T.DisabledBackend(component, "Grounding DINO weights missing")
            return super().__call__(component)

    summary = _masked(session, Half())
    record = summary["transients"]
    assert record["state"] == SM.RECORD_PARTIAL
    assert record["extraction_masked"] is True
    assert record["images_masked"] == N and record["images_unmasked"] == 0
    assert record["rule"] == T.TransientParams(mode=T.MODE_ONEFORMER).rule_id()
    assert record["requested_rule"] == T.TransientParams().rule_id()
    assert "Grounding DINO weights missing" in record["rule_fallback"]


def test_the_mask_step_raising_is_an_unmasked_solve(session, colmap, monkeypatch):
    def boom(*a, **k):
        raise ValueError("unexpected")

    monkeypatch.setattr(SM, "ensure_solver_masks", boom)
    summary = _masked(session, StubDetector())
    assert summary["solved"] is True
    assert summary["transients"]["state"] == SM.RECORD_UNAVAILABLE
    assert "unexpected" in summary["transients"]["detail"]


# ---------------------------------------------------------------------------
# 6. what the solution records, and old solutions
# ---------------------------------------------------------------------------


def _bare_solution(**kw):
    return global_solve.Solution(
        solver="glomap", solved_at=1.0, input_digest="d", keyframe_ids=["s:1"], poses={},
        components=[], xyz=np.zeros((0, 3), np.float32), rgb=np.zeros((0, 3), np.uint8),
        component=np.zeros(0, np.int32), first_keyframe=np.zeros(0, np.int32),
        track_length=np.zeros(0, np.int32), error=np.zeros(0, np.float32),
        observations=np.zeros((0, 3), np.int32), **kw)


def test_a_solution_written_before_these_records_reads_and_merges_as_before(tmp_path):
    ws = global_solve.SolveWorkspace(tmp_path / "solve")
    global_solve.write_solution(ws, _bare_solution())
    meta = json.loads(ws.solution_path.read_text(encoding="utf-8"))
    assert "transients" not in meta and "solve" not in meta

    class _Store:
        def world_dir(self, world_id):
            return tmp_path / "w"

    store = _Store()
    real = global_solve.workspace_for(store, "w", "s")
    real.root.mkdir(parents=True)
    for name in ("solution.json", "solution.npz"):
        (real.root / name).write_bytes((ws.root / name).read_bytes())
    loaded = global_solve.load_solution(store, "w", "s")
    assert loaded.transients is None and loaded.solve is None
    merged = global_solve.merge([], [], [], None, loaded, input_digest="d")
    assert "transients" not in merged.summary and "solve" not in merged.summary


def test_a_killed_solves_mask_staging_file_is_swept(tmp_path):
    """The COLMAP masks are written the atomic way; a solve killed mid-write
    leaves a staging file that the next solve's sweep removes."""
    import subprocess

    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    ws = global_solve.SolveWorkspace(tmp_path / "solve")
    masks = ws.root / SM.MASKS_DIRNAME
    masks.mkdir(parents=True)
    orphan = masks / f"00000001.jpg.png.p{dead.pid}.abcd1234.tmp"
    orphan.write_bytes(b"half a png")
    kept = masks / "00000001.jpg.png"
    kept.write_bytes(b"a finished mask")
    assert global_solve.sweep_workspace(ws) == 1
    assert not orphan.exists() and kept.exists()


def test_the_records_reach_the_published_manifest_summary(tmp_path):
    record = {"state": SM.RECORD_APPLIED, "rule": "r"}
    solve = {"seed": 7, "threads": 1}
    merged = global_solve.merge([], [], [], None, _bare_solution(transients=record, solve=solve),
                                input_digest="d")
    assert merged.summary["transients"] == record
    assert merged.summary["solve"] == solve


# ---------------------------------------------------------------------------
# 7. the surface stage reuses the solver's masks -- for a raw build only
# ---------------------------------------------------------------------------


def _world_with_solver_masks(root):
    """The appearance tests' world, its raw capture frames recorded, and the
    solve's mask cache built over solver images of the solve camera's size."""
    from tests.test_world_builder_appearance import SESSION, WORLD, H, W, World, _paint_forbidden_sources

    world = World(root)
    _paint_forbidden_sources(world)
    ws = global_solve.workspace_for(world.store, WORLD, SESSION)
    names, ids = [], {}
    rng = np.random.default_rng(1)
    for kid in world.kids:
        name = f"{kid.rsplit(':', 1)[-1]}.jpg"
        ok, buf = cv2.imencode(".jpg", rng.integers(0, 255, (H, W, 3), dtype=np.uint8))
        (ws.images_dir / name).write_bytes(buf.tobytes())
        names.append(name)
        ids[name] = kid
    solver_stub = StubDetector(rect=(10, 40, 10, 50))
    result = SM.ensure_solver_masks(ws, names, keyframe_ids=ids, shape=(H, W),
                                    backend_factory=solver_stub, device_probe=_gpu)
    assert result.record_state == SM.RECORD_APPLIED
    return world, ws, names, solver_stub


def _surface_masks(world, policy, stub):
    from tests.test_world_builder_appearance import SESSION, WORLD
    from tower.world_builder.global_solve import load_solution

    solution = load_solution(world.store, WORLD, SESSION)
    return T.ensure_transient_masks(
        world.store, WORLD, SESSION, list(enumerate(world.kids)), intrinsics=world.intrinsics,
        camera=solution.camera, align_records={r["ki"]: r for r in world.records},
        depth_dir=world.dense / "work" / "depth", params=T.TransientParams(),
        policy=policy, backend_factory=stub)


def test_a_raw_imagery_surface_takes_the_solvers_masks_instead_of_recomputing(tmp_path):
    from tests.test_world_builder_appearance import SESSION, WORLD, H, W
    from tower.world_builder import appearance as A
    from tower.world_builder import raw_imagery as RAWIMG

    world, ws, names, _ = _world_with_solver_masks(tmp_path)
    policy = A.resolve_label_policy(world.store, WORLD, SESSION, imagery_source=RAWIMG.IMAGERY_RAW)
    assert policy.raw
    surface_stub = StubDetector()
    report = _surface_masks(world, policy, surface_stub)
    assert report.state == T.STATE_OK
    assert surface_stub.calls == [], "the surface ran no detector"
    assert report.reused_from_solve == len(world.kids)
    assert report.record()["reused_from_solve"] == len(world.kids)
    m = report.mask(0)
    assert m is not None and m[10:40, 10:50].all() and not m[:, 100:].any()
    # The copied file says where it came from.
    _h, _p, rec = T.read_component(T.cache_path(world.dense / "work" / "depth", 0, T.COMPONENT_GDSAM))
    assert rec["origin"] == "solve" and rec["solve_image"] == names[0]
    assert (H, W) == m.shape


def test_a_redacted_surface_never_takes_a_mask_made_from_raw_pixels(tmp_path):
    world, ws, names, _ = _world_with_solver_masks(tmp_path)
    surface_stub = StubDetector()
    report = _surface_masks(world, None, surface_stub)        # the product: redacted
    assert report.state == T.STATE_OK
    assert report.reused_from_solve == 0
    assert "reused_from_solve" not in report.record()
    assert len(surface_stub.calls) == 2 * len(world.kids), "it computed its own, both components"


def test_the_solver_mask_is_not_lent_once_its_image_changed(tmp_path):
    from tests.test_world_builder_appearance import SESSION, WORLD, H, W

    world, ws, names, _ = _world_with_solver_masks(tmp_path)
    donor = SM.solve_mask_donor(world.store, WORLD, SESSION)
    params = T.TransientParams()
    assert donor(world.kids[0], T.COMPONENT_GDSAM, params, (H, W)) is not None
    assert donor(world.kids[0], T.COMPONENT_GDSAM, params, (H + 1, W)) is None   # other grid
    ok, buf = cv2.imencode(".jpg", np.zeros((H, W, 3), np.uint8))
    (ws.images_dir / names[1]).write_bytes(buf.tobytes())
    fresh = SM.solve_mask_donor(world.store, WORLD, SESSION)
    assert fresh(world.kids[1], T.COMPONENT_GDSAM, params, (H, W)) is None
    # And a world whose solve masked nothing has no donor at all: today.
    assert SM.solve_mask_donor(world.store, WORLD, "another-session") is None


# ---------------------------------------------------------------------------
# review V5: M1-2 (GPU out of memory), M1-4 (read-only walk DB), M1-5 (a database
# per solve), M1-6 (why a solve re-extracted -- asserted in the path tests above)
# ---------------------------------------------------------------------------


class _OOMOnce(StubDetector):
    """The detector's first run raises a CUDA OOM after emitting one image; the retry succeeds --
    or, with `always`, fails again."""

    def __init__(self, always=False):
        super().__init__()
        self.always, self.raised = always, 0

    def __call__(self, component):
        backend = super().__call__(component)
        stub, run = self, backend.run

        def run_with_oom(items, params, emit, should_stop=None):
            if stub.raised == 0 or stub.always:
                stub.raised += 1
                if items:
                    i, rgb, _u = items[0]
                    emit(i, np.zeros(rgb.shape[:2], bool), np.zeros(rgb.shape[:2], bool), 0.001)
                raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
            return run(items, params, emit, should_stop)

        backend.run = run_with_oom
        return backend


def test_a_gpu_oom_is_retried_once_on_a_cleared_cache(session, colmap, monkeypatch):
    cleared = []
    monkeypatch.setattr(SM, "_empty_cuda_cache", lambda: cleared.append(1))
    summary = _masked(session, _OOMOnce())
    record = summary["transients"]
    assert record["state"] == SM.RECORD_APPLIED and record["retries"] >= 1 and cleared
    assert record["cause"] is None and record["retryable"] is False


def test_a_gpu_oom_that_persists_is_recorded_as_retryable(session, colmap, monkeypatch):
    monkeypatch.setattr(SM, "_empty_cuda_cache", lambda: None)
    summary = _masked(session, _OOMOnce(always=True))
    record = summary["transients"]
    assert record["state"] == SM.RECORD_UNAVAILABLE
    assert record["cause"] == SM.CAUSE_GPU_OOM and record["retryable"] is True
    assert record["retries"] == 1
    # any other detector failure is not retryable
    summary = _masked(session, StubDetector(run_error=RuntimeError("device lost")))
    assert summary["transients"]["cause"] == SM.CAUSE_DETECTOR_FAILED
    assert summary["transients"]["retryable"] is False


def test_the_walk_database_is_opened_read_only(walked, monkeypatch):
    """No connection to the walk's database may write: every one is `mode=ro`."""
    import sqlite3

    opened = []
    real = sqlite3.connect

    def spy(target, *a, **kw):
        opened.append((str(target), kw.get("uri", False)))
        return real(target, *a, **kw)

    monkeypatch.setattr(sqlite3, "connect", spy)
    walk = walked.workspace.database_path
    assert SM.walk_database_usable(walk)
    masks = SM.SolverMasks(state=SM.STATE_OK, params=SM.solver_params(), requested_rule="r")
    SM.filter_walk_database(walked.workspace, masks)
    to_walk = [(t, uri) for t, uri in opened if Path(t.split("?")[0].replace("file:///", "")).name
               == walk.name or t == str(walk)]
    assert to_walk and all(uri and t.endswith("?mode=ro") for t, uri in to_walk), opened


def test_two_solves_never_share_a_masked_database(walked, colmap):
    a = _masked(walked, StubDetector())["transients"]["database"]
    b = _masked(walked, StubDetector())["transients"]["database"]
    assert a != b and _is_masked_db(a) and _is_masked_db(b)
    # the second solve kept the first's database: its writer (this process) is alive, and the
    # published solution named it until the second published
    assert (walked.workspace.root / b).exists()


def test_a_dead_solves_masked_database_is_swept_the_published_one_kept(walked, colmap):
    import subprocess

    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    root = walked.workspace.root
    stale = root / f"database.masked.p{dead.pid}.0123abcd.db"
    stale.write_bytes(b"x")
    (root / f"database.masked.p{dead.pid}.0123abcd.json").write_text("{}")
    published = root / f"database.masked.p{dead.pid}.feedbeef.db"
    published.write_bytes(b"y")
    meta = json.loads(walked.workspace.solution_path.read_text(encoding="utf-8"))
    meta.setdefault("solve", {})["database"] = published.name
    walked.workspace.solution_path.write_text(json.dumps(meta), encoding="utf-8")
    removed = SM.sweep_masked_databases(walked.workspace)
    assert removed == [stale.name]
    assert not stale.exists() and published.exists() and walked.workspace.database_path.exists()


def test_a_fallback_forced_by_gpu_memory_is_recorded_retryable(session, colmap):
    """V7 L-b: Grounding DINO / SAM out of memory, OneFormer carries on: `partial`, with the cause."""
    class OOMGdsam(StubDetector):
        def __call__(self, component):
            backend = super().__call__(component)
            if component == T.COMPONENT_GDSAM:
                def run(items, params, emit, should_stop=None):
                    raise T.TransientDetectorUnavailable("CUDA out of memory. Tried to allocate 1 GiB")
                backend.run = run
            return backend

    record = _masked(session, OOMGdsam())["transients"]
    assert record["state"] == SM.RECORD_PARTIAL
    assert record["cause"] == SM.CAUSE_GPU_OOM and record["retryable"] is True
