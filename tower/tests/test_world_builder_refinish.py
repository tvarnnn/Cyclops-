"""The re-finish command (WORLD-BUILDER-COMPONENTS.md §7 rule 4, contract T11).

Run on a fixture world built in `tmp_path` -- never the live store. The solve child
(`world_finalize.py` with the product settings) and the GPU stages are faked; what is
pinned is the order, the settings the solve is given, that nothing is deleted, and that
components and areas follow.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import world_build_session as wbs  # noqa: E402
from scripts import world_finish_pending as wfp  # noqa: E402
from scripts import world_refinish as wr  # noqa: E402
from tests.test_world_builder_area_build import (  # noqa: E402
    AREA1,
    AREA2,
    ROOM,
    S1,
    W1,
    _fake_stages,
    _store,
    fake_depth,  # noqa: F401 -- a fixture
)
from tower.world_builder import components as C  # noqa: E402


@pytest.fixture(autouse=True)
def _no_native_warm(monkeypatch):
    monkeypatch.setattr(wfp, "prewarm_world_builder", lambda *a, **k: ())


def _old_world(tmp_path):
    """A saved world as today's Tower leaves it: a solve, a room surface and
    appearance, a derived tree, no components record."""
    store, kids = _store(tmp_path)
    wd = store.world_dir(W1)
    (wd / "solve" / S1 / "sources.json").write_text(json.dumps({"k": "v"}))
    (wd / "solve" / S1 / "database.db").write_bytes(b"old features")
    for stage in ("surface", "appearance"):
        (wd / stage / S1).mkdir(parents=True)
        (wd / stage / S1 / "manifest.json").write_text(json.dumps({"old": stage}))
    (wd / "derived").mkdir()
    (wd / "derived" / "manifest.json").write_text(json.dumps({"old": "derived"}))
    return store, kids


def _files(root: Path) -> dict:
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


class _Solve:
    """The solve child: records its argv and env, and publishes a components record
    the way the evidence gate does."""

    def __init__(self, store, kids, *, gate_writes=True, code=0):
        self.store, self.kids, self.gate_writes, self.code = store, kids, gate_writes, code
        self.calls = []

    def __call__(self, argv, env=None, capture_output=True, text=True):
        self.calls.append({"argv": argv, "env": env})
        if self.code == 0:
            from tests.test_world_builder_area_build import _components
            from tower.world_builder.global_solve import (
                load_solution,
                workspace_for,
                write_solution,
            )

            # The solve is published afresh into the fresh solve directory -- with or
            # without the gate; only a gated one writes a components record.
            old = self.store.world_dir(W1) / wr.REFINISH_DIRNAME
            stamp = next(old.iterdir()).name
            sol = load_solution(_ShadowStore(None, old / stamp), W1, S1)
            write_solution(workspace_for(self.store, W1, S1), sol)
            if self.gate_writes:
                _components(self.store, self.kids)
        out = {"finalized": self.code == 0, "global_solve": {"solved": self.code == 0}}
        return SimpleNamespace(returncode=self.code, stdout=json.dumps(out), stderr="")


class _ShadowStore:
    """Reads the set-aside solve as if it were the world's (for the fake solve)."""

    def __init__(self, base, aside):
        self._base, self._aside = base, aside

    def world_dir(self, world_id):
        return self._aside


@pytest.fixture
def stages(monkeypatch):
    calls = []
    fake = _fake_stages(calls)
    monkeypatch.setattr(wbs, "final_surface_stages", fake)
    monkeypatch.setattr(wfp, "final_surface_stages", fake)
    return calls


def test_it_sets_aside_solves_gated_and_builds_the_room_and_its_areas(
        tmp_path, stages, fake_depth):  # noqa: F811
    store, kids = _old_world(tmp_path)
    before = _files(store.world_dir(W1))
    solve = _Solve(store, kids)
    report = wr.refinish(store, tmp_path, W1, S1, seed=7, solve_runner=solve,
                         stamp="20260923-000000")
    assert report["done"] is True, report
    # 2. the product settings, in the child
    env = solve.calls[0]["env"]
    assert env["TOWER_WORLD_SOLVE_MASKS"] == "1"
    assert env["TOWER_WORLD_SOLVE_GATE"] == "1"
    assert env["TOWER_WORLD_SOLVE_SEED"] == "7"
    argv = solve.calls[0]["argv"]
    assert argv[1].endswith("world_finalize.py")
    assert argv[argv.index("--world") + 1] == W1 and argv[argv.index("--session") + 1] == S1
    # 3. the room, then 4. both areas, each through its own view
    assert [getattr(c["store"], "area_id", None) for c in stages] == [None, AREA1, AREA2]
    assert [c["id"] for c in report["components"]] == [ROOM, AREA1, AREA2]
    # 1. NOTHING DELETED: every file that was there is there, or set aside.
    aside = store.world_dir(W1) / "refinish" / "20260923-000000"
    after = _files(store.world_dir(W1))
    for rel, data in before.items():
        assert rel in after or f"refinish\\20260923-000000\\{rel}" in after or \
            f"refinish/20260923-000000/{rel}" in after, rel
    assert (aside / "solve" / S1 / "database.db").read_bytes() == b"old features"
    assert json.loads((aside / "surface" / S1 / "manifest.json").read_text()) == {
        "old": "surface"}
    assert (aside / "session.json").exists()
    assert (aside / "derived" / "manifest.json").exists()
    # the solve keeps where the raw frames were
    assert (store.world_dir(W1) / "solve" / S1 / "sources.json").exists()
    ledger = json.loads((aside / wr.LEDGER_FILENAME).read_text())
    assert ledger["world_id"] == W1 and ledger["session_id"] == S1
    assert {m["kind"] for m in ledger["moved"]} == {"solve"}
    assert {c["kind"] for c in ledger["copied"]} >= {"surface", "appearance", "derived",
                                                     "session"}
    assert ledger["previous"]["stages"]["surface"]["state"] == "ok"
    assert "human" in ledger["deletion"]
    # the lock is released
    assert store.lock_holder(W1) is None


def test_a_second_refinish_sets_the_first_areas_aside(tmp_path, stages, fake_depth):  # noqa: F811
    store, kids = _old_world(tmp_path)
    wr.refinish(store, tmp_path, W1, S1, solve_runner=_Solve(store, kids), stamp="a")
    assert (C.areas_dir(store, W1) / AREA1).exists()
    # The fake solve reads the most recent set-aside solve; keep only one.
    report = wr.refinish(store, tmp_path, W1, S1, solve_runner=_Solve2(store, kids),
                         stamp="b")
    assert report["done"] is True
    assert (store.world_dir(W1) / "refinish" / "b" / "areas" / AREA1).exists()


class _Solve2(_Solve):
    def __call__(self, argv, env=None, capture_output=True, text=True):
        from tests.test_world_builder_area_build import _components
        from tower.world_builder.global_solve import load_solution, workspace_for, write_solution

        self.calls.append({"argv": argv, "env": env})
        aside = self.store.world_dir(W1) / wr.REFINISH_DIRNAME / "b"
        sol = load_solution(_ShadowStore(None, aside), W1, S1)
        write_solution(workspace_for(self.store, W1, S1), sol)
        _components(self.store, self.kids)
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")


def test_without_the_gate_the_room_is_rebuilt_and_components_stay_null(tmp_path, stages):
    store, kids = _old_world(tmp_path)
    report = wr.refinish(store, tmp_path, W1, S1,
                         solve_runner=_Solve(store, kids, gate_writes=False), stamp="x")
    assert report["done"] is True
    assert report["components"] is None and "components_note" in report
    assert len(stages) == 1                               # the room only


def test_a_failed_solve_puts_the_previous_result_back(tmp_path, stages):
    """Review V6, L3: a final solve that publishes nothing restores the set-aside solve,
    derived tree and session record, so the session is again what is on disk. What the
    failed solve left is kept, never deleted."""
    from tower.world_builder.global_solve import load_solution

    store, kids = _old_world(tmp_path)
    before_session = store.session_path(W1, S1).read_bytes()
    before_solution = load_solution(store, W1, S1)
    report = wr.refinish(store, tmp_path, W1, S1,
                         solve_runner=_Solve(store, kids, code=1), stamp="x")
    assert report["done"] is False and "put back" in report["reason"]
    assert stages == []
    restored = load_solution(store, W1, S1)
    assert restored is not None
    assert restored.input_digest == before_solution.input_digest
    assert (store.world_dir(W1) / "solve" / S1 / "database.db").read_bytes() == b"old features"
    assert store.session_path(W1, S1).read_bytes() == before_session
    assert store.read_session(W1, S1).stages["surface"]["state"] == "ok"
    aside = store.world_dir(W1) / "refinish" / "x"
    assert (aside / "failed-solve" / S1).exists()          # kept, not deleted
    ledger = json.loads((aside / wr.LEDGER_FILENAME).read_text())
    assert ledger["state"] == wr.LEDGER_RESTORED
    assert store.lock_holder(W1) is None


def test_a_solve_that_exits_zero_but_publishes_nothing_is_also_put_back(tmp_path, stages):
    store, kids = _old_world(tmp_path)

    def exits_zero(argv, env=None, capture_output=True, text=True):
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")

    report = wr.refinish(store, tmp_path, W1, S1, solve_runner=exits_zero, stamp="z")
    assert report["done"] is False
    assert (store.world_dir(W1) / "solve" / S1 / "solution.json").exists()
    assert stages == []


def test_it_refuses_a_purged_world_and_writes_nothing(tmp_path, stages):
    import dataclasses

    store, kids = _old_world(tmp_path)
    world = store.read_world(W1)
    store.write_world(dataclasses.replace(world, images_purged=True))
    before = _files(tmp_path)
    with pytest.raises(wr.Refused, match="purged"):
        wr.refinish(store, tmp_path, W1, S1, solve_runner=_Solve(store, kids))
    assert _files(tmp_path) == before


def test_it_refuses_an_unknown_session(tmp_path, stages):
    store, kids = _old_world(tmp_path)
    with pytest.raises(wr.Refused, match="no session"):
        wr.refinish(store, tmp_path, W1, "s9", solve_runner=_Solve(store, kids))


def test_dry_run_writes_nothing(tmp_path, capsys):
    store, _kids = _old_world(tmp_path)
    before = _files(tmp_path)
    assert wr.main(["--root", str(tmp_path), "--world", W1, "--dry-run"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["dry_run"] is True and out["session_id"] == S1
    assert out["final_solve_env"]["TOWER_WORLD_SOLVE_GATE"] == "1"
    assert [m["kind"] for m in out["plan"]["moves"]] == ["solve"]
    assert _files(tmp_path) == before


def test_the_root_goes_through_the_artifact_guard():
    src = (Path(wr.__file__)).read_text(encoding="utf-8")
    assert '"--root", type=artifact_root_arg' in src


# ---------------------------------------------------------------------------
# RV1 M1-1: the masked final solve must filter THE WALK'S OWN database (arm A1h)
# ---------------------------------------------------------------------------


def _walk_database(path: Path) -> None:
    """A COLMAP-shaped feature database holding one image: what a walk leaves."""
    import sqlite3

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()   # the fixture's placeholder bytes, in tmp_path
    con = sqlite3.connect(str(path))
    try:
        con.executescript(
            "create table images (image_id integer primary key, name text, camera_id int);"
            "create table keypoints (image_id int, rows int, cols int, data blob);"
            "create table matches (pair_id int, rows int, cols int, data blob);"
            "create table two_view_geometries (pair_id int, rows int, cols int, data blob,"
            " config int);"
            "insert into images values (1, '00000001.jpg', 1);"
            "insert into keypoints values (1, 1, 2, x'00');")
        con.commit()
    finally:
        con.close()


class _MaskedSolve(_Solve):
    """The solve child with the masks on, deciding its database the way
    `global_solve.solve` does (`_walk_database_usable` on the workspace's
    `database.db`): a walk database is filtered (A1h), none is re-extracted (A1).
    Like the real one it extracts and matches INTO the database it finds."""

    def __call__(self, argv, env=None, capture_output=True, text=True):
        import sqlite3

        from tower.world_builder import global_solve as gs

        self.calls.append({"argv": argv, "env": env})
        assert env["TOWER_WORLD_SOLVE_MASKS"] == "1"
        ws = gs.workspace_for(self.store, W1, S1)
        self.masking = ("walk-database-filtered" if gs._walk_database_usable(ws.database_path)
                        else "re-extracted")
        if self.masking == "walk-database-filtered":
            con = sqlite3.connect(str(ws.database_path))
            con.execute("insert into images values (2, '00000002.jpg', 1)")
            con.commit()
            con.close()
        aside = self.store.world_dir(W1) / wr.REFINISH_DIRNAME
        stamp = next(aside.iterdir()).name
        sol = gs.load_solution(_ShadowStore(None, aside / stamp), W1, S1)
        sol.solve = {"masking": self.masking, "database": ws.database_path.name}
        gs.write_solution(ws, sol)
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")


def test_a_refinish_filters_the_walks_own_database_and_keeps_the_original(tmp_path, stages):
    from tower.world_builder.global_solve import load_solution

    store, kids = _old_world(tmp_path)
    solve_dir = store.world_dir(W1) / "solve" / S1
    _walk_database(solve_dir / "database.db")
    (solve_dir / "images").mkdir()
    (solve_dir / "images" / "00000001.jpg").write_bytes(b"the walk's solver image")
    (solve_dir / "camera.json").write_text(json.dumps({"fx": 1.0}))
    walk_db = (solve_dir / "database.db").read_bytes()

    solve = _MaskedSolve(store, kids)
    report = wr.refinish(store, tmp_path, W1, S1, solve_runner=solve, stamp="m")
    assert report["done"] is True
    # The masked solve found the walk's database and filtered it: arm A1h.
    assert solve.masking == "walk-database-filtered"
    assert load_solution(store, W1, S1).solve["masking"] == "walk-database-filtered"
    # The set-aside original is byte for byte what the walk left; the rebuild wrote
    # into its own copy.
    aside = store.world_dir(W1) / "refinish" / "m" / "solve" / S1
    assert (aside / "database.db").read_bytes() == walk_db
    assert (solve_dir / "database.db").read_bytes() != walk_db
    assert (aside / "images" / "00000001.jpg").read_bytes() == b"the walk's solver image"
    assert (solve_dir / "images" / "00000001.jpg").read_bytes() == b"the walk's solver image"
    assert json.loads((solve_dir / "camera.json").read_text()) == {"fx": 1.0}
    ledger = json.loads((store.world_dir(W1) / "refinish" / "m" / wr.LEDGER_FILENAME)
                        .read_text())
    assert {c["name"] for c in ledger["copied_back"]} >= {"database.db", "images",
                                                          "camera.json", "sources.json"}
    assert {m["kind"] for m in ledger["moved"]} == {"solve"}


def test_without_a_walk_database_the_same_solve_would_re_extract(tmp_path, stages):
    """The contrast that makes the test above mean something: the fixture's
    placeholder `database.db` is not a COLMAP database, so the solve takes arm A1."""
    store, kids = _old_world(tmp_path)
    solve = _MaskedSolve(store, kids)
    wr.refinish(store, tmp_path, W1, S1, solve_runner=solve, stamp="n")
    assert solve.masking == "re-extracted"


def test_the_dry_run_names_what_goes_back(tmp_path, capsys):
    store, _kids = _old_world(tmp_path)
    _walk_database(store.world_dir(W1) / "solve" / S1 / "database.db")
    assert wr.main(["--root", str(tmp_path), "--world", W1, "--dry-run"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert "database.db" in out["plan"]["copy_back_into_fresh_solve"]


def test_the_solvers_mask_cache_is_copied_back_and_the_rest_is_not(tmp_path, stages):
    """`transients/` is keyed by image name AND content, so the rebuild can only use an
    entry for the exact image it was made on; it saves the masked solve its GPU
    minutes. What a solve made for itself (`masks/`, `database.masked.*`) stays set
    aside only."""
    store, kids = _old_world(tmp_path)
    solve_dir = store.world_dir(W1) / "solve" / S1
    (solve_dir / "transients").mkdir()
    entry = solve_dir / "transients" / "00000001.0123456789ab.hands.npz"
    entry.write_bytes(b"cached union mask")
    (solve_dir / "masks").mkdir()
    (solve_dir / "masks" / "00000001.jpg.png").write_bytes(b"colmap mask")
    (solve_dir / "database.masked.db").write_bytes(b"a masked solve's own db")
    wr.refinish(store, tmp_path, W1, S1, solve_runner=_Solve(store, kids, gate_writes=False),
                stamp="t")
    aside = store.world_dir(W1) / "refinish" / "t" / "solve" / S1
    assert (solve_dir / "transients" / entry.name).read_bytes() == b"cached union mask"
    assert (aside / "transients" / entry.name).read_bytes() == b"cached union mask"
    assert not (solve_dir / "masks").exists()
    assert not (solve_dir / "database.masked.db").exists()
    assert (aside / "masks" / "00000001.jpg.png").exists()
    assert (aside / "database.masked.db").exists()


# ---------------------------------------------------------------------------
# review V6: H1 (a build running without the lock), M1 (set-aside atomicity),
# M3 (the room says a re-finish is in progress)
# ---------------------------------------------------------------------------


@pytest.fixture
def live_pid():
    import subprocess

    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        yield sleeper.pid
    finally:
        sleeper.kill()


def _running_status(path: Path, pid: int) -> None:
    import time as _time

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"state": "running", "stage": "depth", "pid": pid,
                                "updated_at": _time.time()}))


def test_it_refuses_while_the_builder_runs_the_rooms_post_stop_surface(tmp_path, stages,
                                                                        live_pid):
    """H1: the builder releases the writer lock and THEN runs the final surface and
    appearance. A `running` room stage under a live pid is a build in progress."""
    store, kids = _old_world(tmp_path)
    _running_status(store.world_dir(W1) / "surface" / S1 / "status.json", live_pid)
    with pytest.raises(wr.Refused, match="running under a live process"):
        wr.refinish(store, tmp_path, W1, S1, solve_runner=_Solve(store, kids), stamp="x")
    assert not (store.world_dir(W1) / "refinish").exists()
    assert (store.world_dir(W1) / "solve" / S1 / "solution.json").exists()
    assert store.lock_holder(W1) is None


def test_it_refuses_while_an_area_of_the_session_is_being_built(tmp_path, stages,
                                                                 fake_depth, live_pid):  # noqa: F811
    store, kids = _old_world(tmp_path)
    wr.refinish(store, tmp_path, W1, S1, solve_runner=_Solve(store, kids), stamp="a")
    area = C.areas_dir(store, W1) / AREA1
    _running_status(area / "appearance" / S1 / "status.json", live_pid)
    with pytest.raises(wr.Refused, match="being built right now"):
        wr.refinish(store, tmp_path, W1, S1, solve_runner=_Solve2(store, kids), stamp="b")
    assert not (store.world_dir(W1) / "refinish" / "b").exists()


def test_a_move_that_keeps_failing_is_rolled_back_and_the_world_is_as_it_was(
        tmp_path, stages, fake_depth, monkeypatch):  # noqa: F811
    """M1: the solve move fails after the areas moved -- every completed move is undone,
    the ledger says so, nothing is stranded, and the error is a refusal."""
    store, kids = _old_world(tmp_path)
    wr.refinish(store, tmp_path, W1, S1, solve_runner=_Solve(store, kids), stamp="a")
    solve_dir = store.world_dir(W1) / "solve" / S1
    before_solve = sorted(p.name for p in solve_dir.iterdir())
    area = C.areas_dir(store, W1) / AREA1
    assert area.exists()
    real = wr._replace_with_retry

    def refuse_the_solve(src, dst):
        if Path(src) == solve_dir:
            raise PermissionError(13, "held open by a reader")
        return real(src, dst)

    monkeypatch.setattr(wr, "_replace_with_retry", refuse_the_solve)
    with pytest.raises(wr.SetAsideFailed, match="nothing was moved"):
        wr.refinish(store, tmp_path, W1, S1, solve_runner=_Solve2(store, kids), stamp="b")
    assert sorted(p.name for p in solve_dir.iterdir()) == before_solve
    assert area.exists()                                   # moved, then moved back
    ledger = json.loads((store.world_dir(W1) / "refinish" / "b" / wr.LEDGER_FILENAME)
                        .read_text())
    assert ledger["state"] == wr.LEDGER_ROLLED_BACK
    assert "PermissionError" in ledger["error"]
    assert all(m.get("moved_back") for m in ledger["moved"] if m.get("moved"))
    assert store.lock_holder(W1) is None
    # The room was not marked: nothing was set aside.
    assert store.read_session(W1, S1).stages["surface"]["state"] == "ok"


@pytest.mark.skipif(os.name != "nt", reason="Windows directory-rename semantics")
def test_an_open_area_file_is_a_clean_refusal_on_windows(tmp_path, stages, fake_depth,
                                                          monkeypatch):  # noqa: F811
    """The reviewer's probe, reversed: a reader holding an area file open no longer
    leaves the solve moved with no ledger."""
    monkeypatch.setattr(wr, "MOVE_BACKOFF_S", 0.01)
    store, kids = _old_world(tmp_path)
    wr.refinish(store, tmp_path, W1, S1, solve_runner=_Solve(store, kids), stamp="a")
    area = C.areas_dir(store, W1) / AREA1
    victim = next(p for p in area.rglob("*") if p.is_file())
    handle = open(victim, "rb")  # noqa: SIM115
    try:
        with pytest.raises(wr.SetAsideFailed):
            wr.refinish(store, tmp_path, W1, S1, solve_runner=_Solve2(store, kids),
                        stamp="b")
    finally:
        handle.close()
    assert (store.world_dir(W1) / "solve" / S1 / "solution.json").exists()
    assert area.exists()
    aside = store.world_dir(W1) / "refinish" / "b"
    assert json.loads((aside / wr.LEDGER_FILENAME).read_text())["state"] == \
        wr.LEDGER_ROLLED_BACK
    assert store.lock_holder(W1) is None


def test_while_the_new_solve_runs_the_room_says_a_refinish_is_in_progress(tmp_path, stages):
    """M3: under the step-1 lock the room's stages are recorded `stopped`, so a
    re-finish that ends before step 3 leaves a room the finisher rebuilds."""
    store, kids = _old_world(tmp_path)
    seen = {}
    inner = _Solve(store, kids)

    def solve(argv, env=None, capture_output=True, text=True):
        stages_now = store.read_session(W1, S1).stages
        seen.update({k: dict(v) for k, v in stages_now.items()})
        return inner(argv, env=env, capture_output=capture_output, text=text)

    wr.refinish(store, tmp_path, W1, S1, solve_runner=solve, stamp="m")
    for stage in ("surface", "appearance"):
        assert seen[stage]["state"] == "stopped"
        assert "re-finish in progress" in seen[stage]["detail"]
    # Step 3 then recorded the rebuilt room.
    assert store.read_session(W1, S1).stages["surface"]["state"] == "ok"
    from scripts import world_finish_pending as wfp2

    # And while it said `stopped`, the finisher would have owed the room.
    assert "stopped" in {v["state"] for v in seen.values()}
    del wfp2
