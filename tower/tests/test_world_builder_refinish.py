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
    # Review V9, M-7: a solver image goes back only when its provenance is proven. This
    # one belongs to no keyframe of the session, so it is withheld -- the set-aside
    # original is untouched.
    assert not (solve_dir / "images" / "00000001.jpg").exists()
    assert json.loads((solve_dir / "camera.json").read_text()) == {"fx": 1.0}
    ledger = json.loads((store.world_dir(W1) / "refinish" / "m" / wr.LEDGER_FILENAME)
                        .read_text())
    assert {c["name"] for c in ledger["copied_back"]} >= {"database.db", "images",
                                                          "camera.json", "sources.json"}
    assert ledger["solver_frames"]["walk_images"]["withheld_images"] == {
        "00000001.jpg": "no-keyframe"}
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


# ---------------------------------------------------------------------------
# P3 (PF): any exception between the set-aside and a published solve puts the previous
# result back (PV attempt 1: a read-only session.json made `mark_stage` raise WinError 5
# right after the set-aside, and the world was left with its solve moved aside)
# ---------------------------------------------------------------------------


def _world_with_areas(tmp_path):
    """A world after one re-finish: a solve, the room, two built areas."""
    store, kids = _old_world(tmp_path)
    wr.refinish(store, tmp_path, W1, S1, solve_runner=_Solve(store, kids), stamp="a")
    assert (C.areas_dir(store, W1) / AREA1).exists()
    return store, kids


def _snapshot(store, *, derived=True):
    from tower.world_builder.global_solve import load_solution

    wd = store.world_dir(W1)
    snap = {"solve": sorted(p.name for p in (wd / "solve" / S1).iterdir()),
            "digest": load_solution(store, W1, S1).input_digest,
            "session": store.session_path(W1, S1).read_bytes(),
            "areas": sorted(p.name for p in C.areas_dir(store, W1).iterdir())}
    if derived:
        snap["derived"] = _files(wd / "derived")
    return snap


def _assert_put_back(store, before, stamp, step):
    wd = store.world_dir(W1)
    assert _snapshot(store) == before
    aside = wd / "refinish" / stamp
    ledger = json.loads((aside / wr.LEDGER_FILENAME).read_text())
    assert ledger["state"] == wr.LEDGER_RESTORED_AFTER_ERROR, ledger
    assert ledger["error_after_set_aside"]["step"] == step
    assert all(m.get("moved_back") for m in ledger["moved"] if m.get("moved"))
    assert ledger["restore_report"]["errors"] == []
    # What the rebuild had put in place is kept, never deleted.
    assert (aside / "failed-solve" / S1).exists()
    assert store.lock_holder(W1) is None
    return ledger


def test_an_error_right_after_the_set_aside_puts_the_previous_result_back(
        tmp_path, stages, fake_depth, monkeypatch):  # noqa: F811
    from tower.world_builder.engine import WorldBuilderEngine

    store, kids = _world_with_areas(tmp_path)
    before = _snapshot(store)

    def denied(self, *a, **k):
        raise PermissionError(13, "Access is denied", "session.json")

    monkeypatch.setattr(WorldBuilderEngine, "mark_stage", denied)
    solve = _Solve2(store, kids)
    report = wr.refinish(store, tmp_path, W1, S1, solve_runner=solve, stamp="b")
    assert report["done"] is False
    assert report["error"].startswith("marking the room's stages: PermissionError")
    assert "put back" in report["reason"]
    assert solve.calls == []                        # the solve never ran
    ledger = _assert_put_back(store, before, "b", "marking the room's stages")
    assert "PermissionError" in ledger["error_after_set_aside"]["error"]
    assert (C.areas_dir(store, W1) / AREA1).exists()   # the areas went back
    assert store.read_session(W1, S1).stages["surface"]["state"] == "ok"


@pytest.mark.skipif(os.name != "nt", reason="Windows refuses to replace a read-only file")
def test_a_read_only_session_record_is_pv_attempt_1_and_is_put_back(tmp_path, stages,
                                                                    fake_depth):  # noqa: F811
    """The failure as it happened: the copied tree kept the read-only attribute, so the
    real `mark_stage` could not replace session.json after the solve was set aside."""
    import stat

    store, kids = _world_with_areas(tmp_path)
    before = _snapshot(store)
    path = store.session_path(W1, S1)
    os.chmod(path, stat.S_IREAD)
    try:
        report = wr.refinish(store, tmp_path, W1, S1, solve_runner=_Solve2(store, kids),
                             stamp="b")
    finally:
        os.chmod(path, stat.S_IREAD | stat.S_IWRITE)
    assert report["done"] is False and "PermissionError" in report["error"]
    ledger = _assert_put_back(store, before, "b", "marking the room's stages")
    # The record was never written, so it is left alone rather than rewritten.
    assert ledger["restore_report"]["unchanged"] == str(path)


def test_a_solve_child_that_cannot_start_puts_the_previous_result_back(tmp_path, stages,
                                                                      fake_depth):  # noqa: F811
    store, kids = _world_with_areas(tmp_path)
    before = _snapshot(store)

    def cannot_start(argv, env=None, capture_output=True, text=True):
        # The room was marked `stopped` before the child was asked for.
        assert store.read_session(W1, S1).stages["surface"]["state"] == "stopped"
        raise OSError(8, "Not enough memory resources are available")

    report = wr.refinish(store, tmp_path, W1, S1, solve_runner=cannot_start, stamp="b")
    assert report["done"] is False and "OSError" in report["error"]
    _assert_put_back(store, before, "b", "running the final solve")
    # The session record this run marked `stopped` is written back from its snapshot.
    assert store.read_session(W1, S1).stages["surface"]["state"] == "ok"


def test_an_interrupt_after_the_set_aside_puts_it_back_and_is_re_raised(tmp_path, stages,
                                                                       fake_depth):  # noqa: F811
    store, kids = _world_with_areas(tmp_path)
    before = _snapshot(store)

    def interrupted(argv, env=None, capture_output=True, text=True):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        wr.refinish(store, tmp_path, W1, S1, solve_runner=interrupted, stamp="b")
    _assert_put_back(store, before, "b", "running the final solve")


def test_a_restore_that_cannot_finish_still_moves_everything_it_can_and_says_so(
        tmp_path, stages, fake_depth, monkeypatch):  # noqa: F811
    store, kids = _world_with_areas(tmp_path)
    session_derived = store.world_dir(W1) / "derived" / S1
    session_derived.mkdir(parents=True)
    (session_derived / "poses.json").write_text('{"poses": "old"}')
    before = _snapshot(store, derived=False)

    def rebuilt_then_cannot_finish(argv, env=None, capture_output=True, text=True):
        # the rebuild's derived tree for this session (review V10, MED-3: the put-back puts
        # back only this session's, and only where it differs from the snapshot)
        (session_derived / "poses.json").write_text('{"poses": "rebuilt"}')
        raise OSError("no child")

    real_copytree = wr.shutil.copytree
    calls = {"n": 0}
    snapshot = store.world_dir(W1) / "refinish" / "b" / "derived" / S1

    def copytree_fails_on_restore(src, dst, *a, **k):
        # the set-aside's own copies succeed; the put-back's copy of the snapshot does not
        if Path(src) == snapshot:
            calls["n"] += 1
            raise OSError(28, "No space left on device")
        return real_copytree(src, dst, *a, **k)

    monkeypatch.setattr(wr.shutil, "copytree", copytree_fails_on_restore)
    report = wr.refinish(store, tmp_path, W1, S1, solve_runner=rebuilt_then_cannot_finish,
                         stamp="b")
    assert calls["n"] == 1
    assert report["done"] is False and "stopped part-way" in report["reason"]
    ledger = json.loads((store.world_dir(W1) / "refinish" / "b" / wr.LEDGER_FILENAME)
                        .read_text())
    assert ledger["state"] == wr.LEDGER_RESTORE_INCOMPLETE
    assert any("derived" in e for e in ledger["restore_report"]["errors"])
    # every other step still ran: the solve, the areas and the session are back
    assert _snapshot(store, derived=False) == before
    # ... and the rebuild's derived tree was not moved before its replacement was copied
    assert (session_derived / "poses.json").read_text() == '{"poses": "rebuilt"}'
    assert store.lock_holder(W1) is None


def test_when_the_lock_is_taken_meanwhile_the_ledger_says_nothing_was_put_back(
        tmp_path, stages, monkeypatch):
    store, kids = _old_world(tmp_path)
    real = store.acquire_writer_lock
    calls = {"n": 0}

    def second_time_taken(world_id):
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("a live writer holds the world")
        return real(world_id)

    monkeypatch.setattr(store, "acquire_writer_lock", second_time_taken)

    def cannot_start(argv, env=None, capture_output=True, text=True):
        raise OSError("no child")

    report = wr.refinish(store, tmp_path, W1, S1, solve_runner=cannot_start, stamp="b")
    assert report["done"] is False and "could not be put back" in report["reason"]
    ledger = _ledger(store, "b")
    assert ledger["state"] == wr.LEDGER_RESTORE_INCOMPLETE
    assert "live writer" in ledger["restore_errors"][0]
    assert ledger["error_after_set_aside"]["step"] == "running the final solve"


# ---------------------------------------------------------------------------
# P3 (PF): the session's raw capture reaches the final solve (finding 6: 991e5a15 and
# af47007c were re-finished from the redacted keyframes with their captures on disk),
# and each keyframe gets ITS OWN capture's frame, never another's (review of 3abe763,
# ACC: the captures of the live walk adc75972 reuse frame names; by name, 67 of 353
# keyframes got another moment's image)
# ---------------------------------------------------------------------------

CAPTURE_ID = "c0ffee00c0ffee00c0ffee00c0ffee00"
CAP_B, CAP_C = "b" * 32, "c" * 32


def _frame_bytes(capture_id, seq, t):
    """A raw frame's bytes name its capture and its moment, so an assignment is
    checked exactly."""
    return f"{capture_id}|{seq}|{t!r}".encode()


def _write_capture(root, capture_id, frames, *, continues=None, started_at=None,
                   journal=True):
    """One capture directory as the recorder leaves it: `capture.json`, `frames/` and
    the `frames.jsonl` journal (`frames` = [(source_seq, received_at)])."""
    d = root / "captures" / capture_id
    (d / "frames").mkdir(parents=True, exist_ok=True)
    (d / "capture.json").write_text(json.dumps({
        "capture_id": capture_id, "continues_capture": continues,
        "started_at": started_at if started_at is not None else
        (min(t for _s, t in frames) - 1.0 if frames else 0.0),
        "retains_raw_imagery": True, "redaction": "none"}))
    lines = []
    for seq, t in frames:
        (d / "frames" / f"{seq:08d}.jpg").write_bytes(_frame_bytes(capture_id, seq, t))
        lines.append(json.dumps({
            "schema_version": 1, "source_seq": seq, "wire_seq": seq, "tx_seq": None,
            "received_at": t, "time_basis": "tower-receipt",
            "relpath": f"frames/{seq:08d}.jpg", "byte_count": 1, "width": 240,
            "height": 180}))
    if journal:
        (d / "frames.jsonl").write_text("\n".join(lines) + "\n")
    return d


def _walk(tmp_path, captures, keyframes):
    """An old world whose session followed a live capture chain. `captures` =
    [(capture_id, continues, [(seq, t), ...])]; `keyframes` = [(capture_id, seq, t)],
    each a frame of that capture -- the id and image name come from `seq` alone, as the
    builder makes them, so a restarted numbering collides exactly as on adc75972."""
    import dataclasses

    from tower.world_builder.records import Keyframe

    store, kids = _old_world(tmp_path)
    store.write_session(dataclasses.replace(store.read_session(W1, S1),
                                            capture_id=captures[0][0]))
    root = tmp_path / "caproot"
    for capture_id, continues, frames in captures:
        _write_capture(root, capture_id, frames, continues=continues)
    for _capture_id, seq, t in keyframes:
        store.append_keyframe(W1, Keyframe(
            keyframe_id=f"{S1}:{seq:08d}", session_id=S1, source_seq=seq, received_at=t,
            image_relpath=f"images/{seq:08d}.jpg", width=240, height=180, byte_count=1,
            wire_seq=seq))
    return store, kids, root


def _live_capture_world(tmp_path, *, n=3):
    """One capture, n keyframes, frames 1..n at t = 100 + seq."""
    frames = [(i, 100.0 + i) for i in range(1, n + 1)]
    return _walk(tmp_path, [(CAPTURE_ID, None, frames)],
                 [(CAPTURE_ID, s, t) for s, t in frames])


def _capture_args(argv):
    return [argv[i + 1] for i, a in enumerate(argv) if a == "--capture-dir"]


def _ledger(store, stamp):
    return json.loads((store.world_dir(W1) / "refinish" / stamp / wr.LEDGER_FILENAME)
                      .read_text())


def _written_sources(store):
    from tower.world_builder.global_solve import read_sources, workspace_for

    return read_sources(workspace_for(store, W1, S1))


def _assert_every_frame_is_its_own(store, keyframes):
    """Every raw frame the solve will read belongs to that keyframe's own capture AND
    moment; returns how many keyframes have one."""
    own = {f"{S1}:{seq:08d}": _frame_bytes(c, seq, t) for c, seq, t in keyframes}
    sources = _written_sources(store)
    for kid, path in sources.items():
        assert Path(path).read_bytes() == own[kid], (kid, path)
    return len(sources)


def test_the_sessions_capture_reaches_the_solve_as_sources_never_as_a_capture_dir(
        tmp_path, stages, monkeypatch):
    store, kids, capture_root = _live_capture_world(tmp_path)
    monkeypatch.setenv(wr.CAPTURE_ROOT_ENV, str(capture_root))
    solve = _Solve(store, kids)
    report = wr.refinish(store, tmp_path, W1, S1, solve_runner=solve, stamp="c")
    assert report["done"] is True
    expected = str(capture_root / "captures" / CAPTURE_ID)
    # The by-name `--capture-dir` lookup is never handed to the solve.
    assert _capture_args(solve.calls[0]["argv"]) == []
    frames = _ledger(store, "c")["solver_frames"]
    assert frames["from"] == wr.CAPTURE_FROM_SESSION
    assert frames["capture_id"] == CAPTURE_ID and frames["capture_dirs"] == [expected]
    assert frames["raw_from_capture_dir"] == 3 and frames["redacted_session_copies"] == 0
    assert frames["source"] == wr.SOURCE_RAW and frames["sources_json_written"] is True
    assert "sources" not in frames                     # the record is on disk, not here
    assert report["solver_frames"] == frames
    assert _assert_every_frame_is_its_own(
        store, [(CAPTURE_ID, i, 100.0 + i) for i in (1, 2, 3)]) == 3
    # The set-aside walk record is untouched.
    aside = store.world_dir(W1) / "refinish" / "c" / "solve" / S1 / "sources.json"
    assert json.loads(aside.read_text()) == {"k": "v"}


def test_colliding_frame_names_across_three_captures_never_cross(tmp_path, stages,
                                                                 monkeypatch):
    """The reviewer's case: a reconnect restarts the numbering, so all three captures
    hold frames 1..4. Every keyframe gets the frame of ITS OWN capture and moment, and
    the two keyframes whose ids collide in the session keep their stored copies."""
    a = [(s, 10.0 + s) for s in (1, 2, 3, 4)]
    b = [(s, 20.0 + s) for s in (1, 2, 3, 4)]
    c = [(s, 30.0 + s) for s in (1, 2, 3, 4)]
    keyframes = [(CAPTURE_ID, 1, 11.0), (CAPTURE_ID, 3, 13.0), (CAP_B, 2, 22.0),
                 (CAP_B, 4, 24.0), (CAP_C, 1, 31.0)]          # id :00000001 twice
    store, kids, root = _walk(tmp_path, [(CAPTURE_ID, None, a), (CAP_B, CAPTURE_ID, b),
                                         (CAP_C, CAP_B, c)], keyframes)
    monkeypatch.setenv(wr.CAPTURE_ROOT_ENV, str(root))
    solve = _Solve(store, kids)
    report = wr.refinish(store, tmp_path, W1, S1, solve_runner=solve, stamp="c")
    assert _capture_args(solve.calls[0]["argv"]) == []
    frames = report["solver_frames"]
    assert [x["capture_id"] for x in frames["captures"]] == [CAPTURE_ID, CAP_B, CAP_C]
    assert _assert_every_frame_is_its_own(store, keyframes) == 3
    assert set(_written_sources(store)) == {f"{S1}:00000003", f"{S1}:00000002",
                                            f"{S1}:00000004"}
    assert frames["raw_from_capture_dir"] == 3
    assert frames["redacted_session_copies"] == 2
    assert frames["redacted_ambiguous_in_session"] == 2
    assert frames["ambiguous_keyframe_ids"] == [f"{S1}:00000001"]
    assert frames["source"] == wr.SOURCE_MIXED


def test_an_adc75972_shaped_walk_gives_no_keyframe_another_captures_frame(
        tmp_path, stages, monkeypatch):
    """Regression, the live walk's structure at small scale: three captures after two
    reconnects, each restarting at 1 and journalling every other frame; the builder
    kept every third journalled frame. The old by-name lookup gives some keyframes a
    frame of another capture -- the identity lookup gives none."""
    from tower.world_builder.global_solve import _source_frame

    captures, keyframes = [], []
    for cid, parent, t0, pick in ((CAPTURE_ID, None, 1000.0, slice(0, None, 3)),
                                  (CAP_B, CAPTURE_ID, 1100.0, slice(1, None, 3)),
                                  (CAP_C, CAP_B, 1200.0, slice(0, None, 4))):
        frames = [(s, t0 + 0.033 * s) for s in range(1, 60, 2)]
        captures.append((cid, parent, frames))
        keyframes += [(cid, s, t) for s, t in frames[pick]]
    store, kids, root = _walk(tmp_path, captures, keyframes)
    monkeypatch.setenv(wr.CAPTURE_ROOT_ENV, str(root))
    kf_records = store.read_keyframes(W1, S1)
    ids = [k.keyframe_id for k in kf_records]
    ambiguous = {k for k in ids if ids.count(k) > 1}
    assert ambiguous                                   # ids collide, as on adc75972
    # The hazard was real: by name, some keyframe got another capture's frame. The by-name
    # fallback now refuses a name more than one capture holds (the stored keyframe instead,
    # `global_solve._source_frame`), so it hands no keyframe a raw frame not its own.
    dirs = [root / "captures" / c for c in (CAPTURE_ID, CAP_B, CAP_C)]
    own = {(k.keyframe_id, k.received_at): _frame_bytes(c, s, t)
           for k, (c, s, t) in zip(kf_records, keyframes)}
    ambiguous_by_name = []
    picked = [_source_frame(k, store.session_dir(W1, S1), dirs, {}, ambiguous_by_name)
              for k in kf_records]
    assert ambiguous_by_name, "the restarted numbering makes names ambiguous"
    for k, path in zip(kf_records, picked):
        if Path(path).is_relative_to(root):
            assert path.read_bytes() == own[(k.keyframe_id, k.received_at)]
        else:
            assert path == store.session_dir(W1, S1) / k.image_relpath
    report = wr.refinish(store, tmp_path, W1, S1, solve_runner=_Solve(store, kids),
                         stamp="c")
    frames = report["solver_frames"]
    mapped = _assert_every_frame_is_its_own(store, keyframes)
    assert mapped == frames["raw_from_capture_dir"] > 0
    assert frames["redacted_ambiguous_in_session"] == sum(1 for k in ids if k in ambiguous)
    assert mapped + frames["redacted_session_copies"] == len(keyframes)
    assert not set(_written_sources(store)) & ambiguous


def test_no_journal_record_or_two_of_them_keeps_the_stored_copy(tmp_path, stages,
                                                                monkeypatch):
    a = [(1, 11.0), (2, 12.0), (3, 13.0)]
    # CAP_B journals frame 3 at the SAME moment as CAPTURE_ID (two claims to one
    # identity); keyframe 2's receipt time matches no record at all.
    b = [(3, 13.0)]
    keyframes = [(CAPTURE_ID, 1, 11.0), (CAPTURE_ID, 2, 12.5), (CAPTURE_ID, 3, 13.0)]
    store, kids, root = _walk(tmp_path, [(CAPTURE_ID, None, a), (CAP_B, CAPTURE_ID, b)],
                              keyframes)
    monkeypatch.setenv(wr.CAPTURE_ROOT_ENV, str(root))
    report = wr.refinish(store, tmp_path, W1, S1, solve_runner=_Solve(store, kids),
                         stamp="c")
    frames = report["solver_frames"]
    assert set(_written_sources(store)) == {f"{S1}:00000001"}
    assert frames["raw_from_capture_dir"] == 1
    assert frames["redacted_no_capture_frame"] == 1
    assert frames["redacted_several_capture_frames"] == 1


def test_a_directory_without_a_journal_is_not_searched_by_name(tmp_path, stages,
                                                                monkeypatch):
    store, kids, _root = _live_capture_world(tmp_path / "one")
    monkeypatch.delenv(wr.CAPTURE_ROOT_ENV, raising=False)
    bare = _write_capture(tmp_path / "bare", "x" * 32, [(1, 101.0), (2, 102.0)],
                          journal=False)
    report = wr.refinish(store, tmp_path / "one", W1, S1, solve_runner=_Solve(store, kids),
                         stamp="c", capture_dirs=[bare])
    frames = report["solver_frames"]
    assert frames["from"] == wr.CAPTURE_FROM_ARGUMENT
    assert frames["raw_from_capture_dir"] == 0 and frames["redacted_session_copies"] == 3
    assert "frames.jsonl" in frames["capture_notes"][0]
    assert frames["sources_json_written"] is False


def test_without_a_capture_root_the_solve_gets_the_session_copies_as_today(tmp_path, stages,
                                                                           monkeypatch):
    store, kids, _capture_root = _live_capture_world(tmp_path)
    monkeypatch.delenv(wr.CAPTURE_ROOT_ENV, raising=False)
    solve = _Solve(store, kids)
    wr.refinish(store, tmp_path, W1, S1, solve_runner=solve, stamp="c")
    assert _capture_args(solve.calls[0]["argv"]) == []
    frames = _ledger(store, "c")["solver_frames"]
    assert frames["from"] is None and wr.CAPTURE_ROOT_ENV in frames["why"]
    assert frames["redacted_session_copies"] == 3
    assert frames["source"] == wr.SOURCE_REDACTED
    assert frames["sources_json_written"] is False
    # the walk's record, copied back, is exactly what it was
    assert json.loads((store.world_dir(W1) / "solve" / S1 / "sources.json").read_text()) \
        == {"k": "v"}


def test_an_explicit_capture_dir_is_searched_by_identity_and_no_capture_turns_it_off(
        tmp_path, stages, monkeypatch):
    store, kids, capture_root = _live_capture_world(tmp_path / "one")
    monkeypatch.setenv(wr.CAPTURE_ROOT_ENV, str(capture_root))
    other = _write_capture(tmp_path / "elsewhere", "e" * 32, [(2, 102.0)])
    wr.refinish(store, tmp_path / "one", W1, S1, solve_runner=_Solve(store, kids),
                stamp="c", capture_dirs=[other])
    frames = _ledger(store, "c")["solver_frames"]
    assert frames["from"] == wr.CAPTURE_FROM_ARGUMENT
    assert frames["raw_from_capture_dir"] == 1 and frames["redacted_session_copies"] == 2
    assert frames["source"] == wr.SOURCE_MIXED
    assert _assert_every_frame_is_its_own(store, [("e" * 32, 2, 102.0)]) == 1

    store2, kids2, capture_root2 = _live_capture_world(tmp_path / "two")
    monkeypatch.setenv(wr.CAPTURE_ROOT_ENV, str(capture_root2))
    wr.refinish(store2, tmp_path / "two", W1, S1, solve_runner=_Solve(store2, kids2),
                stamp="d", use_capture=False)
    frames2 = _ledger(store2, "d")["solver_frames"]
    assert frames2["why"] == "--no-capture" and frames2["redacted_session_copies"] == 3


def test_the_builders_record_wins_except_for_an_ambiguous_keyframe_id(tmp_path, stages,
                                                                      monkeypatch):
    a = [(1, 11.0), (2, 12.0)]
    b = [(1, 21.0)]
    keyframes = [(CAPTURE_ID, 1, 11.0), (CAPTURE_ID, 2, 12.0), (CAP_B, 1, 21.0)]
    store, kids, root = _walk(tmp_path, [(CAPTURE_ID, None, a), (CAP_B, CAPTURE_ID, b)],
                              keyframes)
    monkeypatch.setenv(wr.CAPTURE_ROOT_ENV, str(root))
    raw = tmp_path / "walk-raw" / "00000002.jpg"
    raw.parent.mkdir()
    raw.write_bytes(b"raw frame the builder recorded")
    (store.world_dir(W1) / "solve" / S1 / "sources.json").write_text(json.dumps(
        {"sources": {f"{S1}:00000002": str(raw),
                     f"{S1}:00000001": str(root / "captures" / CAP_B / "frames" /
                                           "00000001.jpg")}}))
    report = wr.refinish(store, tmp_path, W1, S1, solve_runner=_Solve(store, kids),
                         stamp="c")
    frames = report["solver_frames"]
    assert _written_sources(store) == {f"{S1}:00000002": str(raw)}
    assert frames["raw_from_sources_json"] == 1
    assert frames["redacted_ambiguous_in_session"] == 2
    assert frames["dropped_builder_entries"] == [f"{S1}:00000001"]


def test_walk_images_whose_provenance_is_unproven_are_withheld_not_reused(
        tmp_path, stages, monkeypatch):
    """Review V9, M-7. This test used to pin the defect: three walk images of unknown
    provenance were counted `already_undistorted`, the plan called its source
    `walk-solver-images`, and the solve kept them over the raw frames found by identity.
    Now an image goes back only when it is proven to be the planned frame (by the
    provenance record, or by reproducing it); these cannot be, so none goes back, and the
    solve writes each keyframe's image from its own raw frame."""
    store, kids, capture_root = _live_capture_world(tmp_path)
    monkeypatch.setenv(wr.CAPTURE_ROOT_ENV, str(capture_root))
    images = store.world_dir(W1) / "solve" / S1 / "images"
    images.mkdir()
    for i in (1, 2, 3):
        (images / f"{i:08d}.jpg").write_bytes(b"the walk's solver image")
    report = wr.refinish(store, tmp_path, W1, S1, solve_runner=_Solve(store, kids),
                         stamp="c")
    frames = report["solver_frames"]
    assert frames["already_undistorted"] == 0
    assert frames["raw_from_capture_dir"] == 3 and frames["source"] == wr.SOURCE_RAW
    assert frames["walk_images"]["carried_back"] == 0
    assert sum(frames["walk_images"]["withheld"].values()) == 3
    assert not any((store.world_dir(W1) / "solve" / S1 / "images").iterdir())
    # The set-aside originals are untouched.
    aside = store.world_dir(W1) / "refinish" / "c" / "solve" / S1 / "images"
    assert sorted(p.read_bytes() for p in aside.iterdir()) == [b"the walk's solver image"] * 3


def test_capture_resolution_rules(tmp_path, monkeypatch):
    import dataclasses

    store, _kids, capture_root = _live_capture_world(tmp_path)
    # A relative capture root is anchored where sources.json paths are, never the cwd.
    monkeypatch.setenv("TOWER_SOURCES_ROOT", str(tmp_path))
    monkeypatch.setenv(wr.CAPTURE_ROOT_ENV, "caproot")
    monkeypatch.chdir(capture_root)
    got = wr.resolve_capture_dirs(store, W1, S1)
    assert got["capture_dirs"] == [str(tmp_path / "caproot" / "captures" / CAPTURE_ID)]
    # Blank is unset.
    monkeypatch.setenv(wr.CAPTURE_ROOT_ENV, "   ")
    assert wr.resolve_capture_dirs(store, W1, S1)["capture_dirs"] == []
    # A capture root without this capture's frames.
    monkeypatch.setenv(wr.CAPTURE_ROOT_ENV, str(tmp_path / "nowhere"))
    got = wr.resolve_capture_dirs(store, W1, S1)
    assert got["capture_dirs"] == [] and "no capture frames" in got["why"]
    # A session that was not a live capture records none.
    monkeypatch.setenv(wr.CAPTURE_ROOT_ENV, str(capture_root))
    store.write_session(dataclasses.replace(store.read_session(W1, S1), capture_id=None))
    got = wr.resolve_capture_dirs(store, W1, S1)
    assert got["capture_dirs"] == [] and "no capture" in got["why"]
    # A capture id that is not a plain name never walks out of the capture root.
    store.write_session(dataclasses.replace(store.read_session(W1, S1),
                                            capture_id="../" + CAPTURE_ID))
    got = wr.resolve_capture_dirs(store, W1, S1)
    assert got["capture_dirs"] == [] and "plain" in got["why"]


def test_the_chain_is_every_capture_continuing_the_sessions_and_nothing_else(tmp_path,
                                                                            monkeypatch):
    store, _kids, root = _live_capture_world(tmp_path)
    _write_capture(root, CAP_B, [(1, 201.0)], continues=CAPTURE_ID, started_at=200.0)
    _write_capture(root, CAP_C, [(1, 301.0)], continues=CAP_B, started_at=300.0)
    _write_capture(root, "f" * 32, [(1, 251.0)], continues=CAP_B, started_at=250.0)  # fork
    _write_capture(root, "d" * 32, [(1, 401.0)], started_at=400.0)                   # unrelated
    monkeypatch.setenv(wr.CAPTURE_ROOT_ENV, str(root))
    got = wr.resolve_capture_dirs(store, W1, S1)
    assert [c["capture_id"] for c in got["captures"]] == [CAPTURE_ID, CAP_B, "f" * 32, CAP_C]
    assert all(c["retains_raw_imagery"] is True for c in got["captures"])


def test_the_cli_dry_run_names_the_frames_and_writes_nothing(tmp_path, capsys, monkeypatch):
    store, _kids, capture_root = _live_capture_world(tmp_path)
    monkeypatch.delenv(wr.CAPTURE_ROOT_ENV, raising=False)
    cap = capture_root / "captures" / CAPTURE_ID
    before = _files(tmp_path)
    assert wr.main(["--root", str(tmp_path), "--world", W1, "--dry-run",
                    "--capture-dir", str(cap)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["solver_frames"]["from"] == wr.CAPTURE_FROM_ARGUMENT
    assert out["solver_frames"]["raw_from_capture_dir"] == 3
    assert out["solver_frames"]["sources_changed"] is True
    assert "sources" not in out["solver_frames"]
    assert _files(tmp_path) == before
    with pytest.raises(SystemExit):
        wr.main(["--root", str(tmp_path), "--world", W1, "--dry-run",
                 "--capture-dir", str(cap), "--no-capture"])


# ---------------------------------------------------------------------------
# FIN (review V8): the ledger names its process and ends in a terminal state; a
# re-finish restarts the re-gate counter too
# ---------------------------------------------------------------------------


def test_the_ledger_names_its_process_and_ends_done(tmp_path, stages):
    store, kids = _old_world(tmp_path)
    seen = {}

    def solve(argv, env=None, capture_output=True, text=True):
        seen["during"] = _ledger(store, "p")
        return _Solve(store, kids)(argv, env=env, capture_output=capture_output, text=text)

    report = wr.refinish(store, tmp_path, W1, S1, solve_runner=solve, stamp="p")
    assert report["done"] is True
    during, after = seen["during"], _ledger(store, "p")
    assert during["process"]["pid"] == os.getpid()
    assert during["state"] == wr.LEDGER_SET_ASIDE
    assert wr.refinish_process_alive(during) is True          # this process, running
    assert after["state"] == wr.LEDGER_DONE and after["state"] in wr.LEDGER_TERMINAL_STATES
    assert "published_at" in after and "done_at" in after


def test_a_stopped_rebuild_ends_stopped(tmp_path, stages):
    store, kids = _old_world(tmp_path)
    report = wr.refinish(store, tmp_path, W1, S1, solve_runner=_Solve(store, kids),
                         stamp="p", should_stop=lambda: True)
    assert report["done"] is False
    assert _ledger(store, "p")["state"] == wr.LEDGER_STOPPED


def test_a_ledger_whose_process_is_gone_reads_dead(tmp_path):
    import subprocess

    from tower.world_builder.store import _lock_record

    gone = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        record = _lock_record(gone.pid)
    finally:
        gone.kill()
        gone.wait(timeout=30)
    assert "created_at" in record
    assert wr.refinish_process_alive({"process": record}) is False
    # a pid now naming ANOTHER process (recycled) reads dead by its start time
    me = _lock_record(os.getpid())
    assert wr.refinish_process_alive(
        {"process": {"pid": me["pid"], "created_at": me["created_at"] - 3600.0}}) is False
    assert wr.refinish_process_alive({"state": "set-aside"}) is None     # an old ledger


def test_a_refinish_restarts_the_regate_counter_too(tmp_path, stages):
    store, kids = _old_world(tmp_path)
    exhausted = {"attempts": 3, "forgiven": 0, "detail": "re-gate failed"}
    keys = (S1, wfp.area_ledger_key(S1), wfp.regate_ledger_key(S1))
    wfp._write_ledger(store, W1, {k: dict(exhausted) for k in keys})
    wr.refinish(store, tmp_path, W1, S1, solve_runner=_Solve(store, kids), stamp="p")
    sessions, _unreadable = wfp._read_ledger(store, W1)
    for key in keys:
        assert sessions[key]["attempts"] == 0, key
    assert _ledger(store, "p")["previous"]["finish_attempts"][wfp.regate_ledger_key(S1)] \
        == exhausted
