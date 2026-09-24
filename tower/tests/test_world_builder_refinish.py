"""The re-finish command (WORLD-BUILDER-COMPONENTS.md §7 rule 4, contract T11).

Run on a fixture world built in `tmp_path` -- never the live store. The solve child
(`world_finalize.py` with the product settings) and the GPU stages are faked; what is
pinned is the order, the settings the solve is given, that nothing is deleted, and that
components and areas follow.
"""

from __future__ import annotations

import json
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
        if self.gate_writes and self.code == 0:
            from tests.test_world_builder_area_build import _components
            from tower.world_builder.global_solve import (
                load_solution,
                workspace_for,
                write_solution,
            )

            # The solve is published afresh into the (now empty) solve directory.
            old = self.store.world_dir(W1) / wr.REFINISH_DIRNAME
            stamp = next(old.iterdir()).name
            from tower.world_builder.store import WorldStore

            moved = WorldStore(self.store.root)
            sol = load_solution(_ShadowStore(moved, old / stamp), W1, S1)
            write_solution(workspace_for(self.store, W1, S1), sol)
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


def test_a_failed_solve_stops_and_leaves_the_set_aside_named(tmp_path, stages):
    store, kids = _old_world(tmp_path)
    report = wr.refinish(store, tmp_path, W1, S1,
                         solve_runner=_Solve(store, kids, code=1), stamp="x")
    assert report["done"] is False
    assert stages == []
    assert Path(report["set_aside"]["moved"][0]["to"]).exists()


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
