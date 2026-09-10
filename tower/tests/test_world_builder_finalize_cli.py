"""Finishing a session whose finalization did not finish.

Before `scripts/world_finalize.py` there was no such operation. The only
caller of `mark_finalization` was the builder's own `finally`, so a session
whose builder died between the last keyframe and the finished world could
not be repaired by anything the system supported: re-running
`world_build_session.py` opens a NEW session against the same world, and
`world_solve.py` writes a solution that nothing merges, because
`engine.build()` is the sole writer of the derived tree.

The 2026-09-09 walk landed exactly there -- 795 keyframes and 26,634 points
on disk under `finalization: interrupted`, `BadZipFile: File is not a zip
file` -- and recovering it took three hand-written steps in a scratch
directory. These tests pin those three steps as an operation.
"""

import json
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import world_finalize  # noqa: E402
from tower.world_builder.engine import WorldBuilderEngine  # noqa: E402
from tower.world_builder.records import (  # noqa: E402
    FINALIZATION_COMPLETE,
    FINALIZATION_INTERRUPTED,
)
from tower.world_builder.records import Keyframe  # noqa: E402
from tower.world_builder.store import WorldStore  # noqa: E402


def _jpeg() -> bytes:
    rng = np.random.default_rng(0)
    ok, buf = cv2.imencode(".jpg", rng.integers(0, 255, (640, 360, 3), dtype=np.uint8))
    assert ok
    return buf.tobytes()


@pytest.fixture
def interrupted_world(tmp_path):
    """A world in the shape the field walk left behind.

    A finished capture whose session record says `interrupted` with a
    detail naming an exception, and whose authoritative journals and
    images are entirely intact. That combination is the whole point: the
    data is fine, the record is not, and nothing could act on it.
    """
    root = tmp_path / "worlds"
    store = WorldStore(root)
    engine = WorldBuilderEngine(store)
    world_id = engine.create_world()
    session_id = engine.start_session(world_id, frame_source="synthetic")
    jpeg = _jpeg()
    for i in range(6):
        keyframe = Keyframe(
            keyframe_id=f"{session_id}:{i:08d}", session_id=session_id, source_seq=i,
            received_at=float(i), image_relpath=f"images/{i:08d}.jpg",
            width=360, height=640, byte_count=len(jpeg), segment_index=0,
        )
        store.append_keyframe(world_id, keyframe)
        path = store.session_dir(world_id, session_id) / keyframe.image_relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(jpeg)
    engine.stop_session("error", hold_lock=True)
    engine.mark_finalization(
        world_id, session_id,
        state=FINALIZATION_INTERRUPTED, final_solve=None,
        detail="BadZipFile: File is not a zip file",
    )
    engine.release_world(world_id)
    return root, world_id, session_id


def _run(root, world_id, *extra) -> dict:
    """The CLI, in-process, returning its JSON report."""
    import io
    import contextlib

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = world_finalize.main(
            ["--root", str(root), "--world", world_id, *extra]
        )
    return {"exit": code, **json.loads(out.getvalue())}


# ---------------------------------------------------------------------------


def test_it_finishes_a_session_the_builder_could_not(interrupted_world):
    root, world_id, session_id = interrupted_world
    report = _run(root, world_id, "--skip-solve")

    assert report["exit"] == 0
    assert report["finalized"] is True
    assert report["was"]["state"] == FINALIZATION_INTERRUPTED
    assert report["was"]["detail"] == "BadZipFile: File is not a zip file"
    assert report["now"]["state"] == FINALIZATION_COMPLETE

    store = WorldStore(root)
    assert store.read_session(world_id, session_id).finalization["state"] == (
        FINALIZATION_COMPLETE
    )


def test_it_rebuilds_the_derived_tree(interrupted_world):
    """The record moving is not the point; the world becoming readable is.

    The derived tree is what the phone reads, so a repair that only
    rewrote the session record would report success and change nothing
    the user can see.
    """
    root, world_id, session_id = interrupted_world
    store = WorldStore(root)
    store.clear_derived(world_id)
    assert store.read_derived_manifest(world_id) is None

    report = _run(root, world_id, "--skip-solve")
    assert report["finalized"] is True
    manifest = store.read_derived_manifest(world_id)
    assert manifest is not None
    assert manifest["session_id"] == session_id
    assert manifest["keyframes"] == 6
    assert report["build"]["keyframes"] == 6


def test_it_defaults_to_the_most_recent_session(interrupted_world):
    root, world_id, session_id = interrupted_world
    report = _run(root, world_id, "--skip-solve")
    assert report["session_id"] == session_id


def test_running_it_twice_produces_the_same_world(interrupted_world):
    """Idempotent, because every step rewrites rather than appends.

    A repair that had to be run exactly once would be a worse operation
    than no repair at all: the operator cannot tell from the outside
    whether the first run landed.
    """
    root, world_id, session_id = interrupted_world
    first = _run(root, world_id, "--skip-solve")
    store = WorldStore(root)
    manifest_one = dict(store.read_derived_manifest(world_id))
    keyframes_one = len(store.read_keyframes(world_id, session_id))
    events_one = len(store.read_events(world_id, session_id))

    second = _run(root, world_id, "--skip-solve")
    manifest_two = dict(store.read_derived_manifest(world_id))

    assert second["finalized"] is True
    assert first["build"] == second["build"]
    # The journals are AUTHORITATIVE and must not grow.
    assert len(store.read_keyframes(world_id, session_id)) == keyframes_one
    assert len(store.read_events(world_id, session_id)) == events_one
    for key in ("keyframes", "points", "poses_solved", "segments", "input_digest"):
        assert manifest_one[key] == manifest_two[key]


def test_it_refuses_a_world_a_live_builder_is_holding(interrupted_world, monkeypatch):
    """The lock is the only thing standing between this and a live walk.

    A repair that wrote underneath a running builder would corrupt the
    session it was meant to rescue.

    The holder must be a genuinely live OTHER process, and both halves of
    that matter. `acquire_writer_lock` reclaims a lock naming a dead pid,
    and it also reclaims one naming `os.getpid()` -- deliberately, because
    the builder retakes its own lock after `stop_session(hold_lock=True)`.
    So a same-process test would prove nothing: it would be reclaimed by
    the self-reclaim branch and pass for the wrong reason. In production
    the builder is a subprocess, which is what this stages.
    """
    import subprocess
    import sys as _sys

    from tower.world_builder import store as store_module

    root, world_id, _ = interrupted_world
    store = WorldStore(root)
    holder = subprocess.Popen(
        [_sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=subprocess.DEVNULL,
    )
    try:
        # Write the lock as though that live process had taken it.
        monkeypatch.setattr(store_module.os, "getpid", lambda: holder.pid)
        store.acquire_writer_lock(world_id)
        monkeypatch.undo()
        assert store.lock_holder(world_id)["alive"] is True

        report = _run(root, world_id, "--skip-solve")
        assert report["exit"] == 1
        assert report["finalized"] is False
        assert str(holder.pid) in report["reason"]
        # And the world was left alone.
        assert store.read_session(world_id, report["session_id"]).finalization[
            "state"
        ] == FINALIZATION_INTERRUPTED
    finally:
        holder.terminate()
        holder.wait(timeout=10)


def test_it_releases_the_lock_afterwards(interrupted_world):
    root, world_id, _ = interrupted_world
    _run(root, world_id, "--skip-solve")
    store = WorldStore(root)
    assert not store.lock_path(world_id).exists()
    # And the proof that matters: a new session can be started.
    engine = WorldBuilderEngine(store)
    engine.start_session(world_id, frame_source="synthetic")
    engine.stop_session("stopped")


def test_a_world_with_no_sessions_is_refused_not_crashed(tmp_path):
    root = tmp_path / "worlds"
    engine = WorldBuilderEngine(WorldStore(root))
    world_id = engine.create_world()
    report = _run(root, world_id, "--skip-solve")
    assert report["exit"] == 1
    assert report["finalized"] is False


def test_the_root_flag_routes_through_the_artifact_guard():
    """CLAUDE.md's filesystem policy, enforced for this CLI too.

    `tests/test_artifact_paths.py` scans `scripts/` for any `--root` that
    skips `artifact_root_arg`; this asserts it directly so a reader of
    this file sees the requirement.
    """
    import inspect

    source = inspect.getsource(world_finalize.main)
    assert '"--root", type=artifact_root_arg' in source


# ---------------------------------------------------------------------------
# The repair has to reach the SURFACE, not just the record.


def test_a_repaired_session_stops_reading_as_interrupted(interrupted_world):
    """Otherwise this tool cannot deliver what it exists for.

    `end_reason` describes the CAPTURE; `finalization` describes the WORLD.
    The lifecycle classifier answered both with the first, so a session whose
    capture ended badly could never be reported as finished however it was
    repaired. Measured on the recovered 2026-09-09 artifact before the fix:
    finalization complete, final solve solved, 88 of 122 segments registered
    -- and the phone still said Interrupted.

    `end_reason` is deliberately NOT rewritten. That walk really did end in
    an error.
    """
    from tower.results.world_builder import _lifecycle

    root, world_id, session_id = interrupted_world
    store = WorldStore(root)

    before = _lifecycle(
        holder=None, stopped=True, session=store.read_session(world_id, session_id),
        geometry_current=True, has_manifest=True, has_session_geometry=True,
    )
    assert before["state"] == "interrupted"

    # A repair that solved nothing must NOT flip the label -- only a
    # finalization that actually finished, with a final solve that landed.
    _run(root, world_id, "--skip-solve")
    skipped = _lifecycle(
        holder=None, stopped=True, session=store.read_session(world_id, session_id),
        geometry_current=True, has_manifest=True, has_session_geometry=True,
    )
    assert skipped["state"] == "interrupted", (
        "a repair that skipped the solve claimed the world was finished"
    )

    # Now the real thing: record a solved final solve the way a successful
    # repair does, and the surface follows.
    engine = WorldBuilderEngine(store)
    engine.mark_finalization(
        world_id, session_id,
        state=FINALIZATION_COMPLETE, final_solve="solved", detail=None,
    )
    after = _lifecycle(
        holder=None, stopped=True, session=store.read_session(world_id, session_id),
        geometry_current=True, has_manifest=True, has_session_geometry=True,
    )
    assert after["state"] == "ready"
    assert store.read_session(world_id, session_id).end_reason == "error", (
        "the capture's end_reason was rewritten; that erases what happened"
    )
    assert "'error'" in after["reason"], "the reason hides the interrupted capture"


def test_it_registers_when_the_solve_placed_nothing(interrupted_world):
    """The fallback the builder has and this tool did not.

    When no global solve placed anything, the Sim3 registrar is the only
    producer of placements there is, and `world_build_session.py` runs it
    for exactly that case. Without it a repair rebuilt the derived tree and
    left the world with no placements at all -- every fragment its own
    island, which is the outcome this whole campaign is about. Found by an
    adversarial review of the repair tool.
    """
    root, world_id, _session_id = interrupted_world
    report = _run(root, world_id, "--skip-solve")
    assert report["finalized"] is True
    assert report["build"]["placements_source"] is None
    # It was ASKED. What the registrar could place from a six-keyframe
    # synthetic world is its own business; that it ran is this test's.
    assert report["registration"]["attempted"] is True


def test_it_does_not_re_register_over_a_global_solve(interrupted_world, monkeypatch):
    """And the guard the builder uses is the guard this uses."""
    from tower.world_builder import global_solve as gs

    root, world_id, _session_id = interrupted_world

    class _Merged:
        pose_rows: list = []
        point_rows: list = []
        support_rows: list = []
        summary: dict = {}
        segments: dict = {}

    from tower.world_builder.records import SegmentPlacement

    _Merged.placements = [SegmentPlacement(
        segment_index=0, state="registered",
        rotation_wxyz=(1.0, 0.0, 0.0, 0.0), translation=(0.0, 0.0, 0.0),
        scale=1.0, reference_segment=0, refusal_reason=None,
        input_digest="d", evidence={}, frame_revision=1,
    )]
    monkeypatch.setattr(gs, "load_solution", lambda *a, **k: object())
    monkeypatch.setattr(gs, "merge", lambda *a, **k: _Merged())

    report = _run(root, world_id, "--skip-solve")
    assert report["build"]["placements_source"] == "global_solve"
    assert report["registration"]["attempted"] is False


def test_a_failed_repair_does_not_downgrade_a_healthy_record(interrupted_world):
    """A repair that fails must leave the world as it found it.

    This wrote `interrupted` unconditionally, so pointing the tool at an
    already-complete world and hitting any error -- a purged world, a full
    disk, a raising solve -- downgraded a healthy record permanently, with
    no way back: every re-run hits the same error. An adversarial review
    demonstrated it on a purged world.
    """
    root, world_id, session_id = interrupted_world
    store = WorldStore(root)

    # Get it healthy first.
    assert _run(root, world_id, "--skip-solve")["finalized"] is True
    engine = WorldBuilderEngine(store)
    engine.mark_finalization(
        world_id, session_id,
        state=FINALIZATION_COMPLETE, final_solve="solved", detail=None,
    )
    healthy = dict(store.read_session(world_id, session_id).finalization)

    # Now make the rebuild fail the way a purged world does.
    world = store.read_world(world_id)
    store.write_world(replace(world, images_purged=True))

    report = _run(root, world_id, "--skip-solve")
    assert report["finalized"] is False
    assert "ImagesPurgedError" in report["reason"]

    after = store.read_session(world_id, session_id).finalization
    assert after["state"] == FINALIZATION_COMPLETE, (
        "a failed repair downgraded a healthy record, and there is no way back"
    )
    assert after["final_solve"] == healthy["final_solve"]
    assert after["detail"] == healthy["detail"]


def test_a_failed_repair_on_an_already_broken_world_still_says_so(interrupted_world):
    """The other half: nothing to preserve, so record the failure."""
    root, world_id, session_id = interrupted_world
    store = WorldStore(root)
    world = store.read_world(world_id)
    store.write_world(replace(world, images_purged=True))

    report = _run(root, world_id, "--skip-solve")
    assert report["finalized"] is False
    after = store.read_session(world_id, session_id).finalization
    assert after["state"] == FINALIZATION_INTERRUPTED
    assert "ImagesPurgedError" in after["detail"]


def test_an_older_session_of_a_world_walked_twice_is_still_openable(tmp_path):
    """The manifest is the WORLD's; the geometry the phone opens is the
    SESSION's. A guard that asks the first when it means the second calls
    every older session of a multi-session world unopenable.

    Measured by a reviewer on two synthetic sessions in one world, both
    finalized `complete`, both derived trees on disk: the older one read
    `interrupted` over a reason saying its geometry was "no longer on
    disk". That is worse than the hole the guard closed -- that one lied
    about a world with nothing in it, this lied about an intact one and
    invited the wearer to redo the walk.

    So this test builds the state on real disk rather than passing flags:
    the flags are exactly what got it wrong.
    """
    from tower.results.world_builder import _has_session_geometry, _lifecycle
    from tower.world_builder.records import Session
    from tower.world_builder.store import WorldStore

    store = WorldStore(tmp_path / "worlds")
    world_id = "w" * 32
    older, newer = "a" * 32, "b" * 32
    for session_id in (older, newer):
        derived = store.derived_dir(world_id) / session_id
        derived.mkdir(parents=True)
        (derived / "poses.json").write_text("{}", encoding="utf-8")
        (derived / "points.json").write_text("{}", encoding="utf-8")
    # ONE manifest, naming the session that built last -- which is what the
    # producer hands `_lifecycle`, and it discards it for the other session.
    store.write_derived_manifest(world_id, {"session_id": newer})

    def lifecycle_for(session_id, has_manifest):
        session = Session(
            session_id=session_id, world_id=world_id, started_at=0.0,
            frame_source="synthetic", ended_at=1.0, end_reason="stop",
            finalization={
                "state": FINALIZATION_COMPLETE, "final_solve": "solved",
                "started_at": 0.0, "updated_at": 1.0, "detail": None,
            },
        )
        return _lifecycle(
            holder=None, stopped=True, session=session,
            geometry_current=has_manifest, has_manifest=has_manifest,
            has_session_geometry=_has_session_geometry(store, world_id, session_id),
        )

    assert lifecycle_for(newer, True)["state"] == "ready"
    older_lifecycle = lifecycle_for(older, False)
    assert older_lifecycle["state"] == "ready", (
        "the older session of a world walked twice reads "
        f"{older_lifecycle['state']!r} with its poses.json right there: "
        f"{older_lifecycle['reason']}"
    )

    # And the refusal still works, on the thing it is actually about.
    for name in ("poses.json", "points.json"):
        (store.derived_dir(world_id) / older / name).unlink()
    assert lifecycle_for(older, False)["state"] == "interrupted"


def test_an_older_session_with_no_finalization_record_is_still_openable(tmp_path):
    """The same bug, one branch further down, which the first fix left.

    `if not has_manifest: return stopped_unbuilt` was never touched, so a
    session with geometry on disk and NO finalization record read
    `stopped_unbuilt` -- "no geometry has been built for this session yet",
    projected to the phone as a permanent `finalizing`. Verbatim the
    outcome the branch above spends a paragraph forbidding, reached by a
    different door, and found by a reviewer walking the branches the fix
    did not.

    `finalization is None` is the common case for anything offline:
    `stop_session` writes the block only when `hold_lock=True`.
    """
    from tower.results.world_builder import (
        _MODEL_STATE_BY_LIFECYCLE,
        MODEL_STATE_IDLE,
        _has_session_geometry,
        _lifecycle,
    )
    from tower.world_builder.records import Session
    from tower.world_builder.store import WorldStore

    store = WorldStore(tmp_path / "worlds")
    world_id, session_id = "w" * 32, "a" * 32
    derived = store.derived_dir(world_id) / session_id
    derived.mkdir(parents=True)
    for name in ("poses.json", "points.json"):
        (derived / name).write_text("{}", encoding="utf-8")
    # The manifest names a DIFFERENT session, as it does for any world
    # walked twice.
    store.write_derived_manifest(world_id, {"session_id": "b" * 32})

    session = Session(
        session_id=session_id, world_id=world_id, started_at=0.0,
        frame_source="synthetic", ended_at=1.0, end_reason="stop",
        finalization=None,
    )
    lifecycle = _lifecycle(
        holder=None, stopped=True, session=session,
        geometry_current=False, has_manifest=False,
        has_session_geometry=_has_session_geometry(store, world_id, session_id),
    )
    assert lifecycle["state"] == "ready", (
        f"a session with its geometry on disk reads {lifecycle['state']!r}: "
        f"{lifecycle['reason']}"
    )
    projected = _MODEL_STATE_BY_LIFECYCLE.get(lifecycle["state"], MODEL_STATE_IDLE)
    assert projected != "finalizing", (
        "the phone is told to keep waiting for a build that finished"
    )

    # And a session with nothing on disk still says so.
    for name in ("poses.json", "points.json"):
        (derived / name).unlink()
    empty = _lifecycle(
        holder=None, stopped=True, session=session,
        geometry_current=False, has_manifest=False,
        has_session_geometry=_has_session_geometry(store, world_id, session_id),
    )
    assert empty["state"] == "stopped_unbuilt"


def test_the_three_surfaces_ask_the_same_question_of_the_same_files(tmp_path):
    """`session_state`, `_lifecycle` and the render page disagreeing is how
    a picker row, the canvas it opens and the page it draws end up saying
    different things about one session.

    THREE copies, not two -- a reviewer counted while the docstring beside
    one of them still said "if a third caller appears, move it into the
    store". The third is `world_builder_render._has_geometry`, on the
    surface the wearer actually looks at, and the first version of this
    test left it free to drift. The modules must not import each other, so
    the copies stay; this is what keeps them honest.
    """
    from tower.results.world_builder import _has_session_geometry
    from tower.results.world_builder_library import _has_geometry as library_has
    from tower.results.world_builder_render import _has_geometry as render_has
    from tower.world_builder.store import WorldStore

    store = WorldStore(tmp_path / "worlds")
    world_id, session_id = "w" * 32, "s" * 32
    derived = store.derived_dir(world_id) / session_id
    derived.mkdir(parents=True)

    for present in ([], ["poses.json"], ["points.json"], ["poses.json", "points.json"]):
        for name in ("poses.json", "points.json"):
            path = derived / name
            if name in present:
                path.write_text("{}", encoding="utf-8")
            else:
                path.unlink(missing_ok=True)
        answers = {
            "status": _has_session_geometry(store, world_id, session_id),
            "listing": library_has(store, world_id, session_id),
            "render": render_has(store, world_id, session_id),
        }
        assert len(set(answers.values())) == 1, (present, answers)


def test_the_status_predicate_answers_existence_not_openability(tmp_path):
    """And the docstring says so, because a reviewer caught it claiming
    otherwise.

    `_has_session_geometry` stats two files; the serving path
    (`WorldStore.read_derived`) opens and parses them. Empty, truncated or
    wrong-shaped files therefore read `ready` here and 404 there. That is a
    real gap and it is deliberate -- this runs on the 0.5 s status poll and
    `points.json` is megabytes on a real walk -- but "the files the phone
    actually opens" was the wrong way to describe it.

    What must NOT happen is the route turning a corrupt tree into a 500.
    Both are "nothing to serve"; neither is a server fault.
    """
    from tower.results.world_builder import _has_session_geometry
    from tower.world_builder.store import WorldStore

    store = WorldStore(tmp_path / "worlds")
    world_id, session_id = "w" * 32, "s" * 32
    derived = store.derived_dir(world_id) / session_id
    derived.mkdir(parents=True)
    store.write_derived_manifest(world_id, {"session_id": session_id})

    for shape in ("", "{}", "[]", '{"poses": [1', "null"):
        (derived / "poses.json").write_text(shape, encoding="utf-8")
        (derived / "points.json").write_text(shape, encoding="utf-8")
        assert _has_session_geometry(store, world_id, session_id) is True
        assert store.read_derived(world_id, session_id) is None, (
            f"a derived tree holding {shape!r} was served rather than refused"
        )


@pytest.mark.parametrize(
    "has_manifest, expected",
    [(True, "ready"), (False, "interrupted")],
)
def test_ready_is_not_claimed_over_geometry_that_is_not_there(has_manifest, expected):
    """`ready` on a world with nothing to open is the same class of lie as
    "Nothing mapped yet" over 26,634 points, pointing the other way.

    The first version of this branch returned READY without consulting
    `has_manifest`, and an adversarial review reached it end to end: a
    repair whose build failed left `complete / solved` on a world whose
    derived tree it had just deleted, and the phone read `ready`.
    """
    from tower.results.world_builder import _lifecycle
    from tower.world_builder.records import Session

    session = Session(
        session_id="s" * 32, world_id="w" * 32, started_at=0.0,
        frame_source="synthetic", ended_at=1.0, end_reason="error",
        finalization={
            "state": FINALIZATION_COMPLETE, "final_solve": "solved",
            "started_at": 0.0, "updated_at": 1.0, "detail": None,
        },
    )
    lifecycle = _lifecycle(
        holder=None, stopped=True, session=session,
        geometry_current=True, has_manifest=has_manifest,
        has_session_geometry=has_manifest,
    )
    assert lifecycle["state"] == expected


def _finished_stop_session():
    """Finalization complete and solved, capture ended the way a wearer ends
    it. Only the derived tree varies."""
    from tower.world_builder.records import Session

    return Session(
        session_id="s" * 32, world_id="w" * 32, started_at=0.0,
        frame_source="synthetic", ended_at=1.0, end_reason="stop",
        finalization={
            "state": FINALIZATION_COMPLETE, "final_solve": "solved",
            "started_at": 0.0, "updated_at": 1.0, "detail": None,
        },
    )


def test_the_ordinary_stop_path_is_ready_when_the_geometry_is_there():
    from tower.results.world_builder import _lifecycle

    lifecycle = _lifecycle(
        holder=None, stopped=True, session=_finished_stop_session(),
        geometry_current=True, has_manifest=True, has_session_geometry=True,
    )
    assert lifecycle["state"] == "ready"


def test_the_ordinary_stop_path_does_not_claim_ready_over_nothing():
    """The same hole, on the branch nearly every real world takes.

    The first guard went onto the `error`/`interrupted` branch, which is the
    one the recovered field artifact travels. The branch a wearer who simply
    presses Stop travels -- finalization complete, lock released -- did not
    check at all, and a code comment claimed that was "noted in the handoff".
    It was not. An audit of the handoff caught the CLAIM; the hole was still
    open, one `and` away from the fix already sitting twenty lines above it.

    REFUSING `ready` IS NOT ENOUGH, AND THE FIRST FIX STOPPED THERE. It let
    the refusal fall through to `stopped_unbuilt`, and a reviewer took that
    end to end: `stopped_unbuilt` maps to `model_state: finalizing`, which
    `WorldPresentation` renders as a PERMANENT "Finalizing" -- a stage it
    reads as still-changing, with the explaining sentence suppressed --
    over a world where nothing is running and nothing ever will be. It also
    hard-codes `finalization: None`, deleting the only evidence that the
    solve ever completed on its way to the phone.

    So the assertions here are on all three, not on the state alone. The
    narrow version of this test passed against the broken fall-through.
    """
    from tower.results.world_builder import _lifecycle

    lifecycle = _lifecycle(
        holder=None, stopped=True, session=_finished_stop_session(),
        geometry_current=True, has_manifest=False, has_session_geometry=False,
    )
    assert lifecycle["state"] == "interrupted", (
        "a world with nothing to open read as finished, or as still working"
    )
    assert lifecycle["finalization"] is not None, (
        "the finalization record was dropped: the phone cannot see that the "
        "solve completed"
    )
    assert "yet" not in (lifecycle["reason"] or ""), (
        "the reason says the build has not happened; it happened and its "
        "output is gone"
    )
    assert "rebuilt" in (lifecycle["reason"] or ""), (
        "the reason does not say what can be done about it"
    )


def test_the_phone_is_not_told_a_dead_world_is_still_finalizing():
    """The projection, not the classifier -- the defect was only visible there.

    `_lifecycle` returning the wrong state is a fact about a dict. What
    made it a defect was `model_state`, which is what the phone switches
    on: `stopped_unbuilt` becomes `finalizing`, and iOS reads that stage as
    a world the Tower may still change.
    """
    from tower.results.world_builder import (
        _MODEL_STATE_BY_LIFECYCLE,
        MODEL_STATE_IDLE,
        _lifecycle,
    )

    lifecycle = _lifecycle(
        holder=None, stopped=True, session=_finished_stop_session(),
        geometry_current=True, has_manifest=False, has_session_geometry=False,
    )
    projected = _MODEL_STATE_BY_LIFECYCLE.get(lifecycle["state"], MODEL_STATE_IDLE)
    assert projected != "finalizing", (
        "the phone is told to keep waiting for a build that already finished"
    )


def test_the_listing_does_not_call_a_vanished_world_complete():
    """The same hole on the surface a person CHOOSES a walk from.

    `session_state`'s docstring says it "Mirrors `_lifecycle` in the status
    producer". It did not: a complete finalization record short-circuited
    past its own `has_geometry` check, so the picker badge read "Complete"
    over a session with nothing behind it. Found by the reviewer who took
    the classifier fix end to end, on the more damaging of the two surfaces.
    """
    from tower.results.world_builder_library import session_state

    session = _finished_stop_session()
    assert session_state(session, live=False, has_geometry=True) == "complete"
    assert session_state(session, live=False, has_geometry=False) != "complete", (
        "the picker offers a walk that cannot be opened"
    )


def test_a_failed_repair_that_lost_the_derived_tree_does_not_stay_complete(
    interrupted_world, monkeypatch
):
    """The other half of the same hole, at the source.

    `solve()` and `build()` are WRITERS and have already run by the time
    the failure is handled, so a failure between them can leave the tree
    deleted while the preserved record still says `complete`.
    """
    root, world_id, session_id = interrupted_world
    store = WorldStore(root)

    assert _run(root, world_id, "--skip-solve")["finalized"] is True
    engine = WorldBuilderEngine(store)
    engine.mark_finalization(
        world_id, session_id,
        state=FINALIZATION_COMPLETE, final_solve="solved", detail=None,
    )
    assert store.read_derived_manifest(world_id) is not None

    # A build that destroys the tree and then fails.
    def wreck(self, world, session):
        store.clear_derived(world)
        raise OSError("the disk went away")

    monkeypatch.setattr(WorldBuilderEngine, "build", wreck)
    report = _run(root, world_id, "--skip-solve")

    assert report["finalized"] is False
    assert store.read_derived_manifest(world_id) is None
    assert store.read_session(world_id, session_id).finalization["state"] == (
        FINALIZATION_INTERRUPTED
    ), "the record still claims a complete world whose geometry is gone"


def test_the_geometry_block_does_not_say_no_build_ran_over_a_built_session():
    """`ready` beside "no build has run for this session" is the Tower
    disagreeing with itself, which is the failure this campaign is named
    after, pointing the other way.

    The lifecycle branch that reports an older session of a world walked
    twice as `ready` was added first; the geometry block beside it still
    read the WORLD's manifest, found none for this session, and said no
    build had run -- over a derived tree on disk. Found by the lead
    inspecting its own fix rather than by a reviewer, which is the only
    reason it did not ship.
    """
    from tower.results.world_builder import _geometry_block

    absent = _geometry_block(None, False, 10, has_session_geometry=False)
    built = _geometry_block(None, False, 10, has_session_geometry=True)

    assert absent["available"] is False and built["available"] is False
    assert "no build has run" in absent["unavailable_reason"]
    assert "no build has run" not in built["unavailable_reason"], (
        "a session with its geometry on disk is told no build ran: "
        + built["unavailable_reason"]
    )
    assert "another session" in built["unavailable_reason"]


def test_a_second_walk_does_not_take_the_first_walks_geometry_away(tmp_path):
    """The root of the whole family, fixed at the root.

    A world has ONE `derived/manifest.json` and it names whichever session
    built last. The status producer correctly refuses to attribute it to
    another session -- and then had no figures at all, so an older session
    of a world walked twice reported no geometry, no poses and no currency,
    and the phone rendered a red "Needs retry" over a reconstruction on
    disk. Four hand-written branches were added to paper over that, three
    of them wrong, before anyone asked why there was only one copy.

    `write_derived` now writes the manifest beside the poses and points it
    describes. This test builds two real sessions in one world through the
    real engine and asserts the older one still answers for itself.
    """
    from tower.results.world_builder import _lifecycle
    from tower.world_builder.store import WorldStore

    store = WorldStore(tmp_path / "worlds")
    world_id = "w" * 32
    older, newer = "a" * 32, "b" * 32
    for session_id, points in ((older, [[0.0, 0.0, 0.0]]), (newer, [[1.0, 1.0, 1.0]])):
        store.write_derived(
            world_id, session_id,
            poses=[{"keyframe_id": f"{session_id}:1"}],
            points=points,
            manifest={"session_id": session_id, "keyframes": 1, "points": len(points),
                      "input_digest": f"digest-{session_id}"},
        )

    for session_id in (older, newer):
        manifest = store.read_session_manifest(world_id, session_id)
        assert manifest is not None, f"{session_id} cannot describe itself"
        assert manifest["session_id"] == session_id, (
            "a session's own manifest names another session"
        )

    # The world-level one names only the last builder, as it always did.
    assert store.read_derived_manifest(world_id)["session_id"] == newer


def test_the_producer_reads_the_older_sessions_own_manifest(tmp_path, monkeypatch):
    """End to end through the real producer: the figures, the currency and
    the phone-facing state for a session the world manifest does not name."""
    import json as _json

    from tower.results.world_builder import (
        WorldBuilderStatusProducer,
        _keyframes_digest,
    )
    from tower.world_builder.engine import WorldBuilderEngine
    from tower.world_builder.store import WorldStore

    root = tmp_path / "worlds"
    store = WorldStore(root)
    engine = WorldBuilderEngine(store)
    world_id = engine.create_world("twice")

    built = []
    for _ in range(2):
        session_id = engine.start_session(world_id, frame_source="synthetic")
        store.append_keyframes(
            world_id, session_id,
            [{"keyframe_id": f"{session_id}:1", "index": 0, "segment_index": 0,
              "captured_at": 0.0, "width": 8, "height": 8,
              "image_relpath": "images/1.jpg", "pose": None}],
        ) if hasattr(store, "append_keyframes") else None
        store.write_derived(
            world_id, session_id,
            poses=[{"keyframe_id": f"{session_id}:1", "position": [0.0, 0.0, 0.0]}],
            points=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            manifest={
                # The schema the producer validates, not a convenient
                # subset: `_validate_manifest` refuses anything else, and
                # a test that hand-writes a looser shape measures the
                # test's imagination.
                "schema_version": store.read_world(world_id).schema_version,
                "session_id": session_id, "keyframes": 1, "points": 2,
                "poses_solved": 1, "poses_refused": 0, "segments": 1,
                # The producer's OWN digest function, so "current" means
                # what the producer means by it rather than what the test
                # hopes it means.
                "input_digest": _keyframes_digest(store, world_id, session_id),
                "built_at": 0.0, "backend_id": "test",
            },
        )
        engine.stop_session("stop")
        built.append(session_id)
    older, newer = built

    import time

    producer = WorldBuilderStatusProducer(root, time.time)
    payload = producer.snapshot(world_id, older).payload
    assert payload["selection"]["session_id"] == older
    assert payload["geometry"]["available"] is True, (
        "the older session reports no geometry with its manifest right there: "
        + _json.dumps(payload["geometry"])
    )
    assert payload["geometry"]["element_count"] == 2
    assert payload["geometry"]["current"] is True, (
        "currency is unknowable without the session's own manifest, and "
        "that is what this change supplies"
    )
    assert payload["lifecycle"]["state"] == "ready"
    assert payload["model_state"] == "finalized"

    # And the newest session is unaffected: it reads its own copy too.
    newest = producer.snapshot(world_id, newer).payload
    assert newest["geometry"]["available"] is True
    assert newest["lifecycle"]["state"] == "ready"


def test_geometry_and_trajectory_tell_the_same_story_about_one_session():
    """Two blocks, twelve lines apart, describing the same files.

    `_geometry_block` was given a sentence for "a derived tree with no
    manifest describing it"; `_trajectory_block` was not, and still said
    "no build has run for this session" over a `poses.json` on disk. A
    reviewer found it by printing both for one session. Fixing one half of
    a self-contradiction is how the other half survives a review.
    """
    from tower.results.world_builder import (
        WorldBuilderStatusProducer,
        _geometry_block,
    )

    geometry = _geometry_block(None, False, 10, has_session_geometry=True)
    trajectory = WorldBuilderStatusProducer._trajectory_block(
        None, None, None, None, None, False, 10, {},
        has_session_geometry=True,
    )
    for block, name in ((geometry, "geometry"), (trajectory, "trajectory")):
        assert block["available"] is False
        assert "no build has run" not in block["unavailable_reason"], (
            f"the {name} block says no build ran over a session that has one: "
            + block["unavailable_reason"]
        )

    # And with nothing on disk, both still say the plain thing.
    empty_geometry = _geometry_block(None, False, 10, has_session_geometry=False)
    empty_trajectory = WorldBuilderStatusProducer._trajectory_block(
        None, None, None, None, None, False, 10, {},
        has_session_geometry=False,
    )
    assert "no build has run" in empty_geometry["unavailable_reason"]
    assert "no build has run" in empty_trajectory["unavailable_reason"]


def test_an_earlier_walk_can_actually_be_opened(tmp_path):
    """The status channel saying `ready` is worth nothing if the route 404s.

    `read_derived` gates on `derived_is_current`, which read the WORLD's
    manifest -- the one naming whichever session built last. So on a world
    walked twice it judged the older session's keyframes against the newer
    session's digest, found a mismatch, logged "stale; treating as absent"
    and refused to serve it. "Open an earlier walk from Saved Worlds"
    returned a 404 for a reconstruction sitting on disk.

    Four review rounds fixed that bug on the REPORTING path, one branch at
    a time. Nobody asked what the phone does after the status channel says
    `ready` -- which is fetch the geometry, and get nothing.
    """
    from tower.world_builder.engine import WorldBuilderEngine
    from tower.world_builder.records import Keyframe
    from tower.world_builder.store import WorldStore

    store = WorldStore(tmp_path / "worlds")
    engine = WorldBuilderEngine(store)
    world_id = engine.create_world("twice")

    built = []
    for index in range(2):
        session_id = engine.start_session(world_id, frame_source="synthetic")
        # A keyframe each, so the two sessions have DIFFERENT digests. With
        # none, both digest to the same value and the world manifest happens
        # to match both -- which makes the bug invisible and the test
        # vacuous. Found by reverting the fix and watching it still pass.
        store.append_keyframe(world_id, Keyframe(
            keyframe_id=f"{session_id}:1", session_id=session_id,
            source_seq=index, received_at=float(index),
            image_relpath=f"images/{index}.jpg", width=8, height=8,
            byte_count=10 + index,
        ))
        digest = _session_digest(store, world_id, session_id)
        store.write_derived(
            world_id, session_id,
            poses=[{"keyframe_id": f"{session_id}:1", "position": [0.0, 0.0, 0.0]}],
            points=[[float(index), 0.0, 0.0]],
            manifest={
                "schema_version": store.read_world(world_id).schema_version,
                "session_id": session_id, "keyframes": 1, "points": 1,
                "poses_solved": 1, "poses_refused": 0, "segments": 1,
                "input_digest": digest, "built_at": 0.0, "backend_id": "test",
            },
        )
        engine.stop_session("stop")
        built.append(session_id)
    older, newer = built

    for label, session_id, expected_x in (("older", older, 0.0), ("newer", newer, 1.0)):
        derived = store.read_derived(world_id, session_id)
        assert derived is not None, (
            f"the {label} walk's geometry was refused as stale; the phone "
            "gets a 404 after being told the world is ready"
        )
        assert derived["points"][0][0] == expected_x, (
            f"the {label} walk was served another session's points"
        )


def _session_digest(store, world_id, session_id):
    from tower.world_builder.store import compute_input_digest

    return compute_input_digest(store.read_keyframes(world_id, session_id))


def test_currency_has_three_answers_not_two(tmp_path):
    """"Unknown" is not "no", and folding it into one boolean broke both.

    Two questions were being asked of `derived_is_current`: *is this
    geometry current* (the wire contract's `current` flag, "reflects every
    keyframe accepted so far") and *may this be served at all*
    (`read_derived`'s verify gate). For a legacy world walked twice the
    honest answers differ -- unknown, and yes-with-the-flag-off.

    Returning True from one boolean made the ROUTE assert `current: true`
    over geometry a reviewer then made genuinely stale by appending a
    keyframe. Returning False refuses a good reconstruction outright.
    """
    from tower.results.world_builder_geometry import _is_current
    from tower.world_builder.engine import WorldBuilderEngine
    from tower.world_builder.records import Keyframe
    from tower.world_builder.store import WorldStore

    store = WorldStore(tmp_path / "worlds")
    engine = WorldBuilderEngine(store)
    world_id = engine.create_world("legacy")

    built = []
    for index in range(2):
        session_id = engine.start_session(world_id, frame_source="synthetic")
        store.append_keyframe(world_id, Keyframe(
            keyframe_id=f"{session_id}:1", session_id=session_id,
            source_seq=index, received_at=float(index),
            image_relpath=f"images/{index}.jpg", width=8, height=8,
            byte_count=10 + index,
        ))
        store.write_derived(
            world_id, session_id,
            poses=[{"keyframe_id": f"{session_id}:1", "position": [0.0, 0.0, 0.0]}],
            points=[[float(index), 0.0, 0.0]],
            manifest={
                "schema_version": store.read_world(world_id).schema_version,
                "session_id": session_id, "keyframes": 1, "points": 1,
                "poses_solved": 1, "poses_refused": 0, "segments": 1,
                "input_digest": _session_digest(store, world_id, session_id),
                "built_at": 0.0, "backend_id": "test",
            },
        )
        engine.stop_session("stop")
        built.append(session_id)
    older = built[0]

    # A LEGACY world: delete the per-session copies, as anything built
    # before they existed looks on disk.
    for session_id in built:
        store.session_manifest_path(world_id, session_id).unlink()

    digest = _session_digest(store, world_id, older)
    assert store.derived_currency(world_id, digest, older) is None, (
        "a world whose only manifest is about another session was judged, "
        "when the honest answer is that nothing here can judge it"
    )
    assert store.read_derived(world_id, older) is not None, (
        "the reconstruction was refused as stale; the phone gets a 404"
    )
    assert _is_current(store, world_id, older) is False, (
        "the route claimed the geometry is current when nothing had "
        "checked; `current` is a claim, and the status channel says false"
    )

    # And it stays false once the geometry is genuinely behind.
    store.append_keyframe(world_id, Keyframe(
        keyframe_id=f"{older}:2", session_id=older, source_seq=99,
        received_at=99.0, image_relpath="images/99.jpg", width=8, height=8,
        byte_count=99,
    ))
    assert _is_current(store, world_id, older) is False


def test_an_older_walks_placements_are_not_judged_by_another_walks_build(tmp_path):
    """Every placement of an earlier walk was refused, so it could not be
    composited at all -- and its segments were served the OTHER session's
    coverage classes.

    `usable_placements` and the `global_solve.segments` lookup still read
    the WORLD's manifest after `derived_is_current` had been given a
    session id. Found by a reviewer asking what else read the world-level
    copy.
    """
    from tower.results.world_builder_geometry import (
        _session_manifest,
        manifest_for,
        usable_placements,
    )
    from tower.world_builder.engine import WorldBuilderEngine
    from tower.world_builder.records import SegmentPlacement
    from tower.world_builder.store import WorldStore

    store = WorldStore(tmp_path / "worlds")
    engine = WorldBuilderEngine(store)
    world_id = engine.create_world("twice")

    built = []
    for index, coverage in enumerate(("unresolved", "confident")):
        session_id = engine.start_session(world_id, frame_source="synthetic")
        store.write_derived(
            world_id, session_id,
            poses=[], points=[],
            manifest={
                "schema_version": store.read_world(world_id).schema_version,
                "session_id": session_id, "keyframes": 1, "points": 0,
                "poses_solved": 0, "poses_refused": 0, "segments": 1,
                "input_digest": f"digest-{index}", "built_at": 0.0,
                "backend_id": "test",
                "global_solve": {"segments": {"0": {"coverage": coverage}}},
            },
        )
        # A registered placement bound to THIS session's build.
        store.write_placements(world_id, session_id, [SegmentPlacement(
            segment_index=0, state="registered",
            rotation_wxyz=(1.0, 0.0, 0.0, 0.0), translation=(0.0, 0.0, 0.0),
            scale=1.0, reference_segment=0, refusal_reason=None,
            evidence={}, input_digest=f"digest-{index}",
        )])
        engine.stop_session("stop")
        built.append(session_id)
    older, newer = built

    for label, session_id in (("older", older), ("newer", newer)):
        usable = usable_placements(store, world_id, session_id)
        assert 0 in usable, (
            f"the {label} walk's placement was judged against another "
            "walk's build and served as unplaced; the walk cannot be "
            "composited at all"
        )

    assert _session_manifest(store, world_id, older)["session_id"] == older
    older_coverage = manifest_for(store, world_id, older)["global_solve"]["segments"]["0"]
    newer_coverage = manifest_for(store, world_id, newer)["global_solve"]["segments"]["0"]
    assert older_coverage["coverage"] == "unresolved", (
        "the older walk was served the newer walk's coverage class -- the "
        "confident one"
    )
    assert newer_coverage["coverage"] == "confident"


def test_a_world_manifest_that_is_not_a_dict_is_refused_not_a_500():
    """The identically corrupt per-session copy gave a clean refusal.

    `read_session_manifest` guards its return type; `read_derived_manifest`
    does not, and `_validate_manifest` -- extracted to guarantee both copies
    are held to the same checks -- called `.get()` on whatever it was
    handed. A reviewer fed it a top-level list and a string and got
    AttributeError out of the status channel.
    """
    from tower.results.world_builder import _validate_manifest

    for shape in ([], "a string", 42, ()):
        assert _validate_manifest(shape, "w" * 32) is None, shape


def test_a_manifest_without_its_geometry_is_not_an_unbuilt_session():
    """A manifest proves a build ran.

    `_lifecycle` asked `has_session_geometry` before `has_manifest`, so a
    session whose manifest is there and whose derived tree is gone reported
    "no geometry has been built for this session yet" -- projected to the
    phone as a permanent `finalizing`, which the iOS note added in this
    campaign now renders as "worth waiting for Saved", forever.

    Reached through the ORDINARY case: `finalization is None` is what every
    offline caller leaves, so the branch that already handled this never
    fired for them.
    """
    from tower.results.world_builder import (
        _MODEL_STATE_BY_LIFECYCLE,
        MODEL_STATE_IDLE,
        _lifecycle,
    )
    from tower.world_builder.records import Session

    session = Session(
        session_id="s" * 32, world_id="w" * 32, started_at=0.0,
        frame_source="synthetic", ended_at=1.0, end_reason="stop",
        finalization=None,
    )
    lifecycle = _lifecycle(
        holder=None, stopped=True, session=session,
        geometry_current=False, has_manifest=True, has_session_geometry=False,
    )
    assert lifecycle["state"] == "interrupted", lifecycle
    assert "yet" not in (lifecycle["reason"] or "")
    assert "rebuilt" in (lifecycle["reason"] or "")
    projected = _MODEL_STATE_BY_LIFECYCLE.get(lifecycle["state"], MODEL_STATE_IDLE)
    assert projected != "finalizing"


def test_the_file_cache_evicts_rather_than_collapsing(tmp_path):
    """A bound that clears everything is a cliff, not a bound.

    Measured by a reviewer: the hit rate fell from 95.8% to 14.0% the
    moment the entry count crossed MAX_ENTRIES, because crossing it threw
    away every entry. This producer is a process-lifetime singleton on a
    host holding 163 worlds.
    """
    from tower.results.world_builder import _FileCache

    cache = _FileCache()
    paths = []
    for index in range(_FileCache.MAX_ENTRIES + 5):
        path = tmp_path / f"{index}.json"
        path.write_text("{}", encoding="utf-8")
        paths.append(path)
        cache.read(path, lambda index=index: index)

    assert len(cache._entries) <= _FileCache.MAX_ENTRIES

    reads = []
    for path in paths[-10:]:
        cache.read(path, lambda p=path: reads.append(p))
    assert reads == [], (
        f"{len(reads)} of the 10 most recent entries were evicted; the "
        "cache collapsed instead of evicting the oldest"
    )

    # AND IT IS LEAST-RECENTLY-*USED*, NOT FIRST-IN-FIRST-OUT.
    #
    # `d[existing] = v` does not reorder a dict, so a plain re-insert on a
    # hit leaves the hottest entry -- the live session's journal, refreshed
    # every poll -- in its original slot, to be evicted first. A reviewer
    # demonstrated that on a three-key dict. The first version of this test
    # only checked that the newest survived, which FIFO satisfies too.
    hot = paths[-1]
    cache.read(hot, lambda: pytest.fail("the hot entry was not cached"))
    for index in range(_FileCache.MAX_ENTRIES):
        path = tmp_path / f"flood-{index}.json"
        path.write_text("{}", encoding="utf-8")
        cache.read(path, lambda: index)
        cache.read(hot, lambda: reads.append(hot))
    assert reads == [], (
        "the entry read on every single poll was evicted while colder ones "
        "survived: the cache is first-in-first-out, not least-recently-used"
    )


def test_a_manifest_is_not_geometry(tmp_path):
    """One payload cannot say the geometry is gone and count 26,634 points.

    `has_session_geometry` was threaded into the `manifest is None` arm of
    both blocks and not the other one -- so in the state the lifecycle
    branch beside them was written for (a build ran, its tree is gone) the
    blocks reported the manifest's figures as live, `current: true`, while
    the route answered 404. A reviewer printed both halves of that payload
    side by side at field scale.

    It is the "Nothing mapped yet over 26,634 points" failure this
    campaign is named for, inverted.
    """
    from tower.results.world_builder import (
        WorldBuilderStatusProducer,
        _geometry_block,
    )

    manifest = {
        "schema_version": 1, "session_id": "s" * 32, "keyframes": 795,
        "points": 26634, "poses_solved": 561, "poses_refused": 12,
        "segments": 122, "input_digest": "d", "built_at": 0.0,
        "backend_id": "test",
    }

    gone = _geometry_block(manifest, True, 795, has_session_geometry=False)
    there = _geometry_block(manifest, True, 795, has_session_geometry=True)

    assert there["available"] is True and there["element_count"] == 26634
    assert gone["available"] is False, (
        "a manifest was reported as geometry with no poses or points on "
        f"disk: {gone}"
    )
    assert gone["element_count"] is None
    assert "rebuilt" in gone["unavailable_reason"]

    trajectory_gone = WorldBuilderStatusProducer._trajectory_block(
        None, None, None, None, manifest, True, 795, {},
        has_session_geometry=False,
    )
    assert trajectory_gone["available"] is False, trajectory_gone
    assert trajectory_gone["pose_count"] is None


def test_every_reader_prefers_the_copy_beside_the_geometry(tmp_path):
    """Four readers, one rule, or both of this campaign's failures come back.

    The status producer read the WORLD's manifest first and fell back to
    the session's; `store.derived_currency` and
    `world_builder_geometry._session_manifest` do the opposite. A reviewer
    built the states where the two copies disagree and watched the readers
    pick different manifests -- producing `ready` beside a 404 in one
    direction and a route serving geometry the phone was told was still
    finalizing in the other.
    """
    from tower.results.world_builder import WorldBuilderStatusProducer
    from tower.results.world_builder_geometry import _is_current, _session_manifest
    from tower.world_builder.engine import WorldBuilderEngine
    from tower.world_builder.records import Keyframe
    from tower.world_builder.store import WorldStore

    store = WorldStore(tmp_path / "worlds")
    engine = WorldBuilderEngine(store)
    world_id = engine.create_world("torn")
    session_id = engine.start_session(world_id, frame_source="synthetic")
    store.append_keyframe(world_id, Keyframe(
        keyframe_id=f"{session_id}:1", session_id=session_id, source_seq=0,
        received_at=0.0, image_relpath="images/0.jpg", width=8, height=8,
        byte_count=10,
    ))
    digest = _session_digest(store, world_id, session_id)
    store.write_derived(
        world_id, session_id,
        poses=[{"keyframe_id": f"{session_id}:1", "position": [0.0, 0.0, 0.0]}],
        points=[[0.0, 0.0, 0.0]],
        manifest={
            "schema_version": store.read_world(world_id).schema_version,
            "session_id": session_id, "keyframes": 1, "points": 1,
            "poses_solved": 1, "poses_refused": 0, "segments": 1,
            "input_digest": digest, "built_at": 0.0, "backend_id": "test",
        },
    )
    engine.stop_session("stop")

    # A torn write: the world's copy lagged, the session's matches the
    # geometry that is actually on disk.
    import json as _json

    world_path = store.derived_manifest_path(world_id)
    stale = _json.loads(world_path.read_text(encoding="utf-8"))
    stale["input_digest"] = "0" * 64
    world_path.write_text(_json.dumps(stale), encoding="utf-8")

    import time

    payload = WorldBuilderStatusProducer(
        tmp_path / "worlds", time.time
    ).snapshot(world_id, session_id).payload

    assert _session_manifest(store, world_id, session_id)["input_digest"] == digest
    assert store.derived_currency(world_id, digest, session_id) is True
    assert _is_current(store, world_id, session_id) is True
    assert payload["geometry"]["current"] is True, (
        "the status channel judged by the stale copy while the route "
        "judged by the fresh one: " + str(payload["geometry"])
    )
    assert payload["lifecycle"]["state"] == "ready"
