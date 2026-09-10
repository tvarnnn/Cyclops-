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
        geometry_current=True, has_manifest=True,
    )
    assert before["state"] == "interrupted"

    # A repair that solved nothing must NOT flip the label -- only a
    # finalization that actually finished, with a final solve that landed.
    _run(root, world_id, "--skip-solve")
    skipped = _lifecycle(
        holder=None, stopped=True, session=store.read_session(world_id, session_id),
        geometry_current=True, has_manifest=True,
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
        geometry_current=True, has_manifest=True,
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
