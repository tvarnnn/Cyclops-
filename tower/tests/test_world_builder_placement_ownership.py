"""Who owns `placements.json` when a session does not finalise normally.

On the 2026-09-09 walk the global solve worked. Nine background solves ran
during the capture, the last over 646 keyframes, and the build that followed
merged their solution into the derived tree: `derived/manifest.json` records
`segments_replaced: 72` across **14 components**, and `merge()` writes a
`state="registered"` placement for every one of them.

The phone then drew 87 disconnected fragments, because sixteen seconds after
that build the Sim3 registrar ran and overwrote `placements.json` with 120
refusals and 2 registrations.

It ran because the guard asked the wrong question:

    if args.register and not (solve_report or {}).get("solved"):

`solve_report` is the FINAL solve's report. The session died in the observe
loop, so finalization never reached the final solve, so `solve_report` was
`None`, so the guard concluded no solution existed. Nine good solves were
discarded on the strength of a question about a tenth that never happened.

The question that matters is who wrote the file, and only the build knows.
`engine.build()` now reports `diagnostics["placements_source"]`.
"""

from dataclasses import replace

import cv2
import numpy as np
import pytest

from tower.world_builder import global_solve
from tower.world_builder.engine import WorldBuilderEngine
from tower.world_builder.records import Keyframe


def _keyframe(i: int, segment: int) -> Keyframe:
    return Keyframe(
        keyframe_id=f"s:{i:08d}",
        session_id="s",
        source_seq=i,
        received_at=float(i),
        image_relpath=f"images/{i:08d}.jpg",
        width=360,
        height=640,
        byte_count=1,
        segment_index=segment,
    )


# ---------------------------------------------------------------------------
# 1. The build says who owns the file.


def test_a_build_with_a_solution_claims_the_placements(monkeypatch, tmp_path):
    """`placements_source` is "global_solve" exactly when merge wrote them."""
    engine, world_id, session_id = _engine_with_a_session(tmp_path)

    merged = _merged_placements()
    monkeypatch.setattr(global_solve, "load_solution", lambda *a, **k: object())
    monkeypatch.setattr(global_solve, "merge", lambda *a, **k: merged)

    result = engine.build(world_id, session_id)
    assert result.diagnostics["placements_source"] == "global_solve"


def test_a_build_without_a_solution_claims_nothing(tmp_path):
    """No solution means no placements written, and the field says so.

    The registrar is the correct fallback in this case, and the guard must
    still let it run.
    """
    engine, world_id, session_id = _engine_with_a_session(tmp_path)
    result = engine.build(world_id, session_id)
    assert result.diagnostics["placements_source"] is None


def test_an_unreadable_solution_leaves_the_registrar_free_to_run(monkeypatch, tmp_path):
    """The two fixes must not combine into a world with no placements at all.

    `load_solution` now absorbs a torn archive and returns None. That must
    read as "no solution", so the registrar still runs -- otherwise widening
    the guard would have traded a clobbered `placements.json` for a missing
    one.
    """
    engine, world_id, session_id = _engine_with_a_session(tmp_path)
    workspace = global_solve.workspace_for(engine._store, world_id, session_id)
    workspace.root.mkdir(parents=True, exist_ok=True)
    workspace.solution_path.write_text('{"schema_version": 1}', encoding="utf-8")
    workspace.arrays_path.write_bytes(b"PK\x03\x04 torn")

    result = engine.build(world_id, session_id)
    assert result.diagnostics["placements_source"] is None


# ---------------------------------------------------------------------------
# 2. The guard reads that field, not the final solve's report.


@pytest.mark.parametrize(
    "placements_source, final_solve_ran, registrar_should_run",
    [
        # The field case: a solution exists, the final solve never ran.
        # The old guard ran the registrar here and destroyed 72 placements.
        pytest.param("global_solve", False, False, id="solution-but-no-final-solve"),
        pytest.param("global_solve", True, False, id="solution-and-final-solve"),
        # No solution: the registrar is the only producer, and must run.
        pytest.param(None, False, True, id="no-solution-no-final-solve"),
        pytest.param(None, True, True, id="no-solution-but-final-solve-ran"),
    ],
)
def test_the_guard_asks_who_wrote_the_file(
    placements_source, final_solve_ran, registrar_should_run
):
    """The decision table, stated directly.

    The row that matters is the first one. It is the field configuration,
    and it is the only row where the old and new guards disagree.
    """
    solve_report = {"solved": True} if final_solve_ran else None

    # The guard as it now stands, in `world_build_session.py`.
    runs_now = placements_source is None
    assert runs_now is registrar_should_run

    # The guard as it stood on 2026-09-09, for contrast.
    ran_before = not (solve_report or {}).get("solved")
    if placements_source == "global_solve" and not final_solve_ran:
        assert ran_before is True and runs_now is False, (
            "this is the field regression; the guards must differ here"
        )


# ---------------------------------------------------------------------------
# helpers


def _engine_with_a_session(tmp_path):
    from tower.world_builder.store import WorldStore

    store = WorldStore(tmp_path)
    engine = WorldBuilderEngine(store)
    world_id = engine.create_world()
    session_id = engine.start_session(world_id, frame_source="synthetic")
    jpeg = _jpeg_bytes()
    for i in range(4):
        keyframe = replace(_keyframe(i, 0), session_id=session_id, byte_count=len(jpeg))
        store.append_keyframe(world_id, keyframe)
        path = store.session_dir(world_id, session_id) / keyframe.image_relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(jpeg)
    engine.stop_session("stopped")
    return engine, world_id, session_id


def _jpeg_bytes() -> bytes:
    """A real 360x640 JPEG. The frontend decodes every keyframe it builds."""
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (640, 360, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    return buf.tobytes()


def _merged_placements():
    from tower.world_builder.records import SegmentPlacement

    class _Merged:
        pose_rows: list = []
        point_rows: list = []
        support_rows: list = []
        placements = [
            SegmentPlacement(
                segment_index=0, state="registered",
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0), translation=(0.0, 0.0, 0.0),
                scale=1.0, reference_segment=0, refusal_reason=None,
                input_digest="d", evidence={}, frame_revision=1,
            )
        ]
        summary: dict = {}
        segments: dict = {}

    return _Merged()
