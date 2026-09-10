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
from types import SimpleNamespace

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


# The PRODUCTION guard, driven directly.
#
# The first version of this file re-typed the condition into the test and
# asserted the copy -- `runs_now = placements_source is None` -- which would
# have passed against any implementation, including the broken one. An
# adversarial review caught it. `should_register` now exists as a function so
# there is something real to call.


@pytest.mark.parametrize(
    "diagnostics, final_solve_ran, expected",
    [
        # THE FIELD ROW. A solution placed segments; the final solve never
        # ran because the session died in the observe loop. The old guard
        # asked about the final solve, ran the registrar, and destroyed 72
        # placements.
        pytest.param({"placements_source": "global_solve"}, False, False,
                     id="placed-but-no-final-solve"),
        pytest.param({"placements_source": "global_solve"}, True, False,
                     id="placed-and-final-solve"),
        # No solution, or one that placed nothing: the registrar is the only
        # producer there is and must run.
        pytest.param({"placements_source": None}, False, True,
                     id="no-solution"),
        pytest.param({"placements_source": None}, True, True,
                     id="no-solution-but-final-solve-ran"),
        pytest.param({}, False, True, id="a-build-that-reported-nothing"),
    ],
)
def test_the_guard_asks_who_placed_the_segments(diagnostics, final_solve_ran, expected):
    from scripts.world_build_session import should_register

    result = SimpleNamespace(diagnostics=diagnostics)
    assert should_register(result) is expected

    # And the guard it replaced, for contrast: it asked the final solve's
    # report, which is None whenever finalization did not complete normally.
    solve_report = {"solved": True} if final_solve_ran else None
    ran_before = not (solve_report or {}).get("solved")
    if diagnostics.get("placements_source") == "global_solve" and not final_solve_ran:
        assert ran_before is True and should_register(result) is False, (
            "this is the field regression; the two guards must differ here"
        )


def test_the_guard_survives_a_build_with_no_diagnostics_at_all():
    """`result.diagnostics` is a default_factory dict, but the guard reads it
    defensively and must not crash if that ever changes."""
    from scripts.world_build_session import should_register

    assert should_register(SimpleNamespace(diagnostics=None)) is True
    assert should_register(SimpleNamespace()) is True


# ---------------------------------------------------------------------------
# 3. "Placed something", not "merge ran".


def test_a_merge_that_placed_nothing_leaves_the_registrar_free(monkeypatch, tmp_path):
    """`merge()` returning only refusals must NOT stand the registrar down.

    An adversarial review found the first version asking `placements is not
    None`, which is true whenever merge ran at all -- including when every
    segment is still pending (it returns []) and when the solve posed none of
    their keyframes (it returns nothing but refusals). Both place zero
    segments. Suppressing the registrar there is a new way to finish a walk
    with no placements at all.
    """
    engine, world_id, session_id = _engine_with_a_session(tmp_path)
    monkeypatch.setattr(global_solve, "load_solution", lambda *a, **k: object())
    monkeypatch.setattr(global_solve, "merge", lambda *a, **k: _merged_placements(
        states=("refused", "refused")
    ))
    result = engine.build(world_id, session_id)
    assert result.diagnostics["placements_source"] is None


def test_a_merge_that_placed_nothing_at_all_leaves_the_registrar_free(
    monkeypatch, tmp_path
):
    engine, world_id, session_id = _engine_with_a_session(tmp_path)
    monkeypatch.setattr(global_solve, "load_solution", lambda *a, **k: object())
    monkeypatch.setattr(global_solve, "merge", lambda *a, **k: _merged_placements(states=()))
    result = engine.build(world_id, session_id)
    assert result.diagnostics["placements_source"] is None


def test_one_registered_placement_is_enough_to_claim_the_file(monkeypatch, tmp_path):
    engine, world_id, session_id = _engine_with_a_session(tmp_path)
    monkeypatch.setattr(global_solve, "load_solution", lambda *a, **k: object())
    monkeypatch.setattr(global_solve, "merge", lambda *a, **k: _merged_placements(
        states=("refused", "registered", "refused")
    ))
    result = engine.build(world_id, session_id)
    assert result.diagnostics["placements_source"] == "global_solve"


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


def _merged_placements(states=("registered",)):
    """A stand-in for `merge()`'s result, with the placement states it chose.

    Parameterised on state because "did merge run" and "did merge place
    anything" are different questions and conflating them was a real defect.
    """
    from tower.world_builder.records import SegmentPlacement

    def placement(index: int, state: str) -> SegmentPlacement:
        if state == "registered":
            return SegmentPlacement(
                segment_index=index, state="registered",
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0), translation=(0.0, 0.0, 0.0),
                scale=1.0, reference_segment=index, refusal_reason=None,
                input_digest="d", evidence={}, frame_revision=1,
            )
        return SegmentPlacement(
            segment_index=index, state="refused",
            rotation_wxyz=None, translation=None, scale=None,
            reference_segment=None,
            refusal_reason="the global solve posed none of this segment's keyframes",
            input_digest="d", evidence={}, frame_revision=1,
        )

    class _Merged:
        pose_rows: list = []
        point_rows: list = []
        support_rows: list = []
        placements = [placement(i, state) for i, state in enumerate(states)]
        summary: dict = {}
        segments: dict = {}

    return _Merged()


# ---------------------------------------------------------------------------
# 4. The fallback must not be destructive.


def test_the_registrar_will_not_replace_current_registered_placements(tmp_path):
    """The second line of defence, for when the first is told the wrong thing.

    `load_solution` absorbs any unreadable solution and returns None. That is
    right for a torn archive and it is also what it returns if numpy changes
    an exception type, a schema drifts, or the box runs out of memory. In all
    of those `engine.build()` writes no placements, reports
    `placements_source: None`, and the guard concludes there is no global
    solve to defer to -- so it runs the registrar, which used to write
    unconditionally and destroy exactly the placements the fix was written to
    protect. From a cause whose only symptom is one `logger.warning`.

    A placement set is trusted here only if it is REGISTERED and CURRENT: the
    serving layer already drops placements whose digest disagrees with the
    manifest, so a stale set is not worth preserving and re-registering it is
    the correct outcome.
    """
    from scripts.world_build_session import register_session
    from tower.world_builder.records import SegmentPlacement

    engine, world_id, session_id = _engine_with_a_session(tmp_path)
    store = engine._store
    engine.build(world_id, session_id)
    digest = (store.read_derived_manifest(world_id) or {}).get("input_digest")

    good = [SegmentPlacement(
        segment_index=0, state="registered",
        rotation_wxyz=(1.0, 0.0, 0.0, 0.0), translation=(0.0, 0.0, 0.0),
        scale=1.0, reference_segment=0, refusal_reason=None,
        input_digest=digest, evidence={"points": 431}, frame_revision=1,
    )]
    store.write_placements(world_id, session_id, good)

    outcome = register_session(store, world_id, session_id)

    assert outcome["attempted"] is False
    assert outcome["wrote_placements"] is False
    after = store.read_placements(world_id, session_id)
    assert [p.state for p in after] == ["registered"]
    assert after[0].evidence == {"points": 431}, "the good placements were replaced"


def test_the_registrar_still_runs_when_the_placements_are_stale(tmp_path):
    """Stale placements are not something to protect.

    The serving layer drops any placement whose `input_digest` disagrees with
    the manifest, so preserving them would leave the world with nothing drawn
    AND nothing to draw it from.
    """
    from scripts.world_build_session import register_session
    from tower.world_builder.records import SegmentPlacement

    engine, world_id, session_id = _engine_with_a_session(tmp_path)
    store = engine._store
    engine.build(world_id, session_id)

    stale = [SegmentPlacement(
        segment_index=0, state="registered",
        rotation_wxyz=(1.0, 0.0, 0.0, 0.0), translation=(0.0, 0.0, 0.0),
        scale=1.0, reference_segment=0, refusal_reason=None,
        input_digest="a-digest-from-an-older-build", evidence={}, frame_revision=1,
    )]
    store.write_placements(world_id, session_id, stale)

    outcome = register_session(store, world_id, session_id)
    assert outcome["attempted"] is True


def test_the_registrar_runs_on_a_world_with_no_placements(tmp_path):
    from scripts.world_build_session import register_session

    engine, world_id, session_id = _engine_with_a_session(tmp_path)
    engine.build(world_id, session_id)
    outcome = register_session(engine._store, world_id, session_id)
    assert outcome["attempted"] is True
