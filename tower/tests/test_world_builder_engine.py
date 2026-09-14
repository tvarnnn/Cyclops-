"""End-to-end engine: create, map, stop, persist, restart, reload, inspect.

SYNTHETIC, NOT PHYSICAL.
"""

import json

import numpy as np

import pytest

from tests import synthetic_scene as ss
from tower.object_memory.records import Confidence
from tower.world_builder.backends import BACKEND_CLASSICAL, BACKEND_UNPOSED
from tower.world_builder.engine import (
    ImagesPurgedError,
    SessionNotActiveError,
    WorldBuilderEngine,
)
from tower.world_builder.records import CameraIntrinsics
from tower.world_builder.schema import (
    END_REASON_STOP,
    INTRINSICS_SOURCE_SELF_CALIBRATED,
    POSE_STATUS_ANCHOR,
    POSE_STATUS_SOLVED,
    SCALE_RELATIVE,
    SCALE_UNKNOWN,
)
from tower.world_builder.store import WorldStore

WIDTH, HEIGHT = 480, 360


@pytest.fixture(scope="module")
def scene():
    return ss.furnished_room()


@pytest.fixture(scope="module")
def camera_matrix():
    return ss.camera_matrix(WIDTH, HEIGHT)


@pytest.fixture(scope="module")
def intrinsics(camera_matrix):
    return CameraIntrinsics(
        source=INTRINSICS_SOURCE_SELF_CALIBRATED,
        model="pinhole",
        fx=float(camera_matrix[0, 0]),
        fy=float(camera_matrix[1, 1]),
        cx=float(camera_matrix[0, 2]),
        cy=float(camera_matrix[1, 2]),
        calibrated_width=WIDTH,
        calibrated_height=HEIGHT,
    )


@pytest.fixture(scope="module")
def walk_jpegs(scene, camera_matrix):
    """A short sideways walk, encoded exactly as the wire would deliver."""
    poses = ss.strafe(14, step=0.09)
    images = ss.render_sequence(scene, poses, camera_matrix, WIDTH, HEIGHT)
    return [ss.encode_jpeg(image) for image in images]


def _map_session(
    store, walk_jpegs, intrinsics, backend=BACKEND_CLASSICAL, redactor_factory=None
):
    engine = WorldBuilderEngine(
        store, backend_name=backend, redactor_factory=redactor_factory
    )
    world_id = engine.create_world("Test Room")
    session_id = engine.start_session(
        world_id,
        intrinsics=intrinsics,
        frame_source="synthetic",
        declared_size=(WIDTH, HEIGHT),
    )
    for index, payload in enumerate(walk_jpegs):
        engine.observe(payload, source_seq=index * 7, wire_seq=index)
    summary = engine.stop_session(END_REASON_STOP)
    return engine, world_id, session_id, summary


class TestARestartedSequence:
    def test_a_restarted_sequence_overwrites_no_keyframe(
        self, tmp_path, walk_jpegs, intrinsics
    ):
        """The glasses restarted mid-session; every keyframe survives.

        `source_seq` resets when the glasses session restarts, and a
        builder now follows a capture LINEAGE, so one session can see the
        numbers start again. The keyframe id and the image file name were
        both `source_seq`: a repeated number overwrote the earlier image
        on disk, the journal held two keyframes with one id, and the final
        solve aborted on a duplicate file name -- "Partial" over a full
        walk, four times in a dress rehearsal.
        """
        store = WorldStore(tmp_path)
        engine = WorldBuilderEngine(store, backend_name=BACKEND_CLASSICAL)
        world_id = engine.create_world("Test Room")
        session_id = engine.start_session(
            world_id, intrinsics=intrinsics, frame_source="synthetic",
            declared_size=(WIDTH, HEIGHT),
        )
        half = len(walk_jpegs) // 2
        for index, payload in enumerate(walk_jpegs[:half]):
            engine.observe(payload, source_seq=index * 7, wire_seq=index)
        accepted_before = engine._session.keyframes_accepted
        # The sender restarts: the same numbers again.
        for index, payload in enumerate(walk_jpegs[half:]):
            engine.observe(payload, source_seq=index * 7, wire_seq=half + index)
        summary = engine.stop_session(END_REASON_STOP)

        keyframes = store.read_keyframes(world_id, session_id)
        ids = [k.keyframe_id for k in keyframes]
        assert len(ids) == len(set(ids)), "two keyframes share an id"
        names = [k.image_relpath for k in keyframes]
        assert len(names) == len(set(names)), "two keyframes share an image file"
        images = list(store.images_dir(world_id, session_id).glob("*.jpg"))
        assert len(images) == len(keyframes), (
            f"{len(keyframes)} keyframes but {len(images)} images on disk"
        )
        assert summary.keyframes_accepted > accepted_before, (
            "nothing after the restart was kept"
        )
        seqs = [k.source_seq for k in keyframes]
        assert seqs == sorted(seqs) and len(seqs) == len(set(seqs))


class TestSessionLifecycle:
    def test_a_full_session_produces_keyframes_and_a_clean_stop(
        self, tmp_path, walk_jpegs, intrinsics
    ):
        store = WorldStore(tmp_path)

        _, world_id, session_id, summary = _map_session(
            store, walk_jpegs, intrinsics
        )

        assert summary.frames_observed == len(walk_jpegs)
        assert 1 < summary.keyframes_accepted < len(walk_jpegs)
        assert summary.end_reason == END_REASON_STOP
        session = store.read_session(world_id, session_id)
        assert session.ended_at is not None

    def test_rejections_are_counted_by_reason(
        self, tmp_path, walk_jpegs, intrinsics
    ):
        """The histogram is the only tuning instrument before real footage."""
        store = WorldStore(tmp_path)

        _, _, _, summary = _map_session(store, walk_jpegs, intrinsics)

        assert summary.rejected_by_reason
        assert sum(summary.rejected_by_reason.values()) + summary.keyframes_accepted == (
            summary.frames_observed
        )

    def test_observe_without_a_session_is_refused(self, tmp_path):
        engine = WorldBuilderEngine(WorldStore(tmp_path))

        with pytest.raises(SessionNotActiveError):
            engine.observe(b"x", source_seq=0)

    def test_a_malformed_frame_does_not_end_the_session(
        self, tmp_path, walk_jpegs, intrinsics
    ):
        """One bad frame is frame-scoped, never session-scoped."""
        store = WorldStore(tmp_path)
        engine = WorldBuilderEngine(store)
        world_id = engine.create_world()
        engine.start_session(world_id, intrinsics=intrinsics)

        engine.observe(walk_jpegs[0], source_seq=0)
        bad = engine.observe(b"definitely not a jpeg", source_seq=1)
        good = engine.observe(walk_jpegs[5], source_seq=2)

        assert bad.reason == "malformed_frame"
        assert good.frames_observed == 3
        engine.stop_session()

    def test_the_writer_lock_is_released_on_stop(
        self, tmp_path, walk_jpegs, intrinsics
    ):
        store = WorldStore(tmp_path)

        _, world_id, _, _ = _map_session(store, walk_jpegs, intrinsics)

        assert not store.lock_path(world_id).exists()

    def test_keyframe_images_are_written_next_to_their_journal(
        self, tmp_path, walk_jpegs, intrinsics
    ):
        store = WorldStore(tmp_path)

        _, world_id, session_id, _ = _map_session(store, walk_jpegs, intrinsics)

        keyframes = store.read_keyframes(world_id, session_id)
        session_dir = store.session_dir(world_id, session_id)
        for keyframe in keyframes:
            assert (session_dir / keyframe.image_relpath).exists()

    def test_events_are_journalled_with_dense_ids(
        self, tmp_path, walk_jpegs, intrinsics
    ):
        """A gap in event_id means a real drop, not a skipped number."""
        store = WorldStore(tmp_path)

        _, world_id, session_id, _ = _map_session(store, walk_jpegs, intrinsics)

        events = store.read_events(world_id, session_id)
        assert [event["event_id"] for event in events] == list(range(len(events)))
        assert events[0]["kind"] == "session_started"
        assert events[-1]["kind"] == "session_stopped"

    def test_the_journal_is_the_incremental_update_stream(
        self, tmp_path, walk_jpegs, intrinsics
    ):
        """Updates are exposed as they happen, not only at stop."""
        store = WorldStore(tmp_path)
        engine = WorldBuilderEngine(store)
        world_id = engine.create_world()
        session_id = engine.start_session(world_id, intrinsics=intrinsics)

        for index, payload in enumerate(walk_jpegs[:6]):
            engine.observe(payload, source_seq=index * 7)
            # Readable mid-session, from the same process or another one.
            assert store.read_events(world_id, session_id)

        engine.stop_session()


class TestBuildAndReload:
    def test_build_produces_poses_and_points(
        self, tmp_path, walk_jpegs, intrinsics
    ):
        store = WorldStore(tmp_path)
        engine, world_id, session_id, _ = _map_session(
            store, walk_jpegs, intrinsics
        )

        result = engine.build(world_id, session_id)

        assert result.poses_solved >= 1
        assert result.points > 100
        assert result.backend_id == "classical-sfm"

    def test_a_solved_build_declares_relative_scale_never_metric(
        self, tmp_path, walk_jpegs, intrinsics
    ):
        """Internally consistent, arbitrary unit. Not a metre."""
        store = WorldStore(tmp_path)
        engine, world_id, session_id, _ = _map_session(
            store, walk_jpegs, intrinsics
        )

        result = engine.build(world_id, session_id)

        assert result.scale_state == SCALE_RELATIVE
        world = store.read_world(world_id)
        assert world.scale.meters_per_unit is None
        assert not world.scale.allows_metres

    def test_an_uncalibrated_build_yields_no_poses_and_unknown_scale(
        self, tmp_path, walk_jpegs
    ):
        """The honest real-Ray-Ban path today."""
        store = WorldStore(tmp_path)
        engine, world_id, session_id, _ = _map_session(
            store, walk_jpegs, CameraIntrinsics.unknown(), backend=BACKEND_UNPOSED
        )

        result = engine.build(world_id, session_id)

        assert result.poses_solved == 0
        assert result.points == 0
        assert result.scale_state == SCALE_UNKNOWN

    def test_a_downgrade_is_recorded_on_the_session(
        self, tmp_path, walk_jpegs
    ):
        store = WorldStore(tmp_path)
        engine, world_id, session_id, _ = _map_session(
            store, walk_jpegs, CameraIntrinsics.unknown(), backend=BACKEND_CLASSICAL
        )

        engine.build(world_id, session_id)

        session = store.read_session(world_id, session_id)
        assert session.backend_downgraded_from == BACKEND_CLASSICAL
        assert "intrinsics" in session.backend_downgrade_reason

    def test_edges_record_measurements_even_when_a_pose_is_refused(
        self, tmp_path, walk_jpegs
    ):
        """An empty map that explains itself beats one that does not."""
        store = WorldStore(tmp_path)
        engine, world_id, session_id, _ = _map_session(
            store, walk_jpegs, CameraIntrinsics.unknown(), backend=BACKEND_UNPOSED
        )

        engine.build(world_id, session_id)

        edges = store.read_edges(world_id, session_id)
        assert edges
        assert all(edge.pose_status != POSE_STATUS_SOLVED for edge in edges)
        assert any(edge.matches > 0 for edge in edges)

    def test_a_world_survives_process_restart_and_reloads(
        self, tmp_path, walk_jpegs, intrinsics
    ):
        """The definition-of-done item that everything else serves."""
        store = WorldStore(tmp_path)
        engine, world_id, session_id, _ = _map_session(
            store, walk_jpegs, intrinsics
        )
        engine.build(world_id, session_id)

        # Simulate a cold process: brand-new store over the same directory.
        reloaded = WorldStore(tmp_path)
        world = reloaded.read_world(world_id)
        keyframes = reloaded.read_keyframes(world_id, session_id)
        derived = reloaded.read_derived(world_id, session_id)

        assert world.world_id == world_id
        assert len(keyframes) > 1
        assert derived is not None
        assert len(derived["poses"]) == len(keyframes)
        assert len(derived["points"]) > 100

    def test_the_first_pose_is_the_anchor(
        self, tmp_path, walk_jpegs, intrinsics
    ):
        store = WorldStore(tmp_path)
        engine, world_id, session_id, _ = _map_session(
            store, walk_jpegs, intrinsics
        )
        engine.build(world_id, session_id)

        poses = store.read_derived(world_id, session_id)["poses"]

        assert poses[0]["status"] == POSE_STATUS_ANCHOR
        assert poses[0]["translation"] == [0.0, 0.0, 0.0]

    def test_deleting_derived_leaves_the_world_valid_and_rebuildable(
        self, tmp_path, walk_jpegs, intrinsics
    ):
        """The authoritative/derived split, enforced rather than asserted."""
        store = WorldStore(tmp_path)
        engine, world_id, session_id, _ = _map_session(
            store, walk_jpegs, intrinsics
        )
        first = engine.build(world_id, session_id)

        store.clear_derived(world_id)
        assert store.read_derived(world_id, session_id) is None

        second = engine.build(world_id, session_id)

        assert store.read_world(world_id).world_id == world_id
        assert second.keyframes == first.keyframes
        assert second.poses_solved == first.poses_solved

    def test_the_manifest_records_the_input_digest(
        self, tmp_path, walk_jpegs, intrinsics
    ):
        """Staleness is detected rather than trusted."""
        store = WorldStore(tmp_path)
        engine, world_id, session_id, _ = _map_session(
            store, walk_jpegs, intrinsics
        )
        engine.build(world_id, session_id)

        manifest = store.read_derived_manifest(world_id)

        assert manifest["input_digest"]
        assert store.derived_is_current(world_id, manifest["input_digest"])
        assert not store.derived_is_current(world_id, "some-other-digest")

    def test_rebuilding_a_purged_world_fails_loudly(
        self, tmp_path, walk_jpegs, intrinsics
    ):
        """Silently returning nothing would read as a mapping failure
        rather than the consequence of a deliberate privacy action."""
        from dataclasses import replace

        store = WorldStore(tmp_path)
        engine, world_id, session_id, _ = _map_session(
            store, walk_jpegs, intrinsics
        )
        store.write_world(replace(store.read_world(world_id), images_purged=True))

        with pytest.raises(ImagesPurgedError):
            engine.build(world_id, session_id)


class TestPrivacyPosture:
    def test_a_session_truthfully_declares_that_it_retains_imagery(
        self, tmp_path, walk_jpegs, intrinsics
    ):
        """Unlike every module shipped so far, which declares False."""
        store = WorldStore(tmp_path)
        _, world_id, session_id, _ = _map_session(store, walk_jpegs, intrinsics)

        session = store.read_session(world_id, session_id)

        assert session.retains_raw_imagery is True
        assert "raw-imagery" in session.privacy_tags

    def test_redaction_provenance_records_what_actually_ran(
        self, tmp_path, walk_jpegs, intrinsics
    ):
        """The un-retrofittable half of redaction work, now that it ships.

        Without this field there would be no way to tell whether an older
        session's images were filtered, so every historical session would
        have to be assumed unredacted forever.

        The value names the DETECTOR AND ITS THRESHOLD, not an outcome. A
        bare "redacted" would invite a reader to infer a completeness the
        detector cannot support -- it has measured false negatives on
        heavily occluded and ~90-degree-rotated faces.
        """
        store = WorldStore(tmp_path)
        _, world_id, session_id, _ = _map_session(store, walk_jpegs, intrinsics)

        session = store.read_session(world_id, session_id)
        raw = json.loads(store.session_path(world_id, session_id).read_text())

        assert session.redaction == raw["redaction"]
        assert raw["redaction"].startswith("faces-detected-and-filled/")
        assert "@" in raw["redaction"], "the threshold must be recorded"

        # Never an outcome claim, however the label is composed.
        for forbidden in ("anonymised", "anonymized", "privacy-safe", "removed"):
            assert forbidden not in raw["redaction"]

        # And the posture that redaction does NOT change.
        assert raw["retains_raw_imagery"] is True
        assert set(raw["privacy_tags"]) == {"raw-imagery", "first-person"}

    def test_a_session_records_none_when_no_redactor_is_available(
        self, tmp_path, walk_jpegs, intrinsics
    ):
        """A Tower with no model must say so, not claim a filter it lacks."""
        from tower.world_builder.redaction import FaceRedactor

        store = WorldStore(tmp_path)
        _, world_id, session_id, _ = _map_session(
            store,
            walk_jpegs,
            intrinsics,
            redactor_factory=lambda: FaceRedactor(path=tmp_path / "absent.onnx"),
        )

        raw = json.loads(store.session_path(world_id, session_id).read_text())
        assert raw["redaction"] == "none"

    def test_purging_a_world_removes_every_keyframe_image(
        self, tmp_path, walk_jpegs, intrinsics
    ):
        store = WorldStore(tmp_path)
        _, world_id, session_id, _ = _map_session(store, walk_jpegs, intrinsics)
        assert list(store.images_dir(world_id, session_id).glob("*.jpg"))

        report = store.purge_world(world_id)

        assert report.complete
        assert not store.world_dir(world_id).exists()


class TestStorageObservability:
    """Growth must be observable even though it is not capped.

    A hard cap that stopped mapping mid-session would trade a storage
    problem for a silently truncated world, which is worse -- an
    incomplete map with no sign it was cut short.
    """

    def test_world_bytes_reports_by_tier(self, tmp_path, walk_jpegs, intrinsics):
        store = WorldStore(tmp_path)
        engine, world_id, session_id, _ = _map_session(
            store, walk_jpegs, intrinsics
        )
        engine.build(world_id, session_id)

        totals = store.world_bytes(world_id)

        assert totals["images"] > 0
        assert totals["journals"] > 0
        assert totals["derived"] > 0
        assert totals["total"] >= (
            totals["images"] + totals["journals"] + totals["derived"]
        )

    def test_the_derived_tier_is_the_reclaimable_half(
        self, tmp_path, walk_jpegs, intrinsics
    ):
        store = WorldStore(tmp_path)
        engine, world_id, session_id, _ = _map_session(
            store, walk_jpegs, intrinsics
        )
        engine.build(world_id, session_id)
        before = store.world_bytes(world_id)

        store.clear_derived(world_id)
        after = store.world_bytes(world_id)

        assert after["derived"] == 0
        assert after["images"] == before["images"]
        assert after["total"] < before["total"]

    def test_a_missing_world_reports_zero_rather_than_raising(self, tmp_path):
        assert WorldStore(tmp_path).world_bytes("nope")["total"] == 0


class TestPersistedPoseConvention:
    """The persisted contract is T_world_camera: translation IS position.

    Regression cover for a shipped bug. The backends work in OpenCV's
    convention, where recoverPose and solvePnPRansac return (R, t) mapping
    a WORLD point into the CAMERA frame -- so `t` is where the world
    origin sits as seen by the camera, not a position. Writing it raw
    under the T_world_camera contract mirrors every camera through the
    origin.

    It is not a loud failure: a strafe along +X persisted as -X, smooth,
    monotonic and entirely plausible. Nothing caught it until the values
    were compared against ground-truth camera POSITIONS, which is why
    this test compares against those rather than against itself.
    """

    def _built_trajectory(self, tmp_path, walk_jpegs, intrinsics):
        from tower.world_builder.inspect import WorldView

        store = WorldStore(tmp_path)
        engine, world_id, session_id, _ = _map_session(
            store, walk_jpegs, intrinsics
        )
        engine.build(world_id, session_id)
        rows = WorldView(store, world_id).trajectory(session_id)
        return [row for row in rows if row["pose"] is not None]

    def test_persisted_translation_moves_in_the_direction_the_camera_moved(
        self, tmp_path, walk_jpegs, intrinsics
    ):
        """The walk fixture strafes along +X, so persisted X must increase."""
        import numpy as np

        solved = self._built_trajectory(tmp_path, walk_jpegs, intrinsics)
        assert len(solved) >= 3

        x = np.array([row["pose"]["translation"][0] for row in solved])

        assert np.all(np.diff(x) > 0), f"expected +X motion, got {x}"

    def test_persisted_trajectory_matches_ground_truth_camera_positions(
        self, tmp_path, walk_jpegs, intrinsics
    ):
        """Compared up to a similarity, since monocular scale is arbitrary.

        A mirrored trajectory does NOT survive this: Umeyama fits a
        rotation, not a reflection, so a reflected point set cannot be
        aligned onto the truth.
        """
        import numpy as np

        solved = self._built_trajectory(tmp_path, walk_jpegs, intrinsics)
        positions = np.array([row["pose"]["translation"] for row in solved])

        # Ground truth for the accepted keyframes, in walk order.
        indices = [int(row["source_seq"] // 7) for row in solved]
        truth = np.array([ss.strafe(14, step=0.09)[i].position for i in indices])

        scale, rotation, translation = ss.umeyama_similarity(positions, truth)
        aligned = (scale * (rotation @ positions.T).T) + translation
        residual = np.linalg.norm(aligned - truth, axis=1)
        path_length = float(
            np.linalg.norm(np.diff(truth, axis=0), axis=1).sum()
        )

        assert residual.max() / path_length < 0.15


class TestRebuildIsIdempotent:
    """A rebuild must recompute derived state, not accumulate it."""

    def test_rebuilding_does_not_duplicate_edges(
        self, tmp_path, walk_jpegs, intrinsics
    ):
        """Edges are recomputed from keyframes, so they are derived.

        Appending without clearing doubled the edge count on every
        rebuild, and the inspector reported the inflated number.
        """
        store = WorldStore(tmp_path)
        engine, world_id, session_id, _ = _map_session(
            store, walk_jpegs, intrinsics
        )

        engine.build(world_id, session_id)
        first = len(store.read_edges(world_id, session_id))
        engine.build(world_id, session_id)
        second = len(store.read_edges(world_id, session_id))

        assert first > 0
        assert second == first

    def test_rebuilding_preserves_a_measured_scale(
        self, tmp_path, walk_jpegs, intrinsics
    ):
        """A rebuild must not discard a scale it did not earn.

        ScaleState promises superseded estimates are appended, never
        overwritten. A rebuild that reconstructs a fresh ScaleState throws
        away meters_per_unit, method, confidence and history -- so a world
        that could print metres silently reverts to world units with no
        record a measured scale ever existed.
        """
        from dataclasses import replace

        from tower.confidence import Confidence
        from tower.world_builder.records import ScaleState
        from tower.world_builder.schema import SCALE_MEASURED

        store = WorldStore(tmp_path)
        engine, world_id, session_id, _ = _map_session(
            store, walk_jpegs, intrinsics
        )
        measured = ScaleState(
            state=SCALE_MEASURED,
            meters_per_unit=0.37,
            method="tape measure",
            confidence=Confidence.HIGH,
            history=({"note": "earlier estimate"},),
        )
        store.write_world(replace(store.read_world(world_id), scale=measured))

        engine.build(world_id, session_id)

        scale = store.read_world(world_id).scale
        assert scale.state == SCALE_MEASURED
        assert scale.meters_per_unit == 0.37
        assert scale.method == "tape measure"
        assert scale.history


class TestMultiSegmentHonesty:
    """Segments do not share a coordinate frame or a unit."""

    def _two_segment_world(self, tmp_path, scene, camera_matrix, intrinsics):
        """Map, break tracking hard, then map again at a different speed."""
        near = ss.render_sequence(
            scene, ss.strafe(8, step=0.06), camera_matrix, WIDTH, HEIGHT
        )
        far = ss.render_sequence(
            scene, ss.strafe(8, step=0.30, start=(-2.0, -1.6, 2.4)),
            camera_matrix, WIDTH, HEIGHT,
        )
        noise = [
            np.random.default_rng(seed).integers(
                0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8
            )
            for seed in range(4)
        ]
        frames = [ss.encode_jpeg(image) for image in [*near, *noise, *far]]

        store = WorldStore(tmp_path)
        engine = WorldBuilderEngine(store)
        world_id = engine.create_world()
        session_id = engine.start_session(
            world_id, intrinsics=intrinsics, declared_size=(WIDTH, HEIGHT)
        )
        for index, payload in enumerate(frames):
            engine.observe(payload, source_seq=index)
        engine.stop_session()
        return store, engine, world_id, session_id

    def test_a_multi_segment_world_does_not_claim_relative_scale(
        self, tmp_path, scene, camera_matrix, intrinsics
    ):
        """"Relative" asserts internal consistency the world does not have.

        Each segment is solved in its own window, so each carries its own
        arbitrary unit -- measured 4x apart across two segments of one
        session. Claiming a single relative scale would be a fiction.
        """
        store, engine, world_id, session_id = self._two_segment_world(
            tmp_path, scene, camera_matrix, intrinsics
        )

        result = engine.build(world_id, session_id)

        if result.segments > 1:
            assert result.scale_state == SCALE_UNKNOWN
            assert not store.read_world(world_id).scale.allows_metres

    def test_points_carry_the_segment_that_produced_them(
        self, tmp_path, scene, camera_matrix, intrinsics
    ):
        """Untagged concatenation merges incompatible geometry into one blob."""
        store, engine, world_id, session_id = self._two_segment_world(
            tmp_path, scene, camera_matrix, intrinsics
        )
        engine.build(world_id, session_id)

        derived = store.read_derived(world_id, session_id)

        assert derived["points"]
        for point in derived["points"]:
            assert "segment_index" in point
            assert len(point["xyz"]) == 3
