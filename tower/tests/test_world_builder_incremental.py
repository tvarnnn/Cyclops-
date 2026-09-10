"""Extending a live solve must equal solving from scratch. Bit for bit.

SYNTHETIC, NOT PHYSICAL. The rooms are rendered by a perfect pinhole; no
number here says anything about the Ray-Ban camera.

`build()` used to re-solve everything on every call, so a walk rebuilt
every k keyframes cost O(N^2/k) and turning live updates UP made the
session slower. `ClassicalTwoViewBackend` was already strictly
forward-only, so the fix was to stop throwing the solve away -- see the
comment on `GeometryBackend.begin`.

That is a refactor whose entire claim is "the answer does not change",
and this file is where that claim is checked. The comparison is EXACT,
with no tolerance at all, and the tolerance is zero for a reason that is
itself tested here: on this OpenCV build the backend is deterministic,
so the same window solved twice is byte-identical (see
`test_the_oracle_is_deterministic_which_is_why_the_tolerance_is_zero`).
Once that holds, any difference between the two paths is state that was
not promoted, and rounding it away with an `allclose` would hide exactly
the bug this file exists to find.

`estimate_window()` is the oracle. It is deliberately NOT implemented in
terms of the incremental path -- the two share the per-pair geometry
helpers, where a copy would certainly drift, and keep their own
orchestration, which is the part being checked.
"""

import time

import numpy as np
import pytest

from tests import synthetic_scene as ss
from tower.world_builder.backend import (
    BackendCapabilities,
    GeometryBackend,
    GeometryEstimate,
    KeyframeInput,
    PointBlock,
    PoseEstimate,
)
from tower.world_builder.backends import classical as classical_module
from tower.world_builder.backends.classical import ClassicalTwoViewBackend
from tower.world_builder.backends.unposed import UnposedBackend
from tower.world_builder.engine import WorldBuilderEngine
from tower.world_builder.records import CameraIntrinsics
from tower.world_builder.redaction import RedactionResult
from tower.world_builder.schema import (
    INTRINSICS_SOURCE_SELF_CALIBRATED,
    POSE_STATUS_ANCHOR,
    POSE_STATUS_SOLVED,
)
from tower.world_builder.store import WorldStore

WIDTH, HEIGHT = 480, 360

# Every field a PoseEstimate carries, not a chosen few. A refusal that
# differs only in `matches` is still a different answer, and the
# measured signals are what a degeneracy is explained by.
POSE_FIELDS = (
    "keyframe_id",
    "status",
    "degeneracy",
    "matches",
    "inliers",
    "inlier_ratio",
    "median_triangulation_deg",
    "median_displacement_px",
    "cheirality_fraction",
    "r_h",
)


# -- fixtures ----------------------------------------------------------


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


def _grays(scene, camera_matrix, poses):
    import cv2

    return [
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for image in ss.render_sequence(scene, poses, camera_matrix, WIDTH, HEIGHT)
    ]


def _window(grays, prefix="kf"):
    return [
        KeyframeInput(keyframe_id=f"{prefix}{index:04d}", image_gray=gray)
        for index, gray in enumerate(grays)
    ]


@pytest.fixture(scope="module")
def sequences(scene, camera_matrix):
    """One window per motion type, plus the two shapes that break a chain.

    Motion type is not decoration. Lateral motion solves cleanly, forward
    motion puts the epipole in the image, and pure rotation is refused
    outright -- and the refusal paths are exactly where an incremental
    solve could disagree, because that is where estimate_window() stops
    chaining and starts filling in.
    """
    rng = np.random.default_rng(7)
    noise = [
        rng.integers(0, 255, (HEIGHT, WIDTH), dtype=np.uint8) for _ in range(3)
    ]
    strafe = _grays(scene, camera_matrix, ss.strafe(12, step=0.09))
    return {
        "strafe": _window(strafe),
        "forward_walk": _window(
            _grays(scene, camera_matrix, ss.forward_walk(12, step=0.12))
        ),
        "pure_rotation": _window(
            _grays(scene, camera_matrix, ss.pure_rotation(8, degrees_per_step=2.0))
        ),
        # A chain that starts well and then loses correspondence: the
        # branch where estimate_window() stops and marks the tail
        # unavailable with the failing frame's degeneracy.
        "chain_breaks": _window(strafe[:6] + noise + strafe[6:9]),
        # A chain that never starts: the first pair itself refuses.
        "never_starts": _window(noise + strafe[:4]),
        "single": _window(strafe[:1]),
        "pair": _window(strafe[:2]),
    }


SEQUENCE_NAMES = (
    "strafe",
    "forward_walk",
    "pure_rotation",
    "chain_breaks",
    "never_starts",
    "single",
    "pair",
)


# -- exact comparison --------------------------------------------------


def _pose_digest(pose):
    scalars = tuple(getattr(pose, name) for name in POSE_FIELDS)
    arrays = tuple(
        None if value is None else np.asarray(value, dtype=np.float64).tobytes()
        for value in (pose.rotation, pose.translation)
    )
    return scalars + arrays


def _digest(estimate):
    points = estimate.points
    return (
        tuple(_pose_digest(pose) for pose in estimate.poses),
        None if points is None else (
            points.xyz.dtype.str,
            points.xyz.shape,
            points.xyz.tobytes(),
            # The support table was omitted here, so "bit-identical live
            # vs batch" did not cover the association index at all -- the
            # very table publication filtering rewrites. An adversarial
            # review verified equality held; it was simply unasserted.
            None if points.support_views is None else (
                points.support_views.dtype.str,
                points.support_views.shape,
                points.support_views.tobytes(),
            ),
        ),
        estimate.diagnostics,
        estimate.scale_is_metric,
    )


def _solve_cold(backend_factory, intrinsics, window):
    backend = backend_factory()
    backend.prepare(intrinsics)
    return backend.estimate_window(window)


def _solve_live(backend_factory, intrinsics, window, snapshot_every=None):
    backend = backend_factory()
    backend.begin(intrinsics)
    for index, frame in enumerate(window, start=1):
        backend.extend(frame)
        if snapshot_every and index % snapshot_every == 0:
            backend.snapshot()
    return backend.snapshot()


# -- the equivalence property ------------------------------------------


class TestBitIdenticalEquivalence:
    @pytest.mark.parametrize("name", SEQUENCE_NAMES)
    def test_extending_equals_solving_the_whole_window(
        self, name, sequences, intrinsics
    ):
        window = sequences[name]

        oracle = _solve_cold(ClassicalTwoViewBackend, intrinsics, window)
        live = _solve_live(ClassicalTwoViewBackend, intrinsics, window)

        assert len(live.poses) == len(oracle.poses)
        for index, (want, got) in enumerate(zip(oracle.poses, live.poses)):
            assert _pose_digest(got) == _pose_digest(want), (
                f"{name}: pose {index} differs. Not a tolerance question -- "
                "state that was not promoted"
            )
        assert _digest(live)[1] == _digest(oracle)[1], (
            f"{name}: the point cloud differs"
        )

    def test_the_oracle_is_deterministic_which_is_why_the_tolerance_is_zero(
        self, sequences, intrinsics
    ):
        """The justification for comparing with `==` and nothing else.

        findEssentialMat(USAC_MAGSAC) and solvePnPRansac(SQPNP) are
        RANSAC, and a comment in classical.py warns that they are not
        seeded. On this OpenCV build they are nonetheless reproducible:
        the same window solved twice, with unrelated OpenCV RANSAC and
        ORB work in between, comes back byte-identical.

        If that ever stops holding, THIS test fails first and says so,
        rather than the equivalence tests above failing and inviting
        somebody to loosen them into meaninglessness.
        """
        window = sequences["strafe"]
        first = _digest(_solve_cold(ClassicalTwoViewBackend, intrinsics, window))

        for frame in window:
            classical_module.detect_and_describe(frame.image_gray)
        second = _digest(_solve_cold(ClassicalTwoViewBackend, intrinsics, window))

        assert first == second

    @pytest.mark.parametrize("name", ("strafe", "chain_breaks"))
    def test_snapshotting_as_it_goes_changes_nothing(
        self, name, sequences, intrinsics
    ):
        """A viewer must not be a participant.

        This is the backend-level twin of the mid-walk rebuild invariant:
        snapshot() is what a rebuild reads, and reading must not disturb
        the solve it read.
        """
        window = sequences[name]

        oracle = _solve_cold(ClassicalTwoViewBackend, intrinsics, window)
        watched = _solve_live(
            ClassicalTwoViewBackend, intrinsics, window, snapshot_every=1
        )

        assert _digest(watched) == _digest(oracle)

    def test_a_snapshot_partway_equals_solving_that_prefix(
        self, sequences, intrinsics
    ):
        """A mid-walk rebuild is the answer for the keyframes so far."""
        window = sequences["strafe"]
        backend = ClassicalTwoViewBackend()
        backend.begin(intrinsics)

        for count in range(1, len(window) + 1):
            backend.extend(window[count - 1])
            prefix = _solve_cold(
                ClassicalTwoViewBackend, intrinsics, window[:count]
            )
            assert _digest(backend.snapshot()) == _digest(prefix), count

    def test_an_empty_solve_reports_nothing_rather_than_guessing(self, intrinsics):
        backend = ClassicalTwoViewBackend()
        backend.begin(intrinsics)

        assert backend.snapshot() == GeometryEstimate(poses=())

    def test_extend_reports_only_the_structure_that_keyframe_added(
        self, sequences, intrinsics
    ):
        """`new_points` is a delta, and the deltas account for the map.

        The sum used to EQUAL the snapshot. It no longer can, and the
        difference is a real contract change worth stating rather than
        papering over: the local bundle adjustment can move a landmark
        somewhere its own observations no longer support, and
        `_local_adjust` then retires it from publication. A delta is
        therefore what the backend believed when it emitted it, and
        `snapshot()` is what it believes now.

        The shortfall is not slack. It is EXACTLY the number of
        landmarks the adjustment retired, which is what this asserts --
        a delta that vanished for any other reason would fail here.

        A live viewer must consequently re-read `snapshot()` rather than
        append deltas forever. Nothing in production does either today:
        `Extension.new_points` is read by no code outside this test.
        """
        window = sequences["strafe"]
        backend = ClassicalTwoViewBackend()
        backend.begin(intrinsics)

        added = 0
        for frame in window:
            step = backend.extend(frame)
            assert step.pose.keyframe_id == frame.keyframe_id
            if step.new_points is not None:
                added += len(step.new_points)

        published = len(backend.snapshot().points)
        assert added - published == backend._chain.demoted, (
            f"deltas summed to {added}, snapshot publishes {published}, and "
            f"the adjustment retired {backend._chain.demoted} -- those three "
            f"must reconcile exactly"
        )


# -- segments are independent windows, and must stay so ----------------


class TestSegmentIsolation:
    def test_reset_starts_a_genuinely_fresh_solve(self, sequences, intrinsics):
        """A tracking_lost is a new coordinate frame and a new unit.

        Anything carried across it would not be a small error, it would
        be two incompatible geometries reported as one.
        """
        first, second = sequences["strafe"], sequences["forward_walk"]

        backend = ClassicalTwoViewBackend()
        backend.begin(intrinsics)
        for frame in first:
            backend.extend(frame)
        carried = backend.snapshot()
        backend.reset()
        for frame in second:
            backend.extend(frame)
        after_reset = backend.snapshot()

        alone = _solve_cold(ClassicalTwoViewBackend, intrinsics, second)
        assert _digest(after_reset) == _digest(alone)
        # Stated separately from the digest so a failure says which of
        # the two things went wrong.
        assert len(after_reset.poses) == len(second)
        assert len(after_reset.points) == len(alone.points)
        assert len(carried.points) != len(after_reset.points), (
            "the two segments happen to have identical point counts; this "
            "assertion can no longer tell a leak from a coincidence"
        )

    def test_no_landmark_survives_a_reset(self, sequences, intrinsics):
        """Checked on the state itself, not only through the output."""
        backend = ClassicalTwoViewBackend()
        backend.begin(intrinsics)
        for frame in sequences["strafe"]:
            backend.extend(frame)
        assert backend._chain.landmarks

        backend.reset()

        chain = backend._chain
        assert chain.count == 0
        assert chain.landmarks == []
        assert chain.observed == {}
        assert chain.absolute == {}
        assert chain.poses == []
        assert chain.broken is None
        assert chain.references == []
        assert chain.failures == 0


# -- the refusal to invent intrinsics survives the new entry point -----


class TestIntrinsicsAreStillRequired:
    def test_begin_refuses_a_camera_nobody_calibrated(self):
        """An incremental path is not a side door around prepare()."""
        backend = ClassicalTwoViewBackend()

        with pytest.raises(ValueError, match="requires known intrinsics"):
            backend.begin(CameraIntrinsics.unknown())

    def test_extending_before_beginning_is_an_error_not_a_guess(self, sequences):
        backend = ClassicalTwoViewBackend()

        with pytest.raises(RuntimeError, match="begin"):
            backend.extend(sequences["strafe"][0])


# -- the unposed backend gets the same seam, and stays as honest -------


class TestUnposedIncremental:
    @pytest.mark.parametrize("name", ("strafe", "pure_rotation", "single"))
    def test_extending_equals_the_whole_window(self, name, sequences):
        window = sequences[name]
        unknown = CameraIntrinsics.unknown()

        oracle = _solve_cold(UnposedBackend, unknown, window)
        live = _solve_live(UnposedBackend, unknown, window)

        assert _digest(live) == _digest(oracle)

    def test_it_still_withholds_every_pose_and_every_point(self, sequences):
        live = _solve_live(
            UnposedBackend, CameraIntrinsics.unknown(), sequences["strafe"]
        )

        assert live.points is None
        assert live.poses[0].status == POSE_STATUS_ANCHOR
        assert all(pose.rotation is None for pose in live.poses)
        assert all(pose.translation is None for pose in live.poses)
        assert not any(pose.status == POSE_STATUS_SOLVED for pose in live.poses)


# -- the seam is total: a backend that cannot be extended still works --


class _WholeWindowOnlyBackend(GeometryBackend):
    """A backend that only knows how to solve a whole window.

    Stands in for the feed-forward pointmap model the seam exists for:
    it reasons over the submap and cannot be extended a frame at a time.
    The default begin/extend/snapshot must therefore still give the
    engine correct answers, however expensively.
    """

    capabilities = BackendCapabilities(
        backend_id="whole-window-only",
        version="1",
        requires_intrinsics=False,
        estimates_intrinsics=False,
        produces_dense_geometry=False,
        produces_metric_scale=False,
        preferred_window=8,
    )

    def __init__(self):
        self.window_calls = 0

    def prepare(self, intrinsics):
        return None

    def estimate_window(self, window):
        self.window_calls += 1
        if not window:
            return GeometryEstimate(poses=())
        return GeometryEstimate(
            poses=tuple(
                PoseEstimate(keyframe_id=frame.keyframe_id, status=POSE_STATUS_ANCHOR)
                for frame in window
            ),
            points=PointBlock(
                xyz=np.arange(3 * len(window), dtype=np.float32).reshape(-1, 3)
            ),
        )


class TestTheDefaultSeamIsCorrect:
    def test_a_window_only_backend_still_answers_incrementally(self, sequences):
        window = sequences["strafe"]
        unknown = CameraIntrinsics.unknown()

        oracle = _solve_cold(_WholeWindowOnlyBackend, unknown, window)
        live = _solve_live(_WholeWindowOnlyBackend, unknown, window)

        assert _digest(live) == _digest(oracle)

    def test_and_pays_for_it_with_a_re_solve_per_keyframe(self, sequences):
        """The default is correct and quadratic, and says so out loud."""
        window = sequences["strafe"]
        backend = _WholeWindowOnlyBackend()
        backend.begin(CameraIntrinsics.unknown())

        for frame in window:
            backend.extend(frame)

        assert backend.window_calls == len(window)

    def test_reset_discards_the_buffer(self, sequences):
        backend = _WholeWindowOnlyBackend()
        backend.begin(CameraIntrinsics.unknown())
        for frame in sequences["strafe"]:
            backend.extend(frame)

        backend.reset()

        assert backend.snapshot() == GeometryEstimate(poses=())


# -- cost -------------------------------------------------------------


class TestPerKeyframeWorkIsConstant:
    def test_a_keyframe_is_detected_once_no_matter_how_often_we_rebuild(
        self, sequences, intrinsics, monkeypatch
    ):
        """The structural form of the claim, with no clock in it.

        Feature detection is the dominant per-frame cost, so counting the
        calls measures the complexity directly. Re-solving detects every
        keyframe again on every rebuild -- 42 detections for 12 keyframes
        at a cadence of 2, and the ratio widens with N because the
        baseline is quadratic. Extending detects each keyframe once,
        ever, however often it is watched.
        """
        window = sequences["strafe"]
        cadence = 2
        calls = []
        real = classical_module.detect_and_describe
        monkeypatch.setattr(
            classical_module,
            "detect_and_describe",
            lambda gray: (calls.append(1), real(gray))[1],
        )

        re_solving = ClassicalTwoViewBackend()
        re_solving.prepare(intrinsics)
        for count in range(cadence, len(window) + 1, cadence):
            re_solving.estimate_window(window[:count])
        before = len(calls)

        calls.clear()
        extending = ClassicalTwoViewBackend()
        extending.begin(intrinsics)
        for index, frame in enumerate(window, start=1):
            extending.extend(frame)
            if index % cadence == 0:
                extending.snapshot()
        after = len(calls)

        assert after == len(window)
        assert before == sum(
            range(cadence, len(window) + 1, cadence)
        ), "the re-solving baseline is not the O(N^2/k) shape this claims"
        assert before == 42
        assert after * 3 <= before

    def test_the_last_keyframe_costs_about_what_the_first_one_did(
        self, scene, camera_matrix, intrinsics
    ):
        """The timed form. Deliberately a loose bound.

        A wall clock inside a test suite is noise, so this is not trying
        to measure the constant -- it is trying to catch the shape. If
        per-keyframe work were still linear in N, the last quarter of a
        48-keyframe walk would cost around five times the first quarter,
        not the 1.09x measured here at N=64. A 2.5x bound separates those
        two answers with room to spare on a loaded machine.
        """
        window = _window(_grays(scene, camera_matrix, ss.strafe(48, step=0.045)))
        backend = ClassicalTwoViewBackend()
        backend.begin(intrinsics)

        marks = []
        for frame in window:
            started = time.perf_counter()
            backend.extend(frame)
            marks.append(time.perf_counter() - started)

        # Skipping the first eight: they carry ORB and OpenCV allocator
        # warm-up, measured at 2-3x the steady state and nothing to do
        # with N.
        early = float(np.mean(marks[8:20]))
        late = float(np.mean(marks[-12:]))
        assert late < early * 2.5, (
            f"per-keyframe cost grew: {early * 1e3:.2f} ms early, "
            f"{late * 1e3:.2f} ms late"
        )

    def test_retained_state_does_not_grow_with_the_number_of_keyframes(
        self, scene, camera_matrix, intrinsics
    ):
        """Landmarks are the map and must grow. Nothing else may.

        `observed` maps (frame, feature) to landmark, and _extend() only
        ever reads the previous frame's entries. Unpruned it was measured
        at 26.1 MB after 155 keyframes and 142.9 MB after 1000; pruned it
        is flat at about 0.15 MB, because what survives is one frame's
        features.
        """
        window = _window(_grays(scene, camera_matrix, ss.strafe(40, step=0.045)))
        backend = ClassicalTwoViewBackend()
        backend.begin(intrinsics)

        sizes = []
        for index, frame in enumerate(window, start=1):
            backend.extend(frame)
            if index in (10, 20, 40):
                sizes.append(len(backend._chain.observed))

        assert backend._chain.landmarks, "the walk solved nothing to measure"
        assert max(sizes) < min(sizes) * 2, (
            f"observations are accumulating rather than being pruned: {sizes}"
        )


# -- the engine ---------------------------------------------------------


class _AlwaysReencodes:
    """A redactor that changes the bytes on every frame.

    Not a face detector -- it does not need to be. What it reproduces is
    the one property that matters here: the pixels persisted are not the
    pixels that arrived. A live solve fed the incoming frame instead of
    the redacted one agrees with a cold rebuild on clean footage and
    diverges the moment anything is redacted, which is the worst possible
    shape for a bug.
    """

    available = True
    unavailable_reason = None
    label = "test-reencode"

    def redact(self, image_bytes: bytes) -> RedactionResult:
        import cv2

        image = cv2.imdecode(
            np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR
        )
        cv2.rectangle(image, (40, 40), (140, 140), (0, 0, 0), -1)
        ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        assert ok
        return RedactionResult(
            image_bytes=buffer.tobytes(), label=self.label, regions=1
        )


@pytest.fixture(scope="module")
def walk_jpegs(scene, camera_matrix):
    poses = ss.strafe(14, step=0.09)
    return [
        ss.encode_jpeg(image)
        for image in ss.render_sequence(scene, poses, camera_matrix, WIDTH, HEIGHT)
    ]


@pytest.fixture(scope="module")
def two_segment_jpegs(scene, camera_matrix):
    """Map, break tracking hard with noise, then map again elsewhere."""
    near = ss.render_sequence(
        scene, ss.strafe(8, step=0.06), camera_matrix, WIDTH, HEIGHT
    )
    far = ss.render_sequence(
        scene,
        ss.strafe(8, step=0.30, start=(-2.0, -1.6, 2.4)),
        camera_matrix,
        WIDTH,
        HEIGHT,
    )
    rng = np.random.default_rng(3)
    noise = [
        rng.integers(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8) for _ in range(4)
    ]
    return [ss.encode_jpeg(image) for image in [*near, *noise, *far]]


def _observe_all(store, payloads, intrinsics, *, rebuild_every=0, redactor=None):
    engine = WorldBuilderEngine(store, redactor_factory=redactor)
    world_id = engine.create_world("Incremental")
    session_id = engine.start_session(
        world_id,
        intrinsics=intrinsics,
        frame_source="synthetic",
        declared_size=(WIDTH, HEIGHT),
    )
    accepted = 0
    since = 0
    for index, payload in enumerate(payloads):
        outcome = engine.observe(payload, source_seq=index)
        if outcome.keyframe_id is None:
            continue
        accepted += 1
        since += 1
        # A HAND-ROLLED GATE, deliberately, and it is not the production
        # one: `main()` widens the interval with the world
        # (`rebuild_interval`). These walks are well under the 600-keyframe
        # knee, where the two agree exactly, and the property under test --
        # that a mid-walk rebuild does not change the final result -- is
        # about rebuilding AT ALL, not about the cadence. Flagged by a
        # reviewer as pinning a schedule production no longer uses; the
        # answer is that it pins no schedule, and saying so is cheaper than
        # importing the CLI into a unit test.
        if rebuild_every and since >= rebuild_every and accepted >= 2:
            engine.build(world_id, session_id)
            since = 0
    engine.stop_session()
    return engine, world_id, session_id


def _derived(store, world_id, session_id):
    derived = store.read_derived(world_id, session_id)
    assert derived is not None, "the derived output is missing or stale"
    manifest = dict(store.read_derived_manifest(world_id))
    # Not part of the reconstruction: a wall clock.
    manifest.pop("built_at", None)
    return derived["poses"], derived["points"], manifest


def _geometry_without_ids(store, world_id, session_id):
    poses, points, _ = _derived(store, world_id, session_id)
    stripped = [
        {key: value for key, value in pose.items() if key != "keyframe_id"}
        for pose in poses
    ]
    return stripped, points


class TestTheEngineFlushesRatherThanResolves:
    @pytest.mark.parametrize("redactor", (None, _AlwaysReencodes))
    def test_a_live_build_equals_a_cold_build_of_the_same_keyframes(
        self, tmp_path, walk_jpegs, intrinsics, redactor
    ):
        """The whole refactor, stated once.

        The redacted variant is the trap worth naming: build() reads the
        REDACTED bytes back off disk, so a live solve fed the incoming
        frame would agree here on clean footage and quietly disagree the
        moment a face was filled.
        """
        store = WorldStore(tmp_path)
        engine, world_id, session_id = _observe_all(
            store, walk_jpegs, intrinsics, redactor=redactor
        )

        live = engine.build(world_id, session_id)
        live_derived = _derived(store, world_id, session_id)

        # A second engine has no live solve, so this is the from-scratch
        # path over exactly the same journal.
        cold_engine = WorldBuilderEngine(store)
        cold = cold_engine.build(world_id, session_id)
        cold_derived = _derived(store, world_id, session_id)

        assert live == cold
        assert live_derived[0] == cold_derived[0], "poses differ"
        assert live_derived[1] == cold_derived[1], "points differ"
        assert live_derived[2] == cold_derived[2], "manifest differs"
        assert live.poses_solved > 0, "this walk solved nothing to compare"

    def test_rebuilding_mid_walk_does_not_change_the_final_result(
        self, tmp_path, walk_jpegs, intrinsics
    ):
        """The engine-level twin of the follow-CLI invariant.

        If watching cost anything, the viewer would be a participant and
        two operators would get different maps from identical footage.
        """
        plain_store = WorldStore(tmp_path / "plain")
        plain_engine, plain_world, plain_session = _observe_all(
            plain_store, walk_jpegs, intrinsics
        )
        plain = plain_engine.build(plain_world, plain_session)

        watched_store = WorldStore(tmp_path / "watched")
        watched_engine, watched_world, watched_session = _observe_all(
            watched_store, walk_jpegs, intrinsics, rebuild_every=1
        )
        watched = watched_engine.build(watched_world, watched_session)

        for field in (
            "keyframes",
            "poses_solved",
            "poses_refused",
            "points",
            "segments",
            "scale_state",
        ):
            assert getattr(plain, field) == getattr(watched, field), field
        # The geometry itself, not just the counters the CLI test pins.
        # Keyframe ids are dropped because they carry the session id, and
        # these are deliberately two different sessions over identical
        # footage -- which is the comparison the invariant is about.
        assert _geometry_without_ids(
            plain_store, plain_world, plain_session
        ) == _geometry_without_ids(watched_store, watched_world, watched_session)

    def test_a_lost_track_leaks_nothing_across_the_boundary(
        self, tmp_path, two_segment_jpegs, intrinsics
    ):
        store = WorldStore(tmp_path)
        engine, world_id, session_id = _observe_all(
            store, two_segment_jpegs, intrinsics, rebuild_every=2
        )

        live = engine.build(world_id, session_id)
        live_derived = _derived(store, world_id, session_id)
        cold = WorldBuilderEngine(store).build(world_id, session_id)
        cold_derived = _derived(store, world_id, session_id)

        assert live.segments > 1, "tracking never broke; nothing was tested"
        assert live == cold
        assert live_derived[0] == cold_derived[0]
        assert live_derived[1] == cold_derived[1]
        # Per segment as well as in total: a leak that moved points from
        # one segment to another would keep the total intact.
        per_segment = {}
        for point in live_derived[1]:
            per_segment[point["segment_index"]] = (
                per_segment.get(point["segment_index"], 0) + 1
            )
        cold_per_segment = {}
        for point in cold_derived[1]:
            cold_per_segment[point["segment_index"]] = (
                cold_per_segment.get(point["segment_index"], 0) + 1
            )
        assert per_segment == cold_per_segment

    def test_a_world_this_engine_never_observed_is_solved_from_scratch(
        self, tmp_path, walk_jpegs, intrinsics
    ):
        """The from-scratch path is load-bearing, not vestigial.

        A cold rebuild, a re-derive, and `--frames` against an existing
        world all land here, and none of them has a live solve to flush.
        """
        store = WorldStore(tmp_path)
        _, world_id, session_id = _observe_all(store, walk_jpegs, intrinsics)

        stranger = WorldBuilderEngine(store)
        result = stranger.build(world_id, session_id)

        assert stranger._live is None
        assert result.poses_solved > 0

    def test_a_live_solve_that_fails_costs_the_session_nothing(
        self, tmp_path, walk_jpegs, intrinsics, monkeypatch
    ):
        """Live geometry is an optimisation and must fail like one.

        A backend that throws on the frame path would otherwise take the
        keyframes down with it -- and the keyframes are the only thing in
        this pipeline that cannot be recomputed.
        """
        store = WorldStore(tmp_path)

        def explode(self, frame):
            raise RuntimeError("backend fell over")

        monkeypatch.setattr(ClassicalTwoViewBackend, "extend", explode)
        engine, world_id, session_id = _observe_all(store, walk_jpegs, intrinsics)
        monkeypatch.undo()

        assert engine._live is not None and not engine._live.usable
        result = engine.build(world_id, session_id)

        assert result.keyframes > 1
        assert result.poses_solved > 0

    def test_a_second_session_does_not_inherit_the_first_ones_solve(
        self, tmp_path, walk_jpegs, intrinsics
    ):
        store = WorldStore(tmp_path)
        engine, world_id, first_session = _observe_all(
            store, walk_jpegs, intrinsics
        )
        first = engine.build(world_id, first_session)

        second_session = engine.start_session(
            world_id,
            intrinsics=intrinsics,
            frame_source="synthetic",
            declared_size=(WIDTH, HEIGHT),
        )
        for index, payload in enumerate(walk_jpegs):
            engine.observe(payload, source_seq=index)
        engine.stop_session()
        second = engine.build(world_id, second_session)

        assert second.session_id == second_session
        assert second.keyframes == first.keyframes
        assert second.poses_solved == first.poses_solved
        assert second.points == first.points
        # And the first session's world still rebuilds from scratch
        # rather than being handed the second session's solve.
        rebuilt = WorldBuilderEngine(store).build(world_id, first_session)
        assert rebuilt.points == first.points


def test_a_frame_of_a_different_size_does_not_kill_the_walk(tmp_path):
    """An unguarded C assertion on the live frame path.

    `MotionTracker.measure` feeds this frame and a stored reference frame
    straight into `cv2.calcOpticalFlowPyrLK`, which asserts they are the
    same size -- in C, as a `cv2.error`, which is not a `ValueError` and
    walks past `observe`'s decode guard. `world_build_session.py` catches
    only `OSError` around the frame loop, so it reaches the outermost
    `except BaseException`: session `end_reason: error`, finalization
    `interrupted`, every remaining frame discarded.

    A reviewer drove it through a real Tower -- 220 frames with a
    resolution change at frame 120 -- and watched a wearer who walked the
    whole room get "Interrupted".

    Rejecting is the correct answer, not merely the safe one: the
    calibration is per-resolution and exact, so a frame at a size this
    session is not calibrated for could not have produced a usable pose.
    """
    import numpy as np

    from tests import synthetic_scene as ss
    from tower.world_builder.engine import WorldBuilderEngine
    from tower.world_builder.records import CameraIntrinsics
    from tower.world_builder.store import WorldStore

    width, height = 480, 360
    camera_matrix = ss.camera_matrix(width, height)
    scene = ss.furnished_room()
    poses = ss.strafe(8, step=0.09)
    images = ss.render_sequence(scene, poses, camera_matrix, width, height)

    engine = WorldBuilderEngine(WorldStore(tmp_path / "worlds"))
    world_id = engine.create_world("Rung Change")
    session_id = engine.start_session(
        world_id,
        intrinsics=CameraIntrinsics(
            source="self_calibrated", model="pinhole",
            fx=float(camera_matrix[0, 0]), fy=float(camera_matrix[1, 1]),
            cx=float(camera_matrix[0, 2]), cy=float(camera_matrix[1, 2]),
            calibrated_width=width, calibrated_height=height,
        ),
        frame_source="synthetic",
        declared_size=(width, height),
    )
    try:
        for index, image in enumerate(images[:4]):
            engine.observe(ss.encode_jpeg(image), source_seq=index, wire_seq=index)

        # THE RUNG CHANGE. A perfectly decodable JPEG of another size.
        smaller = np.ascontiguousarray(images[4][:288, :384])
        result = engine.observe(ss.encode_jpeg(smaller), source_seq=4, wire_seq=4)
        assert result is not None
        assert getattr(result, "reason", None) == "frame_size_changed", result

        # ...and the walk carries on. This is the half that matters: the
        # remaining frames were being discarded with the session.
        for index, image in enumerate(images[5:], start=5):
            engine.observe(ss.encode_jpeg(image), source_seq=index, wire_seq=index)

        summary = engine.stop_session()
        assert summary is not None
    finally:
        try:
            engine.stop_session()
        except Exception:
            pass

    store = WorldStore(tmp_path / "worlds")
    session = store.read_session(world_id, session_id)
    assert session.end_reason != "error", (
        "one frame of the wrong size ended the whole walk"
    )
    assert session.keyframes_accepted > 0
