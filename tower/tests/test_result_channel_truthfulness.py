"""Does the wire tell the truth about what World Builder actually did?

Checked against INDEPENDENT truth wherever possible: the numbers the
engine returned, or the files on disk, read separately from the code
under test. A test that asserted the payload matched the producer's own
view of the world would pass for a producer that fabricated everything.
"""

import json

import pytest

from tests.result_channel_fixtures import (  # noqa: F401
    _close_result_channel_clients,
    build_world,
    drain,
    make_client,
    start_live_world,
    subscribe,
)
from tower.results.world_builder import WorldBuilderStatusProducer
from tower.world_builder.store import WorldStore


def _payload(monkeypatch, root, **overrides):
    client = make_client(monkeypatch, root)
    with client.websocket_connect("/ws") as ws:
        subscribe(ws, **overrides)
        return drain(ws, expect="cartridge_result")["payload"]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("truth")
    world_id, session_id = build_world(root, frames=10)
    return root, world_id, session_id


# -- the live-session trap ---------------------------------------------


def test_a_live_keyframe_count_comes_from_the_journal_not_session_json(
    monkeypatch, tmp_path
):
    """The single most dangerous fabrication available here.

    session.json is written at start_session with keyframes_accepted=0 and
    is not rewritten until stop_session. So a producer that read the
    obvious field would report ZERO while keyframes were being accepted --
    and zero looks like a measurement, not like a gap.

    Independent truth: the number of lines in keyframes.jsonl, and the
    engine's own in-memory count.
    """
    root = tmp_path / "worlds"
    world_id, session_id, engine = start_live_world(root, frames=10)
    try:
        store = WorldStore(root)
        on_disk = json.loads(
            store.session_path(world_id, session_id).read_text(encoding="utf-8")
        )
        truth = len(store.read_keyframes(world_id, session_id))

        assert on_disk["keyframes_accepted"] == 0, (
            "precondition: session.json really does still hold zero"
        )
        assert truth > 0, "precondition: keyframes really were accepted"

        payload = _payload(monkeypatch, root)
        assert payload["lifecycle"]["state"] == "receiving"
        assert payload["progress"]["keyframes_accepted"] == truth
    finally:
        engine.stop_session()


def test_frames_observed_is_null_while_live_with_its_reason(monkeypatch, tmp_path):
    """Not knowable yet, so not reported. nil and 0 are different claims.

    An ordinary rejected frame writes no journal event -- only a malformed
    one does -- so there is genuinely no live source for this count. A
    producer that reported session.json's zero would be claiming the Tower
    had observed nothing.
    """
    root = tmp_path / "worlds"
    world_id, session_id, engine = start_live_world(root, frames=10)
    try:
        payload = _payload(monkeypatch, root)
        progress = payload["progress"]

        assert progress["frames_observed"] is None
        assert "not knowable yet" in (
            progress["frames_observed_unavailable_reason"]
        )
        assert progress["rejected_by_reason"] is None
        assert progress["keyframes_accepted_source"] == "event journal"
    finally:
        engine.stop_session()


def test_frames_observed_appears_once_the_session_stops(monkeypatch, tmp_path):
    root = tmp_path / "worlds"
    world_id, session_id, engine = start_live_world(root, frames=10)
    summary = engine.stop_session()

    payload = _payload(monkeypatch, root)
    progress = payload["progress"]

    assert progress["frames_observed"] == summary.frames_observed
    assert progress["keyframes_accepted"] == summary.keyframes_accepted
    assert progress["frames_observed_unavailable_reason"] is None


# -- lifecycle ----------------------------------------------------------


def test_a_live_session_is_receiving_on_the_evidence_of_the_lock(
    monkeypatch, tmp_path
):
    """The signal iOS 1.1 asks for: distinct from "frames are arriving".

    The writer lock is held for the lifetime of a mapping session and by
    nothing else, so it answers exactly that question.
    """
    root = tmp_path / "worlds"
    world_id, _, engine = start_live_world(root, frames=6)
    try:
        payload = _payload(monkeypatch, root)
        assert payload["lifecycle"]["state"] == "receiving"
        assert "writer lock" in payload["lifecycle"]["evidence"]
        assert payload["lifecycle"]["build_in_progress"] is False
    finally:
        engine.stop_session()


def test_a_dead_builder_is_reported_as_interrupted_not_as_receiving(
    monkeypatch, tmp_path
):
    """A stale lock is a real, visible failure and must not read as health.

    Reporting `receiving` forever would be a stale observation presented
    as current state. Since 2026-09-06 the word is `interrupted` rather
    than `failed`: the session did not end properly, and the geometry
    block beside it says whether anything was built (on the physical walk
    that made this visible, 463 keyframes were).
    """
    root = tmp_path / "worlds"
    world_id, _, engine = start_live_world(root, frames=6)
    try:
        store = WorldStore(root)
        # A pid that is genuinely not running, found rather than assumed:
        # on Windows pid 0 IS alive (the System Idle Process), so the
        # obvious choice would silently test nothing.
        import psutil

        dead = next(
            pid for pid in range(100_000, 200_000) if not psutil.pid_exists(pid)
        )
        store.lock_path(world_id).write_text(
            json.dumps({"pid": dead}), encoding="utf-8"
        )

        payload = _payload(monkeypatch, root)
        assert payload["lifecycle"]["state"] == "interrupted"
        assert "no longer running" in payload["lifecycle"]["evidence"]
        assert payload["model_state"] == "interrupted"
    finally:
        engine.stop_session()


def test_a_stopped_unbuilt_session_never_claims_a_build_is_running(
    monkeypatch, tmp_path
):
    """`finalizing` would assert something the Tower cannot observe.

    A build does rewrite files before its manifest lands -- an adversarial
    review disproved an earlier "byte-identical" claim here -- but those
    writes are indistinguishable from a build that made them and then
    died. So the state is named for what is visible, and
    `build_in_progress` is NULL, not false, which would be a claim that no
    build is running.
    """
    root = tmp_path / "worlds"
    world_id, session_id, engine = start_live_world(root, frames=8)
    engine.stop_session()

    payload = _payload(monkeypatch, root)
    lifecycle = payload["lifecycle"]

    assert lifecycle["state"] == "stopped_unbuilt"
    assert lifecycle["build_in_progress"] is None
    assert "indistinguishable" in lifecycle["build_in_progress_unavailable_reason"]
    assert payload["geometry"]["available"] is False


def test_a_built_session_is_ready(monkeypatch, built):
    root, _, _ = built
    payload = _payload(monkeypatch, root)
    assert payload["lifecycle"]["state"] == "ready"


# -- geometry and trajectory -------------------------------------------


def test_geometry_matches_the_manifest_the_build_wrote(monkeypatch, built):
    """Independent truth: the manifest on disk, read directly."""
    root, world_id, session_id = built
    manifest = WorldStore(root).read_derived_manifest(world_id)

    payload = _payload(monkeypatch, root)
    geometry = payload["geometry"]

    assert geometry["available"] is True
    assert geometry["element_count"] == manifest["points"]
    assert geometry["backend_id"] == manifest["backend_id"]
    assert geometry["representation"] == "sparse point cloud"
    assert geometry["is_incremental"] is False
    assert geometry["provenance"] == "inferred"


def test_pose_count_is_not_poses_solved_and_not_the_keyframe_count(
    monkeypatch, built
):
    """An anchor has a position and is counted as neither solved nor refused.

    engine.build increments poses_solved only for SOLVED, so reporting it
    as "the number of poses" drops the first keyframe of every segment.
    Reporting the keyframe count instead would claim a position for
    keyframes the backend refused. The trajectory is neither.
    """
    root, world_id, _ = built
    manifest = WorldStore(root).read_derived_manifest(world_id)

    payload = _payload(monkeypatch, root)
    trajectory = payload["trajectory"]

    assert trajectory["pose_count"] == manifest["keyframes"] - manifest["poses_refused"]
    assert trajectory["poses_solved"] == manifest["poses_solved"]
    assert trajectory["pose_count"] >= trajectory["poses_solved"]


def test_no_pose_array_is_sent(monkeypatch, built):
    """IOS-to-Tower.md 1.4 marks a pose array NOT REQUESTED.

    A pose schema needs position, rotation convention, handedness, frame
    and units -- five Tower decisions, each of which renders plausibly and
    wrongly if guessed. Sending a summary is the whole point.
    """
    root, _, _ = built
    payload = _payload(monkeypatch, root)
    encoded = json.dumps(payload)

    # Pose DATA, not the word "pose": `poses_solved` is a count and is
    # exactly what iOS asked for instead of an array.
    for banned in ("translation", "rotation", "quaternion", "wxyz", "xyz"):
        assert banned not in encoded, f"the payload leaked {banned!r}"

    def _no_numeric_arrays(node, path="payload"):
        if isinstance(node, list):
            assert not any(
                isinstance(item, (int, float, list)) for item in node
            ), f"{path} looks like coordinate data"
        elif isinstance(node, dict):
            for key, value in node.items():
                _no_numeric_arrays(value, f"{path}.{key}")

    _no_numeric_arrays(payload)


def test_a_path_length_carries_its_unit_and_scale_semantics(monkeypatch, built):
    """IOS-to-Tower.md 0.5 and 1.5: never a bare number, never metres."""
    root, _, _ = built
    payload = _payload(monkeypatch, root)
    path = payload["trajectory"]["path_length"]

    assert path["available"] is True
    assert path["unit"] == "world units"
    assert path["scale_semantics"] == "relative"
    assert path["provenance"] == "inferred"
    assert "m" != path["display"][-1], "a relative length must not render as metres"
    assert "world units" in path["display"]


def test_a_path_length_is_refused_when_poses_have_gaps(monkeypatch, tmp_path):
    """A refused pose is a HOLE in the path, not a shorter path.

    Summing across it draws a straight line between the keyframes either
    side and calls that distance walked.
    """
    root = tmp_path / "worlds"
    # No intrinsics -> the backend refuses poses, so gaps are guaranteed.
    from tests import synthetic_scene as ss
    from tower.world_builder.engine import WorldBuilderEngine

    engine = WorldBuilderEngine(WorldStore(root))
    world_id = engine.create_world("Uncalibrated")
    session_id = engine.start_session(world_id, frame_source="synthetic")
    scene = ss.furnished_room()
    matrix = ss.camera_matrix(480, 360)
    for index, image in enumerate(
        ss.render_sequence(scene, ss.strafe(8, step=0.09), matrix, 480, 360)
    ):
        engine.observe(ss.encode_jpeg(image), source_seq=index)
    engine.stop_session()
    engine.build(world_id, session_id)

    payload = _payload(monkeypatch, root)
    trajectory = payload["trajectory"]
    if trajectory["available"]:
        path = trajectory["path_length"]
        assert path["available"] is False
        assert "reason" in path


def test_geometry_built_mid_session_is_reported_as_real_but_behind(
    monkeypatch, tmp_path
):
    """The "watch it build" case, and the reason it needed fixing.

    With --rebuild-every, a build finishes and the very next keyframe
    makes its output stale. An earlier version reported anything not
    matching the current keyframes as simply unavailable, so a walk that
    was genuinely producing geometry every few keyframes reported NONE AT
    ALL until it stopped -- the channel hid the exact thing
    --rebuild-every exists to show.

    A build over the first N keyframes is a correct answer to an older
    question, not a wrong answer. It is reported, with `current: false`
    and both counts, so a viewer can show real progress while knowing
    exactly how far behind it is.
    """
    from tests import synthetic_scene as ss
    from tower.world_builder.engine import WorldBuilderEngine
    from tower.world_builder.records import CameraIntrinsics

    root = tmp_path / "worlds"
    matrix = ss.camera_matrix(480, 360)
    engine = WorldBuilderEngine(WorldStore(root))
    world_id = engine.create_world("Growing")
    session_id = engine.start_session(
        world_id,
        intrinsics=CameraIntrinsics(
            source="self_calibrated", model="pinhole",
            fx=float(matrix[0, 0]), fy=float(matrix[1, 1]),
            cx=float(matrix[0, 2]), cy=float(matrix[1, 2]),
            calibrated_width=480, calibrated_height=360,
        ),
        frame_source="synthetic",
        declared_size=(480, 360),
    )
    images = ss.render_sequence(
        ss.furnished_room(), ss.strafe(14, step=0.09), matrix, 480, 360
    )
    accepted = 0
    for index, image in enumerate(images):
        outcome = engine.observe(ss.encode_jpeg(image), source_seq=index)
        if outcome.keyframe_id is not None:
            accepted += 1
            if accepted == 3:
                engine.build(world_id, session_id)
                built_at_keyframes = accepted
    try:
        assert accepted > built_at_keyframes, (
            "precondition: keyframes must have been accepted AFTER the build"
        )
        payload = _payload(monkeypatch, root)
        geometry = payload["geometry"]

        assert geometry["available"] is True, "real geometry must not be hidden"
        assert geometry["current"] is False
        assert geometry["element_count"] > 0
        assert geometry["built_from_keyframes"] == built_at_keyframes
        assert geometry["keyframes_now"] == accepted
        assert geometry["keyframes_now"] > geometry["built_from_keyframes"]
        assert "not the final world" in geometry["stale_reason"]
    finally:
        engine.stop_session()


def test_geometry_and_trajectory_keys_do_not_change_shape(monkeypatch, built):
    """A strict decoder must not choke when a field becomes unavailable.

    Every branch of these two blocks emits the SAME key set; only values
    change. A key that appeared and disappeared would force every consumer
    into optional-chaining for reasons it could not see.
    """
    root, _, _ = built
    available = _payload(monkeypatch, root)
    absent = _payload(monkeypatch, root, world_id="not-a-real-world")

    assert absent["geometry"] is None, "an unresolvable target sends nulls"

    from tower.results.world_builder import (
        _geometry_block,
        _geometry_unavailable,
        _trajectory_unavailable,
    )

    manifest = WorldStore(root).read_derived_manifest(built[1])
    assert set(_geometry_block(manifest, True, 4)) == set(
        _geometry_unavailable("x")
    )
    assert set(available["geometry"]) == set(_geometry_unavailable("x"))
    assert set(available["trajectory"]) == set(_trajectory_unavailable("x"))


# -- scale, calibration, tracking --------------------------------------


def test_scale_is_relative_and_never_licenses_metres(monkeypatch, built):
    """`measured` is unreachable in V1, so no figure may ever be metric."""
    root, _, _ = built
    payload = _payload(monkeypatch, root)
    scale = payload["scale"]

    assert scale["state"] == "relative"
    assert scale["semantics"] == "relative"
    assert scale["allows_metres"] is False
    assert scale["meters_per_unit"] is None
    assert scale["unit"] == "world units"


def test_an_uncalibrated_session_says_so(monkeypatch, tmp_path):
    """Mapped on is_known, not on `source` alone.

    Intrinsics that are present but physically impossible report
    is_known False while source says self_calibrated; mapping on source
    would render a broken calibration as `calibrated`.
    """
    root = tmp_path / "worlds"
    from tower.world_builder.engine import WorldBuilderEngine

    engine = WorldBuilderEngine(WorldStore(root))
    world_id = engine.create_world("No calibration")
    engine.start_session(world_id, frame_source="synthetic")
    try:
        payload = _payload(monkeypatch, root)
        calibration = payload["calibration"]
        assert calibration["state"] == "uncalibrated"
        assert calibration["scope"] == "session"
        assert calibration["calibrating_ever_reported"] is False
    finally:
        engine.stop_session()


def test_a_calibrated_session_says_so(monkeypatch, built):
    root, _, _ = built
    payload = _payload(monkeypatch, root)
    assert payload["calibration"]["state"] == "calibrated"


def test_tracking_never_reports_limited(monkeypatch, built):
    """`limited` would need a threshold nobody has defined.

    The nearest candidate in the code is documented as an untuned
    placeholder, and is not emitted as an event at all -- so it is not
    even available live.
    """
    root, _, _ = built
    payload = _payload(monkeypatch, root)
    tracking = payload["tracking"]

    assert tracking["state"] in ("good", "lost", "unknown")
    assert tracking["state"] != "limited"
    assert tracking["limited_ever_reported"] is False


# -- imagery and privacy ------------------------------------------------


def test_keyframe_imagery_is_reported_present_but_never_fetchable(
    monkeypatch, built
):
    """IOS-to-Tower.md 5: an unstated treatment is not a treatment.

    World Builder keyframes are written with redaction "none" -- raw
    first-person frames. So they are declared, and declared unfetchable,
    and no id or URL is minted. iOS holds no id format, and inventing one
    would be the fabricated contract that document refuses.
    """
    root, _, _ = built
    payload = _payload(monkeypatch, root)
    images = payload["artifacts"]["keyframe_images"]

    assert images["present"] is True
    assert images["count"] > 0
    # What the SESSION recorded, not a constant. Keyframes are now
    # face-redacted before they are written, and a hardcoded "none" here
    # survived that change for exactly as long as it took someone to look.
    assert images["redaction"] == "faces-detected-and-filled/yunet-2023mar@0.30"
    assert images["fetchable"] is False
    assert "id" not in images and "url" not in images
    # Unfetchable REGARDLESS. A best-effort filter with measured false
    # negatives is not grounds to start shipping first-person imagery.
    assert "withhold imagery it cannot verify" in images["reason"]


def test_no_filesystem_path_reaches_the_client(monkeypatch, built):
    """A Tower path is useless to a phone and names a machine's layout."""
    root, _, _ = built
    payload = _payload(monkeypatch, root)
    encoded = json.dumps(payload)

    assert str(root) not in encoded
    assert "C:\\\\" not in encoded and "/tmp/" not in encoded
    assert payload["persistence"]["location_disclosed"] is False


def test_images_purged_is_reported_as_a_declaration_not_a_deletion(
    monkeypatch, built
):
    """The flag deletes nothing; it makes rebuilds refuse.

    Reporting it as "the imagery is gone" would be the false assurance
    06-PRIVACY-DATA forbids.
    """
    root, _, _ = built
    payload = _payload(monkeypatch, root)
    artifacts = payload["artifacts"]

    assert artifacts["images_purged_declared"] is False
    assert artifacts["images_purged_verified"] is None
    assert "not a verified deletion" in artifacts["images_purged_meaning"]


# -- unavailable stays unavailable -------------------------------------


def test_an_absent_world_is_unavailable_with_a_reason(monkeypatch, tmp_path):
    payload = _payload(monkeypatch, tmp_path / "empty")
    assert payload["lifecycle"]["state"] == "unavailable"
    assert payload["world"] is None
    assert payload["geometry"] is None
    assert payload["progress"] is None


def test_naming_a_world_that_does_not_exist_is_unavailable_not_a_guess(
    monkeypatch, built
):
    """Inspection mode must not silently fall back to some other world."""
    root, world_id, _ = built
    payload = _payload(monkeypatch, root, world_id="not-a-real-world")

    assert payload["lifecycle"]["state"] == "unavailable"
    assert "not-a-real-world" in payload["lifecycle"]["reason"]


def test_a_world_with_no_geometry_says_unavailable_never_zero(
    monkeypatch, tmp_path
):
    """"we never built" and "the build found nothing" must not both be 0."""
    root = tmp_path / "worlds"
    world_id, session_id, engine = start_live_world(root, frames=6)
    engine.stop_session()

    payload = _payload(monkeypatch, root)
    geometry = payload["geometry"]

    assert geometry["available"] is False
    assert geometry["element_count"] is None, "a missing count must not be zero"
    assert geometry["representation"] is None
    assert geometry["unavailable_reason"]


def test_geometry_from_another_session_is_not_attributed_to_this_one(
    monkeypatch, tmp_path
):
    """The manifest is per-WORLD but describes one session.

    A world with two built sessions has one manifest, describing whichever
    built last. Attributing it to the other session would report one
    session's geometry as another's.
    """
    root = tmp_path / "worlds"
    world_id, first_session = build_world(root, frames=8, name="Two sessions")

    # A second session in the SAME world, built after the first.
    from tests import synthetic_scene as ss
    from tower.world_builder.engine import WorldBuilderEngine
    from tower.world_builder.records import CameraIntrinsics

    matrix = ss.camera_matrix(480, 360)
    engine = WorldBuilderEngine(WorldStore(root))
    second_session = engine.start_session(
        world_id,
        intrinsics=CameraIntrinsics(
            source="self_calibrated",
            model="pinhole",
            fx=float(matrix[0, 0]),
            fy=float(matrix[1, 1]),
            cx=float(matrix[0, 2]),
            cy=float(matrix[1, 2]),
            calibrated_width=480,
            calibrated_height=360,
        ),
        frame_source="synthetic",
        declared_size=(480, 360),
    )
    for index, image in enumerate(
        ss.render_sequence(ss.furnished_room(), ss.strafe(8, step=0.09), matrix, 480, 360)
    ):
        engine.observe(ss.encode_jpeg(image), source_seq=index)
    engine.stop_session()
    engine.build(world_id, second_session)

    store = WorldStore(root)
    manifest = store.read_derived_manifest(world_id)
    assert manifest["session_id"] == second_session, "precondition"

    first = _payload(monkeypatch, root, world_id=world_id, session_id=first_session)
    second = _payload(monkeypatch, root, world_id=world_id, session_id=second_session)

    # EACH SESSION'S OWN FIGURES, which is a stronger statement than the
    # one this test used to make. It asserted that the first session
    # reported NO geometry -- true at the time, and only because the
    # producer had nothing to read for it. That absence was itself the
    # defect: an older walk reported no geometry, no poses and no currency
    # over a reconstruction on disk, and the phone drew a red "Needs retry"
    # on it. `write_derived` writes a manifest beside each session's own
    # poses and points now, so the question this test asks -- is one
    # session's geometry reported as another's -- can be asked properly.
    own = {
        session: store.read_session_manifest(world_id, session)
        for session in (first_session, second_session)
    }
    assert own[first_session]["session_id"] == first_session
    assert own[second_session]["session_id"] == second_session

    assert first["geometry"]["available"] is True
    assert first["geometry"]["element_count"] == own[first_session]["points"]
    assert second["geometry"]["available"] is True
    assert second["geometry"]["element_count"] == own[second_session]["points"]
    assert second["geometry"]["element_count"] == manifest["points"]

    # And the trajectory beside it, which used to say "no build has run for
    # this session" over the first session's poses.
    assert first["trajectory"]["available"] is True
    # A count, not a formula: `pose_count` includes segment anchors, so it
    # is not `poses_solved`. What matters here is that it is a real number
    # for a session the producer used to have nothing to say about.
    assert isinstance(first["trajectory"]["pose_count"], int)
    assert first["trajectory"]["pose_count"] > 0


# -- the producer itself ------------------------------------------------


def test_the_producer_never_reads_an_unchanged_file_twice(built):
    """Stat-gating, which is a measured necessity rather than an optimisation.

    read_events parses the WHOLE journal on every call -- 26.6 ms at 10k
    events against a 1.98 ms average frame reply -- so a poll loop that
    re-read an unchanged file would spend more time parsing than the Tower
    spends answering frames.
    """
    root, world_id, session_id = built
    import tower.results.world_builder as producer_module

    calls = {"journal": 0, "keyframes": 0}
    original_raw = producer_module.read_raw_jsonl
    original_keyframes = WorldStore.read_keyframes

    def _raw(path, *args, **kwargs):
        calls["journal"] += 1
        return original_raw(path, *args, **kwargs)

    def _keyframes(self, *args, **kwargs):
        calls["keyframes"] += 1
        return original_keyframes(self, *args, **kwargs)

    producer_module.read_raw_jsonl = _raw
    WorldStore.read_keyframes = _keyframes
    try:
        producer = WorldBuilderStatusProducer(root, lambda: 0.0)
        producer.snapshot(world_id, session_id)
        first_pass = dict(calls)
        calls["journal"] = calls["keyframes"] = 0
        for _ in range(5):
            producer.snapshot(world_id, session_id)
        steady_state = dict(calls)
    finally:
        producer_module.read_raw_jsonl = original_raw
        WorldStore.read_keyframes = original_keyframes

    assert first_pass["journal"] == 1
    # Two on the first pass: once for the keyframe digest, and once inside
    # read_derived's own staleness check while the path length is
    # computed. Both are gated afterwards -- what matters is that five
    # further passes over an unchanged world parse nothing at all.
    assert first_pass["keyframes"] == 2
    assert steady_state == {"journal": 0, "keyframes": 0}, (
        f"an unchanged world was re-parsed: {steady_state}"
    )


# -- the iOS projection -------------------------------------------------
#
# `handoff.md` documents the Swift that exists today. These pin the one
# contract shape it says costs the phone nothing: fields mapping 1:1 onto
# `WorldSnapshot`, plus an explicit `WorldModelState`.


IOS_MODEL_STATES = {
    "unsupported",
    "idle",
    "awaiting_first_update",
    "receiving",
    "finalizing",
    "finalized",
    "interrupted",
    "failed",
}
IOS_TRACKING = {"good", "limited", "lost", "unavailable"}
IOS_SCALE = {"relative", "inferredMetric", "measuredMetric", "unknown"}
IOS_CALIBRATION = {"unknown", "uncalibrated", "calibrating", "calibrated"}
IOS_PERSISTENCE = {"unknown", "session", "saved", "reloading"}


def test_the_projection_uses_only_vocabulary_ios_implements(monkeypatch, built):
    """Every enum value must be a case iOS already has.

    A value outside these sets is a value the phone decodes into nothing,
    and iOS's decoder is required to fail rather than downgrade -- so an
    unknown word is a blank screen, not a degraded one.
    """
    root, _, _ = built
    payload = _payload(monkeypatch, root)
    snapshot = payload["world_snapshot"]

    assert payload["model_state"] in IOS_MODEL_STATES
    assert snapshot["tracking"] in IOS_TRACKING
    assert snapshot["scale"] in IOS_SCALE
    assert snapshot["calibration"] in IOS_CALIBRATION
    assert snapshot["persistence"]["state"] in IOS_PERSISTENCE
    assert snapshot["trajectory"]["scale"] in IOS_SCALE


def test_the_projection_has_exactly_the_worldsnapshot_fields(monkeypatch, built):
    """1:1 with handoff.md 8.3, so iOS decodes one flat shape."""
    root, _, _ = built
    snapshot = _payload(monkeypatch, root)["world_snapshot"]

    assert set(snapshot) == {
        "name",
        "world_id",
        "keyframe_count",
        "revision",
        "tracking",
        "scale",
        "mapping_seconds",
        "calibration",
        "geometry",
        "trajectory",
        "persistence",
    }
    assert set(snapshot["geometry"]) == {
        "representation",
        "element_count",
        "is_incremental",
    }
    assert set(snapshot["trajectory"]) == {
        "pose_count",
        "path_length",
        "path_length_unit",
        "scale",
    }
    assert set(snapshot["persistence"]) == {"state", "revision"}


def test_the_projection_never_disagrees_with_the_evidence(monkeypatch, built):
    """It is a projection, not a second source of truth.

    Asserted against the Tower-native blocks it was derived from, so the
    two cannot drift into disagreeing about the same world.
    """
    root, _, _ = built
    payload = _payload(monkeypatch, root)
    snapshot = payload["world_snapshot"]

    assert snapshot["world_id"] == payload["world"]["world_id"]
    assert snapshot["name"] == payload["world"]["display_name"]
    assert snapshot["keyframe_count"] == payload["progress"]["keyframes_accepted"]
    assert snapshot["mapping_seconds"] == payload["progress"]["mapping_seconds"]
    assert snapshot["scale"] == payload["scale"]["semantics"]
    assert snapshot["calibration"] == payload["calibration"]["state"]
    assert (
        snapshot["geometry"]["element_count"] == payload["geometry"]["element_count"]
    )
    assert snapshot["trajectory"]["pose_count"] == payload["trajectory"]["pose_count"]
    assert (
        snapshot["trajectory"]["path_length"]
        == payload["trajectory"]["path_length"]["value"]
    )


def test_the_snapshot_revision_is_the_envelope_revision(monkeypatch, built):
    """iOS holds the revision inside the snapshot; it must be the same one."""
    root, _, _ = built
    client = make_client(monkeypatch, root)
    with client.websocket_connect("/ws") as ws:
        subscribe(ws)
        envelope = drain(ws, expect="cartridge_result")

    assert envelope["payload"]["world_snapshot"]["revision"] == envelope["revision"]


def test_a_built_world_projects_to_finalized(monkeypatch, built):
    root, _, _ = built
    assert _payload(monkeypatch, root)["model_state"] == "finalized"


def test_a_live_session_projects_to_receiving(monkeypatch, tmp_path):
    root = tmp_path / "worlds"
    _, _, engine = start_live_world(root, frames=6)
    try:
        payload = _payload(monkeypatch, root)
        assert payload["model_state"] == "receiving"
        assert payload["world_snapshot"]["keyframe_count"] > 0
    finally:
        engine.stop_session()


def test_a_stopped_unbuilt_session_does_not_ask_the_wearer_to_wait(
    monkeypatch, tmp_path
):
    """`.finalizing` means "wait". Nothing here is coming.

    This asserted `finalizing`, on the reasoning that Tower cannot see
    whether a build is running so "figures may still change" is the honest
    reading. That reasoning was true when it was written and this campaign
    made it false: `stop_session(hold_lock=True)` holds the writer lock
    through finalization, so a build in progress IS visible -- as a live
    lock, which `_lifecycle` answers three branches earlier as
    `finalizing`. Every state that reaches `stopped_unbuilt` has already
    been shown to have no live holder.

    Four review rounds found the consequence by four different routes --
    a permanent "Finalizing" over a world nothing would ever touch again,
    which this campaign's own iOS note now renders as "it usually takes a
    few minutes... worth waiting for Saved". Three of them were answered
    with another branch in `_lifecycle`; the fourth found a route the
    branches still missed. The mapping was the wrong level to keep
    patching around.
    """
    root = tmp_path / "worlds"
    _, _, engine = start_live_world(root, frames=8)
    engine.stop_session()

    payload = _payload(monkeypatch, root)
    assert payload["model_state"] == "interrupted", (
        "a stopped session with nothing to open and nobody working on it "
        "told the wearer to keep waiting"
    )
    assert payload["lifecycle"]["state"] == "stopped_unbuilt"
    assert payload["lifecycle"]["build_in_progress"] is None


def test_a_build_that_really_is_running_still_says_finalizing(
    monkeypatch, tmp_path
):
    """The other half, and the reason the mapping above could change.

    "Wait" is right when something is actually working, and that state is
    distinguishable on disk: the builder holds the writer lock through
    finalization. If that ever stops being true, this test fails and the
    mapping above has to be reconsidered rather than trusted.
    """
    import os

    from tower.world_builder.store import WorldStore

    root = tmp_path / "worlds"
    world_id, _, engine = start_live_world(root, frames=8)
    engine.stop_session()

    # A live holder: this process, which is by definition running.
    store = WorldStore(root)
    store.acquire_writer_lock(world_id)
    try:
        holder = store.lock_holder(world_id)
        assert holder is not None and holder["pid"] == os.getpid()
        payload = _payload(monkeypatch, root)
    finally:
        store.release_writer_lock(world_id)

    assert payload["model_state"] == "finalizing", (
        "a build holding the writer lock was not reported as working"
    )
    assert payload["lifecycle"]["state"] == "finalizing"


def test_no_world_root_projects_to_unsupported_not_idle(monkeypatch):
    """A Tower that cannot serve this at all must not look merely empty.

    `.idle` invites a person to wait for something that is never coming;
    `.unsupported` tells them the Tower cannot do it.
    """
    client = make_client(monkeypatch, None)
    with client.websocket_connect("/ws") as ws:
        reply = subscribe(ws)
    assert reply["type"] == "result_error"
    assert reply["reason"] == "cartridge_unavailable"


def test_an_empty_world_root_projects_to_idle(monkeypatch, tmp_path):
    payload = _payload(monkeypatch, tmp_path / "empty")
    assert payload["model_state"] == "idle"
    assert payload["world_snapshot"] is None


def test_the_projection_never_offers_a_metric_scale(monkeypatch, built):
    """measuredMetric from a monocular pipeline would make the app lie."""
    root, _, _ = built
    payload = _payload(monkeypatch, root)
    assert payload["world_snapshot"]["scale"] != "measuredMetric"
    assert payload["world_snapshot"]["trajectory"]["scale"] != "measuredMetric"


def test_a_spatial_figure_never_arrives_without_its_unit_and_scale(
    monkeypatch, built
):
    """handoff.md 9.6: send scale and unit TOGETHER with any spatial figure."""
    root, _, _ = built
    trajectory = _payload(monkeypatch, root)["world_snapshot"]["trajectory"]

    if trajectory["path_length"] is not None:
        assert trajectory["path_length_unit"] is not None
        assert trajectory["scale"] in IOS_SCALE
        assert trajectory["scale"] != "unknown"
    else:
        assert trajectory["path_length_unit"] is None


# -- reopening a saved world --------------------------------------------


def test_a_saved_world_can_be_reopened_by_id(monkeypatch, tmp_path):
    """The Tower half of iOS's `WorldInspectionMode.inspecting(worldID:)`.

    `handoff.md` 9.7 says that mode exists on iOS but nothing can change
    it -- there is no UI and no client method. The Tower side is here and
    works: name a world (and optionally a session) on subscribe and the
    channel reports that one, not whatever is live.
    """
    root = tmp_path / "worlds"
    first_world, first_session = build_world(root, frames=10, name="First")
    second_world, _ = build_world(root, frames=8, name="Second")
    assert first_world != second_world

    # With no id, the newest world is followed.
    live = _payload(monkeypatch, root)
    assert live["world"]["world_id"] == second_world

    # Named explicitly, the older world is reported instead -- complete,
    # with its own geometry, unaffected by the newer one existing.
    reopened = _payload(monkeypatch, root, world_id=first_world)
    assert reopened["world"]["world_id"] == first_world
    assert reopened["world"]["display_name"] == "First"
    assert reopened["model_state"] == "finalized"
    assert reopened["world_snapshot"]["world_id"] == first_world
    assert reopened["session"]["session_id"] == first_session


def test_reopening_a_specific_session_reports_that_session(monkeypatch, tmp_path):
    root = tmp_path / "worlds"
    world_id, first_session = build_world(root, frames=10, name="Two sessions")

    from tests import synthetic_scene as ss
    from tower.world_builder.engine import WorldBuilderEngine
    from tower.world_builder.records import CameraIntrinsics

    matrix = ss.camera_matrix(480, 360)
    engine = WorldBuilderEngine(WorldStore(root))
    second_session = engine.start_session(
        world_id,
        intrinsics=CameraIntrinsics(
            source="self_calibrated",
            model="pinhole",
            fx=float(matrix[0, 0]),
            fy=float(matrix[1, 1]),
            cx=float(matrix[0, 2]),
            cy=float(matrix[1, 2]),
            calibrated_width=480,
            calibrated_height=360,
        ),
        frame_source="synthetic",
        declared_size=(480, 360),
    )
    for index, image in enumerate(
        ss.render_sequence(
            ss.furnished_room(), ss.strafe(8, step=0.09), matrix, 480, 360
        )
    ):
        engine.observe(ss.encode_jpeg(image), source_seq=index)
    engine.stop_session()

    for session_id in (first_session, second_session):
        payload = _payload(monkeypatch, root, world_id=world_id, session_id=session_id)
        assert payload["session"]["session_id"] == session_id


def test_a_reopened_world_carries_the_replay_data_it_has(monkeypatch, built):
    """What "replay" can honestly mean today, checked against the store.

    Tower keeps, per keyframe, the pose and the PATH TO THE ACTUAL IMAGE
    the glasses saw there -- which is a recorded camera path with a real
    first-person view at every point on it.

    None of it crosses this wire, deliberately. `handoff.md` 9.5 and 14:
    iOS holds summary figures, has no pose schema, and links no 3D
    framework, so a pose array "cannot be displayed and would be dropped".
    The channel reports the SUMMARY, and the replay data stays on the
    Tower where something can actually read it.
    """
    root, world_id, session_id = built
    from tower.world_builder.inspect import open_world

    trajectory = open_world(root, world_id).trajectory(session_id)
    assert trajectory, "precondition: the session has keyframes"
    for row in trajectory:
        assert row["image_relpath"], "a path point with no observed frame"
        assert (
            WorldStore(root).session_dir(world_id, session_id) / row["image_relpath"]
        ).exists()

    payload = _payload(monkeypatch, root)
    assert payload["trajectory"]["pose_count"] is not None
    # ...and no pose data on the wire.
    import json as _json

    encoded = _json.dumps(payload)
    assert "image_relpath" not in encoded
    assert "translation" not in encoded


def test_a_world_that_is_merely_behind_still_says_finalizing(monkeypatch, tmp_path):
    """`stopped_unbuilt` carries two states and only one of them means wait.

    "Built, and behind" is a world that is intact and needs a rebuild:
    "Finalizing" is right. "Nothing was built" is a walk that produced no
    geometry, and telling a wearer to wait for that is the permanent
    "Finalizing" four separate reviews found by four separate routes.

    A previous round fixed the second by pointing the whole state at
    `interrupted`, and a reviewer built the first and watched a complete
    world start rendering a red "Interrupted ... what was built before it
    stopped is here". Both halves are asserted here so neither can be
    fixed at the other's expense again.
    """
    from tower.world_builder.records import Keyframe

    root = tmp_path / "worlds"
    world_id, session_id, engine = start_live_world(root, frames=8)
    engine.build(world_id, session_id)
    engine.stop_session()

    # BEHIND: a keyframe the build never saw.
    store = WorldStore(root)
    store.append_keyframe(world_id, Keyframe(
        keyframe_id=f"{session_id}:behind", session_id=session_id,
        source_seq=9999, received_at=9999.0, image_relpath="images/x.jpg",
        width=8, height=8, byte_count=9,
    ))

    behind = _payload(monkeypatch, root)
    assert behind["lifecycle"]["state"] == "stopped_unbuilt"
    assert behind["geometry"]["available"] is True
    assert behind["geometry"]["current"] is False
    assert behind["model_state"] == "finalizing", (
        "a complete world that merely needs a rebuild was reported as an "
        "interruption"
    )


def test_a_walk_that_built_nothing_does_not_say_finalizing(monkeypatch, tmp_path):
    """The other half: nothing to wait for, so do not say wait."""
    root = tmp_path / "worlds"
    _, _, engine = start_live_world(root, frames=8)
    engine.stop_session()

    payload = _payload(monkeypatch, root)
    assert payload["lifecycle"]["state"] == "stopped_unbuilt"
    assert payload["geometry"]["available"] is False
    assert payload["model_state"] == "interrupted", (
        "a walk with no geometry told the wearer to keep waiting"
    )


# -- round 16: the figures, not the flag -------------------------------


def _featureless(root, *, frames=8):
    """A walk with nothing in it to detect, match or solve.

    Uniform grey. A blank wall, a dark corridor, a lens cap, a
    calibration that never arrived. `engine.build` runs, solves nothing,
    and writes `poses.json`, `points.json` and a manifest saying
    `points: 0, poses_solved: 0` -- because `write_derived` is
    unconditional. That combination is the whole point of these tests.
    """
    import numpy as np

    from tests import synthetic_scene as ss
    from tower.world_builder.engine import WorldBuilderEngine
    from tower.world_builder.records import CameraIntrinsics

    width, height = 480, 360
    camera_matrix = ss.camera_matrix(width, height)
    engine = WorldBuilderEngine(WorldStore(root))
    world_id = engine.create_world("Dark Hallway")
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
    blank = np.full((height, width, 3), 128, dtype=np.uint8)
    for index in range(frames):
        engine.observe(ss.encode_jpeg(blank), source_seq=index, wire_seq=index)
    engine.build(world_id, session_id)
    engine.stop_session()
    return world_id, session_id


def _behind(root, world_id, session_id):
    """One keyframe the build never saw, so the build is stale."""
    from tower.world_builder.records import Keyframe

    WorldStore(root).append_keyframe(world_id, Keyframe(
        keyframe_id=f"{session_id}:behind", session_id=session_id,
        source_seq=9999, received_at=9999.0, image_relpath="images/x.jpg",
        width=8, height=8, byte_count=9,
    ))


def test_a_build_that_solved_nothing_is_not_something_to_wait_for(
    monkeypatch, tmp_path
):
    """`geometry.available` is the WRONG predicate, and this is why.

    The previous round split `stopped_unbuilt` on `geometry.available`,
    which is true as soon as a manifest and a derived tree exist. But
    `engine.build` calls `write_derived` unconditionally, so a walk that
    solved nothing writes a tree too -- and the split kept saying
    "finalizing" over a world with zero points and zero poses. iOS
    decides what to draw from the FIGURES
    (`WorldEvidence.hasGeometry`), so the wearer got a "Finalizing" that
    nothing would ever change: the permanent Finalizing, back through a
    different door one round after it was closed.
    """
    root = tmp_path / "worlds"
    world_id, session_id = _featureless(root)
    _behind(root, world_id, session_id)

    payload = _payload(monkeypatch, root)
    assert payload["lifecycle"]["state"] == "stopped_unbuilt"
    # The flag that used to decide this is TRUE here. That is the trap.
    assert payload["geometry"]["available"] is True
    assert payload["geometry"]["element_count"] == 0
    assert payload["trajectory"]["pose_count"] == 0
    assert payload["model_state"] == "interrupted", (
        "a walk that solved nothing told the wearer to keep waiting for it"
    )


def test_a_real_world_that_is_behind_is_still_something_to_wait_for(
    monkeypatch, tmp_path
):
    """The other half, pinned against the same predicate change."""
    root = tmp_path / "worlds"
    world_id, session_id = build_world(root, frames=10)
    _behind(root, world_id, session_id)

    payload = _payload(monkeypatch, root)
    assert payload["lifecycle"]["state"] == "stopped_unbuilt"
    assert payload["geometry"]["element_count"] > 0
    assert payload["model_state"] == "finalizing", (
        "a complete world that merely needs a rebuild was called an "
        "interruption"
    )


def test_a_world_with_no_manifest_still_reports_its_figures(
    monkeypatch, tmp_path
):
    """The defect this campaign is named for, reintroduced by this campaign.

    A derived tree with no manifest describing it was given
    `lifecycle: ready` and `geometry.available: false,
    element_count: null` -- and the branch's own comment promised "the
    world itself opens normally". It did not. iOS reads those two
    numbers, so a complete reconstruction rendered as "Needs retry:
    nothing usable came of this session. Walking the space again is what
    produces another one", over 1,347 points and 4 camera poses that the
    geometry route was serving 200 at the same moment.

    A manifest is a SUMMARY of these files. The files are still there.
    """
    root = tmp_path / "worlds"
    world_id, session_id = build_world(root, frames=10)
    store = WorldStore(root)
    derived = store.derived_dir(world_id)

    # Independent truth: the files themselves, read separately from the
    # code under test.
    on_disk_points = len(
        json.loads((derived / session_id / "points.json").read_text())["points"]
    )
    with_manifest = _payload(monkeypatch, root)
    assert on_disk_points > 0
    assert with_manifest["geometry"]["element_count"] == on_disk_points

    # LEGACY: no manifest names this session, from either copy.
    (derived / session_id / "manifest.json").unlink()
    (derived / "manifest.json").unlink()

    payload = _payload(monkeypatch, root)
    assert payload["lifecycle"]["state"] == "ready"
    assert payload["geometry"]["available"] is True
    assert payload["geometry"]["element_count"] == on_disk_points, (
        "the figures were not recovered from the files that hold them"
    )
    # And the RECOUNT AGREES WITH THE BUILD for this tree.
    #
    # NOT a check of the anchor rule, and an earlier version of this
    # comment claimed it was. This fixture is one segment with three
    # solved poses and one anchor, so "an anchor counts only in a segment
    # that solved" is never exercised: a reviewer mutated that rule away
    # and watched this test stay green.
    # `test_the_recount_agrees_with_a_manifest_written_by_hand`
    # (`test_world_builder_library.py`) owns that rule and catches the
    # mutation, because its fixture has a segment that resolved nothing.
    #
    # What this DOES check is that the two paths through the same payload
    # -- manifest present, manifest absent -- report the same figures for
    # the same disk, which is the property the recount exists to give.
    assert (
        payload["trajectory"]["pose_count"]
        == with_manifest["trajectory"]["pose_count"]
    )
    assert payload["trajectory"]["segments"] == with_manifest["trajectory"]["segments"]
    assert (
        payload["trajectory"]["poses_solved"]
        == with_manifest["trajectory"]["poses_solved"]
    )


def test_an_unjudgeable_build_does_not_claim_keyframes_arrived_after_it(
    monkeypatch, tmp_path
):
    """`current: false` has two meanings and one message was serving both.

    "Behind" is a build genuinely older than the keyframes. "No manifest"
    is a build whose age nothing here knows. Telling a wearer keyframes
    have been accepted since a build ran, when nothing knows when it ran,
    is a fabrication of exactly the family this campaign is about.
    """
    root = tmp_path / "worlds"
    world_id, session_id = build_world(root, frames=10)
    derived = WorldStore(root).derived_dir(world_id)
    (derived / session_id / "manifest.json").unlink()
    (derived / "manifest.json").unlink()

    payload = _payload(monkeypatch, root)
    assert payload["geometry"]["current"] is False
    for block in ("geometry", "trajectory"):
        stale = payload[block]["stale_reason"]
        assert stale is not None
        assert "have been accepted since this build ran" not in stale, (
            f"{block} claimed to know when a build with no manifest ran"
        )
        assert "counted from" in stale


def test_a_corrupt_manifest_is_not_reported_as_an_absent_one(
    monkeypatch, tmp_path
):
    """`_validate_manifest` refuses cleanly; the refusal was then laundered.

    Four corrupt shapes -- unreadable bytes, a top-level list, a schema
    version from the future, a required key set to null -- all reached a
    branch whose evidence said the world's manifest "names another
    session and this session has no copy of its own". Both files existed
    and both named this session. A reviewer built all four and got that
    sentence for every one.
    """
    root = tmp_path / "worlds"
    world_id, session_id = build_world(root, frames=10)
    derived = WorldStore(root).derived_dir(world_id)
    (derived / session_id / "manifest.json").write_bytes(b"\xff\xfe not json")
    (derived / "manifest.json").write_bytes(b"\xff\xfe not json")

    payload = _payload(monkeypatch, root)
    evidence = payload["lifecycle"]["evidence"]
    assert "no copy of its own" not in evidence, (
        "a manifest that is present and corrupt was described as absent"
    )
    assert "cannot be read" in evidence
    # And the world still opens: the files are fine, only the summary is not.
    assert payload["geometry"]["element_count"] > 0


def test_an_unreadable_tree_is_not_reported_as_an_empty_one(
    monkeypatch, tmp_path
):
    """Counting a tree that cannot be read must not produce zero.

    Zero is a claim about a build. "Could not read" is a claim about a
    file, and the two must not be spelled the same way -- a wearer told
    "this walk produced nothing" over an unreadable file will walk the
    space again for no reason.
    """
    root = tmp_path / "worlds"
    world_id, session_id = build_world(root, frames=10)
    derived = WorldStore(root).derived_dir(world_id)
    (derived / session_id / "manifest.json").unlink()
    (derived / "manifest.json").unlink()
    (derived / session_id / "points.json").write_bytes(b"\xff\xfe not json")

    payload = _payload(monkeypatch, root)
    assert payload["geometry"]["available"] is False
    assert payload["geometry"]["element_count"] is None
    reason = payload["geometry"]["unavailable_reason"]
    # THE EXACT SENTENCE, not a substring that several sentences share.
    # A reviewer showed the earlier `"read" in reason` could not tell this
    # case from a readable tree the producer had refused for another
    # reason -- both said "read" -- which is precisely the distinction
    # the test is named for.
    assert reason == (
        "this session has a derived tree and neither its poses nor its "
        "points could be read, so nothing here can summarise it; the "
        "geometry route reads the same files"
    ), reason
    # The two sentences this must NOT be: both claim something about the
    # BUILD, and nothing here knows anything about the build.
    assert "no geometry exists" not in reason
    assert "no build has run" not in reason


def test_a_wrong_sized_walk_is_visible_on_the_wire_while_it_happens(
    monkeypatch, tmp_path
):
    """Rejecting the frame was right; rejecting it silently was not.

    A reviewer drove seven of eight frames into `frame_size_changed` and
    found no trace of it anywhere a person looks: the live payload had
    no rejection count, the session record's tally is final-only, and
    the follower `continue`d past each one without a log line. A whole
    walk at the wrong rung read "Mapping" with a frozen keyframe count
    and then "Saved" with a truncated world -- where before the guard it
    at least read "Interrupted".

    The engine journals this rejection (an ordinary one writes no event),
    so the live channel can count it. It must, and it must do so while
    the session is still open.
    """
    import numpy as np

    from tests import synthetic_scene as ss

    root = tmp_path / "worlds"
    world_id, session_id, engine = start_live_world(root, frames=4)
    try:
        # Four more frames at another size: all rejected, all journaled.
        smaller = np.full((288, 384, 3), 128, dtype=np.uint8)
        for index in range(4, 8):
            outcome = engine.observe(
                ss.encode_jpeg(smaller), source_seq=index, wire_seq=index
            )
            assert outcome.keyframe_id is None

        payload = _payload(monkeypatch, root)
        assert payload["lifecycle"]["state"] == "receiving"
        assert payload["progress"]["frames_rejected_wrong_size"] == 4, payload["progress"]
        # The final-only tally is still honestly absent mid-session.
        assert payload["progress"]["rejected_by_reason"] is None
    finally:
        engine.stop_session()
