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

    manifest = WorldStore(root).read_derived_manifest(world_id)
    assert manifest["session_id"] == second_session, "precondition"

    first = _payload(monkeypatch, root, world_id=world_id, session_id=first_session)
    second = _payload(monkeypatch, root, world_id=world_id, session_id=second_session)

    assert first["geometry"]["available"] is False
    assert first["geometry"]["element_count"] is None
    assert second["geometry"]["available"] is True
    assert second["geometry"]["element_count"] == manifest["points"]


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


def test_a_stopped_unbuilt_session_projects_to_finalizing(monkeypatch, tmp_path):
    """`.finalizing` is "capture ended, figures may still change".

    That is exactly what `stopped_unbuilt` means, and it is why the two
    map onto each other -- not because Tower can see a build running,
    which lifecycle.build_in_progress still reports it cannot.
    """
    root = tmp_path / "worlds"
    _, _, engine = start_live_world(root, frames=8)
    engine.stop_session()

    payload = _payload(monkeypatch, root)
    assert payload["model_state"] == "finalizing"
    assert payload["lifecycle"]["build_in_progress"] is None


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
