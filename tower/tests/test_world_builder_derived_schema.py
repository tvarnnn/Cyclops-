"""The on-disk shape of the derived reconstruction.

These files are about to become a wire contract. Nothing pinned their key
sets, their dtypes, the wxyz quaternion order, or the difference between
null and zero -- so a rename would have been invisible until a phone
rendered it wrong.
"""

import json


def test_poses_json_has_exactly_the_documented_keys(derived_world):
    store, world_id, session_id = derived_world
    path = store.derived_dir(world_id) / session_id / "poses.json"
    data = json.loads(path.read_text())

    assert set(data) == {"poses"}
    required = {
        "keyframe_id", "segment_index", "status", "degeneracy",
        "rotation", "translation",
    }
    # `observations` is written only by the global solve (the 3-D points a
    # keyframe observes); the chain never writes it. Optional, never zero
    # for a posed row.
    optional = {"observations"}
    for row in data["poses"]:
        assert required <= set(row) <= required | optional


def test_a_refused_pose_keeps_null_and_not_zero(derived_world):
    """null means refused. A zero translation is a claim about the world."""
    store, world_id, session_id = derived_world
    path = store.derived_dir(world_id) / session_id / "poses.json"
    rows = json.loads(path.read_text())["poses"]

    refused = [r for r in rows if r["status"] == "unavailable"]
    assert refused, "fixture must contain a refused pose"
    for row in refused:
        assert row["translation"] is None
        assert row["rotation"] is None


def test_an_anchor_is_identity_and_origin_exactly(derived_world):
    """Anchors are definitional, not measured. Every segment starts at one."""
    store, world_id, session_id = derived_world
    path = store.derived_dir(world_id) / session_id / "poses.json"
    rows = json.loads(path.read_text())["poses"]

    anchors = [r for r in rows if r["status"] == "anchor"]
    assert len(anchors) == 2
    for row in anchors:
        assert row["rotation"] == [1.0, 0.0, 0.0, 0.0]
        assert row["translation"] == [0.0, 0.0, 0.0]


def test_points_json_rows_carry_their_segment(derived_world):
    """Segments share no frame, so an untagged point cannot be placed."""
    store, world_id, session_id = derived_world
    path = store.derived_dir(world_id) / session_id / "points.json"
    data = json.loads(path.read_text())

    assert set(data) == {"points"}
    required = {"segment_index", "xyz"}
    # `rgb` is written only by the global solve, which sees the frames the
    # points came from; the chain never writes it.
    optional = {"rgb"}
    for row in data["points"]:
        assert required <= set(row) <= required | optional
        assert len(row["xyz"]) == 3


def test_derived_manifest_has_exactly_the_documented_keys(derived_world):
    store, world_id, _ = derived_world
    manifest = store.read_derived_manifest(world_id)

    assert set(manifest) == {
        "schema_version", "input_digest", "built_at", "backend_id",
        "session_id", "keyframes", "poses_solved", "poses_refused",
        "poses_anchor", "poses_positioned", "points", "segments",
        "scale_state", "global_solve",
    }


def test_the_fixture_survives_digest_verification(derived_world):
    """If this fails every later geometry test is reading a stale tree."""
    store, world_id, session_id = derived_world
    assert store.read_derived(world_id, session_id) is not None
