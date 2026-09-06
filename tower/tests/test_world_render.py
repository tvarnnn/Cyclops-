"""world_render: what gets composed, what stays apart, what gets written.

A tiny synthetic world with three segments and one non-trivial placement.
The tests pin the three things a picture can lie about:

  * the Sim3 is applied exactly as the store defines it
    (X_ref = s * R @ X_seg + t), checked against a hand-computed point;
  * an unregistered segment is NOT placed into the world PLY;
  * a registered placement bound to a different build is NOT drawn.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import world_render  # noqa: E402
from tower.world_builder.store import WorldStore  # noqa: E402

pytest.importorskip("matplotlib")

WORLD = "w" * 32
SESSION = "s" * 32
DIGEST = "d" * 64

# A 90-degree yaw about y, as wxyz.
YAW_90 = [math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0]


def _placement(index, state, **fields):
    row = {
        "schema_version": 1,
        "segment_index": index,
        "state": state,
        "rotation_wxyz": None,
        "translation": None,
        "scale": None,
        "reference_segment": None,
        "refusal_reason": None,
        "evidence": {},
        "frame_revision": 1,
        "input_digest": DIGEST,
    }
    row.update(fields)
    return row


def _write_world(root: Path, *, placements) -> None:
    derived = root / "worlds" / WORLD / "derived"
    session = derived / SESSION
    session.mkdir(parents=True)
    rng = np.random.default_rng(7)
    points = []
    for index, centre in ((0, (0.0, 0.0, 5.0)), (1, (2.0, 0.0, 5.0)), (2, (0.0, 0.0, 1.0))):
        cloud = rng.normal(size=(40, 3)) * 0.3 + np.asarray(centre)
        points.extend({"segment_index": index, "xyz": row.tolist()} for row in cloud)
    # A known point in segment 1, used for the hand-computed check.
    points.append({"segment_index": 1, "xyz": [1.0, 2.0, 3.0]})
    (session / "points.json").write_text(json.dumps({"points": points}))
    poses = [
        {"keyframe_id": f"{SESSION}:1", "segment_index": 0, "status": "anchor",
         "degeneracy": "", "rotation": [1, 0, 0, 0], "translation": [0, 0, 0]},
        {"keyframe_id": f"{SESSION}:2", "segment_index": 0, "status": "solved",
         "degeneracy": "", "rotation": [1, 0, 0, 0], "translation": [0.5, 0, 0]},
        {"keyframe_id": f"{SESSION}:3", "segment_index": 1, "status": "anchor",
         "degeneracy": "", "rotation": [1, 0, 0, 0], "translation": [0, 0, 0]},
        {"keyframe_id": f"{SESSION}:4", "segment_index": 1, "status": "unavailable",
         "degeneracy": "pure_rotation", "rotation": None, "translation": None},
        {"keyframe_id": f"{SESSION}:5", "segment_index": 2, "status": "anchor",
         "degeneracy": "", "rotation": [1, 0, 0, 0], "translation": [0, 0, 0]},
    ]
    (session / "poses.json").write_text(json.dumps({"poses": poses}))
    (session / "placements.json").write_text(json.dumps({"placements": placements}))
    (derived / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "input_digest": DIGEST, "session_id": SESSION,
        "segments": 3, "points": len(points),
    }))


@pytest.fixture
def rendered(tmp_path):
    root = tmp_path / "data"
    _write_world(root, placements=[
        _placement(0, "registered", rotation_wxyz=[1, 0, 0, 0],
                   translation=[0, 0, 0], scale=1.0, reference_segment=0),
        _placement(1, "registered", rotation_wxyz=YAW_90,
                   translation=[10.0, -1.0, 0.5], scale=2.0, reference_segment=0),
        _placement(2, "refused",
                   refusal_reason="the wearer stood still"),
    ])
    out = tmp_path / "render"
    summary = world_render.render_world(
        WorldStore(root), WORLD, SESSION, out, html_backend="canvas", dpi=60,
    )
    return root, out, summary


def test_world_ply_is_written_and_holds_only_registered_points(rendered):
    _root, out, summary = rendered
    ply = out / "world.ply"
    assert ply.exists() and ply.stat().st_size > 0
    xyz, rgb = world_render.read_ply_xyz_rgb(ply)
    # 40 + 41 registered points; segment 2's 40 are not there.
    assert len(xyz) == 81
    assert summary["frames"][0]["segments"] == [0, 1]
    assert summary["frames"][0]["points"] == 81
    # Segment 2 lives near z=1 in its own frame; nothing in the world PLY
    # sits there, so it was not overlaid at an identity transform.
    assert not np.any(np.abs(xyz[:, 2] - 1.0) < 0.5)
    # Two colours, one per registered segment.
    assert len({tuple(row) for row in rgb}) == 2


def test_composition_matches_a_hand_computed_sim3(rendered):
    _root, out, _summary = rendered
    xyz, _rgb = world_render.read_ply_xyz_rgb(out / "world.ply")
    # X_ref = s * R @ X + t with R = yaw 90 deg about y: (x, y, z) -> (z, y, -x)
    expected = 2.0 * np.array([3.0, 2.0, -1.0]) + np.array([10.0, -1.0, 0.5])
    distances = np.linalg.norm(xyz - expected, axis=1)
    assert distances.min() < 1e-4, f"nearest world point is {distances.min()} away"
    # And the same answer the registration code gives for the same Sim3.
    sim3 = world_render.Sim3(
        2.0, world_render._quaternion_wxyz_to_rotation(YAW_90),
        np.array([10.0, -1.0, 0.5]),
    )
    assert np.allclose(sim3.apply(np.array([1.0, 2.0, 3.0])), expected)


def test_unregistered_segment_is_rendered_separately(rendered):
    _root, out, summary = rendered
    assert (out / "unregistered_segment_2.ply").exists()
    xyz, _rgb = world_render.read_ply_xyz_rgb(out / "unregistered_segment_2.ply")
    assert len(xyz) == 40
    assert (out / "unregistered_tiles.png").stat().st_size > 0
    rows = {row["segment_index"]: row for row in summary["unregistered"]}
    assert list(rows) == [2]
    assert rows[2]["state"] == "refused"
    assert rows[2]["reason"] == "the wearer stood still"
    assert summary["segments"]["2"]["registered"] is False


def test_pngs_html_and_summary_exist(rendered):
    _root, out, summary = rendered
    for name in ("view_top.png", "view_front.png", "view_side.png",
                 "view_iso.png", "overview.png", "world.html", "summary.json",
                 "cameras_world.ply"):
        assert (out / name).stat().st_size > 0, name
    assert summary["files"]["html_backend"] == "canvas"
    # The refused camera row is counted, never drawn as a zero.
    assert summary["segments"]["1"]["cameras"] == 1
    assert summary["segments"]["1"]["cameras_refused"] == 1
    assert summary["frames"][0]["cameras"] == 3
    reread = json.loads((out / "summary.json").read_text())
    assert reread["points_total"] == 121
    assert reread["derived_current"] is None  # no journal in the fixture


def test_a_placement_bound_to_another_build_is_not_drawn(tmp_path):
    root = tmp_path / "data"
    _write_world(root, placements=[
        _placement(0, "registered", rotation_wxyz=[1, 0, 0, 0],
                   translation=[0, 0, 0], scale=1.0, reference_segment=0),
        _placement(1, "registered", rotation_wxyz=[1, 0, 0, 0],
                   translation=[0, 0, 0], scale=1.0, reference_segment=0,
                   input_digest="e" * 64),
    ])
    summary = world_render.render_world(
        WorldStore(root), WORLD, SESSION, tmp_path / "render",
        html=False, dpi=60,
    )
    assert summary["frames"][0]["segments"] == [0]
    assert summary["segments"]["1"]["state"] == "unbound"
    assert summary["segments"]["1"]["registered"] is False


def test_subsampling_spans_the_cloud_instead_of_taking_a_prefix(tmp_path):
    root = tmp_path / "data"
    _write_world(root, placements=[])
    segments = world_render.load_segments(WorldStore(root), WORLD, SESSION)
    original = segments[1].points.copy()
    report = world_render.subsample(segments, 30)
    assert report["subsampled"] is True
    assert report["points_kept"] <= 33
    kept = segments[1].points
    assert 0 < len(kept) < len(original)
    # Not a prefix: the last kept point comes from the back half of the cloud.
    assert not np.array_equal(kept, original[: len(kept)])
    where = np.flatnonzero(np.all(original == kept[-1], axis=1))
    assert where.max() >= len(original) // 2


def test_cli_renders_with_world_root_and_world(tmp_path, capsys):
    root = tmp_path / "data"
    _write_world(root, placements=[])
    out = tmp_path / "render"
    code = world_render.main([
        "--world-root", str(root), "--world", WORLD, "--out", str(out),
        "--no-html", "--dpi", "50",
    ])
    assert code == 0
    assert (out / "summary.json").exists()
    # No registered segments: no world frame, and the overview says so
    # rather than overlaying three unrelated frames.
    summary = json.loads((out / "summary.json").read_text())
    assert summary["frames"] == []
    assert not (out / "world.ply").exists()
    assert (out / "overview.png").exists()
    assert (out / "unregistered_tiles.png").exists()
    assert "rendered separately" in capsys.readouterr().out


def test_cli_accepts_a_world_dir(tmp_path):
    root = tmp_path / "data"
    _write_world(root, placements=[])
    out = tmp_path / "render"
    code = world_render.main([
        "--world-dir", str(root / "worlds" / WORLD), "--out", str(out), "--no-html",
        "--dpi", "50",
    ])
    assert code == 0
    assert (out / "summary.json").exists()
