"""`world_replay.py` stages a walk whose captures restart their numbering.

A walk is a chain of captures, one per transport connection. On some walks
(4cae0b26, adc75972) every reconnect restarts the phone's frame numbering at
`00000001.jpg`, so the raw names collide across captures. `stage_frames` used
to stage frames under their raw names and refused the collision with
"appears in two captures", which made those walks impossible to replay; and a
chain whose later capture happened to number LOWER than an earlier one would
have been sorted out of walk order without any error at all.

The staging directory is what `world_build_session.py --frames` reads, in
sorted order, and each staged path is what the builder records in
`sources.json` as the keyframe's raw frame. So the properties that matter
are: every frame staged once, under a unique name; sorted order is capture
order; and every staged path is the very raw file it claims to be.

Frames here are distinct byte strings named `.jpg`: staging and
`load_frames` never decode them, and distinct content is what lets a test
tell one photograph from another.
"""

import json
import os
from pathlib import Path

import pytest

from scripts import world_replay
from scripts.world_build_session import load_frames


def _capture(root: Path, capture_id: str, names_and_payloads) -> Path:
    frames = root / capture_id / "frames"
    frames.mkdir(parents=True)
    for name, payload in names_and_payloads:
        (frames / name).write_bytes(payload)
    return root / capture_id


def _staged_in_sorted_order(staging: Path) -> list[Path]:
    # Exactly what the builder does with --frames.
    return sorted(staging.glob("*.jpg"))


CAP_A = "aaaaaaaa11112222333344445555aaaa"
CAP_B = "bbbbbbbb11112222333344445555bbbb"


@pytest.fixture
def colliding_chain(tmp_path):
    """Two captures of one walk, both numbered from 00000001."""
    root = tmp_path / "captures"
    _capture(root, CAP_A, [(f"{i:08d}.jpg", f"A{i}".encode()) for i in (1, 2, 3)])
    _capture(root, CAP_B, [(f"{i:08d}.jpg", f"B{i}".encode()) for i in (1, 2, 3, 4)])
    return root


def test_colliding_chain_stages_every_frame_once_in_capture_order(colliding_chain, tmp_path):
    staging = tmp_path / "replay" / "_frames"

    count = world_replay.stage_frames([CAP_A, CAP_B], colliding_chain, staging)

    staged = _staged_in_sorted_order(staging)
    assert count == 7
    assert len(staged) == 7
    assert len({p.name for p in staged}) == 7
    assert [p.read_bytes() for p in staged] == [
        b"A1", b"A2", b"A3", b"B1", b"B2", b"B3", b"B4"
    ]


def test_capture_order_wins_over_raw_numbering(tmp_path):
    """The later capture numbers lower: raw-name order would put it first."""
    root = tmp_path / "captures"
    _capture(root, CAP_A, [("00000500.jpg", b"A500"), ("00000501.jpg", b"A501")])
    _capture(root, CAP_B, [("00000001.jpg", b"B1"), ("00000002.jpg", b"B2")])
    staging = tmp_path / "_frames"

    world_replay.stage_frames([CAP_A, CAP_B], root, staging)

    assert [p.read_bytes() for p in _staged_in_sorted_order(staging)] == [
        b"A500", b"A501", b"B1", b"B2"
    ]


def test_continuing_numbering_keeps_the_raw_name_order(tmp_path):
    """The pinned cases (worldA, worldB) number continuously across captures;
    their replay must see the same frames in the same order as before."""
    root = tmp_path / "captures"
    _capture(root, CAP_A, [("00000001.jpg", b"A1"), ("00000852.jpg", b"A852")])
    _capture(root, CAP_B, [("00000897.jpg", b"B897"), ("00001111.jpg", b"B1111")])
    staging = tmp_path / "_frames"

    world_replay.stage_frames([CAP_A, CAP_B], root, staging)

    raw_order = sorted(
        [p for c in (CAP_A, CAP_B) for p in (root / c / "frames").glob("*.jpg")],
        key=lambda p: p.name,
    )
    staged = _staged_in_sorted_order(staging)
    assert [p.read_bytes() for p in staged] == [p.read_bytes() for p in raw_order]


def test_every_staged_frame_is_the_raw_file_the_manifest_names(colliding_chain, tmp_path):
    staging = tmp_path / "_frames"
    world_replay.stage_frames([CAP_A, CAP_B], colliding_chain, staging)

    manifest = json.loads((staging / world_replay.STAGING_MANIFEST).read_text("utf-8"))
    assert manifest["schema"] == world_replay.STAGING_SCHEMA
    assert manifest["captures"] == [CAP_A, CAP_B]
    entries = manifest["frames"]
    assert [e["staged"] for e in entries] == [p.name for p in _staged_in_sorted_order(staging)]
    for entry in entries:
        staged = staging / entry["staged"]
        raw = Path(entry["raw_path"])
        # A hard link to the raw frame, not a copy and not a neighbour.
        assert os.path.samefile(staged, raw)
        assert raw.parent == (colliding_chain / entry["capture_id"] / "frames").resolve()
        assert raw.name == entry["raw_name"]
        assert entry["capture_id"][:8] in entry["staged"]
        assert entry["capture_ordinal"] == [CAP_A, CAP_B].index(entry["capture_id"])


def test_builder_source_paths_resolve_to_the_right_raw_frames(colliding_chain, tmp_path):
    """`sources.json` is `keyframe_id -> str(frame.source_path)` over exactly
    these frames (world_build_session.py). Each one must be the raw frame the
    builder saw, in walk order, under the enumeration source_seq."""
    staging = tmp_path / "_frames"
    world_replay.stage_frames([CAP_A, CAP_B], colliding_chain, staging)
    manifest = json.loads((staging / world_replay.STAGING_MANIFEST).read_text("utf-8"))
    raw_by_staged = {e["staged"]: Path(e["raw_path"]) for e in manifest["frames"]}

    frames = load_frames(staging)

    assert [f.payload for f in frames] == [
        b"A1", b"A2", b"A3", b"B1", b"B2", b"B3", b"B4"
    ]
    assert [f.source_seq for f in frames] == list(range(7))
    for frame in frames:
        raw = raw_by_staged[frame.source_path.name]
        assert os.path.samefile(frame.source_path, raw)
        assert frame.payload == raw.read_bytes()


def test_restaging_the_same_walk_is_idempotent(colliding_chain, tmp_path):
    staging = tmp_path / "_frames"
    world_replay.stage_frames([CAP_A, CAP_B], colliding_chain, staging)
    first = [(p.name, p.read_bytes()) for p in _staged_in_sorted_order(staging)]

    assert world_replay.stage_frames([CAP_A, CAP_B], colliding_chain, staging) == 7
    assert [(p.name, p.read_bytes()) for p in _staged_in_sorted_order(staging)] == first


def test_a_capture_named_twice_is_refused_before_anything_is_linked(colliding_chain, tmp_path):
    staging = tmp_path / "_frames"
    with pytest.raises(SystemExit, match="more than once"):
        world_replay.stage_frames([CAP_A, CAP_B, CAP_A], colliding_chain, staging)
    assert not staging.exists() or not list(staging.glob("*.jpg"))


def test_foreign_frames_in_the_staging_directory_are_refused(colliding_chain, tmp_path):
    staging = tmp_path / "_frames"
    staging.mkdir()
    (staging / "00000001.jpg").write_bytes(b"left over from another replay")
    with pytest.raises(SystemExit, match="does not stage"):
        world_replay.stage_frames([CAP_A, CAP_B], colliding_chain, staging)
    assert [p.name for p in staging.glob("*.jpg")] == ["00000001.jpg"]


def test_a_staged_name_holding_another_photograph_is_refused(colliding_chain, tmp_path):
    staging = tmp_path / "_frames"
    staging.mkdir()
    name = world_replay.staged_name(0, 3, CAP_A, "00000001.jpg")
    (staging / name).write_bytes(b"a different photograph")
    with pytest.raises(SystemExit, match="not a link"):
        world_replay.stage_frames([CAP_A, CAP_B], colliding_chain, staging)
    assert [p.name for p in staging.glob("*.jpg")] == [name]


def test_a_missing_capture_is_refused_before_anything_is_linked(colliding_chain, tmp_path):
    staging = tmp_path / "_frames"
    with pytest.raises(SystemExit, match="no frames directory"):
        world_replay.stage_frames([CAP_A, "cccccccc", CAP_B], colliding_chain, staging)
    assert not staging.exists() or not list(staging.glob("*.jpg"))


def test_replay_cli_hands_the_builder_the_staged_chain(colliding_chain, tmp_path, monkeypatch, capsys):
    """The CLI path end to end, with the builder stubbed: the colliding chain
    no longer exits, and the builder is pointed at the staged directory."""
    seen = {}

    class _Done:
        returncode = 0
        stdout = json.dumps({"keyframes_accepted": 0})
        stderr = ""

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        return _Done()

    monkeypatch.setattr(world_replay.subprocess, "run", fake_run)
    root = tmp_path / "replay-root"

    code = world_replay.main([
        "--captures", CAP_A, CAP_B,
        "--capture-root", str(colliding_chain),
        "--intrinsics-from", str(tmp_path / "no-calibrations-here"),
        "--root", str(root),
        "--format", "json",
    ])

    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["replay"]["frames_staged"] == 7
    assert report["replay"]["captures"] == [CAP_A, CAP_B]
    argv = seen["argv"]
    assert Path(argv[argv.index("--frames") + 1]) == root.resolve() / "_frames"
    assert Path(report["replay"]["staging_manifest"]).is_file()
