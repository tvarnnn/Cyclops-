"""The live drivers: build a world as frames land, and watch it happen.

SYNTHETIC, NOT PHYSICAL.

Three processes, never one. The Tower records; a builder tails the
capture; an inspector tails the world. That separation is what lets an
expensive rebuild run repeatedly during a walk without costing the frame
path a millisecond -- and it is why none of this needed the blocked module
lifecycle.
"""

import io
import json
import subprocess
import sys

import pytest
from PIL import Image

from tower.capture import CaptureRecorder
from tower.world_builder.store import WorldStore
from tests import synthetic_scene as ss

WIDTH, HEIGHT = 160, 120


def _run(script, *args):
    return subprocess.run(
        [sys.executable, f"scripts/{script}", *args],
        capture_output=True,
        text=True,
    )


@pytest.fixture
def finished_capture(tmp_path):
    """A closed capture directory holding a real synthetic walk."""
    recorder = CaptureRecorder(tmp_path / "capture")
    capture_id = recorder.start()
    scene = ss.furnished_room()
    poses = ss.strafe(10, step=0.09)
    images = ss.render_sequence(
        scene, poses, ss.camera_matrix(WIDTH, HEIGHT), WIDTH, HEIGHT
    )
    for index, image in enumerate(images):
        recorder.write_frame(
            ss.encode_jpeg(image),
            source_seq=index,
            wire_seq=index,
            tx_seq=index,
            width=WIDTH,
            height=HEIGHT,
        )
    recorder.stop()
    return recorder.capture_dir(capture_id), capture_id


class TestFollowCapture:
    def test_a_capture_directory_builds_a_world(self, tmp_path, finished_capture):
        directory, _ = finished_capture

        result = _run(
            "world_build_session.py",
            "--follow-capture",
            str(directory),
            "--root",
            str(tmp_path / "worlds"),
            "--format",
            "json",
        )

        assert result.returncode == 0, result.stderr
        report = json.loads(result.stdout)
        assert report["frames_observed"] == 10
        assert report["keyframes_accepted"] >= 2
        assert report["frame_source"] == "live-capture"

    def test_the_session_records_which_capture_it_came_from(
        self, tmp_path, finished_capture
    ):
        """Provenance: a world must be traceable back to its raw imagery."""
        directory, capture_id = finished_capture
        root = tmp_path / "worlds"

        result = _run(
            "world_build_session.py",
            "--follow-capture",
            str(directory),
            "--root",
            str(root),
            "--format",
            "json",
        )
        report = json.loads(result.stdout)

        store = WorldStore(root)
        session = store.read_session(report["world_id"], report["session_id"])
        assert session.capture_id == capture_id

    def test_frame_metadata_from_the_wire_reaches_the_keyframes(
        self, tmp_path, finished_capture
    ):
        """source_seq/tx_seq are the whole reason for tailing the journal.

        A driver that just globbed the image directory would lose them,
        and with them any ability to reason about dropped frames.
        """
        directory, _ = finished_capture
        root = tmp_path / "worlds"

        report = json.loads(
            _run(
                "world_build_session.py",
                "--follow-capture",
                str(directory),
                "--root",
                str(root),
                "--format",
                "json",
            ).stdout
        )

        keyframes = WorldStore(root).read_keyframes(
            report["world_id"], report["session_id"]
        )
        assert keyframes
        assert all(keyframe.tx_seq is not None for keyframe in keyframes)
        assert all(
            keyframe.wire_seq == keyframe.source_seq for keyframe in keyframes
        )

    def test_refuses_more_than_one_frame_source(self, tmp_path):
        result = _run(
            "world_build_session.py",
            "--synthetic",
            "--follow-capture",
            str(tmp_path),
        )

        assert result.returncode != 0

    def test_a_missing_capture_directory_exits_nonzero(self, tmp_path):
        # `--root` is not decoration. Without it this CLI defaults to a
        # RELATIVE `data/world_builder`, which resolves against pytest's
        # working directory -- so this one test minted a world into the
        # developer's real store on every full-suite run.
        result = _run(
            "world_build_session.py",
            "--root",
            str(tmp_path / "root"),
            "--follow-capture",
            str(tmp_path / "absent"),
        )

        assert result.returncode != 0

    def test_a_session_that_cannot_start_mints_no_world(self, tmp_path):
        """A session that never begins must leave nothing behind.

        This is the mechanism behind the stub worlds: 86 of the 123
        worlds in the corpus on this host hold a `world.json` and no
        sessions at all, and they accumulate with install age.

        `world_build_session.py` creates the world BEFORE the first
        frame, deliberately -- "a Tower whose phone has connected but
        not yet sent a frame reports a world that exists and is empty
        rather than no world at all". That claim is about a session that
        CAN start. This one cannot: there is no capture directory, and
        the process is about to exit nonzero.

        It minted anyway because `follow_capture` is a GENERATOR. Its
        `if not directory.exists(): raise SystemExit` does not run when
        the function is called -- only when the generator is first
        advanced, which happens AFTER `engine.create_world()`. The guard
        fired one statement too late, every time, and left a permanent
        world to prove it.
        """
        root = tmp_path / "root"

        result = _run(
            "world_build_session.py",
            "--root",
            str(root),
            "--follow-capture",
            str(tmp_path / "absent"),
        )

        assert result.returncode != 0
        assert "no capture directory" in (result.stderr + result.stdout)
        minted = sorted((root / "worlds").glob("*/world.json"))
        assert not minted, (
            f"a session that could not start minted {len(minted)} world(s): "
            f"{[path.parent.name for path in minted]}"
        )


class TestIncrementalRebuild:
    def test_rebuilding_during_the_walk_produces_geometry_before_the_stop(
        self, tmp_path, finished_capture
    ):
        """The product ruling's word is *incrementally*.

        Without this the experience is Start -> Walk -> Stop -> the world
        appears. The rebuild is only affordable because it happens in a
        separate process from the one receiving frames.
        """
        directory, _ = finished_capture
        root = tmp_path / "worlds"

        report = json.loads(
            _run(
                "world_build_session.py",
                "--follow-capture",
                str(directory),
                "--root",
                str(root),
                "--rebuild-every",
                "2",
                "--format",
                "json",
            ).stdout
        )

        assert report["rebuilds"] >= 1
        assert report["points"] >= 0

    def test_no_rebuild_cadence_means_one_build_at_the_end(
        self, tmp_path, finished_capture
    ):
        directory, _ = finished_capture

        report = json.loads(
            _run(
                "world_build_session.py",
                "--follow-capture",
                str(directory),
                "--root",
                str(tmp_path / "worlds"),
                "--format",
                "json",
            ).stdout
        )

        assert report["rebuilds"] == 0

    def test_a_mid_walk_rebuild_does_not_change_the_final_result(
        self, tmp_path, finished_capture
    ):
        """Rebuilding is idempotent over the same keyframes.

        If it were not, watching a world build would change the world --
        the viewer would be a participant, and two operators would get
        different maps from identical footage.
        """
        directory, _ = finished_capture

        plain = json.loads(
            _run(
                "world_build_session.py",
                "--follow-capture",
                str(directory),
                "--root",
                str(tmp_path / "a"),
                "--format",
                "json",
            ).stdout
        )
        incremental = json.loads(
            _run(
                "world_build_session.py",
                "--follow-capture",
                str(directory),
                "--root",
                str(tmp_path / "b"),
                "--rebuild-every",
                "2",
                "--format",
                "json",
            ).stdout
        )

        for key in (
            "keyframes_accepted",
            "poses_solved",
            "poses_refused",
            "points",
            "segments",
            "scale_state",
        ):
            assert plain[key] == incremental[key], key


class TestInspectFollow:
    def test_following_a_finished_session_replays_its_events(
        self, tmp_path, finished_capture
    ):
        directory, _ = finished_capture
        root = tmp_path / "worlds"
        report = json.loads(
            _run(
                "world_build_session.py",
                "--follow-capture",
                str(directory),
                "--root",
                str(root),
                "--format",
                "json",
            ).stdout
        )

        result = _run(
            "world_inspect.py",
            "--root",
            str(root),
            "--world",
            report["world_id"],
            "--follow",
        )

        assert result.returncode == 0, result.stderr
        assert "session_started" in result.stdout
        assert "session_stopped" in result.stdout
        assert "keyframe_accepted" in result.stdout

    def test_the_stop_event_records_what_happened_to_the_CAPTURE(
        self, tmp_path, finished_capture
    ):
        """`end_reason` is the world's; `capture_end_reason` is the link's.

        They differ in a case a real walk produces. A capture that ended
        `disconnect` counts as finished, so a walk whose phone never came
        back is recorded as an ordinary `stop` -- deliberately, because
        the alternative puts the campaign's headline symptom back. That
        choice is defensible only while the artifact still says which it
        was, and the session record names only the FIRST capture it
        followed, so after a reconnect the link was a timestamp search.
        """
        from tower.world_builder.store import WorldStore

        directory, capture_id = finished_capture
        root = tmp_path / "worlds"
        report = json.loads(
            _run(
                "world_build_session.py",
                "--follow-capture", str(directory),
                "--root", str(root),
                "--format", "json",
            ).stdout
        )

        store = WorldStore(root)
        events = store.read_events(report["world_id"], report["session_id"])
        stopped = [e for e in events if e["kind"] == "session_stopped"]
        assert stopped, "the walk never recorded a stop"
        payload = stopped[-1]["payload"]
        recorded = json.loads(
            (directory / "capture.json").read_text(encoding="utf-8")
        )["end_reason"]
        assert payload["capture_end_reason"] == recorded, (
            "the journal does not say what happened to the link: "
            f"{payload} against a capture that ended {recorded!r}"
        )

    def test_an_offline_caller_writes_the_journal_it_always_wrote(self, tmp_path):
        """The key is ABSENT when nobody knows, not None and not guessed.

        Every offline caller -- a replay, a test, `world_finalize.py` --
        has no follower to ask. Their journals must be byte-identical to
        what they were before this field existed.
        """
        from tower.world_builder.engine import WorldBuilderEngine
        from tower.world_builder.store import WorldStore

        store = WorldStore(tmp_path / "worlds")
        engine = WorldBuilderEngine(store)
        world_id = engine.create_world("offline")
        session_id = engine.start_session(world_id, frame_source="synthetic")
        engine.stop_session("stop")

        stopped = [
            e for e in store.read_events(world_id, session_id)
            if e["kind"] == "session_stopped"
        ]
        assert stopped[-1]["payload"] == {"end_reason": "stop"}

    def test_follow_emits_json_lines_when_asked(self, tmp_path, finished_capture):
        """A viewer consumes this, not a human. One event per line."""
        directory, _ = finished_capture
        root = tmp_path / "worlds"
        report = json.loads(
            _run(
                "world_build_session.py",
                "--follow-capture",
                str(directory),
                "--root",
                str(root),
                "--format",
                "json",
            ).stdout
        )

        result = _run(
            "world_inspect.py",
            "--root",
            str(root),
            "--world",
            report["world_id"],
            "--follow",
            "--format",
            "json",
        )

        assert result.returncode == 0
        events = [json.loads(line) for line in result.stdout.splitlines() if line]
        assert [event["event_id"] for event in events] == list(range(len(events)))
        assert events[0]["kind"] == "session_started"

    def test_follow_requires_a_world(self, tmp_path):
        result = _run("world_inspect.py", "--root", str(tmp_path), "--follow")

        assert result.returncode != 0
