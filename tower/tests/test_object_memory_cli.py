"""The two drivers, run as a user runs them: separate process, cold start.

`--detector none` throughout: these cases are about the drivers, not
about whether torchvision can find a laptop, and a default suite that
downloaded a model would fail on a train.
"""

import json
import subprocess
import sys

import cv2
import numpy as np
import pytest

from tower.object_memory.records import Confidence, ObjectObservation
from tower.object_memory.store import ObservationStore


def _run(script, *args):
    return subprocess.run(
        [sys.executable, f"scripts/{script}", *args],
        capture_output=True,
        text=True,
    )


def _jpeg(path):
    image = np.full((640, 360, 3), 120, np.uint8)
    path.write_bytes(cv2.imencode(".jpg", image)[1].tobytes())


@pytest.fixture
def capture(tmp_path):
    """A recorded capture: a journal beside its images, as the recorder writes it."""
    directory = tmp_path / "captures" / "cap-1"
    (directory / "frames").mkdir(parents=True)
    lines = []
    for seq in range(3):
        relpath = f"frames/{seq:08d}.jpg"
        _jpeg(directory / relpath)
        lines.append(
            json.dumps(
                {
                    "schema_version": 1,
                    "source_seq": seq,
                    "received_at": 1000.0 + seq,
                    "relpath": relpath,
                }
            )
        )
    (directory / "frames.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return directory


def _observation(
    object_class="laptop", observed_at=1000.0, recorded_at=None, best_score=None
):
    return ObjectObservation(
        object_class=object_class,
        detector_score=0.81,
        confidence=Confidence.HIGH,
        observed_at=observed_at,
        time_basis="tower-receipt",
        recorded_at=observed_at if recorded_at is None else recorded_at,
        source="glasses-camera",
        module_id="object-memory",
        session_id="cap-1",
        frame_seq=7,
        bounding_box=(0.1, 0.1, 0.5, 0.5),
        retention_tag="default",
        privacy_tags=("derived-only",),
        spatial_ref=None,
        external_refs=(),
        best_score=best_score,
    )


class TestSessionDriver:
    def test_help_exits_zero(self):
        assert _run("object_memory_session.py", "--help").returncode == 0

    def test_it_requires_exactly_one_frame_source(self, tmp_path):
        assert _run("object_memory_session.py").returncode != 0
        assert (
            _run(
                "object_memory_session.py",
                "--frames",
                str(tmp_path),
                "--follow-capture",
                str(tmp_path),
            ).returncode
            != 0
        )

    def test_a_missing_capture_directory_exits_nonzero(self, tmp_path):
        # `--root` is not decoration. Without it this CLI defaults to
        # `DEFAULT_OBSERVATION_ROOT`, which is the DEVELOPER'S OWN store
        # inside the checkout -- and this spawn is a subprocess, so no
        # fixture in this suite can reach it. It writes nothing today
        # only because the guard it hits happens to be eager; that is a
        # property of two other functions, not of this test.
        root = tmp_path / "root"

        assert (
            _run(
                "object_memory_session.py",
                "--root",
                str(root),
                "--follow-capture",
                str(tmp_path / "absent"),
            ).returncode
            != 0
        )
        assert not root.exists(), (
            "a session that could not start created its observation store "
            "anyway"
        )

    def test_it_reads_the_journal_rather_than_globbing_when_there_is_one(
        self, tmp_path, capture
    ):
        result = _run(
            "object_memory_session.py",
            "--frames",
            str(capture),
            "--detector",
            "none",
            "--root",
            str(tmp_path / "memory"),
            "--format",
            "json",
        )

        assert result.returncode == 0, result.stderr
        report = json.loads(result.stdout)
        assert report["frames_observed"] == 3
        assert report["timing_source"] == "capture-journal"
        assert report["session_id"] == "cap-1"

    def test_a_run_that_remembers_nothing_says_so_rather_than_failing(
        self, tmp_path, capture
    ):
        # An honest zero. The pipeline ran; there was nothing worth
        # remembering, which is a finding and not an error.
        result = _run(
            "object_memory_session.py",
            "--frames",
            str(capture),
            "--detector",
            "none",
            "--root",
            str(tmp_path / "memory"),
            "--format",
            "json",
        )

        report = json.loads(result.stdout)
        assert report["observations_recorded"] == 0
        assert report["stored_observations"] == 0

    def test_the_report_names_the_classes_this_run_could_record(
        self, tmp_path, capture
    ):
        """Renamed from `persisted_classes` because the answer became
        configuration-dependent: a run with a verifier can record the
        `verify` tier and a run without it cannot, and one key that meant
        both would be a key that meant neither.
        """
        result = _run(
            "object_memory_session.py",
            "--frames",
            str(capture),
            "--detector",
            "none",
            "--root",
            str(tmp_path / "memory"),
            "--format",
            "json",
        )

        report = json.loads(result.stdout)
        assert report["recordable_classes"] == ["laptop", "cell phone"]

    def test_there_is_no_flag_that_widens_the_whitelist(self):
        # Re-admitting `person` must be an edit that meets the comment on
        # PERSISTED_CLASSES, not a command-line switch someone can reach
        # for without reading it.
        help_text = _run("object_memory_session.py", "--help").stdout

        assert "--allow-class" not in help_text
        assert "--classes" not in help_text

    def test_a_directory_of_loose_jpegs_reports_that_it_has_no_clock(self, tmp_path):
        directory = tmp_path / "loose"
        directory.mkdir()
        _jpeg(directory / "a.jpg")

        result = _run(
            "object_memory_session.py",
            "--frames",
            str(directory),
            "--detector",
            "none",
            "--root",
            str(tmp_path / "memory"),
            "--format",
            "json",
        )

        assert json.loads(result.stdout)["timing_source"] == "none"


class TestQueryDriver:
    def test_help_exits_zero(self):
        assert _run("object_query.py", "--help").returncode == 0

    def test_it_requires_exactly_one_question(self, tmp_path):
        assert _run("object_query.py", "--root", str(tmp_path)).returncode != 0

    def test_no_record_exits_nonzero_and_does_not_claim_absence(self, tmp_path):
        result = _run("object_query.py", "--root", str(tmp_path), "--last-seen", "laptop")

        assert result.returncode == 1
        assert "No record" in result.stdout
        assert "not about the world" in result.stdout

    def test_it_answers_when_a_laptop_was_last_seen(self, tmp_path):
        store = ObservationStore(tmp_path, retention_seconds=None)
        store.append(_observation(observed_at=1000.0))
        store.append(_observation(observed_at=3000.0))

        result = _run(
            "object_query.py",
            "--root",
            str(tmp_path),
            "--last-seen",
            "laptop",
            "--now",
            "3060.0",
            "--format",
            "json",
        )

        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        assert payload["observed"] is True
        assert payload["observation"]["observed_at"] == 3000.0

    def test_where_is_a_frame_reference_and_never_a_place(self, tmp_path):
        store = ObservationStore(tmp_path, retention_seconds=None)
        store.append(_observation())

        payload = json.loads(
            _run(
                "object_query.py",
                "--root",
                str(tmp_path),
                "--last-seen",
                "laptop",
                "--now",
                "1060.0",
                "--format",
                "json",
            ).stdout
        )

        assert payload["where"]["kind"] == "frame-reference"
        assert payload["where"]["spatial_ref"] is None
        assert payload["where"]["session_id"] == "cap-1"
        assert payload["where"]["frame_seq"] == 7

    def test_the_text_answer_refuses_to_imply_a_location(self, tmp_path):
        store = ObservationStore(tmp_path, retention_seconds=None)
        store.append(_observation())

        stdout = _run(
            "object_query.py",
            "--root",
            str(tmp_path),
            "--last-seen",
            "laptop",
            "--now",
            "1060.0",
        ).stdout

        assert "not a place" in stdout
        assert "OBSERVED, NOT PRESENT" in stdout

    def test_an_expired_observation_is_not_served(self, tmp_path):
        # The read-time cutoff, end to end through the CLI: retention is
        # a promise about availability, so a query must not answer from
        # data the wearer was told had been forgotten.
        store = ObservationStore(tmp_path, retention_seconds=None)
        store.append(_observation(observed_at=1000.0))

        result = _run(
            "object_query.py",
            "--root",
            str(tmp_path),
            "--last-seen",
            "laptop",
            "--retention-days",
            "1",
            "--now",
            str(1000.0 + 2 * 86400.0),
        )

        assert result.returncode == 1
        assert "No record" in result.stdout

    def test_asking_for_a_class_nothing_records_says_why(self, tmp_path):
        stdout = _run(
            "object_query.py", "--root", str(tmp_path), "--last-seen", "person"
        ).stdout

        assert "only remembers laptop, cell phone" in stdout

    def test_purge_all_really_deletes(self, tmp_path):
        store = ObservationStore(tmp_path, retention_seconds=None)
        store.append(_observation())

        result = _run(
            "object_query.py", "--root", str(tmp_path), "--purge-all", "--format", "json"
        )

        assert json.loads(result.stdout)["observations_removed"] == 1
        assert store.all_observations() == []
        assert not any(tmp_path.iterdir())


class TestReaderCannotWidenRetention:
    """--retention-days is a request, not an authority.

    The store persists the window it was written under, and every read
    clamps to min(persisted, requested). Before that, a reader could hand
    itself 3650 days -- or 0, meaning forever -- and be served a record
    the wearer had been promised was gone.
    """

    WRITTEN_AT = 1_000_000.0
    WINDOW = 30 * 86400.0

    def _store_written_under_the_default(self, tmp_path, age_days):
        recorded_at = self.WRITTEN_AT
        store = ObservationStore(
            tmp_path, retention_seconds=self.WINDOW, clock=lambda: recorded_at
        )
        store.append(_observation(observed_at=recorded_at, recorded_at=recorded_at))
        return recorded_at + age_days * 86400.0

    @pytest.mark.parametrize("requested", ["3650", "0"])
    def test_a_wider_request_does_not_resurrect_an_expired_record(
        self, tmp_path, requested
    ):
        now = self._store_written_under_the_default(tmp_path, age_days=40)

        result = _run(
            "object_query.py",
            "--root",
            str(tmp_path),
            "--last-seen",
            "laptop",
            "--retention-days",
            requested,
            "--now",
            str(now),
        )

        assert result.returncode == 1
        assert "No record" in result.stdout

    def test_a_wider_request_does_not_leak_the_record_as_json_either(self, tmp_path):
        now = self._store_written_under_the_default(tmp_path, age_days=40)

        result = _run(
            "object_query.py",
            "--root",
            str(tmp_path),
            "--last-seen",
            "laptop",
            "--retention-days",
            "0",
            "--now",
            str(now),
            "--format",
            "json",
        )

        payload = json.loads(result.stdout)
        assert payload["observed"] is False
        assert payload["observation"] is None

    def test_a_record_inside_the_promised_window_is_still_answered(self, tmp_path):
        now = self._store_written_under_the_default(tmp_path, age_days=3)

        result = _run(
            "object_query.py",
            "--root",
            str(tmp_path),
            "--last-seen",
            "laptop",
            "--now",
            str(now),
        )

        assert result.returncode == 0, result.stderr

    def test_a_narrower_request_is_still_obeyed(self, tmp_path):
        now = self._store_written_under_the_default(tmp_path, age_days=3)

        result = _run(
            "object_query.py",
            "--root",
            str(tmp_path),
            "--last-seen",
            "laptop",
            "--retention-days",
            "1",
            "--now",
            str(now),
        )

        assert result.returncode == 1
        assert "No record" in result.stdout


class TestTheAnswerSaysWhatItKnows:
    def test_it_reports_the_first_score_and_the_best_one_while_in_view(self, tmp_path):
        store = ObservationStore(tmp_path, retention_seconds=None)
        store.append(_observation(best_score=0.97))

        stdout = _run(
            "object_query.py",
            "--root",
            str(tmp_path),
            "--last-seen",
            "laptop",
            "--now",
            "1060.0",
        ).stdout

        assert "0.81" in stdout
        assert "0.97" in stdout
        assert "best score" in stdout

    def test_the_json_answer_carries_the_best_score(self, tmp_path):
        store = ObservationStore(tmp_path, retention_seconds=None)
        store.append(_observation(best_score=0.97))

        payload = json.loads(
            _run(
                "object_query.py",
                "--root",
                str(tmp_path),
                "--last-seen",
                "laptop",
                "--now",
                "1060.0",
                "--format",
                "json",
            ).stdout
        )

        assert payload["observation"]["best_score"] == pytest.approx(0.97)

    def test_it_says_that_deleting_the_record_does_not_delete_the_frame(
        self, tmp_path
    ):
        # The honest half of `derived-only`: the record holds no imagery,
        # but session_id + frame_seq resolves to a JPEG this cartridge's
        # retention does not govern.
        store = ObservationStore(tmp_path, retention_seconds=None)
        store.append(_observation())

        stdout = _run(
            "object_query.py",
            "--root",
            str(tmp_path),
            "--last-seen",
            "laptop",
            "--now",
            "1060.0",
        ).stdout

        assert "capture retention" in stdout
        assert "data/captures" in stdout
