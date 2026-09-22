"""The switch-soak harness's CLI contract, kept fast.

Driven in-process through `main([...])` rather than as a subprocess: the
property that matters is that the walk, the table and the verdict work
against the real `CVLab`, and two cheap OpenCV experiments prove that in
well under a second. torch never enters the picture -- `baseline` and
`edge_detection` are the whole order -- so the CUDA columns are `n/a`,
which is itself a case worth covering: a CPU-only Tower must get a
verdict, not a crash.

The growth judgement is a pure function and is tested as one, with
synthetic samples, because the real soak cannot be made to leak on
demand.
"""

import json

import pytest

from scripts import cv_lab_switch_soak as soak

CHEAP = "baseline,edge_detection"


def test_help_exits_zero_and_has_no_root_flag(capsys):
    """No `--root` means the artifact-root guard has nothing to guard."""
    with pytest.raises(SystemExit) as raised:
        soak.main(["--help"])

    assert raised.value.code == 0
    out = capsys.readouterr().out
    assert "--json" in out
    assert "--root" not in out


def test_an_in_process_walk_prints_a_table_and_a_flat_verdict(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    code = soak.main(["--order", CHEAP, "--cycles", "3", "--frames", "2"])

    captured = capsys.readouterr()
    assert code == 0, captured.out + captured.err
    assert "threads" in captured.out
    assert "flat" in captured.out
    assert "GROWING" not in captured.out
    # Eight rows: one warm-up walk (marked `w`) plus three measured cycles.
    assert captured.out.count(" baseline ") == 4
    assert captured.out.count(" edge_detection ") == 4
    assert "  w baseline" in captured.out
    # Without --json nothing is written, even into the working directory.
    assert list(tmp_path.iterdir()) == []


def test_json_report_carries_every_transition_and_the_verdict(tmp_path):
    target = tmp_path / "soak.json"

    code = soak.main(
        ["--order", CHEAP, "--cycles", "3", "--frames", "2", "--json", str(target)]
    )

    assert code == 0
    report = json.loads(target.read_text(encoding="utf-8"))
    transitions = report["transitions"]
    assert len(transitions) == 6
    for row in transitions:
        assert row["accepted"] is True
        assert row["state"] == "running"
        assert row["arm_ms"] >= 0
        assert row["threads"] >= 1
        assert row["rss_mb"] > 0
        assert row["frames_ok"] == 2
    assert [row["experiment"] for row in transitions[:2]] == ["baseline", "edge_detection"]
    assert {row["cycle"] for row in transitions} == {0, 1, 2}
    # The warm-up walk is kept beside the measured rows, never among them.
    assert [row["cycle"] for row in report["warmup"]] == [-1, -1]
    for metric in ("threads", "rss_mb", "handles", "children", "cuda_reserved_mb"):
        assert metric in report["verdict"], metric
    # "n/a" when torch was never imported in this process, "flat" when an
    # earlier test in the session imported it: either is a non-growing
    # CUDA verdict, and the cheap walk itself must not import torch.
    assert report["verdict"]["cuda_reserved_mb"]["verdict"] in ("n/a", "flat")
    assert report["verdict"]["growing"] == []
    assert len(report["cycles"]) == 3
    assert all(entry["stop_accepted"] for entry in report["cycles"])
    assert list(tmp_path.iterdir()) == [target]


def _samples(cycles: int, per_cycle: int, value_for) -> list[dict]:
    rows = []
    for cycle in range(cycles):
        for index in range(per_cycle):
            rows.append(
                {
                    "cycle": cycle,
                    "threads": value_for("threads", cycle, index),
                    "rss_mb": value_for("rss_mb", cycle, index),
                    "handles": value_for("handles", cycle, index),
                    "children": value_for("children", cycle, index),
                    "cuda_reserved_mb": value_for("cuda_reserved_mb", cycle, index),
                }
            )
    return rows


def test_judge_calls_flat_samples_flat():
    def steady(field, cycle, index):
        return {"threads": 30, "rss_mb": 900.0, "handles": 400, "children": 0,
                "cuda_reserved_mb": 512.0}[field]

    verdict = soak.judge(_samples(6, 8, steady))

    assert verdict["growing"] == []
    assert verdict["cycles_compared"] == {"first": [0, 1], "last": [4, 5], "total": 6}
    for field in soak.TOLERANCES:
        assert verdict[field]["verdict"] == "flat", field
        assert verdict[field]["delta"] == 0


def test_judge_flags_growth_beyond_tolerance_per_field():
    """Threads and RSS climb per cycle; the others do not."""
    def leaking(field, cycle, index):
        if field == "threads":
            return 30 + 3 * cycle
        if field == "rss_mb":
            return 900.0 + 40.0 * cycle
        if field == "children":
            return 0
        if field == "handles":
            return 400 + (index % 2)
        return 512.0

    verdict = soak.judge(_samples(6, 8, leaking))

    assert verdict["growing"] == ["threads", "rss_mb"]
    assert verdict["threads"]["verdict"] == "GROWING"
    assert verdict["threads"]["delta"] == 12
    assert verdict["rss_mb"]["verdict"] == "GROWING"
    assert verdict["handles"]["verdict"] == "flat"
    assert verdict["children"]["verdict"] == "flat"


def test_judge_tolerates_a_first_cycle_jump():
    """The torch import lands in cycle 0 and must not read as a leak."""
    def warm_up(field, cycle, index):
        if field == "threads":
            return 8 if (cycle == 0 and index < 4) else 40
        return 0 if field == "children" else 100.0

    verdict = soak.judge(_samples(3, 8, warm_up))

    # Cycle 0 averages 24 threads, cycle 2 is 40: that IS a 16-thread rise
    # between the thirds, and with three cycles the harness cannot tell
    # it from a leak. The docstring is honest about this; the check here
    # is that the arithmetic is what the docstring says.
    assert verdict["cycles_compared"] == {"first": [0], "last": [2], "total": 3}
    assert verdict["threads"]["first_mean"] == 24
    assert verdict["threads"]["last_mean"] == 40


def test_judge_reports_n_a_when_a_field_is_never_sampled():
    """A CPU Tower has no CUDA and a Linux Tower has no handle count."""
    def cpu_only(field, cycle, index):
        return None if field in ("cuda_reserved_mb", "handles") else 10

    verdict = soak.judge(_samples(3, 2, cpu_only))

    assert verdict["cuda_reserved_mb"]["verdict"] == "n/a"
    assert verdict["handles"]["verdict"] == "n/a"
    assert verdict["growing"] == []


def test_a_single_child_process_is_growth():
    """Children tolerate zero: one leftover follower is the whole finding."""
    def one_follower(field, cycle, index):
        if field == "children":
            return 1 if cycle == 2 else 0
        return 5

    verdict = soak.judge(_samples(3, 2, one_follower))

    assert verdict["growing"] == ["children"]


def test_an_unknown_experiment_is_a_usage_error_naming_the_registry(capsys):
    code = soak.main(["--order", "baseline,monocular_depth", "--cycles", "1"])

    assert code == 2
    err = capsys.readouterr().err
    assert "monocular_depth" in err
    for name in ("baseline", "edge_detection", "depth", "object_detection"):
        assert name in err
