"""CLI-contract tests for the World Builder environment diagnostic.

Deliberately narrow. The script reports on a GPU, a driver, and a torch
build, none of which the fast suite may assume exist -- so these tests pin
only what is true on any machine: the argument contract, that the
diagnostic still succeeds when the environment it describes is degraded,
and that --strict is the only thing that turns a bad environment into a
non-zero exit.
"""

import json
import subprocess
import sys


def _run(*args):
    return subprocess.run(
        [sys.executable, "scripts/world_builder_env_check.py", *args],
        capture_output=True,
        text=True,
    )


def test_env_check_rejects_unknown_format():
    result = _run("--format", "not-a-real-format")

    assert result.returncode != 0
    assert "--format" in result.stderr


def test_env_check_succeeds_without_arguments():
    """The default run must not fail on an environment it reports as bad.

    This is the whole point of defaulting --strict off: a diagnostic that
    exits non-zero on the broken machine gets read as "the tool is broken"
    and stops being run.
    """
    result = _run()

    assert result.returncode == 0
    assert "=== Verdicts ===" in result.stdout


def test_env_check_json_is_parseable_and_carries_verdicts():
    result = _run("--format", "json")

    assert result.returncode == 0
    report = json.loads(result.stdout)
    assert {"host", "nvidia_smi", "torch", "opencv", "libraries", "verdicts"} <= set(
        report
    )
    assert report["verdicts"], "expected at least one verdict"
    for verdict in report["verdicts"]:
        assert set(verdict) == {"check", "ok", "detail"}
        assert isinstance(verdict["ok"], bool)


def test_strict_exit_code_follows_the_verdicts():
    """--strict must agree with what the report itself says.

    Pinning the relationship rather than a fixed exit code keeps this test
    honest on a machine where every verdict passes and on one where none
    do -- the failure mode being guarded against is --strict quietly
    exiting 0 while the report shows failures.
    """
    report = json.loads(_run("--format", "json").stdout)
    any_failed = any(not verdict["ok"] for verdict in report["verdicts"])

    strict = _run("--strict")

    assert (strict.returncode != 0) == any_failed


def _calibration_verdict(report):
    return next(v for v in report["verdicts"] if v["check"] == "calibration_for_the_camera")


def test_an_uncalibrated_world_root_is_a_red_verdict(tmp_path):
    """The check that would have caught a whole walk producing nothing.

    A reviewer pointed a replay at a `--root` with no `intrinsics/` beside
    it and watched 131 rebuilds produce zero poses and zero points, with a
    WARNING in a log nobody was reading. The pre-flight was all-green
    throughout: its verdicts checked that OpenCV CAN calibrate, never that
    anything HAS.
    """
    import os

    environment = dict(os.environ, TOWER_WORLD_ROOT=str(tmp_path))
    result = subprocess.run(
        [sys.executable, "scripts/world_builder_env_check.py", "--format", "json"],
        capture_output=True, text=True, env=environment,
    )
    assert result.returncode == 0
    verdict = _calibration_verdict(json.loads(result.stdout))
    assert verdict["ok"] is False
    assert str(tmp_path) in verdict["detail"], (
        "the verdict does not say WHICH directory it looked in"
    )


def test_a_calibrated_world_root_is_a_green_verdict(tmp_path):
    """And it goes green for the resolution the last capture actually sent,
    not merely because some calibration exists."""
    import os

    captures = tmp_path / "captures"
    (captures / "cap").mkdir(parents=True)
    (captures / "cap" / "frames.jsonl").write_text(
        json.dumps({"source_seq": 1, "width": 360, "height": 640,
                    "relpath": "frames/00000001.jpg"}) + "\n",
        encoding="utf-8",
    )
    worlds = tmp_path / "worlds"
    (worlds / "intrinsics").mkdir(parents=True)
    (worlds / "intrinsics" / "360x640.json").write_text("{}", encoding="utf-8")

    environment = dict(
        os.environ,
        TOWER_WORLD_ROOT=str(worlds),
        TOWER_CAPTURE_ROOT=str(tmp_path),
    )
    result = subprocess.run(
        [sys.executable, "scripts/world_builder_env_check.py", "--format", "json"],
        capture_output=True, text=True, env=environment,
    )
    verdict = _calibration_verdict(json.loads(result.stdout))
    assert verdict["ok"] is True, verdict["detail"]
    assert "360x640" in verdict["detail"]

    # And red again when the calibration is for a DIFFERENT resolution --
    # the case a "does any calibration exist" check would wave through.
    (worlds / "intrinsics" / "360x640.json").rename(
        worlds / "intrinsics" / "1280x720.json"
    )
    result = subprocess.run(
        [sys.executable, "scripts/world_builder_env_check.py", "--format", "json"],
        capture_output=True, text=True, env=environment,
    )
    verdict = _calibration_verdict(json.loads(result.stdout))
    assert verdict["ok"] is False, verdict["detail"]
    assert "360x640" in verdict["detail"] and "1280x720" in verdict["detail"]


def test_the_env_file_is_read_the_way_the_launcher_reads_it(tmp_path, monkeypatch):
    """A parser that disagrees with the launcher makes the verdict a lie in
    exactly the situation it exists for.

    `start_tower.ps1` hands uvicorn `--env-file`, and uvicorn's reader is
    python-dotenv. The first version of this check split on the first `=`
    and stripped whitespace; a reviewer diffed it against dotenv on twelve
    inputs and it disagreed on six -- `export KEY=v`, single and double
    quotes, a UTF-8 BOM, an inline `# comment`, and a quoted value with a
    space. Every disagreement produced a RED verdict against a correctly
    configured Tower.
    """
    from dotenv import dotenv_values

    worlds = tmp_path / "data" / "world_builder"
    (worlds / "intrinsics").mkdir(parents=True)

    awkward = [
        'TOWER_WORLD_ROOT=data/world_builder',
        'export TOWER_WORLD_ROOT=data/world_builder',
        'TOWER_WORLD_ROOT="data/world_builder"',
        "TOWER_WORLD_ROOT='data/world_builder'",
        'TOWER_WORLD_ROOT=data/world_builder # the world root',
        '﻿TOWER_WORLD_ROOT=data/world_builder',
    ]
    for line in awkward:
        env_file = tmp_path / ".env"
        env_file.write_text(line + "\n", encoding="utf-8")
        # `utf-8-sig`, the encoding the collector passes, and the reason
        # it does. python-dotenv 1.2.1 does NOT strip a byte-order mark:
        # read with the default `utf-8`, the last line here parses to the
        # key `'\ufeffTOWER_WORLD_ROOT'` and the lookup finds nothing --
        # a RED verdict against a correctly configured Tower, which is
        # what this whole test exists to prevent. On Windows a BOM is the
        # DEFAULT from Notepad and from PowerShell's `Out-File`, so this
        # is the ordinary case and not a hostile one.
        assert dotenv_values(
            env_file, encoding="utf-8-sig"
        ).get("TOWER_WORLD_ROOT") == "data/world_builder", (
            f"the test's own premise is wrong for {line!r}"
        )

    # And the collector resolves the same directory for every one of them.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "env_check_under_test", "scripts/world_builder_env_check.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    resolved = set()
    for line in awkward:
        (tmp_path / ".env").write_text(line + "\n", encoding="utf-8")
        monkeypatch.setattr(module, "__file__", str(tmp_path / "scripts" / "x.py"))
        monkeypatch.delenv("TOWER_WORLD_ROOT", raising=False)
        # `str`, because a parser that loses the key entirely returns
        # None here and a bare set would raise on sorting rather than
        # reporting. The BOM case does exactly that.
        resolved.add(str(module.collect_calibrations()["intrinsics_dir"]))
    assert len(resolved) == 1, (
        f"the same world root spelled six legal ways resolved {len(resolved)} "
        f"different directories: {sorted(resolved)}"
    )


def test_a_missing_dotenv_reader_says_so_instead_of_blaming_the_file(tmp_path, monkeypatch):
    """The verdict must not describe a file it can see as absent.

    The first version swallowed any reader failure into an empty dict, and
    the detail then read "there is no .env to read it from" -- about a file
    `is_file()` had confirmed two lines earlier, on a machine whose
    calibration was fine. Reachable because python-dotenv was only
    `uvicorn[standard]`'s transitive dependency; it is declared now, and
    this pins what happens if it goes missing anyway.
    """
    import builtins
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "env_check_dotenvless", "scripts/world_builder_env_check.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    (tmp_path / ".env").write_text(
        "TOWER_WORLD_ROOT=data/world_builder\n", encoding="utf-8"
    )
    monkeypatch.setattr(module, "__file__", str(tmp_path / "scripts" / "x.py"))
    monkeypatch.delenv("TOWER_WORLD_ROOT", raising=False)

    real_import = builtins.__import__

    def no_dotenv(name, *args, **kwargs):
        if name == "dotenv":
            raise ModuleNotFoundError("No module named 'dotenv'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_dotenv)
    result = module.collect_calibrations()

    assert result["covered"] is False
    assert "ModuleNotFoundError" in result["reason"], result["reason"]
    assert "there is no" not in result["reason"], (
        "the verdict blames the file for the reader's absence: " + result["reason"]
    )


def test_a_missing_dotenv_reader_does_not_override_the_environment(tmp_path, monkeypatch):
    """And it gets out of the way when the environment can answer.

    The first fix for the swallowed failure returned early with the
    exception named -- honest, and also wrong: an operator who exports
    `TOWER_WORLD_ROOT` needs no `.env` at all, and the early return skipped
    the env-var lookup, so a correctly configured machine went RED anyway.
    A reviewer measured both versions against the same environment and the
    behaviour being replaced was right about this case.
    """
    import builtins
    import importlib.util

    worlds = tmp_path / "worlds"
    (worlds / "intrinsics").mkdir(parents=True)
    (worlds / "intrinsics" / "360x640.json").write_text("{}", encoding="utf-8")
    captures = tmp_path / "captures"
    (captures / "cap").mkdir(parents=True)
    (captures / "cap" / "frames.jsonl").write_text(
        json.dumps({"source_seq": 1, "width": 360, "height": 640,
                    "relpath": "frames/00000001.jpg"}) + chr(10),
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text("# nothing useful here" + chr(10),
                                   encoding="utf-8")

    spec = importlib.util.spec_from_file_location(
        "env_check_env_wins", "scripts/world_builder_env_check.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "__file__", str(tmp_path / "scripts" / "x.py"))
    monkeypatch.setenv("TOWER_WORLD_ROOT", str(worlds))
    monkeypatch.setenv("TOWER_CAPTURE_ROOT", str(tmp_path))

    real_import = builtins.__import__

    def no_dotenv(name, *args, **kwargs):
        if name == "dotenv":
            raise ModuleNotFoundError("No module named 'dotenv'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_dotenv)
    result = module.collect_calibrations()

    assert result["covered"] is True, (
        "a machine configured entirely through the environment went red "
        "because a file it does not need could not be parsed: "
        + str(result.get("reason"))
    )


def test_python_dotenv_is_a_declared_dependency():
    """Not borrowed from `uvicorn[standard]`'s extras. The pre-flight's
    verdict is only about the Tower that will run if it reads the same
    `.env` uvicorn will."""
    import pathlib

    text = pathlib.Path("pyproject.toml").read_text(encoding="utf-8")
    dependencies = text.split("[project.optional-dependencies]")[0]
    assert "python-dotenv" in dependencies, (
        "python-dotenv is not in the base dependencies; a venv built with "
        "plain uvicorn would make the calibration verdict a silent lie"
    )


def _env_check_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "env_check_backend", "scripts/world_builder_env_check.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stub_report():
    """The keys `build_verdicts` reads, with nothing interesting in them.

    The backend verdict is about the interpreter, not about the host, so
    these tests must not depend on a GPU being present or on what
    `nvidia-smi` says.
    """
    return {
        "nvidia_smi": {"available": False, "reason": "stub"},
        "torch": {"installed": False},
        "opencv": {"installed": False},
        "libraries": {},
        "vocabulary_tree": {},
        "package_origin": {},
        "calibrations": {},
    }


def test_the_preflight_refuses_an_interpreter_that_cannot_solve(monkeypatch):
    """Every other verdict can pass on a Python that reconstructs nothing.

    `import tower` works from the tower directory whether or not the
    package is installed, torch and OpenCV are commonly present
    system-wide, and the vocabulary tree lives in `~/.cache` -- so a
    green pre-flight is achievable on an interpreter with no pycolmap,
    and a walk on it produces zero poses and zero points, announced
    nowhere.

    **The agent that wrote this check ran the whole Tower suite on such
    an interpreter for a working session before noticing**, on this
    machine, with `tower/.venv` sitting beside it. That is who this
    verdict is for.
    """
    import builtins

    module = _env_check_module()
    real_import = builtins.__import__

    def no_pycolmap(name, *args, **kwargs):
        if name == "pycolmap":
            raise ModuleNotFoundError("No module named 'pycolmap'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pycolmap)
    verdicts = {
        name: (ok, detail) for name, ok, detail in module.build_verdicts(_stub_report())
    }

    assert "sfm_backend_importable" in verdicts, sorted(verdicts)
    ok, detail = verdicts["sfm_backend_importable"]
    assert ok is False, detail
    assert "zero poses and zero points" in detail
    # It must name the interpreter, because "why is it failing" is almost
    # always "you are not running the Python you think you are".
    assert module.sys.executable in detail


def test_the_preflight_passes_the_backend_check_where_it_can_solve():
    """The other half: this suite runs on an interpreter that CAN solve.

    If this ever fails, the suite itself is being run on the wrong Python
    and every reconstruction result it reports is about a backend that is
    not the one the Tower uses.
    """
    module = _env_check_module()
    verdicts = {
        name: (ok, detail) for name, ok, detail in module.build_verdicts(_stub_report())
    }
    ok, detail = verdicts["sfm_backend_importable"]
    assert ok is True, (
        "the test suite is running on an interpreter with no SfM backend: "
        + detail
    )
