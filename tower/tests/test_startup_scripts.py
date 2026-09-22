"""Contract tests for the two PowerShell startup scripts.

These scripts cannot be executed to completion from a test -- one installs
packages, the other blocks on a server -- so what is pinned here is
everything that can be checked without running them: that they parse, and
that each of the specific mistakes the 2026-08-24 physical run tripped over
is still absent from the source.

Every assertion below corresponds to a failure that actually happened or
that would silently produce a wrong-looking-but-running tower. None of them
are style checks.
"""

import re
import subprocess
from pathlib import Path

import pytest

TOWER_ROOT = Path(__file__).resolve().parent.parent
SETUP_SCRIPT = TOWER_ROOT / "scripts" / "setup_tower.ps1"
START_SCRIPT = TOWER_ROOT / "scripts" / "start_tower.ps1"
CONFIG_MODULE = TOWER_ROOT / "tower" / "config.py"

_PARSE_COMMAND = (
    "$errs = $null; "
    "$null = [System.Management.Automation.Language.Parser]::ParseFile("
    "'{path}', [ref]$null, [ref]$errs); "
    "if ($errs.Count) {{ $errs | ForEach-Object {{ $_.ToString() }}; exit 1 }}"
)


def _parse_powershell(path):
    """Run the PowerShell parser over a file; return the CompletedProcess."""
    return subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            _PARSE_COMMAND.format(path=str(path)),
        ],
        capture_output=True,
        text=True,
    )


def _source(path):
    return path.read_text(encoding="utf-8")


def _executable_lines(path):
    """Source lines with whole-line PowerShell comments removed.

    Several invariants below are about what the script *does*, and a
    comment explaining a hazard must not read as committing it.
    """
    lines = []
    for line in _source(path).splitlines():
        if line.lstrip().startswith("#"):
            continue
        lines.append(line)
    return lines


def test_the_parse_helper_actually_detects_a_syntax_error(tmp_path):
    """Guard the guard.

    A parser check that silently passes everything is worse than no check:
    it would let a broken script through while reporting success. So prove
    the helper fails on known-bad input before trusting it on the real
    scripts.
    """
    broken = tmp_path / "broken.ps1"
    broken.write_text("if ($true) { Write-Host 'unclosed'\n", encoding="utf-8")

    result = _parse_powershell(broken)

    assert result.returncode != 0, "the parser helper accepted an unbalanced brace"
    assert result.stdout.strip(), "a parse failure must say what is wrong"


@pytest.mark.parametrize("script", [SETUP_SCRIPT, START_SCRIPT])
def test_script_exists_and_is_valid_powershell(script):
    """A script that will not parse fails at the worst possible moment.

    Namely: in front of a phone that is already streaming, in a session
    that was supposed to be about something else.
    """
    assert script.is_file(), f"{script} is missing"

    result = _parse_powershell(script)

    assert result.returncode == 0, f"parse errors in {script.name}:\n{result.stdout}"


def test_start_script_targets_tower_main_app_and_never_the_factory():
    """--factory would load the module container twice.

    tower/main.py calls create_app() at module scope, and create_app() runs
    the container's load_and_start(). Handing uvicorn the factory form makes
    it call create_app() again, so two containers exist and only one of them
    is the one serving requests.
    """
    executable = "\n".join(_executable_lines(START_SCRIPT))

    assert "tower.main:app" in executable
    assert "--factory" not in executable


def test_start_script_passes_host_explicitly():
    """uvicorn's own default host is 127.0.0.1, not 0.0.0.0.

    Omit --host and the tower binds loopback-only: every local check passes,
    /health answers from the browser on the same machine, and the phone
    cannot connect at all with nothing in the log to explain why.
    """
    executable = "\n".join(_executable_lines(START_SCRIPT))

    assert "'--host', $BindHost" in executable, (
        "the uvicorn argument list must pass --host explicitly"
    )
    assert "[string]$BindHost = '0.0.0.0'" in executable, (
        "the default must be the LAN-reachable bind, not uvicorn's loopback default"
    )


def test_start_script_keeps_the_windows_listener_alive():
    """--loop routes uvicorn onto the resilient Proactor loop.

    Without it, uvicorn takes asyncio.ProactorEventLoop on Windows, whose
    accept path (CPython gh-93821) closes the listening socket and never
    re-arms it when a backlogged connection is reset before it is accepted.
    A reconnecting phone does that routinely; the port then vanishes while
    the process lives, which is what a physical test hit. --loop asyncio
    would NOT fix it (still a Proactor loop on win32), so the factory must
    be named explicitly.
    """
    executable = "\n".join(_executable_lines(START_SCRIPT))

    assert "'--loop', 'tower.serve_loop:resilient_loop_factory'" in executable, (
        "the uvicorn argument list must route onto tower/serve_loop.py, or a "
        "reset connection will take port 8000 down while the process stays up"
    )


def test_start_script_bounds_shutdown():
    """--timeout-graceful-shutdown turns an unbounded hang into a bounded exit.

    uvicorn's default (None) makes Ctrl-C wait forever for a stalled
    websocket transport to drain -- a phone suspended with the socket open --
    and a second Ctrl-C does not help. That is the "INFO: Shutting down" that
    then never completed in the field.
    """
    executable = "\n".join(_executable_lines(START_SCRIPT))

    assert "'--timeout-graceful-shutdown'" in executable, (
        "shutdown must be bounded, or a stalled connection hangs Ctrl-C forever"
    )


def test_start_script_names_bindhost_not_host():
    """$Host is an automatic PowerShell variable.

    A -Host parameter shadows it inside the script, which breaks anything
    reading $Host and produces confusing errors far from the cause.
    """
    source = _source(START_SCRIPT)

    assert "[string]$BindHost" in source
    assert not re.search(r"^\s*\[string\]\$Host\b", source, re.MULTILINE)


@pytest.mark.parametrize("script", [SETUP_SCRIPT, START_SCRIPT])
def test_firewall_rule_is_printed_never_executed(script):
    """README.md forbids automating firewall changes.

    Opening an inbound port is a decision about the machine's exposure, and
    it belongs to the person at the keyboard. The scripts may report the
    rule's state and print the command; they must never run it.
    """
    for line in _executable_lines(script):
        if "New-NetFirewallRule" not in line:
            continue
        assert re.search(r"Write-(Host|Output|Detail|Line|Problem)", line), (
            "New-NetFirewallRule appears outside a print context:\n" + line
        )


@pytest.mark.parametrize("script", [SETUP_SCRIPT, START_SCRIPT])
def test_no_bare_interpreter_invocations(script):
    """Bare `pip`/`python`/`pytest` resolve to whatever is on PATH.

    On this machine that is a real hazard: an unactivated shell reaches the
    system interpreter, and an activated one reaches a venv that may not be
    the project's. Every invocation must name .venv\\Scripts\\python.exe, so
    the scripts never depend on an activation step nobody remembers to run.
    """
    pattern = re.compile(r"^\s*&?\s*(pip|python|pytest)(?![\w.\-])", re.IGNORECASE)

    offenders = [line for line in _executable_lines(script) if pattern.match(line)]

    assert not offenders, "bare interpreter invocation:\n" + "\n".join(offenders)


@pytest.mark.parametrize("script", [SETUP_SCRIPT, START_SCRIPT])
def test_the_venv_interpreter_is_the_one_named(script):
    """The interpreter path is spelled out, not inherited from the shell.

    Either literally or composed from the .venv directory -- what matters is
    that no step depends on an Activate.ps1 nobody remembers to run.
    """
    source = _source(script)

    assert "'.venv'" in source or r".venv\Scripts" in source
    assert r"Scripts\python.exe" in source


@pytest.mark.parametrize("script", [SETUP_SCRIPT, START_SCRIPT])
def test_every_install_goes_through_a_module_invocation(script):
    """`-m pip` binds the install to a named interpreter; bare `pip` does not.

    A bare `pip install -e .` in an unactivated shell installs the project
    into the system Python and leaves the venv untouched -- which then looks
    exactly like the install having failed for no reason.
    """
    for line in _executable_lines(script):
        if "pip install" not in line:
            continue
        assert "-m pip install" in line, "install not bound to an interpreter:\n" + line


def test_setup_pins_the_312_launcher():
    """Bare `py` resolves to Python 3.14 on this machine.

    A 3.14 venv installs cleanly enough to look fine and then fails later on
    a wheel that has no 3.14 build. The pin is the only thing standing
    between a fresh checkout and that afternoon.
    """
    source = _source(SETUP_SCRIPT)

    assert "py -3.12" in source

    invocations = re.findall(r"&\s*py\s+(\S+)", source)
    assert invocations, "expected the launcher to be invoked through the call operator"
    for first_argument in invocations:
        assert first_argument == "-3.12", (
            f"unpinned launcher invocation: & py {first_argument}"
        )

    assert not re.search(r"^\s*py\s", source, re.MULTILINE)


def _config_env_names():
    """Every TOWER_* variable tower/config.py actually reads."""
    return set(re.findall(r'"(TOWER_[A-Z0-9_]+)"', _source(CONFIG_MODULE)))


@pytest.mark.parametrize("script", [SETUP_SCRIPT, START_SCRIPT])
def test_scripts_only_name_variables_the_config_reads(script):
    """This is the test that stops the scripts and the config drifting apart.

    A script that writes TOWER_WORLD_ROOTS, or keeps naming a variable after
    config.py stops reading it, produces a tower that starts fine and
    behaves as if the setting were never given. Nothing else in the system
    notices.
    """
    known = _config_env_names()
    assert known, "failed to parse any TOWER_* names out of tower/config.py"

    used = set(re.findall(r"\bTOWER_[A-Z0-9_]+\b", _source(script)))

    assert used <= known, f"names config.py never reads: {sorted(used - known)}"


def test_env_template_writes_exactly_the_variables_that_do_something():
    """The generated .env must carry the two forced roots and no dead vars.

    TOWER_CAPTURE_ROOT and TOWER_WORLD_ROOT are the two whose absence is
    invisible at startup and fatal to the run: no recording, and a World
    Builder that iOS reports as unsupported. TOWER_HOST and TOWER_PORT are
    the opposite case -- config.py reads them into Settings and nothing ever
    reads them back, so writing them into a template would institutionalise
    a setting that binds nothing.
    """
    assignments = dict(
        re.findall(r"'(TOWER_[A-Z0-9_]+)=([^']*)'", _source(SETUP_SCRIPT))
    )

    assert assignments.get("TOWER_CAPTURE_ROOT") == "data", (
        "capture.py appends captures/<id>, so the root is data"
    )
    assert assignments.get("TOWER_WORLD_ROOT") == "data/world_builder", (
        "store.py appends worlds/<id>, and this must equal DEFAULT_ROOT in "
        "scripts/world_build_session.py"
    )
    assert "TOWER_HOST" not in assignments
    assert "TOWER_PORT" not in assignments


def test_world_root_template_matches_the_session_scripts_default():
    """One source of truth for where worlds live.

    If the .env template and scripts/world_build_session.py disagree, the
    result channel reads a different tree than the builder writes, and the
    symptom is an empty world rather than an error.
    """
    session_source = (TOWER_ROOT / "scripts" / "world_build_session.py").read_text(
        encoding="utf-8"
    )
    default_root = re.search(r'DEFAULT_ROOT\s*=\s*Path\("([^"]+)"\)', session_source)
    assert default_root, "could not read DEFAULT_ROOT from world_build_session.py"

    assert f"TOWER_WORLD_ROOT={default_root.group(1)}" in _source(SETUP_SCRIPT)


@pytest.mark.parametrize("script", [SETUP_SCRIPT, START_SCRIPT])
def test_scripts_relocate_to_the_tower_root(script):
    """redaction.py resolves the YuNet weights relative to the CWD.

    Run the server from scripts/, or from wherever the terminal happened to
    be, and the weights are not found -- so face redaction is disabled and
    every keyframe records its redaction as "none". Nothing fails; the
    privacy guarantee just quietly stops holding.
    """
    source = _source(script)

    assert "Split-Path -Parent $PSScriptRoot" in source
    assert re.search(r"^Set-Location -LiteralPath \$root$", source, re.MULTILINE)


def test_start_script_does_not_send_anyone_to_a_second_terminal():
    """The Tower launches the World Builder follower itself.

    The whole point of this workstream is that ordinary development is one
    command in one terminal. A start script that ships a follower script, or
    hands the operator a world_build_session.py command line to run, puts
    the second terminal straight back. Naming that module as the source of
    a path constant is fine; printing it as something to launch is not.
    """
    source = _source(START_SCRIPT)

    assert "start_world_follower" not in source

    for line in source.splitlines():
        if "world_build_session" not in line:
            continue
        assert "python" not in line.lower(), (
            "world_build_session.py is presented as a command to run:\n" + line
        )
