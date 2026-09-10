#!/usr/bin/env python
"""Read-only World Builder readiness diagnostic for this Tower.

Reports, in one place, every environment capability a monocular World
Builder pipeline would depend on: GPU visibility, whether torch can
actually reach that GPU, which OpenCV geometry/calibration/feature
primitives this build ships, and which optional acceleration libraries
are present.

Why this exists separately from ``scripts/verify_cuda.py``: that script
answers one question (can torch see CUDA) and raises SystemExit when the
answer is no. This one answers "what can this machine actually do for
World Builder", installs nothing, writes nothing, and by default exits 0
even when the answer is "not much" -- a diagnostic is only useful if it
still runs on the broken machine it is diagnosing.

Not part of the pytest suite's real work: the suite may not assume a GPU,
a driver, or a particular torch build. ``tests/test_world_builder_env_check_cli.py``
pins only the CLI contract.

    .venv\\Scripts\\python.exe scripts/world_builder_env_check.py
    .venv\\Scripts\\python.exe scripts/world_builder_env_check.py --format json
    .venv\\Scripts\\python.exe scripts/world_builder_env_check.py --strict
"""

import argparse
import importlib
import json
import pathlib
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# The sibling scripts all do this and this one did not, which is how it
# imported a DIFFERENT checkout's `tower` -- see `collect_package_origin`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# OpenCV symbols a monocular pipeline actually calls, grouped by role so a
# partial build reports which capability its absence costs us, rather than
# just a bare symbol name.
OPENCV_REQUIREMENTS = {
    "features": (
        "ORB_create",
        "SIFT_create",
    ),
    "matching": (
        "BFMatcher",
        "FlannBasedMatcher",
    ),
    "flow": (
        "calcOpticalFlowPyrLK",
        "calcOpticalFlowFarneback",
        "DISOpticalFlow_create",
    ),
    "geometry": (
        "findEssentialMat",
        "findFundamentalMat",
        "findHomography",
        "recoverPose",
        "decomposeEssentialMat",
        "triangulatePoints",
        "solvePnPRansac",
        "correctMatches",
    ),
    "calibration": (
        "calibrateCamera",
        "findChessboardCornersSB",
        "initCameraMatrix2D",
        "undistort",
    ),
}

# Optional libraries. Present/absent is informative either way -- absence
# is not a failure here, it is a cost estimate for a future decision.
OPTIONAL_LIBRARIES = (
    "torch",
    "torchvision",
    "timm",
    "scipy",
    "kornia",
    "open3d",
    "onnxruntime",
    "tensorrt",
    "numba",
    "cupy",
    "transformers",
)


def _module_version(name):
    """Import ``name`` and return its version string, or None if absent."""
    try:
        module = importlib.import_module(name)
    except Exception:
        return None
    return str(getattr(module, "__version__", "(no __version__)"))


def collect_host():
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def collect_nvidia_smi():
    """Query the driver directly.

    Deliberately independent of any Python build: the GPU can be perfectly
    healthy while torch cannot reach it, and distinguishing those two
    cases is the main thing this diagnostic exists to do.
    """
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return {"available": False, "reason": "nvidia-smi not on PATH"}

    query = "name,driver_version,memory.total,memory.free,compute_cap"
    try:
        completed = subprocess.run(
            [exe, "--query-gpu=" + query, "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    if completed.returncode != 0:
        return {
            "available": False,
            "reason": f"nvidia-smi exit {completed.returncode}",
        }

    gpus = []
    for line in completed.stdout.strip().splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            continue
        gpus.append(
            {
                "name": fields[0],
                "driver_version": fields[1],
                "memory_total": fields[2],
                "memory_free": fields[3],
                "compute_capability": fields[4],
            }
        )
    return {"available": bool(gpus), "gpus": gpus}


def collect_torch():
    try:
        import torch
    except Exception as exc:
        return {"installed": False, "reason": f"{type(exc).__name__}: {exc}"}

    info = {
        "installed": True,
        "version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": 0,
        "arch_list": [],
        "device_name": None,
        "device_capability": None,
    }

    # A CPU-only wheel can raise from get_arch_list() rather than returning
    # an empty list, so this is guarded rather than assumed.
    try:
        info["arch_list"] = list(torch.cuda.get_arch_list())
    except Exception:
        info["arch_list"] = []

    if info["cuda_available"]:
        info["device_count"] = torch.cuda.device_count()
        try:
            info["device_name"] = torch.cuda.get_device_name(0)
            info["device_capability"] = list(torch.cuda.get_device_capability(0))
        except Exception as exc:
            info["device_query_error"] = f"{type(exc).__name__}: {exc}"

    return info


def collect_opencv():
    try:
        import cv2
    except Exception as exc:
        return {"installed": False, "reason": f"{type(exc).__name__}: {exc}"}

    info = {"installed": True, "version": cv2.__version__, "capabilities": {}}

    for capability, symbols in OPENCV_REQUIREMENTS.items():
        present = [name for name in symbols if hasattr(cv2, name)]
        missing = [name for name in symbols if not hasattr(cv2, name)]
        info["capabilities"][capability] = {
            "complete": not missing,
            "present": present,
            "missing": missing,
        }

    # ChArUco is the self-calibration path that needs no new dependency, so
    # it is reported explicitly rather than left implicit inside `aruco`.
    aruco = getattr(cv2, "aruco", None)
    info["charuco"] = {
        "aruco_module": aruco is not None,
        "CharucoBoard": hasattr(aruco, "CharucoBoard") if aruco else False,
        "CharucoDetector": hasattr(aruco, "CharucoDetector") if aruco else False,
    }

    try:
        info["cuda_device_count"] = cv2.cuda.getCudaEnabledDeviceCount()
    except Exception:
        info["cuda_device_count"] = 0

    return info


def collect_libraries():
    return {name: _module_version(name) for name in OPTIONAL_LIBRARIES}


def collect_package_origin():
    """WHERE THE INSTALL POINTS, not where this script happens to import from.

    Asking `tower.__file__` in THIS process answers nothing: the line at the
    top of this file puts the canonical checkout on `sys.path`, so the
    answer is always "here". The hazard is the editable install underneath
    it, which on this machine maps `tower` to
    `Glasses-worktrees/all-cartridges-field-test-v1/tower/tower` -- a
    different checkout, months of fixes behind.

    Every entry point that sets `sys.path` (this script, and the three
    world_* scripts) wins over that mapping. `tower/main.py` sets none, and
    `import tower` from a neutral working directory therefore loads the
    OTHER checkout -- verified by running it. `scripts/start_tower.ps1`
    happens to `Set-Location` to the tower root first, so the supported
    launcher is safe; a hand-run `python -m uvicorn tower.main:app` from
    anywhere else is not.

    A physical test conducted against a stale worktree would produce
    results about code nobody edited, and nothing would say so.
    """
    here = pathlib.Path(__file__).resolve().parents[1]
    record = {"expected": str(here), "resolved_here": None, "install_points_at": None}
    try:
        import tower
        record["resolved_here"] = str(pathlib.Path(tower.__file__).resolve().parents[1])
    except Exception as exc:  # noqa: BLE001
        record["reason"] = f"{type(exc).__name__}: {exc}"
        return record
    # The editable install's own mapping, read without importing it.
    try:
        import glob
        import re

        # Ask the interpreter where its site-packages is, rather than
        # deriving it from the package path -- the first version of this
        # guessed `parents[2]/.venv` and landed one directory above the
        # venv, so it found no mapping and reported OK for the exact
        # condition it exists to catch.
        finders = []
        for entry in sys.path:
            if entry.endswith("site-packages"):
                finders.extend(glob.glob(str(pathlib.Path(entry) / "__editable__*finder.py")))
        for finder in finders:
            text = pathlib.Path(finder).read_text(encoding="utf-8")
            match = re.search(r"'tower':\s*'([^']+)'", text)
            if match:
                mapped = pathlib.Path(match.group(1).replace("\\\\", "\\")).resolve()
                record["install_points_at"] = str(mapped.parent)
                break
    except Exception:  # noqa: BLE001 -- absence is an answer, not an error
        pass
    record["matches"] = (
        record["install_points_at"] is None
        or pathlib.Path(record["install_points_at"]) == here
    )
    return record


def collect_vocabulary_tree():
    """Whether COLMAP's vocabulary tree is already on this machine.

    IT IS ON THE LIVE PATH NOW. Loop detection used to run only in the
    finalisation solve, and `global_solve` justified that partly as avoiding
    "a network dependency". Since 2026-09-09 every background solve asks for
    it, because that is what makes a live world converge instead of
    fragmenting -- measured on the field capture, 16 components down to 5,
    and the largest component's share of posed keyframes rising 0.75 -> 0.95
    as the walk went on.

    So the network dependency moved onto the walk. pycolmap downloads a
    72 MB tree on first use and caches it in the USER's home, not the venv,
    so a fresh checkout on a warm machine is fine and a fresh machine is
    not. A cold cache during a walk means the first solve child spends the
    download inside its own 120 s budget and is terminated if it overruns:
    the session survives, no solution lands, and the world quietly stays in
    pieces with nothing on screen saying why.

    That is precisely the class of failure this pre-flight exists to catch
    before someone puts the glasses on.
    """
    # The SAME function the solver gates on, so this check cannot disagree
    # with the thing it is checking.
    from tower.world_builder.global_solve import vocabulary_tree_cache_dir

    home = vocabulary_tree_cache_dir()
    trees = sorted(home.glob("*vocab_tree*")) if home.is_dir() else []
    return {
        "cache_dir": str(home),
        "present": bool(trees),
        "files": [{"name": t.name, "bytes": t.stat().st_size} for t in trees],
    }


def collect_calibrations():
    r"""Whether a calibration exists for the resolution the glasses deliver.

    THE PRE-FLIGHT WAS ALL-GREEN AND MISSED THIS. A reviewer pointed a
    replay at a fresh `--root`, which had no `intrinsics/` beside it, and
    watched a whole walk produce nothing:

        world builder: no calibration for 360x640 (looked for
          ...\intrinsics\360x640.json); intrinsics stay unknown and no
          poses will be solved
        world builder: backend selected unposed -- will produce no poses
          and no points
        rebuild 131: 499 keyframes -> 0 positioned poses, 0 points

    131 rebuilds, zero geometry, and a WARNING in a log nobody is reading.
    The seven verdicts this script already emits check that OpenCV *can*
    calibrate; none of them check that anything *has*.

    The resolution cannot be known before the phone connects, so this asks
    the last capture what it sent. That is the best available evidence and
    it is real evidence: every frame of the five most recent captures on
    this host is 360x640.
    """
    from tower.config import get_settings

    settings = get_settings()

    # THE SAME `.env` THE LAUNCHER HANDS UVICORN.
    #
    # `world_root` and `capture_root` are None unless the environment sets
    # them, and this script runs BEFORE `start_tower.ps1` -- that is the
    # point of a pre-flight. The launcher passes `--env-file .env`, so the
    # file beside it is what the Tower will actually use. Reading it here
    # is how this check can be about the Tower that is going to run rather
    # than about this shell. An env var already set still wins, exactly as
    # it does for uvicorn.
    tower_root = pathlib.Path(__file__).resolve().parents[1]
    env_path = tower_root / ".env"
    env = {}
    env_error = None
    if env_path.is_file():
        # PYTHON-DOTENV, NOT A HAND-ROLLED PARSER.
        #
        # The first version split on the first `=` and stripped
        # whitespace, and a reviewer diffed it against dotenv on twelve
        # inputs: it disagreed on SIX. `export KEY=v` filed the key as
        # "export KEY"; quotes were kept as part of the value; a UTF-8 BOM
        # left the key unreachable (`str.strip()` does not remove ﻿,
        # which is Cf, not whitespace); and an inline `# comment` became
        # part of the path. Every disagreement pointed the same way -- a
        # RED verdict saying "every pose and every point of the walk will
        # be missing" against a Tower that is correctly configured, which
        # is a lie told in exactly the situation this check exists for.
        #
        # `start_tower.ps1` hands uvicorn `--env-file`, and uvicorn's
        # reader is dotenv. Using the same library is the only way this
        # check can be about the Tower that is going to run.
        try:
            from dotenv import dotenv_values

            # `utf-8-sig`, NOT the default `utf-8`.
            #
            # python-dotenv 1.2.1 does not strip a byte-order mark: a
            # `.env` beginning EF BB BF parses to the key
            # `'﻿TOWER_WORLD_ROOT'`, and the lookup below then finds
            # nothing and reports a correctly configured Tower as unset --
            # a RED verdict of exactly the kind the paragraph above says
            # this must never produce. On WINDOWS that is the default
            # outcome, not an exotic one: Notepad and PowerShell's
            # `Out-File` both write UTF-8 with a BOM, and this repository's
            # own CLAUDE.md warns about it. Measured here on 1.2.1; the
            # suite's own premise check caught it.
            #
            # `utf-8-sig` consumes a BOM if there is one and is identical
            # to `utf-8` if there is not.
            env = {
                k: v
                for k, v in dotenv_values(env_path, encoding="utf-8-sig").items()
                if v is not None
            }
        except Exception as exc:  # noqa: BLE001 - a diagnostic never fails on its input
            # NAME IT, AND THEN GET OUT OF THE WAY.
            #
            # The first version swallowed this into an empty dict and the
            # verdict then read "there is no .env to read it from" -- about
            # a file `is_file()` had confirmed two lines earlier. The
            # second version returned early with the exception named, which
            # was honest and ALSO WRONG: an operator who exports
            # TOWER_WORLD_ROOT needs no `.env` at all, and the early return
            # skipped the env-var lookup below, so a machine that was
            # correctly configured went RED anyway. A reviewer measured
            # both, and the old fallthrough was right about that case.
            #
            # So: remember the failure, keep going, and let it into the
            # verdict only if the environment could not answer either.
            env_error = f"{type(exc).__name__}: {exc}"
            env = {}

    def _root(configured, key):
        raw = configured if configured is not None else env.get(key)
        if raw is None:
            return None
        path = pathlib.Path(raw)
        return path if path.is_absolute() else (tower_root / path)

    world_root = _root(settings.world_root, "TOWER_WORLD_ROOT")
    capture_root = _root(settings.capture_root, "TOWER_CAPTURE_ROOT")
    if capture_root is not None:
        # `tower/capture.py` appends `captures/<id>`, which `.env` says in
        # its own comment; the root in the file is one level above.
        capture_root = capture_root / "captures"
    if world_root is None:
        # The `.env` sitting there unread is a DIFFERENT failure from no
        # `.env` at all, and the first version of this sentence said the
        # second about the first.
        if env_error is not None:
            detail = (
                f"no world root: TOWER_WORLD_ROOT is unset and {env_path} "
                f"could not be read ({env_error})"
            )
        elif env_path.is_file():
            detail = (
                f"no world root: TOWER_WORLD_ROOT is unset and {env_path} "
                "does not set it"
            )
        else:
            detail = (
                "no world root: TOWER_WORLD_ROOT is unset and there is no "
                f"{env_path} to read it from"
            )
        return {"intrinsics_dir": None, "available": [], "last_capture": None,
                "last_capture_resolution": None, "covered": False,
                "reason": detail}

    directory = world_root / "intrinsics"
    available = sorted(path.stem for path in directory.glob("*.json")) \
        if directory.is_dir() else []

    last, observed = None, None
    if capture_root is not None and capture_root.is_dir():
        captures = sorted(
            (path for path in capture_root.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for capture in captures:
            journal = capture / "frames.jsonl"
            if not journal.is_file():
                continue
            try:
                with journal.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        record = json.loads(line)
                        width, height = record.get("width"), record.get("height")
                        if width and height:
                            last, observed = capture.name, f"{width}x{height}"
                            break
            except (OSError, ValueError):
                continue
            if observed:
                break

    return {
        # Published so `build_verdicts` can ask about locks without
        # resolving the root a second way -- two resolvers for one path is
        # how this campaign's defects kept coming back.
        "world_root": str(world_root),
        "intrinsics_dir": str(directory),
        "available": available,
        "last_capture": last,
        "last_capture_resolution": observed,
        "covered": bool(observed) and observed in available,
    }


def build_verdicts(report):
    """Turn raw facts into the few go/no-go statements an implementer needs.

    Returns a list of ``(key, ok, detail)`` tuples.
    """
    verdicts = []

    nvidia = report["nvidia_smi"]
    verdicts.append(
        (
            "gpu_visible",
            bool(nvidia.get("available")),
            (
                nvidia["gpus"][0]["name"]
                if nvidia.get("gpus")
                else nvidia.get("reason", "no GPU reported by driver")
            ),
        )
    )

    torch_info = report["torch"]
    torch_cuda_ok = bool(torch_info.get("cuda_available"))
    if not torch_info.get("installed"):
        torch_detail = "torch not installed"
    elif torch_cuda_ok:
        capability = torch_info.get("device_capability") or []
        torch_detail = (
            f"torch {torch_info['version']} -> {torch_info.get('device_name')} "
            f"sm_{''.join(str(value) for value in capability)}"
        )
    else:
        torch_detail = (
            f"torch {torch_info['version']} cannot reach CUDA "
            f"(cuda_build={torch_info.get('cuda_build')})"
        )
    verdicts.append(("torch_cuda_usable", torch_cuda_ok, torch_detail))

    # See collect_vocabulary_tree: loop detection is on every live solve now,
    # and a cold cache turns the first solve of a walk into a 72 MB download.
    vocab = report.get("vocabulary_tree") or {}
    if vocab.get("present"):
        total = sum(f["bytes"] for f in vocab["files"])
        vocab_detail = (
            f"{len(vocab['files'])} tree(s), {total / 1e6:.0f} MB, in {vocab['cache_dir']}"
        )
    else:
        vocab_detail = (
            f"absent from {vocab.get('cache_dir')}; loop detection will be OFF "
            "for every solve, so the world will come out in more pieces than it "
            "should. Warm it (needs a network, ~72 MB): .venv/Scripts/python.exe "
            "-c \"import pycolmap; pycolmap.match_vocabtree\" then run any solve "
            "with --loop-detection on a machine that can reach github.com."
        )
    verdicts.append(("vocabulary_tree_cached", bool(vocab.get("present")), vocab_detail))

    # See collect_package_origin. A stale editable install is invisible
    # until it produces results about code nobody edited.
    origin = report.get("package_origin") or {}
    points_at = origin.get("install_points_at")
    if origin.get("reason"):
        origin_detail = origin["reason"]
    elif points_at is None:
        origin_detail = f"no editable install mapping found; tower imports from {origin.get('resolved_here')}"
    elif origin.get("matches"):
        origin_detail = f"the editable install points at {points_at}"
    else:
        origin_detail = (
            f"THE EDITABLE INSTALL POINTS AT {points_at}, not {origin['expected']}. "
            "Scripts that set sys.path are unaffected, but `import tower` from any "
            "other working directory loads that checkout -- tower/main.py sets no "
            "sys.path. start_tower.ps1 Set-Locations to the tower root first, so use "
            "it, or reinstall: .venv/Scripts/python.exe -m pip install -e ."
        )
    verdicts.append(("tower_package_is_this_checkout", bool(origin.get("matches")), origin_detail))

    # See collect_calibrations. A missing calibration is not a degraded
    # walk, it is a walk with no geometry at all, announced only in a log.
    calib = report.get("calibrations") or {}
    observed = calib.get("last_capture_resolution")
    available = calib.get("available") or []
    if calib.get("reason"):
        calib_detail = calib["reason"]
    elif not available:
        calib_detail = f"no calibrations at all under {calib.get('intrinsics_dir')}"
    elif observed is None:
        calib_detail = (
            f"have {', '.join(available)}; no capture on this host says what "
            "resolution the glasses send, so this cannot be checked"
        )
    elif calib.get("covered"):
        calib_detail = (
            f"{observed} is calibrated (last capture {calib.get('last_capture')})"
        )
    else:
        calib_detail = (
            f"the last capture sent {observed} and there is no "
            f"{observed}.json; have {', '.join(available)}. Every pose and "
            "every point of the walk will be missing"
        )
    verdicts.append(("calibration_for_the_camera", bool(calib.get("covered")), calib_detail))

    # THE INTERPRETER RUNNING THIS MUST BE ABLE TO SOLVE, and nothing
    # above establishes that.
    #
    # Every other verdict here can pass on a Python that cannot
    # reconstruct anything. `import tower` works from the tower directory
    # whether or not the package is installed, torch and OpenCV are
    # commonly present system-wide, and the vocabulary tree lives in
    # ~/.cache -- so a green pre-flight is achievable on an interpreter
    # with no pycolmap at all, and a walk on it produces a world with
    # zero poses and zero points, announced nowhere.
    #
    # This is not hypothetical and it is not the operator's mistake to
    # imagine: the agent that wrote this check ran the entire Tower test
    # suite on the wrong interpreter for a whole working session before
    # noticing, on this machine, with `tower/.venv` sitting beside it.
    # `sys.executable` is in the detail for exactly that reason -- the
    # answer to "why is it failing" is usually "you are not running the
    # Python you think you are".
    try:
        import pycolmap  # noqa: PLC0415

        backend_ok = hasattr(pycolmap, "incremental_mapping")
        backend_detail = (
            f"pycolmap {getattr(pycolmap, '__version__', '?')} in "
            f"{sys.executable}"
            if backend_ok
            else (
                f"pycolmap {getattr(pycolmap, '__version__', '?')} is "
                "importable but has no incremental_mapping; this is not a "
                f"build that can reconstruct. Interpreter: {sys.executable}"
            )
        )
    except Exception as exc:  # noqa: BLE001 - a diagnostic never fails on its input
        backend_ok = False
        backend_detail = (
            f"pycolmap is not importable here ({type(exc).__name__}), so "
            "EVERY walk on this interpreter will produce zero poses and "
            f"zero points. Interpreter: {sys.executable}. The Tower's own "
            "environment is tower/.venv -- run start_tower.ps1, or "
            ".venv/Scripts/python.exe, rather than a system Python"
        )
    verdicts.append(("sfm_backend_importable", backend_ok, backend_detail))

    # WHICH WORLDS CLAIM TO BE BEING BUILT RIGHT NOW.
    #
    # A lock file whose pid is alive makes the status producer prefer that
    # world over every saved one, so the phone opens onto it. On this
    # machine 29 of 163 worlds hold a lock and none carries `created_at`
    # -- they predate the field -- so before `_holder_is_running` learned
    # to compare the process start time against the lock file's own mtime,
    # a recycled pid was indistinguishable from a live builder. Two
    # reviewers hit the same alias within an hour and one reproduced the
    # consequence end to end: a real walk built correctly and then, the
    # moment finalization released its own lock, the live screen reverted
    # to a fortnight-old empty world and said "Mapping" forever.
    #
    # The judgement is fixed. This verdict exists so the operator can SEE
    # the condition rather than discover it mid-walk: any world reported
    # live before a walk starts is one to look at.
    try:
        from tower.world_builder.store import WorldStore  # noqa: PLC0415

        world_root = (report.get("calibrations") or {}).get("world_root")
        if not world_root:
            locks_ok, locks_detail = True, (
                "no TOWER_WORLD_ROOT to check; nothing claims to be building"
            )
        else:
            store = WorldStore(pathlib.Path(world_root))
            held = live = 0
            live_ids = []
            for world_id in store.list_world_ids():
                if not store.lock_path(world_id).exists():
                    continue
                held += 1
                holder = store.lock_holder(world_id)
                if holder and holder.get("alive"):
                    live += 1
                    live_ids.append(f"{world_id[:8]} (pid {holder.get('pid')})")
            locks_ok = live == 0
            if live:
                locks_detail = (
                    f"{live} of {held} lock files name a LIVE process: "
                    + ", ".join(live_ids[:4])
                    + ". If no walk is running, one of these is a recycled "
                    "pid and the phone will open onto that world instead of "
                    "the wearer's"
                )
            else:
                locks_detail = (
                    f"{held} lock file(s), none naming a live process"
                    if held
                    else "no world holds a writer lock"
                )
    except Exception as exc:  # noqa: BLE001 - a diagnostic never fails on its input
        locks_ok = True
        locks_detail = f"could not be checked ({type(exc).__name__})"
    verdicts.append(("no_world_claims_to_be_building", locks_ok, locks_detail))

    # The interesting failure is specifically "GPU present, torch blind to
    # it": that is a fixable packaging problem rather than missing
    # hardware, and the two are indistinguishable if you only check torch.
    verdicts.append(
        (
            "gpu_reachable_from_python",
            bool(nvidia.get("available")) and torch_cuda_ok,
            (
                "GPU present but torch cannot use it"
                if nvidia.get("available") and not torch_cuda_ok
                else ("ok" if torch_cuda_ok else "no GPU reachable")
            ),
        )
    )

    cv_info = report["opencv"]
    if not cv_info.get("installed"):
        verdicts.append(("opencv_geometry", False, "opencv not installed"))
        verdicts.append(("opencv_self_calibration", False, "opencv not installed"))
        return verdicts

    capabilities = cv_info["capabilities"]
    incomplete = sorted(
        name for name, data in capabilities.items() if not data["complete"]
    )
    verdicts.append(
        (
            "opencv_geometry",
            not incomplete,
            (
                f"opencv {cv_info['version']}: all {len(capabilities)} "
                "capability groups complete"
                if not incomplete
                else f"incomplete groups: {', '.join(incomplete)}"
            ),
        )
    )

    charuco = cv_info["charuco"]
    charuco_ok = (
        charuco["CharucoBoard"]
        and charuco["CharucoDetector"]
        and capabilities["calibration"]["complete"]
    )
    verdicts.append(
        (
            "opencv_self_calibration",
            charuco_ok,
            (
                "ChArUco board + detector + calibrateCamera available"
                if charuco_ok
                else "ChArUco/calibration primitives incomplete"
            ),
        )
    )

    return verdicts


def render_text(report, verdicts):
    lines = []

    host = report["host"]
    lines.append("=== Host ===")
    lines.append(f"python           {host['python']}")
    lines.append(f"platform         {host['platform']}")

    lines.append("")
    lines.append("=== GPU (driver) ===")
    nvidia = report["nvidia_smi"]
    if nvidia.get("available"):
        for index, gpu in enumerate(nvidia["gpus"]):
            lines.append(
                f"[{index}] {gpu['name']}  driver {gpu['driver_version']}  "
                f"compute {gpu['compute_capability']}  "
                f"{gpu['memory_free']} free / {gpu['memory_total']}"
            )
    else:
        lines.append(f"unavailable: {nvidia.get('reason')}")

    lines.append("")
    lines.append("=== torch ===")
    torch_info = report["torch"]
    if torch_info.get("installed"):
        lines.append(f"version          {torch_info['version']}")
        lines.append(f"cuda build       {torch_info['cuda_build']}")
        lines.append(f"cuda available   {torch_info['cuda_available']}")
        lines.append(f"arch list        {torch_info['arch_list']}")
        if torch_info.get("device_name"):
            lines.append(f"device           {torch_info['device_name']}")
            lines.append(f"capability       {torch_info['device_capability']}")
    else:
        lines.append(f"not installed: {torch_info.get('reason')}")

    lines.append("")
    lines.append("=== OpenCV ===")
    cv_info = report["opencv"]
    if cv_info.get("installed"):
        lines.append(f"version          {cv_info['version']}")
        lines.append(f"cuda devices     {cv_info['cuda_device_count']}")
        for capability, data in cv_info["capabilities"].items():
            status = "complete" if data["complete"] else "INCOMPLETE"
            lines.append(f"{capability:16s} {status}")
            if data["missing"]:
                lines.append(f"                 missing: {', '.join(data['missing'])}")
        charuco = cv_info["charuco"]
        lines.append(
            f"charuco          board={charuco['CharucoBoard']} "
            f"detector={charuco['CharucoDetector']}"
        )
    else:
        lines.append(f"not installed: {cv_info.get('reason')}")

    lines.append("")
    lines.append("=== Optional libraries ===")
    for name, version in report["libraries"].items():
        lines.append(f"{name:16s} {version if version else '-- absent --'}")

    lines.append("")
    lines.append("=== Verdicts ===")
    for key, ok, detail in verdicts:
        lines.append(f"[{'OK ' if ok else 'NO '}] {key:28s} {detail}")

    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Read-only World Builder environment readiness diagnostic. "
            "Installs nothing and writes nothing."
        )
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format (default: text).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Exit non-zero if any verdict fails. Off by default so the "
            "diagnostic still runs cleanly on the machine it is diagnosing."
        ),
    )
    args = parser.parse_args(argv)

    report = {
        "host": collect_host(),
        "nvidia_smi": collect_nvidia_smi(),
        "torch": collect_torch(),
        "opencv": collect_opencv(),
        "libraries": collect_libraries(),
        "vocabulary_tree": collect_vocabulary_tree(),
        "package_origin": collect_package_origin(),
        "calibrations": collect_calibrations(),
    }
    verdicts = build_verdicts(report)
    report["verdicts"] = [
        {"check": key, "ok": ok, "detail": detail} for key, ok, detail in verdicts
    ]

    if args.format == "json":
        print(json.dumps(report, indent=2))
    else:
        print(render_text(report, verdicts))

    if args.strict and any(not ok for _, ok, _ in verdicts):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
