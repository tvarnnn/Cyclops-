#!/usr/bin/env python
"""Rebuild a recorded physical walk, deterministically, from raw frames.

WHY THIS EXISTS

Every World Builder finding before this script was argued from a world
already on disk. That is enough to say what the pipeline DID and useless
for saying what a change would DO, because the only way to re-measure was
to put the glasses back on. A registration change that can only be
validated by another walk is a change nobody dares make.

This replays a walk instead. It reproduces the two physical sessions of
2026-08-29 exactly -- same frames observed, same keyframes accepted, same
rejection histogram, same segment count, same solved poses, same points:

    worldA  1008 frames -> 229 keyframes / 23 segments / 100 solved / 9145 points
    worldB  1074 frames -> 218 keyframes / 36 segments / 108 solved / 13050 points

matching the persisted sessions 815c88ba (world 991e5a15) and 7864d3b3
(world af47007c) figure for figure. That equality is the whole warrant
for the tool: an offline number may then be compared with a physical one.

WHAT A "WALK" IS ON DISK

One walk is usually SEVERAL captures. The Tower mints a capture at
`stream_start`, so a transport disconnect ends one and the reconnect
begins the next, while the World Builder follower keeps a single session
across the gap (under the 90 s new-walk bound). Both sessions here are
three captures. `CASES` records which, in time order.

The frame numbering across captures is NOT a walk-wide clock. In the two
cases above it happens to be disjoint and increasing, but on other walks
(4cae0b26, adc75972) every reconnect restarts it at `00000001.jpg`, so
the raw names collide and a merged directory sorted by raw name would
interleave captures. `stage_frames` therefore names each staged frame
`<capture ordinal>_<capture id[:8]>_<raw name>`: unique by construction,
sorted into capture order first and raw-name order within a capture, and
naming its own source. Where the numbering does continue, that order is
exactly the raw-name order, so the pinned cases replay the same frames in
the same order as before. `staging.json` beside the frames maps every
staged name back to its capture and raw file.

WHAT IT IS NOT

`--frames` reads a directory, not the capture journal, so `source_seq`,
`tx_seq` and receipt time are the enumeration index rather than the
recorder's own values. Nothing in keyframe selection or geometry reads
them -- which is why the replay comes out exact -- but a transport
question must not be asked of this tool. Use `--follow-capture` for that.

Staging is by hard link, never a copy: the frames are the evidence, and a
replay must not be able to cost a gigabyte or to modify them.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tower.artifact_paths import artifact_root_arg  # noqa: E402

TOWER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAPTURE_ROOT = TOWER_ROOT / "data" / "captures"
DEFAULT_WORLD_ROOT = TOWER_ROOT / "data" / "world_builder"

# The captures behind each recorded walk, in the order they were captured.
#
# Named rather than globbed, with the expected totals pinned beside them,
# because a replay that silently drops or gains a capture measures a
# different walk and says nothing about the physical one.
CASES = {
    # 2026-08-29 physical test 1: normal room movement.
    "worldA": {
        "captures": (
            "69a4e59d825948da95e675874d6c3f0b",
            "28f544af41b144f8828de1c9e4acc855",
            "01e4c64fe17a49948844fb821bf2e401",
        ),
        "world_id": "991e5a153c974c9fa81650fcbcc995ff",
        "session_id": "815c88babda64bbca60ca93127e2e2d7",
        "expect": {
            "frames_observed": 1008,
            "keyframes_accepted": 229,
            "segments": 23,
            "poses_solved": 100,
            "points": 9145,
        },
    },
    # 2026-08-29 physical test 2: deliberate lateral motion around a
    # textured drawer, with a return to the starting viewpoint. This is
    # the capture the queued PT-1 asked for, and the one that moved
    # span/depth off the 0.02-0.06 floor every earlier walk sat on.
    "worldB": {
        "captures": (
            "023b5d84b72e41d9b3f0910b6ceb6e29",
            "95831c3475d34ef28d3eafa7ddec11da",
            "e3e8fd2e5451497089f3a4f66a3bdfe0",
        ),
        "world_id": "af47007c56924e568b096cfc0eaf2b24",
        "session_id": "7864d3b370ed42f8b292408090a205ff",
        "expect": {
            "frames_observed": 1074,
            "keyframes_accepted": 218,
            "segments": 36,
            "poses_solved": 108,
            "points": 13050,
        },
    },
}


STAGING_MANIFEST = "staging.json"
STAGING_SCHEMA = "wb-replay-staging/1"


def staged_name(ordinal: int, width: int, capture: str, raw_name: str) -> str:
    """The staged file name of one raw frame: `<ordinal>_<capture[:8]>_<raw>`.

    The zero-padded capture ordinal leads, so a plain sort of the staging
    directory -- which is all `world_build_session.py --frames` does -- is
    capture order first and raw-name order within a capture. The capture id
    is there for the reader: a staged path in `sources.json` then says which
    capture and which raw frame it is without a lookup.
    """
    return f"{ordinal:0{width}d}_{capture[:8]}_{raw_name}"


def stage_frames(captures, capture_root: Path, staging: Path) -> int:
    """Hard-link one walk's frames into a single directory, in walk order.

    Hard links, not copies: a replay of a thousand-frame walk must not
    cost what the capture cost, and the originals are evidence this tool
    has no business rewriting.

    Chained captures may restart their frame numbering (a reconnect begins
    again at `00000001.jpg`), so raw names are not unique across a walk and
    their sort order is not the walk's order. Every frame is staged under
    `staged_name`, which is both. `staging.json` records, per staged name,
    the capture, its position in the chain and the raw file it links to --
    the builder's `sources.json` names the staged path, and this is what
    takes that path back to the raw frame.

    Everything is checked before anything is linked, so a refusal leaves no
    half-staged walk behind:
      * a capture named twice would replay part of the walk twice;
      * a `.jpg` already in the staging directory that this walk does not
        stage (an earlier replay into the same root) would be read as
        part of the walk;
      * a staged name that exists but is a different file would silently
        substitute another photograph.
    """
    captures = list(captures)
    repeated = sorted({c for c in captures if captures.count(c) > 1})
    if repeated:
        raise SystemExit(
            f"capture(s) {', '.join(repeated)} named more than once; a walk "
            "stages each capture exactly once"
        )
    width = max(3, len(str(max(len(captures) - 1, 0))))
    plan = []
    for ordinal, capture in enumerate(captures):
        frames = capture_root / capture / "frames"
        if not frames.is_dir():
            raise SystemExit(f"capture {capture} has no frames directory at {frames}")
        for source in sorted(frames.glob("*.jpg")):
            plan.append((staged_name(ordinal, width, capture, source.name),
                         ordinal, capture, source))

    staging.mkdir(parents=True, exist_ok=True)
    planned = {name for name, _, _, _ in plan}
    foreign = sorted(p.name for p in staging.glob("*.jpg") if p.name not in planned)
    if foreign:
        raise SystemExit(
            f"{staging} already holds {len(foreign)} frame(s) this walk does not "
            f"stage (first: {foreign[0]}); they would be replayed as part of it. "
            "Replay into a fresh --root"
        )
    for name, _, _, source in plan:
        target = staging / name
        if target.exists() and not os.path.samefile(source, target):
            raise SystemExit(
                f"{target} exists and is not a link to {source}; refusing to "
                "replay a different photograph under this name"
            )

    for name, _, _, source in plan:
        target = staging / name
        if not target.exists():
            os.link(source, target)
    manifest = {
        "schema": STAGING_SCHEMA,
        "captures": captures,
        "frames": [
            {
                "staged": name,
                "capture_id": capture,
                "capture_ordinal": ordinal,
                "raw_name": source.name,
                "raw_path": str(source.resolve()),
            }
            for name, ordinal, capture, source in plan
        ],
    }
    (staging / STAGING_MANIFEST).write_text(
        json.dumps(manifest, indent=1), encoding="utf-8"
    )
    return len(plan)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a recorded physical walk through the World Builder "
            "engine, offline and deterministically."
        )
    )
    parser.add_argument("--case", choices=sorted(CASES), help="A pinned recorded walk.")
    parser.add_argument(
        "--captures",
        nargs="+",
        help="Capture ids in time order, instead of a pinned case.",
    )
    parser.add_argument(
        "--capture-root",
        type=Path,
        default=DEFAULT_CAPTURE_ROOT,
        help="Where the raw captures live. Read only; never written.",
    )
    parser.add_argument(
        "--intrinsics-from",
        type=Path,
        default=DEFAULT_WORLD_ROOT / "intrinsics",
        help=(
            "Calibration store to seed the replay root from. Without a "
            "calibration at the observed resolution the backend downgrades "
            "to unposed and the replay reconstructs nothing -- correct "
            "behaviour, and a useless measurement."
        ),
    )
    parser.add_argument(
        "--root",
        type=artifact_root_arg,
        required=True,
        help="Where the replayed world is written. Use a scratch root.",
    )
    parser.add_argument("--rebuild-every", type=int, default=4)
    parser.add_argument(
        "--register",
        action="store_true",
        help="Run cross-segment registration after the final build.",
    )
    parser.add_argument(
        "--solve",
        action="store_true",
        help="run the global solver during and after the replay (see world_build_session.py --solve)",
    )
    parser.add_argument("--solve-every", type=int, default=None)
    parser.add_argument(
        "--surface",
        action="store_true",
        help=(
            "reconstruct a surface as the builder does (passes "
            "world_build_session.py --surface): a coarse one whenever a "
            "background solve lands during the replay, and the full one "
            "after Stop. Needs --solve."
        ),
    )
    parser.add_argument(
        "--appearance",
        action="store_true",
        help=(
            "build the appearance on each surface as the builder does "
            "(passes world_build_session.py --appearance): the walk's own "
            "redacted keyframes, blended over the surface. Needs --surface, "
            "for the same reason tower/main.py gates it on the surface. "
            "Without this flag a replay cannot exercise the top rung at "
            "all, which is why it exists."
        ),
    )
    parser.add_argument(
        "--densify",
        action="store_true",
        help=(
            "run the dense reconstruction after the final build (see "
            "world_build_session.py --densify). Needs --solve: dense "
            "reconstruction is anchored to the global solution's cameras and "
            "sparse points, and there is nothing to anchor to without one."
        ),
    )
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)

    if bool(args.case) == bool(args.captures):
        parser.error("pass exactly one of --case or --captures")

    case = CASES[args.case] if args.case else None
    captures = case["captures"] if case else tuple(args.captures)

    root = Path(args.root)
    staging = root / "_frames"
    frames = stage_frames(captures, args.capture_root, staging)

    # The calibration is copied, not linked: the replay root is a world
    # root in its own right, and a later run must not be able to reach
    # back through it into the real store.
    intrinsics_out = root / "intrinsics"
    if args.intrinsics_from.is_dir():
        intrinsics_out.mkdir(parents=True, exist_ok=True)
        for calibration in args.intrinsics_from.glob("*.json"):
            intrinsics_out.joinpath(calibration.name).write_bytes(
                calibration.read_bytes()
            )

    argv_build = [
        sys.executable,
        str(TOWER_ROOT / "scripts" / "world_build_session.py"),
        "--root", str(root),
        "--frames", str(staging),
        "--rebuild-every", str(args.rebuild_every),
        "--format", "json",
    ]
    if args.register:
        argv_build.append("--register")
    if args.solve:
        argv_build.append("--solve")
        if args.solve_every is not None:
            argv_build += ["--solve-every", str(args.solve_every)]
    if args.densify:
        if not args.solve:
            parser.error("--densify needs --solve: the dense stage is anchored "
                         "to the global solution's cameras and sparse points")
        argv_build.append("--densify")
    if args.surface:
        if not args.solve:
            parser.error("--surface needs --solve: a surface is fused from the "
                         "poses the global solve places, and there is nothing "
                         "trustworthy to fuse without one")
        argv_build.append("--surface")
    if args.appearance:
        if not args.surface:
            parser.error("--appearance needs --surface: the appearance is the "
                         "walk's own keyframes blended over the surface they "
                         "were prepared against, and there is nothing to blend "
                         "them onto without one")
        argv_build.append("--appearance")

    started = time.perf_counter()
    completed = subprocess.run(
        argv_build, cwd=str(TOWER_ROOT), capture_output=True, text=True
    )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        sys.stderr.write(completed.stderr)
        raise SystemExit(f"replay build failed with code {completed.returncode}")

    report = json.loads(completed.stdout[completed.stdout.index("{"):])
    report["replay"] = {
        "case": args.case,
        "captures": list(captures),
        "frames_staged": frames,
        "staging_manifest": str(staging / STAGING_MANIFEST),
        "wall_seconds": round(elapsed, 2),
    }

    # A replay that no longer reproduces its walk is reported, not
    # silently accepted -- and not treated as an error either, because a
    # deliberate engine change is exactly when this number moves.
    if case is not None:
        drift = {
            key: {"recorded": expected, "replayed": report.get(key)}
            for key, expected in case["expect"].items()
            if report.get(key) != expected
        }
        report["replay"]["reproduces_recorded_session"] = not drift
        report["replay"]["recorded_world_id"] = case["world_id"]
        if drift:
            report["replay"]["drift_from_recorded_session"] = drift

    if args.format == "json":
        print(json.dumps(report, indent=2))
    else:
        for key, value in report.items():
            print(f"{key:22s} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
