#!/usr/bin/env python
"""Run the global solver over one session and persist its solution.

    python scripts/world_solve.py --root <world root> --world <id> --session <id>
        [--capture-dir <dir> ...] [--final] [--threads N] [--loop-detection]
        [--format json|text]

What it does, and what it deliberately does not do:

  * It undistorts the session's keyframes (raw capture frames when a
    `--capture-dir` holds them, else the session's redacted copies), extracts
    and matches features INCREMENTALLY into `<world>/solve/<session>/`, and
    solves the whole keyframe set globally (GLOMAP, incremental fallback).
  * It writes `solution.json` + `solution.npz` in that workspace and
    NOTHING under `derived/`. The engine's `build()` is the only writer of
    the derived tree; it merges the solution on its next run. That is what
    lets this run as a background subprocess while the frame path keeps
    building: the two never write the same file.
  * `--final` widens matching (loop detection when a vocabulary tree is
    available) and uses every core. Without it the solver leaves two cores
    to the frame path.

Exit status is 0 whether or not a model was found -- "no model" is a
result, printed in the summary -- and 2 only for a usage error.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tower.artifact_paths import artifact_root_arg  # noqa: E402
from tower.world_builder import global_solve  # noqa: E402
from tower.world_builder.store import WorldStore, compute_input_digest  # noqa: E402


def background_threads() -> int:
    """All cores but two, at least one: the frame path is on this machine."""
    return max(1, (os.cpu_count() or 2) - 2)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=artifact_root_arg, required=True)
    parser.add_argument("--world", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--capture-dir", action="append", default=[],
                        help="capture directory whose frames/ holds raw keyframe frames; repeatable")
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--loop-detection", action="store_true",
                        help="sequential matching with vocabulary-tree loop detection "
                             "(downloads a vocabulary tree on first use)")
    parser.add_argument("--min-image-observations", type=int,
                        default=global_solve.MIN_IMAGE_OBSERVATIONS)
    parser.add_argument("--format", choices=("json", "text"), default="json")
    args = parser.parse_args(argv)

    store = WorldStore(Path(args.root))
    keyframes = store.read_keyframes(args.world, args.session)
    threads = args.threads if args.threads is not None else (-1 if args.final else background_threads())
    summary = global_solve.solve(
        store, args.world, args.session,
        capture_dirs=[Path(d) for d in args.capture_dir],
        final=args.final,
        num_threads=threads,
        min_image_observations=args.min_image_observations,
        loop_detection=args.loop_detection,
        input_digest=compute_input_digest(keyframes),
    )
    if args.format == "json":
        print(json.dumps(summary, indent=2))
    else:
        for key, value in summary.items():
            print(f"{key:20s} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
