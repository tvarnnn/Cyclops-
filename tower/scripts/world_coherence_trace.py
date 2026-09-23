#!/usr/bin/env python
"""Chronological trajectory / sparse / segment forensics for one saved world.

    python scripts/world_coherence_trace.py --world-dir <world dir> --out <dir>
        [--session <id>] [--captures-root <dir> | --capture-dir <dir> ...]
        [--regions <csv>] [--tag <name>] [--no-replay] [--no-live-chain] [--no-pairs]

READ-ONLY with respect to the world and the capture: the COLMAP database is
opened `mode=ro&immutable=1`, images are only read, and every output goes
under `--out`. See `tower/world_builder/coherence_eval/trace.py` for what each
column means and where it comes from.

Outputs under --out:
  keyframes_<tag>.csv   one row per keyframe, capture order
  segments_<tag>.csv    one row per live tracker segment (chain vs global Sim3,
                        merge state, placement)
  frames_<tag>.csv      the frontend replay: one row per raw frame
  pairs_<tag>.csv       every verified pair: rotation agreement + doubling test
  islands_<tag>.csv     rigid (shared-3-D-point) and sequential islands of the solve
  island_links_<tag>.csv  how each island is tied to the others
  gaps_<tag>.csv        raw-frame SIFT matchability across every tracking loss
  regions_summary_<tag>.csv / revisits_<tag>.csv  (when --regions / --revisits)
  summary_<tag>.json    replay/re-run verification, up vector, extents
  plots/*.png
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# numpy/scipy before anything that might start a thread (native_prewarm.py).
import numpy  # noqa: E402,F401

from tower.artifact_paths import artifact_root_arg  # noqa: E402
from tower.world_builder.coherence_eval.trace import run_trace  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--world-dir", type=Path, required=True,
                        help="a world directory (world.json, sessions/, solve/, derived/)")
    parser.add_argument("--out", type=artifact_root_arg, required=True,
                        help="output directory (refused at a drive root or directly in the home directory)")
    parser.add_argument("--session", default=None, help="session id (default: the world's last)")
    parser.add_argument("--captures-root", type=Path, default=None,
                        help="directory holding <capture_id>/ capture directories; the session's capture "
                             "chain is followed through continues_capture")
    parser.add_argument("--capture-dir", type=Path, action="append", default=None,
                        help="explicit capture directory, repeatable, in time order (overrides --captures-root)")
    parser.add_argument("--regions", type=Path, default=None,
                        help="region-label CSV (keyframe_id, region, confidence); joined for reading only")
    parser.add_argument("--revisits", type=Path, default=None,
                        help="revisit-annotation CSV (range_a_start, range_a_end, range_b_start, range_b_end, "
                             "region, confidence); evaluation only")
    parser.add_argument("--no-gaps", action="store_true",
                        help="skip SIFT matchability of the raw frames refused at tracking losses")
    parser.add_argument("--tag", default=None, help="file-name tag (default: first 8 chars of the world id)")
    parser.add_argument("--no-replay", action="store_true", help="skip the frontend replay over raw frames")
    parser.add_argument("--no-live-chain", action="store_true", help="skip re-running the live chain")
    parser.add_argument("--no-pairs", action="store_true", help="skip the all-verified-pairs consistency pass")
    args = parser.parse_args(argv)

    result = run_trace(
        args.world_dir, args.out, session_id=args.session, captures_root=args.captures_root,
        capture_dirs=args.capture_dir, regions_csv=args.regions, revisits_csv=args.revisits, tag=args.tag,
        replay=not args.no_replay, live_chain=not args.no_live_chain, pairs=not args.no_pairs,
        gaps=not args.no_gaps,
    )
    print(json.dumps(result["summary"], indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
