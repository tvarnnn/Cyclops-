#!/usr/bin/env python
"""Same-session A/B for the two product-path optimisations.

WHY THIS EXISTS RATHER THAN COMPARING AGAINST THE STORED BASELINE

Comparing a run made now against `baseline_HEAD_d3d24b5.json` gives 2.19x,
and that number is not real. That baseline was produced by a different
lane, in a separate `git archive` tree, at a different time, with a cold
page cache and different machine load. The component measurements predict
about 1.17x, so most of the 2.19x is measurement conditions.

This runs BOTH arms back to back in ONE process, on the same captures, with
the same warm cache, alternating arm order to cancel drift. The old
implementations are restored by monkeypatch rather than by checking out
old code, so the only thing that differs between arms is the two functions
under test.

Arms:
  new  -- as shipped: cv2.meanStdDev(Laplacian(CV_16S)), dumps+write
  old  -- restored:   cv2.Laplacian(CV_64F).var(),       json.dump(stream)
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

TOWER = Path(__file__).resolve().parents[3]
for extra in (TOWER, TOWER / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))


def _resolve():
    import tower.world_builder.backends.classical as classical

    where = Path(classical.__file__).resolve()
    if "Glasses-world-builder" not in str(where):
        raise SystemExit(f"REFUSING: production code resolved to {where}")
    if not hasattr(classical, "EXTEND_REFERENCE_DEPTH"):
        raise SystemExit("REFUSING: wrong branch (no EXTEND_REFERENCE_DEPTH)")
    return where


def _install_old():
    """Restore the pre-optimisation implementations by monkeypatch."""
    import cv2
    import numpy as np  # noqa: F401
    from tower.world_builder import frontend
    from tower import storage

    def old_sharpness(gray):
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def old_write(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(path.name + storage.TEMP_SUFFIX)
        try:
            with temp_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            storage.replace_with_retry(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)

    frontend.measure_sharpness = old_sharpness
    storage.write_json_atomic = old_write
    # The engine imported these by name at module load, so rebind there too.
    import tower.world_builder.engine as engine
    import tower.world_builder.store as store

    if hasattr(engine, "analyse_frame"):
        engine.analyse_frame.__globals__["measure_sharpness"] = old_sharpness
    store.write_json_atomic = old_write


def _install_new(saved):
    from tower.world_builder import frontend
    from tower import storage
    import tower.world_builder.engine as engine
    import tower.world_builder.store as store

    frontend.measure_sharpness = saved["sharpness"]
    storage.write_json_atomic = saved["write"]
    engine.analyse_frame.__globals__["measure_sharpness"] = saved["sharpness"]
    store.write_json_atomic = saved["write"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", required=True, help="comma-separated prefixes")
    ap.add_argument("--captures-root", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()

    where = _resolve()
    print(f"production code: {where}")

    import world_builder_corpus_benchmark as bench
    from tower.world_builder import frontend
    from tower import storage
    from tower.world_builder.intrinsics_store import IntrinsicsStore

    saved = {
        "sharpness": frontend.measure_sharpness,
        "write": storage.write_json_atomic,
    }
    istore = IntrinsicsStore(Path(args.data_root))
    root = Path(args.captures_root)
    scratch = Path(args.scratch)

    prefixes = [p.strip() for p in args.captures.split(",")]
    resolved = {}
    for prefix in prefixes:
        matches = [d for d in root.iterdir() if d.name.startswith(prefix)]
        if len(matches) != 1:
            raise SystemExit(f"prefix {prefix!r} matched {len(matches)} dirs")
        resolved[prefix] = matches[0]

    results = {"old": {p: [] for p in prefixes}, "new": {p: [] for p in prefixes}}

    for repeat in range(args.repeats):
        # Alternate which arm goes first so any monotonic drift in machine
        # load is charged to both arms equally rather than to whichever
        # ran second.
        order = ("new", "old") if repeat % 2 == 0 else ("old", "new")
        for arm in order:
            if arm == "old":
                _install_old()
            else:
                _install_new(saved)
            for prefix in prefixes:
                out = scratch / f"{arm}{repeat}" / prefix
                started = time.perf_counter()
                res = bench.run_capture(prefix, resolved[prefix], out, istore)
                elapsed = time.perf_counter() - started
                results[arm][prefix].append((elapsed, res["keyframes"],
                                             res["poses_solved"], res["points"]))
                print(f"  repeat{repeat} {arm:3} {prefix}: {elapsed:6.2f}s "
                      f"kf={res['keyframes']} solved={res['poses_solved']} "
                      f"pts={res['points']}")

    _install_new(saved)

    print(f"\n{'capture':10} {'old (s)':>9} {'new (s)':>9} {'speedup':>8}  parity")
    old_total = new_total = 0.0
    for prefix in prefixes:
        o = statistics.median(x[0] for x in results["old"][prefix])
        n = statistics.median(x[0] for x in results["new"][prefix])
        old_total += o
        new_total += n
        parity = ({x[1:] for x in results["old"][prefix]}
                  == {x[1:] for x in results["new"][prefix]})
        print(f"{prefix:10} {o:9.2f} {n:9.2f} {o/n:8.2f}x  "
              f"{'IDENTICAL' if parity else 'DIVERGED'}")
    print(f"{'TOTAL':10} {old_total:9.2f} {new_total:9.2f} "
          f"{old_total/new_total:8.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
