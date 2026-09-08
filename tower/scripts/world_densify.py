#!/usr/bin/env python
"""Densify a saved world, offline.

WHY THIS EXISTS

**THIS IS THE ONLY WAY THE DENSE STAGE RUNS IN THE SERVED PRODUCT TODAY.**
`world_build_session.py` accepts `--densify` and honours it, but `main.py`
never passes it and no setting turns it on -- `config.py` has `world_autobuild`,
`world_rebuild_every`, `world_register` and `world_solve`, and nothing dense. So
a capture through the Tower produces a solved world and no dense artifact until
somebody runs this. An earlier version of this docstring said the opposite.

Beyond that, the whole development loop depends on being able to re-run it on
worlds that already exist, with different parameters, without a new physical
capture. That is the regression laboratory: seven solved worlds on disk,
replayed as often as needed.

It is also the migration path. Worlds solved before the dense stage existed get
their dense artifact by running this; nothing about them is rewritten, and a
world that is never densified keeps working exactly as it does today.

    .venv\\Scripts\\python.exe scripts/world_densify.py --list
    .venv\\Scripts\\python.exe scripts/world_densify.py --world <id>
    .venv\\Scripts\\python.exe scripts/world_densify.py --world <id> --session <id>
    .venv\\Scripts\\python.exe scripts/world_densify.py --world <id> --force
    .venv\\Scripts\\python.exe scripts/world_densify.py --all

Reads the raw capture frames named by the solve's `sources.json` and writes only
under `<world>/dense/<session>/`. It never modifies `derived/`, never modifies
`solve/`, and never touches a capture.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tower.artifact_paths import artifact_root_arg  # noqa: E402
from tower.world_builder.dense import DenseParams, available_backends  # noqa: E402
from tower.world_builder.dense_pipeline import (  # noqa: E402
    STATE_OK,
    densify,
    read_dense_manifest,
)
from tower.world_builder.global_solve import load_solution  # noqa: E402
from tower.world_builder.store import WorldStore  # noqa: E402

DEFAULT_ROOT = Path("data") / "world_builder"


def _sessions_with_solves(store, world_id):
    world = store.read_world(world_id)
    out = []
    for sid in world.session_ids:
        if load_solution(store, world_id, sid) is not None:
            out.append(sid)
    return out


def _progress(stage, n, total):
    print(f"    {stage}: {n}/{total}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world-root", type=artifact_root_arg, default=str(DEFAULT_ROOT))
    ap.add_argument("--world")
    ap.add_argument("--session")
    ap.add_argument("--list", action="store_true",
                    help="list worlds that hold a global solve, and whether they are dense yet")
    ap.add_argument("--all", action="store_true", help="densify every solved world")
    ap.add_argument("--force", action="store_true", help="ignore cached stages")
    ap.add_argument("--backend", default=DenseParams.backend,
                    help=f"depth backend; one of {', '.join(available_backends())}")
    ap.add_argument("--gate-rel", type=float, default=DenseParams.gate_rel)
    ap.add_argument("--tau", type=float, default=DenseParams.tau)
    ap.add_argument("--min-views", type=int, default=DenseParams.min_views)
    ap.add_argument("--neighbours", type=int, default=DenseParams.neighbours)
    ap.add_argument("--stride", type=int, default=DenseParams.stride)
    ap.add_argument("--component", type=int, default=DenseParams.component)
    ap.add_argument("--keep-intermediates", action="store_true",
                    help="keep the per-frame depth maps, the undistorted frames and "
                         "fused.npz. About 470 MB on a 438-keyframe world, and what "
                         "makes re-fusing with different parameters fast -- so this "
                         "is the flag for the development loop")
    ap.add_argument("--format", choices=("text", "json"), default="text")
    a = ap.parse_args(argv)

    store = WorldStore(Path(a.world_root))

    if a.list:
        rows = []
        for world_id in sorted(store.list_world_ids()):
            for sid in _sessions_with_solves(store, world_id):
                man = read_dense_manifest(store, world_id, sid)
                rows.append({
                    "world_id": world_id, "session_id": sid,
                    "dense": bool(man),
                    "points": (man["levels"][0]["points"] if man else None),
                })
        if a.format == "json":
            print(json.dumps(rows, indent=1))
        else:
            print(f"{len(rows)} solved session(s)")
            for r in rows:
                mark = f"dense {r['points']:,} pts" if r["dense"] else "not densified"
                print(f"  {r['world_id']}  {r['session_id'][:12]}  {mark}")
        return 0

    params = DenseParams(
        backend=a.backend, gate_rel=a.gate_rel, tau=a.tau, min_views=a.min_views,
        neighbours=a.neighbours, stride=a.stride, component=a.component,
        keep_intermediates=a.keep_intermediates,
    )

    targets: list[tuple[str, str]] = []
    if a.all:
        for world_id in sorted(store.list_world_ids()):
            for sid in _sessions_with_solves(store, world_id):
                targets.append((world_id, sid))
    elif a.world:
        sids = [a.session] if a.session else _sessions_with_solves(store, a.world)
        if not sids:
            print(f"world {a.world} has no session with a global solve", file=sys.stderr)
            return 2
        targets.extend((a.world, s) for s in sids)
    else:
        ap.error("give --world, or --all, or --list")

    results = []
    failures = 0
    for world_id, sid in targets:
        print(f"\n=== {world_id} / {sid} ===", flush=True)
        t0 = time.time()
        res = densify(store, world_id, sid, params=params,
                      progress=_progress, force=a.force)
        elapsed = time.time() - t0
        row = {"world_id": world_id, "session_id": sid, "elapsed_s": round(elapsed, 1),
               **res.as_dict()}
        results.append(row)
        if res.state != STATE_OK:
            failures += 1
            print(f"  {res.state}: {res.detail}")
            continue
        print(f"  frames {res.frames_used}/{res.frames_total} used "
              f"({res.frames_dropped} dropped by the alignment gate)")
        if res.align_rel_median is not None:
            print(f"  held-out alignment residual (median) {res.align_rel_median * 100:.1f}%")
        for lvl in res.levels:
            print(f"  L{lvl['level']} voxel {lvl['voxel']}: "
                  f"{lvl['points']:,} points, {lvl['bytes'] / 1e6:.1f} MB")
        print(f"  {elapsed:.1f}s total")

    if a.format == "json":
        print(json.dumps(results, indent=1, default=str))
    # A world that legitimately cannot be densified is not a failure of the
    # run. Only a run where nothing at all succeeded exits non-zero.
    return 1 if results and failures == len(results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
