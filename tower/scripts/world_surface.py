#!/usr/bin/env python
"""Build the surface reconstruction of a saved world, offline.

WHY THIS EXISTS

The development loop for reconstruction has to be re-runnable on worlds that
already exist, with different parameters, without a new physical capture.
Wearing the glasses again is not a debugging step. Every world on disk that has
a global solve can be reconstructed here as often as needed, and the SAME
`surfacify()` runs in the served product -- this is a front end to it, not a
parallel implementation. A demo script that reconstructs differently from
production is worse than no script, because it makes the product look finished.

It is also the migration path. Worlds solved before the surface stage existed
get their surface by running this; nothing about them is rewritten, and a world
that is never reconstructed keeps working exactly as it does today.

    .venv\\Scripts\\python.exe scripts/world_surface.py --list
    .venv\\Scripts\\python.exe scripts/world_surface.py --world <id>
    .venv\\Scripts\\python.exe scripts/world_surface.py --world <id> --force
    .venv\\Scripts\\python.exe scripts/world_surface.py --all
    .venv\\Scripts\\python.exe scripts/world_surface.py --world <id> --inspect
    .venv\\Scripts\\python.exe scripts/world_surface.py --world <id> --export-obj out.obj

Writes only under `<world>/surface/<session>/`, and shares the depth maps under
`<world>/dense/<session>/work/` with the point stage rather than recomputing
them. It never modifies `derived/`, never modifies `solve/`, and never touches
a capture.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tower.artifact_paths import artifact_root_arg  # noqa: E402
from tower.world_builder.global_solve import load_solution  # noqa: E402
from tower.world_builder.store import WorldStore  # noqa: E402
from tower.world_builder.surface import (  # noqa: E402
    SurfaceParams,
    read_mesh_bytes,
)
from tower.world_builder.surface_pipeline import (  # noqa: E402
    STATE_OK,
    read_surface_level,
    read_surface_manifest,
    surface_currency,
    surfacify,
)

DEFAULT_ROOT = Path("data") / "world_builder"


def _sessions_with_solves(store, world_id):
    world = store.read_world(world_id)
    return [sid for sid in world.session_ids
            if load_solution(store, world_id, sid) is not None]


def _params_from_args(args) -> SurfaceParams:
    if args.live:
        overrides = {}
        for name in ("voxel_frac", "min_weight", "smooth_iterations",
                     "min_component_frac", "gate_rel"):
            value = getattr(args, name, None)
            if value is not None:
                overrides[name] = value
        return SurfaceParams.live(**overrides)
    kw = {}
    for name in ("voxel_frac", "trunc_voxels", "trunc_error_multiple",
                 "min_weight", "carve_weight", "max_grazing_deg", "edge_rel",
                 "gate_rel", "min_component_frac", "smooth_iterations"):
        value = getattr(args, name, None)
        if value is not None:
            kw[name] = value
    if args.no_carve:
        kw["carve"] = False
    return SurfaceParams(**kw)


def _print_progress(stage, done, total):
    print(f"    {stage}: {done}/{total}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=artifact_root_arg, default=str(DEFAULT_ROOT),
                    help="world root (routed through the artifact-path guard: a "
                         "relative root resolved from the wrong directory is "
                         "how a gigabyte of reconstruction once landed at C:\\)")
    ap.add_argument("--world")
    ap.add_argument("--session")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--inspect", action="store_true",
                    help="report what is on disk and whether it matches the solve")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--export-obj", help="write the canonical level as a .obj")
    ap.add_argument("--export-level", type=int, default=None)
    ap.add_argument("--backend", help="depth backend for the shared depth stage")
    ap.add_argument("--live", action="store_true",
                    help="the coarse preset the builder runs DURING a walk: "
                         "twice the voxel, one level of detail, a lighter "
                         "smooth. Being late is worse than being coarse; the "
                         "evidence rule is not relaxed")

    ap.add_argument("--voxel-frac", dest="voxel_frac", type=float)
    ap.add_argument("--trunc-voxels", dest="trunc_voxels", type=float)
    ap.add_argument("--trunc-error-multiple", dest="trunc_error_multiple", type=float,
                    help="truncation as a multiple of the frames' measured "
                         "disagreement; below 1 they carve away each other's surface")
    ap.add_argument("--min-weight", dest="min_weight", type=float)
    ap.add_argument("--no-carve", action="store_true")
    ap.add_argument("--carve-weight", dest="carve_weight", type=float)
    ap.add_argument("--grazing-deg", dest="max_grazing_deg", type=float)
    ap.add_argument("--edge-rel", dest="edge_rel", type=float)
    ap.add_argument("--gate", dest="gate_rel", type=float)
    ap.add_argument("--min-component-frac", dest="min_component_frac", type=float)
    ap.add_argument("--smooth", dest="smooth_iterations", type=int)
    args = ap.parse_args()

    store = WorldStore(Path(args.root))

    targets: list[tuple[str, str]] = []
    if args.world:
        sessions = ([args.session] if args.session
                    else _sessions_with_solves(store, args.world))
        targets = [(args.world, s) for s in sessions]
    else:
        for world_id in store.list_world_ids():
            for sid in _sessions_with_solves(store, world_id):
                targets.append((world_id, sid))

    if args.list or (not args.world and not args.all and not args.inspect):
        print(f"{len(targets)} solved session(s)")
        for world_id, sid in targets:
            man = read_surface_manifest(store, world_id, sid)
            if man is None:
                state = "no surface"
            else:
                cur = surface_currency(store, world_id, sid, man)
                state = (f"{man['faces']:,} faces, "
                         f"{'current' if cur['current'] else 'BEHIND the solve'}")
            print(f"  {world_id}  {sid[:12]}  {state}")
        return 0

    if args.inspect:
        for world_id, sid in targets:
            man = read_surface_manifest(store, world_id, sid)
            print(f"=== {world_id} / {sid} ===")
            if man is None:
                print("  no surface artifact")
                continue
            cur = surface_currency(store, world_id, sid, man)
            print(json.dumps({
                "format": man["format"], "faces": man["faces"],
                "vertices": man["vertices"], "voxel": man["voxel"],
                "truncation": man["truncation"],
                "frames_used": man["frames_used"],
                "frames_offered": man["frames_offered"],
                "levels": man["levels"], "seconds": man["seconds"],
                "scale": man["scale"], "currency": cur,
            }, indent=2))
        return 0

    if args.export_obj:
        world_id, sid = targets[0]
        man = read_surface_manifest(store, world_id, sid)
        if man is None:
            print("no surface artifact for that session", file=sys.stderr)
            return 2
        level = args.export_level if args.export_level is not None \
            else man["canonical_level"]
        V, F, C, _ = read_mesh_bytes(read_surface_level(store, world_id, sid, level))
        out = Path(args.export_obj)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as handle:
            for v, c in zip(V, C):
                handle.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} "
                             f"{c[0]/255:.4f} {c[1]/255:.4f} {c[2]/255:.4f}\n")
            for f in F + 1:
                handle.write(f"f {f[0]} {f[1]} {f[2]}\n")
        print(f"wrote {out} ({len(V):,} verts, {len(F):,} faces, level {level})")
        return 0

    params = _params_from_args(args)
    failures = 0
    cannot_run_here = False
    for world_id, sid in targets:
        print(f"=== {world_id} / {sid} ===", flush=True)
        t = time.time()
        result = surfacify(store, world_id, sid, params=params, force=args.force,
                           backend=args.backend, progress=_print_progress)
        if result.state != STATE_OK:
            print(f"  {result.state}: {result.detail}")
            failures += 1
            cannot_run_here = cannot_run_here or bool(result.permanent)
            continue
        if result.detail:
            print(f"  {result.detail}")
        print(f"  frames {result.frames_used}/{result.frames_offered} used")
        print(f"  voxel {result.voxel:.5f}  truncation {result.trunc:.5f} "
              f"({result.trunc / result.voxel:.1f} voxels)")
        for lv in result.levels:
            print(f"  L{lv['level']}: {lv['faces']:,} faces, "
                  f"{lv['bytes'] / 1e6:.1f} MB")
        print(f"  {time.time() - t:.1f}s total  {result.seconds}")
    if cannot_run_here:
        # Distinct from an ordinary failure, so the builder's live worker can
        # stop relaunching (SURFACE_EXIT_CANNOT_RUN_HERE in world_build_session).
        return 4
    return 1 if failures and failures == len(targets) else 0


if __name__ == "__main__":
    raise SystemExit(main())
