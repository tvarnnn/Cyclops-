#!/usr/bin/env python
"""Run one solver variant of a frozen world and export it for the harness.

EXPERIMENT CODE (`tower/world_builder/coherence_exp`); the builder never runs
this. The off/off arm (no masks, no gate, no extras, no augment) is the
product recipe of `global_solve.solve`, seeded and single-threaded so that
seeds can be compared. See RUN/experiments/P2-E1/DRIVER-API.md.

SUBCOMMANDS

  run      stage -> (masks) -> SIFT -> sequential(+loop) -> (augment) ->
           one GLOMAP process per seed -> (rigid gate) -> interchange export
  masks    compute and cache transient masks (hands/arms/held phone) for the
           staged keyframes (+ --extras). GPU: run it under RUN/lead/gpulock.py
  map-one  internal: one seeded mapping run (used by `run`)

EXAMPLES (Windows; the Tower venv's python by full path, PYTHONPATH=<lane>/tower)

  world_coherence_variant.py masks --world <frozen>/worlds/<wid> --captures <frozen>/captures
  world_coherence_variant.py run --world <frozen>/worlds/<wid> --captures <frozen>/captures \
      --out <exp>/A3/<wid> --name A3 --masks transients --gate rigid --seeds 0,1,2,3,4
  world_coherence_eval.py eval --world <frozen>/worlds/<wid> --variant <exp>/A3/<wid>/variant ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# numpy/scipy on the main thread before anything spawns threads (native_prewarm rule)
import numpy as np  # noqa: E402,F401
import scipy.linalg  # noqa: E402,F401
import scipy.sparse  # noqa: E402,F401

from tower.artifact_paths import artifact_root_arg  # noqa: E402


def _config_from_args(args):
    from tower.world_builder.coherence_exp.driver import VariantConfig

    d = {}
    if args.config:
        d = json.loads(Path(args.config).read_text(encoding="utf-8"))
    for flag, key in (("name", "name"), ("masks", "masks"), ("gate", "gate"), ("mapper", "mapper"),
                      ("augment", "augment"), ("keyframe_source", "keyframe_source")):
        value = getattr(args, flag, None)
        if value is not None:
            d[key] = value
    if args.seeds:
        d["seeds"] = [int(s) for s in args.seeds.split(",") if s.strip()]
    if args.threads is not None:
        d["num_threads"] = args.threads
    if args.max_parallel is not None:
        d["max_parallel"] = args.max_parallel
    if args.no_loop_detection:
        d["loop_detection"] = False
    if args.product_verification:
        d["verification_seed"] = None
    if args.augment_kwargs:
        d["augment_kwargs"] = json.loads(args.augment_kwargs)
    if args.gate_params:
        d["gate_params"] = json.loads(args.gate_params)
    if args.extras:
        d["extra_images"] = json.loads(Path(args.extras).read_text(encoding="utf-8"))
    if args.database_from:
        d["database_from"] = args.database_from
    if args.reference_seed is not None:
        d["reference_seed"] = args.reference_seed if args.reference_seed == "medoid" else int(args.reference_seed)
    return VariantConfig.from_json(d)


def cmd_run(args) -> int:
    from tower.world_builder.coherence_exp.driver import run_variant

    config = _config_from_args(args)
    result = run_variant(args.world, args.captures, args.out, config, cache_root=args.cache_root,
                         regions_dir=args.regions_dir)
    print(json.dumps({k: result[k] for k in ("out_dir", "export", "timings", "gate_splits")}, indent=1))
    return 0


def cmd_masks(args) -> int:
    from tower.world_builder.coherence_exp import driver, masks

    world = driver.open_world(args.world)
    cache_root = args.cache_root or driver.default_cache_root()
    cdir = masks.cache_dir(cache_root, world.world_id)
    extras = []
    if args.extras:
        extras = json.loads(Path(args.extras).read_text(encoding="utf-8"))
    config = driver.VariantConfig(name="masks", extra_images=extras, keyframe_source=args.keyframe_source)
    staging_dir = cdir / "_staged" / args.keyframe_source
    cindex = driver.capture_index(args.world, args.captures)
    staging = driver.stage_images(world, cindex, staging_dir, config)
    images = [{"name": r["name"], "sha1": r["sha1"], "path": str(staging_dir / "images" / r["name"])}
              for r in staging["images"]]
    if args.limit:
        images = images[:args.limit]
    report = masks.compute_masks(images, cdir, masks.default_params())
    (cdir / "compute_report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("images", "rule")}, indent=1))
    for comp, rep in report["components"].items():
        print(f"{comp}: todo {rep['todo']}, computed {rep['computed']}, {rep.get('seconds')} s")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(q):
        q.add_argument("--world", type=artifact_root_arg, required=True, help="frozen world dir (read-only)")
        q.add_argument("--captures", type=artifact_root_arg, required=True,
                       help="dir holding <capture_id>/ (frozen: RUN/baseline/frozen/captures)")
        q.add_argument("--cache-root", type=artifact_root_arg, default=None,
                       help="masks/db/map caches (default RUN/experiments/cache or $WB_COHERENCE_EXP_CACHE)")
        q.add_argument("--extras", default=None, help="JSON list of ExtraImage dicts (solver-only images)")

    r = sub.add_parser("run", help="run one variant")
    common(r)
    r.add_argument("--out", type=artifact_root_arg, required=True)
    r.add_argument("--config", default=None, help="VariantConfig JSON; flags override it")
    r.add_argument("--name", default=None)
    r.add_argument("--masks", choices=("none", "transients"), default=None)
    r.add_argument("--gate", choices=("none", "rigid"), default=None)
    r.add_argument("--gate-params", default=None, help="JSON overrides of gate.GateParams")
    r.add_argument("--mapper", choices=("glomap", "incremental"), default=None)
    r.add_argument("--seeds", default=None, help="comma-separated, e.g. 0,1,2,3,4")
    r.add_argument("--threads", type=int, default=None, help="mapping threads (1 = deterministic)")
    r.add_argument("--max-parallel", type=int, default=None)
    r.add_argument("--no-loop-detection", action="store_true")
    r.add_argument("--product-verification", action="store_true",
                   help="leave the two-view RANSAC seed at the product's -1 (non-deterministic)")
    r.add_argument("--augment", default=None, help="package.module:function")
    r.add_argument("--augment-kwargs", default=None, help="JSON dict")
    r.add_argument("--keyframe-source", choices=("raw", "redacted"), default=None)
    r.add_argument("--database-from", default=None,
                   help="seed the database from this existing COLMAP database (copied; e.g. the frozen "
                        "product solve's, which carries the live matching history)")
    r.add_argument("--reference-seed", default=None, help="'medoid' (default) or a seed number")
    r.add_argument("--regions-dir", type=artifact_root_arg, default=None,
                   help="C0 labels (evaluation only; components.json histogram)")
    r.set_defaults(func=cmd_run)

    m = sub.add_parser("masks", help="compute transient masks (GPU; run under gpulock)")
    common(m)
    m.add_argument("--limit", type=int, default=None, help="only the first N staged images (smoke)")
    m.add_argument("--keyframe-source", choices=("raw", "redacted"), default="raw",
                   help="mask the raw keyframes (product with capture) or the redacted session copies")
    m.set_defaults(func=cmd_masks)

    o = sub.add_parser("map-one", help="internal: one seeded mapping run")
    o.add_argument("rest", nargs=argparse.REMAINDER)
    o.set_defaults(func=lambda a: __import__("tower.world_builder.coherence_exp.driver",
                                             fromlist=["_main"])._main(["map-one", *a.rest]))

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
