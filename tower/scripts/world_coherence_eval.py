#!/usr/bin/env python
"""One evaluation harness for every reconstruction of a frozen world.

The saved world (baseline) and every future variant -- COLMAP incremental,
GLOMAP with other pairs, hloc/LightGlue, learned SLAM, pose-graph variants --
are measured by the SAME code against the SAME per-world caches, so a claim
that a variant is better is a statement about identical measurements.

SUBCOMMANDS

  cache    build the per-world caches (images only, solver-independent):
             --what depth   MoGe-2 metric depth per keyframe (GPU)
             --what pairs   verified image pairs: adjacent + retrieved revisits,
                            SIFT + essential-matrix RANSAC (CPU)
  eval     metrics.json + metrics.md for one variant of one world
  compare  a diff table of two metrics.json
  table    headline metrics of several metrics.json side by side
  measure  run a command, record wall time, peak RSS and peak VRAM
  adapt    write a world or a COLMAP model in the interchange format

EXAMPLES (Windows; use the Tower venv's python by full path)

  world_coherence_eval.py cache --world <frozen>\\worlds\\<wid> --cache-root <run>\\baseline\\metrics\\cache --what all
  world_coherence_eval.py eval  --world <frozen>\\worlds\\<wid> --cache-root <...>\\cache --out <...>\\metrics\\<wid>
  world_coherence_eval.py eval  --world <...> --variant <exp>\\sparse --cache-root <...> --out <exp>\\metrics
  world_coherence_eval.py compare a\\metrics.json b\\metrics.json
  world_coherence_eval.py measure --out <exp>\\runtime.json --variant <exp>\\variant -- python solve.py ...

See `tower/world_builder/coherence_eval/eval_variant.py` for the interchange
format and `metrics.py` for every metric's definition and limitations.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# numpy/scipy on the main thread before anything spawns threads (the
# loader-lock rule, tower/native_prewarm.py).
import numpy as np  # noqa: E402,F401
import scipy.linalg  # noqa: E402,F401
import scipy.stats  # noqa: E402,F401

from tower.artifact_paths import artifact_root_arg  # noqa: E402

CACHE_ENV = "WB_COHERENCE_CACHE"


def _cache_root(args):
    root = args.cache_root or os.environ.get(CACHE_ENV)
    if not root:
        raise SystemExit(f"--cache-root is required (or set {CACHE_ENV})")
    return artifact_root_arg(str(root))


def cmd_cache(args) -> int:
    from tower.world_builder.coherence_eval.eval_world import open_world

    world = open_world(args.world, args.session)
    root = _cache_root(args)
    what = {"all": ("pairs", "depth")}.get(args.what, (args.what,))
    if "pairs" in what:
        from tower.world_builder.coherence_eval.eval_pairs import build_pair_cache

        m = build_pair_cache(world, root, workers=args.workers, log=lambda s: print(s, flush=True),
                             allow_descriptor_fallback=args.allow_descriptor_fallback)
        print(json.dumps({k: m[k] for k in ("counts", "seconds") if k in m}, indent=1))
    if "depth" in what:
        from tower.world_builder.coherence_eval.eval_depth import build_depth_cache

        m = build_depth_cache(world, root, log=lambda s: print(s, flush=True))
        print(json.dumps({k: m.get(k) for k in ("complete", "new_maps_this_run", "seconds_this_run",
                                                  "peak_vram_mb_this_run")}, indent=1))
    return 0


def _load_variant(args, world):
    from tower.world_builder.coherence_eval import eval_variant as ev

    if args.variant is None:
        return ev.variant_from_world(world)
    kwargs = {}
    if getattr(args, "publish_min_observations", None) is not None:
        kwargs["publish_min_observations"] = args.publish_min_observations
    if getattr(args, "publish_min_model_images", None) is not None:
        kwargs["publish_min_model_images"] = args.publish_min_model_images
    if getattr(args, "image_space", None):
        kwargs["image_space"] = args.image_space
    v = ev.load_any(Path(args.variant), world, **kwargs)
    if args.name:
        v.name = args.name
        v.meta["variant"] = args.name
    return v


def cmd_eval(args) -> int:
    from tower.world_builder.coherence_eval import metrics
    from tower.world_builder.coherence_eval.eval_world import open_world

    world = open_world(args.world, args.session)
    variant = _load_variant(args, world)
    cache_root = _cache_root(args)
    regions_dir = Path(args.regions_dir) if args.regions_dir else None
    result = metrics.evaluate_world_variant(
        world, variant, cache_root=cache_root, regions_dir=regions_dir,
        renders=args.renders, out_dir=args.out)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    metrics.write_outputs(result, out)
    print(f"wrote {out / 'metrics.json'} and {out / 'metrics.md'}")
    return 0


def cmd_compare(args) -> int:
    from tower.world_builder.coherence_eval import metrics

    a = json.loads(Path(args.a).read_text(encoding="utf-8"))
    b = json.loads(Path(args.b).read_text(encoding="utf-8"))
    text = metrics.compare_markdown(a, b)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


def cmd_table(args) -> int:
    from tower.world_builder.coherence_eval import metrics

    results = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.files]
    text = metrics.table_markdown(results, args.names.split(",") if args.names else None)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


def cmd_measure(args) -> int:
    from tower.world_builder.coherence_eval.metrics import measure_command

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("measure: give the command after --")
    record = measure_command(command, poll_s=args.poll)
    text = json.dumps(record, indent=1, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    if args.variant:
        p = Path(args.variant) / "reconstruction.json"
        if p.is_file():
            doc = json.loads(p.read_text(encoding="utf-8"))
            doc.setdefault("meta", {})["runtime"] = record
            p.write_text(json.dumps(doc, indent=1, sort_keys=True), encoding="utf-8")
    print(text)
    return int(record.get("returncode") or 0)


def cmd_adapt(args) -> int:
    from tower.world_builder.coherence_eval import eval_variant as ev
    from tower.world_builder.coherence_eval.eval_world import open_world

    world = open_world(args.world, args.session)
    v = _load_variant(args, world)
    ev.save_variant(v, args.out)
    print(f"wrote {args.out} ({len(v.poses)} posed keyframes, "
          f"{0 if v.xyz is None else len(v.xyz)} points)")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def world_args(q):
        q.add_argument("--world", required=True, help="the frozen world directory (.../worlds/<world_id>)")
        q.add_argument("--session", default=None, help="session id (default: the only one)")

    def cache_arg(q):
        q.add_argument("--cache-root", type=artifact_root_arg, default=None,
                       help=f"per-world caches live in <cache-root>/<world_id>/ (or ${CACHE_ENV})")

    def variant_args(q):
        q.add_argument("--variant", default=None,
                       help="interchange dir (reconstruction.json), COLMAP model dir (or dir of numbered "
                            "models), or a world dir. Default: the saved world itself (the baseline).")
        q.add_argument("--name", default=None, help="variant name to record")
        q.add_argument("--publish-min-observations", type=int, default=None,
                       help="COLMAP input: publish floor on 3-D observations per image (default 30)")
        q.add_argument("--publish-min-model-images", type=int, default=None,
                       help="COLMAP input: minimum registered images for a model to publish (default 5)")
        q.add_argument("--image-space", choices=("canonical", "raw", "custom"), default=None,
                       help="COLMAP input: pixel space of the observations (default: inferred from size)")

    c = sub.add_parser("cache", help="build per-world caches (depth: GPU; pairs: CPU)")
    world_args(c)
    cache_arg(c)
    c.add_argument("--what", choices=("depth", "pairs", "all"), default="all")
    c.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    c.add_argument("--allow-descriptor-fallback", action="store_true",
                   help="pairs: use tiny-image retrieval if DINOv2 cannot load (recorded; not comparable)")
    c.set_defaults(func=cmd_cache)

    e = sub.add_parser("eval", help="metrics for one variant")
    world_args(e)
    cache_arg(e)
    variant_args(e)
    e.add_argument("--out", type=artifact_root_arg, required=True)
    e.add_argument("--regions-dir", default=None,
                   help="dir with <world_id>_regions.csv / <world_id>_revisits.csv (evaluation-only labels)")
    e.add_argument("--renders", action="store_true",
                   help="also render the fixed viewpoint set (needs coherence_eval.layer_renders)")
    e.set_defaults(func=cmd_eval)

    k = sub.add_parser("compare", help="diff two metrics.json")
    k.add_argument("a")
    k.add_argument("b")
    k.add_argument("--out", type=artifact_root_arg, default=None)
    k.set_defaults(func=cmd_compare)

    t = sub.add_parser("table", help="headline metrics of several metrics.json side by side")
    t.add_argument("files", nargs="+")
    t.add_argument("--names", default=None, help="comma-separated column names")
    t.add_argument("--out", type=artifact_root_arg, default=None)
    t.set_defaults(func=cmd_table)

    m = sub.add_parser("measure", help="run a command; record wall time, peak RSS, peak VRAM")
    m.add_argument("--out", type=artifact_root_arg, default=None, help="runtime JSON to write")
    m.add_argument("--variant", default=None, help="interchange dir whose meta.runtime to fill")
    m.add_argument("--poll", type=float, default=0.5, help="seconds between samples")
    m.add_argument("command", nargs=argparse.REMAINDER)
    m.set_defaults(func=cmd_measure)

    a = sub.add_parser("adapt", help="write the interchange format")
    world_args(a)
    variant_args(a)
    a.add_argument("--out", type=artifact_root_arg, required=True)
    a.set_defaults(func=cmd_adapt)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
