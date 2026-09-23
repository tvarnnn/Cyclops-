"""Render a world's layers from the deterministic coherence viewpoint set.

Read-only against the world: it reads ``solve/``, ``surface/`` and
``sessions/`` of WORLD_DIR (a frozen world or a variant directory of the same
shape) and writes PNGs, contact sheets and an index under --out.

    python scripts/world_layer_renders.py WORLD_DIR --out OUT_DIR \
        [--session SID] [--viewpoints VIEWPOINTS.json] [--regions REGIONS.csv] \
        [--layers cameras,sparse,sparse_time,surface,surface_rgb] [--level 0]

Without --viewpoints the rule of
``tower.world_builder.coherence_eval.viewpoints`` is applied to the world's
own trajectory and the set is written to OUT_DIR/viewpoints.json.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# prewarm the BLAS-backed modules on the main thread (Windows loader lock)
import numpy  # noqa: F401
import scipy  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tower.artifact_paths import artifact_root_arg  # noqa: E402
from tower.world_builder.coherence_eval import layer_renders as lr  # noqa: E402
from tower.world_builder.coherence_eval import viewpoints as vp  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("world_dir", help="world (or variant) directory; only read")
    ap.add_argument("--out", required=True, type=artifact_root_arg,
                    help="output directory for PNGs, sheets and index.json")
    ap.add_argument("--session", default=None)
    ap.add_argument("--viewpoints", default=None,
                    help="a viewpoints JSON to reuse instead of deriving one")
    ap.add_argument("--regions", default=None,
                    help="region-label CSV (forensic annotation; colouring only)")
    ap.add_argument("--layers", default=",".join(lr.LAYERS_DEFAULT))
    ap.add_argument("--level", type=int, default=0, help="published surface level")
    ap.add_argument("--orbit-scale", type=float, default=1.0)
    ap.add_argument("--surface-dir", default=None,
                    help="evaluate a surface built elsewhere (same session) instead of "
                         "WORLD_DIR/surface/<sid>")
    ap.add_argument("--insitu", action="store_true",
                    help="also write photo-vs-surface in-situ comparisons and "
                         "per-keyframe surface metrics (surface_keyframe_metrics.csv)")
    ap.add_argument("--skip-layers", action="store_true",
                    help="do not re-render the layer views (with --insitu)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.viewpoints:
        vs = vp.load_viewpoints(a.viewpoints)
    else:
        vs = vp.viewpoints_for_world(a.world_dir, a.session)
        vp.save_viewpoints(vs, out / "viewpoints.json")
    if not a.skip_layers:
        lr.render_world(a.world_dir, out, session_id=a.session, viewpoints=vs,
                        regions_csv=a.regions, layers=[s for s in a.layers.split(",") if s],
                        level=a.level, orbit_scale=a.orbit_scale, surface_dir=a.surface_dir)
    if a.insitu:
        lr.insitu_and_metrics(a.world_dir, out, session_id=a.session, viewpoints=vs,
                              regions_csv=a.regions, level=a.level, surface_dir=a.surface_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
