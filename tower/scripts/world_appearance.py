#!/usr/bin/env python
"""Build the appearance artifact of a saved world, offline.

The appearance is the wearer's own redacted keyframes, prepared to be blended
by view over the surface on the phone: masked (redaction fill, near-black
boxes, the wearer's hands and phone), exposure-balanced, chosen for coverage,
and bundled as GPU texture chunks. Contract:
`docs/contracts/WORLD-BUILDER-APPEARANCE.md`.

The SAME `build_appearance()` runs in the product -- after each live surface
during a walk and after the final surface at Stop. This is a front end to it.

    .venv\\Scripts\\python.exe scripts/world_appearance.py --root <root> --world <id>
    .venv\\Scripts\\python.exe scripts/world_appearance.py --root <root> --world <id> --force
    .venv\\Scripts\\python.exe scripts/world_appearance.py --root <root> --world <id> --inspect
    .venv\\Scripts\\python.exe scripts/world_appearance.py --root <root> --world <id> --live

Needs the session's global solve, its surface artifact, and the depth stage's
per-frame work under `dense/<session>/work/` (run `world_surface.py --force`
first if it was pruned). Writes only under `<world>/appearance/<session>/`.

Exit codes: 0 built or already built; 1 every target failed; 2 bad arguments.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tower.artifact_paths import artifact_root_arg  # noqa: E402
from tower.world_builder.appearance import AppearanceParams  # noqa: E402
from tower.world_builder.appearance_pipeline import (  # noqa: E402
    STATE_OK,
    appearance_currency,
    build_appearance,
    read_appearance_manifest,
)
from tower.world_builder.global_solve import load_solution  # noqa: E402
from tower.world_builder.store import WorldStore  # noqa: E402

DEFAULT_ROOT = Path("data") / "world_builder"


def _sessions_with_solves(store, world_id):
    world = store.read_world(world_id)
    return [sid for sid in world.session_ids
            if load_solution(store, world_id, sid) is not None]


def _print_progress(stage, done, total):
    print(f"    {stage}: {done}/{total}", flush=True)


def _inspect(store, world_id, sid) -> dict:
    man = read_appearance_manifest(store, world_id, sid)
    if man is None:
        return {"present": False}
    kfs = man.get("keyframes") or []
    chunks = man.get("chunks") or []
    by_enc: dict = {}
    for c in chunks:
        e = by_enc.setdefault(c["encoding"], {"chunks": 0, "bytes": 0, "slots": 0})
        e["chunks"] += 1
        e["bytes"] += c["bytes"]
        e["slots"] += c["slots"]
    return {
        "present": True, "format": man["format"], "build_id": man["build_id"],
        "quality": man["quality"], "keyframes": len(kfs),
        "phone": sum(1 for k in kfs if k["tier"] == "phone"),
        "excluded": len(man.get("excluded") or []),
        "encodings": by_enc, "proxy": man["proxy"],
        "provenance": man["appearance_provenance"],
        "selection": man["selection"], "exposure": man["exposure"],
        "occluders": man["occluders"], "seconds": man["seconds"],
        "currency": appearance_currency(store, world_id, sid, man),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=artifact_root_arg, default=str(DEFAULT_ROOT),
                    help="world root (routed through the artifact-path guard)")
    ap.add_argument("--world", required=True)
    ap.add_argument("--session")
    ap.add_argument("--force", action="store_true", help="rebuild even if already built")
    ap.add_argument("--inspect", action="store_true",
                    help="report what is on disk and whether it is current")
    ap.add_argument("--live", action="store_true",
                    help="the walk-time preset: the same rules, less sampling")
    ap.add_argument("--phone-budget", dest="phone_budget", type=int)
    ap.add_argument("--device", help="torch device for the GPU stages (default: cuda if available)")
    args = ap.parse_args(argv)

    store = WorldStore(Path(args.root))
    sessions = [args.session] if args.session else _sessions_with_solves(store, args.world)
    if not sessions:
        print("no solved session in that world", file=sys.stderr)
        return 1

    if args.inspect:
        for sid in sessions:
            print(f"=== {args.world} / {sid} ===")
            print(json.dumps(_inspect(store, args.world, sid), indent=2))
        return 0

    overrides = {}
    if args.phone_budget is not None:
        overrides["phone_budget"] = args.phone_budget
    params = AppearanceParams.live(**overrides) if args.live else AppearanceParams(**overrides)
    failures = 0
    for sid in sessions:
        print(f"=== {args.world} / {sid} ===", flush=True)
        t = time.time()
        result = build_appearance(store, args.world, sid, params=params, force=args.force,
                                  progress=_print_progress, device=args.device)
        if result.state != STATE_OK:
            print(f"  {result.state}: {result.detail}")
            failures += 1
            continue
        if result.detail:
            print(f"  {result.detail}")
        print(f"  {result.keyframes} keyframes ({result.phone} for the phone), "
              f"{result.excluded} excluded, {result.chunks} chunks, "
              f"{result.bytes / 1e6:.1f} MB")
        print(f"  {time.time() - t:.1f}s total  {result.seconds}")
    return 1 if failures and failures == len(sessions) else 0


if __name__ == "__main__":
    raise SystemExit(main())
