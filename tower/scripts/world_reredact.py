#!/usr/bin/env python
"""Re-redact a saved world's keyframes with the CURRENT face redactor, explicitly.

For worlds captured under an older rule of the same YuNet@0.30 family (the
ungated rule, `+plausibility1`, `+plausibility2`), whose false positives black
out hands, walls and furniture. The current rule runs on each keyframe's
verified raw capture frame; a new keyframe set is written BESIDE the stored
one and the session is switched to it by a pointer written last. The stored
`sessions/<sid>/images/` is never modified. Contract:
`docs/contracts/WORLD-BUILDER-APPEARANCE.md` section 6.5.

    .venv\\Scripts\\python.exe scripts/world_reredact.py --root <root> --world <id> --dry-run
    .venv\\Scripts\\python.exe scripts/world_reredact.py --root <root> --world <id> --apply
    .venv\\Scripts\\python.exe scripts/world_reredact.py --root <root> --world <id> --revert

`--dry-run` reports what is recoverable and writes nothing. `--apply` writes the
set and switches; then rebuild the surface (`world_surface.py`, which refits the
depth of the changed frames) and the appearance (`world_appearance.py --force`).
`--revert` switches back to the stored keyframes: a pointer change only.

Raw frame paths come from the solve's `sources.json`, resolved against the Tower
root (`--tower-root`, else `TOWER_SOURCES_ROOT`, else this Tower's `tower/`),
never against the current directory. Nothing here runs during a walk.

Exit codes: 0 done (or nothing to do on --dry-run); 1 refused or failed for
every session; 2 bad arguments.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tower.artifact_paths import artifact_root_arg  # noqa: E402
from tower.world_builder import reredaction as RR  # noqa: E402
from tower.world_builder.global_solve import sources_root  # noqa: E402
from tower.world_builder.store import WorldStore  # noqa: E402

DEFAULT_ROOT = Path("data") / "world_builder"


def _pct(x) -> str:
    return "n/a" if x is None else f"{100.0 * x:.2f}%"


def summarise(plan: RR.Plan, top: int = 10) -> dict:
    totals = plan.totals()
    worst = sorted(plan.recovered_frames, key=lambda f: -f.recovered_px)[:top]
    return {
        "world": plan.world_id, "session": plan.session_id,
        "stored_redaction": plan.stored_redaction, "set_redaction": plan.set_label,
        "set": plan.name, "origins": plan.counts, "totals": totals,
        "largest_recoveries": [
            {"seq": f.seq, "stored_fill": _pct(f.stored_fill_px / f.pixels if f.pixels else None),
             "set_fill": _pct(f.new_fill_px / f.pixels if f.pixels else None)}
            for f in worst],
        "kept_with_reason": [
            {"seq": f.seq, "origin": f.origin, "detail": f.detail}
            for f in plan.frames
            if not f.reredacted and f.origin != RR.KEPT_NOTHING_RECOVERED][:top],
        "refusal": plan.refusal, "seconds": round(plan.seconds, 1),
    }


def _print_summary(s: dict) -> None:
    t = s["totals"]
    print(f"  stored label : {s['stored_redaction']}")
    print(f"  set label    : {s['set_redaction']}  ({s['set']})")
    print(f"  frames       : {t['frames']}  measured against raw: {t['frames_measured']}")
    for origin, n in s["origins"].items():
        print(f"    {origin:<34} {n}")
    print(f"  fill (difference against raw, measured frames):")
    print(f"    stored     : {_pct(t['stored_fill_fraction'])}")
    print(f"    after      : {_pct(t['set_fill_fraction'])}")
    print(f"    recovered  : {t['recovered_px']} px in {t['frames_reredacted']} frames")
    if s["largest_recoveries"]:
        print("  largest recoveries (seq: stored -> set):")
        for r in s["largest_recoveries"]:
            print(f"    {r['seq']}: {r['stored_fill']} -> {r['set_fill']}")
    if s["kept_with_reason"]:
        print("  kept for a reason (first few):")
        for r in s["kept_with_reason"]:
            print(f"    {r['seq']}: {r['origin']} {r['detail'] or ''}")
    if s["refusal"]:
        print(f"  WILL NOT SWITCH: {s['refusal']}")
    print(f"  {s['seconds']}s")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=artifact_root_arg, default=str(DEFAULT_ROOT),
                    help="world root (routed through the artifact-path guard)")
    ap.add_argument("--world", required=True)
    ap.add_argument("--session")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", dest="dry_run", action="store_true",
                      help="report recoverable fill; write nothing")
    mode.add_argument("--apply", action="store_true",
                      help="write the re-redacted set beside images/ and switch to it")
    mode.add_argument("--revert", action="store_true",
                      help="switch back to the stored keyframes (a pointer change)")
    ap.add_argument("--tower-root", dest="tower_root",
                    help="what relative sources.json paths are relative to "
                         f"(default: {sources_root()})")
    ap.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = ap.parse_args(argv)

    store = WorldStore(Path(args.root))
    try:
        world = store.read_world(args.world)
    except Exception as exc:  # noqa: BLE001 -- reported, not a traceback
        print(f"cannot read world {args.world}: {exc}", file=sys.stderr)
        return 1
    sessions = [args.session] if args.session else list(world.session_ids)
    tower_root = Path(args.tower_root).resolve() if args.tower_root else None
    failures = 0
    for sid in sessions:
        print(f"=== {args.world} / {sid} ===", flush=True)
        try:
            if args.revert:
                done = RR.revert_session(store, args.world, sid)
                print(f"  reverted: reads images/ again (was {done['reverted_from']})")
                continue
            plan = RR.plan_session(store, args.world, sid, tower_root=tower_root)
            summary = summarise(plan)
            if args.json:
                print(json.dumps(summary, indent=2))
            else:
                _print_summary(summary)
            if args.apply:
                done = RR.apply_plan(store, plan)
                print(f"  SWITCHED to {done['set']} (set digest {done['set_digest']}"
                      f"{', reused an identical set on disk' if done['reused'] else ''})")
                print("  next: world_surface.py (refits the changed frames' depth), then "
                      "world_appearance.py --force")
        except RR.ReredactionRefused as exc:
            print(f"  refused, nothing switched: {exc}")
            failures += 1
        except Exception as exc:  # noqa: BLE001 -- any failure switches nothing
            print(f"  failed, nothing switched: {type(exc).__name__}: {exc}")
            failures += 1
    return 1 if failures and failures == len(sessions) else 0


if __name__ == "__main__":
    raise SystemExit(main())
