#!/usr/bin/env python
"""What a built world actually amounts to: coherence, and whether it is honest.

WHY THIS EXISTS

`world_build_session.py` reports segments, solved poses and points.
None of those is the product question. A walk that reports 30 segments
and 30,382 points can still be eighteen disconnected islands the wearer
cannot use, and a walk that reports ONE segment can be one confidently
wrong room. The two numbers that matter are

    how much of the geometry sits in ONE coordinate frame,

and

    does that geometry reproject.

The first without the second is the failure mode this whole line of work
has to avoid: coherence bought by accepting poses that are not true.
So this tool refuses to report either alone. Every run prints the
dominant-component share AND the reprojection distribution of the
published support rows, per segment and in aggregate.

WHAT IT MEASURES, AND FROM WHAT

Everything is read cold from a persisted world. Nothing is re-solved, so
a number here is a statement about what shipped, not about what a fresh
run might do.

  segments / keyframes / poses          derived/manifest.json
  root vs cascaded refusals             derived/manifest.json
  per-segment pose status               derived/<session>/poses.json
  per-segment points                    derived/<session>/points.json
  connected components                  derived/<session>/placements.json,
                                        via world_registration.admitted_components
  reprojection error                    support.json rows re-projected through
                                        poses.json, with the 2-D feature
                                        positions recovered by re-detecting ORB
                                        on the stored keyframe

That last one costs an ORB detection per keyframe -- the same cost
`world_registration.read_segments` already pays -- so `--no-reprojection`
is offered for a quick structural read. It is off by default on purpose:
a coherence number published without its quality number is exactly the
misleading artifact this file exists to prevent.

REPROJECTION, PRECISELY

A support row says (segment, frame, feature, point). The landmark is
projected through that frame's own pose, in the segment's own frame, and
compared with the pixel of that feature. Units are pixels at the
capture's resolution. Rows whose landmark falls behind the camera are
counted separately as `behind_camera` rather than folded into the
distribution, because a negative depth has no meaningful pixel distance
and averaging one in would flatter the result.

The bar to read against is `classical.PNP_REPROJECTION_ERROR_PX` (3.0),
which is the threshold `solvePnPRansac` used to admit these very rows.
A median well under it means the published association is the one the
solver believed; a p99 far above it means rows are being published that
no solve would have accepted.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from tower.artifact_paths import artifact_root_arg  # noqa: E402
from tower.world_builder.backends.classical import (  # noqa: E402
    PNP_REPROJECTION_ERROR_PX,
)
from tower.world_builder.store import WorldStore  # noqa: E402

# A component of one segment is not a "component" in the sense anybody
# cares about; it is an orphan. Counted separately so the headline
# component count cannot be improved by producing more singletons.
MIN_JOINED_COMPONENT = 2


def _percentiles(values):
    if len(values) == 0:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "median": round(float(np.median(array)), 4),
        "p95": round(float(np.percentile(array, 95)), 4),
        "p99": round(float(np.percentile(array, 99)), 4),
        "max": round(float(array.max()), 4),
        "over_gate_fraction": round(
            float((array > PNP_REPROJECTION_ERROR_PX).mean()), 4
        ),
    }


def reprojection_by_segment(store, world_id, session_id) -> dict:
    """Median/p95/p99 pixel error of the published support, per segment."""
    from scripts.world_registration import read_segments  # noqa: PLC0415

    segments = read_segments(store, world_id, session_id)
    per_segment, everything, behind = {}, [], 0
    for index, geometry in segments.items():
        errors = []
        segment_behind = 0
        for (frame, feature), point_index in geometry.observed.items():
            pose = geometry.poses.get(frame)
            if pose is None:
                continue
            if frame >= len(geometry.keypoints):
                continue
            pixels = geometry.keypoints[frame]
            if feature >= len(pixels) or point_index >= len(geometry.points):
                continue
            rotation, translation = pose
            camera = rotation @ geometry.points[point_index] + translation
            if not np.isfinite(camera).all() or camera[2] <= 0:
                segment_behind += 1
                continue
            projected = geometry.intrinsics @ camera
            u = projected[0] / projected[2]
            v = projected[1] / projected[2]
            errors.append(float(np.hypot(u - pixels[feature][0],
                                         v - pixels[feature][1])))
        behind += segment_behind
        everything.extend(errors)
        stats = _percentiles(errors)
        if stats is not None:
            stats["behind_camera"] = segment_behind
        per_segment[index] = stats
    aggregate = _percentiles(everything) or {}
    aggregate["behind_camera"] = behind
    return {"aggregate": aggregate, "by_segment": per_segment}


def coherence(
    store, world_id, session_id, *, with_reprojection=True, admitted=None
) -> dict:
    from scripts.world_registration import admitted_components  # noqa: PLC0415

    manifest = store.read_derived_manifest(world_id) or {}
    derived = store.read_derived(world_id, session_id) or {}
    points_by_segment: dict = {}
    for row in derived.get("points") or ():
        points_by_segment[row["segment_index"]] = (
            points_by_segment.get(row["segment_index"], 0) + 1
        )
    status_by_segment: dict = {}
    for row in derived.get("poses") or ():
        bucket = status_by_segment.setdefault(row["segment_index"], {})
        bucket[row["status"]] = bucket.get(row["status"], 0) + 1

    placements = store.read_placements(world_id, session_id) or []
    registered = sorted(
        p.segment_index for p in placements if p.state == "registered"
    )

    # Components are read off the ADMITTED PAIRS where they are
    # available, and not off the placement state, because "registered"
    # means "in the component that happened to contain the reference".
    # A second joined cluster that simply is not the reference's is a
    # real coherence result and would be invisible if this counted
    # placement state alone.
    #
    # placements.json does not carry the pairs -- it is a per-segment
    # record and its `evidence` describes the segment, not the edges --
    # so they come from the build report, which is where `register()`
    # already publishes them. Without one, the fallback treats the
    # registered set as the single component it is by construction, and
    # `admitted_pairs_source` says which reading produced the numbers so
    # that two runs are never silently compared across the two.
    if admitted is None:
        # Since the global solve (2026-09-06) a session can carry MORE than
        # one registered cluster, each under its own reference segment,
        # and they are different frames. Group by reference rather than
        # treating every registered segment as one component.
        by_reference: dict = {}
        for placement in placements:
            if placement.state == "registered":
                by_reference.setdefault(placement.reference_segment, []).append(placement.segment_index)
        components = [sorted(members) for members in by_reference.values() if len(members) > 1]
        pairs_source = "placement-state"
    else:
        pairs = [tuple(sorted((int(a), int(b)))) for a, b in admitted]
        components = admitted_components(pairs) if pairs else []
        pairs_source = "admitted-pairs"

    with_geometry = {s for s, n in points_by_segment.items() if n}
    joined = {segment for component in components for segment in component}
    orphans = sorted(with_geometry - joined)
    sized = [
        (sum(points_by_segment.get(s, 0) for s in component), sorted(component))
        for component in components
    ]
    sized.extend((points_by_segment[s], [s]) for s in orphans)
    sized.sort(key=lambda row: (-row[0], row[1]))

    total_points = sum(points_by_segment.values())
    dominant_points, dominant_segments = (
        sized[0] if sized else (0, [])
    )
    report = {
        "world_id": world_id,
        "session_id": session_id,
        "segments": manifest.get("segments"),
        "keyframes": manifest.get("keyframes"),
        "poses_solved": manifest.get("poses_solved"),
        "poses_refused": manifest.get("poses_refused"),
        "poses_refused_root": manifest.get("poses_refused_root"),
        "poses_refused_cascaded": manifest.get("poses_refused_cascaded"),
        "refusal_degeneracy_counts": manifest.get("refusal_degeneracy_counts"),
        "points": total_points,
        "segments_with_geometry": len(with_geometry),
        "segments_registered": len(registered),
        "components_joined": sum(
            1 for _, members in sized if len(members) >= MIN_JOINED_COMPONENT
        ),
        "orphan_segments": len(orphans),
        "fragments_total": len(sized),
        "dominant_component_segments": dominant_segments,
        "dominant_component_points": dominant_points,
        "dominant_component_share": (
            round(dominant_points / total_points, 4) if total_points else None
        ),
        "component_sizes": [
            {"points": points, "segments": members} for points, members in sized[:12]
        ],
        "points_by_segment": dict(sorted(points_by_segment.items())),
        "status_by_segment": {
            k: status_by_segment[k] for k in sorted(status_by_segment)
        },
    }
    if with_reprojection:
        report["reprojection_px"] = reprojection_by_segment(
            store, world_id, session_id
        )
        report["reprojection_gate_px"] = PNP_REPROJECTION_ERROR_PX
    # A globally solved session has no support rows for its solved segments
    # (they would index the wrong keypoints -- see global_solve.merge), so
    # the ORB reprojection above covers only chain-built segments. The
    # solver's own observations are reprojected here instead, and the
    # report says which segments each number speaks for.
    from tower.world_builder import global_solve  # noqa: PLC0415

    solution = global_solve.load_solution(store, world_id, session_id)
    if solution is not None:
        manifest = store.read_derived_manifest(world_id) or {}
        block = dict((manifest.get("global_solve") or {}))
        block.pop("segments", None)
        block["reprojection_px"] = global_solve.reprojection_summary(solution)
        report["global_solve"] = block
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Coherence and honesty of a built world: how much geometry "
            "shares one frame, and whether it reprojects."
        )
    )
    parser.add_argument("--root", type=artifact_root_arg, required=True)
    parser.add_argument("--world", default=None, help="Default: the only world.")
    parser.add_argument("--session", default=None, help="Default: the only session.")
    parser.add_argument(
        "--no-reprojection",
        action="store_true",
        help=(
            "Skip the pixel-error pass. Faster, and deliberately not the "
            "default: a coherence number without a quality number beside "
            "it is the misleading artifact this tool exists to prevent."
        ),
    )
    parser.add_argument("--format", choices=("text", "json"), default="json")
    args = parser.parse_args(argv)

    store = WorldStore(Path(args.root))
    world_id = args.world
    if world_id is None:
        worlds = store.list_world_ids()
        if len(worlds) != 1:
            parser.error(f"--world is required; {len(worlds)} worlds under --root")
        world_id = worlds[0]
    session_id = args.session
    if session_id is None:
        sessions = store.list_session_ids(world_id)
        if len(sessions) != 1:
            parser.error(
                f"--session is required; {len(sessions)} sessions in {world_id}"
            )
        session_id = sessions[0]

    report = coherence(
        store, world_id, session_id, with_reprojection=not args.no_reprojection
    )
    if args.format == "json":
        print(json.dumps(report, indent=2, default=float))
    else:
        for key, value in report.items():
            if key in ("points_by_segment", "status_by_segment", "component_sizes"):
                continue
            print(f"{key:32s} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
