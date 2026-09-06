#!/usr/bin/env python
"""Look at a built world. Offline, read-only, honest about what is placed.

WHY THIS EXISTS

`world_inspect.py` counts and `world_coherence_report.py` measures. Neither
lets a human LOOK at the room. This tool turns a persisted derived tree
(points.json, poses.json, placements.json) into things a person can open:

    world.ply                world-frame points, one colour per segment
    cameras_world.ply        world-frame camera centres
    view_top.png             orthographic render looking along +y
    view_front.png           orthographic render looking along +z
    view_side.png            orthographic render looking along +x
    view_iso.png             3/4 view
    overview.png             the four views on one sheet
    unregistered_tiles.png   every UNREGISTERED segment, one tile each,
                             each in its own frame
    unregistered_segment_N.ply
    world.html               interactive viewer (plotly if installed,
                             otherwise a self-contained canvas orbit viewer)
    summary.json             counts, flags, bounds, files

WHAT IS AND IS NOT COMPOSED

`docs/contracts/WORLD-BUILDER-GEOMETRY.md` §5 rule 3 and §7 govern the
picture. A segment is drawn in the world frame ONLY when its placement is

    state == "registered", with a complete Sim3,
    bound to the current build (input_digest matches derived/manifest.json).

Segments sharing a `reference_segment` and `frame_revision` are one space.
Segments with different reference segments are different spaces and are
rendered as separate frames, never overlaid. Everything else -- refused,
unplaced, unbound -- is rendered SEPARATELY, in its own frame, with its
reason in the tile title. Their scales disagree by up to ~87x on a real
walk; overlaying them would fabricate a room.

The composition itself is the store's own: `Sim3.apply` from
`tower/world_builder/render.py`, X_ref = scale * R @ X_seg + t, with the
quaternion decoded by the same wxyz convention the registration pass used
to encode it. Nothing here re-derives a convention.

Poses are `T_world_camera` (tower/world_builder/schema.py): the persisted
translation IS the camera centre in the segment frame and the quaternion is
R_world_camera, so the camera's optical axis in the segment frame is
R @ [0, 0, 1]. `translation: null` means refused, never zero; such a camera
is counted and not drawn.

`up_axis` is "unknown" on this hardware, so the view names describe which
axis the eye looks along, not which way is up. "top" looks along +y because
OpenCV's camera y points down.

    python scripts/world_render.py --world-root data/world_builder \\
        --world <id> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from tower.artifact_paths import artifact_root_arg  # noqa: E402
from tower.world_builder.store import WorldStore  # noqa: E402
# The composition -- reading the derived tree, applying each placement,
# grouping shared spaces -- and the dependency-free HTML viewer live in the
# package since 2026-09-06, so the Tower can serve the viewer to the phone
# (GET /worlds/{id}/render). Re-exported here so this script's callers and
# tests keep their names; this file owns the PLY/PNG/plotly writers and
# the CLI.
from tower.world_builder.render import (  # noqa: E402,F401
    CAMERA_COLOUR,
    DEFAULT_MAX_POINTS,
    FRUSTUM_DEPTH_FRACTION,
    PALETTE,
    ROBUST_PERCENTILES,
    Camera,
    Frame,
    Segment,
    Sim3,
    canvas_html,
    compose_frames,
    derived_current,
    frustum_lines,
    html_frames as _html_frames,
    inside,
    load_segments,
    quaternion_wxyz_to_rotation as _quaternion_wxyz_to_rotation,
    robust_bounds,
    segment_colour,
    subsample,
)

PNG_DPI = 130


# -- geometry helpers -----------------------------------------------------


def _rot_x(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


# Each view is a rotation taking world xyz into a screen frame where the
# eye looks along +z and the page is (u=x right, v=-y up). Named by what
# the eye looks along in WORLD axes, because up is unknown.
VIEWS = {
    "top": ("looking along +y (x right, z up the page)", _rot_x(90.0)),
    "front": ("looking along +z (x right, -y up the page)", np.eye(3)),
    "side": ("looking along -x (z right, -y up the page)", _rot_y(90.0)),
    "iso": ("3/4 view: yaw 40 deg, tilt 30 deg", _rot_x(30.0) @ _rot_y(40.0)),
}


def project(xyz: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """World (N,3) -> screen (N,2): u right, v up."""
    if len(xyz) == 0:
        return np.zeros((0, 2))
    screen = xyz @ rotation.T
    return np.column_stack([screen[:, 0], -screen[:, 1]])


# -- writers ---------------------------------------------------------------


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Binary little-endian PLY with per-vertex colour."""
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    rgb = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"comment Glasses world_render {time.strftime('%Y-%m-%d')}\n"
        f"element vertex {len(xyz)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    rows = np.empty(len(xyz), dtype=dtype)
    rows["x"], rows["y"], rows["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rows["red"], rows["green"], rows["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(rows.tobytes())


def read_ply_xyz_rgb(path: Path):
    """The inverse of write_ply, for tests and for checking a file back."""
    data = path.read_bytes()
    marker = b"end_header\n"
    split = data.index(marker) + len(marker)
    header = data[:split].decode("ascii")
    count = int([l for l in header.splitlines() if l.startswith("element vertex")][0].split()[-1])
    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    rows = np.frombuffer(data[split:], dtype=dtype, count=count)
    xyz = np.column_stack([rows["x"], rows["y"], rows["z"]]).astype(np.float64)
    rgb = np.column_stack([rows["red"], rows["green"], rows["blue"]])
    return xyz, rgb


def _colours_for(owner: np.ndarray, ordering: list) -> np.ndarray:
    rgb = np.zeros((len(owner), 3), dtype=np.uint8)
    for index in np.unique(owner):
        rgb[owner == index] = segment_colour(int(index), ordering)
    return rgb


def write_frame_plys(frame: Frame, out: Path, ordering: list) -> dict:
    xyz, owner = frame.points_world()
    files = {}
    ply = out / f"{frame.tag}.ply"
    write_ply(ply, xyz, _colours_for(owner, ordering))
    files["points"] = ply.name
    cameras = frame.cameras_world()
    if cameras:
        centres = np.array([c[0] for c in cameras])
        rgb = np.array([segment_colour(c[2], ordering) for c in cameras], dtype=np.uint8)
        cam_ply = out / f"cameras_{frame.tag}.ply"
        write_ply(cam_ply, centres, rgb)
        files["cameras"] = cam_ply.name
    return files


def write_unregistered_plys(unregistered: list, out: Path, ordering: list) -> dict:
    files = {}
    for segment in unregistered:
        if len(segment.points) == 0:
            continue
        ply = out / f"unregistered_segment_{segment.index}.ply"
        owner = np.full(len(segment.points), segment.index)
        write_ply(ply, segment.points, _colours_for(owner, ordering))
        files[segment.index] = ply.name
    return files


# -- PNG renders -----------------------------------------------------------


def _matplotlib():
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.collections import LineCollection  # noqa: PLC0415

    return plt, LineCollection


def _draw_view(ax, plt, LineCollection, xyz, owner, cameras, rotation, ordering,
               *, bounds, point_size=1.2, legend=True):
    """One orthographic view onto an axes. Returns how many points were
    outside the robust bounds and therefore not drawn."""
    keep = inside(xyz, bounds) if (bounds is not None and len(xyz)) else np.ones(len(xyz), bool)
    dropped = int((~keep).sum())
    shown = xyz[keep]
    shown_owner = owner[keep]
    uv = project(shown, rotation)
    for index in sorted(np.unique(shown_owner), key=int):
        mask = shown_owner == index
        colour = np.array(segment_colour(int(index), ordering)) / 255.0
        ax.scatter(uv[mask, 0], uv[mask, 1], s=point_size, c=[colour], lw=0,
                   alpha=0.85, label=f"seg {int(index)} ({int(mask.sum())} pts)",
                   rasterized=True)
    if cameras:
        extent = float(np.max(bounds[1] - bounds[0])) if bounds is not None else 1.0
        depth = FRUSTUM_DEPTH_FRACTION * extent
        lines, centres, cam_colours = [], [], []
        skipped = 0
        for centre, rot, index, _status in cameras:
            if bounds is not None and not inside(centre[None, :], bounds)[0]:
                skipped += 1
                continue
            for a, b in frustum_lines(centre, rot, depth):
                lines.append(np.vstack([project(a[None, :], rotation),
                                        project(b[None, :], rotation)]))
            centres.append(centre)
            cam_colours.append(np.array(segment_colour(index, ordering)) / 255.0)
        if lines:
            ax.add_collection(LineCollection(lines, colors="black", linewidths=0.4,
                                             alpha=0.6))
            cuv = project(np.array(centres), rotation)
            ax.scatter(cuv[:, 0], cuv[:, 1], s=14, marker="^", c=cam_colours,
                       edgecolors="black", linewidths=0.4, zorder=5,
                       label=(f"cameras ({len(centres)})" if not skipped else
                              f"cameras ({len(centres)} of {len(cameras)} "
                              "inside bounds)"))
    if len(uv):
        lo, hi = uv.min(axis=0), uv.max(axis=0)
        pad = 0.04 * np.maximum(hi - lo, 1e-9)
        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
        ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, lw=0.3, alpha=0.4)
    ax.tick_params(labelsize=7)
    if legend:
        ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                  borderaxespad=0.0, markerscale=4, frameon=False)
    return dropped


def render_frame_pngs(frame: Frame, out: Path, ordering: list, *, dpi=PNG_DPI,
                      title_extra="") -> dict:
    plt, LineCollection = _matplotlib()
    xyz, owner = frame.points_world()
    cameras = frame.cameras_world()
    bounds = robust_bounds(xyz)
    # The cameras stand in front of what they see, so a box over the points
    # alone can leave half the trajectory outside the picture (measured on
    # the 2026-09-01 loop: 232 of 424 cameras clipped). Widen to the
    # cameras' own robust box, so the walk is drawn with the room.
    if bounds is not None and cameras:
        centres = np.array([c[0] for c in cameras], dtype=np.float64)
        cam_bounds = robust_bounds(centres, percentiles=(2, 98), pad=0.02) if len(centres) >= 4 else (centres.min(axis=0), centres.max(axis=0))
        if cam_bounds is not None:
            bounds = (np.minimum(bounds[0], cam_bounds[0]), np.maximum(bounds[1], cam_bounds[1]))
    files = {}
    dropped_by_view = {}
    header = (
        f"{frame.tag}: reference segment {frame.reference_segment}, "
        f"frame revision {frame.frame_revision}, "
        f"{len(frame.segments)} registered segments, {len(xyz)} points, "
        f"{len(cameras)} cameras{title_extra}"
    )
    for name, (caption, rotation) in VIEWS.items():
        fig, ax = plt.subplots(figsize=(9, 7))
        dropped = _draw_view(ax, plt, LineCollection, xyz, owner, cameras, rotation,
                             ordering, bounds=bounds)
        dropped_by_view[name] = dropped
        ax.set_title(f"{header}\n{name}: {caption}; {dropped} points outside "
                     f"p{ROBUST_PERCENTILES[0]:.0f}-p{ROBUST_PERCENTILES[1]:.0f} "
                     "bounds not drawn", fontsize=8)
        fig.tight_layout()
        path = out / f"view_{name}{'' if frame.tag == 'world' else '_' + frame.tag}.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        files[name] = path.name

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    for ax, (name, (caption, rotation)) in zip(axes.ravel(), VIEWS.items()):
        _draw_view(ax, plt, LineCollection, xyz, owner, cameras, rotation, ordering,
                   bounds=bounds, legend=(name == "top"))
        ax.set_title(f"{name}: {caption}", fontsize=9)
    fig.suptitle(header, fontsize=10)
    fig.tight_layout()
    path = out / f"overview{'' if frame.tag == 'world' else '_' + frame.tag}.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    files["overview"] = path.name
    return {"files": files, "points_outside_bounds": dropped_by_view,
            "robust_bounds": None if bounds is None else
            {"min": bounds[0].tolist(), "max": bounds[1].tolist()}}


def render_unregistered_tiles(unregistered: list, out: Path, ordering: list,
                              *, dpi=PNG_DPI) -> dict | None:
    """Every unregistered segment on its own tile in its OWN frame.

    Tiles share nothing: not an origin, not a scale, not an axis range.
    A reader who wants to compare two tiles has to read two scale bars,
    which is the point.
    """
    drawable = [s for s in unregistered if len(s.points)]
    lone = sorted(s.index for s in unregistered if not len(s.points))
    if not drawable:
        return None
    plt, LineCollection = _matplotlib()
    cols = min(4, len(drawable))
    rows = math.ceil(len(drawable) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.0 * rows), squeeze=False)
    rotation = VIEWS["top"][1]
    for ax, segment in zip(axes.ravel(), drawable):
        owner = np.full(len(segment.points), segment.index)
        cameras = [(c.centre, c.rotation_world_camera, segment.index, c.status)
                   for c in segment.cameras]
        bounds = robust_bounds(segment.points)
        dropped = _draw_view(ax, plt, LineCollection, segment.points, owner, cameras,
                             rotation, ordering, bounds=bounds, legend=False)
        reason = (segment.refusal_reason or "no placement row")
        if len(reason) > 70:
            reason = reason[:67] + "..."
        ax.set_title(
            f"segment {segment.index} [{segment.placement_state}]\n"
            f"{len(segment.points)} pts ({dropped} outside p2-p98 not drawn), "
            f"{len(segment.cameras)} cams; own frame, own scale\n{reason}",
            fontsize=7,
        )
    for ax in axes.ravel()[len(drawable):]:
        ax.axis("off")
    fig.suptitle(
        "UNREGISTERED segments: each tile is its own coordinate frame and its "
        "own scale. They are NOT in the world frame and must not be read as "
        "one room. View: top (looking along +y)."
        + (f"\nNot tiled, no geometry (a lone anchor each): segments {lone}"
           if lone else ""),
        fontsize=9, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    path = out / "unregistered_tiles.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return {"file": path.name, "segments": [s.index for s in drawable],
            "no_geometry": lone}


def render_empty_world_png(out: Path, unregistered: list, *, dpi=PNG_DPI) -> str:
    plt, _ = _matplotlib()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axis("off")
    ax.text(0.5, 0.5,
            "No registered segments: there is no world frame to draw.\n"
            f"{len([s for s in unregistered if len(s.points)])} segments with "
            "geometry are rendered separately in unregistered_tiles.png,\n"
            "each in its own frame. Overlaying them would fabricate a room.",
            ha="center", va="center", fontsize=11)
    fig.tight_layout()
    path = out / "overview.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path.name


# -- HTML ------------------------------------------------------------------


def write_html_plotly(path: Path, frames: list, unregistered: list, ordering: list,
                      title: str) -> str:
    import plotly.graph_objects as go  # noqa: PLC0415

    descriptors = _html_frames(frames, unregistered, ordering)
    fig = go.Figure()
    groups = []           # trace indices per frame
    for descriptor in descriptors:
        indices = []
        xyz, owner = descriptor["xyz"], descriptor["owner"]
        for index in sorted(np.unique(owner), key=int):
            mask = owner == index
            r, g, b = segment_colour(int(index), ordering)
            fig.add_trace(go.Scatter3d(
                x=xyz[mask, 0], y=xyz[mask, 1], z=xyz[mask, 2], mode="markers",
                marker=dict(size=1.6, color=f"rgb({r},{g},{b})"),
                name=f"segment {int(index)} ({int(mask.sum())} pts)",
                visible=False,
            ))
            indices.append(len(fig.data) - 1)
        cameras = descriptor["cameras"]
        if cameras:
            centres = np.array([c[0] for c in cameras])
            extent = float(np.max(np.ptp(xyz, axis=0))) if len(xyz) else 1.0
            depth = FRUSTUM_DEPTH_FRACTION * extent
            lx, ly, lz = [], [], []
            for centre, rot, _index, _status in cameras:
                for a, b in frustum_lines(centre, rot, depth):
                    lx += [a[0], b[0], None]
                    ly += [a[1], b[1], None]
                    lz += [a[2], b[2], None]
            fig.add_trace(go.Scatter3d(
                x=centres[:, 0], y=centres[:, 1], z=centres[:, 2], mode="markers",
                marker=dict(size=3.5, color="black", symbol="diamond"),
                text=[f"seg {c[2]} {c[3]}" for c in cameras],
                name=f"cameras ({len(cameras)})", visible=False,
            ))
            indices.append(len(fig.data) - 1)
            fig.add_trace(go.Scatter3d(
                x=lx, y=ly, z=lz, mode="lines",
                line=dict(color="black", width=1), name="frusta",
                visible=False, hoverinfo="skip",
            ))
            indices.append(len(fig.data) - 1)
        groups.append(indices)

    total = len(fig.data)
    buttons = []
    for descriptor, indices in zip(descriptors, groups):
        visible = [i in indices for i in range(total)]
        buttons.append(dict(
            label=descriptor["name"], method="update",
            args=[{"visible": visible},
                  {"title": f"{title} -- {descriptor['name']}"}],
        ))
    if groups:
        for i in groups[0]:
            fig.data[i].visible = True
    fig.update_layout(
        title=f"{title} -- {descriptors[0]['name'] if descriptors else 'nothing to draw'}",
        scene=dict(aspectmode="data",
                   xaxis_title="x", yaxis_title="y (camera down)", zaxis_title="z"),
        updatemenus=[dict(buttons=buttons, direction="down", x=0.0, y=1.12,
                          showactive=True)] if buttons else [],
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=80, b=0),
    )
    fig.write_html(str(path), include_plotlyjs=True, full_html=True)
    return "plotly"


def write_html_canvas(path: Path, frames: list, unregistered: list, ordering: list,
                      title: str, current=None) -> str:
    """Self-contained viewer with no external library: an orthographic
    orbit camera on a 2-D canvas. Chosen so the tool works with nothing
    but a browser. The template is `tower.world_builder.render`'s, so the
    operator's page and the phone's are the same page, BEHIND caption
    included."""
    path.write_text(canvas_html(frames, unregistered, ordering, title, current=current),
                    encoding="utf-8")
    return "canvas"


def write_html(path: Path, frames: list, unregistered: list, ordering: list,
               title: str, backend: str = "auto", current=None) -> str:
    if backend in ("auto", "plotly"):
        try:
            import plotly  # noqa: F401,PLC0415
        except ImportError:
            if backend == "plotly":
                raise
        else:
            return write_html_plotly(path, frames, unregistered, ordering, title)
    return write_html_canvas(path, frames, unregistered, ordering, title, current=current)


# -- orchestration --------------------------------------------------------


def _bounds_dict(xyz: np.ndarray):
    if len(xyz) == 0:
        return None
    return {"min": xyz.min(axis=0).tolist(), "max": xyz.max(axis=0).tolist()}


def render_world(store: WorldStore, world_id: str, session_id: str, out: Path, *,
                 max_points: int = DEFAULT_MAX_POINTS, html_backend: str = "auto",
                 dpi: int = PNG_DPI, html: bool = True) -> dict:
    """Render one session of one world into `out`. Returns the summary."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    segments = load_segments(store, world_id, session_id)
    ordering = sorted(segments)
    current = derived_current(store, world_id, session_id)
    sampling = subsample(segments, max_points)
    frames, unregistered = compose_frames(segments)

    files: dict = {}
    frame_reports = []
    for frame in frames:
        report = {
            "tag": frame.tag,
            "reference_segment": frame.reference_segment,
            "frame_revision": frame.frame_revision,
            "segments": [s.index for s in frame.segments],
            "points": frame.point_count,
            "cameras": len(frame.cameras_world()),
        }
        xyz, _owner = frame.points_world()
        report["bounds"] = _bounds_dict(xyz)
        report["files"] = write_frame_plys(frame, out, ordering)
        png = render_frame_pngs(frame, out, ordering, dpi=dpi,
                                title_extra="" if current else
                                ("; derived tree BEHIND the journal"
                                 if current is False else ""))
        report["files"].update(png["files"])
        report["robust_bounds"] = png["robust_bounds"]
        report["points_outside_robust_bounds"] = png["points_outside_bounds"]
        frame_reports.append(report)
    if not frames:
        files["overview"] = render_empty_world_png(out, unregistered, dpi=dpi)

    unregistered_plys = write_unregistered_plys(unregistered, out, ordering)
    tiles = render_unregistered_tiles(unregistered, out, ordering, dpi=dpi)
    if tiles:
        files["unregistered_tiles"] = tiles["file"]

    if html:
        title = f"world {world_id[:8]} session {session_id[:8]}"
        html_path = out / "world.html"
        files["html"] = html_path.name
        files["html_backend"] = write_html(html_path, frames, unregistered, ordering,
                                           title, backend=html_backend, current=current)

    summary = {
        "world_id": world_id,
        "session_id": session_id,
        "rendered_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "derived_current": current,
        "up_axis": "unknown (views are named by the axis the eye looks along)",
        "composition": "X_ref = scale * R(rotation_wxyz) @ X_segment + translation",
        "points_total": sampling["points_total"],
        "points_rendered": sampling["points_kept"],
        "subsampled": sampling["subsampled"],
        "segments_total": len(segments),
        "segments_registered": sum(1 for s in segments.values() if s.registered),
        "segments_unregistered": len(unregistered),
        "frames": frame_reports,
        "unregistered": [
            {
                "segment_index": s.index,
                "state": s.placement_state,
                "points": len(s.points),
                "cameras": len(s.cameras),
                "cameras_refused": s.cameras_refused,
                "reason": s.refusal_reason,
                "bounds_own_frame": _bounds_dict(s.points),
                "ply": unregistered_plys.get(s.index),
            }
            for s in unregistered
        ],
        "segments": {
            str(s.index): {
                "points": len(s.points),
                "cameras": len(s.cameras),
                "cameras_refused": s.cameras_refused,
                "registered": s.registered,
                "state": s.placement_state,
                "reference_segment": s.reference_segment,
                "frame_revision": s.frame_revision,
                "scale": s.scale,
                "reason": s.refusal_reason,
                "bounds_own_frame": _bounds_dict(s.points),
                "colour_rgb": list(segment_colour(s.index, ordering)),
            }
            for s in segments.values()
        },
        "files": files,
        "seconds": round(time.perf_counter() - started, 2),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _resolve_world(args, parser):
    if args.world_dir is not None:
        world_dir = Path(args.world_dir).resolve()
        if not (world_dir / "derived").exists():
            parser.error(f"{world_dir} has no derived/ directory")
        # worlds/<id> sits under the root; the store knows the layout.
        return WorldStore(world_dir.parent.parent), world_dir.name
    if args.world_root is None:
        parser.error("one of --world-dir or --world-root is required")
    store = WorldStore(Path(args.world_root))
    world_id = args.world
    if world_id is None:
        worlds = store.list_world_ids()
        if len(worlds) != 1:
            parser.error(f"--world is required; {len(worlds)} worlds under --world-root")
        world_id = worlds[0]
    return store, world_id


def _resolve_session(store, world_id, requested, parser):
    if requested is not None:
        return requested
    derived = store.derived_dir(world_id)
    candidates = sorted(
        p.name for p in derived.iterdir()
        if p.is_dir() and (p / "points.json").exists()
    ) if derived.exists() else []
    if len(candidates) != 1:
        parser.error(f"--session is required; {len(candidates)} derived sessions "
                     f"in {world_id}: {candidates}")
    return candidates[0]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Render a saved World Builder world so a human can look at it."
    )
    parser.add_argument("--world-dir", default=None,
                        help="Path to worlds/<id>. Alternative to --world-root/--world.")
    parser.add_argument("--world-root", type=artifact_root_arg, default=None,
                        help="World Builder data root (the directory holding worlds/).")
    parser.add_argument("--world", default=None, help="World id. Default: the only one.")
    parser.add_argument("--session", default=None,
                        help="Session id. Default: the only derived session.")
    parser.add_argument("--out", type=artifact_root_arg, required=True,
                        help="Output directory. Created if missing.")
    parser.add_argument("--max-points", type=int, default=DEFAULT_MAX_POINTS,
                        help="Stride-subsample above this many points.")
    parser.add_argument("--html-backend", choices=("auto", "plotly", "canvas"),
                        default="auto")
    parser.add_argument("--no-html", action="store_true")
    parser.add_argument("--dpi", type=int, default=PNG_DPI)
    args = parser.parse_args(argv)

    store, world_id = _resolve_world(args, parser)
    session_id = _resolve_session(store, world_id, args.session, parser)
    summary = render_world(
        store, world_id, session_id, Path(args.out), max_points=args.max_points,
        html_backend=args.html_backend, dpi=args.dpi, html=not args.no_html,
    )
    print(f"world {world_id} session {session_id}")
    print(f"  derived current      {summary['derived_current']}")
    print(f"  points               {summary['points_rendered']} of "
          f"{summary['points_total']}"
          f"{' (subsampled)' if summary['subsampled'] else ''}")
    print(f"  segments registered  {summary['segments_registered']} of "
          f"{summary['segments_total']}")
    for frame in summary["frames"]:
        print(f"  frame {frame['tag']:<24} ref {frame['reference_segment']} "
              f"segments {frame['segments']} points {frame['points']} "
              f"cameras {frame['cameras']}")
    print(f"  unregistered         {summary['segments_unregistered']} segments, "
          "rendered separately")
    print(f"  out                  {Path(args.out).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
