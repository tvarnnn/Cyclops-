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
`scripts/world_registration.py`, X_ref = scale * R @ X_seg + t, with the
quaternion decoded by the same `_quaternion_wxyz_to_rotation` the
registration pass used to encode it. Nothing here re-derives a convention.

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
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from tower.artifact_paths import artifact_root_arg  # noqa: E402
from tower.world_builder.store import WorldStore, compute_input_digest  # noqa: E402
from scripts.world_registration import (  # noqa: E402
    Sim3,
    _quaternion_wxyz_to_rotation,
)

DEFAULT_MAX_POINTS = 200_000
ROBUST_PERCENTILES = (2.0, 98.0)
# Fraction of the scene's robust extent used for the drawn frustum depth.
FRUSTUM_DEPTH_FRACTION = 0.025
PNG_DPI = 130

# 20 distinguishable colours (matplotlib's tab20, spelled out so the PLY and
# the HTML agree with the PNGs without depending on matplotlib being here).
PALETTE = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
    (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
    (188, 189, 34), (23, 190, 207), (174, 199, 232), (255, 187, 120),
    (152, 223, 138), (255, 152, 150), (197, 176, 213), (196, 156, 148),
    (247, 182, 210), (199, 199, 199), (219, 219, 141), (158, 218, 229),
]
CAMERA_COLOUR = (20, 20, 20)


# -- data ------------------------------------------------------------------


@dataclass
class Camera:
    keyframe_id: str
    status: str
    centre: np.ndarray            # (3,) in the segment's own frame
    rotation_world_camera: np.ndarray  # (3, 3)


@dataclass
class Segment:
    index: int
    points: np.ndarray            # (N, 3) in the segment's own frame
    cameras: list                 # Camera rows that have a centre
    cameras_refused: int          # rows with no translation
    placement_state: str          # registered | refused | unplaced | unbound
    reference_segment: int | None
    frame_revision: int | None
    scale: float | None
    refusal_reason: str | None
    sim3: Sim3 | None

    @property
    def registered(self) -> bool:
        return self.sim3 is not None


@dataclass
class Frame:
    """One shared coordinate space: a reference segment at one revision."""

    tag: str
    reference_segment: int
    frame_revision: int
    segments: list = field(default_factory=list)   # Segment rows

    def points_world(self):
        """(xyz (N,3), segment index per point (N,)) in this frame."""
        xyz, owner = [], []
        for segment in self.segments:
            if len(segment.points):
                xyz.append(segment.sim3.apply(segment.points))
                owner.append(np.full(len(segment.points), segment.index))
        if not xyz:
            return np.zeros((0, 3)), np.zeros(0, dtype=int)
        return np.vstack(xyz), np.concatenate(owner)

    def cameras_world(self):
        """(centre (3,), rotation (3,3), segment index, status) per camera."""
        rows = []
        for segment in self.segments:
            s = segment.sim3
            for camera in segment.cameras:
                rows.append((
                    s.apply(camera.centre),
                    s.rotation @ camera.rotation_world_camera,
                    segment.index,
                    camera.status,
                ))
        return rows

    @property
    def point_count(self) -> int:
        return sum(len(s.points) for s in self.segments)


# -- loading ---------------------------------------------------------------


def segment_colour(index: int, ordering: list) -> tuple:
    """Colour by RANK among the session's segments, so a world whose
    segments are 11..31 still uses the whole palette."""
    rank = ordering.index(index) if index in ordering else index
    return PALETTE[rank % len(PALETTE)]


def load_segments(store: WorldStore, world_id: str, session_id: str) -> dict:
    """Everything the renderer needs, keyed by segment index.

    The derived tree is read with verify=False on purpose: a tree behind
    the journal is a correct answer to an older question (contract §4) and
    is served flagged, not hidden. The flag is computed separately by
    `derived_current`.
    """
    derived = store.read_derived(world_id, session_id, verify=False)
    if derived is None:
        raise FileNotFoundError(
            f"world {world_id} session {session_id} has no derived "
            "points.json/poses.json to render"
        )
    manifest = store.read_derived_manifest(world_id) or {}
    build_digest = manifest.get("input_digest")

    points_by_segment: dict = {}
    for row in derived["points"]:
        points_by_segment.setdefault(int(row["segment_index"]), []).append(row["xyz"])

    cameras_by_segment: dict = {}
    refused_by_segment: dict = {}
    for row in derived["poses"]:
        index = int(row["segment_index"])
        if row.get("translation") is None or row.get("rotation") is None:
            refused_by_segment[index] = refused_by_segment.get(index, 0) + 1
            continue
        cameras_by_segment.setdefault(index, []).append(Camera(
            keyframe_id=row.get("keyframe_id", ""),
            status=row.get("status", ""),
            centre=np.asarray(row["translation"], dtype=np.float64),
            rotation_world_camera=_quaternion_wxyz_to_rotation(row["rotation"]),
        ))

    placements = {
        p.segment_index: p
        for p in (store.read_placements(world_id, session_id) or [])
    }

    indices = sorted(set(points_by_segment) | set(cameras_by_segment)
                     | set(refused_by_segment) | set(placements))
    segments = {}
    for index in indices:
        placement = placements.get(index)
        state, reference, revision, scale, reason, sim3 = (
            "unplaced", None, None, None, None, None
        )
        if placement is not None:
            state = placement.state
            reason = placement.refusal_reason
            if placement.state == "registered":
                reference = placement.reference_segment
                revision = placement.frame_revision
                scale = float(placement.scale)
                if build_digest is None or placement.input_digest != build_digest:
                    # A Sim3 fitted against points that are gone is the
                    # failure the digest exists to catch. Not drawn.
                    state = "unbound"
                    reason = (
                        "placement input_digest does not match the build "
                        "manifest, so the transform was solved against "
                        "different points; not composed"
                    )
                else:
                    sim3 = Sim3(
                        scale=scale,
                        rotation=_quaternion_wxyz_to_rotation(placement.rotation_wxyz),
                        translation=np.asarray(placement.translation, dtype=np.float64),
                    )
        points = np.asarray(points_by_segment.get(index, []), dtype=np.float64)
        if points.size == 0:
            points = np.zeros((0, 3))
        segments[index] = Segment(
            index=index,
            points=points,
            cameras=cameras_by_segment.get(index, []),
            cameras_refused=refused_by_segment.get(index, 0),
            placement_state=state,
            reference_segment=reference,
            frame_revision=revision,
            scale=scale,
            refusal_reason=reason,
            sim3=sim3,
        )
    return segments


def derived_current(store: WorldStore, world_id: str, session_id: str):
    """True/False when it can be known, None when the journal is absent."""
    if not store.keyframes_path(world_id, session_id).exists():
        return None
    try:
        digest = compute_input_digest(store.read_keyframes(world_id, session_id))
    except Exception:  # noqa: BLE001 - an unreadable journal is "unknown", not "stale"
        return None
    return bool(store.derived_is_current(world_id, digest))


def compose_frames(segments: dict):
    """Group registered segments into shared frames; the rest stay apart.

    Returns (frames sorted largest first, unregistered segments). The
    largest frame is tagged "world"; any other reference segment gets its
    own tag so two spaces never share a filename, let alone a picture.
    """
    by_space: dict = {}
    unregistered = []
    for segment in segments.values():
        if segment.registered:
            key = (segment.reference_segment, segment.frame_revision)
            by_space.setdefault(key, []).append(segment)
        else:
            unregistered.append(segment)
    frames = [
        Frame(tag="", reference_segment=ref, frame_revision=rev, segments=members)
        for (ref, rev), members in by_space.items()
    ]
    frames.sort(key=lambda f: (-f.point_count, f.reference_segment))
    for position, frame in enumerate(frames):
        frame.tag = (
            "world" if position == 0
            else f"world_ref{frame.reference_segment}_rev{frame.frame_revision}"
        )
    return frames, unregistered


def subsample(segments: dict, max_points: int) -> dict:
    """Fractional-stride sampling that spans every segment's whole cloud.

    A prefix would be one corner of the room (contract §3). The budget is
    shared proportionally so a big segment cannot starve a small one.
    """
    total = sum(len(s.points) for s in segments.values())
    if total <= max_points:
        return {"subsampled": False, "points_total": total, "points_kept": total}
    kept = 0
    for segment in segments.values():
        n = len(segment.points)
        if n == 0:
            continue
        budget = max(1, int(round(max_points * n / total)))
        if budget < n:
            picks = np.floor(np.arange(budget) * (n / budget)).astype(int)
            segment.points = segment.points[np.minimum(picks, n - 1)]
        kept += len(segment.points)
    return {"subsampled": True, "points_total": total, "points_kept": kept}


# -- geometry helpers -----------------------------------------------------


def robust_bounds(xyz: np.ndarray, percentiles=ROBUST_PERCENTILES, pad=0.05):
    """Per-axis p2..p98 box, padded, so a few diverged points do not blank
    the picture. Returns (lo, hi) or None when there is nothing."""
    if len(xyz) == 0:
        return None
    lo = np.percentile(xyz, percentiles[0], axis=0)
    hi = np.percentile(xyz, percentiles[1], axis=0)
    span = np.maximum(hi - lo, 1e-9)
    return lo - pad * span, hi + pad * span


def inside(xyz: np.ndarray, bounds) -> np.ndarray:
    lo, hi = bounds
    return np.all((xyz >= lo) & (xyz <= hi), axis=1)


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


def frustum_lines(centre, rotation, depth, aspect=0.75):
    """Eight 3-D line segments drawing a small pyramid in front of a camera.

    The optical axis is the camera's +z (OpenCV), so the far face sits at
    centre + rotation @ [±w, ±h, depth].
    """
    w, h = depth * 0.6, depth * 0.6 * aspect
    corners = np.array([
        [-w, -h, depth], [w, -h, depth], [w, h, depth], [-w, h, depth],
    ])
    world = (rotation @ corners.T).T + centre
    lines = [(centre, world[i]) for i in range(4)]
    lines += [(world[i], world[(i + 1) % 4]) for i in range(4)]
    return lines


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


def _html_frames(frames: list, unregistered: list, ordering: list) -> list:
    """Frame descriptors shared by both HTML backends: every drawable
    space, world frames first, each unregistered segment as its own."""
    out = []
    for frame in frames:
        xyz, owner = frame.points_world()
        out.append({
            "name": (f"{frame.tag} (ref segment {frame.reference_segment}, "
                     f"{len(frame.segments)} registered segments)"),
            "shared": True,
            "xyz": xyz, "owner": owner,
            "cameras": frame.cameras_world(),
        })
    for segment in unregistered:
        if len(segment.points) == 0 and not segment.cameras:
            continue
        out.append({
            "name": (f"UNREGISTERED segment {segment.index} "
                     f"[{segment.placement_state}] -- own frame, own scale"),
            "shared": False,
            "xyz": segment.points,
            "owner": np.full(len(segment.points), segment.index),
            "cameras": [(c.centre, c.rotation_world_camera, segment.index, c.status)
                        for c in segment.cameras],
        })
    return out


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


_CANVAS_VIEWER = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
body{margin:0;font:13px system-ui,sans-serif;background:#111;color:#ddd}
#bar{padding:8px;background:#222;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
#legend span{display:inline-block;margin-right:10px}
#legend i{display:inline-block;width:10px;height:10px;margin-right:4px;vertical-align:middle}
canvas{display:block;cursor:grab}
</style></head><body>
<div id="bar"><b>__TITLE__</b>
<select id="frame"></select>
<span>drag: orbit &middot; wheel: zoom &middot; shift-drag: pan &middot; keys 1/2/3: top/front/side</span>
<span id="legend"></span></div>
<canvas id="c"></canvas>
<script>
const FRAMES = __FRAMES__;
const sel = document.getElementById('frame'), canvas = document.getElementById('c'),
      ctx = canvas.getContext('2d'), legend = document.getElementById('legend');
FRAMES.forEach((f, i) => { const o = document.createElement('option'); o.value = i; o.textContent = f.name; sel.appendChild(o); });
let cur = 0, yaw = 0.7, pitch = 0.5, zoom = 1, panX = 0, panY = 0, drag = null;
function resize(){ canvas.width = innerWidth; canvas.height = innerHeight - document.getElementById('bar').offsetHeight; draw(); }
function rot(){ const cy=Math.cos(yaw), sy=Math.sin(yaw), cp=Math.cos(pitch), sp=Math.sin(pitch);
  return [[cy,0,sy],[sy*sp,cp,-cy*sp],[-sy*cp,sp,cy*cp]]; }
function draw(){
  const f = FRAMES[cur], R = rot(), W = canvas.width, H = canvas.height;
  ctx.fillStyle = '#111'; ctx.fillRect(0,0,W,H);
  const c = f.centre, s = zoom * 0.9 * Math.min(W,H) / f.extent;
  function P(p){ const x=p[0]-c[0], y=p[1]-c[1], z=p[2]-c[2];
    const u = R[0][0]*x+R[0][1]*y+R[0][2]*z, v = R[1][0]*x+R[1][1]*y+R[1][2]*z;
    return [W/2 + panX + u*s, H/2 + panY - v*s]; }
  for (const seg of f.segments){ ctx.fillStyle = seg.colour;
    const xs = seg.xyz; for (let i=0;i<xs.length;i+=3){ const q=P([xs[i],xs[i+1],xs[i+2]]); ctx.fillRect(q[0],q[1],1.5,1.5); } }
  ctx.strokeStyle = '#fff'; ctx.lineWidth = 0.6;
  for (const cam of f.cameras){ ctx.beginPath(); for (const ln of cam.lines){ const a=P(ln[0]), b=P(ln[1]); ctx.moveTo(a[0],a[1]); ctx.lineTo(b[0],b[1]); } ctx.stroke();
    const q = P(cam.centre); ctx.fillStyle = cam.colour; ctx.beginPath(); ctx.arc(q[0],q[1],3,0,6.283); ctx.fill(); ctx.stroke(); }
  legend.innerHTML = f.segments.map(sg => `<span><i style="background:${sg.colour}"></i>seg ${sg.index} (${sg.xyz.length/3})</span>`).join('') + ` cameras: ${f.cameras.length}` + (f.shared ? '' : ' <b style="color:#f80">own frame, own scale</b>');
}
canvas.addEventListener('mousedown', e => { drag = {x:e.clientX, y:e.clientY, shift:e.shiftKey}; canvas.style.cursor='grabbing'; });
addEventListener('mouseup', () => { drag = null; canvas.style.cursor='grab'; });
addEventListener('mousemove', e => { if(!drag) return; const dx=e.clientX-drag.x, dy=e.clientY-drag.y; drag.x=e.clientX; drag.y=e.clientY;
  if (drag.shift){ panX += dx; panY += dy; } else { yaw += dx*0.01; pitch = Math.max(-1.55, Math.min(1.55, pitch + dy*0.01)); } draw(); });
canvas.addEventListener('wheel', e => { e.preventDefault(); zoom *= Math.exp(-e.deltaY*0.001); draw(); }, {passive:false});
addEventListener('keydown', e => { if(e.key==='1'){yaw=0;pitch=Math.PI/2;} if(e.key==='2'){yaw=0;pitch=0;} if(e.key==='3'){yaw=Math.PI/2;pitch=0;} draw(); });
sel.addEventListener('change', () => { cur = +sel.value; panX = panY = 0; zoom = 1; draw(); });
addEventListener('resize', resize); resize();
</script></body></html>
"""


def write_html_canvas(path: Path, frames: list, unregistered: list, ordering: list,
                      title: str) -> str:
    """Self-contained viewer with no external library: an orthographic
    orbit camera on a 2-D canvas. Chosen so the tool works with nothing
    but a browser."""
    descriptors = _html_frames(frames, unregistered, ordering)
    payload = []
    for descriptor in descriptors:
        xyz, owner = descriptor["xyz"], descriptor["owner"]
        bounds = robust_bounds(xyz)
        if bounds is None and descriptor["cameras"]:
            centres = np.array([c[0] for c in descriptor["cameras"]])
            bounds = (centres.min(axis=0) - 1.0, centres.max(axis=0) + 1.0)
        if bounds is None:
            continue
        centre = (bounds[0] + bounds[1]) / 2.0
        extent = float(np.max(bounds[1] - bounds[0]))
        segments = []
        for index in sorted(np.unique(owner), key=int):
            r, g, b = segment_colour(int(index), ordering)
            segments.append({
                "index": int(index), "colour": f"rgb({r},{g},{b})",
                "xyz": np.round(xyz[owner == index], 4).ravel().tolist(),
            })
        depth = FRUSTUM_DEPTH_FRACTION * extent
        cameras = []
        for cam_centre, rot, index, _status in descriptor["cameras"]:
            r, g, b = segment_colour(index, ordering)
            cameras.append({
                "centre": np.round(cam_centre, 4).tolist(), "colour": f"rgb({r},{g},{b})",
                "lines": [[np.round(a, 4).tolist(), np.round(b, 4).tolist()]
                          for a, b in frustum_lines(cam_centre, rot, depth)],
            })
        payload.append({
            "name": descriptor["name"], "shared": descriptor["shared"],
            "centre": centre.tolist(), "extent": extent,
            "segments": segments, "cameras": cameras,
        })
    html = (_CANVAS_VIEWER
            .replace("__TITLE__", title)
            .replace("__FRAMES__", json.dumps(payload, separators=(",", ":"))))
    path.write_text(html, encoding="utf-8")
    return "canvas"


def write_html(path: Path, frames: list, unregistered: list, ordering: list,
               title: str, backend: str = "auto") -> str:
    if backend in ("auto", "plotly"):
        try:
            import plotly  # noqa: F401,PLC0415
        except ImportError:
            if backend == "plotly":
                raise
        else:
            return write_html_plotly(path, frames, unregistered, ordering, title)
    return write_html_canvas(path, frames, unregistered, ordering, title)


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
                                           title, backend=html_backend)

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
