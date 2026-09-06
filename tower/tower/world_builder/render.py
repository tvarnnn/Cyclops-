"""Compose a built world into drawable frames, and draw it as HTML.

Lifted out of `scripts/world_render.py` on 2026-09-06 so the Tower's web
process can serve the interactive viewer to the phone (`GET
/worlds/{id}/render`, `docs/contracts/WORLD-BUILDER-WORLDS.md` §4). The
script still owns the PLY and PNG writers, the plotly backend and the CLI;
this module owns what both need: reading the derived tree, applying each
registered placement exactly as the store defines it, grouping segments
that share a reference into one space, and the dependency-free canvas
viewer. Nothing here imports matplotlib or plotly.

WHAT IS AND IS NOT COMPOSED

`docs/contracts/WORLD-BUILDER-GEOMETRY.md` §5 rule 3 and §7 govern the
picture. A segment is drawn in the world frame ONLY when its placement is

    state == "registered", with a complete Sim3,
    bound to the current build (input_digest matches derived/manifest.json).

Segments sharing a `reference_segment` and `frame_revision` are one space.
Segments with different reference segments are different spaces and are
rendered as separate frames, never overlaid. Everything else -- refused,
unplaced, unbound -- is rendered SEPARATELY, in its own frame, with its
reason in the frame's name. Their scales disagree by up to ~87x on a real
walk; overlaying them would fabricate a room.

The composition is the store's own: X_ref = scale * R @ X_seg + t, with the
quaternion decoded by the same wxyz convention the registration pass used
to encode it. Nothing here re-derives a convention.

Poses are `T_world_camera` (tower/world_builder/schema.py): the persisted
translation IS the camera centre in the segment frame and the quaternion is
R_world_camera, so the camera's optical axis in the segment frame is
R @ [0, 0, 1]. `translation: null` means refused, never zero; such a camera
is counted and not drawn.

WHAT THE PICTURE IS

Sparse structure-from-motion output: triangulated feature points and camera
poses. Not a surface, not a mesh, not metric. The viewer says so in its own
caption, because a point cloud drawn without that sentence reads as a
"scan" of the room, and it is not one.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np

from tower.world_builder.store import WorldStore, compute_input_digest

DEFAULT_MAX_POINTS = 200_000
ROBUST_PERCENTILES = (2.0, 98.0)
# Fraction of the scene's robust extent used for the drawn frustum depth.
FRUSTUM_DEPTH_FRACTION = 0.025

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


# -- the transform ---------------------------------------------------------


def quaternion_wxyz_to_rotation(quaternion) -> np.ndarray:
    """wxyz unit quaternion -> 3x3 rotation. A degenerate quaternion is the
    identity rather than NaNs: a picture with one segment unrotated is a
    wrong picture, a picture full of NaN is no picture."""
    w, x, y, z = (float(v) for v in quaternion)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-12:
        return np.eye(3)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


@dataclass(frozen=True)
class Sim3:
    """X_world = scale * rotation @ X_segment + translation. The store's
    placement convention (`records.SegmentPlacement`), applied and nothing
    more; the registrar that fits these lives in `scripts/world_registration.py`."""

    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def apply(self, xyz: np.ndarray) -> np.ndarray:
        xyz = np.asarray(xyz, dtype=np.float64)
        if xyz.ndim == 1:
            return self.scale * (self.rotation @ xyz) + self.translation
        return self.scale * (self.rotation @ xyz.T).T + self.translation


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
            rotation_world_camera=quaternion_wxyz_to_rotation(row["rotation"]),
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
                        rotation=quaternion_wxyz_to_rotation(placement.rotation_wxyz),
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


# -- the HTML viewer -------------------------------------------------------


def html_frames(frames: list, unregistered: list, ordering: list) -> list:
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


# What the picture is, in the viewer's own words. One sentence, always
# shown: the caption is the difference between "a sparse SfM point cloud"
# and "a scan of the room", and only the first is true.
CAPTION = ("Sparse structure-from-motion output: triangulated feature points "
           "and camera poses. Not a surface, not a mesh, not metric scale.")
CAPTION_BEHIND = ("This picture is BEHIND the newest keyframes: the Tower has "
                  "accepted keyframes it has not yet built into geometry.")

# The viewer. Self-contained: no external script, no stylesheet, no fetch,
# so it works with nothing but a browser, and inside a WKWebView with
# outbound navigation refused. Mouse: drag orbits, shift-drag pans, wheel
# zooms. Touch (added 2026-09-06 for the phone): one finger orbits, two
# fingers pinch to zoom and drag to pan. `touch-action: none` keeps the
# page from scrolling or zooming underneath the canvas.
_CANVAS_VIEWER = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>__TITLE__</title>
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<style>
body{margin:0;font:13px system-ui,sans-serif;background:#111;color:#ddd;overflow:hidden}
#bar{padding:8px;background:#222;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
#bar select{max-width:100%;font:inherit}
#caption{padding:4px 8px;background:#1a1a1a;color:#bbb;font-size:12px}
#caption b{color:#f80}
#legend span{display:inline-block;margin-right:10px}
#legend i{display:inline-block;width:10px;height:10px;margin-right:4px;vertical-align:middle}
canvas{display:block;cursor:grab;touch-action:none}
</style></head><body>
<div id="bar"><b>__TITLE__</b>
<select id="frame"></select>
<span>drag: orbit &middot; pinch/wheel: zoom &middot; two fingers/shift-drag: pan &middot; keys 1/2/3: top/front/side</span>
<span id="legend"></span></div>
<div id="caption">__CAPTION__</div>
<canvas id="c"></canvas>
<script>
const FRAMES = __FRAMES__;
const sel = document.getElementById('frame'), canvas = document.getElementById('c'),
      ctx = canvas.getContext('2d'), legend = document.getElementById('legend');
FRAMES.forEach((f, i) => { const o = document.createElement('option'); o.value = i; o.textContent = f.name; sel.appendChild(o); });
let cur = 0, yaw = 0.7, pitch = 0.5, zoom = 1, panX = 0, panY = 0, drag = null, pinch = null;
function chrome(){ return document.getElementById('bar').offsetHeight + document.getElementById('caption').offsetHeight; }
function resize(){ canvas.width = innerWidth; canvas.height = Math.max(50, innerHeight - chrome()); draw(); }
function rot(){ const cy=Math.cos(yaw), sy=Math.sin(yaw), cp=Math.cos(pitch), sp=Math.sin(pitch);
  return [[cy,0,sy],[sy*sp,cp,-cy*sp],[-sy*cp,sp,cy*cp]]; }
function draw(){
  const f = FRAMES[cur], W = canvas.width, H = canvas.height;
  ctx.fillStyle = '#111'; ctx.fillRect(0,0,W,H);
  if (!f){ ctx.fillStyle = '#ddd'; ctx.fillText('Nothing to draw: no segment has points or posed cameras.', 12, 24); legend.innerHTML = ''; return; }
  const R = rot(), c = f.centre, s = zoom * 0.9 * Math.min(W,H) / f.extent;
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
function orbit(dx, dy){ yaw += dx*0.01; pitch = Math.max(-1.55, Math.min(1.55, pitch + dy*0.01)); }
canvas.addEventListener('mousedown', e => { drag = {x:e.clientX, y:e.clientY, shift:e.shiftKey}; canvas.style.cursor='grabbing'; });
addEventListener('mouseup', () => { drag = null; canvas.style.cursor='grab'; });
addEventListener('mousemove', e => { if(!drag) return; const dx=e.clientX-drag.x, dy=e.clientY-drag.y; drag.x=e.clientX; drag.y=e.clientY;
  if (drag.shift){ panX += dx; panY += dy; } else { orbit(dx, dy); } draw(); });
canvas.addEventListener('wheel', e => { e.preventDefault(); zoom *= Math.exp(-e.deltaY*0.001); draw(); }, {passive:false});
function mid(t){ return [(t[0].clientX+t[1].clientX)/2, (t[0].clientY+t[1].clientY)/2]; }
function dist(t){ return Math.hypot(t[0].clientX-t[1].clientX, t[0].clientY-t[1].clientY); }
canvas.addEventListener('touchstart', e => { e.preventDefault(); const t = e.touches;
  if (t.length === 1){ drag = {x:t[0].clientX, y:t[0].clientY, shift:false}; pinch = null; }
  else if (t.length >= 2){ drag = null; pinch = {d:dist(t), m:mid(t)}; } }, {passive:false});
canvas.addEventListener('touchmove', e => { e.preventDefault(); const t = e.touches;
  if (t.length === 1 && drag){ const dx=t[0].clientX-drag.x, dy=t[0].clientY-drag.y; drag.x=t[0].clientX; drag.y=t[0].clientY; orbit(dx, dy); draw(); }
  else if (t.length >= 2 && pinch){ const d = dist(t), m = mid(t); if (pinch.d > 0) zoom *= d / pinch.d;
    panX += m[0]-pinch.m[0]; panY += m[1]-pinch.m[1]; pinch = {d:d, m:m}; draw(); } }, {passive:false});
canvas.addEventListener('touchend', e => { e.preventDefault(); const t = e.touches;
  if (t.length === 0){ drag = null; pinch = null; }
  else if (t.length === 1){ pinch = null; drag = {x:t[0].clientX, y:t[0].clientY, shift:false}; } }, {passive:false});
canvas.addEventListener('touchcancel', () => { drag = null; pinch = null; });
addEventListener('keydown', e => { if(e.key==='1'){yaw=0;pitch=Math.PI/2;} if(e.key==='2'){yaw=0;pitch=0;} if(e.key==='3'){yaw=Math.PI/2;pitch=0;} draw(); });
sel.addEventListener('change', () => { cur = +sel.value; panX = panY = 0; zoom = 1; draw(); });
addEventListener('resize', resize); resize();
</script></body></html>
"""


def _escape(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def canvas_html(frames: list, unregistered: list, ordering: list, title: str, *,
                current=None) -> str:
    """The self-contained viewer as a string. `current` is
    `derived_current`'s answer; False adds the behind-the-journal line to
    the caption, True and None (unknowable) do not."""
    descriptors = html_frames(frames, unregistered, ordering)
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
    caption = _escape(CAPTION)
    if current is False:
        caption += f" <b>{_escape(CAPTION_BEHIND)}</b>"
    # A `<` inside the JSON is the one character that could end the
    # script block early (`</script>`) or, as `<!--`, put the parser into
    # its escaped state. Every string in the payload is server-composed
    # today, but the payload is written once and read by a browser, so it
    # is escaped as `\u003c`, which JSON.parse reads back as `<`.
    frames_json = json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")
    return (_CANVAS_VIEWER
            .replace("__TITLE__", _escape(title))
            .replace("__CAPTION__", caption)
            .replace("__FRAMES__", frames_json))


def render_html(store: WorldStore, world_id: str, session_id: str, *,
                max_points: int = DEFAULT_MAX_POINTS, title: str | None = None) -> str:
    """One session of one world as the interactive viewer. Raises
    `FileNotFoundError` when the session has no derived tree."""
    segments = load_segments(store, world_id, session_id)
    ordering = sorted(segments)
    current = derived_current(store, world_id, session_id)
    subsample(segments, max_points)
    frames, unregistered = compose_frames(segments)
    if title is None:
        title = f"world {world_id[:8]} session {session_id[:8]}"
    return canvas_html(frames, unregistered, ordering, title, current=current)
