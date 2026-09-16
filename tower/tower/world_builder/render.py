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

TWO VIEWS, AND WHICH ONE OPENS (2026-09-09)

Until today this file drew exactly one picture, and it was a debugger's:
every point and every camera coloured by SEGMENT INDEX out of tab20, a
frustum on every keyframe, and each unregistered fragment offered in the
same dropdown as the world. That is the right picture for the question
"did registration work"; it is the wrong picture for "what does my room
look like", which is the question a person opening a saved world is
asking. Worse, it discarded evidence: the global solver writes a
photometric `rgb` on every point it triangulates
(`global_solve.py:1047`), and this renderer read only `xyz`. Measured on
the field world 52ed8e0a, 18,954 of its 26,634 points carry colour, and
on its loop-closed re-solve 19,613 of 19,866 do. That colour was on disk
and on the floor for three weeks.

So there are now two modes over ONE payload, switched in the browser with
no reload and no second request:

  * `world` (the default) -- points in the colour the camera measured,
    a neutral grey where no colour was triangulated, the largest shared
    frame open first, cameras off. This is the product.
  * `diagnostics` -- the picture this file drew before: tab20 by segment,
    frustums, and every unregistered fragment reachable in the dropdown
    with its refusal reason. Nothing that existed became unreachable.

`?view=diagnostics` on the page URL selects the opening mode. It is read
in the browser from `location.search`, NOT declared on the route: FastAPI
ignores query parameters it has not declared, so iOS can deep-link the
diagnostic view without the geometry contract or `routes/geometry.py`
changing at all.

Neither mode invents anything. The world frame still contains only
segments the store registered into it, an unregistered fragment is still
never drawn in it, and a point with no `rgb` is drawn neutral rather than
given a colour from somewhere else.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

import numpy as np

from tower.world_builder.store import WorldStore, compute_input_digest

DEFAULT_MAX_POINTS = 200_000
ROBUST_PERCENTILES = (2.0, 98.0)
# Fraction of the scene's robust extent used for the drawn frustum depth.
FRUSTUM_DEPTH_FRACTION = 0.025

# The two views this page can open in. Named because the value crosses a
# route, a Swift enum and a page template, and a bare "diagnostics" string
# in four places is four chances to misspell it once.
VIEW_PRODUCT = "product"
VIEW_DIAGNOSTICS = "diagnostics"
VIEWS = (VIEW_PRODUCT, VIEW_DIAGNOSTICS)

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

# The colour of a point whose `rgb` the solver never wrote. Deliberately a
# desaturated grey and deliberately NOT drawn from PALETTE: the moment an
# uncoloured point takes a segment colour, the product view is a segment
# view again for exactly the worlds where the evidence is missing, and a
# viewer cannot tell "this wall is grey" from "this point has no colour".
# The page states the count instead.
NEUTRAL_POINT_COLOUR = (138, 138, 138)

# Photometric colour is quantised to 5 bits per channel before it reaches
# the page. Two reasons, both measured on world 52ed8e0a's re-solve:
#
#   * SIZE. 19,613 coloured points hold 17,474 distinct exact colours.
#     Sent per point they are a palette almost as long as the cloud; at 32
#     levels per channel they collapse to 3,099, and the page ships a
#     palette table plus run lengths instead of a colour per point.
#   * SPEED. The canvas draw sets `fillStyle` once per run, not once per
#     point, so an orbit costs a few thousand style changes rather than
#     twenty thousand string allocations per frame.
#
# The error this costs is at most 4/255 per channel (1.6%), which is below
# what a 1.5-pixel dot can show. It is a display quantisation, not a
# recolouring: no point is moved toward any other point's colour.
COLOUR_QUANTISATION_BITS = 5

# Backgrounds, chosen from the measured luminance of real points rather
# than from taste. On world 52ed8e0a's re-solve the Rec.709 luminance of
# the triangulated colours runs p5=3, p25=35, p50=87, p75=138, p95=207.
# Against the near-black #111 this file used before, the darkest 24% of
# the room (luminance < 32) is invisible; against a near-white ground only
# the brightest 5% washes out. So the product view draws on paper and the
# diagnostic view keeps the dark ground the tab20 palette was tuned for.
PRODUCT_BACKGROUND = "#f4f3f0"
DIAGNOSTIC_BACKGROUND = "#111111"


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
    # Photometric colour per point, parallel to `points`. `rgb_known` is
    # the honest half: `rgb` holds NEUTRAL_POINT_COLOUR wherever the
    # solver wrote no colour, and only `rgb_known` distinguishes that from
    # a point the camera really did see as grey. Both are sliced by
    # `subsample` alongside `points`, so a thinned cloud never pairs a
    # point with another point's colour. Defaulted so the one construction
    # site below stays the only one that has to know about them.
    rgb: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.uint8))
    rgb_known: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))

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

    def colours_world(self):
        """(rgb (N,3) uint8, known (N,) bool), row-aligned with
        `points_world`. Separate from `points_world` rather than a third
        return value because `scripts/world_render.py` unpacks that one in
        four places and its PLY writer has no use for photometry."""
        rgb, known = [], []
        for segment in self.segments:
            if len(segment.points):
                rgb.append(segment.rgb)
                known.append(segment.rgb_known)
        if not rgb:
            return np.zeros((0, 3), dtype=np.uint8), np.zeros(0, dtype=bool)
        return np.vstack(rgb), np.concatenate(known)

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

    # `rgb` is OPTIONAL on a point row and read as such. The global solver
    # writes it (`global_solve.py:1047`); the incremental backend that
    # built most of the worlds already on disk does not, and 28 of the 34
    # field worlds measured on 2026-09-09 carry none at all. A missing
    # colour is a missing measurement, not a malformed row: it becomes a
    # False in `rgb_known` and a neutral dot on the page, never an error
    # and never a substituted colour.
    points_by_segment: dict = {}
    colours_by_segment: dict = {}
    for row in derived["points"]:
        index = int(row["segment_index"])
        points_by_segment.setdefault(index, []).append(row["xyz"])
        colour = row.get("rgb")
        known = (isinstance(colour, (list, tuple)) and len(colour) == 3
                 and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                         for v in colour))
        colours_by_segment.setdefault(index, []).append(
            ([int(v) for v in colour], True) if known
            else (list(NEUTRAL_POINT_COLOUR), False)
        )

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
        rows = colours_by_segment.get(index, [])
        rgb = np.asarray([r[0] for r in rows], dtype=np.int64).reshape(-1, 3)
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        rgb_known = np.asarray([r[1] for r in rows], dtype=bool)
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
            rgb=rgb,
            rgb_known=rgb_known,
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
    # WITH the session id -- see `WorldStore.derived_currency`. The render
    # page for an earlier walk showed "stale" against a digest belonging to
    # a different session entirely. `None` ("nothing can judge it") reports
    # as not-current here, which is what this function's own docstring
    # already promises for the unknowable case.
    return bool(store.derived_currency(world_id, digest, session_id) is True)


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
            picks = np.minimum(
                np.floor(np.arange(budget) * (n / budget)).astype(int), n - 1
            )
            segment.points = segment.points[picks]
            # Colour is indexed with the SAME picks, never re-sampled: a
            # thinned cloud that paired point i with colour j would be a
            # fabricated photograph of the room, and it would look
            # plausible, which is worse.
            if len(segment.rgb) == n:
                segment.rgb = segment.rgb[picks]
                segment.rgb_known = segment.rgb_known[picks]
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
        rgb, known = frame.colours_world()
        out.append({
            "name": (f"{frame.tag} (ref segment {frame.reference_segment}, "
                     f"{len(frame.segments)} registered segments)"),
            "shared": True,
            "label": (f"{_count(len(frame.segments), 'segment')} placed together"
                      if frame.tag == "world" else
                      f"separate group around segment {frame.reference_segment} "
                      f"({_count(len(frame.segments), 'segment')})"),
            # How many segments the store PLACED here, which is not the
            # same as how many are drawn: a registered segment that
            # triangulated no points is in the frame and contributes no
            # dots. The summary reports both rather than quietly using
            # whichever number is larger.
            "placed_segments": len(frame.segments),
            "xyz": xyz, "owner": owner,
            "rgb": rgb, "rgb_known": known,
            "cameras": frame.cameras_world(),
        })
    for segment in unregistered:
        if len(segment.points) == 0 and not segment.cameras:
            continue
        out.append({
            "name": (f"UNREGISTERED segment {segment.index} "
                     f"[{segment.placement_state}] -- own frame, own scale"),
            "shared": False,
            "label": (f"unplaced fragment: segment {segment.index} "
                      f"[{segment.placement_state}]"),
            "placed_segments": 0,
            "xyz": segment.points,
            "owner": np.full(len(segment.points), segment.index),
            "rgb": segment.rgb, "rgb_known": segment.rgb_known,
            "cameras": [(c.centre, c.rotation_world_camera, segment.index, c.status)
                        for c in segment.cameras],
        })
    return out


# What the picture is, in the viewer's own words. One sentence, always
# shown: the caption is the difference between "a sparse SfM point cloud"
# and "a scan of the room", and only the first is true.
#
# CAPTION is the DIAGNOSTIC caption and keeps its exact wording. It leads
# with the technique because the person reading it in diagnostics came for
# the technique.
CAPTION = ("Sparse structure-from-motion output: triangulated feature points "
           "and camera poses. Not a surface, not a mesh, not metric scale.")
# PRODUCT_CAPTION says the same true things in the order that answers "what
# am I looking at". Every disclaimer in CAPTION survives -- sparse, not a
# surface, not a mesh, not metric -- because dropping one of them to sound
# friendlier is how a point cloud becomes a "3D scan of your room". The
# added clause is the one this viewer never said and should have: the gaps
# are unmeasured, not measured-as-empty.
PRODUCT_CAPTION = ("The points this walk actually measured, in the colours the "
                   "camera recorded. Sparse structure from motion: only "
                   "textured surfaces produce points, so the empty space is "
                   "unmeasured rather than known to be empty. Not a surface, "
                   "not a mesh, not metric scale.")
CAPTION_BEHIND = ("This picture is BEHIND the newest keyframes: the Tower has "
                  "accepted keyframes it has not yet built into geometry.")

# The viewer. Self-contained: no external script, no stylesheet, no fetch,
# so it works with nothing but a browser, and inside a WKWebView with
# outbound navigation refused. Mouse: drag orbits, shift-drag pans, wheel
# zooms. Touch (added 2026-09-06 for the phone): one finger orbits, two
# fingers pinch to zoom and drag to pan. `touch-action: none` keeps the
# page from scrolling or zooming underneath the canvas.
#
# The CSP meta tag repeats the route's response header INSIDE the page, and
# it is the copy that matters on the phone: iOS drops the response headers
# and calls `loadHTMLString(_:baseURL: nil)`, and WebKit enforces a meta CSP
# in that document but never saw the header. It sits right after
# `<meta charset>`; `wb-representation` must stay inside the first 4096
# characters, where the phone reads the rung.
_CANVAS_VIEWER = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'"><meta name="wb-representation" content="sparse"><title>__TITLE__</title>
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<style>
:root{--bg:__PRODUCT_BG__;--fg:#1c1b19;--bar:#e9e7e2;--sub:#55524d;--edge:#d2cec7;--warn:#a4400a}
body.diag{--bg:__DIAGNOSTIC_BG__;--fg:#dddddd;--bar:#222222;--sub:#bbbbbb;--edge:#333333;--warn:#ff8800}
body{margin:0;font:13px system-ui,sans-serif;background:var(--bg);color:var(--fg);overflow:hidden}
#bar{padding:8px;background:var(--bar);display:flex;gap:10px;align-items:center;flex-wrap:wrap;border-bottom:1px solid var(--edge)}
#bar select,#bar button{max-width:100%;font:inherit;color:var(--fg);background:var(--bg);border:1px solid var(--edge);border-radius:4px;padding:2px 8px}
#bar button{cursor:pointer}
#bar label{color:var(--sub);cursor:pointer}
#caption,#facts{padding:4px 8px;background:var(--bar);color:var(--sub);font-size:12px;border-bottom:1px solid var(--edge)}
#caption b{color:var(--warn)}
#facts{color:var(--fg)}
#hint{color:var(--sub)}
#legend span{display:inline-block;margin-right:10px}
#legend i{display:inline-block;width:10px;height:10px;margin-right:4px;vertical-align:middle;border:1px solid var(--edge)}
#message{padding:16px;max-width:44em;line-height:1.5}
#message h2{font-size:15px;margin:0 0 8px}
#message ul{margin:8px 0;padding-left:18px;color:var(--sub)}
canvas{display:block;cursor:grab;touch-action:none;background:var(--bg)}
/* Our own rule, because `canvas{display:block}` above is an author rule
   and would otherwise beat the user agent's [hidden] no matter how
   specific that is. */
[hidden]{display:none!important}
</style></head><body>
<div id="bar"><b>__TITLE__</b>
<button id="mode" type="button"></button>
<select id="frame"></select>
<label><input type="checkbox" id="cams"> cameras</label>
<span id="hint">drag: orbit &middot; pinch/wheel: zoom &middot; two fingers/shift-drag: pan &middot; keys 1/2/3: top/front/side &middot; d: diagnostics</span>
<span id="legend"></span></div>
<div id="caption"><span id="cap-world">__PRODUCT_CAPTION__</span><span id="cap-diag" hidden>__CAPTION__</span>__CAPTION_SUFFIX__</div>
<div id="facts">__SUMMARY__</div>
<div id="message" hidden><h2>__MESSAGE_TITLE__</h2><div>__MESSAGE_BODY__</div>
<button id="to-diag" type="button">Open diagnostics</button></div>
<canvas id="c"></canvas>
<script>
const FRAMES = __FRAMES__;
const PAL = __PALETTE__, UNREG = __UNREGISTERED__, STATS = __STATS__;
const PRODUCT_BG = "__PRODUCT_BG__", DIAGNOSTIC_BG = "__DIAGNOSTIC_BG__";
const EMPTY_SUMMARY = __EMPTY_SUMMARY__;
const sel = document.getElementById('frame'), canvas = document.getElementById('c'),
      ctx = canvas.getContext('2d'), legend = document.getElementById('legend'),
      facts = document.getElementById('facts'), message = document.getElementById('message'),
      modeButton = document.getElementById('mode'), camsBox = document.getElementById('cams');
// The opening mode is decided WHEN THE PAGE IS COMPOSED, not read from
// the URL here.
//
// It was read from `location.search` until an iOS reviewer pointed out
// that the one client this page has cannot supply one. iOS loads it with
// `loadHTMLString(_:baseURL: nil)` -- no origin, no URL, so
// `location.search` is always empty -- and its navigation policy cancels
// the page's own links, so an in-page href could not reach a second view
// either. The affordance existed and did nothing on the only device that
// matters.
//
// The server substitutes the opening mode below and the route takes
// `?view=`, so the phone asks for the view it wants and gets a page that
// opens in it. The in-page toggle still switches modes for a browser,
// because both views ride on one payload.
let mode = __OPENING_MODE__;
let cur = -1, yaw = 0.7, pitch = 0.5, zoom = 1, panX = 0, panY = 0, drag = null, pinch = null;
// The backing store's scale over CSS pixels, set by resize(). Pointer
// deltas arrive in CSS pixels and the projection works in backing
// pixels, so pan is KEPT in CSS pixels and converted at the one place
// it is used -- otherwise a 2x store makes a drag move the world half
// as far as the finger.
let DPR = 1;
// In the world view a frame is only offered when the store placed its
// segments into a shared space. An unregistered fragment has its own
// scale -- up to ~87x off on a real walk -- so it is not a world and is
// reachable only from diagnostics, where its name says so.
function visibleFrames(){ const out = []; FRAMES.forEach((f, i) => { if (mode === 'diag' || f.shared) out.push(i); }); return out; }
function fillSelect(){
  const vis = visibleFrames(); sel.innerHTML = '';
  vis.forEach(i => { const o = document.createElement('option'); o.value = i;
    o.textContent = mode === 'diag' ? FRAMES[i].name : FRAMES[i].label; sel.appendChild(o); });
  if (vis.indexOf(cur) < 0) cur = vis.length ? vis[0] : -1;
  if (cur >= 0) sel.value = String(cur);
  sel.hidden = vis.length < 2;
}
function applyMode(){
  document.body.classList.toggle('diag', mode === 'diag');
  document.getElementById('cap-world').hidden = mode === 'diag';
  document.getElementById('cap-diag').hidden = mode !== 'diag';
  modeButton.textContent = mode === 'diag' ? 'Back to the world' : 'Diagnostics';
  camsBox.checked = mode === 'diag';
  fillSelect(); resize();
}
function chrome(){ return document.getElementById('bar').offsetHeight + document.getElementById('caption').offsetHeight + facts.offsetHeight + (message.hidden ? 0 : message.offsetHeight); }
// A BACKING STORE THE SCREEN'S SIZE, not the layout's.
//
// This set `canvas.width = innerWidth` with no CSS size, so on a phone the
// bitmap was CSS pixels and the browser upscaled it by the device pixel
// ratio with smoothing on. At dpr 3 a 1.8 px point became a ~5.4 px
// smoothed blob, and a cloud of blobs is a haze. The desktop this was
// inspected on runs at dpr 1, where the defect does not appear at all --
// so "the world reads as a diffuse point cloud" was partly a phone-only
// rendering artifact and not the reconstruction.
//
// Capped at 2 rather than taking dpr 3 whole. A 3x backing store measured
// 1.8x the draw cost, which is affordable at this world's 19k points and
// not at the 40k budget the phone may be handed; 2x buys most of the
// sharpness for less than half the extra cost. `imageSmoothingEnabled`
// off so nothing re-blurs what the extra pixels bought.
function resize(){
  const scale = DPR = Math.min(devicePixelRatio || 1, 2);
  const cssW = innerWidth, cssH = Math.max(50, innerHeight - chrome());
  canvas.style.width = cssW + 'px'; canvas.style.height = cssH + 'px';
  canvas.width = Math.round(cssW * scale); canvas.height = Math.round(cssH * scale);
  ctx.imageSmoothingEnabled = false;
  draw();
}
function rot(){ const cy=Math.cos(yaw), sy=Math.sin(yaw), cp=Math.cos(pitch), sp=Math.sin(pitch);
  return [[cy,0,sy],[sy*sp,cp,-cy*sp],[-sy*cp,sp,cy*cp]]; }
function draw(){
  const f = cur >= 0 ? FRAMES[cur] : null, W = canvas.width, H = canvas.height;
  ctx.fillStyle = mode === 'diag' ? DIAGNOSTIC_BG : PRODUCT_BG; ctx.fillRect(0,0,W,H);
  if (!f){
    // Not an empty box. The page says what WAS reconstructed and why none
    // of it could be put in one frame; the server composed that sentence
    // from the same counts this payload carries.
    legend.innerHTML = ''; facts.textContent = EMPTY_SUMMARY;
    message.hidden = false; canvas.hidden = true;
    return;
  }
  message.hidden = true; canvas.hidden = false;
  facts.textContent = mode === 'diag' ? f.name : f.summary;
  const R = rot(), c = f.centre, s = zoom * 0.9 * Math.min(W,H) / f.extent;
  function P(p){ const x=p[0]-c[0], y=p[1]-c[1], z=p[2]-c[2];
    const u = R[0][0]*x+R[0][1]*y+R[0][2]*z, v = R[1][0]*x+R[1][1]*y+R[1][2]*z;
    return [W/2 + panX*DPR + u*s, H/2 + panY*DPR - v*s]; }
  const dot = mode === 'diag' ? 1.5 : 1.8;
  for (const seg of f.segments){
    const xs = seg.xyz;
    if (mode === 'diag'){
      ctx.fillStyle = seg.colour;
      for (let i=0;i<xs.length;i+=3){ const q=P([xs[i],xs[i+1],xs[i+2]]); ctx.fillRect(q[0],q[1],dot,dot); }
    } else {
      // seg.xyz is ordered by colour run, so fillStyle changes once per
      // run instead of once per point: a few thousand style changes an
      // orbit rather than one string per point per frame. PAL[0] is the
      // neutral grey of a point the solver gave no colour.
      let i = 0;
      for (const run of seg.runs){ ctx.fillStyle = PAL[run[0]]; const end = i + run[1];
        for (; i < end; i++){ const j = i*3; const q=P([xs[j],xs[j+1],xs[j+2]]); ctx.fillRect(q[0],q[1],dot,dot); } }
    }
  }
  if (camsBox.checked){
    if (mode === 'diag'){
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 0.6;
      for (const cam of f.cameras){ ctx.beginPath(); for (const ln of cam.lines){ const a=P(ln[0]), b=P(ln[1]); ctx.moveTo(a[0],a[1]); ctx.lineTo(b[0],b[1]); } ctx.stroke();
        const q = P(cam.centre); ctx.fillStyle = cam.colour; ctx.beginPath(); ctx.arc(q[0],q[1],3,0,6.283); ctx.fill(); ctx.stroke(); }
    } else {
      // Context, not the subject: where the wearer stood, as small dim
      // marks. No frustums, because eight white lines per keyframe over
      // 652 keyframes is a wireframe box that hides the room inside it.
      ctx.fillStyle = 'rgba(30,60,110,0.5)';
      for (const cam of f.cameras){ const q = P(cam.centre); ctx.fillRect(q[0]-1.25,q[1]-1.25,2.5,2.5); }
    }
  }
  if (mode === 'diag'){
    legend.innerHTML = f.segments.map(sg => `<span><i style="background:${sg.colour}"></i>seg ${sg.index} (${sg.xyz.length/3})</span>`).join('')
      + ` cameras: ${f.cameras.length}` + (f.shared ? '' : ' <b>own frame, own scale</b>')
      + (UNREG.length ? ` &middot; ${UNREG.length} unregistered: ` + UNREG.map(u => `${u.index} [${u.state}]`).join(', ') : '');
  } else {
    legend.innerHTML = f.uncoloured ? `<span><i style="background:${PAL[0]}"></i>no measured colour</span>` : '';
  }
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
addEventListener('keydown', e => { if(e.key==='1'){yaw=0;pitch=Math.PI/2;} if(e.key==='2'){yaw=0;pitch=0;} if(e.key==='3'){yaw=Math.PI/2;pitch=0;}
  if(e.key==='d'||e.key==='D'){ toggleMode(); return; } if(e.key==='c'||e.key==='C'){ camsBox.checked = !camsBox.checked; } draw(); });
function toggleMode(){ mode = mode === 'diag' ? 'world' : 'diag'; panX = panY = 0; zoom = 1; applyMode(); }
modeButton.addEventListener('click', toggleMode);
document.getElementById('to-diag').addEventListener('click', () => { if (mode !== 'diag') toggleMode(); });
camsBox.addEventListener('change', draw);
sel.addEventListener('change', () => { cur = +sel.value; panX = panY = 0; zoom = 1; draw(); });
addEventListener('resize', resize); applyMode();
</script></body></html>
"""


def _escape(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def _count(n: int, singular: str, plural: str | None = None) -> str:
    """`1 point` / `26,634 points`. Thousands separators because a viewer
    reading "26634 points" off a phone reads it as a different number."""
    return f"{n:,} {singular if n == 1 else (plural or singular + 's')}"


def _colour_runs(rgb: np.ndarray, known: np.ndarray, palette: dict):
    """(reordering, [[palette index, run length], ...]) for one segment.

    The points are sorted by quantised colour so a run of identical
    colours is contiguous, and only the run boundaries reach the page.
    `palette` is the page-wide table being built; index 0 is reserved for
    NEUTRAL_POINT_COLOUR before this is ever called, so "no measured
    colour" is one specific entry the legend can name rather than a colour
    that merely happens to look grey.

    Sorting the points is safe here for a reason worth stating: this
    viewer draws unsorted dots with no depth test at all, so the order of
    points within a segment carries no information to destroy.
    """
    n = len(rgb)
    if n == 0:
        return np.zeros(0, dtype=int), []
    shift = 8 - COLOUR_QUANTISATION_BITS
    levels = 1 << COLOUR_QUANTISATION_BITS
    q = np.asarray(rgb, dtype=np.int64) >> shift
    code = (q[:, 0] * levels + q[:, 1]) * levels + q[:, 2]
    code = np.where(np.asarray(known, dtype=bool), code, -1)
    order = np.argsort(code, kind="stable")
    ordered = code[order]
    starts = np.concatenate(([0], np.flatnonzero(np.diff(ordered)) + 1))
    lengths = np.diff(np.concatenate((starts, [n])))
    runs = []
    for start, length in zip(starts, lengths):
        value = int(ordered[start])
        if value < 0:
            index = 0
        else:
            if value not in palette:
                # Back to 0..255 from the bucket, scaled so the top bucket
                # is 255 rather than 248: a quantiser that cannot reach
                # white would tint every specular highlight in the room.
                channels = tuple(
                    int(round(((value // levels ** k) % levels) * 255 / (levels - 1)))
                    for k in (2, 1, 0)
                )
                palette[value] = (len(palette) + 1, channels)
            index = palette[value][0]
        runs.append([index, int(length)])
    return order, runs


def _frame_summary(descriptor: dict, *, other_shared: int, unregistered_rows: list,
                   sampling: dict | None) -> str:
    """The sentence under the toolbar in the world view.

    Composed here rather than in the browser so that it is in the SERVED
    HTML: a claim about what the viewer is looking at should be assertable
    by a test that never runs JavaScript.
    """
    points, uncoloured = descriptor["points"], descriptor["uncoloured"]
    if descriptor["shared"]:
        placed, drawn = descriptor["placed_segments"], len(descriptor["segments"])
        parts = [f"{_count(points, 'point')} from "
                 f"{_count(placed, 'segment')} placed in one frame."]
        if drawn < placed:
            parts.append(f"{placed - drawn} of those segments produced no points "
                         "to draw.")
    else:
        parts = [f"{_count(points, 'point')} from one segment that could not be "
                 "placed with any other, drawn in its own frame at its own scale."]
    if points and uncoloured == points:
        parts.append("None of them carry a measured colour, so every point is "
                     "drawn neutral grey.")
    elif uncoloured:
        parts.append(f"{_count(uncoloured, 'point')} carry no measured colour and "
                     "are drawn neutral grey.")
    elif points:
        # Stated positively rather than left out. "How much of this is a
        # measurement and how much is a placeholder" is the question this
        # line exists to answer, and silence answers it either way.
        parts.append("Every point carries a measured colour.")
    if sampling and sampling.get("subsampled"):
        parts.append("Thinned for drawing from "
                     f"{_count(int(sampling['points_total']), 'point')}.")
    if other_shared:
        parts.append(f"{_count(other_shared, 'other group', 'other groups')} of "
                     "segments could not be placed relative to this one.")
    if unregistered_rows:
        parts.append(f"{_count(len(unregistered_rows), 'segment')} could not be "
                     "placed at all.")
    return " ".join(parts)


def _nothing_placed_text(unregistered_rows: list, stats: dict) -> tuple:
    """(title, body) for the panel that replaces the canvas when there is
    no shared frame to draw.

    An empty box is a lie by omission: it reads as "nothing was mapped"
    when what actually happened is that plenty was reconstructed and none
    of it could be tied together. So the panel names what exists, and the
    three degraded cases are worded apart because they are different
    failures with different fixes.
    """
    segments = stats["segments_total"]
    if not segments:
        return ("Nothing was reconstructed for this session.",
                "No segment in the derived tree has points or posed cameras, so "
                "there is nothing to draw. This is what an unbuilt or emptied "
                "session looks like, not a room that failed to map.")
    if stats["points_total"] == 0:
        return (f"None of the {_count(segments, 'segment')} in this session has "
                "triangulated points.",
                f"{_count(stats['cameras_total'], 'solved camera pose')} and "
                f"{_count(stats['cameras_refused'], 'refused pose')} are on "
                "disk, but no 3-D points came out of them, so there is no "
                "geometry to show. The camera poses are still viewable under "
                "diagnostics.")
    states: dict = {}
    for row in unregistered_rows:
        states[row["state"]] = states.get(row["state"], 0) + 1
    breakdown = ", ".join(f"{count} {state}" for state, count in sorted(states.items()))
    return (f"{_count(segments, 'segment')} were reconstructed, but none could be "
            "placed relative to each other.",
            f"There are {_count(stats['points_total'], 'point')} and "
            f"{_count(stats['cameras_total'], 'camera pose')} on disk. What is "
            "missing is registration: no segment has a transform tying it to "
            "another, so there is no single space to draw them in, and drawing "
            "them together anyway would invent a room. "
            f"By placement state: {breakdown}. Each fragment is still viewable on "
            "its own, at its own scale, under diagnostics.")


def canvas_html(frames: list, unregistered: list, ordering: list, title: str, *,
                current=None, sampling: dict | None = None,
                view: str = VIEW_PRODUCT) -> str:
    """The self-contained viewer as a string. `current` is
    `derived_current`'s answer; False adds the behind-the-journal line to
    the caption, True and None (unknowable) do not. `sampling` is
    `subsample`'s report, so the page can say it is drawing a thinned
    cloud rather than quietly showing fewer points than exist.

    ONE payload serves both views. Composing the world page and the
    diagnostic page separately would double the bytes over the wire and
    give the two views two chances to disagree about what is in the frame,
    which is precisely the failure a diagnostic view exists to rule out.

    `view` picks which of the two the page OPENS in, server-side, because
    the phone cannot say it any other way: it loads this string with no URL
    at all. The toggle still works once the page is up.
    """
    descriptors = html_frames(frames, unregistered, ordering)
    # `palette` maps quantised colour code -> (index in the page's table,
    # channels). Index 0 is reserved for the neutral grey and is emitted
    # whether or not any point needs it, so `PAL[0]` in the page is a
    # constant the legend can point at.
    palette: dict = {}
    payload = []
    for descriptor in descriptors:
        xyz, owner = descriptor["xyz"], descriptor["owner"]
        rgb, known = descriptor["rgb"], descriptor["rgb_known"]
        if len(rgb) != len(xyz):
            # A derived tree written before `rgb` existed, or one whose
            # colour column is short. Neutral for all of it, rather than a
            # guess at which points the colours it does have belonged to.
            rgb = np.tile(np.asarray(NEUTRAL_POINT_COLOUR, dtype=np.uint8),
                          (len(xyz), 1))
            known = np.zeros(len(xyz), dtype=bool)
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
            mask = owner == index
            r, g, b = segment_colour(int(index), ordering)
            order, runs = _colour_runs(rgb[mask], known[mask], palette)
            segments.append({
                "index": int(index), "colour": f"rgb({r},{g},{b})",
                # Ordered by colour run. The diagnostic view ignores
                # `runs` and paints the whole array one segment colour, so
                # the two views share these bytes exactly.
                "xyz": np.round(xyz[mask][order], 4).ravel().tolist(),
                "runs": runs,
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
            "label": descriptor["label"],
            "placed_segments": descriptor["placed_segments"],
            "centre": centre.tolist(), "extent": extent,
            "segments": segments, "cameras": cameras,
            "points": int(len(xyz)),
            "uncoloured": int(len(known)) - int(np.count_nonzero(known)),
        })

    palette_entries = [NEUTRAL_POINT_COLOUR] + [
        channels for _code, (_index, channels)
        in sorted(palette.items(), key=lambda kv: kv[1][0])
    ]
    unregistered_rows = [{
        "index": segment.index,
        "state": segment.placement_state,
        "reason": segment.refusal_reason or "",
        "points": int(len(segment.points)),
        "cameras": len(segment.cameras),
    } for segment in unregistered]
    # Counted over the composed structures rather than over `segments`,
    # which this function is not given: a frame's members plus the
    # unregistered list are every segment the loader saw.
    stats = {
        "segments_total": sum(len(f.segments) for f in frames) + len(unregistered),
        "points_total": sum(f.point_count for f in frames)
                        + sum(len(s.points) for s in unregistered),
        "cameras_total": sum(len(f.cameras_world()) for f in frames)
                         + sum(len(s.cameras) for s in unregistered),
        "cameras_refused": sum(s.cameras_refused for f in frames for s in f.segments)
                           + sum(s.cameras_refused for s in unregistered),
        "shared_frames": sum(1 for p in payload if p["shared"]),
        "unregistered": len(unregistered),
    }
    shared_count = stats["shared_frames"]
    for entry in payload:
        entry["summary"] = _frame_summary(
            entry,
            other_shared=(shared_count - 1 if entry["shared"] else shared_count),
            unregistered_rows=unregistered_rows, sampling=sampling,
        )

    message_title, message_body = _nothing_placed_text(unregistered_rows, stats)
    opening = next((p for p in payload if p["shared"]), None)
    summary = opening["summary"] if opening else message_title

    caption_suffix = ""
    if current is False:
        caption_suffix = f" <b>{_escape(CAPTION_BEHIND)}</b>"

    # A `<` inside the JSON is the one character that could end the
    # script block early (`</script>`) or, as `<!--`, put the parser into
    # its escaped state. Every string in the payload is server-composed
    # today, but the payload is written once and read by a browser, so it
    # is escaped as `\u003c`, which JSON.parse reads back as `<`.
    def _json(value) -> str:
        return json.dumps(value, separators=(",", ":")).replace("<", "\\u003c")

    tokens = {
        "__TITLE__": _escape(title),
        "__CAPTION__": _escape(CAPTION),
        "__PRODUCT_CAPTION__": _escape(PRODUCT_CAPTION),
        "__CAPTION_SUFFIX__": caption_suffix,
        "__SUMMARY__": _escape(summary),
        "__MESSAGE_TITLE__": _escape(message_title),
        "__MESSAGE_BODY__": _escape(message_body),
        "__EMPTY_SUMMARY__": _json(f"{message_title} {message_body}"),
        "__FRAMES__": _json(payload),
        "__PALETTE__": _json([f"#{r:02x}{g:02x}{b:02x}" for r, g, b in palette_entries]),
        "__UNREGISTERED__": _json(unregistered_rows),
        "__STATS__": _json(stats),
        # An unrecognised view opens the product page. This is a display
        # mode on an unauthenticated route, so the safe reading of a value
        # nobody understands is "show the normal thing", not an error page.
        "__OPENING_MODE__": (
            "'diag'" if view == VIEW_DIAGNOSTICS else "'world'"
        ),
        "__PRODUCT_BG__": PRODUCT_BACKGROUND,
        "__DIAGNOSTIC_BG__": DIAGNOSTIC_BACKGROUND,
    }
    # One pass over the template, so a placeholder appearing INSIDE a
    # substituted value is never substituted again. A refusal reason is
    # store data and this file does not get to assume what is in it; the
    # chained `.replace()` this used to be would have expanded one.
    return re.sub(r"__[A-Z_]+__", lambda m: tokens.get(m.group(0), m.group(0)),
                  _CANVAS_VIEWER)


def render_html(store: WorldStore, world_id: str, session_id: str, *,
                max_points: int = DEFAULT_MAX_POINTS, title: str | None = None,
                view: str = VIEW_PRODUCT) -> str:
    """One session of one world as the interactive viewer. Raises
    `FileNotFoundError` when the session has no derived tree.

    `view` is the mode the page opens in -- see `canvas_html`. It must be
    honoured here rather than in the browser, because iOS loads the page
    with no URL and cannot pass a query string to it."""
    segments = load_segments(store, world_id, session_id)
    ordering = sorted(segments)
    current = derived_current(store, world_id, session_id)
    sampling = subsample(segments, max_points)
    frames, unregistered = compose_frames(segments)
    if title is None:
        title = f"world {world_id[:8]} session {session_id[:8]}"
    # The sampling report is carried into the page rather than dropped: a
    # phone gets MOBILE_MAX_POINTS of an 80,000-point world and the page
    # said "80,000 points" without saying which 80,000, or of how many.
    return canvas_html(frames, unregistered, ordering, title,
                       current=current, sampling=sampling, view=view)
