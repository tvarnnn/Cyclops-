"""Volumetric surface reconstruction: the field, the mesh, and the format.

Why a distance field rather than more points.

The dense stage measures where surfaces are and stores the measurements as
points. A point cloud has no inside and no outside, so it cannot occlude, and
disagreement between frames has to be resolved per point by discarding the
losers -- which on a plain wall, where frames disagree most, discards the wall.

A truncated signed distance field stores, for every small region of space, how
far it is to the nearest surface AND how much evidence there is for that
answer. Three things follow:

  * The zero crossing is a surface. It occludes. You cannot see through a wall
    that is a wall.
  * Disagreement is averaged once, in three dimensions, weighted by evidence,
    instead of being adjudicated per point.
  * Free space is evidence too. A camera measuring depth d along a ray has also
    measured that everything nearer than d is empty. Carving that emptiness
    removes flying pixels and floaters, and `evidence_filter` removes a thing
    that moved between frames -- a hand, a pet -- WHEN the views that saw
    through where it was outnumber the views that measured it by
    `contradiction_ratio` (2:1). One held through three frames and looked past
    by five stays.

What this deliberately does NOT do is close holes. Space no camera measured
keeps zero weight and emits no triangle; the surface stops at the edge of what
was seen. The converse does not hold: a hole is not proof that nobody looked.
Where frames measured different things, the surface there is outvoted, below
the evidence threshold, or removed by `evidence_filter`, and measured on the
canonical capture that -- not missing coverage -- is most of what a hole is. That is the line between this and Poisson or Delaunay
reconstruction, which are watertight by construction and would turn "never
observed" into "surface here". `WORLD-BUILDER-SURFACE.md` states it as a rule
of the format: any future change that closes unobserved space must break the
format identifier rather than quietly relax it.

Every length here is a FRACTION of the scene's own scale (the median depth
of the solve's sparse observations, `surface_pipeline._scene_scale`), except
the truncation band, which follows each sample's own depth. The SfM gauge
is arbitrary -- `global_solve` never calls COLMAP's `normalize()` -- and the
same room has solved to a ten-unit extent and to a three-hundred-unit one. A
constant voxel size would shatter one world and collapse another.
"""

from __future__ import annotations

import math
import struct
import time
from dataclasses import dataclass, field

import numpy as np


def torch_full_like(t, value):
    import torch

    return torch.full_like(t, float(value))


SURFACE_FORMAT = "wb-surface-mesh/1"
SURFACE_FORMAT_ENCLOSED_FILL = "wb-surface-mesh/1+enclosed-fill"
"""The identifier an artifact built with `fill_gap_frac > 0` must carry.

`WORLD-BUILDER-SURFACE.md` section 3: any change that fills unobserved space
must break the format identifier, because a reader that sees
`wb-surface-mesh/1` is entitled to believe every triangle was measured. The
byte layout is unchanged; the promise is not, so the name is not either, and
every reader that checks for `SURFACE_FORMAT` refuses a filled artifact until
someone decides it should not."""
SURFACE_SCHEMA_VERSION = 1

# What a phone is sent is a PAGE: the viewer template, its configuration, and
# the mobile level's mesh base64-encoded inside it, which is 4/3 of the mesh's
# bytes. The budget was once applied to the mesh bytes alone, and a 5.88 MB
# level shipped as a 7.89 MB page (live replay D).
MOBILE_PAGE_BYTES = 6 * 1024 * 1024
MESH_MAGIC = b"WBSURF01"
SNAP_VERSION = 1

STAGE_DEPTH = "depth"
STAGE_FUSE = "fuse"
STAGE_MESH = "mesh"
STAGE_PACK = "pack"

BLOCK = 8                       # voxels along a block edge
BLOCK_VOXELS = BLOCK ** 3

SHELL_MARGIN_BLOCKS = 0
"""Blocks allocated beyond the truncation band. It was 1. A band of half-width
`tr` around a point already lies within ceil(tr / block) blocks of the point's
own block, so the margin held only voxels OUTSIDE the band -- which no sample
writes a surface value into, and which, left at the +1 default beside the back
of a band, were exactly what built phantom sheets (see `extract_mesh`). With
extraction requiring evidence at every cube corner they add nothing, and with a
depth-proportional band they cost 20% more blocks on the canonical world
(392k against 472k), pushing it over the budget."""

_KEY_OFFSET = 1 << 20
_KEY_SPAN = 1 << 21


class SurfaceUnavailable(RuntimeError):
    """The surface stage cannot run, with a reason a person can read."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SurfaceParams:
    """Everything the fusion needs, in units of the scene's own median depth."""

    # -- resolution ---------------------------------------------------------
    voxel_frac: float = 0.0051
    """Voxel edge as a fraction of the scene scale (`surface_pipeline.
    _scene_scale`: the median depth of the solve's sparse observations).
    0.0051 of the canonical world's 4.72 is 0.0241 -- the same voxel it was
    built at when this was 0.006 of a 4.02 dense-depth median -- which matches
    the spacing the dense stage's L1 ladder shipped at, and is about the point
    where finer voxels store noise: the imagery is 0.23 MP, so a pixel covers
    ~6 mm at 3 m and depth noise is 1-3 cm."""

    trunc_voxels: float = 3.0
    """Truncation floor, in voxels."""

    trunc_depth_proportional: bool = True
    """Size the band per sample: max(floor, multiple * held-out error * d).

    The held-out error is RELATIVE, so the depth disagreement it measures
    grows with range. One absolute band sized at the scene median was 4x the
    error at desk range and 0.65x of it at the far wall (2.3% of 12 units is
    0.28 against a 0.18 band) -- too wide close up, where a 1-4 voxel board
    seen from both sides fused 9-12 voxels thick, and self-erasing far away.
    `trunc_voxels` stays the floor, so a near band never falls under what
    the voxel can represent."""

    trunc_max_voxels: float = 12.0
    """Ceiling on the depth-proportional band, in voxels.

    Blocks follow band width: uncapped, the far wall's 0.55-unit band took the
    canonical world from 290k to 392k blocks, past the 360k budget, and the
    budget then coarsened every voxel by 1.2x -- the whole room paid for the
    far wall. 12 voxels is about one held-out error (1 sigma) at the 12-unit
    far mode rather than two; it is still 1.6x the old absolute band there."""

    trunc_error_multiple: float = 2.0
    """Truncation is raised to this multiple of the measured median held-out
    depth error. THIS IS THE PARAMETER THAT DECIDES COVERAGE, and the reason
    is not obvious: at the conventional 3 voxels the truncation was 0.073
    while frames disagree with each other by 2.3% of a 4.07 scene depth --
    0.094. A frame whose depth ran long therefore wrote free space exactly
    where a correct frame had written surface, and the two carved each other
    away. Truncation narrower than the disagreement between frames is not a
    tighter reconstruction, it is a self-erasing one."""

    # -- evidence -----------------------------------------------------------
    min_weight: float = 2.0
    """Accumulated weight a cell needs before it may emit surface. Weight is
    in units of one square-on pixel at its frame's median depth, and a nearer
    pixel counts up to `max_near_boost` times, so ONE close frame can reach
    2.0 by itself. Weight is therefore not a frame count; the requirement that
    two distinct frames agree is `min_support_frames`, enforced per face."""

    carve: bool = True
    carve_weight: float = 0.35
    """Free space is real evidence but weaker than a surface hit: a depth map
    is confident about the surface it found and only implicitly confident
    about the emptiness in front of it."""

    max_carve_voxels: float = 40.0
    """How far in front of a surface to carve. Unbounded carving would erase
    geometry a distant frame happens to see past."""

    # -- per-pixel rejection ------------------------------------------------
    edge_rel: float = 0.03
    """Relative depth gradient that marks a discontinuity. A pixel spanning
    one gets a depth belonging to neither surface."""

    max_grazing_deg: float = 80.0
    """Beyond this incidence the depth error of a pixel grows without bound.
    Unlike the point pipeline, the incidence cosine BELOW the cut is kept and
    used as the integration weight rather than thrown away, so a surface seen
    at 70 degrees still contributes in proportion to how well it was seen."""

    anchor_depth_multiple: float = 1.5
    """A pixel deeper than this multiple of ITS OWN FRAME's farthest fitted
    sparse anchor (`z_sparse_max` in the depth stage's fit) is not used.

    The far bound is per frame and tied to what the solve supports, because
    the scene is not one depth. This used to be a single clip at 2.6 x the
    scene median, 10.44 units on the canonical world, and it deleted 9.0% of
    all valid depth -- the far wall, the doorway and the ceiling corner that
    45+ frames measured -- while the product told the wearer those gaps were
    places nothing looked. Per-frame medians there run from 1.7 to 11.3; a
    global relative clip cannot be right for both ends of that.

    Depth beyond a frame's farthest anchor is extrapolation of the affine fit,
    not measurement. Measured on the canonical world: 4.5% of valid depth lies
    beyond 1.0 x the frame's farthest anchor, 1.53% beyond 1.2 x, 0.51% beyond
    1.5 x. 1.5 keeps the far wall and refuses what the fit never saw."""

    max_depth_frac: float = 6.0
    """Fallback only, for a fit that records no anchor range: depth beyond this
    multiple of the scene scale is not used. Deliberately generous -- it is a
    guard against absurd depth, not a statement about where the room ends."""

    fill_margin_px: int = 8
    """Pixels around a redaction fill that are not used as depth. See
    `surface_pipeline._dilate_fill`."""

    gate_rel: float = 0.08
    """A frame whose held-out alignment residual exceeds this is not used at
    all. Inherited from the dense stage so the two agree about which frames
    are trustworthy."""

    frame_weight_floor: float = 0.25
    """The worst frame that still passes the gate contributes this much."""

    depth_falloff: bool = True
    """Weight falls as 1/z^2: a far pixel's depth is less certain and covers
    more volume."""

    max_near_boost: float = 4.0
    """Ceiling on the falloff's weight for a pixel NEARER than its frame's
    median depth. At 4.0 a pixel at half its frame's median depth -- a hand, a
    lap, the edge of the desk under the glasses -- counts as four pixels, so
    one close frame at a good angle CAN reach `min_weight` on its own. That is
    not what stops a one-frame surface: `min_support_frames` is, per face, as a
    count of distinct frames.

    Clamping this at 1 was measured and rejected (surface-r2): it removed 16%
    of the surface area, all of it supported by at least two frames and not
    contradicted -- the bed sheet and pieces of floor -- and removed no face
    near the walked path that the frame count had not already removed."""

    drop_back_facing: bool = True
    """Remove a face whose normal points away from EVERY frame that supports
    it (`evidence_filter`). A surface is seen from its front; a face that no
    supporting camera sees from the front is a fold where frames that disagree
    by a few voxels cross -- 18% of the shipped canonical area, the "crumpled
    foil" at grazing views. A thin board seen from both sides keeps both
    faces, because each side has cameras in front of it."""

    min_support_frames: int = 2
    """A face survives only if at least this many DISTINCT frames measured
    depth within truncation of it (`evidence_filter`). Weight is a sum and can
    be reached by one frame; this is a count and cannot."""

    contradiction_ratio: float = 2.0
    """A face is removed when the frames that saw THROUGH it -- measured depth
    beyond it by more than the truncation, with nothing in between -- number at
    least this multiple of the frames that support it. Carving blends free
    space into the field only within `max_carve_voxels` of each frame's own
    surface, so a hand half a metre from the camera was never carved by the
    frames that looked past it at a wall three metres away; the count sees the
    whole ray. 0 disables the test.

    It is a RATIO, so it does not remove every hand: one that 3 frames measured
    and 5 saw past (5 < 2 x 3) stays, and consecutive keyframes often hold the
    same hand. Nor can it tell a frame that saw past a thing from one that
    could not resolve it: a thin pole two near frames measured is removed if
    four distant frames smoothed it into the wall behind."""

    # -- surface cleanup ----------------------------------------------------
    min_component_frac: float = 0.0001
    """Connected components smaller than this fraction of the largest are
    dropped as noise.

    Measured on a WELDED mesh. It was 0.0008 when the tile seams were never
    welded, and "the largest component" was then one tile of wall -- 151,693
    faces -- rather than the room, 1,858,865; so 0.0008 in practice meant
    islands under ~120 faces, and that is what the renders were judged at.
    On the welded mesh 0.0008 would mean ~1,500 faces and delete 15% more
    real surface; 0.0001 is ~190 faces, the strictness that was actually
    approved, now measured against the right component."""

    smooth_iterations: int = 6
    smooth_lambda: float = 0.5
    smooth_mu: float = -0.53
    """Taubin smoothing: a shrinking pass followed by an expanding one, which
    removes the marching-cubes staircase without the volume loss plain
    Laplacian smoothing causes."""

    # -- enclosed-hole fill (OPT-IN, default off) ----------------------------
    fill_gap_frac: float = 0.0
    """Widest gap the enclosed-hole fill may close, as a fraction of median
    scene depth. 0 disables it, and 0 is the default: this is the one step
    that writes surface where no frame measured any, so it stays off until a
    person has reviewed the tinted diagnostic of exactly what it adds.

    When on, an unobserved voxel is filled only if marches along at least
    `fill_enclose_dirs` of the 26 axis/diagonal directions reach observed
    field within `fill_gap_frac` -- the capture BRACKETED it -- and the
    value written is a Laplace interpolation of the surrounding field at
    exactly `min_weight`, never more.

    Measured on the b2a75ab4 capture at 0.018 (3 voxels): unsealed it adds
    63k faces (+2.0%), 77% of them in patches that still own an open rim --
    frontier growth, not closure. Sealed it adds 8k faces (+0.25%) and does
    not visibly change how the room reads, because the dark speckle a viewer
    sees is lace at the edge of what was observed, not bracketed holes. That
    is the reason this is off."""

    fill_enclose_dirs: int = 22
    """Of the 26 directions, how many must reach observed field."""

    fill_sealed_only: bool = True
    """After extraction, revert in the field every filled patch that still
    owns an open boundary edge (`extract_sealed`): it did not close a hole,
    it grew a frontier or left a sliver wall. Turning this off ships that."""

    # -- level of detail ----------------------------------------------------
    lod_face_targets: tuple[int, ...] = (0, 600_000, 300_000)
    """Faces per level; 0 means "no decimation". Level 0 is the archive,
    level 2 is what a phone is sent, and its target is a ceiling: the pack
    stage decimates it further until the page fits `mobile_page_bytes`. Each level is decimated from the one
    before it, not from level 0: decimation grows as faces^1.37, and on a
    20-30 minute walk decimating every level from the full mesh projected to
    41-72 minutes of packing alone."""

    max_blocks: int = 360_000
    """The field's budget, in 8^3 blocks: ~3.4 GiB of field at 20 bytes a
    voxel, which with the frames held in host memory fits a 12 GB card with
    room for the depth network and the viewer's host. A walk that would
    allocate more gets a coarser voxel, chosen before the field is allocated
    and recorded in the manifest, rather than an out-of-memory failure. Block
    count follows the area of surface seen, not the number of frames -- 132
    frames of a tight bathroom made 380k blocks -- so this binds on big or
    cluttered spaces and on long walks alike. The canonical world is 234k."""

    mobile_level: int = 2
    canonical_level: int = 0

    mobile_page_bytes: int = MOBILE_PAGE_BYTES
    """The PAGE a phone is sent must fit this, and the pack stage makes the
    `mobile_level` the largest decimation of its parent level that does:
    `lod_face_targets[mobile_level]` is then a ceiling, not the answer. At
    the canonical world's ~19.6 bytes a face, 6 MiB of page is about 238k
    faces. 0 disables the fit (the target is used as given)."""

    component: int = 0
    """Only the reference component. Components share no unit, so fusing two
    of them into one field would place geometry at meaningless relative
    scales."""

    # -- cross-frame depth consistency (`depth_consistency.py`) --------------
    depth_consistency: bool = True
    """Fuse each frame's depth through the jointly solved smooth correction
    field rather than its plain affine fit. Frames disagreed with each other by
    6-12 voxels about where a wall is; this is what makes them agree. A field
    the held-out checks refuse, or a solve that fails, falls back to the plain
    affine and `manifest.detail.depth_consistency` says so."""

    consistency_outer: int = 5
    """Outer (re-correspondence) iterations of a cold solve."""

    consistency_warm_outer: int = 2
    """Outer iterations when warm-started from the previous solve's field.
    Two matched a cold solve on the canonical capture; one came within
    0.03 points of it in 8 s."""

    # -- plane snap (`snap_planes`) -----------------------------------------
    plane_snap: bool = True
    """After extraction, move vertices that already lie on a large, measured
    plane onto it, within `snap_tol_voxels`. Nothing is added, extended or
    filled; see `snap_planes`."""

    snap_min_area_frac: float = 0.27
    """Smallest plane snapped, as a fraction of the squared scene scale:
    6 square units on the canonical capture (scale 4.72). At 1.5 units the
    gate accepted desk tops and small ceiling pieces and flattened objects."""

    snap_tol_voxels: float = 2.5
    """Fit tolerance and snap distance, in voxels: the cross-frame spread
    that remains on walls and ceiling after the consistency field. A plane
    whose own RMS exceeds it is not snapped; a vertex farther than twice it
    from the plane does not move."""

    snap_min_frames: int = 8
    """Distinct keyframes whose depth must measure a plane before it is
    snapped."""
    transient_detector: str = "union"
    """Which transient detector masks the wearer's hands, arms and held phone
    out of fusion (`transients.py`; `union`, `oneformer` or `off`). A masked
    pixel has zero weight: it neither measures nor carves. Not in
    `digest_fields` (so artifacts built before it keep their digest); the
    pipeline appends the detector's rule id to the params digest instead."""

    quality: str = "final"
    """`final` or `live`. Recorded in the manifest so a reader -- and the
    wearer -- can tell a coarse reconstruction built during the walk from the
    refined one built after Stop. They occupy the same artifact directory and
    the final one replaces the live one, because the product question is
    always "the best available reconstruction of this session"."""

    @classmethod
    def live(cls, **overrides) -> "SurfaceParams":
        """The preset the builder uses DURING a walk.

        Coarser in every dimension that costs wall clock, because the live
        job's budget is the gap between global solves -- roughly 18 s of
        walking at this capture's keyframe rate -- and being late is worse
        than being coarse. Doubling the voxel divides the block count by
        eight, which is where most of the saving is; the rest comes from
        one level of detail instead of three and a lighter smoothing pass.

        What it does NOT relax is the evidence rule. `min_weight` drops from
        2.0 to 1.5 because a live field has had fewer frames to accumulate
        weight, not because a live surface is allowed to be invented.
        """
        base = dict(
            voxel_frac=0.0102,
            min_weight=1.5,
            smooth_iterations=3,
            lod_face_targets=(0, 120_000),
            canonical_level=0,
            mobile_level=1,
            min_component_frac=0.0003,
            # One model during a walk; the union at Stop (WORLD-BUILDER-SURFACE.md §2).
            transient_detector="oneformer",
            quality="live",
            consistency_outer=3,
            consistency_warm_outer=1,
        )
        base.update(overrides)
        return cls(**base)

    def digest_fields(self) -> tuple:
        """The parameters a cached artifact must match to be reusable.

        The fill fields are appended only when the fill is on, so artifacts
        built before the option existed keep their digest and are not rebuilt
        for a parameter they never used.
        """
        base = (
            self.voxel_frac, self.trunc_voxels, self.trunc_error_multiple,
            self.trunc_depth_proportional, self.trunc_max_voxels, "all-corners",
            self.min_weight, self.carve, self.carve_weight,
            self.max_carve_voxels, self.edge_rel, self.max_grazing_deg,
            self.max_depth_frac, self.anchor_depth_multiple, self.fill_margin_px,
            self.gate_rel, self.frame_weight_floor,
            self.depth_falloff, self.max_near_boost, self.min_support_frames,
            self.contradiction_ratio, self.drop_back_facing, self.min_component_frac,
            self.smooth_iterations, self.smooth_lambda, self.smooth_mu,
            self.lod_face_targets, self.component, self.quality,
            self.max_blocks, "lod-cascade", ("mobile-page", self.mobile_page_bytes),
            # Always present, deliberately: a surface fused before the field
            # and the snap existed is NOT what these parameters build.
            ("depth-consistency", self._consistency_digest()),
            ("plane-snap", self.plane_snap, self.snap_min_area_frac,
             self.snap_tol_voxels, self.snap_min_frames, SNAP_VERSION),
        )
        if self.fill_gap_frac > 0:
            base = base + ("fill", self.fill_gap_frac, self.fill_enclose_dirs,
                           self.fill_sealed_only)
        return base

    def _consistency_digest(self):
        if not self.depth_consistency:
            return "off"
        from tower.world_builder.depth_consistency import (  # noqa: PLC0415
            CONSISTENCY_VERSION,
            ConsistencyParams,
        )

        return (CONSISTENCY_VERSION, ConsistencyParams().digest(),
                self.consistency_outer, self.consistency_warm_outer)

    def fill_radius_voxels(self) -> int:
        """March length in voxels for the configured closable gap; 0 = off."""
        if self.fill_gap_frac <= 0:
            return 0
        # A gap closes cleanly only when a corner voxel can see across all
        # of it, so the march radius is the gap's full width, not half.
        return max(1, int(math.floor(self.fill_gap_frac / self.voxel_frac + 1e-9)))


@dataclass
class SurfaceResult:
    state: str
    detail: str | None = None
    frames_used: int = 0
    frames_offered: int = 0
    vertices: int = 0
    faces: int = 0
    blocks: int = 0
    voxel: float = 0.0
    trunc: float = 0.0
    levels: list = field(default_factory=list)
    seconds: dict = field(default_factory=dict)
    stopped_after: str | None = None
    permanent: bool = False
    """True when the stage cannot run on this machine at all -- the depth
    network is not installed -- as opposed to cannot run on this session yet.
    The live worker stops relaunching on it rather than failing every solve."""

    def as_dict(self) -> dict:
        return {
            "state": self.state, "detail": self.detail,
            "frames_used": self.frames_used, "frames_offered": self.frames_offered,
            "vertices": self.vertices, "faces": self.faces, "blocks": self.blocks,
            "voxel": self.voxel, "trunc": self.trunc, "levels": self.levels,
            "seconds": self.seconds, "stopped_after": self.stopped_after,
            "permanent": self.permanent,
        }


# ---------------------------------------------------------------------------
# block keys
# ---------------------------------------------------------------------------


def block_key(bc, *, offset: int = _KEY_OFFSET, span: int = _KEY_SPAN):
    """Block coordinates -> one int64 key.

    Raises rather than aliasing. `voxel_reduce` in the dense stage learned
    this the hard way: outside the keyable range two different cells share a
    key and the reduction silently MERGES them, averaging points from opposite
    ends of a scene into one.
    """
    import torch

    q = bc + offset
    if int(q.min()) < 0 or int(q.max()) >= span:
        raise SurfaceUnavailable(
            "the scene's block coordinates fall outside the keyable range; "
            "the SfM gauge is arbitrary and this world's is too large for the "
            "chosen voxel fraction")
    if isinstance(q, torch.Tensor):
        return (q[:, 0] * span + q[:, 1]) * span + q[:, 2]
    return (q[:, 0] * span + q[:, 1]) * span + q[:, 2]


def block_coords(key, *, offset: int = _KEY_OFFSET, span: int = _KEY_SPAN):
    import torch

    z = key % span
    y = (key // span) % span
    x = key // (span * span)
    return torch.stack([x, y, z], 1) - offset


def occupied_tiles(bc, lo, hi, tile_blocks: int) -> list[tuple[int, int, int]]:
    """The extraction tiles that hold at least one block, in grid order.

    Extraction used to walk the full bounding grid of tiles -- the product of
    three extents -- and pay a device round trip for every tile, including the
    ones that hold nothing. A walk's surface grows with its length; its
    bounding box grows with the product of its extents, and the SfM gauge's
    axes are arbitrary, so even a straight corridor is in general a diagonal
    one. Measured: a diagonal synthetic walk's tile grid grew as frames^2.98
    while its occupied tiles grew as frames^1.0, and the real 795-keyframe field
    walk (`52ed8e0a`), whose component 0 has a few far-flung frames, had a
    1007 x 2288 x 1440-block box: 1,925,280 tiles of which 383 held a block.
    Its mesh stage took 520 s against 5 s of fusion, all of it proving that
    empty tiles are empty.

    This returns exactly the tiles the old loop would have found non-empty, in
    the same order, so the mesh is unchanged. The subtle part is the HALO: a
    tile reads `tile_blocks + 1` blocks per axis, so a block whose coordinate
    sits exactly on a tile boundary is also read by the previous tile, and that
    tile must be kept too -- dropping it would silently change the triangles
    along every seam.
    """
    bc = np.asarray(bc, np.int64).reshape(-1, 3)
    lo = np.asarray(lo, np.int64)
    hi = np.asarray(hi, np.int64)
    if not len(bc):
        return []
    kmax = (hi - lo) // tile_blocks
    q = bc - lo
    f = np.floor_divide(q, tile_blocks)
    on_edge = (q - f * tile_blocks) == 0
    cands = [f]
    for mask in range(1, 8):
        axes = [a for a in range(3) if mask >> a & 1]
        sel = np.all(on_edge[:, axes], axis=1)
        if sel.any():
            g = f[sel].copy()
            g[:, axes] -= 1
            cands.append(g)
    k = np.concatenate(cands)
    k = k[np.all((k >= 0) & (k <= kmax), axis=1)]
    if not len(k):
        return []
    k = np.unique(k, axis=0)                  # lexicographic: the old x, y, z loop order
    origins = lo + k * tile_blocks
    return [tuple(int(v) for v in o) for o in origins]


# ---------------------------------------------------------------------------
# the field
# ---------------------------------------------------------------------------


class SurfaceVolume:
    """A sparse TSDF over int64-keyed 8^3 blocks, held on one device.

    Two allocation modes, because the offline and live paths want different
    things. `reserve` takes the whole block set at once and is what the
    offline path uses: growing the table per frame means concatenating and
    re-sorting it on every keyframe, which is quadratic in blocks and produced
    real allocator pressure at 229k blocks. `grow` adds to an existing table
    and is what the live path uses, where the block set is not known in
    advance.
    """

    def __init__(self, voxel: float, trunc: float, device=None, *,
                 trunc_rel: float = 0.0, trunc_max: float = 0.0):
        """`trunc` is the truncation FLOOR, in scene units. `trunc_rel`, when
        positive, makes the band depth-proportional: a sample measured at
        depth d is truncated at max(trunc, trunc_rel * d). See
        `SurfaceParams.trunc_depth_proportional` for why."""
        import torch

        if not (voxel > 0 and trunc > 0):
            raise SurfaceUnavailable("voxel and truncation must be positive")
        self.voxel = float(voxel)
        self.trunc = float(trunc)
        self.trunc_rel = max(0.0, float(trunc_rel))
        self.trunc_max = max(0.0, float(trunc_max))
        self.dev = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.keys = torch.zeros(0, dtype=torch.int64, device=self.dev)
        self.tsdf = torch.zeros((0, BLOCK_VOXELS), dtype=torch.float32, device=self.dev)
        self.w = torch.zeros((0, BLOCK_VOXELS), dtype=torch.float32, device=self.dev)
        self.rgb = torch.zeros((0, BLOCK_VOXELS, 3), dtype=torch.float32, device=self.dev)

    # -- size ---------------------------------------------------------------

    @property
    def n_blocks(self) -> int:
        return int(self.keys.numel())

    def bytes_used(self) -> int:
        return self.n_blocks * BLOCK_VOXELS * (4 + 4 + 12)

    def release_field(self) -> None:
        """Drop the field's storage, keeping geometry parameters (voxel,
        truncation) usable. For callers that are done extracting."""
        import torch

        self.tsdf = torch.zeros((0, BLOCK_VOXELS), dtype=torch.float32, device=self.dev)
        self.w = torch.zeros((0, BLOCK_VOXELS), dtype=torch.float32, device=self.dev)
        self.rgb = torch.zeros((0, BLOCK_VOXELS, 3), dtype=torch.float32, device=self.dev)

    def trunc_at(self, depth):
        """The truncation of a sample measured at `depth` (tensor or float)."""
        cap = self.trunc_max if self.trunc_max > 0 else float("inf")
        if hasattr(depth, "clamp"):
            if self.trunc_rel <= 0:
                return torch_full_like(depth, self.trunc)
            return (depth * self.trunc_rel).clamp(min=self.trunc, max=max(cap, self.trunc))
        return max(self.trunc, min(cap, float(depth) * self.trunc_rel))

    # -- allocation ---------------------------------------------------------

    def reserve(self, keys) -> None:
        """Allocate exactly this block set, once."""
        import torch

        keys = torch.unique(keys.to(self.dev))
        n = int(keys.numel())
        self.keys = keys
        self.tsdf = torch.ones((n, BLOCK_VOXELS), dtype=torch.float32, device=self.dev)
        self.w = torch.zeros((n, BLOCK_VOXELS), dtype=torch.float32, device=self.dev)
        self.rgb = torch.zeros((n, BLOCK_VOXELS, 3), dtype=torch.float32, device=self.dev)

    def grow(self, keys) -> int:
        """Add blocks, keeping what is already there. Returns how many were new."""
        import torch

        keys = torch.unique(keys.to(self.dev))
        if self.keys.numel():
            pos = torch.searchsorted(self.keys, keys).clamp(max=self.keys.numel() - 1)
            keys = keys[self.keys[pos] != keys]
        if keys.numel() == 0:
            return 0
        n_new = int(keys.numel())
        allk = torch.cat([self.keys, keys])
        order = torch.argsort(allk)
        self.tsdf = torch.cat(
            [self.tsdf, torch.ones((n_new, BLOCK_VOXELS), device=self.dev)])[order]
        self.w = torch.cat(
            [self.w, torch.zeros((n_new, BLOCK_VOXELS), device=self.dev)])[order]
        self.rgb = torch.cat(
            [self.rgb, torch.zeros((n_new, BLOCK_VOXELS, 3), device=self.dev)])[order]
        self.keys = allk[order]
        return n_new

    def lookup(self, keys):
        """Block index for each key, or -1."""
        import torch

        if self.keys.numel() == 0:
            return torch.full_like(keys, -1)
        pos = torch.searchsorted(self.keys, keys).clamp(max=self.keys.numel() - 1)
        return torch.where(self.keys[pos] == keys, pos, torch.full_like(pos, -1))

    def voxel_world(self, bidx):
        """World centres of every voxel of the given blocks -> (n, BLOCK_VOXELS, 3)."""
        import torch

        bc = block_coords(self.keys[bidx]).to(torch.float32)
        r = torch.arange(BLOCK, device=self.dev, dtype=torch.float32)
        gz, gy, gx = torch.meshgrid(r, r, r, indexing="ij")
        off = torch.stack([gx, gy, gz], -1).reshape(1, BLOCK_VOXELS, 3)
        return (bc.unsqueeze(1) * BLOCK + off + 0.5) * self.voxel

    def blocks_for_depth(self, depth, valid, R, t, K):
        """The block keys one posed depth image would touch, surface plus the
        truncation shell around it."""
        import torch

        dev = self.dev
        vy, vx = torch.nonzero(valid, as_tuple=True)
        if vy.numel() == 0:
            return torch.zeros(0, dtype=torch.int64, device=dev)
        # Subsampling is safe: a block is eight voxels across, so neighbouring
        # pixels land in the same block many times over.
        step = max(1, vy.numel() // 60000)
        vy, vx = vy[::step], vx[::step]
        z = depth[vy, vx]
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        x = (vx.to(torch.float32) - cx) / fx * z
        y = (vy.to(torch.float32) - cy) / fy * z
        Xc = torch.stack([x, y, z], 1)
        Xw = (Xc - t) @ R                       # R^T (Xc - t)

        bc = torch.floor(Xw / (self.voxel * BLOCK)).to(torch.int64)
        # The shell radius follows each pixel's own truncation, so a far pixel
        # with a wide band does not widen the shell of every near one.
        rads = (torch.ceil(self.trunc_at(z) / (self.voxel * BLOCK)).to(torch.int64)
                + SHELL_MARGIN_BLOCKS)
        # The shell a pixel touches depends only on the block the pixel lands
        # in, so collapse pixels to blocks BEFORE expanding the shell, and
        # expand in key space rather than coordinate space. A key is affine in
        # its block coordinate -- key(c + o) == key(c) + key(o) - key(0) -- so
        # the shell of every block is its key plus one constant per offset,
        # and deduplication is a 1-D sort instead of `unique(dim=0)`'s
        # lexicographic row sort over (pixels x shell x 3) int64.
        #
        # The row form was the dominant cost of the whole surface stage and it
        # grew with the CUBE of truncation/voxel: at a shell radius of 4 it
        # built a 43.7M-row table per keyframe, peaked at 3 GiB per call, took
        # 530 s of an 873 s build, and once died inside the sort with
        # `cudaErrorIllegalAddress`. Measured paired on real frames: 57x
        # faster at r=3, 119x at r=4, 1.6-3.0 GiB -> 27-53 MiB per call
        # (`Glasses-scratch/wb-final-recon/scaling/SCALING.md`). The block set
        # is identical, key for key and in the same order.
        #
        # The range guard stays exact: the shell's EXTREMES, not only its
        # centres, must be keyable, which is what the row form checked.
        out = []
        for r in torch.unique(rads).tolist():
            bcr = bc[rads == r]
            block_key(torch.stack([bcr.min(0).values - r, bcr.max(0).values + r]))
            centre = torch.unique(block_key(bcr))
            offs = torch.arange(-r, r + 1, device=dev)
            oz, oy, ox = torch.meshgrid(offs, offs, offs, indexing="ij")
            koff = (ox.reshape(-1) * _KEY_SPAN + oy.reshape(-1)) * _KEY_SPAN + oz.reshape(-1)
            out.append(torch.unique((centre.unsqueeze(1) + koff.unsqueeze(0)).reshape(-1)))
        # `.clone()` is load-bearing. On CUDA, `torch.unique` returns a view
        # narrowed out of its full-length sort buffer, so the returned keys
        # keep (centres x shell) int64 alive -- and the offline build holds
        # every frame's keys until `reserve`. Without it the peak allocation
        # of a 346-frame build rose by 823 MiB.
        return torch.unique(torch.cat(out)).clone()

    # -- integration --------------------------------------------------------

    def integrate(self, depth, valid, rgb_img, R, t, K, *, params: SurfaceParams,
                  weight_img=None, chunk_blocks: int = 4096):
        """Fold one posed depth image into the field.

        `R`, `t` are world-to-camera (`x_cam = R @ X + t`), which is the
        convention `solve/<sid>/solution.json` uses -- verified by reprojection
        at 0.700 px median, against 528 px for the inverse. The derived tree's
        `poses.json` is a different frame AND a different convention; do not
        mix them.
        """
        import torch

        if self.n_blocks == 0:
            return
        H, W = depth.shape
        R = R.to(self.dev).to(torch.float32)
        t = t.to(self.dev).to(torch.float32)
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])

        bidx = self._blocks_in_view(R, t, fx, fy, cx, cy, H, W, depth)
        if bidx.numel() == 0:
            return
        # ONCE per frame, not once per chunk of blocks. The depth falloff term
        # needs a reference depth for this image; computing it inside
        # `_integrate_blocks` meant a median over two million sampled values
        # for every 4096-block chunk -- about fifty medians per frame, twenty
        # thousand over a walk, each with a device sync -- and it was the whole
        # cost of the fuse stage.
        med = 1.0
        if params.depth_falloff:
            good = depth[valid]
            if good.numel():
                med = float(torch.median(good))
        for s in range(0, int(bidx.numel()), chunk_blocks):
            self._integrate_blocks(bidx[s:s + chunk_blocks], depth, valid, rgb_img,
                                   R, t, fx, fy, cx, cy, H, W, params, weight_img,
                                   med)

    def _blocks_in_view(self, R, t, fx, fy, cx, cy, H, W, depth):
        import torch

        bc = block_coords(self.keys).to(torch.float32)
        ctr = (bc * BLOCK + BLOCK / 2) * self.voxel
        zmax = torch.nan_to_num(depth, nan=0.0).max()
        tmax = self.trunc_at(float(zmax))
        rad = (BLOCK * self.voxel) * 0.8660254 + tmax
        pc = ctr @ R.T + t
        z = pc[:, 2]
        zc = z.clamp(min=1e-4)
        u = pc[:, 0] / zc * fx + cx
        v = pc[:, 1] / zc * fy + cy
        mu, mv = rad / zc * fx, rad / zc * fy
        return torch.nonzero(
            (z > -rad) & (u > -mu) & (u < W + mu) & (v > -mv) & (v < H + mv)
            & (z < zmax + rad + tmax), as_tuple=False).squeeze(1)

    def _integrate_blocks(self, sel, depth, valid, rgb_img, R, t,
                          fx, fy, cx, cy, H, W, params, weight_img, med):
        import torch

        Xw = self.voxel_world(sel)
        n = Xw.shape[0]
        pc = Xw.reshape(-1, 3) @ R.T + t
        z = pc[:, 2]
        zc = z.clamp(min=1e-4)
        ui = (pc[:, 0] / zc * fx + cx).round().to(torch.int64)
        vi = (pc[:, 1] / zc * fy + cy).round().to(torch.int64)
        ok = (z > 1e-4) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        if not bool(ok.any()):
            return

        flat = vi.clamp(0, H - 1) * W + ui.clamp(0, W - 1)
        d = depth.reshape(-1)[flat]
        ok &= valid.reshape(-1)[flat] & torch.isfinite(d) & (d > 1e-4)

        sdf = d - z
        tr = self.trunc_at(d)
        surface = ok & (sdf >= -tr) & (sdf <= tr)
        if params.carve:
            free = ok & (sdf > tr) & (
                sdf < tr + params.max_carve_voxels * self.voxel)
        else:
            free = torch.zeros_like(surface)

        wpix = (torch.ones_like(z) if weight_img is None
                else weight_img.reshape(-1)[flat])
        if params.depth_falloff:
            wpix = wpix * ((med * med) / (z * z).clamp(min=1e-6)).clamp(
                max=params.max_near_boost)

        col = rgb_img.reshape(-1, 3)[flat]
        val = (sdf / tr).clamp(-1.0, 1.0)

        idx_b = torch.arange(n, device=self.dev).repeat_interleave(BLOCK_VOXELS)
        idx_v = torch.arange(BLOCK_VOXELS, device=self.dev).repeat(n)
        blk = sel[idx_b]

        self._blend(blk, idx_v, surface, val, wpix, col)
        if params.carve:
            self._blend(blk, idx_v, free, torch.ones_like(val),
                        wpix * params.carve_weight, None)

    def _blend(self, blk, idx_v, mask, value, weight, colour):
        if not bool(mask.any()):
            return
        bb, vv = blk[mask], idx_v[mask]
        wo = self.w[bb, vv]
        wn = weight[mask]
        tot = (wo + wn).clamp(min=1e-9)
        self.tsdf[bb, vv] = (self.tsdf[bb, vv] * wo + value[mask] * wn) / tot
        if colour is not None:
            self.rgb[bb, vv] = (
                self.rgb[bb, vv] * wo.unsqueeze(1)
                + colour[mask] * wn.unsqueeze(1)) / tot.unsqueeze(1)
        self.w[bb, vv] = wo + wn

    # -- extraction ---------------------------------------------------------

    def extract_mesh(self, min_weight: float, *, tile_blocks: int = 12,
                     only_blocks=None, progress=None, tag=None):
        """Marching cubes over the observed part of the field only.

        A cube emits triangles only when ALL EIGHT of its corner voxels carry
        at least `min_weight` of evidence, so both ends of every edge a vertex
        is interpolated on were measured. Cells nobody measured keep weight
        zero, and the surface stops at the edge of what was seen instead of
        closing over it.

        This used to test only the voxel NEAREST each vertex. An unobserved
        voxel holds the allocation default +1, which reads as measured free
        space, so a cube with one observed corner behind a surface and one
        never-observed corner beyond the truncation band emitted a triangle
        between them. On clean synthetic input that built a complete phantom
        second wall 7 voxels behind a real one; on the canonical capture it
        was 8,513 vertices whose far edge endpoint had weight exactly 0.

        `only_blocks` restricts extraction to a block set, which is how the
        live path re-meshes just what changed.

        `tag`, a (n_blocks, 512) bool tensor such as the one `fill_enclosed`
        returns, is sampled at every emitted vertex; when it is given the
        return is (V, F, C, G) with G the per-vertex flag, otherwise (V, F, C).
        """
        import torch
        from skimage import measure

        empty = (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64),
                 np.zeros((0, 3), np.uint8))
        if tag is not None:
            empty = empty + (np.zeros(0, bool),)
        if self.n_blocks == 0:
            return empty

        bc_all = block_coords(self.keys).cpu().numpy()
        if only_blocks is not None and len(only_blocks):
            bc_sel = block_coords(only_blocks.to(self.dev)).cpu().numpy()
        else:
            bc_sel = bc_all
        if not len(bc_sel):
            return empty
        lo, hi = bc_sel.min(0), bc_sel.max(0)

        Vs, Fs, Cs, Gs, nv = [], [], [], [], 0
        tiles = occupied_tiles(bc_all, lo, hi, tile_blocks)
        nb = tile_blocks + 1                       # one block of halo
        n = nb * BLOCK
        r = torch.arange(nb, device=self.dev, dtype=torch.int64)
        si, sj, sk = torch.meshgrid(r, r, r, indexing="ij")
        slot_off = torch.stack([si.reshape(-1), sj.reshape(-1), sk.reshape(-1)], 1)

        # Scatter a block's 512 voxels into the tile with one indexed write
        # rather than a Python-level slice assignment per block. At 229k
        # blocks over a thousand tiles the loop version was millions of GPU
        # round trips, and it -- not the marching cubes -- was the whole cost
        # of this stage.
        #
        # A voxel's flat index within a block is z*64 + y*8 + x, because
        # `voxel_world` builds its offsets with meshgrid(z, y, x); the
        # destination array is indexed [x][y][z]. Getting that transposition
        # wrong does not crash, it silently reflects every block.
        v = torch.arange(BLOCK_VOXELS, device=self.dev, dtype=torch.int64)
        vx, vy, vz = v % BLOCK, (v // BLOCK) % BLOCK, v // (BLOCK * BLOCK)
        voxel_off = vx * n * n + vy * n + vz
        slot_base = (slot_off[:, 0] * n * n + slot_off[:, 1] * n
                     + slot_off[:, 2]) * BLOCK

        for ti, (bx, by, bz) in enumerate(tiles):
            origin = torch.tensor([bx, by, bz], dtype=torch.int64, device=self.dev)
            idx = self.lookup(block_key(slot_off + origin))
            present = torch.nonzero(idx >= 0, as_tuple=False).squeeze(1)
            if present.numel() == 0:
                continue
            bidx = idx[present]
            dest = (slot_base[present].unsqueeze(1)
                    + voxel_off.unsqueeze(0)).reshape(-1)

            dt = torch.ones(n * n * n, dtype=torch.float32, device=self.dev)
            dt[dest] = self.tsdf[bidx].reshape(-1)
            a = dt.reshape(n, n, n).cpu().numpy()
            if not (a.min() < 0 < a.max()):
                continue
            try:
                verts, faces, _, _ = measure.marching_cubes(a, level=0.0)
            except (ValueError, RuntimeError):
                continue
            if not len(faces):
                continue

            # Colour is only wanted AT the vertices, so it is sampled on the
            # device and only the sampled values cross the bus.
            vi = np.clip(np.round(verts).astype(np.int64), 0, n - 1)
            lin = torch.as_tensor(vi[:, 0] * n * n + vi[:, 1] * n + vi[:, 2],
                                  device=self.dev)
            dw = torch.zeros(n * n * n, dtype=torch.float32, device=self.dev)
            dw[dest] = self.w[bidx].reshape(-1)
            # The cube a face was generated in: every vertex lies on one of
            # its edges, so the floor of the centroid names it.
            observed = (dw >= min_weight).reshape(1, 1, n, n, n)
            cube_ok = (-torch.nn.functional.max_pool3d(
                -observed.to(torch.float32), 2, 1)).reshape(n - 1, n - 1, n - 1) > 0.5
            ci = np.clip(np.floor(verts[faces].mean(axis=1)).astype(np.int64), 0, n - 2)
            m = n - 1
            clin = torch.as_tensor(ci[:, 0] * m * m + ci[:, 1] * m + ci[:, 2],
                                   device=self.dev)
            keep = cube_ok.reshape(-1)[clin].cpu().numpy()
            faces = faces[keep]
            if not len(faces):
                continue
            used = np.unique(faces)
            remap = np.full(len(verts), -1, np.int64)
            remap[used] = np.arange(len(used))

            dc = torch.zeros((n * n * n, 3), dtype=torch.float32, device=self.dev)
            dc[dest] = self.rgb[bidx].reshape(-1, 3)
            used_t = torch.as_tensor(used, device=self.dev)
            cq = dc[lin[used_t]].cpu().numpy()
            if tag is not None:
                dg = torch.zeros(n * n * n, dtype=torch.bool, device=self.dev)
                dg[dest] = tag[bidx].reshape(-1).to(self.dev)
                Gs.append(dg[lin[used_t]].cpu().numpy())

            world = (verts[used] + np.array([bx, by, bz]) * BLOCK + 0.5) * self.voxel
            Vs.append(world.astype(np.float32))
            Fs.append(remap[faces] + nv)
            Cs.append(np.clip(cq, 0, 255).astype(np.uint8))
            nv += len(used)
            if progress is not None and ti % 50 == 0:
                progress(STAGE_MESH, ti, len(tiles))

        if not Vs:
            return empty
        out = (np.concatenate(Vs), np.concatenate(Fs), np.concatenate(Cs))
        if tag is not None:
            out = out + (np.concatenate(Gs),)
        return out


# ---------------------------------------------------------------------------
# per-frame preparation
# ---------------------------------------------------------------------------


def depth_bound(params: SurfaceParams, scene_scale: float,
                z_sparse_max: float | None) -> float:
    """The far bound for one frame: its farthest fitted anchor times
    `anchor_depth_multiple`, or the generous scene-relative fallback when the
    fit records no anchor range."""
    if z_sparse_max is not None and z_sparse_max > 0 and params.anchor_depth_multiple > 0:
        return float(z_sparse_max) * params.anchor_depth_multiple
    return float(scene_scale) * params.max_depth_frac


def depth_validity(depth, K, params: SurfaceParams, median_depth: float,
                   max_depth: float | None = None):
    """Per-pixel incidence cosine and validity mask, on whatever device the
    depth is on.

    The same three rejections the point pipeline used -- discontinuity,
    grazing incidence, absurd depth -- but the incidence cosine is returned
    rather than discarded, so the caller can weight by it instead of treating
    a 70-degree surface the same as a head-on one right up to the cut.

    `max_depth` is the frame's own far bound (`depth_bound`); without it the
    scene-relative fallback applies.
    """
    import torch

    dev = depth.device
    H, W = depth.shape
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    uu = torch.arange(W, device=dev, dtype=torch.float32).view(1, W).expand(H, W)
    vv = torch.arange(H, device=dev, dtype=torch.float32).view(H, 1).expand(H, W)
    xn, yn = (uu - cx) / fx, (vv - cy) / fy
    Px, Py = xn * depth, yn * depth

    def grad(a, axis):
        g = torch.zeros_like(a)
        if axis == 1:
            g[:, 1:-1] = (a[:, 2:] - a[:, :-2]) * 0.5
            g[:, 0], g[:, -1] = a[:, 1] - a[:, 0], a[:, -1] - a[:, -2]
        else:
            g[1:-1] = (a[2:] - a[:-2]) * 0.5
            g[0], g[-1] = a[1] - a[0], a[-1] - a[-2]
        return g

    zx, zy = grad(depth, 1), grad(depth, 0)
    edge = (zx.abs() + zy.abs()) / depth.clamp(min=1e-6)
    t1 = torch.stack([grad(Px, 1), grad(Py, 1), zx], -1)
    t2 = torch.stack([grad(Px, 0), grad(Py, 0), zy], -1)
    nrm = torch.cross(t1, t2, dim=-1)
    ray = torch.stack([xn, yn, torch.ones_like(xn)], -1)
    ray = ray / ray.norm(dim=-1, keepdim=True)
    cosang = ((nrm * ray).sum(-1) / nrm.norm(dim=-1).clamp(min=1e-12)).abs()

    ok = torch.isfinite(depth) & (depth > 1e-3)
    ok &= depth < (max_depth if max_depth is not None
                   else median_depth * params.max_depth_frac)
    ok &= torch.isfinite(edge) & (edge < params.edge_rel)
    ok &= torch.isfinite(cosang) & (cosang > math.cos(math.radians(params.max_grazing_deg)))
    # A depth edge's neighbour is not trustworthy either.
    ok = torch.nn.functional.max_pool2d(
        (~ok).to(torch.float32)[None, None], 3, 1, 1).squeeze() < 0.5
    return ok, cosang.clamp(0, 1)


def truncation_rel(params: SurfaceParams, median_held_out_rel: float | None) -> float:
    """The depth-proportional part of the truncation: a sample at depth d is
    truncated at max(floor, this * d). 0 when the error was not measured or
    the band is configured absolute."""
    if (not params.trunc_depth_proportional or not median_held_out_rel
            or params.trunc_error_multiple <= 0):
        return 0.0
    return params.trunc_error_multiple * float(median_held_out_rel)


def truncation_for(params: SurfaceParams, voxel: float, median_depth: float,
                   median_held_out_rel: float | None) -> float:
    """Truncation from the voxel floor and the frames' measured disagreement,
    evaluated AT `median_depth`. With `trunc_depth_proportional` a sample at
    the scene scale gets exactly this and the absolute floor is
    `trunc_voxels * voxel`; it stays in the manifest for comparison."""
    trunc = params.trunc_voxels * voxel
    if median_held_out_rel and params.trunc_error_multiple > 0:
        need = params.trunc_error_multiple * float(median_held_out_rel) * median_depth
        trunc = max(trunc, need)
    return trunc


# ---------------------------------------------------------------------------
# enclosed-hole fill (opt-in)
# ---------------------------------------------------------------------------
#
# Why in the field and not on the mesh. A mesh hole-filler sees a loop of
# boundary edges and cannot tell "a few pixels failed the validity mask here,
# but frames on every side measured this surface" from "this is the edge of
# what anyone looked at". The field can: a voxel no frame gave evidence for,
# surrounded by voxels frames did give evidence for, is a gap the capture
# bracketed. A voxel with evidence on one side only is the edge of knowledge.
#
# Three guards, each found by measurement rather than assumed:
#
#   * "Bracketed" is decided against ORIGINAL evidence only. A voxel filled by
#     an earlier tile never counts as observed for a later one; otherwise the
#     fill feeds itself outward across tile borders.
#   * A gap is only closed cleanly when the march radius is at least the gap's
#     WIDTH, not half of it: a voxel in the corner of a gap must see across the
#     whole gap along the diagonal. Between one and two radii the 26-direction
#     test fills the middle and leaves the corners, and marching cubes turns
#     the boundary between a filled voxel and an unfilled one into a sliver
#     wall that no camera saw. `SurfaceParams.fill_radius_voxels` therefore
#     sets radius = gap / voxel.
#   * After extraction, a filled patch that still owns an open boundary edge
#     did not close anything -- it grew a frontier or left a sliver -- and is
#     REVERTED IN THE FIELD, then re-extracted, rather than cut off the mesh.
#     Cutting it off the mesh also removes the ring of measured triangles
#     whose vertices happened to round into a filled voxel, so a rejected fill
#     would leave the hole bigger than it found it.

_DIRS26 = tuple((i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1)
                for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0))


def _shift(t, s: int, ax: int):
    """`torch.roll` without the wrap-around: what enters from outside is zero.

    A tile is a window onto a larger field. Rolling would let the far edge of
    the window stand in for the near side's neighbour -- evidence from
    somewhere else entirely -- so every shift here is a padded one.
    """
    import torch

    if s == 0:
        return t
    n = t.shape[ax]
    out = torch.zeros_like(t)
    if abs(s) >= n:
        return out
    if s > 0:
        out.narrow(ax, s, n - s).copy_(t.narrow(ax, 0, n - s))
    else:
        out.narrow(ax, 0, n + s).copy_(t.narrow(ax, -s, n + s))
    return out


def _enclosure_hits(obs, radius: int):
    """Per voxel: how many of the 26 directions reach `obs` within `radius`."""
    import torch

    hits = torch.zeros(obs.shape, dtype=torch.uint8, device=obs.device)
    for d in _DIRS26:
        acc = torch.zeros_like(obs)
        cur = obs
        for _ in range(radius):
            for ax in (0, 1, 2):
                cur = _shift(cur, -d[ax], ax)
            acc |= cur
        hits += acc.to(torch.uint8)
    return hits


def _diffuse(val, col, known, fill, iters: int):
    """Jacobi Laplace solve inside `fill`, `known` held fixed as Dirichlet data.

    The surrounding zero crossing is continued across the gap and nothing
    else is asserted; a linear field (a plane) is reproduced exactly. Colour
    rides along so a closed hole takes the colour of the surface it continues
    rather than black. Filled voxels start from 0 (no opinion), not from their
    stored value, which for an unobserved voxel is the allocation default of
    +1 and drags the recovered crossing half a voxel toward the camera.
    """
    import torch

    val = torch.where(fill, torch.zeros_like(val), val)
    src = (known | fill).to(torch.float32)
    fill3 = fill.unsqueeze(-1)
    for _ in range(iters):
        acc = torch.zeros_like(val)
        cacc = torch.zeros_like(col)
        cnt = torch.zeros_like(val)
        for ax in (0, 1, 2):
            for s in (1, -1):
                ss = _shift(src, s, ax)
                acc += _shift(val, s, ax) * ss
                cacc += _shift(col, s, ax) * ss.unsqueeze(-1)
                cnt += ss
        safe = cnt.clamp(min=1e-6)
        val = torch.where(fill, acc / safe, val)
        col = torch.where(fill3, cacc / safe.unsqueeze(-1), col)
    return val, col


@dataclass
class EnclosedFill:
    """What `fill_enclosed` wrote, and what was there before it.

    `tag` marks every filled voxel, (n_blocks, 512). The backup rows hold the
    ORIGINAL tsdf / weight / colour of every block the fill touched, sorted by
    block index, so any part of the fill can be taken back out exactly.
    """

    tag: object
    blocks: object
    tsdf: object
    w: object
    rgb: object

    @property
    def voxels(self) -> int:
        return int(self.tag.sum())


def fill_enclosed(vol: SurfaceVolume, min_weight: float, radius: int,
                  need_dirs: int = 22, *, iters: int | None = None,
                  tile_blocks: int = 10, progress=None) -> EnclosedFill:
    """Fill unobserved voxels the capture bracketed. Mutates `vol`.

    A filled voxel gets weight exactly `min_weight` -- the least evidence that
    still emits -- so it can never outvote a measured one, and a measured
    voxel is never modified.
    """
    import torch

    dev = vol.dev
    tag = torch.zeros((vol.n_blocks, BLOCK_VOXELS), dtype=torch.bool, device=dev)
    empty_rows = (torch.zeros(0, dtype=torch.int64, device=dev),
                  torch.zeros((0, BLOCK_VOXELS), device=dev),
                  torch.zeros((0, BLOCK_VOXELS), device=dev),
                  torch.zeros((0, BLOCK_VOXELS, 3), device=dev))
    if vol.n_blocks == 0 or radius <= 0:
        return EnclosedFill(tag, *empty_rows)
    # Jacobi converges in roughly (gap width)^2 sweeps; a closable gap is at
    # most `radius` voxels wide, so this is a margin, not a guess.
    iters = 2 * (radius + 2) ** 2 if iters is None else int(iters)
    # Shifts are padded, not wrapped, so the halo only has to give the march
    # its full reach from any core voxel.
    halo = int(math.ceil(radius / BLOCK)) + 1
    nb = tile_blocks + 2 * halo
    n = nb * BLOCK

    bc_all = block_coords(vol.keys).cpu().numpy()
    lo, hi = bc_all.min(0), bc_all.max(0)
    tiles = [(x, y, z)
             for x in range(int(lo[0]), int(hi[0]) + 1, tile_blocks)
             for y in range(int(lo[1]), int(hi[1]) + 1, tile_blocks)
             for z in range(int(lo[2]), int(hi[2]) + 1, tile_blocks)]

    r = torch.arange(nb, device=dev, dtype=torch.int64)
    si, sj, sk = torch.meshgrid(r, r, r, indexing="ij")
    slot_off = torch.stack([si.reshape(-1), sj.reshape(-1), sk.reshape(-1)], 1)
    core_slot = ((slot_off >= halo) & (slot_off < halo + tile_blocks)).all(1)
    v = torch.arange(BLOCK_VOXELS, device=dev, dtype=torch.int64)
    vx, vy, vz = v % BLOCK, (v // BLOCK) % BLOCK, v // (BLOCK * BLOCK)
    voxel_off = vx * n * n + vy * n + vz            # block z*64+y*8+x -> [x][y][z]
    slot_base = (slot_off[:, 0] * n * n + slot_off[:, 1] * n + slot_off[:, 2]) * BLOCK

    backups = []
    for ti, (bx, by, bz) in enumerate(tiles):
        origin = torch.tensor([bx - halo, by - halo, bz - halo],
                              dtype=torch.int64, device=dev)
        idx = vol.lookup(block_key(slot_off + origin))
        present = torch.nonzero(idx >= 0, as_tuple=False).squeeze(1)
        if present.numel() == 0 or not bool(core_slot[present].any()):
            continue
        bidx = idx[present]
        dest = (slot_base[present].unsqueeze(1) + voxel_off.unsqueeze(0)).reshape(-1)

        dw = torch.zeros(n * n * n, dtype=torch.float32, device=dev)
        dw[dest] = vol.w[bidx].reshape(-1)
        dg = torch.zeros(n * n * n, dtype=torch.bool, device=dev)
        dg[dest] = tag[bidx].reshape(-1)
        obs = ((dw >= min_weight) & ~dg).reshape(n, n, n)   # ORIGINAL evidence
        if not bool(obs.any()):
            continue
        alloc = torch.zeros(n * n * n, dtype=torch.bool, device=dev)
        alloc[dest] = True
        fill = ((~obs) & alloc.reshape(n, n, n)
                & (_enclosure_hits(obs, radius) >= need_dirs))
        core_present = core_slot[present]
        cdest = dest.reshape(-1, BLOCK_VOXELS)[core_present].reshape(-1)
        f = fill.reshape(-1)[cdest]
        if not bool(f.any()):
            continue

        dt = torch.ones(n * n * n, dtype=torch.float32, device=dev)
        dt[dest] = vol.tsdf[bidx].reshape(-1)
        dc = torch.zeros((n * n * n, 3), dtype=torch.float32, device=dev)
        dc[dest] = vol.rgb[bidx].reshape(-1, 3)
        nv, nc = _diffuse(dt.reshape(n, n, n), dc.reshape(n, n, n, 3), obs, fill, iters)

        cb = bidx[core_present]
        fb = f.reshape(-1, BLOCK_VOXELS)
        cbt = cb[fb.any(1)]
        backups.append((cbt, vol.tsdf[cbt].clone(), vol.w[cbt].clone(),
                        vol.rgb[cbt].clone()))
        vol.tsdf[cb] = torch.where(f, nv.reshape(-1)[cdest],
                                   vol.tsdf[cb].reshape(-1)).reshape(-1, BLOCK_VOXELS)
        vol.w[cb] = torch.where(f, torch.full(f.shape, float(min_weight), device=dev),
                                vol.w[cb].reshape(-1)).reshape(-1, BLOCK_VOXELS)
        vol.rgb[cb] = torch.where(f.unsqueeze(1), nc.reshape(-1, 3)[cdest],
                                  vol.rgb[cb].reshape(-1, 3)).reshape(-1, BLOCK_VOXELS, 3)
        tag[cb] |= fb
        if progress is not None and ti % 50 == 0:
            progress(STAGE_MESH, ti, len(tiles))

    if not backups:
        return EnclosedFill(tag, *empty_rows)
    blocks = torch.cat([b[0] for b in backups])
    order = torch.argsort(blocks)
    return EnclosedFill(tag, blocks[order],
                        torch.cat([b[1] for b in backups])[order],
                        torch.cat([b[2] for b in backups])[order],
                        torch.cat([b[3] for b in backups])[order])


def revert_fill_near(vol: SurfaceVolume, fill: EnclosedFill, points,
                     reach_voxels: int = 2) -> int:
    """Restore the original field for filled voxels near `points`.

    Returns how many voxels were reverted. Their tag is cleared, so a later
    extraction no longer counts surface there as filled -- because there is
    none.
    """
    import torch

    dev = vol.dev
    P = torch.as_tensor(np.asarray(points, np.float32).reshape(-1, 3), device=dev)
    if P.numel() == 0 or fill.blocks.numel() == 0:
        return 0
    base = torch.unique(torch.floor(P / vol.voxel).to(torch.int64), dim=0)
    r = torch.arange(-reach_voxels, reach_voxels + 1, device=dev)
    ox, oy, oz = torch.meshgrid(r, r, r, indexing="ij")
    off = torch.stack([ox.reshape(-1), oy.reshape(-1), oz.reshape(-1)], 1)
    reverted = 0
    for s in range(0, base.shape[0], 20000):
        vox = torch.unique((base[s:s + 20000].unsqueeze(1) + off.unsqueeze(0))
                           .reshape(-1, 3), dim=0)
        bc = torch.div(vox, BLOCK, rounding_mode="floor")
        loc = vox - bc * BLOCK
        flat = loc[:, 2] * BLOCK * BLOCK + loc[:, 1] * BLOCK + loc[:, 0]
        bidx = vol.lookup(block_key(bc))
        ok = bidx >= 0
        bidx, flat = bidx[ok], flat[ok]
        hit = fill.tag[bidx, flat]
        bidx, flat = bidx[hit], flat[hit]
        if bidx.numel() == 0:
            continue
        row = torch.searchsorted(fill.blocks, bidx)
        vol.tsdf[bidx, flat] = fill.tsdf[row, flat]
        vol.w[bidx, flat] = fill.w[row, flat]
        vol.rgb[bidx, flat] = fill.rgb[row, flat]
        fill.tag[bidx, flat] = False
        reverted += int(bidx.numel())
    return reverted


def frontier_fill_faces(V, F, G, quantum: float):
    """Faces in filled patches that still own an open boundary edge.

    The mesh is welded first (tile seams duplicate vertices), and an edge is
    open when exactly one DISTINCT triangle uses it: a tile halo emits a
    second copy of every triangle it shares with the next tile, and a
    triangle together with its own copy is still an open rim. Returns
    (bool mask over faces, stats).
    """
    V = np.asarray(V)
    F = np.asarray(F, np.int64)
    G = np.asarray(G, bool)
    stats = {"patches": 0, "sealed_patches": 0, "frontier_patches": 0,
             "filled_faces": 0, "frontier_faces": 0}
    drop = np.zeros(len(F), bool)
    if not len(F) or not G.any():
        return drop, stats
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    q = np.round(np.asarray(V, np.float64) / max(quantum, 1e-12)).astype(np.int64)
    _, weld = np.unique(q, axis=0, return_inverse=True)
    weld = weld.reshape(-1)
    n_w = int(weld.max()) + 1
    Fw = weld[F]
    e = np.sort(np.concatenate([Fw[:, [0, 1]], Fw[:, [1, 2]], Fw[:, [2, 0]]]), axis=1)
    _, inv, cnt = np.unique(e, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    _, triangle_key = np.unique(np.sort(Fw, axis=1), axis=0, return_inverse=True)
    pair = np.unique(np.stack([inv, np.tile(triangle_key.reshape(-1), 3)], 1), axis=0)
    distinct = np.bincount(pair[:, 0], minlength=len(cnt))
    open_per_face = (distinct == 1)[inv].reshape(3, -1).sum(0)

    fmask = G[F].any(axis=1)
    fi = np.nonzero(fmask)[0]
    Ff = Fw[fi]
    ee = np.concatenate([Ff[:, [0, 1]], Ff[:, [1, 2]], Ff[:, [2, 0]]])
    g = coo_matrix((np.ones(len(ee), np.int8), (ee[:, 0], ee[:, 1])), shape=(n_w, n_w))
    _, lab = connected_components(g, directed=False)
    keys, pidx = np.unique(lab[Ff[:, 0]], return_inverse=True)
    n_open = np.bincount(pidx, weights=open_per_face[fi], minlength=len(keys))
    frontier = n_open > 0
    drop[fi[frontier[pidx]]] = True
    stats.update(patches=int(len(keys)), sealed_patches=int((~frontier).sum()),
                 frontier_patches=int(frontier.sum()),
                 filled_faces=int(fmask.sum()), frontier_faces=int(drop.sum()))
    return drop, stats


def keep_sealed_fill(V, F, C, G, quantum: float):
    """Mesh-side fallback: cut frontier patches off the mesh.

    Prefer `extract_sealed`, which reverts them in the field instead. This one
    also removes measured triangles whose vertices round into a filled voxel,
    so a rejected fill can leave a hole slightly larger than it was.
    Returns (V, F, C, G, stats).
    """
    drop, stats = frontier_fill_faces(V, F, G, quantum)
    V = np.asarray(V)
    F = np.asarray(F, np.int64)
    G = np.asarray(G, bool)
    if not drop.any():
        return V, F, C, G, stats
    F2 = F[~drop]
    used = np.unique(F2)
    remap = np.full(len(V), -1, np.int64)
    remap[used] = np.arange(len(used))
    return (V[used], remap[F2], (None if C is None else np.asarray(C)[used]),
            G[used], stats)


def extract_sealed(vol: SurfaceVolume, min_weight: float, fill: EnclosedFill, *,
                   max_rounds: int = 4, reach_voxels: int = 2,
                   progress=None, **extract_kw):
    """Extract; revert in the field every filled patch that did not seal; repeat.

    Reverting part of a filled region can expose a new filled/unfilled
    boundary, hence the rounds; they stop when every remaining filled patch is
    sealed. If `max_rounds` runs out, what is left is cut off the mesh as a
    last resort and the stats say how much.

    Returns (V, F, C, G, stats).
    """
    quantum = min(vol.voxel * 1e-3, 1e-3)
    stats = {"rounds": 0, "voxels_filled": fill.voxels, "voxels_reverted": 0,
             "mesh_side_fallback_faces": 0}
    st = {}
    for rnd in range(max_rounds + 1):
        V, F, C, G = vol.extract_mesh(min_weight, tag=fill.tag,
                                      progress=progress, **extract_kw)
        drop, st = frontier_fill_faces(V, F, G, quantum)
        stats["rounds"] = rnd
        if not drop.any():
            break
        if rnd == max_rounds:
            V, F, C, G, _ = keep_sealed_fill(V, F, C, G, quantum)
            stats["mesh_side_fallback_faces"] = int(drop.sum())
            break
        stats["voxels_reverted"] += revert_fill_near(
            vol, fill, V[np.unique(F[drop])], reach_voxels)
    G = np.asarray(G, bool)
    stats.update({"voxels_kept": fill.voxels, "filled_vertices": int(G.sum()),
                  "faces_touching_fill": int(G[F].any(axis=1).sum()) if len(F) else 0,
                  "sealed_patches": st.get("sealed_patches", 0)})
    return V, F, C, G, stats


# ---------------------------------------------------------------------------
# evidence filter
# ---------------------------------------------------------------------------


def evidence_filter(V, F, views, K, trunc_at, params: SurfaceParams, device=None,
                    *, chunk: int = 2_000_000):
    """Which faces the frames, counted one by one, actually support.

    `views` yields (depth, valid, R, t) per frame, exactly the depth and
    validity the fusion used. For every face centroid and every frame whose
    valid pixel it projects onto, the frame either SUPPORTS the face (measured
    depth within `trunc_at(depth)` of it), saw THROUGH it (measured beyond
    it), or saw something in front of it (no vote). A face is kept when

      * at least `min_support_frames` distinct frames support it, and
      * the see-through frames are fewer than `contradiction_ratio` times the
        supporting ones.

    The field cannot answer either question. Its weight is a sum, which one
    close frame can fill alone, and its free-space carve is bounded to a band
    in front of each frame's own surface, so a thing near the camera -- the
    wearer's hand over the laptop, their lap -- is never carved by the frames
    that look straight past it at the room. On the canonical capture every
    face within 0.5 units of the walked path was contradicted 2:1 and all of
    them survived fusion.

    The contradiction test is the ratio above and nothing stronger: a near
    thing measured by more than half as many frames as saw past it stays.

    Returns (keep mask over faces, stats).
    """
    import torch

    F = np.asarray(F, np.int64)
    nF = len(F)
    stats = {"faces_in": int(nF), "dropped_support": 0, "dropped_contradicted": 0,
             "frames": 0}
    if nF == 0:
        return np.ones(0, bool), stats
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    Vt = torch.as_tensor(np.asarray(V, np.float32), device=dev)
    sup = torch.zeros(nF, dtype=torch.int16, device=dev)
    thru = torch.zeros(nF, dtype=torch.int16, device=dev)
    front = torch.zeros(nF, dtype=torch.int16, device=dev)
    cents, norms = [], []
    for s0 in range(0, nF, chunk):
        Fc = torch.as_tensor(F[s0:s0 + chunk], device=dev)
        tri = Vt[Fc]
        cents.append(tri.mean(1))
        norms.append(torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1))
    P = torch.cat(cents)
    N = torch.cat(norms)
    del Vt, cents, norms, tri
    for depth, valid, R, t in views:
        depth = depth.to(dev).float()
        valid = valid.to(dev)
        R = R.to(dev).float()
        t = t.to(dev).float()
        H, W = depth.shape
        centre = -(R.T @ t)
        stats["frames"] += 1
        for s0 in range(0, nF, chunk):
            pc = P[s0:s0 + chunk] @ R.T + t
            z = pc[:, 2]
            zc = z.clamp(min=1e-6)
            u = torch.round(pc[:, 0] / zc * fx + cx).long()
            v = torch.round(pc[:, 1] / zc * fy + cy).long()
            m = (z > 1e-4) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            flat = v.clamp(0, H - 1) * W + u.clamp(0, W - 1)
            d = depth.reshape(-1)[flat]
            m &= valid.reshape(-1)[flat] & torch.isfinite(d)
            r = d - z
            tr = trunc_at(d)
            s_ = m & (r.abs() <= tr)
            sup[s0:s0 + chunk] += s_.to(torch.int16)
            facing = ((centre - P[s0:s0 + chunk]) * N[s0:s0 + chunk]).sum(1) > 0
            front[s0:s0 + chunk] += (s_ & facing).to(torch.int16)
            thru[s0:s0 + chunk] += (m & (r > tr)).to(torch.int16)
    ok_support = sup >= int(params.min_support_frames)
    keep = ok_support.clone()
    if params.contradiction_ratio > 0:
        contradicted = (thru.float() >= params.contradiction_ratio * sup.float()) & (thru > 0)
        keep &= ~contradicted
        stats["dropped_contradicted"] = int((ok_support & contradicted).sum())
    if params.drop_back_facing:
        back = (front == 0) & ok_support
        stats["dropped_back_facing"] = int((keep & back).sum())
        keep &= ~back
    stats["dropped_support"] = int((~ok_support).sum())
    stats["faces_kept"] = int(keep.sum())
    return keep.cpu().numpy(), stats


def keep_faces(V, F, C, keep):
    """The mesh restricted to `keep`, with unused vertices removed."""
    F = np.asarray(F, np.int64)[np.asarray(keep, bool)]
    used = np.unique(F)
    remap = np.full(len(V), -1, np.int64)
    remap[used] = np.arange(len(used))
    return (np.asarray(V)[used], remap[F], None if C is None else np.asarray(C)[used])


# ---------------------------------------------------------------------------
# mesh cleanup
# ---------------------------------------------------------------------------


def weld_mesh(V, F, C, quantum: float):
    """Merge the duplicate vertices and triangles the tile seams produce.

    `extract_mesh` runs marching cubes per tile with one block of halo, so
    every cube in a halo is meshed twice, by its own tile and by its
    neighbour. The copies come from identical voxel values and land on the
    same positions to within float rounding, but nothing merged them. On the
    canonical world that was 587,178 duplicate triangles -- 18.3% of the mesh
    -- and it did three quiet kinds of damage: component pruning measured
    "largest" against a single tile and deleted real surface a seam had cut
    off, smoothing let the two copies of each seam vertex drift apart, and a
    phone was sent every seam twice.

    Positions are snapped to a `quantum` grid only to decide identity; the
    surviving vertex keeps its own float position. Two triangles are the same
    triangle when they have the same three vertices in any order.
    """
    V = np.asarray(V)
    F = np.asarray(F, np.int64)
    empty_stats = {"vertices_merged": 0, "faces_duplicate": 0, "faces_degenerate": 0}
    if not len(F):
        return V, F, C, empty_stats
    key = np.round(V / quantum).astype(np.int64)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    F2 = inverse[F]
    degenerate = ((F2[:, 0] == F2[:, 1]) | (F2[:, 1] == F2[:, 2])
                  | (F2[:, 0] == F2[:, 2]))
    F2 = F2[~degenerate]
    _, keep = np.unique(np.sort(F2, axis=1), axis=0, return_index=True)
    n_before = len(F2)
    F2 = F2[np.sort(keep)]
    stats = {"vertices_merged": int(len(V) - len(first)),
             "faces_duplicate": int(n_before - len(F2)),
             "faces_degenerate": int(degenerate.sum())}
    return (V[first], F2, None if C is None else np.asarray(C)[first], stats)


def drop_small_components(V, F, C, min_frac: float):
    """Remove connected components smaller than `min_frac` of the largest.

    Flying pixels and single-frame noise survive fusion as tiny islands. A
    genuinely observed but detached object is a real risk here, which is why
    the threshold is a small fraction of the LARGEST component rather than an
    absolute face count -- it scales with the scene and does not delete a
    chair because the room is big.
    """
    if not len(F) or min_frac <= 0:
        return V, F, C, {"components": 0, "dropped": 0, "faces_dropped": 0}
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(V)
    e0 = np.concatenate([F[:, 0], F[:, 1], F[:, 2]])
    e1 = np.concatenate([F[:, 1], F[:, 2], F[:, 0]])
    g = coo_matrix((np.ones(len(e0), np.int8), (e0, e1)), shape=(n, n))
    ncomp, label = connected_components(g, directed=False)
    face_label = label[F[:, 0]]
    sizes = np.bincount(face_label, minlength=ncomp)
    keep_labels = np.nonzero(sizes >= max(1, int(sizes.max() * min_frac)))[0]
    keep = np.isin(face_label, keep_labels)
    F2 = F[keep]
    used = np.unique(F2)
    remap = np.full(n, -1, np.int64)
    remap[used] = np.arange(len(used))
    stats = {"components": int(ncomp), "dropped": int(ncomp - len(keep_labels)),
             "faces_dropped": int((~keep).sum())}
    return V[used], remap[F2], (C[used] if C is not None else None), stats


def taubin_smooth(V, F, iterations: int, lam: float, mu: float):
    """Taubin lambda/mu smoothing.

    A Laplacian pass shrinks the model; Taubin follows each shrinking pass
    with a slightly larger expanding one, so the marching-cubes staircase goes
    without the whole room quietly getting smaller.
    """
    if not len(F) or iterations <= 0:
        return V, 0.0
    from scipy.sparse import coo_matrix

    n = len(V)
    e0 = np.concatenate([F[:, 0], F[:, 1], F[:, 2], F[:, 1], F[:, 2], F[:, 0]])
    e1 = np.concatenate([F[:, 1], F[:, 2], F[:, 0], F[:, 0], F[:, 1], F[:, 2]])
    A = coo_matrix((np.ones(len(e0), np.float32), (e0, e1)), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0

    P = V.astype(np.float64).copy()
    start = P.copy()
    for i in range(iterations * 2):
        step = lam if i % 2 == 0 else mu
        P = P + step * ((A @ P) / deg[:, None] - P)
    moved = float(np.median(np.linalg.norm(P - start, axis=1)))
    return P.astype(np.float32), moved


def decimate(V, F, C, target_faces: int):
    """Quadric decimation to a face budget, carrying vertex colour."""
    if target_faces <= 0 or len(F) <= target_faces:
        return V, F, C
    import open3d as o3d

    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(V, np.float64)),
        o3d.utility.Vector3iVector(np.asarray(F, np.int32)))
    if C is not None:
        m.vertex_colors = o3d.utility.Vector3dVector(np.asarray(C, np.float64) / 255.0)
    m = m.simplify_quadric_decimation(int(target_faces))
    V2 = np.asarray(m.vertices, np.float32)
    F2 = np.asarray(m.triangles, np.int64)
    C2 = (np.clip(np.asarray(m.vertex_colors) * 255.0, 0, 255).astype(np.uint8)
          if len(m.vertex_colors) else None)
    return V2, F2, C2


def vertex_normals(V, F):
    """Area-weighted vertex normals, quantised to int8 by the writer."""
    N = np.zeros_like(V, dtype=np.float32)
    if not len(F):
        return N
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    fn = np.cross(b - a, c - a)
    for i in range(3):
        np.add.at(N, F[:, i], fn)
    ln = np.linalg.norm(N, axis=1, keepdims=True)
    return (N / np.maximum(ln, 1e-12)).astype(np.float32)


# ---------------------------------------------------------------------------
# plane snap
# ---------------------------------------------------------------------------


def _vertex_adjacency(n, F):
    from scipy.sparse import coo_matrix

    e0 = np.concatenate([F[:, 0], F[:, 1], F[:, 2], F[:, 1], F[:, 2], F[:, 0]])
    e1 = np.concatenate([F[:, 1], F[:, 2], F[:, 0], F[:, 0], F[:, 1], F[:, 2]])
    A = coo_matrix((np.ones(len(e0), np.float32), (e0, e1)), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    return A


def snap_planes(V, F, views, K, params: SurfaceParams, voxel: float, median_depth: float,
                device=None, *, max_planes: int = 40, seed: int = 0):
    """Move vertices that already lie on a large, measured plane onto it.

    WHAT THIS IS. After the consistency field, frames agree on a wall to about
    2.5 voxels, and the surface still undulates inside that. For a plane that
    is large (`snap_min_area_frac` of the squared scene scale), flat to within
    the tolerance by its own fit, and measured by at least `snap_min_frames`
    keyframes' depth, each vertex already on it -- within twice the tolerance,
    with its normal within 20 degrees of the plane's -- moves along the
    plane's normal onto it: fully within the tolerance, smoothly less out to
    twice it. The normal gate reads the vertex normal smoothed over its
    neighbourhood, so neighbouring vertices are gated alike.

    WHAT THIS IS NOT. It adds no vertex and no face, closes no hole, and
    moves no vertex farther than twice the tolerance or in any direction but
    the normal: a plane never extends past what was reconstructed. Structure
    standing off the plane by more than twice the tolerance, or turned away
    from it (a shelf edge, a light switch's sides), does not move.

    `views` are the (depth, valid, R, t) the evidence filter used, i.e. the
    corrected depth. Returns (V as float32, record).
    """
    import torch
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    t0 = time.time()
    tol = float(params.snap_tol_voxels) * float(voxel)
    min_area = float(params.snap_min_area_frac) * float(median_depth) ** 2
    record = {"version": SNAP_VERSION, "tol": tol, "tol_voxels": params.snap_tol_voxels,
              "min_area": min_area, "min_frames": params.snap_min_frames,
              "planes": [], "rejected": 0, "rejected_examples": [],
              "vertices_moved": 0, "area_snapped": 0.0}
    V = np.asarray(V)
    F = np.asarray(F, np.int64)
    if not len(F) or tol <= 0:
        record["seconds"] = round(time.time() - t0, 2)
        return V.astype(np.float32), record
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nV = len(V)
    V64 = V.astype(np.float64)
    N = vertex_normals(V.astype(np.float32), F).astype(np.float64)
    A = _vertex_adjacency(nV, F)
    Ns = N.copy()
    for _ in range(3):
        Ns = A @ Ns + Ns
        Ns /= np.maximum(np.linalg.norm(Ns, axis=1, keepdims=True), 1e-12)
    fa = 0.5 * np.linalg.norm(np.cross(V64[F[:, 1]] - V64[F[:, 0]],
                                       V64[F[:, 2]] - V64[F[:, 0]]), axis=1)
    va = np.zeros(nV)
    for k in range(3):
        np.add.at(va, F[:, k], fa / 3)
    cosn = math.cos(math.radians(20.0))
    Vt = torch.as_tensor(V64, device=dev, dtype=torch.float32)
    Nt = torch.as_tensor(N, device=dev, dtype=torch.float32)
    At = torch.as_tensor(va, device=dev, dtype=torch.float32)
    free = np.ones(nV, bool)
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    stack = {}

    def measuring_frames(P):
        if not stack:
            stack["Z"] = torch.stack([z.to(dev).to(torch.float16) for z, _o, _r, _t in views])
            stack["OK"] = torch.stack([o.to(dev) for _z, o, _r, _t in views])
            stack["R"] = torch.stack([r.to(dev).float() for _z, _o, r, _t in views])
            stack["t"] = torch.stack([t.to(dev).float() for _z, _o, _r, t in views])
        Z, OK, Rs, ts = stack["Z"], stack["OK"], stack["R"], stack["t"]
        H, W = Z.shape[1:]
        X = torch.as_tensor(P, device=dev, dtype=torch.float32)
        count = 0
        for s0 in range(0, len(Rs), 64):
            pc = torch.einsum("fij,pj->fpi", Rs[s0:s0 + 64], X) + ts[s0:s0 + 64, None]
            zz = pc[..., 2]
            u = (pc[..., 0] / zz.clamp(min=1e-6) * float(K[0, 0]) + float(K[0, 2])).round().long()
            v = (pc[..., 1] / zz.clamp(min=1e-6) * float(K[1, 1]) + float(K[1, 2])).round().long()
            inb = (zz > 1e-4) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            fidx = torch.arange(s0, s0 + len(pc), device=dev)[:, None].expand_as(u)
            uc, vc = u.clamp(0, W - 1), v.clamp(0, H - 1)
            dm = Z[fidx, vc, uc].float()
            vm = OK[fidx, vc, uc] & inb & torch.isfinite(dm)
            band = tol + 0.01 * zz
            meas = vm & ((dm - zz).abs() < band)
            vis = vm & (dm > zz - band)
            nm = meas.sum(1)
            count += int(((nm >= 25) & (nm.float() >= 0.25 * vis.sum(1).float().clamp(min=1))).sum())
        return count

    snapped_w = np.zeros(nV)
    dist = np.zeros(nV)
    owner = np.full(nV, -1, np.int64)
    planes = []
    for _it in range(max_planes * 3):
        fi = np.nonzero(free)[0]
        if len(fi) < 64:
            break
        fi_t = torch.as_tensor(fi, device=dev)
        seeds = fi[rng.integers(0, len(fi), 512)]
        sn = torch.as_tensor(Ns[seeds], device=dev, dtype=torch.float32)
        sp = Vt[torch.as_tensor(seeds, device=dev)]
        sub = fi_t[torch.randperm(len(fi), generator=gen)[:200_000].to(dev)]
        scale = len(fi) / sub.numel()
        Vs_, Ns_, As_ = Vt[sub], Nt[sub], At[sub]
        best, bscore = None, 0.0
        for s0 in range(0, 512, 64):
            d = ((Vs_[None] - sp[s0:s0 + 64, None]) * sn[s0:s0 + 64, None]).sum(-1)
            al = (Ns_[None] * sn[s0:s0 + 64, None]).sum(-1).abs()
            score = (((d.abs() < tol) & (al > cosn)).float() * As_[None]).sum(1) * scale
            k = int(score.argmax())
            if float(score[k]) > bscore:
                bscore, best = float(score[k]), s0 + k
        if best is None or bscore < min_area:
            break
        n = Ns[seeds[best]].copy()
        c = V64[seeds[best]].copy()
        for _ in range(3):
            d = (V64 - c) @ n
            m = (np.abs(d) < 2 * tol) & (np.abs(N @ n) > cosn) & free
            if m.sum() < 3:
                break
            w = va[m]
            c = (V64[m] * w[:, None]).sum(0) / max(w.sum(), 1e-12)
            _, _, vt = np.linalg.svd((V64[m] - c) * np.sqrt(w)[:, None], full_matrices=False)
            n = vt[2] if vt[2] @ n > 0 else -vt[2]
        d = (V64 - c) @ n
        inl = (np.abs(d) < 2 * tol) & (np.abs(N @ n) > cosn) & free
        fin = inl[F].all(1)
        Ff = F[fin]
        if len(Ff):
            gph = coo_matrix((np.ones(len(Ff) * 2), (np.concatenate([Ff[:, 0], Ff[:, 1]]),
                                                     np.concatenate([Ff[:, 1], Ff[:, 2]]))),
                             shape=(nV, nV))
            ncomp, lab = connected_components(gph, directed=False)
            flab = lab[Ff[:, 0]]
            carea = np.bincount(flab, weights=fa[fin], minlength=ncomp)
            for comp in np.argsort(-carea):
                if carea[comp] < min_area:
                    break
                vid = np.unique(Ff[flab == comp])
                P = V64[vid]
                w = va[vid]
                cc = (P * w[:, None]).sum(0) / max(w.sum(), 1e-12)
                _, _, vt = np.linalg.svd((P - cc) * np.sqrt(w)[:, None], full_matrices=False)
                nn = vt[2] if vt[2] @ n > 0 else -vt[2]
                r = (P - cc) @ nn
                rms = float(np.sqrt((w * r ** 2).sum() / max(w.sum(), 1e-12)))
                smp = vid[rng.choice(len(vid), min(3000, len(vid)), replace=False)]
                nf = measuring_frames(V64[smp]) if rms <= tol else 0
                rec = {"area": float(carea[comp]), "rms_voxels": rms / voxel, "frames": nf,
                       "vertices": int(len(vid))}
                if rms > tol or nf < params.snap_min_frames:
                    record["rejected"] += 1
                    if len(record["rejected_examples"]) < 20:
                        record["rejected_examples"].append(rec)
                    continue
                dd = (V64[vid] - cc) @ nn
                wgt = np.clip((2 * tol - np.abs(dd)) / tol, 0, 1)
                wgt = wgt * wgt * (3 - 2 * wgt)
                # The SMOOTHED normal: a marching-cubes vertex normal swings
                # by tens of degrees between neighbours, and gating on it left
                # single vertices unmoved among moved ones -- spikes. Measured
                # on the canonical capture: wall_b faces >10 deg 17.8% gating
                # on raw normals with weight averaging, 14.2% on smoothed.
                wgt *= np.clip((np.abs(Ns[vid] @ nn) - cosn) / (1 - cosn) * 4, 0, 1)
                better = wgt > snapped_w[vid]
                vv = vid[better]
                snapped_w[vv] = wgt[better]
                dist[vv] = dd[better]
                owner[vv] = len(planes)
                rec.update(normal=[float(x) for x in nn], point=[float(x) for x in cc])
                planes.append(rec)
        free &= ~inl
        if len(planes) >= max_planes:
            break

    Vout = V64.copy()
    if planes:
        own = owner >= 0
        w = snapped_w
        normals = np.array([p["normal"] for p in planes])
        disp = -(dist * w)
        idx = np.nonzero(own & (w > 0))[0]
        Vout[idx] += disp[idx, None] * normals[owner[idx]]
        moved = np.abs(disp[idx])
        for pid, p in enumerate(planes):
            m = owner[idx] == pid
            p["moved_voxels_p50_p99"] = ([float(np.percentile(moved[m], q) / voxel)
                                          for q in (50, 99)] if m.any() else [0.0, 0.0])
        record["vertices_moved"] = int(len(idx))
        record["area_snapped"] = float(va[own & (w > 0.5)].sum())
        record["max_move"] = float(moved.max()) if len(moved) else 0.0
    record["planes"] = planes
    record["plane_count"] = len(planes)
    record["plane_areas"] = [round(p["area"], 3) for p in planes]
    stack.clear()
    record["seconds"] = round(time.time() - t0, 2)
    return Vout.astype(np.float32), record


# ---------------------------------------------------------------------------
# the on-disk mesh format
# ---------------------------------------------------------------------------

_HEADER = struct.Struct("<8sIIII6f")
FLAG_HAS_NORMALS = 1 << 0
FLAG_INDEX_U16 = 1 << 1


def write_mesh_bytes(V, F, C, N=None) -> bytes:
    """`wb-surface-mesh/1`: one self-describing buffer.

    Positions are quantised to 16 bits across the mesh's own bounding box.
    That is not lossy in any way that matters: a 14-unit scene quantises to
    0.0002 units, four orders below the 0.024 voxel the geometry was built at,
    and it halves the bytes that reach a phone.

    The header carries the counts because, unlike the dense point buffer,
    vertices and indices have different strides and a reader cannot recover
    the split from the file length.
    """
    V = np.asarray(V, np.float32).reshape(-1, 3)
    F = np.asarray(F, np.int64).reshape(-1, 3)
    n_v, n_i = len(V), F.size
    if n_v == 0:
        lo = hi = np.zeros(3, np.float32)
    else:
        lo, hi = V.min(0), V.max(0)
    span = np.maximum(hi - lo, 1e-9)

    flags = 0
    if N is not None:
        flags |= FLAG_HAS_NORMALS
    idx_dtype = np.uint32
    if n_v <= 0xFFFF:
        flags |= FLAG_INDEX_U16
        idx_dtype = np.uint16

    parts = [_HEADER.pack(MESH_MAGIC, n_v, n_i, flags, SURFACE_SCHEMA_VERSION,
                          *lo.astype(np.float32), *hi.astype(np.float32))]
    if n_v:
        q = np.clip(((V - lo) / span) * 65535.0, 0, 65535).astype(np.uint16)
        parts.append(q.tobytes())
        col = (np.zeros((n_v, 3), np.uint8) if C is None
               else np.asarray(C, np.uint8).reshape(-1, 3))
        parts.append(col.tobytes())
        if N is not None:
            nq = np.clip(np.asarray(N, np.float32) * 127.0, -127, 127).astype(np.int8)
            parts.append(nq.tobytes())
    if n_i:
        parts.append(F.astype(idx_dtype).ravel().tobytes())
    return b"".join(parts)


def read_mesh_bytes(buf: bytes):
    """Inverse of `write_mesh_bytes`, and the guard a torn file meets."""
    if len(buf) < _HEADER.size:
        raise SurfaceUnavailable("surface mesh buffer is shorter than its header")
    magic, n_v, n_i, flags, version, *box = _HEADER.unpack_from(buf, 0)
    if magic != MESH_MAGIC:
        raise SurfaceUnavailable("surface mesh buffer has the wrong magic")
    if version != SURFACE_SCHEMA_VERSION:
        raise SurfaceUnavailable(
            f"surface mesh schema {version}, expected {SURFACE_SCHEMA_VERSION}")
    lo = np.array(box[:3], np.float32)
    hi = np.array(box[3:], np.float32)
    span = np.maximum(hi - lo, 1e-9)

    if n_i % 3:
        raise SurfaceUnavailable(
            f"surface mesh buffer has {n_i} indices, not a whole number of triangles")
    has_n = bool(flags & FLAG_HAS_NORMALS)
    idx_dtype = np.uint16 if flags & FLAG_INDEX_U16 else np.uint32
    off = _HEADER.size
    need = off + n_v * (6 + 3 + (3 if has_n else 0)) + n_i * np.dtype(idx_dtype).itemsize
    if len(buf) != need:
        raise SurfaceUnavailable(
            f"surface mesh buffer is {len(buf)} bytes, header implies {need}")

    q = np.frombuffer(buf, np.uint16, n_v * 3, off).reshape(-1, 3)
    off += n_v * 6
    C = np.frombuffer(buf, np.uint8, n_v * 3, off).reshape(-1, 3)
    off += n_v * 3
    N = None
    if has_n:
        N = np.frombuffer(buf, np.int8, n_v * 3, off).reshape(-1, 3).astype(np.float32) / 127.0
        off += n_v * 3
    F = np.frombuffer(buf, idx_dtype, n_i, off).reshape(-1, 3).astype(np.int64)
    if len(F) and (int(F.max()) >= n_v or int(F.min()) < 0):
        raise SurfaceUnavailable("surface mesh buffer indexes a vertex it does not have")
    V = lo + (q.astype(np.float32) / 65535.0) * span
    return V, F, C, N
