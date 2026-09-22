"""Calibrated incremental structure-from-motion, degeneracy refused.

Runs today, on CPU, with no dependency beyond the OpenCV the Tower
already requires. It needs intrinsics and says so: prepare() raises rather
than substituting a plausible focal length, because a wrong focal length
does not produce slightly-wrong geometry, it produces a confidently wrong
trajectory.

For a two-frame window this is essential matrix + recoverPose. For longer
windows it chains properly: triangulate from the initial pair, then solve
each subsequent camera by PnP against the accumulated landmarks. Chaining
matters -- returning per-pair relative poses for a long window would give
a set of poses that are each individually correct and collectively
meaningless, because two-view translations are unit-length and carry no
common scale.

Scale is arbitrary by construction. The initial pair's baseline is
declared to be one unit, and everything after is consistent with that
choice. That is "relative", not metric, and nothing here pretends
otherwise.

## A refused keyframe is not a dead coordinate frame

This backend used to latch. The first keyframe whose pose could not be
solved set `_Chain.broken`, and every keyframe after it returned
`unavailable` from an early return that skipped ORB detection entirely.
The engine's only available response was to cut a new segment -- a new
coordinate frame, a new arbitrary unit, another fragment the viewer
cannot connect.

Measured across every calibrated session in the corpus: 1,949 refused
poses, of which 1,812 -- 93% -- were keyframes at which NO SOLVER EVER
RAN. Only 137 were an actual attempt that failed. On the 2026-09-01
long-loop walk the manifest's own split is 21 root against 60 cascaded.

So the latch is replaced by a bounded recovery state, which is the
structure every mature monocular SLAM system converges on (ORB-SLAM3's
RECENTLY_LOST between OK and LOST; stella_vslam's tracker retrying
against the local map before its relocalizer). Two things changed and
nothing else did:

1. **References are the last `EXTEND_REFERENCE_DEPTH` keyframes that
   actually HAVE a pose.** A keyframe that refused contributes no
   reference, so the next keyframe is solved against the last known-good
   one rather than against a keyframe with no entry in `absolute` --
   which could not supply a single 3-D correspondence, and so guaranteed
   the next refusal too.

2. **A refusal costs an attempt, not the chain.** `_Chain.failures`
   counts consecutive refusals; only `MAX_RECOVERY_KEYFRAMES` of them in
   a row set `broken`. One refused keyframe stays refused -- honestly,
   with no pose -- and the walk carries on in the same frame.

WHAT WAS TRIED AND ABANDONED, because it is the obvious next move:
feeding all DEPTH references' correspondences into one PnP. It made the
reconstruction worse. The numbers that comparison produced are
WITHDRAWN -- an adversarial review found the two runs differed in the
recovery budget as well, so nothing was isolated -- but the argument for
not retrying it casually stands, and `_extend` carries it.

NO ACCEPTANCE THRESHOLD MOVED. `MIN_PNP_CORRESPONDENCES`,
`PNP_REPROJECTION_ERROR_PX`, `MIN_INLIERS`, `MIN_INLIER_RATIO` and
`MIN_TRIANGULATION_ANGLE_DEG` are the values they were. Recovery is
asking the question again on a later keyframe against a reference that
still has coordinates; it is not lowering the bar for the answer. That
distinction is the whole safety argument, and
tests/test_world_builder_tracking_recovery.py pins both halves of it.
"""

from collections.abc import Sequence
from dataclasses import replace

import cv2
import numpy as np

from tower.world_builder.backend import (
    BackendCapabilities,
    Extension,
    GeometryBackend,
    GeometryEstimate,
    KeyframeInput,
    PointBlock,
    PoseEstimate,
)
from tower.world_builder.geometry import (
    MIN_INLIER_RATIO,
    MIN_INLIERS,
    MIN_TRIANGULATION_ANGLE_DEG,
    RANSAC_CONFIDENCE,
    RANSAC_THRESHOLD_PX,
    detect_and_describe,
    homography_ratio,
    match_indices,
    median_triangulation_angle_deg,
    landmark_gate,
    min_parallax_deg,
    triangulate_points,
)
from tower.world_builder import bundle
from tower.world_builder.records import CameraIntrinsics
from tower.world_builder.schema import (
    DEGENERACY_LOW_PARALLAX,
    DEGENERACY_NO_CORRESPONDENCE,
    DEGENERACY_NONE,
    DEGENERACY_PURE_ROTATION,
    POSE_STATUS_ANCHOR,
    POSE_STATUS_ROTATION_ONLY,
    POSE_STATUS_SOLVED,
    POSE_STATUS_UNAVAILABLE,
)

# Minimum 3-D/2-D correspondences before PnP is attempted. Six is the
# theoretical minimum for DLT; requiring more keeps RANSAC meaningful.
MIN_PNP_CORRESPONDENCES = 12
PNP_REPROJECTION_ERROR_PX = 3.0

# How many ACCEPTED keyframes back this backend will look for further
# sightings of landmarks it already has. 1 restores the historical
# behaviour exactly -- match the previous keyframe and nothing else --
# and is the control the benchmark neutralises to.
#
# WHY THIS EXISTS. 66.1% of landmarks were seen by exactly two views
# (measured on the previous engine; the HEAD figure is Stage 0's job). A
# two-view landmark is exactly determined -- two rays, one intersection
# -- so it constrains nothing, which is why a bundle adjuster measured
# 0.00% improvement here: two-thirds of the map was invisible to it by
# construction, not by any failure of the solver.
#
# WHY 3. Per pair asked, the useful-edge yield falls off a cliff after
# about three keyframes of separation -- roughly 58% at gap 1, 49% at
# gaps 2-5, 30% at 6-20 -- and this pipeline's own gate accepts 50.4 /
# 49.5 / 47.8% at gaps 1 / 2 / 3 before decaying. Gaps 1-3 are flat and
# cheap. Past that the right instrument is retrieval, not a wider sweep.
#
# WHAT IT COSTS. It is NOT pose-neutral, and an earlier draft of this
# comment claiming otherwise was wrong. The older references' associations
# are merged into `observed`, which the next keyframe's PnP draws
# correspondences from, so later poses can move. Withholding them keeps
# poses frozen and publishes a support table naming one image point as two
# landmarks -- measured at 2 such rows at DEPTH=1 against 147 at DEPTH=3 --
# which is worse, because that table is what cross-segment registration
# solves against. See _reobserve_against_pose.
#
# STILL POST-SOLVE, and deliberately. Feeding these references into the
# PnP itself was measured and refused -- see `_extend`'s docstring for
# the numbers. DEPTH sets the width of the support table, not the width
# of the pose solve.
#
# MEASURED, 30 real segments, DEPTH 1 -> 3: >=3-view share rose on 18
# segments and fell on ZERO (median +3.46 points); poses_solved was
# unchanged on 27, better on 2, worse on 1; the point count FELL on 13,
# and on every one of those 13 observations-per-landmark ROSE. That is
# duplicate landmarks being merged, not structure being lost -- the same
# reuse `_extend` already performs over a one-frame window, extended to
# DEPTH frames.
EXTEND_REFERENCE_DEPTH = 3

# How many CONSECUTIVE keyframes may refuse before the chain is declared
# broken and the engine cuts a new coordinate frame.
#
# WHAT THIS REPLACES. A one-way latch, i.e. this constant was
# effectively 1. The first refusal ended the segment, which is the
# single decision that produced 1,812 of the corpus's 1,949 refused
# poses -- keyframes at which no solver ran because the chain was
# already dead.
#
# WHY A BUDGET AND NOT "NEVER GIVE UP". A camera that has genuinely
# stopped seeing anything it has mapped must be allowed to say so. Every
# mature monocular system bounds this: ORB-SLAM3 gives pure-visual
# RECENTLY_LOST 3.0 seconds (Tracking.cc, the hardcoded 3.0f in the
# non-inertial branch -- the widely-quoted 5.0 is the inertial one)
# before it will consider the map lost; COLMAP retries a failed image
# max_reg_trials = 3 times and re-queues it behind fresher candidates.
#
# WHY 1, WHICH IS THE VALUE THE ADVERSARIAL MEASUREMENT SUPPORTS AND
# NOTHING LARGER IS.
#
# This constant is exactly the largest REFERENCE GAP a solve may be
# admitted at: skip k refused keyframes and the next pose is solved
# against a keyframe k+1 back. That gap is the whole risk, and no
# acceptance gate in this backend is a function of it.
#
# MEASURED against exact synthetic ground truth, target keyframe and its
# image held FIXED, only the reference moved. Median relative rotation
# error of that one solve over six scene seeds:
#
#   gap      1     2     3     4     6     8    10    12
#   forward  0.78  0.86  1.33  2.48  3.04  3.38  4.47  6.09  deg
#   strafe   1.57  2.71  4.55  5.86  9.10 11.89 14.91 19.41  deg
#
# ZERO refusals at any gap, 22-250 PnP inliers throughout. The
# 12-correspondence / 3 px gate is not a function of the gap and cannot
# be made into one.
#
# AND OVER REPEATING TEXTURE A GAP IS NOT DRIFT, IT IS A LIE. A room
# tiled at 1.5 m, a continuous walk at 0.1875 m per keyframe, no
# teleport, no occlusion, no adversary -- only the gap varied, nine
# samples each:
#
#   gap            1      2      3      4      6      8
#   walked      0.188  0.375  0.562  0.750  1.125  1.500   m
#   median err  0.007  0.207  0.241  0.411  0.766  1.499   m
#   over 10 cm    0/9    6/9    9/9    6/6    9/9    9/9
#
# Read the two rows together: whatever the gap, the solve publishes
# roughly ONE keyframe of motion. It is not drifting, it is matching the
# wrong repetition, and the keyframes the camera crossed simply cease to
# have happened. At gap 8 the camera moved 1.500 m and the pose reports
# 0.001 m -- with 169 PnP inliers, 0.14 deg of rotation error, and
# published support reprojecting at 0.22 px median. Every instrument this
# pipeline owns says that pose is excellent.
#
# The corpus agrees from the other direction. At 8, with the bundle
# adjustment running, the 2026-09-01 walk's dominant connected component
# falls from 53.5% of the geometry to 30.5%, and the drawer walk's from
# 55.8% to 27.2%. Recovery makes the world tidier -- 30 segments to 23 --
# and less true, which is exactly the trade this project refuses.
#
# So the mechanism stays and the budget is 1. That is NOT the latch it
# replaced: references are still only keyframes that HAVE poses, a
# refused keyframe still never becomes one, and the seed still holds its
# anchor. What 1 says is that a keyframe which refuses ends the
# coordinate frame, honestly, rather than being stepped over on evidence
# nothing in this pipeline can check.
#
# WHAT WOULD EARN A LARGER VALUE. Not a better matcher -- appearance is
# precisely what lies here. An instrument that checks the displacement a
# recovered pose implies against the number of keyframes it skipped. A
# gap of 8 reporting one keyframe of motion is detectable without any new
# appearance model, and it works exactly where a relocaliser would not.
# See tests/test_world_builder_recovery_safety.py, which carries the full
# tables and is the thing to re-point when that instrument exists.
MAX_RECOVERY_KEYFRAMES = 1

# -- drift control ----------------------------------------------------
#
# THE MEASUREMENT THAT PUT THIS HERE. On a clean synthetic strafe, with
# nothing refused and nothing blurred, this backend's forward-only chain
# drifts monotonically:
#
#     keyframes   rotation error med / max   max drift / path
#          6         0.95 / 1.69 deg               4.7%
#         12         1.69 / 3.46 deg               2.7%
#         20         3.19 / 9.17 deg               9.8%
#         30         5.71 / 18.88 deg             11.3%
#         40         9.21 / 33.98 deg             18.2%
#
# and the per-step recovered/true scale ratio, flat at 7.63 through
# twenty keyframes, falls to 3.46 by thirty and 2.43 by forty: the
# reconstruction CONTRACTS.
#
# That is why a walk does not become a world, and it is a bigger effect
# than the fragmentation everybody can see. A segment longer than about
# twenty keyframes is internally warped, cross-segment registration
# solves a Sim3 and correctly REFUSES two pieces whose geometry
# disagrees, and every extra keyframe of continuity bought by recovery is
# spent making the piece less placeable. Fewer segments without drift
# control is a worse world that looks like a better one.
#
# THE FIX IS BUNDLE ADJUSTMENT, and the record said it would not work:
# a prior attempt measured 0.00% improvement and blamed an observation
# graph whose median covisibility span is 1. That was true then. It is
# not true now -- EXTEND_REFERENCE_DEPTH's guided re-observation built
# the graph the earlier attempt lacked. Measured on the same walk at
# HEAD: mean 4.67 views a landmark, 67.2% with three or more, span
# median 3 / p90 9. And the drift is REACHABLE: the same observations
# reprojected through GROUND-TRUTH poses give RMS 0.49 px against 0.95
# px for the solved ones, so the solved reconstruction is not at the
# minimum.
#
# Measured, this local adjustment over a 24-keyframe chain:
#     rotation error median   3.69 deg -> 0.13 deg
#     scale first/last third  7.63/5.34 -> 9.57/9.42  (drift eliminated)
#     reprojection RMS        1.34 px  -> 0.53 px
#
# WINDOW SIZE. Cameras adjusted together. Wide enough to span the
# ~20-keyframe horizon over which drift becomes visible, small enough
# that the reduced camera system stays trivial (6*12 = 72 square) and
# the cost stays on the keyframe path rather than the frame path.
BUNDLE_WINDOW = 12

# The oldest cameras of the window are held FIXED. Without an anchor the
# window is free to slide as a rigid similarity -- the adjustment would
# be correct and would still move geometry already published, and the
# segment's own origin would stop meaning anything. Two, not one,
# because one fixed camera leaves the scale free.
BUNDLE_ANCHOR_CAMERAS = 2

# Run the adjustment once every this many accepted keyframes. 1 is
# every keyframe; the cost is real and it is paid per KEYFRAME, not per
# frame. See the report for the measured cost at each cadence.
BUNDLE_EVERY = 3

# LM iterations per adjustment. Bounded rather than run to convergence:
# this sits on the keyframe path, the window is re-adjusted a few
# keyframes later anyway, and an unbounded optimiser on a live path is
# how a walk ends mid-room.
BUNDLE_ITERATIONS = 4

# Rows of PointBlock.support_views: [frame index, feature index, landmark
# index]. int32, not int64: ORB is capped at a few thousand features per
# frame and a segment holds tens of thousands of landmarks, so every
# column is bounded three orders of magnitude below the type, and this is
# the one piece of solve state that is never pruned -- half the width is
# half the resident cost for the whole walk.
SUPPORT_DTYPE = np.int32


def _solve_pnp_ransac_or_refuse(object_points, image_points, camera_matrix):
    """solvePnPRansac, converting a solver assertion into a refusal.

    SQPNP raises rather than returning False when the minimal sample
    RANSAC draws has degenerate coordinate variance:

        sqpnp.cpp:236 (-215) point_coordinate_variance >= POINT_VARIANCE_THRESHOLD

    Which sample gets drawn is data-dependent, so this fires on some real
    walks and not others. Reproduced on the 33-segment world built from
    capture 22e9d428.

    A degenerate configuration is exactly what a refusal is FOR. Letting
    the assertion escape turns "this keyframe could not be posed" into
    "the reconstruction process died", which on the live path means a walk
    ends mid-room.

    Inputs are validated BEFORE the call, so a cv2.error reaching the
    handler is a statement about the GEOMETRY rather than about our
    argument marshalling. That distinction needs enforcing rather than
    asserting: OpenCV raises cv2.error for malformed arguments too, so
    catching it without validating first would hide a real bug in this
    repo as an innocent refusal -- which is how a pipeline quietly stops
    reconstructing.
    """
    object_points = np.asarray(object_points, dtype=np.float64)
    image_points = np.asarray(image_points, dtype=np.float64)
    if object_points.ndim != 2 or object_points.shape[1] != 3:
        raise ValueError(
            f"object_points must be (N, 3), got {object_points.shape}"
        )
    if image_points.shape != (len(object_points), 2):
        raise ValueError(
            f"image_points must be ({len(object_points)}, 2), got "
            f"{image_points.shape}"
        )
    try:
        return cv2.solvePnPRansac(
            object_points,
            image_points,
            camera_matrix,
            None,
            reprojectionError=PNP_REPROJECTION_ERROR_PX,
            confidence=RANSAC_CONFIDENCE,
            flags=cv2.SOLVEPNP_SQPNP,
        )
    except cv2.error:
        return False, None, None, None


def _observation_rows(support_rows, keypoints_by_index) -> np.ndarray:
    """(k, 4) [frame, landmark, u, v] for the support rows just created.

    Bundle adjustment needs the PIXEL a support row refers to, and a
    support row stores a feature INDEX. The index is reproducible --
    detection is deterministic -- but only from the image, and the
    backend does not keep images. So the pixel is recorded here, where
    the keypoints are still in hand, rather than recovered later by
    re-detecting from disk (which is what cross-segment registration has
    to do, and which is why registration costs seconds).

    float32 would halve this and is wrong: these are the measurements the
    optimiser's residuals are formed from, and a 1e-7 relative error on a
    pixel coordinate is a tenth of the precision the 3.0 px gate is
    trying to resolve. It is float64, and the array is pruned to the
    bundle window, so the cost is bounded by the window rather than by
    the walk.
    """
    if not len(support_rows):
        return np.zeros((0, 4), dtype=np.float64)
    out = np.empty((len(support_rows), 4), dtype=np.float64)
    write = 0
    for frame, feature, landmark in support_rows:
        keypoints = keypoints_by_index.get(int(frame))
        if keypoints is None or feature >= len(keypoints):
            continue
        u, v = keypoints[feature].pt
        out[write] = (frame, landmark, u, v)
        write += 1
    return out[:write]


def _prune_observations(observations, newest):
    """Drop observations no adjustment can reach.

    The bundle window is the last BUNDLE_WINDOW cameras, so an
    observation by an older one will never enter the reduced system
    again. Without this the array grows with the walk -- roughly five
    rows per landmark -- which is the same unbounded-state failure
    `_Chain.forget_before` exists to prevent for `observed`.
    """
    if not len(observations):
        return observations
    oldest = newest - BUNDLE_WINDOW
    if observations[0, 0] > oldest:
        return observations
    return observations[observations[:, 0] > oldest]


def _rewrite_poses(poses, absolute):
    """Re-publish the poses an adjustment moved.

    `poses` is what snapshot() returns, and it holds PoseEstimate objects
    frozen at solve time. A bundle adjustment writes to `absolute` and
    would otherwise leave the published poses stale -- so a consumer
    would draw cameras in one place and points optimised for another,
    which is worse than not adjusting at all.

    Only SOLVED poses are rewritten. An anchor stays at identity by
    definition -- it is the segment's origin, and BUNDLE_ANCHOR_CAMERAS
    holds it fixed -- and a refusal has no pose to update.
    """
    rewritten = []
    for index, pose in enumerate(poses):
        current = absolute.get(index)
        if pose.status == POSE_STATUS_SOLVED and current is not None:
            rotation, translation = current
            pose = replace(pose, rotation=rotation, translation=translation)
        rewritten.append(pose)
    return rewritten


def _local_adjust(camera_matrix, absolute, landmarks, landmark_ok,
                  first_frame, observations, newest):
    """Bundle-adjust the newest cameras of a chain, in place.

    The window is the cameras with indices greater than
    `newest - BUNDLE_WINDOW` that actually have poses; the oldest
    BUNDLE_ANCHOR_CAMERAS of them are held fixed so the adjustment cannot
    slide geometry that has already been published, and so the seven
    free parameters of a monocular reconstruction stay pinned.

    A landmark that was FIRST SEEN before the window is held fixed. Its
    older observers are outside the window and cannot move to follow it,
    so adjusting it to fit only its recent observers silently breaks
    every support row those cameras published -- measured on the drawer
    walk, before this rule existed, as published reprojection p99 3.97 ->
    12.40 px and rows over the 3 px gate 1.8% -> 10.4%, while the pose
    error genuinely improved at the same time. It is the exact shape of
    the failure this whole branch exists to avoid: a number that looks
    better bought with geometry that is worse.

    Held fixed, those landmarks are not dead weight. They are the anchor
    that ties this window to the geometry already on disk, which is what
    makes a WINDOWED adjustment control drift rather than merely
    redistribute it -- the same job ORB-SLAM gives the keyframes that
    observe a local map point without being local themselves.

    `landmark_ok` is DEMOTED for any landmark the adjustment moved
    somewhere a world may no longer state a coordinate for. An
    under-constrained landmark slides along its viewing ray at almost no
    cost in reprojection -- a two-view point has exactly that degenerate
    direction -- and the publication gate that would have caught it ran
    at creation time. Measured on the pinned eight-capture corpus before
    this demotion existed: the worst bbox blowup, full extent over the
    p2-p98 core, went 11.0 to 35.2 while every reprojection statistic
    improved. Points reprojecting perfectly from a hundred metres behind
    the wall is precisely the failure this branch exists not to ship.

    Returns the report `bundle.optimise` produced, or None if there was
    not enough of a window to adjust. `absolute` and `landmarks` are
    mutated ONLY when the optimiser reports an improvement -- a bundle
    adjustment that made the reprojection worse is a bundle adjustment
    whose result must be thrown away, not a new estimate.
    """
    window = sorted(index for index in absolute if index > newest - BUNDLE_WINDOW)
    if len(window) < BUNDLE_ANCHOR_CAMERAS + 2:
        return None
    if not len(observations):
        return None
    lowest = window[0]
    rows = observations[observations[:, 0] >= lowest]
    if len(rows) < bundle.MIN_OBSERVATIONS_PER_CAMERA * len(window):
        return None
    slot = np.full(newest + 1, -1, dtype=np.int64)
    for position, index in enumerate(window):
        slot[index] = position
    frames = rows[:, 0].astype(np.int64)
    rows = rows[slot[frames] >= 0]
    if not len(rows):
        return None

    # COMPACT the landmarks to the ones this window can see. A segment
    # can hold tens of thousands of landmarks -- the 2026-09-01 walk's
    # largest holds about six thousand -- and the optimiser allocates a
    # 3x3 block and its inverse for every one it is handed. Passing the
    # whole map would make the cost of an adjustment grow with the LENGTH
    # OF THE SEGMENT rather than with the size of the window, which is
    # the property that decides whether this can sit on the keyframe path
    # at all. The landmarks outside the window are not adjusted either
    # way: `bundle.optimise` drops anything below MIN_VIEWS_FOR_ADJUSTMENT
    # in the observations it is given, and an unobserved landmark has
    # zero.
    used, compact = np.unique(rows[:, 1].astype(np.int64), return_inverse=True)
    packed = np.empty((len(rows), 4), dtype=np.float64)
    packed[:, 0] = slot[rows[:, 0].astype(np.int64)]
    packed[:, 1] = compact
    packed[:, 2:] = rows[:, 2:]

    rotations = np.array([absolute[index][0] for index in window])
    translations = np.array([absolute[index][1] for index in window])
    points = np.asarray([landmarks[index] for index in used], dtype=np.float64)
    fixed = np.asarray(
        [first_frame[index] < lowest for index in used], dtype=bool
    )

    rotations, translations, points, report = bundle.optimise(
        rotations, translations, points, packed, camera_matrix,
        iterations=BUNDLE_ITERATIONS,
        fixed_cameras=tuple(range(min(BUNDLE_ANCHOR_CAMERAS, len(window)))),
        fixed_points=fixed,
        min_parallax_deg=min_parallax_deg(camera_matrix),
    )
    report["window_cameras"] = len(window)
    report["window_landmarks"] = len(used)
    if not report.get("improved"):
        return report
    for position, index in enumerate(window):
        absolute[index] = (rotations[position], translations[position])
    point_ok = report.get("point_ok")
    failed_reprojection = report.get("failed_reprojection")
    failed_parallax = report.get("failed_parallax")
    demoted = 0
    by_reprojection = 0
    by_parallax = 0
    for position, index in enumerate(used):
        index = int(index)
        landmarks[index] = points[position]
        # DEMOTE ONLY. A landmark the creation-time gate already refused
        # stays refused; one the adjustment moved out of its own
        # observations' reach stops being publishable. Nothing here can
        # promote, so an adjustment cannot talk a point into a world.
        #
        # `landmark_ok[index]` is what makes the count a DEMOTION rather
        # than a restatement: a landmark that was already unpublishable
        # is not refused twice, and counting it twice is what broke the
        # manifest's accounting identity.
        if point_ok is None or point_ok[position] or not landmark_ok[index]:
            continue
        landmark_ok[index] = False
        demoted += 1
        if failed_reprojection is not None and failed_reprojection[position]:
            by_reprojection += 1
        elif failed_parallax is not None and failed_parallax[position]:
            by_parallax += 1
    report["landmarks_demoted"] = demoted
    report["demoted_high_reprojection"] = by_reprojection
    report["demoted_low_parallax"] = by_parallax
    return report


def _account_for_demotions(tally, report):
    """Fold an adjustment's demotions into the discard tally.

    Into the SAME two buckets `landmark_gate` uses at creation, because
    they are the same two refusals: a landmark the adjustment moved out
    of its observations' reprojection reach is a high-reprojection
    refusal, and one whose observing rays no longer subtend
    `min_parallax_deg` is a low-parallax refusal. Nothing about the
    reason changed; only when it was noticed did.

    This is not bookkeeping for its own sake. The manifest states
    `published + refused == triangulated` so a consumer can tell a sparse
    world from a heavily filtered one, and a demotion that reduced
    `published` without appearing in `refused` would break that identity
    silently -- which is how a filtered world starts reading as an empty
    room.
    """
    if not report:
        return
    tally["high_reprojection"] += report.get("demoted_high_reprojection", 0)
    tally["low_parallax"] += report.get("demoted_low_parallax", 0)


def _support_block(rows) -> np.ndarray:
    """(m, 3) int32 from an iterable of (frame, feature, landmark)."""
    flat = np.fromiter(
        (value for row in rows for value in row), dtype=SUPPORT_DTYPE
    )
    return flat.reshape(-1, 3)


def _support_table(blocks: list) -> np.ndarray:
    """A solve's per-keyframe blocks concatenated, in creation order."""
    if not blocks:
        return np.zeros((0, 3), dtype=SUPPORT_DTYPE)
    return np.concatenate(blocks, axis=0)


CAPABILITIES = BackendCapabilities(
    backend_id="classical-sfm",
    version="2",
    requires_intrinsics=True,
    estimates_intrinsics=False,
    produces_dense_geometry=False,
    produces_metric_scale=False,
    preferred_window=8,
    device="cpu",
)


class ClassicalTwoViewBackend(GeometryBackend):
    """Incremental SfM over a window. Name kept for interface stability."""

    capabilities = CAPABILITIES

    def __init__(self) -> None:
        self._camera_matrix: np.ndarray | None = None
        self._chain: _Chain | None = None

    def prepare(self, intrinsics: CameraIntrinsics) -> None:
        camera_matrix = intrinsics.camera_matrix()
        if camera_matrix is None:
            raise ValueError(
                "ClassicalTwoViewBackend requires known intrinsics; got "
                f"source={intrinsics.source!r}. Refusing to substitute a "
                "guessed focal length -- a wrong one yields a plausible and "
                "entirely wrong trajectory."
            )
        self._camera_matrix = camera_matrix

    def release(self) -> None:
        self._camera_matrix = None
        self._chain = None

    def estimate_window(
        self, window: Sequence[KeyframeInput]
    ) -> GeometryEstimate:
        if self._camera_matrix is None:
            raise RuntimeError("prepare() must be called before estimate_window()")
        if not window:
            return GeometryEstimate(poses=())

        # Created before any early return below. The batch and live paths
        # are asserted bit-identical (test_world_builder_incremental
        # TestBitIdenticalEquivalence), and the live path always reports a
        # tally -- so a degenerate window that returns early must report
        # zeros here, not omit the key. Absent would mean "this build
        # predates the counter", which is a different fact from "nothing
        # was discarded".
        tally = self._new_discard_tally()
        features = [detect_and_describe(frame.image_gray) for frame in window]
        poses: list[PoseEstimate] = [
            PoseEstimate(keyframe_id=window[0].keyframe_id, status=POSE_STATUS_ANCHOR)
        ]
        if len(window) == 1:
            return GeometryEstimate(
                poses=tuple(poses), diagnostics=_discard_diagnostics(tally)
            )

        # World frame == first keyframe's camera frame.
        absolute: dict[int, tuple] = {0: (np.eye(3), np.zeros(3))}
        landmarks: list = []
        # Publishability, index-aligned with `landmarks`.
        landmark_ok: list = []
        # The frame each landmark was FIRST seen from, index-aligned with
        # `landmarks`. `_local_adjust` holds a landmark fixed once that
        # frame falls out of the bundle window -- see its docstring.
        landmark_frame: list = []
        # (frame index, feature index) -> landmark index, so a later frame
        # can find 3-D correspondences for the features it matched.
        observed: dict[tuple[int, int], int] = {}
        # The same association, accumulated rather than derived. `observed`
        # is a LOOKUP -- one landmark per (frame, feature) key, last writer
        # wins -- and the live path prunes it (see _Chain.forget_before), so
        # neither its contents nor its lifetime can stand in for the record
        # of what was triangulated. Rows are appended where landmarks are
        # created, in both this method and extend(), and nowhere else.
        support: list[np.ndarray] = []
        # The keyframes that HAVE a pose, nearest first. A keyframe that
        # refused contributes nothing here, which is the whole point: the
        # next solve is measured against the last known-good views rather
        # than against a frame with no entry in `absolute`.
        references: list[tuple[int, tuple]] = [(0, features[0])]
        failures = 0
        broken: str | None = None
        # [frame, landmark, u, v] for the support rows created so far.
        # The optimiser's residuals are formed from these pixels; see
        # _observation_rows for why they are recorded here rather than
        # recovered by re-detecting later.
        observations = np.zeros((0, 4), dtype=np.float64)
        solved_count = 0

        for current in range(1, len(window)):
            keyframe_id = window[current].keyframe_id
            if broken is not None:
                poses.append(
                    PoseEstimate(
                        keyframe_id=keyframe_id,
                        status=POSE_STATUS_UNAVAILABLE,
                        degeneracy=broken,
                    )
                )
                continue

            if len(absolute) < 2:
                # -- seed ------------------------------------------------
                #
                # Against the ANCHOR, not against `current - 1`. A pair
                # refused for want of parallax is refused because the
                # baseline is too short, and restarting from the frame
                # that just failed resets that baseline to zero. Holding
                # the anchor means the next attempt is WIDER than the one
                # that failed, which is the only direction that fixes it.
                anchor_index, anchor_features = references[0]
                pair = self._estimate_pair(
                    anchor_features, features[current], keyframe_id
                )
                estimate = pair.estimate
                self._add_discards(tally, pair.discarded, len(pair.points))
                if estimate.status == POSE_STATUS_SOLVED:
                    absolute[current] = (estimate.rotation, estimate.translation)
                    landmarks.extend(pair.points)
                    landmark_ok.extend(pair.quality)
                    landmark_frame.extend([anchor_index] * len(pair.points))
                    seed: list[tuple[int, int, int]] = []
                    for offset, (index_a, index_b) in enumerate(
                        pair.inlier_index_pairs
                    ):
                        observed[(anchor_index, index_a)] = offset
                        observed[(current, index_b)] = offset
                        # Emitted regardless of whether the dict write
                        # above collided: match_indices guarantees one
                        # entry per query index, not per train index, so
                        # two of the anchor's features can name the same
                        # feature of `current`. Both statements are true
                        # about the solve, and dropping one would leave a
                        # landmark with a single view -- which is not a
                        # thing that can be triangulated.
                        seed.append((anchor_index, index_a, offset))
                        seed.append((current, index_b, offset))
                    # The seed block IS the whole map so far, so
                    # delta-local and map-relative indices coincide here
                    # and only here.
                    support.append(_support_block(seed))
                    observations = np.concatenate([
                        observations,
                        _observation_rows(seed, {
                            anchor_index: anchor_features[0],
                            current: features[current][0],
                        }),
                    ])
            else:
                # -- extend by PnP ---------------------------------------
                (
                    estimate,
                    new_points,
                    new_observed,
                    reobserved,
                    extend_discards,
                    extend_quality,
                    published_reobserved,
                ) = self._extend(
                    references,
                    features[current],
                    current,
                    absolute,
                    landmarks,
                    observed,
                    keyframe_id,
                )
                self._add_discards(tally, extend_discards, len(new_points))
                if estimate.status == POSE_STATUS_SOLVED:
                    absolute[current] = (estimate.rotation, estimate.translation)
                    observed.update(reobserved)
                    reobserved_rows = [
                        (frame, feature, landmark)
                        for (frame, feature), landmark
                        in published_reobserved.items()
                    ]
                    support.append(_support_block(reobserved_rows))
                    base = len(landmarks)
                    landmarks.extend(new_points)
                    landmark_ok.extend(extend_quality)
                    landmark_frame.extend(
                        [references[0][0]] * len(new_points)
                    )
                    for key, offset in new_observed.items():
                        observed[key] = base + offset
                    created_rows = [
                        (frame, feature, base + offset)
                        for (frame, feature), offset in new_observed.items()
                    ]
                    support.append(_support_block(created_rows))
                    nearest_index, nearest_features = references[0]
                    keypoints_by_index = {
                        nearest_index: nearest_features[0],
                        current: features[current][0],
                    }
                    observations = np.concatenate([
                        observations,
                        _observation_rows(reobserved_rows, keypoints_by_index),
                        _observation_rows(created_rows, keypoints_by_index),
                    ])

            poses.append(estimate)
            if estimate.status == POSE_STATUS_SOLVED:
                failures = 0
                references.insert(0, (current, features[current]))
                del references[EXTEND_REFERENCE_DEPTH:]
                solved_count += 1
                observations = _prune_observations(observations, current)
                if BUNDLE_WINDOW and solved_count % BUNDLE_EVERY == 0:
                    _account_for_demotions(
                        tally,
                        _local_adjust(
                            self._camera_matrix, absolute, landmarks,
                            landmark_ok, landmark_frame, observations,
                            current,
                        ),
                    )
                    poses = _rewrite_poses(poses, absolute)
            else:
                failures += 1
                if failures >= MAX_RECOVERY_KEYFRAMES:
                    broken = estimate.degeneracy

        block = _publishable_block(landmarks, landmark_ok, support)
        return GeometryEstimate(
            poses=tuple(poses),
            points=block,
            diagnostics=_discard_diagnostics(tally),
        )

    # -- the incremental seam -------------------------------------------
    #
    # estimate_window() above is already strictly forward-only: frame i is
    # solved by _extend() against the last EXTEND_REFERENCE_DEPTH keyframes
    # that HAVE poses plus the accumulated landmarks, and never looks
    # forward. There is no LOOP CLOSURE. There IS now a bundle
    # adjustment -- see BUNDLE_WINDOW -- and the record that one was
    # "measured at 0.00% drift improvement at 16, 32 and 104 keyframes"
    # was true of the engine it was measured on, whose observation graph
    # was a chain with a median covisibility span of 1
    # (docs/agent-handoffs/WORLD-BUILDER.md section 10). At HEAD that
    # graph is mean 4.67 views a landmark with 67.2% at three or more,
    # and the adjustment takes a forty-keyframe chain's rotation error
    # from 9.21 deg to 0.15 deg.
    #
    # So `absolute`, `landmarks` and `observed` really are the entire
    # carried state, and the only reason a rebuild re-paid for all of it
    # was that they were local variables. _Chain is those three promoted
    # to instance state, and nothing else.
    #
    # The methods below reuse the SAME _estimate_pair and _extend helpers
    # estimate_window uses, so the two paths cannot drift in their
    # geometry. What they deliberately do NOT share is the orchestration:
    # an oracle that delegates to the thing it is checking checks
    # nothing, and tests/test_world_builder_incremental.py checks this
    # one bit-for-bit.

    def begin(self, intrinsics: CameraIntrinsics) -> None:
        self.prepare(intrinsics)
        self.reset()

    def reset(self) -> None:
        self._chain = _Chain()

    def extend(self, frame: KeyframeInput) -> Extension:
        if self._camera_matrix is None:
            raise RuntimeError("begin() must be called before extend()")
        if self._chain is None:
            self._chain = _Chain()
        chain = self._chain
        index = chain.count

        if chain.broken is not None:
            # estimate_window() stops chaining once MAX_RECOVERY_KEYFRAMES
            # consecutive keyframes have refused, and marks every later
            # frame unavailable carrying the LAST refusal's degeneracy.
            # Latched here for the same reason and with the same value. It
            # skips detection too: estimate_window computes those
            # descriptors up front and then never reads them, so not
            # computing them changes no output.
            pose = PoseEstimate(
                keyframe_id=frame.keyframe_id,
                status=POSE_STATUS_UNAVAILABLE,
                degeneracy=chain.broken,
            )
            chain.poses.append(pose)
            chain.count += 1
            return Extension(pose=pose)

        features = detect_and_describe(frame.image_gray)
        new_points: list = []
        # Publishability for THIS delta, index-aligned with new_points.
        new_points_ok: list = []
        # Rows for THIS keyframe's delta block, landmark indices local to
        # it. The chain's own copy carries the same rows shifted into the
        # accumulated map.
        delta_support: list[tuple[int, int, int]] = []

        if index == 0:
            pose = PoseEstimate(
                keyframe_id=frame.keyframe_id, status=POSE_STATUS_ANCHOR
            )
            # World frame == first keyframe's camera frame.
            chain.absolute[0] = (np.eye(3), np.zeros(3))
        elif len(chain.absolute) < 2:
            # -- seed, against the ANCHOR ----------------------------
            #
            # Not against `index - 1`. See estimate_window(): a pair
            # refused for want of parallax is refused because the
            # baseline is too short, and re-seeding from the frame that
            # just failed resets that baseline to zero. Holding the
            # anchor makes each retry WIDER than the attempt before it.
            anchor_index, anchor_features = chain.references[0]
            pair = self._estimate_pair(
                anchor_features, features, frame.keyframe_id
            )
            pose = pair.estimate
            self._add_discards(chain.discarded, pair.discarded, len(pair.points))
            if pose.status == POSE_STATUS_SOLVED:
                chain.absolute[index] = (pose.rotation, pose.translation)
                chain.landmarks.extend(pair.points)
                chain.landmark_ok.extend(pair.quality)
                chain.landmark_frame.extend([anchor_index] * len(pair.points))
                new_points = pair.points
                new_points_ok = list(pair.quality)
                for offset, (index_a, index_b) in enumerate(
                    pair.inlier_index_pairs
                ):
                    chain.observed[(anchor_index, index_a)] = offset
                    chain.observed[(index, index_b)] = offset
                    delta_support.append((anchor_index, index_a, offset))
                    delta_support.append((index, index_b, offset))
                # The seed block IS the whole map so far, so delta-local
                # and map-relative indices coincide here and only here.
                chain.support.append(_support_block(delta_support))
                chain.observations = np.concatenate([
                    chain.observations,
                    _observation_rows(delta_support, {
                        anchor_index: anchor_features[0],
                        index: features[0],
                    }),
                ])
        else:
            (
                pose,
                triangulated,
                new_observed,
                reobserved,
                extend_discards,
                extend_quality,
                published_reobserved,
            ) = self._extend(
                chain.references,
                features,
                index,
                chain.absolute,
                chain.landmarks,
                chain.observed,
                frame.keyframe_id,
            )
            self._add_discards(
                chain.discarded, extend_discards, len(triangulated)
            )
            if pose.status == POSE_STATUS_SOLVED:
                nearest_index, nearest_features = chain.references[0]
                chain.absolute[index] = (pose.rotation, pose.translation)
                chain.observed.update(reobserved)
                reobserved_rows = [
                    (frame, feature, landmark)
                    for (frame, feature), landmark
                    in published_reobserved.items()
                ]
                chain.support.append(_support_block(reobserved_rows))
                base = len(chain.landmarks)
                chain.landmarks.extend(triangulated)
                chain.landmark_ok.extend(extend_quality)
                chain.landmark_frame.extend([nearest_index] * len(triangulated))
                new_points_ok = list(extend_quality)
                new_points = triangulated
                for key, offset in new_observed.items():
                    chain.observed[key] = base + offset
                    delta_support.append((key[0], key[1], offset))
                created_rows = [
                    (frame, feature, base + landmark)
                    for frame, feature, landmark in delta_support
                ]
                chain.support.append(_support_block(created_rows))
                keypoints_by_index = {
                    nearest_index: nearest_features[0],
                    index: features[0],
                }
                chain.observations = np.concatenate([
                    chain.observations,
                    _observation_rows(reobserved_rows, keypoints_by_index),
                    _observation_rows(created_rows, keypoints_by_index),
                ])
                # A re-observation names a landmark this delta does not
                # carry, so it is not expressible in the delta's own index
                # space. It reaches a consumer through snapshot(), which is
                # the authoritative view anyway.

        chain.poses.append(pose)
        chain.count += 1
        if pose.status in (POSE_STATUS_SOLVED, POSE_STATUS_ANCHOR):
            # ONLY a keyframe that has a pose becomes a reference. This is
            # the invariant every mature system holds and this backend did
            # not: ORB-SLAM's reference keyframe is set in
            # CreateNewKeyFrame() and in UpdateLocalKeyFrames(), both
            # reachable only after a successful track; DSO's coarse
            # tracking reference is set only from makeKeyFrame(), after
            # the window optimisation. A frame with no entry in
            # `absolute` yields no 3-D correspondence, so promoting it
            # would guarantee the NEXT solve fails too -- which is how
            # one refusal used to become a permanent fork.
            chain.failures = 0
            chain.references.insert(0, (index, features))
            del chain.references[EXTEND_REFERENCE_DEPTH:]
            chain.forget_before()
            # SOLVED only, never the anchor. estimate_window() counts the
            # same way -- the anchor is appended before its loop and is
            # not a solve -- and the two paths are asserted bit-identical,
            # so a cadence that differs by one keyframe is a real
            # divergence rather than a cosmetic one.
            if pose.status == POSE_STATUS_SOLVED:
                chain.solved += 1
            chain.observations = _prune_observations(chain.observations, index)
            if (
                BUNDLE_WINDOW
                and chain.solved
                and chain.solved % BUNDLE_EVERY == 0
                and pose.status == POSE_STATUS_SOLVED
            ):
                # Drift control, on the KEYFRAME path. See BUNDLE_WINDOW
                # for the measurement that put it here: without it a
                # chain of forty keyframes is warped by tens of degrees
                # and contracted by a factor of three, and every extra
                # keyframe of continuity recovery buys is spent making
                # the segment less placeable rather than more.
                chain.bundle = _local_adjust(
                    self._camera_matrix, chain.absolute, chain.landmarks,
                    chain.landmark_ok, chain.landmark_frame,
                    chain.observations, index,
                )
                chain.poses = _rewrite_poses(chain.poses, chain.absolute)
                if chain.bundle:
                    chain.demoted += chain.bundle.get("landmarks_demoted", 0)
                    _account_for_demotions(chain.discarded, chain.bundle)
        else:
            chain.failures += 1
            if chain.failures >= MAX_RECOVERY_KEYFRAMES:
                chain.broken = pose.degeneracy
        # THE DELTA IS WHAT THIS KEYFRAME BELIEVED, NOT WHAT THE MAP
        # NOW SAYS, and that distinction arrived with the bundle
        # adjustment. `pose` and `new_points` are captured before
        # `_local_adjust` moves `chain.absolute` and `chain.landmarks`
        # and before `_rewrite_poses` rebuilds `chain.poses`, so on a
        # forty-keyframe synthetic strafe every solved delta pose
        # differs from the snapshot's (worst 0.684 deg) and nearly every
        # delta landmark coordinate does too.
        #
        # `snapshot()` is authoritative -- the Extension docstring
        # already said so about poses -- and a viewer that appends
        # deltas forever would drift away from it. NOTHING IN
        # PRODUCTION DOES: the engine reads only `pose.status` and
        # `chain_broken`, and `_LiveSolve` freezes and serves
        # `snapshot()`. The incremental suite asserts the reconciliation
        # that does still hold: deltas minus retired landmarks equals
        # the snapshot's count.
        return Extension(
            pose=pose,
            # An EDGE. extend() returns early above when the chain is
            # already broken, so reaching here with chain.broken set
            # means THIS keyframe broke it. A `not was_broken` guard
            # used to sit here and was dead code -- it was evaluated
            # after that early return, so it was always False.
            chain_broken=chain.broken is not None,
            new_points=_publishable_block(
                new_points, new_points_ok, [_support_block(delta_support)]
            ),
        )

    def snapshot(self) -> GeometryEstimate:
        chain = self._chain
        if chain is None or chain.count == 0:
            return GeometryEstimate(poses=())
        block = _publishable_block(
            chain.landmarks, chain.landmark_ok, chain.support
        )
        return GeometryEstimate(
            poses=tuple(chain.poses),
            points=block,
            diagnostics=_discard_diagnostics(chain.discarded),
        )

    # -- helpers --------------------------------------------------------

    @staticmethod
    def _new_discard_tally():
        return {"low_parallax": 0, "high_reprojection": 0, "produced": 0}

    @staticmethod
    def _add_discards(tally, counts, kept):
        """Fold one triangulation call into a running tally.

        `produced` is the number of landmarks that entered the MAP.
        Refusals are a subset of it -- a refused landmark is still in the
        map, just not publishable -- so they are not added again here.
        The manifest can then state the identity
        published + refused == triangulated rather than leaving a consumer
        to infer that the points it can see are all that were ever made.
        """
        tally["low_parallax"] += counts.get("low_parallax", 0)
        tally["high_reprojection"] += counts.get("high_reprojection", 0)
        # `kept` is what entered the MAP. Refusals are a subset of it --
        # a refused landmark is still in the map, just not publishable --
        # so it must not be added again here.
        tally["produced"] += kept
        return tally


    class _PairResult:
        __slots__ = (
            "estimate",
            "points",
            "inlier_index_pairs",
            "discarded",
            "quality",
        )

        def __init__(
            self,
            estimate,
            points,
            inlier_index_pairs,
            discarded=None,
            quality=None,
        ):
            self.estimate = estimate
            self.points = points
            self.inlier_index_pairs = inlier_index_pairs
            self.discarded = discarded or {
                "low_parallax": 0,
                "high_reprojection": 0,
            }
            # Per-landmark publishability, index-aligned with `points`.
            # The landmarks stay in the map either way; this only says
            # which of them a world may state a coordinate for.
            self.quality = (
                list(quality) if quality is not None else [True] * len(points)
            )

    def _estimate_pair(self, features_a, features_b, keyframe_id):
        keypoints_a, descriptors_a = features_a
        keypoints_b, descriptors_b = features_b
        index_pairs = match_indices(descriptors_a, descriptors_b)
        matches = len(index_pairs)

        def refuse(degeneracy, **extra):
            return self._PairResult(
                PoseEstimate(
                    keyframe_id=keyframe_id,
                    status=POSE_STATUS_UNAVAILABLE,
                    degeneracy=degeneracy,
                    matches=matches,
                    **extra,
                ),
                [],
                [],
            )

        if matches < MIN_INLIERS:
            return refuse(DEGENERACY_NO_CORRESPONDENCE)

        points_a = np.float32([keypoints_a[i].pt for i, _ in index_pairs])
        points_b = np.float32([keypoints_b[j].pt for _, j in index_pairs])

        camera_matrix = self._camera_matrix
        essential, mask = cv2.findEssentialMat(
            points_a, points_b, camera_matrix,
            method=cv2.USAC_MAGSAC,
            prob=RANSAC_CONFIDENCE,
            threshold=RANSAC_THRESHOLD_PX,
        )
        if essential is None or essential.shape != (3, 3):
            return refuse(DEGENERACY_NO_CORRESPONDENCE)

        # recoverPose takes `mask` as BOTH input and output: it narrows it
        # in place with a cheirality test bounded by an undocumented
        # `distanceThresh` default of 50 baselines. Reading `mask` after
        # the call therefore does NOT give the epipolar inlier count, and
        # the field persisted as `inlier_ratio` was measuring cheirality.
        # Measured, one scene, ORB matches at 640x360:
        #
        #   baseline   matches   epipolar inliers   ratio AFTER recoverPose
        #     0.02 m      1160     1134  (0.978)              0.001
        #     0.04 m      1145     1103  (0.963)              0.004
        #     0.06 m      1154     1120  (0.971)              0.098
        #     0.08 m      1137     1116  (0.982)              0.941
        #     0.30 m       987      958  (0.971)              0.971
        #
        # At short baselines nearly every correspondence is a genuine
        # epipolar inlier and the reported "inlier ratio" is three orders
        # of magnitude smaller. It was a measurement of baseline over
        # depth wearing another field's name -- which also explains two
        # historical results recorded as facts about geometry.
        epipolar_mask = mask.copy()
        cheirality, rotation, translation, _ = cv2.recoverPose(
            essential, points_a, points_b, camera_matrix, mask=mask
        )
        epipolar_kept = epipolar_mask.ravel() > 0
        epipolar_inliers = int(epipolar_kept.sum())
        inlier_ratio = epipolar_inliers / matches if matches else 0.0

        kept = mask.ravel() > 0
        inliers = int(kept.sum())
        # What the gate below has always actually used, now carried in the
        # field that was already declared for it. `KeyframeEdge`'s own
        # comment describes `cheirality_fraction` as "the fraction of
        # correspondences passing recoverPose's cheirality check" -- which
        # is exactly this, and which the code was instead putting into
        # `inlier_ratio` while filling this field with something else.
        cheirality_ratio = inliers / matches if matches else 0.0
        translation = np.asarray(translation, dtype=np.float64).reshape(3)

        inlier_a, inlier_b = points_a[kept], points_b[kept]
        displacement = (
            float(np.median(np.linalg.norm(inlier_a - inlier_b, axis=1)))
            if inliers
            else None
        )
        angle = median_triangulation_angle_deg(
            inlier_a, inlier_b, rotation, translation, camera_matrix
        )
        measured = {
            "matches": matches,
            "inliers": epipolar_inliers,
            "inlier_ratio": inlier_ratio,
            "median_triangulation_deg": angle,
            "median_displacement_px": displacement,
            "cheirality_fraction": cheirality_ratio,
            "r_h": homography_ratio(points_a, points_b),
        }

        # Rotation survives degeneracy; translation does not. Under a pure
        # rotation recoverPose still returns a confident translation whose
        # direction is meaningless -- measured 62 and 106 degrees of error
        # on pairs that reported no other complaint.
        # Gated on the CHEIRALITY ratio, which is what this condition has
        # always used -- the constant is simply named for the wrong thing.
        # Deliberately unchanged: correcting the reporting is a separate
        # act from changing which poses are accepted, and the second needs
        # a sweep this did not have. Note the consequence, measured: a real
        # sideways strafe at a 4-6 cm baseline recovers direction to within
        # 2 degrees and is still refused here.
        degenerate = (
            inliers < MIN_INLIERS
            or cheirality_ratio < MIN_INLIER_RATIO
            or angle is None
            or angle < MIN_TRIANGULATION_ANGLE_DEG
        )
        if degenerate:
            # "pure_rotation" overstates what was observed: a low
            # cheirality ratio means few points are in front of both
            # cameras within 50 baselines, which a genuine short-baseline
            # translation also produces. Kept for now because the label is
            # persisted and consumers switch on it.
            reason = (
                DEGENERACY_PURE_ROTATION
                if cheirality_ratio < MIN_INLIER_RATIO
                else DEGENERACY_LOW_PARALLAX
            )
            return self._PairResult(
                PoseEstimate(
                    keyframe_id=keyframe_id,
                    status=POSE_STATUS_ROTATION_ONLY,
                    rotation=rotation,
                    translation=None,
                    degeneracy=reason,
                    **measured,
                ),
                [],
                [],
            )

        points, keep_mask, quality, discarded = triangulate_points(
            inlier_a, inlier_b, rotation, translation, camera_matrix,
            return_mask=True, return_quality=True,
        )
        surviving_pairs = [
            pair for pair, keep in zip(
                [p for p, k in zip(index_pairs, kept) if k], keep_mask
            ) if keep
        ]
        return self._PairResult(
            PoseEstimate(
                keyframe_id=keyframe_id,
                status=POSE_STATUS_SOLVED,
                rotation=rotation,
                translation=translation,
                degeneracy=DEGENERACY_NONE,
                **measured,
            ),
            list(points),
            surviving_pairs,
            discarded,
            [bool(q) for q in quality],
        )

    def _extend(
        self,
        references,
        features_current,
        current_index,
        absolute,
        landmarks,
        observed,
        keyframe_id,
    ):
        """Solve one keyframe against the nearest keyframe that HAS a pose.

        `references` is the last EXTEND_REFERENCE_DEPTH keyframes with an
        entry in `absolute`, nearest first. `references[0]` is what the
        pose is solved against; the rest supply further sightings AFTER
        the solve, through `_reobserve_against_pose`.

        WHAT CHANGED, AND WHAT DELIBERATELY DID NOT

        What changed is WHICH keyframe `references[0]` is. It used to be
        `current - 1` unconditionally. It is now the nearest keyframe
        that actually has coordinates, so a keyframe whose pose refused
        is stepped over rather than becoming a reference that cannot
        supply a single 3-D correspondence. That is the whole recovery
        mechanism, and it is the invariant every mature system holds:
        ORB-SLAM sets its reference keyframe only in CreateNewKeyFrame()
        and UpdateLocalKeyFrames(), both reachable only after a
        successful track; DSO sets its coarse tracking reference only
        from makeKeyFrame(), after the window optimisation.

        What deliberately did NOT change is the number of references the
        PnP itself draws correspondences from: ONE.

        Feeding all DEPTH references into a single PnP was implemented
        and tried, because it is the obvious widening and because
        ORB-SLAM's TrackLocalMap is exactly that idea done properly. It
        made the reconstruction worse and was abandoned.

        THE NUMBERS THAT COMPARISON PRODUCED ARE WITHDRAWN. An
        adversarial review found the two runs differed in the RECOVERY
        BUDGET as well as in the reference count, and the recovery
        budget is worth tens of points of dominant-component share on
        its own, so the comparison isolated nothing. The variant is no
        longer in the tree and has not been re-measured cleanly. What
        remains is the argument, which stands on its own and is the
        reason not to try it again casually:

        TrackLocalMap is not "match more keyframes". It projects
        landmarks through a PREDICTED pose and searches a small radius
        around the prediction, so appearance is only ever asked to
        choose among candidates that are already geometrically
        plausible. We have no pose prediction at this point -- that is
        what we are solving for -- so an older reference contributes
        pure descriptor matches over a wider baseline, where ORB's
        appearance assumption is weakest. Widening the correspondence
        set is not the same act as widening the local map, and only the
        second is what ORB-SLAM does. Anyone re-attempting this should
        build the pose prediction FIRST.

        So the older references keep the job they were measured doing:
        re-observation after the pose exists, gated on reprojecting
        through it. See `_reobserve_against_pose`.
        """
        keypoints_current, descriptors_current = features_current
        nearest_index, nearest_features = references[0]
        keypoints_nearest, descriptors_nearest = nearest_features
        index_pairs = match_indices(descriptors_nearest, descriptors_current)
        matches = len(index_pairs)

        object_points, image_points, matched_pairs = [], [], []
        # A feature in the current frame can be named by more than one
        # match (knnMatch guarantees one entry per queryIdx, not per
        # trainIdx), so the last writer would otherwise win and silently
        # bind a landmark to the wrong feature. Keep the first claim.
        claimed: set[int] = set()
        reobserved: dict[tuple[int, int], int] = {}
        for index_reference, index_current in index_pairs:
            if index_current in claimed:
                continue
            claimed.add(index_current)
            landmark = observed.get((nearest_index, index_reference))
            if landmark is None:
                matched_pairs.append((index_reference, index_current))
                continue
            object_points.append(landmarks[landmark])
            image_points.append(keypoints_current[index_current].pt)
            # THE propagation. Without this the map is write-only: a
            # landmark seen in frame N-1 and re-seen in frame N cannot
            # be found from frame N, so step N->N+1 re-triangulates
            # the same physical structure instead of reusing it,
            # roughly doubling the point count with duplicates of the
            # same structure and badly degrading the trajectory.
            #
            # Deliberately no percentage here. The figures this
            # comment used to carry were single-run measurements, and
            # findEssentialMat(USAC_MAGSAC)/solvePnPRansac(SQPNP) are
            # not seeded -- a committed test's own docstring claims
            # 1.32% where the same test now measures 1.62% on a
            # different OpenCV build. Point at the report, which can
            # carry the conditions; a bare number in a comment cannot.
            # See reports/2026-08-22-world-builder-closeout.md 5.2.
            reobserved[(current_index, index_current)] = landmark

        if len(object_points) < MIN_PNP_CORRESPONDENCES:
            return (
                PoseEstimate(
                    keyframe_id=keyframe_id,
                    status=POSE_STATUS_UNAVAILABLE,
                    degeneracy=DEGENERACY_NO_CORRESPONDENCE,
                    matches=matches,
                ),
                [],
                {},
                {},
                {"low_parallax": 0, "high_reprojection": 0},
                [],
                {},
            )

        # `reobserved` is built ABOVE, before the solve, because
        # observed.update(reobserved) is what lets the NEXT keyframe find
        # correspondences at all -- starving it is what cost 26 solved
        # poses when the landmark gate first filtered the map.
        #
        # Publication is a different question. An association the solver's
        # own RANSAC called an outlier is not evidence, and support.json
        # is consumed by cross-segment registration, which PnPs against
        # it. Measured on capture 64f48114: 587 of 1851 offered
        # correspondences (31.7%) were outliers and every one was
        # published; re-observation rows reached 723,209 px reprojection
        # on e1c52b9f while creation rows never exceeded 3.0 px.
        reobserved_keys = list(reobserved)
        (
            ok,
            rotation_vector,
            translation,
            inlier_indices,
        ) = _solve_pnp_ransac_or_refuse(
            np.asarray(object_points, dtype=np.float64),
            np.asarray(image_points, dtype=np.float64),
            self._camera_matrix,
        )
        if not ok or inlier_indices is None or len(inlier_indices) < MIN_PNP_CORRESPONDENCES:
            return (
                PoseEstimate(
                    keyframe_id=keyframe_id,
                    status=POSE_STATUS_UNAVAILABLE,
                    degeneracy=DEGENERACY_LOW_PARALLAX,
                    matches=matches,
                    inliers=0 if inlier_indices is None else int(len(inlier_indices)),
                ),
                [],
                {},
                {},
                {"low_parallax": 0, "high_reprojection": 0},
                [],
                {},
            )

        rotation, _ = cv2.Rodrigues(rotation_vector)
        translation = np.asarray(translation, dtype=np.float64).reshape(3)

        # Triangulate the features that had no landmark yet, using the two
        # absolute poses, so new structure lands directly in world frame.
        (
            new_points,
            new_observed,
            discard_counts,
            new_quality,
        ) = self._triangulate_new(
            keypoints_nearest,
            keypoints_current,
            matched_pairs,
            absolute[nearest_index],
            (rotation, translation),
            nearest_index,
            current_index,
        )

        # Only the correspondences the pose solve ACCEPTED are published.
        # The full `reobserved` still goes to `observed` for solving.
        published_reobserved = {
            reobserved_keys[index]: reobserved[reobserved_keys[index]]
            for index in np.asarray(inlier_indices).ravel().tolist()
            if 0 <= index < len(reobserved_keys)
        }

        # Further sightings from the references BEHIND the one the pose
        # was solved against, admitted only if they reproject through
        # that pose. This runs after the solve and cannot change it --
        # see _reobserve_against_pose, and see this method's docstring
        # for the measurement that says it must stay that way.
        # Published on the same terms as the inliers above, because it
        # passed the same reprojection bar those inliers were selected by.
        guided = self._reobserve_against_pose(
            references[1:],
            keypoints_current,
            descriptors_current,
            current_index,
            rotation,
            translation,
            landmarks,
            observed,
            claimed,
        )
        # Merged into `reobserved`, and so into the caller's `observed`,
        # and NOT only into the published support.
        #
        # The pose-neutral variant was built first and measured, because
        # it looked strictly safer: keep guided rows out of `observed`
        # and no later pose can move. It produces an INCONSISTENT support
        # table. `observed` is what the next keyframe consults to decide
        # whether a feature already has a landmark; a guided row withheld
        # from it means the next step finds nothing, triangulates the
        # same physical point a second time, and publishes a support row
        # binding that same (frame, feature) to a different landmark.
        # Measured on the synthetic walk: 2 such rows at DEPTH=1 -- the
        # documented seed-pair case -- against 147 at DEPTH=3.
        #
        # A feature naming two landmarks is not a cosmetic duplicate.
        # `support.json` is what cross-segment registration solves PnP
        # against, so one of those two rows is feeding a wrong 3-D point
        # to the thing that decides where a segment sits in the world.
        #
        # So the association goes where associations go. The cost is that
        # a guided row becomes a correspondence for the NEXT pose solve,
        # which means this change is no longer provably pose-neutral and
        # has to be measured across the corpus rather than argued.
        reobserved.update(guided)
        published_reobserved.update(guided)

        # Re-observations index into the EXISTING landmark list, so they
        # must not be shifted by the caller's `base` offset the way newly
        # triangulated points are. Returned separately for that reason.
        return (
            PoseEstimate(
                keyframe_id=keyframe_id,
                status=POSE_STATUS_SOLVED,
                rotation=rotation,
                translation=translation,
                degeneracy=DEGENERACY_NONE,
                matches=matches,
                inliers=int(len(inlier_indices)),
                inlier_ratio=len(inlier_indices) / matches if matches else None,
            ),
            new_points,
            new_observed,
            reobserved,
            discard_counts,
            new_quality,
            published_reobserved,
        )

    def _reobserve_against_pose(
        self,
        extra_references,
        keypoints_current,
        descriptors_current,
        current_index,
        rotation,
        translation,
        landmarks,
        observed,
        already_claimed,
    ):
        """Further sightings of landmarks we already hold, from keyframes
        older than the one the pose was solved against.

        WHAT THIS IS, AND WHAT IT DELIBERATELY IS NOT

        The obvious way to widen this backend is to feed several
        references' correspondences into ONE PnP. That is a different and
        larger change: it alters the pose directly, and pricing it
        honestly means measuring it across the corpus. It is a reasonable
        eventual move and it is not this.

        This runs AFTER the pose is solved, so it does not participate in
        solving THIS keyframe. It does, however, affect LATER ones: what
        it admits is merged into `observed`, and `observed` is where the
        next keyframe finds its correspondences. That is a deliberate
        choice and the alternative was measured and rejected -- see the
        comment at the call site. What it buys is how many views each
        landmark is known to have been seen from, which is the quantity a
        pose graph and a bundle adjuster actually consume, and the
        quantity cross-segment
        registration PnPs against.

        WHY A REPROJECTION TEST AND NOT A MATCHER SCORE

        A descriptor match is a claim about appearance. Appearance is
        exactly what fails on repeated indoor texture -- two identical
        chair legs match happily -- and `support.json` is consumed by
        registration, which solves against it. So an association is
        admitted here only if the landmark, projected through the pose we
        just solved, lands within PNP_REPROJECTION_ERROR_PX of the
        feature that claims it.

        That threshold is deliberately not a new number. It is the same
        one `solvePnPRansac` used to decide what counted as an inlier for
        this very pose, so an admitted re-observation is one the pose
        solve itself would have accepted had it been offered. Inventing a
        looser threshold here would be inventing evidence.

        Cheirality is checked too: a landmark behind the camera is
        refused before its pixel distance is even considered, because a
        negative depth can still reproject onto a plausible pixel.

        A current feature already bound by the pose solve is never
        rebound, and the first extra reference to claim a free feature
        keeps it -- references are visited nearest-first, so the claim
        order is fixed and the output does not depend on dict iteration.
        """
        if not extra_references:
            return {}

        projection = self._camera_matrix @ np.hstack(
            [rotation, translation.reshape(3, 1)]
        )
        limit_squared = PNP_REPROJECTION_ERROR_PX * PNP_REPROJECTION_ERROR_PX
        admitted: dict[tuple[int, int], int] = {}
        claimed = set(already_claimed)

        for ref_index, (_keypoints_ref, descriptors_ref) in extra_references:
            for index_ref, index_current in match_indices(
                descriptors_ref, descriptors_current
            ):
                if index_current in claimed:
                    continue
                landmark = observed.get((ref_index, index_ref))
                if landmark is None:
                    # Nothing triangulated from that feature, so there is
                    # no landmark to re-observe. Triangulating one here
                    # would be new structure, which is the pose-changing
                    # path this method exists to stay out of.
                    continue
                point = landmarks[landmark]
                projected = projection @ np.array(
                    [point[0], point[1], point[2], 1.0], dtype=np.float64
                )
                depth = projected[2]
                if not np.isfinite(depth) or depth <= 0:
                    continue
                u = projected[0] / depth
                v = projected[1] / depth
                x, y = keypoints_current[index_current].pt
                du = u - x
                dv = v - y
                if du * du + dv * dv > limit_squared:
                    continue
                claimed.add(index_current)
                admitted[(current_index, index_current)] = landmark

        return admitted

    def _triangulate_new(
        self,
        keypoints_previous,
        keypoints_current,
        matched_pairs,
        pose_previous,
        pose_current,
        previous_index,
        current_index,
    ):
        if not matched_pairs:
            return [], {}, {"low_parallax": 0, "high_reprojection": 0}, []

        rotation_p, translation_p = pose_previous
        rotation_c, translation_c = pose_current
        projection_p = self._camera_matrix @ np.hstack(
            [rotation_p, translation_p.reshape(3, 1)]
        )
        projection_c = self._camera_matrix @ np.hstack(
            [rotation_c, translation_c.reshape(3, 1)]
        )

        points_p = np.float32([keypoints_previous[i].pt for i, _ in matched_pairs]).T
        points_c = np.float32([keypoints_current[j].pt for _, j in matched_pairs]).T
        homogeneous = cv2.triangulatePoints(
            projection_p, projection_c, points_p, points_c
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            xyz = (homogeneous[:3] / homogeneous[3]).T

        # Two absolute poses, so these landmarks are already in world
        # frame -- and unlike the seed pair, neither pose is identity.
        #
        # The gate is evaluated ONLY over landmarks that will actually
        # enter the map (finite, and in front of both cameras). Gating a
        # wider set would count refusals for points that were never
        # admitted anyway, and the manifest's accounting identity
        # -- published + refused == triangulated -- would stop closing.
        depth_p = (rotation_p @ xyz.T).T[:, 2] + translation_p[2]
        depth_c = (rotation_c @ xyz.T).T[:, 2] + translation_c[2]
        admissible = (
            np.isfinite(xyz).all(axis=1) & (depth_p > 0) & (depth_c > 0)
        )

        # points_p/points_c are built TRANSPOSED above, (2, N). The gate
        # validates shape rather than reshaping, so passing them without
        # the .T would raise instead of silently rejecting every landmark.
        gate_keep = np.zeros(len(xyz), dtype=bool)
        counts = {"low_parallax": 0, "high_reprojection": 0}
        if admissible.any():
            subset_keep, counts = landmark_gate(
                xyz[admissible],
                points_p.T[admissible],
                points_c.T[admissible],
                pose_previous,
                pose_current,
                self._camera_matrix,
            )
            gate_keep[admissible] = subset_keep

        new_points, new_observed, new_quality = [], {}, []
        for offset, ((index_p, index_c), point) in enumerate(
            zip(matched_pairs, xyz)
        ):
            if not admissible[offset]:
                continue
            landmark = len(new_points)
            new_points.append(point)
            # Kept in the map regardless of the gate -- see the comment in
            # geometry.triangulate_points. The gate decides publication,
            # not whether PnP may use the bearing.
            new_quality.append(bool(gate_keep[offset]))
            new_observed[(previous_index, index_p)] = landmark
            new_observed[(current_index, index_c)] = landmark
        return new_points, new_observed, counts, new_quality


def _publishable_block(landmarks, landmark_ok, support_blocks):
    """The PointBlock a world may actually state, plus remapped support.

    Landmarks the gate refused stay in the solver's map -- they are usable
    bearings -- but a world must not publish a coordinate it cannot
    defend. Filtering happens HERE, at the boundary, so that removing a
    point from the output can never change which poses were solved.

    Support rows name landmarks by index, so dropping landmarks requires
    remapping the survivors and discarding rows that point at the dropped
    ones. Getting this wrong would silently mis-attribute observations to
    the wrong 3-D point, which is worse than publishing nothing.
    """
    if not landmarks:
        return None
    # Deliberately NOT padded. On the one array whose desync means
    # "publish a coordinate you cannot defend", defaulting to True fails
    # in the direction that ships unvetted geometry, and an over-long
    # list would be silently truncated. Both are bugs in this file, not
    # conditions to tolerate at runtime.
    if len(landmark_ok) != len(landmarks):
        raise ValueError(
            f"landmark_ok has {len(landmark_ok)} entries for "
            f"{len(landmarks)} landmarks; they are appended together and "
            f"must stay in lockstep"
        )
    ok = list(landmark_ok)
    remap = {}
    kept = []
    for index, landmark in enumerate(landmarks):
        if ok[index]:
            remap[index] = len(kept)
            kept.append(landmark)
    if not kept:
        return None

    table = _support_table(support_blocks)
    if len(table):
        survives = np.array(
            [row[2] in remap for row in table], dtype=bool
        )
        table = table[survives]
        if len(table):
            table = table.copy()
            table[:, 2] = [remap[row[2]] for row in table]
    return PointBlock(
        xyz=np.asarray(kept, dtype=np.float32),
        support_views=table,
    )


def _discard_diagnostics(tally):
    """Shape the running tally into the diagnostics the manifest carries.

    `points_triangulated` is stated rather than left to be inferred: a
    consumer seeing only the surviving points cannot otherwise tell a
    sparse world from a heavily filtered one, and those need different
    responses from whoever is holding the glasses.
    """
    return {
        "points_discarded": {
            "low_parallax": tally["low_parallax"],
            "high_reprojection": tally["high_reprojection"],
        },
        "points_triangulated": tally["produced"],
    }


class _Chain:
    """The carried state of one forward-only solve, and nothing else.

    Exactly the locals estimate_window() builds -- `absolute`,
    `landmarks`, `observed`, `support`, `references`, `failures` -- plus
    the poses emitted so far and the latch recording where the chain
    stopped. If anything else ever has to live here, this backend has
    stopped being forward-only, and the equivalence test is the thing
    that will say so.
    """

    __slots__ = (
        "absolute",
        "broken",
        "bundle",
        "count",
        "demoted",
        "discarded",
        "failures",
        "landmark_frame",
        "landmark_ok",
        "landmarks",
        "observations",
        "observed",
        "poses",
        "references",
        "solved",
        "support",
    )

    def __init__(self) -> None:
        self.count = 0
        # (frame index, features) for the keyframes that HAVE poses,
        # nearest first, at most EXTEND_REFERENCE_DEPTH of them. Bounded,
        # so this is a constant, not growth -- which is the property
        # test_retained_state_does_not_grow_with_the_number_of_keyframes
        # asserts and which a deque of every past frame would break.
        #
        # A keyframe that REFUSED never enters this list. That is the
        # whole recovery mechanism: the next solve reaches past the
        # failure to the last views that actually have coordinates.
        self.references: list = []
        # Consecutive refusals since the last pose. Reset by any keyframe
        # that solves. MAX_RECOVERY_KEYFRAMES of them sets `broken`.
        self.failures = 0
        # Keyframes that have solved, ever. Drives the bundle cadence,
        # and is deliberately NOT `count`: a run of refusals must not
        # trigger an adjustment of a window nothing new has entered.
        self.solved = 0
        # [frame, landmark, u, v]. PRUNED to the bundle window, so this
        # is a constant and not growth -- the same property
        # `forget_before` maintains for `observed`.
        self.observations = np.zeros((0, 4), dtype=np.float64)
        # The last adjustment's report, or None. Diagnostics only.
        self.bundle = None
        # Landmarks an adjustment retired since the chain began. Exactly
        # the shortfall between "every delta appended" and what
        # snapshot() publishes, which is the invariant
        # test_extend_reports_only_the_structure_that_keyframe_added
        # now asserts.
        self.demoted = 0
        self.landmark_ok = []
        # The frame each landmark was FIRST seen from, index-aligned with
        # `landmarks`. Not pruned: it is one int per landmark, and it is
        # what decides whether an adjustment may move a landmark whose
        # older observers have left the window.
        self.landmark_frame = []
        # Discards accumulate for the LIFE of the chain, so the snapshot
        # reports the whole segment rather than the last window.
        self.discarded = {
            "low_parallax": 0,
            "high_reprojection": 0,
            "produced": 0,
        }
        self.absolute: dict[int, tuple] = {}
        self.landmarks: list = []
        # (frame index, feature index) -> landmark index. PRUNED.
        self.observed: dict[tuple[int, int], int] = {}
        # The support table, one (m, 3) int32 block per keyframe that
        # added something, concatenated by snapshot(). NOT pruned, and it
        # is the only thing here that is not: `observed` can be dropped
        # because nothing will ever look up an old frame again, whereas
        # this IS the output. A list of small arrays rather than one
        # grown array so appending stays O(m) with no reallocation, and
        # rather than a list of tuples so the cost is 12 bytes a row
        # instead of the ~200 a dict entry cost before the prune.
        self.support: list = []
        self.poses: list[PoseEstimate] = []
        # Degeneracy of the refusal that exhausted the recovery budget,
        # or None.
        self.broken: str | None = None

    def forget_before(self) -> None:
        """Drop observations no later step can reach.

        _extend() reads `observed[(reference, f)]` for every keyframe in
        `references` and for nothing else, so once the window has moved
        on nothing will ever look up a frame older than the oldest
        reference. estimate_window() keeps them all because it is over in
        one call. A live solve is not over, and unpruned this dict grows
        by roughly two entries per ORB match per keyframe.

        The bound is `references`, NOT `count - DEPTH`. During recovery
        the references do not advance, so the retained window sits still
        rather than sliding off the last views that have poses -- which
        would delete exactly the correspondences recovery needs. It is
        still a constant: `references` is capped at DEPTH.

        The retained window was one frame when this backend matched only
        its immediate predecessor. It is now DEPTH frames, so the
        constant below is DEPTH times larger -- still a constant, which
        is the property that matters and the one
        test_retained_state_does_not_grow_with_the_number_of_keyframes
        asserts. At DEPTH = 3 that is ~0.45 MB against the ~0.15 MB the
        measurement below records, and it does not grow with the walk.

        `support` is deliberately NOT pruned here. It holds the same
        association and is the reason this backend records anything at
        all about 2-D/3-D linkage, so pruning it would silently give the
        live path one frame's worth of a field the rebuild path fills
        completely. It is affordable precisely because it is not this
        dict: 12 bytes a row against ~200 bytes an entry, so the 26.1 MB
        below is ~1.3 MB, and the 142.9 MB is ~8 MB.

        Measured, 480x360 synthetic walk, retained `observed` unpruned
        against pruned: 26.1 MB vs 0.15 MB at 155 keyframes, 142.9 MB vs
        0.15 MB at 1000. Pruned it is flat, because what survives is one
        frame's features. It changes no output -- which is a claim the
        equivalence test exists to check, not one it is asked to
        tolerate.
        """
        if not self.references:
            return
        oldest = min(index for index, _ in self.references)
        self.observed = {
            key: value for key, value in self.observed.items() if key[0] >= oldest
        }
