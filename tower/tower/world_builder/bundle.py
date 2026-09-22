"""Sparse bundle adjustment: the drift control this pipeline did not have.

WHY THIS EXISTS

Measured on a perfect synthetic strafe, with no tracking failure, no
refusal and no blur -- the shipped forward-only chain drifts:

    keyframes   rotation error (median / max)   max drift / path
        6            0.95 deg / 1.69 deg              4.7%
       12            1.69 deg / 3.46 deg              2.7%
       20            3.19 deg / 9.17 deg              9.8%
       30            5.71 deg / 18.88 deg            11.3%
       40            9.21 deg / 33.98 deg            18.2%

and the per-step recovered/true scale ratio, flat at 7.63 through the
first twenty keyframes, collapses to 3.46 by thirty and 2.43 by forty.
The reconstruction CONTRACTS.

That is the real reason a walk does not become a world. Fragmentation is
the symptom people see; drift is what makes the fragments unmergeable,
because cross-segment registration solves a Sim3 between two segments
and correctly refuses two pieces whose internal geometry disagrees.
Making segments longer without controlling drift trades a visible
failure for an invisible one.

WHY A PRIOR ATTEMPT MEASURED 0.00%, AND WHY THAT IS NOT THE LAST WORD

The record says a bundle adjuster improved drift by 0.00% at 16, 32 and
104 keyframes, and attributes it to an observation graph whose median
covisibility span is 1 -- 66% of landmarks seen by exactly two views, and
a two-view landmark is exactly determined, so it constrains nothing.

That was true of the engine it was measured on. It is not true of this
one. Measured on the same 40-keyframe strafe at HEAD:

    landmarks 5,876   mean views 4.67   >= 3 views 67.2%
    observation span: median 3, p90 9, p99 24; 28% span >= 5 frames

The guided re-observation added by EXTEND_REFERENCE_DEPTH built the
graph the earlier attempt was missing. And the drift is demonstrably
REACHABLE by optimisation rather than baked in: reprojecting the very
same observations through GROUND-TRUTH poses gives RMS 0.49 px against
0.95 px for the solved poses. The solved reconstruction is not at the
minimum. There is something to find.

WHAT THIS IS

Levenberg-Marquardt over camera poses and landmark positions, with the
Schur complement eliminating the landmarks -- the standard structure,
because the normal equations of a bundle problem are block-arrowhead and
solving them densely is the only thing that would make this expensive.
A local window is a few tens of cameras, so the reduced camera system is
a few hundred square and dense linear algebra on it is free; the cost is
all in forming it, which is vectorised.

NumPy and OpenCV only. scipy is not a Tower dependency and this must run
where the rest of the world builder runs.

THE INVARIANT THAT BIT TWICE, stated once so it does not bite again:

    EVERY OBSERVATION OF A CAMERA THAT IS ALLOWED TO MOVE MUST
    PARTICIPATE IN THE OPTIMISATION.

Drop an observation for any reason -- too few views on its landmark, a
per-landmark view cap, a sampling shortcut -- and its landmark stays put
while its camera moves out from under it. The trajectory gets better and
the SUPPORT TABLE, which is what cross-segment registration and the
viewer actually consume, gets worse. Both `MIN_VIEWS_FOR_ADJUSTMENT` and
`MAX_VIEWS_PER_LANDMARK` shipped values that violated this and each
carries the measurement that caught it. A landmark that must not MOVE is
expressed with `fixed_points`, which keeps its observations in the
problem; it is never expressed by removing its rows.

GAUGE. A monocular reconstruction is free up to a similarity: 3 rotation,
3 translation, 1 scale. The Jacobian is therefore rank-deficient by 7 and
an undamped Gauss-Newton step is not unique. LM's damping term makes the
system positive definite, which is the standard resolution and the one
used here -- no parameters are held fixed, so with `fixed_cameras`
empty the optimiser is free to move the whole window and the caller must
NOT assume camera 0 stays where it was. Measured with no fixed camera,
camera 0 moves 0.095 deg and 1.15e-2. Nothing re-anchors afterwards, and
an earlier draft of this docstring claimed it did; a caller that needs an
anchor must say so, which is why `_local_adjust` always passes
`fixed_cameras=(0, 1)`.

CONVENTION. Poses are T_camera_world: `x_camera = R @ x_world + t`, the
same convention `backends/classical.py` publishes and `_triangulate_new`
projects with. Rotation increments are applied on the LEFT
(R <- exp(delta) R), which is why the pose Jacobian below is
-[R X]_cross and not something involving the current rotation vector.
"""

import numpy as np

# Robustifier scale, in pixels. The same 3.0 px that
# `classical.PNP_REPROJECTION_ERROR_PX` uses to call a correspondence an
# inlier, so a residual this optimiser down-weights is one the pose solve
# would already have rejected. Deliberately not a new number.
HUBER_DELTA_PX = 3.0

# A landmark seen from fewer views than this is dropped: neither adjusted
# nor allowed to constrain a camera.
#
# 2, NOT 3, and the difference is not academic -- it was the single worst
# bug on this branch.
#
# The tempting argument for 3 is about INFORMATION, and it is correct as
# far as it goes: two views determine a point exactly, so a two-view
# landmark can always be moved to make both its residuals zero and it
# tells the optimiser nothing about the cameras. Excluding it saves three
# parameters and loses no information.
#
# It loses CONSISTENCY, which is a different property and the one that
# ships. A third of this map is two-view. Drop those observations and the
# cameras move out from under landmarks that are not allowed to follow,
# so every support row those landmarks already published stops
# reprojecting -- and `support.json` is what cross-segment registration
# solves PnP against. Measured on the drawer walk, adjustment ON,
# identical in every other respect:
#
#     min_views   published reproj median / p99   rows over the 3 px gate
#         3            0.723 / 13.698 px                  9.55%
#         2            0.522 /  2.761 px                  0.57%
#
# The baseline with no adjustment at all is 0.543 / 3.974 px and 1.79%.
# So at 3 the adjustment made the published map markedly WORSE than not
# adjusting at all; at 2 it makes it better on every statistic. That is
# the exact shape this branch exists to avoid -- a trajectory that
# genuinely improved while the artifact everybody consumes got worse --
# and it is visible only if you measure the support table rather than the
# trajectory.
MIN_VIEWS_FOR_ADJUSTMENT = 2

# Cameras whose observations are too few to pose. Below this the camera
# would be moved by noise.
MIN_OBSERVATIONS_PER_CAMERA = 12

# Views of one landmark that may enter the reduced camera system. The
# Schur complement is quadratic in views per landmark, so this bounds a
# worst case; 0 disables it.
#
# 16, and it was 8, and 8 was wrong for THE SAME REASON `min_views = 3`
# was wrong. Both dropped observations from the optimisation while
# leaving the cameras that made them free to move. The landmark then
# stays where it was, the camera does not, and every support row between
# them stops reprojecting -- so the published map degrades while the
# trajectory improves.
#
# MEASURED on the 2026-09-01 long-loop walk, adjustment ON, nothing else
# changed:
#
#   cap    published reproj median / p99   over 3 px   dominant component
#     8          0.726 / 15.800 px            9.51%      7,521 pts (24.1%)
#    16          0.546 /  2.781 px            0.69%     16,340 pts (53.5%)
#
# The baseline with no adjustment at all is 0.587 / 4.732 px, 2.54%, and
# 8,285 points (27.3%). So the cap was not a cost knob at all: at 8 it
# was quietly destroying the artifact the whole pipeline exists to
# produce, and at 16 the same adjustment doubles the dominant component
# AND halves the reprojection tail.
#
# 16 rather than uncapped because uncapped changes the SOLVE: measured
# 194 solved poses against 323, as a stronger adjustment moves poses
# enough to change which later keyframes their PnP can place. The
# dominant component is larger (64.7%) and 129 fewer cameras are posed,
# which is not obviously a better world and is certainly a different
# one. 16 keeps the solve where it was and takes the coherence.
#
# HONESTLY: 16 still violates the invariant above, for the rare landmark
# that exceeds it. A bundle window holds at most BUNDLE_WINDOW = 12
# cameras, so a landmark can only pass 16 ROWS by being claimed twice in
# one frame -- which `_extend` explicitly allows and which the support
# table records as two rows. The cap therefore binds on a small tail
# rather than on ordinary landmarks, which is why 16 measures clean where
# 8 did not. It is a residual risk, not a resolved one, and the right
# eventual fix is to stop publishing two rows for one image point rather
# than to raise this number again.
MAX_VIEWS_PER_LANDMARK = 16

# How far an adjustment must COLLAPSE a landmark's parallax before the
# landmark is retired, as a fraction of what it had.
#
# The bound alone was not enough. Demoting every landmark whose angle
# crossed `min_parallax_deg` retires one whose parallax went 0.200 to
# 0.198 degrees exactly as readily as one that was slid along its ray,
# and a map this full of marginal two-view landmarks has a great many of
# the first kind: measured, 29% of the 2026-08-29 normal walk's published
# points, against a handful of landmarks actually responsible for the
# bbox blowup.
#
# A point that genuinely slid does not lose a few percent of its
# parallax, it loses most of it. Requiring the angle to at least HALVE
# separates the two without inventing a second bound -- the floor is
# still `min_parallax_deg`, and this only asks that the adjustment be
# the cause rather than the occasion.
PARALLAX_COLLAPSE_RATIO = 0.5


def _skew(vectors: np.ndarray) -> np.ndarray:
    """(n, 3) -> (n, 3, 3) cross-product matrices."""
    n = len(vectors)
    out = np.zeros((n, 3, 3), dtype=np.float64)
    out[:, 0, 1] = -vectors[:, 2]
    out[:, 0, 2] = vectors[:, 1]
    out[:, 1, 0] = vectors[:, 2]
    out[:, 1, 2] = -vectors[:, 0]
    out[:, 2, 0] = -vectors[:, 1]
    out[:, 2, 1] = vectors[:, 0]
    return out


def _exp_so3(delta: np.ndarray) -> np.ndarray:
    """(n, 3) rotation vectors -> (n, 3, 3) rotation matrices.

    Rodrigues, vectorised, with the small-angle branch written out rather
    than guarded by an `if`: an LM step is overwhelmingly small, so the
    branch that matters is the one where sin(theta)/theta must not be
    evaluated as 0/0.
    """
    theta = np.linalg.norm(delta, axis=1)
    small = theta < 1e-8
    safe = np.where(small, 1.0, theta)
    axis = delta / safe[:, None]
    axis[small] = 0.0
    K = _skew(axis)
    sin = np.where(small, 0.0, np.sin(theta))[:, None, None]
    cos = np.where(small, 0.0, 1.0 - np.cos(theta))[:, None, None]
    identity = np.broadcast_to(np.eye(3), (len(delta), 3, 3))
    rotation = identity + sin * K + cos * (K @ K)
    # Exactly identity where the step is negligible, rather than
    # identity-plus-rounding: a pose that did not move must not acquire a
    # 1e-17 rotation, because these compose over many iterations.
    rotation[small] = np.eye(3)
    return rotation


def _residuals_and_terms(rotations, translations, points, camera_index,
                         point_index, observed, camera_matrix):
    """Reprojection residuals plus everything the normal equations need."""
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    rotated = np.einsum("nij,nj->ni", rotations[camera_index],
                        points[point_index])
    camera = rotated + translations[camera_index]
    depth = camera[:, 2]
    # A landmark behind a camera has no meaningful pixel error. It is
    # given a zero Jacobian and a zero residual rather than being dropped,
    # so the arrays stay index-aligned with `observed` -- and it cannot
    # then drag the solution, which is the behaviour a hard filter would
    # have had anyway.
    valid = np.isfinite(depth) & (depth > 1e-6)
    safe_depth = np.where(valid, depth, 1.0)

    u = fx * camera[:, 0] / safe_depth + cx
    v = fy * camera[:, 1] / safe_depth + cy
    residual = np.stack([u - observed[:, 0], v - observed[:, 1]], axis=1)
    residual[~valid] = 0.0

    # d(u, v) / d(camera point)
    inverse = 1.0 / safe_depth
    projection = np.zeros((len(camera), 2, 3), dtype=np.float64)
    projection[:, 0, 0] = fx * inverse
    projection[:, 0, 2] = -fx * camera[:, 0] * inverse * inverse
    projection[:, 1, 1] = fy * inverse
    projection[:, 1, 2] = -fy * camera[:, 1] * inverse * inverse
    projection[~valid] = 0.0

    # d(camera point) / d(pose). Rotation is updated on the LEFT, so the
    # rotation block is -[R X]_cross and the translation block is I.
    jacobian_pose = np.zeros((len(camera), 2, 6), dtype=np.float64)
    jacobian_pose[:, :, :3] = projection @ (-_skew(rotated))
    jacobian_pose[:, :, 3:] = projection
    # d(camera point) / d(landmark) = R
    jacobian_point = projection @ rotations[camera_index]
    return residual, jacobian_pose, jacobian_point, valid


def _huber_weights(residual: np.ndarray, delta: float) -> np.ndarray:
    """Per-observation IRLS weights. sqrt(w) scales both r and J.

    Huber rather than a hard cut, and applied to the two-dimensional
    residual NORM rather than per axis, because a correspondence is
    wrong as a pair of coordinates or not at all.
    """
    norm = np.linalg.norm(residual, axis=1)
    weight = np.ones_like(norm)
    over = norm > delta
    weight[over] = delta / norm[over]
    return weight


def optimise(
    rotations,
    translations,
    points,
    observations,
    camera_matrix,
    *,
    iterations: int = 10,
    fixed_cameras=(),
    fixed_points=None,
    huber_delta: float | None = None,
    min_views: int | None = None,
    min_parallax_deg: float | None = None,
):
    """Refine poses and landmarks against their observations.

    Arguments
      rotations     (n_cam, 3, 3) R_camera_world, modified in place? NO --
                    a new array is returned and the inputs are untouched.
      translations  (n_cam, 3)
      points        (n_pt, 3)
      observations  (m, 4) float array of [camera, point, u, v]. The first
                    two columns are indices and must be integral; they are
                    taken as a float array because that is what callers
                    have and converting once here is cheaper than
                    converting at every call site.
      camera_matrix (3, 3) K
      fixed_cameras indices whose pose must not move. A caller that has
                    already published a pose downstream passes it here, so
                    an adjustment cannot rewrite geometry someone is
                    holding.
      fixed_points  boolean mask over `points`, or None. A landmark whose
                    observations are not ALL in this window must be
                    passed here. Its older observers are outside the
                    window and cannot move to follow it, so adjusting it
                    to fit only its recent observers silently breaks
                    every support row those older cameras published --
                    measured on the drawer walk as published reprojection
                    p99 3.97 -> 12.40 px, with the pose error genuinely
                    improving at the same time. Its observations STILL
                    constrain the cameras: a landmark held where the
                    older map put it is an anchor tying this window to
                    geometry already on disk, which is most of what makes
                    a windowed adjustment control drift rather than
                    merely redistribute it.

    Returns (rotations, translations, points, report). `report` carries
    the reprojection RMS before and after, the iteration count, whether
    the optimiser actually improved anything -- a caller must be able to
    DISCARD the result, and this is what tells it to -- and `point_ok`, a
    per-point mask of which adjusted landmarks a world may still
    publish. See its construction at the end of this function for why a
    caller that ignores it ships points the creation-time gate would have
    refused.

    Nothing is re-anchored on the way out. Whatever is not named in
    `fixed_cameras` may move, including camera 0.

    The returned poses are re-anchored so that the first camera in
    `rotations` is exactly where it was. The optimiser is gauge-free (see
    the module docstring) so without this the whole window would drift in
    a way no consumer expects.
    """
    # Resolved in the body, not as a default argument: a default is
    # bound at definition time, so a sweep that monkeypatches the module
    # constant would silently measure the same thing every time. That
    # happened, and cost a round of measurements that all came back
    # identical.
    if huber_delta is None:
        huber_delta = HUBER_DELTA_PX
    if min_views is None:
        min_views = MIN_VIEWS_FOR_ADJUSTMENT
    rotations = np.array(rotations, dtype=np.float64, copy=True)
    translations = np.array(translations, dtype=np.float64, copy=True)
    points = np.array(points, dtype=np.float64, copy=True)
    observations = np.asarray(observations, dtype=np.float64)
    n_cam = len(rotations)
    n_pt = len(points)
    if n_cam == 0 or n_pt == 0 or len(observations) == 0:
        return rotations, translations, points, {
            "iterations": 0, "reason": "nothing to adjust"
        }

    camera_index = observations[:, 0].astype(np.int64)
    point_index = observations[:, 1].astype(np.int64)
    observed = observations[:, 2:4]

    # Drop landmarks with too few views. They add three parameters each
    # and no constraint -- a two-view point can always be placed so both
    # residuals vanish -- so they would inflate the normal equations and
    # bias the robustifier's scale without informing a single pose.
    views = np.bincount(point_index, minlength=n_pt)
    usable_point = views >= min_views
    if fixed_points is not None:
        # A fixed landmark is kept whatever its view count: it is not a
        # parameter, so "too few views to determine it" does not apply,
        # and its observations are exactly the anchor this window needs.
        keep = usable_point[point_index] | np.asarray(
            fixed_points, dtype=bool
        )[point_index]
    else:
        keep = usable_point[point_index]
    if not keep.any():
        return rotations, translations, points, {
            "iterations": 0,
            "reason": f"no landmark reaches {min_views} views",
            "landmarks_adjusted": 0,
        }
    camera_index = camera_index[keep]
    point_index = point_index[keep]
    observed = observed[keep]

    # Cap the views a single landmark contributes. The reduced camera
    # system is formed over PAIRS of observations of the same landmark,
    # so cost is quadratic in views: a landmark seen from every camera of
    # a twelve-camera window costs 144 pair blocks on its own, where a
    # typical one (measured mean 4.67 views) costs 22. The cap sits well
    # above the mean so it binds only on the handful that would otherwise
    # dominate.
    #
    # WHICH views are kept is not a detail. Keeping the FIRST k was tried
    # and is wrong: it systematically starves the NEWEST camera, which on
    # an online window is the one camera the adjustment exists to fix.
    # Measured on an eight-camera rig at cap 4, every camera settled
    # within 0.04 deg of the gauge except the last, which sat 0.86 deg
    # off -- a quiet, one-sided error that only appears at the end of the
    # window, which is exactly where it does the most damage.
    #
    # So both ends are kept: the oldest views carry the widest baseline
    # and the newest carry the camera being solved.
    if MAX_VIEWS_PER_LANDMARK:
        order = np.lexsort((camera_index, point_index))
        sorted_points = point_index[order]
        run_start = np.flatnonzero(
            np.r_[True, sorted_points[1:] != sorted_points[:-1]]
        )
        run_length = np.diff(np.r_[run_start, len(order)])
        ranked = np.empty(len(order), dtype=np.int64)
        run_size = np.empty(len(order), dtype=np.int64)
        ranked[order] = np.arange(len(order)) - np.repeat(run_start, run_length)
        run_size[order] = np.repeat(run_length, run_length)
        oldest = MAX_VIEWS_PER_LANDMARK // 2
        newest = MAX_VIEWS_PER_LANDMARK - oldest
        within = (ranked < oldest) | (run_size - 1 - ranked < newest)
        camera_index = camera_index[within]
        point_index = point_index[within]
        observed = observed[within]

    # A fixed landmark keeps its observations -- they constrain the
    # cameras -- but is not a parameter, so it is excluded from V, W and
    # the Schur elimination.
    if fixed_points is None:
        point_is_fixed = np.zeros(n_pt, dtype=bool)
    else:
        point_is_fixed = np.asarray(fixed_points, dtype=bool)
        if point_is_fixed.shape != (n_pt,):
            raise ValueError(
                f"fixed_points must be a mask over {n_pt} points, got "
                f"{point_is_fixed.shape}"
            )
    usable_point = usable_point & ~point_is_fixed
    free_obs = ~point_is_fixed[point_index]

    # And cameras with too few surviving observations to be posed by them.
    per_camera = np.bincount(camera_index, minlength=n_cam)
    frozen = np.zeros(n_cam, dtype=bool)
    frozen[per_camera < MIN_OBSERVATIONS_PER_CAMERA] = True
    for index in fixed_cameras:
        if 0 <= index < n_cam:
            frozen[index] = True

    free = np.nonzero(~frozen)[0]
    if len(free) == 0:
        return rotations, translations, points, {
            "iterations": 0, "reason": "every camera is fixed",
        }
    slot = -np.ones(n_cam, dtype=np.int64)
    slot[free] = np.arange(len(free))
    size = 6 * len(free)

    # -- the sparsity structure, built ONCE ---------------------------
    #
    # WHICH observation pairs share a landmark does not change as the
    # estimate moves, and rebuilding that inside the damping loop was the
    # single largest cost in this function: a Python-level tile() per
    # landmark per attempt, profiled at 140,000 calls over a 30-keyframe
    # walk. Everything below is index arithmetic on the FIXED structure,
    # and only the numbers flowing through it are recomputed.
    live_rows = np.nonzero((~frozen[camera_index]) & free_obs)[0]
    if len(live_rows) == 0:
        return rotations, translations, points, {
            "iterations": 0, "reason": "no observation by a free camera",
        }
    order = np.argsort(point_index[live_rows], kind="stable")
    sorted_rows = live_rows[order]
    sorted_points = point_index[sorted_rows]
    boundaries = np.flatnonzero(
        np.r_[True, sorted_points[1:] != sorted_points[:-1], True]
    )
    starts = boundaries[:-1]
    counts = np.diff(boundaries)
    pair_counts = counts * counts
    total_pairs = int(pair_counts.sum())
    # Every (a, b) within each landmark's run, without a Python loop:
    # the within-run pair ordinal k gives a = k // c and b = k % c.
    offset_within = (
        np.arange(total_pairs)
        - np.repeat(np.r_[0, np.cumsum(pair_counts)[:-1]], pair_counts)
    )
    count_per_pair = np.repeat(counts, pair_counts)
    base = np.repeat(starts, pair_counts)
    rows_a = sorted_rows[base + offset_within // count_per_pair]
    rows_b = sorted_rows[base + offset_within % count_per_pair]
    # Flat destination in the (size, size) reduced matrix for each of the
    # 36 entries of a pair's 6x6 block.
    block_rows = slot[camera_index[rows_a]][:, None] * 6 + np.arange(6)[None, :]
    block_cols = slot[camera_index[rows_b]][:, None] * 6 + np.arange(6)[None, :]
    pair_flat = (
        block_rows[:, :, None] * size + block_cols[:, None, :]
    ).reshape(-1)
    live_slots = slot[camera_index[live_rows]]
    live_points = point_index[live_rows]

    def rms(residual, valid):
        if not valid.any():
            return float("inf")
        return float(np.sqrt((residual[valid] ** 2).sum(axis=1).mean()))

    residual, jac_pose, jac_point, valid = _residuals_and_terms(
        rotations, translations, points, camera_index, point_index,
        observed, camera_matrix)
    before = rms(residual, valid)
    weight = _huber_weights(residual, huber_delta)
    cost = float((weight[:, None] * residual ** 2).sum())

    # The estimate as it arrived, kept so the parallax test at the end can
    # ask what THIS adjustment changed rather than re-judging what the
    # creation gate already ruled on.
    points_before = points.copy()
    rotations_before = rotations.copy()
    translations_before = translations.copy()

    lam = 1e-3
    taken = 0
    for _ in range(iterations):
        root = np.sqrt(weight)[:, None, None]
        A = jac_pose * root          # (m, 2, 6)
        B = jac_point * root         # (m, 2, 3)
        e = residual * np.sqrt(weight)[:, None]   # (m, 2)

        W = np.einsum("mki,mkj->mij", A, B)          # (m, 6, 3)
        ea = -np.einsum("mki,mk->mi", A, e)          # (m, 6)
        eb = -np.einsum("mki,mk->mi", B, e)          # (m, 3)
        AA = np.einsum("mki,mkj->mij", A, A)
        BB = np.einsum("mki,mkj->mij", B, B)

        # bincount, not np.add.at: the same accumulation, and profiled an
        # order of magnitude faster. This runs inside the damping loop,
        # so it is the difference between a keyframe-path cost and an
        # offline one.
        U = np.stack([
            np.bincount(camera_index, weights=AA[:, i, j], minlength=n_cam)
            for i in range(6) for j in range(6)
        ], axis=1).reshape(n_cam, 6, 6)
        # V and g_pt see only FREE points. A fixed landmark has no
        # parameter block, so accumulating one would produce a diagonal
        # entry that is inverted and back-substituted into a point
        # nothing is allowed to move.
        free_weight = free_obs.astype(np.float64)
        V = np.stack([
            np.bincount(point_index, weights=BB[:, i, j] * free_weight,
                        minlength=n_pt)
            for i in range(3) for j in range(3)
        ], axis=1).reshape(n_pt, 3, 3)
        g_cam = np.stack([
            np.bincount(camera_index, weights=ea[:, i], minlength=n_cam)
            for i in range(6)
        ], axis=1)
        g_pt = np.stack([
            np.bincount(point_index, weights=eb[:, i] * free_weight,
                        minlength=n_pt)
            for i in range(3)
        ], axis=1)

        improved = False
        for _attempt in range(6):
            Ud = U.copy()
            Vd = V.copy()
            idx = np.arange(6)
            Ud[:, idx, idx] += lam * np.maximum(U[:, idx, idx], 1e-12)
            idx3 = np.arange(3)
            Vd[:, idx3, idx3] += lam * np.maximum(V[:, idx3, idx3], 1e-12)
            full_Vinv = np.zeros((n_pt, 3, 3))
            try:
                full_Vinv[usable_point] = np.linalg.inv(Vd[usable_point])
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue

            # Schur complement: S = U - sum_j W_ij V_j^-1 W_kj^T, over
            # observation PAIRS sharing a landmark. Two cameras that
            # share one acquire an off-diagonal block, which is exactly
            # the covisibility structure.
            WV = np.einsum("mij,mjk->mik", W, full_Vinv[point_index])
            contribution = np.einsum("pij,pkj->pik", WV[rows_a], W[rows_b])
            S = -np.bincount(
                pair_flat, weights=contribution.reshape(-1),
                minlength=size * size,
            ).reshape(size, size)
            take = np.einsum("mij,mj->mi", WV[live_rows], g_pt[live_points])
            rhs = g_cam[free] - np.stack([
                np.bincount(live_slots, weights=take[:, i], minlength=len(free))
                for i in range(6)
            ], axis=1)
            for a, camera in enumerate(free):
                span = slice(6 * a, 6 * a + 6)
                S[span, span] += Ud[camera]

            try:
                delta_cam = np.linalg.solve(S, rhs.reshape(-1))
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            delta_cam = delta_cam.reshape(len(free), 6)
            full_delta = np.zeros((n_cam, 6))
            full_delta[free] = delta_cam

            # Back-substitute the landmarks.
            pushed = np.einsum("mij,mi->mj", W, full_delta[camera_index])
            correction = np.stack([
                np.bincount(point_index, weights=pushed[:, i] * free_weight,
                            minlength=n_pt)
                for i in range(3)
            ], axis=1)
            delta_pt = np.einsum("nij,nj->ni", full_Vinv, g_pt - correction)
            delta_pt[point_is_fixed] = 0.0

            trial_rotations = rotations.copy()
            trial_translations = translations.copy()
            trial_rotations[free] = _exp_so3(delta_cam[:, :3]) @ rotations[free]
            trial_translations[free] = translations[free] + delta_cam[:, 3:]
            trial_points = points + delta_pt

            trial_residual, tjp, tjq, trial_valid = _residuals_and_terms(
                trial_rotations, trial_translations, trial_points,
                camera_index, point_index, observed, camera_matrix)
            trial_weight = _huber_weights(trial_residual, huber_delta)
            trial_cost = float(
                (trial_weight[:, None] * trial_residual ** 2).sum()
            )
            if trial_cost < cost:
                rotations = trial_rotations
                translations = trial_translations
                points = trial_points
                residual, jac_pose, jac_point, valid = (
                    trial_residual, tjp, tjq, trial_valid)
                weight = trial_weight
                cost = trial_cost
                lam = max(lam * 0.3, 1e-9)
                improved = True
                taken += 1
                break
            lam *= 10.0
        if not improved:
            break

    after = rms(residual, valid)

    # WHICH ADJUSTED LANDMARKS A WORLD MAY STILL PUBLISH.
    #
    # An under-constrained landmark slides along its viewing ray at
    # almost no cost in reprojection -- that is the classic degenerate
    # direction of a bundle problem, and it is exactly what a two-view
    # point has. So the optimiser can leave a point reprojecting
    # beautifully and sitting a hundred metres behind the wall, and the
    # publication gate that would have caught it ran at CREATION time and
    # is not re-run. Measured on the pinned eight-capture corpus, that
    # took the worst bbox blowup -- full extent over the p2-p98 core --
    # from 11.0 to 35.2 while every reprojection statistic improved.
    #
    # So the caller is told which landmarks still pass, on exactly the
    # terms they were admitted by: every observation in front of the
    # camera and within the same `huber_delta` the pose solve used. A
    # landmark with no observations here is reported True and left alone;
    # this can only ever DEMOTE, never promote.
    distance = np.linalg.norm(residual, axis=1)
    ok_row = valid & (distance <= huber_delta)
    point_ok = np.ones(n_pt, dtype=bool)
    np.logical_and.at(point_ok, point_index, ok_row)
    # The two reasons are reported as MASKS, not counts. A count here
    # would be over the whole window and would include landmarks that
    # were already unpublishable, so the caller -- which is the only
    # thing that knows what was publishable a moment ago -- could not
    # tell an actual demotion from a restatement. Counting the wrong one
    # broke the manifest's `published + refused == triangulated` identity
    # by 3,069 points on the 2026-08-29 normal walk.
    failed_reprojection = ~point_ok
    failed_parallax = np.zeros(n_pt, dtype=bool)

    # AND THE PARALLAX, WHICH IS THE HALF REPROJECTION CANNOT SEE.
    #
    # A landmark's degenerate direction is along its own viewing ray, so
    # it can slide arbitrarily far out at almost no cost in reprojection.
    # Demoting on pixel error alone therefore catches only some of it:
    # measured on the pinned corpus, the reprojection test took one
    # capture's worst within-segment bbox blowup from 219.1 to 5.1 and
    # left another at 87.4.
    #
    # The angle the observing camera centres subtend AT the landmark is
    # the quantity that does see it, and the floor is not a new number --
    # `geometry.min_parallax_deg` is the same bound `landmark_gate`
    # already applies at creation, derived from the focal length as
    # sigma_px / f. Below it a landmark's distance is set by pixel noise
    # rather than by geometry, which is exactly what a point that slid
    # along its ray now is.
    #
    # Computed over the same observation PAIRS the Schur complement is
    # built from, so it costs one more pass over a list that already
    # exists.
    if min_parallax_deg is not None and total_pairs:
        point_of_pair = point_index[rows_a]
        seen = np.zeros(n_pt, dtype=bool)
        seen[point_of_pair] = True

        def widest_angle(where, poses_rotations, poses_translations):
            centres = np.einsum(
                "nij,nj->ni",
                np.transpose(poses_rotations, (0, 2, 1)),
                -poses_translations,
            )
            ray_a = where[point_of_pair] - centres[camera_index[rows_a]]
            ray_b = where[point_of_pair] - centres[camera_index[rows_b]]
            norm_a = np.linalg.norm(ray_a, axis=1)
            norm_b = np.linalg.norm(ray_b, axis=1)
            usable = (norm_a > 1e-12) & (norm_b > 1e-12)
            cosine = np.ones(len(ray_a))
            cosine[usable] = np.clip(
                np.einsum("ij,ij->i", ray_a[usable], ray_b[usable])
                / (norm_a[usable] * norm_b[usable]),
                -1.0, 1.0,
            )
            widest = np.zeros(n_pt)
            np.maximum.at(widest, point_of_pair, np.degrees(np.arccos(cosine)))
            return widest

        # BEFORE AND AFTER, and only the DIFFERENCE is actionable.
        #
        # Demoting on the absolute angle was measured and refused: it
        # removes 29% of the corpus's points, because plenty of landmarks
        # sit below this bound honestly and `landmark_gate` already ruled
        # on them at creation with the same number. An adjustment has no
        # standing to re-litigate that.
        #
        # What it does have standing to revoke is what IT broke: a
        # landmark whose observing rays subtended a usable angle before
        # this adjustment, do not after, AND lost most of that angle in
        # the process has been slid along its own ray -- which is the
        # failure this test exists for and the one reprojection cannot
        # see. See PARALLAX_COLLAPSE_RATIO for why the crossing alone is
        # not enough.
        before_angle = widest_angle(
            np.array(points_before, dtype=np.float64), rotations_before,
            translations_before)
        after_angle = widest_angle(points, rotations, translations)
        broke_it = (
            seen
            & (before_angle >= min_parallax_deg)
            & (after_angle < min_parallax_deg)
            & (after_angle < before_angle * PARALLAX_COLLAPSE_RATIO)
        )
        # Counted BEFORE the merge, and only for points reprojection had
        # not already refused, so a point that fails both is attributed
        # once. The engine folds these into the SAME two buckets
        # `landmark_gate` uses, which is what keeps the manifest's
        # `published + refused == triangulated` identity closing.
        failed_parallax = broke_it & ~failed_reprojection
        point_ok &= ~broke_it

    return rotations, translations, points, {
        "iterations": taken,
        "reprojection_rms_before": round(before, 5),
        "reprojection_rms_after": round(after, 5),
        "improved": bool(after < before),
        "landmarks_adjusted": int(usable_point.sum()),
        "landmarks_fixed": int(point_is_fixed.sum()),
        "cameras_free": int((~frozen).sum()),
        "observations": int(len(observed)),
        "point_ok": point_ok,
        "failed_reprojection": failed_reprojection,
        "failed_parallax": failed_parallax,
    }
