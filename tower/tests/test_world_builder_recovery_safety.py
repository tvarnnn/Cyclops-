"""Bounded recovery, attacked. What it costs, and where it lies.

WHY THIS FILE EXISTS

`tests/test_world_builder_tracking_recovery.py` pins that recovery
HAPPENS: one refused keyframe no longer ends a coordinate frame, a
sustained run of refusals still does, and no acceptance constant moved.
Every one of those is a statement about control flow.

None of them is a statement about GEOMETRY. A recovery that carried the
walk forward on poses that are wrong would satisfy every assertion in
that file, and would be exactly the failure this project's epistemics
exist to prevent: coherence bought with poses a consumer cannot tell
from true ones. So this file asks the other question, against the
synthetic scenes' exact ground truth:

    is a pose solved ACROSS a recovery as true as one solved without?

The answer is three-part, and the tests below are ordered to match.

1. WHEN THE SCENE IS RIGID AND UNIQUELY TEXTURED, RECOVERY IS FREE.
   Measured on a 14-keyframe 12 cm strafe with one keyframe replaced by
   noise: the Umeyama-aligned trajectory RMS is 1.50 cm with the refusal
   and 1.50 cm without it, and the per-camera rotation error AFTER the
   refusal is lower, not higher (2.62 deg against 3.20 deg at the last
   keyframe), because stepping over a keyframe widens the baseline the
   next PnP is solved on.

2. THE ERROR OF THE ONE SOLVE GROWS WITH THE REFERENCE GAP, AND NOTHING
   REFUSES. Holding the target keyframe fixed and moving only its
   reference, the relative rotation error of that single solve rises
   monotonically with the gap and the acceptance gate never fires.
   Measured over six scene seeds, median relative rotation error in
   degrees, forward walk and sideways strafe at a 9 cm step:

       gap    1     2     3     4     6     8    10    12    14
       fwd  0.78  0.86  1.33  2.48  3.04  3.38  4.47  6.09  4.39
       str  1.57  2.71  4.55  5.86  9.10 11.89 14.91 19.41 21.36

   Zero refusals at any gap, with 22-250 PnP inliers throughout -- so
   the 12-correspondence / 3 px gate does not see this and cannot be
   asked to. `MAX_RECOVERY_KEYFRAMES` is the only thing bounding it.

3. OVER REPEATING TEXTURE, A GAP IS A LIE. This is the one genuinely
   unsafe result, and it needs no teleport, no covered lens and no
   adversary. On a room whose texture repeats every 1.5 m, walking
   0.1875 m per keyframe, with the target keyframe and the step held
   FIXED and only the gap varied. Nine samples per gap -- three room
   seeds by three refusal seeds -- of the error between the motion the
   camera made and the motion the reconstruction published:

       gap            1      2      3      4      6      8
       walked      0.188  0.375  0.562  0.750  1.125  1.500   m
       median err  0.007  0.207  0.241  0.411  0.766  1.499   m
       worst err   0.023  0.231  0.493  0.557  0.773  1.499   m
       over 10 cm    0/9    6/9    9/9    6/6    9/9    9/9
       refused       0/9    0/9    0/9    3/9    0/9    0/9

   Read the error column against the walked column and the mechanism is
   plain: whatever the gap, the solve publishes roughly ONE keyframe of
   motion. It is not drifting, it is matching the wrong repetition, and
   the keyframes the camera crossed while unmatched simply cease to have
   happened.

   At gap 8 the solve reports 1 millimetre of motion where the camera
   moved 1.500 m, with 169 PnP inliers, 0.14 deg of rotation error, and
   published support that reprojects at 0.22 px median. Every internal
   instrument this pipeline owns says that pose is excellent. It is
   1.5 m wrong.

   This is not a pre-existing defect that recovery merely exposes. At
   gap 1, on the same scene and the same trajectory, the same solver
   recovers the step to within 2.3 cm in all nine samples. The gap is the
   cause, and only bounded recovery can produce a gap.

WHAT THAT IMPLIES FOR THE BOUND, since these tests are the evidence for
it. `MAX_RECOVERY_KEYFRAMES` is precisely the largest reference gap a
solve may be admitted at, so it is the only knob that bounds any of the
above. Gap 1 is free everywhere measured. On unique texture gaps of 2
and 3 cost little. Over repeating texture NO gap above 1 was reliably
correct -- gap 2 already loses 20 cm in six of nine samples -- so no
value above 1 can be defended on geometry alone, and 8 is at the far end
of the trade rather than in the middle of it.

That is not an argument for restoring the latch, which was wrong for
1,812 keyframes in the corpus. It is an argument that the bound is doing
work no measurement here supports it doing, and that the real fix is an
instrument recovery does not currently have: nothing checks that the
displacement a recovered pose implies is consistent with the number of
keyframes it skipped. A gap of 8 that reports one keyframe of motion is
detectable without any new appearance model.

THE PUBLISHED SUPPORT DEGRADES SEGMENT-WIDE, NOT JUST AT THE SEAM.
Fraction of published support rows over the 3 px gate, same walk, only
the number of consecutively refused keyframes varying:

       refusals        0       1       3       7
       reference gap   -       2       4       8
       recovered kf    -    0.35%   5.64%   4.94%
       the rest     0.28%   0.31%   3.60%   2.10%

   At gap 2 the table is indistinguishable from the clean control. At
   gaps 4 and 8 the whole segment's over-gate rate rises roughly tenfold
   and the recovered keyframe carries 1.5-2.4 times the segment's own
   rate. So the cost of a wide gap is not confined to the keyframe that
   paid it.

WHERE THE ERROR WENT. Cross-segment registration refuses a false merge
structurally (tests/test_world_registration.py) because every clause it
owns compares two segments. The failure above puts BOTH SIDES of the lie
inside ONE segment, where no clause reaches. Bounded recovery does not
weaken registration; it moves error out of registration's jurisdiction.

SYNTHETIC, NOT PHYSICAL. Rendered rooms with perfect optics, no rolling
shutter and no compression say nothing about the Ray-Ban camera. Their
value is that the camera poses are inputs, so these are exact answers
rather than plausible ones.
"""

import statistics
from functools import lru_cache

import cv2
import numpy as np
import pytest

from tests import synthetic_scene as ss
from tower.world_builder import geometry
from tower.world_builder.backend import KeyframeInput
from tower.world_builder.backends import classical
from tower.world_builder.backends.classical import ClassicalTwoViewBackend
from tower.world_builder.geometry import detect_and_describe
from tower.world_builder.records import CameraIntrinsics
from tower.world_builder.schema import (
    POSE_STATUS_ANCHOR,
    POSE_STATUS_SOLVED,
    POSE_STATUS_UNAVAILABLE,
)

WIDTH, HEIGHT = 480, 360
POSED = (POSE_STATUS_ANCHOR, POSE_STATUS_SOLVED)

# The texture period of `_periodic_room`, in metres. Named because three
# tests below choose a step or a teleport distance to line up with it,
# and a bare 1.5 in those places would read as an arbitrary number.
PERIOD_M = ss.ROOM_WIDTH_M / 4


def _camera():
    return ss.camera_matrix(WIDTH, HEIGHT)


def _intrinsics(camera):
    return CameraIntrinsics(
        source="self_calibrated",
        fx=float(camera[0][0]),
        fy=float(camera[1][1]),
        cx=float(camera[0][2]),
        cy=float(camera[1][2]),
        calibrated_width=WIDTH,
        calibrated_height=HEIGHT,
    )


def _frames(images, prefix="kf"):
    return [
        KeyframeInput(
            keyframe_id=f"{prefix}{index}",
            image_gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
            image_bgr=image,
        )
        for index, image in enumerate(images)
    ]


def _noise(keyframe_id, seed):
    """Texture with no relationship to the scene.

    Deliberately not black. A featureless frame dies at ORB detection,
    which is an easier and different refusal from "a thousand features,
    none of them yours" -- and the second is what a wearer passing a
    window or a blank wall actually hands the solver.
    """
    rng = np.random.default_rng(seed)
    gray = rng.integers(0, 255, (HEIGHT, WIDTH), dtype=np.uint8)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return KeyframeInput(
        keyframe_id=keyframe_id,
        image_gray=gray,
        image_bgr=cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
    )


def _occluded(keyframe_id, seed):
    """A palm over the lens: dark, low contrast, still faintly textured."""
    rng = np.random.default_rng(seed)
    gray = rng.integers(20, 55, (HEIGHT, WIDTH), dtype=np.uint8)
    return KeyframeInput(
        keyframe_id=keyframe_id,
        image_gray=cv2.GaussianBlur(gray, (5, 5), 0),
        image_bgr=cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
    )


# THE SHIPPED BUDGET IS 1, AND THIS FILE IS WHY.
#
# Every measurement below is a statement about a reference GAP, and
# `MAX_RECOVERY_KEYFRAMES` is exactly the largest gap production will
# admit. Setting it to 1 -- which the evidence in this file is what
# bought -- makes every gap above 1 unreachable, so a file that inherited
# the shipped value would quietly stop measuring anything the day the
# bound was tightened, and the evidence for the bound would evaporate
# along with the danger.
#
# So the budget is raised here, for this file only, and the tests
# manufacture the gap they are about. The tests that want the LATCH set
# it to 1 themselves and restore what they found.
#
# If the bound is ever raised again, this file is the thing that has to
# be re-run first, and its tables are the numbers to beat.
RECOVERY_BUDGET_UNDER_TEST = 999

# The most permissive budget this file's evidence ever considered
# defensible, and the one it rejected. Used by the tests that need the
# chain to actually BREAK, where the claim is "even at the loosest bound
# anybody proposed, this is still refused".
FINITE_BUDGET_UNDER_TEST = 8


@pytest.fixture(autouse=True)
def _exercise_the_mechanism(monkeypatch):
    monkeypatch.setattr(
        classical, "MAX_RECOVERY_KEYFRAMES", RECOVERY_BUDGET_UNDER_TEST
    )


def _run(frames, camera):
    backend = ClassicalTwoViewBackend()
    backend.begin(_intrinsics(camera))
    return [backend.extend(frame) for frame in frames], backend


# -- ground truth ------------------------------------------------------
#
# The backend reconstructs in the FIRST keyframe's camera frame, so
# ground truth has to be expressed there before anything can be compared.
# `CameraPose` is T_world_camera -- its translation IS the camera position
# -- while a solved pose is world-to-camera, which is why the centre below
# is -R.T @ t and never the translation itself. Getting that backwards
# produces errors that look like plausible drift rather than like a bug,
# which is the trap `synthetic_scene.relative_pose` documents.


def _truth_in_frame0(poses):
    rotation0, position0 = poses[0].rotation, poses[0].position
    return [
        (pose.rotation.T @ rotation0, rotation0.T @ (pose.position - position0))
        for pose in poses
    ]


def _centre(pose):
    rotation = np.asarray(pose.rotation, dtype=np.float64)
    translation = np.asarray(pose.translation, dtype=np.float64).reshape(3)
    return -rotation.T @ translation


def _measured(steps, poses):
    """Per-keyframe status, rotation error, direction error and centre."""
    truth = _truth_in_frame0(poses)
    rows = {}
    for index, step in enumerate(steps):
        estimate = step.pose
        if estimate.status == POSE_STATUS_ANCHOR:
            rows[index] = {
                "status": POSE_STATUS_ANCHOR,
                "rotation": np.eye(3),
                "rotation_deg": 0.0,
                "direction_deg": 0.0,
                "centre": np.zeros(3),
                "inliers": 0,
            }
            continue
        if estimate.status != POSE_STATUS_SOLVED:
            rows[index] = {
                "status": estimate.status,
                "degeneracy": estimate.degeneracy,
            }
            continue
        rotation = np.asarray(estimate.rotation, dtype=np.float64)
        centre = _centre(estimate)
        rows[index] = {
            "status": POSE_STATUS_SOLVED,
            "rotation": rotation,
            "centre": centre,
            "rotation_deg": ss.rotation_error_deg(rotation, truth[index][0]),
            "direction_deg": ss.direction_error_deg(centre, truth[index][1]),
            "inliers": int(estimate.inliers or 0),
        }
    return rows


def _trajectory_rms_cm(rows, poses):
    """Umeyama-aligned camera-centre error, in centimetres of ground truth.

    A monocular reconstruction is only ever correct up to a similarity, so
    comparing raw coordinates would fail a perfectly correct answer. The
    scale that alignment recovers is also what converts the arbitrary
    reconstruction unit back into metres, which is the only reason a
    number here can be read as a distance at all.
    """
    truth = _truth_in_frame0(poses)
    indices = [index for index, row in rows.items() if row["status"] in POSED]
    assert len(indices) >= 3, "too few poses to align against ground truth"
    source = np.array([rows[i]["centre"] for i in indices], dtype=np.float64)
    target = np.array([truth[i][1] for i in indices], dtype=np.float64)
    scale, rotation, translation = ss.umeyama_similarity(source, target)
    mapped = (scale * (rotation @ source.T)).T + translation
    error = np.linalg.norm(mapped - target, axis=1)
    return float(np.sqrt((error**2).mean()) * 100.0)


def _metres_moved(rows, poses, reference, target):
    """(published, actual) distance between two keyframes, both in metres.

    The published number is in the reconstruction's arbitrary unit, so it
    is converted by the similarity scale fitted on the part of the walk
    BEFORE the gap -- the part whose correctness is not in question.
    Fitting the scale over the whole trajectory would let the very error
    under test set the ruler it is measured with, which would shrink the
    reported error towards zero exactly when it is largest.
    """
    truth = _truth_in_frame0(poses)
    prefix = [index for index in range(reference + 1) if rows[index]["status"] in POSED]
    assert len(prefix) >= 3, "no clean prefix to fit the scale on"
    scale, _, _ = ss.umeyama_similarity(
        np.array([rows[i]["centre"] for i in prefix], dtype=np.float64),
        np.array([truth[i][1] for i in prefix], dtype=np.float64),
    )
    published = float(
        np.linalg.norm(rows[target]["centre"] - rows[reference]["centre"]) * scale
    )
    actual = float(np.linalg.norm(truth[target][1] - truth[reference][1]))
    return published, actual


# -- scenes ------------------------------------------------------------


def _periodic_room(seed=5, tiles=4):
    """A room whose floor, ceiling and end walls repeat `tiles` times in x.

    Period = ROOM_WIDTH_M / tiles metres. Two camera positions one period
    apart therefore see the same structure, which is the case where a
    descriptor match cannot possibly be evidence and can very easily be
    confident. Real rooms do this constantly: tiled floors, panelled
    corridors, a row of identical desks, a bookshelf.

    The side walls are deliberately NOT tiled. With every surface periodic
    the scene is globally ambiguous and a refusal would be the only
    correct answer, which would make the tests below measure the scene
    rather than the solver. Leaving two unique surfaces means a correct
    answer exists and is even visible -- the camera can see a side wall --
    so a wrong one is a real error rather than an impossible question.
    """
    rng = np.random.default_rng(seed)
    width, depth, height = ss.ROOM_WIDTH_M, ss.ROOM_DEPTH_M, ss.ROOM_HEIGHT_M

    def tiled(a, b):
        return np.tile(ss.make_texture(rng, a, b), (1, tiles, 1))

    half = width / 2
    return [
        ss.Plane(
            np.array([-half, 0.0, 0.0]),
            np.array([width, 0.0, 0.0]),
            np.array([0.0, 0.0, depth]),
            tiled(128, 384),
        ),
        ss.Plane(
            np.array([-half, -height, 0.0]),
            np.array([width, 0.0, 0.0]),
            np.array([0.0, 0.0, depth]),
            tiled(128, 384),
        ),
        ss.Plane(
            np.array([-half, -height, depth]),
            np.array([width, 0.0, 0.0]),
            np.array([0.0, height, 0.0]),
            tiled(128, 256),
        ),
        ss.Plane(
            np.array([-half, -height, 0.0]),
            np.array([width, 0.0, 0.0]),
            np.array([0.0, height, 0.0]),
            tiled(128, 256),
        ),
        ss.Plane(
            np.array([-half, -height, 0.0]),
            np.array([0.0, 0.0, depth]),
            np.array([0.0, height, 0.0]),
            ss.make_texture(rng, 384, 256),
        ),
        ss.Plane(
            np.array([half, -height, 0.0]),
            np.array([0.0, 0.0, depth]),
            np.array([0.0, height, 0.0]),
            ss.make_texture(rng, 384, 256),
        ),
    ]


@lru_cache(maxsize=None)
def _walk(scene, motion, count, step, start):
    """A rendered walk plus its ground-truth poses, cached.

    Ray-traced rendering of a 20-keyframe room costs seconds, and several
    tests below want the SAME pixels so that a difference in the answer is
    attributable to the backend rather than to a re-render. Returned as a
    tuple because `lru_cache` requires hashable arguments and a caller
    that mutated a shared list would corrupt every later test.
    """
    planes = _periodic_room() if scene == "periodic" else ss.furnished_room()
    if motion == "forward":
        poses = ss.forward_walk(count, step=step, start=start)
    else:
        poses = ss.strafe(count, step=step, start=start)
    # A keyframe measured with the camera against a wall is not a
    # measurement of the algorithm; it is a measurement of running out of
    # scene, and it looks exactly like accumulated drift.
    outside = ss.poses_outside_room(poses)
    assert not outside, f"keyframes {outside} are too close to a surface to trust"
    return tuple(ss.render_sequence(planes, poses, _camera(), WIDTH, HEIGHT)), poses


def _refuse_before(frames, target, gap):
    """Replace the gap-1 keyframes before `target` with unrelated texture.

    This is how a reference gap is manufactured without touching the
    target keyframe or its ground-truth pose: the images at `target` and
    at `target - gap` are byte-identical across every gap, so a change in
    the answer is attributable to the reference distance and to nothing
    else. Varying the target instead would confound the gap with where in
    the room the camera happened to be.
    """
    out = list(frames)
    for index in range(target - gap + 1, target):
        out[index] = _noise(f"noise{index}", seed=900 + index)
    return out


def _reprojection_by_keyframe(frames, steps, backend):
    """Published support rows reprojected through the pose that published them.

    The same arithmetic as
    `scripts/world_coherence_report.reprojection_by_segment`, attributed
    PER KEYFRAME rather than per segment, because the question here is
    whether recovery-solved keyframes specifically are carrying the bad
    rows -- which a per-segment median cannot answer. ORB is re-detected
    on the stored image exactly as `world_registration.read_segments`
    does, which is what makes a row's (frame, feature) index resolvable
    back to a pixel at all.

    Rows whose landmark falls behind the camera are dropped rather than
    counted as a large error: a negative depth has no meaningful pixel
    distance, and folding one in would flatter or damn the result
    arbitrarily.
    """
    snapshot = backend.snapshot()
    assert snapshot.points is not None, "nothing was published"
    xyz = np.asarray(snapshot.points.xyz, dtype=np.float64)
    keypoints = [detect_and_describe(frame.image_gray)[0] for frame in frames]
    camera = _camera()
    poses = {}
    for index, step in enumerate(steps):
        estimate = step.pose
        if estimate.status == POSE_STATUS_ANCHOR:
            poses[index] = (np.eye(3), np.zeros(3))
        elif estimate.status == POSE_STATUS_SOLVED:
            poses[index] = (
                np.asarray(estimate.rotation, dtype=np.float64),
                np.asarray(estimate.translation, dtype=np.float64).reshape(3),
            )
    errors: dict[int, list[float]] = {}
    for frame, feature, landmark in snapshot.points.support_views:
        frame, feature, landmark = int(frame), int(feature), int(landmark)
        if frame not in poses or landmark >= len(xyz):
            continue
        if feature >= len(keypoints[frame]):
            continue
        rotation, translation = poses[frame]
        in_camera = rotation @ xyz[landmark] + translation
        if not np.isfinite(in_camera).all() or in_camera[2] <= 0:
            continue
        projected = camera @ in_camera
        x, y = keypoints[frame][feature].pt
        errors.setdefault(frame, []).append(
            float(
                np.hypot(
                    projected[0] / projected[2] - x,
                    projected[1] / projected[2] - y,
                )
            )
        )
    return errors


# -- 1. recovery over unique texture is free ---------------------------


class TestARecoveryCostsNothingOnRigidUniqueTexture:
    """The claim bounded recovery is sold on, checked against ground truth.

    If stepping over a refused keyframe degraded the poses that follow,
    the change would be buying segment continuity with accuracy -- and
    every existing recovery test would still pass, because they only ever
    look at pose STATUS.
    """

    REFUSED = 5
    COUNT = 14

    def _both(self):
        images, poses = _walk("furnished", "strafe", self.COUNT, 0.12, (0.0, -1.6, 0.6))
        camera = _camera()
        clean, _ = _run(_frames(images), camera)
        broken = list(_frames(images))
        broken[self.REFUSED] = _noise("kf5", seed=7)
        recovered, _ = _run(broken, camera)
        return _measured(clean, poses), _measured(recovered, poses), poses

    def test_the_injected_keyframe_is_still_refused(self):
        """Recovery is retrying, not leniency. If this keyframe ever starts
        solving, the bar moved and every other number here is meaningless.
        """
        clean, recovered, _ = self._both()
        assert clean[self.REFUSED]["status"] == POSE_STATUS_SOLVED
        assert recovered[self.REFUSED]["status"] == POSE_STATUS_UNAVAILABLE

    def test_poses_after_the_refusal_are_no_worse_than_without_it(self):
        """Per-camera rotation error against ground truth, keyframe by
        keyframe.

        Measured: recovery is slightly BETTER at every keyframe after the
        refusal (2.62 deg against 3.20 at the last one), because stepping
        over a keyframe widens the baseline the PnP is solved on. The
        assertion allows a full degree of slack in the other direction so
        that it fails on a real regression rather than on solver jitter.
        """
        clean, recovered, _ = self._both()
        for index in range(self.REFUSED + 1, self.COUNT):
            assert recovered[index]["status"] == POSE_STATUS_SOLVED, (
                f"keyframe {index} stopped solving after the refusal"
            )
            assert recovered[index]["rotation_deg"] <= (
                clean[index]["rotation_deg"] + 1.0
            ), (
                f"keyframe {index}: recovery cost "
                f"{recovered[index]['rotation_deg']:.2f} deg of rotation error "
                f"against {clean[index]['rotation_deg']:.2f} deg without it"
            )

    def test_the_recovered_trajectory_matches_the_clean_one(self):
        """Umeyama-aligned RMS against ground truth: measured 1.50 cm both
        with and without the refusal.

        The bound on the difference is generous because the quantity under
        test is a difference and not an absolute; the separate bound on
        the control exists so that a walk which drifted for unrelated
        reasons cannot make the comparison pass by making both sides bad.
        """
        clean, recovered, poses = self._both()
        clean_rms = _trajectory_rms_cm(clean, poses)
        recovered_rms = _trajectory_rms_cm(recovered, poses)
        assert clean_rms < 3.0, f"the control walk itself drifted {clean_rms:.2f} cm"
        assert recovered_rms <= clean_rms + 1.5, (
            f"recovery cost {recovered_rms:.2f} cm RMS against {clean_rms:.2f} cm "
            f"without it"
        )


# -- 2. the gap is the cost, and the gate cannot see it ----------------


def test_solve_error_grows_with_the_reference_gap_and_the_gate_never_fires():
    """How far back `references[0]` may sit is a geometry decision, and
    nothing in the acceptance path is measuring it.

    The target keyframe and its image are IDENTICAL across both gaps --
    only the reference moves -- so this isolates the error of the single
    PnP that the gap changed. Both quantities are expressed relative to
    the reference camera, so drift accumulated BEFORE the reference
    cancels out of each side and what is left is the one solve.

    Measured over six scene seeds on a forward walk, median relative
    rotation error of that solve: 0.78 deg at gap 1 rising to 3.38 at
    gap 8 and 6.09 at gap 12, with ZERO refusals and 22-194 inliers. The
    12-correspondence / 3 px gate is not a function of the gap and cannot
    be made into one, so `MAX_RECOVERY_KEYFRAMES` is the only bound that
    exists on this.

    Two quantities are asserted, because they fail differently. The
    CORRESPONDENCE COUNT is the robust one -- it falls monotonically with
    the gap on every seed measured (167, 158, 142, 107, 54, 47 at gaps
    1, 2, 3, 4, 6, 8 here) and never crosses the gate, which is precisely
    the shape of a signal the gate cannot act on. The ROTATION ERROR is
    noisier on a single seed than the six-seed medians above suggest --
    0.43, 1.60, 3.78, 1.55 at gaps 1, 4, 6, 8 -- so it is asserted as a
    worst case over the wide gaps rather than gap by gap, which is what
    the bound has to survive anyway.

    The failure this catches: a later change that widens the reach on the
    argument that the solve is gap-insensitive. It is not, and the bound
    is resting on that fact.
    """
    images, poses = _walk("furnished", "forward", 20, 0.12, (0.0, -1.6, 0.6))
    camera = _camera()
    truth = _truth_in_frame0(poses)
    target = 14
    errors, inliers = {}, {}
    for gap in (1, 4, 6, 8):
        steps, _ = _run(_refuse_before(_frames(images), target, gap), camera)
        rows = _measured(steps, poses)
        reference = target - gap
        assert rows[target]["status"] == POSE_STATUS_SOLVED, (
            f"gap {gap} refused; this test is about what the gate ADMITS"
        )
        assert rows[reference]["status"] in POSED
        assert rows[target]["inliers"] >= classical.MIN_PNP_CORRESPONDENCES, (
            f"gap {gap} was admitted on "
            f"{rows[target]['inliers']} inliers, below the gate itself"
        )
        inliers[gap] = rows[target]["inliers"]
        errors[gap] = ss.rotation_error_deg(
            rows[target]["rotation"] @ rows[reference]["rotation"].T,
            truth[target][0] @ truth[reference][0].T,
        )

    assert inliers[1] > 2 * inliers[8], (
        f"the evidence a wide gap is solved on should be far thinner than "
        f"at gap 1; measured {inliers[1]} against {inliers[8]}"
    )
    assert errors[1] < 1.5, (
        f"the gap-1 control is already {errors[1]:.2f} deg wrong; the "
        f"comparison below means nothing until that is understood"
    )
    worst = max(errors[gap] for gap in (4, 6, 8))
    assert worst > 2.0, (
        f"the widest gaps solved to at most {worst:.2f} deg, better than "
        f"anything measured. If the solve really has become "
        f"gap-insensitive, re-derive MAX_RECOVERY_KEYFRAMES from the new "
        f"numbers rather than deleting this test"
    )


# -- 3. over repeating texture, a gap erases real motion ---------------


class TestAGapOverRepeatingTextureErasesRealMotion:
    """THE UNSAFE RESULT. A room that repeats every 1.5 m, a camera walking
    0.1875 m per keyframe, and one stretch of refused keyframes.

    No teleport, no covered lens, no adversary -- an ordinary continuous
    walk over a tiled floor. When the gap happens to span one texture
    period, the descriptor match that PnP is solved from is not merely
    noisy: it is confidently answering a different question. "Where were
    you, given that you see this?" has more than one true answer here, and
    the solver holds no instrument that could prefer the right one.

    The control is what makes this a statement about the GAP rather than
    about the scene: at gap 1, same room, same trajectory, same solver,
    the step is recovered to 3 millimetres.
    """

    STEP = PERIOD_M / 8
    COUNT = 20
    START = (-2.4, -1.6, 0.6)
    TARGET = 12

    def _solve(self, gap):
        images, poses = _walk("periodic", "strafe", self.COUNT, self.STEP, self.START)
        frames = _refuse_before(_frames(images), self.TARGET, gap)
        steps, backend = _run(frames, _camera())
        return _measured(steps, poses), poses, steps, backend, frames

    def test_at_gap_one_the_same_scene_is_reconstructed_correctly(self):
        """The control. Repeating texture alone does not break this
        pipeline -- measured 0.194 m published against 0.188 m walked, and
        within 2.3 cm across all nine seed combinations swept."""
        rows, poses, _, _, _ = self._solve(1)
        assert rows[self.TARGET]["status"] == POSE_STATUS_SOLVED
        published, actual = _metres_moved(rows, poses, self.TARGET - 1, self.TARGET)
        assert abs(published - actual) < 0.05, (
            f"the gap-1 control is itself wrong by {abs(published - actual):.3f} m; "
            f"nothing below is interpretable until that is understood"
        )

    def test_at_gap_eight_a_metre_and_a_half_of_walking_is_published_as_centimetres(
        self,
    ):
        """The defect, at the largest gap `MAX_RECOVERY_KEYFRAMES` admits.

        Measured: 1.500 m walked, 0.001 m published, 169 PnP inliers,
        0.14 deg of rotation error, and the same 1.499 m error in all
        nine room-seed by refusal-seed samples swept -- this is the
        deterministic behaviour of the solver on this input, not a tail.
        Gap 8 is reachable on the shipped constant -- seven consecutive
        refusals is one short of the budget -- so this needs no
        monkeypatch to provoke, which is the whole point.

        IF THIS TEST STARTS FAILING BECAUSE THE ERROR SHRANK, the defect
        was fixed and that is good news: re-point the assertion at the new
        magnitude rather than deleting it, and lower
        `MAX_RECOVERY_KEYFRAMES` only if the fix did not remove the need.
        """
        rows, poses, steps, _, _ = self._solve(8)
        assert rows[self.TARGET]["status"] == POSE_STATUS_SOLVED, (
            "the solve refused, which would be the honest outcome and is not "
            "what was measured"
        )
        assert not any(step.chain_broken for step in steps), (
            "seven consecutive refusals is inside the budget, so the chain "
            "must not have broken"
        )
        published, actual = _metres_moved(rows, poses, self.TARGET - 8, self.TARGET)
        assert actual > 1.4, "the walk itself should span about one texture period"
        assert published < 0.3, (
            f"published motion {published:.3f} m against {actual:.3f} m walked; "
            f"the measured figure is 0.001 m"
        )
        assert rows[self.TARGET]["inliers"] >= 100, (
            "the point of this test is that the wrong pose looks EXCELLENT to "
            "every instrument the pipeline owns"
        )

    def test_the_published_support_of_that_wrong_pose_reprojects_beautifully(self):
        """Reprojection cannot detect this, so the coherence report cannot.

        `scripts/world_coherence_report.reprojection_by_segment` reprojects
        every published support row through the pose that published it and
        reads the result against `PNP_REPROJECTION_ERROR_PX`. That is the
        right check for a pose which does not fit its own observations. It
        is structurally blind to a pose that fits observations of the WRONG
        piece of a repeating room, because those observations do fit.

        Measured on the keyframe above: 0.22 px median against 5.54% of
        rows over the 3 px gate, where the keyframes of the SAME walk that
        are not lying sit at 3.91%. A 1.4x ratio on a few per cent is not
        a signal anybody could act on, and the median -- the number a
        reader actually looks at -- is better than the honest keyframes'.
        Meanwhile that camera is 1.5 m from where the world says it is.

        Anyone reading a coherence number as a safety number needs this on
        record. The assertion is therefore a COMPARISON, not an absolute:
        it fails if reprojection ever does start separating this pose from
        an honest one, which would be good news and would make the
        docstring wrong.
        """
        rows, _, steps, backend, frames = self._solve(8)
        assert rows[self.TARGET]["status"] == POSE_STATUS_SOLVED
        errors = _reprojection_by_keyframe(frames, steps, backend)
        target = errors[self.TARGET]
        honest = [
            value
            for frame, values in errors.items()
            if frame != self.TARGET
            for value in values
        ]
        assert len(target) > 100 and len(honest) > 1000, (
            "too few published rows to say anything"
        )
        gate = classical.PNP_REPROJECTION_ERROR_PX

        def over(values):
            return sum(1 for value in values if value > gate) / len(values)

        assert statistics.median(target) < 1.0, (
            f"median reprojection of the wrong pose is "
            f"{statistics.median(target):.2f} px"
        )
        assert over(target) <= 3 * over(honest), (
            f"{100 * over(target):.2f}% of the wrong pose's published rows "
            f"are over the gate against {100 * over(honest):.2f}% for the "
            f"honest keyframes of the same walk. If that gap has widened "
            f"into something separable, reprojection HAS become a detector "
            f"for this and the docstring above is out of date"
        )


# -- 4. covering the lens no longer cuts the segment -------------------


def test_an_occlusion_no_longer_protects_the_seam_over_repeating_texture():
    """Cover the lens, walk one texture period, uncover. The regression.

    Under the one-way latch the first occluded keyframe ended the segment,
    so the second room was never offered to a solver still holding the
    first one's map. That protection was accidental -- the latch was
    maximally pessimistic about everything -- but it was real, and bounded
    recovery removes it for any occlusion shorter than the budget.

    Measured, periodic room, 1.5 m of walking hidden behind the occlusion:

        occluded keyframes    1  2  3  4  5  6  7   8
        poses after the seam
          with the old latch  0  0  0  0  0  0  0   0
          on this branch      8  8  8  8  8  8  8   0

    and every one of those eight reports the 1.5 m jump as 0.013 m. At
    eight occluded keyframes the budget is exhausted and the chain breaks,
    which is why the last column is the honest one and why this is an
    argument about the SIZE of the budget rather than about having one.

    Both columns are measured inside one test on purpose:
    `MAX_RECOVERY_KEYFRAMES = 1` reproduces the latch exactly, so the two
    differ in that constant and in nothing else.
    """
    camera = _camera()
    step = 0.10
    before = ss.strafe(8, step=step, start=(-2.4, -1.6, 0.6))
    after = ss.strafe(8, step=step, start=(-2.4 + 7 * step + PERIOD_M, -1.6, 0.6))
    assert not ss.poses_outside_room(before + after)
    planes = _periodic_room()
    images_before = ss.render_sequence(planes, before, camera, WIDTH, HEIGHT)
    images_after = ss.render_sequence(planes, after, camera, WIDTH, HEIGHT)

    def solved_after_the_seam(occlusions):
        frames = _frames(images_before, "a")
        frames += [_occluded(f"occ{i}", 500 + i) for i in range(occlusions)]
        frames += _frames(images_after, "b")
        steps, _ = _run(frames, camera)
        first_after = len(images_before) + occlusions
        return sum(
            1
            for step in steps[first_after:]
            if step.pose.status == POSE_STATUS_SOLVED
        )

    occlusions = 3
    assert solved_after_the_seam(occlusions) >= 6, (
        "this branch should carry the walk across the occlusion; if it no "
        "longer does, the regression is gone and the table above is stale"
    )

    original = classical.MAX_RECOVERY_KEYFRAMES
    try:
        classical.MAX_RECOVERY_KEYFRAMES = 1
        assert solved_after_the_seam(occlusions) == 0, (
            "MAX_RECOVERY_KEYFRAMES = 1 is the old one-way latch, which "
            "refused every keyframe after the first occluded one"
        )
    finally:
        # Restored in a finally because this is a module global: leaking it
        # would silently reconfigure the backend for every test that runs
        # after this one in the same process.
        classical.MAX_RECOVERY_KEYFRAMES = original


def test_walking_into_an_unrelated_room_is_still_refused_at_every_occlusion(
    monkeypatch,
):
    """The scope of the result above, and the reassuring half of it.

    The failure this file reports needs the two places to LOOK alike. When
    they merely ARE different -- a second room with independent texture,
    which is what "the wearer covered the lens and walked somewhere else"
    usually means -- recovery refuses every keyframe of the new room and
    the budget then breaks the chain, which is the honest answer and the
    same one the old latch gave.

    Measured over occlusions of 0 through 12 keyframes and two scene
    pairs (independent furnished rooms; and the same room re-entered from
    the far side, turned 180 degrees): 0 of 8 second-room keyframes solved
    in every single case, with the chain breaking at the budget.

    So the danger is not "a new place", it is "a place that matches". That
    distinction matters for what a fix should be: a relocaliser or a
    retrieval step would not help here, because appearance is precisely
    what is lying. Only geometry -- how far the camera can have travelled
    for the number of keyframes it skipped -- separates these two cases.
    """
    # A FINITE budget, because the claim here is that the chain
    # BREAKS. The file-wide fixture raises the bound so the gap
    # tables can reach gap 14; these two tests need the bound to
    # exist at all.
    monkeypatch.setattr(
        classical, "MAX_RECOVERY_KEYFRAMES", FINITE_BUDGET_UNDER_TEST
    )
    camera = _camera()
    first = ss.strafe(8, step=0.12, start=(-1.0, -1.6, 0.6))
    second = ss.strafe(8, step=0.12, start=(1.0, -1.6, 0.6))
    assert not ss.poses_outside_room(first + second)
    images_first = ss.render_sequence(
        ss.furnished_room(seed=1234), first, camera, WIDTH, HEIGHT
    )
    images_second = ss.render_sequence(
        ss.furnished_room(seed=909), second, camera, WIDTH, HEIGHT
    )

    for occlusions in (0, 1, 4, 8, 12):
        frames = _frames(images_first, "a")
        frames += [_occluded(f"occ{i}", 700 + i) for i in range(occlusions)]
        frames += _frames(images_second, "b")
        steps, _ = _run(frames, camera)
        first_after = len(images_first) + occlusions
        solved = [
            index - first_after
            for index, step in enumerate(steps[first_after:], first_after)
            if step.pose.status == POSE_STATUS_SOLVED
        ]
        assert solved == [], (
            f"with {occlusions} occluded keyframes, keyframes {solved} of an "
            f"unrelated room were placed in the first room's coordinate frame"
        )
        assert any(step.chain_broken for step in steps), (
            f"with {occlusions} occluded keyframes the chain never broke, so "
            f"the engine would keep the second room in the first one's frame"
        )


# -- 5. the mechanism is what the docstring says it is -----------------


class _LeakyReferences(ClassicalTwoViewBackend):
    """The mutation: a keyframe that REFUSED becomes `references[0]` anyway.

    Production code is not touched. This subclass reaches into the chain
    AFTER the real `extend()` has run, which reproduces the broken
    invariant without changing the thing under test -- a mutation applied
    by editing the implementation would leave nothing to check it against.
    """

    def extend(self, frame):
        index = self._chain.count if self._chain else 0
        features = detect_and_describe(frame.image_gray)
        extension = super().extend(frame)
        if extension.pose.status not in POSED:
            self._chain.references.insert(0, (index, features))
            del self._chain.references[classical.EXTEND_REFERENCE_DEPTH :]
        return extension


def test_recovery_works_because_refused_keyframes_are_not_references(
    monkeypatch,
):
    """Names the mechanism, and proves the naming is not decoration.

    `classical.py` claims recovery works because a keyframe with no entry
    in `absolute` never becomes a reference -- it could not supply a
    single 3-D correspondence, so promoting it would guarantee the next
    refusal too, which is how one refusal used to become a permanent fork.
    That claim is testable: break exactly that invariant and nothing else,
    and recovery must stop working.

    Measured, one refused keyframe in a 14-keyframe strafe:

        as shipped        A S S S S . S S S S S S S S    8 solved after
        leaky references  A S S S S . . . . . . . . .    0 solved, chain broke

    So the mechanism is the reference discipline and not the failure
    counter. A future refactor that keeps the counter and loses the
    discipline would pass every test in
    test_world_builder_tracking_recovery.py while reconstructing nothing.
    """
    # A FINITE budget, because the claim here is that the chain
    # BREAKS. The file-wide fixture raises the bound so the gap
    # tables can reach gap 14; these two tests need the bound to
    # exist at all.
    monkeypatch.setattr(
        classical, "MAX_RECOVERY_KEYFRAMES", FINITE_BUDGET_UNDER_TEST
    )
    images, _ = _walk("furnished", "strafe", 14, 0.12, (0.0, -1.6, 0.6))
    frames = list(_frames(images))
    frames[5] = _noise("kf5", seed=7)
    camera = _camera()

    shipped, _ = _run(frames, camera)
    assert (
        sum(1 for step in shipped[6:] if step.pose.status == POSE_STATUS_SOLVED) >= 6
    )

    leaky = _LeakyReferences()
    leaky.begin(_intrinsics(camera))
    mutated = [leaky.extend(frame) for frame in frames]
    assert (
        sum(1 for step in mutated[6:] if step.pose.status == POSE_STATUS_SOLVED) == 0
    ), (
        "promoting a refused keyframe to a reference should starve every later "
        "solve of correspondences; if it no longer does, the explanation in "
        "classical.py for why recovery works is wrong"
    )
    assert any(step.chain_broken for step in mutated), (
        "the mutated chain should exhaust its budget and break"
    )


# -- 6. the support table stays honest where the geometry is honest ----


def test_a_wide_gap_degrades_the_published_support_of_the_whole_segment():
    """`support.json` is what cross-segment registration solves PnP against,
    so rows the solver itself would not have accepted are wrong 3-D points
    fed to the thing that decides where a segment sits in the world.

    The first thing this measures is publication DISCIPLINE at the seam:
    does the recovered keyframe publish only what its own solve accepted?
    Broadly yes. The second is the thing that was not expected and is the
    reason this test exists at all: a wide gap degrades the published
    support of the ENTIRE segment, including keyframes solved long after
    the recovery, because the landmarks a recovered keyframe triangulates
    stay in the map and every later solve draws correspondences from them.

    Measured, fraction of published rows over the 3 px gate, same walk,
    only the number of consecutively refused keyframes varying:

        refusals        0       1       3       7
        reference gap   -       2       4       8
        recovered kf    -    0.35%   5.64%   4.94%
        the rest     0.28%   0.31%   3.60%   2.10%

    Gap 2 is indistinguishable from the clean control. Gaps 4 and 8 raise
    the whole segment's over-gate rate roughly tenfold. That is the cost
    the corpus-level p99 in `classical.py`'s own docstring would show if
    it were split by gap, and it is a second, independent reason the
    budget of 8 is larger than anything measured here supports.

    NOTE THE SCOPE. None of this is evidence of CORRECTNESS. The
    repeating-texture test above shows this very measurement reading clean
    on a pose 1.5 m out of place.
    """
    images, _ = _walk("furnished", "strafe", 16, 0.12, (0.0, -1.6, 0.6))
    gate = classical.PNP_REPROJECTION_ERROR_PX

    def over(values):
        return sum(1 for value in values if value > gate) / len(values)

    def measure(refusals):
        frames = list(_frames(images))
        for index in range(5, 5 + refusals):
            frames[index] = _noise(f"kf{index}", seed=20 + index)
        steps, backend = _run(frames, _camera())
        recovered = 5 + refusals if refusals else None
        if recovered is not None:
            assert steps[recovered].pose.status == POSE_STATUS_SOLVED, (
                f"{refusals} refusals did not recover, so there is nothing "
                f"to measure"
            )
        errors = _reprojection_by_keyframe(frames, steps, backend)
        rest = [
            value
            for frame, values in errors.items()
            if frame != recovered
            for value in values
        ]
        assert len(rest) > 1000, "too few published rows to say anything"
        if recovered is None:
            return None, over(rest)
        assert len(errors[recovered]) > 100
        return over(errors[recovered]), over(rest)

    _, control = measure(0)
    seam_2, rest_2 = measure(1)
    seam_4, rest_4 = measure(3)
    seam_8, rest_8 = measure(7)

    assert control < 0.01, (
        f"the clean control already publishes {100 * control:.2f}% of its "
        f"rows over the gate; nothing below is interpretable until that is "
        f"understood"
    )
    assert rest_2 < 0.01 and seam_2 < 0.015, (
        f"a gap of 2 should be indistinguishable from clean; measured "
        f"seam {100 * seam_2:.2f}%, rest {100 * rest_2:.2f}%"
    )
    assert rest_4 > 3 * control and rest_8 > 3 * control, (
        f"gaps of 4 and 8 measured {100 * rest_4:.2f}% and "
        f"{100 * rest_8:.2f}% segment-wide against a {100 * control:.2f}% "
        f"control. If a wide gap no longer degrades the rest of the "
        f"segment, that cost is gone and this test should be re-pointed "
        f"rather than deleted"
    )
    for gap, seam, rest in ((2, seam_2, rest_2), (4, seam_4, rest_4),
                            (8, seam_8, rest_8)):
        assert seam <= 3 * rest + 0.005, (
            f"at gap {gap} the recovered keyframe publishes "
            f"{100 * seam:.2f}% of its rows over the gate against "
            f"{100 * rest:.2f}% for the rest of the segment -- the seam is "
            f"no longer publishing on the same terms as everything else"
        )


# -- 7. recovery is retrying, not leniency -----------------------------


def test_a_pure_rotation_walk_yields_no_poses_however_long_recovery_retries():
    """The test that fails if recovery is ever implemented by loosening a
    threshold instead of by retrying.

    A camera rotating about its own centre has zero baseline.
    Triangulation there is not noisy, it is undefined -- and recoverPose
    will still hand back a confident translation whose direction means
    nothing. Every keyframe of this walk must refuse, and recovery must
    keep it refusing however many times it is asked: retrying a question
    whose answer does not exist must not eventually produce one.

    `MAX_RECOVERY_KEYFRAMES` is raised to 999 deliberately, so that the
    walk is not rescued by the chain breaking early. Ten keyframes each
    get a real solver attempt and all ten must come back empty.

    The companion test below is what gives this one teeth.
    """
    camera = _camera()
    poses = ss.pure_rotation(10, degrees_per_step=2.0)
    images = ss.render_sequence(ss.furnished_room(), poses, camera, WIDTH, HEIGHT)
    original = classical.MAX_RECOVERY_KEYFRAMES
    try:
        classical.MAX_RECOVERY_KEYFRAMES = 999
        steps, _ = _run(_frames(images), camera)
    finally:
        classical.MAX_RECOVERY_KEYFRAMES = original
    solved = [
        index
        for index, step in enumerate(steps)
        if step.pose.status == POSE_STATUS_SOLVED
    ]
    assert solved == [], (
        f"keyframes {solved} were given a translation by a camera that never "
        f"translated"
    )


def test_that_refusal_is_bought_by_the_thresholds_and_not_by_luck(monkeypatch):
    """Proves the test above is sensitive to the thing it claims to guard.

    A test asserting "no poses" passes trivially if its input happens to be
    hard for unrelated reasons, and this suite already contains one such:
    `test_recovery_still_refuses_when_the_geometry_is_absent` feeds frames
    that never seed a map at all, so it would keep passing with every
    acceptance constant set to zero.

    This is the mutation that shows the pure-rotation refusal is real.
    Relaxing the two gates that speak to baseline -- `MIN_INLIER_RATIO`
    and `MIN_TRIANGULATION_ANGLE_DEG` -- and nothing else turns the walk
    above from 0 solved poses into 8, each carrying a fabricated
    translation. So if a future change ever buys recovery by moving those
    numbers, the test above is what will say so.

    NOTE THE TWO BINDINGS. `classical.py` does `from ...geometry import
    MIN_INLIER_RATIO`, which copies the value at import time, so patching
    the geometry module alone silently does nothing -- and a mutation test
    that patches only one of them reports "the mutation changed nothing",
    which is a lie in the reassuring direction.
    """
    for name, value in (
        ("MIN_INLIER_RATIO", 0.0),
        ("MIN_TRIANGULATION_ANGLE_DEG", 0.0),
    ):
        monkeypatch.setattr(geometry, name, value)
        monkeypatch.setattr(classical, name, value)
    monkeypatch.setattr(classical, "MAX_RECOVERY_KEYFRAMES", 999)

    camera = _camera()
    poses = ss.pure_rotation(10, degrees_per_step=2.0)
    images = ss.render_sequence(ss.furnished_room(), poses, camera, WIDTH, HEIGHT)
    steps, _ = _run(_frames(images), camera)
    solved = sum(1 for step in steps if step.pose.status == POSE_STATUS_SOLVED)
    assert solved >= 4, (
        f"only {solved} poses appeared with the baseline gates removed. If a "
        f"zero-baseline walk is now refused for some other reason, the "
        f"pure-rotation test above is no longer guarding those two constants "
        f"and needs a different input"
    )


def test_accuracy_is_what_the_reprojection_gate_buys(monkeypatch):
    """A behavioural guard for `PNP_REPROJECTION_ERROR_PX`, because the only
    thing currently guarding it is an assertion that the constant is 3.

    Mutation-tested against the whole world-builder suite with
    `test_thresholds_are_unchanged` deselected: raising
    `PNP_REPROJECTION_ERROR_PX` from 3.0 to 30.0, zeroing
    `MIN_INLIER_RATIO`, zeroing `MIN_TRIANGULATION_ANGLE_DEG`, or dropping
    `MIN_INLIERS` to 4 each leave every remaining test GREEN. A
    constant-equality pin catches a typo; it does not catch someone who
    decided the number should be different and updated both places, which
    is exactly how "recovery" would get implemented as leniency.

    So this measures the consequence instead. Clean 12 cm strafe, no
    refusals, everything else at its shipped value:

        PNP_REPROJECTION_ERROR_PX     3.0    5.0   10.0    30.0
        worst rotation error (deg)   0.52   0.64   0.83    0.82
        worst PUBLISHED support row  3.59   5.65   7.13   15.28  px

    The rotation error moves, but only by a factor of 1.6, and it moves
    further in a warm process than a cold one -- OpenCV's ORB and USAC are
    threaded and their output is not bit-stable across processes here, so
    an assertion resting on that separation is flaky and this test was
    measured failing in isolation while passing in file order before it
    was rewritten. The published support's WORST row is the sturdy
    quantity: `_reobserve_against_pose` admits a re-observation only if it
    lands within `PNP_REPROJECTION_ERROR_PX` of the feature claiming it,
    so that constant is directly the ceiling on what may be published, and
    raising it puts rows into `support.json` that no 3 px solve would have
    accepted. Registration then PnPs against them.

    Hence an ABSOLUTE bound in pixels rather than a bound relative to the
    constant: relative to a loosened constant every row is compliant by
    construction, which is the tautology this test exists to escape.
    """
    images, poses = _walk("furnished", "strafe", 14, 0.12, (0.0, -1.6, 0.6))
    camera = _camera()
    # Twice the shipped gate. A published row is allowed to exceed the gate
    # a little -- landmarks are also admitted by the separate triangulation
    # gate, and a pose refines after the correspondences were chosen -- but
    # not by the multiples a relaxed constant produces.
    ceiling = 2 * 3.0

    def worst_published(frames, steps, backend):
        errors = _reprojection_by_keyframe(frames, steps, backend)
        return max(value for values in errors.values() for value in values)

    frames = _frames(images)
    steps, backend = _run(frames, camera)
    rows = _measured(steps, poses)
    worst_rotation = max(
        row["rotation_deg"]
        for row in rows.values()
        if row["status"] == POSE_STATUS_SOLVED
    )
    assert worst_rotation < 1.5, (
        f"worst rotation error {worst_rotation:.2f} deg on a clean strafe; "
        f"the shipped configuration measures 0.52-0.91 deg"
    )
    strict = worst_published(frames, steps, backend)
    assert strict < ceiling, (
        f"the worst published support row reprojects at {strict:.2f} px, "
        f"over the {ceiling} px ceiling; the shipped configuration measures "
        f"3.59 px"
    )

    monkeypatch.setattr(classical, "PNP_REPROJECTION_ERROR_PX", 30.0)
    loose_frames = _frames(images)
    loose_steps, loose_backend = _run(loose_frames, camera)
    loose = worst_published(loose_frames, loose_steps, loose_backend)
    assert loose > ceiling, (
        f"a ten-fold looser reprojection gate published nothing worse than "
        f"{loose:.2f} px, still inside the ceiling above -- which would mean "
        f"the ceiling is not guarding this constant after all"
    )
