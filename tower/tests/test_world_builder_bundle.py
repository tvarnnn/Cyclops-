"""The bundle adjuster, checked against geometry it cannot see.

WHY THIS FILE EXISTS

`world_builder/bundle.py` is the only optimiser in the world builder and
it moves BOTH camera poses and landmark positions. An optimiser that is
subtly wrong does not crash; it produces a reconstruction that reprojects
beautifully and is not the room. So nothing here checks that the cost
went down -- that is trivially arrangeable by an optimiser that is wrong
in a self-consistent way. Every test compares against a scene whose
answer is known independently of the optimiser.

The three properties that matter:

  1. It recovers KNOWN geometry from a perturbed start.
  2. It respects `fixed_cameras`, exactly, because a caller passes that
     when it has already published a pose someone else is holding.
  3. It never returns a worse estimate than it was given. A caller must
     be able to trust `improved`, because the alternative -- silently
     accepting a step that made the reconstruction worse -- is how an
     optimiser turns a merely-drifting map into a wrong one.
"""

import cv2
import numpy as np
import pytest

from tower.world_builder import bundle


def _rig(seed=0, cameras=6, points=200, noise_px=0.3):
    """A calibrated rig looking at a point cloud, with exact truth."""
    rng = np.random.default_rng(seed)
    camera_matrix = np.array([[400.0, 0, 240], [0, 400.0, 180], [0, 0, 1]])
    rotations = np.array([
        cv2.Rodrigues(np.array([0.0, 0.02 * i, 0.0]))[0] for i in range(cameras)
    ])
    translations = np.array([[-0.15 * i, 0.0, 0.0] for i in range(cameras)])
    xyz = rng.uniform(-1.5, 1.5, (points, 3))
    xyz[:, 2] += 4.0

    rows = []
    for camera in range(cameras):
        in_camera = (rotations[camera] @ xyz.T).T + translations[camera]
        projected = (camera_matrix @ in_camera.T).T
        pixels = projected[:, :2] / projected[:, 2:3]
        pixels += rng.normal(0, noise_px, pixels.shape)
        for point in range(points):
            u, v = pixels[point]
            if 0 <= u < 480 and 0 <= v < 360:
                rows.append([camera, point, u, v])
    return camera_matrix, rotations, translations, xyz, np.array(rows)


def _perturb(rotations, translations, xyz, seed=1,
             rotation_sigma=0.02, translation_sigma=0.03, point_sigma=0.05):
    rng = np.random.default_rng(seed)
    perturbed_rotations = np.array([
        cv2.Rodrigues(cv2.Rodrigues(R)[0].ravel()
                      + rng.normal(0, rotation_sigma, 3))[0]
        for R in rotations
    ])
    return (
        perturbed_rotations,
        translations + rng.normal(0, translation_sigma, translations.shape),
        xyz + rng.normal(0, point_sigma, xyz.shape),
    )


def _rotation_error_deg(a, b):
    return float(np.degrees(np.linalg.norm(cv2.Rodrigues(a @ b.T)[0])))


def test_it_recovers_a_known_rig_from_a_perturbed_start():
    """The whole claim, against truth the optimiser never sees.

    Camera 0 is held fixed at its PERTURBED pose, so the recovered
    solution is correct up to that camera's own error -- which is why
    this asserts that every camera ends up the SAME distance from truth
    as camera 0 rather than at zero error. A solution where the cameras
    disagree with each other is broken; one where they agree and are all
    rotated together is the gauge doing its job.
    """
    camera_matrix, rotations, translations, xyz, rows = _rig()
    started_rotations, started_translations, started_xyz = _perturb(
        rotations, translations, xyz)

    adjusted_r, adjusted_t, adjusted_p, report = bundle.optimise(
        started_rotations, started_translations, started_xyz, rows,
        camera_matrix, iterations=25, fixed_cameras=(0,))

    assert report["improved"]
    assert report["reprojection_rms_after"] < 1.0
    assert report["reprojection_rms_after"] < report["reprojection_rms_before"] / 10

    gauge = _rotation_error_deg(adjusted_r[0], rotations[0])
    errors = [_rotation_error_deg(adjusted_r[i], rotations[i])
              for i in range(len(rotations))]
    assert max(abs(error - gauge) for error in errors) < 0.5, (
        f"cameras disagree with each other: gauge={gauge:.3f} errors={errors}"
    )


def test_a_fixed_camera_does_not_move_at_all():
    """Not "barely moves". A caller passes `fixed_cameras` because it has
    already published that pose; a millimetre of drift there is a
    published pose that is now a lie."""
    camera_matrix, rotations, translations, xyz, rows = _rig()
    started_rotations, started_translations, started_xyz = _perturb(
        rotations, translations, xyz)

    adjusted_r, adjusted_t, _, _ = bundle.optimise(
        started_rotations, started_translations, started_xyz, rows,
        camera_matrix, iterations=15, fixed_cameras=(0, 1))

    for index in (0, 1):
        assert np.array_equal(adjusted_r[index], started_rotations[index])
        assert np.array_equal(adjusted_t[index], started_translations[index])


def test_the_inputs_are_not_mutated():
    """`optimise` returns new arrays. A caller that keeps the old estimate
    to compare against -- which is what makes `improved` actionable --
    must still have it afterwards."""
    camera_matrix, rotations, translations, xyz, rows = _rig()
    started_rotations, started_translations, started_xyz = _perturb(
        rotations, translations, xyz)
    before = (started_rotations.copy(), started_translations.copy(),
              started_xyz.copy())

    bundle.optimise(started_rotations, started_translations, started_xyz,
                    rows, camera_matrix, iterations=10)

    assert np.array_equal(started_rotations, before[0])
    assert np.array_equal(started_translations, before[1])
    assert np.array_equal(started_xyz, before[2])


def test_it_never_returns_a_worse_estimate():
    """Levenberg-Marquardt only accepts a step that lowered the cost, and
    this asserts the consequence rather than the mechanism: whatever it
    hands back reprojects at least as well as what it was given.

    Run over several perturbation magnitudes, including one far outside
    the basin, because "never worse" has to hold when the optimiser
    FAILS as well as when it succeeds.
    """
    camera_matrix, rotations, translations, xyz, rows = _rig()
    for sigma in (0.005, 0.05, 0.4):
        started = _perturb(rotations, translations, xyz,
                           rotation_sigma=sigma, translation_sigma=sigma,
                           point_sigma=sigma)
        _, _, _, report = bundle.optimise(
            *started, rows, camera_matrix, iterations=12)
        assert (
            report["reprojection_rms_after"]
            <= report["reprojection_rms_before"] + 1e-9
        ), f"sigma={sigma}: {report}"


def test_two_view_landmarks_move_with_their_cameras():
    """The bug this branch shipped and then found, pinned as a test.

    A point seen twice is exactly determined and tells the optimiser
    nothing about the cameras, which is a real argument for excluding it
    -- and excluding it is WRONG, because a third of the real map is
    two-view and those landmarks still have published support rows. Drop
    their observations and the cameras move out from under them, and
    every row they published stops reprojecting. Measured on the drawer
    walk: published p99 3.97 px with no adjustment at all, 13.70 px with
    the adjustment excluding two-view landmarks, 2.76 px including them.

    So this asserts the consistency property, not an information one: a
    landmark whose cameras moved must have moved too.
    """
    camera_matrix, rotations, translations, xyz, rows = _rig(cameras=2)
    started = _perturb(rotations, translations, xyz)
    _, _, adjusted_p, report = bundle.optimise(
        *started, rows, camera_matrix, iterations=10, fixed_cameras=(0,))

    assert report.get("landmarks_adjusted", 0) > 0
    assert not np.array_equal(adjusted_p, started[2])


def test_a_landmark_the_window_cannot_hold_is_not_moved():
    """`fixed_points` is how a caller says "this landmark has observers
    you cannot see". The optimiser must leave it exactly where it is
    while still letting its observations constrain the cameras --
    otherwise a windowed adjustment silently rewrites geometry whose
    other half is already on disk."""
    camera_matrix, rotations, translations, xyz, rows = _rig()
    started_rotations, started_translations, started_xyz = _perturb(
        rotations, translations, xyz)
    fixed = np.zeros(len(xyz), dtype=bool)
    fixed[::3] = True

    _, _, adjusted_p, report = bundle.optimise(
        started_rotations, started_translations, started_xyz, rows,
        camera_matrix, iterations=15, fixed_cameras=(0,), fixed_points=fixed)

    assert np.array_equal(adjusted_p[fixed], started_xyz[fixed])
    assert not np.array_equal(adjusted_p[~fixed], started_xyz[~fixed])
    assert report["landmarks_fixed"] == int(fixed.sum())


def test_it_refuses_rather_than_raises_on_degenerate_input():
    """The live path calls this. A solve that cannot proceed must end the
    ADJUSTMENT, not the walk."""
    camera_matrix, rotations, translations, xyz, _ = _rig()
    empty = np.zeros((0, 4))
    r, t, p, report = bundle.optimise(
        rotations, translations, xyz, empty, camera_matrix)
    assert report["iterations"] == 0
    assert np.array_equal(r, rotations)

    r, t, p, report = bundle.optimise(
        np.zeros((0, 3, 3)), np.zeros((0, 3)), xyz, empty, camera_matrix)
    assert report["iterations"] == 0


def test_a_landmark_behind_the_camera_does_not_poison_the_solve():
    """A negative depth has no meaningful pixel error. It is given a zero
    residual and a zero Jacobian rather than a large one, because a
    huge fabricated residual would drag every camera that shares the
    landmark."""
    camera_matrix, rotations, translations, xyz, rows = _rig()
    poisoned = xyz.copy()
    poisoned[:10, 2] = -5.0
    started_rotations, started_translations, _ = _perturb(
        rotations, translations, xyz)

    _, _, adjusted_p, report = bundle.optimise(
        started_rotations, started_translations, poisoned, rows,
        camera_matrix, iterations=15, fixed_cameras=(0,))

    assert np.isfinite(adjusted_p).all()
    assert report["reprojection_rms_after"] < report["reprojection_rms_before"]


def test_the_result_is_deterministic():
    """Two identical calls must agree bit for bit. Nothing here is seeded
    by OpenCV's RNG, and a replay that cannot be repeated is not
    evidence."""
    camera_matrix, rotations, translations, xyz, rows = _rig()
    started = _perturb(rotations, translations, xyz)
    first = bundle.optimise(*started, rows, camera_matrix, iterations=12,
                            fixed_cameras=(0,))
    second = bundle.optimise(*started, rows, camera_matrix, iterations=12,
                             fixed_cameras=(0,))
    for a, b in zip(first[:3], second[:3]):
        assert np.array_equal(a, b)
    # `point_ok` is an array, so the reports are compared field by field
    # rather than with ==, which numpy makes ambiguous.
    assert set(first[3]) == set(second[3])
    for key in first[3]:
        if isinstance(first[3][key], np.ndarray):
            assert np.array_equal(first[3][key], second[3][key]), key
        else:
            assert first[3][key] == second[3][key], key


@pytest.mark.parametrize("cap", [0, 4, 8])
def test_the_view_cap_changes_cost_not_correctness(cap, monkeypatch):
    """MAX_VIEWS_PER_LANDMARK bounds a quadratic. It must not change
    whether the answer is right -- only how long it takes to get there."""
    monkeypatch.setattr(bundle, "MAX_VIEWS_PER_LANDMARK", cap)
    camera_matrix, rotations, translations, xyz, rows = _rig(cameras=8)
    started = _perturb(rotations, translations, xyz)
    adjusted_r, _, _, report = bundle.optimise(
        *started, rows, camera_matrix, iterations=20, fixed_cameras=(0,))

    assert report["improved"]
    gauge = _rotation_error_deg(adjusted_r[0], rotations[0])
    errors = [_rotation_error_deg(adjusted_r[i], rotations[i])
              for i in range(len(rotations))]
    assert max(abs(error - gauge) for error in errors) < 0.5, (
        f"cap={cap} gave inconsistent cameras: {errors}"
    )
