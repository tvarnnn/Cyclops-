"""Review V8, LOW-a: an area is `levelled` only when its normals support the vertical.

Module: `tower/world_builder/area_build.py` (`group_up`). Contract WORLD-BUILDER-COMPONENTS.md
§5.4: "When the vertical cannot be estimated (support below the Tower's floor) ... `area.levelled`
is `false`, and the caption adds *Its vertical could not be estimated, so it may look tilted.*"
§5.4 names no number (OPEN T3), and `group_up` reported `levelled: true` at ANY normal support.

THE FLOOR IS THE ROOM'S OWN. The room page levels its horizon with
`surface_render.surface_up`, which refines toward the surface's level faces only when at least
`min_horizontal_fraction` (10 %) of the surface lies within `cone_deg` (30 deg) of horizontal,
and otherwise does not claim the surface's vertical. The area applies the same test, with the
same two constants, to its own depth normals around the up it chose. Measured on the run's only
area builds whose depth survives (RUN `experiments/P3-PUB/lowa/support.json`, P2-PX's builds):
the target's Area 2 -- §5.4's own example, normal support 0.064, rendered level -- has 0.139 of
its normals within 30 deg of its up, and passes; every other island and room measured 0.21-0.41.
"""

from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

from tower.world_builder import area_build as AB
from tower.world_builder import surface_render as SR


def _level_cameras(n=12):
    """Cameras panning a level head: -Y (OpenCV y-down) is up."""
    out = []
    for yaw in np.radians(np.linspace(0, 330, n)):
        c, s = math.cos(yaw), math.sin(yaw)
        out.append(np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]))
    return np.array(out)


def _normals(n, floor_fraction, seed=0, noise_deg=3.0):
    """`floor_fraction` of the normals on level surfaces (+-Y), the rest on walls."""
    rng = np.random.default_rng(seed)
    k = int(round(n * floor_fraction))
    floor = np.tile([0.0, -1.0, 0.0], (k, 1))
    yaw = rng.uniform(0, 2 * np.pi, n - k)
    walls = np.stack([np.cos(yaw), np.zeros(n - k), np.sin(yaw)], 1)
    N = np.concatenate([floor, walls]) + rng.normal(0, math.radians(noise_deg), (n, 3))
    return N / np.linalg.norm(N, axis=1, keepdims=True)


def test_the_floor_is_the_rooms_own():
    """The two constants are `surface_up`'s defaults, so the room and its areas cannot drift."""
    params = inspect.signature(SR.surface_up).parameters
    assert AB.LEVEL_CONE_DEG == params["cone_deg"].default == 30.0
    assert AB.LEVEL_MIN_HORIZONTAL_FRACTION == params["min_horizontal_fraction"].default == 0.10


def test_an_area_seen_almost_only_as_walls_is_not_levelled():
    """5 % level surface: the vertical is the walls' guess, not measured. Before the fix:
    `levelled: true` (support 0.05, no floor)."""
    up = AB.group_up(_level_cameras(), _normals(4000, 0.05))
    assert up["up"] is not None, "still rotated by its best estimate (review V6, L5)"
    assert up["horizontal_fraction"] < AB.LEVEL_MIN_HORIZONTAL_FRACTION
    assert up["levelled"] is False
    assert "below" in up["source"]


@pytest.mark.parametrize("fraction", [0.14, 0.3, 0.6])
def test_an_area_with_enough_level_surface_is_levelled(fraction):
    """0.14 is the target's Area 2 (0.139 within 30 deg), which rendered level."""
    up = AB.group_up(_level_cameras(), _normals(4000, fraction))
    assert up["levelled"] is True
    assert up["horizontal_fraction"] >= AB.LEVEL_MIN_HORIZONTAL_FRACTION
    assert math.degrees(math.acos(min(1.0, float(np.dot(up["up"], [0, -1, 0]))))) < 2.0


def test_the_fraction_is_recorded_for_audit():
    up = AB.group_up(_level_cameras(), _normals(4000, 0.3))
    assert set(up) >= {"levelled", "up", "support", "ambiguity", "normals",
                       "horizontal_fraction", "source"}
    assert 0.0 <= up["horizontal_fraction"] <= 1.0


def test_the_level_head_fallback_is_unchanged():
    """Too few normals: the roll-free up decides `levelled` by its conditioning, as before
    (no horizontal fraction exists to test)."""
    up = AB.group_up(_level_cameras(), np.zeros((10, 3)))
    assert "roll-free" in up["source"] and "horizontal_fraction" not in up
