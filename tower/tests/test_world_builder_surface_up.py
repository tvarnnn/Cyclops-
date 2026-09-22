"""The page's vertical is measured on the surface, not only on the cameras.

On the canonical world the mean camera up leaned 26 degrees toward the desk
the wearer looked down at, and the phone showed the room rolled.
"""
from __future__ import annotations

import math

import numpy as np

from tower.world_builder.surface_render import surface_up


def _quad(corners):
    """Two triangles over four corners, as (V, F) arrays."""
    V = np.asarray(corners, float)
    return V, np.array([[0, 1, 2], [0, 2, 3]])


def _room(rotation):
    """A floor, a ceiling and two walls of a box room, rotated as a whole.

    Areas are room-like: floor and ceiling 16 units each, walls 12 each.
    Up is +Z before rotation.
    """
    parts = [
        _quad([[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0]]),          # floor
        _quad([[0, 0, 3], [0, 4, 3], [4, 4, 3], [4, 0, 3]]),          # ceiling
        _quad([[0, 0, 0], [0, 0, 3], [4, 0, 3], [4, 0, 0]]),          # wall y=0
        _quad([[0, 0, 0], [0, 4, 0], [0, 4, 3], [0, 0, 3]]),          # wall x=0
    ]
    Vs, Fs, base = [], [], 0
    for V, F in parts:
        Vs.append(V @ rotation.T)
        Fs.append(F + base)
        base += len(V)
    return np.concatenate(Vs), np.concatenate(Fs)


def _rot(axis, deg):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    a = math.radians(deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def _angle(u, v):
    u, v = np.asarray(u, float), np.asarray(v, float)
    return math.degrees(math.acos(min(1.0, abs(u @ v) / np.linalg.norm(u) / np.linalg.norm(v))))


def test_a_camera_estimate_that_leans_is_levelled_by_the_floor_and_ceiling():
    R = _rot([0.3, 1.0, 0.2], 35)
    V, F = _room(R)
    true_up = R @ np.array([0.0, 0.0, 1.0])
    across = np.cross(true_up, [1.0, 0.0, 0.0])
    seed = _rot(across, 22) @ true_up                 # a wearer looking down
    assert _angle(seed, true_up) > 20
    up = surface_up(V, F, seed)
    assert up is not None
    assert _angle(up, true_up) < 0.5
    assert np.dot(up, seed) > 0, "the refinement keeps the seed's sign"


def test_a_surface_with_no_horizontal_evidence_keeps_the_camera_estimate():
    """Walls only: nothing measures the vertical, so nothing is refined."""
    V, F = _room(np.eye(3))
    walls_only = F[4:]
    seed = _rot([1.0, 0.0, 0.0], 10) @ np.array([0.0, 0.0, 1.0])
    assert surface_up(V, walls_only, seed) is None


def test_a_refinement_that_would_tip_the_walls_is_refused():
    """A large slanted surface inside the cone must not drag the vertical
    away from the walls: the wall check refuses it."""
    V, F = _room(np.eye(3))
    ramp_V = np.array([[0, 0, 0], [8, 0, 3.5], [8, 8, 3.5], [0, 8, 0]], float)
    ramp_F = np.array([[0, 1, 2], [0, 2, 3]]) + len(V)
    keep = np.concatenate([F[4:], ramp_F])            # walls + a 24-degree ramp, no floor
    V = np.concatenate([V, ramp_V])
    seed = np.array([0.0, 0.0, 1.0])
    up = surface_up(V, keep, seed)
    assert up is None or _angle(up, seed) < 1e-6


def test_degenerate_input_is_no_refinement():
    seed = [0.0, 0.0, 1.0]
    assert surface_up(np.zeros((0, 3)), np.zeros((0, 3), int), seed) is None
    V, F = _room(np.eye(3))
    assert surface_up(V, F, [0.0, 0.0, 0.0]) is None
    assert surface_up(V, F, [float("nan"), 0.0, 1.0]) is None
