"""Global structure-from-motion over a session's keyframes.

WHY THIS EXISTS

The World Builder reconstructs each tracking segment as a forward PnP
chain and then tries to glue segments together with a Sim3 fitted between
two independent, already-drifted reconstructions. On the 2026-09-06 walk
(438 keyframes, 34 segments) that put 58 keyframes into one frame. Three
independent audits of that walk agree on why:

  * the chain refuses or strands half the walk (210 of 438 poses refused,
    185 of them cascaded from 25 root refusals; a 74-keyframe segment with
    zero geometry although its images link strongly to its neighbours);
  * the segments that DO reconstruct are not rigid -- one segment's scale
    drops ~10x along its own length -- so no Sim3 fits them, and the
    registrar's refusals are largely correct;
  * the features are there: sequential SIFT matching over the same frames
    yields 8,550 verified pairs, 5,559 with >= 15 inliers.

A global solver over pairwise constraints across EVERY keyframe -- rotation
averaging, global positioning, bundle adjustment, i.e. GLOMAP as shipped in
pycolmap 4.2 -- places 428 of those 438 keyframes in one frame at 0.84 px
mean reprojection. Where the local chain is sound (segment 9) the two
solvers agree to 0.3% of the segment's extent; where the chain is known to
have diverged (segment 16) they disagree. See
docs/world-builder-reconstruction-experiments.md, E1-E10.

THE RECIPE (every step measured, see the ledger)

    keyframe image  (raw capture frame when on disk, else the redacted copy)
      -> undistort ONCE with the session calibration     (E8: GLOMAP cannot
         take the 8-parameter FULL_OPENCV model; a pinhole image can go to
         any solver)
      -> SIFT, CPU                                         (~15 ms / frame)
      -> sequential matching, overlap SEQUENTIAL_OVERLAP   (~75 ms / frame)
      -> GLOMAP; incremental mapping if GLOMAP yields nothing
      -> support floor: a camera with fewer than MIN_IMAGE_OBSERVATIONS
         3-D observations is NOT a measurement and stays unplaced (E3:
         GLOMAP's zero-support cameras sat 200 units outside the room)

The feature database persists per session, so a re-solve pays only for
NEW keyframes (E9: 12 s to match 138 new keyframes against 300 old ones).
That is what makes the background path affordable during a live walk.

HOW IT REACHES THE PHONE

Nothing in the geometry contract changes. Tracker segments stay the unit;
`merge()` rewrites each solved segment's poses and points in the segment's
own frame (anchored at its first posed keyframe) and expresses the global
solution through the layer the contract already has for it -- placements.
Every segment of one COLMAP model shares a reference segment, which is
exactly the contract's composition rule. Two models are two frames and are
never composited.

TRUTHFULNESS

  * A keyframe the solver did not pose keeps a refused pose row.
  * A segment with no posed keyframe keeps its LOCAL geometry and a
    `refused` placement that says so.
  * A segment newer than the solution's horizon keeps its local geometry,
    unplaced -- the contract's live "still building" case.
  * Scale stays unknown. Placement scale within a model is exactly 1,
    because the segments are one reconstruction.

pycolmap is OPTIONAL. Without it `solver_available()` is False, the World
Builder behaves exactly as before, and the manifest says why.
"""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tower.storage import (
    read_bytes_closed,
    staging_path,
    replace_with_retry,
    sweep_abandoned_staging,
    read_json_closed,
    write_bytes_atomic,
    write_json_atomic,
)
from tower.world_builder.records import Keyframe, SegmentPlacement
from tower.world_builder.schema import (
    DEGENERACY_NONE,
    POSE_STATUS_ANCHOR,
    POSE_STATUS_SOLVED,
    POSE_STATUS_UNAVAILABLE,
)

logger = logging.getLogger(__name__)

# Degeneracy recorded on a pose row for a keyframe the global solver did
# not place. Distinct from the chain's reasons: the image was in the
# problem and no component claimed it with enough support.
DEGENERACY_UNREGISTERED = "unregistered"

# A camera whose pose rests on fewer 3-D observations than this is not a
# measurement. GLOMAP's rotation averaging poses EVERY image in a connected
# view graph, and on the 2026-09-06 walk the images with 0 observations sat
# 30-214 units from a room 12 units across. The incremental mapper's own
# floor on that walk was 44; 30 keeps every genuinely triangulated camera
# and refuses the ones positioned by rotation alone.
MIN_IMAGE_OBSERVATIONS = 30

# A model with fewer images than this is not a component worth a frame of
# its own; the phone would show a 3-camera fragment as "a world".
MIN_MODEL_IMAGES = 5

# Sequential matching window, in keyframes. At ~3.6 keyframes per second
# of walk this is ~6 s of footage either side; E1 measured it sufficient to
# connect 98% of a two-minute room walk once GLOMAP solves the graph.
SEQUENTIAL_OVERLAP = 20

# SIFT features per undistorted frame. 360x640 frames yield ~900 on
# average; the cap only bites on very textured frames.
MAX_FEATURES = 4096

SOLUTION_SCHEMA_VERSION = 1
SOLVER_GLOMAP = "glomap"
SOLVER_INCREMENTAL = "incremental"

# Coverage classes reported per segment (see docs/contracts/WORLD-BUILDER-GEOMETRY.md
# addendum). Nothing is invented for gaps: `unresolved` is "keyframes exist,
# no geometry", and unseen space simply has no keyframe.
COVERAGE_CONFIDENT = "confident"
COVERAGE_PARTIAL = "partial"
COVERAGE_UNRESOLVED = "unresolved"


def solver_available() -> tuple[bool, str | None]:
    """Whether pycolmap can be imported, and why not if not."""
    try:
        import pycolmap  # noqa: F401
    except Exception as exc:  # ImportError or a broken native wheel
        return False, f"pycolmap is not importable: {exc}"
    return True, None


# ---------------------------------------------------------------------------
# Small rotation helpers. numpy only; the engine keeps its own private copy of
# the quaternion conversion and importing it here would be a circular import.


def rotation_to_quaternion_wxyz(m) -> list[float]:
    m = np.asarray(m, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q)
    return [float(v) for v in q]


def quaternion_wxyz_to_rotation(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Workspace: everything the solver keeps between runs, under the world.


@dataclass(frozen=True)
class SolveWorkspace:
    root: Path

    @property
    def images_dir(self) -> Path:
        return self.root / "images"

    @property
    def database_path(self) -> Path:
        return self.root / "database.db"

    @property
    def sparse_dir(self) -> Path:
        return self.root / "sparse"

    @property
    def camera_path(self) -> Path:
        return self.root / "camera.json"

    @property
    def solution_path(self) -> Path:
        return self.root / "solution.json"

    @property
    def arrays_path(self) -> Path:
        return self.root / "solution.npz"


def workspace_for(store, world_id: str, session_id: str) -> SolveWorkspace:
    """The solver's working directory for one session: `<world>/solve/<session>`.

    Beside `derived/`, never inside it: derived is the published output the
    store digests and serves, this is disposable intermediate state
    (undistorted frames, the feature database, COLMAP's own model files).
    """
    return SolveWorkspace(store.world_dir(world_id) / "solve" / session_id)


def keyframe_image_name(keyframe: Keyframe) -> str:
    """The file name COLMAP knows the keyframe by: the journal's own image
    file name, which is the source sequence number and unique per session."""
    return Path(keyframe.image_relpath).name


@dataclass(frozen=True)
class PinholeCamera:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def to_json_dict(self) -> dict:
        return {
            "fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
            "width": self.width, "height": self.height,
        }

    @classmethod
    def from_json_dict(cls, d: dict) -> "PinholeCamera":
        return cls(
            fx=float(d["fx"]), fy=float(d["fy"]), cx=float(d["cx"]),
            cy=float(d["cy"]), width=int(d["width"]), height=int(d["height"]),
        )


class UndistortionUnavailable(RuntimeError):
    """The session has no usable pinhole calibration."""


def _undistort_maps(intrinsics, width: int, height: int):
    """Rectification maps + the pinhole camera of the cropped result.

    alpha = 0: keep only pixels every source pixel can fill, so no black
    border enters the image -- a black border is a strong, perfectly
    repeatable fake feature in every frame and it would be matched.
    """
    import cv2

    if intrinsics is None or intrinsics.fx is None or intrinsics.cx is None:
        raise UndistortionUnavailable("session has no intrinsics")
    k = np.array(
        [[intrinsics.fx, 0.0, intrinsics.cx],
         [0.0, intrinsics.fy, intrinsics.cy],
         [0.0, 0.0, 1.0]], dtype=np.float64,
    )
    dist = np.array(intrinsics.dist_coeffs or (0.0, 0.0, 0.0, 0.0, 0.0), dtype=np.float64)
    if intrinsics.calibrated_width and intrinsics.calibrated_width != width:
        # The session's frames are a different size from the calibration.
        # The engine already refuses that combination at build time; refuse
        # here too rather than scale a distortion model we cannot check.
        raise UndistortionUnavailable(
            f"calibrated for {intrinsics.calibrated_width}x"
            f"{intrinsics.calibrated_height}, frames are {width}x{height}"
        )
    new_k, roi = cv2.getOptimalNewCameraMatrix(k, dist, (width, height), 0, (width, height))
    x, y, rw, rh = (int(v) for v in roi)
    if rw < 32 or rh < 32:
        raise UndistortionUnavailable(f"valid undistorted region is {rw}x{rh}")
    m1, m2 = cv2.initUndistortRectifyMap(k, dist, None, new_k, (width, height), cv2.CV_16SC2)
    camera = PinholeCamera(
        fx=float(new_k[0, 0]), fy=float(new_k[1, 1]),
        cx=float(new_k[0, 2] - x), cy=float(new_k[1, 2] - y),
        width=rw, height=rh,
    )
    return m1, m2, (x, y, rw, rh), camera


SOURCES_FILENAME = "sources.json"


def write_sources(store, world_id: str, session_id: str, sources: dict) -> None:
    """Record where each keyframe's RAW frame lives: keyframe_id -> path.

    The builder that observed the frames is the only thing that knows
    this (a replay stages frames under enumeration indices, so the name
    alone does not find them). Written by the builder before it launches a
    solve; read by `prepare_images`. Absent means "find by name, else use
    the session's copy".
    """
    write_sources_records(workspace_for(store, world_id, session_id), sources)


def write_sources_records(workspace: SolveWorkspace, sources: dict) -> None:
    """The same, addressed by workspace rather than by store.

    Split out so `prepare_images` can put back what a recalibration's
    `rmtree` just deleted -- it holds a workspace, not a store.
    """
    workspace.root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        workspace.root / SOURCES_FILENAME,
        {"sources": {k: str(v) for k, v in sources.items()}},
    )


# WHAT A RELATIVE `sources.json` PATH IS RELATIVE TO.
#
# The builder records paths as it was given them, and the Tower runs from
# `tower/`, so the canonical capture's 398 entries read `data\captures\...`.
# Every reader then asked `Path(recorded).exists()`, resolved against the
# PROCESS cwd. Measured by the fix-it privacy lane from a scratch directory:
# 95 of 398 raw frames were found, and the dense stage's fill masks fell back
# to the shape guess, which misses solid boxes touching dark scene (PRIVACY.md
# L3). Resolved against `tower/`, 398 of 398 exist (REREDACT.md section 1).
# Same bug class, and the same anchor, as the model path in `redaction.py`.
#
# `TOWER_SOURCES_ROOT` overrides the anchor for a world root read by code that
# is not the Tower that captured it (a worktree, a copy). Never the cwd.
TOWER_ROOT = Path(__file__).resolve().parents[2]
SOURCES_ROOT_ENV = "TOWER_SOURCES_ROOT"


def sources_root() -> Path:
    override = os.environ.get(SOURCES_ROOT_ENV, "").strip()
    return Path(override) if override else TOWER_ROOT


def resolve_source_path(recorded, tower_root=None) -> Path | None:
    """A `sources.json` entry as an absolute path, whatever the cwd."""
    if not recorded:
        return None
    path = Path(str(recorded))
    if path.is_absolute():
        return path
    return (Path(tower_root) if tower_root is not None else sources_root()) / path


def read_sources(workspace: SolveWorkspace) -> dict:
    """keyframe_id -> the path AS RECORDED. Resolve with `resolve_source_path`."""
    path = workspace.root / SOURCES_FILENAME
    if not path.exists():
        return {}
    try:
        return dict(read_json_closed(path).get("sources") or {})
    except (OSError, json.JSONDecodeError, AttributeError):
        return {}


def _source_frame(keyframe: Keyframe, session_dir: Path, capture_dirs, sources=None) -> Path:
    """The raw capture frame when it is on disk, else the session's copy.

    The session's keyframe images are face-redacted, and on real walks the
    redactor blacks out large regions (phone screens, hands) that carry
    exactly the texture a solver needs: E6 measured 307 -> 337 keyframes in
    the main model from using the raw frames. The raw frame never leaves
    this machine and nothing derived from it but points and poses is
    published, exactly as before.
    """
    recorded = resolve_source_path((sources or {}).get(keyframe.keyframe_id))
    if recorded is not None and recorded.is_file():
        return recorded
    name = keyframe_image_name(keyframe)
    for capture_dir in capture_dirs:
        for candidate in (Path(capture_dir) / "frames" / name, Path(capture_dir) / name):
            if candidate.is_file():
                return candidate
    return session_dir / keyframe.image_relpath


def prepare_images(
    store, world_id: str, session_id: str, keyframes: list[Keyframe], *, capture_dirs=()
) -> tuple[PinholeCamera, int]:
    """Undistort every keyframe not yet in the workspace. Returns the pinhole
    camera and how many images were written this call."""
    import cv2

    workspace = workspace_for(store, world_id, session_id)
    session = store.read_session(world_id, session_id)
    session_dir = store.session_dir(world_id, session_id)
    if not keyframes:
        raise UndistortionUnavailable("no keyframes")
    width, height = keyframes[0].width, keyframes[0].height
    m1, m2, (x, y, rw, rh), camera = _undistort_maps(session.intrinsics, width, height)
    workspace.images_dir.mkdir(parents=True, exist_ok=True)
    recalibrated = False
    if workspace.camera_path.exists():
        stored = PinholeCamera.from_json_dict(read_json_closed(workspace.camera_path))
        if stored != camera:
            # Calibration changed under a live workspace. Everything in it
            # was undistorted with the old maps; start over.
            #
            # AND `rmtree(ignore_errors=True)` DOES NOT GUARANTEE THAT. On
            # Windows a file any reader holds open cannot be unlinked, and
            # `ignore_errors` turns that into silence -- the tree survives,
            # the loop below sees `target.exists()` and SKIPS it, and COLMAP
            # is handed a mix of two calibrations under one `camera.json`.
            # An adversarial review demonstrated exactly that: one image
            # from calibration A and three from B, no exception, no warning.
            #
            # `recalibrated` makes the skip conditional instead, so a frame
            # that survived the delete is re-undistorted rather than
            # trusted. The rmtree stays as the cheap path; correctness no
            # longer depends on it succeeding.
            sources_before = read_sources(workspace)
            shutil.rmtree(workspace.root, ignore_errors=True)
            workspace.images_dir.mkdir(parents=True, exist_ok=True)
            recalibrated = True
            if sources_before:
                # `sources.json` maps each keyframe to the RAW capture frame
                # it came from, and the builder that observed those frames is
                # the only thing that knows it -- a replay stages them under
                # enumeration indices, so the name alone does not find them.
                # The rmtree above deletes it three lines before
                # `read_sources` runs, so a recalibration silently demoted
                # every solve to the face-redacted session copies: the
                # ledger's measured 337 images down to 307.
                write_sources_records(workspace, sources_before)
    # THE CAMERA IS COMMITTED AFTER THE IMAGES MATCH IT, not before.
    #
    # This wrote `camera.json` here, before the loop, and gated the
    # re-undistort on a per-CALL `recalibrated` flag. So if the pass did
    # not finish -- and the loop has three ways to abandon a frame without
    # failing: an unreadable source, a wrong-sized source, and any
    # exception -- then the NEXT call read a `camera.json` that already
    # matched, computed `recalibrated = False`, and skipped every stale
    # frame forever. An adversarial review demonstrated the second call
    # re-undistorting zero frames and leaving three of four images from the
    # old calibration under a `camera.json` naming the new one: the same
    # defect this flag was added to close, reached by a different route.
    #
    # Committing the camera last makes the file mean what it says -- "the
    # images beside me were made with these parameters" -- and makes an
    # interrupted pass self-healing, because the next call still sees a
    # mismatch and tries again.
    sources = read_sources(workspace)
    written = 0
    for keyframe in keyframes:
        target = workspace.images_dir / keyframe_image_name(keyframe)
        if target.exists() and not recalibrated:
            continue
        source = _source_frame(keyframe, session_dir, capture_dirs, sources)
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            logger.warning("global solve: unreadable frame %s", source)
            continue
        if image.shape[1] != width or image.shape[0] != height:
            logger.warning("global solve: frame %s is %sx%s, expected %sx%s",
                           source, image.shape[1], image.shape[0], width, height)
            continue
        undistorted = cv2.remap(image, m1, m2, cv2.INTER_LINEAR)[y:y + rh, x:x + rw]
        # A staging name no other solve can be using, and a `finally` that
        # does not leave one behind. Both were missing: the name was
        # `<target>.tmp.jpg`, derived only from the destination, so two
        # solves of one world -- the builder's child and an operator's
        # hand-run `world_solve.py`, which `storage.staging_path` calls an
        # ordinary operator action -- collided on it. One writer's
        # `os.replace` could publish the other's partial JPEG, and the
        # `if target.exists(): continue` above means a corrupt frame is
        # never regenerated: it feeds COLMAP for the life of the workspace.
        # `staging_path`, NOT a hand-rolled name. This spelled the pid bare
        # while `staging_path` spells it `p<pid>`, and when the sweeper was
        # tightened to require the `p` form -- so that a hex uuid or a frame
        # number could not be mistaken for a process -- this writer silently
        # stopped being swept. It is the writer that leaks MOST: the builder
        # terminates a solve child on every stop that outstays its budget,
        # and this is the per-frame write it dies inside.
        #
        # The suite did not notice because the test wrote a name by hand
        # instead of asking the producer for one, so it pinned the fix and
        # not the code. One producer, one convention.
        tmp = staging_path(target).with_suffix(".tmp.jpg")
        try:
            cv2.imwrite(str(tmp), undistorted, [cv2.IMWRITE_JPEG_QUALITY, 95])
            # `replace_with_retry`, not `os.replace`. Windows refuses a
            # replace onto a destination any handle has open, and after a
            # recalibration this path OVERWRITES frames a reader may be
            # holding -- which is exactly the WinError 5 the retry exists
            # for, in a function that had no tests until one held a frame
            # open and it raised.
            replace_with_retry(tmp, target)
        finally:
            tmp.unlink(missing_ok=True)
        written += 1
    write_json_atomic(workspace.camera_path, camera.to_json_dict())
    return camera, written


# ---------------------------------------------------------------------------
# The solve.



# THE SPARSE POINT COLOUR THAT IS PERSISTED (privacy lane L1; review 1, m5).
#
# pycolmap's `point.color` is the mean of the observing pixels of the images
# COLMAP was given, and those are the RAW capture frames, undistorted: on the
# canonical capture 536 of 14,415 points take their colour ONLY from pixels the
# face redactor removed. The page stopped drawing it (`render.DRAWABLE_RGB_SOURCES`),
# but `solution.npz` `rgb` and `derived/*/points.json` `rgb` still persisted it,
# and a persisted colour attribute is an appearance artifact. So every writer
# writes this neutral grey instead -- the same grey the page draws for "no
# colour" (`render.NEUTRAL_POINT_COLOUR`) -- and the fields stay, with their
# shapes, so every reader of either file is unchanged. Nothing recolours from
# redacted keyframes yet; when something does it must say so with an
# `rgb_source` the page allows.
WITHHELD_POINT_RGB = (138, 138, 138)


def withheld_rgb(n: int) -> np.ndarray:
    """(n, 3) uint8 of `WITHHELD_POINT_RGB`."""
    return np.tile(np.asarray(WITHHELD_POINT_RGB, np.uint8), (int(n), 1)).reshape(-1, 3)

@dataclass
class Solution:
    """A global solution over a set of keyframes, as persisted."""

    solver: str
    solved_at: float
    input_digest: str | None
    keyframe_ids: list[str]
    # keyframe_id -> {component, rotation (R_cw, 9 floats row-major), translation (t_cw), observations}
    poses: dict[str, dict]
    components: list[dict]
    xyz: np.ndarray            # (N, 3) float32, in the component's frame
    rgb: np.ndarray            # (N, 3) uint8
    component: np.ndarray      # (N,) int32
    first_keyframe: np.ndarray  # (N,) int32 index into keyframe_ids
    track_length: np.ndarray   # (N,) int32
    error: np.ndarray          # (N,) float32, mean reprojection px
    observations: np.ndarray   # (M, 3) int32 [keyframe index, feature index, point index]
    observation_xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.float32))  # (M, 2) undistorted px
    camera: dict | None = None  # the pinhole camera the observations are expressed in
    timing: dict = field(default_factory=dict)
    # WHAT RAN, for the evidence gate and the harness (contract
    # WORLD-BUILDER-COMPONENTS §2.5). Both None on a solution written before
    # 2026-09-23 (and on any written by hand), which a reader must take as
    # "today's solve: unmasked, unseeded" -- never as an error.
    #   transients: the solver-image masks (`solve_masks`): `state` applied /
    #               partial / unavailable, `detail`, `rule`, `cache_hits`,
    #               `computed`, images masked and unmasked.
    #   solve:      `seed` (None = unseeded), `threads` (the mapper's; 1 when
    #               seeded, -1 = every core), the other steps' threads, the
    #               two-view RANSAC seed, and which feature database was used.
    transients: dict | None = None
    solve: dict | None = None
    #   gate:       the evidence gate on the final solve (`coherence_publish.py`,
    #               `TOWER_WORLD_SOLVE_GATE`): `state` applied / failed, the
    #               gate's `params` and `params_digest`, `masks_applied`,
    #               `metric_available`, the depth stage and metric scale that
    #               fed it. None: the gate did not run -- every solve before it
    #               existed, and every solve with it off.
    gate: dict | None = None

    @property
    def horizon(self) -> set[str]:
        return set(self.keyframe_ids)


def _pose_matrices(entry: dict):
    r_cw = np.asarray(entry["rotation"], dtype=np.float64).reshape(3, 3)
    t_cw = np.asarray(entry["translation"], dtype=np.float64).reshape(3)
    r_wc = r_cw.T
    centre = -r_wc @ t_cw
    return r_wc, centre


def sweep_workspace(workspace: SolveWorkspace) -> int:
    """Clear staging files left by solve children that were killed.

    `BackgroundSolver` terminates a child that outstays its budget and the
    final solve on a hard stop, and `TerminateProcess` runs no `finally`.
    With per-writer staging names that leaves one file per kill, forever --
    up to 13 MB each for a `solution.npz` at 6,000 keyframes. Swept here
    because this is the directory those writers write into, and a solve is
    the moment nobody else is using it.
    """
    if not workspace.root.is_dir():
        return 0
    swept = sweep_abandoned_staging(workspace.root)
    if workspace.images_dir.is_dir():
        swept += sweep_abandoned_staging(workspace.images_dir)
    # The solver's COLMAP masks (`solve_masks`), written the same atomic way.
    masks = workspace.root / "masks"
    if masks.is_dir():
        swept += sweep_abandoned_staging(masks)
    return swept


def write_solution(workspace: SolveWorkspace, solution: Solution) -> None:
    """Publish a solution: the arrays first, the metadata last.

    ORDER IS LOAD-BEARING, because `load_solution` requires BOTH files and
    the metadata is what carries `schema_version` and `solved_at`. Writing
    the arrays first means the newest `solution.json` a reader can see
    always has arrays at least as new behind it. The reverse order
    publishes a claim before the evidence.

    Both writes are atomic. `solution.npz` was not until 2026-09-09, and a
    reader in the builder process caught the solver child mid-zip; see
    `storage.write_bytes_atomic`.
    """
    workspace.root.mkdir(parents=True, exist_ok=True)
    write_bytes_atomic(
        workspace.arrays_path,
        lambda handle: np.savez_compressed(
            handle,
            xyz=solution.xyz.astype(np.float32),
            rgb=withheld_rgb(len(solution.xyz)),   # never the solver's colour
            component=solution.component.astype(np.int32),
            first_keyframe=solution.first_keyframe.astype(np.int32),
            track_length=solution.track_length.astype(np.int32),
            error=solution.error.astype(np.float32),
            observations=solution.observations.astype(np.int32).reshape(-1, 3),
            observation_xy=solution.observation_xy.astype(np.float32).reshape(-1, 2),
        ),
    )
    meta = {
        "schema_version": SOLUTION_SCHEMA_VERSION,
        "solver": solution.solver,
        "solved_at": solution.solved_at,
        "input_digest": solution.input_digest,
        "keyframe_ids": solution.keyframe_ids,
        "components": solution.components,
        "poses": solution.poses,
        "camera": solution.camera,
        "timing": solution.timing,
    }
    # Additive, and only when recorded: the schema version is unchanged, so
    # every reader of a solution written before these fields still reads it.
    if solution.transients is not None:
        meta["transients"] = solution.transients
    if solution.solve is not None:
        meta["solve"] = solution.solve
    if solution.gate is not None:
        meta["gate"] = solution.gate
    write_json_atomic(workspace.solution_path, meta)


def load_solution(store, world_id: str, session_id: str) -> Solution | None:
    """The persisted solution, or None. Never raises: an unreadable or
    half-written solution is absent, and the build proceeds without it."""
    workspace = workspace_for(store, world_id, session_id)
    if not workspace.solution_path.exists() or not workspace.arrays_path.exists():
        return None
    try:
        meta = read_json_closed(workspace.solution_path)
        if meta.get("schema_version") != SOLUTION_SCHEMA_VERSION:
            return None
        # Read the bytes with the handle closed, then parse in memory.
        # `np.load` on a PATH keeps the zip open for the life of the
        # NpzFile, and a held handle is what blocks a writer's os.replace
        # on Windows -- so the lazy form would make this reader the very
        # obstacle the write path has to retry around.
        with np.load(io.BytesIO(read_bytes_closed(workspace.arrays_path))) as arrays:
            return Solution(
                solver=meta["solver"],
                solved_at=float(meta["solved_at"]),
                input_digest=meta.get("input_digest"),
                keyframe_ids=list(meta["keyframe_ids"]),
                poses=dict(meta["poses"]),
                components=list(meta["components"]),
                xyz=arrays["xyz"],
                rgb=arrays["rgb"],
                component=arrays["component"],
                first_keyframe=arrays["first_keyframe"],
                track_length=arrays["track_length"],
                error=arrays["error"],
                observations=arrays["observations"].reshape(-1, 3),
                observation_xy=(arrays["observation_xy"].reshape(-1, 2)
                                if "observation_xy" in arrays else np.zeros((0, 2), np.float32)),
                camera=meta.get("camera"),
                timing=dict(meta.get("timing") or {}),
                transients=meta.get("transients"),
                solve=meta.get("solve"),
                gate=meta.get("gate"),
            )
    except Exception as exc:  # noqa: BLE001 -- see below; the narrow tuple IS the bug
        # DELIBERATELY BROAD, and the breadth is the fix rather than a
        # shortcut around one.
        #
        # This used to catch `(OSError, KeyError, ValueError,
        # json.JSONDecodeError)`, chosen to make the docstring's "never
        # raises" true. It did not. A `.npz` is a zip read by numpy, and a
        # torn one raises out of two libraries whose exception types are
        # not part of anyone's contract: measured over 15 s of a real
        # reader/writer race, 48,854 `EOFError`, 10,295
        # `zipfile.BadZipFile`, plus `BadZipFile: Bad magic number for
        # central directory` and `Truncated file header`. `EOFError` --
        # the MOST common by five to one -- descends from Exception, and
        # `BadZipFile` from Exception alone; neither is an OSError. In the
        # same run `load_solution` returned None exactly ZERO times.
        #
        # One of those escaped on the 2026-09-09 walk and ended a session
        # holding 795 keyframes and 26,634 points: `BadZipFile: File is
        # not a zip file` reached the builder's BaseException handler and
        # became `finalization.interrupted`.
        #
        # An allowlist of exception types for a best-effort read of a
        # foreign binary format is incomplete by construction, and the
        # cost of being wrong is losing a capture. The write is atomic now
        # (`storage.write_bytes_atomic`), so a torn read should not recur
        # -- this is the second line of defence, and it must not have a
        # gap. A solution is DERIVED: re-solving rebuilds it from the
        # journal, so "absent" is always a survivable reading of
        # "unreadable", and never worth ending a capture over.
        #
        # BaseException still propagates, so KeyboardInterrupt and the
        # supervisor's stop still stop this process.
        logger.warning(
            "global solve: solution for %s unreadable (%s: %s); continuing without it",
            world_id, type(exc).__name__, exc,
        )
        return None


def vocabulary_tree_cache_dir() -> Path:
    """Where COLMAP caches the vocabulary tree it downloads.

    `~/.cache/colmap`, and NOT the venv: a fresh checkout on a machine that
    has solved before is fine, and a fresh machine is not. The code never
    sets `vocab_tree_path`, so COLMAP resolves and caches it itself.
    """
    return Path.home() / ".cache" / "colmap"


def vocabulary_tree_cached() -> bool:
    """Whether loop detection can run without reaching the network.

    Deliberately a filesystem check rather than a try/except around the
    solve: COLMAP's failure to fetch the tree is a glog CHECK that aborts
    the process, so by the time it is observable there is nothing left to
    handle. Being wrong in this direction is survivable -- a tree we fail
    to find means a solve without loop detection, not a dead one.
    """
    directory = vocabulary_tree_cache_dir()
    try:
        return any(directory.glob("*vocab_tree*"))
    except OSError:
        return False


def _quiet_pycolmap():
    try:
        import pycolmap
        pycolmap.logging.minloglevel = 2  # errors only; glog is very chatty
    except Exception:
        pass


class _FromSettings:
    """`solve(seed=...)` not given: the final solve reads `TOWER_WORLD_SOLVE_SEED`."""

    def __repr__(self) -> str:
        return "FROM_SETTINGS"


FROM_SETTINGS = _FromSettings()


def resolve_run_options(*, final: bool, masks: bool | None, seed) -> tuple[bool, int | None]:
    """(masks, seed) for one solve.

    The two settings belong to the FINAL solve only. The background solves of
    a walk keep today's recipe whatever the settings say: masks need the GPU the
    live surface is using, and a single-thread mapper would slow the live
    world's convergence 3.3x (research D1). An explicit argument wins either
    way, so a script or a test can ask for exactly what it wants.
    """
    from tower.config import world_solve_masks_setting, world_solve_seed_setting  # noqa: PLC0415

    if masks is None:
        masks = bool(final) and world_solve_masks_setting()
    if seed is FROM_SETTINGS:
        seed = world_solve_seed_setting() if final else None
    if seed is not None:
        seed = int(seed)
        if seed < 0:
            raise ValueError(f"a solve seed is a non-negative integer, not {seed}")
    return bool(masks), seed


def solve(
    store,
    world_id: str,
    session_id: str,
    *,
    capture_dirs=(),
    final: bool = False,
    num_threads: int | None = None,
    min_image_observations: int = MIN_IMAGE_OBSERVATIONS,
    overlap: int = SEQUENTIAL_OVERLAP,
    loop_detection: bool | None = None,
    input_digest: str | None = None,
    masks: bool | None = None,
    seed=FROM_SETTINGS,
    transient_backend_factory=None,
    mask_device_probe=None,
    gate: bool | None = None,
) -> dict:
    """Run the recipe over the session's current keyframes and persist the
    solution. Returns a summary dict (what the CLI prints).

    `gate`: the evidence gate before publish (`coherence_publish.py`). None
    reads `TOWER_WORLD_SOLVE_GATE` for a FINAL solve (off by default) and is
    off for every background solve; off, the solution is published exactly as
    the solver returned it.

    Idempotent and incremental: images already undistorted, features already
    extracted and pairs already matched are skipped by the workspace and by
    pycolmap respectively. The MODEL is rebuilt from scratch every call --
    GLOMAP is fast enough (E8: 43 s for 438 keyframes) and a from-scratch
    solve has no way to inherit a wrong decision.

    TWO OPTIONS, BOTH OFF BY DEFAULT, BOTH FOR THE FINAL SOLVE
    (`resolve_run_options`; the settings are `world_solve_masks` and
    `world_solve_seed` in `tower/config.py`). Off, every call into pycolmap is
    exactly today's.

      * `masks`: no match touching the wearer's hands, arms or held phone
        reaches the model (`solve_masks.py`). With a walk database, it is
        extracted and matched exactly as today and the solve maps a filtered
        copy of it (`walk-database-filtered`, the run's arm A1h); without one,
        the masks go to extraction, into a database of their own
        (`re-extracted`, arm A1). `solve.masking` says which. When the
        detector cannot run the solve runs unmasked on today's database and
        `transients.state` says why.
      * `seed`: every mapper seed, pycolmap's global seed and the two-view
        RANSAC seed are set and mapping runs on one thread -- the determinism
        hygiene of the run's experiment driver (lane `coherence_exp/driver.py`),
        without which GLOMAP is not reproducible (research D1 §2.4).
        Extraction and matching keep their threads, as in the driver. What is
        reproducible is the MAPPING, given its feature database -- measured
        on the target walk (run P3-PM): two seeded one-thread GLOMAP runs on
        one database were bit-identical, two unseeded ones were not. The
        database itself is not reproducible: SIFT extraction is, but
        multi-threaded matching is not (62 of 7,450 sequential pairs matched
        differently twice; one thread matched all 7,450 identically, about 6x
        slower), and loop detection re-run on an existing database proposes
        new pairs. Without masks the database is also the one the walk's
        background solves accumulated.

    Both are recorded in the solution (`transients`, `solve`) and in the
    returned summary, so the evidence gate and the harness can tell what ran.
    """
    available, reason = solver_available()
    if not available:
        return {"solved": False, "reason": reason}
    want_masks, seed = resolve_run_options(final=final, masks=masks, seed=seed)
    sweep_workspace(workspace_for(store, world_id, session_id))
    import pycolmap

    _quiet_pycolmap()
    started = time.perf_counter()
    keyframes = store.read_keyframes(world_id, session_id)
    if len(keyframes) < 2:
        return {"solved": False, "reason": "fewer than two keyframes"}
    workspace = workspace_for(store, world_id, session_id)
    try:
        camera, written = prepare_images(
            store, world_id, session_id, keyframes, capture_dirs=capture_dirs
        )
    except UndistortionUnavailable as exc:
        return {"solved": False, "reason": f"cannot undistort: {exc}"}
    prepared = time.perf_counter()

    names = [keyframe_image_name(k) for k in keyframes]
    present = [n for n in names if (workspace.images_dir / n).exists()]
    if len(present) < 2:
        return {"solved": False, "reason": "fewer than two readable frames"}

    threads = num_threads if num_threads is not None else -1
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "PINHOLE"
    reader.camera_params = ",".join(str(v) for v in (camera.fx, camera.fy, camera.cx, camera.cy))

    # THE FEATURE DATABASE, and the masks that decide which one. Without masks
    # it is `database.db`, shared with the walk's background solves -- today.
    # With masks there are two paths (`solve_masks.MASKING_*`): the walk's
    # database is extracted and matched exactly as today and a FILTERED COPY
    # is mapped, or, with no walk database to filter, the masks go to
    # extraction itself, into a fresh database of their own.
    database_path = workspace.database_path
    database_existed = database_path.exists()
    masks_record = _masks_off_record()
    masks = None
    masking = None
    if want_masks:
        masks, masks_record = _ensure_solver_masks(
            workspace, present, keyframes, camera,
            backend_factory=transient_backend_factory, device_probe=mask_device_probe)
        if masks is not None:
            if _walk_database_usable(database_path):
                masking = _MASKING_FILTERED
            else:
                database_path, masks_record = _reextract_masked(workspace, reader, masks,
                                                                present, masks_record)
                database_existed = masks_record.get("database_reused", False)
                masking = _MASKING_REEXTRACTED
        masks_record["database"] = database_path.name
    masked = time.perf_counter()

    extraction = pycolmap.FeatureExtractionOptions()
    extraction.num_threads = threads
    extraction.sift.max_num_features = MAX_FEATURES
    pycolmap.extract_features(
        database_path, workspace.images_dir, image_names=present,
        camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader,
        extraction_options=extraction,
    )
    extracted = time.perf_counter()

    matching = pycolmap.FeatureMatchingOptions()
    matching.num_threads = threads
    pairing = pycolmap.SequentialPairingOptions()
    pairing.overlap = overlap
    pairing.quadratic_overlap = False
    # Loop detection needs a vocabulary tree COLMAP downloads on first use
    # and caches in the USER's home. It is what makes a live world converge
    # instead of fragmenting -- 16 components to 5 on the 2026-09-09
    # capture -- so every solve asks for it now, not just the final one.
    #
    # WHICH IS WHY THE ABSENCE OF THE TREE HAS TO BE CHECKED HERE. A missing
    # tree is not a refusal: `match_sequential` is outside any try/except,
    # and COLMAP's failure to fetch the file is a glog CHECK, so the process
    # dies of `abort()` with exit code 3 and no Python exception to catch.
    # Measured with an empty cache and no network:
    #
    #     file.cc:507] Check failed: blob.has_value() Failed to download file
    #     *** Aborted ***                                        EXIT=3
    #
    # End to end that means EVERY solve dies, the manifest carries no
    # `global_solve` at all, and the world ships with every segment refused
    # -- the "87 disconnected fragments" outcome, from a new cause, on a
    # machine that merely has no network. Turning loop detection off instead
    # costs the convergence and keeps the walk.
    wanted = bool(loop_detection) if loop_detection is not None else False
    if wanted and not vocabulary_tree_cached():
        logger.warning(
            "global solve: no vocabulary tree in %s, so loop detection is off for "
            "this solve. Fetching it needs a network and COLMAP aborts the process "
            "rather than failing the call, so it is not attempted mid-walk. The "
            "world will still solve, in more pieces than it would otherwise. "
            "scripts/world_builder_env_check.py reports this before a walk.",
            vocabulary_tree_cache_dir(),
        )
        wanted = False
    pairing.loop_detection = wanted
    seeded = seed is not None
    verification = None
    if seeded:
        # The two-view RANSAC is seeded too (the experiment driver's
        # `verification_seed`): otherwise the verified inlier sets, and so
        # the view graph GLOMAP averages, differ run to run.
        verification = pycolmap.TwoViewGeometryOptions()
        verification.ransac.random_seed = seed
    _match_sequential(pycolmap, database_path, matching, pairing, verification)
    revisits = {"listed": 0, "verified": 0, "detail": None}
    if final:
        revisits = _match_revisit_pairs(
            pycolmap, store, world_id, session_id, workspace, database_path, present,
            matching, verification)
    if masking == _MASKING_FILTERED:
        try:
            database_path, masks_record = _filter_walk_database(
                pycolmap, workspace, masks, masks_record, verification)
        except Exception as exc:  # noqa: BLE001 -- fall back to re-extraction, recorded
            logger.exception("global solve: filtering the walk database failed; the masks "
                             "go to extraction instead")
            masks_record = dict(masks_record, filter_failed=f"{type(exc).__name__}: {exc}")
            database_path, masks_record = _reextract_masked(workspace, reader, masks,
                                                            present, masks_record)
            masking = _MASKING_REEXTRACTED
            pycolmap.extract_features(
                database_path, workspace.images_dir, image_names=present,
                camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader,
                extraction_options=extraction,
            )
            _match_sequential(pycolmap, database_path, matching, pairing, verification)
            if final:
                revisits = _match_revisit_pairs(
                    pycolmap, store, world_id, session_id, workspace, database_path, present,
                    matching, verification)
        masks_record["database"] = database_path.name
    if masking is not None:
        masks_record["masking"] = masking
    matched = time.perf_counter()

    shutil.rmtree(workspace.sparse_dir, ignore_errors=True)
    workspace.sparse_dir.mkdir(parents=True)
    # ONE thread when seeded: GLOMAP is reproducible only then (D1 §2.4).
    map_threads = 1 if seeded else threads
    solver = SOLVER_GLOMAP
    options = pycolmap.GlobalPipelineOptions()
    options.num_threads = map_threads
    options.mapper.bundle_adjustment.refine_focal_length = False
    options.mapper.bundle_adjustment.refine_principal_point = False
    options.mapper.bundle_adjustment.refine_extra_params = False
    if seeded:
        _seed_global_options(options, seed)
        pycolmap.set_random_seed(seed)
    try:
        reconstructions = pycolmap.global_mapping(
            database_path, workspace.images_dir, workspace.sparse_dir, options=options
        )
    except Exception as exc:  # a solver failure is a refusal, not a crash
        logger.warning("global solve: global mapping raised %s", exc)
        reconstructions = {}
    if not reconstructions:
        solver = SOLVER_INCREMENTAL
        inc = pycolmap.IncrementalPipelineOptions()
        inc.num_threads = map_threads
        inc.ba_refine_focal_length = False
        inc.ba_refine_principal_point = False
        inc.ba_refine_extra_params = False
        if seeded:
            _seed_incremental_options(inc, seed)
            pycolmap.set_random_seed(seed)
        try:
            reconstructions = pycolmap.incremental_mapping(
                database_path, workspace.images_dir, workspace.sparse_dir, options=inc
            )
        except Exception as exc:
            logger.warning("global solve: incremental mapping raised %s", exc)
            reconstructions = {}
    mapped = time.perf_counter()

    solution = _solution_from_reconstructions(
        reconstructions, keyframes, solver=solver, input_digest=input_digest,
        min_image_observations=min_image_observations, camera=camera,
    )
    solution.timing = {
        "prepare_s": round(prepared - started, 3),
        "extract_s": round(extracted - masked, 3),
        "match_s": round(matched - extracted, 3),
        "map_s": round(mapped - matched, 3),
        "images_undistorted": written,
        "final": bool(final),
    }
    if want_masks:
        solution.timing["masks_s"] = round(masked - prepared, 3)
    solution.transients = masks_record
    solution.solve = {
        "seed": seed,
        # The MAPPER's threads, which is what reproducibility turns on: 1 when
        # seeded; otherwise pycolmap's -1, "every core", or the caller's count.
        "threads": map_threads,
        "seeded": seeded,
        "extraction_threads": threads,
        "matching_threads": threads,
        "verification_seed": seed if seeded else None,
        "database": database_path.name,
        "database_existed": bool(database_existed),
        # How the masks reached the model: `walk-database-filtered`,
        # `re-extracted`, or None for an unmasked solve.
        "masking": masking,
        "final": bool(final),
        "pycolmap": getattr(pycolmap, "__version__", None),
        # The live relocalizer's verified revisit links, matched explicitly.
        "revisit_pairs": revisits,
    }
    # PUBLISH. With the evidence gate on (a final solve, `TOWER_WORLD_SOLVE_GATE`)
    # the candidate first gets its depth stage, metric scale and gate, and is
    # published RELABELLED -- pieces the gate did not attach are their own
    # components -- followed by `components.json` and the depth hand-off to the
    # surface. Off, this is `write_solution(workspace, solution)`.
    from tower.world_builder import coherence_publish  # noqa: PLC0415

    solution, _gate_record = coherence_publish.gate_and_publish(
        store, world_id, session_id, workspace, solution, final=final, gate=gate,
        database_path=database_path, keyframes=keyframes, write=write_solution)
    return {
        "solved": True,
        "solver": solver,
        "keyframes": len(keyframes),
        "keyframes_posed": sum(1 for p in solution.poses.values() if p["observations"] >= min_image_observations),
        "components": solution.components,
        "points": int(len(solution.xyz)),
        "timing": solution.timing,
        "workspace": str(workspace.root),
        "transients": solution.transients,
        "solve": solution.solve,
        "gate": solution.gate,
    }


def _seed_global_options(options, seed: int) -> None:
    """Every random number generator GLOMAP's pipeline owns, and one thread.
    The experiment driver's `_mapper_options` for `glomap`, verbatim."""
    options.mapper.num_threads = 1
    options.random_seed = seed
    options.mapper.random_seed = seed
    options.mapper.rotation_averaging.random_seed = seed
    options.mapper.global_positioning.random_seed = seed
    options.mapper.retriangulation.random_seed = seed


def _seed_incremental_options(inc, seed: int) -> None:
    """The same for the incremental fallback (driver `_mapper_options`), plus
    the mapper's own thread count, which the driver left to `num_threads`."""
    inc.mapper.num_threads = 1
    inc.random_seed = seed
    inc.mapper.random_seed = seed
    inc.triangulation.random_seed = seed


REVISIT_PAIRS_FILENAME = "revisit_pairs.txt"
# A revisit pair counts as verified at COLMAP's own two-view floor
# (`TwoViewGeometryOptions.min_num_inliers`, 15), the floor the solve's view
# graph uses for every other pair.
REVISIT_VERIFIED_MIN_INLIERS = 15


def _match_revisit_pairs(pycolmap, store, world_id, session_id, workspace, database_path,
                         present, matching, verification) -> dict:
    """Match the live relocalizer's revisit links (`relocalizer.revisit_pairs`)
    explicitly, in the final solve's database, with its matching and
    verification options -- and its masks, which apply to every pair because
    they applied at extraction.

    Sequential matching reaches 20 keyframes either side and loop detection
    queries every tenth keyframe; a look-back revisit is exactly the pair
    neither is sure to propose. No links (no relocalizer, an old session, a
    walk that never lost tracking) is today's solve, with no call made.
    Never raises: a failure costs the links, and the record says so.
    """
    record = {"listed": 0, "verified": 0, "detail": None}
    try:
        from tower.world_builder.relocalizer import revisit_pairs  # noqa: PLC0415

        pairs = revisit_pairs(store.session_dir(world_id, session_id))
    except Exception as exc:  # noqa: BLE001 -- no links is today's solve
        record["detail"] = f"revisit links unreadable ({type(exc).__name__}: {exc})"
        return record
    have = set(present)
    pairs = [(a, b) for a, b in pairs if a in have and b in have and a != b]
    record["listed"] = len(pairs)
    if not pairs:
        return record
    path = workspace.root / REVISIT_PAIRS_FILENAME
    data = "".join(f"{a} {b}\n" for a, b in pairs).encode("utf-8")
    try:
        write_bytes_atomic(path, lambda handle: handle.write(data))
        imported = pycolmap.ImportedPairingOptions()
        imported.match_list_path = str(path)
        if verification is not None:
            pycolmap.match_image_pairs(database_path, matching_options=matching,
                                       pairing_options=imported,
                                       verification_options=verification)
        else:
            pycolmap.match_image_pairs(database_path, matching_options=matching,
                                       pairing_options=imported)
    except Exception as exc:  # noqa: BLE001 -- recorded, the solve goes on
        logger.warning("global solve: matching %d revisit links failed: %s", len(pairs), exc)
        record["detail"] = f"matching failed ({type(exc).__name__}: {exc})"
        return record
    record["verified"] = _verified_pair_count(database_path, pairs)
    return record


def _verified_pair_count(database_path, pairs) -> int | None:
    """How many of `pairs` (image names) the database holds as verified pairs,
    or None when the database cannot be read."""
    import sqlite3  # noqa: PLC0415

    base = 2147483647  # COLMAP's kMaxNumImages: pair_id = min_id * base + max_id
    try:
        con = sqlite3.connect(f"file:{Path(database_path).as_posix()}?mode=ro", uri=True)
        try:
            ids = {name: iid for iid, name in con.execute("select image_id, name from images")}
            count = 0
            for a, b in pairs:
                if a not in ids or b not in ids:
                    continue
                lo, hi = sorted((ids[a], ids[b]))
                row = con.execute("select rows from two_view_geometries where pair_id = ?",
                                  (lo * base + hi,)).fetchone()
                if row is not None and (row[0] or 0) >= REVISIT_VERIFIED_MIN_INLIERS:
                    count += 1
        finally:
            con.close()
    except sqlite3.Error:
        return None
    return count


def _masks_off_record() -> dict:
    from tower.world_builder.solve_masks import off_record  # noqa: PLC0415

    return off_record()


# `solve_masks.MASKING_FILTERED` / `MASKING_REEXTRACTED`, spelled here so this
# module does not import the mask module (and its detector) to solve unmasked.
_MASKING_FILTERED = "walk-database-filtered"
_MASKING_REEXTRACTED = "re-extracted"


def _match_sequential(pycolmap, database_path, matching, pairing, verification) -> None:
    """Today's call when unseeded -- no `verification_options` at all -- and the
    seeded two-view RANSAC otherwise."""
    if verification is not None:
        pycolmap.match_sequential(
            database_path, matching_options=matching, pairing_options=pairing,
            verification_options=verification,
        )
    else:
        pycolmap.match_sequential(
            database_path, matching_options=matching, pairing_options=pairing
        )


def _ensure_solver_masks(workspace, present, keyframes, camera, *,
                         backend_factory=None, device_probe=None):
    """(the masks, their record) when they can be applied, else (None, a record
    saying why). Never raises: whatever the mask step does, the solve runs, and
    the record says whether it ran masked."""
    from tower.world_builder import solve_masks  # noqa: PLC0415

    ids = {keyframe_image_name(k): k.keyframe_id for k in keyframes}
    try:
        result = solve_masks.ensure_solver_masks(
            workspace, present, keyframe_ids=ids, shape=(camera.height, camera.width),
            backend_factory=backend_factory, device_probe=device_probe)
    except Exception as exc:  # noqa: BLE001 -- a mask failure is an unmasked solve, recorded
        logger.exception("global solve: the transient mask step failed; this solve is unmasked")
        return None, solve_masks.failed_record(f"{type(exc).__name__}: {exc}")
    record = result.record()
    if not result.available:
        logger.warning(
            "global solve: transient masks were requested for the final solve and "
            "are NOT applied (%s: %s); this solve is unmasked and says so",
            result.state, result.detail)
        return None, record
    return result, record


def _walk_database_usable(path) -> bool:
    from tower.world_builder.solve_masks import walk_database_usable  # noqa: PLC0415

    return walk_database_usable(path)


def _reextract_masked(workspace, reader, masks, present, record):
    """The fallback: the masks go to extraction, into `database.masked.db`."""
    from tower.world_builder import solve_masks  # noqa: PLC0415

    database, info = solve_masks.masked_database(workspace, masks, all_names=present)
    reader.mask_path = str(solve_masks.masks_dir(workspace))
    return database, dict(record, database=info["database"], database_reused=info["reused"],
                          database_rebuilt_because=info["rebuilt_because"])


def _filter_walk_database(pycolmap, workspace, masks, record, verification):
    """The walk database, copied and stripped of every match touching a mask
    (`solve_masks.filter_walk_database`), with the pairs that changed
    re-verified under the solve's own two-view options."""
    from tower.world_builder import solve_masks  # noqa: PLC0415

    database, pairs_path, info = solve_masks.filter_walk_database(workspace, masks)
    info["pairs_reverified"] = 0
    if pairs_path is not None:
        options = verification if verification is not None else pycolmap.TwoViewGeometryOptions()
        pycolmap.verify_matches(database, pairs_path, options=options)
        info["pairs_reverified"] = info["pairs_listed"]
    return database, dict(record, filter=info)


def _solution_from_reconstructions(
    reconstructions, keyframes, *, solver, input_digest, min_image_observations, camera=None
) -> Solution:
    keyframe_ids = [k.keyframe_id for k in keyframes]
    index_by_name = {keyframe_image_name(k): i for i, k in enumerate(keyframes)}
    poses: dict[str, dict] = {}
    components: list[dict] = []
    xyz, rgb, comp, first, track, err, obs, obs_xy = [], [], [], [], [], [], [], []
    point_base = 0
    models = sorted(reconstructions.items(), key=lambda kv: -kv[1].num_reg_images())
    component_index = 0
    for _, rec in models:
        if rec.num_reg_images() < MIN_MODEL_IMAGES:
            continue
        image_index = {}
        keypoints = {}
        posed_here = 0
        for image in rec.images.values():
            if not image.has_pose or image.name not in index_by_name:
                continue
            kf_index = index_by_name[image.name]
            keypoints[image.image_id] = image.points2D
            cam_from_world = image.cam_from_world()
            observations = int(image.num_points3D)
            poses[keyframe_ids[kf_index]] = {
                "component": component_index,
                "rotation": [float(v) for v in np.asarray(cam_from_world.rotation.matrix()).reshape(-1)],
                "translation": [float(v) for v in cam_from_world.translation],
                "observations": observations,
            }
            image_index[image.image_id] = kf_index
            if observations >= min_image_observations:
                posed_here += 1
        n_points = 0
        errors = []
        for point_id, point in rec.points3D.items():
            elements = point.track.elements
            owners = [image_index[e.image_id] for e in elements if e.image_id in image_index]
            if not owners:
                continue
            xyz.append([float(v) for v in point.xyz])
            rgb.append(list(WITHHELD_POINT_RGB))   # never point.color: raw pixels
            comp.append(component_index)
            first.append(min(owners))
            track.append(len(elements))
            err.append(float(point.error))
            errors.append(float(point.error))
            for element in elements:
                if element.image_id in image_index:
                    obs.append([image_index[element.image_id], int(element.point2D_idx), point_base + n_points])
                    xy = keypoints[element.image_id][element.point2D_idx].xy
                    obs_xy.append([float(xy[0]), float(xy[1])])
            n_points += 1
        point_base += n_points
        components.append({
            "index": component_index,
            "images": int(rec.num_reg_images()),
            "images_supported": posed_here,
            "points": n_points,
            "mean_error_px": float(np.mean(errors)) if errors else None,
        })
        component_index += 1
    return Solution(
        solver=solver,
        solved_at=time.time(),
        input_digest=input_digest,
        keyframe_ids=keyframe_ids,
        poses=poses,
        components=components,
        xyz=np.asarray(xyz, dtype=np.float32).reshape(-1, 3),
        rgb=np.asarray(rgb, dtype=np.uint8).reshape(-1, 3),
        component=np.asarray(comp, dtype=np.int32),
        first_keyframe=np.asarray(first, dtype=np.int32),
        track_length=np.asarray(track, dtype=np.int32),
        error=np.asarray(err, dtype=np.float32),
        observations=np.asarray(obs, dtype=np.int32).reshape(-1, 3),
        observation_xy=np.asarray(obs_xy, dtype=np.float32).reshape(-1, 2),
        camera=camera.to_json_dict() if camera is not None else None,
    )


def reprojection_summary(solution: Solution) -> dict | None:
    """Reproject the solution's own observations through its own poses.

    The chain's coherence report reprojects `support.json` through ORB
    keypoints it re-detects on the redacted keyframe copies; the solver's
    features are SIFT on undistorted raw frames, so that index does not
    apply to solved segments and is not written for them. This is the
    equivalent self-check for the solution, in the solver's pinhole camera.
    Returns per-component and overall pixel-error percentiles, or None
    when the solution carries no observation coordinates.
    """
    if solution.camera is None or not len(solution.observation_xy):
        return None
    cam = solution.camera
    fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
    obs = solution.observations
    xy = solution.observation_xy.astype(np.float64)
    errors = np.full(len(obs), np.nan)
    for kf_index, keyframe_id in enumerate(solution.keyframe_ids):
        entry = solution.poses.get(keyframe_id)
        if entry is None:
            continue
        rows = np.nonzero(obs[:, 0] == kf_index)[0]
        if not len(rows):
            continue
        r = np.asarray(entry["rotation"], dtype=np.float64).reshape(3, 3)
        t = np.asarray(entry["translation"], dtype=np.float64)
        pts = solution.xyz[obs[rows, 2]].astype(np.float64)
        cam_pts = (r @ pts.T).T + t
        z = cam_pts[:, 2]
        ok = z > 1e-9
        u = fx * cam_pts[:, 0] / np.where(ok, z, 1.0) + cx
        v = fy * cam_pts[:, 1] / np.where(ok, z, 1.0) + cy
        e = np.hypot(u - xy[rows, 0], v - xy[rows, 1])
        e[~ok] = np.inf
        errors[rows] = e
    valid = np.isfinite(errors)
    if not valid.any():
        return None

    def stats(v):
        return {
            "count": int(len(v)), "median": float(np.median(v)),
            "p95": float(np.percentile(v, 95)), "p99": float(np.percentile(v, 99)),
            "max": float(v.max()), "over_3px": float((v > 3.0).mean()),
        }

    per_component = {}
    for c in sorted(set(solution.component.tolist())):
        sel = valid & (solution.component[obs[:, 2]] == c)
        if sel.any():
            per_component[str(c)] = stats(errors[sel])
    return {
        "camera": cam, "behind_camera": int(np.isinf(errors).sum()),
        "overall": stats(errors[valid]), "per_component": per_component,
    }


# ---------------------------------------------------------------------------
# Merge: express the solution through the derived tree the phone reads.


@dataclass
class MergeResult:
    pose_rows: list[dict]
    point_rows: list[dict]
    support_rows: list[list[int]]
    placements: list[SegmentPlacement]
    # segment_index -> {"replaced": bool, "coverage": str, ...}
    segments: dict[int, dict]
    summary: dict


def coverage_for(posed: int, keyframes: int, median_observations: float | None, points: int) -> str:
    if posed == 0 or points == 0:
        return COVERAGE_UNRESOLVED
    if (
        median_observations is not None
        and median_observations >= 2 * MIN_IMAGE_OBSERVATIONS
        and posed >= max(2, keyframes // 2)
        and points >= 50
    ):
        return COVERAGE_CONFIDENT
    return COVERAGE_PARTIAL


def merge(
    keyframes: list[Keyframe],
    pose_rows: list[dict],
    point_rows: list[dict],
    support_rows: list[list[int]] | None,
    solution: Solution,
    *,
    input_digest: str,
    min_image_observations: int = MIN_IMAGE_OBSERVATIONS,
) -> MergeResult:
    """Rewrite the segments the solution covers; keep the rest as the chain
    built them.

    Per tracker segment, with members M in keyframe order:
      * not every member inside the solution's horizon -> keep the LOCAL
        rows, no placement (the contract's `unplaced`): the segment is
        still being walked and the next solve will cover it;
      * no member posed with >= min_image_observations -> keep local rows,
        placement `refused` with the count;
      * otherwise the segment's frame is its first posed keyframe (identity
        there), every posed member gets a `solved` row in that frame, every
        other member an `unavailable` row with degeneracy `unregistered`,
        the segment's points are the solution's points first seen by one
        of its members, its support rows are dropped (they would index the
        wrong keypoints), and its placement is `registered` into the lowest
        segment index of the same component with scale exactly 1.
    A segment whose posed members fall in more than one component takes the
    component holding most of them; the others become `unregistered` rows.
    """
    horizon = solution.horizon
    members_by_segment: dict[int, list[tuple[int, Keyframe]]] = {}
    for position, keyframe in enumerate(keyframes):
        members_by_segment.setdefault(keyframe.segment_index, []).append((position, keyframe))
    local_poses: dict[int, list[dict]] = {}
    for row in pose_rows:
        local_poses.setdefault(row["segment_index"], []).append(row)
    local_points: dict[int, list[dict]] = {}
    for row in point_rows:
        local_points.setdefault(row["segment_index"], []).append(row)
    local_support: dict[int, list[list[int]]] = {}
    for row in support_rows or []:
        local_support.setdefault(int(row[0]), []).append(list(row))

    # Which segment owns each solution point: the segment of its first observer.
    owner_segment = np.array(
        [keyframes[i].segment_index if 0 <= i < len(keyframes) else -1 for i in solution.first_keyframe],
        dtype=np.int64,
    ) if len(solution.first_keyframe) else np.zeros(0, dtype=np.int64)
    kf_index_by_id = {k.keyframe_id: i for i, k in enumerate(keyframes)}

    # First pass: decide each segment's component and anchor.
    decided: dict[int, dict] = {}
    for segment, members in members_by_segment.items():
        ids = [k.keyframe_id for _, k in members]
        in_horizon = [kid in horizon for kid in ids]
        if not all(in_horizon):
            decided[segment] = {"mode": "pending", "in_horizon": sum(in_horizon)}
            continue
        posed = [
            (pos, k) for pos, k in members
            if k.keyframe_id in solution.poses
            and solution.poses[k.keyframe_id]["observations"] >= min_image_observations
        ]
        if not posed:
            decided[segment] = {"mode": "refused", "posed": 0}
            continue
        by_component: dict[int, int] = {}
        for _, k in posed:
            c = solution.poses[k.keyframe_id]["component"]
            by_component[c] = by_component.get(c, 0) + 1
        component = max(by_component, key=lambda c: (by_component[c], -c))
        posed = [(pos, k) for pos, k in posed if solution.poses[k.keyframe_id]["component"] == component]
        anchor_pos, anchor_kf = posed[0]
        r_wa, c_a = _pose_matrices(solution.poses[anchor_kf.keyframe_id])
        decided[segment] = {
            "mode": "replaced", "component": component, "posed": posed,
            "r_wa": r_wa, "c_a": c_a,
        }

    reference_by_component: dict[int, int] = {}
    for segment in sorted(decided):
        d = decided[segment]
        if d["mode"] == "replaced":
            reference_by_component.setdefault(d["component"], segment)

    new_pose_rows: list[dict] = []
    new_point_rows: list[dict] = []
    new_support_rows: list[list[int]] = []
    placements: list[SegmentPlacement] = []
    per_segment: dict[int, dict] = {}
    frame_revision = 1
    counts = {"segments_replaced": 0, "segments_refused": 0, "segments_pending": 0,
              "poses_solved": 0, "points": 0}

    for segment in sorted(members_by_segment):
        members = members_by_segment[segment]
        d = decided[segment]
        if d["mode"] != "replaced":
            new_pose_rows.extend(local_poses.get(segment, []))
            new_point_rows.extend(local_points.get(segment, []))
            new_support_rows.extend(local_support.get(segment, []))
            local_pts = len(local_points.get(segment, []))
            if d["mode"] == "pending":
                counts["segments_pending"] += 1
                per_segment[segment] = {
                    "replaced": False, "state": "pending",
                    "coverage": COVERAGE_PARTIAL if local_pts else COVERAGE_UNRESOLVED,
                    "keyframes": len(members), "in_horizon": d["in_horizon"],
                }
            else:
                counts["segments_refused"] += 1
                placements.append(SegmentPlacement(
                    segment_index=segment, state="refused",
                    rotation_wxyz=None, translation=None, scale=None,
                    reference_segment=None,
                    refusal_reason=(
                        f"the global solve posed none of this segment's "
                        f"{len(members)} keyframes with at least "
                        f"{min_image_observations} 3-D observations"
                    ),
                    input_digest=input_digest,
                    evidence={"keyframes": len(members), "keyframes_posed": 0, "solver": solution.solver},
                    frame_revision=frame_revision,
                ))
                per_segment[segment] = {
                    "replaced": False, "state": "refused",
                    "coverage": COVERAGE_PARTIAL if local_pts else COVERAGE_UNRESOLVED,
                    "keyframes": len(members), "keyframes_posed": 0,
                }
            continue

        counts["segments_replaced"] += 1
        component = d["component"]
        r_wa, c_a = d["r_wa"], d["c_a"]
        r_aw = r_wa.T
        posed_ids = {k.keyframe_id for _, k in d["posed"]}
        observations = []
        member_position = {k.keyframe_id: i for i, (_, k) in enumerate(members)}
        for i, (_, keyframe) in enumerate(members):
            entry = solution.poses.get(keyframe.keyframe_id)
            if keyframe.keyframe_id in posed_ids:
                r_wk, c_k = _pose_matrices(entry)
                r_ak = r_aw @ r_wk
                t_ak = r_aw @ (c_k - c_a)
                anchor = i == member_position[d["posed"][0][1].keyframe_id]
                new_pose_rows.append({
                    "keyframe_id": keyframe.keyframe_id,
                    "segment_index": segment,
                    "status": POSE_STATUS_ANCHOR if anchor else POSE_STATUS_SOLVED,
                    "degeneracy": DEGENERACY_NONE,
                    "rotation": [1.0, 0.0, 0.0, 0.0] if anchor else rotation_to_quaternion_wxyz(r_ak),
                    "translation": [0.0, 0.0, 0.0] if anchor else [float(v) for v in t_ak],
                    "observations": int(entry["observations"]),
                })
                observations.append(int(entry["observations"]))
                if not anchor:
                    counts["poses_solved"] += 1
            else:
                new_pose_rows.append({
                    "keyframe_id": keyframe.keyframe_id,
                    "segment_index": segment,
                    "status": POSE_STATUS_UNAVAILABLE,
                    "degeneracy": DEGENERACY_UNREGISTERED,
                    "rotation": None,
                    "translation": None,
                    "observations": int(entry["observations"]) if entry else 0,
                })
        # Points first seen by this segment, in this component, into the segment frame.
        mask = (owner_segment == segment) & (solution.component == component)
        point_indices = np.nonzero(mask)[0]
        if len(point_indices):
            world_xyz = solution.xyz[point_indices].astype(np.float64)
            local_xyz = (r_aw @ (world_xyz - c_a).T).T
            for j, p in enumerate(point_indices):
                new_point_rows.append({
                    "segment_index": segment,
                    "xyz": [float(v) for v in local_xyz[j]],
                    "rgb": list(WITHHELD_POINT_RGB),   # never the solver's colour
                })
            # No support rows for a solved segment. support.json's feature
            # index is defined over the ORB keypoints the chain re-detects
            # on the keyframe copies; the solver's observations are SIFT
            # keypoints on undistorted raw frames and live in the solution
            # (`reprojection_summary` is their self-check).
        n_points = int(len(point_indices))
        counts["points"] += n_points
        reference = reference_by_component[component]
        if reference == segment:
            rotation, translation = [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        else:
            r_wr, c_r = decided[reference]["r_wa"], decided[reference]["c_a"]
            rotation = rotation_to_quaternion_wxyz(r_wr.T @ r_wa)
            translation = [float(v) for v in (r_wr.T @ (c_a - c_r))]
        median_obs = float(np.median(observations)) if observations else None
        coverage = coverage_for(len(posed_ids), len(members), median_obs, n_points)
        placements.append(SegmentPlacement(
            segment_index=segment, state="registered",
            rotation_wxyz=tuple(rotation), translation=tuple(translation), scale=1.0,
            reference_segment=reference, refusal_reason=None,
            input_digest=input_digest,
            evidence={
                "keyframes": len(members), "keyframes_posed": len(posed_ids),
                "median_observations": median_obs, "points": n_points,
                "component": component, "solver": solution.solver, "coverage": coverage,
            },
            frame_revision=frame_revision,
        ))
        per_segment[segment] = {
            "replaced": True, "state": "registered", "coverage": coverage,
            "keyframes": len(members), "keyframes_posed": len(posed_ids),
            "median_observations": median_obs, "points": n_points,
            "component": component, "reference_segment": reference,
        }

    components_out = []
    for component, reference in sorted(reference_by_component.items()):
        segs = [s for s, d in decided.items() if d["mode"] == "replaced" and d["component"] == component]
        components_out.append({
            "component": component, "reference_segment": reference,
            "segments": sorted(segs),
            "keyframes": sum(per_segment[s]["keyframes_posed"] for s in segs),
            "points": sum(per_segment[s]["points"] for s in segs),
        })
    summary = {
        "solver": solution.solver,
        "solved_at": solution.solved_at,
        "horizon_keyframes": len(solution.keyframe_ids),
        "components": components_out,
        **counts,
    }
    # What ran (contract WORLD-BUILDER-COMPONENTS §2.5), into the published
    # manifest's `global_solve` beside the rest -- only when the solution
    # recorded it, so a world solved before these records builds as it did.
    if solution.transients is not None:
        summary["transients"] = solution.transients
    if solution.solve is not None:
        summary["solve"] = solution.solve
    # The evidence gate's record (§2.5 `gate.params` and digest, what fed it),
    # when the gate ran on this solution.
    if solution.gate is not None:
        summary["gate"] = solution.gate
    return MergeResult(
        pose_rows=new_pose_rows, point_rows=new_point_rows, support_rows=new_support_rows,
        placements=placements, segments=per_segment, summary=summary,
    )
