"""Transient masks on the global solver's OWN images (the final solve).

WHY THIS EXISTS

The wearer's hands, arms and the phone held in them are in most keyframes and
move WITH the camera, so SIFT matches on them are geometrically consistent
between views that share no scene at all. On the 2026-09-23 coherence run's
target walk (world 6e6d3fc3) the only glue between two rigid islands -- a
bathroom and bed block placed into the bedroom at a wrong tilt, scale and
position -- was 7 verified pairs whose inliers sat on the held phone. Masking
hands, arms and held phones before feature extraction removed 117 of the 136
phone inliers and left 0 cross-island pairs; the same masks removed the
split-off links on 52ed8e0a (93-97 % inside the masks) and on 991e5a15
(run `baseline/FORENSICS.md` H-A; candidate architecture module M).

WHAT IS MASKED

Every image the solve is handed: `<solve>/images/<name>.jpg`, the undistorted
raw capture frame where raw capture exists, else the session's redacted copy.
The recipe is `transients.py`'s own `union` mode (Grounding DINO + SAM 2.1 and
OneFormer, the held-phone rule, the 12 px ellipse dilation), run by its own
backends, with the two differences the experiment ran under (lane
`coherence_exp/masks.py`):

  * the input is the solver image, so there are no redaction fill boxes and
    the unobserved mask is empty -- the fill-drop rule never fires;
  * a mask is keyed by the solver image's name AND the SHA-1 of its bytes, so
    a re-run finds it whatever the keyframe numbering, and a frame that was
    re-undistorted is never matched to an old mask.

LAYOUT, under the solve workspace `<world>/solve/<session>/`:

    transients/<stem>.<sha1[:12]>.<component>.npz   the cache, one file per
                                        component (`transients.write_component`)
    transients/index.json               keyframe id -> solver image + SHA-1: what
                                        the surface stage looks its masks up by
    masks/<image name>.png              COLMAP masks: 0 = transient, 255 = use
    database.masked.p<pid>.<8hex>.db    the masked feature database ONE solve maps:
                                        a name of its own per solve, so two solves
                                        of one workspace never share a file (review
                                        V5, M1-5); `solve.database` names it
    database.masked.p<pid>.<8hex>.json  how it was made (`path`), and for the
                                        re-extracted path, per image, the image
                                        and mask digests it was extracted under
    (database.masked.db/.json: the one fixed name of solves before that; read
    as a re-extraction source, never written)
    reverify_pairs.txt                  the filtered path's re-verified pairs

TWO WAYS TO A MASKED DATABASE (`MASKING_*` below). `database.db` is the walk's
own: its background solves extracted UNMASKED features into it and verified
pairs at every horizon, loop detection included. The final solve extracts and
matches into it exactly as today, then maps from a FILTERED COPY with every
match touching a masked keypoint removed -- the run's arm A1h, which keeps the
target's room whole. Only when there is no walk database do the masks go to
extraction, into a fresh database matched once (arm A1), which on the target
splits the room: a single round of loop detection finds fewer revisits than
the walk's accumulated rounds. The walk's `database.db` is never modified by
either path, and never mapped by a masked solve.

ONE COMPUTATION, TWO CONSUMERS -- FOR A RAW-IMAGERY BUILD ONLY. The surface
and appearance stages compute the same union masks per keyframe today (about
300 s on a long walk). The solve computes them first, and
`transients.ensure_transient_masks` takes a keyframe's mask from this cache
(`solve_mask_donor`) when its own cache has none, saying how many it took
(`reused_from_solve`). But a solver image is the UNREDACTED capture frame, and
a redacted (product) build must never draw from one, not even as the shape of
a mask that becomes its published alpha. So the copy happens only for a build
under `TOWER_WORLD_RAW_IMAGERY`, which reads those frames anyway; a redacted
build computes its own masks exactly as before.

GPU. Exactly as the surface stage: `transients`' backends load one model at a
time, run it over every image that needs it, and free the card before the next
(at most one checkpoint resident, <= 2.6 GB). No scheduler of its own.

NEVER FATAL, NEVER SILENT. No CUDA device, missing packages or weights, a
model that fails to load, an out-of-memory error, anything a detector raises:
the solve runs UNMASKED on today's database, and the record it writes says so.
The record is the solution's `transients` (contract WORLD-BUILDER-COMPONENTS
§2.5): `state` is `applied` only when every piece of evidence the solve holds
was masked under the union rule, `partial` with counts when not (a fallback
rule, or an unmasked image that could not be excluded), and `unavailable` with
a `detail` otherwise -- including when the setting is off. Masks are a hard
dependency of the evidence gate (manager 011), so this record must never read
`applied` for a solve that was not masked. A missing or stale cache entry is
recomputed, never an error.

AN IMAGE THAT CANNOT BE MASKED IS EXCLUDED, not a fail-safe for the walk
(review V8, M2b). It gets an all-0 COLMAP mask, so re-extraction takes no
feature of it, and the walk-database filter drops every match touching it; it
is counted in `images_unmasked` and `images_excluded`. Before, one such image
made the record `partial` -- the gate then attached nothing anywhere -- while
the filtered database still carried its unmasked matches. The cost now is that
image, which is left unposed. The causes are an unreadable or mis-sized solver
image, or one another process held while it was hashed: the backends emit a
mask for every image they are given.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import numpy as np

from tower.storage import read_json_closed, write_bytes_atomic, write_json_atomic
from tower.world_builder import transients as T

logger = logging.getLogger(__name__)

SOLVER_MASK_SCHEMA = 1
# What the detector was shown. Distinct from the surface stage's input (the
# redacted keyframe with its fill), and part of every cache key.
INPUT_RULE = "solver-image|no-fill"

CACHE_DIRNAME = "transients"
MASKS_DIRNAME = "masks"
INDEX_FILENAME = "index.json"
MASKED_DATABASE_NAME = "database.masked.db"
MASKED_DATABASE_RECORD = "database.masked.json"

# What happened inside this module (`SolverMasks.state`, recorded as `outcome`).
STATE_OFF = T.STATE_OFF
STATE_OK = T.STATE_OK
STATE_UNAVAILABLE = T.STATE_UNAVAILABLE
STATE_FAILED = T.STATE_FAILED

# What the published solve says (`transients.state`, contract
# WORLD-BUILDER-COMPONENTS §2.5). Masks are a HARD dependency of the evidence
# gate (manager 011): anything but `applied` switches the gate to its
# fail-safe, so a solve that was not masked must never read `applied`.
RECORD_APPLIED = "applied"          # every solver image was masked
RECORD_PARTIAL = "partial"          # some images unmasked; the counts say how many
RECORD_UNAVAILABLE = "unavailable"  # none: `detail` says why (off, no GPU, load, OOM, ...)

OFF_DETAIL = ("transient masks on the final solve are off (TOWER_WORLD_SOLVE_MASKS "
              "unset or false): this solve is unmasked")

# HOW the masks reached the solve (`solve.masking`, `transients.masking`).
#
# walk-database-filtered: the walk's own `database.db`, extracted and matched
#     exactly as today, is COPIED; every match and every verified inlier that
#     touches a keypoint on a masked pixel is removed from the copy; the pairs
#     that lost any are re-verified; the solve maps from the copy. The
#     experiment driver's `database_from` + masks path -- arm A1h, the arm the
#     approved candidate numbers were measured on (the target's room whole at
#     232 keyframes in 5 of 5 seeds).
# re-extracted: no walk database to filter (a re-finish whose database is
#     gone). Masked keypoints are never extracted (`ImageReaderOptions.
#     mask_path`) into a fresh database matched once -- arm A1, which on the
#     target splits the room (154 + 81, 5 of 5 seeds).
#
# Either way, no match touching a masked pixel survives into the solve.
# `SolverMasks.cause`: why the masks are not applied, for a program.
CAUSE_GPU_OOM = "gpu-oom"
CAUSE_DETECTOR_FAILED = "detector-failed"
# Causes a later run can be expected to clear by itself.
RETRYABLE_CAUSES = (CAUSE_GPU_OOM,)

MASKING_FILTERED = "walk-database-filtered"
MASKING_REEXTRACTED = "re-extracted"
REVERIFY_PAIRS_FILENAME = "reverify_pairs.txt"
_PAIR_ID_BASE = 2147483647   # COLMAP: pair_id = image_id1 * kMaxNumImages + image_id2, id1 < id2


def solver_params() -> T.TransientParams:
    """The rule the solver's masks are made under: `transients`' union mode,
    the recipe the run measured (HANDS.md; FORENSICS H-A)."""
    return T.TransientParams()


def cache_dir(workspace) -> Path:
    return Path(workspace.root) / CACHE_DIRNAME


def masks_dir(workspace) -> Path:
    return Path(workspace.root) / MASKS_DIRNAME


def masked_database_path(workspace) -> Path:
    """The fixed name solves used before names were per solve. Read, never written."""
    return Path(workspace.root) / MASKED_DATABASE_NAME


MASKED_DATABASE_GLOB = "database.masked.p*.db"


def new_masked_database_path(workspace) -> Path:
    """A masked database name no other solve can be using: the pid (so a stray one
    is attributable, and sweepable once its writer is gone) and a nonce."""
    return Path(workspace.root) / f"database.masked.p{os.getpid()}.{uuid.uuid4().hex[:8]}.db"


def database_record_path(db) -> Path:
    """`<db>.json` beside `<db>.db`: what that one database was made from."""
    return Path(db).with_suffix(".json")


def _writer_pid(db) -> int | None:
    parts = Path(db).name.split(".")
    if len(parts) >= 5 and parts[2].startswith("p") and parts[2][1:].isdigit():
        return int(parts[2][1:])
    return None


def _pid_alive(pid: int) -> bool:
    try:
        import psutil  # noqa: PLC0415

        return psutil.pid_exists(int(pid))
    except Exception:  # noqa: BLE001 -- unknown is alive: never sweep a live writer's file
        return True


def _published_database(workspace) -> str | None:
    try:
        meta = read_json_closed(Path(workspace.root) / "solution.json")
        return ((meta or {}).get("solve") or {}).get("database")
    except (OSError, ValueError, AttributeError):
        return None


def sweep_masked_databases(workspace, keep=()) -> list:
    """Remove per-solve masked databases whose writing solve is gone, except the one
    the published solution names and `keep`. The walk's `database.db` and the
    pre-per-solve `database.masked.db` are never touched. Returns the names removed.

    The product's own superseded intermediates, as the fixed name was replaced
    in place before: a masked database is rebuilt from the walk's database and
    the mask cache whenever it is needed."""
    root = Path(workspace.root)
    keep = {Path(k).name for k in keep} | {_published_database(workspace) or ""}
    removed = []
    for db in sorted(root.glob(MASKED_DATABASE_GLOB)):
        pid = _writer_pid(db)
        if db.name in keep or pid is None or pid == os.getpid() or _pid_alive(pid):
            continue
        for side in (db, Path(str(db) + "-wal"), Path(str(db) + "-shm"), database_record_path(db),
                     db.with_suffix(".reverify.txt")):
            try:
                side.unlink(missing_ok=True)
            except OSError:
                pass
        removed.append(db.name)
    return removed


def component_path(cdir: Path, name: str, component: str, sha1: str) -> Path:
    """One file per (solver image name, image bytes, component)."""
    return Path(cdir) / f"{Path(name).stem}.{sha1[:12]}.{component}.npz"


def mask_key(component: str, params: T.TransientParams, name: str, sha1: str) -> dict:
    """Everything that changes a cached component: the image (name and bytes),
    what the detector was shown, the pinned models and the component's own
    parameters. The same key read with a surface build's params finds the mask
    exactly when the surface would have run the same component."""
    return {
        "schema": SOLVER_MASK_SCHEMA,
        "component": component,
        "image": name,
        "image_sha1": sha1,
        "input_rule": INPUT_RULE,
        "models": {mid: rev for mid, rev in T.COMPONENT_MODELS[component]},
        "params": params.component_params(component),
    }


def file_sha1(path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_rgb(path) -> np.ndarray | None:
    import cv2  # noqa: PLC0415

    try:
        bgr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        return None
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def cuda_unavailable_reason() -> str | None:
    """None when a CUDA device is usable, else why not.

    The solver's masks run on the GPU or not at all: on the CPU the union
    recipe costs several seconds a frame, which would hold a final solve for
    most of an hour. A Tower without the GPU solves unmasked and says so."""
    try:
        import torch  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001 -- ImportError or a broken wheel
        return f"torch is not importable ({type(exc).__name__}: {exc})"
    try:
        if torch.cuda.is_available():
            return None
    except Exception as exc:  # noqa: BLE001
        return f"CUDA probe failed ({type(exc).__name__}: {exc})"
    return ("no CUDA device, and no CPU fallback: the solver's transient masks "
            "run on the GPU only")


@dataclass
class SolverMasks:
    """What happened to the solver's masks, and what `record()` publishes."""

    state: str
    params: T.TransientParams
    requested_rule: str
    detail: str | None = None
    partial: str | None = None
    images: int = 0
    # image name -> {"image_sha1", "mask_sha1", "masked_frac"}; exactly the
    # images a PNG mask was written for.
    masked: dict = field(default_factory=dict)
    unmasked: list = field(default_factory=list)
    # image name -> {"image_sha1", "mask_sha1"}: the unmasked images EXCLUDED from
    # the solve (review V8, M2b) -- an all-0 COLMAP mask was written for each, so
    # re-extraction takes no feature of it, and the walk-database filter drops
    # every match touching it (`filter_walk_database`). Its evidence never reaches
    # the solve, so it does not make the solve `partial`.
    excluded: dict = field(default_factory=dict)
    cache_hits: int = 0
    computed: int = 0
    device: str | None = None
    seconds: dict = field(default_factory=dict)
    gpu_peak_mb: float | None = None
    # WHY the masks are not applied, as a code a program can act on (`CAUSE_*`),
    # beside the sentence in `detail`. `gpu-oom` is TRANSIENT: the finisher
    # treats a gated session whose final solve lost its masks to it as owed a
    # re-solve (review V5, M1-2); every other cause is the machine's and stays.
    cause: str | None = None
    retries: int = 0

    @property
    def available(self) -> bool:
        """Whether extraction is given these masks."""
        return self.state == STATE_OK and bool(self.masked)

    @property
    def record_state(self) -> str:
        """`applied` / `partial` / `unavailable`, as the contract spells it.

        `partial` also when every image is masked but under a FALLBACK rule
        (`union` asked for, OneFormer alone could run): the evidence was
        measured with the union rule, and the gate treats anything but
        `applied` as masks unavailable (lead, 2026-09-23).

        An image that could not be masked but was EXCLUDED (`excluded`) does not
        make it `partial` (review V8, M2b): none of its features or matches
        reach the solve, so every piece of evidence the solve holds was masked
        under the union rule -- which is what `applied` promises the gate. One
        unreadable frame used to put the whole walk into the masks fail-safe.
        Only an unmasked image that could NOT be excluded (no mask shape to
        write its exclusion with) is `partial` now."""
        if not self.available:
            return RECORD_UNAVAILABLE
        if self.partial:
            return RECORD_PARTIAL
        if (set(self.unmasked) - set(self.excluded)
                or len(self.masked) + len(self.unmasked) < self.images):
            return RECORD_PARTIAL
        return RECORD_APPLIED

    def record(self) -> dict:
        ok = self.available
        fracs = np.asarray([v["masked_frac"] for v in self.masked.values()], np.float64)
        return {
            "schema": SOLVER_MASK_SCHEMA,
            "requested": True,
            "state": self.record_state,
            "outcome": self.state,
            "extraction_masked": self.available,
            "detail": self.detail,
            "mode": self.params.mode,
            "rule": self.params.rule_id() if ok else None,
            "requested_rule": self.requested_rule,
            # Set when `union` was asked for and only OneFormer could run: the
            # images ARE masked, under `rule`, which is then not `requested_rule`.
            "rule_fallback": self.partial,
            "input_rule": INPUT_RULE,
            "models": self.params.models() if ok else {},
            "images": self.images,
            "images_masked": len(self.masked),
            "images_unmasked": len(self.unmasked),
            "unmasked_examples": list(self.unmasked[:10]),
            # The unmasked images kept out of the solve (review V8, M2b).
            "images_excluded": len(self.excluded),
            "excluded_examples": list(self.excluded)[:10],
            "cache_hits": self.cache_hits,
            "computed": self.computed,
            "device": self.device,
            "masked_fraction": (None if not len(fracs) else {
                "mean": round(float(fracs.mean()), 5),
                "median": round(float(np.median(fracs)), 5),
                "p90": round(float(np.percentile(fracs, 90)), 5),
                "max": round(float(fracs.max()), 5),
                "images_with_pixels": int((fracs > 0).sum()),
            }),
            "seconds": dict(self.seconds),
            "gpu_peak_mb": self.gpu_peak_mb,
            "mask_dir": MASKS_DIRNAME if self.available else None,
            "cause": self.cause,
            "retryable": self.cause in RETRYABLE_CAUSES,
            "retries": self.retries,
        }


def off_record() -> dict:
    """The record of a solve that was not asked for masks: today's solve.

    `unavailable`, not a fourth state: to the gate an unmasked solve is an
    unmasked solve, whatever the reason, and `requested` says it was chosen."""
    return {"schema": SOLVER_MASK_SCHEMA, "requested": False, "state": RECORD_UNAVAILABLE,
            "outcome": STATE_OFF, "extraction_masked": False, "detail": OFF_DETAIL,
            "rule": None}


def failed_record(detail: str, params: T.TransientParams | None = None) -> dict:
    """The record when the mask step itself raised: the solve ran unmasked."""
    params = params or solver_params()
    return SolverMasks(state=STATE_FAILED, params=params, requested_rule=params.rule_id(),
                       detail=detail).record()


def ensure_solver_masks(workspace, names, *, keyframe_ids: dict | None = None, shape=None,
                        params: T.TransientParams | None = None,
                        backend_factory: Callable | None = None,
                        device_probe: Callable[[], str | None] | None = None) -> SolverMasks:
    """Make sure every solver image in `names` has its transient mask, computing
    only what the cache lacks, and write the COLMAP masks for extraction.

    `shape` is the solver camera's (height, width); `keyframe_ids` maps an image
    name to its keyframe id for the surface stage's index. Never raises for a
    machine that cannot run the detector: the result says `unavailable` or
    `failed` and nothing is written to `masks/`.
    """
    t0 = time.time()
    params = params or solver_params()
    out = SolverMasks(state=STATE_OK, params=params, requested_rule=params.rule_id(),
                      images=len(names))
    if params.mode == T.MODE_OFF:
        out.state, out.detail = STATE_OFF, "the solver mask rule is `off`"
        return out
    cdir = cache_dir(workspace)
    images_dir = Path(workspace.images_dir)
    shape = tuple(int(v) for v in shape) if shape is not None else None

    # -- the cheap pass: hash every image, find which components are cached --
    images = []
    unhashed = []
    for name in names:
        try:
            images.append((name, images_dir / name, file_sha1(images_dir / name)))
        except OSError:
            out.unmasked.append(name)
            unhashed.append(name)
    missing: dict = {c: [] for c in params.components}
    for name, path, sha1 in images:
        any_missing = False
        for c in params.components:
            if T.read_component(component_path(cdir, name, c, sha1),
                                mask_key(c, params, name, sha1), shape) is None:
                missing[c].append((name, path, sha1))
                any_missing = True
        if not any_missing:
            out.cache_hits += 1
    out.seconds["lookup"] = round(time.time() - t0, 3)

    # -- inference, for what the cache lacks ---------------------------------
    if any(missing.values()):
        reason = (device_probe or cuda_unavailable_reason)()
        if reason:
            T._log_unavailable_once(reason)
            out.state, out.detail = STATE_UNAVAILABLE, reason
            out.seconds["total"] = round(time.time() - t0, 3)
            return out
        out.device = "cuda"
        factory = backend_factory or T.BACKEND_FACTORY
        backends = {c: factory(c) for c in params.components if missing[c]}
        refused = {}
        for c, backend in backends.items():
            why = backend.probe()
            if why:
                T._log_unavailable_once(why)
                refused[c] = why
        if refused:
            if not T._can_fall_back(params, refused):
                out.state, out.detail = STATE_UNAVAILABLE, next(iter(refused.values()))
                out.seconds["total"] = round(time.time() - t0, 3)
                return out
            params = _fall_back(out, params, refused)
        peak = T._reset_peak()
        # What EACH component produced, by image name (review V8, M2c). One set shared
        # by both components made the OOM retry of the second one run over nothing:
        # the first had already emitted every image, so "not yet emitted" was empty,
        # and a transient out-of-memory error became a permanent `unavailable`.
        computed: dict = {c: set() for c in params.components}
        for c in params.components:
            todo = missing.get(c) or []
            if not todo:
                continue
            backend = backends.get(c) or factory(c)
            t1 = time.time()
            items, by_index = [], {}
            for i, (name, path, sha1) in enumerate(todo):
                rgb = _read_rgb(path)
                if rgb is None or (shape is not None and rgb.shape[:2] != shape):
                    continue
                items.append((i, rgb, np.zeros(rgb.shape[:2], bool)))
                by_index[i] = (name, sha1)
            out.seconds[f"{c}.read"] = round(time.time() - t1, 3)
            emitted = computed.setdefault(c, set())

            def emit(i, hand, phone, seconds, c=c, by_index=by_index, emitted=emitted):
                name, sha1 = by_index[i]
                T.write_component(component_path(cdir, name, c, sha1),
                                  mask_key(c, params, name, sha1),
                                  np.asarray(hand, bool), np.asarray(phone, bool),
                                  image_sha1=sha1, seconds=seconds)
                emitted.add(name)

            t2 = time.time()
            try:
                try:
                    timings = backend.run(items, params, emit)
                except Exception as first:  # noqa: BLE001 -- only an OOM is retried
                    if not is_gpu_oom(first):
                        raise
                    # ONE retry, on a cleared cache, of what THIS component has not yet
                    # emitted: an OOM is usually another process's allocation or
                    # fragmentation, not this batch being too large (review V5, M1-2).
                    out.retries += 1
                    logger.warning("[Tower][WorldBuilder][solve-masks] the %s detector ran out "
                                   "of GPU memory; clearing the cache and retrying once", c)
                    _empty_cuda_cache()
                    items = [it for it in items if by_index[it[0]][0] not in emitted]
                    timings = backend.run(items, params, emit)
            except T.TransientDetectorUnavailable as exc:
                T._log_unavailable_once(str(exc))
                if T._can_fall_back(params, {c: str(exc)}):
                    params = _fall_back(out, params, {c: str(exc)})
                    continue
                out.state, out.detail = STATE_UNAVAILABLE, str(exc)
                out.cause = CAUSE_GPU_OOM if is_gpu_oom(exc) else None
                out.seconds["total"] = round(time.time() - t0, 3)
                return out
            except Exception as exc:  # noqa: BLE001 -- OOM, device loss: a solve, unmasked
                logger.exception("[Tower][WorldBuilder][solve-masks] the %s detector failed; "
                                 "this solve runs unmasked", c)
                out.state = STATE_FAILED
                out.cause = CAUSE_GPU_OOM if is_gpu_oom(exc) else CAUSE_DETECTOR_FAILED
                out.detail = f"the {c} detector failed ({type(exc).__name__}: {exc})"
                out.seconds["total"] = round(time.time() - t0, 3)
                return out
            out.seconds[c] = round(time.time() - t2, 3)
            for k, v in (timings or {}).items():
                if k != "stopped":
                    out.seconds[f"{c}.{k}"] = v
        # Images at least one component was computed for, as before.
        out.computed = len(set().union(*computed.values())) if computed else 0
        out.gpu_peak_mb = T._peak_mb(peak)

    # -- compose, and write what extraction reads ------------------------------
    t3 = time.time()
    mdir = masks_dir(workspace)
    mdir.mkdir(parents=True, exist_ok=True)
    index = {}
    for name, path, sha1 in images:
        parts = []
        for c in params.components:
            got = T.read_component(component_path(cdir, name, c, sha1),
                                   mask_key(c, params, name, sha1), shape)
            if got is None:
                break
            parts.append(got[:2])
        png_path = mdir / f"{name}.png"
        if len(parts) != len(params.components):
            # No mask for this image (unreadable, or its detector output never
            # arrived). It is counted as unmasked and EXCLUDED (review V8, M2b):
            # an all-0 mask ("extract nothing") rather than the all-255 ("extract
            # everything") it used to get, so none of its features reaches a
            # re-extracted solve; the walk-database filter drops its matches.
            out.unmasked.append(name)
            _exclude(out, png_path, name, sha1, shape)
            continue
        H, W = parts[0][0].shape
        m = T.compose(parts, params, (H, W))
        png = np.where(m, 0, 255).astype(np.uint8)
        mask_sha1 = _write_png(png_path, png)
        out.masked[name] = {"image_sha1": sha1, "mask_sha1": mask_sha1,
                            "masked_frac": float(m.mean())}
        kid = (keyframe_ids or {}).get(name)
        if kid is not None:
            index[kid] = {"image": name, "image_sha1": sha1}
    for name in unhashed:
        # An image that could not even be read to hash it (a file another process
        # holds, a vanished file) is excluded the same way.
        _exclude(out, mdir / f"{name}.png", name, None, shape)
    if index:
        _merge_index(cdir, index, params)
    out.seconds["write"] = round(time.time() - t3, 3)
    out.seconds["total"] = round(time.time() - t0, 3)
    if not out.masked:
        out.state = STATE_UNAVAILABLE
        out.detail = out.detail or "no solver image could be masked"
    logger.info("[Tower][WorldBuilder][solve-masks] %d of %d solver images masked "
                "(%d cached, %d computed) under %s in %.1f s",
                len(out.masked), out.images, out.cache_hits, out.computed,
                params.mode, out.seconds["total"])
    return out


def is_gpu_oom(exc: BaseException) -> bool:
    """A CUDA out-of-memory error, from torch or from a library under it."""
    if type(exc).__name__ in ("OutOfMemoryError", "OutOfMemory"):
        return True
    return _text_is_gpu_oom(str(exc))


def _text_is_gpu_oom(text: str) -> bool:
    text = str(text).lower()
    return "out of memory" in text and ("cuda" in text or "gpu" in text or "cublas" in text)


def _empty_cuda_cache() -> None:
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 -- the retry happens either way
        logger.debug("[Tower][WorldBuilder][solve-masks] could not empty the CUDA cache",
                     exc_info=True)


def _fall_back(out: SolverMasks, params: T.TransientParams, missing: dict) -> T.TransientParams:
    """`union` without Grounding DINO / SAM continues as `oneformer`, and says
    so -- the surface stage's own rule (`transients._fall_back`)."""
    why = "; ".join(f"{c}: {r}" for c, r in sorted(missing.items()))
    if any(_text_is_gpu_oom(str(r)) for r in missing.values()):
        # A fallback forced by the GPU running out of memory is transient: recorded so,
        # like an OOM that left the solve unmasked (review V7, L-b).
        out.cause = CAUSE_GPU_OOM
    effective = replace(params, mode=T.MODE_ONEFORMER)
    out.params = effective
    out.partial = (f"{params.mode} was requested but only {T.MODE_ONEFORMER} could run ({why}); "
                   "these masks are OneFormer's alone")
    logger.warning("[Tower][WorldBuilder][solve-masks] %s", out.partial)
    return effective


def _exclude(out: SolverMasks, png_path: Path, name: str, sha1: str | None, shape) -> None:
    """Keep an image with no mask out of the solve: an all-0 COLMAP mask (no feature
    is extracted from it) and an entry in `out.excluded`. Without the solver camera's
    shape no mask can be written, so the image is NOT excluded and the record says
    `partial`. A mask that cannot be written is the same: not excluded."""
    if shape is None:
        return
    try:
        mask_sha1 = _write_png(png_path, np.zeros(shape, np.uint8))
    except OSError:
        logger.warning("[Tower][WorldBuilder][solve-masks] could not write the exclusion mask "
                       "of %s; the solve is partial", name, exc_info=True)
        return
    out.excluded[name] = {"image_sha1": sha1, "mask_sha1": mask_sha1}


def _write_png(path: Path, png: np.ndarray) -> str:
    import cv2  # noqa: PLC0415

    ok, buf = cv2.imencode(".png", png)
    if not ok:
        raise OSError(f"cannot encode mask {path}")
    data = buf.tobytes()
    write_bytes_atomic(path, lambda handle: handle.write(data))
    return hashlib.sha1(data).hexdigest()


def _merge_index(cdir: Path, index: dict, params: T.TransientParams) -> None:
    path = Path(cdir) / INDEX_FILENAME
    images = {}
    try:
        previous = read_json_closed(path) if path.is_file() else {}
        images = dict(previous.get("images") or {})
    except (OSError, ValueError, AttributeError):
        images = {}
    images.update(index)
    write_json_atomic(path, {"schema": SOLVER_MASK_SCHEMA, "input_rule": INPUT_RULE,
                             "rule": params.rule_id(), "images": images})


def masked_database(workspace, masks: SolverMasks, *, all_names=()) -> tuple[Path, dict]:
    """This solve's own masked feature database (a new per-solve name), seeded
    with a copy of the newest earlier RE-EXTRACTED one when every image it
    holds was extracted under the same image and mask. Returns (path, info).

    Incremental as before: an image new since that database is simply
    extracted; one whose bytes or mask changed makes it unusable, because
    COLMAP has no way to re-extract one image in place. The earlier database
    is read, never modified, so a solve running beside this one is not
    disturbed (review V5, M1-5).
    """
    import sqlite3  # noqa: PLC0415

    db = new_masked_database_path(workspace)
    current = {name: [v["image_sha1"], v["mask_sha1"]] for name, v in masks.masked.items()}
    # An EXCLUDED image is extracted under its all-0 mask, which is recorded like any
    # other: a database that extracted it whole (`[None, None]`, before exclusion
    # existed) or under a real mask is then "extracted under a different mask" and
    # not reused -- COLMAP never re-extracts an image a database already holds.
    for name, v in masks.excluded.items():
        current[name] = [v["image_sha1"], v["mask_sha1"]]
    for name in all_names:
        current.setdefault(name, [None, None])
    rule = masks.params.rule_id()
    source, stale_reason, held = None, None, {}
    records = [q for q in Path(workspace.root).glob("database.masked*.json")]
    records.sort(key=lambda q: q.stat().st_mtime, reverse=True)
    for record_path in records:
        candidate = record_path.with_suffix(".db")
        try:
            previous = read_json_closed(record_path)
        except (OSError, ValueError):
            continue
        if not candidate.is_file() or not isinstance(previous, dict):
            continue
        if previous.get("path", MASKING_REEXTRACTED) != MASKING_REEXTRACTED:
            stale_reason = stale_reason or (f"{candidate.name} was made by {previous.get('path')!r}, "
                                            "not by re-extraction")
            continue
        if previous.get("rule") != rule:
            stale_reason = stale_reason or f"mask rule changed from {previous.get('rule')!r}"
            continue
        changed = next((name for name, entry in (previous.get("images") or {}).items()
                        if name in current and list(entry) != current[name]), None)
        if changed is not None:
            stale_reason = stale_reason or f"{changed} was extracted under a different image or mask"
            continue
        source, held = candidate, dict(previous.get("images") or {})
        break
    if source is not None:
        src = _connect_read_only(source)
        out = sqlite3.connect(str(db))
        try:
            src.backup(out)
        finally:
            src.close()
            out.close()
    elif stale_reason:
        logger.info("[Tower][WorldBuilder][solve-masks] a fresh masked database: %s", stale_reason)
    held.update(current)
    # Written BEFORE extraction: an interrupted extraction leaves images in the
    # database that the record already names, never the reverse.
    write_json_atomic(database_record_path(db), {"schema": SOLVER_MASK_SCHEMA,
                                                 "path": MASKING_REEXTRACTED,
                                                 "rule": rule, "images": held})
    swept = sweep_masked_databases(workspace, keep=[db] + ([source] if source else []))
    return db, {"database": db.name, "reused": source is not None,
                "reused_from": source.name if source is not None else None,
                "rebuilt_because": None if source is not None else stale_reason,
                "swept": swept}


def _connect_read_only(path):
    """SQLite `mode=ro`: reads the file and its write-ahead log, never writes
    either, never checkpoints. (`immutable=1` would ignore the log, which holds
    whatever COLMAP committed last.)"""
    import sqlite3  # noqa: PLC0415

    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


def walk_database_usable(path) -> bool:
    """Whether `path` is a COLMAP database holding at least one image: the
    walk's own database, which the filtered path copies. Never raises."""
    import sqlite3  # noqa: PLC0415

    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        # READ-ONLY (review V5, M1-4): this is the walk's database, which the solve
        # never writes; a read-write connection could checkpoint its WAL.
        con = _connect_read_only(path)
        try:
            images = con.execute("select count(*) from images").fetchone()[0]
            con.execute("select count(*) from keypoints").fetchone()
            con.execute("select count(*) from matches").fetchone()
            con.execute("select count(*) from two_view_geometries").fetchone()
        finally:
            con.close()
    except sqlite3.Error:
        return False
    return images > 0


def _keypoints_in_mask(con, names: dict, masks: SolverMasks, mdir: Path, info: dict) -> dict:
    """image_id -> bool per keypoint (True = on a masked pixel, so its matches go).

    An image WITHOUT a mask -- one the detector could not mask, or one in the walk
    database that this solve's mask step never saw -- is EXCLUDED: every keypoint
    counts as masked, so every match and verified inlier touching it is removed
    (review V8, M2b). It used to be skipped, so its unmasked matches reached the
    solve whole while the record said `partial` and the gate attached nothing
    anywhere; now its evidence is dropped and the rest of the walk is gated."""
    import cv2  # noqa: PLC0415

    in_mask = {}
    for iid, rows, cols, data in con.execute("select image_id, rows, cols, data from keypoints"):
        name = names.get(iid)
        if name not in masks.masked:
            in_mask[iid] = np.ones(int(rows or 0), bool)
            info["images_excluded"] += 1
            info["keypoints_excluded"] += int(rows or 0)
            continue
        png = cv2.imread(str(mdir / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        if png is None:
            raise OSError(f"the COLMAP mask of {name} is unreadable")
        if not rows or data is None:
            in_mask[iid] = np.zeros(0, bool)
            info["images_filtered"] += 1
            continue
        xy = np.frombuffer(data, np.float32).reshape(int(rows), int(cols))[:, :2]
        # COLMAP's keypoint (x, y) has the top-left pixel's centre at (0.5, 0.5),
        # so the pixel a keypoint sits on is floor(x), floor(y): the rule the
        # run's driver used.
        x = np.clip(np.floor(xy[:, 0]).astype(np.int64), 0, png.shape[1] - 1)
        y = np.clip(np.floor(xy[:, 1]).astype(np.int64), 0, png.shape[0] - 1)
        hit = png[y, x] == 0
        in_mask[iid] = hit
        info["images_filtered"] += 1
        info["keypoints_total"] += int(rows)
        info["keypoints_in_mask"] += int(hit.sum())
    return in_mask


def _touching(pairs: np.ndarray, first, second) -> np.ndarray:
    """Which rows of an (n, 2) index array touch a masked keypoint. An index
    past the end of an image's keypoints is treated as touching: a match that
    cannot be checked cannot be kept."""
    bad = np.zeros(len(pairs), bool)
    for col, hit in ((0, first), (1, second)):
        if hit is None:
            continue
        idx = pairs[:, col].astype(np.int64)
        outside = idx >= len(hit)
        bad |= outside
        bad[~outside] |= hit[idx[~outside]]
    return bad


def filter_walk_database(workspace, masks: SolverMasks) -> tuple[Path, Path | None, dict]:
    """The walk database, copied, with every match and every verified inlier
    that touches a masked keypoint removed. Returns (the masked database, the
    COLMAP pair list to re-verify or None, what was removed).

    The walk's `database.db` is never modified: it is read through SQLite's
    backup API, which also carries whatever COLMAP left in its write-ahead log.
    A pair that lost a raw match loses its verified geometry and is listed for
    re-verification from what is left; a verified geometry whose inliers touch
    a mask while its raw matches did not (it cannot happen through COLMAP, but
    a database is not a promise) is dropped and re-verified the same way.
    """
    import sqlite3  # noqa: PLC0415

    t0 = time.time()
    walk = Path(workspace.database_path)
    db = new_masked_database_path(workspace)
    record_path = database_record_path(db)
    # `images_unfiltered` stays 0 and is kept for readers of older records: an image
    # without a mask is now `images_excluded` (review V8, M2b).
    info = {"source": walk.name, "images_filtered": 0, "images_unfiltered": 0,
            "images_excluded": 0, "keypoints_excluded": 0,
            "keypoints_total": 0, "keypoints_in_mask": 0, "matches_dropped": 0,
            "pairs_changed": 0, "pairs_emptied": 0, "geometries_dropped": 0}
    src = _connect_read_only(walk)
    out = sqlite3.connect(str(db))
    try:
        src.backup(out)
    finally:
        src.close()
    changed: dict = {}
    try:
        names = dict(out.execute("select image_id, name from images"))
        in_mask = _keypoints_in_mask(out, names, masks, masks_dir(workspace), info)
        with_matches = set()
        for pair_id, rows, cols, data in out.execute(
                "select pair_id, rows, cols, data from matches").fetchall():
            with_matches.add(pair_id)
            if not rows or data is None:
                continue
            second = int(pair_id) % _PAIR_ID_BASE
            first = (int(pair_id) - second) // _PAIR_ID_BASE
            a, b = in_mask.get(first), in_mask.get(second)
            if a is None and b is None:
                continue
            matches = np.frombuffer(data, np.uint32).reshape(int(rows), int(cols))
            bad = _touching(matches, a, b)
            if not bad.any():
                continue
            keep = np.ascontiguousarray(matches[~bad])
            info["matches_dropped"] += int(bad.sum())
            info["pairs_emptied"] += int(len(keep) == 0)
            out.execute("update matches set rows=?, data=? where pair_id=?",
                        (int(len(keep)), keep.tobytes(), pair_id))
            out.execute("delete from two_view_geometries where pair_id=?", (pair_id,))
            changed[pair_id] = (names.get(first), names.get(second))
        for pair_id, rows, cols, data in out.execute(
                "select pair_id, rows, cols, data from two_view_geometries").fetchall():
            if not rows or data is None:
                continue
            second = int(pair_id) % _PAIR_ID_BASE
            first = (int(pair_id) - second) // _PAIR_ID_BASE
            a, b = in_mask.get(first), in_mask.get(second)
            if a is None and b is None:
                continue
            inliers = np.frombuffer(data, np.uint32).reshape(int(rows), int(cols))
            if _touching(inliers, a, b).any():
                out.execute("delete from two_view_geometries where pair_id=?", (pair_id,))
                info["geometries_dropped"] += 1
                if pair_id in with_matches:
                    changed[pair_id] = (names.get(first), names.get(second))
        out.commit()
    finally:
        out.close()
    info["pairs_changed"] = len(changed)
    pairs_path = None
    listed = [(a, b) for a, b in changed.values() if a and b]
    info["pairs_listed"] = len(listed)
    if listed:
        pairs_path = db.with_suffix(".reverify.txt")
        data = "".join(f"{a} {b}\n" for a, b in listed).encode("utf-8")
        write_bytes_atomic(pairs_path, lambda handle: handle.write(data))
    write_json_atomic(record_path, {"schema": SOLVER_MASK_SCHEMA, "path": MASKING_FILTERED,
                                    "rule": masks.params.rule_id(), "source": walk.name})
    info["swept"] = sweep_masked_databases(workspace, keep=[db])
    info["seconds"] = round(time.time() - t0, 3)
    logger.info("[Tower][WorldBuilder][solve-masks] walk database filtered: %d of %d keypoints "
                "on masks, %d matches in %d pairs removed", info["keypoints_in_mask"],
                info["keypoints_total"], info["matches_dropped"], info["pairs_changed"])
    return db, pairs_path, info


def solve_mask_donor(store, world_id: str, session_id: str):
    """A lookup `(keyframe_id, component, params, shape) -> (hand, phone,
    provenance) | None` over the solve's mask cache, or None when the solve has
    masked nothing (every world solved without the setting, i.e. every world
    today). Never raises.

    Each solver image is re-hashed before its mask is lent, so a frame that
    was re-undistorted since is never matched to an old mask.
    """
    try:
        from tower.world_builder.global_solve import workspace_for  # noqa: PLC0415

        workspace = workspace_for(store, world_id, session_id)
        path = cache_dir(workspace) / INDEX_FILENAME
        if not path.is_file():
            return None
        entries = dict(read_json_closed(path).get("images") or {})
    except Exception:  # noqa: BLE001 -- no donor is today's behaviour
        return None
    if not entries:
        return None
    cdir = cache_dir(workspace)
    verified: dict = {}

    def lookup(keyframe_id, component, params, shape):
        entry = entries.get(keyframe_id)
        if not isinstance(entry, dict):
            return None
        name, sha1 = entry.get("image"), entry.get("image_sha1")
        if not name or not sha1:
            return None
        if name not in verified:
            try:
                verified[name] = file_sha1(Path(workspace.images_dir) / name) == sha1
            except OSError:
                verified[name] = False
        if not verified[name]:
            return None
        key = mask_key(component, params, name, sha1)
        got = T.read_component(component_path(cdir, name, component, sha1), key, shape)
        if got is None:
            return None
        return got[0], got[1], {"origin": "solve", "solve_image": name,
                                "solve_image_sha1": sha1,
                                "solve_key_digest": T.key_digest(key)}

    return lookup
