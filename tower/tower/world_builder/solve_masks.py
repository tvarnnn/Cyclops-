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
    masks/<image name>.png              COLMAP extraction masks: 0 = transient,
                                        no keypoint is extracted there; 255 = use
    database.masked.db                  the masked feature database
    database.masked.json                per image, the image and mask digests its
                                        features were extracted under

WHY A SECOND DATABASE. `database.db` is shared with the background solves of
the walk, which extract UNMASKED features, and `extract_features` skips every
image already in a database. A masked final solve on that database would
silently reuse unmasked keypoints for every keyframe the walk had solved. The
masked database is its own file, and it is rebuilt whenever an image it
already holds was extracted under a different image or mask.

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
§2.5): `state` is `applied` only when every solver image was masked, `partial`
with counts when some were not, and `unavailable` with a `detail` otherwise --
including when the setting is off. Masks are a hard dependency of the evidence
gate (manager 011), so this record must never read `applied` for a solve that
was not masked. A missing or stale cache entry is recomputed, never an error.
"""

from __future__ import annotations

import hashlib
import logging
import time
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


def solver_params() -> T.TransientParams:
    """The rule the solver's masks are made under: `transients`' union mode,
    the recipe the run measured (HANDS.md; FORENSICS H-A)."""
    return T.TransientParams()


def cache_dir(workspace) -> Path:
    return Path(workspace.root) / CACHE_DIRNAME


def masks_dir(workspace) -> Path:
    return Path(workspace.root) / MASKS_DIRNAME


def masked_database_path(workspace) -> Path:
    return Path(workspace.root) / MASKED_DATABASE_NAME


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
    cache_hits: int = 0
    computed: int = 0
    device: str | None = None
    seconds: dict = field(default_factory=dict)
    gpu_peak_mb: float | None = None

    @property
    def available(self) -> bool:
        """Whether extraction is given these masks."""
        return self.state == STATE_OK and bool(self.masked)

    @property
    def record_state(self) -> str:
        """`applied` / `partial` / `unavailable`, as the contract spells it."""
        if not self.available:
            return RECORD_UNAVAILABLE
        if self.unmasked or len(self.masked) < self.images:
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
    for name in names:
        try:
            images.append((name, images_dir / name, file_sha1(images_dir / name)))
        except OSError:
            out.unmasked.append(name)
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
        computed: set = set()
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

            def emit(i, hand, phone, seconds, c=c, by_index=by_index):
                name, sha1 = by_index[i]
                T.write_component(component_path(cdir, name, c, sha1),
                                  mask_key(c, params, name, sha1),
                                  np.asarray(hand, bool), np.asarray(phone, bool),
                                  image_sha1=sha1, seconds=seconds)
                computed.add(name)

            t2 = time.time()
            try:
                timings = backend.run(items, params, emit)
            except T.TransientDetectorUnavailable as exc:
                T._log_unavailable_once(str(exc))
                if T._can_fall_back(params, {c: str(exc)}):
                    params = _fall_back(out, params, {c: str(exc)})
                    continue
                out.state, out.detail = STATE_UNAVAILABLE, str(exc)
                out.seconds["total"] = round(time.time() - t0, 3)
                return out
            except Exception as exc:  # noqa: BLE001 -- OOM, device loss: a solve, unmasked
                logger.exception("[Tower][WorldBuilder][solve-masks] the %s detector failed; "
                                 "this solve runs unmasked", c)
                out.state = STATE_FAILED
                out.detail = f"the {c} detector failed ({type(exc).__name__}: {exc})"
                out.seconds["total"] = round(time.time() - t0, 3)
                return out
            out.seconds[c] = round(time.time() - t2, 3)
            for k, v in (timings or {}).items():
                if k != "stopped":
                    out.seconds[f"{c}.{k}"] = v
        out.computed = len(computed)
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
            # arrived). An all-255 mask says "extract everything" explicitly,
            # rather than leaving COLMAP to decide what a missing file means,
            # and the image is counted as unmasked.
            out.unmasked.append(name)
            if shape is not None:
                _write_png(png_path, np.full(shape, 255, np.uint8))
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


def _fall_back(out: SolverMasks, params: T.TransientParams, missing: dict) -> T.TransientParams:
    """`union` without Grounding DINO / SAM continues as `oneformer`, and says
    so -- the surface stage's own rule (`transients._fall_back`)."""
    why = "; ".join(f"{c}: {r}" for c, r in sorted(missing.items()))
    effective = replace(params, mode=T.MODE_ONEFORMER)
    out.params = effective
    out.partial = (f"{params.mode} was requested but only {T.MODE_ONEFORMER} could run ({why}); "
                   "these masks are OneFormer's alone")
    logger.warning("[Tower][WorldBuilder][solve-masks] %s", out.partial)
    return effective


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
    """The masked feature database, emptied first if any image it already holds
    was extracted under a different image or mask. Returns (path, info).

    Incremental like `database.db`: an image new since the last masked solve is
    simply extracted; one whose bytes or mask changed invalidates the file,
    because COLMAP has no way to re-extract one image in place.
    """
    db = masked_database_path(workspace)
    record_path = Path(workspace.root) / MASKED_DATABASE_RECORD
    current = {name: [v["image_sha1"], v["mask_sha1"]] for name, v in masks.masked.items()}
    for name in all_names:
        current.setdefault(name, [None, None])
    rule = masks.params.rule_id()
    previous = None
    try:
        previous = read_json_closed(record_path) if record_path.is_file() else None
    except (OSError, ValueError):
        previous = None
    stale_reason = None
    if db.exists():
        if not isinstance(previous, dict) or previous.get("rule") != rule:
            stale_reason = "no record of what the masked database was extracted under"
            if isinstance(previous, dict):
                stale_reason = f"mask rule changed from {previous.get('rule')!r}"
        else:
            held = previous.get("images") or {}
            for name, entry in held.items():
                if name in current and list(entry) != current[name]:
                    stale_reason = f"{name} was extracted under a different image or mask"
                    break
    reused = db.exists() and stale_reason is None
    if db.exists() and stale_reason is not None:
        logger.info("[Tower][WorldBuilder][solve-masks] rebuilding %s: %s", db.name, stale_reason)
        for side in ("", "-wal", "-shm"):
            Path(str(db) + side).unlink(missing_ok=True)
    held = dict((previous or {}).get("images") or {}) if reused else {}
    held.update(current)
    # Written BEFORE extraction: an interrupted extraction leaves images in the
    # database that the record already names, never the reverse.
    write_json_atomic(record_path, {"schema": SOLVER_MASK_SCHEMA, "rule": rule, "images": held})
    return db, {"database": db.name, "reused": bool(reused), "rebuilt_because": stale_reason}


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
