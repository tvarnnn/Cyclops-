"""The solver-variant driver: one recipe, many arms, every arm scored the same way.

OFF/OFF IS THE PRODUCT. With `masks="none"`, `gate="none"`, no extra images
and no augment hook, `run_variant` runs `global_solve.solve` (lines ~754-909)
step for step, reusing its functions and constants: the keyframe's raw
capture frame when on disk (`_source_frame`), undistorted once with
`_undistort_maps` (alpha 0, cropped) and written JPEG q95 exactly as
`prepare_images`; SIFT on CPU with `MAX_FEATURES`; a single PINHOLE camera
with the session calibration; `match_sequential` with `SEQUENTIAL_OVERLAP`,
`quadratic_overlap=False` and loop detection on (vocabulary tree from the
user cache, refused rather than downloaded); GLOMAP with focal length,
principal point and extra parameters fixed, incremental mapping only if
GLOMAP returns nothing; `MIN_IMAGE_OBSERVATIONS` / `MIN_MODEL_IMAGES` as the
publish floor.

The only deliberate differences are determinism hygiene, because GLOMAP is
not reproducible otherwise (D1 2.4): every mapper seed is set,
`pycolmap.set_random_seed(seed)` is called, mapping runs single-threaded,
and (by default) the two-view RANSAC seed is fixed. Each seed maps in its
OWN process on its OWN database copy (two mappers on one database: "database
is locked"; the 8th in-process run: MemoryError).

NAMES. Staged images are named `f"{capture_seq:06d}_k.jpg"` (keyframes) and
`f"{capture_seq:06d}_g.jpg"` (solver-only extras), `capture_seq` being the
position in the concatenated `frames.jsonl` of the session's capture chain.
COLMAP's sequential matcher orders images by name, so lexical order is
capture order and extras interleave between the keyframes they were
captured between. The product names keyframes by source sequence
(`00000042.jpg`); with keyframes only the two orders are identical (source
sequence is monotone along the chain), so the matched pairs are the same.

PUBLISHED = KEYFRAMES. Solver-only frames help the solve and are never
exported: the interchange output holds keyframes only, in the canonical
image space, so the harness scores the same fixed set of accepted keyframes
for every arm.

EXPERIMENT CODE: not imported by the builder path.
"""

from __future__ import annotations

import concurrent.futures as cf
import hashlib
import importlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tower.world_builder import global_solve as gs
from tower.world_builder.coherence_exp import gate as gate_mod
from tower.world_builder.coherence_exp import masks as masks_mod

DRIVER_VERSION = 2  # 2: the database is checkpointed out of WAL before it is copied
KEYFRAME_TAG = "k"
EXTRA_TAG = "g"
MASKS = ("none", "transients")
MAPPERS = ("glomap", "incremental")
GATES = ("none", "rigid")
SOURCES = ("raw", "redacted")
LANE_TOWER = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# configuration


@dataclass
class ExtraImage:
    """A solver-only image. Never published, never scored."""

    name_key: str
    undistorted_path: str
    capture_seq: int
    source: str = "raw"
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.source not in SOURCES:
            raise ValueError(f"ExtraImage.source {self.source!r} not in {SOURCES}")
        self.capture_seq = int(self.capture_seq)
        self.undistorted_path = str(self.undistorted_path)

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "ExtraImage":
        return cls(**{k: d[k] for k in ("name_key", "undistorted_path", "capture_seq", "source", "meta") if k in d})


@dataclass
class VariantConfig:
    name: str = "variant"
    masks: str = "none"
    extra_images: list = field(default_factory=list)
    augment: str | None = None
    augment_kwargs: dict = field(default_factory=dict)
    seeds: list = field(default_factory=lambda: [0, 1, 2, 3, 4])
    num_threads: int = 1
    mapper: str = "glomap"
    loop_detection: bool = True
    overlap: int = gs.SEQUENTIAL_OVERLAP
    gate: str = "none"
    gate_params: dict = field(default_factory=dict)
    verification_seed: int | None = 0
    min_image_observations: int = gs.MIN_IMAGE_OBSERVATIONS
    publish_min_model_keyframes: int = gs.MIN_MODEL_IMAGES
    max_parallel: int = 5
    keyframe_source: str = "raw"
    # Seed the database from an existing COLMAP database of the same keyframes (e.g. the frozen
    # product solve's, which carries the LIVE matching history) instead of extracting and
    # matching from scratch. Image names are rewritten to staged names on the copy. With
    # masks="transients", matches touching a masked keypoint are removed and those pairs are
    # re-verified (verify_matches, same options); untouched pairs keep their verification.
    database_from: str | None = None
    # Which seed's model is published and anchors the gate: "medoid" = the seed whose main
    # component agrees best with the others (generic, most reproducible); or an int seed.
    reference_seed: object = "medoid"

    def __post_init__(self):
        if self.masks not in MASKS:
            raise ValueError(f"masks {self.masks!r} not in {MASKS}")
        if self.mapper not in MAPPERS:
            raise ValueError(f"mapper {self.mapper!r} not in {MAPPERS}")
        if self.gate not in GATES:
            raise ValueError(f"gate {self.gate!r} not in {GATES}")
        if self.keyframe_source not in SOURCES:
            raise ValueError(f"keyframe_source {self.keyframe_source!r} not in {SOURCES}")
        if not self.seeds:
            raise ValueError("at least one seed")
        self.seeds = [int(s) for s in self.seeds]
        self.extra_images = [e if isinstance(e, ExtraImage) else ExtraImage.from_json(e)
                             for e in (self.extra_images or [])]
        if callable(self.augment):
            self.augment = f"{self.augment.__module__}:{self.augment.__qualname__}"
        if self.database_from is not None:
            self.database_from = str(self.database_from)
            if self.extra_images or self.augment:
                raise ValueError("database_from cannot be combined with extra images or an augment hook")
        if self.reference_seed != "medoid":
            self.reference_seed = int(self.reference_seed)
            if self.reference_seed not in self.seeds:
                raise ValueError(f"reference_seed {self.reference_seed} is not one of the seeds")

    def to_json(self) -> dict:
        d = asdict(self)
        d["extra_images"] = [e.to_json() for e in self.extra_images]
        return d

    @classmethod
    def from_json(cls, d: dict) -> "VariantConfig":
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)

    def is_product_recipe(self) -> bool:
        return (self.masks == "none" and not self.extra_images and self.augment is None
                and self.mapper == "glomap" and self.loop_detection
                and self.overlap == gs.SEQUENTIAL_OVERLAP and self.gate == "none"
                and self.keyframe_source == "raw" and self.database_from is None)


# ---------------------------------------------------------------------------
# the world and its capture chain


def open_world(world_dir):
    from tower.world_builder.coherence_eval.eval_world import open_world as _open

    return _open(Path(world_dir))


@dataclass
class CaptureIndex:
    """Every frame of the session's capture chain, in capture order."""

    frames: list
    chain: list
    keyframe_rows: dict
    notes: list = field(default_factory=list)

    def seq_of_source(self, source_seq: int, capture_id: str | None = None):
        for f in self.frames:
            if f["source_seq"] == int(source_seq) and (capture_id is None or f["capture_id"] == capture_id):
                return f["capture_seq"]
        return None

    def keyframe_seq(self, keyframe_id: str):
        row = self.keyframe_rows.get(keyframe_id)
        return None if row is None else row["capture_seq"]

    def between(self, seq_a: int, seq_b: int) -> list:
        """Frames strictly between two capture positions."""
        return [f for f in self.frames if seq_a < f["capture_seq"] < seq_b]


def capture_index(world_dir, captures_root) -> CaptureIndex:
    from tower.world_builder.coherence_eval import trace

    world = open_world(world_dir)
    chain = []
    if world.capture_id and captures_root is not None:
        chain = trace.capture_chain(Path(captures_root), world.capture_id, world.session)
    rows = []
    for cdir in chain:
        for r in trace._read_jsonl(Path(cdir) / "frames.jsonl"):
            rows.append({"capture_id": Path(cdir).name, "source_seq": int(r["source_seq"]),
                         "path": str(Path(cdir) / r["relpath"]), "received_at": r.get("received_at")})
    notes = []
    seqs = [r["source_seq"] for r in rows]
    if any(b <= a for a, b in zip(seqs, seqs[1:])):
        notes.append("source_seq is not strictly increasing along the chain; capture order = chain/file order")
    by_seq = {}
    for i, r in enumerate(rows):
        by_seq.setdefault(r["source_seq"], i)
    kf_pos = {}
    missing = []
    for k in world.keyframes:
        s = int(k["source_seq"])
        if s in by_seq:
            kf_pos[k["keyframe_id"]] = by_seq[s]
        else:
            missing.append(k)
    if missing:
        # keyframes absent from the capture: merged by source sequence (recorded)
        notes.append(f"{len(missing)} keyframes are not in the capture chain; ordered by source_seq")
        for k in missing:
            rows.append({"capture_id": None, "source_seq": int(k["source_seq"]), "path": None,
                         "received_at": k.get("received_at"), "_kf": k["keyframe_id"]})
        order = sorted(range(len(rows)), key=lambda i: (rows[i]["source_seq"], i))
        rows = [rows[i] for i in order]
        kf_pos = {}
        by_seq = {}
        for i, r in enumerate(rows):
            if "_kf" in r:
                kf_pos[r.pop("_kf")] = i
            else:
                by_seq.setdefault(r["source_seq"], i)
        for k in world.keyframes:
            if k["keyframe_id"] not in kf_pos:
                kf_pos[k["keyframe_id"]] = by_seq[int(k["source_seq"])]
    kf_of_pos = {p: kid for kid, p in kf_pos.items()}
    frames = []
    for i, r in enumerate(rows):
        frames.append(dict(r, capture_seq=i, is_keyframe=i in kf_of_pos, keyframe_id=kf_of_pos.get(i)))
    keyframe_rows = {kid: frames[p] for kid, p in kf_pos.items()}
    return CaptureIndex(frames=frames, chain=[str(c) for c in chain], keyframe_rows=keyframe_rows, notes=notes)


def staged_name(capture_seq: int, tag: str) -> str:
    return f"{int(capture_seq):06d}_{tag}.jpg"


def canonical_camera(world_dir) -> dict:
    return dict(open_world(world_dir).canonical_camera)


def undistort_frame(world_dir, raw_bgr, world=None) -> np.ndarray:
    """A raw capture frame -> the canonical solver image (prepare_images' maps)."""
    import cv2

    world = world or open_world(world_dir)
    m1, m2, (x, y, rw, rh), _ = world._maps()
    return cv2.remap(raw_bgr, m1, m2, cv2.INTER_LINEAR)[y:y + rh, x:x + rw]


def _sha1(path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_jpeg(path: Path, image) -> None:
    import cv2

    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise OSError(f"cannot encode {path}")
    path.write_bytes(buf.tobytes())


def stage_images(world, cindex: CaptureIndex, workspace: Path, config: VariantConfig, *, log=print) -> dict:
    """Undistort every keyframe (and copy every extra) into `workspace/images`
    under capture-order names; write `workspace/staging.json`."""
    import cv2

    images_dir = workspace / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    m1, m2, (x, y, rw, rh), camera = world._maps()
    capture_dirs = [Path(c) for c in cindex.chain]
    rows = []
    skipped = []
    for k in world.keyframes:
        seq = cindex.keyframe_seq(k["keyframe_id"])
        name = staged_name(seq, KEYFRAME_TAG)
        kf = SimpleNamespace(keyframe_id=k["keyframe_id"], image_relpath=k["image_relpath"])
        if config.keyframe_source == "raw":
            src = gs._source_frame(kf, world.session_dir, capture_dirs, sources=None)
        else:
            src = world.session_dir / k["image_relpath"]
        kind = "redacted" if Path(src).resolve().is_relative_to(world.session_dir.resolve()) else "raw"
        image = cv2.imdecode(np.fromfile(str(src), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape[1] != int(k["width"]) or image.shape[0] != int(k["height"]):
            skipped.append({"keyframe_id": k["keyframe_id"], "source": str(src), "why": "unreadable or wrong size"})
            continue
        und = cv2.remap(image, m1, m2, cv2.INTER_LINEAR)[y:y + rh, x:x + rw]
        target = images_dir / name
        _write_jpeg(target, und)
        rows.append({"name": name, "kind": "keyframe", "keyframe_id": k["keyframe_id"], "name_key": None,
                     "capture_seq": int(seq), "source": kind, "source_path": str(src), "sha1": _sha1(target)})
    used = {r["name"] for r in rows}
    for e in config.extra_images:
        name = staged_name(e.capture_seq, EXTRA_TAG)
        if name in used:
            raise ValueError(f"two staged images named {name} (extra {e.name_key})")
        used.add(name)
        img = cv2.imdecode(np.fromfile(e.undistorted_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None or img.shape[1] != camera.width or img.shape[0] != camera.height:
            raise ValueError(f"extra {e.name_key}: {e.undistorted_path} is not a canonical "
                             f"{camera.width}x{camera.height} image")
        target = images_dir / name
        shutil.copyfile(e.undistorted_path, target)
        rows.append({"name": name, "kind": "extra", "keyframe_id": None, "name_key": e.name_key,
                     "capture_seq": int(e.capture_seq), "source": e.source,
                     "source_path": e.undistorted_path, "sha1": _sha1(target), "meta": e.meta})
    rows.sort(key=lambda r: r["name"])
    staging = {"camera": camera.to_json_dict(), "images": rows, "skipped": skipped,
               "capture_chain": cindex.chain, "capture_notes": cindex.notes,
               "keyframes": sum(1 for r in rows if r["kind"] == "keyframe"),
               "extras": sum(1 for r in rows if r["kind"] == "extra")}
    (workspace / "staging.json").write_text(json.dumps(staging, indent=1), encoding="utf-8")
    log(f"staged {staging['keyframes']} keyframes ({sum(1 for r in rows if r['kind'] == 'keyframe' and r['source'] == 'raw')} raw) "
        f"+ {staging['extras']} extras; skipped {len(skipped)}")
    return staging


# ---------------------------------------------------------------------------
# features and matches


def _vocab_ok(wanted: bool) -> bool:
    if wanted and not gs.vocabulary_tree_cached():
        raise RuntimeError(f"loop detection requested but no vocabulary tree in {gs.vocabulary_tree_cache_dir()}; "
                           "the driver never downloads it")
    return wanted


def database_key(staging: dict, config: VariantConfig, mask_rule: str | None) -> str:
    import pycolmap

    doc = {
        "driver": DRIVER_VERSION, "pycolmap": getattr(pycolmap, "__version__", "?"),
        "images": [[r["name"], r["sha1"]] for r in staging["images"]],
        "camera": staging["camera"], "max_features": gs.MAX_FEATURES,
        "masks": config.masks, "mask_rule": mask_rule,
        "overlap": config.overlap, "loop": config.loop_detection,
        "verification_seed": config.verification_seed,
        "augment": config.augment, "augment_kwargs": config.augment_kwargs,
    }
    if config.database_from:  # only when set, so keys of scratch-built databases stay stable
        doc["database_from"] = [str(config.database_from), _sha1(config.database_from)]
    return hashlib.sha1(json.dumps(doc, sort_keys=True, default=str).encode()).hexdigest()


def name_to_image_id(database_path) -> dict:
    con = sqlite3.connect(f"file:{Path(database_path).as_posix()}?mode=ro", uri=True)
    try:
        return {n: int(i) for i, n in con.execute("select image_id, name from images")}
    finally:
        con.close()


def resolve_augment(spec: str):
    module, _, attr = spec.partition(":")
    if not attr:
        raise ValueError(f"augment {spec!r}: expected 'package.module:function'")
    obj = importlib.import_module(module)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def build_database(workspace: Path, staging: dict, config: VariantConfig, *, masks_dir: Path | None,
                   log=print) -> dict:
    """extract_features + match_sequential(+loop) [+ augment] into workspace/database.db."""
    import pycolmap

    gs._quiet_pycolmap()
    db = workspace / "database.db"
    images_dir = workspace / "images"
    cam = staging["camera"]
    names = [r["name"] for r in staging["images"]]
    t0 = time.perf_counter()
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "PINHOLE"
    reader.camera_params = ",".join(str(v) for v in (cam["fx"], cam["fy"], cam["cx"], cam["cy"]))
    if masks_dir is not None:
        reader.mask_path = str(masks_dir)
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.num_threads = -1
    extraction.sift.max_num_features = gs.MAX_FEATURES
    pycolmap.extract_features(db, images_dir, image_names=names, camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=reader, extraction_options=extraction)
    t1 = time.perf_counter()
    matching = pycolmap.FeatureMatchingOptions()
    matching.num_threads = -1
    pairing = pycolmap.SequentialPairingOptions()
    pairing.overlap = config.overlap
    pairing.quadratic_overlap = False
    pairing.loop_detection = _vocab_ok(config.loop_detection)
    verification = pycolmap.TwoViewGeometryOptions()
    if config.verification_seed is not None:
        verification.ransac.random_seed = int(config.verification_seed)
    pycolmap.match_sequential(db, matching_options=matching, pairing_options=pairing,
                              verification_options=verification)
    t2 = time.perf_counter()
    augment_result = None
    if config.augment:
        fn = resolve_augment(config.augment)
        augment_result = fn(db, name_to_image_id(db), images_dir, workspace, **(config.augment_kwargs or {}))
    t3 = time.perf_counter()
    checkpoint_database(db)
    info = {"extract_s": round(t1 - t0, 3), "match_s": round(t2 - t1, 3), "augment_s": round(t3 - t2, 3),
            "augment": augment_result, "loop_detection": bool(pairing.loop_detection),
            "database_stats": database_stats(db)}
    log(f"database: extract {info['extract_s']} s, match {info['match_s']} s, "
        f"{info['database_stats']['verified_pairs']} verified pairs")
    return info


def seed_database_from(src_db, workspace: Path, staging: dict, world, config: VariantConfig, *,
                       cdir=None, log=print) -> dict:
    """Copy an existing database (never opened in place), rename its images to the staged names,
    and -- with masks -- drop matches on masked keypoints and re-verify the pairs that changed."""
    import pycolmap

    t0 = time.perf_counter()
    db = workspace / "database.db"
    shutil.copyfile(src_db, db)
    os.chmod(db, 0o666)
    staged = {r["keyframe_id"]: r for r in staging["images"] if r["kind"] == "keyframe"}
    con = sqlite3.connect(str(db))
    info = {"database_from": str(src_db), "source_sha1": _sha1(src_db)}
    changed = []
    try:
        rows = con.execute("select image_id, name from images").fetchall()
        unknown = []
        image_row = {}
        for iid, name in rows:
            kid = world.resolve_keyframe(name)
            if kid is None or kid not in staged:
                unknown.append(name)
                continue
            con.execute("update images set name=? where image_id=?", (staged[kid]["name"], iid))
            image_row[iid] = staged[kid]
        con.commit()
        if unknown:
            raise ValueError(f"{src_db}: {len(unknown)} images are not staged keyframes (first {unknown[:3]})")
        info["renamed"] = len(image_row)
        if config.masks == "transients":
            params = masks_mod.default_params()
            in_mask = {}
            for iid, r in image_row.items():
                m = masks_mod.load_mask(cdir, r["name"], r["sha1"], params)
                if m is None:
                    raise masks_mod.MasksMissing([r["name"]])
                got = con.execute("select rows, cols, data from keypoints where image_id=?", (iid,)).fetchone()
                if got is None or not got[0]:
                    in_mask[iid] = np.zeros(0, bool)
                    continue
                kp = np.frombuffer(got[2], np.float32).reshape(got[0], got[1])[:, :2]
                x = np.clip(np.floor(kp[:, 0]).astype(int), 0, m.shape[1] - 1)
                y = np.clip(np.floor(kp[:, 1]).astype(int), 0, m.shape[0] - 1)
                in_mask[iid] = m[y, x]
            info["keypoints_in_mask"] = int(sum(v.sum() for v in in_mask.values()))
            info["keypoints_total"] = int(sum(len(v) for v in in_mask.values()))
            base = 2147483647
            dropped = 0
            for pid, n, cols, data in con.execute("select pair_id, rows, cols, data from matches").fetchall():
                if not n:
                    continue
                i2 = pid % base
                i1 = (pid - i2) // base
                mm = np.frombuffer(data, np.uint32).reshape(n, cols)
                bad = in_mask[i1][mm[:, 0]] | in_mask[i2][mm[:, 1]]
                if not bad.any():
                    continue
                keep = np.ascontiguousarray(mm[~bad])
                dropped += int(bad.sum())
                con.execute("update matches set rows=?, data=? where pair_id=?",
                            (int(len(keep)), keep.tobytes(), pid))
                con.execute("delete from two_view_geometries where pair_id=?", (pid,))
                changed.append((image_row[i1]["name"], image_row[i2]["name"]))
            con.commit()
            info.update({"matches_dropped": dropped, "pairs_changed": len(changed)})
    finally:
        con.close()
    if changed:
        pairs = workspace / "reverify_pairs.txt"
        pairs.write_text("".join(f"{a} {b}\n" for a, b in changed), encoding="utf-8")
        verification = pycolmap.TwoViewGeometryOptions()
        if config.verification_seed is not None:
            verification.ransac.random_seed = int(config.verification_seed)
        gs._quiet_pycolmap()
        pycolmap.verify_matches(db, pairs, options=verification)
    checkpoint_database(db)
    info["seconds"] = round(time.perf_counter() - t0, 3)
    info["database_stats"] = database_stats(db)
    log(f"database seeded from {src_db}: {info.get('pairs_changed', 0)} pairs re-verified, "
        f"{info['database_stats']['verified_pairs']} verified pairs")
    return info


def checkpoint_database(db) -> None:
    """Fold COLMAP's write-ahead log into the database file and leave it in
    rollback-journal mode, so that a plain file copy is the whole database.

    COLMAP opens its database in WAL mode; the last matches can sit in
    `database.db-wal` after the call returns, and copying only `database.db`
    (to a cache, or to a seed's own copy) then silently loses them."""
    import gc

    gc.collect()  # drop any pycolmap Database handle before checkpointing
    con = sqlite3.connect(str(db))
    try:
        busy, log_frames, done = con.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        mode = con.execute("PRAGMA journal_mode=DELETE").fetchone()[0]
    finally:
        con.close()
    if busy or str(mode).lower() != "delete":
        raise RuntimeError(f"{db}: WAL checkpoint incomplete (busy={busy}, journal_mode={mode})")
    for side in ("-wal", "-shm"):
        side_path = Path(str(db) + side)
        if side_path.exists() and side_path.stat().st_size:
            raise RuntimeError(f"{side_path} is not empty after the checkpoint")


def database_stats(db) -> dict:
    con = sqlite3.connect(f"file:{Path(db).as_posix()}?mode=ro", uri=True)
    try:
        kp = [r[0] for r in con.execute("select rows from keypoints")]
        tv = [r[0] for r in con.execute("select rows from two_view_geometries")]
    finally:
        con.close()
    return {"images": len(kp), "keypoints_median": float(np.median(kp)) if kp else 0.0,
            "keypoints_total": int(sum(kp)), "pairs": len(tv), "verified_pairs": int(sum(1 for v in tv if v >= 15))}


# ---------------------------------------------------------------------------
# mapping, one seed per process


def _mapper_options(seed: int, threads: int, mapper: str):
    import pycolmap

    if mapper == "glomap":
        o = pycolmap.GlobalPipelineOptions()
        o.num_threads = threads
        o.mapper.num_threads = threads
        o.mapper.bundle_adjustment.refine_focal_length = False
        o.mapper.bundle_adjustment.refine_principal_point = False
        o.mapper.bundle_adjustment.refine_extra_params = False
        o.random_seed = seed
        o.mapper.random_seed = seed
        o.mapper.rotation_averaging.random_seed = seed
        o.mapper.global_positioning.random_seed = seed
        o.mapper.retriangulation.random_seed = seed
        return o
    o = pycolmap.IncrementalPipelineOptions()
    o.num_threads = threads
    o.ba_refine_focal_length = False
    o.ba_refine_principal_point = False
    o.ba_refine_extra_params = False
    o.random_seed = seed
    o.mapper.random_seed = seed
    o.triangulation.random_seed = seed
    return o


def map_one(database: Path, images: Path, out: Path, seed: int, threads: int, mapper: str) -> dict:
    """Runs in its own process. Writes out/model.npz, out/summary.json, out/sparse/<k>/."""
    import pycolmap

    gs._quiet_pycolmap()
    out.mkdir(parents=True, exist_ok=True)
    sparse = out / "sparse_raw"
    sparse.mkdir(exist_ok=True)
    pycolmap.set_random_seed(int(seed))
    t0 = time.perf_counter()
    solver = mapper
    recs = {}
    error = None
    try:
        if mapper == "glomap":
            recs = pycolmap.global_mapping(database, images, sparse, options=_mapper_options(seed, threads, "glomap"))
        else:
            recs = pycolmap.incremental_mapping(database, images, sparse,
                                                options=_mapper_options(seed, threads, "incremental"))
    except Exception as exc:  # a solver failure is a refusal (product)
        error = f"{type(exc).__name__}: {exc}"
        recs = {}
    if not recs and mapper == "glomap":
        solver = "incremental(fallback)"
        pycolmap.set_random_seed(int(seed))
        try:
            recs = pycolmap.incremental_mapping(database, images, sparse,
                                                options=_mapper_options(seed, threads, "incremental"))
        except Exception as exc:
            error = (error or "") + f"; incremental: {type(exc).__name__}: {exc}"
            recs = {}
    seconds = time.perf_counter() - t0
    models = sorted(recs.values(), key=lambda m: -m.num_reg_images())
    names, comp, R, t, nobs = [], [], [], [], []
    xyz, pcomp, perr, oi, op, ouv = [], [], [], [], [], []
    for k, rec in enumerate(models):
        rec.write(str(_mkdir(out / "sparse" / str(k))))
        idx = {}
        for image in rec.images.values():
            if not image.has_pose:
                continue
            cfw = image.cam_from_world()
            idx[image.image_id] = (len(names), image)
            names.append(image.name)
            comp.append(k)
            R.append(np.asarray(cfw.rotation.matrix(), dtype=np.float64))
            t.append(np.asarray(cfw.translation, dtype=np.float64))
            nobs.append(int(image.num_points3D))
        for _, point in rec.points3D.items():
            els = [e for e in point.track.elements if e.image_id in idx]
            if not els:
                continue
            pi = len(xyz)
            xyz.append(np.asarray(point.xyz, dtype=np.float64))
            pcomp.append(k)
            perr.append(float(point.error))
            for e in els:
                ii, image = idx[e.image_id]
                oi.append(ii)
                op.append(pi)
                ouv.append(np.asarray(image.points2D[e.point2D_idx].xy, dtype=np.float64))
    np.savez_compressed(
        out / "model.npz",
        names=np.asarray(names, dtype=str), component=np.asarray(comp, np.int32),
        R_cw=np.asarray(R, np.float64).reshape(-1, 3, 3), t_cw=np.asarray(t, np.float64).reshape(-1, 3),
        n_obs=np.asarray(nobs, np.int32), xyz=np.asarray(xyz, np.float64).reshape(-1, 3),
        point_component=np.asarray(pcomp, np.int32), point_error=np.asarray(perr, np.float32),
        obs_image=np.asarray(oi, np.int32), obs_point=np.asarray(op, np.int32),
        obs_uv=np.asarray(ouv, np.float64).reshape(-1, 2))
    summary = {"seed": int(seed), "threads": int(threads), "mapper": mapper, "solver": solver,
               "seconds": round(seconds, 3), "models": [int(m.num_reg_images()) for m in models],
               "points": int(len(xyz)), "error": error,
               "pycolmap": getattr(pycolmap, "__version__", "?")}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def _mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_seed_model(map_dir: Path, seed: int) -> gate_mod.SeedModel:
    with np.load(Path(map_dir) / "model.npz", allow_pickle=False) as z:
        return gate_mod.SeedModel(
            seed=int(seed), names=[str(s) for s in z["names"]], component=z["component"].astype(np.int64),
            R_cw=z["R_cw"].reshape(-1, 3, 3), t_cw=z["t_cw"].reshape(-1, 3), n_obs=z["n_obs"].astype(np.int64),
            xyz=z["xyz"].reshape(-1, 3), point_component=z["point_component"].astype(np.int64),
            obs_image=z["obs_image"].astype(np.int64), obs_point=z["obs_point"].astype(np.int64),
            obs_uv=z["obs_uv"].reshape(-1, 2),
            meta=json.loads((Path(map_dir) / "summary.json").read_text(encoding="utf-8")))


def _subprocess_env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(LANE_TOWER) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def run_seeds(database: Path, images: Path, seed_dirs: dict, config: VariantConfig, *, log=print) -> dict:
    """Map every seed not already mapped, each in its own process on its own
    database copy, at most `config.max_parallel` at once."""
    from tower.world_builder.coherence_eval.metrics import measure_command

    records = {}
    todo = []
    for seed, d in seed_dirs.items():
        if (d / "model.npz").is_file() and (d / "summary.json").is_file():
            records[seed] = {"cached": True}
        else:
            todo.append(seed)

    def one(seed):
        d = seed_dirs[seed]
        d.mkdir(parents=True, exist_ok=True)
        work = d.parent / f"{d.name}.work"
        work.mkdir(parents=True, exist_ok=True)
        db = work / "database.db"
        shutil.copyfile(database, db)
        cmd = [sys.executable, "-m", "tower.world_builder.coherence_exp.driver", "map-one",
               "--db", str(db), "--images", str(images), "--out", str(work / "map"),
               "--seed", str(seed), "--threads", str(config.num_threads), "--mapper", config.mapper]
        rec = measure_command(cmd, poll_s=1.0)   # inherits os.environ (PYTHONPATH set below)
        if rec.get("returncode") != 0 or not (work / "map" / "model.npz").is_file():
            raise RuntimeError(f"seed {seed}: mapping failed (rc {rec.get('returncode')}); see {work}")
        # publish into the cache dir (rename is atomic; another run may have won)
        for item in ("model.npz", "summary.json"):
            shutil.copyfile(work / "map" / item, d / f"{item}.tmp")
        if (work / "map" / "sparse").is_dir():
            shutil.copytree(work / "map" / "sparse", d / "sparse", dirs_exist_ok=True)
        (d / "runtime.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        os.replace(d / "summary.json.tmp", d / "summary.json")
        os.replace(d / "model.npz.tmp", d / "model.npz")
        return seed, rec

    if todo:
        log(f"mapping seeds {todo} ({config.mapper}, {config.num_threads} thread(s), "
            f"{min(config.max_parallel, len(todo))} at once)")
        # measure_command inherits os.environ: set the children's PYTHONPATH once, before any thread
        saved = os.environ.get("PYTHONPATH")
        os.environ["PYTHONPATH"] = _subprocess_env()["PYTHONPATH"]
        try:
            with cf.ThreadPoolExecutor(max_workers=max(1, min(config.max_parallel, len(todo)))) as pool:
                futures = [pool.submit(one, s) for s in todo]
                for fut in cf.as_completed(futures):
                    seed, rec = fut.result()
                    records[seed] = {"cached": False, "wall_s": rec.get("wall_s"),
                                     "peak_rss_mb": rec.get("peak_rss_mb")}
                    log(f"  seed {seed}: {rec.get('wall_s')} s, peak RSS {rec.get('peak_rss_mb')} MB")
        finally:
            if saved is None:
                os.environ.pop("PYTHONPATH", None)
            else:
                os.environ["PYTHONPATH"] = saved
    return records


# ---------------------------------------------------------------------------
# export


def _T_world_camera(R_cw, t_cw) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R_cw.T
    T[:3, 3] = -R_cw.T @ t_cw
    return T


def export_variant(model: gate_mod.SeedModel, labels: dict | None, staging: dict, world, out_dir: Path,
                   config: VariantConfig, *, name: str, meta: dict) -> dict:
    """Interchange directory for the harness: KEYFRAMES ONLY, canonical image
    space. `labels`: staged name -> final component (None = the model's own).
    Publish rule: a component publishes if it has >= publish_min_model_keyframes
    registered keyframes; a keyframe publishes if its component does and it has
    >= min_image_observations observations."""
    from tower.world_builder.coherence_eval import eval_variant as ev

    by_name = {r["name"]: r for r in staging["images"]}
    lab = np.array([labels[n] if labels is not None else int(c) for n, c in zip(model.names, model.component)])
    is_kf = np.array([by_name.get(n, {}).get("kind") == "keyframe" for n in model.names])
    kf_count = {int(c): int(((lab == c) & is_kf).sum()) for c in np.unique(lab)}
    comp_ok = {c: n >= config.publish_min_model_keyframes for c, n in kf_count.items()}
    cam = staging["camera"]
    v = ev.Variant(name=name, world_id=world.world_id, session_id=world.session_id, image_space="canonical")
    v.cameras = {"default": {"model": "PINHOLE", "width": int(cam["width"]), "height": int(cam["height"]),
                             "params": [float(cam["fx"]), float(cam["fy"]), float(cam["cx"]), float(cam["cy"])]}}
    kid_of = {}
    for i, n in enumerate(model.names):
        if not is_kf[i]:
            continue
        kid = by_name[n]["keyframe_id"]
        kid_of[i] = kid
        v.poses[kid] = _T_world_camera(model.R_cw[i], model.t_cw[i])
        v.component[kid] = str(int(lab[i]))
        v.camera_of[kid] = "default"
        published = comp_ok[int(lab[i])] and int(model.n_obs[i]) >= config.min_image_observations
        v.status[kid] = ev.PUBLISHED if published else ev.POSED
    # points: component = majority of ALL observers' final labels; keep keyframe
    # observations in that component; drop points no keyframe observes
    P = len(model.xyz)
    if P and len(model.obs_point):
        obs_lab = lab[model.obs_image]
        order = np.lexsort((obs_lab, model.obs_point))
        op, ol = model.obs_point[order], obs_lab[order]
        point_lab = np.full(P, -1, dtype=np.int64)
        starts = np.flatnonzero(np.r_[True, op[1:] != op[:-1]])
        ends = np.r_[starts[1:], len(op)]
        for s, e in zip(starts, ends):
            vals, counts = np.unique(ol[s:e], return_counts=True)
            point_lab[op[s]] = vals[np.argmax(counts)]
        keep_obs = is_kf[model.obs_image] & (lab[model.obs_image] == point_lab[model.obs_point])
        used = np.unique(model.obs_point[keep_obs])
        new_index = np.full(P, -1, dtype=np.int64)
        new_index[used] = np.arange(len(used))
        kf_ids = world.keyframe_ids
        kf_index = {k: i for i, k in enumerate(kf_ids)}
        v.xyz = model.xyz[used]
        v.point_component = np.asarray([str(int(c)) for c in point_lab[used]])
        v.obs_keyframe_ids = kf_ids
        v.obs_keyframe = np.asarray([kf_index[kid_of[i]] for i in model.obs_image[keep_obs]], dtype=np.int64)
        v.obs_point = new_index[model.obs_point[keep_obs]]
        v.obs_uv = model.obs_uv[keep_obs]
    v.meta = dict(meta, variant=name, adapter="coherence_exp.driver",
                  publish_rule={"min_observations": config.min_image_observations,
                                "min_model_keyframes": config.publish_min_model_keyframes})
    ev.save_variant(v, out_dir)
    return {"keyframes_posed": len(v.poses), "keyframes_published": len(v.published_ids()),
            "points": 0 if v.xyz is None else int(len(v.xyz))}


def component_report(model: gate_mod.SeedModel, labels: dict | None, staging: dict, world,
                     config: VariantConfig, regions_dir=None) -> list:
    """Every final component: keyframes (registered / published), extras,
    footprint of the published keyframe centres, capture ranges, and -- for
    evaluation only, read after every decision -- a C0 label histogram."""
    by_name = {r["name"]: r for r in staging["images"]}
    lab = np.array([labels[n] if labels is not None else int(c) for n, c in zip(model.names, model.component)])
    centres = model.centres
    kidx = {k: i for i, k in enumerate(world.keyframe_ids)}
    regions = _load_regions(regions_dir, world.world_id)
    out = []
    for c in sorted(np.unique(lab).tolist()):
        sel = np.flatnonzero(lab == c)
        kf = [i for i in sel if by_name.get(model.names[i], {}).get("kind") == "keyframe"]
        n_kf = len(kf)
        pub = [i for i in kf if model.n_obs[i] >= config.min_image_observations] \
            if n_kf >= config.publish_min_model_keyframes else []
        idx = sorted(kidx[by_name[model.names[i]]["keyframe_id"]] for i in kf)
        pc = centres[pub] if pub else np.zeros((0, 3))
        row = {"component": int(c), "keyframes": n_kf, "keyframes_published": len(pub),
               "extras": int(len(sel) - n_kf), "keyframe_index_ranges": _runs(idx)}
        if len(pc):
            lo, hi = pc.min(0), pc.max(0)
            row.update({"bbox_min": lo.tolist(), "bbox_max": hi.tolist(), "extent_p5_p95": gate_mod.extent(pc)})
        if regions is not None:
            hist = {}
            for i in kf:
                r = regions.get(by_name[model.names[i]]["keyframe_id"], "unlabelled")
                hist[r] = hist.get(r, 0) + 1
            row["c0_regions_eval_only"] = dict(sorted(hist.items(), key=lambda kv: -kv[1]))
        out.append(row)
    return out


def _load_regions(regions_dir, world_id):
    if regions_dir is None:
        return None
    p = Path(regions_dir) / f"{world_id}_regions.csv"
    if not p.is_file():
        return None
    import csv

    with open(p, encoding="utf-8") as f:
        return {r["keyframe_id"]: r.get("region") or "unlabelled" for r in csv.DictReader(f)}


def _runs(idx) -> list:
    out, start, prev = [], None, None
    for p in idx:
        if start is None:
            start = prev = p
        elif p == prev + 1:
            prev = p
        else:
            out.append(f"{start}-{prev}" if start != prev else f"{start}")
            start = prev = p
    if start is not None:
        out.append(f"{start}-{prev}" if start != prev else f"{start}")
    return out


def lane_commit() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(LANE_TOWER), "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=20).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def lane_dirty(paths=("tower/world_builder/coherence_exp", "scripts/world_coherence_variant.py")) -> list:
    try:
        r = subprocess.run(["git", "-C", str(LANE_TOWER), "status", "--porcelain", "--", *paths],
                           capture_output=True, text=True, timeout=20)
        return [ln for ln in r.stdout.splitlines() if ln.strip()]
    except Exception:  # noqa: BLE001
        return []


# ---------------------------------------------------------------------------
# the entry point


def default_cache_root() -> Path:
    return Path(os.environ.get("WB_COHERENCE_EXP_CACHE",
                               r"C:\Users\tvllo\Projects\Glasses-scratch\wb-coherence-run-2026-09-23\experiments\cache"))


def run_variant(world_dir, captures_root, out_dir, config: VariantConfig, *, cache_root=None,
                regions_dir=None, log=print) -> dict:
    t_start = time.perf_counter()
    world_dir = Path(world_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_root = Path(cache_root) if cache_root is not None else default_cache_root()
    world = open_world(world_dir)
    (out_dir / "config.json").write_text(json.dumps(config.to_json(), indent=1), encoding="utf-8")
    workspace = out_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    timings = {}

    t = time.perf_counter()
    cindex = capture_index(world_dir, captures_root)
    staging = stage_images(world, cindex, workspace, config, log=log)
    timings["stage_s"] = round(time.perf_counter() - t, 3)

    # masks (CPU: cached only)
    masks_dir = None
    mask_rule = None
    mask_stats = None
    if config.masks == "transients":
        params = masks_mod.default_params()
        mask_rule = params.rule_id()
        cdir = masks_mod.cache_dir(cache_root, world.world_id)
        imgs = [{"name": r["name"], "sha1": r["sha1"]} for r in staging["images"]]
        masks_dir = workspace / "masks"
        mask_stats = masks_mod.write_colmap_masks(imgs, cdir, masks_dir, params)
        (out_dir / "mask_stats.json").write_text(json.dumps(
            {"summary": masks_mod.area_summary(mask_stats), "rule": mask_rule, "per_image": mask_stats},
            indent=1), encoding="utf-8")

    # database (cached by content key)
    t = time.perf_counter()
    key = database_key(staging, config, mask_rule)
    db_cache = cache_root / "db" / world.world_id / key[:16]
    db = workspace / "database.db"
    if (db_cache / "database.db").is_file() and (db_cache / "info.json").is_file():
        shutil.copyfile(db_cache / "database.db", db)
        db_info = json.loads((db_cache / "info.json").read_text(encoding="utf-8"))
        db_info["cached_from"] = str(db_cache)
        log(f"database: reused {db_cache}")
    else:
        if db.exists():
            raise RuntimeError(f"{db} exists from an earlier run; use a fresh out_dir")
        if config.database_from:
            db_info = seed_database_from(Path(config.database_from), workspace, staging, world, config,
                                         cdir=masks_mod.cache_dir(cache_root, world.world_id), log=log)
        else:
            db_info = build_database(workspace, staging, config, masks_dir=masks_dir, log=log)
        db_cache.mkdir(parents=True, exist_ok=True)
        tmp = db_cache / f"database.db.p{os.getpid()}.tmp"
        shutil.copyfile(db, tmp)
        os.replace(tmp, db_cache / "database.db")
        (db_cache / "info.json").write_text(json.dumps(dict(db_info, key=key), indent=1, default=str),
                                            encoding="utf-8")
    timings["database_s"] = round(time.perf_counter() - t, 3)

    # seeds
    t = time.perf_counter()
    map_root = cache_root / "map" / world.world_id / key[:16]
    seed_dirs = {s: map_root / f"{config.mapper}-s{s}-t{config.num_threads}" for s in config.seeds}
    seed_runs = run_seeds(db_cache / "database.db", workspace / "images", seed_dirs, config, log=log)
    models = [load_seed_model(seed_dirs[s], s) for s in config.seeds]
    kf_names_all = {r["name"] for r in staging["images"] if r["kind"] == "keyframe"}
    ref_choice = gate_mod.choose_reference(models, config.reference_seed, names=kf_names_all)
    models = [models[ref_choice["index"]]] + [m for i, m in enumerate(models) if i != ref_choice["index"]]
    timings["map_s"] = round(time.perf_counter() - t, 3)

    kf_names = {r["name"] for r in staging["images"] if r["kind"] == "keyframe"}
    base_meta = {"source_world": str(world_dir), "driver": DRIVER_VERSION, "code_commit": lane_commit(),
                 "code_dirty": lane_dirty(), "params": config.to_json(), "database_key": key,
                 "product_recipe": config.is_product_recipe(),
                 "privacy": "research/unredacted: solver images are raw capture frames; only poses/points exported"}

    # per-seed exports (before the gate)
    seeds_report = {"seeds": config.seeds, "per_seed": [], "components": gate_mod.seed_components(models, kf_names),
                    "spread_main_component": gate_mod.seed_spread(models, kf_names)}
    for m in models:
        d = out_dir / "seeds" / str(m.seed) / "variant"
        info = export_variant(m, None, staging, world, d, config, name=f"{config.name}-s{m.seed}",
                              meta=dict(base_meta, seed=m.seed, mapping=m.meta))
        seeds_report["per_seed"].append({"seed": m.seed, "mapping": m.meta, "export": info,
                                         "cached": seed_runs.get(m.seed, {}).get("cached"),
                                         "wall_s": seed_runs.get(m.seed, {}).get("wall_s"),
                                         "peak_rss_mb": seed_runs.get(m.seed, {}).get("peak_rss_mb")})

    # gate
    t = time.perf_counter()
    labels = None
    gate_report = None
    if config.gate == "rigid":
        gp = gate_mod.GateParams.from_json(config.gate_params)
        gate_report = gate_mod.apply_rigid_gate(models, gp)
        labels = gate_report["labels"]
        (out_dir / "gate.json").write_text(json.dumps(gate_mod._json_clean(
            {k: v for k, v in gate_report.items() if k != "labels"}), indent=1), encoding="utf-8")
    timings["gate_s"] = round(time.perf_counter() - t, 3)

    ref = models[0]
    comps = component_report(ref, labels, staging, world, config, regions_dir=regions_dir)
    total_kf = len(world.keyframes)
    published = sum(c["keyframes_published"] for c in comps)
    main_pub = comps and max(c["keyframes_published"] for c in comps)
    components_doc = {"world_id": world.world_id, "accepted_keyframes": total_kf,
                      "published_keyframes": published, "main_component_published": main_pub,
                      "unplaced_keyframes": total_kf - published,
                      "unplaced_or_outside_main": total_kf - (main_pub or 0),
                      "reference_seed": ref.seed, "gate": config.gate, "components": comps}
    (out_dir / "components.json").write_text(json.dumps(components_doc, indent=1), encoding="utf-8")
    (out_dir / "seeds.json").write_text(json.dumps(gate_mod._json_clean(seeds_report), indent=1), encoding="utf-8")

    timings["total_s"] = round(time.perf_counter() - t_start, 3)
    meta = dict(base_meta, timings=timings, database=db_info, staging={k: staging[k] for k in
                                                                         ("keyframes", "extras", "capture_chain",
                                                                          "capture_notes", "skipped")},
                mask_summary=None if mask_stats is None else masks_mod.area_summary(mask_stats),
                mask_rule=mask_rule, reference_seed=ref.seed, reference_choice=ref_choice,
                seed_runs={str(k): v for k, v in seed_runs.items()},
                gate_splits=None if gate_report is None else gate_report["splits"])
    export = export_variant(ref, labels, staging, world, out_dir / "variant", config, name=config.name, meta=meta)
    meta["export"] = export
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    log(f"{config.name}: {export['keyframes_published']}/{total_kf} keyframes published; "
        f"main {main_pub}; components {[c['keyframes_published'] for c in comps if c['keyframes_published']]}")
    return {"out_dir": str(out_dir), "components": components_doc, "export": export, "timings": timings,
            "gate_splits": meta["gate_splits"]}


# ---------------------------------------------------------------------------
# `python -m tower.world_builder.coherence_exp.driver map-one ...`


def _main(argv=None) -> int:
    import argparse

    from tower.artifact_paths import artifact_root_arg

    p = argparse.ArgumentParser(description="internal: one seeded mapping run in its own process")
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("map-one")
    m.add_argument("--db", type=artifact_root_arg, required=True)
    m.add_argument("--images", type=artifact_root_arg, required=True)
    m.add_argument("--out", type=artifact_root_arg, required=True)
    m.add_argument("--seed", type=int, required=True)
    m.add_argument("--threads", type=int, default=1)
    m.add_argument("--mapper", choices=MAPPERS, default="glomap")
    args = p.parse_args(argv)
    s = map_one(args.db, args.images, args.out, args.seed, args.threads, args.mapper)
    print(json.dumps(s))
    return 0


if __name__ == "__main__":
    import scipy.linalg  # noqa: F401  -- numpy/scipy on the main thread first (native_prewarm rule)

    raise SystemExit(_main())
