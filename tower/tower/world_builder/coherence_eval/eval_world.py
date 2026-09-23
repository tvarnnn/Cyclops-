"""Read-only access to one (frozen) saved world, for the evaluation harness.

Everything the harness needs from a world that does NOT depend on any
reconstruction: the accepted keyframes in capture order, the session
calibration, the canonical undistorted image of every keyframe, and what the
world's own logs recorded about runtime. A variant (any reconstruction of the
same keyframes) is evaluated against this, never against another variant.

Nothing here writes. The world directory is expected to be read-only
evidence (``world_freeze.py``).

KEYFRAME IDENTITY
-----------------
A keyframe is identified by its journal ``keyframe_id``
(``<session_id>:<8-digit source sequence>``). Its IMAGE NAME is the file name
of ``image_relpath`` (``00000042.jpg``); that is the name COLMAP knows it by
(``global_solve.keyframe_image_name``) and the name under ``solve/<sid>/images``.
Its INDEX is its position in ``sessions/<sid>/keyframes.jsonl``, which is
capture order (the journal is append-only in acceptance order). All three are
interchangeable through ``WorldInfo``.

CANONICAL IMAGE SPACE
---------------------
"canonical" = the session's raw 360x640 frames undistorted ONCE with the
session calibration by ``global_solve._undistort_maps`` (alpha 0, cropped to
the valid ROI): a PINHOLE camera, 359x639 on the Ray-Ban captures. It is the
space of ``solve/<sid>/images`` and of the global solve's observations, and it
is the space the harness's depth and pair caches are computed in.
"raw" = the 360x640 frames as captured, with the session's ``pinhole_radtan``
distortion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CANONICAL = "canonical"
RAW = "raw"
CUSTOM = "custom"


def _read_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_concatenated_json(path: Path) -> list:
    """`solve.log` is several pretty-printed JSON objects appended back to back."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    decoder = json.JSONDecoder()
    out, pos = [], 0
    while pos < len(text):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text):
            break
        try:
            obj, pos = decoder.raw_decode(text, pos)
        except json.JSONDecodeError:
            break
        out.append(obj)
    return out


@dataclass
class WorldInfo:
    world_dir: Path
    world_id: str
    session_id: str
    world: dict
    session: dict
    keyframes: list[dict]
    index_of: dict[str, int] = field(default_factory=dict)
    id_of_name: dict[str, str] = field(default_factory=dict)
    canonical_camera: dict | None = None
    canonical_roi: tuple | None = None
    canonical_camera_source: str | None = None

    # -- identity --------------------------------------------------------

    @property
    def n(self) -> int:
        return len(self.keyframes)

    @property
    def keyframe_ids(self) -> list[str]:
        return [k["keyframe_id"] for k in self.keyframes]

    def image_name(self, index: int) -> str:
        return Path(self.keyframes[index]["image_relpath"]).name

    def resolve_keyframe(self, key) -> str | None:
        """keyframe_id, image name (with or without directories) or index -> keyframe_id."""
        if isinstance(key, (int, np.integer)):
            return self.keyframes[int(key)]["keyframe_id"] if 0 <= int(key) < self.n else None
        key = str(key)
        if key in self.index_of:
            return key
        name = key.replace("\\", "/").rsplit("/", 1)[-1]
        return self.id_of_name.get(name)

    # -- paths -----------------------------------------------------------

    @property
    def session_dir(self) -> Path:
        return self.world_dir / "sessions" / self.session_id

    @property
    def solve_dir(self) -> Path:
        return self.world_dir / "solve" / self.session_id

    @property
    def derived_dir(self) -> Path:
        return self.world_dir / "derived" / self.session_id

    @property
    def capture_id(self) -> str | None:
        return self.session.get("capture_id")

    def capture_dirs(self) -> list[Path]:
        """Where the raw capture of this session may be on disk.

        Frozen evidence layout: ``<frozen>/worlds/<wid>`` beside
        ``<frozen>/captures/<cid>``. Live layout: ``<data>/world_builder/worlds/<wid>``
        with ``<data>/captures/<cid>``.
        """
        cid = self.capture_id
        if not cid:
            return []
        out = []
        for base in (self.world_dir.parent.parent / "captures",
                     self.world_dir.parent.parent.parent / "captures"):
            candidate = base / cid
            if candidate.is_dir():
                out.append(candidate)
        return out

    # -- cameras -----------------------------------------------------------

    @property
    def raw_camera(self) -> dict | None:
        intr = self.session.get("intrinsics") or {}
        if intr.get("fx") is None:
            return None
        return {
            "model": "PINHOLE_RADTAN",
            "width": int(intr.get("calibrated_width") or self.keyframes[0]["width"]),
            "height": int(intr.get("calibrated_height") or self.keyframes[0]["height"]),
            "params": [float(intr["fx"]), float(intr["fy"]), float(intr["cx"]), float(intr["cy"]),
                       *[float(v) for v in (intr.get("dist_coeffs") or [0, 0, 0, 0, 0])]],
        }

    def canonical_K(self) -> np.ndarray:
        c = self.canonical_camera
        return np.array([[c["fx"], 0.0, c["cx"]], [0.0, c["fy"], c["cy"]], [0.0, 0.0, 1.0]])

    def raw_to_canonical(self, uv) -> np.ndarray:
        """Raw (distorted, 360x640) pixels -> canonical undistorted pixels."""
        import cv2

        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        raw = self.raw_camera
        fx, fy, cx, cy = raw["params"][:4]
        dist = np.asarray(raw["params"][4:], dtype=np.float64)
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        x, y = self.canonical_roi[:2]
        c = self.canonical_camera
        P = np.array([[c["fx"], 0, c["cx"] + x], [0, c["fy"], c["cy"] + y], [0, 0, 1]], dtype=np.float64)
        if not len(uv):
            return uv
        out = cv2.undistortPoints(uv.reshape(-1, 1, 2), K, dist, P=P).reshape(-1, 2)
        return out - np.array([x, y], dtype=np.float64)

    # -- images ------------------------------------------------------------

    def canonical_image_source(self, index: int) -> tuple[Path | None, str]:
        """(path, kind): kind is 'solve-images' (already canonical), 'raw-capture'
        or 'session-keyframe' (both need undistorting)."""
        name = self.image_name(index)
        p = self.solve_dir / "images" / name
        if p.is_file():
            return p, "solve-images"
        for d in self.capture_dirs():
            for candidate in (d / "frames" / name, d / name):
                if candidate.is_file():
                    return candidate, "raw-capture"
        p = self.session_dir / self.keyframes[index]["image_relpath"]
        if p.is_file():
            return p, "session-keyframe"
        return None, "missing"

    def canonical_image(self, index: int, *, gray: bool = False):
        """The keyframe in canonical space as a uint8 array, and its source kind."""
        import cv2

        path, kind = self.canonical_image_source(index)
        if path is None:
            return None, kind
        flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flag)
        if image is None:
            return None, "unreadable"
        c = self.canonical_camera
        if kind == "solve-images":
            if image.shape[1] != c["width"] or image.shape[0] != c["height"]:
                return None, "solve-image-wrong-size"
            return image, kind
        m1, m2, (x, y, rw, rh), _ = self._maps()
        return cv2.remap(image, m1, m2, cv2.INTER_LINEAR)[y:y + rh, x:x + rw], kind

    _maps_cache = None

    def _maps(self):
        if self._maps_cache is None:
            from tower.world_builder.global_solve import _undistort_maps

            self._maps_cache = _undistort_maps(_IntrinsicsView(self.session["intrinsics"]),
                                               int(self.keyframes[0]["width"]),
                                               int(self.keyframes[0]["height"]))
        return self._maps_cache

    # -- runtime evidence --------------------------------------------------

    def runtime_evidence(self) -> dict:
        """What the world's own logs recorded about time and resources.

        Solve timing: `solve.log` (every background solve, appended) and
        `solution.json` `timing` (the solve that produced the saved poses,
        `final: true`). Stage timing: `session.json` finalization/stages
        timestamps and the surface/appearance `status.json` results. Nothing in
        a saved world records GPU/VRAM or RAM; those are reported as not
        recorded, never as zero.
        """
        out: dict = {"gpu_vram_recorded": False, "ram_recorded": False}
        s = self.session
        if s.get("started_at") and s.get("ended_at"):
            out["capture_seconds"] = float(s["ended_at"]) - float(s["started_at"])
        fin = s.get("finalization") or {}
        if fin.get("started_at") and fin.get("updated_at"):
            out["finalization_seconds"] = float(fin["updated_at"]) - float(fin["started_at"])
            out["finalization_state"] = fin.get("state")
        stages = {}
        for name, st in sorted((s.get("stages") or {}).items()):
            row = {"state": st.get("state"), "attempted": st.get("attempted")}
            if st.get("started_at") and st.get("updated_at"):
                row["wall_seconds"] = float(st["updated_at"]) - float(st["started_at"])
            stages[name] = row
        for name in ("surface", "appearance"):
            p = self.world_dir / name / self.session_id / "status.json"
            if not p.is_file():
                continue
            try:
                st = _read_json(p)
            except (OSError, ValueError):
                continue
            row = stages.setdefault(name, {})
            result = st.get("result") or {}
            if isinstance(result.get("seconds"), dict):
                row["seconds"] = result["seconds"]
            detail = result.get("detail")
            if isinstance(detail, str) and detail.startswith("{"):
                try:
                    d = json.loads(detail)
                    for key in ("seconds", "timing", "timings"):
                        if key in d:
                            row["detail_" + key] = d[key]
                except ValueError:
                    pass
        out["stages"] = stages
        log = self.solve_dir / "solve.log"
        if log.is_file():
            solves = read_concatenated_json(log)
            out["solve_log"] = [
                {"keyframes": r.get("keyframes"), "keyframes_posed": r.get("keyframes_posed"),
                 "timing": r.get("timing")}
                for r in solves if isinstance(r, dict)
            ]
            total = 0.0
            for r in solves:
                t = (r or {}).get("timing") or {}
                total += sum(float(t.get(k) or 0.0) for k in ("prepare_s", "extract_s", "match_s", "map_s"))
            out["solve_log_total_seconds"] = total
        sol = self.solve_dir / "solution.json"
        if sol.is_file():
            try:
                timing = (_read_json(sol).get("timing") or {})
                out["final_solve_timing"] = timing
                out["final_solve_seconds"] = sum(
                    float(timing.get(k) or 0.0) for k in ("prepare_s", "extract_s", "match_s", "map_s"))
            except (OSError, ValueError):
                pass
        return out


class _IntrinsicsView:
    """Duck-typed stand-in for `records.Intrinsics` built from session.json."""

    def __init__(self, d: dict) -> None:
        self.fx = d.get("fx")
        self.fy = d.get("fy")
        self.cx = d.get("cx")
        self.cy = d.get("cy")
        self.dist_coeffs = d.get("dist_coeffs")
        self.calibrated_width = d.get("calibrated_width")
        self.calibrated_height = d.get("calibrated_height")


def open_world(world_dir, session_id: str | None = None) -> WorldInfo:
    world_dir = Path(world_dir)
    world = _read_json(world_dir / "world.json")
    sessions_root = world_dir / "sessions"
    sids = sorted(p.name for p in sessions_root.iterdir() if p.is_dir()) if sessions_root.is_dir() else []
    if session_id is None:
        if len(sids) != 1:
            raise ValueError(f"{world_dir}: pass a session id; sessions present: {sids}")
        session_id = sids[0]
    session = _read_json(sessions_root / session_id / "session.json")
    keyframes = _read_jsonl(sessions_root / session_id / "keyframes.jsonl")
    info = WorldInfo(world_dir=world_dir, world_id=world.get("world_id") or world_dir.name,
                     session_id=session_id, world=world, session=session, keyframes=keyframes)
    for i, k in enumerate(keyframes):
        k.setdefault("index", i)
        info.index_of[k["keyframe_id"]] = i
        info.id_of_name[Path(k["image_relpath"]).name] = k["keyframe_id"]
    # The canonical camera is COMPUTED from the calibration with the solver's
    # own function, so it exists for a world that never ran a solve, and it is
    # compared with solve/camera.json when that exists.
    if session.get("intrinsics", {}).get("fx") is not None and keyframes:
        try:
            _, _, roi, cam = info._maps()
            info.canonical_camera = cam.to_json_dict()
            info.canonical_roi = tuple(int(v) for v in roi)
            info.canonical_camera_source = "computed(global_solve._undistort_maps)"
        except Exception as exc:  # noqa: BLE001 -- recorded, then fall back
            info.canonical_camera_source = f"compute-failed: {type(exc).__name__}: {exc}"
    cam_path = info.solve_dir / "camera.json"
    if cam_path.is_file():
        stored = _read_json(cam_path)
        if info.canonical_camera is None:
            info.canonical_camera = stored
            info.canonical_roi = (0, 0, int(stored["width"]), int(stored["height"]))
            info.canonical_camera_source = "solve/camera.json"
        else:
            same = all(abs(float(stored[k]) - float(info.canonical_camera[k])) < 1e-6
                       for k in ("fx", "fy", "cx", "cy", "width", "height"))
            info.canonical_camera_source += "; matches solve/camera.json" if same else "; DIFFERS from solve/camera.json"
    return info
