"""The appearance stage: provenance, masks, occluders, gains, selection,
publication, routes. Contract: docs/contracts/WORLD-BUILDER-APPEARANCE.md.

Everything runs on a synthetic solved, surfaced world whose inputs are written
by hand: eight cameras looking at the same textured wall of a box room, their
redacted session keyframes rendered exactly from a world-space pattern, the
depth stage's records and maps, and a surface artifact that IS the box. So the
questions are about behaviour, and a leak is detectable by colour signature
rather than by face detection: the unredacted sources are painted a magenta
nothing else in the scene is.

Tests are numbered after the privacy lane's plan
(Glasses-scratch/wb-final-recon/fixit/privacy/PRIVACY.md §4) where they are one.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import io
import json
import pathlib
import time

import cv2
import numpy as np
import pytest

from tests.test_world_builder_surface import _look_from, _render_box_depth

from tower.world_builder import appearance as A
from tower.world_builder import appearance_pipeline as AP

WORLD, SESSION = "w1", "s1"
TRUSTED = "faces-detected-and-filled/yunet-2023mar@0.30+plausibility3"
FX, SW, SH = 300.0, 160, 120          # session keyframe camera
W, H = 159, 119                       # the solve camera: `_undistort_maps` ROI
MAGENTA = np.array([255, 0, 255], np.uint8)
N_FRAMES = 8


def _pattern(X):
    """The room's albedo: smooth, mid-grey, never near-black, never magenta."""
    r = 110 + 50 * np.sin(3.1 * X[..., 0] + 0.7 * X[..., 2])
    g = 115 + 45 * np.cos(2.3 * X[..., 1] - 0.5 * X[..., 0])
    b = 105 + 40 * np.sin(1.7 * (X[..., 0] + X[..., 1]) + X[..., 2])
    return np.stack([r, g, b], -1)


class World:
    """A solved, depth-staged, surfaced world, with helpers to edit it."""

    def __init__(self, root, label=TRUSTED, n=N_FRAMES):
        from tower.world_builder.dense import DenseParams
        from tower.world_builder.dense_pipeline import FILL_RULE
        from tower.world_builder.global_solve import Solution, workspace_for, write_solution
        from tower.world_builder.records import CameraIntrinsics, Session, World as WorldRec
        from tower.world_builder.store import WorldStore

        self.root = root
        self.store = WorldStore(root)
        self.store.write_world(WorldRec(world_id=WORLD, created_at=1.0, updated_at=2.0,
                                        session_ids=(SESSION,)))
        self.intrinsics = CameraIntrinsics(
            source="self_calibrated", model="pinhole", fx=FX, fy=FX, cx=SW / 2, cy=SH / 2,
            calibrated_width=SW, calibrated_height=SH)
        self.set_label(label)
        self.K = np.array([[FX, 0, SW / 2], [0, FX, SH / 2], [0, 0, 1.0]])
        self.kids, self.poses, self.depth = [], {}, []
        self.dense = self.store.world_dir(WORLD) / "dense" / SESSION
        (self.dense / "work" / "depth").mkdir(parents=True)
        self.images = self.store.images_dir(WORLD, SESSION)
        self.images.mkdir(parents=True)
        self.records = []
        self.gain = [1.0] * n
        for i in range(n):
            x = -0.6 + 1.2 * i / max(1, n - 1)
            eye = np.array([x, 0.05 * (i % 3), 0.0])
            target = np.array([0.4 * x, 0.1 * ((i % 2) - 0.5), 3.0])
            R, t = _look_from(eye, target)
            kid = f"{SESSION}:{i + 1:08d}"
            self.kids.append(kid)
            self.poses[kid] = {"component": 0, "rotation": R.ravel().tolist(),
                               "translation": t.tolist(), "observations": 50}
            z = _render_box_depth(R, t, self.K, SW, SH)
            self.depth.append(z)
            self.records.append({"ki": i, "kid": kid, "ok": True, "a": 1.0, "b": 0.0,
                                 "held_out_rel": 0.01, "fill_rule": FILL_RULE,
                                 "image_origin": "world-keyframe"})
            self.set_image(i, self.render(i))
            self.set_fill(i, np.zeros((H, W), bool))
            self.set_pred(i, z[:H, :W])
        self.write_align()
        self.solution = Solution(
            solver="glomap", solved_at=time.time(), input_digest="digest-1",
            keyframe_ids=self.kids, poses=self.poses,
            components=[{"index": 0, "images": n, "points": 16}],
            xyz=np.random.default_rng(0).uniform(-2, 2, (16, 3)).astype(np.float32),
            rgb=np.full((16, 3), 128, np.uint8), component=np.zeros(16, np.int32),
            first_keyframe=np.zeros(16, np.int32), track_length=np.full(16, 3, np.int32),
            error=np.full(16, 0.5, np.float32), observations=np.zeros((0, 3), np.int32),
            camera={"fx": FX, "fy": FX, "cx": SW / 2, "cy": SH / 2, "width": W, "height": H})
        write_solution(workspace_for(self.store, WORLD, SESSION), self.solution)
        self._write_solution = write_solution
        self._workspace = workspace_for(self.store, WORLD, SESSION)
        self.surface_built_at = None
        self.write_surface()
        self.dparams = DenseParams()

    # -- editing ------------------------------------------------------------

    def set_label(self, label):
        from tower.world_builder.records import Session

        self.store.write_session(Session(session_id=SESSION, world_id=WORLD, started_at=1.0,
                                         redaction=label, intrinsics=self.intrinsics))

    def pose(self, i):
        p = self.poses[self.kids[i]]
        return np.array(p["rotation"]).reshape(3, 3), np.array(p["translation"])

    def render(self, i):
        """The session keyframe: the wall pattern seen from camera i."""
        R, t = self.pose(i)
        z = self.depth[i]
        vv, uu = np.mgrid[0:SH, 0:SW].astype(float)
        xc = np.stack([(uu + 0.5 - self.K[0, 2]) / FX * z, (vv + 0.5 - self.K[1, 2]) / FX * z, z], -1)
        X = (xc - t) @ R
        return np.clip(_pattern(X) * self.gain[i], 0, 255).astype(np.uint8)

    def set_image(self, i, rgb):
        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 97])
        data = enc.tobytes()
        (self.images / f"{self.kids[i].rsplit(':', 1)[-1]}.jpg").write_bytes(data)
        self.records[i]["image_sha1"] = hashlib.sha1(data).hexdigest()

    def set_fill(self, i, mask):
        np.save(self.dense / "work" / "depth" / f"{i:05d}_fill.npy", mask.astype(bool))

    def set_pred(self, i, z):
        np.save(self.dense / "work" / "depth" / f"{i:05d}_pred.npy", np.asarray(z, np.float16))

    def write_align(self):
        from tower.world_builder.dense_pipeline import FILL_RULE

        (self.dense / "align.json").write_text(json.dumps({
            "kind": "depth", "backend": "moge2-vitl", "fill_rule": FILL_RULE,
            "input_digest": "digest-1", "cache_key": "digest-1|moge2-vitl|fill2",
            "camera": {"fx": FX, "fy": FX, "cx": SW / 2, "cy": SH / 2, "width": W, "height": H},
            "records": self.records}))

    def write_surface(self, subdivisions=10):
        from tower.world_builder.surface import SURFACE_FORMAT, SURFACE_SCHEMA_VERSION, write_mesh_bytes

        V, F = _box_mesh(3.0, subdivisions)
        buf = write_mesh_bytes(V, F, np.full((len(V), 3), 128, np.uint8))
        root = self.store.world_dir(WORLD) / "surface" / SESSION
        root.mkdir(parents=True, exist_ok=True)
        name = f"mesh_l0.{time.time_ns():x}.bin"
        (root / name).write_bytes(buf)
        self.surface_built_at = time.time()
        (root / "manifest.json").write_text(json.dumps({
            "format": SURFACE_FORMAT, "schema_version": SURFACE_SCHEMA_VERSION,
            "built_at": self.surface_built_at, "input_digest": "digest-1",
            "params_digest": "p", "params": {"quality": "final"},
            "faces": int(len(F)), "vertices": int(len(V)), "mobile_level": 0,
            "canonical_level": 0,
            "levels": [{"level": 0, "faces": int(len(F)), "vertices": int(len(V)),
                        "bytes": len(buf), "file": name}]}))
        return buf

    def build(self, **kw):
        params = kw.pop("params", None) or A.AppearanceParams(selection_samples=4000)
        kw.setdefault("device", "cpu")
        return AP.build_appearance(self.store, WORLD, SESSION, params=params, **kw)

    def manifest(self):
        return AP.read_appearance_manifest(self.store, WORLD, SESSION)

    def decoded(self, ki, encoding=A.ENC_WEBP):
        man = self.manifest()
        entry = next(k for k in man["keyframes"] if k["ki"] == ki)
        ref = entry["chunks"][encoding]
        data = AP.read_appearance_file(self.store, WORLD, SESSION, "chunk", ref["digest"], man)
        chunk = A.read_chunk(data)
        blob = chunk["slots"][ref["slot"]]
        return A.decode_astc(blob, W, H) if encoding == A.ENC_ASTC else A.decode_webp(blob)


def _box_mesh(half, n):
    """The six faces of a box, each an n x n grid of quads."""
    verts, faces = [], []
    g = np.linspace(-half, half, n + 1)
    for axis in range(3):
        for sign in (-1.0, 1.0):
            base = len(verts)
            a, b = [k for k in range(3) if k != axis]
            for i in range(n + 1):
                for j in range(n + 1):
                    p = [0.0, 0.0, 0.0]
                    p[axis], p[a], p[b] = sign * half, g[i], g[j]
                    verts.append(p)
            for i in range(n):
                for j in range(n):
                    v0 = base + i * (n + 1) + j
                    v1, v2, v3 = v0 + 1, v0 + n + 1, v0 + n + 2
                    faces += [[v0, v1, v2], [v1, v3, v2]]
    return np.array(verts, np.float32), np.array(faces, np.int64)


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


class _FakeRedactor:
    """Available; fills a fixed box on every image, under the current label."""

    available = True
    unavailable_reason = None
    label = TRUSTED
    box = (40, 70, 60, 100)   # y0, y1, x0, x1 in the SESSION image
    fail_on = None
    calls = 0

    def redact(self, data):
        from tower.world_builder.redaction import REDACTION_NONE, RedactionResult

        type(self).calls += 1
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if self.fail_on is not None and self.fail_on(img):
            return RedactionResult(image_bytes=data, label=REDACTION_NONE, regions=0,
                                   unavailable_reason="detector threw")
        y0, y1, x0, x1 = self.box
        img[y0:y1, x0:x1] = 0
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 97])
        return RedactionResult(image_bytes=enc.tobytes(), label=self.label, regions=1)


class _UnavailableRedactor:
    available = False
    unavailable_reason = "no model on this test host"
    label = "none"

    def redact(self, data):  # pragma: no cover - must never be called
        raise AssertionError("an unavailable redactor was asked to redact")


def _never_redact():
    raise AssertionError("a trusted label must not construct a redactor")


def _opaque(rgba):
    return rgba[..., 3] >= 128


def _near_colour(rgb, colour, tol=40):
    return (np.abs(rgb.astype(int) - colour.astype(int)).max(-1) <= tol)


# ---------------------------------------------------------------------------
# a build publishes a whole, readable, self-describing artifact
# ---------------------------------------------------------------------------


class TestABuild:

    def test_it_publishes_a_whole_artifact_that_says_what_it_is(self, world):
        result = world.build(redactor_factory=_never_redact)
        assert result.state == AP.STATE_OK, result.detail
        man = world.manifest()
        assert man["format"] == A.APPEARANCE_FORMAT
        assert man["input_digest"] == "digest-1"
        assert man["camera"] == {"fx": FX, "fy": FX, "cx": SW / 2, "cy": SH / 2,
                                 "width": W, "height": H}
        prov = man["appearance_provenance"]
        assert prov["session_redaction"] == TRUSTED and prov["label_trusted"] is True
        assert prov["redactor_applied_here"] is None
        assert prov["unobserved_rule"] == A.UNOBSERVED_RULE
        assert prov["source"] == "session-keyframes"
        assert prov["frames"] == {"used": N_FRAMES, "refused": {}}
        assert "not anonymised" in prov["note"]
        assert len(man["keyframes"]) == N_FRAMES
        for k in man["keyframes"]:
            assert len(k["id"]) == 16 and k["width"] == W and k["height"] == H
            assert len(k["rotation"]) == 9 and len(k["translation"]) == 3
            assert len(k["gain"]) == 3 and k["mask_origin"] == "stored-fill+nearblack"
            assert A.ENC_WEBP in k["chunks"]
            if k["tier"] == A.TIER_PHONE:
                assert A.ENC_ASTC in k["chunks"] and isinstance(k["rank"], int)
        # the proxy is the surface's phone level, byte for byte
        from tower.world_builder.surface_pipeline import read_surface_level

        proxy = AP.read_appearance_file(world.store, WORLD, SESSION, "proxy",
                                        man["proxy"]["digest"], man)
        assert proxy == read_surface_level(world.store, WORLD, SESSION, 0)
        assert man["proxy"]["source"]["surface_built_at"] == world.surface_built_at
        # every chunk reads, holds whole slots, and its name is its content
        for c in man["chunks"]:
            data = AP.read_appearance_file(world.store, WORLD, SESSION, "chunk", c["digest"], man)
            chunk = A.read_chunk(data)
            assert chunk["encoding"] == c["encoding"] and len(chunk["slots"]) == c["slots"]
            if c["encoding"] == A.ENC_ASTC:
                assert all(len(s) == A.astc_bytes(W, H) for s in chunk["slots"])
        names = {p.name for p in AP.appearance_dir(world.store, WORLD, SESSION).iterdir()}
        assert ".appearance.lock" not in names
        assert not [n for n in names if n.endswith(".tmp")]
        assert json.loads((AP.appearance_dir(world.store, WORLD, SESSION)
                           / "status.json").read_text())["state"] == AP.STATE_OK

    def test_the_published_pixels_are_the_keyframe_undistorted(self, world):
        world.build(redactor_factory=_never_redact)
        for ki in (0, 5):
            rgba = world.decoded(ki)
            src = world.render(ki)[:H, :W]
            op = _opaque(rgba)
            assert op.mean() > 0.9
            err = np.abs(rgba[..., :3].astype(int) - src.astype(int))[op]
            assert np.median(err) <= 3
            astc = world.decoded(ki, A.ENC_ASTC) if A.ENC_ASTC in next(
                k for k in world.manifest()["keyframes"] if k["ki"] == ki)["chunks"] else None
            if astc is not None:
                assert np.median(np.abs(astc[..., :3].astype(int) - src.astype(int))[op]) <= 4

    def test_the_chunk_format_refuses_what_does_not_add_up(self):
        buf = A.pack_chunk(A.ENC_WEBP, 4, 4, [b"abc", b"defgh"])
        assert A.read_chunk(buf)["slots"] == [b"abc", b"defgh"]
        with pytest.raises(ValueError):
            A.read_chunk(buf[:-1])
        with pytest.raises(ValueError):
            A.read_chunk(b"XXXXXXXX" + buf[8:])


# ---------------------------------------------------------------------------
# source provenance (privacy plan 1-5)
# ---------------------------------------------------------------------------


def _paint_forbidden_sources(world):
    """Magenta in every place appearance must never read pixels from."""
    solve_images = world.store.world_dir(WORLD) / "solve" / SESSION / "images"
    solve_images.mkdir(parents=True, exist_ok=True)
    undist = world.dense / "work" / "undist"
    undist.mkdir(parents=True, exist_ok=True)
    captures = world.root / "captures" / "c1" / "frames"
    captures.mkdir(parents=True, exist_ok=True)
    magenta = np.zeros((SH, SW, 3), np.uint8)
    magenta[:] = MAGENTA[::-1]  # BGR on disk
    ok, enc = cv2.imencode(".jpg", magenta, [cv2.IMWRITE_JPEG_QUALITY, 97])
    sources = {}
    for i, kid in enumerate(world.kids):
        seq = kid.rsplit(":", 1)[-1]
        (solve_images / f"{seq}.jpg").write_bytes(enc.tobytes())
        (undist / f"{i:05d}.jpg").write_bytes(enc.tobytes())
        (captures / f"{seq}.jpg").write_bytes(enc.tobytes())
        sources[kid] = str(captures / f"{seq}.jpg")
    (world.store.world_dir(WORLD) / "solve" / SESSION / "sources.json").write_text(
        json.dumps({"sources": sources}))
    world.solution.rgb[:] = MAGENTA
    world._write_solution(world._workspace, world.solution)


def test_appearance_never_reads_an_unredacted_source(world, monkeypatch):
    """Privacy plan 1, 2, 3, 18: a magenta signature in the solve's images, the
    depth stage's undist frames, a raw capture named by sources.json and the
    sparse point colours; none of it reaches a texel, and no such file is
    opened."""
    _paint_forbidden_sources(world)
    opened = []
    real_open, real_io_open, real_imread = builtins.open, io.open, cv2.imread
    real_path_open = pathlib.Path.open

    def spy_open(file, *a, **k):
        opened.append(str(file))
        return real_open(file, *a, **k)

    def spy_io_open(file, *a, **k):
        opened.append(str(file))
        return real_io_open(file, *a, **k)

    def spy_path_open(self, *a, **k):
        opened.append(str(self))
        return real_path_open(self, *a, **k)

    def spy_imread(path, *a, **k):
        opened.append(str(path))
        return real_imread(path, *a, **k)

    monkeypatch.setattr(builtins, "open", spy_open)
    monkeypatch.setattr(io, "open", spy_io_open)
    monkeypatch.setattr(pathlib.Path, "open", spy_path_open)
    monkeypatch.setattr(cv2, "imread", spy_imread)
    result = world.build(redactor_factory=_never_redact)
    monkeypatch.undo()
    assert result.state == AP.STATE_OK, result.detail
    norm = [p.replace("\\", "/") for p in opened]
    forbidden = [p for p in norm if f"/solve/{SESSION}/images" in p or "/undist/" in p
                 or "/captures/" in p or p.endswith("sources.json")]
    assert forbidden == []
    assert any(f"/sessions/{SESSION}/images/" in p for p in norm)
    for k in world.manifest()["keyframes"]:
        rgba = world.decoded(k["ki"])
        assert not (_near_colour(rgba[..., :3], MAGENTA) & _opaque(rgba)).any()


def test_appearance_modules_import_no_solve_workspace():
    """Privacy plan 4: structural, not only behavioural."""
    for module in (A, AP):
        source = pathlib.Path(module.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.Name):
                names.add(node.id)
        assert not names & {"SolveWorkspace", "workspace_for", "read_sources",
                            "prepare_images", "_source_frame", "_undistorted_image"}
        # String constants in CODE (docstrings may name what is forbidden).
        docstrings = {id(node.body[0].value) for node in ast.walk(tree)
                      if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef))
                      and node.body and isinstance(node.body[0], ast.Expr)
                      and isinstance(node.body[0].value, ast.Constant)}
        literals = [node.value for node in ast.walk(tree)
                    if isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and id(node) not in docstrings]
        assert not [s for s in literals if "sources" in s or "undist" in s
                    or "captures" in s or s == "images"]
    # Keyframe pixels are located only through the store's one accessor
    # (`keyframe_image_set`, which honours a re-redaction switch), never the
    # capture's `images_dir` directly, and the directory is opened in exactly
    # one function: the provenance function.
    tree = ast.parse(pathlib.Path(A.__file__).read_text(encoding="utf-8"))

    def users_of(attr):
        return {fn.name for fn in ast.walk(tree) if isinstance(fn, ast.FunctionDef)
                for node in ast.walk(fn) if isinstance(node, ast.Attribute)
                and node.attr == attr}

    assert users_of("images_dir") == set()
    assert users_of("directory") == {"keyframe_source"}
    assert users_of("keyframe_image_set") <= {"keyframe_source", "resolve_label_policy",
                                              "keyframe_set_identity"}
    assert "images_dir" not in pathlib.Path(AP.__file__).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# the label (privacy plan 6-14)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label", [
    "faces-detected-and-filled/yunet-2023mar@0.30",
    "faces-detected-and-filled/yunet-2023mar@0.30+plausibility1",
    TRUSTED,
])
def test_allowlisted_labels_are_trusted_without_rerun(tmp_path, label):
    w = World(tmp_path, label=label)
    result = w.build(redactor_factory=_never_redact)
    assert result.state == AP.STATE_OK, result.detail
    assert w.manifest()["appearance_provenance"]["redaction_effective"] == label


@pytest.mark.parametrize("label", [
    "none", "", "redacted", "faces-detected-and-filled/yunet-2023mar@0.50",
    "faces-detected-and-filled/yunet-2023mar@0.30+plausibility9",
    "faces-detected-and-filled/yunet-2023mar@0.30+plausibility2",
])
def test_untrusted_labels_are_redacted_again_before_any_pixel_is_used(tmp_path, label):
    """6, 9, 10: the fake redactor fills a box; that box is transparent in
    every published frame and the effective label says what ran."""
    w = World(tmp_path, label=label)
    _FakeRedactor.calls = 0
    result = w.build(redactor_factory=_FakeRedactor)
    assert result.state == AP.STATE_OK, result.detail
    assert _FakeRedactor.calls >= N_FRAMES
    man = w.manifest()
    prov = man["appearance_provenance"]
    assert prov["label_trusted"] is False
    assert prov["redactor_applied_here"] == TRUSTED
    assert prov["redaction_effective"] == f"{label or 'none'}&{TRUSTED}"
    y0, y1, x0, x1 = _FakeRedactor.box
    for k in man["keyframes"]:
        assert k["origin"] == A.ORIGIN_REDACTED_HERE
        assert k["mask_origin"].startswith(A.MASK_RERUN)
        rgba = w.decoded(k["ki"])
        assert not _opaque(rgba)[y0:y1, x0:x1].any()


def test_label_none_refuses_the_build_when_no_redactor_can_run(tmp_path):
    """7: refused as a whole, and nothing that looks like textures is written."""
    w = World(tmp_path, label="none")
    result = w.build(redactor_factory=_UnavailableRedactor)
    assert result.state == AP.STATE_UNAVAILABLE
    assert "no face redactor is available" in result.detail
    root = AP.appearance_dir(w.store, WORLD, SESSION)
    assert w.manifest() is None
    assert not list(root.glob("c.*.bin")) and not list(root.glob("p.*.bin"))


def test_a_redaction_failure_on_one_frame_refuses_only_that_frame(tmp_path):
    """8."""
    w = World(tmp_path, label="none")
    target = w.render(3)
    signature = target[5:15, 5:15].astype(int)

    class Failing(_FakeRedactor):
        pass

    def fail_on(img_bgr):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(int)
        return np.abs(rgb[5:15, 5:15] - signature).mean() < 3

    Failing.fail_on = staticmethod(fail_on)
    result = w.build(redactor_factory=Failing)
    assert result.state == AP.STATE_OK, result.detail
    man = w.manifest()
    assert man["appearance_provenance"]["frames"]["refused"] == {A.REFUSED_REDACTION_FAILED: 1}
    assert {e["ki"]: e["reason"] for e in man["excluded"]}[3] == A.REFUSED_REDACTION_FAILED
    assert 3 not in {k["ki"] for k in man["keyframes"]}


def test_an_absent_or_unreadable_session_record_is_untrusted(tmp_path):
    """12."""
    w = World(tmp_path)
    w.store.session_path(WORLD, SESSION).write_text("{ not json")
    assert A.read_session_redaction(w.store, WORLD, SESSION) is None
    assert w.build(redactor_factory=_UnavailableRedactor).state == AP.STATE_UNAVAILABLE


def test_a_purged_world_is_neither_built_nor_served(world):
    """13."""
    from fastapi.testclient import TestClient

    from tower.world_builder.records import World as WorldRec

    assert world.build(redactor_factory=_never_redact).state == AP.STATE_OK
    world.store.write_world(WorldRec(world_id=WORLD, created_at=1.0, updated_at=3.0,
                                     session_ids=(SESSION,), images_purged=True))
    client = TestClient(_app(world.root))
    r = client.get(f"/worlds/{WORLD}/appearance/{SESSION}/manifest")
    assert r.status_code == 404 and "purged" in r.json()["detail"]
    assert world.build(force=True, redactor_factory=_never_redact).state == AP.STATE_UNAVAILABLE


def test_the_dense_stage_label_rule_is_the_gap_this_stage_closes():
    """14, as far as this lane goes: the appearance allowlist refuses what the
    dense stage's `label != "none"` would trust. (The dense stage itself is
    another lane's to change; this pins the difference.)"""
    for label in ("redacted", "faces-detected-and-filled/yunet-2023mar@0.30+plausibility2",
                  "faces-detected-and-filled/yunet-2023mar@0.30+plausibility9"):
        assert bool(label) and label != "none"
        assert not A.label_is_trusted(label)


# ---------------------------------------------------------------------------
# fill, near-black, occluders, transients (privacy plan 15-17, 20)
# ---------------------------------------------------------------------------


def test_fill_pixels_are_transparent_and_carry_no_colour(world):
    """15: the stored fill is alpha 0, its RGB zeroed, and its ring alpha 0."""
    fill = np.zeros((H, W), bool)
    fill[30:60, 50:90] = True
    world.set_fill(2, fill)
    world.write_align()
    world.build(redactor_factory=_never_redact)
    rgba = world.decoded(2)
    assert not _opaque(rgba)[30:60, 50:90].any()
    assert (rgba[35:55, 55:85, :3] <= 3).all()
    ring = A.dilate(fill, A.ALPHA_RING_PX) & ~A.dilate(fill, A.UNOBSERVED_DILATE_PX + 1)
    assert not _opaque(rgba)[ring].any()


def test_a_fill_box_the_stored_mask_missed_is_still_transparent(world):
    """16: a solid black box touching dark scene, with an EMPTY stored mask."""
    img = world.render(4)
    img[20:60, 0:50] = 0          # the box
    img[20:60, 50:58] = (8, 9, 7)  # dark scene touching it
    world.set_image(4, img)
    world.write_align()
    result = world.build(redactor_factory=_never_redact)
    assert result.state == AP.STATE_OK, result.detail
    rgba = world.decoded(4)
    assert not _opaque(rgba)[22:58, 2:48].any()


@pytest.mark.parametrize("problem", ["missing", "wrong-shape", "sha-mismatch"])
def test_a_missing_fill_mask_refuses_the_frame(world, problem):
    """17: never read as "nothing was filled"."""
    path = world.dense / "work" / "depth" / "00001_fill.npy"
    if problem == "missing":
        path.unlink()
    elif problem == "wrong-shape":
        np.save(path, np.zeros((10, 10), bool))
    else:
        world.records[1]["image_sha1"] = "0" * 40
        world.write_align()
    result = world.build(redactor_factory=_never_redact)
    assert result.state == AP.STATE_OK, result.detail
    man = world.manifest()
    assert {e["ki"]: e["reason"] for e in man["excluded"]}[1] == A.REFUSED_NO_FILL_MASK
    assert 1 not in {k["ki"] for k in man["keyframes"]}


def test_a_near_occluder_the_proxy_lacks_is_transparent(world):
    """The wearer's hand in front of the wall: that frame's own depth is far
    nearer than the proxy there."""
    z = world.depth[5][:H, :W].copy()
    z[60:110, 20:80] = z[60:110, 20:80] * 0.35
    world.set_pred(5, z)
    world.build(redactor_factory=_never_redact)
    man = world.manifest()
    entry = next(k for k in man["keyframes"] if k["ki"] == 5)
    assert entry["near_fraction"] > 0.1
    rgba = world.decoded(5)
    assert not _opaque(rgba)[65:105, 25:75].any()
    assert _opaque(world.decoded(4))[65:105, 25:75].all()


def test_a_transient_one_frame_saw_is_transparent_in_that_frame_only(world):
    """A hand ON the wall: no depth difference, but no other keyframe saw it."""
    img = world.render(3)
    img[40:80, 60:110] = (40, 210, 60)
    world.set_image(3, img)
    world.write_align()
    world.build(redactor_factory=_never_redact)
    man = world.manifest()
    entry = next(k for k in man["keyframes"] if k["ki"] == 3)
    assert entry["transient_fraction"] > 0.05
    rgba = world.decoded(3)
    green = _near_colour(rgba[..., :3], np.array([40, 210, 60])) & _opaque(rgba)
    assert green.sum() < 20
    for other in (2, 4):
        assert next(k for k in man["keyframes"] if k["ki"] == other)["transient_fraction"] < 0.02


def test_exposure_gains_recover_a_frame_shot_darker(tmp_path):
    w = World(tmp_path)
    w.gain[6] = 0.6
    w.set_image(6, w.render(6))
    w.write_align()
    w.build(redactor_factory=_never_redact)
    man = w.manifest()
    gains = {k["ki"]: np.mean(k["gain"]) for k in man["keyframes"]}
    others = np.median([g for ki, g in gains.items() if ki != 6])
    assert gains[6] / others == pytest.approx(0.6, abs=0.06)
    assert man["exposure"]["observations"] > 0


def test_the_phone_tier_is_capped_and_the_rest_is_kept_for_the_tower(world):
    params = A.AppearanceParams(selection_samples=4000, phone_budget=3)
    world.build(params=params, redactor_factory=_never_redact)
    man = world.manifest()
    tiers = [k["tier"] for k in man["keyframes"]]
    assert tiers.count(A.TIER_PHONE) <= 3 and tiers.count(A.TIER_TOWER) >= N_FRAMES - 3
    ranks = sorted(k["rank"] for k in man["keyframes"] if k["tier"] == A.TIER_PHONE)
    assert ranks == list(range(len(ranks)))
    for c in man["chunks"]:
        if c["encoding"] == A.ENC_ASTC:
            assert c["tier"] == A.TIER_PHONE
    cov = man["selection"]["coverage"]
    # The box proxy has six walls and the cameras see one, so absolute
    # coverage is small; three keyframes must still keep most of what all see.
    assert cov["all"]["seen1"] > 0
    assert cov["phone"]["seen1"] >= 0.9 * cov["all"]["seen1"]


# ---------------------------------------------------------------------------
# labelling, invalidation, publication (privacy plan 22-26)
# ---------------------------------------------------------------------------


class TestItIsRebuiltExactlyWhenSomethingChanged:

    def test_an_unchanged_world_is_already_built(self, world):
        first = world.build(redactor_factory=_never_redact)
        again = world.build(redactor_factory=_never_redact)
        assert again.detail == AP.ALREADY_BUILT and again.build_id == first.build_id

    def test_a_label_change_invalidates(self, world):
        """23."""
        world.build(redactor_factory=_never_redact)
        before = world.manifest()["params_digest"]
        world.set_label("faces-detected-and-filled/yunet-2023mar@0.30+plausibility1")
        result = world.build(redactor_factory=_never_redact)
        assert result.detail != AP.ALREADY_BUILT
        assert world.manifest()["params_digest"] != before

    def test_a_redactor_upgrade_invalidates_a_rerun_build(self, tmp_path):
        """24."""
        w = World(tmp_path, label="none")
        w.build(redactor_factory=_FakeRedactor)
        before = w.manifest()["params_digest"]

        class Upgraded(_FakeRedactor):
            label = "faces-detected-and-filled/yunet-2023mar@0.30+plausibility4"

        assert w.build(redactor_factory=Upgraded).detail != AP.ALREADY_BUILT
        assert w.manifest()["params_digest"] != before

    def test_a_changed_keyframe_image_invalidates(self, world):
        world.build(redactor_factory=_never_redact)
        before = world.manifest()["appearance_provenance"]["per_frame_sha1_digest"]
        img = world.render(0)
        img[0, 0] = (1, 2, 3)
        world.set_image(0, img)
        world.write_align()
        assert world.build(redactor_factory=_never_redact).detail != AP.ALREADY_BUILT
        assert world.manifest()["appearance_provenance"]["per_frame_sha1_digest"] != before

    def test_a_new_surface_invalidates_and_currency_says_so_until_rebuilt(self, world):
        world.build(redactor_factory=_never_redact)
        man = world.manifest()
        assert AP.appearance_currency(world.store, WORLD, SESSION, man)["current"] is True
        world.write_surface(subdivisions=12)
        cur = AP.appearance_currency(world.store, WORLD, SESSION, man)
        assert cur == {"present": True, "current": False, "reason": "built on an earlier surface"}
        assert world.build(redactor_factory=_never_redact).detail != AP.ALREADY_BUILT
        assert AP.appearance_currency(world.store, WORLD, SESSION, world.manifest())["current"]


def test_a_label_change_during_the_build_aborts_the_publish(world, monkeypatch):
    """26."""
    real = A.encode_webp
    changed = []

    def relabel_then_encode(rgba, quality=A.WEBP_QUALITY):
        if not changed:
            world.set_label("faces-detected-and-filled/yunet-2023mar@0.30+plausibility1")
            changed.append(True)
        return real(rgba, quality)

    monkeypatch.setattr(A, "encode_webp", relabel_then_encode)
    result = world.build(redactor_factory=_never_redact)
    assert result.state == AP.STATE_UNAVAILABLE and "label changed" in result.detail
    assert world.manifest() is None
    assert not list(AP.appearance_dir(world.store, WORLD, SESSION).glob("c.*.bin"))


def test_a_stop_leaves_the_previous_artifact_standing(world):
    assert world.build(redactor_factory=_never_redact).state == AP.STATE_OK
    before = world.manifest()
    world.set_label("faces-detected-and-filled/yunet-2023mar@0.30+plausibility1")
    calls = []

    def stop_in_encode():
        calls.append(1)
        status = json.loads((AP.appearance_dir(world.store, WORLD, SESSION)
                             / "status.json").read_text())
        return status.get("stage") == AP.STAGE_ENCODE

    result = world.build(redactor_factory=_never_redact, should_stop=stop_in_encode)
    assert result.state == AP.STATE_STOPPED
    after = world.manifest()
    assert after["build_id"] == before["build_id"]
    for name, size in AP.named_files(after).items():
        assert (AP.appearance_dir(world.store, WORLD, SESSION) / name).stat().st_size == size


def test_superseded_files_are_pruned_only_after_the_grace(world):
    world.build(redactor_factory=_never_redact)
    old = set(AP.named_files(world.manifest()))
    img = world.render(0)
    img[10:20, 10:20] = (200, 30, 30)
    world.set_image(0, img)
    world.write_align()
    world.build(redactor_factory=_never_redact)
    new = set(AP.named_files(world.manifest()))
    root = AP.appearance_dir(world.store, WORLD, SESSION)
    gone = old - new
    assert gone and all((root / n).exists() for n in gone)
    AP._prune(root, new, older_than_s=0.0)
    assert not any((root / n).exists() for n in gone)
    assert all((root / n).exists() for n in new)


def test_live_then_final(world):
    live = A.AppearanceParams.live(selection_samples=3000)
    assert world.build(params=live, redactor_factory=_never_redact).state == AP.STATE_OK
    assert world.manifest()["quality"] == "live"
    final = A.AppearanceParams(selection_samples=4000)
    result = world.build(params=final, redactor_factory=_never_redact)
    assert result.state == AP.STATE_OK and result.detail != AP.ALREADY_BUILT
    assert world.manifest()["quality"] == "final"


def test_no_surface_is_a_clear_refusal(tmp_path):
    w = World(tmp_path)
    (w.store.world_dir(WORLD) / "surface" / SESSION / "manifest.json").unlink()
    result = w.build(redactor_factory=_never_redact)
    assert result.state == AP.STATE_UNAVAILABLE and "surface" in result.detail


# ---------------------------------------------------------------------------
# routes (privacy plan 25, 27, 30) and the revision
# ---------------------------------------------------------------------------


def _app(root):
    from fastapi import FastAPI

    from tower.routes.geometry import router

    app = FastAPI()
    app.include_router(router)
    app.state.world_root = root
    return app


def _assert_private(response):
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["pragma"] == "no-cache"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "etag" not in response.headers and "last-modified" not in response.headers


class TestTheRoutes:

    def test_they_serve_the_artifact_privately(self, world):
        from fastapi.testclient import TestClient

        world.build(redactor_factory=_never_redact)
        client = TestClient(_app(world.root))
        r = client.get(f"/worlds/{WORLD}/appearance/{SESSION}/manifest")
        assert r.status_code == 200
        _assert_private(r)
        assert r.headers["x-world-redaction"] == TRUSTED
        body = r.json()
        assert body["build_id"] == world.manifest()["build_id"]
        assert body["currency"]["current"] is True
        chunk = body["chunks"][0]
        c = client.get(f"/worlds/{WORLD}/appearance/{SESSION}/chunk/{chunk['digest']}")
        assert c.status_code == 200 and len(c.content) == chunk["bytes"]
        assert c.headers["content-type"] == "application/octet-stream"
        _assert_private(c)
        p = client.get(f"/worlds/{WORLD}/appearance/{SESSION}/proxy/{body['proxy']['digest']}")
        assert p.status_code == 200 and p.content[:8] == b"WBSURF01"
        _assert_private(p)

    def test_bodies_are_compressed_only_when_accepted_and_stay_private(self, world):
        """gzip/deflate when the client accepts it (18.2 MB -> ~14.1 MB on the
        canonical world); the privacy headers do not change with the encoding."""
        import gzip
        import zlib

        from fastapi.testclient import TestClient

        world.build(redactor_factory=_never_redact)
        client = TestClient(_app(world.root))
        man = world.manifest()
        base = f"/worlds/{WORLD}/appearance/{SESSION}"
        urls = (f"{base}/manifest", f"{base}/chunk/{man['chunks'][0]['digest']}",
                f"{base}/proxy/{man['proxy']['digest']}")
        for url in urls:
            plain = client.get(url, headers={"Accept-Encoding": "identity"})
            assert plain.status_code == 200, url
            assert "content-encoding" not in plain.headers, url
            assert plain.headers["vary"] == "Accept-Encoding"
            _assert_private(plain)
            raw = plain.content
            for accept, coding, decode in (("gzip", "gzip", gzip.decompress),
                                           ("br, gzip;q=0.5, deflate", "gzip", gzip.decompress),
                                           ("deflate", "deflate", zlib.decompress),
                                           ("gzip;q=0, deflate;q=0.1", "deflate", zlib.decompress),
                                           ("*", "gzip", gzip.decompress)):
                # stream=True-free: read the wire bytes undecoded.
                with client.stream("GET", url, headers={"Accept-Encoding": accept}) as r:
                    wire = b"".join(r.iter_raw())
                    assert r.status_code == 200, (url, accept)
                    assert r.headers["content-encoding"] == coding, (url, accept)
                    assert r.headers["vary"] == "Accept-Encoding"
                    assert int(r.headers["content-length"]) == len(wire)
                    _assert_private(r)
                    assert r.headers["x-world-redaction"] == TRUSTED
                assert decode(wire) == raw, (url, accept)
            for refused in ("gzip;q=0", "br", "identity, *;q=0"):
                r = client.get(url, headers={"Accept-Encoding": refused})
                assert "content-encoding" not in r.headers, (url, refused)
                assert r.content == raw
        # A 404 is a sentence, never compressed, and still private.
        with client.stream("GET", f"{base}/chunk/{'0' * 32}",
                           headers={"Accept-Encoding": "gzip"}) as r:
            assert r.status_code == 404 and "content-encoding" not in r.headers
            _assert_private(r)

    def test_the_encoding_negotiation(self):
        from tower.results.world_builder_appearance import encode_body, negotiate_encoding

        assert negotiate_encoding(None) is None and negotiate_encoding("") is None
        assert negotiate_encoding("gzip, deflate, br") == "gzip"  # URLSession's default
        assert negotiate_encoding("deflate, gzip") == "gzip"
        assert negotiate_encoding("GZIP;Q=1.0") == "gzip"
        assert negotiate_encoding("gzip;q=0, deflate") == "deflate"
        assert negotiate_encoding("gzip;q=bogus") is None
        assert negotiate_encoding("*;q=0.2") == "gzip"
        assert negotiate_encoding("*, gzip;q=0") == "deflate"
        assert negotiate_encoding("identity") is None
        small, headers = encode_body(b"x" * 100, "gzip")
        assert small == b"x" * 100 and "Content-Encoding" not in headers
        data = bytes(range(256)) * 64
        body, headers = encode_body(data, "gzip")
        again, _ = encode_body(data, "gzip")
        assert headers["Content-Encoding"] == "gzip" and body == again, "deterministic"

    def test_anything_not_named_is_404_and_still_private(self, world):
        from fastapi.testclient import TestClient

        world.build(redactor_factory=_never_redact)
        client = TestClient(_app(world.root))
        man = world.manifest()
        proxy = man["proxy"]["digest"]
        for url in (f"/worlds/{WORLD}/appearance/{SESSION}/chunk/{'0' * 32}",
                    f"/worlds/{WORLD}/appearance/{SESSION}/chunk/{proxy}",
                    f"/worlds/{WORLD}/appearance/{SESSION}/chunk/..%5Cmanifest.json",
                    f"/worlds/{WORLD}/appearance/{SESSION}/chunk/{'A' * 32}",
                    f"/worlds/{WORLD}/appearance/nope/manifest",
                    f"/worlds/nope/appearance/{SESSION}/manifest"):
            r = client.get(url)
            assert r.status_code == 404, url
            _assert_private(r)

    def test_a_stale_label_is_not_served_and_the_revision_goes_null(self, world):
        """25."""
        from fastapi.testclient import TestClient

        world.build(redactor_factory=_never_redact)
        client = TestClient(_app(world.root))
        rev = client.get(f"/worlds/{WORLD}/render/revision", params={"session_id": SESSION,
                                                                      "viewer": "appearance-1"})
        assert rev.status_code == 200
        appearance = rev.json()["appearance"]
        assert appearance["revision"] == f"{SESSION}/appearance:{world.manifest()['build_id']}"
        page_revision = rev.json()["revision"]
        world.set_label("faces-detected-and-filled/yunet-2023mar@0.30+plausibility1")
        r = client.get(f"/worlds/{WORLD}/appearance/{SESSION}/manifest")
        assert r.status_code == 404 and r.json()["detail"] == AP.STALE_LABEL_DETAIL
        chunk = world.manifest()["chunks"][0]["digest"]
        assert client.get(f"/worlds/{WORLD}/appearance/{SESSION}/chunk/{chunk}").status_code == 404
        rev2 = client.get(f"/worlds/{WORLD}/render/revision", params={"session_id": SESSION,
                                                                       "viewer": "appearance-1"})
        assert rev2.json()["appearance"]["revision"] is None
        # The appearance page is no longer served: the rung steps down to the
        # surface, and that (not an appearance build) moves the page revision.
        assert rev.json()["representation"] == "appearance"
        assert rev2.json()["representation"] == "surface"
        assert rev2.json()["revision"] != page_revision

    def test_a_rebuild_moves_only_the_appearance_revision(self, world):
        from fastapi.testclient import TestClient

        world.build(redactor_factory=_never_redact)
        client = TestClient(_app(world.root))
        first = client.get(f"/worlds/{WORLD}/render/revision", params={"session_id": SESSION}).json()
        world.build(force=True, redactor_factory=_never_redact)
        second = client.get(f"/worlds/{WORLD}/render/revision", params={"session_id": SESSION}).json()
        assert second["appearance"]["revision"] != first["appearance"]["revision"]
        assert second["revision"] == first["revision"]

    def test_the_manifest_carries_no_paths_or_sequence_numbers(self, world):
        """30."""
        from fastapi.testclient import TestClient

        world.build(redactor_factory=_never_redact)
        body = TestClient(_app(world.root)).get(
            f"/worlds/{WORLD}/appearance/{SESSION}/manifest").json()
        strings = []

        def walk(value):
            if isinstance(value, dict):
                for k, v in value.items():
                    strings.append(k)
                    walk(v)
            elif isinstance(value, list):
                for v in value:
                    walk(v)
            elif isinstance(value, str):
                strings.append(value)

        walk(body)
        seqs = [kid.rsplit(":", 1)[-1] for kid in world.kids]
        for text in strings:
            assert f"{SESSION}:" not in text
            assert not any(seq in text for seq in seqs), text
            assert "\\" not in text and ".jpg" not in text and "/images" not in text, text


# ---------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------


def test_the_product_builds_appearance_with_the_surface():
    from tower.config import Settings
    from tower.main import _world_build_spec

    def settings(**kw):
        return Settings(host="0.0.0.0", port=8000, dev_mode=True, cv_experiment="baseline",
                        cv_device="cpu", world_root="C:/w", world_autobuild=True, **kw)

    assert "--appearance" in _world_build_spec(settings()).argv
    assert "--appearance" not in _world_build_spec(settings(world_appearance=False)).argv
    assert "--appearance" not in _world_build_spec(settings(world_surface=False)).argv


def test_the_live_surface_child_builds_the_appearance_when_asked(tmp_path):
    from scripts.world_build_session import BackgroundSurface

    spawned = []

    class Proc:
        pid = 1

        def poll(self):
            return None

    def spawn(argv, **kw):
        spawned.append(argv)
        return Proc()

    s = BackgroundSurface(root=tmp_path, world_id=WORLD, session_id=SESSION, spawn=spawn,
                          appearance=True)
    assert s.maybe_launch(None)
    assert "--appearance" in spawned[0] and "--live" in spawned[0]
    s._child = None
    plain = BackgroundSurface(root=tmp_path, world_id=WORLD, session_id=SESSION, spawn=spawn)
    assert plain.maybe_launch(None)
    assert "--appearance" not in spawned[1]
    plain._child = None


class TestTheFinalChain:
    """`world_build_session.final_surface_stages`: what runs after Stop, run with
    the three stages replaced by recorders (the stages themselves are tested in
    their own files)."""

    @pytest.fixture
    def calls(self, monkeypatch):
        from tower.world_builder import appearance_pipeline, dense_pipeline, surface_pipeline
        from tower.world_builder.appearance_pipeline import AppearanceResult
        from tower.world_builder.surface import SurfaceResult

        log = {"order": [], "surface_state": "ok", "appearance_state": "ok", "stop_in": None,
               "stop": False}

        def trip(stage):
            if log["stop_in"] == stage:
                log["stop"] = True

        def surfacify(store, world_id, session_id, *, params=None, force=False,
                      should_stop=None, **kw):
            log["order"].append("surface")
            log["surface_params"], log["surface_force"] = params, force
            log["surface_stop"] = should_stop
            trip("surface")
            state = "stopped" if should_stop() else log["surface_state"]
            return SurfaceResult(state=state)

        def build_appearance(store, world_id, session_id, *, params=None, should_stop=None, **kw):
            log["order"].append("appearance")
            log["appearance_params"], log["appearance_stop"] = params, should_stop
            trip("appearance")
            return AppearanceResult(state="stopped" if should_stop() else log["appearance_state"])

        def prune(directory):
            log["order"].append("prune")
            return 123

        monkeypatch.setattr(surface_pipeline, "surfacify", surfacify)
        monkeypatch.setattr(appearance_pipeline, "build_appearance", build_appearance)
        monkeypatch.setattr(dense_pipeline, "prune_intermediates", prune)
        return log

    def _run(self, tmp_path, log, **kw):
        from scripts.world_build_session import final_surface_stages
        from tower.world_builder.store import WorldStore

        args = dict(solved=True, appearance=True, prune_depth_work=True,
                    should_stop=lambda: log["stop"], stop_source=lambda: "stdin")
        args.update(kw)
        return final_surface_stages(WorldStore(tmp_path), WORLD, SESSION, **args)

    def test_surface_then_appearance_then_prune_with_the_final_presets(self, tmp_path, calls):
        report = self._run(tmp_path, calls)
        assert calls["order"] == ["surface", "appearance", "prune"]
        sp, ap = calls["surface_params"], calls["appearance_params"]
        # Union masks at Stop (OneFormer alone during the walk), final consistency.
        assert sp.transient_detector == "union" and ap.transient_detector == "union"
        assert sp.quality != "live" and sp.depth_consistency
        assert (sp.consistency_outer, sp.consistency_warm_outer) == (5, 2)
        assert calls["surface_force"] is True
        assert calls["surface_stop"]() is False and calls["appearance_stop"]() is False
        assert report["surface"]["attempted"] and report["surface"]["depth_work_pruned_bytes"] == 123
        assert report["appearance"]["attempted"] and report["appearance"]["state"] == "ok"

    def test_the_live_child_uses_oneformer_and_the_warm_live_iterations(self):
        from tower.world_builder.appearance import AppearanceParams
        from tower.world_builder.surface import SurfaceParams

        live = SurfaceParams.live()
        assert live.transient_detector == "oneformer" and live.depth_consistency
        assert (live.consistency_outer, live.consistency_warm_outer) == (3, 1)
        assert AppearanceParams.live().transient_detector == "oneformer"

    def test_a_hard_stop_before_it_attempts_nothing(self, tmp_path, calls):
        calls["stop"] = True
        report = self._run(tmp_path, calls)
        assert calls["order"] == []
        assert report == {"surface": {"attempted": False,
                                      "reason": "hard stop (stdin) during finalization"}}

    def test_a_hard_stop_during_the_surface_skips_the_rest(self, tmp_path, calls):
        calls["stop_in"] = "surface"
        report = self._run(tmp_path, calls)
        assert calls["order"] == ["surface"]
        assert report["surface"]["state"] == "stopped" and "appearance" not in report

    def test_a_hard_stop_during_the_appearance_keeps_the_depth_work(self, tmp_path, calls):
        """The stopped appearance is rebuilt by the next run, which needs the
        per-frame depth work; pruning it after a hard stop made that rebuild
        refuse every frame."""
        calls["stop_in"] = "appearance"
        report = self._run(tmp_path, calls)
        assert calls["order"] == ["surface", "appearance"]
        assert report["appearance"]["state"] == "stopped"
        assert "depth_work_pruned_bytes" not in report["surface"]

    def test_a_stopped_appearance_keeps_the_depth_work_even_if_the_stop_is_not_sticky(
            self, tmp_path, calls):
        """The prune decision must not depend on `should_stop` still answering
        True after the appearance returned `stopped`."""
        calls["appearance_state"] = "stopped"
        report = self._run(tmp_path, calls)
        assert calls["order"] == ["surface", "appearance"]
        assert report["appearance"]["state"] == "stopped"
        assert "depth_work_pruned_bytes" not in report["surface"]

    def test_a_surface_that_did_not_build_gets_no_appearance_and_no_prune(self, tmp_path, calls):
        calls["surface_state"] = "failed"
        report = self._run(tmp_path, calls)
        assert calls["order"] == ["surface"] and "appearance" not in report

    def test_a_failed_appearance_still_prunes(self, tmp_path, calls):
        calls["appearance_state"] = "unavailable"
        self._run(tmp_path, calls)
        assert calls["order"] == ["surface", "appearance", "prune"]

    def test_no_solve_no_surface(self, tmp_path, calls):
        report = self._run(tmp_path, calls, solved=False)
        assert calls["order"] == [] and report["surface"]["attempted"] is False

    def test_without_appearance_or_with_densify(self, tmp_path, calls):
        self._run(tmp_path, calls, appearance=False)
        assert calls["order"] == ["surface", "prune"]
        calls["order"].clear()
        self._run(tmp_path, calls, prune_depth_work=False)
        assert calls["order"] == ["surface", "appearance"]

    def test_main_runs_it_after_the_final_solve_and_before_the_dense_stage(self):
        import inspect

        import scripts.world_build_session as B

        src = inspect.getsource(B.main)
        assert src.index("solver.run_final(") < src.index("final_surface_stages(") < src.index(
            "densify(")
        assert "should_stop=stop_request.hard_asked_for" in src[src.index("final_surface_stages("):]


def test_the_cli_inspects_what_it_built(world, capsys):
    import scripts.world_appearance as CLI

    world.build(redactor_factory=_never_redact)
    assert CLI.main(["--root", str(world.root), "--world", WORLD, "--session", SESSION,
                     "--inspect"]) == 0
    out = capsys.readouterr().out
    assert '"present": true' in out and '"current": true' in out
