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

        # The trust decision the depth stage records (review 1, M3): the stored
        # bytes under a trusted label, a re-redaction by the current rule else.
        label = self.store.keyframe_image_set(WORLD, SESSION).redaction
        trust = A.pixel_trust_token(label, None if A.label_is_trusted(label) else TRUSTED)
        (self.dense / "align.json").write_text(json.dumps({
            "kind": "depth", "backend": "moge2-vitl", "fill_rule": FILL_RULE,
            "redaction": label, "redaction_trust": trust,
            "input_digest": "digest-1", "cache_key": "digest-1|moge2-vitl|fill2",
            "camera": {"fx": FX, "fy": FX, "cx": SW / 2, "cy": SH / 2, "width": W, "height": H},
            "records": self.records}))

    def write_surface(self, subdivisions=10, confidence=None):
        from tower.world_builder.surface import (
            CONFIDENCE_FORMAT,
            CONFIDENCE_VERSION,
            SURFACE_FORMAT,
            SURFACE_SCHEMA_VERSION,
            write_confidence_bytes,
            write_mesh_bytes,
        )

        V, F = _box_mesh(3.0, subdivisions)
        buf = write_mesh_bytes(V, F, np.full((len(V), 3), 128, np.uint8))
        root = self.store.world_dir(WORLD) / "surface" / SESSION
        root.mkdir(parents=True, exist_ok=True)
        name = f"mesh_l0.{time.time_ns():x}.bin"
        (root / name).write_bytes(buf)
        # A surface built before the channel existed has none, which is the
        # default here; `confidence` writes one the way the surface stage does.
        self.surface_confidence = None
        conf_record = None
        if confidence is not None:
            self.surface_confidence = np.asarray(confidence, np.uint8)
            cbuf = write_confidence_bytes(self.surface_confidence)
            cname = f"conf_l0.{time.time_ns():x}.bin"
            (root / cname).write_bytes(cbuf)
            conf_record = {"file": cname, "bytes": len(cbuf),
                           "vertices": int(len(V)), "format": CONFIDENCE_FORMAT,
                           "version": CONFIDENCE_VERSION}
        self.surface_built_at = time.time()
        (root / "manifest.json").write_text(json.dumps({
            "format": SURFACE_FORMAT, "schema_version": SURFACE_SCHEMA_VERSION,
            "built_at": self.surface_built_at, "input_digest": "digest-1",
            "params_digest": "p", "params": {"quality": "final"},
            "faces": int(len(F)), "vertices": int(len(V)), "mobile_level": 0,
            "canonical_level": 0,
            "levels": [{"level": 0, "faces": int(len(F)), "vertices": int(len(V)),
                        "bytes": len(buf), "file": name,
                        **({"confidence": conf_record} if conf_record else {})}]}))
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
        # the proxy is the surface's phone level, byte for byte -- except its
        # vertex colours: they are the depth stage's pixels, the page never
        # draws them, and the route must not carry them (review 1, M2). This
        # world's surface has no confidence channel, so they are all zero,
        # which a page must read as "no confidence known" (APPEARANCE 4.2a).
        from tower.world_builder.surface import read_mesh_bytes
        from tower.world_builder.surface_pipeline import read_surface_level

        proxy = AP.read_appearance_file(world.store, WORLD, SESSION, "proxy",
                                        man["proxy"]["digest"], man)
        surface = read_surface_level(world.store, WORLD, SESSION, 0)
        assert len(proxy) == len(surface)
        pV, pF, pC, _pN = read_mesh_bytes(proxy)
        sV, sF, sC, _sN = read_mesh_bytes(surface)
        assert np.array_equal(pV, sV) and np.array_equal(pF, sF)
        assert sC.any() and not pC.any()
        assert proxy == AP.proxy_with_confidence(surface, None)
        assert man["proxy"]["confidence"] == {
            "present": False, "version": AP.PROXY_CONFIDENCE_VERSION,
            "channel": AP.PROXY_CONFIDENCE_CHANNEL, "source_level": 0}
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


class _StubDepth:
    """A depth network that answers without a model."""

    name = "appearance-test-stub"
    licence = "test"
    windowed = False
    window_size = 0
    kind = "depth"

    def predict(self, rgb):
        return np.full(rgb.shape[:2], 2.0, np.float32)


@pytest.fixture
def depth_stub(monkeypatch):
    """The depth stage runnable on the synthetic world: a stub network, and the
    fake redactor wherever the stage constructs `FaceRedactor`."""
    from tower.world_builder import redaction as R
    from tower.world_builder.dense import register_backend

    register_backend(_StubDepth.name, _StubDepth)
    monkeypatch.setattr(R, "FaceRedactor", _FakeRedactor)
    _FakeRedactor.calls = 0
    return _StubDepth.name


UNGATED = "faces-detected-and-filled/yunet-2023mar@0.30"
UNTRUSTED_LABELS = ("faces-detected-and-filled/yunet-2023mar@0.30+plausibility2",
                    "faces-detected-and-filled/yunet-2023mar@0.50", "redacted", "none")


@pytest.mark.parametrize("label", UNTRUSTED_LABELS + (UNGATED, TRUSTED))
def test_the_depth_stage_trusts_exactly_the_labels_the_appearance_stage_trusts(
        tmp_path, depth_stub, label):
    """14 (review 1, M2): the depth stage ran `label != "none"` and read the
    stored bytes of `+plausibility2`, `@0.50` or any string with no redactor
    call, while the appearance stage re-redacted the same session. Now both go
    through `appearance.label_is_trusted`, and the stage records the decision."""
    from tower.world_builder.dense import DenseParams
    from tower.world_builder.dense_pipeline import run_depth_stage

    w = World(tmp_path, label=label)
    trusted = A.label_is_trusted(label)
    payload = run_depth_stage(w.store, WORLD, SESSION, w.solution, w.intrinsics,
                              DenseParams(backend=depth_stub), w.dense)
    origins = {k: v for k, v in payload["image_origins"].items()}
    if trusted:
        assert origins == {"world-keyframe": N_FRAMES}
        assert _FakeRedactor.calls == 0
    else:
        assert origins == {"world-keyframe-redacted-here": N_FRAMES}
        assert _FakeRedactor.calls >= N_FRAMES
    assert payload["redaction_trust"] == A.pixel_trust_token(label, None if trusted else TRUSTED)
    stored = {kid: hashlib.sha1((w.images / f"{kid.rsplit(':', 1)[-1]}.jpg").read_bytes()).hexdigest()
              for kid in w.kids}
    for record in payload["records"]:
        assert (record["image_sha1"] == stored[w.kids[record["ki"]]]) is trusted


@pytest.mark.parametrize("label", UNTRUSTED_LABELS[:3])
def test_the_dense_read_refuses_a_callers_trust_the_allowlist_does_not_give(tmp_path, label):
    """The same rule inside `keyframe_image_bytes` itself (review 1, repro r4):
    a caller that still says "redacted" for a label off the allowlist gets the
    bytes redacted again, never the stored bytes as `world-keyframe`."""
    from tower.world_builder.dense_pipeline import keyframe_image_bytes

    w = World(tmp_path, label=label)
    image_set = w.store.keyframe_image_set(WORLD, SESSION)
    stored = (w.images / f"{w.kids[0].rsplit(':', 1)[-1]}.jpg").read_bytes()
    _FakeRedactor.calls = 0
    data, origin, _fill = keyframe_image_bytes(
        w.store, WORLD, SESSION, w.kids[0], None, _FakeRedactor(),
        keyframes_are_redacted=True, image_set=image_set)
    assert origin == "world-keyframe-redacted-here" and _FakeRedactor.calls == 1
    assert data != stored
    trusted = World(tmp_path / "t", label=TRUSTED)
    data, origin, _fill = keyframe_image_bytes(
        trusted.store, WORLD, SESSION, trusted.kids[0], None, _UnavailableRedactor(),
        keyframes_are_redacted=True, image_set=trusted.store.keyframe_image_set(WORLD, SESSION))
    assert origin == "world-keyframe"


def test_a_walk_time_depth_stage_is_not_reused_after_stop(tmp_path, depth_stub):
    """Review 1, M3 (repro r2, first half). During a walk the label is `none`,
    the depth stage re-redacts, and `align.json` records the RE-REDACTED bytes'
    SHA-1. After Stop the label is trusted and the solve digest is often the
    same; the depth cache used to be reused, so the trusted final appearance
    found no record matching the stored bytes and refused every frame the
    re-redaction had changed. The trust decision is now part of the cache."""
    from tower.world_builder import surface_pipeline as SP
    from tower.world_builder.dense import DenseParams
    from tower.world_builder.dense_pipeline import depth_trust_now

    w = World(tmp_path, label="none")
    (w.dense / "align.json").unlink()
    dparams = DenseParams(backend=depth_stub)
    walk, work = SP.ensure_depth_stage(w.store, WORLD, SESSION, w.solution, w.intrinsics,
                                       gate_rel=dparams.gate_rel, backend=depth_stub)
    assert walk["redaction_trust"] == f"rerun:none&{TRUSTED}"
    # This synthetic solve has no sparse points, so no fit succeeds; mark them
    # good and put their maps on disk, so the trust decision is the only thing
    # between this cache and reuse.
    for r in walk["records"]:
        r["ok"] = True
        np.save(work / "depth" / f"{r['ki']:05d}.npy", np.ones((H, W), np.float32))
    (w.dense / "align.json").write_text(json.dumps(walk))
    walk_trust = depth_trust_now(w.store, WORLD, SESSION)
    assert SP._depth_cache_usable(walk, w.dense, w.solution, dparams, trust=walk_trust)

    w.set_label(TRUSTED)                                   # Stop writes the real label
    final_trust = depth_trust_now(w.store, WORLD, SESSION)
    assert final_trust == f"trusted:{TRUSTED}"
    assert not SP._depth_cache_usable(walk, w.dense, w.solution, dparams, trust=final_trust)
    final, _work = SP.ensure_depth_stage(w.store, WORLD, SESSION, w.solution, w.intrinsics,
                                         gate_rel=dparams.gate_rel, backend=depth_stub)
    assert final["redaction_trust"] == final_trust
    assert final["cache_key"].endswith(f"|trust:{final_trust}")
    stored = {i: hashlib.sha1((w.images / f"{kid.rsplit(':', 1)[-1]}.jpg").read_bytes()).hexdigest()
              for i, kid in enumerate(w.kids)}
    assert {r["ki"]: r["image_sha1"] for r in final["records"]} == stored
    # And the reverse (a relabel to an untrusted label) is not reused either.
    w.set_label("faces-detected-and-filled/yunet-2023mar@0.30+plausibility2")
    assert not SP._depth_cache_usable(final, w.dense, w.solution, dparams,
                                      trust=depth_trust_now(w.store, WORLD, SESSION))


def test_a_depth_cache_from_before_the_trust_token_is_kept_only_if_still_trusted():
    """No canonical world refits for the new key alone: a record that used the
    stored bytes of a label still on the allowlist reads as that trust; one
    that trusted `+plausibility2` (the old rule) does not."""
    from tower.world_builder.dense import DenseParams
    from tower.world_builder.dense_pipeline import _depth_cache_key, depth_cache_matches, recorded_trust

    p = DenseParams()
    old = {"cache_key": _depth_cache_key("d", p), "redaction": UNGATED,
           "keyframes_were_redacted_at_capture": True}
    assert recorded_trust(old) == f"trusted:{UNGATED}"
    assert depth_cache_matches(old, "d", p, None, f"trusted:{UNGATED}")
    weak = {**old, "redaction": UNTRUSTED_LABELS[0]}
    assert recorded_trust(weak) is None
    assert not depth_cache_matches(weak, "d", p, None, f"trusted:{UNGATED}")
    rerun = {**old, "redaction": "none", "keyframes_were_redacted_at_capture": False}
    assert recorded_trust(rerun) is None


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


# ---------------------------------------------------------------------------
# the cross-frame redaction consensus (contract §5.3b)
# ---------------------------------------------------------------------------

SIGNATURE = np.array([250, 60, 10], np.uint8)   # "the face", by colour
WALL_Z = 3.0                                    # the box face every camera sees


def _wall_patch(half, n=41):
    """A square of the far wall, as world points."""
    g = np.linspace(-half, half, n)
    x, y = np.meshgrid(g, g)
    return np.stack([x.ravel(), y.ravel(), np.full(x.size, WALL_Z)], -1)


def _project_patch(world, i, pts, shape):
    """That square, as a boolean mask in keyframe `i`."""
    R, t = world.pose(i)
    pc = np.asarray(pts, float) @ R.T + t
    u = FX * pc[:, 0] / pc[:, 2] + SW / 2
    v = FX * pc[:, 1] / pc[:, 2] + SH / 2
    m = np.zeros(shape, bool)
    ui = np.round(u).astype(int)
    vi = np.round(v).astype(int)
    ok = (ui >= 0) & (vi >= 0) & (ui < shape[1]) & (vi < shape[0])
    assert ok.mean() > 0.99, f"the patch leaves keyframe {i}"
    m[vi[ok], ui[ok]] = True
    return cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)


def _a_face_on_the_wall(world, hidden_in=(3,), half=0.10, fill_half=0.15):
    """Paint one patch of the wall in every keyframe, and fill it -- as the
    redactor would -- in `hidden_in` only. That is the leak exactly: a face the
    detector found in one frame and missed in the others."""
    sig = _wall_patch(half)
    box = _wall_patch(fill_half)
    for i in range(N_FRAMES):
        img = world.render(i)
        img[_project_patch(world, i, sig, (SH, SW))] = SIGNATURE
        world.set_image(i, img)
    for i in hidden_in:
        img = cv2.imdecode(np.frombuffer(
            (world.images / f"{world.kids[i].rsplit(':', 1)[-1]}.jpg").read_bytes(),
            np.uint8), cv2.IMREAD_COLOR)[..., ::-1].copy()
        img[_project_patch(world, i, box, (SH, SW))] = 0
        world.set_image(i, img)
        world.set_fill(i, _project_patch(world, i, box, (H, W)))
    world.write_align()
    return sig


def _signature_texels(world, man):
    """Where the published appearance still shows the painted face."""
    out = {}
    for entry in man["keyframes"]:
        rgba = world.decoded(entry["ki"])
        hit = _near_colour(rgba[..., :3], SIGNATURE, tol=60) & _opaque(rgba)
        out[entry["ki"]] = int(hit.sum())
    return out


def test_without_the_consensus_a_face_one_frame_hid_is_published_by_the_others(world):
    """The leak, as measured on the canonical world: the control for the tests
    below. Every keyframe but one publishes what that one redacted."""
    _a_face_on_the_wall(world)
    result = world.build(redactor_factory=_never_redact,
                         params=A.AppearanceParams(selection_samples=4000,
                                                   redaction_consensus=A.CONSENSUS_OFF))
    assert result.state == AP.STATE_OK, result.detail
    man = world.manifest()
    assert man["appearance_provenance"]["redaction_consensus"]["mode"] == A.CONSENSUS_OFF
    seen = _signature_texels(world, man)
    assert seen[3] == 0, "the frame that hid it must not publish it"
    assert sum(1 for ki, n in seen.items() if ki != 3 and n > 100) >= 5, seen


def test_a_surface_one_keyframe_redacted_is_published_by_no_keyframe(world):
    """§5.3b: the fill is projected onto the proxy and read back in every
    keyframe, so a face one detector pass found is unobserved for all of them.
    Fails with `redaction_consensus=off`, which is the default since the
    campaign deferred privacy preprocessing (the test above)."""
    _a_face_on_the_wall(world)
    result = world.build(redactor_factory=_never_redact, params=A.AppearanceParams(selection_samples=4000,
                                          redaction_consensus=A.CONSENSUS_PLAUSIBLE))
    assert result.state == AP.STATE_OK, result.detail
    man = world.manifest()
    seen = _signature_texels(world, man)
    assert max(seen.values()) == 0, seen
    rec = man["appearance_provenance"]["redaction_consensus"]
    assert rec["mode"] == A.CONSENSUS_PLAUSIBLE
    assert rec["voxels_dilated"] >= rec["voxels_marked"] > 0
    assert rec["frames_masked"] >= N_FRAMES - 1
    entry = next(k for k in man["keyframes"] if k["ki"] == 4)
    assert entry["consensus_fraction"] > 0


def test_the_consensus_covers_that_surface_and_not_the_room(world):
    """The cost side: it takes the patch and its tolerance, not the wall. The
    tolerance is about two voxels -- one dilation cell and the quantisation --
    or 12 px at the median proxy depth, so the mask is roughly twice the
    eroded fill across."""
    _a_face_on_the_wall(world)
    world.build(redactor_factory=_never_redact, params=A.AppearanceParams(selection_samples=4000,
                                          redaction_consensus=A.CONSENSUS_PLAUSIBLE))
    man = world.manifest()
    for entry in man["keyframes"]:
        assert (entry["consensus_fraction"] or 0.0) < 0.20, entry["ki"]
        # and the published alpha carries the 7 px ring every mask carries
        assert _opaque(world.decoded(entry["ki"])).mean() > 0.70, entry["ki"]


def test_plausible_leaves_a_wall_sized_false_positive_to_the_other_frames(world):
    """What `plausible` does NOT do, in code: a fill region too large to be a
    face is not propagated, so the room is still drawn from the frames that saw
    it. `union` propagates it and the room goes dark -- the trade PRIVLEAK.md
    measures at 93% of the canonical world's published texels."""
    big = np.zeros((H, W), bool)
    big[10:100, 10:140] = True          # 61% of the frame
    world.set_fill(3, big)
    world.write_align()

    world.build(redactor_factory=_never_redact, params=A.AppearanceParams(selection_samples=4000,
                                          redaction_consensus=A.CONSENSUS_PLAUSIBLE))
    plausible = world.manifest()
    assert (plausible["appearance_provenance"]["redaction_consensus"]
            ["regions_refused"]) == {"larger than a face": 1}
    assert all((k["consensus_fraction"] or 0.0) == 0.0 for k in plausible["keyframes"])

    world.build(redactor_factory=_never_redact, force=True,
                params=A.AppearanceParams(selection_samples=4000,
                                          redaction_consensus=A.CONSENSUS_UNION))
    union = world.manifest()
    assert max(k["consensus_fraction"] or 0.0 for k in union["keyframes"]) > 0.3


def test_the_consensus_rule_is_in_the_cache_key_and_the_epoch(world):
    """A change to the rule rebuilds, and an open page drops what it drew under
    the old one: those textures may hold what the new rule hides."""
    _a_face_on_the_wall(world)
    # Explicitly under the rule: `off` is the default while privacy
    # preprocessing is deferred, and `off` ignores the rule's own parameters.
    on = A.AppearanceParams(selection_samples=4000,
                            redaction_consensus=A.CONSENSUS_PLAUSIBLE)
    assert world.build(redactor_factory=_never_redact, params=on).state == AP.STATE_OK
    man = world.manifest()
    assert world.build(redactor_factory=_never_redact, params=on).detail == AP.ALREADY_BUILT

    params = A.AppearanceParams(selection_samples=4000, consensus_erode=0.05,
                                redaction_consensus=A.CONSENSUS_PLAUSIBLE)
    assert world.build(redactor_factory=_never_redact, params=params).detail != AP.ALREADY_BUILT
    third = world.manifest()
    assert third["params_digest"] != man["params_digest"]
    assert third["epoch"] != man["epoch"]

    prov = man["appearance_provenance"]
    assert not AP.textures_carry_over(prov, third["appearance_provenance"])
    assert AP.textures_carry_over(prov, prov)
    # the revision route asks the question without naming a rule (§9)
    assert AP.textures_carry_over(prov, {k: prov[k] for k in (
        "session_redaction", "keyframe_image_set", "label_trusted")})


def test_an_unknown_consensus_mode_is_refused():
    with pytest.raises(ValueError):
        A.AppearanceParams(redaction_consensus="sometimes")


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


def _exposure_inputs(gains, slopes, vignette, n=12):
    """Keyframes of the box room in the SOLVE camera, recorded through a known
    photometric model: albedo x gain x exp(slope . (xn, yn) + vignette(r2))."""
    K = np.array([[FX, 0, SW / 2], [0, FX, SH / 2], [0, 0, 1.0]])
    rgbs, zps, Rs, ts = [], [], [], []
    for i in range(n):
        x = -0.9 + 1.8 * i / (n - 1)
        eye = np.array([x, 0.3 * ((i % 3) - 1), 0.4 * ((i % 2) - 0.5)])
        target = np.array([0.2 * x, 0.25 * ((i % 4) - 1.5), 3.0])
        R, t = _look_from(eye, target)
        z = _render_box_depth(R, t, K, W, H)
        vv, uu = np.mgrid[0:H, 0:W].astype(float)
        xc = np.stack([(uu + 0.5 - K[0, 2]) / FX * z, (vv + 0.5 - K[1, 2]) / FX * z, z], -1)
        X = (xc - t) @ R
        field = A.exposure_field(slopes[i], vignette, W, H)
        img = _pattern(X) * np.asarray(gains[i], float)[None, None] * field[..., None]
        rgbs.append(np.clip(img, 0, 255).astype(np.uint8))
        zps.append(z.astype(np.float32))
        Rs.append(R)
        ts.append(t)
    return rgbs, [np.ones((H, W), bool)] * n, zps, Rs, ts, K


def test_the_exposure_field_is_written_in_the_contracts_coordinates():
    """§5.4: xn, yn in [-1, 1] from the image centre, r2 = 1 at the corners."""
    f = A.exposure_field((0.2, -0.1), (-0.3, -0.05), 200, 100)
    assert f.shape == (100, 200) and f.dtype == np.float32
    xn, yn, r2 = A.exposure_coordinates(199.5, 0.5, 200, 100)
    assert f[0, 199] == pytest.approx(np.exp(0.2 * xn - 0.1 * yn - 0.3 * r2 - 0.05 * r2 * r2), rel=1e-5)
    assert A.exposure_coordinates(100.0, 50.0, 200, 100) == (0.0, 0.0, 0.0)
    assert A.exposure_coordinates(200.0, 100.0, 200, 100) == (1.0, 1.0, 1.0)
    # a positive x slope records the right of the image brighter
    assert f[50, 190] > f[50, 10]
    assert A.exposure_field(None, None, 8, 4) == pytest.approx(np.ones((4, 8)))


def test_the_spatial_exposure_model_recovers_a_lens_falloff_and_a_tilt():
    n = 12
    gains = [[1.0, 1.0, 1.0]] * n
    gains[5] = [0.7, 0.7, 0.7]
    slopes = [(0.0, 0.0)] * n
    slopes[3] = (0.3, 0.0)
    inputs = _exposure_inputs(gains, slopes, (-0.35, 0.0), n)
    # Unregularised, the joint solve recovers the model the frames were recorded through.
    params = A.AppearanceParams(exposure_grid_px=6, exposure_border_px=4, exposure_slope_ridge=0.0,
                                exposure_vignette_ridge=0.0)
    g, obs, rec = A.solve_gains(*inputs, params, device="cpu")
    assert rec["model"] == A.EXPOSURE_MODEL_SPATIAL and (obs > 0).all()
    k1, k2 = rec["vignette"]
    assert k1 + k2 == pytest.approx(-0.35, abs=0.03)         # the corner falloff
    assert k1 == pytest.approx(-0.35, abs=0.05)
    sl = rec["slopes"]
    assert sl[3, 0] == pytest.approx(0.3, abs=0.03) and sl[3, 1] == pytest.approx(0.0, abs=0.03)
    assert np.abs(np.delete(sl, 3, axis=0)).max() < 0.03
    ratio = g[5].mean() / np.median(np.delete(g.mean(1), 5))
    assert ratio == pytest.approx(0.7, abs=0.02)
    # With the default ridges (a keyframe that saw little is held flat), the
    # same answers, shrunk: this synthetic walk faces one wall, so a tilt and
    # the falloff are only weakly told apart.
    g1, _obs1, rec1 = A.solve_gains(*inputs, A.AppearanceParams(exposure_grid_px=6, exposure_border_px=4),
                                    device="cpu")
    assert -0.35 - 0.05 < sum(rec1["vignette"]) < -0.15
    assert rec1["slopes"][3, 0] - np.median(np.delete(rec1["slopes"][:, 0], 3)) == pytest.approx(0.3, abs=0.08)

    g0, _obs0, rec0 = A.solve_gains(*inputs, A.AppearanceParams(
        exposure_grid_px=6, exposure_border_px=4, exposure_model=A.EXPOSURE_MODEL_GAIN), device="cpu")
    assert "slopes" not in rec0 and "vignette" not in rec0
    assert rec["abs_log_residual_after"] < 0.5 * rec0["abs_log_residual_after"]


def test_an_unknown_exposure_model_is_refused():
    with pytest.raises(ValueError):
        A.solve_gains([np.zeros((4, 4, 3), np.uint8)], [np.ones((4, 4), bool)], [np.ones((4, 4))],
                      [np.eye(3)], [np.zeros(3)], np.eye(3),
                      A.AppearanceParams(exposure_model="gain+magic"), device="cpu")


def test_the_manifest_carries_the_photometric_model(world):
    world.build(redactor_factory=_never_redact)
    man = world.manifest()
    ex = man["exposure"]
    assert ex["model"] == A.EXPOSURE_MODEL_SPATIAL
    assert len(ex["vignette"]) == 2 and ex["coordinates"] == A.EXPOSURE_COORDINATES
    assert "slopes" not in ex                                  # they travel on the keyframes
    for k in man["keyframes"]:
        assert len(k["gain_slope"]) == 2 and all(np.isfinite(k["gain_slope"]))
    assert man["params"]["exposure_model"] == A.EXPOSURE_MODEL_SPATIAL


def test_mask_edges_fade_inward_and_the_ring_is_untouched():
    """§5.6: alpha 0 over the core dilated by the ring, then a smooth rise over
    the feather; nothing that the hard rule made transparent gains alpha."""
    rgb = np.full((60, 80, 3), 120, np.uint8)
    core = np.zeros((60, 80), bool)
    core[20:30, 30:40] = True
    hard = A.rgba_for(rgb, core)
    soft = A.rgba_for(rgb, core, feather_px=8)
    ring = A.dilate(core, A.ALPHA_RING_PX)
    assert (hard[..., 3][ring] == 0).all() and (hard[..., 3][~ring] == 255).all()
    assert (soft[..., 3][ring] == 0).all()
    assert (soft[..., :3][core] == 0).all() and (soft[..., :3][~core] == 120).all()
    assert (soft[..., 3] <= hard[..., 3]).all()
    far = ~A.dilate(core, A.ALPHA_RING_PX + 10)
    assert (soft[..., 3][far] == 255).all()
    row = soft[25, 40 + A.ALPHA_RING_PX:, 3].astype(int)       # walking away from the patch
    assert (np.diff(row) >= 0).all() and 0 < row[3] < 255
    assert A.rgba_for(rgb, np.zeros((60, 80), bool), feather_px=8)[..., 3].min() == 255


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
# review 1, B1: the label transition at Stop, epochs, and what a page is told
# ---------------------------------------------------------------------------


def _revision(client):
    return client.get(f"/worlds/{WORLD}/render/revision",
                      params={"session_id": SESSION, "viewer": "appearance-1"}).json()


def _records_carry(world, redactor=None):
    """What the depth stage records: the SHA-1 of the bytes it read -- the
    re-redacted ones under an untrusted label, the stored ones under a trusted."""
    for i, kid in enumerate(world.kids):
        data = (world.images / f"{kid.rsplit(':', 1)[-1]}.jpg").read_bytes()
        if redactor is not None:
            data = redactor.redact(data).image_bytes
        world.records[i]["image_sha1"] = hashlib.sha1(data).hexdigest()
    world.write_align()


class TestTheStopTransition:
    """Every ordinary Stop turned the label from `none` to the real one; the
    revision answered `appearance: null` with nothing else, the open page dropped
    its textures for good, and the final build came back under the constant page
    revision the dead page was stamped with, so nothing ever replaced it."""

    def _walk(self, tmp_path):
        from fastapi.testclient import TestClient

        w = World(tmp_path, label="none")
        _records_carry(w, _FakeRedactor())
        live = w.build(params=A.AppearanceParams.live(selection_samples=3000,
                                                       transient_detector="off"),
                       redactor_factory=_FakeRedactor)
        assert live.state == AP.STATE_OK, live.detail
        return w, TestClient(_app(w.root))

    def test_stop_reports_rebuilding_and_the_final_build_keeps_the_epoch(self, tmp_path):
        w, client = self._walk(tmp_path)
        walking = _revision(client)
        assert walking["representation"] == "appearance"
        assert walking["appearance"]["state"] == AP.SERVED
        epoch = walking["appearance"]["epoch"]
        assert epoch == w.manifest()["epoch"] == w.manifest()["build_id"]

        w.set_label(TRUSTED)                                   # Stop
        gap = _revision(client)
        assert gap["appearance"]["revision"] is None
        assert gap["appearance"]["state"] == AP.REBUILDING, "not a withdrawal: keep drawing"
        assert gap["representation"] == "surface"
        # still not SERVED to anyone during the gap: the label check stands
        r = client.get(f"/worlds/{WORLD}/appearance/{SESSION}/manifest")
        assert r.status_code == 404 and r.json()["detail"] == AP.STALE_LABEL_DETAIL

        _records_carry(w)                                      # the final depth stage
        final = w.build(params=A.AppearanceParams(selection_samples=4000,
                                                  transient_detector="off"),
                        redactor_factory=_never_redact)
        assert final.state == AP.STATE_OK, final.detail
        done = _revision(client)
        assert done["appearance"]["state"] == AP.SERVED
        assert done["appearance"]["revision"] != walking["appearance"]["revision"]
        assert done["appearance"]["epoch"] == epoch, "the final build replaces the walk's in place"
        assert done["revision"] == walking["revision"], "so the page is not reloaded"
        assert w.manifest()["appearance_provenance"]["label_trusted"] is True

    def test_a_relabel_withdraws_and_its_rebuild_moves_the_page_revision(self, world):
        from fastapi.testclient import TestClient

        world.build(redactor_factory=_never_redact)
        client = TestClient(_app(world.root))
        before = _revision(client)
        assert before["appearance"]["state"] == AP.SERVED
        world.set_label("faces-detected-and-filled/yunet-2023mar@0.30+plausibility1")
        gap = _revision(client)
        assert gap["appearance"] == {"revision": None, "current": False,
                                     "state": AP.WITHDRAWN, "epoch": None}
        world.build(redactor_factory=_never_redact)
        after = _revision(client)
        assert after["representation"] == "appearance"
        assert after["appearance"]["epoch"] != before["appearance"]["epoch"]
        assert after["revision"] != before["revision"], (
            "a page that dropped its textures must see a new page revision when they return")

    def test_a_label_change_and_rebuild_inside_one_poll_still_changes_the_epoch(self, world):
        """m7: the page never sees the gap, so the epoch is what tells it to drop
        the old textures before drawing the new build."""
        world.build(redactor_factory=_never_redact)
        first = world.manifest()["epoch"]
        world.set_label("faces-detected-and-filled/yunet-2023mar@0.30+plausibility2")
        assert world.build(redactor_factory=_FakeRedactor).state == AP.STATE_OK
        assert world.manifest()["epoch"] != first

    def test_purged_and_absent_are_told_apart(self, world, tmp_path):
        from fastapi.testclient import TestClient

        empty = World(tmp_path / "empty")
        assert _revision(TestClient(_app(empty.root)))["appearance"]["state"] == AP.ABSENT
        world.build(redactor_factory=_never_redact)
        record = world.store.read_world(WORLD)
        world.store.write_world(type(record)(**{**record.__dict__, "images_purged": True}))
        assert _revision(TestClient(_app(world.root)))["appearance"]["state"] == AP.WITHDRAWN

    @pytest.mark.parametrize("previous,current,carries", [
        # the walk's re-redacted build -> the final build under the real label
        ({"label_trusted": False, "session_redaction": None, "redactor_applied_here": TRUSTED,
          "keyframe_image_set": None},
         {"label_trusted": True, "session_redaction": TRUSTED, "keyframe_image_set": None}, True),
        # a re-redaction by a redactor that is not itself trusted
        ({"label_trusted": False, "session_redaction": None, "redactor_applied_here": "x",
          "keyframe_image_set": None},
         {"label_trusted": True, "session_redaction": TRUSTED, "keyframe_image_set": None}, False),
        # stored bytes under one trusted label -> another label
        ({"label_trusted": True, "session_redaction": TRUSTED, "keyframe_image_set": None},
         {"label_trusted": True, "session_redaction": UNGATED, "keyframe_image_set": None}, False),
        ({"label_trusted": True, "session_redaction": TRUSTED, "keyframe_image_set": None},
         {"label_trusted": False, "session_redaction": "none", "keyframe_image_set": None}, False),
        # the same label and set
        ({"label_trusted": True, "session_redaction": TRUSTED, "keyframe_image_set": None},
         {"label_trusted": True, "session_redaction": TRUSTED, "keyframe_image_set": None}, True),
        # a re-redaction switch or revert changes the pixels
        ({"label_trusted": True, "session_redaction": TRUSTED, "keyframe_image_set": "images.redacted-p3@1"},
         {"label_trusted": True, "session_redaction": TRUSTED, "keyframe_image_set": None}, False),
        (None, {"label_trusted": True, "session_redaction": TRUSTED}, False),
    ])
    def test_which_label_changes_carry_textures_over(self, previous, current, carries):
        assert AP.textures_carry_over(previous, current) is carries


class TestAPageLoadingTheLastBuild:
    """Review 1, M4 (repro r5): the chunks a page is loading 404ed the moment the
    next build published, because the route served only the current manifest's
    digests although the files were kept for the prune grace."""

    def _two_builds(self, world):
        from fastapi.testclient import TestClient

        p = A.AppearanceParams(selection_samples=4000, transient_detector="off")
        assert world.build(params=p, redactor_factory=_never_redact).state == AP.STATE_OK
        old = world.manifest()
        img = world.render(3)
        img[10:30, 10:30] = (200, 40, 40)
        world.set_image(3, img)
        world.write_align()
        assert world.build(params=p, redactor_factory=_never_redact).state == AP.STATE_OK
        new = world.manifest()
        gone = [c["digest"] for c in old["chunks"]
                if c["digest"] not in {n["digest"] for n in new["chunks"]}]
        assert gone, "the rebuild replaced some chunks"
        return TestClient(_app(world.root)), gone

    def test_a_superseded_chunk_is_served_through_the_grace(self, world):
        client, gone = self._two_builds(world)
        for digest in gone:
            r = client.get(f"/worlds/{WORLD}/appearance/{SESSION}/chunk/{digest}")
            assert r.status_code == 200, r.text
            _assert_private(r)
            assert AP.content_digest(r.content) == digest

    def test_not_after_the_grace(self, world):
        client, gone = self._two_builds(world)
        root = AP.appearance_dir(world.store, WORLD, SESSION)
        doc = json.loads((root / AP.SUPERSEDED_NAME).read_text())
        for entry in doc["entries"]:
            entry["at"] -= AP.PRUNE_GRACE_S + 1
        (root / AP.SUPERSEDED_NAME).write_text(json.dumps(doc))
        for digest in gone:
            r = client.get(f"/worlds/{WORLD}/appearance/{SESSION}/chunk/{digest}")
            assert r.status_code == 404 and r.json()["detail"] == "no such appearance file"

    def test_not_under_another_label_and_not_when_the_label_is_stale(self, world):
        client, gone = self._two_builds(world)
        root = AP.appearance_dir(world.store, WORLD, SESSION)
        doc = json.loads((root / AP.SUPERSEDED_NAME).read_text())
        for entry in doc["entries"]:
            entry["session_redaction"] = None               # built under `none`
        (root / AP.SUPERSEDED_NAME).write_text(json.dumps(doc))
        assert all(client.get(f"/worlds/{WORLD}/appearance/{SESSION}/chunk/{d}").status_code == 404
                   for d in gone)
        world.set_label("faces-detected-and-filled/yunet-2023mar@0.30+plausibility1")
        doc2 = json.loads((root / AP.SUPERSEDED_NAME).read_text())
        assert doc2  # still on disk; the label check comes first
        current = world.manifest()["chunks"][0]["digest"]
        assert client.get(f"/worlds/{WORLD}/appearance/{SESSION}/chunk/{current}").status_code == 404


def test_the_world_listing_reports_the_appearance_as_imagery(world):
    """29 (review 1, m6): `GET /worlds` names the appearance artifact, its label,
    privacy tags and retention, whether it is served, and never a path, a URL,
    or a claim that it is anonymised or privacy-safe."""
    from fastapi.testclient import TestClient

    client = TestClient(_app(world.root))

    def summary():
        body = client.get("/worlds").json()
        (w,) = [x for x in body["worlds"] if x["world_id"] == WORLD]
        (s,) = w["sessions"]
        return s["appearance"]

    assert summary() is None
    world.build(redactor_factory=_never_redact)
    a = summary()
    assert a["state"] == AP.SERVED and a["redaction"] == TRUSTED
    assert a["redaction_effective"] == TRUSTED and a["label_trusted"] is True
    assert a["keyframes"] == N_FRAMES and a["bytes"] > 0
    assert "retention" in a and "purged" in a["retention"]
    text = json.dumps(a).lower()
    assert "not anonymised" in text
    assert "anonymised" not in text.replace("not anonymised", "")
    assert "privacy-safe" not in text and "privacy safe" not in text
    assert "\\\\" not in text and "/appearance/" not in text.replace("appearance/<session>/", "")
    world.set_label("faces-detected-and-filled/yunet-2023mar@0.30+plausibility2")
    assert summary()["state"] == AP.WITHDRAWN


def test_a_same_size_corrupt_file_is_rewritten_and_is_not_already_built(world):
    """m10: `_write_once` and `_already_built` trusted the name and size."""
    assert world.build(redactor_factory=_never_redact).state == AP.STATE_OK
    man = world.manifest()
    root = AP.appearance_dir(world.store, WORLD, SESSION)
    name = next(iter(AP.named_files(man)))
    good = (root / name).read_bytes()
    (root / name).write_bytes(bytes(len(good)))                # same size, wrong content
    assert AP._already_built(root, man["params_digest"], False) is None
    AP._write_once(root / name, good)
    assert (root / name).read_bytes() == good
    assert AP._already_built(root, man["params_digest"], False) is not None


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

        # The recorder the builder hands in is `engine.mark_stage`; here it is
        # a list, so what reaches the session record can be asserted without a
        # store. `record` is optional and defaults to a no-op, so every caller
        # that predates it -- including `world_finalize.py` -- is unchanged.
        log.setdefault("recorded", [])

        def record(stage, *, state, detail=None, attempted=True):
            log["recorded"].append((stage, state, attempted, detail))

        args = dict(solved=True, appearance=True, prune_depth_work=True,
                    should_stop=lambda: log["stop"], stop_source=lambda: "stdin",
                    record=record)
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

    @pytest.mark.parametrize("state", ["unavailable", "failed"])
    def test_an_appearance_that_did_not_build_keeps_the_depth_work(self, tmp_path, calls, state):
        """Review 1, m2: an `unavailable` appearance (a live child's lock not yet
        released, no redactor) is rebuilt later and needs the per-frame work."""
        calls["appearance_state"] = state
        report = self._run(tmp_path, calls)
        assert calls["order"] == ["surface", "appearance"]
        assert "depth_work_pruned_bytes" not in report["surface"]

    def test_no_solve_no_surface(self, tmp_path, calls):
        report = self._run(tmp_path, calls, solved=False)
        assert calls["order"] == [] and report["surface"]["attempted"] is False

    def test_without_appearance_or_with_densify(self, tmp_path, calls):
        self._run(tmp_path, calls, appearance=False)
        assert calls["order"] == ["surface", "prune"]
        calls["order"].clear()
        self._run(tmp_path, calls, prune_depth_work=False)
        assert calls["order"] == ["surface", "appearance"]

    # -- what the session record is told ------------------------------
    #
    # The builder marks finalization `complete` and releases the world lock in
    # its `finally`, which is BEFORE any of this runs, and that is deliberate:
    # the phone reads the world while the surface builds. What was not
    # deliberate is that nothing on disk then said whether the photographic
    # stages ran, were skipped, or raised -- their only record was a report
    # dict on the child stdout that nobody persists. These pin the second
    # record. They do not move a single call.

    def test_each_stage_is_recorded_running_and_then_with_its_outcome(
            self, tmp_path, calls):
        self._run(tmp_path, calls)
        assert calls["recorded"] == [
            ("surface", "running", True, None),
            ("surface", "ok", True, None),
            ("appearance", "running", True, None),
            ("appearance", "ok", True, None),
        ]

    def test_a_surface_that_raises_is_recorded_as_failed_and_still_raises(
            self, tmp_path, calls, monkeypatch):
        """THE DEFECT THIS EXISTS FOR. `surfacify` raising left a world that
        was sparse forever beside a record saying `complete`, with the reason
        only in a traceback on a stdout nobody keeps. The exception must still
        leave -- the exit code is how the supervisor learns -- but
        the failure is on the record first."""
        from tower.world_builder import surface_pipeline

        def boom(*a, **kw):
            calls["order"].append("surface")
            raise RuntimeError("the depth network fell over")

        monkeypatch.setattr(surface_pipeline, "surfacify", boom)
        with pytest.raises(RuntimeError, match="depth network"):
            self._run(tmp_path, calls)
        assert calls["recorded"][0] == ("surface", "running", True, None)
        stage, state, attempted, detail = calls["recorded"][1]
        assert (stage, state, attempted) == ("surface", "failed", True)
        assert detail == "RuntimeError: the depth network fell over"
        # And the appearance that never got its turn says why, rather than
        # being absent and indistinguishable from an older Tower.
        assert calls["recorded"][2][:3] == ("appearance", "unavailable", False)

    def test_an_appearance_that_raises_is_recorded_as_failed_and_still_raises(
            self, tmp_path, calls, monkeypatch):
        from tower.world_builder import appearance_pipeline

        def boom(*a, **kw):
            raise ValueError("no redactor")

        monkeypatch.setattr(appearance_pipeline, "build_appearance", boom)
        with pytest.raises(ValueError, match="no redactor"):
            self._run(tmp_path, calls)
        assert calls["recorded"][1] == ("surface", "ok", True, None)
        assert calls["recorded"][3] == (
            "appearance", "failed", True, "ValueError: no redactor")

    def test_a_surface_that_did_not_build_records_why_the_appearance_did_not(
            self, tmp_path, calls):
        calls["surface_state"] = "failed"
        self._run(tmp_path, calls)
        assert calls["recorded"][1][:3] == ("surface", "failed", True)
        assert calls["recorded"][2][:3] == ("appearance", "unavailable", False)

    def test_a_hard_stop_records_both_stages_as_stopped_and_unattempted(
            self, tmp_path, calls):
        calls["stop"] = True
        self._run(tmp_path, calls)
        assert [r[:3] for r in calls["recorded"]] == [
            ("surface", "stopped", False), ("appearance", "stopped", False)]
        assert all("hard stop (stdin)" in r[3] for r in calls["recorded"])

    def test_no_solve_records_both_stages_as_unavailable(self, tmp_path, calls):
        self._run(tmp_path, calls, solved=False)
        assert [r[:3] for r in calls["recorded"]] == [
            ("surface", "unavailable", False), ("appearance", "unavailable", False)]

    def test_an_unwanted_appearance_is_not_recorded_at_all(self, tmp_path, calls):
        """`--appearance` off: `main` records that separately, once, rather
        than this function claiming an outcome for a stage it was not given."""
        self._run(tmp_path, calls, appearance=False)
        assert [r[0] for r in calls["recorded"]] == ["surface", "surface"]

    def test_the_recorder_is_optional(self, tmp_path, calls):
        """Every existing caller passes no recorder and must be unchanged."""
        report = self._run(tmp_path, calls, record=None)
        assert calls["order"] == ["surface", "appearance", "prune"]
        assert report["appearance"]["state"] == "ok"

    def test_main_records_the_stages_through_the_engine(self):
        """The builder wires `engine.mark_stage` in, records the dense stage
        itself, and records the stages it was never asked for -- so "never
        attempted" is on disk rather than inferred from an absent key."""
        import inspect

        import scripts.world_build_session as B

        src = inspect.getsource(B.main)
        assert "engine.mark_stage(" in src
        assert "record=record_stage" in src
        # After the finally block that releases the lock, never inside it.
        assert src.index("engine.release_world(") < src.index("record=record_stage")
        for stage in ("STAGE_SURFACE", "STAGE_APPEARANCE", "STAGE_DENSE"):
            assert stage in src

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
