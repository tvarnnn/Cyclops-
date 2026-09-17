"""The transient detector's masks: the wearer's hands, arms and held phone.

Contracts: WORLD-BUILDER-APPEARANCE.md §5.3a, WORLD-BUILDER-SURFACE.md §2,
WORLD-BUILDER-DENSE.md §3a. Module: tower/world_builder/transients.py.

No model is loaded here. Every test that needs a detector passes a STUB
backend that marks rectangles it is told about (and, to prove what it was
shown, records the pixels it received); the suite's conftest switches the real
detector off by environment so nothing downloads by accident.

Runs on the appearance tests' synthetic world: eight cameras on one textured
wall, a surface artifact that is the box, the depth stage's records and maps.
"""

from __future__ import annotations

import ast
import builtins
import io
import logging
import pathlib

import cv2
import numpy as np
import pytest

from tests.test_world_builder_appearance import (
    MAGENTA,
    N_FRAMES,
    SESSION,
    WORLD,
    H,
    W,
    World,
    _FakeRedactor,
    _near_colour,
    _never_redact,
    _opaque,
    _paint_forbidden_sources,
)

from tower.world_builder import appearance as A
from tower.world_builder import appearance_pipeline as AP
from tower.world_builder import transients as T

HAND = (60, 110, 40, 100)      # y0, y1, x0, x1 in the solve camera
HAND_KI = 5


class Stub:
    """A backend factory whose backends mark given rectangles as hand/phone."""

    def __init__(self, hands=None, phones=None, reason=None, run_error=None):
        self.hands = hands if hands is not None else {HAND_KI: HAND}
        self.phones = phones or {}
        self.reason = reason
        self.run_error = run_error
        self.calls = []        # (component, ki)
        self.seen = {}         # (component, ki) -> rgb it was shown

    def __call__(self, component):
        stub = self

        class Backend(T.ComponentBackend):
            def probe(self):
                return stub.reason

            def run(self, items, params, emit, should_stop=None):
                if stub.run_error is not None:
                    raise stub.run_error
                for ki, rgb, unobserved in items:
                    stub.calls.append((component, ki))
                    stub.seen[(component, ki)] = rgb.copy()
                    hand = np.zeros(rgb.shape[:2], bool)
                    phone = np.zeros(rgb.shape[:2], bool)
                    if ki in stub.hands:
                        y0, y1, x0, x1 = stub.hands[ki]
                        hand[y0:y1, x0:x1] = True
                    if ki in stub.phones:
                        y0, y1, x0, x1 = stub.phones[ki]
                        phone[y0:y1, x0:x1] = True
                    emit(ki, hand, phone, 0.001)
                return {"frames": len(items)}

        Backend.component = component
        return Backend()

    def computed(self, component=None):
        return [ki for c, ki in self.calls if component in (None, c)]


def _ensure(world, stub, params=None, frames=None, policy=None, redactor_factory=_never_redact):
    from tower.world_builder.global_solve import load_solution

    solution = load_solution(world.store, WORLD, SESSION)
    records = {r["ki"]: r for r in world.records}
    frames = frames if frames is not None else list(enumerate(world.kids))
    return T.ensure_transient_masks(
        world.store, WORLD, SESSION, frames, intrinsics=world.intrinsics,
        camera=solution.camera, align_records=records,
        depth_dir=world.dense / "work" / "depth", params=params or T.TransientParams(),
        policy=policy, redactor_factory=redactor_factory, backend_factory=stub)


def _rect_mask(rect, shape=(H, W)):
    m = np.zeros(shape, bool)
    y0, y1, x0, x1 = rect
    m[y0:y1, x0:x1] = True
    return m


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


# ---------------------------------------------------------------------------
# composition
# ---------------------------------------------------------------------------


def test_a_phone_counts_only_when_it_touches_a_hand():
    params = T.TransientParams(dilate_px=0)
    hand = _rect_mask((50, 70, 50, 70))
    held = _rect_mask((72, 90, 50, 70))          # 2 px below the hand
    resting = _rect_mask((10, 30, 120, 150))     # far away on the desk
    out = T.compose([(hand, held | resting)], params, (H, W))
    assert out[80, 60] and out[60, 60]
    assert not out[20, 130]
    # no hand at all: no phone either
    assert not T.compose([(np.zeros((H, W), bool), held)], params, (H, W)).any()


def test_the_union_is_dilated_by_an_ellipse_scaled_with_width():
    params = T.TransientParams()
    assert T.dilation_radius(params, 359) == 12
    assert T.dilation_radius(params, 718) == 24
    dot = np.zeros((200, 359), bool)
    dot[100, 180] = True
    other = np.zeros((200, 359), bool)
    other[20, 20] = True
    out = T.compose([(dot, np.zeros_like(dot)), (other, np.zeros_like(dot))], params, (200, 359))
    assert out[100, 192] and not out[100, 193]          # 12 px along an axis
    assert not out[112, 192]                             # an ellipse, not a square
    assert out[20, 20] and out[30, 20]                   # the union of both components


def test_instance_masks_mostly_redaction_fill_or_most_of_the_image_are_dropped():
    params = T.TransientParams()
    fill = _rect_mask((0, 40, 0, 40))
    in_fill = _rect_mask((2, 38, 2, 38))
    huge = np.ones((H, W), bool)
    real = _rect_mask((60, 90, 60, 90))
    phone = _rect_mask((60, 90, 95, 110))
    hand, ph = T.filter_instance_masks([in_fill, huge, real, phone, real],
                                       ["hand", "arm", "sleeve", "mobile phone", "hand"],
                                       [0.9, 0.9, 0.9, 0.9, 0.1], fill, params)
    assert not hand[20, 20] and hand[70, 70] and hand.sum() == real.sum()
    assert ph.sum() == phone.sum()


def test_the_rule_id_names_the_mode_and_moves_with_every_parameter():
    union, one = T.TransientParams(), T.TransientParams.live()
    assert union.mode == T.MODE_UNION and one.mode == T.MODE_ONEFORMER
    assert "gdino" in union.rule_id() and "oneformer" in union.rule_id()
    assert "gdino" not in one.rule_id() and "oneformer" in one.rule_id()
    for change in (dict(dilate_px=6), dict(phone_near_px=8), dict(hand_score=0.4),
                   dict(oneformer_threshold=0.6), dict(fill_drop_frac=0.5)):
        assert T.TransientParams(**change).rule_id() != union.rule_id(), change
    with pytest.raises(ValueError):
        T.TransientParams(mode="everything")


# ---------------------------------------------------------------------------
# the cache: keying and reuse
# ---------------------------------------------------------------------------


class TestTheCache:

    def test_masks_are_cached_per_keyframe_and_component_and_reused(self, world):
        stub = Stub()
        first = _ensure(world, stub)
        assert first.state == T.STATE_OK
        assert first.computed == N_FRAMES and first.cached == 0
        assert sorted(stub.computed(T.COMPONENT_GDSAM)) == list(range(N_FRAMES))
        assert sorted(stub.computed(T.COMPONENT_ONEFORMER)) == list(range(N_FRAMES))
        depth = world.dense / "work" / "depth"
        for ki in range(N_FRAMES):
            for c in (T.COMPONENT_GDSAM, T.COMPONENT_ONEFORMER):
                assert (depth / f"{ki:05d}_transient.{c}.npz").exists()
        assert first.mask(HAND_KI)[80, 70] and not first.mask(4).any()

        again = Stub()
        second = _ensure(world, again)
        assert again.calls == [] and second.computed == 0 and second.cached == N_FRAMES
        np.testing.assert_array_equal(second.mask(HAND_KI), first.mask(HAND_KI))
        assert second.frames_digest() == first.frames_digest()

    def test_only_new_or_changed_keyframes_are_computed(self, world):
        """The live increment: the keyframes of the previous build are cached."""
        stub = Stub()
        _ensure(world, stub, frames=list(enumerate(world.kids))[:5])
        assert sorted(stub.computed(T.COMPONENT_ONEFORMER)) == [0, 1, 2, 3, 4]
        more = Stub()
        _ensure(world, more)
        assert sorted(more.computed(T.COMPONENT_ONEFORMER)) == [5, 6, 7]
        # a keyframe whose stored image changed is a new keyframe
        img = world.render(2)
        img[0, 0] = (1, 2, 3)
        world.set_image(2, img)
        world.write_align()
        changed = Stub()
        report = _ensure(world, changed)
        assert changed.computed(T.COMPONENT_ONEFORMER) == [2]
        assert report.computed == 1 and report.cached == N_FRAMES - 1

    def test_the_key_is_image_rule_models_and_label(self, world, monkeypatch):
        _ensure(world, Stub())
        # a component parameter recomputes that component only
        s = Stub()
        _ensure(world, s, params=T.TransientParams(oneformer_threshold=0.6))
        assert len(s.computed(T.COMPONENT_ONEFORMER)) == N_FRAMES
        assert s.computed(T.COMPONENT_GDSAM) == []
        # a composition parameter recomputes nothing, and still changes the mask
        s = Stub()
        wide = _ensure(world, s, params=T.TransientParams(oneformer_threshold=0.6, dilate_px=20))
        assert s.calls == [] and wide.mask(HAND_KI).sum() > 0
        # the cheaper mode reuses the union's OneFormer masks
        s = Stub()
        _ensure(world, s, params=T.TransientParams.live(oneformer_threshold=0.6))
        assert s.calls == []
        # a new model revision recomputes
        monkeypatch.setitem(T.COMPONENT_MODELS, T.COMPONENT_ONEFORMER,
                            ((T.ONEFORMER_MODEL[0], "0" * 40),))
        s = Stub()
        _ensure(world, s, params=T.TransientParams.live())
        assert len(s.calls) == N_FRAMES
        # a different effective redaction label recomputes
        world.set_label("none")
        s = Stub()
        _ensure(world, s, params=T.TransientParams.live(), redactor_factory=_FakeRedactor)
        assert len(s.calls) == N_FRAMES

    def test_a_cache_file_for_another_keyframe_or_shape_is_not_trusted(self, world):
        report = _ensure(world, Stub())
        depth = world.dense / "work" / "depth"
        src = depth / f"{HAND_KI:05d}_transient.{T.COMPONENT_ONEFORMER}.npz"
        dst = depth / f"{4:05d}_transient.{T.COMPONENT_ONEFORMER}.npz"
        dst.write_bytes(src.read_bytes())            # keyframe 5's mask under 4's name
        assert report.mask(4) is None                # checked again on read
        s = Stub()
        _ensure(world, s)
        assert s.computed(T.COMPONENT_ONEFORMER) == [4]
        key = dict(report.keys[HAND_KI])[T.COMPONENT_ONEFORMER]
        assert T.read_component(src, key, (H + 1, W)) is None


# ---------------------------------------------------------------------------
# the missing-detector policy
# ---------------------------------------------------------------------------


class TestWithoutADetector:

    def test_the_appearance_is_built_and_says_it_is_not_masked(self, world, caplog):
        stub = Stub(reason="no detector on this test host")
        with caplog.at_level(logging.WARNING, logger=T.logger.name):
            result = world.build(redactor_factory=_never_redact, transient_backend_factory=stub)
            world.build(redactor_factory=_never_redact, transient_backend_factory=stub, force=True)
        assert result.state == AP.STATE_OK, result.detail
        man = world.manifest()
        tr = man["transients"]
        assert tr["state"] == T.STATE_UNAVAILABLE
        assert tr["detail"] == "no detector on this test host"
        assert tr["rule"] is None and tr["models"] == {} and tr["frames_masked"] == 0
        assert tr["frames_digest"] is None
        for k in man["keyframes"]:
            assert k["transient_mask"] is None and k["detector_fraction"] is None
        assert man["occluders"]["frames_with_detector_mask"] == 0
        # the photometric vote still ran
        assert "transient_fraction" in man["keyframes"][0]
        # logged once, not once per build
        assert sum("no detector on this test host" in r.getMessage()
                   for r in caplog.records) <= 1
        assert stub.calls == []

    def test_weights_that_cannot_be_fetched_are_unavailable_not_a_failure(self, world):
        err = T.TransientDetectorUnavailable("transient detector model 'x' is not in the cache")
        result = world.build(redactor_factory=_never_redact,
                             transient_backend_factory=Stub(run_error=err))
        assert result.state == AP.STATE_OK, result.detail
        tr = world.manifest()["transients"]
        assert tr["state"] == T.STATE_UNAVAILABLE and "not in the cache" in tr["detail"]
        assert all(k["transient_mask"] is None for k in world.manifest()["keyframes"])

    def test_the_hub_loader_names_an_offline_checkpoint_as_unavailable(self, tmp_path, monkeypatch):
        """`transformers` re-raises the hub's LocalEntryNotFoundError as an
        OSError `from` it; that is still "not here and not fetchable"."""
        from huggingface_hub.errors import LocalEntryNotFoundError

        from tower.world_builder import dense

        monkeypatch.setattr(dense, "hub_model_cache", lambda model_id: tmp_path / "hub")

        def offline():
            try:
                raise LocalEntryNotFoundError("not cached")
            except LocalEntryNotFoundError as exc:
                raise OSError("We couldn't connect to 'https://huggingface.co'") from exc

        with pytest.raises(T.TransientDetectorUnavailable) as caught:
            dense.load_hub_weights("oneformer", T.ONEFORMER_MODEL[0], offline,
                                   what="transient detector model",
                                   error=T.TransientDetectorUnavailable, hint="0.85 GB")
        assert T.ONEFORMER_MODEL[0] in str(caught.value) and "0.85 GB" in str(caught.value)
        with pytest.raises(OSError):
            dense.load_hub_weights("oneformer", "x/y", lambda: (_ for _ in ()).throw(OSError("disk")),
                                   error=T.TransientDetectorUnavailable)

    def test_the_switch_and_a_missing_package_are_unavailable(self, monkeypatch):
        monkeypatch.setenv(T.ENV_SWITCH, "off")
        backend = T.default_backend_factory(T.COMPONENT_ONEFORMER)
        assert "disabled" in backend.probe()
        monkeypatch.delenv(T.ENV_SWITCH)
        assert isinstance(T.default_backend_factory(T.COMPONENT_GDSAM), T.GroundingDinoSamBackend)
        assert isinstance(T.default_backend_factory(T.COMPONENT_ONEFORMER), T.OneFormerBackend)
        assert "not installed" in T._packages_missing("no_such_package_for_transients")

    def test_a_detector_that_becomes_available_rebuilds_the_appearance(self, world):
        world.build(redactor_factory=_never_redact, transient_backend_factory=Stub(reason="later"))
        before = world.manifest()["params_digest"]
        result = world.build(redactor_factory=_never_redact, transient_backend_factory=Stub())
        assert result.detail != AP.ALREADY_BUILT
        assert world.manifest()["params_digest"] != before
        assert world.manifest()["transients"]["state"] == T.STATE_OK

    def test_the_surface_fuses_without_masks_and_says_so(self, world):
        from tower.world_builder import surface_pipeline as SP

        _add_surface_inputs(world)
        result = SP.surfacify(world.store, WORLD, SESSION, params=_surface_params(),
                              force=True, transient_backend_factory=Stub(reason="offline"))
        assert result.state == SP.STATE_OK, result.detail
        tr = SP.read_surface_manifest(world.store, WORLD, SESSION)["transients"]
        assert tr["state"] == T.STATE_UNAVAILABLE and tr["frames_fused_with_mask"] == 0


# ---------------------------------------------------------------------------
# appearance: alpha, exposure, digest
# ---------------------------------------------------------------------------


def test_a_detected_hand_is_transparent_in_its_keyframe_only(world):
    """The rectangle is plain wall: no depth difference, no colour difference.
    Only the detector says it is a hand, so only the detector can remove it."""
    result = world.build(redactor_factory=_never_redact, transient_backend_factory=Stub())
    assert result.state == AP.STATE_OK, result.detail
    man = world.manifest()
    assert man["transients"]["state"] == T.STATE_OK
    assert man["transients"]["frames_masked"] == N_FRAMES
    assert man["transients"]["rule"] == T.TransientParams().rule_id()
    entry = next(k for k in man["keyframes"] if k["ki"] == HAND_KI)
    assert entry["transient_mask"] == {"mode": T.MODE_UNION, "rule": T.TransientParams().rule_id()}
    assert entry["detector_fraction"] > 0.05
    rgba = world.decoded(HAND_KI)
    y0, y1, x0, x1 = HAND
    detector = T.compose([(_rect_mask(HAND), np.zeros((H, W), bool))], T.TransientParams(), (H, W))
    assert detector.sum() > _rect_mask(HAND).sum()          # dilated 12 px
    assert not _opaque(rgba)[A.dilate(detector, A.ALPHA_RING_PX)].any()
    # the same pixels in a neighbouring keyframe are appearance
    assert _opaque(world.decoded(4))[y0:y1, x0:x1].all()
    other = next(k for k in man["keyframes"] if k["ki"] == 4)
    assert other["transient_mask"] is not None and other["detector_fraction"] == 0


def test_the_transient_mask_stays_separate_from_the_fill_mask(world):
    world.build(redactor_factory=_never_redact, transient_backend_factory=Stub())
    entry = next(k for k in world.manifest()["keyframes"] if k["ki"] == HAND_KI)
    assert entry["unobserved_fraction"] < 0.01          # the privacy mask is untouched
    assert entry["mask_origin"] == "stored-fill+nearblack"
    fill = np.load(world.dense / "work" / "depth" / f"{HAND_KI:05d}_fill.npy")
    assert not fill.any()                               # and so is the file it came from


def test_exposure_gains_never_sample_a_detected_hand(world, monkeypatch):
    captured = {}
    real = A.solve_gains

    def spy(rgbs, opaque, *a, **k):
        captured["opaque"] = [np.array(o) for o in opaque]
        return real(rgbs, opaque, *a, **k)

    monkeypatch.setattr(A, "solve_gains", spy)
    world.build(redactor_factory=_never_redact, transient_backend_factory=Stub())
    kis = [k["ki"] for k in world.manifest()["keyframes"]]
    op = captured["opaque"][kis.index(HAND_KI)]
    detector = T.compose([(_rect_mask(HAND), np.zeros((H, W), bool))], T.TransientParams(), (H, W))
    excluded = A.dilate(detector, A.ALPHA_RING_PX)
    assert not op[excluded].any()
    assert op[~excluded].mean() > 0.95


def test_a_rule_change_invalidates_the_appearance(world, monkeypatch):
    world.build(redactor_factory=_never_redact, transient_backend_factory=Stub())
    first = world.manifest()
    again = world.build(redactor_factory=_never_redact, transient_backend_factory=Stub())
    assert again.detail == AP.ALREADY_BUILT
    # a new version of the rule (same mode, same parameters)
    monkeypatch.setattr(T, "TRANSIENT_SCHEMA", T.TRANSIENT_SCHEMA + 1)
    result = world.build(redactor_factory=_never_redact, transient_backend_factory=Stub())
    assert result.detail != AP.ALREADY_BUILT
    second = world.manifest()
    assert second["params_digest"] != first["params_digest"]
    assert second["transients"]["rule"] != first["transients"]["rule"]
    # another mode
    params = A.AppearanceParams(selection_samples=4000, transient_detector=T.MODE_ONEFORMER)
    result = world.build(params=params, redactor_factory=_never_redact,
                         transient_backend_factory=Stub())
    assert result.detail != AP.ALREADY_BUILT
    assert world.manifest()["transients"]["mode"] == T.MODE_ONEFORMER


# ---------------------------------------------------------------------------
# surface fusion: zero weight
# ---------------------------------------------------------------------------


def _add_surface_inputs(world):
    """What the surface stage reads beyond the appearance fixture: the aligned
    depth maps and the depth stage's (grey, deliberately uninformative) frames."""
    depth = world.dense / "work" / "depth"
    undist = world.dense / "work" / "undist"
    undist.mkdir(parents=True, exist_ok=True)
    for i in range(N_FRAMES):
        np.save(depth / f"{i:05d}.npy", np.load(depth / f"{i:05d}_pred.npy"))
        cv2.imwrite(str(undist / f"{i:05d}.jpg"), np.full((H, W, 3), 128, np.uint8))


def _surface_params(**kw):
    from tower.world_builder import surface as S

    base = dict(voxel_frac=0.01, lod_face_targets=(0,), canonical_level=0, mobile_level=0,
                smooth_iterations=0, min_weight=0.5, min_component_frac=0.0,
                min_support_frames=1, carve=False)
    base.update(kw)
    return S.SurfaceParams(**base)


def test_a_detected_hand_has_zero_weight_in_fusion(world):
    """A hand lying ON the wall in one keyframe: its depth is the wall's (a hand
    on a desk is within depth noise of the desk), so no geometric test can drop
    it, and its colour fuses into the surface as a ghost. With the detector's
    mask the keyframe contributes nothing there: the surface keeps only what
    the other keyframes saw."""
    from tower.world_builder import surface as S
    from tower.world_builder import surface_pipeline as SP

    _add_surface_inputs(world)
    y0, y1, x0, x1 = HAND
    undist = world.dense / "work" / "undist" / f"{HAND_KI:05d}.jpg"
    img = np.full((H, W, 3), 128, np.uint8)
    img[y0:y1, x0:x1] = (0, 255, 0)                     # BGR: a green hand
    cv2.imwrite(str(undist), img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    R, t = world.pose(HAND_KI)
    K = world.K

    def hand_colours():
        V, F, C, _N = S.read_mesh_bytes(SP.read_surface_level(world.store, WORLD, SESSION, 0))
        pc = V @ R.T + t
        u = K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2]
        v = K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]
        inside = (u > x0 + 3) & (u < x1 - 3) & (v > y0 + 3) & (v < y1 - 3)
        assert inside.sum() > 20
        c = C[inside].astype(int)
        return int((np.abs(c[:, 1] - c[:, 0]) > 8).sum())

    unmasked = SP.surfacify(world.store, WORLD, SESSION, params=_surface_params(),
                            force=True, transient_backend_factory=Stub(reason="no detector"))
    assert unmasked.state == SP.STATE_OK, unmasked.detail
    assert hand_colours() > 20                  # the fixture does fuse a green ghost

    masked = SP.surfacify(world.store, WORLD, SESSION, params=_surface_params(),
                          force=True, transient_backend_factory=Stub())
    assert masked.state == SP.STATE_OK, masked.detail
    assert hand_colours() == 0
    tr = SP.read_surface_manifest(world.store, WORLD, SESSION)["transients"]
    assert tr["state"] == T.STATE_OK and tr["frames_fused_with_mask"] == N_FRAMES
    assert tr["rule"] == T.TransientParams().rule_id()


def test_a_rule_change_invalidates_the_surface(world):
    from tower.world_builder import surface_pipeline as SP

    _add_surface_inputs(world)
    first = SP.surfacify(world.store, WORLD, SESSION, params=_surface_params(),
                         transient_backend_factory=Stub())
    assert first.state == SP.STATE_OK, first.detail
    again = SP.surfacify(world.store, WORLD, SESSION, params=_surface_params(),
                         transient_backend_factory=Stub())
    assert again.detail == SP.ALREADY_BUILT
    other = SP.surfacify(world.store, WORLD, SESSION,
                         params=_surface_params(transient_detector=T.MODE_ONEFORMER),
                         transient_backend_factory=Stub())
    assert other.detail != SP.ALREADY_BUILT
    assert SP.read_surface_manifest(world.store, WORLD, SESSION)["transients"]["mode"] == \
        T.MODE_ONEFORMER


# ---------------------------------------------------------------------------
# provenance: never from a forbidden source
# ---------------------------------------------------------------------------


def test_masks_are_computed_only_from_the_redacted_session_keyframes(world, monkeypatch):
    """Magenta in the solve's images, the depth stage's undist frames, a raw
    capture and sources.json. The detector is never shown a magenta pixel and
    none of those files is opened."""
    _paint_forbidden_sources(world)
    opened = []
    real_open, real_io_open, real_path_open = builtins.open, io.open, pathlib.Path.open
    real_imread = cv2.imread

    def spy(fn):
        def inner(file, *a, **k):
            opened.append(str(file))
            return fn(file, *a, **k)
        return inner

    monkeypatch.setattr(builtins, "open", spy(real_open))
    monkeypatch.setattr(io, "open", spy(real_io_open))
    monkeypatch.setattr(pathlib.Path, "open", spy(real_path_open))
    monkeypatch.setattr(cv2, "imread", spy(real_imread))
    stub = Stub()
    report = _ensure(world, stub)
    monkeypatch.undo()
    assert report.state == T.STATE_OK and report.computed == N_FRAMES
    norm = [p.replace("\\", "/") for p in opened]
    assert not [p for p in norm if f"/solve/{SESSION}/images" in p or "/undist/" in p
                or "/captures/" in p or p.endswith("sources.json")]
    assert any(f"/sessions/{SESSION}/images/" in p for p in norm)
    for (_c, ki), rgb in stub.seen.items():
        assert rgb.shape == (H, W, 3)
        assert not _near_colour(rgb, MAGENTA).any()
        np.testing.assert_allclose(rgb.astype(int), world.render(ki)[:H, :W].astype(int), atol=12)


def test_an_untrusted_label_shows_the_detector_the_redacted_pixels(tmp_path):
    """Label `none` (a live walk): the detector sees what the redactor wrote,
    fill box included, never the stored bytes as they are."""
    w = World(tmp_path, label="none")
    stub = Stub()
    report = _ensure(w, stub, redactor_factory=_FakeRedactor)
    assert report.state == T.STATE_OK
    y0, y1, x0, x1 = _FakeRedactor.box
    for rgb in stub.seen.values():
        assert rgb[y0 + 4:y1 - 4, x0 + 4:x1 - 4].max() <= 8


def test_the_transient_module_opens_no_image_itself():
    source = pathlib.Path(T.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Name):
            names.add(node.id)
    assert not names & {"images_dir", "imread", "imdecode", "SolveWorkspace", "workspace_for",
                        "read_sources", "prepare_images", "_undistorted_image", "Image.open"}
    assert "keyframe_source" in names
    docstrings = {id(node.body[0].value) for node in ast.walk(tree)
                  if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef))
                  and node.body and isinstance(node.body[0], ast.Expr)
                  and isinstance(node.body[0].value, ast.Constant)}
    literals = [node.value for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
                and id(node) not in docstrings]
    assert not [s for s in literals if "undist" in s or "captures" in s
                or "sources" in s or s == "images"]


# ---------------------------------------------------------------------------
# live and final
# ---------------------------------------------------------------------------


def test_the_walk_uses_oneformer_and_the_finished_world_the_union():
    import inspect

    import scripts.world_build_session as B
    from tower.world_builder.surface import SurfaceParams

    assert SurfaceParams().transient_detector == T.MODE_UNION
    assert SurfaceParams.live().transient_detector == T.MODE_ONEFORMER
    assert A.AppearanceParams().transient_detector == T.MODE_UNION
    assert A.AppearanceParams.live().transient_detector == T.MODE_ONEFORMER
    # the final surface and appearance after Stop take the defaults
    src = inspect.getsource(B.main)
    for name in ("surfacify(", "build_appearance("):
        call = src[src.index(name):src.index("\n            )", src.index(name))]
        code = "\n".join(line.split("#")[0] for line in call.splitlines())
        assert "params=" not in code, code


def test_the_surface_cli_passes_the_detector_mode(tmp_path):
    import argparse

    import scripts.world_surface as CLI

    args = argparse.Namespace(live=True, no_carve=False, transient_detector=None,
                              voxel_frac=None, min_weight=None, smooth_iterations=None,
                              min_component_frac=None, gate_rel=None)
    assert CLI._params_from_args(args).transient_detector == T.MODE_ONEFORMER
    args.transient_detector = T.MODE_UNION
    assert CLI._params_from_args(args).transient_detector == T.MODE_UNION
