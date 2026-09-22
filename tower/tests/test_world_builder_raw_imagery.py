"""The research imagery bypass: off by default, labelled when on, and refused
by a reader that did not ask for it.

Contract: `docs/contracts/WORLD-BUILDER-APPEARANCE.md` §6.6.

The owner asked, on 2026-09-21, for a path that builds a saved world from the
ORIGINAL local capture instead of the redacted keyframes, so the fix-it
campaign could measure what reconstruction quality that imagery supports --
with the privacy implementation preserved and re-enablable, not removed. These
tests are the fence around it. They run on the same synthetic world as
`test_world_builder_appearance.py`, whose forbidden sources are painted a
magenta nothing else in the scene is: there, the assertion is that no magenta
reaches a texel; here, under the flag, it is that it does, and that every
record, header and page says so.
"""

from __future__ import annotations

import ast
import builtins
import importlib.util
import io
import json
import pathlib

import cv2
import numpy as np
import pytest

from tests.test_world_builder_appearance import (
    MAGENTA,
    SESSION,
    TRUSTED,
    WORLD,
    World,
    _near_colour,
    _opaque,
    _paint_forbidden_sources,
)

from tower.world_builder import appearance as A
from tower.world_builder import appearance_pipeline as AP
from tower.world_builder import raw_imagery as RAWIMG

RAW = RAWIMG.IMAGERY_RAW
RED = RAWIMG.IMAGERY_REDACTED


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def _raw_params(**kw):
    kw.setdefault("selection_samples", 4000)
    return A.AppearanceParams(imagery_source=RAW, **kw)


# ---------------------------------------------------------------------------
# 1. the flag is off unless something asks for it
# ---------------------------------------------------------------------------


def test_every_default_is_the_redacted_product():
    """Nothing in the stack reads the environment to decide this. A params
    object built by a library caller, a test, or the product is `redacted`."""
    from tower.world_builder.dense import DenseParams
    from tower.world_builder.surface import SurfaceParams

    assert A.AppearanceParams().imagery_source == RED
    assert A.AppearanceParams.live().imagery_source == RED
    assert DenseParams().imagery_source == RED
    assert SurfaceParams().imagery_source == RED
    assert SurfaceParams.live().imagery_source == RED


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "none", "FALSE", "  "])
def test_the_environment_flag_is_off_for_every_spelling_of_no(value):
    assert RAWIMG.imagery_source_from_env({}) == RED
    assert RAWIMG.imagery_source_from_env({RAWIMG.RAW_IMAGERY_ENV: value}) == RED


@pytest.mark.parametrize("value", ["1", "true", "yes", "raw-local-research"])
def test_the_environment_flag_is_on_for_anything_else(value):
    assert RAWIMG.imagery_source_from_env({RAWIMG.RAW_IMAGERY_ENV: value}) == RAW


def test_an_unknown_imagery_source_is_refused_rather_than_guessed():
    with pytest.raises(ValueError):
        A.AppearanceParams(imagery_source="raw")
    with pytest.raises(ValueError):
        RAWIMG.normalise("unredacted")


def test_the_default_policy_resolves_no_raw_frames_at_all(world):
    """`resolve_label_policy` is where the bypass is decided. Asked for
    nothing, it decides nothing: no capture is looked for, and the frame
    reader's branch is not reachable."""
    _paint_forbidden_sources(world)
    policy = A.resolve_label_policy(world.store, WORLD, SESSION)
    assert policy.raw is False
    assert policy.raw_images is None
    assert policy.effective == TRUSTED


def test_the_raw_label_is_not_on_the_trusted_allowlist():
    """Belt and braces: even if a raw label reached a trust check, it fails."""
    assert not A.label_is_trusted(RAWIMG.RAW_LABEL)
    assert RAWIMG.RAW_LABEL not in A.TRUSTED_REDACTION_LABELS


# ---------------------------------------------------------------------------
# 2. under the flag the pixels really are the original capture
# ---------------------------------------------------------------------------


def test_a_raw_build_draws_the_original_capture_and_a_redacted_one_does_not(world):
    """The same world, built twice. The magenta exists only in the capture
    frames; it reaches the texels under the flag and nowhere else."""
    _paint_forbidden_sources(world)

    assert world.build(redactor_factory=lambda: (_ for _ in ()).throw(
        AssertionError("a trusted label must not construct a redactor"))).state == AP.STATE_OK
    for k in world.manifest()["keyframes"]:
        rgba = world.decoded(k["ki"])
        assert not (_near_colour(rgba[..., :3], MAGENTA) & _opaque(rgba)).any()

    assert world.build(params=_raw_params(), force=True).state == AP.STATE_OK
    man = world.manifest()
    seen = any((_near_colour(world.decoded(k["ki"])[..., :3], MAGENTA)
                & _opaque(world.decoded(k["ki"]))).any()
               for k in man["keyframes"])
    assert seen, "the raw build did not draw the original capture's pixels"


def test_a_raw_build_applies_no_privacy_mask_and_forces_the_consensus_off(world):
    """No fill mask, no near-black rule, no cross-frame consensus -- and the
    records say `off` because it IS off, rather than naming a rule that did
    not run."""
    _paint_forbidden_sources(world)
    params = _raw_params(redaction_consensus=A.CONSENSUS_UNION)
    assert params.redaction_consensus == A.CONSENSUS_OFF
    assert world.build(params=params).state == AP.STATE_OK
    prov = world.manifest()["appearance_provenance"]
    assert prov["redaction_consensus"]["mode"] == A.CONSENSUS_OFF
    assert prov["unobserved_rule"] == RAWIMG.RAW_UNOBSERVED_RULE
    for k in world.manifest()["keyframes"]:
        assert k["unobserved_fraction"] == 0.0
        assert k["mask_origin"] == A.MASK_RAW_NONE
        assert k["origin"] == A.ORIGIN_RAW_LOCAL


def test_the_transient_hand_mask_is_quality_and_survives_the_bypass(world):
    """The bypass removes PRIVACY masks. The detector's hand/arm/phone mask is
    a quality mask and is still applied, through `transparent_core`."""
    _paint_forbidden_sources(world)
    det = np.zeros((119, 159), bool)
    det[10:40, 10:40] = True

    class _Backend:
        def masks(self, *a, **k):
            return det

    frame = A.PreparedFrame(source=A.FrameSource(ki=0, keyframe_id="k"),
                            R=np.eye(3), t=np.zeros(3), zp=None,
                            occluder=np.zeros_like(det), occluder_record=None,
                            detector=det)
    frame.source.unobserved = np.zeros_like(det)
    assert frame.transparent_core.sum() == det.sum()


def test_a_session_with_no_record_of_its_original_frames_is_refused(world):
    """No `sources.json` is a refusal with a reason, never a silent fallback
    to the redacted keyframes under a raw label."""
    with pytest.raises(A.AppearanceUnavailable) as exc:
        A.resolve_label_policy(world.store, WORLD, SESSION, imagery_source=RAW)
    assert "no record of where its original frames live" in str(exc.value)


# ---------------------------------------------------------------------------
# 3. a raw artifact says what it is, in every record
# ---------------------------------------------------------------------------


def test_a_raw_artifact_is_labelled_everywhere_a_reader_looks(world):
    _paint_forbidden_sources(world)
    assert world.build(params=_raw_params()).state == AP.STATE_OK
    man = world.manifest()
    prov = man["appearance_provenance"]
    assert man["imagery_source"] == RAW
    assert man["privacy_safe"] is False
    assert prov["imagery_source"] == RAW
    assert prov["privacy_safe"] is False
    assert prov["redaction_effective"] == RAWIMG.RAW_LABEL
    assert prov["source"] == RAWIMG.SOURCE_RAW_LOCAL
    assert prov["label_trusted"] is False
    assert prov["note"] == RAWIMG.RAW_NOTE
    assert prov["raw_imagery_set"].startswith(RAW + "@")
    assert AP.imagery_source_of(man) == RAW


def test_a_redacted_artifact_is_unchanged_and_reads_as_redacted(world):
    """The product's manifest gains the key and nothing else; a manifest
    written before the key existed still reads as the product."""
    assert world.build().state == AP.STATE_OK
    man = world.manifest()
    prov = man["appearance_provenance"]
    assert man["imagery_source"] == RED
    assert man["privacy_safe"] is True
    assert prov["source"] == A.SOURCE_SESSION_KEYFRAMES
    assert prov["redaction_effective"] == TRUSTED
    assert prov["raw_imagery_set"] is None
    assert AP.imagery_source_of({}) == RED
    assert AP.imagery_source_of({"appearance_provenance": {}}) == RED


def test_the_two_modes_have_different_params_digests(world):
    """`--force` aside, a raw build must never be answered `already built`
    from a redacted one, nor the reverse."""
    _paint_forbidden_sources(world)
    assert world.build().state == AP.STATE_OK
    redacted = world.manifest()["params_digest"]
    assert world.build(params=_raw_params()).state == AP.STATE_OK
    raw = world.manifest()["params_digest"]
    assert raw != redacted


def test_the_depth_and_surface_keys_separate_the_two_modes():
    from tower.world_builder.dense import DenseParams
    from tower.world_builder.dense_pipeline import _depth_cache_key, depth_cache_matches
    from tower.world_builder.surface import SurfaceParams

    red, raw = DenseParams(), DenseParams(imagery_source=RAW)
    kred = _depth_cache_key("d", red, None, "trusted:x")
    kraw = _depth_cache_key("d", raw, None, "trusted:x")
    assert kred != kraw
    # A key written before the bypass existed is byte-identical to today's.
    assert kred == "d|moge2-vitl|0|20|fill2|trust:trusted:x"
    # And the record, not only the key, has to agree.
    cached = {"cache_key": kraw, "redaction_trust": "trusted:x", "imagery_source": RAW}
    assert depth_cache_matches(cached, "d", raw, None, "trusted:x")
    assert not depth_cache_matches(cached, "d", red, None, "trusted:x")
    assert (SurfaceParams().digest_fields()
            != SurfaceParams(imagery_source=RAW).digest_fields())
    assert A.pixel_trust_token(TRUSTED) != A.pixel_trust_token(
        TRUSTED, imagery_source=RAW, raw_token=RAW)


def test_a_prediction_is_never_reused_across_the_two_modes(tmp_path):
    """The sharpest edge of the lot, and it was live until it was measured.

    A keyframe the redactor found nothing in has a stored image
    byte-identical to its original frame, so the (kid, image_sha1) reuse key
    matches -- but the redacted depth stage may still have INPAINTED it,
    because the fill mask is a shape-gated guess with false positives, and
    the prediction it cached was made on invented pixels. On the canonical
    world 215 of 395 frames offered exactly that.
    """
    from tower.world_builder.dense_pipeline import FILL_RULE, reusable_predictions

    align = tmp_path / "align.json"
    records = [{"ki": 0, "kid": "s:1", "ok": True, "image_sha1": "abc",
                "fill_rule": FILL_RULE}]
    align.write_text(json.dumps({"backend": "moge2-vitl", "records": records}))
    assert reusable_predictions(align, "moge2-vitl") == {0: ("s:1", "abc")}
    assert reusable_predictions(align, "moge2-vitl", RAW) == {}
    align.write_text(json.dumps({"backend": "moge2-vitl", "imagery_source": RAW,
                                 "records": records}))
    assert reusable_predictions(align, "moge2-vitl", RAW) == {0: ("s:1", "abc")}
    assert reusable_predictions(align, "moge2-vitl") == {}


# ---------------------------------------------------------------------------
# 4. a reader refuses the imagery it did not ask for, in both directions
# ---------------------------------------------------------------------------


def test_a_redacted_reader_refuses_a_raw_artifact_and_a_raw_reader_the_product(world):
    _paint_forbidden_sources(world)
    assert world.build(params=_raw_params()).state == AP.STATE_OK
    raw_man = world.manifest()
    assert not AP.imagery_matches(raw_man, RED)
    assert AP.imagery_matches(raw_man, RAW)
    assert not AP.label_matches(world.store, WORLD, SESSION, raw_man, imagery_source=RED)
    assert AP.label_matches(world.store, WORLD, SESSION, raw_man, imagery_source=RAW)

    assert world.build(force=True).state == AP.STATE_OK
    red_man = world.manifest()
    assert AP.imagery_matches(red_man, RED)
    assert not AP.imagery_matches(red_man, RAW)
    assert not AP.label_matches(world.store, WORLD, SESSION, red_man, imagery_source=RAW)


def test_the_route_answers_404_with_the_real_reason(world, monkeypatch):
    from tower.results.world_builder_appearance import (
        AppearanceNotServed,
        appearance_manifest,
    )

    _paint_forbidden_sources(world)
    assert world.build(params=_raw_params()).state == AP.STATE_OK
    monkeypatch.delenv(RAWIMG.RAW_IMAGERY_ENV, raising=False)
    with pytest.raises(AppearanceNotServed) as exc:
        appearance_manifest(world.store, WORLD, SESSION)
    assert RAW in str(exc.value) and RED in str(exc.value)
    assert str(exc.value) != AP.STALE_LABEL_DETAIL

    monkeypatch.setenv(RAWIMG.RAW_IMAGERY_ENV, "1")
    payload, label, imagery = appearance_manifest(world.store, WORLD, SESSION)
    assert label == RAWIMG.RAW_LABEL
    assert imagery == RAW


def test_the_served_headers_name_the_imagery(world, monkeypatch):
    from tower.routes.geometry import _appearance_headers

    product = _appearance_headers(TRUSTED)
    assert product["X-World-Imagery"] == RED
    assert "X-World-Imagery-Warning" not in product
    research = _appearance_headers(RAWIMG.RAW_LABEL, RAW)
    assert research["X-World-Imagery"] == RAW
    assert research["X-World-Imagery-Warning"] == RAWIMG.RAW_NOTE
    assert research["Cache-Control"] == "no-store"


def test_the_transport_gets_the_imagery_vocabulary_from_the_adapter():
    """The HTTP layer must not import this cartridge to serve its headers.

    `tower/routes/geometry.py` is transport: it is shared with every other
    cartridge, and `test_shared_code_does_not_import_a_cartridge` exists
    because the moment it learns one cartridge's vocabulary the next
    cartridge's route inherits it. That rule went red when this feature
    landed, because the route read `IMAGERY_REDACTED` and `RAW_NOTE` straight
    out of `tower.world_builder.raw_imagery`. Both strings are now published
    by the appearance ADAPTER -- which is named after the cartridge and is
    allowed to know it -- and the route copies what it is handed.

    Three assertions, because only all three together are the boundary: the
    adapter answers, the route asks it, and the route imports nothing from
    the cartridge.
    """
    from tower.results import world_builder_appearance as ADP

    # 1. the adapter answers, with the same strings the manifest, the
    #    provenance and the page label themselves with
    assert ADP.DEFAULT_IMAGERY == RAWIMG.IMAGERY_REDACTED
    assert ADP.imagery_warning(RAWIMG.IMAGERY_REDACTED) is None
    assert ADP.imagery_warning(RAW) == RAWIMG.RAW_NOTE

    # 2. the route asks it: the default a caller gets is the adapter's
    from tower.routes.geometry import _appearance_headers

    assert _appearance_headers(TRUSTED)["X-World-Imagery"] == ADP.DEFAULT_IMAGERY

    # 3. and the route's source imports nothing from the cartridge. Read the
    #    file rather than the module: an import the route never executes is
    #    still a dependency, and this is the predicate
    #    test_architecture_boundaries uses.
    route = pathlib.Path(
        importlib.util.find_spec("tower.routes.geometry").origin
    ).read_text(encoding="utf-8")
    imported: list[str] = []
    for node in ast.walk(ast.parse(route)):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
    assert [n for n in imported if n.startswith("tower.world_builder")] == []


def test_textures_never_carry_over_between_the_two_modes():
    """The manifest's epoch is the only signal the page and the phone's
    texture cache have. Crossing modes must change it."""
    # Identical in EVERY other field, so only the imagery source can be
    # what stops the carry-over.
    red = {"keyframe_image_set": None, "label_trusted": True,
           "session_redaction": TRUSTED, "imagery_source": RED}
    raw = dict(red, imagery_source=RAW)
    assert AP.textures_carry_over(red, dict(red))
    assert not AP.textures_carry_over(red, raw)
    assert not AP.textures_carry_over(raw, red)
    assert (AP.appearance_epoch({"appearance_provenance": red}, raw, "b2") == "b2")


def test_a_superseded_file_is_never_lent_across_the_two_modes(tmp_path):
    """Chunk names are content digests in one namespace per session, so the
    grace-period lend is where a raw chunk could have been served under a
    redacted manifest."""
    import time as _time

    root = tmp_path
    (root / "superseded.json").write_text(json.dumps({"schema_version": 1, "entries": [
        {"at": _time.time(), "build_id": "b1", "session_redaction": TRUSTED,
         "keyframe_image_set": None, "imagery_source": RAW,
         "files": {"c.0123456789abcdef0123456789abcdef.bin": 10}}]}))
    man = {"appearance_provenance": {"session_redaction": TRUSTED,
                                     "keyframe_image_set": None},
           "imagery_source": RED, "chunks": [], "proxy": {}}
    assert AP.servable_size(root, "c.0123456789abcdef0123456789abcdef.bin", man) is None
    man["imagery_source"] = RAW
    assert AP.servable_size(root, "c.0123456789abcdef0123456789abcdef.bin", man) == 10


def test_the_world_listing_never_presents_a_research_build_as_the_product(world, monkeypatch):
    from tower.results.world_builder_library import _appearance_summary

    _paint_forbidden_sources(world)
    assert world.build().state == AP.STATE_OK
    rec = world.store.read_world(WORLD)
    monkeypatch.delenv(RAWIMG.RAW_IMAGERY_ENV, raising=False)
    product = _appearance_summary(world.store, WORLD, SESSION, rec)
    assert product["imagery_source"] == RED
    assert product["privacy_safe"] is True
    assert "face redaction" in product["imagery"]

    assert world.build(params=_raw_params(), force=True).state == AP.STATE_OK
    monkeypatch.setenv(RAWIMG.RAW_IMAGERY_ENV, "1")
    research = _appearance_summary(world.store, WORLD, SESSION, rec)
    assert research["imagery_source"] == RAW
    assert research["privacy_safe"] is False
    assert research["imagery"] == RAWIMG.RAW_NOTE


def test_the_page_is_told_which_imagery_it_is_drawing(world, monkeypatch):
    from tower.world_builder import appearance_render as AR

    _paint_forbidden_sources(world)
    assert world.build(params=_raw_params()).state == AP.STATE_OK
    monkeypatch.setenv(RAWIMG.RAW_IMAGERY_ENV, "1")
    config = AR.build_appearance_config(world.store, WORLD, SESSION)
    assert config["imagery_source"] == RAW
    assert config["privacy_safe"] is False
    assert config["redaction_effective"] == RAWIMG.RAW_LABEL


def test_the_page_never_claims_redaction_it_cannot_know_about():
    """The caption's "with faces redacted" was hardcoded. Whatever the page
    says about redaction must now be behind the flag it is told about."""
    from tower.world_builder.appearance_render import viewer_template_path

    page = viewer_template_path().read_text(encoding="utf-8")
    assert "const RAW_IMAGERY" in page
    claim = """: "These are the camera's own frames, with faces redacted"""
    assert page.count(claim) == 1
    i = page.index(claim)
    assert "RAW_IMAGERY" in page[max(0, i - 800):i]
    assert 'id="rawmark"' in page


# ---------------------------------------------------------------------------
# 5. the bypass stays local
# ---------------------------------------------------------------------------


def test_a_raw_build_writes_nothing_outside_the_world_directory(world, monkeypatch):
    """The whole point of "local" is that the original frames go into the
    world's own appearance directory and nowhere else. Every file this build
    opens for writing is checked, not only the ones it means to write."""
    _paint_forbidden_sources(world)
    written = []
    real_open, real_io_open = builtins.open, io.open
    real_path_open, real_imwrite, real_save = pathlib.Path.open, cv2.imwrite, np.save

    def record(path, mode):
        # An int is a file descriptor an atomic write already opened through
        # `os.open`; the path it names was recorded when that call was made.
        if isinstance(path, int):
            return
        if any(c in str(mode) for c in ("w", "a", "x", "+")):
            written.append(str(path))

    def spy_open(file, mode="r", *a, **k):
        record(file, mode)
        return real_open(file, mode, *a, **k)

    def spy_io_open(file, mode="r", *a, **k):
        record(file, mode)
        return real_io_open(file, mode, *a, **k)

    def spy_path_open(self, mode="r", *a, **k):
        record(self, mode)
        return real_path_open(self, mode, *a, **k)

    def spy_imwrite(path, *a, **k):
        written.append(str(path))
        return real_imwrite(path, *a, **k)

    def spy_save(path, *a, **k):
        written.append(str(path))
        return real_save(path, *a, **k)

    monkeypatch.setattr(builtins, "open", spy_open)
    monkeypatch.setattr(io, "open", spy_io_open)
    monkeypatch.setattr(pathlib.Path, "open", spy_path_open)
    monkeypatch.setattr(cv2, "imwrite", spy_imwrite)
    monkeypatch.setattr(np, "save", spy_save)
    result = world.build(params=_raw_params())
    monkeypatch.undo()
    assert result.state == AP.STATE_OK, result.detail

    allowed = str(world.store.world_dir(WORLD)).replace("\\", "/").lower()
    stray = [p for p in written
             if not p.replace("\\", "/").lower().startswith(allowed)]
    assert stray == [], stray
    appearance = str(AP.appearance_dir(world.store, WORLD, SESSION)
                     ).replace("\\", "/").lower()
    assert any(p.replace("\\", "/").lower().startswith(appearance) for p in written)


def test_nothing_reads_the_original_capture_unless_the_flag_asked(world, monkeypatch):
    """The complement of the test above, and of the privacy lane's own: with
    the flag off, the capture directory and `sources.json` are not opened even
    though this world has both."""
    _paint_forbidden_sources(world)
    opened = []
    real_open, real_path_open = builtins.open, pathlib.Path.open

    def spy_open(file, *a, **k):
        opened.append(str(file))
        return real_open(file, *a, **k)

    def spy_path_open(self, *a, **k):
        opened.append(str(self))
        return real_path_open(self, *a, **k)

    monkeypatch.setattr(builtins, "open", spy_open)
    monkeypatch.setattr(pathlib.Path, "open", spy_path_open)
    result = world.build()
    monkeypatch.undo()
    assert result.state == AP.STATE_OK, result.detail
    norm = [p.replace("\\", "/") for p in opened]
    assert [p for p in norm if "/captures/" in p or p.endswith("sources.json")] == []
