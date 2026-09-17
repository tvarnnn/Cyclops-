"""The explicit re-redaction step: `world_builder/reredaction.py`,
`scripts/world_reredact.py`, and every reader that honours its switch.

Contract: docs/contracts/WORLD-BUILDER-APPEARANCE.md section 6.5.

The capture here is synthetic and small. Its raw frames live under a fake Tower
root (`<tmp>/tower/data/captures/...`) and `sources.json` names them RELATIVE
to it, exactly as the builder does. Stored keyframes are the raw frames with
two boxes filled, as an older rule would: box A is a false positive the
current rule no longer fills, box B a "face" it still does. The redactor is a
stub that fills whichever boxes it is told to, under the current label.

Tests 1-13 follow the fix-it re-redaction lane's plan
(Glasses-scratch/wb-final-recon/fixit/reredact/REREDACT.md section 6).
"""

from __future__ import annotations

import builtins
import hashlib
import json
import logging
import os
import pathlib
import sys
import tempfile

import cv2
import numpy as np
import pytest

from tower.world_builder import reredaction as RR
from tower.world_builder.redaction import REDACTION_NONE, RedactionResult

W, S = "w1", "s1"
UNGATED = "faces-detected-and-filled/yunet-2023mar@0.30"
P1 = UNGATED + "+plausibility1"
P2 = UNGATED + "+plausibility2"
CURRENT = RR.TARGET_LABEL
H_, W_ = 120, 160
BOX_A = (10, 50, 10, 60)      # y0, y1, x0, x1: filled by the old rule only
BOX_B = (70, 110, 100, 150)   # filled by both rules
N = 5


def _raw_image(i: int) -> np.ndarray:
    """Smooth, never near-black, and different enough from its neighbour that
    one cannot pass for the other (the verification's hard negative)."""
    yy, xx = np.mgrid[0:H_, 0:W_].astype(float)
    ph = 1.7 * i
    base = 130 + 45 * np.sin(xx / 9.0 + ph) + 25 * np.cos(yy / 11.0 - ph)
    return np.clip(np.stack([base - 8, base, base + 8], -1), 0, 255).astype(np.uint8)


def _enc(img: np.ndarray, q: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    assert ok
    return buf.tobytes()


def _dec(data: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def _filled(img: np.ndarray, boxes) -> np.ndarray:
    out = img.copy()
    for y0, y1, x0, x1 in boxes:
        out[y0:y1, x0:x1] = 0
    return out


def _sha1(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


class Rule:
    """The current redactor, stubbed: fills `boxes`, under `label`, and keeps
    `FaceRedactor.redact`'s shapes -- the input bytes back when nothing is
    filled, the ORIGINAL bytes labelled `none` on a failure."""

    available = True
    unavailable_reason = None

    def __init__(self, boxes=(BOX_B,), label=CURRENT, none_on=None, transform=None):
        self.boxes = boxes
        self.label = label
        self.none_on = none_on
        self.transform = transform
        self.calls = 0

    def redact(self, data):
        self.calls += 1
        img = _dec(data)
        if self.none_on is not None and self.none_on(img):
            return RedactionResult(image_bytes=data, label=REDACTION_NONE, regions=0,
                                   unavailable_reason="detector threw")
        if self.transform is not None:
            return RedactionResult(image_bytes=_enc(self.transform(img)), label=self.label,
                                   regions=1)
        if not self.boxes:
            return RedactionResult(image_bytes=data, label=self.label, regions=0)
        return RedactionResult(image_bytes=_enc(_filled(img, self.boxes)), label=self.label,
                               regions=len(self.boxes))


class Unavailable:
    available = False
    unavailable_reason = "no model on this test host"
    label = REDACTION_NONE

    def redact(self, data):  # pragma: no cover - must never be called
        raise AssertionError("an unavailable redactor was asked to redact")


class Capture:
    """A finished session: stored keyframes filled by an older rule, raw frames
    under a fake Tower root, `sources.json` relative to it."""

    def __init__(self, tmp_path, label=UNGATED, n=N, stored_boxes=(BOX_A, BOX_B)):
        from tower.world_builder.global_solve import write_sources
        from tower.world_builder.records import Keyframe, Session, World
        from tower.world_builder.store import WorldStore

        self.tower = tmp_path / "tower"
        self.root = tmp_path / "root"
        self.store = WorldStore(self.root)
        self.store.write_world(World(world_id=W, created_at=1.0, updated_at=2.0,
                                     session_ids=(S,)))
        self.store.write_session(Session(session_id=S, world_id=W, started_at=1.0,
                                         ended_at=2.0, redaction=label))
        self.kids, self.sources, self.raw_bytes = [], {}, {}
        for i in range(n):
            seq = f"{i + 1:08d}"
            kid = f"{S}:{seq}"
            raw = _raw_image(i)
            rel = f"data/captures/cap1/frames/{seq}.jpg"
            (self.tower / rel).parent.mkdir(parents=True, exist_ok=True)
            self.raw_bytes[kid] = _enc(raw, 95)
            (self.tower / rel).write_bytes(self.raw_bytes[kid])
            # What the engine persists: the decoded capture frame, filled, re-encoded.
            stored = _enc(_filled(_dec(self.raw_bytes[kid]), stored_boxes))
            self.store.write_keyframe_image(W, S, f"{seq}.jpg", stored)
            self.store.append_keyframe(W, Keyframe(
                keyframe_id=kid, session_id=S, source_seq=i + 1, received_at=1.0 + i,
                image_relpath=f"images/{seq}.jpg", width=W_, height=H_,
                byte_count=len(stored)))
            self.kids.append(kid)
            self.sources[kid] = rel
        write_sources(self.store, W, S, self.sources)

    @property
    def images(self) -> pathlib.Path:
        return self.store.images_dir(W, S)

    def stored_hashes(self) -> dict:
        return {p.name: _sha1(p.read_bytes()) for p in sorted(self.images.iterdir())}

    def plan(self, redactor=None, **kw):
        return RR.plan_session(self.store, W, S, redactor=redactor or Rule(),
                               tower_root=self.tower, **kw)

    def apply(self, redactor=None):
        plan = self.plan(redactor)
        return plan, RR.apply_plan(self.store, plan)

    def raw_path(self, i) -> pathlib.Path:
        return self.tower / self.sources[self.kids[i]]


@pytest.fixture
def cap(tmp_path):
    return Capture(tmp_path)


# ---------------------------------------------------------------------------
# 1-2. which labels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label", [UNGATED, P1, P2])
def test_reredact_from_raw_only_for_older_same_family_labels(tmp_path, label):
    """1: the three older gates of the family re-redact from raw."""
    c = Capture(tmp_path, label=label)
    plan = c.plan()
    assert plan.counts == {RR.ORIGIN_REREDACTED: N}
    assert plan.refusal is None
    assert plan.set_label == CURRENT


@pytest.mark.parametrize("label", [
    "none", "", "faces-detected-and-filled/yunet-2023mar@0.50", "redacted",
    UNGATED + "+plausibility9",
])
def test_reredact_never_reads_raw_for_none_or_unknown_labels(tmp_path, label, monkeypatch):
    """1: `none` and unknown labels take the build-time re-redaction path; the
    raw frames are never opened."""
    c = Capture(tmp_path, label=label)
    opened = []
    real = pathlib.Path.read_bytes

    def spy(self):
        opened.append(str(self))
        return real(self)

    monkeypatch.setattr(pathlib.Path, "read_bytes", spy)
    rule = Rule()
    with pytest.raises(RR.ReredactionRefused, match="not an older rule"):
        c.plan(rule)
    assert rule.calls == 0
    assert not [p for p in opened if "captures" in p]


def test_reredact_skipped_when_label_is_current(tmp_path):
    """2."""
    c = Capture(tmp_path, label=CURRENT)
    with pytest.raises(RR.ReredactionRefused, match="already redacted under the current rule"):
        c.plan()


# ---------------------------------------------------------------------------
# 3. the raw frame must provably be the keyframe's source
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("problem", ["neighbour", "wrong-size", "undecodable", "extra-object"])
def test_reredact_requires_verified_raw(cap, problem):
    target = cap.raw_path(2)
    if problem == "neighbour":
        target.write_bytes(cap.raw_bytes[cap.kids[3]])
    elif problem == "wrong-size":
        target.write_bytes(_enc(cv2.resize(_raw_image(2), (W_ // 2, H_ // 2)), 95))
    elif problem == "undecodable":
        target.write_bytes(b"not a jpeg at all")
    else:
        raw = _dec(cap.raw_bytes[cap.kids[2]])
        raw[60:100, 20:70] = (255, 0, 255)     # outside both fill boxes
        target.write_bytes(_enc(raw, 95))
    stored = (cap.images / "00000003.jpg").read_bytes()
    plan = cap.plan()
    f = plan.frames[2]
    assert f.origin == RR.KEPT_RAW_UNVERIFIED, (f.origin, f.detail)
    assert f.image_bytes == stored
    others = [g.origin for i, g in enumerate(plan.frames) if i != 2]
    assert set(others) == {RR.ORIGIN_REREDACTED}
    # Under the ungated label the stored frame still meets the current label.
    assert plan.refusal is None
    RR.apply_plan(cap.store, plan)
    record = json.loads((cap.store.keyframe_image_set(W, S).directory
                         / RR.RECORD_FILENAME).read_text())
    rec = {r["seq"]: r for r in record["frames"]}["00000003"]
    assert rec["origin"] == RR.KEPT_RAW_UNVERIFIED and rec["detail"]
    assert rec["sha1"] == _sha1(stored)


def test_a_raw_frame_missing_keeps_the_stored_keyframe(cap):
    cap.raw_path(1).unlink()
    plan = cap.plan()
    assert plan.frames[1].origin == RR.KEPT_RAW_MISSING
    assert plan.frames[1].image_bytes == (cap.images / "00000002.jpg").read_bytes()


# ---------------------------------------------------------------------------
# 4. redactor unavailable, or `none` on a frame
# ---------------------------------------------------------------------------


def test_reredact_refuses_whole_when_the_redactor_is_unavailable_or_another_rule(cap):
    before = cap.stored_hashes()
    with pytest.raises(RR.ReredactionRefused, match="no face redactor is available"):
        cap.plan(Unavailable())
    with pytest.raises(RR.ReredactionRefused, match="only writes sets under"):
        cap.plan(Rule(label=UNGATED + "+plausibility4"))
    assert cap.store.keyframe_image_set(W, S).name == "images"
    assert not list(cap.images.parent.glob("images.redacted-*"))
    assert cap.stored_hashes() == before


def test_reredact_falls_back_to_stored_when_the_redactor_returns_none(tmp_path):
    """4: with the ungated label the frame's bytes are exactly the stored bytes
    and the set still switches; under plausibility2 the same frame would carry
    the weaker gate's fill under the current label, so nothing switches."""
    fail_frame = _raw_image(1)

    def none_on(img):
        return np.abs(img.astype(int) - fail_frame.astype(int)).mean() < 5

    c = Capture(tmp_path / "ungated")
    plan = c.plan(Rule(none_on=none_on))
    assert plan.frames[1].origin == RR.KEPT_REDACTION_NONE
    assert plan.frames[1].image_bytes == (c.images / "00000002.jpg").read_bytes()
    RR.apply_plan(c.store, plan)
    assert c.store.keyframe_image_set(W, S).redaction == CURRENT

    c2 = Capture(tmp_path / "p2", label=P2)
    plan2 = c2.plan(Rule(none_on=none_on))
    assert plan2.frames[1].origin == RR.KEPT_REDACTION_NONE
    assert plan2.refusal and "does not fill everything" in plan2.refusal
    with pytest.raises(RR.ReredactionRefused):
        RR.apply_plan(c2.store, plan2)
    assert c2.store.keyframe_image_set(W, S).name == "images"
    assert c2.store.read_redaction_set_pointer(W, S) is None


def test_an_exception_in_the_redactor_switches_nothing(cap):
    class Throws(Rule):
        def redact(self, data):
            raise RuntimeError("boom")

    before = cap.stored_hashes()
    with pytest.raises(RuntimeError):
        cap.plan(Throws())
    assert cap.store.read_redaction_set_pointer(W, S) is None
    assert cap.store.keyframe_image_set(W, S).name == "images"
    assert cap.stored_hashes() == before


# ---------------------------------------------------------------------------
# 5-6. what comes out
# ---------------------------------------------------------------------------


def test_reredact_never_returns_raw_bytes(cap):
    """5: a redactor that fills nothing hands back the raw bytes; the set holds
    a fresh encode of them instead."""
    plan = cap.plan(Rule(boxes=()))
    for i, f in enumerate(plan.frames):
        assert f.origin == RR.ORIGIN_REREDACTED
        raw_bytes = cap.raw_bytes[cap.kids[i]]
        assert f.image_bytes != raw_bytes
        diff = np.abs(_dec(f.image_bytes).astype(int) - _dec(raw_bytes).astype(int))
        assert diff.mean() < 1.5


def test_reredacted_output_equals_raw_outside_fill(cap):
    """6: mean <= 1.5 outside the fill, and no pixel over 40 further than 4 px
    from a box."""
    plan = cap.plan()
    y0, y1, x0, x1 = BOX_B
    near = np.zeros((H_, W_), bool)
    near[max(0, y0 - 4):y1 + 4, max(0, x0 - 4):x1 + 4] = True
    for i, f in enumerate(plan.frames):
        d = np.abs(_dec(f.image_bytes).astype(int) - _dec(cap.raw_bytes[cap.kids[i]]).astype(int)).max(-1)
        box = np.zeros((H_, W_), bool)
        box[y0:y1, x0:x1] = True
        assert d[~box].mean() <= 1.5
        assert not ((d > 40) & ~near).any()


def test_an_output_that_is_not_raw_plus_boxes_keeps_the_stored_keyframe(cap):
    plan = cap.plan(Rule(transform=lambda img: cv2.GaussianBlur(_filled(img, [BOX_B]), (9, 9), 4)))
    assert {f.origin for f in plan.frames} == {RR.KEPT_OUTPUT_NOT_RAW}
    assert plan.refusal


# ---------------------------------------------------------------------------
# the safety invariant: re-redaction may only un-fill
# ---------------------------------------------------------------------------


def test_new_fill_outside_the_stored_fill_keeps_the_stored_frame(cap, caplog):
    moved = (60, 100, 10, 60)     # overlaps neither stored box
    with caplog.at_level(logging.WARNING, logger="tower.world_builder.reredaction"):
        plan = cap.plan(Rule(boxes=(moved,)))
    for i, f in enumerate(plan.frames):
        assert f.origin == RR.KEPT_FILL_OUTSIDE_STORED
        assert f.outside_stored_px > 0
        assert f.image_bytes == (cap.images / f"{i + 1:08d}.jpg").read_bytes()
    assert "outside the stored fill" in caplog.text
    assert plan.refusal == "no frame has recoverable fill; nothing to switch"


FACE_ONLY_IN_FRAME_0 = (60, 100, 10, 60)


class _FrameZeroDiffers(Rule):
    """Frame 0 gets `first` (an extra "face" the stored ungated keyframe leaves
    raw, or an output that is not raw plus boxes); every other frame BOX_B
    only, so BOX_A is recovered there. Review 1, repro r1."""

    def __init__(self, raw0, first):
        super().__init__()
        self.raw0, self.first = raw0, first

    def redact(self, data):
        img = _dec(data)
        if float(np.abs(img.astype(int) - self.raw0.astype(int)).mean()) < 3.0:
            if self.first == RR.KEPT_OUTPUT_NOT_RAW:
                return RedactionResult(
                    image_bytes=_enc(cv2.GaussianBlur(_filled(img, [BOX_B]), (9, 9), 4)),
                    label=self.label, regions=1)
            self.boxes = (BOX_B, FACE_ONLY_IN_FRAME_0)
        else:
            self.boxes = (BOX_B,)
        return super().redact(data)


@pytest.mark.parametrize("origin", [RR.KEPT_FILL_OUTSIDE_STORED, RR.KEPT_OUTPUT_NOT_RAW])
def test_a_frame_measured_not_to_meet_the_target_refuses_the_apply_under_any_label(cap, origin):
    """M1: the set's label must be true for every frame in it. A frame kept
    because the current rule filled outside the stored fill (or its output is
    not raw plus boxes) is the ungated superset argument failing, so a set
    holding it cannot carry the current label: refused whole, nothing written,
    no pointer, readers stay on images/ under the stored label."""
    plan = cap.plan(_FrameZeroDiffers(_dec(cap.raw_bytes[cap.kids[0]]), origin))
    assert plan.frames[0].origin == origin
    assert len(plan.recovered_frames) == N - 1, "the other frames did recover box A"
    assert plan.refusal and origin in plan.refusal
    before = cap.stored_hashes()
    with pytest.raises(RR.ReredactionRefused):
        RR.apply_plan(cap.store, plan)
    image_set = cap.store.keyframe_image_set(W, S)
    assert not image_set.reredacted and image_set.redaction == UNGATED
    assert cap.store.read_redaction_set_pointer(W, S) is None
    assert not (cap.store.session_dir(W, S) / RR.set_name()).exists()
    assert cap.stored_hashes() == before


def test_reredacted_fill_is_subset_of_ungated_fill_with_the_real_detector():
    """7: the real YuNet. The current gate only removes boxes from the ungated
    detection pass, so on real faces at several sizes and positions (including
    large and at the frame edge, where the gates differ) its fill lies inside
    the ungated fill, and the step's invariant never trips."""
    from tower.world_builder import redaction as R

    if R.model_path() is None:
        pytest.skip("no face-detection model is vendored on this host")
    skimage_data = pytest.importorskip("skimage.data")
    from tests.test_world_builder_redaction import _room

    redactor = R.FaceRedactor()
    face = cv2.cvtColor(skimage_data.astronaut()[20:220, 150:350], cv2.COLOR_RGB2BGR)
    frames = 0
    for size, (x, y) in ((70, (80, 60)), (150, (400, 100)), (300, (0, 30)), (330, (160, 20))):
        # Camera-like: the synthetic room's aliased texture is far harsher on a
        # JPEG generation than a real frame (mean 4.3 against 0.27 measured).
        frame = cv2.GaussianBlur(_room(), (7, 7), 2.0)
        patch = cv2.resize(face, (size, size), interpolation=cv2.INTER_AREA)
        frame[y:y + size, x:x + size] = patch[: frame.shape[0] - y, : frame.shape[1] - x]
        ungated = np.zeros(frame.shape[:2], bool)
        for (bx, by, bw, bh), _lm in redactor._raw_detect(frame, R.UPSCALE):
            cx, cy = bx + bw / 2.0, by + bh / 2.0
            bw, bh = bw * R.HEAD_DILATION, bh * R.HEAD_DILATION
            x0, y0 = max(0, int(cx - bw / 2.0)), max(0, int(cy - bh / 2.0))
            x1 = min(frame.shape[1], int(cx - bw / 2.0 + bw))
            y1 = min(frame.shape[0], int(cy - bh / 2.0 + bh))
            ungated[y0:y1, x0:x1] = True
        current = np.zeros(frame.shape[:2], bool)
        for bx, by, bw, bh in redactor._detect(frame):
            x0, y0 = max(0, int(bx)), max(0, int(by))
            x1, y1 = min(frame.shape[1], int(bx + bw)), min(frame.shape[0], int(by + bh))
            current[y0:y1, x0:x1] = True
        assert not (current & ~ungated).any()
        if not ungated.any():
            continue
        frames += 1
        raw_bytes = _enc(frame, 95)
        stored = _dec(raw_bytes)
        stored[ungated] = 0
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "raw.jpg"
            p.write_bytes(raw_bytes)
            outcome = RR.reredact_frame(f"{S}:x", _enc(stored), p, redactor)
        assert outcome.origin in (RR.ORIGIN_REREDACTED, RR.KEPT_NOTHING_RECOVERED), (
            size, outcome.origin, outcome.detail)
        assert outcome.outside_stored_px == 0
    assert frames >= 3


# ---------------------------------------------------------------------------
# 8. sources.json is relative to the Tower, not to the cwd
# ---------------------------------------------------------------------------


def test_sources_json_resolves_against_tower_root_not_cwd(cap, tmp_path, monkeypatch):
    from tower.world_builder import global_solve as GS
    from tower.world_builder.dense_pipeline import keyframe_image_bytes

    elsewhere = tmp_path / "somewhere-else"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert not pathlib.Path(cap.sources[cap.kids[0]]).exists()   # relative to cwd: absent
    plan = cap.plan()
    assert plan.counts == {RR.ORIGIN_REREDACTED: N}

    # The same anchor for every reader: the default is `tower/`, the override
    # an environment variable, never the cwd.
    assert GS.resolve_source_path("data/x.jpg") == GS.TOWER_ROOT / "data" / "x.jpg"
    monkeypatch.setenv(GS.SOURCES_ROOT_ENV, str(cap.tower))
    assert GS.resolve_source_path(cap.sources[cap.kids[0]]) == cap.raw_path(0)
    assert GS.resolve_source_path(str(cap.raw_path(0))) == cap.raw_path(0)
    # The dense stage's exact fill mask needs the raw frame: found from here.
    data, origin, mask = keyframe_image_bytes(
        cap.store, W, S, cap.kids[0], cap.sources[cap.kids[0]], None,
        keyframes_are_redacted=True)
    assert origin == "world-keyframe" and mask is not None
    y0, y1, x0, x1 = BOX_A
    assert mask[y0 + 2:y1 - 2, x0 + 2:x1 - 2].all()


def test_the_cli_runs_from_any_directory(cap, tmp_path, monkeypatch, capsys):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
    import world_reredact

    monkeypatch.setattr(RR, "plan_session", _plan_with_rule)
    elsewhere = tmp_path / "cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    before = cap.stored_hashes()
    assert world_reredact.main(["--root", str(cap.root), "--world", W, "--dry-run",
                                "--tower-root", str(cap.tower)]) == 0
    out = capsys.readouterr().out
    assert "raw-reredacted" in out and "recovered" in out
    assert cap.store.read_redaction_set_pointer(W, S) is None     # dry run wrote nothing
    assert not list(cap.images.parent.glob("images.redacted-*"))
    assert world_reredact.main(["--root", str(cap.root), "--world", W, "--apply",
                                "--tower-root", str(cap.tower)]) == 0
    assert "SWITCHED" in capsys.readouterr().out
    assert cap.store.keyframe_image_set(W, S).redaction == CURRENT
    assert world_reredact.main(["--root", str(cap.root), "--world", W, "--revert"]) == 0
    assert cap.store.keyframe_image_set(W, S).name == "images"
    assert cap.stored_hashes() == before


_REAL_PLAN = RR.plan_session


def _plan_with_rule(store, w, s, tower_root=None, **kw):
    return _REAL_PLAN(store, w, s, redactor=Rule(), tower_root=tower_root)


# ---------------------------------------------------------------------------
# 9. raw bytes stay in one function
# ---------------------------------------------------------------------------


def test_raw_bytes_never_leave_provenance_function(cap, monkeypatch):
    readers = []
    real_read = pathlib.Path.read_bytes
    real_open = builtins.open
    capture_root = str(cap.tower / "data" / "captures")

    def spy_read(self):
        if str(self).startswith(capture_root):
            readers.append(sys._getframe(1).f_code.co_name)
        return real_read(self)

    def spy_open(file, *a, **kw):
        if str(file).startswith(capture_root):
            readers.append(sys._getframe(1).f_code.co_name)
        return real_open(file, *a, **kw)

    monkeypatch.setattr(pathlib.Path, "read_bytes", spy_read)
    monkeypatch.setattr(builtins, "open", spy_open)
    plan, _ = cap.apply()
    assert readers and set(readers) == {"reredact_frame"}
    raws = set(cap.raw_bytes.values())
    assert not [f for f in plan.frames if f.image_bytes in raws]
    directory = cap.store.keyframe_image_set(W, S).directory
    for p in directory.iterdir():
        assert p.read_bytes() not in raws
    record = (directory / RR.RECORD_FILENAME).read_text()
    assert "captures" not in record and str(cap.tower) not in record


# ---------------------------------------------------------------------------
# 10. the record, the pointer, and what they say
# ---------------------------------------------------------------------------


def test_record_and_pointer_name_origins_hashes_and_labels(cap):
    before = cap.stored_hashes()
    plan, done = cap.apply()
    image_set = cap.store.keyframe_image_set(W, S)
    assert image_set.name == "images.redacted-plausibility3" == done["set"]
    assert image_set.redaction == CURRENT and image_set.stored_redaction == UNGATED
    assert image_set.cache_token == f"{image_set.name}@{done['set_digest']}"
    pointer = cap.store.read_redaction_set_pointer(W, S)
    assert pointer["active"] == image_set.name and pointer["stored_redaction"] == UNGATED
    record = json.loads((image_set.directory / RR.RECORD_FILENAME).read_text())
    assert record["tool_version"] == RR.TOOL_VERSION
    assert record["stored_redaction"] == UNGATED and record["redaction"] == CURRENT
    assert record["redactor_label"] == CURRENT
    assert record["verification_rule"] == RR.VERIFY_RULE
    assert record["counts"] == {RR.ORIGIN_REREDACTED: N}
    assert record["set_digest"] == done["set_digest"]
    for r in record["frames"]:
        assert r["stored_sha1"] == before[f"{r['seq']}.jpg"]
        assert r["sha1"] == _sha1((image_set.directory / f"{r['seq']}.jpg").read_bytes())
        assert r["recovered_px"] >= (BOX_A[1] - BOX_A[0]) * (BOX_A[3] - BOX_A[2]) * 0.9
    totals = record["totals"]
    assert totals["set_fill_fraction"] < totals["stored_fill_fraction"]


def test_stored_images_are_untouched_by_apply_and_revert(cap):
    before = cap.stored_hashes()
    mtimes = {p.name: p.stat().st_mtime_ns for p in cap.images.iterdir()}
    cap.apply()
    assert cap.stored_hashes() == before
    RR.revert_session(cap.store, W, S)
    assert cap.stored_hashes() == before
    assert {p.name: p.stat().st_mtime_ns for p in cap.images.iterdir()} == mtimes


def test_the_pointer_is_written_last_and_a_failure_before_it_switches_nothing(cap, monkeypatch):
    seen = {}
    real = cap.store.write_redaction_set_pointer

    def at_switch(w, s, pointer):
        d = cap.store.session_dir(W, S) / pointer["active"]
        seen["complete"] = (d.is_dir() and (d / RR.RECORD_FILENAME).exists()
                            and len(list(d.glob("*.jpg"))) == N)
        seen["reader_before"] = cap.store.keyframe_image_set(W, S).name
        return real(w, s, pointer)

    monkeypatch.setattr(cap.store, "write_redaction_set_pointer", at_switch)
    cap.apply()
    assert seen == {"complete": True, "reader_before": "images"}
    assert cap.store.keyframe_image_set(W, S).name == "images.redacted-plausibility3"


def test_a_failure_while_writing_the_set_leaves_readers_on_images(tmp_path, monkeypatch):
    c = Capture(tmp_path)
    plan = c.plan()

    def broken(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(RR, "_record", broken)
    with pytest.raises(OSError):
        RR.apply_plan(c.store, plan)
    assert c.store.read_redaction_set_pointer(W, S) is None
    assert c.store.keyframe_image_set(W, S).name == "images"
    assert not (c.store.session_dir(W, S) / plan.name).exists()
    assert not c.store.lock_path(W).exists()


def test_a_pointer_the_store_cannot_honour_reads_as_no_switch(cap):
    cap.apply()
    good = cap.store.read_redaction_set_pointer(W, S)
    # the stored label changed since the switch
    from dataclasses import replace

    session = cap.store.read_session(W, S)
    cap.store.write_session(replace(session, redaction=P1))
    assert cap.store.keyframe_image_set(W, S).name == "images"
    cap.store.write_session(session)
    assert cap.store.keyframe_image_set(W, S).name != "images"
    for bad in ({**good, "active": "../images"}, {**good, "active": "elsewhere"},
                {**good, "active": "images.redacted-missing"}, {**good, "redaction": None}):
        cap.store.write_redaction_set_pointer(W, S, bad)
        assert cap.store.keyframe_image_set(W, S).name == "images", bad
    cap.store.redaction_set_path(W, S).write_text("{ not json")
    assert cap.store.keyframe_image_set(W, S).name == "images"


def test_revert_is_a_pointer_change_and_reapply_reuses_the_set(cap):
    plan, done = cap.apply()
    directory = cap.store.keyframe_image_set(W, S).directory
    files = {p.name: p.read_bytes() for p in directory.iterdir()}
    assert RR.revert_session(cap.store, W, S) == {"reverted_from": done["set"]}
    assert cap.store.keyframe_image_set(W, S).name == "images"
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == files
    with pytest.raises(RR.ReredactionRefused, match="already reads its stored"):
        RR.revert_session(cap.store, W, S)
    again = RR.apply_plan(cap.store, cap.plan())
    assert again["reused"] is True and again["set_digest"] == done["set_digest"]
    assert [h["action"] for h in cap.store.read_redaction_set_pointer(W, S)["history"]] == [
        "apply", "revert", "apply"]
    with pytest.raises(RR.ReredactionRefused, match="--revert first"):
        RR.apply_plan(cap.store, cap.plan())


def test_apply_refuses_a_live_session_and_a_held_world(tmp_path):
    from dataclasses import replace

    c = Capture(tmp_path)
    session = c.store.read_session(W, S)
    c.store.write_session(replace(session, ended_at=None))
    with pytest.raises(RR.ReredactionRefused, match="has not ended"):
        c.plan()
    c.store.write_session(session)
    plan = c.plan()
    from tower.world_builder import store as store_module

    # A live builder holds the world: another process, alive (our parent).
    c.store.lock_path(W).write_text(json.dumps(store_module._lock_record(os.getppid())),
                                    encoding="utf-8")
    with pytest.raises(RR.ReredactionRefused, match="live writer"):
        RR.apply_plan(c.store, plan)
    assert c.store.lock_path(W).exists()          # not taken from its holder
    c.store.lock_path(W).unlink()
    assert c.store.read_redaction_set_pointer(W, S) is None


# ---------------------------------------------------------------------------
# 11. a corroboration pass that throws fills, and so recovers nothing
# ---------------------------------------------------------------------------


def test_corroboration_failure_during_reredaction_fills(tmp_path, monkeypatch):
    from tower.world_builder import redaction as R

    if R.model_path() is None:
        pytest.skip("no face-detection model is vendored on this host")
    big = (20.0, 10.0, 90.0, 60.0)      # 28% of the frame, off the edge margin
    # The ungated rule filled the box, head dilation included: x -7..137, y -8..88.
    c = Capture(tmp_path, n=1, stored_boxes=[(0, 88, 0, 137)])
    landmarks = [45.0, 30.0, 75.0, 30.0, 60.0, 42.0, 48.0, 55.0, 72.0, 55.0]   # facelike

    def detect(self, image, upscale):
        if upscale == R.UPSCALE:
            return [(big, landmarks)]
        raise RuntimeError("corroboration pass failed")

    monkeypatch.setattr(R.FaceRedactor, "_raw_detect", detect)
    plan = c.plan(R.FaceRedactor())
    f = plan.frames[0]
    assert f.new_fill_px and f.new_fill_px > 0.9 * 90 * 60
    assert f.origin == RR.KEPT_NOTHING_RECOVERED
    assert plan.refusal

    def dissolves(self, image, upscale):
        return [(big, landmarks)] if upscale == R.UPSCALE else []

    monkeypatch.setattr(R.FaceRedactor, "_raw_detect", dissolves)
    plan = c.plan(R.FaceRedactor())
    assert plan.frames[0].origin == RR.ORIGIN_REREDACTED
    assert plan.frames[0].new_fill_px == 0


# ---------------------------------------------------------------------------
# 12. every reader follows the switch through the one accessor
# ---------------------------------------------------------------------------


def test_dense_and_appearance_consume_same_frame_sha(cap):
    from tower.world_builder import appearance as A
    from tower.world_builder.dense_pipeline import keyframe_image_bytes

    _plan, done = cap.apply()
    image_set = cap.store.keyframe_image_set(W, S)
    policy = A.resolve_label_policy(cap.store, W, S, redactor_factory=_never)
    assert policy.trusted and policy.session_redaction == CURRENT
    for i, kid in enumerate(cap.kids):
        on_disk = (image_set.directory / f"{i + 1:08d}.jpg").read_bytes()
        data, origin, _ = keyframe_image_bytes(cap.store, W, S, kid, None, None,
                                               keyframes_are_redacted=True)
        src = A.keyframe_source(cap.store, W, S, kid, i, policy=policy, align_record=None,
                                depth_dir=None, undistorter=None, hash_only=True)
        assert data == on_disk and origin == "world-keyframe"
        assert src.source_sha1 == _sha1(data)
        assert data != (cap.images / f"{i + 1:08d}.jpg").read_bytes()
    RR.revert_session(cap.store, W, S)
    data, _, _ = keyframe_image_bytes(cap.store, W, S, cap.kids[0], None, None,
                                      keyframes_are_redacted=True)
    assert data == (cap.images / "00000001.jpg").read_bytes()


def _never():
    raise AssertionError("a trusted label must not construct a redactor")


def test_a_switch_invalidates_the_depth_fuse_and_surface_caches(tmp_path):
    from tower.world_builder import surface_pipeline as SP
    from tower.world_builder.dense import DenseParams
    from tower.world_builder.dense_pipeline import _depth_cache_key, _fuse_cache_key

    p = DenseParams()
    # Unchanged for the capture's own keyframes: no existing cache is lost.
    assert _depth_cache_key("d", p) == f"d|{p.backend}|{p.component}|{p.min_sparse_points}|fill2"
    assert _depth_cache_key("d", p, None) == _depth_cache_key("d", p)
    assert _depth_cache_key("d", p, "images.redacted-x@1") != _depth_cache_key("d", p)
    assert _depth_cache_key("d", p, "images.redacted-x@1") != _depth_cache_key("d", p, "images.redacted-x@2")
    assert _fuse_cache_key("d", p, "images.redacted-x@1") != _fuse_cache_key("d", p)

    root = tmp_path / "dense"
    for sub in ("depth", "undist"):
        (root / "work" / sub).mkdir(parents=True)
    for ki in range(2):
        (root / "work" / "depth" / f"{ki:05d}.npy").write_bytes(b"x")
        (root / "work" / "undist" / f"{ki:05d}.jpg").write_bytes(b"x")

    class Sol:
        input_digest = "d"

    cached = {"input_digest": "d", "backend": p.backend, "fill_rule": 2, "stopped_after": None,
              "records": [{"ki": 0, "ok": True}, {"ki": 1, "ok": True}]}
    assert SP._depth_cache_usable(cached, root, Sol(), p)
    assert not SP._depth_cache_usable(cached, root, Sol(), p, image_set="images.redacted-x@1")
    switched = {**cached, "keyframe_image_set": "images.redacted-x@1"}
    assert SP._depth_cache_usable(switched, root, Sol(), p, image_set="images.redacted-x@1")
    assert not SP._depth_cache_usable(switched, root, Sol(), p)       # after a revert


def test_appearance_follows_the_switch_and_its_served_label(tmp_path):
    """The appearance built before a switch is stale the moment the pointer
    changes (served 404), a rebuild reads the new set under its label, and a
    revert makes that build stale in turn."""
    from fastapi.testclient import TestClient

    from tests.test_world_builder_appearance import SESSION, WORLD, World, _app
    from tower.world_builder import appearance_pipeline as AP

    w = World(tmp_path, label=UNGATED)
    assert w.build(redactor_factory=_never).state == AP.STATE_OK
    first = w.manifest()
    assert first["appearance_provenance"]["keyframe_image_set"] is None
    client = TestClient(_app(w.root))
    url = f"/worlds/{WORLD}/appearance/{SESSION}/manifest"
    assert client.get(url).status_code == 200

    name = "images.redacted-plausibility3"
    set_dir = w.store.session_dir(WORLD, SESSION) / name
    set_dir.mkdir()
    for i in range(len(w.kids)):
        seq = w.kids[i].rsplit(":", 1)[-1]
        img = w.render(i)
        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 91])
        (set_dir / f"{seq}.jpg").write_bytes(enc.tobytes())
        w.records[i]["image_sha1"] = _sha1(enc.tobytes())
    w.write_align()
    w.store.write_redaction_set_pointer(WORLD, SESSION, {
        "active": name, "redaction": CURRENT, "stored_redaction": UNGATED,
        "set_digest": "abc"})
    assert not AP.label_matches(w.store, WORLD, SESSION, first)
    r = client.get(url)
    assert r.status_code == 404 and "stale" in r.json()["detail"]

    assert w.build(redactor_factory=_never).state == AP.STATE_OK   # not "already built"
    second = w.manifest()
    prov = second["appearance_provenance"]
    assert prov["session_redaction"] == CURRENT
    assert prov["keyframe_image_set"] == f"{name}@abc"
    assert second["params_digest"] != first["params_digest"]
    for k in second["keyframes"]:
        seq = w.kids[k["ki"]].rsplit(":", 1)[-1]
        assert k["source_sha1"] == _sha1((set_dir / f"{seq}.jpg").read_bytes())
    assert client.get(url).status_code == 200

    # Another set under the SAME label is still another set: stale.
    pointer = w.store.read_redaction_set_pointer(WORLD, SESSION)
    w.store.write_redaction_set_pointer(WORLD, SESSION, {**pointer, "set_digest": "abd"})
    assert client.get(url).status_code == 404
    w.store.write_redaction_set_pointer(WORLD, SESSION, pointer)
    assert client.get(url).status_code == 200

    w.store.write_redaction_set_pointer(WORLD, SESSION, {**pointer, "active": None})
    assert client.get(url).status_code == 404


@pytest.mark.parametrize("recorded", ["field", "params_digest"])
def test_a_revert_takes_the_sets_surface_off_every_rung_until_it_is_rebuilt(tmp_path, recorded):
    """Review 1, m4: after `--revert` the appearance route 404s at once, but the
    surface built from the set -- its vertex colours are that set's pixels --
    was still drawn. A surface (or dense artifact) recording a set the session
    no longer reads is not drawable: not on the ladder, not in the revision,
    not in the listing, and a pinned request is refused."""
    from tower.results.world_builder_render import (
        WorldRenderUnavailable,
        build_world_render,
    )
    from tower.world_builder.store import (
        built_from_an_inactive_keyframe_set,
        surface_artifact_drawable,
    )

    from tests.test_world_builder_appearance import SESSION, WORLD, World

    w = World(tmp_path, label=UNGATED)
    name = "images.redacted-plausibility3"
    (w.store.session_dir(WORLD, SESSION) / name).mkdir()
    pointer = {"active": name, "redaction": CURRENT, "stored_redaction": UNGATED,
               "set_digest": "abc"}
    w.store.write_redaction_set_pointer(WORLD, SESSION, pointer)
    token = w.store.keyframe_image_set(WORLD, SESSION).cache_token
    manifest_path = w.store.world_dir(WORLD) / "surface" / SESSION / "manifest.json"
    man = json.loads(manifest_path.read_text())
    if recorded == "field":
        man["keyframe_image_set"] = token
    else:                                   # a surface built before the field existed
        man["params_digest"] = f"digest-1|x|moge2-vitl|set:{token}|transients:off"
    manifest_path.write_text(json.dumps(man))

    assert surface_artifact_drawable(w.store, WORLD, SESSION)
    assert "wb-representation\" content=\"surface\"" in build_world_render(
        w.store, WORLD, SESSION, representation="surface")

    w.store.write_redaction_set_pointer(WORLD, SESSION, {**pointer, "active": None})   # --revert
    assert built_from_an_inactive_keyframe_set(w.store, WORLD, SESSION, "surface")
    assert not surface_artifact_drawable(w.store, WORLD, SESSION)
    with pytest.raises(WorldRenderUnavailable):
        build_world_render(w.store, WORLD, SESSION, representation="surface")
    with pytest.raises(WorldRenderUnavailable):
        build_world_render(w.store, WORLD, SESSION)

    # a surface built from the capture's own images/ is never stale by this test
    man.pop("keyframe_image_set", None)
    man["params_digest"] = "digest-1|x|moge2-vitl|transients:off"
    manifest_path.write_text(json.dumps(man))
    assert surface_artifact_drawable(w.store, WORLD, SESSION)
    w.store.write_redaction_set_pointer(WORLD, SESSION, pointer)
    assert surface_artifact_drawable(w.store, WORLD, SESSION)


# ---------------------------------------------------------------------------
# 13. the frozen canonical capture (slow; needs a COPY of the dataset)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_frozen_reredaction_numbers():
    """398 verified; 89 frames re-redacted; the opening pose (ki 148, seq
    00001296) goes to no fill; ki 120 (seq 00001156) keeps its fill.

    Set WB_REREDACT_DATASET to a world root holding the canonical world and
    TOWER_SOURCES_ROOT to the Tower whose data/captures holds its raw frames.
    Dry: nothing is written."""
    root = os.environ.get("WB_REREDACT_DATASET")
    if not root or not os.environ.get("TOWER_SOURCES_ROOT"):
        pytest.skip("WB_REREDACT_DATASET and TOWER_SOURCES_ROOT are not set")
    from tower.world_builder.store import WorldStore

    store = WorldStore(pathlib.Path(root))
    plan = RR.plan_session(store, "b2a75ab40d2d415d8d6ef5e4d5f0fb3d",
                           "a8c6817e14a74e3c977fccfcdacad595")
    assert plan.counts == {RR.ORIGIN_REREDACTED: 89, RR.KEPT_NOTHING_RECOVERED: 309}
    by_seq = {f.seq: f for f in plan.frames}
    assert by_seq["00001296"].reredacted and by_seq["00001296"].new_fill_px == 0
    assert not by_seq["00001156"].reredacted
    totals = plan.totals()
    assert abs(totals["stored_fill_fraction"] - 0.0983) < 0.0005
    assert abs(totals["set_fill_fraction"] - 0.0486) < 0.0005
    assert all((f.outside_stored_px or 0) == 0 for f in plan.frames)
