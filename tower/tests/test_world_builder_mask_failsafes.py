"""Review V8, M2b and M2c: the solver's transient masks never fail a whole walk silently.

Module: `tower/world_builder/solve_masks.py`. Contract: WORLD-BUILDER-COMPONENTS.md §2.5
(`transients.state`, `cause`, `retryable`).

M2c. `ensure_solver_masks` retries a GPU out-of-memory error once. The images already
emitted were tracked in ONE set shared by both union components, so an OOM in the
SECOND component (OneFormer, after Grounding DINO + SAM had emitted every image) was
"retried" over an empty list: a transient OOM became a permanent `unavailable` with
`cause: None` (reproduced in RUN `baseline/review/V8/routes/repro_oom_second_component.py`,
which the first test here is).

M2b. An image the detector could not mask used to make the record `partial`, which puts
the WHOLE walk into the gate's masks fail-safe (nothing attached), while the filtered
walk database kept every match of that image (`_keypoints_in_mask` skipped it) -- so
its unmasked evidence still reached the solve. Now such an image is EXCLUDED: every
match and verified inlier touching it is dropped from the filtered database, and on
re-extraction its COLMAP mask is all 0, so no feature of it is extracted. The solve
then holds no unmasked evidence, the record says `applied` with the image counted in
`images_excluded`, and the walk is gated normally. A fallback RULE (OneFormer alone)
stays `partial`: the images are masked, but not by the rule the evidence was measured
with.
"""

from __future__ import annotations

import sqlite3
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

from tests.test_world_builder_solve_masks import (  # noqa: F401 -- fixtures
    N,
    StubDetector,
    _B,
    _masked,
    _rows,
    _settings_unset,
    _solve,
    colmap,
    session,
    walked,
)
from tower.world_builder import solve_masks as SM
from tower.world_builder import transients as T

H, W = 48, 64


def _workspace(tmp_path, n=4):
    root = tmp_path / "solve"
    images = root / "images"
    images.mkdir(parents=True)
    names = []
    for i in range(n):
        name = f"{i:08d}.jpg"
        names.append(name)
        cv2.imwrite(str(images / name),
                    np.random.default_rng(i).integers(0, 255, (H, W, 3), dtype=np.uint8))
    return types.SimpleNamespace(root=root, images_dir=images), names


def _oom_in(component, calls, *, times=1, emit_first=0):
    """A backend factory whose `component` raises a CUDA OOM on its first `times` runs,
    after emitting `emit_first` images; every other run emits every image."""

    def factory(c):
        class Backend(T.ComponentBackend):
            def probe(self):
                return None

            def run(self, items, params, emit, should_stop=None):
                calls.append((c, len(items)))
                runs = sum(1 for k, _ in calls if k == c)
                if c == component and runs <= times:
                    for i, rgb, _u in items[:emit_first]:
                        emit(i, np.zeros(rgb.shape[:2], bool), np.zeros(rgb.shape[:2], bool), 0.001)
                    raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
                for i, rgb, _u in items:
                    emit(i, np.zeros(rgb.shape[:2], bool), np.zeros(rgb.shape[:2], bool), 0.001)
                return {"frames": len(items)}

        Backend.component = c
        return Backend()

    return factory


# ---------------------------------------------------------------------------
# M2c: the OOM retry of the second component
# ---------------------------------------------------------------------------


def test_an_oom_in_the_second_component_is_retried_over_its_own_missing_images(tmp_path,
                                                                                monkeypatch):
    """The V8 repro. Before the fix: backend calls [(gdsam, 4), (oneformer, 4),
    (oneformer, 0)], state `unavailable`, 0 masked, cause None."""
    monkeypatch.setattr(SM, "_empty_cuda_cache", lambda: None)
    ws, names = _workspace(tmp_path)
    calls = []
    out = SM.ensure_solver_masks(ws, names, shape=(H, W),
                                 backend_factory=_oom_in(T.COMPONENT_ONEFORMER, calls),
                                 device_probe=lambda: None)
    rec = out.record()
    assert calls == [(T.COMPONENT_GDSAM, 4), (T.COMPONENT_ONEFORMER, 4), (T.COMPONENT_ONEFORMER, 4)]
    assert rec["state"] == SM.RECORD_APPLIED
    assert rec["images_masked"] == 4 and rec["images_unmasked"] == 0
    assert rec["retries"] == 1 and rec["cause"] is None and rec["retryable"] is False
    assert rec["computed"] == 4


def test_the_retry_runs_only_what_that_component_had_not_emitted(tmp_path, monkeypatch):
    monkeypatch.setattr(SM, "_empty_cuda_cache", lambda: None)
    ws, names = _workspace(tmp_path)
    calls = []
    out = SM.ensure_solver_masks(ws, names, shape=(H, W),
                                 backend_factory=_oom_in(T.COMPONENT_ONEFORMER, calls, emit_first=1),
                                 device_probe=lambda: None)
    assert calls[-1] == (T.COMPONENT_ONEFORMER, 3)
    assert out.record()["state"] == SM.RECORD_APPLIED


def test_an_oom_in_the_second_component_that_persists_is_retryable_with_its_cause(tmp_path,
                                                                                  monkeypatch):
    """Before the fix the retry never ran, so a persisting OOM ended `unavailable` with
    `cause: None`, not retryable; now it is `gpu-oom`, retryable, as for the first."""
    monkeypatch.setattr(SM, "_empty_cuda_cache", lambda: None)
    ws, names = _workspace(tmp_path)
    calls = []
    out = SM.ensure_solver_masks(ws, names, shape=(H, W),
                                 backend_factory=_oom_in(T.COMPONENT_ONEFORMER, calls, times=2),
                                 device_probe=lambda: None)
    rec = out.record()
    assert calls == [(T.COMPONENT_GDSAM, 4), (T.COMPONENT_ONEFORMER, 4), (T.COMPONENT_ONEFORMER, 4)]
    assert rec["state"] == SM.RECORD_UNAVAILABLE
    assert rec["cause"] == SM.CAUSE_GPU_OOM and rec["retryable"] is True and rec["retries"] == 1


# ---------------------------------------------------------------------------
# M2b: an image without a mask is excluded, not a whole-walk fail-safe
# ---------------------------------------------------------------------------


def _break(session_, name="00000001.jpg"):
    (session_.workspace.images_dir / name).write_bytes(b"not a jpeg")
    return name


def test_an_image_that_cannot_be_masked_is_excluded_and_the_solve_stays_applied(session, colmap):
    """Re-extraction: its COLMAP mask is all 0, so none of its features reach the solve.
    Before: `partial` and an all-255 ("use everything") mask."""
    _solve(session, final=True)            # undistort once
    colmap.log.clear()
    bad = _break(session)
    summary = _masked(session, StubDetector())
    record = summary["transients"]
    assert record["state"] == SM.RECORD_APPLIED
    assert record["images_masked"] == N - 1 and record["images_unmasked"] == 1
    assert record["images_excluded"] == 1 and record["excluded_examples"] == [bad]
    assert (colmap.masks_read[bad] == 0).all(), "no feature of it is extracted"
    for name, m in colmap.masks_read.items():
        if name != bad:
            assert (m == 255).any(), name


def test_an_unmasked_image_loses_every_match_in_the_filtered_walk_database(walked, colmap):
    """Arm A1h: before, `_keypoints_in_mask` skipped an image without a mask, so a pair
    of it with another image kept every match -- (3,4) below survived whole."""
    ws = walked.workspace
    bad = _break(walked, "00000003.jpg")   # image id 4 in the walk database
    summary = _masked(walked, StubDetector())
    record = summary["transients"]
    assert record["state"] == SM.RECORD_APPLIED and record["images_excluded"] == 1
    masked = ws.root / record["database"]
    matches, geometries = _rows(masked, "matches"), _rows(masked, "two_view_geometries")
    assert matches[3 * _B + 4] == [], "every match touching the excluded image is gone"
    assert 3 * _B + 4 not in geometries
    assert matches[1 * _B + 2] == [[2, 2], [3, 3]], "the masked images are filtered as before"
    f = record["filter"]
    assert f["images_excluded"] == 1 and f["images_filtered"] == N - 1
    assert f["images_unfiltered"] == 0
    (_c, _n, _db, listed, _o), = colmap.calls("verify_matches")
    assert "00000002.jpg 00000003.jpg" in listed.splitlines()
    assert record["excluded_examples"] == [bad]


def test_a_walk_database_image_the_mask_step_never_saw_is_excluded_too(walked, colmap):
    """An image in the walk database that is not among this solve's images has no mask:
    its matches cannot be checked, so they are not kept."""
    ws = walked.workspace
    con = sqlite3.connect(str(ws.database_path))
    con.execute("update images set name = 'gone.jpg' where image_id = 4")
    con.commit()
    con.close()
    before = ws.database_path.read_bytes()
    summary = _masked(walked, StubDetector())
    assert ws.database_path.read_bytes() == before
    record = summary["transients"]
    masked = ws.root / record["database"]
    assert _rows(masked, "matches")[3 * _B + 4] == []
    assert record["filter"]["images_excluded"] == 1


def test_an_image_cannot_be_excluded_without_a_mask_shape_so_it_stays_partial(tmp_path):
    """Without the solver camera's shape no COLMAP mask can be written for it, so its
    features would be extracted: that is `partial`, the gate's fail-safe."""
    ws, names = _workspace(tmp_path)
    (ws.images_dir / names[1]).write_bytes(b"not a jpeg")
    out = SM.ensure_solver_masks(ws, names, shape=None, backend_factory=StubDetector(),
                                 device_probe=lambda: None)
    rec = out.record()
    assert rec["state"] == SM.RECORD_PARTIAL
    assert rec["images_unmasked"] == 1 and rec["images_excluded"] == 0


def test_an_image_that_cannot_be_hashed_is_excluded_with_a_mask_of_its_own(tmp_path, monkeypatch):
    ws, names = _workspace(tmp_path)
    real = SM.file_sha1

    def flaky(path):
        if Path(path).name == names[2]:
            raise PermissionError("locked by another process")
        return real(path)

    monkeypatch.setattr(SM, "file_sha1", flaky)
    out = SM.ensure_solver_masks(ws, names, shape=(H, W), backend_factory=StubDetector(),
                                 device_probe=lambda: None)
    rec = out.record()
    assert rec["state"] == SM.RECORD_APPLIED and rec["excluded_examples"] == [names[2]]
    png = cv2.imread(str(SM.masks_dir(ws) / f"{names[2]}.png"), cv2.IMREAD_GRAYSCALE)
    assert png is not None and png.shape == (H, W) and (png == 0).all()


def test_a_fallback_rule_is_still_partial_and_excluding_images_does_not_hide_it(session, colmap):
    class Half(StubDetector):
        def __call__(self, component):
            if component == T.COMPONENT_GDSAM:
                return T.DisabledBackend(component, "Grounding DINO weights missing")
            return super().__call__(component)

    _solve(session, final=True)
    _break(session)
    record = _masked(session, Half())["transients"]
    assert record["state"] == SM.RECORD_PARTIAL and record["rule_fallback"]
    assert record["images_excluded"] == 1


def test_a_re_extracted_database_of_an_image_once_used_whole_is_not_reused(session, colmap):
    """A re-extracted database made when an unmasked image was extracted WHOLE (all-255,
    before this change: `[None, None]` in its record) must not be reused once that image
    is excluded: COLMAP never re-extracts an image already in a database."""
    import json

    _solve(session, final=True)
    colmap.log.clear()
    _break(session)
    first = _masked(session, StubDetector())["transients"]
    record_path = SM.database_record_path(session.workspace.root / first["database"])
    doc = json.loads(record_path.read_text())
    doc["images"]["00000001.jpg"] = [None, None]          # as the old code recorded it
    record_path.write_text(json.dumps(doc))
    second = _masked(session, StubDetector())["transients"]
    assert second["database_reused"] is False
    assert "00000001.jpg" in second["database_rebuilt_because"]
    # ... and one made under the exclusion mask is reused.
    third = _masked(session, StubDetector())["transients"]
    assert third["database_reused"] is True


def test_the_walk_database_filter_without_exclusions_is_unchanged(walked, colmap):
    """Every image masked: the counts are today's, with `images_excluded` 0."""
    record = _masked(walked, StubDetector())["transients"]
    f = record["filter"]
    assert (f["images_filtered"], f["images_unfiltered"], f["images_excluded"]) == (N, 0, 0)
    assert record["images_excluded"] == 0 and record["state"] == SM.RECORD_APPLIED


@pytest.mark.parametrize("rows", [0, 4])
def test_an_excluded_image_with_or_without_keypoints_is_counted_once(walked, colmap, rows):
    ws = walked.workspace
    con = sqlite3.connect(str(ws.database_path))
    if rows == 0:
        con.execute("update keypoints set rows = 0, data = NULL where image_id = 4")
    con.commit()
    con.close()
    _break(walked, "00000003.jpg")
    f = _masked(walked, StubDetector())["transients"]["filter"]
    assert f["images_excluded"] == 1
    assert f["keypoints_excluded"] == rows
