"""Review V9, M-8 and the mask cache's LOW: excluding unmaskable images has a bound and a signal.

Module: `tower/world_builder/solve_masks.py` (and `transients.read_component`). Contract:
WORLD-BUILDER-COMPONENTS.md §2.5 (`transients`).

At 6d4b567 (RV9-F, RUN `baseline/review/V9/m2/probe_exclusion.py`):
  * 9 of 10 solver images unreadable, or held by another process while hashed, gave
    `applied`, `images_excluded: 9`, no cause and no notice -- indistinguishable from a
    clean solve;
  * all of them held gave `unavailable` with `cause: None` and "no solver image could be
    masked", which the notice turned into "an operator can make the transient detector
    run" -- the wrong owner;
  * a hash or a read that failed was never tried again;
  * reusing a re-extracted database kept an image that had left the solve, extracted whole
    (`probe_reuse.py`);
  * an empty or torn cached mask raised (`EOFError`, `BadZipFile`) out of the mask step.

Now: one retry of the hash and of the read (after one pause per batch), `excluded_reasons`,
`exclusion_notice_due`, `none_masked` with `cause: images-unmaskable`, a reuse check, and a
bad cache file that is a logged miss and is rewritten.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import sqlite3
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

from tower.world_builder import solve_masks as SM
from tower.world_builder import transients as T

H, W = 48, 64


@pytest.fixture(autouse=True)
def _no_real_pause(monkeypatch):
    """The retry's pause is a sleep of `RETRY_PAUSE_S`; the tests count it instead."""
    pauses = []
    monkeypatch.setattr(SM, "_pause", lambda: pauses.append(1), raising=False)
    return pauses


def _workspace(tmp_path, n=4, seed=0):
    root = tmp_path / "solve"
    images = root / "images"
    images.mkdir(parents=True)
    names = []
    for i in range(n):
        name = f"{i:08d}.jpg"
        names.append(name)
        cv2.imwrite(str(images / name),
                    np.random.default_rng(seed + i).integers(0, 255, (H, W, 3), dtype=np.uint8))
    return types.SimpleNamespace(root=root, images_dir=images), names


def _empty_detector(component):
    """Emits an empty mask for every image it is given, like `probe_exclusion.stub_factory`."""

    class Backend(T.ComponentBackend):
        def probe(self):
            return None

        def run(self, items, params, emit, should_stop=None):
            for i, rgb, _u in items:
                emit(i, np.zeros(rgb.shape[:2], bool), np.zeros(rgb.shape[:2], bool), 0.001)
            return {"frames": len(items)}

    Backend.component = component
    return Backend()


def _masks(ws, names, **kw):
    kw.setdefault("shape", (H, W))
    return SM.ensure_solver_masks(ws, names, backend_factory=_empty_detector,
                                  device_probe=lambda: None, **kw)


def _held_once(monkeypatch, names_held):
    """`file_sha1` raises a sharing violation the FIRST time each of `names_held` is hashed."""
    real = SM.file_sha1
    seen = set()

    def flaky(path):
        name = Path(path).name
        if name in names_held and name not in seen:
            seen.add(name)
            raise PermissionError(13, "The process cannot access the file because it is being "
                                      "used by another process")
        return real(path)

    monkeypatch.setattr(SM, "file_sha1", flaky)


# ---------------------------------------------------------------------------
# 1. the hash and the read are retried once
# ---------------------------------------------------------------------------


def test_an_image_held_while_it_is_hashed_is_hashed_again_and_masked(tmp_path, monkeypatch,
                                                                      _no_real_pause):
    """RV9-F `mass_locked`, with the lock let go by the retry. At 6d4b567: 9 excluded."""
    ws, names = _workspace(tmp_path, n=10)
    _held_once(monkeypatch, set(names[1:]))
    rec = _masks(ws, names).record()
    assert rec["state"] == SM.RECORD_APPLIED
    assert rec["images_masked"] == 10 and rec["images_excluded"] == 0
    assert rec["retried"] == {"hash": 9, "read": 0}
    assert len(_no_real_pause) == 1, "one pause for the whole batch, not one per image"


def test_an_image_held_while_it_is_read_is_read_again_and_masked(tmp_path, monkeypatch,
                                                                  _no_real_pause):
    """The read for the detector (`np.fromfile`) meets a sharing violation once."""
    ws, names = _workspace(tmp_path)
    real = np.fromfile
    seen = set()

    def flaky(path, *a, **k):
        name = Path(str(path)).name
        if name == names[2] and name not in seen:
            seen.add(name)
            raise PermissionError(13, "being used by another process")
        return real(path, *a, **k)

    monkeypatch.setattr(np, "fromfile", flaky)
    rec = _masks(ws, names).record()
    assert rec["state"] == SM.RECORD_APPLIED
    assert rec["images_masked"] == 4 and rec["images_excluded"] == 0
    assert rec["retried"]["read"] == 1 and len(_no_real_pause) == 1


def test_without_a_failure_nothing_is_retried_and_nothing_pauses(tmp_path, _no_real_pause):
    ws, names = _workspace(tmp_path)
    rec = _masks(ws, names).record()
    assert rec["retried"] == {"hash": 0, "read": 0} and _no_real_pause == []
    assert rec["excluded_reasons"] == {} and rec["exclusion_notice_due"] is False
    assert rec["none_masked"] is False and rec["state"] == SM.RECORD_APPLIED


# ---------------------------------------------------------------------------
# 2. why each image was excluded, and when that is enough to tell the owner
# ---------------------------------------------------------------------------


def test_every_excluded_image_says_why(tmp_path, monkeypatch):
    ws, names = _workspace(tmp_path, n=6)
    (ws.images_dir / names[1]).write_bytes(b"not a jpeg")                       # undecodable
    cv2.imwrite(str(ws.images_dir / names[2]), np.zeros((H + 2, W, 3), np.uint8))  # wrong size
    real = SM.file_sha1

    def held(path):                                                             # held for good
        if Path(path).name == names[3]:
            raise PermissionError(13, "in use")
        return real(path)

    monkeypatch.setattr(SM, "file_sha1", held)
    rec = _masks(ws, names).record()
    assert rec["state"] == SM.RECORD_APPLIED and rec["images_excluded"] == 3
    assert rec["excluded_reasons"] == {SM.REASON_HASH_FAILED: 1, SM.REASON_UNDECODABLE: 1,
                                       SM.REASON_WRONG_SIZE: 1}
    assert sum(rec["excluded_reasons"].values()) == rec["images_excluded"]
    assert rec["exclusion_notice_due"] is True, "3 of 6 excluded: max(3, 2 %) is 3"


def test_nine_unreadable_images_of_ten_are_applied_but_the_owner_is_told(tmp_path):
    """RV9-F `mass_unreadable`. At 6d4b567 the record held no signal at all."""
    ws, names = _workspace(tmp_path, n=10)
    for name in names[1:]:
        (ws.images_dir / name).write_bytes(b"not a jpeg")
    rec = _masks(ws, names).record()
    assert rec["state"] == SM.RECORD_APPLIED, "their evidence never reaches the solve"
    assert rec["images_excluded"] == 9
    assert rec["excluded_reasons"] == {SM.REASON_UNDECODABLE: 9}
    assert rec["exclusion_notice_due"] is True and rec["none_masked"] is False


def _record_with(images, excluded):
    out = SM.SolverMasks(state=SM.STATE_OK, params=SM.solver_params(), requested_rule="r",
                         images=images)
    for i in range(images - excluded):
        out.masked[f"m{i}"] = {"image_sha1": "a", "mask_sha1": "b", "masked_frac": 0.0}
    for i in range(excluded):
        out.unmasked.append(f"x{i}")
        out.unmasked_reasons[f"x{i}"] = SM.REASON_UNDECODABLE
        out.excluded[f"x{i}"] = {"image_sha1": None, "mask_sha1": "c"}
    return out.record()


@pytest.mark.parametrize("images,excluded,due", [
    (10, 2, False), (10, 3, True),          # max(3, 0.2) = 3
    (150, 3, True),                         # max(3, 3.0) = 3
    (200, 3, False), (200, 4, True),        # max(3, 4.0) = 4
    (678, 13, False), (678, 14, True),      # max(3, 13.56)
])
def test_the_notice_is_due_at_max_3_or_2_percent(images, excluded, due):
    rec = _record_with(images, excluded)
    assert rec["images_excluded"] == excluded
    assert rec["exclusion_notice_due"] is due
    assert rec["state"] == SM.RECORD_APPLIED


# ---------------------------------------------------------------------------
# 3. no image could be masked: the images are the owner, not the detector
# ---------------------------------------------------------------------------


def test_when_no_image_could_be_masked_the_cause_names_the_images(tmp_path, monkeypatch):
    """RV9-F `all_locked`. At 6d4b567: `unavailable`, `cause: None`, and the notice blamed the
    detector. Held for good (the retry does not help)."""
    ws, names = _workspace(tmp_path)

    def held(path):
        raise PermissionError(13, "in use")

    monkeypatch.setattr(SM, "file_sha1", held)
    rec = _masks(ws, names).record()
    assert rec["state"] == SM.RECORD_UNAVAILABLE, "never `applied` with nothing masked"
    assert rec["none_masked"] is True
    assert rec["cause"] == SM.CAUSE_IMAGES_UNMASKABLE and rec["retryable"] is False
    # The solve ran unmasked with all four in it: none was excluded (review V10, L-11b). The
    # reasons are in `detail`.
    assert rec["excluded_reasons"] == {} and rec["images_excluded"] == 0
    assert rec["detail"] == ("no solver image could be masked: 4 of 4 solver images were unusable "
                             "(4 hash-failed)")
    assert "detector" not in rec["detail"]
    assert rec["retried"]["hash"] == 4


def test_none_masked_mixed_image_reasons(tmp_path):
    ws, names = _workspace(tmp_path, n=3)
    (ws.images_dir / names[0]).write_bytes(b"not a jpeg")
    cv2.imwrite(str(ws.images_dir / names[1]), np.zeros((H, W + 4, 3), np.uint8))
    (ws.images_dir / names[2]).write_bytes(b"")
    rec = _masks(ws, names).record()
    assert rec["none_masked"] is True and rec["state"] == SM.RECORD_UNAVAILABLE
    assert rec["excluded_reasons"] == {}, "unmasked, not excluded (review V10, L-11b)"
    assert "(2 undecodable, 1 wrong-size)" in rec["detail"]


def test_a_machine_that_cannot_run_the_detector_is_not_none_masked(tmp_path):
    """The operator's problem keeps its own cause and detail: `none_masked` is the images'."""
    ws, names = _workspace(tmp_path)
    rec = SM.ensure_solver_masks(ws, names, shape=(H, W), backend_factory=_empty_detector,
                                 device_probe=lambda: "no CUDA device").record()
    assert rec["state"] == SM.RECORD_UNAVAILABLE and rec["detail"] == "no CUDA device"
    assert rec["none_masked"] is False and rec["cause"] is None
    assert rec["exclusion_notice_due"] is False


def test_the_masks_off_record_is_todays_byte_for_byte():
    """The default (masks off) publishes today's record: the new keys are absent, so CON's
    notice reads today's behaviour."""
    assert SM.off_record() == {
        "schema": 1, "requested": False, "state": "unavailable", "outcome": "off",
        "extraction_masked": False,
        "detail": ("transient masks on the final solve are off (TOWER_WORLD_SOLVE_MASKS "
                   "unset or false): this solve is unmasked"),
        "rule": None}


def test_the_record_no_longer_promises_an_unattended_re_solve():
    """`solve_masks.py:339-342` said the finisher owes a re-solve for `gpu-oom`; nothing
    re-solves unattended (review V7 H2, `world_finish_pending.py`)."""
    text = " ".join(inspect.getsource(SM.SolverMasks).replace("#", " ").split())
    assert "owed a re-solve" not in text
    assert "NOTHING re-solves unattended" in text and "re-finish" in text


# ---------------------------------------------------------------------------
# 4. reusing a re-extracted database (LOW, `probe_reuse.py`)
# ---------------------------------------------------------------------------


def _earlier_database(ws, masks, extra):
    prev = ws.root / f"database.masked.p{os.getpid()}.deadbeef.db"
    sqlite3.connect(str(prev)).close()
    held = {n: [v["image_sha1"], v["mask_sha1"]] for n, v in masks.masked.items()}
    held.update(extra)
    SM.database_record_path(prev).write_text(json.dumps({
        "schema": 1, "path": SM.MASKING_REEXTRACTED, "rule": masks.params.rule_id(), "images": held}))
    return prev


def test_a_database_holding_an_unmasked_image_that_left_the_solve_is_not_reused(tmp_path):
    ws, names = _workspace(tmp_path, n=3)
    masks = _masks(ws, names)
    _earlier_database(ws, masks, {"gone.jpg": [None, None]})
    db, info = SM.masked_database(ws, masks, all_names=names)
    assert info["reused"] is False
    assert "gone.jpg" in info["rebuilt_because"]
    assert "gone.jpg" not in json.loads(SM.database_record_path(db).read_text())["images"]


def test_a_database_whose_departed_image_was_masked_is_still_reused(tmp_path):
    """Its features were extracted under a mask of the same rule: nothing unmasked rides along."""
    ws, names = _workspace(tmp_path, n=3)
    masks = _masks(ws, names)
    prev = _earlier_database(ws, masks, {"gone.jpg": ["ab" * 20, "cd" * 20]})
    db, info = SM.masked_database(ws, masks, all_names=names)
    assert info["reused"] is True and info["reused_from"] == prev.name


# ---------------------------------------------------------------------------
# 5. a bad cached mask is a logged miss, and is rewritten (LOW)
# ---------------------------------------------------------------------------


def _good_component(tmp_path):
    path = tmp_path / "x.abc.gdsam.npz"
    key = {"schema": 1, "component": "gdsam"}
    T.write_component(path, key, np.zeros((H, W), bool), np.zeros((H, W), bool), image_sha1="s")
    return path, key


@pytest.mark.parametrize("damage", ["empty", "truncated", "garbage"])
def test_a_bad_cached_mask_is_a_miss_and_says_so(tmp_path, caplog, damage):
    path, key = _good_component(tmp_path)
    assert T.read_component(path, key, (H, W)) is not None
    data = path.read_bytes()
    path.write_bytes({"empty": b"", "truncated": data[: len(data) // 2], "garbage": b"PK\x03\x04junk"}[damage])
    with caplog.at_level(logging.WARNING, logger=T.logger.name):
        assert T.read_component(path, key, (H, W)) is None
    assert any("unreadable" in r.getMessage() for r in caplog.records)


def test_a_missing_or_stale_cached_mask_is_a_quiet_miss(tmp_path, caplog):
    path, key = _good_component(tmp_path)
    with caplog.at_level(logging.WARNING, logger=T.logger.name):
        assert T.read_component(tmp_path / "absent.npz", key, (H, W)) is None
        assert T.read_component(path, dict(key, component="other"), (H, W)) is None
        assert T.read_component(path, key, (H + 1, W)) is None
    assert not caplog.records


def test_an_empty_cached_mask_is_computed_again_and_rewritten(tmp_path):
    """At 6d4b567 the empty file raised `EOFError` out of `ensure_solver_masks` -- the solve
    then ran unmasked, every time."""
    ws, names = _workspace(tmp_path)
    first = _masks(ws, names).record()
    assert first["computed"] == 4
    victim = sorted(SM.cache_dir(ws).glob("*.npz"))[0]
    victim.write_bytes(b"")
    again = _masks(ws, names).record()
    assert again["state"] == SM.RECORD_APPLIED and again["images_masked"] == 4
    assert again["computed"] == 1 and again["cache_hits"] == 3
    with np.load(victim) as z:
        assert set(z.files) == {"hand", "phone", "shape", "key"}, "rewritten whole"
