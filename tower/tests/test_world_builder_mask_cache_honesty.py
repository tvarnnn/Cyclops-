"""Review V10, L-11b and L-12b: the solve's mask record says only what happened.

Module: `tower/world_builder/solve_masks.py` (and `transients.write_component` /
`transients.ensure_transient_masks`). Contract: WORLD-BUILDER-COMPONENTS.md §2.5 (`transients`).

At 4a95a0d (RV10-D, RUN `baseline/review/V10/`):

  * L-11b: a detector that returned no mask for any image gave `unavailable` -- the solve ran
    UNMASKED on today's database -- yet the record still said `images_excluded: 10` and
    `exclusion_notice_due: true`, so the finalization notice told the owner "10 of 10 images
    could not be masked and were left out of the solve". Nothing was left out.
  * L-12b: the mask cache's write (`solve_masks` `emit`, `transients.write_component`'s
    `os.replace`) was unguarded. One full disk, one cache file another process held, one
    MAX_PATH failure raised out of the detector's run and became `detector-failed`: not
    retryable, and the notice sent an operator after a detector that had worked.

Now the excluded counts and the notice bound apply only when the masks reached the solve, and a
cache write that fails is counted (`cache_write_failed`) while the mask it would have kept is
used for this solve all the same.
"""

from __future__ import annotations

import errno
import os
import types

import cv2
import numpy as np
import pytest

from tower.world_builder import coherence_publish as CP
from tower.world_builder import solve_masks as SM
from tower.world_builder import transients as T

H, W = 48, 64


@pytest.fixture(autouse=True)
def _no_real_pause(monkeypatch):
    monkeypatch.setattr(SM, "_pause", lambda: None, raising=False)


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


def _detector(emit_for=None):
    """A backend factory that marks a small square on every image it is shown (or only on the
    images whose index is in `emit_for`, emitting nothing for the rest)."""

    def factory(component):
        class Backend(T.ComponentBackend):
            def probe(self):
                return None

            def run(self, items, params, emit, should_stop=None):
                for i, rgb, _u in items:
                    if emit_for is not None and i not in emit_for:
                        continue
                    hand = np.zeros(rgb.shape[:2], bool)
                    hand[10:20, 10 + i:20 + i] = True
                    emit(i, hand, np.zeros(rgb.shape[:2], bool), 0.001)
                return {"frames": len(items)}

        Backend.component = component
        return Backend()

    return factory


def _masks(ws, names, factory=None, **kw):
    kw.setdefault("shape", (H, W))
    return SM.ensure_solver_masks(ws, names, backend_factory=factory or _detector(),
                                  device_probe=lambda: None, **kw)


def _summary(record):
    """What the finalization notice is written from: a gate that ran without masks."""
    return {"transients": record,
            "gate": {"state": CP.GATE_STATE_APPLIED, "masks_applied": False, "attach": False}}


# ---------------------------------------------------------------------------
# L-11b: nothing is "left out of the solve" when the solve was not masked
# ---------------------------------------------------------------------------


def test_no_mask_for_any_image_is_not_counted_as_images_left_out(tmp_path):
    """RV10 L-11b. At 4a95a0d: `images_excluded: 10`, `exclusion_notice_due: True`."""
    ws, names = _workspace(tmp_path, n=10)
    rec = _masks(ws, names, _detector(emit_for=set())).record()
    assert rec["state"] == SM.RECORD_UNAVAILABLE and rec["extraction_masked"] is False
    assert rec["images_unmasked"] == 10, "every image is still counted as unmasked"
    assert rec["images_excluded"] == 0 and rec["excluded_examples"] == []
    assert rec["excluded_reasons"] == {}
    assert rec["exclusion_notice_due"] is False
    assert rec["detail"] == "no solver image could be masked", "the detail is today's"


def test_the_notice_of_an_unmasked_solve_never_says_images_were_left_out(tmp_path):
    """RV10 L-11b, through CON's notice: at 4a95a0d it read '...; 10 of 10 images could not be
    masked and were left out of the solve; ...' for a solve that used all ten, unmasked."""
    ws, names = _workspace(tmp_path, n=10)
    rec = _masks(ws, names, _detector(emit_for=set())).record()
    notice = CP.publish_notice(_summary(rec))
    assert notice is not None, "the masks fail-safe itself is still said"
    assert "left out of the solve" not in notice
    assert CP.notice_causes(_summary(rec)) == ["masks-no-image"]


def test_when_no_image_could_be_masked_nothing_is_counted_as_left_out(tmp_path, monkeypatch):
    """`images-unmaskable` (V9 M-8) is unavailable too: the solve ran unmasked. Its reasons are
    in `detail`, and nothing claims the images were excluded."""
    ws, names = _workspace(tmp_path)

    def held(path):
        raise PermissionError(13, "in use")

    monkeypatch.setattr(SM, "file_sha1", held)
    rec = _masks(ws, names).record()
    assert rec["none_masked"] is True and rec["cause"] == SM.CAUSE_IMAGES_UNMASKABLE
    assert rec["images_excluded"] == 0 and rec["excluded_reasons"] == {}
    assert rec["exclusion_notice_due"] is False
    assert rec["detail"].endswith("(4 hash-failed)")
    assert "left out of the solve" not in CP.publish_notice(_summary(rec))


def test_when_the_masks_reach_the_solve_the_excluded_images_are_still_counted(tmp_path):
    """The other side, unchanged: masks applied, the images without one excluded and said."""
    ws, names = _workspace(tmp_path, n=10)
    rec = _masks(ws, names, _detector(emit_for={0, 1, 2, 3, 4, 5})).record()
    assert rec["state"] == SM.RECORD_APPLIED and rec["extraction_masked"] is True
    assert rec["images_masked"] == 6 and rec["images_excluded"] == 4
    assert rec["excluded_reasons"] == {SM.REASON_NO_MASK: 4}
    assert rec["exclusion_notice_due"] is True
    applied = dict(_summary(rec), gate={"state": CP.GATE_STATE_APPLIED, "masks_applied": True})
    assert "4 of 10 images could not be masked and were left out of the solve" in \
        CP.publish_notice(applied)


# ---------------------------------------------------------------------------
# L-12b: a mask-cache write that fails is not a detector failure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("error", [
    OSError(errno.ENOSPC, "No space left on device"),
    PermissionError(13, "The process cannot access the file because it is being used by another process"),
    FileNotFoundError(errno.ENOENT, "No such file or directory"),        # MAX_PATH's face on Windows
])
def test_a_mask_cache_write_that_fails_keeps_the_mask_for_this_solve(tmp_path, monkeypatch, error):
    """RV10 L-12b. At 4a95a0d: `outcome: failed`, `cause: detector-failed`, not retryable, and the
    solve ran unmasked."""
    ref_out = _masks(*_workspace(tmp_path / "ref"))
    reference = ref_out.record()
    ws, names = _workspace(tmp_path / "run")

    def fails(*a, **k):
        raise error

    monkeypatch.setattr(T, "write_component", fails)
    out = _masks(ws, names)
    rec = out.record()
    assert rec["state"] == SM.RECORD_APPLIED and rec["cause"] is None
    assert rec["retryable"] is False and rec["outcome"] == SM.STATE_OK
    assert rec["images_masked"] == 4 and rec["images_excluded"] == 0
    assert rec["computed"] == 4
    assert rec["cache_write_failed"] == 4 * len(SM.solver_params().components)
    assert {n: v["mask_sha1"] for n, v in out.masked.items()} == \
        {n: v["mask_sha1"] for n, v in ref_out.masked.items()}, \
        "the masks this solve used are exactly the ones a working cache gives"
    assert rec["masked_fraction"] == reference["masked_fraction"]
    assert list(SM.cache_dir(ws).glob("*.npz")) == [], "nothing was kept"
    assert CP.notice_causes(_summary(rec)) == [], "no fail-safe, no operator blamed"


def test_without_a_failure_the_record_has_no_cache_write_key(tmp_path):
    """Only when it happened: a working cache writes exactly the record it wrote before."""
    rec = _masks(*_workspace(tmp_path)).record()
    assert "cache_write_failed" not in rec


@pytest.mark.skipif(os.name != "nt", reason="a file held open refuses os.replace only on Windows")
def test_a_cached_mask_another_process_holds_is_masked_anyway(tmp_path):
    """RV10 L-12b with a REAL sharing violation: the cached mask is torn (a miss, recomputed) and
    held open, so `transients.write_component`'s `os.replace` onto it fails (WinError 5)."""
    ws, names = _workspace(tmp_path)
    first = _masks(ws, names)
    victim = sorted(SM.cache_dir(ws).glob("*.npz"))[0]
    victim.write_bytes(b"")
    with open(victim, "rb"):
        again = _masks(ws, names)
    rec = again.record()
    assert rec["state"] == SM.RECORD_APPLIED and rec["cause"] is None
    assert rec["images_masked"] == 4 and rec["cache_write_failed"] == 1
    assert {n: v["mask_sha1"] for n, v in again.masked.items()} == \
        {n: v["mask_sha1"] for n, v in first.masked.items()}
    assert [p.name for p in victim.parent.iterdir() if p.name.startswith(victim.name + ".")] == [], \
        "the failed write left no staging file beside the cache"
    healed = _masks(ws, names).record()
    assert healed["computed"] == 1 and "cache_write_failed" not in healed, "rewritten next time"


def test_a_mask_index_write_that_fails_keeps_the_solve_masked(tmp_path, monkeypatch):
    """The cache's index (`_merge_index`, which the surface stage's donor reads) is an
    optimisation too. At 4a95a0d a failed write raised out of the mask step: `masks-step-failed`."""
    ws, names = _workspace(tmp_path)

    def fails(*a, **k):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(SM, "write_json_atomic", fails)
    rec = _masks(ws, names, keyframe_ids={n: f"kf:{i}" for i, n in enumerate(names)}).record()
    assert rec["state"] == SM.RECORD_APPLIED and rec["images_masked"] == 4
    assert rec["cache_write_failed"] == 1


# ---------------------------------------------------------------------------
# L-12b, the same write in the surface stage's own transient masks
# ---------------------------------------------------------------------------


def test_write_component_leaves_no_staging_file_when_it_fails(tmp_path, monkeypatch):
    path = tmp_path / "x.gdsam.npz"
    real = os.replace

    def refuse(src, dst):
        raise PermissionError(13, "held")

    monkeypatch.setattr(T.os, "replace", refuse)
    with pytest.raises(PermissionError):
        T.write_component(path, {"k": 1}, np.zeros((H, W), bool), np.zeros((H, W), bool),
                          image_sha1="s")
    monkeypatch.setattr(T.os, "replace", real)
    assert list(tmp_path.iterdir()) == []


def test_the_surface_stages_masks_survive_a_cache_that_cannot_be_written(tmp_path, monkeypatch):
    """`ensure_transient_masks`' own `emit` wrote unguarded too. At 4a95a0d: `failed`, with the
    detector's masks for every keyframe thrown away."""
    from tests.test_world_builder_transients import HAND_KI, N_FRAMES, Stub, World, _ensure

    reference = _ensure(World(tmp_path / "ref"), Stub())
    world = World(tmp_path / "run")

    def fails(*a, **k):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(T, "write_component", fails)
    report = _ensure(world, Stub())
    assert report.state == T.STATE_OK and report.computed == N_FRAMES
    assert sorted(report.keys) == list(range(N_FRAMES))
    np.testing.assert_array_equal(report.mask(HAND_KI), reference.mask(HAND_KI))
    assert report.mask(HAND_KI).any() and not report.mask(4).any()
    assert report.frames_digest() == reference.frames_digest()
    rec = report.record()
    assert rec["frames_masked"] == N_FRAMES
    assert rec["cache_write_failed"] == N_FRAMES * len(T.TransientParams().components)
    assert "cache_write_failed" not in reference.record(), "only when it happened"


def test_a_mask_held_in_memory_is_packed_and_given_back_exactly():
    """A full disk fails every write: the masks kept in memory are bits, not bytes."""
    rng = np.random.default_rng(7)
    hand, phone = rng.random((37, 61)) > 0.5, rng.random((37, 61)) > 0.9
    packed = T.pack_masks(hand, phone)
    assert sum(a.nbytes for a in packed[:2]) == 2 * 37 * 8
    back = T.unpack_masks(packed)
    assert all(b.dtype == bool for b in back)
    np.testing.assert_array_equal(back[0], hand)
    np.testing.assert_array_equal(back[1], phone)


def test_a_donor_copy_that_cannot_be_written_is_not_a_failure(tmp_path, monkeypatch):
    """`_take_from_donor` wrote unguarded: a failure raised out of the surface stage's masks.
    Now the copy is simply not taken, and the mask is computed."""
    hand = np.zeros((H, W), bool)

    def donor(kid, component, params, shape):
        return hand, hand, {"origin": "solve"}

    def fails(*a, **k):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(T, "write_component", fails)
    assert T._take_from_donor(donor, "kf:0", "gdsam", T.TransientParams(), (H, W),
                              tmp_path / "x.npz", {"k": 1}) is False
