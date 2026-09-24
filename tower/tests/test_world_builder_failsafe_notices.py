"""Review V8, M2a and M2 (general): no gate fail-safe is silent.

Module: `tower/world_builder/coherence_publish.py` (`publish_notice`, the gate record's
`retryable`). Contract: WORLD-BUILDER-COMPONENTS.md §2.2 (`masks-unavailable`,
`scale-unavailable`) and §2.5 (`gate.retryable`, `gate.cause`, `transients.*`).

Every published gated solve that is NOT "masks applied and metric scale available" carries
  * a row notice -- `publish_notice(summary)`, the sentence `world_finalize.py`,
    `world_build_session.py` and the finisher's re-gate put in the session's
    `finalization.detail`, the existing notice mechanism;
  * an owner hint -- who can fix it: the idle Tower (a re-gate), an owner (re-finish or
    re-capture), or an operator (then an owner);
  * a correct retryable flag -- `gate.retryable` only where a re-gate in place can change
    the outcome, `transients.retryable` only for a transient mask failure.

M2a. Scale unavailable WITH DEPTH IN HAND (fewer than `min_metric_fraction` of the
supported cameras have a metric ratio, the depth stage having run to the end) was
`retryable: False` with no notice at all. It stays NOT retryable -- the depth stage is
deterministic on the same keyframes (its misses are missing, undecodable or re-sized
images), so a re-gate in place reproduces the same shortfall and would only spend the
finisher's attempts -- and it now says so, naming what is missing and who can fix it.
"""

from __future__ import annotations

import pytest

from tests.test_world_builder_coherence_publish import TRIANGLE, _entries, _name, _run
from tower.world_builder import coherence_gate as CG
from tower.world_builder import coherence_publish as CP
from tower.world_builder import solve_masks as SM

APPLIED = {"state": "applied", "requested": True}


def _few_levels(n):
    def metric(solution, name_of, database_path, work):
        return {"metric_log": {_name(i): 0.0 for i in range(n)}}
    return metric


def _notice(result, transients=None):
    return CP.publish_notice({"gate": result.record,
                              "transients": transients if transients is not None else APPLIED})


# ---------------------------------------------------------------------------
# M2a: scale unavailable with depth in hand
# ---------------------------------------------------------------------------


def test_a_scale_shortfall_with_depth_in_hand_says_what_is_missing_and_who_can_fix_it(monkeypatch):
    """Before: `publish_notice` returned None for this record -- a permanent fail-safe the
    row never mentioned."""
    _, result = _run(monkeypatch, cross=TRIANGLE, metric_fn=_few_levels(10))   # 10 of 50
    assert result.record["metric_available"] is False
    assert result.record["depth"]["state"] == CP.DEPTH_OK
    notice = _notice(result)
    assert notice is not None
    assert "10 of 50" in notice, "what is missing: the cameras with a metric level"
    assert "re-capture" in notice, "who can fix it: an owner, with a new walk"
    assert "re-runs" not in notice, "nothing is promised that will not happen"
    assert notice == CP.NOTICE_SCALE_SHORT.format(
        why=result.record["evidence"]["metric_scale"])


def test_a_scale_shortfall_with_depth_in_hand_is_not_retryable(monkeypatch, tmp_path):
    """The choice, pinned: a re-gate would re-run a deterministic depth stage on the same
    keyframes and reproduce the shortfall, so nothing is owed unattended."""
    import json

    from scripts import world_finish_pending as wfp
    from tests.test_world_builder_coherence_publish import _gated_session

    _, result = _run(monkeypatch, cross=TRIANGLE, metric_fn=_few_levels(10))
    assert result.record["retryable"] is False and result.record["cause"] is None
    store = _gated_session(tmp_path, transients=APPLIED,
                           gate=json.loads(json.dumps(result.record)))
    assert CP.regate_owed(store, "w1", "s1") is None
    v = wfp.assess(store, "w1", "s1")
    assert v.code != "owed-regate" and not v.owed


def test_no_metric_ratio_at_all_with_depth_in_hand_is_the_same_notice(monkeypatch):
    _, result = _run(monkeypatch, cross=TRIANGLE, metric_fn=_few_levels(0))
    notice = _notice(result)
    assert notice is not None and "0 of 50" in notice and "re-capture" in notice


def test_a_depth_failure_still_owes_a_re_gate_and_says_so(monkeypatch):
    """Unchanged (review V7, H1b): the depth stage failing is retryable, the Tower re-runs."""
    def no_depth(*a, **kw):
        raise CP.DepthUnavailable("CUDA out of memory")

    _, result = _run(monkeypatch, cross=TRIANGLE, depth_runner=no_depth)
    assert result.record["retryable"] is True
    notice = _notice(result)
    assert "re-runs the gate" in notice and "re-capture" not in notice


# ---------------------------------------------------------------------------
# M2 (general): the masks fail-safe says why, and who can fix it
# ---------------------------------------------------------------------------


MASKS_CASES = {
    # transients record                                     -> the notice, and its owner
    "off": ({"state": "unavailable", "requested": False, "outcome": "off",
             "detail": SM.OFF_DETAIL}, CP.NOTICE_MASKS_OFF),
    "absent": (None, CP.NOTICE_MASKS_OFF),
    "no-gpu": ({"state": "unavailable", "requested": True, "outcome": "unavailable",
                "detail": "no CUDA device, and no CPU fallback", "cause": None,
                "retryable": False},
               CP.NOTICE_MASKS_UNAVAILABLE.format(why="no CUDA device, and no CPU fallback")),
    "detector-failed": ({"state": "unavailable", "requested": True, "outcome": "failed",
                         "detail": "the gdsam detector failed (RuntimeError: device lost)",
                         "cause": SM.CAUSE_DETECTOR_FAILED, "retryable": False},
                        CP.NOTICE_MASKS_UNAVAILABLE.format(
                            why="the gdsam detector failed (RuntimeError: device lost)")),
    "gpu-oom": ({"state": "unavailable", "requested": True, "outcome": "failed",
                 "detail": "the oneformer detector failed (OutOfMemoryError)",
                 "cause": SM.CAUSE_GPU_OOM, "retryable": True}, CP.NOTICE_MASKS_OOM),
    "fallback-rule": ({"state": "partial", "requested": True, "outcome": "ok",
                       "rule_fallback": "union was requested but only oneformer could run",
                       "cause": None, "retryable": False, "images": 50,
                       "images_masked": 50, "images_unmasked": 0}, CP.NOTICE_MASKS_FALLBACK),
    "fallback-rule-oom": ({"state": "partial", "requested": True, "outcome": "ok",
                           "rule_fallback": "union was requested but only oneformer could run",
                           "cause": SM.CAUSE_GPU_OOM, "retryable": True}, CP.NOTICE_MASKS_OOM),
    "unexcluded-images": ({"state": "partial", "requested": True, "outcome": "ok",
                           "rule_fallback": None, "cause": None, "retryable": False,
                           "images": 50, "images_masked": 47, "images_unmasked": 3,
                           "images_excluded": 0},
                          CP.NOTICE_MASKS_PARTIAL.format(unmasked=3, images=50)),
}


@pytest.mark.parametrize("case", sorted(MASKS_CASES))
def test_every_masks_fail_safe_carries_a_notice_with_its_owner(monkeypatch, case):
    """Before: only the GPU-out-of-memory cause said anything on the row."""
    transients, expected = MASKS_CASES[case]
    _, result = _run(monkeypatch, cross=TRIANGLE,
                     transients=("state", (transients or {}).get("state")) if transients else None)
    assert result.record["masks_applied"] is False
    assert [e["reasons"] for e in _entries(result)][1] == [CG.REASON_MASKS_UNAVAILABLE]
    # Never a re-gate: the masks are the SOLVE's; a re-gate in place cannot mask it.
    assert result.record["retryable"] is False and result.record["cause"] is None
    notice = CP.publish_notice({"gate": result.record, "transients": transients})
    assert notice == expected
    assert any(who in notice for who in ("an owner can re-finish", "an operator can"))


@pytest.mark.parametrize("case", sorted(MASKS_CASES))
def test_the_transient_retryable_flag_is_only_for_a_gpu_out_of_memory(case):
    transients, _ = MASKS_CASES[case]
    assert bool((transients or {}).get("retryable")) is (
        (transients or {}).get("cause") == SM.CAUSE_GPU_OOM)


def test_a_notice_names_every_fail_safe_that_applies(monkeypatch):
    """Masks not applied AND a scale shortfall with depth in hand: both, in §2.2's order."""
    transients = MASKS_CASES["no-gpu"][0]
    _, result = _run(monkeypatch, cross=TRIANGLE, transients=("state", "unavailable"),
                     metric_fn=_few_levels(10))
    notice = CP.publish_notice({"gate": result.record, "transients": transients})
    masks = MASKS_CASES["no-gpu"][1]
    scale = CP.NOTICE_SCALE_SHORT.format(why=result.record["evidence"]["metric_scale"])
    assert notice == f"{masks}; {scale}"


def test_masks_applied_and_metric_available_say_nothing(monkeypatch):
    _, result = _run(monkeypatch, cross=TRIANGLE)
    assert result.record["masks_applied"] is True and result.record["metric_available"] is True
    assert _notice(result) is None


def test_an_excluded_image_is_not_a_fail_safe_and_says_nothing(monkeypatch):
    """M2b's other half: an image excluded from the solve leaves it `applied`."""
    _, result = _run(monkeypatch, cross=TRIANGLE)
    transients = dict(APPLIED, images=50, images_masked=49, images_unmasked=1,
                      images_excluded=1)
    assert CP.publish_notice({"gate": result.record, "transients": transients}) is None


@pytest.mark.parametrize("transients", [None, {"state": "unavailable", "requested": False},
                                        {"state": "unavailable", "requested": True,
                                         "detail": "no CUDA device", "cause": None}])
def test_an_ungated_solve_is_told_nothing_new(transients):
    """No gate, no fail-safe: exactly today's `None` (the OOM sentence aside, which is
    today's too)."""
    assert CP.publish_notice({"transients": transients, "gate": None}) is None
    assert CP.publish_notice({"transients": transients}) is None


def test_a_failed_gate_keeps_its_own_notice(monkeypatch):
    def broken(*a, **kw):
        raise ZeroDivisionError("a bug")

    _, result = _run(monkeypatch, metric_fn=broken)
    notice = _notice(result)
    assert notice == CP.NOTICE_GATE_FAILED.format(why="ZeroDivisionError: a bug")


def test_every_notice_is_one_line_with_no_metric_figure():
    """The row's sentence: one line; and contract §2.4 rule 6 -- no metre figure."""
    for text in (CP.NOTICE_MASKS_OOM, CP.NOTICE_MASKS_OFF, CP.NOTICE_MASKS_UNAVAILABLE,
                 CP.NOTICE_MASKS_FALLBACK, CP.NOTICE_MASKS_PARTIAL, CP.NOTICE_SCALE_SHORT,
                 CP.NOTICE_REGATE, CP.NOTICE_GATE_FAILED):
        assert "\n" not in text and " m " not in text and "metre" not in text
