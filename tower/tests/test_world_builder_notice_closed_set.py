"""`finalization.notice` is a closed set of Tower-written sentences (review V9, M-4 and M-8; manager 025, decision 3).

Module: `tower/world_builder/coherence_publish.py` (`publish_notice`, `publish_detail`, `NOTICE_SENTENCES`,
`notice_cause_phrase`, `regate_clause`, `masks_notice`). Contract WORLD-BUILDER-COMPONENTS.md §3.1: `<why>` is "a
short Tower-written phrase", and "`detail` can carry an error string; `notice` never does". The phone shows the
notice word for word (the Mac tip de1b045 turns paths, tracebacks, a leading exception, newlines and more than 700
characters into a generic sentence, so the set must pass that guard verbatim).

Before: `publish_notice` put the raw detail in the parentheses -- "the evidence gate failed (DatabaseError: file
is not a database)", a CUDA out-of-memory message with GiB figures, a `C:\\Users\\...` path (RV9-E, RV9-F).
The raw records below are the ones the product really writes (V9 `notice\\` and `m2\\` probes).
"""

from __future__ import annotations

import copy
import re

import pytest

from tower.world_builder import coherence_publish as CP
from tower.world_builder import solve_masks as SM

APPLIED = {"state": "applied", "requested": True}
GATE_OK = {"state": "applied", "masks_applied": True, "metric_available": True, "retryable": False,
           "attach": True, "depth": {"state": "ok"}}
GATE_NO_MASKS = {"state": "applied", "masks_applied": False, "metric_available": True, "retryable": False,
                 "depth": {"state": "ok"}}


def _depth_lost(detail):
    return {"state": "applied", "masks_applied": True, "metric_available": False, "retryable": True,
            "cause": "depth-unavailable", "depth": {"state": "unavailable", "detail": detail}}


def _gate_failed(detail):
    return {"state": "failed", "retryable": True, "cause": "gate-failed", "detail": detail}


def _masks(detail, *, outcome="unavailable", cause=None, **kw):
    return {"state": "unavailable", "requested": True, "outcome": outcome, "detail": detail, "cause": cause,
            "retryable": cause == SM.CAUSE_GPU_OOM, **kw}


CUDA_OOM = (r"RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 8.00 GiB total capacity; "
            r"6.10 GiB already allocated) at C:\Users\tvllo\Projects\Glasses\tower\.venv\Lib\moge\model.py:411")
TORCH_OOM = ("OutOfMemoryError: CUDA out of memory. Tried to allocate 1.20 GiB. GPU 0 has a total capacity of "
             "8.00 GiB of which 312.00 MiB is free.")
DB_ERROR = "DatabaseError: file is not a database"
STATE_DICT = ('RuntimeError: Error(s) in loading state_dict for MoGeModel:\n\tMissing key(s) in state_dict: '
              '"head.0.weight"')
TRACEBACK = 'Traceback (most recent call last):\n  File "C:/x/y.py", line 1, in <module>\nValueError: boom'
MISSING_FRAME = (r"FileNotFoundError: [Errno 2] No such file or directory: "
                 r"'C:\\Users\\tvllo\\Projects\\Glasses\\tower\\data\\worlds\\7d31e8d7\\solve\\s1\\images\\00000042.jpg'")
HF_CACHE = (r"transient detector model 'IDEA-Research/grounding-dino-base' (backend 'gdsam') is not in the Hugging "
            r"Face cache (C:\Users\tvllo\.cache\huggingface\hub) and could not be downloaded: OSError. Connect this "
            r"machine to the internet for its first build (...), or pre-seed the cache, then rebuild")
WIN_ERROR = (r"the gdsam detector failed (PermissionError: [WinError 5] Access is denied: "
             r"'C:\Users\tvllo\AppData\Local\Temp\tmpux4cczst\transients\00000001.a2d7cfff4fb3.gdsam.npz.p37660.tmp' "
             r"-> 'C:\Users\tvllo\AppData\Local\Temp\tmpux4cczst\transients\00000001.a2d7cfff4fb3.gdsam.npz')")
GROUNDING = r"the grounding_dino detector failed (OSError: [Errno 22] Invalid argument: 'C:\Users\tvllo\.cache\x')"

# (transients, gate) as the product records them -> the cause the notice must name
CASES = {
    "gate failed: DatabaseError": (APPLIED, _gate_failed(DB_ERROR), ["gate-failed-database"]),
    "gate failed: KeyError": (APPLIED, _gate_failed("KeyError: 'kf_000123'"), ["gate-failed"]),
    "gate failed: a traceback": (APPLIED, _gate_failed(TRACEBACK), ["gate-failed"]),
    "gate failed: state dict": (APPLIED, _gate_failed(STATE_DICT), ["gate-failed"]),
    "gate failed: MemoryError": (APPLIED, _gate_failed("MemoryError: "), ["gate-failed-memory"]),
    "depth: CUDA OOM, GiB and a path": (APPLIED, _depth_lost(CUDA_OOM), ["depth-gpu-oom"]),
    "depth: torch OOM, MiB": (APPLIED, _depth_lost(TORCH_OOM), ["depth-gpu-oom"]),
    "depth: moge not installed": (APPLIED, _depth_lost("DepthModelUnavailable: moge is not installed"),
                                  ["depth-model-missing"]),
    "depth: stopped": (APPLIED, _depth_lost("stopped: the depth stage was stopped after 12 frames"),
                       ["depth-stopped"]),
    "depth: lock held": (APPLIED, _depth_lost("another surface build of this session holds its lock"),
                         ["depth-surface-busy"]),
    "depth: no intrinsics": (APPLIED, _depth_lost("session has no intrinsics"), ["depth-no-intrinsics"]),
    "depth: no detail": (APPLIED, dict(_depth_lost(None), depth={"state": "unavailable"}), ["depth-unavailable"]),
    "masks: the mask step raised, with a path": (SM.failed_record(MISSING_FRAME), GATE_NO_MASKS,
                                                 ["masks-step-failed"]),
    "masks: weights not cached": (_masks(HF_CACHE), GATE_NO_MASKS, ["masks-not-installed"]),
    "masks: a refused cache write (WinError)": (_masks(WIN_ERROR, outcome="failed",
                                                       cause=SM.CAUSE_DETECTOR_FAILED), GATE_NO_MASKS,
                                                ["masks-detector-failed"]),
    "masks: grounding dino failed": (_masks(GROUNDING, outcome="failed", cause=SM.CAUSE_DETECTOR_FAILED),
                                     GATE_NO_MASKS, ["masks-detector-failed"]),
    "masks: no CUDA": (_masks("no CUDA device, and no CPU fallback: the solver's transient masks run on the GPU "
                              "only"), GATE_NO_MASKS, ["masks-no-gpu"]),
    "masks: torch not importable": (_masks("torch is not importable (ImportError: DLL load failed while "
                                           "importing _C)"), GATE_NO_MASKS, ["masks-not-installed"]),
    "masks: GPU out of memory": (_masks("the oneformer detector failed (OutOfMemoryError: CUDA out of memory. "
                                        "Tried to allocate 20.00 MiB)", outcome="failed",
                                        cause=SM.CAUSE_GPU_OOM), GATE_NO_MASKS, ["masks-gpu-oom"]),
    "masks and a failed gate": (_masks("no CUDA device"), _gate_failed(DB_ERROR),
                                ["masks-no-gpu", "gate-failed-database"]),
    "masks partial": ({"state": "partial", "requested": True, "images": 50, "images_unmasked": 3,
                       "rule_fallback": None}, GATE_NO_MASKS, ["masks-partial"]),
    "excluded images (M-8)": (dict(APPLIED, images=400, images_excluded=12, exclusion_notice_due=True), GATE_OK,
                              ["masks-excluded"]),
    "no image masked (M-8)": (_masks(r"no solver image could be masked: 9 unreadable, 1 held by another process "
                                     r"(C:\x)", images=10, images_excluded=10, none_masked=True,
                                     exclusion_notice_due=True), GATE_NO_MASKS, ["masks-none"]),
    "scale short": (APPLIED, {"state": "applied", "masks_applied": True, "metric_available": False,
                              "retryable": False, "depth": {"state": "ok"},
                              "evidence": {"metric_scale": "10 of 50 supported cameras with a ratio (20%; "
                                                           "available at >= 50%)"}}, ["scale-short"]),
    "consensus deferred": (APPLIED, dict(GATE_OK, retryable=True, cause="consensus-deferred"),
                           ["consensus-deferred"]),
    # the rest of the set, so every sentence is produced at least once
    "masks off": ({"state": "unavailable", "requested": False, "outcome": "off", "detail": SM.OFF_DETAIL},
                  GATE_NO_MASKS, ["masks-off"]),
    "masks fallback rule": ({"state": "partial", "requested": True, "images": 50, "images_unmasked": 0,
                             "rule_fallback": "union was requested but only oneformer could run (gdsam: x)"},
                            GATE_NO_MASKS, ["masks-fallback"]),
    "masks partial, uncounted": ({"state": "partial", "requested": True, "rule_fallback": None}, GATE_NO_MASKS,
                                 ["masks-partial-uncounted"]),
    "excluded images, uncounted": (dict(APPLIED, exclusion_notice_due=True), GATE_OK,
                                   ["masks-excluded-uncounted"]),
    "masks: no image masked, before M-8": (_masks("no solver image could be masked", outcome="ok"),
                                           GATE_NO_MASKS, ["masks-no-image"]),
    "masks: an unknown refusal": (_masks("something the Tower never wrote before: x/y"), GATE_NO_MASKS,
                                  ["masks-unavailable"]),
    "depth: no camera": (APPLIED, _depth_lost("the solve has no camera"), ["depth-no-camera"]),
}
RAW = {name: case for name, case in CASES.items()
       if (case[1].get("detail") or (case[1].get("depth") or {}).get("detail") or case[0].get("detail"))}


def _notice(case):
    transients, gate, _ = case
    return CP.publish_notice({"transients": transients, "gate": gate})


def _closed_set_pattern():
    sentence = []
    for text in CP.NOTICE_SENTENCES.values():
        rx = re.escape(text)
        for field in ("unmasked", "images", "excluded"):
            rx = rx.replace(re.escape("{" + field + "}"), r"\d+")
        sentence.append(rx)
    one = "(?:" + "|".join(sentence) + ")"
    return re.compile(f"{one}(?:; {one})*")


# ---------------------------------------------------------------------------
# M-4: no raw text reaches the notice


@pytest.mark.parametrize("name", sorted(CASES))
def test_no_notice_carries_an_error_string_a_path_or_a_figure(name):
    notice = _notice(CASES[name])
    assert isinstance(notice, str) and notice
    for bad in ("\\", "/", "\n", "\t", "Error", "Traceback", "GiB", "MiB", "Errno", ".py", ".jpg", ".npz",
                "Users", "tvllo", "C:", '"'):
        assert bad not in notice, (name, bad, notice)
    assert "'" not in notice.replace("'s ", ""), "no quoted name; the set's own apostrophes are possessives"
    assert len(notice) < 700
    assert not re.match(r"^[A-Z]\w*(Error|Exception)\b", notice), "no leading exception"
    assert not re.search(r"\d", re.sub(r"\d+ of \d+ images", "", notice)), "the only numbers are image counts"


@pytest.mark.parametrize("name", sorted(CASES))
def test_every_notice_is_built_from_the_closed_set(name):
    """Checked against the published set by pattern, not by asking the code which causes it chose."""
    notice = _notice(CASES[name])
    assert _closed_set_pattern().fullmatch(notice), notice
    assert CP.notice_causes({"transients": CASES[name][0], "gate": CASES[name][1]}) == CASES[name][2]


def test_the_cases_produce_every_sentence_of_the_set():
    assert {c for case in CASES.values() for c in case[2]} == set(CP.NOTICE_SENTENCES)


def test_the_closed_set_passes_the_phones_guard_verbatim():
    for cause, text in CP.NOTICE_SENTENCES.items():
        filled = text.format(unmasked=12, images=400, excluded=12)
        assert "/" not in filled and "\\" not in filled and "\n" not in filled, cause
        assert len(filled) < 300, cause
        assert not re.match(r"^[A-Z]\w*(Error|Exception)\b", filled)
        assert set(CP.NOTICE_PHRASES) == set(CP.NOTICE_SENTENCES) == set(CP.NOTICE_CLAUSES)
    longest = max(len(CP.NOTICE_SENTENCES[c]) for c in CP.NOTICE_SENTENCES if c.startswith("masks-"))
    longest_gate = max(len(CP.NOTICE_SENTENCES[c]) for c in CP.NOTICE_SENTENCES if not c.startswith("masks-"))
    assert longest + len(CP.NOTICE_SENTENCES["masks-excluded"]) + longest_gate + 20 < 700, "the longest notice"


@pytest.mark.parametrize("name", sorted(RAW))
def test_the_raw_text_stays_in_the_record_and_the_detail_quotes_it_client_safe(name):
    """The record keeps the raw text; the finalization's `detail` quotes it as `client_safe_detail` makes it --
    the exception class and a short message, no path, one line (review V10, MED-5: `detail` reaches the phone and
    the unauthenticated socket, so the P3.6 "raw text in the detail" was a leak)."""
    transients, gate, _ = RAW[name]
    before = copy.deepcopy((transients, gate))
    detail = CP.publish_detail({"transients": transients, "gate": gate})
    raw = gate.get("detail") or (gate.get("depth") or {}).get("detail") or transients.get("detail")
    if CP.notice_causes({"transients": transients, "gate": gate})[0] not in ("masks-gpu-oom", "masks-off"):
        assert CP.client_safe_detail(raw, max_chars=CP.WHY_MAX_CHARS) in detail, "the diagnostics are the detail's"
    assert "\n" not in detail and "Traceback" not in detail and not re.search(r"[A-Za-z]:[\\/]", detail)
    assert "\\" not in detail and "tvllo" not in detail
    assert (transients, gate) == before, "the records keep their text"


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_cause_phrase_is_a_fixed_phrase(name):
    """REF's given-up sentence is composed from these (`notice_cause_phrase`, `regate_clause`)."""
    transients, gate, causes = CASES[name]
    phrases = set(CP.NOTICE_PHRASES.values()) | {""}
    assert CP.notice_cause_phrase(gate) in phrases and CP.notice_cause_phrase(transients) in phrases
    gate_causes = [c for c in causes if not c.startswith("masks-")]
    if gate_causes:
        assert CP.notice_cause_phrase(gate) == CP.NOTICE_PHRASES[gate_causes[0]]
        assert CP.regate_clause(gate) == CP.NOTICE_CLAUSES[gate_causes[0]]
    masks_causes = [c for c in causes if c.startswith("masks-")]
    if masks_causes:
        assert CP.notice_cause_phrase(transients) == CP.NOTICE_PHRASES[masks_causes[0]]
        assert CP.masks_notice(transients, gate) == "; ".join(
            CP.NOTICE_SENTENCES[c].format(unmasked=transients.get("images_unmasked"), images=transients.get("images"),
                                          excluded=transients.get("images_excluded")) for c in masks_causes)


def test_a_real_re_gate_on_an_unreadable_database_puts_no_exception_in_the_notice(tmp_path, monkeypatch):
    """RV9-E F1a: the real `regate_published` and gate on a solve whose database is not a database. The notice
    carried "DatabaseError: file is not a database" verbatim; now it says so in the Tower's words, and the
    exception is in the detail and in `gate.detail`."""
    from tests.test_world_builder_solve_consensus import PIECES, SID, _candidate, _keyframes, _Store
    from tower.world_builder import global_solve as GS
    from tower.world_builder.store import WorldStore

    store = WorldStore(tmp_path)
    n = 30 + sum(PIECES.values())
    monkeypatch.setattr(store, "read_session", _Store().read_session)
    monkeypatch.setattr(store, "read_keyframes", lambda w, s: _keyframes(n))

    def no_depth(*a, **k):
        raise CP.DepthUnavailable("session has no intrinsics")

    monkeypatch.setattr(CP, "run_gate_depth", no_depth)
    ws = GS.workspace_for(store, "w1", SID)
    published = _candidate(tuple(PIECES))
    published.gate = _gate_failed("ZeroDivisionError: a bug")
    GS.write_solution(ws, published)
    ws.database_path.write_bytes(b"this is not a COLMAP database, and sqlite says so" * 100)
    out = CP.regate_published(store, "w1", SID)
    assert out["gate"]["state"] == CP.GATE_STATE_FAILED
    assert out["notice"] == CP.NOTICE_SENTENCES["gate-failed-database"]
    assert "DatabaseError" in out["detail"]
    assert "DatabaseError" in GS.load_solution(store, "w1", SID).gate["detail"]


# ---------------------------------------------------------------------------
# M-8: the sentences for excluded images (the masks stage decides when they are due)


def test_enough_excluded_images_say_how_many_and_that_an_owner_can_re_finish():
    t = dict(APPLIED, images=400, images_excluded=12, exclusion_notice_due=True)
    assert CP.publish_notice({"transients": t, "gate": GATE_OK}) == (
        "12 of 400 images could not be masked and were left out of the solve; an owner can re-finish this walk")


def test_no_masked_image_is_said_with_its_owner_not_the_detectors():
    """RV9-F: when every image failed, the sentence told an operator to make the DETECTOR run."""
    t = _masks("no solver image could be masked", images=10, images_excluded=10, none_masked=True,
               exclusion_notice_due=True)
    notice = CP.publish_notice({"transients": t, "gate": GATE_NO_MASKS})
    assert notice == ("masks were not applied (no solver image could be masked); an owner can re-finish "
                      "this walk")
    assert "operator" not in notice and "detector" not in notice


@pytest.mark.parametrize("transients", [
    dict(APPLIED, images=400, images_excluded=12),                          # no flag: today's silence
    dict(APPLIED, images=400, images_excluded=2, exclusion_notice_due=False),
])
def test_an_exclusion_that_is_not_due_says_nothing(transients):
    assert CP.publish_notice({"transients": transients, "gate": GATE_OK}) is None


def test_absent_keys_keep_todays_sentence_for_a_walk_no_image_of_which_was_masked():
    t = _masks("no solver image could be masked")
    assert CP.publish_notice({"transients": t, "gate": GATE_NO_MASKS}) == (
        "masks were not applied (no solver image could be masked); an operator can make the transient detector "
        "run on this Tower, then an owner can re-finish this walk")


def test_an_ungated_solve_is_told_nothing_about_excluded_images():
    t = dict(APPLIED, images=400, images_excluded=12, exclusion_notice_due=True)
    assert CP.publish_notice({"transients": t, "gate": None}) is None
    assert CP.publish_notice({"transients": t}) is None


# ---------------------------------------------------------------------------
# V9 LOW: a failed gate keeps the masks cause


def test_a_failed_gate_keeps_the_masks_cause():
    notice = CP.publish_notice({"transients": _masks("no CUDA device"), "gate": _gate_failed(DB_ERROR)})
    assert notice.startswith("masks were not applied (no GPU could run the transient detector)")
    assert notice.endswith("the evidence gate failed (the solve's feature database could not be read); "
                           "the Tower re-runs it when it is idle")
    assert CP.publish_notice({"transients": APPLIED, "gate": _gate_failed(DB_ERROR)}) == (
        "the evidence gate failed (the solve's feature database could not be read); the Tower re-runs it when "
        "it is idle")
