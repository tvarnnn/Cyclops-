"""The client-safe scrubber runs in bounded time (review V12, RV12-B LOW-A and LOW-B).

`client_safe_detail` runs on every listing row and every status poll, and `re` keeps the GIL, so one slow call
stalls the whole Tower. At a5001ab a 100,000-character unbroken token took about a minute (the unanchored
backslash pattern, the per-character bracket count in `_given_back`, the dotted exception-name search), and
`_CLASS_PREFIX` backtracked exponentially on a 33-character run of capitals. The fixes keep every output on
text shorter than `SCRUB_MAX_CHARS`: a differential run over 62,366 texts, 138 of them real finalization texts,
found no difference (`RUN\\lead\\p310\\diff_scrub.out`).
"""
from __future__ import annotations

import time

import pytest

from tower.world_builder import coherence_publish as CP

# Generous: each took 20 to 60+ seconds before the fix, and takes well under half a second after it.
SECONDS = 3.0

ADVERSARIAL = {
    "unbroken run": "x" * 100_000,
    "unclosed frame quote": 'File "' + "x" * 100_000,
    "a path and many closing brackets": "C:\\x" + ")" * 100_000,
    "dotted run": "a." * 50_000,
    "traceback header and a dotted run": "Traceback (most recent call last)" + "A." * 50_000,
    "many lines": ("x" * 1000 + "\n") * 100,
}


@pytest.mark.parametrize("text", list(ADVERSARIAL.values()), ids=list(ADVERSARIAL))
def test_a_long_text_is_made_client_safe_in_bounded_time(text):
    t0 = time.perf_counter()
    row = CP.client_safe_finalization({"state": "complete", "detail": text, "notice": text})
    owner = CP.owner_facing_detail(text)
    assert time.perf_counter() - t0 < SECONDS
    assert len(row["detail"]) <= CP.FINALIZATION_TEXT_MAX_CHARS
    assert len(row["notice"]) <= CP.FINALIZATION_TEXT_MAX_CHARS
    for out in (row["detail"], row["notice"], owner):
        assert "\\" not in out and "\n" not in out


def test_a_run_of_capitals_before_a_colon_does_not_backtrack():
    t0 = time.perf_counter()
    for tail in ("!", ": x", ""):
        CP.owner_facing_detail("Ab" + "A" * 60 + tail)
    assert time.perf_counter() - t0 < SECONDS


@pytest.mark.parametrize(("raw", "owner"), [
    ("RuntimeError: CUDA out of memory", "CUDA out of memory"),
    ("DepthModelUnavailable: no depth model", "no depth model"),
    ("torch.OutOfMemoryError: CUDA out of memory", "CUDA out of memory"),
    ("depth failed (KeyboardInterrupt)", "depth failed"),
    ("an ordinary sentence: nothing to drop", "an ordinary sentence: nothing to drop"),
])
def test_class_names_still_leave_owner_facing_text(raw, owner):
    assert CP.owner_facing_detail(raw) == owner


@pytest.mark.parametrize(("path", "kept", "given"), [
    ("C:\\Program Files (x86)", "C:\\Program Files (x86)", ""),
    ("C:\\x.py);", "C:\\x.py", ");"),
    ("C:\\x" + ")" * 5, "C:\\x", ")" * 5),
    ("C:\\a(b)).", "C:\\a(b)", ")."),
    ("C:\\a[b]}]:", "C:\\a[b]", "}]:"),
])
def test_given_back_returns_what_it_did_before(path, kept, given):
    assert CP._given_back(path) == (kept, given)


def test_text_past_the_bound_is_cut_at_a_word_and_a_traceback_keeps_its_exception():
    words = "word " * 2000
    out = CP.client_safe_detail(words)
    assert len(out) <= CP.SCRUB_MAX_CHARS and out.endswith("...")
    assert CP.client_safe_detail(words[:200]) == words[:200]      # short clean text is untouched
    tb = "Traceback (most recent call last):" + " x" * 5000 + " ValueError: bad value"
    assert CP.client_safe_detail(tb) == "ValueError: bad value"
