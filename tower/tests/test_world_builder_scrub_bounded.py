"""The client-safe scrubber runs in bounded time (review V12, RV12-B LOW-A and LOW-B).

`client_safe_detail` runs on every listing row and every status poll, and `re` keeps the GIL, so one slow call
stalls the whole Tower. At a5001ab a 100,000-character unbroken token took about a minute (the unanchored
backslash pattern, the per-character bracket count in `_given_back`, the dotted exception-name search), and
`_CLASS_PREFIX` backtracked exponentially on a 33-character run of capitals. The fixes keep every output on
text shorter than `SCRUB_MAX_CHARS`: a differential run over 62,366 texts, 138 of them real finalization texts,
found no difference (`RUN\\lead\\p310\\diff_scrub.out`). Longer text is cut, and review V13 (LOW-1) found the
first cut could split a user name or a path; `_bounded` removes those first, over the whole text.
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


def test_each_fixed_pattern_is_linear_on_its_own():
    """The cut bounds the whole scrub, so these check the fixes themselves (review V13, NOTE-2): undoing the
    backslash anchor, the once-only bracket count or the flat `_CLASS_PREFIX` fails here."""
    t0 = time.perf_counter()
    assert CP._UNQUOTED_PATHS[-1].findall("x" * 100_000) == []
    assert CP._given_back("C:\\x" + ")" * 100_000) == ("C:\\x", ")" * 100_000)
    assert CP._CLASS_PREFIX.search("Ab" + "A" * 60 + "!") is None
    assert time.perf_counter() - t0 < SECONDS


def test_spaced_text_under_user_directories_is_bounded():
    text = " x\\Users\\A B" * 8000       # the user-directory growth in `_scrub_paths` is quadratic on this
    t0 = time.perf_counter()
    CP.owner_facing_detail(text)
    CP.owner_facing_detail(text.replace("\\", "/"))
    assert time.perf_counter() - t0 < SECONDS


def _at_cut(pad_word: str, needle: str, offset: int, after: str = " more words") -> str:
    """`needle` starting `offset` characters before the cut (`SCRUB_MAX_CHARS - 3`), after a long path."""
    start = CP.SCRUB_MAX_CHARS - 3 - offset
    head = "C:\\" + "d" * 3000 + " "
    head += (pad_word * (start - len(head)))[:start - len(head) - 1] + " "
    assert len(head) == start
    return head + needle + after


@pytest.mark.parametrize(("text", "leak"), [
    ("C:\\" + "d" * 3988 + '",tvllo,more text', "tvll"),                   # no space before the cut
    (_at_cut("w ", "John Smith", 6), "john"),                               # a spaced account name split
    (_at_cut("w ", "tvllo", 2), "tv"),                                      # a name split mid-word
    (_at_cut("w ", "data/jdoe/report.txt", 8), "jdoe"),                     # a relative path split before .txt
    (_at_cut("w ", "'/data/x/Jane Doe/report.txt'", 16), "jane"),           # a quoted path losing its quote
    (_at_cut("w ", 'File "/srv/jdoe/x.py", line 3, in f', 20), "jdoe"),   # a frame losing its line number
])
def test_the_cut_leaves_no_part_of_a_name_or_a_path(monkeypatch, text, leak):
    monkeypatch.setattr(CP, "_user_names", lambda: {"tvllo", "John Smith"})
    assert len(text) > CP.SCRUB_MAX_CHARS
    row = CP.client_safe_finalization({"state": "complete", "detail": text})
    for out in (CP.client_safe_detail(text), row["detail"], CP.owner_facing_detail(text)):
        assert leak not in out.lower()


def test_a_traceback_with_a_long_message_keeps_its_exception():
    tb = "Traceback (most recent call last): torch.OutOfMemoryError: CUDA out of memory " + "z" * 3985
    assert CP.client_safe_detail(tb).startswith("torch.OutOfMemoryError: CUDA out of memory")


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
