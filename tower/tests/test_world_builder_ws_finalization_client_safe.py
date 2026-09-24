"""The status channel's `lifecycle.finalization` is exactly as client-safe as the `GET /worlds` row (review V11
MED-B; CON3 OPEN): `detail` AND `notice` are scrubbed and bounded, and a clean record is the same object."""

from tower.results import world_builder as RW
from tower.world_builder import coherence_publish as CP

RAW = ("Traceback (most recent call last):\n  File \"C:\\Users\\someone\\Projects\\x.py\", line 3, in f\n"
       "RuntimeError: CUDA out of memory at C:\\Users\\someone\\model.bin")


def test_a_raw_notice_is_scrubbed_on_the_status_channel_as_on_the_row():
    record = {"state": "complete", "final_solve": "solved", "detail": RAW, "notice": RAW}
    ws = RW._client_safe_finalization(record)
    row = CP.client_safe_finalization(record)
    assert ws == row
    for key in ("detail", "notice"):
        assert "\n" not in ws[key] and "someone" not in ws[key] and "C:\\" not in ws[key]
    assert record["notice"] == RAW, "the record on disk is not touched"


def test_a_clean_record_is_the_same_object():
    notice = next(iter(CP.NOTICE_SENTENCES.values())).format(unmasked=1, images=2, excluded=1)
    record = {"state": "complete", "final_solve": "solved", "detail": None, "notice": notice}
    assert RW._client_safe_finalization(record) is record
    assert RW._client_safe_finalization(None) is None


def test_the_text_is_bounded():
    long = "x " * 2000
    ws = RW._client_safe_finalization({"detail": long, "notice": long})
    assert len(ws["detail"]) <= CP.FINALIZATION_TEXT_MAX_CHARS
    assert len(ws["notice"]) <= CP.FINALIZATION_TEXT_MAX_CHARS
