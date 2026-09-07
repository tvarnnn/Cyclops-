"""`write_json_atomic` against a reader that holds the destination open.

Windows refuses `os.replace` onto a file any handle has open. The Tower's
web process reads the derived tree while the builder rewrites it, and on
2026-09-06 a live builder died inside `write_derived` (poses.json new,
points.json old, no temp file left) when a reader -- descheduled under a
solver using 18 of 20 cores -- held points.json longer than the 60 ms
retry budget of the day. The writer is the mapping session; it must ride
out a slow reader, and it must still give up on a parked one.
"""

import json
import sys
import threading
import time

import pytest

from tower import storage


def _polling_reader(path, hold_s, idle_s, stop):
    """The shape of the Tower's reader: a route that opens the file a few
    times a second and, under load, may hold it far longer than a read
    takes. Not a reader that reopens the file every millisecond -- no
    replace can win against that on Windows, and nothing here does it."""
    while not stop.is_set():
        try:
            with path.open("rb") as handle:
                handle.read(64)
                time.sleep(hold_s)  # a reader that lost the CPU mid-read
        except PermissionError:
            pass  # a replace won the race against our open; go again
        time.sleep(idle_s)


def test_a_reader_descheduled_for_150ms_does_not_fail_the_writer(tmp_path):
    path = tmp_path / "points.json"
    payload = {"points": [[i, i * 0.5, i * 0.25] for i in range(20000)]}
    storage.write_json_atomic(path, payload)
    stop = threading.Event()
    reader = threading.Thread(target=_polling_reader, args=(path, 0.15, 0.1, stop), daemon=True)
    reader.start()
    try:
        for i in range(20):
            storage.write_json_atomic(path, {"i": i, **payload})
    finally:
        stop.set()
        reader.join(2)
    assert json.loads(path.read_text(encoding="utf-8"))["i"] == 19
    assert not path.with_name(path.name + storage.TEMP_SUFFIX).exists()


@pytest.mark.skipif(sys.platform != "win32", reason="POSIX replaces over open readers")
def test_a_parked_reader_still_fails_the_writer_within_the_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "REPLACE_BUDGET_S", 0.05)
    path = tmp_path / "held.json"
    storage.write_json_atomic(path, {"a": 1})
    release = threading.Event()

    def park():
        with path.open("rb"):
            release.wait(5.0)

    parked = threading.Thread(target=park, daemon=True)
    parked.start()
    time.sleep(0.02)
    started = time.perf_counter()
    try:
        with pytest.raises(PermissionError):
            storage.write_json_atomic(path, {"a": 2})
        assert time.perf_counter() - started < 1.0
    finally:
        release.set()
        parked.join(2)
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1}
    assert not path.with_name(path.name + storage.TEMP_SUFFIX).exists()
