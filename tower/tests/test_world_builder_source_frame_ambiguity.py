"""A keyframe is never given another capture's frame by name (`global_solve._source_frame`; PF's OPEN).

The capture-directory fallback finds a raw frame by its FILE NAME, and the captures of a reconnect chain
restart their numbering (the live walk adc75972: three captures, names restarting). A name held by more
than one of the given capture directories is therefore not a frame: the keyframe's own stored (redacted)
copy is used, and the solve records how many. One directory, or a name in exactly one directory, is found
exactly as before; `sources.json` is asked first, and is never ambiguous.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from tests.test_world_builder_solve_masks import (  # noqa: F401 -- fixtures
    HEIGHT,
    N,
    WIDTH,
    _solve,
    _solution_json,
    colmap,
    session,
)
from tower.world_builder import global_solve as GS


def _old_source_frame(keyframe, session_dir, capture_dirs, sources=None):
    """`_source_frame` before this change, for the byte-identity cases."""
    recorded = GS.resolve_source_path((sources or {}).get(keyframe.keyframe_id))
    if recorded is not None and recorded.is_file():
        return recorded
    name = GS.keyframe_image_name(keyframe)
    for capture_dir in capture_dirs:
        for candidate in (Path(capture_dir) / "frames" / name, Path(capture_dir) / name):
            if candidate.is_file():
                return candidate
    return session_dir / keyframe.image_relpath


def _kf(seq=3):
    return SimpleNamespace(keyframe_id=f"s:{seq:08d}", image_relpath=f"images/{seq:08d}.jpg")


def _put(path: Path, data=b"frame"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_a_name_in_two_capture_directories_is_the_stored_keyframe(tmp_path):
    a, b = tmp_path / "capA", tmp_path / "capB"
    _put(a / "frames" / "00000003.jpg", b"A's frame 3")
    _put(b / "frames" / "00000003.jpg", b"B's frame 3 -- another moment")
    ambiguous = []
    got = GS._source_frame(_kf(), tmp_path / "session", [a, b], {}, ambiguous)
    assert got == tmp_path / "session" / "images" / "00000003.jpg"
    assert ambiguous == ["s:00000003"]
    # the same with the name at a directory's top level in one of them
    (b / "frames" / "00000003.jpg").unlink()
    _put(b / "00000003.jpg")
    assert GS._source_frame(_kf(), tmp_path / "session", [a, b], {}) == \
        tmp_path / "session" / "images" / "00000003.jpg"


def test_sources_json_is_asked_first_and_is_never_ambiguous(tmp_path):
    a, b = tmp_path / "capA", tmp_path / "capB"
    _put(a / "frames" / "00000003.jpg")
    _put(b / "frames" / "00000003.jpg")
    own = _put(tmp_path / "raw" / "exact.jpg", b"this keyframe's own frame")
    ambiguous = []
    got = GS._source_frame(_kf(), tmp_path / "session", [a, b], {"s:00000003": str(own)}, ambiguous)
    assert got == own and ambiguous == []


LAYOUTS = {
    # (files in capA, files in capB): every case where no name is held twice
    "one dir, frames/": ({"frames/00000003.jpg"}, None),
    "one dir, top level": ({"00000003.jpg"}, None),
    "one dir, both places": ({"frames/00000003.jpg", "00000003.jpg"}, None),
    "one dir, absent": ({"frames/00000009.jpg"}, None),
    "two dirs, only the first": ({"frames/00000003.jpg"}, {"frames/00000004.jpg"}),
    "two dirs, only the second": ({"frames/00000004.jpg"}, {"00000003.jpg"}),
    "two dirs, neither": ({"frames/00000004.jpg"}, {"frames/00000005.jpg"}),
    "no dirs": (None, None),
}


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
def test_without_a_repeated_name_the_frame_is_todays(tmp_path, layout):
    first, second = LAYOUTS[layout]
    dirs = []
    for name, files in (("capA", first), ("capB", second)):
        if files is None:
            continue
        d = tmp_path / name
        d.mkdir()
        for rel in files:
            _put(d / rel)
        dirs.append(d)
    ambiguous = []
    for sources in ({}, {"s:00000003": str(tmp_path / "missing.jpg")}):
        assert GS._source_frame(_kf(), tmp_path / "session", dirs, sources, ambiguous) == \
            _old_source_frame(_kf(), tmp_path / "session", dirs, sources)
    assert ambiguous == []


# ---------------------------------------------------------------------------
# through the solve


def _captures(tmp_path, *, repeated: bool):
    """Two capture directories. With `repeated`, both hold every keyframe's name (a restarted numbering);
    without, the first holds the names and the second other ones."""
    dirs = []
    for i, value in ((0, 255), (1, 0)):
        d = tmp_path / f"cap{i}" / "frames"
        d.mkdir(parents=True)
        for k in range(N):
            name = f"{k:08d}.jpg" if (repeated or i == 0) else f"{k + 100:08d}.jpg"
            cv2.imwrite(str(d / name), np.full((HEIGHT, WIDTH, 3), value, np.uint8))
        dirs.append(d.parent)
    return dirs


def test_a_solve_takes_the_stored_keyframe_for_a_repeated_name_and_says_so(session, colmap, tmp_path):
    summary = _solve(session, final=True, capture_dirs=_captures(tmp_path, repeated=True))
    assert summary["solved"] is True
    record = _solution_json(session)["solve"]["frames_ambiguous_by_name"]
    assert record["count"] == N and len(record["examples"]) == N
    for k in range(N):
        img = cv2.imread(str(session.workspace.images_dir / f"{k:08d}.jpg"))
        # the session's own random-noise keyframe, neither capture's flat frame
        assert 60 < float(img.mean()) < 200 and float(img.std()) > 30


def test_a_solve_with_unique_names_records_nothing_new(session, colmap, tmp_path):
    summary = _solve(session, final=True, capture_dirs=_captures(tmp_path, repeated=False))
    assert "frames_ambiguous_by_name" not in _solution_json(session)["solve"]
    assert "frames_ambiguous_by_name" not in summary["solve"]
    img = cv2.imread(str(session.workspace.images_dir / "00000000.jpg"))
    assert float(img.mean()) > 250, "the one capture holding the name gave its frame, as before"
