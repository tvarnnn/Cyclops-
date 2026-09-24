"""Review V8, LOW-c: `_still_building` for a session with `components: null` is exactly
what it was before the components work (contract WORLD-BUILDER-COMPONENTS.md §7 rule 1:
every older world's rows and bodies stay as they were; §3.4: "On a session with
`components: null` nothing changes").

The C1 E13 branch -- a `running` photographic word is `build_in_progress: true` even when
the present-tense probe saw no live room stage -- was added for AREAS, whose status files
that probe does not read. It applied to every session, so a `components: null` session
whose word read `running` while the probe (a moment apart) saw nothing changed from
`build_in_progress: false` with the "could not be determined" reason to `true` with a new
reason. The word carries `scope` exactly when the session has a components record
(`components.combine_photographic`), so the branch now requires it.

The expected dictionaries below are the integration base's (`17e6d3d`) output, written out.
"""

from __future__ import annotations

import pytest

from tower.results.world_builder import LIFECYCLE_FINALIZING, _still_building

BASE = {"state": "finalized", "evidence": "the session record says complete",
        "reason": "the session finished", "build_in_progress": False,
        "build_in_progress_unavailable_reason": None}
RUNNING = {"state": "running", "stage": "surface",
           "detail": "the surface stage is running under pid 4242"}


def test_a_running_word_on_a_session_without_components_is_answered_as_before():
    """Before the fix: `build_in_progress: True` and "is still running" -- the E13 branch."""
    got = _still_building(dict(BASE), None, None, dict(RUNNING))
    detail = RUNNING["detail"]
    assert got == {
        **BASE,
        "photographic": RUNNING,
        "state": LIFECYCLE_FINALIZING,
        "evidence": f"{BASE['evidence']}, and {detail}",
        "reason": ("whether the photographic build for this world is still running could not "
                   "be determined, so this world is not reported as finished: " + detail),
        "build_in_progress": False,
        "build_in_progress_unavailable_reason": None,
    }


def test_the_old_answer_carries_the_probes_caveat_as_before():
    got = _still_building(dict(BASE), None, "the stage probe failed", dict(RUNNING))
    assert got["build_in_progress"] is False
    assert got["build_in_progress_unavailable_reason"] == "the stage probe failed"


@pytest.mark.parametrize("scope,reason", [
    ("area", "the room is saved and an area of this walk is still being built: "),
    ("room", "the photographic build for this world is still running: "),
])
def test_a_session_with_components_keeps_c1_e13(scope, reason):
    word = dict(RUNNING, scope=scope)
    got = _still_building(dict(BASE), None, None, word)
    assert got["state"] == LIFECYCLE_FINALIZING
    assert got["build_in_progress"] is True
    assert got["build_in_progress_unavailable_reason"] is None
    assert got["reason"] == reason + RUNNING["detail"]


@pytest.mark.parametrize("state", ["owed", "unobservable", "complete", "failed", "unattempted"])
def test_every_other_word_is_untouched_by_the_scope_rule(state):
    """The rule reads `scope` only for `running`; for every other word the answer with or
    without a components record differs only by the block itself."""
    word = {"state": state, "stage": "surface", "detail": "d"}
    plain = _still_building(dict(BASE), None, None, dict(word))
    scoped = _still_building(dict(BASE), None, None, dict(word, scope="room"))
    plain.pop("photographic", None)
    scoped.pop("photographic", None)
    assert plain == scoped


def test_a_live_room_stage_is_answered_as_before_with_or_without_components():
    building = "the surface stage is running under pid 4242"
    a = _still_building(dict(BASE), building, None, dict(RUNNING))
    b = _still_building(dict(BASE), building, None, dict(RUNNING, scope="room"))
    assert a["build_in_progress"] is b["build_in_progress"] is True
    assert a["reason"] == b["reason"]
