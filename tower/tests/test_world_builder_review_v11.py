"""Review V11 fixes (P3.9, CON3).

Modules: `tower/results/world_builder_library.py`, `tower/world_builder/coherence_publish.py`,
`tower/world_builder/global_solve.py` (`_publish_draw_0_first`).

- MED-B: the `GET /worlds` session row sent `finalization` raw -- a `C:\\Users\\<user>\\...` path and a
  traceback in `detail`, an old raw `notice` -- on an unauthenticated listing, while `/ws` sent a
  client-safe copy (RV11-E `test_rv11e_channels.py`). Now both texts are client-safe and at most 700
  characters; a clean record is sent byte for byte as before.
- LOW-1: the solve's second pass gated draw 0 again. Now draw 0 is gated once: an uninterrupted two-pass
  finish publishes exactly what one pass over the N draws publishes (RV11-A `cmp_two_pass.py` is the
  model), what the record says draw 0 cost is the gate that ran, and a transient failure in the second
  pass can no longer publish an anchor-only fail-safe over the room the first pass attached.
- LOW-15: `depth-no-intrinsics` and `depth-no-camera` are not retryable; their sentences are unchanged.
- LOW-16, LOW-17: the scrubber's gaps; the owner-facing reason has no "(: " and no square brackets, and
  realistic reasons pass the Mac-guard reading of `test_world_builder_consensus_v10`.
- LOW-4: a stale `consensus.json` is moved aside whenever the publish writes no consensus detail.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import types
from dataclasses import replace

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.test_world_builder_consensus_v10 import _every_phone_sentence, _guard_trips
from tests.test_world_builder_solve_consensus import PIECES, SID, _candidate, _keyframes, _links, _name, _Store
from tower.results import world_builder as RWB
from tower.results.world_builder_library import build_world_listing
from tower.routes import geometry as geometry_routes
from tower.world_builder import coherence_gate as CG
from tower.world_builder import coherence_publish as CP
from tower.world_builder import global_solve as GS
from tower.world_builder.store import WorldStore

USER = "someone"


@pytest.fixture(autouse=True)
def _a_known_user_name(monkeypatch):
    """This machine's user name, fixed, so the user-name cases do not depend on who runs the suite."""
    monkeypatch.setattr(CP, "_user_names", lambda: {USER})


# ---------------------------------------------------------------------------
# MED-B: the listing row's `finalization` is client-safe


RAW_DETAIL = ("PermissionError: [WinError 32] The process cannot access the file because it is being used by "
              "another process: 'C:\\Users\\someone\\Projects\\Glasses\\tower\\data\\worlds\\w1\\derived\\s1\\"
              "points.json'")
TB_DETAIL = ('Traceback (most recent call last):\n  File "C:\\Users\\someone\\Projects\\Glasses\\tower\\scripts\\'
             'world_build_session.py", line 2330, in main\nRuntimeError: boom')
LONG_DETAIL = "final solve failed: " + "x " * 600
RAW_NOTICE = ("masks were not applied (the gdsam detector failed (OSError: [Errno 22] "
              "C:\\Users\\someone\\Projects\\Glasses\\tower\\.venv\\Lib\\x.py)); an owner can re-finish this walk")


def _client(store) -> TestClient:
    app = FastAPI()
    app.include_router(geometry_routes.router)
    app.state.world_root = store.root
    return TestClient(app)


def _set_finalization(store, world_id, session_id, **fields) -> dict:
    session = store.read_session(world_id, session_id)
    finalization = {"state": "complete", "final_solve": "solved", "started_at": 1.0, "updated_at": 2.0, **fields}
    store.write_session(replace(session, finalization=finalization))
    return finalization


def _row_finalization(store) -> dict:
    return build_world_listing(store)["worlds"][0]["sessions"][0]["finalization"]


def _client_safe(text: str) -> None:
    assert USER not in text and "Users" not in text and "\\" not in text and "\n" not in text
    assert "Traceback" not in text and 'File "' not in text and not re.search(r"\b[A-Za-z]:[\\/]", text)
    assert len(text) <= 700, "the phone's own bound (CP.FINALIZATION_TEXT_MAX_CHARS)"


@pytest.mark.parametrize("detail", [RAW_DETAIL, TB_DETAIL, LONG_DETAIL], ids=["path", "traceback", "long"])
def test_the_listing_row_sends_detail_and_notice_client_safe(derived_world, detail):
    """RV11-E `test_rv11e_channels.py`: at a559fc0 the row's `detail` held `C:\\Users\\<user>\\...` and a
    multi-line traceback, and its `notice` the raw text an old writer put there (L-19c)."""
    store, world_id, session_id = derived_world
    _set_finalization(store, world_id, session_id, detail=detail, notice=RAW_NOTICE)
    fin = _row_finalization(store)
    _client_safe(fin["detail"])
    _client_safe(fin["notice"])
    assert fin["detail"] == CP.client_safe_detail(detail, max_chars=700)
    assert fin["notice"] == "masks were not applied (the gdsam detector failed (OSError: [Errno 22] [path])); " \
                            "an owner can re-finish this walk"
    assert (fin["state"], fin["final_solve"], fin["started_at"], fin["updated_at"]) == ("complete", "solved", 1.0, 2.0)
    served = _client(store).get("/worlds").json()["worlds"][0]["sessions"][0]["finalization"]
    assert served == fin, "the route serves the row as built"
    record = store.read_session(world_id, session_id).finalization
    assert record["detail"] == detail and record["notice"] == RAW_NOTICE, "the record on disk keeps its text"


@pytest.mark.parametrize("detail", [RAW_DETAIL, TB_DETAIL], ids=["path", "traceback"])
def test_the_row_and_the_status_channel_send_one_detail(derived_world, detail):
    """The same transform `/ws` applies to `lifecycle.finalization.detail` (CARTRIDGE-RESULTS: "the same
    object as the listing row's `finalization`")."""
    store, world_id, session_id = derived_world
    _set_finalization(store, world_id, session_id, detail=detail)
    session = store.read_session(world_id, session_id)
    ws = RWB._client_safe_finalization(session.finalization)
    assert _row_finalization(store)["detail"] == ws["detail"]


CLEAN_DETAILS = [
    None,
    "",
    "final solve skipped: hard stop (SIGBREAK) during finalization",
    "final solve terminated: hard stop (SIGINT) during finalization; the last background solution stands",
    "final solve failed: world_solve.py exited 3",
    "final solve produced no solution: fewer than two keyframes",
    "KeyboardInterrupt: ",
    "OSError: [Errno 28] No space left on device",
    "the evidence gate could not measure metric scale (RuntimeError: CUDA out of memory. Tried to allocate 2.00 "
    "GiB at [path]); the Tower re-runs the gate when it is idle",
    "about 3/4 of the images, see IDEA-Research/grounding-dino-base and https://pytorch.org/docs/stable/x.html",
]


def test_a_clean_finalization_is_sent_byte_for_byte_as_before(derived_world):
    """At a559fc0 the row sent the record as it is. Every clean detail, and every sentence the phone's closed set
    holds (the gate's and the finisher's), is still sent exactly so -- the same bytes."""
    store, world_id, session_id = derived_world
    notices = [None] + sorted(_every_phone_sentence().values())
    notices.append("; ".join([CP.NOTICE_SENTENCES["masks-gpu-oom"], CP.NOTICE_SENTENCES["consensus-deferred"]]))
    for i, notice in enumerate(notices):
        detail = CLEAN_DETAILS[i % len(CLEAN_DETAILS)]
        fields = {"detail": detail, **({"notice": notice} if notice is not None else {})}
        written = _set_finalization(store, world_id, session_id, **fields)
        fin = _row_finalization(store)
        assert json.dumps(fin) == json.dumps(written), (detail, notice)
        assert CP.client_safe_finalization(written) is written, "a clean record is the same object"


def test_client_safe_finalization_leaves_what_it_cannot_read_alone():
    for value in (None, "complete", 3, [], {"state": "complete"}, {"detail": None}, {"detail": 7, "notice": ["x"]}):
        assert CP.client_safe_finalization(value) is value
    fin = {"state": "interrupted", "detail": RAW_DETAIL}
    out = CP.client_safe_finalization(fin)
    assert out is not fin and fin["detail"] == RAW_DETAIL and out["state"] == "interrupted"


# ---------------------------------------------------------------------------
# LOW-1: draw 0 is gated once


TWO_PASS_SCENARIOS = {
    # rotated pieces per draw (a rotated piece contradicts its links: that draw's gate leaves it out)
    "agree": [(), (), ()],
    "withhold-in-draw-0": [(), ("X", "A"), ("X", "B")],
    "draw-1-chosen": [("A",), (), ()],
    "draw-1-chosen-and-withheld": [("X", "A"), (), ("X", "B")],
    # draw 0's second depth stage finds the surface lock held (RV11-A `d0-second-gate-lock`)
    "draw-0-second-gate-lock": [(), (), ()],
}


def _dump(root) -> dict:
    """Every file a publish leaves in the world: json parsed, npz arrays hashed, anything else hashed."""
    out = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if "lock" in path.name:
            continue
        if path.suffix == ".json":
            out[rel] = json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix == ".npz":
            with np.load(path) as z:
                out[rel] = {k: hashlib.sha1(np.ascontiguousarray(z[k]).tobytes()).hexdigest() for k in z.files}
        else:
            out[rel] = hashlib.sha1(path.read_bytes()).hexdigest()
    return out


def _finish(root, monkeypatch, scenario: str, *, two_pass: bool):
    """One gated, seeded consensus of 3 on the synthetic solve of `test_world_builder_solve_consensus`, with a
    depth stage that keeps a prediction cache and writes its unstamped `align.json` as the product's does.
    `two_pass`: the solve's own flow (`_publish_draw_0_first`: the early publish of draw 0, then the
    consensus). Otherwise ONE pass over the three draws (`gate_and_publish(consensus=...)`), the reference."""
    rotated = TWO_PASS_SCENARIOS[scenario]
    store = WorldStore(root)
    monkeypatch.setattr(store, "read_session", _Store().read_session)
    pieces = tuple(PIECES)
    links, rots, n = _links(pieces)
    monkeypatch.setattr(CG, "read_verified_links", lambda db, min_inliers=15: dict(links))
    monkeypatch.setattr(CG, "read_link_rotations", lambda db, cam, min_inliers=15: dict(rots))
    keyframes = _keyframes(n)
    dparams = CP._depth_params()
    dense = store.world_dir("w1") / "dense" / SID
    cache: set = set()
    depth_seeds: list = []

    def depth(store_, world_id, session_id, solution, intrinsics, should_stop=None, **kw):
        seed = solution.solve["seed"]
        depth_seeds.append(seed)
        if scenario == "draw-0-second-gate-lock" and depth_seeds.count(7) == 2 and seed == 7:
            raise CP.DepthUnavailable("another surface build of this session holds its lock")
        kids = [k for k in solution.keyframe_ids if k in solution.poses]
        hits = sum(1 for k in kids if k in cache)
        cache.update(kids)
        align = {"backend": "moge2-vitl", "known_fov": 42.0, "targets": len(kids), "stopped_after": None,
                 "records": [{"ki": i, "kid": k, "ok": True, "a": 1.0, "b": 0.0}
                             for i, k in enumerate(solution.keyframe_ids) if k in solution.poses],
                 "image_origins": {"stored": len(kids)},
                 "prediction_cache": {"token": "tok", "hits": hits, "predicted": len(kids) - hits,
                                      "write_failed": 0}}
        dense.mkdir(parents=True, exist_ok=True)
        (dense / "align.json").write_text(json.dumps(align), encoding="utf-8")
        return align, dense / "work", dparams

    monkeypatch.setattr(CP, "run_gate_depth", depth)
    monkeypatch.setattr(CP, "measure_metric_scale", lambda solution, name_of, db, work: {
        "metric_log": {_name(i): 0.0 for i in range(n)}, "cameras_published": n, "cameras_measured": n})
    candidates = []
    for k, rot in enumerate(rotated):
        c = _candidate(pieces, rotated=rot)
        c.solve = {"seed": 7 + k}
        c.timing = {"map_s": 1.0}
        candidates.append(c)
    plan = CP.ConsensusPlan(draws=3, seed=7, map_draw=lambda seed: candidates[seed - 7])
    ws = GS.workspace_for(store, "w1", SID)
    ws.root.mkdir(parents=True, exist_ok=True)
    ws.database_path.write_bytes(b"one frozen database")
    kw = dict(final=True, gate=True, database_path=ws.database_path, keyframes=keyframes)
    if two_pass:
        GS._publish_draw_0_first(store, "w1", SID, ws, candidates[0], plan, should_stop=None, **kw)
    else:
        CP.gate_and_publish(store, "w1", SID, ws, candidates[0], write=GS.write_solution, consensus=plan, **kw)
    return _dump(store.world_dir("w1")), depth_seeds


@pytest.fixture
def frozen_clock(monkeypatch):
    """Every `seconds` is 0 in both flows, so a comparison needs no exclusions at all."""
    monkeypatch.setattr(time, "perf_counter", lambda: 100.0)


@pytest.mark.parametrize("scenario", sorted(TWO_PASS_SCENARIOS))
def test_an_uninterrupted_two_pass_finish_publishes_what_one_pass_publishes(tmp_path, monkeypatch, frozen_clock,
                                                                           scenario):
    """RV11-A `cmp_two_pass.py`, as a test. The solve publishes draw 0 first and then the consensus (review V10,
    MED-1a); at a559fc0 the second pass gated draw 0 again, so the published record differed from one pass's in
    draw 0's prediction counts (`gate.depth.predictions`, `gate.consensus.draws[0].predictions`, the stamped
    `align.json`), and on a transient failure of that second gate in the ROOM itself (RV11-A: 61 keyframes
    where one pass gives 200). Now every file both flows leave -- solution, components, consensus detail,
    the depth hand-off -- is identical, byte for byte of its parsed content, with the clock frozen."""
    one, one_depth = _finish(tmp_path / "one", monkeypatch, scenario, two_pass=False)
    two, two_depth = _finish(tmp_path / "two", monkeypatch, scenario, two_pass=True)
    assert two_depth == one_depth == [7, 8, 9], "each draw's depth stage ran once"
    assert set(two) == set(one)
    for rel in sorted(one):
        assert two[rel] == one[rel], rel
    gate = one["solve/s1/solution.json"]["gate"]
    assert gate["consensus"]["state"] == CP.CONSENSUS_APPLIED and gate["attach"] is True
    draw0 = gate["consensus"]["draws"][0]
    assert draw0["predictions"]["predicted"] > 0 and draw0["predictions"]["cached"] == 0, "draw 0's own gate, cold"


def test_the_two_pass_scenarios_reach_the_paths_they_are_named_for(tmp_path, monkeypatch, frozen_clock):
    seen = {}
    for scenario in sorted(TWO_PASS_SCENARIOS):
        dump, _ = _finish(tmp_path / scenario, monkeypatch, scenario, two_pass=True)
        consensus = dump["solve/s1/solution.json"]["gate"]["consensus"]
        seen[scenario] = (consensus["chosen"]["draw"], bool(consensus["detached"]))
    assert seen == {"agree": (0, False), "withhold-in-draw-0": (0, True), "draw-1-chosen": (1, False),
                    "draw-1-chosen-and-withheld": (1, True), "draw-0-second-gate-lock": (0, False)}


def test_the_second_pass_hands_draw_0_to_the_consensus_untouched(tmp_path, monkeypatch, frozen_clock):
    """The pass-1 result carries draw 0's own gate result, before any consensus record, owed cause or publish
    touched it; the second pass is handed exactly that."""
    handed = []
    real = CP.gate_by_consensus

    def recording(*a, **kw):
        handed.append(kw.get("draw_0"))
        return real(*a, **kw)

    monkeypatch.setattr(CP, "gate_by_consensus", recording)
    _finish(tmp_path, monkeypatch, "agree", two_pass=True)
    assert handed[0] is None and isinstance(handed[1], CP.GateResult)
    record = handed[1].record
    assert "consensus" not in record and record["retryable"] is False and record.get("cause") is None
    assert record["attach"] is True and handed[1].consensus_detail is None and handed[1].draw_0 is None
    assert getattr(handed[1].solution, "gate", None) is None, "the early publish did not reach the copy"


# ---------------------------------------------------------------------------
# LOW-15: no intrinsics, no camera -- not retryable, the same sentence


def _gate_with_depth_failure(detail: str) -> dict:
    """The real gate (`gate_final_solution`) on the synthetic solve, its depth stage raising `detail`."""
    pieces = tuple(PIECES)
    links, rots, n = _links(pieces)

    def depth(*a, **kw):
        raise CP.DepthUnavailable(detail)

    result = CP.gate_final_solution(
        _Store(), "w1", SID, _candidate(pieces), database_path="db", keyframes=_keyframes(n), depth_runner=depth,
        link_reader=lambda db, cam, m: (dict(links), dict(rots)))
    return result.record


@pytest.mark.parametrize("detail, cause", [
    ("session has no intrinsics", "depth-no-intrinsics"),
    ("the solve has no camera", "depth-no-camera"),
])
def test_a_depth_stage_that_cannot_start_for_the_walks_own_reason_is_not_retryable(detail, cause):
    """RV11-E (L-19b residual): these were `retryable`, so the finisher re-ran a gate the sentence says cannot
    help, and its given-up and refused sentences then said "re-finish". Now nothing is owed; the notice is the
    sentence the retryable record said, word for word."""
    gate = _gate_with_depth_failure(detail)
    assert gate["state"] == CP.GATE_STATE_APPLIED and gate["attach"] is False
    assert gate["retryable"] is False and gate["cause"] is None
    assert gate["depth"] == {"state": CP.DEPTH_UNAVAILABLE, "detail": detail, "seconds": gate["depth"]["seconds"]}
    summary = {"gate": gate, "transients": {"state": "applied"}}
    assert CP.notice_causes(summary) == [cause]
    assert CP.publish_notice(summary) == CP.NOTICE_SENTENCES[cause]
    as_before = dict(gate, retryable=True, cause=CP.CAUSE_DEPTH_UNAVAILABLE)
    assert CP.publish_notice(summary) == CP.publish_notice(dict(summary, gate=as_before))
    assert CP.publish_detail(summary) == CP.publish_detail(dict(summary, gate=as_before))
    assert CP.notice_cause_phrase(gate) == CP.NOTICE_PHRASES[cause]
    assert CP.regate_clause(gate) == CP.NOTICE_CLAUSES[cause]


@pytest.mark.parametrize("detail", [
    "another surface build of this session holds its lock",
    "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB",
    "DepthModelUnavailable: moge is not installed",
    "stopped: the depth stage was stopped after 3 frames",
])
def test_every_other_depth_failure_is_still_owed_a_re_gate(detail):
    gate = _gate_with_depth_failure(detail)
    assert gate["retryable"] is True and gate["cause"] == CP.CAUSE_DEPTH_UNAVAILABLE


def test_a_published_solve_without_intrinsics_owes_the_finisher_nothing(tmp_path):
    store = WorldStore(tmp_path)
    ws = GS.workspace_for(store, "w1", SID)
    solution = _candidate(tuple(PIECES))
    solution.gate = _gate_with_depth_failure("session has no intrinsics")
    GS.write_solution(ws, solution)
    assert CP.regate_owed(store, "w1", SID) is None


def test_a_consensus_whose_first_draw_has_no_intrinsics_is_not_deferred():
    """Nothing to vote on, and nothing a re-gate can cure: `not-needed`, not `deferred` (owed)."""
    pieces = tuple(PIECES)
    links, rots, n = _links(pieces)

    def gate_runner(store, world_id, session_id, candidate, **kw):
        def depth(*a, **k):
            raise CP.DepthUnavailable("session has no intrinsics")
        return CP.gate_final_solution(store, world_id, session_id, candidate, database_path="db",
                                      keyframes=kw["keyframes"], depth_runner=depth,
                                      link_reader=lambda db, cam, m: (dict(links), dict(rots)))

    plan = CP.ConsensusPlan(draws=3, seed=0, map_draw=lambda seed: pytest.fail("no draw is mapped"))
    result = CP.gate_by_consensus(_Store(), "w1", SID, _candidate(pieces), plan=plan, database_path="db",
                                  keyframes=_keyframes(n), gate_runner=gate_runner)
    assert result.record["consensus"]["state"] == CP.CONSENSUS_NOT_NEEDED
    assert result.record["retryable"] is False


# ---------------------------------------------------------------------------
# LOW-16: the scrubber's gaps


SCRUB = {
    "U+2028": ("line one\u2028line two", "line one line two"),
    "U+2029": ("line one\u2029line two", "line one line two"),
    "U+0085": ("line one\u0085line two", "line one line two"),
    "VT": ("line one\x0bline two", "line one line two"),
    "FF": ("line one\x0cline two", "line one line two"),
    "U+2028 before a path": ("line one\u2028line two C:\\Users\\someone\\x", "line one line two [path]"),
    "a relative path": ("cannot open data/worlds/7d31e8d7/derived/s1/points.json now", "cannot open [path] now"),
    "a relative a/b/c.json": ("cannot read a/b/c.json", "cannot read [path]"),
    "a relative file in one directory": ("cannot read images/00000042.jpg now", "cannot read [path] now"),
    "the tail of a spaced user directory": ("cannot write C:\\Users\\John Smith now", "cannot write [path] now"),
    "the tail of a spaced home": ("cannot open /home/Anne Marie Smith now", "cannot open [path] now"),
    "the tail of a spaced file name": ("cannot write C:\\Users\\someone\\Glasses\\my file.txt now",
                                       "cannot write [path] now"),
    "a frame without its header": ('  File "C:\\Users\\someone\\a.py", line 3, in f\n    x = y\nRuntimeError: boom',
                                   "RuntimeError: boom"),
    "a traceback that ends on a frame": ('Traceback (most recent call last):\n  File "C:\\Users\\someone\\a.py", '
                                         "line 3, in f", "a traceback"),
    "a frame in one line": ('worker failed at File "C:\\Users\\someone\\x.py", line 3, in <module>',
                            "worker failed at"),
    "a frame alone": ('File "[path]", line 12', "a traceback"),
    "a path before );": ("the gate failed (OSError: C:\\Users\\someone\\x.py); the Tower re-runs it when it is idle",
                         "the gate failed (OSError: [path]); the Tower re-runs it when it is idle"),
    "a path before ).": ("the gate failed (at /home/someone/x/y.py).", "the gate failed (at [path])."),
    "a path before :": ("C:\\Users\\someone\\x.db: locked", "[path]: locked"),
    "a path that holds its own parentheses": ("cannot open C:\\Tools (x86)\\a\\b.txt; retry",
                                              "cannot open [path]; retry"),
}

CLEAN = [
    "about 3/4 of the images were masked",
    "the ratio was 3/4 at 1/2 scale",
    "transformers cannot load Grounding DINO / SAM 2",
    "IDEA-Research/grounding-dino-base is not in the cache",
    "see https://pytorch.org/docs/stable/notes/cuda.html for more",
    "Tried to allocate 2.00 GiB (GPU 0; 8.00 GiB total capacity)",
    "final solve skipped: hard stop (SIGBREAK) during finalization",
    "12 of 40 supported cameras with a ratio (30%; available at >= 50%)",
    "the model v1.2 failed, e.g. twice",
    "col1\tcol2",
]


@pytest.mark.parametrize("name", sorted(SCRUB))
def test_the_scrubber_closes_its_gaps(name):
    raw, safe = SCRUB[name]
    assert CP.client_safe_detail(raw) == safe
    assert not re.search(r"[\r\n\x0b\x0c\x1c-\x1e\x85\u2028\u2029]", CP.client_safe_detail(raw))


@pytest.mark.parametrize("text", CLEAN)
def test_clean_text_is_still_returned_unchanged(text):
    assert CP.client_safe_detail(text) == text
    assert CP.client_safe_detail(text, max_chars=CP.WHY_MAX_CHARS) == text


# ---------------------------------------------------------------------------
# LOW-17: the owner-facing reason -- no orphan "(: ", no square brackets


OWNER = {
    "OSError with an errno, in parentheses": (
        "the evidence gate failed (OSError: [Errno 28] No space left on device); the Tower re-runs it when it is idle",
        "the evidence gate failed (No space left on device); the Tower re-runs it when it is idle"),
    "EOFError after a prefix": ("final solve failed: EOFError: Ran out of input", "final solve failed: Ran out of input"),
    "a quoted path": ("PermissionError: [WinError 32] The process cannot access the file: "
                      "'C:\\Users\\someone\\w\\points.json'", "The process cannot access the file: a path"),
    "a path in parentheses": ("the gate failed (OSError: C:\\Users\\someone\\x.py); the Tower re-runs it when it is idle",
                              "the gate failed (a path); the Tower re-runs it when it is idle"),
    "an errno and a path": ("OSError: [Errno 13] Permission denied: '/home/someone/x/y.db'", "Permission denied: a path"),
    "a user name": ("access denied for Someone", "access denied for a user name"),
    "a class in parentheses": ("the gate failed (KeyboardInterrupt); x", "the gate failed; x"),
}


@pytest.mark.parametrize("name", sorted(OWNER))
def test_the_owner_facing_detail_has_no_orphan_and_no_bracket(name):
    raw, owner = OWNER[name]
    out = CP.owner_facing_detail(raw)
    assert out == owner
    assert "[" not in out and "]" not in out and not re.search(r"\(\s*:", out) and not re.search(r":\s+:", out)


@pytest.mark.parametrize("raw", [raw for raw, _ in SCRUB.values()] + [raw for raw, _ in OWNER.values()] + [
    RAW_DETAIL, TB_DETAIL, RAW_NOTICE, "[WinError 5] Access is denied", "KeyError: 'kf_000123'",
    "held by [something]"])
def test_no_owner_facing_detail_emits_a_square_bracket(raw):
    out = CP.owner_facing_detail(raw)
    assert "[" not in out and "]" not in out and not re.search(r"\(\s*:", out)


# Realistic details, as the builder, `world_finalize`, the gate and the finisher write them (RV11-E
# `probe_owner_guard.py`), through the two `lifecycle.reason` branches that quote the detail.
REASON_DETAILS = [
    CP.publish_detail({"gate": {"state": "applied", "masks_applied": True, "retryable": True,
                                "cause": "depth-unavailable",
                                "depth": {"state": "unavailable", "detail": "RuntimeError: CUDA out of memory. "
                                          "Tried to allocate 2.00 GiB at C:\\Users\\someone\\x\\model.py:411"}},
                       "transients": {"state": "applied"}}),
    CP.publish_detail({"gate": {"state": "failed", "retryable": True, "cause": "gate-failed",
                                "detail": "OSError: [Errno 28] No space left on device"},
                       "transients": {"state": "applied"}}),
    "IOError: disk quota exceeded",
    "final solve failed: EOFError: Ran out of input",
    RAW_DETAIL,
    TB_DETAIL,
    "KeyboardInterrupt: ",
    "final solve skipped: hard stop (SIGBREAK) during finalization",
    "final solve failed: OSError: [Errno 28] No space left on device: '/home/someone/Glasses/tower/data/w'",
    "the gate failed (OSError: C:\\Users\\someone\\x.py); the Tower re-runs it when it is idle",
    "cannot write C:\\Users\\John Smith now",
    "line one\u2028line two",
]


def _reasons(detail):
    out = []
    for end_reason, state, geometry in (("error", "complete", False), ("stopped", "interrupted", True)):
        session = types.SimpleNamespace(finalization={"state": state, "final_solve": "solved", "detail": detail},
                                        end_reason=end_reason, ended_at=1.0, stages=None)
        lc = RWB._lifecycle_from_the_record(holder=None, stopped=True, session=session, geometry_current=True,
                                            has_manifest=True, has_session_geometry=geometry,
                                            has_readable_figures=True)
        assert lc["state"] == "interrupted"
        out.append(lc["reason"])
    return out


@pytest.mark.parametrize("detail", REASON_DETAILS)
def test_realistic_reasons_pass_the_macs_guard(detail):
    """CON2's reading of the Mac's `WorldTowerText` guard (Mac 3bb4431), over the reasons the phone shows. At
    a559fc0 a path read "[path]" and an errno "[Errno 28]", which that reading counts as JSON."""
    for reason in _reasons(detail):
        assert _guard_trips(reason) == [], reason


# ---------------------------------------------------------------------------
# LOW-4: a stale consensus.json is moved aside


STALE = {"record": CP.CONSENSUS_RECORD, "session_id": SID, "solve_identity": "an older solve", "draws": []}


def _stale_workspace(tmp_path):
    store = WorldStore(tmp_path)
    ws = GS.workspace_for(store, "w1", SID)
    ws.root.mkdir(parents=True, exist_ok=True)
    (ws.root / CP.CONSENSUS_FILENAME).write_text(json.dumps(STALE), encoding="utf-8")
    return store, ws


@pytest.mark.parametrize("detail", [None, "a gate result without a consensus detail"])
def test_a_publish_without_a_consensus_detail_moves_the_old_one_aside(tmp_path, detail):
    store, ws = _stale_workspace(tmp_path)
    result = None if detail is None else CP.GateResult(solution=None, record={}, components=None)
    out = CP.after_publish(store, "w1", SID, ws.root, None, result)
    assert out.get("consensus_retired") is True
    assert not (ws.root / CP.CONSENSUS_FILENAME).exists()
    assert json.loads((ws.root / CP.CONSENSUS_SUPERSEDED_FILENAME).read_text(encoding="utf-8")) == STALE


def test_a_publish_with_a_consensus_detail_replaces_it(tmp_path):
    store, ws = _stale_workspace(tmp_path)
    detail = {"record": CP.CONSENSUS_RECORD, "session_id": SID, "solve_identity": "this solve", "draws": []}
    result = CP.GateResult(solution=None, record={}, components=None, consensus_detail=detail)
    out = CP.after_publish(store, "w1", SID, ws.root, None, result)
    assert out.get("consensus_written") is True and "consensus_retired" not in out
    assert json.loads((ws.root / CP.CONSENSUS_FILENAME).read_text(encoding="utf-8")) == detail
    assert not (ws.root / "consensus.superseded.json").exists()


def test_with_no_old_consensus_the_publish_says_what_it_said_before(tmp_path):
    store = WorldStore(tmp_path)
    ws = GS.workspace_for(store, "w1", SID)
    ws.root.mkdir(parents=True, exist_ok=True)
    assert CP.after_publish(store, "w1", SID, ws.root, None, None) == {"components_retired": False}
    assert not any(ws.root.iterdir())


def test_the_early_publish_moves_an_older_consensus_aside(tmp_path, monkeypatch, frozen_clock):
    """The window RV11-A named: between the early publish of draw 0 and the consensus, the older solve's
    `consensus.json` sat beside the new solve. A kill in that window left it there for good."""
    root = tmp_path / "w"
    store = WorldStore(root)
    ws = GS.workspace_for(store, "w1", SID)
    ws.root.mkdir(parents=True, exist_ok=True)
    (ws.root / CP.CONSENSUS_FILENAME).write_text(json.dumps(STALE), encoding="utf-8")
    real = CP.gate_by_consensus

    def killed_after_the_early_publish(*a, **kw):
        if not kw.get("stopped"):
            raise KeyboardInterrupt("a hard stop while the further draws are mapped")
        return real(*a, **kw)

    monkeypatch.setattr(CP, "gate_by_consensus", killed_after_the_early_publish)
    with pytest.raises(KeyboardInterrupt):
        _finish(root, monkeypatch, "agree", two_pass=True)
    assert not (ws.root / CP.CONSENSUS_FILENAME).exists()
    assert json.loads((ws.root / CP.CONSENSUS_SUPERSEDED_FILENAME).read_text(encoding="utf-8")) == STALE
    assert CP.regate_owed(store, "w1", SID) == CP.CAUSE_CONSENSUS_DEFERRED
