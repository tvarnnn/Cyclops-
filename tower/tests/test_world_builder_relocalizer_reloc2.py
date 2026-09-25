"""RELOC2 (manager 058): three relocalizer options, each off by default.

1. `window_from: prompt` -- a PROMPTED episode's timeout runs from the
   prompt, so the wearer gets the whole `timeout_s` after being asked;
2. historical references -- up to `history_keyframes` older keyframes,
   spread over the session in runs of `history_group`, added to each
   episode's recent references (MISSION Objective 4);
3. `recovery_summary` -- one line per resolved episode: references, scans,
   best legs, closest triangle, losses joined.

What is pinned here:

* the defaults change NOTHING: no option reaches the relocalizer, its wire
  payload is byte-identical, an engine journals exactly what it journaled;
* the acceptance rule is untouched by every option (the floors are the
  contract's 50 / 8 deg / 100);
* each option does what it says, on the pure machine or with a scripted
  verifier;
* the setting parser: unset, garbage and out-of-range read as today's.
"""

import json

import numpy as np
import pytest

from tests import synthetic_scene as ss
from tower import config
from tower.world_builder import relocalizer as R
from tower.world_builder.engine import WorldBuilderEngine
from tower.world_builder.events import EVENT_KINDS
from tower.world_builder.records import CameraIntrinsics
from tower.world_builder.store import WorldStore

WIDTH, HEIGHT = 480, 360
LIM = R.LimiterParams()


def _kinds(events):
    return [kind for kind, _ in events]


# -- defaults change nothing ---------------------------------------------------------


def test_no_options_is_no_keyword():
    assert R.options_kwargs(None) == {}
    assert R.options_kwargs({}) == {}
    assert R.options_kwargs({"window_from": "loss", "history_keyframes": 0, "summary_events": False}) == {}


def test_the_default_wire_payload_is_byte_identical():
    """What `relocalizer_started` carried before RELOC2, key for key."""
    assert R.LimiterParams().to_wire() == {
        "max_prompts": 2, "window_s": 60.0, "mechanism": "cooldown", "cooldown_s": 30.0,
        "prompt_after_s": 5.0, "timeout_s": 20.0, "speak_window_s": 5.0,
    }
    assert R.AcceptanceParams().to_wire() == {
        "matcher": "sift", "reference_keyframes": 10, "scan_hz": 2.0,
        "triangle": {"min_links": 2, "min_link_inliers": 50, "max_closure_deg": 8.0},
        "strong_link": {"min_inliers": 100},
    }


def test_every_option_keeps_the_acceptance_rule():
    kw = R.options_kwargs({"window_from": "prompt", "history_keyframes": 20, "summary_events": True})
    acc = kw["acceptance"]
    assert (acc.triangle_min_links, acc.triangle_min_link_inliers, acc.triangle_max_closure_deg) == (2, 50, 8.0)
    assert acc.strong_link_min_inliers == 100
    assert (acc.matcher, acc.reference_keyframes, acc.scan_hz) == ("sift", 10, 2.0)
    lim = kw["limiter"]
    assert (lim.prompt_after_s, lim.timeout_s, lim.cooldown_s, lim.max_prompts) == (5.0, 20.0, 30.0, 2)


def test_options_are_on_the_wire_only_when_on():
    kw = R.options_kwargs({"window_from": "prompt", "history_keyframes": 20})
    assert kw["limiter"].to_wire()["window_from"] == "prompt"
    assert kw["acceptance"].to_wire()["history"] == {"keyframes": 20, "group": 2}
    # The payload block keeps exactly the contract's keys whatever was journaled.
    reader = R.RecoveryJournalReader()
    reader.feed({"kind": "relocalizer_started", "at": 1.0, "payload": {
        "acceptance": kw["acceptance"].to_wire(), "limiter": kw["limiter"].to_wire(),
        "prompts_enabled": True}})
    block = reader.block()
    assert "window_from" not in block["limiter"] and "history" not in block["acceptance"]


def test_bad_options_are_refused():
    with pytest.raises(ValueError):
        R.options_kwargs({"window_from": "never"})
    for n in (-1, 1, 3, 21, 40):
        with pytest.raises(ValueError):
            R.options_kwargs({"history_keyframes": n})
    for n in (4, 20):
        assert R.options_kwargs({"history_keyframes": n})["acceptance"].history_keyframes == n


def test_the_history_bounds_are_one_pair_of_numbers():
    """config mirrors the relocalizer's measured range without importing it."""
    assert (config.WORLD_RELOCALIZER_HISTORY_MIN, config.WORLD_RELOCALIZER_HISTORY_MAX) == (
        R.HISTORY_MIN_KEYFRAMES, R.HISTORY_MAX_KEYFRAMES) == (4, 20)


@pytest.mark.parametrize("capacity, group", [(3, 1), (2, 2), (3, 2), (5, 3)])
def test_a_history_below_two_groups_is_refused(capacity, group):
    """Review F7: below two groups nothing can be spread."""
    with pytest.raises(ValueError):
        R._History(capacity=capacity, group=group)


def test_the_summary_is_a_journal_kind():
    assert R.EVENT_SUMMARY == "recovery_summary" and R.EVENT_SUMMARY in EVENT_KINDS
    # ...and not one the payload block moves on.
    assert R.EVENT_SUMMARY not in R.RECOVERY_EVENT_KINDS


# -- 1. the window from the prompt ---------------------------------------------------------


def _window(from_):
    return R.RecoveryStateMachine(R.LimiterParams(window_from=from_), prompts_enabled=True)


def test_by_default_the_timeout_runs_from_the_loss():
    m = _window("loss")
    m.lost(100.0)
    assert _kinds(m.tick(105.0)) == ["recovery_prompted"]
    assert _kinds(m.tick(120.0)) == ["recovery_timed_out"]


def test_from_the_prompt_the_wearer_gets_the_whole_timeout_after_being_asked():
    m = _window("prompt")
    m.lost(100.0)
    assert _kinds(m.tick(105.2)) == ["recovery_prompted"]
    assert m.prompted_at == 105.2
    assert m.tick(120.0) == [] and m.tick(125.1) == []  # past loss + 20 s: still open
    assert m.is_open and m.state == "prompting"
    assert _kinds(m.accepted(124.0, "triangle", {})) == ["recovery_accepted"]
    assert m.counts["recovered_after_prompt"] == 1


def test_from_the_prompt_it_still_times_out_timeout_s_after_the_prompt():
    m = _window("prompt")
    m.lost(100.0)
    m.tick(105.0)
    assert m.tick(124.99) == []
    assert _kinds(m.tick(125.0)) == ["recovery_timed_out"] and m.resolved_at == 125.0


@pytest.mark.parametrize("prompts_enabled", [False, True])
def test_an_unprompted_episode_keeps_the_loss_window(prompts_enabled):
    """Withheld (disabled here; the limiter below) or never prompted: from the loss."""
    m = R.RecoveryStateMachine(R.LimiterParams(window_from="prompt"), prompts_enabled=prompts_enabled)
    if prompts_enabled:  # episode 2 opens inside episode 1's prompt cooldown
        m.lost(0.0)
        m.tick(5.0)
        assert _kinds(m.tick(25.0)) == ["recovery_timed_out"]
        m.lost(26.0)
        assert m.tick(31.0)[0][1]["why"] == "limiter"
        base = 26.0
    else:
        m.lost(100.0)
        m.tick(105.0)
        base = 100.0
    assert m.prompted_at is None
    assert _kinds(m.tick(base + 20.0)) == ["recovery_timed_out"]


def test_the_next_episode_starts_with_no_prompt_time():
    m = _window("prompt")
    m.lost(0.0)
    m.tick(5.0)
    m.tick(25.0)
    assert m.state == "timed_out"
    m.lost(26.0)
    assert m.prompted_at is None and m.losses_joined == 0


def test_losses_that_join_are_counted():
    m = R.RecoveryStateMachine(LIM, prompts_enabled=True)
    m.lost(0.0)
    m.lost(1.0)
    m.lost(2.0)
    assert m.losses_joined == 2 and m.lost_at == 0.0


# -- 2. historical references -----------------------------------------------------------------


def test_history_keeps_the_budget_spread_over_the_walk_in_pairs():
    h = R._History(capacity=6, group=2)
    for s in range(100):
        h.add(f"k{s}", s, None)
    seqs = [s for _n, g in h.groups for _k, s, _g in g]
    assert len(seqs) == 6
    assert all(len(g) == 2 and g[1][1] == g[0][1] + 1 for _n, g in h.groups)  # consecutive pairs
    assert seqs[:2] == [0, 1]  # the walk's first view is kept
    assert seqs[-2:] == [98, 99]  # the newest pair is kept
    gaps = np.diff([g[0][1] for _n, g in h.groups])
    assert gaps.min() >= 20  # spread, not bunched at either end
    assert [k for k, _ in h.members()][:2] == ["k0", "k1"]


def test_history_spacing_stays_within_a_factor_of_two_of_even():
    h = R._History(capacity=20, group=2)
    for s in range(1000):
        h.add(f"k{s}", s, None)
    starts = [g[0][1] for _n, g in h.groups][:-1]  # the newest is kept whatever its number
    gaps = np.diff(starts)
    assert len(set(gaps.tolist())) == 1  # evenly strided
    assert sum(len(g) for _n, g in h.groups) <= 20 and len(h.groups) >= 6


def test_history_group_one_keeps_singles():
    h = R._History(capacity=4, group=1)
    for s in range(40):
        h.add(f"k{s}", s, None)
    assert [len(g) for _n, g in h.groups] == [1, 1, 1, 1]
    assert h.groups[0][1][0][1] == 0 and h.groups[-1][1][0][1] == 39


def _gray(marker):
    g = np.zeros((HEIGHT, WIDTH), np.uint8)
    g[0, 0] = marker
    return g


class _Scripted:
    """A verifier that links by marker: `table[(a, b)]` inliers, identity R."""

    def __init__(self, table):
        self.table = table

    def features(self, gray):
        return R.Features(np.array([[float(gray[0, 0]), 0.0]]), np.zeros((1, 128), np.float32))

    def verify(self, a, b, size):
        key = (int(a.xy[0, 0]), int(b.xy[0, 0]))
        n = self.table.get(key) or self.table.get(key[::-1])
        if not n:
            return None
        return R.Link(n_inliers=n, n_matches=n, R=np.eye(3), t=np.array([1.0, 0, 0]), model="essential")


def _reloc(table, **kw):
    return R.LookBackRelocalizer(camera_matrix=np.eye(3), frame_size=(WIDTH, HEIGHT),
                                 prompts_enabled=True, verifier=_Scripted(table), synchronous=True, **kw)


def _walk_keyframes(reloc, n):
    for i in range(n):
        reloc.note_keyframe(f"k{i}", i, _gray(i))
        reloc.note_frame(_gray(i), i, f"k{i}", float(i) * 0.1)


def test_with_history_off_the_references_are_the_last_ten_exactly():
    reloc = _reloc({})
    _walk_keyframes(reloc, 30)
    reloc.note_lost(10.0)
    assert [k for k, _ in reloc._episode.refs] == [f"k{i}" for i in range(20, 30)]
    assert reloc._history is None
    reloc.close(11.0)


def test_with_history_the_references_add_older_spread_keyframes():
    reloc = _reloc({}, acceptance=R.AcceptanceParams(history_keyframes=6))
    _walk_keyframes(reloc, 60)
    reloc.note_lost(10.0)
    ids = [k for k, _ in reloc._episode.refs]
    assert ids[:10] == [f"k{i}" for i in range(50, 60)]
    old = [int(k[1:]) for k in ids[10:]]
    assert len(old) == 6 and len(set(ids)) == len(ids)
    assert max(old) < 50 and old[:2] == [0, 1]
    assert reloc._episode.n_history == 6
    reloc.close(11.0)


def test_a_revisit_of_an_old_view_relinks_only_with_history():
    """Keyframes 0 and 1 (an old view, linked to each other) are what the
    scanned frame 200 matches. Without history they are not references."""
    table = {(0, 1): 120, (0, 200): 60, (1, 200): 70}
    for history, expect in ((0, []), (6, ["recovery_accepted"])):
        reloc = _reloc(table, acceptance=R.AcceptanceParams(history_keyframes=history))
        _walk_keyframes(reloc, 40)
        ev = reloc.note_lost(10.0) + reloc.note_frame(_gray(200), 200, None, 10.0)
        assert [k for k in _kinds(ev) if k == "recovery_accepted"] == expect
        if expect:
            acc = [p for k, p in ev if k == "recovery_accepted"][0]
            assert acc["by"] == "triangle" and {l["ref_keyframe_id"] for l in acc["links"]} == {"k0", "k1"}
        reloc.close(11.0)


# -- 3. the per-episode summary -------------------------------------------------------------------


def test_no_summary_by_default():
    reloc = _reloc({(8, 9): 90, (8, 200): 60, (9, 200): 55})
    _walk_keyframes(reloc, 10)
    ev = reloc.note_lost(10.0) + reloc.note_frame(_gray(200), 200, None, 10.0)
    ev += reloc.close(11.0)
    assert "recovery_summary" not in _kinds(ev)


def test_a_recovered_episode_is_summarised_once_right_after_its_acceptance():
    reloc = _reloc({(8, 9): 90, (8, 200): 60, (9, 200): 55, (7, 200): 31}, summary_events=True)
    _walk_keyframes(reloc, 10)
    reloc.note_lost(10.0)
    reloc.note_lost(10.1)  # joins
    ev = reloc.note_frame(_gray(200), 200, None, 10.2)
    assert _kinds(ev) == ["recovery_accepted", "recovery_summary"]
    s = ev[1][1]
    assert s["episode"] == 1 and s["outcome"] == "recovered" and s["by"] == "triangle"
    assert s["prompted"] is False and s["losses_joined"] == 1 and s["attempts"] == 1
    assert s["references"] == [f"k{i}" for i in range(10)] and s["history_references"] == 0
    assert s["best_links"] == [{"ref_keyframe_id": "k8", "inliers": 60},
                               {"ref_keyframe_id": "k9", "inliers": 55},
                               {"ref_keyframe_id": "k7", "inliers": 31}]
    assert s["best_pair"]["refs"] == ["k8", "k9"] and s["best_pair"]["refs_linked"] is True
    assert s["best_triangle"] == {"refs": ["k8", "k9"], "inliers": [60, 55], "closure_deg": 0.0,
                                  "frame": {"source_seq": 200}}
    json.dumps(s)  # journal-serialisable
    tail = reloc.note_frame(_gray(201), 201, None, 10.7) + reloc.close(11.0)
    assert "recovery_summary" not in _kinds(tail)  # once


def test_a_timed_out_episode_is_summarised_with_its_near_miss():
    reloc = _reloc({(8, 9): 90, (8, 200): 46, (9, 200): 31}, summary_events=True)
    _walk_keyframes(reloc, 10)
    ev = reloc.note_lost(10.0)
    t = 10.0
    while t < 31.0:
        ev += reloc.note_frame(_gray(200), 200, None, t)
        t += 0.5
    kinds = _kinds(ev)
    assert kinds.index("recovery_summary") == kinds.index("recovery_timed_out") + 1
    s = [p for k, p in ev if k == "recovery_summary"][0]
    assert s["outcome"] == "timed_out" and s["prompted"] is True and s["by"] is None
    # scans at 10.0, 10.5, ... 30.0: the frame that times it out is scanned first
    assert s["attempts"] == 41 and s["open_s"] == 20.0
    assert s["best_pair"]["inliers"] == [46, 31] and s["best_triangle"] is None
    reloc.close(40.0)


def test_an_episode_open_at_stop_is_summarised_before_the_terminal_line():
    reloc = _reloc({}, summary_events=True)
    _walk_keyframes(reloc, 10)
    reloc.note_lost(10.0)
    ev = reloc.close(11.0)
    assert _kinds(ev) == ["recovery_timed_out", "recovery_summary", "relocalizer_stopped"]
    assert ev[1][1]["outcome"] == "timed_out" and ev[1][1]["attempts"] == 0


def test_the_payload_block_ignores_the_summary():
    journal = [
        {"kind": "relocalizer_started", "at": 1.0, "payload": R.LookBackRelocalizer(
            camera_matrix=np.eye(3), frame_size=(WIDTH, HEIGHT), prompts_enabled=True).started_payload()},
        {"kind": "tracking_lost", "at": 2.0, "payload": {}},
        {"kind": "recovery_timed_out", "at": 22.0, "payload": {"episode": 1}},
    ]
    summary = {"kind": "recovery_summary", "at": 22.0, "payload": {"episode": 1, "outcome": "recovered"}}
    assert R.recovery_block(journal + [summary]) == R.recovery_block(journal)


# -- the settings, and the engine ------------------------------------------------------------------


@pytest.mark.parametrize("env, expect", [
    ({}, {}),
    ({"TOWER_WORLD_RELOCALIZER_WINDOW": "loss"}, {}),
    ({"TOWER_WORLD_RELOCALIZER_WINDOW": "prompt"}, {"window_from": "prompt"}),
    ({"TOWER_WORLD_RELOCALIZER_WINDOW": "sometimes"}, {}),
    ({"TOWER_WORLD_RELOCALIZER_HISTORY": "20"}, {"history_keyframes": 20}),
    ({"TOWER_WORLD_RELOCALIZER_HISTORY": "4"}, {"history_keyframes": 4}),
    ({"TOWER_WORLD_RELOCALIZER_HISTORY": "0"}, {}),
    # review F3: 20 is the most the replay measured; F7: below 4 degenerates
    ({"TOWER_WORLD_RELOCALIZER_HISTORY": "21"}, {}),
    ({"TOWER_WORLD_RELOCALIZER_HISTORY": "40"}, {}),
    ({"TOWER_WORLD_RELOCALIZER_HISTORY": "3"}, {}),
    ({"TOWER_WORLD_RELOCALIZER_HISTORY": "-4"}, {}),
    ({"TOWER_WORLD_RELOCALIZER_HISTORY": "lots"}, {}),
    ({"TOWER_WORLD_RELOCALIZER_SUMMARY": "on"}, {"summary_events": True}),
    ({"TOWER_WORLD_RELOCALIZER_SUMMARY": "off"}, {}),
])
def test_the_settings_default_to_today_and_garbage_is_today(monkeypatch, env, expect):
    for name in ("TOWER_WORLD_RELOCALIZER_WINDOW", "TOWER_WORLD_RELOCALIZER_HISTORY",
                 "TOWER_WORLD_RELOCALIZER_SUMMARY"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert config.world_relocalizer_options() == expect


class _Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture(scope="module")
def room_frames():
    K = ss.camera_matrix(WIDTH, HEIGHT)
    images = ss.render_sequence(ss.furnished_room(), ss.strafe(10, step=0.09), K, WIDTH, HEIGHT)
    rng = np.random.default_rng(0)
    noise = [rng.integers(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8) for _ in range(6)]
    return [ss.encode_jpeg(i) for i in images], [ss.encode_jpeg(n) for n in noise]


def _engine_walk(root, frames, **engine_kw):
    K = ss.camera_matrix(WIDTH, HEIGHT)
    intr = CameraIntrinsics(source="self_calibrated", model="pinhole", fx=float(K[0, 0]), fy=float(K[1, 1]),
                            cx=float(K[0, 2]), cy=float(K[1, 2]), calibrated_width=WIDTH, calibrated_height=HEIGHT)
    clock = _Clock()
    engine = WorldBuilderEngine(WorldStore(root), clock=clock, relocalizer="prompt", **engine_kw)
    wid = engine.create_world("reloc2")
    sid = engine.start_session(wid, intrinsics=intr, frame_source="synthetic", declared_size=(WIDTH, HEIGHT))
    for seq, jpeg in enumerate(frames):
        clock.t += 0.3
        engine.observe(jpeg, source_seq=seq, wire_seq=seq)
    engine.stop_session()
    store = WorldStore(root)
    return [json.loads(l) for l in store.events_path(wid, sid).read_text().splitlines()]


def _strip(events):
    """Kinds and payloads, with the run's random ids (world, session) masked."""
    import re

    return [(e["kind"], re.sub(r"[0-9a-f]{32}", "ID", json.dumps(e["payload"], sort_keys=True)))
            for e in events]


def test_an_engine_with_the_environment_unset_journals_exactly_as_before(tmp_path, room_frames, monkeypatch):
    for name in ("TOWER_WORLD_RELOCALIZER_WINDOW", "TOWER_WORLD_RELOCALIZER_HISTORY",
                 "TOWER_WORLD_RELOCALIZER_SUMMARY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(R, "from_session", _synchronous(R.from_session))
    room, noise = room_frames
    frames = room + noise + room[::-1][:6]
    from_env = _engine_walk(tmp_path / "a", frames)
    explicit_off = _engine_walk(tmp_path / "b", frames, relocalizer_options={})
    assert _strip(from_env) == _strip(explicit_off)
    assert "recovery_summary" not in {e["kind"] for e in from_env}


def test_an_engine_with_the_summary_on_journals_one_per_episode(tmp_path, room_frames, monkeypatch):
    monkeypatch.setattr(R, "from_session", _synchronous(R.from_session))
    room, noise = room_frames
    events = _engine_walk(tmp_path / "c", room + noise + room[::-1][:6],
                          relocalizer_options={"summary_events": True, "window_from": "prompt",
                                               "history_keyframes": 4})
    started = [e for e in events if e["kind"] == "relocalizer_started"][0]["payload"]
    assert started["limiter"]["window_from"] == "prompt"
    assert started["acceptance"]["history"] == {"keyframes": 4, "group": 2}
    episodes = {e["payload"]["episode"] for e in events
                if e["kind"] in ("recovery_accepted", "recovery_timed_out")}
    summaries = [e["payload"] for e in events if e["kind"] == "recovery_summary"]
    assert episodes and sorted(s["episode"] for s in summaries) == sorted(episodes)


def _synchronous(real):
    def factory(intrinsics, *, mode, **kw):
        return real(intrinsics, mode=mode, synchronous=True, **kw)
    return factory


class _Counting(_Scripted):
    def __init__(self, table):
        super().__init__(table)
        self.extracted = []

    def features(self, gray):
        self.extracted.append(int(gray[0, 0]))
        return super().features(gray)


def test_a_historical_reference_is_extracted_once_across_episodes():
    """The worker's feature cache (history only): the next episode re-uses
    what the last one extracted for the references the two share."""
    verifier = _Counting({})
    reloc = R.LookBackRelocalizer(camera_matrix=np.eye(3), frame_size=(WIDTH, HEIGHT), prompts_enabled=True,
                                  verifier=verifier, synchronous=True,
                                  acceptance=R.AcceptanceParams(history_keyframes=4))
    _walk_keyframes(reloc, 30)
    reloc.note_lost(10.0)
    reloc.note_frame(_gray(200), 200, None, 10.0)
    assert len([m for m in verifier.extracted if m != 200]) == 14  # 10 recent + 4 historical, once
    for t in (10.5, 11.0, 30.0):  # times out at 30.0
        reloc.note_frame(_gray(201), 201, None, t)
    assert not reloc.machine.is_open
    verifier.extracted.clear()
    reloc.note_keyframe("k30", 30, _gray(30))  # one new recent keyframe
    reloc.note_lost(31.0)
    reloc.note_frame(_gray(202), 202, None, 31.0)
    # Only the new keyframe and the scanned frame are extracted; every other
    # reference (recent or historical) comes from the cache.
    assert sorted(verifier.extracted) == [30, 202]
    # ...and the cache holds exactly this episode's references, no more.
    assert set(reloc._feature_cache) == {kid for kid, _ in reloc._episode.refs}
    reloc.close(32.0)


def test_without_history_there_is_no_cross_episode_cache():
    verifier = _Counting({})
    reloc = R.LookBackRelocalizer(camera_matrix=np.eye(3), frame_size=(WIDTH, HEIGHT), prompts_enabled=True,
                                  verifier=verifier, synchronous=True)
    _walk_keyframes(reloc, 12)
    reloc.note_lost(10.0)
    reloc.note_frame(_gray(200), 200, None, 10.0)
    reloc.note_frame(_gray(201), 201, None, 30.0)
    verifier.extracted.clear()
    reloc.note_lost(31.0)
    reloc.note_frame(_gray(202), 202, None, 31.0)
    assert len([m for m in verifier.extracted if m != 202]) == 10  # all ten again, as today
    assert reloc._feature_cache == {}
    reloc.close(32.0)


# -- review round 3 (RV-RELOC) -------------------------------------------------------------------------
#
# F1: history is PURELY ADDITIVE -- a scan the recent references decide is
# decided exactly as today, and never consults history.


class _Tracing(_Scripted):
    def __init__(self, table):
        super().__init__(table)
        self.verified = []

    def verify(self, a, b, size):
        self.verified.append((int(a.xy[0, 0]), int(b.xy[0, 0])))
        return super().verify(a, b, size)


def _with_history(table):
    reloc = R.LookBackRelocalizer(camera_matrix=np.eye(3), frame_size=(WIDTH, HEIGHT), prompts_enabled=True,
                                  verifier=_Tracing(table), synchronous=True,
                                  acceptance=R.AcceptanceParams(history_keyframes=6))
    _walk_keyframes(reloc, 40)  # recent: k30..k39; history holds k0, k1, ...
    assert [k for k, _ in reloc._history.members()][:2] == ["k0", "k1"]
    return reloc


def test_a_recent_decision_is_never_replaced_by_a_stronger_historical_one():
    """Recent k38/k39 close a triangle (60/55); historical k0 would be a
    STRONG link (150), which the rule prefers over any triangle when both are
    evaluated together. Additive: the recent triangle stands."""
    table = {(38, 39): 90, (38, 200): 60, (39, 200): 55, (0, 200): 150, (0, 1): 120, (1, 200): 70}
    reloc = _with_history(table)
    ev = reloc.note_lost(10.0) + reloc.note_frame(_gray(200), 200, None, 10.0)
    acc = [p for k, p in ev if k == "recovery_accepted"][0]
    assert acc["by"] == "triangle" and [l["ref_keyframe_id"] for l in acc["links"]] == ["k38", "k39"]
    assert not any(l.get("historical") for l in acc["links"])
    # ...and the historical references were never even verified against it.
    assert not any(a in (0, 1) or b in (0, 1) for a, b in reloc._verifier.verified)
    reloc.close(11.0)


class _Rotating(_Tracing):
    """`table[(a, b)] = (inliers, degrees)`: R is a rotation about z."""

    def verify(self, a, b, size):
        self.verified.append((int(a.xy[0, 0]), int(b.xy[0, 0])))
        key = (int(a.xy[0, 0]), int(b.xy[0, 0]))
        v = self.table.get(key) or self.table.get(key[::-1])
        if not v:
            return None
        n, deg = v
        c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
        return R.Link(n_inliers=n, n_matches=n, R=np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]),
                      t=np.array([1.0, 0, 0]), model="essential")


def test_a_recent_triangle_is_never_replaced_by_a_tighter_mixed_one():
    """The triangle rule keeps the LOWEST closure across all pairs. Recent
    k38/k39 close within 5 deg; the mixed pair k38/k0 would close within 0 deg
    and win if history were evaluated with them. Additive: k38/k39 stands."""
    table = {(38, 39): (90, 0.0), (38, 200): (60, 0.0), (39, 200): (55, 5.0),
             (0, 38): (80, 0.0), (0, 200): (70, 0.0)}
    reloc = R.LookBackRelocalizer(camera_matrix=np.eye(3), frame_size=(WIDTH, HEIGHT), prompts_enabled=True,
                                  verifier=_Rotating(table), synchronous=True,
                                  acceptance=R.AcceptanceParams(history_keyframes=6))
    _walk_keyframes(reloc, 40)
    ev = reloc.note_lost(10.0) + reloc.note_frame(_gray(200), 200, None, 10.0)
    acc = [p for k, p in ev if k == "recovery_accepted"][0]
    assert [l["ref_keyframe_id"] for l in acc["links"]] == ["k38", "k39"] and acc["closure_deg"] == 5.0
    reloc.close(11.0)
    # The same scan with everything evaluated together (the reviewed code)
    # would have taken the mixed pair: the rule itself, on the union.
    links = {"k38": R.Link(60, 60, np.eye(3), np.zeros(3), "essential"),
             "k39": _Rotating(table).verify(R.Features(np.array([[39.0, 0]]), None),
                                            R.Features(np.array([[200.0, 0]]), None), None),
             "k0": R.Link(70, 70, np.eye(3), np.zeros(3), "essential")}
    ids = {"k38": 38, "k39": 39, "k0": 0}
    union = R.evaluate_acceptance(
        links, lambda a, b: _Rotating(table).verify(R.Features(np.array([[float(ids[a]), 0]]), None),
                                                    R.Features(np.array([[float(ids[b]), 0]]), None), None),
        R.AcceptanceParams())
    assert union[0] == "triangle" and set(union[1]) == {"k38", "k0"}


def test_history_is_consulted_only_when_the_recent_references_decide_nothing():
    """A single recent leg (k39, 60) decides nothing; with history it closes a
    MIXED triangle with k0 -- marked historical in the journal (F2)."""
    table = {(39, 200): 60, (0, 200): 70, (0, 39): 90}
    reloc = _with_history(table)
    ev = reloc.note_lost(10.0) + reloc.note_frame(_gray(200), 200, None, 10.0)
    acc = [p for k, p in ev if k == "recovery_accepted"][0]
    by_ref = {l["ref_keyframe_id"]: l for l in acc["links"]}
    assert set(by_ref) == {"k39", "k0"}
    assert by_ref["k0"]["historical"] is True and "historical" not in by_ref["k39"]
    reloc.close(11.0)


def test_a_newer_frame_preempts_the_history_pass_and_the_next_scan_resumes_it():
    """History is spare-cycle work (F1, the scan-schedule half): once a newer
    frame waits, the pass stops -- so today's scans are never delayed by more
    than one verify -- and the next scan picks up where it stopped."""
    reloc = _with_history({})  # nothing links: every scan is undecided
    reloc.note_lost(10.0)
    ep = reloc._episode
    hist = [kid for kid, _ in ep.refs[10:]]
    budget = {"n": 3}

    def waiting():  # a newer frame arrives after three historical references
        budget["n"] -= 1
        return budget["n"] < 0
    reloc._newer_job_waiting = waiting
    v = reloc._verifier
    v.verified.clear()
    reloc._run(R._Job(ep.number, 200, None, _gray(200)))
    first = [a for a, b in v.verified if b == 200 and a < 30]
    assert first == [int(k[1:]) for k in hist[:3]] and ep.history_preempted == 1
    budget["n"] = 100
    v.verified.clear()
    reloc._run(R._Job(ep.number, 201, None, _gray(201)))
    second = [a for a, b in v.verified if b == 201 and a < 30]
    assert second == [int(k[1:]) for k in hist[3:] + hist[:3]]  # resumed, then wrapped
    # ...and the recent references were verified in full both times, first.
    assert [a for a, b in v.verified if b == 201][:10] == list(range(30, 40))
    reloc.close(11.0)


def test_with_history_off_no_link_is_marked():
    reloc = _reloc({(8, 9): 90, (8, 200): 60, (9, 200): 55})
    _walk_keyframes(reloc, 10)
    ev = reloc.note_lost(10.0) + reloc.note_frame(_gray(200), 200, None, 10.0)
    acc = [p for k, p in ev if k == "recovery_accepted"][0]
    assert all("historical" not in l for l in acc["links"])
    reloc.close(11.0)


def test_the_historical_mark_stays_in_the_journal():
    """F2: journal-only. The payload block carries no links at all."""
    acc_payload = {"episode": 1, "by": "triangle", "closure_deg": 1.0,
                   "links": [{"ref_keyframe_id": "s:00000001", "inliers": 70, "historical": True}],
                   "frame": {"source_seq": 9, "keyframe_id": "s:00000009"},
                   "anchor": {"keyframe_id": "s:00000009", "identity": True, "inliers": None}}
    started = R.LookBackRelocalizer(camera_matrix=np.eye(3), frame_size=(WIDTH, HEIGHT),
                                    prompts_enabled=True).started_payload()
    journal = [{"kind": "relocalizer_started", "at": 1.0, "payload": started},
               {"kind": "tracking_lost", "at": 2.0, "payload": {}},
               {"kind": "recovery_accepted", "at": 3.0, "payload": acc_payload}]
    block = R.recovery_block(journal)
    assert block["state"] == "recovered" and "historical" not in json.dumps(block)


# F5: observability never costs an accept or a line.


def test_a_failing_scan_summary_never_costs_the_accept(monkeypatch):
    reloc = _reloc({(8, 9): 90, (8, 200): 60, (9, 200): 55}, summary_events=True)
    _walk_keyframes(reloc, 10)

    def boom(*a, **k):
        raise RuntimeError("injected")
    monkeypatch.setattr(reloc, "_note_scan", boom)
    ev = reloc.note_lost(10.0) + reloc.note_frame(_gray(200), 200, None, 10.0)
    assert "recovery_accepted" in _kinds(ev)
    reloc.close(11.0)


@pytest.mark.parametrize("path", ["note_frame", "tick", "close"])
def test_a_failing_episode_summary_never_drops_a_line_and_is_not_retried(monkeypatch, path):
    reloc = _reloc({(8, 9): 90, (8, 200): 60, (9, 200): 55}, summary_events=True)
    _walk_keyframes(reloc, 10)
    calls = []

    def boom():
        calls.append(1)
        raise RuntimeError("injected")
    monkeypatch.setattr(reloc, "_summaries", boom)
    ev = reloc.note_lost(10.0)
    if path == "note_frame":
        ev += reloc.note_frame(_gray(200), 200, None, 10.0)
        assert "recovery_accepted" in _kinds(ev)
        ev += reloc.note_frame(_gray(201), 201, None, 10.1)
        ev += reloc.tick(10.2)
    elif path == "tick":
        ev += reloc.tick(31.0)  # times out on a rejected frame's tick
        assert "recovery_timed_out" in _kinds(ev)
        ev += reloc.tick(31.1)
    ev += reloc.close(40.0)
    assert _kinds(ev)[-1] == "relocalizer_stopped"
    assert len(calls) == 1  # given up for that episode, not retried on every frame


def test_summarised_is_set_only_once_the_payload_is_built():
    reloc = _reloc({}, summary_events=True)
    _walk_keyframes(reloc, 10)
    reloc.note_lost(10.0)
    reloc.machine.close(11.0)  # resolved; nothing written yet
    assert reloc._summarised == 0
    assert _kinds(reloc._summaries()) == ["recovery_summary"] and reloc._summarised == 1
    assert reloc._summaries() == []
    reloc.close(12.0)


# F6: a closing relocalizer does not verify into finalization.


def test_nothing_is_verified_once_stopped():
    table = {(38, 39): 90, (38, 200): 60, (39, 200): 55}
    reloc = _reloc(table)
    _walk_keyframes(reloc, 40)
    reloc.note_lost(10.0)
    ep = reloc._episode
    reloc._stop = True
    reloc._run(R._Job(ep.number, 200, None, _gray(200)))
    assert ep.refref == {} and reloc.attempts == 0 and not reloc._results
    reloc._stop = False
    reloc.close(11.0)


def test_refref_refuses_to_verify_or_cache_once_stopping():
    """Stopping DURING a scan: the frame's links are in, then close() lands
    before the triangle's reference-reference verify."""
    table = {(38, 39): 90, (38, 200): 60, (39, 200): 55}

    class _StopsMidScan(_Scripted):
        def __init__(self, table, reloc_box):
            super().__init__(table)
            self.box = reloc_box

        def verify(self, a, b, size):
            pair = (int(a.xy[0, 0]), int(b.xy[0, 0]))
            if pair == (39, 200):
                self.box[0]._stop = True  # close() lands here
            return super().verify(a, b, size)

    box = []
    reloc = R.LookBackRelocalizer(camera_matrix=np.eye(3), frame_size=(WIDTH, HEIGHT), prompts_enabled=True,
                                  verifier=_StopsMidScan(table, box), synchronous=True)
    box.append(reloc)
    _walk_keyframes(reloc, 40)
    reloc.note_lost(10.0)
    ep = reloc._episode
    reloc._run(R._Job(ep.number, 200, None, _gray(200)))
    # (39, 200) was the last recent verify; the triangle's reference-reference
    # verify that would follow is refused: nothing cached, nothing accepted.
    assert ep.refref == {} and not reloc._results
    reloc._stop = False
    reloc.close(11.0)


# F9: the GOLDEN journal. With every option unset, HEAD journals exactly
# what 9f4766a (before RELOC2) journaled -- recorded from that commit's
# relocalizer.py / engine.py / events.py by RUN/lead/reloc2/golden/
# golden_record.py (RV-RELOC's module swap), ids masked.

GOLDEN = __import__("pathlib").Path(__file__).parent / "golden" / "world_builder_relocalizer_9f4766a.json"


def _golden_frames(name):
    """KEEP IN STEP with golden_record.py: the recorded walks."""
    K = ss.camera_matrix(WIDTH, HEIGHT)
    room = [ss.encode_jpeg(i) for i in ss.render_sequence(ss.furnished_room(), ss.strafe(12, step=0.09), K, WIDTH, HEIGHT)]
    rng = np.random.default_rng(0)
    noise = [ss.encode_jpeg(rng.integers(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8)) for _ in range(8)]
    if name == "rv":  # RV-RELOC's cmp_old_new.py walk
        return room + noise[:6] + room[::-1][:6] + noise * 3 + room[:8] + noise * 10 + room[::-1] + noise[:3] + room[:5]
    if name == "timeout":  # a prompt that times out, then a later loss that re-links
        return room + noise * 9 + room[:6] + noise * 9 + room[::-1] + noise[:2] + room[:4]
    raise KeyError(name)


def _round_floats(line, places=3):
    def walk(x):
        if isinstance(x, float):
            return round(x, places)
        if isinstance(x, list):
            return [walk(v) for v in x]
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items()}
        return x
    return walk(json.loads(line))


def _engine_walk_golden(root, frames):
    K = ss.camera_matrix(WIDTH, HEIGHT)
    intr = CameraIntrinsics(source="self_calibrated", model="pinhole", fx=float(K[0, 0]), fy=float(K[1, 1]),
                            cx=float(K[0, 2]), cy=float(K[1, 2]), calibrated_width=WIDTH, calibrated_height=HEIGHT)
    clock = _Clock()
    engine = WorldBuilderEngine(WorldStore(root), clock=clock, relocalizer="prompt")  # options: the environment
    wid = engine.create_world("golden")
    sid = engine.start_session(wid, intrinsics=intr, frame_source="synthetic", declared_size=(WIDTH, HEIGHT))
    for seq, jpeg in enumerate(frames):
        clock.t += 0.3
        engine.observe(jpeg, source_seq=seq, wire_seq=seq)
    engine.stop_session()
    return WorldStore(root).events_path(wid, sid).read_text().splitlines()


@pytest.mark.parametrize("scenario", ["rv", "timeout"])
def test_golden_the_default_journal_is_the_pre_reloc2_journal(tmp_path, monkeypatch, scenario):
    import re

    import cv2

    for name in ("TOWER_WORLD_RELOCALIZER_WINDOW", "TOWER_WORLD_RELOCALIZER_HISTORY",
                 "TOWER_WORLD_RELOCALIZER_SUMMARY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(R, "from_session", _synchronous(R.from_session))
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    expected = golden["scenarios"][scenario]
    kinds = [json.loads(l)["kind"] for l in expected]
    assert "recovery_accepted" in kinds and "relocalizer_stopped" in kinds  # it exercises the relocalizer
    lines = [re.sub(r"[0-9a-f]{32}", "ID", line) for line in _engine_walk_golden(tmp_path, _golden_frames(scenario))]
    if (golden["cv2"], golden["numpy"]) == (cv2.__version__, np.__version__):
        assert lines == expected  # byte for byte
    else:  # another OpenCV/NumPy build: RANSAC's last digits may move, nothing else may
        assert [_round_floats(l) for l in lines] == [_round_floats(l) for l in expected]
