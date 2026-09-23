"""The live look-back relocalizer, `tracking.recovery`, and the prompt limiter.

Specification: docs/contracts/WORLD-BUILDER-COMPONENTS.md section 6. The
evidence behind every threshold is P2-LOOKBACK / P2-LM (the run's
experiments); the limiter's values were chosen on the P3-PR replay of the
seven frozen walks through `RecoveryStateMachine` itself.

What is pinned here:

* the state machine's transitions (6.1), including "a loss inside an open
  episode joins it" and a session that stops mid-episode;
* acceptance (6.3): a triangle of two >= 50-inlier links closing within 8
  degrees, or one >= 100-inlier link -- and a single smaller link REFUSED;
* the limiter (6.4): never more than 2 prompts in any closed 60 s window
  under a burst of losses, and with the chosen cooldown the cap never has
  to bind;
* the payload block (6.2): exact keys, fixed arity, null for every journal
  without a relocalizer, prompt ids strictly increasing;
* the revisit-link journal round trip (`revisit_pairs`);
* defaults change nothing: the setting is off, an engine without it
  journals exactly what it journaled before, and turning it on does not
  move a single keyframe decision;
* the frame path never waits for the matcher.
"""

import functools
import json
import time

import numpy as np
import pytest

from tests import synthetic_scene as ss
from tests.result_channel_fixtures import _close_result_channel_clients  # noqa: F401
from tower.world_builder import relocalizer as R
from tower.world_builder.engine import WorldBuilderEngine
from tower.world_builder.records import CameraIntrinsics
from tower.world_builder.store import WorldStore

WIDTH, HEIGHT = 480, 360

OLD_EVENT_KINDS = {
    "session_started", "keyframe_accepted", "frame_rejected", "tracking_lost",
    "solve_chain_broken", "segment_started", "backend_downgraded",
    "mapping_stalled", "build_completed", "session_stopped", "source_seq_restarted",
}

CONTRACT_BLOCK_KEYS = {
    "state", "episode", "lost_at", "resolved_at", "recovered_by",
    "prompts_enabled", "prompt", "counts", "limiter", "acceptance",
}
CONTRACT_COUNT_KEYS = {
    "episodes", "recovered", "recovered_after_prompt", "timed_out", "prompts",
    "withheld_by_limiter", "withheld_disabled",
}
CONTRACT_LIMITER_KEYS = {
    "max_prompts", "window_s", "mechanism", "cooldown_s", "prompt_after_s",
    "timeout_s", "speak_window_s",
}
CONTRACT_PROMPT_KEYS = {"id", "episode", "kind", "issued_at", "speak_until"}


# -- helpers --------------------------------------------------------------------


def _kinds(events):
    return [kind for kind, _ in events]


class _Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def _intrinsics():
    K = ss.camera_matrix(WIDTH, HEIGHT)
    return CameraIntrinsics(
        source="self_calibrated", model="pinhole",
        fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]),
        calibrated_width=WIDTH, calibrated_height=HEIGHT,
    )


@pytest.fixture(scope="module")
def room_frames():
    K = ss.camera_matrix(WIDTH, HEIGHT)
    images = ss.render_sequence(ss.furnished_room(), ss.strafe(10, step=0.09), K, WIDTH, HEIGHT)
    rng = np.random.default_rng(0)
    noise = [rng.integers(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8) for _ in range(6)]
    return [ss.encode_jpeg(i) for i in images], [ss.encode_jpeg(n) for n in noise]


@pytest.fixture
def synchronous(monkeypatch):
    """The engine's relocalizer, answering on the frame it was given."""
    monkeypatch.setattr(
        R, "from_session", functools.partial(R.from_session, synchronous=True)
    )


def _walk(root, frames, *, relocalizer, step=0.3, intrinsics=None):
    clock = _Clock()
    engine = WorldBuilderEngine(WorldStore(root), clock=clock, relocalizer=relocalizer)
    world_id = engine.create_world("reloc")
    session_id = engine.start_session(
        world_id, intrinsics=intrinsics or _intrinsics(), frame_source="synthetic",
        declared_size=(WIDTH, HEIGHT),
    )
    outcomes = []
    for seq, (jpeg, dt) in enumerate(frames):
        clock.t += dt if dt is not None else step
        outcomes.append(engine.observe(jpeg, source_seq=seq, wire_seq=seq).outcome)
    engine.stop_session()
    store = WorldStore(root)
    events = [json.loads(line) for line in store.events_path(world_id, session_id).read_text().splitlines()]
    return store, world_id, session_id, events, outcomes


def _lost_then_revisit(room_frames):
    room, noise = room_frames
    return [(f, None) for f in room] + [(n, None) for n in noise] + [(f, None) for f in room[::-1][:6]]


# -- the state machine (contract 6.1) ----------------------------------------------

LIM = R.LimiterParams()  # the shipped values: after 5 s, timeout 20 s, cooldown 30 s


def test_the_shipped_limiter_values_are_the_replayed_ones():
    assert (LIM.max_prompts, LIM.window_s) == (2, 60.0)
    assert (LIM.mechanism, LIM.cooldown_s) == ("cooldown", 30.0)
    assert (LIM.prompt_after_s, LIM.timeout_s) == (5.0, 20.0)
    # The cooldown alone implies the cap: that is the design, not a coincidence.
    assert LIM.cooldown_s >= LIM.window_s / LIM.max_prompts
    # The Mac review's ceiling (mac-001 E7): a stale prompt is never spoken
    # more than 5 s after issue.
    assert 0 < LIM.speak_window_s <= 5.0


def test_the_prompt_kind_is_the_one_the_phone_knows():
    """An unknown kind is never spoken (contract 6.5 rule 2); keep it stable."""
    assert R.PROMPT_KIND_LOOK_BACK == "look-back"


def test_a_match_before_prompt_after_recovers_silently():
    m = R.RecoveryStateMachine(LIM, prompts_enabled=True)
    opened, ev = m.lost(100.0)
    assert opened and ev == [] and m.state == "searching" and m.episode == 1
    assert m.tick(104.9) == []
    ev = m.accepted(104.9, "triangle", {})
    assert _kinds(ev) == ["recovery_accepted"] and ev[0][1]["episode"] == 1
    assert m.state == "recovered" and m.recovered_by == "triangle"
    assert m.counts["prompts"] == 0 and m.counts["recovered"] == 1
    assert m.counts["recovered_after_prompt"] == 0
    assert m.tick(200.0) == []  # a resolved episode never moves again


def test_prompt_at_prompt_after_then_recovered_after_prompt():
    m = R.RecoveryStateMachine(LIM, prompts_enabled=True)
    m.lost(100.0)
    ev = m.tick(105.0)
    assert ev == [("recovery_prompted", {"prompt_id": 1, "episode": 1})]
    assert m.state == "prompting"
    assert m.tick(106.0) == []  # never a second prompt in one episode
    ev = m.accepted(107.0, "strong-link", {})
    assert m.state == "recovered" and m.counts["recovered_after_prompt"] == 1


def test_no_acceptance_before_timeout_is_timed_out():
    m = R.RecoveryStateMachine(LIM, prompts_enabled=True)
    m.lost(100.0)
    assert _kinds(m.tick(105.0)) == ["recovery_prompted"]
    assert _kinds(m.tick(120.0)) == ["recovery_timed_out"]
    assert m.state == "timed_out" and m.resolved_at == 120.0
    assert m.accepted(121.0, "triangle", {}) == []  # too late: nothing moves


def test_a_loss_inside_an_open_episode_joins_it():
    m = R.RecoveryStateMachine(LIM, prompts_enabled=True)
    m.lost(100.0)
    m.tick(105.0)
    opened, ev = m.lost(106.0)
    assert not opened and ev == []
    assert m.episode == 1 and m.counts["episodes"] == 1 and m.lost_at == 100.0
    assert m.tick(110.0) == []  # no second prompt
    # After it resolves the next loss opens episode 2.
    m.tick(120.0)
    opened, _ = m.lost(121.0)
    assert opened and m.episode == 2 and m.state == "searching"


def test_prompts_disabled_withholds_and_never_prompts():
    m = R.RecoveryStateMachine(LIM, prompts_enabled=False)
    m.lost(100.0)
    ev = m.tick(105.0)
    assert ev == [("recovery_withheld", {"episode": 1, "why": "disabled", "layer": None})]
    assert m.state == "searching" and m.counts["withheld_disabled"] == 1
    assert _kinds(m.tick(120.0)) == ["recovery_timed_out"]


def test_a_session_that_stops_mid_episode_times_it_out():
    m = R.RecoveryStateMachine(LIM, prompts_enabled=True)
    m.lost(100.0)
    assert m.close(101.0) == [("recovery_timed_out", {"episode": 1, "why": "session_stopped"})]
    assert m.close(102.0) == []


# -- the limiter under a burst of losses (contract 6.4) -------------------------------


def _burst(limiter, *, minutes=10, every_s=3.0):
    """A loss every few seconds that never relocalizes: the worst case."""
    m = R.RecoveryStateMachine(limiter, prompts_enabled=True)
    events = []
    t = 0.0
    while t < minutes * 60:
        m.lost(t)
        for tick in np.arange(t, t + every_s, 0.1):
            events += [(float(tick), k, p) for k, p in m.tick(float(tick))]
        t += every_s
    return m, events


def _max_in_closed_window(times, window):
    return max((sum(1 for u in times[i:] if u - t <= window) for i, t in enumerate(times)), default=0)


@pytest.mark.parametrize("cooldown", [None, 20.0, 30.0])
def test_never_more_than_two_prompts_in_any_60s_window(cooldown):
    limiter = R.LimiterParams(prompt_after_s=1.0, timeout_s=2.0,
                              mechanism="cooldown" if cooldown else "none", cooldown_s=cooldown)
    m, events = _burst(limiter)
    prompts = [(t, p) for t, k, p in events if k == "recovery_prompted"]
    assert len(prompts) >= 10  # the burst really did ask for prompts
    assert _max_in_closed_window([t for t, _ in prompts], 60.0) <= 2
    ids = [p["prompt_id"] for _, p in prompts]
    assert ids == list(range(1, len(ids) + 1))  # strictly increasing, never reused
    withheld = [p for t, k, p in events if k == "recovery_withheld"]
    assert len(withheld) == m.counts["withheld_by_limiter"] > 0
    assert m.counts["episodes"] == m.counts["prompts"] + m.counts["withheld_by_limiter"]


def test_with_the_shipped_cooldown_the_cap_never_binds():
    m, events = _burst(R.LimiterParams(prompt_after_s=1.0, timeout_s=2.0))
    layers = {p["layer"] for t, k, p in events if k == "recovery_withheld"}
    assert layers == {"cooldown"}


def test_a_refused_prompt_is_never_issued_later():
    limiter = R.LimiterParams(prompt_after_s=1.0, timeout_s=25.0, cooldown_s=30.0)
    m = R.RecoveryStateMachine(limiter, prompts_enabled=True)
    m.lost(0.0)
    assert _kinds(m.tick(1.0)) == ["recovery_prompted"]
    assert _kinds(m.tick(25.0)) == ["recovery_timed_out"]
    m.lost(26.0)
    assert _kinds(m.tick(27.0)) == ["recovery_withheld"]  # 26 s after the last prompt
    # The cooldown expires at 31 s while episode 2 is still open: a late
    # "look back" is wrong advice, so nothing is issued.
    assert m.tick(31.5) == [] and m.state == "searching"
    assert _kinds(m.tick(51.0)) == ["recovery_timed_out"]
    assert m.counts["prompts"] == 1 and m.counts["withheld_by_limiter"] == 1


# -- acceptance (contract 6.3) ---------------------------------------------------------


def _rot(axis, deg):
    import cv2

    v = np.asarray(axis, float)
    v = v / np.linalg.norm(v) * np.radians(deg)
    return cv2.Rodrigues(v)[0]


def _link(inliers, R_=None):
    return R.Link(n_inliers=inliers, n_matches=inliers * 2, R=np.eye(3) if R_ is None else R_,
                  t=np.array([1.0, 0, 0]), model="essential")


A = R.AcceptanceParams()


def test_a_single_link_under_100_inliers_is_refused():
    assert R.evaluate_acceptance({"k1": _link(99)}, lambda a, b: None, A) is None
    assert R.evaluate_acceptance({"k1": _link(60)}, lambda a, b: _link(80), A) is None


def test_one_link_of_100_inliers_is_a_strong_link():
    by, used, closure = R.evaluate_acceptance({"k1": _link(100), "k2": _link(40)}, lambda a, b: None, A)
    assert (by, used, closure) == ("strong-link", ["k1"], None)


def test_a_closing_triangle_of_two_50_inlier_links_is_accepted():
    R_k1 = _rot([0, 1, 0], 10)  # ref1 -> frame
    R_12 = _rot([0, 1, 0], -4)  # ref1 -> ref2
    R_k2 = R_k1 @ R_12.T @ _rot([1, 0, 0], 3)  # ref2 -> frame, 3 deg of closure error
    links = {"k1": _link(50, R_k1), "k2": _link(55, R_k2)}
    by, used, closure = R.evaluate_acceptance(links, lambda a, b: _link(80, R_12), A)
    assert by == "triangle" and set(used) == {"k1", "k2"}
    assert closure == pytest.approx(3.0, abs=0.01)


def test_a_triangle_that_does_not_close_within_8_degrees_is_refused():
    R_k1 = _rot([0, 1, 0], 10)
    R_12 = _rot([0, 1, 0], -4)
    R_k2 = R_k1 @ R_12.T @ _rot([1, 0, 0], 9)
    links = {"k1": _link(50, R_k1), "k2": _link(55, R_k2)}
    assert R.evaluate_acceptance(links, lambda a, b: _link(80, R_12), A) is None


def test_a_triangle_needs_both_legs_at_50_and_a_verified_reference_pair():
    R_k1, R_12 = _rot([0, 1, 0], 10), _rot([0, 1, 0], -4)
    R_k2 = R_k1 @ R_12.T
    assert R.evaluate_acceptance(
        {"k1": _link(49, R_k1), "k2": _link(90, R_k2)}, lambda a, b: _link(80, R_12), A) is None
    assert R.evaluate_acceptance(
        {"k1": _link(60, R_k1), "k2": _link(90, R_k2)}, lambda a, b: None, A) is None


# -- the payload block (contract 6.2) -----------------------------------------------------


def _ev(kind, at, payload=None):
    return {"kind": kind, "at": at, "payload": payload or {}}


def _started(prompts_enabled=True):
    reloc = R.LookBackRelocalizer(camera_matrix=np.eye(3), frame_size=(WIDTH, HEIGHT),
                                  prompts_enabled=prompts_enabled)
    return _ev("relocalizer_started", 1.0, reloc.started_payload())


def test_old_journals_have_no_recovery_and_their_summary_is_unchanged():
    from tower.results.world_builder import _summarise_events, _tracking_block

    events = ([{"kind": "session_started"}] + [{"kind": "keyframe_accepted"}] * 4
              + [{"kind": "tracking_lost"}])
    summary = _summarise_events(events)
    assert "recovery" not in summary
    assert R.recovery_block(events) is None
    assert _tracking_block(summary)["recovery"] is None


def test_the_block_has_exactly_the_contract_keys_and_fixed_arity():
    from tower.results.world_builder import _summarise_events, _tracking_block

    journal = [
        _ev("session_started", 1.0), _started(),
        _ev("keyframe_accepted", 2.0),
        _ev("tracking_lost", 10.0),
        _ev("recovery_prompted", 15.0, {"prompt_id": 1, "episode": 1}),
        _ev("tracking_lost", 16.0),  # joins
        _ev("recovery_accepted", 17.0, {"episode": 1, "by": "triangle", "links": []}),
        _ev("tracking_lost", 50.0),
        _ev("recovery_withheld", 55.0, {"episode": 2, "why": "limiter", "layer": "cooldown"}),
        _ev("recovery_timed_out", 70.0, {"episode": 2}),
    ]
    tracking = _tracking_block(_summarise_events(journal))
    block = tracking["recovery"]
    assert set(block) == CONTRACT_BLOCK_KEYS
    assert set(block["counts"]) == CONTRACT_COUNT_KEYS
    assert set(block["limiter"]) == CONTRACT_LIMITER_KEYS
    assert set(block["acceptance"]) == {"matcher", "reference_keyframes", "scan_hz", "triangle", "strong_link"}
    assert set(block["acceptance"]["triangle"]) == {"min_links", "min_link_inliers", "max_closure_deg"}
    assert set(block["prompt"]) == CONTRACT_PROMPT_KEYS
    assert block["state"] == "timed_out" and block["episode"] == 2
    assert block["lost_at"] == 50.0 and block["resolved_at"] == 70.0
    assert block["recovered_by"] is None
    assert block["prompt"] == {"id": 1, "episode": 1, "kind": "look-back", "issued_at": 15.0,
                               "speak_until": 15.0 + block["limiter"]["speak_window_s"]}
    assert block["counts"] == {"episodes": 2, "recovered": 1, "recovered_after_prompt": 1,
                               "timed_out": 1, "prompts": 1, "withheld_by_limiter": 1,
                               "withheld_disabled": 0}
    # Fixed arity whatever the journal length.
    long = journal[:2] + journal[2:] * 200
    assert set(_summarise_events(long)["recovery"]) == CONTRACT_BLOCK_KEYS
    # No volatile field: the same journal twice gives the same block.
    assert _summarise_events(journal)["recovery"] == _summarise_events(list(journal))["recovery"]


def test_prompt_ids_only_rise_and_stale_events_never_move_the_block():
    journal = [
        _started(), _ev("tracking_lost", 10.0),
        _ev("recovery_prompted", 15.0, {"prompt_id": 3, "episode": 1}),
        _ev("recovery_timed_out", 30.0, {"episode": 1}),
        _ev("tracking_lost", 40.0),
        _ev("recovery_prompted", 45.0, {"prompt_id": 3, "episode": 2}),  # reused id: ignored
        _ev("recovery_accepted", 46.0, {"episode": 1, "by": "triangle"}),  # stale episode: ignored
    ]
    block = R.recovery_block(journal)
    assert block["prompt"]["id"] == 3 and block["prompt"]["episode"] == 1
    assert block["state"] == "searching" and block["episode"] == 2
    assert block["counts"]["prompts"] == 1 and block["counts"]["recovered"] == 0


def test_a_fresh_session_with_a_relocalizer_is_none_not_null():
    block = R.recovery_block([_started(prompts_enabled=False)])
    assert block["state"] == "none" and block["episode"] == 0
    assert block["lost_at"] is None and block["prompt"] is None
    assert block["prompts_enabled"] is False


# -- on the wire ------------------------------------------------------------------------------


def _envelope(monkeypatch, root):
    from tests.result_channel_fixtures import drain, make_client, subscribe

    client = make_client(monkeypatch, root)
    with client.websocket_connect("/ws") as ws:
        subscribe(ws)
        return drain(ws, expect="cartridge_result")


def test_on_the_wire_an_old_world_says_null(monkeypatch, tmp_path):
    from tests.result_channel_fixtures import build_world

    build_world(tmp_path, frames=6)
    envelope = _envelope(monkeypatch, tmp_path)
    assert "tower_sent_at" in envelope
    assert "recovery" in envelope["payload"]["tracking"]
    assert envelope["payload"]["tracking"]["recovery"] is None


def test_on_the_wire_a_relocalizer_session_carries_the_block(
    monkeypatch, tmp_path, room_frames, synchronous
):
    _walk(tmp_path, _lost_then_revisit(room_frames), relocalizer="prompt")
    envelope = _envelope(monkeypatch, tmp_path)
    assert isinstance(envelope.get("tower_sent_at"), (int, float))  # mac-001 E2 needs it
    block = envelope["payload"]["tracking"]["recovery"]
    assert set(block) == CONTRACT_BLOCK_KEYS
    assert block["state"] == "recovered" and block["recovered_by"] in ("triangle", "strong-link")


# -- the engine, end to end ------------------------------------------------------------------


def test_a_revisit_after_a_loss_is_a_verified_link_the_final_solve_can_match(
    tmp_path, room_frames, synchronous
):
    store, world_id, session_id, events, _ = _walk(
        tmp_path, _lost_then_revisit(room_frames), relocalizer="prompt"
    )
    kinds = [e["kind"] for e in events]
    assert kinds[:2] == ["session_started", "relocalizer_started"]
    accepted = [e for e in events if e["kind"] == "recovery_accepted"]
    assert len(accepted) == 1
    payload = accepted[0]["payload"]
    assert payload["by"] in ("triangle", "strong-link")
    assert all(link["inliers"] >= 50 for link in payload["links"])
    # The journal round trip: names that exist on disk, reference first.
    pairs = R.revisit_pairs(store.session_dir(world_id, session_id))
    assert pairs
    names = {k.image_relpath.split("/")[-1] for k in store.read_keyframes(world_id, session_id)}
    for ref, anchor in pairs:
        assert ref in names and anchor in names and ref < anchor
        assert (store.session_dir(world_id, session_id) / "images" / anchor).is_file()
    block = R.recovery_block(events)
    assert block["state"] == "recovered" and block["counts"]["prompts"] == 0


def _lost_for_good(room_frames):
    room, noise = room_frames
    # One second per frame after the loss: only unrelated views for 25 s.
    return [(f, 0.3) for f in room] + [(noise[i % len(noise)], 1.0) for i in range(26)]


def test_an_unrecovered_loss_prompts_once_then_times_out(tmp_path, room_frames, synchronous):
    _, _, _, events, _ = _walk(tmp_path, _lost_for_good(room_frames), relocalizer="prompt")
    kinds = [e["kind"] for e in events]
    assert kinds.count("recovery_prompted") == 1  # later losses meet the cooldown
    assert "recovery_accepted" not in kinds
    prompt = next(e for e in events if e["kind"] == "recovery_prompted")
    lost = next(e for e in events if e["kind"] == "tracking_lost")
    assert prompt["payload"] == {"prompt_id": 1, "episode": 1}
    assert prompt["at"] - lost["at"] >= 5.0
    timed = next(e for e in events if e["kind"] == "recovery_timed_out")
    assert timed["payload"] == {"episode": 1}
    assert timed["at"] - lost["at"] >= 20.0
    block = R.recovery_block(events)
    assert block["prompt"]["id"] == 1 and block["state"] == "timed_out"
    # Every episode is resolved by the time the session record is closed.
    assert kinds.index("session_stopped") > max(
        i for i, k in enumerate(kinds) if k == "recovery_timed_out")


def test_silent_mode_records_and_never_prompts(tmp_path, room_frames, synchronous):
    _, _, _, events, _ = _walk(tmp_path, _lost_for_good(room_frames), relocalizer="silent")
    kinds = [e["kind"] for e in events]
    assert "relocalizer_started" in kinds and "recovery_prompted" not in kinds
    block = R.recovery_block(events)
    assert block["prompts_enabled"] is False and block["counts"]["withheld_disabled"] >= 1
    assert block["prompt"] is None and block["counts"]["prompts"] == 0


# -- defaults change nothing -----------------------------------------------------------------


def test_the_setting_defaults_to_off_and_garbage_is_off(monkeypatch):
    from tower.config import get_settings, world_relocalizer_setting

    monkeypatch.delenv("TOWER_WORLD_RELOCALIZER", raising=False)
    assert get_settings().world_relocalizer == "off"
    for value, expected in (("prompt", "prompt"), ("on", "prompt"), ("silent", "silent"),
                            ("off", "off"), ("", "off"), ("pormpt", "off")):
        monkeypatch.setenv("TOWER_WORLD_RELOCALIZER", value)
        assert world_relocalizer_setting() == expected


def test_an_engine_without_the_setting_journals_exactly_what_it_did(tmp_path, room_frames, monkeypatch):
    monkeypatch.delenv("TOWER_WORLD_RELOCALIZER", raising=False)
    _, _, _, events, _ = _walk(tmp_path, _lost_then_revisit(room_frames), relocalizer=None)
    kinds = {e["kind"] for e in events}
    assert "tracking_lost" in kinds  # the walk did lose tracking
    assert kinds <= OLD_EVENT_KINDS
    assert R.recovery_block(events) is None


def test_turning_it_on_moves_no_keyframe_decision(tmp_path, room_frames, synchronous):
    walk = _lost_then_revisit(room_frames)
    s_off, w_off, id_off, ev_off, out_off = _walk(tmp_path / "off", walk, relocalizer="off")
    s_on, w_on, id_on, ev_on, out_on = _walk(tmp_path / "on", walk, relocalizer="prompt")
    assert out_on == out_off
    strip = [(e["kind"], e["at"], {k: v for k, v in e["payload"].items() if k != "keyframe_id"})
             for e in ev_on if e["kind"] in OLD_EVENT_KINDS]
    assert strip == [(e["kind"], e["at"], {k: v for k, v in e["payload"].items() if k != "keyframe_id"})
                     for e in ev_off]
    seqs = lambda s, w, i: [(k.source_seq, k.segment_index) for k in s.read_keyframes(w, i)]  # noqa: E731
    assert seqs(s_on, w_on, id_on) == seqs(s_off, w_off, id_off)


def test_an_uncalibrated_session_gets_no_relocalizer(tmp_path, room_frames):
    _, _, _, events, _ = _walk(
        tmp_path, _lost_then_revisit(room_frames), relocalizer="prompt",
        intrinsics=CameraIntrinsics.unknown(),
    )
    assert "relocalizer_started" not in {e["kind"] for e in events}
    assert R.recovery_block(events) is None


# -- the revisit-link journal round trip -------------------------------------------------------


def _write_session(tmp_path, keyframe_ids, events):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "keyframes.jsonl").write_text("".join(
        json.dumps({"keyframe_id": k, "image_relpath": f"images/{i:08d}.jpg"}) + "\n"
        for i, k in enumerate(keyframe_ids)), encoding="utf-8")
    (tmp_path / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events)
                                           + '{"torn', encoding="utf-8")
    return tmp_path


def test_revisit_pairs_round_trip(tmp_path):
    kf = [f"s:{i:08d}" for i in range(12)]
    session = _write_session(tmp_path / "s", kf, [
        _started(), _ev("tracking_lost", 10.0),
        # anchored at once (the scanned frame was keyframe 8)
        _ev("recovery_accepted", 11.0, {"episode": 1, "by": "triangle",
            "links": [{"ref_keyframe_id": kf[2]}, {"ref_keyframe_id": kf[3]}],
            "anchor": {"keyframe_id": kf[8], "identity": True, "inliers": None}}),
        _ev("tracking_lost", 30.0),
        # anchored afterwards
        _ev("recovery_accepted", 31.0, {"episode": 2, "by": "strong-link",
            "links": [{"ref_keyframe_id": kf[5]}], "anchor": None}),
        _ev("recovery_anchored", 31.5, {"episode": 2, "anchor": {"keyframe_id": kf[10]}}),
        # an anchor for an episode whose acceptance was never journaled: ignored
        _ev("recovery_anchored", 40.0, {"episode": 3, "anchor": {"keyframe_id": kf[11]}}),
    ])
    assert R.revisit_pairs(session) == [
        ("00000002.jpg", "00000008.jpg"), ("00000003.jpg", "00000008.jpg"),
        ("00000005.jpg", "00000010.jpg"),
    ]


def test_revisit_pairs_of_an_old_or_missing_session_is_empty(tmp_path):
    assert R.revisit_pairs(tmp_path / "nowhere") == []
    session = _write_session(tmp_path / "old", ["s:1"], [_ev("session_started", 1.0),
                                                        _ev("tracking_lost", 2.0)])
    assert R.revisit_pairs(session) == []


# -- the frame path never waits ------------------------------------------------------------------


class _SlowVerifier:
    def __init__(self, delay):
        self.delay = delay

    def features(self, gray):
        time.sleep(self.delay)
        return R.Features(np.zeros((0, 2)), np.zeros((0, 128), np.float32))

    def verify(self, a, b, size):
        return None


def test_the_frame_path_never_waits_for_the_matcher():
    reloc = R.LookBackRelocalizer(camera_matrix=np.eye(3), frame_size=(WIDTH, HEIGHT),
                                  prompts_enabled=True, verifier=_SlowVerifier(0.4))
    gray = np.zeros((HEIGHT, WIDTH), np.uint8)
    for i in range(5):
        reloc.note_keyframe(f"k{i}", i, gray)
    started = time.perf_counter()
    reloc.note_lost(10.0)
    for i in range(20):
        reloc.note_frame(gray, 10 + i, None, 10.0 + 0.5 * i)  # a scan is due on every one
    elapsed = time.perf_counter() - started
    reloc.close(30.0)
    assert elapsed < 0.3  # 20 frames, each of which would cost >= 0.4 s if it waited
    assert reloc.dropped_frames > 0  # latest-wins, not a queue


def test_the_worker_is_idle_until_a_loss():
    reloc = R.LookBackRelocalizer(camera_matrix=np.eye(3), frame_size=(WIDTH, HEIGHT),
                                  prompts_enabled=True, verifier=_SlowVerifier(0.0))
    gray = np.zeros((HEIGHT, WIDTH), np.uint8)
    for i in range(30):
        reloc.note_keyframe(f"k{i}", i, gray)
        assert reloc.note_frame(gray, i, f"k{i}", float(i)) == []
    assert reloc._thread is None and reloc.attempts == 0
    reloc.close(40.0)
