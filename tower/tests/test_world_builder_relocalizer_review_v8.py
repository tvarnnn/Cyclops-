"""Review V8 (reviewer RVX) on the look-back relocalizer: M4's relocalizer half, LOW-1..4.

The findings, and what each test below pins:

* **M4 (the relocalizer's half).** A revisit pair is the (reference, anchor)
  pair the final solve imports. Its reference leg always met the live floor
  (a triangle leg >= 50, a strong link >= 100), but the anchor leg -- the
  scanned frame tied to a post-loss keyframe by `_anchor` / `_run_anchor` --
  took ANY `verify()` link (>= 30 inliers). Now every live leg must meet
  `REVISIT_MIN_INLIERS` (50, `AcceptanceParams.triangle_min_link_inliers`,
  P2-LOOKBACK `tri2_50`) before the relocalizer anchors, and
  `revisit_pairs` never returns a pair with a leg below it, whatever journal
  it reads.
* **LOW-1.** Frames the engine rejects before tracking (undecodable, or at
  another size) never ticked the episode, so it could stay `searching` for
  ever. They tick it now, with no matching work.
* **LOW-2.** A worker that failed on every attempt was only logged. After
  `MAX_WORKER_FAILURES` consecutive failures the relocalizer stops:
  `relocalizer_stopped {why: error}`.
* **LOW-3.** The 2-per-60 s cap was enforced on the duration clock and shown
  on the Tower clock, with one 15.6 ms tick between them. Each clock is now
  checked on its own readings: the Tower-clock check compares the journal's
  own `at`s, so the bound holds exactly on the wire.
* **LOW-4.** The module docstring said revisit pairs are not wired into
  `global_solve`; they are.
"""

import functools
import inspect
import json
import time

import numpy as np
import pytest

from tests import synthetic_scene as ss
from tests.test_world_builder_relocalizer import (  # noqa: F401 -- fixtures
    HEIGHT,
    WIDTH,
    _Clock,
    _intrinsics,
    _kinds,
    _lost_for_good,
    _SlowVerifier,
    _walk,
    _write_session,
    room_frames,
    synchronous,
)
from tower.world_builder import relocalizer as R
from tower.world_builder.engine import WorldBuilderEngine
from tower.world_builder.store import WorldStore

LIM = R.LimiterParams()
# One tick of this machine's `time.monotonic` (GetTickCount64) and `time.time`
# (GetSystemTimeAsFileTime): `time.get_clock_info(...).resolution`.
TICK = 0.015625


def _gray(tag):
    gray = np.zeros((HEIGHT, WIDTH), np.uint8)
    gray[0, 0] = tag
    return gray


class _TableVerifier:
    """Features carry the frame's first pixel as a tag; links come from a table
    of (tag_a, tag_b) -> inliers."""

    def __init__(self, table):
        self.table = dict(table)
        self.features_calls = 0

    def features(self, gray):
        self.features_calls += 1
        return R.Features(np.array([[float(gray[0, 0]), 0.0]]), np.zeros((1, 128), np.float32))

    def verify(self, a, b, size):
        n = self.table.get((int(a.xy[0, 0]), int(b.xy[0, 0])))
        if n is None:
            return None
        return R.Link(n_inliers=n, n_matches=2 * n, R=np.eye(3), t=np.array([1.0, 0.0, 0.0]),
                      model="essential")


def _table_relocalizer(table):
    reloc = R.LookBackRelocalizer(camera_matrix=np.eye(3), frame_size=(WIDTH, HEIGHT),
                                  prompts_enabled=True, verifier=_TableVerifier(table),
                                  synchronous=True)
    for tag in range(1, 6):
        reloc.note_keyframe(f"k{tag}", tag, _gray(tag))
    return reloc


def _session_from(tmp_path, keyframe_ids, events):
    """A session directory whose journal holds `events` ((kind, payload) tuples)."""
    return _write_session(tmp_path, keyframe_ids, [
        {"kind": kind, "at": 100.0 + i, "payload": payload} for i, (kind, payload) in enumerate(events)
    ])


# -- M4: the relocalizer never writes a revisit pair below its own floor ------------------------


def test_the_revisit_floor_is_the_relocalizers_own_50():
    """The floor the final solve is told to import at is the live path's, not a new number."""
    assert R.REVISIT_MIN_INLIERS == R.AcceptanceParams().triangle_min_link_inliers == 50
    # Every link that can make an acceptance clears it: triangle legs at 50, a strong link at 100.
    assert R.AcceptanceParams().strong_link_min_inliers >= R.REVISIT_MIN_INLIERS


def test_an_anchor_link_below_the_floor_is_never_the_revisit_anchor(tmp_path):
    """The nearest post-loss keyframe verifies against the scanned frame with
    40 inliers: before, it became the anchor and (k1, p200) a revisit pair.
    Now the acceptance waits for an anchor at the floor: p202 (45) is a
    failed try, p201 (60) is the anchor."""
    reloc = _table_relocalizer({(1, 100): 120, (100, 200): 40, (100, 202): 45, (100, 201): 60})
    events = reloc.note_lost(10.0)
    reloc.note_keyframe("p200", 20, _gray(200))  # post-loss, before the scan
    events += reloc.note_frame(_gray(100), 21, None, 10.5)  # the scan: a strong link to k1
    accepted = [p for k, p in events if k == "recovery_accepted"]
    assert len(accepted) == 1 and accepted[0]["by"] == "strong-link"
    assert accepted[0]["anchor"] is None  # p200's 40 inliers do not anchor it
    assert all("R_anchor_ref" not in link for link in accepted[0]["links"])
    reloc.note_keyframe("p202", 22, _gray(202))  # 45: still below the floor, one try spent
    events += reloc.note_frame(_gray(202), 22, "p202", 11.0)
    assert "recovery_anchored" not in _kinds(events)
    reloc.note_keyframe("p201", 23, _gray(201))  # 60: the anchor
    events += reloc.note_frame(_gray(201), 23, "p201", 11.5)
    anchored = [p for k, p in events if k == "recovery_anchored"]
    assert [a["anchor"] for a in anchored] == [
        {"keyframe_id": "p201", "identity": False, "inliers": 60}]
    reloc.close(12.0)

    kf = ["k1", "k2", "k3", "k4", "k5", "p200", "p201", "p202"]
    session = _session_from(tmp_path / "s", kf, events)
    assert R.revisit_pairs(session) == [("00000000.jpg", "00000006.jpg")]  # (k1, p201)


def test_the_nearest_anchor_at_the_floor_wins_among_post_loss_keyframes():
    """With two post-loss keyframes already in hand, the nearer one below the
    floor is skipped for the farther one at it."""
    reloc = _table_relocalizer({(1, 100): 120, (100, 200): 49, (100, 201): 50})
    events = reloc.note_lost(10.0)
    reloc.note_keyframe("p201", 18, _gray(201))
    reloc.note_keyframe("p200", 20, _gray(200))
    events += reloc.note_frame(_gray(100), 21, None, 10.5)
    accepted = [p for k, p in events if k == "recovery_accepted"]
    assert accepted[0]["anchor"] == {"keyframe_id": "p201", "identity": False, "inliers": 50}
    reloc.close(12.0)


def test_an_acceptance_no_keyframe_anchors_at_the_floor_writes_no_revisit_pair(tmp_path):
    """Three later keyframes, all below the floor: the pending anchor is given
    up (MAX_ANCHOR_TRIES) and nothing reaches the solve."""
    reloc = _table_relocalizer({(1, 100): 120, (100, 200): 30, (100, 201): 49, (100, 202): 45,
                                (100, 203): 90})
    events = reloc.note_lost(10.0)
    events += reloc.note_frame(_gray(100), 21, None, 10.5)
    assert [p["anchor"] for k, p in events if k == "recovery_accepted"] == [None]
    for i, tag in enumerate((200, 201, 202, 203)):  # the fourth is past MAX_ANCHOR_TRIES
        reloc.note_keyframe(f"p{tag}", 22 + i, _gray(tag))
        events += reloc.note_frame(_gray(tag), 22 + i, f"p{tag}", 11.0 + i)
    assert "recovery_anchored" not in _kinds(events)
    reloc.close(20.0)
    session = _session_from(tmp_path / "s", ["k1", "k2", "k3", "k4", "k5", "p200", "p201", "p202",
                                             "p203"], events)
    assert R.revisit_pairs(session) == []


def test_revisit_pairs_never_returns_a_pair_below_the_live_floor(tmp_path):
    """Whatever journal it reads -- one written before this fix included -- a
    pair is returned only when every live leg met REVISIT_MIN_INLIERS: each
    reference link's `inliers`, and the anchor's unless it IS the scanned
    frame. A leg without a count is below the floor."""
    kf = [f"s:{i:08d}" for i in range(12)]
    session = _write_session(tmp_path / "s", kf, [
        {"kind": "relocalizer_started", "at": 1.0, "payload": {}},
        # 1: a strong link, anchored at 49 -> no pair
        {"kind": "recovery_accepted", "at": 11.0, "payload": {
            "episode": 1, "by": "strong-link", "links": [{"ref_keyframe_id": kf[2], "inliers": 120}],
            "anchor": {"keyframe_id": kf[8], "identity": False, "inliers": 49}}},
        # 2: a triangle whose second leg is 49 (only possible with other params) -> one pair
        {"kind": "recovery_accepted", "at": 21.0, "payload": {
            "episode": 2, "by": "triangle",
            "links": [{"ref_keyframe_id": kf[3], "inliers": 50}, {"ref_keyframe_id": kf[4], "inliers": 49}],
            "anchor": {"keyframe_id": kf[9], "identity": True, "inliers": None}}},
        # 3: anchored afterwards at 30 -> no pair
        {"kind": "recovery_accepted", "at": 31.0, "payload": {
            "episode": 3, "by": "strong-link", "links": [{"ref_keyframe_id": kf[5], "inliers": 100}],
            "anchor": None}},
        {"kind": "recovery_anchored", "at": 31.5, "payload": {
            "episode": 3, "anchor": {"keyframe_id": kf[10], "identity": False, "inliers": 30}}},
        # 4: a link with no count -> no pair
        {"kind": "recovery_accepted", "at": 41.0, "payload": {
            "episode": 4, "by": "strong-link", "links": [{"ref_keyframe_id": kf[6]}],
            "anchor": {"keyframe_id": kf[11], "identity": True}}},
        # 5: anchored afterwards at exactly 50 -> a pair (the floor is inclusive)
        {"kind": "recovery_accepted", "at": 51.0, "payload": {
            "episode": 5, "by": "strong-link", "links": [{"ref_keyframe_id": kf[1], "inliers": 100}],
            "anchor": None}},
        {"kind": "recovery_anchored", "at": 51.5, "payload": {
            "episode": 5, "anchor": {"keyframe_id": kf[7], "identity": False, "inliers": 50}}},
    ])
    assert R.revisit_pairs(session) == [("00000003.jpg", "00000009.jpg"),
                                        ("00000001.jpg", "00000007.jpg")]


# -- LOW-1: rejected frames tick the episode, and do no matching -------------------------------


class _CountingEngine(WorldBuilderEngine):
    """Counts SIFT work done inside each observe(), by outcome reason."""

    work_by_reason: dict = {}

    def observe(self, raw, **kw):
        before = _SIFT_CALLS["n"]
        result = super().observe(raw, **kw)
        self.work_by_reason[result.reason] = (
            self.work_by_reason.get(result.reason, 0) + _SIFT_CALLS["n"] - before)
        return result


_SIFT_CALLS = {"n": 0}


@pytest.fixture
def counted_sift(monkeypatch):
    real = R.SiftVerifier.features

    def features(self, gray):
        _SIFT_CALLS["n"] += 1
        return real(self, gray)

    monkeypatch.setattr(R.SiftVerifier, "features", features)
    _CountingEngine.work_by_reason = {}
    yield _CountingEngine.work_by_reason


def _rejected_frame(kind):
    if kind == "malformed_frame":
        return b"not a jpeg at all"
    rng = np.random.default_rng(7)
    return ss.encode_jpeg(rng.integers(0, 255, (HEIGHT * 2, WIDTH * 2, 3), dtype=np.uint8))


@pytest.mark.parametrize("reason", ["frame_size_changed", "malformed_frame"])
def test_frames_the_engine_rejects_still_tick_the_episode(tmp_path, room_frames, synchronous,
                                                           counted_sift, reason):
    """The reviewer's probe A as a test: after the loss every frame for 30 s
    is rejected before tracking. The episode must time out AT timeout_s, on
    the frames' own clock -- not stay `searching` until the stop line."""
    room, noise = room_frames
    walk = ([(f, 0.3) for f in room] + [(noise[0], 0.3), (noise[1], 0.3)]
            + [(_rejected_frame(reason), 1.0)] * 30)
    _, _, _, events, _ = _walk(tmp_path, walk, relocalizer="prompt", engine_cls=_CountingEngine)
    assert sum(1 for e in events if e["kind"] == "frame_rejected"
               and e["payload"]["reason"] == reason) == 30
    lost = next(e for e in events if e["kind"] == "tracking_lost")
    timed = [e for e in events if e["kind"] == "recovery_timed_out"]
    assert [e["payload"] for e in timed] == [{"episode": 1}]  # by the timeout, not `session_stopped`
    assert 20.0 <= timed[0]["at"] - lost["at"] <= 21.0
    assert R.recovery_block(events)["state"] == "timed_out"
    # A rejected frame is never matched: the relocalizer did its SIFT on the
    # frames that were decoded, and none on the 30 it was only told the time by.
    assert counted_sift.get(reason, 0) == 0
    assert sum(counted_sift.values()) > 0  # the loss frame itself was scanned


def test_a_tick_hands_the_worker_nothing_and_never_waits():
    """`tick` is time only: with a matcher that takes 0.4 s per call, fifty
    ticks inside an open episode start no thread, submit no scan, and cost
    nothing; the prompt and the timeout still happen on time."""
    reloc = R.LookBackRelocalizer(camera_matrix=np.eye(3), frame_size=(WIDTH, HEIGHT),
                                  prompts_enabled=True, verifier=_SlowVerifier(0.4))
    for i in range(5):
        reloc.note_keyframe(f"k{i}", i, np.zeros((HEIGHT, WIDTH), np.uint8))
    events = reloc.note_lost(0.0)
    started = time.perf_counter()
    for i in range(1, 51):
        events += reloc.tick(i * 0.5)
    elapsed = time.perf_counter() - started
    assert elapsed < 0.05
    assert reloc._thread is None and reloc._pending is None and reloc.attempts == 0
    assert _kinds(events) == ["recovery_prompted", "recovery_timed_out"]
    assert reloc.tick(30.0) == []  # a resolved episode: nothing
    reloc.close(31.0)


def test_the_frame_path_never_waits_with_ticks_scans_and_anchors_mixed():
    """The existing never-waits test, extended to every frame-thread entry
    point this review added or touched: decoded frames, rejected-frame
    ticks, keyframes inside an open episode, and the prompt's journal
    feedback -- against a 0.4 s matcher."""
    reloc = R.LookBackRelocalizer(camera_matrix=np.eye(3), frame_size=(WIDTH, HEIGHT),
                                  prompts_enabled=True, verifier=_SlowVerifier(0.4),
                                  wall_clock=_Clock(1000.0))
    gray = np.zeros((HEIGHT, WIDTH), np.uint8)
    for i in range(5):
        reloc.note_keyframe(f"k{i}", i, gray)
    started = time.perf_counter()
    reloc.note_lost(10.0)
    for i in range(20):
        now = 10.0 + 0.5 * i
        for kind, _ in reloc.note_frame(gray, 10 + i, None, now):
            if kind == "recovery_prompted":
                reloc.prompt_journaled(1000.0 + now)
        reloc.tick(now + 0.25)
        reloc.note_keyframe(f"p{i}", 100 + i, gray)
    elapsed = time.perf_counter() - started
    reloc.close(30.0)
    assert elapsed < 0.3  # 20 scans, each of which would cost >= 0.4 s if it waited
    assert reloc.dropped_frames > 0


# -- LOW-2: a worker that keeps failing stops the relocalizer ---------------------------------


class _Boom:
    def features(self, gray):
        raise RuntimeError("injected: cv2.error stand-in")

    def verify(self, a, b, size):
        return None


class _FailOnOddFrames:
    """Scans of a frame whose first pixel is odd raise; the rest succeed."""

    def features(self, gray):
        if int(gray[0, 0]) % 2:
            raise RuntimeError("injected")
        return R.Features(np.zeros((0, 2)), np.zeros((0, 128), np.float32))

    def verify(self, a, b, size):
        return None


_NEVER_TIMES_OUT = R.LimiterParams(prompt_after_s=1e5, timeout_s=1e6)


def _scan_and_wait(reloc, tag, now):
    """Submit one scan and wait (real time) for the worker to finish it."""
    done = reloc.attempts + reloc.failed_attempts
    reloc.note_frame(_gray(tag), int(now * 10), None, now)
    deadline = time.monotonic() + 5.0
    while reloc.attempts + reloc.failed_attempts == done:
        assert time.monotonic() < deadline, "the worker never ran the scan"
        time.sleep(0.005)


def test_consecutive_worker_failures_stop_the_relocalizer_and_one_success_resets_them():
    assert R.MAX_WORKER_FAILURES == R.MAX_ANCHOR_TRIES == 3
    reloc = R.LookBackRelocalizer(camera_matrix=np.eye(3), frame_size=(WIDTH, HEIGHT),
                                  prompts_enabled=True, verifier=_FailOnOddFrames(),
                                  limiter=_NEVER_TIMES_OUT)
    for i in range(3):
        reloc.note_keyframe(f"k{i}", i, _gray(0))
    reloc.note_lost(0.0)
    now = 0.0
    for tag in (1, 1, 0, 1, 1, 0, 1, 1, 0):  # never three failures in a row
        now += 1.0
        _scan_and_wait(reloc, tag, now)
    assert reloc.failed_attempts == 6 and reloc.attempts == 3
    assert reloc.note_frame(_gray(0), 999, None, now + 0.1) == []  # still running
    for _ in range(3):
        now += 1.0
        _scan_and_wait(reloc, 1, now)
    reloc._thread.join(timeout=2.0)
    assert not reloc._thread.is_alive()  # the worker is gone, not spinning
    with pytest.raises(R.RelocalizerWorkerFailed):
        reloc.note_frame(_gray(0), 1000, None, now + 1.0)
    with pytest.raises(R.RelocalizerWorkerFailed):
        reloc.tick(now + 1.5)
    events = reloc.close(now + 2.0, why="error")
    assert _kinds(events) == ["recovery_timed_out", "relocalizer_stopped"]


def test_the_engine_journals_a_worker_that_keeps_failing(tmp_path, room_frames, monkeypatch):
    """The reviewer's probe C through a real engine and the real worker
    thread: before, the journal showed nothing until the stop; now the drop
    is journaled like any other relocalizer failure (review V5 M4-1)."""
    monkeypatch.setattr(R, "from_session", functools.partial(R.from_session, verifier=_Boom()))

    class _Paced(WorldBuilderEngine):
        def observe(self, raw, **kw):
            result = super().observe(raw, **kw)
            time.sleep(0.02)  # the worker thread gets the CPU between frames
            return result

    _, _, _, events, _ = _walk(tmp_path, _lost_for_good(room_frames), relocalizer="prompt",
                               engine_cls=_Paced)
    kinds = [e["kind"] for e in events]
    stopped = [e for e in events if e["kind"] == "relocalizer_stopped"]
    assert [e["payload"] for e in stopped] == [{"why": "error", "error": "RelocalizerWorkerFailed"}]
    timed = [e["payload"] for e in events if e["kind"] == "recovery_timed_out"]
    assert timed == [{"episode": 1, "why": "relocalizer_stopped"}]
    # Dropped mid-episode, long before its own timeout -- not at the stop,
    # ~25 s after the loss -- and nothing relocalizer-shaped after it.
    lost = next(e for e in events if e["kind"] == "tracking_lost")
    assert stopped[0]["at"] - lost["at"] < R.LimiterParams().timeout_s
    after = kinds[kinds.index("relocalizer_stopped") + 1:]
    assert not [k for k in after if k.startswith(("recovery_", "relocalizer_"))]
    assert after[-1] == "session_stopped"


# -- LOW-3: the cap holds exactly on the clock the wire carries --------------------------------


def _one_prompt(m, wall, *, lost, wall_at):
    """Open an episode at duration-clock `lost` and tick it at `lost + 5`,
    the Tower clock reading `wall_at`. Journals a prompt like the engine."""
    assert m.lost(lost)[0]
    wall.t = wall_at
    events = m.tick(lost + 5.0)
    for kind, _ in events:
        if kind == "recovery_prompted":
            m.prompt_journaled(wall_at)  # the journal line's own `at`
    m.close(lost + 6.0)
    return events


@pytest.mark.parametrize("wall_gap, expected", [
    (30.0, [("recovery_withheld", {"episode": 2, "why": "limiter", "layer": "cooldown"})]),
    (30.0 + TICK, [("recovery_prompted", {"prompt_id": 2, "episode": 2})]),
])
def test_the_cooldown_holds_exactly_on_the_tower_clock(wall_gap, expected):
    """Probe B's schedule: on the duration clock the second prompt comes 30 s
    plus one tick after the first, which the inclusive cooldown lets
    through. If the Tower clock -- the wire's `issued_at` -- reads exactly
    30 s, the wire would show a prompt the design refuses; it is withheld."""
    wall = _Clock(0.0)
    m = R.RecoveryStateMachine(LIM, prompts_enabled=True, wall_clock=wall)
    assert _kinds(_one_prompt(m, wall, lost=0.0, wall_at=1005.0)) == ["recovery_prompted"]
    assert _one_prompt(m, wall, lost=30.0 + TICK, wall_at=1005.0 + wall_gap) == expected


@pytest.mark.parametrize("third_wall, expected", [(1065.0, "cap"), (1065.0 + TICK, None)])
def test_the_cap_holds_exactly_on_the_tower_clock(third_wall, expected):
    """The cap alone (no mechanism): duration-clock issue times 5, 35+q,
    65+2q span more than 60 s, so that clock allows the third. On the Tower
    clock they are 1005, 1035 and `third_wall`; a closed 60 s window holding
    three is refused, one tick later it is not."""
    wall = _Clock(0.0)
    m = R.RecoveryStateMachine(R.LimiterParams(mechanism="none", cooldown_s=None),
                               prompts_enabled=True, wall_clock=wall)
    _one_prompt(m, wall, lost=0.0, wall_at=1005.0)
    _one_prompt(m, wall, lost=30.0 + TICK, wall_at=1035.0)
    third = _one_prompt(m, wall, lost=60.0 + 2 * TICK, wall_at=third_wall)
    if expected is None:
        assert _kinds(third) == ["recovery_prompted"]
    else:
        assert third == [("recovery_withheld", {"episode": 3, "why": "limiter", "layer": expected})]
    issued = [1005.0, 1035.0] + ([third_wall] if expected is None else [])
    assert max(sum(1 for u in issued[i:] if u - t <= 60.0) for i, t in enumerate(issued)) <= 2


def test_the_duration_clock_still_governs_and_a_tower_clock_step_never_prompts():
    """Review V5 M4-4 is kept: a forward Tower-clock step cannot release a
    prompt the duration clock's cooldown refuses, and a step back past the
    last prompt voids the Tower-clock history rather than silencing prompts."""
    wall = _Clock(0.0)
    m = R.RecoveryStateMachine(LIM, prompts_enabled=True, wall_clock=wall)
    _one_prompt(m, wall, lost=0.0, wall_at=1005.0)
    # 10 s later on the duration clock, an hour later on the Tower clock.
    assert _one_prompt(m, wall, lost=10.0, wall_at=1005.0 + 3600.0) == [
        ("recovery_withheld", {"episode": 2, "why": "limiter", "layer": "cooldown"})]
    # 31 s after the prompt on the duration clock; the Tower clock stepped back an hour.
    assert _kinds(_one_prompt(m, wall, lost=31.0, wall_at=1005.0 - 3600.0 + 31.0)) == [
        "recovery_prompted"]


def test_without_a_tower_clock_the_limiter_is_the_duration_clocks_alone():
    """A replay or a caller without an engine passes no Tower clock: exactly
    the old behaviour, probe B's three prompts included."""
    m = R.RecoveryStateMachine(LIM, prompts_enabled=True)
    issued = []
    for t0 in (0.0, 30.0 + TICK, 60.0 + 2 * TICK):
        m.lost(t0)
        issued += [t0 + 5.0 for k, _ in m.tick(t0 + 5.0) if k == "recovery_prompted"]
        m.close(t0 + 6.0)
    assert issued == [5.0, 35.0 + TICK, 65.0 + 2 * TICK]


def test_the_engine_hands_the_limiter_the_journals_own_times(tmp_path, room_frames, synchronous,
                                                            monkeypatch):
    """The wiring: the relocalizer reads the engine's Tower clock, and every
    `recovery_prompted` line's `at` -- the wire's `issued_at` -- is fed back
    to it, exactly."""
    fed = []
    real = R.LookBackRelocalizer.prompt_journaled

    def spy(self, at):
        fed.append(at)
        return real(self, at)

    monkeypatch.setattr(R.LookBackRelocalizer, "prompt_journaled", spy)
    _, _, _, events, _ = _walk(tmp_path, _lost_for_good(room_frames), relocalizer="prompt")
    prompted = [e["at"] for e in events if e["kind"] == "recovery_prompted"]
    assert prompted and fed == prompted

    clock = _Clock()
    engine = WorldBuilderEngine(WorldStore(tmp_path / "w"), clock=clock, relocalizer="prompt",
                                monotonic=_Clock(50.0))
    engine.start_session(engine.create_world("w"), intrinsics=_intrinsics(),
                         frame_source="synthetic", declared_size=(WIDTH, HEIGHT))
    assert engine._reloc.machine._wall_clock is clock
    engine.stop_session()


# -- relocalizer off: none of this runs --------------------------------------------------------


class _CountingClock(_Clock):
    reads = 0

    def __call__(self):
        self.reads += 1
        return self.t


@pytest.mark.parametrize("mode", [None, "off"])
def test_with_the_relocalizer_off_no_frame_path_touches_it(tmp_path, room_frames, monkeypatch, mode):
    """Off (the default) runs no relocalizer code on ANY frame path -- a
    loss, a rejected frame of either kind, a keyframe -- not merely writes
    nothing: its factory is never called, and its duration clock is read
    only where it always was (once per `tracking_lost`, before the
    relocalizer hook), never on the rejected paths this review added."""
    monkeypatch.delenv("TOWER_WORLD_RELOCALIZER", raising=False)

    def forbidden(*a, **k):
        raise AssertionError("from_session called with the relocalizer off")

    monkeypatch.setattr(R, "from_session", forbidden)
    room, noise = room_frames
    clock = _Clock()
    mono = _CountingClock(50.0)
    engine = WorldBuilderEngine(WorldStore(tmp_path), clock=clock, relocalizer=mode,
                                monotonic=mono)
    world_id = engine.create_world("off")
    session_id = engine.start_session(world_id, intrinsics=_intrinsics(), frame_source="synthetic",
                                      declared_size=(WIDTH, HEIGHT))
    frames = (room + noise[:2] + [_rejected_frame("frame_size_changed"),
                                  _rejected_frame("malformed_frame")] + room[::-1][:4])
    for seq, frame in enumerate(frames):
        clock.t += 0.3
        engine.observe(frame, source_seq=seq)
    engine.stop_session()
    kinds = [json.loads(line)["kind"] for line in
             WorldStore(tmp_path).events_path(world_id, session_id).read_text().splitlines()]
    assert "tracking_lost" in kinds and kinds.count("frame_rejected") == 2
    assert not [k for k in kinds if k.startswith(("recovery_", "relocalizer_"))]
    assert mono.reads == kinds.count("tracking_lost")


# -- LOW-4: the docstring names the consumer that exists --------------------------------------


def test_the_module_docstring_says_where_revisit_pairs_go():
    from tower.world_builder import global_solve

    assert "Not wired into global_solve" not in R.__doc__
    assert "_match_revisit_pairs" in R.__doc__
    assert "revisit_pairs(" in inspect.getsource(global_solve._match_revisit_pairs)
