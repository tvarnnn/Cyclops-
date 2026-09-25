"""The live look-back relocalizer, and the look-back prompt's rate limiter.

`docs/contracts/WORLD-BUILDER-COMPONENTS.md` section 6 is the specification;
this module is its producer. In one paragraph:

On a `tracking_lost` with no episode open, the builder opens a RECOVERY
EPISODE and matches incoming frames (at `scan_hz`) against the last
`reference_keyframes` keyframes accepted before the loss. `searching` ->
`recovered` if a match is accepted before `prompt_after_s`, silently.
Otherwise, at `prompt_after_s`, -> `prompting` if prompts are enabled and the
limiter allows; if not, the episode stays `searching` and the withheld prompt
is counted. `searching` or `prompting` -> `recovered` on an acceptance before
`timeout_s` from the loss, else -> `timed_out`. A `tracking_lost` inside an
open episode JOINS it: no new episode, no second prompt, the reference set
unchanged.

WHY IT EXISTS. Walks almost never look back unprompted: over 7 replayed
worlds only 14 of 107 unbridged tracking losses relocalized within 10 s
(RUN/experiments/P2-LOOKBACK). Views that ARE seen again relocalize at once.

WHAT "RECOVERED" MEANS (contract 6.3), and nothing weaker:

  triangle     -- in ONE scanned frame, >= 2 verified links of >= 50 RANSAC
                  inliers to two reference keyframes that are themselves
                  verified-linked, the triangle closing within 8 degrees
                  (P2-LOOKBACK `tri2_50`: wrong 4 of 63, SIFT, masked arm);
  strong-link  -- one verified link of >= 100 inliers (`single_100`: wrong 0
                  of 38 de-duplicated).

A single link of any smaller size was wrong 20 of 99 times and is REFUSED.
The thresholds are SIFT's; P2-LM re-tested them with ALIKED+LightGlue and
kept the rule, but the learned matcher is not a product dependency, so the
product matches with SIFT (RootSIFT 4000, mutual + ratio 0.8, essential
RANSAC 1 px -- the lane harness's base tier, `eval_pairs.PAIR_PARAMS`).

WHAT IT WRITES. Every decision is a journal event (events.py), written by
the ENGINE on the frame thread -- the worker never touches the journal:

  relocalizer_started {acceptance, limiter, prompts_enabled}   session start
  recovery_prompted   {prompt_id, episode}
  recovery_withheld   {episode, why: limiter|disabled|late|no-references, layer}
  recovery_accepted   {episode, by, links, frame, anchor, ...}
  recovery_timed_out  {episode[, why]}
  relocalizer_stopped {why: session_stopped|error[, error]}   terminal
  recovery_anchored   {episode, frame, anchor, links}      after an accept

An accepted relocalization is a VERIFIED REVISIT LINK between keyframes the
final solve can match explicitly: `revisit_pairs(session_dir)` reads them
back as image-name pairs, and `global_solve._match_revisit_pairs` matches
them in the final solve's database. Which solves import them, and at what
COLMAP floor, is that module's decision; what this module guarantees is
that every pair it returns met `REVISIT_MIN_INLIERS` on every live leg --
the reference link AND the link tying the scanned frame to its anchor
keyframe (review V8 M4).

THE FRAME PATH NEVER WAITS. The matcher runs on one daemon thread that is
idle (blocked on a condition, 0 % CPU) unless an episode is open. The frame
thread hands it at most one frame per 1/scan_hz seconds through a one-slot
mailbox and never waits for it: a frame that arrives while the worker is
busy replaces the pending one. Results are collected on the next frame --
or on the next frame the engine REJECTED, which ticks the clock without
handing the worker anything (`tick`, review V8 LOW-1). A worker that fails
`MAX_WORKER_FAILURES` times in a row exits, and the frame thread's next
call raises `RelocalizerWorkerFailed`, which the engine journals as
`relocalizer_stopped {why: error}` (review V8 LOW-2).
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# -- vocabulary (contract 6.2) ----------------------------------------------

STATE_NONE = "none"
STATE_SEARCHING = "searching"
STATE_PROMPTING = "prompting"
STATE_RECOVERED = "recovered"
STATE_TIMED_OUT = "timed_out"
OPEN_STATES = (STATE_SEARCHING, STATE_PROMPTING)

BY_TRIANGLE = "triangle"
BY_STRONG_LINK = "strong-link"

PROMPT_KIND_LOOK_BACK = "look-back"

WITHHELD_LIMITER = "limiter"
WITHHELD_DISABLED = "disabled"
# Review V5: withholds the limiter never sees and `counts` has no key for.
WITHHELD_LATE = "late"
WITHHELD_NO_REFERENCES = "no-references"

MECHANISM_COOLDOWN = "cooldown"

# The setting's vocabulary (config.Settings.world_relocalizer).
MODE_OFF = "off"
MODE_PROMPT = "prompt"
MODE_SILENT = "silent"
MODES = (MODE_OFF, MODE_PROMPT, MODE_SILENT)

EVENT_STARTED = "relocalizer_started"
EVENT_PROMPTED = "recovery_prompted"
EVENT_WITHHELD = "recovery_withheld"
EVENT_ACCEPTED = "recovery_accepted"
EVENT_TIMED_OUT = "recovery_timed_out"
# Not a state change: the keyframe an accepted-but-unanchored link was tied
# to afterwards. Only `revisit_pairs` reads it.
EVENT_ANCHORED = "recovery_anchored"
# Terminal (review V5 M4-1): the relocalizer is gone for the rest of the
# session -- `why: session_stopped` at a normal stop, `error` when it raised.
# An episode still open is closed `timed_out` before it; the reader closes
# one itself if the line never came (a builder that died).
EVENT_STOPPED = "relocalizer_stopped"
# Not a state change (RELOC2, observability, OFF by default): one line per
# episode, written right after the line that resolved it, recording what the
# relocalizer tried -- its references, scans, best legs and closest
# triangle, and the losses it joined. The payload block ignores it; it exists
# so that "why did this look-back not re-link?" is a lookup, not a replay.
EVENT_SUMMARY = "recovery_summary"
RECOVERY_EVENT_KINDS = (
    EVENT_STARTED, EVENT_PROMPTED, EVENT_WITHHELD, EVENT_ACCEPTED, EVENT_TIMED_OUT,
)

# Where `timeout_s` is measured from (RELOC2 candidate 1). `loss` (the
# default, contract 6.1): from the loss that opened the episode, so a prompt
# issued at `prompt_after_s` leaves the wearer `timeout_s - prompt_after_s`.
# `prompt`: a PROMPTED episode's timeout runs from the prompt instead, so the
# wearer gets the whole `timeout_s` after being asked; an unprompted episode
# is unchanged.
WINDOW_FROM_LOSS = "loss"
WINDOW_FROM_PROMPT = "prompt"
WINDOW_FROMS = (WINDOW_FROM_LOSS, WINDOW_FROM_PROMPT)


@dataclass(frozen=True)
class AcceptanceParams:
    """What `recovered` requires. Read-only on the wire (contract 6.3).

    Every number is P2-LOOKBACK's (RUN/experiments/P2-LOOKBACK/out/
    TABLE_sift.md), re-tested by P2-LM (RUN/experiments/P2-LM/lookback):

    * `reference_keyframes` 10 and `scan_hz` 2 -- the K10 / 2 Hz arm the
      prompt replay was run on (`prompts_tri2_50_masked_K10_2hz.json`). It
      is the cheaper of the measured settings: ~42 ms of SIFT per scanned
      frame plus ~6-15 ms per reference pair on one core
      (`out/bench_cpu.json`), i.e. roughly 0.25 of a core while an episode
      is open. K20 / 4 Hz relocalizes more but costs ~4x; see the P3-PR
      replay for the prompt-rate difference.
    * 8 deg -- the lane's control-derived consensus bound (control cycle
      p90 5.9-6.2 deg; lb_common.TAU_ROT_DEG).
    * 50 / 100 inliers -- `tri2_50` and `single_100`.
    """

    matcher: str = "sift"
    reference_keyframes: int = 10
    scan_hz: float = 2.0
    triangle_min_links: int = 2
    triangle_min_link_inliers: int = 50
    triangle_max_closure_deg: float = 8.0
    strong_link_min_inliers: int = 100
    # HISTORICAL REFERENCES (RELOC2 candidate 2, MISSION Objective 4's
    # "historical-keyframe matching"; 0 = off, today's behaviour). Up to this
    # many keyframes OLDER than the last `reference_keyframes`, kept spread
    # over the session by keyframe order (see `_History`), in runs of
    # `history_group` consecutive keyframes so a revisited old view can
    # still close a triangle (two references that are themselves linked).
    # They are added to an episode's references; the acceptance rule is
    # untouched.
    history_keyframes: int = 0
    history_group: int = 2

    def to_wire(self) -> dict:
        wire = {
            "matcher": self.matcher,
            "reference_keyframes": self.reference_keyframes,
            "scan_hz": self.scan_hz,
            "triangle": {
                "min_links": self.triangle_min_links,
                "min_link_inliers": self.triangle_min_link_inliers,
                "max_closure_deg": self.triangle_max_closure_deg,
            },
            "strong_link": {"min_inliers": self.strong_link_min_inliers},
        }
        if self.history_keyframes:
            # Only when on: an off relocalizer's journal is byte-identical.
            wire["history"] = {
                "keyframes": self.history_keyframes, "group": self.history_group,
            }
        return wire


# THE REVISIT FLOOR (review V8 M4): the fewest RANSAC inliers any live leg
# of a revisit pair may have -- the reference->frame link, and the
# frame->anchor link unless the scanned frame IS the anchor keyframe. It is
# the live path's own floor, not a new number: the per-leg 50 of `tri2_50`
# (contract 6.3, P2-LOOKBACK), the smallest link that can make an
# acceptance at all (a strong link needs 100). `verify()` on its own
# accepts 30 (VerifyParams.min_inliers), and a single link under 100 was
# wrong 20 of 99 times (contract 6.3) -- which is why the anchor leg, one
# single link, may not sit below the legs it extends. The final solve's
# import floor in `global_solve` should cite this constant, not copy it.
REVISIT_MIN_INLIERS = AcceptanceParams().triangle_min_link_inliers


class RelocalizerWorkerFailed(RuntimeError):
    """The worker failed `MAX_WORKER_FAILURES` scans in a row and exited.

    Raised on the FRAME thread, by the relocalizer's next call, so the
    engine drops it and journals `relocalizer_stopped {why: error, error:
    "RelocalizerWorkerFailed"}` exactly as for any other failure (review V5
    M4-1). The worker's own exceptions are in the log, one per attempt.
    """


@dataclass(frozen=True)
class LimiterParams:
    """The prompt-rate limiter (contract 6.4). Read-only on the wire.

    Two layers:

    1. THE CAP, invariant: never a prompt when `max_prompts` were issued at
       or after `now - window_s`. Any closed 60 s window holds at most 2.
    2. THE MECHANISM, a COOLDOWN: no prompt within `cooldown_s` of the
       previous one. Chosen over hysteresis and motion suppression because
       it is the only one of the three whose bound is a property of the
       parameter rather than of the walk: with `cooldown_s >= window_s /
       max_prompts` it alone already implies the cap, so the cap never has
       to bind and a withheld prompt is always the cooldown's, visibly.
       Hysteresis (re-arm after N s of good tracking) and motion
       suppression both need a threshold on the tracker that nothing has
       measured, and both let a storm of short losses through between
       their re-arm points. The replay that picked the values is
       RUN/experiments/P3-PR (prompts per minute per walk, and how many
       unbridged losses still get a prompt).

    `prompt_after_s` and `timeout_s` come from the same replay, inside the
    range P2-LOOKBACK tried (1-5 s, 10-20 s). `speak_window_s` is the
    phone's staleness bound (contract 6.5 rule 4) at the Mac review's
    ceiling (mac-001 E7 / M4: <= 5 s bounds both a stale "look back" and the
    one repeat after a relaunch). No walk measured how late a spoken prompt
    stops being useful, so it sits AT the ceiling rather than below it.
    """

    max_prompts: int = 2
    window_s: float = 60.0
    mechanism: str = MECHANISM_COOLDOWN
    cooldown_s: float | None = 30.0
    prompt_after_s: float = 5.0
    timeout_s: float = 20.0
    speak_window_s: float = 5.0
    # RELOC2 candidate 1: `loss` (default) or `prompt`; see WINDOW_FROM_*.
    window_from: str = WINDOW_FROM_LOSS

    def to_wire(self) -> dict:
        wire = {
            "max_prompts": self.max_prompts,
            "window_s": self.window_s,
            "mechanism": self.mechanism,
            "cooldown_s": self.cooldown_s,
            "prompt_after_s": self.prompt_after_s,
            "timeout_s": self.timeout_s,
            "speak_window_s": self.speak_window_s,
        }
        if self.window_from != WINDOW_FROM_LOSS:
            # Only when changed: the default journal is byte-identical, and
            # the payload block's `limiter` keeps exactly the contract keys
            # (`_LIMITER_KEYS` is the default's).
            wire["window_from"] = self.window_from
        return wire


# -- the limiter -------------------------------------------------------------


class _IssueHistory:
    """The issue times of the prompts on ONE clock, and the two checks on it."""

    def __init__(self) -> None:
        self.issued: deque[float] = deque()
        self.last: float | None = None

    def clear(self) -> None:
        self.issued.clear()
        self.last = None

    def cooldown_refuses(self, p: LimiterParams, now: float) -> bool:
        # INCLUSIVE: a prompt exactly `cooldown_s` after the last is still
        # refused. With cooldown_s = window_s / max_prompts that is what makes
        # the cooldown alone imply the CLOSED-window cap: three prompts each
        # more than 30 s apart span more than 60 s.
        return (
            p.mechanism == MECHANISM_COOLDOWN
            and p.cooldown_s is not None
            and self.last is not None
            and now - self.last <= p.cooldown_s
        )

    def cap_refuses(self, p: LimiterParams, now: float) -> bool:
        while self.issued and self.issued[0] < now - p.window_s:
            self.issued.popleft()
        return len(self.issued) >= p.max_prompts

    def record(self, now: float) -> None:
        self.issued.append(now)
        self.last = now


class PromptLimiter:
    """The cap, then the mechanism. Pure; the clocks are the caller's.

    EACH CHECK ON ONE CLOCK (review V8 LOW-3). The bound is decided on the
    builder's DURATION clock (review V5 M4-4: what the wearer hears, immune
    to a wall-clock step) but SHOWN on the Tower clock: the wire's
    `prompt.issued_at` is the journal line's `at`. The two are separate
    clocks, each quantised to 15.6 ms on this Windows machine
    (GetTickCount64 and GetSystemTimeAsFileTime), so a duration-clock gap of
    30 s plus one tick can be exactly 30 s on the wire -- the bound held
    there with a margin of one tick, not by construction. So the limiter
    keeps two histories and asks each check of each clock with that clock's
    OWN readings, never a difference across the two:
      * the duration clock: the `now` of every issue;
      * the Tower clock: the `at` the journal actually wrote for each
        prompt (`record_journaled`, fed back by the engine), against a
        Tower-clock reading taken BEFORE the new line is written -- so the
        new line's `at` is no earlier than the reading, and every bound
        holds on the wire's own numbers exactly.
    The Tower clock can only WITHHOLD, never issue: a prompt goes out only
    when both clocks allow it, so a wall-clock step can never cause one
    (M4-4 kept). A Tower clock read BEHIND the last journaled prompt has
    stepped back; its history is then void (the wire's order is already
    broken by the step) and the duration clock alone decides.
    Without a Tower clock (a replay, a caller with no journal) the limiter is
    the duration clock's alone, exactly as before.
    """

    def __init__(self, params: LimiterParams) -> None:
        self._params = params
        self._duration = _IssueHistory()
        self._tower = _IssueHistory()

    def refusal(self, now: float, tower_now: float | None = None) -> str | None:
        """None if a prompt may be issued at `now` (duration clock) and
        `tower_now` (Tower clock, optional), else the layer refusing.

        The mechanism is asked first, on both clocks, so `"cap"` means
        exactly "the cap refused a prompt the mechanism would have let
        through" -- the event the design says should be rare, counted on
        its own.
        """
        p = self._params
        tower = self._tower if tower_now is not None else None
        if tower is not None and tower.last is not None and tower_now < tower.last:
            tower.clear()  # the Tower clock stepped back past the last prompt
        if self._duration.cooldown_refuses(p, now) or (
            tower is not None and tower.cooldown_refuses(p, tower_now)
        ):
            return "cooldown"
        if self._duration.cap_refuses(p, now) or (
            tower is not None and tower.cap_refuses(p, tower_now)
        ):
            return "cap"
        return None

    def record(self, now: float) -> None:
        """A prompt was issued at duration-clock `now`."""
        self._duration.record(now)

    def record_journaled(self, at: float) -> None:
        """The journal wrote that prompt with Tower-clock `at`."""
        self._tower.record(at)


# -- the state machine ---------------------------------------------------------


def _zero_counts() -> dict:
    return {
        "episodes": 0,
        "recovered": 0,
        "recovered_after_prompt": 0,
        "timed_out": 0,
        "prompts": 0,
        "withheld_by_limiter": 0,
        "withheld_disabled": 0,
    }


class RecoveryStateMachine:
    """Contract 6.1, clock-driven and side-effect free.

    Every method returns the journal events the transition implies as
    `(kind, payload)` tuples, in order. The ENGINE writes them. Keeping the
    machine pure is what lets the same code be replayed on the frozen walks
    (RUN/experiments/P3-PR/scripts/replay_prompts.py) and unit-tested with
    a fake clock.

    THE CLOCK IS MONOTONIC (review V5 M4-4). Every `at`/`now` handed to this
    machine is a duration clock (`time.monotonic` in the builder); the
    Tower-clock times on the wire come from the journal lines themselves,
    never from here, so a wall-clock step cannot open, prompt or time out an
    episode. The one use of the Tower clock (`wall_clock`, review V8 LOW-3)
    is the limiter's second history, which can only WITHHOLD a prompt that
    would break the cap as the wire shows it; see `PromptLimiter`.

    ORDER OF A TICK (review V5 M4-2). The timeout is evaluated FIRST, and a
    prompt is issued only while `now - lost_at <= prompt_after_s +
    late_grace_s`, `late_grace_s` being one scan period. A frame that
    arrives later than that -- a stalled stream, a reconnect -- withholds
    the prompt as `late`, which the limiter never sees: a late "look back"
    is wrong advice, and it must not spend the cooldown of the next,
    timely one. An episode whose first frame after a stall is already past
    `timeout_s` simply times out, unprompted.

    AN EPISODE THAT CANNOT BE ACCEPTED (review V5 M4-3) -- no reference
    keyframes before the loss -- withholds its prompt as `no-references`:
    asking the wearer to look back at nothing is a prompt with no possible
    success.
    """

    def __init__(
        self,
        limiter: LimiterParams,
        *,
        prompts_enabled: bool,
        late_grace_s: float | None = None,
        wall_clock=None,
    ) -> None:
        self.limiter_params = limiter
        self.prompts_enabled = bool(prompts_enabled)
        self.late_grace_s = (
            1.0 / AcceptanceParams().scan_hz if late_grace_s is None else float(late_grace_s)
        )
        self._limiter = PromptLimiter(limiter)
        # The clock the journal stamps `at` with (the engine's Tower clock),
        # read only when the limiter decides. None: the duration clock alone.
        self._wall_clock = wall_clock
        self.state = STATE_NONE
        self.episode = 0
        self.lost_at: float | None = None
        self.resolved_at: float | None = None
        self.recovered_by: str | None = None
        self.prompt_id = 0
        self.prompt_episode: int | None = None
        self.counts = _zero_counts()
        # Withholds the contract's `counts` has no key for; journaled with
        # their `why`, counted here for tests and diagnostics only.
        self.withheld_late = 0
        self.withheld_no_references = 0
        self._prompt_decided = False
        self._can_accept = True
        # The open (or last) episode's prompt time on this machine's clock,
        # None when it was not prompted; and the losses that joined it.
        self.prompted_at: float | None = None
        self.losses_joined = 0

    @property
    def is_open(self) -> bool:
        return self.state in OPEN_STATES

    def lost(self, at: float, *, can_accept: bool = True) -> tuple[bool, list]:
        """A journaled `tracking_lost` at `at`. Returns (opened, events)."""
        if self.is_open:
            self.losses_joined += 1
            return False, []  # joins the open episode
        self.episode += 1
        self.counts["episodes"] += 1
        self.state = STATE_SEARCHING
        self.lost_at = at
        self.resolved_at = None
        self.recovered_by = None
        self._prompt_decided = False
        self._can_accept = bool(can_accept)
        self.prompted_at = None
        self.losses_joined = 0
        return True, []

    def cannot_accept(self) -> None:
        """The open episode can no longer be accepted (e.g. no usable frame)."""
        if self.is_open:
            self._can_accept = False

    def accepted(self, at: float, by: str, detail: dict | None = None) -> list:
        if not self.is_open:
            return []
        if by not in (BY_TRIANGLE, BY_STRONG_LINK):
            raise ValueError(f"unknown acceptance {by!r}")
        after_prompt = self.state == STATE_PROMPTING
        self.state = STATE_RECOVERED
        self.resolved_at = at
        self.recovered_by = by
        self.counts["recovered"] += 1
        if after_prompt:
            self.counts["recovered_after_prompt"] += 1
        return [(EVENT_ACCEPTED, {"episode": self.episode, "by": by, **(detail or {})})]

    def tick(self, now: float) -> list:
        """Time-driven transitions, evaluated as frames arrive (OPEN T8)."""
        if not self.is_open:
            return []
        p = self.limiter_params
        elapsed = now - self.lost_at
        # The window's start: the loss, or -- `window_from: prompt` and the
        # episode was prompted -- the prompt (RELOC2 candidate 1).
        window_start = self.lost_at
        if p.window_from == WINDOW_FROM_PROMPT and self.prompted_at is not None:
            window_start = self.prompted_at
        if now - window_start >= p.timeout_s:
            # FIRST: a frame this late never prompts, it only closes.
            self._prompt_decided = True
            return self._time_out(now, why=None)
        if self._prompt_decided or elapsed < p.prompt_after_s:
            return []
        self._prompt_decided = True
        if not self.prompts_enabled:
            self.counts["withheld_disabled"] += 1
            return [(EVENT_WITHHELD, {
                "episode": self.episode, "why": WITHHELD_DISABLED, "layer": None,
            })]
        if not self._can_accept:
            self.withheld_no_references += 1
            return [(EVENT_WITHHELD, {
                "episode": self.episode, "why": WITHHELD_NO_REFERENCES, "layer": None,
            })]
        if elapsed > p.prompt_after_s + self.late_grace_s:
            self.withheld_late += 1
            return [(EVENT_WITHHELD, {
                "episode": self.episode, "why": WITHHELD_LATE, "layer": None,
            })]
        # Read BEFORE the prompt line is journaled, so that line's `at` can
        # only be the same or later (review V8 LOW-3).
        tower_now = self._wall_clock() if self._wall_clock is not None else None
        layer = self._limiter.refusal(now, tower_now)
        if layer is not None:
            # Never issued later: a late "look back" is wrong advice.
            self.counts["withheld_by_limiter"] += 1
            return [(EVENT_WITHHELD, {
                "episode": self.episode, "why": WITHHELD_LIMITER, "layer": layer,
            })]
        self._limiter.record(now)
        self.prompt_id += 1
        self.prompt_episode = self.episode
        self.counts["prompts"] += 1
        self.state = STATE_PROMPTING
        self.prompted_at = now
        return [(EVENT_PROMPTED, {"prompt_id": self.prompt_id, "episode": self.episode})]

    def prompt_journaled(self, at: float) -> None:
        """The journal wrote the last `recovery_prompted` with Tower-clock
        `at` -- the wire's `issued_at`. The limiter's Tower-clock history is
        these numbers, not a reading of its own (review V8 LOW-3)."""
        self._limiter.record_journaled(float(at))

    def close(self, now: float, *, why: str = "session_stopped") -> list:
        """The session stopped (or the relocalizer did): an open episode can
        no longer recover."""
        if not self.is_open:
            return []
        return self._time_out(now, why=why)

    def _time_out(self, now: float, *, why: str | None) -> list:
        self.state = STATE_TIMED_OUT
        self.resolved_at = now
        self.counts["timed_out"] += 1
        payload = {"episode": self.episode}
        if why is not None:
            payload["why"] = why
        return [(EVENT_TIMED_OUT, payload)]


# -- the payload block, from the journal ---------------------------------------


class RecoveryJournalReader:
    """`tracking.recovery` (contract 6.2) from the session's journal.

    Fed one event at a time inside `_summarise_events`' single pass, so the
    cached summary holds this fixed-arity block and never the journal.
    `limiter` and `acceptance` are what the BUILDER journaled at session
    start, never the web process's configuration.
    """

    def __init__(self) -> None:
        self.recorded = False
        self._limiter: dict | None = None
        self._acceptance: dict | None = None
        self._prompts_enabled = False
        self._state = STATE_NONE
        self._episode = 0
        self._lost_at = None
        self._resolved_at = None
        self._recovered_by = None
        self._prompt: dict | None = None
        self._counts = _zero_counts()
        self._stopped = False

    def feed(self, event: dict) -> None:
        kind = event.get("kind")
        if kind == EVENT_STARTED:
            payload = _payload(event)
            self.recorded = True
            self._limiter = _fixed_limiter(payload.get("limiter"))
            self._acceptance = _fixed_acceptance(payload.get("acceptance"))
            self._prompts_enabled = bool(payload.get("prompts_enabled"))
            return
        if not self.recorded:
            return
        at = _number(event.get("at"))
        if kind in (EVENT_STOPPED, "session_stopped"):
            # Review V5 M4-1: nothing can resolve an open episode after the
            # relocalizer or the session is gone. The builder journals the
            # close itself; this covers a builder that died before it could,
            # so the block never reads "searching" forever.
            if self._state in OPEN_STATES:
                self._state = STATE_TIMED_OUT
                self._resolved_at = at
                self._counts["timed_out"] += 1
            self._stopped = True
            return
        if kind == "tracking_lost":
            if self._stopped or self._state in OPEN_STATES:
                return  # no relocalizer any more, or joins the open episode
            self._episode += 1
            self._counts["episodes"] += 1
            self._state = STATE_SEARCHING
            self._lost_at = at
            self._resolved_at = None
            self._recovered_by = None
            return
        if kind not in RECOVERY_EVENT_KINDS:
            return
        payload = _payload(event)
        if payload.get("episode") != self._episode or self._state not in OPEN_STATES:
            return  # not about the open episode: never let it move the block
        if kind == EVENT_PROMPTED:
            prompt_id = payload.get("prompt_id")
            if not isinstance(prompt_id, int) or (
                self._prompt is not None and prompt_id <= self._prompt["id"]
            ):
                return  # ids only ever rise
            self._state = STATE_PROMPTING
            self._counts["prompts"] += 1
            speak_window = (self._limiter or {}).get("speak_window_s") or 0.0
            self._prompt = {
                "id": prompt_id,
                "episode": self._episode,
                "kind": PROMPT_KIND_LOOK_BACK,
                "issued_at": at,
                "speak_until": None if at is None else at + float(speak_window),
            }
        elif kind == EVENT_WITHHELD:
            why = payload.get("why")
            if why == WITHHELD_DISABLED:
                self._counts["withheld_disabled"] += 1
            elif why == WITHHELD_LIMITER:
                self._counts["withheld_by_limiter"] += 1
            # `late` and `no-references` are neither: the limiter never saw
            # them, and the contract's counts carry no key for them.
        elif kind == EVENT_ACCEPTED:
            by = payload.get("by")
            if self._state == STATE_PROMPTING:
                self._counts["recovered_after_prompt"] += 1
            self._state = STATE_RECOVERED
            self._resolved_at = at
            self._recovered_by = by if by in (BY_TRIANGLE, BY_STRONG_LINK) else None
            self._counts["recovered"] += 1
        elif kind == EVENT_TIMED_OUT:
            self._state = STATE_TIMED_OUT
            self._resolved_at = at
            self._counts["timed_out"] += 1

    def block(self) -> dict | None:
        """The fixed-arity object, or None when no relocalizer was recorded."""
        if not self.recorded:
            return None
        return {
            "state": self._state,
            "episode": self._episode,
            "lost_at": self._lost_at if self._episode else None,
            "resolved_at": self._resolved_at,
            "recovered_by": self._recovered_by if self._state == STATE_RECOVERED else None,
            "prompts_enabled": self._prompts_enabled,
            "prompt": dict(self._prompt) if self._prompt is not None else None,
            "counts": dict(self._counts),
            "limiter": dict(self._limiter),
            "acceptance": {
                **self._acceptance,
                "triangle": dict(self._acceptance["triangle"]),
                "strong_link": dict(self._acceptance["strong_link"]),
            },
        }


def recovery_block(events) -> dict | None:
    """`tracking.recovery` for a whole journal (tests and offline readers)."""
    reader = RecoveryJournalReader()
    for event in events:
        reader.feed(event)
    return reader.block()


def _payload(event: dict) -> dict:
    payload = event.get("payload")
    return payload if isinstance(payload, dict) else {}


def _number(value):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


_LIMITER_KEYS = tuple(LimiterParams().to_wire())
_ACCEPTANCE_KEYS = ("matcher", "reference_keyframes", "scan_hz")


def _fixed_limiter(raw) -> dict:
    """Exactly the contract's keys, whatever the journal line carried."""
    raw = raw if isinstance(raw, dict) else {}
    return {key: raw.get(key) for key in _LIMITER_KEYS}


def _fixed_acceptance(raw) -> dict:
    raw = raw if isinstance(raw, dict) else {}
    tri = raw.get("triangle") if isinstance(raw.get("triangle"), dict) else {}
    strong = raw.get("strong_link") if isinstance(raw.get("strong_link"), dict) else {}
    return {
        **{key: raw.get(key) for key in _ACCEPTANCE_KEYS},
        "triangle": {
            "min_links": tri.get("min_links"),
            "min_link_inliers": tri.get("min_link_inliers"),
            "max_closure_deg": tri.get("max_closure_deg"),
        },
        "strong_link": {"min_inliers": strong.get("min_inliers")},
    }


# -- revisit links, for the final solve ------------------------------------------


def revisit_pairs(session_dir) -> list[tuple[str, str]]:
    """Image-name pairs the final solve should match explicitly.

    One pair per (reference keyframe, anchor keyframe) of every journaled
    `recovery_accepted`, where the anchor is the keyframe the accepted scan
    frame is tied to: the frame itself when it was a keyframe, the nearest
    post-loss keyframe it verified against, or -- `recovery_anchored`, for
    the same episode -- a keyframe accepted just after it. Names are
    `Path(image_relpath).name`, i.e. `global_solve.keyframe_image_name`,
    in journal order, each pair once, reference first.

    THE FLOOR (review V8 M4). A pair is returned only when every live leg
    met `REVISIT_MIN_INLIERS` (inclusive) as journaled: the reference
    link's `inliers` in `recovery_accepted.links[]`, and the anchor's
    `inliers` (`recovery_accepted.anchor` or `recovery_anchored.anchor`)
    unless the anchor is the scanned frame itself (`identity: true`). A leg
    with no count is below the floor. The live path no longer anchors below
    it; this read applies it to every journal, older ones included, so a
    caller never needs a marker: below the floor is simply absent.

    An absent journal, an old session, or a session without a relocalizer
    returns `[]` -- never an error (contract 7). An acceptance never
    anchored is not returned: its scanned frame is not on disk, so the
    solve has nothing to match.
    """
    session_dir = Path(session_dir)
    events_path = session_dir / "events.jsonl"
    keyframes_path = session_dir / "keyframes.jsonl"
    if not events_path.is_file() or not keyframes_path.is_file():
        return []
    names: dict[str, str] = {}
    for record in _read_jsonl(keyframes_path):
        kid, rel = record.get("keyframe_id"), record.get("image_relpath")
        if isinstance(kid, str) and isinstance(rel, str):
            names[kid] = Path(rel).name
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    accepted_refs: dict = {}  # episode -> reference ids of its accepted link, at the floor

    def at_floor(inliers) -> bool:
        return (
            isinstance(inliers, (int, float)) and not isinstance(inliers, bool)
            and inliers >= REVISIT_MIN_INLIERS
        )

    def add(refs, anchor):
        anchor_name = names.get(anchor.get("keyframe_id")) if isinstance(anchor, dict) else None
        if anchor_name is None:
            return
        if anchor.get("identity") is not True and not at_floor(anchor.get("inliers")):
            return  # the scanned frame -> anchor leg is below the floor
        for ref in refs:
            ref_name = names.get(ref)
            if ref_name is None or ref_name == anchor_name:
                continue
            pair = (ref_name, anchor_name)
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)

    for event in _read_jsonl(events_path):
        kind = event.get("kind")
        if kind not in (EVENT_ACCEPTED, EVENT_ANCHORED):
            continue
        payload = _payload(event)
        if kind == EVENT_ACCEPTED:
            # Only the accepted event carries each reference link's count;
            # `recovery_anchored` re-lists the ids, never the evidence.
            refs = [
                link.get("ref_keyframe_id")
                for link in payload.get("links") or ()
                if isinstance(link, dict) and at_floor(link.get("inliers"))
            ]
            accepted_refs[payload.get("episode")] = refs
            add(refs, payload.get("anchor"))
        elif payload.get("episode") in accepted_refs:
            # Only an episode whose acceptance was journaled.
            add(accepted_refs[payload.get("episode")], payload.get("anchor"))
    return pairs


def _read_jsonl(path: Path):
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue  # a torn tail line is routine (append without fsync)
        if isinstance(record, dict):
            yield record


# -- two-view verification (SIFT) ------------------------------------------------


@dataclass(frozen=True)
class VerifyParams:
    """The lane harness's base tier (`eval_pairs.PAIR_PARAMS`), unchanged."""

    sift_features: int = 4000
    ratio: float = 0.8
    ransac_px: float = 1.0
    ransac_conf: float = 0.9999
    min_inliers: int = 30
    min_inlier_ratio: float = 0.25
    min_bbox_fraction: float = 0.05
    min_parallax_deg: float = 1.0


@dataclass
class Features:
    xy: object  # (N, 2) float64 pixel coordinates, undistorted
    desc: object  # (N, 128) float32 RootSIFT


@dataclass
class Link:
    """A verified two-view relation. R maps camera A to camera B."""

    n_inliers: int
    n_matches: int
    R: object
    t: object
    model: str


class SiftVerifier:
    """RootSIFT + mutual ratio matching + essential RANSAC (`eval_pairs`).

    Keypoints are undistorted with the session's own calibration, so the
    1 px RANSAC threshold means what it meant on the harness's undistorted
    images; the relocalizer refuses to run without a calibration (the
    inlier thresholds were measured under one).
    """

    def __init__(self, camera_matrix, dist_coeffs=None, params: VerifyParams | None = None):
        import numpy as np

        self.K = np.asarray(camera_matrix, dtype=np.float64)
        self.dist = None if not dist_coeffs else np.asarray(dist_coeffs, dtype=np.float64)
        self.p = params or VerifyParams()
        self._sift = None

    def features(self, gray) -> Features:
        import cv2
        import numpy as np

        if self._sift is None:
            self._sift = cv2.SIFT_create(nfeatures=self.p.sift_features)
        kps, desc = self._sift.detectAndCompute(gray, None)
        if desc is None or not len(kps):
            return Features(np.zeros((0, 2)), np.zeros((0, 128), np.float32))
        xy = np.array([k.pt for k in kps], np.float64)
        if self.dist is not None and np.any(self.dist):
            xy = cv2.undistortPoints(
                xy.reshape(-1, 1, 2), self.K, self.dist, P=self.K
            ).reshape(-1, 2)
        desc = desc.astype(np.float32)
        desc /= np.abs(desc).sum(1, keepdims=True) + 1e-9
        return Features(xy, np.sqrt(desc))

    def verify(self, a: Features, b: Features, size) -> Link | None:
        import cv2
        import numpy as np

        p = self.p
        if len(a.desc) < 2 or len(b.desc) < 2:
            return None
        bf = cv2.BFMatcher(cv2.NORM_L2)
        fwd = bf.knnMatch(a.desc, b.desc, k=2)
        bwd = bf.knnMatch(b.desc, a.desc, k=1)
        back = {m[0].queryIdx: m[0].trainIdx for m in bwd if m}
        pairs = [
            (m[0].queryIdx, m[0].trainIdx)
            for m in fwd
            if len(m) == 2
            and m[0].distance < p.ratio * m[1].distance
            and back.get(m[0].trainIdx) == m[0].queryIdx
        ]
        if len(pairs) < max(p.min_inliers, 5):
            return None
        idx = np.asarray(pairs, np.int64)
        p1, p2 = a.xy[idx[:, 0]], b.xy[idx[:, 1]]
        E, mask = cv2.findEssentialMat(
            p1, p2, self.K, method=cv2.RANSAC, prob=p.ransac_conf, threshold=p.ransac_px
        )
        if E is None or mask is None or E.shape[0] < 3 or E.shape[1] != 3:
            return None
        E = E[:3]
        inl = mask.ravel().astype(bool)
        n_inl = int(inl.sum())
        if n_inl < p.min_inliers or n_inl / max(len(pairs), 1) < p.min_inlier_ratio:
            return None
        w, h = size
        for q in (p1[inl], p2[inl]):
            span = (q[:, 0].max() - q[:, 0].min()) * (q[:, 1].max() - q[:, 1].min())
            if span / float(w * h) < p.min_bbox_fraction:
                return None
        Kinv = np.linalg.inv(self.K)
        n1 = (Kinv @ np.c_[p1[inl], np.ones(n_inl)].T).T
        n2 = (Kinv @ np.c_[p2[inl], np.ones(n_inl)].T).T
        b1 = n1 / np.linalg.norm(n1, axis=1, keepdims=True)
        b2 = n2 / np.linalg.norm(n2, axis=1, keepdims=True)
        out = cv2.recoverPose(E, p1[inl], p2[inl], cameraMatrix=self.K, distanceThresh=1e4)
        n_cheir, R_E, t_E = int(out[0]), out[1], out[2]
        # Rotation-only (Wahba) fit, used when the pair has no parallax.
        U, _, Vt = np.linalg.svd(b1.T @ b2)
        D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
        R_rot = Vt.T @ D @ U.T

        def parallax(R):
            c = np.clip(((b1 @ R.T) * b2).sum(1), -1.0, 1.0)
            return float(np.median(np.degrees(np.arccos(c))))

        if parallax(R_E) >= p.min_parallax_deg and n_cheir >= 0.5 * n_inl:
            R, model = R_E, "essential"
        else:
            R, model = R_rot, "rotation"
        t = t_E.ravel()
        t = t / (np.linalg.norm(t) + 1e-12)
        return Link(n_inliers=n_inl, n_matches=len(pairs), R=R, t=t, model=model)


def rotation_angle_deg(R) -> float:
    import numpy as np

    c = (float(np.trace(np.asarray(R))) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def evaluate_acceptance(links: dict, refref, params: AcceptanceParams):
    """The contract's two rules over one scanned frame's verified links.

    `links` maps a reference keyframe id to its `Link` (reference -> frame).
    `refref(k1, k2)` returns the verified `Link` k1 -> k2 or None; it is only
    asked about pairs whose both legs already clear the triangle floor.
    Returns `(by, used_ids, closure_deg)` or None. The strong link is
    preferred when both hold: it was never wrong on the replay (0 of 38).
    """
    strong = [
        (k, l) for k, l in links.items() if l.n_inliers >= params.strong_link_min_inliers
    ]
    if strong:
        k, _ = max(strong, key=lambda kl: kl[1].n_inliers)
        return BY_STRONG_LINK, [k], None
    legs = [
        (k, l) for k, l in links.items() if l.n_inliers >= params.triangle_min_link_inliers
    ]
    if len(legs) < max(2, params.triangle_min_links):
        return None
    import numpy as np

    best = None
    for i in range(len(legs)):
        for j in range(i + 1, len(legs)):
            (k1, l1), (k2, l2) = legs[i], legs[j]
            o = refref(k1, k2)
            if o is None:
                continue
            # R(r<-k1) against R(r<-k2) R(k2<-k1)
            closure = rotation_angle_deg(np.asarray(l1.R).T @ np.asarray(l2.R) @ np.asarray(o.R))
            if closure <= params.triangle_max_closure_deg and (best is None or closure < best[2]):
                best = (k1, k2, closure)
    if best is None:
        return None
    return BY_TRIANGLE, [best[0], best[1]], best[2]


# -- the worker ------------------------------------------------------------------


@dataclass
class _Episode:
    number: int
    refs: list  # [(keyframe_id, gray)] oldest first
    ref_features: dict = field(default_factory=dict)
    refref: dict = field(default_factory=dict)
    post: list = field(default_factory=list)  # [(keyframe_id, source_seq, gray)]
    done: bool = False
    # What the worker tried (the summary's evidence; written by the worker
    # under the lock, read by the frame thread under it).
    n_history: int = 0
    attempts: int = 0
    best_inliers: dict = field(default_factory=dict)  # ref id -> most inliers seen
    best_pair: dict | None = None  # the scan whose SECOND-best leg was highest
    best_triangle: dict | None = None  # the closest triangle evaluated at the floor
    dropped_at_open: int = 0
    cache_pruned: bool = False  # the worker trimmed the feature cache to these refs


class _History:
    """Keyframes older than the last `reference_keyframes`, spread over the
    session (RELOC2 candidate 2). Pixels in memory only, like `_refs`.

    Keyframes arrive in acceptance order as they fall out of the recent
    window, so consecutive arrivals are consecutive keyframes. They are
    kept in GROUPS of `group` consecutive keyframes (a triangle needs two
    linked references of one view), numbered 0, 1, 2, ... A STRIDE rule
    keeps the spread even: only groups whose number is a multiple of
    `stride` are kept, plus the newest (the most recent old view); when the
    budget is exceeded the stride doubles. So the held groups are always
    evenly spaced over the whole walk in keyframe order -- the first group
    (where the walk began) is never dropped -- at most a factor of 2 apart
    from perfectly even. Keyframe order stands in for time and viewpoint:
    the relocalizer holds no poses, and nothing measured says a cheap image
    descriptor would spread views better.
    """

    def __init__(self, capacity: int, group: int) -> None:
        self.capacity = int(capacity)
        self.group = max(1, int(group))
        self.stride = 1
        self._next = 0
        # [(number, [(keyframe_id, source_seq, gray), ...]), ...] oldest first
        self.groups: list[tuple[int, list]] = []

    def add(self, keyframe_id: str, source_seq: int, gray) -> None:
        member = (keyframe_id, source_seq, gray)
        if self.groups and len(self.groups[-1][1]) < self.group:
            self.groups[-1][1].append(member)
        else:
            # The newest group is complete and settles: kept only on stride.
            if self.groups and self.groups[-1][0] % self.stride:
                self.groups.pop()
            self.groups.append((self._next, [member]))
            self._next += 1
        while sum(len(m) for _, m in self.groups) > self.capacity:
            if len(self.groups) <= 2:
                self.groups.pop(0)  # a budget below two groups: oldest out
                continue
            self.stride *= 2
            newest = self.groups[-1]
            self.groups = [g for g in self.groups[:-1] if g[0] % self.stride == 0] + [newest]

    def members(self) -> list:
        """[(keyframe_id, gray)], oldest first."""
        return [(kid, gray) for _, m in self.groups for kid, _seq, gray in m]


@dataclass
class _Job:
    episode: int
    source_seq: int
    keyframe_id: str | None
    gray: object


@dataclass
class _PendingAnchor:
    """An accepted scan frame no post-loss keyframe has verified against yet."""

    episode: int
    source_seq: int
    frame: Features
    links: dict  # reference keyframe id -> Link (reference -> frame)
    tries: int = 0


# An acceptance whose scanned frame is not a keyframe, and that no keyframe
# accepted since the loss verified against, is tied to the NEXT keyframes as
# they arrive, up to this many. Without an anchor the link names a frame
# that is not on disk, and the final solve has nothing to match. The number
# is OPEN (no walk measured it): three keyframes is about a second of
# walking at the selector's usual cadence, the span over which the view is
# still the one the scanned frame saw. A keyframe whose link to the scanned
# frame is below REVISIT_MIN_INLIERS is a failed try (review V8 M4).
MAX_ANCHOR_TRIES = 3

# A worker whose jobs (scans or anchor checks) raise this many times IN A
# ROW exits, and the relocalizer is stopped (`relocalizer_stopped {why:
# error}`, review V8 LOW-2); one job that completes resets the count.
# OPEN: no walk has ever produced a failing worker, so nothing measured
# this. It is the module's existing retry bound (MAX_ANCHOR_TRIES). At
# `scan_hz` 2 a matcher that fails on every scan is stopped about 1.5 s
# after the loss, normally before `prompt_after_s` (5 s), so it does not
# prompt a look-back it could never recognise.
MAX_WORKER_FAILURES = MAX_ANCHOR_TRIES


class LookBackRelocalizer:
    """The engine's handle: journal-free, clock-free, never blocking.

    The engine calls, from the frame thread:

    * `started_payload()` once, for `relocalizer_started`;
    * `note_keyframe(keyframe_id, source_seq, gray)` on every acceptance;
    * `note_lost(at)` on every journaled `tracking_lost`;
    * `note_frame(gray, source_seq, keyframe_id, now)` on every decoded frame;
    * `tick(now)` on every frame the engine REJECTS before tracking
      (undecodable, or at another size): time only, no matching;
    * `prompt_journaled(at)` with the `at` of every `recovery_prompted`
      line it wrote;
    * `close(now, why=...)` at stop, or when the engine drops it.

    Every time handed in is the builder's MONOTONIC clock (review V5
    M4-4); the Tower-clock times on the wire are the journal lines' own.
    `wall_clock` is the engine's Tower clock, read only by the limiter
    (review V8 LOW-3).

    Each returns the journal events to write, in order. Only while an
    episode is open (or an accepted link still waits for its anchor) does
    any of them hand work to the worker. All but `close` raise
    `RelocalizerWorkerFailed` once the worker has given up.
    """

    def __init__(
        self,
        *,
        camera_matrix,
        dist_coeffs=None,
        frame_size,
        prompts_enabled: bool,
        acceptance: AcceptanceParams | None = None,
        limiter: LimiterParams | None = None,
        verifier=None,
        synchronous: bool = False,
        wall_clock=None,
        summary_events: bool = False,
    ) -> None:
        self.acceptance = acceptance or AcceptanceParams()
        self.limiter = limiter or LimiterParams()
        # RELOC2 candidate 3: `recovery_summary` per resolved episode. Off:
        # not one line is added to the journal.
        self.summary_events = bool(summary_events)
        self._summarised = 0  # the last episode a summary was written for
        self._ref_seq: dict = {}  # recent reference id -> source_seq (history only)
        # keyframe id -> Features, across episodes; history only; the WORKER
        # thread alone reads and writes it (see `_run`).
        self._feature_cache: dict = {}
        # RELOC2 candidate 2: None when off (today's reference set exactly).
        self._history = (
            _History(self.acceptance.history_keyframes, self.acceptance.history_group)
            if self.acceptance.history_keyframes > 0 else None
        )
        # One scan period of lateness is tolerated before a prompt is `late`.
        self.machine = RecoveryStateMachine(
            self.limiter, prompts_enabled=prompts_enabled,
            late_grace_s=1.0 / self.acceptance.scan_hz,
            wall_clock=wall_clock,
        )
        self._verifier = verifier or SiftVerifier(camera_matrix, dist_coeffs)
        self._size = tuple(frame_size)  # (width, height)
        self._refs: deque = deque(maxlen=self.acceptance.reference_keyframes)
        self._episode: _Episode | None = None
        self._last_submit: float | None = None
        self._synchronous = synchronous
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._pending: _Job | None = None
        self._anchor_jobs: deque = deque(maxlen=MAX_ANCHOR_TRIES)
        self._pending_anchor: _PendingAnchor | None = None
        self._results: deque = deque()
        self._stop = False
        self._closed = False
        self._thread: threading.Thread | None = None
        # A frame at a size the calibration does not describe was seen: no
        # scan can run, so no episode can be accepted (review V5 M4-3).
        self._size_mismatch = False
        self._said_no_references = False
        # Measured, per scan attempt, on the worker thread.
        self.attempts = 0
        self.attempt_wall_s: list[float] = []
        self.attempt_cpu_s: list[float] = []
        self.dropped_frames = 0
        # Worker jobs that raised: in total, and in a row (review V8 LOW-2).
        self.failed_attempts = 0
        self._consecutive_failures = 0
        self._worker_error: BaseException | None = None

    # -- engine API ---------------------------------------------------------

    def started_payload(self) -> dict:
        return {
            "acceptance": self.acceptance.to_wire(),
            "limiter": self.limiter.to_wire(),
            "prompts_enabled": self.machine.prompts_enabled,
        }

    def _fits(self, gray) -> bool:
        return tuple(gray.shape[:2]) == (self._size[1], self._size[0])

    def note_keyframe(self, keyframe_id: str, source_seq: int, gray) -> None:
        self._raise_if_worker_failed()
        if not self._fits(gray):
            self._size_mismatch = True
            return  # a frame this calibration does not describe
        ep = self._episode
        with self._lock:
            if ep is not None and not ep.done:
                ep.post.append((keyframe_id, source_seq, gray))
                del ep.post[:-3]  # the anchor candidates: the nearest three
            wants_anchor = self._pending_anchor is not None
        if wants_anchor:
            self._submit_anchor((keyframe_id, source_seq, gray))
        if self._history is not None:
            if len(self._refs) == self._refs.maxlen:
                old_id, old_gray = self._refs[0]  # about to fall out of the window
                self._history.add(old_id, self._ref_seq.pop(old_id, -1), old_gray)
            self._ref_seq[keyframe_id] = source_seq
        self._refs.append((keyframe_id, gray))

    def note_lost(self, at: float) -> list:
        """`at` is the builder's MONOTONIC time of the journaled loss."""
        self._raise_if_worker_failed()
        can_accept = bool(self._refs) and not self._size_mismatch
        opened, events = self.machine.lost(at, can_accept=can_accept)
        if opened:
            refs = list(self._refs)
            history = self._history.members() if self._history is not None and refs else []
            with self._lock:
                if self._episode is not None:
                    self._episode.done = True
                # Recent references first, then the historical ones, oldest
                # first; with history off this is exactly the recent window.
                self._episode = _Episode(
                    number=self.machine.episode, refs=refs + history,
                    n_history=len(history), dropped_at_open=self.dropped_frames,
                )
                self._pending = None
            self._last_submit = None
            if not can_accept:
                self._say_no_references()
        return events

    def note_frame(self, gray, source_seq: int, keyframe_id: str | None, now: float) -> list:
        """`now` is the builder's MONOTONIC time of this frame."""
        self._raise_if_worker_failed()
        events = self._drain(now)
        if self.machine.is_open:
            if not self._fits(gray):
                if not self._size_mismatch:
                    self._size_mismatch = True
                self.machine.cannot_accept()
                self._say_no_references()
            elif self._episode is not None and self._episode.refs:
                due = self._last_submit is None or now - self._last_submit >= 1.0 / self.acceptance.scan_hz
                if due:
                    self._last_submit = now
                    self._submit(_Job(self.machine.episode, source_seq, keyframe_id, gray))
                    events += self._drain(now)  # synchronous mode answers at once
        if self.machine.is_open:
            events += self.machine.tick(now)
        if not self.machine.is_open:
            events += self._summaries()
            self._finish_episode()
        return events

    def tick(self, now: float) -> list:
        """A frame the engine REJECTED before tracking (review V8 LOW-1).

        Time only: results already in are collected and the episode's clock
        moves (prompt, `late`, timeout), so an episode can never stay
        `searching` for ever behind a stream of undecodable or resized
        frames. Nothing is scanned and nothing is handed to the worker: the
        frame is not one the relocalizer can match, and the frame path never
        waits. `now` is the builder's MONOTONIC time of the frame.
        """
        self._raise_if_worker_failed()
        events = self._drain(now)
        if self.machine.is_open:
            events += self.machine.tick(now)
        if not self.machine.is_open:
            events += self._summaries()
            self._finish_episode()
        return events

    def prompt_journaled(self, at: float) -> None:
        """The engine wrote a `recovery_prompted` line with Tower-clock `at`."""
        self.machine.prompt_journaled(at)

    def close(self, now: float, *, why: str = "session_stopped") -> list:
        """Stop for good: close an open episode, then `relocalizer_stopped`.

        `why` is `session_stopped` at a normal stop and `error` when the
        engine drops a relocalizer that raised (review V5 M4-1). A worker
        that had already given up (review V8 LOW-2) makes it `error` either
        way. Idempotent: a second call journals nothing.
        """
        if self._closed:
            return []
        self._closed = True
        stopped = {"why": why}
        with self._lock:
            worker_failed = self._worker_error is not None
        if worker_failed:
            stopped = {"why": "error", "error": RelocalizerWorkerFailed.__name__}
        events = []
        try:
            events += self._drain(now)
        except Exception:  # the terminal lines below must be written regardless
            logger.exception("[Tower][WorldBuilder] relocalizer: collecting results at close failed")
        events += self.machine.close(
            now, why="session_stopped" if stopped["why"] == "session_stopped" else "relocalizer_stopped"
        )
        try:
            events += self._summaries()
        except Exception:  # observability must never cost the terminal line
            logger.exception("[Tower][WorldBuilder] relocalizer: the episode summary at close failed")
        events.append((EVENT_STOPPED, stopped))
        self._finish_episode()
        with self._lock:
            self._stop = True
            self._pending_anchor = None
            self._anchor_jobs.clear()
            self._wake.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return events

    def _say_no_references(self) -> None:
        if self._said_no_references:
            return
        self._said_no_references = True
        logger.warning(
            "[Tower][WorldBuilder] look-back relocalizer: a tracking loss with "
            "nothing to relocalize against (%s); its prompt is withheld as "
            "no-references",
            "frames at a size the calibration does not describe"
            if self._size_mismatch else "no keyframe accepted before it",
        )

    def cpu_summary(self) -> dict:
        def med(xs):
            s = sorted(xs)
            return None if not s else s[len(s) // 2]

        return {
            "attempts": self.attempts,
            "wall_ms_median": None if not self.attempt_wall_s else 1000 * med(self.attempt_wall_s),
            "cpu_ms_median": None if not self.attempt_cpu_s else 1000 * med(self.attempt_cpu_s),
            "dropped_frames": self.dropped_frames,
        }

    # -- internals -------------------------------------------------------------

    def _summaries(self) -> list:
        """`recovery_summary` for the episode that just resolved, once.

        Called on the frame thread right after the line that resolved the
        episode; empty when the setting is off, while an episode is open, or
        when this episode was already summarised. Numbers and keyframe ids
        only, never imagery.
        """
        m = self.machine
        if not self.summary_events or m.is_open or m.episode == 0 or self._summarised >= m.episode:
            return []
        self._summarised = m.episode
        ep = self._episode if self._episode is not None and self._episode.number == m.episode else None
        with self._lock:
            refs = [] if ep is None else [kid for kid, _ in ep.refs]
            best = {} if ep is None else dict(ep.best_inliers)
            payload = {
                "episode": m.episode,
                "outcome": m.state,
                "by": m.recovered_by if m.state == STATE_RECOVERED else None,
                "prompted": m.prompted_at is not None,
                "losses_joined": m.losses_joined,
                "open_s": None if m.resolved_at is None or m.lost_at is None
                else round(m.resolved_at - m.lost_at, 3),
                "references": refs,
                "history_references": 0 if ep is None else ep.n_history,
                "attempts": 0 if ep is None else ep.attempts,
                "dropped": 0 if ep is None else max(0, self.dropped_frames - ep.dropped_at_open),
                "best_links": [
                    {"ref_keyframe_id": kid, "inliers": n}
                    for kid, n in sorted(best.items(), key=lambda kn: (-kn[1], refs.index(kn[0]) if kn[0] in refs else 0))[:3]
                ],
                "best_pair": None if ep is None or ep.best_pair is None else dict(ep.best_pair),
                "best_triangle": None if ep is None or ep.best_triangle is None else dict(ep.best_triangle),
            }
        return [(EVENT_SUMMARY, payload)]

    def _note_scan(self, ep: _Episode, job: _Job, links: dict, refref) -> None:
        """The summary's evidence from one scan (worker thread).

        Costs at most ONE extra verify per new best pair of legs (the
        reference-reference link, cached per episode, that the closure of a
        below-floor pair needs); the floor triangles reuse the links
        `evaluate_acceptance` already computed.
        """
        import numpy as np

        legs = sorted(((l.n_inliers, k) for k, l in links.items()), key=lambda nk: -nk[0])
        with self._lock:
            ep.attempts += 1
            for n, k in legs:
                if n > ep.best_inliers.get(k, 0):
                    ep.best_inliers[k] = n
            improves = len(legs) >= 2 and (
                ep.best_pair is None or legs[1][0] > ep.best_pair["inliers"][1]
            )
        if improves:
            (n1, k1), (n2, k2) = legs[0], legs[1]
            o = refref(k1, k2)
            closure = None if o is None else rotation_angle_deg(
                np.asarray(links[k1].R).T @ np.asarray(links[k2].R) @ np.asarray(o.R))
            pair = {
                "refs": [k1, k2], "inliers": [n1, n2], "refs_linked": o is not None,
                "closure_deg": None if closure is None else round(float(closure), 3),
                "frame": {"source_seq": int(job.source_seq)},
            }
            with self._lock:
                ep.best_pair = pair
        # The closest triangle among legs AT the floor: exactly the pairs
        # `evaluate_acceptance` just verified (cached), so this is free.
        floor = self.acceptance.triangle_min_link_inliers
        at_floor = [(k, links[k]) for k in links if links[k].n_inliers >= floor]
        for i in range(len(at_floor)):
            for j in range(i + 1, len(at_floor)):
                (ka, la), (kb, lb) = at_floor[i], at_floor[j]
                o = ep.refref.get((ka, kb))
                if o is None:
                    continue
                c = rotation_angle_deg(np.asarray(la.R).T @ np.asarray(lb.R) @ np.asarray(o.R))
                with self._lock:
                    if ep.best_triangle is None or c < ep.best_triangle["closure_deg"]:
                        ep.best_triangle = {
                            "refs": [ka, kb], "inliers": [la.n_inliers, lb.n_inliers],
                            "closure_deg": round(float(c), 3),
                            "frame": {"source_seq": int(job.source_seq)},
                        }

    def _finish_episode(self) -> None:
        with self._lock:
            if self._episode is not None:
                self._episode.done = True
            self._pending = None

    def _ensure_thread(self) -> None:
        # Called with the lock held.
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="wb-relocalizer", daemon=True)
            self._thread.start()

    def _submit(self, job: _Job) -> None:
        if self._synchronous:
            self._run(job)
            return
        with self._lock:
            if self._pending is not None:
                self.dropped_frames += 1  # latest wins; the frame path never waits
            self._pending = job
            self._ensure_thread()
            self._wake.notify()

    def _submit_anchor(self, item) -> None:
        if self._synchronous:
            self._run_anchor(item)
            return
        with self._lock:
            self._anchor_jobs.append(item)
            self._ensure_thread()
            self._wake.notify()

    def _loop(self) -> None:
        while True:
            with self._lock:
                while self._pending is None and not self._anchor_jobs and not self._stop:
                    self._wake.wait()
                if self._stop:
                    return
                if self._anchor_jobs:
                    work, arg = self._run_anchor, self._anchor_jobs.popleft()
                else:
                    work, arg = self._run, self._pending
                    self._pending = None
            try:
                work(arg)
            except Exception as exc:  # the relocalizer must never cost the session
                logger.exception("[Tower][WorldBuilder] relocalizer attempt failed")
                with self._lock:
                    self.failed_attempts += 1
                    self._consecutive_failures += 1
                    if self._consecutive_failures >= MAX_WORKER_FAILURES:
                        # Give up (review V8 LOW-2). The worker never touches
                        # the journal: the frame thread's next call raises,
                        # and the engine journals the stop.
                        self._worker_error = exc
                        logger.error(
                            "[Tower][WorldBuilder] relocalizer worker failed %d scans in a "
                            "row; stopping the relocalizer for this session",
                            self._consecutive_failures,
                        )
                        return
            else:
                with self._lock:
                    self._consecutive_failures = 0

    def _raise_if_worker_failed(self) -> None:
        """On the frame thread: the worker gave up, so the relocalizer stops."""
        with self._lock:
            error = self._worker_error
            failures = self._consecutive_failures
        if error is not None:
            raise RelocalizerWorkerFailed(
                f"the relocalizer worker failed {failures} scans in a row; "
                f"the last: {type(error).__name__}: {error}"
            ) from error

    def _run(self, job: _Job) -> None:
        ep = self._episode
        if ep is None or ep.done or ep.number != job.episode:
            return
        wall0, cpu0 = time.perf_counter(), time.thread_time()
        v = self._verifier
        cache = self._feature_cache if self._history is not None else None
        if cache is not None and not ep.cache_pruned:
            # WORKER-ONLY state: kept to this episode's references, so the
            # cache never holds more than `reference_keyframes` +
            # `history_keyframes` feature sets (RELOC2: a historical
            # reference is extracted once, not once per episode).
            wanted = {kid for kid, _ in ep.refs}
            for kid in [k for k in cache if k not in wanted]:
                del cache[kid]
            ep.cache_pruned = True
        for kid, gray in ep.refs:
            if self._stop or ep.done:
                return  # resolved or stopped meanwhile: stop spending CPU
            if kid not in ep.ref_features:
                hit = cache.get(kid) if cache is not None else None
                ep.ref_features[kid] = hit if hit is not None else v.features(gray)
                if cache is not None:
                    cache[kid] = ep.ref_features[kid]
        frame = v.features(job.gray)
        links = {}
        for kid, _ in ep.refs:
            if self._stop or ep.done:
                return
            if kid == job.keyframe_id:
                continue
            link = v.verify(ep.ref_features[kid], frame, self._size)
            if link is not None:
                links[kid] = link

        def refref(k1, k2):
            key = (k1, k2)
            if key not in ep.refref:
                ep.refref[key] = v.verify(ep.ref_features[k1], ep.ref_features[k2], self._size)
            return ep.refref[key]

        decision = evaluate_acceptance(links, refref, self.acceptance)
        if self.summary_events:
            # Before the result is published, so the summary written when it
            # is collected already counts this scan.
            self._note_scan(ep, job, links, refref)
        result = None
        if decision is not None:
            by, used, closure = decision
            anchor = self._anchor(ep, job, frame)
            result = ("accepted", job.episode, by, _accept_detail(job, links, used, closure, anchor))
            if anchor is None:
                with self._lock:
                    self._pending_anchor = _PendingAnchor(
                        episode=job.episode, source_seq=job.source_seq, frame=frame,
                        links={k: links[k] for k in used},
                    )
        self.attempts += 1
        self.attempt_wall_s.append(time.perf_counter() - wall0)
        self.attempt_cpu_s.append(time.thread_time() - cpu0)
        if result is not None:
            with self._lock:
                self._results.append(result)

    def _anchor(self, ep: _Episode, job: _Job, frame: Features):
        """The post-loss keyframe the scanned frame is tied to, verified AT
        THE REVISIT FLOOR (review V8 M4): the nearest post-loss keyframe
        whose link to the scanned frame has >= REVISIT_MIN_INLIERS inliers.
        A nearer one below the floor is passed over; none at it leaves the
        acceptance to wait for a later keyframe (`_run_anchor`)."""
        if job.keyframe_id is not None:
            return {"keyframe_id": job.keyframe_id, "identity": True}
        with self._lock:
            post = list(ep.post)
        for kid, seq, gray in sorted(post, key=lambda p: abs(p[1] - job.source_seq)):
            link = self._verifier.verify(frame, self._verifier.features(gray), self._size)
            if _at_revisit_floor(link):  # R maps the scanned frame to the anchor
                return {"keyframe_id": kid, "identity": False, "link": link}
        return None

    def _run_anchor(self, item) -> None:
        """Tie a pending accepted frame to a keyframe that arrived after it,
        at the revisit floor: a link below it is a failed try (V8 M4)."""
        import numpy as np

        with self._lock:
            pa = self._pending_anchor
        if pa is None:
            return
        kid, _seq, gray = item
        link = self._verifier.verify(pa.frame, self._verifier.features(gray), self._size)
        with self._lock:
            if self._pending_anchor is not pa:
                return
            if not _at_revisit_floor(link):
                pa.tries += 1
                if pa.tries >= MAX_ANCHOR_TRIES:
                    self._pending_anchor = None
                return
            self._pending_anchor = None
            R_af = np.asarray(link.R)  # R maps the scanned frame to the anchor
            self._results.append(("anchored", {
                "episode": pa.episode,
                "frame": {"source_seq": int(pa.source_seq)},
                "anchor": {"keyframe_id": kid, "identity": False, "inliers": int(link.n_inliers)},
                "links": [
                    {"ref_keyframe_id": ref, "R_anchor_ref": _mat(R_af @ np.asarray(lk.R))}
                    for ref, lk in pa.links.items()
                ],
            }))

    def _drain(self, now: float) -> list:
        with self._lock:
            if not self._results:
                return []
            results = list(self._results)
            self._results.clear()
        events = []
        for item in results:
            if item[0] == "anchored":
                events.append((EVENT_ANCHORED, item[1]))
                continue
            _, episode, by, detail = item
            if episode == self.machine.episode and self.machine.is_open:
                events += self.machine.accepted(now, by, detail)
            else:
                # Too late: the episode resolved first. Its link is not
                # journaled, so it must not be anchored later either.
                with self._lock:
                    if self._pending_anchor is not None and self._pending_anchor.episode == episode:
                        self._pending_anchor = None
        return events


def _at_revisit_floor(link) -> bool:
    """A verified link strong enough to be a leg of a revisit pair."""
    return link is not None and int(link.n_inliers) >= REVISIT_MIN_INLIERS


def _accept_detail(job: _Job, links: dict, used: list, closure, anchor) -> dict:
    import numpy as np

    anchor_R = None
    if anchor is not None and not anchor.get("identity"):
        anchor_R = np.asarray(anchor["link"].R)
    out_links = []
    for kid in used:
        link = links[kid]
        R = np.asarray(link.R)
        entry = {
            "ref_keyframe_id": kid,
            "inliers": int(link.n_inliers),
            "model": link.model,
            "R_frame_ref": _mat(R),
            "t_frame_ref": [float(x) for x in np.asarray(link.t).ravel()],
        }
        if anchor is not None:
            entry["R_anchor_ref"] = _mat(R if anchor_R is None else anchor_R @ R)
        out_links.append(entry)
    return {
        "links": out_links,
        "closure_deg": None if closure is None else round(float(closure), 3),
        "frame": {"source_seq": int(job.source_seq), "keyframe_id": job.keyframe_id},
        "anchor": None if anchor is None else {
            "keyframe_id": anchor["keyframe_id"],
            "identity": bool(anchor.get("identity")),
            "inliers": None if anchor.get("identity") else int(anchor["link"].n_inliers),
        },
    }


def _mat(R) -> list:
    return [[round(float(x), 6) for x in row] for row in R]


def options_kwargs(options: dict | None) -> dict:
    """`LookBackRelocalizer` keyword arguments for the RELOC2 options.

    `options` holds only the settings that differ from today's
    (`config.world_relocalizer_options`): `window_from` (`prompt`),
    `history_keyframes` (> 0) and `summary_events` (True). Empty or None:
    no keyword at all, so the relocalizer is built exactly as before.
    """
    options = dict(options or {})
    kwargs: dict = {}
    window_from = options.get("window_from", WINDOW_FROM_LOSS)
    if window_from not in WINDOW_FROMS:
        raise ValueError(f"unknown window_from {window_from!r}")
    if window_from != WINDOW_FROM_LOSS:
        kwargs["limiter"] = LimiterParams(window_from=window_from)
    history = int(options.get("history_keyframes", 0) or 0)
    if history < 0:
        raise ValueError("history_keyframes must be >= 0")
    if history:
        kwargs["acceptance"] = AcceptanceParams(history_keyframes=history)
    if options.get("summary_events"):
        kwargs["summary_events"] = True
    return kwargs


def from_session(intrinsics, *, mode: str, options: dict | None = None, **kwargs) -> LookBackRelocalizer | None:
    """The relocalizer for a session, or None when it cannot or must not run.

    `mode` is the setting (`off` / `prompt` / `silent`). A session without a
    usable calibration gets none: the inlier thresholds were measured on
    calibrated, undistorted images, and `tracking.recovery` then stays
    `null` ("not recorded"). Frames or keyframes at any size other than the
    calibrated one are ignored by the relocalizer (the engine rejects a
    size change anyway, and the build refuses a calibration mismatch).
    `options` are the RELOC2 settings (`options_kwargs`); None is today's.
    """
    if mode not in (MODE_PROMPT, MODE_SILENT):
        return None
    kwargs = {**options_kwargs(options), **kwargs}
    K = intrinsics.camera_matrix() if intrinsics is not None else None
    if K is None:
        logger.warning(
            "[Tower][WorldBuilder] look-back relocalizer not started: the session "
            "has no calibration, and its thresholds were measured under one"
        )
        return None
    width, height = intrinsics.calibrated_width, intrinsics.calibrated_height
    if not width or not height:
        logger.warning(
            "[Tower][WorldBuilder] look-back relocalizer not started: the "
            "calibration names no resolution"
        )
        return None
    return LookBackRelocalizer(
        camera_matrix=K,
        dist_coeffs=intrinsics.dist_coeffs,
        frame_size=(width, height),
        prompts_enabled=(mode == MODE_PROMPT),
        **kwargs,
    )
