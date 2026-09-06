import os
from dataclasses import dataclass
from pathlib import Path

# The tower project root -- the directory holding `scripts/`, `models/`
# and, by default, `data/`. Resolved from this file rather than from the
# working directory: a process started with the wrong CWD would otherwise
# resolve a relative default somewhere nobody chose, and the failure that
# produces is silent. `main.py` imports this rather than computing its
# own, because two copies of a resolved root are two answers to the same
# question.
TOWER_ROOT = Path(__file__).resolve().parent.parent

# Where a producer writes its observations and where the read routes look
# for them, when nobody says otherwise. ONE constant, because the
# alternative shipped and was measured: on 2026-08-26 the producer
# defaulted to `data/object_memory`, the web process defaulted to nothing
# at all, and a real 2,203-frame walk was remembered into a store that
# every HTTP request answered 404 about until an operator set an
# environment variable by hand. Two defaults for one directory is not a
# configuration choice; it is a bug with a settings file in front of it.
#
# Absolute, unlike `capture_root`'s relative default: this value is
# handed to a CHILD PROCESS as an argv, and a relative path that resolves
# differently in the parent and the child is the same disagreement in a
# harder-to-see form.
DEFAULT_OBSERVATION_ROOT = str(TOWER_ROOT / "data" / "object_memory")


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    dev_mode: bool
    cv_experiment: str
    cv_device: str
    # Whether the CV Lab derives a live picture of what the running
    # experiment sees, and serves it over `GET /cv-lab/preview`.
    #
    # ON by default, which is a deliberate choice and not an oversight.
    # The Lab existed for months as a screen of numbers that a person
    # could read for a minute without learning whether the algorithm
    # could see the doorway they were standing in; the picture is most of
    # what makes it a laboratory rather than a telemetry dump. Nothing it
    # serves is photographic (`experiments.scene_structure` explains why),
    # nothing is written to disk, and exactly one frame exists at a time.
    #
    # The switch is real and worth keeping. An operator running the Tower
    # for a measured benchmark can turn every picture off and get the
    # frame path back to exactly what it was, byte for byte -- which is
    # also how anybody re-measuring the physical baselines should run it.
    cv_preview: bool = True
    # The longest side of a served preview. See
    # `cv_lab.contracts.PREVIEW_MAX_EDGE_PX` for where the number comes
    # from. Exposed because a Tower on a slow link may want less and a
    # bench Tower on a desk may want more, and neither should have to
    # edit a constant.
    cv_preview_max_edge_px: int = 320
    # The floor on the gap between two captures, in seconds. This is the
    # whole of "visualisation runs at its own rate": 0.05 is a 20 Hz
    # ceiling on the picture regardless of how fast the experiment runs,
    # and raising it is how a Tower whose link cannot carry 20 previews a
    # second says so.
    cv_preview_min_interval_s: float = 0.05
    # None means the dataset recorder is not armed at all. A path arms
    # it -- which still records nothing until a stream_start arrives.
    # Defaulted, unlike its neighbours: "off" is the only safe value for a
    # raw-imagery recorder, so a caller that forgets it gets no recording
    # rather than an unconfigured one.
    capture_root: str | None = None
    # Where world_build_session.py writes its worlds. Read-only to the web
    # process: it never builds, it only reports what another process has
    # already persisted. None means the result channel declares World
    # Builder's contract but reports it unavailable, which is a different
    # claim from "this Tower has no such cartridge".
    world_root: str | None = None
    # Whether a capture automatically gets a builder process attached to
    # it. On by default WHEN a world root is set, because the alternative
    # is what the first physical test did: record ten captures and build
    # a world from the one a human attached a follower to by hand.
    #
    # Off is a real configuration, not a debug flag. Reprocessing a
    # recorded capture offline wants the result channel reporting worlds
    # while nothing new is being built, and this is the escape hatch if
    # auto-attach ever misbehaves in the field.
    world_autobuild: bool = True
    # Where the observation producer writes and the read routes look.
    # Read-only to the web process, which cannot delete and cannot widen
    # the retention window the records were written under.
    #
    # None means the route answers 404 -- "this Tower serves no object
    # memory", a claim about configuration and not about what was ever
    # observed. It is now reachable ONLY by switching the cartridge off,
    # never by forgetting to set a path.
    #
    # THE DEFAULT REVERSED, AND THE REASON MATTERS.
    #
    # This used to default to None on the grounds that "a memory of what
    # a wearer's camera saw does not go on the network because a process
    # happened to start in a directory that has one". The physical test
    # on 2026-08-26 showed what that actually bought: the producer wrote
    # 64 observations to `data/object_memory` regardless, and the only
    # thing the unset default prevented was the WEARER reading their own
    # memory back. Data existed, nothing served it, and no log line said
    # why. A default that hides data from its owner while still storing
    # it protects nobody.
    #
    # So the switch moved to where the decision actually is:
    # `observation_enabled` governs whether this Tower produces or serves
    # object memory AT ALL, and when it is off, nothing is written and
    # nothing is served. Producing without serving is no longer
    # reachable by accident.
    observation_root: str | None = None
    # Whether this Tower runs the object-memory cartridge at all.
    #
    # On by default, and that is a smaller claim than it looks: an armed
    # cartridge records nothing until a wearer starts a session, exactly
    # as an armed capture recorder writes no byte until `stream_start`.
    # Off means no producer can be attached, no route answers, and
    # `observation_root` is None.
    observation_enabled: bool = True
    # Which device the attached observation producer runs its detector on.
    #
    # "cpu" by default, and the reason is CONTENTION rather than speed --
    # a correction, because the first version of this comment claimed
    # CUDA was slower and cited a figure that was not a measurement.
    #
    # It said "a CUDA pass over the whole corpus measured a WORSE 75
    # ms/frame mean". 75.0 was the running average printed at frame
    # 10,000 of a run competing with a test suite; the run's actual mean
    # was 87.8, and neither number describes a quiet host.
    #
    # Re-measured with no other work of ours running, replaying the same
    # 2,203-frame capture: CPU **39.8 to 46.9 ms/frame** across five
    # consecutive runs, against CUDA at 48.7. An independent audit on the
    # same host measured the ordering the OTHER WAY (CUDA 43.9 against
    # CPU 51.0). The spread within one device exceeds the gap between
    # them, so the honest statement is that this detector costs about the
    # same either way -- the work is single-frame preprocessing and
    # transfer, not the 320x320 forward pass, and neither device is doing
    # much of it.
    #
    # THIS HOST IS NOT QUIET. It carries several autonomous agent lanes
    # at once, and two more appeared in `git worktree list` while these
    # numbers were being taken. Read every latency figure in this
    # cartridge as a range.
    #
    # What is NOT within noise is what else wants the GPU. Object Memory
    # does not own this Tower: World Builder runs on it, the depth
    # experiment runs on it, and this cartridge's own verifier takes
    # 620 MB of VRAM when it is enabled. A producer that follows a
    # capture has no latency requirement at all -- it may fall behind and
    # catch up -- so it is the one stage that should stay off the
    # contended device.
    #
    # "auto" rather than "cpu" since 2026-08-29, resolved in the producer.
    # The measurement above still holds and auto will pick the GPU on a
    # host that has one; what changed is that the value is no longer a
    # constant a second machine has to override by hand.
    observation_device: str = "auto"
    # The retention window the producer WRITES UNDER, recorded in the
    # store manifest at first append. Every later read clamps to
    # min(persisted, requested), so this is the promise and a reader can
    # only ever narrow it.
    observation_retention_days: float = 30.0
    # What, if anything, may second-guess a detector label before a class
    # the detector cannot be trusted to name is written.
    #
    # `owlv2` IS THE DEFAULT, AND IT WAS "none" UNTIL 2026-08-29.
    #
    # The old default was a measurement, not caution. Reading the crops
    # the shipped detector produced over the real corpus found a ceiling
    # fan detected as `airplane` at 0.99 and a laptop keyboard as
    # `remote` at 0.87, so a Tower with nothing to check those labels
    # recorded neither -- and `docs/agent-handoffs/OBJECT-MEMORY-HANDOFF`
    # section 7.4 recorded turning it on as an OPEN DECISION FOR A HUMAN,
    # explicitly not one an agent should close: "the default stays `none`
    # because 94 crops from one home justify building it and not
    # switching it on for everybody."
    #
    # A human closed it. The 2026-08-29 product pass was instructed that
    # OWLv2 is this project's intended standard configuration and that
    # setting `TOWER_OBSERVATION_VERIFIER` by hand before every launch is
    # not acceptable for ordinary use. That is the ruling section 7.4 was
    # waiting for, and it is recorded here rather than in a shell script
    # so that every way of starting this Tower agrees.
    #
    # WHAT IT COSTS, AND WHY IT IS STILL SAFE TO DEFAULT.
    #
    # ~600 MB of weights, fetched once, and ~620 MB of VRAM while the
    # producer runs. A host that cannot get the weights is NOT broken by
    # this: `_build_verifier` reports the failure and runs with no
    # verifier, which narrows what is recorded to the two classes the
    # detector is trusted on. The narrowing direction is the only one
    # this setting is allowed to fail in.
    #
    # It changes what this Tower RECORDS, from two classes to fourteen.
    #
    # `recorded_classes` on the read routes is derived from this value --
    # and that means it reports what was ASKED FOR, not what loaded. The
    # web process cannot know the difference: the weights are loaded in
    # the producer, in another process, minutes later, and there is no
    # channel back. So on a host where the download fails, the routes
    # advertise fourteen classes while the producer records two, and the
    # only place that says so is the producer's own report
    # (`verifier` beside `verifier_requested`) and the loud line it
    # prints on stderr into the Tower console.
    #
    # Documented rather than papered over. Inventing a channel for it
    # would put cartridge state on the web process for a case that is
    # loud and one-off, and a client that needs certainty should read the
    # report.
    #
    # A NAME rather than a boolean, because the answer will eventually be
    # a model identifier and a boolean cannot become one. It is handed to
    # the producer's argv AND used by the read routes to say which
    # classes this Tower records, so the two cannot disagree about it.
    observation_verifier: str = "owlv2"
    # Where the verifier runs, when there is one.
    #
    # CUDA by default even though the DETECTOR defaults to CPU, and the
    # asymmetry is the measurement: the detector costs about the same on
    # either device (within noise, see above), while the verifier
    # measured 126 ms a crop on this GPU against 2,473 ms on this CPU --
    # a factor of nineteen. Splitting the two stages across devices is
    # what keeps a 2.5-second burst off the cores the detector is using,
    # and it costs 620 MB of VRAM on a card that has twelve.
    #
    # A host with no CUDA does not need this set: the verifier reports
    # the downgrade and runs on CPU rather than failing. Since 2026-08-29
    # the default says so explicitly -- "auto" rather than "cuda" -- so
    # the log line at startup names a device this host actually has
    # instead of one it was assumed to have.
    observation_verifier_device: str = "auto"
    # Whether each record gets a small filtered picture of its OWN.
    #
    # ON by default, and the default is the whole point of the setting
    # rather than a convenience.
    #
    # WHAT IT FIXES. A record has 30-day retention. The picture it points
    # at lives in `data/captures/<session_id>/frames/`, which this
    # cartridge does not own and whose lifetime it does not set -- which
    # is why every record carries `frame-referenced` and every imagery
    # payload said `imagery_retention: "capture-side"`. Today nothing
    # prunes captures at all (`CaptureRecorder.purge()` has no production
    # caller), so nothing has gone wrong yet. The first thing that prunes
    # them -- a capture pruner, or a human reclaiming the ~2.1 GB an hour
    # a recording costs -- takes the picture out of EVERY memory at once,
    # and a memory aid whose measured product value is the image becomes
    # a label and a timestamp. With this on, the record keeps a crop this
    # cartridge owns and this cartridge's retention deletes.
    #
    # WHAT IT COSTS. MEASURED over all 116 of this host's records: mean 11.7 KB
    # a keyframe (median 12.3, max 22.1) at a 384 px long side and JPEG
    # quality 80, so about 4.3 MB an hour of walking at the corpus's
    # measured rate of ~380 records an hour -- roughly 1/400th of what
    # the recording itself costs. On the frame path it costs one padded
    # numpy copy per admitted detection; the write itself happens once
    # per sighting, at its end, off the frame path.
    #
    # WHAT IT REQUIRES, AND WHAT HAPPENS WITHOUT IT. A face-detection
    # model. `KeyframeStore.write` FAILS CLOSED: with no weights, or a
    # filter that raises, it writes nothing at all rather than writing an
    # unfiltered crop with an honest label on it -- a label is not a
    # control and a file outlives every label that travelled with it. So
    # a Tower with no model keeps behaving exactly as it did, serves
    # crops out of the capture as before, and says so once, loudly, at
    # producer start rather than refusing silently a few hundred times.
    #
    # OFF is a real configuration and not a degraded one: it reproduces
    # the behaviour that shipped, where this cartridge persisted no
    # pixels whatsoever. A deployment that would rather have no
    # first-person imagery under the observation root at all sets
    # `TOWER_OBSERVATION_KEEP_IMAGERY=false` and gets exactly that.
    observation_keep_imagery: bool = True
    # Whether Scene Understanding may run at all on this Tower.
    #
    # OFF by default, and the default is a resource decision rather than
    # caution -- MEASURED, and the measurement is worse than the estimate
    # that first stood here.
    #
    # `scripts/cartridge_live_benchmark.py`, real corpus frames fed at the
    # delivered 12.0 fps, CPU, with torch capped at 2 threads: **1.4
    # cores, and 0.11% of frames skipped** on a host with room. On a host
    # already at 100% from other work the same run skipped 34%.
    #
    # Both are true and the second is the one to design around: wall-clock
    # service time is ~84 ms against an 83.5 ms interval, so there is no
    # headroom. This cartridge keeps up, and it is the first thing a
    # loaded Tower will starve.
    #
    # See `scene_torch_threads` below: capping torch's pool to 2 cut that
    # to 1.03 cores at IDENTICAL throughput, which is the single most
    # valuable thing an operator can do here.
    #
    # Off means `/cartridges` declares the contract and reports it
    # unavailable, naming this variable. It never means the Tower is
    # silent about the cartridge.
    scene_understanding: bool = False
    # Which device the scene detector loads onto.
    #
    # "cpu" by default, unlike TOWER_CV_DEVICE's "auto", and measured
    # rather than assumed: ssdlite320 is 30.4 ms on CUDA against 32.9 ms
    # on CPU -- an 8% gain, because MobileNetV3 at an internal 320 px is
    # bound by kernel-launch overhead and not arithmetic. Taking a GPU
    # for 2.5 ms a frame while World Builder wants it would be a bad
    # trade made silently.
    scene_device: str = "cpu"
    # Whether the session estimates coarse facing.
    #
    # OFF by default, and this one is not close. The pose model is 956.4
    # ms per call on CPU -- 11.5x the delivered frame interval -- against
    # 43.4 ms on CUDA. It is also entirely unvalidated: no ground truth
    # for facing exists on this host. Enabling it on a CPU Tower would
    # convert the cartridge from "cheap and honest" into "wrong and
    # slow".
    scene_orientation: bool = False
    # Cap torch's intra-op thread pool, or 0 to leave its default.
    #
    # PROCESS-GLOBAL. `torch.set_num_threads` has no per-model scope, so
    # this affects the Experimental CV Lab too. That is the only reason
    # it is not on by default, because the measurement is one-sided:
    #
    #   torch default (20 threads on this host)   4.12 cores, 9.85 fps
    #   capped at 2                               1.03 cores, 9.88 fps
    #
    # Four times the CPU for no throughput at all, which is exactly what
    # `docs/superpowers/research/2026-08-26-scene-understanding-
    # measurements.md` predicts: ssdlite320 at an internal 320 px is
    # bound by kernel-launch overhead, not arithmetic, so more threads
    # buy nothing and cost a core each.
    #
    # WHAT THE ROW ABOVE DOES NOT SAY IS WHAT IT COSTS THE LAB, and the
    # answer is large enough that "one-sided" is the wrong word for it.
    # MEASURED at the delivered 360x640, 5 repeats x 200 frames per cell
    # in separate processes, with a reversed-order control:
    #
    #                     default(20)      capped 4        capped 2
    #   object_detection    26.91 ms     37.34 (+39%)    49.11 (+83%)
    #   depth               19.73 ms     38.24 (+94%)    55.01 (+179%)
    #
    # The CV Lab's `process()` runs SYNCHRONOUSLY ON THE EVENT LOOP, so
    # that is block time every connection shares. With a scene session
    # observing concurrently -- the shipped default, since
    # `scene_autostart` is on -- capping at 2 put 20% of depth frames and
    # 8.5% of object_detection frames OVER the entire 83.3 ms delivery
    # interval, where the default and a cap of 4 put none.
    #
    # So this remains 0 by default, and an operator who sets it should
    # prefer 4 to 2 and should know they are buying CPU with latency on a
    # path that cannot yield. It is a resource lever, not a free win.
    #
    # It is ALSO NOT A FIX for the per-session thread-pool growth someone
    # will be tempted to point it at: each live session runs on a NEW OS
    # thread and torch's intra-op pool is per-thread and never reclaimed,
    # so RSS grows by roughly `get_num_threads() - 1` threads per
    # Start/Stop cycle (measured +19 threads and +8.1 MB per cycle,
    # linear, no plateau). Capping divides that rate; it does not stop it.
    # The fix is to reuse one worker thread, measured to remove the growth
    # entirely at no cost here.
    scene_torch_threads: int = 0
    # Whether `stream_start` starts a scene session and `stream_stop`
    # ends it.
    #
    # ON by default, and the default is what makes this cartridge
    # reachable from a phone at all. `IOS-to-Tower.md` 6.2: opening a
    # cartridge on the phone sends NOTHING, and a test asserts the wire
    # stays silent -- so a session that only an HTTP POST could start is
    # a contract a phone can subscribe to and will watch report "not
    # observing" forever. That is not a safety property; it is a dead
    # product path, and an adversarial review found it as one.
    #
    # Enabling the cartridge is already the opt-in. This does not widen
    # what a Tower may do, only when it does it.
    scene_autostart: bool = True
    # Where a document session writes what it read, and where the
    # document routes read it back.
    #
    # Named `document_root`, not `document_memory_root`, and the name is
    # load-bearing rather than a preference:
    # `test_document_memory_is_not_registered_as_a_production_module`
    # asserts the substring "document_memory" appears nowhere in the raw
    # text of `main.py`, which this value has to reach. Object Memory
    # solved the same problem the same way with `observation_root`.
    #
    # None means the document routes answer 404 and `/cartridges` reports
    # the cartridge unavailable -- a claim about configuration, never
    # about what was ever read.
    document_root: str | None = None
    # Whether a live document session may attach to the stream.
    #
    # OFF by default and separately from the root, because the two
    # answer different questions. A root with capture off is a Tower that
    # will serve a library recorded elsewhere and record nothing itself,
    # which is the right posture for a machine reprocessing captures
    # offline -- and the same escape hatch `world_autobuild` provides.
    document_capture: bool = False
    # Whether `stream_start` starts a DOCUMENT session.
    #
    # OFF by default, unlike Scene Understanding's, and the asymmetry is
    # the difference between the two cartridges: this one WRITES. A
    # session that persists what a wearer read gets an explicit start,
    # which is the standard 06-PRIVACY-DATA.md holds the dataset recorder
    # to -- "arming is not recording".
    #
    # The cost is smaller than it looks: the half of this cartridge a
    # phone reaches is the library, over HTTP, and that works whether or
    # not anything is currently recording.
    document_autostart: bool = False
    # Keyframes between mid-walk rebuilds in the attached builder.
    #
    # NOT the script's own default, which is 0 -- "build once, at the
    # end". That default is correct for a batch reprocess and wrong for a
    # live walk: it is why the 2026-08-24 test showed a climbing keyframe
    # count with no geometry at all until the capture closed, and then
    # every figure appearing at once.
    world_rebuild_every: int = 4
    # Whether the attached builder tries to place the walk's segments in
    # one frame when the walk ends.
    #
    # On by default because off is what we shipped, and off is why every
    # walk so far arrived on the phone as a heap of disconnected
    # fragments: the registration pass existed, was tested, persisted and
    # served, and nothing ever invoked it. Refusal is still the default
    # answer for any individual pair, so this makes the world no more
    # confident -- only less silent.
    #
    # The escape hatch is real and worth keeping: registration is the
    # only step in the walk that costs tens of seconds, it runs once at
    # the end in the builder subprocess, and a host that wants a walk
    # finalised the instant it stops can turn it off without losing the
    # reconstruction.
    world_register: bool = True
    # The global solver (tower/world_builder/global_solve.py): a
    # structure-from-motion solve over EVERY keyframe of the session, run
    # as a child of the builder every few dozen keyframes and once more
    # after Stop. It is what turns a walk into one room rather than a bag
    # of fragments (docs/world-builder-reconstruction-experiments.md, E8:
    # 58 -> 428 of 438 keyframes in one frame on the 2026-09-06 walk).
    # Requires pycolmap; without it the builder says so and behaves as
    # before. The Sim3 registrar (world_register) is skipped whenever the
    # solver produced a solution.
    world_solve: bool = True


def get_settings() -> Settings:
    observation_enabled = _flag("TOWER_OBSERVATION_ENABLED", default=True)
    return Settings(
        host=os.environ.get("TOWER_HOST", "0.0.0.0"),
        port=int(os.environ.get("TOWER_PORT", "8000")),
        dev_mode=os.environ.get("TOWER_DEV_MODE", "true").lower() in ("1", "true", "yes"),
        cv_experiment=os.environ.get("TOWER_CV_EXPERIMENT", "baseline"),
        cv_device=os.environ.get("TOWER_CV_DEVICE", "auto"),
        cv_preview=_flag("TOWER_CV_PREVIEW", default=True),
        cv_preview_max_edge_px=_non_negative_int(
            os.environ.get("TOWER_CV_PREVIEW_MAX_EDGE_PX"), default=320
        )
        or 320,
        cv_preview_min_interval_s=_non_negative_float(
            os.environ.get("TOWER_CV_PREVIEW_MIN_INTERVAL_S"), default=0.05
        ),
        capture_root=_optional_path(os.environ.get("TOWER_CAPTURE_ROOT")),
        world_root=_optional_path(os.environ.get("TOWER_WORLD_ROOT")),
        observation_root=_observation_root(observation_enabled),
        observation_enabled=observation_enabled,
        observation_device=_device(
            os.environ.get("TOWER_OBSERVATION_DEVICE"), default="auto"
        ),
        observation_retention_days=_non_negative_float(
            os.environ.get("TOWER_OBSERVATION_RETENTION_DAYS"), default=30.0
        ),
        observation_verifier=_verifier(
            os.environ.get("TOWER_OBSERVATION_VERIFIER")
        ),
        observation_verifier_device=_device(
            os.environ.get("TOWER_OBSERVATION_VERIFIER_DEVICE"), default="auto"
        ),
        observation_keep_imagery=_flag(
            "TOWER_OBSERVATION_KEEP_IMAGERY", default=True
        ),
        world_autobuild=os.environ.get("TOWER_WORLD_AUTOBUILD", "true").lower()
        in ("1", "true", "yes"),
        world_rebuild_every=_non_negative_int(
            os.environ.get("TOWER_WORLD_REBUILD_EVERY"), default=4
        ),
        world_register=_flag("TOWER_WORLD_REGISTER", default=True),
        world_solve=_flag("TOWER_WORLD_SOLVE", default=True),
        scene_understanding=_flag("TOWER_SCENE_UNDERSTANDING", default=False),
        scene_device=os.environ.get("TOWER_SCENE_DEVICE", "cpu"),
        scene_orientation=_flag("TOWER_SCENE_ORIENTATION", default=False),
        scene_torch_threads=_non_negative_int(
            os.environ.get("TOWER_SCENE_TORCH_THREADS"), default=0
        ),
        scene_autostart=_flag("TOWER_SCENE_AUTOSTART", default=True),
        document_root=_optional_path(os.environ.get("TOWER_DOCUMENT_ROOT")),
        document_capture=_flag("TOWER_DOCUMENT_CAPTURE", default=False),
        document_autostart=_flag("TOWER_DOCUMENT_AUTOSTART", default=False),
    )


def _observation_root(enabled: bool) -> str | None:
    """The one path both the producer and the read routes will use.

    An explicit `TOWER_OBSERVATION_ROOT` still wins, because an operator
    who chose a directory has said something the default cannot know. The
    difference from before is that NOT choosing one is no longer a way to
    end up with two different answers.

    Switching the cartridge off wins over both: a Tower told not to run
    object memory does not get a root because somebody left a variable
    set from last week.
    """
    if not enabled:
        return None
    return _optional_path(os.environ.get("TOWER_OBSERVATION_ROOT")) or (
        DEFAULT_OBSERVATION_ROOT
    )


# Every verifier this build can construct.
#
# Named HERE, in settings, rather than only in the producer script, and
# that duplication is deliberate -- it is the smaller of two evils. The
# alternative was for shared config to import the cartridge, which the
# boundary forbids. What must not happen is what did: `config.py`
# accepted any string and treated "not none" as "a verifier exists", so
# `TOWER_OBSERVATION_VERIFIER=owvl2` (a transposition) told the read
# routes that fourteen classes were recordable AND handed the producer a
# name it refuses, killing it at spawn. A Tower advertising twelve
# classes it had just made unrecordable.
#
# `scripts/object_memory_session.py` still validates its own argument;
# these two lists agreeing is checked by
# `test_the_settings_and_the_producer_agree_about_verifier_names`.
KNOWN_VERIFIERS = ("none", "owlv2")

# What an unset TOWER_OBSERVATION_VERIFIER means. Named rather than
# inlined so `Settings`, `_verifier` and the tests all read the same
# constant; the reasoning for the value is on `Settings`.
DEFAULT_OBSERVATION_VERIFIER = "owlv2"


def _verifier(value: str | None) -> str:
    """Which verifier to run, or "none". An unknown name falls back.

    Falls back rather than raising, for the same reason
    `_non_negative_int` does: a typo in an optional variable must not
    take a Tower down for a cartridge it may not even be running. It is
    logged at startup, so the typo is visible rather than silent -- and
    the fallback is the SAFE direction, because "none" narrows what the
    routes claim rather than widening it.
    """
    if value is None:
        return DEFAULT_OBSERVATION_VERIFIER
    if not value.strip():
        # An explicitly EMPTY variable is a person switching it off, and
        # falling back to the default there would ignore them.
        return "none"
    name = value.strip().lower()
    return name if name in KNOWN_VERIFIERS else "none"


# What this Tower will accept as a device for the Object Memory
# producer. `auto` is the same word `TOWER_CV_DEVICE` uses and resolves
# the same way -- see `cartridge_runtime._resolve_device`: auto
# downgrades, cuda does not.
KNOWN_DEVICES = ("auto", "cpu", "cuda")


def _device(value: str | None, *, default: str = "auto") -> str:
    """"auto", "cpu" or "cuda". Anything else falls back rather than
    failing late.

    A typo here would otherwise reach a child process as an argv, be
    rejected by `torch.device`, and surface as a producer that exits
    immediately -- visible only as a warning in the Tower log, hours into
    a walk that remembered nothing. The fallback is the default.

    `auto` was added because the alternative is a machine-specific
    constant in a shared repository. `cuda` was this cartridge's default
    for the verifier and is correct on the host it was measured on; on a
    host without a GPU it is a value that has to be un-set by hand before
    anything works, which is the same "edit the environment before every
    run" problem the whole session surface exists to remove. It is
    RESOLVED IN THE PRODUCER, not here: this process deliberately does
    not import torch, and a resolution that needed it would put a ~2 s
    import on the web process's startup for a decision a child is about
    to make anyway.
    """
    if value is None:
        return default
    normalised = value.strip().lower()
    if normalised in KNOWN_DEVICES:
        return normalised
    return default


def _non_negative_float(value: str | None, *, default: float) -> float:
    """A retention window, or the default. Never negative.

    `ObservationStore` raises on a negative window, and it is right to:
    a negative retention has no meaning. But raising at STARTUP over a
    typo in an optional variable would take a Tower down for a cartridge
    it may not even be running, so a bad value falls back and the
    effective figure is logged.
    """
    if value is None or not value.strip():
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return parsed if parsed >= 0 else default


def _flag(name: str, *, default: bool) -> bool:
    """An on/off environment variable, read the same way every time.

    Both cartridge lanes arrived at this helper independently and for the
    same reason, which is the reason it is one function: so a fourth flag
    cannot arrive with a fifth spelling of "true". The accepted set is one
    list, below, and nothing else in this file may grow its own.

    `TOWER_DEV_MODE` and `TOWER_WORLD_AUTOBUILD` spell a similar test
    inline and are deliberately left alone: both default to true through
    `os.environ.get(name, "true")`, which makes an EXPLICITLY BLANK value
    false. That is the opposite of what a blank means everywhere else in
    this file -- `_optional_path` reads a blank as "unset, use the
    default" -- and quietly changing two existing settings to fix an
    inconsistency is not this change's business. New flags get the
    consistent reading; the old two keep theirs until somebody decides.
    """
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    # "on" is accepted alongside "yes": the two lanes shipped
    # different sets, and the narrower one read `on` as FALSE --
    # which would silently disable a cartridge whose flag defaults
    # ON. A spelling of true must never mean false.
    return value.strip().lower() in ("1", "true", "yes", "on")


def _non_negative_int(value: str | None, *, default: int) -> int:
    """A malformed cadence falls back rather than taking the Tower down.

    Unlike TOWER_PORT, which raises at import on garbage, this one has a
    safe answer: the default. A typo in a rebuild interval must not stop
    a Tower from serving frames, and the value is reported at startup so
    the typo is still visible.
    """
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed >= 0 else default


def _optional_path(value: str | None) -> str | None:
    """Blank means unset. A shell exporting an empty variable is saying
    "no", and treating that as the current working directory would arm a
    raw-imagery recorder somewhere nobody chose.
    """
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None
