import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

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

# Where Document Memory writes what it read and where `/documents` reads
# it back, when nobody says otherwise. The same reasoning as the
# observation root above, arrived at from the same failure: until
# 2026-09-07 this cartridge had NO default, so a stock Tower declared
# it unavailable and the phone told the wearer to set an environment
# variable. A cartridge a person cannot reach without editing a shell
# is not a product feature. Isolated from the sibling cartridges by
# directory, never shared with them.
DEFAULT_DOCUMENT_ROOT = str(TOWER_ROOT / "data" / "document_memory")


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    dev_mode: bool
    cv_experiment: str
    cv_device: str
    # TOWER_CV_TORCH_THREADS. The intra-op thread budget torch gets while a
    # CV Lab experiment holds a model. "auto" is 2 on CUDA and 4 on CPU,
    # from a measured sweep (`tower/experiments/depth.py`); a positive
    # integer is applied as given; 0 leaves torch's default of one thread
    # per logical CPU, which on this host is twenty OpenMP workers
    # spin-waiting between kernel launches -- the 99% CPU that object
    # detection was seen to cost.
    #
    # PROCESS-WIDE in effect, like torch's own setting, and applied on
    # the thread that runs inference each time a model-backed experiment
    # is armed. `TOWER_SCENE_TORCH_THREADS` caps the same pool for Scene
    # Understanding; if both cartridges run in one process, whichever
    # loaded last decides.
    cv_torch_threads: int | str = "auto"
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
    # PRODUCT-MANAGED SINCE 2026-09-07. `scene_understanding_mode` is the
    # tri-state a person may set -- "auto" (the default when the variable
    # is unset), "on", or "off" -- and this boolean is what that resolves
    # to: True for "on", False for "off", and for "auto" whether the
    # optional [ml] extra LOOKS installed (`find_spec` on torch and
    # torchvision, which locates without importing). The real decision is
    # still made by CONSTRUCTING the session in `cartridge_runtime`, which
    # imports torch for real and reports a failure with its reason; the
    # spec check only stops a Tower without the extra from paying for an
    # import that would fail, and lets the reason say "not installed"
    # rather than "could not be constructed". Nobody using the product
    # has to know this variable exists; an operator who wants the old
    # behaviour sets it off.
    #
    # It used to default OFF, and the default was a resource decision
    # rather than caution -- MEASURED, and the measurement is worse than
    # the estimate that first stood here. That measurement still holds
    # for a session that is RUNNING; what changed is when one runs. A
    # session now runs only while a client is subscribed to the live
    # scene AND a stream is open (`tower/scene/live.py`, WHEN IT RUNS),
    # so a Tower that offers the cartridge and is not showing it to
    # anyone pays nothing for it beyond the import at boot.
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
    scene_understanding_mode: str = "auto"
    # Which device the scene detector loads onto.
    #
    # "auto" since 2026-09-07: CUDA when it is there, CPU when it is not,
    # resolved once at construction by `_resolve_device`. It used to be
    # "cpu", on the measurement that ssdlite320 gains only 8% from the
    # GPU; that measurement is still true of ssdlite320 and is no longer
    # the deciding one, because the detector this cartridge runs is now
    # chosen by the device (`tower/scene/detect.py`): a stronger model on
    # CUDA where it is affordable, the light one on CPU where it is the
    # only one that keeps up. A person should not have to know which GPU
    # the Tower has to get the detector that fits it.
    scene_device: str = "auto"
    # Which detector the scene session runs. "auto" picks by device:
    # RT-DETRv2-R18 on CUDA, SSDLite320 on CPU (`tower/scene/detect.py`
    # has the measurements). The others are named so an operator can
    # trade speed for accuracy deliberately.
    scene_detector: str = "auto"
    # Whether the session estimates coarse facing.
    #
    # OFF by default, and this one is not close. The pose model is 956.4
    # ms per call on CPU -- 11.5x the delivered frame interval -- against
    # 43.4 ms on CUDA. It is also entirely unvalidated: no ground truth
    # for facing exists on this host. Enabling it on a CPU Tower would
    # convert the cartridge from "cheap and honest" into "wrong and
    # slow".
    scene_orientation: bool = True
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
    # about what was ever read. Since 2026-09-07 it is reachable ONLY by
    # switching the cartridge off (`document_enabled`), never by
    # forgetting to set a path: the default is `DEFAULT_DOCUMENT_ROOT`.
    document_root: str | None = None
    # Whether this Tower runs Document Memory at all.
    #
    # On by default, and that is a smaller claim than it looks: an
    # enabled cartridge records nothing until a wearer starts a session.
    # Off means no root, no session, and `/documents` answers 404.
    document_enabled: bool = True
    # Whether a live document session may attach to the stream.
    #
    # ON when the cartridge is enabled, since 2026-09-07. A session that
    # EXISTS is not a session that RECORDS: it starts only when a person
    # starts it (`document_autostart` below stays off). Off is the
    # posture for a machine that serves a library recorded elsewhere and
    # records nothing itself -- the escape hatch `world_autobuild`
    # provides for the builder.
    document_capture: bool = True
    # Where the OCR reader runs: "auto", "cuda" or "cpu". Resolved when a
    # session starts, in the cartridge, because this process must not
    # import torch to read its settings. Measured on the RTX 5070: a page
    # reads in ~0.3 s on CUDA against ~1.9 s on CPU, and the difference
    # is the difference between a flush at Stop that fits inside the
    # session's 5 s join budget and one that does not.
    document_device: str = "auto"
    # How long a recorded document is kept, in days. 30, matching the
    # session's own default rather than the store's constructor default
    # of "forever": documents are the platform's clearest case of
    # sensitive content and an unbounded window must be chosen, never
    # inherited.
    document_retention_days: float = 30.0
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

    # Transient masks on the FINAL global solve's own images. OFF, and off is
    # exactly today's solve; the lead and the manager flip it, measured.
    #
    # The wearer's hands, arms and the phone held in them are in most
    # keyframes and move WITH the camera, so SIFT matches on them are
    # geometrically consistent across rooms that share no scene at all. On
    # the 2026-09-23 coherence run's target walk the only glue between two
    # rigid islands was 7 verified pairs whose inliers sat on the held phone;
    # masking hands, arms and held phones before extraction removed 117 of its
    # 136 inliers and left 0 cross-island pairs (run FORENSICS H-A, candidate
    # architecture module M). The masks are `transients.py`'s union recipe,
    # computed once per solver image on the GPU (about 0.6 s a keyframe) and
    # reused by the surface stage, which computes the same masks today.
    #
    # Only the final solve reads it: the background solves during a walk keep
    # today's unmasked recipe. A machine that cannot run the detector still
    # solves, unmasked, and the solution says so (`transients.state`).
    world_solve_masks: bool = False
    # The seeded single-thread final solve. None (unset) is today's solve:
    # every core, and GLOMAP's own unseeded randomness. An integer seeds every
    # mapper random number generator, pycolmap's global one and the two-view
    # RANSAC, and maps on ONE thread, because GLOMAP is reproducible only
    # then (run research D1 §2.4: bit-identical seeded single-thread runs;
    # about 3.3x today's mapping time). Recorded in the solution as
    # `solve.seed` and `solve.threads`.
    world_solve_seed: int | None = None

    # Dense reconstruction after Stop. OFF, and the default is the decision.
    #
    # The stage turns the sparse solve into a per-pixel point cloud and is what
    # makes a saved world recognisable rather than a scatter of feature points.
    # It is also the most expensive thing this Tower can be asked to do: about
    # 2.4 GB of VRAM and two to four minutes of GPU per world, on a card four
    # other cartridges share. That runs AFTER the world lock is released, so it
    # blocks no capture -- but it does compete for the GPU with whatever the
    # wearer does next.
    #
    # Until this existed the only way to get a dense artifact was to run
    # scripts/world_densify.py by hand, which is not a supported product path.
    # Now it is one setting, and it is off so that turning it on is somebody's
    # decision rather than a surprise.
    world_densify: bool = False

    # Surface reconstruction: ON, and that default is also the decision.
    #
    # The saved world IS the reconstruction. A coarse surface is rebuilt during
    # the walk whenever a background solve lands, and the full one after Stop,
    # so the wearer watches the room assemble and Saved Worlds opens a surface
    # rather than a cloud of feature points. It shares the depth stage with
    # `world_densify` when both are on. It needs the solve, and degrades to the
    # sparse picture -- never to no picture -- when the depth network is not
    # installed.
    world_surface: bool = True

    # The appearance stage (docs/contracts/WORLD-BUILDER-APPEARANCE.md): the
    # wearer's redacted keyframes prepared for view-dependent blending over the
    # surface on the phone. ON with the surface, because the surface alone is
    # geometry and the saved world's appearance comes from what the glasses
    # saw. It runs after each surface build in the same child, needs the
    # surface, and costs about 40 s of mostly-CPU work on a 400-keyframe walk.
    world_appearance: bool = True

    # Finishing photographic work that was INTERRUPTED. ON, and the default is
    # the decision.
    #
    # The surface and the appearance run once, in the builder child, in the six
    # to sixteen minutes after Stop and after the world lock is released.
    # Anything that ends that child first -- a Tower shutdown, a machine sleep,
    # a crash, the supervisor's thirty-second stop grace -- discarded the work
    # permanently: `scripts/world_finalize.py` rebuilds only the sparse derived
    # tree, the serving path never writes and never spawns, and nothing
    # reconciled anything at startup. On 2026-09-22 a Tower was shut down eight
    # minutes into a build and the next start recovered nothing.
    #
    # ON is safe here in a way it would not be for a general rebuild, and the
    # reason is what `scripts/world_finish_pending.py` counts as owed. It takes
    # two kinds of evidence and NEITHER can discover a backlog: an interrupted
    # `Session.stages` entry, which only a Tower from 2026-09-22 onwards
    # writes; or, where there is no stage record at all, a
    # `<world>/surface/<session>/status.json` that says `stopped` or `running`
    # under a dead pid -- a file `surface_pipeline` alone writes, so a Tower
    # that never ran a photographic stage cannot have left one. Measured on
    # this machine: 166 worlds, ONE with a `surface/` directory, and it is the
    # interrupted one. A dry run over the real root selects exactly it.
    #
    # Off is for an operator who wants the GPU to belong to nothing they did
    # not start themselves.
    world_finish_pending: bool = True

    # The live look-back relocalizer and its spoken prompt
    # (tower/world_builder/relocalizer.py; WORLD-BUILDER-COMPONENTS.md s6).
    # `off` (the default, and today's behaviour): nothing runs, nothing new is
    # journaled, `tracking.recovery` is null. `prompt`: after a tracking loss
    # the builder matches incoming frames against the keyframes before it,
    # journals a verified revisit link when it relocalizes, and asks the
    # wearer to look back when it does not -- at most 2 prompts a minute.
    # `silent`: the relocalizer runs and records, prompts are never issued
    # (the physical test's prompt-off arm). Read by the BUILDER process,
    # which inherits the Tower's environment. Garbage reads as `off`.
    world_relocalizer: str = "off"


def get_settings() -> Settings:
    observation_enabled = _flag("TOWER_OBSERVATION_ENABLED", default=True)
    document_enabled = _flag("TOWER_DOCUMENT_ENABLED", default=True)
    scene_mode = _scene_mode(os.environ.get("TOWER_SCENE_UNDERSTANDING"))
    return Settings(
        host=os.environ.get("TOWER_HOST", "0.0.0.0"),
        port=int(os.environ.get("TOWER_PORT", "8000")),
        dev_mode=os.environ.get("TOWER_DEV_MODE", "true").lower() in ("1", "true", "yes"),
        cv_experiment=os.environ.get("TOWER_CV_EXPERIMENT", "baseline"),
        cv_device=os.environ.get("TOWER_CV_DEVICE", "auto"),
        cv_torch_threads=_torch_threads(os.environ.get("TOWER_CV_TORCH_THREADS")),
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
        world_solve_masks=world_solve_masks_setting(),
        world_solve_seed=world_solve_seed_setting(),
        world_densify=_flag("TOWER_WORLD_DENSIFY", default=False),
        world_surface=_flag("TOWER_WORLD_SURFACE", default=True),
        world_appearance=_flag("TOWER_WORLD_APPEARANCE", default=True),
        world_finish_pending=_flag("TOWER_WORLD_FINISH_PENDING", default=True),
        world_relocalizer=world_relocalizer_setting(),
        scene_understanding=_scene_enabled(scene_mode),
        scene_understanding_mode=scene_mode,
        scene_device=_device(os.environ.get("TOWER_SCENE_DEVICE"), default="auto"),
        scene_orientation=_flag("TOWER_SCENE_ORIENTATION", default=True),
        scene_detector=_scene_detector(os.environ.get("TOWER_SCENE_DETECTOR")),
        scene_torch_threads=_non_negative_int(
            os.environ.get("TOWER_SCENE_TORCH_THREADS"), default=0
        ),
        scene_autostart=_flag("TOWER_SCENE_AUTOSTART", default=True),
        document_root=_document_root(document_enabled),
        document_enabled=document_enabled,
        document_capture=(
            document_enabled and _flag("TOWER_DOCUMENT_CAPTURE", default=True)
        ),
        document_autostart=_flag("TOWER_DOCUMENT_AUTOSTART", default=False),
        document_device=_device(
            os.environ.get("TOWER_DOCUMENT_DEVICE"), default="auto"
        ),
        document_retention_days=_non_negative_float(
            os.environ.get("TOWER_DOCUMENT_RETENTION_DAYS"), default=30.0
        ),
    )


def _document_root(enabled: bool) -> str | None:
    """The one path the session writes and the read routes serve.

    Same shape as `_observation_root`, for the same reason: an explicit
    `TOWER_DOCUMENT_ROOT` wins, not choosing one no longer means the
    cartridge is unreachable, and switching it off wins over both.
    """
    if not enabled:
        return None
    return _optional_path(os.environ.get("TOWER_DOCUMENT_ROOT")) or (
        DEFAULT_DOCUMENT_ROOT
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


def _scene_detector(value: str | None) -> str:
    """One of `tower.scene.detect.DETECTOR_CHOICES`, or "auto". Not
    imported from there -- config must not import a cartridge -- so the
    list is checked at construction, where a bad name fails loudly."""
    if value is None or not value.strip():
        return "auto"
    return value.strip().lower()


def _scene_mode(value: str | None) -> str:
    """"auto", "on" or "off". Unset is auto; garbage is OFF, and logged.

    Garbage is off rather than auto for the same reason `_flag` reads
    garbage as false: a typo must never switch a people detector on.
    """
    if value is None or not value.strip():
        return "auto"
    word = value.strip().lower()
    if word == "auto":
        return "auto"
    if word in ("1", "true", "yes", "on"):
        return "on"
    if word in ("0", "false", "no", "off"):
        return "off"
    logger.warning(
        "[Tower][Config] TOWER_SCENE_UNDERSTANDING=%r is not auto, on or "
        "off; treating it as off",
        value,
    )
    return "off"


def _scene_enabled(mode: str) -> bool:
    """What the mode resolves to before anything is constructed.

    "auto" looks for the [ml] extra with `find_spec`, which locates a
    package without importing it. That is deliberately NOT the probe the
    declaration trusts -- 2026-08-27 measured `find_spec` reporting a
    package whose loader raises as present -- it is only what decides
    whether construction is attempted at all, so a Tower with no extra
    boots without paying for an import that would fail, and the reason
    it publishes can say "not installed" instead of "could not be
    constructed". Construction, which imports for real, is still where
    availability is decided.
    """
    if mode == "on":
        return True
    if mode == "off":
        return False
    from importlib.util import find_spec

    try:
        return find_spec("torch") is not None and find_spec("torchvision") is not None
    except (ImportError, ValueError):
        return False


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


# The final solve's two settings, by name. Public, and read through the two
# functions below rather than through `get_settings()`, because the reader is
# the solve itself: a `world_solve.py` child (or `world_finalize.py`) that
# inherits the Tower's environment and must not need the rest of the Tower's
# configuration -- `get_settings()` raises on a malformed `TOWER_PORT`, which is
# no reason for a solve to fail.
WORLD_SOLVE_MASKS_ENV = "TOWER_WORLD_SOLVE_MASKS"
WORLD_SOLVE_SEED_ENV = "TOWER_WORLD_SOLVE_SEED"


def world_solve_masks_setting() -> bool:
    """`TOWER_WORLD_SOLVE_MASKS`: transient masks on the final solve. Off."""
    return _flag(WORLD_SOLVE_MASKS_ENV, default=False)


def world_solve_seed_setting() -> int | None:
    """`TOWER_WORLD_SOLVE_SEED`: the seed of the single-thread final solve.

    Unset, blank, `off` or anything that is not a non-negative integer is
    None -- today's multi-threaded, unseeded solve. A typo therefore turns the
    seeded solve OFF rather than taking a solve down, the same reasoning as
    `_non_negative_int`, and the solution's `solve.seed` record shows which
    one ran.
    """
    value = os.environ.get(WORLD_SOLVE_SEED_ENV)
    if value is None or not value.strip():
        return None
    try:
        parsed = int(value.strip())
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


# The final solve's evidence gate (`world_builder/coherence_publish.py`): depth
# before publish, the gate, the relabelled components and
# `solve/<session>/components.json`. Read like the two above, by the solve
# itself. Off: today's final solve, published as the solver returned it.
WORLD_SOLVE_GATE_ENV = "TOWER_WORLD_SOLVE_GATE"


def world_solve_gate_setting() -> bool:
    """`TOWER_WORLD_SOLVE_GATE`: the evidence gate on the final solve. Off.

    Masks are a hard dependency of the gate (manager 011): with this on and
    `TOWER_WORLD_SOLVE_MASKS` off, every final solve takes the gate's fail-safe
    and attaches nothing outside the room's anchor block."""
    return _flag(WORLD_SOLVE_GATE_ENV, default=False)


# The evidence gate's CONSENSUS (`world_builder/coherence_publish.gate_by_consensus`;
# review V8 H2, manager 019): the number of mapper-seed draws a gated, seeded final
# solve maps on its one frozen database and gates, deciding attachment to the room per
# group by strict majority. 1 -- the default -- is today's single draw exactly. Each
# extra draw costs one single-thread mapping and one gate, no GPU (the gate's depth
# predictions are kept, R3): measured ~3 min per draw on a 700-keyframe walk (RUN P3-PF
# var/map, P3-H2), all of it under the world's writer lock. Read by the solve.
WORLD_SOLVE_CONSENSUS_ENV = "TOWER_WORLD_SOLVE_CONSENSUS"
# The values it accepts (review V9, M-3): ODD, so a strict majority never ties, and at
# most 7, so a typo (30 for 3: about 2 h under the lock) cannot multiply a finish. The
# cap is a bound on the cost, not a measured optimum; the run measured 3 (manager 025).
WORLD_SOLVE_CONSENSUS_VALUES = (1, 3, 5, 7)


def world_solve_consensus_setting() -> int:
    """`TOWER_WORLD_SOLVE_CONSENSUS`: the number of consensus draws. 1 (off).

    Accepts the odd values 1 to 7 (`WORLD_SOLVE_CONSENSUS_VALUES`). Unset or blank is 1,
    silently. Anything else -- garbage, zero, negative, even, or above 7 -- is 1 and is
    logged: a typo never multiplies a finish, and it is still visible."""
    value = os.environ.get(WORLD_SOLVE_CONSENSUS_ENV)
    if value is None or not value.strip():
        return 1
    try:
        parsed = int(value.strip())
    except ValueError:
        parsed = None
    if parsed in WORLD_SOLVE_CONSENSUS_VALUES:
        return parsed
    logger.warning(
        "[Tower][Config] %s=%r is not one of %s (odd, at most 7); treating it as 1, "
        "the single draw",
        WORLD_SOLVE_CONSENSUS_ENV, value,
        ", ".join(str(v) for v in WORLD_SOLVE_CONSENSUS_VALUES),
    )
    return 1


# The AREA builds (`world_builder/area_build.py`, WORLD-BUILDER-COMPONENTS.md
# §5.4): a surface and an appearance for each component the gate showed as an
# area, built by `scripts/world_finish_pending.py` at the Tower's next idle
# moment, 45-90 s each. Read by the finisher itself. Off, and off is today's
# behaviour exactly: no area is ever built, and an area a components record
# names is recorded as declined ("could not be built" on the phone) rather than
# left owed. Only reachable at all when `TOWER_WORLD_SOLVE_GATE` has written a
# components record; `scripts/world_refinish.py` builds areas regardless.
WORLD_AREA_BUILDS_ENV = "TOWER_WORLD_AREA_BUILDS"


def world_area_builds_setting() -> bool:
    """`TOWER_WORLD_AREA_BUILDS`: build the areas a components record names. Off."""
    return _flag(WORLD_AREA_BUILDS_ENV, default=False)


WORLD_RELOCALIZER_ENV = "TOWER_WORLD_RELOCALIZER"


def world_relocalizer_setting() -> str:
    """`TOWER_WORLD_RELOCALIZER`: `off` (default), `prompt` or `silent`.

    Unset or blank is `off`. `on`/`true`/`yes`/`1` mean `prompt`. Anything
    else is `off`, and logged: a typo must never start asking the wearer to
    turn around.
    """
    value = os.environ.get(WORLD_RELOCALIZER_ENV)
    if value is None or not value.strip():
        return "off"
    word = value.strip().lower()
    if word in ("prompt", "on", "1", "true", "yes"):
        return "prompt"
    if word == "silent":
        return "silent"
    if word not in ("off", "0", "false", "no"):
        logger.warning(
            "[Tower][Config] %s=%r is not off, prompt or silent; treating it "
            "as off",
            WORLD_RELOCALIZER_ENV, value,
        )
    return "off"


# RELOC2 (manager 058): three relocalizer options, each OFF by default, so an
# unset environment builds the relocalizer exactly as before and its journal
# is byte-identical. Read by the BUILDER process at each session start.
#   TOWER_WORLD_RELOCALIZER_WINDOW   `loss` (default) | `prompt`: a prompted
#                                    episode's timeout runs from the prompt.
#   TOWER_WORLD_RELOCALIZER_HISTORY  0 (default), or 4 .. 20: historical
#                                    reference keyframes added to each
#                                    episode's 10. 20 is the most the replay
#                                    measured (review F3); below 4 the spread
#                                    rule degenerates (review F7).
#   TOWER_WORLD_RELOCALIZER_SUMMARY  off (default) | on: one
#                                    `recovery_summary` line per episode.
WORLD_RELOCALIZER_WINDOW_ENV = "TOWER_WORLD_RELOCALIZER_WINDOW"
WORLD_RELOCALIZER_HISTORY_ENV = "TOWER_WORLD_RELOCALIZER_HISTORY"
WORLD_RELOCALIZER_SUMMARY_ENV = "TOWER_WORLD_RELOCALIZER_SUMMARY"
# Mirrors relocalizer.HISTORY_MIN_KEYFRAMES / HISTORY_MAX_KEYFRAMES (a test
# pins the pair); config does not import the builder's modules.
WORLD_RELOCALIZER_HISTORY_MIN = 4
WORLD_RELOCALIZER_HISTORY_MAX = 20


def world_relocalizer_options() -> dict:
    """The RELOC2 options that differ from today's; `{}` when none do.

    Garbage reads as the default and is logged: a typo must never change
    what the relocalizer accepts or when it gives up.
    """
    options: dict = {}
    window = (os.environ.get(WORLD_RELOCALIZER_WINDOW_ENV) or "").strip().lower()
    if window == "prompt":
        options["window_from"] = "prompt"
    elif window not in ("", "loss"):
        logger.warning(
            "[Tower][Config] %s=%r is not loss or prompt; treating it as loss",
            WORLD_RELOCALIZER_WINDOW_ENV, window,
        )
    history = (os.environ.get(WORLD_RELOCALIZER_HISTORY_ENV) or "").strip()
    if history:
        try:
            n = int(history)
        except ValueError:
            n = -1
        if WORLD_RELOCALIZER_HISTORY_MIN <= n <= WORLD_RELOCALIZER_HISTORY_MAX:
            options["history_keyframes"] = n
        elif n != 0:
            logger.warning(
                "[Tower][Config] %s=%r is not 0 or an integer %d..%d; treating it as 0",
                WORLD_RELOCALIZER_HISTORY_ENV, history,
                WORLD_RELOCALIZER_HISTORY_MIN, WORLD_RELOCALIZER_HISTORY_MAX,
            )
    if _flag(WORLD_RELOCALIZER_SUMMARY_ENV, default=False):
        options["summary_events"] = True
    return options


def _torch_threads(value: str | None) -> int | str:
    """"auto", or a non-negative integer. Garbage is "auto", not a crash.

    Same reasoning as `_non_negative_int`: a typo in a thread budget must
    not stop a Tower from serving frames, and the value is logged at
    startup so the typo is still visible.
    """
    if value is None or not value.strip() or value.strip().lower() == "auto":
        return "auto"
    try:
        parsed = int(value)
    except ValueError:
        return "auto"
    return parsed if parsed >= 0 else "auto"


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
