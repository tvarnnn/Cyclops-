"""Selecting, arming, pausing and stopping -- and refusing legibly.

The defects this file is written against, in the order an adversarial
reviewer would look for them:

* an experiment that is selectable but fixed -- the wire accepts a
  selection and the same experiment keeps running;
* a Start that arms nothing, so frames arrive and produce no result;
* a result from the previous experiment surviving a switch;
* two experiments running at once;
* a switch that races itself;
* a command that silently no-ops instead of refusing;
* a failed experiment that takes the whole Lab down with it.
"""

import asyncio

import pytest

from tests.cv_lab_fixtures import (  # noqa: F401
    _close_cv_lab_clients,
    armed_lab,
    jpeg_bytes,
    make_lab,
    start_and_wait,
)
from tower.cv_lab.contracts import (
    ERR_INVALID_STATE,
    ERR_LAB_BUSY,
    ERR_LAB_UNAVAILABLE,
    ERR_MALFORMED,
    ERR_STALE_RUN,
    ERR_UNKNOWN_EXPERIMENT,
    FRAME_REFUSED_IDLE,
    FRAME_REFUSED_PAUSED,
    FRAME_REFUSED_STOPPED,
    ORIGIN_CLIENT_REQUEST,
    ORIGIN_STARTUP_DEFAULT,
    STATE_FAILED,
    STATE_PAUSED,
    STATE_RUNNING,
    STATE_STOPPED,
    STATE_UNAVAILABLE,
)
from tower.modules.base import FrameProcessingError, FrameSkippedError
from tower.modules.container import ModuleContainer
from tower.modules.experimental_cv import ExperimentalCVModule


# -- the startup default ------------------------------------------------


def test_a_fresh_tower_is_already_running_its_default():
    """Every client that predates the CV Lab must keep working unchanged.

    The alternative -- boot idle and wait to be asked -- would mean a
    Tower that answers no frame until somebody sends a message no shipped
    build sends.
    """
    lab = asyncio.run(armed_lab("baseline"))
    status = lab.status()

    assert status["lifecycle"]["state"] == STATE_RUNNING
    assert status["selected"] == "baseline"
    assert status["default_experiment"] == "baseline"
    assert status["run"]["origin"] == ORIGIN_STARTUP_DEFAULT


def test_the_startup_default_says_nobody_asked_for_it():
    """`origin` is what stops "the Lab is running" reading as "somebody
    chose this"."""
    lab = asyncio.run(armed_lab("edge_detection"))
    assert lab.status()["run"]["origin"] == ORIGIN_STARTUP_DEFAULT

    asyncio.run(start_and_wait(lab, "baseline"))
    assert lab.status()["run"]["origin"] == ORIGIN_CLIENT_REQUEST


# -- selection actually selects -----------------------------------------


def test_a_selected_experiment_is_the_one_that_runs():
    """The selectable-but-fixed defect, tested against the RESULT rather
    than against the status field that claims it."""

    async def scenario():
        lab = await armed_lab("baseline")
        assert lab.process(jpeg_bytes()).result_label == "mean_intensity"
        await start_and_wait(lab, "edge_detection")
        return lab

    lab = asyncio.run(scenario())
    assert lab.status()["selected"] == "edge_detection"
    assert lab.process(jpeg_bytes()).result_label == "edge_density"


def test_every_registered_cheap_experiment_can_actually_be_selected():
    """A registry entry nobody can reach is a list item, not a capability."""

    async def scenario():
        lab = await armed_lab("baseline")
        seen = {}
        for experiment_id in (
            "edge_detection",
            "frame_quality",
            "feature_detection",
            "redaction_impact",
            "optical_flow",
            "baseline",
        ):
            outcome = await start_and_wait(lab, experiment_id)
            assert outcome.accepted, experiment_id
            result = lab.process(jpeg_bytes(64, 64, textured=True))
            seen[experiment_id] = (lab.status()["selected"], result.result_label)
        return seen

    for experiment_id, (selected, label) in asyncio.run(scenario()).items():
        assert selected == experiment_id
        from tower.experiments import experiment_metadata

        assert label == experiment_metadata(experiment_id).headline_label


def test_a_switch_mints_a_new_run_and_resets_the_counters():
    async def scenario():
        lab = await armed_lab("baseline")
        for _ in range(3):
            lab.process(jpeg_bytes())
        before = lab.status()["run"]
        await start_and_wait(lab, "edge_detection")
        return before, lab.status()["run"]

    before, after = asyncio.run(scenario())
    assert before["frames_processed"] == 3
    assert after["run_id"] != before["run_id"]
    assert after["frames_processed"] == 0
    assert after["metrics"] == []


def test_the_previous_experiment_cannot_produce_a_result_after_a_switch():
    """The staleness defect, at its source rather than at the wire.

    A switch releases the old experiment BEFORE publishing the new run id,
    so there is no window in which a result computed by one experiment can
    carry another name. Asserted by starting a switch and NOT awaiting it:
    while arming, there must be no experiment to run.
    """
    lab = asyncio.run(armed_lab("baseline"))

    async def scenario():
        lab.start("edge_detection")
        # Arming, not armed. Nothing may answer a frame here.
        with pytest.raises(FrameProcessingError) as caught:
            lab.process(jpeg_bytes())
        assert caught.value.reason == "cv_lab_starting"
        await lab.wait_until_armed()

    asyncio.run(scenario())
    assert lab.process(jpeg_bytes()).result_label == "edge_density"


def test_only_one_experiment_is_ever_armed():
    async def scenario():
        lab = await armed_lab("baseline")
        held = [lab._experiment]
        await start_and_wait(lab, "edge_detection")
        held.append(lab._experiment)
        return held

    first, second = asyncio.run(scenario())
    assert first is not second
    assert first.name == "baseline"
    assert second.name == "edge_detection"


# -- pause, resume, stop ------------------------------------------------


def test_pause_stops_processing_and_keeps_the_experiment_loaded():
    lab = asyncio.run(armed_lab("edge_detection"))
    loaded = lab._experiment

    assert lab.pause().accepted
    assert lab.status()["lifecycle"]["state"] == STATE_PAUSED
    with pytest.raises(FrameProcessingError) as caught:
        lab.process(jpeg_bytes())
    assert caught.value.reason == FRAME_REFUSED_PAUSED
    # The whole difference from stop: nothing was released, so a resume
    # costs nothing.
    assert lab._experiment is loaded

    assert lab.resume().accepted
    assert lab.status()["lifecycle"]["state"] == STATE_RUNNING
    assert lab.process(jpeg_bytes()).result_label == "edge_density"


def test_a_paused_run_keeps_the_figures_it_had():
    lab = asyncio.run(armed_lab("baseline"))
    for _ in range(4):
        lab.process(jpeg_bytes())
    lab.pause()

    run = lab.status()["run"]
    assert run["frames_processed"] == 4
    assert run["metrics"][0]["label"] == "mean_intensity"


def test_stop_ends_the_run_releases_the_experiment_and_keeps_the_summary():
    lab = asyncio.run(armed_lab("baseline"))
    for _ in range(2):
        lab.process(jpeg_bytes())

    assert lab.stop().accepted
    status = lab.status()

    assert status["lifecycle"]["state"] == STATE_STOPPED
    assert status["run"]["ended_at"] is not None
    assert status["run"]["frames_processed"] == 2
    assert lab._experiment is None
    with pytest.raises(FrameProcessingError) as caught:
        lab.process(jpeg_bytes())
    assert caught.value.reason == FRAME_REFUSED_STOPPED


def test_a_stopped_run_stops_counting_and_the_tower_does_not():
    """"I pressed Stop, is the phone still streaming?" is a real question,
    and the answer belongs to the TOWER rather than to a run that ended.

    A stopped run publishes `ended_at`, which freezes `elapsed_s`. It used
    to keep counting refused frames against that frozen window, so
    `offered_fps` climbed forever with nothing happening -- measured at
    45.3 fps from a nine-second window that had seen eight frames. The
    Tower says in as many words that "the last run's figures are final";
    now they are.
    """
    lab = asyncio.run(armed_lab("baseline"))
    lab.process(jpeg_bytes())
    lab.stop()
    frozen = lab.status()["run"]

    for _ in range(30):
        with pytest.raises(FrameProcessingError):
            lab.process(jpeg_bytes())

    status = lab.status()
    assert status["run"] == frozen, "a stopped run's figures must not move"
    assert status["run"]["frames_refused"] == 0
    assert status["run"]["frames_offered"] == 1
    # The Tower keeps counting, which is what answers the question.
    assert status["source"]["frames_offered_total"] == 31
    assert status["source"]["receiving_frames"] is True


def test_the_four_frame_counters_always_sum_to_frames_offered():
    """The invariant IS the diagnostic, so it must be true at every read.

    `frames_offered` used to be a fourth counter incremented on arrival,
    with the outcome recorded afterwards. A status read taken between the
    two -- and the status is built on a worker thread while the frame path
    runs on the loop -- published four numbers that did not add up, 2,741
    times over one run. It is derived now.
    """
    lab = asyncio.run(armed_lab("baseline"))
    for index in range(20):
        try:
            lab.process(b"" if index % 5 == 0 else jpeg_bytes())
        except FrameProcessingError:
            pass
        run = lab.status()["run"]
        assert (
            run["frames_processed"] + run["frames_refused"] + run["frames_failed"]
            == run["frames_offered"]
        )


def test_a_lab_that_never_saw_a_frame_says_so():
    lab = asyncio.run(armed_lab("baseline"))
    source = lab.status()["source"]
    assert source["frames_offered_total"] == 0
    assert source["last_frame_at"] is None
    assert source["receiving_frames"] is False


def test_frames_stop_counting_as_arriving_after_the_idle_threshold():
    clock = [1000.0]
    lab = asyncio.run(armed_lab("baseline", clock=lambda: clock[0]))
    lab.process(jpeg_bytes())
    assert lab.status()["source"]["receiving_frames"] is True

    clock[0] += 5.001
    assert lab.status()["source"]["receiving_frames"] is False


def test_stopping_after_a_stop_is_refused_rather_than_pretended():
    lab = asyncio.run(armed_lab("baseline"))
    lab.stop()
    outcome = lab.stop()
    assert outcome.accepted is False
    assert outcome.reason == ERR_INVALID_STATE


def test_resuming_something_that_is_not_paused_is_refused():
    lab = asyncio.run(armed_lab("baseline"))
    outcome = lab.resume()
    assert outcome.accepted is False
    assert outcome.reason == ERR_INVALID_STATE


def test_a_refusal_still_carries_the_whole_status():
    """A refusal that said only "no" would leave a client guessing what
    state it is now in -- and the commonest cause of a refusal is that its
    picture was already out of date."""
    lab = asyncio.run(armed_lab("baseline"))
    outcome = lab.resume()
    assert outcome.status["lifecycle"]["state"] == STATE_RUNNING
    assert outcome.status["available"]


# -- refusals -----------------------------------------------------------


def test_an_unknown_experiment_is_refused_with_what_exists():
    lab = asyncio.run(armed_lab("baseline"))
    outcome = lab.start("not_a_real_experiment")

    assert outcome.accepted is False
    assert outcome.reason == ERR_UNKNOWN_EXPERIMENT
    assert "baseline" in outcome.extra["available"]
    # And the running experiment was not disturbed.
    assert lab.status()["lifecycle"]["state"] == STATE_RUNNING
    assert lab.process(jpeg_bytes()).result_label == "mean_intensity"


@pytest.mark.parametrize("bad", [None, "", 7, {"id": "baseline"}])
def test_a_malformed_start_is_refused(bad):
    lab = asyncio.run(armed_lab("baseline"))
    outcome = lab.start(bad)
    assert outcome.accepted is False
    assert outcome.reason in (ERR_MALFORMED, ERR_UNKNOWN_EXPERIMENT)


def test_a_command_naming_a_run_that_is_gone_is_refused_not_applied():
    """Otherwise a stop drawn against run 1 stops run 2 -- the wrong run
    ended by somebody who could not have known."""

    async def scenario():
        lab = await armed_lab("baseline")
        first = lab.status()["lifecycle"]["run_id"]
        await start_and_wait(lab, "edge_detection")
        return lab, first

    lab, first_run = asyncio.run(scenario())
    outcome = lab.stop(first_run)

    assert outcome.accepted is False
    assert outcome.reason == ERR_STALE_RUN
    assert outcome.extra["current_run_id"] == lab.status()["lifecycle"]["run_id"]
    assert lab.status()["lifecycle"]["state"] == STATE_RUNNING


def test_a_command_naming_the_current_run_is_accepted():
    lab = asyncio.run(armed_lab("baseline"))
    run_id = lab.status()["lifecycle"]["run_id"]
    assert lab.pause(run_id).accepted


def test_a_second_start_during_an_arm_is_refused_not_queued():
    """Queueing would let two clients each believe they chose what runs."""

    async def scenario():
        lab = await armed_lab("baseline")
        first = lab.start("edge_detection")
        second = lab.start("frame_quality")
        await lab.wait_until_armed()
        return lab, first, second

    lab, first, second = asyncio.run(scenario())
    assert first.accepted is True
    assert second.accepted is False
    assert second.reason == ERR_LAB_BUSY
    assert lab.status()["selected"] == "edge_detection"


def test_a_stop_during_an_arm_wins_and_the_arm_installs_nothing():
    class _SlowExperiment:
        name = "slow"
        released = 0

        def load(self, settings):
            self.__class__.loading.set()
            self.__class__.proceed.wait(5)

        def run(self, raw_bytes):
            raise AssertionError("a stopped run must never process a frame")

        def release(self):
            self.__class__.released += 1

    import threading

    _SlowExperiment.loading = threading.Event()
    _SlowExperiment.proceed = threading.Event()

    async def scenario():
        lab = await armed_lab("baseline")
        # Point the registry at the slow experiment for this one start.
        from tower.experiments import EXPERIMENTS

        EXPERIMENTS["baseline"], original = _SlowExperiment, EXPERIMENTS["baseline"]
        try:
            lab.start("baseline")
            await asyncio.get_running_loop().run_in_executor(
                None, _SlowExperiment.loading.wait, 5
            )
            outcome = lab.stop()
            _SlowExperiment.proceed.set()
            await lab.wait_until_armed()
        finally:
            EXPERIMENTS["baseline"] = original
        return lab, outcome

    lab, outcome = asyncio.run(scenario())
    assert outcome.accepted is True
    assert lab.status()["lifecycle"]["state"] == STATE_STOPPED
    assert lab._experiment is None
    # The abandoned load released what it built rather than leaving it to
    # a garbage collector that may never come.
    assert _SlowExperiment.released >= 1


# -- failure is recoverable ---------------------------------------------


def test_a_command_on_a_released_lab_says_the_lab_is_gone():
    """`invalid_state` tells a client to try from another state, of a
    condition no state change can fix. `lab_unavailable` is the true one,
    and they are two codes on a closed set for a reason."""
    lab = asyncio.run(armed_lab("baseline"))
    lab.release("the module was unloaded")

    outcomes = {
        "start": lab.start("baseline"),
        "pause": lab.pause(),
        "resume": lab.resume(),
        "stop": lab.stop(),
    }
    for name, outcome in outcomes.items():
        assert outcome.accepted is False, name
        assert outcome.reason == ERR_LAB_UNAVAILABLE, name


def test_a_released_lab_ends_the_run_it_was_holding():
    """Otherwise the document keeps publishing `ended_at: null` with
    `elapsed_s` growing, for a Lab that is gone."""
    clock = [1000.0]
    lab = asyncio.run(armed_lab("baseline", clock=lambda: clock[0]))
    lab.process(jpeg_bytes())
    lab.release("the module was unloaded")
    ended = lab.status()["run"]["ended_at"]

    clock[0] += 600.0
    assert lab.status()["run"]["ended_at"] == ended
    assert lab.status()["run"]["elapsed_s"] == 0.0


def test_a_failed_arm_ends_the_run_it_never_started():
    clock = [1000.0]
    lab = asyncio.run(armed_lab("baseline", clock=lambda: clock[0]))
    lab._fail_arm(lab.status()["lifecycle"]["run_id"], "no weights")

    assert lab.status()["run"]["ended_at"] is not None
    clock[0] += 600.0
    assert lab.status()["run"]["elapsed_s"] == 0.0


def test_an_interactive_start_that_fails_leaves_the_lab_usable():
    """A load failure from a REQUEST reports `failed` with a reason and the
    next start is accepted. Otherwise a missing torch would mean a restart
    to get back to `baseline`.

    (A load failure at STARTUP used to be the terminal exception to this,
    "because a typo in TOWER_CV_EXPERIMENT must be loud". Since 2026-09-06
    it is loud AND recoverable -- see test_cv_lab_runtime_controls.py.)
    """

    class _Exploding:
        name = "boom"

        def load(self, settings):
            raise RuntimeError("weights unavailable")

        def run(self, raw_bytes):
            raise AssertionError("never armed")

        def release(self):
            return None

    async def scenario():
        lab = await armed_lab("baseline")
        from tower.experiments import EXPERIMENTS

        original = EXPERIMENTS["edge_detection"]
        EXPERIMENTS["edge_detection"] = _Exploding
        try:
            outcome = lab.start("edge_detection")
            await lab.wait_until_armed()
            failed = lab.status()
        finally:
            EXPERIMENTS["edge_detection"] = original
        recovered = await start_and_wait(lab, "baseline")
        return outcome, failed, recovered, lab

    accepted, failed, recovered, lab = asyncio.run(scenario())

    assert accepted.accepted is True  # accepted, then it failed
    assert failed["lifecycle"]["state"] == STATE_FAILED
    assert "weights unavailable" in failed["lifecycle"]["reason"]
    assert recovered.accepted is True
    assert lab.process(jpeg_bytes()).result_label == "mean_intensity"


def test_a_failed_lab_refuses_frames_with_a_reason_a_person_can_act_on():
    lab = asyncio.run(armed_lab("baseline"))
    lab._fail_arm(lab.status()["lifecycle"]["run_id"], "the model was not there")

    with pytest.raises(FrameProcessingError) as caught:
        lab.process(jpeg_bytes())
    assert caught.value.reason == "cv_lab_failed"
    assert "the model was not there" in str(caught.value)


def test_a_released_lab_reports_unavailable_and_refuses_commands():
    lab = asyncio.run(armed_lab("baseline"))
    lab.release("the module was unloaded")

    status = lab.status()
    assert status["lifecycle"]["state"] == STATE_UNAVAILABLE
    available, reason = lab.availability()
    assert available is False
    assert "unloaded" in reason
    assert lab.start("baseline").reason == ERR_LAB_UNAVAILABLE
    # And the switch flag is not left set behind it. Nothing can reach
    # `start()` past the released check today; the flag stopping being an
    # invariant is the defect.
    assert lab._switching is False


def test_an_experiment_needing_a_missing_extra_is_refused_before_it_is_tried(
    monkeypatch,
):
    """Known in advance, so it is a refusal rather than a start that fails.

    Checked PER EXPERIMENT rather than by one global torch probe: `depth`
    needs torch and timm, `object_detection` needs torchvision, and a
    single torch probe reported `object_detection` available on a Tower
    that could not run it.
    """
    import tower.cv_lab.lab as lab_module

    monkeypatch.setattr(
        lab_module,
        "_module_is_installed",
        lambda name: name not in ("torch", "torchvision", "timm"),
    )
    lab = asyncio.run(armed_lab("baseline"))
    outcome = lab.start("depth")

    assert outcome.accepted is False
    assert outcome.reason == "experiment_unavailable"
    assert outcome.extra["experiment_id"] == "depth"
    # And the catalog says so in advance rather than only on refusal.
    entry = next(e for e in outcome.status["available"] if e["id"] == "depth")
    assert entry["available"] is False
    assert "ml" in entry["unavailable_reason"]


# -- the module boundary ------------------------------------------------


def test_a_lab_refusal_reaches_the_container_as_a_skipped_frame():
    """The module must stay ACTIVE. A refusal is not a module failure, and
    `mark_failed()` is terminal."""
    module = ExperimentalCVModule("baseline")
    asyncio.run(module.load())
    asyncio.run(module.start())
    module.lab.pause()

    container = ModuleContainer(module)
    with pytest.raises(FrameSkippedError) as caught:
        container.process(jpeg_bytes())

    assert caught.value.reason == FRAME_REFUSED_PAUSED
    assert module.state.value == "active"
    module.lab.resume()
    assert container.process(jpeg_bytes()).result_label == "mean_intensity"


def test_a_frame_level_failure_still_reports_the_generic_reason():
    """A module that names no reason keeps the sentence every existing
    caller expects."""
    module = ExperimentalCVModule("baseline")
    asyncio.run(module.load())
    asyncio.run(module.start())
    container = ModuleContainer(module)

    with pytest.raises(FrameSkippedError) as caught:
        container.process(b"")

    assert caught.value.reason is None
    assert "could not process this frame" in str(caught.value)
    assert module.state.value == "active"


def test_an_experiment_that_raises_on_a_frame_is_counted_and_not_attributed():
    lab = asyncio.run(armed_lab("baseline"))
    with pytest.raises(FrameProcessingError):
        lab.process(b"")

    assert lab.status()["run"]["frames_failed"] == 1
    assert lab.frame_provenance() is None


def test_an_idle_lab_refuses_with_an_instruction():
    lab = make_lab("baseline")
    with pytest.raises(FrameProcessingError) as caught:
        lab.process(jpeg_bytes())
    assert caught.value.reason == FRAME_REFUSED_IDLE
    assert "cv_lab_start" in str(caught.value)


def test_a_superseded_arm_does_not_clear_its_successors_busy_flag():
    """`lab_busy` must mean what it says, at every moment.

    A cancelled task does not always raise: `stop()` cancels and a
    `start()` may arm again, and if the first task had already passed its
    last await it runs to completion regardless. Clearing `_switching`
    unconditionally then cleared the SECOND start's flag and let a third
    start race it. The wrong experiment still could not be installed --
    the run-id check sees to that -- but a guarantee that holds only most
    of the time is not one.
    """
    import threading

    class _Slow:
        name = "slow"
        gate = threading.Event()
        released = 0

        def load(self, settings):
            type(self).gate.wait(5)

        def run(self, raw_bytes):
            raise AssertionError("never armed")

        def release(self):
            type(self).released += 1

    async def scenario():
        from tower.experiments import EXPERIMENTS

        lab = await armed_lab("baseline")
        original = EXPERIMENTS["edge_detection"]
        EXPERIMENTS["edge_detection"] = _Slow
        try:
            lab.start("edge_detection")          # arm A, blocked in load
            first_task = lab._arm_task
            lab.stop()                            # cancels A, clears the flag
            lab.start("frame_quality")            # arm B sets the flag again
            second_task = lab._arm_task
            # Let A finish. It is cancelled, but its load had no await to
            # be cancelled at, so it runs to completion.
            _Slow.gate.set()
            busy_during_b = lab.start("baseline")
            await lab.wait_until_armed()
            for task in (first_task, second_task):
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        finally:
            EXPERIMENTS["edge_detection"] = original
        return lab, busy_during_b

    lab, busy = asyncio.run(scenario())

    # The third start, issued while B was arming, was refused.
    assert busy.accepted is False
    assert busy.reason == ERR_LAB_BUSY
    # And B is the one that ended up armed.
    assert lab.status()["selected"] == "frame_quality"
