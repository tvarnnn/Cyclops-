import asyncio
import logging
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from tower.capture import END_REASON_DISCONNECT, END_REASON_STOP
from tower.frames import FrameError, parse_and_decode_frame
from tower.metrics import SessionMetrics
from tower.modules.base import FrameSkippedError, ModuleUnavailableError
from tower.results.envelope import json_safe
from tower.routes import cv_lab_ws, results_ws

logger = logging.getLogger(__name__)

router = APIRouter()


class _ConnectionSender:
    """Serialises every send on one socket.

    A WebSocket is one TCP stream and Starlette offers no send-side lock,
    so two tasks awaiting `send_json` concurrently can interleave. Once
    the result channel introduced a second sender -- the per-connection
    push task -- serialising became a correctness requirement rather than
    a precaution.

    The frame path takes this lock too. That is a change to the
    latency-measured path and was measured before it shipped: an
    uncontended asyncio.Lock costs well under a microsecond against a
    frame budget of several milliseconds, and a connection with no
    subscription never contends at all, because no second sender exists.

    Every send is passed through `json_safe`, and THIS is the boundary
    that has to do it rather than any producer. `NaN` and `Infinity` are
    not JSON; Python emits them bare because `allow_nan` defaults True,
    and Starlette's `send_json` takes that default -- so one non-finite
    float anywhere in a payload makes the ENTIRE message unparseable by a
    strict decoder, and Swift's is strict. The result channel sanitised
    at its envelope and the CV Lab at its document builder, and between
    them they still left `frame_result` -- the highest-volume message on
    this socket -- emitting a bare `NaN` for any experiment that produced
    one. Sanitising at the socket is the only place that covers every
    message by construction, including the next one somebody adds.

    Measured at 0.045 ms for a 7.8 KB status document and proportionally
    less for a ~500-byte frame result, against a frame budget of
    milliseconds.
    """

    __slots__ = ("_websocket", "_lock")

    def __init__(self, websocket: WebSocket) -> None:
        self._websocket = websocket
        self._lock = asyncio.Lock()

    async def send(self, payload: dict) -> None:
        async with self._lock:
            await self._websocket.send_json(json_safe(payload))

    async def send_bounded(self, payload: dict, *, lock_timeout: float,
                           send_timeout: float) -> None:
        """Send with the lock wait and the send itself bounded SEPARATELY.

        Lumping them together let a slow frame send consume a result's
        whole budget and trigger a spurious "the consumer is too slow"
        drop -- when the result had not reached the socket at all, it was
        queued behind the frame path. The two waits mean different things
        and are reported differently, so they are measured apart.
        """
        await asyncio.wait_for(self._lock.acquire(), timeout=lock_timeout)
        try:
            await asyncio.wait_for(
                self._websocket.send_json(json_safe(payload)),
                timeout=send_timeout,
            )
        finally:
            self._lock.release()


async def _handle_frame_message(
    websocket: WebSocket,
    message: dict,
    metrics: SessionMetrics | None,
    sender: "_ConnectionSender",
) -> None:
    receive_start = time.perf_counter()
    try:
        frame = parse_and_decode_frame(message)
    except FrameError as exc:
        logger.warning("%s", exc)
        if metrics is not None:
            metrics.record_frame_rejected()
        # The Lab never sees this frame, so it cannot count it -- and
        # without the count, "the phone is sending garbage" and "the phone
        # is sending nothing" are the same reading. Told, not inferred.
        _note_cv_lab_rejection(websocket)
        # seq may legitimately be absent -- a frame message can fail
        # validation before seq is known. Report null rather than
        # inventing one (Rule 3: unknown stays unknown).
        await _send_frame_error(
            sender, _echo_safe(message.get("seq")), "invalid_frame", str(exc)
        )
        return

    logger.info("[Tower][Frame] #%s received: %s bytes", frame.seq, frame.byte_count)
    logger.info(
        "[Tower][Frame] #%s decoded: %sx%s",
        frame.seq,
        frame.decoded_width,
        frame.decoded_height,
    )

    if frame.dimensions_match:
        logger.info("[Tower][Frame] #%s verified", frame.seq)
    else:
        logger.warning(
            "[Tower][Frame] #%s dimension mismatch: declared %sx%s, decoded %sx%s",
            frame.seq,
            frame.declared_width,
            frame.declared_height,
            frame.decoded_width,
            frame.decoded_height,
        )

    module_container = websocket.app.state.module_container
    try:
        result = module_container.process(frame.raw_bytes)
    except FrameSkippedError as exc:
        reason = getattr(exc, "reason", None)
        logger.warning(
            "[Tower][Frame] #%s: frame-level %s, module still active: %s",
            frame.seq,
            "refusal" if reason else "failure",
            exc,
        )
        if metrics is not None:
            # A REFUSAL is not a processing error. A module that names a
            # reason declined the frame on purpose -- it is paused, or
            # idle, or arming -- and nothing failed. Counting those as
            # `frame_processing_errors` made a Lab paused for five minutes
            # report hundreds of errors in its session summary.
            #
            # `frames_rejected` still counts them, and correctly: it
            # answers "how many arriving frames are missing from the
            # numbers below", and a refused frame is missing from them.
            if reason is None:
                metrics.record_frame_processing_error()
            metrics.record_frame_rejected()
        # The module's own code when it named one, `frame_skipped`
        # otherwise. This transport does not know what the codes mean and
        # must not: a module that is deliberately not processing and a
        # frame that could not be decoded are different facts, and only
        # the module knows which one this was. See
        # `FrameProcessingError.reason`.
        await _send_frame_error(
            sender, frame.seq, reason or "frame_skipped", str(exc)
        )
        _fan_out_frame(websocket, frame)
        return
    except ModuleUnavailableError as exc:
        logger.warning(
            "[Tower][Frame] #%s: module unavailable, frame dropped: %s",
            frame.seq,
            exc,
        )
        if metrics is not None:
            metrics.record_frame_rejected()
        await _send_frame_error(sender, frame.seq, "module_unavailable", str(exc))
        _fan_out_frame(websocket, frame)
        return

    logger.info(
        "[Tower][Frame] #%s processed: %s=%.4f",
        frame.seq,
        result.result_label,
        result.result_value,
    )

    receive_to_result_ms = (time.perf_counter() - receive_start) * 1000
    if metrics is not None:
        metrics.record_frame(
            seq=frame.seq,
            byte_count=frame.byte_count,
            receive_to_result_ms=receive_to_result_ms,
            cv_processing_ms=result.processing_ms,
            stage_ms=result.stage_ms,
            tx_seq=frame.tx_seq,
            source_seq=frame.source_seq,
        )

    payload = {
        "type": "frame_result",
        "seq": frame.seq,
        "processing_ms": result.processing_ms,
        "result_value": result.result_value,
        "result_label": result.result_label,
        "stage_ms": result.stage_ms,
    }
    if result.mean_intensity is not None:
        payload["mean_intensity"] = result.mean_intensity
    # Additive, and omitted entirely when empty: a client that ignores
    # this field is unaffected, and an experiment whose headline says
    # everything does not pay for an empty object on every frame.
    # Deliberately name -> number only. This is a MEASUREMENT channel,
    # not the structured result channel -- that one needs the module
    # contract work that is blocked.
    if getattr(result, "metrics", None):
        payload["metrics"] = dict(result.metrics)
    # Who produced this number. Read HERE, between `process()` and the
    # `await` below, and that placement is the whole guarantee: the frame
    # path is synchronous up to this point, so nothing can have changed
    # the running experiment since the result was computed. Additive and
    # omitted entirely when the Tower runs no Lab.
    #
    # Fetched off `app.state` rather than imported: `tower/routes/ws.py`
    # must not know which experiment produced anything, and a test
    # (`test_shared_code_does_not_import_an_experiment_implementation`)
    # enforces that.
    provenance = _cv_lab_provenance(websocket)
    if provenance is not None:
        payload["cv_lab"] = provenance

    try:
        await sender.send(payload)
    except WebSocketDisconnect:
        logger.warning(
            "[Tower][Frame] #%s: could not send result, client disconnected mid-frame",
            frame.seq,
        )
        raise

    # After the reply, never before: see _record_capture.
    _fan_out_frame(websocket, frame)

    if metrics is not None and metrics.should_log_summary():
        logger.info("[Tower][Session] summary: %s", metrics.snapshot())


def _note_cv_lab_rejection(websocket) -> None:
    """Tell the Lab a frame arrived and was refused before it. Never raises."""
    lab = getattr(websocket.app.state, "cv_lab", None)
    if lab is None:
        return
    try:
        lab.note_frame_rejected_before_processing()
    except Exception:
        logger.exception(
            "[Tower][Frame] could not record a pre-Lab rejection; the "
            "frame_error is sent regardless"
        )


def _cv_lab_provenance(websocket):
    """Attribution for the frame result just produced, or None.

    Never raises and never blocks. A diagnostics block must not be able to
    cost a client the result it is attached to.
    """
    lab = getattr(websocket.app.state, "cv_lab", None)
    if lab is None:
        return None
    try:
        return lab.frame_provenance()
    except Exception:
        logger.exception(
            "[Tower][Frame] could not read CV Lab provenance; the result is "
            "sent without it"
        )
        return None


def _frame_observers(websocket):
    """Every registered frame observer.

    A LIST, not a single slot: more than one consumer may eventually want
    frames (a dataset recorder, and later a cartridge wanting occasional
    high-value stills), and a singleton would force the second one to
    either displace the first or patch this module again.
    """
    return getattr(websocket.app.state, "frame_observers", None) or []


def _frame_consumers(websocket):
    """Every live cartridge session that wants frames.

    A SECOND list, and the separation from `frame_observers` is
    deliberate rather than tidiness. That list is the dataset recorder's
    and is shaped around capture lineage: `_start_capture` calls
    `resumable_capture()`, `start()` and then `capture_dir()` -- the last
    of those OUTSIDE the per-observer `try`, so an object without it ends
    the connection -- and `/health` reports `capture: {armed: true}` for
    anything in it, which would be a lie about recording for a cartridge
    that counts frames and drops them.

    A cartridge session implements one method, `offer_frame`, and cannot
    be mistaken for a recorder.
    """
    return getattr(websocket.app.state, "frame_consumers", None) or []


def _fan_out_frame(websocket, frame) -> None:
    """Give a DECODED frame to the recorder and to every live cartridge.

    CALLED ON EVERY PATH THAT DECODED A FRAME, including the two where the
    CV module refused it or was unavailable. That is the whole point of
    this function existing, and it is worth saying why at length, because
    the shape it replaces looked correct for a year.

    The recorder and the live cartridges used to be reached only from the
    success path, after a `frame_result` was sent. Before the 2026-08-27
    unification that was defensible: the CV module had no refusal path, so
    a `FrameSkippedError` meant something had genuinely gone wrong and
    dropping the frame was of a piece with dropping the result.

    The CV Lab lane changed that premise and nothing noticed. It added six
    CLIENT-REACHABLE refusal states -- idle, starting, paused, stopped,
    failed, unavailable -- and a refusal is not a failure. So an ordinary
    `cv_lab_pause`, sendable from ANY connection with no ownership check,
    stopped frames reaching the DATASET RECORDER, Scene Understanding and
    Document Memory, while every one of them went on reporting itself
    healthy: `/health` said `capture: armed`, the recorder said
    `is_recording` with an open capture id, Scene said `running`, and zero
    bytes were written and zero frames observed. The walk was lost, and
    nothing in the logs said so. Worse with a terminal `module_unavailable`
    -- a typo in TOWER_CV_EXPERIMENT -- which made the Tower record nothing
    and feed no cartridge for the life of the process.

    THE RULE: what the CV module thought of a frame is the CV module's
    business. A frame that arrived and decoded is a real frame, and the
    three consumers that are not the CV module have their own lifecycles
    and their own Start and Stop. One experimental sandbox must not gate
    the frame bus.

    Still AFTER the reply on every path, never before, for the reason
    `_record_capture` gives: recording does a durable fsync'd write, and
    putting it ahead of the reply would charge every cartridge's latency
    for a recording it did not ask for.
    """
    _record_capture(websocket, frame)
    _offer_to_cartridges(websocket, frame)


def _offer_to_cartridges(websocket, frame) -> None:
    """Hand one frame to each live cartridge. After the reply, isolated.

    `offer_frame` is required to be non-blocking and to never raise --
    it replaces a slot and signals a worker thread, and everything
    expensive happens over there. The `try` is here anyway, because "is
    required to" is not "does", and a cartridge bug must not cost a
    connection that is successfully answering frames.

    Note what is NOT offered: the decoded array. Each cartridge decodes
    for itself, off the loop, on its own thread. Sharing a decode here
    would put a `cv2.imdecode` on the event loop for every connection
    whether or not any cartridge was running.

    No timestamp is passed, and that is the honest choice rather than a
    gap. `tower/frames.py` carries no time field -- there is no capture
    timestamp anywhere on this wire -- so the only clock available is
    this Tower's, and the session stamps it on receipt. That is exactly
    what `CaptureRecorder` does (`capture.py:250` writes `received_at`
    at WRITE time), and matching it means one definition of
    "tower-receipt" rather than two that differ by a few milliseconds
    and by which code path was slower.
    """
    for consumer in _frame_consumers(websocket):
        try:
            consumer.offer_frame(
                frame.raw_bytes, source_seq=frame.source_seq
            )
        except Exception:
            logger.exception(
                "[Tower][Cartridge] frame #%s was not offered to a live "
                "session; the stream continues",
                frame.seq,
            )


def _record_capture(websocket, frame) -> None:
    """Offer one raw frame to each observer, after the client has its result.

    Called AFTER the frame_result is sent, deliberately. Recording does a
    durable fsync'd write, and this is an async handler with no threadpool
    offload -- doing it before the reply would put a disk write inside the
    latency every other cartridge measures, and would block the event loop
    ahead of inference. Accessibility, which wants minimum latency, would
    pay for a recording it never asked for.

    Each observer is isolated: recording is a side errand, and a full disk
    or a permission error must never cost the client its result or take
    down a session.
    """
    for observer in _frame_observers(websocket):
        try:
            if not observer.is_recording:
                continue
            observer.write_frame(
                frame.raw_bytes,
                source_seq=frame.source_seq,
                wire_seq=frame.seq,
                tx_seq=frame.tx_seq,
                width=frame.decoded_width,
                height=frame.decoded_height,
            )
        except Exception:
            logger.exception(
                "[Tower][Capture] frame #%s not recorded; continuing", frame.seq
            )


def _echo_safe(value):
    """A client-supplied `seq` on its way back out, bounded.

    The contract types `seq` as a number and a well-formed client sends
    one, but a `frame` message can fail validation before `seq` has been
    checked -- so whatever arrived is what gets echoed. A 40,000-character
    string in that field produced a 40,000-character reply. Numbers pass
    through untouched; anything else is clipped, because the alternative
    is letting a remote party choose the size of our messages.
    """
    if value is None or isinstance(value, (int, float)):
        return value
    return str(value)[:120]


async def _send_frame_error(
    sender: "_ConnectionSender", seq: int | None, reason: str, message: str
) -> None:
    try:
        await sender.send(
            {
                "type": "frame_error",
                "seq": seq,
                "reason": reason,
                "message": message,
            }
        )
    except WebSocketDisconnect:
        logger.warning(
            "[Tower][Frame] #%s: could not send frame_error, client disconnected mid-frame",
            seq,
        )
        raise


def _finalize_stream_measurement(metrics: SessionMetrics, end_reason: str) -> None:
    try:
        snapshot = metrics.snapshot()
    except Exception:
        # Deliberately broad, unlike the receive loop's narrow handler:
        # this is a one-shot diagnostics call, and one of its call sites
        # is the endpoint's `finally` block, immediately before
        # session.client_disconnected(). Letting it propagate would skip
        # that cleanup and leave the tracker asserting a client is still
        # connected forever -- a measurement failure must never corrupt
        # connection state (Rule 3) or leak lifecycle (Rule 15). Logged
        # with a traceback rather than swallowed.
        logger.exception(
            "[Tower][Session] could not finalize measurement "
            "(end_reason=%s); connection cleanup continues",
            end_reason,
        )
        return
    logger.info(
        "[Tower][Session] final summary: %s",
        {**snapshot, "end_reason": end_reason},
    )


def _capture_workers(websocket):
    """Whatever supervises per-capture worker processes, or None.

    Fetched by name rather than injected, and tolerant of absence: most
    tests in this repository build an app and never set it, and a missing
    supervisor must mean "nothing follows captures here", not an
    AttributeError on the frame path.

    This module knows nothing about what a worker computes. It hands over
    a capture id and a directory; `main.py` decides what that is worth
    running.
    """
    return getattr(websocket.app.state, "capture_workers", None)


def _tell_cartridges_the_stream_opened(websocket, owner) -> None:
    """The stream boundary, handed to every live cartridge.

    Separate from `_tell_cartridges_about_capture` and it must stay
    separate. A CAPTURE exists only when a recorder is armed; a STREAM
    exists whenever a phone is sending frames. A cartridge that started
    on the capture signal would be silently inert on every Tower with no
    `TOWER_CAPTURE_ROOT`, which is most of them.

    Isolated per consumer for the same reason every other observer loop
    here is: a cartridge that will not start must not end a connection
    that is answering frames.
    """
    for consumer in _frame_consumers(websocket):
        try:
            consumer.stream_opened(owner)
        except Exception:
            logger.exception(
                "[Tower][Cartridge] a live session did not start with the "
                "stream; the stream continues without it"
            )


async def _close_cartridge_streams(websocket, owner) -> None:
    """End the stream for every live cartridge, OFF the event loop.

    `stream_closed` reaches `LiveSession.stop()`, which flushes -- up to
    two pages of OCR at ~1.19 s each for Document Memory -- and then
    joins its worker under a 5 s bound. Run inline, that is a Tower that
    answers no frames, no pings and no `/health` on ANY connection for as
    long as it takes.

    It is not hypothetical and needs no unusual configuration: measured
    at 5.00 s for a scene session whose phone dropped during a cold model
    load, and 2.39 s for a document session mid-dwell. A wearable
    disconnecting abruptly is the NORMAL case.

    `main.py`'s lifespan already says this about the same call -- "a
    bounded join on the event loop is still a join on the event loop" --
    and `routes/scene.py` says it about its own handlers. This is the
    third place the rule applies and the one that had it wrong.

    `stream_opened` stays inline: it starts a thread and returns.

    Started on a plain thread and then awaited, rather than `to_thread`
    alone: this runs in the connection's `finally`, and when the handler
    task is being cancelled (app shutdown, a test client's teardown) a
    second cancellation can arrive before `to_thread` has even started
    its work, leaving a stream open in a live cartridge's book for ever.
    The thread guarantees the close happens; the await is best effort.
    """
    import threading

    done = threading.Event()

    def run():
        try:
            _tell_cartridges_the_stream_closed(websocket, owner)
        finally:
            done.set()

    threading.Thread(target=run, name="tower-stream-teardown", daemon=True).start()
    await asyncio.to_thread(done.wait, 30.0)


def _tell_cartridges_the_stream_closed(websocket, owner) -> None:
    for consumer in _frame_consumers(websocket):
        try:
            consumer.stream_closed(owner)
        except Exception:
            logger.exception(
                "[Tower][Cartridge] a live session did not stop with the "
                "stream; it may still be holding a model"
            )


async def _start_capture(websocket, owner) -> None:
    """Bound a dataset recording to the existing stream window.

    Reuses stream_start/stream_stop rather than adding a message type:
    the session boundary already exists on the wire, and V1 deliberately
    makes no protocol change.

    A capture id is minted HERE, in this process, at this moment. That is
    the structural reason attaching a builder used to be a manual step:
    the id does not exist until the phone connects, so nothing could be
    launched in advance holding it. The process that mints it hands it
    over.
    """
    supervisor = _capture_workers(websocket)
    for observer in _frame_observers(websocket):
        try:
            if observer.is_recording:
                # A superseding stream_start opens a new measurement
                # window; the recording must follow it, or one capture id
                # would span two windows and stop identifying which frames
                # belong to which.
                #
                # Unconditional (no owner): this connection is deliberately
                # taking over, which is different from a dead connection
                # tearing down a live one.
                observer.stop(END_REASON_STOP)
            # Read before `start`, which clears it. The supervisor needs
            # the same lineage the manifest records, or it cannot tell a
            # reconnect from a new walk and starts a second builder on
            # one lineage.
            continues = observer.resumable_capture()
            capture_id = observer.start(owner=owner, continues=continues)
            logger.info("[Tower][Capture] recording started: %s", capture_id)
        except Exception:
            logger.exception("[Tower][Capture] could not start recording")
            continue
        # OFF-THREAD, and only this call. `supervisor.capture_opened`
        # takes the supervisor's lock, and since the supervisor became
        # multi-spec that lock is contended by Pause and Stop -- whose
        # `detach` holds it across a `terminate()` and a bounded
        # `process.wait()`. A producer that ignores SIGTERM therefore
        # blocked this line for up to the terminate timeout PER SPEC, and
        # because this ran on the event loop the whole Tower answered no
        # frames, no pings and no /health on every connection for that
        # long. Measured at 1.95 s with one stubborn worker.
        #
        # It is the same rule `_close_cartridge_streams` already follows:
        # a bounded join on the event loop is still a join on the event
        # loop. Only this call moves; `observer.start()` above stays on
        # the loop, so recorder starts remain serialised exactly as
        # before. Two connections' `capture_opened` calls may now
        # interleave in threads, which is precisely what the supervisor's
        # RLock and its lineage bookkeeping exist to handle.
        await asyncio.to_thread(
            _offer_capture_opened,
            supervisor,
            capture_id,
            observer.capture_dir(capture_id),
            continues,
        )
        _tell_cartridges_about_capture(websocket, capture_id, opened=True)


def _tell_cartridges_about_capture(websocket, capture_id, *, opened: bool) -> None:
    """Hand a live cartridge the lineage of the frames it is about to see.

    A capture id is minted in this process, at `stream_start`, and does
    not exist before a phone connects -- which is why nothing can be
    constructed holding one and why this has to be a notification rather
    than a lookup.

    It matters to exactly one cartridge today. Document Memory stamps it
    onto every document it records, and without it a stored memory says
    only "some frames, some time" -- provenance that cannot be checked is
    not provenance. Scene Understanding ignores it: it writes nothing, so
    there is nothing for a capture id to be provenance FOR.

    Isolated per consumer for the same reason `_record_capture` is: a
    cartridge bug must not end a connection that is answering frames.
    """
    for consumer in _frame_consumers(websocket):
        try:
            if opened:
                consumer.capture_started(capture_id)
            else:
                consumer.capture_stopped(capture_id)
        except Exception:
            logger.exception(
                "[Tower][Cartridge] a live session did not accept the "
                "capture lineage for %s; it will record without it",
                capture_id,
            )


def _offer_capture_opened(supervisor, capture_id, capture_dir, continues) -> None:
    """Tell the supervisor a capture exists. Never let it cost the stream.

    Isolated for the same reason `_record_capture` isolates each
    observer: building a world is a side errand, and the connection
    answering frames must not end because a subprocess could not be
    started.
    """
    if supervisor is None:
        return
    try:
        supervisor.capture_opened(capture_id, capture_dir, continues=continues)
    except Exception:
        logger.exception(
            "[Tower][Capture] could not attach a worker to capture %s; the "
            "recording continues, but nothing may be building a world from it",
            capture_id,
        )


def _stop_capture(websocket, reason: str = END_REASON_STOP, owner=None) -> None:
    supervisor = _capture_workers(websocket)
    for observer in _frame_observers(websocket):
        closed_id = None
        try:
            if not observer.is_recording:
                continue
            status = observer.stop(reason, owner=owner)
            if status is not None and status.is_open:
                # Refused: the capture belongs to a live connection that
                # superseded this one. Its worker belongs to that
                # connection too, and must not be told its capture ended.
                continue
            closed_id = status.capture_id
            logger.info(
                "[Tower][Capture] recording stopped (%s): %s frames, %s bytes",
                reason,
                status.frames_written,
                status.bytes_written,
            )
        except Exception:
            logger.exception("[Tower][Capture] could not stop recording cleanly")
        if closed_id is None:
            continue
        _tell_cartridges_about_capture(websocket, closed_id, opened=False)
        try:
            if supervisor is not None:
                supervisor.capture_closed(closed_id)
        except Exception:
            logger.exception(
                "[Tower][Capture] could not notify the worker supervisor that "
                "capture %s closed",
                closed_id,
            )


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    session = websocket.app.state.session
    active_measurement: SessionMetrics | None = None
    sender = _ConnectionSender(websocket)
    # Identity for this connection, used to stop a DEAD socket's teardown
    # from disarming a LIVE socket's recording. `object()` rather than a
    # counter: it needs to be unique and comparable, nothing more, and it
    # never leaves this process.
    connection_token = object()
    # The result channel reports this connection's subscriptions to the
    # live cartridges as DEMAND under the same token, so a subscription
    # and a stream from one phone are recognisably one phone's.
    channels = results_ws.ChannelHolder(
        time.time,
        owner=connection_token,
        live=getattr(websocket.app.state, "live_cartridges", None),
    )
    await websocket.accept()
    session.client_connected()
    logger.info("client connected")

    try:
        while True:
            try:
                message = await websocket.receive_json()
            except WebSocketDisconnect:
                raise
            except (ValueError, KeyError, UnicodeDecodeError) as exc:
                # Payload-level failures only: JSONDecodeError subclasses
                # ValueError, and a binary frame makes Starlette's
                # message["text"] raise KeyError. Deliberately NOT a bare
                # `except Exception` -- Starlette raises RuntimeError for
                # invalid *state* (socket not connected / already
                # disconnected) from a synchronous check with no await
                # point, so swallowing it and continuing would spin this
                # loop at full CPU with nothing able to cancel it
                # (Rule 15: no tight retry loops). State errors must
                # propagate and end the connection.
                logger.warning("received malformed WS message, ignoring: %s", exc)
                continue

            if not isinstance(message, dict):
                logger.warning(
                    "received malformed WS message, ignoring: not a JSON object: %.200r",
                    message,
                )
                continue

            message_type = message.get("type")
            if not isinstance(message_type, str):
                # `{"type": {"nested": 1}}` used to reach a
                # `message_type in <frozenset>` test and raise
                # `TypeError: unhashable type: 'dict'`, which nothing
                # caught -- one malformed message killed a connection
                # that was answering frames. The dispatch below assumes a
                # string; this is where that assumption is checked.
                logger.warning(
                    "received a message whose type is not a string, "
                    "ignoring: %.80r",
                    message_type,
                )
                continue

            if message_type == "ping":
                await sender.send({"type": "pong"})
            elif message_type == "frame":
                await _handle_frame_message(
                    websocket, message, active_measurement, sender
                )
            elif message_type == "stream_start":
                if active_measurement is not None:
                    _finalize_stream_measurement(
                        active_measurement, end_reason="superseded_by_stream_start"
                    )
                active_measurement = SessionMetrics()
                logger.info(
                    "[Tower][Session] stream_start: measurement window opened"
                )
                await _start_capture(websocket, connection_token)
                # After the recorder, so a cartridge that wants the
                # capture lineage has already been told what it is.
                _tell_cartridges_the_stream_opened(
                    websocket, connection_token
                )
            elif message_type == "stream_stop":
                if active_measurement is not None:
                    _finalize_stream_measurement(
                        active_measurement, end_reason="stream_stop"
                    )
                    active_measurement = None
                else:
                    logger.warning(
                        "[Tower][Session] stream_stop received with no active "
                        "measurement window"
                    )
                _stop_capture(websocket, END_REASON_STOP, owner=connection_token)
                await _close_cartridge_streams(websocket, connection_token)
            elif message_type in cv_lab_ws.CV_LAB_MESSAGE_TYPES:
                await cv_lab_ws.handle(
                    message, websocket=websocket, sender=sender
                )
            elif message_type in results_ws.RESULT_MESSAGE_TYPES:
                await results_ws.handle(
                    message,
                    websocket=websocket,
                    sender=sender,
                    channel_holder=channels,
                )
            else:
                # Answered, not merely logged. IOS-to-Tower.md 2.2: iOS
                # "never lets a request silently no-op", and
                # 04-MODULE-SYSTEM.md already requires an unsupported
                # request to "produce a clear degraded/failed state rather
                # than silently pretending" it applied. Until now this
                # branch wrote a server-side log line that no client could
                # see, so a phone asking for something this Tower does not
                # implement could not tell that from a message lost in
                # flight.
                logger.warning("received unknown message type: %s", message_type)
                await sender.send(
                    {
                        "type": "protocol_error",
                        "reason": "unknown_message_type",
                        # Clipped. This is a value the client chose, going
                        # straight back out, and a remote party must not
                        # be able to pick the size of a message this Tower
                        # sends.
                        "message_type": message_type[:120],
                        "message": (
                            "this Tower does not implement that message type"
                        ),
                    }
                )
    except WebSocketDisconnect:
        logger.info("client disconnected")
    finally:
        # Stop recording on ANY exit, not just a polite stream_stop.
        #
        # A wearable client disconnects abruptly as the normal case --
        # crash, network drop, walking out of range. Without this, a
        # recorder started by one connection stays armed, and because
        # _record_capture gates only on is_recording, the NEXT
        # connection's frames land in the previous connection's capture
        # with no stream_start and no consent. That is incidental capture
        # of someone else's imagery, which 06-PRIVACY-DATA forbids and
        # which capture.py's own docstring promises cannot happen.
        # Before the capture teardown, and unconditionally. A
        # subscription that outlived its socket would keep the shared
        # reader polling disk for a client that is gone.
        #
        # In a try/finally rather than bare, because `close()` CAN raise
        # in one case: when this handler task is itself being cancelled
        # -- app shutdown, or a test client's teardown -- the channel's
        # own sender cancellation is re-raised on purpose so the outer
        # cancellation is not swallowed. Everything below must still run
        # in that case: a capture left armed by a cancelled connection is
        # the incidental-capture defect described above, and a stream
        # left open in a live cartridge's book means its "last stream
        # out" can never come.
        try:
            await channels.close()
        finally:
            _stop_capture(
                websocket, END_REASON_DISCONNECT, owner=connection_token
            )
            # On ANY exit, not only a polite stream_stop, and for the same
            # reason the recorder is torn down here: a wearable client
            # disconnects abruptly as the NORMAL case. A scene session
            # left running by a dropped connection would hold a model,
            # park a worker, and -- worse -- keep serving a scene of a
            # room whose wearer walked out of range.
            await _close_cartridge_streams(websocket, connection_token)
            if active_measurement is not None:
                _finalize_stream_measurement(
                    active_measurement, end_reason="disconnect"
                )
            session.client_disconnected()
