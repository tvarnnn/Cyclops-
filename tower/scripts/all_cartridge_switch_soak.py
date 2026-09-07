#!/usr/bin/env python
"""Five cartridges, one Tower, over and over — the switch nobody could test.

Each of the four autonomous lanes proved its own cartridge starts and stops
cleanly. None of them could prove the FIVE coexist, because each had only its
own branch. This is that test, and it is the reason the all-cartridges
integration exists as a lane rather than as four merges.

`cartridge_switch_soak.py` (World Builder's) walks World Builder against the
CV Lab and is kept as it is; this is its sibling and reuses its vocabulary --
the same sampler, the same lock report, the same verdict shape -- so a reader
of one can read the other. What is new is the other three cartridges and, more
importantly, the assertions BETWEEN them.

One cycle, in the order a wearer would move:

    idle
      -> CV Lab            arm, stream, stop
      -> World Builder     session start, stream, Stop, finalize, session stop
      -> Object Memory     session start, stream, pause, resume, stop
      -> Document Memory   session start, stream, stop  (library must survive)
      -> Scene Understanding
                           stream WITHOUT a watcher   (must stay stopped)
                           + watcher                  (must run)
                           - watcher                  (must stop and release)
      -> CV Lab again      the same Tower still arms

WHAT IS ACTUALLY BEING ASSERTED, AND WHY EACH ONE EARNED ITS PLACE

Every cartridge here activates by a DIFFERENT mechanism, and that is the
integration risk in one sentence. World Builder and Object Memory are gated
subprocess workers behind a `CartridgeSession`. Document Memory is an
in-process `LiveSession` a person starts. Scene Understanding is demand-driven
and runs only while somebody streams AND somebody watches. The CV Lab is
armed on the socket and processes frames inline. Four disciplines, one frame
path, and no lane could see more than its own.

So each step asserts three things rather than one:

  * the cartridge it just entered did what it was asked;
  * no OTHER cartridge started because a camera happened to be streaming --
    the "a generic incoming stream must not start unrelated work" rule, which
    is checked after every stream in this script and is the single most
    valuable thing here;
  * the cartridge it just left gave back what it held -- worker gone, model
    released, subscription closed -- before the next one asks for it.

Plus, at every step: the Tower is alive, `/health` answers, the listener is
still bound, and threads, RSS and child processes are sampled so the verdict
can compare the last third of cycles against the first.

    python scripts/all_cartridge_switch_soak.py --cycles 3
    python scripts/all_cartridge_switch_soak.py --cycles 5 --json report.json

Exit 0 means every cycle held. A non-zero exit names what did not.

Frames are synthetic noise. That is deliberate and it is a limitation: this
measures LIFECYCLE -- ownership, gating, release, leak -- and not perception.
Nothing here says a document was read or a person was seen, and the
`--with-models` half of `unified_cartridge_smoke.py` plus the physical test
are what cover that.
"""

import argparse
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import psutil  # noqa: E402

from cartridge_switch_soak import (  # noqa: E402
    TOWER_ROOT,
    _drain,
    _feed,
    _free_port,
    _get,
    _post,
    _wait_until,
    build_frames,
    builders_in,
    sample,
    stop_tower,
    wait_for_health,
)
from tower.artifact_paths import artifact_root_arg  # noqa: E402
from tower.process_ownership import (  # noqa: E402
    interpreter_command,
    interpreter_environment,
)

DEFAULT_ROOT = Path("data/soak/all_cartridge_switch")
WB_SESSION = "/cartridges/world_builder/session"
OM_SESSION = "/cartridges/object_memory/session"
DOC_SESSION = "/documents-session"
OBSERVATION_MARKER = "object_memory_session"


def start_tower(port: int, root: Path, log_path: Path) -> subprocess.Popen:
    """A Tower with every cartridge reachable and nothing auto-starting.

    Deliberately NOT `cartridge_switch_soak.start_tower`, which sets
    `TOWER_SCENE_AUTOSTART=false`. That flag does not merely stop Scene
    Understanding following a stream -- `SceneLive._reconcile` returns
    early on it, so the whole demand path is off and only
    `POST /scene/start` can run a session. Borrowing that Tower made this
    script wait ninety seconds for a session the configuration had
    disabled, and report the product for it. The demand model IS the thing
    under test here, so it is left on.

    `TOWER_DOCUMENT_AUTOSTART` stays off, which is its shipped default and
    is what makes "a stream does not start a document session" a real
    check rather than a tautology.
    """
    env = interpreter_environment()
    env["PYTHONPATH"] = str(TOWER_ROOT)
    env["TOWER_CAPTURE_ROOT"] = str(root / "capture")
    env["TOWER_WORLD_ROOT"] = str(root / "world")
    env["TOWER_OBSERVATION_ROOT"] = str(root / "object_memory")
    env["TOWER_DOCUMENT_ROOT"] = str(root / "document_memory")
    env["TOWER_WORLD_AUTOBUILD"] = "true"
    env["TOWER_WORLD_REBUILD_EVERY"] = "4"
    env["TOWER_SCENE_UNDERSTANDING"] = "auto"
    env["TOWER_SCENE_AUTOSTART"] = "true"
    env["TOWER_DOCUMENT_ENABLED"] = "true"
    env["TOWER_DOCUMENT_AUTOSTART"] = "false"
    # A soak that pins every core is a soak nobody runs, and torch's
    # default pool would take them all.
    env["TOWER_SCENE_TORCH_THREADS"] = "2"
    return subprocess.Popen(
        [
            *interpreter_command("-m", "uvicorn", "tower.main:app"),
            "--host", "127.0.0.1", "--port", str(port), "--log-level", "info",
        ],
        cwd=str(TOWER_ROOT),
        env=env,
        # A FILE, never a pipe: workers inherit this stdout and log per
        # rebuild; a pipe nobody drains would wedge them.
        stdout=log_path.open("w", encoding="utf-8", errors="replace"),
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        creationflags=(
            subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        ),
    )


def observers_in(row: dict) -> list[dict]:
    """Object Memory producers among the Tower's descendants.

    `sample()` labels anything that is not a builder or a solver `other`, so
    the producer is found by its command line rather than by its kind.
    """
    return [d for d in row["descendants"] if OBSERVATION_MARKER in d.get("cmdline", "")]


def health_workers(base: str) -> list:
    return _get(base, "/health")[1].get("capture_workers", {}).get("workers") or []


async def _subscribe(ws, cartridge: str, result_type: str, request_id: str):
    # Both keys, or the channel answers `malformed_request`. Scene
    # Understanding publishes a `live` result; the other cartridges publish
    # `status`.
    await ws.send(json.dumps({
        "type": "result_subscribe",
        "cartridge": cartridge,
        "result_type": result_type,
        "request_id": request_id,
    }))
    return await _drain(ws, ("result_subscribed", "result_error"))


async def _unsubscribe(ws, subscription_id: str):
    await ws.send(json.dumps({
        "type": "result_unsubscribe", "subscription_id": subscription_id,
    }))
    return await _drain(ws, ("result_unsubscribed", "result_error"))


async def _quiet(ws):
    """A round trip, so the Tower has certainly processed what came before."""
    await ws.send(json.dumps({"type": "ping"}))
    await _drain(ws, ("pong",))


async def _await_until(predicate, timeout: float, poll: float = 0.25) -> bool:
    """`_wait_until`, off this script's own event loop.

    `_wait_until` sleeps and polls synchronously. Called directly from a
    coroutine that also owns the websocket, it blocks the client loop for as
    long as it waits -- and a model load or a bounded stop join is long
    enough that the client stops answering the server's keepalive ping and
    the socket is closed under us with `1011 keepalive ping timeout`. That
    reads exactly like a Tower that stalled, and on the first run of this
    script it was mistaken for one; the Tower's own log showed it answering
    `/scene` 200 throughout. The waiting happens on a thread instead.
    """
    return await asyncio.to_thread(_wait_until, predicate, timeout, poll)


async def soak(args, base: str, tower_pid: int | None, root: Path, findings: list[str]) -> dict:
    import websockets

    world_root = root / "world"
    frames = build_frames()
    rows: list[dict] = []
    seq = 0

    def record(cycle: int, step: str, **extra) -> dict:
        row = {"cycle": cycle, "step": step, "t": round(time.time(), 3), **extra}
        row.update(sample(tower_pid, world_root))
        rows.append(row)
        return row

    def nothing_unrelated(cycle: int, step: str, row: dict, *, allow: tuple = ()) -> None:
        """The rule that matters most: a stream is not a request to run
        everything. Anything expensive that is alive here and was not asked
        for is a finding."""
        if "builder" not in allow:
            leaked = builders_in(row)
            if leaked:
                findings.append(
                    f"cycle {cycle} {step}: a World Builder builder is alive and was "
                    f"not asked for: {leaked}"
                )
        if "observer" not in allow:
            leaked = observers_in(row)
            if leaked:
                findings.append(
                    f"cycle {cycle} {step}: an Object Memory producer is alive and was "
                    f"not asked for: {leaked}"
                )
        if "scene" not in allow:
            state = _get(base, "/scene")[1]
            if isinstance(state, dict):
                running = (state.get("lifecycle") or {}).get("state")
                if running in ("running", "starting"):
                    findings.append(
                        f"cycle {cycle} {step}: Scene Understanding is {running} and "
                        "nobody is watching it"
                    )
        if "document" not in allow:
            state = _get(base, DOC_SESSION)[1]
            if isinstance(state, dict):
                running = (state.get("session") or {}).get("state")
                if running in ("running", "starting"):
                    findings.append(
                        f"cycle {cycle} {step}: a Document Memory session is {running} "
                        "and nobody started it"
                    )

    def tower_still_serving(cycle: int, step: str) -> None:
        status, _ = _get(base, "/health", timeout=10.0)
        if status != 200:
            findings.append(f"cycle {cycle} {step}: /health answered {status}")

    record(-1, "baseline")
    async with websockets.connect(
        f"ws://{args.host}:{args.port}/ws", max_size=8 * 1024 * 1024, open_timeout=60
    ) as ws:
        for cycle in range(args.cycles):
            # -- CV Lab ---------------------------------------------------
            await ws.send(json.dumps({
                "type": "cv_lab_start", "experiment_id": args.experiment,
                "request_id": f"soak-{cycle}-a",
            }))
            reply = await _drain(ws, ("cv_lab_status", "cv_lab_error"))
            if reply["type"] != "cv_lab_status":
                findings.append(f"cycle {cycle}: cv_lab_start refused: {reply.get('reason')}")
            await ws.send(json.dumps({"type": "stream_start"}))
            seq, _ = await _feed(ws, frames, args.frames, seq)
            row = record(cycle, "cv_lab")
            # THE CV LAB IS STREAMING AND NOTHING ELSE MAY WAKE UP. This is
            # the exact defect the World Builder gate closed: before it, this
            # capture attached a builder and ran background global solves.
            nothing_unrelated(cycle, "cv_lab", row)
            tower_still_serving(cycle, "cv_lab")
            await ws.send(json.dumps({"type": "cv_lab_stop"}))
            await _drain(ws, ("cv_lab_status", "cv_lab_error"))
            await ws.send(json.dumps({"type": "stream_stop"}))
            await _quiet(ws)
            record(cycle, "cv_lab_left")

            # -- World Builder --------------------------------------------
            status, body = _post(base, f"{WB_SESSION}/start")
            if status != 200 or body.get("state") != "active":
                findings.append(f"cycle {cycle}: world_builder start answered {status} {body}")
            await ws.send(json.dumps({"type": "stream_start"}))
            seq, _ = await _feed(ws, frames, args.frames, seq)
            attached = await _await_until(
                lambda: builders_in(sample(tower_pid, world_root)), 20.0
            )
            if tower_pid is not None and not attached:
                findings.append(f"cycle {cycle}: no builder attached while World Builder was active")
            row = record(cycle, "world_builder", builder_attached=attached)
            nothing_unrelated(cycle, "world_builder", row, allow=("builder",))
            tower_still_serving(cycle, "world_builder")
            t0 = time.perf_counter()
            await ws.send(json.dumps({"type": "stream_stop"}))
            await _quiet(ws)
            finalized = await _await_until(
                lambda: not health_workers(base), args.finalize_timeout
            )
            finalize_s = round(time.perf_counter() - t0, 2)
            if not finalized:
                findings.append(
                    f"cycle {cycle}: the builder was still registered "
                    f"{args.finalize_timeout}s after Stop"
                )
            _post(base, f"{WB_SESSION}/stop")
            record(cycle, "world_builder_left", finalize_s=finalize_s, finalized=finalized)

            # -- Object Memory --------------------------------------------
            status, body = _post(base, f"{OM_SESSION}/start")
            if status != 200 or body.get("state") != "active":
                findings.append(f"cycle {cycle}: object_memory start answered {status} {body}")
            await ws.send(json.dumps({"type": "stream_start"}))
            seq, _ = await _feed(ws, frames, args.frames, seq)
            row = record(cycle, "object_memory")
            # The producer is allowed; a BUILDER here would mean the World
            # Builder gate did not close when its session stopped.
            nothing_unrelated(cycle, "object_memory", row, allow=("observer",))
            tower_still_serving(cycle, "object_memory")
            status, body = _post(base, f"{OM_SESSION}/pause")
            if body.get("state") != "paused":
                findings.append(f"cycle {cycle}: object_memory pause answered {body}")
            status, body = _post(base, f"{OM_SESSION}/resume")
            if body.get("state") != "active":
                findings.append(f"cycle {cycle}: object_memory resume answered {body}")
            await ws.send(json.dumps({"type": "stream_stop"}))
            await _quiet(ws)
            _post(base, f"{OM_SESSION}/stop")
            gone = await _await_until(
                lambda: not observers_in(sample(tower_pid, world_root)), 20.0
            )
            if tower_pid is not None and not gone:
                findings.append(
                    f"cycle {cycle}: an Object Memory producer outlived its session stop"
                )
            record(cycle, "object_memory_left", producer_gone=gone)

            # -- Document Memory ------------------------------------------
            before = _get(base, "/documents?limit=1")[1]
            before_count = (
                before.get("document_count") if isinstance(before, dict) else None
            )
            status, body = _post(base, f"{DOC_SESSION}/start")
            if status != 200:
                findings.append(f"cycle {cycle}: document start answered {status} {body}")
            running = await _await_until(
                lambda: (_get(base, DOC_SESSION)[1].get("session") or {}).get("state")
                in ("running", "paused"),
                args.model_timeout,
            )
            if not running:
                findings.append(
                    f"cycle {cycle}: the Document Memory session did not reach running "
                    f"within {args.model_timeout}s"
                )
            await ws.send(json.dumps({"type": "stream_start"}))
            seq, _ = await _feed(ws, frames, args.frames, seq)
            row = record(cycle, "document_memory", reached_running=running)
            nothing_unrelated(cycle, "document_memory", row, allow=("document",))
            tower_still_serving(cycle, "document_memory")
            await ws.send(json.dumps({"type": "stream_stop"}))
            await _quiet(ws)
            _post(base, f"{DOC_SESSION}/stop")
            stopped = await _await_until(
                lambda: (_get(base, DOC_SESSION)[1].get("session") or {}).get("state")
                == "stopped",
                30.0,
            )
            if not stopped:
                findings.append(f"cycle {cycle}: the Document Memory session did not stop")
            after = _get(base, "/documents?limit=1")[1]
            after_count = after.get("document_count") if isinstance(after, dict) else None
            # STOP KEEPS WHAT WAS RECORDED. Scene discards, Document does
            # not, and the asymmetry is the contract. Synthetic noise records
            # nothing, so this asserts the library did not go BACKWARDS --
            # which is what a Stop that wiped its store would look like.
            if (
                before_count is not None
                and after_count is not None
                and after_count < before_count
            ):
                findings.append(
                    f"cycle {cycle}: the document library shrank across a Stop "
                    f"({before_count} -> {after_count})"
                )
            record(cycle, "document_memory_left", stopped=stopped, documents=after_count)

            # -- Scene Understanding --------------------------------------
            # A stream with NO watcher must leave it stopped. This is the
            # defect the lane closed and the one a merge could silently undo,
            # because the watcher hooks run through the result channel.
            await ws.send(json.dumps({"type": "stream_start"}))
            seq, _ = await _feed(ws, frames, args.frames, seq)
            await _quiet(ws)
            state = _get(base, "/scene")[1]
            unwatched = (state.get("lifecycle") or {}).get("state")
            if unwatched in ("running", "starting"):
                findings.append(
                    f"cycle {cycle}: a stream with no watcher started Scene "
                    f"Understanding ({unwatched})"
                )
            record(cycle, "scene_unwatched", state=unwatched)

            reply = await _subscribe(
                ws, "scene_understanding", "live", f"soak-{cycle}-scene"
            )
            subscription_id = reply.get("subscription_id")
            if reply["type"] != "result_subscribed":
                findings.append(
                    f"cycle {cycle}: subscribing to scene_understanding failed: {reply}"
                )
            seq, _ = await _feed(ws, frames, args.frames, seq)
            watched = await _await_until(
                lambda: (_get(base, "/scene")[1].get("lifecycle") or {}).get("state")
                == "running",
                args.model_timeout,
            )
            if not watched:
                findings.append(
                    f"cycle {cycle}: Scene Understanding did not run with a stream and "
                    f"a watcher within {args.model_timeout}s"
                )
            row = record(cycle, "scene_watched", running=watched)
            nothing_unrelated(cycle, "scene_watched", row, allow=("scene",))
            tower_still_serving(cycle, "scene_watched")

            if subscription_id:
                await _unsubscribe(ws, subscription_id)
            await _quiet(ws)
            released = await _await_until(
                lambda: (_get(base, "/scene")[1].get("lifecycle") or {}).get("state")
                != "running",
                30.0,
            )
            if not released:
                findings.append(
                    f"cycle {cycle}: Scene Understanding kept running after the last "
                    "watcher left"
                )
            await ws.send(json.dumps({"type": "stream_stop"}))
            await _quiet(ws)
            record(cycle, "scene_left", released=released)

            # -- CV Lab again, on the same Tower --------------------------
            await ws.send(json.dumps({
                "type": "cv_lab_start", "experiment_id": args.experiment,
                "request_id": f"soak-{cycle}-z",
            }))
            reply = await _drain(ws, ("cv_lab_status", "cv_lab_error"))
            if reply["type"] != "cv_lab_status":
                findings.append(
                    f"cycle {cycle}: the CV Lab would not arm again after four other "
                    f"cartridges: {reply.get('reason')}"
                )
            await ws.send(json.dumps({"type": "stream_start"}))
            seq, _ = await _feed(ws, frames, args.frames, seq)
            row = record(cycle, "cv_lab_again")
            nothing_unrelated(cycle, "cv_lab_again", row)
            tower_still_serving(cycle, "cv_lab_again")
            await ws.send(json.dumps({"type": "cv_lab_stop"}))
            await _drain(ws, ("cv_lab_status", "cv_lab_error"))
            await ws.send(json.dumps({"type": "stream_stop"}))
            await _quiet(ws)
            record(cycle, "cycle_end")

    settled = _wait_until(lambda: not health_workers(base), args.finalize_timeout)
    final = record(args.cycles, "settled", settled=settled)
    for d in final["descendants"]:
        if d["kind"] != "other" or OBSERVATION_MARKER in d.get("cmdline", ""):
            findings.append(f"after the last cycle a worker is still alive: {d}")
    for s in final["strays"]:
        findings.append(f"after the last cycle a stray worker is alive: {s}")
    locks = final["locks"]
    if locks["dead_locks"]:
        findings.append(f"locks naming dead pids: {locks['dead_locks']}")
    if locks["open_sessions"]:
        findings.append(f"sessions left open: {locks['open_sessions']}")
    return {"rows": rows, "final": final}


def judge(rows: list[dict], cycles: int) -> dict:
    """The last third of cycles against the first, at the same point in each.

    `cycle_end` is the sample every cycle ends on, with every cartridge left
    and nothing streaming, so a difference between two of them is growth
    rather than phase.
    """
    settled = [r for r in rows if r["step"] == "cycle_end"]
    if len(settled) < 3:
        return {"verdict": "TOO SHORT", "detail": f"{len(settled)} settled samples"}
    third = max(1, len(settled) // 3)
    first, last = settled[:third], settled[-third:]

    def mean(sequence, key):
        values = [r[key] for r in sequence if r.get(key) is not None]
        return round(sum(values) / len(values), 1) if values else None

    out = {}
    for key in ("threads", "rss_mb", "handles"):
        a, b = mean(first, key), mean(last, key)
        out[key] = {"first": a, "last": b, "delta": None if a is None or b is None else round(b - a, 1)}
    return out


def render(report: dict) -> str:
    lines = ["", "== all-cartridge switch soak =="]
    growth = report["growth"]
    for key, row in growth.items():
        if isinstance(row, dict) and "delta" in row:
            lines.append(f"  {key:<10} {row['first']} -> {row['last']}  (delta {row['delta']})")
    final = report["result"]["final"]
    lines.append(f"  workers left: {len(final['descendants'])}  strays: {len(final['strays'])}")
    lines.append(f"  locks: {final['locks']}")
    if report["findings"]:
        lines.append("")
        lines.append(f"  {len(report['findings'])} FINDINGS:")
        for f in report["findings"]:
            lines.append(f"    - {f}")
        lines.append("")
        lines.append("  VERDICT: FAILED")
    else:
        lines.append("")
        lines.append("  VERDICT: STABLE -- every cycle held, nothing leaked")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--experiment", default="baseline")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--finalize-timeout", type=float, default=120.0)
    parser.add_argument(
        "--model-timeout",
        type=float,
        default=90.0,
        help="how long a cartridge that loads a model may take to reach running",
    )
    parser.add_argument("--root", type=artifact_root_arg, default=str(DEFAULT_ROOT))
    parser.add_argument("--json", default=None, help="write the full report here")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    args.port = args.port or _free_port()
    base = f"http://{args.host}:{args.port}"
    log_path = root / "tower.log"

    findings: list[str] = []
    process = start_tower(args.port, root, log_path)
    try:
        wait_for_health(base, process, log_path)
        tower_pid = process.pid
        result = asyncio.run(soak(args, base, tower_pid, root, findings))
    finally:
        if not stop_tower(process):
            findings.append("the Tower did not stop when asked")

    # After the Tower is down: nothing it started may still be running.
    # A WORKER, not merely a process that mentions the root. The first
    # version matched on the root path alone and reported THIS SCRIPT, whose
    # own argv carries `--root`, as a leaked worker. A leftover is a process
    # running one of the worker entry points against this root.
    markers = ("world_build_session", "world_solve", OBSERVATION_MARKER)
    leftovers = []
    mine = {process.pid, __import__("os").getpid()}
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = " ".join(proc.info["cmdline"] or [])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if proc.info["pid"] in mine:
            continue
        if str(root) in cmdline and any(m in cmdline for m in markers):
            leftovers.append({"pid": proc.info["pid"], "cmdline": cmdline[-120:]})
    for leftover in leftovers:
        findings.append(f"a worker outlived the Tower: {leftover}")

    report = {
        "args": vars(args) | {"root": str(root)},
        "result": result,
        "growth": judge(result["rows"], args.cycles),
        "findings": findings,
        "leftovers": leftovers,
    }
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(render(report))
    print(f"\n  the Tower's log: {log_path}")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
