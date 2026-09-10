"""Filesystem primitives shared by every module that persists to disk.

Extracted once a second module needed them. Both object memory and World
Builder had independently converged on the same three operations, and the
capture recorder -- which is shared transport infrastructure -- was
importing them from inside a cartridge.

The atomic-write pattern in particular is not incidental: the try/finally
exists because a failure mid-write once left a temp file holding a live
copy of data that nothing read, pruned, or deleted. That defect shipped
before it was fixed, which is why it lives in one place now.
"""

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import BinaryIO, Callable

logger = logging.getLogger(__name__)

TEMP_SUFFIX = ".tmp"


def staging_path(path: Path) -> Path:
    """A staging name no other writer can be using.

    Public, because two modules need the one convention: this file's two
    atomic writers and `world_builder/store.py`'s keyframe images. A second
    spelling of it is a second chance to get it wrong.

    `path.name + ".tmp"` was the convention here until an adversarial review
    measured what it does with two writers of one destination: 656 torn reads
    at the published path over 12 seconds, plus a writer killed by
    `FileNotFoundError` out of `replace` because its peer's `finally` had
    already unlinked the shared temp. One writer can also `replace` the
    half-written temp the OTHER is still filling straight onto the
    destination -- which is precisely the `BadZipFile` the atomic write was
    introduced to prevent, reintroduced by the staging name.

    Nothing in this Tower serialises writers of `solution.npz`:
    `acquire_writer_lock` is per-world and taken only by the engine, and
    `scripts/world_solve.py` takes no lock at all. A hand-run solve against a
    world with a live builder is two writers, and that is an ordinary
    operator action -- it is how the 2026-09-09 artifact was recovered.

    pid plus uuid4: the pid makes a stray temp attributable when someone
    finds one, and the uuid makes it unique even within one process.
    """
    return path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}{TEMP_SUFFIX}")


def new_id() -> str:
    """Mint an opaque identifier.

    uuid4 hex, never derived from a display name (users rename things) and
    never a content hash (refinement changes content). Anything that
    references an id must survive both.
    """
    return uuid.uuid4().hex


# How long a writer will keep trying to replace a destination a reader
# has open, and how it paces the attempts. Bounded: this exists to ride
# out a reader, not to wait out a process that has parked on the file. A
# writer that cannot win in this budget raises, exactly as it did before.
#
# Until 2026-09-06 the budget was 12 x 5 ms = 60 ms, sized for "a
# reader's sub-millisecond handle". That is the handle's length on an
# idle box. On a live walk with the global solver's background child
# taking 18 of 20 cores, the Tower's web thread reading a 2.2 MB
# points.json for the phone lost the CPU mid-read, held the file past
# 60 ms, and the BUILDER -- the process the retry exists to protect --
# died in `write_derived` with poses.json new and points.json old. Two
# seconds is longer than any reader this Tower has that is still making
# progress, and still short next to a rebuild cadence of one per second.
REPLACE_BUDGET_S = 2.0
REPLACE_BACKOFF_S = 0.005
REPLACE_BACKOFF_MAX_S = 0.05


def sweep_abandoned_staging(directory: Path) -> int:
    """Remove staging files whose writer is gone. Returns how many.

    UNIQUE NAMES FIXED ONE PROBLEM AND CREATED ANOTHER. A shared
    `<name>.tmp` meant a killed writer left one file that the next
    successful write's `finally` cleaned up. Unique names mean a killed
    writer leaves one file EVERY TIME, and nothing removes it: the
    `finally` does not run under `TerminateProcess`, and `purge_world` --
    which the docstrings call the sweeper -- has no production caller at
    all. An adversarial review measured 21 strays and 11.4 MB after six
    hard kills, and the builder kills a solve child on every stop that
    outstays its budget.

    The pid in the name is what makes this safe: a staging file belonging
    to a LIVE process is someone's write in flight and is left alone. Only
    a dead writer's leavings are swept, so this can run at any time.
    """
    swept = 0
    for candidate in directory.iterdir():
        if not candidate.is_file():
            continue
        parts = candidate.name.split(".")
        # `.tmp` as a COMPONENT, not as a suffix. `staging_path` puts it
        # last for a JSON or npz write, but `prepare_images` stages an
        # undistorted frame as `<stem>.<pid>.<uuid>.tmp.jpg` -- so a
        # suffix match swept the writers that rarely die and missed the
        # one the builder terminates on every over-long stop, in the very
        # directory this claims to sweep. Caught by testing the sweeper
        # against every staging shape rather than the one it was written
        # against.
        if TEMP_SUFFIX.lstrip(".") not in parts:
            continue
        pid = None
        for part in parts:
            if part.isdigit():
                pid = int(part)
        if pid is None:
            # A staging name from before they carried a pid, or something
            # that merely looks like one. Not ours to judge, and deleting
            # an unattributable file is not a sweep.
            continue
        try:
            import psutil

            if psutil.pid_exists(pid):
                continue
        except Exception:  # noqa: BLE001 -- absence of psutil is not a reason to delete
            continue
        try:
            candidate.unlink()
        except OSError:
            continue
        swept += 1
    if swept:
        logger.info(
            "storage: swept %s abandoned staging file(s) from %s", swept, directory
        )
    return swept


def write_json_atomic(path: Path, payload: dict) -> None:
    """Replace `path` atomically, leaving no temp file behind either way."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = staging_path(path)
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            # `json.dumps(...)` then one write, NOT `json.dump(payload,
            # handle)`. The streaming form calls handle.write() once per
            # token, and every one of those crosses TextIOWrapper's
            # encode-and-buffer path; building the string once and writing
            # it once measured **3.9x faster** (23.97 ms -> 6.18 ms on a
            # representative payload) for **byte-identical** output. Same
            # encoder, same defaults, so the file on disk does not change.
            #
            # The cost is that the whole document is materialised first,
            # and the peak is worse than "one extra copy": MEASURED at
            # 750k points, peak went 71.5 MB -> 167 MB, roughly 1.4x the
            # document on top of it, because the string and its UTF-8
            # encoding coexist.
            #
            # Negligible at the sizes this Tower writes -- the largest JSON
            # it has ever persisted is a 1.71 MB points.json against
            # ~184 MB RSS. But it scales with payload, so a caller writing
            # something an order of magnitude larger should revisit this
            # rather than inherit it.
            #
            # Deliberately NOT orjson, which was measured and refused
            # because ITS BYTES DIFFER -- separators, and `1e-07` against
            # `1e-7`. That alone disqualifies it here.
            #
            # An earlier version of this comment also claimed orjson would
            # "defeat the allow_nan=False guard". CORRECTED: exactly one
            # caller has that guard (`world_builder/store.py:463`,
            # placements) and it runs `json.dumps(payload,
            # allow_nan=False)` as a SEPARATE validation before calling
            # this function -- so changing the encoder in here could not
            # have defeated it. The byte difference is the real reason and
            # it stands on its own.
            handle.write(json.dumps(payload))
            handle.flush()
            os.fsync(handle.fileno())
        replace_with_retry(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def replace_with_retry(temp_path: Path, path: Path) -> None:
    """os.replace, retried while a concurrent READER holds the destination.

    Public, because every writer of a file a reader can hold needs it, not
    only the JSON ones. `prepare_images` undistorts keyframes with a bare
    `os.replace` and raised `PermissionError` the first time a test held one
    of its outputs open -- which is the same WinError 5 this function was
    written for, in a path that had never been tested.

    Windows refuses `replace()` onto a destination any handle has open,
    and -- measured, not assumed -- `FILE_SHARE_DELETE` does NOT lift
    that: `MOVEFILE_REPLACE_EXISTING` fails with WinError 5 even against
    a share-delete handle. So a reader cannot make itself harmless, and
    the tolerance has to live here.

    Until 2026-08-23 this store's docstring could say "V1 also has no
    concurrent reader -- capture, build and inspect are separate
    processes". The Tower->iOS result channel is that reader: the web
    process now polls world state while a build session writes it. That
    assumption is void, and without this retry the consequence lands on
    the WRITER -- a status channel would crash the mapping session it
    exists to report on.

    Measured on this host, 400 atomic writes against a reader looping as
    fast as it can (a far harsher case than the channel's 2 Hz poll):

        no reader                        0 / 400 failed
        reader, no retry               223 / 400 failed   (55.8%)
        reader, this retry               0 / 400 failed

    The retried run was the harsher of the two: it completed 68,455
    reader opens against the 8,648 of the failing run, because a writer
    that is not erroring out early leaves the reader more time to run.

    Retrying a rename is safe in a way that retrying most IO is not: the
    operation is atomic, so it either happened or it did not. There is no
    partial state to reconcile and no possibility of writing twice.
    """
    deadline = time.monotonic() + REPLACE_BUDGET_S
    backoff = REPLACE_BACKOFF_S
    while True:
        try:
            temp_path.replace(path)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(backoff)
            backoff = min(backoff * 2, REPLACE_BACKOFF_MAX_S)


def write_bytes_atomic(path: Path, write: "Callable[[BinaryIO], None]") -> None:
    """Publish a binary artifact atomically: `write` fills a temp file, and
    the destination is replaced only once the bytes are whole and on disk.

    The binary twin of `write_json_atomic`, and it exists for a measured
    failure rather than for symmetry. `world_builder/global_solve.py` wrote
    `solution.npz` by handing the FINAL path straight to
    `np.savez_compressed`, while `solution.json` beside it went through the
    atomic helper. A `.npz` is a zip, a zip is only a zip once its central
    directory is written last, and the reader lives in a DIFFERENT PROCESS
    -- the builder rebuilding the derived tree while the solver child it
    launched is still writing.

    On the 2026-09-09 walk that reader opened the file mid-write and got
    `BadZipFile: File is not a zip file`, which ended a session holding 795
    keyframes and 26,634 points. The same window is worse when the writer
    is TERMINATED rather than merely slow (`BackgroundSolver.wait` kills a
    child that outstays a stop): the torn file then persists, and every
    later read of that world fails the same way.

    Replacing a whole temp file closes both cases at once: a reader sees
    the previous solution or the next one, never half of either.

    IT DOES NOT CLEAN UP AFTER A KILL, and an earlier version of this
    docstring claimed it did. The `finally` below runs on an exception; it
    does not run on `TerminateProcess`, which is exactly how
    `BackgroundSolver` ends a solve child that outstays a stop. Measured:
    a real kill mid-write leaves the published archive VALID -- which is
    the guarantee that matters -- and a stray `.tmp` beside it that
    nothing prunes. `purge_world` is still the only sweeper.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = staging_path(path)
    try:
        with temp_path.open("wb") as handle:
            write(handle)
            handle.flush()
            # fsync before the replace, not after. The replace is what
            # publishes; bytes still sitting in the OS cache at that
            # moment are bytes a crash can take with the rename already
            # visible, which is the one ordering that produces a file
            # that IS published and IS torn.
            os.fsync(handle.fileno())
        replace_with_retry(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def read_bytes_closed(path: Path) -> bytes:
    """Read a binary artifact whole, with the handle closed before use.

    The binary twin of `read_json_closed`, and for the same Windows
    reason: a handle held across parsing is a handle that blocks a
    writer's `os.replace` (WinError 5), so a reader that parses lazily
    turns itself into the thing the write path has to retry around.

    `np.load` on a path is exactly that lazy reader -- `NpzFile` keeps the
    zip open until it is closed -- so callers hand these bytes to
    `np.load(io.BytesIO(...))` instead. The arrays this Tower persists are
    single-digit MB; the field session's `solution.npz` is 1.7 MB, and the
    extra peak is one copy of the compressed file.

    HOW MUCH THIS BUYS, HONESTLY: measured on that artifact, the lazy form
    holds the handle 23.6-31.7 ms and the eager form 0.9-1.2 ms, against a
    `REPLACE_BUDGET_S` of 2000 ms. So the retry already absorbed the lazy
    reader comfortably and this is a margin, not a rescue. It is still the
    right shape -- the budget is finite and a reader under solver load is
    exactly what descheduled long enough to matter on 2026-09-06 -- but it
    was oversold as closing a hazard, and it is not what fixed the field
    failure. The atomic write is.
    """
    with path.open("rb") as handle:
        return handle.read()


def read_json_closed(path: Path) -> dict:
    """Read and parse with the handle closed before parsing.

    Windows cannot os.replace onto an open destination (verified:
    WinError 5), so a reader holding a handle across other work can block
    a writer. Closing first makes that structurally impossible.
    """
    with path.open("r", encoding="utf-8") as handle:
        text = handle.read()
    return json.loads(text)


def append_jsonl(path: Path, payload: dict) -> None:
    """Append one record, and heal a torn line rather than compounding it.

    The newline check is not defensive noise. An interrupted write leaves a
    partial line with NO trailing newline, and a plain append then glues
    the next record onto the end of it -- so ONE crash destroys TWO
    records: the torn one, which is expected, and the next good one, which
    is not. `read_raw_jsonl` drops the fused line as corrupt, and the
    caller never learns the second record existed.

    Starting the new record on its own line confines the damage to the
    write that was actually interrupted. Two opens instead of one; an
    append is never the hot loop in this codebase, since even the capture
    recorder does one per delivered frame.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_newline = False
    if path.exists() and path.stat().st_size > 0:
        with path.open("rb") as handle:
            handle.seek(-1, os.SEEK_END)
            needs_newline = handle.read(1) != b"\n"
    with path.open("a", encoding="utf-8") as handle:
        if needs_newline:
            handle.write("\n")
        handle.write(json.dumps(payload) + "\n")


def read_raw_jsonl(path: Path) -> tuple[list[dict], int]:
    """Parse a journal, counting genuinely-corrupt lines separately.

    A line that is not valid JSON is corruption. A well-formed line whose
    fields the caller's schema cannot interpret is NOT corruption, and
    callers treat the two differently. A torn final line from an
    interrupted write lands in the first category and is skipped without
    touching the file.
    """
    if not path.exists():
        return [], 0
    raw_records: list[dict] = []
    corrupt = 0
    # errors="replace" rather than the default strict: a write interrupted
    # mid-codepoint leaves an invalid byte sequence, and a UnicodeDecodeError
    # raised from the file iterator would take out the WHOLE journal rather
    # than the one torn line -- exactly the failure this function promises
    # not to have. Not reachable through this module's own writes today
    # (json.dumps defaults to ensure_ascii, so every byte written is ASCII
    # and a tear can only fall on a character boundary), but the promise
    # should not depend on that staying true.
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw_records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("skipping corrupt line at %s:%s", path, line_number)
                corrupt += 1
    return raw_records, corrupt
