"""THE RESEARCH BYPASS: build from the original local capture, not the
redacted keyframes. Off by default, and never privacy-safe.

WHY THIS EXISTS. The saved world's appearance normally comes from the
wearer's REDACTED keyframes (`appearance.py`, contract
`WORLD-BUILDER-APPEARANCE.md`). On the canonical capture that costs a great
deal of the picture before any reconstruction runs: measured over its 398
keyframes, 182 of them carry a solid fill, covering 27.5% of those frames on
average and 93.8% at the worst, and the cross-frame consensus then withdraws
what one frame hid from every other frame as well. The owner asked, on
2026-09-21, for a path that proves what reconstruction quality the ORIGINAL
local imagery supports, with the privacy implementation left in place and
re-enablable rather than deleted.

SO THIS MODULE IS THE WHOLE BYPASS, AND IT IS THE ONLY PLACE THAT NAMES THE
RAW FRAMES. `appearance.py` keeps its one provenance function and its static
boundary test; the bypass enters it through a single branch on
`AppearanceParams.imagery_source`, which is `"redacted"` unless something
asks for otherwise. Nothing here changes what the redacted path does.

WHAT IT IS NOT. An artifact built this way is NOT privacy-safe and must never
be mistaken for one. Every build in this mode is labelled
`raw-local-research/no-redaction` in its manifest, its appearance provenance,
its params digest and the header it is served under, and the page draws a
visible marker. `appearance_pipeline.refuse_mismatched_imagery` is the reader
side: a consumer that expects redacted appearance refuses a raw artifact, and
a consumer that asked for raw refuses a redacted one, so no cache and no page
can quietly hold both.

LOCAL ONLY. The raw frames are read from the capture directory on this
machine, through `global_solve.resolve_source_path`, and nothing is copied
out of it: the bytes go into the same texture chunks under the world's own
`appearance/<session>/` directory that a redacted build writes, served over
the same local Tower->phone route. There is no new persistence and no new
transmission.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

# `params.imagery_source`, and the values it may take.
IMAGERY_REDACTED = "redacted"
IMAGERY_RAW = "raw-local-research"
IMAGERY_SOURCES = (IMAGERY_REDACTED, IMAGERY_RAW)

# The environment flag. Absent, empty, "0", "false", "no" and "off" all mean
# the default -- redacted. Anything else is read as a request for raw, and
# that request is logged wherever it is honoured.
RAW_IMAGERY_ENV = "TOWER_WORLD_RAW_IMAGERY"
_FALSE = frozenset({"", "0", "false", "no", "off", "none"})

# The label an artifact built in this mode carries EVERYWHERE: the manifest,
# the appearance provenance, the served header and the page's caption. It is
# not a redaction label, and `appearance.label_is_trusted` does not admit it:
# no stage may read it as "these pixels were redacted".
RAW_LABEL = "raw-local-research/no-redaction"

# What `appearance_provenance.source` says instead of `session-keyframes`.
SOURCE_RAW_LOCAL = "raw-local-capture"

# What replaces `unobserved_rule` when no privacy mask is applied at all.
RAW_UNOBSERVED_RULE = "none|raw-local-research"

# The one sentence a reader gets. Short enough to fit a header value.
RAW_NOTE = ("RESEARCH BUILD: original unredacted local capture frames; faces "
            "and screens are NOT redacted; not privacy-safe; do not publish")


def is_raw(imagery_source) -> bool:
    return str(imagery_source) == IMAGERY_RAW


def normalise(imagery_source) -> str:
    value = str(imagery_source)
    if value not in IMAGERY_SOURCES:
        raise ValueError(f"unknown imagery source {imagery_source!r}; "
                         f"one of {IMAGERY_SOURCES}")
    return value


def imagery_source_from_env(environ=None) -> str:
    """The default imagery source for a process. `redacted` unless the flag
    is set to something that is not a falsehood.

    Read at a CLI or engine boundary, never inside `AppearanceParams`: the
    dataclass default is `redacted` always, so a params object built in a
    test or a library call never depends on the ambient environment.
    """
    import os  # noqa: PLC0415

    if environ is None:
        environ = os.environ
    value = str(environ.get(RAW_IMAGERY_ENV, "")).strip().lower()
    return IMAGERY_REDACTED if value in _FALSE else IMAGERY_RAW


class RawImageryUnavailable(RuntimeError):
    """Raw mode was asked for and the original frames cannot be found."""


@dataclass(frozen=True)
class RawKeyframeImages:
    """Where each keyframe's ORIGINAL local frame is, resolved once.

    `paths` is keyframe_id -> absolute path, already resolved against the
    Tower root (or `TOWER_SOURCES_ROOT`) rather than the process cwd -- the
    bug `global_solve.resolve_source_path` exists to close.
    """

    paths: dict
    recorded: dict
    missing: tuple
    digest: str

    @property
    def cache_token(self) -> str:
        """Names this exact set of original frames, and says out loud that it
        is not a redacted set. Goes wherever `KeyframeImageSet.cache_token`
        goes, so nothing cached for a redacted build is reused for this one."""
        return f"{IMAGERY_RAW}@{self.digest}"

    def path(self, keyframe_id: str):
        return self.paths.get(keyframe_id)

    def read(self, keyframe_id: str) -> bytes | None:
        path = self.paths.get(keyframe_id)
        if path is None:
            return None
        try:
            return Path(path).read_bytes()
        except OSError:
            return None


def resolve_raw_keyframes(store, world_id: str, session_id: str,
                          keyframe_ids=None, tower_root=None) -> RawKeyframeImages:
    """The original local frame behind each keyframe, or a refusal.

    The mapping is the solve workspace's `sources.json`, which the builder
    that observed the frames wrote; a replay stages frames under enumeration
    indices, so the file name alone does not find them.
    """
    from tower.world_builder.global_solve import (  # noqa: PLC0415
        read_sources,
        resolve_source_path,
        workspace_for,
    )

    recorded = read_sources(workspace_for(store, world_id, session_id))
    if not recorded:
        raise RawImageryUnavailable(
            "this session has no record of where its original frames live, so "
            "raw-local-research imagery cannot be built for it")
    wanted = list(keyframe_ids) if keyframe_ids is not None else sorted(recorded)
    paths: dict = {}
    missing: list = []
    for kid in wanted:
        resolved = resolve_source_path(recorded.get(kid), tower_root)
        if resolved is None or not Path(resolved).is_file():
            missing.append(kid)
            continue
        paths[kid] = Path(resolved)
    if not paths:
        raise RawImageryUnavailable(
            f"none of this session's {len(wanted)} original frames is on this "
            "machine; raw-local-research imagery cannot be built")
    h = hashlib.sha1()
    for kid in sorted(paths):
        p = Path(paths[kid])
        try:
            size = p.stat().st_size
        except OSError:
            size = -1
        h.update(f"{kid}\t{p.name}\t{size}\n".encode())
    return RawKeyframeImages(paths=paths, recorded=dict(recorded),
                             missing=tuple(sorted(missing)), digest=h.hexdigest()[:16])
