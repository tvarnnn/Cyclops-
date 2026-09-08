"""Serving a dense reconstruction to a browser, and to the phone.

WHY THE PAYLOAD IS INLINE

`GET /worlds/{id}/render` returns HTML under

    Content-Security-Policy: default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'

so in a BROWSER `fetch` and `XMLHttpRequest` are blocked outright by the
absent `connect-src`, and a page that fetched its points beside itself would
simply not work.

THE PHONE DOES NOT SEE THAT HEADER, and an earlier version of this note
claimed it did. iOS fetches the page with `URLSession`, keeps the body, drops
the response headers, and hands the string to
`WKWebView.loadHTMLString(_:baseURL:nil)` (`WorldRenderViewer.swift`). No CSP
applies there at all. What blocks a network fetch on the phone is the
navigation delegate refusing every navigation but the first, plus the null
`baseURL` that leaves relative URLs unresolvable -- a different mechanism with
the same effect, and one this module does not control.

Two consequences, both deliberate:

* Inline is still right. It is the only shape that works in BOTH clients, and
  it needs no second request, no relaxed policy and no new route.
* `js_object_literal`'s escaping is the SOLE defence against a `</script>` in
  the config, not the second one. There is no browser-enforced policy behind
  it on the device the product ships on. Treat it accordingly.

WHY THERE IS A BYTE BUDGET AND NOT JUST AN LOD

The LOD ladder is a fraction of each scene's own median depth, so its point
count follows the SIZE of the room, not the size of the payload. Measured over
five real worlds the mobile level ranges from 39k points to 1.45 M -- 0.6 MB to
23 MB, which is 31 MB of base64 for the largest. A phone must not be handed
that, and the existing 2-D viewer has already been seen killing the WebKit
content process. So the page enforces a byte budget, and when a level exceeds
it the points are thinned by CONFIDENCE FIRST: the geometry several cameras
agreed on survives, and the weakest goes first.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path

import numpy as np

from tower.world_builder.dense import (
    POINT_STRIDE_BYTES,
    read_points_bin,
    voxel_reduce,
)
from tower.world_builder.dense_pipeline import (
    dense_currency,
    dense_dir,
    read_dense_manifest,
)

logger = logging.getLogger(__name__)

# The page is one HTML string, so this is the whole transfer. 6 MB of binary is
# about 8 MB of base64; the existing sparse page has been observed killing the
# WKWebView content process well below that in point COUNT, and a GPU buffer is
# far cheaper per point than the 2-D canvas it used.
MOBILE_BYTE_BUDGET = 6 * 1024 * 1024
DESKTOP_BYTE_BUDGET = 48 * 1024 * 1024

VIEWER_FILENAME = "dense_viewer.html"

# What the page template must contain. Kept as constants so a mismatch is a
# test failure rather than a blank page.
TOKEN_CONFIG = "__WB_CONFIG__"
TOKEN_POINTS = "__WB_POINTS_B64__"


# What the picture IS, in one sentence, always shown. `WORLD-BUILDER-WORLDS.md`
# requires the render page to carry it, and rule 2 says the page never claims
# more than it does. The dense page is now what that route serves, so the
# obligation is this module's. The sparse page's sentence would be a lie here
# and this one would be a lie there, so they are different sentences with the
# same job: name the thing, and refuse the word "scan".
CAPTION = ("Dense reconstruction: per-pixel depth from a neural network, "
           "anchored to the structure-from-motion solve and kept only where "
           "several cameras agreed. Not a surface, not a mesh, not metric scale.")
CAPTION_BEHIND_SOLVE = ("This reconstruction is BEHIND the world: it was built "
                        "from an earlier solve, and the world has been solved "
                        "again since.")
CAPTION_BEHIND_KEYFRAMES = ("This picture is BEHIND the newest keyframes: the "
                            "Tower has accepted keyframes it has not yet built "
                            "into geometry.")


class DenseViewerUnavailable(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def viewer_template_path() -> Path:
    return Path(__file__).resolve().parent / VIEWER_FILENAME


def has_dense(store, world_id: str, session_id: str) -> bool:
    return read_dense_manifest(store, world_id, session_id) is not None


def thin_to_budget(X, C, F, budget_bytes: int, *, voxel_hint: float | None = None):
    """Make the buffer fit by making the picture COARSER, not smaller.

    This used to sort by confidence and keep the best N, which sounds honest and
    is not. Confidence is not distributed evenly through a room: it is high
    where the wearer stood still and low at the far end of every space they
    walked past once. A global confidence threshold therefore does not thin the
    room, it DELETES the parts of it that were seen from fewer angles. Measured
    on the three-room chain, whose mobile level is 2.6 M points against a
    393 k budget: the old code shipped `confidence >= 9`, kept 15% of the cloud,
    and what reached the phone was a scatter of isolated wall and ceiling slabs
    with two of the three rooms simply gone. The caption said "some gaps are
    thinning", which was true and no help -- the wearer opens their room and
    does not recognise it.

    So the budget is met the same way the LOD ladder meets it: a coarser voxel
    grid over the WHOLE extent, which keeps every part of the room and lowers
    the density everywhere equally. `voxel_reduce` keeps the best confidence in
    each cell, so the confidence channel still means what it meant.

    The voxel size is solved for rather than searched. These points lie on
    surfaces, so their count scales as roughly `voxel ** -2`; one step of that
    law lands within a few percent and a second corrects it. Two passes cost
    about 0.9 s on 2.6 M points, which is why this is done here, on request,
    rather than baked into a ladder that cannot know the client's budget.

    Any residue after four passes -- overshoot from a scene that is not
    surface-like -- is trimmed by confidence, which at a few percent is the
    thing that trim is actually safe for.

    Returns (X, C, F, thinned_to_confidence). The last value stays non-None
    whenever anything was dropped, because the page has to say so.
    """
    max_points = max(1, budget_bytes // POINT_STRIDE_BYTES)
    if len(X) <= max_points:
        return X, C, F, None

    span = float(np.max(X.max(axis=0) - X.min(axis=0)))
    if not np.isfinite(span) or span <= 0:
        span = 1.0
    # A starting cell: the level's own voxel when the caller knows it, else the
    # extent divided by the cube root of the count, which is the right order of
    # magnitude for anything.
    voxel = float(voxel_hint) if voxel_hint else span / max(len(X) ** (1 / 3), 1.0)

    # Aim slightly under the budget. The scaling law is good to a few percent,
    # and landing 3% over costs a whole extra pass -- or, worse, falls through
    # to the confidence trim this exists to avoid.
    target = max(1.0, max_points * 0.95)
    Xr, Cr, Fr = X, C, F
    for _ in range(6):
        voxel *= float(np.sqrt(len(Xr) / target))
        if not np.isfinite(voxel) or voxel <= 0:
            break
        Xr, Cr, Fr = voxel_reduce(X, C, F, voxel)
        if len(Xr) <= max_points:
            break

    if len(Xr) > max_points:
        # Did not converge. Fall back rather than blow the budget, and say so:
        # this is the branch whose spatial bias the docstring warns about.
        logger.warning(
            "[Tower][WorldBuilder][dense] voxel thinning did not reach the "
            "budget (%d > %d); trimming by confidence",
            len(Xr), max_points,
        )
        order = np.argsort(-Fr.astype(np.int32), kind="stable")[:max_points]
        order.sort()
        Xr, Cr, Fr = Xr[order], Cr[order], Fr[order]

    return Xr, Cr, Fr, int(Fr.min()) if len(Fr) else 0


MAX_VIEWPOINTS = 240


def capture_viewpoints(store, world_id: str, session_id: str, component: int = 0):
    """Where the wearer actually stood, and which way they looked.

    A room reconstructed from the inside looks its worst from the outside: you
    see the backs of walls, through every hole, with the furniture hidden behind
    them. Opening the viewer at an arbitrary orbit distance is therefore the
    least flattering possible first impression of a reconstruction that is
    perfectly good from where it was captured.

    So the page opens where a camera stood, looking where it looked. Returned
    as flat [cx, cy, cz, fx, fy, fz] rows, subsampled, in capture order, so the
    viewer can also step through them.
    """
    try:
        from tower.world_builder.global_solve import load_solution  # noqa: PLC0415

        solution = load_solution(store, world_id, session_id)
    except Exception:  # noqa: BLE001 -- a viewpoint list is a nicety, never a blocker
        return []
    if solution is None:
        return []
    rows = []
    for kid in solution.keyframe_ids:
        pose = (solution.poses or {}).get(kid)
        if not pose or pose.get("rotation") is None or pose.get("translation") is None:
            continue
        if int(pose.get("component", 0)) != component:
            continue
        R = np.asarray(pose["rotation"], float).reshape(3, 3)
        t = np.asarray(pose["translation"], float)
        centre = -R.T @ t
        forward = R.T @ np.array([0.0, 0.0, 1.0])   # the optical axis, in world
        rows.append([float(v) for v in centre] + [float(v) for v in forward])
    if len(rows) > MAX_VIEWPOINTS:
        step = len(rows) / MAX_VIEWPOINTS
        rows = [rows[int(i * step)] for i in range(MAX_VIEWPOINTS)]
    return rows


def build_dense_payload(store, world_id: str, session_id: str, *,
                        budget_bytes: int = MOBILE_BYTE_BUDGET,
                        level: int | None = None):
    """The buffer a page embeds, plus what a viewer needs to describe it."""
    manifest = read_dense_manifest(store, world_id, session_id)
    if manifest is None:
        raise DenseViewerUnavailable("this session has no dense reconstruction")
    levels = manifest.get("levels") or []
    if not levels:
        raise DenseViewerUnavailable("the dense manifest lists no levels")
    idx = level if level is not None else manifest.get("mobile_level", len(levels) - 1)
    idx = max(0, min(int(idx), len(levels) - 1))
    path = dense_dir(store, world_id, session_id) / f"points_l{idx}.bin"
    if not path.exists():
        raise DenseViewerUnavailable(f"dense level {idx} is missing")
    X, C, F = read_points_bin(path)
    if len(X) == 0:
        raise DenseViewerUnavailable("the dense level holds no points")
    X, C, F, min_conf = thin_to_budget(
        X, C, F, budget_bytes, voxel_hint=levels[idx].get("voxel"))

    buf = np.zeros(len(X), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                  ("r", "u1"), ("g", "u1"), ("b", "u1"), ("c", "u1")])
    buf["x"], buf["y"], buf["z"] = X[:, 0], X[:, 1], X[:, 2]
    buf["r"], buf["g"], buf["b"] = C[:, 0], C[:, 1], C[:, 2]
    buf["c"] = F
    raw = buf.tobytes()

    centre = X.mean(0)
    viewpoints = capture_viewpoints(store, world_id, session_id,
                                    (manifest.get("params") or {}).get("component", 0))
    currency = dense_currency(store, world_id, session_id, manifest)
    behind = []
    if currency.get("solve_current") is False:
        behind.append(CAPTION_BEHIND_SOLVE)
    if currency.get("derived_current") is False:
        behind.append(CAPTION_BEHIND_KEYFRAMES)
    config = {
        "caption": CAPTION,
        "caption_behind": behind,
        "viewpoints": viewpoints,
        "world_id": world_id,
        "session_id": session_id,
        "points": int(len(X)),
        "stride": POINT_STRIDE_BYTES,
        "level": idx,
        "level_voxel": levels[idx].get("voxel"),
        "thinned_to_confidence": min_conf,
        "source_points": int(levels[idx].get("points") or 0),
        "bbox_min": manifest.get("bbox_min"),
        "bbox_max": manifest.get("bbox_max"),
        "centre": [float(v) for v in centre],
        "median_scene_depth": manifest.get("median_scene_depth"),
        "confidence_range": [int(F.min()), int(F.max())],
        # Camera axes are OpenCV's: x right, y DOWN, z forward, and the world's
        # up_axis is unknown. A viewer must not assume +y or +z is up.
        "up_axis": "unknown",
        "screen_up": [0, -1, 0],
        # Repeated from the manifest. The dense stage makes no scale claim the
        # sparse solve did not already make, and a viewer must not print metres.
        "scale": manifest.get("scale"),
    }
    return raw, config, manifest



def js_object_literal(obj) -> str:
    """JSON, safe to paste into a <script> body.

    The config becomes a JS source literal inside the page, so three
    characters that are legal in JSON are not legal there: LESS-THAN can
    end the script tag, and U+2028 and U+2029 are line terminators in
    JavaScript but ordinary characters in JSON. Each is replaced by its own
    six-character escape, which JS reads back as the original character.
    """
    return (
        json.dumps(obj)
        .replace(chr(0x3C), chr(92) + 'u003c')
        .replace(chr(0x3E), chr(92) + 'u003e')
        .replace(chr(0x2028), chr(92) + 'u2028')
        .replace(chr(0x2029), chr(92) + 'u2029')
    )

def build_dense_page(store, world_id: str, session_id: str, *,
                     budget_bytes: int = MOBILE_BYTE_BUDGET,
                     level: int | None = None) -> str:
    """One self-contained HTML page with the points inside it."""
    template_path = viewer_template_path()
    if not template_path.exists():
        raise DenseViewerUnavailable("the dense viewer template is not installed")
    template = template_path.read_text(encoding="utf-8")
    for token in (TOKEN_CONFIG, TOKEN_POINTS):
        if token not in template:
            raise DenseViewerUnavailable(f"the dense viewer template has no {token}")
    raw, config, _ = build_dense_payload(
        store, world_id, session_id, budget_bytes=budget_bytes, level=level
    )
    page = template.replace(TOKEN_CONFIG, js_object_literal(config))
    page = page.replace(TOKEN_POINTS, base64.b64encode(raw).decode("ascii"))
    logger.info(
        "[Tower][WorldBuilder][dense] viewer for %s/%s: %s points, %.1f MB of page",
        world_id, session_id, config["points"], len(page) / 1e6,
    )
    return page
