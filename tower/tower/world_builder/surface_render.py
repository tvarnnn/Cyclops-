"""Compose the surface viewer page the phone receives.

The iOS app shows a saved world in a `WKWebView` loaded with
`loadHTMLString(_:baseURL: nil)`. That page has **no origin**: it cannot
fetch, cannot load a script or a stylesheet from anywhere, and the app's
navigation policy cancels its links. So everything -- the renderer, the
styles, and the mesh itself -- is inlined into one string, and the route
serves it under a Content-Security-Policy that says so from the other side.

The mesh is chosen by byte budget, not by triangle count: the artifact ships
a level-of-detail ladder and this picks the largest rung that fits. Points
would be thinned; a mesh cannot be thinned without tearing it, so the choice
is which rung, never which triangles.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path

from tower.world_builder.surface import SurfaceUnavailable, read_mesh_bytes

logger = logging.getLogger(__name__)

# The dense point viewer settled on 6 MB after measuring a desktop draw cliff,
# and that number has never run in a WKWebView. A mesh is cheaper per visible
# surface than a point cloud is -- 150k triangles cover what a million points
# only speckle -- so the same budget buys a great deal more here.
MOBILE_BYTE_BUDGET = 6 * 1024 * 1024

TOKEN_CONFIG = "__WB_SURFACE_CONFIG__"
TOKEN_MESH = "__WB_SURFACE_MESH__"
TEMPLATE_NAME = "surface_viewer.html"


class SurfaceViewerUnavailable(Exception):
    """No surface page can be composed, with a reason a person can read."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def viewer_template_path() -> Path:
    return Path(__file__).resolve().parent / TEMPLATE_NAME


def js_object_literal(value: dict) -> str:
    """JSON that is safe to paste inside a `<script>`.

    `</script>` inside a string literal ends the script element, and U+2028 and
    U+2029 are line terminators to a JavaScript parser but not to JSON. Both
    turn a data value into markup or a syntax error.
    """
    return (
        json.dumps(value)
        .replace(chr(0x3C), chr(92) + "u003c")
        .replace(chr(0x3E), chr(92) + "u003e")
        .replace(chr(0x2028), chr(92) + "u2028")
        .replace(chr(0x2029), chr(92) + "u2029")
    )


def choose_level(manifest: dict, budget_bytes: int) -> dict:
    """The largest level whose bytes fit the budget, else the smallest there is.

    Never returns nothing. A world whose coarsest level still exceeds the
    budget is served anyway and the page says the budget was exceeded -- the
    alternative is telling a wearer their world is empty when it is not.
    """
    levels = [lv for lv in (manifest.get("levels") or [])
              if isinstance(lv.get("bytes"), int)]
    if not levels:
        raise SurfaceViewerUnavailable("the surface manifest lists no levels")
    fitting = [lv for lv in levels if lv["bytes"] <= budget_bytes]
    if fitting:
        return max(fitting, key=lambda lv: lv["bytes"])
    return min(levels, key=lambda lv: lv["bytes"])


def build_surface_payload(store, world_id: str, session_id: str, *,
                          budget_bytes: int = MOBILE_BYTE_BUDGET,
                          level: int | None = None):
    """Return (mesh bytes, config dict) for one session's best fitting level."""
    from tower.world_builder.surface_pipeline import (
        read_surface_level,
        read_surface_manifest,
        surface_currency,
    )

    manifest = read_surface_manifest(store, world_id, session_id)
    if manifest is None:
        raise SurfaceViewerUnavailable(
            "this session has no surface reconstruction")

    chosen = (next((lv for lv in manifest["levels"] if lv["level"] == level), None)
              if level is not None else choose_level(manifest, budget_bytes))
    if chosen is None:
        raise SurfaceViewerUnavailable(f"the surface has no level {level}")

    try:
        raw = read_surface_level(store, world_id, session_id, chosen["level"],
                                 manifest=manifest)
    except SurfaceUnavailable as exc:
        raise SurfaceViewerUnavailable(exc.reason) from None
    # Parse it here rather than trusting the manifest's counts. A torn or
    # mismatched buffer must be caught on THIS side, where it degrades to the
    # next rung of the ladder, not on the phone where it is a blank canvas.
    try:
        vertices, faces, _colors, _normals = read_mesh_bytes(raw)
    except SurfaceUnavailable as exc:
        raise SurfaceViewerUnavailable(exc.reason) from None
    if len(faces) == 0:
        raise SurfaceViewerUnavailable(
            "the surface reconstruction of this session has no triangles")

    currency = surface_currency(store, world_id, session_id, manifest)
    scale = manifest.get("scale") or {}
    config = {
        "world_id": world_id,
        "session_id": session_id,
        "format": manifest.get("format"),
        "level": chosen["level"],
        "levels": manifest.get("levels"),
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "bytes": len(raw),
        "quality": (manifest.get("params") or {}).get("quality", "final"),
        "frames_used": manifest.get("frames_used"),
        "frames_offered": manifest.get("frames_offered"),
        "voxel": manifest.get("voxel"),
        "median_scene_depth": manifest.get("median_scene_depth"),
        "scale_state": scale.get("state", "unknown"),
        "current": bool(currency.get("current")),
        "currency_reason": currency.get("reason"),
        "cameras": _camera_path(store, world_id, session_id),
        "up": _camera_up(store, world_id, session_id),
    }
    return raw, config


def _camera_path(store, world_id: str, session_id: str) -> list:
    """Where the wearer stood, so the viewer can open there and walk it.

    Opening on an arbitrary orbit of an incomplete room shows its missing
    back; opening where a camera stood guarantees geometry in front of you,
    because that geometry is what that camera measured.
    """
    from tower.world_builder.global_solve import load_solution

    try:
        solution = load_solution(store, world_id, session_id)
    except Exception:  # noqa: BLE001 -- the page is better without it than absent
        return []
    if solution is None:
        return []
    import numpy as np

    out = []
    for kid in (solution.keyframe_ids or []):
        pose = (solution.poses or {}).get(kid)
        if pose is None:
            continue
        R = np.array(pose["rotation"], float).reshape(3, 3)
        t = np.array(pose["translation"], float)
        centre = -R.T @ t
        forward = R.T @ np.array([0.0, 0.0, 1.0])
        out.append([round(float(v), 5) for v in (*centre, *forward)])
    # A few hundred is plenty to step through and keeps the page small. The
    # ceiling division is what makes 240 a cap: floor division gave a step of 1
    # for any walk under 480 keyframes and sent all of them.
    step = max(1, -(-len(out) // 240))
    return out[::step]


def _camera_up(store, world_id: str, session_id: str) -> list | None:
    """The walk's own up direction, as the mean of every camera's -y axis.

    The solve's gauge has no declared vertical (`up_axis` is "unknown"), and a
    viewer that assumes world -Y is up tilts the horizon by however far this
    solve's frame is rotated -- 14.3 degrees on the canonical world. A wearer
    holds their head near level, so the average camera up is the vertical to
    within a few degrees, and it is measured rather than assumed.
    """
    from tower.world_builder.global_solve import load_solution

    try:
        solution = load_solution(store, world_id, session_id)
    except Exception:  # noqa: BLE001 -- the page can estimate its own
        return None
    if solution is None or not solution.poses:
        return None
    import numpy as np

    ups = []
    for pose in solution.poses.values():
        R = np.array(pose["rotation"], float).reshape(3, 3)
        ups.append(R.T @ np.array([0.0, -1.0, 0.0]))
    mean = np.mean(ups, axis=0)
    norm = float(np.linalg.norm(mean))
    if norm < 1e-6:
        return None
    return [round(float(v), 6) for v in mean / norm]


def build_surface_page(store, world_id: str, session_id: str, *,
                       budget_bytes: int = MOBILE_BYTE_BUDGET,
                       level: int | None = None,
                       max_points: int | None = None) -> str:
    """One self-contained HTML page with the mesh inside it."""
    template_path = viewer_template_path()
    if not template_path.exists():
        raise SurfaceViewerUnavailable("the surface viewer template is not installed")
    template = template_path.read_text(encoding="utf-8")
    for token in (TOKEN_CONFIG, TOKEN_MESH):
        if token not in template:
            raise SurfaceViewerUnavailable(
                f"the surface viewer template has no {token}")

    # `max_points` is the worlds contract's budget knob and must not be
    # ignored just because this representation is not made of points. It was
    # dropped on the dense path once and `max_points=1` returned a 6 MB page.
    # Here it selects a coarser rung, converted at the dense point format's
    # 16 bytes a point so one `max_points` buys a comparable page on either
    # rung, and it only ever moves the budget downwards.
    if max_points is not None:
        budget_bytes = min(budget_bytes, max(1, int(max_points)) * 16)

    raw, config = build_surface_payload(
        store, world_id, session_id, budget_bytes=budget_bytes, level=level)
    page = template.replace(TOKEN_CONFIG, js_object_literal(config))
    page = page.replace(TOKEN_MESH, base64.b64encode(raw).decode("ascii"))
    logger.info(
        "[Tower][WorldBuilder][surface] viewer for %s/%s: level %s, %s faces, "
        "%.1f MB of page", world_id, session_id, config["level"],
        config["faces"], len(page) / 1e6,
    )
    return page
