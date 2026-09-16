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

from tower.world_builder.surface import (
    MOBILE_PAGE_BYTES,
    SurfaceUnavailable,
    read_mesh_bytes,
)

logger = logging.getLogger(__name__)

# The dense point viewer settled on 6 MB after measuring a desktop draw cliff,
# and that number has never run in a WKWebView. A mesh is cheaper per visible
# surface than a point cloud is -- 150k triangles cover what a million points
# only speckle -- so the same budget buys a great deal more here.
#
# IT BOUNDS THE PAGE, not the mesh. The mesh is inlined base64-encoded, 4/3 of
# its bytes, beside the viewer and its configuration; budgeting the mesh alone
# served a 7.89 MB page for a 5.88 MB level (live replay D).
MOBILE_BYTE_BUDGET = MOBILE_PAGE_BYTES

# The configuration inlined beside the mesh: the camera path (at most 240
# poses), the level list, the currency sentence. About 20 KiB on the canonical
# world; the allowance is generous because under-estimating it is the failure.
PAGE_CONFIG_ALLOWANCE = 64 * 1024


def page_overhead_bytes() -> int:
    """Everything in a surface page except the base64 mesh, estimated."""
    try:
        template = viewer_template_path().stat().st_size
    except OSError:
        template = 64 * 1024
    return int(template) + PAGE_CONFIG_ALLOWANCE


def page_bytes_for_mesh(mesh_bytes: int, overhead: int | None = None) -> int:
    """The page a mesh of `mesh_bytes` makes."""
    overhead = page_overhead_bytes() if overhead is None else int(overhead)
    return overhead + 4 * ((int(mesh_bytes) + 2) // 3)


def mesh_bytes_for_page(page_bytes: int, overhead: int | None = None) -> int:
    """The largest mesh whose page fits `page_bytes`."""
    overhead = page_overhead_bytes() if overhead is None else int(overhead)
    return max(0, (int(page_bytes) - overhead) // 4 * 3)


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


def choose_level(manifest: dict, budget_bytes: int, overhead: int | None = None) -> dict:
    """The largest level whose PAGE fits the budget, else the smallest there is.

    Never returns nothing. A world whose coarsest level still exceeds the
    budget is served anyway and the page says the budget was exceeded -- the
    alternative is telling a wearer their world is empty when it is not.
    """
    levels = [lv for lv in (manifest.get("levels") or [])
              if isinstance(lv.get("bytes"), int)]
    if not levels:
        raise SurfaceViewerUnavailable("the surface manifest lists no levels")
    fitting = [lv for lv in levels
               if page_bytes_for_mesh(lv["bytes"], overhead) <= budget_bytes]
    if fitting:
        return max(fitting, key=lambda lv: lv["bytes"])
    return min(levels, key=lambda lv: lv["bytes"])


def evidence_filter_ran(manifest: dict) -> bool:
    """Whether this artifact's faces went through the per-face frame tests.

    Read off the manifest, because the page's caption promises "at least two
    camera views" and a surface built before the filter existed made no such
    test: its `params` carry neither key. A manifest that records the filter's
    own stats (`detail.evidence_filter`) is believed first."""
    detail = manifest.get("detail")
    if isinstance(detail, dict) and "evidence_filter" in detail:
        return detail["evidence_filter"] is not None
    params = manifest.get("params") or {}
    try:
        return (int(params.get("min_support_frames") or 0) > 0
                or float(params.get("contradiction_ratio") or 0) > 0)
    except (TypeError, ValueError):
        return False


def build_surface_payload(store, world_id: str, session_id: str, *,
                          budget_bytes: int = MOBILE_BYTE_BUDGET,
                          level: int | None = None,
                          overhead: int | None = None):
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
              if level is not None else choose_level(manifest, budget_bytes, overhead))
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
        "evidence_filter": evidence_filter_ran(manifest),
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
        "up": _world_up(store, world_id, session_id, vertices, faces),
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


def _world_up(store, world_id: str, session_id: str, vertices, faces) -> list | None:
    """The vertical the page levels its horizon to.

    Seeded by the cameras (`_camera_up`) and then measured on the surface
    itself. A wearer at a desk looks down, and the mean camera up leans toward
    what they looked at -- 26 degrees on the canonical world, which the phone
    showed as a rolled room. Floors, ceilings and table tops are the surface's
    own evidence of the vertical, and walls are the check: on the canonical
    world the refinement took the median wall tilt from 13.1 to 4.6 degrees and
    the median horizontal-surface tilt from 24.0 to 8.4 (area-weighted).

    The refinement is kept only when the surface supports it: enough
    near-horizontal area, a bounded move from the seed, and walls that end up
    no less vertical than the seed left them. Otherwise the camera estimate
    stands, as before.
    """
    seed = _camera_up(store, world_id, session_id)
    if seed is None:
        return None
    refined = surface_up(vertices, faces, seed)
    return seed if refined is None else refined


def surface_up(vertices, faces, seed, *, cone_deg: float = 30.0,
               min_horizontal_fraction: float = 0.10,
               max_move_deg: float = 45.0) -> list | None:
    """Refine `seed` toward the area-weighted normal of near-horizontal faces,
    or return None when the surface does not support a refinement."""
    import numpy as np

    V = np.asarray(vertices, np.float64)
    F = np.asarray(faces, np.int64)
    u0 = np.asarray(seed, np.float64)
    if len(F) == 0 or not np.isfinite(u0).all() or np.linalg.norm(u0) < 1e-9:
        return None
    u0 = u0 / np.linalg.norm(u0)
    cross = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    area = np.linalg.norm(cross, axis=1)
    total = float(area.sum())
    if not total > 0:
        return None
    n = cross / np.maximum(area, 1e-30)[:, None]
    cos_cone = float(np.cos(np.radians(cone_deg)))
    cos_wall = float(np.cos(np.radians(60.0)))

    def wall_tilt(u):
        dots = np.abs(n @ u)
        wall = dots < cos_wall
        if not wall.any():
            return None
        # The area-weighted MEAN tilt: a median over a few large walls can sit
        # on the one wall a bad vertical happens to leave upright.
        ang = np.arcsin(np.clip(dots[wall], 0.0, 1.0))
        return float((ang * area[wall]).sum() / area[wall].sum())

    u = u0
    for _ in range(5):
        dots = n @ u
        horizontal = np.abs(dots) > cos_cone
        if area[horizontal].sum() < min_horizontal_fraction * total:
            return None
        m = (n[horizontal] * np.sign(dots[horizontal])[:, None]
             * area[horizontal][:, None]).sum(axis=0)
        norm = float(np.linalg.norm(m))
        if norm < 1e-12:
            return None
        u = m / norm
    if u @ u0 < 0:
        u = -u
    if np.degrees(np.arccos(np.clip(u @ u0, -1.0, 1.0))) > max_move_deg:
        return None
    before, after = wall_tilt(u0), wall_tilt(u)
    if before is not None and (after is None or after > before):
        return None
    return [round(float(v), 6) for v in u]


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


def _compose(template: str, raw: bytes, config: dict) -> str:
    page = template.replace(TOKEN_CONFIG, js_object_literal(config))
    return page.replace(TOKEN_MESH, base64.b64encode(raw).decode("ascii"))


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
    page = _compose(template, raw, config)
    size = len(page.encode("utf-8"))
    if level is None and size > budget_bytes:
        # The overhead was an estimate; with the real one, a smaller level may
        # fit. Chosen again once, never looped.
        overhead = size - 4 * ((len(raw) + 2) // 3)
        raw2, config2 = build_surface_payload(
            store, world_id, session_id, budget_bytes=budget_bytes, overhead=overhead)
        if config2["level"] != config["level"]:
            raw, config = raw2, config2
            page = _compose(template, raw, config)
    logger.info(
        "[Tower][WorldBuilder][surface] viewer for %s/%s: level %s, %s faces, "
        "%.1f MB of page", world_id, session_id, config["level"],
        config["faces"], len(page) / 1e6,
    )
    return page
