"""Compose the appearance viewer page: the top rung of the render ladder.

Contract: `docs/contracts/WORLD-BUILDER-WORLDS.md` §4 (the page and its
transport) and `docs/contracts/WORLD-BUILDER-APPEARANCE.md` (what it draws).

Unlike every other rung this page does NOT carry its data. It is a small shell
-- the renderer, the camera path, the vertical and the addresses of the
appearance routes -- and it fetches the manifest, the proxy and the keyframe
bundles itself. Two reasons, both measured by the fix-it mobile lane:

- the imagery is ~13 MB of ASTC blocks, and inlining it as base64 into a page
  string makes a 20 MB string the phone keeps alive beside the textures;
- a live walk lands a new appearance build every solve, and a page that
  carries its data can only follow one by being replaced, which resets the
  wearer's camera. This page overwrites its texture layers in place instead.

WHERE IT FETCHES FROM is the whole security question, and it is decided here,
once, by `transport`:

- `app` (the default, and the only mode the phone uses): every address is
  `glasses-world://tower/...`, the app's private URL scheme. The page's CSP
  allows `connect-src glasses-world:` and nothing else, and the app's scheme
  handler serves exactly the page and this world's appearance and revision
  routes, proxied to the Tower through an ephemeral, cache-less session. A
  desktop browser opening the page this way fetches nothing and says so.
- `tower`: a desktop debug mode, only when asked for by name
  (`?transport=tower`). Addresses are relative to the Tower's own origin and
  the CSP allows `connect-src 'self'`. It reaches exactly the routes any client
  on the Tower's network could already call; it widens nothing else.

Nothing here is cached and nothing is written.
"""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import quote

from tower.world_builder.surface_render import js_object_literal

logger = logging.getLogger(__name__)

TEMPLATE_NAME = "appearance_viewer.html"
TOKEN_CONFIG = "__WB_APPEARANCE_CONFIG__"
TOKEN_CSP = "__WB_APPEARANCE_CSP__"

TRANSPORT_APP = "app"
TRANSPORT_TOWER = "tower"
TRANSPORTS = (TRANSPORT_APP, TRANSPORT_TOWER)

# The iOS app's private scheme and the host its scheme handler answers for.
APP_SCHEME = "glasses-world"
APP_BASE = f"{APP_SCHEME}://tower"

# The page PROGRAM's revision, not the appearance build's. The page follows
# appearance builds itself, so a new build must not change the revision the
# native follower compares (that would reload the page and reset the camera).
# Bump this when a Tower update must replace pages already open.
PAGE_REVISION = "appearance:1"

# Texture-layer caps. ASTC 6x6 at the canonical 359x639 is 102,720 bytes a
# layer (192 layers = 19.7 MB); the RGBA8 fallback is 917,604 bytes a layer, so
# it is capped hard (48 layers = 42 MB) and the page says the set is reduced.
ASTC_LAYER_CAP = 192
RGBA8_LAYER_CAP = 48

# Display mapping: the keyframes' own brightness, with a hue-preserving
# highlight roll-off above the knee that approaches 0.97 and never reaches white
# (not a clamp). It is the same for every pixel; it adds no lighting and moves
# no colour between surfaces. Revised 2026-09-17 (fix-it blotch lane, after the
# visual review): the first mapping lifted the frames by 1.8 with gamma 1.15,
# and at the keyframes' own pose and field of view the render was 1.19-2.04x as
# bright as the keyframe (median 1.66, NCC 0.85), whites blown to cream. At
# 1.0 / 1.0 / knee 0.8 on the same 10 poses: median 1.01x, NCC 0.94, 3.6% of
# pixels >= 235 against the keyframes' 4.0%.
DISPLAY_EXPOSURE = 1.0
DISPLAY_GAMMA = 1.0
TONE_KNEE = 0.8

# The blend (WORLD-BUILDER-WORLDS.md §4), measured on the canonical world by the
# fix-it viewer-polish lane (Glasses-scratch/wb-final-recon/fixit/viewer-polish):
# - a source's WEIGHT rises over 80 source pixels in from its image border (its
#   evidence keeps 20): a keyframe's rectangle showed as a brightness step on
#   walls. Seam excess (gradient on source switches minus nearby, 8-bit, 34
#   views): first page 6.67, soft weights with a 20 px feather 5.59, 80 px 5.16,
#   with the texture gradient beside the seams 3.75 / 3.40 / 3.50;
# - softmax temperature 0.07 rad of penalty: 0.15 removed a little more seam
#   (5.01) but blurred texture (3.06);
# - the room fades out over 8 CSS px where it borders nothing (screen space);
# - robust consensus among the k + 2 best sources, full strength;
# - a changed choice of sources crossfades over 280 ms;
# - proxy no kept image covers is drawn as the background (with the same soft
#   edge), not as a dark tint: on wide views the tint read as dark shards.
BLEND_TEMPERATURE = 0.07
BORDER_FEATHER_PX = 80
EDGE_FADE_PX = 8
CONSENSUS = 1.0
SOURCE_FADE_MS = 280
# The fix-it blotch lane (Glasses-scratch/wb-final-recon/fixit/blotch):
# - the display field of view is the keyframes' own, the viewport fitted inside
#   their frustum with this margin on the tangent;
# - thin cracks in the phone proxy (a third of its edges are open) are closed
#   on the screen across CRACK_FILL_PX CSS px when both sides are one plane,
#   and shaded from the sources like the proxy;
# - a void is an unlit fog of the coarse grey luminance around it, and dims
#   the room near it by up to VOID_WIDE_FADE.
# The fix-it framing lane (Glasses-scratch/wb-final-recon/fixit/framing):
# - cutting the viewport to the keyframe exactly took the damage out and left a
#   viewfinder so tight that 61% of the reachable views were a clean, well-lit,
#   empty wall. The frame is the capture's plus a MARGIN on each tangent:
#   VIEW_MARGIN on the horizontal and VIEW_MARGIN_V on the vertical (which only
#   binds on a portrait canvas, where the keyframe's own vertical decides).
#   Swept 1.0/1.15/1.25/1.4/1.6 over the 40 walk poses and 20 look-arounds:
#   1.4 keeps drawn at 95.9% on the walk poses (96.9% at 1.25) for 12% more
#   detail, and breaks two of the 61 shots by more than 5 points of drawn;
#   1.6 breaks thirteen.
# - STANDOFF is how close the camera may come to the proxy: at 0.55 it could be
#   pressed against a wall and fill the frame with one blurred patch that the
#   envelope scored as fully supported.
# The fix-it interaction lane (Glasses-scratch/wb-final-recon/fixit/interaction):
# - a look is no longer bounded by the capture at all, so the only thing left
#   between a deliberate forward push and the room is this standoff, and at 1.0
#   on a world whose median scene depth is 4.7 it was a fifth of the room. An
#   independent review measured a 4.48-unit push travelling a median 0.94 units.
#   0.6 is the value that keeps the 360x640 keyframes off the lens without
#   turning the envelope into a cage; the blurred close-up it guards against is
#   a MINOR complaint in that review and the cage is a blocking one.
VIEW_MARGIN = 1.4
VIEW_MARGIN_V = 1.15
STANDOFF = 0.6
CRACK_FILL_PX = 4
VOID_FOG = 0.35
VOID_WIDE_FADE = 0.2


class AppearanceViewerUnavailable(Exception):
    """No appearance page can be composed, with a reason a person can read."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def viewer_template_path() -> Path:
    return Path(__file__).resolve().parent / TEMPLATE_NAME


def content_security_policy(transport: str = TRANSPORT_APP) -> str:
    """The page's policy, identical in its `<meta>` and the route's header."""
    connect = "'self'" if transport == TRANSPORT_TOWER else f"{APP_SCHEME}:"
    return ("default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
            f"connect-src {connect}")


def routes_for(world_id: str, session_id: str) -> dict:
    """The only addresses the page fetches, relative to its base."""
    w, s = quote(world_id, safe=""), quote(session_id, safe="")
    return {
        "manifest": f"/worlds/{w}/appearance/{s}/manifest",
        "chunk": f"/worlds/{w}/appearance/{s}/chunk/",
        "proxy": f"/worlds/{w}/appearance/{s}/proxy/",
        "revision": f"/worlds/{w}/render/revision?session_id={s}",
    }


def build_appearance_config(store, world_id: str, session_id: str, *,
                            transport: str = TRANSPORT_APP,
                            appearance_revision: str | None = None) -> dict:
    """The page's configuration. Reads the manifest and the proxy; no imagery."""
    from tower.world_builder import appearance_pipeline as AP
    from tower.world_builder.surface import SurfaceUnavailable, read_mesh_bytes
    from tower.world_builder.surface_render import _camera_path, _world_up

    if transport not in TRANSPORTS:
        transport = TRANSPORT_APP
    manifest = AP.read_appearance_manifest(store, world_id, session_id)
    if manifest is None:
        raise AppearanceViewerUnavailable("this session has no appearance artifact")
    if not AP.label_matches(store, world_id, session_id, manifest):
        raise AppearanceViewerUnavailable(AP.STALE_LABEL_DETAIL)
    phone = [k for k in (manifest.get("keyframes") or []) if k.get("tier") == "phone"]
    if not phone:
        raise AppearanceViewerUnavailable("the appearance artifact has no keyframes for the phone")
    proxy = manifest.get("proxy") or {}
    raw = AP.read_appearance_file(store, world_id, session_id, "proxy",
                                  str(proxy.get("digest")), manifest)
    if raw is None:
        raise AppearanceViewerUnavailable("the appearance proxy is missing")
    try:
        vertices, faces, _colors, _normals = read_mesh_bytes(raw)
    except SurfaceUnavailable as exc:
        raise AppearanceViewerUnavailable(exc.reason) from None
    if len(faces) == 0:
        raise AppearanceViewerUnavailable("the appearance proxy has no triangles")

    currency = AP.appearance_currency(store, world_id, session_id, manifest)
    scale = manifest.get("scale") or {}
    selection = manifest.get("selection") or {}
    return {
        "world_id": world_id,
        "session_id": session_id,
        "transport": transport,
        "base": "" if transport == TRANSPORT_TOWER else APP_BASE,
        "routes": routes_for(world_id, session_id),
        "appearance_revision": appearance_revision,
        "build_id": manifest.get("build_id"),
        "quality": manifest.get("quality"),
        "keyframes_phone": len(phone),
        "keyframes_total": len(manifest.get("keyframes") or []),
        "proxy_faces": int(len(faces)),
        "current": bool(currency.get("current")),
        "currency_reason": currency.get("reason"),
        "scale_state": scale.get("state", "unknown"),
        "median_scene_depth": selection.get("z_ref"),
        "cameras": _camera_path(store, world_id, session_id),
        # The vertical the horizon is levelled to: the surface page's own rule
        # (`surface_render.surface_up`), measured on the proxy this page draws.
        "up": _world_up(store, world_id, session_id, vertices, faces),
        "astc_layer_cap": ASTC_LAYER_CAP,
        "rgba8_layer_cap": RGBA8_LAYER_CAP,
        "exposure": DISPLAY_EXPOSURE,
        "gamma": DISPLAY_GAMMA,
        "tone_knee": TONE_KNEE,
        "k": 4,
        "blend_temperature": BLEND_TEMPERATURE,
        "border_feather_px": BORDER_FEATHER_PX,
        "edge_fade_px": EDGE_FADE_PX,
        "consensus": CONSENSUS,
        "source_fade_ms": SOURCE_FADE_MS,
        "view_margin": VIEW_MARGIN,
        "view_margin_v": VIEW_MARGIN_V,
        "standoff": STANDOFF,
        "crack_fill_px": CRACK_FILL_PX,
        "void_fog": VOID_FOG,
        "void_wide_fade": VOID_WIDE_FADE,
    }


def build_appearance_page(store, world_id: str, session_id: str, *,
                          transport: str = TRANSPORT_APP,
                          appearance_revision: str | None = None) -> str:
    """The page shell. Raises `AppearanceViewerUnavailable`."""
    template_path = viewer_template_path()
    if not template_path.exists():
        raise AppearanceViewerUnavailable("the appearance viewer template is not installed")
    template = template_path.read_text(encoding="utf-8")
    for token in (TOKEN_CONFIG, TOKEN_CSP):
        if token not in template:
            raise AppearanceViewerUnavailable(f"the appearance viewer template has no {token}")
    config = build_appearance_config(store, world_id, session_id, transport=transport,
                                     appearance_revision=appearance_revision)
    page = (template.replace(TOKEN_CSP, content_security_policy(config["transport"]))
            .replace(TOKEN_CONFIG, js_object_literal(config)))
    logger.info(
        "[Tower][WorldBuilder][appearance] viewer for %s/%s: %s phone keyframes, "
        "%.0f KB of page (%s transport)", world_id, session_id, config["keyframes_phone"],
        len(page) / 1e3, config["transport"],
    )
    return page
