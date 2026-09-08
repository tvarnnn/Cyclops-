# World Builder dense reconstruction — architecture

Status: **draft, decisions still open pending benchmarks** (see §6).
Companion: `01-EVIDENCE.md`, which holds the measurements this design rests on.

---

## 1. What is being added, and what is deliberately not touched

The sparse GLOMAP global solve stays exactly as it is. It is a good registration
backbone — 0.753 px median reprojection error over 436 posed frames — and
nothing here changes the solver, the keyframe policy, the intrinsics, the
segment machinery or the existing `derived/` tree.

What is added is a **dense stage that runs after the final solve**, consumes the
solve's poses and sparse points, and writes a new, additive, optional artifact
beside the existing ones. Old worlds keep working untouched; a world without the
new artifact renders exactly as it does today.

```
capture ──► keyframes ──► GLOMAP global solve          (unchanged)
                                │
                                │  poses (R_cw, t), K, sparse points,
                                │  observations, per-point RGB
                                ▼
                    ┌───────────────────────────┐
                    │   DENSE FINALIZATION      │   (new, post-Stop)
                    │  1 undistort keyframes    │
                    │  2 per-frame depth        │
                    │  3 align depth to sparse  │
                    │  4 per-frame quality gate │
                    │  5 geometric validity mask│
                    │  6 multi-view consensus   │
                    │  7 voxel fusion + LODs    │
                    └───────────────────────────┘
                                │
                                ▼
                     dense/<session>/ artifact
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
        Tower WebGL viewer              iOS WKWebView
      (desktop inspection)          (same HTML, inline payload)
```

## 2. Why scale comes from the SfM and not from the depth model

Step 3 is the load-bearing idea and the reason this is honest rather than
generative.

A monocular depth model outputs affine-invariant inverse depth: it knows the
*shape* of the scene but not its size or offset. So per frame we solve

```
disp_pred(u, v)  ≈  a · (1 / z_sfm)  +  b
```

by iteratively reweighted least squares with a Huber loss, using **only the
sparse 3-D points the global solve already placed in that frame** (median 405
per frame). Then `z(u, v) = a / (disp(u, v) − b)`.

Every metre of every depth map is therefore anchored to multi-view triangulated
geometry. The network supplies interpolation between the sparse points; it never
supplies scale, position or orientation. A frame where that fit does not hold is
dropped rather than trusted.

Measured on the reference world, **held out** (fit on even-indexed sparse points,
scored on the odd ones the fit never saw): median relative depth residual
**3.0%**, versus 2.6% in-sample. The held-out number being close to the
in-sample number is what says the fit is not overfitting.

## 3. Why nothing is hallucinated

Four mechanisms, in order of how much they remove:

1. **Per-frame quality gate.** A frame whose held-out residual exceeds 8% is
   excluded entirely. On the reference world this drops 79 of 425 frames. A
   regularised refit rescued only one of them, which is itself the evidence that
   those frames are genuinely unreliable rather than merely ill-conditioned —
   so they are dropped, and their part of the room simply stays empty.
2. **Geometric validity mask.** A pixel is rejected if the local depth gradient
   is large relative to its depth (a "flying pixel" strung across a depth edge)
   or if its surface normal is more oblique than 80° to the view ray (where
   depth error explodes). The mask is then eroded by one pixel.
3. **Multi-view consensus.** A candidate point is projected into its ten nearest
   neighbouring cameras and kept only if at least **three** of them hold a depth
   within 3% of the predicted depth. Independently wrong depths do not agree, so
   this is the mechanism that actually separates evidence from invention. It
   removes 38–40% of candidates.
4. **No hole filling.** There is no Poisson reconstruction, no TSDF closure and
   no generative completion. An unobserved region stays empty. Poisson and
   Delaunay are both *closure* methods — watertight by construction — and would
   convert "never seen" into "surface here".

Every surviving point carries a **confidence** value: the number of independent
cameras that agreed with it. That is persisted and exposed to the viewer, so a
user can raise the threshold and watch weakly supported geometry disappear.

## 4. The artifact

Additive and optional, because `require_schema` refuses any version but 1 and
there is no migration machinery. It follows the existing `support.json` and
`placements.json` precedent: a new subtree that old readers never look at.

```
<world>/dense/<session_id>/
    manifest.json      format id, params, counts, bbox, scale semantics, LOD table
    points_l0.bin      full detail
    points_l1.bin      mid   (the canonical level — see below)
    points_l2.bin      mobile
    diagnostics.json   per-frame alignment, gate decisions, timings
```

`points_lN.bin` is a flat interleaved buffer, little-endian, **16-byte stride**,
so a browser can upload it to the GPU without per-point JavaScript:

```
float32 x, float32 y, float32 z, uint8 r, uint8 g, uint8 b, uint8 confidence, uint8 pad
```

Measured LOD ladder on the reference world:

| level | voxel (world units) | points | bytes |
| --- | --- | --- | --- |
| L0 | 0.020 | 11,939,355 | 203.0 MB |
| L1 | 0.045 | 2,125,935 | 36.1 MB |
| L2 | 0.090 | 387,524 | 6.6 MB |

**L1 is the canonical level.** The corpus is 0.23 MP, which gives roughly
6.4 mm per pixel at 3 m and 1–3 cm of depth noise, so L0's voxel is finer than
the evidence supports and mostly stores noise. L1 sits at the honest resolution.
L0 is kept as an optional export for engineering, not as the product default.

**Scale semantics are inherited unchanged and stated explicitly.** The manifest
repeats `state: "unknown"`, `meters_per_unit: null`. The dense stage introduces
no new scale claim, and because COLMAP normalises each component separately,
only component 0 is densified.

## 5. Serving it — and the CSP constraint that decides the design

`GET /worlds/{id}/render` returns HTML that the iOS app shows in a `WKWebView`,
so Tower owns the saved-world viewer and a 3-D renderer needs no Swift change.

But the route sets

```
Content-Security-Policy: default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'
```

With no `connect-src`, **`fetch` and `XMLHttpRequest` are blocked**. The
documented promise is "a page that loads nothing from anywhere", and that is
worth keeping. So:

- **Product path:** the point payload is embedded **inline** in the page as
  base64 at the mobile LOD, and decoded straight into a GPU buffer. No network
  request, no CSP change, no contract change.
- **Engineering path:** a separate desktop-only route serves the large LODs as
  binary with a relaxed `connect-src 'self'`, for offline inspection. The phone
  never uses it.

Both paths share one renderer.

## 6. Decisions still open

These are being settled by measurement, not by preference. Each has a running
experiment and will be recorded in `03-DECISIONS.md` with its evidence.

| decision | competing options | how it is being decided |
| --- | --- | --- |
| depth source | SfM-anchored monocular depth vs COLMAP CUDA PatchMatch MVS vs Depth Anything 3 in pose-conditioned `colmap` mode | identical metric on identical data: median relative depth error against the SfM sparse points, plus completeness, runtime and VRAM |
| depth model | Depth Anything V2 Small (Apache-2.0) vs MoGe-2 (MIT) vs Metric3D v2 (BSD-2), against the CC-BY-NC V2 Large as a labelled ceiling | the same held-out relative residual, and how many frames pass the 8% gate |
| appearance layer | point cloud only vs adding a Gaussian splat | whether a splat trained without any CUDA compiler produces a recognizable room, and whether it invents geometry off the capture path |
| canonical voxel | 0.02 vs 0.045 | held-out geometric accuracy versus artifact size |

The one decision already made on evidence is that **the sparse solve is kept as
the registration backbone** rather than replaced by a feed-forward reconstructor:
it is already sub-pixel accurate, and every alternative either cannot accept
existing poses or is non-commercially licensed.

## 7. Lifecycle — the part that must not break the product

Measured cost on 429 frames: depth 56 s, fusion 105 s, packing about 60 s —
roughly four minutes. **`stop_grace_seconds` is 30.0 and a Windows Job Object
kills the whole process tree**, so this cannot simply be run inline at Stop.

The design therefore is:

- The dense stage runs **after the final build**, inside the existing
  finalization block in `scripts/world_build_session.py`, where the writer lock
  is already held and `finalization` is already `pending` on disk.
- It is **skipped entirely on a hard stop**, exactly as the final solve is, and
  records that it was skipped rather than failing.
- It is **staged and resumable**: alignment, fusion and packing each write their
  own artifact, so an interrupted run resumes instead of restarting.
- It writes its own `dense/status.json` so an interrupted or skipped run is
  legible from disk alone, with no live session.
- An **offline entry point**, `scripts/world_densify.py --world <id>`, densifies
  an already-saved world. This is what migrates the seven existing solved worlds
  and what makes the whole loop runnable without a new physical capture.
- Nothing about cartridge activation, camera transport or the other cartridges
  is touched. World Builder is a supervised child process and stays one.
