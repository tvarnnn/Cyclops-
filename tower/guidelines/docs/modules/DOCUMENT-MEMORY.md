# Module Concept — Document Memory

## Status

**IMPLEMENTED, AWAITING A PHYSICAL PAGE** as of 2026-09-07. Promoted from
the research seed at `docs/superpowers/research/2026-08-20-document-memory-design.md`,
rebuilt on 2026-09-07 after the reality check of 2026-08-26 showed the
original detector fired on nothing real. Full account:
`docs/agent-handoffs/DOCUMENT-MEMORY-V1-RESEARCH-IMPLEMENTATION.md`.

| Part | Status |
|---|---|
| Managed storage root, capability on by default | **CURRENTLY IMPLEMENTED** (`tower/data/document_memory`, `TOWER_DOCUMENT_ENABLED`) |
| Steadiness gate + text-detector region finder (no page outline needed) | **CURRENTLY IMPLEMENTED** (`tower/document_memory/gate.py`) |
| Dwell tracking with page-turn segments, best-frame selection | **CURRENTLY IMPLEMENTED** (`dwell.py`) |
| OCR via EasyOCR on CUDA, released at Stop | **CURRENTLY IMPLEMENTED** (optional `[ocr]` extra; `TOWER_DOCUMENT_DEVICE`) |
| Same-page identity across dwells (sightings) | **CURRENTLY IMPLEMENTED** (`identity.py`: words AND look must agree) |
| Persistence, retention window, real purge | **CURRENTLY IMPLEMENTED** |
| Retrieval by time, by content (lexical, page-level, OCR-tolerant), by recency | **CURRENTLY IMPLEMENTED** |
| Live library updates to the phone | **CURRENTLY IMPLEMENTED** (`library.revision` on the status channel) |
| Idle self-stop of a session that lost its stream | **CURRENTLY IMPLEMENTED** (600 s) |
| Perspective correction from a quad | **RETAINED, UNUSED** on the live path (`detect.py`) |
| Reading a physical page through the glasses | **UNTESTED** — no capture contains one; see below |
| Semantic (embedding) retrieval | **PLANNED**, with a named trigger |
| Registration as a production module | **BLOCKED** at the same V1.0/V1.1 boundary as World Builder |
| Voice queries | **PLANNED**, deliberately out of V1 |

Earlier report: `reports/2026-08-22-document-memory-v1-report.md`.

## Goal

The wearer reads a document normally — no "scan this" workflow, no
flattening it on a table, no holding it still for the camera. Later they
ask *"what was that document I looked at about thirty minutes ago?"* and
get an answer grounded in text the system actually captured.

## What this module can and cannot know

**It cannot detect attention.** `07-PLATFORM-CONSTRAINTS.md` Limitation 8
is explicit: something appearing in the glasses camera does not prove the
user looked at it, noticed it, read it, or understood it. There is no eye
tracking on this hardware and none is planned.

What it detects is **a page-like region persistently present in the
camera view, held steadily, for long enough to be worth reading**. That is
a good proxy for reading and a poor synonym for it.

This governs the vocabulary, and the vocabulary is not decoration. The
record is `DocumentObservation`; the fields are `observed_at`,
`observed_seconds`, `pages_observed`; the CLI prints **OBSERVED, NOT
READ**. Nothing in this module is named `read` or `viewed`, and nothing
may be.

## Pipeline

```
frames (live stream, 360x640 @ 12 fps, or a recorded capture)
   |
[every frame]   steady-and-sharp gate                       ~1-3 ms
   |              phase correlation vs previous frame,
   |              variance of Laplacian vs rolling median
   |
[stable frames] text detector (EasyOCR/CRAFT), <= 4 Hz     ~35 ms GPU
   |              union of boxes = the region; no quad needed
   |
[every frame]   dwell / same-region tracking                ~0 ms
   |              + content check vs the segment's reference:
   |                a page turned in place opens a new segment
   |
   +-- no region, or not held >= 1 s --> discarded, nothing persisted
   |
[per segment]   best 1-2 frames (sharpness x squareness)
   |
[per frame]     crop the region, OCR                        ~0.3 s GPU
   |
                readability floor, within-dwell merge, title, excerpt
   |
                identity: same words AND same look as a record?
                  yes -> a SIGHTING of that record
                  no  -> a new record
   |
                persist derived text (not pixels); journal stamp moves
   |
                retrieval: by time, by content (per page), by recency
```

**The 100x cost ratio between the gate and OCR is the whole design.**
The pipeline exists to make the expensive stage rare, not fast. Measured
over 5,000+ real frames it spends 1.2 ms on the median frame and 4-8 ms
on the mean, and OCR runs only when a dwell qualifies.

## The binding constraint — REPLACED 2026-09-07

> The 2026-08-26 reality check found the contour-quad detector fired 6
> times in 9,199 real frames, all false, and zero times after
> re-derivation. Detection, not recognition, was the binding constraint.
> Evidence: `docs/superpowers/research/2026-08-26-document-memory-reality-check.md`.

**The detector was replaced, not re-derived.** A text detector answers
"is there text, and where" far better than a contour finder plus a glyph
statistic ever could from 360x640 pixels, and EasyOCR ships one (CRAFT)
that costs ~35 ms on the GPU. The per-frame stage now asks only whether
the camera is steady and the image sharp; the detector runs on steady
frames at most four times a second; the region a dwell tracks is the
union of the boxes it found. No quadrilateral is required, which matters
because only 4.9% of real frames contained a convex four-corner contour
of any size and a partial page, a screen or a sheet under a hand never
does.

**Measured on real footage (2026-09-07, `scripts/document_memory_replay.py`):**
over five captures and 7,124 frames — the sharpest laptop-screen sessions,
the backlit-keyboard capture that fooled the old detector, and the
blurriest walk — the gate admitted 20-67% of frames, the detector ran
111-800 times, and **zero regions, zero dwells and zero records** resulted.
With a rendered page composited onto 48 consecutive real frames of the
same captures, exactly **one record** resulted each time, at word recall
0.979 (sitting) and 0.894 (walking). The premise is still untested on a
physical page: no capture contains one.

### Recall, restated in geometries the hardware can actually produce

Measured word recall against known rendered text:

| Frame size | Word recall | Reachable? |
|---|---|---|
| 1280×720 landscape | 0.957 – 1.000 | **No — DAT has no landscape mode** |
| 640×480 landscape | 0.905 – 1.000 | **No — 4:3 does not exist either** |
| 640×360 landscape | 0.429 – 0.810 | **No** |
| **360×640 portrait — what is actually delivered** | **0.343 – 1.000, mean 0.703** | yes |
| 504×896 portrait still | **0.886 – 1.000** | yes, DAT's middle rung |

Three of the four originally published rows are **unreachable geometries**.
DAT offers 720×1280, 504×896 and 360×640, all 9:16
(`07-PLATFORM-CONSTRAINTS.md:79`).

Portrait is not simply worse. Its *worst* case (0.343) does fall below the
old worst row, confirming the earlier suspicion at the tail — but its mean
beats landscape's (0.703 against 0.572) and square-on it is far better. Same
pixel count, different failure mode.

### What this changes

The old conclusion — "not a Tower problem, it is a requirement on iOS to
raise stream resolution" — no longer follows, and raising the stream is
actively the wrong move: World Builder measured 720p as **harmful**, with
73.3% of frames falling below its absolute sharpness floor.

**The tension dissolves once you notice this cartridge does not need a
stream.** It needs one or two stills per dwell. A 504×896 *still*, captured
on dwell, buys 0.886–1.000 while the video stream stays at 360×640 for
World Builder. That is a capture-mode request, not a resolution negotiation.

Two prerequisites before any of it:
1. Re-derive `MIN_ROW_TRANSITIONS` against the real negatives.
2. Record a capture in which someone actually looks at a page. **The
   cartridge has never had a chance to succeed.**

This is not a Tower problem and no Tower work fixes it.
`CARTRIDGE-GROUNDWORK.md` predicted it — *"Text/Document ... Missing:
resolution negotiation. DAT's adaptive ladder drops resolution first and
cannot be overridden"* — and the numbers above turn that prediction into
a requirement on the iOS/DAT side. It is recorded in
`docs/agent-handoffs/TOWER-TO-IOS.md`.

## Detection: two stages, and what each may claim

**Stage 1 (every frame) claims nothing about text.** Steady means the
phase-correlation shift against the previous frame is under 1% of the
diagonal with a response above 0.6; sharp means the variance of the
Laplacian clears an absolute floor (25, World Builder's) and 0.55 of the
rolling median of the last 30 frames. The first frame of a stream is
never steady (nothing precedes it), and a rung change resets the
comparison.

**Stage 2 (steady frames, <= 4 Hz) is the OCR engine's own text
detector.** Its boxes make a region when there are at least three, their
median height is at least 7 px (body text at 360x640 is ~8 px and is
readable at that height only when the page fills the frame) and their
padded union covers at least 3% of the frame. Between runs the last
region is carried forward for up to 1.5 s while the view stays steady.

**A dwell is made of segments.** The region tracker compares each frame's
region crop against the segment's first frame by phase correlation; three
consecutive frames below 0.45 open a new segment. Each segment keeps its
own best one or two frames, so a three-page read within one dwell yields
three pages. Segments are capped at six per dwell.

### What still gets through, and what is kept out

- **A keyboard or a screen held steady** is text to a text detector. It
  reaches OCR; what OCR makes of it is kept as a page with
  `readable: false` (regions but no words above the noise floor), and the
  session counts it as `dwells_unreadable`. On real footage of exactly
  those surfaces the detector found no region at all, but that is a
  measurement, not a guarantee.
- **Body text at 360x640 needs the page close.** Recall on rendered pages
  is 0.98 with the page filling the frame, 0.74 at 60% of the frame
  height, 0.02 at 40%. At 504x896 it is 1.00 / 0.98 / 0.66.
- **A partial page, a page under a hand, a screen** are all fine: no
  outline is required. The quad detector's known false negatives (sparse
  pages, a white desk, a page closer than 98% of the frame) no longer
  apply.

## Retrieval is lexical, and says so

BM25 over stored OCR text. It matches a document containing the word
"transformer"; it does **not** match a paraphrase that never uses the
word. Every result carries `match_kind: "lexical"`, the terms it matched
on, and a snippet of the captured text, so an answer is always traceable.

Calling this "semantic retrieval" would be the overclaim Rule 16 exists to
prevent.

**Upgrade trigger, named rather than vague:** when a measured query set
shows lexical recall failing on paraphrase. Embeddings then become
justified; until then BM25 is forty lines, needs no dependency, and is
explainable — which matters more here, because a retrieval answer must be
traceable to text that was actually captured.

**Pages are the unit of scoring** (since 2026-09-07), so a result names
the page the snippet came from; a term of five or more characters also
matches a token within one OCR edit of it (with the rn/m confusion
folded), weighted below an exact match and never above one. The corpus
is cached on the journal's `(mtime, size)` stamp, so a query re-tokenises
the library only when it changed and a newly persisted page is
searchable on the next query. Measured before the cache: 0.3 ms over 10
documents, 4.6 ms over 100, 252 ms over 1000; a persistent index becomes
justified in the thousands.

## Anti-hallucination, as data rather than convention

- Every page stores the text OCR **actually returned**, its per-region
  confidence and its region count.
- A page whose OCR found nothing is still recorded. *"We looked and found
  no readable text"* is a different fact from *"we never looked"*.
- Retrieval returns matched text with the record it came from.
- A query with no confident match returns `sufficient_evidence: False`
  and states that the refusal is about **what was captured, not about the
  world** (Core Principle 3).
- `coverage()` reports `pages_observed` with **`pages_total: None`**. The
  system cannot see pages it was never shown, and inventing a denominator
  would turn an observation gap into a claim of completeness.
- Document confidence is the **weakest** page's, not the average: a
  document is only as trustworthy as its worst-read page.
- Titles and summaries are **extractive**. There is no LLM, and a
  generated summary of partially-captured text would read as
  authoritative while describing pages nobody observed.

## Timing provenance

There is still no capture timestamp on the wire, so every timestamp is
`tower-receipt` time and says so.

A directory of loose jpegs carries no timestamps at all, so replaying one
has to assume a frame interval. Every document produced that way is
stamped `timing_source: "assumed-interval"` with the interval used, so an
assumed duration can never be read as a measured one. Frames from a
capture journal carry real receipt times and are stamped
`"capture-journal"`.

## Privacy

Documents are the platform's clearest case of `06-PRIVACY-DATA.md`'s
Sensitive Visual Information: financial, medical, identity and private
correspondence appear as literal readable text, and a document's whole
point is to be read.

- **Derived text is persisted. Page images are not, by default.** Keeping
  the corrected page image is opt-in and off. When enabled the record
  says `redaction: "none"`, because no redaction exists on this platform
  and a crop is not inherently safe.
- **Raw frames are never persisted by this cartridge.** They live in
  memory for the dwell window and are dropped.
- **Real purge**, per document and whole-store, reporting what it could
  **not** delete. A purge that cannot remove everything must never be
  presented as success.
- **Retention is a window**, defaulting to 30 days rather than forever.
- **No third-party transmission.** OCR, indexing and retrieval are local.

## Spatial context

`world_id`, `world_session_id` and `frame_revision` are **supplied by a
caller or absent**. Nothing in this module derives them, and it must not
import World Builder — a test enforces that in both directions. Absent
means unknown, which is not the same as "nowhere".

The import ban has a second reason worth stating: **World Builder's blur
and motion gates would reject exactly the frames this cartridge wants.** A
held-still, high-detail view of a page has near-zero parallax and is
`insufficient_motion` to a mapper. This cartridge inverts World Builder's
signal, so sharing that code would mean sharing an assumption that is
wrong here.

## Query interface

Deliberately independent of any voice path. `scripts/document_query.py`
and the `DocumentMemory` Python API answer three questions:

```
--recent 5                      what have I looked at lately
--minutes-ago 30 --window 15    the one from about half an hour ago
--text "return policy"          the one about the return policy
```

A future Siri shortcut, custom wake word or iOS screen would sit **above**
this. None is required for the feature to work, and building the voice
layer first would have made the memory untestable.

## Integration boundary

Not a registered production module, for the same reason World Builder is
not: the module contract is a registry of one with a scalar-shaped result.
Additionally, 1.2 s of OCR could not sit on the event loop even if a slot
were free. A test pins non-registration.

It runs as an engine plus a driver, in a **separate process** from the
Tower, consuming a capture through `CaptureFollower` — live or recorded.

## Out of scope for V1

Named so nobody assumes otherwise: multi-document cross-referencing,
cross-session physical-document re-identification, handwriting,
non-Latin scripts, multi-column reading-order preservation, real-time
reading assistance, and any generated (as opposed to extracted) summary.
