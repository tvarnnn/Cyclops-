# Document Memory V1 — research and implementation handoff

**Written:** 2026-09-07, on Windows, in the worktree
`C:\Users\tvllo\Projects\Glasses-worktrees\document-memory-v1`, branch
`feature/document-memory-v1`, base `6beaf57` (`integration/wb-cv-ios-validation-v1`).
**Verdict:** see §17 — **READY WITH KNOWN LIMITATIONS**, the limitations being
that the iOS half is unverified by a compiler and no physical page has yet
been read through the glasses.

---

## 1. Executive summary

**What existed.** A complete, carefully reasoned Document Memory cartridge
(~12,000 lines of code, tests and docs) that recorded nothing. Its per-frame
page detector — a contour-quad finder plus a glyph statistic — fired six
times on 9,199 real frames, all on blinds and keyboards, and zero times after
re-derivation. Its storage root had no default, so every stock Tower declared
the cartridge unavailable and the phone told the wearer to set
`TOWER_DOCUMENT_ROOT`. OCR ran on the CPU. Dedup existed only inside one
dwell. The phone dropped the library half of every status push and never
started the camera.

**What was built.**

- A managed root (`tower/data/document_memory`) with `TOWER_DOCUMENT_ENABLED`
  as the switch, capture on by default, sessions still started by a person.
- A new capture gate: phase-correlation steadiness plus rolling-median
  sharpness on every frame (~1-3 ms), and the OCR engine's own text detector
  on steady frames at most four times a second (~35 ms on the GPU). No page
  outline is required.
- Dwells made of segments, so a page turned in place becomes a new page;
  loss tolerance in seconds, because the stream is 12 fps.
- OCR on CUDA (0.27 s a page against 1.86 s on CPU), released with its GPU
  memory at Stop.
- Identity across dwells: a later look whose words AND look agree with a
  page on record becomes a *sighting* of that record. Two invoices on one
  template, a half-shared page and an unreadable re-view stay separate.
- Page-level search with bounded OCR-noise tolerance, cached on the journal
  stamp so a new page is searchable on the next query.
- `library.revision` on the status channel as the live-update trigger; an
  idle session stops itself ten minutes after its stream closes.
- iOS: follows the new identifiers, refreshes on the revision, lists on
  open, starts the camera on Start and stops only what it started.
- A replay benchmark over real captures with a synthetic positive control.

**Readiness.** Every Tower test passes; real footage yields zero false
records over 7,124 frames; an injected page yields exactly one record at
0.94-0.98 word recall. Physical validation is a precise procedure (§16), not
an open question. The iOS code was written without Xcode and must be built
by the Mac lane first.

---

## 2. Existing implementation findings

The prior lane's account is in
`Glasses-scratch/document-memory-v1/research/01-tower-audit.md` (Tower),
`02-ios-audit.md` (iOS) and `03-prior-research-digest.md` (history). The
findings that mattered:

**`TOWER_DOCUMENT_ROOT` was configuration, hiding a working backend.** With
it unset, `config.document_root` was `None`, `registry.declare` marked both
contracts `available: false` with a reason naming the variable, `/documents*`
answered 404, and no `DocumentLive` was constructed. iOS keyed the
"has not declared a contract" headline on `available: false` and appended
the Tower's reason. Object Memory had reversed the identical default on
2026-08-26 for the identical reason.

**But the backend behind it could not record.** `detect.py` required a
convex four-point contour covering ≥6% of the frame, then rows of ink, then a
glyph-transition statistic whose own docstring said it "does not separate
text from structure" and sat in a 10-wide window pinned to 360x640. Only
4.9% of real frames contained any convex quad; the survivors were blinds and
keyboards. The 2026-08-26 reality check concluded "not viable as-is".

**Everything downstream of detection was sound and is retained:** the
shared `LiveSession` lifecycle (single slot, parked worker, invalidation
latch, four-step stop), the append-only JSONL store with atomic rewrites and
real deletion, the record vocabulary ("observed", never "read"), the
three-answer wire envelope, capture lineage on the dwell, retention on a
real cadence, and BM25 with refusals.

**Other gaps:** `EasyOcrRecogniser()` defaulted to `gpu=False`; dedup was
within-dwell only (`identity: "no-document-identity-across-sightings"`);
search re-parsed the journal per query and could not attribute a match to a
page; the phone never refetched the library and never started the camera;
document detail was decoded but unreachable from the UI.

---

## 3. Research findings

Seven investigation agents ran in parallel; reports are under
`Glasses-scratch/document-memory-v1/research/`. The OCR-engine agent was
terminated by a rate limit before writing; its ground is covered by the
lead's own measurements (`bench/`) and the 2026-08-22 dependency dry-run.

**Capture/stability (06).** Variance of Laplacian is far more blur-sensitive
than Tenengrad and 3-7x cheaper; its absolute value is not portable across
rungs but its ratio to a rolling median is (World Builder's 0.55). Phase
correlation on a 180-px downscale gives a shift in pixels and a response
that collapses (~0.24) when content changes without motion — a page-turn
signal a quad tracker lacks. "Capture first stable frame" is wrong on this
platform (the head settles before the eyes focus); best-of-window is right.
MSER counts and edge density are useless as text gates on real negatives;
the learned detector is the gate.

**OCR (lead's measurements).** EasyOCR 1.7.2 on the RTX 5070: reader
construction 1.3 s, 0.27 s per 800x1040 page on CUDA vs 1.86 s CPU; the
detector-only stage 31-47 ms at 360x640, 55-66 ms at 504x896, 0 boxes on
noise. Full-frame recall on rendered pages at 360x640: 0.98 / 0.74 / 0.02 at
90% / 60% / 40% of frame height; at 504x896: 1.00 / 0.98 / 0.66; at 720x1280
1.00 throughout. RapidOCR and PaddleOCR were rejected (they install
`opencv-python` beside `opencv-python-headless`), Tesseract has no binary,
docTR/Surya add a second model stack for no measured gain.

**Deduplication (06).** pHash-64 of the rectified crop: 0-2 bits apart for
the same page under shift, scale, perspective, exposure, JPEG and blur; 8-24
for different prose; but 2 for two invoices on one template. Token-set
Jaccard: 0.91-1.00 for readable re-views, 0.40-0.45 for the hardest
different pages, 0.14 for unrelated prose. Sequence ratio is unusable (OCR
returns lines in different orders). Conclusion: merge only when text and
look agree; unreadable readings abstain.

**Persistence/search (07).** JSONL journal + atomic rewrite is the shape all
three cartridges converged on. Measured: shipped BM25 276 ms at 3k pages,
988 ms at 10k; SQLite FTS5 answers in <1 ms but is a second source of
truth. Decision: no SQLite in V1 — page-level in-process BM25 cached on the
journal stamp, with the trigger for an index in the thousands.

**Privacy (07).** Doctrine: derived text persists, raw imagery does not by
default, a crop is not inherently safe, deletion must be real, retention
must be chosen. The only face redaction in the repo lives inside World
Builder, which Document Memory may not import. Decision: text-only
persistence stays; page images remain an opt-in the web process cannot
enable.

**Performance (07, 04).** 20 cores, 31.7 GB RAM, RTX 5070 12 GB. Known
pitfalls: torch's per-thread OpenMP pool never reclaimed (the parked worker
already solves it), spin-waiting pools, orphaned children on Windows.
Decision: OCR in-process on the reused worker; release drops the reader,
collects, and empties the CUDA cache.

**Real data (04).** 97 captures, 45,594 frames, all 360x640, one apartment.
**None contains a paper document**; a detector sweep of every tenth frame
(4,599) found zero confident text boxes. Real footage is therefore
negative/blur material; OCR ground truth is synthetic.

**Rejected:** raising the stream rung from the Tower (a cross-cartridge
choice the phone makes; 720p breaks World Builder); a subprocess OCR worker
(the thread model is already hardened and a Windows child is an orphan
risk); embeddings (no paraphrase failure has been measured); keeping the
quad detector as the admission test.

---

## 4. Selected architecture

```
glasses (360x640 @ 12 fps) ──ws──▶ Tower frame path ──offer_frame──▶ single slot
                                                                          │ worker thread
   every frame   gate.FrameGate     steady? sharp?                 ~1-3 ms │
   steady frames gate.PageFinder    EasyOCR.detect(), ≤4 Hz       ~35 ms  │ carried forward ≤1.5 s
   every frame   dwell.DwellTracker same region? content changed?  ~0 ms  │ segments, best 1-2 frames each
   dwell ends    engine._record     crop → EasyOCR.read()         ~0.3 s  │ per selected frame
                 readability floor, within-dwell merge, title, excerpt     │
                 identity.compare   words AND look vs recent records       │ → sighting or new record
                 store.append / store.update   documents.jsonl, atomic     │ journal stamp moves
                                                                          ▼
   results.DocumentStatusProducer  library.revision ──status push──▶ iOS re-fetches /documents
   routes/documents.py             /documents, /search (per page), /{id}, /documents-session/*
```

Stop: close the slot, flush the open dwell (at most four OCR calls,
newest frames first), join the worker, release the reader and the CUDA
cache. Disconnect: the session keeps running; if no stream reopens within
600 s it stops itself, dropping (not reading) any dwell that was still
open when the stream closed, and logs that it did. Second session: a fresh reader on the
same parked worker thread, counters reset, same library.

---

## 5. Storage and data model

**Root.** `config.DEFAULT_DOCUMENT_ROOT = TOWER_ROOT/data/document_memory`,
absolute, isolated from `object_memory/` and `world_builder/`. Override with
`TOWER_DOCUMENT_ROOT`; `TOWER_DOCUMENT_ENABLED=false` removes the root and
the routes 404 with a body that still names `TOWER_DOCUMENT_ROOT` (the phone
string-matches it). Other settings: `TOWER_DOCUMENT_CAPTURE` (default on),
`TOWER_DOCUMENT_AUTOSTART` (default off), `TOWER_DOCUMENT_DEVICE`
(auto/cuda/cpu), `TOWER_DOCUMENT_RETENTION_DAYS` (30).

**Files.** `<root>/documents.jsonl`, one `DocumentObservation` per line,
append-only; prune, purge and sighting-merge rewrite the whole file
atomically (temp + fsync + replace) on raw dicts, so unknown keys survive.
`<root>/pages/` exists only when page images were explicitly enabled by a
CLI, never by the web process.

**Record.** `DocumentObservation` (schema_version 1, unchanged — every
addition has a default a version-1 reader tolerates): `document_id`,
`observed_at`, `recorded_at`, `observed_seconds`, `pages[]`, `title`,
`summary` (first 40 words verbatim), `frames_considered`, `frames_ocred`,
`end_reason`, `confidence` (weakest page), `capture_id`, timing provenance,
`retains_raw_imagery`, `redaction`, `privacy_tags`, and new
`sightings[]` (each: `observed_at`, `observed_seconds`, `capture_id`,
`source_seq`, `end_reason`, `frames_considered`). Derived:
`sighting_count`, `last_observed_at`, `total_observed_seconds`.

**Page.** `PageObservation`: `text`, `region_count`, confidences, `sharpness`,
`squareness`, `source_seq`, `observed_at`, `observation_count`, and new
`visual_hash` (16 hex chars, pHash of the text region), `box_count`,
`readable`.

**Session relationship.** A dwell is a record. A record's first observation
is never rewritten; later looks add sightings and may replace a page's text
with a higher-confidence reading of the same page.

**Index.** None persisted. The search corpus is rebuilt in memory when
`(mtime_ns, size, retention)` of the journal changes.

**Versioning.** Integer `schema_version`, readers skip unknown versions with
a warning. No migration was needed: all fields are additive.

---

## 6. Capture quality

**Steady** = shift < 1% of the frame diagonal (7 px at 360x640) with
phase-correlation response ≥ 0.6. **Sharp** = variance of Laplacian ≥ 25
absolute AND ≥ 0.55 of the rolling median of the last 30 frames. The first
frame of a stream is never admitted; a rung change resets the comparison.

**Region** = the padded union of the detector's boxes when there are ≥ 3
boxes, median box height ≥ 7 px, and the union covers ≥ 3% of the frame.
The detector runs at most every 0.25 s on steady frames (floor 0.1 s); the
last region is carried forward ≤ 1.5 s.

**Dwell** = the same region (centre within 18% of the diagonal, area ratio
0.5-2.0) present for ≥ 4 frames AND ≥ 1.0 s, tolerating ≤ 1.5 s of loss,
capped at 180 s. **Segment** = the region's inner 80% crop stops correlating
(response < 0.45) with the segment's reference for 3 consecutive frames; ≤ 6
segments per dwell; the oldest is dropped, not the newest.

**Best frame** = sharpness × squareness within each segment, top 2, floor
40 on the crop's Laplacian variance.

**Redundancy.** Two views of one page inside a dwell merge (token overlap
≥ 0.70, higher confidence wins). Across dwells, §8.

**Why these numbers.** Fractions of the diagonal and seconds rather than
pixels and frames, so the policy means the same thing at every rung and
frame rate. The response thresholds come from synthetic measurement
(shift ≥ 0.9, page turn 0.24) with margin for sensor noise; the sharpness
ratio is World Builder's, measured on a real walk. Every threshold is a
field on `GatePolicy`, `RegionPolicy` or `DwellPolicy` so the benchmark can
sweep it. **None has been calibrated on a real page yet** — that is what the
physical test is for, and `scripts/document_memory_replay.py` is how to
re-derive from the first real capture.

---

## 7. OCR

**Candidates.** EasyOCR (current), RapidOCR/PaddleOCR, Tesseract, docTR,
Surya. Rejected on operational grounds (§3). EasyOCR retained: torch is
already present, no second cv2 distribution, and it ships the detector the
gate needs.

**Selected pipeline.** `EasyOcrRecogniser(device="auto")`, loaded on the
worker thread inside `state: "starting"`. `detect(gray)` for the gate,
`read(crop)` for the page. No preprocessing: EasyOCR reads the grayscale
crop at 0.98 recall square-on and 0.98 at tilt 0.5 (synthetic); perspective
warp was measured to add nothing and needs a quad that real frames do not
supply. `release()` drops the reader, `gc.collect()`s and empties the CUDA
cache.

**Measured (this host, RTX 5070).**

| Measurement | Value |
|---|---|
| Reader load | 1.3 s (5.4-6.2 s cold in the replay, model files paged in) |
| Read, 800x1040 page | 274 ms CUDA / 1,861 ms CPU |
| Read, 360x640 frame | 290-410 ms CUDA |
| Detect only, 360x640 | 31-47 ms; 55-66 ms at 504x896 |
| GPU memory during a session | 388-663 MB reserved; 0-63 MB after release |
| Word recall, synthetic, 360x640 | 0.98 / 0.74 / 0.02 at 90 / 60 / 40 % fill |
| Word recall, synthetic, 504x896 | 1.00 / 0.98 / 0.66 |
| Injected page on real footage | 0.979 (sitting), 0.936 (walking) |

**Failure semantics.** An OCR exception records the page as unreadable and
continues; a page with regions but no words above the noise floor
(mean confidence < 0.15 or no two-character token) is kept with
`readable: false`; a dwell whose every page is unreadable is persisted and
counted as `dwells_unreadable`. Confidence is a label derived from the mean
region confidence; a document's is its weakest page's.

---

## 8. Deduplication and grouping

`identity.compare(existing, incoming)` returns one of `same`, `shares-text`,
`same-look`, `different`, `abstain`:

- both readings must be able to testify (≥ 3 tokens, mean confidence ≥ 0.30)
  or the answer is `abstain` and the dwell is a new record;
- words agree when token-set Jaccard ≥ 0.70, or containment ≥ 0.85 with
  Jaccard ≥ 0.25 (a partial view), AND numeric tokens agree (Jaccard ≥ 0.5
  when both sides carry ≥ 3 numbers);
- look agrees when the pHash Hamming distance ≤ 6 of 64 bits;
- `same` needs words AND look (or words alone when a record has no hash —
  records written before 2026-09-07).

The engine compares a dwell's first readable page against the 200 most
recently recorded documents. On `same`, the record gains a `Sighting`, the
page's text is replaced only by a higher-confidence reading of the same
words, and nothing about the first observation moves. Tested: same frame
twice, shifted, under perspective, revisited later, template twins (kept
apart), the same paragraph in two documents (kept apart), blank re-view
(kept apart).

**Grouping.** A dwell is the record; pages turned within a dwell are its
pages. Records are not grouped into "documents" across dwells — the reliable
signal for that does not exist without page numbers or headers, and a false
merge destroys a page. This is stated on the wire (§15.5 of the contract).

---

## 9. Search

`retrieval.DocumentMemory.search_text`: BM25 (k1 1.5, b 0.75, smoothed
IDF) over PAGES, title tokens weighted double; a document's score is its
best page's and the result carries `page_index`, `snippet` from that page,
`matched_terms`, `fuzzy`. Terms of ≥ 5 characters also match a token within
one edit or one they prefix, after folding the rn/m, vv/w, 0/o, 1/l OCR
confusions, at 0.6 weight; exact-term count is the primary sort key so an
exact hit always outranks a near one. Floor 0.10; refusals say whether the
memory was empty or the terms were absent. Corpus cached per store path on
`(mtime_ns, size, retention)`; a purge or an append invalidates it, so a new
page is searchable on the next query. `recent` orders by last sighting;
`around` counts a sighting inside the window.

**Live updates.** `library.revision` (journal mtime) on the status push;
iOS re-runs its standing query when it changes and never on a timer.

---

## 10. Privacy and retention

**Persisted:** OCR text and per-region confidence; titles and 40-word
excerpts (verbatim); timestamps (Tower receipt); capture id and frame
sequence numbers (a joinable pointer into a recording, stated as such);
a 64-bit perceptual hash per page (not renderable, not a fingerprint of the
words — it is one of two witnesses and never sufficient alone); box counts;
sightings. **Not persisted:** frames, crops, thumbnails, embeddings, model
caches beyond EasyOCR's own weights in `~/.EasyOCR`. Page images remain an
opt-in on the CLI only; `test_the_production_session_keeps_no_page_images`
still holds and no route serves an image.

**Retention:** 30 days by default, enforced at session start and after every
write; readers can only narrow. **Deletion:** `scripts/document_query.py
--purge [document_id]`, real, reporting what it could not remove; no
unauthenticated HTTP delete, by doctrine. The phone still has no delete
control (the Object Memory lane made the same call).

---

## 11. Real project data evaluation

**Used, read-only, from the canonical checkout's `tower/data/captures/`:**
`64f48114` (527 frames, sharp laptop chat UI, blinds), `b5a0d654` (561,
the backlit-keyboard false positive), `ddcf9426` (1,371, desk monitor walk),
`2f37f2d7` (2,395, longest and sharpest sitting capture), `7febdae8` (2,270,
blurriest live walk). Outputs went to `Glasses-scratch/document-memory-v1/replay/`;
nothing under `tower/data` was written.

**Why:** they are the surfaces the old detector confused (screens,
keyboards, blinds), the sharpest steady views (most detector runs), and the
blurriest motion (most gate rejections).

**Results, end to end (`scripts/document_memory_replay.py`):**

| Capture | Frames | Stable | Detector runs | Regions | Records | ms/frame mean / p50 / p95 |
|---|---|---|---|---|---|---|
| 64f48114 | 527 | 67% | 111 | 0 | 0 | 8.0 / 1.25 / 35.6 |
| 7febdae8 | 2,270 | 20% | 200 | 0 | 0 | 4.4 / 1.24 / 28.8 |
| 2f37f2d7 | 2,395 | 73% | 534 | 0 | 0 | 7.7 / 1.25 / 34.7 |
| b5a0d654, ddcf9426 (session script) | 1,932 | — | — | — | 0 | 10-17 |
| 64f48114 + injected page (48 frames) | 527 | 68% | 111 | 47 | **1** (recall 0.979) | 11.8 |
| 7febdae8 + injected page (48 frames) | 2,270 | 22% | 206 | 47 | **1** (recall 0.936) | 4.9 |

**Limitations.** No real footage contains a document, so every recall
figure is against a rendered page with known text (the fixtures in
`tests/document_fixtures.py`) composited onto real frames; real paper is
lower-contrast than the renderer and every synthetic threshold will read
lower on it. Labelled evaluation exists only for that synthetic text; no
accuracy on real paper is claimed anywhere.

---

## 12. Performance

| Quantity | Measured |
|---|---|
| Per-frame cost, real footage | mean 4.4-8.0 ms, p50 1.2 ms, p95 29-36 ms (one detector run) |
| Detector cadence | ≤ 4 Hz on steady frames; 534 runs over 200 s of steady footage |
| OCR per page | 0.27-0.41 s CUDA |
| Session start (reader load) | 1.3 s warm, ~5.5 s cold |
| Stop with a dwell open | one flush of ≤ 12 OCR calls (≤ ~4 s CUDA); worker join bound 5 s |
| Release | 0.09-0.12 s; GPU reserved 388-663 MB → 0-63 MB |
| RSS, one replay process | 830 MB before load → 1,120-1,230 MB after (torch + reader; released to the allocator, not the OS) |
| Threads | no growth over repeated sessions (`test_no_thread_is_left_behind_by_repeated_sessions`) |
| Processes | none spawned; everything is on the parked worker thread |

Not measured: sustained CPU % on the live Tower with a phone attached (the
replay runs the engine in a loop, not through the socket), and the
2-session RSS delta on the live Tower. Both are in the physical test.

---

## 13. Testing

**Added (Tower):** `test_document_memory_config.py` (16), `test_document_gate.py`
(24), `test_document_identity.py` (21), `test_document_segments.py` (12),
`test_document_live.py` (13), `test_document_search.py` (17). **Updated:**
`test_document_memory_engine.py`, `test_document_retrieval.py`,
`test_documents_wire_e2e.py`, `test_result_channel_protocol.py`,
`test_live_cartridge_privacy.py`, `test_new_contracts_are_documented.py`
(via the contract doc).

**Results (this worktree, canonical venv):** the Document Memory suites —
config, gate, identity, segments, live, search, dwell, engine, hostile,
retrieval, store, wire e2e, privacy, CLI, contracts-documented, result
channel protocol/hostile, architecture boundaries — **all pass**. The opt-in
real-OCR suite (`TOWER_RUN_MODEL_TESTS=1 tests/test_document_ocr_integration.py`)
passes, 14 tests, now on CUDA. Full suite: FULL_SUITE_RESULT.

**Skipped and why:** `test_document_detect_corpus.py` (24) needs
`tower/data/captures/` which lives only in the canonical checkout, so it is
skipped in every worktree; it still measures the retained-but-unused quad
detector.

**Replay:** §11.

**iOS:** `DocumentMemoryTests.swift` fixtures updated to the new
identifiers, identity, measurement; two decoder tests added
(`testTheLiveUpdateFieldsDecodeWithDefaults`, `testSightingsDecodeOnADocument`).
**Not run** — no Xcode on this machine.

---

## 14. Independent review

One independent reviewer (the subagent budget allowed one; it covered
architecture, persistence/search, capture quality, identity, privacy,
performance, code and iOS in a single pass, with reproductions under the
scratchpad). Report: `Glasses-scratch/document-memory-v1/research/08-review.md`.

**Blockers, both fixed (`fix(document-memory): what the independent review found`):**

- **A page turn could lose the previous page.** The three frames that
  confirm a content change were scored into the OLD segment's best list;
  a sharper new page evicted the old page's frames. Reproduced by the
  reviewer. Fix: disagreeing frames wait in `Dwell.pending` and seed the
  new segment; `test_a_sharper_second_page_does_not_evict_the_first`.
- **Journal rewrites used a bare `os.replace`.** Windows refuses a rename
  onto a file any reader holds open, and this branch added readers that
  wake on every write (status stamp, phone re-fetch, corpus cache). The
  reviewer measured 8/200 failed updates under one looping reader. Fix:
  `storage._replace_with_retry` everywhere the store renames, readers
  retry a refused open, and a prune with nothing to drop no longer
  rewrites; `test_update_survives_a_reader_holding_the_journal_open`.

**Should-fix, fixed:** `dwells_unreadable` was unreachable (S1) and a
flushed resighting counted as a new record (S11) — `RecordOutcome` now
carries both facts; the corpus cache served expired records until the
next write (S2) — a minute bucket in the stamp; every recorded document
rewrote the journal (S3); the idle timer ran OCR on its own thread (S4)
— it now drops an open dwell and logs it; the flush at Stop was
unbounded (S5) — four frames, newest first; a blank next page folded
into the previous page across a segment (S6); routes read with no
retention window (S8) — the Tower's configured window applies; one-char
OCR noise could testify to identity (S9).

**Should-fix, deliberately not done:** S7 (a resighting keeps the
record's pages and does not merge the dwell's additional pages) is the
documented conservative choice; S10 (the production detector is not run
against hostile surfaces in the default suite) needs the EasyOCR models,
which the default suite must not download — the replay benchmark over
real captures is that test, and it is in §11.

**Nits acknowledged, not changed:** `detect.py` and `MIN_WORDS_FOR_TEXT`
re-export are dead code retained for the corpus tests; `revision` is a
raw mtime (documented as opaque); the Release-build camera copy points at
a DEBUG-only control; document detail is still unreachable on the phone.

**Clean:** privacy (nothing new on the wire discloses paths or pixels;
the hash is a 63-bit DCT sign pattern, not invertible, not served),
import boundaries, `main.py` free of the cartridge's name, iOS syntax
and conformance as far as reading can establish.

---

## 15. Git handoff

- **Branch:** `feature/document-memory-v1`
- **Worktree:** `C:\Users\tvllo\Projects\Glasses-worktrees\document-memory-v1`
- **Base:** `6beaf57` on `integration/wb-cv-ios-validation-v1` (the same base
  the concurrent `fix/object-memory-runtime-v1` lane uses; the two lanes
  touch no common file except `docs/contracts/TOWER-UNIFIED-CARTRIDGES.md`
  and `tower/tower/main.py`, both in disjoint hunks)
- **Final HEAD:** FINAL_HEAD
- **Commits:** `8c909c1` foundation (root, gate, GPU OCR, sightings, search,
  contracts); `cf99b05` tests and the two defects they found; `baced41`
  replay benchmark and its refinements; `f728e88` iOS [BUILD UNVERIFIED];
  plus the review follow-up and this handoff.
- **Major files:** `tower/tower/document_memory/{gate,identity}.py` (new),
  `{dwell,engine,live,ocr,records,store,retrieval}.py`,
  `tower/tower/results/{document_memory,contracts,registry,__init__}.py`,
  `tower/tower/{config,cartridge_runtime,main}.py`,
  `tower/tower/routes/documents.py`, `tower/scripts/document_memory_replay.py`
  (new), `tower/docs/contracts/CARTRIDGE-RESULTS.md`,
  `docs/contracts/TOWER-UNIFIED-CARTRIDGES.md`,
  `tower/guidelines/docs/modules/DOCUMENT-MEMORY.md`,
  `ios/Glasses/Workspaces/DocumentMemory/*`, `ios/Glasses/{ProjectManager,ContentView}.swift`.
- **Scratch (disposable, not committed):**
  `C:\Users\tvllo\Projects\Glasses-scratch\document-memory-v1\` — `research/`
  (agent reports 01-07, review 08), `bench/` (OCR measurements),
  `gate-bench/` and `ocr-bench/` (agent experiments), `replay/` (real-capture
  outputs and roots), `full_suite.log`.
- **Intentionally uncommitted:** nothing in the worktree. The canonical
  checkout's untracked `orb_vocab_stella.fbow` predates this lane and was
  not touched.
- **Retained, unused:** `tower/tower/document_memory/detect.py` and its
  tests; `scripts/document_memory_benchmark.py` still sweeps landscape
  geometries. Removing both is a follow-up, not a risk.

---

## 16. Physical device test

Windows Tower + iPhone + Meta glasses. Prerequisite: the Mac lane has built
the iOS app from this branch (§13) — an app built before `f728e88` will show
Document Memory as **"needs update"** against this Tower, because the
identifiers moved.

1. **Start Tower once**, from the canonical checkout, with no
   `TOWER_DOCUMENT_*` variables set:
   `tower\.venv\Scripts\python.exe -m uvicorn tower.main:app --host 0.0.0.0 --port 8000`.
   Expect the log line `document root ...\data\document_memory (capture on,
   OCR device auto, retention 30.0 days)`.
2. **Connect iOS.** Home shows Glasses Registered, Tower Connected.
3. **Open Document Memory.** Expect the recent listing ("Never observed" on
   a fresh Tower — that is correct) and a Recording panel, NOT "has not
   declared a contract". `GET /cartridges` on the Tower shows both
   `document_memory` entries `available: true`.
4. **Verify capability:** `curl http://<tower>:8000/documents-session` →
   `session.state: "stopped"`, `reason: null`.
5. **Start.** Expect `starting` then `running` within ~2 s (cold ~6 s), the
   panel showing `OCR on cuda`, and the camera starting if it was not
   already (DEBUG build; in Release start the camera from Home first).
6. **Put a clearly printed page in view** (12-14 pt body text, a distinctive
   word on it, e.g. a printed "KUBERNETES"). Hold it so the page fills most
   of the frame height — at 360x640 the text must be ~8 px tall or larger.
7. **Hold it steady for 2-3 s.** Small hand motion is fine; walking is not.
   The panel's "pages detected" should climb while it is in view.
8. **Verify the page is captured:** look away for 2 s; within ~3 s the panel
   shows `1 recorded` and the listing gains a row (no tap needed).
9. **Verify text:** the row's title is the page's first line. Search for the
   distinctive word: one result, `page_index 0`, a snippet containing it.
10. **Library updates live:** confirmed by step 8 arriving without Recent.
11. **Search** a word that is NOT on the page: "Nothing matched".
12. **Correct result:** search a second word from the page body.
13. **Show the same page again** for 2-3 s, look away.
14. **Duplication:** the panel shows `1 recorded, 1 seen again`; the listing
    still has one row, now "Seen 2 times". `documents.jsonl` has one line.
15. **Show a second, different page** the same way.
16. **Separate record:** `2 recorded`; two rows; searching a word unique to
    page two returns only it.
17. **Pause.** State `paused`; a page shown now is not recorded.
18. **Resume.** State `running`; a page shown now is recorded.
19. **Stop.** State `stopped` within ~5 s; if a page was in view the panel
    reports `flushed_document_id`; GPU memory drops (`nvidia-smi`).
20. **Final state:** `GET /documents?limit=10` lists both records with
    `sighting_count` 2 and 1; search still answers.
21. **Second session without restarting Tower:** Start again; `OCR on cuda`
    again; show page one; expect `seen again` (a sighting on the existing
    record), not a third record.
22. **Verify it works:** the listing still has two rows.
23. **Switch to another cartridge** (CV Lab or Object Memory) and run it
    briefly. If the Document Memory session was left running, its panel
    shows "will stop itself" once no stream reaches it, and it stops after
    10 minutes; otherwise Stop it first.
24. **Return to Document Memory.** The listing re-lists on open.
25. **Data persists:** both records, sightings intact.
26. **Tower healthy:** `GET /health` 200; `nvidia-smi` shows the Tower
    process at the CUDA-context baseline (~200 MB), no
    `tower-Document-session` thread left running (`py-spy dump` or the
    thread count in `/health` if exposed); no orphan python processes.

**If anything fails, collect:** the Tower console log (every line with
`[Tower][Document]` or `[Tower][Config]`); `curl /documents-session` and
`curl /cartridges` output; `curl "/documents?limit=5"`; the phone's
Xcode console filtered on `[Glasses]`; and for a capture problem, arm the
recorder (`TOWER_CAPTURE_ROOT`) and re-run the page steps, then replay the
capture:
`python scripts/document_memory_replay.py --capture data\captures\<id> --root C:\Users\<you>\Projects\Glasses-scratch\dm-replay\<id> --format json`.
That output — stable fraction, detector runs, regions, per-frame ms — says
which stage refused the page, and is the input for re-deriving thresholds
on real paper.

---

## 17. Verdict

**DOCUMENT MEMORY V1: READY WITH KNOWN LIMITATIONS.**

Evidence for READY: a stock Tower now offers the cartridge; every Document
Memory suite and the real-OCR suite pass; on 7,124 real frames the pipeline
records nothing and on the same frames with a page composited in it records
exactly one document at 0.94-0.98 recall; a second session, a re-sighting,
Stop with a flush, release of GPU memory, and idle self-stop are each
tested; search finds a newly persisted page on the next query; the phone
refreshes on the library revision.

Known limitations, each with an owner: (1) the iOS code has not been
compiled — the Mac lane must build and run `DocumentMemoryTests` before
integration; (2) no physical page has been read through the glasses — §16
is the test, and every threshold is a policy field for re-derivation
against the first real capture; (3) at 360x640 a page must be held close;
504x896 is the measured remedy and the DEBUG resolution picker is the lever;
(4) document detail (full page text) is decoded on the phone but still not
reachable from a row — search snippets and titles show recognised text
until that view is added; (5) records are not grouped into multi-dwell
documents, by design, and the wire says so.
