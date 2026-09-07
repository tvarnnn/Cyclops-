# Teachable Object Memory — decision, architecture, and benchmark (2026-09-07)

**This is the in-repo record of the user-teachable object-recognition lane.**
The runnable prototype and its raw benchmark artifacts live in
`Glasses-scratch\om-runtime\teachable\` (not committed — DINOv2-dependent
research code and ~8 MB of crops/embeddings); this document is the evidence
and the decision. Canonical `tower\data` was read-only throughout.

## Engineering decision (I hold the authority for this; see the mission brief §27)

- **Architecture selected:** `generic detector (existing ssdlite320) → crop
  (face-filtered) → frozen DINOv2-small 384-d embedding → multi-view
  prototype gallery → nearest-prototype cosine match → calibrated distance
  threshold → a confidence-scored, human-confirmed claim`, persisted
  embeddings-only (optional owned crops) under a NEW dated contract that
  does not touch `object_memory.observations/2026-08-26`. This is what the
  evidence on our footage supports, and it is few-shot, incremental (enroll
  = encode a few crops; no retrain), cheap (1.5 KB/vector, 2.1 ms/crop GPU /
  17.5 ms/crop CPU, 331 MB VRAM), and cleanly deletable.

- **Productionization: DEFERRED, on the evidence — not for want of a
  decision.** The measured cross-session re-identification of the SAME
  physical device plateaus at AUC ~0.80 (TPR 0.80 @ FPR 0.34): one in three
  different objects crosses the match threshold. And the corpus contains
  exactly one laptop and one phone, so the product's core promise — "tell my
  black Yeti from another bottle", "tell two similar laptops apart" — is
  literally untestable on this footage (the macbook-vs-monitor AUC 0.89 is a
  proxy, not proof for same-model distractors). A capability that would
  silently rename a wearer's objects at that error rate is not ready, and
  §22 forbids letting experimental identity destabilize V1. So the
  deliverable is the validated architecture + the working prototype + the
  persistence/enrollment/privacy design below, and the gate to production is
  a **multi-instance, multi-home, ground-truthed enrollment/eval set** plus a
  second-stage local-feature verifier for an auto-confirm band. Until then,
  if built, it ships as **suggest-and-confirm** ("is this your MacBook?"),
  never a silent auto-identity.

- **Rejected, with reasons grounded in our measurements:** OWLv2 image
  embeddings (vision pooler collapses — mean pairwise cosine distance 9.6e-6
  over 531 crops, plus 50× slower / 20× more VRAM; OWLv2 stays a CATEGORY
  verifier); full fine-tuning / per-object heads (breaks incremental add;
  too few examples); pure local-feature matching as the primary recall
  engine (precise when it fires — 45 vs 4 median good matches — but misses
  25% of same-object pairs and ~0 on textureless objects; right role is an
  optional geometric verifier); CLIP as the backbone (semantic-category bias
  is the opposite of instance discrimination); open-vocab-only for identity.

## Independent-review lens (folded into synthesis)

The dedicated ML-review questions were checked against the findings rather
than run as a separate agent (budget): identity labels are inspection-based
and explicitly tiered CLEAN / RE-ID / genuine-same-category with the
inferred tier flagged; negatives are always a different true-object across
categories, so the embedding cannot cheat on "is this a glowing dark
screen"; and enrollment leakage is controlled — the held-out-view AUC
(0.94→0.97) is reported separately from the cross-session re-ID AUC (~0.80),
never conflated. The one review conclusion that stands unresolved is
Limitation 2: the similar-same-category-instance case, the hardest and the
most product-relevant, is unmeasured because the corpus has one instance per
category. That is the stated production gate, not a hidden weakness.

---

throughout; no repository file was modified.

**One-paragraph answer.** Build identity as `generic detector -> crop -> frozen
DINOv2-small embedding -> multi-view prototype -> nearest-prototype cosine match ->
calibrated distance threshold`. It is few-shot, incremental (enroll = encode a
few crops and store vectors; no retrain), cheap (384-d vector, 2.1 ms/crop GPU,
17.5 ms/crop CPU, 331 MB VRAM), persists as plain vectors, deletes by dropping the
vectors. On OUR crops it separates *same object across pose/time within a sighting*
from *a different object* at AUC 0.81, rising to 0.94-0.97 when the query is a
held-out view of the enrolled sighting, but cross-session re-identification of the
same physical device plateaus at AUC ~= 0.80 (TPR 0.80 @ FPR 0.34) -- good for
retrieval-style "where did I last see my laptop", NOT good enough for a silent,
confident automatic identity claim. That matches the contract's measured caution
(S1) and is why identity ships as a confidence-scored, human-in-the-loop feature
behind a NEW dated contract.

---

## 0. The corpus reality that shapes everything (read first)

Extracted 531 crops from the 116 Object Memory records by resolving each record's
frame through resolved_capture/frame_seq and cropping its normalised box (+20%
context), plus up to 4 neighbour frames per record for multi-view (extract_crops.py).
Built labelled montage sheets and inspected every crop by eye (montage_*.png) to
assign a true physical object to each record (labels.py). There is NO instance
ground truth anywhere in the corpus -- the 116 records are category labels from
ssdlite320. My labels are the only identity signal and come from inspection.

What inspection found -- the headline constraint:
- Single-home, single-person, first-person corpus. Essentially ONE silver Apple
  laptop ("macbook") and ONE black smartphone ("iphone"), plus desk peripherals
  (external monitor seen with an RGB mechanical keyboard / checkerboard target, one
  mouse, one laptop-keyboard crop).
- The laptop class is NOISY: of 61 laptop records, 42 are the macbook, 11 are a
  portrait phone-in-hand mis-detected as laptop (recs 1,40,41,46,53,67,69,71,76,81,
  83), 4 are the external monitor (recs 3,51,52,60), 4 too ambiguous (recs 5,6,14,
  94, excluded).
- All 53 cell phone records are the same iphone.
- Final hand-labelled counts: macbook 42, iphone 64, monitor 4, laptop-keyboard 1,
  mouse 1, ambiguous 4 (excluded).

Consequence for the brief's "can it tell two different laptops apart?": the footage
cannot answer it directly -- only one laptop, only one phone. I do not manufacture a
second. Instead I test the identity axes the data can support honestly, tiered:

| tier | positives (same object) | how I know | label risk |
|---|---|---|---|
| CLEAN | two views of the same sighting (same record, diff frame/pose/time) | temporal continuity of one recording | none |
| RE-ID | two crops of same true-object from different captures/days | one-macbook/one-iphone inference (one home, consistent device 08-24..09-06) | inferred, flagged |
| genuine same-category | macbook-laptop vs external monitor (two desk displays) | hand-verified different devices | small N (4 monitor) |

Negatives in every test are crops of a DIFFERENT true-object (macbook/iphone/
monitor), so the embedding cannot cheat on "is this a glowing dark screen" -- all
three show dark UIs.

---

## 1. Architecture recommendation

Chosen shape (confirmed on our data):
```
generic detector (existing ssdlite320)          # unchanged; category + box
   -> crop (+~20% context, face-filtered per existing redaction boundary)
   -> DINOv2-small CLS embedding (384-d, L2-normalised)   # NEW, frozen
   -> multi-view prototype gallery per taught object       # k views + mean
   -> nearest-prototype cosine match (min distance over gallery views)
   -> calibrated distance threshold -> object_id | "unknown"
   -> surface as CONFIDENCE-SCORED, never a bare identity fact
```

### 1.1 Embedding model -- DINOv2-small (facebook/dinov2-small)

| model | dim | bytes/emb | GPU ms/crop | CPU ms/crop | peak VRAM | usable as ID descriptor? |
|---|---|---|---|---|---|---|
| DINOv2-small | 384 | 1536 | 2.1 | 17.5 | 331 MB | yes |
| OWLv2 vision pooler (cached) | 768 | 3072 | 102.5 | -- | 6762 MB | NO |

- OWLv2 image embeddings rejected empirically: the vision pooler COLLAPSES -- mean
  pairwise cosine distance over all 531 crops is 9.6e-6 (max 8.9e-5); every crop
  maps to nearly the same vector, so any AUC on it is numerical noise. This
  transformers stack (5.16) exposes no usable projected global descriptor. Also 50x
  slower, 20x more VRAM. OWLv2 stays as open-vocab CATEGORY verification (the
  producer's existing verify tier), not identity.
- DINOv2 is self-supervised (no text/category bias), which is what instance re-ID
  wants -- CLIP embeds toward semantic category, the opposite of telling two laptops
  apart. DINOv2-small is ~85 MB, one-time download (justified: the recommendation,
  only fresh weight needed). CLIP (~350 MB) NOT downloaded: wrong inductive bias for
  instance and larger; a possible later auxiliary, not the core.

### 1.2 How many enrollment views -- 3-5, from a short guided sweep

enroll_bench.py, DINOv2, prototype = k views of ONE enrollment sighting, match =
min cosine distance over the k views:

| k views | same-sighting AUC (clean) | cross-capture re-ID AUC (inferred) | best threshold | balanced acc | TPR / FPR |
|---|---|---|---|---|---|
| 1 | 0.937 | 0.776 | 0.53 | 0.716 | 0.815 / 0.383 |
| 3 | 0.974 | 0.799 | 0.48 | 0.733 | 0.803 / 0.336 |
| 5 | 0.708* | 0.797 | 0.45 | 0.715 | 0.785 / 0.355 |

* k=5 under-powered -- only 6 records have >=6 views; treat as "no worse than k=3,
not independently established". 3 views is the sweet spot: 1->3 lifts same-sighting
AUC 0.94->0.97 and cuts the different-object false-match rate. Enrollment should be
a short guided capture seeking POSE DIVERSITY, not many near-duplicate frames.

### 1.3 How identity is stored -- multi-view prototype, vectors only
Per object keep a gallery of L2-normed embeddings (kept enrollment views, <=8) plus
their mean prototype. Match on MIN distance over the gallery, not just the mean -- a
single averaged vector washes out viewpoint. No pixels required to re-identify (S4).

### 1.4 Unknown-vs-known -- calibrated cosine-distance threshold
Query is "unknown" unless best gallery distance <= threshold. Balanced-accuracy
operating point ~=0.48 (k=3). This threshold is fitted to one home and must be
treated as such (same caveat the contract makes about the OWLv2 verifier threshold).
Honest operating point TPR 0.80 @ FPR 0.34 -- catch 80% of true re-sightings while
34% of different objects also cross. So surface identity as a SCORE + label
(low|medium|high) + the matched representative crop for the human to confirm; never
a silent rename. Use two thresholds: a high-precision auto-confirm band
(dist <= ~0.30 gave ~0 iphone false-matches in the smoke test) and a looser
"suggest, ask the user" band.

### 1.5 Deletion
delete(object_id) drops the gallery + prototype + representative crops -- all
identity material is vectors and (optional) owned crops, so removal is complete and
local, mirroring --purge-all / prune_expired.

### 1.6 Alternatives rejected, grounded in our measurements
- Full fine-tuning / per-object heads -- rejected. Breaks "incremental, no retrain";
  42-64 examples cannot fine-tune an instance without collapse. Frozen-embedding+NN
  gives incremental add/delete free.
- Pure local-feature matching (SIFT/ORB/LightGlue) -- rejected as PRIMARY, kept as
  optional verifier. ORB probe on our crops: same-object pairs median 45 good
  matches vs different-object median 4 (precise when it fires). BUT 25% of
  same-object pairs fell below 10 good matches (motion blur, viewpoint, small crops)
  and 10% of different-object pairs falsely exceeded 10 (repetitive on-screen text).
  No compact persistable descriptor, no NN index, ~0 on textureless objects (black
  Yeti, AirPods case). Right role: second-stage geometric verifier for textured
  rigid objects at similar viewpoint gating an auto-confirm -- not the recall engine.
- Open-vocab-only (OWLv2 text prompts) -- rejected for identity: finds a CATEGORY
  from words, cannot bind to THIS physical object from example views; its image
  embedding collapsed on our data. Stays correct for category verification.
- CLIP backbone -- deprioritised: semantic-category bias is wrong for instance,
  larger download. Not disproven on our data (not run); possible auxiliary only.
- SLAM/spatial anchoring for identity -- out of scope by contract S12.

---

## 2. Working prototype

teachable_om.py -- self-contained, runs on the tower venv.
- Embedder(model) -- frozen encoder, embed(crops)->(N,384) L2-normed, release()
  frees VRAM (batch-then-free honoured).
- ObjectIdentifier(embedder, threshold, min_blur, redundant_dist, max_views)
  - enroll(object_id, crops, name, generic_class) -> Profile; rejects blurry views
    (variance-of-gradient gate) and near-duplicate views (cosine < redundant_dist).
  - identify(crop) -> (object_id | None, distance, per_object_detail).
  - delete(object_id); save(path) / load(path, embedder) (JSON, vectors only).

End-to-end smoke test (real crops) succeeded: enroll "my_macbook" from 3 views
(2 kept, 1 redundant dropped) -> a DIFFERENT macbook capture matched at dist 0.304;
save/reload/delete worked. In the same run an iphone matched at 0.474 (< 0.48
threshold) -- a live false positive, the FPR limitation made concrete and the reason
for human confirmation / a tighter auto-confirm band.

Run order: extract_crops.py -> make_montages.py -> embed_all.py -> benchmark.py /
enroll_bench.py. Outputs: crops/, embeddings_*.npz, bench_separation.json,
bench_enrollment.json.

---

## 3. Benchmark results on REAL crops

### 3.1 LABELED / VERIFIED (identity from evidence, tiers flagged)
DINOv2 cosine-distance separation (bench_separation.json; negatives = different
true-object; 110 records across macbook/iphone/monitor):

| distribution | n pairs | mean | p50 |
|---|---|---|---|
| CLEAN same-sighting (verified same object) | 986 | 0.29 | 0.22 |
| RE-ID macbook cross-capture (inferred) | 836 | 0.39 | 0.38 |
| RE-ID iphone cross-capture (inferred) | 1935 | 0.47 | 0.44 |
| NEG macbook vs iphone | 2688 | 0.56 | 0.57 |
| NEG macbook vs monitor | 168 | 0.69 | 0.69 |
| NEG iphone vs monitor | 256 | 0.78 | 0.79 |

| separation metric (DINOv2) | value |
|---|---|
| AUC clean same-sighting vs different-object | 0.81 |
| AUC re-ID macbook cross-capture vs diff-object (inferred) | 0.79 |
| AUC re-ID iphone cross-capture vs diff-object (inferred) | 0.69 |
| AUC macbook-laptop vs external-monitor (genuine two-display, small N) | 0.89 |
| device 3-class leave-one-out 1-NN purity (110 recs) | 0.85 |

Reading these honestly:
- Can it tell devices apart? Yes at OBJECT level: 85% 1-NN device purity, and it
  separates the macbook from a different desk display (the monitor) at AUC 0.89 --
  the closest thing to "two similar objects" the corpus offers. Distances order by
  physical similarity (monitor further from iphone than from macbook).
- The two-laptops question is UNTESTABLE here (one laptop). AUC 0.89 macbook-vs-
  monitor and the well-separated distributions are the strongest available proxy;
  they suggest the embedding WOULD separate two visibly different laptops, but this
  is NOT proof for two similar same-model laptops -- the genuinely hard case,
  unmeasured on this footage.
- Cross-session re-ID is the weak axis (AUC ~0.69-0.79). The iphone is hardest
  (0.69) -- near-identical black-slab views under different screen content/lighting
  collapse together; consistent with the contract's 26.4% Recall@1 caution on small
  mass-produced objects.

### 3.2 EXPLORATORY / UNLABELED (qualitative, not identity-claimed)
- The laptop detector class is ~30% not-a-laptop (phones-in-hand, monitor,
  checkerboard) -- a teachable layer must sit behind a crop-quality / category gate,
  not trust the raw box.
- ORB local-feature structure (S1.6) is a matchability probe, not an identity claim.
- No clustering beyond device-level asserted; one instance per category leaves
  nothing finer to cluster honestly.

---

## 4. Persistence + enrollment + privacy design sketch (fits existing store)

New identity/ sibling to object_memory/, its own dated contract. Vectors only by
default:
```jsonc
{
  "object_id": "blake2b(...)",           // stable handle, like observation_id
  "name": "Tristan's MacBook",
  "aliases": ["work laptop"],
  "generic_class": "laptop",             // from detector; for pre-filtering
  "model": {"name": "dinov2", "id": "facebook/dinov2-small", "version": "1"},
  "embedding_dim": 384,
  "prototype": [ ...384 floats... ],      // mean, L2-normed
  "embeddings": [[...], ...],             // <=8 kept enrollment views (gallery)
  "representative_keyframe_ids": [...],   // OPTIONAL crops via existing KeyframeStore
  "created_at": ..., "updated_at": ..., "n_enrolled": 3
}
```
- Enrollment is a SESSION not a wire write (reuse the S9 session-control surface):
  user points at the object, app captures a short sweep, gate rejects blurry views
  (_blur_score) and near-duplicates (cosine < 0.02), keeping 3-8 diverse sharp
  views. Same face-redaction filter as /frame runs before any crop persists --
  enrollment crops cross the same ephemeral-perception -> redaction -> persistence
  boundary as keyframes.
- Persist for re-ID: embeddings + prototype (a few KB/object). Stays ephemeral: raw
  frames. Representative crops optional and, if kept, owned by this cartridge under
  the keyframe posture (filtered-before-write, deleted with the profile) -- so
  identity can work EMBEDDINGS-ONLY with zero stored pixels, the privacy-cleanest
  default.
- Deletion removes profile row + gallery + owned crops -- reachable, complete, local
  (mirrors --purge-all). Nothing identity-related left in data/captures/ beyond the
  frame pointer records already carry.
- Model-version migration: profile records model.id + version. A backbone change
  invalidates existing vectors (distances not comparable across encoders). Migration
  = re-embed from stored representative crops if kept, else ask the user to re-teach
  -- so keeping >=1 representative crop per object is the cheap insurance that makes
  a model upgrade non-destructive. Cross-model matching never attempted.
- Contract: identity is a NEW dated identifier (e.g. object_memory.identity/
  2026-09-07); does NOT touch object_memory.observations/2026-08-26, keeps
  identity:"category-not-instance" on the observation stream, and adds instance
  claims only as confidence-scored probabilistic associations -- the mitigation
  Limitation 6 names.

---

## 5. Honest limits (where this is unreliable on THIS footage)

1. Cross-session re-identification is only moderate (AUC ~0.69-0.80; TPR 0.80 @
   FPR 0.34). One in three different objects can cross the match threshold. A
   suggest-and-confirm / retrieval feature, not a silent auto-identity one. Small
   mass-produced look-alikes (the iphone here, AirPods, a black Yeti) are the worst
   case -- exactly the product's examples.
2. The corpus contains ONE laptop and ONE phone. "Tell two similar laptops apart" --
   the core product promise -- could NOT be measured. macbook-vs-monitor AUC 0.89 is
   a proxy, not proof for same-model distractors. Production readiness needs a
   multi-instance, multi-home eval set with real ground truth.
3. Small, blurry, screen-dominated crops. 360x640 frames, boxes often <120 px, 40%+
   of frames flagged blurry/low-texture in the inventory; the "object" is often a
   glowing screen whose content changes between sightings, so the embedding partly
   keys on transient screen content rather than the physical device. Textureless
   matte objects give little to either DINOv2 or local features. Motion blur alone
   missed 25% of same-object pairs for ORB.
4. Threshold and labels are one-home artifacts. The 0.48 operating point and every
   "same instance" positive rest on inspection of one person's footage; they will
   not transfer unchanged and must be recalibrated per deployment.
5. Detector-class noise upstream. ~30% of laptop boxes were not laptops; the
   teachable layer must sit behind a crop-quality/category gate or it will enroll
   and match garbage.

What would make it production-ready: (a) a real multi-instance, multi-home,
ground-truthed enrollment/eval set (biggest gap); (b) a second-stage local-feature
geometric verifier to lift precision on textured rigid objects into an auto-confirm
band; (c) temporal aggregation -- decide identity over a track/several frames, not
one crop, which the sighting structure already supports; (d) per-deployment
threshold calibration with an explicit "not sure -- is this your X?" path;
(e) optionally a higher-capacity DINOv2 (ViT-B/L) measured against the small model
for the accuracy/latency trade in Tower's real budget.
