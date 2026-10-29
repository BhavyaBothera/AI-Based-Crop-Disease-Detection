# PHYTOSIGNAL — Non-Leaf Input Safety Audit

**Report date:** 29 October 2026  
**Scope:** Why an arbitrary photograph of a person received a tomato Late Blight diagnosis, and what was implemented to prevent unsupported images from being presented as crop disease results.  
**Status:** Implemented safeguard + held-out evaluation completed. No model retraining, no dataset split change, no deployment.

---

## 1. Executive summary

### 1.1 What happened

A photograph containing no plant material received:

- `Tomato — Late Blight`
- `100.00%` primary score
- a top-3 candidate list
- a full disease monograph and IPM advisory

The disease model is a **closed-world 15-class softmax classifier**. It always returns a
distribution over its 15 trained classes, so its argmax is not evidence that the input is a
supported crop leaf. Nothing between the file and the UI checked image *content*. The
`100.00%` figure was a raw softmax output inside that closed set, presented under the label
"Primary Prediction Confidence".

### 1.2 What was implemented

A deterministic, model-independent **content-level input suitability gate** now runs after
physical file validation and **before inference**:

- **Rejected** when the image contains essentially no plant pigmentation. Inference is not
  run and a structured `422` response with `status: "rejected"` is returned.
- **Abstained** when plant material is present but the frame is sparse, scattered,
  underexposed or too featureless to assess. Same non-diagnostic contract with
  `status: "abstained"`.
- **Accepted** only when plant-tissue evidence is present in a framed, usable photograph;
  existing disease prediction is unchanged.

The UI renders a dedicated PHYTOSIGNAL rejection state with no class, candidate, score,
knowledge base or advisory controls, and a direct "Upload another image" action.

### 1.3 Headline measurements

| Measurement | Result | Sample |
|:--|:--|:--|
| Valid leaves receiving **no diagnosis** (held-out evaluation half) | **2 / 1,657 = 0.12%** (2 abstentions, 0 hard rejections) | PlantVillage validation split |
| Valid leaves receiving no diagnosis (calibration half) | **0 / 1,761 = 0.00%** | PlantVillage validation split |
| Real non-leaf photographs **rejected/abstained** | **249 / 558 = 44.6%** | Excluded `Background_without_leaves` photographs |
| Real non-leaf photographs **falsely accepted** | **309 / 558 = 55.4%** | Same |
| Unsupported crop leaves accepted (documented limitation) | **475 / 480 = 99.0%** | Excluded Apple/Corn/Grape/Peach/Soybean/Strawberry leaves |
| Gate latency, assess only | mean **12.1 ms**, p95 **18.5 ms** | 150 validation leaves |
| Programmatic non-plant probes rejected/abstained | **7 / 7** | Person, animal, building, document, object, screenshot, uniform |

**This is not a leaf detector and it does not solve OOD detection.** It eliminates a specific,
demonstrated class of false diagnosis (images with no plant tissue) at no measurable cost on
the validation leaves, and it abstains instead of guessing elsewhere. Foliage-heavy outdoor
scenes and unsupported crop species still reach the classifier; those limits are quantified in
§7 and are **not** claimed as solved.

---

## 2. Root cause

### 2.1 Complete request path (before the fix)

```
Upload
  → content-type allow-list + magic-byte signature + streamed 10 MB cap
  → Pillow decode, pixel ceiling, EXIF transpose, 64 px floor, RGB conversion
  → uniform-canvas check (luminance sigma ≤ 2.0)
  → resize 224×224, float32 [0, 255]
  → MobileNetV2(15-class softmax)
  → argmax rendered as "Tomato — Late Blight", score ×100 as "confidence"
  → top-3 candidates + knowledge base + advisory export
```

The existing safety controls are real but **physical**, not semantic:

| Control | What it proves | What it does not prove |
|:--|:--|:--|
| MIME + magic bytes | The bytes are a JPEG/PNG/WEBP | The pixels show a leaf |
| 10 MB / 20 MP / 64 px | Resource bounds | Subject matter |
| Decode / scanline verification | File is intact | Subject matter |
| Luminance σ ≤ 2.0 | Not a flat canvas | A portrait, document or scene passes easily |
| Softmax score | Relative position within 15 classes | Anything about out-of-distribution inputs |

The uniform-canvas rule is the only content-adjacent check. A person photograph has normal
luminance variance, so it passed.

### 2.2 Contributing product causes

1. **No abstention contract.** `PredictionResponse` had no way to express "no diagnosis"; a
   successful HTTP response was always a full diagnosis.
2. **Misleading label.** A raw closed-set softmax was labelled "Primary Prediction
   Confidence", inviting users to read 100% as certainty.
3. **Frontend rendered any 200 response and retained state on failure.** `lastPrediction`
   and `currentActiveKnowledgeData` were not cleared when a request failed, so a previous
   diagnosis and advisory could survive a later failure in application state (the results
   card was hidden, but state and DOM nodes remained).
4. **No UI rejection state.** Upload/format errors used an alert banner; there was no
   deliberate, non-diagnostic result state.

### 2.3 Why frontend file validation cannot fix this

`accept="image/jpeg,..."`, MIME checks and size checks happen before the bytes are decoded and
are trivially satisfied by a portrait photograph. Content safety must be enforced server-side,
after decoding, where the tensor that the model would see is available.

---

## 3. Implemented safeguard architecture

```
[ Upload ]
   │
   ▼
Layer A — physical integrity (unchanged)
   MIME/format agreement, streamed size cap, decode, 20 MP, 64 px, RGB, σ > 2.0
   │ pass
   ▼
Layer B — content suitability gate (new, deterministic)
   On the same 224×224 RGB frame the model will see
   │
   ├── reject  (no diagnosis) ──► structured 422 ──► rejection UI
   ├── abstain (no diagnosis) ──► structured 422 ──► rejection UI
   └── pass ───────────────────► Layer C
                                    │
                                    ▼
Layer C — existing 15-class MobileNetV2 inference
   └── 200 PredictionResponse (contract unchanged) ──► existing diagnosis UI
```

Layer B uses **no model confidence**, no second neural network, no external API and no
additional download. It is deterministic, CPU-only and runs in the existing decode worker
thread under the existing admission-control semaphore.

### 3.1 Decision rules

Implemented in `backend/app/services/input_gate.py`; thresholds live in
`backend/app/config.py` with measured provenance.

| # | Rule | Condition | Decision | Code |
|:--:|:--|:--|:--|:--|
| 1 | Underexposed | mean luminance < 0.05 | abstain | `UNDEREXPOSED_IMAGE` |
| 2 | No plant evidence | plant-pigment fraction < 0.01 | **reject** | `NON_LEAF_IMAGE` |
| 3 | No detail | Laplacian variance < 1e-4 | abstain | `LOW_DETAIL_IMAGE` |
| 4 | Sparse | 0.01 ≤ plant fraction < 0.08 | abstain | `SPARSE_PLANT_EVIDENCE` |
| 5 | Scattered | ≥ 3 qualifying regions **and** largest region < 2% | abstain | `SCATTERED_PLANT_EVIDENCE` |
| 6 | Otherwise | — | accept → inference | `SUPPORTED_LEAF_INPUT` |

**Plant-pigment mask:** HSV hue 40–170°, saturation ≥ 0.12, value ≥ 0.08. This band covers
green and chlorotic foliage while excluding skin (≈ 20–35°), wood (≈ 25–40°), beige walls,
paper (near-zero saturation), concrete and cyan/blue sky (≈ 190–220°). It is a
*pigmentation presence* measure, not a leaf-shape or species measure.

**Sparse/scattered** are abstentions, not rejections: they say "plant material may be present,
but this is not a framed specimen", which is the honest claim.

### 3.2 Threshold provenance

Thresholds were fixed before the synthetic evaluation probes were generated. The calibration
sweep below is computed on the calibration half only; the disjoint evaluation half is reported
in §5.

From `non_leaf_input_safety_eval.json` (calibration half, plant-evidence rule only):

| Threshold | Calibration leaves below | Calibration real non-leaf photographs below |
|--:|--:|--:|
| 0.005 | 0.00% | 10.6% |
| 0.010 | 0.00% | 15.9% |
| 0.020 | 0.00% | 21.5% |
| 0.030 | 0.00% | 26.7% |
| 0.050 | 0.00% | 33.0% |
| **0.080** (accept line) | **0.00%** | **42.9%** |
| 0.100 | 0.23% | 47.0% |
| 0.150 | 0.97% | 58.0% |

The operating point (`reject < 0.01`, `accept ≥ 0.08`) keeps every calibration leaf, while a
0.08 accept line already costs valid leaves (0.23% at 0.10, 0.97% at 0.15). No threshold can
reject foliage scenes reliably; see §7.

### 3.3 API contract (backward compatible)

**Success (unchanged, HTTP 200):** the existing `PredictionResponse` fields — `predicted_class_id`,
`raw_class`, `crop`, `condition`, `health`, `score`, `top_predictions`, `all_scores`,
`disclaimer`. Existing clients continue to work unchanged.

**Rejected/abstained (new, HTTP 422):** a machine-readable detail object, deliberately shaped
so it cannot be mistaken for a prediction:

```json
{
  "detail": {
    "status": "rejected",
    "code": "NON_LEAF_IMAGE",
    "category": "non_leaf_image",
    "message": "We couldn't verify that this image contains a supported crop leaf.",
    "detail": "No plant pigmentation was detected. ...",
    "retryable": true,
    "signals": { "plant_evidence": 0.0, "dominant_region": 0.0, "region_count": 0,
                 "mean_luminance": 0.3375, "detail_variance": 0.002069 }
  }
}
```

There is no `predicted_class_id`, `crop`, `condition`, `health`, `score`, `top_predictions`
or `all_scores` anywhere in this response. HTTP 422 means old clients fail **safe** (they show
an error, never a diagnosis).

| Outcome | HTTP | Client distinction |
|:--|--:|:--|
| Valid prediction | 200 | Result contract present |
| Unsupported / no plant tissue | 422, `status=rejected` | `InputRejectedError`, category |
| Ambiguous / low quality | 422, `status=abstained` | `InputRejectedError`, category |
| Invalid upload (format, size, dimensions, blank canvas) | 400/413/415/422 string detail | Generic `ApiError` |
| Model/service unavailable or internal failure | 503/500 | Generic `ApiError` |

`rejected` and `abstained` share the same safe UI, but remain distinguishable for clients and
monitoring.

### 3.4 Efficiency and concurrency

- The gate reuses the decoded 224×224 frame; there is no second decode and no extra upload read.
- It runs inside the existing `asyncio.to_thread` decode task, under
  `MAX_DECODE_CONCURRENCY`, before the inference semaphore; the event loop is not blocked.
- Rejected inputs never enter inference, so they consume no model time and do not delay valid
  requests.
- No new model, weights, package, network call or file download. Connected-component analysis
  is a self-contained run-length union-find (no SciPy dependency).
- Measured assess-only latency: mean 12.1 ms, median 10.3 ms, p95 18.5 ms (n = 150). The
  inference stage is an order of magnitude larger; the gate adds roughly 12% to total latency
  on CPU.
- `ENABLE_INPUT_GATE=false` is a documented rollback switch (matching the existing
  `ENABLE_LEVEL_A_PREVALIDATION` pattern). It was tested but is not intended for routine use.

### 3.5 UI behaviour

A new `#results-rejected` panel inside the results card renders:

- the exact required explanation, "We couldn't verify that this image contains a supported crop leaf.";
- a category badge: *No crop leaf detected* / *Image could not be verified* / *Image quality too low*;
- a specific reason and capture guidance;
- **Upload another image** (returns to the upload state) and the existing **New Scan**;
- a note that no diagnosis, candidate list or treatment guidance was produced.

On rejection the UI also:

- hides the diagnosis banner, top-3 list, agronomic reference and disclaimer;
- hides **Print** and **Copy Advisory**;
- clears `lastPrediction`, active candidate, cached knowledge and the rendered candidate DOM;
- preserves the preview and the ability to retry the same image.

Selecting a new file, starting a new analysis, or receiving any failure clears diagnostic state
first, so a previous result cannot survive a rejection.

---

## 4. Files modified / added

| File | Change |
|:--|:--|
| `backend/app/services/input_gate.py` | **New.** Deterministic gate: plant mask, connected-region statistics, exposure/detail checks, decision records, structured error payload |
| `backend/app/config.py` | Gate thresholds + `ENABLE_INPUT_GATE`, each documented with the data used |
| `backend/app/api/predict.py` | Decode + assess in one worker task; reject/abstain before inference with structured 422; OpenAPI 422 description |
| `backend/app/services/__init__.py` | Export `assess_image`, `InputAssessment` |
| `backend/tests/test_input_gate.py` | **New.** 26 unit/endpoint tests for decisions, contract and rejection safety |
| `backend/tests/non_leaf_fixtures.py` | **New.** Deterministic synthetic person/animal/building/document/object/screenshot/plant probes |
| `backend/tests/image_fixtures.py` | Synthetic fixtures gained fine surface texture (a pure gradient is correctly treated as low-detail) |
| `frontend/js/api.js` | `InputRejectedError`, structured-detail parsing, stricter success-contract validation |
| `frontend/js/app.js` | Rejection renderer, full state/DOM clearing, distinct error routing, rejection action |
| `frontend/index.html` | Rejection state markup; score label now "Primary Model Score (Softmax)" |
| `frontend/css/styles.css` | Rejection panel styling (desktop/mobile/print), sticky-header scroll margin |
| `frontend/test_api.js` | +7 tests for structured rejection, abstention, invalid upload, service error, contract validation |
| `frontend/test_browser.py` | **New.** Headless Chrome E2E for desktop and mobile; screenshots |
| `ml/scripts/evaluate_input_gate.py` | **New.** Held-out evaluation harness producing the JSON artifact |
| `docs/audit_reports/non_leaf_input_safety_eval.json` | Generated evaluation artifact |
| `docs/audit_reports/non_leaf_safety_screenshots/` | Desktop/mobile evidence screenshots |
| `README.md`, `docs/README.md` | Updated safety description, test counts, known limitations |

**Unchanged:** all 15 class IDs and mappings, model architecture and weights, train/validation/
test split, existing Layer A validation, CORS, upload limits, inference serialization.

---

## 5. Dataset and evaluation methodology

### 5.1 Data used

| Category | Source | n | Role |
|:--|:--|--:|:--|
| Supported leaves | `val` split, 15 classes | 3,418 | No-diagnosis rate |
| Real non-leaf photographs | `excluded` → `Background_without_leaves` | 1,143 | False acceptance |
| Unsupported crop leaves | `excluded` → 6 non-target crops | 480 (80/class) | Documented limitation |
| Synthetic probes | Generated deterministically after thresholds were fixed | 15 | Branch coverage / UI |
| **Held-out test split** | `split == 'test'` | **Never opened** | — |

The evaluation script asserts the test-split row count without reading images or labels for
model selection; the disease model is not evaluated or tuned here at all.

### 5.2 Calibration / evaluation separation

Every source category is partitioned by SHA-256 hash of the image path into a calibration half
and a disjoint evaluation half. The sweep in §3.2 uses the calibration half only. Reported
metrics in §5.3 use the evaluation half. Synthetic probes were created after the thresholds
were frozen; no probe was used to select a threshold.

### 5.3 Honest provenance caveat

The 1,143 real background photographs were **inspected while choosing the operating point**
(aggregate distributions, plus a handful of images). Their evaluation numbers are therefore an
**exploratory diagnostic**, not a clean, untouched benchmark. The synthetic probes are
genuinely post-hoc, but they are drawings, not photographs. **The repository contains no clean
held-out real-photograph OOD benchmark.** This is a limitation of the evidence, not something
the numbers below can fix.

### 5.4 Metric definitions

- **No-diagnosis rate** = (rejected + abstained) / n. This is the user-visible safety outcome
  and the primary valid-leaf metric.
- **Hard rejection rate** = rejected / n (the system positively asserted "no plant tissue").
- **False acceptance rate (FAR)** = accepted / n for non-leaf categories. Every accepted
  non-leaf image receives a disease diagnosis, so FAR is the residual product risk.
- Decode failures are recorded as rejections (Layer A) and are not double-counted.

---

## 6. Results

### 6.1 Held-out / disjoint evaluation

| Category | n | Accepted | Rejected | Abstained | No diagnosis | FAR |
|:--|--:|--:|--:|--:|--:|--:|
| Supported leaves (val evaluation half) | 1,657 | 1,655 | 0 | 2 | **0.12%** | — |
| Real non-leaf photographs (evaluation half) | 558 | 309 | 88 | 161 | **44.6%** | **55.4%** |
| Unsupported crop leaves (6 species) | 480 | 475 | 1 | 4 | 1.0% | 99.0% accepted |

Valid-leaf reasons: 2 × `SPARSE_PLANT_EVIDENCE` (no hard rejections).  
Non-leaf reasons: 88 × `NON_LEAF_IMAGE`, 160 × `SPARSE_PLANT_EVIDENCE`,
1 × `SCATTERED_PLANT_EVIDENCE`.  
Unsupported-crop reasons: 4 × `SPARSE`, 1 × `NON_LEAF` (5 coincidental failures, not a species
check).

Supporting full-split measurement (context only): across all 3,418 validation leaves, 0 images
fall below the 0.02 reject line and 2 (0.06%) below the 0.08 accept line.

### 6.2 Programmatic probes

| Probe | Category | Result | Code |
|:--|:--|:--|:--|
| Person | non-leaf | rejected | `NON_LEAF_IMAGE` |
| Animal | non-leaf | rejected | `NON_LEAF_IMAGE` |
| Building | non-leaf | rejected | `NON_LEAF_IMAGE` |
| Document | non-leaf | rejected | `NON_LEAF_IMAGE` |
| Random object | non-leaf | rejected | `NON_LEAF_IMAGE` |
| Screenshot with small green accent | non-leaf | abstained | `SPARSE_PLANT_EVIDENCE` |
| Uniform constant colour | non-leaf | rejected by Layer A | blank canvas |
| Leaf close-up | leaf | accepted | `SUPPORTED_LEAF_INPUT` |
| Multiple leaves | leaf | accepted | `SUPPORTED_LEAF_INPUT` |
| Leaf on complex/field background | leaf | accepted | `SUPPORTED_LEAF_INPUT` |
| Sparse vegetation | ambiguous | abstained | `SPARSE_PLANT_EVIDENCE` |
| Scattered foliage | ambiguous | abstained | `SCATTERED_PLANT_EVIDENCE` |
| Tiny leaf | ambiguous | abstained | `SPARSE_PLANT_EVIDENCE` |
| Dark leaf | low quality | abstained | `UNDEREXPOSED_IMAGE` |
| Blurry leaf | low quality | abstained | `LOW_DETAIL_IMAGE` |

Each probe is a single deterministic drawing. These rows demonstrate branch behaviour; they are
**not** a statistical estimate of performance on real people, documents or screenshots.

### 6.3 Why the reported bug is fixed

The reported image was a photograph of a person. Such an image contains skin, hair, clothing
and background with no plant-pigment pixels; in the gate's own terms it is at or near
`plant_evidence = 0`, far below the 0.01 reject line. It is rejected **before** the model is
invoked, so no softmax distribution exists to display. The same path produced a structured
rejection in the automated tests, the real non-leaf photographs, and the browser flow.

**The exact photograph supplied in the bug report was not present in the repository**, so it
could not be replayed byte-for-byte. To confirm it, place the file in the project and submit it
through the UI or:

```bash
curl -i -F "file=@/path/to/the_photo.jpg" http://127.0.0.1:8000/api/predict
```

Expected: HTTP 422 with `detail.status = "rejected"` and `detail.code = "NON_LEAF_IMAGE"`.

---

## 7. Remaining limitations

### 7.1 Residual risk (quantified)

1. **Foliage-heavy scenes are still accepted.** 55.4% of the real background photographs pass
   because they contain grass, trees or foliage. A wide garden or landscape photo can pass this
   gate and reach the classifier. Distinguishing a leaf close-up from a vegetation scene needs
   semantics this gate does not have.
2. **Unsupported crop species are not detected.** 99.0% of Apple/Corn/Grape/Peach/Soybean/
   Strawberry leaves are accepted and diagnosed as one of the 15 Solanaceae classes. This is the
   previously documented closed-world limitation; the gate makes it neither better nor worse.
3. **Colour shortcuts.** A person in a large green field, a green-painted wall, a graphic with
   substantial foliage-coloured regions, or a heavily green-tinted screenshot can pass. A
   photographed dead/brown leaf with almost no green could be rejected (the gate treats it as
   non-plant); measured cost on the dataset is 0 hard rejections, but field imagery is not the
   dataset.
4. **PlantVillage is not field data.** The validation leaves are close-up, uniform-background
   photographs. The 0.12% no-diagnosis rate is measured there, not on farmer field photography.
   Complex backgrounds, soil, shadows and hands are under-represented.
5. **No trained leaf detector.** The repository has no labelled non-leaf corpus of people,
   animals, documents, screens, buildings and objects. Training a gate on PlantVillage
   backgrounds and other crop leaves would learn "background vs foliage", not "leaf vs
   anything", and would not be trustworthy for the reported failure mode.
6. **Synthetic probes are weak evidence.** They check branches and UI behaviour, not
   generalisation.
7. **Softmax scores remain uncalibrated.** They are still displayed, now labelled "Primary Model
   Score (Softmax)". Temperature scaling would change the numbers but not the closed-world
   problem, and is not an acceptance criterion.

### 7.2 What the gate must not be advertised as

- Not a leaf detector.
- Not an unsupported-species detector.
- Not OOD detection.
- Not production-ready protection on the strength of 558 real negative photographs.

---

## 8. Testing

### 8.1 Automated suites

| Suite | Command | Result |
|:--|:--|:--|
| Backend (75 tests) | `.venv/Scripts/python.exe -m unittest discover -s backend/tests -p "test_*.py"` | **OK**, 75 passed |
| Frontend API client (35 tests) | `node --test frontend/test_api.js` | **35 passed** (3 live tests need a running backend; they skip when offline) |
| Browser E2E (desktop 1440×900 + mobile 390×844) | `.venv/Scripts/python.exe frontend/test_browser.py --static-port 5510` | **OK** both viewports |
| Held-out gate evaluation | `.venv/Scripts/python.exe ml/scripts/evaluate_input_gate.py` | **OK**, artifact written |

New coverage includes: non-leaf categories, abstention categories, no diagnosis/treatment/
advisory on rejection, no inference on rejection, queue bookkeeping, rollback flag, malformed
success payloads, structured error parsing, stale-result clearing, and the exact UI assertions
that no disease term, candidate card, print or copy control appears in a rejection.

### 8.2 Browser evidence

Screenshots in `docs/audit_reports/non_leaf_safety_screenshots/`:

| File | State |
|:--|:--|
| `desktop-person-rejected.png` | Desktop rejection state |
| `mobile-person-rejected-viewport.png` | Mobile rejection state below sticky header |
| `desktop-supported-leaf.png`, `mobile-supported-leaf-viewport.png` | Unchanged valid prediction |
| `desktop-stale-result-cleared.png`, `mobile-stale-result-cleared.png` | Previous diagnosis replaced by rejection |

The page was inspected at both viewports. The rejection panel uses existing PHYTOSIGNAL tokens,
keeps the brand identity, and is readable and fully visible below the sticky header on mobile.

### 8.3 Checks that could not be performed

- **The exact reported photograph** was not available in the repository (see §6.3).
- No real-photograph benchmark of people, animals, documents, buildings, screenshots and
  field leaf photos exists in the repository. Collecting one is the primary prerequisite for the
  proposal in §9.2.

---

## 9. Deployment recommendations

### 9.1 Ship the implemented safeguard

1. Deploy with `ENABLE_INPUT_GATE=true` (default). Keep Layer A unchanged.
2. Keep the structured 422 contract; do not collapse reject/abstain into a generic error.
3. Monitor, without images or personal data:
   - rejection/abstention rate and reason-code mix,
   - `non_leaf_image` share by client/device,
   - any user reports of a valid leaf being declined.
4. Review the UI wording with an agronomist: "couldn't verify" must not imply the leaf is
   healthy or that the image is worthless.
5. Do not market the gate as leaf detection or OOD detection.

### 9.2 Proposed path to a trustworthy gate (not implemented)

1. **Collect a dedicated evaluation corpus** under a documented licence and consent process:
   ≥ 2,000 real portraits, ≥ 1,000 animals, ≥ 1,000 documents/screens, ≥ 1,000 buildings/
   objects/screenshots, and ≥ 3,000 real field leaf photographs (supported and unsupported
   species, soil, hands, shadows, multiple subjects, blur, rain, night).
2. **Split by capture group and source**, never by random image, to avoid near-duplicate
   leakage (the discipline already used for the disease dataset).
3. **Train a dedicated gate** (e.g., MobileNetV3-Small or EfficientNet-Lite, ~2–5M params,
   separately from the disease model) with a "supported leaf / other plant / non-plant /
   unusable" head; evaluate species rejection only on held-out species.
4. **Set the operating point for a target false-rejection budget** (e.g., ≤ 1% valid-leaf
   abstention at the best achievable FAR), with confidence intervals by category.
5. **Shadow-mode deploy** the gate (log-only) for 4–8 weeks, then enforce. Keep deterministic
   Layer A and this Layer B as defence in depth.
6. **Calibrate display scores** separately (temperature scaling) and never use them as the
   acceptance criterion.
7. **Retrain/replace the disease model only with documented approval**, as required by this
   task's boundaries; nothing here changes it.

---

## 10. Implemented vs proposed

| Item | Status |
|:--|:--|
| Non-plant inputs rejected before inference | **Implemented** |
| Explicit abstention for sparse/scattered/dark/blurry inputs | **Implemented** |
| Machine-readable rejected/abstained API contract | **Implemented** |
| Dedicated UI rejection state, no diagnosis/advisory/export | **Implemented** |
| Stale diagnosis cleared on rejection/new file/failure | **Implemented** |
| Deterministic Layer B gate with calibrated, documented thresholds | **Implemented** |
| Docstring/README/audit documentation of limits | **Implemented** |
| Trustworthy learned leaf/specimen detector | **Proposed** (needs data; §9.2) |
| Unsupported-species rejection | **Proposed** (existing centroid approach audited and rejected) |
| Field-photograph validation | **Proposed** (needs field data) |
| Production OOD certification | **Not claimed** |
