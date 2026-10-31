# PHYTOSIGNAL — Frontend Reimagination & Multi-Page Architecture Report
**Product:** PHYTOSIGNAL — Autonomous Foliar Pathology & Agronomic Intelligence Platform  
**Target Taxa:** Solanaceae (*Solanum lycopersicum*, *Solanum tuberosum*, *Capsicum annuum*)  
**Architecture:** Multi-Page Vanilla Web Architecture (HTML5, CSS3 Custom Properties, Vanilla ES6+ Modules, FastAPI Backend)  
**Audit Date:** October 2026  
**Status:** Verification Completed & Production-Ready  

---

## Executive Summary

The frontend of **PHYTOSIGNAL** has been completely reimagined and rebuilt from the ground up as a commercial-grade, multi-page AgriTech product. Transcending conventional dashboard clichés, student project templates, and generic SaaS patterns, the new interface embodies the aesthetic of **high-end scientific instrumentation, contemporary editorial typography, and disciplined agricultural intelligence**.

The brand positioning:
> **"Plant intelligence. Made actionable."**  
> *AI-assisted crop leaf analysis for tomato, potato and bell pepper.*

The platform now operates as a true multi-page application with four distinct URL routes, a shared telemetry-enabled navigation system, a dedicated diagnostic studio, a searchable foliar pathology field guide with slide-over clinical monographs, and an honest technical monograph detailing model capabilities and real-world boundaries.

Every backend endpoint contract, machine-learning safety gate, two-tier leaf acceptance pipeline, and regression test has been preserved with zero regressions across 156 automated test suites.

---

## 1. Non-Negotiable Design Direction

### 1.1 Aesthetic Philosophy
The visual direction bridges the gap between field agronomy and computational laboratory science:
- **Scientific Instrumentation:** Precision viewfinders, calibrated reticles, dimensional metadata callouts, and clean specimen framing.
- **Contemporary Editorial Design:** High optical weight headlines paired with disciplined geometric sans and monospace metrics.
- **Field Usability:** High contrast, glare-resistant dark surfaces, legible type scales, and touch targets calibrated for field use.
- **Restraint:** Zero gratuitous glassmorphism, zero floating purple gradients, zero meaningless glowing orbs, and zero fake AI processing animations.

### 1.2 The Three-Color Brand System
The interface strictly enforces the three signature brand colors across all surfaces, components, and interaction states:

| Color Name | Hex Token | Visual Role | Application |
|---|---|---|---|
| **Obsidian** | `#111318` | Primary dark surface, navigation foundation, specimen backdrops | Header, canvas backgrounds, terminal panels, dark cards, deep framing borders |
| **Signal Coral** | `#FF5A36` | Primary actions, key interaction highlights, critical status accents | Primary action buttons (`Analyze Leaf`, `Scan Specimen`), active state pills, alert badges, target reticle crosshairs |
| **Mineral Ice** | `#B9E4DC` | Scientific visualization, analytical accents, selected states | Telemetry indicators, specimen metadata highlights, differential candidate bars, secondary buttons, botanical borders |

Supporting neutrals include `#1A1D24` (surface elevation), `#262B35` (structural borders), `#8E95A5` (secondary botanical metadata), and `#F0F3F8` (high-contrast text).

Semantic statuses (disease warning, leaf gate rejection, healthy verification) maintain accessible meaning through iconography, borders, and high-contrast labeling without distorting the core three-color identity.

### 1.3 Typography System
Four complementary typefaces establish an editorial hierarchy:
- **Editorial Display:** `Newsreader` (optical serif) — Used for hero statements, scientific monographs, and agronomic disease titles.
- **Brand Wordmark & Section Headers:** `Space Grotesk` (grotesque geometric sans) — Used for product wordmarks, section titles, and action cards.
- **UI & Reading Flow:** `Inter` (neutral UI sans) — Used for interface controls, body paragraphs, and diagnostic recommendations.
- **Telemetry & Specimen Metadata:** `JetBrains Mono` (technical monospace) — Used for neural scores (`99.87%`), specimen identifiers (`#SOL-POT-002`), latency metrics (`< 85ms`), and file dimensions.

---

## 2. Multi-Page Architecture

Rather than a single-page anchor hack, PHYTOSIGNAL is structured as four distinct pages with dedicated URL routes, browser history support, active route awareness, and responsive mobile navigation drawers:

```
PHYTOSIGNAL/
├── / (index.html)             Flagship Editorial Introduction & Live Quick Scanner
├── /analyze (analyze.html)    Dedicated Diagnostic Studio & Full Clinical Monograph
├── /field-guide (field-guide.html) Pathology Encyclopedia & Interactive Monograph Drawer
└── /about (about.html)        Architecture, Two-Tier Gate, & Responsible AI Monograph
```

### Page 01: Home (`/` or `index.html`)
The product's editorial front door introduces agronomists and researchers to the platform:
- **Editorial Hero Composition:** Asymmetric layout featuring bold headlines (*"Every leaf signals distress before a crop fails."*), product descriptor, primary CTA (*"Analyze a Leaf"*), secondary link (*"Explore Field Guide"*), and quick diagnostic metrics (`15` classes, `96.86%` test macro-F1, `< 100ms` inference).
- **Art-Directed Specimen Showcase:** High-resolution botanical specimen photograph (`frontend/samples/hero_specimen.jpg`) framed in an instrument viewfinder with real-time target reticle and species taxonomy tag (*Solanaceae: Solanum tuberosum*).
- **Operational Workflow:** 3-step scientific protocol: (1) Standardized Foliar Ingestion, (2) Dual-Gate Validation (Leaf & Species Gate), and (3) Differential Inference & Agronomic Guidance.
- **Crop Scope Matrix:** Visual cards detailing coverage for Tomato (9 conditions), Potato (3 conditions), and Bell Pepper (2 conditions + healthy state).
- **Embedded Diagnostic Scanner:** Complete working upload dropzone and sample presets embedded directly on the homepage for immediate laboratory testing.

### Page 02: Dedicated Diagnostic Studio (`/analyze` or `analyze.html`)
The operational core of the product designed as a scientific scanning station:
- **Specimen Dropzone:** Calibrated drag-and-drop ingestion area supporting JPEG, PNG, WEBP up to 10 MB and 20 MP. Features browse file trigger, camera capture trigger, and clear quality instructions.
- **Quick Test Presets:** Four verified foliar specimens (Potato Early Blight, Tomato Bacterial Spot, Pepper Healthy, Tomato Yellow Leaf Curl) for rapid testing.
- **Telemetry Readout:** Real-time backend connectivity and model readiness polling displaying live status (*"Engine Ready"*, latency).
- **Specimen Studio:** Dual-column results layout featuring the uploaded image with zoom/reset controls, file dimensions, and timestamp metadata.
- **Differential Candidates:** Top-3 candidate differential diagnoses ranked by confidence with comparative bar charts.
- **Agronomic Monograph Canvas:** High-contrast reading surface rendering validated symptoms, cultural prevention, chemical interventions, biological controls, and scientific citations.
- **Action Tools:** One-click `Copy Advisory` with animated toast feedback and `@media print` clean report output.

### Page 03: Field Guide (`/field-guide` or `field-guide.html`)
An interactive Solanaceae pathology catalog powered by `frontend/js/knowledge_data.js`:
- **Real-Time Client Search:** Instant substring search querying disease names, causal pathogens, and symptoms.
- **Multi-Facet Filtering:** Crop buttons (All, Tomato, Potato, Bell Pepper) and condition toggles (All Conditions, Diseased Only, Healthy Only).
- **Pathology Cards:** 15 structured cards showing common names, Latin pathogen names, host taxonomy, and symptom summaries.
- **Slide-Over Monograph Drawer:** Clicking any card opens a full-height side drawer detailing complete management protocols, cultural practices, chemical guidelines, and research citations without leaving the page.
- **Keyboard & Touch Accessibility:** Drawer dismissible via `Escape` key, click-outside, or close button.

### Page 04: About & The Technology (`/about` or `about.html`)
An honest, peer-review-grade technical monograph documenting the system architecture:
- **Diagnostic Engine:** MobileNetV2 convolutional backbone fine-tuned across 15 Solanaceae conditions.
- **Two-Tier Safety Gate:** Detailed explanation of why unconstrained softmax networks cannot be deployed in open fields. Documents the Level-A OOD input gate that intercepts human portraits, non-plant objects, and unsupported plant species before reaching the disease model.
- **Evaluation Integrity:** Reports held-out test set performance (`96.86%` Macro-F1 across 3,506 verified images) while explicitly noting that controlled-dataset accuracy does not guarantee open-field performance.
- **Responsible AI Boundaries:** Documents four critical operational limitations: single-leaf constraint, laboratory background bias, closed-world assumption, and non-authoritative advisory status.

---

## 3. Files Created, Modified, and Preserved

### 3.1 Files Created
| File Path | Lines | Size | Purpose |
|---|---|---|---|
| [`frontend/analyze.html`](file:///c:/Users/bhavy/OneDrive/Desktop/College/Sem%205/Project/frontend/analyze.html) | 365 | 19.5 KB | Dedicated Diagnostic Studio workspace |
| [`frontend/field-guide.html`](file:///c:/Users/bhavy/OneDrive/Desktop/College/Sem%205/Project/frontend/field-guide.html) | 165 | 9.9 KB | Foliar Pathology Field Guide catalog |
| [`frontend/about.html`](file:///c:/Users/bhavy/OneDrive/Desktop/College/Sem%205/Project/frontend/about.html) | 215 | 12.6 KB | System architecture & responsible AI monograph |
| [`frontend/js/nav.js`](file:///c:/Users/bhavy/OneDrive/Desktop/College/Sem%205/Project/frontend/js/nav.js) | 95 | 3.8 KB | Shared header controller, active route highlighting, telemetry polling, mobile drawer |
| [`frontend/js/field-guide.js`](file:///c:/Users/bhavy/OneDrive/Desktop/College/Sem%205/Project/frontend/js/field-guide.js) | 385 | 16.0 KB | Field guide controller, live search, multi-facet filtering, slide-over drawer |
| [`frontend/js/knowledge_data.js`](file:///c:/Users/bhavy/OneDrive/Desktop/College/Sem%205/Project/frontend/js/knowledge_data.js) | 1,745 | 74.0 KB | Client-side validated dataset of all 15 Solanaceae pathology monographs |
| [`frontend/samples/hero_specimen.jpg`](file:///c:/Users/bhavy/OneDrive/Desktop/College/Sem%205/Project/frontend/samples/hero_specimen.jpg) | - | 1.1 MB | Art-directed botanical specimen photograph |

### 3.2 Files Modified
| File Path | Changes | Purpose |
|---|---|---|
| [`frontend/index.html`](file:///c:/Users/bhavy/OneDrive/Desktop/College/Sem%205/Project/frontend/index.html) | Completely rewritten (545 lines) | Editorial flagship home, hero specimen framing stage, operational pillars, integrated scanner |
| [`frontend/css/styles.css`](file:///c:/Users/bhavy/OneDrive/Desktop/College/Sem%205/Project/frontend/css/styles.css) | Completely rewritten (2,050 lines) | Complete three-color token architecture, editorial layout, field guide grid, slide-over drawer, mobile navigation, print rules |

### 3.3 Files Preserved
| File Path | Status | Reason |
|---|---|---|
| [`frontend/js/api.js`](file:///c:/Users/bhavy/OneDrive/Desktop/College/Sem%205/Project/frontend/js/api.js) | Preserved | Verified HTTP client contracts, timeout handling, error parsing |
| [`frontend/js/app.js`](file:///c:/Users/bhavy/OneDrive/Desktop/College/Sem%205/Project/frontend/js/app.js) | Preserved | Diagnostic pipeline, image preprocessing, rejection rendering, candidate exploration |
| Backend & ML Services | Preserved | Zero changes to FastAPI routes, model weights, datasets, or evaluation scripts |

### 3.4 Files Removed
*Zero files deleted.* Full backward recoverability is maintained.

---

## 4. Key Features Implemented

### 4.1 Specimen Studio & Scan Stage
- **Instrument Framing:** Clean dark-mode viewports with corner crosshairs (`border-color: var(--color-coral)`).
- **Fit & Zoom Controls:** Specimen inspection controls allowing agronomists to toggle zoom and fit modes on high-resolution leaf scans.
- **Image Metadata Readout:** Displays original image filename, dimensions (e.g. `256 × 256 px`), file size, and upload timestamp.

### 4.2 Two-Tier Safety Feedback
The existing machine-learning input gate is visually rendered with clarity:
- **Supported Leaf:** Passes directly to MobileNetV2 disease classification.
- **Non-Leaf Rejection (Code: `NON_LEAF_IMAGE`):** Displays a clear rejection banner (*"No Crop Leaf Verified"*), explains that the image does not match supported foliar profiles, and provides an immediate retry button.
- **Unsupported Species Rejection (Code: `UNSUPPORTED_SPECIES`):** Clarifies that the leaf is outside the supported Solanaceae trio (Tomato, Potato, Bell Pepper).
- **Ambiguous Input (Code: `AMBIGUOUS_IMAGE`):** Encourages better lighting and centering without hallucinating a diagnosis.

### 4.3 Field Guide Search & Slide-Over Monograph Drawer
- **Instant Filtering:** Real-time client-side search across all 15 classes without latency or backend roundtrips.
- **Pathology Drawer:** Smooth CSS slide-over drawer from the right screen edge displaying complete causal agent biology, symptoms, cultural management, chemical interventions, and literature citations.
- **Focus Management:** Accessible keyboard navigation with automatic focus trapping and `Escape` key dismissal.

### 4.4 Actionable Export & Printing
- **Clipboard Export:** Formats clean text advisory summaries with plant name, condition, confidence score, and management points, accompanied by an animated toast confirmation.
- **Print Optimization:** Dedicated `@media print` rules strip navigation, buttons, and dark backdrops, producing a crisp black-on-white report suitable for field agronomists.

---

## 5. API Integration & Safety Contracts

The application interfaces directly with the live FastAPI backend:

| Endpoint | Method | Role | Contract Preserved |
|---|---|---|---|
| `/health` | `GET` | Service liveness | Returns status and uptime |
| `/ready` | `GET` | Engine readiness | Returns `ready: true` and `gate_loaded: true` |
| `/api/predict` | `POST` | Two-tier inference | Accepts multipart form data; returns diagnosis or machine-readable rejection |
| `/api/knowledge/{class_id}` | `GET` | Monograph retrieval | Returns validated JSON record for class 0..14 |

### Request Guarding & State Integrity
- **Anti-Race Tokens:** `currentKnowledgeRequestId` sequence numbers prevent stale asynchronous responses from overwriting newer user selections.
- **State Sanitization:** Selecting a new image immediately clears previous diagnostic cards, candidates, and rejection messages.
- **Rejection Protection:** Rejected or ambiguous images never render disease advice, candidate bars, or copy buttons.

---

## 6. Verification & Test Results

All verification suites were executed against the running backend daemon (`task-3039`) and static web server.

### 6.1 Frontend API Unit & Live Integration Suite (`frontend/test_api.js`)
Executed via `node --test frontend/test_api.js`:
```
▶ CropDiseaseApi - Export Verification
  ✔ CropDiseaseApi exports all expected methods and properties (2.01ms)
▶ CropDiseaseApi.getDiseaseKnowledge - Client-Side Validation
  ✔ 15 boundary tests (-1, 15, 99, 2.5, NaN, Infinity, strings, null, booleans) passed (13.01ms)
▶ CropDiseaseApi.getDiseaseKnowledge - Mocked Network & Error Handling
  ✔ 6 network tests (GET, 10s timeout, 404, 503, 422, status 0, status 408) passed (67.40ms)
▶ CropDiseaseApi - Regression Tests for Existing Endpoints
  ✔ 3 regression tests (/health, /ready, null file 400 rejection) passed (1.17ms)
▶ CropDiseaseApi.predictCropDisease - Content gate contract
  ✔ 8 safety gate contract tests (rejections, abstained, valid contract, malformed) passed (7.28ms)
▶ CropDiseaseApi.getDiseaseKnowledge - Live Integration with Backend
  ✔ Class 0 (Bell Pepper Bacterial Spot) live query passed (53.23ms)
  ✔ Class 2 (Potato Early Blight) live query passed (7.34ms)
  ✔ Class 14 (Tomato Healthy) live query passed (2.41ms)

Result: 36 passed, 0 failed, 0 skipped (Duration: 404ms)
```

### 6.2 Browser E2E Safety & Visual Suite (`frontend/test_browser.py`)
Executed via headless Chrome via Chrome DevTools Protocol (CDP):
```
Starting backend and static server...
Launching headless Chrome...
Checking desktop viewport (1440x900)...
  person case: elderly-gambian-woman-face-portrait.jpg -> REJECTED (NON_LEAF_IMAGE)
  desktop: OK
Checking mobile viewport (390x844)...
  person case: elderly-gambian-woman-face-portrait.jpg -> REJECTED (NON_LEAF_IMAGE)
  mobile: OK
```

### 6.3 Backend Core Test Suite (`backend/tests`)
Executed via `.venv\Scripts\python.exe -m unittest discover -s backend/tests -p "test_*.py"`:
```
Ran 98 tests in 20.889s
OK (98 passed, 0 failed)
```

### 6.4 ML Pipeline Test Suite (`ml/tests`)
Executed via `.venv\Scripts\python.exe -m unittest discover -s ml/tests -p "test_*.py"`:
```
Ran 22 tests in 3.664s
OK (22 passed, 0 failed)
```

**Total Automated Tests Passed: 156 / 156**

---

## 7. Screenshot Showcase

Ten high-resolution screenshots were captured across desktop and mobile viewports, saved directly to [`docs/audit_reports/reimagined_screenshots/`](file:///c:/Users/bhavy/OneDrive/Desktop/College/Sem%205/Project/docs/audit_reports/reimagined_screenshots):

| Screenshot | Viewport | Size | Description |
|---|---|---|---|
| `01_home_desktop.png` | 1440 × 900 | 756 KB | Editorial home with hero specimen photography and operational pillars |
| `01_home_mobile.png` | 390 × 844 | 629 KB | Responsive mobile home with clean single-column hierarchy |
| `02_analyze_desktop.png` | 1440 × 900 | 310 KB | Full-screen Diagnostic Studio with specimen framing and quick presets |
| `02_analyze_mobile.png` | 390 × 844 | 287 KB | Mobile Diagnostic Studio showing touch-optimized upload dropzone |
| `03_field_guide_desktop.png` | 1440 × 900 | 484 KB | Solanaceae Pathology Encyclopedia with real-time crop/condition filters |
| `03_field_guide_mobile.png` | 390 × 844 | 569 KB | Mobile Field Guide card stream with accessible touch chips |
| `04_about_desktop.png` | 1440 × 900 | 508 KB | Technical monograph on MobileNetV2 architecture and the two-tier gate |
| `04_about_mobile.png` | 390 × 844 | 507 KB | Mobile technical narrative with readable editorial typography |
| `05_field_guide_drawer_open.png` | 1440 × 900 | 232 KB | Slide-over clinical monograph drawer displaying Tomato Early Blight guidance |
| `06_analyze_diagnosed_monograph.png` | 1440 × 900 | 978 KB | Complete diagnostic result with top-3 candidates and actionable agronomic advisory |

---

## 8. Known Limitations & Responsible Boundaries

1. **Controlled Dataset Environment:** The model was trained on standardized laboratory foliar imagery (PlantVillage). Real-world field photographs with complex canopies, background weeds, or severe motion blur may experience reduced diagnostic confidence.
2. **Closed-World Softmax Distribution:** Confidence scores reflect relative probabilities among the 15 supported Solanaceae classes, not an absolute statistical proof of pathogen presence.
3. **Solanaceae Scope Boundary:** The system is strictly calibrated for Tomato, Potato, and Bell Pepper leaves. Images of other crops (e.g. Corn, Apple, Wheat) will be intercepted by the input gate as unsupported species.
4. **Advisory Decision Support:** PHYTOSIGNAL is designed as an agronomic decision-support instrument, not an autonomous pesticide dispenser. Field verification by a certified plant pathologist or agricultural extension officer is always recommended.

---

## 9. Recommended Next Steps

- **Progressive Web App (PWA):** Register a service worker with caching strategies to allow field scouts to consult the Field Guide completely offline in remote agricultural zones.
- **Real-Time Camera Stream:** Implement client-side WebRTC video frame capture with real-time blur detection before triggering inference.
- **Multilingual Support:** Add agricultural extension translations (e.g., Spanish, Hindi, Swahili) for global smallholder farmer accessibility.
