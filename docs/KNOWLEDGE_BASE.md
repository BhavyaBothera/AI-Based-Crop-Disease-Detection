# Agronomic Knowledge Base — Data Specification

**File:** `backend/app/data/disease_knowledge_base.json` (~72 KB, 15 records)
**Served by:** `GET /api/knowledge/{class_id}`
**Loaded by:** `KnowledgeBaseService` (`backend/app/services/knowledge.py`) once at application startup

The knowledge base is the agronomic half of this project. The model answers *"which of 15
conditions does this leaf match?"*; the knowledge base answers *"what does that mean for the
grower?"* — symptoms, environmental drivers, cultural prevention, IPM principles, escalation
triggers, and citations.

It is a plain versioned JSON file. There is no database, no CMS, and no network access at
runtime: the whole file is parsed and validated in memory at startup, and every request is
served from RAM.

---

## 1. Relationship to the Model

Class IDs `0`–`14` are the single authoritative key shared by three components:

| Component | Location |
|:--|:--|
| Model output index | `ml/scripts/train_model.py` (`CLASS_NAMES`) |
| Serving metadata | `backend/app/services/inference.py` (`CLASS_ID_TO_METADATA`) |
| Agronomic record | `disease_knowledge_base.json` (`class_id`) |

At startup the service asserts **1:1 parity** between each record and
`CLASS_ID_TO_METADATA` on `raw_class`, `crop`, `condition`, and `health_status`. A mismatch
is a hard startup failure, not a warning — the class orderings of Keras' `ImageDataGenerator`
and `flow_from_dataframe` are alphabetical and easy to get subtly wrong, so the service
refuses to serve a knowledge record that does not match what the model was trained to emit.

If parity breaks, `/health` still returns green but `/ready` reports `503` with the reason,
and `/api/knowledge/*` returns `503` with a sanitized message.

---

## 2. Record Structure

Each record is one object with 13 top-level fields:

| Field | Type | Notes |
|:--|:--|:--|
| `class_id` | `int` | Authoritative index `0`–`14`, unique |
| `raw_class` | `str` | Exact dataset class string, e.g. `Tomato___Early_blight` |
| `crop` | `str` | Common name: Tomato, Potato, Bell Pepper |
| `condition` | `str` | Disease name, or `Healthy` |
| `health_status` | `str` | `"Healthy"` or `"Diseased"` |
| `causal_agent` | `str \| null` | Scientific binomial; **must be `null`** for healthy classes |
| `symptoms` | `object` | See below |
| `favorable_conditions` | `object` | See below |
| `prevention_and_cultural` | `object` | See below |
| `treatment` | `object` | See below |
| `extension_consultation` | `object` | See below |
| `sources` | `array` (min 1) | See §4 |
| `safety_and_uncertainty_notes` | `str` | Where this condition is visually confused |

Nested objects:

| Object | Fields |
|:--|:--|
| `symptoms` | `summary` (str), `foliar_symptoms` (list), `stem_and_fruit_symptoms` (list), `look_alike_conditions` (list) |
| `favorable_conditions` | `environmental_factors` (str), `transmission_routes` (list) |
| `prevention_and_cultural` | `sanitation` (list), `crop_rotation` (**str**), `irrigation_and_airflow` (list), `resistant_cultivars` (list) |
| `treatment` | `urgency` (str), `non_chemical_controls` (list), `ipm_chemical_principles` (list), `safety_and_regulatory_constraints` (list) |
| `extension_consultation` | `local_specialist` (str), `escalation_triggers` (list) |
| `sources[i]` | `organization`, `title`, `url`, `claims_supported`, `reference_url_or_publication` (all `str`) |

`crop_rotation` is deliberately a **string**, not a list — it states a rotation interval and
non-host set as a single agronomic recommendation, and the validator enforces the type.

---

## 3. The 15 Records

| ID | Crop | Condition | Health | Causal agent |
|:--:|:--|:--|:--|:--|
| 0 | Bell Pepper | Bacterial Spot | Diseased | *Xanthomonas euvesicatoria* (syn. *X. campestris* pv. *vesicatoria*) |
| 1 | Bell Pepper | Healthy | Healthy | — (`null`) |
| 2 | Potato | Early Blight | Diseased | *Alternaria solani* |
| 3 | Potato | Late Blight | Diseased | *Phytophthora infestans* |
| 4 | Potato | Healthy | Healthy | — (`null`) |
| 5 | Tomato | Bacterial Spot | Diseased | *Xanthomonas* species complex (*X. perforans*, *X. euvesicatoria*, *X. vesicatoria*, *X. gardneri*) |
| 6 | Tomato | Early Blight | Diseased | *Alternaria solani* |
| 7 | Tomato | Late Blight | Diseased | *Phytophthora infestans* |
| 8 | Tomato | Leaf Mold | Diseased | *Passalora fulva* (syn. *Fulvia fulva*, *Cladosporium fulvum*) |
| 9 | Tomato | Septoria Leaf Spot | Diseased | *Septoria lycopersici* |
| 10 | Tomato | Two-spotted Spider Mite | Diseased | *Tetranychus urticae* (Acari: Tetranychidae) |
| 11 | Tomato | Target Spot | Diseased | *Corynespora cassiicola* |
| 12 | Tomato | Yellow Leaf Curl Virus | Diseased | *Tomato yellow leaf curl virus* (TYLCV; *Begomovirus*) |
| 13 | Tomato | Mosaic Virus | Diseased | *Tomato mosaic virus* (ToMV; *Tobamovirus*) |
| 14 | Tomato | Healthy | Healthy | — (`null`) |

Note that class `10` is a mite, not a pathogen, and classes `12`/`13` are viruses — neither
is controllable by fungicide. The record set reflects that in its treatment fields rather
than emitting a generic spray schedule.

---

## 4. Sourcing Standard

Every record carries **2 citations** (30 in total) and every claim-bearing section is tied to
a published extension or university source. Sources are public agricultural extension
material, not commercial product literature:

| Organization | Citations | Domain |
|:--|--:|:--|
| Tamil Nadu Agricultural University (TNAU) Agritech Portal | 12 | `agritech.tnau.ac.in` |
| University of California Statewide IPM Program (UC IPM) | 11 | `ipm.ucanr.edu` |
| The Ohio State University Extension (High Tunnel Disease Facts) | 2 | `u.osu.edu` |
| University of Florida IFAS Extension (EDIS) | 2 | `edis.ifas.ufl.edu` |
| University of Massachusetts Amherst Extension (CAFE) | 1 | `ag.umass.edu` |
| Clemson University Cooperative Extension (HGIC) | 1 | `hgic.clemson.edu` |
| Purdue University Extension (Vegetable Crops Hotline) | 1 | `vegcropshotline.org` |

Each source object names the organization and publication title, states the specific claims
it substantiates in `claims_supported`, and gives both a live `url` and a stable
`reference_url_or_publication` fallback. TNAU and UC IPM carry the bulk of the load because
they publish complete per-disease IPM pages covering the same Solanaceae diseases as
PlantVillage.

**URLs are validated syntactically, not fetched.** The startup check enforces the
`http`/`https` scheme and non-empty metadata entirely offline. A link that later rots will
still validate — periodic link auditing is a maintenance task, not a runtime guarantee.

The frontend only makes a source clickable when `isValidHttpUrl()` passes; otherwise it
renders the citation as inert text. Source text is inserted with `textContent`, never
`innerHTML`.

---

## 5. Validation Guarantees

`KnowledgeBaseService` refuses to start on any of the following, because a silently wrong
agronomic recommendation is worse than a hard failure:

- Root is not a JSON array, or does not contain exactly 15 records.
- Any class ID in `0`–`14` is missing, duplicated, or out of range.
- Any of `raw_class` / `crop` / `condition` / `health_status` disagrees with `CLASS_ID_TO_METADATA`.
- A healthy class (`1`, `4`, `14`) has a non-null `causal_agent`, or its
  `ipm_chemical_principles` recommends a chemical spray.
- A diseased class has an empty or missing `causal_agent`.
- `crop_rotation` is a list rather than a string.
- Any list field contains an empty or non-string element.
- Any source has a non-HTTP(S) URL or empty metadata.

The full rule set and the 21 dedicated tests that enforce it are described in
[`audit_reports/backend_phase3b_implementation.md`](audit_reports/backend_phase3b_implementation.md)
and `backend/tests/test_knowledge.py`.

Retrieved records are returned as `copy.deepcopy`, so a caller cannot mutate server state.

---

## 6. Editing the File

Edit the JSON, restart the backend, and check `/ready`:

```bash
.venv/Scripts/python.exe -m unittest backend.tests.test_knowledge -v
```

The validator is strict on purpose. If the service fails to start after an edit, the
exception names the offending `class_id` and rule, and the reason is written to the server
log while the API continues to return a sanitized `503`.

Two conventions worth keeping:

1. **Healthy means no disease management.** Records `1`, `4`, and `14` describe maintenance,
   scouting, and airflow — not treatment. This is enforced, not merely intended.
2. **Never assert certainty the model cannot support.** Every record's
   `safety_and_uncertainty_notes` names the conditions it is most likely to be confused with,
   because a 15-class softmax score is a relative distribution, not a calibrated diagnosis.
