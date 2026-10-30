# Documentation Index

This folder holds the evidence for the project's claims. The top-level
[`README.md`](../README.md) describes the system; `docs/` records how each number in it was
measured and what was audited to get there.

- [`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md) — the consolidated report: problem statement,
  architecture, dataset engineering, model design, results, backend and knowledge-base design,
  testing, limitations, and viva preparation. Start here for a submission or a walkthrough.
- [`KNOWLEDGE_BASE.md`](KNOWLEDGE_BASE.md) — data specification for the 15 agronomic records:
  schema, sourcing standard, and the startup validation rules.
- [`audit_reports/`](audit_reports/) — measurement and audit reports, indexed below.

---

## Reports

| Report | What it establishes | Headline result |
|:--|:--|:--|
| [`audit_reports/plantvillage/dataset_audit_report.md`](audit_reports/plantvillage/dataset_audit_report.md) | Quality of the raw PlantVillage download: class balance, corrupt files, exact duplicates, perceptual near-duplicates, cross-partition leakage | Audit of all 55,448 images before any split was made |
| [`audit_reports/preparation_report.md`](audit_reports/preparation_report.md) | Reconciliation and stratified split construction (seed 42) | 22,773 target images → 15,942 / 3,418 / 3,413 at 70/15/15 |
| [`audit_reports/split_verification.json`](audit_reports/split_verification.json) | Machine-readable proof the split is leak-free | 0 SHA-256 overlaps across splits; duplicate groups fully group-locked |
| [`audit_reports/near_duplicate_review.md`](audit_reports/near_duplicate_review.md) | Human review of the 5 remaining cross-class near-duplicate pairs (dHash = 2) | Resolved individually; see the [contact sheet](audit_reports/near_duplicates_contact_sheet.png) |
| [`audit_reports/model_validation_comparison.md`](audit_reports/model_validation_comparison.md) | Unweighted vs. balanced class-weighted training, compared on validation only | Weighted model selected on validation macro-F1 0.9686 |
| [`audit_reports/final_test_evaluation_report.md`](audit_reports/final_test_evaluation_report.md) | The single held-out test evaluation of the selected checkpoint | **macro-F1 0.9653**, accuracy **96.89 %** (3,307 / 3,413) |
| [`audit_reports/test_evaluation_integrity_audit.md`](audit_reports/test_evaluation_integrity_audit.md) | Integrity audit of that evaluation — that the test split was untouched and the reported metrics reproducible | Confirms one-shot evaluation, no test-time selection |
| [`audit_reports/phase5b_ood_policy_verification.md`](audit_reports/phase5b_ood_policy_verification.md) | Whether an out-of-distribution gate could reject unsupported crops | **Rejected**: embedding-centroid gate gave 56 % false-accept on unsupported crops and 5.8–7.6 % false rejection of valid diseased leaves |
| [`audit_reports/phase5b_level_a_final_validation_audit.md`](audit_reports/phase5b_level_a_final_validation_audit.md) | That the input-safety thresholds reject no legitimate samples, plus the upload-limit and concurrency measurements behind `config.py` | σ ≤ 2.0 vs. empirical minimum σ = 11.31 over 3,418 validation images (5.65× margin); 20 MP natural PNG = 60 MB → rejected at the 10 MB stream gate |
| [`audit_reports/non_leaf_input_safety_report.md`](audit_reports/non_leaf_input_safety_report.md) | Why a non-leaf photograph received a diagnosis, and the layered gate that now rejects/abstains before inference | Trained 3-class leaf-acceptance gate (§1.4): 1.7 % valid-leaf no-diagnosis (1,698 disjoint leaves); 100 % of 70 real non-leaf photographs and 193 leaf-free field photographs blocked; 97.8 % of seen unsupported species, 75.7 %/53.3 % of held-out Strawberry/Soybean |
| [`audit_reports/backend_phase3b_implementation.md`](audit_reports/backend_phase3b_implementation.md) | Knowledge-base service design: Pydantic schemas, validation rules, `/api/knowledge/{id}` contract and error codes | 15 records validated at startup; sanitized 404/422/503 |

Both `.md` and `.json` variants exist for the generated reports. The Markdown is for reading;
the JSON is the machine-readable output of the script that produced it, kept so results can
be diffed rather than re-measured. The input-gate numbers live in
[`audit_reports/non_leaf_input_safety_eval.json`](audit_reports/non_leaf_input_safety_eval.json)
and are produced by `ml/scripts/evaluate_input_gate.py`.

---

## Reproducing the Results

The dataset, training, and evaluation numbers can be regenerated from `ml/scripts/`:

```bash
# 1. Audit the raw dataset (non-destructive)
.venv/Scripts/python.exe ml/scripts/audit_dataset.py --data-dir "ml/data/raw_dataset"

# 2. Rebuild the stratified split (deterministic, seed 42)
.venv/Scripts/python.exe ml/scripts/prepare_dataset.py

# 3. Train (writes best_model.keras + metadata to the output dir)
.venv/Scripts/python.exe ml/scripts/train_model.py --class-weighting balanced --output-dir ml/models/experiment_weighted

# 4. Compare candidate models on the validation split only
.venv/Scripts/python.exe ml/scripts/evaluate_validation.py

# 5. One-shot held-out test evaluation
.venv/Scripts/python.exe ml/scripts/evaluate_test_set.py

# 6. Input-gate evaluation (real non-leaf photographs + post-hoc synthetic probes)
.venv/Scripts/python.exe ml/scripts/evaluate_input_gate.py
```

The full suite is described in the top-level README: 75 backend tests, 22 ML pipeline tests,
35 frontend API tests, and a headless Chrome end-to-end suite.

### A note on removed measurement scripts

The Phase 5B safety work was measured with throwaway scripts kept in a gitignored `scratch/`
directory — the σ sweep over all 3,418 validation images, the synthetic aliasing fixtures,
the upload-limit benchmark, and the OOD experiment. That directory has been deleted as part
of repository cleanup, so **those particular measurements are now historical records rather
than runnable reproductions.** Each report retains its methodology and full result tables, so
the conclusions remain auditable by inspection, but re-running them would mean re-deriving
the harnesses.

Likewise, eight intermediate phase drafts were removed once their conclusions were folded
into the final reports above — the Phase 1 code-audit, remediation, and verification
sequence; the Phase 5B Step 2 OOD design and Step 4/5 Level A contract drafts; and the two
cleanup-inventory meta-reports. Two retained reports carried links to those files, which now
read as plain references noting the removal.
