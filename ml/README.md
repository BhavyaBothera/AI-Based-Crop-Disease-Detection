# Machine Learning Subsystem

This directory contains the dataset preparation, auditing, model training, and evaluation code for the **AI-Based Crop Disease Detection and Treatment Recommendation System**.

---

## Dataset Audit Tool (`ml/scripts/audit_dataset.py`)

A non-destructive dataset auditing script to inspect any candidate plant disease image dataset before training.

### What It Inspects:
1. **Solanaceae Crop Coverage:** Evaluates whether **Tomato**, **Potato**, and **Bell Pepper** (`Capsicum`) classes exist.
2. **Health Status Classification:** Identifies `Healthy` vs `Diseased` categories.
3. **Class Imbalance & Baseline Metrics:** Computes per-class counts, percentages, imbalance ratio, and the **Zero-Rule Majority Class Baseline Accuracy**.
4. **Data Integrity:** Identifies unreadable/corrupted files, non-image files, and images smaller than minimum resolution (e.g. < 128px).
5. **Exact Duplicates:** Fast chunked SHA-256 byte hashing.
6. **Perceptual Near-Duplicates:** 64-bit Difference Hashing (`dHash`) with multi-index block pruning (Hamming distance threshold).
7. **Cross-Partition Leakage:** Flags images shared across `train`, `val`, and `test` subfolders.

---

### Prerequisites & Dependencies
- **Python:** Python 3.11 recommended (`py -3.11`).
- **Dependencies:** Standard library + `Pillow` (`PIL`).

Verify Pillow is available:
```powershell
py -3.11 -c "import PIL; print(PIL.__version__)"
```

---

### Usage

Run the audit on any extracted dataset folder:

```powershell
# Basic usage
py -3.11 ml/scripts/audit_dataset.py --data-dir "ml/data/your_dataset_folder"

# Custom output directory and sensitivity
py -3.11 ml/scripts/audit_dataset.py `
  --data-dir "ml/data/raw_dataset" `
  --output-dir "docs/audit_reports" `
  --dhash-threshold 2 `
  --min-dimension 128
```

### CLI Arguments
| Argument | Flag | Default | Description |
| :--- | :--- | :--- | :--- |
| `--data-dir` | `-d` | *(Required)* | Path to the directory containing dataset images. |
| `--output-dir` | `-o` | `docs/audit_reports` | Directory where `.md` and `.json` reports are written. |
| `--dhash-threshold` | | `2` | Hamming distance (0-64) to flag perceptual duplicates. |
| `--min-dimension` | | `128` | Threshold in pixels to flag unusually small images. |
| `--max-near-dups` | | `100` | Max near-duplicate pairs to include in the report. |

---

### Output Reports
The script generates two artifacts:
- `docs/audit_reports/dataset_audit_report.md`: Human-readable Markdown summary with tables for your college project documentation.
- `docs/audit_reports/dataset_audit_report.json`: Machine-readable structured metrics for automated verification.

---

### Data Leakage Prevention Notice
> **Important:** Identifying duplicates during the audit is step one. In Phase 2, any near-duplicate pairs or exact duplicate clusters must **never** be split across training, validation, and test sets. Augmentation (rotations, flips, zoom) must only take place dynamically inside the training pipeline.

---

## Model Training Pipeline (`ml/scripts/train_model.py`)

A leak-safe transfer learning and fine-tuning script using **MobileNetV2** for 15-class Solanaceae crop disease detection.

### Pipeline Architecture:
1. **Input Preprocessing:** Raw RGB images of shape `(224, 224, 3)` with pixel values in `[0, 255]`. Normalization to `[-1, 1]` is embedded directly inside the model as a `Rescaling(1./127.5, offset=-1.0)` layer.
2. **Backbone:** MobileNetV2 with ImageNet pretrained weights. In Stage 1, the entire backbone is frozen (`trainable = False`) and called with `training=False` to preserve batch normalization statistics.
3. **Classification Head:**
   - `GlobalAveragePooling2D()`
   - `BatchNormalization()`
   - `Dense(256, activation='relu', kernel_regularizer=l2(1e-4))`
   - `Dropout(0.35)`
   - `Dense(15, activation='softmax')`
4. **Two-Stage Training:**
   - **Stage 1 (Feature Extraction):** Trains only the classification head with Adam optimizer (`lr=1e-3`).
   - **Stage 2 (Fine-Tuning):** Unfreezes upper backbone layers (indices 100 to 153) with Adam optimizer (`lr=1e-5`). All backbone `BatchNormalization` layers remain strictly frozen (`layer.trainable = False`), and the backbone continues to execute in inference mode (`training=False`).
5. **Validation-Based Model Selection:**
   - Calculates the **full-validation-set Macro-F1** score after every epoch (never averaged over mini-batches).
   - The Macro-F1 callback executes first so that learning-rate schedulers and early stopping observe `val_macro_f1`.
   - Global best checkpoint tracking across both stages preserves the model with the highest validation Macro-F1.
   - Differences within `1e-4` are treated as ties and deterministically broken by lower validation loss (`val_loss`).
   - Saves `best_model.keras` along with structured `best_model_metadata.json`, `config.json`, and `training_history.json`.
6. **Strict Split Isolation:**
   - Only rows with `status == "included"` in `ml/data/dataset_manifest.csv` are used.
   - Exactly 15,942 training and 3,418 validation images are streamed via `tf.data`.
   - The 3,413 test images are verified from the manifest schema but are **never loaded or decoded** during training.

---

### Usage & Experiments

```powershell
# Display help and CLI options
py -3.11 ml/scripts/train_model.py --help

# Verify pipeline without training (dry run)
py -3.11 ml/scripts/train_model.py --dry-run --weights none

# Experiment A: Unweighted Cross-Entropy
py -3.11 ml/scripts/train_model.py `
  --class-weighting none `
  --output-dir ml/models/unweighted `
  --batch-size 32 `
  --epochs-stage1 15 `
  --epochs-stage2 15

# Experiment B: Balanced Class-Weighted Cross-Entropy
py -3.11 ml/scripts/train_model.py `
  --class-weighting balanced `
  --output-dir ml/models/balanced `
  --batch-size 32 `
  --epochs-stage1 15 `
  --epochs-stage2 15

# Compare experiments on validation metrics (no test set leakage)
py -3.11 ml/scripts/train_model.py --compare ml/models/unweighted ml/models/balanced
```

---

## Leaf-Acceptance Gate (`ml/scripts/train_input_gate.py`)

The input gate that screens uploads before the disease model is a separate 3-class model
(`supported_leaf` / `other_plant` / `non_leaf`): a linear head on frozen ImageNet MobileNetV2
features, exported as one self-contained Keras file (`ml/models/input_gate/leaf_gate.keras`)
that the backend loads like any other model.

```bash
# Download the licence-cleared real-photo OOD corpus (needs network, one-off)
.venv/Scripts/python.exe ml/scripts/build_ood_corpus.py

# Extract cached features, fit the head, export, verify export parity
.venv/Scripts/python.exe ml/scripts/train_input_gate.py

# Held-out evaluation + behavioural probes + latency (writes docs/audit_reports/)
.venv/Scripts/python.exe ml/scripts/evaluate_input_gate.py
```

Splits are path-hashed and disjoint (gate-train / gate-val / gate-eval); the disease `test`
split is never read, and Strawberry + Soybean are held out of gate training entirely so species
rejection is measured on unseen species. `--train` alone refits from the cached embeddings
(seconds) without re-extracting features.

### Unit Test Suite (`ml/tests/test_train_pipeline.py`)

Run unit tests verifying class mappings, split isolation, class weight computation, tie-breaking, and MobileNetV2 architecture specifications:

```powershell
py -3.11 -m unittest ml/tests/test_train_pipeline.py
```

