# Process: Copra Moisture-Based Grade Classification

Short walkthrough of the pipeline. **The governing documents are `docs/model-development-instructions.md` (build spec) and `docs/model-acceptance-criteria.md` (pass/fail targets).** If this file disagrees with them, they win. Section numbers below (§N) refer to the build spec.

---

## 1. The Dataset (§3)

```
copra-dataset.csv
photos/
    <id>.jpeg
    ...
```

| Column | Meaning |
| --- | --- |
| `id` | Unique photo ID (one row = one photo). |
| `batch_id` | **Cube ID.** Each cube has 4 photos, one per cut meat face, sharing this ID. One copra is cut into 1–3 cubes; which cubes came from which copra was not recorded (no `copra_id`). |
| `copra_class` | Label `A`–`F`: the owner's pasa grade of the whole copra, inherited by its cubes. Shown as "Grade A" … "Grade F". |
| `path` | `photos/<id>.jpeg`, relative to the CSV. Image root is configurable. |

Capture is fixed: 10 cm, no rotation, ISO 100, 1/60 s, WB 6000 K, focus 0.10 m, 4:3 JPEG, mid-grey construction paper, two white lights that dim slowly during a session.

Minimum size: 50 cubes per grade, 300 cubes and 1,200 raw photos in total. Smallest grade's cube count ≥ 90% of the largest (acceptance P1, P2).

Grades and moisture bands: see `docs/classes.md`.

---

## 2. Pipeline Steps

### Step 1: Load and check (§3, §6.1)

Read the CSV, load photos, keep `batch_id` on every row. Detect and resolve duplicate Cube IDs. Report cube count per grade.

### Step 2: Preprocessing (§4)

1. **Background model + GrabCut mask** (Otsu failed on the pilot, ADR-008): fit a quadratic surface per LAB channel to the border band, seed copra at > 4 robust SDs, GrabCut 3 iterations, keep the largest component, fill holes. The mask keeps both white meat and brown skin.
2. **Lighting normalization:** scale each photo so the fitted paper color matches one fixed reference grey (ADR-012).
3. **Scale normalization:** resize to a fixed pixel scale.

A clearly wrong mask: retake the face if the cube still exists, otherwise exclude it.

### Step 3: Feature extraction (§5)

On the masked, normalized image only. All three families always ship; one row per face image:

- **Texture — GLCM:** contrast, homogeneity, energy, entropy at 0°, 45°, 90°, 135° and distances 1–3 px (ADR-006).
- **Color — HSV and LAB:** per-channel mean and SD over copra pixels (ADR-009).
- **Edges — Canny:** edge density and contour stats.

### Step 4: Outlier removal (§6.2)

On feature values, not images. Outlier if |z| > 3 **or** outside `Q1 − 1.5×IQR` to `Q3 + 1.5×IQR`. A photo with even one outlier value is dropped as a whole row, so no null values are left (ADR-001). Never impute, never relax. Report drops per grade (acceptance P3).

### Step 5: Nested split, augmentation, tuning (§7, §8b)

- **Outer:** `StratifiedGroupKFold(5)`, grouped by `batch_id`, stratified by `copra_class`. Reporting only.
- **Inner:** `StratifiedGroupKFold(3)` inside each outer training portion, same grouping. Optuna scores each trial by mean inner Macro F1.
- **Augmentation:** rotations and flips only, more copies for smaller grades (ADR-002). Inner training folds during tuning; whole outer training portion when refitting. Never before a split.
- **Refit** the best trial on the full augmented outer training portion, score once on the held-out outer fold.

### Step 6: Train and compare four models (§8a)

| Model | Role | Scaling |
| --- | --- | --- |
| Logistic Regression | Baseline only, never deployed | Standardized, inside a `Pipeline` |
| Random Forest | Candidate | None |
| XGBoost | Candidate | None |
| LightGBM | Candidate | None |

All four class-weighted natively. No stacking or voting across them.

### Step 7: Select (§8c)

Best of RF / XGBoost / LightGBM by mean outer-fold Macro F1. Do not re-tune or retrain on the outer folds.

### Step 8: Evaluate (§11, acceptance §3–§6)

Per image, on held-out outer folds: pooled confusion matrix, accuracy, Macro F1 (mean ± SD), per-grade precision and recall, Grade A precision. Plus background-only check before and after normalization, feature-family ablation, optional PCA/t-SNE sanity plot. Fill in the acceptance pass/fail table.

### Step 9: Label check (§9)

Six pooled Brown-Duvel readings, one per grade, each checked against its grade's band. Sanity check only, never relabel.

### Step 10: Explainability (§10)

TreeSHAP on the selected model only, aggregated by family (texture / color / edge). Check the top features for a plausible physical explanation.

### Step 11: Final artifact (§8b, §12)

Re-run Optuna for the selected algorithm on the full dataset (grouped 5-fold), train on all data, serialize with its preprocessing and feature config (ADR-005). Expose a per-photo SHAP function for the Streamlit app (one photo in, one grade out).

---

## 3. Summary

| Step | What | Method |
| --- | --- | --- |
| 1 | Load | CSV + `photos/`, duplicate Cube ID check |
| 2 | Preprocess | Background model + GrabCut, lighting normalization, resize |
| 3 | Features | GLCM + HSV/LAB + Canny |
| 4 | Outliers | Drop if \|z\| > 3 or outside 1.5×IQR |
| 5 | Split + tune | Nested StratifiedGroupKFold (5 outer / 3 inner) by `batch_id`; geometric augmentation after split; Optuna on Macro F1 |
| 6 | Train | LR (baseline), RF, XGBoost, LightGBM, all class-weighted |
| 7 | Select | Best ensemble by mean outer-fold Macro F1 |
| 8 | Evaluate | Acceptance criteria M1–M4, L2, F1 |
| 9 | Label check | Pooled Brown-Duvel per grade (L1) |
| 10 | Explain | TreeSHAP by feature family (F2) |
| 11 | Artifact | Selected model + config, for Streamlit |
