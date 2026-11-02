# Model Acceptance Criteria — Copra Moisture-Based Grade Classification

**Purpose:** the pass marks the trained models must be checked against, for the model-development repository. These targets were set by the thesis team **before training** and are stated in Methodology §4.10 of the manuscript. They are the team's own targets, not values taken from the literature.

**Companion document:** `model-development-instructions.md` (the full build spec). This file adds only the pass/fail targets and how to report them. If the two ever disagree, flag it to the thesis team instead of choosing one.

**Classes:** `A`, `B`, `C`, `D`, `E`, `F` (the `copra_class` column), shown as "Grade A" to "Grade F".

---

## 1. Ground rules

1. **Targets are fixed.** Do not lower, round, or reinterpret a target after seeing results.
2. **Never tune to pass.** If a target is missed, report the actual value and the grades that were confused. Do not re-tune, retrain, or change features to push a metric over the line.
3. **Only outer-fold scores count.** Every model metric below comes from the held-out outer folds of the nested cross-validation (5 outer folds of `StratifiedGroupKFold`, grouped by `batch_id` (the Cube ID) and stratified by `copra_class`; 3 inner folds for Optuna). Inner-fold scores and training scores are never used for acceptance.
4. **Metrics are per image.** Each held-out face photo is scored on its own against its copra's pasa grade.
5. **"Selected model"** means the best of Random Forest, XGBoost, and LightGBM by mean outer-fold Macro F1. Logistic Regression is the baseline only and is never eligible.

---

## 2. Preconditions (check before training)

| # | Check | Target | If it fails |
|---|---|---|---|
| P1 | Cubes per grade | At least 50 cubes per grade (at least 300 cubes and 1,200 raw photos in total, before augmentation) | Do not train for reporting. Tell the thesis team that more data is needed. |
| P2 | Grade balance | The smallest grade's cube count is at least 90% of the largest grade's | Report the counts and flag it to the thesis team. |
| P3 | Outlier removal | Drop a feature value if \|z\| > 3 **or** it falls outside Q1 − 1.5×IQR to Q3 + 1.5×IQR. Drop, never impute. | Never relax the thresholds. If dropping pushes any grade below 50 cubes, the answer is more data collection. Report it. |
| P4 | Leakage | No Cube ID appears in both the training and test portion of any fold, at either level. Split before augmenting. | Pipeline bug. Fix before any result is reported. |

---

## 3. Model performance targets (selected model)

| # | Metric | Target | Notes |
|---|---|---|---|
| M1 | Mean Macro F1 across the 5 outer folds | **≥ 0.75** (stretch goal: 0.80) | Report mean ± SD. |
| M2 | Recall for **every** grade A to F, mean across outer folds | **≥ 0.60** for each grade | No single grade may be left poorly recognized. Report all six values. |
| M3 | Grade A precision, mean across outer folds | **≥ 0.85** | A false Grade A on wet copra is the costly error: the station overpays. |
| M4 | Gap over the baseline | Selected model's mean Macro F1 minus Logistic Regression's mean Macro F1 **> one outer-fold SD** | Report both models' means and SDs. |

M1 to M4 together are what the study treats as "accurate enough to support grading decisions at a buying station" (the fourth objective).

**Interpretation defaults (not spelled out in §4.10; flag them as defaults in the code and the results):**
- M2 and M3 use the mean of the per-fold values. Also report the pooled confusion matrix across all outer folds.
- M4 uses the **selected model's** outer-fold Macro F1 SD. Also report Logistic Regression's SD, so the thesis team can see whether the result changes under the other choice.

---

## 4. Label and lighting checks

| # | Check | Target | If it fails |
|---|---|---|---|
| L1 | Pooled Brown-Duvel reading per grade (6 readings, one ~100 g pooled portion per grade) | Each reading falls inside its grade's moisture range: A 6% to <9%, B 9% to <12%, C 12% to <15%, D 15% to <18%, E ≥18%, F <6%. A boundary value (9, 12, 15, 18%) belongs to the higher grade. | Report that grade as a label limitation. **Do not relabel its images.** This is a sanity check on the labels, not an agreement rate and not the main result. |
| L2 | Background-only check: a simple classifier trained on paper-only pixels (no copra) to predict the grade | Near chance (about 17% for six grades) **after** lighting normalization. Accuracy **above 25%** is investigated as lighting leakage. | A failed check means the model's scores cannot be trusted until the leak is fixed. Run it before **and** after normalization, and report both. |

---

## 5. Feature and explainability checks

| # | Check | Target | If it fails |
|---|---|---|---|
| F1 | Feature-family ablation (color-only, texture-only, edge-only, combined) | The combined feature set's mean Macro F1 is **no lower** than the best single family's | Report it. It weakens the case for keeping all three families. |
| F2 | TreeSHAP ranking by feature family (texture / color / edge), on the selected model only | The top features have a plausible physical explanation | Qualitative check, confirmed with the buying station owner. A nonsensical top feature should be investigated as possible leakage or a preprocessing artifact before the result is trusted. |

---

## 6. What to report

Hand back one results summary containing:

1. Cube count per grade, and per grade per outer fold (P1, P2).
2. Number of feature values dropped as outliers, per grade (P3).
3. For each of the four models: mean ± SD outer-fold Macro F1, and each outer fold's chosen hyperparameters.
4. For the selected model: pooled confusion matrix, accuracy, per-grade precision and recall (per fold and mean), and Grade A precision.
5. The six pooled Brown-Duvel readings against their ranges (L1).
6. Background-only accuracy before and after normalization (L2).
7. Ablation table: mean ± SD Macro F1 for color-only, texture-only, edge-only, and combined (F1).
8. TreeSHAP importance aggregated by feature family (F2).
9. A pass/fail table in this form:

| ID | Target | Actual | Pass? | Notes (confused grades, defaults used) |
|---|---|---|---|---|
| P1 | ≥ 50 cubes per grade | | | |
| P2 | Smallest ≥ 90% of largest | | | |
| P4 | No Cube ID across splits | | | |
| M1 | Mean Macro F1 ≥ 0.75 (stretch 0.80) | | | |
| M2 | Recall ≥ 0.60, every grade | | | |
| M3 | Grade A precision ≥ 0.85 | | | |
| M4 | Gap over LR > 1 outer-fold SD | | | |
| L1 | All 6 Brown-Duvel readings in range | | | |
| L2 | Background-only ≤ 25% after normalization | | | |
| F1 | Combined ≥ best single family | | | |
| F2 | TreeSHAP ranking plausible | | | (owner review) |

Also state these known sources of optimism alongside the results: cubes from the same copra can fall in different folds (no Copra ID was recorded), and picking the best of three models by outer-fold scores adds a small optimistic bias.

---

## 7. Out of scope for this repository

These §4.10 checks belong to the Streamlit prototype, not the model pipeline. They are listed here only for awareness:

- **Usability:** a buying station worker submits a photo and gets a grade unaided after a brief walkthrough.
- **Reliability:** a valid grade for every properly formatted image across all six grades, and a predictable failure (no crash) on blurry, wrong-format, or non-copra images.
- **Interpretability:** every prediction can be expanded into a partial TreeSHAP explanation on request.

The pipeline should expose the model artifact and a SHAP function the Streamlit app can call, so these checks can be run later.
