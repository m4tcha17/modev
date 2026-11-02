# Goal: What the Model Must Show

**Study:** *Copra Moisture-Based Grade Classification Using Image-Based Feature Extraction for Quality Grading*
**Date:** 2026-10-31
**For:** the Model Development repository. Read this together with `model-development-instructions.md`, which gives the full build spec. This file says what the results must show for the thesis to count as a success.

> **Superseded (2026-11-02).** `docs/model-acceptance-criteria.md` is now the source of truth for targets and reporting; this file is kept for history. Target mapping: T1→M1, T2→M2, T3→M3, T4→M4, T5→L2 (pass mark: accuracy ≤ 25%), T6→L1, T7→F1 + F2, T8→out of scope (acceptance §7, Streamlit). Within-one-grade accuracy (§3) and "copra per grade" counts are **not** in the acceptance criteria; do not report them as targets unless the thesis team adds them.

---

## 1. Research objectives

These are the thesis's five specific objectives, quoted from Introduction 1.3. Every result the pipeline produces should trace back to one of them.

1. To extract quantitative visual features (texture, color, and edge characteristics) from copra cube images captured at four side angles, the four cut meat faces, using texture analysis, color space transformation, and edge density computation.
2. To train and compare ensemble classification models (Random Forest, XGBoost, LightGBM) against a Logistic Regression baseline for moisture-based grade classification.
3. To evaluate which feature families (texture, color, edge) contribute most to prediction accuracy using TreeSHAP feature importance analysis.
4. To verify the training labels through one pooled Brown-Duvel moisture reading per class, checked against that class's moisture range, and to assess whether the trained model's classifications are accurate enough to support grading decisions at a real buying station.
5. To implement a functional prototype system that applies the trained model to classify copra images and output the corresponding moisture grade (A to F).

General objective (abridged): establish, evaluate, and validate a computer vision pipeline built on tree-based ensemble models, alongside a functional prototype, for classifying copra by moisture grade.

**Wording rule:** Random Forest, XGBoost and LightGBM are each a tree-based ensemble. The study compares them and deploys one. Do not build a voting or stacking ensemble across the three.

---

## 2. Grade ranges (changed 2026-10-31)

The six grades come from the partner buying station's grading scheme. Class labels in code and data are `A` to `F`, displayed as "Grade A" to "Grade F".

| Grade | Old wording | New wording | Note |
|---|---|---|---|
| A | 6 to 9% | 6% to below 9% (6 ≤ MC < 9) | |
| B | 9 to 12% | 9% to below 12% (9 ≤ MC < 12) | |
| C | 12 to 15% | 12% to below 15% (12 ≤ MC < 15) | crosses PCA AO 02's 14% non-merchantable line |
| D | 15 to 18% | 15% to below 18% (15 ≤ MC < 18) | |
| E | above 18% | 18% and above (MC ≥ 18) | lowest payout |
| F | below 6% | below 6% (MC < 6) | penalized |

**What changed:** the old ranges shared their edges, so a reading of exactly 9%, 12%, 15% or 18% fitted two grades. **A reading on a boundary now belongs to the higher grade.** Each range includes its lower edge and excludes its upper edge.

**Where it matters in code:** the per-grade label check in Section 4 below, and any helper that maps a moisture reading to a grade. Training labels themselves come from the owner's pasa grading, not from a reading, so the pasa labels don't change.

```python
def grade_from_moisture(mc: float) -> str:
    """Map a moisture reading (%) to a station grade. Boundaries go to the higher grade."""
    if mc < 6:
        return "F"
    if mc < 9:
        return "A"
    if mc < 12:
        return "B"
    if mc < 15:
        return "C"
    if mc < 18:
        return "D"
    return "E"
```

**PCA framing (for any README text):** the grades are PCA-aligned. The station follows PCA guidelines on acceptable copra and splits PCA's brackets more finely for pricing. PCA AO 02 treats copra at 14% and above as non-merchantable to copra buyers. A buying station can still choose whether to purchase it, depending on how it processes and normalizes the copra.

---

## 3. Metrics to report

All model metrics are **per image**. Each face image's prediction comes from the outer StratifiedGroupKFold fold (5 folds, grouped by Cube ID = `batch_id`, stratified by grade) in which its cube was held out. There is no `copra_id`: which cubes came from which copra was not recorded, so cubes from the same copra can land in different folds and scores may be slightly optimistic. State this in the results. Report every outer-fold metric as **mean ± SD across the 5 folds**.

| Metric | Why it is needed | Objective |
|---|---|---|
| **Macro F1** (headline) | Weights all six grades equally. It is also the Optuna tuning objective. | 2, 4 |
| Overall accuracy | Familiar headline number; read it next to Macro F1 | 2, 4 |
| Confusion matrix (6×6) | Shows which grades get confused, usually neighbours | 2, 4 |
| Per-grade precision and recall | Shows whether any grade is being ignored | 2, 4 |
| **Grade A precision** (emphasis) | A false Grade A on wetter copra means the buyer overpays | 4 |
| Within-one-grade accuracy (secondary, new) | Share of predictions that are the true grade or an adjacent grade (A–B, B–C, C–D, D–E; F is adjacent to A). A one-grade miss costs much less than a three-grade miss. **Not yet in the thesis text.** | 4 |
| Model comparison table | RF, XGBoost, LightGBM and Logistic Regression side by side, all class-weighted, same features | 2 |
| Selected hyperparameters per outer fold | Shows how stable the tuning is | 2 |
| TreeSHAP importance by feature family | Texture (GLCM), color (HSV/LAB), edge (Canny), on the selected model only | 3 |
| Feature-family ablation | Color-only, texture-only, edge-only and combined runs, with Macro F1 for each | 1, 3 |
| Background-only check | Classifier on paper-only pixels, run before and after grey-paper normalization | validity |
| Pooled Brown-Duvel label check | 6 readings, each inside or outside its grade's range | 4 |
| Per-grade counts | Cubes and copra per grade, after outlier removal | all |
| Masking success | Share of face images with a clean mask; count of retaken or excluded images | 1 |

---

## 4. Targets for success

**Status:** these targets were set by the team on 2026-10-31. On 2026-11-01 the student put them into the thesis (Methodology 4.10) and the appeal letter without waiting for separate adviser confirmation; the adviser confirms them by endorsing the letter. They are the team's own pass marks, not values taken from the literature, so don't cite a source for them. Write the pipeline so each threshold is a config value, and report pass or fail for each one.

Reference points:
- Chance with six balanced grades is about 0.17 Macro F1 and about 17% accuracy.
- An earlier run on the old dataset reached about 70% accuracy. That result is why more data is being collected, so the targets sit above it. (Internal note only; the thesis does not mention this figure.)

| # | Target | Threshold | Objective |
|---|---|---|---|
| T1 | Selected model's Macro F1 (mean of 5 outer folds) | **≥ 0.75** (stretch goal 0.80) | 2, 4 |
| T2 | Recall of every grade (mean of outer folds) | **≥ 0.60** for each of A to F | 4 |
| T3 | Grade A precision (mean of outer folds) | **≥ 0.85** | 4 |
| T4 | Best tree model vs the Logistic Regression baseline | Macro F1 gap **larger than one outer-fold SD** | 2 |
| T5 | Background-only check, after normalization | Accuracy **near chance** (about 17%; investigate anything clearly above, e.g. over 25%) | validity |
| T6 | Pooled Brown-Duvel readings | **All 6** fall inside their grade's range (Section 2) | 4 |
| T7 | Feature families | The combined set is not worse than the best single family on Macro F1, and the TreeSHAP ranking has a plausible physical explanation | 1, 3 |
| T8 | Deployable model | Takes one face photo and returns one grade (A to F) with an on-request SHAP explanation, without crashing on any properly formatted image | 5 |

How to read the results:
- **T1 to T4 together** answer "accurate enough to support grading decisions" (objective 4).
- **If T5 fails,** lighting is leaking grade information, and T1 to T4 can't be trusted until it's fixed. Report it; don't tune around it.
- **If T6 fails for a grade,** report it as a label limitation for that grade. Don't relabel the data.
- **If a target is missed,** report the actual number and which grades were confused. A clear miss is still a valid thesis result, but say so plainly.

---

## 5. Not part of the goal

These are out of scope; see Section 13 of `model-development-instructions.md`.

- Exact moisture prediction (regression) or price calculation.
- Combining the three tree models into one voting or stacking ensemble.
- Deep learning or CNN models.
- Model-vs-meter majority-vote comparison, or an agreement rate against Brown-Duvel.
- Comparing copra sources, deliveries or warehouse areas.

---

## 6. Still open on the thesis side

- Whether within-one-grade accuracy goes into the thesis text (4.5.1 Accuracy requirement, 4.10 Validation, SRS).
- T1 to T7 are stated in Methodology 4.10 (added 2026-11-01). T8 is covered by the reliability check in 4.10.
