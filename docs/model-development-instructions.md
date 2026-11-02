# Model Development Instructions — Copra Moisture-Based Grade Classification

**Purpose of this document:** hand-off instructions for a separate Claude Code project/session that will actually write and run the machine learning pipeline (Python code, notebooks, scripts). This document is the single source of truth for *what* to build and *why*; it was compiled from an undergraduate BS Computer Science thesis project's Methodology chapter and supporting planning documents. Follow it precisely — every design decision below was deliberated and defended (or is pending a proposal defense's approval), not arbitrary.

**What this document is not:** it is not the thesis manuscript, and the model-development session should not write manuscript prose. It is a technical build spec. Code, comments, and README output from that session should be normal software-engineering style, not academic voice.

**Study title:** *Copra Moisture-Based Grade Classification Using Image-Based Feature Extraction for Quality Grading*

---

## 1. Domain Context (read this first)

Copra is dried coconut meat, traded commercially in the Philippines. Moisture content (MC) is the main determinant of its value. The Philippine Coconut Authority (PCA) regulates copra pricing under **Administrative Order No. 02**, which is background context only for this study. The classes this model predicts are the **partner buying station's own six moisture grades**, not PCA grades:

| Class (`copra_class`) | Moisture Range | Commercial Impact |
|---|---|---|
| **Grade A** | 6% to <9% MC | — |
| **Grade B** | 9% to <12% MC | |
| **Grade C** | 12% to <15% MC | |
| **Grade D** | 15% to <18% MC | |
| **Grade E** | 18% and above MC | Lowest payout / quality |
| **Grade F** | below 6% MC | Penalized |

Write the ranges exactly as given. A value on a boundary (9, 12, 15, 18%) belongs to the higher grade. The source of this scheme is the buying station (its owner). It is not PCA and not a published citation, so do not attach a citation to it. **Model class labels must be `A`, `B`, `C`, `D`, `E`, `F` (the `copra_class` column), displayed as "Grade A" … "Grade F".** Do not introduce other grade names (e.g. "Resecada Bodega," "Semi-Resecada," "Corriente," PCA "Class 1/2/3") anywhere in code, comments, class labels, UI strings, or output.

**Why classification, not regression:** grading is inherently threshold-based. 8.3% MC and 8.7% MC fall in the same grade with the same consequence. A regression model predicting exact decimal moisture percentages would be solving a harder, noisier problem than the actual business decision requires, and adjacent moisture percentages have negligible visually distinguishable texture/color variance. **The system classifies only — it never predicts a price.** PCA AO 02's actual discount table assigns a distinct discount at every 0.1% increment across 6.1%–13.9%, which is a genuinely continuous schedule a six-grade output cannot and should not try to reproduce. Do not build any price-prediction or regression head into the model.

**Ground truth:** the buying station owner grades each whole copra by the traditional **pasa** method, and the copra's cubes inherit that grade. The owner also confirmed that the collected samples in each class are accurately graded. A **Brown-Duvel** moisture reading (the traditional method PCA AO 02 designates for copra moisture) is taken once per class on a pooled ~100 g portion of that class's cubes. It must fall inside the class's moisture range. This is a class-level sanity check on the labels, not the main result. It cannot catch an individual mislabeled copra.

---

## 2. What Already Exists vs. What This Session Builds

**Already built, out of scope for this session — do not redesign or reimplement:**
- **Jotter**, a React Native/Expo + native Kotlin mobile app used in the field to capture data. For each whole copra, the station owner grades it by pasa and it is cut into 1 to 3 cubes (small copra 1, large copra up to 3). Jotter guides a clerk through photographing each cube's four cut meat faces (capture itself is delegated to the third-party Open Camera app). It logs Cube ID, Face ID (1–4), timestamp, and the confirmed grade (A–F), plus one class-level record per grade that holds the pooled Brown-Duvel reading. It syncs to a Supabase (PostgreSQL + Storage) backend. Jotter is a **data collection method only** — it has no role in the classification pipeline itself, and its own capture/sync logic should not be touched or reasoned about beyond understanding the shape of the data it hands off.
- The eventual **Streamlit deployment interface** is a separate, later concern (Section 12 below covers only what the pipeline needs to hand off to it).

**This session's actual scope — everything from a compiled dataset export through to a trained, evaluated, explainable model:**
1. Preprocessing (background masking, lighting normalization, scale normalization)
2. Feature extraction (three permanent feature families, per face image)
3. Data cleaning / outlier detection
4. Dataset splitting (leakage-safe, StratifiedGroupKFold grouped by Cube ID, `batch_id`)
5. Class-imbalance handling (augmentation + class weighting)
6. Model training and hyperparameter tuning (4 candidate models)
7. Model comparison and algorithm selection
8. Explainability (TreeSHAP)
9. Label check reporting (pooled Brown-Duvel readings vs. class ranges)
10. Evaluation (per-image, on held-out folds)
11. Producing the final deployable model artifact

---

## 3. Input: What the Pipeline Receives

- A dataset export: `copra-dataset.csv` with one row per photo and the columns `id` (unique photo ID), `batch_id` (the **Cube ID**, shared by the cube's 4 photos), `copra_class` (`A`–`F`), and `path` (`photos/<id>.jpeg`). The `photos/` folder sits beside `copra-dataset.csv`. Make the image root path configurable.
- Per cube: **4 photos**, one per cut meat face. Per copra: 1 to 3 cubes, so 4 to 12 photos.
- **Minimum dataset size: 50 cubes per class × 6 classes = 300 cubes, 200 photos per class, 1,200 raw photos before augmentation.** Collection is still ongoing and has no upper limit. Per-class cube counts are kept within 10% of each other (smallest class at least 90% of the largest). Report the **cube count per class**.
- **The camera stays fixed at 10 cm for every capture, with no rotation.** Settings are locked: ISO 100, 1/60 s, manual white balance 6000 K, manual focus 0.10 m, 4:3, JPEG. Each cut meat face is a single shot on mid-grey construction paper under two white lights. Background and framing never change between photos, so one fixed background-model setting (Section 4 below) works across the whole dataset. The two lights dim slowly during a session, and Section 4's lighting normalization reduces that drift.
- The class label is the `copra_class` column (the owner's pasa grade). Do not derive it from a moisture reading. There is no per-sample moisture reading in the export.

---

## 4. Preprocessing: Background Masking and Lighting Normalization

**Algorithm: background model + GrabCut** (Rother et al., 2004), applied per face image. Reference implementation: `src/copra_grading/preprocessing/background.py` in the model repo.

**Otsu's thresholding was tried first and failed.** A pilot on 136 face photos (both methods inspected) found Otsu failed on 136 of 136. In 135 the mask spilled to the photo border and took in the grey paper (typically about 77% of the frame). In the other one, the mask kept only the white meat and lost the brown skin. The cause is the copra itself: its white meat side is bright (L ~200), its brown skin side is dark (L ~30), and the mid-grey paper (L ~100-170) sits between them and drifts across the frame with the lighting, so one global threshold lumps the paper in with the meat. GrabCut was one of the listed fallbacks and is now the method. Only the Otsu border count was checked by rule. All other pilot counts are visual judgement and approximate.

Background model + GrabCut result on the same 136 photos: about 123 clean (about 90%), about 12 with a small edge leak (a thin shadow strip), 1 clear failure (a large shadow patch included). No mask touched the frame edge.

Steps:
1. Work on a downscaled copy (speed; GrabCut is the slow part).
2. Fit a smooth quadratic surface to each LAB channel using only the image border band (assumed to be paper). This captures the lighting gradient. Do one robust refit that drops border outliers (specks, a copra corner touching the edge).
3. Score every pixel by its distance from that surface, in units of the border's robust spread. Pixels above `threshold_k = 4` are copra seeds.
4. Seed GrabCut: eroded seeds = sure copra, the exterior of the dilated seeds + border band = sure background, the rest = probable. Run 3 iterations.
5. Keep the largest connected component and fill its holes. Upscale the mask back to the input resolution.
6. Apply the mask to the original image.

**The mask keeps both the white meat and the brown skin.** The cubes are curved, so some outer brown skin shows at the cut-face edges. Skin pixels shift the color averages slightly. Treat this as a known limitation.

**Bad-mask rule:** a face image whose mask is clearly wrong (like the pilot's one failure) is retaken if the cube is still available, otherwise excluded. The Section 6 outlier step stays as the later safety net.

**Lighting normalization (after masking, before feature extraction).** The two battery-powered lights dim slowly during a session. Grades were photographed in sequence (all Grade A first, then the next grade), so brightness drift can line up with grade. For each photo, measure the paper's average color from the fitted background surface and scale the photo so the paper matches one fixed reference grey. Then resize to a fixed pixel scale, to normalize for physical size variation between cubes.

**The masked, lighting-normalized, scale-normalized image is what feeds every downstream feature extraction step, never the raw, unmasked photo.** Streamlit applies the same masking and normalization to the submitted photo (Section 12).

Keep the masking method behind a single function so it can be swapped without touching feature extraction code.

Recommended libraries: **OpenCV** (including its GrabCut implementation) and/or **Scikit-Image**, both named in the study's own tool inventory.

---

## 5. Feature Extraction (per face image — three permanent feature families)

Run on the **background-masked, lighting-normalized, scale-normalized** image only. All three families run in parallel (none depends on another's output) and their outputs concatenate into **one combined feature vector per face image**. This is not optional/ablation-only — all three ship in the default pipeline; ablation (Section 10) exists to *measure* each family's marginal contribution, not to gate whether it's included.

### 5a. GLCM (Gray-Level Co-occurrence Matrix) — Texture

Captures how pixel intensities co-occur with their spatial neighbors. Compute four statistics from the co-occurrence matrix:
- **Contrast** — local intensity variation (higher = rougher texture)
- **Homogeneity** — closeness of the pixel-pair distribution to the diagonal (higher = smoother/more uniform)
- **Energy** — sum of squared matrix elements (higher = more orderly/uniform)
- **Entropy** — randomness/disorder of the texture

Compute at **multiple mathematical angles (0°, 45°, 90°, 135°) and multiple pixel distances**, all from the single masked image — this is a GLCM computation parameter, not additional photography. Consider computing at a small set of distances (e.g. 1, 2, 3 px) and either averaging or keeping each distance/angle combination as a separate feature — either is defensible; document whichever choice is made.

**Physical rationale:** wet copra has a smoother, more uniform meat surface; as it dries, the surface roughens, cracks, and becomes irregular. GLCM is built to quantify exactly that.

### 5b. Color Space Transforms — HSV and LAB

Convert the masked image's RGB pixels into:
- **HSV** — separates Hue and Saturation from Value/brightness, so color signal stays robust to any residual lighting variation, less confounded than raw RGB would be even with the slow dimming of the capture lights.
- **LAB** — perceptually uniform; its numeric distances correspond to how a human eye perceives color difference, chosen specifically because it matches the gray-to-brown shift copra undergoes while drying.

Compute per-channel statistics from both color spaces per face image (e.g. mean, standard deviation, and/or histogram-based statistics per channel). **Do not use photometric augmentation (brightness/contrast/color-shift)** anywhere in this pipeline — it would directly corrupt the HSV/LAB color signal this feature family is built to measure (see Section 7).

**Physical rationale:** moisture loss visibly shifts copra from pale/white toward tan/brown; color is one of the more direct visual proxies for drying stage.

### 5c. Canny Edge/Contour Density — Edges

Run Canny edge detection on the masked image; compute edge/contour density (proportion of edge pixels, and/or contour distribution statistics) from the resulting edge map.

**Physical rationale:** as copra dries and shrinks, its surface develops more visible cracks, fissures, and contour irregularity. Edge density is a proxy for that structural change.

### Output of Section 5

Each face image → **one row of features** (GLCM stats + HSV/LAB stats + Canny edge density, concatenated), carrying `batch_id` and `copra_class` through. Each cube's 4 photos → 4 rows, all sharing the copra's grade.

---

## 6. Data Cleaning and Outlier Detection

**Completeness is already guaranteed upstream.** Jotter enforces that every synced entry has at least 1 cube with all 4 face images and a confirmed grade before it's marked complete. Do not write imputation logic for missing images or grades; that scenario should not reach this pipeline. What this pipeline does still need to do:

1. **Duplicate Cube ID resolution** at merge/export time (multiple field clerks may have used offline-first apps that couldn't cross-check IDs with each other in real time). Detect and resolve duplicates across the merged dataset before proceeding.
2. **Outlier removal on extracted feature values** (not raw images) — after Section 5's feature extraction, flag anomalous values (e.g. glare, an unexpected object in frame) that passed the file-level completeness check but would still produce unreliable features. A value is an outlier if **either** rule flags it:
   - **Z-score:** `z = (x - mean) / std`; flag `|z| > 3`.
   - **IQR:** `IQR = Q3 - Q1`; flag values below `Q1 - 1.5×IQR` or above `Q3 + 1.5×IQR`.

   Drop flagged values. Do not impute. Do not relax the thresholds; if too much data is lost, the answer is to collect more data. Cite: Aggarwal, C. C. (2017). *Outlier Analysis* (2nd ed.). Springer.

---

## 7. Dataset Splitting and Augmentation

**Order matters and is a hard constraint: split before augmenting, never the reverse.** This was an actual bug caught and fixed during the thesis's own diagram planning — get it right the first time.

### 7a. StratifiedGroupKFold Split

Use **five-fold StratifiedGroupKFold (scikit-learn), grouped by `batch_id` (the Cube ID) and stratified by `copra_class`**. This must run *before* augmentation. The 4 photos of one cube — and every augmented variant later generated from them — must stay together in the same fold, so the model never sees a face of a test cube in training. Stratifying keeps a similar A–F mix in each fold. Which cubes came from which copra was not recorded, so there is no `copra_id` and cubes from the same copra can land in different folds. They share a grade and drying history, so scores may be slightly optimistic; state this in the results. Report the cube count per class (and per fold).

### 7b. Class-Weighted Geometric Augmentation

**Geometric only: rotations and flips.** No photometric augmentation (brightness/contrast/color-shift) — it would corrupt the HSV/LAB color features that are the actual measured signal. No synthetic interpolation methods like SMOTE — these generate non-physical feature vectors, a documented risk on an already-small dataset (Abdelhamid, M., & Desai, A. (2024). *Balancing the scales: A comprehensive study on tackling class imbalance in binary classification.* arXiv:2409.19751).

Apply an **uneven multiplier across classes**: a greater number of augmented copies for the smaller classes than for the larger ones (the class with the fewest images is expected to need the biggest multiplier), partially correcting class imbalance using real rotated/flipped copies of genuine samples rather than synthetic ones.

**Open item — not finalized:** the exact multiplier per class has not been decided by the thesis team. Make the per-class multiplier a configurable parameter (e.g. a dict `{'A': k_a, 'B': k_b, ..., 'F': k_f}`), pick a reasonable starting default, based on the raw per-class image counts, that meaningfully oversamples the smaller classes without producing excessive near-duplicate images (e.g. start by roughly balancing effective class counts post-augmentation), and clearly document that this is a tunable default pending confirmation, not a fixed decision.

### 7c. Feature Scaling

**Not required** for the tree-based ensemble models (Random Forest, XGBoost, LightGBM) — tree splits are scale-invariant. **Required** for Logistic Regression specifically (standardization) — apply it only to that model's inputs, not the others.

---

## 8. Model Training, Tuning, and Selection

### 8a. The Four Candidate Models

| Model | Role | Type | Feature Scaling |
|---|---|---|---|
| **Logistic Regression** | Baseline only — never eligible for deployment | Linear | Yes (standardized) |
| **Random Forest** | Ensemble candidate | Bagging (many trees on bootstrapped samples/feature subsets, averaged/voted) | No |
| **XGBoost** | Ensemble candidate | Gradient boosting (sequential trees, each correcting prior errors) | No |
| **LightGBM** | Ensemble candidate | Histogram-based gradient boosting (bins continuous features for faster splits) | No |

**Important framing:** "ensemble" in the study's title refers to each of Random Forest/XGBoost/LightGBM being internally an ensemble method (many trees combined), **not** to stacking or combining all three into one meta-model. They are compared against each other as separate candidates, not blended together. Do not build a voting/stacking meta-ensemble across the three — that would misrepresent what the study claims and was an explicit defense-panel correction point (a prior title implying a combined meta-ensemble was flagged and revised).

**All four models must be class-weighted natively** (e.g. `class_weight='balanced'` or explicit per-class weights in scikit-learn/XGBoost/LightGBM's own APIs), including Logistic Regression. Weighting the baseline identically to the ensembles is deliberate: it keeps the Section 8c comparison isolating algorithm strength, not which models happened to get help with class imbalance and which didn't. Class weighting is the *model-level* imbalance correction; Section 7b's augmentation is the *data-level* correction — both are used together, not as alternatives to each other.

Recommended libraries: **Scikit-Learn** (Logistic Regression, preprocessing; Pedregosa et al., 2011), **XGBoost** (Chen & Guestrin, 2016), **LightGBM** (Ke et al., 2017).

### 8b. Hyperparameter Tuning

Use **Optuna** (Akiba et al., 2019) to tune all four models' hyperparameters. **Optimize for Macro F1, not raw accuracy.** Macro F1 averages the F1 score across all six classes equally, so the largest class cannot dominate the score the way plain accuracy would let it. This matters because Grade A precision (see Section 11) is the practically important part of this system's accuracy, not overall accuracy. Cite: Maia, W. F. et al. (2024). *Multi-level product category prediction through text classification.* arXiv:2403.01638 (used in the thesis for the Macro-F1-under-imbalance justification).

No specific hyperparameter search space or trial count has been fixed by the thesis team — choose reasonable, well-documented search spaces per model (e.g. `n_estimators`, `max_depth`/`num_leaves`, `learning_rate`, `min_child_samples`/`min_samples_leaf`, regularization terms) and a trial budget appropriate to the small dataset size (a few thousand rows post-augmentation) — this does not need enormous trial counts. Document whatever search space and trial count you pick; do not present it as a decision the thesis team already made.

**Tuning must be nested (decided 2026-10-14).** Tuning on the same five folds used for reporting makes the reported Macro F1 optimistic, since the hyperparameters get picked to suit those exact test folds.

- **Outer loop:** the five StratifiedGroupKFold folds from §7a. Used only to report scores. Optuna never sees the held-out outer fold.
- **Inner loop:** inside each outer training portion, a second `StratifiedGroupKFold(n_splits=3)`, grouped by `batch_id` and stratified by `copra_class`. Each Optuna trial is scored by its mean Macro F1 across the three inner folds.
- **Augmentation:** apply §7b augmentation only to inner training folds during tuning, and to the whole outer training portion when refitting. Never augment before a split, at either level. Logistic Regression's standardization is fit on training data only, at both levels (put it in a `Pipeline`).
- **Per outer fold:** refit the best trial's parameters on the full (augmented) outer training portion, then score once on the held-out outer fold.
- **Report:** mean ± SD Macro F1 across the five outer folds, plus each outer fold's chosen hyperparameters (tuning-stability check).
- **Final deployed model:** after selection (§8c), re-run Optuna for the selected algorithm on the full dataset with a grouped five-fold split, then train on all data. Its expected performance is the nested estimate; there is no held-out data left to re-score it.

### 8c. Algorithm Selection vs. Deployment Configuration — do not conflate these

This is the single most important structural distinction in the whole pipeline; keep them as two clearly separate stages in code, not one:

1. **Algorithm selection (this section):** train all four models on the **per-image feature set** (one feature vector per photographed face). Whichever of the three ensembles (Random Forest, XGBoost, LightGBM) scores best on mean outer-fold Macro F1 from the nested CV in §8b is the **selected algorithm**, and it is what gets deployed. Logistic Regression is the baseline reference point only and is **never** eligible for deployment regardless of its score.
2. **Evaluation (Section 11):** the selected algorithm's per-image metrics come from the nested CV's held-out outer folds (each outer fold's model, tuned on the inner folds only). Do not re-tune or retrain on the outer folds before scoring them. Picking the best of three by outer scores adds a small optimistic bias; note it in the results.

The deployed model always takes exactly one face image, since it was trained on per-image data from the start.

---

## 9. Label Check Reporting — Pooled Brown-Duvel Readings

The model-vs-meter majority-vote comparison and the old per-sample spot-check are **removed**. Do not build them. The only Brown-Duvel data in this study is **six pooled readings**, one per class, each taken on a ~100 g portion pooled from that class's cubes after all of them were photographed.

If the readings are supplied as a small file (class, reading), report for each class whether the reading falls inside that class's moisture range from Section 1. State it as a sanity check on the labels. Do not report it as an agreement rate and do not present it as the main result. It verifies the class as a whole and cannot detect an individual mislabeled copra. The main evaluation is Section 11.

---

## 10. Explainability (TreeSHAP)

Run **TreeSHAP** (Lundberg & Lee, 2017 — the tree-optimized SHAP variant, exact and fast specifically for tree-based models) against **only the selected algorithm** from Section 8 (not all three ensemble candidates separately — running it three times over was explicitly judged not worth the effort for a two-person team).

**Aggregate the resulting feature-importance scores by feature family** — texture (GLCM), color (HSV/LAB), edge (Canny) — rather than reporting per individual feature. This answers "how much did texture vs. color vs. edges matter overall to the model's decisions," which is a separate question from overall accuracy.

Separately, at deployment time, the eventual Streamlit interface computes a **live, per-classification SHAP explanation** against the actual single-photo deployed model for one specific submitted image — that is a different computation from this aggregate research finding and happens outside this pipeline's batch scope, though this pipeline should expose whatever SHAP-computation function/API the deployment layer will call.

Every top SHAP feature identified should ideally be checked against a plausible physical/domain explanation (e.g., "edge density mattered most for separating Grade A from Grade F" is domain-plausible; a nonsensical top feature is worth investigating as a possible leakage or preprocessing artifact before trusting the result).

---

## 11. Evaluation

**Pass marks for every metric below are in `model-acceptance-criteria.md`.** Check results against that file and fill in its pass/fail table.

Report all of the following — do not settle for a single accuracy number:

- **Per-image metrics on the held-out folds:** confusion matrix, overall accuracy, Macro F1, and per-class precision and recall. Each held-out photo is scored individually against its copra's grade.
- **Grade A precision, with particular emphasis:** a false Grade A on wetter copra means the buyer overpays, a direct financial loss.
- **Macro F1** (the same metric used for hyperparameter optimization — also report it here as a headline evaluation number, not only as the tuning objective).
- **Per-class counts:** report cube count per class (see Section 7a).
- **Background-only check:** train a simple classifier on paper-only pixels (no copra) to predict the grade. Accuracy near chance (about 17% for six grades) means lighting does not reveal the grade. Clearly above chance means lighting leaks grade information. Run it before and after the Section 4 lighting normalization and report both.
- **Feature-family ablation** (train on color-only, texture-only, edge-only, and combined feature sets; report the accuracy progression) — this is what justifies keeping all three feature families in the default pipeline (Section 5), particularly Canny edge density, which is the newest of the three additions. Not optional — build this as a standard evaluation step, not an afterthought.
- Consider running a **PCA or t-SNE visualization** on the extracted feature space (GLCM + HSV/LAB + Canny combined) before or alongside model training, purely as a sanity check on whether the six classes look separable at all in feature space. If they overlap heavily, that's a feature-engineering problem, not something more model tuning will fix — flag it rather than continuing to tune blindly.

---

## 12. Final Deployable Artifact

The artifact this whole pipeline produces is: **the algorithm selected in Section 8c**, trained on the per-image feature set — no separate retraining step, since that per-image training is already the same shape of input a live user submits. This is the only model that gets serialized and handed off for deployment.

Serialize the model (and any accompanying preprocessing/feature-extraction config it depends on, e.g. the background-model and GrabCut parameters and the reference grey, the specific angle/distance set used for GLCM, the exact feature ordering) in a way a separate Streamlit app can load and call directly — format not yet finalized by the thesis team (e.g., a standard `joblib`/pickle dump for scikit-learn-compatible models, or each library's native serialization for XGBoost/LightGBM, is reasonable; document whichever is chosen and package the feature-extraction code as an importable, reusable module rather than notebook-only code, since deployment needs to call the exact same preprocessing/feature-extraction logic used during training).

**Deployment context (for downstream awareness only, not this session's build target):** the artifact will eventually be loaded by a Streamlit web interface where a buying-station user submits one photo and receives a moisture grade (A–F) plus an optional on-request SHAP explanation. The interface applies the same masking and lighting normalization to the submitted photo before feature extraction. No price adjustment is ever computed by this system.

---

## 13. Explicit Non-Goals — Do Not Build These

- **No price/discount prediction or regression output.** Classification only, ever.
- **No end-to-end CNN or deep-learning image model.** The dataset (at least 1,200 raw images before augmentation, more after) is judged too small for reliable deep-learning fine-tuning without overfitting; the handcrafted-feature + tree-ensemble approach is the deliberate choice, more interpretable and appropriate at this data scale (Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). *Why do tree-based models still outperform deep learning on tabular data?* arXiv:2207.08815).
- **No meta-ensemble/stacking across Random Forest, XGBoost, and LightGBM.** They are compared, not combined.
- **No SMOTE or other synthetic interpolation for class balance.** Use class weighting + real geometric augmentation only.
- **No photometric augmentation** (brightness/contrast/color/hue shifts) at any point — it corrupts the HSV/LAB color signal the pipeline depends on.
- **No external "wet"/"dried" public dataset merged in** unless a documented, uncertainty-acknowledged label-mapping protocol is separately agreed with the thesis team first — default assumption is this study uses only its own field-collected data.
- **Do not use other grade names** ("Resecada," "Bodega," "Semi-Resecada," "Corriente," "Non-Merchantable," PCA "Class 1/2/3") anywhere in code, labels, or generated output — use `Grade A` … `Grade F` only, per Section 1.

---

## 14. Objective Traceability Map

Every pipeline stage should be traceable back to one of the study's five specific objectives — useful for the model-development session to self-check coverage:

| Objective | What It Requires | Pipeline Section(s) |
|---|---|---|
| 1 | Extract texture/color/edge features from per-image cut-face photos | §5 (Feature Extraction) |
| 2 | Train and compare RF/XGBoost/LightGBM vs. Logistic Regression baseline | §8 (Model Training, Tuning, Selection) |
| 3 | Evaluate which feature families contribute most via TreeSHAP | §10 (Explainability) |
| 4 | Verify training labels through one pooled Brown-Duvel reading per class checked against that class's range, and assess whether the trained model's classifications are accurate enough to support grading decisions at a real buying station | §9 (Label Check Reporting) and §11 (Evaluation) |
| 5 | Implement a functional prototype that applies the trained model to classify copra images and output the moisture grade (A–F) | §12 (Final Artifact) — full Streamlit build is a separate, later effort |

---

## 15. Open Items the Thesis Team Has Not Yet Finalized

Flag these explicitly in any generated code/README rather than silently deciding them as if settled — implement configurable defaults, but say clearly that they are defaults pending confirmation:

1. **Exact per-class augmentation multiplier** — §7b.
2. **Optuna search space and trial budget per model** — §8b.
3. **Exact image-file delivery format alongside the CSV export** (folder convention / zip / cloud folder) — §3.
4. **Model artifact serialization format** for handoff to Streamlit — §12.
5. **GLCM distance set** (this document suggests e.g. 1–3 px as a reasonable default; not fixed by the thesis team) — §5a.

If the actual field dataset is not yet available when this session starts, build and validate the full pipeline against a small synthetic/placeholder dataset with the same schema (`id`, `batch_id`, `copra_class` A–F, `path`; 4 photos per cube) so the code path is proven correct and ready to run the moment real data lands — do not block all development on waiting for the dataset.

---

## 16. Source Documents (for deeper context if needed)

This instruction file was derived from, and should stay consistent with, these files in the thesis project repository (not needed by the coding session unless a question arises that this document doesn't answer):
- `thesis-project-overview.md` — full project history and context
- `Methodology/objective.md` — the five specific objectives and research questions verbatim
- `Methodology/4.3-research-procedure.md`, `4.4-concept.md`, `4.5-analysis-and-design.md`, `4.8-software-development-tools.md` — formal methodology chapter sections
- `process.md` — an earlier, shorter technical walkthrough covering the same pipeline
- `models.md` — a plain-language explainer of what each technique does and why
