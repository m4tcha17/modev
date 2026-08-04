# Model Development Instructions — Copra Moisture-Based Grade Classification

**Purpose of this document:** hand-off instructions for a separate Claude Code project/session that will actually write and run the machine learning pipeline (Python code, notebooks, scripts). This document is the single source of truth for *what* to build and *why*; it was compiled from an undergraduate BS Computer Science thesis project's Methodology chapter and supporting planning documents. Follow it precisely — every design decision below was deliberated and defended (or is pending a proposal defense's approval), not arbitrary.

**What this document is not:** it is not the thesis manuscript, and the model-development session should not write manuscript prose. It is a technical build spec. Code, comments, and README output from that session should be normal software-engineering style, not academic voice.

**Study title:** *Copra Moisture-Based Grade Classification Using Image-Based Feature Extraction for Quality Grading*

---

## 1. Domain Context (read this first)

Copra is dried coconut meat, traded commercially in the Philippines. The Philippine Coconut Authority (PCA) regulates its grading under **Administrative Order No. 02**, which sets moisture-content (MC) thresholds that determine price and acceptance:

| Class (this study's label) | Moisture Range | PCA AO 02 Status | Commercial Impact |
|---|---|---|---|
| **Class 1** | ≤ 6.0% MC | order's term "resecada/bodega" — the 6% equilibrium point, not a named range | 0% deduction (premium) |
| **Class 2** | 6.1% – 13.9% MC | **unnamed by the order** — described only by its deduction schedule | Graduated price/weight deductions |
| **Class 3** | ≥ 14.0% MC | order's own regulatory status: "non-merchantable" | Automatic rejection (aflatoxin/mold risk) |

Important compliance note: PCA AO 02 does **not** officially name the middle bracket. "Class 1/2/3" are this study's own labels for the three brackets, not PCA-assigned grade names. Do not introduce commercial grade names (e.g. "Resecada Bodega," "Semi-Resecada," "Corriente") anywhere in code, comments, class labels, UI strings, or output — a prior defense-panel review flagged this exact issue and it was fully corrected across the manuscript. **Model class labels must be `1`, `2`, `3` (or `class_1`/`class_2`/`class_3`), never a commercial grade name.**

**Why classification, not regression:** grading is inherently threshold-based. 8.3% MC and 8.7% MC fall in the same PCA class with the same accept/reject/deduction-tier consequence. A regression model predicting exact decimal moisture percentages would be solving a harder, noisier problem than the actual business decision requires, and adjacent moisture percentages have negligible visually distinguishable texture/color variance. **The system classifies only — it never predicts a price.** PCA AO 02's actual discount table assigns a distinct discount at every 0.1% increment within Class 2, which is a genuinely continuous schedule a 3-class output cannot and should not try to reproduce. Do not build any price-prediction or regression head into the model.

**Ground truth instrument:** a Brown-Duvel moisture meter (the traditional method PCA AO 02 itself designates for measuring copra moisture) provides the confirmed moisture reading per sample, which is what the class label is derived from.

---

## 2. What Already Exists vs. What This Session Builds

**Already built, out of scope for this session — do not redesign or reimplement:**
- **Jotter**, a React Native/Expo + native Kotlin mobile app used in the field to capture data. It guides a buying-station clerk through six standardized photo angles per physical sample, logs the Brown-Duvel moisture reading, and syncs to a Supabase (PostgreSQL + Storage) backend. Jotter is a **data collection method only** — it has no role in the classification pipeline itself, and its own capture/sync logic should not be touched or reasoned about beyond understanding the shape of the data it hands off.
- The eventual **Streamlit deployment interface** is a separate, later concern (Section 9 below covers only what the pipeline needs to hand off to it).

**This session's actual scope — everything from a compiled dataset export through to a trained, evaluated, explainable model:**
1. Preprocessing (background masking)
2. Feature extraction (three permanent feature families, per angle image)
3. Data cleaning / outlier detection
4. Dataset splitting (leakage-safe)
5. Class-imbalance handling (augmentation + class weighting)
6. Model training and hyperparameter tuning (4 candidate models)
7. Model comparison and algorithm selection
8. Explainability (TreeSHAP)
9. Angle-wise comparison to determine deployment configuration
10. Evaluation (including benchmarking against a human baseline)
11. Producing the final deployable model artifact

---

## 3. Input: What the Pipeline Receives

- A dataset export: one CSV row per angle image, containing at minimum `Sample_ID`, `Angle_ID`, `timestamp`, and the confirmed `moisture_reading` (shared by all six rows of the same sample), plus the referenced image files delivered alongside it (exact delivery format — flat folder, zip, cloud folder — is not yet finalized by the data-collection side; build the loading code to take a configurable image root path rather than hardcoding a layout).
- Per physical sample: **six angle images** — one top-down, four sides at ~90° rotational intervals, one bottom (flipped) — sharing one `Sample_ID` and one moisture reading.
- **Target dataset size: 180–240 physical samples, 60–80 per class** (a working minimum, not a hard ceiling — collection may exceed this). That is **1,080–1,440 raw images before augmentation**. Expect natural class imbalance skewed toward Class 2 (buying-station stock clusters mid-range because traders pre-dry to avoid rejection); Class 1 and Class 3 will likely be the minority classes.
- **The camera was fixed in one position for all six captures; the physical sample itself was rotated beneath it.** This is why a single global preprocessing calibration (Step 4 below) works across the whole six-image set — background and framing never change between angles for a given sample.
- Derive the class label per sample from `moisture_reading` using the thresholds in Section 1's table — do not expect a pre-populated class column; compute it directly from the exact 6.0/6.1/14.0 boundaries given, and treat samples near those two boundaries as a reportable edge case (see Section 10).

---

## 4. Preprocessing: Background Masking

**Algorithm: Otsu's thresholding** (Otsu, 1979), applied per angle image.

Otsu's method automatically finds a single grayscale intensity threshold that best separates an image into two classes (foreground/background) by maximizing between-class variance (equivalently, minimizing within-class variance). It is reliable here specifically *because* the camera never moved between shots — only the sample rotated beneath it — so background and lighting geometry stay consistent across all six images of a given sample, meaning one global threshold per image (no per-angle recalibration) is sufficient.

Steps:
1. Convert the angle image to grayscale (or an appropriate single channel).
2. Compute the image's intensity histogram.
3. Run Otsu's algorithm to find the threshold minimizing intra-class intensity variance.
4. Apply the threshold to produce a binary mask isolating copra pixels from the tray/surface behind them.
5. **The masked image (copra pixels only) is what feeds every downstream feature extraction step — never the raw, unmasked photo.**

**Documented fallback, only if Otsu proves insufficient under real field lighting** (do not implement preemptively — Otsu is the primary/default method; only reach for these if evaluation shows Otsu masks are unreliable): adaptive/local thresholding, an HSV/saturation-based mask calibrated to the known tray color, or GrabCut (graph-cut segmentation seeded with a bounding box). Ambient lighting at the buying station is **not** physically controlled (daylight + indoor lighting shifting through the day) — this is a stated scope decision, not an oversight, so the masking and every downstream feature must be expected to tolerate lighting variability rather than assume a studio-controlled shot. Build in a way that makes it easy to swap the masking method later without touching feature extraction code.

Recommended libraries: **OpenCV** and/or **Scikit-Image** (both are the tools named in the study's own tool inventory).

---

## 5. Feature Extraction (per angle image — three permanent feature families)

Run on the **background-masked** image only. All three families run in parallel (none depends on another's output) and their outputs concatenate into **one combined feature vector per angle image**. This is not optional/ablation-only — all three ship in the default pipeline; ablation (Section 10) exists to *measure* each family's marginal contribution, not to gate whether it's included.

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
- **HSV** — separates Hue and Saturation from Value/brightness, so color signal is less confounded by uncontrolled ambient lighting than raw RGB would be.
- **LAB** — perceptually uniform; its numeric distances correspond to how a human eye perceives color difference, chosen specifically because it matches the gray-to-brown shift copra undergoes while drying.

Compute per-channel statistics from both color spaces per angle image (e.g. mean, standard deviation, and/or histogram-based statistics per channel). **Do not use photometric augmentation (brightness/contrast/color-shift)** anywhere in this pipeline — it would directly corrupt the HSV/LAB color signal this feature family is built to measure (see Section 7).

**Physical rationale:** moisture loss visibly shifts copra from pale/white toward tan/brown; color is one of the more direct visual proxies for drying stage.

### 5c. Canny Edge/Contour Density — Edges

Run Canny edge detection on the masked image; compute edge/contour density (proportion of edge pixels, and/or contour distribution statistics) from the resulting edge map.

**Physical rationale:** as copra dries and shrinks, its surface develops more visible cracks, fissures, and contour irregularity. Edge density is a proxy for that structural change.

### Output of Section 5

Each angle image → **one row of features** (GLCM stats + HSV/LAB stats + Canny edge density, concatenated). Six angle images per physical sample → six such rows before any per-angle vs. combined comparison happens (that comparison is Section 9 below, not this step).

---

## 6. Data Cleaning and Outlier Detection

**Completeness is already guaranteed upstream.** Jotter enforces that every synced entry has all six angle images and a confirmed moisture reading before it's ever marked complete — do not write imputation logic for missing images or readings; that scenario should not reach this pipeline. What this pipeline does still need to do:

1. **Duplicate `Sample_ID` resolution** at merge/export time (multiple field clerks may have used offline-first apps that couldn't cross-check IDs with each other in real time). Detect and resolve duplicates across the merged dataset before proceeding.
2. **Outlier detection on extracted feature values** (not raw images) — after Section 5's feature extraction, flag anomalous rows (e.g. glare, an unexpected object in frame) that passed the file-level completeness check but would still produce unreliable features. Choose **one** of:
   - **Z-score:** `z = (x - mean) / std`; flag `|z| > 3` as an outlier.
   - **IQR:** `IQR = Q3 - Q1`; flag values below `Q1 - 1.5×IQR` or above `Q3 + 1.5×IQR`.

   **Open item — not finalized by the thesis team:** which method, and its exact cutoff/multiplier, has not been decided. Implement both as configurable options behind a single interface, default to IQR (more robust to skewed data, less sensitive to the outliers it's trying to detect than Z-score's own mean/std are), and **flag this choice explicitly in your output/README as a parameter the student should confirm** rather than silently deciding it's final. Cite: Aggarwal, C. C. (2017). *Outlier Analysis* (2nd ed.). Springer.

---

## 7. Dataset Splitting and Augmentation

**Order matters and is a hard constraint: split before augmenting, never the reverse.** This was an actual bug caught and fixed during the thesis's own diagram planning — get it right the first time.

### 7a. GroupKFold Split

Use **GroupKFold cross-validation, grouped by `Sample_ID`**. This must run *before* augmentation. Rationale: all six angle images of one physical sample — and every augmented variant later generated from them — must stay together in the same fold. If splitting happened after augmentation, or without grouping by sample, an augmented copy of a training-fold sample could leak into a validation/test fold, an artificially easy case that inflates apparent accuracy without the model having learned anything generalizable.

### 7b. Class-Weighted Geometric Augmentation

**Geometric only: rotations and flips.** No photometric augmentation (brightness/contrast/color-shift) — it would corrupt the HSV/LAB color features that are the actual measured signal. No synthetic interpolation methods like SMOTE — these generate non-physical feature vectors, a documented risk on an already-small dataset (Abdelhamid, M., & Desai, A. (2024). *Balancing the scales: A comprehensive study on tackling class imbalance in binary classification.* arXiv:2409.19751).

Apply an **uneven multiplier across classes**: a greater number of augmented copies for the minority Class 1 and Class 3 than for majority Class 2, partially correcting class imbalance using real rotated/flipped copies of genuine samples rather than synthetic ones.

**Open item — not finalized:** the exact multiplier per class has not been decided by the thesis team. Make the per-class multiplier a configurable parameter (e.g. a dict `{1: k1, 2: k2, 3: k3}`), pick a reasonable starting default that meaningfully oversamples the minority classes without producing excessive near-duplicate images (e.g. start by roughly balancing effective class counts post-augmentation), and clearly document that this is a tunable default pending confirmation, not a fixed decision.

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

Use **Optuna** (Akiba et al., 2019) to tune all four models' hyperparameters. **Optimize for Macro F1, not raw accuracy.** Macro F1 averages the F1 score across all three classes equally, so majority-class Class 2 cannot dominate the score the way plain accuracy would let it — this matters directly because minority-class performance (Class 1 precision, Class 3 recall — see Section 10) is the practically important part of this system's accuracy, not overall accuracy. Cite: Maia, W. F. et al. (2024). *Multi-level product category prediction through text classification.* arXiv:2403.01638 (used in the thesis for the Macro-F1-under-imbalance justification).

No specific hyperparameter search space or trial count has been fixed by the thesis team — choose reasonable, well-documented search spaces per model (e.g. `n_estimators`, `max_depth`/`num_leaves`, `learning_rate`, `min_child_samples`/`min_samples_leaf`, regularization terms) and a trial budget appropriate to the small dataset size (in the low hundreds to low thousands of rows post-augmentation) — this does not need enormous trial counts. Document whatever search space and trial count you pick; do not present it as a decision the thesis team already made.

### 8c. Algorithm Selection vs. Deployment Configuration — do not conflate these

This is the single most important structural distinction in the whole pipeline; keep them as two clearly separate stages in code, not one:

1. **Algorithm selection (this section):** train all four models on the **combined all-angle feature set** (all six images' features together, per sample). Whichever of the three ensembles (Random Forest, XGBoost, LightGBM) scores best on tuned Macro F1 is the **selected algorithm**. Logistic Regression is the baseline reference point only and is **never** eligible for deployment regardless of its score.
2. **Deployment configuration (Section 9):** the *selected algorithm* (not all four, just the winner) is then **retrained** on single-angle and angle-subset feature sets to determine what a live user actually has to submit.

**The combined all-angle model from step 1 is not what gets deployed.** It exists only to answer "which of the three ensemble algorithms is strongest," not "what does a live user submit." Do not skip straight to deploying the combined-all-angle winner — Section 9's retraining step is mandatory before anything is called "the deployed model."

---

## 9. Angle-Wise Accuracy Comparison — Determines the Deployment Input

Retrain **only the algorithm selected in Section 8c**, separately, on each of:
- Top-only
- Side-only (each individual side separately, and all four sides combined)
- Bottom-only
- Combined all-angle (the Section 8 baseline, kept here purely as the upper-bound comparison point)

Compare accuracy/Macro F1 across these configurations. Whichever single-angle configuration reaches acceptable accuracy becomes the actual deployed model's required input — a live buying-station user should only ever need to submit **one photograph**, never the six-angle set data collection required. If no single angle alone is good enough, fall back to the smallest angle combination that is (e.g. two sides, or top+one side) rather than defaulting back to all six.

This comparison is not a purely academic finding — its output directly determines what image-submission requirement gets built into the eventual Streamlit interface, so treat its result as a hard input to Section 11, not just a reported metric.

---

## 10. Explainability (TreeSHAP)

Run **TreeSHAP** (Lundberg & Lee, 2017 — the tree-optimized SHAP variant, exact and fast specifically for tree-based models) against **only the selected algorithm's combined all-angle version** from Section 8 (not the single-angle deployment version from Section 9, and not all three ensemble candidates separately — running it three times over was explicitly judged not worth the effort for a two-person team).

**Aggregate the resulting feature-importance scores by feature family** — texture (GLCM), color (HSV/LAB), edge (Canny) — rather than reporting per individual feature. This answers "how much did texture vs. color vs. edges matter overall to the model's decisions," which is a distinct question from Section 9's "which capture angle matters most."

Separately, at deployment time, the eventual Streamlit interface computes a **live, per-classification SHAP explanation** against the actual single-photo deployed model for one specific submitted image — that is a different computation from this aggregate research finding and happens outside this pipeline's batch scope, though this pipeline should expose whatever SHAP-computation function/API the deployment layer will call.

Every top SHAP feature identified should ideally be checked against a plausible physical/domain explanation (e.g., "edge density mattered most for the 6% boundary" is domain-plausible; a nonsensical top feature is worth investigating as a possible leakage or preprocessing artifact before trusting the result).

---

## 11. Evaluation

Report all of the following — do not settle for a single accuracy number:

- **Confusion matrix, overall accuracy, F1-score.**
- **Class-specific precision** — report Class 1 precision with particular emphasis: a false positive here (wet/lower-grade copra misclassified as Class 1/premium) is a direct financial loss to the buyer.
- **Class-specific recall** — report Class 3 recall with particular emphasis: a false negative here (non-merchantable copra misclassified as acceptable) is a health/liability risk (aflatoxin/mold).
- **Macro F1** (the same metric used for hyperparameter optimization — also report it here as a headline evaluation number, not only as the tuning objective).
- **Boundary-region performance reported separately from mid-range performance** — specifically, accuracy for samples near the 6.0%/6.1% boundary and near the 13.9%/14.0% boundary, since ground-truth measurement variance (the Brown-Duvel meter has documented reading-to-reading variance) matters most exactly at those two thresholds. Bucket evaluation samples into "near-boundary" (e.g., within some tolerance band you choose and document, such as ±0.5 percentage points of either threshold) vs. "mid-range" and report metrics for each bucket separately.
- **Benchmark against a human baseline:** experienced buying-station personnel classify a validation subset using the traditional manual (pasa) method; compare model accuracy against human accuracy on that same subset. This is a stronger practical-impact argument for the whole study than merely beating the Logistic Regression baseline, and the evaluation code should be built to accept a human-labeled comparison file/column as an input, not assume it's absent.
- **Feature-family ablation** (train on color-only, texture-only, edge-only, and combined feature sets; report the accuracy progression) — this is what justifies keeping all three feature families in the default pipeline (Section 5), particularly Canny edge density, which is the newest of the three additions. Not optional — build this as a standard evaluation step, not an afterthought.
- Consider running a **PCA or t-SNE visualization** on the extracted feature space (GLCM + HSV/LAB + Canny combined) before or alongside model training, purely as a sanity check on whether the three classes look separable at all in feature space. If they overlap heavily, that's a feature-engineering problem, not something more model tuning will fix — flag it rather than continuing to tune blindly.

---

## 12. Final Deployable Artifact

The artifact this whole pipeline produces is: **the algorithm selected in Section 8c, retrained on the single-angle (or minimal-angle) configuration determined in Section 9.** This is the only model that gets serialized and handed off for deployment — not the combined all-angle version, which exists only as an intermediate comparison point.

Serialize the model (and any accompanying preprocessing/feature-extraction config it depends on, e.g. the Otsu parameters, the specific angle/distance set used for GLCM, the exact feature ordering) in a way a separate Streamlit app can load and call directly — format not yet finalized by the thesis team (e.g., a standard `joblib`/pickle dump for scikit-learn-compatible models, or each library's native serialization for XGBoost/LightGBM, is reasonable; document whichever is chosen and package the feature-extraction code as an importable, reusable module rather than notebook-only code, since deployment needs to call the exact same preprocessing/feature-extraction logic used during training).

**Deployment context (for downstream awareness only, not this session's build target):** the artifact will eventually be loaded by a Streamlit web interface where a buying-station user submits one photo and receives a PCA class plus an optional on-request SHAP explanation. No price adjustment is ever computed by this system.

---

## 13. Explicit Non-Goals — Do Not Build These

- **No price/discount prediction or regression output.** Classification only, ever.
- **No end-to-end CNN or deep-learning image model.** The dataset (1,080–1,440 raw images before augmentation, several thousand after) is judged too small for reliable deep-learning fine-tuning without overfitting; the handcrafted-feature + tree-ensemble approach is the deliberate choice, more interpretable and appropriate at this data scale (Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). *Why do tree-based models still outperform deep learning on tabular data?* arXiv:2207.08815).
- **No meta-ensemble/stacking across Random Forest, XGBoost, and LightGBM.** They are compared, not combined.
- **No SMOTE or other synthetic interpolation for class balance.** Use class weighting + real geometric augmentation only.
- **No photometric augmentation** (brightness/contrast/color/hue shifts) at any point — it corrupts the HSV/LAB color signal the pipeline depends on.
- **No external "wet"/"dried" public dataset merged in** unless a documented, uncertainty-acknowledged label-mapping protocol is separately agreed with the thesis team first — default assumption is this study uses only its own field-collected data.
- **Do not use commercial grade names** ("Resecada," "Bodega," "Semi-Resecada," "Corriente," "Non-Merchantable" as a class *name* rather than a status) anywhere in code, labels, or generated output — use `Class 1`/`Class 2`/`Class 3` only, per Section 1's compliance note.

---

## 14. Objective Traceability Map

Every pipeline stage should be traceable back to one of the study's six specific objectives — useful for the model-development session to self-check coverage:

| Objective | What It Requires | Pipeline Section(s) |
|---|---|---|
| 1 | Extract texture/color/edge features from multi-angle images | §5 (Feature Extraction) |
| 2 | Train and compare RF/XGBoost/LightGBM vs. Logistic Regression baseline | §8 (Model Training, Tuning, Selection) |
| 3 | Evaluate which feature families contribute most via TreeSHAP | §10 (Explainability) |
| 4 | Determine which capture angle yields highest accuracy; establish live-deployment photo requirement | §9 (Angle-Wise Comparison) |
| 5 | Evaluate model accuracy vs. traditional pasa method by experienced personnel | §11 (Evaluation — human baseline) |
| 6 | Implement a functional prototype applying the trained model to a single submitted image | §12 (Final Artifact) — full Streamlit build is a separate, later effort |

---

## 15. Open Items the Thesis Team Has Not Yet Finalized

Flag these explicitly in any generated code/README rather than silently deciding them as if settled — implement configurable defaults, but say clearly that they are defaults pending confirmation:

1. **Outlier detection method and exact cutoff** (Z-score `|z|>3` vs. IQR `1.5×IQR`, or a different multiplier) — §6.
2. **Exact per-class augmentation multiplier** — §7b.
3. **Optuna search space and trial budget per model** — §8b.
4. **Exact image-file delivery format alongside the CSV export** (folder convention / zip / cloud folder) — §3.
5. **Model artifact serialization format** for handoff to Streamlit — §12.
6. **GLCM distance set** (this document suggests e.g. 1–3 px as a reasonable default; not fixed by the thesis team) — §5a.
7. **Boundary-region tolerance band** used to bucket "near-boundary" vs. "mid-range" evaluation samples — §11.

If the actual field dataset is not yet available when this session starts, build and validate the full pipeline against a small synthetic/placeholder dataset with the same schema (six angle images per sample, a moisture-reading column, the three-class derivation) so the code path is proven correct and ready to run the moment real data lands — do not block all development on waiting for the dataset.

---

## 16. Source Documents (for deeper context if needed)

This instruction file was derived from, and should stay consistent with, these files in the thesis project repository (not needed by the coding session unless a question arises that this document doesn't answer):
- `thesis-project-overview.md` — full project history and context
- `Methodology/objective.md` — the six specific objectives and research questions verbatim
- `Methodology/4.3-research-procedure.md`, `4.4-concept.md`, `4.5-analysis-and-design.md`, `4.8-software-development-tools.md` — formal methodology chapter sections
- `process.md` — an earlier, shorter technical walkthrough covering the same pipeline
- `models.md` — a plain-language explainer of what each technique does and why
