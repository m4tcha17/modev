# Process: Copra Classification Model

How the model is built, from the dataset to a deployed classifier, step by step.

---

## 1. The Dataset

### File structure

```
copra-dataset.csv
photos/
    <UUID>.jpeg
    <UUID>.jpeg
    ...
```

`copra-dataset.csv` and the `photos/` folder sit in the same directory. Each photo is named after its unique `id`.

### Columns in `copra-dataset.csv`

| Column | Meaning |
| --- | --- |
| `id` | Unique ID of one photo (one row = one photo). |
| `batch_id` | ID of one physical copra sample. Each sample is photographed from 4 angles/sides, so all 4 photos of the same sample share the same `batch_id`. |
| `copra_class` | The label. One of `A`, `B`, `C`, `D`, `E`, `F`. |
| `path` | Relative file path of the photo, e.g. `photos/<UUID>.jpeg`. |

Example rows:

```
id,batch_id,copra_class,path
3f2a...,b01,A,photos/3f2a....jpeg
9c1e...,b01,A,photos/9c1e....jpeg
7d4b...,b01,A,photos/7d4b....jpeg
1a8f...,b01,A,photos/1a8f....jpeg
```

All photos were taken with the same phone, fixed distance (10 cm), same camera settings, on the same mid-grey background. This consistency is what makes the steps below work.

---

## 2. Pipeline Steps

### Step 1: Load the data

1. Read `copra-dataset.csv`.
2. Load each image from its `path`.
3. Keep `batch_id` with every row. It is needed for splitting in Step 4.

### Step 2: Preprocessing (background removal)

1. Convert the image to grayscale.
2. Apply **Otsu's thresholding** to get a binary mask that separates the copra from the grey background.
3. Apply the mask to the original image so only copra pixels remain.
4. Normalize lighting against the grey paper: measure the paper's average color from a surface fitted to the photo's border band, and scale the photo per RGB channel so the paper matches one fixed reference grey (ADR-013). This is one deterministic correction applied to every photo, at training and inference, not augmentation.
5. Resize to a fixed size so all images are on the same scale.

If Otsu turns out unreliable, fallbacks are adaptive thresholding, an HSV-based mask, or GrabCut. Only switch if Otsu actually fails. (It did: the default mask is a background model + GrabCut, ADR-008.)

A photo whose mask is clearly wrong on inspection is excluded (listed in `configs/mask_exclusions.csv`) unless it can be retaken.

### Step 3: Feature extraction

Run on each masked image. Produces one feature vector per photo, made of three feature groups:

**a. Texture — GLCM (Gray-Level Co-occurrence Matrix)**
- Build the GLCM from the grayscale masked image at angles 0°, 45°, 90°, 135° and a few pixel distances.
- Compute: contrast, homogeneity, energy, entropy.

**b. Color — HSV and LAB**
- Convert the masked image from RGB to HSV and to LAB.
- Compute per channel: mean, standard deviation (optionally histograms).
- Only use copra pixels (ignore masked background).

**c. Edges — Canny**
- Run Canny edge detection on the grayscale masked image.
- Compute edge density (edge pixels ÷ copra pixels) and contour stats.

Concatenate a + b + c into one feature vector. Output: a feature table with one row per photo, plus `batch_id` and `copra_class`.

### Step 4: Data preparation

**a. Outlier removal**
- Run on the feature values, not the images.
- A value is an outlier if |z| > 3 **or** it falls outside the IQR fence (`Q1 − 1.5×IQR` to `Q3 + 1.5×IQR`).
- Drop flagged values. Do not impute.
- Never relax the threshold. If too much data is lost, collect more data.

**b. Split with StratifiedGroupKFold (5 folds)**
- Group by the whole copra sample. Several `batch_id`s can come from the same whole sample, so `batch_id` alone is only safe while each batch is its own sample.
- Stratify by `copra_class` so every fold has a similar class mix.
- All 4 photos of one sample always stay in the same fold. Otherwise the model sees one side of a sample in training and another side in testing, which inflates accuracy.
- Split **before** augmentation.

**c. Augmentation (training folds only)**
- Rotations and flips only. No brightness, contrast, or color changes (they would corrupt the color features).
- No SMOTE.
- Smaller classes get more augmented copies than larger ones.
- Augmented images go through Steps 2–3 to get their features.

### Step 5: Train the models

Train four classifiers on the training folds:

| Model | Role | Scaling |
| --- | --- | --- |
| Logistic Regression | Baseline only, never deployed | Standardize features |
| Random Forest | Candidate | None |
| XGBoost | Candidate | None |
| LightGBM | Candidate | None |

- All four use **class weighting** to handle class imbalance.
- Tune hyperparameters with **Optuna**, optimizing **Macro F1** (not accuracy).
- Tune regularization: max depth and min samples per leaf (Random Forest), max depth and min child weight (XGBoost, LightGBM).

### Step 6: Select the best model

- Compare all four on Macro F1 across the 5 folds.
- The best of Random Forest / XGBoost / LightGBM is the selected model.
- Logistic Regression is only a reference point.
- After selection, re-tune the selected model on the full dataset (grouped 5-fold) and train it on all photos plus augmented copies. That model is the deployed one; its reported score stays the nested-CV estimate.

### Step 7: Evaluate

On the held-out folds, report per photo:
- Confusion matrix
- Accuracy
- Macro F1
- Per-class precision and recall

Then check the team's pass marks, reporting every miss with its value (never tuning to pass): mean Macro F1 ≥ 0.75 (stretch 0.80), every class's mean recall ≥ 0.60, class A precision ≥ 0.85, best ensemble beats Logistic Regression by more than one outer-fold SD.

Two further checks (ADR-013):
- **Background-only lighting check.** Predict the class from paper-only pixels, before and after normalization. Near chance means lighting does not reveal the class; above 25% is investigated as leakage.
- **Feature-family ablation.** Rerun the same nested CV on GLCM-only, color-only and edge-only features. The combined set must score no lower than the best single family.

### Step 8: Explainability

- Run **TreeSHAP** on the selected model.
- Sum feature importance by group: texture (GLCM), color (HSV/LAB), edge (Canny).
- Result: which kind of feature matters most for classification.

### Step 9: Deployment

- Save the selected model (the full-data refit from Step 6).
- Load it in a **Streamlit** web app.
- User uploads **one** photo, the app runs Steps 2–3 on it, and the model returns a class (A–F).
- On request, show a SHAP explanation for that photo, grouped by feature type.

---

## 3. Summary

| Step | What | Method |
| --- | --- | --- |
| 1 | Load data | CSV + `photos/` |
| 2 | Background removal | Background model + GrabCut (ADR-008), grey-paper lighting normalization, resize |
| 3 | Features | GLCM + HSV/LAB + Canny |
| 4a | Outliers | Drop if \|z\| > 3 or outside 1.5×IQR |
| 4b | Split | StratifiedGroupKFold (5), grouped by whole sample, stratified by `copra_class` |
| 4c | Augmentation | Rotate/flip, training folds only |
| 5 | Train | LR (baseline), RF, XGBoost, LightGBM; class weights; Optuna on Macro F1 |
| 6 | Select | Best ensemble on Macro F1, then full-data retune + refit |
| 7 | Evaluate | Confusion matrix, accuracy, Macro F1, per-class precision/recall; pass marks; lighting check; feature-family ablation |
| 8 | Explain | TreeSHAP by feature group |
| 9 | Deploy | Streamlit, one photo in, one class out |
