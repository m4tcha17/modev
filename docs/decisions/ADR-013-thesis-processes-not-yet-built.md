# ADR-013: Thesis Processes Not Yet in the Pipeline

**Status:** proposed 2026-11-05. Each item below needs a thesis-team go/no-go before it is built.
**Spec reference:** thesis manuscript `../thesis/ANG-FLORENTINO-THESIS.md` (Chapter 4 and Appendix B, the SRS v0.11). Line numbers below are lines in that file.

## Context

The thesis manuscript was read in full on 2026-11-05 and checked against this repo. It commits to several processes that `process.md` does not list, and several that `process.md` lists but the code does not implement yet. This ADR records every remaining process in one place, so the build and the thesis can be reconciled. Where `process.md` and the thesis disagree, `process.md` stays the build spec until the thesis team updates it.

Already built and matching the thesis, so not listed below: background model + GrabCut masking (REQ-4.1-2), GLCM with entropy, HSV/LAB, Canny (REQ-4.1-3), outlier drop at |z| > 3 or 1.5×IQR (REQ-4.1-4), StratifiedGroupKFold by `batch_id` before augmentation (REQ-4.1-5, ADR-010), class-targeted geometric augmentation (thesis :723, ADR-002), class-weighted LR/RF/XGBoost/LightGBM with Optuna on Macro F1 and nested 5×3 CV (REQ-4.2-1 to 4.2-5), per-fold best params and outer-fold mean/SD (REQ-4.2-6).

## Remaining processes

### Not in `process.md` (thesis-only)

**1. Grey-paper lighting normalization.** REQ-4.1-6; thesis §1.6.3 (:260), §4.1.1 (:605), §4.2.1 (:635), §4.3.2.2 (:721).
- After masking, measure the paper's average color from the fitted background surface, then scale the photo so the paper matches one fixed reference grey. Then resize as now.
- The surface is already fitted in `preprocessing/background.py` (quadratic LAB fit on the border band). Normalization reuses it, so no new background estimate is needed.
- The Streamlit app must apply the same step (REQ-4.4-3). The reference grey goes into the artifact's saved config.
- This is not photometric augmentation (`CLAUDE.md`). It is one deterministic correction applied identically to every photo, at training and at inference, and it creates no extra copies.
- Open: per-channel gain or lightness-only; the reference grey value (median paper color across the export is the obvious default); whether GLCM and Canny take the normalized image too. The thesis only says "before color feature extraction" but resizes after normalization, which implies all three families see it.
- Effect: every feature changes, so outlier counts (ADR-011) and all CV scores must be rerun.

**2. Background-only lighting check.** REQ-4.1-8; thesis §4.10 (:982, :984), §4.2.1 (:635).
- Train a simple classifier on paper-only pixels (no copra) to predict the grade. Run it before and after normalization (item 1) and report both accuracies.
- Pass mark (:984): near chance after normalization. Above 25% is investigated as lighting leakage. A failed check means the model's scores cannot be trusted until the leak is fixed.
- The "before" half can be built now; the "after" half needs item 1.
- Open: paper features (for example per-photo mean and SD of L, a, b over the border band); classifier (Logistic Regression is the simplest); use the same grouped folds as the models.
- Note: the thesis chance level of ~17% assumes six grades. With F still missing, chance is 20%, which leaves little room under the 25% mark.

**3. Feature-family ablation.** Thesis §4.4.2 Table 2 (:755), §4.6 (:850), §4.10 (:984), §4.10.1 (:994, :996).
- Train on GLCM-only, color-only, edge-only and combined feature sets and compare Macro F1.
- Pass mark (:984): combined must score no lower than the best single family.
- Open: which models (selected model only is cheapest; all four is a 4× nested-CV cost); reuse the same outer folds so the numbers are comparable.
- Thesis mismatch: §4.6 (:850) says ablation results "shaped which features were retained". No feature was ever dropped on ablation results (ADR-012's pruning was correlation-based and was rejected). Either reword that sentence or make ablation feed a decision.

**4. Pass-mark report.** Thesis §4.10 (:984).
- Check the selected model against the team's fixed targets and write pass/fail with the actual values:
  - mean outer-fold Macro F1 ≥ 0.75 (stretch 0.80);
  - every grade's mean recall ≥ 0.60;
  - Grade A precision ≥ 0.85;
  - best ensemble beats Logistic Regression by more than one outer-fold SD of Macro F1.
- A missed target is reported with its value and the confused grades. The model is not tuned further to pass it. That matches the 2026-10-15 decision to accept Logistic Regression beating the ensembles; the last target currently fails and is reported as such.
- `evaluation.py` computes pooled metrics only. Per-grade recall and Grade A precision need per-fold values to report a "mean".

**5. Final retune and full-data refit.** REQ-4.2-7; thesis §4.5.3.
- After selection, re-tune the selected algorithm on the full dataset with a grouped five-fold split, then train it on all data with augmentation. That model is the one deployed.
- `process.md` Step 9 only says "save the selected model", but nothing produces a deployable model today. Deployment is blocked on this item.
- The refit model, its tuned params, its feature order and the preprocessing config (including the normalization reference from item 1) are saved through `artifact.py` (ADR-005).

### In `process.md` but not built yet

**6. TreeSHAP by feature family.** `process.md` Step 8; REQ-4.3-1, 4.3-2. Both functions in `explainability.py` raise `NotImplementedError`. Run it on the selected model only, summed into texture / color / edge.

**7. Artifact save/load.** `process.md` Step 9; ADR-005. `save_artifact` and `load_artifact` raise `NotImplementedError`. Needed by items 5 and 8.

**8. Streamlit app.** `process.md` Step 9; SRS §3.1, §4.4, §4.5, §5.2, §5.3. `app/streamlit_app.py` `main()` raises `NotImplementedError`. Beyond one photo in and one grade out, the SRS requires:
- the same masking and normalization as training (REQ-4.4-3);
- a file-signature check rather than an extension check, and a size cap (25 MB in the wireframe text);
- a clear error on model-load failure or a bad image, visually distinct from a grade, and never a default or cached grade;
- the SHAP panel collapsed by default and shown only on request (REQ-4.5-1, 4.5-2).

### Partly built

**9. Exclusion of clearly wrong masks.** REQ-4.1-7; thesis §4.3.2.2 (:719). A face image whose mask is clearly wrong on inspection is excluded unless retaken. `scripts/mask_photos.py` writes the preview used to inspect masks, but nothing removes a flagged photo. A small exclusion list (photo `id`s) read by `scripts/extract_features.py` would close it.

## Not pipeline work

These are thesis processes done by people, recorded here so nothing is lost: the six pooled Brown-Duvel readings checked against each grade's range; the owner's whole-dataset review before preprocessing; the owner validating the trained model's output; the usability walkthrough and reliability edge-case run on the Streamlit app (§4.10, :986–:988).

## Stale thesis text found during the read

Not processes to build. These should be fixed in the manuscript:
- §4.5.3 (:813) still describes per-angle models (top-only, side-only, bottom-only) and a six-angle capture. SRS v0.6 removed the angle comparison; capture is four cut faces, and Face ID is not exported.
- §4.5.3 (:807) still refers to "majority Class 2" and "Class 1 and Class 3" from the old three-class design.
- §4.5.4 figure captions (:823, :831) still describe a six-angle capture and a moisture-reading entry.
- Table 3 (:904) lists OpenCV for "Background masking (Otsu's thresholding)"; Otsu is not used (ADR-008).

## Suggested order

5 and 7 first, since deployment is blocked without them. Then 2 (before half) and 9, which are cheap. Then 1, followed by 2 (after half) and a full rerun. Then 3, 4 and 6 on the rerun results. Item 8 last, once the artifact exists.

## Owner

Thesis team (methodology decision per item); implementation once approved.
