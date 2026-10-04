# ADR-005: Model Artifact Serialization Format

**Status:** proposed-default, pending thesis-team confirmation
**Spec reference:** `process.md` Step 9

## Default

`joblib` dump for the winning model if it's scikit-learn-compatible (Random Forest, Logistic Regression baseline artifacts if ever needed); native serialization (`Booster.save_model` / LightGBM's own `.txt`/`.pkl` save) for XGBoost or LightGBM if either wins selection (Step 6). Bundled alongside a config file capturing Otsu parameters, the resize target, the GLCM angle/distance set (ADR-006), and exact feature ordering — everything `artifact.py` needs to reproduce identical preprocessing/feature-extraction at inference time.

## Rationale

Format should follow whichever ensemble algorithm actually wins selection — each library's native serialization is the most robust option for that library specifically, more so than forcing everything through a single generic format like pickle. The pipeline can't know the winner at scaffold time, so `artifact.py` should branch on model type.

## What would change it

If the Streamlit app (`app/streamlit_app.py`) or its hosting environment needs a single file, bundle into one archive.

## Owner

Thesis team.
