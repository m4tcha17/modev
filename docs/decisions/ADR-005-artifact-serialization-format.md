# ADR-005: Model Artifact Serialization Format

**Status:** proposed-default, pending thesis-team confirmation
**Spec reference:** §12, §15 item 5

## Default

`joblib` dump for the winning model if it's scikit-learn-compatible (Random Forest, Logistic Regression baseline artifacts if ever needed); native serialization (`Booster.save_model` / LightGBM's own `.txt`/`.pkl` save) for XGBoost or LightGBM if either wins selection (§8c). Bundled alongside a config file capturing Otsu parameters, the GLCM angle/distance set (ADR-006), and exact feature ordering — everything `artifact.py` needs to reproduce identical preprocessing/feature-extraction at inference time.

## Rationale

Format should follow whichever ensemble algorithm actually wins selection — each library's native serialization is the most robust option for that library specifically, more so than forcing everything through a single generic format like pickle. The pipeline can't know the winner at scaffold time, so `artifact.py` should branch on model type.

## What would change it

If the Streamlit deployment layer has its own loading constraints (e.g. requiring a single file, or a specific format for its hosting environment), confirm with that project before finalizing.

## Owner

Thesis team, coordinated with whoever builds the Streamlit deployment layer.
