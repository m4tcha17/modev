# Notebooks

Exploration and one-off analysis only — call into `src/copra_grading`, never hold pipeline logic. Real logic lives in `src/` so it's testable and reusable by both notebooks and the Streamlit app.

Planned notebooks (create as real `.ipynb` files when needed):

- `01_eda.ipynb` — dataset shape, class balance across A–F, photos per `batch_id`, `dataset.check_batches` output.
- `02_masking_check.ipynb` — visual check of Otsu masks on the grey background; where a fallback (adaptive / HSV / GrabCut) would be chosen if Otsu fails.
- `03_feature_sanity.ipynb` — spot-check GLCM / HSV-LAB / Canny values per class; how many rows the outlier rule drops.
- `04_shap.ipynb` — TreeSHAP by feature group for the selected model.
