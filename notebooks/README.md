# Notebooks

Exploration and one-off analysis only — call into `src/copra_grading`, never hold pipeline logic. Real logic lives in `src/` so it's testable and reusable by both notebooks and the eventual Streamlit deployment.

Planned notebooks (create as real `.ipynb` files when the implementing session starts; each stub below just marks the slot and its purpose):

- `01_eda.ipynb` — raw dataset shape, class balance, moisture distribution, sanity checks on the CSV/image export before any pipeline code runs.
- `02_masking_check.ipynb` — visual check of Otsu masks (`preprocessing/otsu.py`) across sample lighting conditions; where the fallback-method decision (§4) would get made if Otsu proves unreliable.
- `03_feature_sanity.ipynb` — spot-check extracted GLCM/HSV-LAB/Canny feature values against physical expectations (e.g. do drier samples show higher edge density).
- `04_ablation.ipynb` — run and visualize `evaluation/ablation.py` output (feature-family accuracy progression) and `evaluation/viz.py` (PCA/t-SNE separability check).
- `05_shap.ipynb` — run and visualize `explainability/treeshap.py` aggregate-by-family output; per-feature plausibility check per §10.
