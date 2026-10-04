"""Streamlit deployment app. process.md Step 9.

User uploads ONE photo. The app runs Steps 2-3 on it (Otsu mask, resize,
GLCM + HSV/LAB + Canny features, using the artifact's saved config), the
selected model returns a class A-F (shown with its moisture band from
dataset.CLASS_DESCRIPTIONS), and on request a SHAP explanation for
that photo is shown, grouped by feature type (texture / color / edge).

Run: uv run streamlit run app/streamlit_app.py
"""

from pathlib import Path

ARTIFACT_PATH = Path("models/selected")


def main() -> None:
    """Upload widget -> load_artifact -> mask_image/apply_mask/resize_masked
    -> extract_all_features -> model.predict -> optional explain_single_prediction.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
