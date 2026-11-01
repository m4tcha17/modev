"""Streamlit deployment app. process.md Step 9.

User uploads ONE photo. The app runs Steps 2-3 on it (background-model + GrabCut mask, resize,
GLCM + HSV/LAB + Canny features, using the artifact's saved config), the
selected model returns a class A-F (shown with its moisture band from
dataset.CLASS_DESCRIPTIONS), and on request a SHAP explanation for
that photo is shown, grouped by feature type (texture / color / edge).

Run: uv run streamlit run app/streamlit_app.py
"""
import streamlit as st
from frontend import render_app
from pathlib import Path

ARTIFACT_PATH = Path("models/selected")


#PAGE CONFIGURATION
st.set_page_config(
    page_title="Copra Quality Classification",
    page_icon="🥥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def main() -> None:
    """Upload widget -> load_artifact -> preprocessing.pipeline.preprocess
    -> extract_all_features -> model.predict -> optional explain_single_prediction.
    """
    render_app()


if __name__ == "__main__":
    main()
