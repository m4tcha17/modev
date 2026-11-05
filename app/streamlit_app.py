"""Streamlit deployment app. process.md Step 9.

User uploads ONE photo. The app runs Steps 2-3 on it (background-model + GrabCut mask,
grey-paper normalization, resize, GLCM + HSV/LAB + Canny features, using the
artifact's saved config), the selected model returns a class A-F (shown with its
moisture band from dataset.CLASS_DESCRIPTIONS), and on request a SHAP explanation
for that photo is shown, grouped by feature type (texture / color / edge).

Errors (bad file, no copra, missing model) are shown as error boxes, never as a
grade (SRS section 5.2).

Run: uv run streamlit run app/streamlit_app.py
"""

from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from copra_grading.artifact import load_artifact
from copra_grading.inference import UploadError, classify, decode_upload, explain

ARTIFACT_PATH = Path("models/selected")
# Badge colors run from best (A) to worst (E); F is penalized for over-drying.
GRADE_COLORS = {"A": "#1b7f3b", "B": "#5a9a2f", "C": "#b58900", "D": "#cb6a15", "E": "#b3261e", "F": "#6a3d9a"}


@st.cache_resource
def _artifact():
    return load_artifact(ARTIFACT_PATH)


def _outlined(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = image.copy()
    cv2.drawContours(out, contours, -1, (255, 0, 0), max(2, image.shape[1] // 300))
    return out


def main() -> None:
    """Upload widget -> load_artifact -> preprocessing.pipeline.preprocess
    -> extract_all_features -> model.predict -> optional explain_single_prediction.
    """
    st.set_page_config(page_title="Copra Moisture Grade", layout="centered")
    st.title("Copra moisture grade")

    try:
        artifact = _artifact()
    except Exception as err:  # any load failure blocks classification
        st.error(f"The model could not be loaded from `{ARTIFACT_PATH}`, so no photo can be graded. ({err})")
        st.stop()
    st.caption(f"Model loaded: {artifact.model.name}")

    st.subheader("1. Upload")
    upload = st.file_uploader("One photo of a cube's cut meat face (JPEG or PNG, up to 25 MB)",
                              type=["jpg", "jpeg", "png"])
    if upload is None:
        st.session_state.pop("result", None)
    elif st.button("Classify copra image", type="primary"):
        try:
            image = decode_upload(upload.getvalue())
            with st.spinner("Masking, normalizing lighting, extracting features, classifying..."):
                st.session_state["result"] = (upload.file_id, image, classify(artifact, image))
        except UploadError as err:
            st.session_state.pop("result", None)
            st.error(str(err))

    result = st.session_state.get("result")
    if result is None or upload is None or result[0] != upload.file_id:
        return
    _, image, classification = result

    st.subheader("2. Processing")
    st.image(_outlined(image, classification.mask), caption="Copra mask (red outline)", width="stretch")

    st.subheader("3. Result")
    color = GRADE_COLORS.get(classification.grade, "#444")
    st.markdown(
        f"<div style='background:{color};color:#fff;padding:16px;border-radius:8px'>"
        f"<div style='font-size:14px'>Moisture grade</div>"
        f"<div style='font-size:44px;font-weight:700;line-height:1.1'>{classification.grade}</div>"
        f"<div>{classification.description}</div></div>",
        unsafe_allow_html=True,
    )

    if st.toggle("Show SHAP explanation", value=False):
        shares = explain(artifact, classification.features)
        names = {"texture": "Texture (GLCM)", "color": "Color (HSV/LAB)", "edge": "Edge (Canny)"}
        for family, share in sorted(shares.items(), key=lambda kv: -kv[1]):
            st.write(f"{names[family]}: {share:.0%}")
            st.progress(min(max(share, 0.0), 1.0))
        st.info("These bars show each feature family's share of the TreeSHAP contribution "
                "to this grade. They are relative importance, not model accuracy.")


if __name__ == "__main__":
    main()
