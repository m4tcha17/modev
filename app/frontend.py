# 1. Imports and configuration
from __future__ import annotations

import streamlit as st
from dataclasses import dataclass
from config import (
    APP_TITLE,
    ALLOWED_FORMATS,
    ALLOWED_EXTS,
    MAX_FILE_SIZE
)
import html
import time
from io import BytesIO
from PIL import Image, UnidentifiedImageError

# MOCK DATA
@dataclass(frozen=True)
class ClassificationResult:
    label: str
    grade: str
    status_label: str


MOCK_RESULT = ClassificationResult(
    label="Class A",
    grade="Low Moisture / High Quality",
    status_label="ACCEPT — STANDARD BUYING RATE",
)

SHAP_FAMILIES = (
    ("Texture", 42, "#2563eb"),
    ("Edge", 35, "#e74c3c"),
    ("Color", 23, "#f59e0b"),
)

MAX_FILE_SIZE = 25 * 1024 * 1024

# Custom CSS and styling
GLOBAL_CSS = """
<style>
@import url("https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap");

:root {
    --bg: #f4f4f0;
    --surface: #ffffff;
    --border: #d4d4cc;
    --ink: #1a1a18;
    --muted: #6b6b63;
    --accent: #0a5c3c;
    --accent-light: #e6f4ee;
    --sans: "DM Sans", sans-serif;
    --mono: "JetBrains Mono", monospace;
}

* { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: var(--sans);
    color: var(--ink);
}
html { font-size: 16px; }
body, .stApp, [data-testid="stAppViewContainer"] { background: var(--bg); }
[data-testid="stAppViewContainer"] > .main { background: var(--bg); }

/* Remove Streamlit chrome so the page reads as a purpose-built dashboard. */
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
#MainMenu, footer { display: none !important; }
[data-testid="stSidebar"] { display: none; }
.block-container {
    width: 100%;
    max-width: 860px;
    padding: 104px 24px 60px !important;
}

.app-header {
    position: fixed;
    z-index: 999;
    top: 0;
    left: 0;
    width: 100%;
    min-height: 72px;
    padding: 14px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 24px;
    background: var(--ink);
    border-bottom: 1px solid #2a2a28;
}
.header-title {
    color: #ffffff;
    font-size: 17px;
    line-height: 1.25;
    font-weight: 600;
    letter-spacing: -0.01em;
}
.header-subtitle {
    margin-top: 2px;
    color: #9ca3af;
    font-family: var(--mono);
    font-size: 12px;
}
.system-status {
    flex-shrink: 0;
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 7px 12px;
    border-radius: 2px;
    background: var(--accent-light);
    color: var(--accent);
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.04em;
}
.status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--accent);
}

.section-label {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 0 0 16px;
}
.section-number {
    padding: 2px 6px;
    border: 1px solid var(--border);
    border-radius: 1px;
    color: var(--muted);
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.12em;
}
.section-title {
    color: var(--ink);
    font-size: 15px;
    font-weight: 600;
    letter-spacing: -0.01em;
}
.section-label.small { margin-bottom: 20px; }
.section-label.small .section-number { font-size: 9px; }
.section-label.small .section-title { font-size: 13px; }

/* Native file uploader, restyled to match the designed drop zone. */
/* UPLOAD ZONE */
.st-key-upload_zone [data-testid="stFileUploader"] {
    margin-bottom: 0 !important;
}

.st-key-upload_zone [data-testid="stFileUploaderDropzone"] {
    height: 264px !important;
    min-height: 264px !important;
    padding: 30px 24px !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    border: 2px dashed var(--border) !important;
    border-radius: 4px !important;
    background: #ffffff !important;
    cursor: pointer;
}

.st-key-upload_zone [data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--accent) !important;
    background: var(--accent-light) !important;
}

/* Icon */
.st-key-upload_zone [data-testid="stFileUploaderDropzone"] svg {
    color: #b8b8b3 !important;
    width: 32px !important;
    height: 32px !important;
}

/* Instructions wrapper: stack icon + custom text, centered */
.st-key-upload_zone [data-testid="stFileUploaderDropzoneInstructions"] {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
    gap: 14px !important;
}

/* Hide Streamlit's default text ("Drag and drop file here", "Limit 200MB…") */
.st-key-upload_zone [data-testid="stFileUploaderDropzoneInstructions"] > div {
    display: none !important;
}

/* Inject the custom text */
.st-key-upload_zone [data-testid="stFileUploaderDropzoneInstructions"]::before {
    content: "Drag and drop copra image here or click to browse";
    order: 2;
    margin-top: 18px;
    color: var(--ink);
    font-family: var(--sans);
    font-size: 19px;
    font-weight: 500;
}
.st-key-upload_zone [data-testid="stFileUploaderDropzoneInstructions"]::after {
    content: "Accepts JPG, JPEG, PNG — max 25 MB";
    order: 3;
    color: var(--muted);
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.04em;
}

/* Hide the Browse button (the whole drop zone is still clickable) */
.st-key-upload_zone [data-testid="stFileUploaderDropzone"] button {
    display: none !important;
}

.st-key-image_preview {
    overflow: hidden;
    margin: 8px 0 12px;
    border: 1px solid var(--border);
    border-radius: 3px;
    background: var(--surface);
}
.st-key-image_preview [data-testid="stImage"] img {
    max-height: 360px;
    object-fit: contain;
    background: #ecece7;
}
.file-meta {
    padding: 14px 18px 16px;
    border-top: 1px solid var(--border);
}
.eyebrow {
    margin-bottom: 5px;
    color: var(--muted);
    font-family: var(--mono);
    font-size: 9px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}
.file-name {
    overflow-wrap: anywhere;
    color: var(--ink);
    font-size: 16px;
    font-weight: 600;
}
.file-size {
    margin: 3px 0;
    margin-bottom: 10px;
    color: var(--muted);
    font-family: var(--mono);
    font-size: 11px;
}

/* Button treatments */
.st-key-classify_button button {
    width: 100%;
    min-height: 50px;
    margin-top: 4px;
    border: 0;
    border-radius: 2px;
    background: var(--ink);
    color: #ffffff;
    font-family: var(--sans);
    font-size: 16px;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.st-key-classify_button button:hover {
    border: 0;
    background: #30302d;
    color: #ffffff;
}
.st-key-classify_button button:disabled {
    border: 0;
    background: var(--border);
    color: #8c8c84;
}
.st-key-shap_button button {
    min-height: 45px;
    margin: 10px 0;
    padding: 0 24px;
    border: 1px solid var(--ink);
    border-radius: 2px;
    background: transparent;
    color: var(--ink);
    font-family: var(--sans);
    font-size: 15px;
    font-weight: 600;
}
.st-key-shap_button button:hover {
    border-color: var(--ink);
    background: var(--ink);
    color: #ffffff;
}
.st-key-collapse_button button {
    min-height: 39px;
    padding: 0 18px;
    border: 1px solid var(--border);
    border-radius: 2px;
    background: transparent;
    color: var(--muted);
    font-family: var(--sans);
    font-size: 16px;
}
.st-key-collapse_button button:hover {
    border-color: var(--ink);
    color: var(--ink);
}

.progress-track {
    height: 3px;
    margin-top: 6px;
    overflow: hidden;
    border-radius: 1px;
    background: var(--border);
}
.progress-fill {
    width: 0;
    height: 100%;
    border-radius: 1px;
    background: var(--accent);
    animation: classify-progress 2.2s ease-in-out forwards;
}
@keyframes classify-progress { from { width: 0; } to { width: 100%; } }

.results-wrap { margin-top: 40px; }
.metric-card {
    padding: 20px 24px 18px;
    border: 1px solid var(--border);
    border-radius: 2px;
    background: var(--surface);
}
.metric-label {
    margin-bottom: 8px;
    color: var(--muted);
    font-family: var(--mono);
    font-size: 9px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}
.metric-value {
    margin-bottom: 6px;
    color: var(--ink);
    font-size: 26px;
    line-height: 1.1;
    font-weight: 700;
    letter-spacing: -0.01em;
}
.metric-sub {
    color: var(--muted);
    font-size: 16px;
}
.accept-banner {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 16px 0 32px;
    padding: 12px 20px;
    border-radius: 2px;
    background: var(--accent);
    color: #ffffff;
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 0.06em;
}
.accept-dot {
    width: 8px;
    height: 8px;
    flex-shrink: 0;
    border-radius: 50%;
    background: #6ee7b7;
}

.st-key-shap_panel {
    margin-top: 1px;
    padding: 28px 28px 24px;
    border: 1px solid var(--border);
    border-radius: 2px;
    background: var(--surface);
    animation: panel-in .2s ease;
}
@keyframes panel-in {
    from { opacity: 0; transform: translateY(-6px); }
    to { opacity: 1; transform: translateY(0); }
}
.shap-row { margin-bottom: 20px; }
.shap-row-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 7px;
}
.shap-name {
    color: var(--ink);
    font-size: 15px;
    font-weight: 600;
}
.shap-percent {
    font-family: var(--mono);
    font-size: 15px;
    font-weight: 600;
}
.shap-track {
    height: 22px;
    overflow: hidden;
    border-radius: 2px;
    background: #efefec;
}
.shap-fill {
    height: 100%;
    border-radius: 2px;
    animation: bar-in .7s cubic-bezier(.22, 1, .36, 1);
    transform-origin: left;
}

.st-key-clear_button_wrap { padding: 0 18px 16px; }
.st-key-clear_button_wrap button {
    min-height: 36px;
    padding: 0 16px;
    border: 1px solid var(--border);
    border-radius: 2px;
    background: transparent;
    color: var(--muted);
    font-family: var(--sans);
    font-size: 14px;
}
.st-key-clear_button_wrap button:hover {
    border-color: #c0392b;
    color: #c0392b;
}

/* The preview already shows name and size, so hide the default file row */
.st-key-upload_zone [data-testid="stFileUploaderFile"],
.st-key-upload_zone [data-testid="stFileUploaderFileList"] {
    display: none !important;
}

@keyframes bar-in { from { transform: scaleX(0); } to { transform: scaleX(1); } }
.shap-note {
    margin: 16px 0 20px;
    padding: 12px 14px;
    border: 1px solid var(--border);
    border-left: 3px solid var(--muted);
    border-radius: 2px;
    background: #fafaf8;
    color: var(--muted);
    font-size: 12px;
    line-height: 1.6;
}
.shap-note strong { color: var(--ink); font-weight: 600; }
.error-bar {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 0 0 12px;
    padding: 12px 16px;
    border: 1px solid #e6b8b2;
    border-left: 4px solid #c0392b;
    border-radius: 2px;
    background: #fbefed;
    color: #8e2e23;
    font-size: 14px;
    font-weight: 500;
    line-height: 1.4;
}
.error-icon {
    flex-shrink: 0;
    width: 20px;
    height: 20px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    background: #c0392b;
    color: #ffffff;
    font-family: var(--mono);
    font-size: 12px;
    font-weight: 700;
}

@media (max-width: 680px) {
    .block-container { padding: 126px 16px 44px !important; }
    .app-header {
        min-height: 102px;
        padding: 13px 16px;
        align-items: flex-start;
        flex-direction: column;
        gap: 9px;
    }
    .header-title { font-size: 15px; }
    .header-subtitle { font-size: 10px; }
    .system-status { padding: 5px 9px; font-size: 9px; }
    .st-key-upload_zone [data-testid="stFileUploaderDropzone"] {
        min-height: 150px;
        padding: 28px 14px;
    }
    .metric-card { padding: 18px; }
    .st-key-shap_panel { padding: 22px 18px 20px; }
    .accept-banner { font-size: 12px; line-height: 1.45; }
}
</style>
"""


# 4. Helper functions
def section_label(number: str, title: str, *, small: bool = False) -> None:
    size_class = " small" if small else ""
    st.markdown(
        f"""
        <div class="section-label{size_class}">
            <span class="section-number">{html.escape(number)}</span>
            <span class="section-title">{html.escape(title)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def initialize_state() -> None:
    defaults = {
        "upload_signature": None,
        "phase": "idle",
        "result": None,
        "shap_open": False,
        "stored_file": None,     # {"name", "size", "data"} once accepted
        "uploader_nonce": 0,
        "upload_error": None,

    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_for_new_upload(signature: tuple[str, int] | None) -> None:
    if st.session_state.upload_signature != signature:
        st.session_state.upload_signature = signature
        st.session_state.phase = "idle"
        st.session_state.result = None
        st.session_state.shap_open = False


def format_file_size(size: int) -> str:
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    return f"{size / 1024:.1f} KB"

def clear_upload() -> None:
    # A new widget key makes Streamlit discard the uploaded file.
    # reset_for_new_upload(None) then clears phase/result/SHAP on the rerun.
    st.session_state.stored_file = None
    st.session_state.upload_error = None
    st.session_state.phase = "idle"
    st.session_state.result = None
    st.session_state.shap_open = False
    st.session_state.uploader_nonce += 1

def render_header() -> None:
    st.markdown(
        """
        <div class="app-header">
            <div>
                <div class="header-title">Copra Quality Classification &amp; Interpretability System</div>
                <div class="header-subtitle">Streamlit Thesis Prototype Dashboard</div>
            </div>
            <div class="system-status">
                <span class="status-dot"></span>
                <span>System Ready · Model Loaded</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_preview(file: dict) -> None:
    safe_name = html.escape(file["name"])
    with st.container(key="image_preview"):
        st.image(file["data"], use_container_width=True)
        st.markdown(
            f"""
            <div class="file-meta">
                <div class="eyebrow">Image Loaded</div>
                <div class="file-name">{safe_name}</div>
                <div class="file-size">{format_file_size(file["size"])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.container(key="clear_button_wrap"):
            st.button("Remove image", key="clear_image", on_click=clear_upload)


def render_result(result: ClassificationResult) -> None:
    st.markdown('<div class="results-wrap"></div>', unsafe_allow_html=True)
    section_label("02", "Primary Classification Output")
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Predicted Class</div>
            <div class="metric-value">{html.escape(result.label)}</div>
            <div class="metric-sub">{html.escape(result.grade)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_shap_panel() -> None:
    bars = "".join(
        f"""
        <div class="shap-row">
            <div class="shap-row-head">
                <span class="shap-name">{html.escape(family)}</span>
                <span class="shap-percent" style="color:{color}">{contribution}%</span>
            </div>
            <div class="shap-track">
                <div class="shap-fill" style="width:{contribution}%;background:{color}"></div>
            </div>
        </div>
        """
        for family, contribution, color in SHAP_FAMILIES
    )
    with st.container(key="shap_panel"):
        section_label("03", "TreeSHAP Feature-Family Contributions", small=True)
        st.html(
            f"""
            {bars}
            <div class="shap-note">
                <strong>Note:</strong> These percentages represent the relative contribution of each
                feature family to the model's classification decision as attributed by TreeSHAP.
                They indicate which groups of features (texture patterns, edge characteristics, or
                color properties) had the greatest influence on the predicted class — they do
                <em>not</em> reflect or validate the accuracy of the grade assignment itself.
            </div>
            """,
        )
        with st.container(key="collapse_button"):
            if st.button("Collapse Explanation", key="collapse_explanation"):
                st.session_state.shap_open = False
                st.rerun()


# 5. Mock application behavior
def classify_image() -> ClassificationResult:
    """Simulate local model inference without inventing an external service."""
    time.sleep(2.2)
    return MOCK_RESULT

def validate_upload(uploaded_file) -> str | None:
    name = uploaded_file.name
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in ALLOWED_EXTS:
        return "Wrong file type. Only JPG, JPEG or PNG images are accepted."
    if uploaded_file.size > MAX_FILE_SIZE:
        return f"File too large ({format_file_size(uploaded_file.size)}). The maximum size is 25 MB."
    try:
        data = uploaded_file.getvalue()
    except Exception:
        return "The file could not be read. Please try uploading it again."
    if not data:
        return "The file is empty and could not be read."
    try:
        with Image.open(BytesIO(data)) as img:
            img.verify()
            fmt = img.format
    except (UnidentifiedImageError, OSError, SyntaxError):
        return "The file could not be read as an image. It may be corrupted."
    if fmt not in ALLOWED_FORMATS:
        return "Wrong file type. Only JPG, JPEG or PNG images are accepted."
    return None


def render_error_bar(message: str) -> None:
    st.markdown(
        f"""
        <div class="error-bar" role="alert">
            <span class="error-icon">!</span>
            <span>{html.escape(message)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# 6–9. Session state, UI components, main logic, and entry point
def main() -> None:
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    initialize_state()
    render_header()

    section_label("01", "Image Upload")

    # Error bar sits at the top of the upload zone
    if st.session_state.upload_error:
        render_error_bar(st.session_state.upload_error)

    stored = st.session_state.stored_file

    if stored is None:
        with st.container(key="upload_zone"):
            uploaded = st.file_uploader(
                "Upload copra image",
                accept_multiple_files=False,
                key=f"copra_uploader_{st.session_state.uploader_nonce}",
                label_visibility="collapsed",
            )

        if uploaded is not None:
            error = validate_upload(uploaded)
            if error is None:
                st.session_state.stored_file = {
                    "name": uploaded.name,
                    "size": uploaded.size,
                    "data": uploaded.getvalue(),
                }
                st.session_state.upload_error = None
            else:
                st.session_state.upload_error = error
            st.session_state.uploader_nonce += 1
            st.rerun()
    else:
        render_preview(stored)

    file_is_valid = stored is not None
    button_slot = st.empty()
    with button_slot.container():
        with st.container(key="classify_button"):
            classify_clicked = st.button(
                "Classify Copra Image",
                disabled=not file_is_valid,
                key="classify",
            )

    if classify_clicked and file_is_valid:
        st.session_state.phase = "loading"
        st.session_state.shap_open = False
        button_slot.empty()
        with button_slot.container():
            with st.container(key="classify_button_container"):
                st.button("Loading...", disabled=True, key="classify_loading")
            st.markdown(
                '<div class="progress-track"><div class="progress-fill"></div></div>',
                unsafe_allow_html=True,
            )
        st.session_state.result = classify_image()
        st.session_state.phase = "done"
        st.rerun()

    if st.session_state.phase == "done" and st.session_state.result is not None:
        render_result(st.session_state.result)
        if not st.session_state.shap_open:
            with st.container(key="shap_button"):
                if st.button("Request SHAP Explanation", key="request_shap"):
                    st.session_state.shap_open = True
                    st.rerun()
        else:
            render_shap_panel()

def render_app():
    main()