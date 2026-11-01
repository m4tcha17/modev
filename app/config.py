from pathlib import Path


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "selected"

# Upload settings
ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png"]
MAX_FILE_SIZE = 25 * 1024 * 1024

# Classification settings
VALID_CLASSES = ["A", "B", "C", "D", "E", "F"]

# Application settings
APP_TITLE = "Copra Quality Classification"
APP_LAYOUT = "wide"