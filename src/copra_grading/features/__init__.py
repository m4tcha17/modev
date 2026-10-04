"""Feature extraction: GLCM texture, HSV/LAB color, Canny edges. process.md Step 3.

All three families run on the masked image and concatenate into one feature
row per photo (see combine.py). TreeSHAP (explainability.py) reports which
group matters most; it does not gate inclusion.
"""
