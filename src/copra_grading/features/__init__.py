"""Feature extraction: GLCM texture, HSV/LAB color, Canny edges. process.md Step 3.

All three families run on the masked image and concatenate into one feature
row per photo (see combine.py). TreeSHAP (explainability.py) reports which
group matters most; it does not gate inclusion.
"""


class EmptyRegionError(ValueError):
    """No usable copra pixels to measure (empty mask, or a region too small
    for a GLCM distance). Raised per photo so feature-table building can skip
    and report that photo - it is dropped, never imputed. Subclasses
    ValueError; other ValueErrors (bad dtype, bad config) are real bugs and
    must still propagate.
    """
