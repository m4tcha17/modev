"""Feature extraction: GLCM texture, HSV/LAB color, Canny edges. Spec §5.

All three families run on the masked image, in parallel, and concatenate into
one feature row per angle image (see combine.py). All three ship in the
default pipeline - ablation (evaluation/ablation.py) measures marginal
contribution, it does not gate inclusion.
"""
