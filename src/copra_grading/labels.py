"""Class label derivation from moisture reading. Spec §1, §3.

Thresholds (PCA AO 02, this study's own class numbering):
  Class 1: <= 6.0% MC
  Class 2: > 6.0% and < 14.0% MC (spec states the practical range as 6.1-13.9,
           i.e. the band between the two rounded-to-0.1 boundary values; class
           membership itself is decided by the two boundaries, not a gap)
  Class 3: >= 14.0% MC

Never derive labels from a pre-populated CSV column - compute them here.
"""

VALID_CLASSES = ("1", "2", "3")

LOWER_THRESHOLD = 6.0
UPPER_THRESHOLD = 14.0


def derive_class_label(moisture_reading: float) -> str:
    """Map a moisture percentage to one of "1", "2", "3" per §1's thresholds."""
    if moisture_reading <= LOWER_THRESHOLD:
        return "1"
    if moisture_reading >= UPPER_THRESHOLD:
        return "3"
    return "2"
