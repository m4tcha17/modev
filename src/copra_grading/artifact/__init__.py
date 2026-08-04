"""Final deployable artifact serialization. Spec §12. Format: see ADR-005.

Bundles the angle-retrained deployment model (selection/deployment_config.py's
output) with its exact preprocessing/feature-extraction config - never the
combined all-angle model from selection/algorithm_selection.py.
"""
