"""Lighting normalization. Thesis spec section 4 (after masking, before features).

The two capture lights dim slowly during a session, and grades were shot in
sequence (all of one grade, then the next), so brightness drift can line up
with grade and leak it into the color features. Each photo is rescaled so
its grey paper matches one fixed reference grey.

How the paper color is measured:
1. Downscale (same working width as background.py).
2. Fit a smooth quadratic surface to each RGB channel from the image border
   band (assumed paper), with the same robust refit masking uses. This
   models the lighting gradient across the frame.
3. Average that surface over the copra's own pixels: the paper color the
   copra would have been lit under, not just the paper at the frame edge.

Two modes (`preprocessing.lighting_mode`, ADR-012):
- "gain": each channel multiplied by reference / paper color under the
  copra. Corrects overall brightness and color cast, not the gradient.
- "flat_field": each pixel divided by the fitted paper surface at that
  point, times reference. Also removes the lighting gradient, so light
  set up at a different angle or distance is evened out too.
Both are multiplicative per pixel, so normalizing before or after masking
gives the same copra pixels.
"""

import cv2
import numpy as np

from copra_grading.preprocessing.background import (
    _WORK_WIDTH,
    border_band_mask,
    quadratic_design,
    robust_surface_fit,
)


def paper_surface(image: np.ndarray) -> np.ndarray:
    """Fitted paper color surface, (h, w, 3) float RGB, at working resolution."""
    h0, w0 = image.shape[:2]
    work_w = min(_WORK_WIDTH, w0)
    work_h = max(1, round(h0 * work_w / w0))
    small = cv2.resize(image, (work_w, work_h), interpolation=cv2.INTER_AREA).astype(np.float32)
    border_band = max(2, round(25 * work_w / _WORK_WIDTH))

    yy, xx = np.mgrid[0:work_h, 0:work_w].astype(np.float32)
    xx /= work_w
    yy /= work_h
    border = border_band_mask(work_h, work_w, border_band)
    design = quadratic_design(xx[border], yy[border])
    full_design = quadratic_design(xx.ravel(), yy.ravel())
    surface = np.empty_like(small)
    for c in range(3):
        coef, _ = robust_surface_fit(design, small[..., c][border])
        surface[..., c] = (full_design @ coef).reshape(work_h, work_w)
    return surface


def paper_color(image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Mean fitted paper RGB under the copra (`mask`), or over the whole
    frame if the mask is empty or not given."""
    surface = paper_surface(image)
    h, w = surface.shape[:2]
    if mask is not None and mask.any():
        small_mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        if small_mask.any():
            return surface[small_mask.astype(bool)].mean(axis=0)
    return surface.reshape(-1, 3).mean(axis=0)


def _keep_copra_nonzero(out: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is not None:
        out[mask & ~out.any(axis=-1)] = 1
    return out


def normalize_flat_field(
    image: np.ndarray, reference_rgb, mask: np.ndarray | None = None
) -> np.ndarray:
    """Divide each pixel by the fitted paper surface at that point and scale
    to `reference_rgb`: the paper becomes flat reference grey everywhere."""
    h0, w0 = image.shape[:2]
    surface = cv2.resize(paper_surface(image), (w0, h0), interpolation=cv2.INTER_LINEAR)
    gain = np.asarray(reference_rgb, np.float32) / np.maximum(surface, 1.0)
    out = np.clip(np.rint(image.astype(np.float32) * gain), 0, 255).astype(np.uint8)
    return _keep_copra_nonzero(out, mask)


def normalize(image: np.ndarray, mask: np.ndarray, reference_rgb, mode: str = "gain") -> np.ndarray:
    """Apply the configured lighting normalization mode."""
    if mode == "gain":
        return normalize_lighting(image, paper_color(image, mask), reference_rgb, mask)
    if mode == "flat_field":
        return normalize_flat_field(image, reference_rgb, mask)
    raise ValueError(f"unknown lighting_mode {mode!r}")


def normalize_lighting(
    image: np.ndarray,
    paper_rgb: np.ndarray,
    reference_rgb,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Scale each RGB channel so `paper_rgb` maps to `reference_rgb`.

    Inside `mask`, pixels are kept at least 1 in some channel so the
    nonzero-means-copra convention (otsu.apply_mask) survives a gain < 1.
    """
    gain = np.asarray(reference_rgb, np.float32) / np.maximum(np.asarray(paper_rgb, np.float32), 1.0)
    out = np.clip(np.rint(image.astype(np.float32) * gain), 0, 255).astype(np.uint8)
    return _keep_copra_nonzero(out, mask)
