"""Background-model masking + GrabCut refinement. process.md Step 2 fallback.

Otsu fails on the real photos: copra has a white meat side (L ~200) and a
dark brown skin side (L ~30), and the mid-grey background (L ~100-170,
drifting across the frame with the lighting) sits between them. A single
global threshold therefore lumps the background in with the meat.

This method instead models the background directly:

1. Work on a downscaled copy (speed; GrabCut is the slow part).
2. Fit a smooth quadratic surface to each LAB channel using only the image
   border band, which is assumed to be background. This captures the
   lighting gradient. One robust refit drops border outliers (specks, a
   copra corner touching the edge).
3. Score every pixel by its distance from that surface, in units of the
   border's own robust spread. Pixels above `threshold_k` are copra seeds.
4. Seed GrabCut: eroded seeds = sure copra, dilated-seed exterior + border
   band = sure background, the rest = probable. GrabCut refines the edge.
5. Keep the largest connected piece and fill its holes, then upscale the
   mask back to the input resolution.

Assumes one copra piece per photo, not touching the frame edge, on the
uniform grey background described in process.md.
"""

import cv2
import numpy as np

_WORK_WIDTH = 1020  # downscaled width the mask is computed at


def _quadratic_design(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    return np.stack([np.ones_like(xs), xs, ys, xs * xs, xs * ys, ys * ys], axis=1)


def _largest_component(binary: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8))
    if n <= 1:
        return np.zeros_like(binary, dtype=np.uint8)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8)


def _fill_holes(binary: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(binary)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return filled


def border_band_mask(shape: tuple[int, int], border_band: int) -> np.ndarray:
    """Boolean (H, W) mask of the frame's outer band, assumed to be paper."""
    border = np.zeros(shape, dtype=bool)
    border[:border_band] = border[-border_band:] = True
    border[:, :border_band] = border[:, -border_band:] = True
    return border


def fit_border_surfaces(image: np.ndarray, border_band: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit a quadratic surface per channel to the border band, with one
    robust refit that drops border outliers.

    Returns (surfaces, sigmas): surfaces is (H, W, C) float32, sigmas the
    per-channel robust spread of the border around its surface (at least 1).
    """
    h, w = image.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx /= w
    yy /= h

    border = border_band_mask((h, w), border_band)
    design = _quadratic_design(xx[border], yy[border])
    full_design = _quadratic_design(xx.ravel(), yy.ravel())

    channels = image.shape[2]
    surfaces = np.empty((h, w, channels), dtype=np.float32)
    sigmas = np.empty(channels, dtype=np.float32)
    for c in range(channels):
        values = image[..., c][border].astype(np.float32)
        coef = np.linalg.lstsq(design, values, rcond=None)[0]
        resid = values - design @ coef
        mad = 1.4826 * np.median(np.abs(resid))
        keep = np.abs(resid) < 3 * mad + 1e-6
        coef = np.linalg.lstsq(design[keep], values[keep], rcond=None)[0]
        sigmas[c] = max(1.4826 * float(np.median(np.abs(values[keep] - design[keep] @ coef))), 1.0)
        surfaces[..., c] = (full_design @ coef).reshape(h, w)
    return surfaces, sigmas


def background_distance(lab: np.ndarray, border_band: int) -> np.ndarray:
    """Per-pixel distance from the fitted background surface, in robust
    standard deviations of the border band (combined over L, a, b).
    """
    surfaces, sigmas = fit_border_surfaces(lab, border_band)
    return np.sqrt((((lab - surfaces) / sigmas) ** 2).sum(axis=-1))


def working_copy(image: np.ndarray) -> tuple[np.ndarray, int]:
    """Downscaled copy the background is modelled on, and its border band width."""
    h0, w0 = image.shape[:2]
    work_w = min(_WORK_WIDTH, w0)
    work_h = max(1, round(h0 * work_w / w0))
    small = cv2.resize(image, (work_w, work_h), interpolation=cv2.INTER_AREA)
    return small, max(2, round(25 * work_w / _WORK_WIDTH))


def paper_color(image: np.ndarray) -> np.ndarray:
    """Average RGB color of the grey paper, from a surface fitted to the
    border band. `image` is (H, W, 3) RGB uint8. Returns float (3,).

    The surface, not the raw border pixels, is averaged: it covers the whole
    frame (including under the copra) and its robust refit ignores specks.
    """
    small, border_band = working_copy(image)
    blurred = cv2.GaussianBlur(small, (5, 5), 0)
    surfaces, _ = fit_border_surfaces(blurred, border_band)
    return surfaces.reshape(-1, 3).mean(axis=0).astype(np.float64)


def mask_image_background(
    image: np.ndarray, threshold_k: float = 4.0, grabcut_iters: int = 3
) -> np.ndarray:
    """Return a boolean (H, W) copra mask at the input resolution.

    `image` must be an (H, W, 3) RGB uint8 array - not BGR.
    """
    if image.ndim != 3:
        raise ValueError(f"mask_image_background expects an RGB image, got shape {image.shape}")
    h0, w0 = image.shape[:2]
    small, border_band = working_copy(image)

    # Morphology sizes scale with the working width (tuned at 1020 px).
    unit = small.shape[1] / _WORK_WIDTH
    open_k = max(3, round(5 * unit) | 1)
    seed_k = max(3, round(15 * unit) | 1)

    lab = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_RGB2LAB), (5, 5), 0).astype(np.float32)
    score = background_distance(lab, border_band)

    seeds = (score > threshold_k).astype(np.uint8)
    seeds = cv2.morphologyEx(seeds, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8))
    seeds = _largest_component(seeds)
    if not seeds.any():
        return np.zeros((h0, w0), dtype=bool)

    if grabcut_iters > 0:
        kernel = np.ones((seed_k, seed_k), np.uint8)
        gc = np.full(seeds.shape, cv2.GC_PR_BGD, np.uint8)
        gc[cv2.dilate(seeds, kernel) == 0] = cv2.GC_BGD
        gc[seeds == 1] = cv2.GC_PR_FGD
        gc[cv2.erode(seeds, kernel) == 1] = cv2.GC_FGD
        gc[:border_band] = gc[-border_band:] = cv2.GC_BGD
        gc[:, :border_band] = gc[:, -border_band:] = cv2.GC_BGD
        if (gc == cv2.GC_FGD).any() or (gc == cv2.GC_PR_FGD).any():
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            # GrabCut's k-means init draws from OpenCV's global RNG, whose
            # state depends on earlier calls in the process; reseed so the
            # same photo always gets the same mask.
            cv2.setRNGSeed(0)
            cv2.grabCut(small, gc, None, bgd_model, fgd_model, grabcut_iters, cv2.GC_INIT_WITH_MASK)
            seeds = _largest_component(np.isin(gc, (cv2.GC_FGD, cv2.GC_PR_FGD)))

    mask = _fill_holes(seeds)
    return cv2.resize(mask, (w0, h0), interpolation=cv2.INTER_NEAREST).astype(bool)
