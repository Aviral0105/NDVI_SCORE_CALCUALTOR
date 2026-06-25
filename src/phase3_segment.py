"""
Phase 3 — Wheat Leaf Segmentation
===================================
Produces a binary mask (True = wheat leaf pixel).

Strategy — Multi-Cue Classical Computer Vision
----------------------------------------------
Chosen over ML/DL because:
  • No labelled training data is available yet.
  • The OCN camera provides an NIR channel enabling a genuine vegetation
    signal (NDVI > threshold) that is highly discriminative vs soil/stones.
  • Classical CV is fully reproducible and interpretable — important for
    inclusion in a research paper.
  • It runs in seconds without a GPU.

The pipeline combines three independent cues and AND-s them together:

  Cue A — NDVI threshold
    Pixels with NDVI > ndvi_threshold are candidate vegetation.
    This alone separates green/NIR-reflective leaves from bare soil,
    stones, and dry straw very reliably with the OCN sensor.

  Cue B — HSV colour filter
    Wheat leaves appear greenish in the false-colour OCN image.
    A broad HSV range removes obviously non-leaf pixels that pass
    the NDVI test (e.g. bright glare spots, white reference panels).

  Cue C — Shadow exclusion
    Very dark pixels (HSV Value < shadow_v_min) are excluded regardless
    of their NDVI, since shadowed areas produce unreliable VI values.

  Post-processing
    Morphological opening removes speckle noise.
    Morphological closing fills small holes inside leaves.
    Small disconnected components < min_leaf_area_px are dropped.
"""

import logging
from typing import Dict, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: compute NDVI from the preprocessed array
# ---------------------------------------------------------------------------

def compute_raw_ndvi(arr: np.ndarray, band_cfg: Dict) -> np.ndarray:
    """
    Computes pixel-wise NDVI = (NIR - Red) / (NIR + Red + ε).

    Parameters
    ----------
    arr      : (H, W, C) float32 in [0,1]
    band_cfg : band configuration dict

    Returns
    -------
    ndvi : (H, W) float32 in [-1, 1]
    """
    channels = band_cfg["channels"]
    nir_idx = channels["NIR"]["channel_index"]
    red_idx = channels["Red"]["channel_index"]

    nir = arr[:, :, nir_idx].astype(np.float64)
    red = arr[:, :, red_idx].astype(np.float64)

    eps = 1e-8
    ndvi = (nir - red) / (nir + red + eps)
    ndvi = np.clip(ndvi, -1.0, 1.0).astype(np.float32)
    return ndvi


# ---------------------------------------------------------------------------
# Cue A — NDVI threshold mask
# ---------------------------------------------------------------------------

def ndvi_mask(ndvi: np.ndarray, threshold: float) -> np.ndarray:
    """
    Returns a boolean mask: True where NDVI > threshold.
    """
    mask = ndvi > threshold
    pct = mask.mean() * 100
    logger.debug(f"NDVI mask (threshold={threshold:.2f}): {pct:.1f}% pixels selected")
    return mask


# ---------------------------------------------------------------------------
# Cue B — HSV colour filter
# ---------------------------------------------------------------------------

def hsv_leaf_mask(arr: np.ndarray,
                  lower: Tuple[int, int, int],
                  upper: Tuple[int, int, int]) -> np.ndarray:
    """
    Creates a mask of pixels whose HSV colour falls within [lower, upper].
    For the OCN false-colour image, green hues correspond to healthy leaves.

    Parameters
    ----------
    arr   : (H, W, 3) float32 in [0,1] — the preprocessed image
    lower : (H_min, S_min, V_min) in OpenCV HSV scale (H:0-179, S:0-255, V:0-255)
    upper : (H_max, S_max, V_max)

    Returns
    -------
    Boolean mask (H, W).
    """
    u8 = (arr[:, :, :3] * 255).clip(0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(u8, cv2.COLOR_RGB2HSV)

    lower_arr = np.array(lower, dtype=np.uint8)
    upper_arr = np.array(upper, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_arr, upper_arr).astype(bool)
    pct = mask.mean() * 100
    logger.debug(f"HSV colour mask: {pct:.1f}% pixels selected")
    return mask


# ---------------------------------------------------------------------------
# Cue C — Shadow exclusion
# ---------------------------------------------------------------------------

def shadow_exclusion_mask(arr: np.ndarray, v_min: int = 40) -> np.ndarray:
    """
    Returns True for pixels that are NOT shadowed (HSV Value >= v_min).

    Parameters
    ----------
    arr   : (H, W, 3) float32 in [0,1]
    v_min : minimum HSV Value (0–255) — pixels below this are shadows

    Returns
    -------
    Boolean mask (H, W): True = not a shadow.
    """
    u8 = (arr[:, :, :3] * 255).clip(0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(u8, cv2.COLOR_RGB2HSV)
    value_channel = hsv[:, :, 2]
    mask = value_channel >= v_min
    pct = (~mask).mean() * 100
    logger.debug(f"Shadow exclusion: {pct:.1f}% of pixels identified as shadow")
    return mask


# ---------------------------------------------------------------------------
# Post-processing: morphological cleanup and small-component removal
# ---------------------------------------------------------------------------

def morphological_cleanup(mask: np.ndarray, min_area: int = 200) -> np.ndarray:
    """
    Applies morphological opening (removes noise) and closing (fills holes),
    then drops connected components smaller than min_area pixels.

    Parameters
    ----------
    mask     : (H, W) bool array
    min_area : minimum component area in pixels to retain

    Returns
    -------
    Cleaned boolean mask (H, W).
    """
    m = mask.astype(np.uint8) * 255

    # Opening — remove small isolated noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel_open, iterations=2)

    # Closing — fill small holes in leaf blades
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    # Remove small components
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    clean = np.zeros_like(m)
    kept = 0
    for label in range(1, n_labels):   # label 0 = background
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            clean[labels == label] = 255
            kept += 1
    logger.debug(f"Morphological cleanup: {kept}/{n_labels-1} components retained "
                 f"(min_area={min_area} px)")
    return clean.astype(bool)


# ---------------------------------------------------------------------------
# Full Phase 3 pipeline
# ---------------------------------------------------------------------------

def segment_leaves(arr: np.ndarray,
                   band_cfg: Dict,
                   cfg: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full Phase 3 entry point.

    Parameters
    ----------
    arr      : (H, W, C) float32 in [0,1] — preprocessed image
    band_cfg : band configuration dict from Phase 1
    cfg      : pipeline configuration dict

    Returns
    -------
    mask : (H, W) bool — True = wheat leaf pixel
    ndvi : (H, W) float32 — NDVI map (used again in Phase 5)
    """
    logger.info("=" * 60)
    logger.info("PHASE 3 — Leaf Segmentation")
    logger.info("=" * 60)

    # ---- Compute NDVI for Cue A ----
    ndvi = compute_raw_ndvi(arr, band_cfg)
    ndvi_thresh = cfg.get("ndvi_threshold", 0.10)

    # ---- Cue A: NDVI threshold ----
    mask_a = ndvi_mask(ndvi, ndvi_thresh)

    # ---- Cue B: HSV colour filter ----
    hsv_lower = tuple(cfg.get("hsv_green_lower", [25, 20, 40]))
    hsv_upper = tuple(cfg.get("hsv_green_upper", [95, 255, 255]))
    mask_b = hsv_leaf_mask(arr, hsv_lower, hsv_upper)

    # ---- Cue C: Shadow exclusion ----
    v_min = cfg.get("shadow_v_min", 40)
    mask_c = shadow_exclusion_mask(arr, v_min=v_min)

    # ---- Combine all cues (intersection) ----
    combined = mask_a & mask_b & mask_c

    # ---- Post-process ----
    min_area = cfg.get("min_leaf_area_px", 200)
    final_mask = morphological_cleanup(combined, min_area=min_area)

    leaf_pct = final_mask.mean() * 100
    logger.info(f"Segmentation complete: {leaf_pct:.2f}% of pixels classified as leaf.")

    if leaf_pct < 1.0:
        logger.warning(
            "Very few leaf pixels detected (<1%). Consider lowering "
            "ndvi_threshold or adjusting hsv_green_lower/upper in config.yaml."
        )

    return final_mask, ndvi
