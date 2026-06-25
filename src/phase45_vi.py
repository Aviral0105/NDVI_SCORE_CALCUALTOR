"""
Phase 4 — Background Removal
Phase 5 — Vegetation Index Computation
========================================

Phase 4
-------
Applies the leaf binary mask to the preprocessed image, zeroing out all
pixels that are NOT wheat leaf (soil, stones, shadows, dry vegetation, etc.).

Phase 5
-------
Computes the chosen vegetation index ONLY over leaf pixels.
Supported indices:

  NDVI  = (NIR - Red) / (NIR + Red)
          Requires NIR + Red bands (available for MAPIR OCN sensor).
          Range: [-1, 1] — healthy green vegetation typically 0.3–0.8.

  VARI  = (Green - Red) / (Green + Red - Blue)
          Visible Atmospherically Resistant Index.
          RGB-only fallback when NIR is unavailable.

  ExG   = 2·Green - Red - Blue
          Excess Green Index. Simple RGB-only index.

  GLI   = (2·Green - Red - Blue) / (2·Green + Red + Blue)
          Green Leaf Index. Normalised RGB version.

For the MAPIR Survey3W OCN sensor, NDVI is the primary choice.
"""

import logging
from typing import Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase 4 — Background removal
# ---------------------------------------------------------------------------

def remove_background(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Zeros out all non-leaf pixels in the image.

    Parameters
    ----------
    arr  : (H, W, C) float32 in [0,1]
    mask : (H, W) bool — True = leaf pixel

    Returns
    -------
    leaf_image : (H, W, C) float32 — non-leaf pixels set to 0
    """
    logger.info("=" * 60)
    logger.info("PHASE 4 — Background Removal")
    logger.info("=" * 60)

    leaf_image = arr.copy()
    for c in range(arr.shape[2]):
        leaf_image[:, :, c] = np.where(mask, arr[:, :, c], 0.0)

    n_leaf = mask.sum()
    n_total = mask.size
    logger.info(f"Retained {n_leaf:,} leaf pixels out of {n_total:,} total "
                f"({n_leaf/n_total*100:.2f}%).")
    return leaf_image


# ---------------------------------------------------------------------------
# Vegetation Index formulas
# ---------------------------------------------------------------------------

def compute_ndvi(arr: np.ndarray, band_cfg: Dict) -> np.ndarray:
    """NDVI = (NIR - Red) / (NIR + Red + ε)"""
    nir_idx = band_cfg["channels"]["NIR"]["channel_index"]
    red_idx = band_cfg["channels"]["Red"]["channel_index"]
    nir = arr[:, :, nir_idx].astype(np.float64)
    red = arr[:, :, red_idx].astype(np.float64)
    eps = 1e-8
    vi = (nir - red) / (nir + red + eps)
    return np.clip(vi, -1.0, 1.0).astype(np.float32)


def compute_vari(arr: np.ndarray, band_cfg: Dict) -> np.ndarray:
    """VARI = (Green - Red) / (Green + Red - Blue + ε)"""
    ch = band_cfg["channels"]
    g = arr[:, :, ch["Green"]["channel_index"]].astype(np.float64)
    r = arr[:, :, ch["Red"]["channel_index"]].astype(np.float64)
    b_idx = next((v["channel_index"] for k, v in ch.items() if "blue" in k.lower()), 2)
    b = arr[:, :, b_idx].astype(np.float64)
    eps = 1e-8
    vi = (g - r) / (g + r - b + eps)
    return np.clip(vi, -1.0, 1.0).astype(np.float32)


def compute_exg(arr: np.ndarray, band_cfg: Dict) -> np.ndarray:
    """ExG = 2·G - R - B"""
    ch = band_cfg["channels"]
    g = arr[:, :, ch["Green"]["channel_index"]].astype(np.float64)
    r = arr[:, :, ch["Red"]["channel_index"]].astype(np.float64)
    b_idx = next((v["channel_index"] for k, v in ch.items() if "blue" in k.lower()), 2)
    b = arr[:, :, b_idx].astype(np.float64)
    vi = 2.0 * g - r - b
    return vi.astype(np.float32)


def compute_gli(arr: np.ndarray, band_cfg: Dict) -> np.ndarray:
    """GLI = (2G - R - B) / (2G + R + B + ε)"""
    ch = band_cfg["channels"]
    g = arr[:, :, ch["Green"]["channel_index"]].astype(np.float64)
    r = arr[:, :, ch["Red"]["channel_index"]].astype(np.float64)
    b_idx = next((v["channel_index"] for k, v in ch.items() if "blue" in k.lower()), 2)
    b = arr[:, :, b_idx].astype(np.float64)
    eps = 1e-8
    vi = (2 * g - r - b) / (2 * g + r + b + eps)
    return np.clip(vi, -1.0, 1.0).astype(np.float32)


VI_FUNCTIONS = {
    "NDVI": compute_ndvi,
    "VARI": compute_vari,
    "ExG":  compute_exg,
    "GLI":  compute_gli,
}


# ---------------------------------------------------------------------------
# Phase 5 — Compute vegetation index
# ---------------------------------------------------------------------------

def compute_vi(arr: np.ndarray,
               mask: np.ndarray,
               band_cfg: Dict,
               cfg: Dict) -> Tuple[np.ndarray, float, str]:
    """
    Full Phase 5 entry point.

    Selects and computes the appropriate vegetation index, then calculates
    the average value exclusively over leaf pixels (mask == True).

    Parameters
    ----------
    arr      : (H, W, C) float32 — background-removed image (non-leaf = 0)
    mask     : (H, W) bool — True = leaf pixel
    band_cfg : band configuration from Phase 1
    cfg      : pipeline configuration dict

    Returns
    -------
    vi_map      : (H, W) float32 — full VI map (NaN outside leaf area)
    avg_vi      : scalar float — mean VI over leaf pixels only
    vi_name     : str — name of the VI that was computed
    """
    logger.info("=" * 60)
    logger.info("PHASE 5 — Vegetation Index Computation")
    logger.info("=" * 60)

    # Determine which VI to use
    requested_vi = cfg.get("vi_mode", "NDVI").upper()
    ndvi_possible = band_cfg.get("ndvi_possible", False)

    if requested_vi == "NDVI" and not ndvi_possible:
        logger.warning(
            "NDVI requested but NIR or Red band is not available. "
            "Falling back to VARI (RGB-based)."
        )
        vi_name = "VARI"
    elif requested_vi not in VI_FUNCTIONS:
        logger.warning(f"Unknown VI '{requested_vi}'. Defaulting to NDVI.")
        vi_name = "NDVI"
    else:
        vi_name = requested_vi

    logger.info(f"Computing {vi_name} over the entire image...")
    vi_fn = VI_FUNCTIONS[vi_name]
    vi_full = vi_fn(arr, band_cfg)

    # Create masked VI map — NaN outside leaf pixels
    vi_map = np.full_like(vi_full, np.nan)
    vi_map[mask] = vi_full[mask]

    # Compute mean over leaf pixels
    leaf_vals = vi_full[mask]
    if leaf_vals.size == 0:
        logger.error("No leaf pixels found — cannot compute average VI.")
        avg_vi = float("nan")
    else:
        avg_vi = float(np.nanmean(leaf_vals))
        std_vi = float(np.nanstd(leaf_vals))
        logger.info(f"{vi_name} over leaf pixels: mean={avg_vi:.4f}, std={std_vi:.4f}, "
                    f"n={leaf_vals.size:,}")

    print(f"\n  ╔══════════════════════════════════════╗")
    print(f"  ║  Average {vi_name:<6s} (leaf only): {avg_vi:+.4f}  ║")
    print(f"  ╚══════════════════════════════════════╝\n")

    return vi_map, avg_vi, vi_name
