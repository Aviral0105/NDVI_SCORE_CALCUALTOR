"""
Phase 2 — Preprocessing
========================
Applies the following steps to the raw float32 [0,1] image array:

1. Noise removal      — OpenCV fastNlMeansDenoisingColored (bilateral-style)
2. CLAHE enhancement  — per-channel contrast limited adaptive histogram equalization
3. Radiometric check  — warns if channels look swapped / saturated
4. Optional resize    — only if width > 4096 px (keeps memory manageable)

All operations are applied on a copy; the original array is never modified.
"""

import logging
from typing import Dict, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def remove_noise(arr: np.ndarray, h: float = 10) -> np.ndarray:
    """
    Denoises a float32 [0,1] image using OpenCV Non-Local Means.

    Parameters
    ----------
    arr : (H, W, 3) float32 in [0,1]
    h   : filter strength — higher removes more noise but may blur edges

    Returns
    -------
    Denoised array, same shape and dtype.
    """
    # Convert to uint8 for OpenCV
    u8 = (arr * 255).clip(0, 255).astype(np.uint8)
    if u8.shape[2] == 3:
        denoised = cv2.fastNlMeansDenoisingColored(u8, None, h, h, 7, 21)
    else:
        # Grayscale / single band
        denoised = cv2.fastNlMeansDenoising(u8[:, :, 0], None, h, 7, 21)
        denoised = denoised[:, :, np.newaxis]
    result = denoised.astype(np.float32) / 255.0
    logger.debug("Noise removal applied (h=%.1f)", h)
    return result


def apply_clahe(arr: np.ndarray,
                clip_limit: float = 2.0,
                tile_grid: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Applies CLAHE independently to each channel of a float32 [0,1] image.
    CLAHE improves local contrast without over-amplifying noise.

    Parameters
    ----------
    arr        : (H, W, C) float32 in [0,1]
    clip_limit : threshold for contrast limiting (typ. 1–4)
    tile_grid  : size of grid for histogram equalization

    Returns
    -------
    Contrast-enhanced array, same shape and dtype.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    out = np.empty_like(arr)
    n_ch = arr.shape[2]
    for c in range(n_ch):
        ch_u8 = (arr[:, :, c] * 255).clip(0, 255).astype(np.uint8)
        enhanced = clahe.apply(ch_u8)
        out[:, :, c] = enhanced.astype(np.float32) / 255.0
    logger.debug("CLAHE applied (clip=%.1f, grid=%s)", clip_limit, tile_grid)
    return out


def radiometric_check(arr: np.ndarray, band_cfg: Dict) -> None:
    """
    Logs per-channel statistics and warns about obvious radiometric issues
    such as saturation, near-zero channels, or unexpected channel ordering.

    Parameters
    ----------
    arr      : (H, W, C) float32 in [0,1]
    band_cfg : band configuration dict from phase1_inspect
    """
    channels = band_cfg.get("channels", {})
    ch_names = {info["channel_index"]: name for name, info in channels.items()}

    logger.info("Radiometric channel statistics:")
    for c in range(arr.shape[2]):
        ch = arr[:, :, c]
        name = ch_names.get(c, f"Ch{c}")
        mean_val = ch.mean()
        sat_pct = (ch >= 0.99).mean() * 100
        zero_pct = (ch <= 0.01).mean() * 100
        logger.info(
            f"  [{name:8s}] mean={mean_val:.3f}  "
            f"saturated={sat_pct:.1f}%  near-zero={zero_pct:.1f}%"
        )
        if sat_pct > 20:
            logger.warning(
                f"  ⚠️  Channel {name} is >20% saturated — "
                "consider reducing exposure or using RAW images."
            )
        if zero_pct > 50:
            logger.warning(
                f"  ⚠️  Channel {name} is >50% near-zero — "
                "band may be empty or incorrectly assigned."
            )

    # For OCN camera: NIR channel (ch0) should be brighter than Red (ch2)
    # in vegetated areas — check if that's the case
    if "NIR" in ch_names.values() and "Red" in ch_names.values():
        nir_idx = next(k for k, v in ch_names.items() if v == "NIR")
        red_idx = next(k for k, v in ch_names.items() if v == "Red")
        nir_mean = arr[:, :, nir_idx].mean()
        red_mean = arr[:, :, red_idx].mean()
        if nir_mean > red_mean:
            logger.info("  ✅ NIR mean > Red mean — typical for vegetated scene.")
        else:
            logger.warning(
                "  ⚠️  NIR mean ≤ Red mean. "
                "Scene may be dominated by bare soil, or bands may be swapped."
            )


def maybe_resize(arr: np.ndarray, max_width: int = 4096) -> np.ndarray:
    """
    Resizes the image if it exceeds max_width, preserving aspect ratio.
    Only applied when necessary to keep processing time reasonable.

    Parameters
    ----------
    arr       : (H, W, C) float32
    max_width : maximum allowed width in pixels

    Returns
    -------
    Possibly resized array.
    """
    h, w = arr.shape[:2]
    if w <= max_width:
        return arr
    scale = max_width / w
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    logger.info(f"Image resized from {w}×{h} to {new_w}×{new_h} (scale={scale:.2f})")
    return resized


def preprocess(arr: np.ndarray,
               band_cfg: Dict,
               cfg: Dict) -> np.ndarray:
    """
    Full Phase 2 pipeline entry point.

    Parameters
    ----------
    arr      : (H, W, C) float32 in [0,1] — raw image from Phase 1
    band_cfg : band configuration dict from phase1_inspect
    cfg      : pipeline config dict (from YAML)

    Returns
    -------
    Preprocessed array, same shape and dtype.
    """
    logger.info("=" * 60)
    logger.info("PHASE 2 — Preprocessing")
    logger.info("=" * 60)

    out = arr.copy()

    # 1. Optional resize
    out = maybe_resize(out, max_width=4096)

    # 2. Noise removal
    h_strength = cfg.get("denoise_h", 10)
    out = remove_noise(out, h=h_strength)

    # 3. CLAHE contrast enhancement
    clip = cfg.get("clahe_clip_limit", 2.0)
    grid = tuple(cfg.get("clahe_tile_grid", [8, 8]))
    out = apply_clahe(out, clip_limit=clip, tile_grid=grid)

    # 4. Radiometric statistics & warnings
    radiometric_check(out, band_cfg)

    logger.info("Preprocessing complete.")
    return out
