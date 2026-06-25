"""
Phase 6 — Visualization
Phase 7 — Output Saving
========================

Phase 6 produces a 6-panel figure:
  1. Original image (display channels remapped for human viewing)
  2. Segmentation mask (leaf = white)
  3. Background-removed leaf image
  4. VI heatmap over leaf pixels
  5. Histogram of VI values across leaf pixels
  6. Summary statistics text panel

Phase 7 saves:
  • leaf_mask.png          — binary mask
  • leaf_only.png          — leaf-only image
  • vi_map.tif             — float32 VI raster (GeoTIFF if geo-referenced)
  • results.csv            — per-image summary (average VI, leaf %)
  • visualization.png      — the 6-panel figure
"""

import csv
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Colour-display helper for the OCN false-colour image
# ---------------------------------------------------------------------------

def ocn_to_display_rgb(arr: np.ndarray, band_cfg: Dict) -> np.ndarray:
    """
    Remaps OCN channels (NIR, Green, Red) to an intuitive false-colour
    display: Red channel = Red band, Green = Green band, Blue = NIR.
    This gives a colour-infrared (CIR) composite where healthy vegetation
    appears bright red/magenta, matching the typical CIR convention.

    Falls back to arr[:,:,:3] if channel names are not recognised.
    """
    channels = band_cfg.get("channels", {})
    try:
        r_idx = channels["Red"]["channel_index"]
        g_idx = channels["Green"]["channel_index"]
        n_idx = channels["NIR"]["channel_index"]
        display = np.stack([
            arr[:, :, r_idx],   # Display R ← sensor Red
            arr[:, :, g_idx],   # Display G ← sensor Green
            arr[:, :, n_idx],   # Display B ← NIR (CIR composite)
        ], axis=-1)
    except (KeyError, IndexError):
        display = arr[:, :, :3]
    return np.clip(display, 0, 1)


# ---------------------------------------------------------------------------
# Phase 6 — Visualize
# ---------------------------------------------------------------------------

def visualize(original_arr: np.ndarray,
              preprocessed_arr: np.ndarray,
              mask: np.ndarray,
              leaf_image: np.ndarray,
              vi_map: np.ndarray,
              avg_vi: float,
              vi_name: str,
              band_cfg: Dict,
              cfg: Dict,
              out_path: str) -> None:
    """
    Creates and saves the 6-panel diagnostic figure.

    Parameters
    ----------
    original_arr    : raw image array from Phase 1
    preprocessed_arr: after Phase 2 (used for display)
    mask            : binary leaf mask (H, W) bool
    leaf_image      : background-removed image (H, W, C)
    vi_map          : (H, W) float32 with NaN outside leaf
    avg_vi          : average VI scalar
    vi_name         : e.g. "NDVI"
    band_cfg        : band configuration
    cfg             : pipeline config
    out_path        : path to save the PNG figure
    """
    logger.info("=" * 60)
    logger.info("PHASE 6 — Visualization")
    logger.info("=" * 60)

    cmap_name = cfg.get("colormap", "RdYlGn")
    dpi = cfg.get("dpi", 150)

    # Build display images
    orig_display = ocn_to_display_rgb(original_arr, band_cfg)
    leaf_display = ocn_to_display_rgb(leaf_image, band_cfg)

    # Leaf pixel VI values for histogram
    leaf_vi_vals = vi_map[mask & ~np.isnan(vi_map)]

    fig = plt.figure(figsize=(18, 11), facecolor="#1a1a2e")
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.25,
                  left=0.04, right=0.96, top=0.92, bottom=0.06)

    ax_style = dict(facecolor="#16213e")
    title_kw = dict(color="white", fontsize=11, fontweight="bold", pad=8)
    tick_kw  = dict(colors="white")

    # Panel 1 — Original image
    ax1 = fig.add_subplot(gs[0, 0], **ax_style)
    ax1.imshow(orig_display)
    ax1.set_title("① Original Image (CIR Display)", **title_kw)
    ax1.axis("off")
    ax1.text(0.01, 0.01, "CIR: R=Red, G=Green, B=NIR",
             transform=ax1.transAxes, color="lightgrey", fontsize=7)

    # Panel 2 — Segmentation mask
    ax2 = fig.add_subplot(gs[0, 1], **ax_style)
    ax2.imshow(mask, cmap="Greens", vmin=0, vmax=1)
    ax2.set_title("② Wheat Leaf Mask", **title_kw)
    ax2.axis("off")
    pct = mask.mean() * 100
    ax2.text(0.5, 0.02, f"Leaf coverage: {pct:.1f}%",
             transform=ax2.transAxes, ha="center", color="lightgreen", fontsize=9)

    # Panel 3 — Leaf-only image
    ax3 = fig.add_subplot(gs[0, 2], **ax_style)
    ax3.imshow(leaf_display)
    ax3.set_title("③ Leaf-Only Image (Background Removed)", **title_kw)
    ax3.axis("off")

    # Panel 4 — VI heatmap
    ax4 = fig.add_subplot(gs[1, 0], **ax_style)
    vi_display = vi_map.copy()
    vi_display[~mask] = np.nan   # transparent background
    vmin, vmax = -0.2, 0.9
    im = ax4.imshow(vi_display, cmap=cmap_name, vmin=vmin, vmax=vmax,
                    interpolation="nearest")
    ax4.set_title(f"④ {vi_name} Heatmap (Leaf Pixels Only)", **title_kw)
    ax4.axis("off")
    cbar = fig.colorbar(im, ax=ax4, fraction=0.04, pad=0.02)
    cbar.ax.yaxis.set_tick_params(**tick_kw)
    cbar.set_label(vi_name, color="white", fontsize=9)

    # Panel 5 — Histogram
    ax5 = fig.add_subplot(gs[1, 1], **ax_style)
    if leaf_vi_vals.size > 0:
        n_bins = min(80, max(20, leaf_vi_vals.size // 500))
        n, bins, patches = ax5.hist(leaf_vi_vals, bins=n_bins, edgecolor="none",
                                    density=False)
        # Colour bars by value using the chosen colormap
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cm = plt.get_cmap(cmap_name)
        for patch, left_edge in zip(patches, bins[:-1]):
            patch.set_facecolor(cm(norm(left_edge)))
        ax5.axvline(avg_vi, color="white", linestyle="--", linewidth=1.5,
                    label=f"Mean = {avg_vi:.4f}")
        ax5.legend(facecolor="#16213e", labelcolor="white", fontsize=9)
    ax5.set_xlabel(vi_name, color="white", fontsize=9)
    ax5.set_ylabel("Pixel count", color="white", fontsize=9)
    ax5.set_title(f"⑤ {vi_name} Distribution", **title_kw)
    ax5.tick_params(**tick_kw)
    ax5.spines[["top", "right"]].set_visible(False)
    for sp in ["bottom", "left"]:
        ax5.spines[sp].set_color("grey")

    # Panel 6 — Statistics summary
    ax6 = fig.add_subplot(gs[1, 2], **ax_style)
    ax6.axis("off")
    n_leaf = mask.sum()
    std_vi = float(np.nanstd(leaf_vi_vals)) if leaf_vi_vals.size > 0 else float("nan")
    p25 = float(np.nanpercentile(leaf_vi_vals, 25)) if leaf_vi_vals.size > 0 else float("nan")
    p75 = float(np.nanpercentile(leaf_vi_vals, 75)) if leaf_vi_vals.size > 0 else float("nan")

    stats_text = (
        f"  Vegetation Index : {vi_name}\n\n"
        f"  Leaf pixels      : {n_leaf:,}\n"
        f"  Leaf coverage    : {mask.mean()*100:.2f}%\n\n"
        f"  Mean {vi_name:<6s}       : {avg_vi:+.4f}\n"
        f"  Std dev          : {std_vi:.4f}\n"
        f"  25th percentile  : {p25:+.4f}\n"
        f"  75th percentile  : {p75:+.4f}\n\n"
        f"  Image size       : {original_arr.shape[1]}×{original_arr.shape[0]} px\n"
        f"  Spectral bands   : {original_arr.shape[2]}\n"
    )
    ax6.text(0.05, 0.92, "⑥ Summary Statistics",
             transform=ax6.transAxes, va="top",
             color="white", fontsize=11, fontweight="bold")
    ax6.text(0.05, 0.80, stats_text,
             transform=ax6.transAxes, va="top",
             color="lightcyan", fontsize=10, family="monospace",
             linespacing=1.5)

    fig.suptitle(
        f"Wheat Vegetation Index Pipeline  —  {vi_name}",
        color="white", fontsize=14, fontweight="bold", y=0.97
    )

    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Visualization saved → {out_path}")


# ---------------------------------------------------------------------------
# Phase 7 — Save outputs
# ---------------------------------------------------------------------------

def save_outputs(mask: np.ndarray,
                 leaf_image: np.ndarray,
                 vi_map: np.ndarray,
                 avg_vi: float,
                 vi_name: str,
                 stem: str,
                 out_dir: str,
                 metadata: Dict,
                 band_cfg: Dict,
                 cfg: Dict) -> None:
    """
    Saves all pipeline outputs to the output directory.

    Parameters
    ----------
    mask       : (H, W) bool leaf mask
    leaf_image : (H, W, C) float32 leaf-only image
    vi_map     : (H, W) float32 VI map (NaN outside leaf)
    avg_vi     : scalar average VI
    vi_name    : e.g. "NDVI"
    stem       : base filename (no extension)
    out_dir    : root output directory
    metadata   : from Phase 1
    band_cfg   : from Phase 1
    cfg        : pipeline config
    """
    logger.info("=" * 60)
    logger.info("PHASE 7 — Saving Outputs")
    logger.info("=" * 60)

    masks_dir = Path(out_dir) / "masks"
    vi_dir    = Path(out_dir) / "vi_maps"
    csv_dir   = Path(out_dir) / "csv"
    viz_dir   = Path(out_dir) / "visualizations"
    for d in [masks_dir, vi_dir, csv_dir, viz_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Leaf mask (PNG, 8-bit binary)
    mask_path = masks_dir / f"{stem}_leaf_mask.png"
    cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))
    logger.info(f"  Saved leaf mask       → {mask_path}")

    # 2. Leaf-only image (PNG, 8-bit)
    leaf_u8 = (np.clip(leaf_image[:, :, :3], 0, 1) * 255).astype(np.uint8)
    leaf_bgr = cv2.cvtColor(leaf_u8, cv2.COLOR_RGB2BGR)
    leaf_path = masks_dir / f"{stem}_leaf_only.png"
    cv2.imwrite(str(leaf_path), leaf_bgr)
    logger.info(f"  Saved leaf-only image → {leaf_path}")

    # 3. VI map (float32 TIFF via rasterio if available, else NPY)
    vi_path = vi_dir / f"{stem}_{vi_name}_map.tif"
    try:
        import rasterio
        from rasterio.transform import from_bounds
        h, w = vi_map.shape
        # Use identity transform if no geo-referencing available
        transform = metadata.get("transform") or from_bounds(0, 0, 1, 1, w, h)
        crs = metadata.get("crs") or "EPSG:4326"
        with rasterio.open(
            str(vi_path), "w",
            driver="GTiff",
            height=h, width=w,
            count=1,
            dtype=rasterio.float32,
            crs=crs,
            transform=transform,
            nodata=np.nan,
        ) as dst:
            dst.write(vi_map.astype(np.float32), 1)
        logger.info(f"  Saved VI GeoTIFF      → {vi_path}")
    except Exception as e:
        npy_path = vi_dir / f"{stem}_{vi_name}_map.npy"
        np.save(str(npy_path), vi_map)
        logger.warning(f"  rasterio save failed ({e}). Saved as NPY → {npy_path}")

    # 4. CSV results
    csv_path = csv_dir / "results.csv"
    file_exists = csv_path.exists()
    with open(str(csv_path), "a", newline="") as f:
        fieldnames = ["filename", "vi_name", "avg_vi", "leaf_px",
                      "total_px", "leaf_pct", "camera_model", "n_bands"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "filename":     stem,
            "vi_name":      vi_name,
            "avg_vi":       f"{avg_vi:.6f}",
            "leaf_px":      int(mask.sum()),
            "total_px":     int(mask.size),
            "leaf_pct":     f"{mask.mean()*100:.4f}",
            "camera_model": metadata.get("camera_model", "unknown"),
            "n_bands":      metadata.get("n_bands", 3),
        })
    logger.info(f"  Appended CSV row      → {csv_path}")

    logger.info("All outputs saved successfully.")
