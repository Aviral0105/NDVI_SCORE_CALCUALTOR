"""
main.py — Wheat Vegetation Index Pipeline
==========================================
Orchestrates all phases for single-image or batch processing.

Usage
-----
  # Single image:
  python main.py --input input/my_image.jpg

  # Batch (all images in the input folder):
  python main.py --batch

  # Custom config:
  python main.py --input input/img.jpg --config config/config.yaml
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import yaml

# ---- Setup logging ----
LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)-20s  %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/pipeline.log", mode="a"),
    ],
)
logger = logging.getLogger("wheat_pipeline")

# ---- Import pipeline phases ----
sys.path.insert(0, str(Path(__file__).parent))
from src.phase1_inspect    import inspect_image
from src.phase2_preprocess import preprocess
from src.phase3_segment    import segment_leaves
from src.phase45_vi        import remove_background, compute_vi
from src.phase67_output    import visualize, save_outputs


def load_config(config_path: str) -> dict:
    """Loads YAML configuration file."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Configuration loaded from: {config_path}")
    return cfg


def process_one(image_path: str, cfg: dict) -> None:
    """
    Runs the full 7-phase pipeline on a single image.

    Parameters
    ----------
    image_path : full path to input image
    cfg        : pipeline configuration dict
    """
    t0 = time.time()
    stem = Path(image_path).stem
    out_dir = cfg.get("output_dir", "output")

    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info(f"║  Processing: {stem:<44s} ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")

    # Phase 1 — Inspect
    arr_orig, metadata, band_cfg = inspect_image(image_path)

    # Phase 2 — Preprocess
    arr_pre = preprocess(arr_orig, band_cfg, cfg)

    # Phase 3 — Segment leaves
    mask, ndvi_raw = segment_leaves(arr_pre, band_cfg, cfg)

    # Phase 4 — Remove background
    leaf_image = remove_background(arr_pre, mask)

    # Phase 5 — Compute VI
    vi_map, avg_vi, vi_name = compute_vi(leaf_image, mask, band_cfg, cfg)

    # Phase 6 — Visualize
    viz_path = str(Path(out_dir) / "visualizations" / f"{stem}_visualization.png")
    Path(viz_path).parent.mkdir(parents=True, exist_ok=True)
    visualize(
        original_arr=arr_orig,
        preprocessed_arr=arr_pre,
        mask=mask,
        leaf_image=leaf_image,
        vi_map=vi_map,
        avg_vi=avg_vi,
        vi_name=vi_name,
        band_cfg=band_cfg,
        cfg=cfg,
        out_path=viz_path,
    )

    # Phase 7 — Save outputs
    save_outputs(
        mask=mask,
        leaf_image=leaf_image,
        vi_map=vi_map,
        avg_vi=avg_vi,
        vi_name=vi_name,
        stem=stem,
        out_dir=out_dir,
        metadata=metadata,
        band_cfg=band_cfg,
        cfg=cfg,
    )

    elapsed = time.time() - t0
    logger.info(f"✅  {stem} processed in {elapsed:.1f}s — "
                f"Average {vi_name} (leaf only) = {avg_vi:+.4f}")


def run_batch(cfg: dict) -> None:
    """Processes all images in the input directory."""
    in_dir = Path(cfg.get("input_dir", "input"))
    extensions = cfg.get("extensions", [".tif", ".tiff", ".jpg", ".jpeg", ".png"])
    files = sorted(
        f for f in in_dir.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    )
    if not files:
        logger.warning(f"No images found in '{in_dir}'. "
                       "Copy your images there and re-run.")
        return

    logger.info(f"Batch mode: found {len(files)} image(s) in '{in_dir}'.")
    for fp in files:
        try:
            process_one(str(fp), cfg)
        except Exception as e:
            logger.error(f"Failed on {fp.name}: {e}", exc_info=True)


def main():
    parser = argparse.ArgumentParser(
        description="Wheat Leaf Vegetation Index Pipeline"
    )
    parser.add_argument("--input",  type=str, help="Path to a single image file.")
    parser.add_argument("--batch",  action="store_true",
                        help="Process all images in input_dir (from config).")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Path to YAML config file.")
    args = parser.parse_args()

    # Ensure logs folder exists
    Path("logs").mkdir(exist_ok=True)

    cfg = load_config(args.config)

    if args.input:
        process_one(args.input, cfg)
    elif args.batch:
        run_batch(cfg)
    else:
        logger.info("No --input or --batch specified. Running on demo image...")
        # Try to find any image in the input directory
        in_dir = Path(cfg.get("input_dir", "input"))
        images = sorted(in_dir.glob("*"))
        images = [f for f in images if f.suffix.lower() in
                  [".tif", ".tiff", ".jpg", ".jpeg", ".png"]]
        if images:
            process_one(str(images[0]), cfg)
        else:
            logger.error("No images found. Use --input <path> or --batch.")
            sys.exit(1)


if __name__ == "__main__":
    main()
