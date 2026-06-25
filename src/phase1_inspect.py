"""
Phase 1 — Image Inspection
===========================
Reads any TIFF/JPG/PNG image and:
  • Reports number of bands, dtype, spatial dimensions
  • Detects the camera model from EXIF (MAPIR Survey3W OCN)
  • Determines correct spectral band-to-channel mapping
  • Decides whether true NDVI is computable
"""

import logging
import os
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
from PIL import Image, ExifTags

logger = logging.getLogger(__name__)


MAPIR_BAND_MAP: Dict[str, Dict[str, Any]] = {
    # MAPIR Survey3W OCN: when JPEG is saved, Bayer demosaicing assigns:
    #   R channel = Orange filter (~650 nm)  → Red band
    #   G channel = Cyan filter   (~550 nm)  → Green band
    #   B channel = NIR filter    (~850 nm)  → NIR band
    # This produces the characteristic purple/magenta appearance since NIR >> R,G
    "survey3w_ocn": {
        "description": "MAPIR Survey3W OCN (Orange-Cyan-NIR)",
        "channels": {
            "Red":   {"channel_index": 0, "wavelength_nm": 650},
            "Green": {"channel_index": 1, "wavelength_nm": 550},
            "NIR":   {"channel_index": 2, "wavelength_nm": 850},
        },
        "ndvi_possible": True,
        "recommended_vi": "NDVI",
    },
    "survey3_ocn": {
        "description": "MAPIR Survey3 OCN (Orange-Cyan-NIR)",
        "channels": {
            "Red":   {"channel_index": 0, "wavelength_nm": 650},
            "Green": {"channel_index": 1, "wavelength_nm": 550},
            "NIR":   {"channel_index": 2, "wavelength_nm": 850},
        },
        "ndvi_possible": True,
        "recommended_vi": "NDVI",
    },
    "rgb_fallback": {
        "description": "Standard RGB (no NIR band)",
        "channels": {
            "Red":   {"channel_index": 0, "wavelength_nm": 660},
            "Green": {"channel_index": 1, "wavelength_nm": 550},
            "Blue":  {"channel_index": 2, "wavelength_nm": 470},
        },
        "ndvi_possible": False,
        "recommended_vi": "VARI",
    },
}


def read_image_as_array(image_path: str) -> Tuple[np.ndarray, Dict]:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    metadata: Dict[str, Any] = {"path": str(path), "filename": path.name}

    if path.suffix.lower() in (".tif", ".tiff"):
        try:
            import rasterio
            with rasterio.open(str(path)) as src:
                arr = src.read()
                arr = np.moveaxis(arr, 0, -1).astype(np.float32)
                if src.dtypes[0] == "uint8":
                    arr /= 255.0
                elif src.dtypes[0] == "uint16":
                    arr /= 65535.0
                metadata.update({
                    "format": "GeoTIFF", "crs": str(src.crs),
                    "transform": src.transform, "n_bands": src.count,
                    "height": src.height, "width": src.width,
                    "dtype": str(src.dtypes[0]), "nodata": src.nodata,
                })
                return arr, metadata
        except Exception as e:
            logger.warning(f"rasterio failed ({e}); falling back to Pillow.")

    pil_img = Image.open(str(path))
    exif_data = {}
    try:
        raw_exif = pil_img._getexif() or {}
        exif_data = {ExifTags.TAGS.get(k, k): v for k, v in raw_exif.items()
                     if isinstance(v, (str, int, float))}
    except Exception:
        pass

    arr = np.array(pil_img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]

    metadata.update({
        "format": pil_img.format or path.suffix.upper().lstrip("."),
        "mode": pil_img.mode,
        "n_bands": arr.shape[2],
        "height": arr.shape[0],
        "width": arr.shape[1],
        "dtype": "uint8",
        "exif": exif_data,
        "camera_make":  exif_data.get("Make", "Unknown"),
        "camera_model": exif_data.get("Model", "Unknown"),
    })
    logger.info(f"Loaded image: {path.name} — {arr.shape[2]} bands, "
                f"{arr.shape[1]}×{arr.shape[0]}, mode={pil_img.mode}")
    return arr, metadata


def detect_band_config(metadata: Dict) -> Dict[str, Any]:
    cam_model = metadata.get("camera_model", "").lower().replace(" ", "_")
    for key, config in MAPIR_BAND_MAP.items():
        if key in cam_model:
            logger.info(f"Recognised camera: {config['description']}")
            return config
    n_bands = metadata.get("n_bands", 3)
    if n_bands == 1:
        return {"description": "Single-band", "channels": {"NIR": {"channel_index": 0}},
                "ndvi_possible": False, "recommended_vi": "None"}
    logger.warning(f"Camera model '{cam_model}' not recognised. Defaulting to RGB.")
    return MAPIR_BAND_MAP["rgb_fallback"]


def inspect_image(image_path: str) -> Tuple[np.ndarray, Dict, Dict]:
    logger.info("=" * 60)
    logger.info("PHASE 1 — Image Inspection")
    logger.info("=" * 60)

    arr, metadata = read_image_as_array(image_path)
    band_cfg = detect_band_config(metadata)

    print("\n╔══════════════════════════════════════════════════╗")
    print("║           IMAGE INSPECTION REPORT               ║")
    print("╚══════════════════════════════════════════════════╝")
    print(f"  File          : {metadata['filename']}")
    print(f"  Dimensions    : {metadata['width']} × {metadata['height']} px")
    print(f"  Bands         : {metadata['n_bands']}")
    print(f"  Camera Make   : {metadata.get('camera_make', 'N/A')}")
    print(f"  Camera Model  : {metadata.get('camera_model', 'N/A')}")
    print(f"\n  ── Band Configuration ──────────────────────────")
    print(f"  Detected      : {band_cfg['description']}")
    for band_name, info in band_cfg["channels"].items():
        print(f"    Channel {info['channel_index']} (JPEG {'R' if info['channel_index']==0 else 'G' if info['channel_index']==1 else 'B'}) "
              f"→ {band_name:8s} (~{info['wavelength_nm']} nm)")
    print(f"\n  ── NDVI Assessment ─────────────────────────────")
    if band_cfg["ndvi_possible"]:
        print("  ✅ True NDVI IS computable (NIR in B-channel, Red in R-channel)")
    else:
        print(f"  ⚠️  Fallback VI: {band_cfg['recommended_vi']}")
    print()
    return arr, metadata, band_cfg
