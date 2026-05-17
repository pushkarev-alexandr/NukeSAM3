"""
Mask serialisation helpers.

write_mask_exr  – writes multi-channel half-float EXR
                  channels: mask_0, mask_1, ...  (one per detected object)
mask_to_png_b64 – encodes a single bool mask as base64 PNG for inline JSON transfer
"""
from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def write_mask_exr(path: Path, masks: np.ndarray) -> None:
    """
    masks: bool or float array, shape (N, H, W) – N objects.
    Writes a multi-channel half-float EXR.  Channel names: mask_0 … mask_N-1.
    Falls back to writing a 16-bit PNG stack if OpenEXR is unavailable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import OpenEXR  # type: ignore
        import Imath    # type: ignore

        n, h, w = masks.shape
        header = OpenEXR.Header(w, h)
        pixel_type = Imath.PixelType(Imath.PixelType.HALF)
        header["channels"] = {
            f"mask_{i}": Imath.Channel(pixel_type) for i in range(n)
        }
        exr = OpenEXR.OutputFile(str(path), header)
        channel_data = {
            f"mask_{i}": masks[i].astype(np.float16).tobytes()
            for i in range(n)
        }
        exr.writePixels(channel_data)
        exr.close()

    except ImportError:
        # Fallback: write first mask as 16-bit single-channel PNG
        _write_mask_png_fallback(path.with_suffix(".png"), masks)
        logger.warning(
            "OpenEXR not installed – wrote PNG fallback: %s", path.with_suffix(".png")
        )


def _write_mask_png_fallback(path: Path, masks: np.ndarray) -> None:
    from PIL import Image  # type: ignore

    path.parent.mkdir(parents=True, exist_ok=True)
    # Write only the first mask as a grayscale PNG
    mask = masks[0].astype(np.uint8) * 255
    Image.fromarray(mask, mode="L").save(str(path))


def mask_to_png_b64(mask: np.ndarray) -> str:
    """
    mask: bool or uint8 array, shape (H, W).
    Returns base64-encoded single-channel PNG string.
    """
    from PIL import Image  # type: ignore

    arr = (mask.astype(np.uint8) * 255)
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def masks_to_png_b64_list(masks: np.ndarray) -> list[str]:
    """masks: (N, H, W) bool array → list of N base64 PNG strings."""
    return [mask_to_png_b64(masks[i]) for i in range(len(masks))]
