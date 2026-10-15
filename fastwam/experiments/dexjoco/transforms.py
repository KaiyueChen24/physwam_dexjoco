"""Shared train/serve transforms. No simulator, Torch or model imports here."""
from __future__ import annotations

import numpy as np
from PIL import Image

PROMPT_TEMPLATE = "A video recorded from a robot's point of view executing the following instruction: {task}"


def resize_rgb(image, size):
    """RGB uint8 HWC -> CHW float32 [0,1], aspect-preserving, centered padding."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 RGB HWC image, got {image.shape}/{image.dtype}")
    height, width = map(int, size)
    if min(height, width, *image.shape[:2]) <= 0:
        raise ValueError("Image dimensions must be positive")
    ratio = min(height / image.shape[0], width / image.shape[1])
    resized = Image.fromarray(image).resize(
        (max(1, round(image.shape[1] * ratio)), max(1, round(image.shape[0] * ratio))),
        Image.Resampling.BILINEAR,
    )
    canvas = Image.new("RGB", (width, height))
    canvas.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return np.asarray(canvas).transpose(2, 0, 1).astype(np.float32) / 255.0


def pack_images(images, camera_order, size):
    return np.concatenate([resize_rgb(images[key], size) for key in camera_order], axis=-1) * 2 - 1


class MinMaxNormalizer:
    """Same min/max scale, constant-dimension handling and clamp as upstream."""
    def __init__(self, stats):
        self.parameters = {}
        for field in ("state", "action"):
            values = stats[field]["default"]
            low = np.asarray(values["global_min"], dtype=np.float32)
            high = np.asarray(values["global_max"], dtype=np.float32)
            if low.ndim != 1 or low.shape != high.shape or not np.isfinite([low, high]).all() or (low > high).any():
                raise ValueError(f"Invalid {field} statistics")
            constant = high - low < 1e-4
            scale = 2.0 / np.where(constant, 2.0, high - low)
            offset = np.where(constant, -low, -1.0 - scale * low)
            self.parameters[field] = (scale, offset)

    def _check(self, field, value):
        value = np.asarray(value, dtype=np.float32)
        if value.shape[-1] != len(self.parameters[field][0]) or not np.isfinite(value).all():
            raise ValueError(f"Invalid {field} shape or nonfinite values: {value.shape}")
        return value

    def forward(self, field, value):
        value = self._check(field, value)
        scale, offset = self.parameters[field]
        return np.clip(value * scale + offset, -5, 5).astype(np.float32)

    def backward(self, field, value):
        value = self._check(field, value)
        scale, offset = self.parameters[field]
        return ((value - offset) / scale).astype(np.float32)
