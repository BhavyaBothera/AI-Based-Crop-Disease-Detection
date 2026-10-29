"""
Deterministic synthetic image fixtures shared by the backend test suites.

Solid-colour canvases are rejected by the Level A blank-canvas guard in
``backend.app.services.inference.preprocess_image`` (luminance sigma <= 2.0
yields HTTP 422). Fixtures produced here therefore contain macro-scale spatial
variation - a horizontal luminance ramp plus a low-frequency undulation - so
they exercise the decoding, resizing, and inference paths these tests target
instead of being short-circuited by the uniformity check.

A fine high-frequency component is also included. Without it, a pure smooth
gradient has almost no Laplacian energy and the content-level input gate
treats it as unassessable low-detail imagery (real validation leaves never fall
below a Laplacian variance of 2e-4). Adding micro-texture keeps these fixtures
representative of a framed leaf surface rather than a flat synthetic wash.

Texture is generated at no larger than 512x512 and upscaled to the requested
size, keeping memory bounded for large-image tests such as the 25 MP fixture.
"""
import io

import numpy as np
from PIL import Image

_TEXTURE_MAX_DIMENSION = 512


def create_textured_image_bytes(
    format: str = "JPEG",
    size: tuple = (100, 100),
    color: tuple = (34, 139, 34),
    mode: str = "RGB",
) -> bytes:
    """
    Creates in-memory synthetic image bytes with spatial variation.

    Args:
        format: Pillow format name (JPEG, PNG, WEBP, GIF).
        size: (width, height) of the produced image.
        color: Base RGB(A) colour. The first three values seed the channels;
            the optional fourth value seeds the alpha channel.
        mode: PIL mode, "RGB" or "RGBA".

    Returns:
        Encoded image bytes.
    """
    width, height = size
    tex_w = min(width, _TEXTURE_MAX_DIMENSION)
    tex_h = min(height, _TEXTURE_MAX_DIMENSION)

    # Macro-scale horizontal luminance ramp (+/- 60 levels)
    ramp = np.linspace(-60.0, 60.0, tex_w, dtype=np.float32)[None, :]
    # Low-frequency vertical undulation so the pattern is not a pure 1-D gradient
    undulation = 30.0 * np.cos(np.linspace(0.0, 3.0 * np.pi, tex_h, dtype=np.float32))[:, None]

    # Fine venation-like micro-texture (gives the surface real Laplacian energy)
    xs = np.arange(tex_w, dtype=np.float32)[None, :]
    ys = np.arange(tex_h, dtype=np.float32)[:, None]
    micro = 12.0 * np.sin(xs * 2.1) * np.cos(ys * 1.9)
    micro += 6.0 * np.sign(np.sin((xs + ys) * 2.4))

    base_rgb = np.array(color[:3], dtype=np.float32)
    planes = [
        np.clip(
            base_rgb[channel] + ramp + undulation * (1.0 if channel % 2 == 0 else -1.0) + micro,
            0,
            255,
        )
        for channel in range(3)
    ]
    arr = np.stack(planes, axis=-1).astype(np.uint8)

    if mode == "RGBA":
        alpha_value = color[3] if len(color) > 3 else 255
        alpha = np.full((tex_h, tex_w, 1), alpha_value, dtype=np.uint8)
        arr = np.concatenate([arr, alpha], axis=-1)

    img = Image.fromarray(arr, mode=mode)
    if (tex_w, tex_h) != (width, height):
        img = img.resize((width, height), Image.Resampling.BILINEAR)

    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()
