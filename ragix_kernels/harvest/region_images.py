"""Bounded, hash-checked base64 raster payloads; no HTML and no image inference.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

import base64
import binascii
import hashlib
import io
import re
import warnings
from .region_types import FigureImage, RegionLimits, RegionRefused

MIME = {"image/png": "PNG", "image/jpeg": "JPEG"}


def image_payload(
    data,
    media_type,
    *,
    source_asset=None,
    conversion_rule="source-raster/1",
    limits=RegionLimits(),
    renderer=None,
    renderer_version=None,
    dpi=None,
):
    if media_type not in MIME:
        raise RegionRefused("UNSUPPORTED_FIGURE_MEDIA_TYPE")
    if not isinstance(data, bytes) or not data or len(data) > limits.max_image_bytes:
        raise RegionRefused("FIGURE_BYTE_LIMIT")
    try:
        from PIL import Image
    except ImportError as error:
        raise RegionRefused("FIGURE_VALIDATOR_UNAVAILABLE") from error
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data), formats=[MIME[media_type]]) as picture:
                width, height = picture.size
                if width * height > limits.max_image_pixels:
                    raise RegionRefused("FIGURE_PIXEL_LIMIT")
                if getattr(picture, "n_frames", 1) != 1:
                    raise RegionRefused("ANIMATED_FIGURE_UNSUPPORTED")
                picture.verify()
            with Image.open(io.BytesIO(data), formats=[MIME[media_type]]) as picture:
                picture.load()
    except RegionRefused:
        raise
    except (
        OSError,
        ValueError,
        SyntaxError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as error:
        raise RegionRefused("INVALID_FIGURE_RASTER") from error
    digest = hashlib.sha256(data).hexdigest()
    if source_asset is None:
        source_asset = digest
    if not re.fullmatch(r"[0-9a-f]{64}", source_asset):
        raise RegionRefused("INVALID_FIGURE_ASSET_ID")
    return FigureImage(
        "base64",
        media_type,
        base64.b64encode(data).decode("ascii"),
        digest,
        len(data),
        width,
        height,
        source_asset,
        conversion_rule,
        renderer,
        renderer_version,
        dpi,
    )


def validate_image(image, limits):
    if image.encoding != "base64" or image.media_type not in MIME:
        raise RegionRefused("UNSUPPORTED_FIGURE_ENCODING")
    if len(image.data) > 4 * ((limits.max_image_bytes + 2) // 3):
        raise RegionRefused("FIGURE_BYTE_LIMIT")
    try:
        data = base64.b64decode(image.data, validate=True)
    except (ValueError, binascii.Error) as error:
        raise RegionRefused("INVALID_FIGURE_BASE64") from error
    expected = image_payload(
        data,
        image.media_type,
        source_asset=image.source_asset,
        conversion_rule=image.conversion_rule,
        limits=limits,
        renderer=image.renderer,
        renderer_version=image.renderer_version,
        dpi=image.dpi,
    )
    if expected != image:
        raise RegionRefused("FIGURE_PAYLOAD_MISMATCH")


def image_from_store(store, asset, *, raster_asset=None, limits=RegionLimits()):
    """Reuse an existing raster, or one unambiguous derived raster in its manifest.

    A vector description or raw PDF pixels are not labelled as PNG. If no raster
    is stored, callers may use the existing renderer port separately and supply
    its checked raster. No renderer is imported or invoked by this envelope.
    """
    if not re.fullmatch(r"[0-9a-f]{64}", asset):
        raise RegionRefused("INVALID_FIGURE_ASSET_ID")
    manifest = store.manifest()
    if asset not in manifest:
        raise RegionRefused("FIGURE_ASSET_MISSING")
    try:
        if (store.root / asset).stat().st_size > limits.max_source_asset_bytes:
            raise RegionRefused("FIGURE_SOURCE_ASSET_LIMIT")
        original = store.read(asset)
        if len(original) > limits.max_source_asset_bytes:
            raise RegionRefused("FIGURE_SOURCE_ASSET_LIMIT")
    except (OSError, KeyError) as error:
        raise RegionRefused("FIGURE_ASSET_MISSING_OR_CHANGED") from error
    choices = []
    if manifest[asset].get("media_type") in MIME:
        choices = [asset]
    else:
        choices = [
            key
            for key, value in manifest.items()
            if value.get("media_type") in MIME
            and any(ref.get("derived_from") == asset for ref in value.get("references", ()))
        ]
    if raster_asset is not None:
        if raster_asset not in choices:
            raise RegionRefused("RASTER_NOT_DERIVED_FROM_FIGURE")
        chosen = raster_asset
    elif len(choices) == 1:
        chosen = choices[0]
    else:
        raise RegionRefused("FIGURE_RASTER_AMBIGUOUS" if choices else "FIGURE_RASTER_UNAVAILABLE")
    if not re.fullmatch(r"[0-9a-f]{64}", chosen):
        raise RegionRefused("INVALID_FIGURE_ASSET_ID")
    try:
        if (store.root / chosen).stat().st_size > limits.max_image_bytes:
            raise RegionRefused("FIGURE_BYTE_LIMIT")
        data = original if chosen == asset else store.read(chosen)
    except (OSError, KeyError) as error:
        raise RegionRefused("FIGURE_ASSET_MISSING_OR_CHANGED") from error
    return image_payload(
        data,
        manifest[chosen]["media_type"],
        source_asset=asset,
        conversion_rule="source-raster/1" if chosen == asset else "stored-derived-raster/1",
        limits=limits,
    )
