from __future__ import annotations

import numpy as np


def _check_image(image: np.ndarray) -> None:
    """
    Checks whether the input image is valid.
    Supported formats:
        Grayscale: H x W
        Color:     H x W x C
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a NumPy array.")

    if image.ndim not in (2, 3):
        raise ValueError(
            "Image must be either grayscale (H x W) "
            "or color (H x W x C)."
        )

    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("Image dimensions cannot be zero.")


def _check_scale(scale: float) -> None:
    """Checks whether scale factor is valid."""
    if scale <= 0:
        raise ValueError("Scale factor must be greater than zero.")


def _output_size(
    height: int,
    width: int,
    scale: float
) -> tuple[int, int]:
    """
    Calculates the dimensions of the resized image.
    """
    new_height = max(1, int(round(height * scale)))
    new_width = max(1, int(round(width * scale)))

    return new_height, new_width


def _source_coordinates(
    input_size: int,
    output_size: int
) -> np.ndarray:
    """
    Maps pixel centers from the output image back to
    the coordinate system of the input image.

    Inverse mapping is used:

        src = (dst + 0.5) / scale - 0.5

    Using inverse mapping prevents holes in the output image.
    """
    actual_scale = output_size / input_size

    destination_coordinates = np.arange(
        output_size,
        dtype=np.float32
    )

    source_coordinates = (
        (destination_coordinates + 0.5) / actual_scale
    ) - 0.5

    return source_coordinates


def _restore_dtype(
    image: np.ndarray,
    original_dtype: np.dtype
) -> np.ndarray:
    """
    Converts the calculated floating-point result
    back to the original image data type.
    """
    if np.issubdtype(original_dtype, np.integer):
        info = np.iinfo(original_dtype)

        image = np.clip(
            np.rint(image),
            info.min,
            info.max
        )

    return image.astype(original_dtype)


# =========================================================
# 1) NEAREST NEIGHBOR INTERPOLATION
# =========================================================

def nearest_neighbor_resize(
    image: np.ndarray,
    scale: float
) -> np.ndarray:
    """
    Resizes an image using Nearest Neighbor Interpolation.

    For every output pixel:
        1. Find its corresponding coordinate in input image.
        2. Round that coordinate to the nearest pixel.
        3. Copy the selected pixel value.
    """

    _check_image(image)
    _check_scale(scale)

    height, width = image.shape[:2]

    new_height, new_width = _output_size(
        height,
        width,
        scale
    )

    src_y = _source_coordinates(
        height,
        new_height
    )

    src_x = _source_coordinates(
        width,
        new_width
    )

    # Select nearest integer coordinates
    nearest_y = np.floor(src_y + 0.5).astype(np.int32)
    nearest_x = np.floor(src_x + 0.5).astype(np.int32)

    # Border handling
    nearest_y = np.clip(
        nearest_y,
        0,
        height - 1
    )

    nearest_x = np.clip(
        nearest_x,
        0,
        width - 1
    )

    # Advanced NumPy indexing
    output = image[
        nearest_y[:, None],
        nearest_x[None, :]
    ]

    return output.copy()


# =========================================================
# 2) BILINEAR INTERPOLATION
# =========================================================

def bilinear_resize(
    image: np.ndarray,
    scale: float
) -> np.ndarray:
    """
    Resizes an image using Bilinear Interpolation.

    Each output pixel is calculated using four neighboring
    pixels in the original image.
    """

    _check_image(image)
    _check_scale(scale)

    original_dtype = image.dtype

    # Work in floating point for weighted calculations
    source = image.astype(np.float32)

    grayscale = source.ndim == 2

    if grayscale:
        source = source[:, :, None]

    height, width, _ = source.shape

    new_height, new_width = _output_size(
        height,
        width,
        scale
    )

    src_y = _source_coordinates(
        height,
        new_height
    )

    src_x = _source_coordinates(
        width,
        new_width
    )

    # Coordinates of surrounding pixels
    y0 = np.floor(src_y).astype(np.int32)
    x0 = np.floor(src_x).astype(np.int32)

    y1 = y0 + 1
    x1 = x0 + 1

    # Fractional distances
    dy = src_y - y0
    dx = src_x - x0

    # Border handling
    y0_clip = np.clip(y0, 0, height - 1)
    y1_clip = np.clip(y1, 0, height - 1)

    x0_clip = np.clip(x0, 0, width - 1)
    x1_clip = np.clip(x1, 0, width - 1)

    # Four neighboring pixels
    top_left = source[
        y0_clip[:, None],
        x0_clip[None, :]
    ]

    top_right = source[
        y0_clip[:, None],
        x1_clip[None, :]
    ]

    bottom_left = source[
        y1_clip[:, None],
        x0_clip[None, :]
    ]

    bottom_right = source[
        y1_clip[:, None],
        x1_clip[None, :]
    ]

    # Reshape weights for broadcasting
    dx = dx[None, :, None]
    dy = dy[:, None, None]

    # Horizontal interpolation
    top = (
        top_left * (1.0 - dx)
        + top_right * dx
    )

    bottom = (
        bottom_left * (1.0 - dx)
        + bottom_right * dx
    )

    # Vertical interpolation
    output = (
        top * (1.0 - dy)
        + bottom * dy
    )

    if grayscale:
        output = output[:, :, 0]

    return _restore_dtype(
        output,
        original_dtype
    )


# =========================================================
# 3) BICUBIC INTERPOLATION
# =========================================================

def _cubic_kernel(
    distance: np.ndarray,
    a: float = -0.5
) -> np.ndarray:
    """
    Cubic convolution kernel.

    Parameter a = -0.5 is commonly used and produces
    a smooth cubic interpolation.
    """

    x = np.abs(distance).astype(np.float32)

    result = np.zeros_like(
        x,
        dtype=np.float32
    )

    # |x| <= 1
    mask1 = x <= 1.0

    result[mask1] = (
        (a + 2.0) * x[mask1] ** 3
        - (a + 3.0) * x[mask1] ** 2
        + 1.0
    )

    # 1 < |x| < 2
    mask2 = (x > 1.0) & (x < 2.0)

    result[mask2] = (
        a * x[mask2] ** 3
        - 5.0 * a * x[mask2] ** 2
        + 8.0 * a * x[mask2]
        - 4.0 * a
    )

    return result


def bicubic_resize(
    image: np.ndarray,
    scale: float,
    a: float = -0.5
) -> np.ndarray:
    """
    Resizes an image using Bicubic Interpolation.

    Bicubic interpolation uses a 4 x 4 neighborhood,
    therefore 16 neighboring pixels contribute to
    each output pixel.
    """

    _check_image(image)
    _check_scale(scale)

    original_dtype = image.dtype

    source = image.astype(np.float32)

    grayscale = source.ndim == 2

    if grayscale:
        source = source[:, :, None]

    height, width, channels = source.shape

    new_height, new_width = _output_size(
        height,
        width,
        scale
    )

    src_y = _source_coordinates(
        height,
        new_height
    )

    src_x = _source_coordinates(
        width,
        new_width
    )

    base_y = np.floor(src_y).astype(np.int32)
    base_x = np.floor(src_x).astype(np.int32)

    # Bicubic uses four neighbors in each direction:
    # -1, 0, 1, 2 relative to the base pixel
    offsets = (-1, 0, 1, 2)

    y_indices = []
    x_indices = []

    y_weights = []
    x_weights = []

    for offset in offsets:
        yi = base_y + offset
        xi = base_x + offset

        y_indices.append(
            np.clip(
                yi,
                0,
                height - 1
            )
        )

        x_indices.append(
            np.clip(
                xi,
                0,
                width - 1
            )
        )

        y_weights.append(
            _cubic_kernel(
                src_y - yi,
                a
            )
        )

        x_weights.append(
            _cubic_kernel(
                src_x - xi,
                a
            )
        )

    output = np.zeros(
        (
            new_height,
            new_width,
            channels
        ),
        dtype=np.float32
    )

    # 4 x 4 neighborhood = 16 contributions
    for row in range(4):
        for col in range(4):

            pixels = source[
                y_indices[row][:, None],
                x_indices[col][None, :]
            ]

            weight = (
                y_weights[row][:, None, None]
                * x_weights[col][None, :, None]
            )

            output += pixels * weight

    if grayscale:
        output = output[:, :, 0]

    return _restore_dtype(
        output,
        original_dtype
    )


# =========================================================
# GENERAL FUNCTION
# =========================================================

def resize_image(
    image: np.ndarray,
    scale: float,
    method: str
) -> np.ndarray:
    """
    General interface for all interpolation methods.

    method:
        nearest
        bilinear
        bicubic
    """

    method = method.lower()

    if method == "nearest":
        return nearest_neighbor_resize(
            image,
            scale
        )

    if method == "bilinear":
        return bilinear_resize(
            image,
            scale
        )

    if method == "bicubic":
        return bicubic_resize(
            image,
            scale
        )

    raise ValueError(
        "Unknown method. "
        "Use 'nearest', 'bilinear', or 'bicubic'."
    )