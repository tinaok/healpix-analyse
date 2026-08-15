"""Prototype face-native separable Gaussian filtering for NESTED HEALPix."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np
from scipy.ndimage import convolve1d, maximum_filter

from ._face_topology import (
    clear_topology_cache,
    pix2xyf,
    resolve_face_grid,
)
from ._neighbourhood import _wgs84_distance
from .radial_filter import (
    _validate_cell_ids,
    _validate_domain,
    _validate_positive_finite,
    _validate_values,
    gaussian_filter,
)


@dataclass(frozen=True)
class FaceNativeGaussianStats:
    """Diagnostics and timing for one face-native filter call."""

    dx_m: float
    dy_m: float
    sigma_x_px: float
    sigma_y_px: float
    radius_x: int
    radius_y: int
    face_tiles: int
    topology_halo_cells: int
    fallback_halo_cells: int
    fallback_output_cells: int
    setup_seconds: float
    filtering_seconds: float
    exact_fallback_seconds: float

    @property
    def fallback_halo_fraction(self) -> float:
        """Fraction of cross-face halo lookups routed to fallback."""
        if self.topology_halo_cells == 0:
            return 0.0
        return self.fallback_halo_cells / self.topology_halo_cells

    @property
    def total_seconds(self) -> float:
        return self.setup_seconds + self.filtering_seconds + self.exact_fallback_seconds


def _kernel(sigma_px: float, radius: int) -> np.ndarray:
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    weights = np.exp(-0.5 * (offsets / sigma_px) ** 2)
    return weights / np.sum(weights)


def _separable(
    array: np.ndarray,
    kernel_x: np.ndarray,
    kernel_y: np.ndarray,
    pass_order: Literal["xy", "yx"],
) -> np.ndarray:
    if pass_order == "xy":
        result = convolve1d(array, kernel_x, axis=-1, mode="constant", cval=0.0)
        return convolve1d(result, kernel_y, axis=-2, mode="constant", cval=0.0)
    result = convolve1d(array, kernel_y, axis=-2, mode="constant", cval=0.0)
    return convolve1d(result, kernel_x, axis=-1, mode="constant", cval=0.0)


def _calibrate(
    representative_cell: int,
    refinement_level: int,
) -> tuple[float, float]:
    face, x, y = pix2xyf(
        np.asarray([representative_cell], dtype=np.uint64),
        refinement_level,
    )
    face_i = int(face[0])
    x_i = int(x[0])
    y_i = int(y[0])
    plus_x = resolve_face_grid(face_i, x_i + 1, y_i, refinement_level)
    plus_y = resolve_face_grid(face_i, x_i, y_i + 1, refinement_level)
    if not bool(plus_x.valid_mask) or not bool(plus_y.valid_mask):
        raise RuntimeError("Could not resolve local +x/+y calibration cells.")
    distances = _wgs84_distance(
        np.asarray([representative_cell, representative_cell], dtype=np.uint64),
        np.asarray([plus_x.cell_ids.item(), plus_y.cell_ids.item()], dtype=np.uint64),
        refinement_level,
    )
    if np.any(~np.isfinite(distances)) or np.any(distances <= 0.0):
        raise RuntimeError("Invalid WGS84 face metric calibration.")
    return float(distances[0]), float(distances[1])


def _clear_filter_caches() -> None:
    """Clear face-native reusable data for cold benchmark runs."""
    clear_topology_cache()


def face_native_gaussian_filter(
    values: np.ndarray,
    cell_ids: np.ndarray,
    refinement_level: int,
    *,
    sigma_m: float,
    truncate: float = 4.0,
    domain: np.ndarray | None = None,
    pass_order: Literal["xy", "yx"] = "xy",
    return_stats: bool = False,
) -> np.ndarray | tuple[np.ndarray, FaceNativeGaussianStats]:
    """Apply an approximate Gaussian on face-local NESTED grids.

    A single WGS84 ``dx``/``dy`` calibration is used over the patch.  Face
    interiors and one-edge halo crossings use integer topology.  Outputs whose
    kernel support touches a corner or multi-edge halo location are replaced
    by the exact :func:`gaussian_filter` result.

    Missing values and cells outside ``domain`` are excluded through normalized
    convolution; they are never interpreted as zero-valued observations.
    This first prototype accepts NumPy arrays and supports arbitrary leading
    batch dimensions.
    """
    if not isinstance(values, np.ndarray):
        raise TypeError("'values' must be a NumPy array for this prototype.")
    cells = _validate_cell_ids(cell_ids, name="cell_ids")
    _validate_values(values, cells.size)
    output_domain = _validate_domain(cells, domain)
    sigma = _validate_positive_finite(sigma_m, name="sigma_m")
    truncation = _validate_positive_finite(truncate, name="truncate")
    if pass_order not in ("xy", "yx"):
        raise ValueError("'pass_order' must be either 'xy' or 'yx'.")
    if not isinstance(refinement_level, (int, np.integer)) or isinstance(
        refinement_level, (bool, np.bool_)
    ):
        raise TypeError("'refinement_level' must be an integer.")
    refinement_level = int(refinement_level)
    if not 0 <= refinement_level <= 29:
        raise ValueError("'refinement_level' must be in [0, 29].")

    output_shape = values.shape[:-1] + (output_domain.size,)
    if output_domain.size == 0:
        empty = np.empty(output_shape, dtype=np.result_type(values.dtype, np.float64))
        stats = FaceNativeGaussianStats(
            np.nan, np.nan, np.nan, np.nan, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0
        )
        return (empty, stats) if return_stats else empty

    setup_start = perf_counter()
    representative = int(output_domain[output_domain.size // 2])
    dx_m, dy_m = _calibrate(representative, refinement_level)
    sigma_x_px = sigma / dx_m
    sigma_y_px = sigma / dy_m
    radius_x = int(truncation * sigma_x_px + 0.5)
    radius_y = int(truncation * sigma_y_px + 0.5)
    kernel_x = _kernel(sigma_x_px, radius_x)
    kernel_y = _kernel(sigma_y_px, radius_y)

    faces, xs, ys = pix2xyf(output_domain, refinement_level)
    input_positions = {int(cell): index for index, cell in enumerate(cells)}
    domain_set = {int(cell) for cell in output_domain}
    flat_values = np.asarray(values).reshape((-1, cells.size))
    result = np.full(
        (flat_values.shape[0], output_domain.size),
        np.nan,
        dtype=np.result_type(values.dtype, np.float64),
    )
    tiles = []
    topology_halo_cells = 0
    fallback_halo_cells = 0
    fallback_outputs = np.zeros(output_domain.size, dtype=bool)

    for face in np.unique(faces):
        output_indices = np.flatnonzero(faces == face)
        face_x = xs[output_indices].astype(np.int64)
        face_y = ys[output_indices].astype(np.int64)
        x0, x1 = int(face_x.min()), int(face_x.max())
        y0, y1 = int(face_y.min()), int(face_y.max())
        grid_x = np.arange(x0 - radius_x, x1 + radius_x + 1, dtype=np.int64)
        grid_y = np.arange(y0 - radius_y, y1 + radius_y + 1, dtype=np.int64)
        xx, yy = np.meshgrid(grid_x, grid_y)
        resolved = resolve_face_grid(int(face), xx, yy, refinement_level)
        topology_mask = resolved.single_edge_mask | resolved.fallback_mask
        topology_halo_cells += int(np.count_nonzero(topology_mask))
        fallback_halo_cells += int(np.count_nonzero(resolved.fallback_mask))

        source_positions = np.full(xx.shape, -1, dtype=np.int64)
        for row, column in zip(*np.nonzero(resolved.valid_mask), strict=True):
            cell = int(resolved.cell_ids[row, column])
            if cell in domain_set:
                source_positions[row, column] = input_positions[cell]

        corner_support = maximum_filter(
            resolved.fallback_mask.astype(np.uint8),
            size=(2 * radius_y + 1, 2 * radius_x + 1),
            mode="constant",
            cval=0,
        ).astype(bool)
        local_rows = face_y - (y0 - radius_y)
        local_columns = face_x - (x0 - radius_x)
        fallback_outputs[output_indices] = corner_support[local_rows, local_columns]
        tiles.append((output_indices, local_rows, local_columns, source_positions))

    setup_seconds = perf_counter() - setup_start
    filtering_start = perf_counter()
    for output_indices, local_rows, local_columns, source_positions in tiles:
        tile_shape = (flat_values.shape[0],) + source_positions.shape
        numerator = np.zeros(tile_shape, dtype=result.dtype)
        denominator = np.zeros(tile_shape, dtype=np.float64)
        available = source_positions >= 0
        if np.any(available):
            gathered = flat_values[:, source_positions[available]]
            finite = np.isfinite(gathered)
            numerator[:, available] = np.where(finite, gathered, 0)
            denominator[:, available] = finite
        numerator = _separable(numerator, kernel_x, kernel_y, pass_order)
        denominator = _separable(denominator, kernel_x, kernel_y, pass_order)
        selected_numerator = numerator[:, local_rows, local_columns]
        selected_denominator = denominator[:, local_rows, local_columns]
        filtered = np.full_like(selected_numerator, np.nan)
        np.divide(
            selected_numerator,
            selected_denominator,
            out=filtered,
            where=selected_denominator > 0.0,
        )
        result[:, output_indices] = filtered
    filtering_seconds = perf_counter() - filtering_start

    fallback_seconds = 0.0
    if np.any(fallback_outputs):
        fallback_start = perf_counter()
        exact = gaussian_filter(
            values,
            cells,
            refinement_level,
            sigma_m=sigma,
            truncate=truncation,
            domain=output_domain,
        )
        exact_flat = np.asarray(exact).reshape((-1, output_domain.size))
        result[:, fallback_outputs] = exact_flat[:, fallback_outputs]
        fallback_seconds = perf_counter() - fallback_start

    stats = FaceNativeGaussianStats(
        dx_m=dx_m,
        dy_m=dy_m,
        sigma_x_px=sigma_x_px,
        sigma_y_px=sigma_y_px,
        radius_x=radius_x,
        radius_y=radius_y,
        face_tiles=len(tiles),
        topology_halo_cells=topology_halo_cells,
        fallback_halo_cells=fallback_halo_cells,
        fallback_output_cells=int(np.count_nonzero(fallback_outputs)),
        setup_seconds=setup_seconds,
        filtering_seconds=filtering_seconds,
        exact_fallback_seconds=fallback_seconds,
    )
    shaped = result.reshape(output_shape)
    return (shaped, stats) if return_stats else shaped


__all__ = [
    "FaceNativeGaussianStats",
    "_clear_filter_caches",
    "face_native_gaussian_filter",
]
