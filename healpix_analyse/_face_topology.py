"""Thin adapter for the NESTED face topology API in healpix-geo PR #244."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from healpix_geo import nested


@dataclass(frozen=True)
class FaceGridResolution:
    """Resolved cell IDs and routing information for face-grid locations."""

    cell_ids: np.ndarray
    valid_mask: np.ndarray
    single_edge_mask: np.ndarray
    fallback_mask: np.ndarray


def _require_face_topology_api() -> None:
    missing = [
        name
        for name in ("pix2xyf", "xyf2pix", "face_neighbour_transform")
        if not hasattr(nested, name)
    ]
    if missing:
        raise ImportError(
            "face_native_gaussian_filter requires the NESTED face topology "
            "API introduced by GRID4EARTH/healpix-geo PR #244; missing: "
            + ", ".join(missing)
        )


def pix2xyf(cell_ids: np.ndarray, refinement_level: int):
    """Adapt healpix-geo's public topology API to local terminology."""
    _require_face_topology_api()
    return nested.pix2xyf(cell_ids, refinement_level)


def xyf2pix(face, x, y, refinement_level: int) -> np.ndarray:
    """Adapt healpix-geo's public topology API to local terminology."""
    _require_face_topology_api()
    return np.asarray(
        nested.xyf2pix(face, x, y, refinement_level),
        dtype=np.uint64,
    )


@lru_cache(maxsize=96)
def _face_transform(face: int, direction: str):
    _require_face_topology_api()
    return nested.face_neighbour_transform(face, direction)


def clear_topology_cache() -> None:
    """Clear reusable face-orientation lookups for cold benchmarks."""
    _face_transform.cache_clear()


def _apply_transform(x, y, nside: int, transform):
    if transform.swap_xy:
        x, y = y, x
    if transform.flip_x:
        x = nside - 1 - x
    if transform.flip_y:
        y = nside - 1 - y
    return x, y


def resolve_face_grid(
    face: int,
    x,
    y,
    refinement_level: int,
) -> FaceGridResolution:
    """Resolve same-face and one-edge face-local coordinates.

    Locations outside two coordinate bounds, or farther than one face away,
    are deliberately marked for the caller's correctness-first fallback.
    No face transforms are composed here.
    """
    _require_face_topology_api()
    x, y = np.broadcast_arrays(
        np.atleast_1d(np.asarray(x, dtype=np.int64)),
        np.atleast_1d(np.asarray(y, dtype=np.int64)),
    )
    nside = 1 << refinement_level
    x_low = x < 0
    x_high = x >= nside
    y_low = y < 0
    y_high = y >= nside
    crossings = (
        x_low.astype(np.uint8)
        + x_high.astype(np.uint8)
        + y_low.astype(np.uint8)
        + y_high.astype(np.uint8)
    )
    too_far = (x < -nside) | (x >= 2 * nside) | (y < -nside) | (y >= 2 * nside)
    fallback = (crossings > 1) | too_far
    same_face = crossings == 0
    single_edge = (crossings == 1) & ~too_far

    result = np.zeros(x.shape, dtype=np.uint64)
    valid = same_face.copy()
    if np.any(same_face):
        result[same_face] = xyf2pix(
            face,
            x[same_face],
            y[same_face],
            refinement_level,
        )

    directions = (
        (x_low, "SW"),
        (x_high, "NE"),
        (y_low, "SE"),
        (y_high, "NW"),
    )
    for direction_mask, direction in directions:
        selected = single_edge & direction_mask
        if not np.any(selected):
            continue
        transform = _face_transform(int(face), direction)
        if transform is None:
            fallback[selected] = True
            single_edge[selected] = False
            continue
        local_x = np.mod(x[selected], nside)
        local_y = np.mod(y[selected], nside)
        target_x, target_y = _apply_transform(
            local_x,
            local_y,
            nside,
            transform,
        )
        result[selected] = xyf2pix(
            transform.target_face,
            target_x,
            target_y,
            refinement_level,
        )
        valid[selected] = True

    return FaceGridResolution(
        cell_ids=result,
        valid_mask=valid,
        single_edge_mask=single_edge,
        fallback_mask=fallback,
    )


__all__ = [
    "FaceGridResolution",
    "clear_topology_cache",
    "pix2xyf",
    "resolve_face_grid",
    "xyf2pix",
]
