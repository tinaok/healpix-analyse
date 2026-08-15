"""Tests for the face-native separable Gaussian prototype."""

from __future__ import annotations

import numpy as np
import pytest

from healpix_analyse._face_topology import resolve_face_grid, xyf2pix
from healpix_analyse._neighbourhood import _wgs84_distance
from healpix_analyse.face_gaussian_filter import face_native_gaussian_filter
from healpix_analyse.radial_filter import gaussian_filter


def _spacing(cell: int, level: int) -> float:
    from healpix_analyse.face_gaussian_filter import _calibrate

    dx_m, dy_m = _calibrate(cell, level)
    return float(np.sqrt(dx_m * dy_m))


def _rectangle(face: int, x0: int, x1: int, y0: int, y1: int, level: int):
    x, y = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
    return xyf2pix(face, x.ravel(), y.ravel(), level)


def test_wgs84_distance_helper_is_vectorised_and_zero_on_self():
    level = 3
    cells = xyf2pix(4, [2, 3], [2, 2], level)
    distance = _wgs84_distance(cells, cells[::-1], level)
    assert distance.shape == (2,)
    assert np.all(distance > 0.0)
    np.testing.assert_array_equal(_wgs84_distance(cells, cells, level), 0.0)


def test_topology_adapter_resolves_same_face_and_single_edge():
    level = 3
    nside = 1 << level
    resolved = resolve_face_grid(
        4,
        np.array([2, nside]),
        np.array([3, 3]),
        level,
    )
    assert resolved.valid_mask.tolist() == [True, True]
    assert resolved.single_edge_mask.tolist() == [False, True]
    assert resolved.fallback_mask.tolist() == [False, False]
    assert int(resolved.cell_ids[0]) == int(xyf2pix(4, 2, 3, level)[0])


def test_topology_adapter_marks_corner_without_composing_transforms():
    resolved = resolve_face_grid(4, -1, -1, 3)
    assert not bool(resolved.valid_mask)
    assert bool(resolved.fallback_mask)


@pytest.mark.parametrize(
    ("source_x", "source_y", "halo_x", "halo_y", "healpy_direction"),
    [
        (0, 3, -1, 3, 0),  # -x / SW
        (7, 3, 8, 3, 4),  # +x / NE
        (3, 0, 3, -1, 6),  # -y / SE
        (3, 7, 3, 8, 2),  # +y / NW
    ],
)
def test_single_edge_mapping_matches_healpy_neighbour_direction(
    source_x, source_y, halo_x, halo_y, healpy_direction
):
    healpy = pytest.importorskip("healpy")
    level = 3
    nside = 1 << level
    for face in range(12):
        source = int(xyf2pix(face, source_x, source_y, level)[0])
        expected = healpy.get_all_neighbours(nside, source, nest=True)[healpy_direction]
        resolved = resolve_face_grid(face, halo_x, halo_y, level)
        assert bool(resolved.valid_mask)
        assert int(resolved.cell_ids.item()) == int(expected)


def test_constant_field_preserved_on_partial_interior_domain():
    level = 5
    cells = _rectangle(4, 8, 17, 9, 16, level)
    values = np.full(cells.size, 7.25)
    result, stats = face_native_gaussian_filter(
        values,
        cells,
        level,
        sigma_m=1.3 * _spacing(int(cells[cells.size // 2]), level),
        return_stats=True,
    )
    np.testing.assert_allclose(result, 7.25, rtol=0.0, atol=1e-12)
    assert stats.fallback_halo_cells == 0
    assert stats.fallback_output_cells == 0


def test_missing_samples_use_normalized_convolution():
    level = 5
    cells = _rectangle(4, 8, 17, 9, 16, level)
    values = np.full(cells.size, 3.5)
    values[::5] = np.nan
    result = face_native_gaussian_filter(
        values,
        cells,
        level,
        sigma_m=_spacing(int(cells[cells.size // 2]), level),
    )
    np.testing.assert_allclose(result, 3.5, rtol=0.0, atol=1e-12)


def test_domain_is_both_processing_and_output_domain():
    level = 5
    cells = _rectangle(4, 8, 18, 8, 18, level)
    domain = cells.reshape(10, 10)[2:8, 2:8].ravel()[::-1]
    values = np.full(cells.size, 2.0)
    values[np.isin(cells, domain, invert=True)] = 1_000.0
    result = face_native_gaussian_filter(
        values,
        cells,
        level,
        domain=domain,
        sigma_m=_spacing(int(domain[domain.size // 2]), level),
    )
    assert result.shape == domain.shape
    np.testing.assert_allclose(result, 2.0, rtol=0.0, atol=1e-12)


def test_xy_and_yx_passes_are_equivalent():
    level = 5
    cells = _rectangle(4, 8, 18, 7, 16, level)
    values = np.random.default_rng(40).normal(size=(2, cells.size))
    sigma_m = 1.4 * _spacing(int(cells[cells.size // 2]), level)
    xy = face_native_gaussian_filter(
        values, cells, level, sigma_m=sigma_m, pass_order="xy"
    )
    yx = face_native_gaussian_filter(
        values, cells, level, sigma_m=sigma_m, pass_order="yx"
    )
    np.testing.assert_allclose(xy, yx, rtol=2e-15, atol=2e-15)


@pytest.mark.parametrize("sigma_px", [3.0, 5.0], ids=["S2-G04", "S2-G05"])
def test_level20_s2_fixed_sigma_matches_exact_reference(sigma_px):
    """Exercise application-derived S2 scales at the S2MSI HEALPix level."""
    level = 20
    nside = 1 << level
    size = 16
    start = (nside - size) // 2
    cells = _rectangle(
        4,
        start,
        start + size,
        start,
        start + size,
        level,
    )
    yy, xx = np.indices((size, size))
    values = (
        np.sin(xx / (size / 5.0)) + 0.6 * np.cos(yy / (size / 7.0)) + 0.02 * xx
    ).ravel()
    sigma_m = sigma_px * _spacing(int(cells[cells.size // 2]), level)

    face, stats = face_native_gaussian_filter(
        values,
        cells,
        level,
        sigma_m=sigma_m,
        return_stats=True,
    )
    reverse = face_native_gaussian_filter(
        values,
        cells,
        level,
        sigma_m=sigma_m,
        pass_order="yx",
    )
    exact = gaussian_filter(
        values,
        cells,
        level,
        sigma_m=sigma_m,
    )

    np.testing.assert_allclose(face, reverse, rtol=2e-15, atol=2e-15)
    assert stats.fallback_output_cells == 0
    assert np.sqrt(np.mean((face - exact) ** 2)) < 0.05


def test_single_edge_patch_uses_transform_without_fallback():
    level = 5
    nside = 1 << level
    x, y = np.meshgrid(np.arange(nside - 4, nside + 4), np.arange(10, 18))
    resolved = resolve_face_grid(4, x.ravel(), y.ravel(), level)
    assert np.all(resolved.valid_mask)
    cells = resolved.cell_ids
    values = np.ones(cells.size)
    result, stats = face_native_gaussian_filter(
        values,
        cells,
        level,
        sigma_m=0.7 * _spacing(int(cells[cells.size // 2]), level),
        return_stats=True,
    )
    np.testing.assert_allclose(result, 1.0, rtol=0.0, atol=1e-12)
    assert stats.topology_halo_cells > 0
    assert stats.fallback_halo_cells == 0


def test_corner_outputs_are_replaced_by_exact_reference():
    level = 2
    cells = np.arange(12 * 4**level, dtype=np.uint64)
    values = np.random.default_rng(244).normal(size=cells.size)
    sigma_m = 0.55 * _spacing(0, level)
    actual, stats = face_native_gaussian_filter(
        values,
        cells,
        level,
        sigma_m=sigma_m,
        truncate=2.0,
        return_stats=True,
    )
    expected = gaussian_filter(
        values,
        cells,
        level,
        sigma_m=sigma_m,
        truncate=2.0,
    )
    assert stats.fallback_halo_cells > 0
    assert stats.fallback_output_cells > 0
    corner_cells = xyf2pix(
        np.repeat(np.arange(12), 4),
        np.tile([0, 0, 3, 3], 12),
        np.tile([0, 3, 0, 3], 12),
        level,
    )
    np.testing.assert_allclose(
        actual[corner_cells],
        expected[corner_cells],
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("pass_order", ["x", "", None])
def test_invalid_pass_order_rejected(pass_order):
    with pytest.raises(ValueError, match="pass_order"):
        face_native_gaussian_filter(
            np.array([1.0]),
            np.array([0], dtype=np.uint64),
            1,
            sigma_m=1.0,
            pass_order=pass_order,
        )
