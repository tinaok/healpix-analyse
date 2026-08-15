"""Benchmark exact, Cartesian, and face-native Gaussian implementations."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import tracemalloc
from dataclasses import asdict

import numpy as np
from scipy.ndimage import gaussian_filter as scipy_gaussian_filter

from healpix_analyse._face_topology import resolve_face_grid, xyf2pix
from healpix_analyse.face_gaussian_filter import (
    _calibrate,
    _clear_filter_caches,
    face_native_gaussian_filter,
)
from healpix_analyse.radial_filter import gaussian_filter

APPLICATION_CASES = {"S2-G04": 3.0, "S2-G05": 5.0}


def _runtime(function, *, warmups, repeats):
    """Measure runtime without memory tracing instrumentation."""
    if warmups < 0:
        raise ValueError("warmups must be non-negative")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    for _ in range(warmups):
        function()
    samples = []
    value = None
    for _ in range(repeats):
        start = time.perf_counter()
        value = function()
        samples.append(time.perf_counter() - start)
    return value, {
        "median": statistics.median(samples),
        "minimum": min(samples),
        "maximum": max(samples),
        "repeats": repeats,
    }


def _peak_memory(function):
    """Measure Python-tracked peak memory in a separate invocation."""
    tracemalloc.start()
    value = function()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return value, peak


def _normalized_cartesian(values, valid, sigma_px, truncate):
    numerator = scipy_gaussian_filter(
        np.where(valid, values, 0.0), sigma_px, truncate=truncate, mode="constant"
    )
    denominator = scipy_gaussian_filter(
        valid.astype(float), sigma_px, truncate=truncate, mode="constant"
    )
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0.0,
    )


def _patch(face, level, size, location):
    nside = 1 << level
    if size > nside:
        raise ValueError("size cannot exceed one face width")
    if location == "interior":
        start_x = max(0, (nside - size) // 2)
        start_y = max(0, (nside - size) // 2)
    elif location == "edge":
        start_x = nside - size // 2
        start_y = max(0, (nside - size) // 2)
    elif location == "corner":
        # A complete base face exercises corner fallback while keeping a
        # well-defined rectangular Cartesian comparator.
        start_x = start_y = 0
        size = nside
    else:
        raise ValueError(location)
    x, y = np.meshgrid(
        np.arange(start_x, start_x + size),
        np.arange(start_y, start_y + size),
    )
    resolved = resolve_face_grid(face, x, y, level)
    if not np.all(resolved.valid_mask):
        if location != "corner":
            raise RuntimeError("benchmark patch requires an unsupported transform")
        cells = xyf2pix(face, x.ravel(), y.ravel(), level)
    else:
        cells = resolved.cell_ids.ravel()
    return cells, (size, size)


def benchmark_case(
    *,
    name,
    level,
    face,
    size,
    location,
    sigma_px,
    truncate,
    warmups=1,
    repeats=3,
):
    cells, shape = _patch(face, level, size, location)
    yy, xx = np.indices(shape)
    values_2d = (
        np.sin(xx / max(1.0, size / 5.0))
        + 0.6 * np.cos(yy / max(1.0, size / 7.0))
        + 0.02 * xx
    )
    values = values_2d.ravel()
    valid = np.isfinite(values_2d)
    dx_m, dy_m = _calibrate(int(cells[cells.size // 2]), level)
    sigma_m = sigma_px * np.sqrt(dx_m * dy_m)

    exact_call = lambda: gaussian_filter(
        values,
        cells,
        level,
        sigma_m=sigma_m,
        truncate=truncate,
    )
    cartesian_call = lambda: _normalized_cartesian(values_2d, valid, sigma_px, truncate)
    face_call = lambda: face_native_gaussian_filter(
        values,
        cells,
        level,
        sigma_m=sigma_m,
        truncate=truncate,
        return_stats=True,
    )

    exact, exact_runtime = _runtime(
        exact_call,
        warmups=warmups,
        repeats=repeats,
    )
    cartesian, cart_runtime = _runtime(
        cartesian_call,
        warmups=warmups,
        repeats=repeats,
    )

    def cold_face_call():
        _clear_filter_caches()
        return face_call()

    (face_cold, cold_stats), cold_runtime = _runtime(
        cold_face_call,
        warmups=0,
        repeats=repeats,
    )
    (face_cached, cached_stats), cached_runtime = _runtime(
        face_call,
        warmups=warmups,
        repeats=repeats,
    )
    np.testing.assert_allclose(face_cold, face_cached, rtol=0.0, atol=0.0)

    _, exact_peak = _peak_memory(exact_call)
    _, cart_peak = _peak_memory(cartesian_call)
    _, cold_peak = _peak_memory(cold_face_call)
    _, cached_peak = _peak_memory(face_call)

    difference_exact = face_cached - exact
    difference_cartesian = face_cached - cartesian.ravel()
    return {
        "case": name,
        "level": level,
        "face": face,
        "location": location,
        "shape": list(shape),
        "cells": int(cells.size),
        "sigma_px": sigma_px,
        "sigma_m": sigma_m,
        "truncate": truncate,
        "runtime_seconds": {
            "exact": exact_runtime,
            "cartesian": cart_runtime,
            "face_cold": cold_runtime,
            "face_cached": cached_runtime,
        },
        "peak_memory_bytes": {
            "exact": exact_peak,
            "cartesian": cart_peak,
            "face_cold": cold_peak,
            "face_cached": cached_peak,
        },
        "face_stats_cold": asdict(cold_stats)
        | {"fallback_halo_fraction": cold_stats.fallback_halo_fraction},
        "face_stats_cached": asdict(cached_stats)
        | {"fallback_halo_fraction": cached_stats.fallback_halo_fraction},
        "error_vs_exact": {
            "rmse": float(np.sqrt(np.nanmean(difference_exact**2))),
            "max_abs": float(np.nanmax(np.abs(difference_exact))),
        },
        "error_vs_cartesian": {
            "rmse": float(np.sqrt(np.nanmean(difference_cartesian**2))),
            "max_abs": float(np.nanmax(np.abs(difference_cartesian))),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=20)
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--face", type=int, default=4)
    parser.add_argument(
        "--location", choices=("interior", "edge", "corner"), default="interior"
    )
    parser.add_argument("--sigma-px", type=float, default=3.0)
    parser.add_argument("--truncate", type=float, default=4.0)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--suite",
        action="store_true",
        help="run S2-G04/G05 across north/equatorial/south faces and locations",
    )
    args = parser.parse_args()

    if args.suite:
        cases = [
            (name, face, location, sigma_px)
            for name, sigma_px in APPLICATION_CASES.items()
            for face in (0, 4, 8)
            for location in ("interior", "edge")
        ]
    else:
        cases = [("custom", args.face, args.location, args.sigma_px)]

    for name, face, location, sigma_px in cases:
        record = benchmark_case(
            name=name,
            level=args.level,
            face=face,
            size=args.size,
            location=location,
            sigma_px=sigma_px,
            truncate=args.truncate,
            warmups=args.warmups,
            repeats=args.repeats,
        )
        print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
