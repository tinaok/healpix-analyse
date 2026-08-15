# Face-native Gaussian prototype

`face_native_gaussian_filter` is an experimental fast Gaussian backend for
dense, local NESTED HEALPix patches. It complements rather than replaces
`gaussian_filter`: the latter retains exact WGS84 distance semantics and is the
reference implementation.

```python
from healpix_analyse import face_native_gaussian_filter

smoothed, stats = face_native_gaussian_filter(
    values,
    cell_ids,
    refinement_level=14,
    sigma_m=30.0,
    domain=cell_ids,
    return_stats=True,
)

print(stats.fallback_halo_cells, stats.fallback_halo_fraction)
```

## Approximation

The backend selects one representative patch cell and measures its exact
WGS84 centre-to-centre distance to the face-local `+x` and `+y` cells. These
two scales are held constant over the patch:

```text
sigma_x_px = sigma_m / dx_m
sigma_y_px = sigma_m / dy_m
```

It then applies finite 1-D Gaussian kernels along x and y. `pass_order="xy"`
is the default; `"yx"` is provided as a diagnostic and is numerically
equivalent up to floating-point round-off.

This patch-level metric approximation can vary in accuracy with patch size,
latitude and base face. Use the exact filter where metre-level kernel semantics
are required.

## Topology and fallback

The thin adapter in `_face_topology.py` is the only code coupled to
`healpix_geo.nested.pix2xyf`, `xyf2pix`, and `face_neighbour_transform` from
[healpix-geo PR #244](https://github.com/GRID4EARTH/healpix-geo/pull/244).
Same-face and single-edge halo cells use those integer topology operations.

Face-corner and multi-edge transforms are deliberately not composed. If such
a halo location falls within an output cell's kernel support, that output is
replaced with the exact WGS84 `gaussian_filter` result. `return_stats=True`
reports both halo lookup and affected-output counts, together with setup,
separable filtering and exact fallback timings.

## Missing data and partial domains

Unavailable cells and non-finite samples are excluded with normalized
convolution:

```text
Gaussian(values * valid) / Gaussian(valid)
```

Consequently a finite constant field stays constant at a partial-domain edge.
An output is `NaN` only when its effective denominator is zero.

## Prototype limitations

- NumPy arrays only; the exact reference also supports PyTorch.
- Best suited to compact, tile-like domains. A sparse domain with a very large
  face-local bounding box can require a correspondingly large temporary tile.
- Requires the API currently supplied by healpix-geo PR #244 until that PR is
  merged and released.
- The benchmark is evidence gathering for a candidate backend, not a claim of
  production-ready accuracy or speed.
