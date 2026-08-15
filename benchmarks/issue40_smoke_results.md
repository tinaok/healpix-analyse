# Issue 40 prototype smoke results

These measurements validate the benchmark and expose the first-order tradeoff;
they are not production performance claims. They were collected on an arm64
development Mac with Python 3.12, level 6, `truncate=4`, and tracemalloc enabled.
Values are a deterministic smooth synthetic field.

## Location and face matrix

The S2-G04 (`sigma=3 px`) and S2-G05 (`sigma=5 px`) suite used 16 x 16
patches on north, equatorial, and south faces, both inside a face and across a
single edge.

| Case | Exact runtime | Face runtime | RMSE vs exact | Fallback |
|---|---:|---:|---:|---:|
| S2-G04, all 6 locations | 0.325-0.434 s | 0.017-0.032 s | 0.0068-0.1622 | 0 |
| S2-G05, 5 no-fallback locations | 0.509-0.595 s | 0.024-0.045 s | 0.0055-0.0490 | 0 |
| S2-G05, north-face edge | 0.617 s | 0.619 s | 0.1249 | 32 / 2304 halo cells |

For no-fallback cases, the prototype was roughly 12-23 times faster than the
exact implementation in this small run. Peak Python-tracked memory was roughly
0.16-0.28 MB for face-native versus 7.9-9.1 MB for exact. The north-face edge
S2-G05 case affected 16 outputs; its exact correction raised face-native peak
memory to 9.3 MB and removed the speed advantage.

The largest errors occurred on the north-face edge, where the calibrated x/y
scales were strongly anisotropic. The equatorial-face cases had much smaller
RMSE. This confirms that patch location and face matter and that the exact
implementation must remain the reference.

## Patch-size scaling at an equatorial face interior

| Size | Sigma | Exact | Face | RMSE vs exact | Exact / face peak memory |
|---:|---:|---:|---:|---:|---:|
| 8 x 8 | 3 | 0.091 s | 0.013 s | 0.0158 | 0.58 / 0.09 MB |
| 8 x 8 | 5 | 0.158 s | 0.025 s | 0.0098 | 0.72 / 0.18 MB |
| 16 x 16 | 3 | 0.420 s | 0.019 s | 0.0178 | 7.95 / 0.17 MB |
| 16 x 16 | 5 | 0.724 s | 0.035 s | 0.0172 | 9.08 / 0.28 MB |
| 32 x 32 | 3 | 1.904 s | 0.042 s | 0.0112 | 52.49 / 0.44 MB |
| 32 x 32 | 5 | 3.376 s | 3.431 s | 0.0154 | 112.57 / 113.09 MB |

The final case has a 20-cell halo around a 32-cell patch on a 64-cell-wide
face. That support reaches face corners and invokes exact fallback, erasing the
performance benefit. This is useful evidence for the next design step: a true
multi-hop/corner topology primitive is important for large supports and
tile-scale readiness.

## Initial decision

Candidate A is worth advancing for compact patches whose support avoids face
corners: the separable filtering itself is fast, normalized convolution is
stable, and single-edge routing retains a substantial speed advantage. It is
not yet production-ready for corner-heavy or large-support patches because the
correctness fallback can dominate runtime and memory. Accuracy also requires
further application-specific acceptance thresholds, especially on anisotropic
north/south face locations.
