# Issue 40 level-20 S2MSI benchmark results

These measurements validate the prototype at the HEALPix level used for the
S2MSI comparison. They are not production performance claims.

The run used:

- HEALPix level 20 (`nside = 1,048,576`);
- an arm64 development Mac with Python 3.12;
- `truncate=4`;
- one warm-up and three measured repetitions;
- median wall-clock runtime measured without `tracemalloc`;
- Python-tracked peak memory measured in a separate invocation;
- a deterministic smooth synthetic field.

At the representative locations, S2-G04 (`sigma=3 px`) corresponds to
approximately 18.66-20.44 m and S2-G05 (`sigma=5 px`) to approximately
31.11-34.06 m. The range comes from face/location-dependent WGS84 calibration.

## Location and face matrix

The suite used 16 x 16 patches on north, equatorial and south faces, both
inside a face and across one face edge.

| Case | SciPy | Exact WGS84 | Face-native | Exact / face | RMSE vs exact | Fallback |
|---|---:|---:|---:|---:|---:|---:|
| S2-G04, 6 locations | 0.024-0.027 ms | 185.8-203.0 ms | 2.225-3.560 ms | 54-87x | 0.00515-0.20755 | 0 |
| S2-G05, 6 locations | 0.026-0.028 ms | 239.3-273.9 ms | 2.283-3.930 ms | 62-111x | 0.00508-0.15694 | 0 |

SciPy operates on an already materialized Cartesian array and therefore
remains an idealized filtering lower bound. It does not include HEALPix face
setup, topology-aware halo construction, or restoration to `cell_ids` order.
The complete face-native path was approximately 82-148 times slower than this
SciPy-only lower bound, but 54-111 times faster than exact WGS84 filtering.

Peak Python-tracked memory was approximately:

| Case | Exact WGS84 | Face-native |
|---|---:|---:|
| S2-G04 | 7.94-8.00 MB | 0.160-0.170 MB |
| S2-G05 | 9.05-9.08 MB | 0.270-0.282 MB |

No level-20 S2-G04/G05 case used corner or multi-edge fallback. Unlike the
earlier low-level topology smoke run, a 12- or 20-cell halo is tiny relative
to a level-20 face width.

The largest errors occurred on the north-face edge. Its representative metric
was strongly anisotropic (`dx` approximately 9.16 m and `dy` approximately
5.07 m), whereas equatorial edge cases had much smaller RMSE. This confirms
that location-dependent approximation error, rather than fallback, is the
main accuracy concern for these level-20 S2 cases.

## Patch-size scaling at a level-20 equatorial face interior

| Size | Sigma | SciPy | Exact WGS84 | Face-native | RMSE vs exact | Exact / face peak memory |
|---:|---:|---:|---:|---:|---:|---:|
| 8 x 8 | 3 | 0.023 ms | 40.946 ms | 2.026 ms | 0.01585 | 0.578 / 0.091 MB |
| 8 x 8 | 5 | 0.022 ms | 59.303 ms | 2.059 ms | 0.00984 | 0.723 / 0.182 MB |
| 16 x 16 | 3 | 0.027 ms | 180.514 ms | 2.194 ms | 0.01826 | 7.950 / 0.170 MB |
| 16 x 16 | 5 | 0.027 ms | 237.993 ms | 2.341 ms | 0.01753 | 9.075 / 0.278 MB |
| 32 x 32 | 3 | 0.038 ms | 793.454 ms | 2.642 ms | 0.01275 | 52.484 / 0.451 MB |
| 32 x 32 | 5 | 0.044 ms | 1079.156 ms | 3.128 ms | 0.01733 | 112.579 / 0.595 MB |

No patch-size case used fallback. Exact runtime and memory grew rapidly with
the number of target cells and Gaussian support. Face-native runtime grew much
more slowly because its approximately 2-4 ms topology/setup cost dominates the
roughly 0.05-0.19 ms separable Gaussian passes at these patch sizes.

## Initial decision

At level 20, Candidate A is worth advancing for S2MSI-scale local patches:

- it was 54-111 times faster than exact WGS84 in the face/location matrix;
- its runtime scaled much more slowly than exact filtering from 8 x 8 through
  32 x 32 patches;
- its Python-tracked peak memory was substantially lower;
- single-edge cases required no correctness fallback at S2-G04/G05 support.

It is not yet production-ready. North-face edge accuracy requires explicit
application acceptance thresholds, and true tile-scale testing remains future
work. The next performance target is topology/setup reuse and vectorization,
not the already-small separable Gaussian passes.
