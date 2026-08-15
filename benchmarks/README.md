# Gaussian benchmark

Run one small comparison:

```bash
python benchmarks/benchmark_gaussian_filter.py \
  --level 20 --size 16 --face 4 --location interior --sigma-px 3
```

Run the application-derived S2-G04 (`sigma=3`) and S2-G05 (`sigma=5`)
matrix over north, equatorial and south base faces, for interior and edge
patches:

```bash
python benchmarks/benchmark_gaussian_filter.py --suite --level 20 --size 16
```

Runtime and memory are measured in separate invocations so that
`tracemalloc` does not distort timing. Runtime records report the median,
minimum and maximum after warm-up; `--warmups` and `--repeats` control the
measurement. Peak memory is Python-tracked memory from a separate invocation.

Each JSON line covers exact WGS84, Cartesian SciPy, cold face-native and warm
face-native paths. It also includes face-native setup/filter/fallback timing,
error against both references, and fallback halo count/fraction. Increase
`--size`, vary `--face` and select `--location corner` to study patch scale
and topology sensitivity.

S2-G04/G05 evidence intended for comparison with S2MSI must use level 20.
Lower levels are useful only for topology stress tests and must not be
presented as S2MSI performance results.

The exact implementation is intentionally included in every run and can make
large benchmark matrices slow. Full one-tile performance is outside the scope
of this prototype.

See [issue40_smoke_results.md](issue40_smoke_results.md) for one checked
prototype run and the resulting go/no-go observations.
