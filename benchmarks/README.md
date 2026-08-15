# Gaussian benchmark

Run one small comparison:

```bash
python benchmarks/benchmark_gaussian_filter.py \
  --level 6 --size 16 --face 4 --location interior --sigma-px 3
```

Run the application-derived S2-G04 (`sigma=3`) and S2-G05 (`sigma=5`)
matrix over north, equatorial and south base faces, for interior and edge
patches:

```bash
python benchmarks/benchmark_gaussian_filter.py --suite --level 7 --size 32
```

Each JSON line records total runtime and peak Python-tracked memory for the
exact WGS84, Cartesian SciPy, cold face-native and cached face-native paths.
It also includes face-native setup/filter/fallback timing, error against both
references, and fallback halo count/fraction. Increase `--size`, vary `--face`
and select `--location corner` to study patch scale and topology sensitivity.

The exact implementation is intentionally included in every run and can make
large benchmark matrices slow. Full one-tile performance is outside the scope
of this prototype.

See [issue40_smoke_results.md](issue40_smoke_results.md) for one checked
prototype run and the resulting go/no-go observations.
