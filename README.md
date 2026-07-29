# mojo-bottleneck

`mojo-bottleneck` is a standalone Mojo port of the compute-heavy core of
[Bottleneck](https://github.com/pydata/bottleneck): NaN-aware reductions and
moving-window statistics. It exposes the same function names and call
signatures for the covered subset, so code can use:

```python
import mojo_bottleneck as bn
```

instead of `import bottleneck as bn`.

This is an early port, not a replacement for every Bottleneck function.

## Coverage

The following functions are implemented and parity-tested against Bottleneck
1.6.0:

| reductions | moving windows |
|---|---|
| `nansum` | `move_sum` |
| `nanmean` | `move_mean` |
| `nanvar` | `move_var` |
| `nanstd` | `move_std` |
| `nanmin` | `move_min` |
| `nanmax` | `move_max` |
| `nanmedian` | |
| `nanargmin`, `nanargmax` | |
| `anynan`, `allnan` | |

The reduction functions support `axis=None`, positive and negative axes, and
the applicable `ddof`. Moving functions support `window`, `min_count`, any
valid axis, and `ddof` for variance and standard deviation. Bool, integer,
float16, float32, and float64 inputs follow Bottleneck's output dtype rules.
Long-double reductions use NumPy for matching precision; long-double moving
windows are rejected rather than silently narrowed.

Not covered are the ordinary non-NaN reductions, `move_median`, `move_rank`,
`rankdata`, `nanrankdata`, `partition`, `argpartition`, `push`, `replace`, and
`ss`. The moving kernels retain Bottleneck's documented limitation around
infinities leaving a rolling accumulator.

## Install and run

The repository pins the verified Mojo nightly and all Python dependencies:

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` creates `dist/libmojo-bottleneck.so`. The activated Pixi
environment adds `python/` to `PYTHONPATH`.

```bash
pixi run python - <<'PY'
import numpy as np
import mojo_bottleneck as bn

x = np.array([1.0, np.nan, 3.0, 5.0])
print(bn.nanmean(x))
print(bn.move_mean(x, window=3, min_count=2))
PY
```

This prints:

```text
3.0
[nan nan  2.  4.]
```

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux x86-64, Python 3.13.14, and Bottleneck 1.6.0. Times are the best of five
warm runs on one-dimensional `float64` inputs containing NaNs. A ratio above
1 means Mojo is faster.

| operation | mojo-bottleneck | bottleneck | upstream / Mojo |
|---|---:|---:|---:|
| nansum, 5M | 5.81 ms | 7.05 ms | 1.21x |
| nanmean, 5M | 6.37 ms | 7.89 ms | 1.24x |
| nanstd, 5M | 12.73 ms | 20.13 ms | 1.58x |
| nanmedian, 1M | 15.38 ms | 14.01 ms | 0.91x |
| move_sum, 5M w=100 | 55.01 ms | 50.73 ms | 0.92x |
| move_var, 5M w=100 | 45.86 ms | 56.92 ms | 1.24x |
| move_min, 5M w=100 | 90.10 ms | 119.70 ms | 1.33x |
| move_max, 5M w=100 | 72.65 ms | 95.21 ms | 1.31x |

Mojo wins on six of the eight measured kernels in this run. It is behind on
`nanmedian` and `move_sum`. These timings include Python wrapper and
output-allocation costs. No GPU path is provided.

## How it works

NumPy owns all input, output, and scratch memory. The Python layer normalizes
the requested axis to the last dimension, makes a C-contiguous `float64` view
when necessary, and passes only integer buffer addresses and dimensions
through `ctypes`. The arrays remain strongly referenced for the duration of
each synchronous call. The Mojo shared library validates pointers and
dimensions before reconstructing mutable pointers inside the exported C-ABI
functions, returns an error status to Python, and does not allocate or retain
Python memory.

Reductions traverse independent contiguous rows. Variance uses a stable
two-pass algorithm; numeric reductions process full SIMD-width blocks followed
by a scalar tail. `nanmedian` compacts non-NaN values into a private copy before
in-place quickselect. Moving sums and means use rolling accumulators, moving
variance uses Welford insertion and removal updates, and moving extrema use an
O(n) monotonic deque. Large moving variance and extrema inputs are divided into
independent chunks, while inputs below the parallel threshold stay serial.
Non-last axes are restored to the original array layout before returning.
Float32 results are cast back to float32, integer moving-window results follow
Bottleneck's float64 convention, and uncommon floating reduction dtypes use
NumPy's matching fallback where precision requires it.

## Development

```bash
pixi run build
pixi run test
pixi run bench
```

The benchmark task holds `/tmp/mojo-bench.lock` so parallel jobs do not
silently contaminate measurements.
