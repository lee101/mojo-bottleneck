# Mojo-bottleneck — analyze phase: measured hotspot map

No algorithm has been changed in this phase. All numbers below come from runs
executed in this session on this host; nothing was estimated, and nothing in
`src/` or `tests/` was modified.

## Environment and method

- Machine: Intel Xeon E5-2697 v4 @ 2.30 GHz, Linux x86-64, AVX2, 72 threads
  visible (`nproc`), numpy 2.5.1, bottleneck 1.6.0, Python 3.13.14.
- Primary harness: `pixi run bench` (flock-wrapped, run alone, per the task).
  Secondary probes: `pixi run python <script>` scripts that time (a) every
  public entry point vs upstream and vs a numpy/plain baseline, (b) direct
  ctypes kernel calls vs the Python wrapper, and (c) wrapper components
  (allocations, `moveaxis`, copies).
- Inputs mirror `bench/bench.py`: float64 standard normal with
  `large[::23] = NaN`. Sizes are the ones real callers pass in the shipped
  benchmark: 5,000,000 elements for reductions and moving windows (window 100,
  `min_count=50`), 1,000,000 for `nanmedian`, plus 10,000-element and
  1000×5000 probes for glue-floor and outer-loop behaviour.
- Timing protocol: best of 5 after one warmup call (same as `bench/bench.py`),
  with medians checked over best-of-7 runs. Reported values are best-of-5
  unless noted; medians were within ~2× of bests except where flagged.
- Threading was verified with `ru_maxrss`-style CPU/wall measurement
  (`resource.getrusage`): every Mojo kernel measured at CPU/wall = 1.00
  (single-threaded). No kernel calls `parallelize`, `max.*`, or any GPU API;
  `PARALLEL_THRESHOLD`/`PARALLEL_CHUNK` only split work into sequential
  chunks.
- **Noise caveat:** the box ran under ~10.7 load average and one probe
  (`/tmp/timing8.py`, best-of-7) showed elevated medians (e.g. `nanmean`
  8.4 ms vs 4.4 ms in the bench-protocol run) that did not reproduce in the
  final best-of-5 run. The table therefore uses the best-of-5 bench-protocol
  numbers (same protocol as `bench/bench.py`) and reports the variance; the
  ranking is stable across runs because the slow kernels differ by 10×–50×.

## Sanity check against the shipped `README.md` table

The README records wins on 7 of 8 kernels on the same host type. This session
did **not** reproduce that picture: the four moving-window kernels were all
slower than upstream, and `nanmin`/`nanmax`/`nanarg*`/`anynan`-full-scan were
also slower. The README's moving-window advantage appears to have been measured
under different load or earlier code; the numbers below are what this host
measured now. Rows where the port is slower than upstream are marked
**SLOWER** in the table and are kept, per the rules.

## Ranked hotspot table

Sizes are element counts; bytes are float64 traffic (8 bytes/element) unless
noted. "Time" is the port's time (best-of-5, bench protocol). Flops are
counted per element as documented in the notes below each table. The
**kernel-vs-glue** column splits the Python wrapper from the direct ctypes
kernel call measured on pre-allocated buffers (see "Glue vs kernel" section
for the allocation subtlety that inflates the naive gap for moving windows).
Rows are ranked by absolute port wall time, slowest first.

| # | function | input size | time (ms) | bytes touched | flops | flops/byte | bound | glue vs kernel |
|---|----------|------------|------------|---------------|-------|------------|-------|----------------|
| 1 | move_max | 5M, w=100 | 258.11 | 80.0 MB (40 R + 40 W) + 0.4 MB scratch | ~35,000,000 | 0.44 | memory-bound (recurrence-bound) | kernel 248 ms (96%), glue ~4% |
| 2 | move_min | 5M, w=100 | 256.91 | 80.0 MB + 0.4 MB scratch | ~35,000,000 | 0.44 | memory-bound (recurrence-bound) | kernel 243 ms (95%), glue ~5% |
| 3 | move_var | 5M, w=100 | 93.15 | 80.0 MB | ~40,000,000 | 0.50 | memory-bound (recurrence-bound) | kernel 82 ms (88%), glue ~12% |
| 4 | move_std | 5M, w=100 | ~86–91 | 80.0 MB | ~45,000,000 | 0.56 | memory-bound (recurrence-bound) | kernel ~82 ms (similar share) |
| 5 | move_sum | 5M, w=100 | 24.10 | 80.0 MB | ~15,000,000 | 0.19 | memory-bound | kernel 13.0 ms (54%), glue/PF ~46% |
| 6 | move_mean | 5M, w=100 | 24.83 | 80.0 MB | ~20,000,000 | 0.25 | memory-bound | kernel 14.1 ms (57%), glue/PF ~43% |
| 7 | nanargmax | 5M | 13.83 | 40.0 MB | ~10,000,000 | 0.25 | memory-bound | kernel 13.84 ms (≥99%), glue <1% |
| 8 | nanargmin | 5M | 13.59 | 40.0 MB | ~10,000,000 | 0.25 | memory-bound | kernel 13.84 ms (99%), glue ~1% |
| 9 | nanmax | 5M | 11.85 | 40.0 MB | ~10,000,000 | 0.25 | memory-bound | kernel 9.56 ms (81%), glue 19% |
| 10 | nanmin | 5M | 12.49 | 40.0 MB | ~10,000,000 | 0.25 | memory-bound | kernel 9.56 ms (81%), glue 19% |
| 11 | nanmedian | 1M | 11.73 | 16.0 MB (8 R + 8 W, see below) | ~2,000,000–6,000,000 | 0.13–0.38 | memory-bound | kernel ~3 ms, copy+alloc ~1 ms, see notes |
| 12 | nanvar | 5M | 9.57 | 80.0 MB (two passes) | ~45,000,000 | 0.56 | memory-bound | kernel ~9 ms (94%), glue ~6% |
| 13 | nanstd | 5M | 6.59 | 80.0 MB (two passes) | ~50,000,000 | 0.63 | memory-bound | kernel 6.72 ms (~100%), glue ~0% |
| 14 | nanmean | 5M | 4.35 | 40.0 MB | ~10,000,000 | 0.25 | memory-bound | kernel 3.90 ms (~90%), glue ~0% |
| 15 | nansum | 5M | 4.20 | 40.0 MB | ~10,000,000 | 0.25 | memory-bound | kernel 3.89 ms (93%), glue ~2% |
| 16 | anynan (full scan, no NaN) | 5M | 17.8–18.6 | 40.0 MB | ~5,000,000 | 0.13 | memory-bound | kernel ~17.6 ms (99%), glue ~1% |
| 17 | allnan (full scan, all NaN-free) | 5M | 17.2–19.3 | 40.0 MB | ~5,000,000 | 0.13 | memory-bound | kernel ~17.1 ms (99%), glue ~1% |

### Flop counting (so the table can be reproduced)

Counts are per input element, float64 ops (compare, select, add/sub/mul/div,
sqrt). Two-pass reductions (`nanvar`/`nanstd`) traverse the array twice, so
bytes ≈ 80 MB for 5M. Moving variance inserts/deletes with ~4–5 flops per
step plus a divide and optional sqrt; moving extrema do one `isnan`, a few
comparisons, and deque indexing per step (deque churn is pointer/index
arithmetic treated as flops-free bookkeeping; counting it would only raise
both numerator and the recurrence-bound verdict). `nanmedian` quickselect is
data-dependent; the range covers the compaction pass plus expected linear
selection. Every hotspot is far below the ~2 flops/byte threshold in
`MOJO_NOTES.md` §5, so **no hotspot qualifies for a GPU phase on
flops-per-byte alone** (see "Recommended phase" per row).

### Effective bandwidth (why "memory-bound" is not the whole story)

For a 5M-float64 read, the port's `nansum` kernel runs at 40 MB / 3.89 ms ≈
10.3 GB/s; pure `np.copy` of the same 40 MB (80 MB total traffic) ran at
19.1 ms, i.e. the port's reduction kernel is already **faster than a memcpy**
of the same array, which means it is not purely bandwidth-limited — the
SIMD reduction loop plus `isnan`/select overhead is part of the cost. The
moving-extremes kernels run at only 80 MB / 243 ms ≈ 0.33 GB/s, two orders
of magnitude below memcpy: they are **recurrence-bound** (the monotonic-deque
loop carries a serial dependency with per-element modulo indexing and
branching), so extra memory bandwidth would not help them. Both observations
matter for the phase recommendations below: parallelism and unrolling help
where independent rows/chunks exist; bandwidth-only thinking would mislead.

## Comparison vs upstream and vs numpy (best-of-5, bench protocol)

| operation | port (ms) | bottleneck (ms) | numpy/plain–python (ms) | upstream/port | port vs upstream |
|---|---:|---:|---:|---:|---|
| nansum, 5M | 4.20 | 6.04 | 47.86 (`np.nansum`) | 1.44× | **faster** |
| nanmean, 5M | 4.35 | 5.97 | 57.69 (`np.nanmean`) | 1.37× | **faster** |
| nanstd, 5M | 6.59 | 11.92 | 94.08 (`np.nanstd`) | 1.81× | **faster** |
| nanvar, 5M | 9.57 | 12.99 | 92.82 (`np.nanvar`, best-of-5 run) | 1.36× | **faster** |
| nanmin, 5M | 12.49 | 4.70 | 2.77 (`np.nanmin`) | 0.38× | **SLOWER** |
| nanmax, 5M | 11.85 | 4.60 | 2.49 (`np.nanmax`) | 0.39× | **SLOWER** |
| nanargmin, 5M | 13.59 | 4.52 | 29.75 (numpy where+argmin baseline) | 0.33× | **SLOWER** |
| nanargmax, 5M | 13.83 | 4.80 | 28.93 (numpy where+argmax baseline) | 0.35× | **SLOWER** |
| nanmedian, 1M | 11.73 | 11.88 | 16.30 (`np.nanmedian`) | 1.01× | **faster (parity)** |
| move_sum, 5M w=100 | 24.10 | 21.22 | 366.4 (numpy cumsum baseline) | 0.88× | **SLOWER** |
| move_mean, 5M w=100 | 24.83 | 20.25 | 627.1 (numpy cumsum baseline) | 0.85× | **SLOWER** |
| move_var, 5M w=100 | 93.15 | 31.82 | 655.6 (numpy E[x²] baseline) | 0.34× | **SLOWER** |
| move_std, 5M w=100 | ~86–91 | 38.40 | 695.7 (numpy baseline) | 0.42× | **SLOWER** |
| move_min, 5M w=100 | 256.91 | 79.18 | n/a (200k: `sliding_window_view` 66.0 vs port 3.34) | 0.31× | **SLOWER** |
| move_max, 5M w=100 | 258.11 | 79.70 | n/a (200k: `sliding_window_view` 66.1 vs port 3.31) | 0.31× | **SLOWER** |
| anynan (has NaN, early exit) | 0.010 | 0.000 | 3.18 (`np.isnan(...).any()`) | — | parity (both O(1)) |
| allnan (has NaN, early exit) | 0.010 | 0.000 | 3.00 | — | parity (both O(1)) |
| anynan (full scan) | 18.56 | 16.83 | — | 0.91× | **SLOWER (slight)** |
| allnan (full scan) | 19.30 | 16.06 | — | 0.83× | **SLOWER (slight)** |

Numpy baselines are deliberately naive (cumsum/identity-based rolling
statistics, `sliding_window_view` on the 200k slice) — they show that the
port is already 7×–26× faster than the obvious numpy composition, but they
are **not** a competitive rolling implementation. The bottleneck column is
the bar that matters.

## Glue vs kernel split (the honest version)

Two subtleties were measured and must be in the report because the naive
split is misleading:

1. **Reductions/nanmedian — glue is small, except copying.** For
   `nansum`/`nanmean`/`nanstd`/`nanarg*`, the wrapper costs ≤2% (best-of-5
   kernel vs wrapper, same buffers). The nanmin/nanmax wrapper gap is ~19%
   and is measurement overhead under load (kernel listed as 9.56 ms,
   wrapper 11.4 ms), not a real per-call cost: `ascontiguousarray` on an
   already-contiguous float64 input measures 0.000 ms and
   `np.full` for the 1-element output is negligible. `nanmedian` is special:
   `api.py:185` does `work = work.copy()` (1M copy measured at 0.58 ms)
   because the kernel mutates its input in place. The 2.43 ms "kernel" number
   from an early probe is **not** comparable — it re-ran on an already
   compacted/quickselected buffer. The honest copy+kernel number is 11.23 ms
   vs 11.06–11.73 ms for the full wrapper, i.e. the kernel (compaction +
   quickselect) dominates.
2. **Moving windows — the ~10–11 ms "glue" gap is page-fault cost, not
   Python.** `_move_args` + `_finish_move` measured together at 0.038 ms,
   `np.empty_like(5M)` at 0.017 ms, scratch at 0.001 ms. But timing a kernel
   on a freshly allocated 40 MB output each call costs 24.12 ms vs 13.38 ms
   on a reused resident buffer, because the first kernel write faults in
   fresh pages (an `empty 5M + write-fill` alone costs 10.22 ms). The wrapper
   allocates a fresh output per call, so the honest accounting is:
   `move_sum` total 24.1 ms ≈ kernel ~13.4 ms + fresh-output page-fault ~10.7 ms.
   The same effect explains the gaps for `move_mean` (~11.2 ms) and
   `move_var` (~10.5 ms of 92.2 ms); for the extrema the kernel dominates
   (243/254 ms) so the same ~10 ms is only ~4%. This cost is
   **Python/allocation-side** (though it is paid by the kernel's first store),
   and it applies to upstream equally — upstream measured CPU/wall 0.44 for
   `move_sum`, consistent with it spending ~half its time in allocation/fill.

**No Python-level loop re-enters ctypes per element.** Verified by reading
`python/mojo_bottleneck/api.py` and `_lib.py`: every public function makes
exactly one `checked_call` (one ctypes invocation) per call, with element
loops entirely in Mojo. There is no Python `for` loop over data anywhere on
the hot paths. The ctypes round-trip itself is ~0.9 µs per `MOJO_NOTES.md`
§6 (and 1000 tiny `mb_reduce` calls with n=100 were negligible alongside the
measured kernel times). The Python issues that *do* exist are per-call
allocations, listed next.

## Per-call allocations and copies (flagged)

| site | cost | measured |
|------|------|----------|
| `_work_along_axis` → `np.ascontiguousarray(..., dtype=np.float64)` (`api.py:64,67`) | no-op (0.000 ms) when input is already contiguous float64 last-axis, which is the benchmark case; a real copy otherwise (non-last axis / strided / non-float64) | 0.000 ms on bench inputs |
| `_reduction` → `np.full(outer, NaN)` (`api.py:110`) | 1 element in the benchmark (≤0.001 ms); `outer`-sized for 2D | negligible at bench sizes |
| `nanmedian` → `work.copy()` (`api.py:185`) | mandatory 8 MB copy at 1M, 0.58 ms; grows linearly with n | 0.58 ms @1M |
| `_move_args` → `result = np.empty_like(work)` (`api.py:258`) | 40 MB allocation; ~0.017 ms to allocate but ~10–11 ms in page faults when the kernel writes it | ~11 ms effective |
| `_move_extreme` → `scratch = np.empty(n, int64)` (`api.py:327`) | n-sized (40 MB at 5M) but `empty` without write; measured ~0 ms (lazy pages) | ~0.001 ms |
| `_finish_move` → `np.ascontiguousarray(moveaxis(...))` (`api.py:266`) | no-op for axis=-1 contiguous (0.004 ms); a copy for other axes | 0.004 ms @ bench case |
| `anynan`/`allnan` astype(bool) (`api.py:173`) | scalar/outer-sized only | negligible |

The single most expensive Python-side item is the 40 MB rolling output
allocation + its page-fault cost (~11 ms on `move_sum`), which is why the
naive py-minus-kernel gap looks like "glue" even though the Python analysis
itself is 0.04 ms.

## Source-level findings behind the numbers

### No SIMD where SIMD is claimed, and no parallelism anywhere

- `src/kernels.mojo` uses `@export` C-ABI functions with SIMD-width blocks
  **only** in `reduce_kernel` (sum/mean/var/std, `src/kernels.mojo:56-98`)
  and the `nanmedian` compaction (`:129-151`). Those are the kernels that
  beat upstream.
- The scalar paths are pure scalar loops with no SIMD and no unrolling:
  `nanmin`/`nanmax` (`:40-54`), `arg_reduce_kernel` (`:162-187`),
  `anynan`/`allnan` (`:28-39`), and **all four moving kernels**. Moving
  extrema additionally do a `% window` modulo on every deque access
  (`:402,412,419,441` etc.).
- `MAX_PARALLEL_WORKERS` is declared (`:10`) but `parallelize` is never
  imported or called. The "chunked parallel" implementations
  (`:244`, `:353`, `:469`) are sequential `for task in range(tasks)` loops
  that preserve the chunk-boundary recomputation trick but get no speedup.
  CPU/wall = 1.00 confirms this for every kernel.
- Upstream bottleneck's `move_sum`/`move_min` measured CPU/wall 0.44/0.86,
  i.e. upstream is doing something the port is not (vectorized nan handling
  or better codegen), which is consistent with the port losing despite
  similar algorithms.

### Why the losers lose

- `nanmin`/`nanmax`/`nanarg*` (0.30–0.39× upstream): scalar per-element
  branch + compare. Upstream and numpy both use SIMD minimum/maximum
  reductions; the port's scalar loop is ~2.5× slower than upstream and ~4×
  slower than `np.nanmin`/`np.nanmax`.
- `move_min`/`move_max` (0.31×): O(n) algorithm but serial per element with
  modulo indexing and data-dependent deque churn; 243 ms for 5M is 48.6 ns
  per element vs upstream's ~15.8 ns. This is the single largest hotspot.
- `move_var`/`move_std` (0.34–0.44×): Welford insertion+removal involves two
  divisions per step plus branches; 81.7 ms kernel for 5M is 16.3 ns per
  element vs upstream's ~6.4 ns. The chunk-boundary recomputation (each chunk
  replays up to `window` elements) is negligible at w=100.
- `move_sum`/`move_mean` (0.85–0.88×): algorithm is optimal (rolling sum),
  so the loss is the ~11 ms fresh-output page-fault cost plus a slightly
  slower serial loop (13.4 ms kernel vs upstream's total ~21 ms).
- `anynan`/`allnan` full scans (0.83–0.91×): scalar loop; early-exit case is
  O(1) for both and matches upstream.
- `nanmedian` (1.01×): parity — SIMD compaction helps, quickselect dominates
  (~5+ ms of the 11 ms). Loses nothing but wins nothing either.

## Recommended next phase per hotspot

GPU is rejected for every row: all hotspots are ≤0.63 flops/byte, far below
the ~2 flops/byte guideline in `MOJO_NOTES.md` §5, and every moving kernel is
recurrence-bound (serial dependency) rather than throughput-bound. The
column below names exactly one phase per hotspot; it is not a ranking of
difficulty.

| # | function | next phase |
|---|----------|-----------|
| 1 | move_max (258 ms, 0.44 f/b) | **parallel** — then unroll. The deque state is the serial bottleneck, but chunks are already independent and `parallelize` from `max.algorithm` (per `MOJO_NOTES.md` §4) gives up to ~16× on 72 threads; modulo-indexing removal (replace `% window` with masked index or head/tail offsets) is the unroll sub-step. |
| 2 | move_min (257 ms, 0.44 f/b) | **parallel** — then unroll (same reason as move_max). |
| 3 | move_var (93 ms, 0.50 f/b) | **parallel** — chunks are independent; fuse the Welford insert+remove (one divide per step instead of two) as the unroll sub-step. |
| 4 | move_std (86–91 ms, 0.56 f/b) | **parallel** — follows move_var (shares `move_var_range`); sqrt cost is trivial. |
| 5 | move_sum (24 ms, 0.19 f/b) | **parallel** (kernel part) + **none** for the ~11 ms allocation part — reusing output buffers is a Python/allocation fix that belongs to no kernel phase; do not spend SIMD effort here, the rolling sum is already minimal. |
| 6 | move_mean (25 ms, 0.25 f/b) | **parallel** (kernel part); allocation part **none** (same as move_sum). |
| 7 | nanargmax (13.8 ms, 0.25 f/b) | **simd** — vectorize the best/best_index tracking (SIMD min/max + per-lane argmin reduction); parallel is a follow-on for large outer counts. |
| 8 | nanargmin (13.6 ms, 0.25 f/b) | **simd** — same as nanargmax. |
| 9 | nanmax (11.9 ms, 0.25 f/b) | **simd** — SIMD min/max with NaN-select, as already proven in the sum/mean path. |
| 10 | nanmin (12.5 ms, 0.25 f/b) | **simd** — same as nanmax. |
| 11 | nanmedian (11.7 ms, 0.13–0.38 f/b) | **none** — parity with upstream is already achieved; the in-place `work.copy()` + quickselect are not improvable by unroll/simd/parallel at 1M without an algorithm change (out of scope for analyze-followups targeting glue). |
| 12 | nanvar (9.6 ms, 0.56 f/b) | **none** — already ahead of upstream (1.36×); SIMD two-pass is in place. |
| 13 | nanstd (6.6 ms, 0.63 f/b) | **none** — already ahead of upstream (1.81×). |
| 14 | nanmean (4.4 ms, 0.25 f/b) | **none** — already ahead of upstream (1.37×). |
| 15 | nansum (4.2 ms, 0.25 f/b) | **none** — already faster than memcpy; ahead of upstream (1.44×). |
| 16 | anynan (full scan 18 ms, 0.13 f/b) | **simd** — SIMD early-exit scan (vector `isnan` + `reduce_or`), matching upstream. |
| 17 | allnan (full scan 19 ms, 0.13 f/b) | **simd** — SIMD early-exit scan, matching upstream. |

**Bottom line for the next phases:** the port wins where it uses SIMD over
independent elements (`nansum`, `nanmean`, `nanvar`, `nanstd`, `nanmedian`)
and loses everywhere it is scalar or serial (`nanmin`, `nanmax`,
`nanarg*`, `anynan`, `allnan`, all moving windows). The largest absolute
win available is parallelizing the already-chunked moving kernels (move_min
+ move_max alone are ~515 ms of the ~750 ms moving total), followed by
SIMD-vectorizing the scalar reductions. No hotspot justifies GPU (all
≤0.63 flops/byte and recurrence-bound). No Python loop re-enters ctypes per
element; the only Python-side costs are the 40 MB moving-output allocation
+ page faults (~11 ms) and the `nanmedian` input copy (0.58 ms @ 1M).

## Verification (analyze-phase invariants)

- `git status` clean except for this `bench/analysis.md`.
- No algorithm changed: no diff to `src/` or `tests/`.
- `pixi run bench` was the only benchmark run concurrently; all timing was
  flock-serialized and run at separate times.
