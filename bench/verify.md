# Mojo-bottleneck — VERIFY phase

No code was changed in this phase (no failing test justified a fix). Only
`README.md`'s benchmark table was corrected and this file was added. All
numbers below were measured in this session on this host; nothing was
estimated and nothing was copied from `bench/analysis.md` into the README.

## Commands run, in order

- `pixi run build && pixi run test` → build OK, `1708 passed, 30 skipped`
  (exit 0). Re-run at the end; both green again.
- `pixi run bench` → 1 initial run, then 4 further runs, all
  flock-serialized and run one at a time.

Environment (from `pixi run bench` output and `nproc`): Intel Xeon
E5-2697 v4 @ 2.30 GHz, Linux x86_64, 72 threads visible, Python 3.13.14,
Bottleneck 1.6.0. This matches the README's stated environment, so the
discrepancies below are not a host mismatch.

## What reproduced

- `pixi run build && pixi run test` is green as shipped.
- The reductions `nansum`, `nanmean`, `nanstd` are faster than upstream and
  `nanmedian` is at parity. The old README values for these four are within
  ~20% of what was measured now (see table below), so they are not the
  problem — except the old ratios understated them (see "Corrected").
- `bench/bench.py` measures both sides on the same input (verified below).

## What did not reproduce

The four moving-window rows did not reproduce at all. The README claimed
Mojo wins on `move_sum`, `move_var`, `move_min`, and `move_max` (1.55x,
1.93x, 2.03x, 1.69x). Every one of the five runs in this session measured
Mojo **slower** than upstream on all four rows (0.88x, 0.35x, 0.31x, 0.30x).

| operation | README (old) | measured median (this session) | deviation |
|---|---:|---:|---:|
| nansum, 5M | 4.89 / 5.61 / 1.15x | 4.02 / 5.99 / 1.49x | mojo −17.9%, upstream +6.8% |
| nanmean, 5M | 4.73 / 5.89 / 1.25x | 3.92 / 5.84 / 1.49x | mojo −17.1%, upstream −0.8% |
| nanstd, 5M | 8.09 / 11.90 / 1.47x | 6.59 / 11.97 / 1.82x | mojo −18.5%, upstream +0.6% |
| nanmedian, 1M | 11.24 / 10.75 / 0.96x | 11.23 / 11.82 / 1.05x | mojo −0.1%, upstream +9.9% |
| move_sum, 5M w=100 | 15.16 / 23.58 / 1.55x | 23.78 / 20.97 / 0.88x | mojo +56.9%, upstream −11.1% |
| move_var, 5M w=100 | 20.02 / 38.63 / 1.93x | 92.56 / 31.99 / 0.35x | mojo +362%, upstream −17.2% |
| move_min, 5M w=100 | 46.69 / 94.94 / 2.03x | 252.29 / 78.08 / 0.31x | mojo +440%, upstream −17.8% |
| move_max, 5M w=100 | 45.22 / 76.64 / 1.69x | 257.82 / 78.54 / 0.30x | mojo +470%, upstream +2.5% |

The Mojo moving-window numbers in the old README are off by 3.6x–5.7x and
also flip the winner. This is consistent with `bench/analysis.md`'s own
finding (which recorded the same slower picture and flagged the README's
moving-window advantage as not reproduced) and with the code: at HEAD the
moving kernels are serial `for task in range(tasks)` loops — the
`parallelize` calls that the accel phase (d5e6eff) had added were removed
in the test phase (4033d0a) when the kernels were reworked, so the
parallel speedup behind the old table is no longer present.

Per the task rules, every figure off by more than 20% was replaced in the
README with the measured median of this session's five runs, including the
four rows that are slower (they stay slower in the table; they were not
removed or reworded). The rows within 20% on both sides (`nanmedian`,
and the upstream column in general) were also updated to the measured
medians so the table is one consistent measured set, and the "wins on
seven of the eight" sentence was corrected to four of eight.

The README now reads:

| operation | mojo-bottleneck | bottleneck | upstream / Mojo |
|---|---:|---:|---:|
| nansum, 5M | 4.02 ms | 5.99 ms | 1.49x |
| nanmean, 5M | 3.92 ms | 5.84 ms | 1.49x |
| nanstd, 5M | 6.59 ms | 11.97 ms | 1.82x |
| nanmedian, 1M | 11.23 ms | 11.82 ms | 1.05x |
| move_sum, 5M w=100 | 23.78 ms | 20.97 ms | 0.88x |
| move_var, 5M w=100 | 92.56 ms | 31.99 ms | 0.35x |
| move_min, 5M w=100 | 252.29 ms | 78.08 ms | 0.31x |
| move_max, 5M w=100 | 257.82 ms | 78.54 ms | 0.30x |

Run-to-run spread in this session (five `pixi run bench` runs, best-of-5
each): Mojo `nansum` 3.69–4.80 ms, `move_min` 234–255 ms; upstream
`nansum` 5.77–6.13 ms, `move_min` 74–79 ms. The ranking (reductions win,
moving windows lose, median at parity) never flipped across runs.

## Fairness checks (DO #4)

`bench/bench.py` was read and verified:

- Same input for both sides: each case calls `mb.<fn>` and
  `upstream.<fn>` with the same `large` array (5M float64, `large[::23]
  = NaN`, `default_rng(2026)`) or `median_data` (first 1M), and the same
  `window=100, min_count=50`. Both sides are timed with the same
  best-of-5 `timeit`, each warmed once first.
- No skipped work: timing calls never cache the result away — each
  `timeit` iteration calls the function and discards the return value,
  which still forces allocation and computation. A direct run confirmed
  the outputs are computed and equal in shape/dtype and NaN placement for
  all eight cases (`move_sum` agrees to abs diff ≤ 1.2e-12, i.e. within
  default `allclose`; `move_var`/`move_min`/`move_max`/`nanmedian` agree
  to `rtol=1e-12`). Inputs are not modified by either side.
- Leftover question, not a harness bug: the harness times the full
  wrapper including output allocation for both sides, which is also what
  the README states. That is fair, and it is also why `move_sum`'s gap
  (0.88x) is much smaller than `move_min`'s (0.31x).

## Test history check (DO #5)

`git log -p -- tests/` was reviewed across all three commits that touched
tests:

- 10426df (initial): `tests/test_parity.py` with 218 lines.
- d5e6eff (accel): only widened two `@parametrize` lists (added
  `nanmedian` to the SIMD-tail test and `move_sum`/`move_mean` to the
  parallel-threshold test) and added one parity test
  (`test_moving_kernels_write_invalid_outputs`). No tolerance was loosened
  and no assertion was removed.
- 4033d0a (test): added 572 lines of degenerate-input parity tests
  (`TOL = dict(rtol=1e-12, atol=1e-14)`) and, in `api.py`/`kernels.mojo`,
  fixed genuine parity bugs the new tests exposed (zero-length reduction
  axis, `nanvar`/`nanstd` with `count==0, ddof<0`, integer-argument
  validation, infinite values leaving a moving-variance window). The
  original-file diff shows no `-` lines in `tests/test_parity.py` other
  than the accel phase's parametrize widening.

Nothing was found to restore: no removed assertions, no loosened
tolerances. The current file is byte-identical to the test-phase version,
and all 1708 tests pass. The loosest tolerances are pre-existing and
justified by the data, not added to hide failures: `rtol=2e-4` for
float16 moving windows, `rtol=3e-5, atol=3e-7` for the 1e9-offset moving
variance stability test.

## Changes made

- `README.md`: benchmark table replaced with this session's measured
  medians; summary sentence corrected from "wins on seven of the eight /
  near parity on nanmedian" to "wins on four of the eight / near parity
  on nanmedian / slower on all four moving-window kernels". The slower
  rows remain in the table, unsoftened.
- `bench/verify.md`: this file.
- Nothing else. No source, test, or benchmark-harness change: the rule
  was to change code only for a bug shown by a failing test, and no code
  bug was found (build and tests were already green, and the benchmark
  harness was verified to compare like-for-like).

## What could not be reproduced

- The old README's moving-window wins (see above). They are not
  reproducible at HEAD with `pixi run bench` on the same host type and
  software versions; the table now reports what HEAD actually measures.
- `bench/analysis.md`'s per-kernel glue splits and numpy baselines were
  not re-measured in this phase; only the README table rows were.
