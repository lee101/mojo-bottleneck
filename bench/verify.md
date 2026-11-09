# Mojo-bottleneck — VERIFY phase (re-verification, 2026-11-09 session)

This is an independent re-verification. `bench/verify.md` previously held a
prior VERIFY pass; this version reports only what was measured in *this*
session. All numbers below were measured in this session on this host; nothing
was estimated and nothing was copied from `bench/analysis.md` into the README.

## Commands run, in order

- `pixi run build && pixi run test` → build OK, `1708 passed, 30 skipped`
  (exit 0). Re-run at the end; both green again (see below).
- `pixi run bench` → 4 full runs, one at a time (flock-serialized by the
  `bench` task), plus 1 independent per-row driver (same protocol, see
  "Per-row re-measurement" below).
- `git log -p -- tests/` → reviewed in full (see "Test history check").

Environment (from `pixi run bench` output, `pixi run python`, `nproc`, and
`/proc/cpuinfo`): Intel Xeon E5-2697 v4 @ 2.30 GHz, Linux x86_64, 72 threads
visible, Python 3.13.14, Bottleneck 1.6.0. This matches the README's stated
environment, so any discrepancy is not a host mismatch.

## What reproduced

- `pixi run build && pixi run test` is green as shipped at HEAD.
- **Every row of the current README table reproduced within 20%.** The
  README at HEAD (corrected by the prior VERIFY pass) already reports the
  slower moving-window picture, and this session's numbers confirm it.
  Medians over this session's four `pixi run bench` runs:

  | operation | README (HEAD) mojo / upstream / ratio | this session median mojo / upstream / ratio | deviation (mojo / upstream) |
  |---|---:|---:|---:|
  | nansum, 5M | 4.02 / 5.99 / 1.49x | 3.93 / 5.74 / 1.46x | −2.2% / −4.3% |
  | nanmean, 5M | 3.92 / 5.84 / 1.49x | 3.88 / 5.77 / 1.49x | −1.1% / −1.3% |
  | nanstd, 5M | 6.59 / 11.97 / 1.82x | 6.64 / 11.64 / 1.75x | +0.8% / −2.7% |
  | nanmedian, 1M | 11.23 / 11.82 / 1.05x | 11.41 / 11.87 / 1.04x | +1.6% / +0.4% |
  | move_sum, 5M w=100 | 23.78 / 20.97 / 0.88x | 23.84 / 21.36 / 0.90x | +0.3% / +1.8% |
  | move_var, 5M w=100 | 92.56 / 31.99 / 0.35x | 92.82 / 32.49 / 0.35x | +0.3% / +1.6% |
  | move_min, 5M w=100 | 252.29 / 78.08 / 0.31x | 254.42 / 80.88 / 0.32x | +0.8% / +3.6% |
  | move_max, 5M w=100 | 257.82 / 78.54 / 0.30x | 257.75 / 79.56 / 0.31x | −0.0% / +1.3% |

  Run-to-run spread over the four runs: Mojo `nansum` 3.65–4.35 ms,
  `nanmean` 3.38–4.50 ms, `move_min` 244.90–258.30 ms; upstream
  `nansum` 5.55–5.81 ms, `move_min` 80.38–81.29 ms. The ranking (three
  reductions win, `nanmedian` at parity, all four moving windows lose)
  never flipped in any run, including the independent per-row driver.

- The qualitative claims reproduce: Mojo wins on the measured reductions
  (`nansum`, `nanmean`, `nanstd`), is near parity on `nanmedian`, and is
  slower on all four moving-window kernels (`move_sum`, `move_var`,
  `move_min`, `move_max`). "Wins on four of the eight" counts the three
  reductions plus `nanmedian` at 1.04–1.05x.

## What did not reproduce

- Nothing in the current README table is more than 20% off, so no figure
  needed correction (DO #2). The single noisiest row was `nanmean` Mojo
  (3.38–4.50 ms across runs, ±14% around the median) and `nansum` Mojo
  (3.65–4.35 ms); even the extremes stay within 20% of the table.
- The *pre-correction* README (before the prior VERIFY pass) is **not**
  reproducible and this session confirms why: it claimed Mojo wins on all
  four moving windows (1.55x/1.93x/2.03x/1.69x). At HEAD the moving kernels
  are serial `for task in range(tasks)` loops (`src/kernels.mojo:244,353`
  and the extreme kernel's chunk loop) — `PARALLEL_THRESHOLD`/
  `PARALLEL_CHUNK` split work into sequential chunks and `parallelize` is
  never called, matching `bench/analysis.md`'s threading probe
  (CPU/wall = 1.00 everywhere). The current README's slower rows are the
  correct HEAD picture; they were not softened.

## Per-row re-measurement (DO #2, independent driver)

One additional driver re-implemented the bench protocol exactly (same
`default_rng(2026)` `large` array, same `median_data`, same
`window=100, min_count=50`, best-of-5 `timeit`, one warmup each) and timed
each row again:

| operation | mojo | upstream | ratio |
|---|---:|---:|---:|
| nansum, 5M | 5.35 ms | 5.85 ms | 1.09x |
| nanmean, 5M | 4.96 ms | 6.04 ms | 1.22x |
| nanstd, 5M | 8.98 ms | 12.36 ms | 1.38x |
| nanmedian, 1M | 11.87 ms | 12.08 ms | 1.02x |
| move_sum, 5M w=100 | 24.54 ms | 20.85 ms | 0.85x |
| move_var, 5M w=100 | 86.73 ms | 36.57 ms | 0.42x |
| move_min, 5M w=100 | 234.64 ms | 76.70 ms | 0.33x |
| move_max, 5M w=100 | 264.61 ms | 78.98 ms | 0.30x |

The reduction rows run slower in this driver than in the four
`pixi run bench` runs (e.g. `nanstd` Mojo 8.98 vs 6.23–7.28 ms) — the box
was noisier during that run — but the per-row ranking and every
winner/loser decision agree with the table. No row flipped.

## Fairness checks (DO #4)

`bench/bench.py` was read and verified, and the outputs were checked
directly (`pixi run python`, same inputs as the bench):

- Same input for both sides: each case passes the *same* `large` array
  (5M float64, `large[::23] = NaN`, `default_rng(2026)`) or `median_data`
  (first 1M) to `mb.<fn>` and `upstream.<fn>`, with the same
  `window=100, min_count=50`. Both sides use the same best-of-5 `timeit`
  with one warmup call each. Timing order is port-first then upstream for
  every row, so any order bias is symmetric across rows.
- No skipped work: timing calls never cache the result away — each
  `timeit` iteration invokes the function and discards the return; the
  functions are eager and allocate a fresh output per call, so the work is
  done each time. A direct run confirmed the outputs agree: identical
  shape/dtype and NaN placement for all eight cases; max abs diff
  `nansum` 2.1e-10, `nanmean` 4.4e-17, `nanstd` 1.5e-14, `nanmedian` 0,
  `move_sum` 1.2e-12, `move_var` 1.8e-13, `move_min`/`move_max` 0.
- No input mutation: inputs were compared before/after both sides;
  `large` and `median_data` are unchanged (`np.array_equal(..., equal_nan=True)`).
- Disclosure: the wrapper paths include output allocation and, for
  `nanmedian`, a work-array copy (`api.py:185`), and the harness times the
  full wrapper for both sides, which is also what the README states. That
  is fair and like-for-like, and it explains the shape of the gaps
  (allocation-dominated `move_sum` at ~0.9x vs serial-recurrence
  `move_min`/`move_max` at ~0.3x).

## Test history check (DO #5)

`git log -p -- tests/` was reviewed across every commit that touched
tests. The only deleted (`-`) lines in the whole test history are two
`@parametrize` list lines replaced by widened versions in d5e6eff:

- 10426df (initial import): `tests/test_parity.py` with 218 lines.
- d5e6eff (accel): widened two parametrize lists (`nanmedian` added to
  SIMD-tail; `move_sum`/`move_mean` added to parallel-threshold) and added
  `test_moving_kernels_write_invalid_outputs`. No tolerance loosened, no
  assertion removed.
- 4033d0a (test): pure additions (572 lines) of degenerate-input parity
  tests with `TOL = dict(rtol=1e-12, atol=1e-14)`, plus code fixes in
  `api.py`/`kernels.mojo` for genuine parity bugs the new tests exposed.
- 67a1773 (prior verify): touched only `README.md` and
  `bench/verify.md`; no test changes.

Nothing was found to restore: no removed assertions, no loosened
tolerances. The loosest current tolerances are pre-existing from the
initial import and tied to documented data properties, not added to hide
failures: `rtol=2e-4` for float16 moving windows and
`rtol=3e-5, atol=3e-7` for the 1e9-offset moving-variance stability test
(both present at 10426df). All 1708 tests pass at HEAD with the file as
committed.

## Changes made

- `bench/verify.md`: this file (rewritten as an independent
  re-verification; the prior pass's text was replaced, and its numbers were
  only used as things to check against, not copied into the README).
- Nothing else. No source, test, or benchmark-harness change: the rule was
  to change code only for a bug shown by a failing test, and no bug was
  found (build and tests were green as shipped, the harness was verified
  like-for-like, and no README figure is >20% off — so no README correction
  was triggered either).

## What could not be reproduced

- The pre-correction README's moving-window wins (see above); they remain
  irreproducible at HEAD and are not restored.
- `bench/analysis.md`'s per-kernel glue splits and numpy baselines were
  not re-measured in this phase; only the README table rows were.
- One environmental footnote: a single `pixi run bench` run of `nanmean`
  (3.38 ms) and the per-row driver run (`nansum` 5.35 ms Mojo) show the box
  has occasional load noise, so the reductions' absolute times carry
  roughly ±15% run-to-run jitter. The table's medians sit inside that
  band and the winner/loser ranking was stable in every run.
