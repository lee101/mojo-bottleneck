from __future__ import annotations

import math
import os
import platform
import sys
import time

import bottleneck as upstream
import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python")
)

import mojo_bottleneck as mb  # noqa: E402


def timeit(fn, repeat=5):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def cases():
    rng = np.random.default_rng(2026)
    large = rng.normal(size=5_000_000)
    large[::23] = np.nan
    median_data = large[:1_000_000].copy()
    return [
        ("nansum, 5M", lambda: mb.nansum(large), lambda: upstream.nansum(large)),
        ("nanmean, 5M", lambda: mb.nanmean(large), lambda: upstream.nanmean(large)),
        ("nanstd, 5M", lambda: mb.nanstd(large), lambda: upstream.nanstd(large)),
        (
            "nanmedian, 1M",
            lambda: mb.nanmedian(median_data),
            lambda: upstream.nanmedian(median_data),
        ),
        (
            "move_sum, 5M w=100",
            lambda: mb.move_sum(large, 100, min_count=50),
            lambda: upstream.move_sum(large, 100, min_count=50),
        ),
        (
            "move_var, 5M w=100",
            lambda: mb.move_var(large, 100, min_count=50),
            lambda: upstream.move_var(large, 100, min_count=50),
        ),
        (
            "move_min, 5M w=100",
            lambda: mb.move_min(large, 100, min_count=50),
            lambda: upstream.move_min(large, 100, min_count=50),
        ),
        (
            "move_max, 5M w=100",
            lambda: mb.move_max(large, 100, min_count=50),
            lambda: upstream.move_max(large, 100, min_count=50),
        ),
    ]


def main():
    print(f"Machine: {cpu_name()} ({platform.system()} {platform.machine()})")
    print(f"Python {platform.python_version()}, Bottleneck {upstream.__version__}")
    print()
    print("| operation | mojo-bottleneck | bottleneck | upstream / Mojo |")
    print("|---|---:|---:|---:|")
    for name, ours, theirs in cases():
        ours()
        theirs()
        mojo_time = timeit(ours)
        upstream_time = timeit(theirs)
        print(
            f"| {name} | {mojo_time * 1e3:.2f} ms | "
            f"{upstream_time * 1e3:.2f} ms | {upstream_time / mojo_time:.2f}x |"
        )


if __name__ == "__main__":
    main()
