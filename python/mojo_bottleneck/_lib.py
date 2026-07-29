from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_BOTTLENECK_LIB") or os.path.join(
    ROOT, "dist", "libmojo-bottleneck.so"
)

I = ctypes.c_int64

_SIGNATURES = {
    "mb_reduce": ([I, I, I, I, I, I], I),
    "mb_arg_reduce": ([I, I, I, I, I], I),
    "mb_nanmedian": ([I, I, I, I], I),
    "mb_move_sum_mean": ([I, I, I, I, I, I, I], I),
    "mb_move_var": ([I, I, I, I, I, I, I, I], I),
    "mb_move_extreme": ([I, I, I, I, I, I, I, I], I),
}

_library: ctypes.CDLL | None = None


def build() -> str:
    source = os.path.join(ROOT, "src", "kernels.mojo")
    if os.environ.get("MOJO_BOTTLENECK_LIB") and os.path.exists(LIB):
        return LIB
    if os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(source):
        return LIB
    proc = subprocess.run(
        ["bash", os.path.join(ROOT, "build", "build.sh")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode or not os.path.exists(LIB):
        raise RuntimeError((proc.stderr or proc.stdout).strip())
    return LIB


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_library, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _library


def addr(array: np.ndarray) -> int:
    return array.ctypes.data


def checked_call(name: str, *args: int) -> int:
    """Call an exported kernel and reject an invalid FFI contract."""
    status = int(getattr(lib(), name)(*args))
    if status < 0:
        raise RuntimeError(f"{name} rejected invalid buffer metadata")
    return status
