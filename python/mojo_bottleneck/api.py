from __future__ import annotations

import operator

import numpy as np

from ._lib import addr, checked_call

_REDUCE_OP = {
    "sum": 0,
    "mean": 1,
    "var": 2,
    "std": 3,
    "min": 4,
    "max": 5,
    "any": 6,
    "all": 7,
}


def _array(a) -> np.ndarray:
    array = np.asarray(a)
    if array.dtype.kind not in "biuf":
        raise TypeError(f"unsupported dtype {array.dtype}")
    return array


def _axis(axis, ndim: int, *, allow_none: bool) -> int | None:
    if axis is None:
        if allow_none:
            return None
        raise TypeError("axis must be an integer")
    axis = operator.index(axis)
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise np.exceptions.AxisError(axis, ndim=ndim)
    return axis


def _work_along_axis(array: np.ndarray, axis):
    normalized = _axis(axis, array.ndim, allow_none=True)
    if normalized is None:
        work = np.ascontiguousarray(array, dtype=np.float64).reshape(1, array.size)
        return work, ()
    moved = np.moveaxis(array, normalized, -1)
    work = np.ascontiguousarray(moved, dtype=np.float64)
    outer = int(np.prod(moved.shape[:-1], dtype=np.intp))
    return work.reshape(outer, moved.shape[-1]), moved.shape[:-1]


def _reduction(a, axis, name: str, ddof: int = 0):
    array = _array(a)
    if name in {"sum", "min", "max"} and array.dtype.kind in "biu":
        fn = {"sum": np.sum, "min": np.min, "max": np.max}[name]
        dtype = None
        if name == "sum":
            if array.dtype.kind == "u":
                dtype = np.uint64
            elif array.dtype.kind == "b" or array.dtype.itemsize < 4:
                dtype = np.int64
            else:
                dtype = array.dtype
        result = fn(array, axis=axis, dtype=dtype) if dtype is not None else fn(
            array, axis=axis
        )
        return result.item() if np.ndim(result) == 0 else result
    if array.dtype.kind == "f" and array.dtype not in (np.float32, np.float64):
        fn = {
            "sum": np.nansum,
            "mean": np.nanmean,
            "var": np.nanvar,
            "std": np.nanstd,
            "min": np.nanmin,
            "max": np.nanmax,
        }[name]
        kwargs = {"axis": axis}
        if name in {"var", "std"}:
            kwargs["ddof"] = ddof
        with np.errstate(all="ignore"):
            result = fn(array, **kwargs)
        return result.item() if np.ndim(result) == 0 else result
    work, result_shape = _work_along_axis(array, axis)
    if name in {"min", "max"} and work.shape[1] == 0 and work.shape[0]:
        raise ValueError("zero-size array to reduction operation")
    result = np.full(work.shape[0], np.nan, dtype=np.float64)
    if work.shape[0]:
        checked_call(
            "mb_reduce",
            addr(work),
            addr(result),
            work.shape[0],
            work.shape[1],
            _REDUCE_OP[name],
            operator.index(ddof),
        )
    if array.dtype == np.float32:
        result = result.astype(np.float32)
    shaped = result.reshape(result_shape)
    return shaped.item() if not result_shape else shaped


def nansum(a, axis=None):
    return _reduction(a, axis, "sum")


def nanmean(a, axis=None):
    return _reduction(a, axis, "mean")


def nanvar(a, axis=None, ddof=0):
    return _reduction(a, axis, "var", ddof)


def nanstd(a, axis=None, ddof=0):
    return _reduction(a, axis, "std", ddof)


def nanmin(a, axis=None):
    return _reduction(a, axis, "min")


def nanmax(a, axis=None):
    return _reduction(a, axis, "max")


def anynan(a, axis=None):
    return _boolean_reduction(a, axis, "any")


def allnan(a, axis=None):
    return _boolean_reduction(a, axis, "all")


def _boolean_reduction(a, axis, name):
    array = _array(a)
    if array.dtype.kind in "biu":
        fn = np.any if name == "any" else np.all
        result = fn(np.isnan(array), axis=axis)
        return result.item() if result.ndim == 0 else result
    work, result_shape = _work_along_axis(array, axis)
    result = np.empty(work.shape[0], dtype=np.float64)
    if work.shape[0]:
        checked_call(
            "mb_reduce",
            addr(work), addr(result), work.shape[0], work.shape[1], _REDUCE_OP[name], 0
        )
    shaped = result.astype(bool).reshape(result_shape)
    return shaped.item() if not result_shape else shaped


def nanmedian(a, axis=None):
    array = _array(a)
    if array.dtype.kind == "f" and array.dtype not in (np.float32, np.float64):
        with np.errstate(all="ignore"):
            result = np.nanmedian(array, axis=axis)
        return result.item() if np.ndim(result) == 0 else result
    work, result_shape = _work_along_axis(array, axis)
    work = work.copy()
    result = np.full(work.shape[0], np.nan, dtype=np.float64)
    if work.shape[0]:
        checked_call(
            "mb_nanmedian", addr(work), addr(result), work.shape[0], work.shape[1]
        )
    if array.dtype == np.float32:
        result = result.astype(np.float32)
    shaped = result.reshape(result_shape)
    return shaped.item() if not result_shape else shaped


def _arg_reduction(a, axis, find_max: bool):
    array = _array(a)
    normalized = _axis(axis, array.ndim, allow_none=True)
    if array.dtype.kind in "biu":
        fn = np.argmax if find_max else np.argmin
        return fn(array, axis=normalized)
    if array.dtype not in (np.float32, np.float64):
        fn = np.nanargmax if find_max else np.nanargmin
        return fn(array, axis=normalized)
    work, result_shape = _work_along_axis(array, normalized)
    if work.shape[1] == 0 and work.shape[0]:
        raise ValueError("zero-size array to reduction operation")
    result = np.full(work.shape[0], -1, dtype=np.int64)
    invalid = 0
    if work.shape[0]:
        invalid = checked_call(
            "mb_arg_reduce",
            addr(work), addr(result), work.shape[0], work.shape[1], int(find_max)
        )
    if invalid:
        raise ValueError("All-NaN slice encountered")
    shaped = result.reshape(result_shape)
    return shaped.item() if not result_shape else shaped


def nanargmin(a, axis=None):
    return _arg_reduction(a, axis, False)


def nanargmax(a, axis=None):
    return _arg_reduction(a, axis, True)


def _move_args(a, window, min_count, axis):
    array = _array(a)
    if array.dtype.kind == "f" and array.dtype.itemsize > 8:
        raise TypeError(
            f"moving-window kernels do not support {array.dtype} without narrowing"
        )
    normalized = _axis(axis, array.ndim, allow_none=False)
    window = operator.index(window)
    if window < 1:
        raise ValueError("window must be positive")
    if window > array.shape[normalized]:
        raise ValueError("Moving window (= %d) must between 1 and %d, inclusive" %
                         (window, array.shape[normalized]))
    if min_count is None:
        min_count = window
    else:
        min_count = operator.index(min_count)
    if min_count <= 0:
        raise ValueError("min_count must be greater than zero")
    if min_count > window:
        raise ValueError("min_count must be less than or equal to window")
    moved = np.moveaxis(array, normalized, -1)
    work = np.ascontiguousarray(moved, dtype=np.float64)
    rows = work.reshape(-1, work.shape[-1])
    result = np.empty_like(work, dtype=np.float64)
    return array, normalized, window, min_count, rows, result


def _finish_move(array, normalized, result):
    if array.dtype.kind == "f" and array.dtype != np.float64:
        result = result.astype(array.dtype)
    restored = np.moveaxis(result, -1, normalized)
    return np.ascontiguousarray(restored)


def _move_sum_mean(a, window, min_count, axis, mean_mode):
    array, normalized, window, min_count, rows, result = _move_args(
        a, window, min_count, axis
    )
    if rows.shape[0]:
        checked_call(
            "mb_move_sum_mean",
            addr(rows),
            addr(result),
            rows.shape[0],
            rows.shape[1],
            window,
            min_count,
            int(mean_mode),
        )
    return _finish_move(array, normalized, result)


def move_sum(a, window, min_count=None, axis=-1):
    return _move_sum_mean(a, window, min_count, axis, False)


def move_mean(a, window, min_count=None, axis=-1):
    return _move_sum_mean(a, window, min_count, axis, True)


def _move_variance(a, window, min_count, axis, ddof, std_mode):
    array, normalized, window, min_count, rows, result = _move_args(
        a, window, min_count, axis
    )
    if rows.shape[0]:
        checked_call(
            "mb_move_var",
            addr(rows),
            addr(result),
            rows.shape[0],
            rows.shape[1],
            window,
            min_count,
            operator.index(ddof),
            int(std_mode),
        )
    return _finish_move(array, normalized, result)


def move_var(a, window, min_count=None, axis=-1, ddof=0):
    return _move_variance(a, window, min_count, axis, ddof, False)


def move_std(a, window, min_count=None, axis=-1, ddof=0):
    return _move_variance(a, window, min_count, axis, ddof, True)


def _move_extreme(a, window, min_count, axis, find_max):
    array, normalized, window, min_count, rows, result = _move_args(
        a, window, min_count, axis
    )
    scratch = np.empty(rows.shape[1], dtype=np.int64)
    if rows.shape[0]:
        checked_call(
            "mb_move_extreme",
            addr(rows),
            addr(result),
            addr(scratch),
            rows.shape[0],
            rows.shape[1],
            window,
            min_count,
            int(find_max),
        )
    return _finish_move(array, normalized, result)


def move_min(a, window, min_count=None, axis=-1):
    return _move_extreme(a, window, min_count, axis, False)


def move_max(a, window, min_count=None, axis=-1):
    return _move_extreme(a, window, min_count, axis, True)
