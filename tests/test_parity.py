import inspect
from pathlib import Path

import bottleneck as upstream
import numpy as np
import pytest

import mojo_bottleneck as mb
from mojo_bottleneck._lib import checked_call


@pytest.fixture
def sample():
    rng = np.random.default_rng(123)
    array = rng.normal(size=(4, 7, 31))
    array.ravel()[::9] = np.nan
    array[2, 3, :] = np.nan
    return array


@pytest.mark.parametrize(
    "name", ["nansum", "nanmean", "nanmin", "nanmax", "nanmedian", "anynan", "allnan"]
)
@pytest.mark.parametrize("axis", [None, 0, 1, 2, -1])
def test_reduction_parity(sample, name, axis):
    expected = getattr(upstream, name)(sample, axis=axis)
    actual = getattr(mb, name)(sample, axis=axis)
    assert np.allclose(actual, expected, equal_nan=True)


@pytest.mark.parametrize("name", ["nanvar", "nanstd"])
@pytest.mark.parametrize("axis", [None, 0, 1, -1])
@pytest.mark.parametrize("ddof", [0, 1, 3])
def test_variance_reduction_parity(sample, name, axis, ddof):
    expected = getattr(upstream, name)(sample, axis=axis, ddof=ddof)
    actual = getattr(mb, name)(sample, axis=axis, ddof=ddof)
    assert np.allclose(actual, expected, equal_nan=True, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("name", ["nanargmin", "nanargmax"])
@pytest.mark.parametrize("axis", [None, 0, 1, -1])
def test_arg_reduction_parity(sample, name, axis):
    data = sample.copy()
    data[2, 3, :] = np.arange(data.shape[-1])
    expected = getattr(upstream, name)(data, axis=axis)
    actual = getattr(mb, name)(data, axis=axis)
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("name", ["nanargmin", "nanargmax"])
def test_arg_reduction_rejects_all_nan(name):
    data = np.array([[1.0, 2.0], [np.nan, np.nan]])
    with pytest.raises(ValueError, match="All-NaN"):
        getattr(mb, name)(data, axis=1)


@pytest.mark.parametrize(
    "dtype", [np.bool_, np.int8, np.int32, np.int64, np.uint32, np.float16, np.float32, np.float64]
)
@pytest.mark.parametrize("name", ["nansum", "nanmean", "nanmin", "nanmax", "nanmedian"])
def test_reduction_dtype_and_values(dtype, name):
    data = np.arange(24).reshape(4, 6).astype(dtype)
    if np.issubdtype(dtype, np.floating):
        data[0, 1] = np.nan
    expected = getattr(upstream, name)(data, axis=1)
    actual = getattr(mb, name)(data, axis=1)
    assert actual.dtype == expected.dtype
    assert np.allclose(actual, expected, equal_nan=True)


@pytest.mark.parametrize(
    "name", ["move_sum", "move_mean", "move_var", "move_std", "move_min", "move_max"]
)
@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("min_count", [None, 1, 3])
def test_move_parity(sample, name, axis, min_count):
    kwargs = dict(window=4, min_count=min_count, axis=axis)
    if name in {"move_var", "move_std"}:
        kwargs["ddof"] = 1
    expected = getattr(upstream, name)(sample, **kwargs)
    actual = getattr(mb, name)(sample, **kwargs)
    assert np.allclose(actual, expected, equal_nan=True, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("dtype", [np.int32, np.float16, np.float32, np.float64])
@pytest.mark.parametrize(
    "name", ["move_sum", "move_mean", "move_var", "move_std", "move_min", "move_max"]
)
def test_move_dtype(dtype, name):
    data = np.arange(30, dtype=dtype)
    expected = getattr(upstream, name)(data, 5, min_count=2)
    actual = getattr(mb, name)(data, 5, min_count=2)
    assert actual.dtype == expected.dtype
    assert np.allclose(actual, expected, equal_nan=True, rtol=2e-4)


def test_strided_and_non_last_axis():
    data = np.arange(240.0).reshape(12, 20)[::2, ::3]
    data[2, 2] = np.nan
    for name in ("move_sum", "move_mean", "move_min", "move_max"):
        expected = getattr(upstream, name)(data, 3, min_count=1, axis=0)
        actual = getattr(mb, name)(data, 3, min_count=1, axis=0)
        assert np.allclose(actual, expected, equal_nan=True)


def test_large_offset_moving_variance_is_stable():
    rng = np.random.default_rng(7)
    data = 1e9 + rng.normal(scale=0.1, size=5000)
    expected = upstream.move_var(data, 101, min_count=50, ddof=1)
    actual = mb.move_var(data, 101, min_count=50, ddof=1)
    assert np.allclose(actual, expected, equal_nan=True, rtol=3e-5, atol=3e-7)
    assert np.nanmin(actual) >= 0


@pytest.mark.parametrize("size", [5, 7, 33])
@pytest.mark.parametrize("name", ["nansum", "nanmean", "nanvar", "nanstd"])
def test_reduction_simd_tail(name, size):
    data = np.linspace(-3.0, 5.0, size)
    data[1::4] = np.nan
    expected = getattr(upstream, name)(data)
    actual = getattr(mb, name)(data)
    assert np.allclose(actual, expected, equal_nan=True, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("size", [262_143, 262_145])
@pytest.mark.parametrize("name", ["move_var", "move_min", "move_max"])
def test_moving_parallel_threshold(name, size):
    rng = np.random.default_rng(81)
    data = rng.normal(size=size)
    data[::23] = np.nan
    kwargs = dict(window=17, min_count=8)
    expected = getattr(upstream, name)(data, **kwargs)
    actual = getattr(mb, name)(data, **kwargs)
    assert np.allclose(actual, expected, equal_nan=True, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("name", ["nansum", "nanmean", "nanvar", "nanstd"])
def test_empty_reductions(name):
    data = np.empty((2, 0))
    expected = getattr(upstream, name)(data, axis=1)
    actual = getattr(mb, name)(data, axis=1)
    assert np.allclose(actual, expected, equal_nan=True)


@pytest.mark.parametrize("name", ["nanmin", "nanmax", "nanargmin", "nanargmax"])
def test_empty_extrema_raise(name):
    with pytest.raises(ValueError):
        getattr(mb, name)(np.array([]))


def test_empty_boolean_reductions():
    assert mb.anynan([]) is upstream.anynan([])
    assert mb.allnan([]) is upstream.allnan([])


@pytest.mark.parametrize("name", ["move_sum", "move_mean", "move_min", "move_max"])
def test_moving_validation(name):
    fn = getattr(mb, name)
    data = np.arange(5.0)
    with pytest.raises(ValueError):
        fn(data, 0)
    with pytest.raises(ValueError):
        fn(data, 6)
    with pytest.raises(ValueError):
        fn(data, 3, min_count=0)
    with pytest.raises(ValueError):
        fn(data, 3, min_count=4)


def test_scalar_and_axis_errors():
    assert mb.nanmean(np.array(3.0)) == upstream.nanmean(np.array(3.0))
    with pytest.raises(Exception):
        mb.nanmean(np.array(3.0), axis=0)
    with pytest.raises(Exception):
        mb.move_sum(np.arange(5.0), 2, axis=1)


def test_public_signatures_match_upstream():
    for name in mb.__all__:
        ours = str(inspect.signature(getattr(mb, name)))
        documented = (getattr(upstream, name).__doc__ or "").splitlines()[0]
        assert ours.replace(" ", "") == documented.removeprefix(name).replace(" ", "")


def test_shared_library_was_built():
    root = Path(__file__).resolve().parents[1]
    assert (root / "dist" / "libmojo-bottleneck.so").is_file()


def test_moving_empty_outer_dimension():
    data = np.empty((0, 5), dtype=np.float64)
    for name in ("move_sum", "move_mean", "move_var", "move_std", "move_min", "move_max"):
        expected = getattr(upstream, name)(data, 2, axis=1)
        actual = getattr(mb, name)(data, 2, axis=1)
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype


def test_moving_longdouble_is_not_silently_narrowed():
    data = np.arange(5, dtype=np.longdouble)
    for name in ("move_sum", "move_mean", "move_var", "move_std", "move_min", "move_max"):
        with pytest.raises(TypeError, match="without narrowing"):
            getattr(mb, name)(data, 2)


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("mb_reduce", (0, 0, 1, 1, 0, 0)),
        ("mb_nanmedian", (0, 0, 1, 1)),
        ("mb_move_sum_mean", (0, 0, 1, 1, 1, 1, 0)),
        ("mb_move_var", (0, 0, 1, 1, 1, 1, 0, 0)),
        ("mb_move_extreme", (0, 0, 0, 1, 1, 1, 1, 0)),
    ],
)
def test_ffi_rejects_null_buffers(name, args):
    with pytest.raises(RuntimeError, match="invalid buffer metadata"):
        checked_call(name, *args)
