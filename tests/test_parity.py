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
@pytest.mark.parametrize(
    "name", ["nansum", "nanmean", "nanvar", "nanstd", "nanmedian"]
)
def test_reduction_simd_tail(name, size):
    data = np.linspace(-3.0, 5.0, size)
    data[1::4] = np.nan
    expected = getattr(upstream, name)(data)
    actual = getattr(mb, name)(data)
    assert np.allclose(actual, expected, equal_nan=True, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("size", [262_143, 262_145])
@pytest.mark.parametrize(
    "name", ["move_sum", "move_mean", "move_var", "move_min", "move_max"]
)
def test_moving_parallel_threshold(name, size):
    rng = np.random.default_rng(81)
    data = rng.normal(size=size)
    data[::23] = np.nan
    kwargs = dict(window=17, min_count=8)
    expected = getattr(upstream, name)(data, **kwargs)
    actual = getattr(mb, name)(data, **kwargs)
    assert np.allclose(actual, expected, equal_nan=True, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize(
    "name", ["move_sum", "move_mean", "move_var", "move_std", "move_min", "move_max"]
)
def test_moving_kernels_write_invalid_outputs(name):
    data = np.array([np.nan, 2.0, np.nan, 4.0, np.nan])
    kwargs = dict(window=3, min_count=3)
    expected = getattr(upstream, name)(data, **kwargs)
    actual = getattr(mb, name)(data, **kwargs)
    assert np.allclose(actual, expected, equal_nan=True)


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


# ---------------------------------------------------------------------------
# Degenerate-input parity tests added for the TEST phase. Every comparison
# below calls the installed upstream `bottleneck` (1.6.0) with the same
# arguments and compares against the port with an explicit tolerance, or
# asserts the same exception class upstream raises. Cases where upstream
# itself is inconsistent (fast C path vs bn.slow/numpy) follow the installed
# bottleneck fast path, which is the stated reference; upstream test files
# cited in comments are bottleneck/tests/*.py from the installed package.
# ---------------------------------------------------------------------------

REDUCTIONS = [
    "nansum",
    "nanmean",
    "nanvar",
    "nanstd",
    "nanmin",
    "nanmax",
    "nanmedian",
    "nanargmin",
    "nanargmax",
    "anynan",
    "allnan",
]
MOVING = ["move_sum", "move_mean", "move_var", "move_std", "move_min", "move_max"]
TOL = dict(rtol=1e-12, atol=1e-14)


def _assert_parity(name, *args, **kwargs):
    expected = getattr(upstream, name)(*args, **kwargs)
    actual = getattr(mb, name)(*args, **kwargs)
    exp_arr = np.asarray(expected)
    act_arr = np.asarray(actual)
    assert act_arr.shape == exp_arr.shape
    if exp_arr.ndim:
        assert act_arr.dtype == exp_arr.dtype
    assert np.allclose(act_arr, exp_arr, equal_nan=True, **TOL)


def _assert_same_error(name, *args, **kwargs):
    with pytest.raises(Exception) as exp_info:
        getattr(upstream, name)(*args, **kwargs)
    with pytest.raises(type(exp_info.value)) as act_info:
        getattr(mb, name)(*args, **kwargs)
    assert type(act_info.value) is type(exp_info.value) or isinstance(
        act_info.value, type(exp_info.value)
    )


@pytest.mark.parametrize("name", REDUCTIONS)
@pytest.mark.parametrize(
    "data",
    [
        np.array([42.0]),
        np.array([np.nan]),
        np.array([np.inf]),
        np.array([-np.inf]),
        np.array([7.0, 7.0, 7.0, 7.0]),
        np.array([5.0, 5.0, np.nan, 5.0]),
        np.array([1.0, np.inf, 2.0]),
        np.array([1.0, -np.inf, 2.0]),
        np.array([np.inf, np.nan]),
        np.array([3.0, 1.0, 2.0]),
        np.array([4.0, 1.0, 3.0, 2.0]),
    ],
)
def test_reduction_one_element_duplicates_and_inf(name, data):
    # One element, duplicates (ties), even/odd counts, and +/-inf are the
    # fixture shapes in bottleneck/tests/util.py arrays().
    if name in {"nanargmin", "nanargmax"} and np.all(np.isnan(data)):
        with pytest.raises(ValueError, match="All-NaN"):
            getattr(upstream, name)(data)
        with pytest.raises(ValueError, match="All-NaN"):
            getattr(mb, name)(data)
    else:
        _assert_parity(name, data)


@pytest.mark.parametrize("name", REDUCTIONS)
@pytest.mark.parametrize(
    "data",
    [
        np.array([[9.0]]),
        np.array([[1.0, 2.0], [np.nan, np.nan]]),
        np.array([[np.nan, np.nan], [np.nan, np.nan]]),
        np.array([[3.0, 3.0], [3.0, 3.0]]),
        np.array([[np.inf, 1.0], [2.0, -np.inf]]),
        np.array([[2.0, 1.0, 1.0], [3.0, 3.0, 1.0]]),
    ],
)
@pytest.mark.parametrize("axis", [None, 0, 1, -1, -2])
def test_reduction_2d_degenerate_parity(name, data, axis):
    if name in {"nanargmin", "nanargmax"}:
        # All-NaN slices raise in upstream; compare the error class instead.
        try:
            getattr(upstream, name)(data, axis=axis)
        except ValueError:
            with pytest.raises(ValueError, match="All-NaN"):
                getattr(mb, name)(data, axis=axis)
            return
    _assert_parity(name, data, axis=axis)


@pytest.mark.parametrize("name", ["nanvar", "nanstd"])
@pytest.mark.parametrize("ddof", [-5, -1, 0, 1, 2, 10])
@pytest.mark.parametrize(
    "data",
    [
        np.array([]),
        np.array([np.nan]),
        np.array([np.nan, np.nan]),
        np.array([1.0]),
        np.array([1.0, 2.0]),
        np.array([1.0, 2.0, 3.0]),
    ],
)
def test_variance_ddof_edge_parity(name, data, ddof):
    # Upstream fast and slow agree here: count==0 with ddof<0 yields 0.0
    # (bottleneck.nanvar([], ddof=-1) == 0.0); count-ddof<=0 with count>0
    # yields nan. The port previously returned nan for the count==0 case.
    _assert_parity(name, data, ddof=ddof)


@pytest.mark.parametrize(
    "name", ["nansum", "nanmean", "nanvar", "nanstd", "nanmedian", "anynan", "allnan"]
)
@pytest.mark.parametrize(
    ("shape", "axis"),
    [
        ((0,), None),
        ((0,), 0),
        ((2, 0), None),
        ((2, 0), 1),
        ((2, 0), 0),
        ((0, 5), None),
        ((0, 5), 0),
        ((0, 5), 1),
        ((0, 0), None),
        ((0, 0), 0),
        ((0, 0), 1),
        ((1, 2, 0), None),
    ],
)
def test_empty_reductions_parity(name, shape, axis):
    # Empty inputs: upstream returns 0.0/nan/False/True/empty arrays here
    # (never raises for these functions). Compare shape, dtype, and values.
    _assert_parity(name, np.empty(shape), axis=axis)


@pytest.mark.parametrize("name", ["nanmin", "nanmax", "nanargmin", "nanargmax"])
@pytest.mark.parametrize(
    ("shape", "axis"),
    [
        ((0,), None),
        ((0,), 0),
        ((2, 0), None),
        ((2, 0), 1),
        ((0, 5), 0),
        ((0, 0), 0),
        ((0, 0), 1),
        ((0, 0), None),
        ((1, 2, 0), None),
    ],
)
def test_empty_extrema_raise_parity(name, shape, axis):
    # Upstream bottleneck/tests/reduce_test.py test_nanmin_size_zero and
    # test_nanmax_size_zero: shapes (0,), (2, 0), (1, 2, 0) raise ValueError.
    # The same rule applies whenever the reduction axis has length 0, so
    # (0, 0) with axis=0/1 and (0, 5) with axis=0 raise too, while
    # (0, 5) axis=1 and (2, 0) axis=0 return an empty result.
    data = np.empty(shape)
    with pytest.raises(ValueError):
        getattr(upstream, name)(data, axis=axis)
    with pytest.raises(ValueError):
        getattr(mb, name)(data, axis=axis)


@pytest.mark.parametrize("name", ["nanmin", "nanmax", "nanargmin", "nanargmax"])
@pytest.mark.parametrize(
    ("shape", "axis"), [((0, 5), 1), ((2, 0), 0), ((0, 5), -1), ((2, 0), -2)]
)
def test_empty_outer_extrema_return_empty(name, shape, axis):
    # A zero-length outer dimension does not raise: the reduction axis is
    # non-empty, so upstream returns an empty array of the right shape.
    _assert_parity(name, np.empty(shape), axis=axis)


@pytest.mark.parametrize("name", ["nanargmin", "nanargmax"])
@pytest.mark.parametrize(
    "data",
    [
        np.array([2.0, 1.0, 1.0, 3.0]),
        np.array([np.nan, 1.0, 1.0]),
        np.array([3.0, 3.0, np.nan, 3.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 1.0]),
    ],
)
def test_arg_reduction_duplicates_first_index(name, data):
    # Ties resolve to the first index in upstream (bottleneck/tests/util.py
    # ties fixtures [0, 0, 0] and [1, 1, 1]).
    _assert_parity(name, data)
    assert getattr(mb, name)(data) == getattr(upstream, name)(data)


@pytest.mark.parametrize("name", ["nanargmin", "nanargmax"])
@pytest.mark.parametrize(
    "data",
    [
        np.array([np.nan]),
        np.array([np.nan, np.nan]),
        np.array([[1.0, 2.0], [np.nan, np.nan]]),
    ],
)
def test_arg_reduction_all_nan_raises(name, data):
    with pytest.raises(ValueError, match="All-NaN"):
        getattr(upstream, name)(data, axis=1 if data.ndim == 2 else None)
    with pytest.raises(ValueError, match="All-NaN"):
        getattr(mb, name)(data, axis=1 if data.ndim == 2 else None)


@pytest.mark.parametrize("name", REDUCTIONS)
@pytest.mark.parametrize(
    "data",
    [
        [1.0, 2.0, np.nan],
        [1, 2, 3],
        -9,
        0,
        9.0,
        np.array(-9),
        np.array(9, dtype=np.int64),
        np.array(3.0),
    ],
)
def test_scalar_and_list_input_parity(name, data):
    # bottleneck/tests/scalar_input_test.py and list_input_test.py: scalar
    # and list inputs are valid for reductions. Values are compared with
    # tolerance; scalar Python type (int vs np.int64) is not part of the
    # upstream comparison (upstream itself returns Python scalars for
    # float64/int64 inputs).
    _assert_parity(name, data)


@pytest.mark.parametrize("name", REDUCTIONS)
@pytest.mark.parametrize("axis", [0, -1])
def test_zero_d_axis_raises_parity(name, axis):
    # 0-d input with an explicit axis is a mismatched shape: upstream
    # raises ValueError for every reduction (np.array(9), axis=0/-1).
    with pytest.raises(ValueError):
        getattr(upstream, name)(np.array(9.0), axis=axis)
    with pytest.raises(ValueError):
        getattr(mb, name)(np.array(9.0), axis=axis)
    with pytest.raises(ValueError):
        getattr(upstream, name)(np.array(9), axis=axis)
    with pytest.raises(ValueError):
        getattr(mb, name)(np.array(9), axis=axis)


@pytest.mark.parametrize("name", REDUCTIONS)
@pytest.mark.parametrize("axis", [1, 5, -5])
def test_reduction_axis_out_of_bounds(name, axis):
    with pytest.raises(ValueError):
        getattr(upstream, name)(np.array([1.0, 2.0]), axis=axis)
    with pytest.raises(ValueError):
        getattr(mb, name)(np.array([1.0, 2.0]), axis=axis)


@pytest.mark.parametrize("name", REDUCTIONS)
@pytest.mark.parametrize("axis", ["0", 1.5, True, (0,), (0, 1)])
def test_reduction_axis_type_raises(name, axis):
    # bottleneck/tests/reduce_test.py test_arg_parse_raises: axis="0" is a
    # TypeError; bools/floats/tuples are likewise not integers.
    with pytest.raises(TypeError):
        getattr(upstream, name)(np.array([1.0, 2.0]), axis=axis)
    with pytest.raises(TypeError):
        getattr(mb, name)(np.array([1.0, 2.0]), axis=axis)


@pytest.mark.parametrize("name", ["nanvar", "nanstd"])
@pytest.mark.parametrize("ddof", [0.5, "a", None, True])
def test_variance_ddof_type_raises(name, ddof):
    with pytest.raises(TypeError):
        getattr(upstream, name)(np.array([1.0, 2.0]), ddof=ddof)
    with pytest.raises(TypeError):
        getattr(mb, name)(np.array([1.0, 2.0]), ddof=ddof)


@pytest.mark.parametrize("name", REDUCTIONS)
def test_reduction_arg_parsing_parity(name):
    # bottleneck/tests/reduce_test.py test_arg_parsing: positional axis,
    # axis=None, and a=a keyword forms all work; extra/missing arguments
    # raise TypeError (test_arg_parse_raises).
    a = np.array([1.0, 2.0, 3.0])
    _assert_parity(name, a, 0)
    _assert_parity(name, a, None)
    _assert_parity(name, a=a)
    with pytest.raises(TypeError):
        getattr(upstream, name)()
    with pytest.raises(TypeError):
        getattr(mb, name)()
    with pytest.raises(TypeError):
        getattr(upstream, name)(a, axis=0, extra=0)
    with pytest.raises(TypeError):
        getattr(mb, name)(a, axis=0, extra=0)
    if name not in {"nanvar", "nanstd"}:
        with pytest.raises(TypeError):
            getattr(upstream, name)(a, ddof=0)
        with pytest.raises(TypeError):
            getattr(mb, name)(a, ddof=0)


@pytest.mark.parametrize("name", MOVING)
@pytest.mark.parametrize(
    "data",
    [
        np.array([42.0]),
        np.array([np.nan]),
        np.array([7.0, 7.0, 7.0, 7.0, 7.0]),
        np.array([5.0, 5.0, np.nan, 5.0, 5.0]),
        np.array([np.nan, np.nan, np.nan]),
        np.array([1.0, 2.0]),
        np.array([[1.0, 2.0, 3.0], [4.0, np.nan, 6.0]]),
    ],
)
@pytest.mark.parametrize("window", [1, 2, 3])
def test_moving_degenerate_parity(name, data, window):
    if window > data.shape[-1]:
        pytest.skip("window larger than axis is covered by validation tests")
    for min_count in (None, 1):
        _assert_parity(name, data, window, min_count=min_count)


@pytest.mark.parametrize("name", MOVING)
def test_moving_window_equals_length(name):
    data = np.array([3.0, 1.0, np.nan, 4.0])
    _assert_parity(name, data, 4)
    _assert_parity(name, data, 4, min_count=1)
    _assert_parity(name, data, 1, min_count=1)


@pytest.mark.parametrize("name", MOVING)
@pytest.mark.parametrize(
    "data",
    [
        np.array([1.0, np.inf, 2.0, 3.0, 4.0]),
        np.array([np.inf, 1.0, 2.0, 3.0]),
        np.array([1.0, 2.0, np.inf, 4.0, 5.0, 6.0]),
        np.array([1.0, -np.inf, 2.0, 3.0]),
    ],
)
@pytest.mark.parametrize("window", [1, 2, 3])
def test_moving_inf_parity(name, data, window):
    # Infinities entering a rolling accumulator poison it in upstream
    # (README: the port retains this limitation); move_min/move_max track
    # the inf value itself. Upstream's own fixture generator substitutes
    # the value 8 for inf for move_sum/mean/var/std (util.py), but the
    # installed fast path defines the behaviour compared here.
    _assert_parity(name, data, window, min_count=1)


@pytest.mark.parametrize("name", ["move_var", "move_std"])
@pytest.mark.parametrize("ddof", [-1, 0, 1, 2, 3, 10])
def test_move_variance_ddof_parity(name, ddof):
    # Installed bottleneck move_var/move_std divide by (count-ddof) even
    # when it is zero or negative (e.g. ddof=10 on a 3-wide window yields
    # -0.2857 for move_var), unlike the per-window bn.slow/numpy reference
    # which yields nan. The installed fast path is the reference here.
    data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    _assert_parity(name, data, 3, min_count=1, ddof=ddof)


@pytest.mark.parametrize("name", MOVING)
@pytest.mark.parametrize(
    ("window", "min_count", "axis"),
    [
        (0, None, -1),
        (-1, None, -1),
        (6, None, -1),
        (3, 0, -1),
        (3, -1, -1),
        (3, 4, -1),
        (3, None, 1),
        (3, None, 5),
        (3, None, -5),
        (3, None, None),
    ],
)
def test_moving_validation_parity(name, window, min_count, axis):
    # Mismatched shapes for moving windows: window/min_count/axis ranges
    # are ValueErrors in upstream (bottleneck/slow/move.py validation and
    # the fast path use the same classes), axis=None is a TypeError.
    data = np.arange(5.0)
    kwargs = dict(min_count=min_count, axis=axis)
    with pytest.raises(Exception) as exp_info:
        getattr(upstream, name)(data, window, **kwargs)
    with pytest.raises(type(exp_info.value)):
        getattr(mb, name)(data, window, **kwargs)


@pytest.mark.parametrize("name", MOVING)
@pytest.mark.parametrize("axis", ["0", 0.5, True, (0,)])
def test_moving_axis_type_raises(name, axis):
    # bottleneck/tests/move_test.py test_arg_parse_raises.
    with pytest.raises(TypeError):
        getattr(upstream, name)(np.array([1.0, 2.0, 3.0]), 2, axis=axis)
    with pytest.raises(TypeError):
        getattr(mb, name)(np.array([1.0, 2.0, 3.0]), 2, axis=axis)


@pytest.mark.parametrize("name", MOVING)
@pytest.mark.parametrize("window", [2.0, "2", True, None])
def test_moving_window_type_raises(name, window):
    with pytest.raises(TypeError):
        getattr(upstream, name)(np.arange(5.0), window)
    with pytest.raises(TypeError):
        getattr(mb, name)(np.arange(5.0), window)


@pytest.mark.parametrize("name", MOVING)
@pytest.mark.parametrize("min_count", [1.0, "1", True])
def test_moving_min_count_type_raises(name, min_count):
    with pytest.raises(TypeError):
        getattr(upstream, name)(np.arange(5.0), 2, min_count=min_count)
    with pytest.raises(TypeError):
        getattr(mb, name)(np.arange(5.0), 2, min_count=min_count)


@pytest.mark.parametrize("name", ["move_var", "move_std"])
@pytest.mark.parametrize("ddof", [0.5, "a", None, True])
def test_move_variance_ddof_type_raises(name, ddof):
    with pytest.raises(TypeError):
        getattr(upstream, name)(np.arange(5.0), 2, ddof=ddof)
    with pytest.raises(TypeError):
        getattr(mb, name)(np.arange(5.0), 2, ddof=ddof)


@pytest.mark.parametrize("name", MOVING)
def test_moving_scalar_raises_parity(name):
    # bottleneck/slow/move.py move_func: "moving window functions
    # require ndim > 0" (ValueError) for 0-d input.
    with pytest.raises(ValueError, match="ndim > 0"):
        getattr(upstream, name)(np.array(3.0), 1)
    with pytest.raises(ValueError, match="ndim > 0"):
        getattr(mb, name)(np.array(3.0), 1)


@pytest.mark.parametrize("name", MOVING)
def test_moving_arg_parsing_parity(name):
    # bottleneck/tests/move_test.py test_arg_parsing: positional and
    # keyword (window/min_count/axis/ddof) forms; extra args raise TypeError.
    a = np.array([1.0, 2.0, 3.0])
    _assert_parity(name, a, 2, 1)
    _assert_parity(name, a, window=2)
    _assert_parity(name, a, window=2, min_count=1, axis=0)
    _assert_parity(name, a, min_count=1, window=2, axis=0)
    if name in {"move_var", "move_std"}:
        _assert_parity(name, a, 2, 1, -1, ddof=1)
    with pytest.raises(TypeError):
        getattr(upstream, name)()
    with pytest.raises(TypeError):
        getattr(mb, name)()
    with pytest.raises(TypeError):
        getattr(upstream, name)(a, 2, axis=0, extra=0)
    with pytest.raises(TypeError):
        getattr(mb, name)(a, 2, axis=0, extra=0)
    if name not in {"move_var", "move_std"}:
        with pytest.raises(TypeError):
            getattr(upstream, name)(a, 2, ddof=0)
        with pytest.raises(TypeError):
            getattr(mb, name)(a, 2, ddof=0)


@pytest.mark.parametrize("name", MOVING)
@pytest.mark.parametrize("axis", [0, 1, -1, -2])
def test_moving_2d_axis_parity(name, axis):
    data = np.array([[1.0, 2.0, 3.0], [4.0, np.nan, 6.0]])
    _assert_parity(name, data, 2, min_count=1, axis=axis)


@pytest.mark.parametrize("name", MOVING)
@pytest.mark.parametrize(
    "data",
    [
        np.array([np.inf, np.nan]),
        np.array([np.inf, -np.inf]),
        np.array([np.nan, 2.0, 3.0]),
        np.array([-np.inf, 2.0, 3.0]),
        np.array([1, 2, 3], dtype=np.float16),
        np.array([1, 2, 3]) + 1e9,
        np.array([0, 0, 0]),
        np.array([1, 1, 1]),
        np.array([1.0, np.nan, np.nan, 2.0]),
        np.array([1, 2, 3], dtype=">f4"),
    ],
)
def test_upstream_fixture_arrays_parity(name, data):
    # Fixture arrays copied from bottleneck/tests/util.py arrays():
    # inf/nan pairs, float16, large-offset (1e9) robustness, ties, and
    # byte-swapped float32. Moving variance/std are compared on the same
    # fixtures where upstream defines them (see test_moving_inf_parity).
    _assert_parity(name, data, 2, min_count=1)


@pytest.mark.parametrize("name", REDUCTIONS)
@pytest.mark.parametrize(
    "dtype", [np.int8, np.int32, np.uint8, np.float16, np.float32, np.longdouble]
)
def test_reduction_scalar_dtype_values_parity(name, dtype):
    if np.issubdtype(dtype, np.floating):
        data = np.array([1.0, np.nan, 3.0], dtype=dtype)
    else:
        data = np.array([1, 2, 3], dtype=dtype)
    _assert_parity(name, data)


@pytest.mark.parametrize("name", REDUCTIONS)
@pytest.mark.parametrize(
    "make",
    [
        lambda: np.arange(12.0).reshape(4, 3)[::2],
        lambda: np.arange(12.0).reshape(4, 3)[:, ::2],
        lambda: np.arange(12.0).reshape(4, 3)[::2][:, ::2],
        lambda: np.asfortranarray(np.arange(12.0).reshape(4, 3)),
        lambda: np.arange(12.0).reshape(4, 3)[:, ::-1],
    ],
)
def test_noncontiguous_reduction_parity(name, make):
    # Non-contiguous/strided/Fortran/reversed inputs (util.py fixtures).
    data = make()
    data.flat[0] = np.nan
    _assert_parity(name, data, axis=1)


@pytest.mark.parametrize("name", MOVING)
@pytest.mark.parametrize(
    "make",
    [
        lambda: np.arange(12.0).reshape(4, 3)[::2],
        lambda: np.arange(12.0).reshape(4, 3)[:, ::2],
        lambda: np.asfortranarray(np.arange(12.0).reshape(4, 3)),
    ],
)
def test_noncontiguous_moving_parity(name, make):
    data = make()
    data.flat[0] = np.nan
    _assert_parity(name, data, 2, min_count=1, axis=1)


@pytest.mark.parametrize("name", REDUCTIONS + MOVING)
def test_input_not_modified_parity(name):
    # bottleneck/tests/input_modification_test.py: no covered function
    # modifies its input (nanmedian compacts a private copy).
    data = np.array([3.0, 1.0, np.nan, 2.0])
    original = data.copy()
    if name in MOVING:
        getattr(mb, name)(data, 3)
    else:
        getattr(mb, name)(data)
    assert np.array_equal(data, original, equal_nan=True)


def test_moving_empty_axis_raises_parity():
    # A zero-length moving axis is a mismatched shape: window must be
    # between 1 and 0, so every window raises ValueError in upstream.
    for name in MOVING:
        for data in (np.array([]), np.empty((3, 0))):
            with pytest.raises(ValueError):
                getattr(upstream, name)(data, 2)
            with pytest.raises(ValueError):
                getattr(mb, name)(data, 2)
