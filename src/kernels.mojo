"""Float64 kernels exported through a small C ABI."""

from std.math import isnan, sqrt
from std.sys import simd_width_of

comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int64, AnyOrigin[mut=True]]
comptime PARALLEL_THRESHOLD = 262144
comptime PARALLEL_CHUNK = 262144


def fptr(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def iptr(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def reduce_kernel(src: FPtr, dst: FPtr, outer: Int, n: Int, op: Int, ddof: Int):
    comptime W = simd_width_of[DType.float64]()
    for row in range(outer):
        var base = row * n
        var acc = 0.0
        var count = 0
        if op == 6 or op == 7:
            var answer = op == 7
            for j in range(n):
                if isnan(src[unsafe_offset=base + j]):
                    if op == 6:
                        answer = True
                        break
                elif op == 7:
                    answer = False
                    break
            dst[unsafe_offset=row] = 1.0 if answer else 0.0
            continue
        if op == 4 or op == 5:
            for j in range(n):
                var v = src[unsafe_offset=base + j]
                if isnan(v):
                    continue
                if count == 0:
                    acc = v
                elif op == 4 and v < acc:
                    acc = v
                elif op == 5 and v > acc:
                    acc = v
                count += 1
            if count > 0:
                dst[unsafe_offset=row] = acc
            continue

        var vector_acc = SIMD[DType.float64, W](0.0)
        var j = 0
        while j + W <= n:
            var values = src.unsafe_load[width=W](base + j)
            var valid = ~isnan(values)
            vector_acc += valid.select(values, 0.0)
            count += Int(valid.cast[DType.int64]().reduce_add())
            j += W
        acc = vector_acc.reduce_add()
        while j < n:
            var v = src[unsafe_offset=base + j]
            if not isnan(v):
                acc += v
                count += 1
            j += 1
        if op == 0:
            dst[unsafe_offset=row] = acc
        elif op == 1:
            if count > 0:
                dst[unsafe_offset=row] = acc / Float64(count)
        elif (op == 2 or op == 3) and count > 0 and count - ddof > 0:
            var mean = acc / Float64(count)
            var squared = 0.0
            var vector_squared = SIMD[DType.float64, W](0.0)
            j = 0
            while j + W <= n:
                var values = src.unsafe_load[width=W](base + j)
                var valid = ~isnan(values)
                var delta = valid.select(values - mean, 0.0)
                vector_squared += delta * delta
                j += W
            squared = vector_squared.reduce_add()
            while j < n:
                var v = src[unsafe_offset=base + j]
                if not isnan(v):
                    var delta = v - mean
                    squared += delta * delta
                j += 1
            var variance = squared / Float64(count - ddof)
            dst[unsafe_offset=row] = sqrt(variance) if op == 3 else variance


def select_value(src: FPtr, base: Int, n: Int, kth: Int) -> Float64:
    var left = 0
    var right = n - 1
    while left < right:
        var pivot = src[unsafe_offset=base + (left + right) // 2]
        var i = left
        var j = right
        while i <= j:
            while src[unsafe_offset=base + i] < pivot:
                i += 1
            while src[unsafe_offset=base + j] > pivot:
                j -= 1
            if i <= j:
                var swap = src[unsafe_offset=base + i]
                src[unsafe_offset=base + i] = src[unsafe_offset=base + j]
                src[unsafe_offset=base + j] = swap
                i += 1
                j -= 1
        if kth <= j:
            right = j
        elif kth >= i:
            left = i
        else:
            break
    return src[unsafe_offset=base + kth]


def nanmedian_kernel(src: FPtr, dst: FPtr, outer: Int, n: Int):
    for row in range(outer):
        var base = row * n
        var count = 0
        for j in range(n):
            var v = src[unsafe_offset=base + j]
            if not isnan(v):
                src[unsafe_offset=base + count] = v
                count += 1
        if count == 0:
            continue
        var high = select_value(src, base, count, count // 2)
        if count % 2 == 0:
            var low = select_value(src, base, count, count // 2 - 1)
            dst[unsafe_offset=row] = low * 0.5 + high * 0.5
        else:
            dst[unsafe_offset=row] = high


def arg_reduce_kernel(
    src: FPtr, dst: IPtr, outer: Int, n: Int, find_max: Bool
) -> Int:
    var invalid = 0
    for row in range(outer):
        var base = row * n
        var found = False
        var best = 0.0
        var best_index = 0
        for j in range(n):
            var v = src[unsafe_offset=base + j]
            if isnan(v):
                continue
            if (
                not found
                or (find_max and v > best)
                or (not find_max and v < best)
            ):
                found = True
                best = v
                best_index = j
        if found:
            dst[unsafe_offset=row] = Int64(best_index)
        else:
            invalid += 1
    return invalid


def move_sum_mean_range[
    mean_mode: Bool
](
    src: FPtr,
    dst: FPtr,
    n: Int,
    row: Int,
    start: Int,
    end: Int,
    window: Int,
    min_count: Int,
):
    var base = row * n
    var acc = 0.0
    var count = 0
    var initial = max(0, start - window)
    for j in range(initial, start):
        var value = src[unsafe_offset=base + j]
        if not isnan(value):
            acc += value
            count += 1
    for j in range(start, end):
        var entering = src[unsafe_offset=base + j]
        if not isnan(entering):
            acc += entering
            count += 1
        if j >= window:
            var leaving = src[unsafe_offset=base + j - window]
            if not isnan(leaving):
                acc -= leaving
                count -= 1
        if count >= min_count:
            comptime if mean_mode:
                if count > 0:
                    dst[unsafe_offset=base + j] = acc / Float64(count)
            else:
                dst[unsafe_offset=base + j] = acc


def move_sum_mean_kernel(
    src: FPtr,
    dst: FPtr,
    outer: Int,
    n: Int,
    window: Int,
    min_count: Int,
    mean_mode: Bool,
):
    if mean_mode:
        for row in range(outer):
            move_sum_mean_range[True](src, dst, n, row, 0, n, window, min_count)
    else:
        for row in range(outer):
            move_sum_mean_range[False](
                src, dst, n, row, 0, n, window, min_count
            )


def move_var_range[
    std_mode: Bool
](
    src: FPtr,
    dst: FPtr,
    n: Int,
    row: Int,
    start: Int,
    end: Int,
    window: Int,
    min_count: Int,
    ddof: Int,
):
    var base = row * n
    var count = 0
    var mean = 0.0
    var m2 = 0.0
    var initial = max(0, start - window)
    for j in range(initial, start):
        var value = src[unsafe_offset=base + j]
        if not isnan(value):
            count += 1
            var delta = value - mean
            mean += delta / Float64(count)
            m2 += delta * (value - mean)
    for j in range(start, end):
        if j >= window:
            var leaving = src[unsafe_offset=base + j - window]
            if not isnan(leaving):
                if count == 1:
                    count = 0
                    mean = 0.0
                    m2 = 0.0
                else:
                    var new_count = count - 1
                    var delta = leaving - mean
                    mean -= delta / Float64(new_count)
                    m2 -= delta * (leaving - mean)
                    count = new_count
        var entering = src[unsafe_offset=base + j]
        if not isnan(entering):
            count += 1
            var delta = entering - mean
            mean += delta / Float64(count)
            m2 += delta * (entering - mean)
        if count >= min_count and count > 0 and count - ddof > 0:
            var variance = m2 / Float64(count - ddof)
            if variance < 0.0:
                variance = 0.0
            comptime if std_mode:
                dst[unsafe_offset=base + j] = sqrt(variance)
            else:
                dst[unsafe_offset=base + j] = variance


def move_var_impl[
    std_mode: Bool
](
    src: FPtr,
    dst: FPtr,
    outer: Int,
    n: Int,
    window: Int,
    min_count: Int,
    ddof: Int,
):
    var chunks_per_row = (n + PARALLEL_CHUNK - 1) // PARALLEL_CHUNK
    var tasks = outer * chunks_per_row
    if outer * n < PARALLEL_THRESHOLD or tasks == 1:
        for row in range(outer):
            move_var_range[std_mode](
                src, dst, n, row, 0, n, window, min_count, ddof
            )
        return

    def work(task: Int) capturing:
        var row = task // chunks_per_row
        var chunk = task - row * chunks_per_row
        var start = chunk * PARALLEL_CHUNK
        var end = min(n, start + PARALLEL_CHUNK)
        move_var_range[std_mode](
            src, dst, n, row, start, end, window, min_count, ddof
        )

    for task in range(tasks):
        work(task)


def move_var_kernel(
    src: FPtr,
    dst: FPtr,
    outer: Int,
    n: Int,
    window: Int,
    min_count: Int,
    ddof: Int,
    std_mode: Bool,
):
    if std_mode:
        move_var_impl[True](src, dst, outer, n, window, min_count, ddof)
    else:
        move_var_impl[False](src, dst, outer, n, window, min_count, ddof)


def move_extreme_range[
    find_max: Bool
](
    src: FPtr,
    dst: FPtr,
    scratch: IPtr,
    n: Int,
    start: Int,
    end: Int,
    window: Int,
    min_count: Int,
    scratch_base: Int,
):
    var head = 0
    var tail = 0
    var count = 0
    var initial = max(0, start - window)
    for j in range(initial, start):
        var entering = src[unsafe_offset=j]
        if not isnan(entering):
            count += 1
            while head < tail:
                var last = Int(
                    scratch[unsafe_offset=scratch_base + (tail - 1) % window]
                )
                var last_value = src[unsafe_offset=last]
                comptime if find_max:
                    if last_value > entering:
                        break
                else:
                    if last_value < entering:
                        break
                tail -= 1
            scratch[unsafe_offset=scratch_base + tail % window] = Int64(j)
            tail += 1
    for j in range(start, end):
        if j >= window and not isnan(src[unsafe_offset=j - window]):
            count -= 1
        while (
            head < tail
            and Int(scratch[unsafe_offset=scratch_base + head % window])
            <= j - window
        ):
            head += 1
        var entering = src[unsafe_offset=j]
        if not isnan(entering):
            count += 1
            while head < tail:
                var last = Int(
                    scratch[unsafe_offset=scratch_base + (tail - 1) % window]
                )
                var last_value = src[unsafe_offset=last]
                comptime if find_max:
                    if last_value > entering:
                        break
                else:
                    if last_value < entering:
                        break
                tail -= 1
            scratch[unsafe_offset=scratch_base + tail % window] = Int64(j)
            tail += 1
        if count >= min_count and head < tail:
            dst[unsafe_offset=j] = src[
                unsafe_offset=Int(
                    scratch[unsafe_offset=scratch_base + head % window]
                )
            ]


def move_extreme_impl[
    find_max: Bool
](
    src: FPtr,
    dst: FPtr,
    scratch: IPtr,
    outer: Int,
    n: Int,
    window: Int,
    min_count: Int,
):
    var tasks = (n + PARALLEL_CHUNK - 1) // PARALLEL_CHUNK
    if (
        outer == 1
        and n >= PARALLEL_THRESHOLD
        and tasks > 1
        and tasks * window <= n
    ):

        def work(task: Int) capturing:
            var start = task * PARALLEL_CHUNK
            var end = min(n, start + PARALLEL_CHUNK)
            move_extreme_range[find_max](
                src,
                dst,
                scratch,
                n,
                start,
                end,
                window,
                min_count,
                task * window,
            )

        for task in range(tasks):
            work(task)
        return
    for row in range(outer):
        var base = row * n
        var head = 0
        var tail = 0
        var count = 0
        for j in range(n):
            if j >= window and not isnan(src[unsafe_offset=base + j - window]):
                count -= 1
            while (
                head < tail and Int(scratch[unsafe_offset=head]) <= j - window
            ):
                head += 1
            var entering = src[unsafe_offset=base + j]
            if not isnan(entering):
                count += 1
                while head < tail:
                    var last = Int(scratch[unsafe_offset=tail - 1])
                    var last_value = src[unsafe_offset=base + last]
                    comptime if find_max:
                        if last_value > entering:
                            break
                    else:
                        if last_value < entering:
                            break
                    tail -= 1
                scratch[unsafe_offset=tail] = Int64(j)
                tail += 1
            if count >= min_count and head < tail:
                dst[unsafe_offset=base + j] = src[
                    unsafe_offset=base + Int(scratch[unsafe_offset=head])
                ]


def move_extreme_kernel(
    src: FPtr,
    dst: FPtr,
    scratch: IPtr,
    outer: Int,
    n: Int,
    window: Int,
    min_count: Int,
    find_max: Bool,
):
    if find_max:
        move_extreme_impl[True](src, dst, scratch, outer, n, window, min_count)
    else:
        move_extreme_impl[False](src, dst, scratch, outer, n, window, min_count)


@export("mb_reduce")
def mb_reduce(
    src: Int, dst: Int, outer: Int, n: Int, op: Int, ddof: Int
) abi("C") -> Int:
    if outer < 0 or n < 0 or op < 0 or op > 7:
        return -1
    if outer == 0:
        return 0
    if src == 0 or dst == 0:
        return -1
    reduce_kernel(fptr(src), fptr(dst), outer, n, op, ddof)
    return 0


@export("mb_arg_reduce")
def mb_arg_reduce(
    src: Int, dst: Int, outer: Int, n: Int, find_max: Int
) abi("C") -> Int:
    if outer < 0 or n < 0:
        return -1
    if outer == 0:
        return 0
    if src == 0 or dst == 0:
        return -1
    return arg_reduce_kernel(fptr(src), iptr(dst), outer, n, find_max != 0)


@export("mb_nanmedian")
def mb_nanmedian(src: Int, dst: Int, outer: Int, n: Int) abi("C") -> Int:
    if outer < 0 or n < 0:
        return -1
    if outer == 0:
        return 0
    if src == 0 or dst == 0:
        return -1
    nanmedian_kernel(fptr(src), fptr(dst), outer, n)
    return 0


@export("mb_move_sum_mean")
def mb_move_sum_mean(
    src: Int,
    dst: Int,
    outer: Int,
    n: Int,
    window: Int,
    min_count: Int,
    mean_mode: Int,
) abi("C") -> Int:
    if (
        outer < 0
        or n < 1
        or window < 1
        or window > n
        or min_count < 1
        or min_count > window
    ):
        return -1
    if outer == 0:
        return 0
    if src == 0 or dst == 0:
        return -1
    move_sum_mean_kernel(
        fptr(src), fptr(dst), outer, n, window, min_count, mean_mode != 0
    )
    return 0


@export("mb_move_var")
def mb_move_var(
    src: Int,
    dst: Int,
    outer: Int,
    n: Int,
    window: Int,
    min_count: Int,
    ddof: Int,
    std_mode: Int,
) abi("C") -> Int:
    if (
        outer < 0
        or n < 1
        or window < 1
        or window > n
        or min_count < 1
        or min_count > window
    ):
        return -1
    if outer == 0:
        return 0
    if src == 0 or dst == 0:
        return -1
    move_var_kernel(
        fptr(src), fptr(dst), outer, n, window, min_count, ddof, std_mode != 0
    )
    return 0


@export("mb_move_extreme")
def mb_move_extreme(
    src: Int,
    dst: Int,
    scratch: Int,
    outer: Int,
    n: Int,
    window: Int,
    min_count: Int,
    find_max: Int,
) abi("C") -> Int:
    if (
        outer < 0
        or n < 1
        or window < 1
        or window > n
        or min_count < 1
        or min_count > window
    ):
        return -1
    if outer == 0:
        return 0
    if src == 0 or dst == 0 or scratch == 0:
        return -1
    move_extreme_kernel(
        fptr(src),
        fptr(dst),
        iptr(scratch),
        outer,
        n,
        window,
        min_count,
        find_max != 0,
    )
    return 0
