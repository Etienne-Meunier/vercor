from __future__ import annotations

from typing import Any

import numpy as np


def _format_index(shape: tuple[int, ...], flat_index: int) -> str:
    if not shape:
        return "scalar"
    unraveled = np.unravel_index(flat_index, shape)
    return str(tuple(int(index) for index in unraveled))


def assert_allclose_compact(
    actual: Any,
    expected: Any,
    *,
    rtol: float = 1e-7,
    atol: float = 0.0,
    equal_nan: bool = True,
    label: str = "array",
) -> None:
    """Assert numerical closeness with concise, greppable diagnostics."""

    actual_arr = np.asarray(actual)
    expected_arr = np.asarray(expected)

    if actual_arr.shape != expected_arr.shape:
        if expected_arr.ndim == 0:
            expected_arr = np.broadcast_to(expected_arr, actual_arr.shape)
        elif actual_arr.ndim == 0:
            actual_arr = np.broadcast_to(actual_arr, expected_arr.shape)
        else:
            raise AssertionError(
                f"{label} shape mismatch: actual {actual_arr.shape}, expected {expected_arr.shape}"
            )

    close = np.isclose(
        actual_arr, expected_arr, rtol=rtol, atol=atol, equal_nan=equal_nan
    )
    if np.all(close):
        return

    with np.errstate(divide="ignore", invalid="ignore"):
        abs_error = np.abs(actual_arr - expected_arr)
        abs_error = np.nan_to_num(abs_error, nan=np.inf, posinf=np.inf, neginf=np.inf)
        scale = np.maximum(np.abs(expected_arr), atol)
        rel_error = abs_error / np.where(scale > 0.0, scale, 1.0)
        rel_error = np.nan_to_num(rel_error, nan=np.inf, posinf=np.inf, neginf=np.inf)

    failing = ~close.ravel()
    flat_abs_error = abs_error.ravel()
    failing_abs = np.where(failing, flat_abs_error, -np.inf)
    flat_index = int(np.argmax(failing_abs))
    max_abs_error = float(flat_abs_error[flat_index])
    max_rel_error = float(np.max(rel_error.ravel()[failing]))
    actual_value = actual_arr.ravel()[flat_index]
    expected_value = expected_arr.ravel()[flat_index]

    raise AssertionError(
        f"{label} mismatch: max abs err {max_abs_error:.3e}, "
        f"max rel err {max_rel_error:.3e} at index {_format_index(actual_arr.shape, flat_index)}; "
        f"expected {expected_value!r}, got {actual_value!r}"
    )


def assert_array_equal_compact(
    actual: Any,
    expected: Any,
    *,
    equal_nan: bool = True,
    label: str = "array",
) -> None:
    """Assert exact equality with the same compact diagnostics."""

    actual_arr = np.asarray(actual)
    expected_arr = np.asarray(expected)

    if np.array_equal(actual_arr, expected_arr, equal_nan=equal_nan):
        return

    assert_allclose_compact(
        actual_arr,
        expected_arr,
        rtol=0.0,
        atol=0.0,
        equal_nan=equal_nan,
        label=label,
    )
