"""Focused tests for the sole immutable output accumulator and layout helper."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.assertions import assert_allclose_compact
from vercor.output import OutputFrame, OutputVariable
from vercor.output._period import period_mean_sample_to_output_variable
from vercor.output._session import _OutputAccumulator


def _frame(
    values: object,
    *,
    dims: tuple[str, ...] = ("x",),
    sample_dimension: str | None = None,
) -> OutputFrame:
    return OutputFrame(
        {"temp": OutputVariable(dims, values, {"units": "K"})},
        sample_dimension=sample_dimension,
    )


def test_output_accumulator_is_an_immutable_jax_pytree() -> None:
    accumulator = _OutputAccumulator.zeros_from_frame(_frame(jnp.asarray([0.0, 0.0])))
    updated = jax.jit(lambda value: value.add_frame(_frame(jnp.asarray([1.0, 3.0]))))(
        accumulator
    )

    assert all(
        isinstance(leaf, jax.Array) for leaf in jax.tree_util.tree_leaves(updated)
    )
    assert_allclose_compact(updated.mean_frame().variables["temp"].values, [1.0, 3.0])
    with pytest.raises(AttributeError):
        updated.names = ("changed",)  # type: ignore[misc]


def test_output_accumulator_tracks_coordinate_values_as_dynamic_pytree_leaves() -> None:
    def coordinated_frame(coordinates: jax.Array) -> OutputFrame:
        return OutputFrame(
            {"temp": OutputVariable(("x",), jnp.asarray([1.0, 2.0]))},
            coordinates={"x": OutputVariable(("x",), coordinates)},
        )

    first = _OutputAccumulator.zeros_from_frame(
        coordinated_frame(jnp.asarray([0.0, 1.0]))
    )
    second = _OutputAccumulator.zeros_from_frame(
        coordinated_frame(jnp.asarray([2.0, 3.0]))
    )
    compiled_identity = jax.jit(lambda accumulator: accumulator)

    first_result = compiled_identity(first)
    second_result = compiled_identity(second)

    assert_allclose_compact(first_result.mean_frame().coordinates["x"].values, [0, 1])
    assert_allclose_compact(second_result.mean_frame().coordinates["x"].values, [2, 3])
    assert any(
        np.array_equal(np.asarray(leaf), np.asarray([2.0, 3.0]))
        for leaf in jax.tree_util.tree_leaves(second_result)
    )


def test_output_accumulator_canonicalizes_array_metadata_for_jit_reuse() -> None:
    def metadata_frame(value: float) -> OutputFrame:
        return OutputFrame(
            {
                "temp": OutputVariable(
                    ("x",),
                    jnp.asarray([1.0, 2.0]),
                    {"valid_range": np.asarray([0.0, value])},
                )
            },
            coordinates={
                "x": OutputVariable(
                    ("x",),
                    jnp.asarray([0.0, 1.0]),
                    {"bounds": np.asarray([0.0, value])},
                )
            },
            metadata={"coefficients": np.asarray([1.0, value])},
        )

    first = _OutputAccumulator.zeros_from_frame(metadata_frame(2.0))
    second = _OutputAccumulator.zeros_from_frame(metadata_frame(3.0))
    compiled_identity = jax.jit(lambda accumulator: accumulator)

    first_result = compiled_identity(first)
    second_result = compiled_identity(second)

    assert first_result.mean_frame().metadata["coefficients"] == (1.0, 2.0)
    assert second_result.mean_frame().metadata["coefficients"] == (1.0, 3.0)


def test_output_accumulator_preserves_nanmean_counts_without_mutation() -> None:
    empty = _OutputAccumulator.zeros_from_frame(_frame(np.asarray([0.0, 0.0, 0.0])))
    first = empty.add_frame(_frame(np.asarray([1.0, np.nan, np.nan])))
    second = first.add_frame(_frame(np.asarray([3.0, 5.0, np.nan])))

    assert_allclose_compact(empty.counts[0], np.asarray([0, 0, 0]))
    assert_allclose_compact(first.counts[0], np.asarray([1, 0, 0]))
    assert second.counts[0].dtype == jnp.int32
    assert_allclose_compact(second.counts[0], np.asarray([2, 1, 0]))
    assert_allclose_compact(
        second.mean_frame().variables["temp"].values,
        np.asarray([2.0, 5.0, np.nan]),
    )


def test_output_accumulator_reduces_named_sample_dimension() -> None:
    frame = _frame(
        np.asarray([[1.0, np.nan], [3.0, 5.0]]),
        dims=("time", "x"),
        sample_dimension="time",
    )
    accumulator = _OutputAccumulator.zeros_from_frame(frame).add_frame(frame)

    assert accumulator.dims == (("x",),)
    assert_allclose_compact(accumulator.counts[0], np.asarray([2, 1]))
    assert_allclose_compact(
        accumulator.mean_frame().variables["temp"].values,
        np.asarray([2.0, 5.0]),
    )


def test_output_accumulator_rejects_changed_variables_dimensions_and_shape() -> None:
    base = _OutputAccumulator.zeros_from_frame(_frame(np.asarray([1.0, 2.0])))
    with pytest.raises((KeyError, ValueError), match="variables changed|unknown"):
        base.add_frame(
            OutputFrame({"salt": OutputVariable(("x",), np.asarray([1.0, 2.0]))})
        )
    with pytest.raises(ValueError, match="dimensions changed"):
        base.add_frame(_frame(np.asarray([1.0, 2.0]), dims=("y",)))
    with pytest.raises(ValueError, match="shape changed"):
        base.add_frame(_frame(np.asarray([1.0, 2.0, 3.0])))


def test_period_mean_sample_to_output_variable_orders_explicit_dimensions() -> None:
    values = np.arange(2 * 3 * 4, dtype=float).reshape((2, 3, 4))
    variable = period_mean_sample_to_output_variable(
        OutputVariable(("lat", "lon", "level"), values, {"units": "K"}),
        time_dim="time",
        dimension_order=("time", "level", "lat", "lon"),
    )

    assert variable.dims == ("time", "level", "lat", "lon")
    assert variable.attrs == {"units": "K"}
    assert_allclose_compact(variable.values[0], np.transpose(values, axes=(2, 0, 1)))


def test_period_mean_sample_to_output_variable_accepts_output_value_dims() -> None:
    values = np.arange(2 * 3 * 4, dtype=float).reshape((2, 3, 4))
    variable = period_mean_sample_to_output_variable(
        OutputVariable(("xt", "yt", "zt"), values),
        time_dim="time",
        value_dims=("zt", "yt", "xt"),
    )

    assert variable.dims == ("time", "zt", "yt", "xt")
    assert_allclose_compact(variable.values[0], np.transpose(values, axes=(2, 1, 0)))
