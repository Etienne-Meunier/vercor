"""Shared NetCDF writer boundary tests used by output coordination."""

from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path

import h5netcdf
import numpy as np
import pytest

from tests._coverage_support import capture_logger_output
from tests.assertions import assert_allclose_compact
from vercor.output import OutputVariable
from vercor.output._dataset import time_coordinate_variable
from vercor.output._netcdf import write_netcdf_dataset


def _mean_variables() -> dict[str, OutputVariable]:
    return {
        "temperature": OutputVariable(
            ("time", "x"),
            np.asarray([[1.0, 3.0]]),
            {"units": "K"},
        )
    }


def _coordinate_variables() -> dict[str, OutputVariable]:
    return {
        "time": time_coordinate_variable(datetime(2000, 1, 2)),
        "x": OutputVariable(("x",), np.asarray([10.0, 20.0])),
    }


def test_write_netcdf_dataset_logs_one_coordinator_write(tmp_path: Path) -> None:
    output = tmp_path / "period-average.nc"
    logger_name = "VerCOR.test.period-files"
    logger = logging.getLogger(logger_name)

    with capture_logger_output(logger_name) as stream:
        write_netcdf_dataset(
            output=str(output),
            coordinate_variables=_coordinate_variables(),
            data_variables=_mean_variables(),
            logger=logger,
        )

    with h5netcdf.File(output, "r") as actual:
        temperature = actual.variables["temperature"]
        assert temperature.dimensions == ("time", "x")
        assert temperature.attrs["units"] == "K"
        assert_allclose_compact(np.asarray(temperature), np.asarray([[1.0, 3.0]]))
        assert_allclose_compact(np.asarray(actual.variables["x"]), [10.0, 20.0])
    assert stream.getvalue().count(f"Writing output file:  {output}") == 1


def test_write_netcdf_dataset_persists_coordinator_metadata(tmp_path: Path) -> None:
    output = tmp_path / "decorated.nc"
    variables = {
        name: OutputVariable(
            variable.dims,
            variable.values,
            {**dict(variable.attrs), "long_name": "Air temperature"},
        )
        for name, variable in _mean_variables().items()
    }

    write_netcdf_dataset(
        output=str(output),
        coordinate_variables=_coordinate_variables(),
        data_variables=variables,
    )

    with h5netcdf.File(output, "r") as actual:
        assert actual.variables["temperature"].attrs["long_name"] == "Air temperature"


def test_write_netcdf_dataset_surfaces_dimension_conflicts(tmp_path: Path) -> None:
    conflicting = {
        "temperature": OutputVariable(
            ("time", "x"),
            np.asarray([[1.0, 2.0, 3.0]]),
        )
    }

    with pytest.raises(ValueError, match="dimension 'x'.*existing size 2.*new size 3"):
        write_netcdf_dataset(
            output=str(tmp_path / "conflict.nc"),
            coordinate_variables=_coordinate_variables(),
            data_variables=conflicting,
        )
