import pytest
from verec.regridders import make_rectilinear_grid


@pytest.fixture
def atm_grid():
    return make_rectilinear_grid("atm-grid", 128, 64, 0.0, 360.0, -90.0, 90.0)


@pytest.fixture
def ocn_grid():
    return make_rectilinear_grid("ocn-grid", 64, 32, 0.0, 360.0, -80.0, 80.0)


@pytest.fixture
def seaice_grid():
    return make_rectilinear_grid("seaice-grid", 64, 32, 0.0, 360.0, -80.0, 80.0)


@pytest.fixture
def lnd_grid():
    return make_rectilinear_grid("lnd-grid", 96, 48, 0.0, 360.0, -60.0, 60.0)
