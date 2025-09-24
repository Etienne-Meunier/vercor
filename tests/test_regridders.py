import numpy as np
from vercor.regridders.bilinear import BilinearRectilinear


def test_bilinear_rectilinear(atm_grid, ocn_grid):
    # Bilinear regridder
    regridder = BilinearRectilinear(atm_grid, None, ocn_grid, None)
    assert regridder.src_grid == atm_grid
    assert regridder.dst_grid == ocn_grid
    assert getattr(regridder, 'regridder') == None
    regridder.prepare()
    assert regridder.regridder is not None
