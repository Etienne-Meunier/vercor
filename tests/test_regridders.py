import numpy as np
from verec.regridders import XESMFBilinear


def test_xESMF_regridders(atm_grid, ocn_grid):
    # Bilinear regridder
    regridder = XESMFBilinear(atm_grid, None, ocn_grid, None)
    assert regridder.src_grid == atm_grid
    assert regridder.dst_grid == ocn_grid
    assert not hasattr(regridder, 'regridder')
    regridder.prepare()
    assert regridder.regridder is not None
    atmospheric_t2m = 273.15 + 15.0 * np.ones((regridder.regridder.shape_in))
    oceanic_t2m = regridder.regridder(atmospheric_t2m)
    assert oceanic_t2m.shape == regridder.regridder.shape_out

    # Reuse weights
    # regridder = XESMFBilinear(lnd_grid, None, atm_grid, None)
    # regridder.prepare(reuse_weights=True)
    # assert regridder.regridder is not None

    # Conservative regridder
    # regridder = XESMFConservative_normed(ocn_grid, None, seaice_grid, None)
    # assert regridder.src_grid == ocn_grid
    # assert regridder.dst_grid == seaice_grid
    # assert regridder.regridder is None
    # regridder.prepare()
    # assert regridder.regridder is not None
