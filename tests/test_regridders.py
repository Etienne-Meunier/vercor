from vercor.regridders.bilinear import BilinearRectilinearRegridder


def test_bilinear_rectilinear(atm_grid, ocn_grid):
    # Bilinear regridder
    regridder = BilinearRectilinearRegridder(atm_grid, ocn_grid)
    assert regridder.source_grid == atm_grid
    assert regridder.destination_grid == ocn_grid
    assert regridder.interpolator is not None
