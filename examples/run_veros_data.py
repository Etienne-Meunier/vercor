from datetime import datetime

from vercor import Clock, Coupler, Exchange
from vercor.components import ERA5Atmosphere, ERA5Land, VerosGCM
from vercor.coupler import RunSequence
from vercor.grid import RectilinearGrid
from vercor.regridders import BilinearRectilinearRegridder
from vercor.regridders.conservative import ConservativeRectilinearRegridder


if __name__ == "__main__":
    atm = ERA5Atmosphere()
    ocn = VerosGCM()
    lnd = ERA5Land()

    # Clock and sequence
    clock = Clock(start=datetime(2025, 1, 1, 0, 0, 0), dt_seconds=86400.0, steps=360)
    run_sequence = RunSequence(order=["OCN", "LND", "ATM"])

    # Coupler
    cpl = Coupler(clock=clock)
    components = [ocn, lnd, atm]
    for component in components:
        cpl.register(component)  # type: ignore

    cpl.set_components_run_sequence(run_sequence)

    # Bilinear interpolation
    # Having interpolator factory function allows easy access
    # to different interpolators' args & kwargs
    def bilinear(
        source_grid: RectilinearGrid, destination_grid: RectilinearGrid
    ) -> BilinearRectilinearRegridder:
        return BilinearRectilinearRegridder(source_grid, destination_grid)

    def conservative(
        source_grid: RectilinearGrid, destination_grid: RectilinearGrid
    ) -> ConservativeRectilinearRegridder:
        return ConservativeRectilinearRegridder(source_grid, destination_grid)

    # Exchanges
    cpl.add_exchange(
        Exchange(
            source="ATM",
            destination="OCN",
            field_names=[
                ("ubot", "vbot"),
                "qbot",
                "zbot",
                "rbot",
                "thbot",
                "tbot",
            ],
            regridder_factory=bilinear,
        )
    )

    cpl.add_exchange(
        Exchange(
            source="ATM",
            destination="OCN",
            field_names=[
                "swr_net",
                "lwr_dw",
            ],
            regridder_factory=bilinear,
        )
    )

    cpl.add_exchange(
        Exchange(
            source="OCN",
            destination="ATM",
            field_names=[
                "sst",
            ],
            regridder_factory=bilinear,
        )
    )

    cpl.add_exchange(
        Exchange(
            source="ATM",
            destination="LND",
            field_names=[
                "tbot",
                "qbot",
            ],
            regridder_factory=bilinear,
        )
    )

    cpl.add_exchange(
        Exchange(
            source="LND",
            destination="ATM",
            field_names=[
                "skt",
            ],
            regridder_factory=bilinear,
        )
    )

    cpl.initialize()
    cpl.run()
    cpl.finalize()
