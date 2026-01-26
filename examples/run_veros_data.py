from datetime import datetime

from vercor import Clock, Coupler, Exchange
from vercor.components import ERA5Atmosphere, ERA5Land, VerosGCM
from vercor.coupler import RunSequence
from vercor.regridders import bilinear


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
