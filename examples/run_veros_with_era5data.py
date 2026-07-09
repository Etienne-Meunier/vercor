from datetime import datetime

from vercor import (
    Clock,
    Coupler,
    Exchange,
    OutputConfig,
    PeriodOutput,
    RuntimeOptions,
)
from vercor.setups import make_era5_atmosphere
from vercor.setups import make_era5_land
from vercor.setups import VerosConfig, make_veros_gcm
from vercor.recipes import (
    ATMOSPHERE_TO_LAND_BASIC_FIELDS,
    ATMOSPHERE_TO_VEROS_FORCING_FIELDS,
    LAND_TO_ATMOSPHERE_SURFACE_FIELDS,
    OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
)
from vercor.regridding import bilinear
from vercor.topology import SurfaceMaskPolicy

if __name__ == "__main__":
    atm = make_era5_atmosphere()
    ocn = make_veros_gcm(
        config=VerosConfig(
            restore_to_climatology=True,
            output=OutputConfig(
                period=PeriodOutput(
                    frequency="month",
                    variables=(
                        "temp",
                        "salt",
                        "u",
                        "v",
                        "w",
                        "surface_taux",
                        "surface_tauy",
                        "psi",
                    ),
                ),
            ),
        ),
    )
    lnd = make_era5_land()

    # Clock and sequence
    clock = Clock(
        start=datetime(2000, 1, 1, 0, 0, 0),
        dt_seconds=86400.0,
        steps=365,
        calendar="noleap",
    )
    run_order = ["OCN", "LND", "ATM"]

    components = [ocn, lnd, atm]
    cpl = Coupler(
        clock=clock,
        components=components,
        run_order=run_order,
        runtime=RuntimeOptions(topology=SurfaceMaskPolicy()),
    )

    # Exchanges
    cpl.add_exchanges(
        (
            Exchange(
                source="ATM",
                target="OCN",
                fields=ATMOSPHERE_TO_VEROS_FORCING_FIELDS,
                regrid=bilinear,
            ),
            Exchange(
                source="OCN",
                target="ATM",
                fields=OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
                regrid=bilinear,
            ),
            Exchange(
                source="ATM",
                target="LND",
                fields=ATMOSPHERE_TO_LAND_BASIC_FIELDS,
                regrid=bilinear,
            ),
            Exchange(
                source="LND",
                target="ATM",
                fields=LAND_TO_ATMOSPHERE_SURFACE_FIELDS,
                regrid=bilinear,
            ),
        ),
    )

    cpl.initial_state()
    final_state = cpl.run()
    cpl.write_outputs(final_state)
