from datetime import datetime
from typing import List
import numpy as np

from vercor import Clock, Coupler, Exchange
from vercor.components import ERA5Atmosphere, ERA5Ocean
from vercor.components.data.erainterim_ocean import ERAInterimOcean
from vercor.coupler import RunSequence
from vercor.regridders import BilinearRectilinearRegridder

if __name__ == "__main__":
    # Build components
    atm = ERA5Atmosphere("ATM")
    ocn = ERAInterimOcean("OCN")

    # Clock and sequence
    clock = Clock(start=datetime(2025, 1, 1, 0, 0, 0), dt_seconds=3600, steps=2)
    run_sequence = RunSequence(order=["OCN", "ATM"])

    # Coupler
    cpl = Coupler(clock=clock)
    components: List[ERA5Atmosphere | ERAInterimOcean] = [atm, ocn]
    for component in components:
        cpl.register(component)

    cpl.set_components_run_sequence(run_sequence)

    # Bilinear interpolation
    # Having interpolator factory function allows easy access
    # to different interpolators' args & kwargs
    bilinear = lambda source_grid, destination_grid: BilinearRectilinearRegridder(
        source_grid, destination_grid
    )

    # Exchanges
    # scalar fields (vector field))
    # ["qbot", "zbot", ("ubot", "vbot")]
    cpl.add_exchange(
        Exchange(
            source="ATM",
            destination="OCN",
            field_names=[("ubot", "vbot"), "qbot", "zbot", "rbot", "thbot", "tbot"],
            regridder_factory=bilinear,
            when="pre",
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
            when="pre",
        )
    )

    cpl.initialize()
    cpl.run()

    # Inspect a few fields
    print("SST(OCN) mean:", np.nanmin(ocn.get("sst")))
    print("SST(ERA) mean:", np.nanmin(atm.get("sst")))
    print("qbot(ERA) mean:", np.nanmin(atm.get("qbot")))
    print("qbot(OCN) mean:", np.nanmin(ocn.get("qbot")))
    print("tbot(ERA) mean:", np.nanmin(atm.get("tbot")))
    print("tbot(OCN) mean:", np.nanmin(ocn.get("tbot")))
    print("zbot(ERA) mean:", np.nanmin(atm.get("zbot")))
    print("zbot(OCN) mean:", np.nanmin(ocn.get("zbot")))
    print(
        "speed(ERA) mean:",
        np.nanmean(np.sqrt(atm.get("ubot") ** 2 + atm.get("vbot") ** 2)),
    )
    print(
        "speed(OCN) mean:",
        np.nanmean(np.sqrt(ocn.get("ubot") ** 2 + ocn.get("vbot") ** 2)),
    )
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(15, 10), layout="constrained")

    lon_atm = np.array(atm.grid.longitude)
    lat_atm = np.array(atm.grid.latitude)
    longitude_source_2d, latitude_source_2d = np.meshgrid(
        lon_atm, lat_atm, indexing="ij"
    )
    scalar_source = atm.get("zbot").T
    u_source = atm.get("ubot").T
    v_source = atm.get("vbot").T

    lon_ocn = np.array(ocn.grid.longitude)
    lat_ocn = np.array(ocn.grid.latitude)
    longitude_target_2d, latitude_target_2d = np.meshgrid(
        lon_ocn, lat_ocn, indexing="ij"
    )
    scalar_target = ocn.get("zbot").T
    u_target = ocn.get("ubot").T
    v_target = ocn.get("vbot").T

    im = axs[0, 0].pcolormesh(
        longitude_source_2d,
        latitude_source_2d,
        scalar_source,
        shading="auto",
        cmap="coolwarm",
    )
    axs[0, 0].set_title("Initial Scalar Field")
    axs[0, 0].set_xlabel("Longitude")
    axs[0, 0].set_ylabel("Latitude")

    axs[0, 1].quiver(
        longitude_source_2d,
        latitude_source_2d,
        u_source,
        v_source,
        scale=150,
    )
    axs[0, 1].set_title("Initial Vector Field")
    axs[0, 1].set_xlabel("Longitude")
    axs[0, 1].set_ylabel("Latitude")

    axs[1, 0].pcolormesh(
        longitude_target_2d,
        latitude_target_2d,
        scalar_target,
        shading="auto",
        cmap="coolwarm",
    )
    axs[1, 0].set_title("Interpolated Scalar Field")
    axs[1, 0].set_xlabel("Longitude")
    axs[1, 0].set_ylabel("Latitude")

    axs[1, 1].quiver(
        longitude_target_2d,
        latitude_target_2d,
        u_target,
        v_target,
        scale=150,
    )
    axs[1, 1].set_title("Interpolated Vector Field")
    axs[1, 1].set_xlabel("Longitude")
    axs[1, 1].set_ylabel("Latitude")

    fig.colorbar(im, ax=axs, shrink=0.6)

    plt.show()
