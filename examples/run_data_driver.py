from datetime import datetime
from typing import List
import numpy as np

from vercor import Clock, Coupler, Exchange
from vercor.components import ERA5Atmosphere, Land, Ocean, SeaIce
from vercor.coupler import RunSequence
from vercor.regridders import (BilinearRectilinearRegridder,
                               make_rectilinear_grid)

if __name__ == "__main__":
    # Build grids
    ocn_grid = make_rectilinear_grid("ocn-grid", 64, 32, 0.0, 360.0, -80.0, 80.0)
    ice_grid = make_rectilinear_grid("ice-grid", 64, 32, 0.0, 360.0, -80.0, 80.0)
    lnd_grid = make_rectilinear_grid("lnd-grid", 96, 48, 0.0, 360.0, -60.0, 60.0)

    # Build components
    atm = ERA5Atmosphere("ERA5")
    ocn = Ocean("OCN", ocn_grid)
    ice = SeaIce("ICE", ice_grid)
    lnd = Land("LND", lnd_grid)

    # Clock and sequence
    clock = Clock(start=datetime(2025, 1, 1, 0, 0, 0), dt_seconds=3600, steps=24)
    run_sequence = RunSequence(order=["OCN", "ERA5", "ICE", "LND"])

    # Coupler
    cpl = Coupler(clock=clock)
    components: List[ERA5Atmosphere | Ocean | SeaIce | Land] = [atm, ocn, ice, lnd]
    for component in components:
        cpl.register(component)

    cpl.set_components_run_sequence(run_sequence)

    # Bilinear interpolation
    # Having interpolator factory function allows easy access
    # to different interpolators' args & kwargs
    bilinear = lambda source_grid, destination_grid:\
        BilinearRectilinearRegridder(source_grid, destination_grid)

    # Exchanges
    # scalar fields (vector field))
    # ["qbot", "zbot", ("ubot", "vbot")]
    cpl.add_exchange(Exchange(
        source="ERA5",
        destination="OCN",
        field_names=[("ubot", "vbot"), "qbot", "zbot", "rbot", "thbot", "tbot"],
        regridder_factory=bilinear,
        when="pre",
    ))

    cpl.add_exchange(Exchange(
        source="OCN",
        destination="ERA5",
        field_names=["SST",],
        regridder_factory=bilinear,
        when="pre",
    ))

    cpl.initialize()
    cpl.run()

    # Inspect a few fields
    print("SST(OCN) mean:", ocn.get("SST").mean())
    print("SST(ERA5) mean:", atm.get("SST").mean())
    print("qbot(ERA5) mean:", atm.get("qbot").mean())
    print("qbot(OCN) mean:", ocn.get("qbot").mean())
    print("tbot(ERA5) mean:", atm.get("tbot").mean())
    print("tbot(OCN) mean:", ocn.get("tbot").mean())
    print("zbot(ERA5) mean:", atm.get("zbot").mean())
    print("zbot(OCN) mean:", ocn.get("zbot").mean())
    print("speed(ERA5) mean:", np.sqrt(atm.get("ubot")**2 + atm.get("vbot")**2).mean())
    print("speed(OCN) mean:", np.sqrt(ocn.get("ubot")**2 + ocn.get("vbot")**2).mean())
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(15, 10), layout="constrained")

    lon_atm = np.array(atm.grid.longitude)
    lat_atm = np.array(atm.grid.latitude)
    longitude_source_2d, latitude_source_2d = np.meshgrid(lon_atm, lat_atm, indexing="ij")
    scalar_source = atm.get("zbot").T
    u_source = atm.get("ubot").T
    v_source = atm.get("vbot").T

    lon_ocn = np.array(ocn.grid.longitude)
    lat_ocn = np.array(ocn.grid.latitude)
    longitude_target_2d, latitude_target_2d = np.meshgrid(lon_ocn, lat_ocn, indexing="ij")
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
