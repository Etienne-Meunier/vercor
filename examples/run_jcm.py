from datetime import datetime
from typing import List

import numpy as np

from vercor import Clock, Coupler, Exchange
from vercor.components import Land, Ocean, SeaIce, JCM
from vercor.coupler import RunSequence
from vercor.grid import RectilinearGrid
from vercor.regridders import (
    BilinearRectilinearRegridder,
    ConservativeRectilinearRegridder,
    make_rectilinear_grid,
)

import jcm

if __name__ == "__main__":
    
    land_fraction_threshold = 0.9
    
    # Build grids
    ocn_grid = make_rectilinear_grid("ocn-grid", 64, 32, 0.0, 360.0, -90.0, 90.0)
    ice_grid = make_rectilinear_grid("ice-grid", 64, 32, 0.0, 360.0, -90.0, 90.0)

    # Build components
    atm = JCM("ATM", jcm.model.Model(), jitted=True)
    
    hgrid = atm.model.coords.horizontal
    lnd_grid = RectilinearGrid(
        name="LND",
        longitude=np.array(hgrid.longitudes) * 180.0 / np.pi,
        latitude=np.array(hgrid.latitudes) * 180.0 / np.pi,
        binary_mask=np.where(atm.model.geometry.fmask > land_fraction_threshold, 0.0, 1.0).transpose(),  # 0 = land, 1 = ocean
    )

    ocn_grid = RectilinearGrid(
        name="OCN",
        longitude=np.array(hgrid.longitudes) * 180.0 / np.pi,
        latitude=np.array(hgrid.latitudes) * 180.0 / np.pi,
        binary_mask=np.where(atm.model.geometry.fmask <= land_fraction_threshold, 1.0, 0.0).transpose(),  # 0 = land, 1 = ocean
    )


    ocn = Ocean("OCN", ocn_grid)
    ice = SeaIce("ICE", ice_grid)
    lnd = Land("LND", lnd_grid)

    # Clock and sequence
    clock = Clock(start=datetime(2025, 1, 1, 0, 0, 0), dt_seconds=86400.0, steps=2)
    run_sequence = RunSequence(order=["OCN", "ATM", "ICE", "LND"])

    # Coupler
    cpl = Coupler(clock=clock)
    components = [atm, ocn, ice, lnd]
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
    # scalar fields (vector field))
    # ["SHF", "LHF", ("u10m", "v10m")]
    cpl.add_exchange(
        Exchange(
            source="ATM",
            destination="OCN",
            field_names=[("u10m", "v10m"), "SHF", "LHF"],
            regridder_factory=bilinear,
        )
    )

    cpl.add_exchange(
        Exchange(
            source="OCN",
            destination="ATM",
            field_names=["sst"],
            regridder_factory=bilinear,
        )
    )

    cpl.add_exchange(
        Exchange(
            source="OCN",
            destination="ICE",
            field_names=["sst"],
            regridder_factory=bilinear,
        )
    )

    cpl.add_exchange(
        Exchange(
            source="LND",
            destination="ATM",
            field_names=["SOILM"],
            regridder_factory=bilinear,
        )
    )

    cpl.add_exchange(
        Exchange(
            source="ATM",
            destination="LND",
            field_names=["LHF", "SHF"],
            regridder_factory=conservative,
        )
    )

    cpl.initialize()
    cpl.run()
    cpl.finalize()

    atm._finalize("JCM-output.nc")

    # Inspect a few fields
    print("sst(OCN) mean:", np.nanmean(ocn.get("sst")))
    print("sst(ATM) mean:", np.nanmean(atm.get("sst")))
    print("TA2M mean:", np.nanmean(atm.get("TA2M")))
    print("u10m mean:", np.nanmean(atm.get("u10m")))
    print("v10m mean:", np.nanmean(atm.get("v10m")))
    print("SOILM(LND) mean:", np.nanmean(lnd.get("SOILM")))
    print("SOILM(ATM) mean:", np.nanmean(atm.get("SOILM")))
    print("ICEFRAC mean:", np.nanmean(ice.get("ICEFRAC")))
    print("SHF(ATM) mean:", np.nanmean(atm.get("SHF")))
    print("SHF(LND) mean:", np.nanmean(lnd.get("SHF")))

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(15, 10), layout="constrained")

    lon_atm = np.array(atm.grid.longitude)
    lat_atm = np.array(atm.grid.latitude)
    longitude_source_2d, latitude_source_2d = np.meshgrid(
        lon_atm, lat_atm, indexing="ij"
    )
    scalar_source = atm.get("sst").T
    u_source = atm.get("u10m").T
    v_source = atm.get("v10m").T

    lon_ocn = np.array(ocn.grid.longitude)
    lat_ocn = np.array(ocn.grid.latitude)
    longitude_target_2d, latitude_target_2d = np.meshgrid(
        lon_ocn, lat_ocn, indexing="ij"
    )
    scalar_target = ocn.get("sst").T
    u_target = ocn.get("u10m").T
    v_target = ocn.get("v10m").T

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
        scale=100,
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
        scale=100,
    )
    axs[1, 1].set_title("Interpolated Vector Field")
    axs[1, 1].set_xlabel("Longitude")
    axs[1, 1].set_ylabel("Latitude")

    fig.colorbar(im, ax=axs, shrink=0.6)

    plt.show()
