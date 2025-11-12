from datetime import datetime

from vercor import Clock, Coupler, Exchange
from vercor.components import Atmosphere, Land, Ocean, SeaIce
from vercor.coupler import RunSequence
from vercor.regridders import (BilinearRectilinearRegridder,
                               make_rectilinear_grid)


# Build grids
atm_grid = make_rectilinear_grid("atm-grid", 128, 64, 0.0, 360.0, -90.0, 90.0)
ocn_grid = make_rectilinear_grid("ocn-grid", 64, 32, 0.0, 360.0, -80.0, 80.0)
ice_grid = make_rectilinear_grid("ice-grid", 64, 32, 0.0, 360.0, -80.0, 80.0)
lnd_grid = make_rectilinear_grid("lnd-grid", 96, 48, 0.0, 360.0, -60.0, 60.0)

# Build components
atm = Atmosphere("ATM", atm_grid)
ocn = Ocean("OCN", ocn_grid)
ice = SeaIce("ICE", ice_grid)
lnd = Land("LND", lnd_grid)

# Clock and sequence
clock = Clock(start=datetime(2025, 1, 1, 0, 0, 0), dt_seconds=3600, steps=24)
run_sequence = RunSequence(order=["ATM", "OCN", "ICE", "LND"])

# Coupler
cpl = Coupler(clock=clock)
for component in [atm, ocn, ice, lnd]:
    cpl.register(component)

cpl.set_components_run_sequence(run_sequence)

# Bilinear interpolation
# Having interpolator factory function allows easy access
# to different interpolators' args & kwargs
bilinear = lambda source_grid, destination_grid:\
    BilinearRectilinearRegridder(source_grid, destination_grid)

# Exchanges
# scalar fields (vector field))
# ["SHF", "LHF", ("u10m", "v10m")]
cpl.add_exchange(Exchange(
    source="ATM",
    destination="OCN",
    field_names=[("u10m", "v10m"), "SHF", "LHF"],
    regridder_factory=bilinear,
    when="pre",
))

cpl.add_exchange(Exchange(
    source="OCN",
    destination="ATM",
    field_names=["SST"],
    regridder_factory=bilinear,
    when="pre",
))

cpl.add_exchange(Exchange(
    source="OCN",
    destination="ICE",
    field_names=["SST"],
    regridder_factory=bilinear,
    when="pre",
))

cpl.add_exchange(Exchange(
    source="LND",
    destination="ATM",
    field_names=["SOILM"],
    regridder_factory=bilinear,
    when="pre",
))

cpl.add_exchange(Exchange(
    source="ATM",
    destination="LND",
    field_names=["LHF"],
    regridder_factory=bilinear,
    when="post",
))

cpl.initialize()
cpl.run()

# Inspect a few fields
print("SST mean:", ocn.state["SST"].mean())
print("TA2M mean:", atm.state["TA2M"].mean())
print("u10m mean:", atm.state["u10m"].mean())
print("v10m mean:", atm.state["v10m"].mean())
print("SOILM(LND) mean:", lnd.state["SOILM"].mean())
print("SOILM(ATM) mean:", atm.state["SOILM"].mean())
print("ICEFRAC mean:", ice.state["ICEFRAC"].mean())
