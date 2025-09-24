from datetime import datetime

from vercor import Coupler, Clock, Exchange
from vercor.coupler import RunSequence
from vercor.components import Atmosphere, Ocean, SeaIce, Land
from vercor.regridders import BilinearRectilinear, make_rectilinear_grid

# Build grids
atm_grid = make_rectilinear_grid("atm-grid", 128, 64, 0.0, 360.0, -90.0, 90.0)
ocn_grid = make_rectilinear_grid("ocn-grid", 64, 32, 0.0, 360.0, -80.0, 80.0)
ice_grid = make_rectilinear_grid("ice-grid", 64, 32, 0.0, 360.0, -80.0, 80.0)
lnd_grid = make_rectilinear_grid("lnd-grid", 96, 48, 0.0, 360.0, -60.0, 60.0)

# Build components
ATM = Atmosphere("ATM", atm_grid)
OCN = Ocean("OCN", ocn_grid)
ICE = SeaIce("ICE", ice_grid)
LND = Land("LND", lnd_grid)

# Clock and sequence
clock = Clock(start=datetime(2025, 1, 1, 0, 0, 0), dt_seconds=3600, steps=48)
run_sequence = RunSequence(order=["ATM", "OCN", "ICE", "LND"])

# Choose models/components for concurrent execution when MPI is ON

# Coupler
cpl = Coupler(clock=clock, runseq=run_sequence)
for comp in [ATM, OCN, ICE, LND]:
    cpl.register(comp)

# Bilinear interpolation
bilinear = lambda sg, sm, dg, dm: BilinearRectilinear(sg, sm, dg, dm)

# Exchanges
# scalar fields (vector field)) 
#["SHF", "LHF", ("u10m", "v10m")]
cpl.add_exchange(Exchange(
    name="ATM2OCN",
    source="ATM",
    destination="OCN",
    field_names=[("u10m", "v10m"), "SHF", "LHF"],
    regridder_factory=bilinear,
    when="pre",
))

cpl.add_exchange(Exchange(
    name="OCN2ATM",
    source="OCN",
    destination="ATM",
    field_names=["SST"],
    regridder_factory=bilinear,
    when="pre",
))

cpl.add_exchange(Exchange(
    name="OCN2ICE",
    source="OCN",
    destination="ICE",
    field_names=["SST"],
    regridder_factory=bilinear,
    when="pre",
))

cpl.add_exchange(Exchange(
    name="LND2ATM",
    source="LND",
    destination="ATM",
    field_names=["SOILM"],
    regridder_factory=bilinear,
    when="pre",
))

cpl.add_exchange(Exchange(
    name="ATM2LND",
    source="ATM",
    destination="LND",
    field_names=["LHF"],
    regridder_factory=bilinear,
    when="post",
))

# Run
cpl.run()

# Inspect a few fields
print("SST mean:", OCN.state["SST"].data.mean())
print("u10m mean:", ATM.state["u10m"].data.mean())
print("v10m mean:", ATM.state["v10m"].data.mean())
print("TA2M mean:", ATM.state["TA2M"].data.mean())
print("ICEFRAC mean:", ICE.state["ICEFRAC"].data.mean())
print("SOILM mean:", LND.state["SOILM"].data.mean())
print("SOILM(LND) mean:", ATM.state["SOILM"].data.mean())
