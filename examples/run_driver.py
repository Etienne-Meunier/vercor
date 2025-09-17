from datetime import datetime

from verec import Coupler, Clock, Exchange
from verec.coupler import RunSequence
from verec.components import Atmosphere, Ocean, SeaIce, Land
from verec.regridders import XESMFBilinearRectilinear, make_rectilinear_grid

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
runseq = RunSequence(order=["ATM", "OCN", "ICE", "LND"])
# Choose models/components for concurrent execution when MPI is ON

# Coupler
cpl = Coupler(clock=clock, runseq=runseq)
for comp in [ATM, OCN, ICE, LND]:
    cpl.register(comp)

# Exchanges
cpl.add_exchange(Exchange(
    name="ATM_to_OCN",
    source="ATM",
    destination="OCN",
    field_names=["SHF", "LHF"],
    regridder_factory=lambda sg, sm, dg, dm: XESMFBilinearRectilinear(sg, sm, dg, dm),
    when="pre",
))

cpl.add_exchange(Exchange(
    name="OCN_to_ATM",
    source="OCN",
    destination="ATM",
    field_names=["SST"],
    regridder_factory=lambda sg, sm, dg, dm: XESMFBilinearRectilinear(sg, sm, dg, dm),
    when="pre",
))

cpl.add_exchange(Exchange(
    name="OCN_to_ICE",
    source="OCN",
    destination="ICE",
    field_names=["SST"],
    regridder_factory=lambda sg, sm, dg, dm: XESMFBilinearRectilinear(sg, sm, dg, dm),
    when="pre",
))

cpl.add_exchange(Exchange(
    name="ATM_to_LND",
    source="ATM",
    destination="LND",
    field_names=["LHF"],
    regridder_factory=lambda sg, sm, dg, dm: XESMFBilinearRectilinear(sg, sm, dg, dm),
    when="post",
))

# Run
cpl.run()

# Inspect a few fields
print("SST mean:", OCN.state["SST"].data.mean())
print("TA2M mean:", ATM.state["TA2M"].data.mean())
print("ICEFRAC mean:", ICE.state["ICEFRAC"].data.mean())
print("SOILM mean:", LND.state["SOILM"].data.mean())
