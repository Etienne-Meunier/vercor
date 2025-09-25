from dataclasses import dataclass


@dataclass
class VercorSettings:
    identifier: str = "UNNAMED"  # Identifier of the current simulation
    output_frequency: int = 1  # Frequency of output in timesteps
    max_steps: int = 1000  # Maximum number of timesteps
    dt: float = 60.0  # Timestep size in seconds
    # ------------------------- Physical constants ----------------------------------
    earth_radius: float = 6.371e6  # Earth radius [m]
    # ------------------------- Bulk formula constants ------------------------------
    gravity: float = 9.81  # Acceleration due to gravity [m/s^2]
    rhoAir: float = 1.3  # Density of air [kg/m^3]
    cpdair: float = 1.00464e3  # Specific heat capacity of dry air [J/(kg*K)]
    zvir: float = (
        0.608  # (rwv / rdair) - 1.0 - Dry-air water-vapor molecular mass ratio [-]
    )
    cpwv: float = 1.810e3  # specific heat of water vapor [J/kg/K]
    cpvir: float = 0.802  # cpwv/cpdair - 1.0 specific heat of vaporization [-]
    latice: float = 3.337e5  # latent heat of fusion  [J/kg]
    rgas: float = 8314.47  # avogad * bolzc - Ideal gas constant [J/K/kmole]
    umin_ocean: float = 0.5  # minimum atm. wind speed over ocean surface [m/s]
    umin_ice: float = 1.0  # minimum atm. wind speed over ice surface [m/s]
    karman: float = 0.4  # von Karman constant
    stefBoltz: float = 5.67e-8  # Stefan-Boltzmann constant [W/m^2/K^4]
    ocean_emissivity: float = 0.97  # Long-wave emissivity of ocean surface [-]
    ice_emissivity: float = 0.97  # Long-wave emissivity of sea ice [-]
    snow_emissivity: float = 0.99  # Long-wave emissivity of snow [-]
    latvap: float = 2.501e6  # Latent heat of vaporization [J/kg]
    latfresh: float = 3.34e5  # Latent heat of fusion [J/kg]
    gamma_blk: float = 0.1  # Bulk aerodynamic resistance [-]   
    # --------------------------------------------------------------------------------


