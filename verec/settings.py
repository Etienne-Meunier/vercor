from collections import namedtuple

"""
Constants and settings for verec.
Adapted from Veros: https://github.com/team-ocean/veros/blob/main/veros/settings.py
"""
#
Setting = namedtuple("setting", ("default", "type", "description"))

SETTINGS = {
    "identifier": Setting("UNNAMED", str, "Identifier of the current simulation"),
    "output_frequency": Setting(1, int, "Frequency of output in timesteps"),
    "max_steps": Setting(1000, int, "Maximum number of timesteps"),
    "dt": Setting(60.0, float, "Timestep size in seconds"),
    # ------------------------- Physical constants ----------------------------------
    # ------------------------- Bulk formula constants ------------------------------
    "gravity": Setting(9.81, float, "Acceleration due to gravity [m/s^2]"),
    "rhoAir": Setting(1.3, float, "Density of air [kg/m^3]"),
    "cpdair": Setting(1.00464e3, float, "Specific heat capacity of dry air [J/(kg*K)]"),
    "zvir": Setting(
        0.608,
        float,
        "(RWV / RDAIR) - 1.0 - Dry-air water-vapor molecular mass ratio [-]",
    ),
    "karman": Setting(0.4, float, "von Karman constant"),
    "stefBoltz": Setting(5.67e-8, float, "Stefan-Boltzmann constant [W/m^2/K^4]"),
    "ocean_emissivity": Setting(0.97, float, "Long-wave emissivity of ocean surface"),
    "ice_emissivity": Setting(0.97, float, "Long-wave emissivity of sea ice"),
    "snow_emissivity": Setting(0.99, float, "Long-wave emissivity of snow"),
    "latvap": Setting(2.501e6, float, "Latent heat of vaporization [J/kg]"),
    "latfresh": Setting(3.34e5, float, "Latent heat of fusion [J/kg]"),
    "gamma_blk": Setting(0.1, float, "Bulk aerodynamic resistance"),
    # --------------------------------------------------------------------------------
    "rgas": Setting(8314.47, float, "avogad * bolzc - Ideal gas constant [J/K/kmole]"),
    "cpdair": Setting(1.00464e3, float, "specific heat of dry air [J/K/kg]"),
    "umin_ocean": Setting(0.5, float, "minimum atm. wind speed over ocean surface [m/s]"),
    "umin_ice": Setting(1., float, "minimum atm. wind speed over ice surface [m/s]"),
    "earth_radius": Setting(6.371e6, float, "Earth radius [m]"),
}
