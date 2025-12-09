from typing import TYPE_CHECKING, Union


if TYPE_CHECKING:
    from vercor.components import Atmosphere, Land, Ocean, SeaIce
    from vercor.components.data.era5_atmosphere import ERA5Atmosphere
    from vercor.components.data.era5_land import ERA5Land
    from vercor.components.data.era5_ocean import ERA5Ocean
    from vercor.components.data.erainterim_ocean import ERAInterimOcean


type OceanType = Union[Ocean, ERA5Ocean, ERAInterimOcean]
type LandType = Union[Land, ERA5Land]
type AtmosphereType = Union[Atmosphere, ERA5Atmosphere]
type AllComponentsType = Union[OceanType, LandType, AtmosphereType, SeaIce]
