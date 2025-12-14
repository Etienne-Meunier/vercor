from typing import TYPE_CHECKING, Union


if TYPE_CHECKING:
    from vercor.components import (
        Atmosphere,
        Land,
        Ocean,
        SeaIce,
        ERA5Atmosphere,
        ERA5Land,
        ERA5Ocean,
        ERAInterimOcean,
        JCM,
    )


type OceanType = Union[Ocean, ERA5Ocean, ERAInterimOcean]
type LandType = Union[Land, ERA5Land]
type AtmosphereType = Union[Atmosphere, ERA5Atmosphere, JCM]
type AllComponentsType = Union[OceanType, LandType, AtmosphereType, SeaIce]
