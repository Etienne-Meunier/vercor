from typing import TYPE_CHECKING, Any, TypeAlias, Union

import jax
from numpy.typing import NDArray

if TYPE_CHECKING:
    from vercor.components import (
        Atmosphere,
        Land,
        Ocean,
        SeaIce,
        ERA5Atmosphere,
        ERA5Land,
        JCMLand,
        ERA5Ocean,
        ERAInterimOcean,
        JAXGCM,
        VerosGCM,
        CAMulatorGCM,
        CAMulatorLand,
    )

OceanType: TypeAlias = Union["Ocean", "ERA5Ocean", "ERAInterimOcean", "VerosGCM"]
LandType: TypeAlias = Union["Land", "ERA5Land", "JCMLand", "CAMulatorLand"]
AtmosphereType: TypeAlias = Union[
    "Atmosphere", "ERA5Atmosphere", "JAXGCM", "CAMulatorGCM"
]
AllComponentsType: TypeAlias = Union[OceanType, LandType, AtmosphereType, "SeaIce"]
RuntimeArray: TypeAlias = NDArray[Any] | jax.Array
