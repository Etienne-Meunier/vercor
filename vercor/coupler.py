from dataclasses import dataclass, field
from itertools import chain
from typing import Dict, List, Tuple, Union
import logging

from vercor.components import Atmosphere, Ocean, SeaIce, Land

from vercor.fields import Field
from vercor.clock import Clock
from vercor.exchange import Exchange
from vercor.regridders.base import Regridder
from vercor.run_sequence import RunSequence
from vercor.settings import SETTINGS


logger = logging.getLogger("VerCOR.coupler")
logging.basicConfig(level=logging.INFO)


def _scalar_field_interpolate(
    field_name: str,
    source_fields: Dict[str, Field],
    regridder: Regridder,
) -> Field:
    if not callable(regridder):
        raise TypeError("Regridder must be callable for scalar field interpolation")

    destination_field = regridder(source_fields[field_name])

    return destination_field


def _vector_field_interpolate(
    field_name: Tuple[str, str],
    source_fields: Dict[str, Field],
    regridder: Regridder,
) -> Tuple[Field, Field]:
    if len(field_name) == 2:
        src_field_name, alt_field_name = field_name
    else:
        raise ValueError("Vector field name must be a tuple of two strings")
    if not callable(regridder):
        raise TypeError("Regridder must be callable for vector field interpolation")
    try:
        destination_field_lon, destination_field_lat = regridder(
            source_fields[src_field_name], source_fields[alt_field_name]
        )
    except Exception as e:
        raise TypeError(
            "Regridder for vector fields must accept two arguments and return a tuple of two Fields"
        ) from e
    return (destination_field_lon, destination_field_lat)


@dataclass
class Coupler:
    # Add communicator for MPI
    clock: Clock
    runseq: RunSequence
    components: Dict[str, Union[Atmosphere, Ocean, SeaIce, Land]] = field(
        default_factory=dict
    )
    exchanges: List[Exchange] = field(default_factory=list)
    _regridders: Dict[
        Tuple[str, str],
        Regridder,
    ] = field(default_factory=dict)

    def register(self, component: Union[Atmosphere, Ocean, SeaIce, Land]) -> None:
        if component.name in self.components:
            raise ValueError(f"Component {component.name} already registered")
        self.components[component.name] = component
        logger.info(f"Registered component {component.name}")

    def add_exchange(self, exchange: Exchange) -> None:
        self.exchanges.append(exchange)
        logger.info(
            f"Added exchange {exchange.name}: {exchange.source} -> {exchange.destination} {exchange.field_names} [{exchange.when}]"
        )

    def initialize(self) -> None:
        # Build regridders per (src, dst)
        for exchange in self.exchanges:
            key = (exchange.source, exchange.destination)
            if key not in self._regridders:
                src_grid = self.components[exchange.source].grid
                src_mask = (
                    self.components[exchange.source].grid.mask
                    if self.components[exchange.source].grid.mask
                    else None
                )
                dst_grid = self.components[exchange.destination].grid
                dst_mask = (
                    self.components[exchange.destination].grid.mask
                    if self.components[exchange.destination].grid.mask
                    else None
                )
                self._regridders[key] = exchange.build(
                    src_grid, src_mask, dst_grid, dst_mask
                )

        # Initialize components
        for name, component in self.components.items():
            component.initialize(self)
            logger.info(f"Initialized {name}")

    def _do_exchanges(self, component: Union[Atmosphere, Ocean, SeaIce, Land], when: str) -> None:
        for exchange in self.exchanges:
            if exchange.when != when:
                continue
            if exchange.destination != component.name:
                continue

            source = self.components[exchange.source]
            destination = self.components[exchange.destination]

            logger.info(f" Exchange ({exchange.name}): {source.name} ---> {destination.name} ({when})")

            regridder = self._regridders[(exchange.source, exchange.destination)]
            source_fields = source.export_fields()
            destination_fields = {}

            for field_name in exchange.field_names:
                if isinstance(field_name, tuple):
                    flattened = list(
                        chain.from_iterable(
                            (item,) if isinstance(item, str) else item
                            for item in field_name
                        )
                    )
                    if not all(fkey in source_fields for fkey in flattened):
                        raise ValueError(
                            f"Not all fields in vector {field_name} are present in source fields"
                        )

                    (
                        destination_fields[field_name[0]],
                        destination_fields[field_name[1]],
                    ) = _vector_field_interpolate(field_name, source_fields, regridder)
                else:
                    destination_fields[field_name] = _scalar_field_interpolate(
                        field_name, source_fields, regridder
                    )

            if destination_fields:
                destination.import_fields(destination_fields)
                logger.debug(
                    f"{when.upper()} step: Exchanged {list(destination_fields)} from {exchange.source} to {exchange.destination}"
                )

    def run(self) -> None:
        self.initialize()
        for n, time, dt in self.clock.iter():
            logger.info(f" ====== Step: {n:05d} ====== Date: {time} ====== Δt: {dt} ")

            # Step components in declared order
            for cname in self.runseq:
                # Pre-step exchanges
                self._do_exchanges(self.components[cname], "pre")

                logger.info(f" Execute component: {cname}")
                self.components[cname].step(dt, time, self)

                # Post-step exchanges
                self._do_exchanges(self.components[cname], "post")
