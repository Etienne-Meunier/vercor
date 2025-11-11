import logging
from dataclasses import dataclass, field
from logging import Logger
from typing import Dict, List, Tuple, Union

from vercor.clock import Clock
from vercor.components import Atmosphere, Land, Ocean, SeaIce
from vercor.exchange import Exchange
from vercor.regridders.bilinear import BilinearRectilinearRegridder
from vercor.run_sequence import RunSequence
from vercor.settings import VercorSettings


def setup_logger():
    logger = logging.getLogger("VerCOR")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    return logger


@dataclass
class Coupler:
    clock: Clock
    logger: Logger = field(default_factory=setup_logger)
    run_sequence: RunSequence = field(init=False)
    components: Dict[str, Union[Atmosphere, Ocean, SeaIce, Land]] = field(
        default_factory=dict
    )
    exchanges: List[Exchange] = field(default_factory=list)
    settings: VercorSettings = field(default_factory=VercorSettings)
    _regridders: Dict[
        Tuple[str, str],
        BilinearRectilinearRegridder,
    ] = field(default_factory=dict)

    def register(self, component: Union[Atmosphere, Ocean, SeaIce, Land]) -> None:
        if component.name in self.components:
            raise ValueError(f"Component {component.name} already registered")

        self.components[component.name] = component
        self.logger.info(f" Registered component {component.name}")

    def add_exchange(self, exchange: Exchange) -> None:
        self.exchanges.append(exchange)
        formatted_field_names = ", ".join(
            ", ".join(item) if isinstance(item, tuple) else item
            for item in exchange.field_names
        )
        self.logger.info(
            f" Added exchange {exchange.name}: {exchange.source} -> {exchange.destination}:"
            f" Fields --- {formatted_field_names} --- Call order: {exchange.when}"
        )

    def set_components_run_sequence(self, run_sequence: RunSequence) -> None:
        for cname in run_sequence:
            if cname not in self.components.keys():
                raise ValueError(f"Component {cname} not registered in coupler")
        self.run_sequence = run_sequence
        self.logger.info(
            f" Set coupler components run sequence: {', '.join(self.run_sequence)}"
        )

    def initialize(self) -> None:
        self.logger.info(" Initializing coupler and components")

        # Initialize components
        for name, component in self.components.items():
            component.initialize(self)
            self.logger.info(f" Initialized {name}")

        # Build regridders per (source component, destination component) pair
        for exchange in self.exchanges:
            key = (exchange.source, exchange.destination)
            if key not in self._regridders:
                self._regridders[key] = exchange.create(
                    self.components[exchange.source].grid,
                    self.components[exchange.destination].grid,
                )
            else:
                self.logger.warning(
                    f" Regridder for exchange {exchange.name} already exists, skipping creation"
                )

    def _do_exchanges(
        self, component: Union[Atmosphere, Ocean, SeaIce, Land], when: str
    ) -> None:
        for exchange in self.exchanges:
            # Exchange before or after component stepping
            if exchange.when != when:
                continue

            # Ensure exchange for currently stepping component only
            if exchange.destination != component.name:
                continue

            source_component = self.components[exchange.source]
            destination_component = self.components[exchange.destination]

            self.logger.info(
                f" Exchange fields ({exchange.name}): {source_component.name} ---> {destination_component.name} ({when})"
            )

            regrid = self._regridders[(exchange.source, exchange.destination)]
            source_fields = source_component.export_fields()
            destination_fields = {}

            # Regridder (regrid) checks if components have identical grids internally and
            # returns fields as-is (from source to destination) if so, avoiding unnecessary computation
            for field_name in exchange.field_names:
                # Figure out if scalar or vector field to be regridded & passed to destination
                if isinstance(field_name, tuple):
                    field_name_set = set(field_name)
                    if not field_name_set.issubset(set(source_fields.keys())):
                        raise ValueError(
                            f"Not all fields in vector {field_name} are present in source fields"
                        )
                    (
                        destination_fields[field_name[0]],
                        destination_fields[field_name[1]],
                    ) = regrid(
                        source_fields[field_name[0]], source_fields[field_name[1]]
                    )
                else:
                    if field_name not in source_fields:
                        raise ValueError(
                            f"Field {field_name} not present in source fields"
                        )
                    destination_fields[field_name] = regrid(source_fields[field_name])

            if destination_fields:
                destination_component.import_fields(destination_fields)
                self.logger.debug(
                    f"{when.upper()} step: Exchanged {list(destination_fields)} from {exchange.source} to {exchange.destination}"
                )

    def run(self) -> None:
        for n, time, dt in self.clock.iter():
            self.logger.info(
                f" ====== Step: {n:05d} ====== Date: {time} ====== Δt: {dt} "
            )

            # Step components in declared order
            for cname in self.run_sequence:
                self._do_exchanges(self.components[cname], "pre")

                self.logger.info(f" Run component: {cname}")
                self.components[cname].step(dt, time, self)

                self._do_exchanges(self.components[cname], "post")
