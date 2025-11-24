import logging
from dataclasses import dataclass, field
from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from vercor.clock import Clock
from vercor.components import Atmosphere, Land, Ocean, SeaIce
from vercor.components.base import Shared
from vercor.components.base import TimedNamedArray as TNA
from vercor.components.data.era5_atmosphere import ERA5Atmosphere
from vercor.components.data.era5_ocean import ERA5Ocean
from vercor.components.data.erainterim_ocean import ERAInterimOcean
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
    components: Dict[
        str,
        Union[
            Atmosphere, ERA5Atmosphere, Ocean, ERA5Ocean, ERAInterimOcean, SeaIce, Land
        ],
    ] = field(default_factory=dict)
    exchanges: List[Exchange] = field(default_factory=list)
    settings: VercorSettings = field(default_factory=VercorSettings)
    _regridders: Dict[
        Tuple[str, str],
        BilinearRectilinearRegridder,
    ] = field(default_factory=dict)
    """
    Main coupler class to manage components and exchanges between them.

    Attributes:
        logger: Logger instance for coupler logging
        run_sequence: sequence of component names defining the call (step) order
        components: mapping of component name to component instance
        exchanges: list of all Exchange instances
        _regridders: mapping of (source component name, destination component name)
                     to Regridder instance (a pool of all available regridders)
    """

    def register(
        self,
        component: Union[
            Atmosphere, ERA5Atmosphere, Ocean, ERA5Ocean, ERAInterimOcean, SeaIce, Land
        ],
    ) -> None:
        if component.name in self.components:
            raise KeyError(f"Component {component.name} already registered")

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
            f" Fields ({formatted_field_names})"
        )

    def set_components_run_sequence(self, run_sequence: RunSequence) -> None:
        for cname in run_sequence:
            if cname not in self.components.keys():
                raise KeyError(f"Component {cname} not registered in coupler")
        self.run_sequence = run_sequence
        self.logger.info(
            f" Set coupler components run sequence: {', '.join(self.run_sequence)}"
        )

    def initialize(self) -> None:
        self.logger.info(" Initializing coupler and components")

        # Initialize components
        for name, component in self.components.items():
            component.initialize(self)
            component.check_not_empty_import_export_lists()
            component.send_fields(self.clock.start, self)
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

    def interpolate_and_dispatch_fields(
        self,
        timestamp: datetime,
        component: Union[
            Atmosphere, ERA5Atmosphere, Ocean, ERA5Ocean, ERAInterimOcean, SeaIce, Land
        ],
    ) -> None:
        for exchange in self.exchanges:
            # Ensure exchange for currently stepping component only
            if exchange.destination != component.name:
                continue

            source_component = self.components[exchange.source]
            destination_component = self.components[exchange.destination]

            self.logger.info(
                f" Exchange fields ({exchange.name}): {source_component.name} ---> {destination_component.name}"
            )

            regrid = self._regridders[(exchange.source, exchange.destination)]
            source_fields = source_component.export_fields()
            destination_fields = Shared()

            # Regridder (regrid) checks if components have identical grids internally and
            # returns fields as-is (from source to destination) if so, avoiding unnecessary computation
            for field_name in exchange.field_names:
                # Figure out if scalar or vector field to be regridded & passed to destination
                if isinstance(field_name, tuple):
                    field_name_set = set(field_name)
                    if not field_name_set.issubset(set(source_fields.fields().keys())):
                        raise RuntimeError(
                            f"Not all fields in vector {field_name} are present in source fields"
                        )
                    (
                        u_vector,
                        v_vector,
                    ) = regrid(
                        getattr(source_fields, field_name[0]).data,
                        getattr(source_fields, field_name[1]).data,
                        src_mask=self.components[exchange.source].grid.binary_mask,
                    )
                    setattr(
                        destination_fields,
                        field_name[0],
                        TNA(u_vector, timestamp, exchange.source),
                    )
                    setattr(
                        destination_fields,
                        field_name[1],
                        TNA(v_vector, timestamp, exchange.source),
                    )
                else:
                    if field_name not in source_fields.fields().keys():
                        raise KeyError(
                            f"Field {field_name} not present in source fields"
                        )

                    # to pass mypy type checking
                    scalar = np.asarray(
                        regrid(getattr(source_fields, field_name).data,
                               src_mask=self.components[exchange.source].grid.binary_mask)
                    )

                    setattr(
                        destination_fields,
                        field_name,
                        TNA(scalar, timestamp, exchange.source),
                    )

            if not destination_fields.is_empty:
                destination_component.import_fields(destination_fields)
                self.logger.debug(
                    f" Exchanged {destination_fields.field_names}"
                    f" from {exchange.source} to {exchange.destination}"
                )

    def finalize(self, output_file_mask: Optional[Path] = None) -> None:
        self.logger.info(" ------------ Finalizing coupler and components ------------")
        for name, component in self.components.items():
            component.finalize(output_file_mask)
            self.logger.info(f" Finalized {name}")

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"├── Run start: {self.clock.start}\n"
            f"├── Components: "
            + ", ".join(
                f"<{component.__class__.__name__}>({name})"
                for name, component in self.components.items()
            )
            + "\n"
            f"├── Exchanges: {', '.join(exchange.name for exchange in self.exchanges)}\n"
            f"└── Run sequence: {', '.join(self.run_sequence)}"
        )

    def run(self) -> None:
        # TODO: add setup checks like time step consistency,
        # component's readiness (outgoing fields), etc.
        # Wrap in a class method or function
        for cname in self.run_sequence:
            if self.components[cname].outgoing_fields.is_empty:
                raise RuntimeError(
                    f"Component {cname} outgoing fields were not initialized properly."
                )

        for n, time, dt in self.clock.iter():
            self.logger.info(
                f" ====== Step: {n:05d} ====== Date: {time} ====== Δt: {dt} "
            )

            # Step components in declared order
            for cname in self.run_sequence:
                self.interpolate_and_dispatch_fields(time, self.components[cname])

                self.logger.info(f" Run component: {cname}")
                self.components[cname].receive_fields(time)

                # TODO: add sub-steps for individual components if needed
                self.components[cname].step(dt, time, self)

                self.components[cname].send_fields(time, self)
