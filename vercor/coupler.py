import logging
from dataclasses import dataclass, field
from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from vercor.clock import Clock
from vercor.components import Atmosphere, Land, Ocean
from vercor.components.base import Shared
from vercor.components.base import TimedNamedArray as TNA
from vercor.components.data.era5_atmosphere import ERA5Atmosphere
from vercor.components.data.era5_land import ERA5Land
from vercor.components.data.era5_ocean import ERA5Ocean
from vercor.components.data.erainterim_ocean import ERAInterimOcean
from vercor.exchange import Exchange
from vercor.interpolators.conservative_remap_rectilinear import (
    ConservativeRectilinearRemapper,
)
from vercor.regridders import BilinearRectilinearRegridder
from vercor.regridders import ConservativeRectilinearRegridder
from vercor.regridders import compute_grid_fraction_rectilinear
from vercor.regridders.helpers import centers_to_edges, compute_land_mask
from vercor.run_sequence import RunSequence
from vercor.settings import VercorSettings
from vercor.tools import grids_identical, get_component
from vercor.types import AllComponentsType


def setup_logger():
    """
    Setup and return a logger for the Coupler.
    """
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
        AllComponentsType,
    ] = field(default_factory=dict)
    exchanges: List[Exchange] = field(default_factory=list)
    settings: VercorSettings = field(default_factory=VercorSettings)
    _regridders: Dict[
        Tuple[str, str],
        BilinearRectilinearRegridder | ConservativeRectilinearRegridder,
    ] = field(default_factory=dict)
    _binary_masks: Dict[Tuple[str, str], NDArray] = field(default_factory=dict)
    _fractional_masks: Dict[Tuple[str, str], NDArray] = field(default_factory=dict)

    """
    Main coupler class to manage components and exchanges between them.

    Attributes:
        clock: Clock instance for managing simulation time
        logger: Logger instance for coupler logging
        run_sequence: sequence of component names defining the call (step) order
        components: mapping of component name to component instance
        exchanges: list of all Exchange instances
        settings: VercorSettings instance for coupler settings
        _regridders: mapping of (source component name, destination component name)
                to Regridder instance (a pool of all available regridders)
        _binary_masks: mapping of (source component name, destination component name)
                to a binary mask NDArray. This mask is used during regridding of fields
                to ensure that only valid (e.g., ocean or land) points are considered
                during the regridding process.
        _fractional_masks: mapping of (source component name, destination component name)
                to a fractional mask NDArray. This mask is applied during field exchanges
                after regridding to ensure that only the appropriate portion from source
                grid cells of the forcing or boundary conditions is transferred to 
                destination grid cells, reflecting the partial coverage of source grid cells
                within destination grid cells.
    """

    def register(
        self,
        component: AllComponentsType,
    ) -> None:
        """
        Register a component with the coupler.

        Arguments:
            component: component instance to register
        """

        if component.name in self.components:
            raise KeyError(f"Component {component.name} already registered")

        self.components[component.name] = component
        self.logger.info(f" Registered component {component.name}")

    def add_exchange(self, exchange: Exchange) -> None:
        """
        Add an exchange definition to the coupler.

        Arguments:
            exchange: Exchange instance defining the exchange between components to add
        """

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
        """
        Set the run sequence for the coupler components.

        Arguments:
            run_sequence: RunSequence instance defining the order of components execution
        """

        for cname in run_sequence:
            if cname not in self.components.keys():
                raise KeyError(f"Component {cname} not registered in coupler")
        self.run_sequence = run_sequence
        self.logger.info(
            f" Set coupler components run sequence: {', '.join(self.run_sequence)}"
        )

    def initialize(self) -> None:
        """
        Initialize the coupler and all registered components.
        """

        self.logger.info(" Initializing coupler and components")

        # Initialize each component
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

        self._create_exchange_masks()

    def _create_exchange_masks(self) -> None:
        land_component = get_component(self.components, (Land, ERA5Land), "land")
        atmosphere_component = get_component(
            self.components, (Atmosphere, ERA5Atmosphere), "atmosphere"
        )
        ocean_component = get_component(
            self.components, (Ocean, ERA5Ocean, ERAInterimOcean), "ocean"
        )

        if not grids_identical(land_component.grid, atmosphere_component.grid):
            raise RuntimeError(
                "Land and atmospheric components must use identical horizontal grids"
            )

        # Regrid the binary mask from the mask origin component
        # to the destination component grid
        regridder = ConservativeRectilinearRegridder(
            ocean_component.grid,
            atmosphere_component.grid,
        )

        ocean_bmask = np.asarray(ocean_component.grid.binary_mask)
        if ocean_bmask is None:
            raise RuntimeError(
                f"Ocean component {ocean_component.name} has no binary mask defined"
            )

        ocn_bmask_on_atm_grid = np.asarray(regridder(ocean_bmask))
        ocn_bmask_on_atm_grid = np.clip(ocn_bmask_on_atm_grid, 0.0, 1.0)
        lnd_bmask_on_atm_grid = compute_land_mask(ocn_bmask_on_atm_grid)

        if regridder.interpolator is not None and isinstance(
            regridder.interpolator, ConservativeRectilinearRemapper
        ):
            src_total_mass = regridder.interpolator.get_src_total_mass(ocean_bmask)
            dst_total_mass = regridder.interpolator.get_dst_total_mass(
                ocn_bmask_on_atm_grid
            )

            if not np.isclose(src_total_mass, dst_total_mass, atol=1e-6):
                raise RuntimeError(
                    "Regridding ocean binary mask to atmospheric grid does not conserve total mass "
                    f"(source mass: {src_total_mass}, destination mass: {dst_total_mass})"
                )

        bmask_sum = lnd_bmask_on_atm_grid + ocn_bmask_on_atm_grid
        min_sum = bmask_sum.min()
        if not np.isclose(min_sum, 1.0, atol=1e-12):
            raise RuntimeError(
                "Binary land and ocean masks on atmospheric grid must sum to approx. 1 everywhere "
                f"(minimum sum {min_sum})"
            )

        self._binary_masks[(ocean_component.name, atmosphere_component.name)] = (
            ocn_bmask_on_atm_grid
        )
        self._binary_masks[(land_component.name, atmosphere_component.name)] = (
            lnd_bmask_on_atm_grid
        )

        ocn_fmask_on_atm_grid = compute_grid_fraction_rectilinear(
            centers_to_edges(atmosphere_component.grid.latitude, "lat"),
            centers_to_edges(atmosphere_component.grid.longitude, "lon"),
            centers_to_edges(ocean_component.grid.latitude, "lat"),
            centers_to_edges(ocean_component.grid.longitude, "lon"),
            ocean_bmask.astype(bool),
        )
        lnd_fmask_on_atm_grid = 1.0 - ocn_fmask_on_atm_grid

        fmask_sum = lnd_fmask_on_atm_grid + ocn_fmask_on_atm_grid
        min_fsum = fmask_sum.min()
        max_fsum = fmask_sum.max()
        if not (
            np.isclose(min_fsum, 1.0, atol=1e-3)
            and np.isclose(max_fsum, 1.0, atol=1e-3)
        ):
            raise RuntimeError(
                "Fractional land and ocean masks on atmospheric grid must sum to approx. 1 everywhere "
                f"(minimum sum {min_fsum}, maximum sum {max_fsum})"
            )

        self._fractional_masks[(ocean_component.name, atmosphere_component.name)] = (
            ocn_fmask_on_atm_grid
        )
        self._fractional_masks[(land_component.name, atmosphere_component.name)] = (
            lnd_fmask_on_atm_grid
        )

    def interpolate_and_dispatch_fields(
        self,
        timestamp: datetime,
        component: AllComponentsType,
    ) -> None:
        """
        Interpolate and dispatch fields to the given component at the specified timestamp.

        Arguments:
            timestamp: current simulation (coupler's) time
            component: destination component instance to process exchanges for
        """

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
                    scalar = np.asarray(regrid(getattr(source_fields, field_name).data))

                    setattr(
                        destination_fields,
                        field_name,
                        TNA(scalar, timestamp, exchange.source),
                    )

            # TODO: apply fractional mask if available

            if not destination_fields.is_empty:
                destination_component.import_fields(destination_fields)
                self.logger.debug(
                    f" Exchanged {destination_fields.field_names}"
                    f" from {exchange.source} to {exchange.destination}"
                )

    def finalize(self, output_file_mask: Optional[Path] = None) -> None:
        """
        Finalize the coupler and all registered components.

        Arguments:
            output_file_mask: optional path mask for output files
        """

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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(runstart={self.clock.start}, run_sequence={'-> '.join(self.run_sequence)})"

    def run(self) -> None:
        """
        Run the coupler and all registered components according to the run sequence.
        """

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
