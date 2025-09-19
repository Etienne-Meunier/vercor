from dataclasses import dataclass, field
from itertools import chain
from typing import Dict, List, Tuple, Union, Callable
import logging

import numpy as np
from scipy.fftpack import dct
from verec.components import Atmosphere, Ocean, SeaIce, Land
from verec.regridders import (
    BilinearRectilinear,
    XESMFBilinearRectilinear,
    XESMFConservative_normed,
)

from verec.fields import Field
from verec.clock import Clock
from verec.exchange import Exchange
from verec.run_sequence import RunSequence


logger = logging.getLogger("VerEC.coupler")
logging.basicConfig(level=logging.INFO)


def _scalar_field_interpolate(
    fname: str,
    source_fields: Dict[str, Field],
    regridder: Callable,
) -> Field:
    destination_field = regridder(source_fields[fname])
    return destination_field


def _vector_field_interpolate(
    fname: Tuple[str, str],
    source_fields: Dict[str, Field],
    regridder: Callable,
) -> Tuple[Field, Field]:
    if len(fname) == 2:
        src_fname, alt_fname = fname
    else:
        raise ValueError("Vector field name must be a tuple of two strings")
    destination_field_lon, destination_field_lat = regridder(
        source_fields[src_fname], source_fields[alt_fname]
    )
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
        Union[BilinearRectilinear, XESMFBilinearRectilinear, XESMFConservative_normed],
    ] = field(default_factory=dict)

    def register(self, component: Union[Atmosphere, Ocean, SeaIce, Land]) -> None:
        if component.name in self.components:
            raise ValueError(f"Component {component.name} already registered")
        self.components[component.name] = component
        logger.info(f"Registered component {component.name}")

    def add_exchange(self, ex: Exchange) -> None:
        self.exchanges.append(ex)
        logger.info(
            f"Added exchange {ex.name}: {ex.source} -> {ex.destination} {ex.field_names} [{ex.when}]"
        )

    def initialize(self) -> None:
        # Build regridders per (src, dst)
        for ex in self.exchanges:
            key = (ex.source, ex.destination)
            if key not in self._regridders:
                src_grid = self.components[ex.source].grid
                src_mask = (
                    self.components[ex.source].grid.mask
                    if self.components[ex.source].grid.mask
                    else None
                )
                dst_grid = self.components[ex.destination].grid
                dst_mask = (
                    self.components[ex.destination].grid.mask
                    if self.components[ex.destination].grid.mask
                    else None
                )
                self._regridders[key] = ex.build(src_grid, src_mask, dst_grid, dst_mask)

        # Initialize components
        for name, comp in self.components.items():
            comp.initialize(self)
            logger.info(f"Initialized {name}")

    def _do_exchanges(self, when: str) -> None:
        for ex in self.exchanges:
            if ex.when != when:
                continue

            source = self.components[ex.source]
            destination = self.components[ex.destination]
            regridder = self._regridders[(ex.source, ex.destination)]
            source_fields = source.export_fields()
            destination_fields = {}

            for fname in ex.field_names:
                if isinstance(fname, tuple):
                    flattened = list(
                        chain.from_iterable(
                            (item,) if isinstance(item, str) else item for item in fname
                        )
                    )
                    if not all(fkey in source_fields for fkey in flattened):
                        raise ValueError(
                            f"Not all fields in vector {fname} are present in source fields"
                        )

                    destination_fields[fname[0]], destination_fields[fname[1]] = (
                        _vector_field_interpolate(fname, source_fields, regridder)
                    )
                else:
                    destination_fields[fname] = _scalar_field_interpolate(
                        fname, source_fields, regridder
                    )

            if destination_fields:
                destination.import_fields(destination_fields)
                logger.debug(
                    f"Exchanged {list(destination_fields)} from {ex.source} to {ex.destination}"
                )

    def run(self) -> None:
        self.initialize()
        for n, time, dt in self.clock.iter():
            logger.info(f" ====== Step: {n:05d} ====== Date: {time} ====== Δt: {dt} ")
            # Pre-step exchanges
            self._do_exchanges("pre")
            # Step components in declared order
            for cname in self.runseq:
                self.components[cname].step(dt, time, self)
            # Post-step exchanges
            self._do_exchanges("post")
