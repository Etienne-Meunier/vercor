from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union
import logging

from .components import Atmosphere, Ocean, SeaIce, Land
from .regridders import XESMFBilinear, XESMFConservative_normed

from .clock import Clock
from .exchange import Exchange
from .run_sequence import RunSequence


logger = logging.getLogger("VerEC.coupler")
logging.basicConfig(level=logging.INFO)


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
        Tuple[str, str], Union[XESMFBilinear, XESMFConservative_normed]
    ] = field(default_factory=dict)

    def register(self, component: Union[Atmosphere, Ocean, SeaIce, Land]) -> None:
        if component.name in self.components:
            raise ValueError(f"Component {component.name} already registered")
        self.components[component.name] = component
        logger.info(f"Registered component {component.name}")

    def add_exchange(self, ex: Exchange) -> None:
        self.exchanges.append(ex)
        logger.info(
            f"Added exchange {ex.name}: {ex.src} -> {ex.dst} {ex.field_names} [{ex.when}]"
        )

    def initialize(self) -> None:
        # Build regridders per (src,dst)
        for ex in self.exchanges:
            key = (ex.src, ex.dst)
            if key not in self._regridders:
                srcg = self.components[ex.src].grid
                dstg = self.components[ex.dst].grid
                self._regridders[key] = ex.build(srcg, dstg)

        # Initialize components
        for name, comp in self.components.items():
            comp.initialize(self)
            logger.info(f"Initialized {name}")

    def _do_exchanges(self, when: str) -> None:
        for ex in self.exchanges:
            if ex.when != when:
                continue
            src = self.components[ex.src]
            dst = self.components[ex.dst]
            regridder = self._regridders[(ex.src, ex.dst)]
            src_fields = src.export_fields()
            out_fields = {}
            for fname in ex.field_names:
                if fname not in src_fields:
                    continue
                out_fields[fname] = regridder(src_fields[fname])
            if out_fields:
                dst.receive_fields(out_fields)
                logger.debug(f"Exchanged {list(out_fields)} from {ex.src} to {ex.dst}")

    def run(self) -> None:
        self.initialize()
        for n, t, dt in self.clock.iter():
            logger.info(f"=== Step {n} t={t} dt={dt} ===")
            # Pre-step exchanges
            self._do_exchanges("pre")
            # Step components in declared order
            for cname in self.runseq:
                self.components[cname].step(dt, t, self)
            # Post-step exchanges
            self._do_exchanges("post")
