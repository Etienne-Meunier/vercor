from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class RunSequence:
    """Public ordered component-name schedule used by ``Coupler.run()``."""

    order: list[str] = field(default_factory=list)

    def __iter__(self) -> Iterator[str]:
        return iter(self.order)
