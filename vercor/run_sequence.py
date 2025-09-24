from dataclasses import dataclass, field
from typing import Iterator, List


@dataclass
class RunSequence:
    # Ordered component names for stepping
    order: List[str] = field(default_factory=list)

    def __iter__(self) -> Iterator[str]:
        return iter(self.order)
