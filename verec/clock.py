from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterator, Tuple


@dataclass
class Clock:
    start: datetime
    dt_seconds: float
    steps: int

    def iter(self) -> Iterator[Tuple[int, datetime, timedelta]]:
        t = self.start
        dt = timedelta(seconds=self.dt_seconds)
        for n in range(self.steps):
            yield n, t, dt
            t = t + dt


if __name__ == "__main__":
    clock = Clock(
        start=datetime(2020, 1, 1, 0, 0, 0),
        dt_seconds=3600,
        steps=5,
    )
    for n, t, dt in clock.iter():
        print(f"Step {n}: Time {t}, Delta {dt}")
