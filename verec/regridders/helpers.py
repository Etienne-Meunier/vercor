import numpy as np
from ..grid import RectilinearGrid


def make_rectilinear_grid(
    name: str, nx: int, ny: int, x0: float, x1: float, y0: float, y1: float, mask=None, area=None
) -> RectilinearGrid:
    """Helper to build rectilinear grid"""

    x = np.linspace(x0, x1, nx, dtype=float)
    y = np.linspace(y0, y1, ny, dtype=float)
    return RectilinearGrid(name=name, x=x, y=y, mask=mask, area=area)
