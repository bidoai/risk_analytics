from __future__ import annotations

import numpy as np


class TimeGrid:
    """Utility for building simulation time grids."""

    @staticmethod
    def uniform(maturity: float, n_steps: int) -> np.ndarray:
        """Evenly-spaced time grid from 0 to maturity."""
        return np.linspace(0.0, maturity, n_steps + 1)

    @staticmethod
    def from_dates(dates: list[float]) -> np.ndarray:
        """Build a time grid from a list of year fractions (must start at 0)."""
        grid = np.asarray(dates, dtype=float)
        if grid[0] != 0.0:
            grid = np.concatenate([[0.0], grid])
        return np.sort(grid)

    @staticmethod
    def dt(time_grid: np.ndarray) -> np.ndarray:
        """Time steps between grid points, shape (T-1,)."""
        return np.diff(time_grid)
