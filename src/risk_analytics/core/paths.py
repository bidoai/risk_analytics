from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SimulationResult:
    """Container for Monte Carlo simulation output.

    Attributes
    ----------
    time_grid : np.ndarray, shape (T,)
        Simulation time points in years.
    paths : np.ndarray, shape (n_paths, T, n_factors)
        Simulated state variable paths.
    model_name : str
        Name of the model that produced these paths.
    factor_names : list[str]
        Labels for each factor dimension.
    """

    time_grid: np.ndarray
    paths: np.ndarray
    model_name: str
    factor_names: list[str] = field(default_factory=list)

    @property
    def n_paths(self) -> int:
        return self.paths.shape[0]

    @property
    def n_steps(self) -> int:
        return self.paths.shape[1]

    @property
    def n_factors(self) -> int:
        return self.paths.shape[2]

    def factor(self, name: str) -> np.ndarray:
        """Return paths for a single factor by name, shape (n_paths, T)."""
        idx = self.factor_names.index(name)
        return self.paths[:, :, idx]

    def factor_at(self, name: str, t_idx: int) -> np.ndarray:
        """Return all paths for a factor at a specific time index, shape (n_paths,)."""
        idx = self.factor_names.index(name)
        return self.paths[:, t_idx, idx]
