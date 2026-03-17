from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .paths import SimulationResult


class StochasticModel(ABC):
    """Abstract base class for all stochastic asset models."""

    @abstractmethod
    def simulate(
        self,
        time_grid: np.ndarray,
        n_paths: int,
        random_draws: np.ndarray,
    ) -> SimulationResult:
        """Simulate paths on the given time grid.

        Parameters
        ----------
        time_grid : np.ndarray, shape (T,)
            Simulation time points in years (must start at 0).
        n_paths : int
            Number of Monte Carlo paths.
        random_draws : np.ndarray, shape (n_paths, T-1, n_factors)
            Pre-generated (possibly correlated) standard normal draws.

        Returns
        -------
        SimulationResult
        """
        ...

    @abstractmethod
    def calibrate(self, market_data: dict) -> None:
        """Calibrate model parameters to market data.

        Parameters
        ----------
        market_data : dict
            Asset-class-specific market data (term structure, vol surface, etc.).
        """
        ...

    @abstractmethod
    def get_params(self) -> dict:
        """Return current model parameters as a dict."""
        ...

    @abstractmethod
    def set_params(self, params: dict) -> None:
        """Set model parameters from a dict."""
        ...

    @property
    @abstractmethod
    def n_factors(self) -> int:
        """Number of Brownian motion factors."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Model identifier string."""
        ...


class Pricer(ABC):
    """Abstract base class for all pricing models."""

    @abstractmethod
    def price(self, result: SimulationResult) -> np.ndarray:
        """Price the instrument on each simulated path at each time step.

        Parameters
        ----------
        result : SimulationResult
            Output from a StochasticModel.simulate() call.

        Returns
        -------
        np.ndarray, shape (n_paths, T)
            Mark-to-market value at each time point for each path.
        """
        ...
