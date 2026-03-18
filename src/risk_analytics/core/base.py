from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from .paths import SimulationResult

logger = logging.getLogger(__name__)


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
        """Return current model parameters as a JSON-serialisable dict."""
        ...

    @abstractmethod
    def set_params(self, params: dict) -> None:
        """Set model parameters from a dict (inverse of get_params)."""
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

    @property
    def interpolation_space(self) -> list:
        """
        Interpolation space per factor for sparse-grid path interpolation.
        "log" for log-normally distributed quantities (spots, FX rates).
        "linear" for Gaussian quantities (rates, log-spot processes).
        """
        return ["linear"] * self.n_factors

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialise calibrated parameters to a JSON file.

        The file contains the model name (for validation on load) and the
        full parameter dict returned by ``get_params()``. NumPy arrays are
        stored as JSON lists; scalar numpy types are converted to Python
        built-ins. The file can be reloaded with ``model.load(path)``.

        Parameters
        ----------
        path : str | Path
            Destination file path (e.g. ``"hw1f.json"``).
        """
        payload = {
            "model": self.name,
            "params": _to_serializable(self.get_params()),
        }
        Path(path).write_text(json.dumps(payload, indent=2))
        logger.debug("Saved %s params to %s", self.name, path)

    def load(self, path: str | Path) -> "StochasticModel":
        """Load calibrated parameters from a JSON file into this instance.

        Validates that the saved model name matches ``self.name``, then
        calls ``set_params()`` with the stored values. NumPy array params
        (e.g. ``theta`` in HullWhite1F) are passed as plain Python lists
        — each model's ``set_params()`` converts them as needed.

        Parameters
        ----------
        path : str | Path
            JSON file previously written by ``save()``.

        Returns
        -------
        self (for chaining: ``model = HullWhite1F().load("hw1f.json")``)

        Raises
        ------
        ValueError
            If the saved model name does not match this instance's name.
        FileNotFoundError
            If ``path`` does not exist.
        """
        payload = json.loads(Path(path).read_text())
        saved = payload.get("model", "")
        if saved != self.name:
            raise ValueError(
                f"File contains params for '{saved}', but this model is '{self.name}'."
            )
        self.set_params(payload["params"])
        logger.debug("Loaded %s params from %s", self.name, path)
        return self


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _to_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to JSON-native Python types."""
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj  # None, bool, int, float, str — already JSON-safe


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

    def cashflow_times(self) -> list:
        """
        Return the list of known cashflow/payment times (in years).
        Used by the pipeline to augment the sparse simulation grid with
        hard nodes at each payment date, preventing interpolation across
        discontinuities in MTM.
        Override in subclasses for instruments with scheduled payments.
        """
        return []

    def price_at(self, result: SimulationResult, t_idx: int) -> np.ndarray:
        """Return MTM for all paths at a single time index.

        Default delegates to ``price(result)[:, t_idx]``.
        Override for efficient O(1)-in-T implementations that only need
        the risk-factor slice at ``t_idx`` (e.g. vanilla IRS, bonds).

        Returns
        -------
        np.ndarray, shape (n_paths,)
        """
        return self.price(result)[:, t_idx]
