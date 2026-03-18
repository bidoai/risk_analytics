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
    interpolation_space : list[str]
        Interpolation space per factor. Each element is "log" or "linear".
        "log" for log-normally distributed quantities (spots, FX rates).
        "linear" for Gaussian quantities (rates, log-spot processes).
        Default: ["linear"] for a single-factor result.
    model : object
        Reference to the StochasticModel that produced this result. Used by
        pricers to access model-specific discount factor formulas (e.g.
        Hull-White A(t,T)·exp(-B(t,T)·r)). Optional.
    """

    time_grid: np.ndarray
    paths: np.ndarray
    model_name: str
    factor_names: list[str] = field(default_factory=list)
    interpolation_space: list = field(default_factory=lambda: ["linear"])
    model: object = field(default=None)

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

    def at(self, t: float) -> np.ndarray:
        """Return interpolated factor values at time t.

        Parameters
        ----------
        t : float
            Time point in years.

        Returns
        -------
        np.ndarray, shape (n_paths, n_factors)
            Interpolated state values at time t for each path and factor.
        """
        # Check if t is already in the time grid (within tolerance)
        tol = 1e-9
        diffs = np.abs(self.time_grid - t)
        idx_exact = np.argmin(diffs)
        if diffs[idx_exact] < tol:
            return self.paths[:, idx_exact, :]

        # Find bracketing indices
        lo = np.searchsorted(self.time_grid, t, side="right") - 1
        hi = lo + 1

        # Clamp to valid range
        lo = max(0, min(lo, len(self.time_grid) - 2))
        hi = lo + 1

        t_lo = self.time_grid[lo]
        t_hi = self.time_grid[hi]
        dt = t_hi - t_lo
        if dt < tol:
            return self.paths[:, lo, :]

        frac = (t - t_lo) / dt

        n_factors = self.n_factors
        result = np.empty((self.n_paths, n_factors))

        # Ensure interpolation_space has enough entries
        spaces = list(self.interpolation_space)
        while len(spaces) < n_factors:
            spaces.append("linear")

        for f in range(n_factors):
            v_lo = self.paths[:, lo, f]
            v_hi = self.paths[:, hi, f]
            space = spaces[f]

            if space == "log":
                # Guard against non-positive values
                v_lo_safe = np.where(v_lo > 0, v_lo, 1e-300)
                v_hi_safe = np.where(v_hi > 0, v_hi, 1e-300)
                result[:, f] = np.exp(
                    np.log(v_lo_safe) + frac * (np.log(v_hi_safe) - np.log(v_lo_safe))
                )
            else:
                result[:, f] = v_lo + frac * (v_hi - v_lo)

        return result

    def at_times(self, times: list) -> np.ndarray:
        """Return interpolated factor values at multiple time points.

        Parameters
        ----------
        times : list of float
            Time points in years.

        Returns
        -------
        np.ndarray, shape (n_paths, len(times), n_factors)
        """
        slices = [self.at(t) for t in times]
        return np.stack(slices, axis=1)
