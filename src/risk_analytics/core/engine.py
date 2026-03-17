from __future__ import annotations

import numpy as np
from scipy.stats import qmc

from .base import StochasticModel
from .paths import SimulationResult


class MonteCarloEngine:
    """Drives correlated multi-asset Monte Carlo simulation.

    Parameters
    ----------
    n_paths : int
        Number of simulation paths.
    seed : int | None
        Random seed for reproducibility.
    quasi_random : bool
        Use Sobol quasi-random sequences instead of pseudo-random normals.
    """

    def __init__(
        self,
        n_paths: int,
        seed: int | None = None,
        quasi_random: bool = False,
    ) -> None:
        self.n_paths = n_paths
        self.seed = seed
        self.quasi_random = quasi_random

    def run(
        self,
        models: list[StochasticModel],
        time_grid: np.ndarray,
        correlation_matrix: np.ndarray | None = None,
    ) -> dict[str, SimulationResult]:
        """Simulate all models jointly under a common correlation structure.

        Parameters
        ----------
        models : list[StochasticModel]
            Models to simulate; each declares how many factors it needs.
        time_grid : np.ndarray, shape (T,)
            Simulation time grid in years, starting at 0.
        correlation_matrix : np.ndarray | None, shape (total_factors, total_factors)
            Global correlation matrix across all model factors in the order
            models are listed. If None, identity (independence) is assumed.

        Returns
        -------
        dict[str, SimulationResult]
            Keyed by model.name.
        """
        n_steps = len(time_grid) - 1
        total_factors = sum(m.n_factors for m in models)

        # --- Generate raw standard normals ---
        # Shape: (n_paths, n_steps, total_factors)
        draws = self._generate_draws(n_paths=self.n_paths, n_steps=n_steps, total_factors=total_factors)

        # --- Apply Cholesky correlation ---
        if correlation_matrix is not None:
            corr = np.asarray(correlation_matrix, dtype=float)
            self._validate_correlation(corr, total_factors)
            L = np.linalg.cholesky(corr)
            # draws @ L.T applies correlation; shape preserved
            draws = draws @ L.T

        # --- Slice draws per model and simulate ---
        results: dict[str, SimulationResult] = {}
        offset = 0
        for model in models:
            nf = model.n_factors
            model_draws = draws[:, :, offset : offset + nf]
            results[model.name] = model.simulate(time_grid, self.n_paths, model_draws)
            offset += nf

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_draws(self, n_paths: int, n_steps: int, total_factors: int) -> np.ndarray:
        n_dims = n_steps * total_factors
        if self.quasi_random:
            sampler = qmc.Sobol(d=n_dims, scramble=True, seed=self.seed)
            # Sobol requires power-of-2 sample counts; round up then trim
            m = int(np.ceil(np.log2(max(n_paths, 2))))
            raw = sampler.random_base2(m)[:n_paths]
            draws_flat = qmc.scale(raw, 0, 1)
            # Convert uniform to normal via inverse CDF
            from scipy.stats import norm
            draws_flat = norm.ppf(np.clip(draws_flat, 1e-10, 1 - 1e-10))
        else:
            rng = np.random.default_rng(self.seed)
            draws_flat = rng.standard_normal(size=(n_paths, n_dims))

        return draws_flat.reshape(n_paths, n_steps, total_factors)

    @staticmethod
    def _validate_correlation(corr: np.ndarray, n: int) -> None:
        if corr.shape != (n, n):
            raise ValueError(
                f"correlation_matrix must be ({n}, {n}) for {n} total factors; got {corr.shape}"
            )
        if not np.allclose(corr, corr.T):
            raise ValueError("correlation_matrix must be symmetric")
        eigenvalues = np.linalg.eigvalsh(corr)
        if eigenvalues.min() < -1e-8:
            raise ValueError("correlation_matrix is not positive semi-definite")
