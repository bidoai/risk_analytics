from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.stats import qmc

from .base import StochasticModel
from .paths import SimulationResult

logger = logging.getLogger(__name__)


class MonteCarloEngine:
    """Drives correlated multi-asset Monte Carlo simulation.

    Parameters
    ----------
    n_paths : int
        Number of simulation paths. Must be even when ``antithetic=True``.
    seed : int | None
        Random seed for reproducibility.
    quasi_random : bool
        Use Sobol quasi-random sequences instead of pseudo-random normals.
    antithetic : bool
        Enable antithetic variates variance reduction. Generates ``n_paths // 2``
        base draws Z and mirrors them as -Z, giving n_paths total paths. The
        paired paths are negatively correlated, so estimator variance drops by
        roughly (1 + ρ) / 2 where ρ is the within-pair correlation of the payoff
        — for symmetric payoffs (e.g. log-normal terminal price) the reduction can
        be near 100 % for the mean, and significant for option prices.
        Requires ``n_paths`` to be even.
    """

    def __init__(
        self,
        n_paths: int,
        seed: int | None = None,
        quasi_random: bool = False,
        antithetic: bool = False,
        parallel_models: bool = False,
    ) -> None:
        if antithetic and n_paths % 2 != 0:
            raise ValueError(
                f"n_paths must be even when antithetic=True; got {n_paths}."
            )
        self.n_paths = n_paths
        self.seed = seed
        self.quasi_random = quasi_random
        self.antithetic = antithetic
        self.parallel_models = parallel_models

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
        model_names = [m.name for m in models]

        logger.info(
            "MonteCarloEngine.run: n_paths=%d  n_steps=%d  models=%s  "
            "antithetic=%s  quasi_random=%s",
            self.n_paths, n_steps, model_names, self.antithetic, self.quasi_random,
        )

        # --- Generate raw standard normals ---
        # Shape: (n_paths, n_steps, total_factors)
        draws = self._generate_draws(n_paths=self.n_paths, n_steps=n_steps, total_factors=total_factors)
        logger.debug("Random draws generated: shape=%s", draws.shape)

        # --- Apply Cholesky correlation ---
        if correlation_matrix is not None:
            corr = np.asarray(correlation_matrix, dtype=float)
            self._validate_correlation(corr, total_factors)
            L = np.linalg.cholesky(corr)
            # draws @ L.T applies correlation; shape preserved
            draws = draws @ L.T
            logger.debug("Cholesky correlation applied: corr_matrix shape=%s", corr.shape)

        # --- Slice draws per model ---
        slices: list[tuple[StochasticModel, np.ndarray]] = []
        offset = 0
        for model in models:
            nf = model.n_factors
            slices.append((model, draws[:, :, offset : offset + nf]))
            logger.debug("Simulating %s (factors=%d, draw_slice=[%d:%d])", model.name, nf, offset, offset + nf)
            offset += nf

        # --- Simulate: parallel (threads) or serial ---
        # Each model is independent after draw slicing.  NumPy and Numba both
        # release the GIL, so ThreadPoolExecutor achieves real concurrency with
        # no serialization cost (thread-shared memory, no pickling).
        results: dict[str, SimulationResult] = {}

        def _simulate(pair: tuple[StochasticModel, np.ndarray]) -> tuple[str, SimulationResult]:
            model, model_draws = pair
            result = model.simulate(time_grid, self.n_paths, model_draws)
            result.interpolation_space = model.interpolation_space
            return model.name, result

        if self.parallel_models and len(slices) > 1:
            with ThreadPoolExecutor(max_workers=len(slices)) as pool:
                for name, result in pool.map(_simulate, slices):
                    results[name] = result
        else:
            for pair in slices:
                name, result = _simulate(pair)
                results[name] = result

        logger.info("MonteCarloEngine.run complete: %d models simulated", len(results))
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_draws(self, n_paths: int, n_steps: int, total_factors: int) -> np.ndarray:
        n_dims = n_steps * total_factors

        if self.antithetic:
            # Generate n_paths // 2 base draws, then mirror as -Z.
            n_base = n_paths // 2
            base = self._raw_draws(n_base, n_dims)
            draws_flat = np.concatenate([base, -base], axis=0)  # (n_paths, n_dims)
        else:
            draws_flat = self._raw_draws(n_paths, n_dims)

        return draws_flat.reshape(n_paths, n_steps, total_factors)

    def _raw_draws(self, n: int, n_dims: int) -> np.ndarray:
        """Generate n × n_dims standard normal draws (pseudo- or quasi-random)."""
        if self.quasi_random:
            sampler = qmc.Sobol(d=n_dims, scramble=True, seed=self.seed)
            m = int(np.ceil(np.log2(max(n, 2))))
            raw = sampler.random_base2(m)[:n]
            from scipy.stats import norm
            return norm.ppf(np.clip(raw, 1e-10, 1 - 1e-10))
        else:
            rng = np.random.default_rng(self.seed)
            return rng.standard_normal(size=(n, n_dims))

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
