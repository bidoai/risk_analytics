from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from risk_analytics.core.base import StochasticModel
from risk_analytics.core.paths import SimulationResult


class GeometricBrownianMotion(StochasticModel):
    """Geometric Brownian Motion equity model.

    dS(t) = μ·S(t) dt + σ·S(t) dW(t)

    Simulated via the exact log-normal solution:
    S(t+dt) = S(t) · exp((μ - σ²/2)·dt + σ·√dt·Z)

    Parameters
    ----------
    S0 : float
        Initial spot price.
    mu : float
        Drift (annualised).
    sigma : float
        Volatility (annualised).
    """

    def __init__(self, S0: float = 100.0, mu: float = 0.05, sigma: float = 0.20) -> None:
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma

    @property
    def n_factors(self) -> int:
        return 1

    @property
    def name(self) -> str:
        return "GBM"

    def simulate(
        self,
        time_grid: np.ndarray,
        n_paths: int,
        random_draws: np.ndarray,
    ) -> SimulationResult:
        """Exact log-normal simulation.

        Parameters
        ----------
        time_grid : np.ndarray, shape (T,)
        n_paths : int
        random_draws : np.ndarray, shape (n_paths, T-1, 1)

        Returns
        -------
        SimulationResult with factor 'S', shape (n_paths, T, 1)
        """
        T = len(time_grid)
        dt = np.diff(time_grid)

        log_paths = np.empty((n_paths, T))
        log_paths[:, 0] = np.log(self.S0)

        dW = random_draws[:, :, 0]  # (n_paths, T-1)

        increments = (self.mu - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * dW
        log_paths[:, 1:] = log_paths[:, 0:1] + np.cumsum(increments, axis=1)

        paths = np.exp(log_paths)

        return SimulationResult(
            time_grid=time_grid,
            paths=paths[:, :, np.newaxis],
            model_name=self.name,
            factor_names=["S"],
        )

    def calibrate(self, market_data: dict) -> None:
        """Calibrate sigma to ATM implied vol; optionally set mu from risk-free rate.

        Expected market_data keys:
        - 'S0': float, current spot
        - 'atm_vol': float, ATM implied volatility
        - 'mu' (optional): float, drift (e.g. risk-free rate - dividend yield)
        """
        if "S0" in market_data:
            self.S0 = float(market_data["S0"])
        if "atm_vol" in market_data:
            self.sigma = float(market_data["atm_vol"])
        if "mu" in market_data:
            self.mu = float(market_data["mu"])

    def get_params(self) -> dict:
        return {"S0": self.S0, "mu": self.mu, "sigma": self.sigma}

    def set_params(self, params: dict) -> None:
        if "S0" in params:
            self.S0 = float(params["S0"])
        if "mu" in params:
            self.mu = float(params["mu"])
        if "sigma" in params:
            self.sigma = float(params["sigma"])
