from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import minimize

from risk_analytics.core.base import StochasticModel
from risk_analytics.core.paths import SimulationResult

logger = logging.getLogger(__name__)


class Schwartz1F(StochasticModel):
    """Schwartz (1997) one-factor commodity model.

    Mean-reverting log-price:
    dX(t) = κ(μ - X(t)) dt + σ dW(t)
    S(t) = exp(X(t))

    Parameters
    ----------
    S0 : float
        Initial spot price.
    kappa : float
        Mean reversion speed.
    mu : float
        Long-run log-price level.
    sigma : float
        Log-price volatility.
    """

    def __init__(
        self,
        S0: float = 50.0,
        kappa: float = 1.0,
        mu: float = 3.9,   # ≈ ln(50)
        sigma: float = 0.3,
    ) -> None:
        self.S0 = S0
        self.kappa = kappa
        self.mu = mu
        self.sigma = sigma

    @property
    def n_factors(self) -> int:
        return 1

    @property
    def name(self) -> str:
        return "Schwartz1F"

    @property
    def interpolation_space(self) -> list:
        return ["linear"]

    def simulate(
        self,
        time_grid: np.ndarray,
        n_paths: int,
        random_draws: np.ndarray,
    ) -> SimulationResult:
        """Exact OU discretisation of the log-price.

        X(t+dt) = X(t)·exp(-κ·dt) + μ·(1 - exp(-κ·dt))
                  + σ·√((1-exp(-2κ·dt))/(2κ))·Z

        Parameters
        ----------
        random_draws : np.ndarray, shape (n_paths, T-1, 1)

        Returns
        -------
        SimulationResult with factor 'S', shape (n_paths, T, 1)
        """
        T = len(time_grid)
        dt = np.diff(time_grid)  # (T-1,)
        kappa, mu, sigma = self.kappa, self.mu, self.sigma

        Z = random_draws[:, :, 0]  # (n_paths, T-1)

        # Precompute all per-step coefficients as (T-1,) arrays
        if kappa != 0.0:
            e_kdt = np.exp(-kappa * dt)
            mu_contrib = mu * (1.0 - e_kdt)
            std_step = sigma * np.sqrt(-np.expm1(-2.0 * kappa * dt) / (2.0 * kappa))
        else:
            e_kdt = np.ones(T - 1)
            mu_contrib = np.zeros(T - 1)
            std_step = sigma * np.sqrt(dt)

        X = np.empty((n_paths, T))
        X[:, 0] = np.log(self.S0)

        for i in range(T - 1):
            X[:, i + 1] = X[:, i] * e_kdt[i] + mu_contrib[i] + std_step[i] * Z[:, i]

        paths = np.exp(X)[:, :, np.newaxis]

        return SimulationResult(
            time_grid=time_grid,
            paths=paths,
            model_name=self.name,
            factor_names=["S"],
            interpolation_space=self.interpolation_space,
        )

    def calibrate(self, market_data: dict) -> None:
        """Calibrate to forward curve and historical volatility.

        Expected market_data keys:
        - 'S0': float
        - 'forward_prices': np.ndarray, forward prices at 'forward_tenors'
        - 'forward_tenors': np.ndarray, in years
        - 'hist_vol' (optional): float, historical log-return volatility
        """
        if "S0" in market_data:
            self.S0 = float(market_data["S0"])

        if "hist_vol" in market_data:
            self.sigma = float(market_data["hist_vol"])

        if "forward_prices" in market_data and "forward_tenors" in market_data:
            fwd = np.asarray(market_data["forward_prices"])
            tenors = np.asarray(market_data["forward_tenors"])
            self._fit_to_forward_curve(fwd, tenors)

        logger.info(
            "Schwartz1F calibrated: S0=%.4g  kappa=%.4f  mu=%.4f  sigma=%.4f",
            self.S0, self.kappa, self.mu, self.sigma,
        )

    def get_params(self) -> dict:
        return {"S0": self.S0, "kappa": self.kappa, "mu": self.mu, "sigma": self.sigma}

    def set_params(self, params: dict) -> None:
        for attr in ["S0", "kappa", "mu", "sigma"]:
            if attr in params:
                setattr(self, attr, float(params[attr]))

    def forward_price(self, t: float) -> float:
        """Analytical forward price F(0, t) under Schwartz 1F."""
        X0 = np.log(self.S0)
        e = np.exp(-self.kappa * t)
        mean_X = X0 * e + self.mu * (1 - e)
        var_X = self.sigma**2 * (1 - np.exp(-2 * self.kappa * t)) / (2 * self.kappa)
        return float(np.exp(mean_X + 0.5 * var_X))

    def _fit_to_forward_curve(self, fwd: np.ndarray, tenors: np.ndarray) -> None:
        def objective(x: np.ndarray) -> float:
            kappa, mu = float(x[0]), float(x[1])
            self.kappa, self.mu = kappa, mu
            model_fwd = np.array([self.forward_price(t) for t in tenors])
            return float(np.sum((np.log(model_fwd) - np.log(fwd)) ** 2))

        result = minimize(
            objective,
            x0=[self.kappa, self.mu],
            bounds=[(1e-4, 20), (-10, 10)],
            method="L-BFGS-B",
        )
        self.kappa, self.mu = float(result.x[0]), float(result.x[1])
