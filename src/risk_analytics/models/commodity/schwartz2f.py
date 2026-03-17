from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from risk_analytics.core.base import StochasticModel
from risk_analytics.core.paths import SimulationResult


class Schwartz2F(StochasticModel):
    """Schwartz-Smith (2000) two-factor commodity model.

    Log-spot = long-term component χ + short-term deviation ξ:
    dξ(t) = -κ·ξ(t) dt + σ_ξ dW_ξ
    dχ(t) = (μ_χ - λ_χ) dt + σ_χ dW_χ
    S(t) = exp(ξ(t) + χ(t))
    dW_ξ · dW_χ = ρ dt

    Parameters
    ----------
    S0 : float
        Initial spot price.
    xi0 : float
        Initial short-term deviation.
    chi0 : float
        Initial long-term component (≈ ln(S0) - xi0).
    kappa : float
        Mean reversion speed of short-term factor.
    mu_chi : float
        Drift of long-term factor (risk-neutral).
    lambda_chi : float
        Market price of risk for long-term factor.
    sigma_xi : float
        Volatility of short-term factor.
    sigma_chi : float
        Volatility of long-term factor.
    rho : float
        Correlation between the two Brownians.
    """

    def __init__(
        self,
        S0: float = 50.0,
        xi0: float = 0.0,
        chi0: float | None = None,
        kappa: float = 1.5,
        mu_chi: float = 0.05,
        lambda_chi: float = 0.0,
        sigma_xi: float = 0.3,
        sigma_chi: float = 0.15,
        rho: float = 0.3,
    ) -> None:
        self.S0 = S0
        self.xi0 = xi0
        self.chi0 = np.log(S0) - xi0 if chi0 is None else chi0
        self.kappa = kappa
        self.mu_chi = mu_chi
        self.lambda_chi = lambda_chi
        self.sigma_xi = sigma_xi
        self.sigma_chi = sigma_chi
        self.rho = rho

    @property
    def n_factors(self) -> int:
        return 2

    @property
    def name(self) -> str:
        return "Schwartz2F"

    def simulate(
        self,
        time_grid: np.ndarray,
        n_paths: int,
        random_draws: np.ndarray,
    ) -> SimulationResult:
        """Simulate both factors: exact OU for xi, vectorised BM for chi.

        xi uses exact Gaussian transition (no Euler bias):
            xi(t+dt) = xi(t)·exp(-κ·dt) + σ_xi·√((1-exp(-2κ·dt))/(2κ))·Z

        chi is arithmetic BM with drift (Euler = exact for BM):
            chi(t+dt) = chi(t) + (μ_χ - λ_χ)·dt + σ_χ·√dt·Z
        chi is fully vectorised via cumsum.

        Parameters
        ----------
        random_draws : np.ndarray, shape (n_paths, T-1, 2)
            Factor 0 → W_xi (orthogonal component), Factor 1 → W_chi.

        Returns
        -------
        SimulationResult with factors ['S', 'xi', 'chi'], shape (n_paths, T, 3)
        """
        T = len(time_grid)
        dt = np.diff(time_grid)  # (T-1,)
        sqrt_dt = np.sqrt(dt)    # (T-1,)

        dZ1 = random_draws[:, :, 0]  # orthogonal component, (n_paths, T-1)
        dW_chi = random_draws[:, :, 1]
        dW_xi = self.rho * dW_chi + np.sqrt(1.0 - self.rho**2) * dZ1

        # --- chi: arithmetic BM — fully vectorised via cumsum ---
        chi_increments = (
            (self.mu_chi - self.lambda_chi) * dt
            + self.sigma_chi * sqrt_dt * dW_chi
        )  # (n_paths, T-1)
        chi = np.empty((n_paths, T))
        chi[:, 0] = self.chi0
        chi[:, 1:] = self.chi0 + np.cumsum(chi_increments, axis=1)

        # --- xi: exact OU transition — precompute per-step scalars ---
        e_kdt = np.exp(-self.kappa * dt)                                   # (T-1,)
        std_xi = self.sigma_xi * np.sqrt(
            -np.expm1(-2.0 * self.kappa * dt) / (2.0 * self.kappa)
        )                                                                    # (T-1,)

        xi = np.empty((n_paths, T))
        xi[:, 0] = self.xi0
        for i in range(T - 1):
            xi[:, i + 1] = xi[:, i] * e_kdt[i] + std_xi[i] * dW_xi[:, i]

        S = np.exp(xi + chi)
        paths = np.stack([S, xi, chi], axis=2)  # (n_paths, T, 3)

        return SimulationResult(
            time_grid=time_grid,
            paths=paths,
            model_name=self.name,
            factor_names=["S", "xi", "chi"],
        )

    def calibrate(self, market_data: dict) -> None:
        """Calibrate to forward curve.

        Expected market_data keys:
        - 'S0': float
        - 'forward_prices': np.ndarray
        - 'forward_tenors': np.ndarray (years)
        """
        if "S0" in market_data:
            self.S0 = float(market_data["S0"])
            self.chi0 = np.log(self.S0) - self.xi0

        if "forward_prices" in market_data and "forward_tenors" in market_data:
            fwd = np.asarray(market_data["forward_prices"])
            tenors = np.asarray(market_data["forward_tenors"])
            self._fit_to_forward_curve(fwd, tenors)

    def get_params(self) -> dict:
        return {
            "S0": self.S0,
            "xi0": self.xi0,
            "chi0": self.chi0,
            "kappa": self.kappa,
            "mu_chi": self.mu_chi,
            "lambda_chi": self.lambda_chi,
            "sigma_xi": self.sigma_xi,
            "sigma_chi": self.sigma_chi,
            "rho": self.rho,
        }

    def set_params(self, params: dict) -> None:
        for attr in ["S0", "xi0", "chi0", "kappa", "mu_chi", "lambda_chi",
                     "sigma_xi", "sigma_chi", "rho"]:
            if attr in params:
                setattr(self, attr, float(params[attr]))

    def forward_price(self, t: float) -> float:
        """Analytical forward price F(0, t) under Schwartz 2F."""
        e = np.exp(-self.kappa * t)
        mean_xi = self.xi0 * e
        mean_chi = self.chi0 + (self.mu_chi - self.lambda_chi) * t

        var_xi = self.sigma_xi**2 * (1 - e**2) / (2 * self.kappa)
        var_chi = self.sigma_chi**2 * t
        cov = self.rho * self.sigma_xi * self.sigma_chi * (1 - e) / self.kappa

        total_var = var_xi + var_chi + 2 * cov
        return float(np.exp(mean_xi + mean_chi + 0.5 * total_var))

    def _fit_to_forward_curve(self, fwd: np.ndarray, tenors: np.ndarray) -> None:
        def objective(x: np.ndarray) -> float:
            kappa, mu_chi, lambda_chi = float(x[0]), float(x[1]), float(x[2])
            old = (self.kappa, self.mu_chi, self.lambda_chi)
            self.kappa, self.mu_chi, self.lambda_chi = kappa, mu_chi, lambda_chi
            model_fwd = np.array([self.forward_price(t) for t in tenors])
            self.kappa, self.mu_chi, self.lambda_chi = old
            return float(np.sum((np.log(model_fwd) - np.log(fwd)) ** 2))

        result = minimize(
            objective,
            x0=[self.kappa, self.mu_chi, self.lambda_chi],
            bounds=[(1e-4, 20), (-1, 1), (-1, 1)],
            method="L-BFGS-B",
        )
        self.kappa, self.mu_chi, self.lambda_chi = result.x
