from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from risk_analytics.core.base import StochasticModel
from risk_analytics.core.paths import SimulationResult


class HestonModel(StochasticModel):
    """Heston stochastic volatility model.

    dS(t) = μ·S(t) dt + √v(t)·S(t) dW_S(t)
    dv(t) = κ(θ - v(t)) dt + ξ·√v(t) dW_v(t)
    dW_S · dW_v = ρ dt

    Simulated via Euler-Maruyama with full truncation of the variance process.

    Parameters
    ----------
    S0 : float
        Initial spot price.
    v0 : float
        Initial variance.
    mu : float
        Drift of log-price.
    kappa : float
        Mean reversion speed of variance.
    theta : float
        Long-run variance.
    xi : float
        Volatility of variance (vol-of-vol).
    rho : float
        Correlation between price and variance Brownians.
    """

    def __init__(
        self,
        S0: float = 100.0,
        v0: float = 0.04,
        mu: float = 0.05,
        kappa: float = 2.0,
        theta: float = 0.04,
        xi: float = 0.3,
        rho: float = -0.7,
    ) -> None:
        self.S0 = S0
        self.v0 = v0
        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho

    @property
    def n_factors(self) -> int:
        return 2  # W_S (orthogonalised) and W_v

    @property
    def name(self) -> str:
        return "Heston"

    def simulate(
        self,
        time_grid: np.ndarray,
        n_paths: int,
        random_draws: np.ndarray,
    ) -> SimulationResult:
        """Euler-Maruyama with full truncation for the variance process.

        Parameters
        ----------
        random_draws : np.ndarray, shape (n_paths, T-1, 2)
            Two independent standard normal factors per step.
            Factor 0 → W_S (orthogonal component), Factor 1 → W_v.

        Returns
        -------
        SimulationResult with factors ['S', 'v'], shape (n_paths, T, 2)
        """
        T = len(time_grid)
        dt = np.diff(time_grid)

        S = np.empty((n_paths, T))
        v = np.empty((n_paths, T))
        S[:, 0] = self.S0
        v[:, 0] = self.v0

        # Correlated Brownians: dW_S = rho*dW_v + sqrt(1-rho^2)*dZ
        dZ1 = random_draws[:, :, 0]  # orthogonal to W_v
        dW_v = random_draws[:, :, 1]
        dW_S = self.rho * dW_v + np.sqrt(1 - self.rho**2) * dZ1

        # Precompute per-step scalars outside the loop
        sqrt_dt = np.sqrt(dt)  # (T-1,)

        for i in range(T - 1):
            v_plus = np.maximum(v[:, i], 0.0)  # full truncation — path-dependent, loop unavoidable
            sqrt_v = np.sqrt(v_plus)

            v[:, i + 1] = (
                v[:, i]
                + self.kappa * (self.theta - v_plus) * dt[i]
                + self.xi * sqrt_v * sqrt_dt[i] * dW_v[:, i]
            )

            log_S = np.log(S[:, i]) + (self.mu - 0.5 * v_plus) * dt[i] + sqrt_v * sqrt_dt[i] * dW_S[:, i]
            S[:, i + 1] = np.exp(log_S)

        paths = np.stack([S, v], axis=2)  # (n_paths, T, 2)

        return SimulationResult(
            time_grid=time_grid,
            paths=paths,
            model_name=self.name,
            factor_names=["S", "v"],
        )

    def calibrate(self, market_data: dict) -> None:
        """Calibrate Heston parameters to implied vol surface.

        Expected market_data keys:
        - 'S0': float
        - 'r': float, risk-free rate
        - 'strikes': np.ndarray, shape (N,)
        - 'maturities': np.ndarray, shape (M,)
        - 'implied_vols': np.ndarray, shape (M, N)
        """
        if "S0" in market_data:
            self.S0 = float(market_data["S0"])
        if "r" in market_data:
            self.mu = float(market_data["r"])

        if "implied_vols" not in market_data:
            return

        strikes = np.asarray(market_data["strikes"])
        maturities = np.asarray(market_data["maturities"])
        target_vols = np.asarray(market_data["implied_vols"])
        r = self.mu

        def objective(x: np.ndarray) -> float:
            kappa, theta, xi, rho, v0 = x
            total_error = 0.0
            for j, T_mat in enumerate(maturities):
                for k, K in enumerate(strikes):
                    model_vol = self._heston_implied_vol(self.S0, K, T_mat, r, v0, kappa, theta, xi, rho)
                    total_error += (model_vol - target_vols[j, k]) ** 2
            return total_error

        x0 = [self.kappa, self.theta, self.xi, self.rho, self.v0]
        bounds = [(0.01, 20), (1e-4, 2), (1e-4, 5), (-0.99, 0.99), (1e-4, 2)]
        result = minimize(objective, x0=x0, bounds=bounds, method="L-BFGS-B")
        self.kappa, self.theta, self.xi, self.rho, self.v0 = result.x

    def get_params(self) -> dict:
        return {
            "S0": self.S0,
            "v0": self.v0,
            "mu": self.mu,
            "kappa": self.kappa,
            "theta": self.theta,
            "xi": self.xi,
            "rho": self.rho,
        }

    def set_params(self, params: dict) -> None:
        for attr in ["S0", "v0", "mu", "kappa", "theta", "xi", "rho"]:
            if attr in params:
                setattr(self, attr, float(params[attr]))

    # ------------------------------------------------------------------
    # Heston characteristic function (for calibration)
    # ------------------------------------------------------------------

    def _heston_char_fn(
        self, phi: complex, S0: float, T: float, r: float,
        v0: float, kappa: float, theta: float, xi: float, rho: float
    ) -> complex:
        i = complex(0, 1)
        d = np.sqrt((rho * xi * i * phi - kappa) ** 2 + xi**2 * (i * phi + phi**2))
        g = (kappa - rho * xi * i * phi - d) / (kappa - rho * xi * i * phi + d)
        exp_dT = np.exp(-d * T)

        C = (kappa * theta / xi**2) * (
            (kappa - rho * xi * i * phi - d) * T
            - 2 * np.log((1 - g * exp_dT) / (1 - g))
        )
        D = ((kappa - rho * xi * i * phi - d) / xi**2) * ((1 - exp_dT) / (1 - g * exp_dT))

        return np.exp(C + D * v0 + i * phi * (np.log(S0) + r * T))

    def _heston_call_price(
        self, S0: float, K: float, T: float, r: float,
        v0: float, kappa: float, theta: float, xi: float, rho: float
    ) -> float:
        from scipy.integrate import quad

        def integrand(phi: float, j: int) -> float:
            b = kappa - rho * xi if j == 1 else kappa
            u = 0.5 if j == 1 else -0.5
            i = complex(0, 1)
            cf = self._heston_char_fn(phi - (j * i), S0, T, r, v0, kappa, theta, xi, rho)
            num = np.exp(-i * phi * np.log(K)) * cf
            return (num / (i * phi)).real

        P1, _ = quad(lambda phi: integrand(phi, 1), 1e-10, 200, limit=200)
        P2, _ = quad(lambda phi: integrand(phi, 2), 1e-10, 200, limit=200)

        price = S0 * (0.5 + P1 / np.pi) - K * np.exp(-r * T) * (0.5 + P2 / np.pi)
        return max(float(price), 0.0)

    def _heston_implied_vol(
        self, S0: float, K: float, T: float, r: float,
        v0: float, kappa: float, theta: float, xi: float, rho: float
    ) -> float:
        from scipy.optimize import brentq
        price = self._heston_call_price(S0, K, T, r, v0, kappa, theta, xi, rho)
        intrinsic = max(S0 - K * np.exp(-r * T), 0.0)
        if price <= intrinsic + 1e-8:
            return 0.0
        # Black-Scholes inversion
        from scipy.stats import norm

        def bs_price(sigma: float) -> float:
            d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2) - price

        try:
            return brentq(bs_price, 1e-6, 10.0)
        except ValueError:
            return float(np.sqrt(abs(self.theta)))
