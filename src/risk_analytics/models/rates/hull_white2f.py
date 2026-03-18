from __future__ import annotations

import logging

import numpy as np

from risk_analytics.core.base import StochasticModel
from risk_analytics.core.paths import SimulationResult
from risk_analytics.core.yield_curve import YieldCurve

logger = logging.getLogger(__name__)


class HullWhite2F(StochasticModel):
    """Two-factor Hull-White (G2++) short rate model.

    Short rate: x(t) = r(t) + u(t) + φ(t)
    where:
        dr = -a·r dt + σ dW₁
        du = -b·u dt + η dW₂
        E[dW₁ dW₂] = ρ dt

    The model is internally two-factor but externally compatible with the
    existing HW1F rate pricer interface:
    - Factor 'r'           : x(t) = r_component(t) + u_component(t)  (the full short rate)
    - Factor 'u_component' : u(t) alone (for advanced / diagnostic use)

    The discount_factor() method uses an approximate affine formula based on
    the dominant r-factor B function; a full 2F affine formula is available
    via discount_factor_2f(). Cross-correlation terms are omitted in the
    log-A approximation (Brigo & Mercurio §4.2 simplification).

    Parameters
    ----------
    a : float
        Mean reversion speed for the r factor.
    sigma : float
        Volatility of the r factor.
    b : float
        Mean reversion speed for the u factor.
    eta : float
        Volatility of the u factor.
    rho : float
        Instantaneous correlation between the two Brownian motions.
    r0 : float
        Initial short rate (= r(0) + u(0), with u(0) = 0 by convention).
    """

    name = "HullWhite2F"
    n_factors = 2
    interpolation_space = ["linear", "linear"]

    def __init__(
        self,
        a: float = 0.10,
        sigma: float = 0.01,
        b: float = 0.05,
        eta: float = 0.005,
        rho: float = 0.0,
        r0: float = 0.03,
    ) -> None:
        self.a = a
        self.sigma = sigma
        self.b = b
        self.eta = eta
        self.rho = rho
        self.r0 = r0
        self._curve = None
        self._theta = None  # fitted drift for r component (T-1,)

    # ------------------------------------------------------------------
    # StochasticModel interface
    # ------------------------------------------------------------------

    @property
    def n_factors(self) -> int:
        return 2

    @property
    def name(self) -> str:
        return "HullWhite2F"

    @property
    def interpolation_space(self) -> list:
        return ["linear", "linear"]

    def simulate(
        self,
        time_grid: np.ndarray,
        n_paths: int,
        random_draws: np.ndarray,
    ) -> SimulationResult:
        """Exact Gaussian transition for the 2F Hull-White SDE.

        Parameters
        ----------
        time_grid : np.ndarray, shape (T,)
        n_paths : int
        random_draws : np.ndarray, shape (n_paths, T-1, 2)
            Pre-generated (possibly correlated) standard normal draws.
            Factor 0 drives r, factor 1 drives u.

        Returns
        -------
        SimulationResult with factors ['r', 'u_component'], shape (n_paths, T, 2)
            Factor 'r' = x(t) = r_component(t) + u_component(t) (full short rate)
            Factor 'u_component' = u(t) alone
        """
        T = len(time_grid)
        dt = np.diff(time_grid)

        a, sigma = self.a, self.sigma
        b, eta = self.b, self.eta

        theta = self._theta if self._theta is not None else np.zeros(T - 1)
        if len(theta) != T - 1:
            raise ValueError(f"theta must have length {T - 1}, got {len(theta)}")

        r_comp = np.empty((n_paths, T))
        u_comp = np.empty((n_paths, T))

        # Initial conditions: r_component = r0, u_component = 0
        r_comp[:, 0] = self.r0
        u_comp[:, 0] = 0.0

        Z1 = random_draws[:, :, 0]  # (n_paths, T-1)
        Z2 = random_draws[:, :, 1]  # (n_paths, T-1)

        # Precompute per-step scalar coefficients
        if a != 0.0:
            e_a = np.exp(-a * dt)
            var_r = sigma**2 * (-np.expm1(-2.0 * a * dt)) / (2.0 * a)
            std_r = np.sqrt(var_r)
            theta_contrib = (theta / a) * (1.0 - e_a)
        else:
            e_a = np.ones(T - 1)
            std_r = sigma * np.sqrt(dt)
            theta_contrib = theta * dt

        if b != 0.0:
            e_b = np.exp(-b * dt)
            var_u = eta**2 * (-np.expm1(-2.0 * b * dt)) / (2.0 * b)
            std_u = np.sqrt(var_u)
        else:
            e_b = np.ones(T - 1)
            std_u = eta * np.sqrt(dt)

        for i in range(T - 1):
            r_comp[:, i + 1] = (
                r_comp[:, i] * e_a[i]
                + theta_contrib[i]
                + std_r[i] * Z1[:, i]
            )
            u_comp[:, i + 1] = (
                u_comp[:, i] * e_b[i]
                + std_u[i] * Z2[:, i]
            )

        # Full short rate = r component + u component
        x = r_comp + u_comp

        # Shape: (n_paths, T, 2)
        paths = np.stack([x, u_comp], axis=2)

        return SimulationResult(
            time_grid=time_grid,
            paths=paths,
            model_name=self.name,
            factor_names=["r", "u_component"],
            interpolation_space=self.interpolation_space,
            model=self,
        )

    def calibrate(self, market_data: dict) -> None:
        """Calibrate the r-factor drift theta to match the initial yield curve.

        Uses the same theta-fitting approach as HullWhite1F — the u factor is
        treated as an additional volatility overlay and is not separately
        calibrated here.

        Expected market_data keys:
        - 'time_grid': np.ndarray — simulation time grid (required)
        - 'yield_curve': YieldCurve — preferred
        - 'tenors' + 'zero_rates': np.ndarray pair — legacy interface
        """
        time_grid = np.asarray(market_data["time_grid"])

        if "yield_curve" in market_data:
            curve = market_data["yield_curve"]
        else:
            tenors = np.asarray(market_data["tenors"])
            zero_rates = np.asarray(market_data["zero_rates"])
            curve = YieldCurve(tenors, zero_rates)

        self._curve = curve
        self.r0 = float(curve.zero_rate(0.0))
        self._theta = self._fit_theta(time_grid, curve)
        logger.info(
            "HullWhite2F calibrated: r0=%.4f  a=%.4f  sigma=%.4f  b=%.4f  eta=%.4f",
            self.r0, self.a, self.sigma, self.b, self.eta,
        )

    def get_params(self) -> dict:
        params = {
            "a": self.a,
            "sigma": self.sigma,
            "b": self.b,
            "eta": self.eta,
            "rho": self.rho,
            "r0": self.r0,
        }
        if self._theta is not None:
            params["theta"] = self._theta
        return params

    def set_params(self, params: dict) -> None:
        for k in ("a", "sigma", "b", "eta", "rho", "r0"):
            if k in params:
                setattr(self, k, float(params[k]))
        if "theta" in params:
            self._theta = np.asarray(params["theta"])

    # ------------------------------------------------------------------
    # Analytical discount factor (compatible with existing rate pricers)
    # ------------------------------------------------------------------

    def discount_factor(self, t: float, T_mat: float, x_t: np.ndarray) -> np.ndarray:
        """Approximate affine discount factor P(t, T) using the dominant r factor.

        Uses the HW1F formula with parameter a as an approximation:
            P(t,T) ≈ A(t,T) · exp(-B₁(t,T) · x(t))

        This is exact for the pure r factor and approximate for the 2F model.
        For the full 2F formula, use discount_factor_2f().
        """
        tau = T_mat - t
        a, sigma = self.a, self.sigma

        if a != 0:
            B1 = (1.0 - np.exp(-a * tau)) / a
        else:
            B1 = tau

        if self._curve is not None:
            t_safe = max(float(t), 1e-9)
            p0T = self._curve.discount_factor(float(T_mat))
            p0t = self._curve.discount_factor(float(t)) if t > 1e-9 else 1.0
            f0t = float(self._curve.instantaneous_forward(t_safe))
            if a != 0:
                conv = (sigma**2 / (4.0 * a)) * B1**2 * (1.0 - np.exp(-2.0 * a * float(t)))
            else:
                conv = 0.5 * sigma**2 * float(t) * B1**2
            ln_A = np.log(p0T / p0t) + B1 * f0t - conv
        else:
            b_val = self.r0
            if a != 0:
                ln_A = (b_val - sigma**2 / (2.0 * a**2)) * (B1 - tau) - sigma**2 * B1**2 / (4.0 * a)
            else:
                ln_A = b_val * (B1 - tau) - 0.5 * sigma**2 * tau * B1**2

        return np.exp(ln_A - B1 * x_t)

    def discount_factor_2f(
        self, t: float, T_mat: float, r_t: np.ndarray, u_t: np.ndarray
    ) -> np.ndarray:
        """Full 2F affine discount factor P(t, T | r, u).

        ln A(t,T) = ln(P(0,T)/P(0,t)) + B₁·f(0,t)
                    - (σ²/4a)·(1-exp(-2at))·B₁²
                    - (η²/4b)·(1-exp(-2bt))·B₂²

        Note: the cross-correlation term ρση/(a+b) is omitted (set rho=0 in
        the ln A approximation). It only matters when rho ≠ 0 and both factors
        have significant variance.

        Parameters
        ----------
        t : float
        T_mat : float
        r_t : np.ndarray, shape (n_paths,) — the r_component at time t
        u_t : np.ndarray, shape (n_paths,) — the u_component at time t
        """
        tau = T_mat - t
        a, sigma = self.a, self.sigma
        b, eta = self.b, self.eta

        B1 = (1.0 - np.exp(-a * tau)) / a if a != 0 else tau
        B2 = (1.0 - np.exp(-b * tau)) / b if b != 0 else tau

        if self._curve is not None:
            t_safe = max(float(t), 1e-9)
            p0T = self._curve.discount_factor(float(T_mat))
            p0t = self._curve.discount_factor(float(t)) if t > 1e-9 else 1.0
            f0t = float(self._curve.instantaneous_forward(t_safe))
            conv_r = (sigma**2 / (4.0 * a)) * B1**2 * (1.0 - np.exp(-2.0 * a * float(t))) if a != 0 else 0.0
            conv_u = (eta**2 / (4.0 * b)) * B2**2 * (1.0 - np.exp(-2.0 * b * float(t))) if b != 0 else 0.0
            ln_A = np.log(p0T / p0t) + B1 * f0t - conv_r - conv_u
        else:
            b_val = self.r0
            ln_A = (
                (b_val - sigma**2 / (2.0 * a**2)) * (B1 - tau) - sigma**2 * B1**2 / (4.0 * a)
                if a != 0 else b_val * (B1 - tau) - 0.5 * sigma**2 * tau * B1**2
            )

        return np.exp(ln_A - B1 * r_t - B2 * u_t)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _fit_theta(self, time_grid: np.ndarray, curve: YieldCurve) -> np.ndarray:
        """Fit theta to match the initial term structure (same as HW1F)."""
        t_mid = time_grid[:-1]
        fwd = curve.instantaneous_forward(t_mid)
        dfwd = np.gradient(fwd, t_mid) if len(fwd) > 1 else np.zeros_like(fwd)

        a, sigma = self.a, self.sigma
        if a != 0:
            theta = dfwd + a * fwd + (sigma**2 / (2.0 * a)) * (1.0 - np.exp(-2.0 * a * t_mid))
        else:
            theta = dfwd + sigma**2 * t_mid
        return theta
