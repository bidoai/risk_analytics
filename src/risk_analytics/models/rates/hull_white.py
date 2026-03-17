from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import minimize

from risk_analytics.core.base import StochasticModel
from risk_analytics.core.paths import SimulationResult
from risk_analytics.core.yield_curve import YieldCurve

logger = logging.getLogger(__name__)


class HullWhite1F(StochasticModel):
    """Hull-White one-factor short rate model.

    dr(t) = (θ(t) - a·r(t)) dt + σ dW(t)

    Parameters
    ----------
    a : float
        Mean reversion speed.
    sigma : float
        Short rate volatility.
    r0 : float
        Initial short rate.
    theta : np.ndarray | None
        Deterministic drift term θ(t) evaluated on the simulation time grid.
        If None, assumed zero (pure Vasicek).
    """

    def __init__(
        self,
        a: float = 0.1,
        sigma: float = 0.01,
        r0: float = 0.03,
        theta: np.ndarray | None = None,
    ) -> None:
        self.a = a
        self.sigma = sigma
        self.r0 = r0
        self.theta = theta  # shape (T-1,) — one value per time step

    # ------------------------------------------------------------------
    # StochasticModel interface
    # ------------------------------------------------------------------

    @property
    def n_factors(self) -> int:
        return 1

    @property
    def name(self) -> str:
        return "HullWhite1F"

    def simulate(
        self,
        time_grid: np.ndarray,
        n_paths: int,
        random_draws: np.ndarray,
    ) -> SimulationResult:
        """Exact Gaussian transition for the Hull-White SDE.

        Given r(t) and θ(t) constant over [t, t+dt], the exact solution is:

            r(t+dt) = r(t)·e^{-a·dt}
                      + (θ(t)/a)·(1 - e^{-a·dt})
                      + σ·√((1 - e^{-2a·dt}) / (2a)) · Z

        where Z ~ N(0,1). This is bias-free regardless of step size.

        Parameters
        ----------
        time_grid : np.ndarray, shape (T,)
        n_paths : int
        random_draws : np.ndarray, shape (n_paths, T-1, 1)

        Returns
        -------
        SimulationResult with factor 'r', shape (n_paths, T, 1)
        """
        T = len(time_grid)
        dt = np.diff(time_grid)  # (T-1,)

        theta = self.theta if self.theta is not None else np.zeros(T - 1)
        if len(theta) != T - 1:
            raise ValueError(f"theta must have length {T - 1}, got {len(theta)}")

        a, sigma = self.a, self.sigma
        paths = np.empty((n_paths, T))
        paths[:, 0] = self.r0

        Z = random_draws[:, :, 0]  # (n_paths, T-1)

        # Precompute per-step scalar coefficients (T-1 values, not n_paths×T ops)
        if a != 0.0:
            e_adt = np.exp(-a * dt)                            # (T-1,)
            # Use expm1 for numerical stability when a*dt is small:
            # 1 - e^{-2a·dt} = -expm1(-2a·dt)
            var_step = sigma**2 * (-np.expm1(-2.0 * a * dt)) / (2.0 * a)  # (T-1,)
            std_step = np.sqrt(var_step)                        # (T-1,)
            theta_contrib = (theta / a) * (1.0 - e_adt)        # (T-1,)
        else:
            # a → 0 limit: Vasicek degenerates to arithmetic BM with drift
            e_adt = np.ones(T - 1)
            std_step = sigma * np.sqrt(dt)
            theta_contrib = theta * dt

        for i in range(T - 1):
            paths[:, i + 1] = (
                paths[:, i] * e_adt[i]
                + theta_contrib[i]
                + std_step[i] * Z[:, i]
            )

        return SimulationResult(
            time_grid=time_grid,
            paths=paths[:, :, np.newaxis],
            model_name=self.name,
            factor_names=["r"],
        )

    def calibrate(self, market_data: dict) -> None:
        """Calibrate Hull-White parameters to market data.

        Expected market_data keys:
        - 'time_grid': np.ndarray — simulation time grid (required)
        - 'yield_curve': YieldCurve — preferred; provides interpolated rates
          and instantaneous forwards directly.
        - 'tenors' + 'zero_rates': np.ndarray pair — legacy interface;
          a LOG_LINEAR YieldCurve is constructed automatically.
        - 'cap_vols' (optional): np.ndarray of cap implied vols for (a, sigma) calibration
        """
        time_grid = np.asarray(market_data["time_grid"])

        if "yield_curve" in market_data:
            curve = market_data["yield_curve"]
        else:
            tenors = np.asarray(market_data["tenors"])
            zero_rates = np.asarray(market_data["zero_rates"])
            curve = YieldCurve(tenors, zero_rates)

        self.r0 = float(curve.zero_rate(0.0))
        logger.info(
            "HullWhite1F: fitting theta to yield curve (%s, %d tenors), r0=%.4f",
            curve.interpolation.value, len(curve._t), self.r0,
        )

        # Fit theta(t) to match the initial term structure exactly
        self.theta = self._fit_theta(time_grid, curve)
        logger.debug("HullWhite1F: theta fitted, shape=%s", self.theta.shape)

        # Optionally fit a and sigma to cap vols
        if "cap_vols" in market_data and "cap_tenors" in market_data:
            logger.info("HullWhite1F: calibrating a and sigma to cap vols")
            self._calibrate_vol_params(market_data)

        logger.info(
            "HullWhite1F calibrated: r0=%.4f  a=%.4f  sigma=%.4f",
            self.r0, self.a, self.sigma,
        )

    def get_params(self) -> dict:
        params: dict = {"a": self.a, "sigma": self.sigma, "r0": self.r0}
        if self.theta is not None:
            params["theta"] = self.theta  # np.ndarray — serialised as list by save()
        return params

    def set_params(self, params: dict) -> None:
        if "a" in params:
            self.a = float(params["a"])
        if "sigma" in params:
            self.sigma = float(params["sigma"])
        if "r0" in params:
            self.r0 = float(params["r0"])
        if "theta" in params:
            self.theta = np.asarray(params["theta"])

    # ------------------------------------------------------------------
    # Analytical helpers
    # ------------------------------------------------------------------

    def discount_factor(self, t: float, T_mat: float, r_t: np.ndarray) -> np.ndarray:
        """Analytical zero-coupon bond price P(t, T) under Hull-White.

        B(t,T) = (1 - exp(-a*(T-t))) / a
        ln P(t,T) = -B*r(t) + ln(P(0,T)/P(0,t)) - 0.5*sigma^2*B^2*...
        Simplified here as the Vasicek formula (θ=0 baseline):
        P(t,T) = A(t,T) * exp(-B(t,T) * r(t))
        """
        tau = T_mat - t
        a, sigma = self.a, self.sigma
        B = (1 - np.exp(-a * tau)) / a if a != 0 else tau
        A = np.exp((B - tau) * (a**2 * 0.0 - sigma**2 / (2 * a**2)) - sigma**2 * B**2 / (4 * a))
        return A * np.exp(-B * r_t)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _fit_theta(self, time_grid: np.ndarray, curve: YieldCurve) -> np.ndarray:
        """Derive theta(t) that fits the initial term structure.

        For Hull-White: θ(t) = f'(0,t) + a·f(0,t) + σ²/(2a)·(1 - exp(-2a·t))
        where f(0,t) is the instantaneous forward rate from the yield curve.

        Using YieldCurve.instantaneous_forward() gives analytically correct
        forwards for each interpolation method (piecewise-constant for
        LOG_LINEAR; smooth for CUBIC_SPLINE) rather than finite-differencing
        interpolated zero rates.
        """
        t_mid = time_grid[:-1]   # (T-1,) — one value per simulation step

        # Instantaneous forwards from the curve (analytic, not finite-diff)
        fwd = curve.instantaneous_forward(t_mid)

        # f'(0,t): numerical gradient of f along the time grid mid-points
        dfwd = np.gradient(fwd, t_mid) if len(fwd) > 1 else np.zeros_like(fwd)

        a, sigma = self.a, self.sigma
        if a != 0:
            theta = dfwd + a * fwd + (sigma**2 / (2.0 * a)) * (1.0 - np.exp(-2.0 * a * t_mid))
        else:
            theta = dfwd + sigma**2 * t_mid
        return theta

    def _calibrate_vol_params(self, market_data: dict) -> None:
        """Fit a and sigma to cap implied volatilities via least squares."""
        cap_tenors = np.asarray(market_data["cap_tenors"])
        cap_vols = np.asarray(market_data["cap_vols"])

        def objective(x: np.ndarray) -> float:
            a, sigma = float(x[0]), float(x[1])
            model_vols = self._hull_white_cap_vol(cap_tenors, a, sigma)
            return float(np.sum((model_vols - cap_vols) ** 2))

        result = minimize(
            objective,
            x0=[self.a, self.sigma],
            bounds=[(1e-4, 5.0), (1e-4, 1.0)],
            method="L-BFGS-B",
        )
        self.a, self.sigma = float(result.x[0]), float(result.x[1])

    def _hull_white_cap_vol_integral(self, T: float, a: float, sigma: float) -> float:
        """Variance of log(P(0,T)) under Hull-White (cap vol formula)."""
        if a == 0:
            return sigma**2 * T**3 / 3
        return (sigma**2 / (2 * a**3)) * (
            2 * a * T - 3 + 4 * np.exp(-a * T) - np.exp(-2 * a * T)
        )

    def _hull_white_cap_vol(self, tenors: np.ndarray, a: float, sigma: float) -> np.ndarray:
        return np.array([
            np.sqrt(self._hull_white_cap_vol_integral(T, a, sigma) / T)
            for T in tenors
        ])
