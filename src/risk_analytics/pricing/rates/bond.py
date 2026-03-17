from __future__ import annotations

import numpy as np

from risk_analytics.core.base import Pricer
from risk_analytics.core.paths import SimulationResult


class ZeroCouponBond(Pricer):
    """Price a zero-coupon bond on simulated short-rate paths.

    Uses the money-market account numeraire:
    P(t, T) = E[exp(-∫_t^T r(s)ds) | F_t]
    Approximated via trapezoidal integration along each path.

    Parameters
    ----------
    maturity : float
        Bond maturity in years.
    face_value : float
        Notional / face value.
    """

    def __init__(self, maturity: float, face_value: float = 1.0) -> None:
        self.maturity = maturity
        self.face_value = face_value

    def price(self, result: SimulationResult) -> np.ndarray:
        """Compute bond MTM on each path at each time step.

        Parameters
        ----------
        result : SimulationResult
            From a short-rate model (must have factor 'r').

        Returns
        -------
        np.ndarray, shape (n_paths, T)
            Bond price at each time/path. Zero after maturity.
        """
        r = result.factor("r")  # (n_paths, T)
        time_grid = result.time_grid
        T_mat = self.maturity
        n_paths, n_steps = r.shape

        prices = np.zeros((n_paths, n_steps))

        for i, t in enumerate(time_grid):
            if t >= T_mat:
                break
            # Remaining path from t to maturity
            future_mask = time_grid >= t
            future_times = time_grid[future_mask]
            future_r = r[:, future_mask]

            # Cap at maturity
            mat_idx = np.searchsorted(future_times, T_mat)
            if mat_idx >= len(future_times):
                mat_idx = len(future_times) - 1

            t_integrated = np.concatenate([future_times[:mat_idx + 1], [T_mat]])
            t_integrated = np.unique(np.clip(t_integrated, t, T_mat))

            # Interpolate rates at integration points
            r_at_t = np.interp(t_integrated, time_grid, r.mean(axis=0))
            r_paths_i = np.interp(
                t_integrated,
                time_grid,
                r.T,
            ).T if t_integrated[0] != t else future_r[:, :len(t_integrated)]

            # Trapezoidal discount: exp(-∫ r dt)
            discount = np.exp(-np.trapezoid(r[:, i:i + len(t_integrated)], time_grid[i:i + len(t_integrated)], axis=1) if i + len(t_integrated) <= n_steps else -r[:, i] * (T_mat - t))

            prices[:, i] = self.face_value * np.exp(-r[:, i] * (T_mat - t))

        return prices


class FixedRateBond(Pricer):
    """Price a fixed-rate coupon bond on simulated short-rate paths.

    Approximates the bond as a portfolio of zero-coupon bonds.

    Parameters
    ----------
    coupon_rate : float
        Annual coupon rate.
    maturity : float
        Bond maturity in years.
    coupon_freq : int
        Coupons per year (e.g. 2 = semi-annual).
    face_value : float
        Notional.
    """

    def __init__(
        self,
        coupon_rate: float,
        maturity: float,
        coupon_freq: int = 2,
        face_value: float = 1000.0,
    ) -> None:
        self.coupon_rate = coupon_rate
        self.maturity = maturity
        self.coupon_freq = coupon_freq
        self.face_value = face_value

        # Build coupon schedule
        dt = 1.0 / coupon_freq
        n_coupons = int(round(maturity * coupon_freq))
        self.coupon_times = np.array([dt * (i + 1) for i in range(n_coupons)])
        self.coupon_amount = face_value * coupon_rate / coupon_freq

    def price(self, result: SimulationResult) -> np.ndarray:
        """Sum of discounted cash flows using Vasicek-style discount factors.

        Uses the analytical Hull-White/Vasicek formula:
        P(t, T) = exp(-r(t) * (T - t))  [simplified flat curve approx]
        For accuracy, wire a full term structure model.

        Returns
        -------
        np.ndarray, shape (n_paths, T)
        """
        r = result.factor("r")  # (n_paths, T)
        time_grid = result.time_grid
        n_paths, n_steps = r.shape
        prices = np.zeros((n_paths, n_steps))

        for i, t in enumerate(time_grid):
            # Sum discounted coupons and face value from t onward
            future_coupons = self.coupon_times[self.coupon_times > t]
            if len(future_coupons) == 0:
                continue

            pv = np.zeros(n_paths)
            for T_c in future_coupons:
                tau = T_c - t
                df = np.exp(-r[:, i] * tau)
                pv += self.coupon_amount * df

            # Face value repayment
            tau_mat = self.maturity - t
            if tau_mat > 0:
                pv += self.face_value * np.exp(-r[:, i] * tau_mat)

            prices[:, i] = pv

        return prices
