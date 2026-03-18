from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from risk_analytics.core.base import Pricer
from risk_analytics.core.paths import SimulationResult

if TYPE_CHECKING:
    from risk_analytics.core.schedule import Schedule


def _discount_factors(r_t: np.ndarray, t: float, T_array: np.ndarray, model) -> np.ndarray:
    """Compute P(t, T_i | r_t) for all paths and payment times.

    Uses the Hull-White affine formula when the model has parameters a, sigma,
    and optionally an initial yield curve. Falls back to flat-curve exp(-r·τ)
    when no model information is available.

    Parameters
    ----------
    r_t : np.ndarray, shape (n_paths,)
        Short rate at time t on each path.
    t : float
        Current time.
    T_array : np.ndarray, shape (k,)
        Payment / maturity times, all > t.
    model : StochasticModel or None
        Hull-White model instance (or None for flat-curve fallback).

    Returns
    -------
    np.ndarray, shape (n_paths, k)
    """
    tau_vec = T_array - t                   # (k,)

    if model is not None and hasattr(model, "a") and model.a != 0:
        a = model.a
        sigma = model.sigma
        B_vec = (1.0 - np.exp(-a * tau_vec)) / a   # (k,)

        curve = getattr(model, "_curve", None)
        if curve is not None:
            p0T = np.array([curve.discount_factor(float(T)) for T in T_array])
            p0t = curve.discount_factor(float(t)) if t > 1e-9 else 1.0
            f0t = float(curve.instantaneous_forward(max(float(t), 1e-9)))
            conv = (sigma ** 2 / (4.0 * a)) * B_vec ** 2 * (1.0 - np.exp(-2.0 * a * float(t)))
            ln_A = np.log(p0T / p0t) + B_vec * f0t - conv
        else:
            b = model.r0
            ln_A = ((b - sigma ** 2 / (2.0 * a ** 2)) * (B_vec - tau_vec)
                    - sigma ** 2 * B_vec ** 2 / (4.0 * a))

        # (n_paths, k)  — A and B are scalars per payment time
        return np.exp(ln_A[None, :] - B_vec[None, :] * r_t[:, None])

    # Flat-curve fallback (a=0 or no model)
    return np.exp(-r_t[:, None] * tau_vec[None, :])


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

    def cashflow_times(self) -> list:
        """Return the maturity as the single cashflow time."""
        return [self.maturity]

    def price(self, result: SimulationResult) -> np.ndarray:
        """Compute bond MTM on each path at each time step.

        Uses the Hull-White affine discount factor P(t,T) = A(t,T)·exp(-B(t,T)·r(t))
        when the result carries a Hull-White model reference, otherwise falls back to
        the flat-curve approximation exp(-r(t)·(T-t)).

        Returns
        -------
        np.ndarray, shape (n_paths, T)
            Bond price at each time/path. Zero at or after maturity.
        """
        r = result.factor("r")          # (n_paths, T)
        time_grid = result.time_grid
        T_mat = self.maturity
        n_paths, n_steps = r.shape
        hw_model = getattr(result, "model", None)

        prices = np.zeros((n_paths, n_steps))
        T_arr = np.array([T_mat])

        for i, t in enumerate(time_grid):
            if t >= T_mat:
                break
            r_t = r[:, i]
            prices[:, i] = self.face_value * _discount_factors(r_t, t, T_arr, hw_model)[:, 0]

        return prices

    def price_at(self, result: SimulationResult, t_idx: int) -> np.ndarray:
        """Efficient single-step MTM using only the r_t slice at t_idx."""
        t = float(result.time_grid[t_idx])
        if t >= self.maturity:
            return np.zeros(result.n_paths)
        r_t = result.factor_at("r", t_idx)
        hw_model = getattr(result, "model", None)
        return self.face_value * _discount_factors(r_t, t, np.array([self.maturity]), hw_model)[:, 0]


class FixedRateBond(Pricer):
    """Price a fixed-rate coupon bond on simulated short-rate paths.

    Approximates the bond as a portfolio of zero-coupon bonds.

    Parameters
    ----------
    coupon_rate : float
        Annual coupon rate.
    maturity : float | None
        Bond maturity in years. Required when ``schedule`` is None.
    coupon_freq : int
        Coupons per year (e.g. 2 = semi-annual). Used only when
        ``schedule`` is None.
    face_value : float
        Notional.
    schedule : Schedule | None
        Pre-built payment schedule. When provided, coupon amounts are
        ``face_value * coupon_rate * δᵢ`` per period using the schedule's
        actual day-count fractions rather than the uniform ``1/coupon_freq``.
    """

    def __init__(
        self,
        coupon_rate: float,
        maturity: float | None = None,
        coupon_freq: int = 2,
        face_value: float = 1000.0,
        schedule: "Schedule | None" = None,
    ) -> None:
        self.coupon_rate = coupon_rate
        self.face_value = face_value
        self.schedule = schedule

        if schedule is not None:
            self.coupon_times = schedule.payment_times
            # Coupon amount per period uses actual day-count fraction
            self.coupon_amounts = face_value * coupon_rate * schedule.day_count_fractions
            self.maturity = float(schedule.payment_times[-1])
        else:
            if maturity is None:
                raise ValueError("Either maturity or schedule must be provided.")
            self.maturity = maturity
            self.coupon_freq = coupon_freq
            dt = 1.0 / coupon_freq
            n_coupons = int(round(maturity * coupon_freq))
            self.coupon_times = np.array([dt * (i + 1) for i in range(n_coupons)])
            self.coupon_amounts = np.full(n_coupons, face_value * coupon_rate * dt)

    def cashflow_times(self) -> list:
        """Return coupon times plus maturity (deduplicated)."""
        times = list(self.coupon_times)
        if self.maturity not in times:
            times.append(self.maturity)
        return sorted(times)

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
        hw_model = getattr(result, "model", None)

        for i, t in enumerate(time_grid):
            future_mask = self.coupon_times > t
            if not future_mask.any():
                continue

            future_T = self.coupon_times[future_mask]        # (k,)
            future_C = self.coupon_amounts[future_mask]      # (k,)

            # Discounted coupons — vectorised over payments
            df = _discount_factors(r[:, i], t, future_T, hw_model)   # (n_paths, k)
            pv = (future_C[None, :] * df).sum(axis=1)                # (n_paths,)

            # Face value repayment
            tau_mat = self.maturity - t
            if tau_mat > 0:
                pv = pv + self.face_value * _discount_factors(
                    r[:, i], t, np.array([self.maturity]), hw_model
                )[:, 0]

            prices[:, i] = pv

        return prices

    def price_at(self, result: SimulationResult, t_idx: int) -> np.ndarray:
        """Efficient single-step MTM using only the r_t slice at t_idx."""
        t = float(result.time_grid[t_idx])
        r_t = result.factor_at("r", t_idx)
        hw_model = getattr(result, "model", None)
        n_paths = len(r_t)

        future_mask = self.coupon_times > t
        if not future_mask.any() and self.maturity <= t:
            return np.zeros(n_paths)

        pv = np.zeros(n_paths)
        if future_mask.any():
            future_T = self.coupon_times[future_mask]
            future_C = self.coupon_amounts[future_mask]
            df = _discount_factors(r_t, t, future_T, hw_model)   # (n_paths, k)
            pv += (future_C[None, :] * df).sum(axis=1)

        tau_mat = self.maturity - t
        if tau_mat > 0:
            pv += self.face_value * _discount_factors(
                r_t, t, np.array([self.maturity]), hw_model
            )[:, 0]

        return pv
