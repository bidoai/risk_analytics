from __future__ import annotations

import numpy as np

from risk_analytics.core.base import Pricer
from risk_analytics.core.paths import SimulationResult
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


class InterestRateSwap(Pricer):
    """Plain vanilla interest rate swap (fixed vs floating).

    The payer swap (long fixed, receive floating) MTM at time t is:
    V(t) = PV(floating leg) - PV(fixed leg)
         = N · [P(t, t_0) - P(t, T_N) - K · Σ δ_i · P(t, T_i)]

    Uses simplified flat-curve discount factors from the short rate r(t).

    Parameters
    ----------
    fixed_rate : float
        Fixed leg coupon rate.
    maturity : float | None
        Swap maturity in years. Required when ``schedule`` is None.
    notional : float
        Notional principal.
    payment_freq : int
        Payments per year (e.g. 4 = quarterly). Used only when ``schedule``
        is None.
    payer : bool
        True = payer (pay fixed, receive floating); False = receiver.
    schedule : Schedule | None
        Pre-built payment schedule with calendar- and day-count-adjusted
        payment times and accrual fractions. When provided, ``maturity``
        and ``payment_freq`` are ignored and the schedule's ``payment_times``
        and ``day_count_fractions`` are used instead.
    """

    def __init__(
        self,
        fixed_rate: float,
        maturity: float | None = None,
        notional: float = 1_000_000.0,
        payment_freq: int = 4,
        payer: bool = True,
        schedule: Schedule | None = None,
    ) -> None:
        self.fixed_rate = fixed_rate
        self.notional = notional
        self.payer = payer
        self.schedule = schedule

        if schedule is not None:
            self.payment_times = schedule.payment_times          # (n,)
            self.deltas = schedule.day_count_fractions           # (n,) — δᵢ per period
            self.maturity = float(schedule.payment_times[-1])
        else:
            if maturity is None:
                raise ValueError("Either maturity or schedule must be provided.")
            self.maturity = maturity
            self.payment_freq = payment_freq
            dt = 1.0 / payment_freq
            n = int(round(maturity * payment_freq))
            self.payment_times = np.array([dt * (i + 1) for i in range(n)])
            self.deltas = np.full(n, dt)                         # uniform δ

        self._payment_times = self._build_payment_times()

    def _build_payment_times(self) -> list:
        """Build and store the list of payment times."""
        return list(self.payment_times)

    def cashflow_times(self) -> list:
        """Return the list of payment times for this swap."""
        return self._payment_times

    def price(self, result: SimulationResult) -> np.ndarray:
        """Compute swap MTM at each time step on each path.

        Uses the standard annuity formula:
        V_payer(t) = N · [(1 - P(t, T_N)) - K · A(t)]

        where A(t) = Σ_{T_i > t} δ · P(t, T_i)  (annuity factor)
        P(t, T) ≈ exp(-r(t) · (T - t))           (flat-curve approximation)

        Returns
        -------
        np.ndarray, shape (n_paths, T)
        """
        r = result.factor("r")  # (n_paths, T)
        time_grid = result.time_grid
        n_paths, n_steps = r.shape
        mtm = np.zeros((n_paths, n_steps))
        hw_model = getattr(result, "model", None)

        future_mask = self.payment_times > 0  # updated per step below
        for i, t in enumerate(time_grid):
            future_mask = self.payment_times > t
            if not future_mask.any():
                continue

            r_t = r[:, i]                                     # (n_paths,)
            future_T = self.payment_times[future_mask]        # (k,)
            future_delta = self.deltas[future_mask]           # (k,)

            # Annuity: Σ δᵢ · P(t, Tᵢ)  — vectorised over payments
            df = _discount_factors(r_t, t, future_T, hw_model)   # (n_paths, k)
            annuity = (future_delta[None, :] * df).sum(axis=1)  # (n_paths,)

            # Final discount P(t, T_N)
            tau_N = self.maturity - t
            if tau_N > 0:
                P_tN = _discount_factors(r_t, t, np.array([self.maturity]), hw_model)[:, 0]
            else:
                P_tN = np.ones(n_paths)

            swap_value = self.notional * ((1 - P_tN) - self.fixed_rate * annuity)
            mtm[:, i] = swap_value if self.payer else -swap_value

        return mtm
