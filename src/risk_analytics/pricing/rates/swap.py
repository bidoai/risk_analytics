from __future__ import annotations

import numpy as np

from risk_analytics.core.base import Pricer
from risk_analytics.core.paths import SimulationResult


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
    maturity : float
        Swap maturity in years.
    notional : float
        Notional principal.
    payment_freq : int
        Payments per year (e.g. 4 = quarterly).
    payer : bool
        True = payer (pay fixed, receive floating); False = receiver.
    """

    def __init__(
        self,
        fixed_rate: float,
        maturity: float,
        notional: float = 1_000_000.0,
        payment_freq: int = 4,
        payer: bool = True,
    ) -> None:
        self.fixed_rate = fixed_rate
        self.maturity = maturity
        self.notional = notional
        self.payment_freq = payment_freq
        self.payer = payer

        dt = 1.0 / payment_freq
        n = int(round(maturity * payment_freq))
        self.payment_times = np.array([dt * (i + 1) for i in range(n)])
        self.delta = dt  # day-count fraction per period

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

        for i, t in enumerate(time_grid):
            future_payments = self.payment_times[self.payment_times > t]
            if len(future_payments) == 0:
                continue

            r_t = r[:, i]  # (n_paths,)

            # Annuity factor: sum of discount factors for each future payment date
            annuity = np.zeros(n_paths)
            for T_p in future_payments:
                tau = T_p - t
                annuity += self.delta * np.exp(-r_t * tau)

            # Final discount factor P(t, T_N)
            tau_N = self.maturity - t
            P_tN = np.exp(-r_t * tau_N) if tau_N > 0 else np.ones(n_paths)

            # Swap value (payer = +floating -fixed)
            swap_value = self.notional * ((1 - P_tN) - self.fixed_rate * annuity)

            mtm[:, i] = swap_value if self.payer else -swap_value

        return mtm
