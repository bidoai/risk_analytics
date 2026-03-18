from __future__ import annotations

import numpy as np
from scipy.stats import norm

from pyxva.core.base import Pricer
from pyxva.core.paths import SimulationResult


class EuropeanOption(Pricer):
    """European vanilla call or put option priced on simulated equity paths.

    At expiry, the payoff is realised. Before expiry, the option is
    re-priced analytically using Black-Scholes with the simulated spot and
    implied vol (or the model's own sigma if available).

    Parameters
    ----------
    strike : float
        Option strike price.
    expiry : float
        Option expiry in years.
    sigma : float
        Implied volatility used for Black-Scholes re-pricing before expiry.
    risk_free_rate : float
        Risk-free rate for discounting.
    option_type : str
        'call' or 'put'.
    notional : float
        Number of contracts / multiplier.
    """

    def __init__(
        self,
        strike: float,
        expiry: float,
        sigma: float = 0.20,
        risk_free_rate: float = 0.03,
        option_type: str = "call",
        notional: float = 1.0,
    ) -> None:
        if option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")
        self.strike = strike
        self.expiry = expiry
        self.sigma = sigma
        self.risk_free_rate = risk_free_rate
        self.option_type = option_type
        self.notional = notional

    def cashflow_times(self) -> list:
        """Return the expiry as the single cashflow time."""
        return [self.expiry]

    def price(self, result: SimulationResult) -> np.ndarray:
        """MTM via Black-Scholes before expiry; intrinsic value at expiry.

        Fully vectorised: no Python loop over time steps. Boolean index
        slices select the live, at-expiry, and expired columns and price
        them in one NumPy call each.

        Parameters
        ----------
        result : SimulationResult
            Must have factor 'S'.

        Returns
        -------
        np.ndarray, shape (n_paths, T)
        """
        S = result.factor("S")   # (n_paths, T)
        time_grid = result.time_grid
        tau = self.expiry - time_grid  # (T,) — positive before expiry

        live = tau > 0                          # columns before expiry
        at_exp = np.isclose(time_grid, self.expiry)  # column(s) at expiry
        # expired columns (tau < 0 and not at_exp) stay zero

        mtm = np.zeros_like(S)

        # --- Before expiry: Black-Scholes on all live columns at once ---
        if np.any(live):
            tau_live = tau[live]              # (n_live,)
            S_live = S[:, live]              # (n_paths, n_live)
            sqrt_tau = np.sqrt(tau_live)

            K, r, sig = self.strike, self.risk_free_rate, self.sigma
            with np.errstate(divide="ignore", invalid="ignore"):
                d1 = (np.log(S_live / K) + (r + 0.5 * sig**2) * tau_live) / (sig * sqrt_tau)
            d2 = d1 - sig * sqrt_tau

            if self.option_type == "call":
                mtm[:, live] = self.notional * (
                    S_live * norm.cdf(d1) - K * np.exp(-r * tau_live) * norm.cdf(d2)
                )
            else:
                mtm[:, live] = self.notional * (
                    K * np.exp(-r * tau_live) * norm.cdf(-d2) - S_live * norm.cdf(-d1)
                )

        # --- At expiry: intrinsic payoff ---
        if np.any(at_exp):
            S_exp = S[:, at_exp]             # (n_paths, n_at_exp) — usually 1 column
            if self.option_type == "call":
                mtm[:, at_exp] = self.notional * np.maximum(S_exp - self.strike, 0.0)
            else:
                mtm[:, at_exp] = self.notional * np.maximum(self.strike - S_exp, 0.0)

        return mtm

    def price_at(self, result: SimulationResult, t_idx: int) -> np.ndarray:
        """MTM at a single time step — O(n_paths), avoids allocating (n_paths, T).

        Consistent with IRS and bond overrides: only the spot slice at t_idx
        is needed, making this efficient for StreamingExposureEngine.

        Returns
        -------
        np.ndarray, shape (n_paths,)
        """
        t = result.time_grid[t_idx]
        tau = self.expiry - t
        S = result.factor("S")[:, t_idx]  # (n_paths,)

        if tau < -1e-10:
            # Past expiry: option has expired worthless
            return np.zeros(len(S))

        if abs(tau) <= 1e-10:
            # At expiry: intrinsic payoff
            if self.option_type == "call":
                return self.notional * np.maximum(S - self.strike, 0.0)
            else:
                return self.notional * np.maximum(self.strike - S, 0.0)

        # Before expiry: Black-Scholes on the single spot slice
        return self.notional * self._black_scholes(S, tau)

    def _black_scholes(self, S: np.ndarray, tau: float) -> np.ndarray:
        """Vectorised Black-Scholes price."""
        K = self.strike
        r = self.risk_free_rate
        sigma = self.sigma

        with np.errstate(divide="ignore", invalid="ignore"):
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
            d2 = d1 - sigma * np.sqrt(tau)

        if self.option_type == "call":
            return S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
        else:
            return K * np.exp(-r * tau) * norm.cdf(-d2) - S * norm.cdf(-d1)

    @staticmethod
    def black_scholes_price(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
    ) -> float:
        """Standalone Black-Scholes formula for benchmarking."""
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == "call":
            return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
        else:
            return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
