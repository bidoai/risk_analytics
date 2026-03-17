from __future__ import annotations

import numpy as np
from scipy.stats import norm

from risk_analytics.core.base import Pricer
from risk_analytics.core.paths import SimulationResult


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

    def price(self, result: SimulationResult) -> np.ndarray:
        """MTM via Black-Scholes before expiry; intrinsic value at expiry.

        Parameters
        ----------
        result : SimulationResult
            Must have factor 'S'.

        Returns
        -------
        np.ndarray, shape (n_paths, T)
        """
        S = result.factor("S")  # (n_paths, T)
        time_grid = result.time_grid
        n_paths, n_steps = S.shape
        mtm = np.zeros((n_paths, n_steps))

        for i, t in enumerate(time_grid):
            tau = self.expiry - t
            S_t = S[:, i]

            if tau <= 0:
                # At or past expiry: intrinsic payoff (already paid if past)
                if t == self.expiry or np.isclose(t, self.expiry):
                    if self.option_type == "call":
                        mtm[:, i] = self.notional * np.maximum(S_t - self.strike, 0.0)
                    else:
                        mtm[:, i] = self.notional * np.maximum(self.strike - S_t, 0.0)
                # After expiry: option has expired, MTM = 0
            else:
                mtm[:, i] = self.notional * self._black_scholes(S_t, tau)

        return mtm

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
