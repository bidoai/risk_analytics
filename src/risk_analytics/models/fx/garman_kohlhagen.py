from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from risk_analytics.core.base import StochasticModel
from risk_analytics.core.paths import SimulationResult

logger = logging.getLogger(__name__)


class GarmanKohlhagen(StochasticModel):
    """Garman-Kohlhagen FX spot model.

    Log-normal FX spot with domestic and foreign risk-free rates:

        dS(t) = (r_d - r_f)·S(t) dt + σ·S(t) dW(t)

    Exact discretisation (no Euler bias):

        S(t+dt) = S(t)·exp((r_d - r_f - σ²/2)·dt + σ·√dt·Z)

    Parameters
    ----------
    S0 : float
        Initial spot FX rate (units of domestic per unit of foreign).
    r_d : float
        Domestic risk-free rate (continuously compounded).
    r_f : float
        Foreign risk-free rate / dividend yield of the foreign currency.
    sigma : float
        FX volatility (annualised).
    """

    def __init__(
        self,
        S0: float = 1.10,
        r_d: float = 0.03,
        r_f: float = 0.01,
        sigma: float = 0.10,
    ) -> None:
        self.S0 = S0
        self.r_d = r_d
        self.r_f = r_f
        self.sigma = sigma

    @property
    def n_factors(self) -> int:
        return 1

    @property
    def name(self) -> str:
        return "GarmanKohlhagen"

    def simulate(
        self,
        time_grid: np.ndarray,
        n_paths: int,
        random_draws: np.ndarray,
    ) -> SimulationResult:
        """Exact log-normal simulation of the FX spot.

        Parameters
        ----------
        time_grid : np.ndarray, shape (T,)
        n_paths : int
        random_draws : np.ndarray, shape (n_paths, T-1, 1)

        Returns
        -------
        SimulationResult with factor 'S', shape (n_paths, T, 1)
        """
        dt = np.diff(time_grid)   # (T-1,)
        Z = random_draws[:, :, 0]  # (n_paths, T-1)

        drift = (self.r_d - self.r_f - 0.5 * self.sigma**2) * dt  # (T-1,)
        diffusion = self.sigma * np.sqrt(dt) * Z                    # (n_paths, T-1)

        log_increments = drift + diffusion                          # (n_paths, T-1)
        log_paths = np.empty((n_paths, len(time_grid)))
        log_paths[:, 0] = np.log(self.S0)
        log_paths[:, 1:] = np.log(self.S0) + np.cumsum(log_increments, axis=1)

        return SimulationResult(
            time_grid=time_grid,
            paths=np.exp(log_paths)[:, :, np.newaxis],
            model_name=self.name,
            factor_names=["S"],
        )

    def calibrate(self, market_data: dict) -> None:
        """Calibrate sigma (and optionally rates) to market data.

        Expected market_data keys:
        - 'S0'      : float, current spot
        - 'r_d'     : float, domestic rate
        - 'r_f'     : float, foreign rate
        - 'atm_vol' : float, ATM implied volatility (sets sigma directly)
        - 'option_price', 'strike', 'maturity', 'option_type' : fit sigma
          to a single vanilla FX option price via Brent's method.
        """
        if "S0" in market_data:
            self.S0 = float(market_data["S0"])
        if "r_d" in market_data:
            self.r_d = float(market_data["r_d"])
        if "r_f" in market_data:
            self.r_f = float(market_data["r_f"])

        if "atm_vol" in market_data:
            self.sigma = float(market_data["atm_vol"])
        elif all(k in market_data for k in ("option_price", "strike", "maturity")):
            self._fit_sigma(market_data)

        logger.info(
            "GarmanKohlhagen calibrated: S0=%.4g  r_d=%.4f  r_f=%.4f  sigma=%.4f",
            self.S0, self.r_d, self.r_f, self.sigma,
        )

    def get_params(self) -> dict:
        return {"S0": self.S0, "r_d": self.r_d, "r_f": self.r_f, "sigma": self.sigma}

    def set_params(self, params: dict) -> None:
        for attr in ("S0", "r_d", "r_f", "sigma"):
            if attr in params:
                setattr(self, attr, float(params[attr]))

    # ------------------------------------------------------------------
    # Garman-Kohlhagen analytical formula
    # ------------------------------------------------------------------

    @staticmethod
    def gk_price(
        S: float,
        K: float,
        T: float,
        r_d: float,
        r_f: float,
        sigma: float,
        option_type: str = "call",
    ) -> float:
        """Garman-Kohlhagen price for a European FX vanilla option.

        Call:  S·e^{-r_f·T}·N(d1) - K·e^{-r_d·T}·N(d2)
        Put:   K·e^{-r_d·T}·N(-d2) - S·e^{-r_f·T}·N(-d1)
        """
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r_d - r_f + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        if option_type == "call":
            return float(S * np.exp(-r_f * T) * norm.cdf(d1)
                         - K * np.exp(-r_d * T) * norm.cdf(d2))
        else:
            return float(K * np.exp(-r_d * T) * norm.cdf(-d2)
                         - S * np.exp(-r_f * T) * norm.cdf(-d1))

    def forward(self, T: float) -> float:
        """Analytical FX forward: F(0,T) = S0·exp((r_d - r_f)·T)."""
        return float(self.S0 * np.exp((self.r_d - self.r_f) * T))

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _fit_sigma(self, market_data: dict) -> None:
        target = float(market_data["option_price"])
        K = float(market_data["strike"])
        T = float(market_data["maturity"])
        opt_type = market_data.get("option_type", "call")

        def error(sigma: float) -> float:
            return (self.gk_price(self.S0, K, T, self.r_d, self.r_f, sigma, opt_type) - target) ** 2

        result = minimize_scalar(error, bounds=(1e-4, 5.0), method="bounded")
        self.sigma = float(result.x)
