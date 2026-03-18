from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from risk_analytics.core.paths import SimulationResult
from risk_analytics.core.stateful import PathState, StatefulPricer


def _bs_call(S: np.ndarray, K: float, sigma: float, r: float, tau: float) -> np.ndarray:
    """Vectorised Black-Scholes call price."""
    tau = max(tau, 1e-10)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)


def _bs_put(S: np.ndarray, K: float, sigma: float, r: float, tau: float) -> np.ndarray:
    """Vectorised Black-Scholes put price."""
    tau = max(tau, 1e-10)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return K * np.exp(-r * tau) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _barrier_analytical_mtm(
    S: np.ndarray,
    K: float,
    H: float,
    sigma: float,
    r: float,
    tau: float,
    barrier_type: str,
    option_type: str,
) -> np.ndarray:
    """Analytical Black-Scholes price for a down-and-out or up-and-out barrier option.

    For down-and-out call (H < K):
        Price = BS_call(S, K) - (H/S)^(2λ) * BS_call(H²/S, K)
        where λ = r/σ² + 0.5

    For up-and-out put (H > K):
        Price = BS_put(S, K) - (H/S)^(2λ) * BS_put(H²/S, K)

    Returns 0 for paths already below/above the barrier (knocked out).
    """
    lam = r / (sigma**2) + 0.5
    S_safe = np.where(S > 0, S, 1e-300)

    if option_type == "call":
        vanilla = _bs_call(S_safe, K, sigma, r, tau)
        reflected = _bs_call(H**2 / S_safe, K, sigma, r, tau)
        price = vanilla - (H / S_safe) ** (2 * lam) * reflected
    else:
        vanilla = _bs_put(S_safe, K, sigma, r, tau)
        reflected = _bs_put(H**2 / S_safe, K, sigma, r, tau)
        price = vanilla - (H / S_safe) ** (2 * lam) * reflected

    # Prices cannot be negative (floor at 0)
    return np.maximum(price, 0.0)


class BarrierOption(StatefulPricer):
    """Down-and-out or up-and-out European barrier option.

    The barrier is monitored continuously, approximated by checking at
    each simulation time step.  At expiry T, surviving paths receive
    ``max(S_T - K, 0)`` (call) or ``max(K - S_T, 0)`` (put), discounted
    back to today at the flat risk-free rate.

    Pre-expiry MTM uses the analytical Black-Scholes barrier option formula
    for surviving paths. Knocked-out paths always receive MTM = 0.

    Parameters
    ----------
    strike : float
        Option strike price.
    barrier : float
        Barrier level.  For ``"down-out"``, paths where ``S_t < barrier``
        are knocked out.  For ``"up-out"``, paths where ``S_t > barrier``
        are knocked out.
    expiry : float
        Option expiry in years.
    barrier_type : str
        ``"down-out"`` (default) or ``"up-out"``.
    sigma : float
        Volatility used for the pre-expiry analytical Black-Scholes price.
        Defaults to 0.20 (20% flat vol).
    factor_name : str
        Name of the equity/FX factor in SimulationResult (default ``"S"``).
    risk_free_rate : float
        Flat discount rate used to compute present value at expiry.
    option_type : str
        ``"call"`` (default) or ``"put"``.
    """

    @dataclass
    class State(PathState):
        """Per-path knock-out state."""
        active: np.ndarray   # bool (n_paths,) — True while option is still alive

        @classmethod
        def allocate(cls, n_paths: int) -> "BarrierOption.State":
            return cls(active=np.ones(n_paths, dtype=bool))

        def copy(self) -> "BarrierOption.State":
            return BarrierOption.State(active=self.active.copy())

    def __init__(
        self,
        strike: float,
        barrier: float,
        expiry: float,
        barrier_type: str = "down-out",
        sigma: float = 0.20,
        factor_name: str = "S",
        risk_free_rate: float = 0.04,
        option_type: str = "call",
    ) -> None:
        if barrier_type not in ("down-out", "up-out"):
            raise ValueError(f"barrier_type must be 'down-out' or 'up-out', got '{barrier_type}'")
        if option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

        self.strike = strike
        self.barrier = barrier
        self.expiry = expiry
        self.barrier_type = barrier_type
        self.sigma = sigma
        self.factor_name = factor_name
        self.risk_free_rate = risk_free_rate
        self.option_type = option_type

    def cashflow_times(self) -> list:
        return [self.expiry]

    def allocate_state(self, n_paths: int) -> "BarrierOption.State":
        return BarrierOption.State.allocate(n_paths)

    def step(
        self,
        result: SimulationResult,
        t: float,
        t_idx: int,
        state: "BarrierOption.State",
    ) -> tuple[np.ndarray, "BarrierOption.State"]:
        """Advance one time step: update barrier state, return MTM.

        At expiry: discounted intrinsic payoff for surviving paths.
        Pre-expiry: analytical Black-Scholes barrier option price for surviving
        paths, 0 for knocked-out paths.
        """
        S_t = result.factor_at(self.factor_name, t_idx)   # (n_paths,)

        # Update knock-out state
        new_active = state.active.copy()
        if self.barrier_type == "down-out":
            new_active &= S_t >= self.barrier
        else:  # "up-out"
            new_active &= S_t <= self.barrier

        if t >= self.expiry - 1e-9:
            # At expiry: discounted intrinsic payoff
            if self.option_type == "call":
                intrinsic = np.maximum(S_t - self.strike, 0.0)
            else:
                intrinsic = np.maximum(self.strike - S_t, 0.0)
            df = np.exp(-self.risk_free_rate * self.expiry)
            mtm = new_active.astype(float) * intrinsic * df
        else:
            # Pre-expiry: analytical barrier option price for surviving paths
            tau = self.expiry - t
            analytical = _barrier_analytical_mtm(
                S=S_t,
                K=self.strike,
                H=self.barrier,
                sigma=self.sigma,
                r=self.risk_free_rate,
                tau=tau,
                barrier_type=self.barrier_type,
                option_type=self.option_type,
            )
            mtm = new_active.astype(float) * analytical

        return mtm, BarrierOption.State(active=new_active)
