from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from risk_analytics.core.paths import SimulationResult
from risk_analytics.core.stateful import PathState, StatefulPricer


@dataclass
class AsianState(PathState):
    """Per-path running average state for an arithmetic Asian option.

    Attributes
    ----------
    avg_sum : np.ndarray, shape (n_paths,)
        Running sum of observed spot prices.
    count : int
        Number of observations accumulated so far (starts at 1 at t=0).
    """
    avg_sum: np.ndarray
    count: int

    @classmethod
    def allocate(cls, n_paths: int) -> "AsianState":
        return cls(avg_sum=np.zeros(n_paths), count=0)

    def copy(self) -> "AsianState":
        return AsianState(avg_sum=self.avg_sum.copy(), count=self.count)


class AsianOption(StatefulPricer):
    """Arithmetic average Asian call option (StatefulPricer).

    Payoff at expiry: max(avg(S) - K, 0)
    where avg(S) is the arithmetic mean of spot prices observed at each
    simulation time step up to and including expiry.

    Pre-expiry MTM: 0 (no analytical interim approximation yet).

    Parameters
    ----------
    strike : float
        Option strike price.
    expiry : float
        Option expiry in years.
    risk_free_rate : float
        Flat discount rate used to compute present value at expiry.
    factor_name : str
        Name of the spot factor in SimulationResult (default "S" for GBM).
    """

    def __init__(
        self,
        strike: float,
        expiry: float,
        risk_free_rate: float = 0.0,
        factor_name: str = "S",
    ) -> None:
        self.strike = strike
        self.expiry = expiry
        self.risk_free_rate = risk_free_rate
        self.factor_name = factor_name

    def cashflow_times(self) -> list:
        return [self.expiry]

    def allocate_state(self, n_paths: int) -> AsianState:
        return AsianState.allocate(n_paths)

    def step(
        self,
        result: SimulationResult,
        t: float,
        t_idx: int,
        state: AsianState,
    ) -> tuple[np.ndarray, AsianState]:
        """Advance one time step: update running average and compute MTM.

        At t=0, initialises the running sum with S(0).
        At subsequent steps, accumulates S(t) into the running sum.
        At expiry, returns the discounted payoff max(avg - K, 0).
        Pre-expiry, returns zero (no interim analytical MTM).
        """
        S_t = result.factor_at(self.factor_name, t_idx)  # (n_paths,)

        new_avg_sum = state.avg_sum + S_t
        new_count = state.count + 1
        running_avg = new_avg_sum / new_count

        if t >= self.expiry - 1e-8:
            df = np.exp(-self.risk_free_rate * self.expiry)
            mtm = df * np.maximum(running_avg - self.strike, 0.0)
        else:
            mtm = np.zeros(result.n_paths)

        return mtm, AsianState(avg_sum=new_avg_sum, count=new_count)
