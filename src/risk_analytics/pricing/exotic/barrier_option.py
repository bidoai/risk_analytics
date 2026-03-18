from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from risk_analytics.core.paths import SimulationResult
from risk_analytics.core.stateful import PathState, StatefulPricer


class BarrierOption(StatefulPricer):
    """Down-and-out or up-and-out European barrier option.

    The barrier is monitored continuously, approximated by checking at
    each simulation time step.  At expiry T, surviving paths receive
    ``max(S_T - K, 0)`` (call) or ``max(K - S_T, 0)`` (put), discounted
    back to today at the flat risk-free rate.

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
        """Advance one time step: update barrier state, return MTM at expiry."""
        S_t = result.factor_at(self.factor_name, t_idx)   # (n_paths,)

        # Update knock-out state
        new_active = state.active.copy()
        if self.barrier_type == "down-out":
            new_active &= S_t >= self.barrier
        else:  # "up-out"
            new_active &= S_t <= self.barrier

        # MTM is non-zero only at expiry
        if t >= self.expiry - 1e-9:
            if self.option_type == "call":
                intrinsic = np.maximum(S_t - self.strike, 0.0)
            else:
                intrinsic = np.maximum(self.strike - S_t, 0.0)
            df = np.exp(-self.risk_free_rate * self.expiry)
            mtm = new_active.astype(float) * intrinsic * df
        else:
            mtm = np.zeros(result.n_paths)

        return mtm, BarrierOption.State(active=new_active)
