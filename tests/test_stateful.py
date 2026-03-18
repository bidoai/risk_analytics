"""Tests for PathState, StatefulPricer, and BarrierOption."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from risk_analytics.core.paths import SimulationResult
from risk_analytics.core.stateful import PathState, StatefulPricer
from risk_analytics.pricing.exotic.barrier_option import BarrierOption


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(n_paths: int, n_steps: int, factor_name: str = "S") -> SimulationResult:
    """Create a SimulationResult with linearly increasing spot paths."""
    time_grid = np.linspace(0, 1.0, n_steps)
    # Paths: S increases from 100 to 120 linearly on all paths
    paths = np.linspace(100.0, 120.0, n_steps)[None, :, None] * np.ones((n_paths, 1, 1))
    return SimulationResult(
        time_grid=time_grid,
        paths=paths,
        model_name="GBM",
        factor_names=[factor_name],
        interpolation_space=["log"],
    )


# ---------------------------------------------------------------------------
# PathState
# ---------------------------------------------------------------------------

class TestPathState:
    def test_allocate_not_implemented(self):
        with pytest.raises(NotImplementedError):
            PathState.allocate(10)

    def test_copy_is_deep(self):
        @dataclass
        class MyState(PathState):
            arr: np.ndarray

            @classmethod
            def allocate(cls, n_paths):
                return cls(arr=np.zeros(n_paths))

        s = MyState.allocate(5)
        s2 = s.copy()
        s2.arr[0] = 99.0
        assert s.arr[0] == 0.0, "copy should be independent"


# ---------------------------------------------------------------------------
# StatefulPricer — concrete minimal implementation
# ---------------------------------------------------------------------------

@dataclass
class _RunningMaxState(PathState):
    max_so_far: np.ndarray

    @classmethod
    def allocate(cls, n_paths):
        return cls(max_so_far=np.full(n_paths, -np.inf))


class _LookbackCallPricer(StatefulPricer):
    """MTM = current spot - running max (negative until new high; >=0 at expiry)."""

    def __init__(self, expiry: float):
        self.expiry = expiry

    def cashflow_times(self):
        return [self.expiry]

    def allocate_state(self, n_paths):
        return _RunningMaxState.allocate(n_paths)

    def step(self, result, t, t_idx, state):
        S_t = result.factor_at("S", t_idx)
        new_max = np.maximum(state.max_so_far, S_t)
        new_state = _RunningMaxState(max_so_far=new_max)
        if t >= self.expiry - 1e-9:
            mtm = S_t - new_max   # payoff
        else:
            mtm = np.zeros(result.n_paths)
        return mtm, new_state


class TestStatefulPricer:
    def test_price_shape(self):
        result = _make_result(n_paths=20, n_steps=13)
        pricer = _LookbackCallPricer(expiry=1.0)
        mtm = pricer.price(result)
        assert mtm.shape == (20, 13)

    def test_price_at_matches_price(self):
        result = _make_result(n_paths=10, n_steps=11)
        pricer = _LookbackCallPricer(expiry=1.0)
        full = pricer.price(result)
        for t_idx in [0, 5, 10]:
            np.testing.assert_allclose(
                pricer.price_at(result, t_idx),
                full[:, t_idx],
                err_msg=f"price_at mismatch at t_idx={t_idx}",
            )

    def test_state_accumulates(self):
        """Running max must be non-decreasing along each path."""
        result = _make_result(n_paths=5, n_steps=7)
        pricer = _LookbackCallPricer(expiry=1.0)
        state = pricer.allocate_state(result.n_paths)
        prev_max = np.full(result.n_paths, -np.inf)
        for i, t in enumerate(result.time_grid):
            _, state = pricer.step(result, t, i, state)
            assert np.all(state.max_so_far >= prev_max)
            prev_max = state.max_so_far.copy()


# ---------------------------------------------------------------------------
# BarrierOption
# ---------------------------------------------------------------------------

class TestBarrierOption:
    def test_down_out_all_survive(self):
        """All paths survive when spot always above barrier."""
        n_paths, n_steps = 50, 13
        result = _make_result(n_paths, n_steps)   # S in [100, 120]
        opt = BarrierOption(strike=105.0, barrier=90.0, expiry=1.0)
        mtm = opt.price(result)
        # At expiry (last step), all paths have S>90 so all survive; call payoff = S-105 > 0
        last_S = result.factor_at("S", n_steps - 1)
        expected = np.maximum(last_S - 105.0, 0.0) * np.exp(-0.04 * 1.0)
        np.testing.assert_allclose(mtm[:, -1], expected, rtol=1e-6)

    def test_down_out_all_knocked(self):
        """All paths are knocked out when spot drops below barrier."""
        n_paths, n_steps = 10, 5
        time_grid = np.linspace(0, 1.0, n_steps)
        # Paths: S = 80 < barrier = 100  on all paths at all steps
        paths = np.full((n_paths, n_steps, 1), 80.0)
        result = SimulationResult(
            time_grid=time_grid, paths=paths, model_name="GBM", factor_names=["S"]
        )
        opt = BarrierOption(strike=70.0, barrier=100.0, expiry=1.0, barrier_type="down-out")
        mtm = opt.price(result)
        # All knocked out → payoff is 0 everywhere
        np.testing.assert_array_equal(mtm[:, -1], 0.0)

    def test_up_out_knocked(self):
        """Paths above barrier are knocked out for up-and-out."""
        n_paths, n_steps = 5, 7
        time_grid = np.linspace(0, 1.0, n_steps)
        paths = np.full((n_paths, n_steps, 1), 150.0)   # S > barrier=130
        result = SimulationResult(
            time_grid=time_grid, paths=paths, model_name="GBM", factor_names=["S"]
        )
        opt = BarrierOption(strike=100.0, barrier=130.0, expiry=1.0, barrier_type="up-out")
        mtm = opt.price(result)
        np.testing.assert_array_equal(mtm[:, -1], 0.0)

    def test_put_payoff(self):
        """Put barrier option: payoff = max(K - S, 0) for surviving paths."""
        n_paths, n_steps = 10, 5
        time_grid = np.linspace(0, 1.0, n_steps)
        paths = np.full((n_paths, n_steps, 1), 95.0)  # S=95 < K=100, > barrier=80
        result = SimulationResult(
            time_grid=time_grid, paths=paths, model_name="GBM", factor_names=["S"]
        )
        opt = BarrierOption(
            strike=100.0, barrier=80.0, expiry=1.0,
            barrier_type="down-out", option_type="put",
        )
        mtm = opt.price(result)
        df = np.exp(-0.04 * 1.0)
        np.testing.assert_allclose(mtm[:, -1], 5.0 * df, rtol=1e-6)

    def test_invalid_barrier_type(self):
        with pytest.raises(ValueError, match="barrier_type"):
            BarrierOption(strike=100, barrier=80, expiry=1.0, barrier_type="sideways")

    def test_cashflow_times(self):
        opt = BarrierOption(strike=100, barrier=80, expiry=2.5)
        assert opt.cashflow_times() == [2.5]

    def test_state_type(self):
        opt = BarrierOption(strike=100, barrier=80, expiry=1.0)
        state = opt.allocate_state(10)
        assert isinstance(state, BarrierOption.State)
        assert state.active.shape == (10,)
        assert state.active.all()
