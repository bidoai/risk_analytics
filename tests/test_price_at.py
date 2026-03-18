"""Tests for Pricer.price_at() — default and efficient overrides."""
from __future__ import annotations

import numpy as np
import pytest

from risk_analytics.core.paths import SimulationResult
from risk_analytics.pricing.rates.swap import InterestRateSwap
from risk_analytics.pricing.rates.bond import ZeroCouponBond, FixedRateBond


def _hw_result(n_paths: int = 50, n_steps: int = 21, r0: float = 0.04) -> SimulationResult:
    """Minimal SimulationResult with constant short rates (no HW model attached)."""
    time_grid = np.linspace(0, 5.0, n_steps)
    paths = np.full((n_paths, n_steps, 1), r0)
    return SimulationResult(
        time_grid=time_grid,
        paths=paths,
        model_name="HW",
        factor_names=["r"],
    )


class TestPriceAtIRS:
    def test_matches_full_price(self):
        result = _hw_result()
        swap = InterestRateSwap(fixed_rate=0.04, maturity=5.0, notional=1e6)
        full = swap.price(result)
        for t_idx in [0, 5, 10, 15, 20]:
            np.testing.assert_allclose(
                swap.price_at(result, t_idx),
                full[:, t_idx],
                rtol=1e-10,
                err_msg=f"IRS price_at mismatch at t_idx={t_idx}",
            )

    def test_receiver_swap(self):
        result = _hw_result()
        swap = InterestRateSwap(fixed_rate=0.04, maturity=5.0, payer=False)
        full = swap.price(result)
        t_idx = 3
        np.testing.assert_allclose(swap.price_at(result, t_idx), full[:, t_idx], rtol=1e-10)

    def test_after_maturity_zero(self):
        result = _hw_result(n_steps=11)
        swap = InterestRateSwap(fixed_rate=0.04, maturity=2.0, notional=1e6)
        # t_idx where t > maturity
        late_idx = 10  # t = 5.0 > 2.0
        np.testing.assert_array_equal(swap.price_at(result, late_idx), 0.0)


class TestPriceAtZCB:
    def test_matches_full_price(self):
        result = _hw_result(n_steps=21)
        zcb = ZeroCouponBond(maturity=3.0, face_value=1_000_000)
        full = zcb.price(result)
        for t_idx in [0, 5, 10]:
            np.testing.assert_allclose(
                zcb.price_at(result, t_idx),
                full[:, t_idx],
                rtol=1e-10,
                err_msg=f"ZCB price_at mismatch at t_idx={t_idx}",
            )

    def test_at_or_after_maturity_zero(self):
        result = _hw_result(n_steps=11)
        zcb = ZeroCouponBond(maturity=2.5, face_value=1e6)
        # time_grid goes to 5.0; last step t=5.0 > 2.5
        last_idx = 10
        np.testing.assert_array_equal(zcb.price_at(result, last_idx), 0.0)


class TestPriceAtFixedRateBond:
    def test_matches_full_price(self):
        result = _hw_result(n_paths=30, n_steps=21)
        bond = FixedRateBond(coupon_rate=0.05, maturity=5.0, coupon_freq=2, face_value=1e6)
        full = bond.price(result)
        for t_idx in [0, 5, 15]:
            np.testing.assert_allclose(
                bond.price_at(result, t_idx),
                full[:, t_idx],
                rtol=1e-10,
                err_msg=f"FixedRateBond price_at mismatch at t_idx={t_idx}",
            )

    def test_shape(self):
        result = _hw_result(n_paths=20, n_steps=11)
        bond = FixedRateBond(coupon_rate=0.04, maturity=3.0, coupon_freq=1, face_value=1e6)
        out = bond.price_at(result, 0)
        assert out.shape == (20,)
