"""Tests for Pricer.price_at() — default and efficient overrides."""
from __future__ import annotations

import numpy as np
import pytest

from pyxva.core.paths import SimulationResult
from pyxva.pricing.equity.vanilla_option import EuropeanOption
from pyxva.pricing.rates.swap import InterestRateSwap
from pyxva.pricing.rates.bond import ZeroCouponBond, FixedRateBond


def _equity_result(n_paths: int = 50, n_steps: int = 21, S0: float = 100.0) -> SimulationResult:
    """Flat equity path at constant spot S0."""
    time_grid = np.linspace(0, 5.0, n_steps)
    paths = np.full((n_paths, n_steps, 1), S0)
    return SimulationResult(
        time_grid=time_grid,
        paths=paths,
        model_name="GBM",
        factor_names=["S"],
    )


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


class TestPriceAtEuropeanOption:
    def test_matches_full_price(self):
        """price_at() should agree with price()[:, t_idx] at every step."""
        result = _equity_result(n_paths=50, n_steps=21)
        opt = EuropeanOption(strike=100.0, expiry=3.0, sigma=0.20, risk_free_rate=0.04)
        full = opt.price(result)
        for t_idx in [0, 5, 10, 14]:  # all pre-expiry steps
            np.testing.assert_allclose(
                opt.price_at(result, t_idx),
                full[:, t_idx],
                rtol=1e-8,
                err_msg=f"EuropeanOption price_at mismatch at t_idx={t_idx}",
            )

    def test_at_expiry_intrinsic(self):
        """At expiry, price_at should return the intrinsic payoff."""
        n_paths = 100
        time_grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        S = np.linspace(80.0, 120.0, n_paths)
        paths = np.tile(S[:, None, None], (1, len(time_grid), 1))
        result = SimulationResult(
            time_grid=time_grid,
            paths=paths,
            model_name="GBM",
            factor_names=["S"],
        )
        opt = EuropeanOption(strike=100.0, expiry=3.0, sigma=0.20)
        at_exp_idx = 3  # t = 3.0 == expiry
        payoff = opt.price_at(result, at_exp_idx)
        expected = np.maximum(S - 100.0, 0.0)
        np.testing.assert_allclose(payoff, expected, atol=1e-10)

    def test_after_expiry_zero(self):
        """Past expiry the option has no remaining value."""
        result = _equity_result(n_paths=30, n_steps=11)
        opt = EuropeanOption(strike=100.0, expiry=2.0, sigma=0.20)
        last_idx = 10  # t = 5.0 > 2.0
        np.testing.assert_array_equal(opt.price_at(result, last_idx), 0.0)

    def test_put_matches_full_price(self):
        result = _equity_result(n_paths=40, n_steps=21)
        put = EuropeanOption(strike=95.0, expiry=2.5, sigma=0.25, option_type="put")
        full = put.price(result)
        t_idx = 7
        np.testing.assert_allclose(put.price_at(result, t_idx), full[:, t_idx], rtol=1e-8)


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
