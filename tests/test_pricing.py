"""Tests for pricing models."""
import numpy as np
import pytest

from risk_analytics.core import MonteCarloEngine, TimeGrid
from risk_analytics.models import GeometricBrownianMotion, HullWhite1F
from risk_analytics.pricing import (
    EuropeanOption,
    FixedRateBond,
    InterestRateSwap,
    ZeroCouponBond,
)

N_PATHS = 5000
SEED = 42


class TestEuropeanOption:
    def setup_method(self):
        self.S0 = 100.0
        self.K = 100.0
        self.T = 1.0
        self.r = 0.03
        self.sigma = 0.20
        self.model = GeometricBrownianMotion(S0=self.S0, mu=self.r, sigma=self.sigma)
        self.grid = TimeGrid.uniform(self.T, 52)
        engine = MonteCarloEngine(N_PATHS, seed=SEED)
        self.result = engine.run([self.model], self.grid)["GBM"]

    def test_call_price_matches_black_scholes(self):
        """MC call price at t=0 should match Black-Scholes within tolerance."""
        option = EuropeanOption(
            strike=self.K,
            expiry=self.T,
            sigma=self.sigma,
            risk_free_rate=self.r,
            option_type="call",
        )
        mtm = option.price(self.result)
        mc_price = mtm[:, 0].mean()

        bs_price = EuropeanOption.black_scholes_price(
            self.S0, self.K, self.T, self.r, self.sigma, "call"
        )
        # Allow 2% relative tolerance for MC noise
        assert abs(mc_price - bs_price) / bs_price < 0.02, (
            f"MC={mc_price:.4f}, BS={bs_price:.4f}"
        )

    def test_put_price_matches_black_scholes(self):
        option = EuropeanOption(
            strike=self.K,
            expiry=self.T,
            sigma=self.sigma,
            risk_free_rate=self.r,
            option_type="put",
        )
        mtm = option.price(self.result)
        mc_price = mtm[:, 0].mean()
        bs_price = EuropeanOption.black_scholes_price(
            self.S0, self.K, self.T, self.r, self.sigma, "put"
        )
        assert abs(mc_price - bs_price) / bs_price < 0.02

    def test_put_call_parity(self):
        """C - P = S - K*exp(-rT)"""
        call = EuropeanOption(self.K, self.T, self.sigma, self.r, "call")
        put = EuropeanOption(self.K, self.T, self.sigma, self.r, "put")
        c = call.price(self.result)[:, 0].mean()
        p = put.price(self.result)[:, 0].mean()
        parity = self.S0 - self.K * np.exp(-self.r * self.T)
        assert abs((c - p) - parity) < 0.5

    def test_invalid_option_type(self):
        with pytest.raises(ValueError):
            EuropeanOption(100, 1.0, 0.2, 0.03, "binary")

    def test_mtm_shape(self):
        option = EuropeanOption(self.K, self.T, self.sigma, self.r, "call")
        mtm = option.price(self.result)
        assert mtm.shape == (N_PATHS, len(self.grid))


class TestInterestRateSwap:
    def setup_method(self):
        self.model = HullWhite1F(a=0.1, sigma=0.01, r0=0.05)
        self.grid = TimeGrid.uniform(5.0, 60)
        engine = MonteCarloEngine(N_PATHS, seed=SEED)
        self.result = engine.run([self.model], self.grid)["HullWhite1F"]

    def test_mtm_shape(self):
        swap = InterestRateSwap(fixed_rate=0.05, maturity=5.0, notional=1_000_000)
        mtm = swap.price(self.result)
        assert mtm.shape == (N_PATHS, len(self.grid))

    def test_at_market_swap_near_zero(self):
        """At-market swap (fixed rate = current rate) should have ~0 initial MTM."""
        swap = InterestRateSwap(fixed_rate=0.05, maturity=5.0, notional=1_000_000)
        mtm = swap.price(self.result)
        initial_mtm = mtm[:, 0].mean()
        # At par swap: initial value should be small relative to notional
        assert abs(initial_mtm) / 1_000_000 < 0.05

    def test_payer_receiver_symmetry(self):
        """Payer + receiver should net to zero."""
        payer = InterestRateSwap(fixed_rate=0.05, maturity=5.0, notional=1e6, payer=True)
        receiver = InterestRateSwap(fixed_rate=0.05, maturity=5.0, notional=1e6, payer=False)
        mtm_sum = payer.price(self.result) + receiver.price(self.result)
        assert np.allclose(mtm_sum, 0.0)


class TestFixedRateBond:
    def setup_method(self):
        self.model = HullWhite1F(a=0.1, sigma=0.01, r0=0.05)
        self.grid = TimeGrid.uniform(5.0, 60)
        engine = MonteCarloEngine(N_PATHS, seed=SEED)
        self.result = engine.run([self.model], self.grid)["HullWhite1F"]

    def test_mtm_shape(self):
        bond = FixedRateBond(coupon_rate=0.05, maturity=5.0, face_value=1000.0)
        mtm = bond.price(self.result)
        assert mtm.shape == (N_PATHS, len(self.grid))

    def test_bond_positive(self):
        """Bond prices should be positive before maturity."""
        bond = FixedRateBond(coupon_rate=0.05, maturity=5.0, face_value=1000.0)
        mtm = bond.price(self.result)
        # Prices at t=0 should be positive
        assert (mtm[:, 0] > 0).all()
