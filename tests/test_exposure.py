"""Tests for exposure metrics and netting."""
import numpy as np
import pytest

from risk_analytics.core import MonteCarloEngine, TimeGrid
from risk_analytics.models import GeometricBrownianMotion, HullWhite1F
from risk_analytics.pricing import EuropeanOption, InterestRateSwap
from risk_analytics.exposure import ExposureCalculator, NettingSet

N_PATHS = 2000
SEED = 42


def make_equity_results(n_paths=N_PATHS):
    model = GeometricBrownianMotion(S0=100.0, mu=0.05, sigma=0.20)
    grid = TimeGrid.uniform(1.0, 52)
    engine = MonteCarloEngine(n_paths, seed=SEED)
    results = engine.run([model], grid)
    return results, grid


def make_rates_results(n_paths=N_PATHS):
    model = HullWhite1F(a=0.1, sigma=0.01, r0=0.05)
    grid = TimeGrid.uniform(5.0, 60)
    engine = MonteCarloEngine(n_paths, seed=SEED)
    results = engine.run([model], grid)
    return results, grid


class TestExposureCalculator:
    def setup_method(self):
        results, self.grid = make_equity_results()
        option = EuropeanOption(strike=100, expiry=1.0, sigma=0.20, risk_free_rate=0.03)
        self.mtm = option.price(results["GBM"])
        self.calc = ExposureCalculator()

    def test_exposure_profile_non_negative(self):
        ep = self.calc.exposure_profile(self.mtm)
        assert np.all(ep >= 0)

    def test_expected_exposure_shape(self):
        ee = self.calc.expected_exposure(self.mtm)
        assert ee.shape == (len(self.grid),)

    def test_pse_scalar(self):
        pse = self.calc.pse(self.mtm)
        assert isinstance(pse, float)
        assert pse >= 0

    def test_epe_scalar(self):
        epe = self.calc.epe(self.mtm, self.grid)
        assert isinstance(epe, float)
        assert epe >= 0

    def test_pfe_shape_and_monotone(self):
        pfe_90 = self.calc.pfe(self.mtm, 0.90)
        pfe_95 = self.calc.pfe(self.mtm, 0.95)
        assert pfe_90.shape == (len(self.grid),)
        # 95th percentile should be >= 90th
        assert np.all(pfe_95 >= pfe_90 - 1e-8)

    def test_pse_geq_epe(self):
        """PSE (peak) should be >= EPE (average)."""
        assert self.calc.pse(self.mtm) >= self.calc.epe(self.mtm, self.grid)

    def test_summary_keys(self):
        summary = self.calc.exposure_summary(self.mtm, self.grid)
        for key in ("ee_profile", "pfe_profile", "pse", "epe", "confidence"):
            assert key in summary

    def test_negative_mtm_gives_zero_exposure(self):
        """All-negative MTM should produce zero exposure."""
        mtm_neg = -np.abs(self.mtm)
        ep = self.calc.exposure_profile(mtm_neg)
        assert np.all(ep == 0.0)


class TestNettingSet:
    def setup_method(self):
        self.results, self.grid = make_rates_results()

    def test_single_trade_netting(self):
        """Single trade: netted MTM should equal trade MTM."""
        swap = InterestRateSwap(fixed_rate=0.05, maturity=5.0, notional=1e6, payer=True)
        ns = NettingSet("test")
        ns.add_trade("swap1", swap)
        net = ns.net_mtm(self.results)
        direct = swap.price(self.results["HullWhite1F"])
        assert np.allclose(net, direct)

    def test_netting_benefit(self):
        """Sum of individual positive exposures >= netted positive exposure."""
        payer = InterestRateSwap(fixed_rate=0.05, maturity=5.0, notional=1e6, payer=True)
        receiver = InterestRateSwap(fixed_rate=0.03, maturity=5.0, notional=1e6, payer=False)

        ns = NettingSet("test")
        ns.add_trade("payer", payer)
        ns.add_trade("receiver", receiver)

        calc = ExposureCalculator()
        r = self.results["HullWhite1F"]

        mtm_payer = payer.price(r)
        mtm_receiver = receiver.price(r)

        ee_sum = (
            calc.expected_exposure(mtm_payer) + calc.expected_exposure(mtm_receiver)
        )
        net_mtm = ns.net_mtm(self.results)
        ee_net = calc.expected_exposure(net_mtm)

        # Netting benefit: gross >= net at every time step
        assert np.all(ee_sum >= ee_net - 1e-6)

    def test_trade_ids(self):
        ns = NettingSet("ns1")
        swap = InterestRateSwap(fixed_rate=0.05, maturity=5.0, notional=1e6)
        ns.add_trade("trade_A", swap)
        assert ns.trade_ids == ["trade_A"]

    def test_empty_netting_set_raises(self):
        ns = NettingSet("empty")
        with pytest.raises(ValueError):
            ns.net_mtm(self.results)

    def test_exposure_summary_keys(self):
        swap = InterestRateSwap(fixed_rate=0.05, maturity=5.0, notional=1e6)
        ns = NettingSet("test")
        ns.add_trade("swap1", swap)
        summary = ns.exposure(self.results, self.grid)
        for key in ("ee_profile", "pfe_profile", "pse", "epe", "net_mtm"):
            assert key in summary
