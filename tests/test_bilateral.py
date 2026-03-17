"""Tests for bilateral ISDA exposure calculator."""
import numpy as np
import pytest

from risk_analytics.core import MonteCarloEngine, TimeGrid
from risk_analytics.models import HullWhite1F, GeometricBrownianMotion
from risk_analytics.pricing import InterestRateSwap, EuropeanOption
from risk_analytics.exposure import (
    CSATerms,
    MarginRegime,
    IMModel,
    CollateralAccount,
    HaircutSchedule,
    REGVMEngine,
    REGIMEngine,
    SimmSensitivities,
    SimmCalculator,
    BilateralExposureCalculator,
    ISDAExposureCalculator,
    NettingSet,
)

N_PATHS = 2000
SEED = 42


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def rates_setup():
    model = HullWhite1F(a=0.1, sigma=0.01, r0=0.05)
    grid = TimeGrid.uniform(5.0, 60)
    engine = MonteCarloEngine(N_PATHS, seed=SEED)
    results = engine.run([model], grid)
    return results, grid


def equity_setup():
    model = GeometricBrownianMotion(S0=100.0, mu=0.05, sigma=0.20)
    grid = TimeGrid.uniform(2.0, 52)
    engine = MonteCarloEngine(N_PATHS, seed=SEED)
    results = engine.run([model], grid)
    return results, grid


# ---------------------------------------------------------------------------
# CSATerms
# ---------------------------------------------------------------------------

class TestCSATerms:
    def test_regvm_standard(self):
        csa = CSATerms.regvm_standard("CP1", mta=0.5)
        assert csa.threshold_party == 0.0
        assert csa.threshold_counterparty == 0.0
        assert csa.mta_party == 0.5
        assert csa.margin_regime == MarginRegime.REGVM

    def test_legacy_bilateral(self):
        csa = CSATerms.legacy_bilateral("CP2", threshold=5.0, mta=0.5)
        assert csa.threshold_party == 5.0
        assert csa.threshold_counterparty == 5.0
        assert csa.margin_regime == MarginRegime.LEGACY

    def test_cleared(self):
        csa = CSATerms.cleared("CCP1")
        assert csa.mpor == pytest.approx(5 / 252)
        assert csa.rehypothecatable_vm is False

    def test_default_eligible_collateral(self):
        csa = CSATerms()
        assert "CASH_USD" in csa.eligible_collateral
        assert csa.eligible_collateral["CASH_USD"] == 0.0


# ---------------------------------------------------------------------------
# HaircutSchedule & CollateralAccount
# ---------------------------------------------------------------------------

class TestHaircutSchedule:
    def test_cash_no_haircut(self):
        hs = HaircutSchedule()
        assert hs.apply("CASH_USD", 100.0) == pytest.approx(100.0)

    def test_bond_haircut(self):
        hs = HaircutSchedule(haircuts={"BOND": 0.05})
        assert hs.apply("BOND", 200.0) == pytest.approx(190.0)

    def test_unknown_asset_default_haircut(self):
        hs = HaircutSchedule(default_haircut=1.0)
        assert hs.apply("UNKNOWN", 500.0) == pytest.approx(0.0)


class TestCollateralAccount:
    def test_receive_vm_reduces_exposure(self):
        acc = CollateralAccount()
        acc.receive_vm(np.array([100.0, 200.0]))
        net = acc.net_vm_value()
        assert np.allclose(net, [100.0, 200.0])

    def test_post_vm_increases_balance_magnitude(self):
        acc = CollateralAccount()
        acc.post_vm(np.array([50.0]))
        net = acc.net_vm_value()
        assert np.allclose(net, [-50.0])

    def test_net_zero_on_equal_post_receive(self):
        acc = CollateralAccount()
        acc.receive_vm(np.array([100.0]))
        acc.post_vm(np.array([100.0]))
        assert np.allclose(acc.net_vm_value(), [0.0])

    def test_haircut_applied_on_post(self):
        hs = HaircutSchedule(haircuts={"BOND": 0.10})
        acc = CollateralAccount(haircut_schedule=hs)
        acc.receive_vm(100.0, asset_type="BOND")
        assert acc.net_vm_value() == pytest.approx(90.0)

    def test_im_segregated_not_in_vm(self):
        acc = CollateralAccount()
        acc.receive_vm(100.0)
        acc.receive_im(200.0, segregated=True)
        # net_collateral_value without IM = just VM
        assert acc.net_collateral_value(include_im=False) == pytest.approx(100.0)
        # with IM included
        assert acc.net_collateral_value(include_im=True) == pytest.approx(300.0)

    def test_reset_clears_entries(self):
        acc = CollateralAccount()
        acc.receive_vm(100.0)
        acc.reset()
        assert acc.net_vm_value() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# REGVMEngine
# ---------------------------------------------------------------------------

class TestREGVMEngine:
    def setup_method(self):
        self.csa = CSATerms.regvm_standard("CP", mta=0.0)
        self.vm = REGVMEngine(self.csa)
        # Simple 3-path, 5-step MTM array
        self.mtm = np.array([
            [0,  5, 10, -3,  2],
            [0, -2, -8,  4,  1],
            [0,  0,  0,  0,  0],
        ], dtype=float)

    def test_csb_zero_threshold_positive(self):
        csb = self.vm.credit_support_balance(self.mtm)
        # With zero threshold: CSB = mtm (all positive = received, all negative = posted)
        assert np.allclose(csb, self.mtm)

    def test_csb_with_threshold(self):
        csa = CSATerms.legacy_bilateral("CP", threshold=3.0, mta=0.0)
        vm = REGVMEngine(csa)
        csb = vm.credit_support_balance(self.mtm)
        # V=5: received = max(5-3,0)=2, posted=max(-5-3,0)=0 → CSB=2
        assert csb[0, 1] == pytest.approx(2.0)
        # V=-3: received=0, posted=max(3-3,0)=0 → CSB=0
        assert csb[0, 3] == pytest.approx(0.0)

    def test_ia_offsets_csb(self):
        csa = CSATerms(ia_counterparty=1.0, ia_party=0.0)
        vm = REGVMEngine(csa)
        mtm = np.zeros((2, 3))
        csb = vm.credit_support_balance(mtm)
        # With zero mtm, CSB = ia_counterparty - ia_party = 1.0
        assert np.allclose(csb, 1.0)

    def test_lagged_csb_at_t0_is_ia(self):
        csa = CSATerms(ia_counterparty=2.0, mpor=10 / 252)
        vm = REGVMEngine(csa)
        mtm = np.ones((5, 61)) * 10.0
        grid = TimeGrid.uniform(5.0, 60)
        lagged = vm.lagged_csb(mtm, grid)
        # At t[0], lagged time is clamped to t[0]: CSB at t=0 = max(10-0,0) + IA = 10+2=12
        assert lagged[:, 0].mean() == pytest.approx(12.0, abs=0.5)

    def test_mta_zeros_small_calls(self):
        csa = CSATerms(mta_party=5.0, mta_counterparty=5.0)
        vm = REGVMEngine(csa)
        mtm = np.array([[3.0, -3.0, 10.0]])  # 3 < MTA, so no call
        call = vm.vm_call(mtm)
        assert call[0, 0] == pytest.approx(0.0)   # 3 < MTA=5
        assert call[0, 2] == pytest.approx(10.0)   # 10 > MTA=5

    def test_collateralised_exposure_less_than_uncoll(self):
        results, grid = rates_setup()
        ns = NettingSet("test")
        ns.add_trade("swap", InterestRateSwap(0.05, 5.0, 1e6))
        net_mtm = ns.net_mtm(results)

        csa = CSATerms.regvm_standard("CP")
        vm = REGVMEngine(csa)
        coll_exp = vm.collateralised_exposure(net_mtm, grid)
        uncoll_exp = np.maximum(net_mtm, 0.0)

        ee_coll = coll_exp.mean(axis=0)
        ee_uncoll = uncoll_exp.mean(axis=0)
        # Collateralised EE should generally be ≤ uncollateralised
        assert ee_coll.mean() <= ee_uncoll.mean() + 1e-3


# ---------------------------------------------------------------------------
# REGIMEngine
# ---------------------------------------------------------------------------

class TestREGIMEngine:
    def setup_method(self):
        self.csa = CSATerms(im_model=IMModel.SCHEDULE)
        self.engine = REGIMEngine(self.csa)

    def test_schedule_im_positive(self):
        trades = [
            {"asset_class": "IR", "gross_notional": 1e6, "maturity": 5.0, "net_replacement_cost": 0.0}
        ]
        im = self.engine.schedule_im(trades)
        assert float(im) > 0

    def test_schedule_im_zero_notional(self):
        trades = [{"asset_class": "IR", "gross_notional": 0.0, "maturity": 5.0}]
        im = self.engine.schedule_im(trades)
        assert float(im) == pytest.approx(0.0)

    def test_schedule_ngr_reduces_im(self):
        """Higher NRC / GrossIM ratio (NGR) → higher IM factor.

        IR 5yr bucket → weight = 0.04 → GrossIM = 1e6 * 0.04 = 40_000.
        - NRC = 0       → NGR = 0   → IM = 0.4 × GrossIM = 16_000
        - NRC = GrossIM → NGR = 1.0 → IM = 1.0 × GrossIM = 40_000
        """
        gross = 1_000_000.0
        gross_im = gross * 0.04   # IR 5yr "5_plus" bucket weight

        # NRC = 0: IM = 0.4 * gross_im
        trades_zero = [{"asset_class": "IR", "gross_notional": gross, "maturity": 5.0, "net_replacement_cost": 0.0}]
        im_low = float(self.engine.schedule_im(trades_zero))

        # NRC = gross_im → NGR = 1.0 → IM = gross_im
        trades_high = [{"asset_class": "IR", "gross_notional": gross, "maturity": 5.0, "net_replacement_cost": gross_im}]
        im_high = float(self.engine.schedule_im(trades_high))

        assert im_high > im_low
        assert im_low == pytest.approx(0.4 * gross_im, rel=1e-6)
        assert im_high == pytest.approx(1.0 * gross_im, rel=1e-6)

    def test_broadcast_to_shape(self):
        trades = [{"asset_class": "FX", "gross_notional": 1e6, "maturity": 1.0}]
        im = self.engine.schedule_im(trades, shape=(N_PATHS, 61))
        assert im.shape == (N_PATHS, 61)


# ---------------------------------------------------------------------------
# SimmCalculator
# ---------------------------------------------------------------------------

class TestSimmCalculator:
    def setup_method(self):
        self.calc = SimmCalculator()

    def test_ir_margin_single_currency(self):
        sens = {"USD": {"1y": 1000.0, "5y": 2000.0}}
        im = float(self.calc.ir_margin(sens))
        assert im > 0

    def test_equity_margin_positive(self):
        sens = {1: 5000.0, 2: 3000.0}
        im = float(self.calc.equity_margin(sens))
        assert im > 0

    def test_fx_margin_positive(self):
        sens = {"EUR": 1e6, "JPY": 2e6}
        im = float(self.calc.fx_margin(sens))
        assert im > 0

    def test_total_im_cross_class(self):
        s = SimmSensitivities(
            ir={"USD": {"5y": 1000}},
            equity={1: 2000},
            fx={"EUR": 5e5},
        )
        total = float(self.calc.total_im(s))
        # Cross-class aggregation: total > individual class IMs
        ir_only = float(self.calc.ir_margin(s.ir))
        eq_only = float(self.calc.equity_margin(s.equity))
        assert total >= max(ir_only, eq_only)

    def test_empty_sensitivities_returns_zero(self):
        s = SimmSensitivities()
        assert float(self.calc.total_im(s)) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# BilateralExposureCalculator
# ---------------------------------------------------------------------------

class TestBilateralExposureCalculator:
    def setup_method(self):
        results, self.grid = rates_setup()
        ns = NettingSet("test")
        ns.add_trade("swap", InterestRateSwap(0.05, 5.0, 1e6))
        self.mtm = ns.net_mtm(results)
        self.calc = BilateralExposureCalculator()

    def test_ene_non_positive(self):
        """ENE ≤ 0 by definition."""
        ene = self.calc.ene(self.mtm)
        assert np.all(ene <= 0 + 1e-10)

    def test_ee_plus_ene_spans_full_distribution(self):
        """EE(t) + |ENE(t)| ≤ E[|V(t)|] (upper bound check)."""
        ee = self.calc.expected_exposure(self.mtm)
        ene = np.abs(self.calc.ene(self.mtm))
        abs_mean = np.abs(self.mtm).mean(axis=0)
        assert np.all(ee + ene <= abs_mean + 1e-6)

    def test_eepe_scalar(self):
        eepe = self.calc.eepe(self.mtm, self.grid)
        assert isinstance(eepe, float)
        assert eepe >= 0

    def test_eepe_geq_epe(self):
        """EEPE ≥ EPE because effective EE is non-decreasing (monotone envelope)."""
        eepe = self.calc.eepe(self.mtm, self.grid)
        epe = self.calc.epe(self.mtm, self.grid)
        assert eepe >= epe - 1e-6

    def test_mpor_adjusted_ee_further_right(self):
        """MPOR-shifted EE[t] = EE[t+MPOR]: should be ≥ EE[t] if EE is increasing."""
        ee = self.calc.expected_exposure(self.mtm)
        ee_mpor = self.calc.mpor_adjusted_ee(self.mtm, self.grid, mpor=10 / 252)
        # At t=0, mpor-shifted looks further out; generally different from ee
        assert ee_mpor.shape == ee.shape

    def test_collateralised_ee_leq_uncoll(self):
        csa = CSATerms.regvm_standard("CP")
        vm = REGVMEngine(csa)
        lagged = vm.lagged_csb(self.mtm, self.grid)
        ee_coll = self.calc.collateralised_ee(self.mtm, lagged)
        ee_uncoll = self.calc.expected_exposure(self.mtm)
        # After removing collateral, exposure should be ≤ uncollateralised
        assert ee_coll.mean() <= ee_uncoll.mean() + 1e-3

    def test_cva_positive(self):
        cva = self.calc.cva_approx(self.mtm, self.grid, hazard_rate=0.01, lgd=0.6)
        assert cva >= 0

    def test_dva_positive(self):
        dva = self.calc.dva_approx(self.mtm, self.grid, own_hazard_rate=0.005, own_lgd=0.6)
        assert dva >= 0

    def test_bilateral_cva_keys(self):
        out = self.calc.bilateral_cva(self.mtm, self.grid, cp_hazard_rate=0.01, own_hazard_rate=0.005)
        for k in ("cva", "dva", "bcva"):
            assert k in out
        assert out["bcva"] == pytest.approx(out["cva"] - out["dva"])

    def test_bilateral_summary_keys(self):
        summary = self.calc.bilateral_summary(
            self.mtm, self.grid,
            cp_hazard_rate=0.01, own_hazard_rate=0.005
        )
        for k in ("ee", "ene", "pfe", "pse", "epe", "eepe", "ee_mpor", "cva", "dva", "bcva"):
            assert k in summary


# ---------------------------------------------------------------------------
# ISDAExposureCalculator — integration
# ---------------------------------------------------------------------------

class TestISDAExposureCalculator:
    def setup_method(self):
        self.results, self.grid = rates_setup()
        self.ns = NettingSet("test_ns")
        self.ns.add_trade("payer_swap", InterestRateSwap(0.05, 5.0, notional=1e6, payer=True))
        self.ns.add_trade("receiver_swap", InterestRateSwap(0.03, 5.0, notional=5e5, payer=False))
        self.csa = CSATerms.regvm_standard("CP_BANK", mta=0.1)

    def test_run_returns_all_keys(self):
        calc = ISDAExposureCalculator(self.ns, self.csa)
        out = calc.run(self.results, self.grid)
        for k in ("net_mtm", "csb", "lagged_csb", "ee", "ene", "pfe", "ee_coll", "epe", "eepe"):
            assert k in out

    def test_net_mtm_shape(self):
        calc = ISDAExposureCalculator(self.ns, self.csa)
        out = calc.run(self.results, self.grid)
        assert out["net_mtm"].shape == (N_PATHS, len(self.grid))

    def test_ee_coll_shape_and_non_negative(self):
        """Collateralised EE has correct shape and is non-negative."""
        calc = ISDAExposureCalculator(self.ns, self.csa)
        out = calc.run(self.results, self.grid)
        assert out["ee_coll"].shape == (len(self.grid),)
        assert np.all(out["ee_coll"] >= 0.0)

    def test_with_schedule_im(self):
        im_engine = REGIMEngine(CSATerms(im_model=IMModel.SCHEDULE))
        calc = ISDAExposureCalculator(self.ns, self.csa, im_engine=im_engine)
        trades = [
            {"asset_class": "IR", "gross_notional": 1e6, "maturity": 5.0, "net_replacement_cost": 0.0}
        ]
        out = calc.run(self.results, self.grid, im_trades=trades)
        assert out["im"] is not None
        assert out["im"].shape == (N_PATHS, len(self.grid))

    def test_with_cva_dva(self):
        calc = ISDAExposureCalculator(self.ns, self.csa)
        out = calc.run(
            self.results, self.grid,
            cp_hazard_rate=0.02,
            own_hazard_rate=0.01,
        )
        assert "cva" in out and "dva" in out
        assert out["cva"] >= 0
        assert out["bcva"] == pytest.approx(out["cva"] - out["dva"])

    def test_netting_benefit_vs_gross(self):
        """Net exposure should be ≤ sum of individual positive exposures."""
        calc = ISDAExposureCalculator(self.ns, self.csa)
        out = calc.run(self.results, self.grid)

        # Individual uncollateralised EE
        r = self.results["HullWhite1F"]
        payer_mtm = InterestRateSwap(0.05, 5.0, 1e6, payer=True).price(r)
        recv_mtm = InterestRateSwap(0.03, 5.0, 5e5, payer=False).price(r)

        from risk_analytics.exposure import ExposureCalculator
        ec = ExposureCalculator()
        ee_gross = ec.expected_exposure(payer_mtm) + ec.expected_exposure(recv_mtm)
        ee_net = out["ee"]
        assert ee_net.mean() <= ee_gross.mean() + 1e-3

    def test_legacy_csa_larger_threshold_means_less_collateral(self):
        """Legacy CSA: with a very large threshold, CSB is effectively zero → EE ≈ uncoll EE."""
        ns = NettingSet("ns")
        ns.add_trade("swap", InterestRateSwap(0.05, 5.0, 1e6, payer=True))

        # REGVM: zero threshold → CSB tracks full MTM
        csa_regvm = CSATerms.regvm_standard("CP", mta=0.0)
        vm_regvm = REGVMEngine(csa_regvm)
        net_mtm = ns.net_mtm(self.results)
        csb_regvm = vm_regvm.credit_support_balance(net_mtm)

        # Legacy with very large threshold: CSB ≈ 0 for moderate MTM swings
        csa_large_th = CSATerms.legacy_bilateral("CP", threshold=1e9, mta=0.0)
        vm_large = REGVMEngine(csa_large_th)
        csb_large = vm_large.credit_support_balance(net_mtm)

        # Zero-threshold REGVM should have larger (or equal) mean CSB magnitude
        assert np.abs(csb_regvm).mean() >= np.abs(csb_large).mean() - 1e-3
