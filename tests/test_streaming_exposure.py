"""Tests for REGVMStepper and StreamingExposureEngine."""
from __future__ import annotations

import numpy as np
import pytest

from pyxva.core.paths import SimulationResult
from pyxva.exposure.csa import CSATerms, MarginRegime, IMModel
from pyxva.exposure.streaming.vm_stepper import REGVMStepper
from pyxva.exposure.streaming.engine import StreamingExposureEngine
from pyxva.pricing.rates.swap import InterestRateSwap
from pyxva.pricing.equity.vanilla_option import EuropeanOption
from pyxva.portfolio.trade import Trade


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rate_result(
    n_paths: int = 100,
    n_steps: int = 25,
    r0: float = 0.04,
    r_end: float = 0.04,
) -> SimulationResult:
    time_grid = np.linspace(0, 5.0, n_steps)
    # Fan of paths: r varies linearly from r0 to r_end with cross-sectional spread
    spread = np.linspace(-0.02, 0.02, n_paths)
    r = (
        np.linspace(r0, r_end, n_steps)[None, :]
        + spread[:, None] * np.linspace(0, 1, n_steps)[None, :]
    )
    paths = r[:, :, None]
    return SimulationResult(
        time_grid=time_grid,
        paths=paths,
        model_name="HW",
        factor_names=["r"],
    )


def _csa_zero_th() -> CSATerms:
    return CSATerms(
        counterparty_id="CP",
        margin_regime=MarginRegime.REGVM,
        threshold_party=0.0,
        threshold_counterparty=0.0,
        mta_party=0.0,
        mta_counterparty=0.0,
        mpor=10 / 252,
    )


# ---------------------------------------------------------------------------
# REGVMStepper
# ---------------------------------------------------------------------------

class TestREGVMStepper:
    def test_zero_threshold_fully_collateralised(self):
        """With zero threshold, post-margin exposure should be ~0 after first call."""
        csa = _csa_zero_th()
        stepper = REGVMStepper(csa, n_paths=5)
        net_mtm = np.array([1.0, 2.0, -1.0, 0.5, -0.5])
        exposure = stepper.step(net_mtm)
        # CE = max(V - CSB, 0): after fully collateralised VM call, CE=0 for positive V
        # For positive V: CSB should equal V, so CE = max(V-V,0) = 0
        np.testing.assert_allclose(exposure[net_mtm > 0], 0.0, atol=1e-12)

    def test_csb_updated_after_step(self):
        csa = _csa_zero_th()
        stepper = REGVMStepper(csa, n_paths=3)
        stepper.step(np.array([10.0, -5.0, 0.0]))
        csb = stepper.csb
        # CSB should be ~10 for first path (counterparty posted VM)
        assert csb[0] == pytest.approx(10.0, abs=1e-10)

    def test_mta_suppresses_small_calls(self):
        """Calls below MTA should not update CSB."""
        csa = CSATerms(
            counterparty_id="CP",
            threshold_party=0.0, threshold_counterparty=0.0,
            mta_party=5.0, mta_counterparty=5.0,
        )
        stepper = REGVMStepper(csa, n_paths=2)
        # MTM below MTA
        stepper.step(np.array([2.0, 3.0]))
        np.testing.assert_allclose(stepper.csb, 0.0, atol=1e-10)

    def test_threshold_reduces_csb(self):
        """Threshold reduces delivery amount."""
        csa = CSATerms(
            counterparty_id="CP",
            threshold_party=3.0, threshold_counterparty=3.0,
            mta_party=0.0, mta_counterparty=0.0,
        )
        stepper = REGVMStepper(csa, n_paths=1)
        stepper.step(np.array([10.0]))
        # Should post V - TH = 10 - 3 = 7
        assert stepper.csb[0] == pytest.approx(7.0, abs=1e-10)

    def test_reset(self):
        csa = _csa_zero_th()
        stepper = REGVMStepper(csa, n_paths=3)
        stepper.step(np.array([5.0, 5.0, 5.0]))
        stepper.reset()
        np.testing.assert_allclose(stepper.csb, 0.0, atol=1e-10)

    def test_rounding_delivery_ceil_return_floor(self):
        """ISDA 2016 VM CSA: delivery rounds UP (ceil), return rounds DOWN (floor).

        Regression test for the rounding direction fix. With rounding_nearest=1000:
        - A delivery shortfall of 1500 should round UP  to 2000, not nearest (2000 is same here).
        - A delivery shortfall of 1100 should round UP  to 2000, not nearest (1000).
        - A return  excess   of 1900 should round DOWN to 1000, not nearest (2000).
        """
        csa = CSATerms(
            counterparty_id="CP",
            threshold_party=0.0,
            threshold_counterparty=0.0,
            mta_party=0.0,
            mta_counterparty=0.0,
            rounding_nearest=1000.0,
        )
        # --- delivery rounds UP ---
        stepper = REGVMStepper(csa, n_paths=1)
        # shortfall = 1100 → ceil(1100/1000)*1000 = 2000, NOT round(1100/1000)*1000 = 1000
        stepper.step(np.array([1100.0]))
        assert stepper.csb[0] == pytest.approx(2000.0, abs=1e-10), (
            "delivery should round UP: 1100 → 2000 (ceil), not 1000 (nearest)"
        )

        # --- return rounds DOWN ---
        stepper2 = REGVMStepper(csa, n_paths=1)
        # Seed the CSB at 3000 first (simulate prior delivery)
        stepper2.step(np.array([3000.0]))
        assert stepper2.csb[0] == pytest.approx(3000.0, abs=1e-10)
        # Now MTM drops to 1100: return excess = 3000 - 1100 = 1900
        # floor(1900/1000)*1000 = 1000, NOT round = 2000
        stepper2.step(np.array([1100.0]))
        # After return of 1000: CSB = 3000 - 1000 = 2000
        assert stepper2.csb[0] == pytest.approx(2000.0, abs=1e-10), (
            "return should round DOWN: excess 1900 → return 1000 (floor), not 2000 (nearest)"
        )


# ---------------------------------------------------------------------------
# StreamingExposureEngine
# ---------------------------------------------------------------------------

class TestStreamingExposureEngine:
    def test_output_shapes(self):
        result = _rate_result(n_paths=50, n_steps=20)
        swap = InterestRateSwap(fixed_rate=0.04, maturity=5.0, notional=1e6)
        trades = [("swap1", swap)]
        csa = _csa_zero_th()
        engine = StreamingExposureEngine(trades, csa, confidence=0.95)
        out = engine.run(result)
        n = len(result.time_grid)
        assert out.ee_profile.shape == (n,)
        assert out.ene_profile.shape == (n,)
        assert out.pfe_profile.shape == (n,)
        assert out.ee_mpor_profile.shape == (n,)

    def test_ee_non_negative(self):
        result = _rate_result(n_paths=100, n_steps=25)
        swap = InterestRateSwap(fixed_rate=0.04, maturity=5.0, notional=1e6)
        engine = StreamingExposureEngine([("swap1", swap)], _csa_zero_th())
        out = engine.run(result)
        assert np.all(out.ee_profile >= -1e-10)

    def test_pfe_geq_ee(self):
        """At high confidence, PFE should be >= EE."""
        result = _rate_result(n_paths=200, n_steps=25)
        swap = InterestRateSwap(fixed_rate=0.04, maturity=5.0, notional=1e6)
        engine = StreamingExposureEngine([("swap1", swap)], _csa_zero_th(), confidence=0.99)
        out = engine.run(result)
        assert np.all(out.pfe_profile >= out.ee_profile - 1e-8)

    def test_scalars(self):
        result = _rate_result(n_paths=50, n_steps=15)
        swap = InterestRateSwap(fixed_rate=0.04, maturity=5.0, notional=1e6)
        engine = StreamingExposureEngine([("swap1", swap)], _csa_zero_th())
        out = engine.run(result)
        assert out.peak_ee == pytest.approx(float(np.max(out.ee_profile)))
        assert out.peak_pfe == pytest.approx(float(np.max(out.pfe_profile)))

    def test_two_trades_sum_mtm(self):
        """Two equal offsetting swaps should produce near-zero exposure."""
        result = _rate_result(n_paths=100, n_steps=15)
        pay = InterestRateSwap(fixed_rate=0.04, maturity=5.0, notional=1e6, payer=True)
        rec = InterestRateSwap(fixed_rate=0.04, maturity=5.0, notional=1e6, payer=False)
        trades = [("pay", pay), ("rec", rec)]
        engine = StreamingExposureEngine(trades, _csa_zero_th())
        out = engine.run(result)
        np.testing.assert_allclose(out.ee_profile, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# REGVMStepper property-based invariants
# ---------------------------------------------------------------------------

class TestREGVMStepperProperties:
    """Property tests over random MTM paths — not hand-crafted inputs."""

    def test_csb_tracks_mtm_exactly_zero_threshold_zero_mta(self):
        """With threshold=0 and mta=0, CSB must equal V(t) after every step."""
        rng = np.random.default_rng(42)
        csa = _csa_zero_th()
        n_paths, n_steps = 500, 60
        stepper = REGVMStepper(csa, n_paths=n_paths)

        for _ in range(n_steps):
            mtm = rng.normal(0, 100, size=n_paths)
            stepper.step(mtm)
            np.testing.assert_allclose(stepper.csb, mtm, atol=1e-10,
                err_msg="CSB must equal V(t) at every step when threshold=mta=0")

    def test_post_margin_ce_zero_for_positive_mtm(self):
        """With threshold=0 and mta=0, CE must be 0 for all positive-MTM paths."""
        rng = np.random.default_rng(99)
        csa = _csa_zero_th()
        n_paths, n_steps = 500, 40
        stepper = REGVMStepper(csa, n_paths=n_paths)

        for _ in range(n_steps):
            mtm = rng.normal(50, 30, size=n_paths)  # mostly positive
            exposure = stepper.step(mtm)
            # CE = max(V - CSB, 0); since CSB=V after step, CE must be 0
            positive = mtm > 0
            np.testing.assert_allclose(exposure[positive], 0.0, atol=1e-10,
                err_msg="Post-margin CE must be 0 for positive MTM paths when fully collateralised")

    def test_initial_csb_equals_net_ia(self):
        """Initial CSB (before any step) must equal ia_counterparty - ia_party."""
        ia_cpty, ia_party = 100.0, 30.0
        csa = CSATerms(
            counterparty_id="CP",
            margin_regime=MarginRegime.REGVM,
            threshold_party=0.0,
            threshold_counterparty=0.0,
            mta_party=0.0,
            mta_counterparty=0.0,
            ia_party=ia_party,
            ia_counterparty=ia_cpty,
        )
        stepper = REGVMStepper(csa, n_paths=50)
        expected_floor = ia_cpty - ia_party
        np.testing.assert_allclose(stepper.csb, expected_floor, atol=1e-12)


# ---------------------------------------------------------------------------
# Streaming vs batch EE parity
# ---------------------------------------------------------------------------

def test_streaming_batch_ee_parity():
    """StreamingExposureEngine and ISDAExposureCalculator must produce the same
    EE profile for a plain vanilla swap using a high threshold (no VM), ensuring
    both compute the raw uncollateralised E[max(V,0)] profile.

    Tolerance: 1bp (0.0001) on a unit-notional basis.
    """
    from pyxva.core.engine import MonteCarloEngine
    from pyxva.core.grid import TimeGrid
    from pyxva.models.rates.hull_white import HullWhite1F
    from pyxva.pricing.rates.swap import InterestRateSwap
    from pyxva.exposure.netting import NettingSet
    from pyxva.exposure.bilateral import ISDAExposureCalculator
    from pyxva.exposure.csa import CSATerms, MarginRegime, IMModel
    from pyxva.portfolio.trade import Trade

    N_PATHS = 3000
    SEED = 42

    model = HullWhite1F(a=0.10, sigma=0.01, r0=0.05)
    grid = TimeGrid.uniform(5.0, 60)
    engine = MonteCarloEngine(N_PATHS, seed=SEED)
    results = engine.run([model], grid)

    swap = InterestRateSwap(fixed_rate=0.05, maturity=5.0, notional=1.0, payer=True)

    # Use a very high threshold so no VM is ever called.
    # Both engines will therefore compute raw EE = E[max(V, 0)].
    HIGH_TH = 1e12
    csa = CSATerms(
        counterparty_id="cp",
        margin_regime=MarginRegime.REGVM,
        threshold_party=HIGH_TH,
        threshold_counterparty=HIGH_TH,
        mta_party=0.0,
        mta_counterparty=0.0,
        im_model=IMModel.SCHEDULE,
        mpor=0.0,
    )

    # --- Batch path (ISDAExposureCalculator) ---
    ns_batch = NettingSet("test")
    ns_batch.add_trade(Trade(id="swap", pricer=swap, model_name="HullWhite1F"))

    isda = ISDAExposureCalculator(netting_set=ns_batch, csa=csa)
    batch_out = isda.run(results, grid)
    ee_batch = batch_out["ee"]  # raw E[max(V,0)]

    # --- Streaming path ---
    stream_csa = CSATerms(
        counterparty_id="cp",
        margin_regime=MarginRegime.REGVM,
        threshold_party=HIGH_TH,
        threshold_counterparty=HIGH_TH,
        mta_party=0.0,
        mta_counterparty=0.0,
        im_model=IMModel.SCHEDULE,
        mpor=0.0,
    )
    # StreamingExposureEngine takes (trade_id, pricer) tuples and a single SimulationResult
    trades = [("swap", swap)]
    streaming = StreamingExposureEngine(trades, stream_csa, confidence=0.95)
    stream_out = streaming.run(results["HullWhite1F"])
    ee_stream = stream_out.ee_profile  # post-VM EE ≈ raw EE when threshold=∞

    # Should match within 1bp on unit-notional basis
    np.testing.assert_allclose(
        ee_batch, ee_stream, atol=1e-4,
        err_msg="Streaming and batch EE profiles diverge by more than 1bp",
    )


# ---------------------------------------------------------------------------
# Multi-model netting sets
# ---------------------------------------------------------------------------

def _equity_result(
    n_paths: int = 100,
    n_steps: int = 25,
    s0: float = 100.0,
    sigma: float = 0.20,
) -> SimulationResult:
    """Synthetic GBM-like equity SimulationResult."""
    time_grid = np.linspace(0, 5.0, n_steps)
    rng = np.random.default_rng(77)
    dt = np.diff(time_grid, prepend=0.0)
    log_s = np.log(s0) + np.cumsum(
        (0.04 - 0.5 * sigma ** 2) * dt[None, :]
        + sigma * np.sqrt(dt)[None, :] * rng.standard_normal((n_paths, n_steps)),
        axis=1,
    )
    paths = np.exp(log_s)[:, :, None]
    return SimulationResult(
        time_grid=time_grid,
        paths=paths,
        model_name="GBM",
        factor_names=["S"],
    )


class TestStreamingMultiModel:
    """Multi-model netting sets: Trade objects + dict[str, SimulationResult]."""

    def test_single_result_dict_matches_single_result(self):
        """Passing {model_name: result} and bare result must produce identical output."""
        rate_res = _rate_result(n_paths=50, n_steps=20)
        swap = InterestRateSwap(fixed_rate=0.04, maturity=5.0, notional=1e6)
        csa = _csa_zero_th()

        # Legacy: tuple + bare result
        engine_legacy = StreamingExposureEngine([("swap1", swap)], csa)
        out_legacy = engine_legacy.run(rate_res)

        # New: Trade + dict
        t = Trade(id="swap1", pricer=swap, model_name="HW")
        engine_new = StreamingExposureEngine([t], csa)
        out_new = engine_new.run({"HW": rate_res})

        np.testing.assert_allclose(out_legacy.ee_profile, out_new.ee_profile, atol=1e-12)

    def test_two_models_different_results(self):
        """Netting set with IR swap + equity option should run without error."""
        rate_res = _rate_result(n_paths=100, n_steps=20)
        eq_res = _equity_result(n_paths=100, n_steps=20)

        swap = InterestRateSwap(fixed_rate=0.04, maturity=5.0, notional=1e5)
        option = EuropeanOption(strike=100.0, expiry=2.0, sigma=0.20, risk_free_rate=0.04)

        t_swap = Trade(id="swap1", pricer=swap, model_name="HW")
        t_opt = Trade(id="opt1", pricer=option, model_name="GBM")

        csa = _csa_zero_th()
        engine = StreamingExposureEngine([t_swap, t_opt], csa)
        out = engine.run({"HW": rate_res, "GBM": eq_res})

        assert out.ee_profile.shape == (20,)
        assert np.all(out.ee_profile >= -1e-10)
        assert out.peak_ee >= 0.0

    def test_mismatched_time_grids_raises(self):
        """Dict results with different time grids must raise ValueError."""
        rate_res = _rate_result(n_paths=50, n_steps=20)
        eq_res_diff = _equity_result(n_paths=50, n_steps=15)  # different n_steps

        swap = InterestRateSwap(fixed_rate=0.04, maturity=5.0, notional=1e5)
        option = EuropeanOption(strike=100.0, expiry=2.0, sigma=0.20, risk_free_rate=0.04)
        t_swap = Trade(id="s", pricer=swap, model_name="HW")
        t_opt = Trade(id="o", pricer=option, model_name="GBM")

        engine = StreamingExposureEngine([t_swap, t_opt], _csa_zero_th())
        with pytest.raises(ValueError, match="same time grid"):
            engine.run({"HW": rate_res, "GBM": eq_res_diff})


# ---------------------------------------------------------------------------
# Proper MPOR: CSB ring buffer
# ---------------------------------------------------------------------------

class TestProperMPOR:
    def test_mpor_profile_non_negative(self):
        """ee_mpor_profile must be non-negative."""
        result = _rate_result(n_paths=200, n_steps=30)
        swap = InterestRateSwap(fixed_rate=0.04, maturity=5.0, notional=1e6)
        engine = StreamingExposureEngine([("s", swap)], _csa_zero_th())
        out = engine.run(result, mpor_steps=3)
        assert np.all(out.ee_mpor_profile >= -1e-12)

    def test_mpor_zero_for_first_steps(self):
        """For i < mpor_steps the ee_mpor should be 0 (no history yet)."""
        result = _rate_result(n_paths=100, n_steps=20)
        swap = InterestRateSwap(fixed_rate=0.04, maturity=5.0, notional=1e6)
        mpor_steps = 5
        engine = StreamingExposureEngine([("s", swap)], _csa_zero_th())
        out = engine.run(result, mpor_steps=mpor_steps)
        # First mpor_steps entries should be zero (no ring-buffer history)
        np.testing.assert_allclose(
            out.ee_mpor_profile[:mpor_steps], 0.0, atol=1e-12,
            err_msg="ee_mpor should be 0 for the first mpor_steps steps (no CSB history)"
        )

    def test_mpor_geq_ee_for_well_collateralised_netting_set(self):
        """Under zero-threshold CSA, ee_mpor(t) >= ee_profile(t) for t >= mpor_steps.

        After mpor periods the lagged CSB is lower than the current CSB
        (collateral hasn't been refreshed for mpor steps), so the MPOR exposure
        should weakly exceed the standard post-margin exposure.
        """
        result = _rate_result(n_paths=500, n_steps=40, r0=0.03, r_end=0.06)
        swap = InterestRateSwap(fixed_rate=0.03, maturity=5.0, notional=1e6)
        mpor_steps = 3
        engine = StreamingExposureEngine([("s", swap)], _csa_zero_th())
        out = engine.run(result, mpor_steps=mpor_steps)
        # After mpor_steps, ee_mpor >= ee_profile (up to floating point)
        np.testing.assert_array_less(
            out.ee_profile[mpor_steps:] - 1e-4,
            out.ee_mpor_profile[mpor_steps:],
        )
