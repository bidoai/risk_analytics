"""Tests for YieldCurve interpolation and HullWhite1F integration."""
from __future__ import annotations

import numpy as np
import pytest

from risk_analytics.core import YieldCurve, Interpolation
from risk_analytics.models import HullWhite1F
from risk_analytics.core.grid import TimeGrid


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TENORS = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
ZERO_RATES = np.array([0.035, 0.037, 0.040, 0.043, 0.045, 0.048, 0.051, 0.054])


@pytest.fixture(params=["linear", "log_linear", "cubic_spline"])
def curve(request):
    return YieldCurve(TENORS, ZERO_RATES, interpolation=request.param)


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_string_interpolation_accepted(self):
        c = YieldCurve(TENORS, ZERO_RATES, interpolation="cubic_spline")
        assert c.interpolation == Interpolation.CUBIC_SPLINE

    def test_enum_interpolation_accepted(self):
        c = YieldCurve(TENORS, ZERO_RATES, interpolation=Interpolation.LOG_LINEAR)
        assert c.interpolation == Interpolation.LOG_LINEAR

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            YieldCurve([1, 2, 3], [0.04, 0.05])

    def test_non_increasing_tenors_raises(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            YieldCurve([1.0, 0.5, 2.0], [0.04, 0.05, 0.06])

    def test_duplicate_tenors_raises(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            YieldCurve([1.0, 1.0, 2.0], [0.04, 0.04, 0.06])

    def test_negative_tenor_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            YieldCurve([-0.5, 1.0, 2.0], [0.03, 0.04, 0.05])

    def test_single_tenor_raises(self):
        with pytest.raises(ValueError, match="At least two"):
            YieldCurve([1.0], [0.04])

    def test_repr_contains_interpolation(self):
        c = YieldCurve(TENORS, ZERO_RATES, "linear")
        assert "linear" in repr(c)


# ---------------------------------------------------------------------------
# zero_rate
# ---------------------------------------------------------------------------

class TestZeroRate:
    def test_exact_at_knots(self, curve):
        """All interpolation methods must reproduce exact knot values."""
        for t, z in zip(TENORS, ZERO_RATES):
            assert curve.zero_rate(t) == pytest.approx(z, rel=1e-6)

    def test_scalar_input_returns_scalar(self, curve):
        result = curve.zero_rate(1.0)
        assert isinstance(result, float)

    def test_array_input_returns_array(self, curve):
        ts = np.array([0.5, 1.0, 2.0])
        result = curve.zero_rate(ts)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)

    def test_flat_left_extrapolation(self, curve):
        """Below first tenor → hold first zero rate."""
        assert curve.zero_rate(0.01) == pytest.approx(curve.zero_rate(TENORS[0]), rel=1e-4)

    def test_flat_right_extrapolation(self, curve):
        """Beyond last tenor → hold last zero rate."""
        assert curve.zero_rate(20.0) == pytest.approx(ZERO_RATES[-1], rel=1e-4)

    def test_upward_sloping(self, curve):
        """For an upward-sloping curve, z(5) > z(1)."""
        assert curve.zero_rate(5.0) > curve.zero_rate(1.0)

    def test_monotone_between_knots(self, curve):
        ts = np.linspace(TENORS[0], TENORS[-1], 200)
        zs = curve.zero_rate(ts)
        assert np.all(np.diff(zs) >= -1e-8), "zero rates should be non-decreasing"


# ---------------------------------------------------------------------------
# discount_factor
# ---------------------------------------------------------------------------

class TestDiscountFactor:
    def test_at_zero(self, curve):
        """P(0, 0) = 1 for all methods."""
        assert curve.discount_factor(0.0) == pytest.approx(1.0, rel=1e-8)

    def test_always_positive(self, curve):
        ts = np.linspace(0.0, 15.0, 50)
        dfs = curve.discount_factor(ts)
        assert np.all(dfs > 0)

    def test_strictly_decreasing(self, curve):
        """Upward-sloping curve → discount factors decrease with maturity."""
        ts = np.linspace(0.25, 10.0, 40)
        dfs = curve.discount_factor(ts)
        assert np.all(np.diff(dfs) < 0)

    def test_consistent_with_zero_rate(self, curve):
        """P(0, t) = exp(-z(t) * t) at all knots."""
        for t, z in zip(TENORS, ZERO_RATES):
            expected = np.exp(-z * t)
            assert curve.discount_factor(t) == pytest.approx(expected, rel=1e-6)

    def test_scalar_returns_scalar(self, curve):
        result = curve.discount_factor(2.0)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# forward_rate
# ---------------------------------------------------------------------------

class TestForwardRate:
    def test_forward_above_short_rate_upward_curve(self, curve):
        """On an upward sloping curve, F(1,2) > z(1)."""
        assert curve.forward_rate(1.0, 2.0) > curve.zero_rate(1.0)

    def test_flat_curve_forward_equals_zero_rate(self):
        """On a flat curve, forward rate = zero rate everywhere."""
        flat = YieldCurve([0.5, 1.0, 5.0], [0.05, 0.05, 0.05])
        assert flat.forward_rate(1.0, 3.0) == pytest.approx(0.05, rel=1e-5)

    def test_t2_must_exceed_t1(self, curve):
        with pytest.raises(ValueError, match="t2 must be strictly greater"):
            curve.forward_rate(2.0, 1.0)

    def test_consistent_with_discount_factors(self, curve):
        """F(t1, t2) = -log(P(0,t2)/P(0,t1)) / (t2-t1)."""
        t1, t2 = 1.0, 3.0
        expected = -np.log(curve.discount_factor(t2) / curve.discount_factor(t1)) / (t2 - t1)
        assert curve.forward_rate(t1, t2) == pytest.approx(expected, rel=1e-8)


# ---------------------------------------------------------------------------
# instantaneous_forward
# ---------------------------------------------------------------------------

class TestInstantaneousForward:
    def test_flat_curve_forward_equals_rate(self):
        """On a flat curve f(0,t) = z for all t."""
        flat = YieldCurve([0.5, 1.0, 5.0, 10.0], [0.04, 0.04, 0.04, 0.04])
        for method in ["linear", "log_linear", "cubic_spline"]:
            c = YieldCurve([0.5, 1.0, 5.0, 10.0], [0.04, 0.04, 0.04, 0.04], method)
            ts = np.linspace(0.5, 10.0, 20)
            np.testing.assert_allclose(c.instantaneous_forward(ts), 0.04, atol=1e-8)

    def test_upward_curve_forward_above_spot(self, curve):
        """On upward-sloping curve, f(0, 5) > z(5)."""
        assert curve.instantaneous_forward(5.0) > curve.zero_rate(5.0)

    def test_log_linear_piecewise_constant(self):
        """LOG_LINEAR produces piecewise-constant forwards within each interval."""
        c = YieldCurve(TENORS, ZERO_RATES, "log_linear")
        # Sample densely between knots 2 and 3 (t=2 to t=3)
        ts = np.linspace(2.01, 2.99, 50)
        fwds = c.instantaneous_forward(ts)
        assert np.allclose(fwds, fwds[0], rtol=1e-10), "LOG_LINEAR forwards must be constant within an interval"

    def test_cubic_spline_smooth(self):
        """CUBIC_SPLINE forwards should vary smoothly (no step discontinuities)."""
        c = YieldCurve(TENORS, ZERO_RATES, "cubic_spline")
        ts = np.linspace(TENORS[0], TENORS[-1], 500)
        fwds = c.instantaneous_forward(ts)
        # Max jump between adjacent points should be small
        max_jump = np.max(np.abs(np.diff(fwds)))
        assert max_jump < 0.005, f"Cubic spline forward not smooth: max_jump={max_jump}"

    def test_consistent_with_finite_difference(self):
        """f(0,t) ≈ -d/dt log P(0,t) via finite differences."""
        c = YieldCurve(TENORS, ZERO_RATES, "log_linear")
        t_mid = 2.5   # interior point
        eps = 1e-5
        fd_fwd = -(np.log(c.discount_factor(t_mid + eps)) - np.log(c.discount_factor(t_mid - eps))) / (2 * eps)
        assert c.instantaneous_forward(t_mid) == pytest.approx(fd_fwd, rel=1e-4)

    def test_scalar_returns_scalar(self, curve):
        result = curve.instantaneous_forward(2.0)
        assert isinstance(result, float)

    def test_array_returns_array(self, curve):
        ts = np.array([1.0, 2.0, 5.0])
        result = curve.instantaneous_forward(ts)
        assert result.shape == (3,)


# ---------------------------------------------------------------------------
# HullWhite1F integration
# ---------------------------------------------------------------------------

class TestHullWhiteIntegration:
    def setup_method(self):
        self.grid = TimeGrid.uniform(5.0, 60)
        self.curve = YieldCurve(TENORS, ZERO_RATES, "log_linear")
        self.hw = HullWhite1F(a=0.15, sigma=0.010, r0=0.04)

    def test_calibrate_with_yield_curve_key(self):
        """Passing yield_curve in market_data should calibrate theta."""
        self.hw.calibrate({"yield_curve": self.curve, "time_grid": self.grid})
        assert self.hw.theta is not None
        assert len(self.hw.theta) == len(self.grid) - 1

    def test_calibrate_legacy_interface_still_works(self):
        """Passing tenors + zero_rates (old interface) should still work."""
        self.hw.calibrate({
            "tenors": TENORS,
            "zero_rates": ZERO_RATES,
            "time_grid": self.grid,
        })
        assert self.hw.theta is not None
        assert len(self.hw.theta) == len(self.grid) - 1

    def test_r0_matches_curve_short_end(self):
        self.hw.calibrate({"yield_curve": self.curve, "time_grid": self.grid})
        assert self.hw.r0 == pytest.approx(self.curve.zero_rate(0.0), rel=1e-6)

    def test_both_interfaces_produce_similar_theta(self):
        """YieldCurve and legacy raw-array interface should produce close theta values."""
        hw_curve = HullWhite1F(a=0.15, sigma=0.010)
        hw_curve.calibrate({"yield_curve": self.curve, "time_grid": self.grid})

        hw_legacy = HullWhite1F(a=0.15, sigma=0.010)
        hw_legacy.calibrate({
            "tenors": TENORS,
            "zero_rates": ZERO_RATES,
            "time_grid": self.grid,
        })

        # Both use LOG_LINEAR internally → theta should be identical
        np.testing.assert_allclose(hw_curve.theta, hw_legacy.theta, rtol=1e-10)

    def test_cubic_spline_calibration_runs(self):
        curve_cs = YieldCurve(TENORS, ZERO_RATES, "cubic_spline")
        self.hw.calibrate({"yield_curve": curve_cs, "time_grid": self.grid})
        assert self.hw.theta is not None

    def test_simulation_runs_after_calibration(self):
        from risk_analytics import MonteCarloEngine
        self.hw.calibrate({"yield_curve": self.curve, "time_grid": self.grid})
        engine = MonteCarloEngine(n_paths=100, seed=0)
        results = engine.run([self.hw], self.grid)
        assert results["HullWhite1F"].n_paths == 100
