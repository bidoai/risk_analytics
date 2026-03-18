"""Tests for MarketData — bump, scenario, from_dict, accessors."""
from __future__ import annotations

import pytest
import numpy as np

from risk_analytics.core.market_data import MarketData, BumpType, ScenarioBump
from risk_analytics.core.yield_curve import YieldCurve, Interpolation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_curve(rates_offset: float = 0.0) -> YieldCurve:
    tenors = [0.5, 1.0, 2.0, 5.0, 10.0]
    rates = [0.03 + rates_offset, 0.035 + rates_offset, 0.04 + rates_offset,
             0.045 + rates_offset, 0.05 + rates_offset]
    return YieldCurve(tenors, rates, interpolation=Interpolation.LOG_LINEAR)


def make_md() -> MarketData:
    return MarketData(
        curves={"USD": make_curve()},
        spots={"AAPL": 150.0, "MSFT": 300.0},
        vols={"AAPL_VOL": 0.20, "MSFT_VOL": 0.25},
        forward_curves={"OIL": make_curve(0.01)},
    )


# ---------------------------------------------------------------------------
# Accessor tests
# ---------------------------------------------------------------------------

class TestAccessors:
    def test_discount_factor_delegates_to_yield_curve(self):
        md = make_md()
        expected = md.curves["USD"].discount_factor(2.0)
        assert md.discount_factor("USD", 2.0) == pytest.approx(expected)

    def test_zero_rate_delegates(self):
        md = make_md()
        expected = md.curves["USD"].zero_rate(1.0)
        assert md.zero_rate("USD", 1.0) == pytest.approx(expected)

    def test_forward_rate_delegates(self):
        md = make_md()
        expected = md.curves["USD"].forward_rate(1.0, 2.0)
        assert md.forward_rate("USD", 1.0, 2.0) == pytest.approx(expected)

    def test_spot_returns_correct_value(self):
        md = make_md()
        assert md.spot("AAPL") == pytest.approx(150.0)

    def test_spot_raises_key_error_for_unknown(self):
        md = make_md()
        with pytest.raises(KeyError):
            md.spot("GOOG")

    def test_vol_returns_correct_value(self):
        md = make_md()
        assert md.vol("MSFT_VOL") == pytest.approx(0.25)

    def test_vol_raises_key_error_for_unknown(self):
        md = make_md()
        with pytest.raises(KeyError):
            md.vol("UNKNOWN")

    def test_curve_key_error(self):
        md = make_md()
        with pytest.raises(KeyError):
            md.discount_factor("EUR", 1.0)


# ---------------------------------------------------------------------------
# Bump tests
# ---------------------------------------------------------------------------

class TestBump:
    def test_parallel_bump_shifts_all_rates(self):
        md = make_md()
        original_rates = list(md.curves["USD"]._z)
        size = 0.01
        bumped = md.bump("USD", size, BumpType.PARALLEL)

        # Original object is not mutated
        np.testing.assert_allclose(md.curves["USD"]._z, original_rates)

        # All rates shifted up by size
        np.testing.assert_allclose(bumped.curves["USD"]._z,
                                   [r + size for r in original_rates])

    def test_parallel_bump_returns_new_object(self):
        md = make_md()
        bumped = md.bump("USD", 0.01)
        assert bumped is not md
        assert bumped.curves["USD"] is not md.curves["USD"]

    def test_slope_bump_positive_at_long_end(self):
        md = make_md()
        size = 0.01
        bumped = md.bump("USD", size, BumpType.SLOPE)
        orig_rates = list(md.curves["USD"]._z)
        bumped_rates = list(bumped.curves["USD"]._z)

        # Long end should be higher than original (positive tilt)
        assert bumped_rates[-1] > orig_rates[-1]
        # Short end should be lower than original (negative tilt)
        assert bumped_rates[0] < orig_rates[0]

    def test_point_bump_shifts_only_nearest_pillar(self):
        md = make_md()
        size = 0.005
        # Target tenor=1.0 — the second pillar in our curve
        bumped = md.bump("USD", size, BumpType.POINT, tenor=1.0)
        orig_rates = list(md.curves["USD"]._z)
        bumped_rates = list(bumped.curves["USD"]._z)

        # Only the closest pillar changes; others are identical
        for i, (o, b) in enumerate(zip(orig_rates, bumped_rates)):
            pillar = md.curves["USD"]._t[i]
            if abs(pillar - 1.0) < 1e-9:
                assert b == pytest.approx(o + size)
            else:
                assert b == pytest.approx(o)

    def test_bump_spot_scales_by_one_plus_size(self):
        md = make_md()
        original_spot = md.spot("AAPL")
        size = 0.05
        bumped = md.bump("AAPL", size, BumpType.PARALLEL)
        assert bumped.spot("AAPL") == pytest.approx(original_spot * (1 + size))
        # Original unchanged
        assert md.spot("AAPL") == pytest.approx(original_spot)

    def test_bump_vol_adds_size(self):
        md = make_md()
        original_vol = md.vol("AAPL_VOL")
        size = 0.02
        bumped = md.bump("AAPL_VOL", size, BumpType.PARALLEL)
        assert bumped.vol("AAPL_VOL") == pytest.approx(original_vol + size)

    def test_bump_spot_non_parallel_raises(self):
        md = make_md()
        with pytest.raises(ValueError):
            md.bump("AAPL", 0.01, BumpType.SLOPE)

    def test_bump_unknown_key_raises(self):
        md = make_md()
        with pytest.raises(KeyError):
            md.bump("UNKNOWN_KEY", 0.01)

    def test_point_bump_requires_tenor(self):
        md = make_md()
        with pytest.raises(ValueError):
            md.bump("USD", 0.01, BumpType.POINT, tenor=None)


# ---------------------------------------------------------------------------
# Scenario test
# ---------------------------------------------------------------------------

class TestScenario:
    def test_scenario_applies_multiple_bumps_sequentially(self):
        md = make_md()
        bumps = [
            ScenarioBump("USD", 0.01, BumpType.PARALLEL),
            ScenarioBump("AAPL", 0.10, BumpType.PARALLEL),
        ]
        result = md.scenario(bumps)

        # Check curve bump
        orig_rates = list(md.curves["USD"]._z)
        result_rates = list(result.curves["USD"]._z)
        np.testing.assert_allclose(result_rates, [r + 0.01 for r in orig_rates])

        # Check spot bump
        assert result.spot("AAPL") == pytest.approx(md.spot("AAPL") * 1.10)

        # Original is unchanged
        assert md.spot("AAPL") == pytest.approx(150.0)

    def test_scenario_returns_new_object(self):
        md = make_md()
        result = md.scenario([ScenarioBump("USD", 0.01)])
        assert result is not md


# ---------------------------------------------------------------------------
# from_dict test
# ---------------------------------------------------------------------------

class TestFromDict:
    def test_from_dict_constructs_curves_and_spots(self):
        data = {
            "curves": {
                "USD": {
                    "tenors": [0.5, 1.0, 2.0, 5.0],
                    "rates": [0.03, 0.035, 0.04, 0.045],
                    "interpolation": "LOG_LINEAR",
                }
            },
            "spots": {"AAPL": 150.0},
            "vols": {"AAPL": 0.20},
        }
        md = MarketData.from_dict(data)
        assert "USD" in md.curves
        assert md.spot("AAPL") == pytest.approx(150.0)
        assert md.vols["AAPL"] == pytest.approx(0.20)
        assert md.discount_factor("USD", 1.0) == pytest.approx(
            md.curves["USD"].discount_factor(1.0)
        )

    def test_from_dict_default_interpolation(self):
        data = {
            "curves": {
                "EUR": {
                    "tenors": [1.0, 2.0, 5.0],
                    "rates": [0.02, 0.025, 0.03],
                    # No interpolation key — should default to LOG_LINEAR
                }
            }
        }
        md = MarketData.from_dict(data)
        assert md.curves["EUR"].interpolation == Interpolation.LOG_LINEAR

    def test_from_dict_forward_curves(self):
        data = {
            "forward_curves": {
                "OIL": {
                    "tenors": [0.5, 1.0, 2.0],
                    "rates": [0.05, 0.06, 0.07],
                }
            }
        }
        md = MarketData.from_dict(data)
        assert "OIL" in md.forward_curves
        fc = md.forward_curve("OIL")
        assert isinstance(fc, YieldCurve)
