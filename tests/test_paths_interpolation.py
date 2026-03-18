"""Tests for SimulationResult.at() and at_times() interpolation."""
from __future__ import annotations

import numpy as np
import pytest

from risk_analytics.core.paths import SimulationResult


def make_linear_result() -> SimulationResult:
    """
    2-point grid: t=[0, 1].  2 paths, 1 factor (linear space).
    Path 0: [10, 20]
    Path 1: [5, 15]
    """
    time_grid = np.array([0.0, 1.0])
    paths = np.array([[[10.0], [20.0]],
                      [[5.0],  [15.0]]])  # (2, 2, 1)
    return SimulationResult(
        time_grid=time_grid,
        paths=paths,
        model_name="Test",
        factor_names=["x"],
        interpolation_space=["linear"],
    )


def make_log_result() -> SimulationResult:
    """
    2-point grid: t=[0, 1].  2 paths, 1 factor (log space).
    Path 0: [100, 200]
    Path 1: [50, 50]
    """
    time_grid = np.array([0.0, 1.0])
    paths = np.array([[[100.0], [200.0]],
                      [[50.0],  [50.0]]])  # (2, 2, 1)
    return SimulationResult(
        time_grid=time_grid,
        paths=paths,
        model_name="Test",
        factor_names=["S"],
        interpolation_space=["log"],
    )


def make_two_factor_result() -> SimulationResult:
    """
    3-point grid: t=[0, 0.5, 1.0].  1 path, 2 factors.
    Factor 0 (log): [100, 150, 200]
    Factor 1 (linear): [0.04, 0.06, 0.08]
    """
    time_grid = np.array([0.0, 0.5, 1.0])
    paths = np.array([[[100.0, 0.04],
                       [150.0, 0.06],
                       [200.0, 0.08]]])  # (1, 3, 2)
    return SimulationResult(
        time_grid=time_grid,
        paths=paths,
        model_name="Test",
        factor_names=["S", "v"],
        interpolation_space=["log", "linear"],
    )


class TestAtExactGridPoint:
    def test_at_t0_returns_first_column(self):
        result = make_linear_result()
        vals = result.at(0.0)
        np.testing.assert_allclose(vals[:, 0], [10.0, 5.0])

    def test_at_t1_returns_last_column(self):
        result = make_linear_result()
        vals = result.at(1.0)
        np.testing.assert_allclose(vals[:, 0], [20.0, 15.0])

    def test_at_exact_grid_point_shape(self):
        result = make_linear_result()
        vals = result.at(0.0)
        assert vals.shape == (2, 1)  # (n_paths, n_factors)


class TestLinearInterpolation:
    def test_midpoint_linear(self):
        result = make_linear_result()
        vals = result.at(0.5)
        # Linear interp: path 0: 10 + 0.5*(20-10) = 15, path 1: 5 + 0.5*(15-5) = 10
        np.testing.assert_allclose(vals[:, 0], [15.0, 10.0])

    def test_quarter_point_linear(self):
        result = make_linear_result()
        vals = result.at(0.25)
        # path 0: 10 + 0.25*10 = 12.5, path 1: 5 + 0.25*10 = 7.5
        np.testing.assert_allclose(vals[:, 0], [12.5, 7.5])


class TestLogInterpolation:
    def test_midpoint_log(self):
        result = make_log_result()
        vals = result.at(0.5)
        # Log interp at frac=0.5: exp(log(100) + 0.5*(log(200)-log(100)))
        # = exp(log(100) * 0.5 + log(200) * 0.5) = sqrt(100 * 200) = sqrt(20000)
        expected_path0 = np.sqrt(100.0 * 200.0)
        np.testing.assert_allclose(vals[0, 0], expected_path0, rtol=1e-10)

    def test_constant_path_log(self):
        result = make_log_result()
        vals = result.at(0.5)
        # path 1: both endpoints = 50, log interp = 50
        np.testing.assert_allclose(vals[1, 0], 50.0)


class TestTwoFactors:
    def test_interpolation_at_midpoint_two_factors(self):
        result = make_two_factor_result()
        # t=0.25, bracketed by t=0 and t=0.5, frac=0.5
        vals = result.at(0.25)
        assert vals.shape == (1, 2)

        # Factor 0 (log): exp(log(100) + 0.5*(log(150)-log(100))) = sqrt(100*150)
        expected_s = np.sqrt(100.0 * 150.0)
        np.testing.assert_allclose(vals[0, 0], expected_s, rtol=1e-10)

        # Factor 1 (linear): 0.04 + 0.5*(0.06-0.04) = 0.05
        np.testing.assert_allclose(vals[0, 1], 0.05)

    def test_at_middle_grid_point(self):
        result = make_two_factor_result()
        vals = result.at(0.5)
        np.testing.assert_allclose(vals[0, 0], 150.0)
        np.testing.assert_allclose(vals[0, 1], 0.06)


class TestAtTimes:
    def test_at_times_matches_repeated_at_calls(self):
        result = make_linear_result()
        times = [0.0, 0.25, 0.5, 0.75, 1.0]
        result_batch = result.at_times(times)

        for i, t in enumerate(times):
            single = result.at(t)
            np.testing.assert_allclose(result_batch[:, i, :], single)

    def test_at_times_shape(self):
        result = make_linear_result()
        times = [0.0, 0.5, 1.0]
        out = result.at_times(times)
        # (n_paths=2, len(times)=3, n_factors=1)
        assert out.shape == (2, 3, 1)

    def test_at_times_two_factors_shape(self):
        result = make_two_factor_result()
        times = [0.0, 0.25, 0.75, 1.0]
        out = result.at_times(times)
        assert out.shape == (1, 4, 2)


class TestDefaultInterpolationSpace:
    def test_default_interpolation_space_is_linear(self):
        time_grid = np.array([0.0, 1.0])
        paths = np.array([[[5.0], [15.0]]])
        result = SimulationResult(
            time_grid=time_grid,
            paths=paths,
            model_name="Test",
            factor_names=["x"],
        )
        # No interpolation_space provided — should default to ["linear"]
        assert result.interpolation_space == ["linear"]
        vals = result.at(0.5)
        np.testing.assert_allclose(vals[0, 0], 10.0)  # linear midpoint
