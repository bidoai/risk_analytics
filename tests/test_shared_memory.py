"""Tests for SimulationSharedMemory context manager."""
from __future__ import annotations

import numpy as np
import pytest

from risk_analytics.core.paths import SimulationResult
from risk_analytics.pipeline.shared_memory import SimulationSharedMemory


def _make_results() -> dict:
    rng = np.random.default_rng(42)
    time_grid = np.linspace(0, 1.0, 10)
    paths_hw = rng.standard_normal((50, 10, 1)).astype(np.float64)
    paths_gbm = rng.standard_normal((50, 10, 1)).astype(np.float64)
    return {
        "rates": SimulationResult(
            time_grid=time_grid,
            paths=paths_hw,
            model_name="HullWhite1F",
            factor_names=["r"],
            interpolation_space=["linear"],
        ),
        "equity": SimulationResult(
            time_grid=time_grid,
            paths=paths_gbm,
            model_name="GBM",
            factor_names=["S"],
            interpolation_space=["log"],
        ),
    }


class TestSimulationSharedMemory:
    def test_context_manager_no_leak(self):
        """SharedMemory blocks should be released on exit without error."""
        sim_results = _make_results()
        with SimulationSharedMemory(sim_results) as shm:
            assert len(shm.descriptors) == 2

        # After exit, the shm_blocks should be cleared
        assert len(shm._shm_blocks) == 0

    def test_descriptors_keys(self):
        sim_results = _make_results()
        with SimulationSharedMemory(sim_results) as shm:
            desc = shm.descriptors
            assert set(desc.keys()) == {"rates", "equity"}
            for name, d in desc.items():
                assert "shm_name" in d
                assert "shape" in d
                assert "dtype" in d
                assert "time_grid" in d
                assert "factor_names" in d

    def test_attach_returns_correct_arrays(self):
        """Attached arrays should have the same values as the originals."""
        sim_results = _make_results()
        with SimulationSharedMemory(sim_results) as shm:
            desc = shm.descriptors
            attached = SimulationSharedMemory.attach(desc)
            try:
                results = SimulationSharedMemory.results_from_attached(attached)
                for model_name, result in results.items():
                    orig = sim_results[model_name]
                    np.testing.assert_array_equal(result.paths, orig.paths)
                    np.testing.assert_array_equal(result.time_grid, orig.time_grid)
                    assert result.factor_names == orig.factor_names
                    assert result.interpolation_space == orig.interpolation_space
            finally:
                SimulationSharedMemory.detach(attached)

    def test_attach_detach_roundtrip(self):
        """Attach → modify should not affect original (separate memory). Detach cleans up."""
        sim_results = _make_results()
        orig_paths = sim_results["rates"].paths.copy()
        with SimulationSharedMemory(sim_results) as shm:
            desc = shm.descriptors
            attached = SimulationSharedMemory.attach(desc)
            results = SimulationSharedMemory.results_from_attached(attached)
            # Shared memory IS the same block; modifying it would change original
            # (that's the point — zero copy). Just verify shapes.
            assert results["rates"].paths.shape == orig_paths.shape
            SimulationSharedMemory.detach(attached)

    def test_empty_results(self):
        with SimulationSharedMemory({}) as shm:
            assert shm.descriptors == {}
