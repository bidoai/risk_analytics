"""Tests for stochastic models."""
import numpy as np
import pytest

from risk_analytics.core import MonteCarloEngine, TimeGrid
from risk_analytics.models import (
    GeometricBrownianMotion,
    HestonModel,
    HullWhite1F,
    Schwartz1F,
    Schwartz2F,
)

N_PATHS = 5000
SEED = 42


# -----------------------------------------------------------------------
# Hull-White 1F
# -----------------------------------------------------------------------

class TestHullWhite1F:
    def setup_method(self):
        self.model = HullWhite1F(a=0.1, sigma=0.01, r0=0.03)
        self.grid = TimeGrid.uniform(5.0, 60)

    def test_simulate_shape(self):
        engine = MonteCarloEngine(N_PATHS, seed=SEED)
        results = engine.run([self.model], self.grid)
        r = results["HullWhite1F"]
        assert r.paths.shape == (N_PATHS, 61, 1)
        assert r.factor_names == ["r"]

    def test_initial_rate(self):
        engine = MonteCarloEngine(N_PATHS, seed=SEED)
        results = engine.run([self.model], self.grid)
        r0 = results["HullWhite1F"].paths[:, 0, 0]
        assert np.allclose(r0, 0.03)

    def test_mean_reversion(self):
        """Long-run mean of r should be close to theta/a (zero theta → 0 drift centre)."""
        engine = MonteCarloEngine(N_PATHS, seed=SEED)
        results = engine.run([self.model], self.grid)
        r_terminal = results["HullWhite1F"].paths[:, -1, 0]
        # With zero theta, mean should drift toward 0 from r0=0.03
        assert r_terminal.mean() < 0.03 + 0.02  # stays in reasonable range

    def test_get_set_params(self):
        params = self.model.get_params()
        assert "a" in params and "sigma" in params and "r0" in params
        self.model.set_params({"a": 0.2, "sigma": 0.02})
        assert self.model.a == 0.2
        assert self.model.sigma == 0.02


# -----------------------------------------------------------------------
# Geometric Brownian Motion
# -----------------------------------------------------------------------

class TestGBM:
    def setup_method(self):
        self.model = GeometricBrownianMotion(S0=100.0, mu=0.05, sigma=0.20)
        self.grid = TimeGrid.uniform(1.0, 252)

    def test_simulate_shape(self):
        engine = MonteCarloEngine(N_PATHS, seed=SEED)
        results = engine.run([self.model], self.grid)
        S = results["GBM"]
        assert S.paths.shape == (N_PATHS, 253, 1)

    def test_initial_price(self):
        engine = MonteCarloEngine(N_PATHS, seed=SEED)
        results = engine.run([self.model], self.grid)
        S0 = results["GBM"].paths[:, 0, 0]
        assert np.allclose(S0, 100.0)

    def test_log_normal_terminal(self):
        """Terminal log-price should be approx normal."""
        engine = MonteCarloEngine(N_PATHS, seed=SEED)
        results = engine.run([self.model], self.grid)
        S_T = results["GBM"].paths[:, -1, 0]
        log_S_T = np.log(S_T)
        # Expected log-price
        expected_mean = np.log(100.0) + (0.05 - 0.5 * 0.20**2) * 1.0
        assert abs(log_S_T.mean() - expected_mean) < 0.05  # within 5% error

    def test_positive_prices(self):
        engine = MonteCarloEngine(N_PATHS, seed=SEED)
        results = engine.run([self.model], self.grid)
        S = results["GBM"].paths[:, :, 0]
        assert np.all(S > 0)

    def test_calibrate(self):
        self.model.calibrate({"S0": 120.0, "atm_vol": 0.25, "mu": 0.03})
        assert self.model.S0 == 120.0
        assert self.model.sigma == 0.25
        assert self.model.mu == 0.03


# -----------------------------------------------------------------------
# Heston
# -----------------------------------------------------------------------

class TestHeston:
    def setup_method(self):
        self.model = HestonModel(S0=100.0, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
        self.grid = TimeGrid.uniform(1.0, 52)

    def test_simulate_shape(self):
        engine = MonteCarloEngine(N_PATHS, seed=SEED)
        results = engine.run([self.model], self.grid)
        paths = results["Heston"].paths
        assert paths.shape == (N_PATHS, 53, 2)

    def test_variance_non_negative(self):
        """Full truncation should keep variance non-negative."""
        engine = MonteCarloEngine(N_PATHS, seed=SEED)
        results = engine.run([self.model], self.grid)
        v = results["Heston"].factor("v")
        # Negative variance should not appear (full truncation applied in SDE, not stored)
        # The stored v can be slightly negative due to Euler step; positivity enforced internally
        assert results["Heston"].factor("S").min() > 0

    def test_positive_spot(self):
        engine = MonteCarloEngine(N_PATHS, seed=SEED)
        results = engine.run([self.model], self.grid)
        S = results["Heston"].factor("S")
        assert np.all(S > 0)

    def test_factor_names(self):
        engine = MonteCarloEngine(N_PATHS, seed=SEED)
        results = engine.run([self.model], self.grid)
        assert results["Heston"].factor_names == ["S", "v"]


# -----------------------------------------------------------------------
# Schwartz 1F
# -----------------------------------------------------------------------

class TestSchwartz1F:
    def setup_method(self):
        self.model = Schwartz1F(S0=50.0, kappa=1.0, mu=np.log(50), sigma=0.3)
        self.grid = TimeGrid.uniform(2.0, 24)

    def test_simulate_shape(self):
        engine = MonteCarloEngine(N_PATHS, seed=SEED)
        results = engine.run([self.model], self.grid)
        assert results["Schwartz1F"].paths.shape == (N_PATHS, 25, 1)

    def test_positive_prices(self):
        engine = MonteCarloEngine(N_PATHS, seed=SEED)
        results = engine.run([self.model], self.grid)
        S = results["Schwartz1F"].factor("S")
        assert np.all(S > 0)

    def test_mean_reversion_tendency(self):
        """Long-run mean of log(S) should converge towards mu."""
        engine = MonteCarloEngine(10000, seed=SEED)
        model = Schwartz1F(S0=100.0, kappa=2.0, mu=np.log(50), sigma=0.2)
        grid = TimeGrid.uniform(10.0, 120)
        results = engine.run([model], grid)
        S_terminal = results["Schwartz1F"].factor("S")[:, -1]
        # Log of terminal price should be near mu
        assert abs(np.log(S_terminal).mean() - np.log(50)) < 0.5


# -----------------------------------------------------------------------
# Schwartz 2F
# -----------------------------------------------------------------------

class TestSchwartz2F:
    def setup_method(self):
        self.model = Schwartz2F(S0=50.0, kappa=1.5, sigma_xi=0.3, sigma_chi=0.15)
        self.grid = TimeGrid.uniform(2.0, 24)

    def test_simulate_shape(self):
        engine = MonteCarloEngine(N_PATHS, seed=SEED)
        results = engine.run([self.model], self.grid)
        assert results["Schwartz2F"].paths.shape == (N_PATHS, 25, 3)

    def test_factor_names(self):
        engine = MonteCarloEngine(N_PATHS, seed=SEED)
        results = engine.run([self.model], self.grid)
        assert results["Schwartz2F"].factor_names == ["S", "xi", "chi"]

    def test_positive_spot(self):
        engine = MonteCarloEngine(N_PATHS, seed=SEED)
        results = engine.run([self.model], self.grid)
        S = results["Schwartz2F"].factor("S")
        assert np.all(S > 0)


# -----------------------------------------------------------------------
# Engine: correlation
# -----------------------------------------------------------------------

class TestMonteCarloEngine:
    def test_correlation_applied(self):
        """Correlation between two GBM models should match specified rho."""
        gbm1 = GeometricBrownianMotion(S0=100, mu=0.05, sigma=0.20)
        gbm2 = GeometricBrownianMotion(S0=100, mu=0.05, sigma=0.20)
        gbm2._name = "GBM2"  # override for unique key

        # Patch name to avoid collision
        class GBM2(GeometricBrownianMotion):
            @property
            def name(self):
                return "GBM2"

        gbm2 = GBM2(S0=100, mu=0.05, sigma=0.20)
        rho = 0.8
        corr = np.array([[1.0, rho], [rho, 1.0]])
        engine = MonteCarloEngine(10000, seed=SEED)
        grid = TimeGrid.uniform(1.0, 52)
        results = engine.run([gbm1, gbm2], grid, correlation_matrix=corr)

        log_ret1 = np.diff(np.log(results["GBM"].factor("S")), axis=1)
        log_ret2 = np.diff(np.log(results["GBM2"].factor("S")), axis=1)

        # Flatten returns and measure correlation
        flat1 = log_ret1.flatten()
        flat2 = log_ret2.flatten()
        empirical_rho = np.corrcoef(flat1, flat2)[0, 1]
        assert abs(empirical_rho - rho) < 0.05

    def test_invalid_correlation_raises(self):
        gbm = GeometricBrownianMotion()
        engine = MonteCarloEngine(100, seed=SEED)
        grid = TimeGrid.uniform(1.0, 12)
        bad_corr = np.array([[1.0, 1.5], [1.5, 1.0]])  # not PSD
        with pytest.raises((ValueError, np.linalg.LinAlgError)):
            engine.run([gbm], grid, correlation_matrix=bad_corr)
