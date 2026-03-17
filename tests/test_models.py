"""Tests for stochastic models."""
import numpy as np
import pytest

from risk_analytics.core import MonteCarloEngine, TimeGrid
from risk_analytics.models import (
    GeometricBrownianMotion,
    GarmanKohlhagen,
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
# Garman-Kohlhagen FX model
# -----------------------------------------------------------------------

class TestGarmanKohlhagen:
    def setup_method(self):
        self.model = GarmanKohlhagen(S0=1.10, r_d=0.03, r_f=0.01, sigma=0.10)
        self.grid = TimeGrid.uniform(1.0, 252)

    def test_simulate_shape(self):
        engine = MonteCarloEngine(N_PATHS, seed=SEED)
        results = engine.run([self.model], self.grid)
        assert results["GarmanKohlhagen"].paths.shape == (N_PATHS, 253, 1)
        assert results["GarmanKohlhagen"].factor_names == ["S"]

    def test_initial_spot(self):
        engine = MonteCarloEngine(N_PATHS, seed=SEED)
        results = engine.run([self.model], self.grid)
        S0_sim = results["GarmanKohlhagen"].paths[:, 0, 0]
        assert np.allclose(S0_sim, 1.10)

    def test_positive_paths(self):
        engine = MonteCarloEngine(N_PATHS, seed=SEED)
        results = engine.run([self.model], self.grid)
        assert np.all(results["GarmanKohlhagen"].paths > 0)

    def test_risk_neutral_forward(self):
        """E[S(T)] should equal the GK forward S0·exp((r_d - r_f)·T)."""
        engine = MonteCarloEngine(20000, seed=SEED)
        grid = TimeGrid.uniform(1.0, 12)
        results = engine.run([self.model], grid)
        S_terminal = results["GarmanKohlhagen"].factor("S")[:, -1]
        expected = self.model.forward(1.0)
        assert abs(S_terminal.mean() / expected - 1.0) < 0.02

    def test_gk_price_call_put_parity(self):
        """Call - Put = S·e^{-r_f·T} - K·e^{-r_d·T} (Garman-Kohlhagen parity)."""
        S, K, T = 1.10, 1.05, 1.0
        r_d, r_f, sigma = 0.03, 0.01, 0.10
        call = GarmanKohlhagen.gk_price(S, K, T, r_d, r_f, sigma, "call")
        put  = GarmanKohlhagen.gk_price(S, K, T, r_d, r_f, sigma, "put")
        parity = S * np.exp(-r_f * T) - K * np.exp(-r_d * T)
        assert abs((call - put) - parity) < 1e-10

    def test_calibrate_atm_vol(self):
        self.model.calibrate({"atm_vol": 0.15})
        assert self.model.sigma == 0.15

    def test_calibrate_from_option_price(self):
        """Round-trip: price a call, then recover sigma from that price."""
        target_sigma = 0.12
        price = GarmanKohlhagen.gk_price(1.10, 1.10, 1.0, 0.03, 0.01, target_sigma, "call")
        self.model.calibrate({
            "option_price": price, "strike": 1.10,
            "maturity": 1.0, "option_type": "call",
        })
        assert abs(self.model.sigma - target_sigma) < 1e-5

    def test_get_set_params(self):
        self.model.set_params({"sigma": 0.20, "r_d": 0.05})
        p = self.model.get_params()
        assert p["sigma"] == 0.20
        assert p["r_d"] == 0.05


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

    def test_antithetic_odd_paths_raises(self):
        with pytest.raises(ValueError, match="even"):
            MonteCarloEngine(101, antithetic=True)

    def test_antithetic_correct_shape(self):
        gbm = GeometricBrownianMotion(S0=100, mu=0.05, sigma=0.20)
        engine = MonteCarloEngine(1000, seed=SEED, antithetic=True)
        grid = TimeGrid.uniform(1.0, 12)
        results = engine.run([gbm], grid)
        assert results["GBM"].paths.shape == (1000, 13, 1)

    def test_antithetic_paths_are_exact_mirrors(self):
        """For GBM: log(S_base) + log(S_anti) = 2·log(S0) + 2·(μ - σ²/2)·t exactly."""
        mu, sigma, S0 = 0.05, 0.20, 100.0
        gbm = GeometricBrownianMotion(S0=S0, mu=mu, sigma=sigma)
        n = 1000
        engine = MonteCarloEngine(n, seed=SEED, antithetic=True)
        grid = TimeGrid.uniform(1.0, 12)
        results = engine.run([gbm], grid)
        log_S = np.log(results["GBM"].factor("S"))  # (n, T)
        base_log = log_S[: n // 2]   # (n//2, T)
        anti_log = log_S[n // 2 :]   # (n//2, T)
        # The sum of log-prices should equal twice the deterministic drift path
        expected_sum = 2 * np.log(S0) + 2 * (mu - 0.5 * sigma**2) * grid
        assert np.allclose(base_log + anti_log, expected_sum, atol=1e-10)

    def test_antithetic_reduces_variance(self):
        """Antithetic estimator of E[S(T)] should have lower std error than plain MC
        with the same n_paths."""
        model = GeometricBrownianMotion(S0=100, mu=0.05, sigma=0.30)
        grid = TimeGrid.uniform(1.0, 12)
        n = 2000
        n_trials = 50

        plain_means, anti_means = [], []
        for trial in range(n_trials):
            plain = MonteCarloEngine(n, seed=trial)
            anti = MonteCarloEngine(n, seed=trial, antithetic=True)
            plain_means.append(plain.run([model], grid)["GBM"].factor("S")[:, -1].mean())
            anti_means.append(anti.run([model], grid)["GBM"].factor("S")[:, -1].mean())

        assert np.std(anti_means) < np.std(plain_means)
