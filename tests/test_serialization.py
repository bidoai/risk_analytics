"""Tests for model parameter serialisation (save / load)."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from risk_analytics.models import (
    GarmanKohlhagen,
    GeometricBrownianMotion,
    HestonModel,
    HullWhite1F,
    Schwartz1F,
    Schwartz2F,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def round_trip(model, tmp_path):
    """Save model params, load into a fresh same-type instance, return it."""
    path = tmp_path / "params.json"
    model.save(path)
    fresh = type(model)()
    fresh.load(path)
    return fresh


# ---------------------------------------------------------------------------
# JSON file structure
# ---------------------------------------------------------------------------

class TestJsonFormat:
    def test_file_contains_model_name(self, tmp_path):
        m = GeometricBrownianMotion(S0=120.0, mu=0.06, sigma=0.25)
        m.save(tmp_path / "gbm.json")
        payload = json.loads((tmp_path / "gbm.json").read_text())
        assert payload["model"] == "GBM"

    def test_file_contains_params_key(self, tmp_path):
        m = GeometricBrownianMotion()
        m.save(tmp_path / "gbm.json")
        payload = json.loads((tmp_path / "gbm.json").read_text())
        assert "params" in payload

    def test_numpy_array_serialised_as_list(self, tmp_path):
        grid = np.linspace(0, 5, 61)
        m = HullWhite1F(a=0.1, sigma=0.01, r0=0.03, theta=np.ones(60) * 0.02)
        m.save(tmp_path / "hw.json")
        payload = json.loads((tmp_path / "hw.json").read_text())
        assert isinstance(payload["params"]["theta"], list)
        assert all(isinstance(v, float) for v in payload["params"]["theta"])

    def test_none_param_omitted(self, tmp_path):
        """HullWhite with no theta should not write a theta key."""
        m = HullWhite1F(a=0.1, sigma=0.01, r0=0.03)  # theta=None
        m.save(tmp_path / "hw.json")
        payload = json.loads((tmp_path / "hw.json").read_text())
        assert "theta" not in payload["params"]


# ---------------------------------------------------------------------------
# Round-trip correctness for each model
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_gbm(self, tmp_path):
        m = GeometricBrownianMotion(S0=120.0, mu=0.07, sigma=0.22)
        r = round_trip(m, tmp_path)
        assert r.S0 == pytest.approx(120.0)
        assert r.mu == pytest.approx(0.07)
        assert r.sigma == pytest.approx(0.22)

    def test_hull_white_scalars(self, tmp_path):
        m = HullWhite1F(a=0.15, sigma=0.008, r0=0.04)
        r = round_trip(m, tmp_path)
        assert r.a == pytest.approx(0.15)
        assert r.sigma == pytest.approx(0.008)
        assert r.r0 == pytest.approx(0.04)

    def test_hull_white_with_theta(self, tmp_path):
        theta = np.linspace(0.01, 0.03, 60)
        m = HullWhite1F(a=0.1, sigma=0.01, r0=0.03, theta=theta)
        path = tmp_path / "hw.json"
        m.save(path)
        fresh = HullWhite1F()
        fresh.load(path)
        assert fresh.theta is not None
        assert np.allclose(fresh.theta, theta)

    def test_heston(self, tmp_path):
        m = HestonModel(S0=110.0, v0=0.05, mu=0.04, kappa=3.0,
                        theta=0.06, xi=0.4, rho=-0.6)
        r = round_trip(m, tmp_path)
        assert r.S0 == pytest.approx(110.0)
        assert r.kappa == pytest.approx(3.0)
        assert r.rho == pytest.approx(-0.6)

    def test_schwartz1f(self, tmp_path):
        m = Schwartz1F(S0=60.0, kappa=1.5, mu=4.1, sigma=0.28)
        r = round_trip(m, tmp_path)
        assert r.S0 == pytest.approx(60.0)
        assert r.kappa == pytest.approx(1.5)
        assert r.mu == pytest.approx(4.1)

    def test_schwartz2f(self, tmp_path):
        m = Schwartz2F(S0=55.0, kappa=2.0, sigma_xi=0.35, sigma_chi=0.12, rho=0.4)
        r = round_trip(m, tmp_path)
        assert r.S0 == pytest.approx(55.0)
        assert r.kappa == pytest.approx(2.0)
        assert r.rho == pytest.approx(0.4)

    def test_garman_kohlhagen(self, tmp_path):
        m = GarmanKohlhagen(S0=1.25, r_d=0.04, r_f=0.02, sigma=0.09)
        r = round_trip(m, tmp_path)
        assert r.S0 == pytest.approx(1.25)
        assert r.r_d == pytest.approx(0.04)
        assert r.sigma == pytest.approx(0.09)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrors:
    def test_load_wrong_model_raises(self, tmp_path):
        gbm = GeometricBrownianMotion(S0=100.0)
        gbm.save(tmp_path / "gbm.json")
        hw = HullWhite1F()
        with pytest.raises(ValueError, match="GBM"):
            hw.load(tmp_path / "gbm.json")

    def test_load_missing_file_raises(self, tmp_path):
        m = GeometricBrownianMotion()
        with pytest.raises(FileNotFoundError):
            m.load(tmp_path / "nonexistent.json")

    def test_save_creates_file(self, tmp_path):
        m = GeometricBrownianMotion()
        path = tmp_path / "out.json"
        assert not path.exists()
        m.save(path)
        assert path.exists()


# ---------------------------------------------------------------------------
# Chaining
# ---------------------------------------------------------------------------

class TestChaining:
    def test_load_returns_self(self, tmp_path):
        m = GeometricBrownianMotion(S0=99.0)
        m.save(tmp_path / "m.json")
        fresh = GeometricBrownianMotion()
        result = fresh.load(tmp_path / "m.json")
        assert result is fresh
        assert fresh.S0 == pytest.approx(99.0)

    def test_constructor_load_pattern(self, tmp_path):
        """GeometricBrownianMotion().load('path') one-liner."""
        GeometricBrownianMotion(S0=88.0).save(tmp_path / "m.json")
        m = GeometricBrownianMotion().load(tmp_path / "m.json")
        assert m.S0 == pytest.approx(88.0)
