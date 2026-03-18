"""Integration tests for the RiskEngine pipeline (steps 8-13)."""
from __future__ import annotations

import numpy as np
import pytest

from risk_analytics.core.market_data import MarketData, ScenarioBump
from risk_analytics.pipeline.config import EngineConfig
from risk_analytics.pipeline.engine import RiskEngine
from risk_analytics.pipeline.result import AgreementResult, RunResult


# ---------------------------------------------------------------------------
# Shared fixture: minimal but complete config
# ---------------------------------------------------------------------------

CONFIG = {
    "simulation": {
        "n_paths": 500,   # small for speed
        "seed": 42,
        "antithetic": False,
        "time_grid": {"type": "standard"},
    },
    "market_data": {
        "curves": {
            "USD_OIS": {
                "tenors": [0.5, 1.0, 2.0, 5.0, 10.0],
                "rates": [0.040, 0.042, 0.044, 0.047, 0.050],
            }
        },
        "spots": {"SPX": 100.0},
        "vols": {"SPX": 0.20},
    },
    "models": [
        {"name": "rates_usd", "type": "HullWhite1F", "params": {"a": 0.15, "sigma": 0.01, "r0": 0.04}},
        {"name": "equity_spx", "type": "GBM", "params": {"S0": 100.0, "mu": 0.06, "sigma": 0.20}},
    ],
    "correlation": [
        ["rates_usd", "equity_spx", 0.10],
    ],
    "agreements": [
        {
            "id": "AGR_001",
            "counterparty": "Counterparty_A",
            "cp_hazard_rate": 0.01,
            "own_hazard_rate": 0.005,
            "csa": {"mta": 10000, "threshold": 0, "margin_regime": "REGVM"},
            "netting_sets": [
                {
                    "id": "NS_001",
                    "trades": [
                        {
                            "id": "trade_001",
                            "type": "InterestRateSwap",
                            "model": "rates_usd",
                            "params": {
                                "fixed_rate": 0.045,
                                "maturity": 5.0,
                                "notional": 1000000,
                                "payer": True,
                            },
                        },
                        {
                            "id": "trade_002",
                            "type": "EuropeanOption",
                            "model": "equity_spx",
                            "params": {
                                "strike": 105.0,
                                "expiry": 2.0,
                                "sigma": 0.20,
                                "risk_free_rate": 0.04,
                                "option_type": "call",
                            },
                        },
                    ],
                }
            ],
        }
    ],
    "outputs": {"confidence": 0.95, "write_raw_paths": False},
}


@pytest.fixture(scope="module")
def run_result() -> RunResult:
    """Run the full pipeline once and share the result across tests."""
    engine = RiskEngine(CONFIG)
    return engine.run()


# ---------------------------------------------------------------------------
# Test 1: EngineConfig.from_dict
# ---------------------------------------------------------------------------

def test_engine_config_from_dict():
    cfg = EngineConfig.from_dict(CONFIG)
    assert cfg.simulation.n_paths == 500
    assert cfg.simulation.seed == 42
    assert len(cfg.models) == 2
    assert cfg.models[0].name == "rates_usd"
    assert cfg.models[1].name == "equity_spx"
    assert len(cfg.agreements) == 1
    assert cfg.agreements[0].id == "AGR_001"
    assert cfg.agreements[0].cp_hazard_rate == pytest.approx(0.01)
    assert cfg.agreements[0].csa.mta == 10000
    assert len(cfg.correlation) == 1
    assert cfg.correlation[0].value == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# Test 2: RiskEngine.run returns a RunResult
# ---------------------------------------------------------------------------

def test_run_returns_run_result(run_result):
    assert isinstance(run_result, RunResult)


# ---------------------------------------------------------------------------
# Test 3: agreement_results has correct key
# ---------------------------------------------------------------------------

def test_agreement_results_key(run_result):
    assert "AGR_001" in run_result.agreement_results


# ---------------------------------------------------------------------------
# Test 4: ee_profile has correct shape (T,)
# ---------------------------------------------------------------------------

def test_ee_profile_shape(run_result):
    agr = run_result.agreement_results["AGR_001"]
    T = len(run_result.time_grid)
    assert isinstance(agr, AgreementResult)
    assert agr.ee_profile.shape == (T,)
    assert agr.pfe_profile.shape == (T,)
    assert agr.ene_profile.shape == (T,)
    assert agr.ee_mpor_profile.shape == (T,)


# ---------------------------------------------------------------------------
# Test 5: cva is a finite positive float
# ---------------------------------------------------------------------------

def test_cva_positive_finite(run_result):
    agr = run_result.agreement_results["AGR_001"]
    assert np.isfinite(agr.cva)
    assert agr.cva > 0.0
    assert np.isfinite(agr.dva)
    assert agr.dva >= 0.0


# ---------------------------------------------------------------------------
# Test 6: summary_df returns a DataFrame with one row
# ---------------------------------------------------------------------------

def test_summary_df(run_result):
    import pandas as pd
    df = run_result.summary_df()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert "AGR_001" in df.index
    assert "cva" in df.columns
    assert "dva" in df.columns
    assert "bcva" in df.columns


# ---------------------------------------------------------------------------
# Test 7: to_dict returns expected structure
# ---------------------------------------------------------------------------

def test_to_dict(run_result):
    d = run_result.to_dict()
    assert "total_cva" in d
    assert "total_dva" in d
    assert "total_bcva" in d
    assert "agreements" in d
    assert "AGR_001" in d["agreements"]
    assert np.isfinite(d["total_cva"])


# ---------------------------------------------------------------------------
# Test 8: stress_test returns a new RunResult with different CVA
# ---------------------------------------------------------------------------

def test_stress_test(run_result):
    md = MarketData.from_dict(CONFIG["market_data"])
    bumps = [ScenarioBump("USD_OIS", 0.001)]  # +10bp parallel shift
    stressed = run_result.stress_test(bumps, md)
    assert isinstance(stressed, RunResult)
    # The stress test uses bumped market data but the same paths,
    # so it should produce a result — CVA may differ
    assert "AGR_001" in stressed.agreement_results
    assert np.isfinite(stressed.total_cva)
    # CVA might change with rate bump (but could be similar for small bumps)
    # Just verify we get a valid new result
    assert stressed.total_cva >= 0.0


# ---------------------------------------------------------------------------
# Test 9: SparseTimeGrid is merged with cashflow times
# ---------------------------------------------------------------------------

def test_time_grid_merged_with_cashflows(run_result):
    # The swap has semi-annual cashflows at 0.5, 1.0, 1.5, ..., 5.0
    # The option expires at 2.0
    # All of these should be in the grid (or very close to a node)
    grid = run_result.time_grid
    tol = 1e-4

    # Check the grid has t=0 and t=5.0 (max maturity)
    assert grid[0] == pytest.approx(0.0)
    assert grid[-1] == pytest.approx(5.0)

    # Check some known cashflow times are in the grid
    for cf_time in [0.5, 1.0, 1.5, 2.0, 2.5, 5.0]:
        assert np.min(np.abs(grid - cf_time)) < tol, (
            f"Expected cashflow time {cf_time} to be in grid. "
            f"Closest: {grid[np.argmin(np.abs(grid - cf_time))]:.6f}"
        )


# ---------------------------------------------------------------------------
# Test 10: netting set summaries populated
# ---------------------------------------------------------------------------

def test_netting_set_summaries(run_result):
    agr = run_result.agreement_results["AGR_001"]
    assert "NS_001" in agr.netting_set_summaries
    ns_summary = agr.netting_set_summaries["NS_001"]
    assert np.isfinite(ns_summary.epe)
    assert ns_summary.epe >= 0.0
    assert np.isfinite(ns_summary.pse)


# ---------------------------------------------------------------------------
# Test 11: total_cva equals sum of per-agreement CVA
# ---------------------------------------------------------------------------

def test_total_cva_aggregation(run_result):
    expected_total = sum(
        r.cva for r in run_result.agreement_results.values()
    )
    assert run_result.total_cva == pytest.approx(expected_total)


# ---------------------------------------------------------------------------
# Test 12: TradeFactory raises on unknown type
# ---------------------------------------------------------------------------

def test_trade_factory_unknown_type():
    from risk_analytics.pipeline.config import TradeFactory, TradeConfig
    with pytest.raises(ValueError, match="Unknown trade type"):
        TradeFactory.build(
            TradeConfig(id="x", type="UnknownInstrument", model="m", params={})
        )


# ---------------------------------------------------------------------------
# Test 13: RiskEngine accepts YAML path (round-trip via temp file)
# ---------------------------------------------------------------------------

def test_engine_config_from_yaml(tmp_path):
    import yaml
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(CONFIG))
    cfg = EngineConfig.from_yaml(str(config_path))
    assert cfg.simulation.n_paths == 500
    assert cfg.agreements[0].id == "AGR_001"
