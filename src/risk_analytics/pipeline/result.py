from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd

from risk_analytics.core.market_data import MarketData, ScenarioBump
from risk_analytics.core.paths import SimulationResult


@dataclass
class NettingSetSummary:
    """Pre-collateral exposure summary for a single netting set."""
    id: str
    ee_profile: np.ndarray    # (T,)
    pfe_profile: np.ndarray   # (T,)
    pse: float
    epe: float


@dataclass
class AgreementResult:
    """
    Full exposure result for one Agreement (ISDA MA + CSA scope).

    Post-collateral profiles are at agreement level.
    Pre-collateral summaries are available per netting set.
    """
    id: str
    counterparty_id: str
    time_grid: np.ndarray

    # Agreement-level profiles (post-collateral)
    ee_profile: np.ndarray
    ene_profile: np.ndarray
    pfe_profile: np.ndarray
    ee_mpor_profile: np.ndarray

    # Per netting set (pre-collateral)
    netting_set_summaries: dict  # str -> NettingSetSummary

    # XVA scalars
    cva: float
    dva: float
    bcva: float
    pse: float
    epe: float
    eepe: float

    # Raw MTM arrays — only populated if write_raw_paths=True
    raw_net_mtm: Optional[np.ndarray] = None   # (n_paths, T)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "counterparty_id": self.counterparty_id,
            "cva": self.cva,
            "dva": self.dva,
            "bcva": self.bcva,
            "pse": self.pse,
            "epe": self.epe,
            "eepe": self.eepe,
        }


@dataclass
class RunResult:
    """
    In-memory result of a full RiskEngine pipeline run.

    Holds simulation paths (needed for stress testing) and
    agreement-level summary statistics. Raw per-path MTM arrays
    are not retained unless write_raw_paths=True was set.
    """
    config: object                             # EngineConfig (avoid circular import)
    time_grid: np.ndarray
    simulation_results: dict                   # str -> SimulationResult (kept for stress test)
    agreement_results: dict                    # str -> AgreementResult
    total_cva: float
    total_dva: float
    total_bcva: float

    def summary_df(self) -> pd.DataFrame:
        """One row per agreement with all scalar metrics."""
        rows = []
        for agr_id, result in self.agreement_results.items():
            rows.append(result.to_dict())
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.set_index("id")
        return df

    def to_parquet(self, path: str) -> None:
        """
        Write consolidated output:
        - {path}/summary.parquet  — scalar metrics per agreement
        - {path}/ee_profiles.parquet — EE time profile per agreement
        - {path}/pfe_profiles.parquet — PFE time profile per agreement
        If raw_net_mtm is populated on any AgreementResult, also writes
        - {path}/raw_mtm_{agreement_id}.parquet
        """
        import os
        os.makedirs(path, exist_ok=True)

        # Summary scalars
        self.summary_df().to_parquet(os.path.join(path, "summary.parquet"))

        # EE profiles
        ee_data = {
            agr_id: result.ee_profile
            for agr_id, result in self.agreement_results.items()
        }
        pd.DataFrame(ee_data, index=self.time_grid).to_parquet(
            os.path.join(path, "ee_profiles.parquet")
        )

        # PFE profiles
        pfe_data = {
            agr_id: result.pfe_profile
            for agr_id, result in self.agreement_results.items()
        }
        pd.DataFrame(pfe_data, index=self.time_grid).to_parquet(
            os.path.join(path, "pfe_profiles.parquet")
        )

        # Optional raw MTM
        for agr_id, result in self.agreement_results.items():
            if result.raw_net_mtm is not None:
                pd.DataFrame(result.raw_net_mtm, columns=self.time_grid).to_parquet(
                    os.path.join(path, f"raw_mtm_{agr_id}.parquet")
                )

    def to_dict(self) -> dict:
        return {
            "total_cva": self.total_cva,
            "total_dva": self.total_dva,
            "total_bcva": self.total_bcva,
            "agreements": {
                agr_id: result.to_dict()
                for agr_id, result in self.agreement_results.items()
            },
        }

    def stress_test(
        self,
        bumps: list,
        market_data: MarketData,
    ) -> "RunResult":
        """
        Reprice on existing simulation paths with bumped market data.
        No re-simulation. Returns a new RunResult.

        bumps: list[ScenarioBump]
        market_data: the base MarketData to apply bumps to
        """
        bumped_md = market_data.scenario(bumps)
        # Re-run phases 2+3 with the bumped market data but same simulation_results
        from risk_analytics.pipeline.engine import RiskEngine
        return RiskEngine._run_exposure_phase(
            config=self.config,
            time_grid=self.time_grid,
            simulation_results=self.simulation_results,
            market_data=bumped_md,
        )
