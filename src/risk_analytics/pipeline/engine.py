from __future__ import annotations
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import numpy as np

from risk_analytics.core.engine import MonteCarloEngine
from risk_analytics.core.grid import SparseTimeGrid
from risk_analytics.core.market_data import MarketData
from risk_analytics.exposure.bilateral import BilateralExposureCalculator, ISDAExposureCalculator
from risk_analytics.exposure.csa import CSATerms, MarginRegime, IMModel
from risk_analytics.exposure.metrics import ExposureCalculator
from risk_analytics.exposure.netting import NettingSet
from risk_analytics.pipeline.config import (
    AgreementConfig, EngineConfig, ModelConfig, TradeFactory,
)
from risk_analytics.pipeline.result import AgreementResult, NettingSetSummary, RunResult
from risk_analytics.portfolio.agreement import Agreement

logger = logging.getLogger(__name__)


class RiskEngine:
    """
    End-to-end Monte Carlo risk pipeline.

    Usage (programmatic):
        engine = RiskEngine(config_dict_or_yaml_path)
        result = engine.run(market_data)

    Usage (YAML):
        engine = RiskEngine.from_yaml("config.yaml")
        result = engine.run()
    """

    def __init__(self, config):
        if isinstance(config, str):
            config = EngineConfig.from_yaml(config)
        elif isinstance(config, dict):
            config = EngineConfig.from_dict(config)
        self.config: EngineConfig = config

    @classmethod
    def from_yaml(cls, path: str) -> "RiskEngine":
        return cls(EngineConfig.from_yaml(path))

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, market_data: Optional[MarketData] = None) -> RunResult:
        """
        Run the full pipeline.

        Phase 1 (serial):   simulate all models on the sparse grid.
        Phase 2 (parallel): compute exposure per agreement.
        Phase 3 (serial):   aggregate into RunResult.
        """
        # ---- Phase 1 -----------------------------------------------
        md = market_data or MarketData.from_dict(self.config.market_data)

        agreements = _build_agreements(self.config, md)

        # Sparse grid: standard grid for max maturity, merged with all cashflow times
        max_maturity = _max_maturity(self.config)
        sim_cfg = self.config.simulation
        if sim_cfg.time_grid.type == "custom" and sim_cfg.time_grid.anchor_points:
            grid = SparseTimeGrid.custom(sim_cfg.time_grid.anchor_points)
        else:
            grid = SparseTimeGrid.standard(max_maturity)

        all_cf_times = []
        for agr in agreements:
            all_cf_times.extend(agr.all_cashflow_times())
        grid = SparseTimeGrid.merge_cashflows(grid, all_cf_times)

        logger.info("Time grid: %d nodes, maturity=%.2f", len(grid), grid[-1])

        models, model_names = _build_models(self.config, md, grid)
        corr_matrix = _build_correlation_matrix(self.config, model_names)

        mc_engine = MonteCarloEngine(
            n_paths=sim_cfg.n_paths,
            seed=sim_cfg.seed,
            antithetic=sim_cfg.antithetic,
            quasi_random=sim_cfg.quasi_random,
        )
        simulation_results = mc_engine.run(models, grid, corr_matrix)
        logger.info("Simulation complete: %d paths × %d steps", sim_cfg.n_paths, len(grid))

        # ---- Phase 2 + 3 -------------------------------------------
        return RiskEngine._run_exposure_phase(
            config=self.config,
            time_grid=grid,
            simulation_results=simulation_results,
            market_data=md,
            agreements=agreements,
        )

    # ------------------------------------------------------------------
    # Exposure phase (callable independently for stress testing)
    # ------------------------------------------------------------------

    @staticmethod
    def _run_exposure_phase(
        config: EngineConfig,
        time_grid: np.ndarray,
        simulation_results: dict,
        market_data: MarketData,
        agreements: Optional[list] = None,
    ) -> RunResult:
        if agreements is None:
            agreements = _build_agreements(config, market_data)

        confidence = config.outputs.confidence
        write_raw = config.outputs.write_raw_paths

        # Decide serial vs parallel
        use_parallel = len(agreements) > 2

        if use_parallel:
            agr_results = _run_parallel(
                agreements, simulation_results, time_grid, confidence, write_raw, config
            )
        else:
            agr_results = {}
            for agr in agreements:
                agr_result = _compute_agreement_result(
                    agr, simulation_results, time_grid, confidence, write_raw, config
                )
                agr_results[agr.id] = agr_result

        total_cva = sum(r.cva for r in agr_results.values())
        total_dva = sum(r.dva for r in agr_results.values())
        total_bcva = sum(r.bcva for r in agr_results.values())

        return RunResult(
            config=config,
            time_grid=time_grid,
            simulation_results=simulation_results,
            agreement_results=agr_results,
            total_cva=total_cva,
            total_dva=total_dva,
            total_bcva=total_bcva,
        )


# ---------------------------------------------------------------------------
# Module-level helpers (picklable for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

class _ModelWrapper:
    """
    Wraps a StochasticModel and overrides its ``name`` property
    with a user-supplied name from the config.

    This ensures that MonteCarloEngine.run() keys simulation results
    by the user-defined model name (e.g. "rates_usd") rather than the
    class-level default (e.g. "HullWhite1F").
    """

    def __init__(self, model, user_name: str):
        self._model = model
        self._user_name = user_name

    # Forward all attribute access to the wrapped model
    def __getattr__(self, item):
        return getattr(self._model, item)

    @property
    def name(self) -> str:
        return self._user_name

    # Required by MonteCarloEngine — delegate explicitly
    @property
    def n_factors(self) -> int:
        return self._model.n_factors

    @property
    def interpolation_space(self) -> list:
        return self._model.interpolation_space

    def simulate(self, time_grid, n_paths, random_draws):
        return self._model.simulate(time_grid, n_paths, random_draws)

    def calibrate(self, market_data):
        return self._model.calibrate(market_data)

    def get_params(self):
        return self._model.get_params()

    def set_params(self, params):
        return self._model.set_params(params)


class _AggregateNettingSet:
    """
    Thin wrapper that presents a pre-computed aggregate MTM array
    as a NettingSet-compatible object for ISDAExposureCalculator.

    ISDAExposureCalculator calls:
      - self.netting_set.net_mtm(simulation_results)  -> (n_paths, T)
      - self.netting_set.name  (used in logging)
    """
    def __init__(self, id_: str, mtm: np.ndarray):
        self.name = id_   # ISDAExposureCalculator uses .name
        self.id = id_
        self._mtm = mtm
        self.trades = {}

    def net_mtm(self, simulation_results=None) -> np.ndarray:
        return self._mtm


def _run_parallel(agreements, simulation_results, time_grid, confidence, write_raw, config):
    """Run exposure computation in parallel using ProcessPoolExecutor."""
    max_workers = min(len(agreements), os.cpu_count() or 1)
    agr_results = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _compute_agreement_result,
                agr, simulation_results, time_grid, confidence, write_raw, config
            ): agr.id
            for agr in agreements
        }
        for future in as_completed(futures):
            agr_id = futures[future]
            try:
                agr_results[agr_id] = future.result()
            except Exception as exc:
                logger.error("Agreement %s raised an exception: %s", agr_id, exc)
                raise

    return agr_results


def _compute_agreement_result(
    agreement: Agreement,
    simulation_results: dict,
    time_grid: np.ndarray,
    confidence: float,
    write_raw: bool,
    config: EngineConfig,
) -> AgreementResult:
    """Compute full exposure result for one Agreement. Runs in a worker process."""
    logger.info("Computing exposure for agreement %s", agreement.id)

    # Find the matching AgreementConfig for hazard rates
    agr_cfg = next(
        (a for a in config.agreements if a.id == agreement.id), None
    )
    cp_hazard = agr_cfg.cp_hazard_rate if agr_cfg else None
    own_hazard = agr_cfg.own_hazard_rate if agr_cfg else None

    # Compute netting set pre-collateral summaries
    calc = ExposureCalculator()
    ns_summaries = {}
    for ns in agreement.netting_sets:
        ns_mtm = ns.net_mtm(simulation_results)
        ee = calc.expected_exposure(ns_mtm)
        pfe = calc.pfe(ns_mtm, confidence)
        pse = float(calc.pse(ns_mtm))
        epe = float(calc.epe(ns_mtm, time_grid))
        ns_summaries[ns.id] = NettingSetSummary(
            id=ns.id,
            ee_profile=ee,
            pfe_profile=pfe,
            pse=pse,
            epe=epe,
        )

    # Aggregate MTM across netting sets (no floor) — VM base
    agg_mtm = agreement.aggregate_mtm(simulation_results)

    # Wrap aggregated MTM in a NettingSet-compatible object
    agg_ns = _AggregateNettingSet(agreement.id, agg_mtm)

    isda_calc = ISDAExposureCalculator(
        netting_set=agg_ns,
        csa=agreement.csa,
    )

    out = isda_calc.run(
        simulation_results=simulation_results,
        time_grid=time_grid,
        confidence=confidence,
        cp_hazard_rate=cp_hazard,
        own_hazard_rate=own_hazard,
    )

    raw_mtm = agg_mtm if write_raw else None

    return AgreementResult(
        id=agreement.id,
        counterparty_id=agreement.counterparty_id,
        time_grid=time_grid,
        ee_profile=out.get("ee", np.zeros(len(time_grid))),
        ene_profile=out.get("ene", np.zeros(len(time_grid))),
        pfe_profile=out.get("pfe", np.zeros(len(time_grid))),
        ee_mpor_profile=out.get("ee_mpor", np.zeros(len(time_grid))),
        netting_set_summaries=ns_summaries,
        cva=float(out.get("cva", 0.0)),
        dva=float(out.get("dva", 0.0)),
        bcva=float(out.get("bcva", 0.0)),
        pse=float(out.get("pse", 0.0)),
        epe=float(out.get("epe", 0.0)),
        eepe=float(out.get("eepe", 0.0)),
        raw_net_mtm=raw_mtm,
    )


def _build_models(config: EngineConfig, market_data: MarketData, time_grid: np.ndarray):
    """Instantiate and optionally calibrate all models. Returns (models_list, names_list)."""
    from risk_analytics.models.rates.hull_white import HullWhite1F
    from risk_analytics.models.equity.gbm import GeometricBrownianMotion
    from risk_analytics.models.equity.heston import HestonModel
    from risk_analytics.models.commodity.schwartz1f import Schwartz1F
    from risk_analytics.models.commodity.schwartz2f import Schwartz2F
    from risk_analytics.models.fx.garman_kohlhagen import GarmanKohlhagen

    MODEL_REGISTRY = {
        "HullWhite1F": HullWhite1F,
        "GBM": GeometricBrownianMotion,
        "GeometricBrownianMotion": GeometricBrownianMotion,
        "HestonModel": HestonModel,
        "Schwartz1F": Schwartz1F,
        "Schwartz2F": Schwartz2F,
        "GarmanKohlhagen": GarmanKohlhagen,
    }

    models = []
    names = []
    for m_cfg in config.models:
        if m_cfg.type not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model type '{m_cfg.type}'. Known: {list(MODEL_REGISTRY)}")
        model_cls = MODEL_REGISTRY[m_cfg.type]
        model = model_cls(**m_cfg.params)

        if m_cfg.calibrate_to:
            if m_cfg.calibrate_to in market_data.curves:
                curve = market_data.curves[m_cfg.calibrate_to]
                calib_data = {
                    "tenors": list(curve._t),
                    "zero_rates": list(curve._z),
                    "time_grid": time_grid,
                }
                model.calibrate(calib_data)
                logger.info("Calibrated %s to curve '%s'", m_cfg.name, m_cfg.calibrate_to)
            else:
                logger.warning(
                    "calibrate_to curve '%s' not found in MarketData; skipping calibration.",
                    m_cfg.calibrate_to,
                )

        # Wrap model so MonteCarloEngine keys results by user-defined name
        wrapped = _ModelWrapper(model, m_cfg.name)
        models.append(wrapped)
        names.append(m_cfg.name)

    return models, names


def _build_correlation_matrix(config: EngineConfig, model_names: list) -> np.ndarray:
    """Build correlation matrix from config entries."""
    n = len(model_names)
    corr = np.eye(n)
    name_to_idx = {name: i for i, name in enumerate(model_names)}

    for entry in config.correlation:
        if entry.model_a not in name_to_idx or entry.model_b not in name_to_idx:
            raise ValueError(
                f"Correlation entry references unknown model: "
                f"'{entry.model_a}' or '{entry.model_b}'. "
                f"Known: {model_names}"
            )
        i = name_to_idx[entry.model_a]
        j = name_to_idx[entry.model_b]
        corr[i, j] = entry.value
        corr[j, i] = entry.value

    return corr


def _build_agreements(config: EngineConfig, market_data: MarketData) -> list:
    """Build Agreement objects from config, including all trades."""
    agreements = []
    for agr_cfg in config.agreements:
        csa_cfg = agr_cfg.csa
        regime = MarginRegime[csa_cfg.margin_regime]
        im_model = IMModel[csa_cfg.im_model]

        # CSATerms uses threshold_party/threshold_counterparty and mta_party/mta_counterparty
        # CSAConfig uses a single threshold and mta (symmetric)
        # mpor in CSAConfig is business days (int); CSATerms expects year-fraction
        mpor_years = csa_cfg.mpor / 252.0

        csa = CSATerms(
            counterparty_id=agr_cfg.counterparty,
            margin_regime=regime,
            threshold_party=csa_cfg.threshold,
            threshold_counterparty=csa_cfg.threshold,
            mta_party=csa_cfg.mta,
            mta_counterparty=csa_cfg.mta,
            ia_party=csa_cfg.ia_posted,
            ia_counterparty=csa_cfg.ia_held,
            im_model=im_model,
            mpor=mpor_years,
        )

        netting_sets = []
        for ns_cfg in agr_cfg.netting_sets:
            ns = NettingSet(ns_cfg.id)
            for t_cfg in ns_cfg.trades:
                trade = TradeFactory.build(t_cfg)
                ns.add_trade(trade)
            netting_sets.append(ns)

        agr = Agreement(
            id=agr_cfg.id,
            counterparty_id=agr_cfg.counterparty,
            netting_sets=netting_sets,
            csa=csa,
        )
        agreements.append(agr)

    return agreements


def _max_maturity(config: EngineConfig) -> float:
    """Find the maximum trade maturity across all agreements."""
    max_mat = 1.0
    for agr_cfg in config.agreements:
        for ns_cfg in agr_cfg.netting_sets:
            for t_cfg in ns_cfg.trades:
                mat = t_cfg.params.get("maturity") or t_cfg.params.get("expiry")
                if mat is not None:
                    max_mat = max(max_mat, float(mat))
    return max_mat


def _make_aggregate_ns(agreement: Agreement) -> NettingSet:
    """Not used — replaced by _AggregateNettingSet."""
    raise NotImplementedError
