"""risk_analytics — Monte Carlo simulation and exposure analytics library."""

from .backtest import BacktestEngine, BacktestResult
from .core import MonteCarloEngine, Pricer, SimulationResult, StochasticModel, TimeGrid, YieldCurve, Interpolation
from .core.market_data import MarketData, BumpType, ScenarioBump
from .core.grid import SparseTimeGrid
from .models import GeometricBrownianMotion, HestonModel, HullWhite1F, Schwartz1F, Schwartz2F
from .pricing import EuropeanOption, FixedRateBond, InterestRateSwap, ZeroCouponBond
from .exposure import (
    ExposureCalculator,
    NettingSet,
    CSATerms,
    MarginRegime,
    IMModel,
    CollateralAccount,
    HaircutSchedule,
    REGVMEngine,
    REGIMEngine,
    SimmSensitivities,
    SimmCalculator,
    BilateralExposureCalculator,
    ISDAExposureCalculator,
)
from .portfolio import Trade, Agreement
from .pipeline import RiskEngine, RunResult, AgreementResult, EngineConfig

__all__ = [
    # Backtest
    "BacktestEngine",
    "BacktestResult",
    # Core
    "YieldCurve",
    "Interpolation",
    "StochasticModel",
    "Pricer",
    "SimulationResult",
    "MonteCarloEngine",
    "TimeGrid",
    # Market data
    "MarketData",
    "BumpType",
    "ScenarioBump",
    # Sparse grid
    "SparseTimeGrid",
    # Models
    "HullWhite1F",
    "GeometricBrownianMotion",
    "HestonModel",
    "Schwartz1F",
    "Schwartz2F",
    # Pricing
    "ZeroCouponBond",
    "FixedRateBond",
    "InterestRateSwap",
    "EuropeanOption",
    # Exposure — basic
    "ExposureCalculator",
    "NettingSet",
    # Exposure — ISDA/bilateral
    "CSATerms",
    "MarginRegime",
    "IMModel",
    "CollateralAccount",
    "HaircutSchedule",
    "REGVMEngine",
    "REGIMEngine",
    "SimmSensitivities",
    "SimmCalculator",
    "BilateralExposureCalculator",
    "ISDAExposureCalculator",
    # Portfolio
    "Trade",
    "Agreement",
    # Pipeline
    "RiskEngine",
    "RunResult",
    "AgreementResult",
    "EngineConfig",
]
