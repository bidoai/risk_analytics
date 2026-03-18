from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# ------------------------------------------------------------------
# Config dataclasses
# ------------------------------------------------------------------

@dataclass
class TimeGridConfig:
    type: str = "standard"          # "standard" or "custom"
    anchor_points: Optional[list] = None


@dataclass
class SimulationConfig:
    n_paths: int = 10_000
    seed: int = 42
    antithetic: bool = False
    quasi_random: bool = False
    time_grid: TimeGridConfig = field(default_factory=TimeGridConfig)


@dataclass
class ModelConfig:
    name: str
    type: str
    params: dict = field(default_factory=dict)
    calibrate_to: Optional[str] = None   # name of a curve in MarketData


@dataclass
class CorrelationEntry:
    model_a: str
    model_b: str
    value: float


@dataclass
class CSAConfig:
    mta: float = 0.0
    threshold: float = 0.0
    ia_held: float = 0.0
    ia_posted: float = 0.0
    margin_regime: str = "REGVM"
    im_model: str = "SCHEDULE"
    mpor: int = 10


@dataclass
class TradeConfig:
    id: str
    type: str
    model: str
    params: dict = field(default_factory=dict)


@dataclass
class NettingSetConfig:
    id: str
    trades: list = field(default_factory=list)   # list[TradeConfig]


@dataclass
class AgreementConfig:
    id: str
    counterparty: str
    csa: CSAConfig = field(default_factory=CSAConfig)
    netting_sets: list = field(default_factory=list)   # list[NettingSetConfig]
    cp_hazard_rate: Optional[float] = None
    own_hazard_rate: Optional[float] = None


@dataclass
class OutputConfig:
    metrics: list = field(default_factory=lambda: ["EE", "PFE", "CVA"])
    confidence: float = 0.95
    format: str = "parquet"
    path: str = "./results/"
    write_raw_paths: bool = False


@dataclass
class EngineConfig:
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    market_data: dict = field(default_factory=dict)
    models: list = field(default_factory=list)          # list[ModelConfig]
    correlation: list = field(default_factory=list)     # list[CorrelationEntry]
    agreements: list = field(default_factory=list)      # list[AgreementConfig]
    outputs: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "EngineConfig":
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "EngineConfig":
        sim_raw = data.get("simulation", {})
        tg_raw = sim_raw.get("time_grid", {})
        time_grid_cfg = TimeGridConfig(
            type=tg_raw.get("type", "standard"),
            anchor_points=tg_raw.get("anchor_points"),
        )
        sim_cfg = SimulationConfig(
            n_paths=sim_raw.get("n_paths", 10_000),
            seed=sim_raw.get("seed", 42),
            antithetic=sim_raw.get("antithetic", False),
            quasi_random=sim_raw.get("quasi_random", False),
            time_grid=time_grid_cfg,
        )

        models = [
            ModelConfig(
                name=m["name"],
                type=m["type"],
                params=m.get("params", {}),
                calibrate_to=m.get("calibrate_to"),
            )
            for m in data.get("models", [])
        ]

        # Correlation: list of [model_a, model_b, value] triples
        corr_entries = []
        for entry in data.get("correlation", []):
            corr_entries.append(CorrelationEntry(
                model_a=entry[0], model_b=entry[1], value=float(entry[2])
            ))

        agreements = []
        for agr_raw in data.get("agreements", []):
            csa_raw = agr_raw.get("csa", {})
            csa_cfg = CSAConfig(
                mta=csa_raw.get("mta", 0.0),
                threshold=csa_raw.get("threshold", 0.0),
                ia_held=csa_raw.get("ia_held", 0.0),
                ia_posted=csa_raw.get("ia_posted", 0.0),
                margin_regime=csa_raw.get("margin_regime", "REGVM"),
                im_model=csa_raw.get("im_model", "SCHEDULE"),
                mpor=csa_raw.get("mpor", 10),
            )
            netting_sets = []
            for ns_raw in agr_raw.get("netting_sets", []):
                trades = [
                    TradeConfig(
                        id=t["id"],
                        type=t["type"],
                        model=t["model"],
                        params=t.get("params", {}),
                    )
                    for t in ns_raw.get("trades", [])
                ]
                netting_sets.append(NettingSetConfig(id=ns_raw["id"], trades=trades))
            agreements.append(AgreementConfig(
                id=agr_raw["id"],
                counterparty=agr_raw["counterparty"],
                csa=csa_cfg,
                netting_sets=netting_sets,
                cp_hazard_rate=agr_raw.get("cp_hazard_rate"),
                own_hazard_rate=agr_raw.get("own_hazard_rate"),
            ))

        out_raw = data.get("outputs", {})
        outputs = OutputConfig(
            metrics=out_raw.get("metrics", ["EE", "PFE", "CVA"]),
            confidence=out_raw.get("confidence", 0.95),
            format=out_raw.get("format", "parquet"),
            path=out_raw.get("path", "./results/"),
            write_raw_paths=out_raw.get("write_raw_paths", False),
        )

        return cls(
            simulation=sim_cfg,
            market_data=data.get("market_data", {}),
            models=models,
            correlation=corr_entries,
            agreements=agreements,
            outputs=outputs,
        )


class TradeFactory:
    """Builds Pricer instances from TradeConfig dicts.

    Built-in types: InterestRateSwap, ZeroCouponBond, FixedRateBond, EuropeanOption.

    Register custom instrument types with the ``register`` decorator::

        @TradeFactory.register("MyOption")
        def _build_my_option(params):
            return MyOption(strike=params["strike"], ...)

    The registered function receives the raw ``params`` dict from the
    trade config and must return a ``Pricer`` instance.
    """

    _CUSTOM_REGISTRY: dict = {}

    @classmethod
    def register(cls, type_name: str):
        """Decorator to register a custom pricer builder.

        Parameters
        ----------
        type_name : str
            The ``type`` string used in YAML/dict trade configs.

        Example
        -------
        ::

            @TradeFactory.register("BarrierOption")
            def _build_barrier(params):
                return BarrierOption(
                    strike=params["strike"],
                    barrier=params["barrier"],
                    expiry=params["expiry"],
                )
        """
        def decorator(fn):
            cls._CUSTOM_REGISTRY[type_name] = fn
            return fn
        return decorator

    @staticmethod
    def build(trade_cfg: TradeConfig):
        """Returns a Trade object."""
        from risk_analytics.portfolio.trade import Trade
        pricer = TradeFactory._build_pricer(trade_cfg)
        return Trade(id=trade_cfg.id, pricer=pricer, model_name=trade_cfg.model)

    @staticmethod
    def _build_pricer(cfg: TradeConfig):
        t = cfg.type
        p = cfg.params

        # Check custom registry first
        if t in TradeFactory._CUSTOM_REGISTRY:
            return TradeFactory._CUSTOM_REGISTRY[t](p)

        if t == "InterestRateSwap":
            from risk_analytics.pricing.rates.swap import InterestRateSwap
            # frequency is the year fraction between payments (e.g. 0.5 = semi-annual)
            # InterestRateSwap takes payment_freq = payments per year (int)
            frequency = p.get("frequency", 0.5)
            payment_freq = max(1, int(round(1.0 / frequency)))
            return InterestRateSwap(
                fixed_rate=p["fixed_rate"],
                maturity=p["maturity"],
                notional=p.get("notional", 1_000_000),
                payer=p.get("payer", True),
                payment_freq=payment_freq,
            )
        elif t == "ZeroCouponBond":
            from risk_analytics.pricing.rates.bond import ZeroCouponBond
            return ZeroCouponBond(
                face_value=p.get("face_value", 1_000_000),
                maturity=p["maturity"],
            )
        elif t == "FixedRateBond":
            from risk_analytics.pricing.rates.bond import FixedRateBond
            return FixedRateBond(
                face_value=p.get("face_value", 1_000_000),
                coupon_rate=p["coupon_rate"],
                maturity=p["maturity"],
                coupon_freq=max(1, int(round(1.0 / p.get("frequency", 0.5)))),
            )
        elif t == "EuropeanOption":
            from risk_analytics.pricing.equity.vanilla_option import EuropeanOption
            return EuropeanOption(
                strike=p["strike"],
                expiry=p["expiry"],
                sigma=p["sigma"],
                risk_free_rate=p.get("risk_free_rate", 0.04),
                option_type=p.get("option_type", "call"),
            )
        else:
            known = ", ".join(
                ["InterestRateSwap", "ZeroCouponBond", "FixedRateBond", "EuropeanOption"]
                + list(TradeFactory._CUSTOM_REGISTRY)
            )
            raise ValueError(
                f"Unknown trade type '{t}'. "
                f"Built-in and registered types: {known}. "
                f"Use @TradeFactory.register('{t}') to add custom types."
            )


def _register_builtin_trades():
    """Auto-register built-in exotic pricers so they work in YAML configs without
    manual @TradeFactory.register() calls."""
    try:
        from risk_analytics.pricing.exotic.barrier_option import BarrierOption

        if "BarrierOption" not in TradeFactory._CUSTOM_REGISTRY:
            @TradeFactory.register("BarrierOption")
            def _build_barrier(params):
                return BarrierOption(
                    strike=params["strike"],
                    barrier=params["barrier"],
                    expiry=params["expiry"],
                    barrier_type=params.get("barrier_type", "down-out"),
                    sigma=params.get("sigma", 0.20),
                    risk_free_rate=params.get("risk_free_rate", 0.0),
                    option_type=params.get("option_type", "call"),
                )
    except ImportError:
        pass

    try:
        from risk_analytics.pricing.exotic.asian_option import AsianOption

        if "AsianOption" not in TradeFactory._CUSTOM_REGISTRY:
            @TradeFactory.register("AsianOption")
            def _build_asian(params):
                return AsianOption(
                    strike=params["strike"],
                    expiry=params["expiry"],
                    risk_free_rate=params.get("risk_free_rate", 0.0),
                )
    except ImportError:
        pass


_register_builtin_trades()
