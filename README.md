# risk_analytics

A Python library for Monte Carlo counterparty credit risk (CCR) and XVA analytics.
Designed to demonstrate production-quality architecture for simulation, pricing, margining,
and exposure under realistic ISDA CSA dynamics.



- **Correlated multi-asset simulation** — interest rates (HW1F, G2++), equity (GBM, Heston), FX (Garman-Kohlhagen), commodities (Schwartz 1F/2F)
- **Path-dependent pricing** — barrier options, Asian options, IRS, bonds, European options
- **Explicit path-wise state modeling** — both pricing (`StatefulPricer`) and margining (`REGVMStepper`) carry per-path state, enabling correct handling of path-dependent instruments and CSA dynamics
- **Full ISDA CSA margining** — VM and IM (Schedule + SIMM), MPOR, collateral accounts
- **Two exposure engines** — batch (`ISDAExposureCalculator`) and streaming (`StreamingExposureEngine`, O(n_paths) memory)
- **YAML-driven `RiskEngine`** — end-to-end CVA/DVA/EE/PFE from a single config file

Not included (left to production systems): market data connectors (Bloomberg, Refinitiv),
trade ingestion, governance controls, audit logging, and monitoring.

---

## Quick Start

```python
from risk_analytics import RiskEngine

result = RiskEngine.from_yaml("examples/single_swap.yaml").run()

print(result.total_cva)
print(result.summary_df())
```

Or from a dict config:

```python
from risk_analytics import RiskEngine

result = RiskEngine({
    "simulation": {"n_paths": 10000, "seed": 42, "time_grid": {"type": "standard"}},
    "market_data": {"curves": {"USD_OIS": {"tenors": [1, 2, 5], "rates": [0.04, 0.044, 0.047]}}},
    "models": [{"name": "rates", "type": "HullWhite1F", "params": {"a": 0.15, "sigma": 0.01, "r0": 0.04},
                "calibrate_to": "USD_OIS"}],
    "agreements": [{"id": "AGR_001", "counterparty": "Bank_A", "cp_hazard_rate": 0.01,
                    "own_hazard_rate": 0.005, "csa": {"mta": 10000, "threshold": 0, "margin_regime": "REGVM"},
                    "netting_sets": [{"id": "NS_1", "trades": [
                        {"id": "swap_5y", "type": "InterestRateSwap", "model": "rates",
                         "params": {"fixed_rate": 0.045, "maturity": 5.0, "notional": 1000000, "payer": True}}
                    ]}]}],
}).run()

print(f"CVA: {result.total_cva:,.0f}")
```

---

## Installation

```bash
uv sync          # installs all dependencies
uv run pytest    # 405 tests
uv run python demo.py
```

**Requirements:** Python 3.12+, `numpy`, `scipy`, `pandas`, `pyyaml`

See [DESIGN.md](DESIGN.md) for the reasoning behind key architectural decisions.

---

## Design Philosophy

- **Correctness over premature optimisation** — the default pipeline materialises paths and MTM matrices for clarity and debuggability; optimise only when needed
- **Explicit state over hidden abstractions** — path-dependent pricing (`StatefulPricer`) and collateral dynamics (`REGVMStepper`) carry explicit per-path state rather than hiding it in object mutation
- **Separation of concerns** — simulation, pricing, margining, and exposure are cleanly decoupled; each layer can be used independently
- **Scalability as an extension, not a constraint** — the streaming engine handles large-scale workloads, but is not required for typical use cases (1k–10k paths)

---

## Scope

This library focuses on Monte Carlo exposure and XVA analytics. It is not:

- a low-latency pricing system
- real-time risk infrastructure
- an enterprise data integration framework

It is designed to demonstrate correct modelling, clean architecture, and extensibility.

---

## Architecture

```
risk_analytics/
├── core/
│   ├── base.py          # StochasticModel + Pricer ABCs
│   │                    #   interpolation_space — sparse path interpolation space per factor
│   │                    #   cashflow_times()    — payment dates for grid augmentation
│   │                    #   price_at()          — single time-step MTM slice
│   ├── stateful.py      # PathState + StatefulPricer ABC (path-dependent instruments)
│   ├── engine.py        # MonteCarloEngine (Cholesky correlation, antithetic, Sobol)
│   ├── grid.py          # TimeGrid (uniform); SparseTimeGrid (daily→weekly→monthly)
│   ├── paths.py         # SimulationResult — at(t) / at_times(ts) sparse interpolation
│   ├── market_data.py   # MarketData — curves, spots, vols; bump() / scenario()
│   ├── conventions.py   # DayCountConvention, BusinessDayConvention, Calendar hierarchy
│   └── schedule.py      # Schedule (payment dates, day-count fractions)
├── models/
│   ├── rates/hull_white.py        # HullWhite1F — exact Gaussian discretisation
│   ├── rates/hull_white2f.py      # HullWhite2F (G2++) — two-factor short rate
│   ├── equity/gbm.py              # GeometricBrownianMotion — exact log-normal
│   ├── equity/heston.py           # HestonModel — Euler + full truncation
│   ├── commodity/schwartz1f.py    # Schwartz1F — exact OU
│   ├── commodity/schwartz2f.py    # Schwartz2F — exact OU + BM cumsum
│   └── fx/garman_kohlhagen.py     # GarmanKohlhagen — exact log-normal
├── pricing/
│   ├── rates/swap.py              # InterestRateSwap (uniform or Schedule)
│   ├── rates/bond.py              # ZeroCouponBond, FixedRateBond
│   ├── equity/vanilla_option.py   # EuropeanOption — vectorised Black-Scholes
│   ├── exotic/barrier_option.py   # BarrierOption — down/up-and-out (StatefulPricer)
│   └── exotic/asian_option.py     # AsianOption — arithmetic average (StatefulPricer)
├── portfolio/
│   ├── trade.py                   # Trade — binds Pricer to a named model
│   └── agreement.py               # Agreement — ISDA MA scope; aggregate VM, CVA
├── exposure/
│   ├── metrics.py                 # ExposureCalculator: EE, PFE, PSE, EPE
│   ├── netting.py                 # NettingSet
│   ├── bilateral.py               # BilateralExposureCalculator, ISDAExposureCalculator
│   ├── saccr.py                   # SACCRCalculator — Basel III SA-CCR EAD formula
│   ├── margin/vm.py               # REGVMEngine — path-dependent MTA-gated CSB
│   ├── margin/im.py               # REGIMEngine — Schedule IM
│   ├── margin/simm.py             # SimmCalculator — ISDA SIMM IR/Equity delta
│   ├── collateral.py              # CollateralAccount + HaircutSchedule
│   ├── csa.py                     # CSATerms (threshold, MTA, MPOR, IM model)
│   └── streaming/
│       ├── engine.py              # StreamingExposureEngine — step-loop, never full MTM matrix
│       └── vm_stepper.py          # REGVMStepper — per-path CSB state updated each step
├── pipeline/
│   ├── config.py                  # EngineConfig — YAML/dict parser + TradeFactory.register()
│   ├── engine.py                  # RiskEngine — 3-phase pipeline + parallel execution
│   ├── result.py                  # RunResult, AgreementResult, NettingSetSummary
│   └── shared_memory.py           # SimulationSharedMemory — zero-copy paths via SharedMemory
├── examples/
│   ├── single_swap.yaml           # Single IRS with HullWhite1F
│   ├── multi_asset.yaml           # IRS + European equity option, correlated models
│   └── stress_test.yaml           # Three agreements (triggers parallel execution path)
└── backtest/
    ├── engine.py                  # BacktestEngine — PFE exceedances, Kupiec, bias t-test
    └── result.py                  # BacktestResult
```

---

## Market Data

`MarketData` is the central data container. It serves as both the calibration input for
models and the discount/forward provider during pricing. It never mutates — `bump()` and
`scenario()` always return a new copy.

```python
from risk_analytics import MarketData, BumpType, ScenarioBump

md = MarketData.from_dict({
    "curves": {
        "USD_OIS": {
            "tenors": [0.5, 1, 2, 5, 10],
            "rates":  [0.040, 0.042, 0.044, 0.047, 0.050],
            "interpolation": "LOG_LINEAR",   # LINEAR | LOG_LINEAR | CUBIC_SPLINE
        },
    },
    "spots": {"CRUDE_WTI": 80.0, "EURUSD": 1.08},
    "vols":  {"SPX": 0.20},
})

# Accessors
md.discount_factor("USD_OIS", 5.0)      # P(0, 5)
md.zero_rate("USD_OIS", 2.0)            # z(2)
md.forward_rate("USD_OIS", 1.0, 2.0)   # f(1, 2)
md.spot("CRUDE_WTI")                    # 80.0
md.vol("SPX")                           # 0.20

# Stress testing — returns a new MarketData, never mutates
md_up    = md.bump("USD_OIS", 0.001)                          # +10bps parallel
md_tilt  = md.bump("USD_OIS", 0.002, BumpType.SLOPE)          # slope steepener
md_point = md.bump("USD_OIS", 0.001, BumpType.POINT, tenor=5.0)  # 5y point bump
md_spot  = md.bump("CRUDE_WTI", 0.10)                         # +10% spot move

# Multi-factor scenario
md_scenario = md.scenario([
    ScenarioBump("USD_OIS",   0.001,  BumpType.PARALLEL),
    ScenarioBump("CRUDE_WTI", 0.05),
    ScenarioBump("SPX",       0.02),  # vol bump (additive)
])

# Load from YAML file
md = MarketData.from_yaml("market_data.yaml")
```

---

## Sparse Time Grid

`SparseTimeGrid` controls memory usage for long-dated trades. The standard grid uses
daily steps for the first two weeks, weekly for the rest of year one, and monthly
thereafter — roughly 80 nodes for a 5-year deal vs 60 nodes on a uniform monthly grid,
but only ~120 nodes for a 30-year deal vs 360 on a monthly uniform grid.

All known cashflow dates are merged as hard nodes, so interpolation never crosses a
discontinuity in instrument MTM.

```python
from risk_analytics import SparseTimeGrid

# Standard grid: daily 2w → weekly 52w → monthly to maturity
grid = SparseTimeGrid.standard(30.0)    # 30-year deal: ~170 nodes

# Custom anchor points (always includes t=0)
grid = SparseTimeGrid.custom([0.083, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0])

# Merge trade cashflow dates as hard nodes
cf_times = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]   # semi-annual swap
grid = SparseTimeGrid.merge_cashflows(grid, cf_times)
```

### Sparse path interpolation

`SimulationResult` interpolates on demand between sparse nodes. Each model declares its
`interpolation_space` per factor — `"log"` for positive quantities (spots, FX), `"linear"`
for Gaussian quantities (rates, log-spot processes):

| Model | Factor(s) | Space |
|---|---|---|
| `HullWhite1F` | `r` | `["linear"]` |
| `HullWhite2F` | `r`, `u_component` | `["linear", "linear"]` |
| `GBM`, `GarmanKohlhagen` | `S` | `["log"]` |
| `HestonModel` | `S`, `v` | `["log", "linear"]` |
| `Schwartz1F` | `X` (log-spot) | `["linear"]` |
| `Schwartz2F` | `X`, `δ` | `["linear", "linear"]` |

```python
result = simulation_results["HullWhite1F"]

r_at_1y     = result.at(1.0)                           # (n_paths, n_factors)
r_at_tenors = result.at_times(np.array([0.5, 1, 2]))   # (n_paths, 3, n_factors)
```

---

## Stochastic Models

All models implement `StochasticModel` with `simulate()`, `calibrate()`, `get_params()`,
`set_params()`, `save(path)`, and `load(path)`.

| Model | Asset class | SDE |
|---|---|---|
| `HullWhite1F` | Interest rates | `dr = (θ(t) − a·r) dt + σ dW` — exact Gaussian |
| `HullWhite2F` | Interest rates (2F) | `dr = (θ(t)−ar) dt + σ dW₁`, `du = −bu dt + η dW₂` — G2++ |
| `GeometricBrownianMotion` | Equity | `dS = μS dt + σS dW` — exact log-normal |
| `HestonModel` | Equity (stoch vol) | `dS`, `dv` joint — Euler + full truncation |
| `Schwartz1F` | Commodity | `dX = κ(μ − X) dt + σ dW` — exact OU |
| `Schwartz2F` | Commodity | log-spot + convenience yield — exact OU + BM |
| `GarmanKohlhagen` | FX | `dS = (r_d − r_f)S dt + σS dW` — exact log-normal |

### Serialisation

```python
hw.save("hw.json")
hw2 = HullWhite1F().load("hw.json")
```

---

## Two-Engine Architecture

`risk_analytics` has two distinct engine layers that work in sequence:

```
MonteCarloEngine          →   SimulationResult (paths)
                                      │
                    ┌─────────────────┴──────────────────┐
                    ▼                                     ▼
      ISDAExposureCalculator                  StreamingExposureEngine
      (batch — full MTM matrix)               (streaming — O(n_paths) memory)
      Used by RiskEngine pipeline             Used standalone for large portfolios
                    │                                     │
                    └─────────────────┬──────────────────┘
                                      ▼
                            EE / PFE / ENE / CVA profiles
```

`MonteCarloEngine` generates correlated factor paths. Both exposure engines consume those
paths — they differ only in memory strategy. The pipeline always uses the batch engine;
see the [Streaming Exposure Engine](#streaming-exposure-engine) section for when to use
the other.

### MonteCarloEngine (path generation)

```python
from risk_analytics import MonteCarloEngine, SparseTimeGrid

grid   = SparseTimeGrid.standard(5.0)
engine = MonteCarloEngine(n_paths=10_000, seed=42, antithetic=False, quasi_random=False)

results = engine.run(
    models=[hw, gbm, sch],
    time_grid=grid,
    correlation_matrix=corr,   # None → independence
)
# results: dict[str, SimulationResult]
```

`SimulationResult` exposes:
- `.factor(name)` → `(n_paths, T)` — full path at all grid points
- `.at(t)` → `(n_paths, n_factors)` — interpolated at arbitrary time
- `.at_times(ts)` → `(n_paths, len(ts), n_factors)` — batch interpolation

---

## Portfolio Hierarchy

The legal/computation hierarchy distinguishes two levels:

**`Trade`** binds a `Pricer` to the named model whose `SimulationResult` it consumes.
One trade belongs to exactly one `NettingSet`.

**`Agreement`** is the ISDA Master Agreement + CSA scope. It may contain multiple
`NettingSet`s. VM is computed on the **aggregate MTM** across all netting sets (no
per-netting-set floor), because a single CSA governs the combined margin obligation.
CVA/DVA are computed at agreement level on the aggregated expected exposure.

```python
from risk_analytics import Trade, Agreement, NettingSet, CSATerms

# Build trades
swap = InterestRateSwap(fixed_rate=0.045, maturity=5.0, notional=1_000_000, payer=True)
opt  = EuropeanOption(strike=105.0, expiry=2.0, sigma=0.22, risk_free_rate=0.04)

# Netting sets (close-out netting scope)
ns_ir = NettingSet("NS_IR")
ns_ir.add_trade(Trade(id="swap_5y", pricer=swap, model_name="rates_usd"))

ns_eq = NettingSet("NS_EQ")
ns_eq.add_trade(Trade(id="call_2y", pricer=opt,  model_name="equity_spx"))

# Agreement: single CSA covers both netting sets
csa = CSATerms.regvm_standard("Goldman_Sachs", mta=10_000)
agr = Agreement(id="AGR_001", counterparty_id="Goldman_Sachs",
                netting_sets=[ns_ir, ns_eq], csa=csa)

# VM base = sum across netting sets, no floor
agg_mtm = agr.aggregate_mtm(simulation_results)    # (n_paths, T)

# Pre-collateral per netting set
ns_mtms = agr.netting_set_mtms(simulation_results) # dict[ns_id → (n_paths, T)]

# Cashflow times union (used by pipeline to augment sparse grid)
cf_times = agr.all_cashflow_times()
```

---

## Pipeline Engine

`RiskEngine` runs the full end-to-end pipeline from a single YAML or dict config.

**Execution phases:**
1. **Serial** — build `MarketData`, construct sparse grid (standard + all cashflow dates
   merged), instantiate and calibrate models, run `MonteCarloEngine`.
2. **Parallel** (`ProcessPoolExecutor`) — price trades, compute VM/IM/collateral, and
   bilateral exposure independently per `Agreement`. Falls back to serial when ≤ 2
   agreements.
3. **Serial** — aggregate `AgreementResult`s into `RunResult`.

### YAML config

```yaml
simulation:
  n_paths: 10000
  seed: 42
  antithetic: false
  time_grid:
    type: standard        # or "custom" with anchor_points: [...]

market_data:
  curves:
    USD_OIS:
      tenors: [0.5, 1, 2, 5, 10]
      rates:  [0.040, 0.042, 0.044, 0.047, 0.050]
      interpolation: LOG_LINEAR
  spots:
    CRUDE_WTI: 80.0
  vols:
    SPX: 0.20

models:
  - name: rates_usd
    type: HullWhite1F
    params: {a: 0.15, sigma: 0.01, r0: 0.04}
    calibrate_to: USD_OIS          # optional: calibrate theta to this curve
  - name: equity_spx
    type: GBM
    params: {S0: 100.0, mu: 0.06, sigma: 0.20}

correlation:
  - [rates_usd, equity_spx, 0.10]

agreements:
  - id: AGR_001
    counterparty: Goldman_Sachs
    cp_hazard_rate: 0.010
    own_hazard_rate: 0.005
    csa:
      mta: 10000
      threshold: 0
      margin_regime: REGVM
    netting_sets:
      - id: NS_IR
        trades:
          - id: payer_5y
            type: InterestRateSwap
            model: rates_usd
            params: {fixed_rate: 0.045, maturity: 5.0, notional: 1000000, payer: true}
      - id: NS_EQ
        trades:
          - id: call_2y
            type: EuropeanOption
            model: equity_spx
            params: {strike: 105.0, expiry: 2.0, sigma: 0.20,
                     risk_free_rate: 0.04, option_type: call}

outputs:
  metrics: [EE, PFE, CVA]
  confidence: 0.95
  format: parquet
  path: ./results/
  write_raw_paths: false   # true → raw MTM arrays written per agreement
```

### Programmatic usage

```python
from risk_analytics import RiskEngine, MarketData, ScenarioBump, BumpType

# From YAML file
engine = RiskEngine.from_yaml("config.yaml")
result = engine.run()

# From dict (same structure as YAML)
engine = RiskEngine(config_dict)
result = engine.run()

# Override market data at run time (e.g. latest EOD data)
result = engine.run(market_data=live_market_data)
```

### `RunResult`

```python
result.agreement_results          # dict[agreement_id → AgreementResult]
result.total_cva                  # sum of CVA across all agreements
result.summary_df()               # pd.DataFrame — one row per agreement
result.to_dict()                  # plain dict for serialisation
result.to_parquet("./results/")   # summary.parquet, ee_profiles.parquet, pfe_profiles.parquet
```

`AgreementResult` fields:

| Field | Description |
|---|---|
| `ee_profile` / `pfe_profile` / `ene_profile` | Post-collateral profiles `(T,)` |
| `ee_mpor_profile` | MPOR-shifted EE profile `(T,)` |
| `netting_set_summaries` | Pre-collateral EE/PFE/PSE/EPE per netting set |
| `cva` / `dva` / `bcva` | XVA scalars |
| `pse` / `epe` / `eepe` | Exposure scalars |

Supported trade types in YAML: `InterestRateSwap`, `ZeroCouponBond`, `FixedRateBond`,
`EuropeanOption`, `BarrierOption`, `AsianOption` (auto-registered). Register additional
custom types with the `@TradeFactory.register` decorator (see below).

---

## Path-Dependent Instruments (StatefulPricer)

`StatefulPricer` extends `Pricer` for instruments that require accumulating information
along each simulated path — barrier monitoring, Asian averaging, target-redemption
features, etc.

```python
from dataclasses import dataclass
import numpy as np
from risk_analytics.core.stateful import PathState, StatefulPricer

@dataclass
class MyState(PathState):
    running_max: np.ndarray   # (n_paths,)

    @classmethod
    def allocate(cls, n_paths):
        return cls(running_max=np.full(n_paths, -np.inf))

class LookbackCall(StatefulPricer):
    def __init__(self, expiry):
        self.expiry = expiry

    def allocate_state(self, n_paths):
        return MyState.allocate(n_paths)

    def step(self, result, t, t_idx, state):
        S_t = result.factor_at("S", t_idx)
        new_max = np.maximum(state.running_max, S_t)
        mtm = np.maximum(S_t - new_max, 0.0) if t >= self.expiry else np.zeros(len(S_t))
        return mtm, MyState(running_max=new_max)
```

### BarrierOption

`BarrierOption` is a ready-made `StatefulPricer` for European down-and-out / up-and-out
options. The barrier is monitored at each simulation step. For `t < expiry`, `step()`
uses analytical Black-Scholes barrier pricing for pre-expiry MTM (see DESIGN.md §16),
giving smooth EE/PFE profiles. At expiry, it returns the payoff if
the barrier was never breached, otherwise zero.

```python
from risk_analytics.pricing.exotic import BarrierOption

opt = BarrierOption(
    strike=105.0,
    barrier=90.0,
    expiry=1.0,
    barrier_type="down-out",   # or "up-out"
    factor_name="S",
    option_type="call",        # or "put"
    risk_free_rate=0.04,
    sigma=0.20,                # vol for analytical pre-expiry MTM
)
mtm = opt.price(simulation_result)   # (n_paths, T)
```

### AsianOption

`AsianOption` is a `StatefulPricer` for arithmetic-average Asian options. State is
the running sum and count of spot observations; payoff at expiry is
`max(avg(S) − K, 0)`.

```python
from risk_analytics.pricing.exotic import AsianOption

opt = AsianOption(
    strike=100.0,
    expiry=1.0,
    risk_free_rate=0.04,
    factor_name="S",
)
mtm = opt.price(simulation_result)   # (n_paths, T)
```

### price_at()

All `Pricer` subclasses expose `price_at(result, t_idx) -> (n_paths,)` for single-step
MTM. The default delegates to `price(result)[:, t_idx]`. `InterestRateSwap`,
`ZeroCouponBond`, and `FixedRateBond` override it to compute only the requested slice,
avoiding materialising the full `(n_paths, T)` matrix. `StatefulPricer.price_at()` replays
the step loop from t=0 to accumulate state correctly.

---

## Streaming Exposure Engine

Use `StreamingExposureEngine` directly — outside the `RiskEngine` pipeline — when memory
is the binding constraint or when you want fine-grained control over the exposure loop.
It steps through the time grid one step at a time, never holding the full `(n_paths, T)`
MTM matrix. For large simulations (e.g. 100k+ paths), this can reduce memory usage by
several GB compared to the batch approach.

```python
from risk_analytics.exposure.streaming import StreamingExposureEngine
from risk_analytics.exposure.csa import CSATerms

trades = [("payer_5y", swap), ("barrier_call", barrier_opt)]
csa    = CSATerms.regvm_standard("CP_A", mta=10_000)

engine = StreamingExposureEngine(trades, csa, confidence=0.95)
out    = engine.run(simulation_result)

out.ee_profile        # (T,) expected exposure
out.pfe_profile       # (T,) peak future exposure
out.ene_profile       # (T,) expected negative exposure
out.ee_mpor_profile   # (T,) approximate EE under MPOR look-ahead (see DESIGN.md §6)
out.peak_ee           # scalar
out.peak_pfe          # scalar
```

`REGVMStepper` can also be used stand-alone to test margining logic:

```python
from risk_analytics.exposure.streaming import REGVMStepper

stepper = REGVMStepper(csa, n_paths=1000)
for t_idx, net_mtm_t in enumerate(net_mtm_steps):   # (n_paths,) slices
    post_margin_exposure = stepper.step(net_mtm_t)
```

---

## Custom Instrument Types

`BarrierOption` and `AsianOption` are **auto-registered** on import — no decorator call
required. Use them directly in YAML configs:

```yaml
- id: barrier_1
  type: BarrierOption
  model: equity_spx
  params: {strike: 100.0, barrier: 80.0, expiry: 1.0, risk_free_rate: 0.04}

- id: asian_1
  type: AsianOption
  model: equity_spx
  params: {strike: 100.0, expiry: 1.0, risk_free_rate: 0.04}
```

Register additional trade types with the `@TradeFactory.register` decorator:

```python
from risk_analytics.pipeline.config import TradeFactory

@TradeFactory.register("LookbackCall")
def _build_lookback(params):
    return LookbackCall(expiry=params["expiry"])
```

Then use `type: LookbackCall` in the YAML `trades` list as normal.

---

## Shared Memory for Parallel Workers

`SimulationSharedMemory` allocates one named `SharedMemory` block per model result and
exposes lightweight descriptors that worker processes can use to attach numpy views without
copying data:

```python
from risk_analytics.pipeline.shared_memory import SimulationSharedMemory

with SimulationSharedMemory(simulation_results) as shm:
    desc = shm.descriptors   # picklable — safe to send to ProcessPoolExecutor

    def worker(descriptors, agreement):
        attached = SimulationSharedMemory.attach(descriptors)
        results  = SimulationSharedMemory.results_from_attached(attached)
        try:
            return compute_exposure(agreement, results)
        finally:
            SimulationSharedMemory.detach(attached)

    with ProcessPoolExecutor() as ex:
        futures = [ex.submit(worker, desc, agr) for agr in agreements]
```

All blocks are unlinked automatically on `__exit__`.

---

## Stress Testing

`RunResult.stress_test()` reprices on the **existing simulation paths** — no
re-simulation. This is cheap enough for interactive sensitivity analysis.

```python
from risk_analytics import ScenarioBump, BumpType

# +25bps parallel shift on USD rates
stressed = result.stress_test(
    bumps=[ScenarioBump("USD_OIS", 0.0025, BumpType.PARALLEL)],
    market_data=base_market_data,
)

print(f"CVA delta: {stressed.total_cva - result.total_cva:+,.0f}")

# Multi-factor scenario
stressed = result.stress_test(
    bumps=[
        ScenarioBump("USD_OIS",   0.001,  BumpType.PARALLEL),
        ScenarioBump("CRUDE_WTI", 0.10),   # +10% spot
        ScenarioBump("SPX",       0.05),   # +5 vol points
    ],
    market_data=base_market_data,
)
```

---

## SA-CCR (Regulatory Capital)

`SACCRCalculator` computes the Basel III Standardised Approach for Counterparty Credit
Risk EAD formula: **EAD = 1.4 × (RC + PFE add-on)**.

```python
from risk_analytics.exposure.saccr import SACCRCalculator, SACCRTrade

calc = SACCRCalculator()
calc.add_trade(SACCRTrade(
    trade_id="swap_5y",
    asset_class="ir",         # "ir", "equity_single", "equity_index", "fx", ...
    notional=10_000_000,
    maturity=5.0,
    current_mtm=50_000,
    delta=1.0,                # +1 receiver / -1 payer; BS delta for options
))
calc.add_trade(SACCRTrade("opt_2y", "equity_single", 500_000, 2.0, -10_000, 0.6))

print(f"RC:        {calc.replacement_cost():,.0f}")
print(f"PFE addon: {calc.pfe_addon():,.0f}")
print(f"EAD:       {calc.ead():,.0f}")
```

Build directly from pipeline `Trade` objects:

```python
calc = SACCRCalculator.from_trades(netting_set.trades, current_mtm={"swap_5y": 50_000})
ead = calc.ead()
```

IR supervisory factors (Basel III CRE52): < 1Y: 0.20% / 1–5Y: 0.50% / 5–10Y: 1.00% / > 10Y: 1.50%.
Equity single-name: 32% / Equity index: 20% / FX: 4%.

---

## Example Configs

Ready-to-run YAML configs live in `examples/`. Run with:

```bash
uv run python -c "
from risk_analytics import RiskEngine
result = RiskEngine.from_yaml('examples/single_swap.yaml').run()
print(result.summary_df())
"
```

| File | Description |
|---|---|
| `examples/single_swap.yaml` | Single 5-year payer IRS, HullWhite1F calibrated to USD OIS, 10k paths |
| `examples/multi_asset.yaml` | IRS + European equity call, correlated HW1F + GBM, 10k paths |
| `examples/stress_test.yaml` | Three agreements triggering the parallel execution path, 50k paths |

---

## Exposure Metrics

### Basic (uncollateralised)

```python
from risk_analytics import ExposureCalculator, NettingSet

calc = ExposureCalculator()
summary = calc.exposure_summary(mtm, time_grid, confidence=0.95)
# keys: ee_profile, pfe_profile, pse, epe

ns = NettingSet("Counterparty_A")
ns.add_trade("payer_5y", payer_swap)
net_mtm = ns.net_mtm(results)
```

### Bilateral (ISDA/regulatory)

```python
from risk_analytics import CSATerms, ISDAExposureCalculator

csa  = CSATerms.regvm_standard("Counterparty_A", mta=10_000)
isda = ISDAExposureCalculator(ns, csa)
out  = isda.run(results, time_grid, confidence=0.95,
                cp_hazard_rate=0.008, own_hazard_rate=0.004)
# out keys: ee, ene, pfe, ee_coll, ee_mpor, pse, epe, eepe,
#           cva, dva, bcva, net_mtm, csb, lagged_csb, im, collateral
```

---

## Regulatory Initial Margin

```python
from risk_analytics import REGIMEngine, CSATerms, IMModel, SimmSensitivities
from risk_analytics.exposure import SimmCalculator

# Schedule IM
im_engine = REGIMEngine(CSATerms(im_model=IMModel.SCHEDULE))
schedule_im = im_engine.schedule_im(trades=[
    {"asset_class": "IR", "gross_notional": 1_000_000, "maturity": 5.0,
     "net_replacement_cost": 8_000},
])

# SIMM
sens    = SimmSensitivities(ir={"USD": {"1y": 200.0, "5y": 800.0}}, equity={})
simm_im = SimmCalculator().total_im(sens)
```

---

## Day-Count Conventions and Schedules

```python
from risk_analytics.core import (
    DayCountConvention, BusinessDayConvention,
    NullCalendar, TARGET, USCalendar,
    Frequency, Schedule,
)
from datetime import date

yf = DayCountConvention.ACT_ACT_ISDA.year_fraction(date(2024, 7, 1), date(2025, 7, 1))

sched = Schedule.from_dates(
    date(2024, 1, 1), date(2029, 1, 1),
    Frequency.SEMI_ANNUAL,
    calendar=TARGET(),
    day_count=DayCountConvention.ACT_360,
    bdc=BusinessDayConvention.MODIFIED_FOLLOWING,
)
```

Supported conventions: `ACT_360`, `ACT_365`, `ACT_ACT_ISDA`, `THIRTY_360`, `THIRTY_E_360`.
Calendars: `NullCalendar` (weekends only), `TARGET` (ECB), `USCalendar` (Federal).

---

## Backtesting

`BacktestEngine` is model-agnostic: it accepts any MTM forecast distribution and a
realised MTM series.

```python
from risk_analytics import BacktestEngine

bt     = BacktestEngine(confidence=0.95)
result = bt.run(forecast_mtm, realized_mtm, time_grid)
s      = result.summary()

print(f"Exceptions:     {s['n_exceptions']}/{s['n_observations']}")
print(f"Basel zone:     {s['basel_zone']}")
print(f"Kupiec p-value: {s['kupiec_pvalue']:.3f}")
print(f"EE bias:        {s['ee_bias']:,.0f}")
```

| Key | Description |
|---|---|
| `n_exceptions` / `exception_rate` | PFE exceedance count and rate |
| `basel_zone` | "Green" / "Amber" / "Red" (scaled to 250-obs equivalent) |
| `kupiec_lr` / `kupiec_pvalue` | Kupiec POF test — exception frequency vs model confidence |
| `ee_rmse` / `ee_bias` / `ee_mae` | EE forecast accuracy vs realised MTM |
| `bias_tstat` / `bias_pvalue` | t-test for H₀: mean(EE − realized) = 0 |

---

## Logging

```python
import logging
logging.basicConfig(level=logging.INFO, format="%(name)s — %(levelname)s — %(message)s")
```

Use `level=logging.DEBUG` for per-step internals. No output by default (library-friendly).

---

## Run the demo

```bash
uv run risk-analytics-demo
# or: uv run python -m risk_analytics.demo
# or: uv run python demo.py
```

The demo covers all features end-to-end: MarketData construction and stress bumps,
sparse grid, library-API exposure workflow, and the pipeline engine with a two-agreement
config including stress testing.
