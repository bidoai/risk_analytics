# Changelog

All notable changes to `risk-analytics` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.0.0] — 2026-03-18

### Added

- **HullWhite2F** two-factor Hull-White (G2++) short rate model
  (`src/risk_analytics/models/rates/hull_white2f.py`). Compatible with
  existing rate pricers via the combined short-rate factor "r" = r(t)+u(t).

- **AsianOption** arithmetic average `StatefulPricer`
  (`src/risk_analytics/pricing/exotic/asian_option.py`). Payoff at expiry:
  `max(avg(S) - K, 0)` where avg is the arithmetic mean over all simulation
  steps up to expiry.

- **SA-CCR calculator** (`src/risk_analytics/exposure/saccr.py`). Formula-
  based Basel III Standardised Approach for measuring counterparty credit risk
  EAD. Supports IR, equity, and FX asset classes.

- **TradeFactory auto-registration**: `BarrierOption` and `AsianOption` are
  now automatically registered on import — no manual `@TradeFactory.register`
  call required.

- **BarrierOption pre-expiry analytical MTM**: `step()` now returns the
  Black-Scholes barrier option price (down-and-out / up-and-out, call / put)
  for `t < expiry`, making EE/PFE profiles smooth and informative pre-expiry.

- **SharedMemory in parallel execution**: `_run_parallel()` in the pipeline
  engine now uses `SimulationSharedMemory` to share simulation paths across
  worker processes via OS shared memory, avoiding O(workers × data) memory
  duplication from pickling.

- **Example YAML configs** in `examples/`:
  - `single_swap.yaml` — single IRS with HullWhite1F
  - `multi_asset.yaml` — IRS + European equity option with correlation
  - `stress_test.yaml` — three agreements triggering the parallel execution path

- **Streaming vs batch EE parity test**: `tests/test_streaming_exposure.py`
  now includes `test_streaming_batch_ee_parity` which asserts that
  `StreamingExposureEngine` and `ISDAExposureCalculator` produce matching EE
  profiles within 1bp for a plain vanilla swap with zero threshold/MTA.

### Changed

- `pyproject.toml` version bumped to `1.0.0`.
- `pyproject.toml` description updated from placeholder to meaningful text.

### Exports

- `HullWhite2F` added to `risk_analytics.models`.
- `AsianOption` added to `risk_analytics.pricing`.
- `SACCRCalculator` added to `risk_analytics.exposure`.

---

## [0.1.0] — initial release

Initial implementation including:
- `MonteCarloEngine` with antithetic and quasi-random variance reduction
- `HullWhite1F`, `GeometricBrownianMotion`, `HestonModel`, `Schwartz1F/2F`, `GarmanKohlhagen`
- `InterestRateSwap`, `ZeroCouponBond`, `FixedRateBond`, `EuropeanOption`
- `BarrierOption` (StatefulPricer) with knock-out monitoring
- `NettingSet`, `Agreement`, `ISDAExposureCalculator`
- `REGVMEngine`, `REGIMEngine` (Schedule + SIMM)
- `StreamingExposureEngine` for memory-efficient path-by-step exposure
- `SparseTimeGrid` with cashflow-date merging
- `SimulationSharedMemory` for zero-copy cross-process sharing
- Full `RiskEngine` pipeline with YAML config support
- CVA, DVA, bCVA, EE, ENE, PFE, EPE, EEPE metrics
