# TODO / Improvements

Tracked improvements and known gaps in the `risk_analytics` library.
Items are grouped by area; priority labels: `[P1]` urgent / `[P2]` important / `[P3]` nice-to-have.

---

## Pipeline

- [ ] **[P1] Wire `SimulationSharedMemory` into `RiskEngine` parallel path.**
  `_run_parallel()` currently pickles the full `simulation_results` dict into each
  worker process.  Replace with `SimulationSharedMemory` to avoid the O(workers × data)
  memory blow-up described in DESIGN.md §9.

- [x] **[P1] Remove dead code: `_make_aggregate_ns()` in `pipeline/engine.py`.**
  The function raises `NotImplementedError` and is no longer referenced.

- [ ] **[P2] `pipeline/persistence.py` — lazy RunResult loading.**
  `RunResult.to_parquet()` exists but there is no `RunResult.from_parquet()`.  A lazy
  implementation should memory-map per-agreement profile arrays and load scalars eagerly.
  DuckDB can be used for querying across runs.

- [x] **[P2] Support `StatefulPricer` instruments inside `RiskEngine`.**
  `NettingSet._price_trade()` calls `pricer.price(result)` which is correct for both
  plain and stateful pricers, but it tries all models in order until one succeeds.  For
  stateful pricers this is unnecessarily expensive and can produce wrong results if an
  incompatible model "succeeds" by accident.  `Trade.model_name` should be used to look
  up the right `SimulationResult` directly (see netting set improvement below).

- [ ] **[P3] `TradeFactory`: auto-register `BarrierOption` and other built-in exotics.**
  Users currently have to call `@TradeFactory.register("BarrierOption")` manually.
  The built-in exotic pricers should be registered by default on import.

---

## Exposure / Netting

- [x] **[P1] `NettingSet._price_trade()` model lookup is fragile.**
  It iterates all models until one doesn't raise, so the first model whose factor names
  happen to match wins — even if it is the wrong model.  Fix: `NettingSet` should
  store trades as `Trade` objects (not `(str, Pricer)` tuples) so that `trade.model_name`
  can be used to retrieve the correct `SimulationResult` directly.

- [ ] **[P2] `StreamingExposureEngine` MPOR calculation is a proxy.**
  The current `ee_mpor_profile` is `E[max(V(t) - V(t - mpor), 0)]` as a rough stand-in.
  A proper implementation replays the CSB from the last successful margin call at
  `t - mpor`, matching the Basel SA-CCR / FRTB definition.

- [ ] **[P2] `REGVMStepper` rounding convention.**
  The current implementation rounds to the nearest multiple.  ISDA 2016 VM CSA specifies
  that delivery amounts round *up* and return amounts round *down*.  Fix the rounding
  branches accordingly.

- [ ] **[P3] `StreamingExposureEngine`: support multi-model netting sets.**
  Currently each trade is priced via a single `SimulationResult` passed to `run()`.
  Support a `results: dict` argument so trades backed by different models (rates +
  equity in the same netting set) can be priced correctly.

---

## Pricing

- [ ] **[P2] `EuropeanOption.price_at()` override.**
  `EuropeanOption` currently falls back to the default `price(result)[:, t_idx]`.
  An efficient override computing only the `(n_paths,)` Black-Scholes MTM from the spot
  slice would be consistent with the IRS/bond overrides.

- [ ] **[P2] `BarrierOption` pre-expiry MTM.**
  Currently `step()` returns 0 for `t < expiry`.  A more useful implementation would
  return the Black-Scholes barrier option price at `t` (analytical formula exists),
  making the MTM profile smooth and usable for EE/PFE pre-expiry.

- [ ] **[P3] Asian option pricer.**
  Good second example of `StatefulPricer` (state = running arithmetic average).
  Payoff = `max(avg(S) - K, 0)` at expiry; interim MTM via Monte Carlo sub-simulation
  or control-variate approximation.

---

## Models

- [ ] **[P2] Verify `HestonModel.interpolation_space` returns `["log", "linear"]`.**
  The variance factor `v` should be interpolated in linear space (it is mean-reverting
  and can approach zero), while the spot `S` should be log-space.  Confirm this is
  correctly declared and tested.

- [ ] **[P3] Multi-curve Hull-White (HullWhite2F).**
  Second mean-reverting factor captures a richer term structure of interest rate
  volatility.  Useful for swaption calibration and longer-dated exposure.

- [ ] **[P3] Local volatility / SABR for FX/equity.**
  Current GBM and GK models use constant volatility.  A local vol surface would
  improve calibration accuracy for vanilla option portfolios.

---

## Core

- [ ] **[P2] `SimulationResult.at(t)` boundary handling.**
  When `t > max(time_grid)`, `at()` currently clamps to the last node silently.
  It should raise or return zeros with a warning, since extrapolating beyond maturity
  is almost always a logic error.

- [ ] **[P3] `SparseTimeGrid.dt()` method.**
  Return a `(T-1,)` array of step sizes; useful for trapezoidal integration in EPE
  calculations inside pricers without re-differencing `time_grid`.

---

## Testing

- [x] **[P2] Integration test: `RiskEngine` end-to-end with `StatefulPricer`.**
  Add a `BarrierOption` trade to a YAML config and run the full pipeline to confirm
  `NettingSet` + `Agreement` + `RiskEngine` handle stateful pricers correctly.

- [x] **[P2] Property-based tests for `REGVMStepper`.**
  CSB should be non-negative when `threshold=0`, `mta=0`; post-margin CE should equal 0
  when `V(t) > 0`; CSB should never decrease below `ia_counterparty - ia_party`.

- [x] **[P2] Regression tests for `_discount_factors()` against known Hull-White prices.**
  ZCB prices computed via the affine formula should match the closed-form
  `P(0,T) * exp(B*f(0,t) - ...)` expression to machine precision.

- [ ] **[P3] Benchmark: streaming vs batch EE profiles match to <1bp.**
  `StreamingExposureEngine` and `ISDAExposureCalculator` should produce identical (up to
  float precision) EE profiles for plain vanilla swaps with zero threshold/MTA.

- [ ] **[P2] `calibrate_to` missing curve should raise, not warn.**
  When a model config's `calibrate_to` key references a curve not present in
  `MarketData`, `pipeline/engine.py` logs a warning and continues with an uncalibrated
  model. An uncalibrated Hull-White model produces systematically wrong CVA/EE profiles.
  Fix: raise `ValueError` with a clear message identifying the missing curve name.
  Add a test that asserts the error is raised (vs. silently proceeding).

---

## Documentation / DX

- [ ] **[P3] Example YAML configs in `examples/`.**
  Ship a `single_swap.yaml`, `multi_asset.yaml`, and `stress_test.yaml` that users can
  run immediately with `uv run risk-analytics-demo`.

- [ ] **[P3] Changelog (`CHANGELOG.md`).**
  Track breaking changes and notable additions per version so library consumers know
  what to expect on upgrade.
