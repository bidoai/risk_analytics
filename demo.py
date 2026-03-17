"""
risk_analytics — End-to-End Demo
=================================
Demonstrates the complete pipeline:
  1.  Model setup and calibration
  2.  Monte Carlo simulation (correlated multi-asset)
  3.  Pricing on simulated paths
  4.  Basic exposure metrics (PSE / EPE / PFE)
  5.  ISDA bilateral exposure with CSA, VM, IM, collateral
  6.  CVA / DVA / BCVA

Run with:  uv run python demo.py
"""

import numpy as np

from risk_analytics import (
    # Core
    MonteCarloEngine,
    TimeGrid,
    # Models
    HullWhite1F,
    GeometricBrownianMotion,
    Schwartz1F,
    # Pricing
    InterestRateSwap,
    EuropeanOption,
    # Exposure — basic
    ExposureCalculator,
    NettingSet,
    # Exposure — ISDA/bilateral
    CSATerms,
    IMModel,
    REGIMEngine,
    SimmSensitivities,
    BilateralExposureCalculator,
    ISDAExposureCalculator,
)

np.set_printoptions(precision=4, suppress=True)
SEP = "─" * 64


def section(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


# ===========================================================================
# 1. SIMULATION SETUP
# ===========================================================================
section("1. Model Setup & Calibration")

# --- Interest Rate: Hull-White 1F ---
hw = HullWhite1F(a=0.15, sigma=0.010, r0=0.04)

# Calibrate theta to a simple upward-sloping zero curve
tenors = np.array([0.5, 1, 2, 3, 5, 7, 10])
zero_rates = np.array([0.038, 0.040, 0.042, 0.044, 0.047, 0.050, 0.053])
grid = TimeGrid.uniform(5.0, 60)  # 5-year horizon, monthly steps
hw.calibrate({"tenors": tenors, "zero_rates": zero_rates, "time_grid": grid})

print(f"Hull-White 1F params: {hw.get_params()}")

# --- Equity: GBM ---
gbm = GeometricBrownianMotion(S0=100.0, mu=0.06, sigma=0.22)
gbm.calibrate({"S0": 100.0, "atm_vol": 0.22, "mu": 0.06})
print(f"GBM params:           {gbm.get_params()}")

# --- Commodity: Schwartz 1F ---
sch = Schwartz1F(S0=80.0, kappa=1.2, mu=np.log(80), sigma=0.35)
sch.calibrate({
    "S0": 80.0,
    "hist_vol": 0.35,
    "forward_prices": np.array([81, 83, 86, 88]),
    "forward_tenors": np.array([0.5, 1.0, 2.0, 3.0]),
})
print(f"Schwartz 1F params:   {sch.get_params()}")


# ===========================================================================
# 2. CORRELATED MONTE CARLO SIMULATION
# ===========================================================================
section("2. Correlated Multi-Asset Monte Carlo")

# 3 models × 1 factor each = 3 total factors: r, S_equity, S_commodity
# Specify a correlation matrix (IR × Equity × Commodity)
corr = np.array([
    [1.00, 0.10, 0.15],   # IR  vs IR, Equity, Commodity
    [0.10, 1.00, 0.25],   # Equity vs ...
    [0.15, 0.25, 1.00],   # Commodity vs ...
])

engine = MonteCarloEngine(n_paths=10_000, seed=42)
results = engine.run(
    models=[hw, gbm, sch],
    time_grid=grid,
    correlation_matrix=corr,
)

for name, res in results.items():
    print(f"  {name:15s}  paths={res.n_paths}  steps={res.n_steps}  "
          f"factors={res.factor_names}")


# ===========================================================================
# 3. PRICING ON SIMULATED PATHS
# ===========================================================================
section("3. Instrument Pricing on Simulated Paths")

# Payer IRS: pay fixed 4.5%, receive floating, 5yr, 1M notional
payer_swap = InterestRateSwap(
    fixed_rate=0.045, maturity=5.0, notional=1_000_000, payer=True
)
payer_mtm = payer_swap.price(results["HullWhite1F"])     # (10_000, 61)

# Receiver IRS: receive fixed 3.5%, pay floating, 3yr, 500K notional
recv_swap = InterestRateSwap(
    fixed_rate=0.035, maturity=3.0, notional=500_000, payer=False
)
recv_mtm = recv_swap.price(results["HullWhite1F"])

# European call option on equity
call = EuropeanOption(
    strike=105.0, expiry=2.0, sigma=0.22, risk_free_rate=0.04, option_type="call"
)
call_mtm = call.price(results["GBM"])                     # (10_000, 61)

print(f"  Payer IRS   t=0 mean MTM: {payer_mtm[:,0].mean():>12,.0f}")
print(f"  Receiver IRS t=0 mean MTM: {recv_mtm[:,0].mean():>11,.0f}")
bs_ref = EuropeanOption.black_scholes_price(100, 105, 2.0, 0.04, 0.22, "call")
print(f"  Call option t=0 MC price: {call_mtm[:,0].mean():>12.4f}  (BS={bs_ref:.4f})")


# ===========================================================================
# 4. BASIC EXPOSURE METRICS (UNCOLLATERALISED)
# ===========================================================================
section("4. Basic Uncollateralised Exposure Metrics")

calc = ExposureCalculator()
summary = calc.exposure_summary(payer_mtm, grid, confidence=0.95)

print(f"  PSE (Peak Simulated Exposure):   {summary['pse']:>12,.0f}")
print(f"  EPE (Expected Positive Exposure):{summary['epe']:>12,.0f}")
print(f"  PFE 95th pct at 2yr:             "
      f"{summary['pfe_profile'][np.searchsorted(grid, 2.0)]:>12,.0f}")
print(f"  EE  profile shape:               {summary['ee_profile'].shape}")


# ===========================================================================
# 5. NETTING SET
# ===========================================================================
section("5. Netting Set (IRS Payer + IRS Receiver)")

ns = NettingSet("Counterparty_A")
ns.add_trade("payer_5y", payer_swap)
ns.add_trade("receiver_3y", recv_swap)

net_mtm = ns.net_mtm(results)                         # (10_000, 61)
ns_summary = ns.exposure(results, grid, confidence=0.95)

gross_ee = (
    calc.expected_exposure(payer_mtm)
    + calc.expected_exposure(recv_mtm)
)
net_ee = ns_summary["ee_profile"]
netting_benefit = 1.0 - net_ee.mean() / gross_ee.mean()

print(f"  Gross EE (sum of trades):  {gross_ee.mean():>10,.0f}")
print(f"  Net EE (netting set):      {net_ee.mean():>10,.0f}")
print(f"  Netting benefit:           {netting_benefit:>10.1%}")
print(f"  Net PSE:                   {ns_summary['pse']:>10,.0f}")
print(f"  Net EPE:                   {ns_summary['epe']:>10,.0f}")


# ===========================================================================
# 6. ISDA BILATERAL EXPOSURE WITH CSA
# ===========================================================================
section("6. ISDA Bilateral Exposure — REGVM Zero-Threshold CSA")

csa_regvm = CSATerms.regvm_standard("Counterparty_A", mta=10_000)
isda_calc = ISDAExposureCalculator(ns, csa_regvm)

out_regvm = isda_calc.run(
    results, grid,
    confidence=0.95,
    cp_hazard_rate=0.008,   # 80 bps CDS spread / LGD≈60% → λ≈0.013
    own_hazard_rate=0.004,
)

print(f"  EE  (uncoll):    {out_regvm['ee'].mean():>12,.0f}")
print(f"  EE  (coll+MPOR): {out_regvm['ee_coll'].mean():>12,.0f}")
print(f"  ENE:             {out_regvm['ene'].mean():>12,.0f}")
print(f"  EEPE (reg cap):  {out_regvm['eepe']:>12,.0f}")
print(f"  PSE:             {out_regvm['pse']:>12,.0f}")
print(f"  EPE:             {out_regvm['epe']:>12,.0f}")
print(f"  CVA:             {out_regvm['cva']:>12,.0f}")
print(f"  DVA:             {out_regvm['dva']:>12,.0f}")
print(f"  BCVA (CVA-DVA):  {out_regvm['bcva']:>12,.0f}")

# -----------------------------------------------------------------
section("6b. Legacy CSA — Symmetric Threshold (50K per side)")

csa_legacy = CSATerms.legacy_bilateral("Counterparty_A", threshold=50_000, mta=10_000)
out_legacy = ISDAExposureCalculator(ns, csa_legacy).run(
    results, grid, confidence=0.95
)

print(f"  EE  (coll+MPOR, REGVM):  {out_regvm['ee_coll'].mean():>12,.0f}")
print(f"  EE  (coll+MPOR, Legacy): {out_legacy['ee_coll'].mean():>12,.0f}")
print(f"  → Legacy threshold adds EE: "
      f"{out_legacy['ee_coll'].mean() - out_regvm['ee_coll'].mean():>10,.0f}")


# ===========================================================================
# 7. REGULATORY IM — SCHEDULE AND SIMM
# ===========================================================================
section("7. Initial Margin — Schedule vs SIMM")

# Schedule IM
csa_im = CSATerms(im_model=IMModel.SCHEDULE)
im_engine = REGIMEngine(csa_im)
trades_descriptor = [
    {
        "asset_class": "IR",
        "gross_notional": 1_000_000,
        "maturity": 5.0,
        "net_replacement_cost": float(payer_mtm[:, 0].mean()),
    },
    {
        "asset_class": "IR",
        "gross_notional": 500_000,
        "maturity": 3.0,
        "net_replacement_cost": float(recv_mtm[:, 0].mean()),
    },
]
schedule_im = im_engine.schedule_im(trades_descriptor)
print(f"  Schedule IM (netting set): {float(schedule_im):>12,.0f}")

# SIMM IM
from risk_analytics.exposure import SimmCalculator
simm_calc = SimmCalculator()
sens = SimmSensitivities(
    ir={
        "USD": {
            "1y": 200.0,
            "2y": 350.0,
            "5y": 800.0,
        }
    },
    equity={},  # no equity sensitivity (equity option not in netting set here)
)
simm_im = float(simm_calc.total_im(sens))
print(f"  SIMM IM (IR delta only):   {simm_im:>12,.2f}")

# ISDAExposureCalculator with IM
csa_with_im = CSATerms.regvm_standard("CP", mta=10_000)
csa_with_im.im_model = IMModel.SCHEDULE
im_engine2 = REGIMEngine(csa_with_im)
out_with_im = ISDAExposureCalculator(ns, csa_with_im, im_engine=im_engine2).run(
    results, grid, im_trades=trades_descriptor, confidence=0.95
)
print(f"\n  EE_coll (VM only):         {out_regvm['ee_coll'].mean():>12,.0f}")
print(f"  EE_coll (VM + Schedule IM):{out_with_im['ee_coll'].mean():>12,.0f}")
reduction = out_regvm['ee_coll'].mean() - out_with_im['ee_coll'].mean()
print(f"  IM benefit to EE:          {reduction:>12,.0f}")


# ===========================================================================
# 8. SUMMARY TABLE
# ===========================================================================
section("8. Summary: Exposure Profile at Selected Tenors")

tenors_out = [0.5, 1.0, 2.0, 3.0, 5.0]
indices = [np.searchsorted(grid, t) for t in tenors_out]

print(f"\n  {'Tenor':>6}  {'EE_uncoll':>12}  {'EE_coll(VM)':>13}  "
      f"{'ENE':>10}  {'PFE 95%':>10}")
print("  " + "─" * 60)
for t, i in zip(tenors_out, indices):
    ee_u = out_regvm["ee"][i]
    ee_c = out_regvm["ee_coll"][i]
    ene  = out_regvm["ene"][i]
    pfe  = out_regvm["pfe"][i]
    print(f"  {t:>6.1f}  {ee_u:>12,.0f}  {ee_c:>13,.0f}  "
          f"{ene:>10,.0f}  {pfe:>10,.0f}")

print(f"\n{SEP}")
print("  Demo complete.")
print(SEP)
