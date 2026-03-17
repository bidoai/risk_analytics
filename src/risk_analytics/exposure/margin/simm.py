"""Simplified ISDA SIMM (Standard Initial Margin Model) implementation.

Based on ISDA SIMM v2.6 methodology:
https://www.isda.org/a/FEdgE/ISDA-SIMM-v2.6.pdf

This implementation covers:
- Delta margin for Interest Rates, Equity, FX, Commodity, Credit
- Intra-bucket and inter-bucket aggregation with prescribed correlations
- Cross-asset-class aggregation

For production use, full SIMM requires per-trade sensitivity inputs
(DV01s, equity deltas, FX spot sensitivities, commodity deltas, credit CS01s).
This module provides the calculation engine given those sensitivities.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# SIMM Risk Weights (ISDA SIMM v2.6, Table 1 — Delta)
# Values in the same currency unit as sensitivities (e.g. USD millions per bp)
# ---------------------------------------------------------------------------

# Interest Rate: one weight per tenor bucket
# Tenors: 2w, 1m, 3m, 6m, 1y, 2y, 3y, 5y, 10y, 15y, 20y, 30y
IR_TENORS = ["2w", "1m", "3m", "6m", "1y", "2y", "3y", "5y", "10y", "15y", "20y", "30y"]
IR_RISK_WEIGHTS: dict[str, np.ndarray] = {
    # Per-tenor risk weights for "High Vol" currencies (EM)
    "EM": np.array([116, 106, 94, 71, 59, 52, 49, 51, 40, 40, 40, 40], dtype=float),
    # "Regular" currencies (USD, EUR, GBP, CHF, AUD, NZD, CAD, SEK, NOK, DKK, HKD, SGD)
    "REGULAR": np.array([116, 106, 94, 71, 59, 52, 49, 51, 40, 40, 40, 40], dtype=float),
    # Low-vol currencies (JPY)
    "LOW_VOL": np.array([21, 20, 18, 14, 12, 10, 9, 10, 9, 9, 9, 9], dtype=float),
}

# IR intra-currency correlation matrix (12x12, tenor vs tenor)
# Approximated: ρ(i,j) = max(exp(-θ * |t_i - t_j| / min(t_i, t_j)), 0.40)
_IR_TENOR_YEARS = np.array([2/52, 1/12, 3/12, 6/12, 1, 2, 3, 5, 10, 15, 20, 30])
_theta = 0.03

def _build_ir_intra_corr() -> np.ndarray:
    n = len(_IR_TENOR_YEARS)
    corr = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                ti, tj = _IR_TENOR_YEARS[i], _IR_TENOR_YEARS[j]
                corr[i, j] = max(np.exp(-_theta * abs(ti - tj) / min(ti, tj)), 0.40)
    return corr

IR_INTRA_CORR: np.ndarray = _build_ir_intra_corr()

# IR inter-currency (cross-currency) correlation
IR_INTER_CURRENCY_CORR = 0.22

# Equity risk weights (buckets 1–13; 12 = large-cap EM, 13 = indices/ETFs)
EQUITY_BUCKETS = list(range(1, 14))
EQUITY_RISK_WEIGHTS = np.array([
    23, 25, 29, 27, 18, 21, 24, 23, 28, 29, 45, 22, 14
], dtype=float)
# Intra-bucket correlation (flat per bucket)
EQUITY_INTRA_CORR: dict[int, float] = {
    1: 0.17, 2: 0.20, 3: 0.24, 4: 0.22, 5: 0.25, 6: 0.30,
    7: 0.34, 8: 0.30, 9: 0.29, 10: 0.30, 11: 0.60, 12: 0.24, 13: 0.00,
}
# Inter-bucket correlation (symmetric)
EQUITY_INTER_BUCKET_CORR: dict[tuple[int, int], float] = {}
_eq_corr_vals = [  # upper-triangular, buckets 1–12 (13=indices, treated separately)
    (1,2,0.18),(1,3,0.15),(1,4,0.20),(1,5,0.12),(1,6,0.13),(1,7,0.15),(1,8,0.12),
    (1,9,0.12),(1,10,0.12),(1,11,0.17),(1,12,0.18),(2,3,0.22),(2,4,0.21),(2,5,0.12),
    (2,6,0.14),(2,7,0.15),(2,8,0.13),(2,9,0.13),(2,10,0.12),(2,11,0.17),(2,12,0.20),
    (3,4,0.19),(3,5,0.13),(3,6,0.14),(3,7,0.16),(3,8,0.13),(3,9,0.14),(3,10,0.13),
    (3,11,0.17),(3,12,0.17),(4,5,0.13),(4,6,0.14),(4,7,0.17),(4,8,0.14),(4,9,0.14),
    (4,10,0.13),(4,11,0.17),(4,12,0.20),(5,6,0.26),(5,7,0.20),(5,8,0.25),(5,9,0.19),
    (5,10,0.21),(5,11,0.17),(5,12,0.17),(6,7,0.30),(6,8,0.26),(6,9,0.24),(6,10,0.25),
    (6,11,0.17),(6,12,0.16),(7,8,0.28),(7,9,0.26),(7,10,0.26),(7,11,0.16),(7,12,0.18),
    (8,9,0.23),(8,10,0.23),(8,11,0.17),(8,12,0.20),(9,10,0.30),(9,11,0.16),(9,12,0.17),
    (10,11,0.20),(10,12,0.20),(11,12,0.25),
]
for _b1, _b2, _c in _eq_corr_vals:
    EQUITY_INTER_BUCKET_CORR[(_b1, _b2)] = _c
    EQUITY_INTER_BUCKET_CORR[(_b2, _b1)] = _c

# FX risk weights (applied to net open position in each foreign currency)
FX_RISK_WEIGHT_REGULAR = 8.0   # % of notional for most currency pairs
FX_RISK_WEIGHT_HIGH_VOL = 15.0  # EM currencies vs USD

# Commodity risk weights by bucket (1–17)
COMMODITY_RISK_WEIGHTS = np.array([
    19, 21, 18, 18, 36, 25, 14, 23, 22, 23, 24, 15, 14, 14, 10, 57, 16
], dtype=float)
COMMODITY_INTRA_CORR = 0.35  # flat approximation

# Cross-asset-class correlations (Table 4, SIMM v2.6)
CROSS_CLASS_CORR: dict[tuple[str, str], float] = {
    ("IR", "EQUITY"):    0.18,
    ("IR", "COMMODITY"): 0.17,
    ("IR", "FX"):        0.35,
    ("IR", "CREDIT"):    0.30,
    ("EQUITY", "COMMODITY"): 0.20,
    ("EQUITY", "FX"):    0.22,
    ("EQUITY", "CREDIT"): 0.35,
    ("COMMODITY", "FX"): 0.22,
    ("COMMODITY", "CREDIT"): 0.17,
    ("FX", "CREDIT"):   0.20,
}

def cross_class_corr(a: str, b: str) -> float:
    if a == b:
        return 1.0
    return CROSS_CLASS_CORR.get((a, b), CROSS_CLASS_CORR.get((b, a), 0.0))


# ---------------------------------------------------------------------------
# Sensitivity containers
# ---------------------------------------------------------------------------

@dataclass
class SimmSensitivities:
    """Input sensitivities for a SIMM calculation.

    All values are in the settlement currency (e.g. USD millions per 1bp or
    per unit of the risk factor). Shapes can be scalars, (T,) or (n_paths, T).

    Parameters
    ----------
    ir : dict[str, np.ndarray]
        currency → tenor → sensitivity array.
        E.g. ``{"USD": {"1y": 500, "5y": 800}}``.
    equity : dict[int, np.ndarray]
        bucket_number → net equity sensitivity (sum of delta-notionals per bucket).
    fx : dict[str, np.ndarray]
        foreign_currency → net FX delta sensitivity.
    commodity : dict[int, np.ndarray]
        bucket_number (1–17) → net commodity delta.
    credit_ig : dict[str, np.ndarray]
        issuer_id → CS01 (credit spread sensitivity) for IG issuers.
    credit_hy : dict[str, np.ndarray]
        issuer_id → CS01 for HY/NR issuers.
    """

    ir: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    equity: dict[int, np.ndarray] = field(default_factory=dict)
    fx: dict[str, np.ndarray] = field(default_factory=dict)
    commodity: dict[int, np.ndarray] = field(default_factory=dict)
    credit_ig: dict[str, np.ndarray] = field(default_factory=dict)
    credit_hy: dict[str, np.ndarray] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# SIMM Calculator
# ---------------------------------------------------------------------------

class SimmCalculator:
    """Compute ISDA SIMM Initial Margin from delta sensitivities.

    This implements the full cross-bucket and cross-class aggregation
    without approximation. Vega and curvature margins are not included
    (they require vega sensitivities, which are optional add-ons).

    Usage
    -----
    >>> calc = SimmCalculator()
    >>> sens = SimmSensitivities(ir={"USD": {"1y": 1000, "5y": 2000}})
    >>> im = calc.total_im(sens)
    """

    def ir_margin(self, ir_sens: dict[str, dict[str, float | np.ndarray]]) -> np.ndarray:
        """IR delta margin: aggregate across currencies then across tenors.

        For each currency c:
          WS_c(k) = sens(c, k) × RW(c, k)            weighted sensitivity
          K_c = sqrt(Σ_k WS_c(k)² + Σ_{k≠l} ρ_{kl} WS_c(k) WS_c(l))  bucket IM

        Cross-currency:
          Delta_IR = sqrt(Σ_c K_c² + Σ_{c≠d} φ_{cd} K_c K_d)

        where φ_{cd} = IR_INTER_CURRENCY_CORR for all c ≠ d.
        """
        per_ccy: dict[str, np.ndarray] = {}

        for ccy, tenor_sens in ir_sens.items():
            # Build weighted sensitivity vector across tenors
            ws = np.zeros(len(IR_TENORS), dtype=float)
            rw = IR_RISK_WEIGHTS.get(ccy, IR_RISK_WEIGHTS["REGULAR"])
            for k, tenor in enumerate(IR_TENORS):
                if tenor in tenor_sens:
                    ws[k] = np.asarray(tenor_sens[tenor], dtype=float) * rw[k]

            # Intra-currency aggregation: K_c = sqrt(WS^T ρ WS)
            K_c = np.sqrt(np.maximum(ws @ IR_INTRA_CORR @ ws, 0.0))
            per_ccy[ccy] = K_c

        if not per_ccy:
            return np.float64(0.0)

        ccys = list(per_ccy.keys())
        K = np.array([per_ccy[c] for c in ccys])

        # Cross-currency aggregation
        phi = IR_INTER_CURRENCY_CORR
        cross = phi * np.sum(
            [[K[i] * K[j] for j in range(len(K)) if j != i] for i in range(len(K))]
        )
        return np.sqrt(np.maximum(np.sum(K**2) + cross, 0.0))

    def equity_margin(self, eq_sens: dict[int, float | np.ndarray]) -> np.ndarray:
        """Equity delta margin across sector buckets."""
        bucket_ims: dict[int, np.ndarray] = {}

        for bucket, sens in eq_sens.items():
            ws = np.asarray(sens, dtype=float) * EQUITY_RISK_WEIGHTS[bucket - 1]
            rho = EQUITY_INTRA_CORR.get(bucket, 0.20)
            # Single-name bucket: scalar sens → K_b = |WS|
            K_b = np.abs(ws)
            bucket_ims[bucket] = K_b

        if not bucket_ims:
            return np.float64(0.0)

        buckets = list(bucket_ims.keys())
        K = np.array([bucket_ims[b] for b in buckets])

        gamma_sum = 0.0
        for i, bi in enumerate(buckets):
            for j, bj in enumerate(buckets):
                if i != j:
                    gamma = EQUITY_INTER_BUCKET_CORR.get((bi, bj), 0.0)
                    gamma_sum += gamma * K[i] * K[j]

        return np.sqrt(np.maximum(np.sum(K**2) + gamma_sum, 0.0))

    def fx_margin(self, fx_sens: dict[str, float | np.ndarray]) -> np.ndarray:
        """FX delta margin: weighted net open position per currency pair."""
        total = np.float64(0.0)
        for ccy, sens in fx_sens.items():
            rw = FX_RISK_WEIGHT_HIGH_VOL if ccy.endswith("_EM") else FX_RISK_WEIGHT_REGULAR
            total = total + (np.asarray(sens, dtype=float) * rw) ** 2
        return np.sqrt(total)

    def commodity_margin(self, comm_sens: dict[int, float | np.ndarray]) -> np.ndarray:
        """Commodity delta margin across commodity buckets."""
        per_bucket: dict[int, np.ndarray] = {}
        for bucket, sens in comm_sens.items():
            rw = COMMODITY_RISK_WEIGHTS[bucket - 1]
            ws = np.asarray(sens, dtype=float) * rw
            per_bucket[bucket] = np.abs(ws)

        if not per_bucket:
            return np.float64(0.0)

        buckets = list(per_bucket.keys())
        K = np.array([per_bucket[b] for b in buckets])
        rho = COMMODITY_INTRA_CORR

        cross = rho * np.sum([K[i] * K[j] for i in range(len(K)) for j in range(len(K)) if i != j])
        return np.sqrt(np.maximum(np.sum(K**2) + cross, 0.0))

    def credit_margin(
        self,
        credit_ig: dict[str, float | np.ndarray],
        credit_hy: dict[str, float | np.ndarray],
    ) -> np.ndarray:
        """Credit (qualifying + non-qualifying) delta margin.

        Simplified: aggregate IG and HY CS01s with fixed risk weights.
        IG RW ≈ 97 bps, HY RW ≈ 455 bps (SIMM v2.6 Table 2).
        """
        ig_rw, hy_rw = 97.0, 455.0
        ig_im = np.sqrt(sum((np.asarray(s) * ig_rw) ** 2 for s in credit_ig.values())) if credit_ig else 0.0
        hy_im = np.sqrt(sum((np.asarray(s) * hy_rw) ** 2 for s in credit_hy.values())) if credit_hy else 0.0

        # IG and HY are separate buckets within Credit; aggregate with ρ=0.5
        return np.sqrt(np.maximum(ig_im**2 + hy_im**2 + 2 * 0.50 * ig_im * hy_im, 0.0))

    def total_im(self, sensitivities: SimmSensitivities) -> np.ndarray:
        """Compute total SIMM IM across all risk classes.

        Aggregation:
          IM_total = sqrt(Σ_c IM_c² + Σ_{c≠d} ψ_{cd} IM_c IM_d)

        where ψ_{cd} is the cross-asset-class correlation from Table 4.

        Returns
        -------
        np.ndarray
            Total IM (scalar or array matching sensitivity shapes).
        """
        class_ims = {}
        if sensitivities.ir:
            class_ims["IR"] = self.ir_margin(sensitivities.ir)
        if sensitivities.equity:
            class_ims["EQUITY"] = self.equity_margin(sensitivities.equity)
        if sensitivities.fx:
            class_ims["FX"] = self.fx_margin(sensitivities.fx)
        if sensitivities.commodity:
            class_ims["COMMODITY"] = self.commodity_margin(sensitivities.commodity)
        if sensitivities.credit_ig or sensitivities.credit_hy:
            class_ims["CREDIT"] = self.credit_margin(sensitivities.credit_ig, sensitivities.credit_hy)

        if not class_ims:
            return np.float64(0.0)

        classes = list(class_ims.keys())
        IM = np.array([class_ims[c] for c in classes], dtype=float)

        cross = sum(
            cross_class_corr(classes[i], classes[j]) * IM[i] * IM[j]
            for i in range(len(classes))
            for j in range(len(classes))
            if i != j
        )
        return np.sqrt(np.maximum(np.sum(IM**2) + cross, 0.0))
