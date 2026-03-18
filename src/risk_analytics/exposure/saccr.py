"""SA-CCR (Standardised Approach for Counterparty Credit Risk) calculator.

Implements the Basel III SA-CCR formula:
    EAD = 1.4 × (RC + PFE_add_on)

where:
    RC  = max(V, 0)         — replacement cost (current MTM if positive)
    PFE = sum of asset-class add-ons aggregated across hedging sets

This is a formula-based (not simulation-based) regulatory capital calculation.
It uses trade-level notionals, maturities, and asset classes inferred from the
pricer type.

Asset class supervisory factors (Basel III CRE52):
    IR   : 0.20% (<1Y), 0.50% (1–5Y), 1.00% (5–10Y), 1.50% (>10Y)
    Equity single-name : 32%
    Equity index       : 20%
    FX                 : 4%
    Commodity energy   : 18%
    Commodity metals   : 18%
    Commodity other    : 18%
    Credit IG          : 0.38%
    Credit SG          : 1.06%

Reference: Basel Committee on Banking Supervision, "The standardised approach
for measuring counterparty credit risk exposures", March 2014 (rev. April 2014).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Supervisory factor table (Basel III CRE52.72)
# ---------------------------------------------------------------------------

# IR supervisory factors by maturity bucket
_IR_FACTORS = {
    "lt1y": 0.0020,    # < 1 year
    "1y5y": 0.0050,    # 1–5 years
    "gt5y_lt10y": 0.0100,  # 5–10 years
    "gt10y": 0.0150,   # > 10 years
}

_SF = {
    "equity_single": 0.32,
    "equity_index":  0.20,
    "fx":            0.04,
    "commodity_energy":  0.18,
    "commodity_metals":  0.18,
    "commodity_other":   0.18,
    "credit_ig":     0.0038,
    "credit_sg":     0.0106,
}

# Alpha multiplier (Basel III)
_ALPHA = 1.4


def _ir_supervisory_factor(maturity: float) -> float:
    """Return the IR supervisory factor for a given maturity (years)."""
    if maturity < 1.0:
        return _IR_FACTORS["lt1y"]
    elif maturity <= 5.0:
        return _IR_FACTORS["1y5y"]
    elif maturity <= 10.0:
        return _IR_FACTORS["gt5y_lt10y"]
    else:
        return _IR_FACTORS["gt10y"]


# ---------------------------------------------------------------------------
# Trade descriptor
# ---------------------------------------------------------------------------

@dataclass
class SACCRTrade:
    """Description of a single trade for SA-CCR purposes.

    Parameters
    ----------
    trade_id : str
    asset_class : str
        One of: "ir", "equity_single", "equity_index", "fx",
        "commodity_energy", "commodity_metals", "commodity_other",
        "credit_ig", "credit_sg".
    notional : float
        Supervisory notional (e.g. swap notional, option notional).
    maturity : float
        Residual maturity in years (relevant for IR; ignored for equity/FX).
    current_mtm : float
        Current mark-to-market value (from model or market quote).
    delta : float
        Adjusted notional sign/delta: +1 for long/receiver, -1 for short/payer.
        For options, use the Black-Scholes delta.
    """
    trade_id: str
    asset_class: str
    notional: float
    maturity: float = 1.0
    current_mtm: float = 0.0
    delta: float = 1.0


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------

class SACCRCalculator:
    """SA-CCR EAD calculator for a single netting set.

    Usage::

        calc = SACCRCalculator()
        calc.add_trade(SACCRTrade("swap1", "ir", notional=10_000_000, maturity=5.0, current_mtm=50_000))
        calc.add_trade(SACCRTrade("opt1", "equity_single", notional=500_000, maturity=2.0, current_mtm=-10_000))
        ead = calc.ead()

    The calculator aggregates add-ons within each asset class (no cross-class
    netting) and computes EAD = alpha × (RC + PFE_add_on).

    Notes
    -----
    This is a simplified implementation. Production SA-CCR requires:
    - Maturity factor M_i (the supervisory duration for IR)
    - Within-hedging-set aggregation with correlation ρ
    - Netting set-level replacement cost with threshold / MTA
    For simplicity, this implementation uses full-notional add-ons (no duration
    adjustment for IR swaps, no within-hedging-set netting).
    """

    def __init__(self) -> None:
        self._trades: list[SACCRTrade] = []

    def add_trade(self, trade: SACCRTrade) -> None:
        """Add a trade to the netting set."""
        self._trades.append(trade)

    def replacement_cost(self) -> float:
        """RC = max(V_net, 0) where V_net is the net MTM across all trades."""
        net_mtm = sum(t.current_mtm for t in self._trades)
        return max(net_mtm, 0.0)

    def pfe_addon(self) -> float:
        """PFE add-on = sum of asset-class add-ons (no cross-class netting)."""
        total = 0.0
        for trade in self._trades:
            sf = self._supervisory_factor(trade)
            # Effective notional = delta × notional × supervisory_factor
            addon = abs(trade.delta) * trade.notional * sf
            total += addon
        return total

    def ead(self) -> float:
        """EAD = alpha × (RC + PFE_add_on)."""
        return _ALPHA * (self.replacement_cost() + self.pfe_addon())

    # ------------------------------------------------------------------
    # Factory helpers: infer SACCRTrade from pipeline Trade objects
    # ------------------------------------------------------------------

    @classmethod
    def from_trades(
        cls,
        trades: list,
        current_mtm: Optional[dict] = None,
    ) -> "SACCRCalculator":
        """Build a SACCRCalculator from a list of Trade objects.

        Parameters
        ----------
        trades : list of Trade
            Trade objects from the pipeline (must have .pricer and .id).
        current_mtm : dict, optional
            Mapping of trade_id -> current MTM value. Defaults to 0 for
            all trades if not provided.

        Returns
        -------
        SACCRCalculator
        """
        calc = cls()
        mtm_map = current_mtm or {}

        for trade in trades:
            saccr_trade = cls._infer_trade(trade, mtm_map.get(trade.id, 0.0))
            if saccr_trade is not None:
                calc.add_trade(saccr_trade)

        return calc

    @staticmethod
    def _infer_trade(trade, current_mtm: float) -> Optional[SACCRTrade]:
        """Infer SA-CCR attributes from a pipeline Trade object."""
        pricer = trade.pricer
        pricer_type = type(pricer).__name__

        if pricer_type == "InterestRateSwap":
            return SACCRTrade(
                trade_id=trade.id,
                asset_class="ir",
                notional=getattr(pricer, "notional", 1_000_000),
                maturity=getattr(pricer, "maturity", 1.0),
                current_mtm=current_mtm,
                delta=1.0 if getattr(pricer, "payer", True) else -1.0,
            )
        elif pricer_type in ("ZeroCouponBond", "FixedRateBond"):
            return SACCRTrade(
                trade_id=trade.id,
                asset_class="ir",
                notional=getattr(pricer, "face_value", 1_000_000),
                maturity=getattr(pricer, "maturity", 1.0),
                current_mtm=current_mtm,
                delta=1.0,
            )
        elif pricer_type in ("EuropeanOption", "BarrierOption", "AsianOption"):
            return SACCRTrade(
                trade_id=trade.id,
                asset_class="equity_single",
                notional=getattr(pricer, "strike", 100.0) * 1_000,  # nominal
                maturity=getattr(pricer, "expiry", 1.0),
                current_mtm=current_mtm,
                delta=1.0,
            )
        elif pricer_type == "GarmanKohlhagen":
            return SACCRTrade(
                trade_id=trade.id,
                asset_class="fx",
                notional=getattr(pricer, "notional", 1_000_000),
                maturity=getattr(pricer, "expiry", 1.0),
                current_mtm=current_mtm,
                delta=1.0,
            )
        else:
            # Unknown type: skip
            return None

    @staticmethod
    def _supervisory_factor(trade: SACCRTrade) -> float:
        if trade.asset_class == "ir":
            return _ir_supervisory_factor(trade.maturity)
        return _SF.get(trade.asset_class, 0.0)
