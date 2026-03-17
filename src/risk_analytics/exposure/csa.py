"""Credit Support Annex (CSA) terms for bilateral ISDA agreements."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class MarginRegime(Enum):
    """Regulatory margin framework governing the agreement."""

    REGVM = "REGVM"       # Regulatory VM: EMIR Art.11 / CFTC §23.504 — no threshold
    REGIM = "REGIM"       # Regulatory IM: BCBS-IOSCO Phase 1-6 — segregated, SIMM or Schedule
    LEGACY = "LEGACY"     # Pre-regulation bilateral CSA — may have non-zero threshold
    CLEARED = "CLEARED"   # CCP-cleared: daily VM, IM via CCP waterfall


class IMModel(Enum):
    """Initial margin calculation method."""

    SIMM = "SIMM"           # ISDA Standard Initial Margin Model (sensitivity-based)
    SCHEDULE = "SCHEDULE"   # BCBS-IOSCO Schedule (gross notional × table weight)
    INTERNAL = "INTERNAL"   # Internal model approved by regulator


@dataclass
class CSATerms:
    """Full Credit Support Annex parameter set for bilateral OTC agreements.

    All monetary values are in the same currency and unit as the MTM arrays
    (typically millions). Time values are in year-fractions, consistent with
    the simulation ``time_grid``.

    Parameters
    ----------
    counterparty_id : str
        Identifier for the counterparty.
    margin_regime : MarginRegime
        Regulatory framework (REGVM, REGIM, LEGACY, CLEARED).

    Thresholds
    ----------
    threshold_party : float
        TH_p — amount below which the counterparty need not post VM to us.
        Under REGVM this must be 0. A positive value means they can owe us
        up to this amount before we call margin.
    threshold_counterparty : float
        TH_c — amount below which we need not post VM to the counterparty.

    Minimum Transfer Amounts
    ------------------------
    mta_party : float
        Smallest VM call we will make on the counterparty (calls below this
        are not made; amounts aggregate until the threshold is reached).
    mta_counterparty : float
        Smallest VM call the counterparty will make on us.

    Independent Amounts
    -------------------
    ia_party : float
        IA we post to the counterparty unconditionally (reduces our CSB).
    ia_counterparty : float
        IA the counterparty posts to us unconditionally (increases our CSB).

    Rounding
    --------
    rounding_nearest : float
        Round VM transfer amounts to the nearest multiple of this value.
        Per ISDA 2016 VM CSA: delivery amounts round *up*, return amounts
        round *down* (conservative for the posting party).

    Initial Margin
    --------------
    im_model : IMModel
        IM calculation method (SIMM / Schedule / Internal).
    im_threshold : float
        ISDA IM exemption: if the aggregate average notional of both parties
        falls below this level, IM exchange is not required (BCBS-IOSCO
        Phase-in threshold; currently 8 bn USD for Phase 6).
    im_mta : float
        Minimum Transfer Amount for IM calls.
    segregated_im : bool
        Whether IM is held in a bankruptcy-remote segregated account.
        True under EMIR/CFTC; means IM offsets PFE but NOT current EE.
    rehypothecatable_vm : bool
        Whether received VM can be reused (re-hypothecated) by the receiver.
        Affects FVA/MVA but not counterparty credit exposure directly.

    Margin Period of Risk
    ---------------------
    mpor : float
        Time in years from the last successful margin call to when the
        portfolio is closed out after a default. Regulatory minimum:
        - Bilateral OTC: 10 business days ≈ 10/252
        - Netting sets > 5000 trades: 20 BD
        - Illiquid collateral: 20 BD
    margin_call_frequency : float
        How often VM calls can be made (years). Daily = 1/252.

    Settlement
    ----------
    currency : str
        Settlement currency for all collateral flows.
    eligible_collateral : dict[str, float]
        Map of eligible collateral asset type → haircut (0–1).
        E.g. ``{"CASH_USD": 0.0, "UST_10Y": 0.02}``.
    """

    counterparty_id: str = "COUNTERPARTY"
    margin_regime: MarginRegime = MarginRegime.REGVM

    # Thresholds
    threshold_party: float = 0.0
    threshold_counterparty: float = 0.0

    # Minimum Transfer Amounts
    mta_party: float = 0.0
    mta_counterparty: float = 0.0

    # Independent Amounts (positive = received by us)
    ia_party: float = 0.0
    ia_counterparty: float = 0.0

    # Rounding
    rounding_nearest: float = 0.0

    # Initial Margin
    im_model: IMModel = IMModel.SCHEDULE
    im_threshold: float = 8_000.0       # 8 bn default (in millions)
    im_mta: float = 0.0
    segregated_im: bool = True
    rehypothecatable_vm: bool = True

    # Margin Period of Risk
    mpor: float = 10 / 252              # 10 business days
    margin_call_frequency: float = 1 / 252

    # Settlement
    currency: str = "USD"
    eligible_collateral: dict[str, float] = field(
        default_factory=lambda: {
            "CASH_USD": 0.000,
            "CASH_EUR": 0.000,
            "UST_2Y":   0.005,
            "UST_5Y":   0.010,
            "UST_10Y":  0.020,
            "UST_30Y":  0.040,
            "EUR_GOVT_IG": 0.005,
            "CORP_IG":  0.100,
            "EQUITY":   0.150,
        }
    )

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def regvm_standard(cls, counterparty_id: str = "CP", mta: float = 0.5) -> "CSATerms":
        """EMIR-compliant Regulatory VM: zero threshold, daily margining."""
        return cls(
            counterparty_id=counterparty_id,
            margin_regime=MarginRegime.REGVM,
            threshold_party=0.0,
            threshold_counterparty=0.0,
            mta_party=mta,
            mta_counterparty=mta,
            mpor=10 / 252,
            margin_call_frequency=1 / 252,
        )

    @classmethod
    def legacy_bilateral(
        cls,
        counterparty_id: str = "CP",
        threshold: float = 5.0,
        mta: float = 0.5,
    ) -> "CSATerms":
        """Pre-regulation bilateral CSA with symmetric threshold."""
        return cls(
            counterparty_id=counterparty_id,
            margin_regime=MarginRegime.LEGACY,
            threshold_party=threshold,
            threshold_counterparty=threshold,
            mta_party=mta,
            mta_counterparty=mta,
            mpor=10 / 252,
        )

    @classmethod
    def cleared(cls, counterparty_id: str = "CCP") -> "CSATerms":
        """CCP-cleared: zero threshold, 5 BD MPOR, daily VM."""
        return cls(
            counterparty_id=counterparty_id,
            margin_regime=MarginRegime.CLEARED,
            threshold_party=0.0,
            threshold_counterparty=0.0,
            mta_party=0.0,
            mta_counterparty=0.0,
            mpor=5 / 252,
            margin_call_frequency=1 / 252,
            segregated_im=True,
            rehypothecatable_vm=False,
        )
