"""Regulatory Initial Margin (REGIM) engine.

Implements two IM calculation approaches:
1. BCBS-IOSCO Schedule (notional-based, simple)
2. ISDA SIMM (sensitivity-based, via SimmCalculator)

References:
- BCBS-IOSCO "Margin requirements for non-centrally cleared derivatives" (2020)
- ISDA SIMM v2.6
"""
from __future__ import annotations

import numpy as np

from risk_analytics.exposure.csa import CSATerms, IMModel
from .simm import SimmCalculator, SimmSensitivities


# BCBS-IOSCO Schedule: gross notional IM weights by asset class and maturity bucket
# (Table 1, 2020 framework)
# Keys: (asset_class, maturity_bucket) → weight fraction
SCHEDULE_WEIGHTS: dict[tuple[str, str], float] = {
    # Interest Rates
    ("IR", "0_2"):   0.01,
    ("IR", "2_5"):   0.02,
    ("IR", "5_plus"): 0.04,
    # FX
    ("FX", "any"):   0.04,
    # Equity (listed)
    ("EQUITY", "any"): 0.15,
    # Commodity
    ("COMMODITY", "any"): 0.15,
    # Credit (IG)
    ("CREDIT_IG", "0_2"):   0.02,
    ("CREDIT_IG", "2_5"):   0.05,
    ("CREDIT_IG", "5_plus"): 0.10,
    # Credit (HY)
    ("CREDIT_HY", "0_2"):   0.04,
    ("CREDIT_HY", "2_5"):   0.09,
    ("CREDIT_HY", "5_plus"): 0.17,
}


def _maturity_bucket(maturity_years: float) -> str:
    if maturity_years < 2:
        return "0_2"
    elif maturity_years < 5:
        return "2_5"
    else:
        return "5_plus"


class REGIMEngine:
    """Regulatory Initial Margin engine.

    Supports Schedule and SIMM methods. Results can be path-constant
    (computed once at t=0) or time-varying if sensitivity paths are provided.

    Parameters
    ----------
    csa : CSATerms
        Controls the IM method (SIMM or Schedule) and segregation flag.
    """

    def __init__(self, csa: CSATerms) -> None:
        self.csa = csa
        self._simm_calc = SimmCalculator()

    # ------------------------------------------------------------------
    # Schedule IM
    # ------------------------------------------------------------------

    def schedule_im(
        self,
        trades: list[dict],
        shape: tuple | None = None,
    ) -> np.ndarray:
        """Compute Schedule IM for a list of trades.

        Parameters
        ----------
        trades : list[dict]
            Each dict must have:
            - 'asset_class': str  (IR, FX, EQUITY, COMMODITY, CREDIT_IG, CREDIT_HY)
            - 'gross_notional': float | np.ndarray  (mark-to-market or notional)
            - 'maturity': float  (years; used for maturity bucket; ignored for FX/EQ/COMM)
            - 'net_replacement_cost': float | np.ndarray  (current MTM; can be negative)
        shape : tuple | None
            If set, broadcast scalar results to this shape (e.g. (n_paths, T)).

        Returns
        -------
        np.ndarray
            Total Schedule IM = 0.4 × GrossIM + 0.6 × NGR × GrossIM
            where NGR = max(NetRC, 0) / GrossIM (capped [0,1]).

        References
        ----------
        BCBS-IOSCO Schedule formula per Art. 7 of the 2020 framework.
        """
        gross_im = np.float64(0.0)
        net_rc = np.float64(0.0)

        for trade in trades:
            ac = trade["asset_class"]
            gn = np.asarray(trade["gross_notional"], dtype=float)
            mat = float(trade.get("maturity", 5.0))
            nrc = np.asarray(trade.get("net_replacement_cost", 0.0), dtype=float)

            bucket = _maturity_bucket(mat) if ac not in ("FX", "EQUITY", "COMMODITY") else "any"
            weight = SCHEDULE_WEIGHTS.get((ac, bucket), SCHEDULE_WEIGHTS.get((ac, "any"), 0.04))

            gross_im = gross_im + np.abs(gn) * weight
            net_rc = net_rc + nrc

        if np.all(gross_im == 0):
            result = np.float64(0.0)
        else:
            ngr = np.clip(net_rc / gross_im, 0.0, 1.0)
            result = (0.4 + 0.6 * ngr) * gross_im

        # Apply IM threshold: if aggregate notional < threshold, no IM required
        # (simplified: threshold check delegated to ISDAExposureCalculator)

        if self.csa.im_mta > 0:
            result = np.where(result > self.csa.im_mta, result, 0.0)

        if shape is not None:
            result = np.broadcast_to(np.asarray(result), shape).copy()

        return np.asarray(result, dtype=float)

    # ------------------------------------------------------------------
    # SIMM IM
    # ------------------------------------------------------------------

    def simm_im(
        self,
        sensitivities: SimmSensitivities,
        shape: tuple | None = None,
    ) -> np.ndarray:
        """Compute ISDA SIMM IM from trade-level delta sensitivities.

        Parameters
        ----------
        sensitivities : SimmSensitivities
            Aggregated netting-set sensitivities.
        shape : tuple | None
            Broadcast result to this shape if given.

        Returns
        -------
        np.ndarray
            Total SIMM IM (scalar or array).
        """
        im = self._simm_calc.total_im(sensitivities)

        if self.csa.im_mta > 0:
            im = np.where(im > self.csa.im_mta, im, 0.0)

        if shape is not None:
            im = np.broadcast_to(np.asarray(im), shape).copy()

        return np.asarray(im, dtype=float)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def compute_im(
        self,
        trades: list[dict] | None = None,
        sensitivities: SimmSensitivities | None = None,
        shape: tuple | None = None,
    ) -> np.ndarray:
        """Compute IM using the method specified in the CSA.

        Dispatches to ``schedule_im`` or ``simm_im`` based on
        ``csa.im_model``. If both inputs provided, the CSA method wins.

        Parameters
        ----------
        trades : list[dict] | None
            Required for Schedule method.
        sensitivities : SimmSensitivities | None
            Required for SIMM method.
        shape : tuple | None
            Output shape for broadcasting.
        """
        if self.csa.im_model == IMModel.SIMM:
            if sensitivities is None:
                raise ValueError("SimmSensitivities required for SIMM IM method.")
            return self.simm_im(sensitivities, shape=shape)
        else:  # SCHEDULE or INTERNAL
            if trades is None:
                raise ValueError("trades list required for Schedule IM method.")
            return self.schedule_im(trades, shape=shape)
