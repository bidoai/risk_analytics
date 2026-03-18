from __future__ import annotations

import numpy as np

from risk_analytics.exposure.csa import CSATerms


class REGVMStepper:
    """Step-by-step variation margin calculator under ISDA CSA terms.

    Maintains per-path collateral state (Credit Support Balance) across
    time steps. Designed for use inside ``StreamingExposureEngine``; can
    also be used stand-alone for testing margining logic.

    The CSB (Credit Support Balance) represents the net collateral held
    by us from the counterparty.  A positive CSB means we hold collateral.

    Margin call logic (simplified ISDA 2016 VM CSA):
    - Delivery Amount (we call):  max(V(t) - TH_p  - CSB, MTA_p)  if > 0
    - Return Amount   (they call): max(CSB - V(t) + TH_c, MTA_c)  if > 0

    Post-margin exposure (Credit Exposure, CE):
    CE(t) = max(V(t) - CSB(t), 0)

    The MPOR adjustment (for EE_MPOR used in CVA) is handled by the
    streaming engine by looking ahead ``mpor`` steps.
    """

    def __init__(self, csa: CSATerms, n_paths: int) -> None:
        self.csa = csa
        self.n_paths = n_paths
        # Per-path Credit Support Balance (collateral held by us)
        self._csb: np.ndarray = np.full(n_paths, csa.ia_counterparty - csa.ia_party)

    @property
    def csb(self) -> np.ndarray:
        """Current Credit Support Balance per path, shape (n_paths,)."""
        return self._csb.copy()

    def step(self, net_mtm: np.ndarray) -> np.ndarray:
        """Apply one margin call cycle and return post-margin exposure.

        Parameters
        ----------
        net_mtm : np.ndarray, shape (n_paths,)
            Aggregate net MTM of the netting set at this time step.

        Returns
        -------
        np.ndarray, shape (n_paths,)
            Post-margin credit exposure CE(t) = max(V(t) - CSB(t), 0).
        """
        csa = self.csa
        csb = self._csb

        # Delivery Amount: counterparty must post to us
        # We call when V(t) > TH_p + CSB  (our uncollateralised exposure exceeds threshold)
        shortfall = net_mtm - csa.threshold_party - csb
        delivery = np.where(shortfall > csa.mta_party, shortfall, 0.0)
        delivery = np.maximum(delivery, 0.0)

        # Return Amount: we must return collateral to counterparty
        # They call when CSB > V(t) + TH_c  (overcollateralised relative to their threshold)
        excess = csb - net_mtm - csa.threshold_counterparty
        ret = np.where(excess > csa.mta_counterparty, excess, 0.0)
        ret = np.maximum(ret, 0.0)

        # Apply rounding per ISDA 2016 VM CSA:
        #   Delivery amounts round UP   (ceil) — we receive at least what we're owed
        #   Return amounts   round DOWN (floor) — we return no more than the excess
        if csa.rounding_nearest > 0:
            delivery = (
                np.ceil(delivery / csa.rounding_nearest) * csa.rounding_nearest
            )
            ret = (
                np.floor(ret / csa.rounding_nearest) * csa.rounding_nearest
            )

        # Update CSB
        self._csb = csb + delivery - ret

        # Post-margin exposure
        return np.maximum(net_mtm - self._csb, 0.0)

    def reset(self) -> None:
        """Reset CSB to initial IA position."""
        csa = self.csa
        self._csb[:] = csa.ia_counterparty - csa.ia_party
