"""Regulatory Variation Margin (REGVM) engine.

Implements the bilateral VM call calculation per:
- ISDA 2016 Credit Support Annex for Variation Margin (VM CSA)
- EMIR Article 11 / CFTC §23.504 (for REGVM regime)
- ISDA 1994 CSA (for LEGACY regime)
"""
from __future__ import annotations

import numpy as np

from risk_analytics.exposure.csa import CSATerms, MarginRegime


class REGVMEngine:
    """Computes Variation Margin calls and Credit Support Balance on simulation paths.

    Under a bilateral CSA, at each time step:
    - If net MtM V(t) > threshold_counterparty: we call VM from the counterparty.
    - If net MtM V(t) < -threshold_party: the counterparty calls VM from us.
    - Transfers only occur if the change exceeds the MTA.

    For Monte Carlo path-wise computation we use the **stationary approximation**:
    the Credit Support Balance (CSB) is set to its stationary target value at each
    time step, rather than tracking the rolling balance. This is standard practice
    in XVA/PFE engines where the simulation is not path-recurrent. The MPOR
    adjustments in ``BilateralExposureCalculator`` then add the gap-risk effect.

    Parameters
    ----------
    csa : CSATerms
        CSA terms governing this netting set.
    """

    def __init__(self, csa: CSATerms) -> None:
        self.csa = csa

    def credit_support_balance(self, net_mtm: np.ndarray) -> np.ndarray:
        """Net collateral balance held by us (Credit Support Balance).

        CSB(t) = max(V(t) - TH_c, 0)        [VM received from cp]
               - max(-V(t) - TH_p, 0)       [VM posted by us]
               + IA_counterparty - IA_party  [Independent Amounts]

        Positive CSB means we hold net collateral (reduces our EE).
        Negative CSB means we posted net collateral (increases our EE).

        Parameters
        ----------
        net_mtm : np.ndarray, shape (n_paths, T)
            Net MtM of the netting set.

        Returns
        -------
        np.ndarray, shape (n_paths, T)
        """
        th_c = self.csa.threshold_counterparty
        th_p = self.csa.threshold_party
        ia_net = self.csa.ia_counterparty - self.csa.ia_party

        vm_received = np.maximum(net_mtm - th_c, 0.0)   # cp posts to us
        vm_posted = np.maximum(-net_mtm - th_p, 0.0)    # we post to cp

        csb = vm_received - vm_posted + ia_net

        if self.csa.rounding_nearest > 0:
            csb = self._round_conservative(csb, net_mtm)

        return csb

    def vm_call(self, net_mtm: np.ndarray) -> np.ndarray:
        """VM call amount from our perspective (positive = we call from cp).

        This is the gross delivery/return amount before MTA filtering.
        For path simulation, this equals ``credit_support_balance`` minus
        any IA offset, representing the pure VM component.

        Parameters
        ----------
        net_mtm : np.ndarray, shape (n_paths, T)

        Returns
        -------
        np.ndarray, shape (n_paths, T)
        """
        th_c = self.csa.threshold_counterparty
        th_p = self.csa.threshold_party

        call = np.maximum(net_mtm - th_c, 0.0) - np.maximum(-net_mtm - th_p, 0.0)

        if self.csa.mta_party > 0 or self.csa.mta_counterparty > 0:
            call = self._apply_mta(call)

        return call

    def lagged_csb(
        self,
        net_mtm: np.ndarray,
        time_grid: np.ndarray,
        lag: float | None = None,
    ) -> np.ndarray:
        """CSB lagged by MPOR to simulate the last-good-margin state.

        Under a default scenario, the last posted collateral was set at
        ``t - MPOR``. Exposure at time ``t`` is thus:
            E(t) = max(V(t) - CSB(t - MPOR), 0)

        This method computes ``CSB(t - MPOR)`` by:
        1. Computing the full CSB(t) on each path.
        2. Interpolating back by ``mpor`` on the time axis.

        For ``t < MPOR``, we clamp to t=0 (only IA, no VM yet).

        Parameters
        ----------
        net_mtm : np.ndarray, shape (n_paths, T)
        time_grid : np.ndarray, shape (T,)
        lag : float | None
            Override MPOR (years). Defaults to ``csa.mpor``.

        Returns
        -------
        np.ndarray, shape (n_paths, T)
        """
        lag = lag if lag is not None else self.csa.mpor
        csb_full = self.credit_support_balance(net_mtm)  # (n_paths, T)
        lagged_times = np.clip(time_grid - lag, time_grid[0], time_grid[-1])

        # Vectorised interpolation across all paths
        lagged = np.empty_like(csb_full)
        for p in range(csb_full.shape[0]):
            lagged[p] = np.interp(lagged_times, time_grid, csb_full[p])

        return lagged

    def uncollateralised_exposure(self, net_mtm: np.ndarray) -> np.ndarray:
        """Positive exposure with no collateral: max(V(t), 0)."""
        return np.maximum(net_mtm, 0.0)

    def collateralised_exposure(
        self,
        net_mtm: np.ndarray,
        time_grid: np.ndarray,
        im_balance: np.ndarray | None = None,
    ) -> np.ndarray:
        """Path-wise positive exposure after VM (with MPOR lag) and IM.

        E_coll(t) = max(V(t) - CSB(t - MPOR) - IM(t), 0)

        Parameters
        ----------
        net_mtm : np.ndarray, shape (n_paths, T)
        time_grid : np.ndarray, shape (T,)
        im_balance : np.ndarray | None, shape (n_paths, T) or broadcastable
            Initial margin held (received from counterparty). If None, no IM.

        Returns
        -------
        np.ndarray, shape (n_paths, T)
        """
        csb = self.lagged_csb(net_mtm, time_grid)
        net = net_mtm - csb
        if im_balance is not None:
            net = net - np.asarray(im_balance)
        return np.maximum(net, 0.0)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_mta(self, call: np.ndarray) -> np.ndarray:
        """Zero out calls smaller than MTA (both directions)."""
        mta_recv = self.csa.mta_party        # minimum we will call
        mta_send = self.csa.mta_counterparty  # minimum cp will call
        # Positive call (we receive): zero if below mta_recv
        # Negative call (we post): zero if abs below mta_send
        result = call.copy()
        result[(call > 0) & (call < mta_recv)] = 0.0
        result[(call < 0) & (-call < mta_send)] = 0.0
        return result

    def _round_conservative(self, csb: np.ndarray, net_mtm: np.ndarray) -> np.ndarray:
        """Round CSB conservatively per ISDA 2016 VM CSA para 3(b).

        - Delivery amounts (we receive) round DOWN to nearest unit.
        - Return amounts (we return to cp) round UP to nearest unit.
        This is conservative (we keep slightly less than the target CSB).
        """
        unit = self.csa.rounding_nearest
        rounded = np.where(
            csb >= 0,
            np.floor(csb / unit) * unit,   # receiving: round down
            np.ceil(csb / unit) * unit,    # posting: round up (less negative)
        )
        return rounded
