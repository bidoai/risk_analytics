"""Bilateral ISDA exposure calculator.

Combines netting, VM, IM, collateral, and all bilateral exposure metrics
into a single orchestrated calculator.

Key references:
- Gregory (2012) "Counterparty Credit Risk"
- Basel III SA-CCR (CRE52) / IMM (CRE53)
- ISDA 2016 VM CSA / 2018 IM CSA
"""
from __future__ import annotations

import logging
import warnings

import numpy as np

logger = logging.getLogger(__name__)

from pyxva.core.paths import SimulationResult
from .csa import CSATerms
from .collateral import CollateralAccount, HaircutSchedule
from .hazard_curve import HazardCurve
from .margin.vm import REGVMEngine
from .margin.im import REGIMEngine
from .margin.simm import SimmSensitivities
from .metrics import ExposureCalculator
from .netting import NettingSet

# Type alias used in signatures throughout this module
_HazardArg = "float | HazardCurve"


def _marginal_pd(
    hazard: "float | HazardCurve",
    t_prev: float,
    t: float,
) -> float:
    """Q(t_prev) − Q(t): marginal default probability in (t_prev, t].

    Dispatches to HazardCurve.marginal_default_prob for term-structure
    inputs, or uses exp(-λ t) for a flat scalar hazard rate.
    """
    if isinstance(hazard, HazardCurve):
        return hazard.marginal_default_prob(t_prev, t)
    # Flat scalar hazard rate
    return float(np.exp(-hazard * t_prev) - np.exp(-hazard * t))


def _marginal_pd_vec(
    hazard: "float | HazardCurve",
    time_grid: np.ndarray,
) -> np.ndarray:
    """Vectorized marginal default probabilities for every bucket in time_grid.

    Returns Q(t_{i-1}) − Q(t_i) for i = 1 … T−1 in a single call, avoiding a
    Python loop over T−1 scalar ``_marginal_pd`` queries.

    Parameters
    ----------
    hazard : float | HazardCurve
    time_grid : np.ndarray, shape (T,)

    Returns
    -------
    np.ndarray, shape (T-1,)
    """
    if isinstance(hazard, HazardCurve):
        q = hazard.survival_probability_vec(time_grid)
        return q[:-1] - q[1:]
    # Flat scalar hazard rate
    return np.exp(-hazard * time_grid[:-1]) - np.exp(-hazard * time_grid[1:])


class BilateralExposureCalculator(ExposureCalculator):
    """Extends ExposureCalculator with bilateral and regulatory metrics.

    Added metrics:
    - ENE  (Expected Negative Exposure): the counterparty's EE; used for DVA.
    - EEPE (Effective EPE): monotone-EE average over 1yr; SA-CCR/IMM capital.
    - EE_mpor: EE profile shifted forward by MPOR (gap-risk view).
    - collateralised_ee: EE after lagged-CSB and IM.
    - cva_approx: Credit Valuation Adjustment given a hazard rate.
    - dva_approx: Debt Valuation Adjustment.
    """

    # ------------------------------------------------------------------
    # ENE
    # ------------------------------------------------------------------

    def ene(self, mtm: np.ndarray) -> np.ndarray:
        """Expected Negative Exposure profile.

        ENE(t) = E[min(V(t), 0)]

        The counterparty's exposure from their perspective. Used in DVA:
        DVA ≈ LGD_own × Σ ENE(t_i) × PD_own(t_{i-1}, t_i)

        Returns
        -------
        np.ndarray, shape (T,)
        """
        return np.minimum(mtm, 0.0).mean(axis=0)

    # ------------------------------------------------------------------
    # EEPE (regulatory capital metric)
    # ------------------------------------------------------------------

    def eepe(
        self,
        mtm: np.ndarray,
        time_grid: np.ndarray,
        window: float = 1.0,
    ) -> float:
        """Effective Expected Positive Exposure (SA-CCR / IMM regulatory metric).

        Algorithm (Basel III CRE53.22):
        1. Compute EE(t) profile.
        2. Enforce monotonicity: EffEE(t_k) = max(EE(t_k), EffEE(t_{k-1})).
        3. EEPE = time-weighted average of EffEE over [0, min(T, window)].

        Used as the exposure measure for IMM capital. For non-IMM SA-CCR,
        use the alpha-multiplied EEPE: EEPE_reg = α × EEPE, α = 1.4.

        Parameters
        ----------
        mtm : np.ndarray, shape (n_paths, T)
        time_grid : np.ndarray, shape (T,)
        window : float
            Integration window in years (default 1yr per Basel III).

        Returns
        -------
        float
        """
        ee = self.expected_exposure(mtm)
        mask = time_grid <= window

        if not mask.any():
            warnings.warn(
                f"time_grid max ({time_grid[-1]:.2f}yr) < EEPE window ({window}yr). "
                "Using full grid.",
                UserWarning,
                stacklevel=2,
            )
            mask = np.ones(len(time_grid), dtype=bool)

        ee_w = ee[mask]
        tg_w = time_grid[mask]

        # Effective EE: non-decreasing (running maximum)
        eff_ee = np.maximum.accumulate(ee_w)

        T_w = tg_w[-1] - tg_w[0]
        if T_w == 0:
            return float(eff_ee[0])
        return float(np.trapezoid(eff_ee, tg_w) / T_w)

    # ------------------------------------------------------------------
    # MPOR-shifted EE
    # ------------------------------------------------------------------

    def mpor_adjusted_ee(
        self,
        mtm: np.ndarray,
        time_grid: np.ndarray,
        mpor: float,
    ) -> np.ndarray:
        """EE profile shifted forward by MPOR (gap-risk approximation).

        Under a default scenario, the last successful margin call was at
        ``t - MPOR``. The exposure at ``t`` therefore looks like the
        uncollateralised EE evaluated at ``t + MPOR`` (we see further out):

            EE_mpor(t) = EE(t + mpor)   [approximation]

        Boundary: beyond the last grid point, the last EE value is held flat.

        Returns
        -------
        np.ndarray, shape (T,)
        """
        ee = self.expected_exposure(mtm)
        return np.interp(time_grid + mpor, time_grid, ee, right=ee[-1])

    # ------------------------------------------------------------------
    # Collateralised EE
    # ------------------------------------------------------------------

    def collateralised_ee(
        self,
        net_mtm: np.ndarray,
        collateral_balance: np.ndarray,
    ) -> np.ndarray:
        """EE after collateral, incorporating MPOR gap risk.

        E_coll(t) = E[ max(V(t) - C(t - MPOR), 0) ]

        where ``collateral_balance`` is the lagged CSB (already shifted by
        MPOR by ``REGVMEngine.lagged_csb``).

        Returns
        -------
        np.ndarray, shape (T,)
        """
        exposure = np.maximum(net_mtm - collateral_balance, 0.0)
        return exposure.mean(axis=0)

    # ------------------------------------------------------------------
    # CVA / DVA approximation
    # ------------------------------------------------------------------

    def cva_approx(
        self,
        mtm: np.ndarray,
        time_grid: np.ndarray,
        hazard_rate: "float | HazardCurve",
        lgd: float = 0.6,
    ) -> float:
        """Approximate unilateral CVA via hazard rate or term-structure curve.

        CVA ≈ LGD × Σ_i EE(t_i) × [Q(t_{i-1}) - Q(t_i)]

        Parameters
        ----------
        hazard_rate : float | HazardCurve
            Counterparty default intensity λ (annualised flat, e.g. spread/LGD)
            or a ``HazardCurve`` for a term-structure.
        lgd : float
            Loss Given Default (fraction, default 60%).

        Returns
        -------
        float
        """
        ee = self.expected_exposure(mtm)
        return self._integral_xva(time_grid, ee, hazard_rate, lgd)

    def dva_approx(
        self,
        mtm: np.ndarray,
        time_grid: np.ndarray,
        own_hazard_rate: "float | HazardCurve",
        own_lgd: float = 0.6,
    ) -> float:
        """Approximate DVA (own-default benefit).

        DVA ≈ LGD_own × Σ_i |ENE(t_i)| × [Q_own(t_{i-1}) - Q_own(t_i)]

        Returns
        -------
        float
        """
        ene_profile = np.abs(self.ene(mtm))
        return self._integral_xva(time_grid, ene_profile, own_hazard_rate, own_lgd)

    def fva_approx(
        self,
        mtm: np.ndarray,
        time_grid: np.ndarray,
        funding: "float | HazardCurve",
    ) -> float:
        """Approximate FVA (Funding Valuation Adjustment).

        Cost of funding uncollateralised exposure and benefit from negative
        exposure (economically: we borrow from the counterparty):

            FVA ≈ s_fund × Σ_i EE(t_i) × Δt_i
                - s_fund × Σ_i |ENE(t_i)| × Δt_i

        When ``funding`` is a ``HazardCurve``, the funding spread at each
        step is implied from the marginal default probability: the integral
        uses the survival-weighted time measure so that the formula aligns
        with CVA for consistency in BCVA-FVA frameworks.

        Parameters
        ----------
        funding : float | HazardCurve
            Flat funding spread (annualised, e.g. OIS + funding margin) or
            a ``HazardCurve`` for term-structure funding spreads.

        Returns
        -------
        float — positive = funding cost (unfavourable), negative = funding benefit
        """
        ee = self.expected_exposure(mtm)
        ene_abs = np.abs(self.ene(mtm))
        fva_cost = self._integral_xva(time_grid, ee, funding, lgd=1.0)
        fva_benefit = self._integral_xva(time_grid, ene_abs, funding, lgd=1.0)
        return fva_cost - fva_benefit

    def mva_approx(
        self,
        im_profile: np.ndarray,
        time_grid: np.ndarray,
        funding: "float | HazardCurve",
    ) -> float:
        """Approximate MVA (Margin Valuation Adjustment).

        Cost of funding initial margin over the trade life:

            MVA ≈ s_fund × Σ_i IM(t_i) × Δt_i

        Parameters
        ----------
        im_profile : np.ndarray, shape (T,)
            Expected IM at each time step (e.g. from ``REGIMEngine.im_time_profile``).
        funding : float | HazardCurve
            Flat funding spread or term-structure ``HazardCurve``.

        Returns
        -------
        float — non-negative funding cost
        """
        return self._integral_xva(time_grid, im_profile, funding, lgd=1.0)

    def kva_approx(
        self,
        ead_t0: float,
        time_grid: np.ndarray,
        cost_of_capital: float = 0.10,
    ) -> float:
        """Approximate KVA (Capital Valuation Adjustment).

        Cost of holding regulatory capital over the trade life using a flat
        t=0 SA-CCR EAD profile (documented approximation):

            KVA ≈ CoC × EAD_0 × T

        where T = time_grid[-1].

        Parameters
        ----------
        ead_t0 : float
            SA-CCR Exposure-at-Default at t=0 (from ``SACCRCalculator``).
        time_grid : np.ndarray, shape (T,)
        cost_of_capital : float
            Annual cost-of-capital rate (default 10%).

        Returns
        -------
        float
        """
        T = float(time_grid[-1] - time_grid[0])
        return float(cost_of_capital * ead_t0 * T)

    def xva_attribution(
        self,
        mtm: np.ndarray,
        time_grid: np.ndarray,
        hazard: "float | HazardCurve",
        funding: "float | HazardCurve | None" = None,
        im_profile: "np.ndarray | None" = None,
        lgd: float = 0.6,
        own_hazard: "float | HazardCurve | None" = None,
    ) -> dict:
        """Per-time-bucket xVA attribution waterfall.

        Returns the contribution of each time bucket [t_{i-1}, t_i] to the
        total xVA, enabling drill-down by tenor region.

        Parameters
        ----------
        mtm : np.ndarray, shape (n_paths, T)
        time_grid : np.ndarray, shape (T,)
        hazard : float | HazardCurve
            Counterparty hazard rate / curve (for CVA).
        funding : float | HazardCurve | None
            Funding spread / curve (for FVA/MVA). If None, FVA/MVA not computed.
        im_profile : np.ndarray, shape (T,) | None
            Expected IM profile for MVA attribution. If None, MVA not computed.
        lgd : float
            Loss Given Default for CVA/DVA.
        own_hazard : float | HazardCurve | None
            Own hazard rate / curve for DVA.

        Returns
        -------
        dict with keys:
            'time'  : np.ndarray, shape (T-1,) — bucket midpoints
            'cva'   : np.ndarray, shape (T-1,) — CVA per bucket
            'dva'   : np.ndarray, shape (T-1,) — DVA per bucket (zeros if no own_hazard)
            'fva'   : np.ndarray, shape (T-1,) — FVA per bucket (zeros if no funding)
            'mva'   : np.ndarray, shape (T-1,) — MVA per bucket (zeros if no im_profile/funding)
            'total' : np.ndarray, shape (T-1,) — CVA - DVA + FVA + MVA per bucket
        """
        n_buckets = len(time_grid) - 1

        ee = self.expected_exposure(mtm)
        ene_abs = np.abs(self.ene(mtm))

        # Vectorized marginal PDs for all buckets in one pass — avoids T-1 Python iterations
        bucket_mids = 0.5 * (time_grid[:-1] + time_grid[1:])
        pd = _marginal_pd_vec(hazard, time_grid)
        cva_buckets = lgd * ee[1:] * pd

        dva_buckets = np.zeros(n_buckets)
        if own_hazard is not None:
            pd_own = _marginal_pd_vec(own_hazard, time_grid)
            dva_buckets = lgd * ene_abs[1:] * pd_own

        fva_buckets = np.zeros(n_buckets)
        mva_buckets = np.zeros(n_buckets)
        if funding is not None:
            fund_pd = _marginal_pd_vec(funding, time_grid)
            fva_buckets = (ee[1:] - ene_abs[1:]) * fund_pd
            if im_profile is not None:
                mva_buckets = im_profile[1:] * fund_pd

        total_buckets = cva_buckets - dva_buckets + fva_buckets + mva_buckets

        return {
            "time": bucket_mids,
            "cva": cva_buckets,
            "dva": dva_buckets,
            "fva": fva_buckets,
            "mva": mva_buckets,
            "total": total_buckets,
        }

    def bilateral_cva(
        self,
        mtm: np.ndarray,
        time_grid: np.ndarray,
        cp_hazard_rate: "float | HazardCurve",
        own_hazard_rate: "float | HazardCurve",
        cp_lgd: float = 0.6,
        own_lgd: float = 0.6,
    ) -> dict:
        """Compute CVA, DVA, and BCVA = CVA - DVA.

        Returns
        -------
        dict with keys: 'cva', 'dva', 'bcva'
        """
        cva = self.cva_approx(mtm, time_grid, cp_hazard_rate, cp_lgd)
        dva = self.dva_approx(mtm, time_grid, own_hazard_rate, own_lgd)
        return {"cva": cva, "dva": dva, "bcva": cva - dva}

    # ------------------------------------------------------------------
    # Full summary
    # ------------------------------------------------------------------

    def bilateral_summary(
        self,
        net_mtm: np.ndarray,
        time_grid: np.ndarray,
        collateral_balance: np.ndarray | None = None,
        mpor: float = 10 / 252,
        confidence: float = 0.95,
        cp_hazard_rate: "float | HazardCurve | None" = None,
        own_hazard_rate: "float | HazardCurve | None" = None,
        funding_spread: "float | HazardCurve | None" = None,
        im_profile: "np.ndarray | None" = None,
        cost_of_capital: float = 0.10,
        ead_t0: float = 0.0,
    ) -> dict:
        """Full bilateral exposure summary.

        Returns
        -------
        dict with keys:
            ee, ene, pfe, pse, epe, eepe,
            ee_mpor, ee_coll (if collateral given),
            cva, dva, bcva (if hazard rates given),
            time_grid, confidence
        """
        result: dict = {
            "ee": self.expected_exposure(net_mtm),
            "ene": self.ene(net_mtm),
            "pfe": self.pfe(net_mtm, confidence),
            "pse": self.pse(net_mtm),
            "epe": self.epe(net_mtm, time_grid),
            "eepe": self.eepe(net_mtm, time_grid),
            "ee_mpor": self.mpor_adjusted_ee(net_mtm, time_grid, mpor),
            "time_grid": time_grid,
            "confidence": confidence,
        }

        if collateral_balance is not None:
            result["ee_coll"] = self.collateralised_ee(net_mtm, collateral_balance)

        if cp_hazard_rate is not None and own_hazard_rate is not None:
            xva = self.bilateral_cva(
                net_mtm, time_grid, cp_hazard_rate, own_hazard_rate
            )
            result.update(xva)
        elif cp_hazard_rate is not None:
            result["cva"] = self.cva_approx(net_mtm, time_grid, cp_hazard_rate)
            result["dva"] = 0.0
            result["bcva"] = result["cva"]

        result["fva"] = 0.0
        result["mva"] = 0.0
        result["kva"] = 0.0

        if funding_spread is not None:
            result["fva"] = self.fva_approx(net_mtm, time_grid, funding_spread)
            if im_profile is not None:
                result["mva"] = self.mva_approx(im_profile, time_grid, funding_spread)

        if ead_t0 > 0.0:
            result["kva"] = self.kva_approx(ead_t0, time_grid, cost_of_capital)

        return result

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _integral_xva(
        time_grid: np.ndarray,
        profile: np.ndarray,
        hazard: "float | HazardCurve",
        lgd: float,
    ) -> float:
        """Discrete integral: lgd × Σ profile(t_i) × [Q(t_{i-1}) - Q(t_i)].

        Shared helper for CVA, DVA, FVA, and MVA. All four xVA integrals
        share this survival-probability-weighted structure.

        Parameters
        ----------
        time_grid : np.ndarray, shape (T,)
        profile : np.ndarray, shape (T,)
            EE, |ENE|, or IM profile.
        hazard : float | HazardCurve
            Flat hazard rate λ (annualised) or term-structure HazardCurve.
            For a flat rate, Q(t) = exp(-λ t).
        lgd : float
            Scaling factor (LGD for CVA/DVA; 1.0 for FVA/MVA).
        """
        total = 0.0
        for i in range(1, len(time_grid)):
            t0 = float(time_grid[i - 1])
            t1 = float(time_grid[i])
            pd_i = _marginal_pd(hazard, t0, t1)
            total += profile[i] * pd_i
        return float(lgd * total)


# ---------------------------------------------------------------------------
# ISDAExposureCalculator — full orchestrator
# ---------------------------------------------------------------------------

class ISDAExposureCalculator:
    """Full bilateral ISDA counterparty credit exposure engine.

    Orchestrates the complete pipeline:
    1. Net MTM via NettingSet
    2. VM via REGVMEngine (threshold, MTA, MPOR lag)
    3. IM via REGIMEngine (Schedule or SIMM)
    4. Collateral account reconciliation
    5. Bilateral exposure metrics via BilateralExposureCalculator

    Parameters
    ----------
    netting_set : NettingSet
    csa : CSATerms
    vm_engine : REGVMEngine | None
        Defaults to a standard REGVMEngine built from the CSA.
    im_engine : REGIMEngine | None
        If None, IM is not computed.
    collateral : CollateralAccount | None
        Pre-seeded collateral account. Defaults to an empty account.
    """

    def __init__(
        self,
        netting_set: NettingSet,
        csa: CSATerms,
        vm_engine: REGVMEngine | None = None,
        im_engine: REGIMEngine | None = None,
        collateral: CollateralAccount | None = None,
    ) -> None:
        self.netting_set = netting_set
        self.csa = csa
        self.vm_engine = vm_engine or REGVMEngine(csa)
        self.im_engine = im_engine
        self.collateral = collateral or CollateralAccount(
            haircut_schedule=HaircutSchedule.from_csa(csa)
            if hasattr(HaircutSchedule, "from_csa")
            else HaircutSchedule(),
            rehypothecation=csa.rehypothecatable_vm,
        )
        self._calc = BilateralExposureCalculator()

    def run(
        self,
        simulation_results: dict[str, SimulationResult],
        time_grid: np.ndarray,
        confidence: float = 0.95,
        im_trades: list[dict] | None = None,
        im_sensitivities: SimmSensitivities | None = None,
        cp_hazard_rate: "float | HazardCurve | None" = None,
        own_hazard_rate: "float | HazardCurve | None" = None,
        funding_spread: "float | HazardCurve | None" = None,
        cost_of_capital: float = 0.10,
    ) -> dict:
        """Execute the full bilateral exposure pipeline.

        Parameters
        ----------
        simulation_results : dict[str, SimulationResult]
            Output from MonteCarloEngine.run().
        time_grid : np.ndarray, shape (T,)
        confidence : float
            PFE confidence level.
        im_trades : list[dict] | None
            Trade descriptors for Schedule IM. Each dict: asset_class, gross_notional,
            maturity, net_replacement_cost.
        im_sensitivities : SimmSensitivities | None
            SIMM sensitivities (used if csa.im_model == SIMM).
        cp_hazard_rate : float | None
            Counterparty annualised default intensity (spread/LGD). If given,
            CVA/DVA are included in the output.
        own_hazard_rate : float | None
            Our own annualised default intensity (for DVA).

        Returns
        -------
        dict with keys:
            net_mtm          : (n_paths, T)
            csb              : (n_paths, T)
            lagged_csb       : (n_paths, T)
            im               : (n_paths, T) or None
            ee               : (T,)
            ene              : (T,)
            pfe              : (T,)
            ee_coll          : (T,)
            ee_mpor          : (T,)
            pse              : float
            epe              : float
            eepe             : float
            cva, dva, bcva   : float (if hazard rates given)
            collateral       : CollateralAccount
        """
        logger.info(
            "ISDAExposureCalculator.run: netting_set=%s  n_steps=%d  confidence=%.2f",
            self.netting_set.name, len(time_grid) - 1, confidence,
        )

        # 1. Net MTM
        net_mtm = self.netting_set.net_mtm(simulation_results)  # (n_paths, T)
        logger.debug("Step 1 complete: net_mtm shape=%s", net_mtm.shape)

        # 2. VM: path-dependent CSB (MTA-gated) and MPOR-lagged version for exposure
        csb = self.vm_engine.path_csb(net_mtm, time_grid)          # (n_paths, T)
        lagged_csb = self.vm_engine.lagged_csb(net_mtm, time_grid)  # (n_paths, T)
        logger.debug(
            "Step 2 complete: VM CSB mean=%.2f  lagged_CSB mean=%.2f",
            float(csb.mean()), float(lagged_csb.mean()),
        )

        # 3. IM
        im: np.ndarray | None = None
        if self.im_engine is not None:
            shape = net_mtm.shape
            try:
                im = self.im_engine.compute_im(
                    trades=im_trades,
                    sensitivities=im_sensitivities,
                    shape=shape,
                )
                logger.debug("Step 3 complete: IM computed, mean=%.2f", float(im.mean()))
            except ValueError:
                warnings.warn("IM computation skipped: missing inputs.", UserWarning, stacklevel=2)
                logger.warning("IM computation skipped: missing im_trades or im_sensitivities")
        else:
            logger.debug("Step 3 skipped: no IM engine configured")

        # 4. Reconcile collateral account
        self.collateral.reset()
        vm_recv = np.maximum(csb, 0.0)
        vm_post = np.maximum(-csb, 0.0)
        self.collateral.receive_vm(vm_recv, "CASH_USD")
        self.collateral.post_vm(vm_post, "CASH_USD")
        if im is not None:
            self.collateral.receive_im(im, "CASH_USD", segregated=self.csa.segregated_im)

        # 5. Collateralised exposure paths
        total_lagged = lagged_csb + (im if im is not None else 0.0)

        # 5b. IM time profile for MVA (if im_engine and im_trades provided)
        im_profile: np.ndarray | None = None
        if self.im_engine is not None and im_trades is not None and funding_spread is not None:
            try:
                im_profile = self.im_engine.im_time_profile(im_trades, time_grid)
            except Exception:
                logger.warning("im_time_profile() failed; MVA will be 0")

        # 6. Bilateral metrics
        summary = self._calc.bilateral_summary(
            net_mtm=net_mtm,
            time_grid=time_grid,
            collateral_balance=total_lagged,
            mpor=self.csa.mpor,
            confidence=confidence,
            cp_hazard_rate=cp_hazard_rate,
            own_hazard_rate=own_hazard_rate,
            funding_spread=funding_spread,
            im_profile=im_profile,
            cost_of_capital=cost_of_capital,
        )

        summary.update(
            {
                "net_mtm": net_mtm,
                "csb": csb,
                "lagged_csb": lagged_csb,
                "im": im,
                "collateral": self.collateral,
                "netting_set": self.netting_set.name,
                "csa": self.csa,
            }
        )
        logger.info(
            "ISDAExposureCalculator.run complete: EPE=%.2f  PSE=%.2f  EEPE=%.2f",
            float(summary["epe"]), float(summary["pse"]), float(summary["eepe"]),
        )
        return summary
