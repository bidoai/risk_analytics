"""Bilateral ISDA exposure calculator.

Combines netting, VM, IM, collateral, and all bilateral exposure metrics
into a single orchestrated calculator.

Key references:
- Gregory (2012) "Counterparty Credit Risk"
- Basel III SA-CCR (CRE52) / IMM (CRE53)
- ISDA 2016 VM CSA / 2018 IM CSA
"""
from __future__ import annotations

import warnings

import numpy as np

from risk_analytics.core.paths import SimulationResult
from .csa import CSATerms
from .collateral import CollateralAccount, HaircutSchedule
from .margin.vm import REGVMEngine
from .margin.im import REGIMEngine
from .margin.simm import SimmSensitivities
from .metrics import ExposureCalculator
from .netting import NettingSet


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
        hazard_rate: float,
        lgd: float = 0.6,
    ) -> float:
        """Approximate unilateral CVA via hazard rate.

        CVA ≈ LGD × Σ_i EE(t_i) × [exp(-λ t_{i-1}) - exp(-λ t_i)]

        This is the piecewise-constant hazard rate approximation. For bilateral
        CVA, call both ``cva_approx`` (own default) and ``dva_approx``.

        Parameters
        ----------
        hazard_rate : float
            Counterparty default intensity λ (annualised, e.g. spread/LGD).
        lgd : float
            Loss Given Default (fraction, default 60%).

        Returns
        -------
        float
        """
        ee = self.expected_exposure(mtm)
        return self._integral_cva(time_grid, ee, hazard_rate, lgd)

    def dva_approx(
        self,
        mtm: np.ndarray,
        time_grid: np.ndarray,
        own_hazard_rate: float,
        own_lgd: float = 0.6,
    ) -> float:
        """Approximate DVA (own-default benefit).

        DVA ≈ LGD_own × Σ_i |ENE(t_i)| × [exp(-λ_own t_{i-1}) - exp(-λ_own t_i)]

        Returns
        -------
        float
        """
        ene_profile = np.abs(self.ene(mtm))
        return self._integral_cva(time_grid, ene_profile, own_hazard_rate, own_lgd)

    def bilateral_cva(
        self,
        mtm: np.ndarray,
        time_grid: np.ndarray,
        cp_hazard_rate: float,
        own_hazard_rate: float,
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
        cp_hazard_rate: float | None = None,
        own_hazard_rate: float | None = None,
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

        return result

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _integral_cva(
        time_grid: np.ndarray,
        ee_profile: np.ndarray,
        hazard_rate: float,
        lgd: float,
    ) -> float:
        """Discrete integral: LGD × Σ EE(t_i) × [PD(t_{i-1}, t_i)]."""
        total = 0.0
        for i in range(1, len(time_grid)):
            pd_i = np.exp(-hazard_rate * time_grid[i - 1]) - np.exp(-hazard_rate * time_grid[i])
            total += ee_profile[i] * pd_i
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
        cp_hazard_rate: float | None = None,
        own_hazard_rate: float | None = None,
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
        # 1. Net MTM
        net_mtm = self.netting_set.net_mtm(simulation_results)  # (n_paths, T)

        # 2. VM: Credit Support Balance (stationary approximation)
        csb = self.vm_engine.credit_support_balance(net_mtm)  # (n_paths, T)
        lagged_csb = self.vm_engine.lagged_csb(net_mtm, time_grid)  # (n_paths, T)

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
            except ValueError:
                warnings.warn("IM computation skipped: missing inputs.", UserWarning, stacklevel=2)

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

        # 6. Bilateral metrics
        summary = self._calc.bilateral_summary(
            net_mtm=net_mtm,
            time_grid=time_grid,
            collateral_balance=total_lagged,
            mpor=self.csa.mpor,
            confidence=confidence,
            cp_hazard_rate=cp_hazard_rate,
            own_hazard_rate=own_hazard_rate,
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
        return summary
