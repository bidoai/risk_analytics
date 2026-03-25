from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np

from pyxva.core.paths import SimulationResult
from pyxva.core.stateful import StatefulPricer, PathState
from pyxva.exposure.csa import CSATerms
from pyxva.exposure.streaming.vm_stepper import REGVMStepper

logger = logging.getLogger(__name__)


@dataclass
class StreamingExposureResult:
    """Output from a single StreamingExposureEngine run.

    All profiles are over the simulation time grid.
    """
    time_grid: np.ndarray
    ee_profile: np.ndarray        # expected exposure E[max(V,0)] per step
    ene_profile: np.ndarray       # expected negative exposure E[max(-V,0)] per step
    pfe_profile: np.ndarray       # peak future exposure at given confidence
    ee_mpor_profile: np.ndarray   # EE under MPOR: CE(t) = max(V(t) - CSB(t-mpor), 0)
    peak_ee: float                # max of ee_profile
    peak_pfe: float               # max of pfe_profile


class StreamingExposureEngine:
    """Memory-efficient Monte Carlo exposure engine.

    Computes EE, PFE, and MPOR-adjusted EE by stepping through the simulation
    time grid one step at a time, never materialising the full
    ``(n_paths, T)`` MTM matrix in memory.

    Supports both standard ``Pricer`` and ``StatefulPricer`` instruments.
    For stateful pricers the path-state is tracked across steps.

    Multi-model netting sets
    ------------------------
    Trades may be:

    * ``(trade_id: str, pricer)`` tuples — all priced against a single
      ``SimulationResult`` passed to ``run()``.
    * ``Trade`` objects (anything with ``model_name`` and ``pricer``
      attributes) — each trade's pricer is called with the ``SimulationResult``
      whose key matches ``trade.model_name``.

    When using ``Trade`` objects, pass ``results`` as a
    ``dict[str, SimulationResult]`` keyed by model name.  All results must
    share the same time grid.

    Parameters
    ----------
    trades : list
        Mix of ``(trade_id, pricer)`` tuples and/or ``Trade`` objects.
    csa : CSATerms
        CSA parameters driving the VM stepper.
    confidence : float
        Quantile for PFE (default 0.95).
    """

    def __init__(
        self,
        trades: list,
        csa: CSATerms,
        confidence: float = 0.95,
    ) -> None:
        self.trades = trades
        self.csa = csa
        self.confidence = confidence

    def run(
        self,
        results: Union[SimulationResult, dict],
        mpor_steps: Optional[int] = None,
    ) -> StreamingExposureResult:
        """Run the streaming exposure calculation.

        Parameters
        ----------
        results : SimulationResult or dict[str, SimulationResult]
            Single-model result (backwards-compatible) or a dict mapping
            model name → ``SimulationResult`` for multi-model netting sets.
        mpor_steps : int, optional
            Number of time steps corresponding to the MPOR look-ahead.
            Defaults to the number of steps nearest to ``csa.mpor`` years.

        Returns
        -------
        StreamingExposureResult
        """
        # --- Normalise to dict ---
        if isinstance(results, SimulationResult):
            result_dict: dict[str, SimulationResult] = {results.model_name: results}
        else:
            result_dict = results

        # Validate all results share the same time grid
        grids = [r.time_grid for r in result_dict.values()]
        if len(grids) > 1:
            ref = grids[0]
            for g in grids[1:]:
                if g.shape != ref.shape or not np.allclose(g, ref):
                    raise ValueError(
                        "All SimulationResults in a multi-model netting set must "
                        "share the same time grid.  Found mismatched grids."
                    )

        # Reference result for grid / n_paths / n_steps
        ref_result = next(iter(result_dict.values()))
        time_grid = ref_result.time_grid
        n_paths = ref_result.n_paths
        n_steps = ref_result.n_steps

        # Resolve MPOR look-ahead in steps
        if mpor_steps is None:
            dt_grid = np.mean(np.diff(time_grid)) if n_steps > 1 else 1 / 252
            mpor_steps = max(1, int(round(self.csa.mpor / dt_grid)))

        # --- Initialise per-pricer stateful state ---
        states: list[Optional[PathState]] = []
        for trade in self.trades:
            pricer = trade.pricer if hasattr(trade, "model_name") else trade[1]
            if isinstance(pricer, StatefulPricer):
                states.append(pricer.allocate_state(n_paths))
            else:
                states.append(None)

        vm_stepper = REGVMStepper(self.csa, n_paths)

        # Buffers: full profiles
        ee_profile = np.zeros(n_steps)
        ene_profile = np.zeros(n_steps)
        pfe_profile = np.zeros(n_steps)
        ee_mpor_profile = np.zeros(n_steps)

        # Ring buffer of settled CSB values for proper MPOR calculation.
        # At each step i we store the CSB *after* the VM call (i.e. the settled
        # collateral at t_i).  For EE_MPOR at step i we look up the CSB from
        # mpor_steps ago: CE_mpor(t_i) = max(V(t_i) - CSB(t_{i-mpor_steps}), 0).
        csb_buf: list[Optional[np.ndarray]] = [None] * mpor_steps

        logger.info(
            "StreamingExposureEngine: %d trades, %d paths, %d steps, MPOR=%d steps",
            len(self.trades), n_paths, n_steps, mpor_steps,
        )

        for i, t in enumerate(time_grid):
            # --- Price all trades at this step ---
            net_mtm = np.zeros(n_paths)
            new_states: list[Optional[PathState]] = []

            for j, trade in enumerate(self.trades):
                if hasattr(trade, "model_name"):
                    trade_result = result_dict[trade.model_name]
                    pricer = trade.pricer
                else:
                    _, pricer = trade
                    trade_result = ref_result

                if isinstance(pricer, StatefulPricer):
                    mtm_j, new_state_j = pricer.step(trade_result, t, i, states[j])
                    new_states.append(new_state_j)
                else:
                    mtm_j = pricer.price_at(trade_result, i)
                    new_states.append(None)
                net_mtm += mtm_j

            states = new_states

            # --- VM margin step ---
            post_margin = vm_stepper.step(net_mtm)

            # Store the settled CSB after this margin call
            csb_buf[i % mpor_steps] = vm_stepper.csb

            # --- Accumulate exposure statistics ---
            ee_profile[i] = float(np.mean(post_margin))
            ene_profile[i] = float(np.mean(np.maximum(-net_mtm, 0.0)))
            pfe_profile[i] = float(np.quantile(post_margin, self.confidence))

            # --- Proper MPOR EE ---
            # CE_mpor(t_i) = max(V(t_i) - CSB(t_{i-mpor_steps}), 0)
            # Only defined once we have mpor_steps of history.
            if i >= mpor_steps:
                old_csb = csb_buf[(i - mpor_steps) % mpor_steps]
                ee_mpor_profile[i] = float(np.mean(np.maximum(net_mtm - old_csb, 0.0)))

        return StreamingExposureResult(
            time_grid=time_grid,
            ee_profile=ee_profile,
            ene_profile=ene_profile,
            pfe_profile=pfe_profile,
            ee_mpor_profile=ee_mpor_profile,
            peak_ee=float(np.max(ee_profile)),
            peak_pfe=float(np.max(pfe_profile)),
        )
