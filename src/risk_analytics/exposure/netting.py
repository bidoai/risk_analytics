from __future__ import annotations

import numpy as np

from risk_analytics.core.base import Pricer
from risk_analytics.core.paths import SimulationResult
from .metrics import ExposureCalculator


class NettingSet:
    """Aggregate exposure across trades under a bilateral netting agreement.

    Under netting, the exposure is applied to the *net* MTM of the portfolio,
    not the sum of individual positive exposures. This reduces credit risk.

    Parameters
    ----------
    name : str
        Identifier for this netting set (e.g. counterparty name).
    """

    def __init__(self, name: str = "default") -> None:
        self.name = name
        self._trades: list[tuple[str, Pricer]] = []

    def add_trade(self, trade_id: str, pricer: Pricer) -> None:
        """Add a trade to the netting set.

        Parameters
        ----------
        trade_id : str
            Unique identifier for the trade.
        pricer : Pricer
            Pricing model for the trade.
        """
        self._trades.append((trade_id, pricer))

    def net_mtm(self, simulation_results: dict[str, SimulationResult]) -> np.ndarray:
        """Compute the net MTM across all trades in the netting set.

        Each pricer is called with the SimulationResult matching its model.
        If a pricer requires a specific model result, the simulation_results dict
        should contain the key matching that model's name.

        The engine tries each model's result until one is compatible. To be
        explicit, pricers should accept exactly the SimulationResult they need.

        Parameters
        ----------
        simulation_results : dict[str, SimulationResult]
            Output from MonteCarloEngine.run().

        Returns
        -------
        np.ndarray, shape (n_paths, T)
            Net MTM across all trades.
        """
        if not self._trades:
            raise ValueError("NettingSet has no trades.")

        net = None
        for trade_id, pricer in self._trades:
            mtm = self._price_trade(pricer, simulation_results, trade_id)
            if net is None:
                net = mtm
            else:
                net = net + mtm

        return net

    def exposure(
        self,
        simulation_results: dict[str, SimulationResult],
        time_grid: np.ndarray,
        confidence: float = 0.95,
    ) -> dict:
        """Compute full exposure summary for this netting set.

        Returns
        -------
        dict with keys: 'ee_profile', 'pfe_profile', 'pse', 'epe', 'net_mtm'
        """
        net = self.net_mtm(simulation_results)
        calc = ExposureCalculator()
        summary = calc.exposure_summary(net, time_grid, confidence)
        summary["net_mtm"] = net
        summary["netting_set"] = self.name
        return summary

    @property
    def trade_ids(self) -> list[str]:
        return [tid for tid, _ in self._trades]

    def _price_trade(
        self,
        pricer: Pricer,
        results: dict[str, SimulationResult],
        trade_id: str,
    ) -> np.ndarray:
        """Try to price a trade against available simulation results."""
        errors = []
        for model_name, result in results.items():
            try:
                return pricer.price(result)
            except (KeyError, ValueError, IndexError) as e:
                errors.append(f"  {model_name}: {e}")

        raise RuntimeError(
            f"Could not price trade '{trade_id}' with any available simulation result.\n"
            + "\n".join(errors)
        )
