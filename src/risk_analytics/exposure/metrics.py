from __future__ import annotations

import numpy as np


class ExposureCalculator:
    """Computes counterparty credit exposure metrics from MTM path arrays.

    All methods accept an MTM array of shape (n_paths, T) representing the
    mark-to-market value of a trade (or netted portfolio) at each time step.
    """

    def exposure_profile(self, mtm: np.ndarray) -> np.ndarray:
        """Positive exposure at each path and time step.

        EE(t, path) = max(V(t, path), 0)

        Returns
        -------
        np.ndarray, shape (n_paths, T)
        """
        return np.maximum(mtm, 0.0)

    def expected_exposure(self, mtm: np.ndarray) -> np.ndarray:
        """Expected Exposure (EE) profile: mean of positive exposures.

        EE(t) = E[max(V(t), 0)]

        Returns
        -------
        np.ndarray, shape (T,)
        """
        return self.exposure_profile(mtm).mean(axis=0)

    def pse(self, mtm: np.ndarray) -> float:
        """Pre-Settlement Exposure: peak of the EE profile over time.

        PSE = max_t { EE(t) }

        Returns
        -------
        float
        """
        return float(self.expected_exposure(mtm).max())

    def epe(self, mtm: np.ndarray, time_grid: np.ndarray) -> float:
        """Effective Positive Exposure: time-weighted average of EE(t).

        EPE = (1/T) ∫_0^T EE(t) dt

        Approximated via trapezoidal integration.

        Returns
        -------
        float
        """
        ee = self.expected_exposure(mtm)
        T_total = time_grid[-1] - time_grid[0]
        if T_total == 0:
            return float(ee[0])
        return float(np.trapezoid(ee, time_grid) / T_total)

    def pfe(self, mtm: np.ndarray, confidence: float = 0.95) -> np.ndarray:
        """Potential Future Exposure: exposure at a given confidence level.

        PFE(t, α) = quantile_α{ max(V(t), 0) }

        Parameters
        ----------
        confidence : float
            Confidence level, e.g. 0.95 for 95th percentile.

        Returns
        -------
        np.ndarray, shape (T,)
        """
        positive_exposure = self.exposure_profile(mtm)
        return np.quantile(positive_exposure, confidence, axis=0)

    def exposure_summary(
        self, mtm: np.ndarray, time_grid: np.ndarray, confidence: float = 0.95
    ) -> dict:
        """Compute all exposure metrics in one call.

        Returns
        -------
        dict with keys: 'ee_profile', 'pfe_profile', 'pse', 'epe', 'confidence'
        """
        return {
            "ee_profile": self.expected_exposure(mtm),
            "pfe_profile": self.pfe(mtm, confidence),
            "pse": self.pse(mtm),
            "epe": self.epe(mtm, time_grid),
            "confidence": confidence,
            "time_grid": time_grid,
        }
