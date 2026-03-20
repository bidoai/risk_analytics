"""Term-structure hazard (default/funding) curve.

Models a piecewise-constant hazard rate curve that can be constructed
from a flat spread, from explicit tenors, or bootstrapped from CDS market data.

References
----------
- Gregory (2012) "Counterparty Credit Risk", Chapter 8
- O'Kane (2008) "Modelling Single-name and Multi-name Credit Derivatives"
"""
from __future__ import annotations

import numpy as np


class HazardCurve:
    """Piecewise-constant hazard rate curve.

    Stores a sorted sequence of (tenor, hazard_rate) pairs and provides
    survival probability and marginal default probability queries.

    Parameters
    ----------
    tenors : array-like, shape (N,)
        Tenor breakpoints in years. Must be positive and strictly increasing.
    hazard_rates : array-like, shape (N,)
        Piecewise-constant hazard rates (annualised) for each bucket.
        Bucket k covers (tenors[k-1], tenors[k]] with tenors[0] treated as
        the first bucket starting from t=0.
    """

    def __init__(self, tenors: np.ndarray, hazard_rates: np.ndarray) -> None:
        tenors = np.asarray(tenors, dtype=float)
        hazard_rates = np.asarray(hazard_rates, dtype=float)
        if tenors.shape != hazard_rates.shape:
            raise ValueError("tenors and hazard_rates must have the same length")
        if len(tenors) == 0:
            raise ValueError("tenors must be non-empty")
        if np.any(tenors <= 0):
            raise ValueError("tenors must be positive")
        if np.any(np.diff(tenors) <= 0):
            raise ValueError("tenors must be strictly increasing")
        if np.any(hazard_rates < 0):
            raise ValueError("hazard_rates must be non-negative")
        self._tenors = tenors
        self._hazard_rates = hazard_rates

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_flat_spread(cls, spread: float, lgd: float = 0.6) -> "HazardCurve":
        """Construct from a flat CDS spread (one-bucket approximation).

        Parameters
        ----------
        spread : float
            Flat credit/funding spread (annualised).
        lgd : float
            Loss Given Default fraction; hazard_rate = spread / lgd.
        """
        if lgd <= 0:
            raise ValueError("lgd must be positive")
        hazard = spread / lgd
        return cls(tenors=np.array([100.0]), hazard_rates=np.array([hazard]))

    @classmethod
    def from_tenors(
        cls, tenors: np.ndarray, hazard_rates: np.ndarray
    ) -> "HazardCurve":
        """Construct directly from tenor-hazard rate pairs."""
        return cls(tenors=np.asarray(tenors), hazard_rates=np.asarray(hazard_rates))

    @classmethod
    def calibrate(
        cls,
        cds_tenors: np.ndarray,
        cds_spreads: np.ndarray,
        recovery: float = 0.4,
        risk_free_rate: float = 0.0,
    ) -> "HazardCurve":
        """Bootstrap piecewise-constant hazard rates from CDS par spreads.

        Uses the standard CDS pricing formula with piecewise-constant hazard rates
        and a flat risk-free discount rate (r=0 default, suitable for most library
        use cases). Solves analytically for each bucket in sequence.

        CDS par spread formula (simplified):
            s(T_n) = (1-R) × protection_leg / risky_annuity

        Each bucket k with hazard rate λ_k in [T_{k-1}, T_k] is solved for
        in closed form given the contributions from all preceding buckets.

        Parameters
        ----------
        cds_tenors : array-like, shape (N,)
            CDS maturity tenors in years, strictly increasing.
        cds_spreads : array-like, shape (N,)
            Par CDS spreads (annualised) corresponding to each tenor.
        recovery : float
            Recovery rate fraction (0 ≤ R < 1). Default 40%.
        risk_free_rate : float
            Flat risk-free rate for discounting. Default 0 (conservative; omits
            discounting which is standard for library-level approximations).

        Returns
        -------
        HazardCurve
            Calibrated term-structure with len(cds_tenors) buckets.

        Raises
        ------
        ValueError
            If calibration produces a negative hazard rate (implies spread
            inconsistency — forward spread would be negative).
        """
        cds_tenors = np.asarray(cds_tenors, dtype=float)
        cds_spreads = np.asarray(cds_spreads, dtype=float)
        if cds_tenors.shape != cds_spreads.shape:
            raise ValueError("cds_tenors and cds_spreads must have the same length")
        if np.any(cds_spreads < 0):
            raise ValueError("cds_spreads must be non-negative")
        if np.any(np.diff(cds_tenors) <= 0):
            raise ValueError("cds_tenors must be strictly increasing")

        lgd = 1.0 - recovery
        n = len(cds_tenors)
        hazard_rates = np.empty(n)

        # Accumulated protection leg and premium (risky annuity) from prior buckets
        accum_prot = 0.0  # (1-R) × Σ_{j<k} disc_j × (Q(T_{j-1}) - Q(T_j))
        accum_prem = 0.0  # Σ_{j<k} Δt_j × disc_j × Q(T_j)

        q_prev = 1.0          # Q(T_0) = Q(0) = 1
        t_prev = 0.0

        for i in range(n):
            t_k = cds_tenors[i]
            s_k = cds_spreads[i]
            dt_k = t_k - t_prev
            disc_k = np.exp(-risk_free_rate * t_k)

            # Solve for α = exp(-λ_k × dt_k) such that CDS par spread = s_k
            # Closed-form derivation:
            #   s_k × (accum_prem + dt_k × disc_k × q_prev × α)
            #     = accum_prot + lgd × disc_k × q_prev × (1 - α)
            #
            #   α × disc_k × q_prev × (s_k × dt_k + lgd)
            #     = accum_prot + lgd × disc_k × q_prev - s_k × accum_prem
            #
            #   α = (accum_prot + lgd × disc_k × q_prev - s_k × accum_prem)
            #       / (disc_k × q_prev × (s_k × dt_k + lgd))

            denom = disc_k * q_prev * (s_k * dt_k + lgd)
            if denom == 0.0:
                # Degenerate: q_prev = 0 or disc_k = 0 — hazard rate → ∞
                hazard_rates[i] = 0.0
            else:
                numer = accum_prot + lgd * disc_k * q_prev - s_k * accum_prem
                alpha = numer / denom
                alpha = np.clip(alpha, 1e-12, 1.0)  # survival fraction ∈ (0, 1]
                lam_k = -np.log(alpha) / dt_k
                if lam_k < 0:
                    raise ValueError(
                        f"Calibration at tenor {t_k:.1f}yr produced negative hazard rate "
                        f"({lam_k:.4f}). Check CDS spreads for monotonicity violations."
                    )
                hazard_rates[i] = lam_k

            # Update survival probability and accumulators
            q_k = q_prev * np.exp(-hazard_rates[i] * dt_k)
            accum_prot += lgd * disc_k * q_prev * (1.0 - q_k / q_prev)  # = lgd × disc_k × (q_prev - q_k)
            accum_prem += dt_k * disc_k * q_k

            q_prev = q_k
            t_prev = t_k

        return cls(tenors=cds_tenors, hazard_rates=hazard_rates)

    # ------------------------------------------------------------------
    # Core queries
    # ------------------------------------------------------------------

    def survival_probability(self, t: float) -> float:
        """Q(t) — probability of surviving to time t.

        Q(t) = exp(−Σ_k λ_k × min(Δt_k, max(0, t − T_{k-1})))

        Parameters
        ----------
        t : float
            Time in years. Q(0) = 1. Returns Q(T_last) for t > T_last.
        """
        if t <= 0:
            return 1.0
        cumulative = 0.0
        t_prev = 0.0
        for tenor, lam in zip(self._tenors, self._hazard_rates):
            if t <= t_prev:
                break
            dt = min(tenor, t) - t_prev
            cumulative += lam * dt
            t_prev = tenor
            if t <= tenor:
                break
        else:
            # t > last tenor: extend flat (last hazard rate continues)
            if t > self._tenors[-1]:
                dt = t - self._tenors[-1]
                cumulative += self._hazard_rates[-1] * dt
        return float(np.exp(-cumulative))

    def marginal_default_prob(self, t_prev: float, t: float) -> float:
        """Q(t_prev) − Q(t) — probability of defaulting in (t_prev, t].

        Parameters
        ----------
        t_prev : float
            Start of interval.
        t : float
            End of interval. Must be ≥ t_prev.
        """
        if t <= t_prev:
            return 0.0
        return float(self.survival_probability(t_prev) - self.survival_probability(t))

    def survival_probability_vec(self, t_array: np.ndarray) -> np.ndarray:
        """Vectorized survival probabilities for an array of times.

        Equivalent to ``[survival_probability(t) for t in t_array]`` but performs
        a single O(N_buckets) pass over the piecewise-constant segments rather than
        O(len(t_array) × N_buckets) scalar calls.  Intended for bulk time-grid queries.

        Parameters
        ----------
        t_array : np.ndarray, shape (T,)
            Query times in years, must be non-decreasing.

        Returns
        -------
        np.ndarray, shape (T,)
        """
        t_array = np.asarray(t_array, dtype=float)
        cumulative = np.zeros(len(t_array))
        t_prev = 0.0
        for tenor, lam in zip(self._tenors, self._hazard_rates):
            dt = np.clip(np.minimum(tenor, t_array) - t_prev, 0.0, None)
            cumulative += lam * dt
            t_prev = tenor
        # Extend flat beyond the last tenor bucket
        if len(t_array) > 0 and t_array[-1] > self._tenors[-1]:
            cumulative += self._hazard_rates[-1] * np.maximum(t_array - self._tenors[-1], 0.0)
        return np.exp(-cumulative)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def tenors(self) -> np.ndarray:
        return self._tenors.copy()

    @property
    def hazard_rates(self) -> np.ndarray:
        return self._hazard_rates.copy()

    def __repr__(self) -> str:
        pairs = ", ".join(
            f"{t:.1f}yr→{h:.4f}" for t, h in zip(self._tenors, self._hazard_rates)
        )
        return f"HazardCurve([{pairs}])"
