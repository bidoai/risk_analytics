"""YieldCurve — interpolated zero-rate term structure.

Supports three interpolation methods with differing trade-offs:

* ``LINEAR``       — linear on zero rates. Simple, but produces kinked
                     instantaneous forward curves.
* ``LOG_LINEAR``   — linear on log discount factors, equivalent to
                     piecewise-constant instantaneous forwards. This is
                     the market-standard method for bootstrapping and
                     is the default.
* ``CUBIC_SPLINE`` — cubic spline on zero rates. Produces smooth
                     forward curves; preferred when the term structure
                     will be differentiated (e.g. Hull-White theta fitting).

All methods extrapolate flat beyond the supplied tenor range.
"""
from __future__ import annotations

import logging
from enum import Enum

import numpy as np
from scipy.interpolate import CubicSpline

logger = logging.getLogger(__name__)


class Interpolation(str, Enum):
    """Interpolation method for zero rates / discount factors."""

    LINEAR = "linear"
    LOG_LINEAR = "log_linear"
    CUBIC_SPLINE = "cubic_spline"


class YieldCurve:
    """Continuously-compounded zero-rate term structure with interpolation.

    Parameters
    ----------
    tenors : array-like
        Tenor points in years (strictly increasing, non-negative).
    zero_rates : array-like
        Continuously-compounded zero rates at each tenor (same length).
    interpolation : Interpolation | str
        Interpolation method. Default ``"log_linear"``.

    Examples
    --------
    >>> import numpy as np
    >>> from risk_analytics.core import YieldCurve
    >>> curve = YieldCurve([0.5, 1, 2, 5, 10],
    ...                    [0.038, 0.040, 0.042, 0.047, 0.053])
    >>> curve.zero_rate(1.5)          # interpolated between 1yr and 2yr
    >>> curve.discount_factor(5.0)    # P(0, 5)
    >>> curve.instantaneous_forward(2.0)   # f(0, 2)
    """

    def __init__(
        self,
        tenors: np.ndarray,
        zero_rates: np.ndarray,
        interpolation: Interpolation | str = Interpolation.LOG_LINEAR,
    ) -> None:
        self._t = np.asarray(tenors, dtype=float)
        self._z = np.asarray(zero_rates, dtype=float)

        if self._t.ndim != 1 or self._z.ndim != 1:
            raise ValueError("tenors and zero_rates must be 1-D arrays")
        if len(self._t) != len(self._z):
            raise ValueError(
                f"tenors and zero_rates must have the same length; "
                f"got {len(self._t)} and {len(self._z)}"
            )
        if len(self._t) < 2:
            raise ValueError("At least two tenor points are required")
        if not np.all(np.diff(self._t) > 0):
            raise ValueError("tenors must be strictly increasing")
        if self._t[0] < 0:
            raise ValueError("tenors must be non-negative")

        self.interpolation = Interpolation(interpolation)

        # Log discount factors at knots: log P(0, t) = -z(t) * t
        self._log_df = -self._z * self._t

        # Piecewise slopes used for LOG_LINEAR forwards
        self._fwd_rates = np.diff(self._log_df) / np.diff(self._t) * -1  # (N-1,)

        if self.interpolation == Interpolation.CUBIC_SPLINE:
            self._spline = CubicSpline(self._t, self._z, extrapolate=True)
            self._dspline = self._spline.derivative()

        logger.debug(
            "YieldCurve: %d tenors, interpolation=%s, "
            "z[0]=%.4f  z[-1]=%.4f  t[0]=%.2f  t[-1]=%.2f",
            len(self._t), self.interpolation.value,
            self._z[0], self._z[-1], self._t[0], self._t[-1],
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def zero_rate(self, t: float | np.ndarray) -> float | np.ndarray:
        """Interpolated continuously-compounded zero rate.

        Extrapolates flat beyond the supplied tenor range.

        Parameters
        ----------
        t : float or array-like
            Tenor(s) in years (non-negative).

        Returns
        -------
        float or np.ndarray matching the shape of ``t``.
        """
        t_arr, scalar = self._prepare(t)

        if self.interpolation == Interpolation.LINEAR:
            z = np.interp(t_arr, self._t, self._z)

        elif self.interpolation == Interpolation.LOG_LINEAR:
            ldf = np.interp(t_arr, self._t, self._log_df)
            # z = -log_df / t; at t=0 return the short-end rate
            with np.errstate(divide="ignore", invalid="ignore"):
                z = np.where(t_arr > 0, -ldf / t_arr, self._z[0])
            # np.interp extrapolates log_df flat, which yields z = log_df_boundary/t
            # (not constant). Override with flat zero-rate extrapolation instead.
            z = np.where(t_arr < self._t[0], self._z[0], z)
            z = np.where(t_arr > self._t[-1], self._z[-1], z)

        else:  # CUBIC_SPLINE — clamp extrapolation to boundary rates
            z = self._spline(t_arr)
            z = np.where(t_arr < self._t[0], self._z[0], z)
            z = np.where(t_arr > self._t[-1], self._z[-1], z)

        return float(z[0]) if scalar else z

    def discount_factor(self, t: float | np.ndarray) -> float | np.ndarray:
        """Zero-coupon bond price P(0, t) = exp(-z(t) · t).

        Parameters
        ----------
        t : float or array-like

        Returns
        -------
        float or np.ndarray
        """
        t_arr = np.asarray(t, dtype=float)
        scalar = t_arr.ndim == 0
        t_arr = np.atleast_1d(t_arr)
        z = self.zero_rate(t_arr)
        df = np.exp(-z * t_arr)
        return float(df[0]) if scalar else df

    def forward_rate(
        self, t1: float | np.ndarray, t2: float | np.ndarray
    ) -> float | np.ndarray:
        """Continuously-compounded period forward rate for [t1, t2].

        F(t1, t2) = -log(P(0,t2) / P(0,t1)) / (t2 - t1)

        Parameters
        ----------
        t1, t2 : float or array-like
            Start and end of the forward period in years (t2 > t1).

        Returns
        -------
        float or np.ndarray
        """
        t1_arr = np.asarray(t1, dtype=float)
        t2_arr = np.asarray(t2, dtype=float)
        scalar = t1_arr.ndim == 0 and t2_arr.ndim == 0
        t1_arr = np.atleast_1d(t1_arr)
        t2_arr = np.atleast_1d(t2_arr)

        tau = t2_arr - t1_arr
        if np.any(tau <= 0):
            raise ValueError("t2 must be strictly greater than t1")

        df1 = self.discount_factor(t1_arr)
        df2 = self.discount_factor(t2_arr)
        fwd = -np.log(df2 / df1) / tau
        return float(fwd[0]) if scalar else fwd

    def instantaneous_forward(self, t: float | np.ndarray) -> float | np.ndarray:
        """Instantaneous forward rate f(0, t) = -d/dt log P(0, t).

        For ``LOG_LINEAR``: piecewise constant (step function) — exact
        analytic derivative of the linear log-DF interpolant.

        For ``LINEAR``: f(0,t) = z(t) + t · dz/dt where dz/dt is the
        piecewise-constant slope of the linear zero-rate interpolant.

        For ``CUBIC_SPLINE``: f(0,t) = z(t) + t · z'(t) — smooth,
        using the analytic spline derivative.

        Extrapolates flat (holds the boundary forward rate) outside the
        supplied tenor range.

        Parameters
        ----------
        t : float or array-like

        Returns
        -------
        float or np.ndarray
        """
        t_arr, scalar = self._prepare(t)

        if self.interpolation == Interpolation.LOG_LINEAR:
            # Each interval has a constant forward = -slope of log_df
            # Flat extrapolation: use first / last interval rate outside range
            idx = np.searchsorted(self._t, t_arr, side="right") - 1
            idx = np.clip(idx, 0, len(self._fwd_rates) - 1)
            fwd = self._fwd_rates[idx]

        elif self.interpolation == Interpolation.LINEAR:
            z = self.zero_rate(t_arr)
            # dz/dt is the slope of the linear zero-rate interpolant
            dz_dt = np.diff(self._z) / np.diff(self._t)  # (N-1,)
            idx = np.searchsorted(self._t, t_arr, side="right") - 1
            idx = np.clip(idx, 0, len(dz_dt) - 1)
            fwd = z + t_arr * dz_dt[idx]

        else:  # CUBIC_SPLINE
            z = self._spline(t_arr)
            dz = self._dspline(t_arr)
            # Clamp to boundary values outside tenor range
            z = np.where(t_arr < self._t[0], self._z[0], z)
            z = np.where(t_arr > self._t[-1], self._z[-1], z)
            dz = np.where(t_arr < self._t[0], 0.0, dz)
            dz = np.where(t_arr > self._t[-1], 0.0, dz)
            fwd = z + t_arr * dz

        return float(fwd[0]) if scalar else fwd

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"YieldCurve(tenors={self._t.tolist()}, "
            f"zero_rates={self._z.tolist()}, "
            f"interpolation='{self.interpolation.value}')"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare(self, t: float | np.ndarray) -> tuple[np.ndarray, bool]:
        """Convert t to 1-D float array; return (array, was_scalar)."""
        t_arr = np.asarray(t, dtype=float)
        scalar = t_arr.ndim == 0
        return np.atleast_1d(t_arr), scalar
