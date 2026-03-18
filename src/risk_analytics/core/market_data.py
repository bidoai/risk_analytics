from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import copy

from risk_analytics.core.yield_curve import YieldCurve, Interpolation


class BumpType(Enum):
    PARALLEL = "parallel"
    SLOPE = "slope"
    POINT = "point"


@dataclass
class ScenarioBump:
    key: str
    size: float
    bump_type: BumpType = BumpType.PARALLEL
    tenor: Optional[float] = None


class MarketData:
    """
    Central container for all market data. Serves as:
    - Calibration input for stochastic models
    - Discount / forward provider during pricing
    - Stress-test target via bump() and scenario()
    """

    def __init__(
        self,
        curves: Optional[dict[str, YieldCurve]] = None,
        spots: Optional[dict[str, float]] = None,
        vols: Optional[dict[str, float]] = None,
        forward_curves: Optional[dict[str, YieldCurve]] = None,
    ):
        self.curves: dict[str, YieldCurve] = curves or {}
        self.spots: dict[str, float] = spots or {}
        self.vols: dict[str, float] = vols or {}
        self.forward_curves: dict[str, YieldCurve] = forward_curves or {}

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def discount_factor(self, curve: str, t: float) -> float:
        return self._get_curve(curve).discount_factor(t)

    def zero_rate(self, curve: str, t: float) -> float:
        return self._get_curve(curve).zero_rate(t)

    def forward_rate(self, curve: str, t1: float, t2: float) -> float:
        return self._get_curve(curve).forward_rate(t1, t2)

    def spot(self, asset: str) -> float:
        if asset not in self.spots:
            raise KeyError(f"Spot not found for '{asset}'. Available: {list(self.spots)}")
        return self.spots[asset]

    def vol(self, asset: str) -> float:
        if asset not in self.vols:
            raise KeyError(f"Vol not found for '{asset}'. Available: {list(self.vols)}")
        return self.vols[asset]

    def forward_curve(self, asset: str) -> YieldCurve:
        if asset not in self.forward_curves:
            raise KeyError(f"Forward curve not found for '{asset}'. Available: {list(self.forward_curves)}")
        return self.forward_curves[asset]

    # ------------------------------------------------------------------
    # Stress testing — always returns a new copy, never mutates self
    # ------------------------------------------------------------------

    def bump(
        self,
        key: str,
        size: float,
        bump_type: BumpType = BumpType.PARALLEL,
        tenor: Optional[float] = None,
    ) -> "MarketData":
        """Return a bumped copy of this MarketData.

        Bump conventions:
          - Yield curves / forward curves: additive shift on zero rates (e.g. size=0.001 → +10bps)
          - Spots: multiplicative shift (e.g. size=0.10 → +10%)
          - Vols: additive shift on implied vol (e.g. size=0.02 → +2 vol points)
        """
        md = self._copy()
        if key in md.curves:
            md.curves[key] = _bump_curve(md.curves[key], size, bump_type, tenor)
        elif key in md.forward_curves:
            md.forward_curves[key] = _bump_curve(md.forward_curves[key], size, bump_type, tenor)
        elif key in md.spots:
            if bump_type != BumpType.PARALLEL:
                raise ValueError("Spots only support PARALLEL bumps.")
            md.spots[key] = md.spots[key] * (1.0 + size)
        elif key in md.vols:
            if bump_type != BumpType.PARALLEL:
                raise ValueError("Vols only support PARALLEL bumps.")
            md.vols[key] = md.vols[key] + size
        else:
            raise KeyError(f"Key '{key}' not found in any market data container.")
        return md

    def scenario(self, bumps: list[ScenarioBump]) -> "MarketData":
        """Apply multiple bumps sequentially. Returns a new copy."""
        md = self
        for b in bumps:
            md = md.bump(b.key, b.size, b.bump_type, b.tenor)
        return md

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict) -> "MarketData":
        """Build from a plain dict (e.g. parsed from YAML)."""
        curves = {}
        for name, cfg in data.get("curves", {}).items():
            interp = Interpolation[cfg.get("interpolation", "LOG_LINEAR").upper()]
            curves[name] = YieldCurve(
                tenors=cfg["tenors"],
                zero_rates=cfg["rates"],
                interpolation=interp,
            )

        forward_curves = {}
        for name, cfg in data.get("forward_curves", {}).items():
            interp = Interpolation[cfg.get("interpolation", "LOG_LINEAR").upper()]
            forward_curves[name] = YieldCurve(
                tenors=cfg["tenors"],
                zero_rates=cfg["rates"],
                interpolation=interp,
            )

        return cls(
            curves=curves,
            spots=dict(data.get("spots", {})),
            vols=dict(data.get("vols", {})),
            forward_curves=forward_curves,
        )

    @classmethod
    def from_yaml(cls, path: str) -> "MarketData":
        import yaml  # optional dependency

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data.get("market_data", data))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _copy(self) -> "MarketData":
        return MarketData(
            curves=dict(self.curves),
            spots=dict(self.spots),
            vols=dict(self.vols),
            forward_curves=dict(self.forward_curves),
        )

    def _get_curve(self, name: str) -> YieldCurve:
        if name not in self.curves:
            raise KeyError(f"Curve '{name}' not found. Available: {list(self.curves)}")
        return self.curves[name]

    def __repr__(self) -> str:
        return (
            f"MarketData(curves={list(self.curves)}, "
            f"spots={list(self.spots)}, "
            f"vols={list(self.vols)}, "
            f"forward_curves={list(self.forward_curves)})"
        )


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _bump_curve(
    curve: YieldCurve,
    size: float,
    bump_type: BumpType,
    tenor: Optional[float],
) -> YieldCurve:
    tenors = list(curve._t)
    rates = list(curve._z)

    if bump_type == BumpType.PARALLEL:
        rates = [r + size for r in rates]

    elif bump_type == BumpType.SLOPE:
        # Linear tilt: short end gets -size, long end gets +size
        t_min, t_max = tenors[0], tenors[-1]
        span = t_max - t_min if t_max > t_min else 1.0
        rates = [r + size * (t - t_min) / span * 2 - size for r, t in zip(rates, tenors)]

    elif bump_type == BumpType.POINT:
        if tenor is None:
            raise ValueError("tenor must be specified for POINT bumps.")
        # Find closest pillar and bump it
        idx = int(min(range(len(tenors)), key=lambda i: abs(tenors[i] - tenor)))
        rates[idx] = rates[idx] + size

    return YieldCurve(
        tenors=tenors,
        zero_rates=rates,
        interpolation=curve.interpolation,
    )
