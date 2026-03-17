"""Collateral account and haircut schedule for bilateral CSA agreements."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .csa import CSATerms


@dataclass
class HaircutSchedule:
    """Maps collateral asset type to regulatory/CSA haircut fraction (0–1).

    Haircuts reduce the collateral value recognised by the receiver, making
    the system conservative: a bond posted with a 2% haircut is only worth
    98% of its market value as collateral.

    Parameters
    ----------
    haircuts : dict[str, float]
        Asset type → haircut fraction. Keys should match those used in
        ``CSATerms.eligible_collateral``.
    default_haircut : float
        Haircut applied to unknown asset types (conservative default = 1.0
        means unknown assets have zero collateral value).
    """

    haircuts: dict[str, float] = field(
        default_factory=lambda: {
            "CASH_USD": 0.000,
            "CASH_EUR": 0.000,
            "UST_2Y":   0.005,
            "UST_5Y":   0.010,
            "UST_10Y":  0.020,
            "UST_30Y":  0.040,
            "EUR_GOVT_IG": 0.005,
            "CORP_IG":  0.100,
            "EQUITY":   0.150,
        }
    )
    default_haircut: float = 1.0  # unknown asset = no collateral value

    def apply(self, asset_type: str, market_value: np.ndarray | float) -> np.ndarray | float:
        """Return post-haircut collateral value: MV × (1 - h)."""
        h = self.haircuts.get(asset_type, self.default_haircut)
        return market_value * (1.0 - h)

    @classmethod
    def from_csa(cls, csa: CSATerms) -> "HaircutSchedule":
        """Build a HaircutSchedule from the eligible collateral in a CSA."""
        return cls(haircuts=dict(csa.eligible_collateral))


@dataclass
class _CollateralEntry:
    """A single collateral position (for internal tracking)."""

    asset_type: str
    amount: np.ndarray   # post-haircut value, shape (n_paths, T) or scalar
    is_im: bool = False
    segregated: bool = False


class CollateralAccount:
    """Tracks VM and IM collateral posted and received under a CSA.

    All amounts stored internally are **post-haircut market values** in the
    CSA settlement currency. The account distinguishes:

    - **VM** (Variation Margin): offsets current MtM; reduces EE and ENE.
    - **IM** (Initial Margin): covers MPOR gap risk; if segregated, reduces
      PFE but *not* current MtM (per EMIR/CFTC segregation rules).

    Parameters
    ----------
    haircut_schedule : HaircutSchedule | None
        Haircut table. If None, defaults to standard schedule.
    rehypothecation : bool
        Whether received VM can be reused. Affects FVA; not credit exposure.
    """

    def __init__(
        self,
        haircut_schedule: HaircutSchedule | None = None,
        rehypothecation: bool = True,
    ) -> None:
        self.haircut_schedule = haircut_schedule or HaircutSchedule()
        self.rehypothecation = rehypothecation
        self._posted: list[_CollateralEntry] = []
        self._received: list[_CollateralEntry] = []

    # ------------------------------------------------------------------
    # VM
    # ------------------------------------------------------------------

    def post_vm(self, amount: np.ndarray | float, asset_type: str = "CASH_USD") -> None:
        """Record VM posted by us to the counterparty.

        Increases our credit exposure to the counterparty (we lose this
        collateral if they default).
        """
        net_value = self.haircut_schedule.apply(asset_type, np.asarray(amount, dtype=float))
        self._posted.append(_CollateralEntry(asset_type, net_value, is_im=False))

    def receive_vm(self, amount: np.ndarray | float, asset_type: str = "CASH_USD") -> None:
        """Record VM received from the counterparty.

        Reduces our credit exposure (collateral offsets what they owe us).
        """
        net_value = self.haircut_schedule.apply(asset_type, np.asarray(amount, dtype=float))
        self._received.append(_CollateralEntry(asset_type, net_value, is_im=False))

    # ------------------------------------------------------------------
    # IM
    # ------------------------------------------------------------------

    def post_im(
        self,
        amount: np.ndarray | float,
        asset_type: str = "CASH_USD",
        segregated: bool = True,
    ) -> None:
        """Record IM posted to a (possibly segregated) account.

        Segregated IM posted by us is bankruptcy-remote: we get it back
        even if the counterparty defaults, so it does NOT increase our
        credit exposure to them.
        """
        net_value = self.haircut_schedule.apply(asset_type, np.asarray(amount, dtype=float))
        self._posted.append(
            _CollateralEntry(asset_type, net_value, is_im=True, segregated=segregated)
        )

    def receive_im(
        self,
        amount: np.ndarray | float,
        asset_type: str = "CASH_USD",
        segregated: bool = True,
    ) -> None:
        """Record IM received into a (possibly segregated) account.

        Segregated IM received from the counterparty reduces PFE but
        cannot be netted against the current MtM under EMIR/CFTC rules.
        """
        net_value = self.haircut_schedule.apply(asset_type, np.asarray(amount, dtype=float))
        self._received.append(
            _CollateralEntry(asset_type, net_value, is_im=True, segregated=segregated)
        )

    # ------------------------------------------------------------------
    # Aggregated values
    # ------------------------------------------------------------------

    def net_vm_value(self) -> np.ndarray:
        """Net VM collateral value from our perspective (received - posted).

        Positive = net VM receiver (reduces our EE).
        """
        received = self._sum_entries(self._received, im_only=False)
        posted = self._sum_entries(self._posted, im_only=False)
        return received - posted

    def net_im_received(self) -> np.ndarray:
        """IM received into segregated account.

        Reduces our PFE over the MPOR window. Under EMIR segregation,
        this is NOT netted against current MtM.
        """
        return self._sum_entries(self._received, im_only=True)

    def net_im_posted(self) -> np.ndarray:
        """IM posted to segregated account.

        Represents capital tied up (drives MVA) but does not affect our
        credit exposure to the counterparty.
        """
        return self._sum_entries(self._posted, im_only=True)

    def net_collateral_value(self, include_im: bool = False) -> np.ndarray:
        """Total net collateral from our perspective.

        Parameters
        ----------
        include_im : bool
            If True, include segregated IM in the net collateral value.
            Should be False for EE calculation (per EMIR) and True only
            for close-out netting analysis.
        """
        net_vm = self.net_vm_value()
        if include_im:
            return net_vm + self.net_im_received() - self.net_im_posted()
        return net_vm

    def reset(self) -> None:
        """Clear all posted and received collateral entries."""
        self._posted.clear()
        self._received.clear()

    def summary(self) -> dict:
        """Snapshot of current account positions."""
        return {
            "net_vm": self.net_vm_value(),
            "im_received": self.net_im_received(),
            "im_posted": self.net_im_posted(),
            "rehypothecation": self.rehypothecation,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sum_entries(self, entries: list[_CollateralEntry], im_only: bool) -> np.ndarray:
        total = np.float64(0.0)
        for e in entries:
            if e.is_im == im_only:
                total = total + e.amount
        return np.asarray(total, dtype=float)
