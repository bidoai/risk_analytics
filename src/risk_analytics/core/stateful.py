from __future__ import annotations

import copy
from abc import abstractmethod
from dataclasses import dataclass

import numpy as np

from risk_analytics.core.base import Pricer
from risk_analytics.core.paths import SimulationResult


@dataclass
class PathState:
    """Base class for per-path mutable state in StatefulPricers.

    Subclasses define named numpy arrays as dataclass fields, each
    of shape (n_paths,). The ``allocate`` class method constructs a
    fresh zero/False-initialised state; ``copy`` returns a deep copy
    so that state from one time step does not alias the next.

    Example subclass::

        @dataclass
        class MyState(PathState):
            running_max: np.ndarray   # (n_paths,)

            @classmethod
            def allocate(cls, n_paths: int) -> "MyState":
                return cls(running_max=np.zeros(n_paths))
    """

    @classmethod
    def allocate(cls, n_paths: int) -> "PathState":
        """Return a zero-initialised state for n_paths."""
        raise NotImplementedError(
            f"{cls.__name__} must implement allocate(n_paths)."
        )

    def copy(self) -> "PathState":
        """Return a deep copy of this state object."""
        return copy.deepcopy(self)


class StatefulPricer(Pricer):
    """Abstract pricer for path-dependent instruments.

    Use this when correct pricing requires accumulating information
    along each simulated path (barrier monitoring, Asian averaging,
    target redemption features, etc.).  Subclasses implement:

    - ``allocate_state(n_paths)`` — return a fresh ``PathState``
    - ``step(result, t, t_idx, state)`` — advance one time step

    The base ``price()`` drives the step loop and assembles the full
    ``(n_paths, T)`` MTM matrix.  The base ``price_at()`` replays from
    ``t=0`` to ``t_idx`` to produce a single ``(n_paths,)`` slice.
    Both can be overridden for performance where appropriate.
    """

    @abstractmethod
    def allocate_state(self, n_paths: int) -> PathState:
        """Return a fresh initial state for n_paths."""
        ...

    @abstractmethod
    def step(
        self,
        result: SimulationResult,
        t: float,
        t_idx: int,
        state: PathState,
    ) -> tuple[np.ndarray, PathState]:
        """Advance the pricer by one time step.

        Parameters
        ----------
        result : SimulationResult
        t : float
            Current time in years.
        t_idx : int
            Index into ``result.time_grid``.
        state : PathState
            Per-path state entering this step (treat as read-only).

        Returns
        -------
        (mtm, new_state)
            ``mtm`` has shape ``(n_paths,)``.
        """
        ...

    def price(self, result: SimulationResult) -> np.ndarray:
        """Full ``(n_paths, T)`` MTM matrix via sequential ``step()`` calls."""
        n_paths = result.n_paths
        n_steps = result.n_steps
        out = np.zeros((n_paths, n_steps))
        state = self.allocate_state(n_paths)
        for i, t in enumerate(result.time_grid):
            mtm, state = self.step(result, t, i, state)
            out[:, i] = mtm
        return out

    def price_at(self, result: SimulationResult, t_idx: int) -> np.ndarray:
        """MTM at a single time index by replaying from t=0.

        This is equivalent to ``self.price(result)[:, t_idx]`` but
        makes it explicit that state must be accumulated from the start.
        Subclasses may override with a more efficient implementation
        when full replay is unnecessary.

        Returns
        -------
        np.ndarray, shape (n_paths,)
        """
        state = self.allocate_state(result.n_paths)
        mtm = np.zeros(result.n_paths)
        for i, t in enumerate(result.time_grid[: t_idx + 1]):
            mtm, state = self.step(result, t, i, state)
        return mtm
