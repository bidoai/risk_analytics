from __future__ import annotations

import logging
from multiprocessing.shared_memory import SharedMemory
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class SimulationSharedMemory:
    """Context manager for zero-copy sharing of simulation paths across worker processes.

    Allocates one named ``SharedMemory`` block per model result and
    exposes a lightweight descriptor dict that worker processes can use
    to reconstruct numpy array views without copying data.

    Usage::

        with SimulationSharedMemory(simulation_results) as shm:
            descriptors = shm.descriptors   # pass to worker via pickle
            # Workers call SimulationSharedMemory.attach(descriptors) to get arrays
            # ... ProcessPoolExecutor.submit(worker, descriptors, ...) ...

        # SharedMemory is automatically released on __exit__

    Notes
    -----
    SharedMemory blocks are identified by name.  Names are chosen to be
    unique per run (using ``id(paths_array)``), but callers should treat
    the context manager as the sole owner and not share descriptors
    outside the ``with`` block.

    Worker processes must call ``SimulationSharedMemory.attach()`` and
    ``SimulationSharedMemory.detach()`` explicitly (or use the
    ``attached_result`` context manager).
    """

    def __init__(self, simulation_results: dict) -> None:
        """
        Parameters
        ----------
        simulation_results : dict[str, SimulationResult]
            Output of MonteCarloEngine.run().
        """
        self._simulation_results = simulation_results
        self._shm_blocks: dict[str, SharedMemory] = {}
        self._descriptors: dict[str, dict] = {}

    def __enter__(self) -> "SimulationSharedMemory":
        for model_name, result in self._simulation_results.items():
            paths = result.paths  # (n_paths, T, n_factors)
            shm = SharedMemory(create=True, size=paths.nbytes)
            buf = np.ndarray(paths.shape, dtype=paths.dtype, buffer=shm.buf)
            np.copyto(buf, paths)
            self._shm_blocks[model_name] = shm
            self._descriptors[model_name] = {
                "shm_name": shm.name,
                "shape": paths.shape,
                "dtype": str(paths.dtype),
                "time_grid": result.time_grid,
                "model_name": result.model_name,
                "factor_names": result.factor_names,
                "interpolation_space": result.interpolation_space,
            }
            logger.debug(
                "Allocated shared memory '%s' for model '%s': %.1f MB",
                shm.name, model_name, paths.nbytes / 1e6,
            )
        return self

    def __exit__(self, *args) -> None:
        for model_name, shm in self._shm_blocks.items():
            shm.close()
            shm.unlink()
            logger.debug("Released shared memory for model '%s'", model_name)
        self._shm_blocks.clear()
        self._descriptors.clear()

    @property
    def descriptors(self) -> dict:
        """Lightweight descriptor dict safe to pickle across process boundaries."""
        return dict(self._descriptors)

    # ------------------------------------------------------------------
    # Worker-side helpers
    # ------------------------------------------------------------------

    @staticmethod
    def attach(descriptors: dict) -> dict:
        """Attach to shared memory blocks in a worker process.

        Returns a ``{model_name: (SimulationResult, SharedMemory)}`` dict.
        Call ``SimulationSharedMemory.detach(attached)`` when done to
        avoid resource leaks.
        """
        from risk_analytics.core.paths import SimulationResult

        attached = {}
        for model_name, desc in descriptors.items():
            shm = SharedMemory(name=desc["shm_name"], create=False)
            paths = np.ndarray(
                desc["shape"],
                dtype=np.dtype(desc["dtype"]),
                buffer=shm.buf,
            )
            result = SimulationResult(
                time_grid=desc["time_grid"],
                paths=paths,
                model_name=desc["model_name"],
                factor_names=desc["factor_names"],
                interpolation_space=desc["interpolation_space"],
            )
            attached[model_name] = (result, shm)
        return attached

    @staticmethod
    def detach(attached: dict) -> None:
        """Close shared memory handles opened by ``attach()``."""
        for _, (_, shm) in attached.items():
            shm.close()

    @staticmethod
    def results_from_attached(attached: dict) -> dict:
        """Extract the SimulationResult dict from an ``attach()`` return value."""
        return {name: result for name, (result, _) in attached.items()}
