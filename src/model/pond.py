from dataclasses import dataclass
import numpy as np

from utils import funcs

@dataclass
class PondInformation:
    max_capacity: float
    used_capacity: float

    _runoff_in: float | None = None
    _sediment_in: float | None = None

    @property
    def _available_capacity(self) -> float: return self.max_capacity - self.used_capacity

    @property
    def _trapped_runoff(self) -> float:
        if self._runoff_in is None:
            raise RuntimeError("Someone wrote bad code... PondInformation doesn't know _runoff_in")
        return min(self._available_capacity, self._runoff_in)

    @property
    def _runoff_out(self) -> float:
        if self._runoff_in is None:
            raise RuntimeError("Someone wrote bad code... PondInformation doesn't know _runoff_in")
        return self._runoff_in - self._trapped_runoff

    @property
    def runoff_percent_difference(self) -> float:
        if self._runoff_in is None:
            raise RuntimeError("Someone wrote bad code... PondInformation doesn't know _runoff_in")
        return funcs.percent_difference(self._runoff_out, self._runoff_in)

    @property
    def _efficiency(self) -> float:
        if self._runoff_in is None:
            raise RuntimeError("Someone wrote bad code... PondInformation doesn't know _runoff_in")

        if self._runoff_out == 0:
            return 1.0
        else:
            return float(np.clip(
                -22 + ( ( 119 * ( self._available_capacity / self._runoff_in ) ) / ( 0.012 + 1.02 * (self._available_capacity / self._runoff_in ) ) ),
                0, # Minimum Efficiency
                100 # Max Efficiency
            ) / 100) # Convert to percent

    @property
    def _trapped_sediment(self) -> float:
        if self._sediment_in is None:
            raise RuntimeError("Someone wrote bad code... PondInformation doesn't know _sediment_in")
        return self._sediment_in * self._efficiency

    @property
    def _sediment_out(self) -> float:
        if self._sediment_in is None:
            raise RuntimeError("Someone wrote bad code... PondInformation doesn't know _sediment_in")
        return self._sediment_in - self._trapped_sediment

    @property
    def sediment_percent_difference(self) -> float:
        if self._sediment_in is None:
            raise RuntimeError("Someone wrote bad code... PondInformation doesn't know _sediment_in")
        return funcs.percent_difference(self._sediment_out, self._sediment_in)
