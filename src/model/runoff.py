from dataclasses import dataclass, field
from typing import Dict

from utils import funcs, config

@dataclass
class RunoffInformation:

    _ancestor: Dict[str, float] = field(default_factory=dict)
    _local: Dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> Dict[str, float]: return funcs.combine_dict(self._ancestor, self._local)
    @property
    def sum(self) -> float: return funcs.sum_dict(self.total)

    def _calculate_local_runoff(self, area: Dict[str, float], rainfall_amount: float, coefficients: Dict[str, config.RoadTypeData]) -> None:

        for surface_type, surface_area in area.items():
            road_type_data: config.RoadTypeData = coefficients[surface_type] # The reason I don't have any error handling here is because I check to make sure that all road types exist in data/roads.py with __vd_road_types()

            # Calculate runoff volume: Area * Rainfall * Runoff Coefficient
            # Assumes rainfall is in mm and area is in square meters
            runoff_volume = surface_area * (rainfall_amount / 1000) * road_type_data['runoff_coefficient']

            self._local[surface_type] = runoff_volume
