from dataclasses import dataclass, field
from typing import Dict, Set
import shapely.geometry
import networkx as nx

from utils import funcs

@dataclass
class RoadInformation:

    # Helper references
    graph: nx.DiGraph | None = field(default=None, repr=False)
    point: shapely.geometry.point.Point | None = field(default=None, repr=False)

    @property
    def indices(self) -> Dict[str, Set[int]]:
        return funcs.combine_dict_set(self._ancestor_indices, self._local_indices)

    @property
    def length(self) -> Dict[str, float]:
        return self.__sum_geometry_stats(self.indices, 'LENGTH')

    @property
    def area(self) -> Dict[str, float]:
        return self.__sum_geometry_stats(self.indices, 'AREA')

    @property
    def _ancestor_indices(self) -> Dict[str, Set[int]]:
        if self.graph is None or self.point is None:
            raise KeyError("Missing required references in RoadInformation")

        from model.graph import GraphNode

        return funcs.combine_dict_set(*(
            nd.road._local_indices
            for n in nx.ancestors(self.graph, self.point)
            # Fetch 'nd', check if it's a GraphNode, and check if 'nd.road' is truthy
            if isinstance(nd := self.graph.nodes[n].get('nodedata'), GraphNode) and nd.road
        ))

    @property
    def _ancestor_length(self) -> Dict[str, float]:
        return self.__sum_geometry_stats(self._ancestor_indices, 'LENGTH')

    @property
    def _ancestor_area(self) -> Dict[str, float]:
        return self.__sum_geometry_stats(self._ancestor_indices, 'AREA')

    _local_indices: Dict[str, Set[int]] = field(default_factory=dict)

    @property
    def _local_length(self) -> Dict[str, float]:
        return self.__sum_geometry_stats(self._local_indices, 'LENGTH')

    @property
    def _local_area(self) -> Dict[str, float]:
        return self.__sum_geometry_stats(self._local_indices, 'AREA')

    def __sum_geometry_stats(self, indices_map: Dict[str, Set[int]], stat: str) -> Dict[str, float]:
        from model.data import roads

        match stat:
            case 'LENGTH':
                lookup_map = roads.length_map
            case 'AREA':
                lookup_map = roads.area_map
            case _:
                raise ValueError(f"Unknown stat type: {stat}")

        return {
                surface: sum(lookup_map[idx] for idx in road_indices)
                for surface, road_indices in indices_map.items()
            }
