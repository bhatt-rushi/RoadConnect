from dataclasses import dataclass
from typing import List, TYPE_CHECKING
import shapely.geometry
import networkx as nx
from utils import funcs

if TYPE_CHECKING:
    from model.graph import GraphNode

@dataclass
class visualization_information:
    """
    Stores EWKT (Extended Well-Known Text) strings for visualizing various spatial
    components related to a graph node within a GIS environment.
    """
    graph: nx.DiGraph
    node: "GraphNode"

    def __get_srid(self) -> int:
        from .data.drains import _gdf as drains_gdf
        return drains_gdf.crs.to_epsg()

    def __get_ancestor_point_geoms(self) -> List[shapely.geometry.Point]:
        return list(nx.ancestors(self.graph, self.node.point))

    def __get_local_point_geoms(self) -> List[shapely.geometry.Point]:
        return [self.node.point]

    def __get_descendant_point_geoms(self) -> List[shapely.geometry.Point]:
        return list(nx.descendants(self.graph, self.node.point))

    def __get_ancestor_road_geoms(self) -> List[shapely.geometry.base.BaseGeometry]:
        from .data.roads import _gdf as road_gdf
        indices = []
        for ancestor_point in nx.ancestors(self.graph, self.node.point):
            nodedata = self.graph.nodes[ancestor_point]['nodedata']
            for idx_list in nodedata.road._local_indices.values():
                indices.extend(idx_list)

        if not indices:
            return []
        return list(road_gdf.loc[list(set(indices))].geometry)

    def __get_local_road_geoms(self) -> List[shapely.geometry.base.BaseGeometry]:
        from .data.roads import _gdf as road_gdf
        indices = []
        for idx_list in self.node.road._local_indices.values():
            indices.extend(idx_list)

        if not indices:
            return []
        return list(road_gdf.loc[list(set(indices))].geometry)

    def __get_descendant_road_geoms(self) -> List[shapely.geometry.base.BaseGeometry]:
        from .data.roads import _gdf as road_gdf
        indices = []
        for descendant_point in nx.descendants(self.graph, self.node.point):
            nodedata = self.graph.nodes[descendant_point]['nodedata']
            for idx_list in nodedata.road._local_indices.values():
                indices.extend(idx_list)

        if not indices:
            return []
        return list(road_gdf.loc[list(set(indices))].geometry)

    def __get_ancestor_flowpath_geoms(self) -> List[shapely.geometry.base.BaseGeometry]:
        from .data.flowpaths import _gdf as flowpath_gdf
        indices = []
        ancestors = nx.ancestors(self.graph, self.node.point)
        for ancestor_point in ancestors:
            for predecessor_point in self.graph.predecessors(ancestor_point):
                predecessor_data = self.graph.nodes[predecessor_point]['nodedata']
                if predecessor_data.index_of_flowpath_to_child is not None:
                    indices.append(predecessor_data.index_of_flowpath_to_child)

        if not indices:
            return []
        return list(flowpath_gdf.loc[list(set(indices))].geometry)

    def __get_source_flowpath_geoms(self) -> List[shapely.geometry.base.BaseGeometry]:
        from .data.flowpaths import _gdf as flowpath_gdf
        indices = []
        for predecessor_point in self.graph.predecessors(self.node.point):
            predecessor_data = self.graph.nodes[predecessor_point]['nodedata']
            if predecessor_data.index_of_flowpath_to_child is not None:
                indices.append(predecessor_data.index_of_flowpath_to_child)

        if not indices:
            return []
        return list(flowpath_gdf.loc[list(set(indices))].geometry)

    def __get_descendant_flowpath_geoms(self) -> List[shapely.geometry.base.BaseGeometry]:
        from .data.flowpaths import _gdf as flowpath_gdf
        indices = []
        descendants = nx.descendants(self.graph, self.node.point)
        for descendant_point in descendants:
            for predecessor_point in self.graph.predecessors(descendant_point):
                predecessor_data = self.graph.nodes[predecessor_point]['nodedata']
                if predecessor_data.index_of_flowpath_to_child is not None:
                    indices.append(predecessor_data.index_of_flowpath_to_child)

        if not indices:
            return []
        return list(flowpath_gdf.loc[list(set(indices))].geometry)

    @property
    def ancestor_ewkt(self) -> str:
        """
        Combines and returns the EWKT strings for all ancestor-related geometries:
        points, roads, and flowpaths.
        """
        geometries = (
            self.__get_ancestor_point_geoms() +
            self.__get_ancestor_road_geoms() +
            self.__get_ancestor_flowpath_geoms()
        )
        return funcs.generate_wkt(geometries, self.__get_srid())

    @property
    def local_ewkt(self)-> str:
        """
        Combines and returns the EWKT strings for all local geometries:
        the current node's point, local roads, and source flowpaths.
        """
        geometries = (
            self.__get_local_point_geoms() +
            self.__get_local_road_geoms() +
            self.__get_source_flowpath_geoms()
        )
        return funcs.generate_wkt(geometries, self.__get_srid())

    @property
    def descendant_ewkt(self) -> str:
        """
        Combines and returns the EWKT strings for all descendant-related geometries:
        points, roads, and flowpaths.
        """
        geometries = (
            self.__get_descendant_point_geoms() +
            self.__get_descendant_road_geoms() +
            self.__get_descendant_flowpath_geoms()
        )
        return funcs.generate_wkt(geometries, self.__get_srid())
