from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List
import shapely.geometry
import networkx as nx
import numpy as np

from utils import funcs, config

class NodeType(Enum):
    DRAIN = 1 # Anywhere you have runoff converging (i.e., road drains and converging flowpaths)
    POND = 2
    TERMINATION = 3

@dataclass
class RoadInformation:

    # NOTE: Inherited
    _ancestor_indices: Dict[str, List[int]] = field(default_factory=dict)
    _ancestor_length: Dict[str, float] = field(default_factory=dict)
    _ancestor_area: Dict[str, float] = field(default_factory=dict)

    # NOTE: Local
    _local_indices: Dict[str, List[int]] = field(default_factory=dict)
    _local_length: Dict[str, float] = field(default_factory=dict)
    _local_area: Dict[str, float] = field(default_factory=dict)

    # NOTE: Inherited + Local
    @property
    def indices(self) -> Dict[str, List[int]]: return funcs.combine_dict_list(self._ancestor_indices, self._local_indices)

    @property
    def length(self) -> Dict[str, float]: return funcs.combine_dict(self._ancestor_length, self._local_length)

    @property
    def area(self) -> Dict[str, float]: return funcs.combine_dict(self._ancestor_area, self._local_area)

@dataclass
class RunoffInformation:

    # NOTE: Inherited
    _ancestor: Dict[str, float] = field(default_factory=dict)

    # NOTE: Local
    _local: Dict[str, float] = field(default_factory=dict)

    # NOTE: Inherited + Local
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

@dataclass
class SedimentInformation:

    # NOTE: Inherited
    _ancestor: Dict[str, float] = field(default_factory=dict)

    # NOTE: Local
    _local: Dict[str, float] = field(default_factory=dict)

    # NOTE: Inherited + Local
    @property
    def total(self) -> Dict[str, float]: return funcs.combine_dict(self._ancestor, self._local)
    @property
    def sum(self) -> float: return funcs.sum_dict(self.total)

    def _calculate_local_sediment(self, area: Dict[str, float], rainfall_amount: float, coefficients: Dict[str, config.RoadTypeData]) -> None:

        for surface_type, surface_area in area.items():
            road_type_data: config.RoadTypeData = coefficients[surface_type] # The reason I don't have any error handling here is because I check to make sure that all road types exist in data/roads.py with __vd_road_types()

            # Calculate sediment mass: Area * Rainfall * Erosion Coefficient
            # Assumes rainfall is in mm and area is in square meters
            sediment_mass = surface_area * (rainfall_amount / 1000) * road_type_data['erosion_rate']

            self._local[surface_type] = sediment_mass

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

@dataclass
class GraphNode:
    point: shapely.geometry.point.Point
    node_type: NodeType
    elevation: float

    # Information
    road: RoadInformation = field(default_factory=RoadInformation)
    runoff: RunoffInformation = field(default_factory=RunoffInformation)
    sediment: SedimentInformation = field(default_factory=SedimentInformation)
    pond: PondInformation | None = None

    # Node Relationships
    child: shapely.geometry.point.Point | None = None
    distance_to_child: float | None = None
    cost_to_connect_child: float | None = None
    volume_reaching_child: float | None = None
    sediment_reaching_child: float | None = None
    percent_reaching_child: float | None = None

class Graph:
    def __init__(self) -> None:
        self.__G : nx.DiGraph = nx.DiGraph()

    def print(self):
        for node, data in self.__G.nodes(data=True):
            print(f"Node {node}: {data.get('nodedata')}")

    # This function exists because when populating the graph,
    # it is not garunteed that the child node exists so
    # we add a provisional node.
    def conditionally_add_provisional_node(
        self,
        point: shapely.geometry.point.Point,
    ) -> None:

        from .data import elevation

        if not self.__G.has_node(point):
            terminal_node = GraphNode(point=point, node_type=NodeType.TERMINATION, elevation=elevation.sample_point(point))
            self.__G.add_node(point, nodedata=terminal_node)

    def add_node(
        self,
        node: GraphNode,
    ) -> None:
        if bool(node.child) != bool(node.distance_to_child):
            raise ValueError("child_node and distance_to_child must either both be None or non-None")

        self.__G.add_node(node.point, nodedata=node)

        if node.child is not None:
            self.conditionally_add_provisional_node(node.child)
            self.__G.add_edge(node.point, node.child, weight=node.distance_to_child)

    def add_nodes(
        self,
        nodes: List[GraphNode]
    ) -> None:
        for node in nodes: 
            self.add_node(node)
            if not nx.is_directed_acyclic_graph(self.__G):
                raise ValueError(f"Adding point {node.point} made the graph cycle.")

    def to_networkx_graph(self) -> nx.DiGraph:
        return self.__G

    def get_topological_order(self) -> List[shapely.geometry.point.Point]:
        return list(nx.topological_sort(self.__G))

    def prepare_graph(self, rainfall_event_size: float) -> None:
        from utils import config

        self.flowpath_travel_cost: float = config.get_flowpath_travel_cost()
        self.road_types: Dict[str, config.RoadTypeData] = config.get_road_types()
        self.rainfall_event_size = rainfall_event_size

        self.__G.clear_edges() # We're only going to add edges if runoff > cost

    def process_node(self, point: shapely.geometry.point.Point) -> None:
        nodedata = self.__G.nodes[point]['nodedata']

        if not isinstance(nodedata, GraphNode):
            raise ValueError("Node in processing list is somehow not in the graph, this should never happen!")

        nodedata.runoff._calculate_local_runoff(
            nodedata.road._local_area,
            self.rainfall_event_size,
            self.road_types
        )

        nodedata.sediment._calculate_local_sediment(
            nodedata.road._local_area,
            self.rainfall_event_size,
            self.road_types
        )

        match nodedata.node_type:
            case NodeType.POND:
                self.__process_pond_node(nodedata)
            case _:
               pass

        if nodedata.child is not None:
            self.__process_child_node(
                parent_node_data=nodedata,
                child_node_data=self.__G.nodes[nodedata.child]['nodedata']
            )

    def __process_pond_node(self, nodedata: GraphNode) -> None:
        # TODO: Get the bulk density of sediments to update used_capacity between rainfall events

        if not nodedata.pond:
            raise ValueError(f"Pond node {nodedata.point} does not have have a pond structure!") # This should never be the case
        if (funcs.sum_dict(nodedata.runoff._local) != 0) or (funcs.sum_dict(nodedata.sediment._local) != 0):
            raise ValueError(f"Expected [runoff/sediment]._local to be zero for pond node {nodedata.point}.") # This should also never be the case!

        # I don't like how I have this implemented but I don't know how else to do it right now...
        nodedata.pond._runoff_in = nodedata.runoff.sum
        nodedata.pond._sediment_in = nodedata.sediment.sum

        nodedata.runoff._local = funcs.scale_dict(nodedata.runoff._ancestor, nodedata.pond.runoff_percent_difference)
        nodedata.sediment._local = funcs.scale_dict(nodedata.sediment._ancestor, nodedata.pond.sediment_percent_difference)

    def __process_child_node(self, parent_node_data: GraphNode, child_node_data: GraphNode) -> None:
        if not parent_node_data.cost_to_connect_child:
            raise ValueError(f"{parent_node_data.node_type} {parent_node_data.point} is incomplete to compute child node (missing cost_to_connect_child)")

        parent_node_data.volume_reaching_child = max(0, parent_node_data.runoff.sum - parent_node_data.cost_to_connect_child)
        if parent_node_data.volume_reaching_child == 0:
            return
        else:
            self.__G.add_edge(parent_node_data.point, parent_node_data.child, weight=parent_node_data.distance_to_child)

        parent_node_data.percent_reaching_child = parent_node_data.volume_reaching_child / parent_node_data.runoff.sum
        parent_node_data.sediment_reaching_child = parent_node_data.sediment.sum * parent_node_data.percent_reaching_child 

        child_node_data.road._ancestor_indices = funcs.combine_dict_list(
            child_node_data.road._ancestor_indices,
            parent_node_data.road.indices
        )

        child_node_data.road._ancestor_length = funcs.combine_dict(
            child_node_data.road._ancestor_length,
            parent_node_data.road.length
        )

        child_node_data.road._ancestor_area = funcs.combine_dict(
            child_node_data.road._ancestor_area,
            parent_node_data.road.area
        )

        child_node_data.runoff._ancestor = funcs.combine_dict(
            child_node_data.runoff._ancestor,
            funcs.scale_dict(parent_node_data.runoff.total, parent_node_data.percent_reaching_child)
        )

        child_node_data.sediment._ancestor = funcs.combine_dict(
            child_node_data.sediment._ancestor,
            funcs.scale_dict(parent_node_data.sediment.total, parent_node_data.percent_reaching_child)
        )
