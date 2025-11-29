from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple
import shapely.geometry
import networkx as nx
from copy import deepcopy

from utils import funcs

from .road import RoadInformation
from .runoff import RunoffInformation
from .sediment import SedimentInformation
from .pond import PondInformation
from .visualization import visualization_information

class NodeType(Enum):
    DRAIN = 1 # Anywhere you have runoff converging (i.e., road drains and converging flowpaths)
    POND = 2
    TERMINATING_FLOWPATH = 3

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
    terminal_in_base_graph: bool = False
    child: shapely.geometry.point.Point | None = None
    index_of_flowpath_to_child: int | None = None
    distance_to_child: float | None = None
    cost_to_connect_child: float | None = None
    volume_reaching_child: float | None = None
    sediment_reaching_child: float | None = None
    percent_reaching_child: float | None = None
    visualization: visualization_information | None = None

class Graph:
    def __init__(self) -> None:
        from model import data
        print("Generating Base Graph...")
        self.__G : nx.DiGraph = nx.DiGraph()
        self.add_nodes(data.ponds.get_nodes())
        self.add_nodes(data.drains.get_nodes())
        self._mark_terminal_nodes()

    def copy(self) -> 'Graph':
        return deepcopy(self)

    def print(self):
        for node, data in self.__G.nodes(data=True):
            print(f"Node {node}: {data.get('nodedata')}")

    def _mark_terminal_nodes(self) -> None:
        for node_point in self.__G.nodes:
            if self.__G.out_degree(node_point) == 0:
                nodedata = self.__G.nodes[node_point]['nodedata']
                if isinstance(nodedata, GraphNode):
                    nodedata.terminal_in_base_graph = True

    # This function exists because when populating the graph,
    # it is not garunteed that the child node exists so
    # we add a provisional node. The child should always exist
    # unless this was a terminating flowpath without a node
    # marked at the end.
    def conditionally_add_provisional_node(
        self,
        point: shapely.geometry.point.Point,
    ) -> None:

        from .data import elevation

        if not self.__G.has_node(point):
            unlabeled_node = GraphNode(point=point, node_type=NodeType.TERMINATING_FLOWPATH, elevation=elevation.sample_point(point))
            unlabeled_node.road.graph = self.__G
            unlabeled_node.road.point = point
            self.__G.add_node(point, nodedata=unlabeled_node)

    def add_node(
        self,
        node: GraphNode,
    ) -> None:
        if bool(node.child) != bool(node.distance_to_child):
            raise ValueError("child_node and distance_to_child must either both be None or non-None")

        node.road.graph = self.__G
        node.road.point = node.point
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
        self.bulk_density: float = config.get_bulk_density()
        self.rainfall_event_size = rainfall_event_size

        self.__G.clear_edges() # We're only going to add edges if runoff > cost

    def simulate_rainfall(self, rainfall_event_size: float) -> None:
        processing_order = self.get_topological_order()
        self.prepare_graph(rainfall_event_size)

        for node in processing_order:
            self.process_node(node)

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

        if not nodedata.pond:
            raise ValueError(f"Pond node {nodedata.point} does not have have a pond structure!") # This should never be the case
        if (funcs.sum_dict(nodedata.runoff._local) != 0) or (funcs.sum_dict(nodedata.sediment._local) != 0):
            raise ValueError(f"Expected [runoff/sediment]._local to be zero for pond node {nodedata.point}.") # This should also never be the case!

        # I don't like how I have this implemented but I don't know how else to do it right now...
        nodedata.pond._runoff_in = nodedata.runoff.sum
        nodedata.pond._sediment_in = nodedata.sediment.sum

        nodedata.runoff._local = funcs.scale_dict(nodedata.runoff._ancestor, nodedata.pond.runoff_percent_difference)
        nodedata.sediment._local = funcs.scale_dict(nodedata.sediment._ancestor, nodedata.pond.sediment_percent_difference)

    def get_pond_update_data(self) -> List[Tuple[shapely.geometry.point.Point, float]]:
        updates = []
        for node_point in self.__G.nodes:
            nodedata = self.__G.nodes[node_point]['nodedata']
            if isinstance(nodedata, GraphNode) and nodedata.node_type == NodeType.POND and nodedata.pond:

                current_capacity = nodedata.pond.used_capacity
                additional_capacity = 0.0

                # Calculate sediment accumulation if density is set and sediment entered
                if self.bulk_density > 0 and nodedata.pond._sediment_in is not None:
                     additional_capacity = nodedata.pond._sediment_in / self.bulk_density

                new_capacity = min(nodedata.pond.max_capacity, current_capacity + additional_capacity)
                updates.append((node_point, new_capacity))
        return updates

    def update_pond_capacities(self, updates: List[Tuple[shapely.geometry.point.Point, float]]) -> None:
        for point, used_capacity in updates:
             if self.__G.has_node(point):
                nodedata = self.__G.nodes[point]['nodedata']
                if isinstance(nodedata, GraphNode) and nodedata.pond:
                    nodedata.pond.used_capacity = used_capacity

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

        child_node_data.runoff._ancestor = funcs.combine_dict(
            child_node_data.runoff._ancestor,
            funcs.scale_dict(parent_node_data.runoff.total, parent_node_data.percent_reaching_child)
        )

        child_node_data.sediment._ancestor = funcs.combine_dict(
            child_node_data.sediment._ancestor,
            funcs.scale_dict(parent_node_data.sediment.total, parent_node_data.percent_reaching_child)
        )
