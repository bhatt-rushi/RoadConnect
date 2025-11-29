from typing import List, Tuple
import networkx as nx
from copy import deepcopy
from model.graph import GraphNode
from model.runoff import RunoffInformation
from model.sediment import SedimentInformation
from utils import funcs

def create_summary_graph(results: List[Tuple[float, nx.DiGraph]]) -> nx.DiGraph:

    summary_graph = nx.DiGraph()

    for _, graph in results:
        # Add edges (union logic is handled by NetworkX: adding existing edge updates attributes)
        # We assume static edge attributes (distance) are constant.
        summary_graph.add_edges_from(graph.edges(data=True))

        for node, attrs in graph.nodes(data=True):
            nodedata: GraphNode = attrs['nodedata']

            if node not in summary_graph or 'nodedata' not in summary_graph.nodes[node]:
                # Initialize node in summary graph
                # We deepcopy to avoid modifying the original result graphs
                # and to start with a clean slate for dynamic attributes
                new_nodedata = deepcopy(nodedata)

                # Reset dynamic accumulators to zero/empty
                new_nodedata.runoff = RunoffInformation()
                new_nodedata.sediment = SedimentInformation()

                # Reset child flow metrics that we will accumulate
                new_nodedata.volume_reaching_child = 0.0
                new_nodedata.sediment_reaching_child = 0.0
                # Percent doesn't make sense to sum directly, so we'll derive it later or set to None
                new_nodedata.percent_reaching_child = None

                # If the node was added by add_edges_from, it exists but has no data.
                # We update it (or add it if it didn't exist).
                summary_graph.add_node(node, nodedata=new_nodedata)

            # Accumulate values
            summary_node_data: GraphNode = summary_graph.nodes[node]['nodedata']

            # Accumulate Runoff
            summary_node_data.runoff._ancestor = funcs.combine_dict(
                summary_node_data.runoff._ancestor, nodedata.runoff._ancestor
            )
            summary_node_data.runoff._local = funcs.combine_dict(
                summary_node_data.runoff._local, nodedata.runoff._local
            )

            # Accumulate Sediment
            summary_node_data.sediment._ancestor = funcs.combine_dict(
                summary_node_data.sediment._ancestor, nodedata.sediment._ancestor
            )
            summary_node_data.sediment._local = funcs.combine_dict(
                summary_node_data.sediment._local, nodedata.sediment._local
            )

            # Accumulate flows to child (if applicable)
            if nodedata.volume_reaching_child is not None:
                summary_node_data.volume_reaching_child = (summary_node_data.volume_reaching_child or 0) + nodedata.volume_reaching_child

            if nodedata.sediment_reaching_child is not None:
                summary_node_data.sediment_reaching_child = (summary_node_data.sediment_reaching_child or 0) + nodedata.sediment_reaching_child

    # Post-processing for derived metrics
    for node in summary_graph.nodes:
        nodedata: GraphNode = summary_graph.nodes[node]['nodedata']
        if nodedata.volume_reaching_child is not None and nodedata.runoff.sum > 0:
             nodedata.percent_reaching_child = nodedata.volume_reaching_child / nodedata.runoff.sum

    return summary_graph
