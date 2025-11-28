from typing import List
import geopandas as gpd
from pathlib import Path

from model.graph import GraphNode, NodeType
from utils import config
from . import roads, flowpaths, elevation

path: Path = config.resolve_drains_data_path()
_gdf = gpd.read_file(path)

_gdf['ELEVATION'] = _gdf['geometry'].apply(lambda point: elevation.sample_point(point) )

def get_nodes() -> List[GraphNode]:
    nodes: List[GraphNode] = []
    for _, row in _gdf.iterrows():

        point = row.geometry
        node_type = NodeType.DRAIN
        elevation = float(row['ELEVATION']) # Already is a float, this is just for the LSP
        node = GraphNode(
            point=point,
            node_type=node_type,
            elevation=elevation
        )

        filtered_roads = roads._gdf[roads._gdf['DRAIN_IDX'] == point]
        node.road._local_indices = filtered_roads.groupby('TYPE')['index'].apply(list).to_dict()
        node.road._local_length = filtered_roads.groupby('TYPE')['LENGTH'].sum().to_dict()
        node.road._local_area = filtered_roads.groupby('TYPE')['AREA'].sum().to_dict()

        node.child, node.distance_to_child, node.cost_to_connect_child, node.index_of_flowpath_to_child = flowpaths.trace_drainage_endpoint(point)

        nodes.append(node)

    return nodes
