from typing import List
import geopandas as gpd
from pathlib import Path

from model.graph import GraphNode, NodeType, PondInformation
from utils import config
from . import flowpaths, elevation

path: Path = config.resolve_ponds_data_path()
_gdf = gpd.read_file(path)

_gdf['ELEVATION'] = _gdf['geometry'].apply(lambda point: elevation.sample_point(point) )

def get_nodes() -> List[GraphNode]:
    nodes: List[GraphNode] = []
    for _, row in _gdf.iterrows():

        point = row.geometry
        node_type = NodeType.POND
        elevation = float(row['ELEVATION']) # Already is a float, this is just for the LSP
        node = GraphNode(
            point=point,
            node_type=node_type,
            elevation=elevation
        )

        max_capacity = row['MAX_CAP']
        if not isinstance(max_capacity, float | int):
            raise ValueError(f"Invalid pond MAX_CAP at {point}, expected float or int type, got {type(max_capacity)}")

        used_capacity = row['USED_CAP']
        if not isinstance(used_capacity, float | int):
            raise ValueError(f"Invalid pond USED_CAP at {point}, expected float or int type, got {type(used_capacity)}")

        if used_capacity > max_capacity:
            raise ValueError(f"Pond {point} has 'USED_CAP' > 'MAX_CAP'")

        node.pond = PondInformation(
            max_capacity=max_capacity,
            used_capacity=used_capacity
        )

        node.child, node.distance_to_child, node.cost_to_connect_child, node.index_of_flowpath_to_child = flowpaths.trace_drainage_endpoint(point)

        nodes.append(node)

    return nodes
