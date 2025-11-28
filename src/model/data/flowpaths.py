from typing import Tuple
import geopandas as gpd
from pathlib import Path

import shapely
from utils import config
from . import roads, elevation

path: Path = config.resolve_flowpaths_data_path()
__flowpath_travel_cost = config.get_flowpath_travel_cost()
_gdf = gpd.read_file(path)

def __vd_lines() -> None:
    invalid_indices =  [
        idx for idx, line in enumerate(_gdf.geometry) if not line.is_simple
    ]

    if invalid_indices: 
        raise ValueError(f"Self-intersecting flowpaths at indices: {invalid_indices}")

__vd_lines()



def trace_drainage_endpoint(reference_point: shapely.geometry.Point) -> Tuple[shapely.geometry.point.Point | None, float | None, float | None, int | None] :
    # Find flowpaths intersecting with the reference point
    intersecting_flowpaths = _gdf[_gdf.intersects(reference_point)]
    if len(intersecting_flowpaths) == 0:
        return None, None, None, None

    # Identify candidate downhill flowpaths
    candidate_flowpaths = []
    for idx, flowpath in intersecting_flowpaths.iterrows():
        # Extract line coordinates and create start/end points
        line_coordinates = list(flowpath.geometry.coords)
        start_point = shapely.geometry.Point(line_coordinates[0])
        end_point = shapely.geometry.Point(line_coordinates[-1])

        # Sample elevations at start and end points
        start_elevation = elevation.sample_point(start_point)
        end_elevation = elevation.sample_point(end_point)

        # Check for downhill flow conditions
        if (start_point.equals(reference_point)) and (start_elevation > end_elevation):
            candidate_flowpaths.append((flowpath, end_point, idx))
        elif (end_point.equals(reference_point)) and (end_elevation > start_elevation):
            candidate_flowpaths.append((flowpath, start_point, idx))

    # Validate number of candidate flowpaths
    if len(candidate_flowpaths) != 1:
        raise ValueError(f"Expected exactly one downhill flowpath, found {len(candidate_flowpaths)} at {reference_point}")

    # Extract the selected flowpath and its endpoint
    selected_flowpath, terminal_point, flowpath_index = candidate_flowpaths[0]

    # Create a small buffer around the terminal point
    terminal_point_buffer = terminal_point.buffer(0.3)

    # Check for road intersections
    road_intersections = roads._gdf[roads._gdf.intersects(terminal_point_buffer)]
    if not road_intersections.empty:
        # Return the DRAIN_IDX of the intersecting road segment
        return road_intersections.iloc[0]['DRAIN_IDX'], selected_flowpath.geometry.length, selected_flowpath.geometry.length * __flowpath_travel_cost, flowpath_index

    return terminal_point, selected_flowpath.geometry.length, selected_flowpath.geometry.length * __flowpath_travel_cost, flowpath_index
