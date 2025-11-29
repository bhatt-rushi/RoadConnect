import geopandas as gpd
from pathlib import Path
from typing import Dict

from utils import config
from . import drains

path: Path = config.resolve_roads_data_path()
_gdf = gpd.read_file(path)

# Data Validation Functions
def __vd_index() -> None:
    if 'index' not in _gdf.columns:
        _gdf['index'] = range(len(_gdf))

    _gdf.set_index('index', inplace=True, drop=True, verify_integrity=True)

def __vd_road_types() -> None:
    unknown_types = set(_gdf['TYPE']) - set(config.get_road_types())
    if unknown_types:
        raise ValueError(f"Unknown road types in shapefile: {unknown_types}")

def __vd_length_and_area() -> None:
    if (zero_indices := [idx for idx, (length, area) in enumerate(zip(_gdf['LENGTH'], _gdf['AREA'])) if length <= 0 or area <= 0]):
            raise ValueError(f"Zero or negative LENGTH/AREA at road indices: {zero_indices}")

__vd_index()
__vd_road_types()
__vd_length_and_area()
# TODO: Add slope to attribute for erosion

# Export fast lookup maps
length_map: Dict[int, float] = dict(zip(_gdf.index, _gdf['LENGTH']))
area_map: Dict[int, float] = dict(zip(_gdf.index, _gdf['AREA']))

# Pre-Processing Functions

def ___pp_calculate_drain_connectivity() -> None:
    _gdf['INCL_DRAIN'], _gdf['DRAIN_IDX'] = False, None

    # First, mark drain-intersecting road segments
    for drain in drains._gdf.geometry:
        intersecting_roads = _gdf[_gdf.geometry.apply(lambda road: drain.distance(road) <= 1e-9)]

        if len(intersecting_roads) == 0:
            raise ValueError(f"Drain point {drain} does not intersect any road.")

        # Mark the first intersecting road segment
        idx = intersecting_roads.index[0]
        _gdf.at[idx, 'INCL_DRAIN'] = True
        _gdf.at[idx, 'DRAIN_IDX'] = drain

    # Process road segments
    unroutable_segments = []
    for idx, road_segment in _gdf[~_gdf['INCL_DRAIN']].iterrows():
        current_segment = road_segment
        visited_segments = set()

        while not current_segment['INCL_DRAIN']:
            # Find touching segments excluding already visited ones
            touching_segments = _gdf[
                _gdf.touches(current_segment.geometry) &
                (~_gdf.index.isin(visited_segments))
            ]

            # If no touching segments, mark as unroutable
            if touching_segments.empty:
                unroutable_segments.append(current_segment)
                break

            # Check for drain-touching segments
            drain_connected_segments = touching_segments[touching_segments['INCL_DRAIN']]

            if not drain_connected_segments.empty:
                # Assign the drain of the first drain-touching segment
                _gdf.at[idx, 'DRAIN_IDX'] = drain_connected_segments.iloc[0]['DRAIN_IDX']
                break

            # Progress to lowest elevation segment to continue tracing
            current_segment = touching_segments.loc[touching_segments['ELEVATION'].idxmin()]
            visited_segments.add(current_segment.name)

    _gdf.to_file('user_data/roads_w_connectivity.shp' )
    # Log unroutable segments if any exist
    if unroutable_segments:
        unroutable_length = sum(seg.geometry.length for seg in unroutable_segments)
        print(f"\n\nTotal unroutable segment length: {unroutable_length}")
        print(f"\nNumber of unroutable segments: {len(unroutable_segments)}")

___pp_calculate_drain_connectivity()
