import json
from geopandas import gpd
import networkx as nx
from typing import List, Tuple, Dict, Any
from shapely.ops import linemerge
from shapely.geometry import MultiLineString, LineString

from model.data.roads import _gdf

# NOTE: Serialized JSON Output
"""

{
  "rainfall_events": [
    {
      "total_mm": <Float>,
      "nodes": [
        {
          // LOCATION & TYPE
          "point": "SRID=<int>;POINT(...)", // Format: EWKT
          "node_type": "<String>",          // e.g., "DRAIN", "TERMINATION", "POND"
          "elevation": <Float>,

          // ROAD INFORMATION
          // Note: All "wkt" fields contain lists of merged LineStrings in EWKT format.
          "road_information": {
            // 1. Geometries
            "wkt":          { "<SurfaceType>": ["<EWKT>"] }, // Derived from combined @property indices
            "local_wkt":    { "<SurfaceType>": ["<EWKT>"] }, // Derived from _local_indices
            "ancestor_wkt": { "<SurfaceType>": ["<EWKT>"] }, // Derived from _ancestor_indices

            // 2. Combined Statistics (from @property length/area)
            "length_m": { "<SurfaceType>": <Float> },
            "area_sqm": { "<SurfaceType>": <Float> },

            // 3. Split Statistics (Restored)
            "local_length_m":    { "<SurfaceType>": <Float> },
            "ancestor_length_m": { "<SurfaceType>": <Float> },
            "local_area_sqm":    { "<SurfaceType>": <Float> },
            "ancestor_area_sqm": { "<SurfaceType>": <Float> }
          },

          // HYDROLOGY: RUNOFF
          "runoff_information": {
            "local_m3":    { "<SurfaceType>": <Float> },
            "ancestor_m3": { "<SurfaceType>": <Float> },
            "total_m3":    { "<SurfaceType>": <Float> }, // Derived from @property total
            "sum_m3":      <Float>                       // Derived from @property sum (Scalar)
          },

          // HYDROLOGY: SEDIMENT
          "sediment_information": {
            "local_kg":    { "<SurfaceType>": <Float> },
            "ancestor_kg": { "<SurfaceType>": <Float> },
            "total_kg":    { "<SurfaceType>": <Float> }, // Derived from @property total
            "sum_kg":      <Float>                       // Derived from @property sum (Scalar)
          },

          // POND DATA (Nullable: null if node is not a POND)
          "pond_information": {
            "max_capacity": <Float>,
            "used_capacity": <Float>,
            "available_capacity": <Float>,
            "efficiency": <Float>,                       // 0.0 to 1.0

            "runoff_in": <Float>,
            "trapped_runoff": <Float>,
            "runoff_out": <Float>,
            "runoff_percent_difference": <Float>,

            "sediment_in": <Float>,
            "trapped_sediment": <Float>,
            "sediment_out": <Float>,
            "sediment_percent_difference": <Float>
          } | null,

          // CONNECTIVITY (Nullable: null if node has no child)
          "child_connection": {
            "child_point": "SRID=<int>;POINT(...)",
            "distance_to_child_m": <Float>,
            "cost_to_connect_child": <Float>,
            "volume_reaching_child_m3": <Float>,
            "sediment_reaching_child_kg": <Float> | null,
            "percent_reaching_child": <Float> | null
          } | null
        }
      ]
    }
  ]
}

"""

# --- DEFINE SUFFIXES ---
HYDROLOGY_SUFFIXES = {"local": "_m3", "ancestor": "_m3", "total": "_m3", "sum": "_m3"}
SEDIMENT_SUFFIXES = {"local": "_kg", "ancestor": "_kg", "total": "_kg", "sum": "_kg"}
ROAD_STAT_SUFFIXES = {"length": "_m", "area": "_sqm"}


# --- ABSTRACTION LAYERS ---

# TODO: Rewrite this class to handle not just roads but also polygons for coffee farms
class GeometryProcessor:
    def __init__(self):
        self.gdf: gpd.GeoDataFrame = _gdf

        if self.gdf.crs is None:
            raise ValueError("_gdf has not CRS")
        self.srid = self.gdf.crs.to_epsg()

    def to_ewkt(self, geom: Any) -> str:
        return f"SRID={self.srid};{getattr(geom, 'wkt', str(geom))}"

    def indices_to_ewkt_map(self, indices_map: Dict[str, List[int]]) -> Dict[str, List[str]]:
            result_map = {}

            if not indices_map:
                return result_map

            for surface_type, road_segment_indices in indices_map.items():
                # If the list of indices is empty, we return an empty list for this surface.
                # This is considered expected behavior, not an error.
                if not road_segment_indices:
                    result_map[surface_type] = []
                    continue

                try:
                    # 1. GDF Lookup
                    # This will raise a KeyError natively if indices are missing in standard Pandas usage
                    selected_rows = self.gdf.loc[road_segment_indices]

                    # Safety check: Verify we actually got geometries back
                    road_segment_geometries = selected_rows.geometry.tolist()

                    if not road_segment_geometries:
                        raise ValueError(f"Indices provided for '{surface_type}' but no geometries were extracted.")

                    # 2. Geometry Merge
                    # linemerge can fail if geometries are invalid, raising a TopologicalError or similar
                    merged_road_geometry = linemerge(road_segment_geometries)

                    # 3. Convert to EWKT List
                    ewkt_strings_list = []

                    if isinstance(merged_road_geometry, MultiLineString):
                        for geometry_component in merged_road_geometry.geoms:
                            ewkt_strings_list.append(self.to_ewkt(geometry_component))

                    elif isinstance(merged_road_geometry, LineString):
                        ewkt_strings_list.append(self.to_ewkt(merged_road_geometry))

                    else:
                        # Unexpected behavior: linemerge returned a Point, Polygon, or GeometryCollection
                        raise TypeError(
                            f"Unexpected geometry type '{type(merged_road_geometry)}' returned "
                            f"after merging road segments for surface '{surface_type}'."
                        )

                    result_map[surface_type] = ewkt_strings_list

                except KeyError as lookup_error:
                    raise KeyError(
                        f"Critical Error: One or more road indices in {road_segment_indices} "
                        f"for surface '{surface_type}' were not found in the global road GeoDataFrame."
                    ) from lookup_error

                except Exception as unexpected_error:
                    raise RuntimeError(
                        f"An unexpected error occurred while processing road geometries "
                        f"for surface '{surface_type}': {str(unexpected_error)}"
                    ) from unexpected_error

            return result_map


class AttributeExtractor:
    """
    Automatically extracts:
    1. Instance variables (via vars())
    2. Calculated Properties (via class inspection)
    """
    @staticmethod
    def extract(
        obj: Any,
        suffix_map: Dict[str, str] | None = None,
        exclude_patterns: List[str] | None = None
    ) -> Dict[str, Any]:

        if not obj: return {}

        # 1. Start with standard instance variables
        data = vars(obj).copy() if hasattr(obj, "__dict__") else {}

        # 2. Automatically find @properties
        cls = type(obj)
        for name in dir(cls):
            # Get the attribute from the Class (not the instance)
            attr = getattr(cls, name)
            # Check if it is a property
            if isinstance(attr, property):
                # Retrieve the value from the instance
                data[name] = getattr(obj, name)

        # 3. Clean, Filter, Suffix
        cleaned_result = {}
        for key, value in data.items():

            # Skip nulls
            if value is None: continue

            # Check Exclusions
            if exclude_patterns and any(p in key for p in exclude_patterns):
                continue

            # Remove underscores
            clean_key = key.lstrip('_')

            # Apply Suffixes
            if suffix_map:
                for keyword, suffix in suffix_map.items():
                    if keyword in clean_key and not clean_key.endswith(suffix):
                        clean_key += suffix
                        break

            cleaned_result[clean_key] = value

        return cleaned_result


# --- MAIN SERIALIZER ---

def serialize_rainfall_data(data: List[Tuple[float, nx.DiGraph]], output_filename: str):
    geo = GeometryProcessor()
    output = {"rainfall_events": []}

    for rainfall_total, graph in data:
        event = {"total_mm": rainfall_total, "nodes": []}
        for n_id in graph.nodes():
            node_data = graph.nodes[n_id].get("nodedata")
            if node_data:
                event["nodes"].append(_serialize_node(node_data, geo))
        output["rainfall_events"].append(event)

    with open(output_filename, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved {len(data)} events to {output_filename}")


def _serialize_node(node: Any, geo: GeometryProcessor) -> Dict[str, Any]:

    child_info = None
    if node.child is not None:
        child_info = {
            "child_point": geo.to_ewkt(node.child),
            "distance_to_child_m": node.distance_to_child,
            "cost_to_connect_child": node.cost_to_connect_child,
            "volume_reaching_child_m3": node.volume_reaching_child,
            "sediment_reaching_child_kg": node.sediment_reaching_child,
            "percent_reaching_child": node.percent_reaching_child
        }

    road_data = {}
    if node.road:
        # Explicitly handle Geometry
        road_data["wkt"] = geo.indices_to_ewkt_map(getattr(node.road, "indices", {}))
        road_data["local_wkt"] = geo.indices_to_ewkt_map(getattr(node.road, "_local_indices", {}))
        road_data["ancestor_wkt"] = geo.indices_to_ewkt_map(getattr(node.road, "_ancestor_indices", {}))

        stats = AttributeExtractor.extract(
            node.road,
            suffix_map=ROAD_STAT_SUFFIXES,
            exclude_patterns=["indices"] # Exclude because we processed them into WKTs above
        )
        road_data.update(stats)

    runoff_data = AttributeExtractor.extract(
        node.runoff,
        suffix_map=HYDROLOGY_SUFFIXES
    )

    sediment_data = AttributeExtractor.extract(
        node.sediment,
        suffix_map=SEDIMENT_SUFFIXES
    )

    pond_data = AttributeExtractor.extract(node.pond)

    return {
        "point": geo.to_ewkt(node.point),
        "node_type": node.node_type.name if hasattr(node.node_type, "name") else str(node.node_type),
        "elevation": node.elevation,
        "road_information": road_data,
        "runoff_information": runoff_data,
        "sediment_information": sediment_data,
        "pond_information": pond_data or None,
        "child_connection": child_info
    }
