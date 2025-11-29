import os
import json
from pathlib import Path
from typing import List, Dict, TypedDict

CONFIG_PATH = os.path.join(
        os.getcwd(),
        'config',
        'config.json'
    )

def get_rainfall_values() -> List[float]:
    try:
        with open(CONFIG_PATH, 'r') as config_file:
            config_data = json.load(config_file)

            rainfall_values = config_data['rainfall_values']

            try:
                validated_values = [float(value) for value in rainfall_values]
                return validated_values
            except (TypeError, ValueError):
                raise ValueError("rainfall_values must be a list of numbers that can be converted to float")

    except KeyError:
        raise KeyError("'rainfall_values' not found in the configuration file")

def get_bulk_density() -> float:
    try:
        with open(CONFIG_PATH, 'r') as config_file:
            config_data = json.load(config_file)

            # Extract bulk density
            bulk_density = config_data['bulk_density']

            # Validate type (must be int or float)
            if not isinstance(bulk_density, (int, float)):
                raise ValueError("bulk_density must be a number (int or float)")

            # Convert to float
            bulk_density = float(bulk_density)

            # Validate non-negative
            if bulk_density <= 0:
                raise ValueError("bulk_density must be a positive number")

            return bulk_density

    except KeyError:
        raise KeyError("'bulk_density' not found in the configuration file")

def get_flowpath_travel_cost() -> float:
    try:
        with open(CONFIG_PATH, 'r') as config_file:
            config_data = json.load(config_file)

            # Extract travel cost
            travel_cost = config_data['travel_cost']

            # Validate type (must be int or float)
            if not isinstance(travel_cost, (int, float)):
                raise ValueError("travel_cost must be a number (int or float)")

            # Convert to float
            travel_cost = float(travel_cost)

            # Validate non-negative
            if travel_cost < 0:
                raise ValueError("travel_cost must be zero or a positive number")

            return travel_cost

    except KeyError:
        raise KeyError("'travel_cost' not found in the configuration file")

class RoadTypeData(TypedDict):
    runoff_coefficient: float
    erosion_rate: float

def get_road_types() -> Dict[str, RoadTypeData]:

    try:
        with open(CONFIG_PATH, 'r') as config_file:
            config_data = json.load(config_file)

            # Extract road types
            road_types = config_data['road_types']

            # Validate overall structure
            if not isinstance(road_types, dict):
                raise ValueError("road_types must be a dictionary")

            # Validate each road type
            validated_road_types: Dict[str, RoadTypeData] = {}
            for road_type, type_data in road_types.items():
                # Validate road type is a string
                if not isinstance(road_type, str):
                    raise ValueError(f"Road type key must be a string, got {type(road_type)}")

                # Validate each road type has the correct structure
                if not isinstance(type_data, dict):
                    raise ValueError(f"Road type data for '{road_type}' must be a dictionary")

                # Validate required keys and their types
                if set(type_data.keys()) != {'runoff_coefficient', 'erosion_rate'}:
                    raise ValueError(f"Road type '{road_type}' must have exactly 'runoff_coefficient' and 'erosion_rate' keys")

                # Validate and extract runoff coefficient
                runoff_coefficient = type_data['runoff_coefficient']
                if not isinstance(runoff_coefficient, (int, float)):
                    raise ValueError(f"runoff_coefficient for '{road_type}' must be a number")

                # Validate and extract erosion rate
                erosion_rate = type_data['erosion_rate']
                if not isinstance(erosion_rate, (int, float)):
                    raise ValueError(f"erosion_rate for '{road_type}' must be a number")

                # Store validated and converted data
                validated_road_types[road_type] = {
                    'runoff_coefficient': float(runoff_coefficient),
                    'erosion_rate': float(erosion_rate)
                }

            return validated_road_types

    except KeyError:
        raise KeyError("'road_types' not found in the configuration file")

def __resolve_data_path(key: str) -> Path:
    try:
        with open(CONFIG_PATH, 'r') as config_file:
            config_data = json.load(config_file)

            # Extract path string
            path_str = config_data['datapaths'][key]

            # Validate type
            if not isinstance(path_str, str):
                raise ValueError(f"{key} datapath must be a string")

            # Convert to Path
            path = Path(path_str)

            # Validate existence
            if path.is_file():
                return path
            else:
                raise FileNotFoundError(f"{key} file:{path} does not exist!")

    except KeyError:
        raise KeyError(f"'datapaths' -> {key} not found in the configuration file")

def resolve_roads_data_path() -> Path:
    return __resolve_data_path('roads')

def resolve_flowpaths_data_path() -> Path:
    return __resolve_data_path('flowpaths')

def resolve_drains_data_path() -> Path:
    return __resolve_data_path('drains')

def resolve_ponds_data_path() -> Path:
    return __resolve_data_path('ponds')

def resolve_elevation_data_path() -> Path:
    return __resolve_data_path('elevation')

def validate_crs() -> None:
    import geopandas as gpd
    import rasterio
    from pyproj import CRS

    # Reference CRS from drains
    drains_path = resolve_drains_data_path()
    try:
        drains_gdf = gpd.read_file(drains_path)
    except Exception as e:
         raise ValueError(f"Failed to read drains file: {e}")

    target_crs = drains_gdf.crs

    if target_crs is None:
         raise ValueError(f"Drains file {drains_path} has no CRS defined.")

    # Check other vector files
    vector_paths = {
        'roads': resolve_roads_data_path(),
        'flowpaths': resolve_flowpaths_data_path(),
        'ponds': resolve_ponds_data_path()
    }

    for name, path in vector_paths.items():
        try:
            gdf = gpd.read_file(path)
        except Exception as e:
             raise ValueError(f"Failed to read {name} file: {e}")

        if gdf.crs != target_crs:
             raise ValueError(f"{name} file {path} CRS ({gdf.crs}) does not match drains CRS ({target_crs})")

    # Check raster file
    elevation_path = resolve_elevation_data_path()
    try:
        with rasterio.open(elevation_path) as src:
            if src.crs is None:
                 raise ValueError(f"Elevation file {elevation_path} has no CRS defined.")

            # Convert rasterio CRS to pyproj CRS for comparison
            src_crs_pyproj = CRS(src.crs)

            if src_crs_pyproj != target_crs:
                 raise ValueError(f"Elevation file {elevation_path} CRS ({src_crs_pyproj}) does not match drains CRS ({target_crs})")
    except Exception as e:
        raise ValueError(f"Failed to read elevation file or validate CRS: {e}")
