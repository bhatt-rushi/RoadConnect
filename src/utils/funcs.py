from typing import Dict, List, Set
import shapely.ops
import geopandas as gpd

def generate_wkt(geometries: List[shapely.geometry.base.BaseGeometry], srid: int) -> str:
    if not geometries:
        return ""

    # Filter valid geometries
    valid_geoms = [g for g in geometries if g is not None and not g.is_empty]

    if not valid_geoms:
        return ""

    # union
    merged = shapely.ops.unary_union(valid_geoms)

    # simplify (remove collinear points)
    simplified = merged.simplify(0, preserve_topology=True)

    return f"SRID={srid};{simplified.wkt}"

def combine_dict(*dicts: Dict[str, float]) -> Dict[str, float]:
    from collections import Counter
    # Sum all Counters, starting with an empty Counter to handle the first addition
    return dict(sum((Counter(d) for d in dicts), Counter()))

def combine_dict_list(*dicts: Dict[str, List[int]]) -> Dict[str, List[int]]:
    return {
        k: [x for d in dicts for x in d.get(k, [])]
        for k in set().union(*dicts)
    }

def combine_dict_set(*dicts: Dict[str, Set[int]]) -> Dict[str, Set[int]]:
    return {
        k: set().union(*(d.get(k, set()) for d in dicts))
        for k in set().union(*dicts)
    }

def scale_dict(input_dict: Dict[str, float], scaling_factor: float) -> Dict[str, float]:
    return {key: value * scaling_factor for key, value in input_dict.items()}

def sum_dict(input_dict: Dict[str, float]) -> float:
    return sum(input_dict.values())


def percent_difference(new: float, orig: float) -> float:
    if orig == 0.0:
        return 0
    return (new - orig) / orig
