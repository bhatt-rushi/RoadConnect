import sqlite3
import networkx as nx
from typing import List, Tuple, Dict, Any, Set
import os

# NOTE: Serialized SQLite Schema
"""
The database (sqlite3) structure is designed to allow efficient querying of results
by rainfall event or by specific nodes.

Tables and Relationships:

events
  - id (INTEGER PRIMARY KEY)
  - total_mm (REAL)
  - is_summary (BOOLEAN)

nodes
  - id (INTEGER PRIMARY KEY)
  - event_id (INTEGER, FK -> events.id)
  - point (TEXT)
  - node_type (TEXT)
  - elevation (REAL)
  - terminal (BOOLEAN)

visualization
  - node_id (INTEGER PRIMARY KEY, FK -> nodes.id)
  - ancestor_ewkt (TEXT)
  - local_ewkt (TEXT)
  - descendant_ewkt (TEXT)

road_surfaces
  - id (INTEGER PRIMARY KEY)
  - node_id (INTEGER, FK -> nodes.id)
  - surface_type (TEXT)
  - length_m (REAL)
  - area_sqm (REAL)
  - local_length_m (REAL)
  - ancestor_length_m (REAL)
  - local_area_sqm (REAL)
  - ancestor_area_sqm (REAL)

runoff
  - node_id (INTEGER PRIMARY KEY, FK -> nodes.id)
  - sum_m3 (REAL)

runoff_surfaces
  - id (INTEGER PRIMARY KEY)
  - node_id (INTEGER, FK -> nodes.id)
  - surface_type (TEXT)
  - local_m3 (REAL)
  - ancestor_m3 (REAL)
  - total_m3 (REAL)

sediment
  - node_id (INTEGER PRIMARY KEY, FK -> nodes.id)
  - sum_kg (REAL)

sediment_surfaces
  - id (INTEGER PRIMARY KEY)
  - node_id (INTEGER, FK -> nodes.id)
  - surface_type (TEXT)
  - local_kg (REAL)
  - ancestor_kg (REAL)
  - total_kg (REAL)

ponds
  - node_id (INTEGER PRIMARY KEY, FK -> nodes.id)
  - max_capacity (REAL)
  - used_capacity (REAL)
  - available_capacity (REAL)
  - efficiency (REAL)
  - runoff_in (REAL)
  - trapped_runoff (REAL)
  - runoff_out (REAL)
  - runoff_percent_difference (REAL)
  - sediment_in (REAL)
  - trapped_sediment (REAL)
  - sediment_out (REAL)
  - sediment_percent_difference (REAL)

connections
  - node_id (INTEGER PRIMARY KEY, FK -> nodes.id)
  - child_point (TEXT)
  - distance_to_child_m (REAL)
  - cost_to_connect_child (REAL)
  - volume_reaching_child_m3 (REAL)
  - sediment_reaching_child_kg (REAL)
  - percent_reaching_child (REAL)
"""

# --- DEFINE SUFFIXES ---
HYDROLOGY_SUFFIXES = {"local": "_m3", "ancestor": "_m3", "total": "_m3", "sum": "_m3"}
SEDIMENT_SUFFIXES = {"local": "_kg", "ancestor": "_kg", "total": "_kg", "sum": "_kg"}
ROAD_STAT_SUFFIXES = {"length": "_m", "area": "_sqm"}


# --- ABSTRACTION LAYERS ---

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


# --- DATABASE MANAGER ---

class DatabaseManager:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._init_schema()

    def close(self):
        self.conn.commit()
        self.conn.close()

    def _init_schema(self):
        commands = [
            """CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                total_mm REAL NOT NULL,
                is_summary BOOLEAN DEFAULT 0
            )""",
            """CREATE TABLE IF NOT EXISTS nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER,
                point TEXT,
                node_type TEXT,
                elevation REAL,
                terminal BOOLEAN,
                FOREIGN KEY(event_id) REFERENCES events(id)
            )""",
            """CREATE TABLE IF NOT EXISTS visualization (
                node_id INTEGER PRIMARY KEY,
                ancestor_ewkt TEXT,
                local_ewkt TEXT,
                descendant_ewkt TEXT,
                FOREIGN KEY(node_id) REFERENCES nodes(id)
            )""",
            """CREATE TABLE IF NOT EXISTS road_surfaces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id INTEGER,
                surface_type TEXT,
                length_m REAL,
                area_sqm REAL,
                local_length_m REAL,
                ancestor_length_m REAL,
                local_area_sqm REAL,
                ancestor_area_sqm REAL,
                FOREIGN KEY(node_id) REFERENCES nodes(id)
            )""",
            """CREATE TABLE IF NOT EXISTS runoff (
                node_id INTEGER PRIMARY KEY,
                sum_m3 REAL,
                FOREIGN KEY(node_id) REFERENCES nodes(id)
            )""",
            """CREATE TABLE IF NOT EXISTS runoff_surfaces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id INTEGER,
                surface_type TEXT,
                local_m3 REAL,
                ancestor_m3 REAL,
                total_m3 REAL,
                FOREIGN KEY(node_id) REFERENCES nodes(id)
            )""",
            """CREATE TABLE IF NOT EXISTS sediment (
                node_id INTEGER PRIMARY KEY,
                sum_kg REAL,
                FOREIGN KEY(node_id) REFERENCES nodes(id)
            )""",
            """CREATE TABLE IF NOT EXISTS sediment_surfaces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id INTEGER,
                surface_type TEXT,
                local_kg REAL,
                ancestor_kg REAL,
                total_kg REAL,
                FOREIGN KEY(node_id) REFERENCES nodes(id)
            )""",
            """CREATE TABLE IF NOT EXISTS ponds (
                node_id INTEGER PRIMARY KEY,
                max_capacity REAL,
                used_capacity REAL,
                available_capacity REAL,
                efficiency REAL,
                runoff_in REAL,
                trapped_runoff REAL,
                runoff_out REAL,
                runoff_percent_difference REAL,
                sediment_in REAL,
                trapped_sediment REAL,
                sediment_out REAL,
                sediment_percent_difference REAL,
                FOREIGN KEY(node_id) REFERENCES nodes(id)
            )""",
            """CREATE TABLE IF NOT EXISTS connections (
                node_id INTEGER PRIMARY KEY,
                child_point TEXT,
                distance_to_child_m REAL,
                cost_to_connect_child REAL,
                volume_reaching_child_m3 REAL,
                sediment_reaching_child_kg REAL,
                percent_reaching_child REAL,
                FOREIGN KEY(node_id) REFERENCES nodes(id)
            )""",
            # Indices for performance
            "CREATE INDEX IF NOT EXISTS idx_nodes_event_id ON nodes(event_id)",
            "CREATE INDEX IF NOT EXISTS idx_road_surfaces_node_id ON road_surfaces(node_id)",
            "CREATE INDEX IF NOT EXISTS idx_runoff_surfaces_node_id ON runoff_surfaces(node_id)",
            "CREATE INDEX IF NOT EXISTS idx_sediment_surfaces_node_id ON sediment_surfaces(node_id)"
        ]
        for cmd in commands:
            self.cursor.execute(cmd)
        self.conn.commit()

    def insert_event(self, total_mm: float, is_summary: bool = False) -> int:
        self.cursor.execute(
            "INSERT INTO events (total_mm, is_summary) VALUES (?, ?)",
            (total_mm, is_summary)
        )
        return self.cursor.lastrowid

    def insert_node(self, event_id: int, node_data: Any) -> int:
        point = str(node_data.point)
        node_type = node_data.node_type.name if hasattr(node_data.node_type, "name") else str(node_data.node_type)

        self.cursor.execute(
            """INSERT INTO nodes (event_id, point, node_type, elevation, terminal)
               VALUES (?, ?, ?, ?, ?)""",
            (event_id, point, node_type, node_data.elevation, node_data.terminal_in_base_graph)
        )
        node_id = self.cursor.lastrowid

        # Insert related data
        self._insert_visualization(node_id, node_data.visualization)
        self._insert_road_info(node_id, node_data.road)
        self._insert_runoff_info(node_id, node_data.runoff)
        self._insert_sediment_info(node_id, node_data.sediment)
        self._insert_pond_info(node_id, node_data.pond)
        self._insert_connection_info(node_id, node_data) # node_data has child info

        return node_id

    def _insert_visualization(self, node_id: int, viz: Any):
        if not viz: return
        self.cursor.execute(
            """INSERT INTO visualization (node_id, ancestor_ewkt, local_ewkt, descendant_ewkt)
               VALUES (?, ?, ?, ?)""",
            (node_id, viz.ancestor_ewkt, viz.local_ewkt, viz.descendant_ewkt)
        )

    def _insert_road_info(self, node_id: int, road: Any):
        if not road: return
        stats = AttributeExtractor.extract(
            road,
            suffix_map=ROAD_STAT_SUFFIXES,
            exclude_patterns=["indices", "graph", "point"]
        )

        # Transpose stats to surface-based rows
        surfaces = self._transpose_stats(stats)

        for surface, data in surfaces.items():
            self.cursor.execute(
                """INSERT INTO road_surfaces
                   (node_id, surface_type, length_m, area_sqm, local_length_m, ancestor_length_m, local_area_sqm, ancestor_area_sqm)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    node_id, surface,
                    data.get('length_m'), data.get('area_sqm'),
                    data.get('local_length_m'), data.get('ancestor_length_m'),
                    data.get('local_area_sqm'), data.get('ancestor_area_sqm')
                )
            )

    def _insert_runoff_info(self, node_id: int, runoff: Any):
        if not runoff: return
        data = AttributeExtractor.extract(runoff, suffix_map=HYDROLOGY_SUFFIXES)

        # Scalar
        if 'sum_m3' in data:
            self.cursor.execute(
                "INSERT INTO runoff (node_id, sum_m3) VALUES (?, ?)",
                (node_id, data['sum_m3'])
            )

        # Surfaces
        surfaces = self._transpose_stats(data, exclude_keys={'sum_m3'})
        for surface, s_data in surfaces.items():
             self.cursor.execute(
                """INSERT INTO runoff_surfaces (node_id, surface_type, local_m3, ancestor_m3, total_m3)
                   VALUES (?, ?, ?, ?, ?)""",
                (node_id, surface, s_data.get('local_m3'), s_data.get('ancestor_m3'), s_data.get('total_m3'))
            )

    def _insert_sediment_info(self, node_id: int, sediment: Any):
        if not sediment: return
        data = AttributeExtractor.extract(sediment, suffix_map=SEDIMENT_SUFFIXES)

        # Scalar
        if 'sum_kg' in data:
            self.cursor.execute(
                "INSERT INTO sediment (node_id, sum_kg) VALUES (?, ?)",
                (node_id, data['sum_kg'])
            )

        # Surfaces
        surfaces = self._transpose_stats(data, exclude_keys={'sum_kg'})
        for surface, s_data in surfaces.items():
             self.cursor.execute(
                """INSERT INTO sediment_surfaces (node_id, surface_type, local_kg, ancestor_kg, total_kg)
                   VALUES (?, ?, ?, ?, ?)""",
                (node_id, surface, s_data.get('local_kg'), s_data.get('ancestor_kg'), s_data.get('total_kg'))
            )

    def _insert_pond_info(self, node_id: int, pond: Any):
        if not pond: return
        data = AttributeExtractor.extract(pond)

        self.cursor.execute(
            """INSERT INTO ponds (
                node_id, max_capacity, used_capacity, available_capacity, efficiency,
                runoff_in, trapped_runoff, runoff_out, runoff_percent_difference,
                sediment_in, trapped_sediment, sediment_out, sediment_percent_difference
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                node_id, data.get('max_capacity'), data.get('used_capacity'), data.get('available_capacity'), data.get('efficiency'),
                data.get('runoff_in'), data.get('trapped_runoff'), data.get('runoff_out'), data.get('runoff_percent_difference'),
                data.get('sediment_in'), data.get('trapped_sediment'), data.get('sediment_out'), data.get('sediment_percent_difference')
            )
        )

    def _insert_connection_info(self, node_id: int, node: Any):
        if node.child is None: return

        child_point = str(node.child)

        self.cursor.execute(
            """INSERT INTO connections (
                node_id, child_point, distance_to_child_m, cost_to_connect_child,
                volume_reaching_child_m3, sediment_reaching_child_kg, percent_reaching_child
            ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                node_id, child_point, node.distance_to_child, node.cost_to_connect_child,
                node.volume_reaching_child, node.sediment_reaching_child, node.percent_reaching_child
            )
        )

    def _transpose_stats(self, flat_data: Dict[str, Any], exclude_keys: Set[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Converts { 'stat_name': { 'surface': val, ... }, ... }
        into { 'surface': { 'stat_name': val, ... }, ... }
        """
        surfaces = {}
        for stat_name, val_map in flat_data.items():
            if exclude_keys and stat_name in exclude_keys: continue

            if isinstance(val_map, dict):
                for surface, val in val_map.items():
                    if surface not in surfaces:
                        surfaces[surface] = {}
                    surfaces[surface][stat_name] = val
            # Handle non-dict values if any (though normally they are scalars handled separately)
        return surfaces


# --- MAIN SERIALIZER ---

def serialize_rainfall_data(data: List[Tuple[float, nx.DiGraph]], output_filename: str, summary_graph: nx.DiGraph | None = None):
    """
    Serializes rainfall data and summary graph to a SQLite database.
    """
    if os.path.exists(output_filename):
        os.remove(output_filename)

    db = DatabaseManager(output_filename)

    try:
        # Save Events
        for rainfall_total, graph in data:
            event_id = db.insert_event(rainfall_total, is_summary=False)
            for n_id in graph.nodes():
                node_data = graph.nodes[n_id].get("nodedata")
                if node_data:
                    db.insert_node(event_id, node_data)

        # Save Summary
        if summary_graph:
            summary_id = db.insert_event(-1.0, is_summary=True) # -1.0 as placeholder for summary
            for n_id in summary_graph.nodes():
                node_data = summary_graph.nodes[n_id].get("nodedata")
                if node_data:
                    db.insert_node(summary_id, node_data)

        print(f"Saved {len(data)} events (and summary={bool(summary_graph)}) to {output_filename}")

    finally:
        db.close()
