r"""°°°
Class Structure
°°°"""
#|%%--%%| <o7S3MGfNLq|d1C9LohqaR>
import shapely
import pandas as pd
import geopandas as gpd
import rasterio
import json
import networkx as nx
from tqdm import tqdm
from copy import deepcopy
import numpy as np

class Model:
    def __init__(self, roadpath, roadtypepath, drainpath, pondpath, flowpathpath, elevationpath, configpath):
        self.rainfall_data = json.load(open(configpath))['rainfall_values']
        self.graph = Graph(model=self)
        self.data = Data(model=self, roadpath=roadpath, roadtypepath=roadtypepath, drainpath=drainpath, pondpath=pondpath, flowpathpath=flowpathpath, elevationpath=elevationpath)

    def run(self):
        print("BUILDING GRAPH...")
        self.graph.build_graph()

        # Get Resulting Graphs
        self.result_graphs = []
        for rainfall_event in tqdm(self.rainfall_data, total=len(self.rainfall_data)):
            g = deepcopy(self.graph.graph)

            for k, v in g.items():

                # Calculate Direct Runoff
                v['Directly_Connected_Segments']['Runoff'] = {key: value * (rainfall_event/1000) * self.data.roads.types[key]['runoff_coefficient'] for key, value in v['Directly_Connected_Segments']['Area'].items()}

                # Calculate Direct Sediment
                v['Directly_Connected_Segments']['Sediment'] = {key: (rainfall_event/1000) * self.data.roads.types[key]['erosion_rate'] * v['Directly_Connected_Segments']['Area'][key] for key, value in v['Directly_Connected_Segments']['Runoff'].items() if value > 0}

                # Calculate Direct Road Indicies
                for key, value in v['Directly_Connected_Segments']['Indices'].items():
                    v['All_Connected_Segments']['Indices'][key] = v['All_Connected_Segments']['Indices'].get(key, []) + value

                # Calculate Direct Road Length
                for key, value in v['Directly_Connected_Segments']['Length'].items():
                    v['All_Connected_Segments']['Length'][key] = v['All_Connected_Segments']['Length'].get(key, 0) + value

                # Calculate Direct Road Area
                for key, value in v['Directly_Connected_Segments']['Area'].items():
                    v['All_Connected_Segments']['Area'][key] = v['All_Connected_Segments']['Area'].get(key, 0) + value

                # Add Direct Runoff To All Runoff
                for key, value in v['Directly_Connected_Segments']['Runoff'].items():
                    v['All_Connected_Segments']['Runoff'][key] = v['All_Connected_Segments']['Runoff'].get(key, 0) + value

                # Add Direct Sediment To All Sediment
                for key, value in v['Directly_Connected_Segments']['Sediment'].items():
                    v['All_Connected_Segments']['Sediment'][key] = v['All_Connected_Segments']['Sediment'].get(key, 0) + value

                # Calculate Totals For Runoff & Sediment
                v['Runoff_Total'] = sum( v['All_Connected_Segments']['Runoff'].values() )
                v['Sediment_Total'] = sum( v['All_Connected_Segments']['Sediment'].values() )

                # Pond Traps Sediment
                if v['Type'] == 'P' and v['Runoff_Total'] > 0:

                    # TODO: Pond consumes runoff = pond volume
                    trapping_efficiency = float(np.clip(
                        -22 + ( (119 * ((v['Pond_Max_Capacity'] - v['Pond_Used_Capacity']) / v['Runoff_Total'])) / (0.012 + 1.02 * ((v['Pond_Max_Capacity'] - v['Pond_Used_Capacity']) / v['Runoff_Total']))),
                        0, # Min Eff
                        100 # Max Eff
                    ) / 100) # Turn to percent

                    v['Pond_Efficiency'] = trapping_efficiency
                    v['Sediment_Trapped'] = v['Sediment_Total'] * trapping_efficiency
                    v['Sediment_Total'] *= (1 - trapping_efficiency)

                    for key, value in v['All_Connected_Segments']['Sediment'].items():
                        v['All_Connected_Segments']['Sediment'][key] = value * trapping_efficiency

                if v['Child_Node'] is not None:
                    cost_to_child = v['Distance_To_Child'] * self.data.flowpaths.travel_cost
                    v['Cost_Required_To_Connect_Child'] = cost_to_child

                    volume_reaching_child = float( np.maximum(v['Runoff_Total'] - cost_to_child, 0))

                    if volume_reaching_child > 0:

                        delivery_ratio = volume_reaching_child / v['Runoff_Total']

                        g[v['Child_Node']]['Parent_Nodes'].append(k)
                        g[v['Child_Node']]['Ancestor_Nodes'] += v['Ancestor_Nodes'] + [k]

                        # Pass All Connected Road Indices to Child
                        for key, value in v['All_Connected_Segments']['Indices'].items():
                            g[v['Child_Node']]['All_Connected_Segments']['Indices'][key] = g[v['Child_Node']]['All_Connected_Segments']['Indices'].get(key, []) + value

                        # Pass All Connected Road Lengths to Child
                        for key, value in v['All_Connected_Segments']['Length'].items():
                            g[v['Child_Node']]['All_Connected_Segments']['Length'][key] = g[v['Child_Node']]['All_Connected_Segments']['Length'].get(key, 0) + value

                        # Pass All Connected Road Areas to Child
                        for key, value in v['All_Connected_Segments']['Area'].items():
                            g[v['Child_Node']]['All_Connected_Segments']['Area'][key] = g[v['Child_Node']]['All_Connected_Segments']['Area'].get(key, 0) + value

                        # Pass All Connected Road Runoff to Child, Applying V2B
                        for key, value in v['All_Connected_Segments']['Runoff'].items():
                            g[v['Child_Node']]['All_Connected_Segments']['Runoff'][key] = g[v['Child_Node']]['All_Connected_Segments']['Runoff'].get(key, 0) + ( value * delivery_ratio)

                        # Pass All Connected Road Sediment to Child, Applying V2B
                        for key, value in v['All_Connected_Segments']['Sediment'].items():
                            g[v['Child_Node']]['All_Connected_Segments']['Sediment'][key] = g[v['Child_Node']]['All_Connected_Segments']['Sediment'].get(key, 0) + ( value * delivery_ratio)

            # Append the result to result_graphs
            self.result_graphs.append( (rainfall_event, g) )

class GraphNode:
    def __init__(
            self,
            point : shapely.geometry.Point,
            node_type : str,
            elevation : float,
    ):
        self.index = point
        self.node = {
            'Type': node_type,  # Pond or Drain
            'Elevation': elevation,

            'Directly_Connected_Segments': {
                'Indices': {},  # Map organizing indices by ROAD_TYPE
                'Length': {},   # Total length per ROAD_TYPE
                'Area': {},     # Total area per ROAD_TYPE
                'Runoff': {},   # Total runoff from each ROAD_TYPE
                'Sediment': {}, # Total sediment from each ROAD_TYPE
            },

            'All_Connected_Segments': {
                'Indices': {},
                'Length': {},
                'Area': {},
                'Runoff': {},
                'Sediment': {},
            },

            'Runoff_Total': None,
            'Sediment_Total': None,

            'Pond_Max_Capacity': None,
            'Pond_Used_Capacity': None,
            'Pond_Efficiency': None,
            'Sediment_Trapped': None,

            'Parent_Nodes': [],
            'Child_Node': None,
            'Distance_To_Child': None,
            'Cost_Required_To_Connect_Child': None,

            'Ancestor_Nodes': []
        }

class Graph(Model):
    def __init__(self, model:Model):
        self.model = model

        self.graph = {}  # Dictionary of GraphNodes
        self.G = nx.DiGraph()

    # Graph access
    def add_node(self, node:GraphNode):
        self.graph[node.index] = node.node
        pass

    # run() / helper functions
    def build_graph(self):
        self.G = nx.DiGraph()
        for index, node_data in self.graph.items():
            self.G.add_node(index, label=node_data)

            # Add edge to child if exists
            if node_data['Child_Node'] is not None:
                self.G.add_edge(index, node_data['Child_Node'])

        try:
            p = list(nx.topological_sort(self.G))

            new_dict = {}
            for node in p:
                new_dict[node] = self.graph[node]

            self.graph = new_dict.copy()
        except Exception as e:
            raise e

class Data(Model):
    def __init__(self, model:Model, roadpath, roadtypepath, drainpath, pondpath, flowpathpath, elevationpath):
        self.model = model

        self.elevation = self.Elevation(elevationpath)
        self.drains = self.Drains(drainpath)
        self.ponds = self.Ponds(pondpath)
        self.flowpaths = self.Flowpaths(flowpathpath)
        self.roads = self.Roads(self, roadpath, roadtypepath)

        self._vd_projections()
        self.create_graph()

    def _vd_projections(self):
        epsg_codes = [ obj.gdf.crs.to_epsg() for obj in [self.roads, self.drains, self.ponds, self.flowpaths] ]
        epsg_codes.extend( [ obj.md['crs'].to_epsg() for obj in [self.elevation] ] )

        if len(set(epsg_codes)) != 1: raise RuntimeError("Mismatching CRS")

    def create_graph(self):

        self._vd_data()

        # For point in drains+ponds, create a node in the graph, calculating Directly Connected Segments, Type, Elevation, Child Node, and Distance to Child
        for _, row in self.drains.gdf.iterrows():
            point = row.geometry
            node_type = "D"
            elevation = row['ELEVATION']

            filtered_roads = self.roads.gdf[self.roads.gdf['DRAIN_IDX'] == point]
            road_type_index = filtered_roads.groupby('TYPE')['index'].apply(list).to_dict()
            road_type_length = filtered_roads.groupby('TYPE')['LENGTH'].sum().to_dict()
            road_type_area = filtered_roads.groupby('TYPE')['AREA'].sum().to_dict()
            child_node, distance_to_child = self.find_downhill_flowpath(point)

            node = GraphNode(point=point, node_type=node_type, elevation=elevation)
            node.node['Directly_Connected_Segments']['Indices'] = road_type_index
            node.node['Directly_Connected_Segments']['Length'] = road_type_length
            node.node['Directly_Connected_Segments']['Area'] = road_type_area
            node.node['Child_Node'] = child_node
            node.node['Distance_To_Child'] = distance_to_child

            self.model.graph.add_node(node=node)

        for _, row in self.ponds.gdf.iterrows():
            point = row.geometry
            node_type = "P"
            elevation = row['ELEVATION']
            pond_cap = row['MAX_CAP']
            pond_used = row['USED_CAP']
            child_node, distance_to_child = self.find_downhill_flowpath(point)

            node = GraphNode(point=point, node_type=node_type, elevation=elevation)
            node.node['Child_Node'] = child_node
            node.node['Distance_To_Child'] = distance_to_child
            node.node['Pond_Max_Capacity'] = pond_cap
            node.node['Pond_Used_Capacity'] = pond_used

            self.model.graph.add_node(node=node)

    def _vd_data(self):

        # Roads - already called during Roads.__init__() but we run again in case changes were made
        self.roads._vd_road_types()
        self.roads._vd_length_and_area()


        # Flowpaths - already called during Flowpaths.__init__() but we run again in case changes were made
        self.flowpaths._vd_lines()


        # Validate that for all drain and pond points, there is only a single (or perhaps zero) flowpath connected to it that goes downhill.
        points_gdf = pd.concat([self.drains.gdf, self.ponds.gdf], ignore_index=True)
        invalid_flowpath_indexes = []

        with rasterio.open(elevationpath) as src:
            for _, point in points_gdf.iterrows():
                intersecting_paths = self.flowpaths.gdf[self.flowpaths.gdf.intersects(point['geometry'])]

                # Count downhill flowpaths
                downhill_paths = []
                for path_idx, path in intersecting_paths.iterrows():
                    # Sample start and end point elevations
                    start = shapely.geometry.Point(path.geometry.coords[0])
                    end = shapely.geometry.Point(path.geometry.coords[-1])

                    start_elev = float(list(src.sample([(start.x, start.y)]))[0][0])
                    end_elev = float(list(src.sample([(end.x, end.y)]))[0][0])

                    # Check if path goes downhill
                    if (start.intersects(point.geometry) and end_elev < point['ELEVATION']) or \
                       (end.intersects(point.geometry) and start_elev < point['ELEVATION']):
                        downhill_paths.append(path_idx)

                # Track invalid flowpath indexes
                if len(downhill_paths) > 1:
                    invalid_flowpath_indexes.extend(downhill_paths)

        if invalid_flowpath_indexes: raise ValueError(f"Multiple downhill flowpaths connected to the same node: {invalid_flowpath_indexes}")

        # Validation Continued...
        # TODO: This validation is incomplete, complete validation would require checking for intersecting flowpaths, ...

        pass

    def find_downhill_flowpath(self, point):
        intersecting_flowpaths = self.flowpaths.gdf[self.flowpaths.gdf.intersects(point)]
        if len(intersecting_flowpaths) == 0: return None, None

        with rasterio.open(elevationpath) as elevation_raster:
            def get_point_elevation(pt):
                # Sample the raster at the point's coordinates
                elevation = float(list(elevation_raster.sample([(pt.x, pt.y)]))[0][0])
                return elevation

            candidate_flowpaths = []
            for _, flowpath in intersecting_flowpaths.iterrows():
                line_coords = list(flowpath.geometry.coords)
                start_point = shapely.geometry.Point(line_coords[0])
                end_point = shapely.geometry.Point(line_coords[-1])

                start_elev = get_point_elevation(start_point)
                end_elev = get_point_elevation(end_point)

                if (start_point.equals(point)) and (start_elev > end_elev):
                    candidate_flowpaths.append((flowpath, end_point))
                elif (end_point.equals(point)) and (end_elev > start_elev):
                    candidate_flowpaths.append((flowpath, start_point))

            if len(candidate_flowpaths) != 1:
                raise ValueError(f"Expected exactly one downhill flowpath, found {len(candidate_flowpaths)} at {point}")

            end_point = candidate_flowpaths[0][1]
            end_point_buffer = end_point.buffer(0.3)

            # Check intersection with roads
            road_intersections = self.roads.gdf[self.roads.gdf.intersects(end_point_buffer)]
            if not road_intersections.empty:
                # Return the DRAIN_IDX of the intersecting road segment
                return road_intersections.iloc[0]['DRAIN_IDX'], candidate_flowpaths[0][0].geometry.length

            # Check intersection with ponds
            pond_intersections = self.ponds.gdf[self.ponds.gdf.intersects(end_point)]
            if not pond_intersections.empty:
                # Return the existing end point and flowpath length
                return candidate_flowpaths[0][1], candidate_flowpaths[0][0].geometry.length

            # Check intersection with drains
            drain_intersections = self.drains.gdf[self.drains.gdf.intersects(end_point)]
            if not drain_intersections.empty:
                # Return the existing end point and flowpath length
                return candidate_flowpaths[0][1], candidate_flowpaths[0][0].geometry.length

            # At this point, we assume this is a termination point because the flowpath doesn't connect to anything
            # And because this won't exist in any gdb right now, we must create it
            point = candidate_flowpaths[0][1]
            node_type = "T"
            elevation = get_point_elevation(point)
            node = GraphNode(point=point, node_type=node_type, elevation=elevation)
            self.model.graph.add_node(node=node)

            return candidate_flowpaths[0][1], candidate_flowpaths[0][0].geometry.length


    class Roads:
        def __init__(self, data, roadpath, roadtypepath):
            self.data = data
            self.gdf : gpd.GeoDataFrame = gpd.read_file(roadpath)
            self.types = json.load(open(roadtypepath))['road_types']

            self._vd_index()
            self._vd_road_types()
            self._vd_length_and_area()

            self._pp_calculate_drain_connectivity()

        # Data Validation Functions
        def _vd_index(self):
            self.gdf['index'] = self.gdf['index'] if 'index' in self.gdf.columns else range(len(self.gdf))

        def _vd_road_types(self):
            unknown_types = set(self.gdf['TYPE']) - set(self.types)
            if unknown_types: raise ValueError(f"Unknown road types: {unknown_types}")

        def _vd_length_and_area(self):
            if (zero_indexes := [idx for idx, (length, area) in enumerate(zip(self.gdf['LENGTH'], self.gdf['AREA'])) if length <= 0 or area <= 0]):
                raise ValueError(f"Zero or negative LENGTH/AREA at road indexes: {zero_indexes}")


        # Pre-Processing Functions

        def _pp_calculate_drain_connectivity(self):
            self.gdf['INCL_DRAIN'], self.gdf['DRAIN_IDX'] = False, None

            # First, mark drain-intersecting road segments
            for drain in self.data.drains.gdf.geometry:
                intersecting_roads = self.gdf[self.gdf.geometry.apply(lambda road: drain.distance(road) <= 1e-9)]

                if len(intersecting_roads) == 0:
                    raise ValueError(f"Drain point {drain} does not intersect any road.")

                # Mark the first intersecting road segment
                idx = intersecting_roads.index[0]
                self.gdf.at[idx, 'INCL_DRAIN'] = True
                self.gdf.at[idx, 'DRAIN_IDX'] = drain

            # Process road segments
            unroutable_segments = []
            for idx, road_segment in self.gdf[~self.gdf['INCL_DRAIN']].iterrows():
                current_segment = road_segment
                visited_segments = set()

                while not current_segment['INCL_DRAIN']:
                    # Find touching segments excluding already visited ones
                    touching_segments = self.gdf[
                        self.gdf.touches(current_segment.geometry) &
                        (~self.gdf.index.isin(visited_segments))
                    ]

                    # If no touching segments, mark as unroutable
                    if touching_segments.empty:
                        unroutable_segments.append(current_segment)
                        break

                    # Check for drain-touching segments
                    drain_connected_segments = touching_segments[touching_segments['INCL_DRAIN']]

                    if not drain_connected_segments.empty:
                        # Assign the drain of the first drain-touching segment
                        self.gdf.at[idx, 'DRAIN_IDX'] = drain_connected_segments.iloc[0]['DRAIN_IDX']
                        break

                    # Progress to lowest elevation segment to continue tracing
                    current_segment = touching_segments.loc[touching_segments['ELEVATION'].idxmin()]
                    visited_segments.add(current_segment.name)

            # Log unroutable segments if any exist
            if unroutable_segments:
                unroutable_length = sum(seg.geometry.length for seg in unroutable_segments)
                print(f"Total unroutable segment length: {unroutable_length}")
                print(f"Number of unroutable segments: {len(unroutable_segments)}")


    class Drains:
        def __init__(self, drainpath):
            self.gdf : gpd.GeoDataFrame = gpd.read_file(drainpath)
            self._calculate_elevation()

        def _calculate_elevation(self):
            with rasterio.open(elevationpath) as src:
                # src.sample() returns a generator, so we call list, 
                self.gdf['ELEVATION'] = [float(e[0]) for e in list(src.sample([(x, y) for x, y in zip(self.gdf["geometry"].x, self.gdf["geometry"].y)]))]
                if (null_indexes := [i for i, x in enumerate(self.gdf['ELEVATION']) if x == src.nodata]):
                    raise ValueError(f"Features with null elevation at drain indexes: {null_indexes}")

    class Ponds:
        def __init__(self, pondpath):
            self.gdf : gpd.GeoDataFrame = gpd.read_file(pondpath)
            self._calculate_elevation()

        def _calculate_elevation(self):
            with rasterio.open(elevationpath) as src:
                self.gdf['ELEVATION'] = [float(e[0]) for e in list(src.sample([(x, y) for x, y in zip(self.gdf["geometry"].x, self.gdf["geometry"].y)]))]
                if (null_indexes := [i for i, x in enumerate(self.gdf['ELEVATION']) if x == src.nodata]):
                    raise ValueError(f"Features with null elevation at pond indexes: {null_indexes}")

    class Flowpaths:
        def __init__(self, flowpathpath):
            self.travel_cost = json.load(open(configpath))['travel_cost']
            self.gdf = gpd.read_file(flowpathpath)
            self._vd_lines()

        def _vd_lines(self):
            invalid_indexes =  [idx for idx, line in enumerate(self.gdf.geometry) if not line.is_simple]
            if invalid_indexes: raise ValueError(f"Self-intersecting lines at indexes: {invalid_indexes}")

        # Never called, ignore this function
        def _is_downhill_path(self, path, point, elevation_src):
            start = shapely.geometry.Point(path.geometry.coords[0])
            end = shapely.geometry.Point(path.geometry.coords[-1])

            start_elev = float(list(elevation_src.sample([(start.x, start.y)]))[0][0])
            end_elev = float(list(elevation_src.sample([(end.x, end.y)]))[0][0])

            return (start.intersects(point.geometry) and end_elev < point['ELEVATION']) or \
                   (end.intersects(point.geometry) and start_elev < point['ELEVATION'])


    class Elevation:
        def __init__(self, elevationpath):

            with rasterio.open(elevationpath) as src:
                self.array = src.read()
                self.md = src.meta


roadpath = 'user_data/roads.shp'
roadtypepath = 'config/config.json'
drainpath = 'user_data/drains.shp'
pondpath = 'user_data/ponds.shp'
flowpathpath = 'user_data/flowpaths.shp'
elevationpath = 'user_data/elevation.tif'
configpath = 'config/config.json'

m = Model(
    roadpath=roadpath,
    roadtypepath=roadtypepath,
    drainpath=drainpath,
    pondpath=pondpath,
    flowpathpath=flowpathpath,
    elevationpath=elevationpath,
    configpath=configpath
)

m.run()
#|%%--%%| <d1C9LohqaR|mMfRqQPgyA>


import graphviz

def visualize_node_graph(nodes_dict):
    # Create a new directed graph
    dot = graphviz.Digraph(comment='Node Graph', format='pdf')

    # Add nodes to the graph
    for index, node_data in nodes_dict.items():
        # Create a node label with key information
        node_label = f"Index: {index}\n" \
                     f"Type: {node_data['Type']}\n" \
                     f"Runoff Total: {node_data['Runoff_Total']}\n" \
                     f"Runoff By Roadtype: {node_data['All_Connected_Segments']['Runoff']}\n" \
                     f"Sediment Total: {node_data['Sediment_Total']}\n" \
                     f"Sediment By Roadtype: {node_data['All_Connected_Segments']['Sediment']}"

        if node_data['Type'] == 'P':
            node_label = node_label + f"\nPond Efficiency: {node_data['Pond_Efficiency']}\nSediment Trapped: {node_data['Sediment_Trapped']}"

        # Use the string representation of the index as the node identifier
        node_id = str(index)

        # Add the node to the graph
        dot.node(node_id, node_label)

        # Add edge to child if exists
        if node_data['Child_Node'] is not None:
            child_id = str(node_data['Child_Node'])
            dot.edge(node_id, child_id)

    # Render the graph
    dot.render('node_graph', view=True)

# Usage example
# Assuming 'nodes' is your dictionary of nodes
visualize_node_graph(m.result_graphs[0][1])


#|%%--%%| <mMfRqQPgyA|POqeFoW0bl>


import plotly.graph_objs as go
import networkx as nx
import json

def truncate_hover_text(text, max_length=5000):
    """
    Truncate hover text if it's too long, with an ellipsis indicator
    """
    if len(text) <= max_length:
        return text

    # Truncate and add ellipsis
    return text[:max_length] + "... <i>(text truncated)</i>"

def create_interactive_network(data_dict):
    # Create NetworkX graph
    G = nx.DiGraph()

    for parent, node_info in data_dict.items():
        # Add node with all its attributes
        G.add_node(parent, **node_info)

        if 'Child_Node' in node_info and node_info['Child_Node']:
            G.add_edge(parent, node_info['Child_Node'])

    # Compute layout
    pos = nx.spring_layout(G, k=0.5, iterations=50)

    # Prepare node trace
    node_x = []
    node_y = []
    node_hover_text = []
    node_colors = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        # Create hover text with all node attributes
        node_attrs = data_dict.get(node, {})

        # Build hover text with smart formatting
        hover_lines = []
        hover_lines.append(f"<b>{node}</b>")

        for key, value in node_attrs.items():
            if key == 'Child_Node':
                continue

            # Convert value to string and handle potential long values
            str_value = str(value)
            if len(str_value) > 100:
                str_value = str_value[:100] + "..."

            hover_lines.append(f"{key}: {str_value}")

        # Join lines and truncate if necessary
        hover_text = truncate_hover_text("<br>".join(hover_lines))
        node_hover_text.append(hover_text)

        # Color nodes based on connectivity
        node_colors.append(len(list(G.neighbors(node))))

    # Create node trace
    node_trace = go.Scatter(
        x=node_x, 
        y=node_y,
        mode='markers',  # Remove text labels
        hoverinfo='text',
        text=node_hover_text,
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            size=15,
            color=node_colors,
            line_width=2
        )
    )

    # Prepare edge trace
    edge_x = []
    edge_y = []
    for edge in G.edges():
        start_x, start_y = pos[edge[0]]
        end_x, end_y = pos[edge[1]]
        edge_x.extend([start_x, end_x, None])
        edge_y.extend([start_y, end_y, None])

    edge_trace = go.Scatter(
        x=edge_x, 
        y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Interactive Network Graph',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[ 
                            dict(
                                text="Node size and color represent connectivity",
                                showarrow=False,
                                xref="paper", yref="paper",
                                x=0.005, y=-0.002 
                            )
                        ],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))

    # Customize hover template for more control
    fig.update_traces(
        hovertemplate='%{text}<extra></extra>',  # Remove secondary hover box
        hoverlabel=dict(
            bgcolor='white',  # Background color of hover box
            font_size=10,     # Font size of hover text
            font_family='Arial'  # Font family
        )
    )

    return fig

# Optional: Add interactive features
def enhance_figure_interactivity(fig):
    # Add zoom and pan capabilities
    fig.update_layout(
        dragmode='zoom',  # Allow zooming by dragging
        hovermode='closest'
    )

    # Optional: Add buttons for reset view
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(
                        args=[{"visible": [True, True]},
                              {"xaxis.autorange": True,
                               "yaxis.autorange": True}],
                        label="Reset Zoom",
                        method="relayout"
                    )
                ],
                pad={"r": 10, "t": 10},
                showactive=False,
                x=0.11,
                xanchor="left",
                y=1.1,
                yanchor="top"
            )
        ]
    )

    return fig

# Usage
interactive_fig = create_interactive_network(m.result_graphs[0][1])
interactive_fig = enhance_figure_interactivity(interactive_fig)
interactive_fig.show()
